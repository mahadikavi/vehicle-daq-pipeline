""""
Calibration.py
======================================================
PHASE 1 - DAQ CALIBRATION ENGINE

Responsibilities:
  1. Pull raw ultrasonic data from SensorSimulator (Phase 0)
  2. Z-score outlier detection & rejection on noisy_mm
  3. Kalman filter pass on cleaned ultrasonic signal
  4. Linear regression calibration (filtered → true_mm)
  5. IMU static bias estimation (gyro startup window)
  6. Persist calibration coefficients back into config.yaml

Outputs:
  - Calibrated slope, intercept, R² printed to console
  - Gyro bias (yaw/pitch/roll) printed to console
  - config.yaml updated with cal_slope, cal_intercept, gyro biases
  - outputs/processed/calibration_report.csv saved

"""

# ─────── Import Phase 0 ─────────────────────────────

from scipy import stats
import numpy as np
import pandas as pd
import yaml
import sys
import os

_BASE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _BASE)
from Phase_0___Hardware_Abstraction import SensorSimulator  # noqa: E402

_DEFAULT_CONFIG = os.path.join(_BASE, "config.yaml")


# ══════════════════════════════════════════════════════════════════════
#  KALMAN FILTER  (1-D: state = [position, velocity])
# ══════════════════════════════════════════════════════════════════════

class KalmanFilter1D:
    """
    Lightweight 1-D Kalman filter for scalar sensor streams.

    State vector : x = [value, rate]
    Measurement  : z = [value]
    """

    def __init__(self, q: float = 1e-4, r: float = 0.5,
                 dt: float = 0.01):
        self.dt = dt
        # State transition
        self.F = np.array([[1, dt],
                           [0,  1]])
        # Measurement matrix
        self.H = np.array([[1, 0]])
        # Process noise covariance
        self.Q = np.array([[q,   0],
                           [0,   q]])
        # Measurement noise covariance
        self.R = np.array([[r]])
        # State & covariance
        self.x = np.zeros((2, 1))
        self.P = np.eye(2) * 1.0

    def update(self, z: float) -> float:
        """Feed one measurement, return filtered value."""
        # Predict
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q

        # Update
        z_vec = np.array([[z]])
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)
        self.x = self.x + K @ (z_vec - self.H @ self.x)
        self.P = (np.eye(2) - K @ self.H) @ self.P

        return float(self.x[0, 0])

    def reset(self, init_val: float = 0.0):
        self.x = np.array([[init_val], [0.0]])
        self.P = np.eye(2) * 1.0


# ══════════════════════════════════════════════════════════════════════
#  CALIBRATION ENGINE
# ══════════════════════════════════════════════════════════════════════

class CalibrationEngine:
    """
    Full DAQ calibration pipeline for ultrasonic + IMU sensors.

    Usage
    -----
    engine = CalibrationEngine()
    results = engine.run()
    """

    def __init__(self, config_path: str = _DEFAULT_CONFIG):
        self.config_path = config_path

        with open(config_path) as f:
            self.cfg = yaml.safe_load(f)

        # Init simulator (Phase 0)
        self.sim = SensorSimulator(config_path, mode="replay")

        # Calibration parameters from config
        u = self.cfg["ultrasonic"]
        self._z_thresh = 3.0
        self._kalman_q = float(u.get("kalman_q", 1e-4))
        self._kalman_r = float(u.get("kalman_r", 0.5))
        self._bias_window = int(
            self.cfg["imu"].get("startup_bias_samples", 50))

        # Output dir
        out = os.path.join(_BASE, self.cfg["paths"]["processed_data_dir"])
        os.makedirs(out, exist_ok=True)
        self._out_dir = out

    # ─────── STEP 1: Outlier Rejection ─────────────────────────────

    def _reject_outliers(self, series: pd.Series):
        """
        Z-score outlier detection.
        Samples with |z| > threshold are replaced by forward-fill.
        Returns cleaned series + boolean mask of outliers.
        """
        mean = series.mean()
        std = series.std()
        if std == 0:
            return series.copy(), pd.Series([False] * len(series))

        z_scores = (series - mean) / std
        is_outlier = z_scores.abs() > self._z_thresh

        cleaned = series.copy()
        cleaned[is_outlier] = np.nan
        cleaned = cleaned.ffill().bfill()

        return cleaned, is_outlier

    # ─────── STEP 2: Kalman Filter Pass ────────────────────────────

    def _apply_kalman(self, series: pd.Series,
                      dt: float = 0.01) -> pd.Series:
        """Apply 1-D Kalman filter to a pandas Series."""
        kf = KalmanFilter1D(q=self._kalman_q,
                            r=self._kalman_r,
                            dt=dt)
        kf.reset(init_val=float(series.iloc[0]))
        return pd.Series(
            [kf.update(v) for v in series],
            index=series.index
        )

    # ─────── STEP 3: Linear Regression Calibration ───────────────────

    def _linear_calibration(self, filtered: pd.Series,
                            truth: pd.Series):
        """
        Fit: truth = slope * filtered + intercept
        Returns slope, intercept, r_value, p_value, std_err
        """
        result = stats.linregress(filtered.values, truth.values)
        return result   # slope, intercept, rvalue, pvalue, stderr

    # ─────── STEP 4: IMU Gyro Bias Estimation ──────────────────────────

    def _estimate_gyro_bias(self, imu_df: pd.DataFrame) -> dict:
        """
        Estimate static gyro bias from startup window (first N samples).
        Assumes vehicle is stationary at startup.
        """
        window = imu_df.iloc[:self._bias_window]
        bias = {
            "yaw_bias_deg_s":   float(window["raw_yaw_rate_deg_s"].mean()),
            "pitch_bias_deg_s": float(window["raw_pitch_rate_deg_s"].mean()),
            "roll_bias_deg_s":  float(window["raw_roll_rate_deg_s"].mean()),
        }
        return bias

    # ─────── MAIN RUN ───────────────────────────────────────────────────

    def run(self) -> dict:
        print("\n" + "=" * 60)
        print("  Phase 1 — DAQ Calibration Engine")
        print("=" * 60)

        # ── Load raw data ──────────────────────────────────────────

        ultra_df = self.sim.get_all_ultrasonic()
        imu_df = self.sim.get_all_imu()

        dt = float(ultra_df["time_s"].diff().median())

        # ── Ultrasonic Pipeline ────────────────────────────────────

        print("\n[1/4] Outlier rejection (Z-score threshold = "
              f"{self._z_thresh})...")
        cleaned, outlier_mask = self._reject_outliers(ultra_df["noisy_mm"])
        ultra_df["cleaned_mm"] = cleaned
        ultra_df["is_outlier"] = outlier_mask
        n_out = outlier_mask.sum()
        pct = 100 * n_out / len(ultra_df)
        print(f"      Outliers detected : {n_out} / {len(ultra_df)}"
              f" ({pct:.2f}%)")

        print("\n[2/4] Kalman filtering cleaned ultrasonic signal...")
        ultra_df["kalman_mm"] = self._apply_kalman(cleaned, dt=dt)
        rmse_raw = float(np.sqrt(
            ((ultra_df["noisy_mm"] - ultra_df["true_distance_mm"])**2).mean()
        ))
        rmse_kal = float(np.sqrt(
            ((ultra_df["kalman_mm"] - ultra_df["true_distance_mm"])**2).mean()
        ))
        print(f"      RMSE (raw)    : {rmse_raw:.4f} mm")
        print(f"      RMSE (Kalman) : {rmse_kal:.4f} mm")
        print(f"      Improvement   : {rmse_raw - rmse_kal:.4f} mm")

        print("\n[3/4] Linear regression calibration...")
        reg = self._linear_calibration(
            ultra_df["kalman_mm"],
            ultra_df["true_distance_mm"]
        )
        print(f"      Slope      : {reg.slope:.6f}")
        print(f"      Intercept  : {reg.intercept:.6f} mm")
        print(f"      R²         : {reg.rvalue**2:.8f}")
        print(f"      Std Error  : {reg.stderr:.6f}")

        ultra_df["calibrated_mm"] = (
            reg.slope * ultra_df["kalman_mm"] + reg.intercept
        )
        rmse_cal = float(np.sqrt(
            ((ultra_df["calibrated_mm"] -
             ultra_df["true_distance_mm"])**2).mean()
        ))
        print(f"      RMSE (calibrated) : {rmse_cal:.6f} mm")

    # ─────── IMU Bias Estimation ────────────────────────────────────

        print(f"\n[4/4] IMU gyro bias estimation "
              f"(window = {self._bias_window} samples)...")
        bias = self._estimate_gyro_bias(imu_df)
        print(f"      Yaw   bias : {bias['yaw_bias_deg_s']:+.4f} °/s")
        print(f"      Pitch bias : {bias['pitch_bias_deg_s']:+.4f} °/s")
        print(f"      Roll  bias : {bias['roll_bias_deg_s']:+.4f} °/s")

        # ── Persist Calibration Coefficients to config.yaml ───────

        self._save_to_config(reg.slope, reg.intercept, reg.rvalue**2, bias)

        # ── Save calibration report CSV ────────────────────────────

        report_path = os.path.join(self._out_dir, "calibration_report.csv")
        ultra_df[[
            "time_s", "noisy_mm", "cleaned_mm",
            "kalman_mm", "calibrated_mm",
            "true_distance_mm", "is_outlier"
        ]].to_csv(report_path, index=False)
        print(f"\n✓ Calibration report saved → {report_path}")

        results = {
            "cal_slope":       reg.slope,
            "cal_intercept":   reg.intercept,
            "cal_r2":          reg.rvalue**2,
            "rmse_raw_mm":     rmse_raw,
            "rmse_kalman_mm":  rmse_kal,
            "rmse_cal_mm":     rmse_cal,
            "n_outliers":      int(n_out),
            "gyro_bias":       bias,
            "ultra_df":        ultra_df,
            "imu_df":          imu_df,
        }
        return results

    # ─────── Write calibration back into config.yaml ────────────────────

    def _save_to_config(self, slope: float, intercept: float,
                        r2: float, bias: dict):
        with open(self.config_path) as f:
            cfg = yaml.safe_load(f)

        cfg["ultrasonic"]["cal_slope"] = float(round(slope,     8))
        cfg["ultrasonic"]["cal_intercept"] = float(round(intercept, 8))
        cfg["ultrasonic"]["cal_r2"] = float(round(r2,        8))
        cfg["imu"]["yaw_bias_deg_s"] = float(
            round(bias["yaw_bias_deg_s"],   6))
        cfg["imu"]["pitch_bias_deg_s"] = float(
            round(bias["pitch_bias_deg_s"], 6))
        cfg["imu"]["roll_bias_deg_s"] = float(
            round(bias["roll_bias_deg_s"],  6))

        with open(self.config_path, "w") as f:
            yaml.dump(cfg, f, default_flow_style=False, sort_keys=False)

        print(f"\n✓ Calibration coefficients saved → config.yaml")


# ══════════════════════════════════════════════════════════════════════
#  STANDALONE TEST
# ══════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    engine = CalibrationEngine()
    results = engine.run()

    print("\n" + "─" * 60)
    print("  CALIBRATION SUMMARY")
    print("─" * 60)
    print(f"  Slope         : {results['cal_slope']:.6f}")
    print(f"  Intercept     : {results['cal_intercept']:.6f} mm")
    print(f"  R²            : {results['cal_r2']:.8f}")
    print(f"  RMSE raw      : {results['rmse_raw_mm']:.4f} mm")
    print(f"  RMSE Kalman   : {results['rmse_kalman_mm']:.4f} mm")
    print(f"  RMSE cal      : {results['rmse_cal_mm']:.6f} mm")
    print(f"  Outliers      : {results['n_outliers']}")
    print(
        f"  Gyro bias yaw : {results['gyro_bias']['yaw_bias_deg_s']:+.4f} °/s")
    print("─" * 60)
