"""
imu_processor.py
================================================
Full IMU processing pipeline — Yaw, Pitch, Roll

Pipeline
--------
1.  Load CSV  →  standardise column names
2. Compute sample interval (dt) from timestamps
3. Estimate static gyroscope bias from startup samples
4. Subtract bias to get corrected rate signals
5. Run a 2-state Kalman filter  [angle, gyro_bias]
6. Run a Complementary filter   (lightweight comparison)
7. Evaluate RMSE against ground-truth angle columns
8. Return enriched DataFrame — scalars stored in df.attrs

Expected CSV columns (from imu_raw.csv)
------------------------------------------
time,
raw/true _yaw/pitch/roll_ rate_deg_s,
raw/true _yaw/pitch/roll_ angle_deg

"""

import numpy as np
import pandas as pd
import sys
import importlib
import os

sys.path.insert(0, r"E:\Projects\Python\Vehicle DAQ Pipeline")
config = importlib.import_module("config")

# ═══════════════════════════════════════════════════════════════════
#  Kalman Filter — 2-State: [angle, gyro_bias]
# ═══════════════════════════════════════════════════════════════════

class IMUKalmanFilter:
    """
    Estimates angle and gyroscope bias for a single rotational axis.

    State:  x = [angle (°),  gyro_bias (°/s)]

    Prediction:
        angle_pred = angle + (raw_rate − bias) × dt
        bias_pred  = bias                          ← random-walk assumption

    Observation:
        z = raw_angle_deg   (noisy integrated angle from sensor)
        H = [1, 0]          (we observe angle only, not bias directly)

    """

    def __init__(self,
                 dt:      float,
                 q_angle: float = config.KALMAN_Q_ANGLE,
                 q_bias:  float = config.KALMAN_Q_BIAS,
                 r_angle: float = config.KALMAN_R_ANGLE):

        self.dt = dt

        # State transition matrix  (2×2)
        self.F = np.array([[1.0, -dt],
                           [0.0,  1.0]])

        # Control-input matrix  B  (2×1) — maps raw gyro rate to angle change
        self.B = np.array([[dt],
                           [0.0]])

        # Observation matrix  H  (1×2) — we see angle, not bias
        self.H = np.array([[1.0, 0.0]])

        # Process noise  Q  (2×2)
        self.Q = np.array([[q_angle, 0.0],
                           [0.0,     q_bias]])

        # Measurement noise  R  (1×1)
        self.R = np.array([[r_angle]])

        # Initial state  [angle=0, bias=0] with high initial uncertainty
        self.x = np.zeros((2, 1))
        self.P = np.eye(2) * 1.0

    def seed(self, angle_deg: float, bias_deg_s: float = 0.0):
        """Warm-start the filter with a known angle and bias."""
        self.x[0, 0] = angle_deg
        self.x[1, 0] = bias_deg_s

    def step(self,
             gyro_rate_deg_s: float,
             angle_meas_deg:  float):
        """
        Run one predict → Update cycle.

        Parameters
        ----------
        gyro_rate_deg_s : raw (uncorrected) gyro reading  (°/s)
        angle_meas_deg  : noisy integrated angle from sensor  (°)

        Returns
        -------
        estimated_angle : float  (°)
        estimated_bias  : float  (°/s)
        """
        u = np.array([[gyro_rate_deg_s]])   

        # ── Predict ──────────────────────────────────────────────
        x_pred = self.F @ self.x + self.B @ u
        P_pred = self.F @ self.P @ self.F.T + self.Q

        # ── Update ───────────────────────────────────────────────
        z = np.array([[angle_meas_deg]])
        S = self.H @ P_pred @ self.H.T + self.R   # Innovation covariance
        K = P_pred @ self.H.T @ np.linalg.inv(S)  # Kalman gain        
        y = z - self.H @ x_pred                   # Residual

        self.x = x_pred + K @ y
        self.P = (np.eye(2) - K @ self.H) @ P_pred

        return float(self.x[0, 0]), float(self.x[1, 0])


# ═══════════════════════════════════════════════════════════════════
#  Complementary Filter  —  Single Axis
# ═══════════════════════════════════════════════════════════════════
class ComplementaryFilter:
    """
    Fuses gyroscope integration with a direct angle measurement.

    Formula
    -------
    angle = alpha * (angle_prev + rate * dt)
          + (1 - alpha) * angle_meas

    alpha close to 1  →  trusts gyro more (good short-term, drifts long-term)
    alpha close to 0  →  trusts sensor measurement more (removes drift)

    """

    def __init__(self, alpha: float = config.COMPLEMENTARY_ALPHA):
        self.alpha = alpha
        self.angle = 0.0

    def seed(self, angle_deg: float):
        self.angle = angle_deg

    def step(self,
             gyro_rate_deg_s: float,
             angle_meas_deg:  float,
             dt:              float) -> float:
        gyro_angle = self.angle + gyro_rate_deg_s * dt
        self.angle = self.alpha * gyro_angle + \
            (1.0 - self.alpha) * angle_meas_deg
        return self.angle


# ═══════════════════════════════════════════════════════════════════
#  Static Bias Estimator
# ═══════════════════════════════════════════════════════════════════
def estimate_gyro_bias(rate_series: pd.Series,
                       window: int = config.GYRO_BIAS_WINDOW) -> float:
    """
    Average the first `window` samples to estimate gyro static bias.
    Assumes the sensor is stationary during startup.

    Returns bias in °/s.

    """
    n = min(window, len(rate_series))
    bias = float(rate_series.iloc[:n].mean())
    return bias


# ═══════════════════════════════════════════════════════════════════
#  Per-Axis Processing
# ═══════════════════════════════════════════════════════════════════
def process_single_axis(df: pd.DataFrame,
                        axis: str,
                        dt:   float) -> dict:
    """
    Run bias estimation, Kalman filter, complementary filter, and
    RMSE evaluation for one rotational axis.
    Run the full filter stack for one axis (yaw, pitch, or roll).

    Returns a dict of:
      - scalar metrics  →  stored in df.attrs
      - array columns  →  added as DataFrame columns

    """
    raw_rate = df[f"raw_{axis}_rate_deg_s"].values
    raw_angle = df[f"raw_{axis}_angle_deg"].values
    true_angle = df[f"true_{axis}_angle_deg"].values
    n = len(df)

    # ── 1. Static bias removal ───────────────────────────────────────
    bias = estimate_gyro_bias(df[f"raw_{axis}_rate_deg_s"])
    corrected_rate = raw_rate - bias

    # ── 2. Kalman filter ─────────────────────────────────────────
    kf = IMUKalmanFilter(dt=dt)
    kf.seed(angle_deg=raw_angle[0], bias_deg_s=bias)

    kalman_angles = np.zeros(n)
    kalman_biases = np.zeros(n)

    for i in range(n):
        ang, b = kf.step(gyro_rate_deg_s=raw_rate[i],
                         angle_meas_deg=raw_angle[i])
        kalman_angles[i] = ang
        kalman_biases[i] = b

    # ── 3. Complementary filter ──────────────────────────────────
    cf = ComplementaryFilter()
    cf.seed(raw_angle[0])

    comp_angles = np.zeros(n)
    for i in range(n):
        comp_angles[i] = cf.step(gyro_rate_deg_s=corrected_rate[i],
                                 angle_meas_deg=raw_angle[i],
                                 dt=dt)

    # ── 4. RMSE vs Ground Truth ───────────────────────────────────────
    rmse_kalman = float(np.sqrt(np.mean((kalman_angles - true_angle) ** 2)))
    rmse_comp = float(np.sqrt(np.mean((comp_angles - true_angle) ** 2)))
    rmse_raw = float(np.sqrt(np.mean((raw_angle - true_angle) ** 2)))

    return {
        f"{axis}_bias_deg_s": bias,
        f"{axis}_rmse_kalman": rmse_kalman,
        f"{axis}_rmse_comp": rmse_comp,
        f"{axis}_rmse_raw": rmse_raw,
        f"{axis}_corrected_rate": corrected_rate,
        f"{axis}_angle_kalman": kalman_angles,
        f"{axis}_bias_kalman": kalman_biases,
        f"{axis}_angle_comp": comp_angles,
    }


# ═══════════════════════════════════════════════════════════════════
#  Column Name Standardiser
# ═══════════════════════════════════════════════════════════════════
def _standardise_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Maps raw CSV headers to the schema used throughout this module:

        raw_{axis}_rate_deg_s  /  true_{axis}_rate_deg_s
        raw_{axis}_angle_deg   /  true_{axis}_angle_deg

    Detection is substring-based so minor naming variations are handled.

    """

    df.columns = df.columns.str.strip().str.lower()

    rename = {}
    for col in df.columns:
        for axis in ["yaw", "pitch", "roll"]:
            if axis not in col:
                continue
            if "raw" in col and ("rate" in col):
                rename[col] = f"raw_{axis}_rate_deg_s"
            elif "true" in col and ("rate" in col):
                rename[col] = f"true_{axis}_rate_deg_s"
            elif "raw" in col and "angle" in col:
                rename[col] = f"raw_{axis}_angle_deg"
            elif "true" in col and "angle" in col:
                rename[col] = f"true_{axis}_angle_deg"

    df = df.rename(columns=rename)
    return df


# ═══════════════════════════════════════════════════════════════════
#  Main Entry Point
# ═══════════════════════════════════════════════════════════════════
def process_imu(filepath: str) -> pd.DataFrame:
    """
    Load, clean, filter, and evaluate IMU data for all three axes.

    Returns an enriched DataFrame with new columns per axis:
        {axis}_corrected_rate    bias-removed gyro rate  (°/s)
        {axis}_angle_kalman      Kalman-filtered angle   (°)
        {axis}_bias_kalman       dynamic bias estimate   (°/s)
        {axis}_angle_comp        complementary angle     (°)

    Scalar metrics are stored in df.attrs:
        {axis}_bias_deg_s        static startup bias     (°/s)
        {axis}_rmse_raw          raw angle RMSE vs truth
        {axis}_rmse_comp         complementary RMSE
        {axis}_rmse_kalman       Kalman RMSE

    """

    # ── Step 1: Load CSV ─────────────────────────────────────────
    df = pd.read_csv(filepath)

    # ── Step 2: Standardise column names ─────────────────────────
    df = _standardise_columns(df)

    time_candidates = [c for c in df.columns
                       if c == "time" or c.startswith("time")]
    if not time_candidates:
        raise ValueError("No 'time' column found in IMU CSV.")
    df = df.rename(columns={time_candidates[0]: "time_s"})

    # ── Step 3: Clean timestamps ───────────────────────
    df["time_s"] = pd.to_numeric(df["time_s"], errors="coerce")
    df = df.dropna(subset=["time_s"]).reset_index(drop=True)

    # ── Step 4: Sample interval ────────────────────────────────────────
    dt = float(df["time_s"].diff().median())
    if np.isnan(dt) or dt <= 0:
        dt = 0.01   # fallback: assume 100 Hz
    print(f"      Sample interval dt = {dt*1000:.2f} ms  "
          f"(~{1/dt:.0f} Hz)")

    # ── Step 5: Validate required columns ───────────────────
    required_cols = []
    for axis in ["yaw", "pitch", "roll"]:
        for prefix in ["raw", "true"]:
            for suffix in ["rate_deg_s", "angle_deg"]:
                required_cols.append(f"{prefix}_{axis}_{suffix}")

    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(
            f"Missing columns after standardisation: {missing}\n"
            f"Available columns: {df.columns.tolist()}"
        )

    for col in required_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=required_cols).reset_index(drop=True)

    print(f"      Samples loaded: {len(df)}")

    # ── Step 6: Process all three axes ─────────────────────────────────
    scalar_results = {}

    for axis in ["yaw", "pitch", "roll"]:
        results = process_single_axis(df, axis, dt)

        scalars = [f"{axis}_bias_deg_s",
                   f"{axis}_rmse_kalman",
                   f"{axis}_rmse_comp",
                   f"{axis}_rmse_raw"]

        for key in scalars:
            scalar_results[key] = results.pop(key)

        for col, arr in results.items():
            df[col] = arr

    df.attrs.update(scalar_results)

    # ── Step 7: Print summary table ─────────────────────────────────────
    print(f"\n  {'Axis':<8} {'Bias(°/s)':>12} "
          f"{'RMSE Raw':>12} {'RMSE Comp':>12} {'RMSE Kalman':>13}")
    print("  " + "─" * 62)
    for axis in ["yaw", "pitch", "roll"]:
        print(
            f"  {axis:<8} "
            f"{scalar_results[f'{axis}_bias_deg_s']:>12.4f} "
            f"{scalar_results[f'{axis}_rmse_raw']:>12.4f} "
            f"{scalar_results[f'{axis}_rmse_comp']:>12.4f} "
            f"{scalar_results[f'{axis}_rmse_kalman']:>13.4f}"
        )

    return df
