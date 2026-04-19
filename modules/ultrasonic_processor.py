"""
ultrasonic_processor.py
================================================
Ultrasonic Sensor DAQ Processing Pipeline

Pipeline
--------
1.  Load CSV  →  standardise column names
2.  Convert echo time to distance using physics formula
3.  Z-score outlier rejection  +  forward-fill
4.  1-D Kalman Filter  (state: position + velocity)
5.  Linear calibration against ground-truth column
6.  Return enriched DataFrame (scalar results in df.attrs)

Expected CSV columns (from ultrasonic_raw.csv)
-----------------------------------------------
time_s, echo_time_us, noisy_mm, true_distance_mm
"""
import importlib
import os
from scipy import stats
import pandas as pd
import numpy as np
import sys
# Permanently fix import path on Windows
sys.path.insert(0, r"E:\Projects\Python\Vehicle DAQ Pipeline")
config = importlib.import_module("config")

# ═══════════════════════════════════════════════════════════════════
#  1-D Kalman Filter  —  state: [position, velocity]
# ═══════════════════════════════════════════════════════════════════


class KalmanFilter1D:
    """
    Tracks distance (mm) as a position + velocity state.

    State:  x = [position (mm),  velocity (mm/s)]

    Observation:
        z = position only
        H = [1, 0]  (velocity is hidden — we never measure it directly)

    Process noise Q is scaled by dt so faster sampling means
    proportionally less assumed process noise.

    """

    def __init__(self,
                 dt: float,
                 q:  float = config.KALMAN_Q_ULTRA,
                 r:  float = config.KALMAN_R_ULTRA):

        self.dt = dt

        # State transition — position integrates velocity over dt
        self.F = np.array([[1.0, dt],
                           [0.0, 1.0]])

        # We observe position only
        self.H = np.array([[1.0, 0.0]])

        # Process noise — standard kinematic Q matrix scaled by q
        self.Q = q * np.array([[dt**4 / 4, dt**3 / 2],
                               [dt**3 / 2, dt**2]])

        # Measurement noise (scalar variance in mm²)
        self.R = np.array([[r]])

        # Initial state and covariance (start uncertain)
        self.x = np.zeros((2, 1))
        self.P = np.eye(2) * 1.0

    def seed(self, position: float):
        """Warm-start the filter with a known initial position."""
        self.x[0, 0] = position

    def update(self, z: float) -> float:
        """
        Run one predict → update cycle.

        Parameters
        ----------
        z : position measurement  (mm)

        Returns
        -------
        estimated_position :  (mm)
        """
        # ── Predict ──────────────────────────────────────────────
        x_pred = self.F @ self.x
        P_pred = self.F @ self.P @ self.F.T + self.Q

        # ── Update ───────────────────────────────────────────────
        S = self.H @ P_pred @ self.H.T + self.R   # Innovation covariance
        K = P_pred @ self.H.T @ np.linalg.inv(S)  # Kalman gain  (2×1)
        y = np.array([[z]]) - self.H @ x_pred     # Innovation

        self.x = x_pred + K @ y
        self.P = (np.eye(2) - K @ self.H) @ P_pred

        return float(self.x[0, 0])


# ═══════════════════════════════════════════════════════════════════
#  Outlier Rejection  —  Z-score thresholding
# ═══════════════════════════════════════════════════════════════════
def reject_outliers(series: pd.Series,
                    threshold: float = config.OUTLIER_Z_THRESH) -> pd.Series:
    """
    Flag samples whose Z-score exceeds the threshold.
    NaNs are filled with the series mean before scoring.

    Returns
    -------
    Boolean Series  (True = outlier)

    """
    filled = series.fillna(series.mean())
    z_scores = np.abs(stats.zscore(filled))
    return pd.Series(z_scores > threshold, index=series.index)


# ═══════════════════════════════════════════════════════════════════
#  Linear Calibration — sensor vs ground truth
# ═══════════════════════════════════════════════════════════════════
def calibrate(filtered: pd.Series,
              truth:    pd.Series):
    """
    Fit a linear model:  truth = slope * filtered + intercept
    Only uses rows where both signals are valid (non-NaN).

    Parameters
    ----------
    filtered : Kalman-filtered distance column
    truth    : ground-truth distance column

    Returns
    -------
    slope, intercept, r_squared
    """
    valid = ~(filtered.isna() | truth.isna())
    slope, intercept, r, _, _ = stats.linregress(
        filtered[valid], truth[valid]
    )
    return slope, intercept, r ** 2


# ═══════════════════════════════════════════════════════════════════
#  Main Entry Point
# ═══════════════════════════════════════════════════════════════════
def process_ultrasonic(filepath: str) -> pd.DataFrame:
    """
    Run the full ultrasonic processing pipeline on a raw CSV file.

    Returns an enriched DataFrame with new columns:
        physics_mm      distance derived from echo-time physics formula
        is_outlier      True where Z-score spike was detected
        cleaned_mm      spike-replaced signal (forward-filled)
        kalman_mm       Kalman-filtered distance
        calibrated_mm   final calibrated output

    Scalar calibration results in df.attrs:
        cal_slope, cal_intercept, cal_r2

    """

    # ── Step 1: Load CSV ─────────────────────────────────────────
    df = pd.read_csv(filepath)
    df.columns = (df.columns
                    .str.strip()
                    .str.lower()
                    .str.replace(" ", "_"))

    time_col = [c for c in df.columns if "time" in c][0]
    echo_col = [c for c in df.columns if "echo" in c][0]
    noisy_col = [c for c in df.columns if "noisy" in c][0]
    true_col = [c for c in df.columns if "true" in c][0]

    df = df.rename(columns={
        time_col:  "time_s",
        echo_col:  "echo_time_us",
        noisy_col: "noisy_mm",
        true_col:  "true_distance_mm"
    })

    for col in ["time_s", "echo_time_us", "noisy_mm", "true_distance_mm"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=["time_s", "noisy_mm"]).reset_index(drop=True)

    print(f"      Samples loaded : {len(df)}")

    # ── Step 2: Physics Formula ───────────────────────────────────
    # d_mm = (echo_time_us × 1e-6 × speed_of_sound × 1000) / 2
    df["physics_mm"] = (
        df["echo_time_us"] * 1e-6
        * config.SPEED_OF_SOUND_MS
        * 1000
    ) / 2.0

    # ── Step 3: Outlier rejection ─────────────────────────────────
    df["is_outlier"] = reject_outliers(df["noisy_mm"])

    cleaned = df["noisy_mm"].copy()
    cleaned[df["is_outlier"]] = np.nan
    cleaned = cleaned.ffill().bfill()
    df["cleaned_mm"] = cleaned

    print(f"      Outliers found : {df['is_outlier'].sum()}")

    # ── Step 4: Kalman filter ─────────────────────────────────────
    dt = float(df["time_s"].diff().median())
    if np.isnan(dt) or dt <= 0:
        dt = 0.01  # fallback: assume 100 Hz

    print(f"      Sample interval: {dt*1000:.2f} ms  (~{1/dt:.0f} Hz)")

    kf = KalmanFilter1D(dt=dt)
    kf.seed(df["cleaned_mm"].iloc[0])

    kalman_out = []
    for val in df["cleaned_mm"]:
        kalman_out.append(kf.update(val))
    df["kalman_mm"] = kalman_out

    # ── Step 5: Linear calibration against ground truth ───────────────────────────────────────
    slope, intercept, r2 = calibrate(
        df["kalman_mm"],
        df["true_distance_mm"]
    )
    df["calibrated_mm"] = df["kalman_mm"] * slope + intercept

    df.attrs["cal_slope"] = slope
    df.attrs["cal_intercept"] = intercept
    df.attrs["cal_r2"] = r2

    print(f"      Cal slope      : {slope:.4f}")
    print(f"      Cal intercept  : {intercept:.4f}")
    print(f"      Cal R²         : {r2:.6f}")

    rmse = float(
        np.sqrt(
            ((df["calibrated_mm"] - df["true_distance_mm"]) ** 2).mean()
        )
    )
    print(f"      RMSE           : {rmse:.3f} mm")

    return df
