"""
fusion.py
================================================
Sensor Fusion + Control Logic

Steps
-----
1.  Time-synchronise ultrasonic and IMU streams (merge_asof)
2.  Compute obstacle confidence score  (distance + tilt)
3.  Generate proximity and tilt alert flags
4.  Run PID proximity controller  →  brake command [0, 1]
5.  Build timestamped event log CSV
"""

import numpy as np
import pandas as pd
import sys
import importlib
import os

sys.path.insert(0, r"E:\Projects\Python\Vehicle DAQ Pipeline")
config = importlib.import_module("config")

# ═══════════════════════════════════════════════════════════════════
#  PID Controller
# ═══════════════════════════════════════════════════════════════════


class PIDController:
    """
    Discrete-time PID controller that outputs a brake command in [0, 1].

    0 = no braking required
    1 = full emergency brake
    The error is defined as (setpoint − measurement), so a positive error
    means the vehicle is too close and needs to brake.

    """

    def __init__(self,
                 kp=config.PID_KP,
                 ki=config.PID_KI,
                 kd=config.PID_KD,
                 setpoint=config.PID_SETPOINT_MM):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.setpoint = setpoint
        self._integral = 0.0
        self._prev_error = 0.0

    def reset(self):
        """Clear integrator and derivative memory."""
        self._integral = 0.0
        self._prev_error = 0.0

    def step(self, measurement: float, dt: float) -> float:
        """
        Advance one time step and return the brake command.

        Parameters
        ----------
        measurement : current distance reading (mm)
        dt          : time step (s)

        Returns
        -------
        brake : float clamped to [0, 1]

        """
        error = self.setpoint - measurement   
        self._integral += error * dt
        derivative = (error - self._prev_error) / dt if dt > 0 else 0.0
        self._prev_error = error

        raw = (self.kp * error
               + self.ki * self._integral
               + self.kd * derivative)

        return float(np.clip(raw, 0.0, 1.0))


# ═══════════════════════════════════════════════════════════════════
#  Obstacle Confidence Score
# ═══════════════════════════════════════════════════════════════════
def obstacle_confidence(dist_mm:   float,
                        pitch_deg: float,
                        roll_deg:  float,
                        dist_thresh: float = config.PROXIMITY_THRESH_MM,
                        tilt_thresh: float = config.TILT_THRESH_DEG) -> float:
    """
    Weighted score [0, 1] estimating how dangerously close an obstacle is.

    Distance component:  (1 − d / d_thresh)  when d < d_thresh, else 0
    Tilt component:      sqrt(pitch² + roll²) / tilt_thresh, capped at 1

    Final score = w_dist × dist_component + w_tilt × tilt_component

    """
    if dist_mm < dist_thresh:
        dist_component = 1.0 - (dist_mm / dist_thresh)
    else:
        dist_component = 0.0

    tilt_mag = np.sqrt(pitch_deg ** 2 + roll_deg ** 2)
    tilt_component = min(tilt_mag / tilt_thresh, 1.0)

    score = (config.OBSTACLE_W_DIST * dist_component
             + config.OBSTACLE_W_TILT * tilt_component)

    return float(np.clip(score, 0.0, 1.0))


# ═══════════════════════════════════════════════════════════════════
#  Main Fusion Function
# ═══════════════════════════════════════════════════════════════════
def run_fusion(ultra_df: pd.DataFrame,
               imu_df:   pd.DataFrame) -> dict:
    """
    Merge ultrasonic and IMU streams, compute alerts, run PID, log events.

    Parameters
    ----------
    ultra_df : output of process_ultrasonic()
    imu_df   : output of process_imu()

    Returns
    -------
    dict with:
        "fused"  : merged DataFrame with all control signals
        "events" : timestamped event-log DataFrame
    """

    # ── 1. Select and sort relevant columns from each stream ──────────────
    u = ultra_df[[
        "time_s",
        "calibrated_mm",
        "kalman_mm",
        "is_outlier"
    ]].copy().sort_values("time_s").reset_index(drop=True)

    i = imu_df[[
        "time_s",
        "yaw_angle_kalman",
        "pitch_angle_kalman",
        "roll_angle_kalman"
    ]].copy().sort_values("time_s").reset_index(drop=True)

    # ── 2. Time-align with merge_asof ────────────────────────────
    # For each ultrasonic timestamp, find the nearest IMU timestamp
    # within FUSION_TOLERANCE_MS milliseconds.
    tolerance_s = config.FUSION_TOLERANCE_MS * 1e-3

    fused = pd.merge_asof(
        u, i,
        on="time_s",
        tolerance=tolerance_s,
        direction="nearest"
    )

    fused = fused.dropna(subset=[
        "pitch_angle_kalman",
        "roll_angle_kalman"
    ]).reset_index(drop=True)

    # ── 3. Obstacle confidence score ─────────────────────────────
    fused["confidence"] = [
        obstacle_confidence(
            dist_mm=row["calibrated_mm"],
            pitch_deg=row["pitch_angle_kalman"],
            roll_deg=row["roll_angle_kalman"]
        )
        for _, row in fused.iterrows()
    ]

    # ── 4. Alert flags ───────────────────────────────────────────
    fused["proximity_alert"] = (
        fused["calibrated_mm"] < config.PROXIMITY_THRESH_MM
    )

    fused["tilt_alert"] = (
        (fused["pitch_angle_kalman"].abs() > config.TILT_THRESH_DEG) |
        (fused["roll_angle_kalman"].abs() > config.TILT_THRESH_DEG)
    )

    # ── 5. PID controller ────────────────────────────────────────
    pid = PIDController()
    dt = float(fused["time_s"].diff().median())

    brake_cmds = []
    for dist in fused["calibrated_mm"]:
        brake_cmds.append(pid.step(dist, dt))
    fused["brake_cmd"] = brake_cmds

    # ── 6. Event log ─────────────────────────────────────────────
    events = []

    for _, row in fused[fused["proximity_alert"]].iterrows():
        events.append({
            "time_s": row["time_s"],
            "event_type": "PROXIMITY_ALERT",
            "value": round(row["calibrated_mm"], 3),
            "unit": "mm",
            "note": (f"Distance {row['calibrated_mm']:.1f} mm "
                     f"< threshold {config.PROXIMITY_THRESH_MM} mm")
        })

    # Tilt alert events
    for _, row in fused[fused["tilt_alert"]].iterrows():
        max_tilt = max(abs(row["pitch_angle_kalman"]),
                       abs(row["roll_angle_kalman"]))
        events.append({
            "time_s": row["time_s"],
            "event_type": "TILT_ALERT",
            "value": round(max_tilt, 3),
            "unit": "deg",
            "note": (f"Tilt {max_tilt:.1f}° "
                     f"> threshold {config.TILT_THRESH_DEG}°")
        })

    # Outlier detection events (from ultrasonic stage)
    for t in ultra_df.loc[ultra_df["is_outlier"], "time_s"]:
        events.append({
            "time_s": t,
            "event_type": "OUTLIER_DETECTED",
            "value": float("nan"),
            "unit": "mm",
            "note": "Ultrasonic spike rejected by Z-score filter"
        })

    events_df = pd.DataFrame(events)
    if not events_df.empty:
        events_df = (events_df
                     .sort_values("time_s")
                     .reset_index(drop=True))

    return {"fused": fused, "events": events_df}
