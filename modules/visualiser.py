"""
visualiser.py
================================================
Generates three matplotlib dashboards saved to output/:

  01_ultrasonic_pipeline.png  — 3-panel signal processing chain
  02_imu_axes.png             — 3×2 grid: angle + bias per axis
  03_fusion_dashboard.png     — 4-panel fusion + control overview

"""

import sys
import importlib
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")          

sys.path.insert(0, r"E:\Projects\Python\Vehicle DAQ Pipeline")
config = importlib.import_module("config")

# ── Colour palette ────────────────────────────────────────────────

C = {
    "raw": "#adb5bd",
    "cleaned": "#74c0fc",
    "kalman": "#339af0",
    "calibrated": "#2f9e44",
    "truth": "#e03131",
    "outlier": "#f76707",
    "yaw": "#7048e8",
    "pitch": "#1971c2",
    "roll": "#2f9e44",
    "comp": "#f59f00",
    "confidence": "#e03131",
    "brake": "#862e9c",
}


# ═══════════════════════════════════════════════════════════════════
#  Helper — shade alert regions on an axis
# ═══════════════════════════════════════════════════════════════════
def _shade_alerts(ax, time_arr, bool_arr, color, alpha=0.15):
    """Fill background wherever bool_arr is True."""
    in_region = False
    t_start = None
    for i, flag in enumerate(bool_arr):
        if flag and not in_region:
            t_start = time_arr[i]
            in_region = True
        elif not flag and in_region:
            ax.axvspan(t_start, time_arr[i], color=color, alpha=alpha)
            in_region = False
    if in_region:
        ax.axvspan(t_start, time_arr[-1], color=color, alpha=alpha)


# ═══════════════════════════════════════════════════════════════════
#  Plot 1 — Ultrasonic Pipeline
# ═══════════════════════════════════════════════════════════════════
def plot_ultrasonic_pipeline(df, out_dir: str):
    """
    Three-panel figure showing the full ultrasonic processing chain:
      Top    — Raw signal, outlier removal, Kalman filter vs ground truth
      Middle — Pre- vs post-calibration vs ground truth
      Bottom — Residual error (calibrated − truth) with RMSE annotation
    """
    fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True)
    fig.suptitle("Ultrasonic DAQ Pipeline",
                 fontsize=15, fontweight="bold")

    t = df["time_s"]

    # Panel 1 — Signal Chain
    ax = axes[0]
    ax.plot(t, df["noisy_mm"],
            color=C["raw"], alpha=0.5, lw=0.8, label="Noisy (raw)")
    ax.plot(t, df["cleaned_mm"],
            color=C["cleaned"], lw=1.0, label="Cleaned (outliers filled)")
    ax.plot(t, df["kalman_mm"],
            color=C["kalman"], lw=1.5, label="Kalman filtered")
    ax.plot(t, df["true_distance_mm"],
            color=C["truth"], lw=1.5, ls="--", label="Ground truth")

    outlier_mask = df["is_outlier"]
    ax.scatter(t[outlier_mask], df["noisy_mm"][outlier_mask],
               color=C["outlier"], zorder=5, s=20,
               label=f"Outliers ({outlier_mask.sum()})")

    ax.set_ylabel("Distance (mm)")
    ax.set_title("Signal Processing Chain")
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(alpha=0.3)

    # Panel 2 — Calibration Quality
    ax = axes[1]
    ax.plot(t, df["kalman_mm"],
            color=C["kalman"], lw=1.2, alpha=0.7,
            label="Pre-calibration (Kalman)")
    ax.plot(t, df["calibrated_mm"],
            color=C["calibrated"], lw=1.5,
            label="Post-calibration")
    ax.plot(t, df["true_distance_mm"],
            color=C["truth"], lw=1.5, ls="--",
            label="Ground truth")

    r2 = df.attrs.get("cal_r2",        float("nan"))
    slp = df.attrs.get("cal_slope",     float("nan"))
    icp = df.attrs.get("cal_intercept", float("nan"))
    ax.set_title(
        f"Calibration  |  slope = {slp:.4f}  "
        f"intercept = {icp:.4f}  R² = {r2:.6f}"
    )
    ax.set_ylabel("Distance (mm)")
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(alpha=0.3)

    # Panel 3 — Residual
    ax = axes[2]
    resid = df["calibrated_mm"] - df["true_distance_mm"]
    ax.plot(t, resid, color=C["calibrated"], lw=1.0)
    ax.axhline(0, color="k", lw=0.8, ls="--")
    rmse = float(np.sqrt((resid ** 2).mean()))
    ax.set_title(
        f"Residual Error  (calibrated − truth)  |  RMSE = {rmse:.3f} mm"
    )
    ax.set_ylabel("Error (mm)")
    ax.set_xlabel("Time (s)")
    ax.grid(alpha=0.3)

    plt.tight_layout()
    path = os.path.join(out_dir, "01_ultrasonic_pipeline.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"      Saved: {path}")


# ═══════════════════════════════════════════════════════════════════
#  Plot 2 — IMU Axes Dashboard
# ═══════════════════════════════════════════════════════════════════
def plot_imu_axes(df, out_dir: str):
    """
    3-row × 2-column grid.
    Left column  : angle signals  (raw / Kalman / Complementary / truth)
    Right column : Kalman bias estimate + corrected rate (twin y-axis)
    """
    fig = plt.figure(figsize=(16, 12))
    fig.suptitle("IMU Processing — Yaw / Pitch / Roll",
                 fontsize=15, fontweight="bold")
    gs = gridspec.GridSpec(3, 2, figure=fig, hspace=0.45, wspace=0.32)

    t = df["time_s"]

    axis_info = [
        ("yaw",   C["yaw"],   0),
        ("pitch", C["pitch"], 1),
        ("roll",  C["roll"],  2),
    ]

    for axis, color, row in axis_info:

        # ── Left: Angle Signals ──────────────────────────────────
        ax_l = fig.add_subplot(gs[row, 0])
        ax_l.plot(t, df[f"raw_{axis}_angle_deg"],
                  color=C["raw"], alpha=0.5, lw=0.8, label="Raw angle")
        ax_l.plot(t, df[f"{axis}_angle_comp"],
                  color=C["comp"], lw=1.0, alpha=0.85,
                  label="Complementary")
        ax_l.plot(t, df[f"{axis}_angle_kalman"],
                  color=color, lw=1.5, label="Kalman")
        ax_l.plot(t, df[f"true_{axis}_angle_deg"],
                  color=C["truth"], lw=1.5, ls="--",
                  label="Ground truth")

        rmse_k = df.attrs.get(f"{axis}_rmse_kalman", float("nan"))
        rmse_c = df.attrs.get(f"{axis}_rmse_comp",   float("nan"))
        ax_l.set_title(
            f"{axis.capitalize()} Angle  "
            f"[Kalman RMSE = {rmse_k:.4f}°  |  Comp RMSE = {rmse_c:.4f}°]",
            fontsize=9
        )
        ax_l.set_ylabel("Angle (°)")
        ax_l.legend(fontsize=7, loc="upper right")
        ax_l.grid(alpha=0.3)
        if row == 2:
            ax_l.set_xlabel("Time (s)")

        # ── Right: Bias Estimate + Corrected Rate ────────────────
        ax_r = fig.add_subplot(gs[row, 1])
        ax_r2 = ax_r.twinx()

        ax_r.plot(t, df[f"{axis}_bias_kalman"],
                  color=color, lw=1.2,
                  label="Kalman bias estimate")
        ax_r.axhline(
            df.attrs.get(f"{axis}_bias_deg_s", 0),
            color=color, lw=0.8, ls=":", alpha=0.6,
            label="Static bias (startup avg)"
        )
        ax_r.set_ylabel("Bias (°/s)", color=color)
        ax_r.tick_params(axis="y", labelcolor=color)

        ax_r2.plot(t, df[f"{axis}_corrected_rate"],
                   color=C["raw"], lw=0.6, alpha=0.4,
                   label="Corrected rate")
        ax_r2.set_ylabel("Rate (°/s)", color=C["raw"])
        ax_r2.tick_params(axis="y", labelcolor=C["raw"])

        ax_r.set_title(
            f"{axis.capitalize()} — Bias Tracking & Corrected Rate",
            fontsize=9
        )
        ax_r.grid(alpha=0.3)
        if row == 2:
            ax_r.set_xlabel("Time (s)")

        h1, l1 = ax_r.get_legend_handles_labels()
        h2, l2 = ax_r2.get_legend_handles_labels()
        ax_r.legend(h1 + h2, l1 + l2, fontsize=7, loc="upper right")

    path = os.path.join(out_dir, "02_imu_axes.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"      Saved: {path}")


# ═══════════════════════════════════════════════════════════════════
#  Plot 3 — Fusion Dashboard
# ═══════════════════════════════════════════════════════════════════
def plot_fusion_dashboard(fused, events, out_dir: str):
    """
    4-panel dashboard:
      1. Calibrated distance  +  proximity-alert shading
      2. IMU pitch & roll     +  tilt-alert shading
      3. Obstacle confidence score
      4. PID brake command
    """
    fig, axes = plt.subplots(4, 1, figsize=(14, 14), sharex=True)
    fig.suptitle("Sensor Fusion & Control Dashboard",
                 fontsize=15, fontweight="bold")

    t = fused["time_s"]

    # ── Panel 1: Distance ────────────────────────────────────────
    ax = axes[0]
    ax.plot(t, fused["calibrated_mm"],
            color=C["calibrated"], lw=1.5,
            label="Calibrated distance (mm)")
    ax.axhline(config.PROXIMITY_THRESH_MM,
               color=C["outlier"], lw=1.0, ls="--",
               label=f"Threshold ({config.PROXIMITY_THRESH_MM} mm)")

    _shade_alerts(ax, t.values, fused["proximity_alert"].values,
                  color=C["outlier"], alpha=0.15)

    ax.set_ylabel("Distance (mm)")
    ax.set_title("Ultrasonic Distance with Proximity Alerts")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    # ── Panel 2: IMU Angles ──────────────────────────────────────
    ax = axes[1]
    ax.plot(t, fused["pitch_angle_kalman"],
            color=C["pitch"], lw=1.2, label="Pitch (°)")
    ax.plot(t, fused["roll_angle_kalman"],
            color=C["roll"],  lw=1.2, label="Roll (°)")
    ax.axhline(config.TILT_THRESH_DEG,
               color="grey", lw=0.8, ls=":", alpha=0.7)
    ax.axhline(-config.TILT_THRESH_DEG,
               color="grey", lw=0.8, ls=":", alpha=0.7)

    _shade_alerts(ax, t.values, fused["tilt_alert"].values,
                  color="purple", alpha=0.10)

    ax.set_ylabel("Angle (°)")
    ax.set_title("IMU Pitch & Roll with Tilt Alerts")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    # ── Panel 3: Confidence Score ─────────────────────────────────
    ax = axes[2]
    ax.fill_between(t, fused["confidence"],
                    color=C["confidence"], alpha=0.35)
    ax.plot(t, fused["confidence"],
            color=C["confidence"], lw=1.2,
            label="Obstacle confidence")
    ax.axhline(0.5, color="k", lw=0.8, ls="--", alpha=0.5,
               label="Alert level (0.5)")
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Confidence [0–1]")
    ax.set_title("Fused Obstacle Confidence Score")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    # ── Panel 4: PID Brake Command ───────────────────────────────
    ax = axes[3]
    ax.fill_between(t, fused["brake_cmd"],
                    color=C["brake"], alpha=0.30)
    ax.plot(t, fused["brake_cmd"],
            color=C["brake"], lw=1.3,
            label=(f"PID brake  "
                   f"Kp={config.PID_KP} "
                   f"Ki={config.PID_KI} "
                   f"Kd={config.PID_KD}"))
    ax.set_ylim(-0.05, 1.05)
    ax.set_ylabel("Brake [0–1]")
    ax.set_xlabel("Time (s)")
    ax.set_title(
        f"PID Proximity Controller  "
        f"(setpoint = {config.PID_SETPOINT_MM} mm)"
    )
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    # Annotate first few proximity events on panel 1
    if not events.empty:
        for _, ev in (events[events["event_type"] == "PROXIMITY_ALERT"]
                      .head(5).iterrows()):
            axes[0].axvline(ev["time_s"],
                            color=C["outlier"], lw=0.7, alpha=0.6)

    plt.tight_layout()
    path = os.path.join(out_dir, "03_fusion_dashboard.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"      Saved: {path}")
