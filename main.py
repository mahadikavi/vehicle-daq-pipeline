"""
main.py
Automated Control System & Sensor Integration — Full Pipeline

"""

import importlib
import os
import sys

PROJECT_ROOT = r"E:\Projects\Python\Vehicle DAQ Pipeline"
sys.path.insert(0, PROJECT_ROOT)

config = importlib.import_module("config")

from modules.fusion import run_fusion
from modules.imu_processor import process_imu
from modules.ultrasonic_processor import process_ultrasonic
from modules.visualiser import (plot_fusion_dashboard, plot_imu_axes,
                                plot_ultrasonic_pipeline)

def main():
    os.makedirs(config.OUTPUT_DIR, exist_ok=True)

    print("=" * 60)
    print("  DAQ Control System — Sensor Integration Pipeline")
    print("=" * 60)

    # ── Phase 1: Ultrasonic ──────────────────────────────────────
    print("\n[1/4] Processing ultrasonic sensor data...")
    ultra_df = process_ultrasonic(
        os.path.join(config.DATA_DIR, "ultrasonic_raw.csv")
    )
    rmse_u = ((ultra_df["calibrated_mm"]
               - ultra_df["true_distance_mm"]) ** 2).mean() ** 0.5
    print(f"      Samples    : {len(ultra_df)}")
    print(f"      Outliers   : {ultra_df['is_outlier'].sum()}")
    print(f"      Cal R²     : {ultra_df.attrs['cal_r2']:.6f}")
    print(f"      RMSE       : {rmse_u:.3f} mm")

    # ── Phase 2: IMU ─────────────────────────────────────────────
    print("\n[2/4] Processing IMU data (yaw / pitch / roll)...")
    imu_df = process_imu(
        os.path.join(config.DATA_DIR, "imu_raw.csv")
    )

    # ── Phase 3: Sensor Fusion ───────────────────────────────────
    print("\n[3/4] Running sensor fusion and control logic...")
    result = run_fusion(ultra_df, imu_df)
    fused  = result["fused"]
    events = result["events"]
    print(f"      Fused samples    : {len(fused)}")
    print(f"      Total events     : {len(events)}")
    print(f"      Proximity alerts : {fused['proximity_alert'].sum()}")
    print(f"      Tilt alerts      : {fused['tilt_alert'].sum()}")

    # ── Phase 4: Save outputs & plots ────────────────────────────
    print("\n[4/4] Generating outputs...")
    fused.to_csv(
        os.path.join(config.OUTPUT_DIR, "fused_data.csv"), index=False
    )
    events.to_csv(
        os.path.join(config.OUTPUT_DIR, "event_log.csv"), index=False
    )

    plot_ultrasonic_pipeline(ultra_df, config.OUTPUT_DIR)
    plot_imu_axes(imu_df, config.OUTPUT_DIR)
    plot_fusion_dashboard(fused, events, config.OUTPUT_DIR)

    print("\n✓  Done. Outputs saved to:", config.OUTPUT_DIR)
    print("   fused_data.csv  |  event_log.csv")
    print("   01_ultrasonic_pipeline.png")
    print("   02_imu_axes.png")
    print("   03_fusion_dashboard.png")


if __name__ == "__main__":
    main()