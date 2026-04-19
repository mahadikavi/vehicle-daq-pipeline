""""
Sensor_Simulator.py
==============================
PHASE 0 - HARDWARE ABSTRACTION LAYER
Abstracts hardware vs. CSV replay

Modes:
"replay" - replays data from a CSV file
"live" - simulates live sensor data (for testing purposes)

Switching from CSV to live hardware = change ONE line in main.py which is given in README file.
"""

import time
import numpy as np
import pandas as pd
import yaml
import os

_BASE = os.path.dirname(os.path.abspath(__file__))
_DEFAULT_CONFIG = os.path.join(_BASE, "config.yaml")


class SensorSimulator:
    def __init__(self, config_path=_DEFAULT_CONFIG, mode="replay", port=None):

        with open(config_path) as f:
            self.cfg = yaml.safe_load(f)
        self.mode = mode
        self._ultra_df = None
        self._imu_df = None
        self._serial = None

        if mode == "replay":
            self._load_csv_data()
        elif mode == "live":
            self._init_serial(port)


# ─────── CSV Replay Methods ───────────────────────────────────────────


    def _load_csv_data(self):
        # Build absolute path to Data folder relative to this .py file
        BASE_DIR = os.path.dirname(os.path.abspath(__file__))
        raw = os.path.join(BASE_DIR, self.cfg["paths"]["raw_data_dir"])

        self._ultra_df = pd.read_csv(os.path.join(raw, "ultrasonic_raw.csv"))
        self._imu_df = pd.read_csv(os.path.join(raw, "imu_raw.csv"))
        self._ultra_df = self._ultra_df.sort_values(
            "time_s").reset_index(drop=True)
        self._imu_df = self._imu_df.sort_values(
            "time_s").reset_index(drop=True)
        print(
            f"[Simulator] Replay: {len(self._ultra_df)} ultrasonic,"
            f" {len(self._imu_df)} IMU samples loaded."
        )

    def get_ultrasonic_sample(self, idx):
        row = self._ultra_df.iloc[idx]
        return {
            "time_s":       float(row["time_s"]),
            "echo_time_us": float(row["echo_time_us"]),
            "noisy_mm":     float(row["noisy_mm"]),
            "true_mm":      float(row["true_distance_mm"]),
        }

    def get_imu_sample(self, idx):
        row = self._imu_df.iloc[idx]
        return {
            "time_s":               float(row["time_s"]),
            "raw_yaw_rate_deg_s":   float(row["raw_yaw_rate_deg_s"]),
            "raw_pitch_rate_deg_s": float(row["raw_pitch_rate_deg_s"]),
            "raw_roll_rate_deg_s":  float(row["raw_roll_rate_deg_s"]),
            "raw_yaw_angle_deg":    float(row["raw_yaw_angle_deg"]),
            "raw_pitch_angle_deg":  float(row["raw_pitch_angle_deg"]),
            "raw_roll_angle_deg":   float(row["raw_roll_angle_deg"]),
        }

    def get_all_ultrasonic(self):
        return self._ultra_df.copy()

    def get_all_imu(self):
        return self._imu_df.copy()


# ─────── Live Serial ───────────────────────────────────────────


    def _init_serial(self, port):
        try:
            import serial
            self._serial = serial.Serial(port, baudrate=115200, timeout=1.0)
            time.sleep(2)
            print(f"[Simulator] Live: connected to {port}")
        except ImportError:
            raise ImportError("Run: pip install pyserial")
        except Exception as e:
            raise ConnectionError(f"Cannot open {port}: {e}")

    def read_live_sample(self):
        """Read one CSV line from Arduino: t_ms,echo_us,gx,gy,gz"""
        if not self._serial:
            return None
        try:
            line = self._serial.readline().decode("utf-8").strip()
            parts = [float(x) for x in line.split(",")]
            return {
                "time_s":               parts[0] / 1000.0,
                "echo_time_us":         parts[1],
                "raw_yaw_rate_deg_s":   parts[2],
                "raw_pitch_rate_deg_s": parts[3],
                "raw_roll_rate_deg_s":  parts[4],
            }
        except Exception:
            return None

    def close(self):
        if self._serial:
            self._serial.close()


# ─────── Testing Code ───────────────────────────────────────────

""" if __name__ == "__main__":
    import os

    # Automatically find config.yaml next to this .py file
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    CONFIG = os.path.join(BASE_DIR, "config.yaml")

    sim = SensorSimulator(CONFIG, mode="replay")

    print("\n--- Ultrasonic sample 0 ---")
    print(sim.get_ultrasonic_sample(0))

    print("\n--- IMU sample 0 ---")
    print(sim.get_imu_sample(0))

    print("\n--- Ultrasonic sample 5 ---")
    print(sim.get_ultrasonic_sample(5)) """
