# 🚗 Vehicle DAQ Pipeline

> A Python pipeline that takes raw, noisy sensor data from a vehicle and turns it
> into clean, fused, actionable control signals — in real time.

![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python)
![License](https://img.shields.io/badge/License-MIT-green)
![Status](https://img.shields.io/badge/Status-Complete-brightgreen)
![Sensors](https://img.shields.io/badge/Sensors-Ultrasonic%20%2B%20IMU-orange)

---

## What is this?

This project builds a complete sensor data pipeline for a vehicle system. It reads
raw ultrasonic and IMU sensor data, cleans it up, filters it, fuses the two streams
together, and outputs a PID brake command — all in one automated pipeline.

Think of it as the signal processing brain between raw hardware and a vehicle's
control system.

---

## What it does

- Cleans ultrasonic distance data — rejects spikes and noise using Z-score
  filtering, then smooths the signal with a Kalman filter
- Tracks vehicle orientation — estimates Yaw, Pitch, and Roll angles from a
  noisy gyroscope using both a Kalman filter and a Complementary filter
- Fuses both sensors — combines distance and tilt into a single obstacle
  confidence score, time-aligned to within 5ms
- Controls braking — runs a PID controller that outputs a brake command from
  0 (no braking) to 1 (full stop) based on proximity to an obstacle
- Logs everything — saves a timestamped event log of every outlier, proximity
  alert, and tilt alert detected

---

## Results at a glance

| What was measured | Result |
|---|---|
| Ultrasonic samples processed | 3,000 |
| Sensor spikes caught & removed | 100 (3.3%) |
| Distance accuracy (RMSE) | **3.15 mm** |
| Yaw angle accuracy (Kalman) | **2.57°** |
| Pitch angle accuracy (Kalman) | **1.80°** |
| Roll angle accuracy (Kalman) | **1.37°** |
| Sample rate | 100 Hz |

The Kalman filter outperformed the Complementary filter on every axis —
up to **5.5× lower RMSE** on Roll.

---

## Project Structure
vehicle-daq-pipeline/
│
├── main.py # Run this — executes all 4 pipeline phases
├── config.py # All tunable parameters in one place
│
├── modules/
│ ├── _init_.py
│ ├── ultrasonic_processor.py # Outlier removal, Kalman filter, calibration
│ ├── imu_processor.py # Bias removal, Kalman & Complementary filters
│ ├── fusion.py # Sensor fusion, alerts, PID controller
│ └── visualiser.py # All three diagnostic plots
│
├── Data/
│ ├── ultrasonic_raw.csv
│ └── imu_raw.csv
│
└── outputs/ # Generated at runtime — not tracked by git
├── fused_data.csv
├── event_log.csv
├── calibration_report.csv
├── 01_ultrasonic_pipeline.png
├── 02_imu_axes.png
└── 03_fusion_dashboard.png

---

## Getting started

**1. Clone the repository**
```bash
git clone https://github.com/YOUR_USERNAME/vehicle-daq-pipeline.git
cd vehicle-daq-pipeline
```

**2. Install dependencies**
```bash
pip install numpy pandas scipy matplotlib
```

**3. Run the pipeline**
```bash
python main.py
```

All outputs are saved automatically to the `outputs/` folder.

---

## Pipeline overview
Raw CSV files
│
├──▶ Phase 1: Ultrasonic Processing
│ Z-score outlier rejection
│ Kalman filter (position + velocity state)
│ Linear calibration against ground truth
│
├──▶ Phase 2: IMU Processing
│ Static gyro bias removal
│ Complementary filter (α = 0.98)
│ Kalman filter [angle, bias] per axis
│ RMSE evaluation vs ground truth
│
├──▶ Phase 3: Sensor Fusion
│ Time-synchronised merge (±5 ms tolerance)
│ Obstacle confidence score (distance + tilt)
│ Proximity and tilt alert flags
│ PID proximity controller → brake command
│
└──▶ Phase 4: Outputs
3 diagnostic PNG plots
fused_data.csv | event_log.csv


---

## Configuration

Everything is controlled from `config.py` — no magic numbers anywhere else:

```python
# Proximity
PROXIMITY_THRESH_MM = 40.0     # Alert when closer than this
TILT_THRESH_DEG     = 30.0     # Alert when tilted beyond this

# PID controller
PID_KP              = 0.8
PID_KI              = 0.05
PID_KD              = 0.1
PID_SETPOINT_MM     = 50.0     # Target distance the PID tries to maintain

# Kalman (IMU)
KALMAN_Q_ANGLE      = 0.001
KALMAN_Q_BIAS       = 0.003
KALMAN_R_ANGLE      = 0.03

# Complementary filter
COMPLEMENTARY_ALPHA = 0.98

# Sensor fusion
FUSION_TOLERANCE_MS = 5.0      # Max time gap allowed when merging streams
```

---

## Output plots

### Ultrasonic DAQ Pipeline
Shows the full signal chain — raw spikes, Kalman-filtered output, calibration
quality, and residual error over 30 seconds.

### IMU Processing — Yaw / Pitch / Roll
Per-axis angle estimation: raw vs Complementary vs Kalman vs ground truth,
alongside dynamic bias tracking.

### Sensor Fusion & Control Dashboard
Four-panel view of calibrated distance, tilt angles, obstacle confidence score,
and PID brake output over the full run.

---

## Output files

| File | What's in it |
|---|---|
| `fused_data.csv` | Per-sample merged data: distance, angles, confidence, brake command |
| `event_log.csv` | Every detected outlier, proximity alert, and tilt alert with timestamps |
| `calibration_report.csv` | Ultrasonic calibration statistics per sample |

---

## Dependencies

| Package | Version | Used for |
|---|---|---|
| `numpy` | ≥ 1.24 | Matrix operations, Kalman math |
| `pandas` | ≥ 2.0 | Data loading and processing |
| `scipy` | ≥ 1.10 | Z-score filtering, linear regression |
| `matplotlib` | ≥ 3.7 | All three output plots |

---

## License

MIT — free to use, modify, and distribute.

---

## Author

Built by **Avishkar Mahadik** — feel free to open an issue or pull request if you
want to improve something or adapt it for your own sensor setup.

⭐ If this was useful, a star on the repo goes a long way!

