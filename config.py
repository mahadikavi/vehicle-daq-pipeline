# config.py — DAQ Control System Configuration

import os

BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
DATA_DIR   = os.path.join(BASE_DIR, "Data")        
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")    

# ── Ultrasonic settings ──────────────────────────────────────────
SPEED_OF_SOUND_MS   = 343.0
OUTLIER_Z_THRESH    = 3.0
KALMAN_Q_ULTRA      = 1e-4
KALMAN_R_ULTRA      = 0.5
CAL_SLOPE           = 0.94825454
CAL_INTERCEPT       = 2.55147578
CHASSIS_OFFSET_MM   = 15.0
RIDE_HEIGHT_MIN_MM  = 30.0
RIDE_HEIGHT_MAX_MM  = 120.0

# ── IMU settings ─────────────────────────────────────────────────
GYRO_BIAS_WINDOW    = 50
KALMAN_Q_ANGLE      = 1e-4
KALMAN_Q_BIAS       = 1e-5
KALMAN_R_ANGLE      = 0.03
COMPLEMENTARY_ALPHA = 0.98
TILT_ALERT_DEG      = 30.0
YAW_RATE_ALERT      = 40.0
YAW_BIAS            = 18.30484
PITCH_BIAS          = 3.48519
ROLL_BIAS           = 18.01555

# ── Fusion / control settings ────────────────────────────────────
FUSION_TOLERANCE_MS = 5
PROXIMITY_THRESH_MM = 40.0
TILT_THRESH_DEG     = 30.0
OBSTACLE_W_DIST     = 0.7
OBSTACLE_W_TILT     = 0.3

# ── PID gains ────────────────────────────────────────────────────
PID_KP          = 0.8
PID_KI          = 0.05
PID_KD          = 0.1
PID_SETPOINT_MM = 50.0