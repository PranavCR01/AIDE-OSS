## config.py

import os

# Base directory (auto-detected from current file location)
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Path to raw log file
RAW_LOG_FILE = os.path.join(BASE_DIR, "data", "sample_logs_small.txt")

# Intermediate file (after anomaly detection)
ANOMALY_CSV_FILE = os.path.join(BASE_DIR, "data", "logs_with_anomalies.csv")

# Final output file (after summarization)
SUMMARY_CSV_FILE = os.path.join(BASE_DIR, "data", "logs_with_summaries.csv")

# Directory where models are stored
MODEL_DIR = os.path.join(BASE_DIR, "models")

# Path to anomaly detection model
MODEL_PATH = os.path.join(MODEL_DIR, "isolation_forest.pkl")

# Path to encoder used for anomaly detection
ENCODER_PATH = os.path.join(MODEL_DIR, "encoder.pkl")
