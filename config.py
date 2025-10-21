# config.py
import os

# Paths
DATA_DIR = "data"
RAW_PATH = os.path.join(DATA_DIR, "raw", "daily_transactions.csv")
PROCESSED_PATH = os.path.join(DATA_DIR, "processed", "cleaned_transactions.csv")

# Model output folders
MODEL_DIR = "models"
OUTPUT_DIR = "outputs"
FORECAST_DIR = os.path.join(OUTPUT_DIR, "forecasts")
ANOMALY_DIR = os.path.join(OUTPUT_DIR, "anomalies")
PLOT_DIR = os.path.join(OUTPUT_DIR, "plots")

# Conversion rate (INR -> PHP)
INR_TO_PHP = 0.661

# Train/Validation/Test ratios
TRAIN_FRAC = 0.7
VAL_FRAC = 0.15

# Forecasting config
WEEKS_AHEAD = 4
SEQ_LENGTH = 8
RANDOM_SEED = 42
