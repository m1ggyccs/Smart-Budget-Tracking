"""
training.py

Trains three forecasting models (Moving Average, Holt-Winters, LSTM)
for each spending category using preprocessed data.

Inputs:
 - data/processed/cleaned_transactions.csv (from preprocessing.py)

Outputs:
 - models/moving_average_<cat>.json
 - models/holtwinters_<cat>.pkl
 - models/lstm_<cat>.h5
 - models/scaler_<cat>.pkl
 - models/metrics_<cat>.json
"""

import os
import json
import pickle
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam

from statsmodels.tsa.holtwinters import ExponentialSmoothing

# Paths
ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "data" / "processed" / "cleaned_transactions.csv"
MODELS_DIR = ROOT / "models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)

# Hyperparameters
SEQ_LEN = 12           # 12 months or 12 weeks
EPOCHS = 80
BATCH_SIZE = 8


# ==========================
# Helper functions
# ==========================

def safe_name(cat: str) -> str:
    return cat.lower().replace(" ", "_")


def make_sequence_data(series, seq_len=SEQ_LEN):
    X, y = [], []
    vals = series.values
    for i in range(len(vals) - seq_len):
        X.append(vals[i:i + seq_len])
        y.append(vals[i + seq_len])
    X = np.array(X).reshape(-1, seq_len, 1)
    y = np.array(y)
    return X, y


def build_lstm(seq_len=SEQ_LEN):
    model = Sequential([
        LSTM(64, return_sequences=True, input_shape=(seq_len, 1)),
        LSTM(32, activation="tanh"),
        Dense(16, activation="relu"),
        Dense(1)
    ])
    model.compile(optimizer=Adam(learning_rate=0.001), loss="mse")
    return model


def moving_average_forecast(series, window=3):
    """Save last window values for later rolling predictions"""
    vals = series.values
    if len(vals) < window:
        avg = np.mean(vals) if len(vals) > 0 else 0.0
    else:
        avg = np.mean(vals[-window:])
    return {"window": window, "mean": float(avg), "last_values": vals[-window:].tolist()}


# ==========================
# Model training
# ==========================

def train_models_for_category(category: str, series: pd.Series) -> Dict[str, bool]:
    """Train 3 models (MA, HW, LSTM) for one category"""
    results = {"ma": False, "hw": False, "lstm": False}

    # Skip empty
    if len(series.dropna()) < 5:
        print(f"[!] Skipping '{category}' - insufficient data.")
        return results

    print(f"\n=== Training models for category: {category} ===")

    # ----- Moving Average -----
    ma_params = moving_average_forecast(series)
    ma_path = MODELS_DIR / f"moving_average_{safe_name(category)}.json"
    with open(ma_path, "w", encoding="utf-8") as f:
        json.dump(ma_params, f, indent=2)
    results["ma"] = True
    print(f"[MA] Saved -> {ma_path}")

    # ----- Holt-Winters -----
    try:
        hw_model = ExponentialSmoothing(series, trend="add", seasonal=None, damped_trend=True).fit()
        hw_path = MODELS_DIR / f"holtwinters_{safe_name(category)}.pkl"
        with open(hw_path, "wb") as f:
            pickle.dump(hw_model, f)
        results["hw"] = True
        print(f"[HW] Saved -> {hw_path}")
    except Exception as e:
        print(f"[HW] Failed for {category}: {e}")

    # ----- LSTM -----
    try:
        vals = series.values.astype("float32")
        scaler = MinMaxScaler(feature_range=(0, 1))
        vals_scaled = scaler.fit_transform(vals.reshape(-1, 1)).flatten()

        if len(vals_scaled) <= SEQ_LEN + 2:
            print(f"[LSTM] Skipping {category}: not enough history ({len(vals_scaled)} pts)")
            return results

        X, y = make_sequence_data(pd.Series(vals_scaled), SEQ_LEN)
        split = max(1, int(len(X) * 0.9))
        X_train, X_val = X[:split], X[split:]
        y_train, y_val = y[:split], y[split:]

        model = build_lstm(SEQ_LEN)
        es = EarlyStopping(monitor="val_loss", patience=6, restore_best_weights=True, verbose=1)
        model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val) if len(X_val) > 0 else None,
            epochs=EPOCHS,
            batch_size=BATCH_SIZE,
            verbose=2,
            callbacks=[es]
        )

        # evaluate
        preds_val = model.predict(X_val).flatten() if len(X_val) > 0 else []
        if len(preds_val) > 0:
            mae = mean_absolute_error(y_val, preds_val)
            rmse = mean_squared_error(y_val, preds_val, squared=False)
        else:
            mae, rmse = None, None

        # save model & scaler
        model_path = MODELS_DIR / f"lstm_{safe_name(category)}.h5"
        model.save(model_path, include_optimizer=False)
        scaler_path = MODELS_DIR / f"scaler_{safe_name(category)}.pkl"
        with open(scaler_path, "wb") as f:
            pickle.dump(scaler, f)

        # save metrics
        metrics = {"mae": mae, "rmse": rmse, "points": len(series)}
        metrics_path = MODELS_DIR / f"metrics_{safe_name(category)}.json"
        with open(metrics_path, "w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2)

        results["lstm"] = True
        print(f"[LSTM] Saved -> {model_path} | Metrics: {metrics}")
    except Exception as e:
        print(f"[LSTM] Failed for {category}: {e}")

    return results


# ==========================
# Main orchestrator
# ==========================

def load_cleaned_data() -> pd.DataFrame:
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Cleaned transactions not found at {DATA_PATH}. Run preprocessing.py first.")
    df = pd.read_csv(DATA_PATH, parse_dates=["date"])
    df["category"] = df["category"].astype(str).str.lower().str.strip()
    return df


def aggregate_series(df: pd.DataFrame, freq: str = "ME") -> Dict[str, pd.Series]:
    """Aggregate per category by given frequency (ME=monthly, W-MON=weekly)"""
    grouped = {}
    for cat, group in df.groupby("category"):
        s = group.set_index("date")["amount"].resample(freq).sum().sort_index()
        s = s.fillna(0.0)
        grouped[cat] = s
    return grouped


def train_all(freq: str = "ME"):
    df = load_cleaned_data()
    all_series = aggregate_series(df, freq=freq)
    results = {}
    for cat, series in all_series.items():
        results[cat] = train_models_for_category(cat, series)
    # Summary
    trained = sum(sum(v.values()) for v in results.values())
    print(f"\n✅ Training complete: {trained} models saved.")
    return results

# =====================================================
# Append new data directly into processed cleaned CSV and retrain
# =====================================================

def append_and_retrain(new_data=None, retrain_all=False, freq="ME"):
    """
    Append new user data (from CSV or manual entry) to processed dataset,
    then retrain models for affected categories.
    """
    from pandas.tseries.offsets import MonthEnd

    processed_path = os.path.join("data", "processed", "cleaned_transactions.csv")
    if not os.path.exists(processed_path):
        print(f"⚠️ No existing processed data found at {processed_path}.")
        return

    df = pd.read_csv(processed_path, parse_dates=["date"])
    df["category"] = df["category"].astype(str).str.lower().str.strip()

    # --- Interactive data entry (manual monthly mode) ---
    if new_data is None:
        print("\n🆕 Add monthly transactions (enter blank Month to finish).")
        entries = []
        while True:
            month_str = input("Month (YYYY-MM) or ENTER to finish: ").strip()
            if not month_str:
                break

            # Allow YYYY/MM as well
            month_str = month_str.replace("/", "-")

            # Robust month parsing
            try:
                ts = pd.to_datetime(month_str + "-01", errors="coerce")
                if pd.isna(ts):
                    raise ValueError("Invalid month format")
                # Convert to month-end safely (avoid deprecated 'ME')
                month_end = (ts + MonthEnd(0)).normalize()
            except Exception as e:
                print(f"  ❌ Could not parse month '{month_str}': {e}. Try format YYYY-MM (e.g. 2025-11).")
                continue

            # Category
            cat = input("Category: ").strip().lower()
            if not cat:
                print("  ❌ Category cannot be empty. Try again.")
                continue

            # Amount
            amt_str = input("Amount (₱): ").strip()
            try:
                amt = float(amt_str) if amt_str else 0.0
            except:
                print("  ❌ Invalid amount. Enter a numeric value.")
                continue

            entries.append({"date": month_end, "category": cat, "amount": amt})
            print(f"  ✓ Added: {month_end.date()} | {cat} | ₱{amt:.2f}")

        if not entries:
            print("⚠️ No new data entered. Nothing to update.")
            return

        new_df = pd.DataFrame(entries)

    else:
        # CSV path or DataFrame provided externally
        new_df = new_data.copy()
        if "date" not in new_df.columns or "category" not in new_df.columns or "amount" not in new_df.columns:
            print("⚠️ CSV must contain columns: date, category, amount.")
            return
        new_df["date"] = pd.to_datetime(new_df["date"], errors="coerce")
        new_df["category"] = new_df["category"].astype(str).str.lower().str.strip()

    # --- Append & deduplicate ---
    updated_df = pd.concat([df, new_df], ignore_index=True).drop_duplicates(subset=["date", "category"], keep="last")
    updated_df.sort_values("date", inplace=True)
    updated_df.to_csv(processed_path, index=False)
    print(f"\n✅ Updated processed data saved → {processed_path}")

    # --- Retrain affected categories ---
    affected_cats = sorted(new_df["category"].dropna().unique().tolist())
    if not affected_cats:
        print("No affected categories detected; nothing to retrain.")
        return

    print(f"\n🔁 Retraining models for categories: {affected_cats}")

    # For each affected category, build a time series at the requested freq and train
    for cat in affected_cats:
        cat_df = updated_df[updated_df["category"] == cat].copy()
        # aggregate using the provided freq (e.g., "ME" for month-end or "W-MON" for weekly)
        try:
            series = cat_df.set_index("date")["amount"].resample(freq).sum().sort_index()
            series = series.fillna(0.0)
        except Exception as e:
            print(f"  ⚠️ Failed to resample for {cat} with freq='{freq}': {e}. Skipping.")
            continue

        # train_models_for_category expects (category, series)
        try:
            res = train_models_for_category(cat, series)
            ok_any = any(res.values()) if isinstance(res, dict) else bool(res)
            if ok_any:
                print(f"  ✅ Retrained models for category: {cat}")
            else:
                print(f"  ⚠️ Retraining skipped/failed for {cat} (insufficient data or error).")
        except Exception as e:
            print(f"  ⚠️ Error while retraining {cat}: {e}")

    print("\nAll done — data updated and models refreshed.")



if __name__ == "__main__":
    print("=== Smart Budget Tracker — Training ===")
    print(f"Loading cleaned data from {DATA_PATH}")
    results = train_all(freq="ME")  # use "W-MON" for weekly LSTM
