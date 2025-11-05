"""
simulation_realworld_v2.py
CLI simulation using three forecasting approaches:
 - Moving Average
 - Holt-Winters (Exponential Smoothing)
 - Pretrained LSTM models (per category-group) + saved scalers

Assumptions:
 - Historical transactions CSV: data/cleaned_transactions.csv
   expected columns: ['date', 'amount', 'category'] (date parseable)
 - LSTM models & scalers (optional): models/lstm_<group>.h5 and models/scaler_<group>.pkl
 - Outputs written to outputs/
"""

import os
import sys
import json
from pathlib import Path
import warnings

import pandas as pd
import numpy as np

# Optional libs (LSTM & Holt-Winters)
try:
    import tensorflow as tf
    from tensorflow.keras.models import load_model
except Exception:
    tf = None
try:
    import pickle
except Exception:
    pickle = None
try:
    from statsmodels.tsa.holtwinters import ExponentialSmoothing
except Exception:
    ExponentialSmoothing = None

ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "data" / "processed" / "cleaned_transactions.csv"
MODELS_DIR = ROOT / "models"
OUTPUTS_DIR = ROOT / "outputs"
OUTPUTS_DIR.mkdir(exist_ok=True)

# Map category names or keywords to LSTM model group filenames
LSTM_GROUP_MAP = {
    "groceries": "groceries",
    "food": "groceries",
    "transportation": "transportation",
    "transpo": "transportation",
    # default to "others"
}

DEFAULT_GROUP = "others"
LSTM_MODEL_TEMPLATE = "lstm_{}.h5"
SCALER_TEMPLATE = "scaler_{}.pkl"


def safe_load_csv(path):
    if not path.exists():
        return None
    df = pd.read_csv(path, parse_dates=["date"])
    return df


def list_categories(df):
    if df is None or df.empty:
        return []
    return sorted(df["category"].dropna().unique().tolist())


def monthly_category_series(df, category):
    # Returns a monthly series of total spend per month for given category (descending dates -> ascending index)
    dfc = df[df["category"].str.lower().str.contains(category.lower(), na=False)]
    if dfc.empty:
        # try exact match
        dfc = df[df["category"].str.lower() == category.lower()]
    if dfc.empty:
        return pd.Series(dtype=float)
    s = dfc.set_index("date").amount.resample("M").sum().sort_index()
    return s


def moving_average_forecast(series, months, window=3):
    if len(series) == 0:
        return [0.0] * months
    ma = []
    arr = series.values
    for i in range(months):
        if len(arr) >= window:
            ma_val = arr[-window:].mean()
        else:
            ma_val = arr.mean()
        ma.append(float(ma_val))
        # append the forecast as if it were real for iterative multi-step forecasting
        arr = np.append(arr, ma_val)
    return ma


def holt_winters_forecast(series, months):
    if ExponentialSmoothing is None:
        raise RuntimeError("statsmodels is required for Holt-Winters. Install statsmodels.")
    if len(series) < 2:
        # not enough data to fit, fallback to last value or zero
        last = float(series.iloc[-1]) if len(series) else 0.0
        return [last] * months
    # multiplicative vs additive: choose additive for sums
    model = ExponentialSmoothing(series, trend="add", seasonal=None, damped_trend=True)
    fit = model.fit(optimized=True)
    pred = fit.forecast(months)
    return [float(x) for x in pred]


def load_lstm_and_scaler(group):
    """
    Loads model and scaler for group if present.
    returns (model, scaler) or (None, None)
    """
    model_path = MODELS_DIR / LSTM_MODEL_TEMPLATE.format(group)
    scaler_path = MODELS_DIR / SCALER_TEMPLATE.format(group)
    model = None
    scaler = None
    if tf is not None and model_path.exists():
        try:
            model = load_model(str(model_path))
        except Exception as e:
            print(f"Warning: failed to load LSTM model for {group}: {e}")
            model = None
    if pickle is not None and scaler_path.exists():
        try:
            with open(scaler_path, "rb") as f:
                scaler = pickle.load(f)
        except Exception as e:
            print(f"Warning: failed to load scaler for {group}: {e}")
            scaler = None
    return model, scaler


def lstm_forecast(series, months, model, scaler, seq_len=12):
    """
    Multi-step prediction using pretrained LSTM.
    series: pandas Series (monthly sums)
    model: keras model
    scaler: sklearn scaler fit on training monthly values shape (n,1)
    seq_len: length of input sequence expected by model
    """
    if model is None or scaler is None:
        # fallback to moving average if no model
        return moving_average_forecast(series, months)
    # build input array
    arr = series.values.astype(float)
    # if not enough length, pad with last value
    if len(arr) < seq_len:
        pad = np.full((seq_len - len(arr),), arr[-1] if len(arr) else 0.0)
        arr_in = np.concatenate([pad, arr])
    else:
        arr_in = arr[-seq_len:]
    preds = []
    for _ in range(months):
        x = arr_in.reshape(-1, 1)
        # scale
        x_scaled = scaler.transform(x)  # shape (seq_len,1)
        x_scaled = x_scaled.reshape(1, seq_len, 1)
        yhat = model.predict(x_scaled, verbose=0)
        # model output shape depends; assume scalar
        yhat_inv = scaler.inverse_transform(yhat.reshape(-1, 1)).flatten()[0]
        preds.append(float(yhat_inv))
        # append predicted value to arr_in and shift
        arr_in = np.append(arr_in[1:], yhat_inv)
    return preds


def choose_lstm_group(category):
    cat_lower = category.lower()
    for k, v in LSTM_GROUP_MAP.items():
        if k in cat_lower:
            return v
    return DEFAULT_GROUP


def ensure_data_file():
    if not DATA_PATH.exists():
        print(f"No historical data found at {DATA_PATH}.")
        # Ask user to input last-month expenses per category
        print("Let's create a baseline. Please enter your last month's expenses per category.")
        entries = []
        # minimal categories fallback - read from your project categories file if you have it.
        default_cats = ["Groceries", "Transportation", "Utilities", "Entertainment", "Others"]
        print("Enter values for categories (press Enter to accept 0):")
        for c in default_cats:
            v = input(f"  {c}: ").strip()
            amt = float(v) if v else 0.0
            # write a single transaction at end-of-last-month date
            last_month = pd.Timestamp.today().normalize() - pd.offsets.MonthBegin(1)
            entries.append({"date": last_month, "amount": amt, "category": c})
        df_new = pd.DataFrame(entries)
        df_new.to_csv(DATA_PATH, index=False)
        print(f"Baseline saved to {DATA_PATH}. Restart the script to continue.")
        sys.exit(0)


def append_user_csv_to_data(user_csv_path):
    df_user = pd.read_csv(user_csv_path, parse_dates=["date"])
    df_main = safe_load_csv(DATA_PATH) or pd.DataFrame(columns=df_user.columns)
    df_combined = pd.concat([df_main, df_user], ignore_index=True)
    df_combined.to_csv(DATA_PATH, index=False)
    print(f"Appended {len(df_user)} rows to {DATA_PATH}.")


def prediction_flow(df):
    cats = list_categories(df)
    print("Categories detected in your data:")
    for i, c in enumerate(cats, 1):
        print(f"  {i}. {c}")
    print("You may enter a category name from the list above or 'ALL' to forecast for every category.")
    cat_input = input("Category (or ALL): ").strip()
    months_str = input("How many future months to predict? (e.g., 1 or 3): ").strip()
    try:
        months = int(months_str)
        if months < 1:
            months = 1
    except:
        months = 1
    budget_str = input("Enter your monthly budget (numeric): ").strip()
    try:
        monthly_budget = float(budget_str)
    except:
        monthly_budget = None

    if cat_input.lower() == "all":
        target_cats = cats
    else:
        target_cats = [cat_input] if cat_input else cats

    results = []
    for cat in target_cats:
        series = monthly_category_series(df, cat)
        # If completely empty, we allow user to add last-month baseline of 0
        if series.empty:
            # create a short series of zeros to allow methods to run
            series = pd.Series([0.0] * 3, index=pd.date_range(end=pd.Timestamp.today(), periods=3, freq="M"))

        # Moving average
        ma_preds = moving_average_forecast(series, months)

        # Holt-Winters
        try:
            hw_preds = holt_winters_forecast(series, months)
        except Exception as e:
            print(f"Holt-Winters failed for {cat}: {e}. Using MA fallback.")
            hw_preds = ma_preds

        # LSTM
        group = choose_lstm_group(cat)
        model, scaler = load_lstm_and_scaler(group)
        try:
            lstm_preds = lstm_forecast(series, months, model, scaler)
        except Exception as e:
            print(f"LSTM forecasting failed for {cat}: {e}. Using MA fallback.")
            lstm_preds = ma_preds

        # aggregate predictions per month with model label
        for m in range(months):
            results.append({
                "category": cat,
                "month_ahead": m + 1,
                "moving_average": ma_preds[m],
                "holt_winters": hw_preds[m],
                "lstm": lstm_preds[m],
            })

    df_res = pd.DataFrame(results)
    # write outputs per-model CSVs and a summary
    for model_label in ["moving_average", "holt_winters", "lstm"]:
        out_path = OUTPUTS_DIR / f"predictions_{model_label}.csv"
        model_df = df_res[["category", "month_ahead", model_label]].rename(columns={model_label: "predicted"})
        model_df.to_csv(out_path, index=False)
        print(f"Wrote predictions for {model_label} to {out_path}")

    # Budget check: sum predictions across categories for each month for each model
    if monthly_budget is not None:
        for model_label in ["moving_average", "holt_winters", "lstm"]:
            pivot = df_res.pivot_table(index="month_ahead", columns="category", values=model_label, aggfunc="sum", fill_value=0.0)
            pivot["total"] = pivot.sum(axis=1)
            print(f"\nBudget check for model: {model_label}")
            for idx, row in pivot.iterrows():
                over = row["total"] - monthly_budget
                status = "OVER BUDGET" if over > 0 else "within budget"
                print(f"  Month +{idx}: predicted total {row['total']:.2f} => {status} (diff {over:.2f})")
    else:
        print("Monthly budget not provided; skipping budget check.")

    print("\nPrediction flow complete. CSVs saved in outputs/.")


def add_data_flow():
    print("Add Data Options:")
    print("  1) Provide path to a CSV to append (must have date, amount, category columns)")
    print("  2) Manually add a single transaction")
    choice = input("Choose 1 or 2: ").strip()
    if choice == "1":
        p = input("Path to CSV: ").strip()
        user_csv = Path(p)
        if user_csv.exists():
            append_user_csv_to_data(user_csv)
        else:
            print("File not found.")
    else:
        d = input("Date (YYYY-MM-DD): ").strip()
        a = float(input("Amount: ").strip() or 0.0)
        c = input("Category: ").strip()
        df_main = safe_load_csv(DATA_PATH) or pd.DataFrame(columns=["date", "amount", "category"])
        df_new = pd.DataFrame([{"date": d, "amount": a, "category": c}])
        df_combined = pd.concat([df_main, df_new], ignore_index=True)
        df_combined.to_csv(DATA_PATH, index=False)
        print("Transaction added.")


def main():
    print("=== Smart Budget Predictor — Simulation ===")
    ensure_data_file()
    df = safe_load_csv(DATA_PATH)
    if df is None:
        print("No data found after ensure step, exiting.")
        return

    # clean columns expectations
    if "amount" not in df.columns or "date" not in df.columns or "category" not in df.columns:
        print("Data file missing required columns (date, amount, category).")
        return

    action = input("Type 'predict' to predict next months, 'add' to add data, or 'both': ").strip().lower()
    if action not in ("predict", "add", "both"):
        print("Unknown option; defaulting to 'predict'.")
        action = "predict"

    if action in ("add", "both"):
        add_data_flow()
        # reload data after addition
        df = safe_load_csv(DATA_PATH)

    if action in ("predict", "both"):
        prediction_flow(df)


if __name__ == "__main__":
    main()
