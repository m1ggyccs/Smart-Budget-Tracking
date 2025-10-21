# simulation_realworld.py
"""
Interactive real-world simulation for LSTM weekly forecasts.

Run:
    python simulation_realworld.py

This version:
- Prompts user for category, history length, prediction weeks, and optional manual input.
- Loads your model (.keras or .h5) and scaler (.pkl).
- Predicts next weekly spending using your trained LSTM models.
"""

import numpy as np
import pandas as pd
from pathlib import Path
from tensorflow.keras.models import load_model
import joblib

# ---- Config ----
MODELS_DIR = Path("models")
OUTPUT_DIR = Path("outputs")
OUTPUT_DIR.mkdir(exist_ok=True, parents=True)
DEFAULT_TIMESTEPS = 4


# ------------------ Utility Functions ------------------

def safe_load_model(model_path):
    model_path = Path(model_path)
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")
    try:
        model = load_model(str(model_path), compile=False)
    except Exception:
        from tensorflow.keras.metrics import MeanSquaredError
        model = load_model(str(model_path),
                           custom_objects={"mse": MeanSquaredError()},
                           compile=False)
    model.compile(optimizer="adam", loss="mse", metrics=["mse"])
    return model


def load_scaler(scaler_path):
    return joblib.load(str(scaler_path))


def prepare_window(series, timesteps=DEFAULT_TIMESTEPS):
    arr = np.asarray(series).astype(float).flatten()
    if arr.size < timesteps:
        raise ValueError(f"Need at least {timesteps} weeks of data; got {arr.size}")
    last_window = arr[-timesteps:]
    x = last_window.reshape((1, timesteps, 1))
    return x


def baseline_median_predictor(series, n_weeks=1):
    med = float(np.median(series))
    return [med] * n_weeks


def predict_next_weeks(category, recent_series, n_weeks=1, timesteps=DEFAULT_TIMESTEPS):
    """Predict next N weeks using trained model + scaler."""
    model_path = None
    for ext in [".keras", ".h5"]:
        candidate = MODELS_DIR / f"lstm_{category}{ext}"
        if candidate.exists():
            model_path = candidate
            break
    if not model_path:
        print(f"[warning] No model found for '{category}'. Using baseline median predictor.")
        return baseline_median_predictor(recent_series, n_weeks)

    scaler_path = MODELS_DIR / f"scaler_{category}.pkl"
    scaler = load_scaler(scaler_path) if scaler_path.exists() else None

    model = safe_load_model(model_path)
    history = list(recent_series)
    preds = []

    for _ in range(n_weeks):
        x = prepare_window(history, timesteps)
        if scaler is not None:
            n_features_in = getattr(scaler, "n_features_in_", None)
            if n_features_in in (None, 1):
                flat_vals = x.reshape((timesteps, 1))
                flat_scaled = scaler.transform(flat_vals)
                x_scaled = flat_scaled.reshape((1, timesteps, 1))
            else:
                try:
                    flat_row = x.reshape((1, timesteps))
                    flat_scaled = scaler.transform(flat_row)
                    x_scaled = flat_scaled.reshape((1, timesteps, 1))
                except Exception:
                    x_scaled = x
        else:
            x_scaled = x

        pred_scaled = float(model.predict(x_scaled, verbose=0).flatten()[0])
        if scaler is not None:
            try:
                inv = scaler.inverse_transform(np.array([[pred_scaled]]))
                next_pred = float(inv.flatten()[0])
            except Exception:
                next_pred = float(pred_scaled)
        else:
            next_pred = float(pred_scaled)

        preds.append(next_pred)
        history.append(next_pred)

    return preds


# ------------------ Main Interactive Flow ------------------

def load_and_detect(csv_path="data/processed/cleaned_transactions.csv"):
    csv_path = Path(csv_path)
    df = pd.read_csv(csv_path)
    df.columns = [c.strip() for c in df.columns]

    # detect date
    possible_date_cols = [c for c in df.columns if c.lower() in ("date", "datetime", "time", "timestamp")]
    date_col = possible_date_cols[0] if possible_date_cols else df.columns[0]
    df[date_col] = pd.to_datetime(df[date_col], errors="coerce")

    # detect amount
    possible_amount_cols = [c for c in df.columns if c.lower() in ("amount", "value", "amt", "transaction_amount", "price")]
    amount_col = possible_amount_cols[0] if possible_amount_cols else df.columns[-1]

    # detect category
    possible_cat_cols = [c for c in df.columns if c.lower() in ("category", "cat", "label")]
    cat_col = possible_cat_cols[0] if possible_cat_cols else df.columns[1]

    df = df.rename(columns={date_col: "Date", amount_col: "amount", cat_col: "category"})
    df = df.set_index("Date").sort_index()
    return df


def main():
    print("🧠 Smart Budget Predictor — Interactive Mode")
    df = load_and_detect()

    # Step 1: Pick category
    cats = sorted(df["category"].dropna().astype(str).unique())
    print("\nAvailable categories:")
    for i, c in enumerate(cats, start=1):
        print(f"  {i}. {c}")

    choice = input("\nEnter category name or number: ").strip()
    if choice.isdigit():
        idx = int(choice)
        if 1 <= idx <= len(cats):
            category = cats[idx - 1]
        else:
            print("Invalid choice.")
            return
    else:
        category = choice

    # Step 2: Weeks history & prediction
    try:
        weeks_history = int(input("How many past weeks to use? (default=16): ") or "16")
        predict_weeks = int(input("How many future weeks to predict? (default=2): ") or "2")
    except ValueError:
        print("Invalid numeric input.")
        return

    # Step 3: Optional manual values
    manual_input = input("Enter recent weekly values (comma-separated, oldest→newest) or press Enter to use CSV: ").strip()
    if manual_input:
        recent_series = [float(v) for v in manual_input.split(",") if v.strip()]
    else:
        weekly = df[df["category"].astype(str).str.lower() == category.lower()]["amount"].resample("W").sum().dropna()
        if weekly.empty:
            print(f"No data found for '{category}'.")
            return
        recent_series = weekly.values[-weeks_history:]

    print(f"\nUsing {len(recent_series)} recent weekly values for '{category}'...")
    print(f"Recent series: {np.round(recent_series, 2)}")

    preds = predict_next_weeks(category, recent_series, n_weeks=predict_weeks)

    # Step 4: Save and display results
    last_week = df.index.max()
    out_rows = []
    for i, p in enumerate(preds, start=1):
        out_rows.append({
            "category": category,
            "prediction_date": (last_week + pd.Timedelta(weeks=i)).strftime("%Y-%m-%d"),
            "predicted_amount": p
        })
    out_df = pd.DataFrame(out_rows)
    out_path = OUTPUT_DIR / f"predictions_{category}.csv"
    out_df.to_csv(out_path, index=False)

    print("\n✅ Prediction complete!")
    print(out_df)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
