"""
simulation_v3.py — Smart Budget Predictor (Enhanced Version)

Uses processed data from data/processed/cleaned_transactions.csv
and predicts future expenses using Moving Average, Holt-Winters, and LSTM models.

Features:
 - Smart model selection based on best validation RMSE
 - Ensemble forecast (weighted average)
 - Interactive CLI for prediction or monthly-entry data addition
 - Auto retraining when user adds monthly entries or provides a CSV
 - Budget check summary with LSTM / Moving Average / Holt-Winters breakdown
 - Predictive Insights (risk classification and top contributors)
 - Unified output file (outputs/predictions_summary.csv)
"""

import os
import sys
import json
import pickle
import warnings
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# import training functions (training.py must expose append_and_retrain and train_all)
from training import train_all, append_and_retrain  # append_and_retrain should be present in training.py
from tensorflow.keras.models import load_model

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "data" / "processed" / "cleaned_transactions.csv"
MODELS_DIR = ROOT / "models"
OUTPUTS_DIR = ROOT / "outputs"
OUTPUTS_DIR.mkdir(exist_ok=True)


# ---------- Helper Functions ----------

def safe_name(cat):
    return cat.lower().replace(" ", "_")


def load_holt_winters(cat):
    p = MODELS_DIR / f"holtwinters_{safe_name(cat)}.pkl"
    if not p.exists():
        return None
    with open(p, "rb") as f:
        return pickle.load(f)


def load_moving_average(cat):
    p = MODELS_DIR / f"moving_average_{safe_name(cat)}.json"
    if not p.exists():
        return None
    with open(p, "r") as f:
        return json.load(f)


def load_lstm_and_scaler(cat):
    m = MODELS_DIR / f"lstm_{safe_name(cat)}.h5"
    s = MODELS_DIR / f"scaler_{safe_name(cat)}.pkl"
    if not m.exists() or not s.exists():
        return None, None
    model = load_model(str(m), compile=False)
    with open(s, "rb") as f:
        scaler = pickle.load(f)
    return model, scaler


def load_metrics(cat):
    p = MODELS_DIR / f"metrics_{safe_name(cat)}.json"
    if not p.exists():
        return {}
    with open(p, "r") as f:
        return json.load(f)


MODEL_DISPLAY_NAMES = {
    "lstm": "LSTM",
    "holt_winters": "Holt-Winters",
    "moving_average": "Moving Average",
}


def best_model_for_category(cat):
    """Select the model with the lowest recorded RMSE (fallbacks provided)."""
    metrics = load_metrics(cat)
    candidates = {}
    for key in ("lstm", "holt_winters", "moving_average"):
        rmse = (
            metrics.get(key, {}).get("rmse")
            if isinstance(metrics.get(key), dict)
            else None
        )
        if rmse is not None:
            candidates[key] = rmse

    if candidates:
        return min(candidates, key=candidates.get)

    # default fallback when no metrics exist yet
    return "lstm"


# ---------- Forecast Functions ----------

def forecast_moving_average(series, months, params):
    vals = series.values
    window = int(params.get("window", 3)) if params else 3
    last_vals = vals[-window:] if len(vals) >= window else vals
    avg = float(np.mean(last_vals)) if len(last_vals) else 0.0
    return [avg] * months


def forecast_holt_winters(model, months):
    try:
        preds = model.forecast(months)
        return [float(x) for x in preds]
    except Exception:
        return [0.0] * months


def forecast_lstm(series, months, model, scaler, seq_len=12):
    vals = series.values.astype(float).copy()
    if len(vals) == 0:
        return [0.0] * months
    # pad if not enough history
    if len(vals) < seq_len:
        pad_val = vals[-1] if len(vals) else 0.0
        vals = np.pad(vals, (seq_len - len(vals), 0), constant_values=pad_val)
    preds = []
    for _ in range(months):
        x = vals[-seq_len:].reshape(-1, 1)
        x_scaled = scaler.transform(x)
        yhat_scaled = model.predict(x_scaled.reshape(1, seq_len, 1), verbose=0)
        yhat = scaler.inverse_transform(yhat_scaled.reshape(-1, 1))[0][0]
        preds.append(float(yhat))
        vals = np.append(vals, yhat)
    return preds


def collect_model_metrics(categories):
    """Build a long-form DataFrame with MAE/RMSE per model & category."""
    rows = []
    for cat in categories:
        metrics = load_metrics(cat)
        if not metrics:
            continue
        for model_key in ("moving_average", "holt_winters", "lstm"):
            model_metrics = metrics.get(model_key)
            if not isinstance(model_metrics, dict):
                continue
            mae = model_metrics.get("mae")
            rmse = model_metrics.get("rmse")
            if mae is None and rmse is None:
                continue
            rows.append(
                {
                    "category": cat,
                    "model": model_key,
                    "mae": mae,
                    "rmse": rmse,
                }
            )

    return pd.DataFrame(rows)


def plot_model_metrics(metrics_df, show=False):
    """Create and optionally display a dual-panel bar chart for MAE/RMSE."""
    if metrics_df.empty:
        return None

    summary = (
        metrics_df.groupby("model")[["mae", "rmse"]]
        .mean()
        .sort_values("rmse")
    )

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for ax, metric in zip(axes, ["mae", "rmse"]):
        summary[metric].plot(
            kind="barh",
            ax=ax,
            color="#3f51b5" if metric == "rmse" else "#009688",
            legend=False,
        )
        ax.set_xlabel(f"Average {metric.upper()}")
        ax.set_ylabel("")
        ax.set_title(f"Average {metric.upper()} by Model")
        ax.grid(axis="x", linestyle="--", alpha=0.3)
        ax.set_yticklabels(
            [MODEL_DISPLAY_NAMES.get(lbl.get_text(), lbl.get_text()) for lbl in ax.get_yticklabels()]
        )

    fig.suptitle("Model Validation Metrics Overview", fontsize=14, fontweight="bold")
    fig.tight_layout()
    out_path = OUTPUTS_DIR / "model_metrics_summary.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")

    if show:
        plt.show()
    else:
        plt.close(fig)

    print(f"[Metrics] Saved visualization to {out_path}")
    return out_path


# ---------- Data Utilities ----------

def load_cleaned_data():
    if not DATA_PATH.exists():
        print(f"Missing processed data: {DATA_PATH}. Run preprocessing.py first.")
        sys.exit(1)
    df = pd.read_csv(DATA_PATH, parse_dates=["date"])
    df["category"] = df["category"].astype(str).str.lower().str.strip()
    return df


def monthly_series(df, cat):
    s = df[df["category"] == cat].set_index("date")["amount"].resample("ME").sum().sort_index()
    return s.fillna(0.0)


# ---------- Core Simulation Logic ----------

def predict_future(df, months, budget):
    categories = sorted(df["category"].unique())
    results = []
    for cat in categories:
        print(f"\n🔹 Predicting for category: {cat}")
        series = monthly_series(df, cat)
        if series.empty:
            print("  (no data for this category; skipping)")
            continue

        # Load models
        ma_params = load_moving_average(cat)
        hw_model = load_holt_winters(cat)
        lstm_model, scaler = load_lstm_and_scaler(cat)

        # Compute predictions
        ma_pred = forecast_moving_average(series, months, ma_params) if ma_params else [0.0] * months
        hw_pred = forecast_holt_winters(hw_model, months) if hw_model else [0.0] * months
        lstm_pred = forecast_lstm(series, months, lstm_model, scaler) if lstm_model and scaler else ma_pred

        # Ensemble (weighted)
        ensemble_pred = (0.5 * np.array(lstm_pred)) + (0.3 * np.array(hw_pred)) + (0.2 * np.array(ma_pred))
        ensemble_pred = [float(x) for x in ensemble_pred]

        # Choose best model dynamically (per-category)
        best_model = best_model_for_category(cat)
        best_pred = {
            "lstm": lstm_pred,
            "holt_winters": hw_pred,
            "moving_average": ma_pred,
            "ensemble": ensemble_pred
        }.get(best_model, ensemble_pred)

        # Save per month
        for i in range(months):
            results.append({
                "category": cat,
                "month_ahead": i + 1,
                "moving_average": float(ma_pred[i]),
                "holt_winters": float(hw_pred[i]),
                "lstm": float(lstm_pred[i]),
                "ensemble": float(ensemble_pred[i]),
                "best_model": best_model,
                "best_pred": float(best_pred[i])
            })

    df_pred = pd.DataFrame(results)
    out_path = OUTPUTS_DIR / "predictions_summary.csv"
    df_pred.to_csv(out_path, index=False)
    print(f"\n✅ Predictions saved to {out_path}\n")

    # -----------------------
    # Budget-check formatting
    # -----------------------
    if df_pred.empty:
        print("No predictions generated.")
    else:
        n_months = int(df_pred['month_ahead'].max())

        for m in range(1, n_months + 1):
            print("1st Month" if m == 1 else f"Month +{m}:")
            month_df = df_pred[df_pred["month_ahead"] == m]

            total_lstm = month_df["lstm"].sum()
            total_ma = month_df["moving_average"].sum()
            total_hw = month_df["holt_winters"].sum()
            total_ensemble = month_df["ensemble"].sum()
            total_best = month_df["best_pred"].sum()

            def status_and_diff(value, budget):
                diff = value - budget
                status = "🟢 within budget" if diff <= 0 else "🔴 over budget"
                sign = f"+{diff:,.2f}" if diff > 0 else f"{diff:,.2f}"
                return status, sign, diff

            # Ensemble section shows each model totals and the ensemble total
            print("📊 Budget Check (ensemble):")
            status, sign, _ = status_and_diff(total_lstm, budget)
            print(f"  LSTM           : ₱{total_lstm:,.2f} → {status} (diff {sign})")
            status, sign, _ = status_and_diff(total_ma, budget)
            print(f"  Moving Average : ₱{total_ma:,.2f} → {status} (diff {sign})")
            status, sign, _ = status_and_diff(total_hw, budget)
            print(f"  Holt-Winters   : ₱{total_hw:,.2f} → {status} (diff {sign})")
            status_e, sign_e, _ = status_and_diff(total_ensemble, budget)
            print(f"  Ensemble total : ₱{total_ensemble:,.2f} → {status_e} (diff {sign_e})\n")

            # Best-pred section shows same breakdown and the aggregated best_pred total
            print("📊 Budget Check (best_pred):")
            status, sign, _ = status_and_diff(total_lstm, budget)
            print(f"  LSTM           : ₱{total_lstm:,.2f} → {status} (diff {sign})")
            status, sign, _ = status_and_diff(total_ma, budget)
            print(f"  Moving Average : ₱{total_ma:,.2f} → {status} (diff {sign})")
            status, sign, _ = status_and_diff(total_hw, budget)
            print(f"  Holt-Winters   : ₱{total_hw:,.2f} → {status} (diff {sign})")
            status_b, sign_b, _ = status_and_diff(total_best, budget)
            print(f"  Best Pred total: ₱{total_best:,.2f} → {status_b} (diff {sign_b})")
            print("")  # spacer between months

    # -----------------------
    # Predictive Insights Section
    # -----------------------
    # Uses processed df (passed into predict_future) to compute past averages
    print("\n🔮 Predictive Insights — Smart Budget Analysis\n")

    # Compute total predicted spend per category (sum of ensemble predictions)
    predicted_totals = df_pred.groupby("category")["ensemble"].sum() if not df_pred.empty else pd.Series(dtype=float)

    # Compute average historical monthly spending
    past_avg = (
        df.groupby("category")["amount"]
        .mean()
        .reindex(predicted_totals.index)
        .fillna(0.0)
    )

    # Compute percentage change from past average to predicted
    # Avoid division by zero: replace zeros with NaN for ratio calculation
    past_avg_nonzero = past_avg.replace(0, np.nan)
    change_pct = (predicted_totals / past_avg_nonzero) * 100

    insights = []
    for cat in predicted_totals.index if not predicted_totals.empty else []:
        pred_val = float(predicted_totals[cat])
        avg_val = float(past_avg[cat])
        pct = float(change_pct[cat]) if not np.isnan(change_pct[cat]) else 0.0
        if avg_val == 0:
            trend = "⚪ new category (no past data)"
            risk = "neutral"
        elif pct > 120:
            trend = "🔴 Overspending risk (↑ {:.1f}%)".format(pct - 100)
            risk = "high"
        elif pct > 90:
            trend = "🟡 Stable spending ({:.1f}%)".format(pct)
            risk = "medium"
        else:
            trend = "🟢 Improving (↓ {:.1f}%)".format(100 - pct)
            risk = "low"
        insights.append((cat, avg_val, pred_val, trend, risk))

    if insights:
        df_insights = pd.DataFrame(insights, columns=["Category", "Past Avg", "Predicted", "Trend", "Risk Level"])

        # Display insights nicely
        for _, row in df_insights.iterrows():
            print(f"📂 {row['Category'].capitalize():15} | Past ₱{row['Past Avg']:,.2f} → Pred ₱{row['Predicted']:,.2f} | {row['Trend']}")

        # Highlight top categories contributing to total predicted spend
        top_cats = predicted_totals.sort_values(ascending=False).head(3)
        print("\n🏆 Top 3 Spending Contributors (Predicted):")
        for i, (cat, val) in enumerate(top_cats.items(), 1):
            print(f"  {i}. {cat.capitalize()} — ₱{val:,.2f}")

        # Save insights CSV for later reference
        insights_path = OUTPUTS_DIR / "predictive_insights.csv"
        df_insights.to_csv(insights_path, index=False)
        print(f"\n✅ Predictive insights saved to {insights_path}\n")
    else:
        print("No predictive insights (no predicted totals).")

    # -----------------------
    # Model Metrics Snapshot & Visualization
    # -----------------------
    metrics_df = collect_model_metrics(categories)
    if not metrics_df.empty:
        metrics_csv = OUTPUTS_DIR / "model_metrics_snapshot.csv"
        metrics_df.to_csv(metrics_csv, index=False)
        print(f"[Metrics] Saved raw metrics snapshot to {metrics_csv}")
        plot_model_metrics(metrics_df, show=False)
    else:
        print("[Metrics] No stored evaluation metrics found. Run training.py to generate them.")

    return df_pred


# ---------- CLI Interaction ----------

def main():
    print("=== 💡 Smart Budget Predictor — Simulation v3 ===")
    df = load_cleaned_data()

    choice = input("Type 'predict' to forecast, 'add' to add new data, or 'both': ").strip().lower()

    if choice in ("add", "both"):
        new_csv = input("Enter path to new data CSV (leave empty to add monthly transactions manually): ").strip()
        if new_csv:
            new_path = Path(new_csv)
            if not new_path.exists():
                print(f"File not found: {new_path}. Aborting add step.")
            else:
                try:
                    new_df = pd.read_csv(new_path)
                except Exception as e:
                    print(f"Failed to read CSV: {e}")
                    new_df = None
                if new_df is not None:
                    append_and_retrain(new_data=new_df, retrain_all=False, freq="ME")
                    df = load_cleaned_data()
                    print("✅ Data appended from CSV and models retrained.")
        else:
            # Interactive monthly-entry → append into processed file
            append_and_retrain(new_data=None, retrain_all=False, freq="ME")
            df = load_cleaned_data()
            print("✅ Manual monthly transactions appended and models retrained.")


    if choice in ("predict", "both"):
        # ensure df is up-to-date
        months_input = input("How many months ahead to predict? (e.g., 3): ").strip()
        try:
            months = int(months_input) if months_input else 3
        except:
            months = 3
        budget_input = input("Enter your monthly budget (e.g., 25000): ").strip()
        try:
            budget = float(budget_input) if budget_input else 25000.0
        except:
            budget = 25000.0

        df_pred = predict_future(df, months, budget)

        # Optional visualization
        view = input("View chart? (y/n): ").strip().lower()
        if view == "y":
            if not df_pred.empty:
                summary = df_pred.groupby("category")["ensemble"].sum().sort_values()
                summary.plot(kind="barh", title="Predicted Total (Next Months)", figsize=(8, 5))
                plt.xlabel("Predicted Spending (₱)")
                plt.tight_layout()
                plt.show()
            else:
                print("No predictions to plot.")

        metrics_plot = OUTPUTS_DIR / "model_metrics_summary.png"
        if metrics_plot.exists():
            metrics_view = input("View model metrics chart? (y/n): ").strip().lower()
            if metrics_view == "y":
                img = plt.imread(str(metrics_plot))
                plt.figure(figsize=(8, 5))
                plt.imshow(img)
                plt.axis("off")
                plt.title("Model Metrics Overview")
                plt.show()

if __name__ == "__main__":
    main()
