"""
Streamlit dashboard for the Smart Budget Tracking project.

Allows users to trigger the forecasting pipeline defined in simulation.py
and explore interactive summaries, category drilldowns, and predictive insights.
"""

from __future__ import annotations

import io
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd
import streamlit as st

from simulation import predict_future
from training import append_and_retrain


DATA_PATH = Path(__file__).resolve().parent / "data" / "processed" / "cleaned_transactions.csv"
REQUIRED_COLUMNS = {"date", "category", "amount"}


st.set_page_config(
    page_title="Smart Budget Tracking",
    page_icon="💰",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("💰 Smart Budget Tracking Dashboard")
st.caption("AI-powered budget forecasting with Moving Average, Holt-Winters, and LSTM models")


@st.cache_data(show_spinner=False)
def load_history_data() -> pd.DataFrame:
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Processed transactions not found at {DATA_PATH}. "
            "Run processing.py to build the cleaned dataset."
        )
    df = pd.read_csv(DATA_PATH, parse_dates=["date"])
    df["category"] = df["category"].astype(str).str.lower().str.strip()
    return df


def format_currency(value: float) -> str:
    return f"₱{value:,.2f}"


def compute_insights(history_df: pd.DataFrame, preds_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    if preds_df.empty:
        return pd.DataFrame(), pd.DataFrame()

    predicted_totals = preds_df.groupby("category")["ensemble"].sum()
    past_avg = (
        history_df.groupby("category")["amount"]
        .mean()
        .reindex(predicted_totals.index)
        .fillna(0.0)
    )

    insights = []
    for cat in predicted_totals.index:
        pred_val = float(predicted_totals[cat])
        avg_val = float(past_avg.get(cat, 0.0))
        pct = (pred_val / avg_val * 100) if avg_val else np.nan

        if avg_val == 0:
            trend = "⚪ New category (no past data)"
            risk = "neutral"
        elif pct > 120:
            trend = f"🔴 Overspending risk (↑ {pct - 100:.1f}%)"
            risk = "high"
        elif pct > 90:
            trend = f"🟡 Stable spending ({pct:.1f}% of past avg)"
            risk = "medium"
        else:
            trend = f"🟢 Improving (↓ {100 - pct:.1f}%)"
            risk = "low"

        insights.append(
            {
                "Category": cat.title(),
                "Past Avg": avg_val,
                "Predicted": pred_val,
                "Change vs Past": pct - 100 if not np.isnan(pct) else np.nan,
                "Risk Level": risk.title(),
                "Trend": trend,
            }
        )

    insights_df = pd.DataFrame(insights).sort_values("Predicted", ascending=False)

    top_contributors = (
        predicted_totals.sort_values(ascending=False).head(3).reset_index()
        if not predicted_totals.empty
        else pd.DataFrame(columns=["category", "ensemble"])
    )
    if not top_contributors.empty:
        top_contributors["category"] = top_contributors["category"].str.title()
        top_contributors.rename(columns={"category": "Category", "ensemble": "Predicted"}, inplace=True)

    return insights_df, top_contributors


def build_monthly_summary(preds_df: pd.DataFrame, budget: float) -> pd.DataFrame:
    if preds_df.empty:
        return pd.DataFrame()

    summary = (
        preds_df.groupby("month_ahead")[["moving_average", "holt_winters", "lstm", "ensemble", "best_pred"]]
        .sum()
        .reset_index()
        .sort_values("month_ahead")
    )
    summary["budget"] = budget
    summary["diff_vs_budget"] = summary["ensemble"] - budget
    summary["status"] = np.where(summary["diff_vs_budget"] <= 0, "Within budget", "Over budget")
    summary.rename(columns={"month_ahead": "Month Ahead"}, inplace=True)
    return summary




def render_overview(preds_df: pd.DataFrame, insights_df: pd.DataFrame, months: int, budget: float):
    total_forecast = preds_df["ensemble"].sum()
    monthly_avg = total_forecast / months if months else total_forecast

    risk_rank = {"High": 2, "Medium": 1, "Low": 0, "Neutral": 0}
    highest_risk = None
    if not insights_df.empty:
        ranking_df = insights_df.copy()
        ranking_df["_rank"] = ranking_df["Risk Level"].map(risk_rank).fillna(0)
        highest_risk = (
            ranking_df.sort_values(["_rank", "Predicted"], ascending=[False, False])
            .head(1)
            .drop(columns=["_rank"])
        )

    col1, col2, col3 = st.columns(3)
    col1.metric("Total Ensemble Forecast", format_currency(total_forecast), help="Sum of ensemble predictions for the selected horizon.")
    col2.metric("Avg Monthly Spend", format_currency(monthly_avg), help="Total forecast divided by number of months selected.")

    if highest_risk is not None and not highest_risk.empty:
        row = highest_risk.iloc[0]
        col3.metric(
            "Highest Risk Category",
            row["Category"],
            f"{row['Trend']}",
            help="Category with the highest predicted spend relative to its historical average.",
        )
    else:
        col3.metric("Highest Risk Category", "N/A", help="Run predictions to see category risks.")

    # Model-level average monthly spend & share
    model_labels = [
        ("lstm", "LSTM"),
        ("holt_winters", "Holt-Winters"),
        ("moving_average", "Moving Avg"),
    ]
    model_totals = preds_df[[key for key, _ in model_labels]].sum()
    st.markdown("**Average Monthly Spend by Model**")
    model_cols = st.columns(len(model_labels))
    for col, (key, label) in zip(model_cols, model_labels):
        total = model_totals.get(key, 0.0)
        avg = total / months if months else total
        pct = (avg / monthly_avg * 100) if monthly_avg else 0.0
        delta_text = f"{pct:.1f}% of ensemble avg" if monthly_avg else "n/a"
        col.metric(
            label,
            format_currency(avg),
            delta_text,
            help=f"Average monthly prediction from the {label} model over the selected horizon.",
        )


def render_monthly_outlook(summary_df: pd.DataFrame):
    st.subheader("Monthly Budget Outlook")
    if summary_df.empty:
        st.info("No predictions yet. Run the forecast to populate this section.")
        return

    styled = summary_df.copy()
    numeric_cols = ["moving_average", "holt_winters", "lstm", "ensemble", "best_pred", "budget", "diff_vs_budget"]
    styled[numeric_cols] = styled[numeric_cols].map(format_currency)
    st.dataframe(
        styled.rename(
            columns={
                "moving_average": "Moving Avg",
                "holt_winters": "Holt-Winters",
                "lstm": "LSTM",
                "ensemble": "Ensemble",
                "best_pred": "Best Pred",
                "budget": "Budget",
                "diff_vs_budget": "Diff vs Budget",
                "status": "Status",
            }
        ),
        use_container_width=True,
        hide_index=True,
    )


def render_category_breakdown(preds_df: pd.DataFrame):
    st.subheader("Category Breakdown")
    if preds_df.empty:
        st.info("No predictions found for any category.")
        return

    category_totals = (
        preds_df.groupby("category")["ensemble"]
        .sum()
        .sort_values(ascending=False)
        .rename("Ensemble")
        .reset_index()
    )
    category_totals["Category"] = category_totals["category"].str.title()

    st.bar_chart(
        category_totals.set_index("Category")[["Ensemble"]],
        use_container_width=True,
    )

    categories = category_totals["category"].tolist()
    pretty_labels = {cat: cat.title() for cat in categories}
    selected = st.selectbox(
        "Select a category to inspect detailed forecasts",
        options=categories,
        format_func=lambda x: pretty_labels.get(x, x.title()),
    )

    cat_df = preds_df[preds_df["category"] == selected].copy()
    cat_df.sort_values("month_ahead", inplace=True)
    cat_df["Month"] = cat_df["month_ahead"].apply(lambda m: f"Month {m}")

    st.line_chart(
        cat_df.set_index("Month")[["moving_average", "holt_winters", "lstm", "ensemble"]],
        use_container_width=True,
    )

    display_cols = ["Month", "moving_average", "holt_winters", "lstm", "ensemble", "best_model", "best_pred"]
    formatted = cat_df[display_cols].rename(
        columns={
            "moving_average": "Moving Avg",
            "holt_winters": "Holt-Winters",
            "lstm": "LSTM",
            "ensemble": "Ensemble",
            "best_model": "Model Choice",
            "best_pred": "Best Pred",
        }
    )
    for col in ["Moving Avg", "Holt-Winters", "LSTM", "Ensemble", "Best Pred"]:
        formatted[col] = formatted[col].apply(format_currency)
    st.dataframe(formatted, hide_index=True, use_container_width=True)


def render_insights(insights_df: pd.DataFrame, top_contributors: pd.DataFrame):
    st.subheader("Predictive Insights")
    if insights_df.empty:
        st.info("Insights will appear after generating predictions.")
        return

    display = insights_df.copy()
    display["Past Avg"] = display["Past Avg"].apply(format_currency)
    display["Predicted"] = display["Predicted"].apply(format_currency)
    display["Change vs Past"] = display["Change vs Past"].apply(lambda x: f"{x:+.1f}%" if pd.notna(x) else "N/A")
    st.dataframe(display, use_container_width=True, hide_index=True)

    if not top_contributors.empty:
        st.markdown("**Top 3 Predicted Spending Contributors**")
        rows = []
        for idx, row in top_contributors.iterrows():
            rows.append(f"{idx + 1}. {row['Category']} — {format_currency(row['Predicted'])}")
        st.write("\n".join(rows))


def download_predictions_button(preds_df: pd.DataFrame):
    if preds_df.empty:
        return
    csv_buf = io.StringIO()
    preds_df.to_csv(csv_buf, index=False)
    st.download_button(
        label="Download predictions (CSV)",
        data=csv_buf.getvalue(),
        file_name="smart_budget_predictions.csv",
        mime="text/csv",
    )


def main():
    sidebar_csv_file = None
    with st.sidebar:
        st.header("Forecast Settings")
        months = st.slider("Months Ahead", min_value=1, max_value=12, value=6)
        budget = st.number_input("Monthly Budget (₱)", min_value=0.0, value=25000.0, step=1000.0)
        run_forecast = st.button("Run Forecast", type="primary")
        st.caption("Tip: Ensure processing.py and training.py have been run so models and data exist.")
        st.markdown("---")
        st.subheader("➕ Manual Entry & Retrain")
        entry_date = st.date_input("Date", value=pd.Timestamp.today().date(), key="sidebar_entry_date")
        entry_category = st.text_input("Category (e.g., groceries)", key="sidebar_entry_cat")
        entry_amount = st.number_input("Amount (₱)", min_value=0.0, step=100.0, key="sidebar_entry_amt")
        manual_feedback = st.empty()
        manual_submit = st.button("Add Transaction & Retrain", key="sidebar_manual_submit")

        st.markdown("---")
        st.subheader("📁 Append CSV & Retrain")
        with st.form("sidebar_csv_form"):
            sidebar_csv_file = st.file_uploader("Select CSV file", type=["csv"], key="sidebar_csv_upload")
            csv_submit = st.form_submit_button("Append CSV & Retrain")
        csv_feedback = st.empty()

    if "results" not in st.session_state:
        st.session_state["results"] = None

    notice = st.session_state.pop("data_update_notice", None)
    if notice:
        st.success(notice)

    data_updated = False
    if manual_submit:
        if not DATA_PATH.exists():
            manual_feedback.info("Processed data not found yet. Run processing.py before adding entries.")
        else:
            category = entry_category.strip().lower()
            if not category:
                manual_feedback.warning("Category cannot be empty.")
            else:
                manual_df = pd.DataFrame(
                    [
                        {
                            "date": pd.to_datetime(entry_date),
                            "category": category,
                            "amount": float(entry_amount),
                        }
                    ]
                )
                try:
                    append_and_retrain(new_data=manual_df, retrain_all=False, freq="ME")
                    manual_feedback.success("Manual transaction added and models retrained.")
                    data_updated = True
                except Exception as exc:
                    manual_feedback.error(f"Unable to append manual entry: {exc}")

    if 'csv_submit' in locals() and csv_submit:
        if not DATA_PATH.exists():
            csv_feedback.info("Processed data not found yet. Run processing.py before appending CSV data.")
        elif sidebar_csv_file is None:
            csv_feedback.warning("Please choose a CSV file before submitting.")
        else:
            try:
                new_df = pd.read_csv(sidebar_csv_file)
                new_df = new_df.rename(columns={col: col.strip().lower() for col in new_df.columns})
            except Exception as exc:
                csv_feedback.error(f"Failed to read CSV: {exc}")
            else:
                missing_cols = REQUIRED_COLUMNS - set(new_df.columns)
                if missing_cols:
                    csv_feedback.error(f"CSV must include columns: {', '.join(sorted(REQUIRED_COLUMNS))}")
                else:
                    try:
                        append_and_retrain(new_data=new_df, retrain_all=False, freq="ME")
                        csv_feedback.success("CSV data appended and models retrained.")
                        data_updated = True
                    except Exception as exc:
                        csv_feedback.error(f"Unable to append CSV data: {exc}")
    if data_updated:
        load_history_data.clear()
        st.session_state["results"] = None
        st.session_state["data_update_notice"] = (
            "New data added and models retrained. Run a forecast to see the updated predictions."
        )
        st.experimental_rerun()

    if run_forecast:
        try:
            history_df = load_history_data()
        except FileNotFoundError as exc:
            st.error(str(exc))
            st.stop()

        with st.spinner("Running multi-model forecast..."):
            preds_df = predict_future(history_df.copy(), months, budget)
            if preds_df is None:
                preds_df = pd.DataFrame()
            insights_df, top_contributors = compute_insights(history_df, preds_df)
            summary_df = build_monthly_summary(preds_df, budget)

        st.session_state["results"] = {
            "months": months,
            "budget": budget,
            "predictions": preds_df,
            "insights": insights_df,
            "top_contributors": top_contributors,
            "monthly_summary": summary_df,
        }

    results = st.session_state.get("results")
    if not results:
        st.info("Configure the forecast in the sidebar and click **Run Forecast** to get started.")
        return

    preds_df = results["predictions"]
    if preds_df is None or preds_df.empty:
        st.warning("No predictions were generated. Check that your processed data and trained models exist.")
        return

    render_overview(preds_df, results["insights"], results["months"], results["budget"])
    render_monthly_outlook(results["monthly_summary"])
    render_category_breakdown(preds_df)
    render_insights(results["insights"], results["top_contributors"])
    download_predictions_button(preds_df)


if __name__ == "__main__":
    main()

