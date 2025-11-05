"""
preprocessing.py

Preprocessing utilities for Smart-Budget-Tracking.

Primary goals:
 - Load raw daily transactions CSV (e.g., data/daily_transactions.csv)
 - Normalize columns, parse dates (dayfirst=True)
 - Filter expenses (optional)
 - Map fine-grained categories into broader groups via a JSON mapping
 - Clip outliers per-category to reduce scaling/skew problems
 - Produce:
     - cleaned transactions file: data/processed/cleaned_transactions.csv
     - aggregated monthly totals: data/processed/aggregated_monthly.csv
     - aggregated weekly totals: data/processed/aggregated_weekly.csv

Usage (CLI):
    python preprocessing.py                # runs default pipeline and writes processed files
    from preprocessing import ...          # import the functions into other modules
"""

from pathlib import Path
import json
from typing import Dict, Optional, Tuple

import pandas as pd
import numpy as np

ROOT = Path(__file__).resolve().parent
RAW_DEFAULT = ROOT / "data" / "raw" / "daily_transactions.csv"
PROCESSED_DIR = ROOT / "data" / "processed"
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
CLEANED_CSV_OUT = PROCESSED_DIR / "cleaned_transactions.csv"
AGG_MONTHLY_OUT = PROCESSED_DIR / "aggregated_monthly.csv"
AGG_WEEKLY_OUT = PROCESSED_DIR / "aggregated_weekly.csv"
CATEGORY_MAP_PATH = ROOT / "config" / "category_map.json"
CATEGORY_MAP_PATH.parent.mkdir(parents=True, exist_ok=True)


# ---------- helpers ----------
def detect_columns(df: pd.DataFrame) -> Tuple[str, str, str, Optional[str]]:
    """
    Try to find (date_col, category_col, amount_col, income_expense_col)
    Raises KeyError if required fields not found.
    """
    date_col = None
    for c in df.columns:
        if c.lower() == "date" or "date" in c.lower() or "time" in c.lower():
            date_col = c
            break
    if date_col is None:
        raise KeyError("No date-like column found in raw CSV. Columns: {}".format(list(df.columns)))

    category_col = None
    for c in df.columns:
        if c.lower() == "category" or "cat" in c.lower():
            category_col = c
            break
    if category_col is None:
        raise KeyError("No category-like column found in raw CSV. Columns: {}".format(list(df.columns)))

    amount_col = None
    for c in df.columns:
        if "amount" in c.lower() or "amt" in c.lower() or "value" in c.lower() or "php" in c.lower():
            amount_col = c
            break
    if amount_col is None:
        raise KeyError("No amount-like column found in raw CSV. Columns: {}".format(list(df.columns)))

    income_expense_col = None
    for c in df.columns:
        if "income" in c.lower() or "expense" in c.lower() or "type" in c.lower():
            income_expense_col = c
            break

    return date_col, category_col, amount_col, income_expense_col


def load_raw_csv(path: Path = RAW_DEFAULT, dayfirst: bool = True) -> pd.DataFrame:
    """
    Load the raw CSV and parse dates. Returns a DataFrame with original columns.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Raw CSV not found: {path}")
    df = pd.read_csv(path)
    date_col, _, _, _ = detect_columns(df)
    df[date_col] = pd.to_datetime(df[date_col], dayfirst=dayfirst, errors="coerce")
    # warn about unparsable dates
    n_bad = df[date_col].isna().sum()
    if n_bad:
        print(f"Warning: {n_bad} rows have unparsable dates (set to NaT).")
    return df


# ---------- category mapping ----------
DEFAULT_CATEGORY_MAP = {
    "groceries": ["food", "snacks", "groceries", "grocery", "supermarket"],
    "transportation": ["train", "bus", "taxi", "grab", "transportation", "fuel"],
    "utilities": ["electricity", "water", "internet", "mobile", "utilities", "telco"],
    "entertainment": ["subscription", "netflix", "music", "festivals", "leisure", "entertainment"],
    "others": []  # fallback
}


def load_or_create_category_map(path: Path = CATEGORY_MAP_PATH) -> Dict[str, list]:
    """
    Load mapping if exists, else write default mapping and return it.
    Mapping format: { "group_name": ["possible_keyword1", "keyword2", ...], ... }
    Matching is case-insensitive substring match.
    """
    if path.exists():
        with open(path, "r", encoding="utf-8") as fh:
            mapping = json.load(fh)
        print(f"Loaded category mapping from {path}")
        return mapping
    else:
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(DEFAULT_CATEGORY_MAP, fh, indent=2)
        print(f"No mapping found. Wrote default mapping to {path}")
        return DEFAULT_CATEGORY_MAP


def map_category(raw_cat: str, mapping: Dict[str, list]) -> str:
    """
    Map a raw category string into one of the mapping keys via substring matching.
    Falls back to 'others'.
    """
    if not isinstance(raw_cat, str) or not raw_cat.strip():
        return "others"
    rc = raw_cat.strip().lower()
    for target, keywords in mapping.items():
        # exact and substring matches
        for kw in keywords:
            if kw.lower() in rc:
                return target
        if rc == target:
            return target
    return "others"


# ---------- cleaning & aggregation ----------
def clip_outliers_series(s: pd.Series, q_low: float = 0.01, q_high: float = 0.99) -> pd.Series:
    """
    Clip series values to given quantile range to reduce impact of extreme one-off transactions.
    """
    if s.empty:
        return s
    low, high = s.quantile([q_low, q_high])
    return s.clip(lower=low, upper=high)


def preprocess_transactions(
    raw_df: pd.DataFrame,
    mapping: Optional[Dict[str, list]] = None,
    keep_income: bool = False,
    clip_q: Tuple[float, float] = (0.01, 0.99),
    save_cleaned: bool = True,
) -> pd.DataFrame:
    """
    Main preprocessing:
     - detect cols, normalize names to: date, category, amount
     - filter income/expense if desired
     - map categories using mapping
     - clip outliers per-category (on the raw transaction amounts aggregated per category-month)
     - save cleaned transactions CSV for downstream training

    Returns cleaned per-transaction DataFrame (date, category, amount).
    """
    date_col, category_col, amount_col, income_col = detect_columns(raw_df)
    df = raw_df.copy()
    df[date_col] = pd.to_datetime(df[date_col], dayfirst=True, errors="coerce")
    df = df.rename(columns={date_col: "date", category_col: "raw_category", amount_col: "amount"})

    # filter by income/expense column if requested and available
    if not keep_income and income_col:
        df = df[df[income_col].astype(str).str.lower().str.contains("expense")]
    # normalize category text
    df["raw_category"] = df["raw_category"].astype(str).str.strip()
    if mapping is None:
        mapping = load_or_create_category_map()
    df["category"] = df["raw_category"].apply(lambda x: map_category(x, mapping))

    # ensure numeric amount
    df["amount"] = pd.to_numeric(df["amount"], errors="coerce").fillna(0.0)

    # Optional: clip outliers by looking at aggregated monthly totals per category first
    # compute monthly totals
    monthly = df.set_index("date").groupby("category")["amount"].resample("M").sum().reset_index().rename(columns={"amount": "month_amount"})
    # clip month_amount per category
    clipped_months = []
    q_low, q_high = clip_q
    for cat in monthly["category"].unique():
        cat_month = monthly[monthly["category"] == cat]
        clipped = clip_outliers_series(cat_month["month_amount"], q_low=q_low, q_high=q_high)
        # store clipped ratios (we will scale original transactions within months proportionally)
        # build a mapping month_index -> clipped_value / original_value
        ratios = (clipped / cat_month["month_amount"]).replace([np.inf, -np.inf], 1.0).fillna(1.0).values
        # add to list aligned with cat_month index
        clipped_months.append(pd.DataFrame({
            "category": cat,
            "date": cat_month["date"].values,
            "month_amount_original": cat_month["month_amount"].values,
            "month_amount_clipped": clipped.values,
            "ratio": ratios
        }))
    if clipped_months:
        clipped_df = pd.concat(clipped_months, ignore_index=True)
        # merge back ratios to transactions (month-granularity)
        df["_month"] = df["date"].dt.to_period("M").dt.to_timestamp("M")
        clipped_df["_month"] = pd.to_datetime(clipped_df["date"])
        clipped_df = clipped_df[["_month", "category", "ratio"]]
        df = df.merge(clipped_df, left_on=["_month", "category"], right_on=["_month", "category"], how="left")
        df["ratio"] = df["ratio"].fillna(1.0)
        # scale each transaction amount by ratio so that extreme months are toned down
        df["amount_clipped"] = df["amount"] * df["ratio"]
        # choose clipped as final amount for modeling
        df["amount_final"] = df["amount_clipped"]
        df = df.drop(columns=["amount_clipped", "_month"])
    else:
        df["amount_final"] = df["amount"]

    # final select and reorder
    clean = df[["date", "category", "amount_final"]].rename(columns={"amount_final": "amount"})
    # sort
    clean = clean.sort_values("date").reset_index(drop=True)

    if save_cleaned:
        CLEANED_CSV_OUT.parent.mkdir(parents=True, exist_ok=True)
        clean.to_csv(CLEANED_CSV_OUT, index=False)
        print(f"Wrote cleaned transactions to: {CLEANED_CSV_OUT}")

    return clean


def aggregate_monthly(cleaned_df: pd.DataFrame, save_to: Optional[Path] = AGG_MONTHLY_OUT) -> pd.DataFrame:
    """
    Aggregate cleaned per-transaction data into monthly totals per category.
    Returns DataFrame with columns: date (month-end), category, amount
    """
    s = cleaned_df.set_index("date").groupby("category").amount.resample("ME").sum().reset_index()
    s = s.rename(columns={"amount": "amount_monthly", "date": "month_end"})
    if save_to:
        s.to_csv(save_to, index=False)
        print(f"Wrote monthly aggregation to: {save_to}")
    return s


def aggregate_weekly(cleaned_df: pd.DataFrame, save_to: Optional[Path] = AGG_WEEKLY_OUT) -> pd.DataFrame:
    """
    Aggregate cleaned per-transaction data into weekly totals (W-MON) per category.
    Returns DataFrame with columns: date (week end Mon?), category, amount
    """
    s = cleaned_df.set_index("date").groupby("category").amount.resample("W-MON").sum().reset_index()
    s = s.rename(columns={"amount": "amount_weekly", "date": "week_end"})
    if save_to:
        s.to_csv(save_to, index=False)
        print(f"Wrote weekly aggregation to: {save_to}")
    return s


# ---------- convenience CLI ----------
def run_pipeline(
    raw_path: Path = RAW_DEFAULT,
    keep_income: bool = False,
    clip_q: Tuple[float, float] = (0.01, 0.99),
    mapping: Optional[Dict[str, list]] = None
):
    raw = load_raw_csv(raw_path)
    if mapping is None:
        mapping = load_or_create_category_map()
    clean = preprocess_transactions(raw, mapping=mapping, keep_income=keep_income, clip_q=clip_q, save_cleaned=True)
    agg_m = aggregate_monthly(clean)
    agg_w = aggregate_weekly(clean)
    # Print quick summary
    print("\nSummary:")
    print("Total cleaned transactions:", len(clean))
    print("Categories:", sorted(clean["category"].unique().tolist()))
    print("Monthly aggregation sample:")
    print(agg_m.groupby("category")["amount_monthly"].agg(["count", "mean", "sum"]).sort_values("sum", ascending=False).head())
    return {"clean": clean, "monthly": agg_m, "weekly": agg_w}


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Preprocessing pipeline for Smart-Budget-Tracking")
    parser.add_argument("--raw", type=str, default=str(RAW_DEFAULT), help="Path to raw daily transactions CSV")
    parser.add_argument("--keep-income", action="store_true", help="Keep income rows as well as expenses")
    parser.add_argument("--no-clip", action="store_true", help="Disable outlier clipping")
    args = parser.parse_args()

    mapping = load_or_create_category_map()
    clip_q = (0.01, 0.99) if not args.no_clip else (0.0, 1.0)
    run_pipeline(raw_path=Path(args.raw), keep_income=args.keep_income, clip_q=clip_q, mapping=mapping)
