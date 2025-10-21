# processing.py (robust date parsing & friendly errors)
import os
import pandas as pd
import numpy as np
from dateutil import parser as dparser
from config import RAW_PATH, PROCESSED_PATH, INR_TO_PHP

def find_csv(raw_path):
    if os.path.exists(raw_path):
        return raw_path
    folder = os.path.dirname(raw_path)
    if not os.path.isdir(folder):
        os.makedirs(folder, exist_ok=True)
    # fallback: first CSV in folder
    csvs = [f for f in os.listdir(folder) if f.lower().endswith('.csv')]
    if csvs:
        chosen = os.path.join(folder, csvs[0])
        print(f"Using fallback CSV: {chosen}")
        return chosen
    raise FileNotFoundError(f"No CSV found at {raw_path} or in folder {folder}.")

def robust_parse_dates(series):
    """
    Try pandas to_datetime with dayfirst=True; for any NaT, attempt dateutil.parser.parse.
    Returns datetime64[ns] series.
    """
    # first try fast vectorized parse with dayfirst=True
    parsed = pd.to_datetime(series, errors='coerce', dayfirst=True)
    # find indices that failed
    mask_failed = parsed.isna()
    if mask_failed.any():
        # try dateutil on failed rows
        def try_date(s):
            if pd.isna(s):
                return pd.NaT
            try:
                return dparser.parse(str(s), dayfirst=True)
            except Exception:
                return pd.NaT
        parsed_filled = series[mask_failed].apply(try_date)
        parsed.loc[mask_failed] = parsed_filled
    return parsed

def map_category(x):
    x = str(x).lower()
    if any(k in x for k in ['grocery','groceries','food','supermarket','vegetable','dairy','market']):
        return 'groceries'
    if any(k in x for k in ['utility','utilities','electric','water','gas','bill']):
        return 'utilities'
    if any(k in x for k in ['transport','taxi','uber','grab','bus','train','fuel','petrol','transportation']):
        return 'transportation'
    return 'others'

def preprocess_data():
    print("🔄 Loading dataset...")

    csv_path = find_csv(RAW_PATH)

    # Read without parse_dates to avoid early failures
    df = pd.read_csv(csv_path, low_memory=False)
    print(f"Loaded CSV with shape: {df.shape}")

    # Drop unneeded columns if present
    for col in ['subcategory', 'note']:
        if col in df.columns:
            df = df.drop(columns=col)
            print(f"Dropped column: {col}")

    # Identify date column
    date_col = next((c for c in df.columns if c.lower() == 'date'), None)
    if date_col is None:
        # fallback: any column containing 'date' in its name
        date_col = next((c for c in df.columns if 'date' in c.lower()), None)
    if date_col is None:
        raise ValueError("Could not find a date column. Make sure the CSV has a 'Date' column.")

    # Identify amount column
    amt_col = next((c for c in df.columns if c.lower() in ['amount', 'price', 'total']), None)
    if amt_col is None:
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        if numeric_cols:
            amt_col = numeric_cols[0]
            print(f"No explicit Amount column found; using numeric column: {amt_col}")
        else:
            raise ValueError("Could not find an Amount column. Please ensure there's a numeric column for amounts.")

    # Parse dates robustly
    print(f"Parsing dates from column: {date_col} (using dayfirst=True, with fallbacks)...")
    parsed_dates = robust_parse_dates(df[date_col])
    if parsed_dates.isna().all():
        raise ValueError("Date parsing failed for all rows. Inspect the Date column format.")
    df['date_parsed'] = parsed_dates
    # Drop rows with unparseable dates
    n_before = len(df)
    df = df[~df['date_parsed'].isna()].copy()
    n_after = len(df)
    if n_after < n_before:
        print(f"Dropped {n_before - n_after} rows with unparseable dates.")

    # Normalize amounts
    df[amt_col] = pd.to_numeric(df[amt_col], errors='coerce').fillna(0.0)
    df['amount_php'] = df[amt_col] * INR_TO_PHP

    # Category detection
    cat_col = next((c for c in df.columns if 'cat' in c.lower() or 'type' in c.lower()), None)
    if cat_col is None:
        df['category'] = 'others'
        print("No category-like column found; defaulting to 'others'.")
    else:
        df['category'] = df[cat_col].astype(str)

    # Map to simplified groups
    df['category_group'] = df['category'].apply(map_category)

    # Prepare final cleaned DF
    cleaned = df[['date_parsed', 'category_group', 'amount_php']].copy()
    cleaned = cleaned.rename(columns={'date_parsed': 'date', 'category_group': 'category'})
    cleaned['date'] = pd.to_datetime(cleaned['date']).dt.date

    os.makedirs(os.path.dirname(PROCESSED_PATH), exist_ok=True)
    cleaned.to_csv(PROCESSED_PATH, index=False)
    print(f"✅ Cleaned data saved to: {PROCESSED_PATH}")
    print(f"Cleaned shape: {cleaned.shape}")
    # show top categories counts
    print("Category distribution (top):")
    print(cleaned['category'].value_counts().head())

if __name__ == "__main__":
    preprocess_data()
