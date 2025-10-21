# processing.py
import pandas as pd
import numpy as np
import os
from config import RAW_PATH, PROCESSED_PATH, INR_TO_PHP

def preprocess_data():
    print("🔄 Loading dataset...")
    df = pd.read_csv(RAW_PATH, parse_dates=['Date'])

    # Drop unnecessary columns
    for col in ['subcategory', 'note']:
        if col in df.columns:
            df = df.drop(columns=col)

    # Detect and convert amount column
    amt_col = next((c for c in df.columns if c.lower() in ['amount', 'price', 'total']), None)
    df[amt_col] = pd.to_numeric(df[amt_col], errors='coerce').fillna(0)

    # Normalize currency
    df['amount_php'] = df[amt_col] * INR_TO_PHP
    df['date'] = pd.to_datetime(df['Date']).dt.date

    # Category simplification
    cat_col = next((c for c in df.columns if 'cat' in c.lower() or 'type' in c.lower()), None)
    df['category'] = df[cat_col].astype(str).str.lower()

    def map_category(x):
        if any(k in x for k in ['grocery', 'food']): return 'groceries'
        if any(k in x for k in ['utility', 'bill', 'water', 'electric']): return 'utilities'
        if any(k in x for k in ['fuel', 'taxi', 'bus', 'train', 'transport']): return 'transportation'
        return 'others'

    df['category'] = df['category'].apply(map_category)

    cleaned = df[['date', 'category', 'amount_php']]
    os.makedirs(os.path.dirname(PROCESSED_PATH), exist_ok=True)
    cleaned.to_csv(PROCESSED_PATH, index=False)
    print(f"✅ Cleaned dataset saved to {PROCESSED_PATH}")

if __name__ == "__main__":
    preprocess_data()
