# training.py
import os
import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.optimizers import Adam

# reuse your helper functions (or import them if in a module)
from pathlib import Path

CLEANED_CSV = os.path.join("data", "processed", "cleaned_transactions.csv")
MODELS_DIR = os.path.join("models")
os.makedirs(MODELS_DIR, exist_ok=True)

# ---------- load helpers (copied from your snippet) ----------
def load_and_prepare_dataframe(csv_path=CLEANED_CSV, dayfirst=True):
    df = pd.read_csv(csv_path)
    # detect date col
    date_col = None
    for c in df.columns:
        if c.lower() == "date":
            date_col = c
            break
    if date_col is None:
        for c in df.columns:
            if "date" in c.lower() or "time" in c.lower():
                date_col = c
                break
    if date_col is None:
        raise KeyError("Could not find a date column in the cleaned CSV. Columns found: {}".format(list(df.columns)))
    df[date_col] = pd.to_datetime(df[date_col], dayfirst=dayfirst, errors="coerce")
    n_missing = df[date_col].isna().sum()
    if n_missing > 0:
        print(f"Warning: {n_missing} entries could not be parsed as datetimes in column '{date_col}'. They will be NaT.")
    if date_col != "date":
        df = df.rename(columns={date_col: "date"})
    # detect amount column
    possible_amount_cols = [c for c in df.columns if "amount" in c.lower() or "amt" in c.lower() or "value" in c.lower()]
    if not possible_amount_cols:
        raise KeyError("Could not detect an 'amount' column automatically. Columns: {}".format(list(df.columns)))
    amount_col = possible_amount_cols[0]
    if not pd.api.types.is_numeric_dtype(df[amount_col]):
        df[amount_col] = pd.to_numeric(df[amount_col], errors="coerce")
    return df

def get_weekly_sums(df, value_col="amount_php", week_offset='MON'):
    set_index_df = df.set_index('date')
    weekly = set_index_df[value_col].resample('W-' + week_offset).sum()
    weekly = weekly.sort_index()
    # fill short gaps with 0 (you may prefer forward/backfill depending on semantics)
    weekly = weekly.fillna(0.0)
    return weekly

# ---------- model helpers ----------
def make_sequence_data(series_values, seq_len=4):
    """
    Given a 1D numpy array of weekly totals, make (X, y) sequences.
    X shape: (n_samples, seq_len, 1), y shape: (n_samples,)
    """
    X, y = [], []
    for i in range(len(series_values) - seq_len):
        X.append(series_values[i : i + seq_len])
        y.append(series_values[i + seq_len])
    if not X:
        return np.empty((0, seq_len, 1)), np.empty((0,))
    X = np.array(X).reshape(-1, seq_len, 1)
    y = np.array(y)
    return X, y

def build_lstm(seq_len=4):
    model = Sequential()
    model.add(LSTM(32, input_shape=(seq_len, 1), activation="tanh"))
    model.add(Dense(16, activation="relu"))
    model.add(Dense(1))
    model.compile(optimizer=Adam(learning_rate=0.001), loss="mse")
    return model

# ---------- training loop ----------
def train_for_category(weekly_series, category_name, seq_len=4, epochs=50, batch_size=8):
    vals = weekly_series.values.astype("float32")
    if len(vals) < seq_len + 2:
        print(f"Skipping '{category_name}': not enough weekly data ({len(vals)} entries).")
        return False

    # scale per-category
    scaler = MinMaxScaler(feature_range=(0, 1))
    vals_scaled = scaler.fit_transform(vals.reshape(-1, 1)).flatten()

    X, y = make_sequence_data(vals_scaled, seq_len=seq_len)
    if X.shape[0] < 3:
        print(f"Skipping '{category_name}': insufficient sequences ({X.shape[0]} samples).")
        return False

    # simple train/val split (last 10% val)
    split = max(1, int(len(X) * 0.9))
    X_train, X_val = X[:split], X[split:]
    y_train, y_val = y[:split], y[split:]

    model = build_lstm(seq_len=seq_len)
    es = EarlyStopping(monitor="val_loss", patience=6, restore_best_weights=True, verbose=1)

    print(f"Training model for category='{category_name}' | X_train={X_train.shape} y_train={y_train.shape}")
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val) if len(X_val) > 0 else None,
        epochs=epochs,
        batch_size=batch_size,
        callbacks=[es],
        verbose=2
    )

    # Save model + scaler artifacts
    safe_name = category_name.replace(" ", "_").lower()
    model_path = os.path.join(MODELS_DIR, f"lstm_{safe_name}.h5")
    model.save(model_path)
    print(f"Saved model to {model_path}")

    # Save scaler so simulation can inverse-transform predictions (optional)
    import pickle
    scaler_path = os.path.join(MODELS_DIR, f"scaler_{safe_name}.pkl")
    with open(scaler_path, "wb") as fh:
        pickle.dump(scaler, fh)
    print(f"Saved scaler to {scaler_path}")
    return True

def main(train_all=True, categories_to_train=None):
    df = load_and_prepare_dataframe(CLEANED_CSV, dayfirst=True)
    # ensure category column exists
    if "category" not in df.columns:
        raise KeyError("No 'category' column in cleaned CSV. Columns: {}".format(list(df.columns)))

    # unify column name for amount
    amount_cols = [c for c in df.columns if "amount" in c.lower() or "amt" in c.lower() or "value" in c.lower()]
    amount_col = amount_cols[0]
    # compute weekly sums per category
    all_cats = df['category'].dropna().unique().tolist()
    print("Found categories:", all_cats)

    if not train_all and categories_to_train:
        target_cats = [c for c in all_cats if c in categories_to_train]
    else:
        target_cats = all_cats

    trained_any = False
    for cat in target_cats:
        cat_df = df[df['category'] == cat][['date', amount_col]].copy()
        cat_df = cat_df.rename(columns={amount_col: "amount_php"})  # used by get_weekly_sums
        weekly = get_weekly_sums(cat_df, value_col="amount_php", week_offset='MON')
        # optionally restrict to recent data, e.g., last N weeks:
        # weekly = weekly.last("520W")  # last 10 years of weeks

        ok = train_for_category(weekly, cat, seq_len=4, epochs=50, batch_size=8)
        trained_any = trained_any or ok

    if not trained_any:
        print("No models were trained. Check data volume and categories.")

if __name__ == "__main__":
    # default: train for all categories found in the cleaned CSV
    main(train_all=True)
