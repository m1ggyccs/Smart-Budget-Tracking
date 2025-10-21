# training.py
import pandas as pd
import numpy as np
import os, joblib
from config import PROCESSED_PATH, MODEL_DIR, TRAIN_FRAC, VAL_FRAC, RANDOM_SEED
import pmdarima as pm
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.preprocessing import MinMaxScaler
import warnings
warnings.filterwarnings("ignore")

def time_split(df, train_frac=TRAIN_FRAC, val_frac=VAL_FRAC):
    n = len(df)
    n_train = int(n * train_frac)
    n_val = int(n * val_frac)
    return df[:n_train], df[n_train:n_train+n_val], df[n_train+n_val:]

def train_models():
    os.makedirs(MODEL_DIR, exist_ok=True)
    df = pd.read_csv(PROCESSED_PATH, parse_dates=['date'])
    df = df.groupby(['date', 'category']).amount_php.sum().reset_index()
    weekly = df.pivot(index='date', columns='category', values='amount_php').fillna(0)
    weekly = weekly.resample('W-MON', on='date').sum()

    for cat in weekly.columns:
        print(f"\n📈 Training models for {cat}...")
        series = weekly[cat].fillna(0)

        train, val, test = time_split(series)

        # --- ARIMA ---
        try:
            arima_model = pm.auto_arima(train, seasonal=True, m=52, suppress_warnings=True)
            joblib.dump(arima_model, os.path.join(MODEL_DIR, f"arima_{cat}.pkl"))
            print("✅ Saved ARIMA model")
        except Exception as e:
            print("ARIMA failed:", e)

        # --- LSTM ---
        scaler = MinMaxScaler()
        scaled = scaler.fit_transform(series.values.reshape(-1, 1))
        SEQ = 8
        X, y = [], []
        for i in range(len(scaled)-SEQ):
            X.append(scaled[i:i+SEQ])
            y.append(scaled[i+SEQ])
        X, y = np.array(X), np.array(y)

        model = Sequential([
            LSTM(32, input_shape=(SEQ,1)),
            Dense(1)
        ])
        model.compile(optimizer='adam', loss='mse')
        es = EarlyStopping(monitor='loss', patience=10, restore_best_weights=True)
        model.fit(X, y, epochs=100, batch_size=8, verbose=0, callbacks=[es])
        model.save(os.path.join(MODEL_DIR, f"lstm_{cat}.h5"))
        print("✅ Saved LSTM model")

if __name__ == "__main__":
    train_models()
