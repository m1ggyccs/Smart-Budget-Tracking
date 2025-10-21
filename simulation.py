# simulation.py
import pandas as pd
import numpy as np
import os, joblib
from config import PROCESSED_PATH, MODEL_DIR, OUTPUT_DIR
from sklearn.ensemble import IsolationForest
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt

def run_simulation():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    df = pd.read_csv(PROCESSED_PATH, parse_dates=['date'])
    weekly = df.groupby(['date', 'category']).amount_php.sum().reset_index()
    weekly = weekly.pivot(index='date', columns='category', values='amount_php').fillna(0)
    weekly = weekly.resample('W-MON').sum()

    anomalies = []
    for cat in weekly.columns:
        print(f"🔍 Simulating {cat}...")

        arima_path = os.path.join(MODEL_DIR, f"arima_{cat}.pkl")
        lstm_path = os.path.join(MODEL_DIR, f"lstm_{cat}.h5")

        if os.path.exists(arima_path):
            model = joblib.load(arima_path)
            preds = model.predict(n_periods=len(weekly))
        else:
            model = load_model(lstm_path)
            vals = weekly[cat].values.astype('float32')
            sc = MinMaxScaler()
            scaled = sc.fit_transform(vals.reshape(-1,1))
            seq = 8
            X = np.array([scaled[i:i+seq] for i in range(len(scaled)-seq)])
            preds_scaled = model.predict(X)
            preds = sc.inverse_transform(preds_scaled).ravel()

        residuals = weekly[cat].values[-len(preds):] - preds
        iso = IsolationForest(contamination=0.05, random_state=42)
        flags = iso.fit_predict(residuals.reshape(-1,1))
        weekly[f'{cat}_anomaly'] = (flags == -1)
        anomalies.extend(weekly[weekly[f'{cat}_anomaly']].index.tolist())

        plt.figure(figsize=(10,4))
        plt.plot(weekly.index, weekly[cat], label='Actual')
        plt.plot(weekly.index[-len(preds):], preds, label='Predicted')
        plt.legend()
        plt.title(f"{cat} Forecast Simulation")
        plt.savefig(os.path.join(OUTPUT_DIR, f"forecast_{cat}.png"))
        plt.close()

    print(f"✅ Simulation complete. {len(set(anomalies))} anomaly weeks detected.")
    weekly.to_csv(os.path.join(OUTPUT_DIR, "weekly_simulation_output.csv"))

if __name__ == "__main__":
    run_simulation()
