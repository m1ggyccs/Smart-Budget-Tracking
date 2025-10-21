from tensorflow.keras.models import load_model
from tensorflow.keras.optimizers import Adam
import pickle
import os

lstm_path = os.path.join("models", "lstm_groceries.h5")
scaler_path = os.path.join("models", "scaler_groceries.pkl")

def safe_load_model(path, compile_args=None):
    """
    Try loading normally. If deserialization of loss/metrics fails (TypeError about e.g. 'mse'),
    load with compile=False and re-compile using compile_args (dict).
    """
    try:
        print("Trying to load model (normal)...", path)
        model = load_model(path)  # attempt normal load+compile from HDF5 config
        print("Loaded model (compiled) from", path)
        return model
    except TypeError as ex:
        # Known issue: cannot locate function 'mse' (or other metric/loss) during deserialization.
        print("Model load produced TypeError (likely unresolved loss/metric).")
        print("Error:", ex)
        print("Attempting to load with compile=False and then re-compile manually.")
        model = load_model(path, compile=False)
        if compile_args is not None:
            print("Re-compiling model with provided compile_args.")
            model.compile(**compile_args)
        else:
            # sensible default compile config
            model.compile(optimizer=Adam(learning_rate=0.001), loss="mse")
        print("Model loaded and compiled manually.")
        return model

def run_simulation():
    # load model safely
    model = safe_load_model(lstm_path, compile_args={"optimizer": Adam(1e-3), "loss": "mse"})

    # load scaler if exists
    scaler = None
    if os.path.exists(scaler_path):
        with open(scaler_path, "rb") as fh:
            scaler = pickle.load(fh)
    else:
        print("No scaler found at", scaler_path, "- predictions will be in scaled space.")

    # prepare X_input same as training seq_len
    # example: seq_len=4, build numpy array shape (1, seq_len, 1)
    import numpy as np
    # replace this with your real last-4-weeks values (unscaled), then scale before predicting
    last_vals = np.array([1000.0, 1100.0, 900.0, 1200.0]).reshape(-1, 1)  # example
    if scaler is not None:
        last_vals_scaled = scaler.transform(last_vals).flatten()
    else:
        # if no scaler, try basic minmax scaling as fallback (not ideal)
        mn, mx = last_vals.min(), last_vals.max()
        if mx == mn:
            last_vals_scaled = np.zeros_like(last_vals).flatten()
        else:
            last_vals_scaled = (last_vals.flatten() - mn) / (mx - mn)

    X_input = last_vals_scaled.reshape(1, len(last_vals_scaled), 1)
    preds_scaled = model.predict(X_input)
    if scaler is not None:
        pred = scaler.inverse_transform(preds_scaled.reshape(-1, 1)).flatten()[0]
    else:
        pred = float(preds_scaled.flatten()[0])
    print("Predicted next-week groceries (approx):", pred)

if __name__ == "__main__":
    run_simulation()
