"""
Deep Learning Time-Series Forecasting Module
Implements LSTM, Bi-LSTM, GRU, and CNN-LSTM for monsoon rainfall forecasting.
"""

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers, models, callbacks
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from pathlib import Path
import joblib
import warnings
warnings.filterwarnings("ignore")

tf.random.set_seed(42)
np.random.seed(42)

SEQ_LEN = 30  # lookback window in days


def create_sequences(data: np.ndarray, seq_len: int = SEQ_LEN):
    """Convert time-series array into (X, y) sequences."""
    X, y = [], []
    for i in range(len(data) - seq_len):
        X.append(data[i:i + seq_len])
        y.append(data[i + seq_len])
    return np.array(X), np.array(y)


def build_lstm(input_shape, units=64):
    model = models.Sequential([
        layers.Input(shape=input_shape),
        layers.LSTM(units, return_sequences=True),
        layers.Dropout(0.2),
        layers.LSTM(units // 2),
        layers.Dropout(0.2),
        layers.Dense(1)
    ], name="LSTM")
    model.compile(optimizer="adam", loss="mse", metrics=["mae"])
    return model


def build_bilstm(input_shape, units=64):
    model = models.Sequential([
        layers.Input(shape=input_shape),
        layers.Bidirectional(layers.LSTM(units, return_sequences=True)),
        layers.Dropout(0.2),
        layers.Bidirectional(layers.LSTM(units // 2)),
        layers.Dropout(0.2),
        layers.Dense(1)
    ], name="BiLSTM")
    model.compile(optimizer="adam", loss="mse", metrics=["mae"])
    return model


def build_gru(input_shape, units=64):
    model = models.Sequential([
        layers.Input(shape=input_shape),
        layers.GRU(units, return_sequences=True),
        layers.Dropout(0.2),
        layers.GRU(units // 2),
        layers.Dropout(0.2),
        layers.Dense(1)
    ], name="GRU")
    model.compile(optimizer="adam", loss="mse", metrics=["mae"])
    return model


def build_cnn_lstm(input_shape, filters=64, units=64):
    model = models.Sequential([
        layers.Input(shape=input_shape),
        layers.Conv1D(filters, kernel_size=3, activation="relu", padding="same"),
        layers.MaxPooling1D(pool_size=2),
        layers.LSTM(units, return_sequences=False),
        layers.Dropout(0.2),
        layers.Dense(1)
    ], name="CNN_LSTM")
    model.compile(optimizer="adam", loss="mse", metrics=["mae"])
    return model


def evaluate(y_true, y_pred) -> dict:
    mse = mean_squared_error(y_true, y_pred)
    return {
        "MAE":  round(mean_absolute_error(y_true, y_pred), 4),
        "RMSE": round(np.sqrt(mse), 4),
        "R2":   round(r2_score(y_true, y_pred), 4),
    }


def train_dl_models(series: np.ndarray, save_dir: str = "models",
                    epochs: int = 50, batch_size: int = 32) -> pd.DataFrame:
    """
    Train all DL models on a univariate time series.
    series: 1D numpy array of scaled target values.
    """
    Path(save_dir).mkdir(parents=True, exist_ok=True)

    X, y = create_sequences(series, SEQ_LEN)
    split = int(len(X) * 0.8)
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

    # Reshape for models: (samples, timesteps, features)
    X_train = X_train.reshape(*X_train.shape, 1)
    X_test  = X_test.reshape(*X_test.shape, 1)
    input_shape = (SEQ_LEN, 1)

    cb = [
        callbacks.EarlyStopping(patience=10, restore_best_weights=True),
        callbacks.ReduceLROnPlateau(patience=5, factor=0.5, verbose=0)
    ]

    builders = {
        "LSTM":    build_lstm,
        "BiLSTM":  build_bilstm,
        "GRU":     build_gru,
        "CNN_LSTM": build_cnn_lstm,
    }

    results = []
    for name, builder in builders.items():
        print(f"  Training {name}...")
        model = builder(input_shape)
        model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size,
                  validation_split=0.1, callbacks=cb, verbose=0)
        preds = model.predict(X_test, verbose=0).ravel()
        metrics = evaluate(y_test, preds)
        metrics["model"] = name
        results.append(metrics)
        model.save(f"{save_dir}/{name}.keras")
        print(f"    RMSE={metrics['RMSE']}, R2={metrics['R2']}")

    df = pd.DataFrame(results).set_index("model").sort_values("RMSE")
    df.to_csv(f"{save_dir}/dl_comparison.csv")
    return df


def load_dl_model(name: str, save_dir: str = "models") -> tf.keras.Model:
    return tf.keras.models.load_model(f"{save_dir}/{name}.keras")


def forecast_next_n(model: tf.keras.Model, last_sequence: np.ndarray,
                    n_steps: int = 7) -> np.ndarray:
    """Autoregressively forecast n_steps ahead."""
    seq = last_sequence.copy().reshape(1, SEQ_LEN, 1)
    preds = []
    for _ in range(n_steps):
        p = model.predict(seq, verbose=0)[0, 0]
        preds.append(p)
        seq = np.roll(seq, -1, axis=1)
        seq[0, -1, 0] = p
    return np.array(preds)


if __name__ == "__main__":
    from src.data.preprocessing import run_pipeline, split_train_test
    from sklearn.preprocessing import MinMaxScaler

    df = run_pipeline()
    city_df = df[df["city"] == "Mumbai"].sort_values("date")
    scaler = MinMaxScaler()
    series = scaler.fit_transform(city_df[["precipitation_sum"]]).ravel()

    results = train_dl_models(series, epochs=30)
    print(results)
