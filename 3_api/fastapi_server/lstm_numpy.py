"""
LSTM prediction using NumPy and the weights exported from Keras.

Model (same as the notebook): LSTM(64) -> Dropout -> LSTM(32) -> Dropout -> Dense(16, relu) -> Dense(1)
Dropout is not used during prediction so it is skipped.
Gate order in Keras weights: input, forget, cell, output
"""
import json
import numpy as np


def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def _lstm(x_seq, kernel, recurrent, bias, return_sequences):
    """x_seq: (timesteps, features) -> (timesteps, units) or (units,)"""
    units = recurrent.shape[0]
    h = np.zeros(units)
    c = np.zeros(units)
    outputs = []
    for x_t in x_seq:
        z = x_t @ kernel + h @ recurrent + bias
        i = _sigmoid(z[:units])
        f = _sigmoid(z[units:2 * units])
        g = np.tanh(z[2 * units:3 * units])
        o = _sigmoid(z[3 * units:])
        c = f * c + i * g
        h = o * np.tanh(c)
        outputs.append(h)
    return np.array(outputs) if return_sequences else h


class LSTMForecaster:
    def __init__(self, weights_path="model_weights.json"):
        with open(weights_path) as f:
            d = json.load(f)
        L = {k: [np.array(w, dtype=np.float64) for w in v] for k, v in d["layers"].items()}
        self.l1, self.l2, self.d1, self.d2 = (L[name] for name in d["order"])
        s = d["scaler"]
        self.s_min = float(s["min"][0])        # MinMaxScaler: x_scaled = x * scale + min
        self.s_scale = float(s["scale"][0])
        self.window = int(d["window"])
        self.interval_minutes = int(d["interval_minutes"])

    def predict(self, recent_requests):
        """recent_requests: last 12 request counts (oldest first). Returns the forecast for the next interval."""
        x = np.asarray(recent_requests, dtype=np.float64)[-self.window:]
        x = (x * self.s_scale + self.s_min).reshape(-1, 1)
        h = _lstm(x, *self.l1, return_sequences=True)
        h = _lstm(h, *self.l2, return_sequences=False)
        h = np.maximum(0.0, h @ self.d1[0] + self.d1[1])
        y_scaled = float((h @ self.d2[0] + self.d2[1])[0])
        return max(0.0, (y_scaled - self.s_min) / self.s_scale)
