"""
LSTM prediction using NumPy and the weights exported from Keras (model_weights.json).

Model (same as the notebook): LSTM(64) -> Dropout -> Dense(16, relu) -> Dense(1)
The layer list, window size and scaler are read from model_weights.json, so the
same code also works for a stacked LSTM. Dropout is only used during training,
so it is skipped here.
Gate order in Keras LSTM weights: input, forget, cell, output
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
        i = _sigmoid(z[:units])                 # input gate: how much new information to store
        f = _sigmoid(z[units:2 * units])        # forget gate: how much old memory to keep
        g = np.tanh(z[2 * units:3 * units])     # candidate values for the memory
        o = _sigmoid(z[3 * units:])             # output gate: how much memory to show
        c = f * c + i * g                       # cell state (long-term memory)
        h = o * np.tanh(c)                      # hidden state (output at this step)
        outputs.append(h)
    return np.array(outputs) if return_sequences else h


class LSTMForecaster:
    def __init__(self, weights_path="model_weights.json"):
        with open(weights_path) as f:
            d = json.load(f)
        self.order = d["order"]
        # older exports have no "types" list, so fall back to the layer names
        self.types = d.get("types") or ["lstm" if n.startswith("lstm") else "dense" for n in self.order]
        self.layers = [[np.array(w, dtype=np.float64) for w in d["layers"][n]] for n in self.order]
        s = d["scaler"]
        self.s_min = float(s["min"][0])        # MinMaxScaler: x_scaled = x * scale + min
        self.s_scale = float(s["scale"][0])
        self.window = int(d["window"])
        self.interval_minutes = int(d["interval_minutes"])
        self.n_params = int(d.get("n_params") or sum(w.size for layer in self.layers for w in layer))
        n_lstm = self.types.count("lstm")
        units = [layer[1].shape[0] if t == "lstm" else layer[0].shape[1] for layer, t in zip(self.layers, self.types)]
        self.description = (f"{n_lstm}-layer LSTM ("
                            + "-".join(f"{'LSTM' if t == 'lstm' else 'Dense'}({u})" for t, u in zip(self.types, units))
                            + f"), window {self.window}")

    def predict(self, recent_requests):
        """recent_requests: the last `window` request counts (oldest first). Returns the forecast for the next interval."""
        x = np.asarray(recent_requests, dtype=np.float64)[-self.window:]
        h = (x * self.s_scale + self.s_min).reshape(-1, 1)
        last_lstm = max(i for i, t in enumerate(self.types) if t == "lstm")
        for i, (t, w) in enumerate(zip(self.types, self.layers)):
            if t == "lstm":
                h = _lstm(h, *w, return_sequences=i < last_lstm)
            else:
                h = h @ w[0] + w[1]
                if i < len(self.layers) - 1:   # hidden Dense uses ReLU, the output layer is linear
                    h = np.maximum(0.0, h)
        y_scaled = float(h[0])
        return max(0.0, (y_scaled - self.s_min) / self.s_scale)
