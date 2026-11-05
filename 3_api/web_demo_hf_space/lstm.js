// LSTM prediction in JavaScript using the weights exported from Keras.
// Model: LSTM(64) -> LSTM(32) -> Dense(16, relu) -> Dense(1)
// Gate order in Keras weights: input, forget, cell, output
(function (root) {
  const sigmoid = (x) => 1 / (1 + Math.exp(-x));

  function lstm(xs, kernel, recurrent, bias, returnSequences) {
    const units = recurrent.length;            // recurrent: units x 4*units
    let h = new Array(units).fill(0), c = new Array(units).fill(0);
    const out = [];
    for (const x of xs) {
      const z = bias.slice();
      for (let k = 0; k < x.length; k++) { const row = kernel[k], xv = x[k]; for (let j = 0; j < z.length; j++) z[j] += xv * row[j]; }
      for (let k = 0; k < units; k++) { const row = recurrent[k], hv = h[k]; for (let j = 0; j < z.length; j++) z[j] += hv * row[j]; }
      const nh = new Array(units), nc = new Array(units);
      for (let u = 0; u < units; u++) {
        const i = sigmoid(z[u]), f = sigmoid(z[units + u]), g = Math.tanh(z[2 * units + u]), o = sigmoid(z[3 * units + u]);
        nc[u] = f * c[u] + i * g; nh[u] = o * Math.tanh(nc[u]);
      }
      h = nh; c = nc; out.push(h);
    }
    return returnSequences ? out : h;
  }

  function dense(v, W, b, relu) {
    const y = b.slice();
    for (let k = 0; k < v.length; k++) for (let j = 0; j < y.length; j++) y[j] += v[k] * W[k][j];
    return relu ? y.map((t) => Math.max(0, t)) : y;
  }

  function makeForecaster(d) {
    const [l1, l2, d1, d2] = d.order.map((n) => d.layers[n]);
    const sMin = d.scaler.min[0], sScale = d.scaler.scale[0], win = d.window;
    return {
      window: win,
      predict(recent) {
        const x = recent.slice(-win).map((v) => [v * sScale + sMin]);
        let h = lstm(x, l1[0], l1[1], l1[2], true);
        h = lstm(h, l2[0], l2[1], l2[2], false);
        h = dense(h, d1[0], d1[1], true);
        const y = dense(h, d2[0], d2[1], false)[0];
        return Math.max(0, (y - sMin) / sScale);
      },
    };
  }

  root.makeForecaster = makeForecaster;
  if (typeof module !== "undefined") module.exports = { makeForecaster };
})(typeof window !== "undefined" ? window : globalThis);
