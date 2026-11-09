// LSTM prediction in JavaScript using the weights exported from Keras (model_weights.json).
// Model: LSTM(64) -> Dense(16, relu) -> Dense(1). The layer list and window size come
// from model_weights.json, so the same code also works for a stacked LSTM.
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
    const types = d.types || d.order.map((n) => (n.startsWith('lstm') ? 'lstm' : 'dense'));
    const layers = d.order.map((n) => d.layers[n]);
    const lastLstm = types.lastIndexOf('lstm');
    const sMin = d.scaler.min[0], sScale = d.scaler.scale[0], win = d.window;
    let nParams = d.n_params;
    if (!nParams) {
      const count = (a) => (Array.isArray(a) ? a.reduce((s, v) => s + count(v), 0) : 1);
      nParams = count(layers);
    }
    return {
      window: win,
      nParams,
      nLstm: types.filter((t) => t === 'lstm').length,
      predict(recent) {
        let h = recent.slice(-win).map((v) => [v * sScale + sMin]);
        layers.forEach((w, i) => {
          if (types[i] === 'lstm') h = lstm(h, w[0], w[1], w[2], i < lastLstm);
          else h = dense(h, w[0], w[1], i < layers.length - 1);
        });
        return Math.max(0, (h[0] - sMin) / sScale);
      },
    };
  }

  root.makeForecaster = makeForecaster;
  if (typeof module !== "undefined") module.exports = { makeForecaster };
})(typeof window !== "undefined" ? window : globalThis);
