# Part 5: ARIMA Baseline

A classical forecasting baseline for the LSTM, evaluated on the same data, split and test intervals, and fed into the same CloudSim Plus scaling rule.

## How it was run

1. Open `arima_baseline.ipynb` in Google Colab (a CPU runtime is enough) and run all cells. It rebuilds the 5-minute request series from the NASA log, chooses the ARIMA order by AIC on the training data only, and makes one-step-ahead (5-minute) forecasts for the 1,192 test intervals with the parameters fixed.
2. The notebook downloads `predictions_arima.csv`. Copy it into `2_cloudsim_simulation/`.
3. Run the simulation on that file, for example in IntelliJ with program arguments `predictions_arima.csv 0.4`, or:

```
mvn compile exec:java -Dexec.args="predictions_arima.csv 0.4"
```

The ARIMA forecast is stored in the `lstm_pred` column, so in this run the **PREDICTIVE_LSTM row of the output is the ARIMA-driven policy**. Run the simulation again without arguments afterwards to restore the LSTM results in `2_cloudsim_simulation/results/`.

## Results

Selected model: **ARIMA(2, 0, 1)**, five parameters (two AR, one MA, constant, noise variance).

Forecasting error on the test set (requests per 5 minutes):

| Model | Parameters | MAE | RMSE | MAPE (%) |
|---|---|---|---|---|
| Naive (last value) | 0 | 44.45 | 58.27 | 33.28 |
| Moving Average (1 hour) | 0 | 42.43 | 55.81 | 57.53 |
| LSTM (1 layer, 3-hour window) | 17,953 | 39.62 | 51.69 | 46.36 |
| **ARIMA(2, 0, 1)** | 5 | **38.93** | **50.93** | 42.77 |

CloudSim Plus simulation, safety margin 0.40 (same settings as Part 2):

| Policy | VM-hours | Cost ($) | Avg response (s) | SLA violations | Under-provisioned |
|---|---|---|---|---|---|
| Static (sized for peak) | 596.0 | 29.80 | 15.06 | 0.00% | 0.00% |
| Reactive (threshold) | 373.3 | 18.66 | 17.81 | 1.49% | 3.61% |
| Predictive (Moving Average) | 353.0 | 17.65 | 17.29 | 1.53% | 1.76% |
| Predictive (LSTM) | 352.1 | 17.60 | 16.15 | 0.79% | 1.34% |
| **Predictive (ARIMA)** | 352.8 | 17.64 | **15.63** | **0.48%** | **1.01%** |

On this trace ARIMA forecast slightly better than the LSTM (1.7% lower MAE) and led to fewer SLA violations at almost the same cost. The scaling results follow forecast accuracy: the more accurate the forecast, the fewer violations, while all three predictive policies use about the same VM-hours.

## Files

- `arima_baseline.ipynb`: the notebook
- `results/arima_metrics.csv`: forecasting error of the selected ARIMA model
- `results/summary_arima_simulation.csv`: CloudSim Plus result for the ARIMA-driven policy
