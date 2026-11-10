# ML-Based Predictive Load Balancing and Auto-Scaling in Cloud Environments

Minor Project, Cloud Computing | B.Tech ECE (AI), IGDTUW
Team: Akshita, Kanak, Julie

Live demo: https://huggingface.co/spaces/Akshitacat/predictive-autoscaling-lstm

In this project we use an LSTM model to predict web traffic 5 minutes ahead from the last 3 hours of traffic. The predicted load is used to add or remove VMs in advance, instead of waiting for the servers to get overloaded first. We compare this with normal threshold-based (reactive) scaling and static provisioning using CloudSim Plus, on real NASA web server logs.

Workflow:

1. NASA HTTP logs are converted to requests per 5 minutes
2. LSTM is trained to forecast the next interval (Google Colab). Experiments compare window sizes, time features, model types (1-layer LSTM, 2-layer LSTM, GRU) and prediction horizons
3. Predictions are exported to predictions.csv
4. CloudSim Plus runs 4 scaling policies on the same workload
5. The trained model is deployed as a web demo on Hugging Face

## Results

CloudSim Plus simulation on the test period (1192 intervals of 5 minutes, around 4 days):

| Policy | VM-hours | Cost ($) | Avg response (s) | SLA violations | Under-provisioned |
|---|---|---|---|---|---|
| Static (sized for peak) | 596.0 | 29.80 | 15.06 | 0.00% | 0.00% |
| Reactive (threshold 80/40%) | 373.3 | 18.66 | 17.81 | 1.49% | 3.61% |
| Predictive (Moving Average) | 353.0 | 17.65 | 17.29 | 1.53% | 1.76% |
| Predictive (LSTM) | 352.1 | 17.60 | 16.15 | 0.79% | 1.34% |
| Predictive (ARIMA) | 352.8 | 17.64 | 15.63 | 0.48% | 1.01% |

Compared to reactive scaling, the LSTM based scaler had 47% fewer SLA violations, 63% less under-provisioned time, 9% lower average response time and 5.7% lower cost. Compared to the moving average scaler it had 48% fewer SLA violations at almost the same cost. Compared to static provisioning it was 41% cheaper.

Forecast error on the test set (requests per 5 min):

| Model | MAE | RMSE |
|---|---|---|
| Naive (last value) | 44.45 | 58.27 |
| Moving Average (1 hour) | 42.43 | 55.81 |
| LSTM (1 layer, 64 units, 3-hour window) | 39.62 | 51.69 |
| ARIMA(2, 0, 1) | 38.93 | 50.93 |

The final LSTM is 10.9% more accurate (MAE) than the naive forecast and 6.6% more accurate than the 1-hour moving average. In the model type experiment a 2-layer GRU (MAE 39.27) and the 1-layer LSTM (39.62) were within 1% of each other; we kept the LSTM as the proposed model. Adding time-of-day and day-of-week features did not help (MAE 41.98 vs 40.82). All experiment tables and graphs are in `1_lstm_forecasting/outputs`.

As a classical baseline we also fitted ARIMA(2, 0, 1) on the same data (`5_arima_baseline`). It forecast slightly better than the LSTM (MAE 38.93) and, driving the same scaling rule, gave 0.48% SLA violations. Forecast-driven scaling helps in proportion to forecast accuracy: on this smooth, single-signal trace a five-parameter ARIMA matched the LSTM.

## Folders

| Folder | Contents |
|---|---|
| 1_lstm_forecasting | Colab notebook and outputs folder (model, scaler, model_weights.json, predictions.csv, experiment tables exp1 to exp4, fig1 to fig4, fig_exp1 to fig_exp4) |
| 2_cloudsim_simulation | CloudSim Plus Java project (Maven) and results folder (summary.csv, timelines, fig5 to fig10) |
| 3_api | Web demo (Hugging Face) and FastAPI server for the model |
| 4_report | LaTeX files for the report |
| 5_arima_baseline | ARIMA baseline notebook and results (forecast accuracy and CloudSim run) |

## How to run

1. Forecasting: open `1_lstm_forecasting/lstm_workload_forecasting.ipynb` in Google Colab, select T4 GPU and run all cells.
2. Simulation: open `2_cloudsim_simulation` in IntelliJ IDEA as a Maven project and run `AutoScalingSimulation.java`.
3. Graphs: run `python plot_results.py` inside `2_cloudsim_simulation`.

## Dataset

NASA Kennedy Space Center WWW server logs (July 1995), Internet Traffic Archive (ita.ee.lbl.gov). We aggregated the requests into 5 minute intervals.

## Tools used

Python, TensorFlow/Keras, scikit-learn, pandas, matplotlib, Java 17, CloudSim Plus 8.5.7, Maven, FastAPI, Hugging Face Spaces
