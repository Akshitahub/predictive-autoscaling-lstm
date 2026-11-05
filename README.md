# ML-Based Predictive Load Balancing and Auto-Scaling in Cloud Environments

Minor Project, Cloud Computing | B.Tech ECE (AI), IGDTUW
Team: Akshita, Kanak, Julie

Live demo: https://huggingface.co/spaces/Akshitacat/predictive-autoscaling-lstm

In this project we use an LSTM model to predict web traffic 5 minutes ahead. The predicted load is used to add or remove VMs in advance, instead of waiting for the servers to get overloaded first. We compare this with normal threshold-based (reactive) scaling and static provisioning using CloudSim Plus, on real NASA web server logs.

Workflow:

1. NASA HTTP logs are converted to requests per 5 minutes
2. LSTM is trained to forecast the next interval (Google Colab)
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
| Predictive (LSTM) | 351.8 | 17.59 | 16.67 | 0.90% | 1.01% |

Compared to reactive scaling, the LSTM based scaler had 40% fewer SLA violations, 72% less under-provisioned time and 5.8% lower cost. Compared to static provisioning it was 41% cheaper.

Forecast error on the test set (requests per 5 min):

| Model | MAE | RMSE |
|---|---|---|
| Naive (last value) | 44.4 | 58.3 |
| Moving Average (1 hour) | 42.4 | 55.8 |
| LSTM | 41.0 | 53.5 |

## Folders

| Folder | Contents |
|---|---|
| 1_lstm_forecasting | Colab notebook and outputs folder (model, scaler, predictions.csv, fig1 to fig4) |
| 2_cloudsim_simulation | CloudSim Plus Java project (Maven) and results folder (summary.csv, timelines, fig5 to fig10) |
| 3_api | Web demo (Hugging Face) and FastAPI server for the model |
| 4_report | LaTeX files for the report |

## How to run

1. Forecasting: open `1_lstm_forecasting/lstm_workload_forecasting.ipynb` in Google Colab, select T4 GPU and run all cells.
2. Simulation: open `2_cloudsim_simulation` in IntelliJ IDEA as a Maven project and run `AutoScalingSimulation.java`.
3. Graphs: run `python plot_results.py` inside `2_cloudsim_simulation`.

## Dataset

NASA Kennedy Space Center WWW server logs (July 1995), Internet Traffic Archive (ita.ee.lbl.gov). We aggregated the requests into 5 minute intervals.

## Tools used

Python, TensorFlow/Keras, scikit-learn, pandas, matplotlib, Java 17, CloudSim Plus 8.5.7, Maven, FastAPI, Hugging Face Spaces
