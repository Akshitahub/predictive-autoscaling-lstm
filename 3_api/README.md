# Part 3: Model Deployment

| Folder | Description |
|---|---|
| web_demo_hf_space | Web page that runs the trained LSTM in the browser. Hosted on Hugging Face Spaces |
| fastapi_server | REST API version (POST /predict) using the same model. Can be run locally with `uvicorn app:app` or deployed with Docker |

Both use model_weights.json, which has the weights exported from the trained Keras model (1_lstm_forecasting/outputs/lstm_model.keras). The notebook writes this file (1_lstm_forecasting/outputs/model_weights.json). We checked the NumPy and JavaScript predictions against the Keras model on all 1156 test windows and they matched.

Live demo: https://huggingface.co/spaces/Akshitacat/predictive-autoscaling-lstm
