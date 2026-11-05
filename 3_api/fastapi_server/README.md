---
title: LSTM Auto-Scaling API
colorFrom: indigo
colorTo: blue
sdk: docker
app_port: 7860
pinned: false
---

# LSTM Auto-Scaling API

REST API for our minor project "ML-Based Predictive Load Balancing and Auto-Scaling in Cloud Environments" (IGDTUW).

Send the request counts for the last hour (12 values, one for every 5 minutes). The API returns the LSTM forecast for the next 5 minutes and the number of VMs needed.

Endpoints:

- `/` : demo page
- `/docs` : API documentation
- `POST /predict` : `{"recent_requests": [12 numbers], "vm_capacity": 100, "safety_margin": 0.4, "current_vms": 3}`

Run locally:

```
pip install -r requirements.txt
uvicorn app:app --port 7860
```
