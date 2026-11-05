---
title: LSTM Auto-Scaling Demo
colorFrom: indigo
colorTo: blue
sdk: static
pinned: false
---

# LSTM Auto-Scaling Demo

Demo for our minor project "ML-Based Predictive Load Balancing and Auto-Scaling in Cloud Environments" (IGDTUW).

Enter the request counts for the last hour (12 values, one for every 5 minutes). The LSTM predicts the next 5 minutes and suggests how many VMs are needed.

The model was trained with TensorFlow/Keras on the NASA Kennedy Space Center HTTP logs (July 1995). The weights are saved in model_weights.json and lstm.js does the prediction in the browser.
