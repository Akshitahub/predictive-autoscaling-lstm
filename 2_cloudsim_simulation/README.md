# Part 2: CloudSim Plus Auto-Scaling Simulation

This project runs the NASA workload (from predictions.csv, made by the LSTM notebook) in CloudSim Plus 8.5.7 with 4 different auto-scaling policies, and compares their cost and SLA.

| Policy | How the number of VMs is decided |
|---|---|
| STATIC | Fixed number of VMs, enough for the peak load |
| REACTIVE | Uses the load of the previous interval. Scales up above 80% utilisation and down below 40% |
| PREDICTIVE_MA | Uses the moving average forecast plus a safety margin |
| PREDICTIVE_LSTM | Uses the LSTM forecast plus a safety margin |

## Requirements

- Java 17 or above
- Maven (already included in IntelliJ IDEA)
- Python with pandas and matplotlib (only needed for the graphs)

## Running in IntelliJ IDEA

1. Open this folder in IntelliJ (File, Open) and let it load as a Maven project. It downloads CloudSim Plus the first time.
2. If it asks for a JDK, download JDK 21 from the prompt.
3. Run `src/main/java/org/minorproject/AutoScalingSimulation.java`.
4. The results table is printed in the console and the CSV files are saved in the results folder.

Using the terminal instead:

```
mvn compile exec:java
```

Note: annotation processing is turned off in pom.xml (`<proc>none</proc>`). Without this the build fails on newer JDKs because of Lombok, which comes with CloudSim Plus.

## Graphs

```
pip install pandas matplotlib
python plot_results.py
```

The graphs (fig5 to fig10) are saved in results/figures.

## Output files

- results/summary.csv: main results (VM-hours, cost, response time, SLA violation %, under-provisioned %, utilisation)
- results/timeline_POLICY.csv: demand, forecast, active VMs and capacity for every interval
- results/figures: graphs for the report and presentation

## Parameters

These are at the top of AutoScalingSimulation.java:

| Parameter | Value | Meaning |
|---|---|---|
| STEP | 300 s | One interval is 5 minutes, same as the LSTM |
| VM_CAPACITY | 100 | Requests one VM can handle in one interval |
| BOOT_TIME | 120 s | Time a new VM takes to start |
| SLA_RESPONSE_TIME | 60 s | Response time above this counts as an SLA violation |
| SAFETY_MARGIN | 0.40 | Extra capacity added on top of the forecast |
| PRICE_PER_VM_HOUR | $0.05 | Assumed price per VM hour |

To try a different safety margin:

```
mvn compile exec:java -Dexec.args="predictions.csv 0.3"
```

A lower margin is cheaper but gives more SLA violations.

## Results (safety margin 0.40)

| Policy | VM-hours | SLA violations | Under-provisioned |
|---|---|---|---|
| Static | 596.0 | 0.00% | 0.00% |
| Reactive | 373.3 | 1.49% | 3.61% |
| Predictive (MA) | 353.0 | 1.53% | 1.76% |
| Predictive (LSTM) | 351.8 | 0.90% | 1.01% |
