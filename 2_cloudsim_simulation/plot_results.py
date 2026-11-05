"""
Makes the result graphs from the CloudSim Plus output.
Run AFTER the Java simulation:   python plot_results.py
Needs:  pip install pandas matplotlib
Saves PNGs into results/figures/
"""
import os
import pandas as pd
import matplotlib.pyplot as plt

RES = "results"
FIG = os.path.join(RES, "figures")
os.makedirs(FIG, exist_ok=True)

ORDER = ["STATIC", "REACTIVE", "PREDICTIVE_MA", "PREDICTIVE_LSTM"]
LABEL = {"STATIC": "Static (peak)", "REACTIVE": "Reactive (threshold)",
         "PREDICTIVE_MA": "Predictive (MA)", "PREDICTIVE_LSTM": "Predictive (LSTM)"}
COLOR = {"STATIC": "#8c8c8c", "REACTIVE": "#dd8452",
         "PREDICTIVE_MA": "#55a868", "PREDICTIVE_LSTM": "#c44e52"}

s = pd.read_csv(os.path.join(RES, "summary.csv")).set_index("policy").loc[ORDER]
names = [LABEL[p] for p in ORDER]
cols = [COLOR[p] for p in ORDER]


def bar(metric, title, ylabel, fname, fmt="{:.1f}"):
    fig, ax = plt.subplots(figsize=(8, 4.5))
    bars = ax.bar(names, s[metric], color=cols)
    for b, v in zip(bars, s[metric]):
        ax.text(b.get_x() + b.get_width() / 2, b.get_height(), fmt.format(v),
                ha="center", va="bottom", fontsize=10)
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.grid(axis="y", alpha=0.3)
    plt.xticks(rotation=10)
    plt.tight_layout()
    plt.savefig(os.path.join(FIG, fname), dpi=200)
    plt.close()


bar("vm_hours", "Resource usage (lower is cheaper)", "Active VM-hours", "fig5_vm_hours.png")
bar("sla_violation_pct", "SLA violations (response time > 60 s)", "% of request batches",
    "fig6_sla_violations.png", "{:.2f}%")
bar("underprovisioned_steps_pct", "Under-provisioned intervals", "% of 5-min intervals",
    "fig7_underprovisioned.png", "{:.2f}%")
bar("avg_response_s", "Average response time", "Seconds", "fig8_response_time.png", "{:.2f}")

# Cost vs SLA trade-off
fig, ax = plt.subplots(figsize=(7, 5))
for p in ORDER:
    ax.scatter(s.loc[p, "vm_hours"], s.loc[p, "sla_violation_pct"], s=160,
               color=COLOR[p], label=LABEL[p], edgecolor="black", zorder=3)
ax.set_xlabel("Active VM-hours (cost)")
ax.set_ylabel("SLA violations (%)")
ax.set_title("Cost vs SLA trade-off (bottom-left is best)")
ax.grid(alpha=0.3)
ax.legend()
plt.tight_layout()
plt.savefig(os.path.join(FIG, "fig9_cost_vs_sla.png"), dpi=200)
plt.close()

# Scaling behaviour over time: demand vs provisioned capacity (first 2 days)
k = 288 * 2
fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True)
for ax, p in zip(axes, ["REACTIVE", "PREDICTIVE_LSTM"]):
    t = pd.read_csv(os.path.join(RES, f"timeline_{p}.csv")).iloc[:k]
    ax.plot(t["step"], t["actual_requests"], color="black", lw=1, label="Actual demand (requests)")
    ax.step(t["step"], t["ready_capacity"], color=COLOR[p], lw=1.4, where="post",
            label="Ready capacity (requests)")
    ax.fill_between(t["step"], t["actual_requests"], t["ready_capacity"],
                    where=t["actual_requests"] > t["ready_capacity"], color="red", alpha=0.35,
                    step="post", label="Under-provisioned")
    ax.set_title(f"{LABEL[p]}: demand vs provisioned capacity")
    ax.set_ylabel("Requests / 5 min")
    ax.grid(alpha=0.3)
    ax.legend(loc="upper right")
axes[-1].set_xlabel("Time step (5 min)")
plt.tight_layout()
plt.savefig(os.path.join(FIG, "fig10_scaling_timeline.png"), dpi=200)
plt.close()

print(s.round(2).to_string())
print(f"\nSaved figures to {FIG}/")
