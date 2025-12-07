import matplotlib.pyplot as plt
import numpy as np

# ----- Your data -----
delays = [0, 10, 30, 45, 60]  # signal delays
waiting_times = [86.9, 66.5, 53.9, 67, 95.972]   # replace with your real data
throughputs   = [0.169, 0.193, 0.203, 0.202, 0.182]    # replace with your real data

# ----- Bar positions -----
x = np.arange(len(delays))          # [0, 1, 2, 3]
bar_width = 0.35

fig, ax1 = plt.subplots(figsize=(10, 4))

# Waiting time bars (left axis)
bars_wait = ax1.bar(
    x - bar_width / 2,
    waiting_times,
    bar_width,
    label="Waiting Time"
)
ax1.set_ylabel("Waiting Time (steps)")

# Throughput bars (right axis), shifted to the right and colored orange
ax2 = ax1.twinx()
bars_through = ax2.bar(
    x + bar_width / 2,
    throughputs,
    bar_width,
    label="Throughput",
    color="orange"
)
ax2.set_ylabel("Throughput (vehicles/step)")

# X-axis labels
ax1.set_xticks(x)
#ax1.set_xticklabels([f"Delay {d}" for d in delays])
ax1.set_xticklabels(["delay 0 steps","delay 10 steps", "delay 30 steps", "delay 45 steps","delay 60 steps"])
# ----- Add value labels on top of bars -----
# Waiting time labels
ax1.bar_label(bars_wait, fmt="%.1f", padding=3)

# Throughput labels
ax2.bar_label(bars_through, fmt="%.2f", padding=3)

# ----- Combined legend placed above the plot -----
handles1, labels1 = ax1.get_legend_handles_labels()
handles2, labels2 = ax2.get_legend_handles_labels()

fig.legend(
    handles1 + handles2,
    labels1 + labels2,
    loc="upper center",
    bbox_to_anchor=(0.5, 1.08),
    ncol=2
)

# Adjust layout so legend/title fit nicely
fig.tight_layout(rect=[0, 0, 1, 0.95])

plt.show()