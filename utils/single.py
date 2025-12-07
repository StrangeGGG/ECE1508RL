import matplotlib.pyplot as plt
import numpy as np

# ----- Your data -----
# Plan for 4 delays (4 slots on x-axis)
delays = [0, 10, 30, 45]    # 4 positions

# Only first result is real, others are 0 for now
waiting_times = [86.9, 0, 0, 0]
throughputs   = [0.169, 0, 0, 0]

# ----- Bar positions -----
x = np.arange(len(delays))   # [0, 1, 2, 3]
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

# X-axis labels: first has text, others are blank for now
ax1.set_xticks(x)
ax1.set_xticklabels([
    "vanilla trained result",  # slot 1
    "",                        # slot 2 (empty)
    "",                        # slot 3 (empty)
    ""                         # slot 4 (empty)
])

# ----- Add value labels on top of bars -----
ax1.bar_label(bars_wait, fmt="%.1f", padding=3)
ax2.bar_label(bars_through, fmt="%.3f", padding=3)

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

fig.tight_layout(rect=[0, 0, 1, 0.95])
plt.show()