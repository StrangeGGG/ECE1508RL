import matplotlib.pyplot as plt
import numpy as np

# ----- Your data -----
delays = [1, 2, 3, 4]  
waiting_times = [86.9, 104, 68.1, 86]   
throughputs   = [0.169, 0.189, 0.157, 0.189]  

# ----- Bar positions -----
x = np.arange(len(delays))          # [0, 1, 2, 3]
bar_width = 0.35

fig, ax1 = plt.subplots(figsize=(10, 4))

# Waiting time bars (left axis) – BLUE
bars_wait = ax1.bar(
    x - bar_width / 2,
    waiting_times,
    bar_width,
    label="Waiting Time",      # used in legend
    color="tab:blue"
)
ax1.set_ylabel("Waiting Time (steps)")

# Throughput bars (right axis) – ORANGE
ax2 = ax1.twinx()
bars_through = ax2.bar(
    x + bar_width / 2,
    throughputs,
    bar_width,
    label="Throughput",        # used in legend
    color="tab:orange"
)
ax2.set_ylabel("Throughput (vehicles/step)")

# X-axis labels
ax1.set_xticks(x)
ax1.set_xticklabels([
    "origin PPO",
    "max queue PPO",
    "max waiting time PPO",
    "max throughput PPO"
])

# ----- Add value labels on top of bars -----
ax1.bar_label(bars_wait, fmt="%.1f", padding=3)
ax2.bar_label(bars_through, fmt="%.3f", padding=3)

# ----- Legend with colored bars -----
# collect handles from both axes and show as one legend
handles1, labels1 = ax1.get_legend_handles_labels()
handles2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(
    handles1 + handles2,
    labels1 + labels2,
    loc="upper left"   # change position if you like
)

fig.tight_layout()
plt.show()