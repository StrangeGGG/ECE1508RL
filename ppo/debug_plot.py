import matplotlib.pyplot as plt
import numpy as np

def plot_episode_metrics(episode_metrics):
    """
    episode_metrics must be a dict like:
    {
        "avg_wait": [...],
        "queue_sum": [...],
        "passed": [...],
    }
    collected during a single PPO episode.
    """

    steps = np.arange(len(episode_metrics["avg_wait"]))

    fig, axs = plt.subplots(3, 1, figsize=(10, 10))
    fig.suptitle("Traffic Simulation Debug Plot (per episode)")

    # 1. Average waiting time
    axs[0].plot(steps, episode_metrics["avg_wait"], label="Avg Wait")
    axs[0].set_ylabel("Avg Wait")
    axs[0].legend()
    axs[0].grid(True)

    # 2. Queue sum
    axs[1].plot(steps, episode_metrics["queue_sum"], color='orange', label="Queue Sum")
    axs[1].set_ylabel("Queue Sum")
    axs[1].legend()
    axs[1].grid(True)

    # 3. Passed vehicles this step
    axs[2].plot(steps, episode_metrics["passed"], color='green', label="Passed This Step")
    axs[2].set_ylabel("Passed")
    axs[2].set_xlabel("Step")
    axs[2].legend()
    axs[2].grid(True)

    plt.tight_layout()
    plt.show()
