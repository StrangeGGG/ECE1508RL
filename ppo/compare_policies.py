import numpy as np
import torch
import matplotlib.pyplot as plt

from env import TrafficSimulation
from ppo_agent import PPOAgent, PPOTrafficLightController


# ------------------------------------------------------------
# Run PPO policy once
# ------------------------------------------------------------
def run_ppo(model_path, max_steps=4000, phase_duration=12):
    sim = TrafficSimulation()

    state_dim = 13
    action_dim = 4

    agent = PPOAgent(state_dim=state_dim, action_dim=action_dim)
    agent.policy.load_state_dict(torch.load(model_path, map_location="cpu"))
    agent.policy.eval()

    sim.reset()
    metrics = sim.get_metrics()
    state = PPOTrafficLightController(sim).build_state(metrics)

    avg_wait = []
    qsum = []
    passed = []

    for _ in range(max_steps):
        with torch.no_grad():
            action, _, _ = agent.select_action(state)

        for _ in range(phase_duration):
            _, metrics = sim.step(action)
            avg_wait.append(metrics["average_waiting_time"])
            qsum.append(metrics["current_queue_sum"])
            passed.append(metrics["passed_this_step"])
            state = PPOTrafficLightController(sim).build_state(metrics)

    return avg_wait, qsum, passed


# ------------------------------------------------------------
# Run Round-Robin baseline once
# ------------------------------------------------------------
def run_round_robin(max_steps=4000, phase_duration=12):
    sim = TrafficSimulation()

    phase = 0
    avg_wait = []
    qsum = []
    passed = []

    sim.reset()

    for _ in range(max_steps):
        for _ in range(phase_duration):
            _, metrics = sim.step(phase)
            avg_wait.append(metrics["average_waiting_time"])
            qsum.append(metrics["current_queue_sum"])
            passed.append(metrics["passed_this_step"])

        phase = (phase + 1) % 4   # rotate 0→1→2→3→0

    return avg_wait, qsum, passed


# ------------------------------------------------------------
# MAIN: run both and plot
# ------------------------------------------------------------
if __name__ == "__main__":
    print("Running PPO evaluation...")
    ppo_wait, ppo_qsum, ppo_pass = run_ppo("results/ppo/ppo_model.pth")

    print("Running Round-Robin baseline...")
    rr_wait, rr_qsum, rr_pass = run_round_robin()

    # ---------------------- PLOTS --------------------------
    plt.figure(figsize=(16, 10))

    # Avg wait
    plt.subplot(3, 1, 1)
    plt.plot(ppo_wait, label="PPO")
    plt.plot(rr_wait, label="Round Robin", alpha=0.7)
    plt.title("Average Waiting Time")
    plt.legend()

    # Queue sum
    plt.subplot(3, 1, 2)
    plt.plot(ppo_qsum, label="PPO")
    plt.plot(rr_qsum, label="Round Robin", alpha=0.7)
    plt.title("Queue Sum")
    plt.legend()

    # Passed each step
    plt.subplot(3, 1, 3)
    plt.plot(ppo_pass, label="PPO")
    plt.plot(rr_pass, label="Round Robin", alpha=0.7)
    plt.title("Vehicles Passed Each Step")
    plt.legend()

    plt.tight_layout()
    plt.show()
