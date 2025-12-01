import numpy as np
import torch
import matplotlib.pyplot as plt
import os
from env import TrafficSimulation
from ppo_agent import PPOAgent, PPOTrafficLightController


def evaluate_ppo(
        model_path="results/ppo/ppo_model.pth",
        max_steps=400,
        phase_duration=12,
        plot=True,
        save_png=True):

    sim = TrafficSimulation()

    state_dim = 13
    action_dim = 4

    agent = PPOAgent(state_dim=state_dim, action_dim=action_dim)
    agent.policy.load_state_dict(torch.load(model_path, map_location="cpu"))
    agent.policy.eval()

    sim.reset()
    metrics = sim.get_metrics()
    state = PPOTrafficLightController(sim).build_state(metrics)

    avg_wait_list = []
    queue_sum_list = []
    passed_list = []

    steps = 0
    total_passed = 0

    for _ in range(max_steps):

        with torch.no_grad():
            action, _, _ = agent.select_action(state)

        for _ in range(phase_duration):

            _, metrics = sim.step(action)

            avg_wait_list.append(metrics["average_waiting_time"])
            queue_sum_list.append(metrics["current_queue_sum"])
            passed_list.append(metrics["passed_this_step"])

            total_passed = metrics["total_vehicles_passed"]
            steps += 1

            state = PPOTrafficLightController(sim).build_state(metrics)

    # ===== SUMMARY METRICS =====
    avg_wait = np.mean(avg_wait_list)
    avg_queue_sum = np.mean(queue_sum_list)
    avg_throughput = total_passed / steps if steps > 0 else 0

    print("\n========== PPO Evaluation Summary ==========")
    print(f"{'Average waiting time':<30} {avg_wait:.2f}")
    print(f"{'Average queue length sum':<30} {avg_queue_sum:.2f}")
    print(f"{'Throughput (veh/step)':<30} {avg_throughput:.4f}")
    print("============================================\n")

    # ===== PLOTTING & PNG SAVING =====
    if plot or save_png:

        os.makedirs("results/ppo", exist_ok=True)

        # 1) Avg wait
        plt.figure(figsize=(10, 4))
        plt.plot(avg_wait_list)
        plt.title("PPO – Average Waiting Time per Step")
        if save_png:
            plt.savefig("results/ppo/ppo_eval_wait.png", dpi=150)
        if plot:
            plt.show()
        plt.close()

        # 2) Queue sum
        plt.figure(figsize=(10, 4))
        plt.plot(queue_sum_list)
        plt.title("PPO – Queue Length Sum per Step")
        if save_png:
            plt.savefig("results/ppo/ppo_eval_queue.png", dpi=150)
        if plot:
            plt.show()
        plt.close()

        # 3) Passed per step
        plt.figure(figsize=(10, 4))
        plt.plot(passed_list)
        plt.title("PPO – Vehicles Passed Per Step")
        if save_png:
            plt.savefig("results/ppo/ppo_eval_passed.png", dpi=150)
        if plot:
            plt.show()
        plt.close()

        # 4) Combined full plot
        plt.figure(figsize=(18, 7))

        plt.subplot(3, 1, 1)
        plt.plot(avg_wait_list)
        plt.title("Average Waiting Time")

        plt.subplot(3, 1, 2)
        plt.plot(queue_sum_list)
        plt.title("Queue Length Sum")

        plt.subplot(3, 1, 3)
        plt.plot(passed_list)
        plt.title("Vehicles Passed Per Step")

        plt.tight_layout()

        if save_png:
            plt.savefig("results/ppo/ppo_eval_full.png", dpi=150)

        if plot:
            plt.show()

        plt.close()

    return avg_wait, avg_queue_sum, avg_throughput


if __name__ == "__main__":
    evaluate_ppo(plot=True, save_png=True)
