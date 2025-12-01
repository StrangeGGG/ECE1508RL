import numpy as np
import matplotlib.pyplot as plt
import os
from env import TrafficSimulation
from ppo_agent import PPOTrafficLightController   # only for state-builder (not used here)


def evaluate_round_robin(
        max_steps=400,
        phase_duration=12,
        plot=True,
        save_png=True):

    sim = TrafficSimulation()
    sim.reset()

    avg_wait_list = []
    queue_sum_list = []
    passed_list = []

    steps = 0
    total_passed = 0

    phase = 0   # round-robin pointer

    for _ in range(max_steps):

        action = phase % 4     # 0,1,2,3 cycle
        phase += 1

        for _ in range(phase_duration):
            _, metrics = sim.step(action)

            avg_wait_list.append(metrics["average_waiting_time"])
            queue_sum_list.append(metrics["current_queue_sum"])
            passed_list.append(metrics["passed_this_step"])

            total_passed = metrics["total_vehicles_passed"]
            steps += 1

    # ===== SUMMARY METRICS =====
    avg_wait = np.mean(avg_wait_list)
    avg_queue_sum = np.mean(queue_sum_list)
    avg_throughput = total_passed / steps if steps > 0 else 0

    print("\n========== Round Robin Evaluation Summary ==========")
    print(f"{'Average waiting time':<30} {avg_wait:.2f}")
    print(f"{'Average queue length sum':<30} {avg_queue_sum:.2f}")
    print(f"{'Throughput (veh/step)':<30} {avg_throughput:.4f}")
    print("====================================================\n")

    # ===== PLOTTING & PNG SAVING =====
    if plot or save_png:

        os.makedirs("results/rr", exist_ok=True)

        # 1) Avg wait
        plt.figure(figsize=(10, 4))
        plt.plot(avg_wait_list)
        plt.title("Round Robin – Average Waiting Time per Step")
        if save_png:
            plt.savefig("results/rr/rr_eval_wait.png", dpi=150)
        if plot:
            plt.show()
        plt.close()

        # 2) Queue length sum
        plt.figure(figsize=(10, 4))
        plt.plot(queue_sum_list)
        plt.title("Round Robin – Queue Length Sum per Step")
        if save_png:
            plt.savefig("results/rr/rr_eval_queue.png", dpi=150)
        if plot:
            plt.show()
        plt.close()

        # 3) Vehicles passed
        plt.figure(figsize=(10, 4))
        plt.plot(passed_list)
        plt.title("Round Robin – Vehicles Passed Per Step")
        if save_png:
            plt.savefig("results/rr/rr_eval_passed.png", dpi=150)
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
            plt.savefig("results/rr/rr_eval_full.png", dpi=150)

        if plot:
            plt.show()

        plt.close()

    return avg_wait, avg_queue_sum, avg_throughput


if __name__ == "__main__":
    evaluate_round_robin(plot=True, save_png=True)
