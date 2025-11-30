import os
import numpy as np
import matplotlib.pyplot as plt
import random
import torch

from traffic_env import TrafficEngine_ideal, TrafficEngine_realistic
from dqn_agent import DQNAgent

# reproducibility
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)


def train(episodes=300, max_steps=800, save_model_path="traffic_dqn.pth"):
    # Switching environment  
    #env = TrafficEngine_ideal(seed=SEED)
    env = TrafficEngine_realistic(peak_steps=1000, seed=SEED, sensor_noise_std=0.5)
    state = env.reset()
    state_size = len(state)
    action_size = env.action_size

    agent = DQNAgent(state_size, action_size)

    scores = []
    avg_scores = []
    episode_throughputs = []
    episode_waits = []
    losses = []

    # Warm-up steps before learning starts
    warmup_steps = 10000

    for ep in range(1, episodes + 1):

        if ep == 300:  # decrease learning rate
            for param_group in agent.optimizer.param_groups:
                param_group['lr'] = 5e-5

        # reset exploration
        if ep % 150 == 0 and agent.eps < 0.3:
            agent.eps = 0.3
            print(f"Ep {ep}: Reset exploration to 0.3")


        state = env.reset()
        total_reward = 0.0
        tput_list = []
        wait_list = []
        ep_losses = []

        for step in range(max_steps):
            # Agent picks action
            action = agent.act(state, training=True)

            # Interact with environment
            next_state, reward, done, info = env.step(action)

            agent.global_step += 1

            # Warm-up: collect buffer but do NOT train
            if agent.global_step <= warmup_steps:
                agent.replay.push(state, action, reward, next_state, done)
                loss = None
            else:
                loss = agent.step(state, action, reward, next_state, done)

            # Log losses
            if loss is not None:
                ep_losses.append(loss)
                losses.append(loss)

            total_reward += reward
            state = next_state

            # Metrics for logging
            tput_list.append(info.get("throughput", 0.0))
            wait_list.append(info.get("avg_wait", 0.0))

        # per-episode logs
        scores.append(total_reward)
        avg_scores.append(np.mean(scores[-100:]))
        episode_throughputs.append(np.mean(tput_list) if tput_list else 0)
        episode_waits.append(np.mean(wait_list) if wait_list else 0)

        if ep % 10 == 0 or ep == 1:
            print(
                f"Ep {ep}/{episodes} | Score {total_reward:.2f} | "
                f"AvgScore {avg_scores[-1]:.2f} | "
                f"Tput {episode_throughputs[-1]:.4f} | "
                f"Wait {episode_waits[-1]:.1f} | "
                f"Eps {agent.eps:.3f} | "
                f"Loss(avg) {np.mean(ep_losses) if ep_losses else 0:.4f}"
            )

    # === Save trained model ===
    agent.save(save_model_path)
    print(f"Model saved to {save_model_path}")

    # === Plot training curves ===
    plt.figure(figsize=(12, 6))
    plt.subplot(2, 2, 1)
    plt.plot(scores)
    plt.title("Episode Scores")

    plt.subplot(2, 2, 2)
    plt.plot(avg_scores)
    plt.title("Avg Score (last 100)")

    plt.subplot(2, 2, 3)
    plt.plot(episode_throughputs)
    plt.title("Episode Throughput")

    plt.subplot(2, 2, 4)
    plt.plot(episode_waits)
    plt.title("Episode Waiting Time")

    plt.tight_layout()
    plt.savefig("training_summary.png", dpi=200)
    plt.show()

    # Save history
    np.savez(
        "training_history.npz",
        scores=scores,
        avg_scores=avg_scores,
        tput=episode_throughputs,
        wait=episode_waits,
        losses=losses,
    )

    return agent, env, (scores, avg_scores, episode_throughputs, episode_waits, losses)


def test(agent, env, max_steps=1000):
    state = env.reset()
    agent.eps = agent.eps_end  # minimal exploration during testing

    thr = []
    waits = []
    phase_changes = []

    prev_phase = env.current_phase

    for t in range(max_steps):
        action = agent.act(state, training=False)
        next_state, reward, done, info = env.step(action)

        # count phase changes
        if info.get("phase") != prev_phase:
            phase_changes.append(t)
        prev_phase = info.get("phase")

        state = next_state

        thr.append(info.get("throughput", 0.0))
        waits.append(info.get("avg_wait", 0.0))

    # ============================
    #  Last 30% statistics
    # ============================
    n = len(thr)
    tail_len = max(1, int(n * 0.3))

    tail_thr = thr[-tail_len:]
    tail_wait = waits[-tail_len:]

    mean_tail_thr = float(np.mean(tail_thr))
    mean_tail_wait = float(np.mean(tail_wait))

    print("=== Test Performance Summary ===")
    print(f"Total Steps: {n}")
    print(f"Last 30% Steps: {tail_len}")
    print(f"Overall avg throughput: {np.mean(thr):.4f}")
    print(f"Overall avg wait:       {np.mean(waits):.2f}")
    print("--- Last 30% performance ---")
    print(f"Tail avg throughput:    {mean_tail_thr:.4f}")
    print(f"Tail avg wait:          {mean_tail_wait:.2f}")
    print(f"Phase changes:          {len(phase_changes)}")
    print("================================")

    # plot
    plt.figure(figsize=(10, 4))
    plt.subplot(2, 1, 1)
    plt.plot(thr)
    plt.title("Throughput")

    plt.subplot(2, 1, 2)
    plt.plot(waits)
    plt.title("Average Waiting Time")

    plt.tight_layout()
    plt.savefig("test_results.png", dpi=200)
    plt.show()

    # return stats for external comparison
    tail_stats = {
        "tail_thr": mean_tail_thr,
        "tail_wait": mean_tail_wait,
        "tail_len": tail_len,
        "overall_thr": float(np.mean(thr)),
        "overall_wait": float(np.mean(waits)),
        "phase_changes": len(phase_changes),
    }

    return thr, waits, phase_changes, tail_stats

if __name__ == "__main__":
    agent, env, _ = train(episodes=500, max_steps=1000, save_model_path="traffic_dqn.pth")
    test(agent, env, max_steps=2000)
