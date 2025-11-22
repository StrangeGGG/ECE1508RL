# train.py
import os
import numpy as np
import matplotlib.pyplot as plt
import random
import torch

from traffic_env import TrafficRLWrapper
from dqn_agent import DQNAgent

# reproducibility
SEED = 42
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)

def train(episodes=300, max_steps=800, save_model_path="traffic_dqn.pth"):
    # create environment and agent
    env = TrafficRLWrapper(min_phase_steps=8, max_phase_steps=60)
    state = env.reset()
    state_size = len(state)
    action_size = env.action_size

    agent = DQNAgent(state_size, action_size,
                     lr=1e-4, gamma=0.98, batch_size=128,
                     eps_start=1.0, eps_end=0.1, eps_decay=0.9995,
                     update_every=10)

    scores = []
    avg_scores = []
    episode_throughputs = []
    episode_waits = []
    losses = []

    warmup_steps = 5000

    for ep in range(1, episodes + 1):
        state = env.reset()
        total_reward = 0.0
        tput_list = []
        wait_list = []
        ep_losses = []

        for step in range(max_steps):
            action = agent.act(state, training=True)
            next_state, reward, done, info = env.step(action)

            agent.global_step += 1

            if agent.global_step <= warmup_steps:
                agent.replay.push(state, action, reward, next_state, done)
                loss = None
            else:
                loss = agent.step(state, action, reward, next_state, done)


            # loss = agent.step(state, action, reward, next_state, done)
            if loss is not None:
                ep_losses.append(loss)
                losses.append(loss)

            total_reward += reward
            state = next_state

            # metrics for logging
            metrics = env.last_metrics
            tput_list.append(metrics.get('throughput', 0))
            wait_list.append(metrics.get('average_waiting_time', 0))

        scores.append(total_reward)
        avg_scores.append(np.mean(scores[-100:]))
        episode_throughputs.append(np.mean(tput_list) if tput_list else 0)
        episode_waits.append(np.mean(wait_list) if wait_list else 0)

        if ep % 10 == 0 or ep == 1:
            print(f"Ep {ep}/{episodes} | Score {total_reward:.2f} | AvgScore {avg_scores[-1]:.2f} | "
                  f"Throughput {episode_throughputs[-1]:.4f} | Wait {episode_waits[-1]:.1f} | Eps {agent.eps:.3f} "
                  f"| Loss(avg) {np.mean(ep_losses) if ep_losses else 0:.4f}")

    # save model
    agent.save(save_model_path)
    print(f"Model saved to {save_model_path}")

    # plotting
    plt.figure(figsize=(12, 6))
    plt.subplot(2,2,1)
    plt.plot(scores); plt.title('Episode Scores')
    plt.subplot(2,2,2)
    plt.plot(avg_scores); plt.title('Avg Score (last 100)')
    plt.subplot(2,2,3)
    plt.plot(episode_throughputs); plt.title('Episode Throughput')
    plt.subplot(2,2,4)
    plt.plot(episode_waits); plt.title('Episode Waiting Time')
    plt.tight_layout()
    plt.savefig('training_summary.png', dpi=200)
    plt.show()

    # Save history
    np.savez('training_history.npz', scores=scores, avg_scores=avg_scores,
             tput=episode_throughputs, wait=episode_waits, losses=losses)

    return agent, env, (scores, avg_scores, episode_throughputs, episode_waits, losses)

def test(agent, env, max_steps=1000):
    state = env.reset()
    agent.eps = agent.eps_end  # set low exploration for testing
    thr = []
    waits = []
    phase_changes = []
    prev_phase = env.current_phase

    for t in range(max_steps):
        action = agent.act(state, training=False)
        next_state, reward, done, info = env.step(action)
        if info.get('phase', None) != prev_phase:
            phase_changes.append(t)
        prev_phase = info.get('phase', None)
        state = next_state
        metrics = env.last_metrics
        thr.append(metrics.get('throughput', 0))
        waits.append(metrics.get('average_waiting_time', 0))

    print("Test done: avg_throughput=%.4f avg_wait=%.2f phase_changes=%d" %
          (np.mean(thr), np.mean(waits), len(phase_changes)))

    # quick plots
    plt.figure(figsize=(10,4))
    plt.subplot(1,2,1); plt.plot(thr); plt.title('Throughput')
    plt.subplot(1,2,2); plt.plot(waits); plt.title('Waiting time')
    plt.tight_layout()
    plt.savefig('test_results.png', dpi=200)
    plt.show()

    return thr, waits, phase_changes

if __name__ == "__main__":
    agent, env, _ = train(episodes=400, max_steps=600, save_model_path="traffic_dqn.pth")
    test(agent, env, max_steps=1000)
