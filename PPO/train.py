import numpy as np
import torch
import matplotlib.pyplot as plt

from traffic_env import TrafficRLWrapper
from agent import PPOAgent


def train_ppo_on_traffic(
    num_episodes: int = 200,
    max_steps_per_episode: int = 1000,
    steps_per_batch: int = 5000,
    min_phase_steps: int = 0,
    max_phase_steps: int = 5,
    seed: int = 42,
    model_path: str = "ppo_traffic.pth",
):
    # -----------------------
    # Reproducibility
    # -----------------------
    np.random.seed(seed)
    torch.manual_seed(seed)

    # -----------------------
    # Environment & Agent
    # -----------------------
    env = TrafficRLWrapper(
        min_phase_steps=min_phase_steps,
        max_phase_steps=max_phase_steps,
    )

    state = env.reset()
    state_dim = state.shape[0]
    action_dim = env.action_size  # 4 actions: NS left, EW left, NS straight, EW straight

    agent = PPOAgent(
        state_size=state_dim,
        action_size=action_dim,
        gamma=0.99,
        lam=0.95,
        lr=3e-4,
        clip_ratio=0.2,
        update_epochs=10,
        minibatch_size=64,
        reward_scale=1.0,
        entropy_coef=0.01,
        value_coef=0.5,
        max_grad_norm=0.5,
        use_state_norm=True,
        hidden_sizes=(128, 128),
    )

    # Tracking stats
    total_steps = 0
    episode_returns = []
    episode_mean_wait = []
    episode_mean_throughput = []

    print("Starting PPO training on TrafficRLWrapper...")
    print(f"State dim = {state_dim}, action dim = {action_dim}")
    print(f"num_episodes = {num_episodes}, max_steps_per_episode = {max_steps_per_episode}")
    print(f"steps_per_batch = {steps_per_batch}\n")

    for ep in range(num_episodes):
        state = env.reset()
        ep_return = 0.0

        # We'll average wait/throughput over the episode manually
        wait_accum = 0.0
        throughput_accum = 0.0
        wait_count = 0
        throughput_count = 0

        for t in range(max_steps_per_episode):
            # Select action from current policy
            action = agent.select_action(state)  # int in [0, 3]

            next_state, reward, done_env, info = env.step(action)

            # We treat episodes as fixed-horizon; env.done is always False
            done = done_env or (t == max_steps_per_episode - 1)

            # Store reward and done for PPO (for GAE)
            agent.store_reward(reward, done)

            ep_return += reward
            total_steps += 1

            # Track metrics from `info`
            if "average_waiting_time" in info:
                wait_accum += info["average_waiting_time"]
                wait_count += 1
            if "throughput" in info:
                throughput_accum += info["throughput"]
                throughput_count += 1

            state = next_state

            # When we have enough steps, run a PPO update
            if agent.buffer_size() >= steps_per_batch:
                agent.update()

            if done:
                break

        # Episode-level stats
        episode_returns.append(ep_return)
        mean_wait = wait_accum / wait_count if wait_count > 0 else 0.0
        mean_throughput = (
            throughput_accum / throughput_count if throughput_count > 0 else 0.0
        )
        episode_mean_wait.append(mean_wait)
        episode_mean_throughput.append(mean_throughput)

        print(
            f"Episode {ep + 1:4d}/{num_episodes:4d} | "
            f"steps={total_steps:6d} | "
            f"return={ep_return:8.3f} | "
            f"mean_wait={mean_wait:8.3f} | "
            f"mean_throughput={mean_throughput:8.3f}"
        )

    # Final update on any leftover data
    if agent.buffer_size() > 0:
        agent.update()

    # Save trained model
    agent.save(model_path)
    print(f"\nTraining finished. Model saved to {model_path}")

    # -----------------------
    # Save metrics as .npy
    # -----------------------
    np.save("episode_returns.npy", np.array(episode_returns, dtype=np.float32))
    np.save("episode_mean_wait.npy", np.array(episode_mean_wait, dtype=np.float32))
    np.save(
        "episode_mean_throughput.npy",
        np.array(episode_mean_throughput, dtype=np.float32),
    )
    print("Saved metrics: episode_returns.npy, episode_mean_wait.npy, episode_mean_throughput.npy")

    # -----------------------
    # Plot training curves
    # -----------------------
    episodes = np.arange(1, num_episodes + 1)

    # 1) Return curve
    plt.figure()
    plt.plot(episodes, episode_returns)
    plt.xlabel("Episode")
    plt.ylabel("Return")
    plt.title("PPO Training: Episode Return")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("ppo_return_curve.png")

    # 2) Mean waiting time
    plt.figure()
    plt.plot(episodes, episode_mean_wait)
    plt.xlabel("Episode")
    plt.ylabel("Average waiting time")
    plt.title("PPO Training: Mean Waiting Time per Episode")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("ppo_waiting_time_curve.png")

    # 3) Mean throughput
    plt.figure()
    plt.plot(episodes, episode_mean_throughput)
    plt.xlabel("Episode")
    plt.ylabel("Throughput (vehicles / step)")
    plt.title("PPO Training: Mean Throughput per Episode")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("ppo_throughput_curve.png")

    print("Saved plots: ppo_return_curve.png, ppo_waiting_time_curve.png, ppo_throughput_curve.png")

    return {
        "episode_returns": episode_returns,
        "episode_mean_wait": episode_mean_wait,
        "episode_mean_throughput": episode_mean_throughput,
    }


if __name__ == "__main__":
    train_ppo_on_traffic()