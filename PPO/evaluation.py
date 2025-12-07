import random
import numpy as np
import torch

from traffic_env import TrafficRLWrapper
from agent import PPOAgent


def evaluate_ppo_on_traffic(
    model_path: str = r"PPO_result/balance_queue/ppo_traffic.pth",
    num_episodes: int = 10,
    max_steps_per_episode: int = 1000,
    min_phase_steps: int = 0,
    max_phase_steps: int = 60, # doesn't used in the end
    seed: int = 42,
):
    """
    Evaluate a trained PPO model on the TrafficRLWrapper environment.

    We DO NOT update the agent here; we just:
      - load the trained weights,
      - run the policy for several episodes,
      - collect return, mean waiting time, and mean throughput.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    # -----------------------
    # Environment
    # -----------------------
    env = TrafficRLWrapper(
        min_phase_steps=min_phase_steps,
        max_phase_steps=max_phase_steps,
    )

    # Get state/action dimensions from env
    state = env.reset()
    state_dim = state.shape[0]
    action_dim = env.action_size  # 4 actions

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

    # Load trained weights + normalizer
    agent.load(model_path)
    # print(f"Loaded trained PPO model from '{model_path}'")
    # print(f"State dim = {state_dim}, action dim = {action_dim}")
    # print(f"Evaluating for {num_episodes} episodes, {max_steps_per_episode} steps each.\n")

    all_returns = []
    all_mean_wait = []
    all_mean_throughput = []

    for ep in range(num_episodes):
        state = env.reset()
        ep_return = 0.0

        wait_accum = 0.0
        throughput_accum = 0.0
        wait_count = 0
        throughput_count = 0

        for t in range(max_steps_per_episode):
            s = np.asarray(state, dtype=np.float32)
            
            if agent.use_state_norm and agent.state_normalizer is not None:
                norm_s = agent.state_normalizer(s, update=False)
            else:
                norm_s = s

            s_tensor = torch.tensor(
                norm_s, dtype=torch.float32, device=agent.device
            ).unsqueeze(0)

            with torch.no_grad():
                action_int, _ = agent.policy_net.select_action(s_tensor)

            next_state, reward, done_env, info = env.step(action_int)

            ep_return += reward

            if "average_waiting_time" in info:
                wait_accum += info["average_waiting_time"]
                wait_count += 1
            if "throughput" in info:
                throughput_accum += info["throughput"]
                throughput_count += 1

            state = next_state

            if t == max_steps_per_episode - 1:
                break

        mean_wait = wait_accum / wait_count if wait_count > 0 else 0.0
        mean_throughput = (
            throughput_accum / throughput_count if throughput_count > 0 else 0.0
        )

        all_returns.append(ep_return)
        all_mean_wait.append(mean_wait)
        all_mean_throughput.append(mean_throughput)

        print(
            f"[EVAL] Episode {ep + 1:3d}/{num_episodes:3d} | "
            f"return={ep_return:8.3f} | "
            f"mean_wait={mean_wait:8.3f} | "
            f"mean_throughput={mean_throughput:8.3f}"
        )

    return {
        "returns": np.mean(all_returns),
        "mean_wait": np.mean(all_mean_wait),
        "mean_throughput": np.mean(all_mean_throughput),
    }


if __name__ == "__main__":
    result = evaluate_ppo_on_traffic()
    print("\n=== PPO Evaluation Summary ===")
    print(f" Avg return         : {np.mean(result['returns']):.3f}")
    print(f" Avg mean wait      : {np.mean(result['mean_wait']):.3f}")
    print(f" Avg mean throughput: {np.mean(result['mean_throughput']):.3f}")
    
    