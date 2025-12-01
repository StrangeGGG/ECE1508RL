# ================================================================
# FULL UPDATED agent.py WITH THROUGHPUT & WAIT-TIME GRAPHS
# ================================================================
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt

from env import TrafficSimulation  # your environment
import random

class RunningNormalizer:
    """
    Tracks running mean/var for state normalization.
    """

    def __init__(self, state_size):
        self.state_size = state_size
        self.mean = np.zeros(state_size, dtype=np.float32)
        self.var = np.ones(state_size, dtype=np.float32)
        self.count = 1e-4  # avoid div-by-zero

    def update(self, x: np.ndarray):
        assert x.shape[0] == self.state_size
        self.count += 1.0
        delta = x - self.mean
        self.mean += delta / self.count
        delta2 = x - self.mean
        self.var = ((self.count - 1) * self.var + delta * delta2) / self.count

    def normalize(self, x: np.ndarray) -> np.ndarray:
        return (x - self.mean) / (np.sqrt(self.var) + 1e-8)

    def __call__(self, x: np.ndarray) -> np.ndarray:
        self.update(x)
        return self.normalize(x)

class PPOActorCritic(nn.Module):
    def __init__(self, state_size, action_size, hidden_size=128):
        super().__init__()

        self.fc1 = nn.Linear(state_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, hidden_size)

        self.policy_head = nn.Linear(hidden_size, action_size)
        self.value_head = nn.Linear(hidden_size, 1)

    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))

        logits = self.policy_head(x)
        value = self.value_head(x)
        return logits, value

class PPOAgent:
    def __init__(
        self,
        state_size,
        action_size,
        gamma=0.99,
        lam=0.95,
        lr=1e-3,
        clip_ratio=0.1,
        update_epochs=5,
        minibatch_size=64,
        reward_scale=1e-3,
        entropy_coef=0.05,
        max_grad_norm=0.5,
    ):
        self.state_size = state_size
        self.action_size = action_size

        self.gamma = gamma
        self.lam = lam
        self.clip_ratio = clip_ratio
        self.update_epochs = update_epochs
        self.minibatch_size = minibatch_size
        self.reward_scale = reward_scale
        self.entropy_coef = entropy_coef
        self.max_grad_norm = max_grad_norm

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.ac = PPOActorCritic(state_size, action_size).to(self.device)
        self.optimizer = optim.Adam(self.ac.parameters(), lr=lr)

        self.states = []
        self.actions = []
        self.log_probs = []
        self.values = []
        self.rewards = []
        self.dones = []

        self.losses = []

        self.state_normalizer = RunningNormalizer(state_size)

    def select_action(self, state_np: np.ndarray) -> int:
        norm_state = self.state_normalizer(state_np)
        state_tensor = torch.from_numpy(norm_state).float().to(self.device).unsqueeze(0)

        logits, value = self.ac(state_tensor)

        probs = F.softmax(logits, dim=-1)
        dist = torch.distributions.Categorical(probs)
        action = dist.sample()

        self.states.append(norm_state.copy())
        self.actions.append(action.item())
        self.log_probs.append(dist.log_prob(action).item())
        self.values.append(value.item())

        return action.item()

    def store_reward(self, reward, done):
        self.rewards.append(reward * self.reward_scale)
        self.dones.append(done)

    def compute_gae(self):
        advantages = []
        gae = 0.0
        values = self.values + [0.0]

        for t in reversed(range(len(self.rewards))):
            delta = (
                self.rewards[t]
                + self.gamma * values[t + 1] * (1 - self.dones[t])
                - values[t]
            )
            gae = delta + self.gamma * self.lam * (1 - self.dones[t]) * gae
            advantages.insert(0, gae)

        returns = [adv + val for adv, val in zip(advantages, self.values)]
        advantages = np.array(advantages, dtype=np.float32)
        returns = np.array(returns, dtype=np.float32)

        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        return advantages, returns

    def update(self):
        if len(self.states) == 0:
            return

        advantages, returns = self.compute_gae()

        states_np = np.array(self.states, dtype=np.float32)
        actions_np = np.array(self.actions, dtype=np.int64)
        old_log_probs_np = np.array(self.log_probs, dtype=np.float32)

        states = torch.from_numpy(states_np).to(self.device)
        actions = torch.from_numpy(actions_np).to(self.device)
        old_log_probs = torch.from_numpy(old_log_probs_np).to(self.device)
        returns = torch.from_numpy(returns).to(self.device)
        advantages = torch.from_numpy(advantages).to(self.device)

        dataset_size = len(states)

        for _ in range(self.update_epochs):
            idx = np.random.permutation(dataset_size)
            for start in range(0, dataset_size, self.minibatch_size):
                end = start + self.minibatch_size
                batch_idx = idx[start:end]

                batch_states = states[batch_idx]
                batch_actions = actions[batch_idx]
                batch_old_log_probs = old_log_probs[batch_idx]
                batch_returns = returns[batch_idx]
                batch_advantages = advantages[batch_idx]

                logits, values = self.ac(batch_states)
                dist = torch.distributions.Categorical(F.softmax(logits, dim=-1))
                new_log_probs = dist.log_prob(batch_actions)

                ratio = torch.exp(new_log_probs - batch_old_log_probs)

                surr1 = ratio * batch_advantages
                surr2 = torch.clamp(
                    ratio, 1.0 - self.clip_ratio, 1.0 + self.clip_ratio
                ) * batch_advantages
                policy_loss = -torch.min(surr1, surr2).mean()

                value_pred = values.squeeze(-1)
                value_loss = 0.5 * (value_pred - batch_returns).pow(2).clamp(max=10.0).mean()

                entropy = dist.entropy().mean()
                loss = policy_loss + value_loss - self.entropy_coef * entropy

                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.ac.parameters(), self.max_grad_norm)
                self.optimizer.step()

                self.losses.append(loss.item())

        self.states.clear()
        self.actions.clear()
        self.log_probs.clear()
        self.values.clear()
        self.rewards.clear()
        self.dones.clear()

    def buffer_size(self) -> int:
        return len(self.rewards)

    def save(self, path: str):
        torch.save(self.ac.state_dict(), path)

    def load(self, path: str):
        state_dict = torch.load(path, map_location=self.device)
        self.ac.load_state_dict(state_dict)


class SmartTrafficLightControllerPPO:

    def __init__(self, simulation: TrafficSimulation):
        self.simulation = simulation
        self.state_size = 24
        self.action_size = 4

        self.agent = PPOAgent(self.state_size, self.action_size)

        self.current_phase = 0
        self.phase_duration = 0
        self.min_phase_duration = 10
        self.max_phase_duration = 60

        self.scores = []
        self.avg_scores = []

        # NEW METRIC LOGGING ARRAYS
        self.throughput_history = []
        self.waiting_time_history = []

    def get_state(self) -> np.ndarray:
        metrics = self.simulation.metrics_collector.get_metrics()
        queue_lengths = metrics["queue_lengths"]

        state = []

        for direction in ["north", "south", "east", "west"]:
            for lane_type in ["straight", "left"]:
                key = f"{direction}_{lane_type}"

                queue_len = queue_lengths[key]

                lane_vehicles = [
                    v
                    for v in self.simulation.vehicles.values()
                    if v.direction == direction
                    and v.lane_type == lane_type
                    and not v.passed
                    and not v.collided
                ]

                if lane_vehicles:
                    avg_wait = np.mean([v.waiting_time for v in lane_vehicles])
                    vehicle_count = len(lane_vehicles)
                else:
                    avg_wait = 0.0
                    vehicle_count = 0

                state.extend([queue_len, avg_wait, vehicle_count])

        return np.array(state, dtype=np.float32)

    def calculate_reward(self, metrics, old_metrics=None) -> float:
        reward = 0.0
        reward += metrics["throughput"] * 10.0
        reward -= metrics["average_waiting_time"] * 0.1

        total_queue = sum(metrics["queue_lengths"].values())
        reward -= total_queue * 0.2

        if old_metrics is not None and self.phase_duration == 1:
            reward -= 0.1

        if self.phase_duration >= self.min_phase_duration:
            old_total_passed = old_metrics["total_vehicles_passed"] if old_metrics else 0
            vehicles_passed = metrics["total_vehicles_passed"] - old_total_passed
            reward += vehicles_passed * 2.0

        return reward

    def train(self, episodes=200, max_steps=1000, steps_per_batch=4096):
        print("Starting PPO training with batch updates...")

        for ep in range(episodes):
            self.simulation.reset()
            self.current_phase = 0
            self.phase_duration = 0

            state = self.get_state()
            total_reward = 0.0
            old_metrics = None

            # METRICS FOR THIS EPISODE
            ep_through_list = []
            ep_wait_list = []

            for step in range(max_steps):
                action = self.agent.select_action(state)

                if self.phase_duration >= self.min_phase_duration:
                    if action != self.current_phase:
                        self.current_phase = action
                        self.phase_duration = 0

                _, metrics = self.simulation.step(self.current_phase)
                self.phase_duration += 1

                if self.phase_duration >= self.max_phase_duration:
                    self.current_phase = (self.current_phase + 1) % self.action_size
                    self.phase_duration = 0

                reward = self.calculate_reward(metrics, old_metrics)
                total_reward += reward

                # ---- COLLECT METRICS ----
                ep_through_list.append(metrics["throughput"])
                ep_wait_list.append(metrics["average_waiting_time"])

                next_state = self.get_state()
                done = (step == max_steps - 1)

                self.agent.store_reward(reward, done)

                state = next_state
                old_metrics = metrics.copy()

                if done:
                    break

                if self.agent.buffer_size() >= steps_per_batch:
                    self.agent.update()

            # End of episode logging
            self.scores.append(total_reward)
            avg_reward = np.mean(self.scores[-50:])
            self.avg_scores.append(avg_reward)

            # FINAL METRIC AGGREGATION
            mean_throughput = np.mean(ep_through_list)
            mean_wait = np.mean(ep_wait_list)
            self.throughput_history.append(mean_throughput)
            self.waiting_time_history.append(mean_wait)

            print(
                f"[Episode {ep}] Reward={total_reward:.2f} | Avg(50)={avg_reward:.2f} | "
                f"Through={mean_throughput:.2f} | Wait={mean_wait:.2f} | Buffer={self.agent.buffer_size()}"
            )

        if self.agent.buffer_size() > 0:
            print("Final PPO update on remaining batch...")
            self.agent.update()

        return self.scores, self.avg_scores, self.throughput_history, self.waiting_time_history


def main():
    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)

    simulation = TrafficSimulation()
    controller = SmartTrafficLightControllerPPO(simulation)

    print("Training PPO traffic-light controller (batch PPO)...")
    scores, avg_scores, through, wait = controller.train(
        episodes=200,
        max_steps=1000,
        steps_per_batch=4096,
    )

    # -----------------------------
    # PLOT TRAINING CURVES
    # -----------------------------
    plt.figure(figsize=(10, 4))
    plt.plot(scores, label="Episode Reward", alpha=0.6)
    plt.plot(avg_scores, label="Avg Reward (50 episodes)", linewidth=2)
    plt.xlabel("Episode")
    plt.ylabel("Total Reward")
    plt.title("PPO Training Curve (Batch Updates)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("ppo_training_batch.png", dpi=300)
    plt.show()

    # -----------------------------
    # NEW: THROUGHPUT GRAPH
    # -----------------------------
    plt.figure(figsize=(10, 4))
    plt.plot(through, color="blue", label="Throughput")
    plt.xlabel("Episode")
    plt.ylabel("Vehicles Passed")
    plt.title("Throughput Over Episodes")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("ppo_throughput.png", dpi=300)
    plt.show()

    # -----------------------------
    # NEW: WAITING TIME GRAPH
    # -----------------------------
    plt.figure(figsize=(10, 4))
    plt.plot(wait, color="red", label="Avg Waiting Time")
    plt.xlabel("Episode")
    plt.ylabel("Seconds")
    plt.title("Average Waiting Time Over Episodes")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("ppo_waiting_time.png", dpi=300)
    plt.show()

    controller.agent.save("traffic_ppo_batch_model.pth")
    print("Saved PPO model to 'traffic_ppo_batch_model.pth'")


if __name__ == "__main__":
    main()