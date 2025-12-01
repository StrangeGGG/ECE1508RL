# ppo_agent.py

import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class PPOMemory:
    def __init__(self):
        self.states = []
        self.actions = []
        self.logprobs = []
        self.rewards = []
        self.dones = []
        self.values = []

    def clear(self):
        self.states.clear()
        self.actions.clear()
        self.logprobs.clear()
        self.rewards.clear()
        self.dones.clear()
        self.values.clear()


class ActorCritic(nn.Module):
    def __init__(self, state_dim: int, action_dim: int):
        super().__init__()
        hidden = 128

        self.actor = nn.Sequential(
            nn.Linear(state_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, action_dim),
        )

        self.critic = nn.Sequential(
            nn.Linear(state_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, x):
        raise NotImplementedError

    def act(self, state):
        logits = self.actor(state)
        dist = Categorical(logits=logits)
        action = dist.sample()
        logprob = dist.log_prob(action)
        value = self.critic(state)
        return action, logprob, value

    def evaluate(self, states, actions):
        logits = self.actor(states)
        dist = Categorical(logits=logits)

        action_logprobs = dist.log_prob(actions)
        dist_entropy = dist.entropy()
        values = self.critic(states).squeeze(-1)

        return action_logprobs, values, dist_entropy


class PPOAgent:
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        lr: float = 1e-4,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        eps_clip: float = 0.2,
        k_epochs: int = 10,
        entropy_coef: float = 0.01,
    ):
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.eps_clip = eps_clip
        self.k_epochs = k_epochs
        self.entropy_coef = entropy_coef

        self.policy = ActorCritic(state_dim, action_dim).to(device)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=lr)

        self.memory = PPOMemory()

    def select_action(self, state_np: np.ndarray):
        state = torch.tensor(state_np, dtype=torch.float32, device=device).unsqueeze(0)
        with torch.no_grad():
            action, logprob, value = self.policy.act(state)
        # action is categorical: convert to Python int and clip to valid range
        action_int = int(action.item())
        action_int = max(0, min(3, action_int))
        return action_int, float(logprob.item()), float(value.item())

    def store_transition(self, state, action, logprob, reward, done, value):
        self.memory.states.append(state)
        self.memory.actions.append(action)
        self.memory.logprobs.append(logprob)
        self.memory.rewards.append(reward)
        self.memory.dones.append(done)
        self.memory.values.append(value)

    def _compute_gae(self, rewards, dones, values):
        values = np.append(values, 0.0)
        gae = 0.0
        returns = np.zeros_like(rewards)
        for t in reversed(range(len(rewards))):
            delta = rewards[t] + self.gamma * values[t + 1] * (1 - dones[t]) - values[t]
            gae = delta + self.gamma * self.gae_lambda * (1 - dones[t]) * gae
            returns[t] = gae + values[t]
        advantages = returns - values[:-1]
        return returns, advantages

    def update(self):
        if len(self.memory.states) == 0:
            return

        states = torch.tensor(np.array(self.memory.states), dtype=torch.float32, device=device)
        actions = torch.tensor(self.memory.actions, dtype=torch.long, device=device)
        old_logprobs = torch.tensor(self.memory.logprobs, dtype=torch.float32, device=device)
        rewards = np.array(self.memory.rewards, dtype=np.float32)
        dones = np.array(self.memory.dones, dtype=np.float32)
        values = np.array(self.memory.values, dtype=np.float32)

        returns, advantages = self._compute_gae(rewards, dones, values)

        returns = torch.tensor(returns, dtype=torch.float32, device=device)
        advantages = torch.tensor(advantages, dtype=torch.float32, device=device)

        # normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        for _ in range(self.k_epochs):
            logprobs, state_values, dist_entropy = self.policy.evaluate(states, actions)

            ratios = torch.exp(logprobs - old_logprobs)

            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1 - self.eps_clip, 1 + self.eps_clip) * advantages

            critic_loss = (returns - state_values) ** 2
            actor_loss = -torch.min(surr1, surr2)

            loss = actor_loss.mean() + 0.5 * critic_loss.mean() - self.entropy_coef * dist_entropy.mean()

            self.optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(self.policy.parameters(), 0.5)
            self.optimizer.step()

        self.memory.clear()

    def save(self, path: str):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save(self.policy.state_dict(), path)


class PPOTrafficLightController:
    """
    Wraps the TrafficSimulation environment and PPOAgent.
    """

    def __init__(self, simulation, episodes: int = 800, max_steps: int = 400):
        self.sim = simulation
        self.episodes = episodes          # number of PPO decisions per episode
        self.max_steps = max_steps

        # State = [avg_wait, throughput, total_passed_norm, current_queue_sum_norm,
        #          current_vehicles_norm, 8 lane queues]
        self.state_dim = 5 + 8
        self.action_dim = 4  # 4 phases

        self.agent = PPOAgent(
            state_dim=self.state_dim,
            action_dim=self.action_dim,
            lr=1e-4,
            gamma=0.99,
            gae_lambda=0.95,
            eps_clip=0.1,
            k_epochs=8,
            entropy_coef=0.02,
        )

        self.episode_rewards = []

    def build_state(self, metrics: dict) -> np.ndarray:
        # scalar metrics
        avg_wait = metrics.get("average_waiting_time", 0.0)
        throughput = metrics.get("throughput", 0.0)
        total_passed = metrics.get("total_vehicles_passed", 0.0)
        qsum = metrics.get("current_queue_sum", 0.0)
        current_vehicles = metrics.get("current_vehicles", 0.0)

        # queue lengths
        q_dict = metrics.get("queue_lengths", {})
        lane_keys = [
            "north_straight",
            "north_left",
            "south_straight",
            "south_left",
            "east_straight",
            "east_left",
            "west_straight",
            "west_left",
        ]
        lane_queues = [q_dict.get(k, 0.0) for k in lane_keys]

        # basic normalization
        avg_wait_norm = avg_wait / 100.0
        throughput_norm = throughput
        total_passed_norm = total_passed / 100.0
        qsum_norm = qsum / 20.0
        current_vehicles_norm = current_vehicles / 20.0
        lane_queues_norm = [q / 10.0 for q in lane_queues]

        state = np.array(
            [
                avg_wait_norm,
                throughput_norm,
                total_passed_norm,
                qsum_norm,
                current_vehicles_norm,
                *lane_queues_norm,
            ],
            dtype=np.float32,
        )
        return state

    def compute_reward(self, metrics: dict, old_metrics: dict | None) -> float:
        avg_wait = metrics.get("average_waiting_time", 0.0)
        qsum = metrics.get("current_queue_sum", 0.0)
        total_passed = metrics.get("total_vehicles_passed", 0.0)

        if old_metrics is None:
            passed_this_step = 0.0
        else:
            passed_this_step = total_passed - old_metrics.get("total_vehicles_passed", 0.0)
            if passed_this_step < 0:
                passed_this_step = 0.0

        reward = 0.0
        reward += 3.0 * passed_this_step
        lane_queues = metrics["queue_lengths"]
        imbalance = max(lane_queues.values()) - min(lane_queues.values())
        reward -= 0.02 * imbalance
        reward -= 0.015 * avg_wait
        #reward += 0.1
        #reward /= 2.0

        reward = np.clip(reward, -10.0, 10.0)
        return float(reward)

    def train(self, episodes=None, max_steps=None):

        if episodes is None:
            episodes = self.episodes
        if max_steps is None:
            max_steps = self.max_steps

        PHASE_DURATION = 12

        print("Training PPO...")
        self.episode_rewards.clear()

        for ep in range(1, episodes + 1):

            # NEW — debug lists for this episode
            debug_avg_wait = []
            debug_queue_sum = []
            debug_passed = []

            self.sim.reset()
            metrics = self.sim.get_metrics()
            old_metrics = None
            state = self.build_state(metrics)

            ep_reward = 0.0

            for step in range(max_steps):
                action, logprob, value = self.agent.select_action(state)

                block_reward = 0.0

                for _ in range(PHASE_DURATION):
                    _, metrics = self.sim.step(action)
                    r = self.compute_reward(metrics, old_metrics)
                    block_reward += r
                    ep_reward += r

                    # ---- DEBUG DATA HERE ----
                    debug_avg_wait.append(metrics["average_waiting_time"])
                    debug_queue_sum.append(metrics["current_queue_sum"])
                    debug_passed.append(metrics["passed_this_step"])
                    # --------------------------

                    old_metrics = metrics
                    state = self.build_state(metrics)

                self.agent.store_transition(
                    state=state, action=action, logprob=logprob,
                    reward=block_reward, done=False, value=value
                )

            self.agent.update()
            self.episode_rewards.append(ep_reward)

            # ---- MAKE THE PLOT ----
            if ep % 1000 == 0:
                import matplotlib.pyplot as plt
                plt.figure(figsize=(18,7))

                plt.subplot(3,1,1)
                plt.plot(debug_avg_wait)
                plt.title(f"Avg Wait (Episode {ep})")

                plt.subplot(3,1,2)
                plt.plot(debug_queue_sum)
                plt.title("Queue Sum")

                plt.subplot(3,1,3)
                plt.plot(debug_passed)
                plt.title("Passed This Step")

                plt.tight_layout()
                plt.show()
            if ep % 10 == 0:
                print(f"[PPO] Episode {ep} Reward = {np.mean(self.episode_rewards[-10:]):.2f}")
