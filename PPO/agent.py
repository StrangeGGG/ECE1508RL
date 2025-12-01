import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical


class RunningNormalizer:
    """
    Running mean / variance normalizer for states.
    This is purely a preprocessing helper and not part of PPO itself.
    """

    def __init__(self, state_size: int, eps: float = 1e-8):
        self.state_size = state_size
        self.eps = eps

        self.mean = np.zeros(state_size, dtype=np.float32)
        self.var = np.ones(state_size, dtype=np.float32)
        self.count = eps  # avoid divide-by-zero at the very beginning

    def update(self, x: np.ndarray):
        """
        Update running statistics from a batch of states x (shape [B, state_size]).
        """
        x = np.asarray(x, dtype=np.float32)
        if x.ndim == 1:
            x = x[None, :]

        batch_mean = np.mean(x, axis=0)
        batch_var = np.var(x, axis=0)
        batch_count = x.shape[0]

        # From OpenAI Baselines running mean/std
        delta = batch_mean - self.mean
        total_count = self.count + batch_count

        new_mean = self.mean + delta * batch_count / total_count
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        M2 = m_a + m_b + delta**2 * self.count * batch_count / total_count
        new_var = M2 / total_count

        self.mean = new_mean
        self.var = new_var
        self.count = total_count

    def normalize(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=np.float32)
        return (x - self.mean) / (np.sqrt(self.var) + self.eps)

    def __call__(self, x: np.ndarray, update: bool = True) -> np.ndarray:
        """
        Convenience wrapper: optionally update stats, then return normalized x.
        """
        if update:
            self.update(x)
        return self.normalize(x)


class ActorCritic(nn.Module):
    """
    Shared-torso actor-critic network used by PPO.
    """

    def __init__(self, state_size: int, action_size: int, hidden_sizes=(128, 128)):
        super().__init__()
        layers = []
        last_size = state_size
        for h in hidden_sizes:
            layers.append(nn.Linear(last_size, h))
            layers.append(nn.Tanh())
            last_size = h
        self.shared = nn.Sequential(*layers)

        self.policy_head = nn.Linear(last_size, action_size)
        self.value_head = nn.Linear(last_size, 1)

    def forward(self, x):
        x = self.shared(x)
        logits = self.policy_head(x)
        value = self.value_head(x).squeeze(-1)
        return logits, value


class PPOAgent:
    """
    PPO agent for discrete action spaces.

    Usage pattern (pseudo-code):

        env = TrafficRLWrapper(...)
        state = env.reset()
        agent = PPOAgent(state_size=len(state), action_size=4, ...)

        for each training step:
            action = agent.select_action(state)
            next_state, reward, _, info = env.step(action)
            done = ...  # e.g. horizon-based
            agent.store_reward(reward, done)
            state = next_state

            if agent.buffer_size() >= steps_per_batch:
                agent.update()
    """

    def __init__(
        self,
        state_size: int,
        action_size: int,
        gamma: float = 0.99,
        lam: float = 0.95,
        lr: float = 3e-4,
        clip_ratio: float = 0.2,
        update_epochs: int = 10,
        minibatch_size: int = 64,
        reward_scale: float = 1.0,
        entropy_coef: float = 0.01,
        value_coef: float = 0.5,
        max_grad_norm: float = 0.5,
        use_state_norm: bool = True,
        hidden_sizes=(128, 128),
        device: str | None = None,
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
        self.value_coef = value_coef
        self.max_grad_norm = max_grad_norm

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)

        # Networks
        self.ac = ActorCritic(state_size, action_size, hidden_sizes).to(self.device)
        self.optimizer = optim.Adam(self.ac.parameters(), lr=lr)

        # Optional state normalizer
        self.use_state_norm = use_state_norm
        self.state_normalizer = (
            RunningNormalizer(state_size) if use_state_norm else None
        )

        # Rollout storage
        self.states: list[np.ndarray] = []
        self.actions: list[int] = []
        self.log_probs: list[float] = []
        self.rewards: list[float] = []
        self.dones: list[bool] = []
        self.values: list[float] = []

    # ---------------- interaction API ----------------

    def select_action(self, state: np.ndarray) -> int:
        """
        Given a state (np.array of shape [state_size]), sample an action from π_θ.
        Also stores the transition pieces needed for PPO update.
        """
        state = np.asarray(state, dtype=np.float32)

        if self.use_state_norm:
            # update running stats with the *current* state, then normalize
            norm_state = self.state_normalizer(state, update=True)
        else:
            norm_state = state

        state_tensor = torch.from_numpy(norm_state).float().to(self.device).unsqueeze(0)

        with torch.no_grad():
            logits, value = self.ac(state_tensor)
            dist = Categorical(logits=logits)
            action = dist.sample()
            log_prob = dist.log_prob(action)

        action_int = int(action.item())

        # Store transition pieces
        self.states.append(norm_state)  # store normalized state if using normalizer
        self.actions.append(action_int)
        self.log_probs.append(log_prob.item())
        self.values.append(value.item())

        return action_int

    def store_reward(self, reward: float, done: bool):
        """
        Store reward and done flag for the last action taken.
        """
        self.rewards.append(float(reward) * self.reward_scale)
        self.dones.append(bool(done))

    def buffer_size(self) -> int:
        return len(self.rewards)

    def clear_buffer(self):
        self.states.clear()
        self.actions.clear()
        self.log_probs.clear()
        self.rewards.clear()
        self.dones.clear()
        self.values.clear()

    # ---------------- PPO update ----------------

    def _compute_returns_and_advantages(self):
        """
        Compute GAE advantages and bootstrap returns from the stored rollout.
        Assumes the rollout is a sequence of full episodes or fixed-length segments
        where 'done' is True at episode boundaries you define.
        """
        rewards = np.array(self.rewards, dtype=np.float32)
        values = np.array(self.values, dtype=np.float32)
        dones = np.array(self.dones, dtype=np.float32)

        T = len(rewards)
        advantages = np.zeros(T, dtype=np.float32)
        last_adv = 0.0
        last_value = 0.0  # we treat the value after terminal as 0

        for t in reversed(range(T)):
            mask = 1.0 - dones[t]  # 0 if done at t, 1 otherwise
            delta = rewards[t] + self.gamma * last_value * mask - values[t]
            last_adv = delta + self.gamma * self.lam * last_adv * mask
            advantages[t] = last_adv
            last_value = values[t]

        returns = advantages + values
        # Normalize advantages for better conditioning
        adv_mean = advantages.mean()
        adv_std = advantages.std() + 1e-8
        advantages = (advantages - adv_mean) / adv_std
        return returns, advantages

    def update(self):
        """
        Run PPO update using all data currently in the buffer.
        After update, the buffer is cleared.
        """
        if self.buffer_size() == 0:
            return

        # Convert buffers to tensors
        states = torch.tensor(np.array(self.states), dtype=torch.float32).to(self.device)
        actions = torch.tensor(self.actions, dtype=torch.long).to(self.device)
        old_log_probs = torch.tensor(self.log_probs, dtype=torch.float32).to(self.device)

        returns, advantages = self._compute_returns_and_advantages()
        returns = torch.tensor(returns, dtype=torch.float32).to(self.device)
        advantages = torch.tensor(advantages, dtype=torch.float32).to(self.device)

        dataset_size = states.size(0)
        batch_size = self.minibatch_size

        for _ in range(self.update_epochs):
            # Shuffle indices for mini-batches
            indices = torch.randperm(dataset_size)
            for start in range(0, dataset_size, batch_size):
                end = start + batch_size
                mb_idx = indices[start:end]

                mb_states = states[mb_idx]
                mb_actions = actions[mb_idx]
                mb_old_log_probs = old_log_probs[mb_idx]
                mb_returns = returns[mb_idx]
                mb_advantages = advantages[mb_idx]

                logits, values = self.ac(mb_states)
                dist = Categorical(logits=logits)
                new_log_probs = dist.log_prob(mb_actions)
                entropy = dist.entropy().mean()

                # PPO ratio
                ratio = torch.exp(new_log_probs - mb_old_log_probs)

                # Clipped surrogate objective
                surr1 = ratio * mb_advantages
                surr2 = torch.clamp(
                    ratio, 1.0 - self.clip_ratio, 1.0 + self.clip_ratio
                ) * mb_advantages
                policy_loss = -torch.min(surr1, surr2).mean()

                # Value function loss (MSE)
                value_loss = F.mse_loss(values, mb_returns)

                loss = policy_loss + self.value_coef * value_loss - self.entropy_coef * entropy

                self.optimizer.zero_grad()
                loss.backward()
                if self.max_grad_norm is not None:
                    torch.nn.utils.clip_grad_norm_(self.ac.parameters(), self.max_grad_norm)
                self.optimizer.step()

        # Clear rollout buffer after update
        self.clear_buffer()

    # ---------------- save / load ----------------

    def save(self, path: str):
        torch.save(
            {
                "ac_state_dict": self.ac.state_dict(),
                "normalizer_mean": None
                if not self.use_state_norm
                else self.state_normalizer.mean,
                "normalizer_var": None
                if not self.use_state_norm
                else self.state_normalizer.var,
                "normalizer_count": None
                if not self.use_state_norm
                else self.state_normalizer.count,
            },
            path,
        )

    def load(self, path: str):
        checkpoint = torch.load(path, map_location=self.device)
        self.ac.load_state_dict(checkpoint["ac_state_dict"])

        if self.use_state_norm and self.state_normalizer is not None:
            if checkpoint.get("normalizer_mean") is not None:
                self.state_normalizer.mean = checkpoint["normalizer_mean"]
                self.state_normalizer.var = checkpoint["normalizer_var"]
                self.state_normalizer.count = checkpoint["normalizer_count"]