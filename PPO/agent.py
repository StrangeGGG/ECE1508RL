import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical

#technique mention in the tut, which will make the model trained smooth and stable
class RunningNormalizer:
    def __init__(self, state_size: int, eps: float = 1e-8):
        self.state_size = state_size
        self.eps = eps

        self.mean = np.zeros(state_size, dtype=np.float32)
        self.var = np.ones(state_size, dtype=np.float32)
        self.count = eps 

    def update(self, x: np.ndarray):
        x = np.asarray(x, dtype=np.float32)
        if x.ndim == 1:
            x = x[None, :]

        batch_mean = np.mean(x, axis=0)
        batch_var = np.var(x, axis=0)
        batch_count = x.shape[0]

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
        
        if update:
            self.update(x)
        return self.normalize(x)
    
class PolicyNet(nn.Module):
    def __init__(self, state_size: int, action_size: int, hidden_sizes=(128, 128)):
        super().__init__()
        layers = []
        last_size = state_size
        for h in hidden_sizes:
            layers.append(nn.Linear(last_size, h))
            layers.append(nn.Tanh())
            last_size = h
        self.shared = nn.Sequential(*layers)
        self.action_head = nn.Linear(last_size, action_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.shared(x)
        logits = self.action_head(x)
        return logits

    def get_log_prob(self, states: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        logits = self.forward(states)
        dist = Categorical(logits=logits)
        log_probs = dist.log_prob(actions)
        return log_probs

    def select_action(self, state_tensor: torch.Tensor):
        logits = self.forward(state_tensor)
        dist = Categorical(logits=logits)
        action = dist.sample()
        log_prob = dist.log_prob(action)
        return int(action.item()), float(log_prob.item())

class ValueNet(nn.Module):
    def __init__(self, state_size: int, hidden_sizes=(128, 128)):
        super().__init__()
        layers = []
        last_size = state_size
        for h in hidden_sizes:
            layers.append(nn.Linear(last_size, h))
            layers.append(nn.Tanh())
            last_size = h
        self.shared = nn.Sequential(*layers)
        self.value_head = nn.Linear(last_size, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        v = self.shared(x)
        v = self.value_head(v).squeeze(-1)
        return v

class PPOAgent:
    def __init__(
        self,
        state_size: int,
        action_size: int,
        gamma: float = 0.99,
        lam: float = 0.95,
        lr: float = 3e-4,
        clip_ratio: float = 0.2,
        update_epochs: int = 10,
        minibatch_size: int = 128,
        reward_scale: float = 1.0,
        entropy_coef: float = 0.02,
        value_coef: float = 0.5,  
        max_grad_norm: float = 0.5,
        use_state_norm: bool = True,
        hidden_sizes=(128, 128),
        device: str | None = None,
        l2_reg: float = 1e-3,
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
        self.l2_reg = l2_reg

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)

        self.policy_net = PolicyNet(state_size, action_size, hidden_sizes).to(self.device)
        self.value_net = ValueNet(state_size, hidden_sizes).to(self.device)

        self.optimizer_policy = optim.Adam(self.policy_net.parameters(), lr=lr)
        self.optimizer_value = optim.Adam(self.value_net.parameters(), lr=lr)
        
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

    def select_action(self, state: np.ndarray) -> int:
        
        state = np.asarray(state, dtype=np.float32)

        if self.use_state_norm and self.state_normalizer is not None:
            norm_state = self.state_normalizer(state, update=True)
        else:
            norm_state = state

        state_tensor = torch.tensor(norm_state, dtype=torch.float32, device=self.device).unsqueeze(0)

        with torch.no_grad():
            action_int, log_prob = self.policy_net.select_action(state_tensor)
            value = self.value_net(state_tensor).item()

        self.states.append(norm_state)  
        self.actions.append(action_int)
        self.log_probs.append(log_prob)
        self.values.append(value)

        return action_int

    def store_reward(self, reward: float, done: bool):
        """
        Store reward and done flag for the last action taken.
        """
        self.rewards.append(float(reward) * self.reward_scale)
        self.dones.append(bool(done))

    def buffer_size(self) -> int:
        """
        Numinibatcher of time steps currently stored in the rollout buffer.
        """
        return len(self.rewards)

    def _clear_buffer(self):
        self.states.clear()
        self.actions.clear()
        self.log_probs.clear()
        self.rewards.clear()
        self.dones.clear()
        self.values.clear()

    def _compute_returns_and_advantages(self):
        """
        Compute GAE advantages and returns from the current buffer.
        """
        rewards = np.array(self.rewards, dtype=np.float32)
        dones = np.array(self.dones, dtype=np.bool_)
        values = np.array(self.values, dtype=np.float32)

        T = len(rewards)
        advantages = np.zeros(T, dtype=np.float32)
        last_adv = 0.0
        last_value = 0.0  

        for t in reversed(range(T)):
            mask = 1.0 - float(dones[t])
            r_t = rewards[t]
            v_t = values[t]

            delta = r_t + self.gamma * last_value * mask - v_t
            last_adv = delta + self.gamma * self.lam * mask * last_adv
            advantages[t] = last_adv

            last_value = v_t

        returns = advantages + values

        adv_mean = advantages.mean()
        adv_std = advantages.std() + 1e-8
        advantages = (advantages - adv_mean) / adv_std

        return returns, advantages

    def update(self):
        if self.buffer_size() == 0:
            return

        states = torch.tensor(np.array(self.states), dtype=torch.float32).to(self.device)
        actions = torch.tensor(self.actions, dtype=torch.long).to(self.device)
        old_log_probs = torch.tensor(self.log_probs, dtype=torch.float32).to(self.device)

        returns, advantages = self._compute_returns_and_advantages()
        returns = torch.tensor(returns, dtype=torch.float32).to(self.device)
        advantages = torch.tensor(advantages, dtype=torch.float32).to(self.device)

        dataset_size = states.size(0)
        batch_size = self.minibatch_size

        for _ in range(self.update_epochs):
            
            indices = torch.randperm(dataset_size, device=self.device)
            for start in range(0, dataset_size, batch_size):
                end = start + batch_size
                minibatch_idx = indices[start:end]

                minibatch_states = states[minibatch_idx]
                minibatch_actions = actions[minibatch_idx]
                minibatch_old_log_probs = old_log_probs[minibatch_idx]
                minibatch_returns = returns[minibatch_idx]
                minibatch_advantages = advantages[minibatch_idx]
                
                values_pred = self.value_net(minibatch_states)
                value_loss = F.mse_loss(values_pred, minibatch_returns)

                if self.l2_reg > 0.0:
                    l2 = 0.0
                    for param in self.value_net.parameters():
                        l2 = l2 + torch.sum(param.pow(2))
                    value_loss = value_loss + self.l2_reg * l2

                self.optimizer_value.zero_grad()
                value_loss.backward()
                self.optimizer_value.step()

                new_log_probs = self.policy_net.get_log_prob(minibatch_states, minibatch_actions)
                ratio = torch.exp(new_log_probs - minibatch_old_log_probs)

                clip_eps = self.clip_ratio
                surr1 = ratio * minibatch_advantages
                surr2 = torch.clamp(ratio, 1.0 - clip_eps, 1.0 + clip_eps) * minibatch_advantages
                policy_loss = -torch.min(surr1, surr2).mean()

                logits = self.policy_net(minibatch_states)
                dist = Categorical(logits=logits)
                entropy = dist.entropy().mean()

                total_policy_loss = policy_loss - self.entropy_coef * entropy

                self.optimizer_policy.zero_grad()
                total_policy_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), self.max_grad_norm)
                self.optimizer_policy.step()

        self._clear_buffer()

    def save(self, path: str):
        torch.save(
            {
                "policy_state_dict": self.policy_net.state_dict(),
                "value_state_dict": self.value_net.state_dict(),
                "use_state_norm": self.use_state_norm,
                "normalizer_mean": None
                if not self.use_state_norm or self.state_normalizer is None
                else self.state_normalizer.mean,
                "normalizer_var": None
                if not self.use_state_norm or self.state_normalizer is None
                else self.state_normalizer.var,
                "normalizer_count": 0
                if not self.use_state_norm or self.state_normalizer is None
                else self.state_normalizer.count,
            },
            path,
        )

    def load(self, path: str):
        checkpoint = torch.load(path, map_location=self.device)

        self.policy_net.load_state_dict(checkpoint["policy_state_dict"])
        self.value_net.load_state_dict(checkpoint["value_state_dict"])

        self.use_state_norm = checkpoint.get("use_state_norm", self.use_state_norm)
        if self.use_state_norm:
            if self.state_normalizer is None:
                self.state_normalizer = RunningNormalizer(self.state_size)
            mean = checkpoint.get("normalizer_mean", None)
            var = checkpoint.get("normalizer_var", None)
            count = checkpoint.get("normalizer_count", None)
            if mean is not None:
                self.state_normalizer.mean = mean
            if var is not None:
                self.state_normalizer.var = var
            if count is not None:
                self.state_normalizer.count = count
