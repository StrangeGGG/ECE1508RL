import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque

class DQN(nn.Module):
    def __init__(self, state_size, action_size, hidden1=256, hidden2=128):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(state_size, hidden1)
        self.fc2 = nn.Linear(hidden1, hidden2)
        self.fc3 = nn.Linear(hidden2, action_size)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        return self.fc3(x)

class ReplayBuffer:
    def __init__(self, capacity=50000):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        s, a, r, s2, d = zip(*batch)
        return (np.stack(s), np.array(a), np.array(r, dtype=np.float32),
                np.stack(s2), np.array(d, dtype=np.uint8))

    def __len__(self):
        return len(self.buffer)

class DQNAgent:
    def __init__(self, state_size, action_size, device=None,
                 lr=1e-4, gamma=0.95, batch_size=512, buffer_size=100000,
                 eps_start=1.0, eps_end=0.1, eps_decay=0.9995, update_every=50, max_grad_norm=1.0, global_step=0,
                 eps_decay_steps=600000):
        self.state_size = state_size
        self.action_size = action_size
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.online = DQN(state_size, action_size).to(self.device)
        self.target = DQN(state_size, action_size).to(self.device)
        self.target.load_state_dict(self.online.state_dict())

        self.optimizer = optim.Adam(self.online.parameters(), lr=lr)
        self.loss_fn = nn.SmoothL1Loss()

        self.replay = ReplayBuffer(capacity=buffer_size)
        self.batch_size = batch_size
        self.gamma = gamma

        # epsilon schedule (per step - stable)
        self.eps = eps_start
        self.eps_start = eps_start
        self.eps_end = eps_end
        self.eps_decay = eps_decay
        self.global_step = global_step
        self.eps_decay_steps = eps_decay_steps

        self.update_every = update_every
        self.learn_step = 0
        self.max_grad_norm = max_grad_norm

    def act(self, state, training=True):
        if training and random.random() < self.eps:
            return random.randrange(self.action_size)
        state_t = torch.from_numpy(state).float().unsqueeze(0).to(self.device)
        with torch.no_grad():
            q = self.online(state_t)
        return int(q.argmax(1).cpu().numpy()[0])

    def step(self, state, action, reward, next_state, done):
        self.replay.push(state, action, reward, next_state, done)
        # decay epsilon per experience step (only when a new sample added)
        if self.eps > self.eps_end:
            # self.eps = max(self.eps_end, self.eps * self.eps_decay)
            self.global_step += 1
            self.eps = max(
                self.eps_end,
                self.eps_start - (self.global_step / self.eps_decay_steps) * (self.eps_start - self.eps_end)
            )

        # learn if enough samples
        loss = None
        if len(self.replay) >= self.batch_size:
            loss = self.learn()
        return loss

    def learn(self):
        s, a, r, s2, d = self.replay.sample(self.batch_size)
        states = torch.from_numpy(s).float().to(self.device)
        actions = torch.from_numpy(a).long().unsqueeze(1).to(self.device)
        rewards = torch.from_numpy(r).float().unsqueeze(1).to(self.device)
        next_states = torch.from_numpy(s2).float().to(self.device)
        dones = torch.from_numpy(d.astype(np.uint8)).float().unsqueeze(1).to(self.device)

        # Double DQN target
        with torch.no_grad():
            best_next_actions = self.online(next_states).argmax(dim=1, keepdim=True)
            target_q_next = self.target(next_states).gather(1, best_next_actions)

        target_q = rewards + (self.gamma * target_q_next * (1.0 - dones))
        current_q = self.online(states).gather(1, actions)

        loss = self.loss_fn(current_q, target_q)

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.online.parameters(), self.max_grad_norm)
        self.optimizer.step()

        self.learn_step += 1
        if self.learn_step % self.update_every == 0:
            self.target.load_state_dict(self.online.state_dict())

        return loss.item()

    def save(self, path):
        torch.save(self.online.state_dict(), path)

    def load(self, path):
        self.online.load_state_dict(torch.load(path, map_location=self.device))
        self.target.load_state_dict(self.online.state_dict())
