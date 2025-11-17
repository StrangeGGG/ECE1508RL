import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from collections import deque
import matplotlib.pyplot as plt

from env import TrafficSimulation


class DQN(nn.Module):
    """Deep Q-Network for traffic light control"""

    def __init__(self, state_size, action_size, hidden_size=128):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(state_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, hidden_size)
        self.fc4 = nn.Linear(hidden_size, action_size)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.relu(self.fc3(x))
        return self.fc4(x)


class ReplayBuffer:
    """Experience replay buffer"""

    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = map(np.stack, zip(*batch))
        return state, action, reward, next_state, done

    def __len__(self):
        return len(self.buffer)


class DQNAgent:
    """DQN Agent for traffic light control"""

    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = ReplayBuffer(50000)
        self.batch_size = 64
        self.gamma = 0.95  # discount factor
        self.epsilon = 1.0  # exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = 0.0001
        self.update_every = 100  # target network update frequency

        # Device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(self.device)

        # Networks
        self.q_network = DQN(state_size, action_size).to(self.device)
        self.target_network = DQN(state_size, action_size).to(self.device)
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=self.learning_rate)

        # Training metrics
        self.losses = []

        self.update_target_network()
        self.t_step = 0

    def update_target_network(self):
        """Update target network with Q-network weights"""
        self.target_network.load_state_dict(self.q_network.state_dict())

    def act(self, state, training=True):
        """Choose action using epsilon-greedy policy"""
        if training and random.random() < self.epsilon:
            return random.choice(range(self.action_size))

        state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        self.q_network.eval()
        with torch.no_grad():
            action_values = self.q_network(state)
        self.q_network.train()

        return np.argmax(action_values.cpu().data.numpy())

    def step(self, state, action, reward, next_state, done):
        """Save experience and learn"""
        self.memory.push(state, action, reward, next_state, done)

        self.t_step = (self.t_step + 1) % self.update_every
        if self.t_step == 0 and len(self.memory) > self.batch_size:
            experiences = self.memory.sample(self.batch_size)
            loss = self.learn(experiences)
            self.losses.append(loss)

    def learn(self, experiences):
        """Update Q-network using experiences"""
        states, actions, rewards, next_states, dones = experiences

        states = torch.FloatTensor(states).to(self.device)
        actions = torch.LongTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.FloatTensor(dones).to(self.device)

        # Get current Q values
        current_q = self.q_network(states).gather(1, actions.unsqueeze(1))

        # Get next Q values from target network
        next_q = self.target_network(next_states).max(1)[0].detach()
        target_q = rewards + (self.gamma * next_q * (1 - dones))

        # Compute loss
        loss = nn.MSELoss()(current_q.squeeze(), target_q)

        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), 1.0)
        self.optimizer.step()

        # Update epsilon
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

        return loss.item()

    def save(self, filename):
        """Save model weights"""
        torch.save(self.q_network.state_dict(), filename)

    def load(self, filename):
        """Load model weights"""
        self.q_network.load_state_dict(torch.load(filename))
        self.update_target_network()


class SmartTrafficLightController:
    """Smart traffic light controller using DQN"""

    def __init__(self, simulation):
        self.simulation = simulation
        self.state_size = 24  # 8 lanes × 3 features (queue_length, avg_waiting_time, vehicle_count)
        self.action_size = 4  # 4 phases

        self.agent = DQNAgent(self.state_size, self.action_size)
        self.current_phase = 0
        self.phase_duration = 0
        self.min_phase_duration = 10  # minimum time steps per phase
        self.max_phase_duration = 60  # maximum time steps per phase

        # Training tracking
        self.scores = []
        self.avg_scores = []
        self.episode_rewards = []

        # Performance metrics
        self.throughputs_history = []
        self.waiting_times_history = []
        self.episode_throughputs = []
        self.episode_waiting_times = []

    def get_state(self):
        """Get current state representation for DQN"""
        metrics = self.simulation.metrics_collector.get_metrics()
        queue_lengths = metrics['queue_lengths']

        state = []

        # For each direction and lane type, extract features
        for direction in ['north', 'south', 'east', 'west']:
            for lane_type in ['straight', 'left']:
                key = f'{direction}_{lane_type}'

                # Feature 1: Queue length
                queue_len = queue_lengths[key]

                # Feature 2: Average waiting time for vehicles in this lane
                lane_vehicles = [v for v in self.simulation.vehicles.values()
                                 if v.direction == direction and v.lane_type == lane_type
                                 and not v.passed and not v.collided]
                avg_wait = np.mean([v.waiting_time for v in lane_vehicles]) if lane_vehicles else 0

                # Feature 3: Vehicle count
                vehicle_count = len(lane_vehicles)

                state.extend([queue_len, avg_wait, vehicle_count])

        return np.array(state)

    def calculate_reward(self, metrics, old_metrics=None):
        """Calculate reward based on traffic metrics"""
        reward = 0

        # Reward for throughput (vehicles passed per step)
        current_throughput = metrics['throughput']
        reward += current_throughput * 10

        # Penalty for long waiting times
        avg_wait = metrics['average_waiting_time']
        reward -= avg_wait * 0.1

        # Penalty for long queues
        total_queue = sum(metrics['queue_lengths'].values())
        reward -= total_queue * 0.05

        # Small penalty for phase changes to encourage stability
        if old_metrics and self.phase_duration == 1:
            reward -= 0.1

        # Bonus for efficient phase completion
        if self.phase_duration >= self.min_phase_duration:
            vehicles_passed = metrics['total_vehicles_passed'] - (
                old_metrics['total_vehicles_passed'] if old_metrics else 0)
            reward += vehicles_passed * 2

        return reward

    def train(self, episodes=1000, max_steps=1000):
        """Train the DQN agent"""
        print("Starting DQN training...")

        for episode in range(episodes):
            self.simulation.reset()
            state = self.get_state()
            total_reward = 0
            old_metrics = None

            # Episode-specific metrics
            episode_throughputs = []
            episode_waiting_times = []

            for step in range(max_steps):
                # Choose action
                action = self.agent.act(state)

                # Execute phase if it's time to change
                if self.phase_duration >= self.min_phase_duration:
                    # Check if we should change phase based on DQN action
                    if action != self.current_phase:
                        self.current_phase = action
                        self.phase_duration = 0

                # Run simulation step
                observation, metrics = self.simulation.step(self.current_phase)
                self.phase_duration += 1

                # Force phase change if maximum duration reached
                if self.phase_duration >= self.max_phase_duration:
                    self.current_phase = (self.current_phase + 1) % self.action_size
                    self.phase_duration = 0

                # Calculate reward
                reward = self.calculate_reward(metrics, old_metrics)
                total_reward += reward

                # Record metrics
                episode_throughputs.append(metrics['throughput'])
                episode_waiting_times.append(metrics['average_waiting_time'])

                # Get next state
                next_state = self.get_state()

                # Store experience and learn
                done = (step == max_steps - 1)
                self.agent.step(state, action, reward, next_state, done)

                state = next_state
                old_metrics = metrics.copy()

                if done:
                    break

            # Update target network periodically
            if episode % 10 == 0:
                self.agent.update_target_network()

            # Record scores and metrics
            self.scores.append(total_reward)
            avg_score = np.mean(self.scores[-100:])
            self.avg_scores.append(avg_score)
            self.episode_rewards.append(total_reward)

            # Record episode averages
            self.episode_throughputs.append(np.mean(episode_throughputs))
            self.episode_waiting_times.append(np.mean(episode_waiting_times))

            if episode % 10 == 0:
                avg_throughput = np.mean(episode_throughputs)
                avg_waiting = np.mean(episode_waiting_times)
                print(f"Episode {episode}, Score: {total_reward:.2f}, Avg Score: {avg_score:.2f}, "
                      f"Throughput: {avg_throughput:.4f}, Wait Time: {avg_waiting:.2f}, "
                      f"Epsilon: {self.agent.epsilon:.3f}")

        print("Training completed!")
        return self.scores, self.avg_scores

    def test(self, max_steps=1000):
        """Test the trained agent"""
        print("Testing trained agent...")
        self.simulation.reset()
        state = self.get_state()
        total_reward = 0
        old_metrics = None

        # Set epsilon to minimum for testing
        original_epsilon = self.agent.epsilon
        self.agent.epsilon = self.agent.epsilon_min

        phase_changes = []
        waiting_times = []
        throughputs = []

        for step in range(max_steps):
            # Choose action
            action = self.agent.act(state, training=False)

            # Execute phase if it's time to change
            if self.phase_duration >= self.min_phase_duration:
                if action != self.current_phase:
                    phase_changes.append(step)
                    self.current_phase = action
                    self.phase_duration = 0

            # Run simulation step
            observation, metrics = self.simulation.step(self.current_phase)
            self.phase_duration += 1

            # Force phase change if maximum duration reached
            if self.phase_duration >= self.max_phase_duration:
                self.current_phase = (self.current_phase + 1) % self.action_size
                self.phase_duration = 0

            # Calculate reward and collect metrics
            reward = self.calculate_reward(metrics, old_metrics)
            total_reward += reward

            waiting_times.append(metrics['average_waiting_time'])
            throughputs.append(metrics['throughput'])

            state = self.get_state()
            old_metrics = metrics.copy()

        # Restore original epsilon
        self.agent.epsilon = original_epsilon

        print(f"Test completed. Total reward: {total_reward:.2f}")
        print(f"Average waiting time: {np.mean(waiting_times):.2f}")
        print(f"Average throughput: {np.mean(throughputs):.4f}")
        print(f"Number of phase changes: {len(phase_changes)}")

        with open('testing_results.txt', 'w', encoding='utf-8') as f:
            f.write("TRAFFIC SIMULATION RESULTS\n")
            f.write("=" * 50 + "\n")
            f.write(f"Test completed. Total reward: {total_reward:.2f}\n")
            f.write(f"Average waiting time: {np.mean(waiting_times):.2f}\n")
            f.write(f"Average throughput: {np.mean(throughputs):.4f}\n")
            f.write(f"Number of phase changes: {len(phase_changes)}\n")

        return total_reward, waiting_times, throughputs, phase_changes

    def plot_training(self):
        """Plot comprehensive training results"""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))

        # Plot 1: Training scores
        ax1.plot(self.scores, alpha=0.6, label='Episode Score', color='blue', linewidth=1)
        ax1.plot(self.avg_scores, label='Average Score (100 episodes)', color='red', linewidth=2)
        ax1.set_xlabel('Episode')
        ax1.set_ylabel('Score')
        ax1.legend()
        ax1.set_title('Training Scores')
        ax1.grid(True, alpha=0.3)

        # Plot 2: Loss over time
        if self.agent.losses:
            # Smooth losses for better visualization
            window_size = max(1, len(self.agent.losses) // 100)
            smoothed_losses = np.convolve(self.agent.losses, np.ones(window_size) / window_size, mode='valid')
            ax2.plot(smoothed_losses, color='green', linewidth=1)
            ax2.set_xlabel('Training Steps')
            ax2.set_ylabel('Loss')
            ax2.set_title('Training Loss (Smoothed)')
            ax2.set_yscale('log')  # Use log scale for better visualization
            ax2.grid(True, alpha=0.3)

        # Plot 3: Throughput over episodes
        ax3.plot(self.episode_throughputs, color='purple', linewidth=2)
        ax3.set_xlabel('Episode')
        ax3.set_ylabel('Throughput (vehicles/step)')
        ax3.set_title('Average Throughput per Episode')
        ax3.grid(True, alpha=0.3)

        # Plot 4: Waiting time over episodes
        ax4.plot(self.episode_waiting_times, color='orange', linewidth=2)
        ax4.set_xlabel('Episode')
        ax4.set_ylabel('Waiting Time (steps)')
        ax4.set_title('Average Waiting Time per Episode')
        ax4.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig('training_progress.png', dpi=300, bbox_inches='tight')
        plt.show()

        # Print summary statistics
        self.print_training_summary()

    def print_training_summary(self):
        """Print training summary statistics"""
        print("\n=== Training Summary ===")
        print(f"Final Average Score: {self.avg_scores[-1]:.2f}")
        print(f"Final Throughput: {self.episode_throughputs[-1]:.4f} vehicles/step")
        print(f"Final Waiting Time: {self.episode_waiting_times[-1]:.2f} steps")
        print(f"Final Epsilon: {self.agent.epsilon:.3f}")

        # Improvement statistics
        if len(self.episode_throughputs) > 10:
            initial_throughput = np.mean(self.episode_throughputs[:10])
            final_throughput = np.mean(self.episode_throughputs[-10:])
            throughput_improvement = ((final_throughput - initial_throughput) / initial_throughput) * 100

            initial_waiting = np.mean(self.episode_waiting_times[:10])
            final_waiting = np.mean(self.episode_waiting_times[-10:])
            waiting_improvement = ((initial_waiting - final_waiting) / initial_waiting) * 100

            print(f"Throughput Improvement: {throughput_improvement:+.1f}%")
            print(f"Waiting Time Improvement: {waiting_improvement:+.1f}%")


# Usage example
def main():
    # Initialize simulation
    simulation = TrafficSimulation()

    # Create smart controller
    controller = SmartTrafficLightController(simulation)

    # Train the agent
    print("Starting training...")
    scores, avg_scores = controller.train(episodes=500, max_steps=1000)

    # Plot comprehensive training results
    controller.plot_training()

    # Test the trained agent
    print("\nStarting testing...")
    test_reward, waiting_times, throughputs, phase_changes = controller.test(max_steps=500)

    # Plot test results
    plt.figure(figsize=(12, 4))

    plt.subplot(1, 2, 1)
    plt.plot(throughputs)
    plt.xlabel('Time Step')
    plt.ylabel('Throughput')
    plt.title('Test: Throughput over Time')
    plt.grid(True, alpha=0.3)

    plt.subplot(1, 2, 2)
    plt.plot(waiting_times)
    plt.xlabel('Time Step')
    plt.ylabel('Waiting Time')
    plt.title('Test: Waiting Time over Time')
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('testing_result.png', dpi=300, bbox_inches='tight')
    plt.show()

    # Save the trained model
    controller.agent.save("traffic_dqn_model.pth")
    print("Model saved as 'traffic_dqn_model.pth'")


if __name__ == "__main__":
    main()