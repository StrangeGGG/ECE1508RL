import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from datetime import datetime
import os
from FixedTime_basic import FixedTimingTrafficLight, create_intersection_env

class AdvancedTrafficController:
    """Advanced traffic light controller, including data logging and analysis"""

    def __init__(self, phase_durations=[30, 30, 20, 20]):
        self.traffic_light = FixedTimingTrafficLight(phase_durations)

        # data logging
        self.episode_data = {
            'steps': [],
            'phases': [],
            'rewards': [],
            'collisions': [],
            'successes': []
        }

        # Create results directory
        os.makedirs('./results', exist_ok=True)

    def run_episode(self, env, max_steps=300):
        """Run an episode of simulation"""
        obs, info = env.reset()
        self.traffic_light.reset()

        episode_reward = 0
        steps = 0
        collisions = 0
        successes = 0

        phase_history = []
        reward_history = []

        for step in range(max_steps):
            # Update traffic light
            current_phase = self.traffic_light.step()
            phase_history.append(current_phase)

            # simple control strategy
            action = self._choose_action(current_phase, obs)

            # environment step
            obs, reward, terminated, truncated, info = env.step(action)

            episode_reward += reward
            reward_history.append(reward)
            steps += 1

            # Statistics
            if info.get('crashed', False):
                collisions += 1
            if info.get('is_success', False):
                successes += 1

            if terminated or truncated:
                break

        # Record data
        self.episode_data['steps'].append(steps)
        self.episode_data['phases'].append(phase_history)
        self.episode_data['rewards'].append(episode_reward)
        self.episode_data['collisions'].append(collisions)
        self.episode_data['successes'].append(successes)

        return episode_reward, steps, collisions, successes

    def _choose_action(self, phase, obs):
        """Action selection based on current phase"""
        # Simple phase-based strategy
        if phase in [0, 2]:  # north-south green light
            # Encouragement to move forward
            return 1  # accelerate
        else:  # East-west green light
            # conservative behavior
            return 0  # Keep
        # This is a simplified strategy; the actual approach should be based on the observed state.

    def analyze_performance(self):
        """Analyze performance and generate reports"""
        print("\n" + "=" * 60)
        print("Performance analysis report")
        print("=" * 60)

        # basic statistics
        rewards = self.episode_data['rewards']
        steps = self.episode_data['steps']
        collisions = self.episode_data['collisions']

        print(f"Basic record:")
        print(f"Average reward: {np.mean(rewards):.2f} ± {np.std(rewards):.2f}")
        print(f"Average step: {np.mean(steps):.2f} ± {np.std(steps):.2f}")
        print(f"Average collision: {np.mean(collisions):.2f}")
        print(f"Total number of simulation episodes: {len(rewards)}")

        # Phase analysis
        all_phases = []
        for phase_list in self.episode_data['phases']:
            all_phases.extend(phase_list)

        phase_percentages = np.bincount(all_phases, minlength=4) / len(all_phases) * 100
        phase_names = ['Go straight north-south', 'Go straight east-west', 'Turn left north-south', 'Turn left east-west']

        print(f"\n Phase distribution:")
        for i, (name, percentage) in enumerate(zip(phase_names, phase_percentages)):
            print(f"   {name}: {percentage:5.1f}%")

        # Generate visualization
        self._create_visualizations()

        # save data
        self._save_data()

    def _create_visualizations(self):
        """Create visualization charts"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))

        # 1. Reward Trends
        axes[0, 0].plot(self.episode_data['rewards'], 'b-o', linewidth=2)
        axes[0, 0].set_title('Total rewards per episode', fontsize=14, fontweight='bold')
        axes[0, 0].set_xlabel('episode')
        axes[0, 0].set_ylabel('rewards')
        axes[0, 0].grid(True, alpha=0.3)

        # 2. step distribution
        axes[0, 1].plot(self.episode_data['steps'], 'g-o', linewidth=2)
        axes[0, 1].set_title('steps per episode', fontsize=14, fontweight='bold')
        axes[0, 1].set_xlabel('episode')
        axes[0, 1].set_ylabel('steps')
        axes[0, 1].grid(True, alpha=0.3)

        # 3. Collision statistics
        axes[1, 0].bar(range(len(self.episode_data['collisions'])),
                       self.episode_data['collisions'],
                       color='red', alpha=0.7)
        axes[1, 0].set_title('Number of collisions per episode', fontsize=14, fontweight='bold')
        axes[1, 0].set_xlabel('episode')
        axes[1, 0].set_ylabel('number of collisions')

        # 4. Phase distribution
        all_phases = []
        for phase_list in self.episode_data['phases']:
            all_phases.extend(phase_list)

        phase_counts = np.bincount(all_phases, minlength=4)
        phase_names = ['Go straight north-south', 'Go straight east or west', 'Turn left north or south', 'Turn left east or west']
        colors = ['#ff9999', '#66b3ff', '#99ff99', '#ffcc99']

        axes[1, 1].pie(phase_counts, labels=phase_names, autopct='%1.1f%%',
                       colors=colors, startangle=90)
        axes[1, 1].set_title('Phase time distribution', fontsize=14, fontweight='bold')

        plt.tight_layout()

        # save image
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        plt.savefig(f'./results/traffic_analysis_{timestamp}.png',
                    dpi=300, bbox_inches='tight')
        plt.show()

    def _save_data(self):
        """Save data to CSV"""
        df = pd.DataFrame({
            'episode': range(len(self.episode_data['rewards'])),
            'total_reward': self.episode_data['rewards'],
            'steps': self.episode_data['steps'],
            'collisions': self.episode_data['collisions'],
            'successes': self.episode_data['successes']
        })

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        df.to_csv(f'./results/traffic_data_{timestamp}.csv', index=False)
        print(f"Data has been saved to: ./results/traffic_data_{timestamp}.csv")


def main():
    """main function"""
    print("Advanced traffic signal control system")
    print("=" * 50)

    # create environment
    env = create_intersection_env()

    # create controller
    controller = AdvancedTrafficController(phase_durations=[30, 30, 15, 15])

    # Running multi-episode simulations
    num_episodes = 5

    for episode in range(num_episodes):
        print(f"\n Episode {episode + 1}/{num_episodes}")

        reward, steps, collisions, successes = controller.run_episode(env, max_steps=200)

        print(f"Result: Reward = {reward:.2f}, Steps = {steps}"
              f"Collisions = {collisions}, Successes = {successes}")

    # close environment
    env.close()

    # analyze result
    controller.analyze_performance()

    print("\nAll simulations completed！")


if __name__ == "__main__":
    main()