import highway_env
import gymnasium as gym
import numpy as np


class FixedTimingTrafficLight:
    """Fixed-time traffic light controller"""

    def __init__(self, phase_durations=[30, 30, 20, 20]):
        """
        Args:
            phase_durations: The duration of each phase [phase0, phase1, phase2, phase3]
        """
        self.phase_durations = phase_durations
        self.num_phases = len(phase_durations)
        self.current_phase = 0
        self.phase_timer = 0
        self.total_steps = 0

        # phase description
        self.phase_descriptions = [
            "North-south direction straight green light",
            "East-west direction straight green light",
            "Green light for left turns from north and south",
            "Green light for left turns from east and west"
        ]

    def step(self):
        """Update traffic light status"""
        self.phase_timer += 1
        self.total_steps += 1

        # Check if a phase switch is needed.
        if self.phase_timer >= self.phase_durations[self.current_phase]:
            self.current_phase = (self.current_phase + 1) % self.num_phases
            self.phase_timer = 0
            print(f"signal light switch: {self.phase_descriptions[self.current_phase]}")

        return self.current_phase

    def get_status(self):
        """Get current status"""
        progress = self.phase_timer / self.phase_durations[self.current_phase]
        remaining = self.phase_durations[self.current_phase] - self.phase_timer

        return {
            'current_phase': self.current_phase,
            'phase_description': self.phase_descriptions[self.current_phase],
            'progress': progress,
            'remaining_time': remaining,
            'total_steps': self.total_steps
        }

    def reset(self):
        """Reset controller"""
        self.current_phase = 0
        self.phase_timer = 0
        self.total_steps = 0


def create_intersection_env():
    """Create and configure the intersection environment"""

    env = gym.make('intersection-v0', render_mode='rgb_array')

    # Environment configuration
    config = {
        "observation": {
            "type": "Kinematics",
            "vehicles_count": 15,
            "features": ["presence", "x", "y", "vx", "vy", "cos_h", "sin_h"],
            "absolute": True,
            "flatten": False,
        },
        "action": {
            "type": "DiscreteMetaAction",
            "longitudinal": True,
            "lateral": True
        },
        "duration": 50,
        "controlled_vehicles": 1,
        "initial_vehicle_count": 10,
        "spawn_probability": 0.6,
        "screen_width": 800,
        "screen_height": 800,
        "scaling": 5.5,
        "show_trajectories": True,
        "render_agent": True,
    }

    env.unwrapped.configure(config)
    return env


def run_basic_demo():
    """Run the basic demo"""
    print("Start traffic light control demonstration")
    print("=" * 50)

    # Creating environments and controllers
    env = create_intersection_env()
    traffic_light = FixedTimingTrafficLight(phase_durations=[25, 25, 15, 15])

    # Run a demo episode
    obs, info = env.reset()
    traffic_light.reset()

    episode_reward = 0
    steps = 0
    max_steps = 200

    phase_history = []
    reward_history = []

    print("Start simulation...")

    for step in range(max_steps):
        # 1. Update traffic lights
        current_phase = traffic_light.step()
        phase_history.append(current_phase)

        # 2. Choose an action (simple strategy)
        action = 0  # IDLE action

        # 3. environment step
        obs, reward, terminated, truncated, info = env.step(action)

        episode_reward += reward
        reward_history.append(reward)
        steps += 1

        # 4. Show status
        if step % 30 == 0:
            status = traffic_light.get_status()
            print(f"step  {step:3d}: {status['phase_description']} "
                  f"({status['remaining_time']:2d}second left) | "
                  f"reward: {reward:6.2f}")

        # 5. Check if it is finished
        if terminated or truncated:
            print("\nDetailed termination information:")
            # Additional diagnostic information provided by the printing environment
            print(info)
            print(f" terminated: {terminated}, truncated: {truncated}")
            print("Simulation ends early")
            break

    # display result
    print(f"\nSimulation results:")
    print(f"Total steps: {steps}")
    print(f"Total reward: {episode_reward:.2f}")
    print(f"Average reward: {np.mean(reward_history):.2f}")

    env.close()
    return phase_history, reward_history


if __name__ == "__main__":
    phase_history, reward_history = run_basic_demo()