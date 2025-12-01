# train_ppo.py

from env import TrafficSimulation
from ppo_agent import PPOTrafficLightController


def main():
    sim = TrafficSimulation()

    # You can adjust these
    episodes = 400
    max_steps = 200

    controller = PPOTrafficLightController(simulation=sim, episodes=episodes, max_steps=max_steps)
    controller.train()

    # Save model
    controller.agent.save("results/ppo/ppo_model.pth")


if __name__ == "__main__":
    main()
