from env import TrafficSimulation

class RoundRobinController:
    def __init__(self, phase_durations=None):
        self.phases = [0, 1, 2, 3]
        self.phase_durations = phase_durations or [60, 60, 40, 40]

    def run(self, steps=1000):
        sim = TrafficSimulation()
        sim.reset()

        idx = 0
        timer = 0
        total_reward = 0
        old_metrics = None

        for _ in range(steps):
            phase = self.phases[idx]
            _, metrics = sim.step(phase)

            reward = -(metrics["average_waiting_time"]) - sum(metrics["queue_lengths"].values()) * 0.05
            total_reward += reward

            timer += 1
            if timer >= self.phase_durations[idx]:
                idx = (idx + 1) % len(self.phases)
                timer = 0

            old_metrics = metrics

        return total_reward
