from env import TrafficSimulation

class FixedTimeController:
    def __init__(self, phase_duration=60):
        self.phase_duration = phase_duration

    def run(self, steps=1000):
        sim = TrafficSimulation()
        sim.reset()

        total_reward = 0
        phase = 0
        timer = 0
        old_metrics = None

        for _ in range(steps):
            _, metrics = sim.step(phase)
            reward = -(metrics["average_waiting_time"]) - sum(metrics["queue_lengths"].values()) * 0.05
            total_reward += reward

            timer += 1
            if timer >= self.phase_duration:
                phase = (phase + 1) % 4
                timer = 0

        return total_reward
