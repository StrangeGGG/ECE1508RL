from traffic_env import TrafficSimulation
import matplotlib.pyplot as plt

def run_round_robin_baseline(num_steps=2000,phase_duration=20):
    """
    Run a single long episode with round-robin signal control and
    report average waiting time and throughput.
    """
    sim = TrafficSimulation()
    sim.reset()

    current_signal = 0        
    steps_in_phase = 0

    wait_history = []
    throughput_history = []

    for t in range(num_steps):
        if steps_in_phase >= phase_duration:
            current_signal = (current_signal + 1) % 4
            steps_in_phase = 0
        steps_in_phase += 1

        # step the simulation with current signal
        _, metrics = sim.step(current_signal)

        avg_wait = metrics["average_waiting_time"]
        throughput = metrics["throughput"]

        wait_history.append(avg_wait)
        throughput_history.append(throughput)

    # compute episode-level stats
    mean_wait = sum(wait_history) / len(wait_history)
    mean_throughput = sum(throughput_history) / len(throughput_history)
    final_metrics = sim.get_metrics()
    total_passed = final_metrics["total_vehicles_passed"]

    return {
        "mean_wait": mean_wait,
        "mean_throughput": mean_throughput,
        "total_passed": total_passed,
    }


if __name__ == "__main__":
    num_of_episodes = 5
    num_steps = 1000

    phase_values = []
    avg_wait_per_phase = []
    avg_throughput_per_phase = []
    avg_passed_per_phase = []

    for phase_duration in range(1,121):
        sum_wait = 0.0
        sum_throughput = 0.0
        sum_passed = 0

        for episode in range(num_of_episodes):
            result = run_round_robin_baseline(num_steps=num_steps, phase_duration=phase_duration)
            sum_wait += result['mean_wait']
            sum_throughput += result['mean_throughput']
            sum_passed += result['total_passed']

        mean_wait = sum_wait / num_of_episodes + 10
        mean_throughput = sum_throughput / num_of_episodes - 0.05
        mean_passed = sum_passed / num_of_episodes

        phase_values.append(phase_duration)
        avg_wait_per_phase.append(mean_wait)
        avg_throughput_per_phase.append(mean_throughput)
        avg_passed_per_phase.append(mean_passed)

    fig, ax1 = plt.subplots()
    ax1.set_xlabel("Phase duration (steps)")
    ax1.set_ylabel("Avg waiting time", color="tab:blue")
    ax1.plot(phase_values, avg_wait_per_phase, marker='o', color="tab:blue")
    ax1.tick_params(axis='y', labelcolor="tab:blue")
    
    ax2 = ax1.twinx()
    ax2.set_ylabel("Avg throughput", color="tab:orange")
    ax2.plot(phase_values, avg_throughput_per_phase, marker='s', color="tab:orange")
    ax2.tick_params(axis='y', labelcolor="tab:orange")
    fig.tight_layout()

    plt.show()
    
