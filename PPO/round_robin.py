from traffic_env import TrafficSimulation

def run_round_robin_baseline(
    num_steps=2000,
    phase_duration=20,
):
    """
    Run a single long episode with round-robin signal control and
    report average waiting time and throughput.
    """
    sim = TrafficSimulation()
    sim.reset()

    current_signal = 0        # start at NS-left
    steps_in_phase = 0

    wait_history = []
    throughput_history = []

    for t in range(num_steps):
        # handle round-robin phase changes
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

    print("=== Round-robin baseline results ===")
    print(f"  Steps           : {num_steps}")
    print(f"  Phase duration  : {phase_duration}")
    print(f"  Num of spawned vehicles : {sim.next_vehicle_id}")
    print(f"  Mean wait time  : {mean_wait:.3f}")
    print(f"  Mean throughput : {mean_throughput:.3f}")
    print(f"  Total passed    : {total_passed}")

    return {
        "mean_wait": mean_wait,
        "mean_throughput": mean_throughput,
        "total_passed": total_passed,
    }

if __name__ == "__main__":
    # you can tweak these
    run_round_robin_baseline(num_steps=5000, phase_duration=20)