from traffic_env import TrafficSimulation

def run_round_robin_baseline(
    num_steps=2000,
    phase_duration=20,
    episode=1,
):
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

    print(f"=== Round-robin baseline episode {episode} results ===")
    print(f"  Steps                            : {num_steps}")
    print(f"  Phase duration                   : {phase_duration}")
    print(f"  Num of spawned vehicles          : {sim.next_vehicle_id}")
    print(f"  Num of current tracking vehicles : {len(sim.vehicles)}")
    print(f"  Total passed vehicles            : {total_passed}")
    print(f"  Mean wait time                   : {mean_wait:.3f}")
    print(f"  Mean throughput                  : {mean_throughput:.3f}")
    

    return {
        "mean_wait": mean_wait,
        "mean_throughput": mean_throughput,
        "total_passed": total_passed,
    }

if __name__ == "__main__":
    # you can tweak these
    num_of_episodes = 50
    num_steps = 5000
    phase_duration = 20
    sum_wait = 0.0
    sum_throughput = 0.0
    sum_passed = 0
    
    for episode in range(num_of_episodes):
        result = run_round_robin_baseline(num_steps=num_steps, phase_duration=phase_duration, episode = episode + 1)
        sum_wait += result['mean_wait']
        sum_throughput += result['mean_throughput']
        sum_passed += result['total_passed']
        
    print(f"=== Round-robin baseline results ===")
    print(f"  Steps                            : {num_steps}")
    print(f"  Phase duration                   : {phase_duration}")
    print(f"  average wait time                   : {sum_wait/num_of_episodes:.3f}")
    print(f"  average throughput                  : {sum_throughput/num_of_episodes:.3f}")    