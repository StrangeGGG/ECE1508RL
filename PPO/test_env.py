from traffic_env import TrafficSimulation

def main():
    sim = TrafficSimulation()
    sim.reset()

    current_signal = 0  # start with NS-left
    steps = 5000

    for t in range(steps):
        # keep signal fixed (0) just to see something moving
        _, metrics = sim.step(t%4)

        avg_wait = metrics["average_waiting_time"]
        throughput = metrics["throughput"]
        total_passed = metrics["total_vehicles_passed"]

        if t % 5 == 0:
            print(
                f"t={t:03d}  "
                f"signal={current_signal}  "
                f"avg_wait={avg_wait:6.3f}  "
                f"throughput={throughput:4.1f}  "
                f"total_passed={total_passed}"
            )

if __name__ == "__main__":
    main()