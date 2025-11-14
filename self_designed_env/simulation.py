import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.patches import Rectangle
from new_env import TrafficSimulation, TrafficMetricsCollector
#from fixedtime_controller import FixedTimeTrafficLight
from fixed_timing_Tianrui import FixedCycleTrafficLight as FixedTimeTrafficLight

class TrafficVisualizer:
    """Traffic visualization class"""

    def __init__(self, simulation, traffic_light):
        self.simulation = simulation
        self.traffic_light = traffic_light
        # Create 1x3 subplots: intersection, waiting time, throughput
        self.fig, (self.ax1, self.ax2, self.ax3) = plt.subplots(1, 3, figsize=(18, 6))
        self.setup_plots()

    def setup_plots(self):
        """Setup plot areas"""
        # Intersection visualization
        self.ax1.set_xlim(-80, 80)
        self.ax1.set_ylim(-80, 80)
        self.ax1.set_aspect('equal')
        self.ax1.set_title('Intersection Simulation - Real-time Traffic Flow')
        self.ax1.grid(True, alpha=0.3)

        # Draw roads and lanes
        self.draw_roads_and_lanes()

        # Waiting time chart
        self.ax2.set_title('Average Waiting Time Over Time')
        self.ax2.set_xlabel('Simulation Steps')
        self.ax2.set_ylabel('Waiting Time (steps)')
        self.ax2.grid(True, alpha=0.3)
        self.ax2.set_ylim(0, 100)  # Set appropriate y-axis range

        # Throughput chart
        self.ax3.set_title('Throughput Over Time')
        self.ax3.set_xlabel('Simulation Steps')
        self.ax3.set_ylabel('Throughput (vehicles/step)')
        self.ax3.grid(True, alpha=0.3)
        self.ax3.set_ylim(0, 0.2)  # Set appropriate y-axis range

        # Initialize metric curves
        self.steps = []
        self.waiting_times = []
        self.throughputs = []

        # Waiting time curve (blue)
        self.waiting_line, = self.ax2.plot([], [], 'b-', linewidth=2, label='Waiting Time')
        self.ax2.legend()

        # Throughput curve (green)
        self.throughput_line, = self.ax3.plot([], [], 'g-', linewidth=2, label='Throughput')
        self.ax3.legend()

        plt.tight_layout()

    def draw_roads_and_lanes(self):
        """Draw roads and lane markings"""
        # Main roads
        self.ax1.add_patch(Rectangle((-80, -30), 160, 60, color='gray', alpha=0.4))
        self.ax1.add_patch(Rectangle((-30, -80), 60, 160, color='gray', alpha=0.4))

        # Lane markings - clearer lane markers
        # North-South direction lanes (4 lanes)
        north_south_lanes = [-20, -7, 7, 20]
        for pos in north_south_lanes:
            # Northbound lanes
            self.ax1.plot([pos, pos], [-80, -30], 'w-', alpha=0.8, linewidth=2)
            # Southbound lanes
            self.ax1.plot([pos, pos], [30, 80], 'w-', alpha=0.8, linewidth=2)

        # East-West direction lanes (4 lanes)
        east_west_lanes = [-20, -7, 7, 20]
        for pos in east_west_lanes:
            # Eastbound lanes
            self.ax1.plot([-80, -30], [pos, pos], 'w-', alpha=0.8, linewidth=2)
            # Westbound lanes
            self.ax1.plot([30, 80], [pos, pos], 'w-', alpha=0.8, linewidth=2)

        # Intersection boundaries
        self.ax1.plot([-30, 30], [-30, -30], 'r-', alpha=0.5, linewidth=2)
        self.ax1.plot([-30, 30], [30, 30], 'r-', alpha=0.5, linewidth=2)
        self.ax1.plot([-30, -30], [-30, 30], 'r-', alpha=0.5, linewidth=2)
        self.ax1.plot([30, 30], [-30, 30], 'r-', alpha=0.5, linewidth=2)

        # Add lane type labels
        self.ax1.text(-22, -70, 'Left', fontsize=8, color='black', ha='center')
        self.ax1.text(-7, -70, 'Straight', fontsize=8, color='black', ha='center')
        self.ax1.text(7, -70, 'Straight', fontsize=8, color='black', ha='center')
        self.ax1.text(22, -70, 'Left', fontsize=8, color='black', ha='center')

        self.ax1.text(-70, -22, 'Left', fontsize=8, color='black', ha='center', rotation=90)
        self.ax1.text(-70, -7, 'Straight', fontsize=8, color='black', ha='center', rotation=90)
        self.ax1.text(-70, 7, 'Straight', fontsize=8, color='black', ha='center', rotation=90)
        self.ax1.text(-70, 22, 'Left', fontsize=8, color='black', ha='center', rotation=90)

    def update(self, frame):
        """Update visualization"""
        # Update traffic light
        current_phase = self.traffic_light.step()

        # Execute simulation step
        observation, metrics = self.simulation.step(current_phase)

        # Clear vehicles
        for artist in self.ax1.collections[:]:
            artist.remove()

        # Draw vehicles
        colors = {'north': 'red', 'south': 'blue', 'east': 'green', 'west': 'orange'}
        shapes = {'straight': 'o', 'left': 's'}

        for vehicle in observation:
            if not vehicle.passed and not vehicle.collided:
                color = colors[vehicle.direction]
                marker = shapes[vehicle.lane_type]
                size = 100 if vehicle.in_intersection_zone else 60
                alpha = 1.0 if vehicle.in_intersection_zone else 0.8

                self.ax1.scatter(vehicle.x, vehicle.y, c=color, marker=marker,
                                 s=size, alpha=alpha, edgecolors='black', linewidth=1)

        # Update traffic light status
        light_status = self.traffic_light.get_status()
        phase_info = f"Phase {light_status['current_phase']}: {light_status['phase_description']}\n"
        phase_info += f"Time left: {light_status['remaining_time']} steps"

        self.ax1.set_title(f'Intersection Simulation:\n{phase_info}\n'
                           f'Step: {self.simulation.current_step}, Vehicles: {len(observation)}',
                           fontsize=10)

        # Update metric charts
        self.steps.append(self.simulation.current_step)
        self.waiting_times.append(metrics['average_waiting_time'])
        self.throughputs.append(metrics['throughput'])

        # Update waiting time chart
        self.waiting_line.set_data(self.steps, self.waiting_times)
        self.ax2.relim()
        self.ax2.autoscale_view()

        # Update throughput chart
        self.throughput_line.set_data(self.steps, self.throughputs)
        self.ax3.relim()
        self.ax3.autoscale_view()

        # Periodic status output
        if self.simulation.current_step % 50 == 0:
            print(f"\nStep {self.simulation.current_step}:")
            print(f"  {light_status['phase_description']}")
            print(f"  Active vehicles: {len(observation)}")
            print(f"  Avg waiting time: {metrics['average_waiting_time']:.2f} steps")
            print(f"  Throughput: {metrics['throughput']:.4f} vehicles/step")
            print(f"  Total passed: {metrics['total_vehicles_passed']}")

        return self.ax1.collections + [self.waiting_line, self.throughput_line]

    def save_plots(self):
        """Save charts to files"""
        # Save waiting time chart
        fig_waiting, ax_waiting = plt.subplots(figsize=(10, 6))
        ax_waiting.plot(self.steps, self.waiting_times, 'b-', linewidth=2, label='Waiting Time')
        ax_waiting.set_title('Average Waiting Time Over Time')
        ax_waiting.set_xlabel('Simulation Steps')
        ax_waiting.set_ylabel('Waiting Time (steps)')
        ax_waiting.grid(True, alpha=0.3)
        ax_waiting.legend()
        plt.tight_layout()
        plt.savefig('waiting_time_plot.png', dpi=300, bbox_inches='tight')
        plt.close(fig_waiting)
        print("Saved waiting_time_plot.png")

        # Save throughput chart
        fig_throughput, ax_throughput = plt.subplots(figsize=(10, 6))
        ax_throughput.plot(self.steps, self.throughputs, 'g-', linewidth=2, label='Throughput')
        ax_throughput.set_title('Throughput Over Time')
        ax_throughput.set_xlabel('Simulation Steps')
        ax_throughput.set_ylabel('Throughput (vehicles/step)')
        ax_throughput.grid(True, alpha=0.3)
        ax_throughput.legend()
        plt.tight_layout()
        plt.savefig('throughput_plot.png', dpi=300, bbox_inches='tight')
        plt.close(fig_throughput)
        print("Saved throughput_plot.png")


def run_simulation():
    """Run complete traffic simulation"""
    print("Starting Custom Traffic Simulation")
    print("=" * 50)
    print("Legend:")
    print("  🟥 Red: Northbound vehicles")
    print("  🟦 Blue: Southbound vehicles")
    print("  🟩 Green: Eastbound vehicles")
    print("  🟧 Orange: Westbound vehicles")
    print("  ○ Circle: Straight lane")
    print("  □ Square: Left turn lane")
    print("=" * 50)

    # Create simulation environment and traffic light
    simulation = TrafficSimulation()
    """ Version from Mingjie"""
    #traffic_light = FixedTimeTrafficLight(phase_durations=[80, 80, 50, 50])
    """ Version from Tianrui"""
    traffic_light = FixedTimeTrafficLight()
    visualizer = TrafficVisualizer(simulation, traffic_light)

    # Run animation
    ani = animation.FuncAnimation(
        visualizer.fig, visualizer.update, frames=1000,
        interval=100, blit=False, repeat=False
    )

    plt.show()

    # Final statistics
    print("\n" + "=" * 50)
    print("SIMULATION COMPLETE")
    print("=" * 50)

    final_metrics = simulation.metrics_collector.get_metrics()
    visualizer.save_plots()

    # Save results to txt file
    with open('simulation_results.txt', 'w', encoding='utf-8') as f:
        f.write("TRAFFIC SIMULATION RESULTS\n")
        f.write("=" * 50 + "\n")
        f.write(f"Total simulation steps: {simulation.current_step}\n")
        f.write(f"Total vehicles passed: {final_metrics['total_vehicles_passed']}\n")
        f.write(f"Final average waiting time: {final_metrics['average_waiting_time']:.2f} steps\n")
        f.write(f"Final throughput: {final_metrics['throughput']:.4f} vehicles/step\n")
        # f.write("\nAverage Queue Lengths by Lane:\n")
        # for lane, length in final_metrics['queue_lengths'].items():
        #     f.write(f"  {lane}: {length:.2f} vehicles\n")

    print("Saved simulation_results.txt")

    # Console output
    print(f"Total simulation steps: {simulation.current_step}")
    print(f"Total vehicles passed: {final_metrics['total_vehicles_passed']}")
    print(f"Final average waiting time: {final_metrics['average_waiting_time']:.2f} steps")
    print(f"Final throughput: {final_metrics['throughput']:.4f} vehicles/step")

    return simulation


if __name__ == "__main__":
    simulation = run_simulation()