import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import random
from collections import deque


class Vehicle:
    """Vehicle class representing a car in the simulation"""

    def __init__(self, vehicle_id, direction, lane_type, spawn_time):
        self.id = vehicle_id
        self.direction = direction  # 'north', 'south', 'east', 'west'
        self.lane_type = lane_type  # 'straight', 'left'
        self.prev_x, self.prev_y = 0.0, 0.0
        self.spawn_time = spawn_time
        self.waiting_time = 0
        self.passed = False
        self.has_turned = False
        self.collided = False
        self.in_intersection_zone = False

        # Initial position and speed
        self.set_initial_position()
        self.prev_x, self.prev_y = self.x, self.y
        self.speed = 1.5  # Initial speed
        self.max_speed = 2.5
        self.acceleration = 0.1
        self.deceleration = 0.3

    def set_initial_position(self):
        """Set initial position based on direction and lane type"""
        if self.direction == 'north':
            if self.lane_type == 'straight':
                self.x = -7  # Northbound straight lane
            else:
                self.x = -20  # Northbound left turn lane
            self.y = -80
        elif self.direction == 'south':
            if self.lane_type == 'straight':
                self.x = 7  # Southbound straight lane
            else:
                self.x = 20  # Southbound left turn lane
            self.y = 80
        elif self.direction == 'east':
            if self.lane_type == 'straight':
                self.y = -7  # Eastbound straight lane
            else:
                self.y = -20  # Eastbound left turn lane
            self.x = -80
        elif self.direction == 'west':
            if self.lane_type == 'straight':
                self.y = 7  # Westbound straight lane
            else:
                self.y = 20  # Westbound left turn lane
            self.x = 80

    def _crossed_centerline(self) -> bool:
        if self.direction == 'north':
            return self.prev_y < 0 <= self.y
        if self.direction == 'south':
            return self.prev_y > 0 >= self.y
        if self.direction == 'east':
            return self.prev_x < 0 <= self.x
        if self.direction == 'west':
            return self.prev_x > 0 >= self.x
        return False

    def _should_turn_left_now(self) -> bool:
        """Decide when to perform the instantaneous left turn for left-lane vehicles."""
        if self.lane_type != 'left' or self.has_turned:
            return False

        # Tolerances
        center_tol = 6.0     # closeness to intersection centerline along travel axis
        lane_tol = 4.0       # closeness to the left-lane offset

        if not (abs(self.x) <= 30 and abs(self.y) <= 30):
            return False
        
        if self.direction == 'north':   # left lane ~ x = -20; turn when near/after y≈0
            return abs(self.x + 20) < lane_tol and self.y >= -center_tol
        if self.direction == 'south':   # left lane ~ x = +20; turn when near/after y≈0
            return abs(self.x - 20) < lane_tol and self.y <=  center_tol
        if self.direction == 'east':    # left lane ~ y = -20; turn when near/after x≈0
            return abs(self.y + 20) < lane_tol and self.x >= -center_tol
        if self.direction == 'west':    # left lane ~ y = +20; turn when near/after x≈0
            return abs(self.y - 20) < lane_tol and self.x <=  center_tol
        return False

    def update(self, current_phase, intersection_clear):
        """Update vehicle state"""
        if self.passed or self.collided:
            return

        # Update intersection zone status
        self.in_intersection_zone = self.in_intersection()

        # Check if vehicle should wait
        should_wait = self.should_wait(current_phase)

        if should_wait:
            # Decelerate or stop when approaching intersection
            if self.approaching_intersection():
                self.speed = max(0, self.speed - self.deceleration)
                self.waiting_time += 1
            else:
                # Not in waiting zone, drive normally
                self.speed = min(self.max_speed, self.speed + self.acceleration)
        else:
            # Can proceed, accelerate
            self.speed = min(self.max_speed, self.speed + self.acceleration)

        # If this is a left-turn vehicle, turn once when it reaches its lane's centerline
        if (not should_wait) and self._should_turn_left_now():
            self._turn_left()

        # Update position
        self.move()

        # Check if passed intersection
        self.check_passed()

    def should_wait(self, current_phase):
        """Determine if vehicle should wait based on current traffic light phase"""
        # If not approaching intersection, no need to wait
        if not self.approaching_intersection():
            return False

        # Determine based on phase
        if current_phase == 0:  # North-South straight green
            return not (self.direction in ['north', 'south'] and self.lane_type == 'straight')

        elif current_phase == 1:  # East-West straight green
            return not (self.direction in ['east', 'west'] and self.lane_type == 'straight')

        elif current_phase == 2:  # North-South left turn green
            return not (self.direction in ['north', 'south'] and self.lane_type == 'left')

        elif current_phase == 3:  # East-West left turn green
            return not (self.direction in ['east', 'west'] and self.lane_type == 'left')

        # Default to waiting
        return True

    def approaching_intersection(self):
        """Check if vehicle is approaching the intersection"""
        if self.direction == 'north':
            return self.y < -10
        elif self.direction == 'south':
            return self.y > 10
        elif self.direction == 'east':
            return self.x < -10
        elif self.direction == 'west':
            return self.x > 10
        return False

    def in_intersection(self):
        """Check if vehicle is in intersection area"""
        return abs(self.x) <= 30 and abs(self.y) <= 30

    def move(self):
        """Move vehicle based on direction"""
        if self.direction == 'north':
            self.y += self.speed
        elif self.direction == 'south':
            self.y -= self.speed
        elif self.direction == 'east':
            self.x += self.speed
        elif self.direction == 'west':
            self.x -= self.speed
        
        self.prev_x, self.prev_y = self.x, self.y

    def check_passed(self):
        """Check if vehicle has passed the intersection"""
        if self.direction in ['north', 'south']:
            if (self.direction == 'north' and self.y > 60) or \
                    (self.direction == 'south' and self.y < -60):
                self.passed = True
        else:
            if (self.direction == 'east' and self.x > 60) or \
                    (self.direction == 'west' and self.x < -60):
                self.passed = True

    def _turn_left(self):
        """Instant left into a safe point aligned with the outbound left-turn lane."""
        # pick a point slightly inside the box so we don't sweep through straight queues
        if self.direction == 'north':        # to WEST
            self.direction = 'west'
            self.x, self.y = -5.0, 20.0      # x near center, y on westbound-left lane
        elif self.direction == 'south':      # to EAST
            self.direction = 'east'
            self.x, self.y =  5.0, -20.0
        elif self.direction == 'east':       # to NORTH
            self.direction = 'north'
            self.x, self.y = -20.0, -5.0
        elif self.direction == 'west':       # to SOUTH
            self.direction = 'south'
            self.x, self.y =  20.0,  5.0
        self.has_turned = True

class TrafficMetricsCollector:
    """Collects traffic performance metrics"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.waiting_times = []
        self.throughputs = []
        self.queue_lengths = {
            'north_straight': deque(maxlen=100),
            'north_left': deque(maxlen=100),
            'south_straight': deque(maxlen=100),
            'south_left': deque(maxlen=100),
            'east_straight': deque(maxlen=100),
            'east_left': deque(maxlen=100),
            'west_straight': deque(maxlen=100),
            'west_left': deque(maxlen=100)
        }
        self.total_passed_vehicles = 0
        self.collision_count = 0
        self.total_steps = 0

    def update_metrics(self, vehicles_info, current_phase, current_step,
                       passed_this_step, total_passed):
        """Update all metrics"""
        self.total_steps = current_step
        self.total_passed_vehicles = total_passed

        # Reset queue length counts
        queue_counts = {key: 0 for key in self.queue_lengths.keys()}
        current_waiting_times = []

        for vehicle_id, vehicle_data in vehicles_info.items():
            if vehicle_data['passed']:
                continue

            # Collect waiting times
            current_waiting_times.append(vehicle_data['waiting_time'])

            # Calculate queue length (low speed vehicles in waiting zone)
            if vehicle_data['speed'] < 0.5 and self.is_in_waiting_zone(vehicle_data):
                lane_key = f"{vehicle_data['direction']}_{vehicle_data['lane_type']}"
                if lane_key in queue_counts:
                    queue_counts[lane_key] += 1

        # Update queue history
        for lane_key, count in queue_counts.items():
            self.queue_lengths[lane_key].append(count)

        # Update waiting times
        if current_waiting_times:
            self.waiting_times.append(np.mean(current_waiting_times))
        else:
            self.waiting_times.append(0)

        # Calculate current throughput
        if current_step > 0:
            current_throughput = total_passed / current_step
            self.throughputs.append(current_throughput)

    def is_in_waiting_zone(self, vehicle_data):
        """Check if vehicle is in waiting zone"""
        x, y = vehicle_data['x'], vehicle_data['y']
        direction = vehicle_data['direction']

        if direction == 'north':
            return -40 < y < -15
        elif direction == 'south':
            return 15 < y < 40
        elif direction == 'east':
            return -40 < x < -15
        elif direction == 'west':
            return 15 < x < 40
        return False

    def get_metrics(self):
        """Get comprehensive metrics"""
        metrics = {}

        # Average waiting time
        metrics['average_waiting_time'] = np.mean(self.waiting_times) if self.waiting_times else 0

        # Average queue lengths by direction
        metrics['queue_lengths'] = {}
        for direction in ['north', 'south', 'east', 'west']:
            for lane_type in ['straight', 'left']:
                key = f'{direction}_{lane_type}'
                metrics['queue_lengths'][key] = np.mean(self.queue_lengths[key]) if self.queue_lengths[key] else 0

        # Throughput
        metrics['throughput'] = self.throughputs[-1] if self.throughputs else 0
        metrics['total_vehicles_passed'] = self.total_passed_vehicles

        # Current active vehicles
        metrics['current_vehicles'] = len(self.waiting_times)

        return metrics


class TrafficSimulation:
    """Traffic simulation environment"""

    def __init__(self):
        self.vehicles = {}
        self.next_vehicle_id = 0
        self.current_step = 0
        self.metrics_collector = TrafficMetricsCollector()

        # Vehicle spawn probabilities
        self.spawn_probabilities = {
            'north': {'straight': 0.04, 'left': 0.03},
            'south': {'straight': 0.04, 'left': 0.03},
            'east': {'straight': 0.04, 'left': 0.03},
            'west': {'straight': 0.04, 'left': 0.03}
        }

        # Track passed vehicles
        self.total_passed_vehicles = 0

    def reset(self):
        """Reset simulation"""
        self.vehicles = {}
        self.next_vehicle_id = 0
        self.current_step = 0
        self.total_passed_vehicles = 0
        self.metrics_collector.reset()

    def step(self, current_phase):
        """Execute one simulation step"""
        self.current_step += 1

        # Spawn new vehicles
        self.spawn_vehicles()

        # Update all vehicles
        vehicles_to_remove = []
        passed_count = 0

        for vehicle_id, vehicle in self.vehicles.items():
            vehicle.update(current_phase, True)
            if vehicle.passed:
                vehicles_to_remove.append(vehicle_id)
                passed_count += 1

        # Remove passed vehicles and update statistics
        for vehicle_id in vehicles_to_remove:
            del self.vehicles[vehicle_id]
        self.total_passed_vehicles += passed_count

        # Collect metrics
        self.collect_metrics(current_phase, passed_count)

        return self.get_observation(), self.get_metrics()

    def spawn_vehicles(self):
        """Spawn new vehicles based on probability"""
        for direction in ['north', 'south', 'east', 'west']:
            for lane_type in ['straight', 'left']:
                prob = self.spawn_probabilities[direction][lane_type]
                if random.random() < prob:
                    # Check if lane already has waiting vehicles
                    lane_vehicles = [v for v in self.vehicles.values()
                                     if v.direction == direction and v.lane_type == lane_type
                                     and not v.passed and not v.collided]
                    if len(lane_vehicles) < 4:
                        self.add_vehicle(direction, lane_type)

    def add_vehicle(self, direction, lane_type):
        """Add new vehicle to simulation"""
        vehicle = Vehicle(self.next_vehicle_id, direction, lane_type, self.current_step)
        self.vehicles[self.next_vehicle_id] = vehicle
        self.next_vehicle_id += 1

    def collect_metrics(self, current_phase, passed_this_step):
        """Collect traffic metrics"""
        vehicles_info = {}
        for vehicle_id, vehicle in self.vehicles.items():
            vehicles_info[vehicle_id] = {
                'x': vehicle.x, 'y': vehicle.y,
                'direction': vehicle.direction,
                'lane_type': vehicle.lane_type,
                'waiting_time': vehicle.waiting_time,
                'speed': vehicle.speed,
                'passed': vehicle.passed
            }

        self.metrics_collector.update_metrics(vehicles_info, current_phase,
                                              self.current_step, passed_this_step,
                                              self.total_passed_vehicles)

    def get_observation(self):
        """Get current observation state"""
        return list(self.vehicles.values())

    def get_metrics(self):
        """Get current metrics"""
        return self.metrics_collector.get_metrics()