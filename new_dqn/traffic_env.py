# traffic_env.py
import random
import numpy as np

# -------------------------
# Vehicle (your original)
# -------------------------
class Vehicle:
    def __init__(self, vehicle_id, direction, lane_type, spawn_time):
        self.id = vehicle_id
        self.direction = direction  # 'north','south','east','west'
        self.lane_type = lane_type  # 'straight','left'
        self.prev_x, self.prev_y = 0.0, 0.0
        self.spawn_time = spawn_time
        self.waiting_time = 0
        self.passed = False
        self.has_turned = False
        self.collided = False
        self.in_intersection_zone = False

        self.set_initial_position()
        self.prev_x, self.prev_y = self.x, self.y
        self.speed = 1.5
        self.max_speed = 2.5
        self.acceleration = 0.1
        self.deceleration = 0.3

    def set_initial_position(self):
        if self.direction == 'north':
            self.x = 7 if not self.lane_type == 'straight' else 20
            self.y = -80
        elif self.direction == 'south':
            self.x = -7 if not self.lane_type == 'straight' else -20
            self.y = 80
        elif self.direction == 'east':
            self.y = -7 if not self.lane_type == 'straight' else -20
            self.x = -80
        elif self.direction == 'west':
            self.y = 7 if not self.lane_type == 'straight' else 20
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
        if self.lane_type != 'left' or self.has_turned:
            return False
        center_tol = 6.0
        lane_tol = 4.0
        if not (abs(self.x) <= 30 and abs(self.y) <= 30):
            return False
        if self.direction == 'north':
            return abs(self.x + 20) < lane_tol and self.y >= -center_tol
        if self.direction == 'south':
            return abs(self.x - 20) < lane_tol and self.y <= center_tol
        if self.direction == 'east':
            return abs(self.y + 20) < lane_tol and self.x >= -center_tol
        if self.direction == 'west':
            return abs(self.y - 20) < lane_tol and self.x <= center_tol
        return False

    def update(self, current_phase, intersection_clear):
        if self.passed or self.collided:
            return
        self.in_intersection_zone = self.in_intersection()
        should_wait = self.should_wait(current_phase)
        if should_wait:
            if self.approaching_intersection():
                self.speed = max(0, self.speed - self.deceleration)
                self.waiting_time += 1
            else:
                self.speed = min(self.max_speed, self.speed + self.acceleration)
        else:
            self.speed = min(self.max_speed, self.speed + self.acceleration)

        if (not should_wait) and self._should_turn_left_now():
            self._turn_left()
        self.move()
        self.check_passed()

    def should_wait(self, current_phase):
        if not self.approaching_intersection():
            return False
        if current_phase == 0:
            return not (self.direction in ['north', 'south'] and self.lane_type == 'straight')
        elif current_phase == 1:
            return not (self.direction in ['east', 'west'] and self.lane_type == 'straight')
        elif current_phase == 2:
            return not (self.direction in ['north', 'south'] and self.lane_type == 'left')
        elif current_phase == 3:
            return not (self.direction in ['east', 'west'] and self.lane_type == 'left')
        return True

    def approaching_intersection(self):
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
        return abs(self.x) <= 30 and abs(self.y) <= 30

    def move(self):
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
        if self.direction in ['north', 'south']:
            if (self.direction == 'north' and self.y > 60) or (self.direction == 'south' and self.y < -60):
                self.passed = True
        else:
            if (self.direction == 'east' and self.x > 60) or (self.direction == 'west' and self.x < -60):
                self.passed = True

    def _turn_left(self):
        if self.direction == 'north':
            self.direction = 'west'; self.x, self.y = -5.0, 20.0
        elif self.direction == 'south':
            self.direction = 'east'; self.x, self.y = 5.0, -20.0
        elif self.direction == 'east':
            self.direction = 'north'; self.x, self.y = -20.0, -5.0
        elif self.direction == 'west':
            self.direction = 'south'; self.x, self.y = 20.0, 5.0
        self.has_turned = True

# -------------------------
# TrafficMetricsCollector (your original)
# -------------------------
class TrafficMetricsCollector:
    def __init__(self):
        self.reset()

    def reset(self):
        self.instant_waiting_times_history = []
        self.throughputs = []
        self.instant_queue_lengths = {
            'north_straight': 0, 'north_left': 0,
            'south_straight': 0, 'south_left': 0,
            'east_straight': 0, 'east_left': 0,
            'west_straight': 0, 'west_left': 0
        }
        self.total_passed_vehicles = 0
        self.total_steps = 0
        self.previous_total_wait_time = 0
        self.wait_time_increase = 0

    def update_metrics(self, vehicles_info, current_phase, current_step, passed_this_step, total_passed):
        self.total_steps = current_step
        self.total_passed_vehicles = total_passed
        current_total_wait_time = 0
        instantaneous_queue_counts = {key: 0 for key in self.instant_queue_lengths.keys()}
        active_vehicle_wait_times = []

        for vehicle_id, vehicle_data in vehicles_info.items():
            if vehicle_data['passed'] or vehicle_data['collided']:
                continue
            active_vehicle_wait_times.append(vehicle_data['waiting_time'])
            current_total_wait_time += vehicle_data['waiting_time']
            if vehicle_data['speed'] < 0.5 and self.is_in_waiting_zone(vehicle_data):
                lane_key = f"{vehicle_data['direction']}_{vehicle_data['lane_type']}"
                if lane_key in instantaneous_queue_counts:
                    instantaneous_queue_counts[lane_key] += 1

        self.instant_queue_lengths = instantaneous_queue_counts
        self.wait_time_increase = current_total_wait_time - self.previous_total_wait_time
        self.previous_total_wait_time = current_total_wait_time
        avg_wait_time = np.mean(active_vehicle_wait_times) if active_vehicle_wait_times else 0
        self.instant_waiting_times_history.append(avg_wait_time)
        if current_step > 0:
            current_throughput = total_passed / current_step
            self.throughputs.append(current_throughput)

    def is_in_waiting_zone(self, vehicle_data):
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
        metrics = {}
        metrics['average_waiting_time'] = self.instant_waiting_times_history[-1] if self.instant_waiting_times_history else 0
        metrics['queue_lengths'] = self.instant_queue_lengths
        metrics['wait_time_increase'] = self.wait_time_increase
        metrics['throughput'] = self.throughputs[-1] if self.throughputs else 0
        metrics['total_vehicles_passed'] = self.total_passed_vehicles
        metrics['current_vehicles'] = len(self.instant_waiting_times_history)
        return metrics

# -------------------------
# TrafficSimulation (your original)
# -------------------------
class TrafficSimulation:
    def __init__(self):
        self.vehicles = {}
        self.next_vehicle_id = 0
        self.current_step = 0
        self.metrics_collector = TrafficMetricsCollector()
        self.spawn_probabilities = {
            'north': {'straight': 0.02, 'left': 0.01},
            'south': {'straight': 0.02, 'left': 0.01},
            'east': {'straight': 0.02, 'left': 0.01},
            'west': {'straight': 0.02, 'left': 0.01}
        }
        self.total_passed_vehicles = 0

    def reset(self):
        self.vehicles = {}
        self.next_vehicle_id = 0
        self.current_step = 0
        self.total_passed_vehicles = 0
        self.metrics_collector.reset()

    def step(self, current_phase):
        self.current_step += 1
        self.spawn_vehicles()
        vehicles_to_remove = []
        passed_count = 0
        for vehicle_id, vehicle in list(self.vehicles.items()):
            vehicle.update(current_phase, True)
            if vehicle.passed:
                vehicles_to_remove.append(vehicle_id)
                passed_count += 1
        for vehicle_id in vehicles_to_remove:
            del self.vehicles[vehicle_id]
        self.total_passed_vehicles += passed_count
        self.collect_metrics(current_phase, passed_count)
        return self.get_observation(), self.get_metrics()

    def spawn_vehicles(self):
        for direction in ['north', 'south', 'east', 'west']:
            for lane_type in ['straight', 'left']:
                prob = self.spawn_probabilities[direction][lane_type]
                if random.random() < prob:
                    lane_vehicles = [v for v in self.vehicles.values()
                                     if v.direction == direction and v.lane_type == lane_type
                                     and not v.passed and not v.collided]
                    if len(lane_vehicles) < 6:
                        self.add_vehicle(direction, lane_type)

    def add_vehicle(self, direction, lane_type):
        vehicle = Vehicle(self.next_vehicle_id, direction, lane_type, self.current_step)
        self.vehicles[self.next_vehicle_id] = vehicle
        self.next_vehicle_id += 1

    def collect_metrics(self, current_phase, passed_this_step):
        vehicles_info = {}
        for vehicle_id, vehicle in self.vehicles.items():
            vehicles_info[vehicle_id] = {
                'x': vehicle.x, 'y': vehicle.y,
                'direction': vehicle.direction,
                'lane_type': vehicle.lane_type,
                'waiting_time': vehicle.waiting_time,
                'speed': vehicle.speed,
                'passed': vehicle.passed,
                'collided': vehicle.collided
            }
        self.metrics_collector.update_metrics(vehicles_info, current_phase,
                                              self.current_step, passed_this_step,
                                              self.total_passed_vehicles)

    def get_observation(self):
        return list(self.vehicles.values())

    def get_metrics(self):
        return self.metrics_collector.get_metrics()

# -------------------------
# Gym-like wrapper that enforces phase locking and returns RL state/reward
# -------------------------
class TrafficRLWrapper:
    """
    Wrapper around TrafficSimulation to be used by DQN.
    - Enforces a minimum phase duration (min_phase_steps)
    - Maintains last_action and force_action_steps to avoid flickering
    - Builds a numeric state vector (29-dim: per-lane features + one-hot phase + phase duration)
    - Computes stable, clipped rewards
    """

    def __init__(self, sim: TrafficSimulation = None, min_phase_steps=8, max_phase_steps=60):
        self.sim = sim if sim is not None else TrafficSimulation()
        self.action_size = 4
        self.min_phase_steps = min_phase_steps
        self.max_phase_steps = max_phase_steps

        # action locking state
        self.last_action = 0
        self.force_action_steps = 0
        self.current_phase = 0
        self.phase_duration = 0

        # last metrics used for delta computations
        self.last_metrics = {'total_vehicles_passed': 0}
        self.reset()

    def reset(self):
        self.sim.reset()
        self.current_phase = random.randrange(self.action_size)
        self.last_action = self.current_phase
        self.force_action_steps = self.min_phase_steps
        self.phase_duration = 0
        self.last_metrics = {'total_vehicles_passed': 0}
        return self.get_state()

    def step(self, action):
        # ----------------------
        # 1) action -> enforce min_phase_steps locking (happens BEFORE sim.step)
        # ----------------------
        if self.force_action_steps > 0:
            # still locked: override action with last_action
            executed_action = self.last_action
            self.force_action_steps -= 1
        else:
            # not locked: if change requested, lock for min_phase_steps
            if action != self.last_action:
                executed_action = action
                self.force_action_steps = self.min_phase_steps
            else:
                executed_action = action

        # update phase bookkeeping (phase changes are allowed only via wrapper)
        if executed_action != self.current_phase:
            self.current_phase = executed_action
            self.phase_duration = 0
        else:
            self.phase_duration += 1
            # if exceeding max, we won't force a change here; reward will penalize long phases

        self.last_action = executed_action

        # ----------------------
        # 2) run the micro-simulator with executed phase
        # ----------------------
        _, metrics = self.sim.step(self.current_phase)

        # ----------------------
        # 3) compute reward (stable, clipped)
        # ----------------------
        reward = self.compute_reward(self.last_metrics, metrics)
        self.last_metrics = metrics.copy()

        # ----------------------
        # 4) prepare next_state and done
        # ----------------------
        next_state = self.get_state()
        done = False  # episodes will be handled by trainer's loop (or you can set a horizon)
        info = {'phase': self.current_phase, 'phase_duration': self.phase_duration}
        return next_state, reward, done, info

    def get_state(self):
        """
        Build a 29-dim state:
        For each of 8 lanes: [queue_len_norm, avg_wait_norm, vehicle_count_norm] => 24 dims
        + 4-d one-hot current phase
        + 1-d normalized phase duration -> total 29
        """
        state = []
        MAX_VALUE = 300.0
        MAX_QUEUE = 20.0

        # lanes in fixed order
        for direction in ['north', 'south', 'east', 'west']:
            for lane_type in ['straight', 'left']:
                lane_vehicles = [v for v in self.sim.vehicles.values()
                                 if v.direction == direction and v.lane_type == lane_type
                                 and not v.passed and not v.collided]
                avg_wait = (np.mean([v.waiting_time for v in lane_vehicles]) if lane_vehicles else 0.0)
                queue_len = len([v for v in lane_vehicles if v.speed < 0.5])
                vehicle_count = len(lane_vehicles)
                state.extend([
                    min(queue_len / MAX_QUEUE, 1.0),
                    min(avg_wait / MAX_VALUE, 1.0),
                    min(vehicle_count / MAX_QUEUE, 1.0)
                ])
        # phase one-hot
        ph = [0] * self.action_size
        ph[self.current_phase] = 1
        state.extend(ph)
        state.append(self.phase_duration / float(self.max_phase_steps))
        return np.array(state, dtype=np.float32)

    def compute_reward(self, last_metrics, metrics):
        """
        Stable reward:
         - penalize total instantaneous queue length
         - penalize increase in total waiting time
         - small positive reward for throughput (vehicles passed)
         - penalize if phase_duration > max_phase_steps
         - finally clip to [-5, +5]
        """

        # Default: Original presets
        # A: Balanced presets
        # B: Throughput-focused presets
        # C: Waiting-Time minimizer presets

        q_dict = metrics.get('queue_lengths', {})
        total_queue = sum(q_dict.values()) if q_dict else 0
        #queue_penalty = -0.06 * total_queue 
        #queue_penalty = -0.05 * total_queue # Default
        #queue_penalty = -0.015 * total_queue # A
        #queue_penalty = -0.01 * total_queue # B
        queue_penalty = -0.02 * total_queue # C

        wait_inc = metrics.get('wait_time_increase', 0)
        #wait_penalty = -0.02 * wait_inc # Default
        #wait_penalty = -0.04 * wait_inc # A
        #wait_penalty = -0.025 * wait_inc # B
        wait_penalty = -0.06 * wait_inc # C
        

        passed_old = last_metrics.get('total_vehicles_passed', 0)
        passed_new = metrics.get('total_vehicles_passed', 0)
        #passed_reward = (passed_new - passed_old) * 1.0 # Default
        #passed_reward = (passed_new - passed_old) * 1.5 # A
        #passed_reward = (passed_new - passed_old) * 2.5 # B
        passed_reward = (passed_new - passed_old) * 1.0 # C

        phase_penalty = -1.0 if self.phase_duration > self.max_phase_steps else 0.0

        reward = queue_penalty + wait_penalty + passed_reward + phase_penalty
        reward = float(np.clip(reward, -5.0, 5.0))
        return reward