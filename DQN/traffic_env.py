import random
import numpy as np

class Vehicle:
    def __init__(self, vehicle_id, direction, lane_type, spawn_time):
        self.id = vehicle_id
        self.direction = direction  # 'north','south','east','west'
        self.lane_type = lane_type  # 'straight' or 'left'
        self.spawn_time = spawn_time

        self.set_initial_position()

        self.speed = 1.5
        self.max_speed = 2.5
        self.acceleration = 0.1
        self.deceleration = 0.3

        # stats
        self.waiting_time = 0
        self.passed = False
        self.has_turned = False
        self.collided = False


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

    def distance_to_stopline(self):
        if self.direction == 'north':
            return max(0.0, 30.0 - self.y)
        elif self.direction == 'south':
            return max(0.0, self.y + 30.0)
        elif self.direction == 'east':
            return max(0.0, 30.0 - self.x)
        elif self.direction == 'west':
            return max(0.0, self.x + 30.0)
        return 1e6

    def approaching_intersection(self):
        """Check if vehicle is approaching the intersection"""
        if self.direction == 'north':
            return self.y < -40
        elif self.direction == 'south':
            return self.y > 40
        elif self.direction == 'east':
            return self.x < -40
        elif self.direction == 'west':
            return self.x > 40
        return False

    def in_intersection(self):
        return abs(self.x) <= 30.0 and abs(self.y) <= 30.0

    def should_stop_for_red(self, current_phase):
        if not self.approaching_intersection():
            return False
        if current_phase == 0:
            return not (self.direction in ['north', 'south'] and self.lane_type == 'straight')
        if current_phase == 1:
            return not (self.direction in ['east', 'west'] and self.lane_type == 'straight')
        if current_phase == 2:
            return not (self.direction in ['north', 'south'] and self.lane_type == 'left')
        if current_phase == 3:
            return not (self.direction in ['east', 'west'] and self.lane_type == 'left')
        return True

    def update(self, current_phase):
        if self.passed or self.collided:
            return

        should_wait = self.should_stop_for_red(current_phase)

        if should_wait and self.approaching_intersection():
            self.speed = max(0, self.speed - self.deceleration)
            if self.speed < 0.5:
                self.waiting_time += 1
        else:
            self.speed = min(self.max_speed, self.speed + self.acceleration)
            if self._should_turn_left_now():
                self._turn_left()
            self.move()

        self.check_passed()

    def move(self):
        if self.direction == 'north':
            self.y += self.speed
        elif self.direction == 'south':
            self.y -= self.speed
        elif self.direction == 'east':
            self.x += self.speed
        elif self.direction == 'west':
            self.x -= self.speed

    def check_passed(self):
        if self.direction == 'north' and self.y > 60:
            self.passed = True
        elif self.direction == 'south' and self.y < -60:
            self.passed = True
        elif self.direction == 'east' and self.x > 60:
            self.passed = True
        elif self.direction == 'west' and self.x < -60:
            self.passed = True

    def _should_turn_left_now(self):
        if self.lane_type != 'left' or self.has_turned:
            return False
        return abs(self.x) <= 20.0 and abs(self.y) <= 20.0

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

# Ideal Environment without peak/offpeak distinction & noise   
class TrafficEngine_ideal:
    def __init__(self, min_phase_steps=4, max_phase_steps=40, seed=None):
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)

        self.vehicles = {}
        self.next_vehicle_id = 0
        self.current_step = 0
        self.total_passed_vehicles = 0

        # spawn probabilities (per step per lane)
        default_prob = {
            'north': {'straight': 0.05, 'left': 0.02},
            'south': {'straight': 0.05, 'left': 0.02},
            'east': {'straight': 0.05, 'left': 0.02},
            'west': {'straight': 0.05, 'left': 0.02}
        }
        self.spawn_probabilities = default_prob

        # RL-related
        self.action_size = 4
        self.current_phase = 0
        self.phase_duration = 0
        self.force_action_steps = 0
        self.min_phase_steps = min_phase_steps
        self.max_phase_steps = max_phase_steps

        # last metrics
        self.last_metrics = {'total_vehicles_passed': 0, 'avg_wait': 0.0, 'throughput': 0.0}

    # ------------------ API ------------------
    def reset(self):
        self.vehicles = {}
        self.next_vehicle_id = 0
        self.current_step = 0
        self.total_passed_vehicles = 0
        self.current_phase = 0
        self.phase_duration = 0
        self.force_action_steps = 0
        self.last_metrics = {'total_vehicles_passed': 0, 'avg_wait': 0.0, 'throughput': 0.0}
        return self.get_state()

    def step(self, action):
        # enforce lock
        if self.force_action_steps > 0:
            executed_action = self.current_phase
            self.force_action_steps -= 1
        else:
            if action != self.current_phase:
                executed_action = action
                self.force_action_steps = self.min_phase_steps - 1
                self.phase_duration = 0
            else:
                executed_action = action

        if executed_action != self.current_phase:
            self.current_phase = executed_action
            self.phase_duration = 0
        else:
            self.phase_duration += 1

        # advance simulation
        self.current_step += 1
        self._spawn_vehicles()

        passed_this_step = 0
        for vid, v in list(self.vehicles.items()):
            v.update(self.current_phase)
            if v.passed:
                passed_this_step += 1

        # remove passed/collided
        for vid in [vid for vid, v in list(self.vehicles.items()) if v.passed or v.collided]:
            del self.vehicles[vid]

        self.total_passed_vehicles += passed_this_step

        # metrics
        metrics = self._compute_metrics(passed_this_step)
        reward = self._compute_reward(self.last_metrics, metrics)
        self.last_metrics = metrics.copy()

        next_state = self.get_state()
        done = False
        info = {
            'phase': self.current_phase,
            'phase_duration': self.phase_duration,
            'avg_wait': metrics['avg_wait'],
            'throughput': metrics['throughput'],
            'total_wait': metrics['total_wait'],
            'current_vehicles': metrics['current_vehicles']
        }
        return next_state, reward, done, info
        
    # ------------------ Simulation helpers ------------------
    def _spawn_vehicles(self):
        current_probs = self.spawn_probabilities
        for direction in ['north', 'south', 'east', 'west']:
            for lane_type in ['straight', 'left']:
                prob = current_probs[direction][lane_type]
                if random.random() < prob:
                    lane_vehicles = [v for v in self.vehicles.values() if v.direction == direction and v.lane_type == lane_type and not v.passed]
                    if len(lane_vehicles) < 6:
                        self._add_vehicle(direction, lane_type)

    def _add_vehicle(self, direction, lane_type):
        v = Vehicle(self.next_vehicle_id, direction, lane_type, self.current_step)
        self.vehicles[self.next_vehicle_id] = v
        self.next_vehicle_id += 1

    def _is_in_waiting_zone(self, vehicle):
        x, y = vehicle.x, vehicle.y
        direction = vehicle.direction

        if direction == 'north':
            return -40 < y < -15
        elif direction == 'south':
            return 15 < y < 40
        elif direction == 'east':
            return -40 < x < -15
        elif direction == 'west':
            return 15 < x < 40
        return False

    # ------------------ Metrics & reward ------------------
    def _compute_metrics(self, passed_this_step):
        queue_lengths = {
            'north_straight': 0, 'north_left': 0,
            'south_straight': 0, 'south_left': 0,
            'east_straight': 0, 'east_left': 0,
            'west_straight': 0, 'west_left': 0
        }

        wait_times = []
        total_wait = 0

        for vid, v in self.vehicles.items():
            if v.passed or v.collided:
                continue

            lane_key = f"{v.direction}_{v.lane_type}"

            # queue length
            if v.speed < 0.5 and self._is_in_waiting_zone(v):
                queue_lengths[lane_key] = queue_lengths.get(lane_key, 0) + 1

            # waiting metrics
            wait_times.append(v.waiting_time)
            total_wait += v.waiting_time

        avg_wait = float(np.mean(wait_times)) if wait_times else 0.0

        # throughput
        if self.current_step > 0:
            throughput = float(self.total_passed_vehicles) / self.current_step
        else:
            throughput = 0.0

        metrics = {
            'queue_lengths': queue_lengths,
            'avg_wait': avg_wait,
            'total_wait': float(total_wait),
            'throughput': throughput,
            'total_vehicles_passed': self.total_passed_vehicles,
            'current_vehicles': len(wait_times)
        }
        return metrics


    def _compute_reward(self, last_metrics, metrics):
        qdict = metrics.get('queue_lengths', {})
        total_queue = sum(qdict.values()) if qdict else 0

        current_total_wait = metrics.get('total_wait', 0)
        last_total_wait = last_metrics.get('total_wait', 0)
        wait_increase = current_total_wait - last_total_wait

        current_throughput = metrics.get('throughput', 0)
        last_throughput = last_metrics.get('throughput', 0)
        throughput_improvement = current_throughput - last_throughput

        # reward = (-0.05 * float(total_queue)
        #           - 0.005 * wait_increase
        #           + 0.5 * metrics['throughput'])
        reward = (-0.05 * float(total_queue)
                  - 0.005 * wait_increase
                  + 1.0 * throughput_improvement
                  + 0.5 * current_throughput)

        return float(np.clip(reward, -2.0, 2.0))

    # ------------------ State ------------------
    def get_state(self):
        state = []
        MAX_QUEUE = 10.0
        MAX_DISTANCE = 120.0

        for direction in ['north', 'south', 'east', 'west']:
            for lane_type in ['straight', 'left']:
                lane_vehicles = [v for v in self.vehicles.values() if v.direction == direction and v.lane_type == lane_type and not v.passed]
                # queue: stopped cars near stopline
                queue_len = len([v for v in lane_vehicles if v.speed < 0.1 and v.distance_to_stopline() < 10.0])
                queue_norm = min(queue_len / MAX_QUEUE, 1.0)

                if lane_vehicles:
                    dists = [v.distance_to_stopline() for v in lane_vehicles]
                    nearest = max(min(dists), 0.0)
                    dist_norm = min(nearest / MAX_DISTANCE, 1.0)
                else:
                    dist_norm = 1.0

                state.extend([queue_norm, dist_norm])

        ph = [0.0] * self.action_size
        ph[self.current_phase] = 1.0
        state.extend(ph)

        return np.array(state, dtype=np.float32)

    # ------------------ Convenience ------------------
    def get_metrics(self):
        return self.last_metrics.copy()

   # Realistic Environment with peak/offpeak distinction & noise  
class TrafficEngine_realistic:
    def __init__(self, peak_steps=None, min_phase_steps=4, max_phase_steps=40, seed=None, sensor_noise_std=0.5):
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)

        self.vehicles = {}
        self.next_vehicle_id = 0
        self.current_step = 0
        self.total_passed_vehicles = 0

        # spawn probabilities (per step per lane)
        default_peak = {
            'north': {'straight': 0.1, 'left': 0.05},
            'south': {'straight': 0.1, 'left': 0.05},
            'east': {'straight': 0.1, 'left': 0.05},
            'west': {'straight': 0.1, 'left': 0.05}
        }
        default_offpeak = {
            'north': {'straight': 0.05, 'left': 0.02},
            'south': {'straight': 0.05, 'left': 0.02},
            'east': {'straight': 0.05, 'left': 0.02},
            'west': {'straight': 0.05, 'left': 0.02}
        }
        self.spawn_probabilities_offpeak = default_offpeak
        self.spawn_probabilities_peak = default_peak
        # How many steps to consider as peak hours
        self.peak_steps = peak_steps if peak_steps is not None else 1000

        # RL-related
        self.action_size = 4
        self.current_phase = 0
        self.phase_duration = 0
        self.force_action_steps = 0
        self.min_phase_steps = min_phase_steps
        self.max_phase_steps = max_phase_steps
        self.sensor_noise_std = sensor_noise_std

        # last metrics
        self.last_metrics = {'total_vehicles_passed': 0, 'avg_wait': 0.0, 'throughput': 0.0}

    # ------------------ API ------------------
    def reset(self):
        self.vehicles = {}
        self.next_vehicle_id = 0
        self.current_step = 0
        self.total_passed_vehicles = 0
        self.current_phase = 0
        self.phase_duration = 0
        self.force_action_steps = 0
        self.last_metrics = {'total_vehicles_passed': 0, 'avg_wait': 0.0, 'throughput': 0.0}
        return self.get_state()

    def step(self, action):
        # enforce lock
        if self.force_action_steps > 0:
            executed_action = self.current_phase
            self.force_action_steps -= 1
        else:
            if action != self.current_phase:
                executed_action = action
                self.force_action_steps = self.min_phase_steps - 1
                self.phase_duration = 0
            else:
                executed_action = action

        if executed_action != self.current_phase:
            self.current_phase = executed_action
            self.phase_duration = 0
        else:
            self.phase_duration += 1

        # advance simulation
        self.current_step += 1
        self._spawn_vehicles()

        passed_this_step = 0
        for vid, v in list(self.vehicles.items()):
            v.update(self.current_phase)
            if v.passed:
                passed_this_step += 1

        # remove passed/collided
        for vid in [vid for vid, v in list(self.vehicles.items()) if v.passed or v.collided]:
            del self.vehicles[vid]

        self.total_passed_vehicles += passed_this_step

        # metrics
        metrics = self._compute_metrics(passed_this_step)
        reward = self._compute_reward(self.last_metrics, metrics)
        self.last_metrics = metrics.copy()

        next_state = self.get_state()
        done = False
        info = {
            'phase': self.current_phase,
            'phase_duration': self.phase_duration,
            'avg_wait': metrics['avg_wait'],
            'throughput': metrics['throughput'],
            'total_wait': metrics['total_wait'],
            'current_vehicles': metrics['current_vehicles']
        }
        return next_state, reward, done, info

    def _get_current_spawn_probs(self):
        """
        Decide whether we are in peak or off-peak period based on current_step.
        If peak_steps is None, just treat everything as off-peak.
        """
        if self.peak_steps is not None and self.current_step <= self.peak_steps:
            return self.spawn_probabilities_peak
        else:
            return self.spawn_probabilities_offpeak
        
    # ------------------ Simulation helpers ------------------
    def _spawn_vehicles(self):
        current_probs = self._get_current_spawn_probs()
        for direction in ['north', 'south', 'east', 'west']:
            for lane_type in ['straight', 'left']:
                prob = current_probs[direction][lane_type]
                if random.random() < prob:
                    lane_vehicles = [v for v in self.vehicles.values() if v.direction == direction and v.lane_type == lane_type and not v.passed]
                    if len(lane_vehicles) < 6:
                        self._add_vehicle(direction, lane_type)

    def _add_vehicle(self, direction, lane_type):
        v = Vehicle(self.next_vehicle_id, direction, lane_type, self.current_step)
        self.vehicles[self.next_vehicle_id] = v
        self.next_vehicle_id += 1

    def _is_in_waiting_zone(self, vehicle):
        x, y = vehicle.x, vehicle.y
        direction = vehicle.direction

        if direction == 'north':
            return -40 < y < -15
        elif direction == 'south':
            return 15 < y < 40
        elif direction == 'east':
            return -40 < x < -15
        elif direction == 'west':
            return 15 < x < 40
        return False

    # ------------------ Metrics & reward ------------------
    def _compute_metrics(self, passed_this_step):
        queue_lengths = {
            'north_straight': 0, 'north_left': 0,
            'south_straight': 0, 'south_left': 0,
            'east_straight': 0, 'east_left': 0,
            'west_straight': 0, 'west_left': 0
        }

        wait_times = []
        total_wait = 0

        for vid, v in self.vehicles.items():
            if v.passed or v.collided:
                continue

            lane_key = f"{v.direction}_{v.lane_type}"

            # queue length
            if v.speed < 0.5 and self._is_in_waiting_zone(v):
                queue_lengths[lane_key] = queue_lengths.get(lane_key, 0) + 1

            # waiting metrics
            wait_times.append(v.waiting_time)
            total_wait += v.waiting_time

        avg_wait = float(np.mean(wait_times)) if wait_times else 0.0

        # throughput
        if self.current_step > 0:
            throughput = float(self.total_passed_vehicles) / self.current_step
        else:
            throughput = 0.0

        metrics = {
            'queue_lengths': queue_lengths,
            'avg_wait': avg_wait,
            'total_wait': float(total_wait),
            'throughput': throughput,
            'total_vehicles_passed': self.total_passed_vehicles,
            'current_vehicles': len(wait_times)
        }
        return metrics


    def _compute_reward(self, last_metrics, metrics):
        qdict = metrics.get('queue_lengths', {})
        queues = list(qdict.values()) if qdict else [0]
        total_queue = float(sum(queues))
        max_queue = float(max(queues))

        current_total_wait = metrics.get('total_wait', 0)
        last_total_wait = last_metrics.get('total_wait', 0)
        wait_increase = current_total_wait - last_total_wait
        avg_wait = metrics.get('avg_wait', 0.0)
        avg_wait_norm = avg_wait / 100.0

        current_throughput = metrics.get('throughput', 0)
        last_throughput = last_metrics.get('throughput', 0)
        throughput_improvement = current_throughput - last_throughput

        imbalance_penalty = max_queue - (total_queue / len(queues))

        # === A. Original reward function===
        reward = (-0.05 * total_queue
                  - 0.005 * wait_increase
                  + 1.0 * throughput_improvement
                  + 0.5 * current_throughput)
        
        # === B. Stable reward function(without throughput improvement)===
        #reward = (-0.1 * total_queue
        #          - 0.05 * wait_increase
        #          + 1 * current_throughput)

        # === C. Delay-oriented reward function===
        #reward = (-0.01 * avg_wait_norm
        #          - 0.05 * wait_increase
        #          + 1.5 * current_throughput)

        # === D. Balanced queue reward function===
        #reward = (-0.05 * total_queue
        #          - 0.01 * avg_wait_norm
        #          - 0.05 * imbalance_penalty
        #          + 1.5 * current_throughput)
        
        return float(np.clip(reward, -5.0, 5.0))

    # ------------------ State ------------------
    def get_state(self):
        state = []
        MAX_QUEUE = 10.0
        MAX_DISTANCE = 120.0

        for direction in ['north', 'south', 'east', 'west']:
            for lane_type in ['straight', 'left']:
                lane_vehicles = [v for v in self.vehicles.values() if v.direction == direction and v.lane_type == lane_type and not v.passed]
                # queue: stopped cars near stopline
                true_queue_len = len([v for v in lane_vehicles if v.speed < 0.1 and v.distance_to_stopline() < 10.0])
                if self.sensor_noise_std > 0:
                    noisy_queue = true_queue_len + np.random.normal(
                        loc=0.0,
                        scale=self.sensor_noise_std
                    )
                    # keep it >= 0
                    noisy_queue = max(0.0, noisy_queue)
                else:
                    noisy_queue = float(true_queue_len)
                queue_norm = min(noisy_queue / MAX_QUEUE, 1.0)

                if lane_vehicles:
                    dists = [v.distance_to_stopline() for v in lane_vehicles]
                    nearest = max(min(dists), 0.0)
                    dist_norm = min(nearest / MAX_DISTANCE, 1.0)
                else:
                    dist_norm = 1.0

                state.extend([queue_norm, dist_norm])

        ph = [0.0] * self.action_size
        ph[self.current_phase] = 1.0
        state.extend(ph)

        return np.array(state, dtype=np.float32)

    # ------------------ Convenience ------------------
    def get_metrics(self):
        return self.last_metrics.copy()


if __name__ == '__main__': 
    # Switching environment
    #env = TrafficEngine_ideal(seed=42)
    env = TrafficEngine_realistic(peak_steps=1000, seed=42, sensor_noise_std=0.5)
    s = env.reset()
    for i in range(10):
        a = random.randint(0, env.action_size - 1)
        ns, r, d, info = env.step(a)
        print(f"step={i} reward={r:.3f} phase={info['phase']} throughput={info['throughput']} avg_wait={info['avg_wait']:.2f}")
