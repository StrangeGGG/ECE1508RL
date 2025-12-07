import random
import numpy as np

from config import (
    LANE_WIDTH,
    INTERSECTION_SIZE,
    SPAWN_DISTANCE_TO_INTERSECTION,
    SPAWN_DISTANCE_TO_CENRELINE,
    BASE_SPAWN_PROB_STRAIGHT,
    BASE_SPAWN_PROB_LEFT,
    NUM_LANES_PER_DIRECTION_IN_STRAIGHT,
    NUM_LANES_PER_DIRECTION_IN_LEFT,
    MAX_SPEED,
    ACCELERATION,
    DECELERATION,
    MAX_VEHICLES_PER_LANE,
    MAX_PHASE_DURATION,
    MIN_PHASE_DURATION,
)

class Vehicle:
    """
    Single vehicle moving through the intersection.

    Coordinates:
        - Intersection center is at (0, 0).
        - For each direction we spawn at +/- SPAWN_DISTANCE_TO_INTERSECTION
          along the travel axis, and shift laterally by lane_index * LANE_WIDTH.

    Directions:
        'north' : moves +y
        'south' : moves -y
        'east'  : moves +x
        'west'  : moves -x

    lane_type:
        'straight' or 'left'
        (right turns are ignored in this simplified model)
    """

    def __init__(self, vehicle_id, direction, lane_type, lane_index, spawn_time):
        self.id = vehicle_id
        self.direction = direction  # 'north','south','east','west'
        self.lane_type = lane_type  # 'straight','left'
        self.lane_index = lane_index  # start from 1

        self.spawn_time = spawn_time
        self.waiting_time = 0
        self.passed = False
        self.has_turned = False
        self.collided = False
        self.in_intersection_zone = False
        self.cross_intersection = False

        # kinematics
        self.speed = 0.0
        self.max_speed = MAX_SPEED
        self.acceleration = ACCELERATION
        self.deceleration = DECELERATION

        # position
        self.x, self.y = 0.0, 0.0
        self.set_initial_position()
        self.prev_x, self.prev_y = self.x, self.y

    def set_initial_position(self):

        if self.direction == 'north':
            # coming from bottom to top, y starts negative
            self.x = SPAWN_DISTANCE_TO_CENRELINE + LANE_WIDTH * self.lane_index
            self.y = -SPAWN_DISTANCE_TO_INTERSECTION

        elif self.direction == 'south':
            # top to bottom, y starts positive
            self.x = SPAWN_DISTANCE_TO_CENRELINE - LANE_WIDTH * self.lane_index
            self.y = SPAWN_DISTANCE_TO_INTERSECTION

        elif self.direction == 'east':
            # left to right, x starts negative
            self.x = -SPAWN_DISTANCE_TO_INTERSECTION
            self.y = SPAWN_DISTANCE_TO_CENRELINE - LANE_WIDTH * self.lane_index

        elif self.direction == 'west':
            # right to left, x starts positive
            self.x = SPAWN_DISTANCE_TO_INTERSECTION
            self.y = SPAWN_DISTANCE_TO_CENRELINE + LANE_WIDTH * self.lane_index

    def in_intersection(self) -> bool:
        return (
            -INTERSECTION_SIZE <= self.x <= INTERSECTION_SIZE
            and -INTERSECTION_SIZE <= self.y <= INTERSECTION_SIZE
        )

    def distance_to_stop_line(self) -> float:
        if self.direction == 'north':
            # coming from y < 0 moving +y, stop line at -INTERSECTION_SIZE
            return -INTERSECTION_SIZE - self.y
        elif self.direction == 'south':
            # coming from y > 0 moving -y, stop line at +INTERSECTION_SIZE
            return self.y - INTERSECTION_SIZE
        elif self.direction == 'east':
            # coming from x < 0 moving +x, stop line at -INTERSECTION_SIZE
            return -INTERSECTION_SIZE - self.x
        elif self.direction == 'west':
            # coming from x > 0 moving -x, stop line at +INTERSECTION_SIZE
            return self.x - INTERSECTION_SIZE
        return 0.0

    def need_track(self) -> bool:
        """
        within some distance of the intersection along travel axis.
        If not, no need to track signal or wait.
        """
        if self.direction in ['north', 'south']:
            return abs(self.y) <= SPAWN_DISTANCE_TO_INTERSECTION
        else:
            return abs(self.x) <= SPAWN_DISTANCE_TO_INTERSECTION

    def move_along_direction(self, dist: float):
        """
            Move distance 'dist' along the current direction.
        """
        
        if self.direction == 'north':
            self.y += dist
        elif self.direction == 'south':
            self.y -= dist
        elif self.direction == 'east':
            self.x += dist
        elif self.direction == 'west':
            self.x -= dist
    
    def has_cross_intersection(self) -> bool:
        """
        Check if the vehicle has crossed the intersection area.
        """
        if self.direction in ['north']:
            return (self.y >= -INTERSECTION_SIZE)
        elif self.direction in ['south']:
            return (self.y <= INTERSECTION_SIZE)
        elif self.direction in ['east']:
            return (self.x >= -INTERSECTION_SIZE)
        elif self.direction in ['west']:
            return (self.x <= INTERSECTION_SIZE)
        return False

    def should_wait(self, current_signal: int) -> bool:
        """
        Signal control phases:
          0: NS left
          1: EW left
          2: NS straight
          3: EW straight
        """
        self.cross_intersection = self.has_cross_intersection()
        if not self.need_track() or self.cross_intersection:
            return False

        if current_signal == 0:  # NS left
            return not (self.direction in ['north', 'south'] and self.lane_type == 'left')
        elif current_signal == 1:  # EW left
            return not (self.direction in ['east', 'west'] and self.lane_type == 'left')
        elif current_signal == 2:  # NS straight
            return not (self.direction in ['north', 'south'] and self.lane_type == 'straight')
        elif current_signal == 3:  # EW straight
            return not (self.direction in ['east', 'west'] and self.lane_type == 'straight')
        return True

    def should_turn_left_now(self) -> bool:
        """
        Decide when to start turning left. 
        All vehicles should turn left when it reaches the lane index 1 of new direction
        """
        condition_met = False
        condition_met = self.lane_type == 'left' and (not self.has_turned) and self.in_intersection()
        if condition_met:
            if self.direction == 'north':
                if self.y >= LANE_WIDTH:
                    dist_add_to_new_direction = self.y - LANE_WIDTH
                    self.x = -dist_add_to_new_direction + LANE_WIDTH * self.lane_index
                    self.y = LANE_WIDTH
                    return True
            elif self.direction == 'south':
                if self.y <= -LANE_WIDTH:
                    dist_add_to_new_direction = -LANE_WIDTH - self.y
                    self.x = dist_add_to_new_direction - LANE_WIDTH * self.lane_index
                    self.y = -LANE_WIDTH
                    return True 
            elif self.direction == 'east':
                if self.x >= LANE_WIDTH:
                    dist_add_to_new_direction = self.x - LANE_WIDTH
                    self.y = dist_add_to_new_direction - LANE_WIDTH * self.lane_index
                    self.x = LANE_WIDTH
                    return True
            elif self.direction == 'west':
                if self.x <= -LANE_WIDTH:
                    dist_add_to_new_direction = -LANE_WIDTH - self.x
                    self.y = -dist_add_to_new_direction + LANE_WIDTH * self.lane_index
                    self.x = -LANE_WIDTH
                    return True
            
        return False

    def turn_left(self):
        if self.direction == 'north':
            self.direction = 'west'
        elif self.direction == 'south':
            self.direction = 'east'
        elif self.direction == 'east':
            self.direction = 'north'
        elif self.direction == 'west':
            self.direction = 'south'
        self.has_turned = True
        
    def check_passed(self):
        limit = SPAWN_DISTANCE_TO_INTERSECTION + 10.0  
        if self.direction == 'north' and self.y > limit:
            self.passed = True
        elif self.direction == 'south' and self.y < -limit:
            self.passed = True
        elif self.direction == 'east' and self.x > limit:
            self.passed = True
        elif self.direction == 'west' and self.x < -limit:
            self.passed = True

    # ---------------- update per time step ----------------

    def update(self, current_signal: int):
        if self.passed or self.collided:
            return

        self.in_intersection_zone = self.in_intersection()
        wait = self.should_wait(current_signal)

        dist_to_stopline = self.distance_to_stop_line()

        if wait and dist_to_stopline <= 0.0:
            wait = False

        if wait:
            self.speed = max(0.0, self.speed - self.deceleration)
            self.waiting_time += 1
        else:
            self.speed = min(self.max_speed, self.speed + self.acceleration)

        move_dist = self.speed
        if wait and dist_to_stopline > 0.0:
            move_dist = min(move_dist, dist_to_stopline)

        self.prev_x, self.prev_y = self.x, self.y
        self.move_along_direction(move_dist)

        if (not wait) and self.should_turn_left_now():
            self.turn_left()

        self.check_passed()

class TrafficMetricsCollector:
    """
    Keeps track of simple intersection-level metrics.
    """

    def __init__(self):
        self.reset()

    def reset(self):
        self.metrics = {
            'queue_lengths': {},            # lane_key -> queue length
            'lane_waiting_times': {},       # lane_key -> avg waiting
            'lane_counts': {},              # lane_key -> number of vehicles
            'average_waiting_time': 0.0,    # global avg
            'throughput': 0.0,              # vehicles passed this step
            'total_vehicles_passed': 0,     # cumulative
        }
        
        self.each_lane_count = {}           # each_lane_key -> number of vehicles => used for spawning
        
    def update_metrics(
        self,
        vehicles_info: dict,
        current_signal: int,
        current_step: int,
        passed_this_step: int,
        total_passed_vehicles: int,
    ):
        # per-lane accumulators
        queue_lengths = {}
        lane_wait_sums = {}
        lane_counts = {}
        
        self.each_lane_count = {}

        total_wait_sum = 0.0
        total_wait_count = 0

        for vid, info in vehicles_info.items():
            direction = info['direction']
            lane_type = info['lane_type']
            lane_index = info['lane_index']
            lane_key = f"{direction}_{lane_type}"
            each_lane_key = f"{direction}_{lane_index}"

            waiting_time = info['waiting_time']
            speed = info['speed']
            in_intersection = info['in_intersection']
            cross_intersection = info['cross_intersection']
            passed = info['passed']
            collided = info['collided']

            if passed or collided:
                continue

            lane_counts[lane_key] = lane_counts.get(lane_key, 0) + 1
            self.each_lane_count[each_lane_key] = self.each_lane_count.get(each_lane_key, 0) + 1

            # waiting sums (for lane avg)
            lane_wait_sums[lane_key] = lane_wait_sums.get(lane_key, 0.0) + waiting_time

            # global waiting stats
            total_wait_sum += waiting_time
            total_wait_count += 1

            # queue definition: near intersection and almost stopped, and not yet inside
            if (not cross_intersection) and speed < 1.0:
                queue_lengths[lane_key] = queue_lengths.get(lane_key, 0) + 1

        # derive lane_waiting_times
        lane_waiting_times = {}
        for lane_key, count in lane_counts.items():
            if count > 0:
                lane_waiting_times[lane_key] = lane_wait_sums[lane_key] / count
            else:
                lane_waiting_times[lane_key] = 0.0

        avg_wait = (
            total_wait_sum / total_wait_count if total_wait_count > 0 else 0.0
        )

        self.metrics['queue_lengths'] = queue_lengths
        self.metrics['lane_waiting_times'] = lane_waiting_times
        self.metrics['lane_counts'] = lane_counts
        self.metrics['average_waiting_time'] = avg_wait
        self.metrics['throughput'] = float(passed_this_step)
        self.metrics['total_vehicles_passed'] = int(total_passed_vehicles)
        
    def get_metrics(self) -> dict:
        return self.metrics.copy()

class TrafficSimulation:
    """
    traffic simulation for one intersection.
    - Four directions: north, south, east, west
    - Two movement types: straight, left
    """

    def __init__(self):
        self.vehicles = {}
        self.next_vehicle_id = 0
        self.current_step = 0

        self.metrics_collector = TrafficMetricsCollector()
        self.total_passed_vehicles = 0

        self.directions = ['north', 'south', 'east', 'west']
        self.lane_types = ['straight', 'left']

        # number of lanes per direction (shared by straight & left)
        self.num_lanes_per_direction = NUM_LANES_PER_DIRECTION_IN_STRAIGHT + NUM_LANES_PER_DIRECTION_IN_LEFT
        
        # spawn probs per direction & movement
        self.spawn_probabilities = {
            d: {
                'straight': BASE_SPAWN_PROB_STRAIGHT,
                'left': BASE_SPAWN_PROB_LEFT,
            }
            for d in self.directions
        }

        self.max_vehicles_per_lane = MAX_VEHICLES_PER_LANE  # capacity per (direction, lane_index)

    def reset(self):
        self.vehicles = {}
        self.next_vehicle_id = 0
        self.current_step = 0
        self.total_passed_vehicles = 0
        self.metrics_collector.reset()
        return self.get_observation(), self.get_metrics()

    # ---------------- vehicle management ----------------

    def add_vehicle(self, direction: str, lane_type: str, lane_index: int):
        v = Vehicle(
            self.next_vehicle_id,
            direction,
            lane_type,
            lane_index,
            spawn_time=self.current_step,
        )
        self.vehicles[self.next_vehicle_id] = v
        self.next_vehicle_id += 1

    def spawn_vehicles(self):
        """
        For each physical lane (direction, lane_index), try to spawn at most
        one new vehicle per time step.

        - Capacity check is done using metrics_collector.each_lane_count,
        so lane_count never exceeds max_vehicles_per_lane.
        - If we spawn, lane_type ('straight' or 'left') is chosen using
        self.spawn_probabilities[direction][lane_type] as weights.
        """
        # latest per-lane counts from metrics (may be empty at very start)
        lane_counts_by_index = getattr(self.metrics_collector, "each_lane_count", {})

        for direction in self.directions:
            for lane_index in range(1, self.num_lanes_per_direction + 1):
                each_lane_key = f"{direction}_{lane_index}"
                current_count = lane_counts_by_index.get(each_lane_key, 0)

                # capacity guard: do not spawn if lane is full
                if current_count >= self.max_vehicles_per_lane:
                    continue

                # get base spawn probabilities for this direction
                p_straight = self.spawn_probabilities[direction]['straight']
                p_left = self.spawn_probabilities[direction]['left']
                p_total = p_straight + p_left

                if p_total <= 0.0:
                    continue  # nothing to spawn for this lane

                # one spawn attempt per lane per time step
                if random.random() < p_total:
                    # decide movement type (straight vs left)
                    r = random.random()
                    if r < p_straight / p_total:
                        lane_type = 'straight'
                    else:
                        lane_type = 'left'

                    self.add_vehicle(direction, lane_type, lane_index)

    def step(self, current_signal: int):
        self.current_step += 1

        self.spawn_vehicles()

        passed_ids = []
        passed_this_step = 0

        for vid, v in list(self.vehicles.items()):
            v.update(current_signal)
            if v.passed:
                passed_ids.append(vid)
                passed_this_step += 1

        for vid in passed_ids:
            del self.vehicles[vid]

        self.total_passed_vehicles += passed_this_step

        vehicles_info = {}
        for vid, v in self.vehicles.items():
            vehicles_info[vid] = {
                'x': v.x,
                'y': v.y,
                'direction': v.direction,
                'lane_type': v.lane_type,
                'lane_index': v.lane_index,
                'waiting_time': v.waiting_time,
                'speed': v.speed,
                'passed': v.passed,
                'collided': v.collided,
                'in_intersection': v.in_intersection_zone,
                'cross_intersection': v.cross_intersection,
            }

        self.metrics_collector.update_metrics(
            vehicles_info,
            current_signal,
            self.current_step,
            passed_this_step,
            self.total_passed_vehicles,
        )

        return self.get_observation(), self.get_metrics()

    def get_observation(self):
        """
        For now, we do not use a detailed observation here.
        The RL wrapper will build a compact state from metrics instead.
        """
        return None

    def get_metrics(self):
        return self.metrics_collector.get_metrics()

class TrafficRLWrapper:
    """
    A lightweight RL environment that wraps TrafficSimulation.

    Action space:
        0: NS left
        1: EW left
        2: NS straight
        3: EW straight

    State:
        For each of 8 lanes (N/S/E/W x straight/left):
            [queue_len_norm, avg_wait_norm, vehicle_count_norm] => 24 dims
        +   one-hot of current phase
        => total 28 dims
    """

    def __init__(self, sim: TrafficSimulation = None,
                 min_phase_steps: int = MIN_PHASE_DURATION,
                 max_phase_steps: int = MAX_PHASE_DURATION    # max_phase_duration doesn't use
                 ):
        self.sim = sim if sim is not None else TrafficSimulation()

        self.action_size = 4  # 0:NS left, 1:EW left, 2:NS straight, 3:EW straight
        self.min_phase_steps = min_phase_steps
        self.max_phase_steps = max_phase_steps

        self.current_signal = 0
        self.phase_duration = 0

        self.last_metrics = {
            'queue_lengths': {},
            'lane_waiting_times': {},
            'lane_counts': {},
            'average_waiting_time': 0.0,
            'throughput': 0.0,
            'total_vehicles_passed': 0,
        }

    def reset(self):
        _, metrics = self.sim.reset()
        self.current_signal = 0
        self.phase_duration = 0
        self.last_metrics = metrics.copy()
        return self.get_state()

    def step(self, action: int):
        if self.phase_duration < self.min_phase_steps:
            pass
        else:
            if action != self.current_signal:
                self.current_signal = int(action)
                self.phase_duration = 0

        self.phase_duration += 1

        _, metrics = self.sim.step(self.current_signal)

        reward = self.compute_reward(self.last_metrics, metrics)
        self.last_metrics = metrics.copy()

        next_state = self.get_state()
        done = False  
        info = {
            'phase': self.current_signal,
            'phase_duration': self.phase_duration,
        }
        info.update(metrics)
        return next_state, reward, done, info

    def get_state(self):
        """
        Build 28-dim state.
        For each of 8 lanes (N/S/E/W x straight/left):
            [queue_len_norm, avg_wait_norm, vehicle_count_norm] => 24 dims
        + 4-d one-hot of current phase
        """
        metrics = self.sim.get_metrics()
        queue_lengths = metrics.get('queue_lengths', {})
        lane_wait = metrics.get('lane_waiting_times', {})
        lane_counts = metrics.get('lane_counts', {})
        
        MAX_QUEUE = MAX_VEHICLES_PER_LANE*(NUM_LANES_PER_DIRECTION_IN_LEFT+NUM_LANES_PER_DIRECTION_IN_STRAIGHT)*4
        MAX_WAIT = 500
        MAX_COUNT = MAX_VEHICLES_PER_LANE*(NUM_LANES_PER_DIRECTION_IN_LEFT+NUM_LANES_PER_DIRECTION_IN_STRAIGHT)*4

        state = []

        for direction in ['north', 'south', 'east', 'west']:
            for lane_type in ['straight', 'left']:
                lane_key = f"{direction}_{lane_type}"

                q = queue_lengths.get(lane_key, 0.0)
                w = lane_wait.get(lane_key, 0.0)
                c = lane_counts.get(lane_key, 0.0)

                state.append(q / MAX_QUEUE)
                state.append(w / MAX_WAIT)
                state.append(c / MAX_COUNT)

        phase_vec = [0.0] * 4
        phase_vec[self.current_signal] = 1.0
        state.extend(phase_vec)

        #norm_phase_dur = min(self.phase_duration, self.max_phase_steps) / float(self.max_phase_steps)
        #state.append(norm_phase_dur)

        return np.array(state, dtype=np.float32)

    @staticmethod
    def compute_reward(last_metrics: dict, metrics: dict) -> float:
        """
        Simple reward combining:
          + positive for extra passed vehicles
          + penalties for increased queue lengths and waiting time
        """
        q_old = sum(last_metrics.get('queue_lengths', {}).values())
        q_new = sum(metrics.get('queue_lengths', {}).values())
        queue_inc = q_new - q_old

        w_old = last_metrics.get('average_waiting_time', 0.0)
        w_new = metrics.get('average_waiting_time', 0.0)
        wait_inc = w_new - w_old
        
        passed_old = last_metrics.get('total_vehicles_passed', 0)
        passed_new = metrics.get('total_vehicles_passed', 0)
        
        #original
        # queue_penalty = -0.03 * queue_inc
        # wait_penalty = -0.06 * wait_inc
        
        # passed_reward = (passed_new - passed_old) * 1.0
        
        # max_queue_increamental_penalth
        # queue_penalty = -3 * queue_inc
        # wait_penalty = -0.06 * wait_inc

        # passed_reward = (passed_new - passed_old) * 1.0
        
        
        # max_waiting_time penalty
        # queue_penalty = -0.03 * queue_inc
        # wait_penalty = -0.6 * wait_inc

        # passed_reward = (passed_new - passed_old) * 0.1
        
        # max throughput penalty
        # queue_penalty = -0.03 * queue_inc
        # wait_penalty = -0.06 * wait_inc

        # passed_old = last_metrics.get('total_vehicles_passed', 0)
        # passed_new = metrics.get('total_vehicles_passed', 0)
        # passed_reward = (passed_new - passed_old) * 10
        
        # balanced queue
        q_dict = metrics.get('queue_lengths', {})
        queues = np.array(list(q_dict.values()), dtype=np.float32) if q_dict else np.array([0.0])

        mean_queue = queues.mean()
        queue_imbalanced_penalty = np.mean(np.sqrt(np.abs(queues - mean_queue)))
        
        queue_penalty = -0.03 * queue_imbalanced_penalty
        wait_penalty = -0.06 * wait_inc
        passed_reward = (passed_new - passed_old) * 1
        
        reward = queue_penalty 
        #reward = float(np.clip(reward, -5.0, 5.0))
        return reward