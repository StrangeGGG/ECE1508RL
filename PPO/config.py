# traffic_env configuration parameters
LANE_WIDTH = 1.0 # gud for position representation (x,y)
SPAWN_DISTANCE_TO_INTERSECTION = 80.0 # distance from intersection where vehicles spawn
SPAWN_DISTANCE_TO_CENRELINE = 0.0 # x = 0 & y = 0 is centerline
BASE_SPAWN_PROB_STRAIGHT = 0.02 # base spawn probability per lane per step
BASE_SPAWN_PROB_LEFT = 0.01 # base spawn probability per lane per step
NUM_LANES_PER_DIRECTION_IN_STRAIGHT = 1 # default straight =1 and left =1
NUM_LANES_PER_DIRECTION_IN_LEFT = 1 # can be adjusted to increase traffic
INTERSECTION_SIZE = (NUM_LANES_PER_DIRECTION_IN_STRAIGHT+NUM_LANES_PER_DIRECTION_IN_LEFT)*LANE_WIDTH # half-size of intersection box
MAX_VEHICLES_PER_LANE= 10 # max number of vehicles in simulation

# Vehicle parameters
vehicle_length = 2.0 # length of vehicle
MAX_SPEED = 3.0 # max speed of vehicles
ACCELERATION = 0.5 # acceleration per step
DECELERATION = 3 # deceleration per step when braking

SPAWN_SPEED = 2.0 # initial speed when spawning


# Traffic light parameters
YELLOW_DURATION = 3 # duration of yellow light in steps
MIN_GREEN_DURATION = 0 # min duration of green light in steps
MAX_GREEN_DURATION = 60 # max duration of green light in steps