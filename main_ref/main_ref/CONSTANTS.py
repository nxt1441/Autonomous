# ── Simulation ──────────────────────────────────────────────────────────────
TIME_STEP = 32              # ms per control step

# ── Robot geometry ───────────────────────────────────────────────────────────
MAX_VELOCITY = 6.28         # rad/s  (hardware limit)
WHEEL_RADIUS = 0.043        # m
AXLE_LENGTH  = 0.22         # m

# ── Starting pose ────────────────────────────────────────────────────────────
INITIAL_X     = -0.828208
INITIAL_Y     =  0.81297
INITIAL_THETA =  0.0

# ── Occupancy grid ───────────────────────────────────────────────────────────
MAP_SIZE        = 500               # cells (covers ~10 m × 10 m)
RESOLUTION      = 10.0 / MAP_SIZE   # m/cell = 0.02

INITIAL_LOG_ODD = 1.0
OBSTACLE        = 1
DEPTH_OBSTACLE  = 180
FREESPACE       = 0
UNKNOWN         = 255
BLUE_COLUMN     = 100
YELLOW_COLUMN   = 150
CLOSED          = 200
GREEN_CARPET    = 190

# ── A* planner ───────────────────────────────────────────────────────────────
PATH_MIN_LENGTH_M          = 0.8
ASTAR_INFLATION_LEVELS     = [4, 3, 2]
ASTAR_EXPANSION_PIXELS     = 5
ASTAR_FRONTIER_INFLATION   = 4

# ── Exploration timing ───────────────────────────────────────────────────────
EXPLORATION_STEP_STUCK_CHECK        = 15
EXPLORATION_MAP_UPDATE_FREQ         = 20
EXPLORATION_FRONTIER_SELECTION_FREQ = 5
EXPLORATION_START_FRONTIER_AFTER    = 50
EXPLORATION_PATH_PLANNING_FREQ      = 100

# ── Frontier parameters ──────────────────────────────────────────────────────
FRONTIER_MIN_DISTANCE_NEW             = 20
FRONTIER_APPROACH_DISTANCE            = 10
FRONTIER_VISUALIZATION_COLOR_SMALL    = 50
FRONTIER_VISUALIZATION_COLOR_MEDIUM   = 100
FRONTIER_VISUALIZATION_COLOR_LARGE    = 200
FRONTIER_VISUALIZATION_COLOR_LARGEST  = 220

# ── Obstacle avoidance ───────────────────────────────────────────────────────
WALL_DETECTION_THRESHOLD_FRONTIER       = 0.3
WALL_DETECTION_THRESHOLD_PATH_FOLLOWING = 0.2
OBSTACLE_AVOID_THRESHOLD                = 0.25
OBSTACLE_AVOID_MAX_ATTEMPTS             = 2

# ── DWA planner ──────────────────────────────────────────────────────────────
DWA_VELOCITY_SAMPLES              = [0.05, 0.10, 0.15, 0.20, 0.25, 0.27]
DWA_ANGULAR_SAMPLES               = [0, 1.5, -1.5, 2.5, -2.5, 3.0, -3.0, 3.5, -3.5, 4.0, -4.0]
DWA_HEADING_WEIGHT                = 4.0
DWA_DISTANCE_WEIGHT               = 3.5
DWA_SPEED_WEIGHT                  = 0.05
DWA_COST_MAP_WEIGHT               = 1.5
DWA_PREDICTION_DISTANCE_THRESHOLD = 0.2

# ── Path following ───────────────────────────────────────────────────────────
PATH_FOLLOWING_TARGET_REACH_DISTANCE = 4   # pixels

# ── Motor control ────────────────────────────────────────────────────────────
MOTOR_VELOCITY_FORWARD  =  6.28
MOTOR_VELOCITY_TURN     =  6.0
MOTOR_VELOCITY_BACKWARD = -5.0
RED_WALL_ALIGNMENT_SPEED = -2.0

# ── PID alignment ────────────────────────────────────────────────────────────
ALIGN_RED_WALL_KP            = 0.008
ALIGN_RED_WALL_KD            = 0.006
ALIGN_RED_WALL_ERROR_THRESHOLD = 5
ALIGN_COLUMN_KP              = 0.008
ALIGN_COLUMN_KD              = 0.002
ALIGN_COLUMN_ERROR_THRESHOLD = 20
ALIGN_COLUMN_FORWARD_SPEED   = 2.0
ALIGN_PATH_ANGLE_THRESHOLD   = 15
ALIGN_PATH_CLEAR_DISTANCE    = 0.6
ALIGN_PATH_BACK_DISTANCE     = 0.18
ALIGN_PATH_ROTATION_SPEED    = 1.0

# ── Closure marking ──────────────────────────────────────────────────────────
CLOSURE_MARK_COOLDOWN       = 5.0
CLOSURE_MARK_FORWARD        = 0.8
CLOSURE_MARK_BACKWARD       = -0.4
CLOSURE_MARK_WIDTH          = 0.8
CLOSURE_MARK_IOU_THRESHOLD  = 0.4

# ── Colour detection (HSV) ───────────────────────────────────────────────────
COLOR_DETECTION_DEPTH_THRESHOLD = 80
COLOR_DETECTION_RED_PIXEL_RATIO = 0.4
RED_WALL_HSV_LOWER1  = [0,   120, 70]
RED_WALL_HSV_UPPER1  = [10,  255, 255]
RED_WALL_HSV_LOWER2  = [170, 120, 70]
RED_WALL_HSV_UPPER2  = [180, 255, 255]
BLUE_HSV_LOWER       = [100, 150, 50]
BLUE_HSV_UPPER       = [140, 255, 255]
YELLOW_HSV_LOWER     = [20,  100, 100]
YELLOW_HSV_UPPER     = [35,  255, 255]
GREEN_HSV_LOWER      = [36,  100, 100]
GREEN_HSV_UPPER      = [86,  255, 255]
GREEN_CARPET_DILATION_KERNEL_SIZE = 10
GREEN_CARPET_DILATION_ITERATIONS  = 1

# ── Turning ──────────────────────────────────────────────────────────────────
TURN_DURATION_MIN               = 80
TURN_DURATION_MAX               = 200
TURN_ANGLE_COMPLETION_THRESHOLD = 0.02

# ── Sensor ───────────────────────────────────────────────────────────────────
LIDAR_FRONT_CONE_ANGLE = 15
LIDAR_MAX_RANGE        = 2.5   # m — cap far readings to reduce noise
DEPTH_OBSTACLE_POST_TURN_STABILIZATION_STEPS = 3
DEPTH_OBSTACLE_MIN_HEIGHT = 0.04
DEPTH_OBSTACLE_MAX_HEIGHT = 1.2
DEPTH_OBSTACLE_SUPPORT_MAX_HEIGHT = 0.16
DEPTH_OBSTACLE_SUPPORT_FORWARD_TOLERANCE = 0.06
DEPTH_OBSTACLE_SUPPORT_LATERAL_TOLERANCE = 0.08
DEPTH_OBSTACLE_FLOATING_MIN_HEIGHT = 0.05
DEPTH_OBSTACLE_FLOATING_MAX_HEIGHT = 0.22
DEPTH_OBSTACLE_LOG_ODDS_INCREMENT = 1.5
DEPTH_OBSTACLE_LOG_ODDS_CAP = 3.0
DEPTH_OBSTACLE_STAMP_RADIUS = 1
DEPTH_OBSTACLE_CONFIRMATION_COUNT = 3
DEPTH_OBSTACLE_DECAY_STEPS = 1
DEPTH_OBSTACLE_BRIDGE_GAP_CELLS = 2
DEPTH_OBSTACLE_FRONT_WIDTH = 0.38
DEPTH_OBSTACLE_STOP_DISTANCE = 0.35
FLOATING_WALL_MAX_STAMP_DIST = 2.5   # m — don't stamp if wall is farther than this
FLOATING_WALL_THICKNESS      = 1    # cells — dilation radius for stamped wall cells

COST_MAP_MAX_DIST = 14     # cells — distance transform horizon for cost/dist maps
