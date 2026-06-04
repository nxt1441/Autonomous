"""
ROSbot Maze Controller v3 — Python with Bayes Binary Filter Grid Mapping.

Proper occupancy grid mapping based on:
  - Bayes Binary Filter (log-odds representation)
  - Reference: github.com/lukovicaleksa/grid-mapping-in-ROS
  - Bresenham line drawing for ray casting
  - Inverse sensor model for probabilistic updates

Features:
  - VFH obstacle avoidance with corner-stuck recovery
  - Floating/overhead wall detection via depth camera
  - Proper Bayesian occupancy grid mapping (LiDAR + IR + depth camera)
  - Compass-fused odometry (ENU coordinate system)
  - Built-in real-time matplotlib occupancy grid display

To use:
  1. Set controller to "mycontroller_py" in Maze1.wbt
  2. Run the simulation — live grid plot opens automatically

All tunable parameters marked with [TUNABLE].
"""

import math
import numpy as np

from controller import Robot

# ====================== MATPLOTLIB SETUP ======================
PLOTTING_AVAILABLE = False
plt = None
matplotlib = None
try:
    import matplotlib as _mpl
    matplotlib = _mpl
    _backend_ok = False
    for _backend in ["macosx", "TkAgg", "QtAgg", "Qt5Agg"]:
        try:
            matplotlib.use(_backend, force=True)
            import matplotlib.pyplot as _plt
            plt = _plt
            _backend_ok = True
            print(f"[PLOT] Using backend: {_backend}")
            break
        except Exception:
            continue
    if not _backend_ok:
        matplotlib.use("Agg", force=True)
        import matplotlib.pyplot as _plt
        plt = _plt
        print("[PLOT] Agg backend (saves PNG files, no live window)")
    PLOTTING_AVAILABLE = True
except ImportError:
    print("[WARN] matplotlib not found — pip3 install matplotlib")

# ======================== TUNABLE PARAMETERS ========================

# --- Simulation ---
TIME_STEP = 16  # [TUNABLE] ms per step. Lower = more precise (16-64)

# --- Occupancy Grid ---
# Maze1.wbt walls span approximately X:[-2.0, 2.3], Y:[-1.6, 3.2]
# Add 0.5m padding on each side for sensor coverage
GRID_RESOLUTION = 0.03     # [TUNABLE] Meters per cell. 3cm matches reference (0.02-0.10)
MAP_X_LIM = [-2.5, 3.0]   # [TUNABLE] World X coverage in meters (maze spans ~-2.0 to 2.3)
MAP_Y_LIM = [-2.0, 3.7]   # [TUNABLE] World Y coverage in meters (maze spans ~-1.6 to 3.2)

# --- Bayes Binary Filter Probabilities ---
P_PRIOR = 0.5   # Prior probability (unknown). 0.5 = maximum uncertainty. Don't change
P_OCC = 0.9     # [TUNABLE] Probability cell is occupied given sensor hit (0.7-0.95)
P_FREE = 0.3    # [TUNABLE] Probability cell is free given sensor ray passes (0.1-0.4)

# --- Log-odds clamp (prevents over-confidence) ---
L_OCC_MAX = 40.0   # [TUNABLE] Max log-odds (very confident occupied). Higher = stickier walls (20-80)
L_FREE_MIN = -40.0  # [TUNABLE] Min log-odds (very confident free). Lower = stickier free space (-20 to -80)

# --- MLE thresholds for visualization ---
THRESH_P_FREE = 0.2   # [TUNABLE] Below this probability → cell is "free" (white)
THRESH_P_OCC = 0.5    # [TUNABLE] Above this probability → cell is "occupied" (black)

# --- Obstacle Avoidance Thresholds (meters) ---
DANGER_DIST = 0.10     # [TUNABLE] Emergency stop/reverse distance (0.10-0.25)
CAUTION_DIST = 0.35    # [TUNABLE] Start slowing/steering distance (0.25-0.50)
CLEAR_DIST = 0.55      # [TUNABLE] Considered safe distance (0.40-0.80)

# --- Floating/Overhead Wall Detection ---
OVERHEAD_DANGER_DIST = 0.50     # [TUNABLE] Depth range for overhead detection (0.3-1.0)
OVERHEAD_HEIGHT_RATIO = 0.5   # [TUNABLE] Top fraction of depth image to scan (0.2-0.5)
OVERHEAD_PIXEL_THRESHOLD = 0.15 # [TUNABLE] Fraction of pixels needed to trigger (0.05-0.30)

# --- VFH ---
VFH_SECTORS = 72           # [TUNABLE] Angular sectors, each = 360/72 = 5° (36, 72, 120)
VFH_THRESHOLD = 2.5        # [TUNABLE] Sector obstacle density to block (1.0-5.0)
VFH_WIDE_OPENING = 5       # [TUNABLE] Min sectors for "wide" opening (3-10)
VFH_OVERHEAD_BOOST = 20.0  # [TUNABLE] Forward-blocking strength for overhead walls (10-50)

# --- Motion Speeds (rad/s) ---
# Slower speeds = better mapping (less motion blur) and fewer collisions in tight maze
MAX_SPEED = 6.28       # [TUNABLE] Max wheel speed (3.0-10.0)
CRUISE_SPEED = 3.0     # [TUNABLE] Normal speed — slower for better mapping (2.0-6.0)
SLOW_SPEED = 1.8       # [TUNABLE] Near-obstacle speed (1.0-3.0)
TURN_SPEED = 2.0       # [TUNABLE] Rotation speed — slower to reduce map smearing (1.5-4.0)

# --- PD Controller ---
KP_TURN = 3.5   # [TUNABLE] Proportional gain (1.0-8.0)
KD_TURN = 0.6   # [TUNABLE] Derivative gain (0.1-2.0)

# --- Wall Following ---
WALL_FOLLOW_DIST = 0.40    # [TUNABLE] Target wall distance (0.15-0.50)
KP_WALL = 6.0              # [TUNABLE] Wall-follow gain (3.0-12.0)
WALL_FOLLOW_TIMEOUT = 250  # [TUNABLE] Max steps in wall-follow (100-500)

# --- Robot Physical ---
WHEEL_RADIUS = 0.043  # Don't change
WHEEL_BASE = 0.22     # Don't change

# --- Stuck Detection ---
STUCK_CHECK_INTERVAL = 50    # [TUNABLE] Check every N steps (40-150)
STUCK_DIST_THRESHOLD = 0.02  # [TUNABLE] Min movement to not be stuck (0.01-0.05)
MAX_STUCK_COUNT = 3           # [TUNABLE] Consecutive stuck before escape (2-5)

# --- Corner Escape ---
ESCAPE_BACKUP_STEPS = 40          # [TUNABLE] Reverse duration (15-50)
ESCAPE_ROTATE_STEPS = 50          # [TUNABLE] Rotation duration (30-100)
ESCAPE_BACKUP_SPEED = 1.0         # [TUNABLE] Backup speed fraction (0.5-1.0)
ESCAPE_ROTATE_MULTIPLIER = 1.5    # [TUNABLE] Turn speed multiplier (1.0-2.5)

# --- Live Plot ---
PLOT_UPDATE_INTERVAL = 15  # [TUNABLE] Update plot every N steps (5-50)
PLOT_ENABLED = True         # [TUNABLE] False to disable

# --- Initial Robot Position (must match .wbt) ---
INITIAL_X = -0.828208  # [TUNABLE]
INITIAL_Y = 0.81297    # [TUNABLE]

# --- Distance Sensor Mounting Angles (radians, from proto) ---
DS_ANGLES = [0.13, -0.13, math.pi - 0.13, -(math.pi - 0.13)]

# ======================== STATE MACHINE ========================

STATE_FORWARD = 0
STATE_AVOID = 1
STATE_WALL_FOLLOW = 2
STATE_ROTATE = 3
STATE_REVERSE = 4
STATE_ESCAPE = 5

STATE_NAMES = {0: "FORWARD", 1: "AVOID", 2: "WALL_FOLLOW",
               3: "ROTATE", 4: "REVERSE", 5: "ESCAPE"}


# ======================== BAYES BINARY FILTER ========================

def log_odds(p):
    """Convert probability to log-odds: l(x) = log(p / (1-p))"""
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(p / (1.0 - p))


def retrieve_p(l):
    """Convert log-odds back to probability: p = 1 - 1/(1+exp(l))"""
    return 1.0 - 1.0 / (1.0 + np.exp(l))


# Pre-compute log-odds constants
L_PRIOR = log_odds(P_PRIOR)
L_OCC = log_odds(P_OCC)
L_FREE = log_odds(P_FREE)


class OccupancyGridMap:
    """
    Proper Bayesian Occupancy Grid Map using log-odds representation.

    Based on: Thrun, Burgard, Fox - "Probabilistic Robotics", Chapter 9
    and reference implementation from lukovicaleksa/grid-mapping-in-ROS.

    The Bayes Binary Filter update rule in log-odds form:
        l(x|z) = l(x|z_prev) + log_odds(p(x|z)) - log_odds(p_prior)

    This is equivalent to the recursive Bayes filter:
        P(occ|z1..zt) ∝ P(zt|occ) * P(occ|z1..zt-1) / P(zt)
    """

    def __init__(self, x_lim, y_lim, resolution, p_prior=0.5):
        self.x_lim = x_lim
        self.y_lim = y_lim
        self.resolution = resolution

        # Grid dimensions
        x_cells = int((x_lim[1] - x_lim[0]) / resolution) + 1
        y_cells = int((y_lim[1] - y_lim[0]) / resolution) + 1
        self.shape = (x_cells, y_cells)

        # Log-odds grid — initialized with prior
        self.l = np.full(self.shape, log_odds(p_prior), dtype=np.float64)

        print(f"[MAP] Grid: {x_cells}x{y_cells} cells, "
              f"resolution={resolution*100:.0f}cm, "
              f"area={x_lim[1]-x_lim[0]:.0f}x{y_lim[1]-y_lim[0]:.0f}m")

    def discretize(self, x_cont, y_cont):
        """Convert continuous world coordinates to grid indices."""
        gx = int((x_cont - self.x_lim[0]) / self.resolution)
        gy = int((y_cont - self.y_lim[0]) / self.resolution)
        return gx, gy

    def in_bounds(self, gx, gy):
        """Check if grid cell is within bounds."""
        return 0 <= gx < self.shape[0] and 0 <= gy < self.shape[1]

    def update(self, gx, gy, p_sensor):
        """
        Bayesian update of a single cell using inverse sensor model.

        l(x|z) = l(x|z_prev) + log_odds(p_sensor) - l_prior

        p_sensor = P_FREE for cells along the ray (free space)
        p_sensor = P_OCC for the hit endpoint (occupied)
        """
        if not self.in_bounds(gx, gy):
            return
        self.l[gx, gy] = np.clip(
            self.l[gx, gy] + log_odds(p_sensor) - L_PRIOR,
            L_FREE_MIN, L_OCC_MAX
        )

    def get_probability(self, gx, gy):
        """Get occupancy probability of a cell."""
        if not self.in_bounds(gx, gy):
            return P_PRIOR
        return retrieve_p(self.l[gx, gy])

    def to_probability_image(self):
        """Convert entire grid to probability image [0, 1]."""
        return retrieve_p(self.l)

    def to_grayscale_image(self):
        """Convert to grayscale: 1.0=free(white), 0.0=occupied(black)."""
        return 1.0 - retrieve_p(self.l)

    def to_rgb_image(self):
        """
        Standard occupancy grid visualization:
          - Black (0,0,0) = occupied (high probability)
          - White (1,1,1) = free (low probability)
          - Gray  (0.5)   = unknown (prior)
        Exactly like the reference: grayscale = 1 - P(occupied)
        """
        prob = retrieve_p(self.l)
        gray = 1.0 - prob  # occupied→0(black), free→1(white), unknown→0.5(gray)
        # Stack to RGB
        return np.stack([gray, gray, gray], axis=-1).astype(np.float32)

    def calc_mle(self):
        """Calculate Maximum Likelihood Estimate of the map."""
        mle = np.full(self.shape, log_odds(0.5))
        prob = retrieve_p(self.l)
        mle[prob < THRESH_P_FREE] = log_odds(0.01)
        mle[prob > THRESH_P_OCC] = log_odds(0.99)
        return mle

    def find_neighbours(self, gx, gy):
        """Find valid 8-connected neighbours of a cell."""
        neighbours = []
        for dx in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                if dx == 0 and dy == 0:
                    continue
                nx, ny = gx + dx, gy + dy
                if self.in_bounds(nx, ny):
                    neighbours.append((nx, ny))
        return neighbours


# ======================== BRESENHAM ========================

def bresenham(grid_map, x1, y1, x2, y2):
    """
    Bresenham's line drawing algorithm — all 4 quadrants.
    Returns list of (x, y) grid cells along the line from (x1,y1) to (x2,y2).
    The endpoint (x2, y2) is NOT included (it's the hit point, handled separately).

    Reference: lukovicaleksa/grid-mapping-in-ROS/scripts/bresenham.py
    """
    cells = []
    x, y = x1, y1

    delta_x = abs(x2 - x1)
    delta_y = abs(y2 - y1)

    s_x = 1 if x2 > x1 else -1 if x2 < x1 else 0
    s_y = 1 if y2 > y1 else -1 if y2 < y1 else 0

    if delta_x == 0 and delta_y == 0:
        return cells

    interchange = delta_y > delta_x
    if interchange:
        delta_x, delta_y = delta_y, delta_x

    A = 2 * delta_y
    B = 2 * (delta_y - delta_x)
    E = 2 * delta_y - delta_x

    cells.append((x, y))

    for _ in range(1, delta_x):
        if E < 0:
            if interchange:
                y += s_y
            else:
                x += s_x
            E += A
        else:
            y += s_y
            x += s_x
            E += B
        cells.append((x, y))

    return cells


# ======================== HELPERS ========================

def clamp(val, lo, hi):
    return max(lo, min(hi, val))


def normalize_angle(a):
    while a > math.pi:
        a -= 2.0 * math.pi
    while a < -math.pi:
        a += 2.0 * math.pi
    return a


# ======================== CONTROLLER ========================

# --- Turn-rate gating for mapping ---
MAX_TURN_RATE_FOR_MAPPING = 0.15  # [TUNABLE] rad/step — skip mapping when turning faster than this (0.05-0.30)


class RosbotController:
    def __init__(self):
        self.robot = Robot()

        # Create proper Bayesian grid map
        self.grid_map = OccupancyGridMap(
            x_lim=MAP_X_LIM, y_lim=MAP_Y_LIM,
            resolution=GRID_RESOLUTION, p_prior=P_PRIOR
        )

        # Robot pose
        self.x = 0.0
        self.y = 0.0
        self.theta = 0.0

        # Odometry
        self.prev_enc_left = 0.0
        self.prev_enc_right = 0.0
        self.odom_initialized = False

        # Angular velocity tracking (for turn-rate gating)
        self.prev_theta = 0.0
        self.omega = 0.0  # current angular velocity (rad/step)

        # PD controller
        self.prev_heading_error = 0.0

        # State machine
        self.state = STATE_FORWARD
        self.state_timer = 0
        self.target_heading = 0.0

        # Stuck detection
        self.stuck_check_x = 0.0
        self.stuck_check_y = 0.0
        self.stuck_check_timer = 0
        self.stuck_count = 0
        self.total_timesteps = 0

        # Overhead obstacle
        self.overhead_detected = False

        # VFH
        self.vfh_histogram = np.zeros(VFH_SECTORS)

        # LiDAR config
        self.lidar_fov = 0.0
        self.lidar_min_range = 0.0
        self.lidar_max_range = 0.0
        self.lidar_ok = False
        self.compass_ok = False
        self.depth_ok = False

        # Init hardware
        self._init_devices()

        # Init plot
        self._interactive_plot = False
        self.plot_fig = None
        self.plot_ax = None
        self.plot_img = None
        self.plot_arrow = None
        if PLOT_ENABLED and PLOTTING_AVAILABLE:
            self._init_plot()

    # ================== DEVICES ==================

    def _init_devices(self):
        # Motors
        self.motors = []
        for name in ["fl_wheel_joint", "fr_wheel_joint",
                     "rl_wheel_joint", "rr_wheel_joint"]:
            m = self.robot.getDevice(name)
            m.setPosition(float("inf"))
            m.setVelocity(0.0)
            self.motors.append(m)

        # Distance sensors
        self.ds = []
        for name in ["fl_range", "fr_range", "rl_range", "rr_range"]:
            d = self.robot.getDevice(name)
            d.enable(TIME_STEP)
            self.ds.append(d)

        # Encoders
        self.encoders = []
        for name in ["front left wheel motor sensor",
                     "front right wheel motor sensor",
                     "rear left wheel motor sensor",
                     "rear right wheel motor sensor"]:
            e = self.robot.getDevice(name)
            e.enable(TIME_STEP)
            self.encoders.append(e)

        # LiDAR
        self.lidar = self.robot.getDevice("laser")
        if self.lidar:
            self.lidar.enable(TIME_STEP)
            self.lidar.enablePointCloud()
            self.lidar_ok = True
            self.lidar_fov = self.lidar.getFov()
            self.lidar_min_range = self.lidar.getMinRange()
            self.lidar_max_range = self.lidar.getMaxRange()
            print(f"[INIT] LiDAR: fov={math.degrees(self.lidar_fov):.1f}° "
                  f"range=[{self.lidar_min_range:.2f}, {self.lidar_max_range:.1f}]m")

        # Compass
        self.compass = self.robot.getDevice("imu compass")
        if self.compass:
            self.compass.enable(TIME_STEP)
            self.compass_ok = True
            print("[INIT] Compass OK")

        # Gyro
        gyro = self.robot.getDevice("imu gyro")
        if gyro:
            gyro.enable(TIME_STEP)
            print("[INIT] Gyro OK")

        # Accelerometer
        accel = self.robot.getDevice("imu accelerometer")
        if accel:
            accel.enable(TIME_STEP)
            print("[INIT] Accel OK")

        # Depth camera
        self.depth_cam = self.robot.getDevice("camera depth")
        if self.depth_cam:
            self.depth_cam.enable(TIME_STEP)
            self.depth_ok = True
            print(f"[INIT] Depth camera OK "
                  f"({self.depth_cam.getWidth()}x{self.depth_cam.getHeight()})")

        # RGB camera
        rgb = self.robot.getDevice("camera rgb")
        if rgb:
            rgb.enable(TIME_STEP)
            print("[INIT] RGB camera OK")

    # ================== PLOT ==================

    def _init_plot(self):
        backend = matplotlib.get_backend().lower()
        self._interactive_plot = (backend != "agg")

        if self._interactive_plot:
            plt.ion()

        self.plot_fig, self.plot_ax = plt.subplots(1, 1, figsize=(9, 7))
        if self._interactive_plot:
            try:
                self.plot_fig.canvas.manager.set_window_title(
                    "ROSbot — Bayesian Occupancy Grid (Live)")
            except Exception:
                pass

        extent = [MAP_X_LIM[0], MAP_X_LIM[1], MAP_Y_LIM[0], MAP_Y_LIM[1]]
        # Initial gray image (unknown)
        init_img = np.full((self.grid_map.shape[1], self.grid_map.shape[0], 3), 0.5)
        self.plot_img = self.plot_ax.imshow(
            init_img, extent=extent, origin="lower", interpolation="nearest")

        self.plot_ax.set_xlabel("X (m)", fontsize=11)
        self.plot_ax.set_ylabel("Y (m)", fontsize=11)
        self.plot_ax.set_title("Bayesian Occupancy Grid Map", fontsize=13, fontweight="bold")
        self.plot_ax.grid(True, alpha=0.15, linestyle="--")
        self.plot_ax.set_aspect("equal")

        # Robot marker (red dot + heading arrow)
        self.plot_robot_dot, = self.plot_ax.plot([], [], "ro", markersize=8, zorder=5)
        self.plot_arrow = None

        # Status text overlay
        self.plot_status = self.plot_ax.text(
            0.02, 0.98, "", transform=self.plot_ax.transAxes,
            fontsize=9, verticalalignment="top", fontfamily="monospace",
            bbox=dict(boxstyle="round,pad=0.4", facecolor="black", alpha=0.85),
            color="lime"
        )

        plt.tight_layout()
        if self._interactive_plot:
            self.plot_fig.canvas.draw()
            self.plot_fig.canvas.flush_events()

    def update_plot(self, laser_hits_xy=None):
        if not PLOT_ENABLED or not PLOTTING_AVAILABLE or self.plot_fig is None:
            return

        # Standard occupancy grid image: black=wall, white=free, gray=unknown
        rgb = self.grid_map.to_rgb_image()

        # Mark robot position in blue on the grid image
        gx, gy = self.grid_map.discretize(self.x, self.y)
        if self.grid_map.in_bounds(gx, gy):
            rgb[gx, gy] = [0.0, 0.0, 1.0]
        for nx, ny in self.grid_map.find_neighbours(gx, gy):
            if self.grid_map.in_bounds(nx, ny):
                rgb[nx, ny] = [0.0, 0.0, 1.0]

        # Transpose grid[gx,gy] → image[gy,gx] for imshow (row=Y, col=X)
        # NO flip — origin="lower" already puts Y=0 at bottom
        display = np.transpose(rgb, (1, 0, 2))
        self.plot_img.set_data(np.clip(display, 0, 1))

        # Robot marker + heading arrow (plotted in world coordinates)
        self.plot_robot_dot.set_data([self.x], [self.y])
        if self.plot_arrow is not None:
            self.plot_arrow.remove()
            self.plot_arrow = None
        al = 0.15
        self.plot_arrow = self.plot_ax.annotate(
            "", xy=(self.x + al * math.cos(self.theta),
                    self.y + al * math.sin(self.theta)),
            xytext=(self.x, self.y),
            arrowprops=dict(arrowstyle="->", color="red", lw=2.5), zorder=6)

        # Status text
        state_name = STATE_NAMES.get(self.state, "?")
        self.plot_status.set_text(
            f"Pose: ({self.x:.2f}, {self.y:.2f}, {math.degrees(self.theta):.1f}°)\n"
            f"State: {state_name}  Stuck: {self.stuck_count}\n"
            f"Overhead: {self.overhead_detected}")

        if self._interactive_plot:
            self.plot_fig.canvas.draw_idle()
            self.plot_fig.canvas.flush_events()
        else:
            self.plot_fig.savefig("occupancy_grid_live.png", dpi=100,
                                 bbox_inches="tight")

    # ================== COMPASS ==================

    def get_compass_heading(self):
        """
        Webots R2025a with default northDirection = (1, 0, 0).
        Ground plane = XY, Z = Up. Robot FLU: X=Forward, Y=Left, Z=Up.

        Compass returns north vector in robot's local FLU frame.
        With northDirection=(1,0,0), the compass vector is:
          cn = (cos θ, -sin θ, 0)

        To recover heading θ (angle from +X world, CCW):
          θ = atan2(sin θ, cos θ) = atan2(-cn[1], cn[0])

        Verification:
          θ=0   (facing +X): cn=(1, 0,0)  → atan2( 0, 1) = 0    ✓
          θ=π/2 (facing +Y): cn=(0,-1,0)  → atan2( 1, 0) = π/2  ✓
          θ=π   (facing -X): cn=(-1,0,0)  → atan2( 0,-1) = π    ✓
          θ=-π/2(facing -Y): cn=(0, 1,0)  → atan2(-1, 0) = -π/2 ✓
        """
        if not self.compass_ok:
            return self.theta
        cn = self.compass.getValues()
        return math.atan2(-cn[1], cn[0])

    # ================== ODOMETRY ==================

    def update_odometry(self):
        enc_left = self.encoders[0].getValue()
        enc_right = self.encoders[1].getValue()

        if not self.odom_initialized:
            self.prev_enc_left = enc_left
            self.prev_enc_right = enc_right
            self.odom_initialized = True
            self.theta = self.get_compass_heading()
            self.prev_theta = self.theta
            self.x = INITIAL_X
            self.y = INITIAL_Y
            self.stuck_check_x = self.x
            self.stuck_check_y = self.y
            # Compass diagnostic — verify heading formula
            if self.compass_ok:
                cn = self.compass.getValues()
                print(f"[COMPASS] Raw values: ({cn[0]:.4f}, {cn[1]:.4f}, {cn[2]:.4f})")
                print(f"[COMPASS] Heading: {math.degrees(self.theta):.1f}° "
                      f"(expected ~0° for robot facing +X)")
            return

        dl = (enc_left - self.prev_enc_left) * WHEEL_RADIUS
        dr = (enc_right - self.prev_enc_right) * WHEEL_RADIUS
        self.prev_enc_left = enc_left
        self.prev_enc_right = enc_right

        ds = (dl + dr) / 2.0

        # Use compass directly — it's ground truth in Webots, no need to blend
        # Blending (0.95*compass + 0.05*odom) adds jitter during turns
        if self.compass_ok:
            self.theta = self.get_compass_heading()
        else:
            dtheta = (dr - dl) / WHEEL_BASE
            self.theta = normalize_angle(self.theta + dtheta)

        # Track angular velocity (rad/step) for turn-rate gating
        self.omega = abs(normalize_angle(self.theta - self.prev_theta))
        self.prev_theta = self.theta

        self.x += ds * math.cos(self.theta)
        self.y += ds * math.sin(self.theta)

    # ================== OVERHEAD DETECTION ==================

    def check_overhead_obstacles(self):
        self.overhead_detected = False
        if not self.depth_ok:
            return
        w = self.depth_cam.getWidth()
        h = self.depth_cam.getHeight()
        depth = self.depth_cam.getRangeImage()
        if depth is None or len(depth) == 0:
            return

        overhead_row_end = int(h * OVERHEAD_HEIGHT_RATIO)
        col_start, col_end = w // 5, w * 4 // 5
        close, total = 0, 0

        for row in range(0, overhead_row_end, 2):
            for col in range(col_start, col_end, 2):
                d = depth[row * w + col]
                total += 1
                if 0.05 < d < OVERHEAD_DANGER_DIST and not math.isinf(d) and not math.isnan(d):
                    close += 1

        if total > 0 and close / total > OVERHEAD_PIXEL_THRESHOLD:
            self.overhead_detected = True

    # ================== BAYESIAN GRID MAPPING ==================

    def update_map_from_lidar(self):
        """
        Proper Bayesian grid mapping from LiDAR — matches reference exactly:
          https://github.com/lukovicaleksa/grid-mapping-in-ROS

        Key design (from reference utils.py lidar_scan_xy):
          hit_x = x_odom + dist * cos(angle + theta_odom)
          hit_y = y_odom + dist * sin(angle + theta_odom)

        Then:
          x1, y1 = discretize(x_odom, y_odom)   — robot position
          x2, y2 = discretize(hit_x, hit_y)     — laser hit
          bresenham(x1, y1, x2, y2)             — free cells
          update(x2, y2, P_occ)                 — occupied cell

        Turn-rate gating: skip mapping when robot is rotating fast,
        to prevent heading mismatch from smearing the map.
        """
        if not self.lidar_ok:
            return None

        # TURN-RATE GATING: skip mapping during fast rotation
        # This is the key fix — the reference gets a consistent odom+scan snapshot
        # at 10Hz. We run at ~30Hz but during turns the heading changes between
        # consecutive steps, causing radial smearing.
        if self.omega > MAX_TURN_RATE_FOR_MAPPING:
            return None

        ranges = self.lidar.getRangeImage()
        if ranges is None or len(ranges) == 0:
            return None

        n = len(ranges)
        angle_inc = self.lidar_fov / n

        # Snapshot current pose (consistent for all rays in this scan)
        x_odom = self.x
        y_odom = self.y
        theta_odom = self.theta

        # Robot position in grid coordinates (Bresenham start point)
        # Exactly like reference: x1, y1 = gridMap.discretize(x_odom, y_odom)
        x1, y1 = self.grid_map.discretize(x_odom, y_odom)

        laser_hits = []

        for i in range(n):
            r = ranges[i]
            if math.isnan(r) or math.isinf(r):
                continue
            if r < self.lidar_min_range:
                continue

            # Webots LiDAR: angle runs from -fov/2 to +fov/2
            angle = -self.lidar_fov / 2.0 + i * angle_inc

            # Clamp to max range (reference: if dist > range_max: dist = range_max)
            dist = min(r, self.lidar_max_range)

            # Hit point in world coordinates
            # Exactly like reference: x_odom + dist * cos(angle + theta_odom)
            hit_x = x_odom + dist * math.cos(angle + theta_odom)
            hit_y = y_odom + dist * math.sin(angle + theta_odom)

            # Discretize hit point
            x2, y2 = self.grid_map.discretize(hit_x, hit_y)

            # Bresenham ray: mark all cells along ray as FREE
            # Exactly like reference: bresenham(gridMap, x1, y1, x2, y2)
            for (bx, by) in bresenham(self.grid_map, x1, y1, x2, y2):
                self.grid_map.update(bx, by, P_FREE)

            # Mark hit endpoint as OCCUPIED (only if actual hit, not max range)
            # Exactly like reference: if dist < msgScan.range_max
            if r < self.lidar_max_range:
                self.grid_map.update(x2, y2, P_OCC)
                laser_hits.append([hit_x, hit_y])

        return np.array(laser_hits) if laser_hits else None

    def update_map_from_distance_sensors(self):
        """Bayesian update from IR distance sensors using same Bresenham approach."""
        x1, y1 = self.grid_map.discretize(self.x, self.y)

        for i in range(4):
            r = self.ds[i].getValue()
            if r < 0.02 or r > 1.95:
                continue

            abs_angle = self.theta + DS_ANGLES[i]
            hit_x = self.x + r * math.cos(abs_angle)
            hit_y = self.y + r * math.sin(abs_angle)

            x2, y2 = self.grid_map.discretize(hit_x, hit_y)

            for (bx, by) in bresenham(self.grid_map, x1, y1, x2, y2):
                self.grid_map.update(bx, by, P_FREE)

            if r < 1.9:
                self.grid_map.update(x2, y2, P_OCC)

    def update_map_from_depth_camera(self):
        """Bayesian update from depth camera middle band."""
        if not self.depth_ok:
            return

        w = self.depth_cam.getWidth()
        h = self.depth_cam.getHeight()
        fov = self.depth_cam.getFov()
        depth = self.depth_cam.getRangeImage()
        if depth is None:
            return

        x1, y1 = self.grid_map.discretize(self.x, self.y)
        mid_row = h // 2

        for col in range(0, w, 4):  # Sample every 4th column
            d = depth[mid_row * w + col]
            if d <= 0.05 or math.isinf(d) or math.isnan(d) or d > 4.0:
                continue

            pixel_angle = (col / w - 0.5) * fov
            abs_angle = self.theta + pixel_angle
            hit_x = self.x + d * math.cos(abs_angle)
            hit_y = self.y + d * math.sin(abs_angle)

            x2, y2 = self.grid_map.discretize(hit_x, hit_y)

            for (bx, by) in bresenham(self.grid_map, x1, y1, x2, y2):
                self.grid_map.update(bx, by, P_FREE)

            self.grid_map.update(x2, y2, P_OCC)

    # ================== VFH ==================

    def build_vfh_histogram(self):
        self.vfh_histogram[:] = 0.0

        if self.lidar_ok:
            ranges = self.lidar.getRangeImage()
            if ranges and len(ranges) > 0:
                n = len(ranges)
                ai = self.lidar_fov / n
                for i in range(n):
                    r = ranges[i]
                    if r < self.lidar_min_range or r > 2.0 or math.isnan(r) or math.isinf(r):
                        continue
                    angle = -self.lidar_fov / 2.0 + i * ai
                    if angle < 0:
                        angle += 2.0 * math.pi
                    sec = int(angle / (2.0 * math.pi) * VFH_SECTORS) % VFH_SECTORS
                    self.vfh_histogram[sec] += 4.0 / (r * r)

        for i in range(4):
            r = self.ds[i].getValue()
            if r < 0.02 or r > 1.8:
                continue
            angle = DS_ANGLES[i]
            if angle < 0:
                angle += 2.0 * math.pi
            sec = int(angle / (2.0 * math.pi) * VFH_SECTORS) % VFH_SECTORS
            mag = 4.0 / (r * r)
            for s in range(-2, 3):
                idx = (sec + s + VFH_SECTORS) % VFH_SECTORS
                self.vfh_histogram[idx] += mag * (1.0 - 0.2 * abs(s))

        if self.overhead_detected:
            for s in range(VFH_SECTORS):
                sa = s / VFH_SECTORS * 2.0 * math.pi
                if sa > math.pi:
                    sa -= 2.0 * math.pi
                if abs(sa) < math.pi / 3.0:
                    self.vfh_histogram[s] += VFH_OVERHEAD_BOOST

    def find_best_vfh_direction(self):
        blocked = [self.vfh_histogram[i] > VFH_THRESHOLD for i in range(VFH_SECTORS)]
        smoothed = [False] * VFH_SECTORS
        for i in range(VFH_SECTORS):
            s = sum(1 for j in range(-1, 2) if blocked[(i + j) % VFH_SECTORS])
            smoothed[i] = s >= 2

        openings = []
        i = 0
        while i < VFH_SECTORS:
            if not smoothed[i]:
                start = i
                while i < VFH_SECTORS and not smoothed[i]:
                    i += 1
                openings.append({"s": start, "e": i - 1, "w": i - start})
            else:
                i += 1

        if len(openings) >= 2 and openings[0]["s"] == 0 and openings[-1]["e"] == VFH_SECTORS - 1:
            openings[0]["s"] = openings[-1]["s"]
            openings[0]["w"] += openings[-1]["w"]
            openings.pop()

        if not openings:
            return None

        goal_sec = 0
        best_score, best_center = -1e9, 0

        for o in openings:
            if o["w"] >= VFH_WIDE_OPENING:
                ld = abs(o["s"] - goal_sec)
                rd = abs(o["e"] - goal_sec)
                if ld > VFH_SECTORS // 2: ld = VFH_SECTORS - ld
                if rd > VFH_SECTORS // 2: rd = VFH_SECTORS - rd
                c = (o["s"] + VFH_WIDE_OPENING // 2) % VFH_SECTORS if ld < rd else \
                    (o["e"] - VFH_WIDE_OPENING // 2 + VFH_SECTORS) % VFH_SECTORS
            else:
                c = (o["s"] + o["w"] // 2) % VFH_SECTORS
            ad = abs(c - goal_sec)
            if ad > VFH_SECTORS // 2: ad = VFH_SECTORS - ad
            score = -float(ad) + 0.5 * o["w"]
            if score > best_score:
                best_score, best_center = score, c

        d = best_center / VFH_SECTORS * 2.0 * math.pi
        if d > math.pi:
            d -= 2.0 * math.pi
        return d

    # ================== MOTOR CONTROL ==================

    def set_speeds(self, left, right):
        left = clamp(left, -MAX_SPEED, MAX_SPEED)
        right = clamp(right, -MAX_SPEED, MAX_SPEED)
        for i, spd in enumerate([left, right, left, right]):
            self.motors[i].setVelocity(spd)

    def steer_toward(self, heading_error, speed):
        heading_error = normalize_angle(heading_error)
        de = heading_error - self.prev_heading_error
        self.prev_heading_error = heading_error
        c = clamp(KP_TURN * heading_error + KD_TURN * de, -MAX_SPEED, MAX_SPEED)
        self.set_speeds(speed - c, speed + c)

    # ================== SENSOR QUERIES ==================

    def get_front_min_dist(self):
        min_d = 99.0
        for i in range(2):
            d = self.ds[i].getValue()
            if d < min_d:
                min_d = d
        if self.lidar_ok:
            ranges = self.lidar.getRangeImage()
            if ranges and len(ranges) > 0:
                n = len(ranges)
                ai = self.lidar_fov / n
                for i in range(n):
                    a = -self.lidar_fov / 2.0 + i * ai
                    if abs(a) > 0.5:
                        continue
                    r = ranges[i]
                    if self.lidar_min_range <= r < min_d and not math.isnan(r) and not math.isinf(r):
                        min_d = r
        return min_d

    def get_side_min_dist(self, a_min, a_max):
        if not self.lidar_ok:
            return 99.0
        ranges = self.lidar.getRangeImage()
        if not ranges:
            return 99.0
        n = len(ranges)
        min_d = 99.0
        ai = self.lidar_fov / n
        for i in range(n):
            a = -self.lidar_fov / 2.0 + i * ai
            if a < a_min or a > a_max:
                continue
            r = ranges[i]
            if self.lidar_min_range <= r < min_d and not math.isnan(r) and not math.isinf(r):
                min_d = r
        return min_d

    # ================== STUCK DETECTION ==================

    def check_stuck(self):
        self.stuck_check_timer += 1
        if self.stuck_check_timer < STUCK_CHECK_INTERVAL:
            return
        self.stuck_check_timer = 0
        dx = self.x - self.stuck_check_x
        dy = self.y - self.stuck_check_y
        if math.sqrt(dx * dx + dy * dy) < STUCK_DIST_THRESHOLD:
            self.stuck_count += 1
            if self.stuck_count >= MAX_STUCK_COUNT:
                print(f"[STUCK] Aggressive recovery (count={self.stuck_count})")
                self.state = STATE_ESCAPE
                self.state_timer = 0
        else:
            self.stuck_count = 0
        self.stuck_check_x = self.x
        self.stuck_check_y = self.y

    # ================== STATE MACHINE ==================

    def run_state_machine(self):
        self.state_timer += 1
        self.total_timesteps += 1

        front = self.get_front_min_dist()
        left = self.get_side_min_dist(0.5, 1.5)
        right = self.get_side_min_dist(-1.5, -0.5)

        eff = front
        if self.overhead_detected:
            eff = min(eff, DANGER_DIST * 0.8)
            if self.state_timer % 20 == 0:
                print("[OVERHEAD] Floating wall!")

        self.build_vfh_histogram()
        self.check_stuck()

        if self.state == STATE_FORWARD:
            if eff < DANGER_DIST:
                self.state, self.state_timer = STATE_REVERSE, 0
            elif eff < CAUTION_DIST:
                self.state, self.state_timer = STATE_AVOID, 0
            else:
                vfh = self.find_best_vfh_direction()
                if vfh is not None:
                    self.steer_toward(clamp(vfh * 0.3, -0.5, 0.5), CRUISE_SPEED)
                else:
                    self.set_speeds(CRUISE_SPEED, CRUISE_SPEED)

        elif self.state == STATE_AVOID:
            if eff < DANGER_DIST:
                self.state, self.state_timer = STATE_REVERSE, 0
                return
            vfh = self.find_best_vfh_direction()
            if vfh is None:
                self.state, self.state_timer = STATE_ROTATE, 0
                self.target_heading = math.pi / 2 if left > right else -math.pi / 2
                return
            if abs(vfh) > math.pi / 4 and (left < CLEAR_DIST or right < CLEAR_DIST):
                self.state, self.state_timer = STATE_WALL_FOLLOW, 0
                return
            self.steer_toward(vfh, SLOW_SPEED if eff < CAUTION_DIST else CRUISE_SPEED)
            if eff > CLEAR_DIST and abs(vfh) < 0.3:
                self.state, self.state_timer = STATE_FORWARD, 0

        elif self.state == STATE_WALL_FOLLOW:
            if eff < DANGER_DIST:
                self.state, self.state_timer = STATE_REVERSE, 0
                return
            if eff < CAUTION_DIST * 0.7:
                td = 1.0 if left > right else -1.0
                self.set_speeds(-TURN_SPEED * td, TURN_SPEED * td)
            else:
                if left < right:
                    err = -(left - WALL_FOLLOW_DIST)
                else:
                    err = (right - WALL_FOLLOW_DIST)
                c = clamp(KP_WALL * err, -SLOW_SPEED, SLOW_SPEED)
                sp = SLOW_SPEED if eff < CAUTION_DIST else CRUISE_SPEED * 0.7
                self.set_speeds(sp - c, sp + c)
            if eff > CLEAR_DIST * 1.5 and left > CLEAR_DIST and right > CLEAR_DIST:
                self.state, self.state_timer = STATE_FORWARD, 0
            if self.state_timer > WALL_FOLLOW_TIMEOUT:
                self.state, self.state_timer = STATE_FORWARD, 0

        elif self.state == STATE_ROTATE:
            td = 1.0 if self.target_heading > 0 else -1.0
            self.set_speeds(-TURN_SPEED * td, TURN_SPEED * td)
            vfh = self.find_best_vfh_direction()
            if vfh is not None and eff > CAUTION_DIST:
                self.state, self.state_timer = STATE_AVOID, 0
            if self.state_timer > 80:
                self.target_heading = -self.target_heading
                self.state_timer = 0

        elif self.state == STATE_REVERSE:
            self.set_speeds(-SLOW_SPEED, -SLOW_SPEED)
            if self.state_timer > 20 or front > CAUTION_DIST:
                self.state, self.state_timer = STATE_ROTATE, 0
                self.target_heading = math.pi / 2 if left > right else -math.pi / 2

        elif self.state == STATE_ESCAPE:
            if self.state_timer < ESCAPE_BACKUP_STEPS:
                self.set_speeds(-CRUISE_SPEED * ESCAPE_BACKUP_SPEED,
                                -CRUISE_SPEED * ESCAPE_BACKUP_SPEED)
            elif self.state_timer < ESCAPE_BACKUP_STEPS + ESCAPE_ROTATE_STEPS:
                td = 1.0 if (self.total_timesteps // 100) % 2 == 0 else -1.0
                if self.stuck_count > 4:
                    td = -td
                self.set_speeds(-TURN_SPEED * ESCAPE_ROTATE_MULTIPLIER * td,
                                TURN_SPEED * ESCAPE_ROTATE_MULTIPLIER * td)
            else:
                self.state, self.state_timer, self.stuck_count = STATE_FORWARD, 0, 0
                print("[ESCAPE] Complete")

    # ================== MAIN LOOP ==================

    def run(self):
        print("=== ROSbot Controller v3 (Bayes Binary Filter) ===")
        print(f"  P_OCC={P_OCC}  P_FREE={P_FREE}  P_PRIOR={P_PRIOR}")
        print(f"  Plot: {'ON' if PLOT_ENABLED and PLOTTING_AVAILABLE else 'OFF'}")
        print("=" * 50)

        t = 0
        while self.robot.step(TIME_STEP) != -1:
            t += 1

            self.update_odometry()
            self.check_overhead_obstacles()

            # Bayesian grid mapping — LiDAR only (clean Bayes Binary Filter)
            # Turn-rate gating is inside update_map_from_lidar:
            #   skips updates when omega > MAX_TURN_RATE_FOR_MAPPING
            laser_hits = self.update_map_from_lidar()

            # Navigation
            self.run_state_machine()

            # Console
            if t % 100 == 0:
                sn = STATE_NAMES.get(self.state, "?")
                print(f"[t={t}] ({self.x:.3f}, {self.y:.3f}, "
                      f"{math.degrees(self.theta):.1f}°) "
                      f"state={sn} stuck={self.stuck_count} "
                      f"omega={math.degrees(self.omega):.1f}°/step")

            # Live plot
            if t % PLOT_UPDATE_INTERVAL == 0:
                self.update_plot(laser_hits)

        # Final save
        if PLOTTING_AVAILABLE and self.plot_fig:
            self.plot_fig.savefig("occupancy_grid_final.png", dpi=150,
                                 bbox_inches="tight")
            print("[END] Final map saved to occupancy_grid_final.png")


if __name__ == "__main__":
    RosbotController().run()
