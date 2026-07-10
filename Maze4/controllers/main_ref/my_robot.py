from controller import Robot
import numpy as np
import math
import cv2
import random
import time
import threading
from collections import deque
import utils
from CONSTANTS import *
from map import OccupancyGrid
from setup import setup_robot


class MyRobot(Robot):

    # Confidence weights for column-position observations, used by
    # update_column_estimation to weight-blend new sightings into the
    # running estimate -- a close, front-facing depth fix is trusted far
    # more than an ordinary depth fix, which in turn outweighs a distant
    # bearing-only guess (see _column_bearing_local).
    COLUMN_WEIGHT_CLOSE  = 5.0
    COLUMN_WEIGHT_NORMAL = 2.0
    COLUMN_WEIGHT_ROUGH  = 0.3
    COLUMN_WEIGHT_CAP    = 20.0

    # Minimum colour-mask pixel count before a column sighting is trusted
    # enough to even ask the LiDAR for a range -- rejects specular
    # highlights and single-pixel colour noise (see Fairfield & Maxwell's
    # 5px floor for the same reason).
    COLUMN_MIN_LIDAR_BLOB_PX = 10

    # ── Construction ─────────────────────────────────────────────────────────

    def __init__(self):
        super().__init__()
        (self.motors, self.wheel_sensors,
         self.imu, self.camera_rgb, self.camera_depth,
         self.lidar, self.distance_sensors) = setup_robot(self)

        self.time_step    = TIME_STEP
        self.wheel_radius = WHEEL_RADIUS
        self.axle_length  = AXLE_LENGTH

        self.occ_map  = OccupancyGrid(robot=self)
        self.grid_map = self.occ_map.grid_map   # shared reference

        self.last_turn  = 'right'
        self.start_point = None
        self.end_point   = None
        self.path        = []
        self.interrupt_path = False
        self.chosen_frontier_count = 0

        self.blue_estimated_pos   = None
        self.yellow_estimated_pos = None
        # Running confidence total per colour, used to weight-blend new
        # observations into the estimate (see update_column_estimation) --
        # a close, front-facing depth fix counts far more than a distant
        # bearing-only guess, so a stray rough sighting can never drag a
        # confident estimate around once one exists.
        self.blue_estimate_weight    = 0.0
        self.yellow_estimate_weight  = 0.0
        # Explicit per-colour state: UNSEEN -> ROUGH -> PROVISIONAL -> CONFIRMED.
        # UNSEEN: nothing seen yet. ROUGH: only low-confidence bearing-only
        # sightings so far. PROVISIONAL: a real depth-based fix has landed
        # and accumulated confidence clears the commit-ready threshold --
        # start_point/end_point get set here. CONFIRMED: a close, front-facing
        # observation landed -- position is frozen from this point on and
        # used as the reachability anchor for the final pillar-to-pillar path.
        self._column_state = {'blue': 'UNSEEN', 'yellow': 'UNSEEN'}
        self.blue_estimation_positions   = []
        self.yellow_estimation_positions = []
        self.blue_prev_estimate_position   = None
        self.yellow_prev_estimate_position = None
        self.estimation_distance_threshold = 0.5
        # Floating-wall detection: height-band marking with confirmed,
        # frustum-gated persistence (see _refresh_map_depth). Every point
        # `_depth_obstacle_points_local` returns already lies inside the
        # robot-blocking height band [GROUND_EPSILON_M, ROBOT_CLEARANCE_HEIGHT_M]
        # by construction, so there is no separate "is it low enough to
        # block" decision left to make — every map cell the depth camera
        # (or the IR short-range backstop) ever reports here casts a vote;
        # a cell is only trusted and drawn once it has
        # FLOATING_WALL_CONFIRM_VOTES independent votes, and from that
        # point on it is frozen in position (never re-fit, never moved) and
        # can only leave the map via the much stricter frustum-gated
        # clearing pass (_frustum_clear_floating), never by simple
        # per-frame reclassification.
        self._floating_votes = {}
        # Closest forward distance (m) each candidate cell has ever been
        # observed at, across frames -- lets a cell first glimpsed only up
        # close qualify for the reduced FLOATING_WALL_CONFIRM_VOTES_CLOSE
        # threshold (see _refresh_map_depth) instead of always needing the
        # full FLOATING_WALL_CONFIRM_VOTES.
        self._floating_best_range = {}
        self._floating_confirmed = set()
        # Contradiction counter for frustum-gated clearing: how many
        # distinct frames have shown the camera's own line of sight passing
        # clean through a confirmed cell to something farther away. Reset
        # whenever the cell is NOT contradicted; the cell is only removed
        # once this reaches DEPTH_CLEAR_CONFIRMATIONS (see
        # _frustum_clear_floating) — clearing is deliberately much harder
        # to trigger than marking.
        self._floating_clear_votes = {}
        # Union-find over confirmed cells: cells belonging to the same
        # physical wall (touching or bridged together) share one root. Used
        # only to close interior/seam gaps (_floating_solidify_group,
        # _floating_bridge_gap) — every confirmed cell already blocks the
        # robot on its own, so no per-group verdict is needed any more.
        self._floating_group = {}         # cell -> parent cell (path-compressed)
        self._floating_group_members = {} # root cell -> set of member cells

        self.steps_since_turning  = 0
        self.is_currently_turning = False

        self.detection_thread        = None
        self.camera_thread_running   = False
        self.camera_detection_signal = None
        self.camera_detection_queue  = deque(maxlen=6)
        self.detection_lock          = threading.Lock()

        self.lidar_thread        = None
        self.lidar_thread_running = False
        self.lidar_lock          = threading.Lock()

        self._rt_planner_thread  = None
        self._rt_planner_running = False
        self._rt_planner_goal    = None   # goal (map coords) being monitored
        self._rt_active_path     = None   # path currently being followed (written by nav)
        self._rt_new_path        = None   # freshly replanned path (written by planner)
        self._rt_replan_ready    = False  # True when _rt_new_path is ready to consume
        self._rt_last_replan_time = 0.0
        self._rt_planner_lock    = threading.Lock()

        self.stuck_thread        = None
        self.stuck_thread_running = False
        self.stuck_signal        = False
        self.stuck_last_position = None
        self.stuck_lock          = threading.Lock()

        self.follow_target_last_position      = None
        self.follow_target_stuck_count        = 0
        self.follow_target_position_threshold = 0.005
        self.follow_target_stuck_threshold    = 25
        self._phys_stuck_count                = 0
        self._last_scan_fp                    = None  # LiDAR fingerprint for slip detection
        self._scan_stuck_count                = 0
        self._waypoint_motion_commanded       = False
        self._last_cmd_left                   = 0.0
        self._last_cmd_right                  = 0.0
        self._dwa_no_motion_count             = 0
        self._robot_trail                     = set() # coarsened path history for frontier anti-revisit
        self._last_frontier_goal              = None  # last successfully reached frontier centroid
        self._active_frontier_goal            = None
        self._column_focus_color              = None
        self._column_focus_target             = None
        self._column_focus_blocked_until      = {}
        self._last_column_signal_time         = 0.0
        self._last_column_signal_color        = None
        self._last_detection_emit             = {}
        self._last_detection_handled          = {}
        self._camera_seen_counts              = {}

        try:
            self.cam_width   = self.camera_rgb.getWidth()
            self.cam_height  = self.camera_rgb.getHeight()
            self.cam_fov_rad = self.camera_rgb.getFov()
            self.fx = self.cam_width / (2.0 * np.tan(self.cam_fov_rad / 2.0))
            self.fy = self.fx
            self.cx = self.cam_width  / 2.0
            self.cy = self.cam_height / 2.0
        except Exception:
            self.cam_width, self.cam_height = 320, 240
            self.cam_fov_rad = 1.0472
            self.fx = self.fy = 240.0
            self.cx, self.cy = 160.0, 120.0

        # Sensor mount extrinsics, derived from the actual Webots proto chain
        # (Rosbot.proto -> Astra.proto / RpLidarA2.proto), relative to the
        # robot's own local origin — the same origin the wheel-encoder
        # odometry (_odom_x/_odom_y) tracks. These are composed poses, not a
        # single translation: Rosbot.proto mounts the Astra HOUSING at
        # (-0.027, 0, 0.165), but the RangeFinder ("camera depth") element
        # inside Astra.proto has its own local offset of (0.027, 0.037,
        # 0.034) relative to that housing. Composing them:
        #   X = -0.027 + 0.027 = 0.000   (the -0.027 housing offset and the
        #                                 +0.027 internal offset cancel — the
        #                                 depth sensor sits exactly on the
        #                                 robot's forward centerline)
        #   Y =  0.000 + 0.037 = 0.037   (offset sideways — previously assumed
        #                                 zero, which is wrong)
        #   Z =  0.165 + 0.034 = 0.199   (not the housing's own 0.165)
        # Using the housing translation alone (as an earlier pass here did,
        # with X_offset=-0.027) still left a real bias: any forward/lateral
        # offset error, expressed in the robot's frame, lands in a different
        # world (x,y) position depending on the robot's heading when rotated
        # into world coordinates — which looks exactly like a residual
        # per-viewing-angle "angle" error even after the PCA/cardinal-snap
        # fix and the earlier (wrong) X_offset correction.
        self.camera_height_m  = 0.199
        # Refined once at runtime by _calibrate_camera_height() from real
        # floor pixels — the proto composition above is the starting point,
        # but manufacturing/mount tolerance can still leave a residual bias
        # that only an empirical floor measurement can remove.
        self._camera_height_calibrated = False
        self.camera_pitch_rad = 0.0     # Astra mount rotation is identity
        self.X_offset = 0.0
        self.Y_offset = 0.037
        # RpLidarA2.proto's own default `translation` field is (0, 0, 0.031),
        # composed with Rosbot.proto's lidarSlot pose (0.02, 0, 0.1). Only Z
        # is affected (0.1 + 0.031 = 0.131) and the lidar's 2D point cloud
        # doesn't carry height, so X/Y are unaffected by that inner offset.
        self.LIDAR_X_OFFSET = 0.02      # RpLidarA2 translation x
        self.LIDAR_Y_OFFSET = 0.0       # RpLidarA2 translation y
        # Front IR range sensors (Rosbot.proto DistanceSensor nodes fl_range/
        # fr_range): mounted at x=0.10, y=+-0.05, z=0.053 m in the chassis
        # frame, yawed +-0.13 rad outward. Height 0.053 m sits inside the
        # floating-wall band (DEPTH_OBSTACLE_FLOATING_MIN/MAX_HEIGHT), and
        # unlike the Astra depth camera these read down to near 0 m — used
        # only to cover the depth camera's own blind gap below its minRange
        # (DEPTH_CAMERA_MIN_RANGE_M), see _ir_floating_wall_points_local.
        self.IR_FRONT_MOUNTS = (
            (0.10, 0.05, 0.130),
            (0.10, -0.05, -0.130),
        )
        self.IR_RANGE_HEIGHT_M = 0.053

        self.green_carpet_patches             = []
        self.green_carpet_proximity_threshold = GREEN_CARPET_PATCH_PROXIMITY_CELLS
        self.last_green_mark_time  = 0.0
        self.green_mark_cooldown   = 8.0
        self.last_green_carpet_points = []
        self.green_carpet_active   = False
        self.green_carpet_lock     = threading.Lock()

        self.counter_obstacle_recoveries = 0

        self._odom_x      = INITIAL_X
        self._odom_y      = INITIAL_Y
        self._odom_theta  = INITIAL_THETA
        self._odom_prev_left  = 0.0
        self._odom_prev_right = 0.0
        self._odom_initialized = False
        self._step_lock = threading.RLock()

    # ── step() override ───────────────────────────────────────────────────────

    def step(self, duration_ms=None):
        if duration_ms is None:
            duration_ms = self.time_step
        with self._step_lock:
            result = super().step(int(duration_ms))
            if result != -1:
                self._tick_odometry()
            return result

    # ── Odometry ──────────────────────────────────────────────────────────────

    def _tick_odometry(self):
        cn      = self.imu['compass'].getValues()
        heading = math.atan2(-cn[1], cn[0])

        left_enc  = (self.wheel_sensors['fl'].getValue() +
                     self.wheel_sensors['rl'].getValue()) / 2.0
        right_enc = (self.wheel_sensors['fr'].getValue() +
                     self.wheel_sensors['rr'].getValue()) / 2.0

        if not self._odom_initialized:
            self._odom_prev_left  = left_enc
            self._odom_prev_right = right_enc
            self._odom_theta      = heading
            self._odom_initialized = True
            return

        dl = (left_enc  - self._odom_prev_left)  * self.wheel_radius
        dr = (right_enc - self._odom_prev_right) * self.wheel_radius
        self._odom_prev_left  = left_enc
        self._odom_prev_right = right_enc

        ds          = (dl + dr) / 2.0
        prev_theta  = self._odom_theta
        self._odom_theta = heading
        mid_theta   = (prev_theta + self._odom_theta) / 2.0

        self._odom_x = self._odom_x + ds * math.cos(mid_theta)
        self._odom_y = self._odom_y + ds * math.sin(mid_theta)
        # Record coarsened position for frontier anti-revisit
        half = MAP_SIZE // 2
        _mx = half + int(round(self._odom_x / RESOLUTION))
        _my = half - int(round(self._odom_y / RESOLUTION))
        self._robot_trail.add((_mx >> 3, _my >> 3))

    # ── Pose accessors ────────────────────────────────────────────────────────

    def get_position(self):
        return np.array([self._odom_x, self._odom_y])

    def get_heading(self, kind='deg'):
        # Called from both the main thread and the lidar background thread
        # (via transform_points_to_world); serialize against step() and
        # other device reads (see get_pointcloud_2d).
        with self._step_lock:
            cn = self.imu['compass'].getValues()
        rad = math.atan2(-cn[1], cn[0])
        return rad if kind == 'rad' else np.degrees(rad)

    def get_map_position(self):
        half = self.occ_map.map_size // 2
        # Round-to-nearest, matching OccupancyGrid.world_pts_to_map's
        # np.rint (the LiDAR-built wall grid's own conversion). This used
        # to truncate x and ceil y -- two different, asymmetric roundings
        # that disagreed with world_pts_to_map by up to a cell on each
        # axis, which is exactly what let a projected point (a column
        # marker, a path waypoint, ...) land a cell or two off from where
        # the LiDAR wall grid actually put the same physical geometry.
        mx = half + int(round(self._odom_x / RESOLUTION))
        my = half - int(round(self._odom_y / RESOLUTION))
        return np.array([mx, my])

    def get_map_distance(self, map_target):
        return np.linalg.norm(self.get_map_position() - np.array(map_target))

    def convert_to_map_coordinates(self, x, y):
        half = self.occ_map.map_size // 2
        # See get_map_position -- same round-to-nearest fix, kept consistent
        # with OccupancyGrid.world_pts_to_map.
        mx = half + int(round(x / RESOLUTION))
        my = half - int(round(y / RESOLUTION))
        return int(mx), int(my)

    def convert_to_world_coordinates(self, mx, my):
        half = self.occ_map.map_size // 2
        x = (mx - half) * RESOLUTION
        y = (half - my) * RESOLUTION
        return float(x), float(y)

    def convert_to_map_coordinate_matrix(self, pts_world):
        return self.occ_map.world_pts_to_map(pts_world)


    # ── Ground check ──────────────────────────────────────────────────────────

    def robot_on_ground(self, max_tan=0.08):
        try:
            with self._step_lock:
                ax, ay, az = self.imu['accelerometer'].getValues()
            pitch = math.atan2(-ax, max(1e-9, math.sqrt(ay * ay + az * az)))
            return abs(math.tan(pitch)) < max_tan
        except Exception:
            return True

    # ── Motor control ─────────────────────────────────────────────────────────

    def stop_motor(self):
        for m in self.motors.values():
            m.setVelocity(0.0)
        self._last_cmd_left = 0.0
        self._last_cmd_right = 0.0

    def set_robot_velocity(self, left, right):
        left  = max(-MAX_VELOCITY, min(MAX_VELOCITY, left))
        right = max(-MAX_VELOCITY, min(MAX_VELOCITY, right))
        self.motors['fl'].setVelocity(left)
        self.motors['rl'].setVelocity(left)
        self.motors['fr'].setVelocity(right)
        self.motors['rr'].setVelocity(right)
        if left < right:
            self.last_turn = 'left'
        elif right < left:
            self.last_turn = 'right'

    def _set_path_velocity(self, left, right, alpha=0.55):
        left = alpha * left + (1.0 - alpha) * self._last_cmd_left
        right = alpha * right + (1.0 - alpha) * self._last_cmd_right
        if abs(left) < 0.12:
            left = 0.0
        if abs(right) < 0.12:
            right = 0.0
        self.set_robot_velocity(left, right)
        self._last_cmd_left = left
        self._last_cmd_right = right

    def velocity_to_wheel_speeds(self, v, w):
        half = self.axle_length / 2.0
        return (v - half * w) / self.wheel_radius, (v + half * w) / self.wheel_radius

    def is_turning(self):
        return abs(self.motors['fl'].getVelocity() - self.motors['fr'].getVelocity()) > 0.02

    def get_distances(self):
        with self._step_lock:
            return [s.getValue() for s in self.distance_sensors]

    # ── Timed manoeuvres ──────────────────────────────────────────────────────

    def turn_right_milisecond(self, ms=200):
        self.set_robot_velocity(MOTOR_VELOCITY_TURN, -MOTOR_VELOCITY_TURN)
        self.step(ms)
        self.stop_motor()

    def turn_left_milisecond(self, ms=200):
        self.set_robot_velocity(-MOTOR_VELOCITY_TURN, MOTOR_VELOCITY_TURN)
        self.step(ms)
        self.stop_motor()


    # ── LiDAR helpers ─────────────────────────────────────────────────────────

    def get_pointcloud_2d(self):
        if self.lidar is None:
            return np.array([])
        # self.lidar.getPointCloud() is read from both the main thread and
        # the lidar background thread (_refresh_map_lidar); Webots device
        # I/O is not documented as safe for concurrent cross-thread access,
        # and racing it against step() (which also touches the same
        # controller<->simulator IPC channel) is a plausible cause of the
        # simulation intermittently freezing at 0x. Reuse the RLock that
        # already guards step() to serialize every raw device read against
        # it and against each other.
        with self._step_lock:
            pts = self.lidar.getPointCloud()
        if not pts:
            return np.array([])
        arr = np.array([[p.x, p.y] for p in pts], dtype=np.float32)
        arr = arr[~np.isinf(arr).any(axis=1)]
        if len(arr) == 0:
            return arr
        # Points come back in the lidar's own sensor frame; shift into the
        # robot's local (odometry-origin) frame by the lidar's mount offset
        # before any caller rotates/translates them into world coordinates.
        arr[:, 0] += self.LIDAR_X_OFFSET
        arr[:, 1] += self.LIDAR_Y_OFFSET
        return arr

    def transform_points_to_world(self, pts_local):
        if len(pts_local) == 0:
            return pts_local
        theta = self.get_heading('rad')
        R = np.array([[np.cos(theta), -np.sin(theta)],
                      [np.sin(theta),  np.cos(theta)]])
        return pts_local @ R.T + np.array([self._odom_x, self._odom_y])


    def get_lidar_front_min_dist(self, angle_range_deg=30):
        pts = self.get_pointcloud_2d()
        if len(pts) == 0:
            return float('inf')
        angles = np.arctan2(pts[:, 1], pts[:, 0])
        dists  = np.linalg.norm(pts, axis=1)
        lim    = np.radians(angle_range_deg)
        front  = dists[(angles > -lim) & (angles < lim)]
        return float(np.min(front)) if len(front) > 0 else float('inf')

    def _lidar_min_dist_at_bearing(self, bearing_rad, angle_range_deg=8):
        """Nearest LiDAR hit within a narrow angular window around an
        arbitrary bearing (not just straight ahead)."""
        pts = self.get_pointcloud_2d()
        if len(pts) == 0:
            return float('inf')
        angles = np.arctan2(pts[:, 1], pts[:, 0])
        dists  = np.linalg.norm(pts, axis=1)
        lim    = np.radians(angle_range_deg)
        diff   = np.abs(((angles - bearing_rad + np.pi) % (2 * np.pi)) - np.pi)
        sector = dists[diff < lim]
        return float(np.min(sector)) if len(sector) > 0 else float('inf')


    def _get_scan_fp(self, sectors=12):
        """Return per-sector minimum LiDAR distances (robot frame).

        Used to detect encoder slip: if the scan doesn't change between steps
        while the encoders report movement, the robot is physically stuck.
        """
        try:
            with self._step_lock:
                pts = self.lidar.getPointCloud()
            if not pts:
                return None
            arr = np.array([[p.x, p.y] for p in pts], dtype=np.float32)
            valid = ~np.isinf(arr).any(axis=1) & ~np.isnan(arr).any(axis=1)
            arr = arr[valid]
            if len(arr) == 0:
                return None
            angles = np.arctan2(arr[:, 1], arr[:, 0])
            dists  = np.linalg.norm(arr, axis=1)
            edges  = np.linspace(-math.pi, math.pi, sectors + 1)
            fp = np.array([
                float(dists[(angles >= edges[i]) & (angles < edges[i + 1])].min())
                if np.any((angles >= edges[i]) & (angles < edges[i + 1])) else math.inf
                for i in range(sectors)
            ], dtype=np.float32)
            return fp
        except Exception:
            return None

    def _scan_delta(self, a, b):
        if a is None or b is None:
            return None
        finite = np.isfinite(a) & np.isfinite(b)
        if not np.any(finite):
            return None
        return float(np.mean(np.abs(a[finite] - b[finite])))

    def _refresh_map_lidar(self):
        pts_local = self.get_pointcloud_2d()
        if pts_local.shape[0] == 0:
            return
        dists = np.linalg.norm(pts_local, axis=1)
        valid = (dists >= 0.05) & np.isfinite(dists)
        pts_local = pts_local[valid]
        if pts_local.shape[0] == 0:
            return
        pts     = self.transform_points_to_world(pts_local)
        map_pos = self.get_map_position()
        self.occ_map.process_scan(map_pos, pts)

    def _ir_floating_wall_points_local(self):
        """Cover the Astra depth camera's own blind gap: it cannot report
        ANY depth closer than DEPTH_CAMERA_MIN_RANGE_M (Astra.proto's
        RangeFinder.minRange), so a floating wall the robot has approached
        closer than that simply vanishes from `_depth_obstacle_points_local`
        every frame, regardless of tuning. The front IR range sensors read
        down to near 0 m and cover exactly this gap.

        A close IR hit is only trusted as a floating-wall sighting if the
        LiDAR looking along that same bearing reports the ground-level
        space beyond it as clear — that combination (something close at IR
        height, nothing at floor height) is the signature of a panel whose
        underside the IR beam grazed but whose base never reaches the
        floor, as opposed to a normal wall or the maze's own perimeter,
        which LiDAR would already be reporting solidly.
        """
        if not self.distance_sensors or len(self.distance_sensors) < 3:
            return np.empty((0, 3), dtype=np.float32)
        try:
            with self._step_lock:
                fl_val = float(self.distance_sensors[0].getValue())
                fr_val = float(self.distance_sensors[2].getValue())
        except Exception:
            return np.empty((0, 3), dtype=np.float32)

        pts = []
        for dist, (mx, my, myaw) in zip((fl_val, fr_val), self.IR_FRONT_MOUNTS):
            if not np.isfinite(dist) or dist <= 0.01 or dist >= DEPTH_CAMERA_MIN_RANGE_M:
                continue
            px = mx + dist * math.cos(myaw)
            py = my + dist * math.sin(myaw)
            beam_len = math.hypot(px, py)
            bearing = math.atan2(py, px)
            ground_clear = self._lidar_min_dist_at_bearing(bearing)
            if ground_clear <= beam_len + 0.05:
                continue  # LiDAR sees something solid there too — not floating
            pts.append((px, py, self.IR_RANGE_HEIGHT_M))
        if not pts:
            return np.empty((0, 3), dtype=np.float32)
        return np.array(pts, dtype=np.float32)

    def _refresh_map_depth(self, depth_stride=2, max_depth=3.5):
        """Height-band marking with confirmed, frustum-gated persistence.

        depth_stride=2 (not 3): a floating wall that is narrow in the
        camera's HORIZONTAL field of view — e.g. one oriented so the robot
        sees mostly its edge rather than its full face — can project onto
        only a handful of image columns even at moderate range. A stride-3
        sample grid can step over that entire narrow column run on every
        single frame (always landing between it, never on it) for as long
        as the relative geometry stays similar, so it collects effectively
        zero votes no matter how many frames go by. Denser sampling makes
        that systematic miss far less likely; this pipeline is now fully
        vectorized (no more per-pixel Python loop), so the extra samples
        cost little.

        `_depth_obstacle_points_local` already restricts every point it
        returns to the robot-blocking height band, so every point seen here
        is a collision hazard by construction (Nav2 ObstacleLayer / STVL
        style — see the module-level docs). Each frame only casts one vote
        per map cell that band covers; a cell is not drawn or acted on
        until it has FLOATING_WALL_CONFIRM_VOTES independent votes, and
        from that point on it is frozen in position — this function never
        re-fits a line through it, never redraws it, and never moves it.
        It can only leave the map via the much stricter frustum-gated
        `_frustum_clear_floating` pass below, never by simple per-frame
        reclassification.
        """
        if self.camera_depth is None:
            return
        if not self._camera_height_calibrated:
            self._calibrate_camera_height()

        pts_local = self._depth_obstacle_points_local(pixel_stride=depth_stride,
                                                       max_depth=max_depth)
        # Supplement with IR-sensor sightings for whatever the depth camera's
        # own minimum-range blind gap missed (see _ir_floating_wall_points_local).
        # This only ADDS candidate points that flow into the exact same
        # vote/confirm/freeze/bridge pipeline below — nothing about how an
        # already-confirmed cell is drawn or classified changes.
        ir_pts = self._ir_floating_wall_points_local()
        if ir_pts.shape[0] > 0:
            pts_local = np.concatenate([pts_local, ir_pts], axis=0) if pts_local.shape[0] > 0 else ir_pts

        heading = self.get_heading('rad')

        if pts_local.shape[0] > 0:
            R = np.array([[np.cos(heading), -np.sin(heading)],
                          [np.sin(heading),  np.cos(heading)]])
            pts_world = pts_local[:, :2] @ R.T + np.array([self._odom_x, self._odom_y])
            map_pts = self.convert_to_map_coordinate_matrix(pts_world)

            grid = self.occ_map.grid_map
            h, w = grid.shape

            veto_r = FLOATING_WALL_NEAR_LIDAR_VETO_CELLS
            # Track the closest forward distance each candidate cell was
            # observed at this frame -- a wall seen nearly edge-on (the
            # "vertical" case) is very often visible to the depth camera
            # only during a brief close-range window, sometimes just a
            # frame or two, which may never reach FLOATING_WALL_CONFIRM_VOTES
            # before the robot moves past it or the wall leaves the FOV.
            # Close-range depth measurements also carry much less angular
            # error than far ones (the same 1-pixel angular uncertainty
            # translates to far less real-world position error up close),
            # so trusting fewer independent close-range votes is not a
            # noise-tolerance regression -- see FLOATING_WALL_CONFIRM_VOTES_CLOSE.
            cell_min_forward = {}
            candidate_cells = set()
            for (mx, my), forward in zip(map_pts, pts_local[:, 0]):
                mx_i, my_i = int(mx), int(my)
                if not (0 <= mx_i < w and 0 <= my_i < h):
                    continue
                # Never overlay a floating reading on ground truth that
                # lidar or colour detection has already solidly established.
                if grid[my_i, mx_i] in (OBSTACLE, GREEN_CARPET, CLOSED):
                    continue
                # Nor within a small radius of an already-confirmed LiDAR
                # wall: the depth camera's own position error over distance
                # can place a real, LiDAR-visible wall's computed position a
                # cell or two off from where LiDAR fixed it -- close enough
                # that it is clearly the same physical wall, not a separate
                # floating one, and should defer entirely to LiDAR's fix
                # rather than getting voted in nearby under the wrong label.
                y0, y1 = max(0, my_i - veto_r), min(h, my_i + veto_r + 1)
                x0, x1 = max(0, mx_i - veto_r), min(w, mx_i + veto_r + 1)
                if np.any(grid[y0:y1, x0:x1] == OBSTACLE):
                    continue
                cell = (mx_i, my_i)
                candidate_cells.add(cell)
                prev = cell_min_forward.get(cell)
                if prev is None or forward < prev:
                    cell_min_forward[cell] = float(forward)

            newly_confirmed = set()
            for cell in candidate_cells:
                if cell in self._floating_confirmed:
                    continue
                votes = self._floating_votes.get(cell, 0) + 1
                self._floating_votes[cell] = min(votes, FLOATING_WALL_VOTE_CAP)
                best_range = min(cell_min_forward[cell],
                                  self._floating_best_range.get(cell, float('inf')))
                self._floating_best_range[cell] = best_range
                required = (FLOATING_WALL_CONFIRM_VOTES_CLOSE
                            if best_range <= FLOATING_WALL_CLOSE_RANGE_M
                            else FLOATING_WALL_CONFIRM_VOTES)
                if self._floating_votes[cell] >= required:
                    newly_confirmed.add(cell)

            for cell in newly_confirmed:
                self._floating_confirmed.add(cell)
                self._floating_group[cell] = cell
                self._floating_group_members[cell] = {cell}
                self._floating_merge_touching(cell)
                self._floating_merge_colinear_gap(cell)
                self._floating_bridge_gap(cell)
                self._floating_solidify_group(self._floating_find(cell))

        self._frustum_clear_floating(heading)

        # A cell can end up here that LiDAR has since independently
        # confirmed as a real, grounded OBSTACLE for that SAME cell, or for
        # one immediately next to it (map.py's rebuild_grid lets LiDAR's
        # own strong evidence win for an exact-cell match; the small-radius
        # check here catches the same physical wall being confirmed one
        # cell over, from the camera's own position error over distance —
        # see FLOATING_WALL_NEAR_LIDAR_VETO_CELLS). Once that happens it is
        # an ordinary wall, not a floating one — drop it from floating-wall
        # bookkeeping entirely so it renders as a normal wall, not red.
        grid = self.occ_map.grid_map
        gh, gw = grid.shape
        veto_r = FLOATING_WALL_NEAR_LIDAR_VETO_CELLS
        lidar_owned = set()
        for c in self._floating_confirmed:
            cx, cy = c
            y0, y1 = max(0, cy - veto_r), min(gh, cy + veto_r + 1)
            x0, x1 = max(0, cx - veto_r), min(gw, cx + veto_r + 1)
            if np.any(grid[y0:y1, x0:x1] == OBSTACLE):
                lidar_owned.add(c)
        if lidar_owned:
            for cell in lidar_owned:
                self._floating_confirmed.discard(cell)
                self._floating_votes.pop(cell, None)
                self._floating_best_range.pop(cell, None)
                self._floating_clear_votes.pop(cell, None)
                root = self._floating_find(cell)
                members = self._floating_group_members.get(root)
                if members is not None:
                    members.discard(cell)
                self._floating_group.pop(cell, None)

        if not self._floating_confirmed:
            self.occ_map.floating_points = []
            self.occ_map.build_cost_map()
            return

        for mx, my in self._floating_confirmed:
            if grid[my, mx] not in (GREEN_CARPET, CLOSED):
                grid[my, mx] = DEPTH_OBSTACLE
        self.occ_map._depth_obstacle_cells = set(self._floating_confirmed)
        self.occ_map.floating_points = [(mx, my, 'blocking') for mx, my in self._floating_confirmed]
        self.occ_map.build_cost_map()

    def _frustum_clear_floating(self, heading):
        """Frustum-gated clearing (STVL: marking and clearing are different
        decisions with different evidence requirements — clearing must be
        much harder to trigger). A confirmed cell only loses ground when
        the camera's OWN current line of sight demonstrably passes clean
        through it to something farther away, inside the camera's valid
        range and FOV — never from LiDAR (which is blind at this height
        band by definition) and never for a cell inside or near the depth
        camera's 0.6 m blind zone, where the camera simply has no evidence
        either way. Even then, a single contradicting frame only increments
        a per-cell counter; removal requires DEPTH_CLEAR_CONFIRMATIONS
        distinct contradicting frames, and any frame that does NOT
        contradict a cell resets its counter back to zero."""
        if not self._floating_confirmed or self.camera_depth is None:
            return
        frame = self._get_depth_frame()
        if frame is None:
            return
        depth_arr, fx, fy, cx, cy, w, h = frame
        cos_h, sin_h = math.cos(heading), math.sin(heading)
        fov_half = math.atan2(w / 2.0, fx)

        to_remove = set()
        for cell in list(self._floating_confirmed):
            wx, wy = self.convert_to_world_coordinates(*cell)
            dx, dy = wx - self._odom_x, wy - self._odom_y
            fwd = dx * cos_h + dy * sin_h
            lat = -dx * sin_h + dy * cos_h
            if not (DEPTH_CLEAR_MIN_RANGE_M < fwd < 3.5):
                continue  # outside the valid, non-blind range: no evidence
            bearing = math.atan2(lat, fwd)
            if abs(bearing) > fov_half * 0.9:
                continue  # outside FOV (with a small border margin)
            u = int(round(cx + math.tan(bearing) * fx))
            if not (0 <= u < w):
                continue
            col = depth_arr[:, u]
            valid_rows = np.isfinite(col) & (col > 0.05) & (col < max(3.5, fwd + 0.5))
            if not np.any(valid_rows):
                continue
            nearest = float(np.min(col[valid_rows]))
            if nearest > fwd + 0.10:
                # Line of sight demonstrably passes through/beyond where
                # this cell should be — contradicting evidence.
                votes = self._floating_clear_votes.get(cell, 0) + 1
                self._floating_clear_votes[cell] = votes
                if votes >= DEPTH_CLEAR_CONFIRMATIONS:
                    to_remove.add(cell)
            else:
                self._floating_clear_votes.pop(cell, None)

        if not to_remove:
            return
        grid = self.occ_map.grid_map
        for cell in to_remove:
            self._floating_clear_votes.pop(cell, None)
            self._floating_confirmed.discard(cell)
            root = self._floating_find(cell)
            members = self._floating_group_members.get(root)
            if members is not None:
                members.discard(cell)
            self._floating_group.pop(cell, None)
            self._floating_votes.pop(cell, None)
            self._floating_best_range.pop(cell, None)
            mx, my = cell
            if grid[my, mx] == DEPTH_OBSTACLE:
                grid[my, mx] = FREESPACE
        self.occ_map._depth_obstacle_cells.difference_update(to_remove)

    def _floating_find(self, cell):
        """Union-find root lookup with path compression."""
        parent = self._floating_group.get(cell, cell)
        if parent == cell:
            return cell
        root = self._floating_find(parent)
        self._floating_group[cell] = root
        return root

    def _floating_union(self, a, b):
        ra, rb = self._floating_find(a), self._floating_find(b)
        if ra == rb:
            return ra
        members = self._floating_group_members.pop(ra, {ra}) | self._floating_group_members.pop(rb, {rb})
        self._floating_group[rb] = ra
        self._floating_group_members[ra] = members
        return ra

    def _floating_merge_touching(self, cell):
        """Union a newly confirmed cell with any already-confirmed cell it
        directly touches (8-connected), so gap-bridging/solidify treat one
        physical wall as one connected group instead of many disjoint dots."""
        cx, cy = cell
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                if dx == 0 and dy == 0:
                    continue
                neighbor = (cx + dx, cy + dy)
                if neighbor in self._floating_confirmed:
                    self._floating_union(cell, neighbor)

    def _floating_merge_colinear_gap(self, cell):
        """Close a coverage gap WITHIN one physical floating-wall panel:
        its two visible ends got confirmed, but sampling/occlusion/a brief
        viewing window never confirmed the cells between them, leaving a
        hole the planner would happily route the robot through. Searches
        only along the same row and the same column (never a general
        radius) for another confirmed cell belonging to a different group
        — restricting to the two cardinal directions means this can only
        ever merge two points that are candidates for being the SAME
        straight wall run, never an unrelated object that merely happens
        to be nearby in some other direction. The search distance is
        capped at FLOATING_WALL_GAP_CLOSE_MAX_CELLS (the robot's own
        physical width), which is what makes closing it always safe: a
        gap narrower than the robot itself could never have been a real,
        driveable passage regardless of what's on either side of it."""
        cx, cy = cell
        grid = self.occ_map.grid_map
        h, w = grid.shape
        root = self._floating_find(cell)
        limit = FLOATING_WALL_GAP_CLOSE_MAX_CELLS

        best_x, best_dx = None, None
        for nx in range(max(0, cx - limit), min(w, cx + limit + 1)):
            if nx == cx:
                continue
            target = (nx, cy)
            if target in self._floating_confirmed and self._floating_find(target) != root:
                d = abs(nx - cx)
                if best_dx is None or d < best_dx:
                    best_dx, best_x = d, nx
        if best_x is not None:
            for x in range(min(cx, best_x) + 1, max(cx, best_x)):
                c2 = (x, cy)
                if c2 not in self._floating_confirmed:
                    self._floating_confirmed.add(c2)
                    self._floating_group[c2] = c2
                    self._floating_group_members[c2] = {c2}
                    self._floating_union(c2, cell)
            self._floating_union(cell, (best_x, cy))
            root = self._floating_find(cell)

        best_y, best_dy = None, None
        for ny in range(max(0, cy - limit), min(h, cy + limit + 1)):
            if ny == cy:
                continue
            target = (cx, ny)
            if target in self._floating_confirmed and self._floating_find(target) != root:
                d = abs(ny - cy)
                if best_dy is None or d < best_dy:
                    best_dy, best_y = d, ny
        if best_y is not None:
            for y in range(min(cy, best_y) + 1, max(cy, best_y)):
                c2 = (cx, y)
                if c2 not in self._floating_confirmed:
                    self._floating_confirmed.add(c2)
                    self._floating_group[c2] = c2
                    self._floating_group_members[c2] = {c2}
                    self._floating_union(c2, cell)
            self._floating_union(cell, (cx, best_y))

    def _floating_bridge_gap(self, cell):
        """Close the space between a newly confirmed cell and whatever
        structure sits within FLOATING_WALL_BRIDGE_RADIUS_CELLS of it — a
        lidar wall/CLOSED cell, or another confirmed floating-wall cell not
        already directly touching it. This covers two distinct real gaps at
        once: (1) a genuine seam where this floating wall meets the next
        wall or floating wall, and (2) a hole *inside* one physical wall
        where depth-camera sampling simply hasn't confirmed every cell yet
        (stride/occlusion) even though flanking cells of the same object
        are already confirmed. Both cases render as a small strip of
        unknown/free space the planner would otherwise slip through, so
        both get bridged the same way: fill the straight-line gap and merge
        into one connected wall."""
        grid = self.occ_map.grid_map
        h, w = grid.shape
        cx, cy = cell
        r = FLOATING_WALL_BRIDGE_RADIUS_CELLS
        best, best_d2, best_is_floating = None, None, False
        for dy in range(-r, r + 1):
            for dx in range(-r, r + 1):
                d2 = dx * dx + dy * dy
                if d2 == 0 or d2 > r * r:
                    continue
                nx, ny = cx + dx, cy + dy
                if not (0 <= nx < w and 0 <= ny < h):
                    continue
                target = (nx, ny)
                is_floating = target in self._floating_confirmed
                if is_floating and self._floating_find(target) == self._floating_find(cell):
                    continue  # already the same connected wall
                is_hard_wall = grid[ny, nx] in (OBSTACLE, CLOSED)
                if not (is_floating or is_hard_wall):
                    continue
                if best_d2 is None or d2 < best_d2:
                    best_d2, best, best_is_floating = d2, target, is_floating
        if best is None:
            return
        bridge_mask = np.zeros((h, w), dtype=np.uint8)
        cv2.line(bridge_mask, (cx, cy), best, 1, thickness=1)
        bys, bxs = np.where(bridge_mask > 0)
        bridge_cells = set(zip(bxs.tolist(), bys.tolist())) - {cell, best}
        for bc in bridge_cells:
            if bc not in self._floating_confirmed:
                self._floating_confirmed.add(bc)
                self._floating_group[bc] = bc
                self._floating_group_members[bc] = {bc}
                self._floating_union(bc, cell)
        if best_is_floating:
            self._floating_union(cell, best)

    def _floating_solidify_group(self, root):
        """Close any remaining interior gap in an already-established wall
        group by filling straight between its own confirmed members —
        independently per row and per column, so an L-shaped group (two
        real walls meeting at a corner) gets each arm filled correctly
        instead of one diagonal line cutting across the corner.

        This is why a floating wall that is only ever sampled sparsely
        (a handful of cells scattered along its true run, e.g. because the
        robot only ever saw it from close range where the depth camera's
        stride sampling missed most of the interior) still ends up as one
        solid, gap-free span: as soon as ANY two cells of the same physical
        wall are confirmed, the entire straight run between them is filled
        in immediately, without needing every individual cell in between to
        independently reach FLOATING_WALL_CONFIRM_VOTES on its own."""
        members = self._floating_group_members.get(root)
        if not members or len(members) < 2:
            return
        grid = self.occ_map.grid_map
        h, w = grid.shape
        by_row, by_col = {}, {}
        for (x, y) in members:
            by_row.setdefault(y, []).append(x)
            by_col.setdefault(x, []).append(y)

        added = set()
        for y, xs in by_row.items():
            if len(xs) < 2:
                continue
            x0, x1 = min(xs), max(xs)
            if x1 - x0 > FLOATING_WALL_MAX_SOLIDIFY_SPAN_CELLS:
                continue
            for x in range(x0, x1 + 1):
                if 0 <= x < w and (x, y) not in members:
                    added.add((x, y))
        for x, ys in by_col.items():
            if len(ys) < 2:
                continue
            y0, y1 = min(ys), max(ys)
            if y1 - y0 > FLOATING_WALL_MAX_SOLIDIFY_SPAN_CELLS:
                continue
            for y in range(y0, y1 + 1):
                if 0 <= y < h and (x, y) not in members:
                    added.add((x, y))

        for cell in added:
            self._floating_confirmed.add(cell)
            self._floating_group[cell] = root
            members.add(cell)

    def _reset_transient_depth_obstacles(self):
        cells = getattr(self.occ_map, '_depth_obstacle_cells', set())
        if cells:
            for mx, my in list(cells):
                if (0 <= mx < self.occ_map.grid_map.shape[1] and
                        0 <= my < self.occ_map.grid_map.shape[0] and
                        self.occ_map.grid_map[my, mx] == DEPTH_OBSTACLE):
                    self.occ_map.grid_map[my, mx] = FREESPACE
            cells.clear()
        self._floating_votes = {}
        self._floating_best_range = {}
        self._floating_confirmed = set()
        self._floating_clear_votes = {}
        self._floating_group = {}
        self._floating_group_members = {}
        self.occ_map.floating_points = []
        self.occ_map.build_cost_map()

    def _signal_priority(self, sig):
        if sig == 'green_carpet':
            return 2
        if isinstance(sig, tuple) and len(sig) >= 2 and sig[0] == 'column':
            return 1
        return 0

    def _signal_key(self, sig):
        if isinstance(sig, tuple) and len(sig) >= 2:
            return (sig[0], sig[1])
        return sig

    def _signal_cooldown(self, sig):
        if sig == 'green_carpet':
            return CAMERA_GREEN_SIGNAL_COOLDOWN
        if isinstance(sig, tuple) and len(sig) >= 2 and sig[0] == 'column':
            return CAMERA_COLUMN_SIGNAL_COOLDOWN
        return 0.0

    def _signal_recently_active(self, sig, now=None):
        now = time.time() if now is None else now
        key = self._signal_key(sig)
        cooldown = self._signal_cooldown(sig)
        last_emit = self._last_detection_emit.get(key, -float('inf'))
        last_handled = self._last_detection_handled.get(key, -float('inf'))
        return (now - max(last_emit, last_handled)) < cooldown

    def _mark_detection_handled(self, sig):
        if sig is not None:
            self._last_detection_handled[self._signal_key(sig)] = time.time()

    def _push_detection_signal(self, sig):
        if sig is None:
            return
        now = time.time()
        if self._signal_recently_active(sig, now=now):
            return
        q = self.camera_detection_queue
        if sig == 'green_carpet':
            q = deque([s for s in q if not (isinstance(s, tuple) and s[0] == 'column')],
                      maxlen=q.maxlen)
            self.camera_detection_queue = q
        if sig not in q:
            q.append(sig)
            self._last_detection_emit[self._signal_key(sig)] = now
        self.camera_detection_signal = max(q, key=self._signal_priority) if q else None

    def _pop_detection_signal(self):
        q = self.camera_detection_queue
        if q:
            best_idx = max(range(len(q)), key=lambda i: self._signal_priority(q[i]))
            sig = q[best_idx]
            del q[best_idx]
            self.camera_detection_signal = max(q, key=self._signal_priority) if q else None
            self._mark_detection_handled(sig)
            return sig
        sig = self.camera_detection_signal
        self.camera_detection_signal = None
        self._mark_detection_handled(sig)
        return sig

    def _pop_scan_detection_signal(self):
        q = self.camera_detection_queue
        if q:
            for predicate in (
                    lambda s: isinstance(s, tuple) and len(s) >= 2 and s[0] == 'column',
                    lambda s: s == 'green_carpet'):
                for i, sig in enumerate(q):
                    if predicate(sig):
                        del q[i]
                        self.camera_detection_signal = max(q, key=self._signal_priority) if q else None
                        self._mark_detection_handled(sig)
                        return sig
        sig = self.camera_detection_signal
        self.camera_detection_signal = None
        self._mark_detection_handled(sig)
        return sig

    # ── Camera helpers ────────────────────────────────────────────────────────

    def get_hsv_image(self, scale=1.0):
        if self.camera_rgb is None:
            return None
        try:
            # Read from both the main thread and the camera background
            # thread -- serialize against step() and other device reads via
            # the same lock (see get_pointcloud_2d for why).
            with self._step_lock:
                raw = self.camera_rgb.getImage()
                w = self.camera_rgb.getWidth()
                h = self.camera_rgb.getHeight()
        except Exception:
            return None
        if not raw:
            return None
        try:
            img = np.frombuffer(raw, np.uint8).reshape((h, w, 4))
            if scale != 1.0:
                img = cv2.resize(img, (max(1, int(w * scale)),
                                       max(1, int(h * scale))),
                                 interpolation=cv2.INTER_AREA)
            bgr = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
            return cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
        except Exception:
            return None

    def get_bottom_half_hsv(self):
        hsv = self.get_hsv_image()
        if hsv is None:
            return None
        return hsv[hsv.shape[0] // 2:, :]

    # ── Colour detection ──────────────────────────────────────────────────────

    def detect_green(self, hsv=None, min_pixels=50):
        # Accept a pre-fetched full-frame HSV image so the camera loop can share a
        # single getImage()/cvtColor with detect_column instead of fetching twice.
        if hsv is None:
            hsv = self.get_bottom_half_hsv()
        else:
            hsv = hsv[hsv.shape[0] // 2:, :]
        if hsv is None:
            return None
        return cv2.countNonZero(utils.extract_color_mask(hsv, 'green')) > min_pixels

    def detect_column(self, hsv=None, min_pixels=20):
        if hsv is None:
            hsv = self.get_hsv_image()
        if hsv is None:
            return None
        if cv2.countNonZero(utils.extract_color_mask(hsv, 'yellow')) >= min_pixels:
            return 'yellow'
        if cv2.countNonZero(utils.extract_color_mask(hsv, 'blue')) >= min_pixels:
            return 'blue'
        return None


    def found_all_2_columns(self):
        return self.start_point is not None and self.end_point is not None


    def get_column_center_ratio(self, color):
        hsv = self.get_hsv_image()
        if hsv is None:
            return 0.0
        center = hsv[:, hsv.shape[1] // 3: 2 * hsv.shape[1] // 3]
        mask   = utils.extract_color_mask(center, color)
        ppr    = np.count_nonzero(mask, axis=1)
        return np.count_nonzero(ppr) / mask.shape[0]

    def get_column_center_pixels(self, color):
        hsv = self.get_hsv_image()
        if hsv is None:
            return 0
        center = hsv[:, hsv.shape[1] // 3: 2 * hsv.shape[1] // 3]
        return np.count_nonzero(np.count_nonzero(utils.extract_color_mask(center, color), axis=1))

    def _get_depth_frame(self):
        """Fetch the raw depth image plus its pinhole intrinsics once, shared
        by both point extraction and frustum-gated clearing so a frame is
        only ever read from the device a single time per refresh."""
        if self.camera_depth is None:
            return None
        try:
            with self._step_lock:
                depth_data = self.camera_depth.getRangeImage()
                if not depth_data:
                    return None
                w   = self.camera_depth.getWidth()
                h   = self.camera_depth.getHeight()
                fov = self.camera_depth.getFov()
        except Exception:
            return None
        fx = w / (2.0 * np.tan(fov / 2.0))
        fy = fx
        cx = w / 2.0
        cy = h / 2.0
        depth_arr = np.array(depth_data, dtype=np.float32).reshape(h, w)
        return depth_arr, fx, fy, cx, cy, w, h

    def _calibrate_camera_height(self):
        """One-shot empirical refinement of camera_height_m (Nav2/STVL-style
        height-band marking is only as good as this one number). The proto
        chain gives a starting estimate, but the true optical-center height
        above the floor also depends on manufacturing tolerance and mount
        slop that the proto alone can't capture. Point-blank floor pixels in
        the near, central bottom strip of the depth image should compute to
        height == 0 by construction (camera_height_m - d*y_n == 0), so
        solving that equation for camera_height_m over real floor pixels
        calibrates out whatever residual bias remains. Runs once, only
        while the patch looks like flat open floor (tight variance); tries
        again on later frames otherwise, and simply keeps the proto-derived
        default forever if a clean floor patch never appears.
        """
        frame = self._get_depth_frame()
        if frame is None:
            return
        depth_arr, fx, fy, cx, cy, w, h = frame
        row_lo = int(h * 0.85)
        col_lo, col_hi = int(w * 0.35), int(w * 0.65)
        patch = depth_arr[row_lo:h, col_lo:col_hi]
        valid = np.isfinite(patch) & (patch > 0.05) & (patch < 2.0)
        if np.count_nonzero(valid) < 20:
            return
        vv, uu = np.where(valid)
        d = patch[vv, uu]
        y_n = ((vv + row_lo).astype(np.float32) - cy) / fy
        implied_height_if_zero_bias = d * y_n
        if float(np.std(implied_height_if_zero_bias)) > 0.03:
            return  # not flat/consistent enough to trust as open floor
        estimate = float(np.median(implied_height_if_zero_bias))
        if not (0.10 < estimate < 0.30):
            return  # implausible relative to the proto-derived starting point
        self.camera_height_m = estimate
        self._camera_height_calibrated = True

    def _depth_obstacle_points_local(self, pixel_stride=3, max_depth=3.5):
        """Height-band point extraction (Nav2 ObstacleLayer / STVL style):
        every 3D point whose height falls inside
        [GROUND_EPSILON_M, ROBOT_CLEARANCE_HEIGHT_M] is a collision hazard,
        full stop — no distinction between "floating" and "grounded" is
        computed or needed. A point below the band is the floor; a point
        above it is something the robot fits under (an overhang the LiDAR
        being blind to is irrelevant, since it doesn't block anything); a
        point already in the band blocks the robot whether the surface it
        belongs to touches the ground somewhere else or not. LiDAR-visible
        walls fall in this band too and get marked redundantly, which is
        harmless (map.py already lets a LiDAR OBSTACLE win over
        DEPTH_OBSTACLE), and dropping the previous per-pixel downward
        "ground support" column scan removes both an O(candidates ×
        image-height) pure-Python loop and the exact failure mode it
        caused: a genuinely floating wall whose underside numerically
        looked "connected" to nearby floor within noise tolerance was
        silently discarded as a false positive.
        """
        frame = self._get_depth_frame()
        if frame is None:
            return np.empty((0, 3), dtype=np.float32)
        depth_arr, fx, fy, cx, cy, w, h = frame
        valid = np.isfinite(depth_arr) & (depth_arr > 0.05) & (depth_arr < max_depth)
        if not np.any(valid):
            return np.empty((0, 3), dtype=np.float32)

        rows = np.arange(0, h, pixel_stride)
        cols = np.arange(0, w, pixel_stride)
        vv, uu = np.meshgrid(rows, cols, indexing='ij')
        d  = depth_arr[vv, uu]
        ok = valid[vv, uu]
        if not np.any(ok):
            return np.empty((0, 3), dtype=np.float32)

        # Reject silhouette "flying pixel" artifacts: a genuine surface
        # point has depth close to its immediate neighbours; a pixel that
        # straddles an object's edge against a much farther background does
        # not. Vectorized against the full-resolution neighbours, not the
        # strided sample, so it catches the true local discontinuity. The
        # threshold scales with the pixel's own depth (floored at
        # DEPTH_EDGE_DISCONTINUITY_M) so a wall seen nearly edge-on — whose
        # real, non-artifact depth gradient across neighbouring pixels
        # grows with distance from ordinary perspective — isn't mistaken
        # for a silhouette jump and dropped wholesale.
        # d itself, and either neighbour, can be inf/NaN here (a "no return"
        # pixel — e.g. open space beyond max range) since `ok` from `valid`
        # hasn't been applied yet. inf-inf / NaN arithmetic below is
        # harmless (any comparison against NaN is already False, so it
        # can't wrongly ADMIT a bad point) but does emit spurious
        # RuntimeWarnings, and — more importantly — treating a genuinely
        # MISSING neighbour reading as automatic proof of a "silhouette
        # jump" was wrong: a wall seen nearly edge-on (a "vertical" wall
        # the robot mostly passes alongside) often has open space or
        # out-of-range background immediately next to its face in the
        # image, which is exactly a missing/invalid neighbour, not
        # evidence of a flying-pixel artifact — so it was being rejected
        # for the wrong reason. A missing neighbour now contributes no
        # evidence either way; only an actually-measured, too-different
        # neighbour rejects a point.
        d_right = depth_arr[vv, np.minimum(uu + 1, w - 1)]
        d_down  = depth_arr[np.minimum(vv + 1, h - 1), uu]
        finite_d = np.isfinite(d)
        right_finite = np.isfinite(d_right)
        down_finite  = np.isfinite(d_down)
        with np.errstate(invalid='ignore'):
            diff_right = np.abs(d - d_right)
            diff_down  = np.abs(d - d_down)
            edge_thresh = np.maximum(DEPTH_EDGE_DISCONTINUITY_M,
                                      DEPTH_EDGE_RELATIVE_FRACTION * np.where(finite_d, d, 0.0))
        right_ok = ~right_finite | (diff_right < edge_thresh)
        down_ok  = ~down_finite  | (diff_down  < edge_thresh)
        edge_ok = finite_d & right_ok & down_ok
        ok &= edge_ok
        if not np.any(ok):
            return np.empty((0, 3), dtype=np.float32)

        uu = uu[ok].astype(np.int32)
        vv = vv[ok].astype(np.int32)
        d  = d[ok].astype(np.float32)
        x_n = (uu.astype(np.float32) - cx) / fx
        y_n = (vv.astype(np.float32) - cy) / fy

        forward = d + self.X_offset
        lateral = -d * x_n + self.Y_offset
        height  = self.camera_height_m - d * y_n

        band = (height >= GROUND_EPSILON_M) & (height <= ROBOT_CLEARANCE_HEIGHT_M)
        if not np.any(band):
            return np.empty((0, 3), dtype=np.float32)
        return np.stack(
            [forward[band], lateral[band], height[band]], axis=1
        ).astype(np.float32)

    def _column_lidar_position_local(self, color, base_angle_window_deg=4.0,
                                      max_angle_window_deg=24.0):
        """Robot-local (forward, lateral, height) of `color`'s pillar.

        The colour camera contributes exactly one thing: bearing (which
        direction the pillar is in, from its mask centroid) -- a camera
        measures angle far more reliably than range. The range comes
        straight from the LiDAR's own nearest hit along that bearing, so
        the resulting point sits exactly where the LiDAR detected a solid
        obstacle there. No depth-camera projection, near-field percentile
        estimate, or robot-position stand-in is involved -- this is the
        only source of a pillar's placed position.

        The angular search window starts at roughly the blob's own apparent
        width (plus a fixed margin) and, if that finds nothing, doubles up
        to `max_angle_window_deg` before giving up. A distant pillar has a
        tiny apparent width in the image, so its own-width window collapses
        to almost nothing -- too narrow to reliably catch it against a
        LiDAR whose angular ray spacing is coarse relative to a thin
        cylinder at range, or against a slight bearing mismatch between the
        camera's and LiDAR's own mount points. Widening only kicks in when
        the tight window comes up empty, so close-range precision (picking
        the pillar's own return, not a wall behind it) is unaffected.
        """
        hsv_img = self.get_hsv_image()
        if hsv_img is None:
            return None
        mask = utils.extract_color_mask(hsv_img, color)
        if mask is None or not np.any(mask):
            return None
        if np.count_nonzero(mask) < self.COLUMN_MIN_LIDAR_BLOB_PX:
            return None
        ys, xs = np.where(mask != 0)
        u = float(np.median(xs))
        u_lo, u_hi = float(np.min(xs)), float(np.max(xs))
        x_n = (u - self.cx) / self.fx
        bearing = -math.atan(x_n)
        half_width = abs(math.atan((u_hi - self.cx) / self.fx) -
                          math.atan((u_lo - self.cx) / self.fx)) / 2.0
        window_deg = base_angle_window_deg + math.degrees(half_width)
        lidar_r = float('inf')
        while not np.isfinite(lidar_r):
            lidar_r = self._lidar_min_dist_at_bearing(bearing, angle_range_deg=window_deg)
            if np.isfinite(lidar_r) or window_deg >= max_angle_window_deg:
                break
            window_deg = min(window_deg * 2.0, max_angle_window_deg)
        if not np.isfinite(lidar_r):
            return None
        forward = lidar_r * math.cos(bearing)
        lateral = lidar_r * math.sin(bearing)
        return np.array([forward, lateral, self.camera_height_m], dtype=np.float32)

    def _column_bearing_local(self, color, assumed_forward_m=3.2, max_area_ratio=0.06):
        """Rough direction-only fallback for when the RGB camera can see a
        column's colour but the LiDAR has no return along that bearing yet
        (see _column_lidar_position_local) -- range is unavailable, so no
        estimate/marking ever happened for a genuinely distant sighting and
        exploration had no bias toward it at all.

        Uses only the colour mask's horizontal centroid (bearing), combined
        with an assumed forward distance, in the same local [forward,
        lateral, height] convention as the LiDAR-grounded estimate, so it
        can flow through the exact same projection/estimation path -- just
        good enough to steer frontier exploration toward the right general
        direction until the robot gets close enough for a real fix.

        Gated to only fire when the colour blob is still small (a large
        blob means it's actually close and depth failing is a transient
        glitch, not an out-of-range sighting -- using the far-away
        assumption there would plant a wildly wrong point).
        """
        hsv = self.get_hsv_image()
        if hsv is None:
            return None
        mask = utils.extract_color_mask(hsv, color)
        if mask is None or not np.any(mask):
            return None
        h, w = mask.shape[:2]
        area_ratio = float(np.count_nonzero(mask)) / float(mask.size)
        if area_ratio <= 0.0 or area_ratio > max_area_ratio:
            return None
        M = cv2.moments(mask)
        if M['m00'] <= 0:
            return None
        u = M['m10'] / M['m00']
        # Reuse the intrinsics cached at init (same camera, same resolution)
        # instead of re-querying the device under _step_lock.
        fx = self.fx * (w / float(self.cam_width))
        cx = w / 2.0
        x_n = (u - cx) / fx
        forward = assumed_forward_m + self.X_offset
        lateral = -assumed_forward_m * x_n + self.Y_offset
        height = self.camera_height_m
        return np.array([forward, lateral, height], dtype=np.float32)

    def _column_is_front_facing(self, color):
        return self.get_column_center_ratio(color) >= 0.20

    # ── Obstacle detection ────────────────────────────────────────────────────

    def obstacle_in_front(self):
        # Exact port from github.com/duchieuvn/autonomous2's obstacle_in_front.
        # 1. Check original distance sensors (V-shape)
        ds_distances = self.get_distances()
        ds_obstacle = min(ds_distances[0], ds_distances[2]) < 0.05
        if ds_obstacle:
            print(f"[Virtual Bumper] Distance sensors detected obstacle at {min(ds_distances[0], ds_distances[2]):.2f}m")

        # 2. Check Lidar Virtual Bumper (The "Semicircle")
        lidar_dist = self.get_lidar_front_min_dist(angle_range_deg=35)
        lidar_obstacle = lidar_dist < 0.10
        if lidar_obstacle:
            print(f"[Virtual Bumper] Lidar detected obstacle at {lidar_dist:.2f}m")

        return ds_obstacle or lidar_obstacle

    def _unknown_neighborhood_score(self, mx, my, radius=5):
        grid = self.grid_map
        h, w = grid.shape
        mx, my = int(mx), int(my)
        if not (0 <= mx < w and 0 <= my < h):
            return 0.0
        x0, x1 = max(0, mx - radius), min(w, mx + radius + 1)
        y0, y1 = max(0, my - radius), min(h, my + radius + 1)
        area = max(1, (x1 - x0) * (y1 - y0))
        return float(np.count_nonzero(grid[y0:y1, x0:x1] == UNKNOWN)) / float(area)

    # ── DWA planner ───────────────────────────────────────────────────────────

    def there_is_obstacle(self, map_target):
        return self.occ_map.cell_blocked(map_target)

    def _current_wheel_speeds(self):
        """Last commanded wheel angular velocities (rad/s) -- the true
        starting point for a dynamic window, since the motors (and the
        smoothing filter in _set_path_velocity) haven't necessarily reached
        whatever velocity_to_wheel_speeds() last asked for."""
        return self._last_cmd_left, self._last_cmd_right

    def dwa_planner(self, world_target):
        best_score, best_v, best_w = -float('inf'), 0.0, 0.0
        x, y    = self._odom_x, self._odom_y
        theta   = self.get_heading('rad')
        cur_dist = np.linalg.norm(world_target - np.array([x, y]))
        dt = TIME_STEP / 1000.0

        for v in DWA_VELOCITY_SAMPLES:
            for w in DWA_ANGULAR_SAMPLES:
                cx, cy, ct = x, y, theta
                ok = True
                unknown_bonus = 0.0
                for _ in range(5):           # 5-step horizon → ~160 ms look-ahead
                    cx += v * np.cos(ct) * dt
                    cy += v * np.sin(ct) * dt
                    ct += w * dt
                    pmx, pmy = self.convert_to_map_coordinates(cx, cy)
                    if self.there_is_obstacle([pmx, pmy]):
                        ok = False
                        break
                    if (self.occ_map.cost_map is not None and
                            0 <= pmx < MAP_SIZE and 0 <= pmy < MAP_SIZE and
                            float(self.occ_map.cost_map[pmy, pmx]) > 0.92):
                        ok = False
                        break
                    if not self.found_all_2_columns():
                        unknown_bonus = max(
                            unknown_bonus,
                            self._unknown_neighborhood_score(pmx, pmy, radius=4)
                        )
                if not ok:
                    continue
                pd   = np.linalg.norm(world_target - np.array([cx, cy]))
                pa   = np.arctan2(world_target[1] - cy, world_target[0] - cx)
                herr = utils.angle_wrap(pa, ct)
                cost_penalty = 0.0
                if self.occ_map.cost_map is not None:
                    pmx, pmy = self.convert_to_map_coordinates(cx, cy)
                    if (0 <= pmx < MAP_SIZE and 0 <= pmy < MAP_SIZE):
                        cost_penalty = float(self.occ_map.cost_map[pmy, pmx])
                score = (DWA_HEADING_WEIGHT  * np.cos(herr) +
                         DWA_DISTANCE_WEIGHT * (1.0 - pd / max(0.1, cur_dist * 2)) +
                         DWA_SPEED_WEIGHT    * v / max(DWA_VELOCITY_SAMPLES) -
                         DWA_COST_MAP_WEIGHT * cost_penalty -
                         0.18 * abs(w) +
                         DWA_UNKNOWN_WEIGHT  * unknown_bonus)
                if score > best_score:
                    best_score, best_v, best_w = score, v, w

        return best_v, best_w

    # ── Waypoint following ────────────────────────────────────────────────────

    def advance_to_waypoint(self, map_target):
        """Returns (reached, is_stuck)."""
        if self.get_map_distance(map_target) < PATH_FOLLOWING_TARGET_REACH_DISTANCE:
            self.follow_target_last_position = None
            self.follow_target_stuck_count   = 0
            self._phys_stuck_count           = 0
            self._scan_stuck_count           = 0
            self._last_scan_fp               = None
            self._waypoint_motion_commanded  = False
            self.stop_motor()
            return True, False

        cur = self.get_position()
        scan_fp = self._get_scan_fp()
        if self.follow_target_last_position is not None:
            moved = np.linalg.norm(cur - self.follow_target_last_position)
            if moved < self.follow_target_position_threshold:
                self.follow_target_stuck_count += 1
                if self.follow_target_stuck_count >= self.follow_target_stuck_threshold:
                    self.stop_motor()
                    self.follow_target_stuck_count   = 0
                    self.follow_target_last_position = None
                    self._phys_stuck_count           = 0
                    self._scan_stuck_count           = 0
                    self._waypoint_motion_commanded  = False
                    return False, True
            else:
                self.follow_target_stuck_count = 0
        self.follow_target_last_position = cur

        if self._waypoint_motion_commanded:
            scan_delta = self._scan_delta(scan_fp, self._last_scan_fp)
            if scan_delta is not None and scan_delta < 0.006:
                self._scan_stuck_count += 1
                if self._scan_stuck_count >= 10:
                    self.stop_motor()
                    self._scan_stuck_count           = 0
                    self.follow_target_last_position = None
                    self.follow_target_stuck_count   = 0
                    self._last_scan_fp               = scan_fp
                    self._waypoint_motion_commanded  = False
                    return False, True
            else:
                self._scan_stuck_count = 0
        self._last_scan_fp = scan_fp

        # Obstacle stop: cut motors the instant a wall is detected so the
        # wheel encoders stop counting.  This is the core drift fix — no wheel
        # spin means no encoder accumulation and no phantom map movement.
        if self.obstacle_in_front():
            self.stop_motor()                  # encoders freeze immediately
            self._waypoint_motion_commanded = False
            self._phys_stuck_count += 1
            if self._phys_stuck_count >= 4:    # 4 × 32 ms = 128 ms of solid contact
                self._phys_stuck_count           = 0
                self.follow_target_last_position = None
                self.follow_target_stuck_count   = 0
                self._scan_stuck_count           = 0
                return False, True
            return False, False                # motors off, wait one more step
        self._phys_stuck_count = 0

        wx, wy = self.convert_to_world_coordinates(map_target[0], map_target[1])

        # If heading is far off, rotate in place before engaging DWA.
        # DWA with angular samples up to ±4 rad/s would otherwise spin the robot
        # at near-max speed, which looks erratic and clips walls with the chassis edge.
        target_angle = math.atan2(wy - self._odom_y, wx - self._odom_x)
        herr = utils.angle_wrap(target_angle, self.get_heading('rad'))
        if abs(herr) > math.radians(25):
            self.follow_target_stuck_count = 0   # rotation is intentional, not stuck
            w_rot = 2.0 if herr > 0 else -2.0
            lsp, rsp = self.velocity_to_wheel_speeds(0.0, w_rot)
            self._set_path_velocity(lsp, rsp, alpha=0.7)
            self._waypoint_motion_commanded = True
            return False, False

        v, w   = self.dwa_planner(np.array([wx, wy]))
        if v <= 0.001 and abs(w) <= 0.001:
            self.stop_motor()
            self._dwa_no_motion_count += 1
            if self._dwa_no_motion_count >= 3:
                self._dwa_no_motion_count = 0
                self.follow_target_last_position = None
                self._waypoint_motion_commanded = False
                return False, True
            return False, False
        self._dwa_no_motion_count = 0
        if v < 0.08 and abs(w) < 0.25:
            v = 0.08
        ls, rs = self.velocity_to_wheel_speeds(v, w)
        self._set_path_velocity(ls, rs)
        self._waypoint_motion_commanded = abs(ls) > 0.2 or abs(rs) > 0.2
        return False, False

    # ── Recovery / direction adaptation ──────────────────────────────────────


    def unstick(self, turn_range=(400, 600)):
        self.stop_motor()
        for _ in range(40):
            if self.step(self.time_step) == -1:
                return
        ms = random.randint(turn_range[0], turn_range[1])
        if random.random() < 0.5:
            self.turn_left_milisecond(ms)
        else:
            self.turn_right_milisecond(ms)
        self.set_robot_velocity(MOTOR_VELOCITY_FORWARD * 0.8, MOTOR_VELOCITY_FORWARD * 0.8)
        self.step(250)
        self.stop_motor()
        for _ in range(40):
            if self.step(self.time_step) == -1:
                return

    def clear_obstacle(self, turn_range=(400, 600)):
        self.stop_motor()
        for _ in range(30):
            if self.step(self.time_step) == -1:
                return
        ms = random.randint(turn_range[0], turn_range[1])
        if random.random() < 0.5:
            self.turn_left_milisecond(ms)
        else:
            self.turn_right_milisecond(ms)
        for _ in range(30):
            if self.step(self.time_step) == -1:
                return

    def _correct_odom_after_collision(self):
        """Snap _odom_x/_odom_y to the nearest FREESPACE cell when encoder drift after
        a collision places the robot inside an obstacle cell on the map.
        Also refreshes the live visualisation with the corrected position."""
        mx, my = self.get_map_position()
        H, W = self.occ_map.grid_map.shape
        mx = int(np.clip(mx, 1, W - 2))
        my = int(np.clip(my, 1, H - 2))
        cell = self.occ_map.grid_map[my, mx]
        if cell not in (OBSTACLE, CLOSED):
            # Already in traversable space — just refresh the viz
            with self.occ_map.vis_lock:
                self.occ_map.robot_position = (mx, my)
            self.occ_map.refresh_viz()
            return
        column_cells = set()
        for p in (self.start_point, self.end_point):
            if p is not None:
                column_cells.add((int(round(float(p[0]))), int(round(float(p[1])))))
        for p in (self.occ_map.column_points or []):
            if isinstance(p, (list, tuple)) and len(p) >= 2:
                column_cells.add((int(round(float(p[0]))), int(round(float(p[1])))))

        def near_column_cell(x, y, radius=3):
            return any((x - cx) * (x - cx) + (y - cy) * (y - cy) <= radius * radius
                       for cx, cy in column_cells)

        half = self.occ_map.map_size // 2
        for r in range(1, 30):
            for dy in range(-r, r + 1):
                for dx in range(-r, r + 1):
                    if abs(dx) != r and abs(dy) != r:
                        continue  # walk only the ring perimeter
                    nx, ny = mx + dx, my + dy
                    if 0 <= nx < W and 0 <= ny < H:
                        if (self.occ_map.grid_map[ny, nx] == FREESPACE and
                                not near_column_cell(nx, ny)):
                            self._odom_x = (nx - half) * RESOLUTION
                            self._odom_y = (half - ny) * RESOLUTION
                            print(f'[Recovery] Odometry snapped map ({mx},{my})'
                                  f' → ({nx},{ny})  world ({self._odom_x:.3f},{self._odom_y:.3f})')
                            with self.occ_map.vis_lock:
                                self.occ_map.robot_position = (nx, ny)
                            self.occ_map.refresh_viz()
                            return
        print('[Recovery] Could not find a free cell to snap to — position unchanged')

    # ── Alignment helpers ─────────────────────────────────────────────────────

    def center_column_in_view(self, color):
        Kp, Kd    = ALIGN_COLUMN_KP, ALIGN_COLUMN_KD
        last_err  = 0
        count     = 0
        while self.step(self.time_step) != -1:
            count += 1
            if count > 100:
                self.stop_motor()
                break
            hsv = self.get_hsv_image()
            if hsv is None:
                self.stop_motor()
                break
            _, w, _ = hsv.shape
            mask = utils.extract_color_mask(hsv, color)
            M    = cv2.moments(mask)
            if M['m00'] > 0:
                cx  = int(M['m10'] / M['m00'])
                err = cx - (w // 2)
                if abs(err) < ALIGN_COLUMN_ERROR_THRESHOLD:
                    self.stop_motor()
                    break
                self.set_robot_velocity(Kp * err + Kd * (err - last_err),
                                        -(Kp * err + Kd * (err - last_err)))
                last_err = err
            else:
                self.stop_motor()
                break


    def align_to_path(self, map_target, angle_threshold_deg=15):
        if map_target is None:
            return True
        try:
            tx_w, ty_w = self.convert_to_world_coordinates(map_target[0], map_target[1])
        except Exception:
            return True
        heading      = self.get_heading('rad')
        target_angle = np.arctan2(ty_w - self._odom_y, tx_w - self._odom_x)
        angle = target_angle - heading
        while angle > np.pi:  angle -= 2 * np.pi
        while angle < -np.pi: angle += 2 * np.pi
        if abs(np.degrees(angle)) < angle_threshold_deg:
            return True
        w_cmd = 1.0 if angle > 0 else -1.0
        lsp, rsp = self.velocity_to_wheel_speeds(0.0, w_cmd)
        self.set_robot_velocity(lsp, rsp)
        init_h     = self.get_heading('rad')
        rad_needed = abs(angle)
        while self.step(self.time_step) != -1:
            turned = self.get_heading('rad') - init_h
            while turned >  np.pi: turned -= 2 * np.pi
            while turned < -np.pi: turned += 2 * np.pi
            if abs(turned) >= rad_needed - 0.03:
                break
        self.stop_motor()
        return True

    # ── Column marking / estimation ───────────────────────────────────────────


    def _advance_column_state(self, color, weight=0.0, close=False):
        """Single place a colour's state (_column_state) transitions:
        UNSEEN -> ROUGH -> PROVISIONAL -> CONFIRMED. CONFIRMED is frozen --
        once reached, no further transitions happen (replaces the old
        _column_confirmed_close set). ROUGH marks that at least one
        low/any-weight sighting has landed with nothing committed yet.
        PROVISIONAL follows _column_commit_ready() (real fix + enough
        accumulated confidence, or already close) -- this is also the point
        start_point/end_point get set, by the caller in
        _set_committed_column. CONFIRMED is asserted by the caller once a
        close, front-facing observation has actually been committed.
        """
        state = self._column_state.get(color, 'UNSEEN')
        if state == 'CONFIRMED':
            return state
        if state == 'UNSEEN' and weight > 0.0:
            state = 'ROUGH'
        if self._column_commit_ready(color):
            state = 'PROVISIONAL'
        if close:
            state = 'CONFIRMED'
        self._column_state[color] = state
        return state

    def _carve_reachability(self, color, mp, radius=3):
        """One-time free-space carve near a just-CONFIRMED pillar position,
        guaranteeing `_final_pillar_access_cell`'s outward ring-search has a
        reachable anchor to find even if the corridor around the pillar has
        only been thinly ray-traced so far.

        `mp` is now the pillar's real surface point (a LiDAR hit -- see
        _column_lidar_position_local), so carving free space AT `mp` itself
        would falsely erase part of the pillar's own footprint from the
        map. Instead carve a small patch pulled back a few cells from `mp`
        towards the robot's current position -- the side the robot was
        actually standing/looking from when it took this sighting, which is
        real free space by construction (the LiDAR ray that produced `mp`
        travelled unobstructed through it).
        """
        robot = np.array(self.get_map_position(), dtype=float)
        pillar = np.array(mp, dtype=float)
        to_robot = robot - pillar
        dist = float(np.linalg.norm(to_robot))
        if dist < 1e-6:
            anchor = mp
        else:
            pull_back = min(radius + 2, dist)
            anchor = tuple((pillar + to_robot / dist * pull_back).astype(int))
        self._carve_freespace_near(anchor, radius=radius)
        self._record_column_marker(color, mp)

    def _set_committed_column(self, color, mp, close=False):
        """Returns True iff a commit actually happened. Callers
        (_commit_column_from_estimate) must propagate this -- they used to
        return True unconditionally right after calling this, which
        silently lied about blue having committed whenever the commit-order
        gate below blocked it (yellow not done yet). That made
        _handle_scan_signal/_confirm_pillar_close believe blue was already
        confirmed and stop pursuing it, even though it was still plainly
        visible in the camera -- blue looked "skipped" even on a good
        sighting.

        `mp` is a LiDAR-grounded surface point (see
        _column_lidar_position_local / _estimate_column_pos) -- it is
        committed as-is, never snapped, refined, or substituted with the
        robot's own position.
        """
        # Commit-order priority: the final path runs blue -> yellow, so
        # yellow gets committed first. If blue turns up before yellow is
        # committed, its sighting still updates the running estimate and
        # steers exploration toward it (see _estimate_column_pos /
        # _column_biased_target) -- it just isn't allowed to actually commit
        # (stamp start_point) until yellow is locked in as end_point.
        if color == 'blue' and self.end_point is None:
            return False
        # Once a pillar has been close-confirmed (CONFIRMED), its committed
        # position is frozen. Without this guard, every later
        # close+front-facing sighting (e.g. during exploration, before the
        # final approach) would keep re-committing, which is what used to
        # make start_point/end_point flicker.
        if close and self._column_state.get(color) == 'CONFIRMED':
            return False
        # A close confirmation gets the wider reachability carve (pulled
        # back off the pillar's own footprint); a provisional commit just
        # records the marker, no grid mutation.
        if close:
            self._carve_reachability(color, mp)
        else:
            self._record_column_marker(color, mp)
        if color == 'blue':
            self.start_point = mp
        elif color == 'yellow':
            self.end_point = mp
        # Any commit (provisional or close) satisfies _column_is_committed(),
        # so release the exploration focus now -- otherwise _estimate_column_pos
        # keeps refreshing it every sighting and exploration never moves on to
        # look for the other pillar.
        if self._column_focus_color == color:
            self._column_focus_color = None
            self._column_focus_target = None
        self._advance_column_state(color, close=close)
        return True

    def _column_is_committed(self, color):
        return self.start_point is not None if color == 'blue' else self.end_point is not None

    def _column_commit_ready(self, color):
        est = self.blue_estimated_pos if color == 'blue' else self.yellow_estimated_pos
        if est is None:
            return False
        if self._column_close_front_visible(color):
            return True
        weight = self.blue_estimate_weight if color == 'blue' else self.yellow_estimate_weight
        if weight < self.COLUMN_WEIGHT_NORMAL * COLUMN_COMMIT_MIN_ESTIMATES:
            return False
        return self.get_map_distance(est) <= COLUMN_COMMIT_MAX_MAP_DISTANCE

    def _column_close_front_visible(self, color):
        if not self._column_is_front_facing(color):
            return False
        column_local = self._column_lidar_position_local(color)
        if column_local is None:
            return False
        r_cm = math.hypot(float(column_local[0]), float(column_local[1])) * 100.0
        return r_cm < COLOR_DETECTION_DEPTH_THRESHOLD

    def _commit_column_from_estimate(self, color, force=False):
        if not force and not self._column_commit_ready(color):
            return False
        est = self.blue_estimated_pos if color == 'blue' else self.yellow_estimated_pos
        if est is None:
            return False
        mp = (int(round(float(est[0]))), int(round(float(est[1]))))
        close = force or self._column_close_front_visible(color)
        if self._column_is_front_facing(color):
            # This frame's own precise LiDAR-grounded fix, if one can still
            # be taken right now -- use it instead of the blended running
            # average (`est`), which can still be dragged short of the
            # pillar's true position by earlier lower-confidence sightings
            # (e.g. a long straight-on approach that accumulated several
            # normal-weight updates before the robot ever got close) even
            # once a clean fix exists. Not gated on `close` (< 80 cm): a
            # straight-on sighting already has an accurate LiDAR range and
            # reliable bearing well before that range, so restricting this
            # to only the close case left a pillar seen straight-on from
            # far away committed short of its true position. Mirrors the
            # same precise-vs-blended choice _estimate_column_pos makes for
            # its own inline commit; this covers the other callers that
            # commit through this function instead (_confirm_pillar_close,
            # _handle_scan_signal).
            column_local = self._column_lidar_position_local(color)
            if column_local is not None:
                heading = self.get_heading('rad')
                R = np.array([[np.cos(heading), -np.sin(heading)],
                              [np.sin(heading),  np.cos(heading)]])
                wp = column_local[:2] @ R.T + np.array([self._odom_x, self._odom_y])
                fresh_mp = self.convert_to_map_coordinates(float(wp[0]), float(wp[1]))
                mp = (int(round(float(fresh_mp[0]))), int(round(float(fresh_mp[1]))))
        return self._set_committed_column(color, mp, close=close)

    def update_column_estimation(self, color, position, weight=1.0):
        """Weight-blend a new observation into the running position estimate.

        A confidence total is tracked per colour (blue_estimate_weight /
        yellow_estimate_weight) and used, along with the new observation's
        own weight, to compute a weighted average -- so a high-confidence
        close depth fix barely moves once several have landed, while a
        low-weight bearing-only guess can still nudge the estimate before
        any real fix exists, but can't meaningfully perturb one once it
        does. Total weight is capped so the filter stays responsive to a
        genuinely different later reading rather than becoming immovable.
        """
        position = np.array(position, dtype=float)
        if color == 'blue':
            old_pos, old_weight = self.blue_estimated_pos, self.blue_estimate_weight
        else:
            old_pos, old_weight = self.yellow_estimated_pos, self.yellow_estimate_weight
        if old_pos is None or old_weight <= 0.0:
            new_pos, new_weight = position, weight
        else:
            new_weight = old_weight + weight
            new_pos = (np.array(old_pos, dtype=float) * old_weight +
                       position * weight) / new_weight
        new_weight = min(self.COLUMN_WEIGHT_CAP, new_weight)
        if color == 'blue':
            self.blue_estimated_pos    = new_pos
            self.blue_estimate_weight  = new_weight
        elif color == 'yellow':
            self.yellow_estimated_pos   = new_pos
            self.yellow_estimate_weight = new_weight

    def _column_line_of_sight_clear(self, mp, end_margin=3):
        robot_mp = self.get_map_position()
        ray = utils.ray_cells((int(robot_mp[0]), int(robot_mp[1])),
                              (int(mp[0]), int(mp[1])))
        if len(ray) <= end_margin + 2:
            return True
        grid = self.grid_map
        h, w = grid.shape
        for rx, ry in ray[1:-end_margin]:
            if 0 <= rx < w and 0 <= ry < h:
                # DEPTH_OBSTACLE (floating walls) are at a different height than
                # columns; if the RGB camera can see the column colour the visual
                # LOS is clear, so only solid LiDAR walls block the ray here.
                if grid[ry, rx] in (OBSTACLE, CLOSED, GREEN_CARPET):
                    return False
        return True


    def _column_approach_target(self, pos, min_radius=6, max_radius=18):
        if pos is None:
            return None
        grid = self.grid_map
        h, w = grid.shape
        cx, cy = int(round(float(pos[0]))), int(round(float(pos[1])))
        robot = np.array(self.get_map_position(), dtype=float)
        best, best_score = None, float('inf')
        for radius in range(min_radius, max_radius + 1, 2):
            for dx in range(-radius, radius + 1):
                for dy in range(-radius, radius + 1):
                    if abs(dx) != radius and abs(dy) != radius:
                        continue
                    x, y = cx + dx, cy + dy
                    if not (0 <= x < w and 0 <= y < h):
                        continue
                    if grid[y, x] not in (FREESPACE, UNKNOWN):
                        continue
                    if self.occ_map.cell_blocked((x, y)):
                        continue
                    if not self._column_line_of_sight_clear((x, y), end_margin=1):
                        continue
                    unknown_score = self._unknown_neighborhood_score(x, y, radius=5)
                    if unknown_score <= 0.0:
                        continue
                    score = float(np.linalg.norm(np.array([x, y], dtype=float) - robot))
                    score += 0.25 * abs(math.hypot(dx, dy) - 10.0)
                    score -= 18.0 * unknown_score
                    if score < best_score:
                        best_score, best = score, (x, y)
            if best is not None:
                return best
        return None

    def _record_column_marker(self, color, marker_pos):
        """Record/refresh the visualization + semantic marker for `color`'s
        pillar at `marker_pos` in occ_map.column_points. Pure bookkeeping --
        never touches the grid/log-odds, since `marker_pos` is the pillar's
        real (LiDAR-grounded) surface point and must keep reading as a
        genuine obstacle, not be overwritten as free space.
        """
        vmx, vmy = int(round(float(marker_pos[0]))), int(round(float(marker_pos[1])))
        color_rgb = (0, 255, 255) if color == 'blue' else (255, 255, 0)
        pts = list(self.occ_map.column_points) if self.occ_map.column_points else []
        pts = [(px, py, c) for px, py, c in pts if c != color_rgb]
        pts.append((vmx, vmy, color_rgb))
        self.occ_map.column_points = pts

    def _carve_freespace_near(self, mp, radius=1):
        """Force a small circular patch to read as FREESPACE (grid +
        log-odds), guaranteeing a reachable anchor exists near a just-placed
        pillar marker even if the corridor around it has only been thinly
        ray-traced so far. Only ever called with an anchor point known/
        expected to be real free space (see _carve_reachability) -- never
        with the pillar's own surface point, which must stay a genuine
        obstacle on the map.
        """
        mx = int(round(float(mp[0]))); my = int(round(float(mp[1])))
        h, w = self.grid_map.shape
        for dy in range(-radius, radius + 1):
            for dx in range(-radius, radius + 1):
                if dx * dx + dy * dy > radius * radius:
                    continue
                nx, ny = mx + dx, my + dy
                if 0 <= nx < w and 0 <= ny < h:
                    try:
                        self.occ_map.log_odds[ny, nx] = min(self.occ_map.log_odds[ny, nx], -2.5)
                    except Exception:
                        pass
                    self.grid_map[ny, nx] = FREESPACE

    def _estimate_column_pos(self, color):
        """Produce one fresh sighting of `color`'s pillar and fold it into
        the running per-colour estimate.

        The colour camera supplies bearing only (which direction, and which
        pillar); the LiDAR supplies the entire range (see
        _column_lidar_position_local) -- so the resulting map position is
        exactly where the LiDAR detected the pillar's surface, never a
        camera-depth projection. Because a LiDAR hit is by construction an
        unobstructed, in-bounds return, there is nothing left to gate here:
        no line-of-sight check (the ray that produced it was already clear),
        no floating-wall veto, no independent-sensor corroboration -- those
        all existed to catch a bad *camera* projection, and none of them
        apply to a position that already came straight from the LiDAR.
        """
        column_local = self._column_lidar_position_local(color)
        rough = column_local is None
        if rough:
            # Nothing in the LiDAR's range/window along that bearing yet --
            # pillar beyond LiDAR range, or a gap swallows the beam. Fall
            # back to a bearing-only rough estimate so exploration still
            # steers the right way; assigned a low weight below (see
            # update_column_estimation) so it can nudge the running
            # estimate before a real LiDAR fix exists, but never commits
            # (see below) and can't meaningfully drag a real fix around.
            column_local = self._column_bearing_local(color)
            if column_local is None:
                return False
        heading = self.get_heading('rad')
        R = np.array([[np.cos(heading), -np.sin(heading)],
                      [np.sin(heading),  np.cos(heading)]])
        wp = column_local[:2] @ R.T + np.array([self._odom_x, self._odom_y])
        mp = self.convert_to_map_coordinates(float(wp[0]), float(wp[1]))
        mp_cell = (int(round(float(mp[0]))), int(round(float(mp[1]))))
        front_facing = self._column_is_front_facing(color)
        is_close = (not rough) and front_facing and self._column_close_front_visible(color)
        if rough:
            weight = self.COLUMN_WEIGHT_ROUGH
        elif is_close:
            weight = self.COLUMN_WEIGHT_CLOSE
        else:
            weight = self.COLUMN_WEIGHT_NORMAL
        self.update_column_estimation(color, mp, weight=weight)
        self._advance_column_state(color, weight=weight)
        est = self.blue_estimated_pos if color == 'blue' else self.yellow_estimated_pos
        if est is not None:
            est_cell = (int(round(float(est[0]))), int(round(float(est[1]))))
            # Any non-rough, front-facing sighting is this frame's own
            # precise LiDAR-grounded fix -- use it as-is rather than the
            # running weighted-average estimate, which can still be dragged
            # off by earlier low-confidence rough/normal sightings (taken
            # from farther away or bad angles) even after a clean fix
            # finally lands. This used to be gated on is_close (< 80 cm)
            # as well, but a straight-on sighting still well outside that
            # range already has an accurate LiDAR range + reliable bearing
            # (see _column_is_front_facing) -- gating the fresh fix on
            # close range alone left a pillar seen straight-on from far
            # away committed short of its true position, dragged there by
            # earlier lower-confidence sightings, since it might never
            # register as "close" before commit fires. Only fall back to
            # the blended estimate when there is no fresh, reliable fix to
            # anchor on this frame at all (rough, or badly off-angle).
            has_fresh_fix = (not rough) and front_facing
            commit_cell = mp_cell if has_fresh_fix else est_cell
            # A rough bearing-only guess is not accurate enough to mark on
            # the map -- only steer exploration toward it. Once the robot
            # is close enough for a real LiDAR fix, this same path marks it.
            if not rough:
                self._record_column_marker(color, commit_cell)
            if self._column_commit_ready(color):
                self._set_committed_column(color, commit_cell, close=is_close)
            # Keep steering toward the pillar (refreshing the approach target
            # with each fresher estimate) only until it has been committed
            # at all (start_point/end_point set) -- close-range confirmation
            # for both pillars happens later in _confirm_pillar_close(), once
            # exploration is done. Continuing to refocus here after a
            # provisional commit would fixate exploration on one pillar
            # indefinitely and starve the search for the other one.
            if not self._column_is_committed(color):
                self._column_focus_color = color
                self._column_focus_target = self._column_approach_target(est)
        return True

    # ── Dynamic path validation and replanning helpers ─────────────────────────

    def _path_blocked(self, path, lookahead=PATH_BLOCKED_LOOKAHEAD_CELLS, cost_thresh=0.82):
        """Return True if any of the next `lookahead` path cells are blocked or
        have high cost near walls. Also consider persistent depth-only obstacles
        that may not yet be reflected in the grid_map."""
        if path is None or len(path) == 0:
            return False
        lm = min(len(path), lookahead)
        grid = self.occ_map.grid_map
        depth_cells = getattr(self.occ_map, '_depth_obstacle_cells', set())
        for px, py in path[:lm]:
            x, y = int(px), int(py)
            if not (0 <= x < grid.shape[1] and 0 <= y < grid.shape[0]):
                continue
            if grid[y, x] in (OBSTACLE, DEPTH_OBSTACLE):
                return True
            if (x, y) in depth_cells:
                return True
            if self.occ_map.cost_map is not None:
                try:
                    if self.occ_map.cost_map[y, x] > cost_thresh:
                        return True
                except Exception:
                    pass
        return False

    def _path_blocked_from_pose(self, path, lookahead=PATH_BLOCKED_LOOKAHEAD_CELLS, cost_thresh=0.82):
        """Check path blocking starting from the closest waypoint to the robot."""
        if path is None or len(path) == 0:
            return False
        try:
            rx, ry = self.get_map_position()
            rpos = np.array([float(rx), float(ry)], dtype=np.float32)
            pts = np.array(path, dtype=np.float32)
            if pts.ndim != 2 or pts.shape[0] == 0:
                return False
            d2 = np.sum((pts - rpos) ** 2, axis=1)
            start_idx = int(np.argmin(d2))
            segment = pts[start_idx:start_idx + max(1, lookahead)]
            return self._path_blocked(segment, lookahead=len(segment), cost_thresh=cost_thresh)
        except Exception:
            return False

    def _attempt_replan(self, goal):
        """Try several replanning strategies to handle dynamic obstacles and
        narrow passages. Returns a new path or None."""
        start = self.get_map_position()
        # 1) Try default planner
        try:
            new = self.occ_map.astar_path(start, goal)
            if new:
                return new
        except Exception:
            pass
        # 2) Try planner with relaxed cost_map but keep real clearance.
        try:
            if self.occ_map.cost_map is not None:
                scaled = (self.occ_map.cost_map * 0.2).astype(np.float32)
            else:
                scaled = None
            new = self.occ_map.astar_path(start, goal, inflation_levels=[3, 2], cost_map_override=scaled)
            if new:
                return new
        except Exception:
            pass
        # 3) Try ignoring cost map, still with enough inflation to avoid tiny gaps.
        try:
            new = self.occ_map.astar_path(start, goal, inflation_levels=[2], cost_map_override=None)
            if new:
                return new
        except Exception:
            pass
        return None

    def _path_usable_from_pose(self, path, min_len=3, lookahead=PATH_USABLE_LOOKAHEAD_CELLS, cost_thresh=0.82):
        if not path or len(path) < min_len:
            return False
        grid = self.occ_map.grid_map
        for px, py in path:
            x, y = int(px), int(py)
            if 0 <= x < grid.shape[1] and 0 <= y < grid.shape[0] and grid[y, x] == GREEN_CARPET:
                return False
        return not self._path_blocked_from_pose(path, lookahead=lookahead, cost_thresh=cost_thresh)

    def _final_path_clear(self, path, endpoint_margin=2, blocked_values=None,
                           use_live_depth_cells=True):
        if not path or len(path) < 2:
            return False
        grid = self.occ_map.grid_map
        h, w = grid.shape
        if blocked_values is None:
            blocked_values = (OBSTACLE, DEPTH_OBSTACLE, CLOSED, GREEN_CARPET)
        depth_cells = (getattr(self.occ_map, '_depth_obstacle_cells', set())
                       if use_live_depth_cells else ())
        for i in range(1, len(path)):
            x0, y0 = int(path[i - 1][0]), int(path[i - 1][1])
            x1, y1 = int(path[i][0]), int(path[i][1])
            cells = utils.ray_cells((x0, y0), (x1, y1))
            for j, (x, y) in enumerate(cells):
                if i == 1 and j < endpoint_margin:
                    continue
                if i == len(path) - 1 and j >= len(cells) - endpoint_margin:
                    continue
                if not (0 <= x < w and 0 <= y < h):
                    return False
                if grid[y, x] in blocked_values or (x, y) in depth_cells:
                    return False
        return True

    def _final_path_usable(self, path, min_len=3):
        if not path or len(path) < min_len:
            return False
        return self._final_path_clear(path)

    def _final_pillar_access_cell(self, pillar, prefer=None, min_radius=4, max_radius=18):
        if pillar is None:
            return None
        grid = self.occ_map.grid_map
        h, w = grid.shape
        px, py = int(round(float(pillar[0]))), int(round(float(pillar[1])))
        pref = np.array(prefer if prefer is not None else self.get_map_position(), dtype=float)
        # A close-confirmed pillar's own cell gets a guaranteed-FREESPACE
        # carve around it at confirmation time (see _carve_reachability),
        # so it's usually directly usable -- try it first instead of
        # searching outward from min_radius, which would otherwise skip the
        # one cell already known good. Falls through to the ring search
        # below if the carve didn't leave this exact cell free (e.g. it was
        # re-covered by a later obstacle observation).
        if (0 <= px < w and 0 <= py < h and grid[py, px] == FREESPACE and
                not self.occ_map.cell_blocked((px, py))):
            return (px, py)
        best, best_score = None, float('inf')
        for radius in range(min_radius, max_radius + 1):
            found_at_radius = False
            for dx in range(-radius, radius + 1):
                for dy in range(-radius, radius + 1):
                    if abs(dx) != radius and abs(dy) != radius:
                        continue
                    x, y = px + dx, py + dy
                    if not (0 <= x < w and 0 <= y < h):
                        continue
                    if grid[y, x] != FREESPACE:
                        continue
                    if self.occ_map.cell_blocked((x, y)):
                        continue
                    score = float(np.linalg.norm(np.array([x, y], dtype=float) - pref))
                    if score < best_score:
                        best_score, best = score, (x, y)
                        found_at_radius = True
            if found_at_radius:
                return best
        return best

    def _reset_waypoint_follow_state(self):
        self.follow_target_last_position = None
        self.follow_target_stuck_count = 0
        self._phys_stuck_count = 0
        self._scan_stuck_count = 0
        self._last_scan_fp = None
        self._waypoint_motion_commanded = False
        self._dwa_no_motion_count = 0

    def _refresh_after_recovery(self):
        try:
            self._refresh_map_lidar()
        except Exception:
            pass
        try:
            self._refresh_map_depth()
        except Exception:
            pass
        try:
            self.occ_map.build_cost_map(force=True)
        except Exception:
            pass
        self._correct_odom_after_collision()

    def _recover_and_replan(self, goal, prefer_frontier=False, min_len=3):
        """Move out of contact, refresh the map, then return a clear path."""
        self.stop_motor()
        self._reset_waypoint_follow_state()
        self.clear_obstacle(turn_range=(450, 750))
        self._refresh_after_recovery()

        if self.obstacle_in_front():
            self.unstick(turn_range=(550, 850))
            self._refresh_after_recovery()

        planners = []
        if prefer_frontier:
            planners.append(lambda: self.occ_map.frontier_path(self.get_map_position(), goal))
        planners.append(lambda: self._attempt_replan(goal))
        planners.append(lambda: self.occ_map.astar_path(self.get_map_position(), goal, inflation_levels=[3, 2]))
        if not prefer_frontier:
            planners.append(lambda: self.occ_map.frontier_path(self.get_map_position(), goal))

        for plan in planners:
            try:
                new = plan()
            except Exception:
                new = None
            if self._path_usable_from_pose(new, min_len=min_len):
                return list(new)
        return None

    # ── Closure marking ───────────────────────────────────────────────────────


    # ── 360° scan ─────────────────────────────────────────────────────────────

    def scan_360(self):
        print('[Scan] 360° rotation...')
        # scan_360 is a pure in-place spin — the robot ends up at the same
        # (x, y) it started at. Wheel/ground contact keeps the two encoders
        # from cancelling out exactly, so over hundreds of steps of continuous
        # rotation _tick_odometry's ds accumulates real position drift, which
        # then throws off pillar projection right after the scan. Snapshot the
        # pre-scan position and restore it afterwards — heading still comes
        # straight from the compass each tick, so this only discards the
        # spurious translational drift, not real orientation changes.
        origin_x, origin_y = self._odom_x, self._odom_y
        self.set_robot_velocity(3.0, -3.0)
        steps_done  = 0
        target_steps = 8500 // self.time_step
        found_sig = None
        while steps_done < target_steps:
            if self.step(self.time_step) == -1:
                break
            steps_done += 1
            with self.detection_lock:
                sig = self._pop_scan_detection_signal()
            if sig is not None:
                if isinstance(sig, tuple) and sig[0] == 'column':
                    print(f'[Scan] Column {sig[1]} — stopping 360')
                    found_sig = sig
                    break
                elif sig == 'green_carpet':
                    print(f'[Scan] {sig} — stopping 360')
                    found_sig = sig
                    break
        self.stop_motor()
        self._odom_x, self._odom_y = origin_x, origin_y
        return found_sig

    def _suspend_column_focus(self, color, seconds=8.0):
        if color is None:
            return
        self._column_focus_blocked_until[color] = time.time() + seconds
        if self._column_focus_color == color:
            self._column_focus_color = None
            self._column_focus_target = None

    def _clear_frontier_for_column_focus(self):
        if self._column_focus_target is not None:
            self._active_frontier_goal = None

    def _handle_scan_signal(self, sig):
        if sig is None:
            return False
        if sig == 'green_carpet':
            self.mark_green_carpet_permanently(min_pixel_threshold=1500, force=True)
            return False
        if isinstance(sig, tuple) and len(sig) >= 2 and sig[0] == 'column':
            color = sig[1]
            estimated = self._estimate_column_pos(color)
            if not estimated:
                self.center_column_in_view(color)
                estimated = self._estimate_column_pos(color)
            committed = self._column_is_committed(color)
            if estimated and not committed:
                committed = self._commit_column_from_estimate(
                    color, force=self._column_close_front_visible(color))
            if committed:
                self._suspend_column_focus(color, seconds=2.0)
                return True
            if not estimated:
                self._suspend_column_focus(color, seconds=8.0)
        return False

    # ── Frontier selection ────────────────────────────────────────────────────
    # Based on select_frontier_target / select_frontier_target2 from
    # github.com/duchieuvn/autonomous2, but made rigidly exploration-first:
    # frontiers are ranked by how much UNKNOWN space they open up (info gain)
    # rather than plain proximity, and the goal is aimed at the cell most
    # surrounded by unknown so the robot commits straight into unexplored
    # territory instead of skimming the edge.

    def _frontier_info_gain(self, region, radius=25):
        """Count UNKNOWN cells near the frontier centroid.

        High count → genuinely unexplored territory.
        Low count → frontier is stale (area already mapped around it).
        Radius 25 px ≈ 0.83 m gives enough context to distinguish a new room
        from a small pocket left after partial exploration.
        """
        grid = self.grid_map
        h, w = grid.shape
        cells = np.array(region)
        cx = int(cells[:, 0].mean())
        cy = int(cells[:, 1].mean())
        x0, x1 = max(0, cx - radius), min(w, cx + radius + 1)
        y0, y1 = max(0, cy - radius), min(h, cy + radius + 1)
        return int((grid[y0:y1, x0:x1] == UNKNOWN).sum())

    def _frontier_unknown_target(self, region):
        if not region:
            return None
        rpos = np.array(self.get_map_position(), dtype=float)
        best_cell, best_score = None, -float('inf')
        for x, y in region:
            unknown_score = self._unknown_neighborhood_score(x, y, radius=7)
            if unknown_score <= 0.0:
                continue
            dist = float(np.linalg.norm(np.array([x, y], dtype=float) - rpos)) + 1.0
            trail_penalty = 0.55 if ((int(x) >> 3, int(y) >> 3) in self._robot_trail) else 1.0
            score = (unknown_score ** 1.35) * trail_penalty / math.log1p(dist / 6.0)
            if score > best_score:
                best_score = score
                best_cell = (int(x), int(y))
        if best_cell is not None:
            return best_cell
        cells = np.array(region)
        return (int(np.mean(cells[:, 0])), int(np.mean(cells[:, 1])))

    def _score_frontier(self, regions, blacklist=None):
        if not regions:
            return None
        # Prevent visited_frontiers from permanently blacklisting the whole map
        if len(self.occ_map.visited_frontiers) > 20:
            self.occ_map.visited_frontiers = self.occ_map.visited_frontiers[-20:]
        rpos    = np.array(self.get_map_position(), dtype=float)
        visited = [(int(vx), int(vy)) for vx, vy in self.occ_map.visited_frontiers]
        best_region, best_score = None, -1.0

        for region in regions:
            cells    = np.array(region, dtype=float)
            centroid = cells.mean(axis=0)
            cx, cy   = int(centroid[0]), int(centroid[1])
            d        = float(np.linalg.norm(centroid - rpos)) + 1e-3

            info_gain = self._frontier_info_gain(region)
            if info_gain < 10:
                continue  # stale — fewer than 10 unknowns in radius

            known_near = max(1, len(region))
            unknown_ratio = info_gain / float(info_gain + known_near)

            # Exploration-first: prefer frontiers that expand unknown space,
            # while avoiding long trips through already known cells.
            local_unknown = self._unknown_neighborhood_score(cx, cy, radius=10)
            score = ((info_gain ** 1.85) *
                     (0.25 + 3.5 * unknown_ratio + 2.0 * local_unknown) /
                     math.log1p(d / 5.0))

            # Frontier persistence: strongly prefer continuing into the same unknown
            # region rather than jumping across the map after each frontier.
            if self._last_frontier_goal is not None:
                glx, gly = self._last_frontier_goal
                gdist = math.sqrt((cx - glx) ** 2 + (cy - gly) ** 2)
                score *= 1.0 + 3.0 * max(0.0, 1.0 - gdist / 35.0)

            # Short-term blacklist penalty
            if blacklist and any(abs(cx - bx) < 10 and abs(cy - by) < 10
                                 for bx, by in blacklist):
                score *= 0.1

            # Already-visited frontier — reduced radius so the robot can re-approach
            # the same region from a different angle after it's been partially mapped
            if any(abs(cx - vx) < 12 and abs(cy - vy) < 12 for vx, vy in visited):
                score *= 0.15

            # Trail penalty — softened: frontier centroids are by definition in
            # explored space, so a hard penalty here biases against forward-direction
            # frontiers that naturally sit near the robot's recent path
            cxc, cyc = cx >> 3, cy >> 3
            if (cxc, cyc) in self._robot_trail:
                score *= 0.5
            elif any((cxc + dx, cyc + dy) in self._robot_trail
                     for dx in (-1, 0, 1) for dy in (-1, 0, 1)):
                score *= 0.75

            if score > best_score:
                best_score, best_region = score, region

        if best_region is None:
            return self._score_frontier_fallback(regions, blacklist)

        cells = np.array(best_region)
        target = self._frontier_unknown_target(best_region)
        if target is not None:
            return target
        return (int(np.mean(cells[:, 0]) + random.randint(-2, 2)),
                int(np.mean(cells[:, 1]) + random.randint(-2, 2)))

    def _score_frontier_fallback(self, regions, blacklist=None):
        """Fallback when all frontiers are below info_gain threshold.
        Uses the same exploration-first formula as the primary scorer so large
        stale explored regions don't beat small but novel narrow-corridor ones."""
        if not regions:
            return None
        rpos = np.array(self.get_map_position(), dtype=float)
        best_region, best_score = None, -1.0
        for region in regions:
            cells    = np.array(region, dtype=float)
            centroid = cells.mean(axis=0)
            cx, cy   = int(centroid[0]), int(centroid[1])
            d        = float(np.linalg.norm(centroid - rpos)) + 1e-3
            ig       = self._frontier_info_gain(region)
            unknown_ratio = ig / float(ig + max(1, len(region)))
            local_unknown = self._unknown_neighborhood_score(cx, cy, radius=10)
            score    = ((max(ig, 5) ** 1.65) *
                        (0.25 + 3.0 * unknown_ratio + 1.5 * local_unknown) /
                        math.log1p(d / 5.0))
            if self._last_frontier_goal is not None:
                glx, gly = self._last_frontier_goal
                gdist = math.sqrt((cx - glx) ** 2 + (cy - gly) ** 2)
                score *= 1.0 + 3.0 * max(0.0, 1.0 - gdist / 35.0)
            if score > best_score:
                best_score, best_region = score, region
        if best_region is None:
            return None
        cells = np.array(best_region)
        target = self._frontier_unknown_target(best_region)
        if target is not None:
            return target
        return (int(np.mean(cells[:, 0]) + random.randint(-2, 2)),
                int(np.mean(cells[:, 1]) + random.randint(-2, 2)))

    def _random_frontier(self, regions):
        if not regions:
            return None
        rpos  = np.array(self.get_map_position(), dtype=float)
        valid = [r for r in regions if self._frontier_info_gain(r) >= 10]
        pool  = valid if valid else regions
        weights = []
        for r in pool:
            ig = self._frontier_info_gain(r)
            c  = np.array(r, dtype=float).mean(axis=0)
            cx, cy = float(c[0]), float(c[1])
            d  = float(np.linalg.norm(c - rpos)) + 1.0
            unknown_ratio = ig / float(ig + max(1, len(r)))
            local_unknown = self._unknown_neighborhood_score(cx, cy, radius=10)
            w  = max(0.01, ((max(ig, 5) ** 1.65) *
                            (0.25 + 3.0 * unknown_ratio + 1.5 * local_unknown) /
                            math.log1p(d / 5.0)))
            if self._last_frontier_goal is not None:
                glx, gly = self._last_frontier_goal
                gdist = math.sqrt((cx - glx) ** 2 + (cy - gly) ** 2)
                w *= 1.0 + 3.0 * max(0.0, 1.0 - gdist / 35.0)
            weights.append(w)
        chosen = random.choices(pool, weights=weights, k=1)[0]
        cells  = np.array(chosen)
        target = self._frontier_unknown_target(chosen)
        if target is not None:
            return target
        return (int(np.mean(cells[:, 0]) + random.randint(-2, 2)),
                int(np.mean(cells[:, 1]) + random.randint(-2, 2)))

    def _column_biased_target(self, max_jitter=8):
        now = time.time()
        if self._column_focus_target is not None:
            color = self._column_focus_color or 'column'
            # A suspended color (see _suspend_column_focus) must not keep
            # re-latching its own stale focus_target here -- this early-
            # return path used to skip the blocked_until check entirely, so
            # a color that keeps failing to commit (e.g. blue seen before
            # yellow is committed, blocked by the commit-order gate) could
            # stay "focused" forever once set, hijacking every subsequent
            # _update_frontier() call into chasing/scanning it and starving
            # real frontier exploration completely.
            blocked = self._column_focus_blocked_until.get(color, 0.0) > now
            if not blocked:
                tx, ty = self._column_focus_target
                if 0 <= tx < self.grid_map.shape[1] and 0 <= ty < self.grid_map.shape[0]:
                    if (self.grid_map[ty, tx] in (FREESPACE, UNKNOWN) and
                            not self.occ_map.cell_blocked((tx, ty)) and
                            self._unknown_neighborhood_score(tx, ty, radius=5) > 0.0):
                        print(f'[Frontier] Focusing {color.upper()} coordinate')
                        return (int(tx), int(ty))
            self._column_focus_target = None
            self._column_focus_color = None
        candidates = []
        yellow_blocked = self._column_focus_blocked_until.get('yellow', 0.0) > now
        blue_blocked = self._column_focus_blocked_until.get('blue', 0.0) > now
        if self.yellow_estimated_pos is not None and self.end_point is None and not yellow_blocked:
            candidates.append(('yellow', self.yellow_estimated_pos))
        if self.blue_estimated_pos is not None and self.start_point is None and not blue_blocked:
            candidates.append(('blue', self.blue_estimated_pos))
        if not candidates:
            return None
        # Priority: the final path runs blue -> yellow, and yellow is
        # committed first (see _set_committed_column's commit-order gate).
        # Always prefer chasing yellow when both are candidates, so
        # exploration heads toward it directly instead of a coin flip.
        by_color = {c: p for c, p in candidates}
        if 'yellow' in by_color:
            color, pos = 'yellow', by_color['yellow']
        else:
            color, pos = 'blue', by_color['blue']
        target = self._column_approach_target(pos)
        if target is not None:
            print(f'[Frontier] Biasing toward {color.upper()} coordinate')
            self._column_focus_color = color
            self._column_focus_target = target
            self._clear_frontier_for_column_focus()
            return target
        return None

    def _column_focus_path(self):
        target = self._column_biased_target()
        if target is None:
            return None
        start = self.get_map_position()
        path = self.occ_map.frontier_path(start, target)
        if not path:
            path = self.occ_map.astar_path(start, target, inflation_levels=[2, 1, 0])
        return path


    # ── Background threads ────────────────────────────────────────────────────

    def start_camera_thread(self):
        if self.detection_thread and self.detection_thread.is_alive():
            return
        with self.detection_lock:
            self.camera_detection_signal = None
            self.camera_detection_queue.clear()
        self.camera_thread_running = True
        self.detection_thread = threading.Thread(target=self._camera_loop, daemon=True)
        self.detection_thread.start()
        print('[Detection] Camera thread started')

    def stop_camera_thread(self):
        self.camera_thread_running = False
        if self.detection_thread:
            self.detection_thread.join(timeout=1.0)

    def start_lidar_thread(self):
        if self.lidar_thread and self.lidar_thread.is_alive():
            return
        self.lidar_thread_running = True
        self.lidar_thread = threading.Thread(target=self._lidar_loop, daemon=True)
        self.lidar_thread.start()
        print('[LiDAR] Mapping thread started')

    def stop_lidar_thread(self):
        self.lidar_thread_running = False
        if self.lidar_thread:
            self.lidar_thread.join(timeout=1.0)

    # ── Real-time path planner ────────────────────────────────────────────────

    def start_realtime_planner(self, goal):
        """Launch the background path-watchdog targeting `goal` (map coords)."""
        self.stop_realtime_planner()
        with self._rt_planner_lock:
            self._rt_planner_goal = (int(goal[0]), int(goal[1]))
            self._rt_active_path  = None
            self._rt_new_path     = None
            self._rt_replan_ready = False
            self._rt_last_replan_time = 0.0
        self._rt_planner_running = True
        self._rt_planner_thread  = threading.Thread(
            target=self._realtime_planner_loop, daemon=True)
        self._rt_planner_thread.start()

    def stop_realtime_planner(self):
        """Stop the background path-watchdog."""
        self._rt_planner_running = False
        t = self._rt_planner_thread
        if t and t.is_alive():
            t.join(timeout=1.0)
        self._rt_planner_thread = None
        with self._rt_planner_lock:
            self._rt_planner_goal = None
            self._rt_active_path  = None
            self._rt_new_path     = None
            self._rt_replan_ready = False

    def update_realtime_path(self, path):
        """Tell the watchdog which path is currently being followed."""
        with self._rt_planner_lock:
            self._rt_active_path = list(path) if path else None

    def poll_realtime_planner(self):
        """Return a fresh replanned path if one is waiting, else None.
        Consuming the result clears the ready flag."""
        with self._rt_planner_lock:
            if self._rt_replan_ready and self._rt_new_path:
                path = self._rt_new_path
                self._rt_new_path     = None
                self._rt_replan_ready = False
                return path
        return None

    def _realtime_planner_loop(self):
        """Background thread: every 80 ms check the active path against the
        live map and replan immediately (same goal) when a segment is blocked."""
        while self._rt_planner_running:
            try:
                time.sleep(0.08)
                with self._rt_planner_lock:
                    goal          = self._rt_planner_goal
                    cur           = self._rt_active_path
                    already_ready = self._rt_replan_ready
                if goal is None or already_ready:
                    continue
                if cur is None:
                    with self.occ_map.vis_lock:
                        cur = self.occ_map.current_path
                path_clear = (cur is not None and len(cur) > 2 and
                              not self._path_blocked_from_pose(cur, lookahead=20))
                if path_clear:
                    continue
                now = time.time()
                if now - self._rt_last_replan_time < 0.6:
                    continue
                self._rt_last_replan_time = now
                new = self._attempt_replan(goal)
                if not self._path_usable_from_pose(new, min_len=3, lookahead=18):
                    new = self.occ_map.frontier_path(self.get_map_position(), goal)
                if self._path_usable_from_pose(new, min_len=3, lookahead=18):
                    with self._rt_planner_lock:
                        self._rt_new_path     = new
                        self._rt_replan_ready = True
                        self._rt_active_path  = new
            except Exception as e:
                print(f'[RT-Planner] {e}')
                time.sleep(0.2)

    def _camera_loop(self):
        time.sleep(1.0)
        while self.camera_thread_running:
            time.sleep(CAMERA_LOOP_INTERVAL)
            now = time.time()
            # Maze1 smoothness: while a detection signal is still pending and
            # unhandled by the main loop, skip ALL image processing this iteration.
            # The previous code fetched the frame and ran the detectors every tick
            # regardless, doing redundant heavy work that slowed the simulation.
            with self.detection_lock:
                pending = (self.camera_detection_signal is not None or
                           len(self.camera_detection_queue) > 0)
            if pending:
                continue
            # Run detections outside the lock so each is independent and
            # the lock is not held during slow image-processing calls.  This ensures
            # green carpet and column detection are never skipped because the other
            # detector is slow or triggered a skip condition.
            # Fetch the camera frame once per iteration and reuse it for both
            # detectors. Previously detect_green() and detect_column() each called
            # getImage() + a full BGRA→BGR→HSV conversion, doubling the per-frame
            # image work in this 10 Hz loop and slowing the simulation.
            scale = float(CAMERA_DETECTION_SCALE)
            hsv = self.get_hsv_image(scale=scale)
            area_scale = max(0.01, scale * scale)
            green = self.detect_green(hsv, min_pixels=max(8, int(50 * area_scale)))
            color = None
            if not self.found_all_2_columns():
                color = self.detect_column(hsv, min_pixels=max(8, int(20 * area_scale)))
            with self.detection_lock:
                candidates = {
                    'green_carpet': bool(green),
                    ('column', color) if color else ('column', None): bool(color),
                }
                for key, active in candidates.items():
                    if key == ('column', None):
                        continue
                    if active:
                        self._camera_seen_counts[key] = self._camera_seen_counts.get(key, 0) + 1
                    else:
                        self._camera_seen_counts[key] = 0
                if color != 'blue':
                    self._camera_seen_counts[('column', 'blue')] = 0
                if color != 'yellow':
                    self._camera_seen_counts[('column', 'yellow')] = 0
                if green and self._camera_seen_counts.get('green_carpet', 0) >= CAMERA_SIGNAL_MIN_FRAMES:
                    self._push_detection_signal('green_carpet')
                if color and self._camera_seen_counts.get(('column', color), 0) >= CAMERA_SIGNAL_MIN_FRAMES:
                    if (self._last_column_signal_color == color and
                            (now - self._last_column_signal_time) < 1.0):
                        pass  # duplicate within 1 s — do not re-push
                    elif color == 'blue'   and self.start_point is not None:
                        pass
                    elif color == 'yellow' and self.end_point   is not None:
                        pass
                    else:
                        self._last_column_signal_color = color
                        self._last_column_signal_time  = now
                        # The camera thread must not call movement helpers: they call
                        # Robot.step(), which can race the main control loop in Webots.
                        self._push_detection_signal(('column', color))

    def _lidar_loop(self):
        depth_tick = 0
        last_map_update = -float('inf')
        while self.lidar_thread_running:
            try:
                if self.lidar is None:
                    time.sleep(0.5)
                    continue
                if not self.robot_on_ground():
                    time.sleep(0.1)
                    continue
                try:
                    with self._step_lock:
                        now = float(self.getTime())
                except Exception:
                    now = time.time()
                if (not self.is_turning() and
                        now - last_map_update >= TIME_STEP / 1000.0):
                    last_map_update = now
                    with self.lidar_lock:
                        self._refresh_map_lidar()
                        depth_tick += 1
                        if depth_tick % 5 == 0:
                            self._refresh_map_depth()
                        if depth_tick % 30 == 0:
                            self.occ_map.build_cost_map()
                time.sleep(0.002)
            except Exception as e:
                print(f'[LiDAR] Error: {e}')
                time.sleep(1.0)

    # ── Frontier exploration ──────────────────────────────────────────────────

    def _update_frontier(self, count, map_diff):
        regions = []
        chosen  = None
        path    = None
        chasing = False

        if not hasattr(self, '_frontier_blacklist'):
            self._frontier_blacklist = []
        self._frontier_blacklist = [
            (x, y, exp) for x, y, exp in self._frontier_blacklist if exp > count]
        bl_positions = [(x, y) for x, y, _ in self._frontier_blacklist]

        focus = self._column_biased_target()
        if focus is not None:
            path = self.occ_map.astar_path(self.get_map_position(), focus, inflation_levels=[2, 1, 0])
            if not path:
                path = self.occ_map.frontier_path(self.get_map_position(), focus)
            if path:
                return regions, focus, path, True
            self._column_focus_target = None
            self._column_focus_color = None

        if self._active_frontier_goal is not None:
            ax, ay = self._active_frontier_goal
            if self.get_map_distance((ax, ay)) <= PATH_FOLLOWING_TARGET_REACH_DISTANCE:
                self._active_frontier_goal = None
            elif (0 <= ax < self.grid_map.shape[1] and 0 <= ay < self.grid_map.shape[0] and
                  not self.occ_map.cell_blocked((ax, ay))):
                path = self.occ_map.astar_path(self.get_map_position(), (ax, ay), inflation_levels=[2, 1, 0])
                if not path:
                    path = self.occ_map.frontier_path(self.get_map_position(), (ax, ay))
                if path and len(path) > 2:
                    return regions, (ax, ay), path, False
                self._active_frontier_goal = None
            else:
                self._active_frontier_goal = None

        if (count >= EXPLORATION_START_FRONTIER_AFTER and
                count % EXPLORATION_FRONTIER_SELECTION_FREQ == 0):
            regions = self.occ_map.compute_frontiers()

            chosen  = self._column_biased_target()
            chasing = chosen is not None

            if chosen is None:
                if random.random() < 0.90:
                    chosen = self._score_frontier(regions, blacklist=bl_positions)
                    self.chosen_frontier_count += 1
                else:
                    chosen = self._random_frontier(regions)
                    self.chosen_frontier_count = 0

            if chosen:
                self._frontier_blacklist.append((chosen[0], chosen[1], count + 100))
                path = self.occ_map.astar_path(self.get_map_position(), chosen)
                if not path:  # narrow corridor — frontier_path inflation may have blocked it
                    path = self.occ_map.frontier_path(self.get_map_position(), chosen)
                if path:
                    self._active_frontier_goal = tuple(chosen)

        return regions, chosen, path, chasing

    def navigate_frontier(self, path, replan_interval=20,
                          use_global_planner=False):
        if not path:
            return False

        goal         = path[-1]
        cur_path     = list(path)
        tick         = 0
        stuck_count  = 0
        replan_count = 0
        # Clamped so a short path (<= 3 waypoints, e.g. from the
        # stall-fallback's nearby-target planner) still gets driven --
        # an unclamped tidx=3 satisfies `tidx >= len(cur_path)` immediately
        # for such paths, so the loop body never runs and the robot never
        # moves at all even though a valid path was found.
        tidx         = min(3, len(cur_path) - 1)
        MAX_STUCK    = 3
        last_replan_tick = -max(1, replan_interval or 1)
        with self.occ_map.vis_lock:
            self.occ_map.target_position = None
            self.occ_map.current_path    = cur_path
        self.start_realtime_planner(goal)
        self.update_realtime_path(cur_path)
        try:
            while tidx < len(cur_path):
                target = cur_path[tidx]

                while self.step(self.time_step) != -1:
                    tick += 1

                    rt = self.poll_realtime_planner()
                    if rt:
                        cur_path = rt
                        self.update_realtime_path(cur_path)
                        tidx = 0  # outer loop adds 3 → starts at path[3]
                        with self.occ_map.vis_lock:
                            self.occ_map.current_path = cur_path
                        print('[RT-Planner] Path swapped in navigate_frontier')
                        break

                    if self.interrupt_path:
                        self.stop_motor()
                        self.interrupt_path = False
                        return True

                    sig = None
                    with self.detection_lock:
                        sig = self._pop_detection_signal()

                    if self.obstacle_in_front():
                        replan_count += 1
                        new = self._recover_and_replan(goal, prefer_frontier=not use_global_planner, min_len=5)
                        if new and len(new) > 5:
                            cur_path = new
                            self.update_realtime_path(cur_path)
                            tidx = 0
                            last_replan_tick = tick
                            break
                        if replan_count >= 4:
                            return False
                        continue

                    if self._path_blocked_from_pose(cur_path, lookahead=18, cost_thresh=0.82):
                        new = self._recover_and_replan(goal, prefer_frontier=not use_global_planner, min_len=5)
                        if new and len(new) > 5:
                            cur_path = new
                            self.update_realtime_path(cur_path)
                            tidx = 0
                            last_replan_tick = tick
                            break
                        return False

                    if sig == 'green_carpet':
                        self.stop_motor()
                        self.mark_green_carpet_permanently(min_pixel_threshold=1500, force=True)
                        time.sleep(0.5)
                        return False
                    elif tick % 20 == 0:
                        if self.detect_green():
                            self.stop_motor()
                            self.mark_green_carpet_permanently(min_pixel_threshold=2500, force=True)
                            time.sleep(0.5)
                            return False

                    if isinstance(sig, tuple) and len(sig) == 2:
                        _, color = sig
                        cur_pos  = self.get_position()
                        prev_pos = (self.blue_prev_estimate_position if color == 'blue'
                                    else self.yellow_prev_estimate_position)
                        dist_prev = (float(np.linalg.norm(cur_pos - np.array(prev_pos)))
                                     if prev_pos is not None else float('inf'))
                        if dist_prev >= 0.8:
                            self.stop_motor()
                            found = self._estimate_column_pos(color)
                            if not found:
                                self.center_column_in_view(color)
                                found = self.get_column_center_pixels(color) >= 4 and self._estimate_column_pos(color)
                            # Always record attempt position — prevents infinite retry
                            # from the same spot when the estimation keeps failing.
                            if color == 'blue':
                                self.blue_prev_estimate_position  = cur_pos
                            else:
                                self.yellow_prev_estimate_position = cur_pos
                            if found:
                                self._commit_column_from_estimate(
                                    color, force=self._column_close_front_visible(color))
                                return True

                    try:
                        interval_hit = bool(replan_interval) and (tick - last_replan_tick >= replan_interval)
                        if interval_hit:
                            new = self.occ_map.frontier_path(self.get_map_position(), goal)
                            if not new:
                                new = self.occ_map.astar_path(self.get_map_position(), goal)
                            if self._path_usable_from_pose(new, min_len=5):
                                cur_path = list(new)
                                self.update_realtime_path(cur_path)
                                tidx = 0
                                last_replan_tick = tick
                                break
                    except Exception:
                        pass

                    with self.occ_map.vis_lock:
                        rx, ry = self.get_map_position()
                        self.occ_map.robot_position  = (int(rx), int(ry))
                        self.occ_map.current_path    = cur_path
                        self.occ_map.target_position = target
                    self.occ_map.refresh_viz()

                    reached, is_stuck = self.advance_to_waypoint(target)
                    if is_stuck:
                        stuck_count += 1
                        if stuck_count >= MAX_STUCK:
                            self.occ_map.visited_frontiers.append(tuple(goal))
                            self.stop_motor()
                            return False
                        self.stop_motor()
                        new = self._recover_and_replan(goal, prefer_frontier=not use_global_planner, min_len=5)
                        if new and len(new) > 5:
                            cur_path = list(new)
                            self.update_realtime_path(cur_path)
                            tidx = 0
                            last_replan_tick = tick
                            break
                        self.occ_map.visited_frontiers.append(tuple(goal))
                        return False
                    if reached:
                        break

                tidx += 3

            self.stop_motor()
            with self.occ_map.vis_lock:
                self.occ_map.target_position = None
                self.occ_map.current_path    = None
            return True
        finally:
            self.stop_realtime_planner()

    # ── Final path following ──────────────────────────────────────────────────

    def follow_final_path(self, path, debug_vis=False, replan_interval=60):
        """Pillar-to-pillar final approach: both columns are already found
        and committed by the time this runs, so there is nothing left to
        detect or scan for -- this is pure path following plus obstacle-
        triggered replanning. The camera thread (column/green-carpet
        detection, including its stop-and-recenter-on-a-column behaviour)
        is deliberately NOT restarted here; the LiDAR thread stays running
        since the occupancy grid still needs live updates for the
        obstacle-based realtime replanning below."""
        if not path:
            print('[FinalPath] Empty path')
            return False

        if not self.lidar_thread_running:
            self.start_lidar_thread()
        if debug_vis:
            try:
                self.occ_map.start_viz()
            except Exception:
                pass

        goal = tuple(path[-1])
        if self.get_map_distance(goal) < PATH_FOLLOWING_TARGET_REACH_DISTANCE:
            self.stop_motor()
            print('[FinalPath] Already at goal')
            return True

        def _plan_from_current():
            start = tuple(self.get_map_position())
            try:
                new = self.occ_map.astar_path(start, goal)
                return new if self._final_path_usable(new) else None
            except Exception as e:
                print(f'[FinalPath] planner error: {e}')
                return None

        # Reference-style final follow, but preserve the final pillar-to-pillar
        # route produced by explore() instead of replacing it at startup.
        # _final_path_usable() ray-casts every path segment against the grid,
        # which can be tripped up by noisy/transient DEPTH_OBSTACLE cells near
        # the pillars themselves; treat it as a preference, not a hard gate --
        # a nominally "unusable" path is still driven, with in-loop obstacle
        # detection and replanning (below) handling any real blockage as the
        # robot actually reaches it, same as the known-working reference.
        cur_path = list(path)
        if not self._final_path_usable(cur_path):
            replanned = _plan_from_current()
            if replanned:
                cur_path = replanned
        if not cur_path:
            print('[FinalPath] No path available')
            return False

        with self.occ_map.vis_lock:
            rx, ry = self.get_map_position()
            self.occ_map.robot_position  = (int(rx), int(ry))
            self.occ_map.current_path    = cur_path
            self.occ_map.target_position = goal
        if debug_vis:
            self.occ_map.refresh_viz()

        try:
            align_idx = min(1, len(cur_path) - 1)
            self.align_to_path(cur_path[align_idx])
        except Exception:
            pass

        def _build_waypoints(p):
            # Both pillars are already committed and the camera thread is
            # off -- unlike exploration, there is no reason to hug a dense
            # trail of waypoints only 2 cells (~4-6 cm) apart. A coarser
            # stride here means fewer stop-to-realign transitions and lets
            # DWA drive longer, smoother stretches instead of jerky
            # start/stop segments.
            p = list(p)
            if len(p) <= 1:
                return [goal]
            if len(p) < 25:
                stride = 6
            elif len(p) < 60:
                stride = 8
            else:
                stride = 10
            idxs   = list(range(stride, len(p), stride))
            if len(p) - 1 not in idxs:
                idxs.append(len(p) - 1)
            # Explicit first target: p[0] is the start pillar's own cell
            # (explore() prepends it) -- the stride-based indices above
            # start at `stride` and would otherwise skip driving to it as a
            # waypoint at all, leaving the start pillar untouched even
            # though it's now part of the path.
            if 0 not in idxs:
                idxs.insert(0, 0)
            idxs = [i for i in idxs if 0 <= i < len(p)]
            return [tuple(p[i]) for i in idxs] or [goal]

        waypoints = _build_waypoints(cur_path)
        tick = 0
        i    = 0
        last_replan_tick = 0
        map_wait_steps = 40

        while i < len(waypoints):
            target = waypoints[i]
            while self.step(self.time_step) != -1:
                tick += 1

                if self.obstacle_in_front():
                    new = self._recover_and_replan(goal, prefer_frontier=False, min_len=3)
                    if self._final_path_usable(new):
                        cur_path = list(new)
                        waypoints = _build_waypoints(cur_path)
                        i = -1
                        last_replan_tick = tick
                        break
                    for _ in range(map_wait_steps):
                        if self.step(self.time_step) == -1:
                            self.stop_motor()
                            return False
                    new = _plan_from_current()
                    if self._final_path_usable(new):
                        cur_path = list(new)
                        waypoints = _build_waypoints(cur_path)
                        i = -1
                        last_replan_tick = tick
                        break
                    print('[FinalPath] Replan failed after obstacle; continuing supplied path')

                try:
                    # Looser cost_thresh than exploration's default (0.82) --
                    # the final leg shouldn't replan over cost-map noise near
                    # already-committed pillars, only real blockage.
                    blocked = self._path_blocked_from_pose(cur_path, lookahead=FINAL_PATH_LOOKAHEAD_CELLS, cost_thresh=0.92)
                    interval_hit = bool(replan_interval) and (tick - last_replan_tick >= replan_interval)
                    if blocked or (interval_hit and not self._path_usable_from_pose(cur_path, min_len=3, lookahead=FINAL_PATH_LOOKAHEAD_CELLS, cost_thresh=0.92)):
                        new = self._attempt_replan(goal)
                        if self._final_path_usable(new):
                            cur_path = list(new)
                            waypoints = _build_waypoints(cur_path)
                            i = -1
                            last_replan_tick = tick
                            break
                        new = _plan_from_current()
                        if self._final_path_usable(new):
                            cur_path = list(new)
                            waypoints = _build_waypoints(cur_path)
                            i = -1
                            last_replan_tick = tick
                            break
                        last_replan_tick = tick
                except Exception:
                    pass

                with self.occ_map.vis_lock:
                    rx, ry = self.get_map_position()
                    self.occ_map.robot_position  = (int(rx), int(ry))
                    self.occ_map.current_path    = cur_path
                    self.occ_map.target_position = target
                self.occ_map.refresh_viz()

                reached, is_stuck = self.advance_to_waypoint(target)
                if is_stuck or (len(self.get_distances()) and
                                min(self.get_distances()) < 0.035):
                    self.stop_motor()
                    new = self._recover_and_replan(goal, prefer_frontier=False, min_len=3)
                    if self._final_path_usable(new):
                        cur_path  = list(new)
                        waypoints = _build_waypoints(cur_path)
                        i = -1
                        last_replan_tick = tick
                        break
                    for _ in range(map_wait_steps):
                        if self.step(self.time_step) == -1:
                            self.stop_motor()
                            return False
                    new = _plan_from_current()
                    if self._final_path_usable(new):
                        cur_path  = list(new)
                        waypoints = _build_waypoints(cur_path)
                        i = -1
                        last_replan_tick = tick
                        break
                    print(f'[FinalPath] Recovery: skipping waypoint {i}, continuing')
                    break

                if reached:
                    if self.get_map_distance(goal) < PATH_FOLLOWING_TARGET_REACH_DISTANCE:
                        self.stop_motor()
                        print('[FinalPath] Goal reached')
                        return True
                    break
            i += 1
            if self.get_map_distance(goal) < PATH_FOLLOWING_TARGET_REACH_DISTANCE:
                self.stop_motor()
                print('[FinalPath] Goal reached (post-check)')
                return True

        self.stop_motor()
        success = self.get_map_distance(goal) < PATH_FOLLOWING_TARGET_REACH_DISTANCE
        print(f'[FinalPath] Finished, success={success}')
        return success

    def _confirm_pillar_close(self, color, max_attempts=4):
        """Drive toward the current best estimate of `color`'s pillar until a
        close-range, front-facing, LiDAR-grounded confirmation lands,
        refining/overwriting the committed position each attempt. Bounded by
        max_attempts so a pillar boxed in by obstacles can't stall the run
        forever -- if it never confirms, the best (possibly still-far)
        position found so far is kept and used as-is for the final path."""
        if self._column_state.get(color) == 'CONFIRMED':
            return True
        for _ in range(max_attempts):
            est = self.blue_estimated_pos if color == 'blue' else self.yellow_estimated_pos
            if est is None:
                return False
            ex, ey = int(round(float(est[0]))), int(round(float(est[1])))
            target = self._column_approach_target(est, min_radius=3, max_radius=10)
            if target is None:
                target = self._final_pillar_access_cell(
                    (ex, ey), prefer=self.get_map_position(),
                    min_radius=3, max_radius=10)
            if target is None:
                return False
            start = self.get_map_position()
            path = self.occ_map.astar_path(start, target, inflation_levels=[2, 1, 0])
            if path and len(path) > 1:
                self.navigate_frontier(path)
            self.center_column_in_view(color)
            found = self._estimate_column_pos(color)
            if not found:
                found = (self.get_column_center_pixels(color) >= 4 and
                          self._estimate_column_pos(color))
            if found:
                self._commit_column_from_estimate(
                    color, force=self._column_close_front_visible(color))
            if self._column_state.get(color) == 'CONFIRMED':
                return True
        return self._column_state.get(color) == 'CONFIRMED'

    # ── Main exploration loop ─────────────────────────────────────────────────

    def explore(self, debug=True):
        # Fixed seed so frontier-selection randomness (the 90/10 score/random
        # branch, fallback-target jitter) is reproducible run-to-run -- this
        # does not remove variance from background-thread scheduling or
        # sim/sensor timing, only from this module's own random.* calls.
        random.seed(0)

        self.stop_realtime_planner()
        self.stop_camera_thread()
        self.stop_lidar_thread()
        self._reset_transient_depth_obstacles()
        self.start_camera_thread()
        self.start_lidar_thread()

        if debug:
            self.occ_map.start_viz()

        prev_grid = self.occ_map.grid_map.copy()
        map_diff  = 1.0
        count     = 0

        time.sleep(0.2)
        initial_sig = self.scan_360()
        initial_committed = self._handle_scan_signal(initial_sig)
        if isinstance(initial_sig, tuple) and len(initial_sig) >= 2 and initial_sig[0] == 'column':
            print(f'[Scan] Initial {initial_sig[1]} committed={initial_committed} '
                  f'blue={self.start_point is not None} yellow={self.end_point is not None}')

        while self.step(self.time_step) != -1 and not self.found_all_2_columns():

            with self.detection_lock:
                sig = self._pop_detection_signal()

            if sig == 'green_carpet':
                self.stop_motor()
                self.mark_green_carpet_permanently(min_pixel_threshold=1500, force=True)
                continue

            if isinstance(sig, tuple) and len(sig) == 2:
                _, color = sig
                cur_pos  = self.get_position()
                prev_pos = (self.blue_prev_estimate_position if color == 'blue'
                            else self.yellow_prev_estimate_position)
                dist_prev = (float(np.linalg.norm(cur_pos - np.array(prev_pos)))
                             if prev_pos is not None else float('inf'))
                if dist_prev >= 0.8:
                    self.stop_motor()
                    found = self._estimate_column_pos(color)
                    if not found:
                        self.center_column_in_view(color)
                        found = self.get_column_center_pixels(color) >= 4 and self._estimate_column_pos(color)
                    # Always record the attempt position so the robot must move
                    # before retrying — prevents the infinite-stop-retry loop when
                    # the estimation keeps failing from the same spot.
                    if color == 'blue':
                        self.blue_prev_estimate_position  = cur_pos
                    else:
                        self.yellow_prev_estimate_position = cur_pos
                    if found:
                        self._commit_column_from_estimate(
                            color, force=self._column_close_front_visible(color))
                        focus_path = self._column_focus_path()
                        if focus_path and len(focus_path) > 2:
                            self.navigate_frontier(focus_path)
                    continue
                # dist_prev < 0.8: too soon to retry an estimate for this
                # color. Don't skip the tick outright -- fall through into
                # the normal _update_frontier()/navigate_frontier() path
                # below so the robot keeps moving (and can build up the
                # distance needed for the next estimate attempt) instead of
                # sitting still re-seeing the same column forever.

            map_diff = utils.map_delta_ratio(prev_grid, self.occ_map.grid_map)

            # Single goal selection: _update_frontier owns planning; the loop owns
            # navigation. This removes the dual-goal confusion where _update_frontier
            # navigated internally AND the outer loop also tried to navigate.
            _, chosen, path_to_chosen, chasing = self._update_frontier(count, map_diff)

            # One navigation call per iteration — no competing goals
            if path_to_chosen:
                ok = self.navigate_frontier(path_to_chosen)
                if ok:
                    self._last_frontier_goal = chosen
                    if self._active_frontier_goal == tuple(chosen):
                        self._active_frontier_goal = None
                if ok and chasing and not self.found_all_2_columns():
                    scan_sig = self.scan_360()
                    committed = self._handle_scan_signal(scan_sig)
                    if (not committed and isinstance(scan_sig, tuple) and
                            len(scan_sig) >= 2 and scan_sig[0] == 'column'):
                        self._suspend_column_focus(scan_sig[1], seconds=8.0)

            prev_grid = self.occ_map.grid_map.copy()

            if debug:
                with self.occ_map.vis_lock:
                    rx, ry = self.get_map_position()
                    self.occ_map.robot_position  = (rx, ry)
                    self.occ_map.current_path    = path_to_chosen
                    self.occ_map.target_position = chosen
                    col_pts = []
                    if self.blue_estimated_pos is not None:
                        col_pts.append((int(self.blue_estimated_pos[0]),
                                        int(self.blue_estimated_pos[1]),
                                        (0, 255, 255)))
                    if self.yellow_estimated_pos is not None:
                        col_pts.append((int(self.yellow_estimated_pos[0]),
                                        int(self.yellow_estimated_pos[1]),
                                        (255, 255, 0)))
                    self.occ_map.column_points = col_pts
                self.occ_map.refresh_viz()

            count += 1

        # found_all_2_columns() only requires a provisional commit (which can
        # happen from up to COLUMN_COMMIT_MAX_MAP_DISTANCE_M away), so it can end
        # the loop above before the robot ever actually got close to either
        # pillar. Use the camera/lidar threads one last time, while they're
        # still running, to drive in and get a close-range confirmation for
        # each pillar before the final path is planned from these positions.
        if self.found_all_2_columns():
            for color in ('blue', 'yellow'):
                self._confirm_pillar_close(color)

        self.stop_camera_thread()
        self.stop_lidar_thread()
        # Deliberately no stop_viz() here -- the same plotter window/process
        # started above must carry straight through into follow_final_path()
        # rather than being torn down and respawned as a new OS window.
        with self.detection_lock:
            self.camera_detection_signal  = None
            self.camera_detection_queue.clear()
            self.green_carpet_active      = False
            self.last_green_carpet_points = []
            time.sleep(0.5)
        self.stop_motor()
        print('[Explore] Done.')

        if self.start_point is not None and self.end_point is not None:
            blue   = tuple(self.start_point)
            yellow = tuple(self.end_point)
            # Plan the final path pillar-to-pillar with A*, not from the
            # robot's current pose. Always blue -> yellow, regardless of
            # which one was physically discovered first during exploration.
            start_pillar, end_pillar = blue, yellow
            start_access = self._final_pillar_access_cell(
                start_pillar, prefer=self.get_map_position())
            end_access = self._final_pillar_access_cell(
                end_pillar, prefer=start_access or self.get_map_position())
            if start_access is None or end_access is None:
                print('[Explore] WARNING: both pillars found but no free access '
                      'cell exists near a pillar.')
                return []
            path = self._plan_final_pillar_path(start_access, end_access)
            if not path:
                # astar_path only ever traverses cells already confirmed
                # FREESPACE -- UNKNOWN cells are treated as blocked by design
                # (astar_2_spline.py checks `grid == 0`, and UNKNOWN == 255).
                # The main loop above stops exploring the instant both
                # pillars are found, which can be well before the region
                # between them has ever actually been mapped -- so "no
                # route" here can mean the route is genuinely absent from
                # the known map yet, not that planning failed. Resume
                # exploration, bounded, to bridge the gap and retry instead
                # of giving up on the first attempt.
                path = self._bridge_explore_and_replan(start_access, end_access, debug=debug)
            if path:
                # astar_path only ever plans between FREESPACE cells, so
                # start_access/end_access are the nearest reachable cells
                # *next to* each pillar, not the pillars themselves (an
                # obstacle cell can't be a search endpoint) -- the robot
                # would otherwise stop short of actually touching either
                # one. Extend the route with the pillars' own cells so
                # following it drives all the way to contact at both ends;
                # _final_path_clear()'s endpoint_margin already tolerates an
                # obstacle right at a path's start/end for exactly this case.
                path = [tuple(int(c) for c in start_pillar)] + list(path) + \
                       [tuple(int(c) for c in end_pillar)]
                print(f'[Explore] Final pillar-to-pillar path: {len(path)} waypoints')
            else:
                print('[Explore] WARNING: both pillars found but planner found no '
                      'route at all — pillars are unreachable on the current map.')
            return path or []
        return []

    def _bridge_explore_and_replan(self, start_access, end_access, debug=False,
                                    max_iters=400, retry_every=20):
        """Resume ordinary frontier exploration (same machinery as explore()'s
        main loop) to grow map connectivity between the two already-found
        pillars, retrying the final plan every `retry_every` iterations, up
        to `max_iters`. Bounded so an actually-unreachable pair of pillars
        (e.g. separated by a wall with no corridor) still terminates."""
        if not self.camera_thread_running:
            self.start_camera_thread()
        if not self.lidar_thread_running:
            self.start_lidar_thread()
        if debug:
            try:
                self.occ_map.start_viz()
            except Exception:
                pass
        print('[Explore] Final route not yet connected — resuming exploration '
              'to bridge the gap...')
        prev_grid = self.occ_map.grid_map.copy()
        path = None
        count = 0
        while count < max_iters and self.step(self.time_step) != -1:
            map_diff = utils.map_delta_ratio(prev_grid, self.occ_map.grid_map)
            _, chosen, path_to_chosen, _ = self._update_frontier(count, map_diff)
            if path_to_chosen:
                self.navigate_frontier(path_to_chosen)
            prev_grid = self.occ_map.grid_map.copy()
            count += 1
            if count % retry_every == 0:
                path = self._plan_final_pillar_path(start_access, end_access)
                if path:
                    break
        if not path:
            path = self._plan_final_pillar_path(start_access, end_access)
        self.stop_camera_thread()
        self.stop_lidar_thread()
        # No stop_viz() here either -- same persistent window as explore()
        # and follow_final_path() share, see the comment in explore().
        self.stop_motor()
        return path

    def _plan_final_pillar_path(self, start_access, end_access):
        """Final-path planner for the pillar-to-pillar route: a single
        astar_path() call, used as-is.

        The previous version ran a multi-tier inflation/clearance search
        and then discarded candidates over _final_path_hard_clear/
        _final_path_usable -- soft/live obstacle flags (e.g. a transient
        DEPTH_OBSTACLE vote) could reject a geometrically fine path,
        which is exactly why this sometimes reported no route even when
        one clearly existed on the mapped grid. follow_final_path()
        already does live obstacle detection and replanning while
        driving (obstacle_in_front, _path_blocked_from_pose,
        _attempt_replan), so the plan-time path doesn't need to be
        pre-filtered this defensively -- real blockages get handled as
        the robot actually reaches them. Only _bridge_explore_and_replan
        should kick in, and only when astar_path genuinely finds no route
        at all in the known map.
        """
        return self.occ_map.astar_path(start_access, end_access)


    # ── Green carpet ──────────────────────────────────────────────────────────

    def get_green_carpet_points(self):
        hsv_img = self.get_hsv_image()
        if hsv_img is None:
            return np.array([])
        green_mask = utils.extract_color_mask(hsv_img, 'green')
        kernel = cv2.getStructuringElement(
            cv2.MORPH_RECT,
            (GREEN_CARPET_DILATION_KERNEL_SIZE, GREEN_CARPET_DILATION_KERNEL_SIZE))
        if GREEN_CARPET_DILATION_ITERATIONS > 0:
            green_mask = cv2.dilate(green_mask, kernel, iterations=GREEN_CARPET_DILATION_ITERATIONS)
        green_mask = cv2.morphologyEx(green_mask, cv2.MORPH_OPEN, kernel, iterations=1)
        h_start = int(self.cam_height * 0.5)
        green_mask[:h_start, :] = 0
        try:
            n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
                (green_mask == 255).astype(np.uint8), connectivity=8
            )
            if n_labels > 1:
                areas = stats[1:, cv2.CC_STAT_AREA]
                best = 1 + int(np.argmax(areas))
                green_mask = np.where(labels == best, 255, 0).astype(np.uint8)
        except Exception:
            pass
        v_px, u_px = np.where(green_mask == 255)
        if len(v_px) == 0:
            return np.array([])
        x_n = (u_px - self.cx) / self.fx
        y_n = (v_px - self.cy) / self.fy
        D   = self.camera_height_m / (y_n + 1e-6)
        ok  = (y_n > 0.001) & (D > 0.1) & (D < GREEN_CARPET_MAX_PROJECTION_DISTANCE)
        D_v, x_nv = D[ok], x_n[ok]
        pts_local = np.stack([D_v + self.X_offset, -D_v * x_nv + self.Y_offset], axis=1)
        pts_world = self.transform_points_to_world(pts_local)
        return self.convert_to_map_coordinate_matrix(pts_world)

    def mark_green_carpet_permanently(self, min_pixel_threshold=10,
                                      new_area_threshold=0.5, force=False):
        now = time.time()
        if not force and (now - self.last_green_mark_time) < self.green_mark_cooldown:
            return False
        if not self.green_carpet_lock.acquire(blocking=False):
            return False
        self.green_carpet_active = True
        try:
            quick_pts = self.get_green_carpet_points()
            if quick_pts.shape[0] < min_pixel_threshold:
                return False
            pts = quick_pts.astype(np.int32)
            H, W = self.grid_map.shape
            xi, yi = pts[:, 0], pts[:, 1]
            valid  = (xi >= 0) & (xi < W) & (yi >= 0) & (yi < H)
            xi, yi = xi[valid], yi[valid]
            if xi.size == 0:
                return False
            green_protect = (self.grid_map == GREEN_CARPET)
            if np.sum(~green_protect[yi, xi]) / xi.size < new_area_threshold:
                return False
            centroid   = np.array([float(np.mean(xi)), float(np.mean(yi))])
            robot_mp = self.get_map_position()
            if np.linalg.norm(centroid - np.array(robot_mp, dtype=float)) > GREEN_CARPET_MARK_MAX_MAP_DISTANCE:
                return False
            min_d, closest_idx = float('inf'), -1
            for i, (ox, oy, _) in enumerate(self.green_carpet_patches):
                d = np.linalg.norm(centroid - np.array([ox, oy]))
                if d < min_d:
                    min_d, closest_idx = d, i
            if closest_idx != -1 and min_d < self.green_carpet_proximity_threshold:
                if xi.size > self.green_carpet_patches[closest_idx][2] * 1.2:
                    self.green_carpet_patches[closest_idx] = (centroid[0], centroid[1], int(xi.size))
                else:
                    return False
            for _ in range(15):
                if self.step(self.time_step) == -1:
                    break
            self._back_from_green()
            self.stop_motor()
            time.sleep(1.0)
            confirm_threshold = max(
                50,
                int(min_pixel_threshold * GREEN_CARPET_CONFIRM_MIN_POINTS_RATIO)
            )
            cur_pts = self.get_green_carpet_points()
            if cur_pts.shape[0] < confirm_threshold:
                return False
            pts2 = cur_pts.astype(np.int32)
            xi2, yi2 = pts2[:, 0], pts2[:, 1]
            valid2   = (xi2 >= 0) & (xi2 < W) & (yi2 >= 0) & (yi2 < H)
            xi2, yi2 = xi2[valid2], yi2[valid2]
            if xi2.size == 0:
                return False
            # Hard per-cell guard: drop any projected carpet cell farther than the
            # max-cell distance from the robot. This guarantees the carpet is never
            # plotted from far away even when the centroid passes the gate above.
            rmp = np.array(self.get_map_position(), dtype=np.float32)
            cell_d = np.hypot(xi2 - rmp[0], yi2 - rmp[1])
            near = cell_d <= GREEN_CARPET_MAX_CELL_MAP_DISTANCE
            xi2, yi2 = xi2[near], yi2[near]
            if xi2.size == 0:
                return False
            centroid2 = np.array([float(np.mean(xi2)), float(np.mean(yi2))])
            if np.linalg.norm(centroid2 - np.array(self.get_map_position(), dtype=float)) > GREEN_CARPET_MARK_MAX_MAP_DISTANCE:
                return False
            try:
                fill_mask = np.zeros(self.occ_map.grid_map.shape, dtype=np.uint8)
                fill_mask[yi2, xi2] = 1
                kernel = np.ones((2, 2), dtype=np.uint8)
                fill_mask = cv2.morphologyEx(fill_mask, cv2.MORPH_OPEN, kernel, iterations=1)
                self.occ_map.grid_map[fill_mask == 1] = int(GREEN_CARPET)
                if min_d >= self.green_carpet_proximity_threshold:
                    self.green_carpet_patches.append((centroid[0], centroid[1], int(xi2.size)))
                self.last_green_mark_time = time.time()
                return True
            except Exception as e:
                print(f'[warning] green carpet mark failed: {e}')
                return False
        finally:
            self.green_carpet_active = False
            try:
                self.green_carpet_lock.release()
            except Exception:
                pass

    def _back_from_green(self):
        back_speed = -3.0
        self.set_robot_velocity(back_speed, back_speed)
        for _ in range(300):
            if self.step(self.time_step) == -1:
                break
            hsv = self.get_hsv_image()
            if hsv is None:
                break
            h, w, _ = hsv.shape
            bottom  = hsv[h - 30:, :]
            if cv2.countNonZero(utils.extract_color_mask(bottom, 'green')) < 50:
                break
            self.set_robot_velocity(back_speed, back_speed)
        self.stop_motor()
