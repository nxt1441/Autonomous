from controller import Robot
import numpy as np
import math
import cv2
import random
import time
import threading
import utils
from CONSTANTS import *
from map import OccupancyGrid
from setup import setup_robot


class MyRobot(Robot):

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
        self.blue_pos_update_count    = 0
        self.yellow_pos_update_count  = 0
        self.blue_estimation_positions   = []
        self.yellow_estimation_positions = []
        self.blue_prev_estimate_position   = None
        self.yellow_prev_estimate_position = None
        self.estimation_distance_threshold = 0.5
        self.last_found_color = None
        self.last_found_point = None

        self.last_closure_time = 0.0
        self.closure_cooldown  = CLOSURE_MARK_COOLDOWN

        self.steps_since_turning  = 0
        self.is_currently_turning = False

        self.detection_thread        = None
        self.camera_thread_running   = False
        self.camera_detection_signal = None
        self.detection_lock          = threading.Lock()

        self.lidar_thread        = None
        self.lidar_thread_running = False
        self.lidar_lock          = threading.Lock()

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
        self._robot_trail                     = set() # coarsened path history for frontier anti-revisit
        self._last_frontier_goal              = None  # last successfully reached frontier centroid

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

        self.camera_height_m  = 0.17
        self.camera_pitch_rad = 0.0
        self.X_offset = 0.03
        self.Y_offset = 0.0

        self.green_carpet_patches             = []
        self.green_carpet_proximity_threshold = 200.0
        self.last_green_mark_time  = 0.0
        self.green_mark_cooldown   = 8.0
        self.last_green_carpet_points = []
        self.green_carpet_active   = False

        self.counter_obstacle_recoveries = 0

        self._odom_x      = INITIAL_X
        self._odom_y      = INITIAL_Y
        self._odom_theta  = INITIAL_THETA
        self._odom_prev_left  = 0.0
        self._odom_prev_right = 0.0
        self._odom_initialized = False

    # ── step() override ───────────────────────────────────────────────────────

    def step(self, duration_ms=None):
        if duration_ms is None:
            duration_ms = self.time_step
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
        _mx = half + int(self._odom_x / RESOLUTION)
        _my = half - int(math.ceil(self._odom_y / RESOLUTION))
        self._robot_trail.add((_mx >> 3, _my >> 3))

    # ── Pose accessors ────────────────────────────────────────────────────────

    def get_position(self):
        return np.array([self._odom_x, self._odom_y])

    def get_heading(self, kind='deg'):
        cn  = self.imu['compass'].getValues()
        rad = math.atan2(-cn[1], cn[0])
        return rad if kind == 'rad' else np.degrees(rad)

    def get_map_position(self):
        half = self.occ_map.map_size // 2
        mx = half + int(self._odom_x / RESOLUTION)
        my = half - int(math.ceil(self._odom_y / RESOLUTION))
        return np.array([mx, my])

    def get_map_distance(self, map_target):
        return np.linalg.norm(self.get_map_position() - np.array(map_target))

    def convert_to_map_coordinates(self, x, y):
        half = self.occ_map.map_size // 2
        mx   = half + int(x / RESOLUTION)
        my   = half - int(math.ceil(y / RESOLUTION))
        return int(mx), int(my)

    def convert_to_world_coordinates(self, mx, my):
        half = self.occ_map.map_size // 2
        x = (mx - half) * RESOLUTION
        y = (half - my) * RESOLUTION
        return float(x), float(y)

    def convert_to_map_coordinate_matrix(self, pts_world):
        return self.occ_map.world_pts_to_map(pts_world)

    def position_ahead(self, distance_m):
        h = self.get_heading('rad')
        return np.array([self._odom_x + distance_m * np.cos(h),
                         self._odom_y + distance_m * np.sin(h)])

    def get_angle_diff(self, map_target, kind='deg'):
        heading     = self.get_heading('rad')
        mx, my      = self.get_map_position()
        target_ang  = np.arctan2(map_target[1] - my, map_target[0] - mx)
        diff        = abs(target_ang - heading)
        if diff > np.pi:
            diff = 2 * np.pi - diff
        return diff if kind == 'rad' else np.degrees(diff)

    # ── Ground check ──────────────────────────────────────────────────────────

    def robot_on_ground(self, max_tan=0.08):
        try:
            ax, ay, az = self.imu['accelerometer'].getValues()
            pitch = math.atan2(-ax, max(1e-9, math.sqrt(ay * ay + az * az)))
            return abs(math.tan(pitch)) < max_tan
        except Exception:
            return True

    # ── Motor control ─────────────────────────────────────────────────────────

    def stop_motor(self):
        for m in self.motors.values():
            m.setVelocity(0.0)

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

    def velocity_to_wheel_speeds(self, v, w):
        half = self.axle_length / 2.0
        return (v - half * w) / self.wheel_radius, (v + half * w) / self.wheel_radius

    def is_turning(self):
        return abs(self.motors['fl'].getVelocity() - self.motors['fr'].getVelocity()) > 0.02

    def get_distances(self):
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

    def move_backward_milisecond(self, ms=300):
        self.set_robot_velocity(MOTOR_VELOCITY_BACKWARD, MOTOR_VELOCITY_BACKWARD)
        self.step(ms)
        self.stop_motor()

    def turn_degrees(self, degrees, direction='right'):
        rad_target    = np.deg2rad(degrees)
        start_heading = self.get_heading('rad')
        if direction == 'right':
            self.set_robot_velocity(MOTOR_VELOCITY_TURN, -MOTOR_VELOCITY_TURN)
        else:
            self.set_robot_velocity(-MOTOR_VELOCITY_TURN, MOTOR_VELOCITY_TURN)
        while self.step(self.time_step) != -1:
            turned = abs(utils.angle_wrap(self.get_heading('rad'), start_heading))
            if turned >= rad_target - TURN_ANGLE_COMPLETION_THRESHOLD:
                break
        self.stop_motor()

    # ── LiDAR helpers ─────────────────────────────────────────────────────────

    def get_pointcloud_2d(self):
        if self.lidar is None:
            return np.array([])
        pts = self.lidar.getPointCloud()
        if not pts:
            return np.array([])
        arr = np.array([[p.x, p.y] for p in pts], dtype=np.float32)
        return arr[~np.isinf(arr).any(axis=1)]

    def transform_points_to_world(self, pts_local):
        if len(pts_local) == 0:
            return pts_local
        theta = self.get_heading('rad')
        R = np.array([[np.cos(theta), -np.sin(theta)],
                      [np.sin(theta),  np.cos(theta)]])
        return pts_local @ R.T + np.array([self._odom_x, self._odom_y])

    def get_pointcloud_world_coordinates(self):
        return self.transform_points_to_world(self.get_pointcloud_2d())

    def get_lidar_front_min_dist(self, angle_range_deg=30):
        pts = self.get_pointcloud_2d()
        if len(pts) == 0:
            return float('inf')
        angles = np.arctan2(pts[:, 1], pts[:, 0])
        dists  = np.linalg.norm(pts, axis=1)
        lim    = np.radians(angle_range_deg)
        front  = dists[(angles > -lim) & (angles < lim)]
        return float(np.min(front)) if len(front) > 0 else float('inf')

    def get_min_front_distance(self, angle_deg=None):
        if angle_deg is None:
            angle_deg = LIDAR_FRONT_CONE_ANGLE
        pts = self.get_pointcloud_2d()
        if pts.shape[0] == 0:
            ds = self.get_distances()
            return float(min(ds[0], ds[2])) if len(ds) >= 3 else float('inf')
        angles = np.arctan2(pts[:, 1], pts[:, 0]) * 180.0 / np.pi
        front  = pts[(angles > -angle_deg) & (angles < angle_deg)]
        if front.shape[0] == 0:
            ds = self.get_distances()
            return float(min(ds[0], ds[2])) if len(ds) >= 3 else float('inf')
        return float(np.min(np.linalg.norm(front, axis=1)))

    def _get_scan_fp(self, sectors=12):
        """Return per-sector minimum LiDAR distances (robot frame).

        Used to detect encoder slip: if the scan doesn't change between steps
        while the encoders report movement, the robot is physically stuck.
        """
        try:
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

    def _refresh_map_lidar(self):
        pts     = self.get_pointcloud_world_coordinates()
        map_pos = self.get_map_position()
        self.occ_map.process_scan(map_pos, pts)

    def _refresh_map_depth(self, depth_stride=3, max_depth=3.5):
        if self.camera_depth is None:
            return
        try:
            depth_data = self.camera_depth.getRangeImage()
        except Exception:
            return
        if not depth_data:
            return
        try:
            w   = self.camera_depth.getWidth()
            h   = self.camera_depth.getHeight()
            fov = self.camera_depth.getFov()
        except Exception:
            return

        depth_arr = np.array(depth_data, dtype=np.float32).reshape(h, w)
        valid = np.isfinite(depth_arr) & (depth_arr > 0.05) & (depth_arr < max_depth)
        depth_arr[~valid] = np.inf

        row_lo  = max(0, int(h * 0.05))
        row_hi  = min(h, int(h * 0.90))
        col_min = depth_arr[row_lo:row_hi, :].min(axis=0)

        heading = self.get_heading('rad')
        rmap    = self.get_map_position()
        rx_m, ry_m = int(rmap[0]), int(rmap[1])

        for col in range(0, w, depth_stride):
            d = float(col_min[col])
            if not math.isfinite(d):
                continue
            beta  = -(col - w / 2.0) * (fov / max(1, w))
            alpha = heading + beta
            wx = self._odom_x + d * math.cos(alpha)
            wy = self._odom_y + d * math.sin(alpha)
            mx, my = self.convert_to_map_coordinates(wx, wy)
            for px, py in utils.ray_cells((rx_m, ry_m), (mx, my))[:-1]:
                if 0 <= px < MAP_SIZE and 0 <= py < MAP_SIZE:
                    if self.occ_map.log_odds[py, px] < 3.5:
                        self.occ_map.log_odds[py, px] -= 0.36
            if 0 <= mx < MAP_SIZE and 0 <= my < MAP_SIZE:
                self.occ_map.log_odds[my, mx] += 0.70

        self.occ_map.rebuild_grid()

    # ── Camera helpers ────────────────────────────────────────────────────────

    def get_hsv_image(self):
        if self.camera_rgb is None:
            return None
        try:
            raw = self.camera_rgb.getImage()
        except Exception:
            return None
        if not raw:
            return None
        try:
            w = self.camera_rgb.getWidth()
            h = self.camera_rgb.getHeight()
            img = np.frombuffer(raw, np.uint8).reshape((h, w, 4))
            bgr = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
            return cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
        except Exception:
            return None

    def get_bottom_half_hsv(self):
        hsv = self.get_hsv_image()
        if hsv is None:
            return None
        return hsv[hsv.shape[0] // 2:, :]

    def get_camera_depth_cm(self):
        if self.camera_depth is None:
            return None
        try:
            raw = self.camera_depth.getRangeImage()
        except Exception:
            return None
        if not raw:
            return None
        try:
            w = self.camera_depth.getWidth()
            h = self.camera_depth.getHeight()
            depth = np.array(raw).reshape((h, w)) * 100.0
            return np.where(np.isinf(depth), -1, depth).astype(np.int16)
        except Exception:
            return None

    # ── Colour detection ──────────────────────────────────────────────────────

    def there_is_red_wall(self):
        hsv = self.get_hsv_image()
        if hsv is None:
            return False
        h, w, _ = hsv.shape
        m1 = cv2.inRange(hsv, np.array(RED_WALL_HSV_LOWER1), np.array(RED_WALL_HSV_UPPER1))
        m2 = cv2.inRange(hsv, np.array(RED_WALL_HSV_LOWER2), np.array(RED_WALL_HSV_UPPER2))
        return cv2.countNonZero(cv2.bitwise_or(m1, m2)) > w * h * COLOR_DETECTION_RED_PIXEL_RATIO

    def detect_green(self):
        hsv = self.get_bottom_half_hsv()
        if hsv is None:
            return None
        return cv2.countNonZero(utils.extract_color_mask(hsv, 'green')) > 50

    def detect_column(self):
        hsv = self.get_hsv_image()
        if hsv is None:
            return None
        if cv2.countNonZero(utils.extract_color_mask(hsv, 'yellow')):
            return 'yellow'
        if cv2.countNonZero(utils.extract_color_mask(hsv, 'blue')):
            return 'blue'
        return None

    def detect_column_strict(self, min_pixels=800):
        hsv = self.get_hsv_image()
        if hsv is None:
            return None
        roi = hsv[hsv.shape[0] // 2:, :]
        if cv2.countNonZero(utils.extract_color_mask(roi, 'yellow')) > min_pixels:
            return 'yellow'
        if cv2.countNonZero(utils.extract_color_mask(roi, 'blue')) > min_pixels:
            return 'blue'
        return None

    def found_all_2_columns(self):
        return self.start_point is not None and self.end_point is not None

    def column_close(self, column_mask):
        return (np.count_nonzero(column_mask) /
                (column_mask.shape[0] * column_mask.shape[1])) > 0.25

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

    def is_column_fully_in_frame(self, hsv, color, top_strip=50):
        if hsv is None:
            return False
        return cv2.countNonZero(utils.extract_color_mask(hsv, color)[:top_strip, :]) == 0

    def estimate_column_distance(self, color):
        self.stop_motor()
        depth_img = self.get_camera_depth_cm()
        hsv_img   = self.get_hsv_image()
        if depth_img is None or hsv_img is None:
            return None
        mask = utils.extract_color_mask(hsv_img, color)
        if not np.any(mask):
            return None
        valid_depths = depth_img[mask != 0]
        valid_depths = valid_depths[valid_depths > 0]
        if len(valid_depths) == 0:
            area_ratio = float(np.count_nonzero(mask)) / float(mask.size)
            return 0.5 if area_ratio > 0.20 else None
        max_d = float(np.max(valid_depths))
        if not self.is_column_fully_in_frame(hsv_img, color):
            max_d *= 1.25 if max_d < 110.0 else 1.1
        col_h = 125.0
        if max_d <= col_h:
            return float(np.mean(valid_depths))
        return float(np.sqrt(max_d ** 2 - col_h ** 2)) + 10.0

    # ── Obstacle detection ────────────────────────────────────────────────────

    def obstacle_in_front(self):
        ds     = self.get_distances()
        ds_ok  = len(ds) >= 3 and min(ds[0], ds[2]) < 0.08
        lid_ok = self.get_lidar_front_min_dist(angle_range_deg=35) < 0.15
        return ds_ok or lid_ok

    def there_is_obstacle(self, map_target):
        return self.occ_map.cell_blocked(map_target)

    def robot_stuck(self, last_pos, stuck_distance=0.16):
        return np.linalg.norm(self.get_position() - last_pos) < stuck_distance

    # ── DWA planner ───────────────────────────────────────────────────────────

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
                for _ in range(5):           # 5-step horizon → ~160 ms look-ahead
                    cx += v * np.cos(ct) * dt
                    cy += v * np.sin(ct) * dt
                    ct += w * dt
                    pmx, pmy = self.convert_to_map_coordinates(cx, cy)
                    if self.there_is_obstacle([pmx, pmy]):
                        ok = False
                        break
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
                         DWA_COST_MAP_WEIGHT * cost_penalty)
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
            return True, False

        cur = self.get_position()
        if self.follow_target_last_position is not None:
            moved = np.linalg.norm(cur - self.follow_target_last_position)
            if moved < self.follow_target_position_threshold:
                self.follow_target_stuck_count += 1
                if self.follow_target_stuck_count >= self.follow_target_stuck_threshold:
                    self.follow_target_stuck_count   = 0
                    self.follow_target_last_position = None
                    self._phys_stuck_count           = 0
                    return False, True
            else:
                self.follow_target_stuck_count = 0
        self.follow_target_last_position = cur

        # Obstacle stop: cut motors the instant a wall is detected so the
        # wheel encoders stop counting.  This is the core drift fix — no wheel
        # spin means no encoder accumulation and no phantom map movement.
        if self.obstacle_in_front():
            self.stop_motor()                  # encoders freeze immediately
            self._phys_stuck_count += 1
            if self._phys_stuck_count >= 4:    # 4 × 32 ms = 128 ms of solid contact
                self._phys_stuck_count           = 0
                self.follow_target_last_position = None
                self.follow_target_stuck_count   = 0
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
            self.set_robot_velocity(lsp, rsp)
            return False, False

        v, w   = self.dwa_planner(np.array([wx, wy]))
        ls, rs = self.velocity_to_wheel_speeds(v, w)
        self.set_robot_velocity(ls, rs)
        return False, False

    # ── Recovery / direction adaptation ──────────────────────────────────────

    def avoid_obstacle(self):
        ds = self.get_distances()
        if len(ds) >= 4 and np.mean(ds) < 0.3:
            self.move_backward_milisecond(50)
            return
        attempts = 0
        while (len(ds) >= 3 and
               min(ds[0], ds[2]) < OBSTACLE_AVOID_THRESHOLD and
               attempts < OBSTACLE_AVOID_MAX_ATTEMPTS):
            ms = random.randint(TURN_DURATION_MIN, TURN_DURATION_MAX)
            if ds[0] < ds[2]:
                self.turn_right_milisecond(ms)
            else:
                self.turn_left_milisecond(ms)
            attempts += 1
            ds = self.get_distances()

    def unstick(self, turn_range=(400, 600)):
        self.set_robot_velocity(MOTOR_VELOCITY_BACKWARD, MOTOR_VELOCITY_BACKWARD)
        d = self.get_distances()
        self.step(500 if (len(d) >= 4 and d[1] > 0.25 and d[3] > 0.25) else 250)
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
        self.set_robot_velocity(MOTOR_VELOCITY_BACKWARD, MOTOR_VELOCITY_BACKWARD)
        d = self.get_distances()
        self.step(500 if (len(d) >= 4 and d[2] > 0.25 and d[3] > 0.25) else 250)
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

    # ── Alignment helpers ─────────────────────────────────────────────────────

    def align_to_red_wall(self):
        Kp, Kd    = ALIGN_RED_WALL_KP, ALIGN_RED_WALL_KD
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
            h, w, _ = hsv.shape
            is_mid, red_mask = utils.red_wall_centered(hsv)
            M = cv2.moments(red_mask)
            if M['m00'] > 0:
                cx    = int(M['m10'] / M['m00'])
                err   = cx - (w // 2)
                if abs(err) < ALIGN_RED_WALL_ERROR_THRESHOLD and is_mid:
                    self.stop_motor()
                    break
                spd = RED_WALL_ALIGNMENT_SPEED / self.wheel_radius
                self.set_robot_velocity(spd + Kp * err + Kd * (err - last_err),
                                        spd - Kp * err - Kd * (err - last_err))
                last_err = err
            else:
                self.stop_motor()
                break

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

    def align_to_column(self, color):
        Kp, Kd = ALIGN_COLUMN_KP, ALIGN_COLUMN_KD
        last_err = 0
        fwd = ALIGN_COLUMN_FORWARD_SPEED / self.wheel_radius
        while self.step(self.time_step) != -1:
            hsv = self.get_hsv_image()
            if hsv is None:
                self.stop_motor()
                break
            h, w, _ = hsv.shape
            mask = utils.extract_color_mask(hsv, color)
            M    = cv2.moments(mask)
            if M['m00'] > 0:
                cx  = int(M['m10'] / M['m00'])
                err = cx - (w // 2)
                if abs(err) < ALIGN_COLUMN_ERROR_THRESHOLD or np.sum(mask) / (h * w) > 0.7:
                    self.stop_motor()
                    break
                turn = Kp * err + Kd * (err - last_err)
                self.set_robot_velocity(fwd + turn, fwd - turn)
                last_err = err
                self.step(self.time_step)
            else:
                self.stop_motor()
                break

    def align_to_path(self, map_target, angle_threshold_deg=15,
                      clear_distance=0.6, back_distance=0.18):
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
        if self.get_min_front_distance() <= clear_distance and back_distance > 0:
            back_speed = 0.12
            dt    = TIME_STEP / 1000.0
            steps = max(1, int(back_distance / (back_speed * dt)))
            lw, rw = self.velocity_to_wheel_speeds(-back_speed, 0.0)
            self.set_robot_velocity(lw, rw)
            for _ in range(steps):
                if self.step(self.time_step) == -1:
                    break
            self.stop_motor()
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

    def mark_column(self, color):
        wp = self.position_ahead(0.2)
        mp = self.convert_to_map_coordinates(wp[0], wp[1])
        if color == 'blue':
            self.start_point = mp
        elif color == 'yellow':
            self.end_point = mp
        self.last_found_color = color
        self.last_found_point = mp

    def mark_on_map(self, distance_cm, color='blue'):
        if distance_cm is None:
            distance_cm = self.estimate_column_distance(color)
        if distance_cm is None:
            return
        if distance_cm < COLOR_DETECTION_DEPTH_THRESHOLD:
            if color == 'blue':
                self.start_point = tuple(self.get_map_position())
            elif color == 'yellow':
                self.end_point = tuple(self.get_map_position())

    def update_column_estimation(self, color, position):
        if color == 'blue':
            if self.blue_estimated_pos is None:
                self.blue_estimated_pos = position
            else:
                self.blue_estimated_pos = (0.3 * np.array(self.blue_estimated_pos) +
                                           0.7 * np.array(position))
        elif color == 'yellow':
            if self.yellow_estimated_pos is None:
                self.yellow_estimated_pos = position
            else:
                self.yellow_estimated_pos = (0.3 * np.array(self.yellow_estimated_pos) +
                                             0.7 * np.array(position))

    def _estimate_column_pos(self, color):
        dist = self.estimate_column_distance(color)
        if dist is None:
            return
        wp = self.position_ahead(dist / 100.0)
        mp = self.convert_to_map_coordinates(wp[0], wp[1])
        self.update_column_estimation(color, mp)
        if color == 'blue':
            self.blue_pos_update_count += 1
        else:
            self.yellow_pos_update_count += 1

    # ── Closure marking ───────────────────────────────────────────────────────

    def _mark_closure(self):
        now = time.time()
        if now - self.last_closure_time < self.closure_cooldown:
            return False
        try:
            ok = self.occ_map.stamp_closure(
                forward_m=CLOSURE_MARK_FORWARD,
                back_m=CLOSURE_MARK_BACKWARD,
                width_m=CLOSURE_MARK_WIDTH,
            )
            if ok:
                self.last_closure_time = now
            return bool(ok)
        except Exception as e:
            print(f'[warning] _mark_closure: {e}')
            return False

    # ── 360° scan ─────────────────────────────────────────────────────────────

    def scan_360(self):
        print('[Scan] 360° rotation...')
        self.set_robot_velocity(3.0, -3.0)
        steps_done  = 0
        target_steps = 8500 // self.time_step
        while steps_done < target_steps:
            if self.step(self.time_step) == -1:
                break
            steps_done += 1
            with self.detection_lock:
                sig = self.camera_detection_signal
            if sig is not None:
                if isinstance(sig, tuple) and sig[0] == 'column':
                    print(f'[Scan] Column {sig[1]} — stopping 360')
                    break
                elif sig in ('red_wall', 'green_carpet'):
                    print(f'[Scan] {sig} — stopping 360')
                    break
        self.stop_motor()

    # ── Frontier selection ────────────────────────────────────────────────────

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

            # Exploration-first: moderate info_gain exponent, stronger distance weight
            score = (info_gain ** 1.2) * math.log1p(d / 6.0)

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
        return (int(np.mean(cells[:, 0]) + random.randint(-3, 3)),
                int(np.mean(cells[:, 1]) + random.randint(-3, 3)))

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
            score    = (max(ig, 5) ** 1.2) * math.log1p(d / 6.0)
            if self._last_frontier_goal is not None:
                glx, gly = self._last_frontier_goal
                gdist = math.sqrt((cx - glx) ** 2 + (cy - gly) ** 2)
                score *= 1.0 + 3.0 * max(0.0, 1.0 - gdist / 35.0)
            if score > best_score:
                best_score, best_region = score, region
        if best_region is None:
            return None
        cells = np.array(best_region)
        return (int(np.mean(cells[:, 0]) + random.randint(-3, 3)),
                int(np.mean(cells[:, 1]) + random.randint(-3, 3)))

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
            w  = max(0.01, (ig ** 1.2) * math.log1p(d / 6.0))
            if self._last_frontier_goal is not None:
                glx, gly = self._last_frontier_goal
                gdist = math.sqrt((cx - glx) ** 2 + (cy - gly) ** 2)
                w *= 1.0 + 3.0 * max(0.0, 1.0 - gdist / 35.0)
            weights.append(w)
        chosen = random.choices(pool, weights=weights, k=1)[0]
        cells  = np.array(chosen)
        return (int(np.mean(cells[:, 0]) + random.randint(-3, 3)),
                int(np.mean(cells[:, 1]) + random.randint(-3, 3)))

    def _column_biased_target(self, max_jitter=8):
        candidates = []
        if self.yellow_estimated_pos is not None and self.end_point is None:
            candidates.append(('yellow', self.yellow_estimated_pos))
        if self.blue_estimated_pos is not None and self.start_point is None:
            candidates.append(('blue', self.blue_estimated_pos))
        if not candidates:
            return None
        color, pos = random.choice(candidates)
        print(f'[Frontier] Biasing toward {color.upper()}')
        h, w = self.grid_map.shape
        for _ in range(100):
            gx = int(pos[0] + random.randint(-max_jitter, max_jitter))
            gy = int(pos[1] + random.randint(-max_jitter, max_jitter))
            if 0 <= gx < w and 0 <= gy < h:
                if self.grid_map[gy, gx] in (FREESPACE, UNKNOWN):
                    return (gx, gy)
        return None

    def _nearby_free_cell(self, radius=40, max_tries=200):
        grid  = self.grid_map
        h, w  = grid.shape
        rx, ry = self.get_map_position()
        for _ in range(max_tries):
            x = int(rx + random.randint(-radius, radius))
            y = int(ry + random.randint(-radius, radius))
            if 0 <= x < w and 0 <= y < h and grid[y, x] == FREESPACE:
                if not self.occ_map.cell_blocked((x, y)):
                    return (x, y)
        return None

    # ── Background threads ────────────────────────────────────────────────────

    def start_camera_thread(self):
        if self.detection_thread and self.detection_thread.is_alive():
            return
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

    def _camera_loop(self):
        time.sleep(1.0)
        while self.camera_thread_running:
            time.sleep(0.1)
            with self.detection_lock:
                if self.camera_detection_signal is not None:
                    continue
                if self.there_is_red_wall():
                    self.camera_detection_signal = 'red_wall'
                    continue
                if self.detect_green():
                    self.camera_detection_signal = 'green_carpet'
                    continue
                if self.found_all_2_columns():
                    continue
                color = self.detect_column()
                if color:
                    if color == 'blue'   and self.start_point is not None:
                        continue
                    if color == 'yellow' and self.end_point   is not None:
                        continue
                    hsv  = self.get_hsv_image()
                    mask = utils.extract_color_mask(hsv, color) if hsv is not None else None
                    if mask is not None and self.column_close(mask):
                        self.center_column_in_view(color)
                        self.mark_column(color)
                        self.interrupt_path = True
                        self.turn_right_milisecond(600)
                    else:
                        self.camera_detection_signal = ('column', color)

    def _lidar_loop(self):
        depth_tick = 0
        while self.lidar_thread_running:
            try:
                if self.lidar is None:
                    time.sleep(0.5)
                    continue
                if not self.robot_on_ground():
                    time.sleep(0.1)
                    continue
                if not self.is_turning():
                    with self.lidar_lock:
                        self._refresh_map_lidar()
                        depth_tick += 1
                        if depth_tick % 3 == 0:
                            self._refresh_map_depth()
                        if depth_tick % 15 == 0:
                            self.occ_map.build_cost_map()
                time.sleep(0.05)
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

        if (count >= EXPLORATION_START_FRONTIER_AFTER and
                count % EXPLORATION_FRONTIER_SELECTION_FREQ == 0):
            regions = self.occ_map.compute_frontiers()

            if random.random() < 0.9:
                chosen  = self._column_biased_target()
                chasing = chosen is not None

            if chosen is None:
                if random.random() < 0.4:
                    chosen = self._score_frontier(regions, blacklist=bl_positions)
                    self.chosen_frontier_count += 1
                else:
                    chosen = self._random_frontier(regions)
                    self.chosen_frontier_count = 0

            if chosen:
                self._frontier_blacklist.append((chosen[0], chosen[1], count + 100))
                path = self.occ_map.frontier_path(self.get_map_position(), chosen)
                if not path:  # narrow corridor — frontier_path inflation may have blocked it
                    path = self.occ_map.astar_path(self.get_map_position(), chosen)
                if path:
                    ok = self.navigate_frontier(path)
                    if ok:
                        self._last_frontier_goal = chosen
                    if ok and chasing and not self.found_all_2_columns():
                        self.scan_360()

        return regions, chosen, path

    def navigate_frontier(self, path, replan_interval=20,
                          drop_on_red_wall=True, use_global_planner=False):
        if not path:
            return False

        goal         = path[-1]
        cur_path     = list(path)
        tick         = 0
        stuck_count  = 0
        replan_count = 0
        tidx         = 3
        MAX_STUCK    = 3
        with self.occ_map.vis_lock:
            self.occ_map.target_position = None
            self.occ_map.current_path    = cur_path

        while tidx < len(cur_path):
            target = cur_path[tidx]

            while self.step(self.time_step) != -1:
                tick += 1

                if self.interrupt_path:
                    self.stop_motor()
                    self.interrupt_path = False
                    return True

                sig = None
                with self.detection_lock:
                    if self.camera_detection_signal is not None:
                        sig = self.camera_detection_signal
                        self.camera_detection_signal = None

                if self.obstacle_in_front():
                    replan_count += 1
                    self.clear_obstacle()
                    self._refresh_map_lidar()
                    if replan_count >= 3:
                        return False
                    new = self.occ_map.frontier_path(self.get_map_position(), goal)
                    if not new:  # try less-inflated planner for narrow corridors
                        new = self.occ_map.astar_path(self.get_map_position(), goal)
                    if new and len(new) > 5:
                        cur_path = new
                        tidx = 0  # outer loop adds 3, so navigation starts at path[3]
                        break
                    return False

                if sig == 'red_wall':
                    self.stop_motor()
                    self.align_to_red_wall()
                    self.stop_motor()
                    for _ in range(10):
                        if not self.is_turning(): break
                        if self.step(self.time_step) == -1: break
                    self._mark_closure()
                    for _ in range(10):
                        if self.step(self.time_step) == -1: break
                    self.turn_right_milisecond(350)
                    if drop_on_red_wall:
                        self.occ_map.visited_frontiers.append(tuple(goal))
                        self.stop_motor()
                        return False

                if sig == 'green_carpet' or tick % 20 == 0:
                    self.stop_motor()
                    if self.mark_green_carpet_permanently(min_pixel_threshold=10000):
                        self.stop_motor()
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
                        self.center_column_in_view(color)
                        if self.get_column_center_pixels(color) >= 8:
                            self._estimate_column_pos(color)
                            if color == 'blue':
                                self.blue_prev_estimate_position  = cur_pos
                            else:
                                self.yellow_prev_estimate_position = cur_pos

                if tick % replan_interval == 0:
                    try:
                        new = self.occ_map.frontier_path(self.get_map_position(), goal)
                        if new and len(new) > 5:
                            cur_path = new
                            tidx = 0  # outer loop adds 3 → starts at path[3]
                            break
                    except Exception:
                        pass

                with self.occ_map.vis_lock:
                    rx, ry = self.get_map_position()
                    self.occ_map.robot_position  = (int(rx), int(ry))
                    self.occ_map.current_path    = cur_path
                    self.occ_map.target_position = target
                if tick % 8 == 0:
                    self.occ_map.refresh_viz()

                reached, is_stuck = self.advance_to_waypoint(target)
                if is_stuck:
                    stuck_count += 1
                    if stuck_count >= MAX_STUCK:
                        self.occ_map.visited_frontiers.append(tuple(goal))
                        self.stop_motor()
                        return False
                    self.stop_motor()
                    self._refresh_map_lidar()
                    self.unstick()
                    try:
                        fn  = (self.occ_map.astar_path if use_global_planner
                               else self.occ_map.frontier_path)
                        new = fn(self.get_map_position(), goal)
                        if new and len(new) > 5:
                            cur_path = list(new)
                            tidx = 0  # outer loop adds 3 → starts at path[3]
                            break
                        else:
                            self.occ_map.visited_frontiers.append(tuple(goal))
                            return False
                    except Exception:
                        return False
                if reached:
                    break

            tidx += 3

        self.stop_motor()
        with self.occ_map.vis_lock:
            self.occ_map.target_position = None
            self.occ_map.current_path    = None
        return True

    # ── Final path following ──────────────────────────────────────────────────

    def follow_final_path(self, path, debug_vis=False, replan_interval=60):
        if not path:
            print('[FinalPath] Empty path')
            return False

        if not self.camera_thread_running:
            self.start_camera_thread()
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
            return True

        def _plan():
            start = tuple(self.get_map_position())
            try:
                return self.occ_map.astar_path(start, goal)
            except Exception as e:
                print(f'[FinalPath] planner error: {e}')
                return None

        cur_path = _plan() or list(path)

        try:
            # Use a waypoint further ahead for a stable direction estimate;
            # path[1] is too close and noisy when the spline is dense.
            align_idx = min(10, len(cur_path) - 1)
            self.align_to_path(cur_path[align_idx])
        except Exception:
            pass

        def _build_waypoints(p):
            p = list(p)
            if len(p) <= 1:
                return [goal]
            stride = 1
            idxs   = list(range(stride, len(p), stride))
            if len(p) - 1 not in idxs:
                idxs.append(len(p) - 1)
            idxs = [i for i in idxs if 0 <= i < len(p)]
            return [tuple(p[i]) for i in idxs] or [goal]

        waypoints = _build_waypoints(cur_path)
        tick = 0
        i    = 0

        while i < len(waypoints):
            target = waypoints[i]
            while self.step(self.time_step) != -1:
                tick += 1

                sig = None
                with self.detection_lock:
                    if self.camera_detection_signal is not None:
                        sig = self.camera_detection_signal
                        self.camera_detection_signal = None

                if sig == 'red_wall':
                    self.stop_motor()
                    try:
                        self.align_to_red_wall()
                        self.stop_motor()
                    except Exception:
                        pass
                    for _ in range(10):
                        if not self.is_turning(): break
                        if self.step(self.time_step) == -1: break
                    try:
                        self._mark_closure()
                    except Exception:
                        pass
                    for _ in range(10):
                        if self.step(self.time_step) == -1: break
                    try:
                        self.turn_right_milisecond(350)
                    except Exception:
                        pass
                    for _ in range(40):
                        if self.step(self.time_step) == -1:
                            self.stop_motor()
                            return False
                    new = _plan()
                    if new and len(new) > 2:
                        cur_path  = list(new)
                        waypoints = _build_waypoints(cur_path)
                        i = -1  # outer loop adds 1 → starts at waypoints[0]
                        break
                    else:
                        self.stop_motor()
                        return False

                if isinstance(sig, tuple) and len(sig) >= 2 and sig[0] == 'column':
                    _, color = sig
                    try:
                        ratio = self.get_column_center_ratio(color)
                        d     = self.estimate_column_distance(color)
                        close = (d is not None and d < COLOR_DETECTION_DEPTH_THRESHOLD)
                        if ratio >= 0.20 or close:
                            self.stop_motor()
                            for _ in range(10):
                                if not self.is_turning(): break
                                if self.step(self.time_step) == -1: break
                            self.center_column_in_view(color)
                            dist = self.estimate_column_distance(color)
                            if dist is not None:
                                self.mark_on_map(dist, color=color)
                            else:
                                self.mark_column(color)
                        else:
                            self._estimate_column_pos(color)
                    except Exception:
                        pass

                if tick % 10 == 0:
                    try:
                        self.mark_green_carpet_permanently(min_pixel_threshold=10000)
                    except Exception:
                        pass

                front_dist = self.get_min_front_distance()
                if front_dist < 0.08:
                    self.stop_motor()
                    back_speed = 0.12
                    dt    = (TIME_STEP + 80) / 1000.0
                    steps = max(1, int(0.18 / (back_speed * dt)))
                    lw, rw = self.velocity_to_wheel_speeds(-back_speed, 0.0)
                    self.set_robot_velocity(lw, rw)
                    for _ in range(steps):
                        if self.step(self.time_step) == -1: break
                    self.stop_motor()
                    for _ in range(40):
                        if self.step(self.time_step) == -1: break
                    new = _plan()
                    if new and len(new) > 2:
                        cur_path  = list(new)
                        waypoints = _build_waypoints(cur_path)
                        i = -1  # outer loop adds 1 → starts at waypoints[0]
                        break
                    else:
                        self.stop_motor()
                        return False

                if replan_interval and tick % replan_interval == 0:
                    new = _plan()
                    if new and len(new) > 2:
                        cur_path  = list(new)
                        waypoints = _build_waypoints(cur_path)
                        i = -1  # outer loop adds 1 → starts at waypoints[0]
                        break

                with self.occ_map.vis_lock:
                    rx, ry = self.get_map_position()
                    self.occ_map.robot_position  = (int(rx), int(ry))
                    self.occ_map.current_path    = cur_path
                    self.occ_map.target_position = target
                if tick % 8 == 0:
                    self.occ_map.refresh_viz()

                reached, is_stuck = self.advance_to_waypoint(target)
                if is_stuck or (len(self.get_distances()) and
                                min(self.get_distances()) < 0.05):
                    self.stop_motor()
                    try:
                        self.unstick()
                    except Exception:
                        pass
                    for _ in range(40):
                        if self.step(self.time_step) == -1: break
                    new = _plan()
                    if new and len(new) > 2:
                        cur_path  = list(new)
                        waypoints = _build_waypoints(cur_path)
                        i = -1  # outer loop adds 1 → starts at waypoints[0]
                        break
                    else:
                        self.stop_motor()
                        return False

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

    # ── Main exploration loop ─────────────────────────────────────────────────

    def explore(self, debug=True):
        self.start_camera_thread()
        self.start_lidar_thread()

        if debug:
            self.occ_map.start_viz()

        active_path    = None
        active_goal    = None
        prev_grid      = self.occ_map.grid_map.copy()
        map_diff       = 1.0
        count          = 0

        time.sleep(0.2)
        self.scan_360()

        while self.step(self.time_step) != -1 and not self.found_all_2_columns():

            with self.detection_lock:
                sig = self.camera_detection_signal
                self.camera_detection_signal = None

            if sig == 'red_wall':
                self.stop_motor()
                self.align_to_red_wall()
                self.stop_motor()
                for _ in range(10):
                    if not self.is_turning(): break
                    if self.step(self.time_step) == -1: break
                self._mark_closure()
                for _ in range(5):
                    if self.step(self.time_step) == -1: break
                self.turn_right_milisecond(350)
                active_path = active_goal = None
                continue

            if sig == 'green_carpet':
                self.stop_motor()
                self.mark_green_carpet_permanently(min_pixel_threshold=10000)
                active_path = active_goal = None
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
                    self.center_column_in_view(color)
                    if self.get_column_center_pixels(color) >= 8:
                        self._estimate_column_pos(color)
                        if color == 'blue':
                            self.blue_prev_estimate_position  = cur_pos
                        else:
                            self.yellow_prev_estimate_position = cur_pos
                active_path = active_goal = None
                continue

            map_diff = utils.map_delta_ratio(prev_grid, self.occ_map.grid_map)
            _, chosen, path_to_chosen = self._update_frontier(count, map_diff)
            frontier_navigated = path_to_chosen is not None

            if path_to_chosen is None and random.random() < 0.2:
                fb = self._nearby_free_cell()
                if fb is not None:
                    chosen         = fb
                    path_to_chosen = self.occ_map.frontier_path(self.get_map_position(), chosen)

            # Only navigate if _update_frontier didn't already navigate to the frontier
            if not frontier_navigated and active_path is None and path_to_chosen:
                active_path = path_to_chosen
                active_goal = chosen

            if active_path is not None:
                self.navigate_frontier(active_path)
                active_path = active_goal = None

            prev_grid = self.occ_map.grid_map.copy()

            if debug:
                with self.occ_map.vis_lock:
                    rx, ry = self.get_map_position()
                    self.occ_map.robot_position  = (rx, ry)
                    self.occ_map.current_path    = active_path or path_to_chosen
                    self.occ_map.target_position = active_goal or chosen
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
                if count % 8 == 0:
                    self.occ_map.refresh_viz()

            count += 1

        self.stop_camera_thread()
        self.stop_lidar_thread()
        self.occ_map.stop_viz()
        with self.detection_lock:
            self.camera_detection_signal  = None
            self.green_carpet_active      = False
            self.last_green_carpet_points = []
            time.sleep(0.5)
        self.stop_motor()
        print('[Explore] Done.')

        if self.start_point is not None and self.end_point is not None:
            blue   = tuple(self.start_point)
            yellow = tuple(self.end_point)
            if self.last_found_color == 'yellow':
                start, end = yellow, blue
            else:
                start, end = blue, yellow
            return self.find_path(start, end)
        return []

    def find_path(self, start, end):
        return self.occ_map.astar_path(start, end)

    def find_path_for_frontier(self, start, end):
        return self.occ_map.frontier_path(start, end)

    # ── Green carpet ──────────────────────────────────────────────────────────

    def get_green_carpet_points(self):
        hsv_img = self.get_hsv_image()
        if hsv_img is None:
            return np.array([])
        green_mask = utils.extract_color_mask(hsv_img, 'green')
        kernel = cv2.getStructuringElement(
            cv2.MORPH_RECT,
            (GREEN_CARPET_DILATION_KERNEL_SIZE, GREEN_CARPET_DILATION_KERNEL_SIZE))
        green_mask = cv2.dilate(green_mask, kernel, iterations=GREEN_CARPET_DILATION_ITERATIONS)
        h_start = int(self.cam_height * 0.5)
        green_mask[:h_start, :] = 0
        v_px, u_px = np.where(green_mask == 255)
        if len(v_px) == 0:
            return np.array([])
        x_n = (u_px - self.cx) / self.fx
        y_n = (v_px - self.cy) / self.fy
        D   = self.camera_height_m / (y_n + 1e-6)
        ok  = (y_n > 0.001) & (D > 0.1) & (D < 4.0)
        D_v, x_nv = D[ok], x_n[ok]
        pts_local = np.stack([D_v + self.X_offset, -D_v * x_nv + self.Y_offset], axis=1)
        pts_world = self.transform_points_to_world(pts_local)
        return self.convert_to_map_coordinate_matrix(pts_world)

    def mark_green_carpet_permanently(self, min_pixel_threshold=10,
                                      new_area_threshold=0.5):
        self.green_carpet_active = True
        quick_pts = self.get_green_carpet_points()
        if quick_pts.shape[0] < min_pixel_threshold:
            self.green_carpet_active = False
            return False
        pts = quick_pts.astype(np.int32)
        H, W = self.grid_map.shape
        xi, yi = pts[:, 0], pts[:, 1]
        valid  = (xi >= 0) & (xi < W) & (yi >= 0) & (yi < H)
        xi, yi = xi[valid], yi[valid]
        if xi.size == 0:
            self.green_carpet_active = False
            return False
        green_protect = (self.grid_map == GREEN_CARPET)
        if np.sum(~green_protect[yi, xi]) / xi.size < new_area_threshold:
            self.green_carpet_active = False
            return False
        centroid   = np.array([float(np.mean(xi)), float(np.mean(yi))])
        min_d, closest_idx = float('inf'), -1
        for i, (ox, oy, _) in enumerate(self.green_carpet_patches):
            d = np.linalg.norm(centroid - np.array([ox, oy]))
            if d < min_d:
                min_d, closest_idx = d, i
        if closest_idx != -1 and min_d < self.green_carpet_proximity_threshold:
            if xi.size > self.green_carpet_patches[closest_idx][2] * 1.2:
                self.green_carpet_patches[closest_idx] = (centroid[0], centroid[1], int(xi.size))
            else:
                self.green_carpet_active = False
                return False
        for _ in range(15):
            if self.step(self.time_step) == -1:
                break
        self._back_from_green()
        self.stop_motor()
        time.sleep(1.0)
        cur_pts = self.get_green_carpet_points()
        if cur_pts.shape[0] < min_pixel_threshold:
            self.green_carpet_active = False
            return False
        pts2 = cur_pts.astype(np.int32)
        xi2, yi2 = pts2[:, 0], pts2[:, 1]
        valid2   = (xi2 >= 0) & (xi2 < W) & (yi2 >= 0) & (yi2 < H)
        xi2, yi2 = xi2[valid2], yi2[valid2]
        if xi2.size == 0:
            self.green_carpet_active = False
            return False
        try:
            self.grid_map[yi2, xi2] = int(GREEN_CARPET)
            if min_d >= self.green_carpet_proximity_threshold:
                self.green_carpet_patches.append((centroid[0], centroid[1], int(xi2.size)))
            self.last_green_mark_time = time.time()
            self.green_carpet_active  = False
            return True
        except Exception as e:
            print(f'[warning] green carpet mark failed: {e}')
            self.green_carpet_active = False
            return False

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
