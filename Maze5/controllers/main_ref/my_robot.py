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
        self.first_found_color = None
        self.last_floating_wall_orientation = None

        self.last_closure_time = 0.0
        self.closure_cooldown  = CLOSURE_MARK_COOLDOWN

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
        self._last_column_signal_time         = 0.0
        self._last_column_signal_color        = None
        self._last_detection_emit             = {}
        self._last_detection_handled          = {}
        self._camera_seen_counts              = {}
        self._column_focus_blocked_until      = {}

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

    def _scan_delta(self, a, b):
        if a is None or b is None:
            return None
        finite = np.isfinite(a) & np.isfinite(b)
        if not np.any(finite):
            return None
        return float(np.mean(np.abs(a[finite] - b[finite])))

    def _refresh_map_lidar(self):
        pts     = self.get_pointcloud_world_coordinates()
        map_pos = self.get_map_position()
        self.occ_map.process_scan(map_pos, pts)

    def _refresh_map_depth(self, depth_stride=3, max_depth=3.5):
        """Process depth-camera obstacles and plot floating walls.
        Behaviour inspired by Maze1: detect floating-wall points (require_ground_support=True)
        and fall back to general depth obstacles otherwise. Floating walls are stored
        in occ_map.floating_points for visualization (orientation classified).
        """
        if self.camera_depth is None:
            return
        # First try to get floating-wall points (require_ground_support=True)
        pts_local = self._depth_obstacle_points_local(pixel_stride=depth_stride,
                                                     max_depth=max_depth,
                                                     require_ground_support=True)
        if pts_local is None:
            return
        heading = self.get_heading('rad')
        rmap    = self.get_map_position()
        rx_m, ry_m = int(rmap[0]), int(rmap[1])
        R = np.array([[np.cos(heading), -np.sin(heading)],
                      [np.sin(heading),  np.cos(heading)]])

        # If floating points detected, process and mark them specially
        if pts_local.shape[0] > 0:
            pts_world = pts_local[:, :2] @ R.T + np.array([self._odom_x, self._odom_y])
            map_pts = self.convert_to_map_coordinate_matrix(pts_world)
            if map_pts.shape[0] == 0:
                return

            h, w = self.occ_map.grid_map.shape
            hit_mask = np.zeros((h, w), dtype=np.uint8)
            for mx, my in map_pts:
                mx_i, my_i = int(mx), int(my)
                if 0 <= mx_i < w and 0 <= my_i < h:
                    hit_mask[my_i, mx_i] = 1
            # Morphological cleanup for cleaner depth-wall mask shape (Problem 3)
            _dm_kernel = np.ones((3, 3), np.uint8)
            hit_mask = cv2.dilate(hit_mask, _dm_kernel, iterations=1)
            hit_mask = cv2.morphologyEx(hit_mask, cv2.MORPH_CLOSE, _dm_kernel)
            ys, xs = np.where(hit_mask > 0)
            map_pts = np.stack([xs, ys], axis=1).astype(np.int32) if len(xs) else np.empty((0, 2), dtype=np.int32)

            unique_set = set((int(mx), int(my)) for mx, my in map_pts
                             if 0 <= int(mx) < MAP_SIZE and 0 <= int(my) < MAP_SIZE)
            filtered_set = set()
            rejected_set = set()
            grid = self.occ_map.grid_map
            for mx, my in unique_set:
                # Do not promote an already mapped lidar wall to a floating wall.
                # This keeps normal walls behind/near a floating wall from being
                # re-painted by the depth overlay.
                y0 = max(0, my - 1); y1 = min(grid.shape[0], my + 2)
                x0 = max(0, mx - 1); x1 = min(grid.shape[1], mx + 2)
                near_lidar_wall = np.any(grid[y0:y1, x0:x1] == OBSTACLE)
                if grid[my, mx] not in (GREEN_CARPET, CLOSED, OBSTACLE) and not near_lidar_wall:
                    filtered_set.add((mx, my))
                else:
                    rejected_set.add((mx, my))
            if rejected_set:
                self.occ_map._depth_obstacle_cells.difference_update(rejected_set)
                for mx, my in rejected_set:
                    if grid[my, mx] == DEPTH_OBSTACLE:
                        grid[my, mx] = FREESPACE
            wall_like_set = self._wall_like_floating_cells(filtered_set)
            not_wall_like = filtered_set - wall_like_set
            if not_wall_like:
                self.occ_map._depth_obstacle_cells.difference_update(not_wall_like)
                for mx, my in not_wall_like:
                    if 0 <= mx < grid.shape[1] and 0 <= my < grid.shape[0] and grid[my, mx] == DEPTH_OBSTACLE:
                        grid[my, mx] = FREESPACE
                filtered_set = wall_like_set
            stale_behind = self._depth_cells_behind_current_surface(filtered_set, heading, pts_local)
            if stale_behind:
                self.occ_map._depth_obstacle_cells.difference_update(stale_behind)
                for mx, my in stale_behind:
                    if 0 <= mx < grid.shape[1] and 0 <= my < grid.shape[0] and grid[my, mx] == DEPTH_OBSTACLE:
                        grid[my, mx] = FREESPACE
            if not filtered_set:
                self.occ_map.floating_points = []
                self.occ_map.build_cost_map()
                return
            orient = self._classify_floating_wall(pts_local)
            if orient is None:
                orient = self.last_floating_wall_orientation
            elif len(filtered_set) >= DEPTH_OBSTACLE_MIN_WALL_CELLS:
                self.last_floating_wall_orientation = orient
            color_tag = orient if orient is not None else 'vertical'
            # Generate continuous wall segment with thickness (Problems 1, 2, 5)
            wall_cells = self._build_wall_segment_cells(filtered_set, orient, heading=heading)
            blocks_robot = self._floating_wall_blocks_robot(pts_local)
            if not blocks_robot:
                self.occ_map._depth_obstacle_cells.difference_update(wall_cells)
                for mx, my in wall_cells:
                    if (0 <= mx < MAP_SIZE and 0 <= my < MAP_SIZE and
                            self.occ_map.grid_map[my, mx] == DEPTH_OBSTACLE):
                        self.occ_map.grid_map[my, mx] = FREESPACE
                self.occ_map.floating_points = [
                    (mx, my, f'passable_{color_tag}') for mx, my in wall_cells
                ]
                self.occ_map.build_cost_map()
                return
            # Register in the persistent depth-obstacle set so rebuild_grid (triggered by
            # every lidar scan) cannot erase them — lidar passes under/over these walls.
            self.occ_map._depth_obstacle_cells.update(wall_cells)
            # Mark immediately in the live grid as well
            for mx, my in wall_cells:
                if (0 <= mx < MAP_SIZE and 0 <= my < MAP_SIZE and
                        self.occ_map.grid_map[my, mx] not in (GREEN_CARPET, CLOSED)):
                    self.occ_map.grid_map[my, mx] = DEPTH_OBSTACLE
            self.occ_map.build_cost_map()
            self.occ_map.floating_points = [(mx, my, color_tag) for mx, my in wall_cells]
            return

        # No floating-wall points: do not plot any depth obstacles.
        self.occ_map.floating_points = []
        return

    def _floating_wall_blocks_robot(self, pts_local):
        if pts_local is None or len(pts_local) == 0:
            return False
        heights = pts_local[:, 2].astype(np.float32)
        heights = heights[np.isfinite(heights)]
        if len(heights) == 0:
            return False
        low_ratio = float(np.mean(heights <= DEPTH_OBSTACLE_BLOCKING_HEIGHT))
        low_quantile = float(np.percentile(heights, 25))
        return (low_ratio >= DEPTH_OBSTACLE_BLOCKING_LOW_RATIO or
                low_quantile <= DEPTH_OBSTACLE_BLOCKING_HEIGHT)

    def _depth_cells_behind_current_surface(self, keep_cells, heading, pts_local):
        """Drop persistent depth cells that are farther along the same bearing."""
        depth_cells = getattr(self.occ_map, '_depth_obstacle_cells', set())
        if not depth_cells or pts_local.shape[0] == 0:
            return set()

        front_by_bearing = {}
        for fwd, lat, _ in pts_local:
            if fwd <= 0.05:
                continue
            key = int(round((lat / max(fwd, 1e-6)) * 40.0))
            prev = front_by_bearing.get(key)
            if prev is None or fwd < prev:
                front_by_bearing[key] = float(fwd)
        if not front_by_bearing:
            return set()

        cos_h, sin_h = math.cos(heading), math.sin(heading)
        stale = set()
        keep = set(keep_cells)
        for mx, my in list(depth_cells):
            cell = (int(mx), int(my))
            if cell in keep:
                continue
            wx, wy = self.convert_to_world_coordinates(cell[0], cell[1])
            dx, dy = wx - self._odom_x, wy - self._odom_y
            fwd = dx * cos_h + dy * sin_h
            lat = -dx * sin_h + dy * cos_h
            if not (0.05 < fwd < 3.5):
                continue
            bearing = lat / max(fwd, 1e-6)
            if abs(bearing) > 0.65:
                continue
            key = int(round(bearing * 40.0))
            front = min(
                (front_by_bearing[k] for k in (key - 1, key, key + 1)
                 if k in front_by_bearing),
                default=None
            )
            # Protect the full wall-thickness depth so cells on the far face of a
            # wall segment are not purged when the robot views the wall from the
            # opposite side.  The wall spans up to FLOATING_WALL_THICKNESS_CELLS
            # cells (≈ 0.133 m) perpendicular to the sensor; adding a 0.10 m margin
            # gives clearance for odometry drift and quantisation.
            _stale_tol = FLOATING_WALL_THICKNESS_CELLS * RESOLUTION + 0.10
            if front is not None and fwd > front + _stale_tol:
                stale.add(cell)
        return stale

    def _wall_like_floating_cells(self, cells):
        if not cells:
            return set()
        h, w = self.occ_map.grid_map.shape
        mask = np.zeros((h, w), dtype=np.uint8)
        for mx, my in cells:
            if 0 <= mx < w and 0 <= my < h:
                mask[my, mx] = 1
        n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
        keep = set()
        for label in range(1, n_labels):
            area = int(stats[label, cv2.CC_STAT_AREA])
            if area < DEPTH_OBSTACLE_MIN_WALL_CELLS:
                continue
            ys, xs = np.where(labels == label)
            pts = np.stack([xs, ys], axis=1).astype(np.float32)
            span_x = float(xs.max() - xs.min() + 1)
            span_y = float(ys.max() - ys.min() + 1)
            compact = max(span_x, span_y) <= 3.0
            if compact and area < DEPTH_OBSTACLE_MIN_WALL_CELLS + 2:
                continue
            if compact:
                keep.update((int(x), int(y)) for x, y in zip(xs, ys))
                continue
            if len(pts) >= 3:
                centered = pts - pts.mean(axis=0)
                _, s, _ = np.linalg.svd(centered, full_matrices=False)
                major = float(s[0]) if len(s) else 0.0
                minor = float(s[1]) if len(s) > 1 else 0.0
                elongated = major >= 3.0 and (minor < 1e-3 or major / max(minor, 1e-3) >= 1.8)
                if not elongated and max(span_x, span_y) < 5.0:
                    continue
            keep.update((int(x), int(y)) for x, y in zip(xs, ys))
        return keep

    def _reset_transient_depth_obstacles(self):
        cells = getattr(self.occ_map, '_depth_obstacle_cells', set())
        if cells:
            for mx, my in list(cells):
                if (0 <= mx < self.occ_map.grid_map.shape[1] and
                        0 <= my < self.occ_map.grid_map.shape[0] and
                        self.occ_map.grid_map[my, mx] == DEPTH_OBSTACLE):
                    self.occ_map.grid_map[my, mx] = FREESPACE
            cells.clear()
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

    def _peek_detection_signal(self):
        q = self.camera_detection_queue
        if q:
            return max(q, key=self._signal_priority)
        return self.camera_detection_signal

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

    def detect_green(self):
        hsv = self.get_bottom_half_hsv()
        if hsv is None:
            return None
        return cv2.countNonZero(utils.extract_color_mask(hsv, 'green')) > 50

    def detect_column(self):
        hsv = self.get_hsv_image()
        if hsv is None:
            return None
        min_pixels = 20
        if cv2.countNonZero(utils.extract_color_mask(hsv, 'yellow')) >= min_pixels:
            return 'yellow'
        if cv2.countNonZero(utils.extract_color_mask(hsv, 'blue')) >= min_pixels:
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

    def _depth_obstacle_points_local(self, pixel_stride=3, max_depth=3.5,
                                     require_ground_support=True):
        if self.camera_depth is None:
            return np.empty((0, 3), dtype=np.float32)
        try:
            depth_data = self.camera_depth.getRangeImage()
            if not depth_data:
                return np.empty((0, 3), dtype=np.float32)
        except Exception:
            return np.empty((0, 3), dtype=np.float32)
        try:
            w   = self.camera_depth.getWidth()
            h   = self.camera_depth.getHeight()
            fov = self.camera_depth.getFov()
        except Exception:
            return np.empty((0, 3), dtype=np.float32)
        fx = w / (2.0 * np.tan(fov / 2.0))
        fy = fx
        cx = w / 2.0
        cy = h / 2.0

        depth_arr = np.array(depth_data, dtype=np.float32).reshape(h, w)
        valid = np.isfinite(depth_arr) & (depth_arr > 0.05) & (depth_arr < max_depth)
        if not np.any(valid):
            return np.empty((0, 3), dtype=np.float32)

        rows = np.arange(0, h, pixel_stride)
        cols = np.arange(0, w, pixel_stride)
        vv, uu = np.meshgrid(rows, cols, indexing='ij')
        d = depth_arr[vv, uu]
        ok = valid[vv, uu]
        if not np.any(ok):
            return np.empty((0, 3), dtype=np.float32)

        uu = uu[ok].astype(np.int32)
        vv = vv[ok].astype(np.int32)
        d  = d[ok].astype(np.float32)
        uu_f = uu.astype(np.float32)
        vv_f = vv.astype(np.float32)
        x_n = (uu_f - cx) / fx
        y_n = (vv_f - cy) / fy

        forward = d + self.X_offset
        lateral = -d * x_n + self.Y_offset
        height  = self.camera_height_m - d * y_n
        obstacle_height = ((height >= DEPTH_OBSTACLE_MIN_HEIGHT) &
                           (height <= DEPTH_OBSTACLE_MAX_HEIGHT))
        if not np.any(obstacle_height):
            return np.empty((0, 3), dtype=np.float32)

        if require_ground_support:
            floating_collision_height = (
                (height >= DEPTH_OBSTACLE_FLOATING_MIN_HEIGHT) &
                (height <= DEPTH_OBSTACLE_FLOATING_MAX_HEIGHT)
            )
            if not np.any(floating_collision_height):
                return np.empty((0, 3), dtype=np.float32)

            # Follow the wall surface downward in each image column to verify it
            # does NOT connect to the ground. If it reaches ground height, it's
            # a normal wall and should be handled by lidar only.
            support_ok = np.zeros(len(height), dtype=bool)
            for i in np.where(floating_collision_height)[0]:
                u_col = int(uu[i])
                v_row = int(vv[i])
                d_ref = float(d[i])
                for sv in range(v_row + 1, h):
                    sd = depth_arr[sv, u_col]
                    if not (np.isfinite(sd) and 0.05 < sd < max_depth):
                        break  # surface ended → floating
                    if abs(sd - d_ref) > 0.15:
                        break  # different surface below → floating
                    y_n2 = (float(sv) - cy) / fy
                    h2 = self.camera_height_m - sd * y_n2
                    if h2 <= DEPTH_OBSTACLE_SUPPORT_MAX_HEIGHT:
                        support_ok[i] = True
                        break

            floating_mask = floating_collision_height & ~support_ok
            if not np.any(floating_mask):
                return np.empty((0, 3), dtype=np.float32)
            nearest_by_col = np.full(w, np.inf, dtype=np.float32)
            for u_col, depth_val in zip(uu[floating_collision_height], d[floating_collision_height]):
                if depth_val < nearest_by_col[u_col]:
                    nearest_by_col[u_col] = depth_val
            nearest_near_col = np.full(w, np.inf, dtype=np.float32)
            col_radius = max(1, int(pixel_stride) * 2)
            for u_col in np.unique(uu[floating_collision_height]):
                lo = max(0, int(u_col) - col_radius)
                hi = min(w, int(u_col) + col_radius + 1)
                nearest_near_col[u_col] = np.min(nearest_by_col[lo:hi])
            occluded_by_front_surface = d > (nearest_near_col[uu] + 0.12)
            floating_mask &= ~occluded_by_front_surface
            if not np.any(floating_mask):
                return np.empty((0, 3), dtype=np.float32)
            return np.stack(
                [forward[floating_mask],
                 lateral[floating_mask],
                 height[floating_mask]],
                axis=1
            ).astype(np.float32)

        return np.stack(
            [forward[obstacle_height],
             lateral[obstacle_height],
             height[obstacle_height]],
            axis=1
        ).astype(np.float32)

    def _column_depth_position_local(self, color):
        if self.camera_depth is None:
            return None
        hsv_img = self.get_hsv_image()
        if hsv_img is None:
            return None
        try:
            raw_depth = self.camera_depth.getRangeImage()
            w = self.camera_depth.getWidth()
            h = self.camera_depth.getHeight()
            fov = self.camera_depth.getFov()
        except Exception:
            return None
        if not raw_depth:
            return None
        mask = utils.extract_color_mask(hsv_img, color)
        if mask is None or not np.any(mask):
            return None
        depth = np.array(raw_depth, dtype=np.float32).reshape(h, w)
        valid = (mask != 0) & np.isfinite(depth) & (depth > 0.05) & (depth < 3.5)
        if not np.any(valid):
            return None

        vv, uu = np.where(valid)
        d = depth[vv, uu]
        keep = d <= np.percentile(d, 35)
        if not np.any(keep):
            return None
        uu = uu[keep].astype(np.float32)
        vv = vv[keep].astype(np.float32)
        d = d[keep]

        fx = w / (2.0 * np.tan(fov / 2.0))
        cx = w / 2.0
        cy = h / 2.0
        u = float(np.median(uu))
        v = float(np.median(vv))
        depth_m = float(np.median(d))
        x_n = (u - cx) / fx
        y_n = (v - cy) / fx
        forward = depth_m + self.X_offset
        lateral = -depth_m * x_n + self.Y_offset
        height = self.camera_height_m - depth_m * y_n
        return np.array([forward, lateral, height], dtype=np.float32)

    def _classify_floating_wall(self, pts):
        """Classify a set of floating-wall points as 'horizontal' (front-back, parallel)
        or 'vertical' (left-right, perpendicular) relative to the robot frame.
        Returns 'horizontal', 'vertical' or None if ambiguous.
        """
        try:
            if pts is None or len(pts) < 3:
                return None
            xy = pts[:, :2].astype(np.float64)  # forward, lateral
            mean = xy.mean(axis=0)
            X = xy - mean
            cov = X.T @ X
            w, v = np.linalg.eigh(cov)
            principal = v[:, np.argmax(w)]
            angle = math.atan2(principal[1], principal[0])  # radians
            # Normalize to [0, pi/2]
            ang = abs(angle)
            if ang > math.pi/2:
                ang = abs(ang - math.pi)
            # User-defined: horizontal = front-back (parallel to robot heading)
            if abs(ang - 0.0) < 0.35:
                return 'horizontal'
            if abs(ang - (math.pi / 2.0)) < 0.35:
                return 'vertical'
            return None
        except Exception:
            return None

    def _build_wall_segment_cells(self, filtered_set, orient, heading=None):
        """Build a full wall segment with thickness from sparse depth-wall cells.

        Uses PCA to find the principal axis of the point cloud, projects all
        points onto it to locate the segment endpoints, draws the wall with
        cv2.line at FLOATING_WALL_THICKNESS_CELLS width, and returns all map
        cells that belong to the resulting segment.
        """
        if not filtered_set:
            return set()
        h, w = self.occ_map.grid_map.shape
        pts = np.array(list(filtered_set), dtype=np.float32)
        if len(pts) < 2:
            result = set()
            half_t = FLOATING_WALL_THICKNESS_CELLS // 2
            for mx, my in filtered_set:
                for dy in range(-half_t, half_t + 1):
                    for dx in range(-half_t, half_t + 1):
                        nx, ny = mx + dx, my + dy
                        if 0 <= nx < w and 0 <= ny < h:
                            result.add((nx, ny))
            return result
        # PCA: find principal direction of the wall
        mean = pts.mean(axis=0)
        centered = pts - mean
        _, _, Vt = np.linalg.svd(centered, full_matrices=False)
        principal = Vt[0]  # unit vector along wall's longest axis
        projections = centered @ principal
        observed_len = float(projections.max() - projections.min())
        if observed_len < DEPTH_OBSTACLE_SIDE_VIEW_MIN_LENGTH_CELLS:
            principal = self._fallback_wall_axis_from_orientation(orient, heading)
            if principal is None:
                robot = np.array(self.get_map_position(), dtype=np.float32)
                bearing = mean - robot
                norm = float(np.linalg.norm(bearing))
                if norm > 1e-6:
                    bearing /= norm
                    principal = np.array([-bearing[1], bearing[0]], dtype=np.float32)
                else:
                    principal = Vt[0]
            principal = np.array(principal, dtype=np.float32)
            principal /= max(float(np.linalg.norm(principal)), 1e-6)
            centered = pts - mean
        # Project points onto principal axis and find segment endpoints
        projections = centered @ principal
        pmin = float(projections.min())
        pmax = float(projections.max())
        if (pmax - pmin) < DEPTH_OBSTACLE_SIDE_VIEW_MIN_LENGTH_CELLS:
            half_len = DEPTH_OBSTACLE_SIDE_VIEW_MIN_LENGTH_CELLS / 2.0
            pmin, pmax = -half_len, half_len
        pt1 = mean + pmin * principal
        pt2 = mean + pmax * principal
        x1, y1 = int(round(float(pt1[0]))), int(round(float(pt1[1])))
        x2, y2 = int(round(float(pt2[0]))), int(round(float(pt2[1])))
        # Draw line with thickness equal to FLOATING_WALL_THICKNESS_CELLS;
        # this expands the segment perpendicular to the wall direction.
        line_mask = np.zeros((h, w), dtype=np.uint8)
        cv2.line(line_mask, (x1, y1), (x2, y2), 1,
                 thickness=FLOATING_WALL_THICKNESS_CELLS)
        ys, xs = np.where(line_mask > 0)
        result = set((int(x), int(y)) for x, y in zip(xs, ys))
        return result if result else set(filtered_set)

    def _fallback_wall_axis_from_orientation(self, orient, heading):
        if orient is None or heading is None:
            return None
        if orient == 'horizontal':
            angle = heading
        elif orient == 'vertical':
            angle = heading + math.pi / 2.0
        else:
            return None
        return np.array([math.cos(angle), math.sin(angle)], dtype=np.float32)

    def _depth_wall_blocks_column(self, column_local):
        wall_pts = self._depth_obstacle_points_local(pixel_stride=2, max_depth=3.5)
        if wall_pts.shape[0] == 0:
            return False
        forward, lateral, _ = column_local
        if forward <= 0.05:
            return False
        bearing = lateral / max(forward, 1e-6)
        wall_bearing = wall_pts[:, 1] / np.maximum(wall_pts[:, 0], 1e-6)
        same_bearing = np.abs(wall_bearing - bearing) < 0.10
        before_column = wall_pts[:, 0] < forward - 0.08
        blocking_height = (
            (wall_pts[:, 2] >= DEPTH_OBSTACLE_FLOATING_MIN_HEIGHT) &
            (wall_pts[:, 2] <= DEPTH_OBSTACLE_FLOATING_MAX_HEIGHT)
        )
        mask = same_bearing & before_column & blocking_height
        if not np.any(mask):
            self.last_floating_wall_orientation = None
            return False
        # Classify orientation from the subset of blocking points and store it
        orientation = self._classify_floating_wall(wall_pts[mask])
        self.last_floating_wall_orientation = orientation
        return True

    def _column_is_front_facing(self, color):
        return self.get_column_center_ratio(color) >= 0.20

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
        half = self.occ_map.map_size // 2
        for r in range(1, 30):
            for dy in range(-r, r + 1):
                for dx in range(-r, r + 1):
                    if abs(dx) != r and abs(dy) != r:
                        continue  # walk only the ring perimeter
                    nx, ny = mx + dx, my + dy
                    if 0 <= nx < W and 0 <= ny < H:
                        if self.occ_map.grid_map[ny, nx] == FREESPACE:
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
        est = self.blue_estimated_pos if color == 'blue' else self.yellow_estimated_pos
        refined_from_depth = False
        if est is not None:
            if not self._column_commit_ready(color):
                return False
            mp = (int(round(float(est[0]))), int(round(float(est[1]))))
        else:
            column_local = self._column_depth_position_local(color)
            if column_local is None:
                return False
            if not self._column_is_front_facing(color) and self._depth_wall_blocks_column(column_local):
                return False
            if column_local[0] * 100.0 > COLUMN_COMMIT_MAX_DISTANCE_CM:
                return False
            heading = self.get_heading('rad')
            R = np.array([[np.cos(heading), -np.sin(heading)],
                          [np.sin(heading),  np.cos(heading)]])
            wp = column_local[:2] @ R.T + np.array([self._odom_x, self._odom_y])
            mp = self.convert_to_map_coordinates(float(wp[0]), float(wp[1]))
            refined_from_depth = True
        if refined_from_depth:
            mp = self._refine_column_map_position(mp)
        self._set_committed_column(color, mp)
        return True

    def mark_on_map(self, distance_cm, color='blue'):
        column_local = self._column_depth_position_local(color)
        if column_local is None:
            return False
        if not self._column_is_front_facing(color) and self._depth_wall_blocks_column(column_local):
            return False
        if column_local[0] * 100.0 < COLOR_DETECTION_DEPTH_THRESHOLD:
            heading = self.get_heading('rad')
            R = np.array([[np.cos(heading), -np.sin(heading)],
                          [np.sin(heading),  np.cos(heading)]])
            wp = column_local[:2] @ R.T + np.array([self._odom_x, self._odom_y])
            mp = self.convert_to_map_coordinates(float(wp[0]), float(wp[1]))
            mp = self._refine_column_map_position(mp)
            # Skip LOS check when front-facing: the pillar's own OBSTACLE cells
            # would block the ray even though the camera has direct line of sight.
            if not self._column_is_front_facing(color) and not self._column_line_of_sight_clear(mp):
                return False
            self._set_committed_column(color, mp)
            return True
        return False

    def _set_committed_column(self, color, mp):
        self._commit_column_cell(color, mp)
        if color == 'blue':
            self.start_point = mp
        elif color == 'yellow':
            self.end_point = mp
        if self.first_found_color is None:
            self.first_found_color = color
        if self._column_focus_color == color:
            self._column_focus_color = None
            self._column_focus_target = None
        self.last_found_color = color
        self.last_found_point = mp

    def _column_is_committed(self, color):
        return self.start_point is not None if color == 'blue' else self.end_point is not None

    def _column_commit_ready(self, color):
        est = self.blue_estimated_pos if color == 'blue' else self.yellow_estimated_pos
        if est is None:
            return False
        updates = self.blue_pos_update_count if color == 'blue' else self.yellow_pos_update_count
        if updates < COLUMN_COMMIT_MIN_ESTIMATES:
            return False
        if self.get_map_distance(est) <= COLUMN_COMMIT_MAX_MAP_DISTANCE:
            return True
        dist = self.estimate_column_distance(color)
        return dist is not None and dist <= COLUMN_COMMIT_MAX_DISTANCE_CM

    def _commit_column_from_estimate(self, color, force=False):
        if not force and not self._column_commit_ready(color):
            return False
        est = self.blue_estimated_pos if color == 'blue' else self.yellow_estimated_pos
        if est is None:
            return False
        mp = (int(round(float(est[0]))), int(round(float(est[1]))))
        self._set_committed_column(color, mp)
        return True

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

    def _refine_column_map_position(self, mp):
        """Move the visual surface hit toward the pillar center.

        RGB/depth returns the visible front surface, which biases markers toward
        the robot. Prefer the center of the nearby LiDAR obstacle blob; otherwise
        offset one column radius along the robot-to-hit bearing.
        """
        mx = int(round(float(mp[0]))); my = int(round(float(mp[1])))
        h, w = self.grid_map.shape
        if not (0 <= mx < w and 0 <= my < h):
            return (mx, my)

        robot = np.array(self.get_map_position(), dtype=np.float32)
        hit = np.array([mx, my], dtype=np.float32)
        ray = hit - robot
        ray_norm = float(np.linalg.norm(ray))
        if ray_norm > 1e-6:
            ray_unit = ray / ray_norm
            offset_cells = COLUMN_CENTER_OFFSET_M / max(RESOLUTION, 1e-6)
            expected = hit + ray_unit * offset_cells
        else:
            expected = hit

        radius = int(COLUMN_OBSTACLE_SNAP_RADIUS_CELLS)
        cx = int(round(float(expected[0]))); cy = int(round(float(expected[1])))
        x0 = max(0, min(mx, cx) - radius)
        x1 = min(w, max(mx, cx) + radius + 1)
        y0 = max(0, min(my, cy) - radius)
        y1 = min(h, max(my, cy) + radius + 1)
        roi = (self.grid_map[y0:y1, x0:x1] == OBSTACLE).astype(np.uint8)
        if roi.size:
            n_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(roi, connectivity=8)
            best = None
            best_score = float('inf')
            for label in range(1, n_labels):
                area = int(stats[label, cv2.CC_STAT_AREA])
                if area < 3 or area > 120:
                    continue
                centroid = np.array([
                    float(centroids[label][0]) + x0,
                    float(centroids[label][1]) + y0,
                ], dtype=np.float32)
                dist_expected = float(np.linalg.norm(centroid - expected))
                dist_hit = float(np.linalg.norm(centroid - hit))
                if dist_hit > radius + 3:
                    continue
                score = dist_expected + 0.25 * dist_hit
                if score < best_score:
                    best_score = score
                    best = centroid
            if best is not None:
                return (int(round(float(best[0]))), int(round(float(best[1]))))

        ex = int(round(float(expected[0]))); ey = int(round(float(expected[1])))
        return (max(0, min(w - 1, ex)), max(0, min(h - 1, ey)))

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

    def _commit_column_cell(self, color, mp, radius=1):
        """Mark a small circular marker on the map for the detected column and
        reduce log-odds so the cell is not treated as an obstacle. Also add to
        occ_map.column_points for reliable visualization overlay.
        """
        cell_value = BLUE_COLUMN if color == 'blue' else YELLOW_COLUMN
        mx = int(round(float(mp[0]))); my = int(round(float(mp[1])))
        h, w = self.grid_map.shape
        # Stamp a small circular region in the log-odds to lower obstacle belief
        for dy in range(-radius, radius + 1):
            for dx in range(-radius, radius + 1):
                if dx * dx + dy * dy > radius * radius:
                    continue
                nx, ny = mx + dx, my + dy
                if 0 <= nx < w and 0 <= ny < h:
                    # Lower log-odds so rebuild_grid won't mark it as obstacle
                    try:
                        self.occ_map.log_odds[ny, nx] = min(self.occ_map.log_odds[ny, nx], -2.5)
                    except Exception:
                        pass
                    # Keep grid_map as freespace here; visualization uses occ_map.column_points
                    self.grid_map[ny, nx] = FREESPACE
        # Ensure column_points contains this marker for renderer overlays (avoid duplicates)
        color_rgb = (0, 255, 255) if color == 'blue' else (255, 255, 0)
        pts = list(self.occ_map.column_points) if self.occ_map.column_points else []
        if not any(px == mx and py == my for px, py, _ in pts):
            pts.append((mx, my, color_rgb))
            self.occ_map.column_points = pts

    def _estimate_column_pos(self, color):
        column_local = self._column_depth_position_local(color)
        if column_local is None:
            return False
        # If a floating wall blocks the column, record orientation and abort
        if (not self._column_is_front_facing(color) and
                self._depth_wall_blocks_column(column_local)):
            return False
        heading = self.get_heading('rad')
        R = np.array([[np.cos(heading), -np.sin(heading)],
                      [np.sin(heading),  np.cos(heading)]])
        wp = column_local[:2] @ R.T + np.array([self._odom_x, self._odom_y])
        mp = self.convert_to_map_coordinates(float(wp[0]), float(wp[1]))
        mp = self._refine_column_map_position(mp)
        # Skip LOS check when the column is centered in the camera frame: the
        # camera is looking directly at it so the ray has no real obstruction,
        # but the pillar's own LiDAR-mapped OBSTACLE cells would incorrectly
        # block the ray and prevent the estimate from ever being committed.
        front_facing = self._column_is_front_facing(color)
        if not front_facing and not self._column_line_of_sight_clear(mp):
            return False
        self.update_column_estimation(color, mp)
        est = self.blue_estimated_pos if color == 'blue' else self.yellow_estimated_pos
        if est is not None:
            self._commit_column_cell(color, (int(round(float(est[0]))),
                                            int(round(float(est[1])))))
            if ((color == 'blue' and self.start_point is None) or
                    (color == 'yellow' and self.end_point is None)):
                self._column_focus_color = color
                self._column_focus_target = self._column_approach_target(est)
        if color == 'blue':
            self.blue_pos_update_count += 1
        else:
            self.yellow_pos_update_count += 1
        return True

    # ── Dynamic path validation and replanning helpers ─────────────────────────

    def _path_blocked(self, path, lookahead=12, cost_thresh=0.65):
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

    def _path_blocked_from_pose(self, path, lookahead=12, cost_thresh=0.65):
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

    def _path_usable_from_pose(self, path, min_len=3, lookahead=14):
        if not path or len(path) < min_len:
            return False
        grid = self.occ_map.grid_map
        for px, py in path:
            x, y = int(px), int(py)
            if 0 <= x < grid.shape[1] and 0 <= y < grid.shape[0] and grid[y, x] == GREEN_CARPET:
                return False
        return not self._path_blocked_from_pose(path, lookahead=lookahead, cost_thresh=0.82)

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
            self.occ_map.build_cost_map()
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
            dist = self.estimate_column_distance(color)
            if dist is not None and dist < COLOR_DETECTION_DEPTH_THRESHOLD:
                self.mark_on_map(dist, color=color)
                committed = self._column_is_committed(color)
            if estimated and not committed:
                self.mark_column(color)
                committed = self._column_is_committed(color)
            if estimated and not committed:
                committed = self._commit_column_from_estimate(color)
            if committed:
                self._suspend_column_focus(color, seconds=2.0)
                return True
            if not estimated:
                self._suspend_column_focus(color, seconds=8.0)
        return False

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
        if self._column_focus_target is not None:
            color = self._column_focus_color or 'column'
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
        now = time.time()
        yellow_blocked = self._column_focus_blocked_until.get('yellow', 0.0) > now
        blue_blocked = self._column_focus_blocked_until.get('blue', 0.0) > now
        if self.yellow_estimated_pos is not None and self.end_point is None and not yellow_blocked:
            candidates.append(('yellow', self.yellow_estimated_pos))
        if self.blue_estimated_pos is not None and self.start_point is None and not blue_blocked:
            candidates.append(('blue', self.blue_estimated_pos))
        if not candidates:
            return None
        color, pos = random.choice(candidates)
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
            time.sleep(0.1)
            now = time.time()
            # Run detections outside the lock so each is independent and
            # the lock is not held during slow image-processing calls.  This ensures
            # green carpet and column detection are never skipped because the other
            # detector is slow or triggered a skip condition.
            green = self.detect_green()
            color = None
            if not self.found_all_2_columns():
                color = self.detect_column()
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
        tidx         = 3
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
                                self._commit_column_from_estimate(color)
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
        last_replan_tick = -max(1, replan_interval or 1)

        self.start_realtime_planner(goal)
        try:
          while i < len(waypoints):
            target = waypoints[i]
            while self.step(self.time_step) != -1:
                tick += 1

                rt = self.poll_realtime_planner()
                if rt:
                    cur_path  = rt
                    waypoints = _build_waypoints(cur_path)
                    i = -1  # outer loop adds 1 → starts at waypoints[0]
                    with self.occ_map.vis_lock:
                        self.occ_map.current_path = cur_path
                    print('[RT-Planner] Path swapped in follow_final_path')
                    break

                sig = None
                with self.detection_lock:
                    sig = self._pop_detection_signal()

                if sig == 'green_carpet':
                    self.mark_green_carpet_permanently(min_pixel_threshold=1500, force=True)
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
                            found = self._estimate_column_pos(color)
                            if not found:
                                self.center_column_in_view(color)
                                found = self._estimate_column_pos(color)
                            dist = self.estimate_column_distance(color)
                            if dist is not None:
                                self.mark_on_map(dist, color=color)
                            if found and not self._column_is_committed(color):
                                self.mark_column(color)
                            if found and not self._column_is_committed(color):
                                self._commit_column_from_estimate(color)
                        else:
                            self._estimate_column_pos(color)
                    except Exception:
                        pass

                if tick % 5 == 0:
                    try:
                        if self.detect_green():
                            self.stop_motor()
                            self.mark_green_carpet_permanently(min_pixel_threshold=2500, force=True)
                            return False
                    except Exception:
                        pass

                if self.obstacle_in_front():
                    new = self._recover_and_replan(goal, prefer_frontier=False, min_len=3)
                    if new and len(new) > 2:
                        cur_path = list(new)
                        waypoints = _build_waypoints(cur_path)
                        i = -1
                        last_replan_tick = tick
                        break
                    self.stop_motor()
                    return False

                try:
                    blocked = self._path_blocked_from_pose(cur_path, lookahead=24)
                    interval_hit = bool(replan_interval) and (tick - last_replan_tick >= replan_interval)
                    if blocked or interval_hit:
                        new = self._attempt_replan(goal)
                        if self._path_usable_from_pose(new, min_len=3):
                            cur_path = list(new)
                            waypoints = _build_waypoints(cur_path)
                            i = -1  # outer loop adds 1 → starts at waypoints[0]
                            last_replan_tick = tick
                            break
                        new = _plan()
                        if self._path_usable_from_pose(new, min_len=3):
                            cur_path = list(new)
                            waypoints = _build_waypoints(cur_path)
                            i = -1
                            last_replan_tick = tick
                            break
                        new = self._recover_and_replan(goal, prefer_frontier=False, min_len=3)
                        if new and len(new) > 2:
                            cur_path = list(new)
                            waypoints = _build_waypoints(cur_path)
                            i = -1
                            last_replan_tick = tick
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
                if is_stuck or (len(self.get_distances()) and
                                min(self.get_distances()) < 0.05):
                    self.stop_motor()
                    new = self._recover_and_replan(goal, prefer_frontier=False, min_len=3)
                    if new and len(new) > 2:
                        cur_path  = list(new)
                        waypoints = _build_waypoints(cur_path)
                        i = -1  # outer loop adds 1 → starts at waypoints[0]
                        break
                    print(f'[FinalPath] Recovery: skipping waypoint {i}, continuing')
                    i += 1
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
        finally:
            self.stop_realtime_planner()

        self.stop_motor()
        success = self.get_map_distance(goal) < PATH_FOLLOWING_TARGET_REACH_DISTANCE
        print(f'[FinalPath] Finished, success={success}')
        return success

    # ── Main exploration loop ─────────────────────────────────────────────────

    def explore(self, debug=True):
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
        self._handle_scan_signal(self.scan_360())

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
                        self._commit_column_from_estimate(color)
                        focus_path = self._column_focus_path()
                        if focus_path and len(focus_path) > 2:
                            self.navigate_frontier(focus_path)
                continue

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
                if count % 8 == 0:
                    self.occ_map.refresh_viz()

            count += 1

        self.stop_camera_thread()
        self.stop_lidar_thread()
        self.occ_map.stop_viz()
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
            end    = blue if self.first_found_color == 'yellow' else yellow
            robot_start = tuple(self.get_map_position())
            return self.find_path(robot_start, end) or []
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
            cur_pts = self.get_green_carpet_points()
            if cur_pts.shape[0] < min_pixel_threshold:
                return False
            pts2 = cur_pts.astype(np.int32)
            xi2, yi2 = pts2[:, 0], pts2[:, 1]
            valid2   = (xi2 >= 0) & (xi2 < W) & (yi2 >= 0) & (yi2 < H)
            xi2, yi2 = xi2[valid2], yi2[valid2]
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
