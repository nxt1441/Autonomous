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
        self.last_floating_wall_orientation = None

        self.last_closure_time = 0.0
        self.closure_cooldown  = CLOSURE_MARK_COOLDOWN

        self.steps_since_turning  = DEPTH_OBSTACLE_POST_TURN_STABILIZATION_STEPS
        self.is_currently_turning = False

        self.detection_thread        = None
        self.camera_thread_running   = False
        self.camera_detection_signal = None
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
        self._rt_planner_lock    = threading.Lock()
        self._last_rt_swap_time  = 0.0

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
        self.green_carpet_lock     = threading.Lock()

        self.counter_obstacle_recoveries = 0

        self._odom_x      = INITIAL_X
        self._odom_y      = INITIAL_Y
        self._odom_theta  = INITIAL_THETA
        self._odom_prev_left  = 0.0
        self._odom_prev_right = 0.0
        self._odom_initialized = False
        self._last_odom_step_distance = 0.0
        self._last_odom_step_heading_delta = 0.0
        # Temporal counters for confirming observed depth-only obstacles
        self._depth_cell_counters = {}

    # ── step() override ───────────────────────────────────────────────────────

    def step(self, duration_ms=None):
        if duration_ms is None:
            duration_ms = self.time_step
        result = super().step(int(duration_ms))
        if result != -1:
            self._tick_odometry()
            turning_now = self.is_turning()
            if turning_now:
                self.is_currently_turning = True
                self.steps_since_turning = 0
            else:
                if self.is_currently_turning:
                    self.is_currently_turning = False
                    self.steps_since_turning = 0
                else:
                    self.steps_since_turning += 1
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

        ds         = (dl + dr) / 2.0
        prev_theta = self._odom_theta
        self._odom_theta = heading
        dtheta     = utils.angle_wrap(self._odom_theta, prev_theta)

        self._last_odom_step_distance      = float(abs(ds))
        self._last_odom_step_heading_delta = float(abs(dtheta))

        # Skip sub-millimetre ticks to prevent noise accumulation when stationary.
        if abs(ds) < 1e-4:
            return

        # Exact arc formula for differential drive.
        # compass supplies absolute heading so there is no heading drift;
        # this formula is strictly more accurate than the mid-theta linear
        # approximation, especially during turns.
        if abs(dtheta) < 1e-6:
            self._odom_x += ds * math.cos(self._odom_theta)
            self._odom_y += ds * math.sin(self._odom_theta)
        else:
            R = ds / dtheta
            self._odom_x += R * (math.sin(self._odom_theta) - math.sin(prev_theta))
            self._odom_y += R * (-math.cos(self._odom_theta) + math.cos(prev_theta))
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
        mx = half + int(np.rint(self._odom_x / RESOLUTION))
        my = half - int(np.rint(self._odom_y / RESOLUTION))
        return np.array([mx, my])

    def get_map_distance(self, map_target):
        return np.linalg.norm(self.get_map_position() - np.array(map_target))

    def convert_to_map_coordinates(self, x, y):
        half = self.occ_map.map_size // 2
        mx   = half + int(np.rint(x / RESOLUTION))
        my   = half - int(np.rint(y / RESOLUTION))
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

    def _refresh_map_sensors(self):
        if self.lidar is None or self.is_turning() or not self.robot_on_ground():
            return
        with self.lidar_lock:
            self._refresh_map_lidar()
            self._refresh_map_depth()
            self.occ_map.build_cost_map()

    def _refresh_map_depth(self, depth_stride=3, max_depth=5.0):
        """Process depth-camera obstacles and plot floating walls.
        Behaviour inspired by Maze1: detect floating-wall points (require_ground_support=True)
        and fall back to general depth obstacles otherwise. Floating walls are stored
        in occ_map.floating_points for visualization (orientation classified).
        """
        if self.camera_depth is None:
            return
        self._stamp_floor_supported_depth_planes(depth_stride=depth_stride,
                                                 max_depth=max_depth)
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
            cell_heights = {}
            for (mx, my), pt in zip(map_pts, pts_local):
                mx_i, my_i = int(mx), int(my)
                if 0 <= mx_i < w and 0 <= my_i < h:
                    cell_heights.setdefault((mx_i, my_i), []).append(float(pt[2]))

            unique_set = set(cell_heights.keys())
            filtered_set = set()
            grid = self.occ_map.grid_map
            self._clear_depth_cells_near_lidar_walls(radius=2)
            for mx, my in unique_set:
                if (grid[my, mx] not in (GREEN_CARPET, CLOSED, OBSTACLE) and
                        not self._cell_near_lidar_wall(mx, my, radius=2)):
                    filtered_set.add((mx, my))
            if not filtered_set:
                self.occ_map.floating_points = []
                return

            orient = self._classify_floating_wall(pts_local)
            if orient is not None:
                self.last_floating_wall_orientation = orient
            else:
                orient = self.last_floating_wall_orientation

            wall_cells = set()
            for cluster in self._cluster_floating_cells(filtered_set):
                if not self._floating_cells_wall_like(cluster):
                    continue
                cluster_heights = []
                for cell in cluster:
                    cluster_heights.extend(cell_heights.get(cell, ()))
                cluster_pts = np.array([[0.0, 0.0, h] for h in cluster_heights], dtype=np.float32)
                if self._floating_wall_is_passable(cluster_pts):
                    self._clear_depth_cells_near_cells(cluster, radius=5)
                    continue
                wall_cells.update(self._rasterize_floating_wall_cells(cluster, orient,
                                                                      robot_cell=(rx_m, ry_m)))

            if not wall_cells:
                self.occ_map.floating_points = []
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
            color_tag = orient if orient is not None else 'vertical'
            self.occ_map.floating_points = [(mx, my, color_tag) for mx, my in wall_cells]
            return

        # No floating-wall points: do not plot any depth obstacles.
        self.occ_map.floating_points = []
        return

    def _stamp_floor_supported_depth_planes(self, depth_stride=3, max_depth=5.0):
        """Stamp low floor-supported planes that a horizontal LiDAR scan can miss."""
        if self.camera_depth is None:
            return False
        old_floor_cells = set(getattr(self.occ_map, '_floor_depth_obstacle_cells', set()))
        try:
            depth_data = self.camera_depth.getRangeImage()
            if not depth_data:
                return False
            w = self.camera_depth.getWidth()
            h = self.camera_depth.getHeight()
            fov = self.camera_depth.getFov()
        except Exception:
            return False

        depth_arr = np.array(depth_data, dtype=np.float32).reshape(h, w)
        valid = np.isfinite(depth_arr) & (depth_arr > 0.05) & (depth_arr < max_depth)
        if not np.any(valid):
            return False

        stride = max(1, int(depth_stride))
        rows = np.arange(0, h, stride)
        cols = np.arange(0, w, stride)
        vv, uu = np.meshgrid(rows, cols, indexing='ij')
        d = depth_arr[vv, uu]
        ok = valid[vv, uu]
        if not np.any(ok):
            return False

        fx = w / (2.0 * np.tan(fov / 2.0))
        fy = fx
        cx = w / 2.0
        cy = h / 2.0

        uu_i = uu[ok].astype(np.int32)
        vv_i = vv[ok].astype(np.int32)
        uu = uu_i.astype(np.float32)
        vv = vv_i.astype(np.float32)
        d = d[ok].astype(np.float32)
        x_n = (uu - cx) / fx
        y_n = (vv - cy) / fy

        forward = d + self.X_offset
        lateral = -d * x_n + self.Y_offset
        height = self.camera_height_m - d * y_n

        low_plane = (
            (height >= DEPTH_FLOOR_PLANE_MIN_HEIGHT) &
            (height <= DEPTH_FLOOR_PLANE_MAX_HEIGHT)
        )
        if np.any(low_plane):
            vertical_wall = np.zeros(len(height), dtype=bool)
            for i in np.where(low_plane)[0]:
                u_col = int(uu_i[i])
                d_ref = float(d[i])
                tall_same_surface_pixels = 0
                for sv in range(int(vv_i[i]) - 1, -1, -1):
                    sd = depth_arr[sv, u_col]
                    if not (np.isfinite(sd) and 0.05 < sd < max_depth):
                        break
                    if abs(float(sd) - d_ref) > DEPTH_FLOOR_PLANE_SURFACE_DEPTH_TOLERANCE:
                        break
                    y_n2 = (float(sv) - cy) / fy
                    h2 = self.camera_height_m - float(sd) * y_n2
                    if h2 >= DEPTH_FLOOR_PLANE_VERTICAL_REJECT_HEIGHT:
                        tall_same_surface_pixels += 1
                    if tall_same_surface_pixels >= DEPTH_FLOOR_PLANE_VERTICAL_REJECT_PIXELS:
                        vertical_wall[i] = True
                        break
            low_plane &= ~vertical_wall
        if not np.any(low_plane):
            return False

        pts_local = np.stack(
            [forward[low_plane], lateral[low_plane]],
            axis=1
        ).astype(np.float32)
        heading = self.get_heading('rad')
        R = np.array([[np.cos(heading), -np.sin(heading)],
                      [np.sin(heading),  np.cos(heading)]])
        pts_world = pts_local @ R.T + np.array([self._odom_x, self._odom_y])
        map_pts = self.convert_to_map_coordinate_matrix(pts_world)
        if map_pts.shape[0] == 0:
            return False

        grid = self.occ_map.grid_map
        h_map, w_map = grid.shape
        hit_mask = np.zeros((h_map, w_map), dtype=np.uint8)
        for mx, my in map_pts:
            mx_i, my_i = int(mx), int(my)
            if not (0 <= mx_i < w_map and 0 <= my_i < h_map):
                continue
            if grid[my_i, mx_i] in (GREEN_CARPET, CLOSED, OBSTACLE):
                continue
            if self._cell_near_lidar_wall(
                    mx_i, my_i, radius=DEPTH_FLOOR_PLANE_LIDAR_CLEAR_RADIUS):
                continue
            hit_mask[my_i, mx_i] = 1

        if int(np.count_nonzero(hit_mask)) < DEPTH_FLOOR_PLANE_MIN_CELLS:
            return False

        kernel_size = max(1, int(DEPTH_FLOOR_PLANE_CLOSE_KERNEL))
        if kernel_size > 1:
            kernel = np.ones((kernel_size, kernel_size), dtype=np.uint8)
            hit_mask = cv2.morphologyEx(hit_mask, cv2.MORPH_CLOSE, kernel)

        stamped = self._floor_plane_components_from_mask(hit_mask)

        if len(stamped) < DEPTH_FLOOR_PLANE_MIN_CELLS:
            return False

        filtered = set()
        for mx, my in stamped:
            if grid[my, mx] in (GREEN_CARPET, CLOSED, OBSTACLE):
                continue
            if self._cell_near_lidar_wall(
                    mx, my, radius=DEPTH_FLOOR_PLANE_LIDAR_CLEAR_RADIUS):
                continue
            filtered.add((mx, my))
        stamped = filtered
        if len(stamped) < DEPTH_FLOOR_PLANE_MIN_CELLS:
            return False

        stamped = self._confirm_floor_depth_cells(old_floor_cells, stamped)
        if len(stamped) < DEPTH_FLOOR_PLANE_MIN_CELLS:
            return False

        self._merge_floor_depth_cells(old_floor_cells, stamped)
        return True

    def _floor_plane_components_from_mask(self, hit_mask):
        n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
            hit_mask.astype(np.uint8), connectivity=8
        )
        stamped = set()
        for label in range(1, n_labels):
            area = int(stats[label, cv2.CC_STAT_AREA])
            if area < DEPTH_FLOOR_PLANE_MIN_COMPONENT_CELLS:
                continue
            x = int(stats[label, cv2.CC_STAT_LEFT])
            y = int(stats[label, cv2.CC_STAT_TOP])
            width = int(stats[label, cv2.CC_STAT_WIDTH])
            height = int(stats[label, cv2.CC_STAT_HEIGHT])
            minor_span = min(width, height)
            aspect = max(width, height) / max(float(minor_span), 1.0)
            if minor_span < DEPTH_FLOOR_PLANE_MIN_MINOR_SPAN:
                continue
            if aspect > DEPTH_FLOOR_PLANE_MAX_ASPECT_RATIO:
                continue
            ys, xs = np.where(labels == label)
            stamped.update((int(mx), int(my)) for mx, my in zip(xs, ys))
        return stamped

    def _confirm_floor_depth_cells(self, old_cells, candidate_cells):
        counters = getattr(self, '_floor_depth_cell_counters', {})
        old_cells = set(old_cells)
        candidate_cells = set(candidate_cells)
        if len(candidate_cells) >= DEPTH_FLOOR_PLANE_IMMEDIATE_CELLS:
            for cell in candidate_cells:
                counters[cell] = int(DEPTH_FLOOR_PLANE_CONFIRMATION_COUNT)
            self._floor_depth_cell_counters = counters
            return candidate_cells
        confirmed = set()
        near_radius = int(DEPTH_FLOOR_PLANE_CONFIRM_NEAR_EXISTING_RADIUS)
        near_sq = near_radius * near_radius
        old_arr = (np.array(list(old_cells), dtype=np.int32)
                   if old_cells else np.empty((0, 2), dtype=np.int32))

        for cell in candidate_cells:
            counters[cell] = min(
                int(counters.get(cell, 0)) + 1,
                int(DEPTH_FLOOR_PLANE_CONFIRMATION_COUNT)
            )
            near_existing = False
            if old_arr.size:
                delta = old_arr - np.array(cell, dtype=np.int32)
                near_existing = bool(np.any(np.sum(delta * delta, axis=1) <= near_sq))
            if (cell in old_cells or near_existing or
                    counters[cell] >= DEPTH_FLOOR_PLANE_CONFIRMATION_COUNT):
                confirmed.add(cell)

        for cell in list(counters.keys()):
            if cell not in candidate_cells:
                counters[cell] = int(counters[cell]) - 1
                if counters[cell] <= 0:
                    counters.pop(cell, None)

        self._floor_depth_cell_counters = counters
        return confirmed

    def _merge_floor_depth_cells(self, old_cells, new_cells):
        depth_cells = getattr(self.occ_map, '_depth_obstacle_cells', set())
        grid = self.occ_map.grid_map
        keep_cells = set(old_cells) | set(new_cells)
        remove = set()
        for mx, my in keep_cells:
            if not (0 <= mx < grid.shape[1] and 0 <= my < grid.shape[0]):
                remove.add((mx, my))
                continue
            if grid[my, mx] in (GREEN_CARPET, CLOSED, OBSTACLE):
                remove.add((mx, my))
                continue
        keep_cells -= remove

        for mx, my in set(old_cells) - keep_cells:
            depth_cells.discard((mx, my))
            if (0 <= mx < grid.shape[1] and 0 <= my < grid.shape[0] and
                    grid[my, mx] == DEPTH_OBSTACLE):
                grid[my, mx] = FREESPACE
        self.occ_map._floor_depth_obstacle_cells = set(keep_cells)
        self.occ_map._depth_obstacle_cells.update(keep_cells)
        for mx, my in keep_cells:
            if grid[my, mx] not in (GREEN_CARPET, CLOSED, OBSTACLE):
                grid[my, mx] = DEPTH_OBSTACLE
        self.occ_map.build_cost_map()

    def _cell_near_lidar_wall(self, mx, my, radius=3):
        grid = self.occ_map.grid_map
        y0 = max(0, int(my) - radius)
        y1 = min(grid.shape[0], int(my) + radius + 1)
        x0 = max(0, int(mx) - radius)
        x1 = min(grid.shape[1], int(mx) + radius + 1)
        return bool(np.any(grid[y0:y1, x0:x1] == OBSTACLE))

    def _clear_depth_cells_near_lidar_walls(self, radius=3):
        cells = getattr(self.occ_map, '_depth_obstacle_cells', set())
        if not cells:
            return
        floor_cells = getattr(self.occ_map, '_floor_depth_obstacle_cells', set())
        remove = {cell for cell in cells
                  if cell not in floor_cells and
                  self._cell_near_lidar_wall(cell[0], cell[1], radius=radius)}
        if not remove:
            return
        cells.difference_update(remove)
        grid = self.occ_map.grid_map
        for mx, my in remove:
            if 0 <= mx < grid.shape[1] and 0 <= my < grid.shape[0] and grid[my, mx] == DEPTH_OBSTACLE:
                grid[my, mx] = FREESPACE

    def _clear_depth_cells_near_cells(self, reference_cells, radius=3):
        cells = getattr(self.occ_map, '_depth_obstacle_cells', set())
        if not cells or not reference_cells:
            return
        floor_cells = getattr(self.occ_map, '_floor_depth_obstacle_cells', set())
        refs = np.array(list(reference_cells), dtype=np.int32)
        remove = set()
        radius_sq = int(radius) * int(radius)
        for cell in cells:
            if cell in floor_cells:
                continue
            delta = refs - np.array(cell, dtype=np.int32)
            if np.any(np.sum(delta * delta, axis=1) <= radius_sq):
                remove.add(cell)
        if not remove:
            return
        cells.difference_update(remove)
        grid = self.occ_map.grid_map
        for mx, my in remove:
            if 0 <= mx < grid.shape[1] and 0 <= my < grid.shape[0] and grid[my, mx] == DEPTH_OBSTACLE:
                grid[my, mx] = FREESPACE

    def _floating_cells_wall_like(self, cells):
        if len(cells) < 2:
            return False
        pts = np.array(list(cells), dtype=np.float32)
        if len(cells) == 2:
            return float(np.linalg.norm(pts[1] - pts[0])) >= 2.0
        centered = pts - pts.mean(axis=0)
        try:
            _, s, _ = np.linalg.svd(centered, full_matrices=False)
        except Exception:
            return False
        major = float(s[0]) if len(s) else 0.0
        minor = float(s[1]) if len(s) > 1 else 0.0
        span_x = float(np.max(pts[:, 0]) - np.min(pts[:, 0]) + 1.0)
        span_y = float(np.max(pts[:, 1]) - np.min(pts[:, 1]) + 1.0)
        if max(span_x, span_y) < 3.0:
            return False
        return major >= 1.5 and major / max(minor, 1e-3) >= 1.4

    def _cluster_floating_cells(self, cells, link_radius=5):
        cells = list(cells)
        if not cells:
            return []
        pts = np.array(cells, dtype=np.int32)
        remaining = set(range(len(cells)))
        clusters = []
        link_sq = int(link_radius) * int(link_radius)
        while remaining:
            seed = remaining.pop()
            queue = [seed]
            cluster_idx = {seed}
            while queue:
                cur = queue.pop()
                remaining_list = list(remaining)
                if not remaining_list:
                    continue
                delta = pts[remaining_list] - pts[cur]
                close_positions = np.where(np.sum(delta * delta, axis=1) <= link_sq)[0]
                close = [remaining_list[int(i)] for i in close_positions]
                for idx in close:
                    remaining.remove(idx)
                    cluster_idx.add(idx)
                    queue.append(idx)
            clusters.append({cells[i] for i in cluster_idx})
        return clusters

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

    def _depth_obstacle_points_local(self, pixel_stride=3, max_depth=5.0,
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
        # diagnostics: record counts so we can inspect why detection fails
        try:
            self._last_depth_diag = {
                'valid_pixels': int(np.count_nonzero(valid)),
                'h': int(h), 'w': int(w), 'pixel_stride': int(pixel_stride)
            }
        except Exception:
            pass
        if not np.any(valid):
            return np.empty((0, 3), dtype=np.float32)

        rows = np.arange(0, h, pixel_stride)
        cols = np.arange(0, w, pixel_stride)
        vv, uu = np.meshgrid(rows, cols, indexing='ij')
        d = depth_arr[vv, uu]
        ok = valid[vv, uu]
        try:
            if hasattr(self, '_last_depth_diag'):
                self._last_depth_diag['sampled_candidates'] = int(np.count_nonzero(ok))
        except Exception:
            pass
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

            # Pre-compute minimum valid depth per column to detect back-wall points.
            # A point whose depth exceeds the column minimum by >0.20 m is behind a
            # closer obstacle (the floating wall itself) and must not be treated as
            # a floating wall candidate.
            valid_depth_full = np.where(
                np.isfinite(depth_arr) & (depth_arr > 0.05) & (depth_arr < max_depth),
                depth_arr, np.inf
            )
            min_depth_per_col = np.min(valid_depth_full, axis=0)

            support_ok = np.zeros(len(height), dtype=bool)
            for i in np.where(floating_collision_height)[0]:
                u_col = int(uu[i])
                v_row = int(vv[i])
                d_ref = float(d[i])
                # Skip only obvious background behind a closer surface. A lower
                # threshold drops the rear/side edge of floating planes, so the
                # map never gets updated from the opposite approach angle.
                if d_ref - min_depth_per_col[u_col] > 0.30:
                    support_ok[i] = True  # treat as supported so it is excluded
                    continue
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

    def _floating_wall_is_passable(self, pts):
        """Ignore depth walls that sit above the robot's usable clearance.

        These are visible in depth, but the robot can drive underneath them, so
        they should not be stamped into the occupancy grid as blocking cells.
        """
        if pts is None or len(pts) == 0:
            return True
        heights = pts[:, 2].astype(np.float32)
        heights = heights[np.isfinite(heights)]
        if len(heights) == 0:
            return True

        robot_clearance_height = self.camera_height_m + 0.005
        min_h = float(np.min(heights))
        lower_edge = float(np.percentile(heights, 10))
        median_h = float(np.median(heights))
        height_span = float(np.max(heights) - np.min(heights))

        # A floating wall only blocks if its visible lower edge is inside the
        # robot clearance. Using the camera-height clearance avoids treating the
        # side face of an overhead wall as a blocking wall.
        if lower_edge <= robot_clearance_height:
            return False
        if height_span >= 0.16 and min_h <= robot_clearance_height + 0.02 and median_h <= 0.32:
            return False
        return True

    def _floating_cluster_stats(self, cells):
        pts = np.array(list(cells), dtype=np.float32)
        if pts.shape[0] < 2:
            return pts, np.array([1.0, 0.0], dtype=np.float32), 0.0, 0.0
        centered = pts - pts.mean(axis=0)
        try:
            _, s, vh = np.linalg.svd(centered, full_matrices=False)
            axis = vh[0]
            if not np.all(np.isfinite(axis)) or np.linalg.norm(axis) < 1e-6:
                axis = np.array([1.0, 0.0], dtype=np.float32)
        except Exception:
            s = np.array([0.0, 0.0], dtype=np.float32)
            axis = np.array([1.0, 0.0], dtype=np.float32)
        major = float(s[0]) if len(s) else 0.0
        minor = float(s[1]) if len(s) > 1 else 0.0
        return pts, axis.astype(np.float32), major, minor

    def _front_edge_cells(self, cells, robot_cell):
        pts, axis, major, minor = self._floating_cluster_stats(cells)
        if pts.shape[0] < 4 or robot_cell is None:
            return set(cells)
        # Broad depth clusters are horizontal planes/top faces. For those, draw
        # only the edge facing the robot instead of filling/drawing through the
        # visible surface.
        if minor < 1.8 or major / max(minor, 1e-3) > 4.0:
            return set(cells)
        center = pts.mean(axis=0)
        view = center - np.array(robot_cell, dtype=np.float32)
        norm = float(np.linalg.norm(view))
        if norm < 1e-6:
            return set(cells)
        view /= norm
        depth = pts @ view
        near = float(np.min(depth))
        edge_pts = pts[depth <= near + 1.5]
        if edge_pts.shape[0] < 2:
            return set(cells)
        return {(int(round(float(x))), int(round(float(y)))) for x, y in edge_pts}

    def _rasterize_floating_wall_cells(self, cells, orientation=None, robot_cell=None):
        if not cells:
            return set()
        cells = self._front_edge_cells(cells, robot_cell)
        pts = np.array(list(cells), dtype=np.int32)
        if pts.shape[0] == 1:
            return {(int(pts[0, 0]), int(pts[0, 1]))}

        pts_f, axis, _, _ = self._floating_cluster_stats(cells)
        centered = pts_f - pts_f.mean(axis=0)

        projection = centered @ axis
        order = np.argsort(projection)
        pts = pts[order]
        projection = projection[order]

        raster = set()
        max_gap = max(4, int(DEPTH_OBSTACLE_BRIDGE_GAP_CELLS))

        # Mark directly observed cells and only bridge tiny gaps between adjacent
        # observed cells. No fitted centerline or endpoint extension is used here.
        for x, y in pts:
            raster.add((int(x), int(y)))
        for i, (p0, p1) in enumerate(zip(pts[:-1], pts[1:])):
            if projection[i + 1] - projection[i] > max_gap:
                continue
            if float(np.linalg.norm(p1.astype(np.float32) - p0.astype(np.float32))) > max_gap:
                continue
            for x, y in utils.ray_cells((int(p0[0]), int(p0[1])),
                                        (int(p1[0]), int(p1[1]))):
                raster.add((x, y))

        # Thicken the trace a little so the wall reads as a wall in the map.
        radius = max(0, int(DEPTH_OBSTACLE_STAMP_RADIUS))
        if radius == 0:
            return raster
        expanded = set()
        for x, y in raster:
            for dy in range(-radius, radius + 1):
                for dx in range(-radius, radius + 1):
                    if dx * dx + dy * dy > radius * radius:
                        continue
                    expanded.add((x + dx, y + dy))
        return expanded

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
        fy = fx
        cx = w / 2.0
        cy = h / 2.0
        u = float(np.median(uu))
        v = float(np.median(vv))
        depth_m = float(np.median(d))
        x_n = (u - cx) / fx
        y_n = (v - cy) / fy
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

    def _depth_wall_blocks_column(self, column_local):
        wall_pts = self._depth_obstacle_points_local(pixel_stride=2, max_depth=5.0)
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
            (wall_pts[:, 2] <= DEPTH_OBSTACLE_FLOATING_MAX_HEIGHT) &
            (wall_pts[:, 2] <= self.camera_height_m + 0.02)
        )
        mask = same_bearing & before_column & blocking_height
        if not np.any(mask):
            self.last_floating_wall_orientation = None
            return False
        # Classify orientation from the subset of blocking points and store it
        orientation = self._classify_floating_wall(wall_pts[mask])
        self.last_floating_wall_orientation = orientation
        return True

    # ── Obstacle detection ────────────────────────────────────────────────────

    def obstacle_in_front(self):
        ds     = self.get_distances()
        ds_ok  = len(ds) >= 3 and min(ds[0], ds[2]) < 0.08
        lid_ok = self.get_lidar_front_min_dist(angle_range_deg=35) < 0.15
        return ds_ok or lid_ok

    def there_is_obstacle(self, map_target):
        return self.occ_map.cell_blocked(map_target)

    def footprint_has_obstacle(self, mx, my, radius=3):
        grid = self.occ_map.grid_map
        for dy in range(-radius, radius + 1):
            for dx in range(-radius, radius + 1):
                if dx * dx + dy * dy > radius * radius:
                    continue
                x, y = int(mx) + dx, int(my) + dy
                if 0 <= x < grid.shape[1] and 0 <= y < grid.shape[0]:
                    if self.occ_map.cell_blocked((x, y)):
                        return True
        return False

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
                    if self.footprint_has_obstacle(pmx, pmy, radius=3):
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
        est = self.blue_estimated_pos if color == 'blue' else self.yellow_estimated_pos
        if est is not None:
            mp = (int(round(float(est[0]))), int(round(float(est[1]))))
        else:
            column_local = self._column_depth_position_local(color)
            if column_local is None or self._depth_wall_blocks_column(column_local):
                return
            heading = self.get_heading('rad')
            R = np.array([[np.cos(heading), -np.sin(heading)],
                          [np.sin(heading),  np.cos(heading)]])
            wp = column_local[:2] @ R.T + np.array([self._odom_x, self._odom_y])
            mp = self.convert_to_map_coordinates(float(wp[0]), float(wp[1]))
        self._commit_column_cell(color, mp)
        if color == 'blue':
            self.start_point = mp
        elif color == 'yellow':
            self.end_point = mp
        self.last_found_color = color
        self.last_found_point = mp

    def mark_on_map(self, distance_cm, color='blue'):
        column_local = self._column_depth_position_local(color)
        if column_local is None or self._depth_wall_blocks_column(column_local):
            return
        if column_local[0] * 100.0 < COLOR_DETECTION_DEPTH_THRESHOLD:
            heading = self.get_heading('rad')
            R = np.array([[np.cos(heading), -np.sin(heading)],
                          [np.sin(heading),  np.cos(heading)]])
            wp = column_local[:2] @ R.T + np.array([self._odom_x, self._odom_y])
            mp = self.convert_to_map_coordinates(float(wp[0]), float(wp[1]))
            if not self._column_line_of_sight_clear(mp):
                return
            self._commit_column_cell(color, mp)
            if color == 'blue':
                self.start_point = mp
            elif color == 'yellow':
                self.end_point = mp

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
                if grid[ry, rx] in (OBSTACLE, DEPTH_OBSTACLE, CLOSED, GREEN_CARPET):
                    return False
        return True

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
        if self._depth_wall_blocks_column(column_local):
            return False
        heading = self.get_heading('rad')
        R = np.array([[np.cos(heading), -np.sin(heading)],
                      [np.sin(heading),  np.cos(heading)]])
        wp = column_local[:2] @ R.T + np.array([self._odom_x, self._odom_y])
        mp = self.convert_to_map_coordinates(float(wp[0]), float(wp[1]))
        if not self._column_line_of_sight_clear(mp):
            return False
        self.update_column_estimation(color, mp)
        est = self.blue_estimated_pos if color == 'blue' else self.yellow_estimated_pos
        if est is not None:
            self._commit_column_cell(color, (int(round(float(est[0]))),
                                            int(round(float(est[1])))))
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
        # 2) Try planner with relaxed cost_map (reduce wall penalty)
        try:
            if self.occ_map.cost_map is not None:
                scaled = (self.occ_map.cost_map * 0.2).astype(np.float32)
            else:
                scaled = None
            new = self.occ_map.astar_path(start, goal, inflation_levels=[1, 0], cost_map_override=scaled)
            if new:
                return new
        except Exception:
            pass
        # 3) Try ignoring cost map and using minimal inflation (allow narrow passages)
        try:
            new = self.occ_map.astar_path(start, goal, inflation_levels=[0], cost_map_override=None)
            if new:
                return new
        except Exception:
            pass
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

    # ── Real-time path planner ────────────────────────────────────────────────

    def start_realtime_planner(self, goal):
        """Launch the background path-watchdog targeting `goal` (map coords)."""
        self.stop_realtime_planner()
        with self._rt_planner_lock:
            self._rt_planner_goal = (int(goal[0]), int(goal[1]))
            self._rt_active_path  = None
            self._rt_new_path     = None
            self._rt_replan_ready = False
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

    def _paths_similar(self, a, b, stride=5, max_delta=3.0):
        if not a or not b:
            return False
        if abs(len(a) - len(b)) <= 2:
            sample_count = min(len(a), len(b), 6)
            if sample_count <= 1:
                return True
            idxs = np.linspace(0, min(len(a), len(b)) - 1, sample_count).astype(int)
            deltas = [
                np.linalg.norm(np.array(a[i], dtype=float) - np.array(b[i], dtype=float))
                for i in idxs
            ]
            return max(deltas, default=0.0) <= max_delta
        return False

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
                start = tuple(int(v) for v in self.get_map_position())
                new   = self.occ_map.astar_path(start, goal)
                if not new or len(new) <= 2:
                    new = self.occ_map.frontier_path(start, goal)
                if new and len(new) > 2:
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

        return regions, chosen, path, chasing

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
                        now = time.time()
                        if (now - self._last_rt_swap_time >= 1.0 and
                                not self._paths_similar(cur_path, rt)):
                            cur_path = rt
                            self.update_realtime_path(cur_path)
                            self._last_rt_swap_time = now
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
                        if self.camera_detection_signal is not None:
                            sig = self.camera_detection_signal
                            self.camera_detection_signal = None

                    if self.obstacle_in_front():
                        replan_count += 1
                        self.clear_obstacle()
                        self._refresh_map_sensors()
                        if replan_count >= 3:
                            return False
                        new = self._attempt_replan(goal)
                        if new and len(new) > 5:
                            cur_path = new
                            self.update_realtime_path(cur_path)
                            tidx = 0
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

                    try:
                        if tick % 4 == 0:
                            self._refresh_map_sensors()
                        interval_hit = bool(replan_interval) and (tick - last_replan_tick >= replan_interval)
                        if interval_hit:
                            new = self.occ_map.frontier_path(self.get_map_position(), goal)
                            if not new:
                                new = self.occ_map.astar_path(self.get_map_position(), goal)
                            if new and len(new) > 5:
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
                        self._refresh_map_sensors()
                        self.unstick()
                        try:
                            fn  = (self.occ_map.astar_path if use_global_planner
                                   else self.occ_map.frontier_path)
                            new = fn(self.get_map_position(), goal)
                            if new and len(new) > 5:
                                cur_path = list(new)
                                self.update_realtime_path(cur_path)
                                tidx = 0
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

                try:
                    if tick % 4 == 0:
                        self._refresh_map_sensors()
                    blocked = self._path_blocked_from_pose(cur_path, lookahead=24)
                    interval_hit = bool(replan_interval) and (tick - last_replan_tick >= replan_interval)
                    if blocked or interval_hit:
                        new = self._attempt_replan(goal)
                        if new and len(new) > 2:
                            cur_path = list(new)
                            waypoints = _build_waypoints(cur_path)
                            i = -1  # outer loop adds 1 → starts at waypoints[0]
                            last_replan_tick = tick
                            break
                        new = _plan()
                        if new and len(new) > 2:
                            cur_path = list(new)
                            waypoints = _build_waypoints(cur_path)
                            i = -1
                            last_replan_tick = tick
                            break
                        try:
                            self.avoid_obstacle()
                        except Exception:
                            pass
                        self._correct_odom_after_collision()
                        for _ in range(30):
                            if self.step(self.time_step) == -1: break
                        new = _plan()
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
                    try:
                        self.avoid_obstacle()
                    except Exception:
                        pass
                    self._correct_odom_after_collision()
                    for _ in range(30):
                        if self.step(self.time_step) == -1: break
                    new = _plan()
                    if new and len(new) > 2:
                        cur_path  = list(new)
                        waypoints = _build_waypoints(cur_path)
                        i = -1
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
        self.start_camera_thread()
        self.start_lidar_thread()

        if debug:
            self.occ_map.start_viz()

        prev_grid = self.occ_map.grid_map.copy()
        map_diff  = 1.0
        count     = 0

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
                continue

            if sig == 'green_carpet':
                self.stop_motor()
                self.mark_green_carpet_permanently(min_pixel_threshold=10000)
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
                continue

            map_diff = utils.map_delta_ratio(prev_grid, self.occ_map.grid_map)

            # Single goal selection: _update_frontier owns planning; the loop owns
            # navigation. This removes the dual-goal confusion where _update_frontier
            # navigated internally AND the outer loop also tried to navigate.
            _, chosen, path_to_chosen, chasing = self._update_frontier(count, map_diff)

            # Fallback: if no frontier path, occasionally explore a nearby free cell
            if path_to_chosen is None and random.random() < 0.2:
                fb = self._nearby_free_cell()
                if fb is not None:
                    chosen         = fb
                    path_to_chosen = self.occ_map.frontier_path(self.get_map_position(), chosen)
                    chasing        = False

            # One navigation call per iteration — no competing goals
            if path_to_chosen:
                ok = self.navigate_frontier(path_to_chosen)
                if ok:
                    self._last_frontier_goal = chosen
                if ok and chasing and not self.found_all_2_columns():
                    self.scan_360()

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
        ok  = (y_n > 0.001) & (D > 0.1) & (D < 4.0)
        D_v, x_nv = D[ok], x_n[ok]
        pts_local = np.stack([D_v + self.X_offset, -D_v * x_nv + self.Y_offset], axis=1)
        pts_world = self.transform_points_to_world(pts_local)
        return self.convert_to_map_coordinate_matrix(pts_world)

    def mark_green_carpet_permanently(self, min_pixel_threshold=10,
                                      new_area_threshold=0.5):
        now = time.time()
        if (now - self.last_green_mark_time) < self.green_mark_cooldown:
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
            try:
                fill_mask = np.zeros(self.occ_map.grid_map.shape, dtype=np.uint8)
                fill_mask[yi2, xi2] = 1
                kernel = np.ones((3, 3), dtype=np.uint8)
                fill_mask = cv2.morphologyEx(fill_mask, cv2.MORPH_CLOSE, kernel, iterations=2)
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
