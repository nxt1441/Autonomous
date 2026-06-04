import utils
from CONSTANTS import *
import numpy as np
import cv2
from collections import deque
from astar_2_spline import runAStarSearch as _astar
import threading


class MapRenderer:
    """Non-blocking matplotlib display, refreshed from the main thread."""

    _PALETTE = {
        FREESPACE:     (255, 255, 255),
        OBSTACLE:      (0,   0,   0),
        DEPTH_OBSTACLE:(255, 0,   0),
        UNKNOWN:       (80,  80,  80),
        BLUE_COLUMN:   (0,   0,   255),
        YELLOW_COLUMN: (255, 255, 0),
        CLOSED:        (128, 0,   128),
        GREEN_CARPET:  (0,   200, 0),
        50:            (0,   0,   200),
        101:           (0,   200, 200),
        200:           (200, 200, 0),
        220:           (200, 0,   0),
    }

    def __init__(self, grid_size):
        self._size  = grid_size
        self._fig   = None
        self._ax    = None
        self._im    = None
        self._live  = False

    def start(self):
        import matplotlib
        for backend in ['macosx', 'TkAgg', 'QtAgg', 'Qt5Agg', 'Agg']:
            try:
                matplotlib.use(backend)
                break
            except Exception:
                pass
        import matplotlib.pyplot as plt
        plt.ion()
        self._fig, self._ax = plt.subplots(figsize=(8, 8))
        self._ax.set_title('Occupancy Grid — Live')
        self._ax.axis('off')
        blank = np.full((self._size, self._size, 3), 80, dtype=np.uint8)
        self._im = self._ax.imshow(blank, interpolation='nearest')
        self._fig.tight_layout()
        self._fig.canvas.draw()
        try:
            self._fig.canvas.flush_events()
        except Exception:
            pass
        self._live = True

    def _to_rgb(self, grid, frontier_regions=None, cost_map=None):
        h, w = grid.shape
        rgb = np.full((h, w, 3), 80, dtype=np.uint8)
        for val, color in self._PALETTE.items():
            rgb[grid == val] = color

        # Cost map heat overlay: tint freespace cells orange near walls
        if cost_map is not None:
            free_mask = (grid == FREESPACE)
            heat = np.clip(cost_map * 220, 0, 220).astype(np.int16)
            active = free_mask & (heat > 10)
            rgb[active, 0] = np.clip(
                rgb[active, 0].astype(np.int16) + heat[active], 0, 255
            ).astype(np.uint8)
            rgb[active, 1] = np.clip(
                rgb[active, 1].astype(np.int16) - heat[active] // 2, 0, 255
            ).astype(np.uint8)

        if frontier_regions:
            ordered = sorted(frontier_regions, key=lambda r: len(r))
            n = len(ordered)
            for idx, region in enumerate(ordered):
                if n == 1:
                    c = (0, 200, 200)
                elif idx == n - 1:
                    c = (200, 0, 0)
                elif idx > n // 2:
                    c = (200, 200, 0)
                else:
                    c = (0, 0, 200)
                for x, y in region:
                    if 0 <= x < w and 0 <= y < h:
                        rgb[y, x] = c
        return rgb

    def draw(self, grid, robot_pos=None, path=None, target=None,
             columns=None, frontier_regions=None,
             start_point=None, end_point=None, cost_map=None,
             floating_points=None):
        if not self._live:
            return
        import matplotlib.pyplot as plt
        rgb = self._to_rgb(grid, frontier_regions, cost_map=cost_map)
        h, w = grid.shape

        def _mark(x, y, color, r=3):
            x, y = int(round(x)), int(round(y))
            for dy in range(-r, r + 1):
                for dx in range(-r, r + 1):
                    if dx*dx + dy*dy > r*r:
                        continue
                    nx, ny = x + dx, y + dy
                    if 0 <= nx < w and 0 <= ny < h:
                        rgb[ny, nx] = color

        if path:
            for px, py in path:
                px, py = int(px), int(py)
                if 0 <= px < w and 0 <= py < h:
                    rgb[py, px] = (255, 0, 0)
        if target:
            _mark(target[0], target[1], (0, 255, 0), r=4)
        if columns:
            for col in columns:
                if isinstance(col, (list, tuple)) and len(col) >= 3:
                    _mark(col[0], col[1], col[2], r=2)
        if start_point:
            _mark(start_point[0], start_point[1], (0, 0, 255), r=6)
        if end_point:
            _mark(end_point[0], end_point[1], (255, 200, 0), r=6)
        if robot_pos:
            _mark(robot_pos[0], robot_pos[1], (0, 100, 255), r=4)

        # Floating wall overlay: list of (x, y, orientation) where orientation is
        # 'horizontal' or 'vertical' — choose colors accordingly
        if floating_points:
            for fx, fy, orient in floating_points:
                col = (200, 0, 200) if orient == 'horizontal' else (0, 200, 200)
                _mark(fx, fy, col, r=2)

        self._im.set_data(rgb)
        try:
            self._fig.canvas.draw_idle()
            self._fig.canvas.flush_events()
        except Exception:
            pass

    def shutdown(self):
        if self._live and self._fig is not None:
            import matplotlib.pyplot as plt
            try:
                plt.close(self._fig)
            except Exception:
                pass
        self._live = False


class OccupancyGrid:
    """Occupancy grid: log-odds updates, frontier clustering, A* path planning."""

    def __init__(self, robot=None):
        self.robot      = robot
        self.map_size   = MAP_SIZE
        self.resolution = RESOLUTION
        self.log_odds   = np.full((MAP_SIZE, MAP_SIZE), INITIAL_LOG_ODD, dtype=np.float32)
        self.grid_map   = np.full((MAP_SIZE, MAP_SIZE), UNKNOWN, dtype=np.uint8)

        self.frontier_regions  = []
        self.visited_frontiers = []

        self.current_path    = None
        self.robot_position  = None
        self.target_position = None
        self.column_points   = []
        self.floating_points  = []
        self.vis_lock        = threading.Lock()
        self.cost_map        = None

        # Depth-camera obstacles that lidar cannot see (floating walls, ground-level walls).
        # Stored as a set of (x, y) map cells and re-applied after every rebuild_grid so
        # lidar log-odds updates can never erase them.
        self._depth_obstacle_cells = set()

        self._renderer = MapRenderer(MAP_SIZE)

    # ── Obstacle query ────────────────────────────────────────────────────────

    def cell_blocked(self, map_target):
        cell = self.grid_map[map_target[1], map_target[0]]
        return cell in (OBSTACLE, DEPTH_OBSTACLE, GREEN_CARPET, CLOSED)

    # ── Log-odds update ───────────────────────────────────────────────────────

    def _apply_ray_update(self, robot_pos, lidar_pts):
        # Floating-wall cells are owned by the depth sensor; LiDAR must not touch them.
        depth_cells = self._depth_obstacle_cells
        for pt in lidar_pts:
            cells = utils.ray_cells(robot_pos, pt)
            for x, y in cells[:-1]:
                if 0 <= x < MAP_SIZE and 0 <= y < MAP_SIZE:
                    if (x, y) in depth_cells:
                        continue
                    if self.log_odds[y, x] < 3.5:
                        self.log_odds[y, x] -= 0.36
            x, y = cells[-1]
            if 0 <= x < MAP_SIZE and 0 <= y < MAP_SIZE:
                if (x, y) not in depth_cells:
                    self.log_odds[y, x] += 0.85

    def rebuild_grid(self):
        clipped = np.clip(self.log_odds, -5, 5)
        P = 1.0 / (1.0 + np.exp(-clipped))

        closed_mask = (self.grid_map == CLOSED)
        green_mask  = (self.grid_map == GREEN_CARPET)
        depth_mask = np.zeros(self.grid_map.shape, dtype=bool)
        if self._depth_obstacle_cells:
            da = np.array(list(self._depth_obstacle_cells), dtype=np.int32)
            xs_d, ys_d = da[:, 0], da[:, 1]
            in_bounds = (xs_d >= 0) & (xs_d < self.map_size) & \
                        (ys_d >= 0) & (ys_d < self.map_size)
            xs_d, ys_d = xs_d[in_bounds], ys_d[in_bounds]
            if len(xs_d):
                depth_mask[ys_d, xs_d] = True

        protected   = closed_mask | green_mask | depth_mask

        unknown_mask  = (self.log_odds == INITIAL_LOG_ODD) & ~protected
        obstacle_mask = (P > 0.7) & ~protected
        free_mask     = (P < 0.5) & ~protected

        # Connected-component filter: drop noise blobs < 8 px
        obs_bin = obstacle_mask.astype(np.uint8)
        n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(obs_bin, connectivity=4)
        clean = np.zeros_like(obs_bin)
        for i in range(1, n_labels):
            if stats[i, cv2.CC_STAT_AREA] >= 8:
                clean[labels == i] = 1
        obstacle_mask = clean.astype(bool) & ~protected

        self.grid_map[obstacle_mask] = OBSTACLE
        self.grid_map[free_mask]     = FREESPACE
        self.grid_map[unknown_mask]  = UNKNOWN
        self.grid_map[green_mask]    = GREEN_CARPET
        self.grid_map[closed_mask]   = CLOSED

        # Re-stamp depth-camera obstacles: lidar cannot see floating walls or very low
        # walls, so its log-odds updates would clear them. Re-applying here after every
        # rebuild ensures they survive lidar scans.
        if self._depth_obstacle_cells:
            da = np.array(list(self._depth_obstacle_cells), dtype=np.int32)
            xs_d, ys_d = da[:, 0], da[:, 1]
            in_bounds = (xs_d >= 0) & (xs_d < self.map_size) & \
                        (ys_d >= 0) & (ys_d < self.map_size)
            xs_d, ys_d = xs_d[in_bounds], ys_d[in_bounds]
            if len(xs_d):
                cur = self.grid_map[ys_d, xs_d]
                unprotected = (cur != GREEN_CARPET) & (cur != CLOSED)
                self.grid_map[ys_d[unprotected], xs_d[unprotected]] = DEPTH_OBSTACLE

    def process_scan(self, robot_pos, lidar_points):
        map_pts = self.world_pts_to_map(lidar_points)
        self._apply_ray_update(robot_pos, map_pts)
        self.rebuild_grid()

    def set_cell(self, map_point, value):
        x, y = map_point
        if 0 <= x < self.map_size and 0 <= y < self.map_size:
            self.grid_map[y, x] = value

    def build_cost_map(self, max_dist=12):
        from scipy.ndimage import distance_transform_edt
        obs = ((self.grid_map == OBSTACLE) |
               (self.grid_map == DEPTH_OBSTACLE) |
               (self.grid_map == CLOSED)   |
               (self.grid_map == GREEN_CARPET))
        dist = distance_transform_edt(~obs)
        self.cost_map = np.where(
            dist < max_dist,
            ((1.0 - dist / max_dist) ** 2),
            0.0
        ).astype(np.float32)
        return self.cost_map

    # ── Closure marking ───────────────────────────────────────────────────────

    def stamp_closure(self, forward_m=0.7, back_m=-0.2, width_m=0.6, value=CLOSED):
        if self.robot is None or self.robot.is_turning():
            return False
        rx, ry  = self.robot.get_position()
        heading = self.robot.get_heading('rad')
        hw = width_m / 2.0
        front_offset = getattr(self.robot, 'axle_length', 0.0) / 2.0
        front_x = front_offset + forward_m
        rear_x  = front_offset - back_m
        corners_local = np.array([
            [front_x,  hw], [front_x, -hw],
            [rear_x,  -hw], [rear_x,   hw],
        ])
        R = np.array([[np.cos(heading), -np.sin(heading)],
                      [np.sin(heading),  np.cos(heading)]])
        corners_world = corners_local @ R.T + np.array([rx, ry])
        map_pts = [self.robot.convert_to_map_coordinates(float(x), float(y))
                   for x, y in corners_world]
        pts = np.array(map_pts, dtype=np.int32).reshape((-1, 1, 2))
        try:
            mask = np.zeros_like(self.grid_map, dtype=np.uint8)
            cv2.fillPoly(mask, [pts], color=1)
            poly_area = int(mask.sum())
            if poly_area == 0:
                return False
            existing = (self.grid_map == value).astype(np.uint8)
            if int((existing & mask).sum()) / poly_area >= CLOSURE_MARK_IOU_THRESHOLD:
                return False
            ys, xs = np.where(mask)
            x0 = max(0, xs.min()); x1 = min(self.grid_map.shape[1] - 1, xs.max())
            y0 = max(0, ys.min()); y1 = min(self.grid_map.shape[0] - 1, ys.max())
            roi = self.grid_map[y0:y1 + 1, x0:x1 + 1]
            roi[mask[y0:y1 + 1, x0:x1 + 1] == 1] = int(value)
            self.grid_map[y0:y1 + 1, x0:x1 + 1] = roi
            return True
        except Exception as e:
            print(f'[warning] stamp_closure failed: {e}')
            return False

    # ── Frontier detection ────────────────────────────────────────────────────

    def compute_frontiers(self):
        frontier_cells = []
        N4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        for x in range(1, self.map_size - 1):
            for y in range(1, self.map_size - 1):
                if self.grid_map[y, x] == FREESPACE:
                    if any(self.grid_map[y + dy, x + dx] == UNKNOWN for dx, dy in N4):
                        frontier_cells.append((x, y))
        self.frontier_regions = self._cluster_bfs(frontier_cells)
        return self.frontier_regions

    def _cluster_bfs(self, cells, min_size=8):
        if not cells:
            return []
        cell_set = set(cells)
        visited  = set()
        clusters = []
        N8 = [(-1,0),(1,0),(0,-1),(0,1),(-1,-1),(-1,1),(1,-1),(1,1)]
        for seed in cells:
            if seed in visited:
                continue
            cluster = []
            queue   = deque([seed])
            visited.add(seed)
            while queue:
                x, y = queue.popleft()
                cluster.append((x, y))
                for dx, dy in N8:
                    nb = (x + dx, y + dy)
                    nx, ny = nb
                    if (0 < nx < self.map_size - 1 and
                            0 < ny < self.map_size - 1 and
                            nb in cell_set and nb not in visited):
                        visited.add(nb)
                        queue.append(nb)
            if len(cluster) >= min_size:
                clusters.append(cluster)
        return clusters

    # ── Path planning ─────────────────────────────────────────────────────────

    def astar_path(self, start, end, inflation_levels=None, cost_map_override=None):
        """A* wrapper that tries multiple obstacle inflation levels.

        Parameters:
        - start, end: map coordinates
        - inflation_levels: list of inflation pixel values to try (defaults to ASTAR_INFLATION_LEVELS)
        - cost_map_override: numpy array to pass to the planner instead of self.cost_map
        """
        if inflation_levels is None:
            inflation_levels = ASTAR_INFLATION_LEVELS
        best_path, best_len = None, 0.0
        cost_map_to_use = self.cost_map if cost_map_override is None else cost_map_override
        for inflation in inflation_levels:
            base = self.grid_map.copy().astype(np.float32)
            base[base == DEPTH_OBSTACLE] = OBSTACLE
            c_mask = (base == CLOSED)
            g_mask = (base == GREEN_CARPET)
            tmp = base.copy()
            tmp[c_mask] = FREESPACE
            tmp[g_mask] = OBSTACLE
            tmp = utils.remove_small_blobs(tmp, obstacle_value=OBSTACLE, min_size=6, connectivity=4)
            tmp = utils.drop_single_pixels(tmp, obstacle_value=OBSTACLE, connectivity=4)
            tmp = utils.dilate_obstacles(tmp, inflation_pixels=inflation)
            tmp[c_mask] = OBSTACLE
            tmp[g_mask] = OBSTACLE
            utils.clear_around_point(tmp, end,   inflation_pixels=ASTAR_EXPANSION_PIXELS)
            utils.clear_around_point(tmp, start, inflation_pixels=ASTAR_EXPANSION_PIXELS)
            path = _astar(tmp, start, end, cost_map=cost_map_to_use)
            if path is None or len(path) <= 1:
                continue
            try:
                total = 0.0
                pw = self.robot.convert_to_world_coordinates(path[0][0], path[0][1])
                for p in path[1:]:
                    cw = self.robot.convert_to_world_coordinates(p[0], p[1])
                    total += np.hypot(cw[0] - pw[0], cw[1] - pw[1])
                    pw = cw
            except Exception:
                total = 0.0
            if total >= PATH_MIN_LENGTH_M:
                return path
            if total > best_len:
                best_len, best_path = total, path
        return best_path

    def frontier_path(self, start, end):
        if start is None or end is None:
            return []
        base = self.grid_map.copy().astype(np.float32)
        base[base == DEPTH_OBSTACLE] = OBSTACLE
        base = utils.remove_small_blobs(base, obstacle_value=OBSTACLE, min_size=6, connectivity=4)
        c_mask = (base == CLOSED)
        g_mask = (base == GREEN_CARPET)
        tmp = base.copy()
        tmp[c_mask] = FREESPACE
        tmp[g_mask] = OBSTACLE
        tmp = utils.dilate_obstacles(tmp, inflation_pixels=ASTAR_FRONTIER_INFLATION)
        tmp[c_mask] = OBSTACLE
        tmp[g_mask] = OBSTACLE
        utils.clear_around_point(tmp, end,   inflation_pixels=ASTAR_EXPANSION_PIXELS)
        utils.clear_around_point(tmp, start, inflation_pixels=ASTAR_EXPANSION_PIXELS)
        return _astar(tmp, start, end, cost_map=self.cost_map)

    # ── Coordinate conversion ─────────────────────────────────────────────────

    def world_pts_to_map(self, pts_world):
        R = np.array([[1 / RESOLUTION, 0], [0, -1 / RESOLUTION]])
        t = np.array([MAP_SIZE // 2, MAP_SIZE // 2])
        return (pts_world @ R.T + t).astype(np.int32)

    # ── Visualisation ─────────────────────────────────────────────────────────

    def start_viz(self):
        self._renderer.start()

    def stop_viz(self):
        self._renderer.shutdown()

    def refresh_viz(self):
        with self.vis_lock:
            grid_snap = self.grid_map.copy()
            rpos      = self.robot_position
            path      = self.current_path
            tgt       = self.target_position
            cols      = list(self.column_points) if self.column_points else []
            fronts    = list(self.frontier_regions)
            floating  = list(self.floating_points) if getattr(self, 'floating_points', None) else []
            spt = getattr(self.robot, 'start_point', None) if self.robot else None
            ept = getattr(self.robot, 'end_point',   None) if self.robot else None
            cost      = self.cost_map.copy() if self.cost_map is not None else None
        self._renderer.draw(
            grid_snap,
            robot_pos=rpos, path=path, target=tgt,
            columns=cols, frontier_regions=fronts,
            start_point=spt, end_point=ept,
            cost_map=cost,
            floating_points=floating,
        )
