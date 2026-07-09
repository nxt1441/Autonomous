import utils
from CONSTANTS import *
import numpy as np
import cv2
from collections import deque
from astar_2_spline import runAStarSearch as _astar
import threading
import time
import os
import tempfile
import multiprocessing as mp
import queue as queue_mod
import atexit


def _parent_alive(parent_pid):
    if parent_pid <= 0:
        return False
    try:
        os.kill(parent_pid, 0)
        return True
    except OSError:
        return False


def _plotter_process_main(frame_queue, stop_event, size, parent_pid):
    cache_root = os.path.join(tempfile.gettempdir(), 'main_ref_plot_cache')
    os.makedirs(cache_root, exist_ok=True)
    os.environ.setdefault('MPLCONFIGDIR', os.path.join(cache_root, 'matplotlib'))
    os.environ.setdefault('XDG_CACHE_HOME', cache_root)

    try:
        import matplotlib
        for backend in ['macosx', 'QtAgg', 'Qt5Agg', 'Agg']:
            try:
                matplotlib.use(backend)
                break
            except Exception:
                pass
        import matplotlib.pyplot as plt

        plt.ion()
        fig, ax = plt.subplots(figsize=(8, 8))
        ax.set_title('Occupancy Grid - Live')
        ax.axis('off')
        blank = np.full((size, size, 3), 80, dtype=np.uint8)
        # 'nearest', not 'bilinear': this is a categorical/label image (each
        # colour is a discrete map state), not continuous data. Bilinear
        # blending fabricates fake intermediate colours at every cell
        # boundary (e.g. a red/white edge rendering as pink) that don't
        # correspond to any real map state, actively hurting readability.
        im = ax.imshow(blank, interpolation='nearest')
        fig.tight_layout()
        fig.canvas.draw()
        try:
            fig.canvas.flush_events()
        except Exception:
            pass

        last_parent_check = time.time()
        while not stop_event.is_set():
            now = time.time()
            if now - last_parent_check >= 0.25:
                last_parent_check = now
                if not _parent_alive(parent_pid):
                    break
            try:
                rgb = frame_queue.get(timeout=0.05)
                while True:
                    try:
                        rgb = frame_queue.get_nowait()
                    except queue_mod.Empty:
                        break
            except queue_mod.Empty:
                try:
                    fig.canvas.flush_events()
                except Exception:
                    pass
                continue

            im.set_data(rgb)
            try:
                fig.canvas.draw_idle()
                fig.canvas.flush_events()
            except Exception:
                pass
        try:
            plt.close(fig)
        except Exception:
            pass
    except Exception as e:
        print(f'[MapRenderer] plotter process failed: {e}')


class MapRenderer:
    """Latest-frame display that never blocks the robot control loop."""

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
        self._process = None
        self._queue = None
        self._stop_event = None
        self._atexit_registered = False

    def start(self):
        if self._process and self._process.is_alive():
            return
        self._queue = mp.Queue(maxsize=1)
        self._stop_event = mp.Event()
        size = self._size * max(1, int(MAP_RENDER_SCALE))
        self._process = mp.Process(
            target=_plotter_process_main,
            args=(self._queue, self._stop_event, size, os.getpid()),
            daemon=True,
        )
        self._process.start()
        if not self._atexit_registered:
            atexit.register(self.shutdown)
            self._atexit_registered = True
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
            visible_regions = [
                r for r in frontier_regions
                if len(r) >= FRONTIER_RENDER_MIN_CELLS
            ]
            ordered = sorted(visible_regions, key=lambda r: len(r))
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
                pts = np.array(region, dtype=np.float32)
                cx, cy = np.mean(pts, axis=0)
                mx, my = int(round(float(cx))), int(round(float(cy)))
                for dy in range(-2, 3):
                    for dx in range(-2, 3):
                        if dx * dx + dy * dy > 4:
                            continue
                        x, y = mx + dx, my + dy
                        if 0 <= x < w and 0 <= y < h:
                            rgb[y, x] = c
        return rgb

    def draw(self, grid, robot_pos=None, path=None, target=None,
             columns=None, frontier_regions=None,
             start_point=None, end_point=None, cost_map=None,
             floating_points=None):
        if not self._live or self._queue is None:
            return
        rgb = self._compose_rgb(
            grid, robot_pos=robot_pos, path=path, target=target,
            columns=columns, frontier_regions=frontier_regions,
            start_point=start_point, end_point=end_point,
            cost_map=cost_map, floating_points=floating_points,
        )
        try:
            self._queue.put_nowait(rgb)
        except queue_mod.Full:
            try:
                self._queue.get_nowait()
            except queue_mod.Empty:
                pass
            try:
                self._queue.put_nowait(rgb)
            except queue_mod.Full:
                pass

    def _compose_rgb(self, grid, robot_pos=None, path=None, target=None,
                     columns=None, frontier_regions=None,
                     start_point=None, end_point=None, cost_map=None,
                     floating_points=None):
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

        # Path/target/start markers deliberately avoid the base grid's own
        # semantic colours (red = floating wall/DEPTH_OBSTACLE, green =
        # GREEN_CARPET, blue = BLUE_COLUMN) so an overlay marker can never
        # be mistaken for the map data it's drawn on top of.
        if path:
            for px, py in path:
                px, py = int(px), int(py)
                if 0 <= px < w and 0 <= py < h:
                    rgb[py, px] = (255, 0, 255)  # magenta
        if target:
            _mark(target[0], target[1], (255, 140, 0), r=4)  # orange
        if columns:
            for col in columns:
                if isinstance(col, (list, tuple)) and len(col) >= 3:
                    _mark(col[0], col[1], col[2], r=2)
        if start_point:
            _mark(start_point[0], start_point[1], (0, 255, 255), r=6)  # cyan
        if end_point:
            _mark(end_point[0], end_point[1], (255, 200, 0), r=6)
        if robot_pos:
            _mark(robot_pos[0], robot_pos[1], (0, 100, 255), r=4)

        # Floating wall overlay: every entry is a confirmed, blocking
        # floating-wall cell (see MyRobot._refresh_map_depth) -- mark it
        # solid red, matching DEPTH_OBSTACLE's own base grid colour above,
        # so floating walls read unambiguously as red everywhere on the map.
        if floating_points:
            for fx, fy, _tag in floating_points:
                _mark(fx, fy, (255, 0, 0), r=2)

        if MAP_RENDER_SCALE > 1:
            # INTER_NEAREST, not INTER_LINEAR: same reasoning as the
            # matplotlib imshow call above -- this upscales discrete
            # category colours, and linear blending would blur crisp cell
            # boundaries into misleading intermediate colours.
            rgb = cv2.resize(rgb, (w * MAP_RENDER_SCALE, h * MAP_RENDER_SCALE),
                             interpolation=cv2.INTER_NEAREST)
        return rgb

    def shutdown(self):
        if self._stop_event is not None:
            self._stop_event.set()
        if self._process and self._process.is_alive():
            self._process.join(timeout=1.0)
            if self._process.is_alive():
                self._process.terminate()
                self._process.join(timeout=0.5)
        if self._queue is not None:
            try:
                self._queue.cancel_join_thread()
            except Exception:
                pass
            try:
                self._queue.close()
            except Exception:
                pass
        self._process = None
        self._queue = None
        self._stop_event = None
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
        self._last_cost_map_time = 0.0

        # Wall-clock throttle for the live Matplotlib view. The control loop asks for
        # a redraw on a fixed tick count, but a single draw on the macOS backend can
        # take 100-300 ms and blocks robot.step(), which makes the Webots real-time
        # factor stutter between ~1.5x and ~0x. Capping the *draw* rate (the redraw is
        # purely cosmetic — it never feeds mapping, planning, or motion) keeps the step
        # loop fed and the simulation speed smooth. Logic is unchanged.
        self._last_viz_time  = 0.0
        self._viz_min_interval = 1.0 / max(1, int(MAP_RENDER_FPS))

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
        if lidar_pts is None or len(lidar_pts) == 0:
            return
        free_mask = np.zeros_like(self.grid_map, dtype=np.uint8)
        hit_mask  = np.zeros_like(self.grid_map, dtype=np.uint8)
        sx, sy = int(robot_pos[0]), int(robot_pos[1])
        for pt in lidar_pts:
            x, y = int(pt[0]), int(pt[1])
            cv2.line(free_mask, (sx, sy), (x, y), 1, 1)
            if 0 <= x < MAP_SIZE and 0 <= y < MAP_SIZE:
                hit_mask[y, x] = 1

        free = free_mask.astype(bool)
        hit  = hit_mask.astype(bool)
        free[hit] = False

        if self._depth_obstacle_cells:
            da = np.array(list(self._depth_obstacle_cells), dtype=np.int32)
            xs, ys = da[:, 0], da[:, 1]
            ok = (xs >= 0) & (xs < MAP_SIZE) & (ys >= 0) & (ys < MAP_SIZE)
            if np.any(ok):
                free[ys[ok], xs[ok]] = False
                hit[ys[ok], xs[ok]] = False

        free_update = free & (self.log_odds < 3.5)
        self.log_odds[free_update] -= 0.28
        self.log_odds[hit] += 1.00

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

        # A depth-camera cell must never be erased by LiDAR's clearing
        # evidence (LiDAR is blind at the floating-wall height band by
        # definition, so a "clear" ray there proves nothing), but LiDAR's
        # own OBSTACLE-strength evidence for that SAME cell is real ground
        # truth and must be allowed through — excluding depth cells from
        # obstacle_mask here would silently block the "let LiDAR win"
        # behaviour the re-stamp step below already assumes happens.
        unknown_mask  = (self.log_odds == INITIAL_LOG_ODD) & ~protected
        # log_odds != INITIAL_LOG_ODD guards against a subtlety specific to
        # depth cells: INITIAL_LOG_ODD (1.0) already sits above the P>0.70
        # obstacle threshold on its own, so an untouched cell would satisfy
        # obstacle_mask purely from the neutral starting value. For ordinary
        # cells that's harmless (unknown_mask overwrites it back to UNKNOWN
        # right below), but a depth cell is excluded from unknown_mask by
        # `protected`, so without this guard a floating wall LiDAR has never
        # actually scanned would get wrongly promoted to a true OBSTACLE.
        obstacle_mask = (P > 0.70) & (self.log_odds != INITIAL_LOG_ODD) & ~(closed_mask | green_mask)
        free_mask     = (P < 0.42) & ~protected

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
                # Do not override a LiDAR-confirmed regular wall (OBSTACLE) with
                # DEPTH_OBSTACLE — let LiDAR win so wrongly-detected cells self-correct.
                unprotected = (cur != GREEN_CARPET) & (cur != CLOSED) & (cur != OBSTACLE)
                self.grid_map[ys_d[unprotected], xs_d[unprotected]] = DEPTH_OBSTACLE

    def process_scan(self, robot_pos, lidar_points):
        map_pts = self.world_pts_to_map(lidar_points)
        self._apply_ray_update(robot_pos, map_pts)
        self.rebuild_grid()


    def build_cost_map(self, max_dist=12, force=False):
        now = time.time()
        if (not force and self.cost_map is not None and
                now - self._last_cost_map_time < COST_MAP_UPDATE_INTERVAL):
            return self.cost_map
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
        self._last_cost_map_time = now
        return self.cost_map

    # ── Frontier detection ────────────────────────────────────────────────────

    def compute_frontiers(self):
        grid = self.grid_map
        free = (grid == FREESPACE).astype(np.uint8)
        unknown = (grid == UNKNOWN).astype(np.uint8)
        kernel = np.array([[0, 1, 0],
                           [1, 0, 1],
                           [0, 1, 0]], dtype=np.uint8)
        unknown_adjacent = cv2.dilate(unknown, kernel, iterations=1).astype(bool)
        frontier_mask = free.astype(bool) & unknown_adjacent
        frontier_mask[[0, -1], :] = False
        frontier_mask[:, [0, -1]] = False
        ys, xs = np.where(frontier_mask)
        frontier_cells = list(zip(xs.tolist(), ys.tolist()))
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
        best_path, best_len = None, float('inf')
        fallback_path, fallback_len = None, 0.0
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
            if not g_mask[int(end[1]), int(end[0])]:
                utils.clear_around_point(tmp, end, inflation_pixels=ASTAR_EXPANSION_PIXELS)
            if not g_mask[int(start[1]), int(start[0])]:
                utils.clear_around_point(tmp, start, inflation_pixels=ASTAR_EXPANSION_PIXELS)
            path = _astar(tmp, start, end, cost_map=cost_map_to_use)
            if path is None or len(path) <= 1:
                continue
            if any(g_mask[int(py), int(px)] for px, py in path
                   if 0 <= int(px) < self.map_size and 0 <= int(py) < self.map_size):
                continue
            clearance_tmp = utils.dilate_obstacles(tmp.copy(), inflation_pixels=ASTAR_MIN_CLEARANCE_PIXELS)
            if any(clearance_tmp[int(py), int(px)] == OBSTACLE for px, py in path[2:-2]
                   if 0 <= int(px) < self.map_size and 0 <= int(py) < self.map_size):
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
                if total < best_len:
                    best_len, best_path = total, path
            elif total > fallback_len:
                fallback_len, fallback_path = total, path
        return best_path if best_path is not None else fallback_path

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
        if not g_mask[int(end[1]), int(end[0])]:
            utils.clear_around_point(tmp, end, inflation_pixels=ASTAR_EXPANSION_PIXELS)
        if not g_mask[int(start[1]), int(start[0])]:
            utils.clear_around_point(tmp, start, inflation_pixels=ASTAR_EXPANSION_PIXELS)
        path = _astar(tmp, start, end, cost_map=self.cost_map)
        if path and any(g_mask[int(py), int(px)] for px, py in path
                        if 0 <= int(px) < self.map_size and 0 <= int(py) < self.map_size):
            return None
        return path

    # ── Coordinate conversion ─────────────────────────────────────────────────

    def world_pts_to_map(self, pts_world):
        R = np.array([[1 / RESOLUTION, 0], [0, -1 / RESOLUTION]])
        t = np.array([MAP_SIZE // 2, MAP_SIZE // 2])
        return np.rint(pts_world @ R.T + t).astype(np.int32)

    # ── Visualisation ─────────────────────────────────────────────────────────

    def start_viz(self):
        self._renderer.start()

    def stop_viz(self):
        self._renderer.shutdown()

    def refresh_viz(self):
        # Skip draws that arrive faster than the display rate so a slow backend redraw
        # can never stall the control loop and tank the simulation real-time factor.
        now = time.time()
        if now - self._last_viz_time < self._viz_min_interval:
            return
        self._last_viz_time = now
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
