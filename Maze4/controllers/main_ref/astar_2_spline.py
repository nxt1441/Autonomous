import heapq
import numpy as np
from scipy.ndimage import distance_transform_edt
from scipy.interpolate import CubicSpline, splprep, splev


class _GridPlanner:
    """A* path search on a binary occupancy grid with wall-clearance penalty."""

    _WALL_MARGIN = 5.0
    _WALL_COST   = 2.0
    _H_SCALE     = 1.2
    # Standard A* tie-breaking (Amit Patel, "Heuristics" -- redblobgames):
    # without this, two routes of genuinely equal cost (very common in a
    # grid maze with symmetric corridors) tie on f = g + h, and heapq then
    # falls back to comparing the raw (row, col) heap tuple, which has
    # nothing to do with path quality. That tie-break is a pure function of
    # the grid, so replanning from the exact same pose gives the same
    # result -- but replans fire from a slightly different robot pose and
    # against a slightly-updated cost map each time, so which of the two
    # equal-cost routes "wins" the row/col tiebreak can flip between
    # replans even though nothing meaningful changed, and the robot swaps
    # between two different-looking paths to the same goal. Multiplying
    # the heuristic by a hair over 1 breaks ties in favour of nodes closer
    # to the direct line toward the goal instead, which is both consistent
    # across replans and the more sensible choice when costs are equal.
    _TIE_BREAK_EPS = 0.001

    # 8-connected 2-step moves: (row_step, col_step, travel_cost)
    _MOVES = [
        ( 0,  2, 2.0), ( 0, -2, 2.0), ( 2,  0, 2.0), (-2,  0, 2.0),
        ( 2,  2, 2.828), ( 2, -2, 2.828), (-2,  2, 2.828), (-2, -2, 2.828),
    ]

    def __init__(self, grid, cost_map=None, cost_weight=1.5):
        self._rows, self._cols = grid.shape
        self._grid = grid
        wall_dist = distance_transform_edt(grid == 0)
        self._penalty = np.where(
            wall_dist < self._WALL_MARGIN,
            self._WALL_COST * (1.0 - wall_dist / self._WALL_MARGIN),
            0.0,
        ).astype(np.float32)
        self._cost_map    = cost_map
        self._cost_weight = cost_weight

    def _heuristic(self, cx, cy, gx, gy):
        return np.sqrt((cx - gx) ** 2 + (cy - gy) ** 2) * self._H_SCALE * (1.0 + self._TIE_BREAK_EPS)

    def search(self, sx, sy, gx, gy):
        rows, cols = self._rows, self._cols
        g_cost = np.full((rows, cols), np.inf, dtype=np.float32)
        came_from = np.full((rows, cols, 2), -1, dtype=np.int32)
        g_cost[sy, sx] = 0.0
        heap = [(self._heuristic(sx, sy, gx, gy), sy, sx)]

        while heap:
            f, cy, cx = heapq.heappop(heap)

            if abs(cx - gx) <= 2 and abs(cy - gy) <= 2:
                return _trace_path(came_from, cy, cx)

            if f > g_cost[cy, cx] + self._heuristic(cx, cy, gx, gy):
                continue

            for dy, dx, step in self._MOVES:
                ny, nx = cy + dy, cx + dx
                if 0 <= ny < rows and 0 <= nx < cols and self._grid[ny, nx] == 0:
                    extra = (float(self._cost_map[ny, nx]) * self._cost_weight
                             if self._cost_map is not None else 0.0)
                    nc = g_cost[cy, cx] + step + self._penalty[ny, nx] + extra
                    if nc < g_cost[ny, nx]:
                        g_cost[ny, nx] = nc
                        came_from[ny, nx] = [cy, cx]
                        heapq.heappush(heap, (nc + self._heuristic(nx, ny, gx, gy), ny, nx))
        return []


def _trace_path(came_from, cy, cx):
    waypoints = []
    cur = [cy, cx]
    while cur[0] != -1:
        waypoints.append((cur[1], cur[0]))
        cur = came_from[cur[0], cur[1]]
    return waypoints[::-1]


def _dedup_path(xs, ys):
    seen, out = set(), []
    for p in ((int(round(x)), int(round(y))) for x, y in zip(xs, ys)):
        if p not in seen:
            seen.add(p)
            out.append(p)
    return out


def _cubic_smooth(arr, s):
    t = np.arange(len(arr))
    cx = CubicSpline(t, arr[:, 0], bc_type='natural')
    cy = CubicSpline(t, arr[:, 1], bc_type='natural')
    sps = max(5, int(20 * s))
    tf = np.linspace(0, len(arr) - 1, len(arr) * sps)
    return _dedup_path(cx(tf), cy(tf))


def _parametric_smooth(arr, s):
    if len(arr) < 4:
        return _cubic_smooth(arr, s)
    sf = len(arr) * (1 - s) * 10
    tck, _ = splprep([arr[:, 0], arr[:, 1]], s=sf, k=3)
    n = len(arr) * max(10, int(30 * s))
    coords = splev(np.linspace(0, 1, n), tck)
    return _dedup_path(coords[0], coords[1])


def _spline_smooth(arr, s):
    if len(arr) < 4:
        return arr.tolist()
    sv = max(0.0, s * len(arr))
    try:
        tck, _ = splprep([arr[:, 0], arr[:, 1]], s=sv, k=3)
        n = max(10, len(arr) * 8)
        coords = splev(np.linspace(0, 1, n), tck)
        return _dedup_path(coords[0], coords[1])
    except Exception:
        return arr.tolist()


def _apply_smoothing(path, method='bspline', smoothness=0.3):
    if len(path) < 2:
        return path
    arr = np.array(path, dtype=float)
    if method == 'natural_spline':
        return _cubic_smooth(arr, smoothness)
    if method == 'parametric_spline':
        return _parametric_smooth(arr, smoothness)
    return _spline_smooth(arr, smoothness)


def runAStarSearch(grid, start, goal, cost_map=None, cost_weight=1.5):
    planner = _GridPlanner(grid, cost_map=cost_map, cost_weight=cost_weight)
    raw = planner.search(int(start[0]), int(start[1]), int(goal[0]), int(goal[1]))
    if not raw:
        return []
    return _apply_smoothing(raw, method='bspline', smoothness=0.1)
