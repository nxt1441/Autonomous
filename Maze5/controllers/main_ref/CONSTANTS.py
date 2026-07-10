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
# Physical area the grid covers, centered on the robot's start pose. 10 m
# was tight enough to clip parts of this maze; 12 m gives more margin
# around the maze bounds. MAP_SIZE=800 (1.5 cm/cell) made every per-frame
# whole-grid operation (cost map, rebuild, rendering) noticeably slower in
# simulation without a proportionate accuracy gain for this maze's feature
# sizes, since cell count -- and so per-frame cost -- grows with the SQUARE
# of MAP_SIZE. 500 cells (2.4 cm/cell) keeps most of the resolution
# improvement over the original 300/10m (3.33 cm/cell) at under half the
# cell count of 800 (500² vs 800² ≈ 2.56x fewer cells to touch every frame).
# MAP_RENDER_SCALE below handles the on-screen viewing size independently,
# so the displayed map doesn't look small just because the grid is lighter.
MAP_PHYSICAL_SIZE_M = 10
MAP_SIZE        = 500               # cells
RESOLUTION      = MAP_PHYSICAL_SIZE_M / MAP_SIZE
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
# The robot cannot physically fit through a gap narrower than its own
# body, no matter how many inflation levels the planner falls back
# through. Half of AXLE_LENGTH (wheel-to-wheel) plus a small margin for
# wheel/chassis width is the real minimum clearance needed from centreline
# to a wall on either side; every inflation level -- including the most
# relaxed last-resort one -- must stay at or above this floor, or the
# "last resort" fallback can hand back a path through a space the robot
# genuinely cannot drive through.
ROBOT_MIN_CLEARANCE_M = AXLE_LENGTH / 2.0 + 0.02
# Inflation/clearance distances expressed in metres, then converted to
# cells for the current RESOLUTION -- so changing MAP_SIZE/RESOLUTION can
# never silently loosen or tighten the robot's real-world safety margin
# the way a fixed cell count would. Levels progressively relax from a
# generous safety margin down to (but never below) ROBOT_MIN_CLEARANCE_M.
ASTAR_INFLATION_LEVELS_M   = [max(0.20, ROBOT_MIN_CLEARANCE_M + 0.07),
                              max(0.16, ROBOT_MIN_CLEARANCE_M + 0.03),
                              ROBOT_MIN_CLEARANCE_M]
ASTAR_INFLATION_LEVELS     = [max(1, round(m / RESOLUTION)) for m in ASTAR_INFLATION_LEVELS_M]
ASTAR_EXPANSION_M          = 0.10
ASTAR_EXPANSION_PIXELS     = max(1, round(ASTAR_EXPANSION_M / RESOLUTION))
ASTAR_FRONTIER_INFLATION_M = max(0.1333, ROBOT_MIN_CLEARANCE_M)
ASTAR_FRONTIER_INFLATION   = max(1, round(ASTAR_FRONTIER_INFLATION_M / RESOLUTION))
ASTAR_MIN_CLEARANCE_M      = 0.10
ASTAR_MIN_CLEARANCE_PIXELS = max(1, round(ASTAR_MIN_CLEARANCE_M / RESOLUTION))
ASTAR_COST_WEIGHT          = 9.0

# ── Exploration timing ───────────────────────────────────────────────────────
EXPLORATION_FRONTIER_SELECTION_FREQ = 5
EXPLORATION_START_FRONTIER_AFTER    = 50

# ── DWA planner ──────────────────────────────────────────────────────────────
DWA_VELOCITY_SAMPLES              = [0.05, 0.10, 0.15, 0.20, 0.25, 0.27]
DWA_ANGULAR_SAMPLES               = [0, 1.5, -1.5, 2.5, -2.5, 3.0, -3.0, 3.5, -3.5, 4.0, -4.0]
DWA_HEADING_WEIGHT                = 4.0
DWA_DISTANCE_WEIGHT               = 3.5
DWA_SPEED_WEIGHT                  = 0.05
DWA_COST_MAP_WEIGHT               = 1.5
DWA_UNKNOWN_WEIGHT                = 1.2
DWA_COST_MAP_REJECT_THRESHOLD     = 0.6

# ── Path following ───────────────────────────────────────────────────────────
PATH_FOLLOWING_TARGET_REACH_DISTANCE_M = 0.1333
PATH_FOLLOWING_TARGET_REACH_DISTANCE = max(1, round(PATH_FOLLOWING_TARGET_REACH_DISTANCE_M / RESOLUTION))  # cells

# ── Motor control ────────────────────────────────────────────────────────────
MOTOR_VELOCITY_FORWARD  =  6.28
MOTOR_VELOCITY_TURN     =  6.0

# ── PID alignment ────────────────────────────────────────────────────────────
ALIGN_COLUMN_KP              = 0.008
ALIGN_COLUMN_KD              = 0.002
ALIGN_COLUMN_ERROR_THRESHOLD = 20

# ── Colour detection (HSV) ───────────────────────────────────────────────────
COLOR_DETECTION_DEPTH_THRESHOLD = 80
BLUE_HSV_LOWER       = [100, 150, 50]
BLUE_HSV_UPPER       = [140, 255, 255]
YELLOW_HSV_LOWER     = [20,  100, 100]
YELLOW_HSV_UPPER     = [35,  255, 255]
GREEN_HSV_LOWER      = [36,  100, 100]
GREEN_HSV_UPPER      = [86,  255, 255]
GREEN_CARPET_DILATION_KERNEL_SIZE = 2
GREEN_CARPET_DILATION_ITERATIONS  = 0
# Max ground-plane distance (m) a green pixel may project to. Larger captures
# more of the carpet in a single frame (bigger plotted radius); too large lets
# inaccurate near-horizon pixels of a distant carpet project as false floor.
GREEN_CARPET_MAX_PROJECTION_DISTANCE = 1.5
# Centroid gate (cells) — carpet centroid must be this close to the robot to
# be marked. Scaled directly off the projection distance above so it always
# matches it in cells, regardless of MAP_SIZE/RESOLUTION.
GREEN_CARPET_MARK_MAX_MAP_DISTANCE = round(GREEN_CARPET_MAX_PROJECTION_DISTANCE / RESOLUTION)
# Hard per-cell gate: individual projected carpet cells farther than this
# (in metres) from the robot are dropped, so the carpet is never plotted
# from far away even if the centroid happens to fall within range.
GREEN_CARPET_MAX_CELL_DISTANCE_M = 1.6
GREEN_CARPET_MAX_CELL_MAP_DISTANCE = round(GREEN_CARPET_MAX_CELL_DISTANCE_M / RESOLUTION)
GREEN_CARPET_PATCH_PROXIMITY_M = 1.1667
GREEN_CARPET_PATCH_PROXIMITY_CELLS = round(GREEN_CARPET_PATCH_PROXIMITY_M / RESOLUTION)
GREEN_CARPET_CONFIRM_MIN_POINTS_RATIO = 0.35
GREEN_CARPET_FILL_HULL_MIN_POINTS = 20
GREEN_CARPET_FILL_CLOSE_KERNEL_CELLS = 5
GREEN_CARPET_FILL_DILATE_CELLS = 2

# ── Visualization filtering ──────────────────────────────────────────────────
# Minimum rendered frontier-region AREA, expressed in m² so it doesn't
# shrink to noise-sized specks (or balloon to swallow real small frontiers)
# just because RESOLUTION changed.
FRONTIER_RENDER_MIN_AREA_M2 = 18 * (10.0 / 300) ** 2
FRONTIER_RENDER_MIN_CELLS = max(1, round(FRONTIER_RENDER_MIN_AREA_M2 / (RESOLUTION ** 2)))
MAP_RENDER_FPS = 30

# ── Realtime obstacle-replan lookahead ───────────────────────────────────────
# Frontier target scoring (_frontier_info_gain, _score_frontier, etc.) and
# navigate_frontier/the realtime-planner watchdog now use the exact literal
# cell counts from the reference implementation directly, not constants
# derived here. The lookahead distances below are still expressed in
# metres and converted via RESOLUTION, since they remain in active use by
# _path_blocked/_path_blocked_from_pose/_path_usable_from_pose's own
# default parameters and by follow_final_path.
_FRONTIER_TUNING_BASIS_RES = 10.0 / 300
PATH_BLOCKED_LOOKAHEAD_M = 12 * _FRONTIER_TUNING_BASIS_RES
PATH_BLOCKED_LOOKAHEAD_CELLS = max(1, round(PATH_BLOCKED_LOOKAHEAD_M / RESOLUTION))
PATH_USABLE_LOOKAHEAD_M = 14 * _FRONTIER_TUNING_BASIS_RES
PATH_USABLE_LOOKAHEAD_CELLS = max(1, round(PATH_USABLE_LOOKAHEAD_M / RESOLUTION))
# Used by follow_final_path's own blocked/usable checks on the pillar-to-
# pillar final path.
FINAL_PATH_LOOKAHEAD_M = 24 * _FRONTIER_TUNING_BASIS_RES
FINAL_PATH_LOOKAHEAD_CELLS = max(1, round(FINAL_PATH_LOOKAHEAD_M / RESOLUTION))

# ── Sensor ───────────────────────────────────────────────────────────────────
# The Astra depth camera (Astra.proto, RangeFinder node) has a hard
# minRange of 0.6 m -- it reports NO depth at all closer than this,
# regardless of scene content. This is the exact reason a floating wall
# stops being detected once the robot gets very close to it: the depth
# camera goes physically blind, not a tuning issue. The front IR range
# sensors (fl_range/fr_range) cover this blind gap instead, since they
# read down to near 0 m -- see MyRobot._ir_floating_wall_points_local.
DEPTH_CAMERA_MIN_RANGE_M = 0.6
# A confirmed floating cell may only be cleared while it sits inside the
# camera's genuinely valid (non-blind) range -- a small margin past the
# hard minRange above, so a cell right at the boundary is never cleared on
# the strength of ambiguous near-limit readings.
DEPTH_CLEAR_MIN_RANGE_M = DEPTH_CAMERA_MIN_RANGE_M + 0.05
COST_MAP_UPDATE_INTERVAL = 0.20
# Upscales only the final rendered RGB image for viewing (cv2.resize, once
# per rendered frame) -- decoupled from MAP_SIZE so the on-screen map can
# be a comfortable, readable size (500*2 = 1000 px) without the grid
# itself (and every per-frame numpy op over it) paying an 800+-cell cost.
MAP_RENDER_SCALE = 2
CAMERA_LOOP_INTERVAL = 0.25
CAMERA_DETECTION_SCALE = 0.4
# ── Floating-wall height band (Nav2 ObstacleLayer / STVL style) ─────────────
# Every depth point whose height falls in [GROUND_EPSILON_M,
# ROBOT_CLEARANCE_HEIGHT_M] is a collision hazard, full stop -- see
# MyRobot._depth_obstacle_points_local. No per-object "is this floating"
# classification is needed: LiDAR-visible walls fall in this band too and
# get marked redundantly (harmless -- map.py lets a LiDAR OBSTACLE win over
# DEPTH_OBSTACLE), while a genuinely floating wall's face is marked simply
# because its 3D points land in the band, whether or not the surface
# happens to also touch the ground somewhere else in view.
GROUND_EPSILON_M = 0.03
# Robot's tallest physical point plus a small margin for sensor/odometry
# noise. The tallest point (0.2135 m) comes from Astra.proto's own
# boundingObject (the camera housing box, NOT the RangeFinder element):
# housing mount z (0.165) + local box center z (0.034) + half box height
# (0.029/2 = 0.0145) = 0.2135 m — the tallest solid part of the robot in
# the simulator's own collision geometry, not guessed or reverse-engineered
# from any one maze's wall placement. A point above this height is
# something the robot fits under and must NOT be marked -- this single
# threshold IS the passability decision (no separate ratio/quantile check
# needed downstream), which is also why the two tall hanging walls in this
# maze that differ by only 5 cm of clearance (one blocking, one passable)
# are still classified correctly: any point at/below this line blocks,
# every point above it doesn't, decided per point at the moment it's seen.
ROBOT_CLEARANCE_HEIGHT_M = 0.23
# Depth-image "flying pixel" silhouette artifacts appear at object edges
# where the sensor interpolates between a near surface and a much farther
# background; a genuine surface point's depth is close to its immediate
# neighbours. Points whose neighbour differs by more than this are dropped
# before height is even computed. This is a FLOOR, not the only bound —
# see DEPTH_EDGE_RELATIVE_FRACTION below.
DEPTH_EDGE_DISCONTINUITY_M = 0.3
# A wall viewed nearly edge-on (its face close to parallel with the
# viewing ray, e.g. a "vertical" wall the robot mostly passes alongside
# rather than facing) has a genuinely steep depth gradient across
# neighbouring pixels purely from perspective, growing with distance —
# not a silhouette artifact, just a real surface seen at a shallow angle.
# A flat 0.3 m cutoff rejects that legitimate gradient at longer range,
# which is why a wall in that orientation could fail to accumulate ANY
# surviving points/votes even though it was clearly in view. The actual
# per-pixel threshold used is max(DEPTH_EDGE_DISCONTINUITY_M, this
# fraction x the pixel's own depth) — scaling with distance follows a real
# surface's perspective gradient, while a genuine silhouette jump (near
# object against a much farther background) is still far larger in
# proportion than this fraction at any range and gets rejected either way.
DEPTH_EDGE_RELATIVE_FRACTION = 0.15
# Frustum-gated clearing (see MyRobot._frustum_clear_floating) requires
# this many DISTINCT frames of the camera's own line of sight passing
# clean through a confirmed cell before it is removed -- clearing must be
# much harder to trigger than marking, since a single noisy frame
# misjudging a wall's edge must never be able to erase a real wall.
DEPTH_CLEAR_CONFIRMATIONS = 5
# A candidate floating-wall cell within this distance of an already
# LiDAR-confirmed OBSTACLE is dropped before it ever earns a vote. This is
# the recurring "wall behind a floating wall gets marked floating" case:
# the depth camera's own small height/position error over distance can
# place its computed world position for a real, LiDAR-visible wall's
# surface a cell or two away from where LiDAR itself placed that same
# surface -- close enough that they are clearly the same physical wall,
# not two different objects, so the camera reading is redundant at best
# and should just defer to LiDAR's more accurate fix on it entirely,
# rather than getting voted in as its own separate (wrongly labelled)
# floating cell nearby.
FLOATING_WALL_NEAR_LIDAR_VETO_M = 0.05
FLOATING_WALL_NEAR_LIDAR_VETO_CELLS = max(1, round(FLOATING_WALL_NEAR_LIDAR_VETO_M / RESOLUTION))

# ── Floating wall detection (vote-confirm-freeze) ────────────────────────────
# A map cell is only trusted as a floating wall once the depth camera has
# independently seen it as floating this many separate frames. Below the
# threshold it is a candidate only and is never drawn or blocked on — this
# is what filters out one-off sensor noise instead of a shape/size heuristic.
FLOATING_WALL_CONFIRM_VOTES = 4
# A wall seen nearly edge-on (a "vertical" wall the robot mostly passes
# alongside rather than faces) is often only visible to the depth camera
# for a brief, close-range window -- sometimes too few frames to reach
# FLOATING_WALL_CONFIRM_VOTES before the robot moves past it or the wall
# leaves the FOV. Depth measurements taken this close also carry much less
# angular/position error than far ones (the same per-pixel angular
# uncertainty maps to far less real-world distance error up close), so
# trusting fewer independent votes for a cell first seen this close is a
# reduction in required SAMPLE COUNT, not in required CONFIDENCE.
FLOATING_WALL_CLOSE_RANGE_M = 1.2
FLOATING_WALL_CONFIRM_VOTES_CLOSE = 2
# Vote counter ceiling per cell (just prevents unbounded growth; irrelevant
# once a cell has already crossed FLOATING_WALL_CONFIRM_VOTES).
FLOATING_WALL_VOTE_CAP = 8
# Once a cell crosses the confirmation threshold above, it is frozen forever:
# this pipeline never re-fits, erases, or moves a confirmed cell again. A
# newly confirmed cell whose nearest already-mapped wall (lidar OBSTACLE,
# another confirmed floating cell, or CLOSED) is within this many cells gets
# a straight bridge drawn to it once, at confirmation time only — real walls
# in this maze always meet flush, so a leftover few-cell sliver is a mapping
# gap, not a real passage, and A* would otherwise route straight through it.
FLOATING_WALL_BRIDGE_RADIUS_M = 0.10
FLOATING_WALL_BRIDGE_RADIUS_CELLS = max(1, round(FLOATING_WALL_BRIDGE_RADIUS_M / RESOLUTION))
# A coverage gap WITHIN one physical floating-wall panel (its two visible
# ends got confirmed, but sampling/occlusion/a brief viewing window never
# confirmed the cells between them) can be much wider than the seam-bridge
# radius above, which is deliberately kept small so it can never close off
# an actually-usable opening. This second, larger threshold is still
# provably safe to close unconditionally: it is capped at the robot's own
# physical width, so a gap narrower than this could never have been a
# real, driveable passage regardless -- the robot could not fit through it
# either way. Applied only along the same row or column (never a general
# radius), so it can only ever merge two points that are candidates for
# being the SAME straight wall run, not an unrelated object that merely
# happens to be nearby in some other direction.
FLOATING_WALL_GAP_CLOSE_MAX_M = AXLE_LENGTH + 0.05
FLOATING_WALL_GAP_CLOSE_MAX_CELLS = max(1, round(FLOATING_WALL_GAP_CLOSE_MAX_M / RESOLUTION))
# Once two cells of the SAME already-established wall group are confirmed,
# the straight run between them (per row/column) is filled in immediately —
# see MyRobot._floating_solidify_group. Capped to this many cells (~1.5 m at
# this map's resolution, matching the longest WallMedium box run used
# elsewhere in this maze) so a same-row/column false merge far away can
# never paint a wall clear across an unrelated part of the map.
FLOATING_WALL_MAX_SOLIDIFY_SPAN_CELLS = int(1.5 / RESOLUTION)

# ── Camera detection debounce ────────────────────────────────────────────────
CAMERA_SIGNAL_MIN_FRAMES = 2
CAMERA_GREEN_SIGNAL_COOLDOWN = 4.0
CAMERA_COLUMN_SIGNAL_COOLDOWN = 2.0

# ── Column mapping ───────────────────────────────────────────────────────────
COLUMN_COMMIT_MAX_MAP_DISTANCE_M = 0.6
COLUMN_COMMIT_MAX_MAP_DISTANCE = round(COLUMN_COMMIT_MAX_MAP_DISTANCE_M / RESOLUTION)
COLUMN_COMMIT_MIN_ESTIMATES = 2
