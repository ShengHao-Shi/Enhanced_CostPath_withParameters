"""
Cost-Aware Straightening LCP Algorithm (方案C)
================================================

This module keeps the standard 8-direction Dijkstra search from
``enhanced_lcp.py`` but **improves the post-processing straightening**
step.  Instead of only checking whether a shortcut line crosses NODATA
barriers (which can cause the straightened path to traverse high-cost
cells), this version computes the **accumulated cost** along each
proposed shortcut and only accepts the shortcut if its cost does not
exceed a configurable tolerance relative to the original grid-path cost
for the same segment.

Key differences from ``enhanced_lcp.py``:

* ``straighten_path()`` is replaced with ``cost_aware_straighten_path()``
  which compares shortcut cost vs. original path cost.
* A ``cost_tolerance`` parameter (default 1.05 = 5% overhead allowed)
  controls how much extra cost a shortcut may incur.
* The Dijkstra search, smoothing, and all other logic are identical.

Dependencies: numpy (required).
"""

import heapq
import math
from typing import Dict, List, Optional, Tuple

import numpy as np

# ---------------------------------------------------------------------------
# Direction definitions (same as enhanced_lcp)
# ---------------------------------------------------------------------------
DIRECTIONS: List[Tuple[int, int]] = [
    (-1, 0),   # 0: N
    (-1, 1),   # 1: NE
    (0, 1),    # 2: E
    (1, 1),    # 3: SE
    (1, 0),    # 4: S
    (1, -1),   # 5: SW
    (0, -1),   # 6: W
    (-1, -1),  # 7: NW
]

NUM_DIRS: int = len(DIRECTIONS)
NO_DIR: int = -1

_CURVATURE_AMPLIFIER: float = 5.0
_STRAIGHTNESS_PENALTY: float = 0.3


def turning_angle(dir_from: int, dir_to: int) -> float:
    """Return turning angle in degrees between two direction indices."""
    diff = abs(dir_from - dir_to)
    steps = min(diff, NUM_DIRS - diff)
    return steps * 45.0


_TURN_ANGLE_LUT: List[List[float]] = [
    [turning_angle(d_in, d_out) for d_out in range(8)]
    for d_in in range(8)
]


# ---------------------------------------------------------------------------
# Supercover line
# ---------------------------------------------------------------------------

def _supercover_line(
    r0: int, c0: int, r1: int, c1: int,
) -> List[Tuple[int, int]]:
    """Return all grid cells the line segment passes through."""
    cells: List[Tuple[int, int]] = []
    dr = abs(r1 - r0)
    dc = abs(c1 - c0)
    r, c = r0, c0
    sr = 1 if r0 < r1 else -1
    sc = 1 if c0 < c1 else -1
    err = dr - dc

    while True:
        cells.append((r, c))
        if r == r1 and c == c1:
            break
        e2 = 2 * err
        step_r = e2 > -dc
        step_c = e2 < dr

        if step_r and step_c:
            cells.append((r + sr, c))
            cells.append((r, c + sc))
            err -= dc
            r += sr
            err += dr
            c += sc
        elif step_r:
            err -= dc
            r += sr
        else:
            err += dr
            c += sc

    return cells


def _is_line_clear(
    p0: Tuple[int, int],
    p1: Tuple[int, int],
    cost_data: np.ndarray,
    rows: int,
    cols: int,
) -> bool:
    """Check if the line from *p0* to *p1* is free of barriers."""
    _isfinite = math.isfinite
    for r, c in _supercover_line(p0[0], p0[1], p1[0], p1[1]):
        if not (0 <= r < rows and 0 <= c < cols):
            return False
        val = float(cost_data[r, c])
        if not _isfinite(val) or val < 0:
            return False
    return True


# ---------------------------------------------------------------------------
# Cost helpers
# ---------------------------------------------------------------------------

def _cost_scale(cost_raster: np.ndarray) -> float:
    valid = cost_raster[np.isfinite(cost_raster)]
    valid = valid[valid >= 0]
    if valid.size == 0:
        return 1.0
    mean_val = float(np.mean(valid))
    return mean_val if mean_val > 0 else 1.0


def _min_cost(cost_raster: np.ndarray) -> float:
    valid = cost_raster[np.isfinite(cost_raster)]
    valid = valid[valid >= 0]
    if valid.size == 0:
        return 1.0
    return float(np.min(valid))


def _step_distances(cell_size: Tuple[float, float]) -> List[float]:
    cy, cx = cell_size
    return [math.sqrt((dr * cy) ** 2 + (dc * cx) ** 2) for dr, dc in DIRECTIONS]


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def _validate_params(
    cost_raster: np.ndarray,
    start: Tuple[int, int],
    end: Tuple[int, int],
    curvature_factor: float,
    max_turning_angle: float,
    distance_factor: float,
    straighten_factor: float,
    cost_tolerance: float,
) -> None:
    if cost_raster.ndim != 2:
        raise ValueError("cost_raster must be a 2-D array")
    if not 0.0 <= curvature_factor <= 1.0:
        raise ValueError(
            f"curvature_factor must be between 0.0 and 1.0, got {curvature_factor}"
        )
    if not 0.0 <= max_turning_angle <= 180.0:
        raise ValueError(
            f"max_turning_angle must be between 0.0 and 180.0, got {max_turning_angle}"
        )
    if not 0.0 <= distance_factor <= 1.0:
        raise ValueError(
            f"distance_factor must be between 0.0 and 1.0, got {distance_factor}"
        )
    if not 0.0 <= straighten_factor <= 0.5:
        raise ValueError(
            f"straighten_factor must be between 0.0 and 0.5, got {straighten_factor}"
        )
    if cost_tolerance < 1.0:
        raise ValueError(
            f"cost_tolerance must be >= 1.0, got {cost_tolerance}"
        )
    rows, cols = cost_raster.shape
    sr, sc = start
    er, ec = end
    if not (0 <= sr < rows and 0 <= sc < cols):
        raise ValueError(
            f"Start point {start} is outside raster bounds ({rows}, {cols})"
        )
    if not (0 <= er < rows and 0 <= ec < cols):
        raise ValueError(
            f"End point {end} is outside raster bounds ({rows}, {cols})"
        )
    if not np.isfinite(cost_raster[sr, sc]):
        raise ValueError(f"Start point {start} is on an invalid (NaN/Inf) cell")
    if not np.isfinite(cost_raster[er, ec]):
        raise ValueError(f"End point {end} is on an invalid (NaN/Inf) cell")


# ---------------------------------------------------------------------------
# Standard Dijkstra (no direction tracking) — same as enhanced_lcp
# ---------------------------------------------------------------------------

def _dijkstra_standard(
    cost_raster: np.ndarray,
    start: Tuple[int, int],
    end: Tuple[int, int],
    distance_factor: float,
    cell_size: Tuple[float, float],
    straighten_factor: float,
    cost_tolerance: float,
) -> Dict:
    rows, cols = cost_raster.shape
    sr, sc = start
    er, ec = end

    step_dists = _step_distances(cell_size)
    scale = _cost_scale(cost_raster)
    dist_weight = distance_factor * scale
    straight_weight = _STRAIGHTNESS_PENALTY * scale

    cy, cx = cell_size
    h_weight = _min_cost(cost_raster) + dist_weight

    r_idx = np.arange(rows, dtype=np.float64)
    c_idx = np.arange(cols, dtype=np.float64)
    h_map = (np.sqrt(((r_idx[:, None] - er) * cy) ** 2
                     + ((c_idx[None, :] - ec) * cx) ** 2)
             * h_weight).astype(np.float32)

    cost_data = np.ascontiguousarray(cost_raster, dtype=np.float64)

    best = np.full((rows, cols), np.inf, dtype=np.float32)
    best[sr, sc] = 0.0
    parent_dir = np.full((rows, cols), -1, dtype=np.int8)

    _heappop = heapq.heappop
    _heappush = heapq.heappush
    _isfinite = math.isfinite
    _dirs = DIRECTIONS

    counter = 0
    pq: List = [(float(h_map[sr, sc]), counter, 0.0, sr, sc)]

    while pq:
        _, _, g, r, c = _heappop(pq)
        if g > best[r, c]:
            continue
        if r == er and c == ec:
            break

        d_in = int(parent_dir[r, c])

        for d in range(NUM_DIRS):
            dr, dc = _dirs[d]
            nr, nc = r + dr, c + dc
            if not (0 <= nr < rows and 0 <= nc < cols):
                continue
            cell_val = float(cost_data[nr, nc])
            if not _isfinite(cell_val) or cell_val < 0:
                continue
            sd = step_dists[d]
            new_cost = g + cell_val * sd + dist_weight * sd

            if d_in >= 0 and d != d_in:
                str_dr, str_dc = _dirs[d_in]
                str_r, str_c = r + str_dr, c + str_dc
                if 0 <= str_r < rows and 0 <= str_c < cols:
                    str_val = float(cost_data[str_r, str_c])
                    if _isfinite(str_val) and str_val >= 0:
                        similarity = max(0.0, 1.0 - abs(cell_val - str_val) / scale)
                        new_cost += similarity * straight_weight * sd

            if new_cost < best[nr, nc]:
                best[nr, nc] = new_cost
                parent_dir[nr, nc] = d
                counter += 1
                g_stored = float(best[nr, nc])
                _heappush(pq, (g_stored + float(h_map[nr, nc]), counter, g_stored, nr, nc))

    if not np.isfinite(best[er, ec]):
        raise RuntimeError("No path found between start and end")

    return _build_result(parent_dir, start, end, best, float(best[er, ec]),
                         cell_size, cost_data, straighten_factor,
                         cost_tolerance)


# ---------------------------------------------------------------------------
# Direction-aware Dijkstra (curvature support)
# ---------------------------------------------------------------------------

def _dijkstra_with_direction(
    cost_raster: np.ndarray,
    start: Tuple[int, int],
    end: Tuple[int, int],
    curvature_factor: float,
    max_turning_angle: float,
    distance_factor: float,
    cell_size: Tuple[float, float],
    straighten_factor: float,
    cost_tolerance: float,
) -> Dict:
    rows, cols = cost_raster.shape
    sr, sc = start
    er, ec = end

    step_dists = _step_distances(cell_size)
    scale = _cost_scale(cost_raster)
    curv_weight = curvature_factor * _CURVATURE_AMPLIFIER * scale
    dist_weight = distance_factor * scale
    straight_weight = _STRAIGHTNESS_PENALTY * scale

    cy, cx = cell_size
    h_weight = _min_cost(cost_raster) + dist_weight

    r_idx = np.arange(rows, dtype=np.float64)
    c_idx = np.arange(cols, dtype=np.float64)
    h_map = (np.sqrt(((r_idx[:, None] - er) * cy) ** 2
                     + ((c_idx[None, :] - ec) * cx) ** 2)
             * h_weight).astype(np.float32)

    cost_data = np.ascontiguousarray(cost_raster, dtype=np.float64)

    n_states = NUM_DIRS + 1
    best = np.full((rows, cols, n_states), np.inf, dtype=np.float32)
    best[sr, sc, NUM_DIRS] = 0.0

    parent_d = np.full((rows, cols, n_states), -1, dtype=np.int8)

    _heappop = heapq.heappop
    _heappush = heapq.heappush
    _isfinite = math.isfinite
    _dirs = DIRECTIONS
    _turn_lut = _TURN_ANGLE_LUT
    _no_dir = NO_DIR
    _curv_div = curv_weight / 180.0

    counter = 0
    pq: List = [(float(h_map[sr, sc]), counter, 0.0, sr, sc, _no_dir)]

    found = False
    end_state: Optional[Tuple[int, int, int]] = None

    while pq:
        _, _, g, r, c, d_in = _heappop(pq)
        d_idx = d_in if d_in >= 0 else NUM_DIRS

        if g > best[r, c, d_idx]:
            continue
        if r == er and c == ec:
            found = True
            end_state = (r, c, d_in)
            break

        for d_out in range(NUM_DIRS):
            dr, dc = _dirs[d_out]
            nr, nc = r + dr, c + dc
            if not (0 <= nr < rows and 0 <= nc < cols):
                continue
            cell_val = float(cost_data[nr, nc])
            if not _isfinite(cell_val) or cell_val < 0:
                continue

            sd = step_dists[d_out]
            base = cell_val * sd

            curv_penalty = 0.0
            if d_in != _no_dir:
                angle = _turn_lut[d_in][d_out]
                if angle > max_turning_angle:
                    continue
                curv_penalty = _curv_div * angle * sd

            straightness_penalty = 0.0
            if d_in != _no_dir and d_out != d_in:
                str_dr, str_dc = _dirs[d_in]
                str_r, str_c = r + str_dr, c + str_dc
                if (0 <= str_r < rows and 0 <= str_c < cols):
                    str_val = float(cost_data[str_r, str_c])
                    if _isfinite(str_val) and str_val >= 0:
                        similarity = max(0.0, 1.0 - abs(cell_val - str_val) / scale)
                        straightness_penalty = similarity * straight_weight * sd

            new_cost = g + base + curv_penalty + straightness_penalty + dist_weight * sd

            nd_idx = d_out
            if new_cost < best[nr, nc, nd_idx]:
                best[nr, nc, nd_idx] = new_cost
                parent_d[nr, nc, nd_idx] = d_in
                counter += 1
                g_stored = float(best[nr, nc, nd_idx])
                _heappush(pq, (g_stored + float(h_map[nr, nc]), counter, g_stored, nr, nc, d_out))

    if not found:
        raise RuntimeError("No path found between start and end")

    return _build_result_directed(parent_d, end_state, best, cell_size,
                                   cost_data, straighten_factor,
                                   cost_tolerance)


# ---------------------------------------------------------------------------
# Path reconstruction
# ---------------------------------------------------------------------------

def _build_result(
    parent_dir: np.ndarray,
    start: Tuple[int, int],
    end: Tuple[int, int],
    best: np.ndarray,
    total_cost: float,
    cell_size: Tuple[float, float],
    cost_raster: np.ndarray,
    straighten_factor: float,
    cost_tolerance: float,
) -> Dict:
    """Reconstruct path for the standard (non-directional) Dijkstra."""
    path: List[Tuple[int, int]] = []
    r, c = end
    while True:
        path.append((int(r), int(c)))
        d = int(parent_dir[r, c])
        if d < 0:
            break
        r = r - DIRECTIONS[d][0]
        c = c - DIRECTIONS[d][1]
    path.reverse()

    cy, cx = cell_size
    path_length = 0.0
    directions: List[int] = []
    for i in range(1, len(path)):
        r1, c1 = path[i - 1]
        r2, c2 = path[i]
        dr, dc = r2 - r1, c2 - c1
        path_length += math.sqrt((dr * cy) ** 2 + (dc * cx) ** 2)
        directions.append(DIRECTIONS.index((dr, dc)))

    turning_angles: List[float] = []
    for i in range(1, len(directions)):
        turning_angles.append(turning_angle(directions[i - 1], directions[i]))

    straightened = cost_aware_straighten_path(
        path, cost_raster, best, cell_size, straighten_factor, cost_tolerance
    )

    return {
        "path": path,
        "straightened_path": straightened,
        "smoothed_path": _smooth_path_nodata_safe(straightened, cost_raster),
        "total_cost": total_cost,
        "path_length": path_length,
        "directions": directions,
        "turning_angles": turning_angles,
    }


def _build_result_directed(
    parent_d: np.ndarray,
    end_state: Tuple[int, int, int],
    best: np.ndarray,
    cell_size: Tuple[float, float],
    cost_raster: np.ndarray,
    straighten_factor: float,
    cost_tolerance: float,
) -> Dict:
    """Reconstruct path for the direction-aware Dijkstra."""
    states: List[Tuple[int, int, int]] = []
    r, c, d = end_state
    d_idx = d if d >= 0 else NUM_DIRS
    while True:
        states.append((int(r), int(c), int(d)))
        if d_idx >= NUM_DIRS:
            break
        nd = int(parent_d[r, c, d_idx])
        pr = r - DIRECTIONS[d_idx][0]
        pc = c - DIRECTIONS[d_idx][1]
        d_idx = nd if nd >= 0 else NUM_DIRS
        r, c, d = pr, pc, nd
    states.reverse()

    path = [(r, c) for r, c, _ in states]
    directions = [d for _, _, d in states if d != NO_DIR]

    cy, cx = cell_size
    path_length = 0.0
    for i in range(1, len(path)):
        r1, c1 = path[i - 1]
        r2, c2 = path[i]
        dr, dc = r2 - r1, c2 - c1
        path_length += math.sqrt((dr * cy) ** 2 + (dc * cx) ** 2)

    turning_angles: List[float] = []
    for i in range(1, len(directions)):
        turning_angles.append(turning_angle(directions[i - 1], directions[i]))

    er, ec, ed = end_state
    ed_idx = ed if ed >= 0 else NUM_DIRS
    total_cost = float(best[er, ec, ed_idx])

    # Build a 2D best array from the 3D one (min across directions)
    best_2d = np.min(best, axis=2)

    straightened = cost_aware_straighten_path(
        path, cost_raster, best_2d, cell_size, straighten_factor, cost_tolerance
    )

    return {
        "path": path,
        "straightened_path": straightened,
        "smoothed_path": _smooth_path_nodata_safe(straightened, cost_raster),
        "total_cost": total_cost,
        "path_length": path_length,
        "directions": directions,
        "turning_angles": turning_angles,
    }


# ---------------------------------------------------------------------------
# Cost-aware straightening (the key improvement over enhanced_lcp)
# ---------------------------------------------------------------------------

def _line_cost(
    p0: Tuple[int, int],
    p1: Tuple[int, int],
    cost_data: np.ndarray,
    rows: int,
    cols: int,
    cell_size: Tuple[float, float],
) -> float:
    """Compute cost of travelling along the straight line from p0 to p1.

    Returns ``inf`` if the line crosses a barrier.
    """
    return _line_cost_or_inf(p0, p1, cost_data, rows, cols, cell_size)


def _line_cost_or_inf(
    p0: Tuple[int, int],
    p1: Tuple[int, int],
    cost_data: np.ndarray,
    rows: int,
    cols: int,
    cell_size: Tuple[float, float],
) -> float:
    """Check line clearance and compute cost in a single pass.

    Combines the logic of ``_is_line_clear`` and ``_line_cost`` to avoid
    computing the supercover line twice for the same segment.

    Returns ``inf`` if the line crosses a barrier or out-of-bounds cell,
    otherwise returns ``avg_cost * euclidean_distance``.
    """
    _isfinite = math.isfinite
    cells = _supercover_line(p0[0], p0[1], p1[0], p1[1])
    n = len(cells)
    if n == 0:
        return 0.0

    total_cost_sum = 0.0
    for r, c in cells:
        if not (0 <= r < rows and 0 <= c < cols):
            return float("inf")
        val = float(cost_data[r, c])
        if not _isfinite(val) or val < 0:
            return float("inf")
        total_cost_sum += val

    avg_cost = total_cost_sum / n
    cy, cx = cell_size
    dr = (p1[0] - p0[0]) * cy
    dc = (p1[1] - p0[1]) * cx
    dist = math.sqrt(dr * dr + dc * dc)

    return avg_cost * dist


def cost_aware_straighten_path(
    path: List[Tuple[int, int]],
    cost_raster: np.ndarray,
    best: np.ndarray,
    cell_size: Tuple[float, float],
    straighten_factor: float = 0.3,
    cost_tolerance: float = 1.05,
) -> List[Tuple[float, float]]:
    """Straighten path with cost awareness.

    Unlike ``enhanced_lcp.straighten_path()`` which only checks NODATA
    barriers, this version also compares the cost of the shortcut line
    against the cost of the original grid path segment.  A shortcut is
    only accepted if:

    1. The line does not cross any NODATA / barrier cells, **AND**
    2. The estimated cost along the shortcut line is at most
       ``cost_tolerance`` times the original Dijkstra cost for the
       same start→end segment.

    Parameters
    ----------
    path : list[tuple[int, int]]
        Grid-cell path from pathfinding.
    cost_raster : numpy.ndarray
        The cost raster.
    best : numpy.ndarray
        2-D array of best known cost-to-reach for each cell (from Dijkstra).
    cell_size : tuple[float, float]
        ``(y_size, x_size)`` of each raster cell.
    straighten_factor : float
        Controls lookahead distance (0.0 – 0.5).
    cost_tolerance : float
        Maximum ratio of shortcut cost to original path cost (>= 1.0).
        1.0 = only accept shortcuts that are exactly as cheap or cheaper.
        1.05 = accept shortcuts up to 5% more expensive (default).

    Returns
    -------
    list[tuple[float, float]]
        Straightened path as float coordinates.
    """
    if len(path) <= 2:
        return [(float(r), float(c)) for r, c in path]

    if straighten_factor <= 0.0:
        return [(float(r), float(c)) for r, c in path]

    rows, cols = cost_raster.shape
    # Accept pre-converted cost_data if the caller already has one;
    # otherwise convert here.  This avoids a redundant copy when called
    # from _build_result / _build_result_directed.
    if cost_raster.dtype == np.float64 and cost_raster.flags["C_CONTIGUOUS"]:
        cost_data = cost_raster
    else:
        cost_data = np.ascontiguousarray(cost_raster, dtype=np.float64)

    n = len(path)
    max_skip = max(2, int(straighten_factor * (n - 1)))

    # Pre-compute cumulative cost along the original path
    # so we can quickly compute the cost of any sub-segment.
    cum_cost = [0.0] * n
    cy, cx = cell_size
    for k in range(1, n):
        r1, c1 = path[k - 1]
        r2, c2 = path[k]
        dr, dc = r2 - r1, c2 - c1
        sd = math.sqrt((dr * cy) ** 2 + (dc * cx) ** 2)
        cell_val = float(cost_data[r2, c2])
        if not math.isfinite(cell_val) or cell_val < 0:
            cell_val = 0.0
        cum_cost[k] = cum_cost[k - 1] + cell_val * sd

    result_indices: List[int] = [0]
    i = 0

    while i < n - 1:
        j_limit = min(n - 1, i + max_skip)
        best_j = i + 1
        for j in range(j_limit, i + 1, -1):
            # Combined barrier check + cost computation in one pass
            # (avoids computing the supercover line twice).
            shortcut_cost = _line_cost_or_inf(
                path[i], path[j], cost_data, rows, cols, cell_size
            )
            if math.isinf(shortcut_cost):
                continue

            # Shortcut cost must be within tolerance of original
            orig_segment_cost = cum_cost[j] - cum_cost[i]

            # Avoid division by zero for zero-cost segments
            if orig_segment_cost <= 0:
                # Zero-cost segment: accept any clear shortcut
                best_j = j
                break

            if shortcut_cost <= orig_segment_cost * cost_tolerance:
                best_j = j
                break

        result_indices.append(best_j)
        i = best_j

    return [(float(path[k][0]), float(path[k][1])) for k in result_indices]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def cost_aware_least_cost_path(
    cost_raster: np.ndarray,
    start: Tuple[int, int],
    end: Tuple[int, int],
    curvature_factor: float = 0.0,
    max_turning_angle: float = 180.0,
    distance_factor: float = 0.0,
    straighten_factor: float = 0.3,
    cost_tolerance: float = 1.05,
    cell_size: Tuple[float, float] = (1.0, 1.0),
) -> Dict:
    """Compute an LCP with cost-aware post-processing straightening.

    This function uses the same 8-direction Dijkstra search as
    ``enhanced_lcp.enhanced_least_cost_path()``, but replaces the
    NODATA-only straightening with a cost-aware version that only
    accepts shortcuts whose cost is within ``cost_tolerance`` of the
    original grid-path cost.

    Parameters
    ----------
    cost_raster : numpy.ndarray
        2-D array of traversal costs.
    start : tuple[int, int]
        ``(row, col)`` of the start cell.
    end : tuple[int, int]
        ``(row, col)`` of the end cell.
    curvature_factor : float, optional
        Soft penalty weight for sharp turns (0.0 – 1.0, default 0.0).
    max_turning_angle : float, optional
        Hard upper limit on turning angle (0 – 180, default 180).
    distance_factor : float, optional
        Weight for path length (0.0 – 1.0, default 0.0).
    straighten_factor : float, optional
        Lookahead for straightening (0.0 – 0.5, default 0.3).
    cost_tolerance : float, optional
        Maximum shortcut-cost / original-cost ratio (>= 1.0, default 1.05).
        * 1.0 — only accept shortcuts that are equally cheap or cheaper.
        * 1.05 — accept shortcuts up to 5% more expensive.
        * 2.0 — accept shortcuts up to 100% more expensive.
    cell_size : tuple[float, float], optional
        ``(y_size, x_size)`` of each raster cell.

    Returns
    -------
    dict
        Same keys as ``enhanced_lcp.enhanced_least_cost_path()``.
    """
    _validate_params(cost_raster, start, end, curvature_factor,
                     max_turning_angle, distance_factor, straighten_factor,
                     cost_tolerance)

    use_curvature = curvature_factor > 0.0 or max_turning_angle < 180.0

    if use_curvature:
        return _dijkstra_with_direction(
            cost_raster, start, end,
            curvature_factor, max_turning_angle, distance_factor, cell_size,
            straighten_factor, cost_tolerance,
        )
    return _dijkstra_standard(
        cost_raster, start, end, distance_factor, cell_size,
        straighten_factor, cost_tolerance,
    )


# ---------------------------------------------------------------------------
# Smoothing (same as enhanced_lcp)
# ---------------------------------------------------------------------------

def smooth_path(
    path,
    iterations: int = 3,
) -> List[Tuple[float, float]]:
    """Chaikin corner-cutting smoothing."""
    if len(path) <= 2:
        return [(float(r), float(c)) for r, c in path]

    pts: List[Tuple[float, float]] = [(float(r), float(c)) for r, c in path]

    for _ in range(iterations):
        if len(pts) <= 2:
            break
        new_pts: List[Tuple[float, float]] = [pts[0]]
        for i in range(len(pts) - 1):
            r0, c0 = pts[i]
            r1, c1 = pts[i + 1]
            qr = 0.75 * r0 + 0.25 * r1
            qc = 0.75 * c0 + 0.25 * c1
            rr = 0.25 * r0 + 0.75 * r1
            rc = 0.25 * c0 + 0.75 * c1
            new_pts.append((qr, qc))
            new_pts.append((rr, rc))
        new_pts.append(pts[-1])
        pts = new_pts

    return pts


def _smooth_path_nodata_safe(
    straightened: List[Tuple[float, float]],
    cost_raster: np.ndarray,
    iterations: int = 3,
) -> List[Tuple[float, float]]:
    """Smooth via Chaikin, falling back when NODATA is crossed."""
    smoothed = smooth_path(straightened, iterations)

    if len(smoothed) <= 2:
        return smoothed

    rows, cols = cost_raster.shape
    cost_data = np.ascontiguousarray(cost_raster, dtype=np.float64)
    _isfinite = math.isfinite

    for i in range(len(smoothed) - 1):
        r0 = int(round(smoothed[i][0]))
        c0 = int(round(smoothed[i][1]))
        r1 = int(round(smoothed[i + 1][0]))
        c1 = int(round(smoothed[i + 1][1]))
        r0 = max(0, min(rows - 1, r0))
        c0 = max(0, min(cols - 1, c0))
        r1 = max(0, min(rows - 1, r1))
        c1 = max(0, min(cols - 1, c1))
        for r, c in _supercover_line(r0, c0, r1, c1):
            if not (0 <= r < rows and 0 <= c < cols):
                return list(straightened)
            val = float(cost_data[r, c])
            if not _isfinite(val) or val < 0:
                return list(straightened)

    return smoothed
