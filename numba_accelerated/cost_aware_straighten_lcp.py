"""
Numba-Accelerated Cost-Aware Straightening LCP Algorithm
=========================================================

Drop-in replacement for ``pure_python.cost_aware_straighten_lcp`` with
the Dijkstra core loops and supercover-line helpers compiled via
**Numba JIT** for 20–50× speedup on large rasters (e.g. 7000×7000).

The public API is identical:

    >>> from numba_accelerated.cost_aware_straighten_lcp import (
    ...     cost_aware_least_cost_path,
    ... )

Numba-compiled functions
------------------------
* ``_dijkstra_core_standard``  — standard 8-dir A* / Dijkstra
* ``_dijkstra_core_directed``  — direction-aware variant (curvature)
* ``_supercover_line_nb``      — supercover rasterisation
* ``_line_cost_or_inf_nb``     — barrier-check + cost in one pass
* ``_straighten_core_nb``      — cost-aware straightening inner loop

Everything else (validation, path reconstruction, smoothing, public API)
is plain Python and delegates to the pure-Python module where possible.

Dependencies: numpy, numba.
"""

import math
from typing import Dict, List, Optional, Tuple

import numpy as np
import numba
from numba import njit, int32, int64, float64, float32, boolean, types
from numba.typed import List as NumbaList

# Re-use constants and helpers from the pure-Python module
from pure_python.cost_aware_straighten_lcp import (
    DIRECTIONS,
    NUM_DIRS,
    NO_DIR,
    _CURVATURE_AMPLIFIER,
    _STRAIGHTNESS_PENALTY,
    turning_angle,
    _TURN_ANGLE_LUT,
    _cost_scale,
    _min_cost,
    _step_distances,
    _validate_params,
    smooth_path,
    _smooth_path_nodata_safe,
    _is_line_clear,
    _supercover_line,
    _line_cost,
    _line_cost_or_inf,
    cost_aware_straighten_path,
)

# ---------------------------------------------------------------------------
# Direction arrays for Numba (int32 2-D array, shape (8, 2))
# ---------------------------------------------------------------------------
_DIRS_ARRAY = np.array(DIRECTIONS, dtype=np.int32)

# Turn angle LUT as 2-D float64 array
_TURN_LUT_ARRAY = np.array(_TURN_ANGLE_LUT, dtype=np.float64)

# ---------------------------------------------------------------------------
# Numba-compiled min-heap (array-based binary heap)
# ---------------------------------------------------------------------------
# Heap element layout:
#   [f_score, counter, g_score, row, col]            — standard (5 cols)
#   [f_score, counter, g_score, row, col, dir_in]    — directed  (6 cols)
#
# We use a pre-allocated float64 array and maintain the heap invariant
# manually, since Numba does not support Python's heapq.
# ---------------------------------------------------------------------------

@njit(cache=True)
def _heap_parent(i):
    return (i - 1) >> 1

@njit(cache=True)
def _heap_left(i):
    return (i << 1) + 1

@njit(cache=True)
def _heap_right(i):
    return (i << 1) + 2

@njit(cache=True)
def _heap_push(heap, size, element):
    """Push *element* (1-D array) onto the heap; return new size."""
    pos = size
    heap[pos, :] = element
    # Sift up
    while pos > 0:
        parent = _heap_parent(pos)
        if heap[pos, 0] < heap[parent, 0]:
            # Swap
            for k in range(heap.shape[1]):
                tmp = heap[pos, k]
                heap[pos, k] = heap[parent, k]
                heap[parent, k] = tmp
            pos = parent
        else:
            break
    return size + 1

@njit(cache=True)
def _heap_pop(heap, size):
    """Pop the smallest element; return (element_copy, new_size)."""
    top = heap[0].copy()
    new_size = size - 1
    # Move last element to root
    heap[0, :] = heap[new_size, :]
    # Sift down
    pos = 0
    while True:
        left = _heap_left(pos)
        right = _heap_right(pos)
        smallest = pos
        if left < new_size and heap[left, 0] < heap[smallest, 0]:
            smallest = left
        if right < new_size and heap[right, 0] < heap[smallest, 0]:
            smallest = right
        if smallest == pos:
            break
        for k in range(heap.shape[1]):
            tmp = heap[pos, k]
            heap[pos, k] = heap[smallest, k]
            heap[smallest, k] = tmp
        pos = smallest
    return top, new_size


# ---------------------------------------------------------------------------
# Numba-compiled supercover line
# ---------------------------------------------------------------------------

@njit(cache=True)
def _supercover_line_nb(r0, c0, r1, c1, buf):
    """Fill *buf* with (row, col) pairs; return the count of pairs.

    *buf* must be pre-allocated with shape (max_cells, 2).
    max_cells >= 2*(|dr|+|dc|) + 1 is sufficient.
    """
    dr = abs(r1 - r0)
    dc = abs(c1 - c0)
    r = r0
    c = c0
    sr = 1 if r0 < r1 else -1
    sc = 1 if c0 < c1 else -1
    err = dr - dc
    n = 0
    while True:
        buf[n, 0] = r
        buf[n, 1] = c
        n += 1
        if r == r1 and c == c1:
            break
        e2 = 2 * err
        step_r = e2 > -dc
        step_c = e2 < dr
        if step_r and step_c:
            buf[n, 0] = r + sr
            buf[n, 1] = c
            n += 1
            buf[n, 0] = r
            buf[n, 1] = c + sc
            n += 1
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
    return n


# ---------------------------------------------------------------------------
# Numba-compiled line cost (barrier check + cost in one pass)
# ---------------------------------------------------------------------------

@njit(cache=True)
def _line_cost_or_inf_nb(r0, c0, r1, c1, cost_data, rows, cols, cy, cx, buf):
    """Combined barrier check + line cost.  Returns inf on barrier."""
    n = _supercover_line_nb(r0, c0, r1, c1, buf)
    if n == 0:
        return 0.0
    total = 0.0
    for i in range(n):
        r = buf[i, 0]
        c = buf[i, 1]
        if r < 0 or r >= rows or c < 0 or c >= cols:
            return np.inf
        val = cost_data[r, c]
        if not np.isfinite(val) or val < 0.0:
            return np.inf
        total += val
    avg = total / n
    dr = (r1 - r0) * cy
    dc = (c1 - c0) * cx
    dist = math.sqrt(dr * dr + dc * dc)
    return avg * dist


# ---------------------------------------------------------------------------
# Numba-compiled cost-aware straightening inner loop
# ---------------------------------------------------------------------------

@njit(cache=True)
def _straighten_core_nb(path_arr, cost_data, rows, cols, cy, cx,
                        cum_cost, max_skip, cost_tolerance):
    """Return an int32 array of selected path indices.

    Parameters
    ----------
    path_arr : ndarray, shape (n, 2), int32
        Path as row/col pairs.
    cost_data : ndarray, float64
        Cost raster.
    cum_cost : ndarray, float64, shape (n,)
        Cumulative cost along the original path.
    """
    n = path_arr.shape[0]
    # Worst case: every point is selected
    result = np.empty(n, dtype=np.int32)
    result[0] = 0
    result_len = 1
    i = 0

    # Allocate supercover line buffer once
    max_buf = 2 * (rows + cols) + 1
    buf = np.empty((max_buf, 2), dtype=np.int64)

    while i < n - 1:
        j_limit = min(n - 1, i + max_skip)
        best_j = i + 1
        for j in range(j_limit, i + 1, -1):
            shortcut_cost = _line_cost_or_inf_nb(
                path_arr[i, 0], path_arr[i, 1],
                path_arr[j, 0], path_arr[j, 1],
                cost_data, rows, cols, cy, cx, buf
            )
            if np.isinf(shortcut_cost):
                continue
            orig_segment_cost = cum_cost[j] - cum_cost[i]
            if orig_segment_cost <= 0.0:
                best_j = j
                break
            if shortcut_cost <= orig_segment_cost * cost_tolerance:
                best_j = j
                break
        result[result_len] = best_j
        result_len += 1
        i = best_j

    return result[:result_len]


# ---------------------------------------------------------------------------
# Numba-compiled standard Dijkstra core
# ---------------------------------------------------------------------------

@njit(cache=True)
def _dijkstra_core_standard(
    cost_data, rows, cols,
    sr, sc, er, ec,
    step_dists, h_map,
    dist_weight, straight_weight, scale,
    dirs_arr,
):
    """Standard 8-direction A*/Dijkstra — returns (best, parent_dir).

    Uses an array-based min-heap instead of Python heapq.
    """
    best = np.full((rows, cols), np.inf, dtype=np.float32)
    best[sr, sc] = 0.0
    parent_dir = np.full((rows, cols), -1, dtype=np.int8)

    # Pre-allocate heap: worst case ~rows*cols entries, but typically
    # far fewer.  We start with a generous estimate.
    max_heap = rows * cols * 2
    # Columns: f_score(0), counter(1), g_score(2), row(3), col(4)
    heap = np.empty((max_heap, 5), dtype=np.float64)
    heap_size = 0

    counter = 0.0
    elem = np.empty(5, dtype=np.float64)
    elem[0] = h_map[sr, sc]
    elem[1] = counter
    elem[2] = 0.0
    elem[3] = float(sr)
    elem[4] = float(sc)
    heap_size = _heap_push(heap, heap_size, elem)

    while heap_size > 0:
        top, heap_size = _heap_pop(heap, heap_size)
        g = top[2]
        r = int(top[3])
        c = int(top[4])
        if g > best[r, c]:
            continue
        if r == er and c == ec:
            break

        d_in = int(parent_dir[r, c])

        for d in range(8):
            dr = dirs_arr[d, 0]
            dc = dirs_arr[d, 1]
            nr = r + dr
            nc = c + dc
            if nr < 0 or nr >= rows or nc < 0 or nc >= cols:
                continue
            cell_val = cost_data[nr, nc]
            if not np.isfinite(cell_val) or cell_val < 0.0:
                continue

            sd = step_dists[d]
            new_cost = g + cell_val * sd + dist_weight * sd

            if d_in >= 0 and d != d_in:
                str_dr = dirs_arr[d_in, 0]
                str_dc = dirs_arr[d_in, 1]
                str_r = r + str_dr
                str_c = c + str_dc
                if 0 <= str_r < rows and 0 <= str_c < cols:
                    str_val = cost_data[str_r, str_c]
                    if np.isfinite(str_val) and str_val >= 0.0:
                        diff = cell_val - str_val
                        if diff < 0.0:
                            diff = -diff
                        similarity = 1.0 - diff / scale
                        if similarity < 0.0:
                            similarity = 0.0
                        new_cost += similarity * straight_weight * sd

            if new_cost < best[nr, nc]:
                best[nr, nc] = np.float32(new_cost)
                parent_dir[nr, nc] = np.int8(d)
                counter += 1.0
                g_stored = float(best[nr, nc])
                elem[0] = g_stored + h_map[nr, nc]
                elem[1] = counter
                elem[2] = g_stored
                elem[3] = float(nr)
                elem[4] = float(nc)
                # Grow heap if needed
                if heap_size >= heap.shape[0]:
                    new_heap = np.empty((heap.shape[0] * 2, 5), dtype=np.float64)
                    new_heap[:heap.shape[0], :] = heap
                    heap = new_heap
                heap_size = _heap_push(heap, heap_size, elem)

    return best, parent_dir


# ---------------------------------------------------------------------------
# Numba-compiled direction-aware Dijkstra core (curvature support)
# ---------------------------------------------------------------------------

@njit(cache=True)
def _dijkstra_core_directed(
    cost_data, rows, cols,
    sr, sc, er, ec,
    step_dists, h_map,
    curv_div, max_turning_angle, dist_weight, straight_weight, scale,
    dirs_arr, turn_lut,
):
    """Direction-aware 8-direction A*/Dijkstra — returns (best3d, parent_d3d, found, end_d).

    State space: (row, col, direction), with direction in [0..8] (8 = no-dir).
    """
    n_states = 9  # 8 directions + 1 for "no direction"
    best = np.full((rows, cols, n_states), np.inf, dtype=np.float32)
    best[sr, sc, 8] = 0.0
    parent_d = np.full((rows, cols, n_states), -1, dtype=np.int8)

    # Columns: f_score(0), counter(1), g_score(2), row(3), col(4), dir_in(5)
    max_heap = rows * cols * 2
    heap = np.empty((max_heap, 6), dtype=np.float64)
    heap_size = 0

    counter = 0.0
    elem = np.empty(6, dtype=np.float64)
    elem[0] = float(h_map[sr, sc])
    elem[1] = counter
    elem[2] = 0.0
    elem[3] = float(sr)
    elem[4] = float(sc)
    elem[5] = -1.0  # NO_DIR
    heap_size = _heap_push(heap, heap_size, elem)

    found = False
    end_d = -1

    while heap_size > 0:
        top, heap_size = _heap_pop(heap, heap_size)
        g = top[2]
        r = int(top[3])
        c = int(top[4])
        d_in = int(top[5])
        d_idx = d_in if d_in >= 0 else 8

        if g > best[r, c, d_idx]:
            continue
        if r == er and c == ec:
            found = True
            end_d = d_in
            break

        for d_out in range(8):
            dr = dirs_arr[d_out, 0]
            dc = dirs_arr[d_out, 1]
            nr = r + dr
            nc = c + dc
            if nr < 0 or nr >= rows or nc < 0 or nc >= cols:
                continue
            cell_val = cost_data[nr, nc]
            if not np.isfinite(cell_val) or cell_val < 0.0:
                continue

            sd = step_dists[d_out]
            base = cell_val * sd

            curv_penalty = 0.0
            if d_in >= 0:
                angle = turn_lut[d_in, d_out]
                if angle > max_turning_angle:
                    continue
                curv_penalty = curv_div * angle * sd

            straightness_penalty = 0.0
            if d_in >= 0 and d_out != d_in:
                str_dr = dirs_arr[d_in, 0]
                str_dc = dirs_arr[d_in, 1]
                str_r = r + str_dr
                str_c = c + str_dc
                if 0 <= str_r < rows and 0 <= str_c < cols:
                    str_val = cost_data[str_r, str_c]
                    if np.isfinite(str_val) and str_val >= 0.0:
                        diff = cell_val - str_val
                        if diff < 0.0:
                            diff = -diff
                        similarity = 1.0 - diff / scale
                        if similarity < 0.0:
                            similarity = 0.0
                        straightness_penalty = similarity * straight_weight * sd

            new_cost = g + base + curv_penalty + straightness_penalty + dist_weight * sd
            nd_idx = d_out
            if new_cost < best[nr, nc, nd_idx]:
                best[nr, nc, nd_idx] = np.float32(new_cost)
                parent_d[nr, nc, nd_idx] = np.int8(d_in)
                counter += 1.0
                g_stored = float(best[nr, nc, nd_idx])
                elem[0] = g_stored + float(h_map[nr, nc])
                elem[1] = counter
                elem[2] = g_stored
                elem[3] = float(nr)
                elem[4] = float(nc)
                elem[5] = float(d_out)
                if heap_size >= heap.shape[0]:
                    new_heap = np.empty((heap.shape[0] * 2, 6), dtype=np.float64)
                    new_heap[:heap.shape[0], :] = heap
                    heap = new_heap
                heap_size = _heap_push(heap, heap_size, elem)

    return best, parent_d, found, end_d


# ---------------------------------------------------------------------------
# Path reconstruction (plain Python — mirrors pure_python version)
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

    straightened = _cost_aware_straighten_path_numba(
        path, cost_raster, cell_size, straighten_factor, cost_tolerance
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

    straightened = _cost_aware_straighten_path_numba(
        path, cost_raster, cell_size, straighten_factor, cost_tolerance
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
# Numba-accelerated cost-aware straightening (wrapper)
# ---------------------------------------------------------------------------

def _cost_aware_straighten_path_numba(
    path: List[Tuple[int, int]],
    cost_raster: np.ndarray,
    cell_size: Tuple[float, float],
    straighten_factor: float = 0.3,
    cost_tolerance: float = 1.05,
) -> List[Tuple[float, float]]:
    """Straighten path with cost awareness using Numba-compiled inner loop."""
    if len(path) <= 2:
        return [(float(r), float(c)) for r, c in path]

    if straighten_factor <= 0.0:
        return [(float(r), float(c)) for r, c in path]

    rows, cols = cost_raster.shape
    if cost_raster.dtype == np.float64 and cost_raster.flags["C_CONTIGUOUS"]:
        cost_data = cost_raster
    else:
        cost_data = np.ascontiguousarray(cost_raster, dtype=np.float64)

    n = len(path)
    max_skip = max(2, int(straighten_factor * (n - 1)))

    # Convert path to int32 array for Numba
    path_arr = np.array(path, dtype=np.int32)

    # Pre-compute cumulative cost
    cy, cx = cell_size
    cum_cost = np.zeros(n, dtype=np.float64)
    for k in range(1, n):
        r1, c1 = path[k - 1]
        r2, c2 = path[k]
        dr, dc = r2 - r1, c2 - c1
        sd = math.sqrt((dr * cy) ** 2 + (dc * cx) ** 2)
        cell_val = float(cost_data[r2, c2])
        if not math.isfinite(cell_val) or cell_val < 0:
            cell_val = 0.0
        cum_cost[k] = cum_cost[k - 1] + cell_val * sd

    indices = _straighten_core_nb(
        path_arr, cost_data, rows, cols, cy, cx,
        cum_cost, max_skip, cost_tolerance
    )

    return [(float(path[k][0]), float(path[k][1])) for k in indices]


# ---------------------------------------------------------------------------
# Standard Dijkstra wrapper
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

    step_dists_list = _step_distances(cell_size)
    step_dists = np.array(step_dists_list, dtype=np.float64)
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

    best, parent_dir = _dijkstra_core_standard(
        cost_data, rows, cols, sr, sc, er, ec,
        step_dists, h_map, dist_weight, straight_weight, scale,
        _DIRS_ARRAY,
    )

    if not np.isfinite(best[er, ec]):
        raise RuntimeError("No path found between start and end")

    return _build_result(parent_dir, start, end, best, float(best[er, ec]),
                         cell_size, cost_data, straighten_factor,
                         cost_tolerance)


# ---------------------------------------------------------------------------
# Direction-aware Dijkstra wrapper
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

    step_dists_list = _step_distances(cell_size)
    step_dists = np.array(step_dists_list, dtype=np.float64)
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

    curv_div = curv_weight / 180.0

    best, parent_d, found, end_d = _dijkstra_core_directed(
        cost_data, rows, cols, sr, sc, er, ec,
        step_dists, h_map, curv_div, max_turning_angle,
        dist_weight, straight_weight, scale,
        _DIRS_ARRAY, _TURN_LUT_ARRAY,
    )

    if not found:
        raise RuntimeError("No path found between start and end")

    end_state = (er, ec, end_d)
    return _build_result_directed(parent_d, end_state, best, cell_size,
                                   cost_data, straighten_factor,
                                   cost_tolerance)


# ---------------------------------------------------------------------------
# Public API (identical signature to pure_python version)
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

    This is the **Numba-accelerated** version.  The API is identical to
    ``pure_python.cost_aware_straighten_lcp.cost_aware_least_cost_path()``.

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
    cell_size : tuple[float, float], optional
        ``(y_size, x_size)`` of each raster cell.

    Returns
    -------
    dict
        Same keys as ``pure_python.cost_aware_straighten_lcp.cost_aware_least_cost_path()``.
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
