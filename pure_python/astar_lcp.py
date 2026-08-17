"""
Cost-Aware A* LCP Algorithm
============================

This module provides an A* (A-Star) implementation of the least cost path
algorithm as a drop-in alternative to the Dijkstra-based
``cost_aware_straighten_lcp`` module.  All parameters, return values, and
post-processing steps (cost-aware straightening + Chaikin smoothing) are
identical; only the graph search is different.

A* vs Dijkstra
--------------
Dijkstra expands nodes in order of their accumulated cost from the start
(``g``).  A* adds an admissible heuristic ``h`` — a lower-bound estimate of
the remaining cost to the goal — so that nodes closer to the goal are
prioritised::

    f(n) = g(n) + h(n)

This typically results in far fewer node expansions when the heuristic is
informative, especially on large open rasters.

Heuristic
---------
The heuristic used here is::

    h(r, c) = euclidean_distance((r, c), goal) * (min_cost + dist_weight)

where ``min_cost`` is the minimum finite cell cost in the raster and
``dist_weight = distance_factor * cost_scale``.  This heuristic is
*consistent* (and therefore admissible): the true travel cost from any cell to
the goal cannot be less than the minimum per-unit-distance cost times the
straight-line distance.

A* guarantees an optimal path when the heuristic is consistent.  Note that
the optional straightness penalty (``_STRAIGHTNESS_PENALTY``) can make the
effective cost function slightly non-monotone; it is kept here to preserve the
same path character as the Dijkstra version, but for a strictly optimal search
set ``distance_factor=0`` and ``curvature_factor=0``.

Closed set
----------
Unlike the Dijkstra variant in ``cost_aware_straighten_lcp``, this
implementation maintains an explicit *closed set*.  Once a node (or state)
is popped from the priority queue it is immediately added to the closed set
and never re-expanded.  With a consistent heuristic the first time a node
is popped it already holds its optimal g-value, so this is both correct and
efficient.

Public API
----------
``cost_aware_astar_least_cost_path(cost_raster, start, end, ...)``
    Main entry point — identical signature to
    ``cost_aware_straighten_lcp.cost_aware_least_cost_path``.

Dependencies: numpy (required).
"""

import heapq
import math
from typing import Dict, List, Optional, Tuple

import numpy as np

# ---------------------------------------------------------------------------
# Re-use shared constants and utilities from the Dijkstra module
# ---------------------------------------------------------------------------
from pure_python.cost_aware_straighten_lcp import (
    DIRECTIONS,
    NO_DIR,
    NUM_DIRS,
    _CURVATURE_AMPLIFIER,
    _STRAIGHTNESS_PENALTY,
    _TURN_ANGLE_LUT,
    _build_result,
    _build_result_directed,
    _cost_scale,
    _min_cost,
    _smooth_path_nodata_safe,
    _step_distances,
    _validate_params,
    cost_aware_straighten_path,
)


# ---------------------------------------------------------------------------
# Standard A* (no direction tracking)
# ---------------------------------------------------------------------------

def _astar_standard(
    cost_raster: np.ndarray,
    start: Tuple[int, int],
    end: Tuple[int, int],
    distance_factor: float,
    cell_size: Tuple[float, float],
    straighten_factor: float,
    cost_tolerance: float,
    progress_callback=None,
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

    # Pre-compute the heuristic map: h(r, c) = euclidean_distance * h_weight.
    r_idx = np.arange(rows, dtype=np.float64)
    c_idx = np.arange(cols, dtype=np.float64)
    h_map = (
        np.sqrt(
            ((r_idx[:, None] - er) * cy) ** 2
            + ((c_idx[None, :] - ec) * cx) ** 2
        )
        * h_weight
    ).astype(np.float32)

    cost_data = np.ascontiguousarray(cost_raster, dtype=np.float64)

    if progress_callback:
        progress_callback(
            "A* search: initialization complete, starting search..."
        )

    # g-values (cost from start); infinity = not yet reached.
    best = np.full((rows, cols), np.inf, dtype=np.float64)
    best[sr, sc] = 0.0
    parent_dir = np.full((rows, cols), -1, dtype=np.int8)

    # Closed set: numpy bool array is significantly faster than a Python set of
    # (row, col) tuples because it avoids tuple allocation and dict hashing on
    # every membership test and insertion.
    closed = np.zeros((rows, cols), dtype=bool)

    _heappop = heapq.heappop
    _heappush = heapq.heappush
    _isfinite = math.isfinite
    _dirs = DIRECTIONS

    counter = 0
    cells_expanded = 0
    total_cells = rows * cols
    next_pct = 10
    # Priority queue entries: (f, tie-breaker, g, row, col)
    pq: List = [(float(h_map[sr, sc]), counter, 0.0, sr, sc)]

    while pq:
        _, _, g, r, c = _heappop(pq)

        # Skip stale entries or already-settled nodes.
        if closed[r, c]:
            continue
        if g > best[r, c]:
            continue

        # Settle this node.
        closed[r, c] = True

        if r == er and c == ec:
            break

        cells_expanded += 1
        if progress_callback:
            pct = min(int(cells_expanded * 100 / total_cells), 99)
            if pct >= next_pct:
                progress_callback(
                    f"A* search: expanded {cells_expanded} nodes (~{pct}%)"
                )
                next_pct = pct + 10

        d_in = int(parent_dir[r, c])

        for d in range(NUM_DIRS):
            dr, dc = _dirs[d]
            nr, nc = r + dr, c + dc
            if not (0 <= nr < rows and 0 <= nc < cols):
                continue
            if closed[nr, nc]:
                continue
            cell_val = float(cost_data[nr, nc])
            if not _isfinite(cell_val) or cell_val < 0:
                continue

            sd = step_dists[d]
            new_cost = g + cell_val * sd + dist_weight * sd

            # Optional straightness penalty (same as Dijkstra variant).
            if d_in >= 0 and d != d_in:
                str_dr, str_dc = _dirs[d_in]
                str_r, str_c = r + str_dr, c + str_dc
                if 0 <= str_r < rows and 0 <= str_c < cols:
                    str_val = float(cost_data[str_r, str_c])
                    if _isfinite(str_val) and str_val >= 0:
                        similarity = max(
                            0.0, 1.0 - abs(cell_val - str_val) / scale
                        )
                        new_cost += similarity * straight_weight * sd

            if new_cost < best[nr, nc]:
                best[nr, nc] = new_cost
                parent_dir[nr, nc] = d
                counter += 1
                g_stored = float(best[nr, nc])
                _heappush(
                    pq,
                    (
                        g_stored + float(h_map[nr, nc]),
                        counter,
                        g_stored,
                        nr,
                        nc,
                    ),
                )

    if progress_callback:
        progress_callback(
            f"A* search: complete ({cells_expanded} nodes expanded)"
        )

    if not np.isfinite(best[er, ec]):
        raise RuntimeError("No path found between start and end")

    return _build_result(
        parent_dir, start, end, best, float(best[er, ec]),
        cell_size, cost_data, straighten_factor, cost_tolerance,
        progress_callback,
    )


# ---------------------------------------------------------------------------
# Direction-aware A* (curvature / turning-angle support)
# ---------------------------------------------------------------------------

def _astar_with_direction(
    cost_raster: np.ndarray,
    start: Tuple[int, int],
    end: Tuple[int, int],
    curvature_factor: float,
    max_turning_angle: float,
    distance_factor: float,
    cell_size: Tuple[float, float],
    straighten_factor: float,
    cost_tolerance: float,
    progress_callback=None,
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
    h_map = (
        np.sqrt(
            ((r_idx[:, None] - er) * cy) ** 2
            + ((c_idx[None, :] - ec) * cx) ** 2
        )
        * h_weight
    ).astype(np.float32)

    cost_data = np.ascontiguousarray(cost_raster, dtype=np.float64)

    if progress_callback:
        progress_callback(
            "A* search (direction-aware): initialization complete, starting search..."
        )

    # State space: (row, col, incoming_direction_index)
    # NUM_DIRS is the "no prior direction" sentinel index.
    n_states = NUM_DIRS + 1
    best = np.full((rows, cols, n_states), np.inf, dtype=np.float64)
    best[sr, sc, NUM_DIRS] = 0.0

    parent_d = np.full((rows, cols, n_states), -1, dtype=np.int8)

    # Closed set: numpy bool array is significantly faster than a Python set of
    # (row, col, direction) triples because it avoids tuple allocation and dict
    # hashing on every membership test and insertion.
    closed = np.zeros((rows, cols, n_states), dtype=bool)

    _heappop = heapq.heappop
    _heappush = heapq.heappush
    _isfinite = math.isfinite
    _dirs = DIRECTIONS
    _turn_lut = _TURN_ANGLE_LUT
    _no_dir = NO_DIR
    _curv_div = curv_weight / 180.0

    counter = 0
    cells_expanded = 0
    total_cells = rows * cols * n_states
    next_pct = 10
    pq: List = [(float(h_map[sr, sc]), counter, 0.0, sr, sc, _no_dir)]

    found = False
    end_state: Optional[Tuple[int, int, int]] = None

    while pq:
        _, _, g, r, c, d_in = _heappop(pq)
        d_idx = d_in if d_in >= 0 else NUM_DIRS

        if closed[r, c, d_idx]:
            continue
        if g > best[r, c, d_idx]:
            continue

        closed[r, c, d_idx] = True

        if r == er and c == ec:
            found = True
            end_state = (r, c, d_in)
            break

        cells_expanded += 1
        if progress_callback:
            pct = min(int(cells_expanded * 100 / total_cells), 99)
            if pct >= next_pct:
                progress_callback(
                    f"A* search (direction-aware): expanded {cells_expanded} nodes (~{pct}%)"
                )
                next_pct = pct + 10

        for d_out in range(NUM_DIRS):
            dr, dc = _dirs[d_out]
            nr, nc = r + dr, c + dc
            if not (0 <= nr < rows and 0 <= nc < cols):
                continue
            nd_idx = d_out
            if closed[nr, nc, nd_idx]:
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
                if 0 <= str_r < rows and 0 <= str_c < cols:
                    str_val = float(cost_data[str_r, str_c])
                    if _isfinite(str_val) and str_val >= 0:
                        similarity = max(
                            0.0, 1.0 - abs(cell_val - str_val) / scale
                        )
                        straightness_penalty = similarity * straight_weight * sd

            new_cost = (
                g + base + curv_penalty + straightness_penalty + dist_weight * sd
            )

            if new_cost < best[nr, nc, nd_idx]:
                best[nr, nc, nd_idx] = new_cost
                parent_d[nr, nc, nd_idx] = d_in
                counter += 1
                g_stored = float(best[nr, nc, nd_idx])
                _heappush(
                    pq,
                    (
                        g_stored + float(h_map[nr, nc]),
                        counter,
                        g_stored,
                        nr,
                        nc,
                        d_out,
                    ),
                )

    if progress_callback:
        progress_callback(
            f"A* search (direction-aware): complete ({cells_expanded} nodes expanded)"
        )

    if not found:
        raise RuntimeError("No path found between start and end")

    return _build_result_directed(
        parent_d, end_state, best, cell_size, cost_data,
        straighten_factor, cost_tolerance, progress_callback,
    )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def cost_aware_astar_least_cost_path(
    cost_raster: np.ndarray,
    start: Tuple[int, int],
    end: Tuple[int, int],
    curvature_factor: float = 0.0,
    max_turning_angle: float = 180.0,
    distance_factor: float = 0.0,
    straighten_factor: float = 0.3,
    cost_tolerance: float = 1.05,
    cell_size: Tuple[float, float] = (1.0, 1.0),
    progress_callback=None,
) -> Dict:
    """Compute an LCP using A* search with cost-aware post-processing.

    This function is a drop-in replacement for
    ``cost_aware_straighten_lcp.cost_aware_least_cost_path``.  The graph
    search is A* with an admissible, consistent heuristic instead of
    Dijkstra, which typically expands far fewer nodes on open rasters.
    All post-processing (cost-aware straightening + Chaikin smoothing) is
    identical.

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
        Hard upper limit on turning angle in degrees (0 – 180, default 180).
    distance_factor : float, optional
        Weight for path length (0.0 – 1.0, default 0.0).
    straighten_factor : float, optional
        Lookahead for cost-aware straightening (0.0 – 0.5, default 0.3).
    cost_tolerance : float, optional
        Maximum shortcut-cost / original-cost ratio (>= 1.0, default 1.05).
    cell_size : tuple[float, float], optional
        ``(y_size, x_size)`` of each raster cell.
    progress_callback : callable, optional
        A function that accepts a single string argument for progress
        messages.

    Returns
    -------
    dict
        Keys: ``path``, ``straightened_path``, ``smoothed_path``,
        ``total_cost``, ``path_length``, ``directions``, ``turning_angles``.
        Identical structure to ``cost_aware_least_cost_path``.
    """
    _validate_params(
        cost_raster, start, end, curvature_factor,
        max_turning_angle, distance_factor, straighten_factor, cost_tolerance,
    )

    if progress_callback:
        rows, cols = cost_raster.shape
        progress_callback(
            f"Parameters validated. Raster size: {rows}×{cols}, "
            f"start: {start}, end: {end}"
        )

    use_curvature = curvature_factor > 0.0 or max_turning_angle < 180.0

    if use_curvature:
        return _astar_with_direction(
            cost_raster, start, end,
            curvature_factor, max_turning_angle, distance_factor, cell_size,
            straighten_factor, cost_tolerance, progress_callback,
        )
    return _astar_standard(
        cost_raster, start, end, distance_factor, cell_size,
        straighten_factor, cost_tolerance, progress_callback,
    )
