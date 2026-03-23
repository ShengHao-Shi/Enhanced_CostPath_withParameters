"""
Theta* Least Cost Path Algorithm (方案B)
==========================================

Implements a **Theta*** variant of the least cost path algorithm.  Unlike
the standard 8-direction grid search in ``enhanced_lcp.py``, this module
integrates line-of-sight checks directly into the Dijkstra / A* search.
When a cell *s* is expanded and its parent *p = parent(s)* can see a
neighbour *s'* along a clear line, the algorithm connects *s'* directly
to *p* instead of *s*.  This produces **any-angle** paths during the
search itself — no post-processing straightening is needed.

The cost along a straight-line shortcut is computed by integrating the
cost raster values along the line (using the supercover algorithm),
weighted by Euclidean distance.

Key differences from ``enhanced_lcp.py``:

* Parent pointers store arbitrary ``(row, col)`` — not limited to
  8-connected neighbours.
* Path output is naturally any-angle; ``straighten_factor`` is removed.
* Chaikin smoothing is still applied for rounded corners.
* Curvature / distance parameters are supported with the same semantics.

Dependencies: numpy (required).
"""

import heapq
import math
from typing import Dict, List, Optional, Tuple

import numpy as np

# ---------------------------------------------------------------------------
# Direction definitions for 8-connectivity (same as enhanced_lcp)
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


# ---------------------------------------------------------------------------
# Supercover line (shared with enhanced_lcp)
# ---------------------------------------------------------------------------

def _supercover_line(
    r0: int, c0: int, r1: int, c1: int,
) -> List[Tuple[int, int]]:
    """Return all grid cells the line segment (r0,c0)->(r1,c1) passes through."""
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


def _line_cost(
    p0: Tuple[int, int],
    p1: Tuple[int, int],
    cost_data: np.ndarray,
    rows: int,
    cols: int,
    cell_size: Tuple[float, float],
) -> float:
    """Compute cost of travelling along the straight line from *p0* to *p1*.

    The cost is the sum of ``cost_raster[cell] × fraction_of_line_in_cell``
    for every cell the supercover line passes through, multiplied by the
    Euclidean length of the line.  This approximates the integral of the
    cost surface along the line segment.

    Returns ``inf`` if the line crosses a barrier (NODATA / negative / OOB).
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

    # Average cost along line × Euclidean distance
    avg_cost = total_cost_sum / n
    cy, cx = cell_size
    dr = (p1[0] - p0[0]) * cy
    dc = (p1[1] - p0[1]) * cx
    dist = math.sqrt(dr * dr + dc * dc)

    return avg_cost * dist


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
# Cost-scale helpers
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
# Theta* public API
# ---------------------------------------------------------------------------

def theta_star_least_cost_path(
    cost_raster: np.ndarray,
    start: Tuple[int, int],
    end: Tuple[int, int],
    curvature_factor: float = 0.0,
    max_turning_angle: float = 180.0,
    distance_factor: float = 0.0,
    cell_size: Tuple[float, float] = (1.0, 1.0),
) -> Dict:
    """Compute an any-angle least cost path using the Theta* algorithm.

    Unlike the standard 8-direction grid search, Theta* integrates
    line-of-sight checks into the A* search.  When expanding a cell *s*,
    if the parent of *s* can see a neighbour *s'* along a clear line,
    *s'* is connected directly to *parent(s)* at any angle.

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
        Weight for raw path length in the cost function (0.0 – 1.0,
        default 0.0).
    cell_size : tuple[float, float], optional
        ``(y_size, x_size)`` of each raster cell in map units.

    Returns
    -------
    dict
        ``path``          – any-angle waypoints ``(row, col)`` int coords.
        ``smoothed_path`` – Chaikin-smoothed version with rounded turns.
        ``total_cost``    – accumulated cost.
        ``path_length``   – physical length in map units.
    """
    _validate_params(cost_raster, start, end, curvature_factor,
                     max_turning_angle, distance_factor)

    return _theta_star_search(
        cost_raster, start, end,
        curvature_factor, max_turning_angle, distance_factor, cell_size,
    )


# ---------------------------------------------------------------------------
# Theta* search implementation
# ---------------------------------------------------------------------------

def _theta_star_search(
    cost_raster: np.ndarray,
    start: Tuple[int, int],
    end: Tuple[int, int],
    curvature_factor: float,
    max_turning_angle: float,
    distance_factor: float,
    cell_size: Tuple[float, float],
) -> Dict:
    rows, cols = cost_raster.shape
    sr, sc = start
    er, ec = end

    step_dists = _step_distances(cell_size)
    scale = _cost_scale(cost_raster)
    curv_weight = curvature_factor * _CURVATURE_AMPLIFIER * scale
    dist_weight = distance_factor * scale

    cy, cx = cell_size
    h_weight = _min_cost(cost_raster) + dist_weight

    # Pre-compute A* heuristic map.
    r_idx = np.arange(rows, dtype=np.float64)
    c_idx = np.arange(cols, dtype=np.float64)
    h_map = (np.sqrt(((r_idx[:, None] - er) * cy) ** 2
                     + ((c_idx[None, :] - ec) * cx) ** 2)
             * h_weight).astype(np.float32)

    cost_data = np.ascontiguousarray(cost_raster, dtype=np.float64)

    best = np.full((rows, cols), np.inf, dtype=np.float32)
    best[sr, sc] = 0.0

    # Parent pointers: store (row, col) of parent — can be any cell.
    parent_r = np.full((rows, cols), -1, dtype=np.int32)
    parent_c = np.full((rows, cols), -1, dtype=np.int32)

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

        # Get parent of current cell (for Theta* line-of-sight).
        pr, pc = int(parent_r[r, c]), int(parent_c[r, c])
        has_parent = pr >= 0

        for d in range(NUM_DIRS):
            dr, dc = _dirs[d]
            nr, nc = r + dr, c + dc
            if not (0 <= nr < rows and 0 <= nc < cols):
                continue
            cell_val = float(cost_data[nr, nc])
            if not _isfinite(cell_val) or cell_val < 0:
                continue

            # --- Theta* core: try to connect via parent(s) -----------------
            # Path 1: parent(s) -> s' via straight line (any-angle).
            # Path 2: s -> s' via grid step (standard 8-direction).
            # Pick whichever is cheaper, provided the line is clear.

            used_parent = False
            if has_parent:
                # Check line-of-sight from parent to neighbour
                if _is_line_clear((pr, pc), (nr, nc), cost_data, rows, cols):
                    # Cost via parent: g(parent) + line_cost(parent -> s')
                    g_parent = float(best[pr, pc])
                    lc = _line_cost(
                        (pr, pc), (nr, nc), cost_data, rows, cols, cell_size
                    )
                    new_cost_via_parent = g_parent + lc + dist_weight * math.sqrt(
                        ((nr - pr) * cy) ** 2 + ((nc - pc) * cx) ** 2
                    )

                    # Curvature penalty for parent path
                    if curvature_factor > 0.0 or max_turning_angle < 180.0:
                        # Compute angle at parent
                        gpr, gpc = int(parent_r[pr, pc]), int(parent_c[pr, pc])
                        if gpr >= 0:
                            # Direction from grandparent to parent
                            d_in_r = pr - gpr
                            d_in_c = pc - gpc
                            # Direction from parent to s'
                            d_out_r = nr - pr
                            d_out_c = nc - pc
                            angle = _vector_angle(d_in_r, d_in_c, d_out_r, d_out_c)
                            if angle > max_turning_angle:
                                new_cost_via_parent = float("inf")
                            else:
                                sd_parent = math.sqrt(
                                    ((nr - pr) * cy) ** 2 + ((nc - pc) * cx) ** 2
                                )
                                new_cost_via_parent += (curv_weight / 180.0) * angle * sd_parent

                    if new_cost_via_parent < best[nr, nc]:
                        best[nr, nc] = new_cost_via_parent
                        parent_r[nr, nc] = pr
                        parent_c[nr, nc] = pc
                        counter += 1
                        g_stored = float(best[nr, nc])
                        _heappush(pq, (g_stored + float(h_map[nr, nc]),
                                       counter, g_stored, nr, nc))
                        used_parent = True

            # Path 2: standard grid step s -> s'
            sd = step_dists[d]
            new_cost = g + cell_val * sd + dist_weight * sd

            # Curvature penalty for grid path
            if curvature_factor > 0.0 or max_turning_angle < 180.0:
                if has_parent:
                    d_in_r = r - pr
                    d_in_c = c - pc
                    d_out_r = nr - r
                    d_out_c = nc - c
                    angle = _vector_angle(d_in_r, d_in_c, d_out_r, d_out_c)
                    if angle > max_turning_angle:
                        if not used_parent:
                            continue
                        else:
                            # Already handled via parent path
                            continue
                    new_cost += (curv_weight / 180.0) * angle * sd

            if new_cost < best[nr, nc]:
                best[nr, nc] = new_cost
                parent_r[nr, nc] = r
                parent_c[nr, nc] = c
                counter += 1
                g_stored = float(best[nr, nc])
                _heappush(pq, (g_stored + float(h_map[nr, nc]),
                               counter, g_stored, nr, nc))

    if not np.isfinite(best[er, ec]):
        raise RuntimeError("No path found between start and end")

    return _build_result(parent_r, parent_c, start, end,
                         float(best[er, ec]), cell_size, cost_raster)


def _vector_angle(dr1: float, dc1: float, dr2: float, dc2: float) -> float:
    """Return the angle (degrees) between two direction vectors.

    Returns 0 for same direction, 180 for opposite.
    """
    len1 = math.sqrt(dr1 * dr1 + dc1 * dc1)
    len2 = math.sqrt(dr2 * dr2 + dc2 * dc2)
    if len1 == 0 or len2 == 0:
        return 0.0
    cos_angle = (dr1 * dr2 + dc1 * dc2) / (len1 * len2)
    # Clamp for floating-point safety
    cos_angle = max(-1.0, min(1.0, cos_angle))
    return math.degrees(math.acos(cos_angle))


# ---------------------------------------------------------------------------
# Path reconstruction
# ---------------------------------------------------------------------------

def _build_result(
    parent_r: np.ndarray,
    parent_c: np.ndarray,
    start: Tuple[int, int],
    end: Tuple[int, int],
    total_cost: float,
    cell_size: Tuple[float, float],
    cost_raster: np.ndarray,
) -> Dict:
    """Reconstruct the any-angle path from parent pointers."""
    path: List[Tuple[int, int]] = []
    r, c = end
    sr, sc = start
    visited = set()
    while True:
        path.append((int(r), int(c)))
        if r == sr and c == sc:
            break
        key = (int(r), int(c))
        if key in visited:
            raise RuntimeError("Cycle detected in parent pointers")
        visited.add(key)
        pr, pc = int(parent_r[r, c]), int(parent_c[r, c])
        if pr < 0:
            break
        r, c = pr, pc
    path.reverse()

    cy, cx = cell_size
    path_length = 0.0
    for i in range(1, len(path)):
        r1, c1 = path[i - 1]
        r2, c2 = path[i]
        dr, dc = (r2 - r1) * cy, (c2 - c1) * cx
        path_length += math.sqrt(dr * dr + dc * dc)

    # Convert to float for smoothing
    float_path = [(float(r), float(c)) for r, c in path]
    smoothed = _smooth_path_nodata_safe(float_path, cost_raster)

    return {
        "path": path,
        "smoothed_path": smoothed,
        "total_cost": total_cost,
        "path_length": path_length,
    }


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
    path: List[Tuple[float, float]],
    cost_raster: np.ndarray,
    iterations: int = 3,
) -> List[Tuple[float, float]]:
    """Smooth via Chaikin, falling back when NODATA is crossed."""
    smoothed = smooth_path(path, iterations)

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
                return list(path)
            val = float(cost_data[r, c])
            if not _isfinite(val) or val < 0:
                return list(path)

    return smoothed
