"""
Enhanced Least Cost Path Algorithm
===================================

Extends the standard Least Cost Path (LCP) algorithm with two additional
controls beyond the basic cost raster, start point, and end point:

1. **Curvature Control**: Penalizes sharp turns to produce smoother paths.
   - ``curvature_factor`` (0.0-1.0): Soft penalty weight for turns.
   - ``min_turning_angle`` (0-180 degrees): Minimum interior angle at turns.
     Higher values enforce gentler turns.
   - Built-in anti-zigzag preference that favours straight-line continuation
     when the cost surface varies little.

2. **Distance Factor**: Weights path length as an additional cost component,
   encouraging shorter paths when set above zero.

3. **Path Smoothing**: After computing the grid-cell path, Chaikin's
   corner-cutting algorithm is applied to produce a smooth curve that
   replaces sharp corners with rounded arcs.

The algorithm uses a modified Dijkstra's search on a raster grid with
8-connectivity. When curvature control is active, the search state includes
the incoming travel direction so that turning angles can be computed.

Dependencies: numpy (required), rasterio (optional, for file I/O).
"""

import heapq
import math
from typing import Dict, List, Optional, Tuple

import numpy as np

# ---------------------------------------------------------------------------
# Direction definitions for 8-connectivity
# Index:  0=N,  1=NE,  2=E,  3=SE,  4=S,  5=SW,  6=W,  7=NW
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

# Sentinel value representing "no incoming direction" at the start point.
NO_DIR: int = -1

# Internal amplifier so that curvature_factor in [0, 1] has a visible effect.
_CURVATURE_AMPLIFIER: float = 5.0

# Straightness penalty multiplier applied when changing direction and
# the straight-ahead cell has similar cost.  Reduces zigzag artefacts.
_STRAIGHTNESS_PENALTY: float = 0.3


def turning_angle(dir_from: int, dir_to: int) -> float:
    """Return the turning angle in degrees between two direction indices.

    Parameters
    ----------
    dir_from : int
        Incoming direction index (0-7).
    dir_to : int
        Outgoing direction index (0-7).

    Returns
    -------
    float
        Turning angle in degrees (0, 45, 90, 135, or 180).
    """
    diff = abs(dir_from - dir_to)
    steps = min(diff, NUM_DIRS - diff)
    return steps * 45.0


# ---------------------------------------------------------------------------
# Core algorithm
# ---------------------------------------------------------------------------


def _step_distances(cell_size: Tuple[float, float]) -> List[float]:
    """Pre-compute Euclidean step distances for each of the 8 directions."""
    cy, cx = cell_size
    return [math.sqrt((dr * cy) ** 2 + (dc * cx) ** 2) for dr, dc in DIRECTIONS]


def enhanced_least_cost_path(
    cost_raster: np.ndarray,
    start: Tuple[int, int],
    end: Tuple[int, int],
    curvature_factor: float = 0.0,
    min_turning_angle: float = 0.0,
    distance_factor: float = 0.0,
    cell_size: Tuple[float, float] = (1.0, 1.0),
) -> Dict:
    """Compute an enhanced least cost path on a cost raster.

    Parameters
    ----------
    cost_raster : numpy.ndarray
        2-D array of traversal costs.  Higher values mean higher cost.
        Cells with ``numpy.nan`` or ``numpy.inf`` are treated as barriers.
    start : tuple[int, int]
        ``(row, col)`` of the start cell.
    end : tuple[int, int]
        ``(row, col)`` of the end cell.
    curvature_factor : float, optional
        Soft penalty weight for sharp turns (0.0 – 1.0, default 0.0).
        0.0 disables the penalty (standard LCP).  Higher values produce
        smoother, more gently curving paths.
    min_turning_angle : float, optional
        Minimum interior angle at any turn vertex in degrees (0 – 180,
        default 0).  0 allows any turn; higher values enforce gentler
        turns.  For example, ``min_turning_angle=135`` means the path
        can only deflect by up to 45° at each step.
    distance_factor : float, optional
        Weight for raw path length in the cost function (0.0 – 1.0,
        default 0.0).  Higher values encourage shorter paths even if they
        cross higher-cost cells.
    cell_size : tuple[float, float], optional
        ``(y_size, x_size)`` of each raster cell in map units.
        Default ``(1.0, 1.0)``.

    Returns
    -------
    dict
        ``path``            – list of ``(row, col)`` tuples from start to end.
        ``smoothed_path``   – list of ``(row, col)`` float tuples after
                              Chaikin corner-cutting smoothing.
        ``total_cost``      – accumulated cost along the optimal path.
        ``path_length``     – physical length of the path in map units.
        ``directions``      – direction index at each step.
        ``turning_angles``  – turning angle (degrees) at each interior vertex.

    Raises
    ------
    ValueError
        If any parameter is out of range or the start/end cell is invalid.
    RuntimeError
        If no path can be found between start and end.

    Notes
    -----
    **Cost function per step** (moving from cell *A* to neighbour *B* while
    arriving at *A* from direction *d_in* and leaving toward direction
    *d_out*)::

        base_cost        = cost_raster[B] * step_distance
        curvature_penalty = curvature_factor * amplifier
                            * (turning_angle / 180) * step_distance * cost_scale
        straightness_pen  = similarity * penalty * cost_scale * step_distance
        distance_penalty  = distance_factor * step_distance * cost_scale

        total_step_cost  = base_cost + curvature_penalty
                           + straightness_pen + distance_penalty

    where *cost_scale* is the mean of all finite cost-raster values,
    *similarity* measures how close the target-cell cost is to the
    straight-ahead cell cost (1 when equal, 0 when very different), and
    *amplifier* is an internal constant (5.0) that ensures ``curvature_factor``
    values between 0 and 1 produce a noticeable effect.

    When ``curvature_factor == 0`` **and** ``min_turning_angle == 0`` the
    algorithm automatically drops direction tracking, reducing memory and
    run-time to that of a standard Dijkstra LCP.

    A **Chaikin corner-cutting** post-processing pass is always applied to
    produce a ``smoothed_path`` with rounded turns instead of sharp corners.
    """
    # ---- Validate inputs --------------------------------------------------
    _validate_params(cost_raster, start, end, curvature_factor,
                     min_turning_angle, distance_factor)

    use_curvature = curvature_factor > 0.0 or min_turning_angle > 0.0

    if use_curvature:
        return _dijkstra_with_direction(
            cost_raster, start, end,
            curvature_factor, min_turning_angle, distance_factor, cell_size,
        )
    return _dijkstra_standard(
        cost_raster, start, end, distance_factor, cell_size,
    )


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


def _validate_params(
    cost_raster: np.ndarray,
    start: Tuple[int, int],
    end: Tuple[int, int],
    curvature_factor: float,
    min_turning_angle: float,
    distance_factor: float,
) -> None:
    if cost_raster.ndim != 2:
        raise ValueError("cost_raster must be a 2-D array")
    if not 0.0 <= curvature_factor <= 1.0:
        raise ValueError(
            f"curvature_factor must be between 0.0 and 1.0, got {curvature_factor}"
        )
    if not 0.0 <= min_turning_angle <= 180.0:
        raise ValueError(
            f"min_turning_angle must be between 0.0 and 180.0, got {min_turning_angle}"
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
# Cost-scale helper
# ---------------------------------------------------------------------------


def _cost_scale(cost_raster: np.ndarray) -> float:
    """Return the mean of all finite, non-negative values in the raster."""
    valid = cost_raster[np.isfinite(cost_raster)]
    valid = valid[valid >= 0]
    if valid.size == 0:
        return 1.0
    mean_val = float(np.mean(valid))
    return mean_val if mean_val > 0 else 1.0


# ---------------------------------------------------------------------------
# Standard Dijkstra (no direction tracking)
# ---------------------------------------------------------------------------


def _dijkstra_standard(
    cost_raster: np.ndarray,
    start: Tuple[int, int],
    end: Tuple[int, int],
    distance_factor: float,
    cell_size: Tuple[float, float],
) -> Dict:
    rows, cols = cost_raster.shape
    sr, sc = start
    er, ec = end

    step_dists = _step_distances(cell_size)
    scale = _cost_scale(cost_raster)
    dist_weight = distance_factor * scale

    best = np.full((rows, cols), np.inf)
    best[sr, sc] = 0.0
    parent: Dict[Tuple[int, int], Optional[Tuple[int, int]]] = {start: None}

    # counter prevents comparison of tuples when costs are equal
    counter = 0
    pq: List = [(0.0, counter, sr, sc)]

    while pq:
        cost, _, r, c = heapq.heappop(pq)
        if cost > best[r, c]:
            continue
        if r == er and c == ec:
            break

        for d in range(NUM_DIRS):
            dr, dc = DIRECTIONS[d]
            nr, nc = r + dr, c + dc
            if not (0 <= nr < rows and 0 <= nc < cols):
                continue
            cell_val = cost_raster[nr, nc]
            if not np.isfinite(cell_val) or cell_val < 0:
                continue
            sd = step_dists[d]
            step_cost = cell_val * sd + dist_weight * sd
            new_cost = cost + step_cost
            if new_cost < best[nr, nc]:
                best[nr, nc] = new_cost
                parent[(nr, nc)] = (r, c)
                counter += 1
                heapq.heappush(pq, (new_cost, counter, nr, nc))

    if not np.isfinite(best[er, ec]):
        raise RuntimeError("No path found between start and end")

    return _build_result(parent, start, end, best[er, ec], cell_size)


# ---------------------------------------------------------------------------
# Direction-aware Dijkstra (curvature support)
# ---------------------------------------------------------------------------


def _dijkstra_with_direction(
    cost_raster: np.ndarray,
    start: Tuple[int, int],
    end: Tuple[int, int],
    curvature_factor: float,
    min_turning_angle: float,
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
    # Maximum deflection allowed, derived from the minimum interior angle.
    max_deflection = 180.0 - min_turning_angle

    # best_cost shape: (rows, cols, NUM_DIRS + 1)
    # Index NUM_DIRS stores the NO_DIR sentinel for the start cell.
    n_states = NUM_DIRS + 1
    best = np.full((rows, cols, n_states), np.inf)
    best[sr, sc, NUM_DIRS] = 0.0  # start with NO_DIR

    # Parent map: (r, c, d) -> (pr, pc, pd) or None
    parent: Dict[Tuple[int, int, int], Optional[Tuple[int, int, int]]] = {
        (sr, sc, NO_DIR): None,
    }

    counter = 0
    pq: List = [(0.0, counter, sr, sc, NO_DIR)]

    found = False
    end_state: Optional[Tuple[int, int, int]] = None

    while pq:
        cost, _, r, c, d_in = heapq.heappop(pq)
        d_idx = d_in if d_in >= 0 else NUM_DIRS

        if cost > best[r, c, d_idx]:
            continue
        if r == er and c == ec:
            found = True
            end_state = (r, c, d_in)
            break

        for d_out in range(NUM_DIRS):
            dr, dc = DIRECTIONS[d_out]
            nr, nc = r + dr, c + dc
            if not (0 <= nr < rows and 0 <= nc < cols):
                continue
            cell_val = cost_raster[nr, nc]
            if not np.isfinite(cell_val) or cell_val < 0:
                continue

            sd = step_dists[d_out]
            base = cell_val * sd

            # Curvature penalty and hard turn constraint
            curv_penalty = 0.0
            if d_in != NO_DIR:
                angle = turning_angle(d_in, d_out)
                if angle > max_deflection:
                    continue
                curv_penalty = curv_weight * (angle / 180.0) * sd

            # Anti-zigzag: penalize direction changes when cost variation
            # is low (going straight would cost about the same).
            straightness_penalty = 0.0
            if d_in != NO_DIR and d_out != d_in:
                str_dr, str_dc = DIRECTIONS[d_in]
                str_r, str_c = r + str_dr, c + str_dc
                if (0 <= str_r < rows and 0 <= str_c < cols
                        and np.isfinite(cost_raster[str_r, str_c])
                        and cost_raster[str_r, str_c] >= 0):
                    cost_ratio = abs(cell_val - cost_raster[str_r, str_c]) / scale
                    similarity = max(0.0, 1.0 - cost_ratio)
                else:
                    similarity = 0.0
                straightness_penalty = similarity * _STRAIGHTNESS_PENALTY * scale * sd

            dist_penalty = dist_weight * sd
            new_cost = cost + base + curv_penalty + straightness_penalty + dist_penalty

            nd_idx = d_out
            if new_cost < best[nr, nc, nd_idx]:
                best[nr, nc, nd_idx] = new_cost
                parent[(nr, nc, d_out)] = (r, c, d_in)
                counter += 1
                heapq.heappush(pq, (new_cost, counter, nr, nc, d_out))

    if not found:
        raise RuntimeError("No path found between start and end")

    return _build_result_directed(parent, end_state, best, cell_size)


# ---------------------------------------------------------------------------
# Path reconstruction helpers
# ---------------------------------------------------------------------------


def _build_result(
    parent: Dict,
    start: Tuple[int, int],
    end: Tuple[int, int],
    total_cost: float,
    cell_size: Tuple[float, float],
) -> Dict:
    """Reconstruct path for the standard (non-directional) Dijkstra."""
    path: List[Tuple[int, int]] = []
    cur: Optional[Tuple[int, int]] = end
    while cur is not None:
        path.append(cur)
        cur = parent.get(cur)
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

    return {
        "path": path,
        "smoothed_path": smooth_path(path),
        "total_cost": total_cost,
        "path_length": path_length,
        "directions": directions,
        "turning_angles": turning_angles,
    }


def _build_result_directed(
    parent: Dict,
    end_state: Tuple[int, int, int],
    best: np.ndarray,
    cell_size: Tuple[float, float],
) -> Dict:
    """Reconstruct path for the direction-aware Dijkstra."""
    states: List[Tuple[int, int, int]] = []
    cur: Optional[Tuple[int, int, int]] = end_state
    while cur is not None:
        states.append(cur)
        cur = parent.get(cur)
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

    return {
        "path": path,
        "smoothed_path": smooth_path(path),
        "total_cost": total_cost,
        "path_length": path_length,
        "directions": directions,
        "turning_angles": turning_angles,
    }


# ---------------------------------------------------------------------------
# Path smoothing
# ---------------------------------------------------------------------------


def smooth_path(
    path: List[Tuple[int, int]],
    iterations: int = 3,
) -> List[Tuple[float, float]]:
    """Smooth a grid path into a curve using Chaikin's corner-cutting algorithm.

    Converts the discrete grid-cell path into a smooth curve by iteratively
    cutting corners.  Start and end points are preserved exactly.  The result
    replaces sharp angular turns with rounded arcs.

    Parameters
    ----------
    path : list[tuple[int, int]]
        Grid cells ``(row, col)`` from the pathfinding result.
    iterations : int, optional
        Number of smoothing passes (default 3).  More iterations produce
        smoother curves.

    Returns
    -------
    list[tuple[float, float]]
        Smoothed path as fractional ``(row, col)`` coordinates.
    """
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
            # Q point at 1/4 from p0 toward p1
            qr = 0.75 * r0 + 0.25 * r1
            qc = 0.75 * c0 + 0.25 * c1
            # R point at 3/4 from p0 toward p1
            rr = 0.25 * r0 + 0.75 * r1
            rc = 0.25 * c0 + 0.75 * c1
            new_pts.append((qr, qc))
            new_pts.append((rr, rc))
        new_pts.append(pts[-1])
        pts = new_pts

    return pts


# ---------------------------------------------------------------------------
# Raster file I/O helpers (require rasterio)
# ---------------------------------------------------------------------------


def load_cost_raster(filepath: str) -> Tuple[np.ndarray, Dict]:
    """Read a cost raster from a GeoTIFF or other GDAL-supported format.

    Parameters
    ----------
    filepath : str
        Path to the raster file.

    Returns
    -------
    tuple[numpy.ndarray, dict]
        ``(data, metadata)`` where *metadata* contains ``transform``,
        ``crs``, ``cell_size``, ``nodata``, ``width``, and ``height``.
    """
    try:
        import rasterio
    except ImportError as exc:
        raise ImportError(
            "rasterio is required for file I/O.  Install it with: "
            "pip install rasterio"
        ) from exc

    with rasterio.open(filepath) as src:
        data = src.read(1).astype(np.float64)
        nodata = src.nodata
        if nodata is not None:
            data[data == nodata] = np.nan
        transform = src.transform
        cell_size = (abs(transform.e), abs(transform.a))  # (y_size, x_size)
        metadata = {
            "transform": transform,
            "crs": src.crs,
            "cell_size": cell_size,
            "nodata": nodata,
            "width": src.width,
            "height": src.height,
        }
    return data, metadata


def save_path_raster(
    filepath: str,
    path: List[Tuple[int, int]],
    reference_metadata: Dict,
) -> None:
    """Save the path as a binary raster (1 = on path, 0 = off path).

    Parameters
    ----------
    filepath : str
        Output GeoTIFF path.
    path : list[tuple[int, int]]
        List of ``(row, col)`` cells on the path.
    reference_metadata : dict
        Metadata dict returned by :func:`load_cost_raster`.
    """
    try:
        import rasterio
    except ImportError as exc:
        raise ImportError(
            "rasterio is required for file I/O.  Install it with: "
            "pip install rasterio"
        ) from exc

    height = reference_metadata["height"]
    width = reference_metadata["width"]
    out = np.zeros((height, width), dtype=np.uint8)
    for r, c in path:
        out[r, c] = 1

    with rasterio.open(
        filepath,
        "w",
        driver="GTiff",
        height=height,
        width=width,
        count=1,
        dtype="uint8",
        crs=reference_metadata.get("crs"),
        transform=reference_metadata.get("transform"),
    ) as dst:
        dst.write(out, 1)
