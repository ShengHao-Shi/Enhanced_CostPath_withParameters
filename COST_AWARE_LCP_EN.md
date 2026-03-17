# Cost-Aware Straightened Least Cost Path — Technical Documentation

> This document explains the working principle and parameter tuning of the
> **Cost-Aware Straightened LCP** tool implemented in
> `cost_aware_straighten_lcp.py`.

---

## Table of Contents

1. [Overview](#1-overview)
2. [Why Cost-Aware Straightening?](#2-why-cost-aware-straightening)
3. [Algorithm Pipeline](#3-algorithm-pipeline)
   - 3.1 [Step 1 — Dijkstra Path Search](#31-step-1--dijkstra-path-search)
   - 3.2 [Step 2 — Cost-Aware Straightening](#32-step-2--cost-aware-straightening)
   - 3.3 [Step 3 — Chaikin Smoothing](#33-step-3--chaikin-smoothing)
4. [Cost Function Details](#4-cost-function-details)
5. [Parameter Reference & Tuning Guide](#5-parameter-reference--tuning-guide)
   - 5.1 [cost_raster](#51-cost_raster)
   - 5.2 [start / end](#52-start--end)
   - 5.3 [curvature_factor](#53-curvature_factor)
   - 5.4 [max_turning_angle](#54-max_turning_angle)
   - 5.5 [distance_factor](#55-distance_factor)
   - 5.6 [straighten_factor](#56-straighten_factor)
   - 5.7 [cost_tolerance](#57-cost_tolerance)
   - 5.8 [cell_size](#58-cell_size)
6. [Output Dictionary](#6-output-dictionary)
7. [Quick-Start Example](#7-quick-start-example)
8. [Comparison with enhanced_lcp.py](#8-comparison-with-enhanced_lcppy)

---

## 1. Overview

`cost_aware_straighten_lcp.py` provides a least-cost-path algorithm that
produces visually clean, cost-efficient paths on raster cost surfaces.  It
was developed as **Approach C (方案C)** to address the "lightning-bolt"
zigzag artifacts inherent to grid-based pathfinding while ensuring that the
straightened path does not silently traverse expensive cells.

The core idea is:

> Keep the robust 8-direction Dijkstra search, but **upgrade the
> post-processing straightening** so that every proposed shortcut is
> evaluated against the original path cost — not only checked for NODATA
> barriers.

---

## 2. Why Cost-Aware Straightening?

Standard grid-based pathfinding on 8-connected rasters produces paths that
can only move in 8 directions (N, NE, E, SE, S, SW, W, NW).  This creates
staircasing / zigzag artifacts that look unnatural.

The original `enhanced_lcp.py` addresses this with a **line-of-sight
straightening** post-process: it tries to skip intermediate waypoints and
draw direct lines, only rejecting shortcuts that cross NODATA / barrier
cells.

However, "NODATA-only" checking has a blind spot: a shortcut line may pass
through cells that are *valid* but have very high cost.  The result is a
visually straight path that is economically poor.

**Cost-aware straightening** closes this gap.  Before accepting a shortcut
it computes the accumulated cost along the shortcut line and compares it to
the original grid-path cost for the same segment.  Only shortcuts within a
configurable tolerance ratio are accepted.

```
Original grid path:   A ──► B ──► C ──► D ──► E
Proposed shortcut:    A ─────────────────────► E

Accept shortcut only if:
  cost(shortcut) ≤ cost(A→B→C→D→E) × cost_tolerance
```

---

## 3. Algorithm Pipeline

The algorithm proceeds in three sequential stages.

### 3.1 Step 1 — Dijkstra Path Search

An **8-direction modified Dijkstra / A\*** search finds the optimal
grid-cell path from start to end.  This step is identical to
`enhanced_lcp.py` and supports:

- **Base cost**: `cost_raster[cell] × step_distance`
- **Curvature penalty** (when `curvature_factor > 0`)
- **Hard turn constraint** (when `max_turning_angle < 180`)
- **Anti-zigzag straightness penalty**: discourages direction changes when
  adjacent cell costs are similar
- **Distance penalty** (when `distance_factor > 0`)

When curvature control is not needed (`curvature_factor == 0` and
`max_turning_angle == 180`), the algorithm automatically uses a simpler
direction-free Dijkstra variant with lower memory usage.

### 3.2 Step 2 — Cost-Aware Straightening

After the grid path is found, the **cost-aware straightening** removes
unnecessary waypoints:

1. **Cumulative cost pre-computation**: Walk along the grid path and
   accumulate `cost_raster[cell] × step_distance` at each step.  This
   allows O(1) lookup of the original-path cost for any sub-segment.

2. **Greedy shortcut scan**: Starting from the first waypoint, look ahead
   up to `max_skip = max(2, int(straighten_factor × (n − 1)))` positions
   (scanning farthest first).  For each candidate endpoint:
   - **Barrier check**: Use the supercover line algorithm to verify that
     the straight line does not cross any NODATA / out-of-bounds cell.
   - **Cost check**: Compute the integrated cost along the shortcut line
     (average cell cost × Euclidean distance) and compare it to
     `original_segment_cost × cost_tolerance`.
   - If both checks pass, accept the shortcut and advance to the new
     position.

3. The result is a reduced list of waypoints connected by straight-line
   segments that are both passable and economically acceptable.

### 3.3 Step 3 — Chaikin Smoothing

**Chaikin's corner-cutting algorithm** (3 iterations by default) is applied
to the straightened path to round sharp corners into smooth arcs.  After
smoothing, a NODATA safety check verifies every segment; if any smoothed
segment crosses a barrier, the function falls back to the un-smoothed
straightened path.

---

## 4. Cost Function Details

The per-step cost in the Dijkstra search is:

```
step_cost = base_cost
          + curvature_penalty
          + straightness_penalty
          + distance_penalty
```

| Component | Formula | Active When |
|---|---|---|
| **base_cost** | `cost_raster[B] × step_distance` | Always |
| **curvature_penalty** | `curvature_factor × 5.0 × cost_scale × (angle / 180) × step_distance` | `curvature_factor > 0` |
| **straightness_penalty** | `similarity × 0.3 × cost_scale × step_distance` | Direction changed & direction-aware mode |
| **distance_penalty** | `distance_factor × cost_scale × step_distance` | `distance_factor > 0` |

- **step_distance** — Euclidean distance between cell centres (1.0 for
  cardinal, √2 for diagonal, scaled by `cell_size`).
- **cost_scale** — Mean of all finite, non-negative raster values.
  Normalises penalty terms to the same magnitude as the base cost.
- **similarity** — `max(0, 1 − |cost_target − cost_straight| / cost_scale)`.
  High when the target cell and the straight-ahead cell have similar cost;
  suppresses the anti-zigzag penalty when costs differ substantially.

---

## 5. Parameter Reference & Tuning Guide

### 5.1 `cost_raster`

| | |
|---|---|
| **Type** | 2-D NumPy array (`float32` or `float64`) |
| **Required** | Yes |
| **Description** | The traversal cost surface. Each cell value represents the cost of crossing that cell. `NaN` and `Inf` values are treated as impassable barriers. Negative values are also treated as barriers. |

**Tuning notes**:
- Ensure barrier cells use `NaN`, not zero (zero is a valid low cost).
- Large rasters (> 5000 × 5000) are supported but may require several
  seconds.  Use `float32` to reduce memory.

---

### 5.2 `start` / `end`

| | |
|---|---|
| **Type** | `(row, col)` integer tuple |
| **Required** | Yes |
| **Description** | The start and end cells of the path, in zero-based row/column indices. Both must fall within the raster bounds and on finite-cost cells. |

---

### 5.3 `curvature_factor`

| | |
|---|---|
| **Type** | `float` |
| **Range** | 0.0 – 1.0 |
| **Default** | 0.0 |
| **Description** | Soft penalty weight for sharp turns. Higher values produce smoother, more gently curving paths but may increase total cost. |

**Tuning guide**:

| Value | Effect |
|---|---|
| **0.0** | No curvature penalty — standard LCP behaviour. Fastest. |
| **0.1 – 0.3** | Mild smoothing. Reduces the most extreme zigzags without significantly altering the route. Good starting point for most use cases. |
| **0.4 – 0.6** | Moderate smoothing. Paths clearly avoid sharp turns but may detour around cost valleys. |
| **0.7 – 1.0** | Aggressive smoothing. Very gentle curves; suited for pipelines or highways where wide turning radii are critical. Expect noticeable cost increase. |

> **Tip**: Start with `0.0` and increase in steps of `0.1` until the visual
> result meets your needs.  Combine with `max_turning_angle` for a hard
> constraint.

---

### 5.4 `max_turning_angle`

| | |
|---|---|
| **Type** | `float` |
| **Range** | 0 – 180 degrees |
| **Default** | 180.0 |
| **Description** | Hard upper limit on the turning angle at each grid step. Any turn exceeding this angle is completely forbidden during the search. |

**Tuning guide**:

| Value | Meaning |
|---|---|
| **180** | No constraint — all turns allowed. |
| **135** | Forbids U-turns (180° turns). Mild constraint. |
| **90** | Only forward or sideways movement; no backward turns. Produces broadly sweeping paths. |
| **45** | Only straight or slight turns. Very constrained — may fail to find a path on complex rasters. |

> **Warning**: Very low values (< 45°) may cause "No path found" errors
> if the start/end configuration requires sharp turns.  Always pair with
> a reasonable `curvature_factor` instead of relying solely on this
> parameter.

---

### 5.5 `distance_factor`

| | |
|---|---|
| **Type** | `float` |
| **Range** | 0.0 – 1.0 |
| **Default** | 0.0 |
| **Description** | Weight for raw Euclidean path length in the cost function. Higher values encourage shorter paths, even if they cross slightly more expensive cells. |

**Tuning guide**:

| Value | Effect |
|---|---|
| **0.0** | Pure cost optimisation — the cheapest path regardless of length. |
| **0.1 – 0.3** | Mild shortening. Eliminates major detours while still preferring low-cost corridors. Recommended starting range. |
| **0.4 – 0.6** | Balanced: roughly equal weight on cost and distance. |
| **0.7 – 1.0** | Strong distance preference. The path tends toward a straight line; cost is downweighted. |

> **Tip**: If the path makes a large detour through a low-cost area, try
> `distance_factor = 0.2` to pull it back toward a more direct route.

---

### 5.6 `straighten_factor`

| | |
|---|---|
| **Type** | `float` |
| **Range** | 0.0 – 0.5 |
| **Default** | 0.3 |
| **Description** | Controls how far ahead the straightening step looks when trying to replace grid-aligned segments with straight-line shortcuts. Internally converted to a max lookahead of `max(2, int(factor × (n − 1)))` path indices. |

**Tuning guide**:

| Value | Effect |
|---|---|
| **0.0** | No straightening — the output retains all 8-direction grid steps. Use this to inspect the raw Dijkstra path. |
| **0.05 – 0.15** | Conservative: only very short zigzag segments are straightened. The path stays very close to the original grid path. |
| **0.2 – 0.3** | Moderate (default range): produces visually clean paths while respecting cost barriers. Recommended for most use cases. |
| **0.4 – 0.5** | Aggressive: allows long straight-line shortcuts. Good for open, uniform-cost areas but may over-simplify paths in complex terrain. |

> **Interaction with `cost_tolerance`**: A large `straighten_factor` only
> has effect if `cost_tolerance` also allows the longer shortcuts.  If
> `cost_tolerance` is very tight (e.g. 1.0), even far-ahead shortcuts will
> be rejected whenever they cost slightly more than the original.

---

### 5.7 `cost_tolerance`

| | |
|---|---|
| **Type** | `float` |
| **Range** | ≥ 1.0 |
| **Default** | 1.05 |
| **Description** | Maximum allowed ratio of shortcut cost to original grid-path cost for the same segment. This is the **key parameter unique to this tool**. |

**How it works**:

```
Accept shortcut A → E  ⟺  cost(shortcut) ≤ cost(A→…→E) × cost_tolerance
```

**Tuning guide**:

| Value | Meaning | Use Case |
|---|---|---|
| **1.0** | Only accept shortcuts that are exactly as cheap or cheaper than the original path segment. Very strict — may result in little or no straightening on non-uniform surfaces. | When absolute cost optimality is critical. |
| **1.01 – 1.05** | Allow 1 – 5% cost overhead. Good balance between visual quality and cost efficiency. **Recommended default range.** | General-purpose use. |
| **1.05 – 1.15** | Allow 5 – 15% overhead. More aggressive straightening; the path becomes visually cleaner but may pay a moderate cost premium. | When visual straightness matters more than minimal cost. |
| **1.2 – 1.5** | Allow 20 – 50% overhead. Strongly favours straight paths; significant cost increase possible. | Aesthetics-focused or when the cost surface has low variation. |
| **2.0+** | Essentially disables cost checking — behaves like the original NODATA-only straightening in `enhanced_lcp.py`. | When you want maximum straightening regardless of cost. |

> **Tip**: For most GIS applications, start with the default `1.05`.  If
> the path still shows visible zigzags, increase to `1.10`.  If the path
> cuts through high-cost areas unacceptably, lower to `1.02` or `1.0`.

---

### 5.8 `cell_size`

| | |
|---|---|
| **Type** | `(y_size, x_size)` float tuple |
| **Default** | `(1.0, 1.0)` |
| **Description** | Physical dimensions of each raster cell in map units (e.g. metres). Used to compute accurate Euclidean distances for step costs, the distance penalty, and straightening line costs. |

**Tuning notes**:
- If using the ArcGIS toolbox wrapper, `cell_size` is read automatically
  from the input raster.
- For standalone use with a GeoTIFF, extract the cell size from the raster
  metadata (e.g. via `rasterio`).
- Ensure `y_size` and `x_size` are in the same units as the cost raster
  values are intended to be scaled to.

---

## 6. Output Dictionary

The function returns a dictionary with the following keys:

| Key | Type | Description |
|---|---|---|
| `path` | `list[(int, int)]` | Grid-cell path from start to end (8-connected). |
| `straightened_path` | `list[(float, float)]` | Path after cost-aware straightening — reduced waypoints with straight-line connections. |
| `smoothed_path` | `list[(float, float)]` | Chaikin-smoothed version of the straightened path with rounded turns. |
| `total_cost` | `float` | Accumulated cost along the optimal grid path. |
| `path_length` | `float` | Physical length of the grid path in map units. |
| `directions` | `list[int]` | Direction index (0–7) at each step of the grid path. |
| `turning_angles` | `list[float]` | Turning angle in degrees at each interior vertex of the grid path. |

---

## 7. Quick-Start Example

```python
import numpy as np
from cost_aware_straighten_lcp import cost_aware_least_cost_path

# Create a sample cost raster
raster = np.random.default_rng(42).uniform(1, 10, (100, 100))

result = cost_aware_least_cost_path(
    raster,
    start=(0, 0),
    end=(99, 99),
    curvature_factor=0.2,
    max_turning_angle=90.0,
    distance_factor=0.1,
    straighten_factor=0.3,
    cost_tolerance=1.05,
)

print(f"Grid path cells    : {len(result['path'])}")
print(f"Straightened points: {len(result['straightened_path'])}")
print(f"Smoothed points    : {len(result['smoothed_path'])}")
print(f"Total cost         : {result['total_cost']:.2f}")
print(f"Path length        : {result['path_length']:.2f}")
```

---

## 8. Comparison with enhanced_lcp.py

| Feature | `enhanced_lcp.py` | `cost_aware_straighten_lcp.py` |
|---|---|---|
| Dijkstra search | 8-direction, curvature-aware | **Identical** |
| Straightening check | NODATA barriers only | NODATA barriers **+ accumulated cost comparison** |
| Extra parameter | — | `cost_tolerance` (≥ 1.0) |
| Risk of crossing high-cost cells after straightening | Possible | Controlled by `cost_tolerance` |
| Performance | Slightly faster (no cost integration during straightening) | Slightly slower (supercover cost summation per shortcut candidate) |
| Best for | Uniform or low-variation cost surfaces | Non-uniform surfaces where cost fidelity matters |
