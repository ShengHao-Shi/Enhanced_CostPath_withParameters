# CostPath_withParameters

A GIS tool that extends the standard Least Cost Path (LCP) algorithm with
**curvature control** and a **distance factor**, addressing limitations of
ESRI's built-in LCP tool.

---

## Motivation

The ArcGIS *Least Cost Path* tool only exposes three inputs — a cost raster,
a start point, and an end point — plus a boolean option for handling zero
values.  In practice two additional controls are needed:

| Need | Why |
|---|---|
| **Curvature / turn control** | Standard LCP often produces paths with abrupt, sharp turns that are unrealistic for roads, pipelines, or similar linear features. |
| **Distance weighting** | Standard LCP considers only the cost surface; it may route a path through a long detour of low-cost cells when a slightly more expensive but much shorter route would be preferable. |

## Approach — ArcGIS vs. Standalone

| Option | Verdict |
|---|---|
| **ArcGIS / ArcPy** | ArcPy can read rasters and feature classes, but ESRI's own cost-distance / cost-path functions do **not** expose curvature or distance parameters.  A custom algorithm must be written in Python regardless. |
| **Standalone (GDAL / rasterio + NumPy)** | More portable, no licence dependency, easier to test and deploy.  Recommended as the primary implementation. |
| **Best of both** | Implement the core algorithm with NumPy only, then provide a thin ArcGIS Python Toolbox (`.pyt`) wrapper for users who prefer the ArcGIS Pro UI. |

This repository follows the *best-of-both* strategy.

---

## Repository Layout

```
CostPath_withParameters/
├── enhanced_lcp.py        # Core algorithm (NumPy only, no ArcGIS dependency)
├── arcgis_toolbox.pyt     # ArcGIS Python Toolbox wrapper (requires arcpy)
├── requirements.txt       # Python dependencies for standalone use
├── tests/
│   └── test_enhanced_lcp.py
└── README.md
```

## Algorithm

The tool uses a **modified Dijkstra's algorithm** on a raster grid with
8-connectivity.  When curvature control is active the search state is
extended to include the incoming travel direction, enabling the algorithm to
compute turning angles.

### Cost Function

For each step from cell *A* to neighbour *B*, arriving at *A* from direction
*d_in* and leaving toward direction *d_out*:

```
step_cost = base_cost + curvature_penalty + distance_penalty
```

| Component | Formula | When active |
|---|---|---|
| **base_cost** | `cost_raster[B] × step_distance` | Always |
| **curvature_penalty** | `curvature_factor × amplifier × (angle / 180) × step_distance × cost_scale` | `curvature_factor > 0` |
| **distance_penalty** | `distance_factor × step_distance × cost_scale` | `distance_factor > 0` |

* `step_distance` — Euclidean distance between cell centres (1 for cardinal,
  √2 for diagonal, scaled by `cell_size`).
* `cost_scale` — mean of all finite cost-raster values; used to keep penalty
  terms in the same order of magnitude as the base cost.
* `amplifier` — internal constant (5.0) that makes `curvature_factor` values
  between 0 and 1 produce a visible effect.

### Hard Turn Constraint

If `max_turning_angle` is set below 180°, any transition whose turning angle
exceeds that threshold is simply disallowed (pruned from the search).

### Performance Optimisation

When `curvature_factor == 0` **and** `max_turning_angle == 180`, the
algorithm automatically falls back to a standard (direction-free) Dijkstra
search with a smaller state space.

---

## Parameters

| Parameter | Type | Range | Default | Description |
|---|---|---|---|---|
| `cost_raster` | 2-D NumPy array | — | *(required)* | Traversal cost surface.  `NaN` / `Inf` cells are barriers. |
| `start` | `(row, col)` | — | *(required)* | Start cell. |
| `end` | `(row, col)` | — | *(required)* | End cell. |
| `curvature_factor` | float | 0.0 – 1.0 | 0.0 | Soft penalty weight for sharp turns.  0 = standard LCP. |
| `max_turning_angle` | float | 0 – 180 | 180.0 | Hard limit on turning angle (degrees).  180 = unrestricted. |
| `distance_factor` | float | 0.0 – 1.0 | 0.0 | Weight for raw path length.  Higher ⇒ shorter paths preferred. |
| `cell_size` | `(y, x)` | — | `(1, 1)` | Physical cell dimensions in map units. |

## Output

A dictionary with:

| Key | Type | Description |
|---|---|---|
| `path` | `list[(row, col)]` | Ordered cell coordinates from start to end. |
| `total_cost` | `float` | Accumulated cost along the path. |
| `path_length` | `float` | Physical path length in map units. |
| `directions` | `list[int]` | Direction index (0–7) at each step. |
| `turning_angles` | `list[float]` | Turning angle in degrees at each interior vertex. |

---

## Quick Start (Standalone)

```bash
pip install numpy rasterio
```

```python
import numpy as np
from enhanced_lcp import enhanced_least_cost_path

# Example: 50×50 cost raster with random costs 1–10
raster = np.random.default_rng(0).uniform(1, 10, (50, 50))

result = enhanced_least_cost_path(
    raster,
    start=(0, 0),
    end=(49, 49),
    curvature_factor=0.5,      # moderate smoothing
    max_turning_angle=90.0,    # no turns sharper than 90°
    distance_factor=0.3,       # mildly prefer shorter paths
)

print(f"Path length : {result['path_length']:.1f}")
print(f"Total cost  : {result['total_cost']:.1f}")
print(f"Max turn    : {max(result['turning_angles']):.0f}°")
```

## Quick Start (ArcGIS Pro)

1. Copy `enhanced_lcp.py` and `arcgis_toolbox.pyt` into the same folder.
2. In ArcGIS Pro → Catalog → Toolboxes → **Add Toolbox** → select
   `arcgis_toolbox.pyt`.
3. Open the *Enhanced Least Cost Path* tool and fill in the parameters.

## File I/O Helpers

```python
from enhanced_lcp import load_cost_raster, save_path_raster

data, meta = load_cost_raster("cost_surface.tif")
result = enhanced_least_cost_path(data, (10, 20), (200, 300),
                                  cell_size=meta["cell_size"])
save_path_raster("path_output.tif", result["path"], meta)
```

---

## Running Tests

```bash
pip install pytest numpy
python -m pytest tests/ -v
```

---

## Licence

MIT
