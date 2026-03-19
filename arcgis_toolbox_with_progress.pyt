"""
ArcGIS Python Toolbox – Cost-Aware LCP with Progress Reporting
===============================================================

This ``.pyt`` file provides **two** geoprocessing tools that report
step-by-step progress in the ArcGIS Geoprocessing pane:

1. **Cost-Aware LCP (Pure Python)** — uses
   ``pure_python.cost_aware_straighten_lcp`` with progress messages
   throughout the Dijkstra search, straightening, and smoothing phases.

2. **Cost-Aware LCP (Numba Accelerated)** — uses
   ``numba_accelerated.cost_aware_straighten_lcp`` for 20–50× faster
   execution on large rasters, with progress messages before/after
   each computational phase.

Both tools accept the same parameters and produce the same output
format.  The progress messages appear as text lines in the
*Geoprocessing → Messages* section in ArcGIS Pro.

Usage
-----
1. In ArcGIS Pro, open the *Catalog* pane.
2. Right-click **Toolboxes** → **Add Toolbox** and select this file.
3. Expand the toolbox and double-click the desired tool.

Requirements
------------
* ArcGIS Pro with Python 3 (arcpy must be available).
* ``pure_python/cost_aware_straighten_lcp.py`` (always required).
* ``numba_accelerated/cost_aware_straighten_lcp.py`` + ``numba``
  (only required for the Numba-accelerated tool).
* ``numpy`` (bundled with ArcGIS Pro).
"""

import importlib
import os
import sys
import time

import arcpy
import numpy as np

# Ensure the toolbox directory is on sys.path so packages can be imported.
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
if _THIS_DIR not in sys.path:
    sys.path.insert(0, _THIS_DIR)

import pure_python.cost_aware_straighten_lcp as _pure_python_mod  # noqa: E402


class Toolbox:
    """ArcGIS Python Toolbox container."""

    def __init__(self):
        self.label = "Cost-Aware LCP with Progress"
        self.alias = "CostAwareLCPProgress"
        self.tools = [
            CostAwareLCPTool,
            CostAwareNumbaLCPTool,
        ]


# ---------------------------------------------------------------------------
# Helper functions (shared by both tools)
# ---------------------------------------------------------------------------


def _fc_to_point(fc_path):
    """Extract the first point geometry from a feature class."""
    with arcpy.da.SearchCursor(fc_path, ["SHAPE@XY"]) as cur:
        for row in cur:
            return row[0]
    raise ValueError(f"No features found in {fc_path}")


def _xy_to_rowcol(xy, extent, cell_x, cell_y, shape):
    """Convert map coordinates to raster (row, col)."""
    x, y = xy
    col = int((x - extent.XMin) / cell_x)
    row = int((extent.YMax - y) / cell_y)
    rows, cols = shape
    row = max(0, min(row, rows - 1))
    col = max(0, min(col, cols - 1))
    return (row, col)


def _write_polyline(path, extent, cell_x, cell_y, sr, output_fc):
    """Write the path as a single polyline feature class."""
    points = []
    for r, c in path:
        x = extent.XMin + (c + 0.5) * cell_x
        y = extent.YMax - (r + 0.5) * cell_y
        points.append(arcpy.Point(x, y))

    polyline = arcpy.Polyline(arcpy.Array(points), sr)

    out_dir = os.path.dirname(output_fc)
    out_name = os.path.basename(output_fc)
    arcpy.management.CreateFeatureclass(
        out_dir, out_name, "POLYLINE", spatial_reference=sr,
    )
    with arcpy.da.InsertCursor(output_fc, ["SHAPE@"]) as cur:
        cur.insertRow([polyline])


def _make_cost_aware_params():
    """Create the parameter list shared by both tools."""
    params = []

    p_cost = arcpy.Parameter(
        displayName="Cost Raster",
        name="cost_raster",
        datatype="GPRasterLayer",
        parameterType="Required",
        direction="Input",
    )
    params.append(p_cost)

    p_start = arcpy.Parameter(
        displayName="Start Point",
        name="start_point",
        datatype="GPFeatureLayer",
        parameterType="Required",
        direction="Input",
    )
    p_start.filter.list = ["Point"]
    params.append(p_start)

    p_end = arcpy.Parameter(
        displayName="End Point",
        name="end_point",
        datatype="GPFeatureLayer",
        parameterType="Required",
        direction="Input",
    )
    p_end.filter.list = ["Point"]
    params.append(p_end)

    p_curv = arcpy.Parameter(
        displayName="Curvature Factor (0.0 – 1.0)",
        name="curvature_factor",
        datatype="GPDouble",
        parameterType="Optional",
        direction="Input",
    )
    p_curv.value = 0.0
    p_curv.filter.type = "Range"
    p_curv.filter.list = [0.0, 1.0]
    params.append(p_curv)

    p_angle = arcpy.Parameter(
        displayName="Maximum Turning Angle (degrees, 0 – 180)",
        name="max_turning_angle",
        datatype="GPDouble",
        parameterType="Optional",
        direction="Input",
    )
    p_angle.value = 180.0
    p_angle.filter.type = "Range"
    p_angle.filter.list = [0.0, 180.0]
    params.append(p_angle)

    p_dist = arcpy.Parameter(
        displayName="Distance Factor (0.0 – 1.0)",
        name="distance_factor",
        datatype="GPDouble",
        parameterType="Optional",
        direction="Input",
    )
    p_dist.value = 0.0
    p_dist.filter.type = "Range"
    p_dist.filter.list = [0.0, 1.0]
    params.append(p_dist)

    p_straighten = arcpy.Parameter(
        displayName="Straighten Factor (0.00 – 0.50)",
        name="straighten_factor",
        datatype="GPDouble",
        parameterType="Optional",
        direction="Input",
    )
    p_straighten.value = 0.3
    p_straighten.filter.type = "Range"
    p_straighten.filter.list = [0.0, 0.5]
    params.append(p_straighten)

    p_tolerance = arcpy.Parameter(
        displayName="Cost Tolerance (>= 1.0)",
        name="cost_tolerance",
        datatype="GPDouble",
        parameterType="Optional",
        direction="Input",
    )
    p_tolerance.value = 1.05
    p_tolerance.filter.type = "Range"
    p_tolerance.filter.list = [1.0, 100.0]
    params.append(p_tolerance)

    p_out = arcpy.Parameter(
        displayName="Output Path Feature Class",
        name="output_path",
        datatype="DEFeatureClass",
        parameterType="Required",
        direction="Output",
    )
    params.append(p_out)

    return params


def _read_inputs(parameters):
    """Parse tool parameters into a dict of algorithm inputs."""
    cost_raster_path = parameters[0].valueAsText
    start_fc = parameters[1].valueAsText
    end_fc = parameters[2].valueAsText
    curvature_factor = float(parameters[3].value or 0.0)
    max_turning_angle_val = parameters[4].value
    max_turning_angle = float(
        max_turning_angle_val if max_turning_angle_val is not None else 180.0
    )
    distance_factor = float(parameters[5].value or 0.0)
    straighten_factor_val = parameters[6].value
    straighten_factor = float(
        straighten_factor_val if straighten_factor_val is not None else 0.3
    )
    cost_tolerance_val = parameters[7].value
    cost_tolerance = float(
        cost_tolerance_val if cost_tolerance_val is not None else 1.05
    )
    output_fc = parameters[8].valueAsText

    raster = arcpy.Raster(cost_raster_path)
    cost_array = arcpy.RasterToNumPyArray(raster, nodata_to_value=np.nan)
    cost_array = cost_array.astype(np.float32)
    cell_x = raster.meanCellWidth
    cell_y = raster.meanCellHeight
    extent = raster.extent
    sr = raster.spatialReference

    start_pt = _fc_to_point(start_fc)
    end_pt = _fc_to_point(end_fc)
    start_rc = _xy_to_rowcol(start_pt, extent, cell_x, cell_y,
                              cost_array.shape)
    end_rc = _xy_to_rowcol(end_pt, extent, cell_x, cell_y,
                            cost_array.shape)

    return {
        "cost_array": cost_array,
        "start_rc": start_rc,
        "end_rc": end_rc,
        "curvature_factor": curvature_factor,
        "max_turning_angle": max_turning_angle,
        "distance_factor": distance_factor,
        "straighten_factor": straighten_factor,
        "cost_tolerance": cost_tolerance,
        "cell_size": (cell_y, cell_x),
        "output_fc": output_fc,
        "extent": extent,
        "cell_x": cell_x,
        "cell_y": cell_y,
        "sr": sr,
    }


# =========================================================================
# Tool 1: Cost-Aware LCP (Pure Python) with Progress
# =========================================================================

class CostAwareLCPTool:
    """ArcGIS tool for cost-aware LCP (pure Python) with progress reporting."""

    def __init__(self):
        self.label = "Cost-Aware LCP (Pure Python)"
        self.description = (
            "Compute a least cost path using the standard 8-direction "
            "Dijkstra search with cost-aware straightening.  "
            "Reports step-by-step progress in the Messages pane.  "
            "Use the Numba-accelerated variant for faster execution "
            "on large rasters."
        )
        self.canRunInBackground = True

    def getParameterInfo(self):  # noqa: N802
        return _make_cost_aware_params()

    def isLicensed(self):  # noqa: N802
        return True

    def updateParameters(self, parameters):  # noqa: N802
        return

    def updateMessages(self, parameters):  # noqa: N802
        return

    def execute(self, parameters, messages):  # noqa: N802
        importlib.reload(_pure_python_mod)

        inputs = _read_inputs(parameters)

        messages.addMessage(
            f"[Cost-Aware / Pure Python] 开始计算...\n"
            f"  起点: {inputs['start_rc']}, 终点: {inputs['end_rc']}\n"
            f"  栅格大小: {inputs['cost_array'].shape}"
        )

        t0 = time.time()

        result = _pure_python_mod.cost_aware_least_cost_path(
            inputs["cost_array"],
            inputs["start_rc"],
            inputs["end_rc"],
            curvature_factor=inputs["curvature_factor"],
            max_turning_angle=inputs["max_turning_angle"],
            distance_factor=inputs["distance_factor"],
            straighten_factor=inputs["straighten_factor"],
            cost_tolerance=inputs["cost_tolerance"],
            cell_size=inputs["cell_size"],
            progress_callback=messages.addMessage,
        )

        elapsed = time.time() - t0

        messages.addMessage(
            f"[Cost-Aware / Pure Python] 计算完成 (耗时 {elapsed:.1f}s)\n"
            f"  路径节点: {len(result['path'])} 个\n"
            f"  拉直后节点: {len(result['straightened_path'])} 个\n"
            f"  平滑后节点: {len(result['smoothed_path'])} 个\n"
            f"  总成本: {result['total_cost']:.2f}\n"
            f"  路径长度: {result['path_length']:.2f}"
        )

        _write_polyline(
            result["smoothed_path"],
            inputs["extent"],
            inputs["cell_x"],
            inputs["cell_y"],
            inputs["sr"],
            inputs["output_fc"],
        )
        messages.addMessage(f"输出已写入: {inputs['output_fc']}")

    def postExecute(self, parameters):  # noqa: N802
        return


# =========================================================================
# Tool 2: Cost-Aware LCP (Numba Accelerated) with Progress
# =========================================================================

class CostAwareNumbaLCPTool:
    """ArcGIS tool for Numba-accelerated cost-aware LCP with progress."""

    def __init__(self):
        self.label = "Cost-Aware LCP (Numba Accelerated)"
        self.description = (
            "Compute a least cost path using the Numba JIT-compiled "
            "8-direction Dijkstra search with cost-aware straightening.  "
            "20–50× faster than the pure Python variant on large rasters.  "
            "Reports step-by-step progress in the Messages pane.  "
            "Note: the first call has ~6s JIT compilation overhead."
        )
        self.canRunInBackground = True

    def getParameterInfo(self):  # noqa: N802
        return _make_cost_aware_params()

    def isLicensed(self):  # noqa: N802
        return True

    def updateParameters(self, parameters):  # noqa: N802
        return

    def updateMessages(self, parameters):  # noqa: N802
        return

    def execute(self, parameters, messages):  # noqa: N802
        # Import here so the toolbox still loads even if numba is missing
        try:
            import numba_accelerated.cost_aware_straighten_lcp as _numba_mod
            importlib.reload(_numba_mod)
        except ImportError:
            messages.addErrorMessage(
                "无法导入 numba_accelerated 模块。请确保 numba 已安装: "
                "pip install numba"
            )
            raise

        inputs = _read_inputs(parameters)

        messages.addMessage(
            f"[Cost-Aware / Numba] 开始计算...\n"
            f"  起点: {inputs['start_rc']}, 终点: {inputs['end_rc']}\n"
            f"  栅格大小: {inputs['cost_array'].shape}"
        )

        t0 = time.time()

        result = _numba_mod.cost_aware_least_cost_path(
            inputs["cost_array"],
            inputs["start_rc"],
            inputs["end_rc"],
            curvature_factor=inputs["curvature_factor"],
            max_turning_angle=inputs["max_turning_angle"],
            distance_factor=inputs["distance_factor"],
            straighten_factor=inputs["straighten_factor"],
            cost_tolerance=inputs["cost_tolerance"],
            cell_size=inputs["cell_size"],
            progress_callback=messages.addMessage,
        )

        elapsed = time.time() - t0

        messages.addMessage(
            f"[Cost-Aware / Numba] 计算完成 (耗时 {elapsed:.1f}s)\n"
            f"  路径节点: {len(result['path'])} 个\n"
            f"  拉直后节点: {len(result['straightened_path'])} 个\n"
            f"  平滑后节点: {len(result['smoothed_path'])} 个\n"
            f"  总成本: {result['total_cost']:.2f}\n"
            f"  路径长度: {result['path_length']:.2f}"
        )

        _write_polyline(
            result["smoothed_path"],
            inputs["extent"],
            inputs["cell_x"],
            inputs["cell_y"],
            inputs["sr"],
            inputs["output_fc"],
        )
        messages.addMessage(f"输出已写入: {inputs['output_fc']}")

    def postExecute(self, parameters):  # noqa: N802
        return
