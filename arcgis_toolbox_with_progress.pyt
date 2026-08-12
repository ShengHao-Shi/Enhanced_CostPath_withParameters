"""
ArcGIS Python Toolbox – Cost-Aware LCP with Progress Reporting
===============================================================

This ``.pyt`` file provides **four** geoprocessing tools that report
step-by-step progress in the ArcGIS Geoprocessing pane:

1. **Cost-Aware LCP (Pure Python)** — uses
   ``pure_python.cost_aware_straighten_lcp`` with progress messages
   throughout the Dijkstra search, straightening, and smoothing phases.

2. **Cost-Aware LCP (Numba Accelerated)** — uses
   ``numba_accelerated.cost_aware_straighten_lcp`` for 20–50× faster
   execution on large rasters, with progress messages before/after
   each computational phase.

3. **Survey-Aware LCP (Pure Python)** — extends the standard LCP for
   hydrographic survey vessels.  Accepts a CATZOC coverage raster in
   addition to the risk raster, applies a hard safety threshold, and
   attracts the path towards under-surveyed areas.

4. **Survey-Aware LCP (Numba Accelerated)** — same as (3) but uses
   the Numba-accelerated Dijkstra search for large-raster performance.

All tools accept the same base parameters and produce the same path
output.  The progress is shown via the ArcGIS step progressor bar
and as text lines in the *Geoprocessing → Messages* section.

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
import pure_python.survey_aware_lcp as _survey_pure_mod  # noqa: E402


# ---------------------------------------------------------------------------
# Progress bar helper
# ---------------------------------------------------------------------------

# Map algorithm progress messages to percentage positions on the progress bar.
_PROGRESS_MAP = [
    ("Parameters validated", 5),
    ("Dijkstra search", 10),          # start of search
    ("Dijkstra search: complete", 75),
    ("Path reconstruction", 78),
    ("Path straightening: starting", 82),
    ("Path straightening: complete", 88),
    ("Path smoothing: starting", 92),
    ("Path smoothing: complete", 96),
]


def _make_progress_callback(messages):
    """Return a callable that updates both the ArcGIS progressor bar and messages.

    Uses ``arcpy.SetProgressor("step", ...)`` for an actual progress bar,
    and ``messages.addMessage()`` for text logging.
    """
    arcpy.SetProgressor("step", "Initializing...", 0, 100, 1)

    def _callback(msg):
        messages.addMessage(msg)
        arcpy.SetProgressorLabel(msg)

        # For Dijkstra per-iteration messages like "expanded X nodes (~Y%)",
        # map the embedded percentage to the overall 10-75% range.
        if "expanded" in msg and "(~" in msg:
            try:
                pct_str = msg.split("(~")[1].split("%")[0]
                dijkstra_pct = int(pct_str)
                arcpy.SetProgressorPosition(10 + int(dijkstra_pct * 0.65))
            except (IndexError, ValueError):
                pass
            return

        # Determine percentage from the message content
        best_pct = None
        for pattern, pct in _PROGRESS_MAP:
            if pattern in msg:
                best_pct = pct

        if best_pct is not None:
            arcpy.SetProgressorPosition(best_pct)

    return _callback


class Toolbox:
    """ArcGIS Python Toolbox container."""

    def __init__(self):
        self.label = "Cost-Aware LCP with Progress"
        self.alias = "CostAwareLCPProgress"
        self.tools = [
            CostAwareLCPTool,
            CostAwareNumbaLCPTool,
            SurveyAwareLCPTool,
            SurveyAwareNumbaLCPTool,
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


def _raster_to_array_clipped(raster, start_xy, end_xy, is_integer, sentinel):
    """Read only the region of a raster that surrounds start and end points.

    Avoids the ArcPy 'pixel block exceeds maximum size' error for large rasters
    by requesting only the sub-region needed for routing.  A buffer of at least
    25 % of the straight-line distance between the two points (minimum 50 cells)
    is added so that the algorithm has room to route around obstacles.

    Parameters
    ----------
    raster : arcpy.Raster
        Source raster (already opened).
    start_xy, end_xy : (float, float)
        Map coordinates of the start and end points.
    is_integer : bool
        Whether the raster stores integer values.
    sentinel : int or float
        Value used as a stand-in for NoData when *is_integer* is True.
        For floating-point rasters pass ``np.nan`` (it is not used).

    Returns
    -------
    arr : numpy.ndarray  (float32)
        Clipped raster values; NoData cells are represented as ``np.nan``.
    clipped_extent : arcpy.Extent
        The geographic extent of the returned array (grid-snapped).
    """
    cell_x = raster.meanCellWidth
    cell_y = raster.meanCellHeight
    full_ext = raster.extent

    # Bounding box of the two points plus a generous buffer.
    dx = abs(end_xy[0] - start_xy[0])
    dy = abs(end_xy[1] - start_xy[1])
    diag = max((dx ** 2 + dy ** 2) ** 0.5, cell_x * 50)
    buf = max(diag * 0.25, cell_x * 50)

    raw_xmin = min(start_xy[0], end_xy[0]) - buf
    raw_xmax = max(start_xy[0], end_xy[0]) + buf
    raw_ymin = min(start_xy[1], end_xy[1]) - buf
    raw_ymax = max(start_xy[1], end_xy[1]) + buf

    # Clamp to the full raster extent.
    clip_xmin = max(raw_xmin, full_ext.XMin)
    clip_xmax = min(raw_xmax, full_ext.XMax)
    clip_ymin = max(raw_ymin, full_ext.YMin)
    clip_ymax = min(raw_ymax, full_ext.YMax)

    # Snap to the raster grid so llCorner aligns with a cell boundary.
    col_offset = int((clip_xmin - full_ext.XMin) / cell_x)
    row_offset = int((full_ext.YMax - clip_ymax) / cell_y)
    ncols = max(1, int((clip_xmax - clip_xmin) / cell_x))
    nrows = max(1, int((clip_ymax - clip_ymin) / cell_y))

    snap_xmin = full_ext.XMin + col_offset * cell_x
    snap_ymax = full_ext.YMax - row_offset * cell_y
    snap_xmax = snap_xmin + ncols * cell_x
    snap_ymin = snap_ymax - nrows * cell_y

    ll_corner = arcpy.Point(snap_xmin, snap_ymin)

    if is_integer:
        arr = arcpy.RasterToNumPyArray(
            raster, ll_corner, ncols, nrows, nodata_to_value=sentinel
        )
        arr = arr.astype(np.float32)
        arr[arr == sentinel] = np.nan
    else:
        arr = arcpy.RasterToNumPyArray(
            raster, ll_corner, ncols, nrows, nodata_to_value=np.nan
        )
        arr = arr.astype(np.float32)

    clipped_extent = arcpy.Extent(snap_xmin, snap_ymin, snap_xmax, snap_ymax)
    return arr, clipped_extent


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
    cell_x = raster.meanCellWidth
    cell_y = raster.meanCellHeight
    sr = raster.spatialReference

    # Read the start/end points before loading the raster array so that only
    # the sub-region around the route is read (avoids the ArcPy pixel-block
    # size limit on very large rasters).
    start_pt = _fc_to_point(start_fc)
    end_pt = _fc_to_point(end_fc)

    nodata_val = raster.noDataValue
    if raster.isInteger:
        # Integer rasters do not support NaN as nodata_to_value;
        # use the raster's own nodata value as sentinel (or a fallback),
        # convert to float, then replace with NaN.
        sentinel = int(nodata_val) if nodata_val is not None else -9999
        cost_array, extent = _raster_to_array_clipped(
            raster, start_pt, end_pt, is_integer=True, sentinel=sentinel
        )
    else:
        cost_array, extent = _raster_to_array_clipped(
            raster, start_pt, end_pt, is_integer=False, sentinel=np.nan
        )

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
        progress_cb = _make_progress_callback(messages)

        messages.addMessage(
            f"[Cost-Aware / Pure Python] Starting computation...\n"
            f"  Start: {inputs['start_rc']}, End: {inputs['end_rc']}\n"
            f"  Raster size: {inputs['cost_array'].shape}"
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
            progress_callback=progress_cb,
        )

        elapsed = time.time() - t0

        messages.addMessage(
            f"[Cost-Aware / Pure Python] Computation complete ({elapsed:.1f}s)\n"
            f"  Path nodes: {len(result['path'])}\n"
            f"  Straightened nodes: {len(result['straightened_path'])}\n"
            f"  Smoothed nodes: {len(result['smoothed_path'])}\n"
            f"  Total cost: {result['total_cost']:.2f}\n"
            f"  Path length: {result['path_length']:.2f}"
        )

        _write_polyline(
            result["smoothed_path"],
            inputs["extent"],
            inputs["cell_x"],
            inputs["cell_y"],
            inputs["sr"],
            inputs["output_fc"],
        )
        arcpy.SetProgressorPosition(100)
        arcpy.ResetProgressor()
        messages.addMessage(f"Output written to: {inputs['output_fc']}")

    def postExecute(self, parameters):  # noqa: N802
        return


# ---------------------------------------------------------------------------
# Shared helpers for the survey-aware tools
# ---------------------------------------------------------------------------


def _make_survey_aware_params():
    """Create the parameter list for both Survey-Aware LCP tools."""
    # Start with all standard cost-aware parameters.
    params = _make_cost_aware_params()

    # Insert the survey raster as the second parameter (after the cost raster).
    p_survey = arcpy.Parameter(
        displayName="Survey Coverage Raster (CATZOC 0–3)",
        name="survey_raster",
        datatype="GPRasterLayer",
        parameterType="Required",
        direction="Input",
    )
    # Insert at index 1 (after cost_raster at index 0).
    params.insert(1, p_survey)

    # Safety threshold – optional, sits after survey raster.
    p_threshold = arcpy.Parameter(
        displayName="Safety Threshold (risk cells above this are penalised)",
        name="safety_threshold",
        datatype="GPDouble",
        parameterType="Optional",
        direction="Input",
    )
    p_threshold.value = 100.0
    p_threshold.filter.type = "Range"
    p_threshold.filter.list = [0.0, 1000.0]
    params.insert(2, p_threshold)

    # Survey weight.
    p_weight = arcpy.Parameter(
        displayName="Survey Weight (0.0 = risk only, 1.0 = survey only)",
        name="survey_weight",
        datatype="GPDouble",
        parameterType="Optional",
        direction="Input",
    )
    p_weight.value = 0.3
    p_weight.filter.type = "Range"
    p_weight.filter.list = [0.0, 1.0]
    params.insert(3, p_weight)

    # Survey NODATA handling.
    p_nodata = arcpy.Parameter(
        displayName="Survey NODATA Treatment",
        name="survey_nodata_as",
        datatype="GPString",
        parameterType="Optional",
        direction="Input",
    )
    p_nodata.filter.type = "ValueList"
    p_nodata.filter.list = ["unsurveyed", "surveyed"]
    p_nodata.value = "unsurveyed"
    params.insert(4, p_nodata)

    # Penalty multiplier – controls how strongly above-threshold cells are penalised.
    p_penalty = arcpy.Parameter(
        displayName="Safety Penalty Multiplier (>= 0.0; higher = stronger avoidance)",
        name="penalty_multiplier",
        datatype="GPDouble",
        parameterType="Optional",
        direction="Input",
    )
    p_penalty.value = 10.0
    p_penalty.filter.type = "Range"
    p_penalty.filter.list = [0.0, 1000.0]
    params.insert(5, p_penalty)

    return params


def _read_survey_inputs(parameters):
    """Parse survey-aware tool parameters.

    Parameter order (after insertion):
      0  cost_raster
      1  survey_raster        <- new
      2  safety_threshold     <- new
      3  survey_weight        <- new
      4  survey_nodata_as     <- new
      5  penalty_multiplier   <- new
      6  start_point
      7  end_point
      8  curvature_factor
      9  max_turning_angle
      10 distance_factor
      11 straighten_factor
      12 cost_tolerance
      13 output_path
    """
    cost_raster_path = parameters[0].valueAsText
    survey_raster_path = parameters[1].valueAsText
    safety_threshold = float(parameters[2].value or 100.0)
    survey_weight = float(parameters[3].value or 0.3)
    survey_nodata_as = parameters[4].valueAsText or "unsurveyed"
    penalty_multiplier_val = parameters[5].value
    penalty_multiplier = float(
        penalty_multiplier_val if penalty_multiplier_val is not None else 10.0
    )
    start_fc = parameters[6].valueAsText
    end_fc = parameters[7].valueAsText
    curvature_factor = float(parameters[8].value or 0.0)
    max_turning_angle_val = parameters[9].value
    max_turning_angle = float(
        max_turning_angle_val if max_turning_angle_val is not None else 180.0
    )
    distance_factor = float(parameters[10].value or 0.0)
    straighten_factor_val = parameters[11].value
    straighten_factor = float(
        straighten_factor_val if straighten_factor_val is not None else 0.3
    )
    cost_tolerance_val = parameters[12].value
    cost_tolerance = float(
        cost_tolerance_val if cost_tolerance_val is not None else 1.05
    )
    output_fc = parameters[13].valueAsText

    # Read the start/end points before loading raster arrays so that only the
    # sub-region around the route is read (avoids the ArcPy pixel-block size
    # limit on very large rasters).
    start_pt = _fc_to_point(start_fc)
    end_pt = _fc_to_point(end_fc)

    # Load risk raster.
    risk_raster = arcpy.Raster(cost_raster_path)
    nodata_val = risk_raster.noDataValue
    if risk_raster.isInteger:
        sentinel = int(nodata_val) if nodata_val is not None else -9999
        risk_array, extent = _raster_to_array_clipped(
            risk_raster, start_pt, end_pt, is_integer=True, sentinel=sentinel
        )
    else:
        risk_array, extent = _raster_to_array_clipped(
            risk_raster, start_pt, end_pt, is_integer=False, sentinel=np.nan
        )

    cell_x = risk_raster.meanCellWidth
    cell_y = risk_raster.meanCellHeight
    sr = risk_raster.spatialReference

    # Load survey (CATZOC) raster clipped to the same bounding box.
    survey_raster_obj = arcpy.Raster(survey_raster_path)
    survey_nodata = survey_raster_obj.noDataValue
    if survey_raster_obj.isInteger:
        sentinel_s = int(survey_nodata) if survey_nodata is not None else -9999
        survey_array, _ = _raster_to_array_clipped(
            survey_raster_obj, start_pt, end_pt, is_integer=True, sentinel=sentinel_s
        )
    else:
        survey_array, _ = _raster_to_array_clipped(
            survey_raster_obj, start_pt, end_pt, is_integer=False, sentinel=np.nan
        )

    start_rc = _xy_to_rowcol(start_pt, extent, cell_x, cell_y, risk_array.shape)
    end_rc = _xy_to_rowcol(end_pt, extent, cell_x, cell_y, risk_array.shape)

    return {
        "risk_array": risk_array,
        "survey_array": survey_array,
        "safety_threshold": safety_threshold,
        "survey_weight": survey_weight,
        "survey_nodata_as": survey_nodata_as,
        "penalty_multiplier": penalty_multiplier,
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


def _log_survey_result(messages, tag, result, elapsed):
    """Log survey-aware result statistics to the messages pane."""
    messages.addMessage(
        f"[{tag}] Computation complete ({elapsed:.1f}s)\n"
        f"  Path nodes: {len(result['path'])}\n"
        f"  Straightened nodes: {len(result['straightened_path'])}\n"
        f"  Smoothed nodes: {len(result['smoothed_path'])}\n"
        f"  Total cost: {result['total_cost']:.2f}\n"
        f"  Path length: {result['path_length']:.2f}\n"
        f"  Survey score: {result['survey_score']:.2%} "
        f"(gap/minimal coverage cells on path)\n"
        f"  Cells above safety threshold (penalised): {result['above_threshold_count']}"
    )


# =========================================================================
# Tool 3: Survey-Aware LCP (Pure Python)
# =========================================================================


class SurveyAwareLCPTool:
    """ArcGIS tool for survey-vessel LCP routing (pure Python)."""

    def __init__(self):
        self.label = "Survey-Aware LCP (Pure Python)"
        self.description = (
            "Plan routes for hydrographic survey vessels. "
            "Applies a soft safety penalty on risk cells above a threshold "
            "(controlled by the penalty multiplier) and a survey-value "
            "objective based on a CATZOC coverage raster, "
            "attracting the path through under-surveyed areas. "
            "Uses pure Python for maximum compatibility."
        )
        self.canRunInBackground = True

    def getParameterInfo(self):  # noqa: N802
        return _make_survey_aware_params()

    def isLicensed(self):  # noqa: N802
        return True

    def updateParameters(self, parameters):  # noqa: N802
        return

    def updateMessages(self, parameters):  # noqa: N802
        return

    def execute(self, parameters, messages):  # noqa: N802
        importlib.reload(_survey_pure_mod)

        inputs = _read_survey_inputs(parameters)
        progress_cb = _make_progress_callback(messages)

        messages.addMessage(
            f"[Survey-Aware / Pure Python] Starting computation...\n"
            f"  Start: {inputs['start_rc']}, End: {inputs['end_rc']}\n"
            f"  Raster size: {inputs['risk_array'].shape}\n"
            f"  Safety threshold: {inputs['safety_threshold']}\n"
            f"  Survey weight: {inputs['survey_weight']}"
        )

        t0 = time.time()

        result = _survey_pure_mod.survey_aware_least_cost_path(
            inputs["risk_array"],
            inputs["survey_array"],
            inputs["start_rc"],
            inputs["end_rc"],
            safety_threshold=inputs["safety_threshold"],
            survey_weight=inputs["survey_weight"],
            survey_nodata_as=inputs["survey_nodata_as"],
            penalty_multiplier=inputs["penalty_multiplier"],
            curvature_factor=inputs["curvature_factor"],
            max_turning_angle=inputs["max_turning_angle"],
            distance_factor=inputs["distance_factor"],
            straighten_factor=inputs["straighten_factor"],
            cost_tolerance=inputs["cost_tolerance"],
            cell_size=inputs["cell_size"],
            progress_callback=progress_cb,
        )

        elapsed = time.time() - t0
        _log_survey_result(messages, "Survey-Aware / Pure Python", result, elapsed)

        _write_polyline(
            result["smoothed_path"],
            inputs["extent"],
            inputs["cell_x"],
            inputs["cell_y"],
            inputs["sr"],
            inputs["output_fc"],
        )
        arcpy.SetProgressorPosition(100)
        arcpy.ResetProgressor()
        messages.addMessage(f"Output written to: {inputs['output_fc']}")

    def postExecute(self, parameters):  # noqa: N802
        return


# =========================================================================
# Tool 4: Survey-Aware LCP (Numba Accelerated)
# =========================================================================


class SurveyAwareNumbaLCPTool:
    """ArcGIS tool for survey-vessel LCP routing (Numba accelerated)."""

    def __init__(self):
        self.label = "Survey-Aware LCP (Numba Accelerated)"
        self.description = (
            "Plan routes for hydrographic survey vessels using the "
            "Numba JIT-compiled Dijkstra search (20–50x faster on large "
            "rasters). Applies a soft safety penalty on risk cells above a "
            "threshold (controlled by the penalty multiplier) and a "
            "survey-value objective based on a CATZOC coverage raster. "
            "Note: the first call has ~6 s JIT compilation overhead."
        )
        self.canRunInBackground = True

    def getParameterInfo(self):  # noqa: N802
        return _make_survey_aware_params()

    def isLicensed(self):  # noqa: N802
        return True

    def updateParameters(self, parameters):  # noqa: N802
        return

    def updateMessages(self, parameters):  # noqa: N802
        return

    def execute(self, parameters, messages):  # noqa: N802
        try:
            import numba_accelerated.survey_aware_lcp as _survey_numba_mod
            importlib.reload(_survey_numba_mod)
        except ImportError:
            messages.addErrorMessage(
                "Cannot import numba_accelerated module. "
                "Please ensure numba is installed: pip install numba"
            )
            raise

        inputs = _read_survey_inputs(parameters)
        progress_cb = _make_progress_callback(messages)

        messages.addMessage(
            f"[Survey-Aware / Numba] Starting computation...\n"
            f"  Start: {inputs['start_rc']}, End: {inputs['end_rc']}\n"
            f"  Raster size: {inputs['risk_array'].shape}\n"
            f"  Safety threshold: {inputs['safety_threshold']}\n"
            f"  Survey weight: {inputs['survey_weight']}"
        )

        t0 = time.time()

        result = _survey_numba_mod.survey_aware_least_cost_path(
            inputs["risk_array"],
            inputs["survey_array"],
            inputs["start_rc"],
            inputs["end_rc"],
            safety_threshold=inputs["safety_threshold"],
            survey_weight=inputs["survey_weight"],
            survey_nodata_as=inputs["survey_nodata_as"],
            penalty_multiplier=inputs["penalty_multiplier"],
            curvature_factor=inputs["curvature_factor"],
            max_turning_angle=inputs["max_turning_angle"],
            distance_factor=inputs["distance_factor"],
            straighten_factor=inputs["straighten_factor"],
            cost_tolerance=inputs["cost_tolerance"],
            cell_size=inputs["cell_size"],
            progress_callback=progress_cb,
        )

        elapsed = time.time() - t0
        _log_survey_result(messages, "Survey-Aware / Numba", result, elapsed)

        _write_polyline(
            result["smoothed_path"],
            inputs["extent"],
            inputs["cell_x"],
            inputs["cell_y"],
            inputs["sr"],
            inputs["output_fc"],
        )
        arcpy.SetProgressorPosition(100)
        arcpy.ResetProgressor()
        messages.addMessage(f"Output written to: {inputs['output_fc']}")

    def postExecute(self, parameters):  # noqa: N802
        return

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
                "Cannot import numba_accelerated module. "
                "Please ensure numba is installed: pip install numba"
            )
            raise

        inputs = _read_inputs(parameters)
        progress_cb = _make_progress_callback(messages)

        messages.addMessage(
            f"[Cost-Aware / Numba] Starting computation...\n"
            f"  Start: {inputs['start_rc']}, End: {inputs['end_rc']}\n"
            f"  Raster size: {inputs['cost_array'].shape}"
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
            progress_callback=progress_cb,
        )

        elapsed = time.time() - t0

        messages.addMessage(
            f"[Cost-Aware / Numba] Computation complete ({elapsed:.1f}s)\n"
            f"  Path nodes: {len(result['path'])}\n"
            f"  Straightened nodes: {len(result['straightened_path'])}\n"
            f"  Smoothed nodes: {len(result['smoothed_path'])}\n"
            f"  Total cost: {result['total_cost']:.2f}\n"
            f"  Path length: {result['path_length']:.2f}"
        )

        _write_polyline(
            result["smoothed_path"],
            inputs["extent"],
            inputs["cell_x"],
            inputs["cell_y"],
            inputs["sr"],
            inputs["output_fc"],
        )
        arcpy.SetProgressorPosition(100)
        arcpy.ResetProgressor()
        messages.addMessage(f"Output written to: {inputs['output_fc']}")

    def postExecute(self, parameters):  # noqa: N802
        return
