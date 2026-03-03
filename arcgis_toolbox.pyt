"""
ArcGIS Python Toolbox – Enhanced Least Cost Path
=================================================

This ``.pyt`` file defines an ArcGIS geoprocessing tool that wraps the
enhanced LCP algorithm implemented in ``enhanced_lcp.py``.

Usage
-----
1. In ArcGIS Pro, open the *Catalog* pane.
2. Right-click **Toolboxes** → **Add Toolbox** and select this file.
3. Expand the toolbox and double-click **Enhanced Least Cost Path**.

Requirements
------------
* ArcGIS Pro with Python 3 (arcpy must be available).
* ``enhanced_lcp.py`` must be on ``sys.path`` (same directory is fine).
* ``numpy`` (bundled with ArcGIS Pro).
"""

import os
import sys

import arcpy
import numpy as np

# Ensure the toolbox directory is on sys.path so enhanced_lcp can be imported.
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
if _THIS_DIR not in sys.path:
    sys.path.insert(0, _THIS_DIR)

from enhanced_lcp import enhanced_least_cost_path  # noqa: E402


class Toolbox:
    """ArcGIS Python Toolbox container."""

    def __init__(self):
        self.label = "Enhanced Least Cost Path Toolbox"
        self.alias = "EnhancedLCP"
        self.tools = [EnhancedLeastCostPathTool]


class EnhancedLeastCostPathTool:
    """ArcGIS geoprocessing tool for enhanced least-cost-path analysis."""

    def __init__(self):
        self.label = "Enhanced Least Cost Path"
        self.description = (
            "Compute a least cost path with additional curvature and "
            "distance controls that go beyond the standard ESRI LCP tool."
        )
        self.canRunInBackground = True

    # ------------------------------------------------------------------
    # Parameter definitions
    # ------------------------------------------------------------------

    def getParameterInfo(self):  # noqa: N802
        """Define the tool parameters."""
        params = []

        # 0 – Input cost raster
        p_cost = arcpy.Parameter(
            displayName="Cost Raster",
            name="cost_raster",
            datatype="DERasterDataset",
            parameterType="Required",
            direction="Input",
        )
        params.append(p_cost)

        # 1 – Start point
        p_start = arcpy.Parameter(
            displayName="Start Point",
            name="start_point",
            datatype="DEFeatureClass",
            parameterType="Required",
            direction="Input",
        )
        p_start.filter.list = ["Point"]
        params.append(p_start)

        # 2 – End point
        p_end = arcpy.Parameter(
            displayName="End Point",
            name="end_point",
            datatype="DEFeatureClass",
            parameterType="Required",
            direction="Input",
        )
        p_end.filter.list = ["Point"]
        params.append(p_end)

        # 3 – Curvature factor
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

        # 4 – Maximum turning angle
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

        # 5 – Distance factor
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

        # 6 – Output path feature class
        p_out = arcpy.Parameter(
            displayName="Output Path Feature Class",
            name="output_path",
            datatype="DEFeatureClass",
            parameterType="Required",
            direction="Output",
        )
        params.append(p_out)

        return params

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    def isLicensed(self):  # noqa: N802
        return True

    def updateParameters(self, parameters):  # noqa: N802
        return

    def updateMessages(self, parameters):  # noqa: N802
        return

    # ------------------------------------------------------------------
    # Execution
    # ------------------------------------------------------------------

    def execute(self, parameters, messages):  # noqa: N802
        cost_raster_path = parameters[0].valueAsText
        start_fc = parameters[1].valueAsText
        end_fc = parameters[2].valueAsText
        curvature_factor = float(parameters[3].value or 0.0)
        max_turning_angle = float(parameters[4].value or 180.0)
        distance_factor = float(parameters[5].value or 0.0)
        output_fc = parameters[6].valueAsText

        # --- Read cost raster via arcpy ------------------------------------
        raster = arcpy.Raster(cost_raster_path)
        cost_array = arcpy.RasterToNumPyArray(raster, nodata_to_value=np.nan)
        cost_array = cost_array.astype(np.float64)
        cell_x = raster.meanCellWidth
        cell_y = raster.meanCellHeight
        extent = raster.extent
        sr = raster.spatialReference

        # --- Convert point feature classes to (row, col) ------------------
        start_pt = _fc_to_point(start_fc)
        end_pt = _fc_to_point(end_fc)

        start_rc = _xy_to_rowcol(start_pt, extent, cell_x, cell_y,
                                 cost_array.shape)
        end_rc = _xy_to_rowcol(end_pt, extent, cell_x, cell_y,
                               cost_array.shape)

        messages.addMessage(
            f"Start cell: {start_rc}, End cell: {end_rc}, "
            f"Raster shape: {cost_array.shape}"
        )

        # --- Run the algorithm ---------------------------------------------
        result = enhanced_least_cost_path(
            cost_array,
            start_rc,
            end_rc,
            curvature_factor=curvature_factor,
            max_turning_angle=max_turning_angle,
            distance_factor=distance_factor,
            cell_size=(cell_y, cell_x),
        )

        messages.addMessage(
            f"Path found: {len(result['path'])} cells, "
            f"cost={result['total_cost']:.2f}, "
            f"length={result['path_length']:.2f}"
        )

        # --- Write output polyline -----------------------------------------
        _write_polyline(result["path"], extent, cell_x, cell_y, sr, output_fc)

        messages.addMessage(f"Output written to {output_fc}")
        return

    def postExecute(self, parameters):  # noqa: N802
        return


# ---------------------------------------------------------------------------
# Helper functions
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
    """Write the path cells as a single polyline feature class."""
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
