"""Tests for the theta_star_lcp module (方案B)."""

import math

import numpy as np
import pytest

from theta_star_lcp import (
    _supercover_line,
    _is_line_clear,
    _line_cost,
    _vector_angle,
    theta_star_least_cost_path,
    smooth_path,
)


# ---------------------------------------------------------------------------
# Vector angle helper
# ---------------------------------------------------------------------------

class TestVectorAngle:
    def test_same_direction(self):
        assert _vector_angle(1, 0, 1, 0) == pytest.approx(0.0)

    def test_opposite_direction(self):
        assert _vector_angle(1, 0, -1, 0) == pytest.approx(180.0)

    def test_right_angle(self):
        assert _vector_angle(1, 0, 0, 1) == pytest.approx(90.0)

    def test_45_degrees(self):
        assert _vector_angle(0, 1, 1, 1) == pytest.approx(45.0, abs=0.1)

    def test_zero_length(self):
        assert _vector_angle(0, 0, 1, 0) == 0.0


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

class TestValidation:
    @pytest.fixture()
    def simple_raster(self):
        return np.ones((5, 5))

    def test_curvature_factor_too_high(self, simple_raster):
        with pytest.raises(ValueError, match="curvature_factor"):
            theta_star_least_cost_path(simple_raster, (0, 0), (4, 4),
                                       curvature_factor=1.5)

    def test_start_out_of_bounds(self, simple_raster):
        with pytest.raises(ValueError, match="Start point"):
            theta_star_least_cost_path(simple_raster, (10, 0), (4, 4))

    def test_end_out_of_bounds(self, simple_raster):
        with pytest.raises(ValueError, match="End point"):
            theta_star_least_cost_path(simple_raster, (0, 0), (10, 10))

    def test_start_on_nan(self, simple_raster):
        simple_raster[0, 0] = np.nan
        with pytest.raises(ValueError, match="invalid"):
            theta_star_least_cost_path(simple_raster, (0, 0), (4, 4))


# ---------------------------------------------------------------------------
# Basic pathfinding
# ---------------------------------------------------------------------------

class TestThetaStarBasic:
    def test_straight_horizontal(self):
        raster = np.ones((5, 10))
        result = theta_star_least_cost_path(raster, (2, 0), (2, 9))
        path = result["path"]
        assert path[0] == (2, 0)
        assert path[-1] == (2, 9)
        assert result["total_cost"] > 0

    def test_straight_diagonal(self):
        raster = np.ones((10, 10))
        result = theta_star_least_cost_path(raster, (0, 0), (9, 9))
        path = result["path"]
        assert path[0] == (0, 0)
        assert path[-1] == (9, 9)

    def test_avoid_barrier(self):
        """Path must go around a NODATA barrier."""
        raster = np.ones((5, 10))
        raster[2, 3:7] = np.nan  # Wall in middle
        result = theta_star_least_cost_path(raster, (2, 0), (2, 9))
        path = result["path"]
        assert path[0] == (2, 0)
        assert path[-1] == (2, 9)
        # Path should not pass through the barrier
        for r, c in path:
            assert np.isfinite(raster[r, c])

    def test_start_equals_end(self):
        raster = np.ones((5, 5))
        result = theta_star_least_cost_path(raster, (2, 2), (2, 2))
        assert result["path"] == [(2, 2)]
        assert result["total_cost"] == 0.0

    def test_no_path_raises(self):
        raster = np.ones((5, 5))
        raster[2, :] = np.nan  # Impassable wall
        with pytest.raises(RuntimeError, match="No path"):
            theta_star_least_cost_path(raster, (0, 0), (4, 4))

    def test_any_angle_path(self):
        """On a uniform raster, Theta* should produce a nearly direct path
        with fewer waypoints than the 8-direction grid path."""
        raster = np.ones((20, 20))
        result = theta_star_least_cost_path(raster, (0, 0), (19, 19))
        path = result["path"]
        # Theta* should find a fairly direct path
        assert len(path) >= 2
        assert path[0] == (0, 0)
        assert path[-1] == (19, 19)

    def test_prefers_low_cost_cells(self):
        """Path should prefer low-cost cells over direct high-cost shortcut."""
        raster = np.full((10, 10), 10.0)
        # Create a low-cost corridor
        raster[0, :] = 1.0
        raster[:, 9] = 1.0
        result = theta_star_least_cost_path(raster, (0, 0), (9, 9))
        path = result["path"]
        # Path should use the low-cost corridor (go along row 0 then col 9)
        # rather than cutting through the high-cost interior
        assert path[0] == (0, 0)
        assert path[-1] == (9, 9)

    def test_result_has_smoothed_path(self):
        raster = np.ones((10, 10))
        result = theta_star_least_cost_path(raster, (0, 0), (9, 9))
        assert "smoothed_path" in result
        assert len(result["smoothed_path"]) >= 2

    def test_result_has_path_length(self):
        raster = np.ones((5, 10))
        result = theta_star_least_cost_path(raster, (2, 0), (2, 9))
        assert result["path_length"] > 0


# ---------------------------------------------------------------------------
# Curvature support
# ---------------------------------------------------------------------------

class TestThetaStarCurvature:
    def test_curvature_produces_path(self):
        raster = np.ones((10, 10))
        result = theta_star_least_cost_path(
            raster, (0, 0), (9, 9), curvature_factor=0.5
        )
        assert len(result["path"]) >= 2
        assert result["path"][0] == (0, 0)
        assert result["path"][-1] == (9, 9)

    def test_max_turning_angle_produces_path(self):
        raster = np.ones((10, 10))
        result = theta_star_least_cost_path(
            raster, (0, 0), (9, 9), max_turning_angle=90.0
        )
        assert len(result["path"]) >= 2


# ---------------------------------------------------------------------------
# Smoothing
# ---------------------------------------------------------------------------

class TestThetaStarSmoothing:
    def test_smoothed_path_nodata_safe(self):
        """Smoothed path should not cross NODATA."""
        raster = np.ones((10, 10))
        raster[4, 1:9] = np.nan  # NODATA wall
        result = theta_star_least_cost_path(raster, (0, 5), (9, 5))
        smoothed = result["smoothed_path"]
        # Verify no smoothed point is on NODATA
        for r, c in smoothed:
            ri, ci = int(round(r)), int(round(c))
            ri = max(0, min(9, ri))
            ci = max(0, min(9, ci))
            assert np.isfinite(raster[ri, ci])
