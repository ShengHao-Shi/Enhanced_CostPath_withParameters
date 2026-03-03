"""Tests for the enhanced_lcp module."""

import math

import numpy as np
import pytest

from enhanced_lcp import (
    DIRECTIONS,
    NUM_DIRS,
    enhanced_least_cost_path,
    smooth_path,
    turning_angle,
)

# ---------------------------------------------------------------------------
# turning_angle
# ---------------------------------------------------------------------------


class TestTurningAngle:
    """Unit tests for the turning_angle helper."""

    def test_same_direction(self):
        for d in range(NUM_DIRS):
            assert turning_angle(d, d) == 0.0

    def test_opposite_direction(self):
        assert turning_angle(0, 4) == 180.0  # N -> S
        assert turning_angle(2, 6) == 180.0  # E -> W

    def test_right_angle(self):
        assert turning_angle(0, 2) == 90.0   # N -> E
        assert turning_angle(2, 4) == 90.0   # E -> S

    def test_45_degree(self):
        assert turning_angle(0, 1) == 45.0   # N -> NE

    def test_135_degree(self):
        assert turning_angle(0, 3) == 135.0  # N -> SE

    def test_symmetry(self):
        for a in range(NUM_DIRS):
            for b in range(NUM_DIRS):
                assert turning_angle(a, b) == turning_angle(b, a)


# ---------------------------------------------------------------------------
# Parameter validation
# ---------------------------------------------------------------------------


class TestValidation:
    """Ensure invalid parameters raise ValueError."""

    @pytest.fixture()
    def simple_raster(self):
        return np.ones((5, 5))

    def test_curvature_factor_too_high(self, simple_raster):
        with pytest.raises(ValueError, match="curvature_factor"):
            enhanced_least_cost_path(simple_raster, (0, 0), (4, 4),
                                     curvature_factor=1.5)

    def test_curvature_factor_negative(self, simple_raster):
        with pytest.raises(ValueError, match="curvature_factor"):
            enhanced_least_cost_path(simple_raster, (0, 0), (4, 4),
                                     curvature_factor=-0.1)

    def test_min_turning_angle_too_high(self, simple_raster):
        with pytest.raises(ValueError, match="min_turning_angle"):
            enhanced_least_cost_path(simple_raster, (0, 0), (4, 4),
                                     min_turning_angle=200.0)

    def test_distance_factor_too_high(self, simple_raster):
        with pytest.raises(ValueError, match="distance_factor"):
            enhanced_least_cost_path(simple_raster, (0, 0), (4, 4),
                                     distance_factor=2.0)

    def test_start_out_of_bounds(self, simple_raster):
        with pytest.raises(ValueError, match="Start point"):
            enhanced_least_cost_path(simple_raster, (10, 0), (4, 4))

    def test_end_out_of_bounds(self, simple_raster):
        with pytest.raises(ValueError, match="End point"):
            enhanced_least_cost_path(simple_raster, (0, 0), (10, 10))

    def test_start_on_nan(self):
        raster = np.ones((5, 5))
        raster[0, 0] = np.nan
        with pytest.raises(ValueError, match="invalid"):
            enhanced_least_cost_path(raster, (0, 0), (4, 4))

    def test_non_2d_raster(self):
        with pytest.raises(ValueError, match="2-D"):
            enhanced_least_cost_path(np.ones((3, 3, 3)), (0, 0), (2, 2))


# ---------------------------------------------------------------------------
# Basic pathfinding (standard Dijkstra branch)
# ---------------------------------------------------------------------------


class TestStandardDijkstra:
    """Tests that exercise the standard (no-curvature) code path."""

    def test_straight_horizontal(self):
        """Path across a uniform-cost row should be a straight line."""
        raster = np.ones((3, 10))
        result = enhanced_least_cost_path(raster, (1, 0), (1, 9))
        path = result["path"]
        assert path[0] == (1, 0)
        assert path[-1] == (1, 9)
        # All cells should be on row 1
        assert all(r == 1 for r, _ in path)

    def test_diagonal_shortcut(self):
        """On a uniform raster, the diagonal path should be preferred."""
        raster = np.ones((5, 5))
        result = enhanced_least_cost_path(raster, (0, 0), (4, 4))
        path = result["path"]
        assert path[0] == (0, 0)
        assert path[-1] == (4, 4)
        # Diagonal path length = 4 * sqrt(2)
        assert abs(result["path_length"] - 4 * math.sqrt(2)) < 1e-6

    def test_avoid_barrier(self):
        """Path should navigate around NaN barriers."""
        raster = np.ones((5, 5))
        raster[2, 1] = np.nan
        raster[2, 2] = np.nan
        raster[2, 3] = np.nan
        result = enhanced_least_cost_path(raster, (0, 2), (4, 2))
        path = result["path"]
        assert path[0] == (0, 2)
        assert path[-1] == (4, 2)
        # Path should not pass through the barrier row at col 1-3
        for r, c in path:
            assert not (r == 2 and 1 <= c <= 3)

    def test_start_equals_end(self):
        """When start == end, the path has one cell and zero cost."""
        raster = np.ones((3, 3))
        result = enhanced_least_cost_path(raster, (1, 1), (1, 1))
        assert result["path"] == [(1, 1)]
        assert result["total_cost"] == 0.0
        assert result["path_length"] == 0.0

    def test_no_path_raises(self):
        """Isolated start should raise RuntimeError."""
        raster = np.ones((5, 5))
        # Wall off the start cell completely
        raster[0, 1] = np.nan
        raster[1, 0] = np.nan
        raster[1, 1] = np.nan
        with pytest.raises(RuntimeError, match="No path"):
            enhanced_least_cost_path(raster, (0, 0), (4, 4))


# ---------------------------------------------------------------------------
# Distance factor
# ---------------------------------------------------------------------------


class TestDistanceFactor:
    """Verify that ``distance_factor`` encourages shorter paths."""

    def test_distance_factor_shortens_path(self):
        """With a high distance factor, the algorithm should prefer
        a shorter path even if it crosses slightly higher cost cells."""
        # Create a raster where a detour through low-cost cells is long
        raster = np.full((5, 9), 5.0)
        # Direct corridor (row 2) has moderate cost
        raster[2, :] = 3.0
        # Low-cost detour around the top
        raster[0, :] = 1.0

        start, end = (2, 0), (2, 8)

        # Without distance factor -> may detour through cheap top row
        res_no_dist = enhanced_least_cost_path(raster, start, end,
                                               distance_factor=0.0)
        # With high distance factor -> should prefer shorter path
        res_hi_dist = enhanced_least_cost_path(raster, start, end,
                                               distance_factor=1.0)
        assert res_hi_dist["path_length"] <= res_no_dist["path_length"] + 1e-6


# ---------------------------------------------------------------------------
# Curvature factor
# ---------------------------------------------------------------------------


class TestCurvatureFactor:
    """Verify that curvature controls produce smoother paths."""

    def test_curvature_reduces_sharp_turns(self):
        """Higher curvature factor should result in fewer/smaller turns."""
        raster = np.ones((10, 10))
        start, end = (0, 0), (9, 9)

        res_no_curv = enhanced_least_cost_path(
            raster, start, end, curvature_factor=0.0,
        )
        res_hi_curv = enhanced_least_cost_path(
            raster, start, end, curvature_factor=0.8,
        )
        max_angle_no = max(res_no_curv["turning_angles"], default=0)
        max_angle_hi = max(res_hi_curv["turning_angles"], default=0)
        assert max_angle_hi <= max_angle_no

    def test_min_turning_angle_constraint(self):
        """With min_turning_angle=135, no deflection should exceed 45 degrees."""
        raster = np.ones((10, 10))
        # Add some cost variation to encourage turns
        raster[3:7, 3:7] = 5.0

        start, end = (0, 0), (9, 9)
        result = enhanced_least_cost_path(
            raster, start, end,
            curvature_factor=0.5, min_turning_angle=135.0,
        )
        for angle in result["turning_angles"]:
            assert angle <= 45.0 + 1e-9

    def test_min_turning_angle_90(self):
        """With min_turning_angle=90, no deflection should exceed 90 degrees."""
        raster = np.ones((8, 8))
        raster[2:6, 2:6] = 10.0
        start, end = (0, 0), (7, 7)
        result = enhanced_least_cost_path(
            raster, start, end,
            curvature_factor=0.3, min_turning_angle=90.0,
        )
        for angle in result["turning_angles"]:
            assert angle <= 90.0 + 1e-9


# ---------------------------------------------------------------------------
# Cell size
# ---------------------------------------------------------------------------


class TestCellSize:
    """Verify cell_size affects path length calculations."""

    def test_cell_size_scales_length(self):
        raster = np.ones((3, 5))
        start, end = (1, 0), (1, 4)
        res_unit = enhanced_least_cost_path(raster, start, end,
                                            cell_size=(1.0, 1.0))
        res_scaled = enhanced_least_cost_path(raster, start, end,
                                              cell_size=(1.0, 2.0))
        # With cell_size x doubled, path length should double
        assert abs(res_scaled["path_length"] - 2 * res_unit["path_length"]) < 1e-6


# ---------------------------------------------------------------------------
# Combined parameters
# ---------------------------------------------------------------------------


class TestCombined:
    """Integration-style tests using multiple parameters together."""

    def test_all_parameters(self):
        """Smoke test: all three extra parameters active at once."""
        raster = np.random.default_rng(42).uniform(1, 10, size=(15, 15))
        result = enhanced_least_cost_path(
            raster, (0, 0), (14, 14),
            curvature_factor=0.5,
            min_turning_angle=90.0,
            distance_factor=0.3,
        )
        assert len(result["path"]) >= 2
        assert result["path"][0] == (0, 0)
        assert result["path"][-1] == (14, 14)
        assert result["total_cost"] > 0
        assert result["path_length"] > 0
        for angle in result["turning_angles"]:
            assert angle <= 90.0 + 1e-9


# ---------------------------------------------------------------------------
# Straightness (anti-zigzag)
# ---------------------------------------------------------------------------


class TestStraightness:
    """Verify that the anti-zigzag straightness penalty works."""

    def test_uniform_raster_fewer_turns(self):
        """On a uniform raster with curvature enabled, the path should
        have very few direction changes thanks to anti-zigzag."""
        raster = np.ones((15, 15))
        start, end = (0, 0), (14, 14)
        result = enhanced_least_cost_path(
            raster, start, end, curvature_factor=0.3,
        )
        # On a uniform raster going diagonally, ideal path is a straight
        # diagonal with zero direction changes.
        assert len(result["turning_angles"]) == 0 or all(
            a == 0.0 for a in result["turning_angles"]
        )

    def test_low_variation_prefers_straight(self):
        """When cost variation is small, the path should prefer going
        straight rather than zigzagging to slightly cheaper cells."""
        rng = np.random.default_rng(123)
        raster = 5.0 + rng.uniform(-0.1, 0.1, (12, 12))
        start, end = (0, 0), (11, 11)
        result = enhanced_least_cost_path(
            raster, start, end, curvature_factor=0.5,
        )
        # Count actual direction changes (non-zero turning angles)
        n_turns = sum(1 for a in result["turning_angles"] if a > 0)
        # Should be very few turns on a nearly uniform raster
        assert n_turns <= 3


# ---------------------------------------------------------------------------
# Path smoothing
# ---------------------------------------------------------------------------


class TestSmoothing:
    """Verify smooth_path (Chaikin's corner-cutting) post-processing."""

    def test_endpoints_preserved(self):
        """Start and end points must be preserved exactly."""
        path = [(0, 0), (1, 1), (2, 0), (3, 1)]
        smoothed = smooth_path(path)
        assert smoothed[0] == (0.0, 0.0)
        assert smoothed[-1] == (3.0, 1.0)

    def test_more_points_than_original(self):
        """Smoothing should produce more points than the input."""
        path = [(0, 0), (3, 0), (3, 3), (0, 3)]
        smoothed = smooth_path(path)
        assert len(smoothed) > len(path)

    def test_two_point_path_unchanged(self):
        """A straight two-point path should remain unchanged."""
        path = [(0, 0), (5, 5)]
        smoothed = smooth_path(path)
        assert len(smoothed) == 2
        assert smoothed[0] == (0.0, 0.0)
        assert smoothed[-1] == (5.0, 5.0)

    def test_single_point_path(self):
        """A single-point path should be returned as-is."""
        path = [(3, 4)]
        smoothed = smooth_path(path)
        assert smoothed == [(3.0, 4.0)]

    def test_smoothed_path_in_result(self):
        """The result dict must contain a 'smoothed_path' key."""
        raster = np.ones((5, 5))
        result = enhanced_least_cost_path(raster, (0, 0), (4, 4))
        assert "smoothed_path" in result
        assert len(result["smoothed_path"]) >= len(result["path"])
        # Check start/end preserved
        sp = result["smoothed_path"]
        assert sp[0] == (0.0, 0.0)
        assert sp[-1] == (4.0, 4.0)
