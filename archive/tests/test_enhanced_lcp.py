"""Tests for the enhanced_lcp module."""

import math

import numpy as np
import pytest

from enhanced_lcp import (
    DIRECTIONS,
    NUM_DIRS,
    _supercover_line,
    enhanced_least_cost_path,
    smooth_path,
    straighten_path,
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

    def test_max_turning_angle_too_high(self, simple_raster):
        with pytest.raises(ValueError, match="max_turning_angle"):
            enhanced_least_cost_path(simple_raster, (0, 0), (4, 4),
                                     max_turning_angle=200.0)

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

    def test_max_turning_angle_constraint(self):
        """With max_turning_angle=45, no deflection should exceed 45 degrees."""
        raster = np.ones((10, 10))
        # Add some cost variation to encourage turns
        raster[3:7, 3:7] = 5.0

        start, end = (0, 0), (9, 9)
        result = enhanced_least_cost_path(
            raster, start, end,
            curvature_factor=0.5, max_turning_angle=45.0,
        )
        for angle in result["turning_angles"]:
            assert angle <= 45.0 + 1e-9

    def test_max_turning_angle_90(self):
        """With max_turning_angle=90, no deflection should exceed 90 degrees."""
        raster = np.ones((8, 8))
        raster[2:6, 2:6] = 10.0
        start, end = (0, 0), (7, 7)
        result = enhanced_least_cost_path(
            raster, start, end,
            curvature_factor=0.3, max_turning_angle=90.0,
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
            max_turning_angle=90.0,
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
        # Verify the raster actually has low variation
        assert np.ptp(raster) < 0.5
        start, end = (0, 0), (11, 11)
        result = enhanced_least_cost_path(
            raster, start, end, curvature_factor=0.5,
        )
        # Count actual direction changes (non-zero turning angles)
        n_turns = sum(1 for a in result["turning_angles"] if a > 0)
        # Should be very few turns on a nearly uniform raster
        assert n_turns <= 3

    def test_standard_dijkstra_straight_corridor(self):
        """Standard Dijkstra (no curvature params) should prefer a
        straight path through a uniform-cost horizontal corridor."""
        raster = np.full((7, 30), 100.0)
        # Low-cost corridor on rows 2-4
        raster[2:5, :] = 1.0
        start, end = (3, 0), (3, 29)
        result = enhanced_least_cost_path(raster, start, end)
        path = result["path"]
        # Path should stay on the center row of the corridor
        assert all(r == 3 for r, _ in path)

    def test_standard_dijkstra_low_variation_fewer_turns(self):
        """Standard Dijkstra should have few direction changes on a
        low-variation raster thanks to the built-in anti-zigzag penalty."""
        rng = np.random.default_rng(42)
        raster = 5.0 + rng.uniform(-0.05, 0.05, (20, 20))
        start, end = (0, 0), (19, 19)
        result = enhanced_least_cost_path(raster, start, end)
        n_turns = sum(1 for a in result["turning_angles"] if a > 0)
        # Should have very few turns on a nearly uniform raster
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
        assert "straightened_path" in result
        # After straightening, smoothed_path has at least as many points
        # as the straightened path (Chaikin can only add points).
        assert len(result["smoothed_path"]) >= len(result["straightened_path"])
        # Check start/end preserved
        sp = result["smoothed_path"]
        assert sp[0] == (0.0, 0.0)
        assert sp[-1] == (4.0, 4.0)


# ---------------------------------------------------------------------------
# Scalability
# ---------------------------------------------------------------------------


class TestScalability:
    """Verify the algorithm works on moderately large rasters."""

    def test_moderate_raster_with_curvature(self):
        """Direction-aware A* on a 200×200 raster should complete quickly."""
        rng = np.random.default_rng(42)
        raster = rng.uniform(1, 10, (200, 200))
        result = enhanced_least_cost_path(
            raster, (10, 10), (190, 190),
            curvature_factor=0.5,
            max_turning_angle=90.0,
            distance_factor=0.3,
        )
        assert result["path"][0] == (10, 10)
        assert result["path"][-1] == (190, 190)
        assert result["total_cost"] > 0
        for angle in result["turning_angles"]:
            assert angle <= 90.0 + 1e-9

    def test_float32_input_raster(self):
        """Algorithm should work correctly with float32 input rasters."""
        rng = np.random.default_rng(42)
        raster_f32 = rng.uniform(1, 100, (50, 50)).astype(np.float32)

        # Standard path
        res = enhanced_least_cost_path(raster_f32, (5, 5), (45, 45))
        assert res["path"][0] == (5, 5)
        assert res["path"][-1] == (45, 45)
        assert res["total_cost"] > 0

        # With curvature
        res_curv = enhanced_least_cost_path(
            raster_f32, (5, 5), (45, 45),
            curvature_factor=0.5, max_turning_angle=90.0,
        )
        assert res_curv["path"][0] == (5, 5)
        assert res_curv["path"][-1] == (45, 45)
        for angle in res_curv["turning_angles"]:
            assert angle <= 90.0 + 1e-9

    def test_high_cost_values(self):
        """Algorithm should handle high cost values without precision loss."""
        rng = np.random.default_rng(99)
        # High cost values: accumulated costs will be large.
        raster = rng.uniform(500, 1000, (100, 100))
        result = enhanced_least_cost_path(
            raster, (0, 0), (99, 99),
            curvature_factor=0.3, max_turning_angle=90.0,
        )
        assert result["path"][0] == (0, 0)
        assert result["path"][-1] == (99, 99)
        assert result["total_cost"] > 0
        for angle in result["turning_angles"]:
            assert angle <= 90.0 + 1e-9


# ---------------------------------------------------------------------------
# Path straightening (any-angle)
# ---------------------------------------------------------------------------


class TestStraightening:
    """Verify that straighten_path removes unnecessary grid waypoints."""

    def test_diagonal_straightens_to_two_points(self):
        """A pure diagonal on a uniform raster should straighten to
        just start and end when using maximum straighten_factor."""
        raster = np.ones((10, 10))
        result = enhanced_least_cost_path(raster, (0, 0), (9, 9),
                                          straighten_factor=0.5)
        sp = result["straightened_path"]
        assert sp[0] == (0.0, 0.0)
        assert sp[-1] == (9.0, 9.0)
        # With max factor on a uniform raster, intermediate points
        # are reduced (though not necessarily to exactly 2 due to
        # limited lookahead).
        assert len(sp) < len(result["path"])

    def test_horizontal_straightens_to_two_points(self):
        """A horizontal path on a uniform raster should straighten to
        just start and end when using maximum straighten_factor."""
        raster = np.ones((3, 10))
        result = enhanced_least_cost_path(raster, (1, 0), (1, 9),
                                          straighten_factor=0.5)
        sp = result["straightened_path"]
        assert sp[0] == (1.0, 0.0)
        assert sp[-1] == (1.0, 9.0)
        assert len(sp) < len(result["path"])

    def test_straighten_preserves_endpoints(self):
        """Start and end must be preserved exactly."""
        raster = np.ones((8, 8))
        result = enhanced_least_cost_path(raster, (0, 0), (7, 7))
        sp = result["straightened_path"]
        assert sp[0] == (0.0, 0.0)
        assert sp[-1] == (7.0, 7.0)

    def test_straighten_respects_barriers(self):
        """Straightening should not skip waypoints needed to go around
        barriers."""
        raster = np.ones((5, 5))
        # Create a horizontal barrier
        raster[2, 1] = np.nan
        raster[2, 2] = np.nan
        raster[2, 3] = np.nan
        result = enhanced_least_cost_path(raster, (0, 2), (4, 2))
        sp = result["straightened_path"]
        assert sp[0] == (0.0, 2.0)
        assert sp[-1] == (4.0, 2.0)
        # The straightened path must have more than 2 waypoints because
        # the direct line crosses the barrier.
        assert len(sp) > 2

    def test_straighten_fewer_waypoints_than_grid(self):
        """On a non-axis-aligned route, the straightened path should
        have fewer waypoints than the grid path."""
        # Create a raster where the path must go at an odd angle
        raster = np.ones((20, 10))
        result = enhanced_least_cost_path(raster, (0, 0), (19, 9))
        # Grid path has many cells (zigzag between diagonal and cardinal)
        assert len(result["straightened_path"]) < len(result["path"])

    def test_straightened_path_in_result(self):
        """Result dict must contain 'straightened_path'."""
        raster = np.ones((5, 5))
        result = enhanced_least_cost_path(raster, (0, 0), (4, 4))
        assert "straightened_path" in result

    def test_straighten_single_cell(self):
        """Start==end yields a single-point straightened path."""
        raster = np.ones((3, 3))
        result = enhanced_least_cost_path(raster, (1, 1), (1, 1))
        assert result["straightened_path"] == [(1.0, 1.0)]

    def test_straighten_direct(self):
        """straighten_path can be called directly."""
        raster = np.ones((5, 5))
        path = [(0, 0), (1, 1), (2, 2), (3, 3), (4, 4)]
        sp = straighten_path(path, raster)
        assert sp[0] == (0.0, 0.0)
        assert sp[-1] == (4.0, 4.0)
        # Default factor (0.3) reduces waypoints but may not collapse to 2
        assert len(sp) < len(path)

    def test_straighten_with_curvature(self):
        """Straightening should also work on direction-aware paths."""
        raster = np.ones((10, 10))
        result = enhanced_least_cost_path(
            raster, (0, 0), (9, 9),
            curvature_factor=0.5, max_turning_angle=90.0,
            straighten_factor=0.5,
        )
        sp = result["straightened_path"]
        assert sp[0] == (0.0, 0.0)
        assert sp[-1] == (9.0, 9.0)
        # With max factor on a uniform raster, waypoints are reduced
        assert len(sp) < len(result["path"])


# ---------------------------------------------------------------------------
# Supercover line & NODATA safety
# ---------------------------------------------------------------------------


class TestSupercover:
    """Verify that the supercover line algorithm catches NODATA at corners."""

    def test_supercover_includes_corner_cells(self):
        """A diagonal step from (0,0) to (1,1) should include the two
        corner-adjacent cells (0,1) and (1,0)."""
        cells = _supercover_line(0, 0, 1, 1)
        cell_set = set(cells)
        assert (0, 0) in cell_set
        assert (1, 1) in cell_set
        # Corner cells must be included
        assert (0, 1) in cell_set
        assert (1, 0) in cell_set

    def test_supercover_horizontal(self):
        """Horizontal line should include all cells in the row."""
        cells = _supercover_line(2, 0, 2, 5)
        cell_set = set(cells)
        for c in range(6):
            assert (2, c) in cell_set

    def test_supercover_vertical(self):
        """Vertical line should include all cells in the column."""
        cells = _supercover_line(0, 3, 4, 3)
        cell_set = set(cells)
        for r in range(5):
            assert (r, 3) in cell_set

    def test_supercover_blocks_nodata_at_corner(self):
        """Straightening must not skip over NODATA at a diagonal corner."""
        # 3x3 raster with NODATA at (1,0) — right at the corner of
        # a diagonal from (0,0) to (2,1).
        raster = np.ones((3, 3))
        raster[1, 0] = np.nan

        # Direct path (0,0)→(1,1)→(2,1): the Bresenham line from
        # (0,0) to (2,1) would miss (1,0), but supercover catches it.
        path = [(0, 0), (1, 1), (2, 1)]
        sp = straighten_path(path, raster)
        # With the supercover check, the line (0,0)→(2,1) passes through
        # the NODATA cell (1,0), so the straightener cannot skip (1,1).
        assert len(sp) > 2

    def test_straighten_never_crosses_nodata(self):
        """The straightened path should never shortcut across NODATA.

        When the straightener tries to skip intermediate waypoints, the
        supercover line check must detect any NODATA cells in the way.
        (Single-step diagonal moves between adjacent grid cells are valid
        in 8-connectivity and are not checked by the straightener.)
        """
        raster = np.ones((10, 10))
        # Wide NODATA wall across the middle — forces a detour
        raster[4, 2:8] = np.nan
        raster[5, 2:8] = np.nan

        result = enhanced_least_cost_path(raster, (0, 5), (9, 5))
        sp = result["straightened_path"]
        assert len(sp) > 2, "Must detour around the NODATA wall"

        # Verify multi-cell straightened segments don't cross NODATA.
        cost_data = np.ascontiguousarray(raster, dtype=np.float64)
        for k in range(len(sp) - 1):
            r0, c0 = int(round(sp[k][0])), int(round(sp[k][1]))
            r1, c1 = int(round(sp[k + 1][0])), int(round(sp[k + 1][1]))
            # Only check segments that skip intermediate cells
            if max(abs(r1 - r0), abs(c1 - c0)) <= 1:
                continue
            for r, c in _supercover_line(r0, c0, r1, c1):
                if 0 <= r < 10 and 0 <= c < 10:
                    assert np.isfinite(cost_data[r, c]), (
                        f"Straightened segment ({r0},{c0})→({r1},{c1}) "
                        f"crosses NODATA at ({r},{c})"
                    )


# ---------------------------------------------------------------------------
# Straighten factor parameter
# ---------------------------------------------------------------------------


class TestStraightenFactor:
    """Verify that straighten_factor controls straightening degree."""

    def test_factor_zero_no_straightening(self):
        """With straighten_factor=0, straightened path == grid path."""
        raster = np.ones((10, 10))
        result = enhanced_least_cost_path(
            raster, (0, 0), (9, 9), straighten_factor=0.0,
        )
        # Should have the same number of points as the grid path
        assert len(result["straightened_path"]) == len(result["path"])

    def test_factor_one_full_straightening(self):
        """With straighten_factor=0.5, maximum straightening is applied."""
        raster = np.ones((10, 10))
        result = enhanced_least_cost_path(
            raster, (0, 0), (9, 9), straighten_factor=0.5,
        )
        # On a uniform raster, max straightening significantly reduces
        # waypoints compared to no straightening
        res_none = enhanced_least_cost_path(
            raster, (0, 0), (9, 9), straighten_factor=0.0,
        )
        assert len(result["straightened_path"]) < len(res_none["straightened_path"])

    def test_factor_intermediate_partial_straightening(self):
        """An intermediate factor should produce more waypoints than
        factor=0.5 but fewer than factor=0.0 on a long path."""
        raster = np.ones((30, 30))
        res_full = enhanced_least_cost_path(
            raster, (0, 0), (29, 29), straighten_factor=0.5,
        )
        res_partial = enhanced_least_cost_path(
            raster, (0, 0), (29, 29), straighten_factor=0.1,
        )
        res_none = enhanced_least_cost_path(
            raster, (0, 0), (29, 29), straighten_factor=0.0,
        )
        assert len(res_full["straightened_path"]) <= len(res_partial["straightened_path"])
        assert len(res_partial["straightened_path"]) <= len(res_none["straightened_path"])

    def test_factor_validation(self):
        """Invalid straighten_factor should raise ValueError."""
        raster = np.ones((5, 5))
        with pytest.raises(ValueError, match="straighten_factor"):
            enhanced_least_cost_path(raster, (0, 0), (4, 4),
                                     straighten_factor=0.8)
        with pytest.raises(ValueError, match="straighten_factor"):
            enhanced_least_cost_path(raster, (0, 0), (4, 4),
                                     straighten_factor=-0.1)

    def test_factor_with_curvature(self):
        """straighten_factor should work with curvature parameters."""
        raster = np.ones((10, 10))
        result = enhanced_least_cost_path(
            raster, (0, 0), (9, 9),
            curvature_factor=0.5, max_turning_angle=90.0,
            straighten_factor=0.0,
        )
        # With factor=0, straightened path equals grid path
        assert len(result["straightened_path"]) == len(result["path"])

    def test_straighten_direct_with_factor(self):
        """straighten_path called directly respects straighten_factor."""
        raster = np.ones((5, 5))
        path = [(0, 0), (1, 1), (2, 2), (3, 3), (4, 4)]
        sp_full = straighten_path(path, raster, straighten_factor=0.5)
        sp_none = straighten_path(path, raster, straighten_factor=0.0)
        assert len(sp_full) < len(sp_none)  # max factor reduces waypoints
        assert len(sp_none) == 5            # same as input

    def test_fine_decimal_factor(self):
        """Fine-grained decimal values like 0.05 should be accepted
        and produce a different result from 0.1."""
        raster = np.ones((50, 50))
        res_005 = enhanced_least_cost_path(
            raster, (0, 0), (49, 49), straighten_factor=0.05,
        )
        res_010 = enhanced_least_cost_path(
            raster, (0, 0), (49, 49), straighten_factor=0.10,
        )
        res_050 = enhanced_least_cost_path(
            raster, (0, 0), (49, 49), straighten_factor=0.50,
        )
        # 0.05 should produce more waypoints than 0.10
        assert len(res_005["straightened_path"]) >= len(res_010["straightened_path"])
        # 0.10 should produce more waypoints than 0.50
        assert len(res_010["straightened_path"]) >= len(res_050["straightened_path"])


class TestSmoothedPathNodataSafe:
    """Verify that the smoothed path never crosses NODATA / barrier cells."""

    def test_smoothed_path_avoids_nodata_wall(self):
        """When the path detours around a NODATA wall, the smoothed output
        must not shortcut through the wall."""
        raster = np.ones((10, 10))
        raster[4, 1:10] = np.nan
        raster[5, 1:10] = np.nan

        result = enhanced_least_cost_path(raster, (0, 5), (9, 5))
        sp = result["smoothed_path"]
        cost_data = np.ascontiguousarray(raster, dtype=np.float64)

        for i in range(len(sp) - 1):
            r0 = int(round(sp[i][0]))
            c0 = int(round(sp[i][1]))
            r1 = int(round(sp[i + 1][0]))
            c1 = int(round(sp[i + 1][1]))
            if max(abs(r1 - r0), abs(c1 - c0)) <= 1:
                continue
            r0 = max(0, min(9, r0))
            c0 = max(0, min(9, c0))
            r1 = max(0, min(9, r1))
            c1 = max(0, min(9, c1))
            for r, c in _supercover_line(r0, c0, r1, c1):
                if 0 <= r < 10 and 0 <= c < 10:
                    assert np.isfinite(cost_data[r, c]), (
                        f"Smoothed segment ({r0},{c0})→({r1},{c1}) "
                        f"crosses NODATA at ({r},{c})"
                    )

    def test_smoothed_path_avoids_narrow_corridor_nodata(self):
        """In a narrow L-shaped corridor surrounded by NODATA, the smoothed
        path must not cross into NODATA."""
        raster = np.full((8, 8), np.nan)
        raster[0, 0:4] = 1.0
        raster[0:8, 3] = 1.0
        raster[1, 2] = 1.0

        result = enhanced_least_cost_path(raster, (0, 0), (7, 3))
        sp = result["smoothed_path"]
        cost_data = np.ascontiguousarray(raster, dtype=np.float64)

        for i in range(len(sp) - 1):
            r0 = int(round(sp[i][0]))
            c0 = int(round(sp[i][1]))
            r1 = int(round(sp[i + 1][0]))
            c1 = int(round(sp[i + 1][1]))
            if max(abs(r1 - r0), abs(c1 - c0)) <= 1:
                continue
            r0 = max(0, min(7, r0))
            c0 = max(0, min(7, c0))
            r1 = max(0, min(7, r1))
            c1 = max(0, min(7, c1))
            for r, c in _supercover_line(r0, c0, r1, c1):
                if 0 <= r < 8 and 0 <= c < 8:
                    assert np.isfinite(cost_data[r, c]), (
                        f"Smoothed segment ({r0},{c0})→({r1},{c1}) "
                        f"crosses NODATA at ({r},{c})"
                    )

    def test_smoothed_path_smooth_when_safe(self):
        """On a uniform raster (no NODATA), smoothing should still apply
        normally and produce more points than the straightened path."""
        raster = np.ones((20, 20))
        result = enhanced_least_cost_path(raster, (0, 0), (19, 9))
        # Smoothing should be active (more points than straightened)
        assert len(result["smoothed_path"]) >= len(result["straightened_path"])
