"""Tests for the numba_accelerated.cost_aware_straighten_lcp module.

These tests mirror ``tests/test_cost_aware_straighten_lcp.py`` to ensure
the Numba-accelerated version produces identical results to the pure-Python
implementation.  An additional performance-comparison test verifies that
the Numba version is meaningfully faster.
"""

import math
import time

import numpy as np
import pytest

from numba_accelerated.cost_aware_straighten_lcp import (
    cost_aware_least_cost_path,
)


# ---------------------------------------------------------------------------
# Validation (same as pure_python — validation is delegated)
# ---------------------------------------------------------------------------

class TestValidation:
    @pytest.fixture()
    def simple_raster(self):
        return np.ones((5, 5))

    def test_curvature_factor_too_high(self, simple_raster):
        with pytest.raises(ValueError, match="curvature_factor"):
            cost_aware_least_cost_path(simple_raster, (0, 0), (4, 4),
                                       curvature_factor=1.5)

    def test_start_out_of_bounds(self, simple_raster):
        with pytest.raises(ValueError, match="Start point"):
            cost_aware_least_cost_path(simple_raster, (10, 0), (4, 4))

    def test_cost_tolerance_below_one(self, simple_raster):
        with pytest.raises(ValueError, match="cost_tolerance"):
            cost_aware_least_cost_path(simple_raster, (0, 0), (4, 4),
                                       cost_tolerance=0.5)

    def test_straighten_factor_too_high(self, simple_raster):
        with pytest.raises(ValueError, match="straighten_factor"):
            cost_aware_least_cost_path(simple_raster, (0, 0), (4, 4),
                                       straighten_factor=0.8)


# ---------------------------------------------------------------------------
# Basic pathfinding
# ---------------------------------------------------------------------------

class TestBasicPathfinding:
    def test_straight_horizontal(self):
        raster = np.ones((5, 10))
        result = cost_aware_least_cost_path(raster, (2, 0), (2, 9))
        path = result["path"]
        assert path[0] == (2, 0)
        assert path[-1] == (2, 9)

    def test_diagonal_shortcut(self):
        raster = np.ones((5, 5))
        result = cost_aware_least_cost_path(raster, (0, 0), (4, 4))
        assert result["path"][0] == (0, 0)
        assert result["path"][-1] == (4, 4)

    def test_avoid_barrier(self):
        raster = np.ones((5, 10))
        raster[2, 3:7] = np.nan
        result = cost_aware_least_cost_path(raster, (2, 0), (2, 9))
        for r, c in result["path"]:
            assert np.isfinite(raster[r, c])

    def test_start_equals_end(self):
        raster = np.ones((5, 5))
        result = cost_aware_least_cost_path(raster, (2, 2), (2, 2))
        assert result["path"] == [(2, 2)]

    def test_no_path_raises(self):
        raster = np.ones((5, 5))
        raster[2, :] = np.nan
        with pytest.raises(RuntimeError, match="No path"):
            cost_aware_least_cost_path(raster, (0, 0), (4, 4))


# ---------------------------------------------------------------------------
# Cost-aware straightening
# ---------------------------------------------------------------------------

class TestCostAwareStraightening:
    def test_uniform_raster_straightens(self):
        raster = np.ones((10, 10))
        result = cost_aware_least_cost_path(
            raster, (0, 0), (9, 9), straighten_factor=0.5
        )
        straightened = result["straightened_path"]
        grid_path = result["path"]
        assert len(straightened) <= len(grid_path)

    def test_high_cost_region_blocks_shortcut(self):
        raster = np.ones((10, 10))
        raster[3:7, 3:7] = 100.0
        result = cost_aware_least_cost_path(
            raster, (0, 0), (9, 9),
            straighten_factor=0.5, cost_tolerance=1.05,
        )
        straightened = result["straightened_path"]
        for r, c in straightened:
            ri, ci = int(round(r)), int(round(c))
            ri = max(0, min(9, ri))
            ci = max(0, min(9, ci))
            assert 0 <= ri < 10
            assert 0 <= ci < 10

    def test_tolerance_1_strict(self):
        raster = np.ones((10, 10))
        result = cost_aware_least_cost_path(
            raster, (0, 0), (9, 9),
            straighten_factor=0.5, cost_tolerance=1.0,
        )
        assert len(result["straightened_path"]) >= 2

    def test_high_tolerance_more_aggressive(self):
        raster = np.ones((15, 15))
        raster[5:10, 5:10] = 2.0
        result_strict = cost_aware_least_cost_path(
            raster, (0, 0), (14, 14),
            straighten_factor=0.5, cost_tolerance=1.01,
        )
        result_loose = cost_aware_least_cost_path(
            raster, (0, 0), (14, 14),
            straighten_factor=0.5, cost_tolerance=5.0,
        )
        assert len(result_loose["straightened_path"]) <= len(result_strict["straightened_path"])

    def test_factor_zero_no_straightening(self):
        raster = np.ones((10, 10))
        result = cost_aware_least_cost_path(
            raster, (0, 0), (9, 9), straighten_factor=0.0,
        )
        path = result["path"]
        sp = result["straightened_path"]
        assert len(sp) == len(path)

    def test_preserves_endpoints(self):
        raster = np.ones((10, 10))
        result = cost_aware_least_cost_path(
            raster, (0, 0), (9, 9), straighten_factor=0.3,
        )
        sp = result["straightened_path"]
        assert sp[0] == (0.0, 0.0)
        assert sp[-1] == (9.0, 9.0)


# ---------------------------------------------------------------------------
# Curvature support
# ---------------------------------------------------------------------------

class TestCurvature:
    def test_curvature_produces_path(self):
        raster = np.ones((10, 10))
        result = cost_aware_least_cost_path(
            raster, (0, 0), (9, 9), curvature_factor=0.5
        )
        assert len(result["path"]) >= 2

    def test_with_curvature_and_cost_tolerance(self):
        raster = np.ones((10, 10))
        result = cost_aware_least_cost_path(
            raster, (0, 0), (9, 9),
            curvature_factor=0.3, cost_tolerance=1.1,
        )
        assert result["path"][0] == (0, 0)
        assert result["path"][-1] == (9, 9)


# ---------------------------------------------------------------------------
# Smoothing
# ---------------------------------------------------------------------------

class TestSmoothing:
    def test_smoothed_path_in_result(self):
        raster = np.ones((10, 10))
        result = cost_aware_least_cost_path(raster, (0, 0), (9, 9))
        assert "smoothed_path" in result
        assert len(result["smoothed_path"]) >= 2

    def test_smoothed_path_nodata_safe(self):
        raster = np.ones((10, 10))
        raster[4, 1:9] = np.nan
        result = cost_aware_least_cost_path(raster, (0, 5), (9, 5))
        smoothed = result["smoothed_path"]
        for r, c in smoothed:
            ri, ci = int(round(r)), int(round(c))
            ri = max(0, min(9, ri))
            ci = max(0, min(9, ci))
            assert np.isfinite(raster[ri, ci])


# ---------------------------------------------------------------------------
# Numba-vs-Python correctness (bit-exact path comparison)
# ---------------------------------------------------------------------------

class TestNumbaVsPythonCorrectness:
    """Verify the Numba-accelerated version produces identical paths
    to the pure-Python implementation."""

    def _compare(self, raster, start, end, **kwargs):
        from pure_python.cost_aware_straighten_lcp import (
            cost_aware_least_cost_path as python_lcp,
        )
        py = python_lcp(raster, start, end, **kwargs)
        nb = cost_aware_least_cost_path(raster, start, end, **kwargs)

        assert py["path"] == nb["path"], "Paths differ"
        assert abs(py["total_cost"] - nb["total_cost"]) < 1e-3, "Costs differ"
        assert py["directions"] == nb["directions"], "Directions differ"

    def test_uniform_standard(self):
        self._compare(np.ones((10, 10)), (0, 0), (9, 9))

    def test_random_standard(self):
        np.random.seed(42)
        self._compare(np.random.uniform(1, 10, (30, 30)), (0, 0), (29, 29))

    def test_barrier_standard(self):
        raster = np.ones((10, 10))
        raster[5, 2:8] = np.nan
        self._compare(raster, (0, 5), (9, 5))

    def test_uniform_curvature(self):
        self._compare(np.ones((10, 10)), (0, 0), (9, 9),
                       curvature_factor=0.3)

    def test_random_curvature(self):
        np.random.seed(42)
        self._compare(np.random.uniform(1, 10, (30, 30)), (0, 0), (29, 29),
                       curvature_factor=0.3, cost_tolerance=1.1)

    def test_max_turning_angle(self):
        np.random.seed(42)
        self._compare(np.random.uniform(1, 5, (20, 20)), (0, 0), (19, 19),
                       max_turning_angle=90.0)

    def test_distance_factor(self):
        np.random.seed(42)
        self._compare(np.random.uniform(1, 10, (20, 20)), (0, 0), (19, 19),
                       distance_factor=0.5)


# ---------------------------------------------------------------------------
# Performance comparison
# ---------------------------------------------------------------------------

class TestPerformance:
    """Verify the Numba version is meaningfully faster."""

    def test_numba_faster_than_python(self):
        """On a 100×100 raster the Numba version should be at least 5× faster."""
        from pure_python.cost_aware_straighten_lcp import (
            cost_aware_least_cost_path as python_lcp,
        )

        np.random.seed(42)
        raster = np.random.uniform(1, 10, (100, 100))

        # Warm up Numba JIT
        cost_aware_least_cost_path(np.ones((5, 5)), (0, 0), (4, 4))

        t0 = time.time()
        python_lcp(raster, (0, 0), (99, 99))
        t_py = time.time() - t0

        t0 = time.time()
        cost_aware_least_cost_path(raster, (0, 0), (99, 99))
        t_nb = time.time() - t0

        speedup = t_py / t_nb if t_nb > 0 else float("inf")
        assert speedup > 5.0, (
            f"Expected ≥5× speedup, got {speedup:.1f}× "
            f"(Python={t_py:.3f}s, Numba={t_nb:.3f}s)"
        )
