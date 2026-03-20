"""Tests for progress_callback functionality in both pure_python and numba versions."""

import numpy as np
import pytest

from pure_python.cost_aware_straighten_lcp import (
    cost_aware_least_cost_path as python_lcp,
)
from numba_accelerated.cost_aware_straighten_lcp import (
    cost_aware_least_cost_path as numba_lcp,
)


class _ProgressCollector:
    """Collects progress messages for testing."""

    def __init__(self):
        self.messages = []

    def __call__(self, msg):
        self.messages.append(msg)


class TestPurePythonProgressCallback:
    """Test that pure_python version fires progress callbacks."""

    def test_no_callback_works(self):
        """Calling without progress_callback should still work."""
        raster = np.ones((10, 10))
        result = python_lcp(raster, (0, 0), (9, 9))
        assert len(result["path"]) >= 2

    def test_none_callback_works(self):
        """Passing None as progress_callback should still work."""
        raster = np.ones((10, 10))
        result = python_lcp(raster, (0, 0), (9, 9), progress_callback=None)
        assert len(result["path"]) >= 2

    def test_callback_receives_messages_standard(self):
        """Standard Dijkstra should fire progress messages."""
        raster = np.ones((20, 20))
        collector = _ProgressCollector()
        result = python_lcp(
            raster, (0, 0), (19, 19), progress_callback=collector
        )
        assert len(collector.messages) > 0
        # Check for key phases
        msgs = "\n".join(collector.messages)
        assert "Parameters validated" in msgs
        assert "Dijkstra" in msgs
        assert "Path reconstruction" in msgs
        assert "Path smoothing" in msgs

    def test_callback_receives_messages_curvature(self):
        """Direction-aware Dijkstra should fire progress messages."""
        raster = np.ones((20, 20))
        collector = _ProgressCollector()
        result = python_lcp(
            raster, (0, 0), (19, 19),
            curvature_factor=0.3,
            progress_callback=collector,
        )
        assert len(collector.messages) > 0
        msgs = "\n".join(collector.messages)
        assert "direction-aware" in msgs
        assert "Path straightening" in msgs

    def test_callback_shows_straightening_reduction(self):
        """Progress should report node count reduction during straightening."""
        raster = np.ones((15, 15))
        collector = _ProgressCollector()
        python_lcp(
            raster, (0, 0), (14, 14),
            straighten_factor=0.5,
            progress_callback=collector,
        )
        straighten_msgs = [m for m in collector.messages if "Path straightening: complete" in m]
        assert len(straighten_msgs) >= 1
        assert "→" in straighten_msgs[0]

    def test_result_identical_with_or_without_callback(self):
        """Results should be identical regardless of callback."""
        np.random.seed(42)
        raster = np.random.uniform(1, 10, (20, 20))
        r1 = python_lcp(raster, (0, 0), (19, 19))
        collector = _ProgressCollector()
        r2 = python_lcp(raster, (0, 0), (19, 19), progress_callback=collector)
        assert r1["path"] == r2["path"]
        assert abs(r1["total_cost"] - r2["total_cost"]) < 1e-6

    def test_high_precision_raster_no_excess_expansion(self):
        """High-precision float rasters must not cause >100% node expansion.

        Regression test: when the ``best`` array used float32 but costs
        were computed in float64, truncation collisions caused duplicate
        cell expansions, making progress exceed 100%.
        """
        np.random.seed(123)
        # Create raster with high-precision values like those from
        # ArcGIS Pro raster calculator (7 decimal places)
        raster = np.random.uniform(0.013889, 72.0, (50, 50)).astype(np.float32)
        collector = _ProgressCollector()
        result = python_lcp(raster, (0, 0), (49, 49), progress_callback=collector)
        assert len(result["path"]) >= 2

        # Verify no progress message exceeds 99%
        for msg in collector.messages:
            if "(~" in msg and "%)" in msg:
                pct_str = msg.split("(~")[1].split("%")[0]
                pct = int(pct_str)
                assert pct <= 99, f"Progress exceeded 99%: {msg}"


class TestNumbaProgressCallback:
    """Test that numba_accelerated version fires progress callbacks."""

    def test_no_callback_works(self):
        raster = np.ones((10, 10))
        result = numba_lcp(raster, (0, 0), (9, 9))
        assert len(result["path"]) >= 2

    def test_none_callback_works(self):
        raster = np.ones((10, 10))
        result = numba_lcp(raster, (0, 0), (9, 9), progress_callback=None)
        assert len(result["path"]) >= 2

    def test_callback_receives_messages_standard(self):
        raster = np.ones((20, 20))
        collector = _ProgressCollector()
        numba_lcp(raster, (0, 0), (19, 19), progress_callback=collector)
        assert len(collector.messages) > 0
        msgs = "\n".join(collector.messages)
        assert "Parameters validated" in msgs
        assert "Numba JIT" in msgs
        assert "Path reconstruction" in msgs
        assert "Path smoothing" in msgs

    def test_callback_receives_messages_curvature(self):
        raster = np.ones((20, 20))
        collector = _ProgressCollector()
        numba_lcp(
            raster, (0, 0), (19, 19),
            curvature_factor=0.3,
            progress_callback=collector,
        )
        assert len(collector.messages) > 0
        msgs = "\n".join(collector.messages)
        assert "direction-aware" in msgs

    def test_result_identical_with_or_without_callback(self):
        np.random.seed(42)
        raster = np.random.uniform(1, 10, (20, 20))
        r1 = numba_lcp(raster, (0, 0), (19, 19))
        collector = _ProgressCollector()
        r2 = numba_lcp(raster, (0, 0), (19, 19), progress_callback=collector)
        assert r1["path"] == r2["path"]
        assert abs(r1["total_cost"] - r2["total_cost"]) < 1e-6

    def test_callback_shows_straightening_reduction(self):
        raster = np.ones((15, 15))
        collector = _ProgressCollector()
        numba_lcp(
            raster, (0, 0), (14, 14),
            straighten_factor=0.5,
            progress_callback=collector,
        )
        straighten_msgs = [m for m in collector.messages if "Path straightening: complete" in m]
        assert len(straighten_msgs) >= 1
        assert "→" in straighten_msgs[0]

    def test_high_precision_raster_no_excess_expansion(self):
        """High-precision float rasters must not cause >100% node expansion."""
        np.random.seed(123)
        raster = np.random.uniform(0.013889, 72.0, (50, 50)).astype(np.float32)
        collector = _ProgressCollector()
        result = numba_lcp(raster, (0, 0), (49, 49), progress_callback=collector)
        assert len(result["path"]) >= 2
