"""Tests for pure_python.survey_aware_lcp."""

import math

import numpy as np
import pytest

from pure_python.survey_aware_lcp import (
    build_composite_cost,
    survey_aware_least_cost_path,
    _compute_survey_stats,
    CATZOC_MAX,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def uniform_risk():
    """5×10 risk raster with uniform cost of 50."""
    return np.full((5, 10), 50.0)


@pytest.fixture()
def uniform_survey():
    """5×10 survey raster: all CATZOC 0 (fully unsurveyed)."""
    return np.zeros((5, 10), dtype=np.float32)


@pytest.fixture()
def mixed_survey():
    """5×10 survey raster with a vertical band of full coverage (CATZOC 3)
    in the centre and gaps (CATZOC 0) on the sides."""
    s = np.zeros((5, 10), dtype=np.float32)
    s[:, 4:7] = 3.0  # full coverage in the middle
    return s


# ---------------------------------------------------------------------------
# Validation tests
# ---------------------------------------------------------------------------

class TestValidation:

    def test_shape_mismatch_raises(self):
        risk = np.ones((5, 5))
        survey = np.ones((6, 5))
        with pytest.raises(ValueError, match="shape"):
            survey_aware_least_cost_path(risk, survey, (0, 0), (4, 4))

    def test_survey_weight_out_of_range(self):
        risk = np.ones((5, 5))
        survey = np.ones((5, 5))
        with pytest.raises(ValueError, match="survey_weight"):
            survey_aware_least_cost_path(risk, survey, (0, 0), (4, 4),
                                         survey_weight=1.5)

    def test_negative_safety_threshold(self):
        risk = np.ones((5, 5))
        survey = np.ones((5, 5))
        with pytest.raises(ValueError, match="safety_threshold"):
            survey_aware_least_cost_path(risk, survey, (0, 0), (4, 4),
                                         safety_threshold=-1)

    def test_invalid_survey_nodata_as(self):
        risk = np.ones((5, 5))
        survey = np.ones((5, 5))
        with pytest.raises(ValueError, match="survey_nodata_as"):
            survey_aware_least_cost_path(risk, survey, (0, 0), (4, 4),
                                         survey_nodata_as="bad_value")

    def test_start_blocked_by_threshold(self):
        # With the soft-penalty design, start/end above the safety threshold are
        # *not* errors – they just receive a higher cost.  Only NaN/Inf cells
        # are truly impassable.
        risk = np.full((5, 5), 50.0)
        risk[0, 0] = 200.0  # start cell exceeds threshold but is still passable
        survey = np.zeros((5, 5))
        # Should succeed without raising
        result = survey_aware_least_cost_path(risk, survey, (0, 0), (4, 4),
                                              safety_threshold=100)
        assert result["path"][0] == (0, 0)

    def test_end_blocked_by_threshold(self):
        # Same as above: end above threshold is penalised, not blocked.
        risk = np.full((5, 5), 50.0)
        risk[4, 4] = 200.0  # end cell exceeds threshold but is still passable
        survey = np.zeros((5, 5))
        result = survey_aware_least_cost_path(risk, survey, (0, 0), (4, 4),
                                              safety_threshold=100)
        assert result["path"][-1] == (4, 4)

    def test_start_out_of_bounds(self):
        risk = np.ones((5, 5))
        survey = np.ones((5, 5))
        with pytest.raises(ValueError, match="Start point"):
            survey_aware_least_cost_path(risk, survey, (10, 0), (4, 4))

    def test_risk_raster_not_2d(self):
        risk = np.ones((5, 5, 2))
        survey = np.ones((5, 5))
        with pytest.raises(ValueError, match="2-D"):
            survey_aware_least_cost_path(risk, survey, (0, 0), (4, 4))


# ---------------------------------------------------------------------------
# build_composite_cost
# ---------------------------------------------------------------------------

class TestBuildCompositeCost:

    def test_weight_zero_equals_norm_risk(self):
        """survey_weight=0 → composite == norm_risk (survey has no effect)."""
        risk = np.array([[10.0, 20.0], [30.0, 40.0]])
        survey = np.array([[0.0, 3.0], [0.0, 3.0]])
        composite, _ = build_composite_cost(risk, survey, safety_threshold=200,
                                            survey_weight=0.0)
        risk_max = 40.0
        expected = risk / risk_max
        np.testing.assert_allclose(composite, expected)

    def test_weight_one_survey_only(self):
        """survey_weight=1 → composite == survey / CATZOC_MAX."""
        risk = np.array([[10.0, 20.0], [30.0, 40.0]])
        survey = np.array([[0.0, 1.0], [2.0, 3.0]])
        composite, _ = build_composite_cost(risk, survey, safety_threshold=200,
                                            survey_weight=1.0)
        # With survey_weight=1: composite = survey / 3
        expected = survey / CATZOC_MAX
        np.testing.assert_allclose(composite, expected)

    def test_safety_masking(self):
        """Cells above threshold must be penalised but still finite (passable)."""
        risk = np.array([[10.0, 150.0], [30.0, 40.0]])
        survey = np.zeros((2, 2))
        composite, above_count = build_composite_cost(risk, survey,
                                                       safety_threshold=100,
                                                       survey_weight=0.5)
        assert above_count == 1
        # The above-threshold cell should be finite (penalised, not masked).
        assert np.isfinite(composite[0, 1]), "Penalised cell should remain finite"
        assert np.isfinite(composite[0, 0]), "Non-penalised cell should be finite"
        # With penalty_multiplier > 0 the above-threshold cell has a higher
        # composite cost than when the penalty is disabled.
        composite_no_penalty, _ = build_composite_cost(risk, survey,
                                                        safety_threshold=100,
                                                        survey_weight=0.5,
                                                        penalty_multiplier=0.0)
        assert composite[0, 1] >= composite_no_penalty[0, 1], (
            "Penalised cell should cost at least as much as with no penalty"
        )

    def test_survey_nodata_as_unsurveyed(self):
        """NaN survey cells treated as CATZOC 0 should lower composite cost."""
        risk = np.ones((3, 3)) * 10.0
        survey = np.full((3, 3), 3.0)  # all fully surveyed
        survey[1, 1] = np.nan          # centre cell: nodata → treat as CATZOC 0
        composite_unsurveyed, _ = build_composite_cost(
            risk, survey, safety_threshold=100,
            survey_weight=0.5, survey_nodata_as="unsurveyed"
        )
        composite_surveyed, _ = build_composite_cost(
            risk, survey, safety_threshold=100,
            survey_weight=0.5, survey_nodata_as="surveyed"
        )
        # When treated as CATZOC 0, survey_component = 0/3 = 0 → lower cost
        # When treated as CATZOC 3, survey_component = 3/3 = 1 → higher cost
        assert composite_unsurveyed[1, 1] < composite_surveyed[1, 1]

    def test_original_nan_preserved(self):
        """NaN cells in risk_raster stay NaN regardless of threshold."""
        risk = np.array([[np.nan, 50.0], [30.0, 40.0]])
        survey = np.zeros((2, 2))
        composite, above_count = build_composite_cost(risk, survey,
                                                       safety_threshold=200,
                                                       survey_weight=0.3)
        assert above_count == 0  # threshold didn't penalise anything
        assert not np.isfinite(composite[0, 0])

    def test_composite_values_in_range(self):
        """Without a penalty multiplier, all finite composite values must lie in [0, 1]."""
        rng = np.random.default_rng(42)
        risk = rng.uniform(1, 100, (20, 20))
        survey = rng.integers(0, 4, (20, 20)).astype(float)
        # Use penalty_multiplier=0 so no amplification occurs.
        composite, _ = build_composite_cost(risk, survey, safety_threshold=80,
                                            survey_weight=0.4,
                                            penalty_multiplier=0.0)
        finite = composite[np.isfinite(composite)]
        assert finite.min() >= -1e-9
        assert finite.max() <= 1.0 + 1e-9


# ---------------------------------------------------------------------------
# survey_score and profile
# ---------------------------------------------------------------------------

class TestSurveyStats:

    def test_all_gap_path_score_is_one(self):
        survey = np.zeros((5, 5))
        path = [(0, 0), (1, 1), (2, 2), (3, 3), (4, 4)]
        profile, score = _compute_survey_stats(path, survey)
        assert score == pytest.approx(1.0)
        assert all(v == 0.0 for v in profile)

    def test_all_full_coverage_score_is_zero(self):
        survey = np.full((5, 5), 3.0)
        path = [(0, 0), (1, 1), (2, 2), (3, 3), (4, 4)]
        profile, score = _compute_survey_stats(path, survey)
        assert score == pytest.approx(0.0)

    def test_half_unsurveyed_score(self):
        survey = np.array([[0.0, 0.0, 3.0, 3.0]])
        path = [(0, 0), (0, 1), (0, 2), (0, 3)]
        profile, score = _compute_survey_stats(path, survey)
        assert score == pytest.approx(0.5)


# ---------------------------------------------------------------------------
# End-to-end pathfinding
# ---------------------------------------------------------------------------

class TestEndToEnd:

    def test_basic_path_found(self, uniform_risk, uniform_survey):
        result = survey_aware_least_cost_path(
            uniform_risk, uniform_survey, (0, 0), (4, 9)
        )
        assert result["path"][0] == (0, 0)
        assert result["path"][-1] == (4, 9)

    def test_output_keys(self, uniform_risk, uniform_survey):
        result = survey_aware_least_cost_path(
            uniform_risk, uniform_survey, (0, 0), (4, 9)
        )
        for key in ("path", "straightened_path", "smoothed_path",
                    "total_cost", "path_length",
                    "survey_coverage_profile", "survey_score",
                    "above_threshold_count", "composite_cost_raster"):
            assert key in result, f"Missing key: {key}"

    def test_weight_zero_ignores_survey(self, uniform_risk):
        """survey_weight=0 should find the same path as the base algorithm."""
        survey_all_gap = np.zeros((5, 10))
        survey_all_full = np.full((5, 10), 3.0)

        r0 = survey_aware_least_cost_path(
            uniform_risk, survey_all_gap, (0, 0), (4, 9), survey_weight=0.0
        )
        r1 = survey_aware_least_cost_path(
            uniform_risk, survey_all_full, (0, 0), (4, 9), survey_weight=0.0
        )
        # Composite costs are identical (survey ignored), so paths should match.
        assert r0["path"] == r1["path"]

    def test_survey_weight_attracts_to_gap_areas(self):
        """With high survey_weight, the path should prefer the unsurveyed corridor."""
        # Build a raster where the top half is CATZOC 0 (gap) and
        # the bottom half is CATZOC 3 (full), with uniform risk.
        risk = np.ones((10, 10)) * 10.0
        survey = np.zeros((10, 10))
        survey[5:, :] = 3.0  # bottom half: fully surveyed

        result_high_weight = survey_aware_least_cost_path(
            risk, survey, (0, 0), (9, 9), survey_weight=1.0, safety_threshold=200
        )
        result_no_weight = survey_aware_least_cost_path(
            risk, survey, (0, 0), (9, 9), survey_weight=0.0, safety_threshold=200
        )

        # With full survey weight the path should score higher (more gap cells).
        assert result_high_weight["survey_score"] >= result_no_weight["survey_score"]

    def test_barrier_avoidance(self):
        """Cells with high risk should be strongly avoided (soft penalty).
        With a large penalty_multiplier, the path should route around them."""
        risk = np.ones((5, 10)) * 20.0
        risk[2, 3:7] = 200.0  # high-risk row – penalised but not blocked
        survey = np.zeros((5, 10))

        result = survey_aware_least_cost_path(
            risk, survey, (2, 0), (2, 9),
            safety_threshold=100, penalty_multiplier=50.0
        )
        # With a 50× multiplier the path should avoid the high-risk cells.
        for r, c in result["path"]:
            assert risk[r, c] <= 100.0 or (r, c) == (2, 0) or (r, c) == (2, 9), (
                f"Path passed through high-risk cell ({r}, {c}) "
                f"with risk {risk[r, c]}"
            )

    def test_above_threshold_count(self):
        """above_threshold_count must equal the number of risk cells above threshold."""
        risk = np.ones((5, 5)) * 30.0
        risk[2, 2] = 150.0
        risk[3, 3] = 120.0
        survey = np.zeros((5, 5))

        result = survey_aware_least_cost_path(
            risk, survey, (0, 0), (4, 4), safety_threshold=100
        )
        assert result["above_threshold_count"] == 2

    def test_progress_callback_called(self, uniform_risk, uniform_survey):
        messages = []
        survey_aware_least_cost_path(
            uniform_risk, uniform_survey, (0, 0), (4, 9),
            progress_callback=messages.append,
        )
        assert len(messages) > 0
        assert any("[Survey-Aware]" in m for m in messages)

    def test_survey_coverage_profile_length(self, uniform_risk, uniform_survey):
        result = survey_aware_least_cost_path(
            uniform_risk, uniform_survey, (0, 0), (4, 9)
        )
        assert len(result["survey_coverage_profile"]) == len(result["path"])

    def test_start_equals_end(self, uniform_risk, uniform_survey):
        result = survey_aware_least_cost_path(
            uniform_risk, uniform_survey, (2, 5), (2, 5)
        )
        assert result["path"] == [(2, 5)]
        assert result["survey_score"] == pytest.approx(1.0)

    def test_no_path_raises(self):
        risk = np.ones((5, 10)) * 20.0
        # Create a full vertical wall of NaN cells that blocks all paths.
        risk[:, 5] = np.nan
        survey = np.zeros((5, 10))
        with pytest.raises(RuntimeError, match="No path"):
            survey_aware_least_cost_path(
                risk, survey, (2, 0), (2, 9), safety_threshold=100
            )

    def test_curvature_and_distance_factors_accepted(
        self, uniform_risk, uniform_survey
    ):
        """Verify that curvature/distance kwargs are forwarded correctly."""
        result = survey_aware_least_cost_path(
            uniform_risk, uniform_survey, (0, 0), (4, 9),
            curvature_factor=0.5,
            distance_factor=0.3,
            max_turning_angle=135.0,
        )
        assert result["path"][-1] == (4, 9)
