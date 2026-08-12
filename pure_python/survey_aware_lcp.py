"""
Survey-Aware Least Cost Path (thin wrapper over cost_aware_straighten_lcp)
==========================================================================

This module extends the standard Cost-Aware LCP tool for survey vessels.
Instead of minimising navigation risk alone, it also rewards traversal
through under-surveyed areas (low CATZOC value).

Design
------
Two independent objectives are balanced via a composite cost raster built
before the Dijkstra search:

1. **Safety constraint (hard)**: cells whose risk value exceeds
   ``safety_threshold`` are masked to NaN and are unreachable.
2. **Survey-value objective (soft)**: cells with low CATZOC coverage
   (Gap / void-of-soundings) are cheaper to traverse, encouraging the
   path to collect data in under-surveyed areas.

The composite cost formula is::

    composite = (1 - survey_weight) * norm_risk
              + survey_weight       * (1 - norm_survey_cost)

where::

    norm_risk         = risk_raster / max(finite risk values after masking)
    norm_survey_cost  = (3 - survey_raster) / 3   # inverted and normalised

CATZOC values and their meaning
---------------------------------
* 0 – Gap / Void of Soundings  → highest survey priority (lowest composite cost)
* 1 – Minimal Coverage          → high survey priority
* 2 – Moderate Coverage         → low survey priority
* 3 – Full Bottom Coverage      → lowest survey priority (highest composite cost)

NaN cells in ``survey_raster`` are treated according to the
``survey_nodata_as`` parameter (default ``"unsurveyed"`` → treated as 0).

Public API
----------
``survey_aware_least_cost_path(risk_raster, survey_raster, start, end, ...)``
    Main entry point – mirrors the signature of
    ``cost_aware_least_cost_path`` plus the three new parameters.

Dependencies: numpy (required).  Delegates all path-finding to
``pure_python.cost_aware_straighten_lcp``.
"""

from typing import Dict, List, Optional, Tuple

import numpy as np

from pure_python.cost_aware_straighten_lcp import (
    cost_aware_least_cost_path,
)

# ---------------------------------------------------------------------------
# CATZOC constants
# ---------------------------------------------------------------------------

#: Maximum CATZOC value (Full Bottom Coverage).
CATZOC_MAX: int = 3

#: Human-readable labels for CATZOC values 0-3.
CATZOC_LABELS: List[str] = [
    "Gap / Void of Soundings",   # 0
    "Minimal Coverage",          # 1
    "Moderate Coverage",         # 2
    "Full Bottom Coverage",      # 3
]

# ---------------------------------------------------------------------------
# Input validation
# ---------------------------------------------------------------------------


def _validate_survey_params(
    risk_raster: np.ndarray,
    survey_raster: np.ndarray,
    start: Tuple[int, int],
    end: Tuple[int, int],
    safety_threshold: float,
    survey_weight: float,
    survey_nodata_as: str,
) -> None:
    """Validate survey-specific parameters before preprocessing."""
    if risk_raster.ndim != 2:
        raise ValueError("risk_raster must be a 2-D array")
    if survey_raster.ndim != 2:
        raise ValueError("survey_raster must be a 2-D array")
    if risk_raster.shape != survey_raster.shape:
        raise ValueError(
            f"risk_raster shape {risk_raster.shape} does not match "
            f"survey_raster shape {survey_raster.shape}. "
            "Ensure both rasters are snapped to the same grid."
        )
    if not 0.0 <= survey_weight <= 1.0:
        raise ValueError(
            f"survey_weight must be between 0.0 and 1.0, got {survey_weight}"
        )
    if safety_threshold < 0:
        raise ValueError(
            f"safety_threshold must be >= 0, got {safety_threshold}"
        )
    if survey_nodata_as not in ("unsurveyed", "surveyed"):
        raise ValueError(
            "survey_nodata_as must be 'unsurveyed' (treat as CATZOC 0) "
            "or 'surveyed' (treat as CATZOC 3), "
            f"got '{survey_nodata_as}'"
        )

    # Check that start/end are not masked by the safety threshold.
    sr, sc = start
    rows, cols = risk_raster.shape
    if not (0 <= sr < rows and 0 <= sc < cols):
        raise ValueError(
            f"Start point {start} is outside raster bounds ({rows}, {cols})"
        )
    er, ec = end
    if not (0 <= er < rows and 0 <= ec < cols):
        raise ValueError(
            f"End point {end} is outside raster bounds ({rows}, {cols})"
        )

    # Warn if start/end are already NaN before threshold masking
    if not np.isfinite(risk_raster[sr, sc]):
        raise ValueError(
            f"Start point {start} is on an invalid (NaN/Inf) cell in risk_raster"
        )
    if not np.isfinite(risk_raster[er, ec]):
        raise ValueError(
            f"End point {end} is on an invalid (NaN/Inf) cell in risk_raster"
        )

    # Check start/end are not blocked by safety threshold
    if risk_raster[sr, sc] > safety_threshold:
        raise ValueError(
            f"Start point {start} has risk value {risk_raster[sr, sc]:.1f} "
            f"which exceeds safety_threshold {safety_threshold}. "
            "Lower the threshold or choose a different start point."
        )
    if risk_raster[er, ec] > safety_threshold:
        raise ValueError(
            f"End point {end} has risk value {risk_raster[er, ec]:.1f} "
            f"which exceeds safety_threshold {safety_threshold}. "
            "Lower the threshold or choose a different end point."
        )


# ---------------------------------------------------------------------------
# Preprocessing
# ---------------------------------------------------------------------------


def build_composite_cost(
    risk_raster: np.ndarray,
    survey_raster: np.ndarray,
    safety_threshold: float,
    survey_weight: float,
    survey_nodata_as: str = "unsurveyed",
) -> Tuple[np.ndarray, int]:
    """Build the composite cost raster used as input to the LCP algorithm.

    Parameters
    ----------
    risk_raster : numpy.ndarray
        2-D array of navigation risk values (higher = more dangerous).
        NaN/Inf cells are treated as impassable barriers.
    survey_raster : numpy.ndarray
        2-D array of CATZOC coverage values (0–3).
        NaN cells are handled according to ``survey_nodata_as``.
    safety_threshold : float
        Risk cells above this value are masked to NaN (impassable).
    survey_weight : float
        Weight of the survey objective in [0.0, 1.0].
        * 0.0 → pure risk minimisation (identical to original ELCP).
        * 1.0 → pure survey-value maximisation (safety threshold still applies).
    survey_nodata_as : str
        How to handle NaN cells in ``survey_raster``.
        * ``"unsurveyed"`` (default) → treat as CATZOC 0 (highest priority).
        * ``"surveyed"``              → treat as CATZOC 3 (no survey value).

    Returns
    -------
    composite : numpy.ndarray
        Float64 composite cost array ready for the Dijkstra search.
        NaN cells are impassable.
    blocked_cells_count : int
        Number of cells masked by the safety threshold.
    """
    risk = np.array(risk_raster, dtype=np.float64)
    survey = np.array(survey_raster, dtype=np.float64)

    # --- Step 1: Safety masking -----------------------------------------------
    # Cells originally NaN/Inf are already barriers; additionally mask risk > threshold.
    originally_invalid = ~np.isfinite(risk)
    threshold_blocked = np.isfinite(risk) & (risk > safety_threshold)
    risk[threshold_blocked] = np.nan
    blocked_cells_count = int(np.sum(threshold_blocked))

    # --- Step 2: Handle CATZOC NaN values ------------------------------------
    survey_nan_mask = ~np.isfinite(survey)
    if survey_nan_mask.any():
        fill_value = 0.0 if survey_nodata_as == "unsurveyed" else float(CATZOC_MAX)
        survey[survey_nan_mask] = fill_value

    # Clip survey values to valid range [0, 3] in case input is noisy.
    survey = np.clip(survey, 0.0, float(CATZOC_MAX))

    # --- Step 3: Normalise risk -----------------------------------------------
    # Use only the finite (passable) cells for normalisation.
    finite_mask = np.isfinite(risk)
    risk_max = float(risk[finite_mask].max()) if finite_mask.any() else 1.0
    if risk_max == 0.0:
        risk_max = 1.0  # Avoid division by zero on a zero-cost raster.
    norm_risk = risk / risk_max  # NaN cells stay NaN

    # --- Step 4: Invert and normalise survey cost ----------------------------
    # survey_cost = 3 - survey  (0=Gap → cost=3, 3=Full → cost=0)
    # norm_survey_cost in [0, 1]: (3 - survey) / 3
    survey_cost = (CATZOC_MAX - survey) / float(CATZOC_MAX)

    # The "survey attractiveness" as a cost component is the complement:
    # lower norm_survey_cost → we want to pass through → subtract from composite.
    # Using (1 - survey_cost) gives us: Gap→1 (most attractive), Full→0 (least).
    # But we need cost semantics (low = prefer), so we use survey_cost directly:
    # Gap → survey_cost = 1.0 (lowest cost component), Full → 0.0 (highest cost).
    # Wait – this is inverted from what we want.
    #
    # We want:
    #   Gap (CATZOC 0)  → LOW composite cost  → path attracted here
    #   Full (CATZOC 3) → HIGH composite cost → path avoids here
    #
    # survey_cost = (3 - survey) / 3:
    #   CATZOC 0 → survey_cost = 1.0  (high – would raise composite cost)
    #   CATZOC 3 → survey_cost = 0.0  (low  – would lower composite cost)
    #
    # That is the WRONG direction. We need to invert again:
    #   survey_component = 1 - survey_cost = survey / 3
    #   CATZOC 0 → 0.0 (lowest cost → preferred)  ✓
    #   CATZOC 3 → 1.0 (highest cost → avoided)   ✓
    #
    # The formula from the design doc is:
    #   composite = (1 - w) * norm_risk + w * (1 - norm_survey_cost)
    # where norm_survey_cost = (3 - survey) / 3.
    # Substituting: (1 - norm_survey_cost) = 1 - (3-survey)/3 = survey/3
    # So this equals survey / CATZOC_MAX — confirmed correct direction.
    survey_component = survey / float(CATZOC_MAX)
    # Apply the NaN mask from the risk layer so barrier cells stay NaN.
    survey_component[~np.isfinite(risk)] = np.nan

    # --- Step 5: Composite cost -----------------------------------------------
    composite = (1.0 - survey_weight) * norm_risk + survey_weight * survey_component

    # Propagate NaN from original barriers (already done via norm_risk NaN).
    # Restore originally_invalid cells that might have been overwritten.
    composite[originally_invalid] = np.nan

    return composite, blocked_cells_count


# ---------------------------------------------------------------------------
# Survey statistics on the resulting path
# ---------------------------------------------------------------------------


def _compute_survey_stats(
    path: List[Tuple[int, int]],
    survey_raster: np.ndarray,
) -> Tuple[List[float], float]:
    """Compute survey coverage statistics along a grid path.

    Parameters
    ----------
    path : list of (row, col)
        Grid path cells.
    survey_raster : numpy.ndarray
        Original CATZOC raster (0–3, NaN allowed).

    Returns
    -------
    profile : list of float
        CATZOC value at each path cell (NaN where survey is missing).
    survey_score : float
        Fraction of path cells with CATZOC <= 1 (Gap or Minimal Coverage).
        Ranges from 0.0 (fully surveyed route) to 1.0 (entirely through gap areas).
    """
    profile: List[float] = []
    under_surveyed_count = 0
    valid_count = 0

    for r, c in path:
        val = float(survey_raster[r, c])
        profile.append(val)
        if np.isfinite(val):
            valid_count += 1
            if val <= 1.0:  # Gap (0) or Minimal Coverage (1)
                under_surveyed_count += 1

    survey_score = under_surveyed_count / valid_count if valid_count > 0 else 0.0
    return profile, survey_score


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def survey_aware_least_cost_path(
    risk_raster: np.ndarray,
    survey_raster: np.ndarray,
    start: Tuple[int, int],
    end: Tuple[int, int],
    safety_threshold: float = 100.0,
    survey_weight: float = 0.3,
    survey_nodata_as: str = "unsurveyed",
    curvature_factor: float = 0.0,
    max_turning_angle: float = 180.0,
    distance_factor: float = 0.0,
    straighten_factor: float = 0.3,
    cost_tolerance: float = 1.05,
    cell_size: Tuple[float, float] = (1.0, 1.0),
    progress_callback=None,
) -> Dict:
    """Compute a survey-aware least cost path for hydrographic survey vessels.

    Extends ``cost_aware_least_cost_path`` with two additional objectives:

    * **Safety constraint (hard)**: cells with risk > ``safety_threshold``
      are treated as impassable.
    * **Survey-value objective (soft)**: the path is attracted to cells with
      low CATZOC coverage (Gap / Void of Soundings) so that data can be
      collected during transit.

    Parameters
    ----------
    risk_raster : numpy.ndarray
        2-D navigation risk surface (higher values = more dangerous).
        NaN/Inf cells are unconditionally impassable.
    survey_raster : numpy.ndarray
        2-D CATZOC coverage raster, same shape and alignment as
        ``risk_raster``.  Values must be 0–3:

        * 0 – Gap / Void of Soundings  (highest survey priority)
        * 1 – Minimal Coverage
        * 2 – Moderate Coverage
        * 3 – Full Bottom Coverage     (lowest survey priority)

    start : tuple[int, int]
        ``(row, col)`` of the start cell (zero-based).
    end : tuple[int, int]
        ``(row, col)`` of the end cell (zero-based).
    safety_threshold : float, optional
        Risk values strictly above this are masked to NaN.  Default 100.
    survey_weight : float, optional
        Weight of the survey objective in the composite cost [0.0, 1.0].
        Default 0.3.
    survey_nodata_as : str, optional
        How NaN cells in ``survey_raster`` are handled.
        ``"unsurveyed"`` (default) treats them as CATZOC 0 (highest priority).
        ``"surveyed"`` treats them as CATZOC 3 (no survey value added).
    curvature_factor : float, optional
        Soft penalty for sharp turns (0.0 – 1.0). Default 0.0.
    max_turning_angle : float, optional
        Hard limit on turning angle in degrees (0 – 180). Default 180.
    distance_factor : float, optional
        Weight for path length (0.0 – 1.0). Default 0.0.
    straighten_factor : float, optional
        Lookahead fraction for cost-aware straightening (0.0 – 0.5).
        Default 0.3.
    cost_tolerance : float, optional
        Maximum shortcut-cost / original-cost ratio (>= 1.0). Default 1.05.
    cell_size : tuple[float, float], optional
        ``(y_size, x_size)`` of each raster cell. Default ``(1.0, 1.0)``.
    progress_callback : callable, optional
        Function accepting a single string for progress reporting.

    Returns
    -------
    dict
        All keys from ``cost_aware_least_cost_path``, plus:

        ``survey_coverage_profile`` : list of float
            CATZOC value at each cell of the raw grid path.
        ``survey_score`` : float
            Fraction of grid-path cells with CATZOC <= 1 (Gap or Minimal).
            0.0 = entirely through well-surveyed water;
            1.0 = entirely through unsurveyed / minimally surveyed water.
        ``blocked_cells_count`` : int
            Number of cells masked by ``safety_threshold``.
        ``composite_cost_raster`` : numpy.ndarray
            The composite cost array actually used by the Dijkstra search.
    """
    _validate_survey_params(
        risk_raster, survey_raster, start, end,
        safety_threshold, survey_weight, survey_nodata_as,
    )

    if progress_callback:
        rows, cols = risk_raster.shape
        progress_callback(
            f"[Survey-Aware] Parameters validated. "
            f"Raster: {rows}×{cols}, threshold: {safety_threshold}, "
            f"survey_weight: {survey_weight}"
        )

    # --- Preprocessing --------------------------------------------------------
    composite, blocked_cells_count = build_composite_cost(
        risk_raster, survey_raster, safety_threshold, survey_weight, survey_nodata_as,
    )

    if progress_callback:
        progress_callback(
            f"[Survey-Aware] Composite cost raster built. "
            f"Cells blocked by safety threshold: {blocked_cells_count}"
        )

    # --- Delegate to the core algorithm --------------------------------------
    result = cost_aware_least_cost_path(
        composite,
        start,
        end,
        curvature_factor=curvature_factor,
        max_turning_angle=max_turning_angle,
        distance_factor=distance_factor,
        straighten_factor=straighten_factor,
        cost_tolerance=cost_tolerance,
        cell_size=cell_size,
        progress_callback=progress_callback,
    )

    # --- Compute survey statistics on the returned path ----------------------
    profile, survey_score = _compute_survey_stats(result["path"], survey_raster)

    result["survey_coverage_profile"] = profile
    result["survey_score"] = survey_score
    result["blocked_cells_count"] = blocked_cells_count
    result["composite_cost_raster"] = composite

    if progress_callback:
        progress_callback(
            f"[Survey-Aware] Done. "
            f"Survey score: {survey_score:.2%} of path through gap/minimal-coverage areas."
        )

    return result
