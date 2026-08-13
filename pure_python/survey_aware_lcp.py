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

1. **Safety penalty (soft)**: cells whose risk value exceeds
   ``safety_threshold`` receive a linear cost multiplier so that the path
   strongly prefers to avoid them while still being able to pass through
   when no safer alternative exists.  The multiplier is::

       penalty_factor = 1 + penalty_multiplier * (risk - threshold) / threshold

   Cells that are originally NaN/Inf remain fully impassable (hard barrier).
2. **Survey-value objective (soft)**: cells with low CATZOC coverage
   (Gap / void-of-soundings) are cheaper to traverse, encouraging the
   path to collect data in under-surveyed areas.

The composite cost formula is::

    composite = (1 - survey_weight) * norm_risk_penalised
              + survey_weight       * (1 - norm_survey_cost)

where::

    norm_risk_penalised = risk_penalised / max(finite penalised risk values)
    norm_survey_cost    = (3 - survey_raster) / 3   # inverted and normalised

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
    ``cost_aware_least_cost_path`` plus the four new parameters.

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
    penalty_multiplier: float = 10.0,
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
    if penalty_multiplier < 0:
        raise ValueError(
            f"penalty_multiplier must be >= 0, got {penalty_multiplier}"
        )
    if survey_nodata_as not in ("unsurveyed", "surveyed"):
        raise ValueError(
            "survey_nodata_as must be 'unsurveyed' (treat as CATZOC 0) "
            "or 'surveyed' (treat as CATZOC 3), "
            f"got '{survey_nodata_as}'"
        )

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

    # Only hard barriers (NaN/Inf) make a cell truly impassable.
    if not np.isfinite(risk_raster[sr, sc]):
        raise ValueError(
            f"Start point {start} is on an invalid (NaN/Inf) cell in risk_raster"
        )
    if not np.isfinite(risk_raster[er, ec]):
        raise ValueError(
            f"End point {end} is on an invalid (NaN/Inf) cell in risk_raster"
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
    penalty_multiplier: float = 10.0,
    max_avoidance_level: int = CATZOC_MAX,
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
        Risk value above which a linear cost penalty is applied.
        Cells are *not* masked to NaN; instead their composite cost
        is multiplied by ``1 + penalty_multiplier * excess / threshold``,
        where ``excess = risk - safety_threshold``.
        Only the original NaN/Inf cells remain truly impassable.
    survey_weight : float
        Weight of the survey objective in [0.0, 1.0].
        * 0.0 → pure risk minimisation (identical to original ELCP).
        * 1.0 → pure survey-value maximisation.
    survey_nodata_as : str
        How to handle NaN cells in ``survey_raster``.
        * ``"unsurveyed"`` (default) → treat as CATZOC 0 (highest priority).
        * ``"surveyed"``              → treat as CATZOC 3 (no survey value).
    penalty_multiplier : float
        Controls how strongly cells above ``safety_threshold`` are penalised.
        A value of 10.0 (default) means a cell whose risk is exactly
        2× the threshold receives 11× the baseline composite cost.
        Set to 0.0 to disable the penalty entirely.
    max_avoidance_level : int
        Maximum CATZOC level at which the survey objective starts penalising
        cells.  Cells with CATZOC >= ``max_avoidance_level`` are treated as
        fully surveyed (maximum survey component cost).  Cells below this level
        are penalised proportionally.

        * 3 (default) – avoid Class A only (CATZOC 3).
        * 2 – avoid Class A and Class B (CATZOC 2–3).
        * 1 – avoid Class A, B, and C (CATZOC 1–3).

        The effective survey component formula is::

            survey_component = min(survey, max_avoidance_level) / max_avoidance_level

    Returns
    -------
    composite : numpy.ndarray
        Float64 composite cost array ready for the Dijkstra search.
        NaN cells are impassable.
    above_threshold_count : int
        Number of cells whose risk exceeds ``safety_threshold``
        (informational; these cells are penalised but still passable).
    """
    risk = np.array(risk_raster, dtype=np.float64)
    survey = np.array(survey_raster, dtype=np.float64)

    # Validate max_avoidance_level.
    if not (1 <= int(max_avoidance_level) <= CATZOC_MAX):
        raise ValueError(
            f"max_avoidance_level must be between 1 and {CATZOC_MAX}, "
            f"got {max_avoidance_level}"
        )

    # --- Step 1: Identify cells that are intrinsically invalid ----------------
    # Only NaN/Inf cells are true hard barriers; threshold excess is soft.
    originally_invalid = ~np.isfinite(risk)

    # --- Step 2: Apply linear penalty for cells above safety_threshold --------
    above_threshold_mask = np.isfinite(risk) & (risk > safety_threshold)
    above_threshold_count = int(np.sum(above_threshold_mask))
    if above_threshold_count > 0 and penalty_multiplier > 0.0 and safety_threshold > 0.0:
        excess = risk[above_threshold_mask] - safety_threshold
        penalty_factor = 1.0 + penalty_multiplier * excess / safety_threshold
        risk[above_threshold_mask] *= penalty_factor

    # --- Step 3: Handle CATZOC NaN values ------------------------------------
    survey_nan_mask = ~np.isfinite(survey)
    if survey_nan_mask.any():
        fill_value = 0.0 if survey_nodata_as == "unsurveyed" else float(CATZOC_MAX)
        survey[survey_nan_mask] = fill_value

    # Clip survey values to valid range [0, 3] in case input is noisy.
    survey = np.clip(survey, 0.0, float(CATZOC_MAX))

    # Apply max_avoidance_level: cells with CATZOC >= max_avoidance_level are
    # clamped to max_avoidance_level so they all receive maximum survey cost.
    effective_max = max(1, int(max_avoidance_level))
    survey = np.minimum(survey, float(effective_max))

    # --- Step 4: Normalise risk -----------------------------------------------
    # Use only the finite (passable) cells for normalisation.
    finite_mask = np.isfinite(risk)
    risk_max = float(risk[finite_mask].max()) if finite_mask.any() else 1.0
    if risk_max == 0.0:
        risk_max = 1.0  # Avoid division by zero on a zero-cost raster.
    norm_risk = risk / risk_max  # NaN cells stay NaN

    # --- Step 5: Compute survey component -----------------------------------
    # survey_component = survey / effective_max:
    #   CATZOC 0 → 0.0 (lowest cost → preferred)
    #   CATZOC >= effective_max → 1.0 (highest cost → avoided)
    survey_component = survey / float(effective_max)
    # Apply the NaN mask from the risk layer so barrier cells stay NaN.
    survey_component[~np.isfinite(risk)] = np.nan

    # --- Step 6: Composite cost -----------------------------------------------
    composite = (1.0 - survey_weight) * norm_risk + survey_weight * survey_component

    # Propagate NaN from original barriers (already done via norm_risk NaN).
    # Restore originally_invalid cells that might have been overwritten.
    composite[originally_invalid] = np.nan

    return composite, above_threshold_count


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
    penalty_multiplier: float = 10.0,
    max_avoidance_level: int = CATZOC_MAX,
    corridor_mask: Optional[np.ndarray] = None,
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

    * **Safety penalty (soft)**: cells with risk > ``safety_threshold``
      receive a linearly scaled cost multiplier so that the path strongly
      prefers safer alternatives while still being able to cross high-risk
      areas when no other route exists.
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
        * 1 – Minimal Coverage / Class C
        * 2 – Moderate Coverage / Class B
        * 3 – Full Bottom Coverage / Class A (lowest survey priority)

    start : tuple[int, int]
        ``(row, col)`` of the start cell (zero-based).
    end : tuple[int, int]
        ``(row, col)`` of the end cell (zero-based).
    safety_threshold : float, optional
        Risk value above which a linear penalty is applied.  Cells are
        *not* masked to NaN; the path avoids them via higher cost.
        Default 100.
    survey_weight : float, optional
        Weight of the survey objective in the composite cost [0.0, 1.0].
        Default 0.3.
    survey_nodata_as : str, optional
        How NaN cells in ``survey_raster`` are handled.
        ``"unsurveyed"`` (default) treats them as CATZOC 0 (highest priority).
        ``"surveyed"`` treats them as CATZOC 3 (no survey value added).
    penalty_multiplier : float, optional
        Strength of the linear cost penalty for cells above
        ``safety_threshold``.  The effective cost multiplier for a cell is::

            1 + penalty_multiplier * (risk - threshold) / threshold

        Default 10.0.  Set to 0.0 to disable the penalty.
    max_avoidance_level : int, optional
        Maximum CATZOC level at which the survey objective starts penalising
        cells.  Cells with CATZOC >= ``max_avoidance_level`` are treated as
        fully surveyed (maximum survey component cost).  Default 3 (avoid
        Class A only).  Set to 2 to also avoid Class B; set to 1 to avoid
        Class C, B, and A.
    corridor_mask : numpy.ndarray of bool, optional
        2-D boolean array, same shape as ``risk_raster``.  Where
        ``corridor_mask`` is ``False`` the cell is treated as impassable
        (NaN) so that the path is constrained to stay inside the corridor.
        ``None`` (default) means no corridor constraint.
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
        ``above_threshold_count`` : int
            Number of cells whose risk exceeds ``safety_threshold``
            (these cells are penalised but still passable).
        ``composite_cost_raster`` : numpy.ndarray
            The composite cost array actually used by the Dijkstra search.
    """
    _validate_survey_params(
        risk_raster, survey_raster, start, end,
        safety_threshold, survey_weight, survey_nodata_as, penalty_multiplier,
    )

    # --- Apply corridor mask (cells outside → impassable) ---------------------
    if corridor_mask is not None:
        risk_raster = np.array(risk_raster, dtype=np.float64)
        risk_raster[~corridor_mask] = np.nan

    if progress_callback:
        rows, cols = risk_raster.shape
        corridor_info = (
            f", corridor: {int(np.sum(corridor_mask))} passable cells"
            if corridor_mask is not None else ""
        )
        progress_callback(
            f"[Survey-Aware] Parameters validated. "
            f"Raster: {rows}×{cols}, threshold: {safety_threshold}, "
            f"survey_weight: {survey_weight}"
            f"{corridor_info}"
        )

    # --- Preprocessing --------------------------------------------------------
    composite, above_threshold_count = build_composite_cost(
        risk_raster, survey_raster, safety_threshold, survey_weight,
        survey_nodata_as, penalty_multiplier, max_avoidance_level,
    )

    if progress_callback:
        progress_callback(
            f"[Survey-Aware] Composite cost raster built. "
            f"Cells above safety threshold (penalised): {above_threshold_count}"
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
    result["above_threshold_count"] = above_threshold_count
    result["composite_cost_raster"] = composite

    if progress_callback:
        progress_callback(
            f"[Survey-Aware] Done. "
            f"Survey score: {survey_score:.2%} of path through gap/minimal-coverage areas."
        )

    return result
