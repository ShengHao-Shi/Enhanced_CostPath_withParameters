"""
Survey-Aware LCP – Numba-Accelerated Variant
=============================================

Drop-in replacement for ``pure_python.survey_aware_lcp``.

The preprocessing (composite cost construction) and survey statistics are
pure NumPy, so no Numba compilation overhead is incurred for those steps.
The heavy Dijkstra search is delegated to
``numba_accelerated.cost_aware_straighten_lcp.cost_aware_least_cost_path``,
which provides 20–50× speedup on large rasters.

The public API is identical to the pure-Python version::

    from numba_accelerated.survey_aware_lcp import survey_aware_least_cost_path

Dependencies: numpy, numba.  Falls back with a clear error if numba is
not installed.
"""

from typing import Dict, List, Optional, Tuple

import numpy as np

# Re-use all preprocessing and validation helpers from the pure-Python module.
from pure_python.survey_aware_lcp import (
    CATZOC_MAX,
    CATZOC_LABELS,
    _validate_survey_params,
    build_composite_cost,
    _compute_survey_stats,
)


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
    """Compute a survey-aware LCP using the Numba-accelerated Dijkstra search.

    Identical to ``pure_python.survey_aware_lcp.survey_aware_least_cost_path``
    in both API and output format, but uses the JIT-compiled Dijkstra from
    ``numba_accelerated.cost_aware_straighten_lcp`` for large-raster performance.

    Note: the first call incurs ~6 s of Numba JIT compilation overhead.
    Subsequent calls are 20–50× faster than the pure-Python variant.

    Parameters
    ----------
    risk_raster : numpy.ndarray
        2-D navigation risk surface (higher = more dangerous).
    survey_raster : numpy.ndarray
        2-D CATZOC coverage raster (0–3), same shape as ``risk_raster``.
    start : tuple[int, int]
        ``(row, col)`` of the start cell.
    end : tuple[int, int]
        ``(row, col)`` of the end cell.
    safety_threshold : float, optional
        Risk value above which a linear penalty is applied.  Cells are
        *not* masked to NaN; the path avoids them via higher cost.
        Default 100.
    survey_weight : float, optional
        Survey-objective weight in [0.0, 1.0]. Default 0.3.
    survey_nodata_as : str, optional
        How NaN survey cells are handled: ``"unsurveyed"`` or ``"surveyed"``.
        Default ``"unsurveyed"``.
    penalty_multiplier : float, optional
        Strength of the linear cost penalty for cells above
        ``safety_threshold``.  Default 10.0.  Set to 0.0 to disable.
    max_avoidance_level : int, optional
        Maximum CATZOC level at which the survey objective penalises cells.
        Cells with CATZOC >= ``max_avoidance_level`` receive the maximum
        survey component cost.  Default 3 (avoid Class A only).  Set to 2
        to also avoid Class B; set to 1 to avoid Class C, B, and A.
    corridor_mask : numpy.ndarray of bool, optional
        2-D boolean array, same shape as ``risk_raster``.  Where ``False``
        the cell is treated as impassable (NaN) so that the path is
        constrained to stay inside the corridor.  ``None`` (default) means
        no corridor constraint.
    curvature_factor : float, optional
        Soft turn penalty (0.0–1.0). Default 0.0.
    max_turning_angle : float, optional
        Hard turn limit in degrees. Default 180.
    distance_factor : float, optional
        Path-length weight (0.0–1.0). Default 0.0.
    straighten_factor : float, optional
        Straightening lookahead (0.0–0.5). Default 0.3.
    cost_tolerance : float, optional
        Shortcut cost tolerance (>= 1.0). Default 1.05.
    cell_size : tuple[float, float], optional
        ``(y_size, x_size)`` per cell. Default ``(1.0, 1.0)``.
    progress_callback : callable, optional
        String progress reporter.

    Returns
    -------
    dict
        Same keys as ``pure_python.survey_aware_lcp.survey_aware_least_cost_path``.
    """
    # Import here so the module still loads even if numba is missing.
    try:
        import numba_accelerated.cost_aware_straighten_lcp as _numba_mod
    except ImportError as exc:
        raise ImportError(
            "numba_accelerated package requires numba. "
            "Install it with: pip install numba"
        ) from exc

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
            f"[Survey-Aware / Numba] Parameters validated. "
            f"Raster: {rows}×{cols}, threshold: {safety_threshold}, "
            f"survey_weight: {survey_weight}"
            f"{corridor_info}"
        )

    # --- Preprocessing (pure NumPy – no Numba needed) -------------------------
    composite, above_threshold_count = build_composite_cost(
        risk_raster, survey_raster, safety_threshold, survey_weight,
        survey_nodata_as, penalty_multiplier, max_avoidance_level,
    )

    if progress_callback:
        progress_callback(
            f"[Survey-Aware / Numba] Composite cost raster built. "
            f"Cells above safety threshold (penalised): {above_threshold_count}"
        )

    # --- Numba-accelerated Dijkstra search ------------------------------------
    result = _numba_mod.cost_aware_least_cost_path(
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

    # --- Survey statistics ----------------------------------------------------
    profile, survey_score = _compute_survey_stats(result["path"], survey_raster)

    result["survey_coverage_profile"] = profile
    result["survey_score"] = survey_score
    result["above_threshold_count"] = above_threshold_count
    result["composite_cost_raster"] = composite

    if progress_callback:
        progress_callback(
            f"[Survey-Aware / Numba] Done. "
            f"Survey score: {survey_score:.2%} of path through gap/minimal-coverage areas."
        )

    return result
