"""
Survey-Aware A* Least Cost Path
=================================

This module is the A* counterpart of ``pure_python.survey_aware_lcp``.
It exposes an identical public API — ``survey_aware_astar_least_cost_path`` —
but delegates the graph search to the A* implementation in
``pure_python.astar_lcp`` instead of the Dijkstra-based
``pure_python.cost_aware_straighten_lcp``.

All composite-cost preprocessing (safety penalty + CATZOC survey weight),
input validation, corridor handling, and survey-statistics computation are
re-used directly from ``pure_python.survey_aware_lcp``.

See ``pure_python.survey_aware_lcp`` for full parameter and return-value
documentation.
"""

from typing import Dict, Optional, Tuple

import numpy as np

from pure_python.astar_lcp import cost_aware_astar_least_cost_path
from pure_python.survey_aware_lcp import (
    CATZOC_MAX,
    _apply_corridor_mask,
    _compute_survey_stats,
    _validate_survey_params,
    build_composite_cost,
)
from pure_python.cost_aware_straighten_lcp import _check_connectivity


def survey_aware_astar_least_cost_path(
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
    """Compute a survey-aware least cost path using A* search.

    Drop-in replacement for
    ``pure_python.survey_aware_lcp.survey_aware_least_cost_path``.
    Uses A* for the graph search instead of Dijkstra; all other
    parameters, preprocessing steps, and return values are identical.

    Parameters
    ----------
    risk_raster : numpy.ndarray
        2-D navigation risk surface (higher values = more dangerous).
        NaN/Inf cells are unconditionally impassable.
    survey_raster : numpy.ndarray
        2-D CATZOC coverage raster, same shape as ``risk_raster``.
        Values 0–3 (0 = Gap / unsurveyed, 3 = Full coverage).
    start : tuple[int, int]
        ``(row, col)`` of the start cell.
    end : tuple[int, int]
        ``(row, col)`` of the end cell.
    safety_threshold : float, optional
        Risk value above which a linear penalty is applied. Default 100.
    survey_weight : float, optional
        Weight of the survey objective [0.0, 1.0]. Default 0.3.
    survey_nodata_as : str, optional
        How NaN cells in ``survey_raster`` are treated.
        ``"unsurveyed"`` (default) or ``"surveyed"``.
    penalty_multiplier : float, optional
        Strength of the safety penalty. Default 10.0.
    max_avoidance_level : int, optional
        Maximum CATZOC level at which the survey objective penalises
        cells (1–3). Default 3.
    corridor_mask : numpy.ndarray of bool, optional
        2-D boolean mask; ``False`` cells are treated as impassable.
    curvature_factor : float, optional
        Soft turn-penalty weight (0.0 – 1.0). Default 0.0.
    max_turning_angle : float, optional
        Hard limit on turning angle in degrees. Default 180.
    distance_factor : float, optional
        Weight for path length (0.0 – 1.0). Default 0.0.
    straighten_factor : float, optional
        Lookahead fraction for cost-aware straightening. Default 0.3.
    cost_tolerance : float, optional
        Maximum shortcut-cost / original-cost ratio. Default 1.05.
    cell_size : tuple[float, float], optional
        ``(y_size, x_size)`` of each raster cell. Default ``(1.0, 1.0)``.
    progress_callback : callable, optional
        Function accepting a single string for progress reporting.

    Returns
    -------
    dict
        All keys from ``cost_aware_astar_least_cost_path``, plus:

        ``survey_coverage_profile`` : list of float
            CATZOC value at each cell of the raw grid path.
        ``survey_score`` : float
            Fraction of grid-path cells with CATZOC <= 1.
        ``above_threshold_count`` : int
            Number of cells whose risk exceeds ``safety_threshold``.
        ``composite_cost_raster`` : numpy.ndarray
            The composite cost array used by the A* search.
    """
    _validate_survey_params(
        risk_raster, survey_raster, start, end,
        safety_threshold, survey_weight, survey_nodata_as, penalty_multiplier,
    )

    # Apply corridor mask (cells outside → impassable).
    if corridor_mask is not None:
        risk_raster = _apply_corridor_mask(risk_raster, corridor_mask, start, end)

    if progress_callback:
        rows, cols = risk_raster.shape
        corridor_info = (
            f", corridor: {int(np.sum(corridor_mask))} passable cells"
            if corridor_mask is not None else ""
        )
        progress_callback(
            f"[Survey-Aware A*] Parameters validated. "
            f"Raster: {rows}×{cols}, threshold: {safety_threshold}, "
            f"survey_weight: {survey_weight}"
            f"{corridor_info}"
        )

    # Build composite cost raster (same preprocessing as Dijkstra version).
    composite, above_threshold_count = build_composite_cost(
        risk_raster, survey_raster, safety_threshold, survey_weight,
        survey_nodata_as, penalty_multiplier, max_avoidance_level,
    )

    if progress_callback:
        progress_callback(
            f"[Survey-Aware A*] Composite cost raster built. "
            f"Cells above safety threshold (penalised): {above_threshold_count}"
        )

    # Pre-check connectivity before starting the expensive A* search.
    # Raises ValueError immediately if start/end are in disconnected regions.
    _check_connectivity(composite, start, end, progress_callback)

    # Delegate to the A* core algorithm.
    result = cost_aware_astar_least_cost_path(
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

    # Compute survey statistics on the returned path.
    profile, survey_score = _compute_survey_stats(result["path"], survey_raster)

    result["survey_coverage_profile"] = profile
    result["survey_score"] = survey_score
    result["above_threshold_count"] = above_threshold_count
    result["composite_cost_raster"] = composite

    if progress_callback:
        progress_callback(
            f"[Survey-Aware A*] Done. "
            f"Survey score: {survey_score:.2%} of path through gap/minimal-coverage areas."
        )

    return result
