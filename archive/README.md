# Archive

This directory contains earlier algorithm variants that have been superseded
by the **Cost-Aware Straightened LCP** implementation now found in
`pure_python/cost_aware_straighten_lcp.py` and
`numba_accelerated/cost_aware_straighten_lcp.py`.

These files are retained for reference and reproducibility but are **not
actively maintained**.

---

## Contents

| File / Folder | Description | Reason archived |
|---|---|---|
| `enhanced_lcp.py` | Backward-compat root wrapper for Approach A | Superseded by Approach C |
| `theta_star_lcp.py` | Backward-compat root wrapper for Approach B (Theta*) | Superseded by Approach C |
| `cost_aware_straighten_lcp.py` | Backward-compat root wrapper for Approach C | Wrapper no longer needed; use `pure_python.*` directly |
| `enhanced_lcp_high_straightness.py` | Experimental high-straightness variant wrapper | Experimental; not recommended for production |
| `arcgis_toolbox.pyt` | Original ArcGIS toolbox (wraps Approaches A, B, C) | Replaced by `arcgis_toolbox_with_progress.pyt` in repo root |
| `pure_python/enhanced_lcp.py` | Approach A — 8-dir Dijkstra + LOS straightening | Straightening does not check cost, can silently cross expensive cells |
| `pure_python/enhanced_lcp_high_straightness.py` | Approach A variant with higher straightness penalty | Experimental; not recommended |
| `pure_python/theta_star_lcp.py` | Approach B — Theta* any-angle pathfinding | Any-angle search without post-processing cost control |
| `tests/test_enhanced_lcp.py` | Tests for Approach A | Tests for archived module |
| `tests/test_theta_star_lcp.py` | Tests for Approach B | Tests for archived module |

---

## Algorithm Evolution

```
Approach A  enhanced_lcp.py
  8-dir Dijkstra
  + curvature penalty
  + hard turn constraint
  + distance penalty
  + LOS straightening (NODATA-only check)
  ↓ Problem: straightening may silently cross expensive cells

Approach B  theta_star_lcp.py
  Theta* any-angle search
  + line-of-sight baked into the search
  ↓ Problem: no post-search cost tolerance control

Approach C  pure_python/cost_aware_straighten_lcp.py  ← CURRENT RECOMMENDED
  8-dir Dijkstra (same as A)
  + cost-aware straightening (compares shortcut cost vs original path cost)
  + cost_tolerance parameter controls acceptable cost overhead
  + Chaikin smoothing with NODATA safety check
  + Numba-accelerated variant for large rasters (20–50× faster)
```
