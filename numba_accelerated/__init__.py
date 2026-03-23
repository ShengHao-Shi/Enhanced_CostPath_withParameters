"""
Numba-Accelerated LCP Algorithm implementations.

This package contains Numba JIT-compiled versions of the LCP algorithms
for significantly improved performance on large rasters.

* ``cost_aware_straighten_lcp`` — Numba-accelerated 8-dir Dijkstra +
  cost-aware straightening (20–50× faster than pure Python on large grids)
"""
