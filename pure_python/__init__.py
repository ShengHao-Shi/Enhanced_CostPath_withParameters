"""
Pure-Python LCP algorithm implementations.

This package contains the recommended pure-Python implementation of the
Cost-Aware Straightened LCP algorithm (Approach C):

* ``cost_aware_straighten_lcp``  — 8-dir Dijkstra + cost-aware
  straightening + Chaikin smoothing.  This is the primary algorithm.
* ``survey_aware_lcp``           — Survey-vessel extension (Dijkstra);
  adds CATZOC objective and safety penalty on top of the core LCP.
* ``astar_lcp``                  — Drop-in A* replacement for
  ``cost_aware_straighten_lcp``; uses an admissible consistent heuristic
  so fewer nodes are typically expanded on open rasters.
* ``astar_survey_aware_lcp``     — Survey-vessel extension using A*;
  identical parameters and outputs to ``survey_aware_lcp``.

Earlier variants (Approaches A and B) have been moved to ``archive/``.
"""
