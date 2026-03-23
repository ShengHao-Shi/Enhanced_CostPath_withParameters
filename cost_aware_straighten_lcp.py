# Backward-compatible wrapper — delegates to pure_python.cost_aware_straighten_lcp
from pure_python.cost_aware_straighten_lcp import *           # noqa: F401,F403
from pure_python.cost_aware_straighten_lcp import (           # noqa: F401
    _supercover_line,
    _is_line_clear,
    _line_cost,
    _line_cost_or_inf,
    _smooth_path_nodata_safe,
)
