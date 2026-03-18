# Backward-compatible wrapper — delegates to pure_python.theta_star_lcp
from pure_python.theta_star_lcp import *           # noqa: F401,F403
from pure_python.theta_star_lcp import (           # noqa: F401
    _supercover_line,
    _is_line_clear,
    _line_cost,
    _vector_angle,
    _smooth_path_nodata_safe,
)
