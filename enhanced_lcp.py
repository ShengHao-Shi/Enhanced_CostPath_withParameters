# Backward-compatible wrapper — delegates to pure_python.enhanced_lcp
from pure_python.enhanced_lcp import *           # noqa: F401,F403
from pure_python.enhanced_lcp import (           # noqa: F401
    _supercover_line,
    _is_line_clear,
    _smooth_path_nodata_safe,
)
