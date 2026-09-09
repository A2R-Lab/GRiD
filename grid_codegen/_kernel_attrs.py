# BACK-COMPAT ALIAS (de-underscore rename, 2026-09-09): now
# grid_codegen.kernel_attrs. Remove in the Wave D receipt window.
import sys
from . import kernel_attrs as _real
sys.modules[__name__] = _real
