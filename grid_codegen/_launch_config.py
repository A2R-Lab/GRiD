# BACK-COMPAT ALIAS (de-underscore rename, 2026-09-09): the module is now
# grid_codegen.launch_config. sys.modules aliasing makes `import
# grid_codegen._launch_config` yield the REAL module object, so monkeypatching
# through the old name still affects internal lookups (the fingerprinted
# test_batch_switch.py does exactly that). Remove in the Wave D receipt window
# along with that test's import.
import sys
from . import launch_config as _real
sys.modules[__name__] = _real
