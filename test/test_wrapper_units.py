"""bindings/grid_rbd/_units.py: fence grammar, unit assembly, partition referee."""
from __future__ import annotations
import pytest
from pathlib import Path
from grid_rbd import _units

TEMPLATE = Path(__file__).resolve().parents[1] / "bindings" / "grid_rbd" / "wrapper_template.cu"

SYNTH = """\
#include "grid.cuh"
int core_a();
// [[unit: prelude; surface: jax]]
#define JAX_HELPER 1
// [[unit: end]]
// [[unit: fam_x]]
int fam_x_cabi();
// [[unit: end]]
int core_b();
// [[unit: fam_x; surface: jax]]
int fam_x_jax();
// [[unit: end]]
// [[unit: fam_y; surface: torch]]
int fam_y_torch();
// [[unit: end]]
// [[unit: prelude; surface: torch]]
#define TORCH_HELPER 1
// [[unit: end]]
int core_c();
"""


def test_split_synthetic_template():
    units = _units.split(SYNTH)
    assert set(units) == {"core", "fam_x", "fam_y"}
    core = units["core"]
    assert "core_a" in core and "core_b" in core and "core_c" in core
    assert "fam_x" not in core and "JAX_HELPER" not in core and "TORCH_HELPER" not in core
    x = units["fam_x"]
    assert x.index('#include "wrapper_common.cuh"') < x.index("fam_x_cabi")
    assert "JAX_HELPER" in x and "TORCH_HELPER" not in x
    assert x.index("JAX_HELPER") < x.index("fam_x_jax")
    assert "#if defined(GRID_RBD_WITH_JAX)" in x and "GRID_RBD_WITH_TORCH" not in x
    y = units["fam_y"]
    assert "TORCH_HELPER" in y and "JAX_HELPER" not in y and "#if defined(GRID_RBD_WITH_TORCH)" in y


def test_partition_referee_counts_every_line():
    counts = _units.check_partition(SYNTH)
    assert counts["core"] == 5 and counts["fam_x"] == 2 and counts["fam_y"] == 1  # core incl. the trailing empty line
    assert counts["prelude:jax"] == 1 and counts["prelude:torch"] == 1


@pytest.mark.parametrize("bad", [
    "// [[unit: end]]\nint x;\n",
    "// [[unit: a]]\n// [[unit: b]]\nint x;\n// [[unit: end]]\n",
    "// [[unit: a]]\nint x;\n",
    "// [[unit: prelude]]\nint x;\n// [[unit: end]]\n",
])
def test_malformed_fences_are_refused(bad):
    with pytest.raises(_units.FenceError):
        _units.check_partition(bad)


def test_real_template_partitions_cleanly():
    """The checked-in template's fences are well-formed and lose no line."""
    counts = _units.check_partition(TEMPLATE.read_text())
    assert counts["core"] > 0


def test_core_unit_references_no_kernel():
    """The structural guarantee of the split: the core unit launches nothing,
    address-takes no kernel (registration, ceiling, torch impl table) and
    binds no JAX handler — every such reference lives in a family unit."""
    core = _units.split(TEMPLATE.read_text())["core"]
    for needle in ("<<<", "init_grid_kernel_attr_", "GRID_KERNEL_CEIL(", "m.impl(",
                   "XLA_FFI_DEFINE_HANDLER_SYMBOL(", "TORCH_LIBRARY_FRAGMENT(GRID_RBD_TORCH_LIB",
                   "TORCH_LIBRARY_IMPL(GRID_RBD_TORCH_LIB"):
        assert needle not in core, f"core unit still contains {needle!r}"
    import re
    assert not re.search(r"grid::[a-z_0-9]+_kernel<", core), "core unit instantiates a kernel"
    assert "GRID_RBD_TORCH_LIBRARY(GRID_RBD_TORCH_LIB, m)" in core


def test_every_family_unit_is_self_describing():
    units = _units.split(TEMPLATE.read_text())
    from grid_codegen.abi_specs import FAMILIES
    assert set(units) == {"core", *FAMILIES}
    for fam in FAMILIES:
        u = units[fam]
        assert u.startswith(f"// wrapper unit `{fam}`")
        assert '#include "wrapper_common.cuh"' in u
        assert f"grid_rbd_family_{fam}_init_attrs(const char **failed_op) {{" in u
        assert f"grid_rbd_family_{fam}_kernel_ceiling(const char* algo) {{" in u
