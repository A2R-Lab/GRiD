"""CPU referee for the release CUDA host-call bridge (test/benchmarks/release/kernel_bridge.cu).

test/test_runner_registration.py covers only test/cuda_equivalents/cuda_*_runner.cu;
this applies the same rule to the bridge: streams-only init plus per-algorithm
`init_grid_kernel_attr_<algo>_checked` registration of every operation it can
launch, never the aggregate `grid::init_grid<T>()`, and an operation table that
agrees with grid_adapter.KERNEL_OPS."""
from __future__ import annotations
import re
from pathlib import Path

from test.benchmarks.release.grid_adapter import KERNEL_OPS

BRIDGE = Path(__file__).resolve().parent / "release" / "kernel_bridge.cu"
# operation -> the registration twins its host launcher dispatches to
REGISTRATIONS = {op: {op} for op in KERNEL_OPS}
REGISTRATIONS["idsva_so"] = {"idsva_so_body_frame", "idsva_so_world_frame"}
OP_NAMES = {"inverse_dynamics": "OP_ID", "inverse_dynamics_gradient": "OP_ID_GRAD", "idsva_so": "OP_IDSVA_SO",
            "minv": "OP_MINV", "forward_dynamics": "OP_FD", "forward_dynamics_gradient": "OP_FD_GRAD",
            "fdsva_so": "OP_FDSVA_SO", "end_effector_pose": "OP_EE_POSE", "crba": "OP_CRBA",
            "nonlinear_effects": "OP_NLE", "generalized_gravity": "OP_GRAVITY", "ccrba": "OP_CCRBA",
            "coriolis_matrix": "OP_CORIOLIS", "end_effector_pose_gradient": "OP_EE_GRAD"}


def _code():
    return "\n".join(line for line in BRIDGE.read_text().splitlines() if not line.lstrip().startswith("//"))


def test_bridge_uses_streams_only_init_and_per_algorithm_registration():
    text = _code()
    assert "grid::init_grid<T>()" not in text, "aggregate registration is back"
    assert "grid::init_grid_streams<T>()" in text
    registered = set(re.findall(r"init_grid_kernel_attr_([a-z_0-9]+)_checked<T>\(", text))
    assert registered == set().union(*REGISTRATIONS.values()), registered
    from grid_codegen import kernel_attrs
    shorts = {entry[1] for entry in kernel_attrs.KERNEL_ATTR_MANIFEST}
    assert registered <= shorts, registered - shorts


def test_bridge_operation_codes_match_the_adapter_table():
    text = _code()
    codes = dict(re.findall(r"\b(OP_[A-Z_]+) = (\d+)", text))
    assert {OP_NAMES[op]: str(i) for op, i in KERNEL_OPS.items()} == codes
    for op, name in OP_NAMES.items():
        # every operation has a registration case, a with-memory launch, a compute-only launch and both output buffers
        assert len(re.findall(rf"case {name}:", text)) >= 6, (op, len(re.findall(rf"case {name}:", text)))
    assert "h_end_effector_pose_gradient" in text and "d_end_effector_pose_gradient" in text
    assert "KB_CAT(GRID_RBD_EE_POSE_GRADIENT_FN, _compute_only)" in text
    assert 'grid_kernel_create(int op)' in text
