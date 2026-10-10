"""CPU referee for receipt-infra lever R1 (2026-10-10): the heavy CUDA test
runners register only the kernels they launch. `grid::init_grid<T>()` calls the
header's aggregate `init_grid_kernel_attrs`, which address-takes EVERY kernel
(bench-only `_single_timing` twins included) — and whoever address-takes a
kernel compiles it. A mimic robot's cells share one full header, so a single
aggregate call made each cell compile the whole robot (h1_2 fixed flagship:
4.5 h of prewarm). These runners must stay on `init_grid_streams` + the
generated per-algorithm `init_grid_kernel_attr_<algo>_checked` twins."""
from __future__ import annotations

import re
from pathlib import Path

import pytest

RUNNERS = Path(__file__).resolve().parent / "cuda_equivalents"

# runner -> the per-algorithm registrations it must make (its launched kernels)
CONVERTED = {
    "cuda_equivalence_runner.cu": {
        "inverse_dynamics", "minv", "forward_dynamics", "aba", "crba",
        "end_effector_pose", "end_effector_pose_gradient", "end_effector_pose_hessian",
        "inverse_dynamics_gradient", "forward_dynamics_gradient"},
    "cuda_second_order_smoke_runner.cu": {"idsva_so_body_frame", "fdsva_so"},
    "cuda_integrator_smoke_runner.cu": {"integrator", "integrator_gradient", "integrator_with_gradient"},
    "cuda_plant_smoke_runner.cu": {"integrator", "integrator_gradient"},
    "cuda_centroidal_smoke_runner.cu": set(),
    "cuda_centroidal_mimic_smoke_runner.cu": set(),
}
_REG = re.compile(r"init_grid_kernel_attr_([a-z_0-9]+)_checked<T>\(\)")
_HOST_CALL = re.compile(r"grid::([a-z_0-9]+)<T(?:,[^<>()]*)?>\(")


def _code(path: Path) -> str:
    """Source minus `//` comment lines (comments may quote the forbidden call)."""
    return "\n".join(line for line in path.read_text().splitlines()
                     if not line.lstrip().startswith("//"))


@pytest.mark.parametrize("runner", sorted(CONVERTED))
def test_runner_registers_only_what_it_launches(runner):
    text = _code(RUNNERS / runner)
    assert "grid::init_grid<T>()" not in text, f"{runner}: aggregate registration is back"
    assert "grid::init_grid_streams<T>()" in text, f"{runner}: streams-only init missing"
    registered = set(_REG.findall(text))
    assert registered == CONVERTED[runner], (runner, registered)
    # every generated host wrapper the runner calls is covered by a registration
    # (kernels launched directly are `<name>_kernel`; runner-local kernels are not grid::)
    launched = {m for m in _HOST_CALL.findall(text)
                if not m.startswith(("init_", "close_", "free_", "release_", "grid_"))
                and not m.endswith("_device")}
    launched = {m[:-len("_kernel")] if m.endswith("_kernel") else m for m in launched}
    assert launched <= registered, (runner, launched - registered)


def test_every_registration_twin_is_a_manifest_algorithm():
    """A registration must name a real per-algorithm twin (the manifest's short
    names are what kernel_attrs.py emits `init_grid_kernel_attr_<short>_checked` for)."""
    from grid_codegen import kernel_attrs
    shorts = {entry[1] for entry in kernel_attrs.KERNEL_ATTR_MANIFEST}
    for runner, expected in CONVERTED.items():
        assert expected <= shorts, (runner, expected - shorts)
