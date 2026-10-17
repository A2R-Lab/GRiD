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
    "cuda_integrator_batch_forces_runner.cu": {"integrator", "integrator_gradient", "integrator_with_gradient"},
    "cuda_equivalence_runner.cu": {
        "inverse_dynamics", "minv", "forward_dynamics", "aba", "crba",
        "end_effector_pose", "end_effector_pose_gradient", "end_effector_pose_hessian",
        "inverse_dynamics_gradient", "forward_dynamics_gradient"},
    "cuda_second_order_smoke_runner.cu": {"idsva_so_body_frame", "fdsva_so"},
    "cuda_integrator_smoke_runner.cu": {"integrator", "integrator_gradient", "integrator_with_gradient"},
    "cuda_plant_smoke_runner.cu": {"integrator", "integrator_gradient"},
    "cuda_centroidal_smoke_runner.cu": set(),
    "cuda_centroidal_mimic_smoke_runner.cu": set(),
    # 2026-10-12 sweep (the remaining runners)
    "cuda_batched_host_runner.cu": {"minv", "crba"},
    "cuda_energy_regressor_smoke_runner.cu": {"kinetic_energy_regressor", "potential_energy_regressor"},
    "cuda_spherical_integrator_runner.cu": {"integrator"},
    "cuda_branching_skew_runner.cu": {"inverse_dynamics", "minv", "aba", "crba"},
    "cuda_coriolis_smoke_runner.cu": {"coriolis_matrix"},
    "cuda_spherical_kinematics_runner.cu": {"end_effector_pose", "frame_jacobian"},
    "cuda_frame_jacobian_host_runner.cu": {"frame_jacobian", "frame_jacobian_dot", "osc_inertia"},
    "cuda_regressor_smoke_runner.cu": {"inverse_dynamics_regressor"},
    "cuda_workspace_slots_runner.cu": {"idsva_so_world_frame", "idsva_so_body_frame", "fdsva_so", "f_ext_gradient_dq"},
    "cuda_spherical_eepose_grad_hess_runner.cu": {"end_effector_pose_gradient", "end_effector_pose_hessian"},
    "cuda_ee_named_depth1_runner.cu": set(),
    "cuda_idsva_so_world_frame_smoke_runner.cu": {"idsva_so_world_frame"},
    "cuda_centroidal_host_spill_runner.cu": {"com", "ccrba", "energy"},
    "cuda_fd_parameter_gradient_smoke_runner.cu": {"forward_dynamics_parameter_gradient"},
    "cuda_spherical_runner.cu": {"inverse_dynamics", "crba", "minv", "forward_dynamics", "aba",
                                 "inverse_dynamics_gradient", "forward_dynamics_gradient", "fdsva_so"},
    "cuda_eepose_runtime_smoke_runner.cu": set(),
    "cuda_spherical_fdsva_so_runner.cu": {"fdsva_so"},
    "cuda_dccrba_smoke_runner.cu": {"cmm_time_variation", "dccrba"},
    "cuda_frame_jacobian_smoke_runner.cu": set(),
    "cuda_f_ext_gradient_runner.cu": {"f_ext_gradient", "f_ext_gradient_dq"},
    "cuda_regressor_gradient_runner.cu": {"inverse_dynamics_regressor_gradient"},
    "cuda_spherical_integrator_gradient_runner.cu": {"integrator_gradient", "integrator_with_gradient"},
    "cuda_spherical_so_runner.cu": {"idsva_so_world_frame", "idsva_so_body_frame"},
}
# a launched host wrapper that dispatches to frame-specific kernels
_DISPATCH = {"idsva_so": {"idsva_so_world_frame", "idsva_so_body_frame"}}
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
    launched = set().union(*(_DISPATCH.get(m, {m}) for m in launched)) if launched else set()
    assert launched <= registered, (runner, launched - registered)


def test_no_runner_calls_the_aggregate_registration():
    """Every runner in the directory, listed here or not: the aggregate is banned."""
    offenders = [p.name for p in sorted(RUNNERS.glob("cuda_*_runner.cu"))
                 if "grid::init_grid<T>()" in _code(p)]
    assert not offenders, offenders
    unlisted = {p.name for p in RUNNERS.glob("cuda_*_runner.cu")
                if "grid::init_grid_streams<T>()" in _code(p)} - set(CONVERTED)
    assert not unlisted, f"runners using streams-only init but missing from CONVERTED: {sorted(unlisted)}"


def test_every_registration_twin_is_a_manifest_algorithm():
    """A registration must name a real per-algorithm twin (the manifest's short
    names are what kernel_attrs.py emits `init_grid_kernel_attr_<short>_checked` for)."""
    from grid_codegen import kernel_attrs
    shorts = {entry[1] for entry in kernel_attrs.KERNEL_ATTR_MANIFEST}
    for runner, expected in CONVERTED.items():
        assert expected <= shorts, (runner, expected - shorts)
