"""Registration-argument forwarding (audit W08, 2026-09-19).

`grid_rbd.register_robot(..., backend="jax"|"torch")` delegates to the backend
`register_robot`, which in turn delegates to the NumPy root registration. Every
material build option must survive BOTH hops — the 2026-09-17 `contact_frames`
kwarg reached the NumPy path only and was silently DROPPED on the jax/torch
dispatch (the backend signatures lacked it). These tests are CPU-only: the
delegate is replaced by a capturing stub that raises before any build.
"""
from __future__ import annotations

import pytest

import grid_rbd


class _Captured(Exception):
    def __init__(self, kwargs):
        super().__init__("captured")
        self.kwargs = kwargs


def _stub(*args, **kwargs):
    raise _Captured(kwargs)


# Every option a caller may pass at the root that MUST reach the backend build.
_MATERIAL = dict(
    floating_base=True, ee_joint_names=["j7"], max_batch_size=64,
    algorithm_list=["inverse_dynamics"], use_joint_dynamics=True,
    runtime_joint_dynamics=True, runtime_inertia=True, runtime_transform=True,
    enable_tool=True, contact_frames=["foot_fl", "foot_fr"],
    enable_mujoco_kernels=False, dtype="float32",
)


@pytest.mark.parametrize("backend", ["jax", "torch"])
def test_root_dispatch_forwards_every_material_option(monkeypatch, backend):
    mod = pytest.importorskip(f"grid_rbd.{backend}")
    monkeypatch.setattr(mod, "register_robot", _stub)
    with pytest.raises(_Captured) as ei:
        grid_rbd.register_robot("fwd_probe", "/nonexistent/robot.urdf", backend=backend, **_MATERIAL)
    got = ei.value.kwargs
    missing = {k: v for k, v in _MATERIAL.items() if got.get(k) != v}
    assert not missing, f"root->{backend} dispatch dropped/changed: {missing}"


@pytest.mark.parametrize("backend", ["jax", "torch"])
def test_backend_register_forwards_to_the_numpy_root(monkeypatch, backend):
    mod = pytest.importorskip(f"grid_rbd.{backend}")
    if backend == "jax":
        pytest.importorskip("jax")
    else:
        pytest.importorskip("torch")
    # The backend calls `_grid_rbd.register_robot` (the package attribute).
    monkeypatch.setattr(grid_rbd, "register_robot", _stub)
    with pytest.raises(_Captured) as ei:
        mod.register_robot("fwd_probe", "/nonexistent/robot.urdf", **_MATERIAL)
    got = ei.value.kwargs
    missing = {k: v for k, v in _MATERIAL.items() if got.get(k) != v}
    assert not missing, f"{backend}.register_robot dropped/changed on the way to the root: {missing}"
