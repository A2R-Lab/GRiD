"""Interim mjx-guard test: while the codegen mjx fusion is in progress, the binding
must RAISE on a floating-base derivative / second-order / convention-sensitive method
in ``output_convention="mujoco"`` rather than silently returning a PINOCCHIO-frame
result (the dangerous caveat). The value methods (id/fd/aba/crba/minv) DO transform.

These run with no compiled .so: a bare handle is built via ``__new__`` with just the
metadata the guard reads (``_meta['floating_base']`` + ``_output_convention``), and
each guarded method is invoked with dummy args — the guard is the first statement, so
it fires before any Runner access. When an algorithm's mjx fusion lands, drop its
guard AND its name here (the test then proves the method no longer raises).
"""
import inspect

import pytest

from grid_rbd._handle import RobotHandle


# Convention-sensitive methods that must be guarded until their mjx transform lands.
# (Invariant methods — com, energy, osc_inertia, fk_batched, end_effector_pose value,
#  the regressors, the joint barriers — are NOT here: mjx mode is already correct for
#  them, so guarding would wrongly block valid calls.)
_MUST_GUARD = {
    "inverse_dynamics_gradient", "forward_dynamics_gradient", "idsva_so", "fdsva_so",
    "end_effector_pose_gradient", "end_effector_pose_hessian",
    "integrator", "integrator_gradient",
    "plant_step", "plant_step_gradient", "plant_step_hessian",
    "generalized_gravity", "nonlinear_effects", "coriolis_matrix",
    "frame_jacobian", "frame_jacobian_dot", "ccrba", "dccrba", "cmm_time_variation",
    "ee_pos_cost", "com_cost", "momentum_cost",
}
# Value methods that MUST NOT be guarded (mjx mode transforms them correctly).
_MUST_NOT_GUARD = {"inverse_dynamics", "forward_dynamics", "aba", "crba", "minv"}


def _bare_handle(floating: bool, convention: str) -> RobotHandle:
    h = RobotHandle.__new__(RobotHandle)
    h._meta = {"floating_base": floating}
    h._output_convention = convention
    return h


def _dummy_args(method):
    """Required positional args (excluding self) as None — the guard fires first."""
    sig = inspect.signature(method)
    return [None for p in sig.parameters.values()
            if p.kind in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD)
            and p.default is inspect._empty and p.name != "self"]


@pytest.mark.parametrize("name", sorted(_MUST_GUARD))
def test_guarded_method_raises_in_floating_mujoco_mode(name):
    h = _bare_handle(floating=True, convention="mujoco")
    method = getattr(h, name)
    with pytest.raises(NotImplementedError, match="mujoco"):
        method(*_dummy_args(method))


@pytest.mark.parametrize("name", sorted(_MUST_GUARD))
def test_guard_is_noop_in_pinocchio_and_fixed_base(name):
    # pinocchio convention: the guard must not fire (floating + pin)
    h_pin = _bare_handle(floating=True, convention="pinocchio")
    assert h_pin._mjx_guard_unsupported(name) is None
    # fixed base in mujoco mode: no free-flyer, guard is a no-op
    h_fixed = _bare_handle(floating=False, convention="mujoco")
    assert h_fixed._mjx_guard_unsupported(name) is None


def test_value_methods_are_not_guarded():
    """The supported value methods must NOT call the guard (would block valid mjx)."""
    for name in _MUST_NOT_GUARD:
        src = inspect.getsource(getattr(RobotHandle, name))
        assert "_mjx_guard_unsupported" not in src, (
            f"{name} is a supported mjx value method and must not be guarded")
