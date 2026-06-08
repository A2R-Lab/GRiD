"""Tests for the thread-safe ``handle.mujoco`` view + the per-call output
convention override (no compiled .so needed — a bare handle exercises the
convention-resolution + forwarding logic directly).

The design contract: ``handle.mujoco.<method>(...)`` applies the mjx convention
PER CALL by forwarding an explicit ``_convention="mujoco"``; it never mutates the
handle's shared ``output_convention``, so concurrent pinocchio- and mujoco-mode
calls on the same handle cannot clobber each other.
"""
import threading

import pytest

from grid_rbd._handle import RobotHandle, _MujocoView


def _bare_handle(floating=True, convention="pinocchio"):
    h = RobotHandle.__new__(RobotHandle)
    h._meta = {"floating_base": floating}
    h._output_convention = convention
    return h


def test_per_call_override_does_not_mutate_shared_state():
    h = _bare_handle(floating=True, convention="pinocchio")
    assert h._mjx_active() is False                  # handle default = pinocchio
    assert h._mjx_active("mujoco") is True            # explicit per-call override
    assert h._output_convention == "pinocchio"        # override did NOT mutate the handle


def test_fixed_base_override_is_noop():
    h = _bare_handle(floating=False, convention="pinocchio")
    assert h._mjx_active("mujoco") is False            # no free-flyer ⇒ mjx is a no-op


def test_resolve_convention_validates():
    h = _bare_handle()
    assert h._resolve_convention() == "pinocchio"
    assert h._resolve_convention("mujoco") == "mujoco"
    with pytest.raises(ValueError):
        h._resolve_convention("bogus")


def test_mujoco_view_is_cached_and_exposes_value_methods():
    h = _bare_handle()
    v = h.mujoco
    assert isinstance(v, _MujocoView)
    assert h.mujoco is v                              # cached
    for m in ("inverse_dynamics", "forward_dynamics", "aba", "crba", "minv"):
        assert hasattr(v, m)
    # guarded surfaces are deliberately NOT on the view
    for m in ("inverse_dynamics_gradient", "idsva_so", "integrator"):
        assert not hasattr(v, m)


def test_view_forwards_mujoco_convention():
    h = _bare_handle()
    seen = {}

    def fake_id(q, qd, qdd=None, *, gravity=-9.81, f_ext=None, _convention=None):
        seen["conv"] = _convention
        return "tau"

    h.inverse_dynamics = fake_id
    assert h.mujoco.inverse_dynamics(1, 2, 3) == "tau"
    assert seen["conv"] == "mujoco"
    assert h._output_convention == "pinocchio"        # view never mutates the handle


def test_concurrent_pin_and_mujoco_calls_do_not_clobber():
    """A pinocchio-mode call and a mujoco-view call running concurrently on the same
    handle each observe their own convention (the override is per-call, not shared)."""
    h = _bare_handle(floating=True, convention="pinocchio")
    results = {}
    barrier = threading.Barrier(2)

    def record(tag, _convention):
        # both threads resolve at the same time; per-call override must be independent
        barrier.wait()
        results[tag] = h._mjx_active(_convention)

    t_pin = threading.Thread(target=record, args=("pin", None))
    t_mjx = threading.Thread(target=record, args=("mjx", "mujoco"))
    t_pin.start(); t_mjx.start(); t_pin.join(); t_mjx.join()
    assert results == {"pin": False, "mjx": True}
    assert h._output_convention == "pinocchio"
