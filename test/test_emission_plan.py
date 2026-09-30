"""gen_all_code's option resolution is a pure, generation-free plan
(`_resolve_emission_plan`, hygiene 5/9, 2026-09-24). These assert the plan
directly — the §7.z15 class of bug (a feature gate derived wrongly) becomes a
CPU assertion instead of a red CUDA cell. The closure/subset tests remain the
referee for what a plan actually builds; this only pins the derivation."""
from __future__ import annotations

import contextlib
import io
from pathlib import Path

import pytest

from grid_codegen import GRiDCodeGenerator
from grid_codegen.algorithms._f_ext_contact import contact_frames_from_urdf
from URDFParser import URDFParser

REPO = Path(__file__).resolve().parents[1]


def _gen(name="iiwa14", floating=False):
    with contextlib.redirect_stdout(io.StringIO()):
        robot = URDFParser().parse(str(REPO / "config" / "robot_assets" / f"{name}.urdf"), floating_base=floating)
    return GRiDCodeGenerator(robot, DEBUG_MODE=False, NEED_PRINT_MAT=False, FILE_NAMESPACE="grid")


def _plan(gen, **kw):
    before = gen.code_str
    with contextlib.redirect_stdout(io.StringIO()):
        algorithms, any_kin, hom = gen._resolve_emission_plan(**kw)
    assert gen.code_str == before, "resolution must emit nothing"
    return set(algorithms), any_kin, hom


def test_bare_forward_dynamics_pulls_its_composed_inners():
    algos, any_kin, hom = _plan(_gen(), algorithm_list=["forward_dynamics"], enable_mujoco_kernels=False)
    assert {"forward_dynamics", "inverse_dynamics", "minv"} <= algos
    assert any_kin is False


def test_contact_frames_imply_homogenous_transforms_without_kinematics():
    gen = _gen()
    contacts = contact_frames_from_urdf(gen.robot, ["iiwa_joint_ee"])
    _, any_kin, hom = _plan(gen, algorithm_list=["inverse_dynamics"], contact_frames=contacts, enable_mujoco_kernels=False)
    assert any_kin is False and hom is True


def test_profile_and_algorithm_spellings_are_distinct():
    gen = _gen()
    prof, any_kin_p, _ = _plan(gen, algorithm_list=["frame-jacobian"], enable_mujoco_kernels=False)
    alg, any_kin_a, _ = _plan(gen, algorithm_list=["frame_jacobian"], enable_mujoco_kernels=False)
    assert "frame_jacobian" in alg and len(prof) > len(alg), (sorted(prof), sorted(alg))
    assert any_kin_p and any_kin_a


def test_floating_mjx_second_order_adds_crba():
    gen = _gen("go2", floating=True)
    algos, _, _ = _plan(gen, algorithm_list=["inverse_dynamics_gradient"], enable_mujoco_kernels=True)
    assert "crba" in algos
    algos_pin, _, _ = _plan(gen, algorithm_list=["inverse_dynamics_gradient"], enable_mujoco_kernels=False)
    assert "crba" not in algos_pin
