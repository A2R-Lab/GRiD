"""Algorithm-list / profile closure contracts (audit W07, 2026-09-19).

`normalize_codegen_algorithms` turns a requested algorithm list or profile name
into the set the generator actually emits. Two defects were found by the
2026-09-19 audit and are pinned here:

* a bare ``algorithm_list=["forward_dynamics"]`` emitted a header that nvcc
  rejected (``minv_inner`` / ``inverse_dynamics_inner`` undefined) — the
  dependency rules had no row for forward_dynamics itself;
* ``algorithm_list=["dynamics-core"]`` raised — the single-pass token rewrite
  turned the hyphenated PROFILE key into ``dynamics_core`` and then missed it.

The last test is the general net: for every registry algorithm, the SINGLETON
header must define every ``*_inner`` device function it calls (a static scan,
no nvcc). CPU-only; iiwa14-fixed from config/robot_assets.
"""
from __future__ import annotations

import contextlib
import os
import re
from pathlib import Path

import pytest

from URDFParser import URDFParser
from grid_codegen import GRiDCodeGenerator
from grid_codegen._algo_profiles import normalize_codegen_algorithms
from grid_codegen.algo_registry import ALGO_DESCRIPTORS

REPO = Path(__file__).resolve().parent.parent
URDF = REPO / "config" / "robot_assets" / "iiwa14.urdf"


def _gen():
    with open(os.devnull, "w") as devnull, contextlib.redirect_stdout(devnull):
        robot = URDFParser().parse(str(URDF), floating_base=False)
        return GRiDCodeGenerator(robot, DEBUG_MODE=False, NEED_PRINT_MAT=False, FILE_NAMESPACE="grid")


def _closure(gen, algorithm_list=None, codegen_profile="all"):
    return set(normalize_codegen_algorithms(gen, codegen_profile=codegen_profile,
                                            algorithm_list=algorithm_list))


def test_forward_dynamics_singleton_pulls_its_composed_inners():
    got = _closure(_gen(), ["forward_dynamics"])
    assert {"forward_dynamics", "inverse_dynamics", "minv"} <= got, got


@pytest.mark.parametrize("spelling", ["dynamics-core", "dynamics_core", "DYNAMICS-CORE", " dynamics-core "])
def test_profile_names_resolve_in_every_spelling(spelling):
    gen = _gen()
    via_list = _closure(gen, [spelling])
    via_profile = _closure(gen, None, codegen_profile="dynamics-core")
    assert via_list == via_profile and via_list, (spelling, via_list)


def test_aliases_still_resolve():
    gen = _gen()
    assert _closure(gen, ["all-dynamics"]) == _closure(gen, None, "dynamics")
    assert _closure(gen, ["kinematics-only"]) == _closure(gen, None, "kinematics")


def test_closure_is_idempotent_and_order_independent():
    gen = _gen()
    req = ["fdsva_so", "end_effector_pose_gradient", "forward_dynamics"]
    a = _closure(gen, req)
    b = _closure(gen, list(reversed(req)))
    c = _closure(gen, sorted(a))
    assert a == b == c


# Device-function families a subset header composes: the per-algorithm
# `*_inner` bodies AND the shared `load_update_*_helpers` loaders (XImats,
# XmatsHom, ...) — a contact-frame subset without kinematics once referenced
# load_update_XmatsHom_helpers that only the kinematics block emitted.
# A definition is `name(<params>) {` (any return type — init_topology_helpers
# returns int*); a call is `name<...>(` or `name(` anywhere. defs ⊆ calls textually.
_DEF = re.compile(r"\b([a-z0-9_]+(?:_inner|_helpers))\s*\([^;{}]*\)\s*\{", re.S)
_CALL = re.compile(r"\b([a-z0-9_]+(?:_inner|_helpers))\s*(?:<[^;{}]*?>)?\s*\(")


_COMMENTS = re.compile(r"/\*.*?\*/|//[^\n]*", re.S)


def _undefined_inners(header_text: str) -> set[str]:
    code = _COMMENTS.sub("", header_text)   # the interface doc block names every inner
    defs = set(_DEF.findall(code))
    calls = set(_CALL.findall(code))
    return calls - defs


# Registry keys that are NOT requestable through algorithm_list: they are opt-in
# through their own gen_all_code input (multi_target_batch=, collision_spec=,
# contact_frames=) or are a composite host layer (plant). Requesting one by
# name must raise the "Unknown GRiD algorithm selection" ValueError — pinned
# below so a key that becomes requestable is moved into the closure net.
NOT_REQUESTABLE = {
    "collision", "f_ext_contact", "multi_target_position",
    "multi_target_position_gradient", "plant",
    # variant / dispatcher keys: requested through their parent algorithm
    # (f_ext_gradient emits both f_ext_gradient + f_ext_gradient_dq; idsva_so is
    # the body/world-frame dispatcher over idsva_so_body_frame/_world_frame).
    "f_ext_gradient_dq", "idsva_so",
}


@pytest.mark.parametrize("key", sorted(d.key for d in ALGO_DESCRIPTORS))
def test_singleton_header_defines_every_inner_it_calls(key, tmp_path):
    """Static closure net: a singleton build must not reference an `*_inner`
    device function it did not emit (the exact FD-only nvcc failure class)."""
    gen = _gen()
    out = tmp_path / f"{key}.cuh"
    with open(os.devnull, "w") as devnull, contextlib.redirect_stdout(devnull):
        if key in NOT_REQUESTABLE:
            with pytest.raises(ValueError, match="Unknown GRiD algorithm selection"):
                gen.gen_all_code(algorithm_list=[key], output_path=str(out), enable_mujoco_kernels=False)
            return
        gen.gen_all_code(algorithm_list=[key], output_path=str(out), enable_mujoco_kernels=False)
    text = out.read_text()
    assert not _undefined_inners(text), f"{key}: calls undefined inners {sorted(_undefined_inners(text))}"


def test_contact_frames_subset_without_kinematics_is_self_contained(tmp_path):
    """contact_frames= is an opt-in input, not an algorithm: a dynamics-only
    subset build with contact frames must still emit the whole contact section
    AND every helper it composes (audit W07/W08 2026-09-19: it used to sit
    inside the kinematics block and silently vanished; once hoisted it still
    referenced the kinematics-only XmatsHom loader)."""
    from grid_codegen.algorithms._f_ext_contact import contact_frames_from_urdf
    gen = _gen()
    frames = contact_frames_from_urdf(gen.robot, ["iiwa_joint_ee", "tool0_joint"])
    out = tmp_path / "contact_subset.cuh"
    with open(os.devnull, "w") as devnull, contextlib.redirect_stdout(devnull):
        gen.gen_all_code(algorithm_list=["inverse_dynamics", "forward_dynamics"], contact_frames=frames,
                         output_path=str(out), enable_mujoco_kernels=False)
    text = out.read_text()
    assert "#define GRID_HAS_CONTACT_FRAMES 1" in text
    assert "const int NUM_CONTACT_FRAMES = 2;" in text
    assert not _undefined_inners(text), sorted(_undefined_inners(text))


# --------------------------------------------------------------------------
# mjx twins (floating, non-mimic robots emit a MuJoCo-convention twin of every
# twinned kernel). The twins compose extra inners of their own (the ID-gradient
# twin's epilogue rebuilds M via crba_inner), so the closure must hold with the
# twins ON as well — found 2026-09-19 by a go2 subset build that failed in ptxas
# with "Unresolved extern function grid::crba_inner".
# --------------------------------------------------------------------------
GO2 = REPO / "config" / "robot_assets" / "go2.urdf"


def _gen_go2_floating():
    with open(os.devnull, "w") as devnull, contextlib.redirect_stdout(devnull):
        robot = URDFParser().parse(str(GO2), floating_base=True)
        return GRiDCodeGenerator(robot, DEBUG_MODE=False, NEED_PRINT_MAT=False, FILE_NAMESPACE="grid")


def _twinned_keys():
    from grid_codegen.abi_specs import ABI_SPECS
    return sorted(k for k, s in ABI_SPECS.items()
                  if getattr(s, "sig_mjx_macro", None) and k not in NOT_REQUESTABLE
                  and any(d.key == k for d in ALGO_DESCRIPTORS))


@pytest.mark.parametrize("key", _twinned_keys())
def test_singleton_with_mjx_twins_defines_every_inner_it_calls(key, tmp_path):
    if not GO2.exists():
        pytest.skip("go2 asset missing")
    gen = _gen_go2_floating()
    out = tmp_path / f"{key}_mjx.cuh"
    with open(os.devnull, "w") as devnull, contextlib.redirect_stdout(devnull):
        gen.gen_all_code(algorithm_list=[key], output_path=str(out), enable_mujoco_kernels=True)
    text = out.read_text()
    assert not _undefined_inners(text), f"{key}+mjx: calls undefined inners {sorted(_undefined_inners(text))}"
