"""'Adding one algorithm' completion invariants (audit handoff §8, 2026-09-22).

The dY/dx and contact_fext integration tails showed that a new kernel has a
long tail of registries that must agree. The per-registry parity tests already
exist (descriptor parity, ABI cross-check, feature-macro coverage, profile
closure); this module pins the CROSS-registry joins that had no referee:

  registry key  <->  AlgoDescriptor            (kernel attrs / launch config)
  registry key  <->  PER_ALGO_SPECS bench row   (or a named, documented exception)
  ABI op        <->  a Python method on at least one backend (explicit alias map)
  ABI op        <->  a mention in the python_wrappers tutorial (capability docs)
  registry key  <->  a mention somewhere in docs/source

CPU-only, no codegen run. When you add an algorithm and one of these fails,
the fix is to complete the tail, or to add the key to the relevant exception
set WITH the reason — never to widen a set silently.
"""
from __future__ import annotations

import glob
import importlib.util
import sys
from pathlib import Path

import pytest

from grid_codegen.algo_registry import ALGO_DESCRIPTORS, ALGO_REGISTRY
from grid_codegen.abi_specs import ABI_SPECS

REPO = Path(__file__).resolve().parents[1]
DOCS = REPO / "docs" / "source"
TUTORIAL = DOCS / "user_guide" / "tutorials" / "python_wrappers.rst"

# Registry algorithms with NO per-algo bench row, each with the reason.
BENCH_EXEMPT = {
    "collision": "collision-geometry family is timed by its own harness (native vs config_free A/B), not the per-algo bench",
    "f_ext_contact": "contact f_ext is a device helper consumed inside ID/FD launches; timed through those rows",
    "integrator_hessian": "plant_step_hessian surface; second-order integrator timing is covered by the SO rows + plant bench",
    "plant": "grid_plant is a composition layer over the timed kernels (cost/barrier/step); no kernel of its own",
}
# Bench rows that are twins of a registry key (mjx convention variants), not algorithms.
BENCH_TWIN_SUFFIXES = ("_mjx",)

# ABI ops whose Python spelling differs from the C-ABI key, or that are
# internal to a differentiable path (consumed by another method's backward).
ABI_PY_ALIAS = {
    "f_ext_contact": "contact_fext",
    "f_ext_gradient": None,                 # internal: *_gradient(f_ext=...) backward path
    "f_ext_gradient_dq": None,              # internal: same
    "forward_dynamics_parameter_gradient": "forward_dynamics_parameter_gradient",  # jax/torch only
    "forward_dynamics_wrt_params": "forward_dynamics_wrt_params",                  # jax/torch only
    "inverse_dynamics_wrt_params": "inverse_dynamics_wrt_params",                  # jax/torch only
    "plant_com_cost": "com_cost",
    "plant_ee_pos_cost": "ee_pos_cost",
    "plant_joint_position_barrier": "joint_position_barrier",
    "plant_joint_torque_barrier": "joint_torque_barrier",
    "plant_joint_velocity_barrier": "joint_velocity_barrier",
    "plant_momentum_cost": "momentum_cost",
    "plant_quadratic_input_cost": "quadratic_input_cost",
    "plant_quadratic_state_cost": "quadratic_state_cost",
}


def _bench_specs():
    spec = importlib.util.spec_from_file_location(
        "grid_bench_run", REPO / "test" / "benchmarks" / "baselines" / "grid" / "run.py")
    sys.path.insert(0, str(REPO / "test" / "benchmarks" / "baselines" / "grid"))
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return set(m.PER_ALGO_SPECS)


def _backend_method_names():
    from grid_rbd._handle import RobotHandle
    names = {a for a in dir(RobotHandle) if not a.startswith("_")}
    for mod in ("grid_rbd.jax", "grid_rbd.torch"):
        try:
            m = __import__(mod, fromlist=["_"])
        except Exception:
            continue
        for cls_name in ("JaxRobotHandle", "TorchRobotHandle"):
            cls = getattr(m, cls_name, None)
            if cls is not None:
                names |= {a for a in dir(cls) if not a.startswith("_")}
        # fall back: any class in the module with a forward_dynamics attr
        for v in vars(m).values():
            if isinstance(v, type) and hasattr(v, "forward_dynamics"):
                names |= {a for a in dir(v) if not a.startswith("_")}
    return names


REGISTRY = {e.key for e in ALGO_REGISTRY}


def test_every_registry_key_has_a_descriptor_and_vice_versa():
    desc = {d.key for d in ALGO_DESCRIPTORS}
    assert REGISTRY == desc, (sorted(REGISTRY - desc), sorted(desc - REGISTRY))


def test_every_registry_key_has_a_bench_row_or_a_named_exemption():
    bench = _bench_specs()
    missing = REGISTRY - bench - set(BENCH_EXEMPT)
    assert not missing, f"registry algorithms without a PER_ALGO_SPECS row (add the row or a BENCH_EXEMPT reason): {sorted(missing)}"
    stale = set(BENCH_EXEMPT) & bench
    assert not stale, f"BENCH_EXEMPT entries that now HAVE a bench row (drop the exemption): {sorted(stale)}"
    unknown = {b for b in bench - REGISTRY if not b.endswith(BENCH_TWIN_SUFFIXES)}
    assert not unknown, f"bench rows naming no registry algorithm: {sorted(unknown)}"
    twins_without_base = {b for b in bench - REGISTRY if b.endswith(BENCH_TWIN_SUFFIXES)
                          and b.rsplit("_", 1)[0] not in REGISTRY}
    assert not twins_without_base, sorted(twins_without_base)


def test_every_abi_op_is_reachable_from_python_or_declared_internal():
    names = _backend_method_names()
    unmapped = []
    for key in ABI_SPECS:
        py = ABI_PY_ALIAS.get(key, key)
        if py is None:
            continue  # declared internal
        if py not in names:
            unmapped.append((key, py))
    assert not unmapped, f"ABI ops with no Python method on any backend (add the method or an ABI_PY_ALIAS entry): {unmapped}"
    stale = [k for k in ABI_PY_ALIAS if k not in ABI_SPECS]
    assert not stale, f"ABI_PY_ALIAS names ops that no longer exist: {stale}"


def test_every_public_abi_op_is_documented_in_the_tutorial():
    doc = TUTORIAL.read_text()
    missing = []
    for key in ABI_SPECS:
        py = ABI_PY_ALIAS.get(key, key)
        if py is None:
            continue
        if f"``{py}(" not in doc and f"``{py}``" not in doc and f"{py}(" not in doc:
            missing.append(py)
    assert not missing, f"public methods absent from the python_wrappers tutorial: {missing}"


def test_every_registry_algorithm_is_mentioned_in_the_docs():
    text = "\n".join(Path(f).read_text() for f in glob.glob(str(DOCS / "**" / "*.rst"), recursive=True))
    missing = sorted(k for k in REGISTRY if k not in text)
    assert not missing, f"registry algorithms never mentioned under docs/source: {missing}"


# ─── R1 (2026-09-23): the tutorial's VJP table must MATCH VjpSpec, not just mention names ──

import re as _re


def _tutorial_vjp_rows():
    """Parse the 'What is differentiable' list-table: {method: (inputs, notes)}."""
    doc = TUTORIAL.read_text()
    start = doc.index("**What is differentiable.**")
    end = doc.index("Each of these is a **reverse-mode VJP only**")
    block = doc[start:end]
    rows = {}
    for m in _re.finditer(r"\* - (.+?)\n\s+- (.+?)\n\s+- (.+?)(?=\n\s+\* - |\Z)", block, _re.S):
        names = _re.findall(r"``([a-z_]+)``", m.group(1))
        inputs = tuple(_re.findall(r"``([a-z_]+)``", m.group(2)))
        for n in names:
            rows[n] = (inputs, m.group(3))
    return rows


def _spec_differentiable_inputs(spec):
    v = spec.vjp
    inputs = list(v.wrt)
    if getattr(v, "u_via_minv", False) and "u" not in inputs:
        inputs.append("u")
    if getattr(v, "param_grad_op", None):
        inputs.append("params")
    return set(inputs)


def test_tutorial_vjp_table_matches_vjpspec():
    rows = _tutorial_vjp_rows()
    assert rows, "could not parse the tutorial's VJP table"
    specced = {k: s for k, s in ABI_SPECS.items() if getattr(s, "vjp", None) is not None}
    documented_public = {k for k in rows}
    missing = sorted(k for k in specced if k not in documented_public)
    assert not missing, f"differentiable ops absent from the tutorial VJP table: {missing}"
    extra = sorted(k for k in documented_public if k not in specced)
    assert not extra, f"tutorial VJP table lists ops with no VjpSpec: {extra}"
    for k, spec in specced.items():
        inputs, notes = rows[k]
        assert set(inputs) == _spec_differentiable_inputs(spec), (
            f"{k}: tutorial says {sorted(inputs)}, VjpSpec implies {sorted(_spec_differentiable_inputs(spec))}")
        if getattr(spec.vjp, "fixed_base_only", False):
            assert "fixed-base only" in notes, f"{k}: fixed-base restriction not stated in the table"


def test_tutorial_ee_pose_representation_matches_implementation():
    """VjpSpec does not encode output representations; assert them separately."""
    rows = _tutorial_vjp_rows()
    _, notes = rows["end_effector_pose"]
    assert "[xyz, rpy]" in notes and "6 per EE" in notes, notes
    assert "fk_batched" in notes and "7-coordinate" in notes, notes
    from grid_rbd._handle import RobotHandle
    doc = RobotHandle.end_effector_pose.__doc__ or ""
    assert "rpy" in doc and "6*NUM_EES" in doc, doc[:120]
