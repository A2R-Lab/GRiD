"""Regression sentinel for FAST-COMPILE hygiene of example/test robot builds.

Motivation (2026-07-24): a `register_robot(...)` on a floating-base or humanoid robot
that omits `algorithm_list` compiles the WHOLE ~35-algorithm surface into the .so, and
on a floating/non-mimic robot it ALSO emits the MuJoCo-convention ("mjx") twin of every
kernel -- ~2.1M of ~2.4M SASS lines on g1-floating. A single such build is 10-40 min and
many GB of ptxas RAM. The A1/A2 work made the CUDA-equivalence suite pin-only + subset;
this sentinel generalizes that discipline to every example/test that builds a robot, so a
new needless monolith can't creep back in.

The fix for a flagged call is ONE of:
  * `algorithm_list=[...]`  -- build only the algorithms the test/example actually exercises
    (the per-algo subset path; see test/python_wrappers/test_subset_build.py), OR
  * `enable_mujoco_kernels=False` -- for a FLOATING robot whose test never calls
    `handle.mujoco.*`, this drops the mjx twins (the bulk of the cost) while keeping the
    full pinocchio surface.

A file whose POINT is the whole surface (e.g. the native-mjx reference tests, or a
packaging smoke that must build everything) is allowlisted below with a reason.

The check is AST-only (no import, no nvcc) so it runs in milliseconds under the CPU-only
filter -- exactly when you want to catch a monolith build sneaking in.
"""

import ast
import pathlib

import pytest

_REPO = pathlib.Path(__file__).resolve().parents[1]

# Directories whose robot builds should stay fast.
_SCAN_DIRS = [
    _REPO / "test" / "python_wrappers",
    _REPO / "bindings" / "tests",
    _REPO / "bindings" / "examples",
    _REPO / "examples",
]

# Robot-name / urdf substrings that denote a LARGE robot whose full-surface monolith is
# expensive even fixed-base (humanoids / high-DOF quadrupeds).
_BIG_ROBOT_HINTS = ("g1", "h1_2", "h2_plus", "humanoid", "atlas")

# {relpath: reason} -- deliberate full-surface builds. Each MUST justify why the whole
# surface (or the mjx twins) is the point of that file.
_ALLOWLIST = {
    # The native-mjx reference tests: their entire purpose is to exercise the mjx-convention
    # twin of the second-order + gradient surface against the numpy oracle, across jax/torch.
    "bindings/tests/test_mujoco_kernel.py": "native mjx reference: must build the mjx surface",
    "bindings/tests/check_mujoco_jax_torch_parity.py": "native mjx jax/torch parity: mjx surface is the point",
}


def _kw(call, name):
    for k in call.keywords:
        if k.arg == name:
            return k.value
    return None


def _is_falsey_literal(node):
    return isinstance(node, ast.Constant) and node.value is False


def _is_floating(call):
    """floating_base present and NOT literally False -> potentially floating."""
    fb = _kw(call, "floating_base")
    if fb is None:
        return False
    return not _is_falsey_literal(fb)


def _mentions_big_robot(call):
    # any string arg/kwarg that names a known-big robot
    parts = []
    for a in call.args:
        if isinstance(a, ast.Constant) and isinstance(a.value, str):
            parts.append(a.value)
    for k in call.keywords:
        v = k.value
        if isinstance(v, ast.Constant) and isinstance(v.value, str):
            parts.append(v.value)
        # str(_URDF) / f"..." are common -- fall back to a source slice below
    blob = " ".join(parts).lower()
    return any(h in blob for h in _BIG_ROBOT_HINTS)


def _has_mitigation(call, floating):
    # algorithm_list (the subset build) always mitigates.
    if _kw(call, "algorithm_list") is not None:
        return True
    # enable_mujoco_kernels=False drops the mjx twins -- but ONLY floating/non-mimic robots
    # HAVE mjx twins, so on a fixed-base big robot it is a no-op and does NOT count.
    if floating:
        emk = _kw(call, "enable_mujoco_kernels")
        if emk is not None and _is_falsey_literal(emk):
            return True
    # **common / **kw pass-through: a dict may carry the mitigation -- be lenient (avoids
    # false-flagging kwargs-splat helper wrappers whose caller supplies the mitigation).
    if any(isinstance(k, ast.keyword) and k.arg is None for k in call.keywords):
        return True
    return False


def _register_robot_calls(tree):
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            fn = node.func
            name = fn.attr if isinstance(fn, ast.Attribute) else getattr(fn, "id", None)
            if name == "register_robot":
                yield node


def _scan():
    """Return list of (relpath, lineno, reason) offenders."""
    offenders = []
    for d in _SCAN_DIRS:
        if not d.exists():
            continue
        for path in d.rglob("*.py"):
            rel = path.relative_to(_REPO).as_posix()
            if rel in _ALLOWLIST:
                continue
            src = path.read_text()
            try:
                tree = ast.parse(src)
            except SyntaxError:
                continue
            for call in _register_robot_calls(tree):
                seg = ast.get_source_segment(src, call) or ""
                floating = _is_floating(call)
                big = _mentions_big_robot(call) or any(
                    h in seg.lower() for h in _BIG_ROBOT_HINTS)
                if not (floating or big):
                    continue
                if _has_mitigation(call, floating):
                    continue
                why = "floating (mjx monolith)" if floating else "large-robot monolith"
                offenders.append((rel, call.lineno, why))
    return offenders


def test_no_unmitigated_monolith_robot_builds():
    offenders = _scan()
    if offenders:
        lines = [f"  {rel}:{ln}  -- {why}" for rel, ln, why in offenders]
        pytest.fail(
            "register_robot(...) builds the full ~35-algo surface (and, floating, the mjx "
            "twins = ~2.1M SASS lines). Add `algorithm_list=[...]` (subset) or, for a "
            "floating non-mjx test, `enable_mujoco_kernels=False`; or allowlist with a "
            "reason in this file:\n" + "\n".join(lines))


def test_allowlist_entries_still_exist():
    """An allowlist entry pointing at a deleted/renamed file is stale -- remove it."""
    for rel in _ALLOWLIST:
        assert (_REPO / rel).exists(), f"stale fast-compile allowlist entry: {rel}"
