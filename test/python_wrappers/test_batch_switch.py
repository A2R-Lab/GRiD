"""E6 batch-switch (per-call batch-regime threads) tests.

The .so keeps a second per-algo thread count plus a threshold; a call whose
batch is <= the threshold launches at the small-batch count, otherwise the
per-algo overlay / baked default applies. The pick is a stateless threshold
compare on the current call's batch — no hysteresis, no cross-call state.

Covers: C-ABI set/readback/clear semantics, value-equality across the regime
boundary (kernels are thread-count-invariant, so the switch must never change
results), and apply_batch_overlay arming from a launch-config
``ffi_bases_by_n`` block with the tier-mismatch skip rule.

Run with:
    pytest test/python_wrappers/test_batch_switch.py -m python_wrappers -v
"""
from __future__ import annotations

import shutil
import sys
from pathlib import Path

import numpy as np
import pytest


_REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO_ROOT))

_grid_rbd = pytest.importorskip("grid_rbd", reason="grid-rbd not installed")

_URDF = (
    Path.home()
    / ".cache/robot_descriptions/drake/manipulation/models/iiwa_description/urdf/iiwa14_primitive_collision.urdf"
)
if not _URDF.exists():
    pytest.skip(f"iiwa14 URDF fixture not present at {_URDF}", allow_module_level=True)

if shutil.which("nvcc") is None:
    pytest.skip("nvcc not on PATH; grid-rbd register_robot requires it", allow_module_level=True)


pytestmark = pytest.mark.python_wrappers

# Same-algorithm float32 runs at two block sizes differ only by FP-summation
# order in the block-stride loops (see test_any_thread_count).
_TOL = 5e-5


@pytest.fixture(scope="module")
def handle():
    return _grid_rbd.register_robot(
        name="iiwa14_batch_switch",
        urdf_path=str(_URDF),
        floating_base=False,
        max_batch_size=64,
    )


@pytest.fixture(scope="module")
def samples(handle):
    rng = np.random.default_rng(7)
    nj = handle.num_joints
    q = rng.uniform(-1.0, 1.0, (64, nj)).astype(np.float32)
    qd = rng.uniform(-1.0, 1.0, (64, nj)).astype(np.float32)
    return q, qd


def test_so_exposes_batch_switch(handle):
    assert handle._runner.has_batch_switch()


def test_set_readback_clear(handle):
    r = handle._runner
    n_algo = r.algo_count()
    assert n_algo > 0
    # unarmed by default
    thr, small = r.get_batch_switch(0)
    assert thr == 0 and small == -1
    # arm, readback
    r.set_threads_for_n(0, 64, 128)
    assert r.get_batch_switch(0) == (64, 128)
    # clear via threshold=0 (n_small ignored)
    r.set_threads_for_n(0, 0, 999)
    assert r.get_batch_switch(0) == (0, -1)
    # invalid algo / negative threshold raise (C rc=1 -> RuntimeError)
    with pytest.raises(RuntimeError):
        r.set_threads_for_n(n_algo, 64, 128)
    with pytest.raises(RuntimeError):
        r.set_threads_for_n(0, -1, 128)
    # armed threshold with invalid n_small refuses
    with pytest.raises(RuntimeError):
        r.set_threads_for_n(0, 64, 0)


def test_values_identical_across_regimes(handle, samples):
    """The switch changes ONLY the launch dim3 — outputs must match at both
    batch sizes with the switch armed vs unarmed (thread-count invariance)."""
    q, qd = samples
    r = handle._runner
    n_algo = r.algo_count()
    ref_small = {}
    ref_large = {}
    for name in ("inverse_dynamics", "crba", "minv"):
        fn = getattr(handle, name)
        args_small = (q[:16], qd[:16]) if name == "inverse_dynamics" else (q[:16],)
        args_large = (q, qd) if name == "inverse_dynamics" else (q,)
        ref_small[name] = np.asarray(fn(*args_small)).copy()
        ref_large[name] = np.asarray(fn(*args_large)).copy()
    # arm EVERY algo with an aggressive small-batch count at threshold 32
    for i in range(n_algo):
        r.set_threads_for_n(i, 32, 96)
    try:
        for name in ("inverse_dynamics", "crba", "minv"):
            fn = getattr(handle, name)
            args_small = (q[:16], qd[:16]) if name == "inverse_dynamics" else (q[:16],)
            args_large = (q, qd) if name == "inverse_dynamics" else (q,)
            got_small = np.asarray(fn(*args_small))  # batch 16 <= 32: small regime
            got_large = np.asarray(fn(*args_large))  # batch 64 > 32: large regime
            np.testing.assert_allclose(got_small, ref_small[name], atol=_TOL, rtol=0)
            np.testing.assert_allclose(got_large, ref_large[name], atol=_TOL, rtol=0)
    finally:
        for i in range(n_algo):
            r.set_threads_for_n(i, 0, 0)


def test_global_override_beats_switch(handle, samples):
    """set_threads_per_block (explicit user intent) outranks the batch switch;
    correctness still holds either way — this pins the precedence contract by
    exercising both paths with values compared to the unarmed reference."""
    q, qd = samples
    r = handle._runner
    ref = np.asarray(handle.inverse_dynamics(q[:8], qd[:8])).copy()
    r.set_threads_for_n(0, 32, 96)
    handle.set_threads_per_block(160)
    try:
        got = np.asarray(handle.inverse_dynamics(q[:8], qd[:8]))
        np.testing.assert_allclose(got, ref, atol=_TOL, rtol=0)
    finally:
        handle.set_threads_per_block(0)
        r.set_threads_for_n(0, 0, 0)


def test_apply_batch_overlay_from_config(handle, tmp_path, monkeypatch):
    """apply_batch_overlay arms from ffi_bases_by_n and skips tier mismatches."""
    import json
    import grid_codegen._launch_config as lc

    robot_key = handle._meta.get("launch_config_robot")
    if not robot_key:
        pytest.skip("handle has no launch_config_robot in meta")
    import os
    real = os.path.join(lc._launch_configs_dir(), str(robot_key),
                        lc.LAUNCH_CONFIG_DEFAULT_GPU + ".json")
    if not os.path.exists(real):
        pytest.skip(f"no launch config for {robot_key}")
    doc = json.loads(Path(real).read_text())
    baked = lc.load_launch_config(robot_key, handle.floating_base, profile="ffi")
    # synthesize a by-n block: first entry matches its baked tier (arms), the
    # rest are forced to a WRONG tier (must be skipped)
    from grid_codegen.algo_registry import build_launch_config_algo_to_symbol
    a2s = build_launch_config_algo_to_symbol()
    inv_tier = {v: k for k, v in lc.LAUNCH_CONFIG_TIER_SYMBOL.items()}
    base = "floating" if handle.floating_base else "fixed"
    block, armed_expect = {}, 0
    for i, (key, sym) in enumerate(a2s.items()):
        baked_tier = inv_tier.get((baked.get(sym) or {}).get("tier"))
        if baked_tier is None:
            continue
        if armed_expect == 0:
            block[key] = {"tier": baked_tier, "threads": 96}
            armed_expect = 1
        else:
            wrong = "lite" if baked_tier != "lite" else "minimal"
            block[key] = {"tier": wrong, "threads": 96}
    doc["ffi_bases_by_n"] = {"16": {base: block}}
    doc["ffi_by_n_threshold"] = 24
    fake_dir = tmp_path / "launch_configs" / str(robot_key)
    fake_dir.mkdir(parents=True)
    (fake_dir / (lc.LAUNCH_CONFIG_DEFAULT_GPU + ".json")).write_text(json.dumps(doc))
    monkeypatch.setattr(lc, "_launch_configs_dir",
                        lambda: str(tmp_path / "launch_configs"))
    # _handle imports the names off GRiDCodeGenerator's re-export — patch there
    # too. NOTE: `import grid_codegen.GRiDCodeGenerator as x` binds the CLASS
    # (the package __init__ rebinds that attribute), so go via sys.modules.
    import importlib
    gcg_mod = importlib.import_module("grid_codegen.GRiDCodeGenerator")
    monkeypatch.setattr(gcg_mod, "_launch_configs_dir",
                        lambda: str(tmp_path / "launch_configs"))
    try:
        n = handle.apply_batch_overlay("ffi")
        assert n == armed_expect, f"armed {n}, expected {armed_expect} (tier-skip rule)"
        # the armed algo reads back threshold 24
        armed = [i for i in range(handle._runner.algo_count())
                 if handle._runner.get_batch_switch(i)[0] != 0]
        assert len(armed) == armed_expect
        assert handle._runner.get_batch_switch(armed[0]) == (24, 96)
    finally:
        for i in range(handle._runner.algo_count()):
            handle._runner.set_threads_for_n(i, 0, 0)
