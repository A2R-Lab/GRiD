"""CPU gates for the Wave A' header content-key replayer
(test/header_key_replay.py) — the SPLIT_REFRESH consumer of the A4 records.

The end-to-end case really generates a subset header twice (record side and
replay side) so the byte-identity contract is exercised against the actual
codegen, not a stub.
"""
from __future__ import annotations

import contextlib
import hashlib
import json
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(Path(__file__).resolve().parent))

import header_key_replay as hkr  # noqa: E402

_ENV = {"GRID_ENABLE_MUJOCO_KERNELS": "0", "GRID_CODEGEN_PROFILE": None,
        "GRID_CUDA_TARGET_SHARED_MEM_BYTES": None,
        "GRID_CUDA_SHARED_MEM_TYPE_SIZE_BYTES": None}

_CTOR = {"DEBUG_MODE": False, "gen_print_mat": False, "file_namespace": "grid",
         "USE_JOINT_DYNAMICS": False, "MUJOCO_OUTPUT": False,
         "launch_config_profile": "host", "runtime_joint_dynamics": False,
         "launch_config_robot": False}


def _make_direct_record(tmp_path) -> dict:
    """Generate an id-only iiwa14 header the way the A4 recorder would see it
    and return the record the conftest wrapper would have emitted."""
    from config import robot_urdf
    from URDFParser import URDFParser
    from grid_codegen.GRiDCodeGenerator import GRiDCodeGenerator

    urdf = Path(robot_urdf("iiwa14"))
    out = tmp_path / "grid.cuh"
    with hkr._apply_env(_ENV), open(os.devnull, "w") as devnull, \
            contextlib.redirect_stdout(devnull):
        robot = URDFParser().parse(str(urdf), floating_base=False)
        GRiDCodeGenerator(robot, FILE_NAMESPACE="grid").gen_all_code(
            algorithm_list=["inverse_dynamics"], output_path=str(out))
    return {
        "kind": "direct", "robot": "iiwa14", "floating": False,
        "kwargs": {"algorithm_list": ["inverse_dynamics"]},
        "opaque": {}, "codegen": dict(_CTOR), "env": dict(_ENV),
        "urdf_sha256": hashlib.sha256(urdf.read_bytes()).hexdigest(),
        "content_sha256": hashlib.sha256(out.read_bytes()).hexdigest(),
    }


def test_direct_record_replays_byte_identical(tmp_path):
    record = _make_direct_record(tmp_path)
    cache: dict = {}
    ok, why = hkr.replay_record(record, cache)
    assert ok is True, why
    # a rotated content hash must be detected as staleness, and the cache must
    # serve the second verdict without regenerating (same identity key)
    rotated = dict(record, content_sha256="0" * 64)
    ok2, why2 = hkr.replay_record(rotated, cache)
    assert ok2 is False
    assert len(cache) == 1


def test_unreplayable_records_stay_conservative():
    base = {"kind": "direct", "robot": "iiwa14", "floating": False,
            "kwargs": {}, "opaque": {}, "codegen": dict(_CTOR),
            "env": {}, "content_sha256": "x" * 64}
    ok, why = hkr.replay_record(dict(base, opaque={"collision_spec": "Spec()"}))
    assert ok is None and "recipes" in why
    ok, why = hkr.replay_record(dict(base, robot=None))
    assert ok is None
    ok, why = hkr.replay_record(dict(base, codegen=dict(_CTOR, launch_config_robot=True)))
    assert ok is None and "launch_config_robot" in why
    ok, why = hkr.replay_record({"kind": "mystery", "content_sha256": "x"})
    assert ok is None
    ok, why = hkr.replay_record({"kind": "direct"})  # no content hash
    assert ok is None


def test_urdf_rotation_is_honest_staleness(tmp_path):
    record = _make_direct_record(tmp_path)
    record["urdf_sha256"] = "f" * 64  # pretend the asset changed since record
    ok, why = hkr.replay_record(record)
    assert ok is False and "urdf" in why


def test_shard_verdict_folding(monkeypatch):
    rows = [{"content_sha256": "a", "kind": "direct"},
            {"content_sha256": "b", "kind": "direct"}]
    verdicts = {"a": (True, "ok"), "b": (True, "ok")}
    monkeypatch.setattr(hkr, "replay_record",
                        lambda r, c=None: verdicts[r["content_sha256"]])
    assert hkr.shard_replay_verdict(rows)[0] is True
    verdicts["b"] = (None, "no recipe")
    assert hkr.shard_replay_verdict(rows)[0] is None
    verdicts["b"] = (False, "rotated")
    assert hkr.shard_replay_verdict(rows)[0] is False
    assert hkr.shard_replay_verdict([])[0] is None


def test_plan_refresh_per_shard_replay(monkeypatch, tmp_path):
    """A' P2 in plan_refresh: with generator inputs changed, a carried cuda
    shard whose records replay clean stays carried; one whose bytes rotated
    is demoted; a record-less shard takes the (stubbed) fallback verdict —
    computed at most once."""
    import codegen_neutrality
    import run_split_suite as rss

    ids = ["test/cuda_equivalents/test_a.py::t1",
           "test/cuda_equivalents/test_b.py::t2",
           "test/cuda_equivalents/test_c.py::t3"]
    old = {"repo": {"commit_sha": "deadbeef"},
           "shards": [
               {"name": "cuda_clean", "domain": "cuda", "node_ids": [ids[0]],
                "fingerprint": {"included_paths": ["p"], "digest": "F"}},
               {"name": "cuda_rotated", "domain": "cuda", "node_ids": [ids[1]],
                "fingerprint": {"included_paths": ["p"], "digest": "F"}},
               {"name": "cuda_norec", "domain": "cuda", "node_ids": [ids[2]],
                "fingerprint": {"included_paths": ["p"], "digest": "F"}}],
           "tests": []}
    monkeypatch.setattr(codegen_neutrality, "codegen_inputs_changed",
                        lambda sha: True)
    fallback_calls = []

    def fallback(receipt):
        fallback_calls.append(1)
        return True, "stubbed fallback carry"
    monkeypatch.setattr(codegen_neutrality, "cuda_carry_soundness", fallback)

    keys = tmp_path / "gpu-proof-header-keys.json"
    keys.write_text(json.dumps({"schema": 1, "shards": {
        "cuda_clean": [{"content_sha256": "same"}],
        "cuda_rotated": [{"content_sha256": "rot"}]}}))
    monkeypatch.setattr(rss, "HEADER_KEYS_PATH", keys)

    import header_key_replay
    monkeypatch.setattr(
        header_key_replay, "shard_replay_verdict",
        lambda rows, cache=None: (
            (None, "no records") if not rows else
            (True, "identical") if rows[0]["content_sha256"] == "same" else
            (False, "rotated")))

    stale, carried, _wrap, cuda_fresh = rss.plan_refresh(
        old, ids, [], {}, 7200.0, lambda paths: "F")
    assert "cuda_clean" in carried and "cuda_norec" in carried
    assert "cuda_rotated" in stale
    assert len(fallback_calls) == 1  # lazy, computed once
    assert {s.name for s in cuda_fresh} == {"cuda_rotated"}


def test_collision_spec_record_replays(tmp_path):
    """P3 outcome: collision specs are plain JSON-clean dicts, so a
    collision-cell record replays GENERICALLY — no HEADER_RECIPES entry.
    Guards the tuples->lists JSON round-trip staying byte-identical (if a
    future spec field breaks serializability, the recorder routes it to
    `opaque` and this test's record would go conservative instead of green)."""
    from config import robot_urdf
    from URDFParser import URDFParser
    from grid_codegen.GRiDCodeGenerator import GRiDCodeGenerator
    from grid_codegen.algorithms._collision import build_self_cc_ranges

    urdf = Path(robot_urdf("iiwa14"))
    out = tmp_path / "grid.cuh"
    with hkr._apply_env(_ENV), open(os.devnull, "w") as devnull, \
            contextlib.redirect_stdout(devnull):
        robot = URDFParser().parse(str(urdf), floating_base=False)
        spec = {"anchor": [2, 4],
                "offset": [0.02, -0.01, 0.03, -0.02, 0.01, -0.03],
                "radius": [0.5, 0.5],
                "self_cc_ranges": build_self_cc_ranges(robot, [2, 4])}
        GRiDCodeGenerator(robot, FILE_NAMESPACE="grid").gen_all_code(
            codegen_profile="kinematics", output_path=str(out),
            collision_spec=spec)
    record = {
        "kind": "direct", "robot": "iiwa14", "floating": False,
        # what the recorder stores: the JSON round-trip of the kwargs
        "kwargs": json.loads(json.dumps(
            {"codegen_profile": "kinematics", "collision_spec": spec})),
        "opaque": {}, "codegen": dict(_CTOR), "env": dict(_ENV),
        "urdf_sha256": hashlib.sha256(urdf.read_bytes()).hexdigest(),
        "content_sha256": hashlib.sha256(out.read_bytes()).hexdigest(),
    }
    ok, why = hkr.replay_record(record)
    assert ok is True, why
