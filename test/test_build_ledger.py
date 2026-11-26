"""CPU-only tests for the record of wrapper build requests (test/build_ledger.py)
and its use by the split driver's compile pool. No GPU and no compiler: the
build step is a stand-in."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
import build_ledger  # noqa: E402
import run_split_suite as rss  # noqa: E402

grid_rbd = pytest.importorskip("grid_rbd")
from grid_assets import robot_urdf  # noqa: E402

_MODEL = "<robot name='written_by_the_test'><link name='base'/></robot>"


@pytest.fixture(autouse=True)
def _own_ledger(tmp_path, monkeypatch):
    monkeypatch.setattr(build_ledger, "LEDGER_PATH", tmp_path / "ledger" / "wrapper_builds.json")


@pytest.fixture
def recorded(tmp_path, monkeypatch):
    """A recorder over a stand-in build step; returns (record file, calls)."""
    calls = []

    def warm_robot(name, urdf_path=None, **kwargs):
        calls.append((name, urdf_path, kwargs))
        return f"key-{len(calls)}", tmp_path / "robot.so", {}

    monkeypatch.setattr(grid_rbd, "warm_robot", warm_robot)
    record = tmp_path / "builds" / "module.jsonl"
    build_ledger.install_recorder(record)
    return record, calls


def test_a_request_is_recorded_with_its_model_text_and_plain_options(recorded, tmp_path):
    record, calls = recorded
    urdf = tmp_path / "model.urdf"
    urdf.write_text(_MODEL)
    result = grid_rbd.warm_robot("a", str(urdf), floating_base=True, max_batch_size=4,
                                 armature=np.array([0.0, 0.25]), ee_joint_names=("tip",), cache_dir=None)
    assert result[0] == "key-1" and len(calls) == 1
    grid_rbd.warm_robot("b", urdf_string=_MODEL, dtype="float64")
    first, second = build_ledger.read_records(record)
    assert first == {"urdf": _MODEL, "key": "key-1", "options": {
        "floating_base": True, "max_batch_size": 4, "armature": [0.0, 0.25], "ee_joint_names": ["tip"]}}
    assert second == {"urdf": _MODEL, "key": "key-2", "options": {"dtype": "float64"}}


def test_private_cache_forced_planned_and_failed_requests_are_not_recorded(recorded, tmp_path, monkeypatch):
    record, _ = recorded
    grid_rbd.warm_robot("a", urdf_string=_MODEL, cache_dir=tmp_path / "own")
    grid_rbd.warm_robot("a", urdf_string=_MODEL, force_rebuild=True)
    grid_rbd.warm_robot("a", urdf_string=_MODEL, plan_only=True)
    grid_rbd.warm_robot("a", urdf_string=_MODEL, armature=object())  # nothing a replay could rebuild
    assert build_ledger.read_records(record) == []


def test_a_request_that_raises_is_not_recorded(tmp_path, monkeypatch):
    def failing(name, urdf_path=None, **kwargs):
        raise RuntimeError("compile failed")

    monkeypatch.setattr(grid_rbd, "warm_robot", failing)
    record = tmp_path / "module.jsonl"
    build_ledger.install_recorder(record)
    with pytest.raises(RuntimeError):
        grid_rbd.warm_robot("a", urdf_string=_MODEL)
    assert not record.exists()


def test_repeated_requests_are_stored_once_and_garbage_lines_are_skipped(tmp_path):
    record = tmp_path / "module.jsonl"
    line = json.dumps({"urdf": _MODEL, "options": {"max_batch_size": 4}, "key": "k"})
    record.write_text(line + "\n" + line + "\nnot json\n" + json.dumps({"urdf": 3, "options": {}}) + "\n")
    assert len(build_ledger.read_records(record)) == 1
    assert build_ledger.read_records(tmp_path / "missing.jsonl") == []


@pytest.fixture
def module(tmp_path, monkeypatch):
    """A wrapper module file in a private wrappers directory, with the driver pointed at it."""
    folder = tmp_path / "python_wrappers"
    folder.mkdir()
    path = folder / "test_recorded_module.py"
    path.write_text("# version 1\n")
    monkeypatch.setattr(rss, "WRAPPERS_DIR", folder)
    return path


def test_stored_requests_become_pool_entries_until_the_module_changes(module):
    iiwa = robot_urdf("iiwa14").read_text()
    build_ledger.store(module.stem, module, [
        {"urdf": iiwa, "options": {"floating_base": False, "max_batch_size": 8, "armature": None,
                                   "enable_mujoco_kernels": True, "dtype": "float32"}, "key": "k1"},
        {"urdf": _MODEL, "options": {"floating_base": True, "max_batch_size": 256,
                                     "algorithm_list": ["aba"], "enable_mujoco_kernels": False}, "key": "k2"},
    ])
    packaged, written = build_ledger.entries(module.stem, module)
    # Options equal to the defaults are dropped, so a packaged robot's entry has
    # the description of the same entry listed by hand.
    assert packaged == {"robot": "iiwa14", "max_batch_size": 8}
    assert rss._warm_desc(packaged) == rss._warm_desc(rss._r("iiwa14", max_batch_size=8))
    assert written["urdf_string"] == _MODEL and written["robot"].startswith("model_")
    assert {k: v for k, v in written.items() if k not in ("robot", "urdf_string")} == {
        "floating_base": True, "algorithm_list": ["aba"], "enable_mujoco_kernels": False}
    assert "urdf_string" not in rss._warm_desc(written) and _MODEL not in rss._warm_desc(written)
    module.write_text("# version 2\n")
    assert build_ledger.entries(module.stem, module) == []


def test_the_driver_warms_recorded_requests_and_stops_reporting_the_module(module):
    module.write_text("import grid_rbd\n\ndef test_it():\n    grid_rbd.register_robot('x', 'y')\n")
    assert rss.module_warm(module.stem) == []
    assert any(module.stem in line for line in rss.drift_guard([module.stem]))
    build_ledger.store(module.stem, module, [
        {"urdf": _MODEL, "options": {"max_batch_size": 4}, "key": "k"}])
    (entry,) = rss.module_warm(module.stem)
    assert entry["urdf_string"] == _MODEL
    assert rss.drift_guard([module.stem]) == []
    assert [desc for desc, _ in rss.warm_entries([module.stem])] == [rss._warm_desc(entry)]


def test_a_request_listed_by_hand_and_recorded_is_one_pool_job(monkeypatch, tmp_path):
    mod = "test_iiwa14_smoke"
    assert rss.WARM_MANIFEST[mod] == [rss._r("iiwa14", max_batch_size=8)]
    build_ledger.store(mod, rss.WRAPPERS_DIR / f"{mod}.py", [
        {"urdf": robot_urdf("iiwa14").read_text(), "options": {"max_batch_size": 8}, "key": "k"},
        {"urdf": robot_urdf("iiwa14").read_text(), "options": {"max_batch_size": 16}, "key": "k2"}])
    assert rss.module_warm(mod) == [rss._r("iiwa14", max_batch_size=8), rss._r("iiwa14", max_batch_size=16)]


def test_a_recorded_model_is_replayed_from_its_text(monkeypatch):
    seen = {}

    def warm_robot(name, urdf_path=None, **kwargs):
        seen.update(name=name, urdf_path=urdf_path, kwargs=kwargs)
        return "0123456789abcdef", Path("robot.so"), {}

    monkeypatch.setattr(grid_rbd, "warm_robot", warm_robot)
    entry = {"robot": "model_0123456789", "urdf_string": _MODEL, "max_batch_size": 4}
    status, _, detail = rss._warm_one_entry(rss._warm_desc(entry), entry)
    assert status == "HIT" and detail == "0123456789ab"
    assert seen["urdf_path"] is None and seen["kwargs"] == {"urdf_string": _MODEL, "max_batch_size": 4}


def test_a_clean_wrapper_shard_stores_its_requests_under_the_module_file(module, tmp_path, monkeypatch):
    monkeypatch.setattr(rss, "REPO_ROOT", module.parents[1])
    spec = rss.ShardSpec(name="recycled_name", domain="wrappers",
                         targets=[f"python_wrappers/{module.name}"], fingerprint_paths=[])
    record = rss._build_record_path(tmp_path, spec)
    record.parent.mkdir()
    record.write_text(json.dumps({"urdf": _MODEL, "options": {"max_batch_size": 4}, "key": "k"}) + "\n")
    rss._store_build_requests(spec, tmp_path)
    assert len(build_ledger.entries(module.stem, module)) == 1
    # A native shard has no module record.
    rss._store_build_requests(rss.ShardSpec(name="cuda_00", domain="cuda", targets=["a::b"],
                                            fingerprint_paths=[]), tmp_path)
    assert set(json.loads(build_ledger.LEDGER_PATH.read_text())["modules"]) == {module.stem}
