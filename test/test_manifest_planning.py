"""The pre-freeze writer must describe exactly the driver's scheduled IDs."""
from types import SimpleNamespace

import pytest

from test import run_split_suite as driver
from test.receipt_integrity import write_manifest, write_manifest_ids


def test_receipt_and_collection_writers_are_byte_identical(tmp_path):
    ids = ["test/b.py::test_two", "test/a.py::test_one"]
    paths = [tmp_path / "receipt", tmp_path / "collection"]
    for path in paths:
        path.write_text("# Reviewed scope\nobsolete\n")
    write_manifest({"tests": [{"node_id": node} for node in ids]}, paths[0])
    write_manifest_ids(ids, paths[1])
    assert paths[0].read_bytes() == paths[1].read_bytes()


@pytest.mark.parametrize("ids", [[], ["duplicate", "duplicate"]])
def test_manifest_refuses_empty_or_duplicate_ids(tmp_path, ids):
    with pytest.raises(ValueError):
        write_manifest_ids(ids, tmp_path / "scope")
    assert not (tmp_path / "scope").exists()


@pytest.fixture
def planned(monkeypatch):
    monkeypatch.setattr(driver, "discover_modules", lambda: ["test_a"])
    monkeypatch.setattr(driver, "collect_wrapper_node_ids",
                        lambda specs: ["test/python_wrappers/test_a.py::test_a"])
    ids = ["test/cuda_equivalents/test_b.py::test_b"]
    shards = [driver.ShardSpec("cuda_00", "cuda", ids.copy(), [])]
    monkeypatch.setattr(driver, "build_cuda_shards", lambda *args: (shards, ids))
    return shards, ids


def test_plan_includes_both_domains_once(planned):
    assert driver.plan_release_scope(60) == [
        "test/cuda_equivalents/test_b.py::test_b",
        "test/python_wrappers/test_a.py::test_a"]


@pytest.mark.parametrize("damage", ["missing", "duplicate", "unexpected"])
def test_plan_refuses_partial_or_duplicated_partition(planned, damage):
    shards, ids = planned
    if damage == "missing":
        shards.clear()
    elif damage == "duplicate":
        shards[0].targets *= 2
    else:
        shards[0].targets.append("test/cuda_equivalents/test_b.py::unexpected")
    with pytest.raises(ValueError):
        driver.plan_release_scope(60)


@pytest.mark.parametrize("rc", [1, 2, 5])
def test_partial_wrapper_collection_fails_even_with_ids(monkeypatch, rc):
    monkeypatch.setattr(driver.subprocess, "run", lambda *a, **k: SimpleNamespace(
        returncode=rc, stdout="test/python_wrappers/test_a.py::test_a\n", stderr="error"))
    with pytest.raises(RuntimeError):
        driver.collect_wrapper_node_ids(driver.wrapper_specs(["test_a"]))


def test_wrapper_collection_uses_execution_marker(monkeypatch):
    def run(cmd, **kwargs):
        assert cmd[cmd.index("-m", 3) + 1] == "gpu_proof"
        return SimpleNamespace(returncode=0,
            stdout="test/python_wrappers/test_a.py::test_a\n", stderr="")
    monkeypatch.setattr(driver.subprocess, "run", run)
    specs = driver.wrapper_specs(["test_a"])
    assert specs[0].apply_marker
    assert driver.collect_wrapper_node_ids(specs) == ["test/python_wrappers/test_a.py::test_a"]


def test_secondary_gpu_cannot_define_primary_manifest(tmp_path, monkeypatch):
    monkeypatch.setattr("sys.argv", ["driver", "--plan-manifest", str(tmp_path / "scope")])
    monkeypatch.setattr(driver, "receipt_row_for_this_gpu", lambda: SimpleNamespace(primary=False))
    monkeypatch.setattr(driver, "plan_release_scope", lambda *args: pytest.fail("must refuse first"))
    with pytest.raises(ValueError, match="primary"):
        driver.main()
    assert not (tmp_path / "scope").exists()
