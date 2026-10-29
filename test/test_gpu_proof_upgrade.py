"""CPU integration contracts for the pinned receipt tool; never sign evidence."""
from copy import deepcopy
from pathlib import Path

import pytest
import yaml

pytest.importorskip("pytest_gpu_proof")
from pytest_gpu_proof.merge import MergeError, merge_payloads


def payload(node, compiler=None):
    return {
        "schema_version": "3",
        "repo": {"commit_sha": "synthetic", "dirty": False},
        "fingerprint": {"digest": "synthetic"},
        "mode": "local",
        "environment": {"nvcc_version": compiler},
        "session": {"started_at": "2026-01-01T00:00:00Z",
                    "ended_at": "2026-01-01T00:00:01Z"},
        "tests": [{"node_id": node, "outcome": "passed", "duration_s": 0}],
    }


@pytest.mark.parametrize("other", ["CUDA V12.8", None])
def test_fresh_merge_rejects_different_or_unknown_compiler(other):
    with pytest.raises(MergeError, match="environment.nvcc_version"):
        merge_payloads([payload("a", "CUDA V13.2"), payload("b", other)], ["a", "b"])


def test_matching_compiler_survives_unsigned_merge():
    merged = merge_payloads([payload("a", "CUDA V13.2"), payload("b", "CUDA V13.2")], ["a", "b"])
    assert merged["environment"]["nvcc_version"] == "CUDA V13.2"
    assert [t["node_id"] for t in merged["tests"]] == ["a", "b"]
    assert not merged.get("signature")


def test_legacy_missing_compiler_is_not_invented_or_input_mutated():
    old = payload("a")
    del old["environment"]["nvcc_version"]
    before = deepcopy(old)
    merged = merge_payloads([old, payload("b")], ["old", "unknown"])
    assert old == before
    assert merged["environment"].get("nvcc_version") is None
    assert len(merged["tests"]) == 2


def test_cpu_ci_installs_receipt_tool_on_every_supported_interpreter():
    root = Path(__file__).resolve().parents[1]
    workflow = yaml.safe_load((root / ".github/workflows/verify-gpu-proof.yml").read_text())
    lane = workflow["jobs"]["cpu-lane"]
    assert "3.10" in lane["strategy"]["matrix"]["python"]
    step = next(s for s in lane["steps"] if s.get("name") == "Install GRiD (CPU deps only)")
    assert "if" not in step
    assert "sys.version_info" not in step["run"]
    assert "pytest-gpu-proof==" in step["run"]


def test_plugin_compiler_metadata_matches_wrapper_selection(monkeypatch):
    # No compiler/GPU required: both paths must honor PATH, not a separately
    # selected compiler override that the signed plugin field cannot describe.
    import subprocess
    from grid_rbd._compile import find_nvcc
    from pytest_gpu_proof.receipt import _nvcc_version

    monkeypatch.setattr("grid_rbd._compile.shutil.which", lambda name: "/private/cuda/bin/nvcc" if name == "nvcc" else None)
    def run(args, **kwargs):
        assert args == ["nvcc", "--version"]
        return subprocess.CompletedProcess(args, 0, "  compiler identity\n", "")
    monkeypatch.setattr("pytest_gpu_proof.receipt.subprocess.run", run)
    assert find_nvcc() == "/private/cuda/bin/nvcc"
    assert _nvcc_version() == "compiler identity"
