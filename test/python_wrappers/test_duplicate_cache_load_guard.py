"""One process cannot load one model from two cache directories.

Two builds of one model carry the same PyTorch operator namespace, and PyTorch
aborts the process when the second registers. The handle constructor must
raise ``RuntimeError`` before that load, on every backend, and leave the first
artifact working. test/test_torch_namespace_claim.py checks the guard's logic
on stand-in files. This builds the two real artifacts and loads them.

The loads run in a child process: without the guard the second load aborts,
and the child's exit status says so instead of taking the test session down.

Run with:
    pytest test/python_wrappers/test_duplicate_cache_load_guard.py -m python_wrappers -v
"""
from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO_ROOT))

pytest.importorskip("grid_rbd", reason="grid-rbd not installed")
pytest.importorskip("jax", reason="jax not installed (pip install grid-rbd[jax])")
pytest.importorskip("torch", reason="torch not installed")
from grid_assets import robot_urdf  # noqa: E402

if shutil.which("nvcc") is None:
    pytest.skip("nvcc not on PATH; grid-rbd register_robot requires it", allow_module_level=True)

pytestmark = pytest.mark.python_wrappers

_CHILD = r"""
import json, sys
import numpy as np
import grid_rbd

urdf, cache_a, cache_b, backend = sys.argv[1:5]
name = "iiwa14_duplicate_cache_pytest"
options = dict(max_batch_size=4, algorithm_list=["inverse_dynamics"], enable_mujoco_kernels=False)
rng = np.random.default_rng(3)


def register(cache, kind):
    return grid_rbd.register_robot(name, urdf, cache_dir=cache, backend=kind, **options)


def values(handle, kind="numpy"):
    nq = first.nq
    q, qd, qdd = (np.linspace(lo, hi, 4 * nq, dtype=np.float32).reshape(4, nq)
                  for lo, hi in ((-1.0, 1.0), (-0.5, 0.7), (0.2, -0.9)))
    if kind == "torch":
        import torch
        out = handle.inverse_dynamics(*(torch.as_tensor(x, device="cuda") for x in (q, qd, qdd)))
        return out.detach().cpu().numpy().reshape(4, -1)
    return np.asarray(handle.inverse_dynamics(q, qd, qdd)).reshape(4, -1)


first = register(cache_a, "numpy")
report = dict(first=first._so_path, backend=backend)
before = values(first)
report["finite"] = bool(np.all(np.isfinite(before)) and np.any(before != 0))

# The second cache builds its own artifact, then the load must be refused.
try:
    register(cache_b, backend)
    report["second"] = "loaded"
except RuntimeError as exc:
    report["second"] = str(exc)

# The first artifact still loads and still computes, on this backend too.
again = register(cache_a, backend)
report["same_after"] = bool(np.array_equal(before, values(first)))
report["backend_matches"] = bool(np.allclose(before, values(again, backend), rtol=1e-5, atol=1e-5))
print("REPORT " + json.dumps(report))
"""


@pytest.mark.parametrize("backend", ["numpy", "torch", "jax"])
def test_second_cache_is_refused_and_the_first_keeps_working(tmp_path_factory, backend):
    # One pair of caches for the three backends: the artifact does not depend
    # on the backend, so the builds are shared and each backend is a new process.
    root = Path(tmp_path_factory.getbasetemp()) / "duplicate_cache_load_guard"
    cache_a, cache_b = root / "a", root / "b"
    run = subprocess.run(
        [sys.executable, "-c", _CHILD, str(robot_urdf("iiwa14")), str(cache_a), str(cache_b), backend],
        capture_output=True, text=True, cwd=str(_REPO_ROOT))
    assert run.returncode == 0, (
        f"the child exited with {run.returncode} (a negative value is a signal: the second "
        f"load reached PyTorch's registration and aborted)\n{run.stderr[-4000:]}")
    line = next(l for l in run.stdout.splitlines() if l.startswith("REPORT "))
    report = json.loads(line[len("REPORT "):])
    assert report["finite"]
    message = report["second"]
    assert message != "loaded", "the same model loaded from a second cache directory"
    assert "another cache" in message and "grid_rbd_torch_k" in message
    assert report["first"] in message and str(cache_b) in message
    assert report["same_after"] and report["backend_matches"]
