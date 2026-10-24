"""Gate for the environment ownership API (HJCD-IK ask G8a, 2026-10-23).

`grid_collision::make_host_environment` turns plain rows (spheres 4, capsules 7, cuboids 10 with
a quaternion and full dimensions, planes 4) into validated host lists; `HostEnvironment::view()`
feeds the host primitives; `DeviceEnvironment` deep-copies the lists to the current device,
records it, frees on destruction and moves but never copies. The runner checks the row layouts
and the quaternion box, the rejections, bit-identical verdicts and clearances against a hand
upload of the same lists, host-versus-device agreement of the primitives, move semantics and
that destruction returns the device memory (cuda_collision_environment_runner.cu).
"""
from __future__ import annotations
import re
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from test.cuda_equivalents.cuda_harness import _detect_cuda_arch
from test.cuda_equivalents.test_cuda_collision_config_free import _robot, _gen_header
from test.cuda_equivalents.test_cuda_collision_warp import _mixed_spec, _two_tier_spec

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

COLLISION_INCLUDE = REPO_ROOT / "grid_codegen" / "collision"
RUNNER_SOURCE = Path(__file__).with_name("cuda_collision_environment_runner.cu")


def _compile(build_dir):
    nvcc = shutil.which("nvcc")
    if nvcc is None:
        pytest.skip("nvcc not found; install CUDA Toolkit to run CUDA tests.")
    runner_copy = build_dir / RUNNER_SOURCE.name
    shutil.copyfile(RUNNER_SOURCE, runner_copy)
    arch = _detect_cuda_arch()
    exe = build_dir / (RUNNER_SOURCE.stem + ".exe")
    cmd = [nvcc, "-std=c++17", "-O2", "-gencode", f"arch=compute_{arch},code=sm_{arch}",
           "-I", str(build_dir), "-I", str(COLLISION_INCLUDE), "-o", str(exe), str(runner_copy)]
    result = subprocess.run(cmd, cwd=build_dir, capture_output=True, text=True)
    if result.returncode != 0:
        pytest.fail(f"{RUNNER_SOURCE.name} compile FAILED.\ncmd: {' '.join(cmd)}\nstdout:\n{result.stdout}\nstderr:\n{result.stderr}")
    return exe


def _run(exe, threads, n_cfg, seed):
    run = subprocess.run([str(exe), str(threads), str(n_cfg), str(seed)], capture_output=True, text=True)
    assert run.returncode == 0, f"environment runner FAILED:\nstdout:\n{run.stdout[-3000:]}\nstderr:\n{run.stderr}"
    checks = dict(re.findall(r"^CHECK (\w+) (PASS|FAIL)$", run.stdout, re.M))
    assert set(checks) == {"layout", "rejects", "parity", "host", "move", "lifetime"}, run.stdout
    assert all(v == "PASS" for v in checks.values()), run.stdout
    assert "RESULT PASS" in run.stdout
    return run.stdout


@pytest.mark.cuda_equivalence
@pytest.mark.developer_only
@pytest.mark.robot_smoke
@pytest.mark.parametrize("tiers", ["single", "two"])
def test_owned_environment_matches_a_hand_upload(tmp_path, tiers):
    robot = _robot()
    build_dir = tmp_path / f"env_{tiers}"
    _gen_header(robot, build_dir, _mixed_spec(robot) if tiers == "single" else _two_tier_spec(robot))
    exe = _compile(build_dir)
    print(_run(exe, 64, 120, 5))
