"""CUDA gate for W3 Step 3 `grid_collision::config_free` (the namespace emitter).

Certifies the END-TO-END binding: the codegen emits a `grid_collision` namespace whose
`config_free` runs the W1b batched extractor (grid::multi_target_position_device) then the
static SDF checks (grid_collision_geometry.cuh). Self-consistent (no external oracle):
  * empty / far environment + tiny radii  => config_free == free
  * obstacle placed ON sphere 0           => config_free == in-collision
The SDF math + baked-range self-collision are unit-tested by test_cuda_collision_geometry.py;
this gate covers the generated data tables + the extractor->config_free wiring.
"""
from __future__ import annotations

import contextlib
import os
import shutil
import subprocess
from pathlib import Path

import pytest

from GRiDCodeGenerator import GRiDCodeGenerator
from GRiDCodeGenerator.algorithms._collision import build_self_cc_ranges
from test.cuda_equivalents.test_cuda_executable_equivalence import _detect_cuda_arch
from RBDReference.tests.model_sources import resolve_robot_spec, iter_robot_cases
from RBDReference.tests import MANIFEST_PATH
from RBDReference.equivalents.reference_backend import build_project_adapter

REPO_ROOT = Path(__file__).resolve().parents[2]
COLLISION_INCLUDE = REPO_ROOT / "collision"
RUNNER_SOURCE = Path(__file__).with_name("cuda_collision_config_free_runner.cu")


def _robot(robot_id="iiwa14", base_mode="fixed"):
    for case in iter_robot_cases(MANIFEST_PATH, base_mode=base_mode):
        if case["spec"].robot_id == robot_id:
            spec = case["spec"]
            break
    else:
        pytest.skip(f"{robot_id}-{base_mode} not in manifest")
    try:
        resolved = resolve_robot_spec(spec)
    except RuntimeError as exc:
        pytest.skip(f"could not resolve {robot_id}: {exc}")
    return build_project_adapter(spec, resolved, base_mode=base_mode).robot


def _collision_spec(robot):
    """One tiny sphere per movable joint frame (anchor = jid), small nonzero offset so the
    extractor's offset epilogue is exercised; radii tiny so no self-collision at the test q."""
    # get_joints_ordered_by_id() returns MOVABLE joints only (fixed joints are removed by
    # remove_fixed_joints); one sphere per movable frame.
    anchors = [int(j.get_id()) for j in robot.get_joints_ordered_by_id()]
    offset, radius = [], []
    for k, _a in enumerate(anchors):
        offset.extend([0.02 + 0.005 * k, -0.01, 0.03])
        radius.append(0.01)
    return {"anchor": anchors, "offset": offset, "radius": radius,
            "self_cc_ranges": build_self_cc_ranges(robot, anchors)}


@pytest.mark.cuda_equivalence
@pytest.mark.developer_only
@pytest.mark.robot_smoke
def test_collision_config_free(tmp_path):
    robot = _robot()
    spec = _collision_spec(robot)
    build_dir = tmp_path / "collision_config_free"
    build_dir.mkdir()
    header = build_dir / "grid.cuh"
    codegen = GRiDCodeGenerator(robot, FILE_NAMESPACE="grid")
    with open(os.devnull, "w") as devnull, contextlib.redirect_stdout(devnull):
        codegen.gen_all_code(codegen_profile="all", output_path=str(header), collision_spec=spec)

    nvcc = shutil.which("nvcc")
    if nvcc is None:
        pytest.skip("nvcc not found; install CUDA Toolkit to run CUDA tests.")
    runner_copy = build_dir / RUNNER_SOURCE.name
    shutil.copyfile(RUNNER_SOURCE, runner_copy)
    arch = _detect_cuda_arch()
    exe = build_dir / "cuda_collision_config_free_runner.exe"
    cmd = [nvcc, "-std=c++17", "-O2", "-gencode", f"arch=compute_{arch},code=sm_{arch}",
           "-I", str(build_dir), "-I", str(COLLISION_INCLUDE), "-o", str(exe), str(runner_copy)]
    result = subprocess.run(cmd, cwd=build_dir, capture_output=True, text=True)
    if result.returncode != 0:
        pytest.fail(f"config_free runner compile FAILED.\ncmd: {' '.join(cmd)}\n"
                    f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}")

    run = subprocess.run([str(exe)], capture_output=True, text=True)
    assert run.returncode == 0, f"config_free runner FAILED:\nstdout:\n{run.stdout}\nstderr:\n{run.stderr}"
    assert run.stdout.strip().endswith("RESULT: PASS"), run.stdout
    print(run.stdout)
