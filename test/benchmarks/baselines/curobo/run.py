#!/usr/bin/env python3
"""Run cuRobo inverse-dynamics timing benchmark for one robot/base combination.

UNTESTED: there is no `curobo` package installed on this machine yet. This
adapter is modeled byte-for-byte on baselines/mjx/run.py and
baselines/mujoco_warp/run.py (the closest GPU analogs). It shells out to
`timeCurobo.py` and reuses the SAME shared helpers (parse_grid_output /
fill_nulls / build_metadata), so its JSON is drop-in compatible with
generate_report.py under the baseline key "curobo".

WHY cuRobo IS A TIMEABLE GPU BASELINE
-------------------------------------
The cuRobo author wrote an inverse-dynamics CUDA-kernel benchmark *specifically*
for this comparison:
    https://github.com/NVlabs/curobo/blob/main/benchmark/inverse_dynamics_kernel_benchmark.py
It builds `Dynamics` from a cuRobo robot YAML and times the RNEA-forward
(`compute_inverse_dynamics`) and RNEA-backward (the analytic gradient via
`torch.autograd.backward`) CUDA kernels with `torch.profiler`. So cuRobo is a
real GPU competitive baseline for inverse_dynamics + inverse_dynamics_gradient
(earlier we wrongly concluded it was paper-reference-only). `timeCurobo.py`
mirrors that benchmark's sync/measurement approach but reports in OUR label
format + batch sweep (16..256) so parse_grid_output can ingest it.

INSTALL (VERIFIED 2026-06-13 on RTX 5090 / sm_120 / torch 2.12-dev cu128):
    git clone --depth 1 https://github.com/NVlabs/curobo.git
    cd curobo && TORCH_CUDA_ARCH_LIST="12.0" pip install -e . --no-build-isolation
    pip install cuda-core   # REQUIRED: the _src dynamics backend imports cuda.core.LaunchConfig
  Notes from the real run: the editable install is FAST (kernels JIT on first use, no
  20-min precompile); git-lfs is NOT needed for g1 (its URDF is plain text in the repo);
  WITHOUT cuda-core the dynamics backend raises ModuleNotFoundError 'cuda.core' and every
  algo nulls. cuRobo's unitree_g1_29dof_retarget.yml loads **35 DOF** (not 29) — it does
  ~20% more work than our g1_29dof, so the comparison slightly favors cuRobo. Forward
  kinematics (END_EFFECTOR_POSE) currently nulls ('Tensor' has no attribute 'joint_names'
  — the Kinematics FK entry needs a JointState, not a bare tensor; id + id_du work + are
  the meaningful comparison). Measured g1 fixed N=256: id 173.9us, id_du 644.8us (with-mem)
  — GRiD-autotuned beats both (5.75x / 10.44x).

ROBOT COVERAGE (cuRobo content/configs/robot/*.yml)
---------------------------------------------------
cuRobo ships configs for: franka.yml, dual_ur10e.yml, ur10e.yml,
unitree_g1.yml, unitree_g1_29dof_retarget.yml, simple_mimic_robot.yml.
Of OUR three comparison robots (iiwa14, go2, g1):
  * g1     -> unitree_g1_29dof_retarget.yml  (URDF g1_29dof_rev_1_0.urdf, the
             29-DoF body without hands — matches our g1_29dof). unitree_g1.yml
             is the 29-DoF-with-hands (49 joints) variant; we use the retarget
             (body-only) config for an apples-to-apples 29-DoF comparison. Set
             CUROBO_G1_YML=unitree_g1.yml to switch.
  * iiwa14 -> N/A. cuRobo ships NO iiwa/kuka config -> all-null column.
  * go2    -> N/A. cuRobo ships NO go2 config         -> all-null column.

BASE AXIS (fixed vs floating)
-----------------------------
cuRobo's Kinematics roots the tree at `base_link` (no free-flyer joint); its
Dynamics models a FIXED base only. So for our fixed-vs-floating axis:
  * g1 fixed    -> cuRobo fixed-base dynamics (the meaningful comparison cell).
  * g1 floating -> cuRobo CANNOT represent a free-flyer base -> NULL the whole
                   column rather than mislabel a fixed-base number as floating.
This adapter emits a null column for base == floating (and for iiwa14/go2 any
base), and a real timed column only for g1 fixed.

ALGORITHM COVERAGE
------------------
  inverse_dynamics           -> dynamics.compute_inverse_dynamics(joint_state)
                                (RNEA forward kernel, "rnea_forward").
  inverse_dynamics_gradient  -> torch.autograd.backward(tau, ...) on the above
                                (RNEA backward kernel, "rnea_backward"). This is
                                the analytic d(tau)/d(q,qd,qdd) that GRiD's
                                inverse_dynamics_gradient also computes.
  end_effector_pose          -> cuRobo's forward-kinematics kernel
                                (kin.forward / compute_kinematics). The cuRobo
                                ID benchmark times a "forward kernel" too; FK is
                                exposed cleanly via Kinematics, so we time it as
                                end_effector_pose. ASSESS on first run whether
                                the FK kernel maps to our EE-pose semantics; null
                                it if not (see checklist).
All other GRiD algos -> null (cuRobo's dynamics surface only exposes RNEA
fwd/bwd; no forward_dynamics / crba / minv / SO / Jacobian dynamics kernels).

FIRST-RUN VALIDATION CHECKLIST (see docs/open-tasks/curobo_baseline_plan.md)
  * confirm dynamics.compute_inverse_dynamics(JointState(q,qd,qdd)) returns tau
    of shape (batch, dof) on cuda.
  * confirm the RNEA-backward (torch.autograd.backward) path is the gradient we
    mean: q/qd/qdd.grad populated, shape (batch, dof); compare a couple of
    columns vs GRiD's inverse_dynamics_gradient at the same state.
  * confirm setup_batch_size(batch_size=N) is required before each batch N.
  * sanity-check µs/iter vs mjx + grid (same robot, same N) — same ballpark or
    cuRobo faster on g1 id/id_grad (the author tuned g1).
  * confirm torch.cuda.synchronize() actually brackets the timed launch (no
    async enqueue leaking out of the timer).

Usage:
    python test/benchmarks/baselines/curobo/run.py \
        --robot g1 --base fixed [--output results/g1_fixed_curobo_<host>.json]
"""
from __future__ import annotations

import argparse
import json
import os
import platform
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[4]
THIS_DIR  = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO_ROOT))

from test.benchmarks.timing_parser import (  # noqa: E402
    parse_grid_output, fill_nulls, build_metadata,
)

# ---------------------------------------------------------------------------
# cuRobo-available algorithms (others -> null after fill_nulls).
# ---------------------------------------------------------------------------
CUROBO_ALGOS = ["inverse_dynamics", "inverse_dynamics_gradient", "end_effector_pose"]

# ---------------------------------------------------------------------------
# Our robot -> cuRobo robot-config YAML. Only robots cuRobo ships a config for
# appear here; iiwa14 and go2 have NO cuRobo config -> produce a null column.
# g1 -> the 29-DoF body-only retarget config (matches our g1_29dof). Override
# the g1 yml via env CUROBO_G1_YML (e.g. unitree_g1.yml for the 49-joint hand
# variant).
# ---------------------------------------------------------------------------
ROBOT_CUROBO_YML: dict[str, str | None] = {
    "g1":      os.environ.get("CUROBO_G1_YML", "unitree_g1_29dof_retarget.yml"),
    "iiwa14":  None,   # cuRobo ships no iiwa/kuka config
    "go2":     None,   # cuRobo ships no go2 config
    "h2_plus": None,   # H2+ is GRiD-internal; cuRobo ships no config -> null column
}

# cuRobo Dynamics models a FIXED base only (tree rooted at base_link, no
# free-flyer). The floating column cannot be represented -> null it.
SUPPORTED_BASES = {"fixed"}


# ---------------------------------------------------------------------------
# Run timeCurobo.py and parse its output
# ---------------------------------------------------------------------------
TIMING_SCRIPT = THIS_DIR / "timeCurobo.py"


def run_timing(curobo_yml: str, base: str,
               test_iters: int | None = None) -> str:
    floating_arg = "T" if base == "floating" else "F"
    cmd = [sys.executable, str(TIMING_SCRIPT), curobo_yml, floating_arg]
    env = os.environ.copy()
    if test_iters is not None:
        env["BENCH_TEST_ITERS"] = str(int(test_iters))
    result = subprocess.run(cmd, capture_output=True, text=True, env=env)
    if result.returncode != 0:
        raise RuntimeError(
            f"timeCurobo.py exited with code {result.returncode}:\n{result.stderr}"
        )
    return result.stdout + "\n" + result.stderr


# ---------------------------------------------------------------------------
# cuRobo-specific metadata from timeCurobo stdout
# ---------------------------------------------------------------------------
def _parse_curobo_metadata(stdout: str) -> dict[str, str]:
    meta: dict[str, str] = {}
    in_block = False
    for line in stdout.splitlines():
        if "=== BEGIN CUROBO METADATA ===" in line:
            in_block = True
            continue
        if "=== END CUROBO METADATA ===" in line:
            break
        if in_block and ":" in line:
            k, _, v = line.partition(":")
            key = k.strip().lower().replace(" ", "_")
            if key in ("curobo_version", "torch_version", "torch_device",
                       "robot_yml", "dof"):
                meta[key] = v.strip()
    return meta


def _null_column(robot: str, base: str, args, reason: str) -> None:
    """Emit an all-null curobo column (cell N/A) without crashing."""
    filled = fill_nulls({})  # every algo -> None
    meta = build_metadata(include_gpu=True, include_pinocchio=False)
    meta["robot"]      = robot
    meta["base"]       = base
    meta["curobo_na"]  = reason
    result = {"metadata": meta, "results": {robot: {base: {"curobo": filled}}}}
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(f"  [curobo] N/A ({reason}) — wrote null column: {args.output}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(description="Run cuRobo benchmark for one robot/base")
    parser.add_argument("--robot", required=True, choices=list(ROBOT_CUROBO_YML))
    parser.add_argument("--base", required=True, choices=["fixed", "floating"])
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--ee-frame", default=None,
                        help="Accepted for harness CLI parity; cuRobo FK writes all "
                             "link poses, so the EE frame is not separately sliced here.")
    parser.add_argument("--device", choices=["gpu"], default="gpu",
                        help="cuRobo is GPU-only (CUDA). Kept for CLI parity; only "
                             "'gpu' is valid.")
    parser.add_argument("--test-iters", type=int, default=None,
                        help="Override TEST_ITERS (default 500). Number of timed reps "
                             "per single-call or per batch size.")
    args = parser.parse_args()

    build_dir = REPO_ROOT / "test" / "benchmarks" / "results"
    build_dir.mkdir(parents=True, exist_ok=True)
    if args.output is None:
        host = platform.node().replace(" ", "_")
        args.output = build_dir / f"{args.robot}_{args.base}_curobo_{host}.json"

    curobo_yml = ROBOT_CUROBO_YML.get(args.robot)

    # --- N/A cells: no cuRobo config, or a base cuRobo can't represent. ---
    if curobo_yml is None:
        _null_column(args.robot, args.base, args,
                     reason=f"cuRobo ships no robot config for '{args.robot}'")
        return
    if args.base not in SUPPORTED_BASES:
        _null_column(args.robot, args.base, args,
                     reason="cuRobo Dynamics models a fixed base only "
                            "(no free-flyer); floating not representable")
        return

    print(f"[curobo] {args.robot} {args.base} — cuRobo yml: {curobo_yml}")
    print(f"  [curobo] running timeCurobo.py ...")

    try:
        output = run_timing(curobo_yml, args.base, test_iters=args.test_iters)
    except Exception as e:
        # A cuRobo import/build crash -> degrade to a null column (don't take
        # down the orchestrator), matching the mjx/mujoco_warp shell-out pattern.
        print(f"  [curobo] ERROR (writing null column): {e}", file=sys.stderr)
        _null_column(args.robot, args.base, args, reason=f"timeCurobo failed: {e}")
        return

    # parse_grid_output handles the same label format that timeCurobo.py emits.
    timings = parse_grid_output(output)
    # Zero out algos cuRobo doesn't support (so they appear as null, not absent).
    for algo in list(timings.keys()):
        if algo not in CUROBO_ALGOS:
            timings[algo] = None
    filled = fill_nulls(timings)

    meta = build_metadata(include_gpu=True, include_pinocchio=False)
    meta.update(_parse_curobo_metadata(output))
    meta["robot"]    = args.robot
    meta["base"]     = args.base
    meta["ee_frame"] = args.ee_frame or ""

    result = {"metadata": meta, "results": {args.robot: {args.base: {"curobo": filled}}}}
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(f"  [curobo] results saved: {args.output}")

    for algo, entry in sorted(filled.items()):
        if entry is None:
            print(f"    {algo}: null")
        elif "single_us" in entry:
            v = entry["single_us"]["mean"]
            print(f"    {algo}: {v:.2f}us (single)")


if __name__ == "__main__":
    main()
