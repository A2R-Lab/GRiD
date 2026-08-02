#!/usr/bin/env python3
"""GRiD bench LIBRARY (no CLI): header generation + the autotune tier/thread picker.

The per-exe bench cutover (2026-07) retired this module's monolithic timeGRiD_{batch,single}.cu path
and its `main()`. Timing now runs through `test/benchmarks/per_algo_bench.py` (per-algo TUs -> one
exe/process, RAM-safe + crash-isolated). This file is imported for its reusable pieces:

  * generate_header()          -- content-cached grid.cuh codegen (baked + runtime-param + multi-target)
  * PER_ALGO_SPECS             -- the single source of truth for each algo's bench call
  * _per_algo_batch_tu_source  -- the per-algo measure entry the wrapper compiles
  * _autotune_pick_winners + the tier/thread sweep + SASS-dedup helpers (the picker, reused VERBATIM)
  * tier caps, thread grids, kernel-symbol helpers

Consumers: per_algo_bench.py, collect_kernel_limits.py, build_autotune_matrix.py, autotune_ffi.py.
The autotune_best merge lives in config/sweep_to_autotune_best.py; the launch-config bake in
config/autotune_to_launch_config.py.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import contextlib
import hashlib
import io
import json
import os
import shutil
import subprocess
import sys
import time
from datetime import date
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "external"))  # peer submodules (RBDReference/URDFParser/GLASS)

from config import robot_urdf  # noqa: E402
from grid_codegen import GRiDCodeGenerator  # noqa: E402
from grid_codegen.algo_registry import ALGO_REGISTRY  # noqa: E402
from RBDReference.equivalents.reference_backend import strict_parse_robot  # noqa: E402
from test.benchmarks.timing_parser import (  # noqa: E402
    parse_grid_output, fill_nulls, build_metadata,
)

# ---------------------------------------------------------------------------
# Canonical EE frames per robot (fixed joint / link name used as generator target)
# These correspond to the last fixed-joint frame in each robot's kinematic chain.
# ---------------------------------------------------------------------------
DEFAULT_EE_FRAMES: dict[str, str] = {
    "iiwa14": "iiwa_joint_ee",    # fixed joint at EE of iiwa14 URDF
    "go2":    "FR_foot_joint",    # fixed joint at FR foot
    "g1":     "right_hand_palm_joint",  # fixed joint at right hand palm
    "h1_2":   "R_base_link_joint",      # fixed joint at base of right hand (h1_2)
    "h2_plus": "right_hand_joint",      # fixed joint at right hand (H2+ large-robot scaling target)
    "baxter":  "left_endpoint",         # fixed joint mounting the left gripper (dual-arm; single-EE convention)
}

# ---------------------------------------------------------------------------
# Robot URDF resolution via robot_descriptions
# ---------------------------------------------------------------------------
ROBOT_DESCRIPTION_MODULE: dict[str, str] = {
    "iiwa14": "robot_descriptions.iiwa14_description",
    "go2":    "robot_descriptions.go2_description",
    "g1":     "robot_descriptions.g1_description",
    "h1_2":   "robot_descriptions.h1_2_description",
}

# Robots vendored locally (not in robot_descriptions). H2+ = Unitree H2+, the large
# non-mimic humanoid (75-DOF fixed / 81-DOF floating) — the large-robot SCALING target
# replacing the now-deprecated h1_2. No mjx/frax/pinocchio/cuRobo model exists for it, so
# it is a GRiD-internal scaling study, not a competitive cell.
LOCAL_URDF: dict[str, str] = {
    "h2_plus": str(robot_urdf("h2_plus")),
    # Baxter = Rethink dual-arm (14-DOF actuated, fixed-base) — vendored URDF for
    # fixed-base benchmark variety. GRiD-internal (treated GRID_ONLY); single-EE
    # (left_endpoint) per the one-EE-per-robot convention (both-grippers backlog).
    "baxter":  str(robot_urdf("baxter")),
}


def robot_is_mimic(urdf_path: str) -> bool:
    """True if the URDF the sweep actually loads has any ``<mimic>`` joint.

    Data-driven mimic detection on the EXACT cached URDF (`get_urdf_path` →
    ~/.cache/robot_descriptions/...), parsed the same way the header generator
    parses it. Used to drop codegen-skipped families (com/ccrba/energy) for
    mimic robots — h1_2 IS mimic (12 <mimic> finger joints), iiwa14/go2/g1 are
    not. Mirrors GRiDCodeGenerator.helpers.robot_has_mimic_joints, which keys on
    the same per-joint `is_mimic` flag.
    """
    capture = io.StringIO()
    with contextlib.redirect_stdout(capture):
        robot_obj, _ = strict_parse_robot(urdf_path, floating_base=False)
    return any(getattr(j, "is_mimic", False) for j in robot_obj.joints)


def get_urdf_path(robot: str) -> str:
    if robot in LOCAL_URDF:
        path = LOCAL_URDF[robot]
        if not os.path.exists(path):
            raise RuntimeError(f"local URDF for '{robot}' not found at {path}")
        return path
    mod_name = ROBOT_DESCRIPTION_MODULE.get(robot)
    if mod_name is None:
        raise ValueError(f"Unknown robot '{robot}'. Known: "
                         f"{list(ROBOT_DESCRIPTION_MODULE) + list(LOCAL_URDF)}")
    import importlib
    mod = importlib.import_module(mod_name)
    path = getattr(mod, "URDF_PATH", None)
    if path is None:
        raise RuntimeError(f"robot_descriptions module {mod_name} has no URDF_PATH attribute")
    return str(path)


# ---------------------------------------------------------------------------
# CUDA arch detection
# ---------------------------------------------------------------------------
def detect_cuda_arch() -> str:
    env_arch = os.environ.get("GRID_CUDA_ARCH")
    if env_arch:
        return env_arch.replace(".", "")
    nvidia_smi = shutil.which("nvidia-smi")
    if nvidia_smi:
        result = subprocess.run(
            [nvidia_smi, "--query-gpu=compute_cap", "--format=csv,noheader,nounits"],
            capture_output=True, text=True,
        )
        if result.returncode == 0:
            for line in result.stdout.splitlines():
                cc = line.strip()
                if cc:
                    return cc.replace(".", "")
    return "86"


# ---------------------------------------------------------------------------
# Header generation with caching
# ---------------------------------------------------------------------------
def _hash_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _hash_file(path: Path) -> str:
    return _hash_bytes(path.read_bytes())


def _hash_tree(root: Path, suffixes: tuple[str, ...]) -> str:
    digest = hashlib.sha256()
    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        if "__pycache__" in path.parts:
            continue
        if path.suffix in suffixes:
            digest.update(path.read_bytes())
    return digest.hexdigest()


CACHE_ROOT = REPO_ROOT / ".pytest_cache" / "grid_cuda"


def generate_header(
    urdf_path: str,
    robot: str,
    base: str,
    ee_frame: str,
    build_dir: Path,
    no_recompile: bool = False,
    runtime_inertia: bool = False,
    runtime_transform: bool = False,
    runtime_joint_dynamics: bool = False,
    multi_target_from_collision: bool = False,
    emit_alloc_gating: bool = False,
    emit_workspace_chunking: bool = False,
) -> Path:
    """Generate grid.cuh for the given robot/base, using content-hash cache."""
    floating_base = (base == "floating")

    urdf_hash = _hash_file(Path(urdf_path))
    codegen_hash = _hash_tree(REPO_ROOT / "grid_codegen", (".py",))
    # GRID_NO_LICM_BARRIER suppresses the anti-LICM machinery in _single_timing
    # rep loops (volatile reload + __noinline__ barrier). When toggled, the
    # generated header changes — must bust the header cache.
    no_licm_barrier_env = os.environ.get("GRID_NO_LICM_BARRIER", "0")
    # GRID_BENCH_ALGORITHM_LIST (subset timing) changes the generated header, so it
    # MUST be part of the cache key — otherwise a cached full-set binary is served
    # and the requested subset is silently ignored.
    bench_algo_list_env = os.environ.get("GRID_BENCH_ALGORITHM_LIST", "")
    # RUNTIME-PARAM VARIANTS (hardware co-design A/B). Each of these sources a class of model
    # parameters from a MUTABLE device table instead of baking it as a compile-time literal --
    # keeping the SPARSITY PATTERN baked either way, so the only cost is losing value-folding.
    # They change the generated header, so they MUST be in the cache key: without this the baked
    # header is silently served for a "runtime" run and the A/B compares a header to ITSELF,
    # reporting a perfect (and completely fake) wash.
    cache_key = _hash_bytes(
        json.dumps({
            "urdf_hash": urdf_hash,
            "codegen_hash": codegen_hash,
            "robot": robot,
            "base": base,
            "runtime_inertia": runtime_inertia,
            "runtime_transform": runtime_transform,
            "runtime_joint_dynamics": runtime_joint_dynamics,
            "multi_target_from_collision": multi_target_from_collision,
            # 2a: per-algo alloc gating changes init_gridData's emitted guards, so it
            # MUST key the header cache (a gated exe compiled against an ungated
            # cached header would silently allocate everything again).
            "emit_alloc_gating": emit_alloc_gating,
            # chunked-workspace seam: emitting the chunk-capable wrappers changes the
            # header (macro-unset stays behavior-identical, but the bytes differ), so
            # it must key the cache like alloc gating does.
            "emit_workspace_chunking": emit_workspace_chunking,
            "profile": bench_algo_list_env or "all+frame_jacobian",
            "homogenous": True,
            "no_licm_barrier": no_licm_barrier_env,
            "idsva_so_world_frame": True,
            "enable_floating_second_order": True,
            # ee_frame intentionally excluded: not passed to gen_all_code
        }, sort_keys=True).encode()
    )[:24]

    header_path = build_dir / f"{robot}_{base}.cuh"
    cached_header = CACHE_ROOT / "headers" / cache_key / "grid.cuh"

    if no_recompile or (cached_header.exists() and not _recompile_requested()):
        if cached_header.exists():
            shutil.copyfile(cached_header, header_path)
            print(f"  [grid] header cache hit (key={cache_key[:12]})")
            return header_path

    print(f"  [grid] generating header for {robot}-{base} (cache key={cache_key[:12]})...")
    capture = io.StringIO()
    with contextlib.redirect_stdout(capture):
        robot_obj, _ = strict_parse_robot(urdf_path, floating_base=floating_base)
    codegen = GRiDCodeGenerator(
        robot_obj,
        DEBUG_MODE=False,
        NEED_PRINT_MAT=True,
        FILE_NAMESPACE="grid",
    )
    # multi_target_position's cost SCALES WITH THE BATCH SIZE, so a timing number is meaningless
    # without saying how many targets it was taken at. We use the robot's own COLLISION SPHERIZATION
    # as the batch: collision is multi_target's actual consumer, so this keeps the multi_target and
    # (future) config_free numbers describing the SAME geometry rather than two unrelated batches.
    # The batch size is baked into the JSON + printed, so the number is never quoted bare.
    mt_batch = None
    if multi_target_from_collision:
        from grid_codegen.algorithms._collision import collision_spec_from_urdf, normalize_collision_tiers
        with contextlib.redirect_stdout(io.StringIO()):
            spec = collision_spec_from_urdf(robot_obj, str(urdf_path), resolution=0.05)
        finest = normalize_collision_tiers(spec)[-1]
        mt_batch = [{"anchor_jid": int(a),
                     "offset": (finest["offset"][3 * i], finest["offset"][3 * i + 1], finest["offset"][3 * i + 2])}
                    for i, a in enumerate(finest["anchor"])]
        print(f"  [grid] multi_target batch from collision spherization: N={len(mt_batch)} targets")

    with contextlib.redirect_stdout(io.StringIO()):
        codegen.gen_all_code(
            # Runtime-param variants: default False => the baked path, byte-identical to before.
            runtime_inertia=runtime_inertia,
            runtime_transform=runtime_transform,
            runtime_joint_dynamics=runtime_joint_dynamics,
            # multi_target_batch and collision_spec are EXCLUSIVE (each defines NUM_MULTI_TARGETS).
            # We pass the BATCH, which is what emits the timeable kernels + hosts; config_free itself
            # is still a __device__ composite with no __global__ wrapper, so it is not timed here.
            multi_target_batch=mt_batch,
            include_homogenous_transforms=True,
            # fixed_target_name omitted: passing it with the 'all' set triggers a
            # generator bug where kinematics_only() references an _hessian_{name} variant
            # that isn't generated. EE pose timing is unaffected by this omission.
            output_path=str(header_path),
            # S1: the opt-in frame_jacobian family (frame_jacobian / _dot / osc_inertia)
            # is NOT in the default 'all' profile (so the default header stays
            # byte-identical), but the bench DOES want to time it. Request the 'all'
            # set PLUS the three opt-in keys via algorithm_list (which supersedes
            # codegen_profile) so their kernels emit and the PER_ALGO_SPECS rows fire.
            # GRID_BENCH_ALGORITHM_LIST (comma-separated) overrides for SUBSET timing
            # (e.g. =crba to time a single algo without paying the SO/gradient compile).
            algorithm_list=(
                [a.strip() for a in os.environ["GRID_BENCH_ALGORITHM_LIST"].split(",") if a.strip()]
                if os.environ.get("GRID_BENCH_ALGORITHM_LIST")
                else ["all", "frame_jacobian", "frame_jacobian_dot", "osc_inertia"]
            ),
            enable_idsva_so_world_frame=True,
            enable_floating_second_order=True,
            emit_alloc_gating=emit_alloc_gating,
            emit_workspace_chunking=emit_workspace_chunking,
        )

    cached_header.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(header_path, cached_header)
    print(f"  [grid] header generated: {header_path.name}")
    return header_path


def _recompile_requested() -> bool:
    return os.environ.get("GRID_BENCH_RECOMPILE", "0") == "1"


# ---------------------------------------------------------------------------
# Header generation with content-hash caching
# ---------------------------------------------------------------------------
THIS_DIR = Path(__file__).resolve().parent


# ---------------------------------------------------------------------------
# Per-algo call-site specs (PER_ALGO_SPECS) + the per-algo batch measure entry.
#
# PER_ALGO_SPECS is the single source of truth for how each algo's `grid::*`
# entrypoints get called. per_algo_bench.py compiles ONE self-contained TU per
# algo from `_per_algo_batch_tu_source` (below): a `measure_<algo>_batch_entry`
# that #includes the generated grid.cuh + timeGRiD_common.h and calls the
# matching `grid::*` template. Each becomes its own .exe/process (RAM-safe,
# crash-isolated). If you add an algo to ALGO_REGISTRY, add a matching row.
# ---------------------------------------------------------------------------

# Per-algo call-site specs. Each entry encodes the unique knowledge of how
# each algo's `grid::*` entrypoints get called, plus optional preprocessor
# gating and a runtime shared-memory skip guard for kernels that may not
# fit on the device.
#
# Fields:
#   single_call:  body for grid::*_single_timing<...>(...). Receives the
#                 macro args (hd_data, d_robotModel, GRAVITY, ITERS, BLOCK_DIM,
#                 dimms, streams) as appropriate per-algo.
#   batch_with_mem:    body for grid::*<...>(d,m,...,N,dim3(N,1,1),dimms,streams)
#   batch_compute_only: body for grid::*_compute_only<...>(d,m,...,N,dim3(N,1,1),dimms)
#   batch_label:  string passed to measure_batch_pair as the printf label.
#   gate:         optional preprocessor macro that must be 1 to compile the
#                 algo (e.g. GRID_HAS_FDSVA_SO). When None, always compiled.
#   shared_mem_skip:  optional name of a grid::* template that returns the
#                 dynamic shared-memory bytes the kernel needs. When set,
#                 the wrapper emits a runtime skip if grid_kernel_fits_device
#                 reports the kernel can't fit.
#
# The codegen calls below match the monolithic TUs verbatim (see
# timeGRiD_single.cu and timeGRiD_batch.cu). Do NOT freelance template-arg
# patterns here — keep them in lockstep.
PER_ALGO_SPECS: dict[str, dict] = {
    "inverse_dynamics": {
        "single_call":        "grid::inverse_dynamics_single_timing<float,false,true>(hd_data,d_robotModel,GRAVITY,SINGLE_CALL_ITERS_GLOBAL,dim3(1,1,1),dimms,streams)",
        "batch_with_mem":     "grid::inverse_dynamics<float,false,true>(d,m,GRAVITY,N,dim3(N,1,1),dimms,streams)",
        "batch_compute_only": "grid::inverse_dynamics_compute_only<float,false,true>(d,m,GRAVITY,N,dim3(N,1,1),dimms)",
        "batch_label": "INVERSE_DYNAMICS",
        "gate": None,
        "shared_mem_skip": "INVERSE_DYNAMICS_DYNAMIC_SHARED_MEM_BYTES",
    },
    "minv": {
        "single_call":        "grid::minv_single_timing<float,true>(hd_data,d_robotModel,SINGLE_CALL_ITERS_GLOBAL,dim3(1,1,1),dimms,streams)",
        "batch_with_mem":     "grid::minv<float,true>(d,m,N,dim3(N,1,1),dimms,streams)",
        "batch_compute_only": "grid::minv_compute_only<float,true>(d,m,N,dim3(N,1,1),dimms)",
        "batch_label": "Minv",
        "gate": None,
        "shared_mem_skip": "MINV_DYNAMIC_SHARED_MEM_BYTES",
    },
    "forward_dynamics": {
        "single_call":        "grid::forward_dynamics_single_timing<float>(hd_data,d_robotModel,GRAVITY,SINGLE_CALL_ITERS_GLOBAL,dim3(1,1,1),dimms,streams)",
        "batch_with_mem":     "grid::forward_dynamics<float>(d,m,GRAVITY,N,dim3(N,1,1),dimms,streams)",
        "batch_compute_only": "grid::forward_dynamics_compute_only<float>(d,m,GRAVITY,N,dim3(N,1,1),dimms)",
        "batch_label": "FORWARD_DYNAMICS",
        "gate": None,
        "shared_mem_skip": "FORWARD_DYNAMICS_DYNAMIC_SHARED_MEM_BYTES",
    },
    "aba": {
        "single_call":        "grid::aba_single_timing<float>(hd_data,d_robotModel,GRAVITY,SINGLE_CALL_ITERS_GLOBAL,dim3(1,1,1),dimms,streams)",
        "batch_with_mem":     "grid::aba<float>(d,m,GRAVITY,N,dim3(N,1,1),dimms,streams)",
        "batch_compute_only": "grid::aba_compute_only<float>(d,m,GRAVITY,N,dim3(N,1,1),dimms)",
        "batch_label": "ABA",
        "gate": None,
        "shared_mem_skip": "ABA_DYNAMIC_SHARED_MEM_BYTES",
    },
    "crba": {
        "single_call":        "grid::crba_single_timing<float>(hd_data,d_robotModel,GRAVITY,SINGLE_CALL_ITERS_GLOBAL,dim3(1,1,1),dimms,streams)",
        "batch_with_mem":     "grid::crba<float>(d,m,GRAVITY,N,dim3(N,1,1),dimms,streams)",
        "batch_compute_only": "grid::crba_compute_only<float>(d,m,GRAVITY,N,dim3(N,1,1),dimms)",
        "batch_label": "CRBA",
        "gate": None,
        "shared_mem_skip": "CRBA_DYNAMIC_SHARED_MEM_BYTES",
    },
    "inverse_dynamics_gradient": {
        "single_call":        "grid::inverse_dynamics_gradient_single_timing<float,false,true>(hd_data,d_robotModel,GRAVITY,SINGLE_CALL_ITERS_GLOBAL,dim3(1,1,1),dimms,streams)",
        "batch_with_mem":     "grid::inverse_dynamics_gradient<float,false,true>(d,m,GRAVITY,N,dim3(N,1,1),dimms,streams)",
        "batch_compute_only": "grid::inverse_dynamics_gradient_compute_only<float,false,true>(d,m,GRAVITY,N,dim3(N,1,1),dimms)",
        "batch_label": "INVERSE_DYNAMICS_GRADIENT",
        "gate": None,
        "shared_mem_skip": "INVERSE_DYNAMICS_GRADIENT_DYNAMIC_SHARED_MEM_BYTES",
    },
    "forward_dynamics_gradient": {
        "single_call":        "grid::forward_dynamics_gradient_single_timing<float,false>(hd_data,d_robotModel,GRAVITY,SINGLE_CALL_ITERS_GLOBAL,dim3(1,1,1),dimms,streams)",
        "batch_with_mem":     "grid::forward_dynamics_gradient<float,false>(d,m,GRAVITY,N,dim3(N,1,1),dimms,streams)",
        "batch_compute_only": "grid::forward_dynamics_gradient_compute_only<float,false>(d,m,GRAVITY,N,dim3(N,1,1),dimms)",
        "batch_label": "FORWARD_DYNAMICS_GRADIENT",
        "gate": None,
        "shared_mem_skip": "FORWARD_DYNAMICS_GRADIENT_DYNAMIC_SHARED_MEM_BYTES",
    },
    # f_ext gradients (A1). Host wrappers write into gridData's d_dtau_dfext /
    # d_dqdd_dfext / d_f_ext_gradient_dq buffers (allocated in gen_init_gridData), so
    # the call convention matches the standard (hd_data, d_robotModel, N, ...)
    # shape — no gravity arg (RNEA bias is folded into the kernel) and no extra
    # caller buffer. See grid_codegen/algorithms/_f_ext_gradient.py:
    # gen_f_ext_gradient_host (mode 0/1/2) and gen_f_ext_gradient_dq_host.
    "f_ext_gradient": {
        "single_call":        "grid::f_ext_gradient_single_timing<float>(hd_data,d_robotModel,SINGLE_CALL_ITERS_GLOBAL,dim3(1,1,1),dimms,streams)",
        "batch_with_mem":     "grid::f_ext_gradient<float>(d,m,N,dim3(N,1,1),dimms,streams)",
        "batch_compute_only": "grid::f_ext_gradient_compute_only<float>(d,m,N,dim3(N,1,1),dimms)",
        "batch_label": "F_EXT_GRADIENT",
        "gate": None,
        "shared_mem_skip": "F_EXT_GRADIENT_DYNAMIC_SHARED_MEM_BYTES",
    },
    "f_ext_gradient_dq": {
        "single_call":        "grid::f_ext_gradient_dq_single_timing<float>(hd_data,d_robotModel,SINGLE_CALL_ITERS_GLOBAL,dim3(1,1,1),dimms,streams)",
        "batch_with_mem":     "grid::f_ext_gradient_dq<float>(d,m,N,dim3(N,1,1),dimms,streams)",
        "batch_compute_only": "grid::f_ext_gradient_dq_compute_only<float>(d,m,N,dim3(N,1,1),dimms)",
        "batch_label": "F_EXT_GRADIENT_DQ",
        "gate": None,
        "shared_mem_skip": "F_EXT_GRADIENT_DQ_DYNAMIC_SHARED_MEM_BYTES",
    },
    # Joint-torque regressor (A1). The grid:: symbol AND registry key are now both
    # `inverse_dynamics_regressor`; R2: its output buffer `d_Y` is now part of
    # gridData (hd_data->d_Y), so the bench no longer allocates a TU-static buffer
    # — the host wrapper reads/writes hd_data->d_Y directly. It takes the gravity
    # arg (RNEA forward sweep). See _regressor.py:gen_inverse_dynamics_regressor_host.
    "inverse_dynamics_regressor": {
        "single_call":        "grid::inverse_dynamics_regressor_single_timing<float>(hd_data,d_robotModel,GRAVITY,SINGLE_CALL_ITERS_GLOBAL,dim3(1,1,1),dimms,streams)",
        "batch_with_mem":     "grid::inverse_dynamics_regressor<float>(d,m,GRAVITY,N,dim3(N,1,1),dimms,streams)",
        "batch_compute_only": "grid::inverse_dynamics_regressor_compute_only<float>(d,m,GRAVITY,N,dim3(N,1,1),dimms)",
        "batch_label": "INVERSE_DYNAMICS_REGRESSOR",
        "gate": None,
        "shared_mem_skip": "INVERSE_DYNAMICS_REGRESSOR_DYNAMIC_SHARED_MEM_BYTES",
    },
    # FD parameter gradient dqdd/dpi = -Minv.Y (A1). grid:: symbol AND registry key
    # are now both `forward_dynamics_parameter_gradient`; R2: its output buffer
    # `d_dqdd_dpi` is now part of gridData (hd_data->d_dqdd_dpi), so the bench no
    # longer allocates a TU-static buffer. It takes the gravity arg. See
    # _regressor.py:gen_forward_dynamics_parameter_gradient_host.
    "forward_dynamics_parameter_gradient": {
        "single_call":        "grid::forward_dynamics_parameter_gradient_single_timing<float>(hd_data,d_robotModel,GRAVITY,SINGLE_CALL_ITERS_GLOBAL,dim3(1,1,1),dimms,streams)",
        "batch_with_mem":     "grid::forward_dynamics_parameter_gradient<float>(d,m,GRAVITY,N,dim3(N,1,1),dimms,streams)",
        "batch_compute_only": "grid::forward_dynamics_parameter_gradient_compute_only<float>(d,m,GRAVITY,N,dim3(N,1,1),dimms)",
        "batch_label": "FORWARD_DYNAMICS_PARAMETER_GRADIENT",
        "gate": None,
        "shared_mem_skip": "FORWARD_DYNAMICS_PARAMETER_GRADIENT_DYNAMIC_SHARED_MEM_BYTES",
    },
    "end_effector_pose": {
        "single_call":        "grid::end_effector_pose_single_timing<float>(hd_data,d_robotModel,SINGLE_CALL_ITERS_GLOBAL,dim3(1,1,1),dimms,streams)",
        "batch_with_mem":     "grid::end_effector_pose<float>(d,m,N,dim3(N,1,1),dimms,streams)",
        "batch_compute_only": "grid::end_effector_pose_compute_only<float>(d,m,N,dim3(N,1,1),dimms)",
        "batch_label": "END_EFFECTOR_POSE",
        "gate": None,
        "shared_mem_skip": "END_EFFECTOR_POSE_DYNAMIC_SHARED_MEM_BYTES",
    },
    # multi_target_position{,_gradient}: kernels + hosts landed in Inc4b and are registered in
    # algo_registry, but were never wired into the bench harness -- so they were fully emitted, fully
    # correct, and invisible to timing. These rows close that. They are GATED, so a robot generated
    # WITHOUT a target batch simply compiles the blocks out (the default competitive robots are
    # unaffected); pass --multi-target-from-collision to make them fire.
    # ⚠ The cost SCALES WITH THE BATCH SIZE (N targets), so these numbers are only meaningful next to
    # the target count -- which is why the count is recorded in the JSON. No competitor has a
    # counterpart, so this is a CAPABILITY-LEAD cell (like config_free), not a W/L.
    "multi_target_position": {
        "single_call":        "grid::multi_target_position_single_timing<float>(hd_data,d_robotModel,SINGLE_CALL_ITERS_GLOBAL,dim3(1,1,1),dimms,streams)",
        "batch_with_mem":     "grid::multi_target_position<float>(d,m,N,dim3(N,1,1),dimms,streams)",
        "batch_compute_only": "grid::multi_target_position_compute_only<float>(d,m,N,dim3(N,1,1),dimms)",
        "batch_label": "MULTI_TARGET_POSITION",
        "gate": "GRID_HAS_MULTI_TARGET_POSITION",
        "shared_mem_skip": "MULTI_TARGET_POSITION_DYNAMIC_SHARED_MEM_BYTES",
    },
    "multi_target_position_gradient": {
        "single_call":        "grid::multi_target_position_gradient_single_timing<float>(hd_data,d_robotModel,SINGLE_CALL_ITERS_GLOBAL,dim3(1,1,1),dimms,streams)",
        "batch_with_mem":     "grid::multi_target_position_gradient<float>(d,m,N,dim3(N,1,1),dimms,streams)",
        "batch_compute_only": "grid::multi_target_position_gradient_compute_only<float>(d,m,N,dim3(N,1,1),dimms)",
        "batch_label": "MULTI_TARGET_POSITION_GRADIENT",
        "gate": "GRID_HAS_MULTI_TARGET_POSITION",
        "shared_mem_skip": "MULTI_TARGET_POSITION_GRADIENT_DYNAMIC_SHARED_MEM_BYTES",
    },
    "end_effector_pose_gradient": {
        "single_call":        "grid::end_effector_pose_gradient_single_timing<float>(hd_data,d_robotModel,SINGLE_CALL_ITERS_GLOBAL,dim3(1,1,1),dimms,streams)",
        "batch_with_mem":     "grid::end_effector_pose_gradient<float>(d,m,N,dim3(N,1,1),dimms,streams)",
        "batch_compute_only": "grid::end_effector_pose_gradient_compute_only<float>(d,m,N,dim3(N,1,1),dimms)",
        "batch_label": "END_EFFECTOR_POSE_GRADIENT",
        "gate": None,
        "shared_mem_skip": "END_EFFECTOR_POSE_GRADIENT_DYNAMIC_SHARED_MEM_BYTES",
    },
    "frame_jacobian": {
        "single_call":        "grid::frame_jacobian_single_timing<float>(hd_data,d_robotModel,SINGLE_CALL_ITERS_GLOBAL,dim3(1,1,1),dimms,streams)",
        "batch_with_mem":     "grid::frame_jacobian<float>(d,m,N,dim3(N,1,1),dimms,streams)",
        "batch_compute_only": "grid::frame_jacobian_compute_only<float>(d,m,N,dim3(N,1,1),dimms)",
        "batch_label": "FRAME_JACOBIAN",
        "gate": "GRID_HAS_FRAME_JACOBIAN",
        "shared_mem_skip": "FRAME_JACOBIAN_DYNAMIC_SHARED_MEM_BYTES",
    },
    "frame_jacobian_dot": {
        "single_call":        "grid::frame_jacobian_dot_single_timing<float>(hd_data,d_robotModel,SINGLE_CALL_ITERS_GLOBAL,dim3(1,1,1),dimms,streams)",
        "batch_with_mem":     "grid::frame_jacobian_dot<float>(d,m,N,dim3(N,1,1),dimms,streams)",
        "batch_compute_only": "grid::frame_jacobian_dot_compute_only<float>(d,m,N,dim3(N,1,1),dimms)",
        "batch_label": "FRAME_JACOBIAN_DOT",
        "gate": "GRID_HAS_FRAME_JACOBIAN_DOT",
        "shared_mem_skip": "FRAME_JACOBIAN_DOT_DYNAMIC_SHARED_MEM_BYTES",
    },
    "osc_inertia": {
        "single_call":        "grid::osc_inertia_single_timing<float>(hd_data,d_robotModel,SINGLE_CALL_ITERS_GLOBAL,dim3(1,1,1),dimms,streams)",
        "batch_with_mem":     "grid::osc_inertia<float>(d,m,N,dim3(N,1,1),dimms,streams)",
        "batch_compute_only": "grid::osc_inertia_compute_only<float>(d,m,N,dim3(N,1,1),dimms)",
        "batch_label": "OSC_INERTIA",
        "gate": "GRID_HAS_OSC_INERTIA",
        "shared_mem_skip": "OSC_INERTIA_DYNAMIC_SHARED_MEM_BYTES",
    },
    "end_effector_pose_hessian": {
        "single_call":        "grid::end_effector_pose_hessian_single_timing<float>(hd_data,d_robotModel,SINGLE_CALL_ITERS_GLOBAL,dim3(1,1,1),dimms,streams)",
        "batch_with_mem":     "grid::end_effector_pose_hessian<float>(d,m,N,dim3(N,1,1),dimms,streams)",
        "batch_compute_only": "grid::end_effector_pose_hessian_compute_only<float>(d,m,N,dim3(N,1,1),dimms)",
        "batch_label": "END_EFFECTOR_POSE_HESSIAN",
        "gate": None,
        "shared_mem_skip": "END_EFFECTOR_POSE_HESSIAN_DYNAMIC_SHARED_MEM_BYTES",
    },
    "idsva_so": {
        "single_call":        "grid::idsva_so_single_timing<float>(hd_data,d_robotModel,GRAVITY,SINGLE_CALL_ITERS_GLOBAL,dim3(1,1,1),dimms,streams)",
        "batch_with_mem":     "grid::idsva_so<float>(d,m,GRAVITY,N,dim3(N,1,1),dimms,streams)",
        "batch_compute_only": "grid::idsva_so_compute_only<float>(d,m,GRAVITY,N,dim3(N,1,1),dimms)",
        "batch_label": "IDSVA_SO",
        "gate": "GRID_HAS_IDSVA_SO",
        # grid::idsva_so forwards at CODEGEN time (world for floating/spherical/
        # high-DOF fixed) — probe the DISPATCHED kernel's bytes. Probing the body
        # frame on a floating header false-skips: the floating body frame is a
        # no-ladder diagnostic (242 KB on h2_plus) the dispatcher never launches,
        # while the dispatched world frame fits at every tier. Headers predating
        # the macro fall back to the body probe (identical to the old behavior).
        "shared_mem_skip": "IDSVA_SO_BODY_FRAME_DYNAMIC_SHARED_MEM_BYTES",
        "shared_mem_skip_dispatch": ("GRID_IDSVA_SO_DISPATCHES_WORLD_FRAME",
                                     "IDSVA_SO_WORLD_FRAME_DYNAMIC_SHARED_MEM_BYTES"),
    },
    "idsva_so_body_frame": {
        "single_call":        "grid::idsva_so_body_frame_single_timing<float>(hd_data,d_robotModel,GRAVITY,SINGLE_CALL_ITERS_GLOBAL,dim3(1,1,1),dimms,streams)",
        "batch_with_mem":     "grid::idsva_so_body_frame<float>(d,m,GRAVITY,N,dim3(N,1,1),dimms,streams)",
        "batch_compute_only": "grid::idsva_so_body_frame_compute_only<float>(d,m,GRAVITY,N,dim3(N,1,1),dimms)",
        "batch_label": "IDSVA_SO_BODY_FRAME",
        "gate": "GRID_HAS_IDSVA_SO_BODY_FRAME",
        "shared_mem_skip": "IDSVA_SO_BODY_FRAME_DYNAMIC_SHARED_MEM_BYTES",
    },
    "idsva_so_world_frame": {
        "single_call":        "grid::idsva_so_world_frame_single_timing<float>(hd_data,d_robotModel,GRAVITY,SINGLE_CALL_ITERS_GLOBAL,dim3(1,1,1),dimms,streams)",
        "batch_with_mem":     "grid::idsva_so_world_frame<float>(d,m,GRAVITY,N,dim3(N,1,1),dimms,streams)",
        "batch_compute_only": "grid::idsva_so_world_frame_compute_only<float>(d,m,GRAVITY,N,dim3(N,1,1),dimms)",
        "batch_label": "IDSVA_SO_WORLD_FRAME",
        "gate": "GRID_HAS_IDSVA_SO_WORLD_FRAME",
        "shared_mem_skip": "IDSVA_SO_WORLD_FRAME_DYNAMIC_SHARED_MEM_BYTES",
    },
    "fdsva_so": {
        "single_call":        "grid::fdsva_so_single_timing<float>(hd_data,d_robotModel,GRAVITY,SINGLE_CALL_ITERS_GLOBAL,dim3(1,1,1),dimms,streams)",
        "batch_with_mem":     "grid::fdsva_so<float>(d,m,GRAVITY,N,dim3(N,1,1),dimms,streams)",
        "batch_compute_only": "grid::fdsva_so_compute_only<float>(d,m,GRAVITY,N,dim3(N,1,1),dimms)",
        "batch_label": "FDSVA_SO",
        "gate": "GRID_HAS_FDSVA_SO",
        "shared_mem_skip": "FDSVA_SO_DYNAMIC_SHARED_MEM_BYTES",
    },
    # --- mjx-convention timing twins (B4) -------------------------------------------------
    # Time the MUJOCO_OUTPUT=true kernel instantiation (the previously-missing mjx
    # second-order/gradient bars). Emitted ONLY on floating non-mimic robots, so gate on
    # GRID_RBD_WITH_MUJOCO (defined =1 in those headers, absent otherwise) so a fixed/mimic
    # header still compiles the bench with the mjx rows gated out. Key "<algo>_mjx" auto-
    # wires the attr: _attr_init_call emits grid::init_grid_kernel_attr_<algo>_mjx<float>().
    # Pair each with its pin row for a clean mjx-vs-pin A/B.
    "idsva_so_world_frame_mjx": {
        "single_call":        "grid::idsva_so_world_frame_single_timing<float,grid::GRID_DATA_ALL,true>(hd_data,d_robotModel,GRAVITY,SINGLE_CALL_ITERS_GLOBAL,dim3(1,1,1),dimms,streams)",
        "batch_with_mem":     "grid::idsva_so_world_frame<float,grid::GRID_DATA_ALL,true>(d,m,GRAVITY,N,dim3(N,1,1),dimms,streams)",
        "batch_compute_only": "grid::idsva_so_world_frame_compute_only<float,grid::GRID_DATA_ALL,true>(d,m,GRAVITY,N,dim3(N,1,1),dimms)",
        "batch_label": "IDSVA_SO_WORLD_FRAME(mjx)",
        "gate": "GRID_HAS_IDSVA_SO_WORLD_FRAME && GRID_RBD_WITH_MUJOCO",
        "shared_mem_skip": "IDSVA_SO_WORLD_FRAME_DYNAMIC_SHARED_MEM_BYTES",
    },
    "fdsva_so_mjx": {
        "single_call":        "grid::fdsva_so_single_timing<float,grid::GRID_DATA_ALL,true>(hd_data,d_robotModel,GRAVITY,SINGLE_CALL_ITERS_GLOBAL,dim3(1,1,1),dimms,streams)",
        "batch_with_mem":     "grid::fdsva_so<float,grid::GRID_DATA_ALL,true>(d,m,GRAVITY,N,dim3(N,1,1),dimms,streams)",
        "batch_compute_only": "grid::fdsva_so_compute_only<float,grid::GRID_DATA_ALL,true>(d,m,GRAVITY,N,dim3(N,1,1),dimms)",
        "batch_label": "FDSVA_SO(mjx)",
        "gate": "GRID_HAS_FDSVA_SO && GRID_RBD_WITH_MUJOCO",
        "shared_mem_skip": "FDSVA_SO_DYNAMIC_SHARED_MEM_BYTES",
    },
    "inverse_dynamics_gradient_mjx": {
        "single_call":        "grid::inverse_dynamics_gradient_single_timing<float,false,true,grid::GRID_DATA_ALL,true>(hd_data,d_robotModel,GRAVITY,SINGLE_CALL_ITERS_GLOBAL,dim3(1,1,1),dimms,streams)",
        "batch_with_mem":     "grid::inverse_dynamics_gradient<float,false,true,grid::GRID_DATA_ALL,true>(d,m,GRAVITY,N,dim3(N,1,1),dimms,streams)",
        "batch_compute_only": "grid::inverse_dynamics_gradient_compute_only<float,false,true,grid::GRID_DATA_ALL,true>(d,m,GRAVITY,N,dim3(N,1,1),dimms)",
        "batch_label": "INVERSE_DYNAMICS_GRADIENT(mjx)",
        "gate": "GRID_RBD_WITH_MUJOCO",
        "shared_mem_skip": "INVERSE_DYNAMICS_GRADIENT_DYNAMIC_SHARED_MEM_BYTES",
    },
    "forward_dynamics_gradient_mjx": {
        "single_call":        "grid::forward_dynamics_gradient_single_timing<float,false,grid::GRID_DATA_ALL,true>(hd_data,d_robotModel,GRAVITY,SINGLE_CALL_ITERS_GLOBAL,dim3(1,1,1),dimms,streams)",
        "batch_with_mem":     "grid::forward_dynamics_gradient<float,false,grid::GRID_DATA_ALL,true>(d,m,GRAVITY,N,dim3(N,1,1),dimms,streams)",
        "batch_compute_only": "grid::forward_dynamics_gradient_compute_only<float,false,grid::GRID_DATA_ALL,true>(d,m,GRAVITY,N,dim3(N,1,1),dimms)",
        "batch_label": "FORWARD_DYNAMICS_GRADIENT(mjx)",
        "gate": "GRID_RBD_WITH_MUJOCO",
        "shared_mem_skip": "FORWARD_DYNAMICS_GRADIENT_DYNAMIC_SHARED_MEM_BYTES",
    },
    # Centroidal / energy quick-wins (A1). Host wrappers write into gridData
    # buffers (d_c for the RNEA-bias families; d_com / d_ccrba / d_energy).
    #   - generalized_gravity / nonlinear_effects: RNEA-bias wrappers, take the
    #     gravity arg; signature mirrors `id` + gravity. Emitted whenever `id`
    #     is generated (always, under codegen_profile='all'). Shared smem macro
    #     is INVERSE_DYNAMICS_BIAS_DYNAMIC_SHARED_MEM_BYTES for BOTH.
    #     See grid_codegen/algorithms/_centroidal.py:gen_id_bias_host.
    #   - com: kinematics-domain, NO gravity / NO qd. ccrba: NO gravity (uses qd).
    #     energy: takes the gravity arg (uses qd). Output sizes: com=3+3*NUM_VEL,
    #     ccrba=6*NUM_VEL+6, energy=3.  See _centroidal.py:_gen_kin_centroidal_host.
    # NOTE: com/ccrba/energy are SKIPPED at codegen for MIMIC robots (their
    # per-body Jacobian fold is not mimic-reduced) — for a mimic robot these
    # grid:: symbols are absent and the TU would fail to compile. There is no
    # GRID_HAS_* preprocessor macro emitted for these families to #if-gate on,
    # so the bench DROPS them in Python for mimic robots (MIMIC_UNSUPPORTED_ALGOS,
    # filtered by _algo_keys_in_registry_order via the data-driven has_mimic flag
    # from robot_is_mimic()). Mimic status of the sweep robots (checked on the
    # exact loaded URDF): iiwa14=NO, go2=NO, g1=NO, h1_2=YES (12 <mimic> finger
    # joints) — i.e. h1_2 IS mimic, so for h1_2 these three rows are skipped.
    # (generalized_gravity / nonlinear_effects always emit — they reuse the
    # mimic-aware RNEA inner — and are never dropped.)
    "generalized_gravity": {
        "single_call":        "grid::generalized_gravity_single_timing<float>(hd_data,d_robotModel,GRAVITY,SINGLE_CALL_ITERS_GLOBAL,dim3(1,1,1),dimms,streams)",
        "batch_with_mem":     "grid::generalized_gravity<float>(d,m,GRAVITY,N,dim3(N,1,1),dimms,streams)",
        "batch_compute_only": "grid::generalized_gravity_compute_only<float>(d,m,GRAVITY,N,dim3(N,1,1),dimms)",
        "batch_label": "GENERALIZED_GRAVITY",
        "gate": None,
        "shared_mem_skip": "INVERSE_DYNAMICS_BIAS_DYNAMIC_SHARED_MEM_BYTES",
    },
    "nonlinear_effects": {
        "single_call":        "grid::nonlinear_effects_single_timing<float>(hd_data,d_robotModel,GRAVITY,SINGLE_CALL_ITERS_GLOBAL,dim3(1,1,1),dimms,streams)",
        "batch_with_mem":     "grid::nonlinear_effects<float>(d,m,GRAVITY,N,dim3(N,1,1),dimms,streams)",
        "batch_compute_only": "grid::nonlinear_effects_compute_only<float>(d,m,GRAVITY,N,dim3(N,1,1),dimms)",
        "batch_label": "NONLINEAR_EFFECTS",
        "gate": None,
        "shared_mem_skip": "INVERSE_DYNAMICS_BIAS_DYNAMIC_SHARED_MEM_BYTES",
    },
    "energy": {
        "single_call":        "grid::energy_single_timing<float>(hd_data,d_robotModel,GRAVITY,SINGLE_CALL_ITERS_GLOBAL,dim3(1,1,1),dimms,streams)",
        "batch_with_mem":     "grid::energy<float>(d,m,GRAVITY,N,dim3(N,1,1),dimms,streams)",
        "batch_compute_only": "grid::energy_compute_only<float>(d,m,GRAVITY,N,dim3(N,1,1),dimms)",
        "batch_label": "ENERGY",
        "gate": None,
        "shared_mem_skip": "ENERGY_DYNAMIC_SHARED_MEM_BYTES",
    },
    "com": {
        "single_call":        "grid::com_single_timing<float>(hd_data,d_robotModel,SINGLE_CALL_ITERS_GLOBAL,dim3(1,1,1),dimms,streams)",
        "batch_with_mem":     "grid::com<float>(d,m,N,dim3(N,1,1),dimms,streams)",
        "batch_compute_only": "grid::com_compute_only<float>(d,m,N,dim3(N,1,1),dimms)",
        "batch_label": "COM",
        "gate": None,
        "shared_mem_skip": "COM_DYNAMIC_SHARED_MEM_BYTES",
    },
    "ccrba": {
        "single_call":        "grid::ccrba_single_timing<float>(hd_data,d_robotModel,SINGLE_CALL_ITERS_GLOBAL,dim3(1,1,1),dimms,streams)",
        "batch_with_mem":     "grid::ccrba<float>(d,m,N,dim3(N,1,1),dimms,streams)",
        "batch_compute_only": "grid::ccrba_compute_only<float>(d,m,N,dim3(N,1,1),dimms)",
        "batch_label": "CCRBA",
        "gate": None,
        "shared_mem_skip": "CCRBA_DYNAMIC_SHARED_MEM_BYTES",
    },
    # Time integrators. Host signatures take an extra `dt` (const T) between
    # `gravity` and `num_timesteps`; IntegratorType defaults to EULER and gridData
    # already allocates the integrator I/O (d_x_kp1, d_dAB). A fixed bench dt is
    # used (its value doesn't affect timing).
    "integrator": {
        "single_call":        "grid::integrator_single_timing<float>(hd_data,d_robotModel,GRAVITY,static_cast<float>(0.01),SINGLE_CALL_ITERS_GLOBAL,dim3(1,1,1),dimms,streams)",
        "batch_with_mem":     "grid::integrator<float>(d,m,GRAVITY,static_cast<float>(0.01),N,dim3(N,1,1),dimms,streams)",
        "batch_compute_only": "grid::integrator_compute_only<float>(d,m,GRAVITY,static_cast<float>(0.01),N,dim3(N,1,1),dimms)",
        "batch_label": "INTEGRATOR",
        "gate": "GRID_HAS_INTEGRATOR",
        "shared_mem_skip": "INTEGRATOR_DYNAMIC_SHARED_MEM_BYTES",
    },
    "integrator_gradient": {
        "single_call":        "grid::integrator_gradient_single_timing<float>(hd_data,d_robotModel,GRAVITY,static_cast<float>(0.01),SINGLE_CALL_ITERS_GLOBAL,dim3(1,1,1),dimms,streams)",
        "batch_with_mem":     "grid::integrator_gradient<float>(d,m,GRAVITY,static_cast<float>(0.01),N,dim3(N,1,1),dimms,streams)",
        "batch_compute_only": "grid::integrator_gradient_compute_only<float>(d,m,GRAVITY,static_cast<float>(0.01),N,dim3(N,1,1),dimms)",
        "batch_label": "INTEGRATOR_GRADIENT",
        "gate": "GRID_HAS_INTEGRATOR_GRADIENT",
        "shared_mem_skip": "INTEGRATOR_DU_DYNAMIC_SHARED_MEM_BYTES",
    },
    "integrator_with_gradient": {
        "single_call":        "grid::integrator_with_gradient_single_timing<float>(hd_data,d_robotModel,GRAVITY,static_cast<float>(0.01),SINGLE_CALL_ITERS_GLOBAL,dim3(1,1,1),dimms,streams)",
        "batch_with_mem":     "grid::integrator_with_gradient<float>(d,m,GRAVITY,static_cast<float>(0.01),N,dim3(N,1,1),dimms,streams)",
        "batch_compute_only": "grid::integrator_with_gradient_compute_only<float>(d,m,GRAVITY,static_cast<float>(0.01),N,dim3(N,1,1),dimms)",
        "batch_label": "INTEGRATOR_WITH_GRADIENT",
        "gate": "GRID_HAS_INTEGRATOR_GRADIENT",
        "shared_mem_skip": "INTEGRATOR_DU_DYNAMIC_SHARED_MEM_BYTES",
    },
    # --- Registry<->specs bijection completion (2026-07-16) ---
    # These 7 keys each have a real benchmarkable kernel + host symbols (verified:
    # <algo>{,_compute_only,_single_timing} + <ALGO>_DYNAMIC_SHARED_MEM_BYTES) but
    # had no bench spec, so the wrapper printed "skipping algos missing a
    # PER_ALGO_SPECS row" and could never time them. test_per_algo_specs_bijection
    # now asserts set(PER_ALGO_SPECS) == {registry keys with a benchmarkable
    # kernel}, so this cannot silently drift again -- the same HAS-vs-emission
    # class that produced the f_ext_gradient_dq null-alloc bug. All 7 carry a
    # GRID_HAS_* gate so a robot whose codegen omits the family compiles the TU
    # out cleanly (never a link failure).
    #   coriolis_matrix / *_energy_regressor : take the gravity arg (RNEA sweep).
    #   dccrba / cmm_time_variation          : centroidal time-derivatives, NO gravity.
    #   end_effector_pose{,_gradient}_runtime: runtime-target pose surfaces (emitted
    #     only for runtime_transform builds), NO gravity; host mirrors end_effector_pose
    #     with a trailing target_jid defaulted to the leaf-EE joint.
    "coriolis_matrix": {
        "single_call":        "grid::coriolis_matrix_single_timing<float>(hd_data,d_robotModel,GRAVITY,SINGLE_CALL_ITERS_GLOBAL,dim3(1,1,1),dimms,streams)",
        "batch_with_mem":     "grid::coriolis_matrix<float>(d,m,GRAVITY,N,dim3(N,1,1),dimms,streams)",
        "batch_compute_only": "grid::coriolis_matrix_compute_only<float>(d,m,GRAVITY,N,dim3(N,1,1),dimms)",
        "batch_label": "CORIOLIS_MATRIX",
        "gate": "GRID_HAS_CORIOLIS_MATRIX",
        "shared_mem_skip": "CORIOLIS_MATRIX_DYNAMIC_SHARED_MEM_BYTES",
    },
    "dccrba": {
        "single_call":        "grid::dccrba_single_timing<float>(hd_data,d_robotModel,SINGLE_CALL_ITERS_GLOBAL,dim3(1,1,1),dimms,streams)",
        "batch_with_mem":     "grid::dccrba<float>(d,m,N,dim3(N,1,1),dimms,streams)",
        "batch_compute_only": "grid::dccrba_compute_only<float>(d,m,N,dim3(N,1,1),dimms)",
        "batch_label": "DCCRBA",
        "gate": "GRID_HAS_DCCRBA",
        "shared_mem_skip": "DCCRBA_DYNAMIC_SHARED_MEM_BYTES",
    },
    "cmm_time_variation": {
        "single_call":        "grid::cmm_time_variation_single_timing<float>(hd_data,d_robotModel,SINGLE_CALL_ITERS_GLOBAL,dim3(1,1,1),dimms,streams)",
        "batch_with_mem":     "grid::cmm_time_variation<float>(d,m,N,dim3(N,1,1),dimms,streams)",
        "batch_compute_only": "grid::cmm_time_variation_compute_only<float>(d,m,N,dim3(N,1,1),dimms)",
        "batch_label": "CMM_TIME_VARIATION",
        "gate": "GRID_HAS_CMM_TIME_VARIATION",
        "shared_mem_skip": "CMM_TIME_VARIATION_DYNAMIC_SHARED_MEM_BYTES",
    },
    "kinetic_energy_regressor": {
        "single_call":        "grid::kinetic_energy_regressor_single_timing<float>(hd_data,d_robotModel,GRAVITY,SINGLE_CALL_ITERS_GLOBAL,dim3(1,1,1),dimms,streams)",
        "batch_with_mem":     "grid::kinetic_energy_regressor<float>(d,m,GRAVITY,N,dim3(N,1,1),dimms,streams)",
        "batch_compute_only": "grid::kinetic_energy_regressor_compute_only<float>(d,m,GRAVITY,N,dim3(N,1,1),dimms)",
        "batch_label": "KINETIC_ENERGY_REGRESSOR",
        "gate": "GRID_HAS_KINETIC_ENERGY_REGRESSOR",
        "shared_mem_skip": "KINETIC_ENERGY_REGRESSOR_DYNAMIC_SHARED_MEM_BYTES",
    },
    "potential_energy_regressor": {
        "single_call":        "grid::potential_energy_regressor_single_timing<float>(hd_data,d_robotModel,GRAVITY,SINGLE_CALL_ITERS_GLOBAL,dim3(1,1,1),dimms,streams)",
        "batch_with_mem":     "grid::potential_energy_regressor<float>(d,m,GRAVITY,N,dim3(N,1,1),dimms,streams)",
        "batch_compute_only": "grid::potential_energy_regressor_compute_only<float>(d,m,GRAVITY,N,dim3(N,1,1),dimms)",
        "batch_label": "POTENTIAL_ENERGY_REGRESSOR",
        "gate": "GRID_HAS_POTENTIAL_ENERGY_REGRESSOR",
        "shared_mem_skip": "POTENTIAL_ENERGY_REGRESSOR_DYNAMIC_SHARED_MEM_BYTES",
    },
    "end_effector_pose_runtime": {
        "single_call":        "grid::end_effector_pose_runtime_single_timing<float>(hd_data,d_robotModel,SINGLE_CALL_ITERS_GLOBAL,dim3(1,1,1),dimms,streams)",
        "batch_with_mem":     "grid::end_effector_pose_runtime<float>(d,m,N,dim3(N,1,1),dimms,streams)",
        "batch_compute_only": "grid::end_effector_pose_runtime_compute_only<float>(d,m,N,dim3(N,1,1),dimms)",
        "batch_label": "END_EFFECTOR_POSE_RUNTIME",
        "gate": "GRID_HAS_END_EFFECTOR_POSE_RUNTIME",
        "shared_mem_skip": "END_EFFECTOR_POSE_RUNTIME_DYNAMIC_SHARED_MEM_BYTES",
    },
    "end_effector_pose_gradient_runtime": {
        "single_call":        "grid::end_effector_pose_gradient_runtime_single_timing<float>(hd_data,d_robotModel,SINGLE_CALL_ITERS_GLOBAL,dim3(1,1,1),dimms,streams)",
        "batch_with_mem":     "grid::end_effector_pose_gradient_runtime<float>(d,m,N,dim3(N,1,1),dimms,streams)",
        "batch_compute_only": "grid::end_effector_pose_gradient_runtime_compute_only<float>(d,m,N,dim3(N,1,1),dimms)",
        "batch_label": "END_EFFECTOR_POSE_GRADIENT_RUNTIME",
        "gate": "GRID_HAS_END_EFFECTOR_POSE_GRADIENT_RUNTIME",
        "shared_mem_skip": "END_EFFECTOR_POSE_GRADIENT_RUNTIME_DYNAMIC_SHARED_MEM_BYTES",
    },
}


# Algos whose generated kernel is a NON-PRODUCTION reference path under a given
# base, and must NOT be benchmarked there (timing them reports fake losses).
#
#   idsva_so_body_frame @ floating:
#     The floating-base body-frame SO kernel compiles
#     `gen_idsva_so_body_frame_floating_reference_inner` — explicitly documented
#     in grid_codegen/algorithms/_idsva_so.py (~:1170) as NOT emitted in
#     production: the dispatcher routes ALL floating-base SO to the WORLD frame,
#     so this body-frame floating branch is unreachable/dispatch-dead and is
#     ~single-threaded. Benchmarking it produced the spurious g1/iiwa/go2
#     "10–36× behind pinocchio" floating-SO losses (docs/open-tasks/
#     perf_behind_pin_analysis.md, Cluster A — MEASUREMENT ARTIFACT).
#     The PRODUCTION floating SO is still timed by the `idsva_so` dispatcher
#     (→ world-frame) and the `idsva_so_world_frame` rows, which remain in the
#     sweep — so dropping this row loses no production coverage.
NON_PRODUCTION_ALGOS_BY_BASE: dict[str, frozenset[str]] = {
    "floating": frozenset({"idsva_so_body_frame"}),
    "fixed": frozenset(),
}

# Algos whose per-base generated kernel is BIT-IDENTICAL to the `idsva_so`
# dispatcher's, so building BOTH compiles the same __global__ twice for zero
# added coverage. The `idsva_so` host wrapper forwards, at codegen time, to
# whichever frame the dispatcher picks for this base (grid.cuh:
# `void idsva_so(...) { idsva_so_world_frame<...>(...); }` on floating), and its
# `_single_timing` launches that frame's kernel DIRECTLY (no extra indirection),
# so the dispatcher row already times the exact same kernel.
#
#   floating: dispatcher -> world_frame  => `idsva_so_world_frame` is redundant
#   fixed:    dispatcher -> body_frame   => `idsva_so_body_frame`  is redundant
#
# On g1-floating that lone duplicate is the WORLD-frame SO kernel — the single
# most expensive compile in the tree (28x its pin SASS): ~978 s + ~982 s built
# back-to-back, ~16 min of pure waste. We keep the `idsva_so` dispatcher row (the
# production-honest one, and the only SO row analyze_competitive scores — it
# canonicalizes body/world to 'idsva_so' and drops the standalone entries), and
# drop the dispatched-frame standalone. The OTHER frame's standalone row (the
# non-dispatched A/B alternative) is unaffected: on fixed it stays for the
# body-vs-world A/B; on floating it is separately dropped as NON_PRODUCTION.
REDUNDANT_WITH_DISPATCHER_BY_BASE: dict[str, frozenset[str]] = {
    "floating": frozenset({"idsva_so_world_frame"}),
    "fixed": frozenset({"idsva_so_body_frame"}),
}

# Algos the codegen SKIPS for mimic robots, so their grid:: symbols are absent
# and the matching bench TU would fail to compile/link. com/ccrba/energy are the
# (Was MIMIC_UNSUPPORTED_ALGOS = {com, ccrba, energy}.) EMPTIED 2026-07-14: the drop was STALE. Its two
# justifications were both false — (1) gen_centroidal_quickwins now states "Mimic robots are SUPPORTED"
# (the per-body Jacobian + dccrba per-unit phi are alpha-folded, mirroring the mimic-aware oracle), the
# skip it cited no longer exists; (2) GRID_HAS_COM / GRID_HAS_CCRBA / GRID_HAS_ENERGY ARE emitted (=1 on
# fr3-mimic, kernels present — verified), so a #if gate is available and the Python drop is unnecessary.
# Consequence of the stale drop: every mimic robot (h1_2, h2_plus) silently lost com/ccrba/energy from
# all sweep coverage. Kept as an (empty) frozenset so the _algo_keys_in_registry_order call site is
# unchanged; add a key back ONLY with a passing mimic equivalence test proving the drop is real.
MIMIC_UNSUPPORTED_ALGOS: frozenset[str] = frozenset()


# Registry keys intentionally NOT benchmarked: has_kernel_attr=False composites
# with no standalone benchmarkable host (no <algo>_single_timing / _compute_only).
# Kept explicit so (a) the "missing spec row" warning below only fires on genuine
# registry<->specs drift, and (b) test_per_algo_specs_bijection can assert this is
# the COMPLETE exclusion set. Adding a new algo forces a choice: give it a
# PER_ALGO_SPECS row (benchmarkable) or list it here with justification.
#   integrator_hessian : plant_step_hessian composite (no standalone host)
#   plant              : cost/constraint/step primitives (no single kernel)
#   collision          : device-composite; a timing wrapper is separate staged work
#   f_ext_contact      : has_kernel_attr=False device composite (contact-frame
#                        wrench -> joint-local f_ext + derivatives, GATO ask 1 C.2).
#                        Emits *_inner/*_device only -- no __global__ kernel and no
#                        _single_timing/_compute_only host, so there is nothing to
#                        time. Correctness is covered by the go2-floating FD-oracle
#                        test/cuda_equivalents/test_cuda_f_ext_contact.py.
BENCH_EXCLUDED_ALGOS: frozenset[str] = frozenset({"integrator_hessian", "plant", "collision",
                                                  "f_ext_contact"})


def _algo_keys_in_registry_order(floating_base: bool | None = None,
                                 has_mimic: bool | None = None,
                                 dedup_dispatcher_redundant: bool = True) -> list[str]:
    """Return algo keys in ALGO_REGISTRY order, filtered to those in PER_ALGO_SPECS.

    When `floating_base` is given, also drop any algo whose generated kernel is a
    non-production reference path for that base (see NON_PRODUCTION_ALGOS_BY_BASE)
    so the competitive sweep never times dispatch-dead code, AND — unless
    `dedup_dispatcher_redundant=False` — the standalone SO row that is bit-identical
    to the `idsva_so` dispatcher on this base (REDUNDANT_WITH_DISPATCHER_BY_BASE), so
    the same expensive kernel is not compiled twice. Pass
    `dedup_dispatcher_redundant=False` when the caller has EXPLICITLY named the algos
    (e.g. `--algos idsva_so_world_frame`) and must be able to build that exact row.
    When `has_mimic` is True, drop algos the codegen omits for mimic robots
    (MIMIC_UNSUPPORTED_ALGOS) whose grid:: symbols would otherwise be absent and break
    the build. `floating_base`/`has_mimic` default to None (keep everything) — used by
    callers that only need the full key universe (e.g. cache-key bookkeeping).
    """
    drop: frozenset[str] = frozenset()
    if floating_base is not None:
        base_key = "floating" if floating_base else "fixed"
        drop |= NON_PRODUCTION_ALGOS_BY_BASE[base_key]
        if dedup_dispatcher_redundant:
            drop |= REDUNDANT_WITH_DISPATCHER_BY_BASE[base_key]
    if has_mimic:
        drop |= MIMIC_UNSUPPORTED_ALGOS

    keys: list[str] = []
    missing: list[str] = []
    for entry in ALGO_REGISTRY:
        if entry.key in drop:
            continue
        if entry.key in PER_ALGO_SPECS:
            keys.append(entry.key)
        elif entry.key not in BENCH_EXCLUDED_ALGOS:
            missing.append(entry.key)
    if missing:
        # A registry algo with a benchmarkable kernel lost its spec row (genuine
        # drift). BENCH_EXCLUDED_ALGOS (integrator_hessian/plant/collision) are
        # intentionally un-benchmarked and filtered out above, so anything that
        # reaches here is a real gap — warn loudly. test_per_algo_specs_bijection
        # is the hard gate; this warning catches it at sweep time too.
        print(f"  [grid] WARNING: skipping algos missing a PER_ALGO_SPECS row: {', '.join(missing)}",
              file=sys.stderr)
    return keys


def _gate_open(spec: dict) -> str:
    """Open preprocessor block for the per-algo gate. Returns '' when ungated."""
    return f"#if {spec['gate']}\n" if spec.get("gate") else ""


def _gate_close(spec: dict) -> str:
    return "#endif\n" if spec.get("gate") else ""


def _attr_init_call(algo_key: str, spec: dict, guarded: bool) -> str:
    """C++ statement(s) registering ONLY this algo's kernel attributes -- the per-algo
    split (grid::init_grid_kernel_attr_<short>) that REPLACES the init_grid_kernel_attrs
    monolith. The monolith address-takes all ~35 kernels, so every solo TU that called it
    instantiated the WHOLE set + hit ptxas (the OOM/slow-compile on big humanoids). This
    pulls in only this algo's kernel.

    `idsva_so` is a dispatch alias with no own kernel -> register whichever concrete
    variant(s) the header emitted (a floating header has only world_frame). When `guarded`,
    wrap the call in the algo's GRID_HAS_* gate -- needed OUTSIDE the measure entry (which is
    already inside its own gate) so a robot that didn't generate the algo still compiles."""
    if algo_key == "idsva_so":
        return (
            "\n#if GRID_HAS_IDSVA_SO_WORLD_FRAME\n"
            "        grid::init_grid_kernel_attr_idsva_so_world_frame<float>();\n"
            "#endif\n#if GRID_HAS_IDSVA_SO_BODY_FRAME\n"
            "        grid::init_grid_kernel_attr_idsva_so_body_frame<float>();\n"
            "#endif\n        "
        )
    call = f"grid::init_grid_kernel_attr_{algo_key}<float>();"
    gate = spec.get("gate")
    if guarded and gate:
        return f"\n#if {gate}\n        {call}\n#endif\n        "
    return call


def _per_algo_batch_tu_source(algo_key: str) -> str:
    """Source for timeGRiD_batch_<algo>.cu: defines measure_<algo>_batch_entry."""
    spec = PER_ALGO_SPECS[algo_key]
    skip_block = ""
    if "shared_mem_skip" in spec:
        def _skip_probe(fn: str) -> str:
            return (
                f"    if (!grid_kernel_fits_device(grid::{fn}<float>())) {{\n"
                f"        printf(\"[N:%d]: {spec['batch_label']} SKIPPED (kernel needs %zu bytes shared mem, exceeds device cap)\\n\",\n"
                f"               N, grid::{fn}<float>()); return;\n"
                f"    }}\n"
            )
        skip_block = _skip_probe(spec["shared_mem_skip"])
        if "shared_mem_skip_dispatch" in spec:
            # Dispatch-aware probe: when the header says the dispatching wrapper
            # forwards to the alternate kernel, check THAT kernel's bytes instead.
            macro, alt_fn = spec["shared_mem_skip_dispatch"]
            skip_block = (
                f"#if defined({macro}) && {macro}\n"
                + _skip_probe(alt_fn)
                + "#else\n"
                + _skip_probe(spec["shared_mem_skip"])
                + "#endif\n"
            )
    body = (
        f"{_gate_open(spec)}"
        f"void measure_{algo_key}_batch_entry(int N, cudaStream_t *streams, grid::robotModel<float> *m, grid::gridData<float> *d){{\n"
        f"    static const bool _attrs_set = []() {{ {_attr_init_call(algo_key, spec, guarded=False)} return true; }}();\n"
        f"    (void)_attrs_set;\n"
        f"    dim3 dimms = grid_timing_dimms();\n"
        f"{skip_block}"
        f"    measure_batch_pair<TEST_ITERS_GLOBAL>(\"{spec['batch_label']}\", N,\n"
        f"        [&]{{ {spec['batch_with_mem']}; }},\n"
        f"        [&]{{ {spec['batch_compute_only']}; }});\n"
        f"}}\n"
        f"{_gate_close(spec)}"
    )
    return (
        "// AUTO-GENERATED by test/benchmarks/baselines/grid/run.py — do not hand-edit.\n"
        "// One TU per (algo, kind). Compiled WITHOUT -rdc=true so nvcc can\n"
        "// aggressively inline ::glass::* / dot_prod / grid_xhom_or_dxhom_ptr.\n"
        "#include \"timeGRiD_common.h\"\n"
        "\n"
        + body
    )


def _write_if_changed(path: Path, content: str) -> None:
    """Write content to path only if it differs — preserves mtime so ccache stays warm."""
    if path.exists():
        try:
            if path.read_text() == content:
                return
        except OSError:
            pass
    path.write_text(content)


# A1a (autotune-matrix Phase 1): widened to reach the warp floor (32) and the
# hardware ceiling (1024). The earlier narrow {96..384} grid was tuned to small
# fixed-base robots, but the fast regime SCALES UP with robot size + batch
# (go2>=512, g1>=640, up to 1024 — see project_grid_jax_ffi_thread_pathology):
# clamping at 384 hid the genuine large-robot optima. `_clip_grid_to_cap` drops
# every probe above each tier's __launch_bounds__ (tier_max_threads<TIER>()), so
# adding 512..1024 never launches above a tier's register cap — a probe that
# would exceed it is silently dropped, NOT timed (guards against the §1c
# bogus-fast / failed-launch-reads-fastest trap). The one-level refinement around
# each winner (_refine_grid_for_winner) still probes immediate neighbors, so
# edge-case optima between grid points are not missed. Override with
# --autotune-thread-grid.
DEFAULT_AUTOTUNE_THREAD_GRID: tuple[int, ...] = (
    32, 64, 96, 128, 192, 256, 320, 384, 512, 640, 768, 896, 1024)
DEFAULT_AUTOTUNE_N: int = 256            # batch size on which we tune (matches default bench)
AUTOTUNE_TIERS: tuple[str, ...] = ("shared", "lite", "minimal")


def _read_max_perf_level_threads(header_path: Path) -> int | None:
    """Extract `const int MAX_PERF_LEVEL_THREADS = N;` from a generated grid.cuh.

    Used for cap-aware thread-grid clipping. Returns None if not found (callers
    then fall back to the hardware cap of 1024)."""
    import re
    try:
        text = header_path.read_text()
    except OSError:
        return None
    m = re.search(r"MAX_PERF_LEVEL_THREADS\s*=\s*(\d+)\s*;", text)
    return int(m.group(1)) if m else None


def _tier_thread_cap(tier: str, max_perf: int | None) -> int:
    """Mirror grid.cuh's tier_max_threads<TIER>() so we don't probe above the
    kernel's __launch_bounds__ (which would fail the launch)."""
    mp = max_perf if max_perf is not None else 1024
    if tier == "minimal":
        return 1024
    if tier == "lite":
        return min(mp * 2, 768)
    return mp  # shared (== ex-PERF)


def _clip_grid_to_cap(thread_grid: tuple[int, ...], cap: int) -> tuple[int, ...]:
    """Drop probes exceeding `cap`; keep at least the largest fitting one."""
    fit = tuple(t for t in thread_grid if t <= cap)
    if fit:
        return fit
    # Degenerate: every grid point exceeds the cap — fall back to the cap itself.
    return (cap,)
DEFAULT_AUTOTUNE_BATCH_ITERS: int = 50    # outer rep count per (algo, thread count) cell


def _autotune_batch_iters_for_binary(batch_binary: Path, base: str, threads: int,
                                     env_extra: dict[str, str]) -> str:
    """Run the batch binary with GRID_AUTOTUNE_THREAD_COUNT=threads and return stdout.
    Caller is responsible for handling parsing/errors."""
    env = os.environ.copy()
    env.update(env_extra)
    env["GRID_AUTOTUNE_THREAD_COUNT"] = str(int(threads))
    floating_arg = "T" if base == "floating" else "F"
    result = subprocess.run(
        [str(batch_binary), floating_arg],
        capture_output=True, text=True, env=env,
    )
    if result.returncode != 0:
        print(f"  [autotune] WARN: batch binary exited {result.returncode} "
              f"at threads={threads}; stderr (tail):\n{result.stderr[-400:]}",
              file=sys.stderr)
    return result.stdout


def _target_key_for_mode(mode: str, autotune_N: int) -> str:
    """Parser record key the autotune minimizes. batch → batch-N compute-only;
    single → single-call µs."""
    if mode == "single":
        return "single_us"
    return f"batch_{autotune_N}_compute_only_us"


def _sweep_one_binary(
    binary: Path,
    base: str,
    thread_grid: tuple[int, ...],
    target_key: str,
) -> dict[str, dict[int, float]]:
    """Run `binary` once per thread count in `thread_grid`, parse `target_key`
    per algo, and return {algo: {threads: us}}. Thread counts already clipped to
    the tier cap by the caller."""
    sweeps: dict[str, dict[int, float]] = {}
    for threads in thread_grid:
        stdout = _autotune_batch_iters_for_binary(binary, base, threads, {})
        parsed = parse_grid_output(stdout)
        for algo, entry in parsed.items():
            if not isinstance(entry, dict):
                continue
            bucket = entry.get(target_key)
            if not bucket:
                continue
            us = bucket.get("median") or bucket.get("mean")
            if us is None:
                continue
            sweeps.setdefault(algo, {})[int(threads)] = float(us)
    return sweeps


def _refine_grid_for_winner(winner: int, sorted_grid: list[int], cap: int) -> set[int]:
    """Midpoints between `winner` and its grid neighbours (one-level refinement),
    clipped to [32, cap]."""
    out: set[int] = set()
    if winner not in sorted_grid:
        return out
    idx = sorted_grid.index(winner)
    if idx > 0:
        mid = (sorted_grid[idx - 1] + winner) // 2
        if mid not in sorted_grid and 32 <= mid <= cap:
            out.add(mid)
    if idx < len(sorted_grid) - 1:
        mid = (winner + sorted_grid[idx + 1]) // 2
        if mid not in sorted_grid and 32 <= mid <= cap:
            out.add(mid)
    return out


# ---------------------------------------------------------------------------
# Tier-equivalence dedup (provable, structural — NOT a timing heuristic).
#
# The per-tier binaries differ ONLY in the compile-time RESOURCE_TIER baked in
# via -DGRID_DEFAULT_RESOURCE_TIER. For a given algo, that tier feeds three
# tier-varying inputs into the kernel template: the resolved dynamic-smem bytes,
# the boolean smem/inner-level toggles (*_IN_SMEM<TIER>, *_INNER_LEVEL<TIER>),
# AND the per-tier __launch_bounds__ cap (tier_max_threads<TIER>()). All three
# are captured *exactly* by the kernel's emitted machine code (SASS). So the
# provable signature for an (algo, tier) is the hash of that algo's kernel SASS,
# with the tier-enum template immediate (Li0/Li1/Li2) in the mangled symbol name
# normalized out (it is a pure name artifact, not a code difference). Two tiers
# whose kernel SASS hashes match for an algo are BYTE-IDENTICAL kernels: the
# thread sweep on one is, by construction, the thread sweep on the other, so we
# time it once and copy. Tiers whose SASS differs (the common small-robot case,
# where the launch_bounds caps 352/704/1024 already force distinct register
# allocation) are NEVER collapsed — they are swept normally. There is no
# timing-closeness fudge anywhere in this path.
#
# Granularity is per-(tier, algo): a single algo can be deduped between two tiers
# even if other algos in the same binary are not.
# ---------------------------------------------------------------------------
import re as _re  # module-level imports don't include re; alias keeps it local


def _algo_kernel_symbol_base(algo: str) -> str | None:
    """Map an algo key to its CUDA kernel symbol base (e.g. 'id' ->
    'inverse_dynamics_kernel'). Derived from the algo's batch_compute_only call
    `grid::<base>_compute_only<...>` (the launched kernel is `<base>_kernel`).
    Returns None if the spec is missing/unparseable (algo then never dedups)."""
    spec = PER_ALGO_SPECS.get(algo)
    if not spec:
        return None
    m = _re.search(r"grid::([A-Za-z0-9_]+)_compute_only<", spec.get("batch_compute_only", ""))
    if not m:
        return None
    return f"{m.group(1)}_kernel"


def _cuobjdump_elf(binary: Path) -> str | None:
    """`cuobjdump -elf <binary>` stdout, or None on failure."""
    try:
        elf = subprocess.run(["cuobjdump", "-elf", str(binary)],
                             capture_output=True, text=True)
    except OSError:
        return None
    return elf.stdout if elf.returncode == 0 else None


def _kernel_sass_hash(binary: Path, kernel_base: str,
                      elf_text: str | None = None) -> str | None:
    """SHA-256 of the SASS for `grid::<kernel_base><float, TIER>` in `binary`,
    normalized so the result is tier-independent EXCEPT for genuine code diffs.

    Normalization: (1) the mangled tier immediate `IfLi[0-2]E` in the function
    symbol is collapsed to a placeholder, so two tiers that emit identical code
    don't differ merely by their template-enum name; (2) the per-instruction
    encoded-hex columns (/* 0x... */) and absolute addresses are stripped, so we
    hash the instruction stream + operands, not load addresses.

    `elf_text` may be a pre-fetched `cuobjdump -elf` dump (avoids re-running it
    per algo). Returns None (→ algo is not deduped, swept normally) if the symbol
    is absent or cuobjdump fails."""
    # Find the exact mangled symbol for this kernel base (the _kernel form, NOT
    # _kernel_single_timing). Match `<...NNkernel_baseIfLi[0-2]E...>` in the ELF
    # section table, then dump just that function's SASS.
    if elf_text is None:
        elf_text = _cuobjdump_elf(binary)
    if elf_text is None:
        return None
    # Section names look like `.text._ZN4grid23inverse_dynamics_kernelIfLi0EEEv...`.
    # Require the base immediately followed by `IfLi<d>E` so `crba_kernel` does
    # not also match `crba_kernel_single_timing`.
    pat = _re.compile(r"(_ZN4grid\d+" + _re.escape(kernel_base) + r"IfLi[0-2]E\S*)")
    sym = None
    for m in pat.finditer(elf_text):
        cand = m.group(1)
        # Exclude the single_timing variant (its base is `<base>_single_timing`,
        # so it won't match here, but guard defensively).
        if "_single_timing" in cand:
            continue
        sym = cand
        break
    if sym is None:
        return None
    try:
        sass = subprocess.run(["cuobjdump", "-sass", "-fun", sym, str(binary)],
                              capture_output=True, text=True)
    except OSError:
        return None
    if sass.returncode != 0 or not sass.stdout:
        return None
    lines: list[str] = []
    for ln in sass.stdout.splitlines():
        ln = _re.sub(r"/\*[0-9a-fA-F]+\*/", "", ln)  # encoded hex / addr columns
        ln = ln.strip()
        if not ln:
            continue
        lines.append(ln)
    body = "\n".join(lines)
    # Collapse the tier immediate in the symbol name so it doesn't pollute the
    # hash (the symbol name appears in the SASS header line).
    body = _re.sub(r"IfLi[0-2]E", "IfLiXE", body)
    return hashlib.sha256(body.encode()).hexdigest()


def _present_algos_in_binary(binary: Path, elf_text: str | None = None) -> list[str]:
    """Registry algos (PER_ALGO_SPECS order) whose compute kernel symbol is
    actually present in `binary`. These are the algos the bench will time, so
    they are exactly the ones the dedup must account for. Empty list if the ELF
    can't be read."""
    if elf_text is None:
        elf_text = _cuobjdump_elf(binary)
    if elf_text is None:
        return []
    out: list[str] = []
    for algo in _algo_keys_in_registry_order():
        base = _algo_kernel_symbol_base(algo)
        if base is None:
            continue
        # `<NN><base>IfLi<d>E` — base immediately followed by the tier immediate,
        # so `crba_kernel` does not match `crba_kernel_single_timing`.
        if _re.search(r"\d+" + _re.escape(base) + r"IfLi[0-2]E", elf_text):
            out.append(algo)
    return out


def _tier_algo_signature(binary: Path, algo: str,
                         elf_text: str | None = None) -> str | None:
    """Provable per-(tier, algo) kernel signature, or None if unavailable
    (caller then sweeps that (tier, algo) normally — never wrongly collapses)."""
    base = _algo_kernel_symbol_base(algo)
    if base is None:
        return None
    return _kernel_sass_hash(binary, base, elf_text=elf_text)


def _autotune_pick_winners(
    tier_binaries: dict[str, Path],
    base: str,
    thread_grid: tuple[int, ...] = DEFAULT_AUTOTUNE_THREAD_GRID,
    autotune_N: int = DEFAULT_AUTOTUNE_N,
    *,
    max_perf_level_threads: int | None = None,
    mode: str = "batch",
) -> dict[str, dict]:
    """Joint (tier × thread-count) autotune.

    For each tier in `tier_binaries` (its batch — or, in single mode, single —
    binary), sweep the per-tier-cap-clipped `thread_grid` and parse the target
    timing per algo. The per-algo winner is the global argmin µs over
    (tier, threads).

    Returns schema-2 picks:
        {
            algo: {
                "schema": 2,
                "tier_optimal": str, "threads_optimal": int, "us_at_optimal": float,
                "sweep": {tier: {threads: us, ...}, ...},
                "sweep_us": {threads: us, ...},   # flat back-compat (winning tier)
            }
        }

    Algos with no readings at any (tier, thread) cell are omitted.
    """
    target_key = _target_key_for_mode(mode, autotune_N)
    # algo -> tier -> {threads: us}
    sweeps: dict[str, dict[str, dict[int, float]]] = {}
    # algo -> tier -> "tier this (algo, tier) was proven byte-identical to and
    # whose sweep numbers were copied" (omitted when the algo was swept fresh).
    tier_equiv: dict[str, dict[str, str]] = {}
    # tier -> {algo: provable kernel-SASS signature}. Cached so each tier's
    # binary is fingerprinted once.
    tier_sigs: dict[str, dict[str, str]] = {}

    for tier, binary in tier_binaries.items():
        if binary is None:
            continue
        cap = _tier_thread_cap(tier, max_perf_level_threads)
        tier_grid = _clip_grid_to_cap(thread_grid, cap)

        # --- Provable tier-equivalence dedup ---------------------------------
        # Fingerprint every algo's kernel SASS in this tier, then for each algo
        # whose signature exactly matches an ALREADY-SWEPT tier's signature,
        # copy that tier's sweep numbers (byte-identical kernel ⇒ identical
        # timing) instead of re-timing. If EVERY fingerprintable algo collapses
        # this way, the whole binary run is skipped (the big-robot case). Algos
        # with a unique signature (or no obtainable signature) are swept.
        # `present` = registry algos whose kernel symbol exists in THIS binary
        # (so it will actually be timed). `my_sigs` = those of them we could
        # fingerprint. A present algo that we could NOT fingerprint (sig is None
        # despite the symbol existing) is a "blind spot": we cannot prove it
        # identical, so it forces the binary to be swept (never skipped).
        elf_text = _cuobjdump_elf(binary)  # one dump reused for all algos here
        present = _present_algos_in_binary(binary, elf_text=elf_text)
        my_sigs: dict[str, str] = {}
        blind: set[str] = set()
        for algo in present:
            sig = _tier_algo_signature(binary, algo, elf_text=elf_text)
            if sig is None:
                blind.add(algo)
            else:
                my_sigs[algo] = sig
        tier_sigs[tier] = my_sigs

        # algo -> source tier it is byte-identical to (first prior tier that has
        # the same signature AND was actually swept for that algo).
        copy_from: dict[str, str] = {}
        for algo, sig in my_sigs.items():
            # Earliest prior tier (insertion order) with a matching signature
            # that we already have fresh sweep numbers for.
            for prev_tier in tier_binaries:
                if prev_tier == tier:
                    break  # only consider tiers swept before this one
                if (tier_sigs.get(prev_tier, {}).get(algo) == sig
                        and algo in sweeps and prev_tier in sweeps[algo]):
                    copy_from[algo] = prev_tier
                    break

        # Skip the whole binary run ONLY if every present algo is a proven
        # dedup copy (no blind spots, at least one algo present).
        all_deduped = (bool(present)
                       and not blind
                       and all(a in copy_from for a in present))

        if all_deduped:
            src = sorted(set(copy_from.values()))
            print(f"  [autotune] tier={tier:<7s} cap={cap:>4d} DEDUP: all "
                  f"{len(copy_from)} algos byte-identical to tier(s) {src} "
                  f"(SASS-equal) — skipping thread sweep, copying numbers",
                  file=sys.stderr)
            tier_sweep = {}
        else:
            print(f"  [autotune] tier={tier:<7s} cap={cap:>4d} sweeping "
                  f"{len(tier_grid)} thread counts ({mode}): {list(tier_grid)}"
                  + (f" (dedup-copied {len(copy_from)} algo(s): "
                     f"{sorted(copy_from)})" if copy_from else ""),
                  file=sys.stderr)
            tier_sweep = _sweep_one_binary(binary, base, tier_grid, target_key)

        # Fresh-swept algos.
        for algo, by_threads in tier_sweep.items():
            if algo in copy_from:
                continue  # deduped algos get copied numbers below, not raw ones
            sweeps.setdefault(algo, {})[tier] = by_threads
        # Deduped algos: copy the byte-identical source tier's sweep verbatim so
        # the per-tier `sweep` column is still fully populated and the global
        # argmin sees this tier as a (tied) candidate.
        for algo, src_tier in copy_from.items():
            src_sweep = sweeps.get(algo, {}).get(src_tier)
            if src_sweep is None:
                continue
            sweeps.setdefault(algo, {})[tier] = dict(src_sweep)
            tier_equiv.setdefault(algo, {})[tier] = src_tier

    # One-level refinement around each algo's current (tier, threads) winner.
    # Probe per tier so we never exceed that tier's launch_bounds cap.
    refine_by_tier: dict[str, set[int]] = {t: set() for t in tier_binaries}
    for algo, by_tier in sweeps.items():
        best = _argmin_tier_threads(by_tier)
        if best is None:
            continue
        wtier, wthreads, _ = best
        cap = _tier_thread_cap(wtier, max_perf_level_threads)
        grid_for_tier = sorted(_clip_grid_to_cap(thread_grid, cap))
        refine_by_tier[wtier] |= _refine_grid_for_winner(wthreads, grid_for_tier, cap)

    # Algos whose (tier) cell was a dedup copy must never be re-timed in
    # refinement (they have no independent kernel). They are re-synced from
    # their byte-identical source tier after refinement instead.
    def _is_deduped(algo: str, tier: str) -> bool:
        return tier in tier_equiv.get(algo, {})

    for tier, extra in refine_by_tier.items():
        extra = {t for t in extra if t not in (sweeps_for_tier_threads(sweeps, tier))}
        if not extra:
            continue
        binary = tier_binaries.get(tier)
        if binary is None:
            continue
        # If every algo that would be refined in this tier is a dedup copy,
        # there is no fresh kernel to time — skip the launch (its numbers come
        # from the source tier's refinement via the re-sync below).
        fresh_algos = {a for a in sweeps if tier in sweeps.get(a, {})
                       and not _is_deduped(a, tier)}
        if not fresh_algos:
            continue
        print(f"  [autotune] tier={tier} refinement probes: {sorted(extra)}", file=sys.stderr)
        tier_sweep = _sweep_one_binary(binary, base, tuple(sorted(extra)), target_key)
        for algo, by_threads in tier_sweep.items():
            if _is_deduped(algo, tier):
                continue  # don't overwrite a dedup copy with this tier's own run
            sweeps.setdefault(algo, {}).setdefault(tier, {}).update(by_threads)

    # Re-sync every deduped (algo, tier) cell from its byte-identical source so
    # any refinement points the source gained are reflected (the cells stay
    # exactly equal, as they must — same kernel).
    for algo, equivs in tier_equiv.items():
        for tier, src_tier in equivs.items():
            src_sweep = sweeps.get(algo, {}).get(src_tier)
            if src_sweep is not None:
                sweeps[algo][tier] = dict(src_sweep)

    picks: dict[str, dict] = {}
    for algo, by_tier in sweeps.items():
        best = _argmin_tier_threads(by_tier)
        if best is None:
            continue
        wtier, wthreads, wus = best
        pick = {
            "schema": 2,
            "tier_optimal": wtier,
            "threads_optimal": int(wthreads),
            "us_at_optimal": float(wus),
            "sweep": {
                t: {str(int(th)): float(us) for th, us in sorted(s.items())}
                for t, s in sorted(by_tier.items())
            },
            # Flat back-compat view (schema-1 'sweep_us'): the winning tier's
            # per-threads sweep, so existing generate_report.py consumers work.
            "sweep_us": {
                str(int(th)): float(us)
                for th, us in sorted(by_tier.get(wtier, {}).items())
            },
        }
        # Record which tiers were proven byte-identical (SASS-equal) to an
        # earlier-swept tier and therefore had their sweep numbers COPIED rather
        # than re-timed. {deduped_tier: source_tier}. Absent ⇒ all tiers were
        # swept independently for this algo.
        if algo in tier_equiv and tier_equiv[algo]:
            pick["tier_equiv_to"] = dict(sorted(tier_equiv[algo].items()))
        picks[algo] = pick
    return picks


def sweeps_for_tier_threads(sweeps: dict, tier: str) -> set[int]:
    """All thread counts already probed for `tier` across every algo."""
    out: set[int] = set()
    for by_tier in sweeps.values():
        out |= set(by_tier.get(tier, {}).keys())
    return out


def _argmin_tier_threads(
    by_tier: dict[str, dict[int, float]],
) -> tuple[str, int, float] | None:
    """Global argmin over (tier, threads). Returns (tier, threads, us) or None."""
    best: tuple[str, int, float] | None = None
    for tier, sweep in by_tier.items():
        for threads, us in sweep.items():
            if best is None or us < best[2]:
                best = (tier, int(threads), float(us))
    return best
