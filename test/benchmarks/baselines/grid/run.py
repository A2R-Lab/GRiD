#!/usr/bin/env python3
"""Run GRiD timing benchmark for one robot/base combination.

Usage:
    python test/benchmarks/baselines/grid/run.py \
        --robot iiwa14 --base fixed [--output results/iiwa14_fixed_rtx5090.json] \
        [--no-recompile] [--ee-frame iiwa_link_ee]
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

from GRiDCodeGenerator import GRiDCodeGenerator  # noqa: E402
from GRiDCodeGenerator.algo_registry import ALGO_REGISTRY  # noqa: E402
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


def get_urdf_path(robot: str) -> str:
    mod_name = ROBOT_DESCRIPTION_MODULE.get(robot)
    if mod_name is None:
        raise ValueError(f"Unknown robot '{robot}'. Known: {list(ROBOT_DESCRIPTION_MODULE)}")
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
) -> Path:
    """Generate grid.cuh for the given robot/base, using content-hash cache."""
    floating_base = (base == "floating")

    urdf_hash = _hash_file(Path(urdf_path))
    codegen_hash = _hash_tree(REPO_ROOT / "GRiDCodeGenerator", (".py",))
    # GRID_NO_LICM_BARRIER suppresses the anti-LICM machinery in _single_timing
    # rep loops (volatile reload + __noinline__ barrier). When toggled, the
    # generated header changes — must bust the header cache.
    no_licm_barrier_env = os.environ.get("GRID_NO_LICM_BARRIER", "0")
    cache_key = _hash_bytes(
        json.dumps({
            "urdf_hash": urdf_hash,
            "codegen_hash": codegen_hash,
            "robot": robot,
            "base": base,
            "profile": "all",
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
    with contextlib.redirect_stdout(io.StringIO()):
        codegen.gen_all_code(
            include_homogenous_transforms=True,
            # fixed_target_name omitted: passing it with codegen_profile='all' triggers a
            # generator bug where kinematics_only() references an _hessian_{name} variant
            # that isn't generated. EE pose timing is unaffected by this omission.
            output_path=str(header_path),
            codegen_profile="all",
            enable_idsva_so_world_frame=True,
            enable_floating_second_order=True,
        )

    cached_header.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(header_path, cached_header)
    print(f"  [grid] header generated: {header_path.name}")
    return header_path


def _recompile_requested() -> bool:
    return os.environ.get("GRID_BENCH_RECOMPILE", "0") == "1"


# ---------------------------------------------------------------------------
# Binary compilation with caching
# ---------------------------------------------------------------------------
THIS_DIR = Path(__file__).resolve().parent
# Two-binary split (2026-05-15):
#   timeGRiD_single.cu — compiled WITH -rdc=true (anti-LICM correctness)
#   timeGRiD_batch.cu  — compiled WITHOUT -rdc=true (recovers 3× SIMT batch perf)
# Each TU has its own main() and produces its own .exe. We run both and
# concatenate their stdout for the parser. timeGRiD_common.h holds the
# shared init/load/warmup scaffolding.
TIMING_SOURCE_SINGLE = THIS_DIR / "timeGRiD_single.cu"
TIMING_SOURCE_BATCH  = THIS_DIR / "timeGRiD_batch.cu"
TIMING_SOURCE_COMMON = THIS_DIR / "timeGRiD_common.h"


# ---------------------------------------------------------------------------
# Per-algo TU split (P6-7b).
#
# The monolithic timeGRiD_{single,batch}.cu translation units each include the
# generated grid.cuh and instantiate every algo's kernel in one go. cicc (the
# nvcc front-end) processes that 1+M-line TU per build; the giant per-algo
# template stack is the bottleneck for sweep wall time.
#
# The per-algo split below emits N small TUs per timing kind (single, batch),
# each defining one `measure_<algo>_<kind>_entry(...)` host function that
# calls the matching `grid::*` template. A dispatcher main TU declares all
# `extern` entries and calls them in registry order. nvcc compiles all
# .cu → .o in parallel, then one link step produces the final binary.
#
# Behavior must remain byte-identical to the monolithic flow:
#   - same printf labels (those come from algo_registry / grid.cuh, untouched)
#   - same iteration counts (SINGLE_CALL_ITERS_GLOBAL / TEST_ITERS_GLOBAL)
#   - same dispatch order
#   - same -rdc=true on single, no-rdc on batch
#   - same GRID_HAS_* gating
#
# The per-algo spec table below mirrors exactly what the monolithic TUs
# encode by hand. If you add a new algo to ALGO_REGISTRY, add a matching row.
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
        "shared_mem_skip": "ID_DYNAMIC_SHARED_MEM_BYTES",
    },
    "minv": {
        "single_call":        "grid::direct_minv_single_timing<float,true>(hd_data,d_robotModel,SINGLE_CALL_ITERS_GLOBAL,dim3(1,1,1),dimms,streams)",
        "batch_with_mem":     "grid::direct_minv<float,true>(d,m,N,dim3(N,1,1),dimms,streams)",
        "batch_compute_only": "grid::direct_minv_compute_only<float,true>(d,m,N,dim3(N,1,1),dimms)",
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
        "shared_mem_skip": "FD_DYNAMIC_SHARED_MEM_BYTES",
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
        "shared_mem_skip": "ID_DU_DYNAMIC_SHARED_MEM_BYTES",
    },
    "forward_dynamics_gradient": {
        "single_call":        "grid::forward_dynamics_gradient_single_timing<float,false>(hd_data,d_robotModel,GRAVITY,SINGLE_CALL_ITERS_GLOBAL,dim3(1,1,1),dimms,streams)",
        "batch_with_mem":     "grid::forward_dynamics_gradient<float,false>(d,m,GRAVITY,N,dim3(N,1,1),dimms,streams)",
        "batch_compute_only": "grid::forward_dynamics_gradient_compute_only<float,false>(d,m,GRAVITY,N,dim3(N,1,1),dimms)",
        "batch_label": "FORWARD_DYNAMICS_GRADIENT",
        "gate": None,
        "shared_mem_skip": "FD_DU_DYNAMIC_SHARED_MEM_BYTES",
    },
    # f_ext gradients (A1). Host wrappers write into gridData's d_dtau_dfext /
    # d_dqdd_dfext / d_did_du_dfext buffers (allocated in gen_init_gridData), so
    # the call convention matches the standard (hd_data, d_robotModel, N, ...)
    # shape — no gravity arg (RNEA bias is folded into the kernel) and no extra
    # caller buffer. See GRiDCodeGenerator/algorithms/_f_ext_gradient.py:
    # gen_f_ext_gradient_host (mode 0/1/2) and gen_f_ext_gradient_dq_host.
    "f_ext_gradient": {
        "single_call":        "grid::f_ext_gradient_single_timing<float>(hd_data,d_robotModel,SINGLE_CALL_ITERS_GLOBAL,dim3(1,1,1),dimms,streams)",
        "batch_with_mem":     "grid::f_ext_gradient<float>(d,m,N,dim3(N,1,1),dimms,streams)",
        "batch_compute_only": "grid::f_ext_gradient_compute_only<float>(d,m,N,dim3(N,1,1),dimms)",
        "batch_label": "F_EXT_GRADIENT",
        "gate": None,
        "shared_mem_skip": "F_EXT_GRAD_DYNAMIC_SHARED_MEM_BYTES",
    },
    "f_ext_gradient_dq": {
        "single_call":        "grid::f_ext_gradient_dq_single_timing<float>(hd_data,d_robotModel,SINGLE_CALL_ITERS_GLOBAL,dim3(1,1,1),dimms,streams)",
        "batch_with_mem":     "grid::f_ext_gradient_dq<float>(d,m,N,dim3(N,1,1),dimms,streams)",
        "batch_compute_only": "grid::f_ext_gradient_dq_compute_only<float>(d,m,N,dim3(N,1,1),dimms)",
        "batch_label": "F_EXT_GRADIENT_DQ",
        "gate": None,
        "shared_mem_skip": "F_EXT_GRAD_DQ_DYNAMIC_SHARED_MEM_BYTES",
    },
    # Joint-torque regressor (A1). The grid:: symbol is `inverse_dynamics_regressor`
    # (the registry key is `regressor`); its host wrapper takes an extra
    # CALLER-OWNED output buffer `d_Y` (10*NUM_BODIES*NUM_VEL floats per timestep)
    # that is NOT part of gridData. We provide it as a TU-static device buffer
    # sized for the batch max (256). The kernel writes out_size per timestep with
    # stride out_size, so the batch buffer is out_size*256. It also takes the
    # gravity arg (RNEA forward sweep). See _regressor.py:gen_inverse_dynamics_regressor_host.
    "regressor": {
        "single_call":        "static float *_d_Y_s=[]{float*p;cudaMalloc(&p,sizeof(float)*10*grid::NUM_BODIES*grid::NUM_VEL);return p;}(); grid::inverse_dynamics_regressor_single_timing<float>(hd_data,_d_Y_s,d_robotModel,GRAVITY,SINGLE_CALL_ITERS_GLOBAL,dim3(1,1,1),dimms,streams)",
        "batch_with_mem":     "static float *_d_Y_b=[]{float*p;cudaMalloc(&p,sizeof(float)*10*grid::NUM_BODIES*grid::NUM_VEL*256);return p;}(); grid::inverse_dynamics_regressor<float>(d,_d_Y_b,m,GRAVITY,N,dim3(N,1,1),dimms,streams)",
        "batch_compute_only": "static float *_d_Y_c=[]{float*p;cudaMalloc(&p,sizeof(float)*10*grid::NUM_BODIES*grid::NUM_VEL*256);return p;}(); grid::inverse_dynamics_regressor_compute_only<float>(d,_d_Y_c,m,GRAVITY,N,dim3(N,1,1),dimms)",
        "batch_label": "REGRESSOR",
        "gate": None,
        "shared_mem_skip": "INVERSE_DYNAMICS_REGRESSOR_DYNAMIC_SHARED_MEM_BYTES",
    },
    # FD parameter gradient dqdd/dpi = -Minv.Y (A1). grid:: symbol is
    # `fd_parameter_gradient`; like the regressor its host wrapper takes a
    # CALLER-OWNED output buffer `d_dqdd_dpi` (10*NUM_BODIES*NUM_VEL floats per
    # timestep, NOT in gridData) plus the gravity arg. TU-static device buffer,
    # batch sized for N=256. See _regressor.py:gen_fd_parameter_gradient_host.
    "fd_parameter_gradient": {
        "single_call":        "static float *_d_dpi_s=[]{float*p;cudaMalloc(&p,sizeof(float)*10*grid::NUM_BODIES*grid::NUM_VEL);return p;}(); grid::fd_parameter_gradient_single_timing<float>(hd_data,_d_dpi_s,d_robotModel,GRAVITY,SINGLE_CALL_ITERS_GLOBAL,dim3(1,1,1),dimms,streams)",
        "batch_with_mem":     "static float *_d_dpi_b=[]{float*p;cudaMalloc(&p,sizeof(float)*10*grid::NUM_BODIES*grid::NUM_VEL*256);return p;}(); grid::fd_parameter_gradient<float>(d,_d_dpi_b,m,GRAVITY,N,dim3(N,1,1),dimms,streams)",
        "batch_compute_only": "static float *_d_dpi_c=[]{float*p;cudaMalloc(&p,sizeof(float)*10*grid::NUM_BODIES*grid::NUM_VEL*256);return p;}(); grid::fd_parameter_gradient_compute_only<float>(d,_d_dpi_c,m,GRAVITY,N,dim3(N,1,1),dimms)",
        "batch_label": "FD_PARAMETER_GRADIENT",
        "gate": None,
        "shared_mem_skip": "FD_PARAMETER_GRADIENT_DYNAMIC_SHARED_MEM_BYTES",
    },
    "end_effector_pose": {
        "single_call":        "grid::end_effector_pose_single_timing<float>(hd_data,d_robotModel,SINGLE_CALL_ITERS_GLOBAL,dim3(1,1,1),dimms,streams)",
        "batch_with_mem":     "grid::end_effector_pose<float>(d,m,N,dim3(N,1,1),dimms,streams)",
        "batch_compute_only": "grid::end_effector_pose_compute_only<float>(d,m,N,dim3(N,1,1),dimms)",
        "batch_label": "END_EFFECTOR_POSE",
        "gate": None,
        "shared_mem_skip": "EE_POS_DYNAMIC_SHARED_MEM_BYTES",
    },
    "end_effector_pose_gradient": {
        "single_call":        "grid::end_effector_pose_gradient_single_timing<float>(hd_data,d_robotModel,SINGLE_CALL_ITERS_GLOBAL,dim3(1,1,1),dimms,streams)",
        "batch_with_mem":     "grid::end_effector_pose_gradient<float>(d,m,N,dim3(N,1,1),dimms,streams)",
        "batch_compute_only": "grid::end_effector_pose_gradient_compute_only<float>(d,m,N,dim3(N,1,1),dimms)",
        "batch_label": "END_EFFECTOR_POSE_GRADIENT",
        "gate": None,
        "shared_mem_skip": "DEE_POS_DYNAMIC_SHARED_MEM_BYTES",
    },
    "end_effector_pose_hessian": {
        "single_call":        "grid::end_effector_pose_hessian_single_timing<float>(hd_data,d_robotModel,SINGLE_CALL_ITERS_GLOBAL,dim3(1,1,1),dimms,streams)",
        "batch_with_mem":     "grid::end_effector_pose_hessian<float>(d,m,N,dim3(N,1,1),dimms,streams)",
        "batch_compute_only": "grid::end_effector_pose_hessian_compute_only<float>(d,m,N,dim3(N,1,1),dimms)",
        "batch_label": "END_EFFECTOR_POSE_HESSIAN",
        "gate": None,
        "shared_mem_skip": "D2EE_POS_DYNAMIC_SHARED_MEM_BYTES",
    },
    "idsva_so": {
        "single_call":        "grid::idsva_so_single_timing<float>(hd_data,d_robotModel,GRAVITY,SINGLE_CALL_ITERS_GLOBAL,dim3(1,1,1),dimms,streams)",
        "batch_with_mem":     "grid::idsva_so<float>(d,m,GRAVITY,N,dim3(N,1,1),dimms,streams)",
        "batch_compute_only": "grid::idsva_so_compute_only<float>(d,m,GRAVITY,N,dim3(N,1,1),dimms)",
        "batch_label": "IDSVA_SO",
        "gate": "GRID_HAS_IDSVA_SO",
        "shared_mem_skip": "IDSVA_SO_BODY_FRAME_DYNAMIC_SHARED_MEM_BYTES",
    },
    "idsva_so_body_frame": {
        "single_call":        "grid::idsva_so_body_frame_host_single_timing<float>(hd_data,d_robotModel,GRAVITY,SINGLE_CALL_ITERS_GLOBAL,dim3(1,1,1),dimms,streams)",
        "batch_with_mem":     "grid::idsva_so_body_frame_host<float>(d,m,GRAVITY,N,dim3(N,1,1),dimms,streams)",
        "batch_compute_only": "grid::idsva_so_body_frame_host_compute_only<float>(d,m,GRAVITY,N,dim3(N,1,1),dimms)",
        "batch_label": "IDSVA_SO_BODY_FRAME",
        "gate": "GRID_HAS_IDSVA_SO_BODY_FRAME",
        "shared_mem_skip": "IDSVA_SO_BODY_FRAME_DYNAMIC_SHARED_MEM_BYTES",
    },
    "idsva_so_world_frame": {
        "single_call":        "grid::idsva_so_world_frame_host_single_timing<float>(hd_data,d_robotModel,GRAVITY,SINGLE_CALL_ITERS_GLOBAL,dim3(1,1,1),dimms,streams)",
        "batch_with_mem":     "grid::idsva_so_world_frame_host<float>(d,m,GRAVITY,N,dim3(N,1,1),dimms,streams)",
        "batch_compute_only": "grid::idsva_so_world_frame_host_compute_only<float>(d,m,GRAVITY,N,dim3(N,1,1),dimms)",
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
    # Centroidal / energy quick-wins (A1). Host wrappers write into gridData
    # buffers (d_c for the RNEA-bias families; d_com / d_ccrba / d_energy).
    #   - generalized_gravity / nonlinear_effects: RNEA-bias wrappers, take the
    #     gravity arg; signature mirrors `id` + gravity. Emitted whenever `id`
    #     is generated (always, under codegen_profile='all'). Shared smem macro
    #     is ID_BIAS_DYNAMIC_SHARED_MEM_BYTES for BOTH.
    #     See GRiDCodeGenerator/algorithms/_centroidal.py:gen_id_bias_host.
    #   - com: kinematics-domain, NO gravity / NO qd. ccrba: NO gravity (uses qd).
    #     energy: takes the gravity arg (uses qd). Output sizes: com=3+3*NUM_VEL,
    #     ccrba=6*NUM_VEL+6, energy=3.  See _centroidal.py:_gen_kin_centroidal_host.
    # NOTE: com/ccrba/energy are SKIPPED at codegen for MIMIC robots (their
    # per-body Jacobian fold is not mimic-reduced) — for a mimic robot these
    # grid:: symbols are absent and the TU would fail to compile. The current
    # sweep robots (iiwa14/go2/g1/h1_2) are all non-mimic, so no gate is needed
    # here; there is no GRID_HAS_* preprocessor macro emitted for these families
    # to gate on. (generalized_gravity / nonlinear_effects are always emitted.)
    "generalized_gravity": {
        "single_call":        "grid::generalized_gravity_single_timing<float>(hd_data,d_robotModel,GRAVITY,SINGLE_CALL_ITERS_GLOBAL,dim3(1,1,1),dimms,streams)",
        "batch_with_mem":     "grid::generalized_gravity<float>(d,m,GRAVITY,N,dim3(N,1,1),dimms,streams)",
        "batch_compute_only": "grid::generalized_gravity_compute_only<float>(d,m,GRAVITY,N,dim3(N,1,1),dimms)",
        "batch_label": "GENERALIZED_GRAVITY",
        "gate": None,
        "shared_mem_skip": "ID_BIAS_DYNAMIC_SHARED_MEM_BYTES",
    },
    "nonlinear_effects": {
        "single_call":        "grid::nonlinear_effects_single_timing<float>(hd_data,d_robotModel,GRAVITY,SINGLE_CALL_ITERS_GLOBAL,dim3(1,1,1),dimms,streams)",
        "batch_with_mem":     "grid::nonlinear_effects<float>(d,m,GRAVITY,N,dim3(N,1,1),dimms,streams)",
        "batch_compute_only": "grid::nonlinear_effects_compute_only<float>(d,m,GRAVITY,N,dim3(N,1,1),dimms)",
        "batch_label": "NONLINEAR_EFFECTS",
        "gate": None,
        "shared_mem_skip": "ID_BIAS_DYNAMIC_SHARED_MEM_BYTES",
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
        "single_call":        "grid::integrator_gradient_with_x_kp1_single_timing<float>(hd_data,d_robotModel,GRAVITY,static_cast<float>(0.01),SINGLE_CALL_ITERS_GLOBAL,dim3(1,1,1),dimms,streams)",
        "batch_with_mem":     "grid::integrator_gradient_with_x_kp1<float>(d,m,GRAVITY,static_cast<float>(0.01),N,dim3(N,1,1),dimms,streams)",
        "batch_compute_only": "grid::integrator_gradient_with_x_kp1_compute_only<float>(d,m,GRAVITY,static_cast<float>(0.01),N,dim3(N,1,1),dimms)",
        "batch_label": "INTEGRATOR_WITH_GRADIENT",
        "gate": "GRID_HAS_INTEGRATOR_GRADIENT",
        "shared_mem_skip": "INTEGRATOR_DU_DYNAMIC_SHARED_MEM_BYTES",
    },
}


def _algo_keys_in_registry_order() -> list[str]:
    """Return algo keys in ALGO_REGISTRY order, filtered to those in PER_ALGO_SPECS."""
    keys: list[str] = []
    missing: list[str] = []
    for entry in ALGO_REGISTRY:
        if entry.key in PER_ALGO_SPECS:
            keys.append(entry.key)
        else:
            missing.append(entry.key)
    if missing:
        # Skip (don't hard-fail) algos that have no bench spec yet — e.g. the
        # integrator family, whose host signature takes extra dt/IntegratorType
        # args and needs custom buffer setup (a separate follow-up). Warn so the
        # omission stays visible.
        print(f"  [grid] WARNING: skipping algos missing a PER_ALGO_SPECS row: {', '.join(missing)}",
              file=sys.stderr)
    return keys


def _gate_open(spec: dict) -> str:
    """Open preprocessor block for the per-algo gate. Returns '' when ungated."""
    return f"#if {spec['gate']}\n" if spec.get("gate") else ""


def _gate_close(spec: dict) -> str:
    return "#endif\n" if spec.get("gate") else ""


def _per_algo_single_tu_source(algo_key: str) -> str:
    """Source for timeGRiD_single_<algo>.cu: defines measure_<algo>_single_entry."""
    spec = PER_ALGO_SPECS[algo_key]
    skip_block = ""
    if "shared_mem_skip" in spec:
        # Match the monolithic wrapper's runtime skip: if the kernel's
        # requested dynamic shared mem exceeds the device cap, print a
        # "Single Call X SKIPPED" line and return early.
        skip_block = (
            f"    if (!grid_kernel_fits_device(grid::{spec['shared_mem_skip']}<float>())) {{\n"
            f"        printf(\"Single Call {spec['batch_label']} SKIPPED (kernel needs %zu bytes shared mem, exceeds device cap)\\n\",\n"
            f"               grid::{spec['shared_mem_skip']}<float>()); return;\n"
            f"    }}\n"
        )
    # Per-algo TU split fix: call init_grid_kernel_attrs<float>() once on first
    # entry so cudaFuncSetAttribute is applied to THIS TU's kernel stubs
    # (the launch goes through this TU's stubs, not the dispatcher main's).
    body = (
        f"{_gate_open(spec)}"
        f"void measure_{algo_key}_single_entry(cudaStream_t *streams, grid::robotModel<float> *d_robotModel, grid::gridData<float> *hd_data){{\n"
        f"    static const bool _attrs_set = []() {{ grid::init_grid_kernel_attrs<float>(); return true; }}();\n"
        f"    (void)_attrs_set;\n"
        f"    dim3 dimms = grid_timing_dimms();\n"
        f"{skip_block}"
        f"    {spec['single_call']};\n"
        f"}}\n"
        f"{_gate_close(spec)}"
    )
    return (
        "// AUTO-GENERATED by test/benchmarks/baselines/grid/run.py — do not hand-edit.\n"
        "// One TU per (algo, kind). Compiled with -rdc=true to keep the anti-LICM\n"
        "// machinery (volatile reload + __noinline__ grid_licm_barrier) effective\n"
        "// inside the _single_timing rep loops.\n"
        "#include \"timeGRiD_common.h\"\n"
        "\n"
        + body
    )


def _per_algo_batch_tu_source(algo_key: str) -> str:
    """Source for timeGRiD_batch_<algo>.cu: defines measure_<algo>_batch_entry."""
    spec = PER_ALGO_SPECS[algo_key]
    skip_block = ""
    if "shared_mem_skip" in spec:
        skip_block = (
            f"    if (!grid_kernel_fits_device(grid::{spec['shared_mem_skip']}<float>())) {{\n"
            f"        printf(\"[N:%d]: {spec['batch_label']} SKIPPED (kernel needs %zu bytes shared mem, exceeds device cap)\\n\",\n"
            f"               N, grid::{spec['shared_mem_skip']}<float>()); return;\n"
            f"    }}\n"
        )
    body = (
        f"{_gate_open(spec)}"
        f"void measure_{algo_key}_batch_entry(int N, cudaStream_t *streams, grid::robotModel<float> *m, grid::gridData<float> *d){{\n"
        f"    static const bool _attrs_set = []() {{ grid::init_grid_kernel_attrs<float>(); return true; }}();\n"
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


def _per_algo_single_main_source() -> str:
    """Dispatcher TU for the single-call binary."""
    keys = _algo_keys_in_registry_order()
    decls: list[str] = []
    calls: list[str] = []
    for k in keys:
        spec = PER_ALGO_SPECS[k]
        gate = spec.get("gate")
        prefix = f"#if {gate}\n" if gate else ""
        suffix = "#endif\n" if gate else ""
        decls.append(
            f"{prefix}"
            f"extern void measure_{k}_single_entry(cudaStream_t*, grid::robotModel<float>*, grid::gridData<float>*);\n"
            f"{suffix}"
        )
        calls.append(
            f"{prefix}"
            f"    measure_{k}_single_entry(streams, m, d);\n"
            f"{suffix}"
        )
    return (
        "// AUTO-GENERATED by test/benchmarks/baselines/grid/run.py — do not hand-edit.\n"
        "// Dispatcher for the per-algo single-call binary. One extern decl per algo;\n"
        "// calls them in ALGO_REGISTRY order to match the monolithic timeGRiD_single.cu.\n"
        "#include \"timeGRiD_common.h\"\n"
        "\n"
        + "".join(decls)
        + "\n"
        "int main(int argc, const char **argv){\n"
        "    bool floating_base = parse_floating_base_arg(argc, argv);\n"
        "    run_all_tests<float, 256>(floating_base, [&](cudaStream_t *streams, grid::robotModel<float> *m, grid::gridData<float> *d){\n"
        "#if !TEST_FOR_EQUIVALENCE\n"
        "        (void)floating_base;\n"
        + "".join(calls) +
        "#else\n"
        "        (void)floating_base; (void)streams; (void)m; (void)d;\n"
        "#endif\n"
        "    });\n"
        "    return 0;\n"
        "}\n"
    )


def _per_algo_batch_main_source() -> str:
    """Dispatcher TU for the batch binary."""
    keys = _algo_keys_in_registry_order()
    decls: list[str] = []
    calls: list[str] = []
    for k in keys:
        spec = PER_ALGO_SPECS[k]
        gate = spec.get("gate")
        prefix = f"#if {gate}\n" if gate else ""
        suffix = "#endif\n" if gate else ""
        decls.append(
            f"{prefix}"
            f"extern void measure_{k}_batch_entry(int, cudaStream_t*, grid::robotModel<float>*, grid::gridData<float>*);\n"
            f"{suffix}"
        )
        calls.append(
            f"{prefix}"
            f"        measure_{k}_batch_entry(N, streams, m, d);\n"
            f"{suffix}"
        )
    return (
        "// AUTO-GENERATED by test/benchmarks/baselines/grid/run.py — do not hand-edit.\n"
        "// Dispatcher for the per-algo batch binary. One extern decl per algo;\n"
        "// loops over N in {16, 32, 64, 128, 256} to match timeGRiD_batch.cu.\n"
        "#include \"timeGRiD_common.h\"\n"
        "\n"
        + "".join(decls)
        + "\n"
        "static void run_batch_at(int N, cudaStream_t *streams, grid::robotModel<float> *m, grid::gridData<float> *d){\n"
        + "".join(calls) +
        "}\n"
        "\n"
        "int main(int argc, const char **argv){\n"
        "    bool floating_base = parse_floating_base_arg(argc, argv);\n"
        "    run_all_tests<float, 256>(floating_base, [&](cudaStream_t *streams, grid::robotModel<float> *m, grid::gridData<float> *d){\n"
        "#if !TEST_FOR_EQUIVALENCE\n"
        "        (void)floating_base;\n"
        "        run_batch_at(16, streams, m, d);\n"
        "        run_batch_at(32, streams, m, d);\n"
        "        run_batch_at(64, streams, m, d);\n"
        "        run_batch_at(128, streams, m, d);\n"
        "        run_batch_at(256, streams, m, d);\n"
        "#else\n"
        "        (void)floating_base; (void)streams; (void)m; (void)d;\n"
        "#endif\n"
        "    });\n"
        "    return 0;\n"
        "}\n"
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


def generate_per_algo_sources(build_dir: Path) -> tuple[list[tuple[str, Path]], list[tuple[str, Path]], Path, Path]:
    """Write per-algo TUs + dispatcher mains into `build_dir`.

    Returns:
        (single_tus, batch_tus, single_main, batch_main)
    where each *_tus list is [(algo_key, source_path), ...] in registry order.
    """
    keys = _algo_keys_in_registry_order()
    single_tus: list[tuple[str, Path]] = []
    batch_tus: list[tuple[str, Path]] = []
    for k in keys:
        s_path = build_dir / f"timeGRiD_single_{k}.cu"
        b_path = build_dir / f"timeGRiD_batch_{k}.cu"
        _write_if_changed(s_path, _per_algo_single_tu_source(k))
        _write_if_changed(b_path, _per_algo_batch_tu_source(k))
        single_tus.append((k, s_path))
        batch_tus.append((k, b_path))
    single_main = build_dir / "timeGRiD_single_main.cu"
    batch_main  = build_dir / "timeGRiD_batch_main.cu"
    _write_if_changed(single_main, _per_algo_single_main_source())
    _write_if_changed(batch_main,  _per_algo_batch_main_source())
    return single_tus, batch_tus, single_main, batch_main


def _compile_one_source(
    source_path: Path,
    out_name: str,
    *,
    header_path: Path,
    arch: str,
    build_dir: Path,
    cxx_standard: str,
    compile_linalg_flags: list[str],
    link_linalg_flags: list[str],
    common_arch: list[str],
    ptxas_opt_level: int | None,
    split_compile: int | None,
    ofast_compile: str | None,
    runner_key: str,
    ccache_prefix: list[str],
    nvcc: str,
) -> Path:
    """Compile one .cu source → .exe with the given flags. Returns the .exe
    path. Two-stage (compile→link) so ccache caches the heavy compile pass."""
    object_path = build_dir / f"{out_name}.o"
    binary_path = build_dir / f"{out_name}.exe"
    cached_binary = CACHE_ROOT / "grid_benchmarks" / runner_key / f"{out_name}.exe"

    if cached_binary.exists():
        shutil.copyfile(cached_binary, binary_path)
        os.chmod(binary_path, 0o755)
        return binary_path

    compile_cmd = [
        *ccache_prefix,
        nvcc, cxx_standard, "-c", "-o", str(object_path), str(source_path),
        f"-DGRID_HEADER_FILE=\"{header_path}\"",
        *common_arch,
        "-O3", "-ftz=true", "-prec-div=false", "-prec-sqrt=false",
        *compile_linalg_flags,
    ]
    if ptxas_opt_level is not None:
        compile_cmd.extend(["-Xptxas", f"-O{int(ptxas_opt_level)}"])
    if split_compile is not None:
        compile_cmd.extend([f"--split-compile={int(split_compile)}"])
    if ofast_compile is not None:
        compile_cmd.extend([f"-Ofc={ofast_compile}"])
    result = subprocess.run(compile_cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(
            f"nvcc compile (-c) failed for {source_path.name}:\n{result.stdout}\n{result.stderr}"
        )

    link_cmd = [
        nvcc, "-o", str(binary_path), str(object_path),
        *common_arch,
        *link_linalg_flags,
    ]
    result = subprocess.run(link_cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(
            f"nvcc link failed for {source_path.name}:\n{result.stdout}\n{result.stderr}"
        )
    cached_binary.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(binary_path, cached_binary)
    os.chmod(cached_binary, 0o755)
    return binary_path


def _compile_to_object(
    source_path: Path,
    object_path: Path,
    *,
    header_path: Path,
    cxx_standard: str,
    compile_linalg_flags: list[str],
    common_arch: list[str],
    ptxas_opt_level: int | None,
    split_compile: int | None,
    ofast_compile: str | None,
    ccache_prefix: list[str],
    nvcc: str,
) -> None:
    """Compile one .cu → .o (no link). Used by the per-algo split compile path."""
    # `timeGRiD_common.h` lives next to run.py (THIS_DIR), but the per-algo
    # generated .cu files live in test/benchmarks/results/. nvcc only adds
    # the source file's directory to its include search path by default, so
    # we need an explicit -I pointing at THIS_DIR for the common header.
    compile_cmd = [
        *ccache_prefix,
        nvcc, cxx_standard, "-c", "-o", str(object_path), str(source_path),
        f"-DGRID_HEADER_FILE=\"{header_path}\"",
        f"-I{THIS_DIR}",
        *common_arch,
        "-O3", "-ftz=true", "-prec-div=false", "-prec-sqrt=false",
        *compile_linalg_flags,
    ]
    if ptxas_opt_level is not None:
        compile_cmd.extend(["-Xptxas", f"-O{int(ptxas_opt_level)}"])
    if split_compile is not None:
        compile_cmd.extend([f"--split-compile={int(split_compile)}"])
    if ofast_compile is not None:
        compile_cmd.extend([f"-Ofc={ofast_compile}"])
    result = subprocess.run(compile_cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(
            f"nvcc compile (-c) failed for {source_path.name}:\n{result.stdout}\n{result.stderr}"
        )


def _link_objects(
    object_paths: list[Path],
    binary_path: Path,
    *,
    common_arch: list[str],
    link_linalg_flags: list[str],
    use_rdc: bool,
    nvcc: str,
) -> None:
    """Link a set of .o files into one .exe. -rdc=true must be passed at link too
    if any of the objects were compiled with it (single-call binary)."""
    link_cmd = [nvcc, "-o", str(binary_path), *[str(p) for p in object_paths], *common_arch]
    if use_rdc:
        link_cmd.append("-rdc=true")
    link_cmd.extend(link_linalg_flags)
    result = subprocess.run(link_cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(
            f"nvcc link failed:\n{result.stdout}\n{result.stderr}"
        )


def _compile_per_algo_binary(
    *,
    kind: str,
    tu_sources: list[Path],
    main_source: Path,
    out_name: str,
    header_path: Path,
    build_dir: Path,
    cxx_standard: str,
    compile_linalg_flags: list[str],
    link_linalg_flags: list[str],
    common_arch: list[str],
    ptxas_opt_level: int | None,
    split_compile: int | None,
    ofast_compile: str | None,
    use_rdc: bool,
    runner_key: str,
    ccache_prefix: list[str],
    nvcc: str,
    max_workers: int,
) -> Path:
    """Compile all per-algo TUs + the dispatcher main in parallel, then link.
    Returns the final binary path. Caches the linked binary by runner_key."""
    binary_path = build_dir / f"{out_name}.exe"
    cached_binary = CACHE_ROOT / "grid_benchmarks" / runner_key / f"{out_name}.exe"

    if cached_binary.exists():
        shutil.copyfile(cached_binary, binary_path)
        os.chmod(binary_path, 0o755)
        print(f"  [grid] {out_name} cache hit (key={runner_key[:12]})")
        return binary_path

    all_sources = list(tu_sources) + [main_source]
    # Object path is alongside the .cu so ccache (compiler invocation hash)
    # sees a stable -o argument; one .o per .cu.
    object_paths = [build_dir / (s.stem + ".o") for s in all_sources]

    def _compile_one(idx: int) -> tuple[int, str | None]:
        s = all_sources[idx]
        o = object_paths[idx]
        try:
            t0 = time.perf_counter()
            _compile_to_object(
                s, o,
                header_path=header_path,
                cxx_standard=cxx_standard,
                compile_linalg_flags=compile_linalg_flags,
                common_arch=common_arch,
                ptxas_opt_level=ptxas_opt_level,
                split_compile=split_compile,
                ofast_compile=ofast_compile,
                ccache_prefix=ccache_prefix,
                nvcc=nvcc,
            )
            dt = time.perf_counter() - t0
            return (idx, f"OK {s.name} ({dt:.1f}s)")
        except Exception as e:
            return (idx, f"FAIL {s.name}: {e}")

    print(
        f"  [grid] {out_name}: compiling {len(all_sources)} TUs in parallel "
        f"(workers={max_workers}, rdc={use_rdc})..."
    )
    t_start = time.perf_counter()
    failures: list[str] = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as ex:
        futures = {ex.submit(_compile_one, i): i for i in range(len(all_sources))}
        for fut in concurrent.futures.as_completed(futures):
            _, msg = fut.result()
            if msg.startswith("FAIL"):
                failures.append(msg)
    if failures:
        raise RuntimeError(f"per-algo compile failed:\n" + "\n".join(failures))
    compile_secs = time.perf_counter() - t_start
    print(f"  [grid] {out_name}: compiled {len(all_sources)} TUs in {compile_secs:.1f}s")

    t_link = time.perf_counter()
    _link_objects(
        object_paths, binary_path,
        common_arch=common_arch,
        link_linalg_flags=link_linalg_flags,
        use_rdc=use_rdc,
        nvcc=nvcc,
    )
    print(f"  [grid] {out_name}: linked in {time.perf_counter() - t_link:.1f}s")

    cached_binary.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(binary_path, cached_binary)
    os.chmod(cached_binary, 0o755)
    return binary_path


def compile_binaries(
    header_path: Path,
    arch: str,
    build_dir: Path,
    no_recompile: bool = False,
    no_rdc: bool = False,
    single_call_iters: int | None = None,
    batch_iters: int | None = None,
    ptxas_opt_level: int | None = None,
    split_compile: int | None = None,
    ofast_compile: str | None = None,
    per_algo_tus: bool = True,
    compile_workers: int | None = None,
    tier: str | None = None,
) -> tuple[Path, Path]:
    """Compile the single-call binary (with -rdc=true) and the batch binary
    (without -rdc=true) against the generated header. Returns
    (single_binary, batch_binary).

    When `per_algo_tus=True` (default), each binary is built from N small
    per-algo TUs + a dispatcher main, all compiled in parallel via a thread
    pool. This was added as P6-7b to cut cicc-bound sweep wall time by
    ~3-5×. Set `per_algo_tus=False` (or pass `--no-per-algo-tus` on the
    command line) to fall back to the monolithic timeGRiD_single.cu /
    timeGRiD_batch.cu translation units."""
    # Hash all three monolithic sources (.cu + .cu + .h) so a touch of the
    # shared header busts both caches. When per_algo_tus=True the
    # auto-generated per-algo TUs are themselves derived from PER_ALGO_SPECS
    # + the registry, both of which are content in this file — see the
    # `per_algo_specs_hash` field added to runner_key below.
    source_hash = _hash_bytes(
        _hash_file(TIMING_SOURCE_SINGLE).encode() +
        _hash_file(TIMING_SOURCE_BATCH).encode() +
        _hash_file(TIMING_SOURCE_COMMON).encode()
    )
    header_hash = _hash_file(header_path)
    # Bust the binary cache whenever PER_ALGO_SPECS, the registry order, or
    # the generator templates change — any of which can alter the bytes of
    # the per-algo .cu files we emit.
    per_algo_specs_hash = _hash_bytes(
        json.dumps({
            "specs": {k: PER_ALGO_SPECS[k] for k in _algo_keys_in_registry_order()},
            "single_main_src": _per_algo_single_main_source(),
            "batch_main_src":  _per_algo_batch_main_source(),
            "per_algo_single_template_v": 1,  # bump if template body changes
            "per_algo_batch_template_v":  1,
        }, sort_keys=True).encode()
    )

    # c++17 baseline: needed for inline variables (timeGRiD_common.h carries
    # the random-state singletons) and for clean ODR semantics in the
    # per-algo TU split.
    cxx_standard = "-std=c++17"
    linalg_flags: list[str] = []

    # Override defaults in test/benchmarks/baselines/util/experiment_helpers.h
    # (SINGLE_CALL_ITERS_GLOBAL=10000, TEST_ITERS_GLOBAL=100) by re-defining at
    # the compile line. Bumping iter counts is the simplest way to reduce
    # measurement noise on fast kernels.
    if single_call_iters is not None:
        linalg_flags.append(f"-DSINGLE_CALL_ITERS_GLOBAL={int(single_call_iters)}")
    if batch_iters is not None:
        linalg_flags.append(f"-DTEST_ITERS_GLOBAL={int(batch_iters)}")
    # Resource-tier macro override: Phase 4 perf-validation sweep launches
    # every kernel at the chosen tier (PERF default; LITE/MINIMAL pick the
    # spill body). Defaulting via #ifndef in grid.cuh keeps PERF as the
    # baseline when --tier is not passed.
    if tier is not None:
        tier_macro = {
            "shared": "grid::TIER_SHARED",
            "perf": "grid::TIER_SHARED",  # deprecated alias for "shared"
            "lite": "grid::TIER_LITE",
            "minimal": "grid::TIER_MINIMAL",
        }.get(tier)
        if tier_macro is None:
            raise ValueError(f"unknown tier: {tier!r}; expected shared/lite/minimal (perf=shared alias)")
        linalg_flags.append(f"-DGRID_DEFAULT_RESOURCE_TIER={tier_macro}")
    # Optional: GRID_BENCH_D2EE_ONLY=1 in env forwards a -D into nvcc so the
    # batch dispatcher's #if GRID_BENCH_D2EE_ONLY path is taken, measuring only
    # ee_pose_hessian. Cuts compile + run time dramatically for d2ee-focused
    # sweeps. Default off.
    if os.environ.get("GRID_BENCH_D2EE_ONLY", "0") != "0":
        linalg_flags.append("-DGRID_BENCH_D2EE_ONLY=1")

    runner_key = _hash_bytes(
        json.dumps({
            "source_hash": source_hash,
            "header_hash": header_hash,
            "cuda_arch": arch,
            "cxx_standard": cxx_standard,
            "linalg_flags": linalg_flags,
            "no_rdc": no_rdc,
            "ptxas_opt_level": ptxas_opt_level,
            "split_compile": split_compile,
            "ofast_compile": ofast_compile,
            "split_binaries": True,  # cache-bust against the pre-split layout
            "per_algo_tus": per_algo_tus,
            "per_algo_specs_hash": per_algo_specs_hash if per_algo_tus else None,
        }, sort_keys=True).encode()
    )[:24]

    print(
        f"  [grid] compiling timeGRiD_{{single,batch}}.cu "
        f"(arch=sm_{arch}, linalg=glass, cache key={runner_key[:12]})..."
    )
    nvcc = shutil.which("nvcc")
    if nvcc is None:
        raise RuntimeError("nvcc not found — install CUDA Toolkit to compile timeGRiD")

    # Two-stage compile→link so ccache caches the heavy compile pass.
    # ccache treats single-shot `nvcc source.cu -o exe` as a link op and
    # bypasses caching. Disable via GRID_NO_CCACHE=1.
    ccache_prefix: list[str] = []
    if not os.environ.get("GRID_NO_CCACHE"):
        ccache = shutil.which("ccache")
        if ccache is not None:
            ccache_prefix = [ccache]
            print(f"  [grid] ccache enabled (CCACHE_DIR={os.environ.get('CCACHE_DIR', '~/.cache/ccache')})")

    # Partition shared linalg_flags into compile-only vs link-only. Linker
    # flags: -L<dir>, -l<lib>, -dlto. Everything else goes to compile.
    # (-rdc=true is NOT in linalg_flags here — it's added per-TU below.)
    compile_linalg_flags_shared: list[str] = []
    link_linalg_flags_shared: list[str] = []
    for f in linalg_flags:
        if f.startswith(("-L", "-l")) or f == "-dlto":
            link_linalg_flags_shared.append(f)
        else:
            compile_linalg_flags_shared.append(f)

    common_arch = ["-gencode", f"arch=compute_{arch},code=sm_{arch}"]

    # Per-TU flag sets. single needs -rdc=true (unless --no-rdc was passed)
    # for the anti-LICM machinery; batch does NOT, so nvcc can inline
    # ::glass::* / dot_prod / grid_xhom_or_dxhom_ptr aggressively.
    def _flags_for(use_rdc: bool) -> tuple[list[str], list[str]]:
        compile_flags = list(compile_linalg_flags_shared)
        link_flags = list(link_linalg_flags_shared)
        if use_rdc:
            compile_flags.append("-rdc=true")
            link_flags.append("-rdc=true")
        return compile_flags, link_flags

    single_rdc = not no_rdc
    batch_rdc = False  # batch never wants -rdc — that's the whole point of the split
    single_c, single_l = _flags_for(single_rdc)
    batch_c,  batch_l  = _flags_for(batch_rdc)

    t_total = time.perf_counter()
    if per_algo_tus:
        # Per-algo TU split (P6-7b). Each algo gets its own small TU; one
        # dispatcher main per kind. nvcc compiles all in parallel.
        single_tu_pairs, batch_tu_pairs, single_main, batch_main = generate_per_algo_sources(build_dir)
        # Workers: don't oversubscribe. nvcc forks cicc + ptxas + cudafe1
        # subprocesses already, so 0.75x CPU count is the sweet spot we've
        # seen empirically.
        if compile_workers is None:
            cpu = os.cpu_count() or 4
            compile_workers = max(2, int(cpu * 0.75))

        # Build single and batch INDEPENDENTLY so one failing (e.g. the
        # rdc-only single build hitting a ptxas regcount error on integrator
        # kernels) doesn't take down the other — the batch (no-rdc) build
        # inlines and still produces data. Only fail the combo if BOTH die.
        single_binary = None
        try:
            single_binary = _compile_per_algo_binary(
                kind="single",
                tu_sources=[p for (_k, p) in single_tu_pairs],
                main_source=single_main,
                out_name="timeGRiD_single",
                header_path=header_path,
                build_dir=build_dir,
                cxx_standard=cxx_standard,
                compile_linalg_flags=single_c,
                link_linalg_flags=single_l,
                common_arch=common_arch,
                ptxas_opt_level=ptxas_opt_level,
                split_compile=split_compile,
                ofast_compile=ofast_compile,
                use_rdc=single_rdc,
                runner_key=runner_key,
                ccache_prefix=ccache_prefix,
                nvcc=nvcc,
                max_workers=compile_workers,
            )
        except Exception as e:
            print(f"  [grid] WARNING: single-call build failed (batch will still run): {e}", file=sys.stderr)
        batch_binary = None
        try:
            batch_binary = _compile_per_algo_binary(
                kind="batch",
                tu_sources=[p for (_k, p) in batch_tu_pairs],
                main_source=batch_main,
                out_name="timeGRiD_batch",
                header_path=header_path,
                build_dir=build_dir,
                cxx_standard=cxx_standard,
                compile_linalg_flags=batch_c,
                link_linalg_flags=batch_l,
                common_arch=common_arch,
                ptxas_opt_level=ptxas_opt_level,
                split_compile=split_compile,
                ofast_compile=ofast_compile,
                use_rdc=batch_rdc,
                runner_key=runner_key,
                ccache_prefix=ccache_prefix,
                nvcc=nvcc,
                max_workers=compile_workers,
            )
        except Exception as e:
            print(f"  [grid] WARNING: batch build failed (single will still run): {e}", file=sys.stderr)
        if single_binary is None and batch_binary is None:
            raise RuntimeError("both single-call and batch builds failed")
        print(
            f"  [grid] per-algo TU compile+link finished in "
            f"{time.perf_counter() - t_total:.1f}s "
            f"(single rdc={single_rdc} ok={single_binary is not None}, "
            f"batch rdc={batch_rdc} ok={batch_binary is not None}, workers={compile_workers})"
        )
        return single_binary, batch_binary

    # ----- Monolithic fallback (pre-P6-7b layout) -----
    # Independent single/batch builds (see the per-algo path above for rationale).
    single_binary = None
    try:
        single_binary = _compile_one_source(
            TIMING_SOURCE_SINGLE, "timeGRiD_single",
            header_path=header_path, arch=arch, build_dir=build_dir,
            cxx_standard=cxx_standard,
            compile_linalg_flags=single_c, link_linalg_flags=single_l,
            common_arch=common_arch,
            ptxas_opt_level=ptxas_opt_level,
            split_compile=split_compile, ofast_compile=ofast_compile,
            runner_key=runner_key, ccache_prefix=ccache_prefix, nvcc=nvcc,
        )
    except Exception as e:
        print(f"  [grid] WARNING: single-call build failed (batch will still run): {e}", file=sys.stderr)
    batch_binary = None
    try:
        batch_binary = _compile_one_source(
            TIMING_SOURCE_BATCH, "timeGRiD_batch",
            header_path=header_path, arch=arch, build_dir=build_dir,
            cxx_standard=cxx_standard,
            compile_linalg_flags=batch_c, link_linalg_flags=batch_l,
            common_arch=common_arch,
            ptxas_opt_level=ptxas_opt_level,
            split_compile=split_compile, ofast_compile=ofast_compile,
            runner_key=runner_key, ccache_prefix=ccache_prefix, nvcc=nvcc,
        )
    except Exception as e:
        print(f"  [grid] WARNING: batch build failed (single will still run): {e}", file=sys.stderr)
    if single_binary is None and batch_binary is None:
        raise RuntimeError("both single-call and batch builds failed")
    print(
        f"  [grid] monolithic compile finished in "
        f"{time.perf_counter() - t_total:.1f}s "
        f"(single rdc={single_rdc} ok={single_binary is not None}, "
        f"batch rdc={batch_rdc} ok={batch_binary is not None})"
    )
    return single_binary, batch_binary


# ---------------------------------------------------------------------------
# Run and parse
# ---------------------------------------------------------------------------
def run_timing(binaries: tuple[Path | None, Path | None], base: str) -> str:
    """Run whichever of the single / batch binaries built, INDEPENDENTLY, so a
    crash (or missing build) in one doesn't lose the other's data. Returns
    concatenated stdout so parse_grid_output picks up `Single Call X us` lines
    (single binary) AND `[N:K]: X` lines (batch binary). Raises only if neither
    binary produced any output."""
    single_binary, batch_binary = binaries
    floating_arg = "T" if base == "floating" else "F"
    outputs = []
    produced_any = False
    for label, binary in (("single", single_binary), ("batch", batch_binary)):
        if binary is None:
            print(f"  [grid] skipping {label} run (build unavailable)", file=sys.stderr)
            continue
        result = subprocess.run(
            [str(binary), floating_arg],
            capture_output=True, text=True,
        )
        # Capture whatever stdout was emitted before any crash (a runtime smem
        # overflow aborts at init and yields nothing — that's fine, the other
        # binary still contributes).
        outputs.append(result.stdout)
        if result.returncode != 0:
            print(
                f"  [grid] WARNING: timeGRiD_{label} exited {result.returncode} "
                f"(continuing with the other binary); stderr:\n{result.stderr}",
                file=sys.stderr,
            )
            continue
        outputs.append(result.stderr)
        produced_any = True
    if not produced_any and not any(o.strip() for o in outputs):
        raise RuntimeError("neither single nor batch produced any timing output")
    return "\n".join(outputs)


# ---------------------------------------------------------------------------
# Per-(robot, base, algo) JOINT (tier × thread-count) autotune
#   C.4 (2026-05-29): thread-count sweep on the SHARED-tier batch binary.
#   T5  (2026-05-30): generalized to a joint (tier × threads) pick — the inner
#                     thread-grid sweep is run on EACH per-tier batch binary
#                     (shared / lite / minimal), and the winner is the global
#                     argmin µs over (tier, threads).
#
# The benchmark binary's kernels honor a runtime override of the per-block
# thread count via the GRID_AUTOTUNE_THREAD_COUNT env var (see
# timeGRiD_common.h::grid_resolve_threads_per_block). The per-tier *binaries*
# are produced by recompiling with -DGRID_DEFAULT_RESOURCE_TIER=grid::TIER_<X>;
# they are content-cached (see compile_binaries' runner_key), so the per-tier
# binaries that run_multi_version.py's BUILD phase already produced are reused
# here as cache hits — no new compiles in the common case.
#
# Cap-aware clipping (C.4 follow-up): each tier's launch_bounds is
# tier_max_threads<TIER>() (SHARED=MAX_PERF_LEVEL_THREADS, LITE=min(2×,768),
# MINIMAL=1024). Probing a thread count above that cap would exceed the kernel's
# __launch_bounds__ and fail the launch, so the grid is clipped per tier before
# sweeping. (cudaFuncAttributes.maxThreadsPerBlock is the runtime equivalent;
# the tier cap is the tighter static bound and is what we clip to.)
#
# Output (schema 2): result["algo_picks"][algo] = {
#     "schema": 2,
#     "tier_optimal": "shared"|"lite"|"minimal",
#     "threads_optimal": N,
#     "us_at_optimal": <µs>,
#     "sweep": {"shared": {threads: us, ...}, "lite": {...}, "minimal": {...}},
#     # flattened back-compat view (the C.4 schema-1 keys), pointing at the
#     # winning tier's per-threads sweep so existing generate_report.py
#     # consumers keep working unchanged:
#     "sweep_us": {threads: us, ...},
# }
# ---------------------------------------------------------------------------
# Narrowed from {32,64,96,128,192,256,384,512}: sweep data
# (results/perf_sweep_20260601_014157) shows winners cluster in 128-320 with
# almost none below 96 or above 384, so we drop the rarely-winning 32/64/512
# probes. The one-level refinement around each winner (_refine_grid_for_winner)
# still probes the immediate neighbors, so genuine edge-case optima are not
# missed. Override with --autotune-thread-grid.
DEFAULT_AUTOTUNE_THREAD_GRID: tuple[int, ...] = (96, 128, 192, 256, 320, 384)
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


def build_tier_binaries(
    header_path: Path,
    arch: str,
    build_dir: Path,
    *,
    base: str,
    mode: str,
    tiers: tuple[str, ...] = AUTOTUNE_TIERS,
    **compile_kwargs,
) -> dict[str, Path]:
    """Build (or cache-hit) the per-tier binary needed for the autotune sweep.

    Reuses `compile_binaries` per tier; because compile_binaries content-keys its
    binary cache (runner_key includes the GRID_DEFAULT_RESOURCE_TIER macro), the
    per-tier binaries that run_multi_version.py's BUILD phase already compiled are
    cache hits here — no new compiles in the common case.

    `mode` selects which binary the autotune needs: 'batch'/'both' → the batch
    binary; 'single' → the single binary. Returns {tier: binary_path} omitting
    tiers whose required binary failed to build.
    """
    want_single = mode == "single"
    out: dict[str, Path] = {}
    for tier in tiers:
        try:
            single_bin, batch_bin = compile_binaries(
                header_path, arch, build_dir, tier=tier, **compile_kwargs,
            )
        except Exception as e:  # noqa: BLE001 — collect, don't crash the sweep
            print(f"  [autotune] WARN: tier={tier} build failed, skipping: {e}",
                  file=sys.stderr)
            continue
        binary = single_bin if want_single else batch_bin
        if binary is None:
            print(f"  [autotune] WARN: tier={tier} {'single' if want_single else 'batch'} "
                  f"binary unavailable, skipping", file=sys.stderr)
            continue
        # compile_binaries returns the WORKING-DIR binary (build_dir/<name>.exe),
        # which it overwrites on every tier (each tier's cache binary is copied
        # onto the same path). Snapshot each tier's binary to a tier-stamped
        # filename so the three coexist — otherwise all tiers would alias the
        # last-built file and the sweep (and the tier-equivalence dedup) would
        # see them as identical regardless of their real per-tier contents.
        stamped = binary.with_name(f"{binary.stem}__tier_{tier}{binary.suffix}")
        try:
            shutil.copyfile(binary, stamped)
            os.chmod(stamped, 0o755)
            out[tier] = stamped
        except OSError as e:
            print(f"  [autotune] WARN: tier={tier} could not snapshot binary "
                  f"({e}); using shared working path (tiers may alias)",
                  file=sys.stderr)
            out[tier] = binary
    return out


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


def _update_autotune_best(best_path: Path, robot: str, base: str,
                          algo_picks: dict[str, dict], meta: dict) -> None:
    """Merge this (robot, base)'s autotune winners into the canonical
    autotune_best_<host>.json artifact (read-modify-write; other cells preserved).

    Layout:
        {"metadata": {...host/gpu...},
         "best": {robot: {base: {algo: {tier, threads, us}}}}}
    """
    doc: dict = {}
    if best_path.exists():
        try:
            doc = json.loads(best_path.read_text())
        except (OSError, json.JSONDecodeError):
            doc = {}
    doc.setdefault("metadata", {})
    # Keep a light host/gpu fingerprint so a best file isn't silently reused on
    # a different GPU (the winners are device-specific).
    for k in ("hostname", "gpu_name", "cuda_arch"):
        if k in meta:
            doc["metadata"][k] = meta[k]
    best = doc.setdefault("best", {})
    cell = best.setdefault(robot, {}).setdefault(base, {})
    for algo, info in algo_picks.items():
        cell[algo] = {
            "tier": info["tier_optimal"],
            "threads": int(info["threads_optimal"]),
            "us": float(info["us_at_optimal"]),
        }
    best_path.parent.mkdir(parents=True, exist_ok=True)
    best_path.write_text(json.dumps(doc, indent=2, sort_keys=True) + "\n")


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


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(description="Run GRiD benchmark for one robot/base")
    parser.add_argument("--robot", required=True, choices=list(ROBOT_DESCRIPTION_MODULE))
    parser.add_argument("--base", required=True, choices=["fixed", "floating"])
    parser.add_argument("--output", type=Path, default=None,
                        help="JSON output path (default: results/<robot>_<base>_grid_<host>.json)")
    parser.add_argument("--no-recompile", action="store_true",
                        help="Use cached binary even if header changed")
    parser.add_argument("--ee-frame", default=None,
                        help="EE target joint/link name for generator (default: per-robot canonical)")
    parser.add_argument("--no-rdc", action="store_true",
                        default=os.environ.get("GRID_BENCH_NO_RDC", "0") == "1",
                        help="Drop -rdc=true from the compile line. Speeds up ptxas on older "
                             "toolkits/GPUs at the cost of LICM defeat: single-call timings "
                             "for _single_timing kernels may elide their internal rep loop. "
                             "Batch timings (N=16..256) are unaffected. Use when builds hang.")
    parser.add_argument("--no-licm-barrier", action="store_true",
                        default=os.environ.get("GRID_NO_LICM_BARRIER", "0") == "1",
                        help="Suppress the anti-LICM machinery in codegen (volatile reload + "
                             "__noinline__ grid_licm_barrier call inside _single_timing rep loops). "
                             "Strongest hammer for ptxas hangs on floating-base kernels. Sets "
                             "GRID_NO_LICM_BARRIER=1 for the codegen subprocess. Batch timings "
                             "unaffected; single-call may LICM-elide.")
    parser.add_argument("--ptxas-opt-level", type=int, default=None,
                        choices=[0, 1, 2, 3],
                        help="Pass `-Xptxas -O<n>` to nvcc, lowering the device-backend "
                             "(ptxas / SASS) optimization tier. SM_86-SPECIFIC WORKAROUND: "
                             "on sm_86 / CUDA 12.6, ptxas -O3 wedges at 100%% CPU on heavy "
                             "floating-base kernels. Not needed on Blackwell (sm_120). "
                             "Default: nvcc default (-O3 to ptxas).")
    parser.add_argument("--single-call-iters", type=int, default=None,
                        help="Override SINGLE_CALL_ITERS_GLOBAL (default 10000). Inner-kernel "
                             "rep count for single-call timings; bump for more stable medians "
                             "on noisy machines.")
    parser.add_argument("--batch-iters", type=int, default=None,
                        help="Override TEST_ITERS_GLOBAL (default 100). Outer rep count for "
                             "batch timings at each N; bump for more stable medians.")
    parser.add_argument("--split-compile", type=int, default=None,
                        help="Pass `--split-compile=N` to nvcc (12.x). Parallelizes cicc "
                             "optimization passes across N threads within a single TU "
                             "(0 = all CPU cores).\n\n"
                             "*** DO NOT USE FOR MEASUREMENT RUNS *** — empirically defeats "
                             "the anti-LICM machinery (volatile reload + __noinline__ "
                             "grid_licm_barrier) at ALL values N>=2. Single-call and "
                             "batch-compute-only timings collapse to ~0us / launch-overhead "
                             "(~2us) for FD/ABA/MINV/ID_DU/FD_DU/EE_POSE_GRAD. Use ONLY for "
                             "non-timing dev iteration (e.g., verifying codegen output "
                             "compiles).")
    parser.add_argument("--ofast-compile", choices=["min", "mid", "max"], default=None,
                        help="Pass `-Ofc=<level>` to nvcc (12.x). Fast-compile mode for "
                             "device code: 'min' (mild compile-speed win), 'mid' (balanced), "
                             "'max' (focuses only on fastest compilation, disables many "
                             "optimizations). Trades device-code runtime perf for compile "
                             "time — opt-in dev knob, NOT for perf measurement runs.")
    # Per-algo TU split (P6-7b). Default: OFF. Correctness fix landed
    # (init_grid_kernel_attrs<T>() called via static initializer with
    # __forceinline__ so &kernel<T> resolves to per-TU stubs), but the
    # speedup hasn't materialized — measured 458s vs 263s monolithic for
    # iiwa14_floating, because each per-algo TU still parses the full
    # grid.cuh and cicc is parse-bound for small robots. Worth revisiting
    # with precompiled headers or finer-grained codegen. Toggle via
    # --per-algo-tus or GRID_BENCH_PER_ALGO_TUS=1.
    _per_algo_default = os.environ.get("GRID_BENCH_PER_ALGO_TUS", "0") != "0"
    parser.set_defaults(per_algo_tus=_per_algo_default)
    per_algo_group = parser.add_mutually_exclusive_group()
    per_algo_group.add_argument("--per-algo-tus", dest="per_algo_tus", action="store_true",
                                help="Use per-algo TU split (P6-7b). Compiles N small TUs per "
                                     "binary in parallel. Default ON; ~3-5× faster sweep wall time.")
    per_algo_group.add_argument("--no-per-algo-tus", dest="per_algo_tus", action="store_false",
                                help="Fall back to the monolithic timeGRiD_{single,batch}.cu "
                                     "translation units (pre-P6-7b layout).")
    parser.add_argument("--compile-workers", type=int, default=None,
                        help="Thread pool size for the per-algo parallel compile. Default: "
                             "0.75 × CPU count (capped at 2 minimum). nvcc forks cicc/ptxas "
                             "subprocesses already, so don't oversubscribe.")
    parser.add_argument("--tier", default=None, choices=["shared", "perf", "lite", "minimal"],
                        help="Resource-tier override. Compiles bench with "
                             "-DGRID_DEFAULT_RESOURCE_TIER=grid::TIER_<X>. SHARED (a.k.a. the "
                             "deprecated alias 'perf') is the default and preserves current "
                             "behavior; LITE/MINIMAL launch the spill bodies "
                             "with their tier-specific launch_bounds + smem. Used for Phase 4 "
                             "per-tier perf validation; output JSON gains a 'tier' field.")
    parser.add_argument("--build-dir", type=Path, default=None,
                        help="Override the working build directory (default: "
                             "test/benchmarks/results). Give each parallel invocation its own "
                             "dir so the per-case working grid.cuh / .o / .exe don't collide "
                             "(the binary cache is content-keyed + shared, so cache hits still "
                             "work across dirs). Used by the orchestrator's parallel build phase.")
    parser.add_argument("--compile-only", action="store_true",
                        help="Compile + populate the binary cache, then exit WITHOUT timing. "
                             "Used by the orchestrator to fan compiles across cores in a build "
                             "phase; the serial measure phase then re-runs with --no-recompile "
                             "(instant cache hit) so timing stays isolated on the GPU.")
    parser.add_argument("--autotune-threads", action="store_true",
                        help="After the standard timing run, do a JOINT (tier × thread-count) "
                             "autotune: for each tier (shared/lite/minimal) sweep a small grid of "
                             "per-block thread counts (default: 96,128,192,256,320,384 + "
                             "one-level refinement, clipped per tier to its launch_bounds cap) and "
                             "pick the global min-µs/sample winner (tier, threads) per algo. The "
                             "per-tier binaries are reused from the content-keyed binary cache (no "
                             "new compiles in the common case). Picks land in the JSON under "
                             "'algo_picks[algo]' (schema 2) = {'tier_optimal','threads_optimal',"
                             "'us_at_optimal','sweep':{tier:{threads:us}},'sweep_us':{threads:us}}. "
                             "Default OFF; opt-in. Thread overrides are applied via "
                             "GRID_AUTOTUNE_THREAD_COUNT env var read at first launch by "
                             "timeGRiD_common.h::grid_timing_dimms.")
    parser.add_argument("--autotune-thread-grid", type=str, default=None,
                        help="Comma-separated thread counts to sweep when --autotune-threads is "
                             "set. Default: '96,128,192,256,320,384'. Useful for narrowing "
                             "the sweep on slow robots (e.g. '128,256,384' for a quick re-tune).")
    parser.add_argument("--autotune-N", type=int, default=DEFAULT_AUTOTUNE_N,
                        help=f"Batch size to autotune on (default: {DEFAULT_AUTOTUNE_N}). The "
                             "winner is the (tier, thread count) that minimizes "
                             "batch_<N>_compute_only µs/sample.")
    parser.add_argument("--autotune-mode", default="batch", choices=["batch", "single", "both"],
                        help="Which timing path the autotune minimizes. 'batch' (default) tunes "
                             "the batch-N compute-only path; 'single' tunes the single-call "
                             "(single_us) path; 'both' runs each and writes batch picks under "
                             "'algo_picks' + single picks under 'algo_picks_single'.")
    args = parser.parse_args()

    ee_frame = args.ee_frame or DEFAULT_EE_FRAMES.get(args.robot, "")
    # MUST be absolute: the generated header path is baked into the nvcc compile
    # line (-DGRID_HEADER_FILE=...) and nvcc may run from a different cwd, so a
    # relative --build-dir (e.g. from a relative --output-dir) would fail to
    # resolve the header. The default REPO_ROOT-based path was already absolute.
    build_dir = (args.build_dir if args.build_dir is not None else (
        REPO_ROOT / "test" / "benchmarks" / "results")).resolve()
    build_dir.mkdir(parents=True, exist_ok=True)

    # Propagate the CLI flag to codegen via env var (the helpers read it at
    # call time). Must be set BEFORE generate_header() so codegen picks it up.
    if args.no_licm_barrier:
        os.environ["GRID_NO_LICM_BARRIER"] = "1"

    if args.output is None:
        import platform
        host = platform.node().replace(" ", "_")
        args.output = build_dir / f"{args.robot}_{args.base}_grid_{host}.json"

    arch = detect_cuda_arch()
    urdf_path = get_urdf_path(args.robot)
    print(f"[grid] {args.robot} {args.base} — URDF: {urdf_path}")

    try:
        header_path = generate_header(urdf_path, args.robot, args.base, ee_frame, build_dir, args.no_recompile)
    except Exception as e:
        print(f"  [grid] ERROR generating header: {e}", file=sys.stderr)
        sys.exit(1)

    t_compile = time.perf_counter()
    try:
        binaries = compile_binaries(
            header_path,
            arch,
            build_dir,
            args.no_recompile,
            no_rdc=args.no_rdc,
            single_call_iters=args.single_call_iters,
            batch_iters=args.batch_iters,
            ptxas_opt_level=args.ptxas_opt_level,
            split_compile=args.split_compile,
            ofast_compile=args.ofast_compile,
            per_algo_tus=args.per_algo_tus,
            compile_workers=args.compile_workers,
            tier=args.tier,
        )
    except Exception as e:
        print(f"  [grid] ERROR compiling binaries: {e}", file=sys.stderr)
        sys.exit(1)
    print(f"  [grid] compile wall time: {time.perf_counter() - t_compile:.1f}s")

    if args.compile_only:
        print(f"  [grid] --compile-only: binary cache populated, skipping timing.")
        return

    print(f"  [grid] running timing binaries (single + batch)...")
    try:
        output = run_timing(binaries, args.base)
    except Exception as e:
        print(f"  [grid] ERROR running binary: {e}", file=sys.stderr)
        sys.exit(1)

    timings = parse_grid_output(output)
    filled = fill_nulls(timings)

    # --autotune-threads: JOINT (tier × thread-count) autotune. For each tier
    # (shared/lite/minimal) we cache-hit/build its batch (or single) binary and
    # sweep the per-tier-cap-clipped thread grid; the winner is the global argmin
    # µs over (tier, threads). Opt-in; when off the output schema is identical to
    # pre-C.4. Thread overrides use the env-var read by
    # timeGRiD_common.h::grid_resolve_threads_per_block — no recompile per probe.
    algo_picks: dict[str, dict] = {}
    algo_picks_single: dict[str, dict] = {}
    autotune_grid_used: tuple[int, ...] | None = None
    if args.autotune_threads:
        if args.autotune_thread_grid:
            try:
                autotune_grid_used = tuple(int(x.strip()) for x in args.autotune_thread_grid.split(",")
                                           if x.strip())
            except ValueError:
                print(f"  [grid] ERROR: could not parse --autotune-thread-grid "
                      f"{args.autotune_thread_grid!r}", file=sys.stderr)
                sys.exit(1)
            if not autotune_grid_used:
                print(f"  [grid] ERROR: --autotune-thread-grid must list at least one int",
                      file=sys.stderr)
                sys.exit(1)
        else:
            autotune_grid_used = DEFAULT_AUTOTUNE_THREAD_GRID

        max_perf = _read_max_perf_level_threads(header_path)
        # Shared compile kwargs so the per-tier (cache-hit) rebuilds match the
        # main build's flags exactly (→ same runner_key → cache hit).
        _tier_compile_kwargs = dict(
            no_recompile=args.no_recompile, no_rdc=args.no_rdc,
            single_call_iters=args.single_call_iters, batch_iters=args.batch_iters,
            ptxas_opt_level=args.ptxas_opt_level, split_compile=args.split_compile,
            ofast_compile=args.ofast_compile, per_algo_tus=args.per_algo_tus,
            compile_workers=args.compile_workers,
        )

        # Modes to run: 'both' → batch then single.
        _modes = ["batch", "single"] if args.autotune_mode == "both" else [args.autotune_mode]
        for _mode in _modes:
            t_autotune = time.perf_counter()
            tier_binaries = build_tier_binaries(
                header_path, arch, build_dir, base=args.base, mode=_mode,
                **_tier_compile_kwargs,
            )
            if not tier_binaries:
                print(f"  [grid] WARN: --autotune-threads ({_mode}) requested but no per-tier "
                      f"binary available; skipping", file=sys.stderr)
                continue
            picks = _autotune_pick_winners(
                tier_binaries, args.base, autotune_grid_used, autotune_N=args.autotune_N,
                max_perf_level_threads=max_perf, mode=_mode,
            )
            if _mode == "single":
                algo_picks_single = picks
            else:
                algo_picks = picks
            print(f"  [grid] autotune ({_mode}) wall time: "
                  f"{time.perf_counter() - t_autotune:.1f}s "
                  f"({len(picks)} algos picked; tiers={list(tier_binaries)})")
            for algo, info in sorted(picks.items()):
                print(f"    [autotune:{_mode}] {algo:22s} tier={info['tier_optimal']:<7s} "
                      f"threads_optimal={info['threads_optimal']:>4d} "
                      f"us_at_optimal={info['us_at_optimal']:>8.2f}")

    meta = build_metadata(include_gpu=True)
    meta["robot"] = args.robot
    meta["base"] = args.base
    meta["ee_frame"] = ee_frame
    meta["cuda_arch"] = arch
    meta["grid_linalg_backend"] = "glass"
    # Canonicalize the reported tier name: None (no flag) and the deprecated
    # "perf" alias both report as "shared" (the TIER_SHARED default).
    meta["resource_tier"] = "shared" if (args.tier in (None, "perf")) else args.tier
    if args.autotune_threads:
        meta["autotune_threads"] = {
            "thread_grid": list(autotune_grid_used or DEFAULT_AUTOTUNE_THREAD_GRID),
            "autotune_N":  int(args.autotune_N),
            "mode": args.autotune_mode,
            "tiers": list(AUTOTUNE_TIERS),
            "schema": 2,
        }

    grid_block: dict = {"grid": filled}
    if args.autotune_threads:
        grid_block["algo_picks"] = algo_picks
        if algo_picks_single:
            grid_block["algo_picks_single"] = algo_picks_single
    result = {"metadata": meta, "results": {args.robot: {args.base: grid_block}}}
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(f"  [grid] results saved: {args.output}")

    # Canonical best artifact: a flat per-(robot, base, algo) record of the
    # autotuned (tier, threads) winner, merged across invocations so a full
    # sweep accumulates one authoritative file. generate_report.py's grid_best
    # column reads this.
    if args.autotune_threads and algo_picks:
        import platform
        host = platform.node().replace(" ", "_")
        best_path = (REPO_ROOT / "test" / "benchmarks" / "results"
                     / f"autotune_best_{host}.json")
        _update_autotune_best(best_path, args.robot, args.base, algo_picks, meta)
        print(f"  [grid] autotune_best updated: {best_path}")

    # Print quick summary: single + N=16 + N=256 compute-only so it's obvious
    # the batch tests actually ran. Full data (all 5 batch sizes, with_mem +
    # compute_only) lives in the JSON.
    def _us(entry, key):
        v = (entry.get(key) or {}).get("median") or (entry.get(key) or {}).get("mean")
        return f"{v:.2f}" if v is not None else "—"

    for algo, entry in sorted(filled.items()):
        if entry is None:
            print(f"    {algo}: null")
            continue
        single   = _us(entry, "single_us")
        n16_co   = _us(entry, "batch_16_compute_only_us")
        n256_co  = _us(entry, "batch_256_compute_only_us")
        print(f"    {algo:18s} single={single:>8} us   N=16(compute)={n16_co:>7} us   N=256(compute)={n256_co:>7} us")


if __name__ == "__main__":
    main()
