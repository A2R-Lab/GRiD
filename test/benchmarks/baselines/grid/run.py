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
from test.pinocchio_equivalents.utils.project_adapter import strict_parse_robot  # noqa: E402
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
    "id": {
        "single_call":        "grid::inverse_dynamics_single_timing<float,false,true>(hd_data,d_robotModel,GRAVITY,SINGLE_CALL_ITERS_GLOBAL,dim3(1,1,1),dimms,streams)",
        "batch_with_mem":     "grid::inverse_dynamics<float,false,true>(d,m,GRAVITY,N,dim3(N,1,1),dimms,streams)",
        "batch_compute_only": "grid::inverse_dynamics_compute_only<float,false,true>(d,m,GRAVITY,N,dim3(N,1,1),dimms)",
        "batch_label": "ID",
        "gate": None,
    },
    "minv": {
        "single_call":        "grid::direct_minv_single_timing<float,true>(hd_data,d_robotModel,SINGLE_CALL_ITERS_GLOBAL,dim3(1,1,1),dimms,streams)",
        "batch_with_mem":     "grid::direct_minv<float,true>(d,m,N,dim3(N,1,1),dimms,streams)",
        "batch_compute_only": "grid::direct_minv_compute_only<float,true>(d,m,N,dim3(N,1,1),dimms)",
        "batch_label": "Minv",
        "gate": None,
    },
    "fd": {
        "single_call":        "grid::forward_dynamics_single_timing<float>(hd_data,d_robotModel,GRAVITY,SINGLE_CALL_ITERS_GLOBAL,dim3(1,1,1),dimms,streams)",
        "batch_with_mem":     "grid::forward_dynamics<float>(d,m,GRAVITY,N,dim3(N,1,1),dimms,streams)",
        "batch_compute_only": "grid::forward_dynamics_compute_only<float>(d,m,GRAVITY,N,dim3(N,1,1),dimms)",
        "batch_label": "FD",
        "gate": None,
    },
    "aba": {
        "single_call":        "grid::aba_single_timing<float>(hd_data,d_robotModel,GRAVITY,SINGLE_CALL_ITERS_GLOBAL,dim3(1,1,1),dimms,streams)",
        "batch_with_mem":     "grid::aba<float>(d,m,GRAVITY,N,dim3(N,1,1),dimms,streams)",
        "batch_compute_only": "grid::aba_compute_only<float>(d,m,GRAVITY,N,dim3(N,1,1),dimms)",
        "batch_label": "ABA",
        "gate": None,
    },
    "crba": {
        "single_call":        "grid::crba_single_timing<float>(hd_data,d_robotModel,GRAVITY,SINGLE_CALL_ITERS_GLOBAL,dim3(1,1,1),dimms,streams)",
        "batch_with_mem":     "grid::crba<float>(d,m,GRAVITY,N,dim3(N,1,1),dimms,streams)",
        "batch_compute_only": "grid::crba_compute_only<float>(d,m,GRAVITY,N,dim3(N,1,1),dimms)",
        "batch_label": "CRBA",
        "gate": None,
    },
    "id_du": {
        "single_call":        "grid::inverse_dynamics_gradient_single_timing<float,false,true>(hd_data,d_robotModel,GRAVITY,SINGLE_CALL_ITERS_GLOBAL,dim3(1,1,1),dimms,streams)",
        "batch_with_mem":     "grid::inverse_dynamics_gradient<float,false,true>(d,m,GRAVITY,N,dim3(N,1,1),dimms,streams)",
        "batch_compute_only": "grid::inverse_dynamics_gradient_compute_only<float,false,true>(d,m,GRAVITY,N,dim3(N,1,1),dimms)",
        "batch_label": "ID_DU",
        "gate": None,
    },
    "fd_du": {
        "single_call":        "grid::forward_dynamics_gradient_single_timing<float,false>(hd_data,d_robotModel,GRAVITY,SINGLE_CALL_ITERS_GLOBAL,dim3(1,1,1),dimms,streams)",
        "batch_with_mem":     "grid::forward_dynamics_gradient<float,false>(d,m,GRAVITY,N,dim3(N,1,1),dimms,streams)",
        "batch_compute_only": "grid::forward_dynamics_gradient_compute_only<float,false>(d,m,GRAVITY,N,dim3(N,1,1),dimms)",
        "batch_label": "FD_DU",
        "gate": None,
    },
    "ee_pose": {
        "single_call":        "grid::end_effector_pose_single_timing<float>(hd_data,d_robotModel,SINGLE_CALL_ITERS_GLOBAL,dim3(1,1,1),dimms,streams)",
        "batch_with_mem":     "grid::end_effector_pose<float>(d,m,N,dim3(N,1,1),dimms,streams)",
        "batch_compute_only": "grid::end_effector_pose_compute_only<float>(d,m,N,dim3(N,1,1),dimms)",
        "batch_label": "EE_POSE",
        "gate": None,
    },
    "ee_pose_gradient": {
        "single_call":        "grid::end_effector_pose_gradient_single_timing<float>(hd_data,d_robotModel,SINGLE_CALL_ITERS_GLOBAL,dim3(1,1,1),dimms,streams)",
        "batch_with_mem":     "grid::end_effector_pose_gradient<float>(d,m,N,dim3(N,1,1),dimms,streams)",
        "batch_compute_only": "grid::end_effector_pose_gradient_compute_only<float>(d,m,N,dim3(N,1,1),dimms)",
        "batch_label": "EE_POSE_GRADIENT",
        "gate": None,
    },
    "idsva_so": {
        "single_call":        "grid::idsva_so_single_timing<float>(hd_data,d_robotModel,GRAVITY,SINGLE_CALL_ITERS_GLOBAL,dim3(1,1,1),dimms,streams)",
        "batch_with_mem":     "grid::idsva_so<float>(d,m,GRAVITY,N,dim3(N,1,1),dimms,streams)",
        "batch_compute_only": "grid::idsva_so_compute_only<float>(d,m,GRAVITY,N,dim3(N,1,1),dimms)",
        "batch_label": "IDSVA_SO",
        "gate": "GRID_HAS_IDSVA_SO",
    },
    "idsva_so_body_frame": {
        "single_call":        "grid::idsva_so_body_frame_host_single_timing<float>(hd_data,d_robotModel,GRAVITY,SINGLE_CALL_ITERS_GLOBAL,dim3(1,1,1),dimms,streams)",
        "batch_with_mem":     "grid::idsva_so_body_frame_host<float>(d,m,GRAVITY,N,dim3(N,1,1),dimms,streams)",
        "batch_compute_only": "grid::idsva_so_body_frame_host_compute_only<float>(d,m,GRAVITY,N,dim3(N,1,1),dimms)",
        "batch_label": "IDSVA_SO_BODY_FRAME",
        "gate": "GRID_HAS_IDSVA_SO_BODY_FRAME",
    },
    "idsva_so_world_frame": {
        "single_call":        "grid::idsva_so_world_frame_host_single_timing<float>(hd_data,d_robotModel,GRAVITY,SINGLE_CALL_ITERS_GLOBAL,dim3(1,1,1),dimms,streams)",
        "batch_with_mem":     "grid::idsva_so_world_frame_host<float>(d,m,GRAVITY,N,dim3(N,1,1),dimms,streams)",
        "batch_compute_only": "grid::idsva_so_world_frame_host_compute_only<float>(d,m,GRAVITY,N,dim3(N,1,1),dimms)",
        "batch_label": "IDSVA_SO_WORLD_FRAME",
        "gate": "GRID_HAS_IDSVA_SO_WORLD_FRAME",
    },
    "fdsva_so": {
        "single_call":        "grid::fdsva_so_single_timing<float>(hd_data,d_robotModel,GRAVITY,SINGLE_CALL_ITERS_GLOBAL,dim3(1,1,1),dimms,streams)",
        "batch_with_mem":     "grid::fdsva_so<float>(d,m,GRAVITY,N,dim3(N,1,1),dimms,streams)",
        "batch_compute_only": "grid::fdsva_so_compute_only<float>(d,m,GRAVITY,N,dim3(N,1,1),dimms)",
        "batch_label": "FDSVA_SO",
        "gate": "GRID_HAS_FDSVA_SO",
        "shared_mem_skip": "FDSVA_SO_DYNAMIC_SHARED_MEM_BYTES",
    },
}


def _algo_keys_in_registry_order() -> list[str]:
    """Return algo keys in ALGO_REGISTRY order, filtered to those in PER_ALGO_SPECS."""
    keys: list[str] = []
    for entry in ALGO_REGISTRY:
        if entry.key in PER_ALGO_SPECS:
            keys.append(entry.key)
        else:
            raise RuntimeError(
                f"Algo {entry.key!r} is in ALGO_REGISTRY but missing from "
                f"PER_ALGO_SPECS in run.py. Add a spec row."
            )
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
        print(
            f"  [grid] per-algo TU compile+link finished in "
            f"{time.perf_counter() - t_total:.1f}s "
            f"(single rdc={single_rdc}, batch rdc={batch_rdc}, workers={compile_workers})"
        )
        return single_binary, batch_binary

    # ----- Monolithic fallback (pre-P6-7b layout) -----
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
    print(
        f"  [grid] monolithic compile finished in "
        f"{time.perf_counter() - t_total:.1f}s "
        f"(single rdc={single_rdc}, batch rdc={batch_rdc})"
    )
    return single_binary, batch_binary


# ---------------------------------------------------------------------------
# Run and parse
# ---------------------------------------------------------------------------
def run_timing(binaries: tuple[Path, Path], base: str) -> str:
    """Run the single + batch binaries in sequence; return concatenated stdout
    so parse_grid_output picks up `Single Call X us` lines (from the single
    binary) AND `[N:K]: X` lines (from the batch binary)."""
    single_binary, batch_binary = binaries
    floating_arg = "T" if base == "floating" else "F"
    outputs = []
    for label, binary in (("single", single_binary), ("batch", batch_binary)):
        result = subprocess.run(
            [str(binary), floating_arg],
            capture_output=True, text=True,
        )
        if result.returncode != 0:
            raise RuntimeError(
                f"timeGRiD_{label} exited with code {result.returncode}:\n{result.stderr}"
            )
        outputs.append(result.stdout)
        outputs.append(result.stderr)
    return "\n".join(outputs)


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
    args = parser.parse_args()

    ee_frame = args.ee_frame or DEFAULT_EE_FRAMES.get(args.robot, "")
    build_dir = REPO_ROOT / "test" / "benchmarks" / "results"
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
        )
    except Exception as e:
        print(f"  [grid] ERROR compiling binaries: {e}", file=sys.stderr)
        sys.exit(1)
    print(f"  [grid] compile wall time: {time.perf_counter() - t_compile:.1f}s")

    print(f"  [grid] running timing binaries (single + batch)...")
    try:
        output = run_timing(binaries, args.base)
    except Exception as e:
        print(f"  [grid] ERROR running binary: {e}", file=sys.stderr)
        sys.exit(1)

    timings = parse_grid_output(output)
    filled = fill_nulls(timings)

    meta = build_metadata(include_gpu=True)
    meta["robot"] = args.robot
    meta["base"] = args.base
    meta["ee_frame"] = ee_frame
    meta["cuda_arch"] = arch
    meta["grid_linalg_backend"] = "glass"

    result = {"metadata": meta, "results": {args.robot: {args.base: {"grid": filled}}}}
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(f"  [grid] results saved: {args.output}")

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
