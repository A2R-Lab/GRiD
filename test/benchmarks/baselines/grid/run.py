#!/usr/bin/env python3
"""Run GRiD timing benchmark for one robot/base combination.

Usage:
    python test/benchmarks/baselines/grid/run.py \
        --robot iiwa14 --base fixed [--output results/iiwa14_fixed_rtx5090.json] \
        [--no-recompile] [--ee-frame iiwa_link_ee] \
        [--linalg-backend glass|glass-nvidia]
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import io
import json
import os
import shutil
import subprocess
import sys
from datetime import date
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(REPO_ROOT))

from GRiDCodeGenerator import GRiDCodeGenerator  # noqa: E402
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


def resolve_mathdx_root(user_root: str | None = None) -> Path | None:
    """Find a MathDx installation that contains cuBLASDx headers."""
    candidates: list[Path] = []
    if user_root:
        candidates.append(Path(user_root))
    env_root = os.environ.get("MATHDX_ROOT")
    if env_root:
        candidates.append(Path(env_root))
    candidates.append(Path("/opt/nvidia/mathdx/25.12"))

    seen: set[Path] = set()
    for root in candidates:
        root = root.expanduser()
        if root in seen:
            continue
        seen.add(root)
        if (root / "include" / "cublasdx.hpp").exists():
            return root
    return None


def cublasdx_sm_from_arch(arch: str) -> str:
    """Convert GRID_CUDA_ARCH-style values to the MathDx SM macro convention."""
    return f"{arch}0"


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
    # GRID_BENCH_NVIDIA_MIN_DIM controls codegen's per-call backend choice via
    # linalg_smem_for(); changing it must bust the header cache.
    nvidia_min_dim_env = os.environ.get("GRID_BENCH_NVIDIA_MIN_DIM", "16")
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
            "nvidia_min_dim": nvidia_min_dim_env,
            "no_licm_barrier": no_licm_barrier_env,
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
TIMING_SOURCE = THIS_DIR / "timeGRiD.cu"


def compile_binary(
    header_path: Path,
    arch: str,
    build_dir: Path,
    no_recompile: bool = False,
    linalg_backend: str = "glass",
    mathdx_root: str | None = None,
    with_cusolverdx: bool = False,
    no_rdc: bool = False,
    single_call_iters: int | None = None,
    batch_iters: int | None = None,
) -> Path:
    """Compile timeGRiD.cu against the generated header, using content-hash cache."""
    source_hash = _hash_file(TIMING_SOURCE)
    header_hash = _hash_file(header_path)

    cxx_standard = "-std=c++11"
    linalg_flags: list[str] = []
    resolved_mathdx_root: Path | None = None
    cublasdx_sm: str | None = None

    # Override defaults in test/benchmarks/baselines/util/experiment_helpers.h
    # (SINGLE_CALL_ITERS_GLOBAL=10000, TEST_ITERS_GLOBAL=100) by re-defining at
    # the compile line. Bumping iter counts is the simplest way to reduce
    # measurement noise on fast kernels.
    if single_call_iters is not None:
        linalg_flags.append(f"-DSINGLE_CALL_ITERS_GLOBAL={int(single_call_iters)}")
    if batch_iters is not None:
        linalg_flags.append(f"-DTEST_ITERS_GLOBAL={int(batch_iters)}")

    # -rdc=true (relocatable device code) is required so nvcc treats the
    # `__noinline__` device function `grid_licm_barrier` as opaque across the
    # call boundary. Without separate-compilation semantics, nvcc inlines it
    # despite the annotation and hoists the surrounding _single_timing inner
    # work out of the rep loop (the LICM elision we hit on go2/g1 floating).
    # -dlto is added on top only for the cuSOLVERDx link, since that path
    # needs the device-link-time optimizer to merge the precompiled library.
    if linalg_backend == "glass":
        linalg_flags.append("-DGRID_CUDA_LINALG_BACKEND=GRID_LINALG_GLASS")
        if not no_rdc:
            linalg_flags.append("-rdc=true")
    elif linalg_backend == "glass-nvidia":
        resolved_mathdx_root = resolve_mathdx_root(mathdx_root)
        if resolved_mathdx_root is None:
            raise RuntimeError(
                "glass-nvidia backend requested, but cublasdx.hpp was not found. "
                "Set --mathdx-root or MATHDX_ROOT to a MathDx installation."
            )
        cxx_standard = "-std=c++17"
        cublasdx_sm = cublasdx_sm_from_arch(arch)
        linalg_flags.extend([
            "-DGRID_CUDA_LINALG_BACKEND=GRID_LINALG_GLASS_NVIDIA",
            f"-DGRID_CUBLASDX_SM={cublasdx_sm}",
            f"-I{resolved_mathdx_root / 'include'}",
            f"-I{resolved_mathdx_root / 'external' / 'cutlass' / 'include'}",
            # cuBLASDx L2/L3 require relaxed constexpr (see GLASS README).
            "--expt-relaxed-constexpr",
        ])
        if with_cusolverdx and no_rdc:
            raise RuntimeError(
                "--with-cusolverdx requires -rdc=true; cannot combine with --no-rdc."
            )
        if not no_rdc:
            linalg_flags.append("-rdc=true")
        if with_cusolverdx:
            # cuSOLVERDx ships a precompiled device library; needs -dlto
            # on top of -rdc=true (added above) and links against
            # cusolverdx + cublas + cusolver + cudart.
            cusolverdx_lib_dir = resolved_mathdx_root / "lib"
            linalg_flags.extend([
                "-DGRID_CUDA_USE_GLASS_NVIDIA_LAPACK=1",
                "-dlto",
                f"-L{cusolverdx_lib_dir}",
                "-lcusolverdx",
                "-lcublas",
                "-lcusolver",
                "-lcudart",
            ])
    else:
        raise ValueError(f"Unknown linear algebra backend '{linalg_backend}'")

    runner_key = _hash_bytes(
        json.dumps({
            "source_hash": source_hash,
            "header_hash": header_hash,
            "cuda_arch": arch,
            "cxx_standard": cxx_standard,
            "linalg_backend": linalg_backend,
            "mathdx_root": str(resolved_mathdx_root) if resolved_mathdx_root else None,
            "cublasdx_sm": cublasdx_sm,
            "linalg_flags": linalg_flags,
            "with_cusolverdx": with_cusolverdx,
            "no_rdc": no_rdc,
        }, sort_keys=True).encode()
    )[:24]

    binary_path = build_dir / "timeGRiD.exe"
    cached_binary = CACHE_ROOT / "grid_benchmarks" / runner_key / "timeGRiD.exe"

    if no_recompile or cached_binary.exists():
        if cached_binary.exists():
            shutil.copyfile(cached_binary, binary_path)
            os.chmod(binary_path, 0o755)
            print(f"  [grid] binary cache hit (key={runner_key[:12]})")
            return binary_path

    backend_note = linalg_backend
    if cublasdx_sm is not None:
        backend_note += f", GRID_CUBLASDX_SM={cublasdx_sm}"
    print(
        f"  [grid] compiling timeGRiD.cu "
        f"(arch=sm_{arch}, linalg={backend_note}, cache key={runner_key[:12]})..."
    )
    nvcc = shutil.which("nvcc")
    if nvcc is None:
        raise RuntimeError("nvcc not found — install CUDA Toolkit to compile timeGRiD")

    cmd = [
        nvcc, cxx_standard, "-o", str(binary_path), str(TIMING_SOURCE),
        f"-DGRID_HEADER_FILE=\"{header_path}\"",
        "-gencode", f"arch=compute_{arch},code=sm_{arch}",
        "-O3", "-ftz=true", "-prec-div=false", "-prec-sqrt=false",
    ]
    if linalg_backend == "glass-nvidia":
        cmd.extend(["-gencode", f"arch=compute_{arch},code=compute_{arch}"])
    cmd.extend(linalg_flags)
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(
            f"nvcc compilation failed:\n{result.stdout}\n{result.stderr}"
        )

    cached_binary.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(binary_path, cached_binary)
    os.chmod(cached_binary, 0o755)
    print(f"  [grid] compiled successfully")
    return binary_path


# ---------------------------------------------------------------------------
# Run and parse
# ---------------------------------------------------------------------------
def run_timing(binary_path: Path, base: str) -> str:
    floating_arg = "T" if base == "floating" else "F"
    result = subprocess.run(
        [str(binary_path), floating_arg],
        capture_output=True, text=True,
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"timeGRiD exited with code {result.returncode}:\n{result.stderr}"
        )
    return result.stdout + "\n" + result.stderr


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
    parser.add_argument("--linalg-backend",
                        choices=["glass", "glass-nvidia"],
                        default=os.environ.get("GRID_BENCH_LINALG_BACKEND", "glass"),
                        help="Linear algebra backend for generated GRiD helpers")
    parser.add_argument("--mathdx-root", default=os.environ.get("MATHDX_ROOT"),
                        help="MathDx root used when --linalg-backend=glass-nvidia")
    parser.add_argument("--with-cusolverdx", action="store_true",
                        default=os.environ.get("GRID_BENCH_WITH_CUSOLVERDX", "0") == "1",
                        help="Enable cuSOLVERDx LAPACK wrappers (chol/trsm/posv). Adds "
                             "-rdc=true -dlto -lcusolverdx -lcublas -lcusolver -lcudart to "
                             "the link line. Only takes effect with --linalg-backend=glass-nvidia.")
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
    parser.add_argument("--single-call-iters", type=int, default=None,
                        help="Override SINGLE_CALL_ITERS_GLOBAL (default 10000). Inner-kernel "
                             "rep count for single-call timings; bump for more stable medians "
                             "on noisy machines.")
    parser.add_argument("--batch-iters", type=int, default=None,
                        help="Override TEST_ITERS_GLOBAL (default 100). Outer rep count for "
                             "batch timings at each N; bump for more stable medians.")
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
    print(f"  [grid] linear algebra backend: {args.linalg_backend}")

    try:
        header_path = generate_header(urdf_path, args.robot, args.base, ee_frame, build_dir, args.no_recompile)
    except Exception as e:
        print(f"  [grid] ERROR generating header: {e}", file=sys.stderr)
        sys.exit(1)

    try:
        binary_path = compile_binary(
            header_path,
            arch,
            build_dir,
            args.no_recompile,
            linalg_backend=args.linalg_backend,
            mathdx_root=args.mathdx_root,
            with_cusolverdx=args.with_cusolverdx and args.linalg_backend == "glass-nvidia",
            no_rdc=args.no_rdc,
            single_call_iters=args.single_call_iters,
            batch_iters=args.batch_iters,
        )
    except Exception as e:
        print(f"  [grid] ERROR compiling binary: {e}", file=sys.stderr)
        sys.exit(1)

    print(f"  [grid] running timing binary...")
    try:
        output = run_timing(binary_path, args.base)
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
    meta["grid_linalg_backend"] = args.linalg_backend
    if args.linalg_backend == "glass-nvidia":
        resolved_mathdx_root = resolve_mathdx_root(args.mathdx_root)
        meta["mathdx_root"] = str(resolved_mathdx_root) if resolved_mathdx_root else None
        meta["grid_cublasdx_sm"] = cublasdx_sm_from_arch(arch)

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
