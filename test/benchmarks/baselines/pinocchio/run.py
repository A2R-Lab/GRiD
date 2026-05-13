#!/usr/bin/env python3
"""Run Pinocchio timing benchmark for one robot/base combination.

Usage:
    python test/benchmarks/baselines/pinocchio/run.py \
        --robot iiwa14 --base fixed [--output results/iiwa14_fixed_pin_<host>.json] \
        [--no-recompile] [--ee-frame iiwa_link_ee]
"""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import os
import platform
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[4]
THIS_DIR  = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO_ROOT))

from test.benchmarks.timing_parser import (  # noqa: E402
    parse_pinocchio_output, fill_nulls, build_metadata,
)

# ---------------------------------------------------------------------------
# Canonical EE frame names per robot
# ---------------------------------------------------------------------------
DEFAULT_EE_FRAMES: dict[str, str] = {
    "iiwa14": "iiwa_link_ee",
    "go2":    "FR_foot",
    "g1":     "right_rubber_hand",
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
# CPU frequency locking (Linux only, optional)
# ---------------------------------------------------------------------------
def try_lock_cpu_freq() -> bool:
    """Attempt to lock CPU to performance governor.  Returns True if successful."""
    if platform.system() != "Linux":
        print("  [pinocchio] CPU freq locking not supported on this OS — timing may be noisier")
        return False
    cpupower = shutil.which("cpupower")
    if cpupower is None:
        print("  [pinocchio] cpupower not found — timing may be noisier")
        return False
    # Try with sudo (passwordless via /etc/sudoers.d/cpupower — see README.md)
    result = subprocess.run(
        ["sudo", "-n", cpupower, "frequency-set", "-g", "performance"],
        capture_output=True, text=True,
    )
    if result.returncode == 0:
        print("  [pinocchio] CPU locked to performance governor")
        return True
    # Also try the repo-bundled setCPU.sh
    setcpu = REPO_ROOT / "test" / "benchmarks" / "setCPU.sh"
    if setcpu.exists():
        result = subprocess.run(["sudo", "-n", "bash", str(setcpu)], capture_output=True, text=True)
        if result.returncode == 0:
            print("  [pinocchio] CPU locked via setCPU.sh")
            return True
    print("  [pinocchio] Could not lock CPU freq (sudo required) — timing may be noisier")
    return False


# ---------------------------------------------------------------------------
# Pinocchio binary resolution (find pkg-config prefix)
# ---------------------------------------------------------------------------
def has_cppadcg() -> bool:
    """Return True if CppADCodeGen headers are findable."""
    result = subprocess.run(
        ["pkg-config", "--exists", "cppadcg"],
        capture_output=True,
    )
    if result.returncode == 0:
        return True
    # Check cmeel prefix
    venv = Path(sys.prefix)
    cmeel = venv / "lib" / f"python{sys.version_info.major}.{sys.version_info.minor}" / \
            "site-packages" / "cmeel.prefix"
    return (cmeel / "include" / "cppad" / "cg.hpp").exists()


def pinocchio_cflags() -> list[str]:
    result = subprocess.run(
        ["pkg-config", "--cflags", "pinocchio", "cppadcg"],
        capture_output=True, text=True,
    )
    if result.returncode != 0:
        # Fall back to cmeel prefix + system Eigen
        venv = Path(sys.prefix)
        cmeel = venv / "lib" / f"python{sys.version_info.major}.{sys.version_info.minor}" / \
                "site-packages" / "cmeel.prefix"
        includes = []
        if cmeel.exists():
            includes = [f"-I{cmeel}/include"]
        # Eigen may live in cmeel or system; try both
        eigen_pkg = subprocess.run(["pkg-config", "--cflags", "eigen3"],
                                   capture_output=True, text=True)
        if eigen_pkg.returncode == 0:
            includes += eigen_pkg.stdout.strip().split()
        return includes
    return result.stdout.strip().split()


def pinocchio_libs() -> list[str]:
    result = subprocess.run(
        ["pkg-config", "--libs", "pinocchio", "cppadcg"],
        capture_output=True, text=True,
    )
    if result.returncode != 0:
        venv = Path(sys.prefix)
        cmeel = venv / "lib" / f"python{sys.version_info.major}.{sys.version_info.minor}" / \
                "site-packages" / "cmeel.prefix"
        libs = []
        if cmeel.exists():
            libs = [f"-L{cmeel}/lib", "-lpinocchio_default", "-lpinocchio_parsers"]
            # CppAD ships libcppad_lib.so (exports CppAD::local::temp_file etc.)
            # which CppADCodeGen uses at link time. cmeel-cppad puts the library
            # in cmeel.prefix/lib alongside libpinocchio_*, so just add it when
            # cppadcg is enabled.
            if (cmeel / "lib" / "libcppad_lib.so").exists():
                libs.append("-lcppad_lib")
                # Add rpath so the binary finds libcppad_lib.so at runtime
                # without LD_LIBRARY_PATH gymnastics.
                libs.append(f"-Wl,-rpath,{cmeel}/lib")
        return libs
    return result.stdout.strip().split()


# ---------------------------------------------------------------------------
# Binary compilation with caching
# ---------------------------------------------------------------------------
TIMING_SOURCE = THIS_DIR / "timePinocchio.cpp"
CACHE_ROOT = REPO_ROOT / ".pytest_cache" / "grid_cuda"


def _hash_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _hash_file(path: Path) -> str:
    return _hash_bytes(path.read_bytes())


def compile_binary(
    robot: str,
    base: str,
    build_dir: Path,
    no_recompile: bool = False,
    single_call_iters: int | None = None,
    batch_iters: int | None = None,
) -> Path:
    """Compile timePinocchio.cpp, using content-hash cache."""
    source_hash = _hash_file(TIMING_SOURCE)
    util_dir = THIS_DIR.parent / "util"
    util_hash = _hash_bytes(
        b"".join(f.read_bytes() for f in sorted(util_dir.rglob("*.h")) if f.is_file())
    )
    iter_defs: list[str] = []
    if single_call_iters is not None:
        iter_defs.append(f"-DSINGLE_CALL_ITERS_GLOBAL={int(single_call_iters)}")
    if batch_iters is not None:
        iter_defs.append(f"-DTEST_ITERS_GLOBAL={int(batch_iters)}")
    runner_key = _hash_bytes(
        json.dumps({
            "source_hash": source_hash,
            "util_hash": util_hash,
            "robot": robot,
            "base": base,
            "have_cppadcg": has_cppadcg(),
            "iter_defs": iter_defs,
        }, sort_keys=True).encode()
    )[:24]

    binary_path = build_dir / "timePinocchio.exe"
    cached_binary = CACHE_ROOT / "pinocchio_benchmarks" / runner_key / "timePinocchio.exe"

    if not no_recompile and cached_binary.exists():
        shutil.copyfile(cached_binary, binary_path)
        os.chmod(binary_path, 0o755)
        print(f"  [pinocchio] binary cache hit (key={runner_key[:12]})")
        return binary_path

    print(f"  [pinocchio] compiling timePinocchio.cpp (cache key={runner_key[:12]})...")
    gxx = shutil.which("g++")
    if gxx is None:
        raise RuntimeError("g++ not found — install build-essential")

    cflags = pinocchio_cflags()
    libs   = pinocchio_libs()

    codegen_flag = ["-DHAVE_CPPADCG"] if has_cppadcg() else []
    if not codegen_flag:
        print("  [pinocchio] cppadcg not found — codegen algorithms will be null")

    cmd = [
        gxx, "-std=c++14", "-O3", "-DNDEBUG",
        str(TIMING_SOURCE),
        "-o", str(binary_path),
        *iter_defs, *codegen_flag, *cflags, *libs,
    ]
    # -DNDEBUG disables Pinocchio's debug isUnitary check on the rotation matrix.
    # The harness stores q as float32, normalizes the quaternion segment in float32
    # precision (~1e-7), then Pinocchio casts to double and checks unitarity at
    # double precision (~1e-12) — which fails for float32-normalized quaternions on
    # floating-base robots (go2/g1). Release-mode benchmarks should disable asserts.
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(
            f"g++ compilation failed:\n{result.stdout}\n{result.stderr}"
        )

    cached_binary.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(binary_path, cached_binary)
    os.chmod(cached_binary, 0o755)
    print(f"  [pinocchio] compiled successfully")
    return binary_path


# ---------------------------------------------------------------------------
# Run and parse
# ---------------------------------------------------------------------------
def _runtime_env() -> dict[str, str]:
    """Build environment with cmeel lib path prepended to LD_LIBRARY_PATH."""
    env = os.environ.copy()
    venv = Path(sys.prefix)
    cmeel_lib = (
        venv / "lib"
        / f"python{sys.version_info.major}.{sys.version_info.minor}"
        / "site-packages" / "cmeel.prefix" / "lib"
    )
    if cmeel_lib.exists():
        existing = env.get("LD_LIBRARY_PATH", "")
        env["LD_LIBRARY_PATH"] = f"{cmeel_lib}:{existing}" if existing else str(cmeel_lib)
    return env


# Pinocchio actively-timed algos. Skip "fdsva_so" — Pinocchio has no equivalent;
# fill_nulls() will keep it as None in the final JSON. EE algos still need the
# binary built with a valid frame_name; they run as fast as the others. The
# main expense (cppadcg JIT) is gated inside timePinocchio.cpp by --algo.
PINOCCHIO_ALGOS: tuple[str, ...] = (
    "id", "minv", "fd", "aba", "crba", "id_du", "fd_du",
    "ee_pose", "ee_pose_gradient", "idsva_so",
)

# Per-algo subprocess wall-clock timeout. g1 codegen for any one algo
# (e.g., fd_du which needs rnea + minv + rnea_d) can take 20+ minutes;
# headroom keeps us under a single 25-min ceiling per subprocess.
PER_ALGO_TIMEOUT_S = 1500


def run_timing_one_algo(binary_path: Path, urdf_path: str, base: str,
                        ee_frame: str, algo: str) -> tuple[str, str | None]:
    """Run timePinocchio.exe for a single algo. Returns (stdout+stderr, error_msg or None).

    Each invocation runs in its own isolated tmpdir so the cppadcg JIT (which
    writes <algo>_codegen.so + cppadcg_tmp/ to the CWD with a hardcoded name)
    doesn't race when we fan out parallel subprocesses.
    """
    floating_arg = "T" if base == "floating" else "F"
    # The CLI is positional: <urdf> <T/F> <frame_name> <algo>.
    # frame_name must be present (even if empty) for algo to be parsed at argv[4].
    cmd = [str(binary_path), urdf_path, floating_arg, ee_frame or "", algo]
    tmpdir = tempfile.mkdtemp(prefix=f"pin_cg_{algo}_")
    try:
        try:
            result = subprocess.run(
                cmd, capture_output=True, text=True, env=_runtime_env(),
                timeout=PER_ALGO_TIMEOUT_S, cwd=tmpdir,
            )
        except subprocess.TimeoutExpired:
            return ("", f"timeout after {PER_ALGO_TIMEOUT_S}s")
        if result.returncode != 0:
            err = result.stderr[-500:] if result.stderr else f"exit {result.returncode}"
            return (result.stdout + "\n" + result.stderr, err)
        return (result.stdout + "\n" + result.stderr, None)
    finally:
        # Clean up the per-algo cppadcg artifacts.
        shutil.rmtree(tmpdir, ignore_errors=True)


def run_timings_parallel(
    binary_path: Path, urdf_path: str, base: str, ee_frame: str,
    algos: tuple[str, ...] = PINOCCHIO_ALGOS, max_workers: int | None = None,
) -> dict[str, str]:
    """Fan out one subprocess per algo and gather per-algo stdout.

    Returns dict {algo: combined_stdout_stderr}. Algos that timed out or
    errored get an error placeholder in the value (still parsed as null
    downstream).
    """
    if max_workers is None:
        # timePinocchio.cpp uses CPU_THREADS_GLOBAL=8 internal worker threads
        # per subprocess to parallelize the batch loop. Spawning N subprocesses
        # × 8 threads each on M cores must satisfy N * 8 <= M to avoid CPU
        # oversubscription that wrecks batch timings. Conservative cap:
        # cpu_count // 8 subprocesses, minimum 1, maximum len(algos).
        # Override via PIN_MAX_WORKERS env var to force a different value.
        env_override = os.environ.get("PIN_MAX_WORKERS")
        if env_override:
            max_workers = max(1, int(env_override))
        else:
            cores = os.cpu_count() or 4
            max_workers = max(1, cores // 8)

    print(f"  [pinocchio] fanning out {len(algos)} per-algo subprocesses "
          f"(max_workers={max_workers}, timeout={PER_ALGO_TIMEOUT_S}s each)...")
    outputs: dict[str, str] = {}
    started = time.time()
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as ex:
        future_to_algo = {
            ex.submit(run_timing_one_algo, binary_path, urdf_path, base, ee_frame, algo): algo
            for algo in algos
        }
        for fut in concurrent.futures.as_completed(future_to_algo):
            algo = future_to_algo[fut]
            output, err = fut.result()
            elapsed = time.time() - started
            if err is not None:
                print(f"  [pinocchio] [{elapsed:7.1f}s] {algo}: FAILED — {err}")
            else:
                print(f"  [pinocchio] [{elapsed:7.1f}s] {algo}: ok")
            outputs[algo] = output
    return outputs


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(description="Run Pinocchio benchmark for one robot/base")
    parser.add_argument("--robot", required=True, choices=list(ROBOT_DESCRIPTION_MODULE))
    parser.add_argument("--base", required=True, choices=["fixed", "floating"])
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--no-recompile", action="store_true")
    parser.add_argument("--ee-frame", default=None,
                        help="Pinocchio frame name for EE timing (default: per-robot canonical)")
    parser.add_argument("--no-cpu-lock", action="store_true",
                        help="Skip CPU frequency locking even if available")
    parser.add_argument("--single-call-iters", type=int, default=None,
                        help="Override SINGLE_CALL_ITERS_GLOBAL (default 10000).")
    parser.add_argument("--batch-iters", type=int, default=None,
                        help="Override TEST_ITERS_GLOBAL (default 100).")
    args = parser.parse_args()

    ee_frame = args.ee_frame or DEFAULT_EE_FRAMES.get(args.robot, "")
    build_dir = REPO_ROOT / "test" / "benchmarks" / "results"
    build_dir.mkdir(parents=True, exist_ok=True)

    if args.output is None:
        host = platform.node().replace(" ", "_")
        args.output = build_dir / f"{args.robot}_{args.base}_pinocchio_{host}.json"

    if not args.no_cpu_lock:
        try_lock_cpu_freq()

    urdf_path = get_urdf_path(args.robot)
    print(f"[pinocchio] {args.robot} {args.base} — URDF: {urdf_path}")

    try:
        binary_path = compile_binary(args.robot, args.base, build_dir, args.no_recompile,
                                     single_call_iters=args.single_call_iters,
                                     batch_iters=args.batch_iters)
    except Exception as e:
        print(f"  [pinocchio] ERROR compiling: {e}", file=sys.stderr)
        sys.exit(1)

    print(f"  [pinocchio] running timing binary (EE frame: {ee_frame or 'none'})...")
    try:
        per_algo_outputs = run_timings_parallel(
            binary_path, urdf_path, args.base, ee_frame,
        )
    except Exception as e:
        print(f"  [pinocchio] ERROR running binaries: {e}", file=sys.stderr)
        sys.exit(1)

    # Merge per-algo subprocess outputs. Each subprocess only emits timings for
    # its own algo (others are gated out in timePinocchio.cpp); parse each
    # subprocess's stdout independently and take that algo's entry from the
    # parsed dict. Algos that errored or timed out stay null via fill_nulls().
    timings: dict[str, object] = {}
    for algo, output in per_algo_outputs.items():
        if not output:
            continue
        parsed = parse_pinocchio_output(output)
        if algo in parsed and parsed[algo] is not None:
            timings[algo] = parsed[algo]
    filled = fill_nulls(timings)

    meta = build_metadata(include_gpu=False, include_pinocchio=True)
    meta["robot"] = args.robot
    meta["base"] = args.base
    meta["ee_frame"] = ee_frame

    result = {"metadata": meta, "results": {args.robot: {args.base: {"pinocchio": filled}}}}
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(f"  [pinocchio] results saved: {args.output}")

    def _us(entry, key):
        v = (entry.get(key) or {}).get("median") or (entry.get(key) or {}).get("mean")
        return f"{v:.2f}" if v is not None else "—"
    for algo, entry in sorted(filled.items()):
        if entry is None:
            print(f"    {algo}: null")
            continue
        single  = _us(entry, "single_us")
        n16     = _us(entry, "batch_16_with_mem_us")
        n256    = _us(entry, "batch_256_with_mem_us")
        print(f"    {algo:18s} single={single:>8} us   N=16(w/mem)={n16:>7} us   N=256(w/mem)={n256:>7} us")


if __name__ == "__main__":
    main()
