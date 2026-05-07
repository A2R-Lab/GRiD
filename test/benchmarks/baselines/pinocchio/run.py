#!/usr/bin/env python3
"""Run Pinocchio timing benchmark for one robot/base combination.

Usage:
    python test/benchmarks/baselines/pinocchio/run.py \
        --robot iiwa14 --base fixed [--output results/iiwa14_fixed_pin_<host>.json] \
        [--no-recompile] [--ee-frame iiwa_link_ee]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import shutil
import subprocess
import sys
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
) -> Path:
    """Compile timePinocchio.cpp, using content-hash cache."""
    source_hash = _hash_file(TIMING_SOURCE)
    util_dir = THIS_DIR.parent / "util"
    util_hash = _hash_bytes(
        b"".join(f.read_bytes() for f in sorted(util_dir.rglob("*.h")) if f.is_file())
    )
    runner_key = _hash_bytes(
        json.dumps({
            "source_hash": source_hash,
            "util_hash": util_hash,
            "robot": robot,
            "base": base,
            "have_cppadcg": has_cppadcg(),
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
        gxx, "-std=c++14", "-O3",
        str(TIMING_SOURCE),
        "-o", str(binary_path),
        *codegen_flag, *cflags, *libs,
    ]
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


def run_timing(binary_path: Path, urdf_path: str, base: str, ee_frame: str) -> str:
    floating_arg = "T" if base == "floating" else "F"
    cmd = [str(binary_path), urdf_path, floating_arg]
    if ee_frame:
        cmd.append(ee_frame)
    result = subprocess.run(cmd, capture_output=True, text=True, env=_runtime_env())
    if result.returncode != 0:
        raise RuntimeError(
            f"timePinocchio exited with code {result.returncode}:\n{result.stderr}"
        )
    return result.stdout + "\n" + result.stderr


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
        binary_path = compile_binary(args.robot, args.base, build_dir, args.no_recompile)
    except Exception as e:
        print(f"  [pinocchio] ERROR compiling: {e}", file=sys.stderr)
        sys.exit(1)

    print(f"  [pinocchio] running timing binary (EE frame: {ee_frame or 'none'})...")
    try:
        output = run_timing(binary_path, urdf_path, args.base, ee_frame)
    except Exception as e:
        print(f"  [pinocchio] ERROR running binary: {e}", file=sys.stderr)
        sys.exit(1)

    timings = parse_pinocchio_output(output)
    filled = fill_nulls(timings)

    meta = build_metadata(include_gpu=False, include_pinocchio=True)
    meta["robot"] = args.robot
    meta["base"] = args.base
    meta["ee_frame"] = ee_frame

    result = {"metadata": meta, "results": {args.robot: {args.base: {"pinocchio": filled}}}}
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(f"  [pinocchio] results saved: {args.output}")

    for algo, entry in sorted(filled.items()):
        if entry is None:
            print(f"    {algo}: null")
        elif "single_us" in entry:
            v = entry["single_us"]["mean"]
            print(f"    {algo}: {v:.2f}us (single)")


if __name__ == "__main__":
    main()
