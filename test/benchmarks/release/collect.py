#!/usr/bin/env python3
"""Plan or execute isolated release jobs; defaults to a no-GPU dry run.

python -m test.benchmarks.release.collect --stage core
python -m test.benchmarks.release.collect --stage wrappers --smoke --execute --output /tmp/grid-smoke
"""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import importlib.metadata
import json
import os
from pathlib import Path
import signal
import subprocess
import sys

from .protocol import BATCHES, ROBOTS, ROOT, jobs, digest, write_json, source_fingerprints
from .protocol import ACCURACY_POLICIES, ACCURACY_POLICY_VERSION, FD_WARNING_MAX_RELATIVE_L2, WARM_SECONDS

JAX_CACHE = ROOT / "test/benchmarks/results/release-jax-cache"


def worker_environment():
    # Executable cache: local, owner-controlled, never supplied by a third party.
    JAX_CACHE.mkdir(parents=True, exist_ok=True, mode=0o700)
    stat = JAX_CACHE.stat()
    if stat.st_uid != os.getuid() or stat.st_mode & 0o022:
        raise PermissionError(f"JAX cache must be owner-controlled: {JAX_CACHE}")
    return {**os.environ, "OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1", "MKL_NUM_THREADS": "1",
            "JAX_DEFAULT_MATMUL_PRECISION": "highest", "JAX_ENABLE_X64": "false",
            "JAX_COMPILATION_CACHE_DIR": str(JAX_CACHE),
            "JAX_PERSISTENT_CACHE_MIN_COMPILE_TIME_SECS": "0",
            "JAX_PERSISTENT_CACHE_MIN_ENTRY_SIZE_BYTES": "-1",
            "XLA_PYTHON_CLIENT_MEM_FRACTION": os.environ.get("XLA_PYTHON_CLIENT_MEM_FRACTION", "0.35")}


def command_output(cmd):
    try:
        p = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True, timeout=30)
    except (OSError, subprocess.TimeoutExpired) as error:
        return f"unavailable: {error}"
    return p.stdout.strip() if p.returncode == 0 else p.stderr.strip()


def cpu_power():
    """CPU power-management state the timing ran under: governor, energy
    preference, frequency limits and the collector's CPU affinity. Part of the
    cross-capture contract — timings taken under different settings are never
    combined silently (the Python-driven paths are sensitive to it)."""
    base = "/sys/devices/system/cpu/cpu0/cpufreq/"
    def read(name):
        try:
            return open(base + name).read().strip()
        except OSError:
            return None
    return {"governor": read("scaling_governor"), "driver": read("scaling_driver"),
            "energy_performance_preference": read("energy_performance_preference"),
            "scaling_min_khz": read("scaling_min_freq"), "scaling_max_khz": read("scaling_max_freq"),
            "affinity": sorted(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else None}


def provenance():
    packages = {}
    for name in ("numpy", "jax", "torch", "mujoco", "mujoco-mjx", "mujoco-warp", "pin", "bard", "frax"):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    return {"captured_utc": datetime.now(timezone.utc).isoformat(),
        "commit": command_output(["git", "rev-parse", "HEAD"]),
        "worktree_status": command_output(["git", "status", "--short"]),
        "diff_sha256": __import__("hashlib").sha256(command_output(["git", "diff", "HEAD"]).encode()).hexdigest(),
        "submodules": command_output(["git", "submodule", "status", "--recursive"]),
        "collector_sources": source_fingerprints(),
        "packages": packages, "python": sys.version,
        "compiler": command_output(["g++", "--version"]),
        "gpu": command_output(["nvidia-smi", "--query-gpu=name,driver_version,memory.total,clocks.current.sm,power.limit", "--format=csv,noheader"]),
        "cpu": command_output(["lscpu"]), "cpu_power": cpu_power()}


def run_job(cmd, log, timeout):
    """Kill the entire compiler/worker process group on timeout, not just Python."""
    with log.open("w") as stream:
        env = worker_environment()
        proc = subprocess.Popen(cmd, cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT,
                                start_new_session=True, env=env)
        try:
            return proc.wait(timeout=timeout), False
        except (subprocess.TimeoutExpired, KeyboardInterrupt) as error:
            try:
                os.killpg(proc.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
            try:
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                os.killpg(proc.pid, signal.SIGKILL)
                proc.wait()
            if isinstance(error, KeyboardInterrupt):
                raise
            return proc.returncode, True


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--stage", choices=("core", "wrappers", "table"), default="core")
    ap.add_argument("--robots", nargs="+", choices=list(ROBOTS), default=list(ROBOTS))
    ap.add_argument("--backends", nargs="+")
    ap.add_argument("--operations", nargs="+")
    ap.add_argument("--batches", nargs="+", type=int)
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--iterations", type=int, default=30)
    ap.add_argument("--warmups", type=int, default=5)
    ap.add_argument("--warm-seconds", type=float, default=WARM_SECONDS, help="Sustain calls at least this long before sampling (device reaches its steady clock); applies to every backend")
    ap.add_argument("--cpu-threads", type=int, default=min(8, os.cpu_count() or 1), help="Pinocchio CPU worker ceiling; recorded, never changes host clock settings")
    ap.add_argument("--timeout", type=int, default=1800, help="Per-job wall-time ceiling, including build")
    mode = ap.add_mutually_exclusive_group()
    mode.add_argument("--smoke", action="store_true", help="Defaults to B=16; 1 repeat, 2 warmups, 2 samples; never release evidence")
    mode.add_argument("--prepare-only", action="store_true", help="Populate builds/JIT caches at all requested shapes; no oracle validation or timings")
    ap.add_argument("--execute", action="store_true")
    ap.add_argument("--accuracy-policy", choices=ACCURACY_POLICIES, default="strict")
    ap.add_argument("--output", type=Path)
    args = ap.parse_args()
    if args.smoke:
        args.batches, args.repeats, args.iterations, args.warmups = args.batches or [16], 1, 2, 2
    else:
        args.batches = args.batches or list(BATCHES)
    if args.prepare_only:
        args.repeats, args.iterations, args.warmups = 1, 1, 1
    if any(x < 1 for x in [args.repeats, args.iterations, args.warmups, args.timeout, args.cpu_threads]) or args.warm_seconds < 0 or not args.batches or any(b not in BATCHES for b in args.batches):
        ap.error("Positive repeat/warmup/iteration/timeout values, non-negative warm seconds and batches from 16,32,64,128,256 are required")
    try:
        plan = list(jobs(args.stage, args.robots, args.backends, args.operations))
    except ValueError as error:
        ap.error(str(error))
    payload = {"schema": 1, "purpose": "preparation" if args.prepare_only else "smoke" if args.smoke else "collection",
        "stage": args.stage, "batches": args.batches, "repeats": args.repeats,
        "iterations": args.iterations, "warmups": args.warmups, "warm_seconds": args.warm_seconds,
        "cpu_threads": args.cpu_threads, "jobs": plan}
    payload["arithmetic_policy"] = {"jax_default_matmul_precision": "highest", "jax_enable_x64": False, "cpu_blas_threads": 1}
    payload["accuracy_policy"] = args.accuracy_policy
    payload["accuracy_policy_version"] = ACCURACY_POLICY_VERSION
    payload["fd_warning_max_relative_l2"] = FD_WARNING_MAX_RELATIVE_L2
    payload["compilation_cache"] = {"jax_directory": str(JAX_CACHE), "min_compile_seconds": 0, "min_entry_bytes": -1}
    if not args.execute:
        print(json.dumps(payload, indent=2))
        return
    if args.output is None:
        ap.error("--execute requires a fresh --output directory")
    args.output = args.output.resolve()
    args.output.mkdir(parents=True, exist_ok=False)
    payload["provenance"] = provenance()
    write_json(args.output / "plan.json", payload)
    def interrupted(signum, frame):
        raise KeyboardInterrupt(f"collector received signal {signum}")
    signal.signal(signal.SIGTERM, interrupted)
    results = []
    for job in plan:
        for repeat in range(args.repeats):
            key = f"{job['robot']}-{job['backend']}-{job['operation']}-r{repeat}"
            if job["unavailable"]:
                status, reason = job["unavailable"].split(": ", 1)
                results.append({**job, "repeat": repeat, "status": status, "reason": reason})
                continue
            path = args.output / f"{key}.json"
            cmd = [sys.executable, "-m", "test.benchmarks.release.worker", "--robot", job["robot"],
                   "--backend", job["backend"], "--operation", job["operation"],
                   "--batches", *map(str, args.batches), "--iterations", str(args.iterations),
                   "--warmups", str(args.warmups), "--warm-seconds", str(args.warm_seconds),
                   "--cpu-threads", str(args.cpu_threads),
                   "--accuracy-policy", args.accuracy_policy, "--output", str(path)]
            if args.prepare_only:
                cmd.append("--prepare-only")
            print(f"[{key}] starting", flush=True)
            try:
                rc, expired = run_job(cmd, args.output / f"{key}.log", args.timeout)
            except KeyboardInterrupt:
                results.append({**job, "repeat": repeat, "status": "interrupted", "reason": "collection interrupted; worker group terminated"})
                write_json(args.output / "results.json", {"schema": 1, "purpose": payload["purpose"], "jobs": results})
                raise SystemExit(130)
            record = {**job, "repeat": repeat, "command": cmd, "returncode": rc,
                      "status": "timeout" if expired else "error" if rc else "completed"}
            if path.exists():
                record.update(capture=path.name, sha256=digest(path))
                record["cells"] = json.loads(path.read_text()).get("cells", [])
            results.append(record)
            write_json(args.output / "results.json", {"schema": 1, "purpose": payload["purpose"], "jobs": results})
            print(f"[{key}] {record['status']}", flush=True)
    write_json(args.output / "results.json", {"schema": 1, "purpose": payload["purpose"], "jobs": results})
    write_json(args.output / "manifest.json", {p.name: digest(p) for p in sorted(args.output.iterdir())
        if p.is_file() and p.suffix in {".json", ".npz", ".xml", ".log"}})
    failed = any(r["status"] in {"error", "timeout"} for r in results)
    validated = sum(c.get("status") == "validated" for r in results for c in r.get("cells", []))
    warnings = sum(c.get("status") == "accuracy_warning" for r in results for c in r.get("cells", []))
    prepared = sum(c.get("status") == "prepared" for r in results for c in r.get("cells", []))
    if args.prepare_only:
        print(f"Prepared cells: {prepared}; no accuracy validation or timing samples collected")
    print(f"Strict validated cells: {validated}; accuracy warnings retained: {warnings}; failed jobs: {sum(r['status'] in {'error', 'timeout'} for r in results)}; "
          f"explicit unavailable jobs: {sum(bool(r['unavailable']) for r in results)}")
    print(f"Capture: {args.output}; no speedup/publication approval is implied")
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
