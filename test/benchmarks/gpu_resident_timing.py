#!/usr/bin/env python3
"""GPU-resident wrapper timing harness (overnight leg C, 2026-07-29).

Drives the two existing GPU-residency examples as subprocesses across batch sizes and
persists their timing lines as JSON:

  * bindings/examples/jax_gpu_resident.py   -> resident (lax.scan) vs host-roundtrip rollout (ms)
  * bindings/examples/torch_cuda_graphs.py  -> eager vs CUDA-graph replay forward_dynamics (us)

Each (example, batch) config runs SERIALLY (timing isolation); the example's own in-process
.so build happens before its timed region. Zero parsed lines from a run is recorded as a
FAILED leg (never silently absent — bench-orchestration-traps).

Usage: python test/benchmarks/gpu_resident_timing.py [--batches 64,256,1024]
                                                     [--output results/gpu_resident_<host>.json]
"""
from __future__ import annotations

import argparse
import json
import platform
import re
import subprocess
import sys
from datetime import datetime
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
EXAMPLES = REPO_ROOT / "bindings" / "examples"

# [6] 50-step rollout (B=256):  RESIDENT (lax.scan)   12.34 ms   vs   host-roundtrip-per-step   345.67 ms   ->  28.0x faster...
_JAX_RE = re.compile(
    r"\[6\]\s+(?P<steps>\d+)-step rollout \(B=(?P<batch>\d+)\):\s+RESIDENT \(lax\.scan\)\s+"
    r"(?P<resident_ms>[0-9.]+) ms\s+vs\s+host-roundtrip-per-step\s+(?P<host_ms>[0-9.]+) ms")
#     eager  1234.56 us   .   graph-replay   345.67 us   ->  3.57x less launch overhead (B=256)
_TORCH_RE = re.compile(
    r"eager\s+(?P<eager_us>[0-9.]+) us\s+.\s+graph-replay\s+(?P<graph_us>[0-9.]+) us")


def _run(cmd: list[str], log_path: Path, timeout_s: int) -> tuple[int, str]:
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout_s,
                           cwd=REPO_ROOT)
        out = r.stdout + "\n" + r.stderr
        log_path.write_text(out)
        return r.returncode, out
    except subprocess.TimeoutExpired as e:
        out = (e.stdout or "") + "\n" + (e.stderr or "") + f"\n!! TIMEOUT after {timeout_s}s"
        log_path.write_text(out)
        return -9, out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--batches", default="64,256,1024")
    ap.add_argument("--steps", type=int, default=50)
    ap.add_argument("--iters", type=int, default=200)
    ap.add_argument("--timeout", type=int, default=1800, help="per-config timeout (s)")
    ap.add_argument("--output", type=Path, default=None)
    args = ap.parse_args()

    batches = [int(b) for b in args.batches.split(",") if b.strip()]
    host = platform.node().replace(" ", "_")
    out_path = args.output or (REPO_ROOT / "test" / "benchmarks" / "results"
                               / f"gpu_resident_{host}.json")
    log_dir = out_path.parent / "gpu_resident_logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    py = sys.executable

    results: dict = {"metadata": {"host": host, "when": datetime.now().isoformat(),
                                  "steps": args.steps, "iters": args.iters},
                     "jax": {}, "torch": {}, "failures": []}

    for B in batches:
        # --- jax resident-vs-roundtrip rollout ---
        log = log_dir / f"jax_B{B}.log"
        rc, out = _run([py, str(EXAMPLES / "jax_gpu_resident.py"),
                        "--batch", str(B), "--steps", str(args.steps)], log, args.timeout)
        m = _JAX_RE.search(out)
        if rc == 0 and m:
            results["jax"][str(B)] = {
                "resident_rollout_ms": float(m.group("resident_ms")),
                "host_roundtrip_rollout_ms": float(m.group("host_ms")),
                "speedup_x": round(float(m.group("host_ms")) / float(m.group("resident_ms")), 2),
                "steps": int(m.group("steps")),
            }
            print(f"[jax B={B}] resident {m.group('resident_ms')}ms vs host {m.group('host_ms')}ms")
        else:
            results["failures"].append({"leg": "jax", "batch": B, "rc": rc,
                                        "parsed": bool(m), "log": str(log)})
            print(f"[jax B={B}] FAILED rc={rc} parsed={bool(m)} (see {log})")

        # --- torch eager-vs-graph ---
        log = log_dir / f"torch_B{B}.log"
        rc, out = _run([py, str(EXAMPLES / "torch_cuda_graphs.py"),
                        "--batch", str(B), "--iters", str(args.iters)], log, args.timeout)
        m = _TORCH_RE.search(out)
        if rc == 0 and m:
            results["torch"][str(B)] = {
                "eager_us": float(m.group("eager_us")),
                "graph_replay_us": float(m.group("graph_us")),
                "speedup_x": round(float(m.group("eager_us")) / float(m.group("graph_us")), 2),
            }
            print(f"[torch B={B}] eager {m.group('eager_us')}us vs graph {m.group('graph_us')}us")
        else:
            results["failures"].append({"leg": "torch", "batch": B, "rc": rc,
                                        "parsed": bool(m), "log": str(log)})
            print(f"[torch B={B}] FAILED rc={rc} parsed={bool(m)} (see {log})")

    out_path.write_text(json.dumps(results, indent=2))
    print(f"wrote {out_path}  ({len(results['failures'])} failures)")
    sys.exit(1 if len(results["jax"]) + len(results["torch"]) == 0 else 0)


if __name__ == "__main__":
    main()
