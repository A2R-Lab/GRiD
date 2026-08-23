#!/usr/bin/env python3
"""GPU-resident wrapper timing harness (overnight leg C, 2026-07-29; go2 legs 2026-08-01).

Drives the GPU-residency examples as subprocesses across batch sizes and persists their
timing lines as JSON:

  * bindings/examples/jax_gpu_resident.py       -> resident (lax.scan) vs host-roundtrip rollout (ms)
  * bindings/examples/torch_cuda_graphs.py      -> eager vs CUDA-graph replay forward_dynamics (us)
  * bindings/examples/jax_gpu_resident_go2.py   -> go2-FLOATING twin (quaternion state, nv != nq)
  * bindings/examples/torch_cuda_graphs_go2.py  -> go2-FLOATING twin

torch legs (2026-08-01 fix): graph_replay_us is the REPLAY-ONLY wall time (repeated
identical launches — apples-to-apples with the eager loop). The old harness timed
GraphCallable.__call__, whose per-iteration D->D input copy-in made "replay" lose to
eager (0.71-0.94x) — a harness artifact, not a graph property. The copy-in variant and
the CPU submission times (where the graph win actually lives for GPU-bound batches) are
persisted as graph_fresh_inputs_us / cpu_submit_{eager,replay}_us.

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
#     eager  1234.56 us   .   graph-replay   345.67 us   ->  1.02x wall (B=256)
_TORCH_RE = re.compile(
    r"eager\s+(?P<eager_us>[0-9.]+) us\s+.\s+graph-replay\s+(?P<graph_us>[0-9.]+) us")
#     fresh-inputs graph call   456.78 us  (adds D->D copy-in per call)
_TORCH_FRESH_RE = re.compile(r"fresh-inputs graph call\s+(?P<fresh_us>[0-9.]+) us")
#     CPU submit/iter: eager  13.64 us vs replay   1.95 us   ->  7.0x less launch overhead
_TORCH_SUBMIT_RE = re.compile(
    r"CPU submit/iter: eager\s+(?P<submit_eager_us>[0-9.]+) us vs replay\s+"
    r"(?P<submit_replay_us>[0-9.]+) us")


def _run(cmd: list[str], log_path: Path, timeout_s: int) -> tuple[int, str]:
    # ⚠NEVER SIGKILL a GPU process (guide §7.x): subprocess.run's TimeoutExpired
    # path SIGKILLs, and killing a process with live device allocations can
    # wedge the driver (2026-08-22 recurrence). Escalate SIGTERM → grace; if it
    # still won't exit, LEAVE it and report — a stuck pid beats a wedged box.
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                            text=True, cwd=REPO_ROOT)
    try:
        stdout, stderr = proc.communicate(timeout=timeout_s)
        out = stdout + "\n" + stderr
        log_path.write_text(out)
        return proc.returncode, out
    except subprocess.TimeoutExpired:
        proc.terminate()  # SIGTERM: let CUDA teardown sync + free
        try:
            stdout, stderr = proc.communicate(timeout=120)
            out = (stdout or "") + "\n" + (stderr or "") + \
                f"\n!! TIMEOUT after {timeout_s}s (SIGTERM honored)"
        except subprocess.TimeoutExpired:
            out = (f"!! HUNG: no exit {timeout_s}s after start + 120s post-"
                   f"SIGTERM; pid={proc.pid} LEFT RUNNING (SIGKILL would wedge "
                   f"the driver) — stop the run and investigate.")
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

    # (leg_key, kind, example filename) — the go2 legs are the FLOATING twins
    # (quaternion state, nv != nq); same output line formats, same parsers.
    legs = [
        ("jax", "jax", "jax_gpu_resident.py"),
        ("torch", "torch", "torch_cuda_graphs.py"),
        ("jax_go2", "jax", "jax_gpu_resident_go2.py"),
        ("torch_go2", "torch", "torch_cuda_graphs_go2.py"),
    ]

    results: dict = {"metadata": {"host": host, "when": datetime.now().isoformat(),
                                  "steps": args.steps, "iters": args.iters},
                     **{leg: {} for leg, _, _ in legs}, "failures": []}

    for B in batches:
        for leg, kind, script in legs:
            log = log_dir / f"{leg}_B{B}.log"
            if kind == "jax":
                cmd = [py, str(EXAMPLES / script), "--batch", str(B), "--steps", str(args.steps)]
            else:
                cmd = [py, str(EXAMPLES / script), "--batch", str(B), "--iters", str(args.iters)]
            rc, out = _run(cmd, log, args.timeout)
            m = (_JAX_RE if kind == "jax" else _TORCH_RE).search(out)
            if rc == 0 and m:
                if kind == "jax":
                    results[leg][str(B)] = {
                        "resident_rollout_ms": float(m.group("resident_ms")),
                        "host_roundtrip_rollout_ms": float(m.group("host_ms")),
                        "speedup_x": round(float(m.group("host_ms")) / float(m.group("resident_ms")), 2),
                        "steps": int(m.group("steps")),
                    }
                    print(f"[{leg} B={B}] resident {m.group('resident_ms')}ms "
                          f"vs host {m.group('host_ms')}ms")
                else:
                    rec = {
                        "eager_us": float(m.group("eager_us")),
                        "graph_replay_us": float(m.group("graph_us")),
                        "speedup_x": round(float(m.group("eager_us")) / float(m.group("graph_us")), 2),
                    }
                    mf = _TORCH_FRESH_RE.search(out)
                    if mf:
                        rec["graph_fresh_inputs_us"] = float(mf.group("fresh_us"))
                    ms = _TORCH_SUBMIT_RE.search(out)
                    if ms:
                        rec["cpu_submit_eager_us"] = float(ms.group("submit_eager_us"))
                        rec["cpu_submit_replay_us"] = float(ms.group("submit_replay_us"))
                    results[leg][str(B)] = rec
                    print(f"[{leg} B={B}] eager {m.group('eager_us')}us "
                          f"vs graph-replay {m.group('graph_us')}us")
            else:
                results["failures"].append({"leg": leg, "batch": B, "rc": rc,
                                            "parsed": bool(m), "log": str(log)})
                print(f"[{leg} B={B}] FAILED rc={rc} parsed={bool(m)} (see {log})")

    out_path.write_text(json.dumps(results, indent=2))
    print(f"wrote {out_path}  ({len(results['failures'])} failures)")
    sys.exit(1 if sum(len(results[leg]) for leg, _, _ in legs) == 0 else 0)


if __name__ == "__main__":
    main()
