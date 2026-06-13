#!/usr/bin/env python3
"""Time BARD algorithms for one robot.

BARD = "Batched Articulated Robot Dynamics" — a batched PyTorch rigid-body
dynamics library (https://github.com/YueWang996/bard-pytorch-dynamics).

Prints results in the same format as timeGRiD / timeFrax so
timing_parser.parse_grid_output can reuse it. Available algorithms:
  - rnea  -> inverse_dynamics
  - aba   -> forward_dynamics
  - crba  -> crba
BARD has no explicit M^-1 (minv) entry point, and no gradient / second-order /
ee_pose algorithms, so those map to null (rendered `—` by the report).

Conventions (verified to match GRiD/Pinocchio):
  * URDF loading via bard.build_model_from_urdf; joint order == URDF order
    (same order Pinocchio + GRiD use).
  * Gravity passed explicitly as the 3-vector [0, 0, -9.81] (matches GRiD).
  * Fixed base: nq == nv. Floating base: nq == nv + 1 (quaternion free-flyer,
    like Pinocchio) so the config vector q has shape (B, nq) while
    qd/qdd/tau have shape (B, nv).
  * Batched: inputs are (B, n) tensors; the whole batch runs in one call.

Timing discipline (matches timeFrax / timeMJX):
  * Pick the device once (CPU or CUDA) — selected by the BARD_DEVICE env var.
  * Warm up N_WARMUP_PASSES calls, then time TEST_ITERS reps.
  * COMPUTE ONLY: inputs pre-staged on the device.
  * WITH MEMORY: inputs built as host (CPU) tensors and transferred each call
    (host->device transfer included in the timed region — only meaningful for
    the CUDA device; for CPU it degenerates to compute-only + a no-op .to()).

Usage:
    python timeBARD.py <urdf_path> [T/F]
"""

from __future__ import annotations

import os
import sys
import time

import numpy as np

TEST_ITERS  = int(os.environ.get("BENCH_TEST_ITERS", "500"))
BATCH_SIZES = [16, 32, 64, 128, 256, 1024]
N_WARMUP_PASSES = 3

# Select the torch device before timing. BARD advertises CPU + CUDA; the
# multi-version harness runs this script twice (once per device) to capture
# both the bard_cpu and bard_gpu columns. Default: cpu.
_BARD_DEVICE = os.environ.get("BARD_DEVICE", "cpu").strip().lower()
if _BARD_DEVICE not in ("cpu", "gpu", "cuda"):
    _BARD_DEVICE = "cpu"
_TORCH_DEVICE = "cuda" if _BARD_DEVICE in ("gpu", "cuda") else "cpu"


# ---------------------------------------------------------------------------
# Output helpers (same format as timeGRiD / timeFrax)
# ---------------------------------------------------------------------------

def _print_stats(label: str, n: int, times: np.ndarray) -> None:
    print(
        f"[N:{n}]: {label}: "
        f"Average[{np.mean(times):.4f}us] "
        f"Std Dev [{np.std(times):.4f}us] "
        f"Min [{np.min(times):.4f}us] "
        f"Max [{np.max(times):.4f}us]"
    )


def main() -> None:
    if len(sys.argv) < 2:
        print(f"Usage: {sys.argv[0]} <urdf_path> [T/F]", file=sys.stderr)
        sys.exit(1)

    urdf_path     = sys.argv[1]
    floating_base = len(sys.argv) > 2 and sys.argv[2].upper() == "T"

    try:
        import torch
        import bard
    except ImportError as e:
        print(f"BARD import failed: {e}", file=sys.stderr)
        sys.exit(1)

    if _TORCH_DEVICE == "cuda" and not torch.cuda.is_available():
        print("BARD_DEVICE=gpu but torch.cuda.is_available() is False", file=sys.stderr)
        sys.exit(1)

    device = torch.device(_TORCH_DEVICE)
    dtype  = torch.float32  # match the GRiD/Frax/MJX GPU comparison (fp32)

    def _sync():
        if _TORCH_DEVICE == "cuda":
            torch.cuda.synchronize()

    # ------------------------------------------------------------------
    # Load model. nq may differ from nv for floating base (quaternion).
    # ------------------------------------------------------------------
    model = bard.build_model_from_urdf(
        str(urdf_path), floating_base=floating_base, dtype=dtype, device=device
    )
    nq = int(model.nq)
    nv = int(model.nv)
    print(f"# bard: nq={nq}, nv={nv}, floating_base={floating_base}, device={_TORCH_DEVICE}")

    gravity = torch.tensor([0.0, 0.0, -9.81], dtype=dtype, device=device)

    # ------------------------------------------------------------------
    # Metadata block (parsed by run.py::_parse_bard_metadata)
    # ------------------------------------------------------------------
    bard_version = getattr(bard, "__version__", "unknown")
    print("=== BEGIN BARD METADATA ===")
    print(f"bard_version: {bard_version}")
    print(f"torch_version: {torch.__version__}")
    print(f"bard_device: {_TORCH_DEVICE}")
    print(f"nq: {nq}")
    print(f"nv: {nv}")
    print("ID codegen: false")
    print("FD codegen: false")
    print("CRBA codegen: false")
    print("=== END BARD METADATA ===")

    # ------------------------------------------------------------------
    # State factory (device tensors)
    # ------------------------------------------------------------------
    def _make_state(batch: int):
        q   = torch.randn(batch, nq, dtype=dtype, device=device)
        qd  = torch.randn(batch, nv, dtype=dtype, device=device)
        qdd = torch.randn(batch, nv, dtype=dtype, device=device)
        tau = torch.randn(batch, nv, dtype=dtype, device=device)
        return q, qd, qdd, tau

    def _make_state_cpu(batch: int):
        # Host tensors, for the WITH MEMORY (host->device transfer) timing.
        q   = torch.randn(batch, nq, dtype=dtype)
        qd  = torch.randn(batch, nv, dtype=dtype)
        qdd = torch.randn(batch, nv, dtype=dtype)
        tau = torch.randn(batch, nv, dtype=dtype)
        return q, qd, qdd, tau

    # Each algorithm needs update_kinematics(q, qd) first, then the algo call.
    # Time the full (update_kinematics + algo) pipeline, which is what a caller
    # actually pays per state — mirrors GRiD/Frax timing the whole computation.
    def _run_id(data, q, qd, qdd):
        bard.update_kinematics(model, data, q, qd)
        return bard.rnea(model, data, qdd, gravity=gravity)

    def _run_fd(data, q, qd, tau):
        bard.update_kinematics(model, data, q, qd)
        return bard.aba(model, data, tau, gravity=gravity)

    def _run_crba(data, q, qd):
        bard.update_kinematics(model, data, q, qd)
        return bard.crba(model, data)

    # ------------------------------------------------------------------
    # Single-call timing (batch size 1)
    # ------------------------------------------------------------------
    data1 = bard.create_data(model, max_batch_size=1)
    q1, qd1, qdd1, tau1 = _make_state(1)

    def _time_single(fn, *args, n_iters=TEST_ITERS):
        with torch.no_grad():
            for _ in range(N_WARMUP_PASSES + 2):
                fn(*args)
            _sync()
            times = []
            for _ in range(n_iters):
                t0 = time.perf_counter()
                fn(*args)
                _sync()
                times.append((time.perf_counter() - t0) * 1e6)
        return np.array(times)

    for label, fn, args in [
        ("INVERSE_DYNAMICS", _run_id,   (data1, q1, qd1, qdd1)),
        ("FORWARD_DYNAMICS", _run_fd,   (data1, q1, qd1, tau1)),
        ("CRBA",             _run_crba, (data1, q1, qd1)),
    ]:
        try:
            t = _time_single(fn, *args)
            print(f"Single Call {label} {np.median(t):.4f}us")
        except Exception as e:
            print(f"# Single Call {label} skipped: {e}", file=sys.stderr)

    # ------------------------------------------------------------------
    # Batch timing
    # ------------------------------------------------------------------
    n_batch_iters = max(10, TEST_ITERS // 5)

    for N in BATCH_SIZES:
        data = bard.create_data(model, max_batch_size=N)
        q, qd, qdd, tau = _make_state(N)

        # COMPUTE ONLY: device tensors staged ahead of time.
        with torch.no_grad():
            for label, fn, args in [
                ("INVERSE_DYNAMICS", _run_id,   (data, q, qd, qdd)),
                ("FORWARD_DYNAMICS", _run_fd,   (data, q, qd, tau)),
                ("CRBA",             _run_crba, (data, q, qd)),
            ]:
                try:
                    for _ in range(N_WARMUP_PASSES + 2):
                        fn(*args)
                    _sync()
                    times = []
                    for _ in range(n_batch_iters):
                        t0 = time.perf_counter()
                        fn(*args)
                        _sync()
                        times.append((time.perf_counter() - t0) * 1e6)
                    _print_stats(f"{label} COMPUTE ONLY", N, np.array(times))
                except Exception as e:
                    print(f"# [N:{N}] {label} COMPUTE ONLY skipped: {e}", file=sys.stderr)

            # WITH MEMORY: host tensors transferred each call (only meaningful on CUDA).
            def _to_dev(*ts):
                return tuple(t.to(device, non_blocking=True) for t in ts)

            for label, fn, idxs in [
                ("INVERSE_DYNAMICS", _run_id,   (0, 1, 2)),  # q, qd, qdd
                ("FORWARD_DYNAMICS", _run_fd,   (0, 1, 3)),  # q, qd, tau
                ("CRBA",             _run_crba, (0, 1)),      # q, qd
            ]:
                try:
                    # warmup
                    host = _make_state_cpu(N)
                    for _ in range(N_WARMUP_PASSES + 2):
                        fn(data, *_to_dev(*[host[i] for i in idxs]))
                    _sync()
                    times = []
                    for _ in range(n_batch_iters):
                        host = _make_state_cpu(N)   # outside the timer
                        staged = [host[i] for i in idxs]
                        t0 = time.perf_counter()
                        fn(data, *_to_dev(*staged))
                        _sync()
                        times.append((time.perf_counter() - t0) * 1e6)
                    _print_stats(f"{label} WITH MEMORY", N, np.array(times))
                except Exception as e:
                    print(f"# [N:{N}] {label} WITH MEMORY skipped: {e}", file=sys.stderr)


if __name__ == "__main__":
    main()
