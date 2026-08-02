"""grid-rbd × PyTorch on a FLOATING-BASE robot (Unitree Go2): CUDA-resident tensors,
autograd, and CUDA-Graphs replay.

The go2 twin of ``torch_cuda_graphs.py`` (iiwa14, fixed base). What changes with a
floating base:

  * the configuration is nq = 7 + 12 = 19 wide: ``q = [base_pos(3), base_quat_xyzw(4),
    joint_angles(12)]`` with a NORMALIZED quaternion;
  * the tangent space is nv = 6 + 12 = 18 (nv != nq): velocity/torque buffers are still
    passed nq-wide with the trailing quaternion-padding slot zeroed, and only the leading
    nv entries of nq-wide outputs are meaningful;
  * autograd handles the nq<->nv bridge internally (cotangents are sliced to nv for the
    tangent-space Jacobians and padded back to nq).

The .so is built pin-convention-only (``enable_mujoco_kernels=False``) to keep the
one-time nvcc build light; the timing structure (eager wall vs graph-replay wall vs CPU
submission cost, plus the bit-identical replay==eager check) mirrors the iiwa14 example
so ``test/benchmarks/gpu_resident_timing.py`` can drive both with the same parser.

Run:  python bindings/examples/torch_cuda_graphs_go2.py [--urdf PATH] [--batch 256]
Needs: pip install -e .[torch]   ·   a CUDA GPU + torch built with CUDA   ·   go2 URDF
       (auto-resolved via robot_descriptions if installed)
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

_DEFAULT_URDF = (
    Path.home()
    / ".cache/robot_descriptions/unitree_ros/robots/go2_description/urdf/go2_description.urdf"
)


def _resolve_urdf(arg: str) -> Path:
    p = Path(arg).expanduser()
    if p.exists():
        return p
    try:  # robot_descriptions downloads/caches on first use
        from robot_descriptions import go2_description
        return Path(go2_description.URDF_PATH)
    except Exception:
        sys.exit(f"URDF not found: {p} (pass --urdf, or pip install robot_descriptions)")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--urdf", default=str(_DEFAULT_URDF))
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--iters", type=int, default=200)
    args = ap.parse_args()

    urdf = _resolve_urdf(args.urdf)

    import torch
    import grid_rbd
    import grid_rbd.torch as grid_torch

    if not torch.cuda.is_available():
        sys.exit("CUDA not available to torch — this demo is about GPU residency.")
    dev = torch.device("cuda")
    print(f"grid_rbd v{grid_rbd.__version__} · torch {torch.__version__} · {torch.cuda.get_device_name()}")

    grid_rbd.precompile("go2_torch", str(urdf), floating_base=True,
                        max_batch_size=max(args.batch, 256), backends=("torch",),
                        tiers=[{"enable_mujoco_kernels": False}])
    h = grid_torch.get_robot("go2_torch")
    nq, nv, B = h.num_joints, h.num_vel, args.batch
    print(f"  go2 (floating): nq={nq} nv={nv} (nv != nq: quaternion base)  batch B={B}")

    # ── floating-base state: q = [pos(3), quat_xyzw(4) normalized, joints(12)] ──
    g = torch.Generator(device="cuda").manual_seed(0)
    q = torch.rand(B, nq, device=dev, generator=g) * 2 - 1
    q[:, 3:7] = torch.nn.functional.normalize(q[:, 3:7], dim=1)   # unit quaternion
    # qd/u are nq-wide buffers; only the leading nv entries are tangent-space data —
    # zero the trailing quaternion-padding slot.
    qd = torch.rand(B, nq, device=dev, generator=g) * 2 - 1
    u  = torch.rand(B, nq, device=dev, generator=g) * 2 - 1
    qd[:, nv:] = 0.0
    u[:, nv:] = 0.0

    # ── 1. resident call ─────────────────────────────────────────────────────
    qdd = h.forward_dynamics(q, qd, u)
    torch.cuda.synchronize()
    print(f"\n[1] forward_dynamics output {tuple(qdd.shape)} on {qdd.device} "
          f"(nq-wide; leading nv={nv} entries are the tangent qdd)")

    # ── 2. autograd through the analytic backward (nq<->nv bridge inside) ────
    qg = q.clone().requires_grad_(True)
    ug = u.clone().requires_grad_(True)
    loss = h.forward_dynamics(qg, qd, ug)[:, :nv].pow(2).mean() + 1e-3 * ug.pow(2).mean()
    loss.backward()
    print(f"[2] loss={float(loss.detach()):.4f}  →  grads via GRiD analytic Jacobian: "
          f"|∂/∂q|={qg.grad.norm():.4f}  |∂/∂u|={ug.grad.norm():.4f}")

    # ── 3. CUDA-Graphs capture + replay vs eager ─────────────────────────────
    fd_graph = h.capture("forward_dynamics", q, qd, u)   # warmup + record
    out = fd_graph(q, qd, u)                              # copy-in + replay
    torch.cuda.synchronize()
    print(f"[3] captured graph replay output {tuple(out.shape)} on {out.device}")

    out_eager = h.forward_dynamics(q, qd, u)
    torch.cuda.synchronize()
    if not torch.equal(out_eager, out):
        diff = (out_eager - out).abs().max().item()
        sys.exit(f"FATAL: graph replay != eager (max |diff| = {diff:.3e})")
    print(f"    replay == eager: bit-identical (max |diff| = 0)")

    def _time(fn, n):
        """Wall time per call: sync once at the end (throughput of a hot loop)."""
        fn(); torch.cuda.synchronize()                   # warm
        t0 = time.perf_counter()
        for _ in range(n):
            fn()
        torch.cuda.synchronize()
        return (time.perf_counter() - t0) / n * 1e6

    def _submit(fn, n):
        """CPU submission time per call: enqueue only, sync OUTSIDE the timer."""
        fn(); torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(n):
            fn()
        t1 = time.perf_counter()
        torch.cuda.synchronize()
        return (t1 - t0) / n * 1e6

    t_eager = _time(lambda: h.forward_dynamics(q, qd, u), args.iters)
    t_graph = _time(lambda: fd_graph.replay(), args.iters)
    t_fresh = _time(lambda: fd_graph(q, qd, u), args.iters)
    s_eager = _submit(lambda: h.forward_dynamics(q, qd, u), args.iters)
    s_graph = _submit(lambda: fd_graph.replay(), args.iters)
    print(f"    eager {t_eager:8.2f} us   ·   graph-replay {t_graph:8.2f} us"
          f"   →  {t_eager/t_graph:.2f}× wall (B={B})")
    print(f"    fresh-inputs graph call {t_fresh:8.2f} us  (adds D->D copy-in per call)")
    print(f"    CPU submit/iter: eager {s_eager:6.2f} us vs replay {s_graph:6.2f} us"
          f"   →  {s_eager/s_graph:.1f}× less launch overhead")
    print("    note: at these batch sizes forward_dynamics is GPU-execution-bound, so"
          " wall ~= eager; the graph win is the collapsed CPU submission cost (freed"
          " python thread) — it becomes a wall-clock win only in launch-bound regimes.")

    print("\nTakeaway: the floating base changes the STATE handling (normalized quaternion,"
          " nq != nv padding) but not the residency story — capture() the hot loop and"
          " replay() on in-place-updated static_in buffers, exactly as with a fixed base.")


if __name__ == "__main__":
    main()
