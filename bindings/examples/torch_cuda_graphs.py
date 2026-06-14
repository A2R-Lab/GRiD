"""grid-rbd × PyTorch: CUDA-resident tensors, autograd, and CUDA-Graphs replay.

THE POINT: the `.torch` handle returns CUDA ``torch.Tensor`` from every method and the
forwards are autograd-aware (backward contracts the cotangent with GRiD's ANALYTIC
Jacobian). Inputs and outputs stay on the GPU, so GRiD drops straight into a torch training
or MPC loop. For fixed-batch hot loops, ``handle.capture(method, *example_inputs)`` records a
CUDA Graph: subsequent calls are a memcpy-in + graph replay + memcpy-out — the per-launch CPU
overhead of dozens of kernels collapses to a single replay.

Demonstrates:
  1. CUDA-resident inputs → CUDA-resident outputs (no .cpu() anywhere in the hot path).
  2. Autograd: ``loss.backward()`` flows analytic gradients to q / qd / u.
  3. ``capture()`` → ``GraphCallable``: replay the same kernel graph at fixed batch, timed
     against the eager path to show the launch-overhead win.
  4. (optional) zero-copy dlpack handoff PyTorch → JAX.

Run:  python bindings/examples/torch_cuda_graphs.py [--urdf PATH] [--batch 256]
Needs: pip install -e bindings/[torch]   ·   a CUDA GPU + torch built with CUDA   ·   iiwa14 URDF
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

_DEFAULT_URDF = (
    Path.home()
    / ".cache/robot_descriptions/drake/manipulation/models/iiwa_description/urdf/iiwa14_primitive_collision.urdf"
)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--urdf", default=str(_DEFAULT_URDF))
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--iters", type=int, default=200)
    args = ap.parse_args()

    urdf = Path(args.urdf).expanduser()
    if not urdf.exists():
        sys.exit(f"URDF not found: {urdf} (pass --urdf)")

    import torch
    import grid_rbd
    import grid_rbd.torch as grid_torch

    if not torch.cuda.is_available():
        sys.exit("CUDA not available to torch — this demo is about GPU residency.")
    dev = torch.device("cuda")
    print(f"grid_rbd v{grid_rbd.__version__} · torch {torch.__version__} · {torch.cuda.get_device_name()}")

    grid_rbd.precompile("iiwa14_torch", str(urdf),
                        max_batch_size=max(args.batch, 256), backends=("torch",))
    h = grid_torch.get_robot("iiwa14_torch")
    nq, nv, B = h.num_joints, h.num_vel, args.batch
    print(f"  iiwa14: nq={nq} nv={nv}  batch B={B}")

    g = torch.Generator(device="cuda").manual_seed(0)
    q  = torch.rand(B, nq, device=dev, generator=g) * 2 - 1
    qd = torch.rand(B, nv, device=dev, generator=g) * 2 - 1
    u  = torch.rand(B, nv, device=dev, generator=g) * 2 - 1

    # ── 1. resident call ─────────────────────────────────────────────────────
    qdd = h.forward_dynamics(q, qd, u)
    torch.cuda.synchronize()
    print(f"\n[1] forward_dynamics output {tuple(qdd.shape)} on {qdd.device} (stays on GPU)")

    # ── 2. autograd through the analytic backward ────────────────────────────
    qg = q.clone().requires_grad_(True)
    ug = u.clone().requires_grad_(True)
    loss = h.forward_dynamics(qg, qd, ug).pow(2).mean() + 1e-3 * ug.pow(2).mean()
    loss.backward()
    print(f"[2] loss={float(loss):.4f}  →  grads via GRiD analytic Jacobian: "
          f"|∂/∂q|={qg.grad.norm():.4f}  |∂/∂u|={ug.grad.norm():.4f}")

    # ── 3. CUDA-Graphs capture + replay vs eager ─────────────────────────────
    fd_graph = h.capture("forward_dynamics", q, qd, u)   # warmup + record
    out = fd_graph(q, qd, u)                              # copy-in + replay + copy-out
    torch.cuda.synchronize()
    print(f"[3] captured graph replay output {tuple(out.shape)} on {out.device}")

    def _time(fn, n):
        fn(); torch.cuda.synchronize()                   # warm
        t0 = time.perf_counter()
        for _ in range(n):
            fn()
        torch.cuda.synchronize()
        return (time.perf_counter() - t0) / n * 1e6
    t_eager = _time(lambda: h.forward_dynamics(q, qd, u), args.iters)
    t_graph = _time(lambda: fd_graph(q, qd, u), args.iters)
    print(f"    eager {t_eager:8.2f} us   ·   graph-replay {t_graph:8.2f} us"
          f"   →  {t_eager/t_graph:.2f}× less launch overhead (B={B})")

    # ── 4. zero-copy dlpack handoff torch → JAX ──────────────────────────────
    try:
        import jax
        j = jax.dlpack.from_dlpack(qdd.contiguous())
        print(f"[4] dlpack torch→JAX: jax.Array on {j.devices()} sharing the same GPU buffer")
    except Exception as e:
        print(f"[4] dlpack handoff skipped ({type(e).__name__}: {e})")

    print("\nTakeaway: keep tensors on CUDA, let autograd use GRiD's analytic Jacobians, and "
          "capture() the hot loop for graph-replay throughput in MPC / RL.")


if __name__ == "__main__":
    main()
