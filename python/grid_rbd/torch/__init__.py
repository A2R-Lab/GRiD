"""grid_rbd.torch — PyTorch backend (D.3).

Mirrors the standard grid_rbd API but returns a :py:class:`TorchRobotHandle`
whose methods are autograd-aware torch ops returning ``torch.Tensor`` and
running on the current torch CUDA stream. The four differentiable algorithms
(rnea / forward_dynamics / aba / integrator) carry analytic backward passes
that reuse the existing ``*_gradient`` kernels; the rest are forward-only ops.

Usage:

    import grid_rbd
    h = grid_rbd.register_robot("iiwa14", urdf_path="iiwa.urdf", backend="torch")
    qdd = h.forward_dynamics(q, qd, u)   # torch.Tensor, autograd-aware
    qdd.sum().backward()                 # gradients flow to q, qd, u

The underlying ``.so`` is shared with the plain / JAX surfaces (same content-
addressed cache); the torch op block is compiled into it behind
``-DGRID_RBD_WITH_TORCH`` when torch is installed at register time.

CUDA-Graphs: ``h.capture(method, *example_inputs, **kw)`` returns a
``GraphCallable`` for fixed-batch, low-launch-overhead replay (MPC / training).
A mandatory off-graph warmup runs the one-time device-global setup (the >48 KB
dynamic-smem opt-in via grid_rbd_init) BEFORE capture, since those calls are
illegal during stream capture.

GPU/torch compatibility: the backward VJP contractions (bmm / eye) and tensor
allocation run torch's own CUDA kernels, so the installed torch must support the
GPU's compute capability. On an RTX 5090 (sm_120) you need a cu128 (or newer)
torch build — a cu124 wheel (max sm_90) cannot launch any CUDA kernel on
sm_120 ("no kernel image is available for execution on the device"). The
grid_plant/grid kernels themselves are always nvcc-built for the detected arch
and are unaffected; only torch's own kernels carry this requirement.
"""
from __future__ import annotations

import ctypes
import threading
from pathlib import Path
from typing import Any

import grid_rbd as _grid_rbd
from grid_rbd._handle import RobotHandle, _integrator_code


# ─── op-library registry (process-global, idempotent) ───────────────────────

# Maps cache_key → the torch op namespace (e.g. "grid_rbd_torch_k03e0c8d05d7e").
# torch.ops.load_library is idempotent per .so path, but we guard so a later
# notebook cell calling a method never re-loads.
_LOADED: dict[str, str] = {}
_LOCK = threading.Lock()


def _torch_op_namespace(cache_key: str) -> str:
    """Mirror _compile.generate_and_compile's torch_op_key = 'k' + key[:12]."""
    return f"grid_rbd_torch_k{cache_key[:12]}"


def _load_ops(so_path: Path, cache_key: str) -> str:
    """Load the .so's torch ops (once per cache_key); return the op namespace."""
    import torch
    ns = _torch_op_namespace(cache_key)
    with _LOCK:
        if _LOADED.get(cache_key):
            return ns
        # Sanity: the .so must export the torch op block. If it was built
        # without torch (torch installed after register), raise a clear error.
        try:
            lib = ctypes.CDLL(str(so_path))
            del lib
        except OSError as e:
            raise RuntimeError(f"failed to dlopen {so_path}: {e}") from e
        torch.ops.load_library(str(so_path))
        # torch.ops.<ns> is created lazily, so its mere existence proves nothing;
        # probe for a concrete op to confirm the TORCH_LIBRARY block registered.
        try:
            getattr(getattr(torch.ops, ns), "rnea")
        except AttributeError as e:
            raise RuntimeError(
                f"torch op {ns}.rnea not found in {so_path}; was the .so compiled "
                f"with GRID_RBD_WITH_TORCH (torch installed at register time)? "
                f"Re-register with force_rebuild=True."
            ) from e
        _LOADED[cache_key] = ns
    return ns


# ─── autograd Functions (backward = batched VJP via the *_gradient kernels) ──
#
# GRiD emits the FULL analytic per-timestep Jacobian; torch autograd needs vᵀJ.
# Each backward calls the gradient op (forward of the gradient), reshapes to the
# row-major convention, then bmm's the upstream grad against it — on-GPU and
# itself graph-capturable.


def _make_autograd(ns):
    import torch

    ops = getattr(torch.ops, ns)

    class RneaFn(torch.autograd.Function):
        @staticmethod
        def forward(ctx, q, qd, gravity):
            ctx.save_for_backward(q, qd)
            ctx.gravity = gravity
            return ops.rnea(q, qd, gravity)

        @staticmethod
        def backward(ctx, grad_c):
            q, qd = ctx.saved_tensors
            nj = q.shape[1]
            raw = ops.rnea_grad(q, qd, ctx.gravity)  # (B, 2*NJ*NJ) col-major blocks
            B = raw.shape[0]
            blocks = raw.reshape(B, 2, nj, nj).transpose(2, 3)  # row-major (B,2,NJ,NJ)
            dc_dq, dc_dqd = blocks[:, 0], blocks[:, 1]  # (B, NJ, NJ): rows=out, cols=in
            # VJP: grad_in = grad_c · J  →  (B,1,NJ) bmm (B,NJ,NJ) = (B,1,NJ)
            gc = grad_c.unsqueeze(1)
            grad_q = torch.bmm(gc, dc_dq).squeeze(1)
            grad_qd = torch.bmm(gc, dc_dqd).squeeze(1)
            return grad_q, grad_qd, None

    def _make_fd_like(fwd_op):
        # forward_dynamics & aba share the qdd output and the fd-grad backward
        # (∂qdd/∂(q,qd) from fd_grad; ∂qdd/∂u = M⁻¹). One factory, two ops.
        class FDLikeFn(torch.autograd.Function):
            @staticmethod
            def forward(ctx, q, qd, u, gravity):
                ctx.save_for_backward(q, qd, u)
                ctx.gravity = gravity
                return fwd_op(q, qd, u, gravity)

            @staticmethod
            def backward(ctx, grad_qdd):
                q, qd, u = ctx.saved_tensors
                nj = q.shape[1]
                raw = ops.forward_dynamics_grad(q, qd, u, ctx.gravity)
                B = raw.shape[0]
                blocks = raw.reshape(B, 2, nj, nj).transpose(2, 3)
                df_dq, df_dqd = blocks[:, 0], blocks[:, 1]
                # ∂qdd/∂u = M⁻¹ (symmetric); minv writes lower triangle → symmetrize.
                m = ops.minv(q).reshape(B, nj, nj)
                eye = torch.eye(nj, dtype=m.dtype, device=m.device)
                minv = m + m.transpose(1, 2) - m * eye
                g = grad_qdd.unsqueeze(1)
                grad_q = torch.bmm(g, df_dq).squeeze(1)
                grad_qd = torch.bmm(g, df_dqd).squeeze(1)
                grad_u = torch.bmm(g, minv).squeeze(1)
                return grad_q, grad_qd, grad_u, None
        return FDLikeFn

    FDFn = _make_fd_like(lambda q, qd, u, g: ops.forward_dynamics(q, qd, u, g))
    AbaFn = _make_fd_like(lambda q, qd, u, g: ops.aba(q, qd, u, g))

    class IntegratorFn(torch.autograd.Function):
        @staticmethod
        def forward(ctx, q, qd, u, dt, it, gravity):
            ctx.save_for_backward(q, qd, u)
            ctx.dt, ctx.it, ctx.gravity = dt, it, gravity
            return ops.integrator(q, qd, u, dt, it, gravity)

        @staticmethod
        def backward(ctx, grad_x):
            q, qd, u = ctx.saved_tensors
            nv = q.shape[1]
            raw = ops.integrator_gradient(q, qd, u, ctx.dt, ctx.it, ctx.gravity)
            B = raw.shape[0]
            # h_dAB is (2*NV x 3*NV) column-major per ts → row-major (B, 2*NV, 3*NV).
            dAB = raw.reshape(B, 3 * nv, 2 * nv).transpose(1, 2)  # (B, 2NV, 3NV)
            # output x_{k+1} is 2*NV (fixed-base: nq==nv). grad_x is (B, 2*NV).
            gx = grad_x.unsqueeze(1)  # (B,1,2NV)
            vjp = torch.bmm(gx, dAB).squeeze(1)  # (B, 3NV) = [d/dq | d/dqd | d/du]
            grad_q = vjp[:, :nv]
            grad_qd = vjp[:, nv:2 * nv]
            grad_u = vjp[:, 2 * nv:3 * nv]
            return grad_q, grad_qd, grad_u, None, None, None

    return {"rnea": RneaFn, "fd": FDFn, "aba": AbaFn, "integrator": IntegratorFn}


# ─── CUDA-Graphs callable ───────────────────────────────────────────────────


class GraphCallable:
    """A CUDA-Graphs-captured op for fixed-batch replay.

    ``static_in`` are the captured input tensors (``.copy_()`` new data in);
    ``static_out`` is the captured output; ``replay()`` re-runs the graph.
    """

    def __init__(self, op, example_inputs, kwargs):
        import torch
        self._torch = torch
        self.static_in = [t.clone() for t in example_inputs]
        self._kwargs = kwargs
        # 1. WARMUP off-graph: forces grid_rbd_init (>48KB smem opt-in) + first
        #    allocs. These device-global registrations are illegal during capture.
        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            for _ in range(3):
                out = op(*self.static_in, **kwargs)
        torch.cuda.current_stream().wait_stream(s)
        torch.cuda.synchronize()
        # 2. CAPTURE: only the memcpy-repack + kernel launch + memcpy-out remain,
        #    all stream-ordered / capturable.
        self.graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.graph):
            self.static_out = op(*self.static_in, **kwargs)
        self._op = op

    def replay(self):
        self.graph.replay()
        return self.static_out

    def __call__(self, *inputs):
        if len(inputs) != len(self.static_in):
            raise ValueError(f"expected {len(self.static_in)} inputs, got {len(inputs)}")
        for dst, src in zip(self.static_in, inputs):
            dst.copy_(src)
        return self.replay()


# ─── TorchRobotHandle ───────────────────────────────────────────────────────


class TorchRobotHandle:
    """Torch-flavored wrapper. Methods return ``torch.Tensor`` (autograd-aware
    for rnea / forward_dynamics / aba / integrator)."""

    def __init__(self, base: RobotHandle, cache_key: str, so_path: str):
        self._base = base
        self._cache_key = cache_key
        self._so_path = Path(so_path)
        self._ns = _load_ops(self._so_path, cache_key)
        import torch
        self._ops = getattr(torch.ops, self._ns)
        self._fns = _make_autograd(self._ns)

    # ─── metadata (delegated) ────────────────────────────────────────────
    @property
    def name(self) -> str:        return self._base.name
    @property
    def num_joints(self) -> int:  return self._base.num_joints
    @property
    def num_vel(self) -> int:     return self._base.num_vel
    @property
    def num_ees(self) -> int:     return self._base.num_ees
    @property
    def floating_base(self) -> bool: return self._base.floating_base
    @property
    def max_batch(self) -> int:   return self._base.max_batch

    # ─── differentiable algorithms ───────────────────────────────────────

    def rnea(self, q, qd, *, gravity: float = 9.81):
        """Inverse dynamics c (B, NJ). Autograd-aware wrt (q, qd)."""
        return self._fns["rnea"].apply(q, qd, float(gravity))

    def forward_dynamics(self, q, qd, u, *, gravity: float = 9.81):
        """qdd = M⁻¹(τ − c) (B, NJ). Autograd-aware wrt (q, qd, u)."""
        return self._fns["fd"].apply(q, qd, u, float(gravity))

    def aba(self, q, qd, u, *, gravity: float = 9.81):
        """qdd via ABA (B, NJ). Autograd-aware wrt (q, qd, u)."""
        return self._fns["aba"].apply(q, qd, u, float(gravity))

    def integrator(self, q, qd, u, dt, *, integrator_type: str = "euler", gravity: float = 9.81):
        """x_{k+1} (B, NP+NV). Autograd-aware wrt (q, qd, u)."""
        it = _integrator_code(integrator_type)
        return self._fns["integrator"].apply(q, qd, u, float(dt), it, float(gravity))

    # ─── forward-only algorithms (raw kernel ops; reshapes mirror _handle) ──

    def minv(self, q):
        """Minv(q) (B, NJ, NJ), symmetrized (kernel writes lower triangle)."""
        import torch
        nj = self.num_joints
        m = self._ops.minv(q).reshape(-1, nj, nj)
        eye = torch.eye(nj, dtype=m.dtype, device=m.device)
        return m + m.transpose(1, 2) - m * eye

    def crba(self, q, *, gravity: float = 9.81):
        """Mass matrix M(q) (B, NJ, NJ)."""
        nj = self.num_joints
        return self._ops.crba(q, float(gravity)).reshape(-1, nj, nj)

    def end_effector_pose(self, q):
        """EE pose [xyz, rpy] per EE (B, 6*NUM_EES)."""
        return self._ops.end_effector_pose(q)

    def end_effector_pose_gradient(self, q):
        """EE pose Jacobian d/dv (B, 6*NEE, NV), pinocchio tangent convention."""
        nee, nv = self.num_ees, self.num_vel
        raw = self._ops.end_effector_pose_gradient(q)
        B = raw.shape[0]
        return raw.reshape(B, nee, nv, 6).permute(0, 1, 3, 2).reshape(B, 6 * nee, nv)

    def end_effector_pose_hessian(self, q):
        """EE pose Hessian d²/dv² (B, 6*NEE, NV, NV)."""
        nee, nv = self.num_ees, self.num_vel
        return self._ops.end_effector_pose_hessian(q).reshape(-1, 6 * nee, nv, nv)

    def rnea_grad(self, q, qd, *, gravity: float = 9.81):
        """∂c/∂(q,qd) (B, NJ, 2*NJ) = [dc_dq | dc_dqd]."""
        nj = self.num_joints
        raw = self._ops.rnea_grad(q, qd, float(gravity))
        B = raw.shape[0]
        blocks = raw.reshape(B, 2, nj, nj).transpose(2, 3)
        return _concat_blocks(blocks)

    def forward_dynamics_grad(self, q, qd, u, *, gravity: float = 9.81):
        """∂qdd/∂(q,qd) (B, NJ, 2*NJ)."""
        nj = self.num_joints
        raw = self._ops.forward_dynamics_grad(q, qd, u, float(gravity))
        B = raw.shape[0]
        blocks = raw.reshape(B, 2, nj, nj).transpose(2, 3)
        return _concat_blocks(blocks)

    def idsva_so(self, q, qd, *, gravity: float = 9.81):
        """Second-order ID: 4 tensors each (B, NV, NV, NV)."""
        nv = self.num_vel
        flat = self._ops.idsva_so(q, qd, float(gravity))
        B = flat.shape[0]
        return tuple(flat[:, i*nv**3:(i+1)*nv**3].reshape(B, nv, nv, nv) for i in range(4))

    def fdsva_so(self, q, qd, u, *, gravity: float = 9.81):
        """Second-order FD: 4 tensors each (B, NV, NV, NV)."""
        nv = self.num_vel
        flat = self._ops.fdsva_so(q, qd, u, float(gravity))
        B = flat.shape[0]
        return tuple(flat[:, i*nv**3:(i+1)*nv**3].reshape(B, nv, nv, nv) for i in range(4))

    def integrator_gradient(self, q, qd, u, dt, *, integrator_type: str = "euler", gravity: float = 9.81):
        """dAB (B, 2*NV, 3*NV) = [d/dq | d/dqd | d/du] tangent."""
        nv = self.num_vel
        it = _integrator_code(integrator_type)
        raw = self._ops.integrator_gradient(q, qd, u, float(dt), it, float(gravity))
        B = raw.shape[0]
        return raw.reshape(B, 3 * nv, 2 * nv).transpose(1, 2)

    # ─── CUDA-Graphs capture ─────────────────────────────────────────────

    def capture(self, method: str, *example_inputs, **kwargs) -> GraphCallable:
        """Capture ``method`` at the example inputs' fixed batch into a
        replayable CUDA graph. A mandatory off-graph warmup runs the >48 KB
        dynamic-smem opt-in (grid_rbd_init) before capture.

        Returns a :py:class:`GraphCallable` — call it with new inputs (same
        shapes) to ``.copy_()`` + ``.replay()``, or use ``.static_in`` /
        ``.replay()`` directly.
        """
        op = getattr(self, method)
        return GraphCallable(op, example_inputs, kwargs)

    def __repr__(self) -> str:
        return (f"TorchRobotHandle(name={self.name!r}, num_joints={self.num_joints}, "
                f"num_vel={self.num_vel}, num_ees={self.num_ees}, "
                f"floating_base={self.floating_base})")


def _concat_blocks(blocks):
    import torch
    return torch.cat([blocks[:, 0], blocks[:, 1]], dim=-1)


# ─── public API ─────────────────────────────────────────────────────────────


def register_robot(
    name: str,
    urdf_path: str | None = None,
    *,
    urdf_string: str | None = None,
    floating_base: bool = False,
    ee_joint_names: list[str] | tuple[str, ...] | None = None,
    max_batch_size: int = 256,
    cache_dir: str | Path | None = None,
    force_rebuild: bool = False,
    cuda_arch: int | None = None,
) -> TorchRobotHandle:
    """Register a robot for the torch backend (same cache as the plain/JAX
    surfaces). Returns a :py:class:`TorchRobotHandle`."""
    base = _grid_rbd.register_robot(
        name=name, urdf_path=urdf_path, urdf_string=urdf_string,
        floating_base=floating_base, ee_joint_names=ee_joint_names,
        max_batch_size=max_batch_size, cache_dir=cache_dir,
        force_rebuild=force_rebuild, cuda_arch=cuda_arch,
    )
    cache_key, so_path = _lookup(name, cache_dir)
    return TorchRobotHandle(base, cache_key, so_path)


def get_robot(name: str, cache_dir: str | Path | None = None) -> TorchRobotHandle:
    """Look up a previously-registered robot (same cache as grid_rbd.get_robot)."""
    base = _grid_rbd.get_robot(name, cache_dir=cache_dir)
    cache_key, so_path = _lookup(name, cache_dir)
    return TorchRobotHandle(base, cache_key, so_path)


def _lookup(name, cache_dir):
    from grid_rbd._cache import default_cache_dir, manifest_lookup, store_dir
    cd = Path(cache_dir).expanduser() if cache_dir else default_cache_dir()
    entry = manifest_lookup(cd, name)
    if entry is None:
        raise RuntimeError(f"{name!r} isn't in manifest; cache may be corrupted")
    so_path = store_dir(cd, entry["cache_key"]) / "robot.so"
    return entry["cache_key"], str(so_path)


__all__ = ["TorchRobotHandle", "GraphCallable", "register_robot", "get_robot"]
