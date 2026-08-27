"""grid_rbd.torch — PyTorch backend (D.3).

Mirrors the standard grid_rbd API but returns a :py:class:`TorchRobotHandle`
whose methods are autograd-aware torch ops returning ``torch.Tensor`` and
running on the current torch CUDA stream. The differentiable algorithms
(inverse_dynamics / forward_dynamics / aba / integrator, plus the inertial-param
sysID ops inverse_dynamics_wrt_params / forward_dynamics_wrt_params) carry
analytic backward passes that reuse the existing ``*_gradient`` / regressor
kernels; the rest are forward-only ops.

The forward-only centroidal / energy / kinematics surface mirrors the numpy
handle: ``generalized_gravity``, ``nonlinear_effects``, ``energy``, ``com``,
``ccrba``, ``dccrba``, ``cmm_time_variation``, ``coriolis_matrix``,
``kinetic_energy_regressor``, ``potential_energy_regressor``, ``frame_jacobian``,
``frame_jacobian_dot``, ``osc_inertia``, ``end_effector_pose_runtime``,
``end_effector_pose_gradient_runtime`` (the gated ones raise an actionable
"not generated for this robot" error when their kernel is absent from the .so).

The grid_plant cost / barrier / plant-step surface is also exposed
(``plant_step``, ``plant_step_gradient``, ``quadratic_state_cost``,
``quadratic_input_cost``, ``ee_pos_cost``, ``joint_position_barrier``,
``joint_velocity_barrier``, ``joint_torque_barrier``, ``com_cost``,
``momentum_cost``) as forward-only ops matching the numpy handle's shapes; the
ops needing a gated kernel (plant_step[_gradient], ee/com/momentum cost) are
registered only when the per-robot ``.so`` was built with that kernel.

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
from grid_rbd._handle import RobotHandle, SecondOrderID, SecondOrderFD, _integrator_code


# ─── op-library registry (process-global, idempotent) ───────────────────────

# Maps cache_key → the torch op namespace (e.g. "grid_rbd_torch_k03e0c8d05d7e").
# torch.ops.load_library is idempotent per .so path, but we guard so a later
# notebook cell calling a method never re-loads.
_LOADED: dict[str, str] = {}
_LOCK = threading.Lock()


def _require_torch():
    """Import + return the ``torch`` module, or raise a clear, actionable error.

    Routed through the surface entry points (register_robot / get_robot /
    _load_ops) so a missing optional dep gives install guidance instead of a
    bare ``ModuleNotFoundError`` from deep inside a method."""
    try:
        import torch
    except ImportError as e:
        raise ImportError(
            "grid_rbd.torch requires PyTorch, which isn't installed. Install "
            "the optional extra:  pip install -e 'grid-rbd[torch]'  (or "
            "`pip install torch`; on an RTX 5090 / sm_120 you need a cu128+ "
            "build). The numpy and jax backends do not need torch."
        ) from e
    return torch


def _torch_op_namespace(cache_key: str) -> str:
    """Mirror _compile.compile_sources's torch_op_key = 'k' + key[:12]."""
    return f"grid_rbd_torch_k{cache_key[:12]}"


# A torch op that is ALWAYS registered when the torch surface compiles (an ungated
# plant cost op — not behind any per-algo `#if GRID_HAS_*`). Used to tell a
# SUBSET-omitted CORE algo (this present, the requested op absent) apart from a
# never-built op, so the clean "not built — add to algorithm_list" error only fires
# for a genuine subset gap.
_TORCH_SURFACE_SENTINEL = "quadratic_input_cost"


def _resolve_core_op(ops, name: str):
    """Resolve a CORE torch op (id/fd/minv/.../regressor/integrator/...), mapping a
    missing op to the clean subset error the numpy rc=3 path raises. A core op is
    absent only on a SUBSET build (algorithm_list) that omitted it — distinguished
    from a whole-surface-missing .so via the always-present sentinel op."""
    try:
        return getattr(ops, name)
    except AttributeError as e:
        # mjx-suffixed names that resolve through here strip the suffix for the msg.
        base = name[:-len("_mujoco")] if name.endswith("_mujoco") else name
        if hasattr(ops, _TORCH_SURFACE_SENTINEL):
            raise RuntimeError(
                f"{base!r} not built into this robot .so — add {base!r} to "
                f"algorithm_list in register_robot() and rebuild (force_rebuild=True). "
                f"The torch op surface is present but this algorithm was excluded by "
                f"the subset build."
            ) from e
        raise


def _load_ops(so_path: Path, cache_key: str) -> str:
    """Load the .so's torch ops (once per cache_key); return the op namespace."""
    torch = _require_torch()
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
        # probe for the ALWAYS-PRESENT sentinel op (an ungated plant cost op) to
        # confirm the TORCH_LIBRARY block registered. Probing a core op (e.g.
        # inverse_dynamics) would FALSELY fail a SUBSET .so that omits that core,
        # so use the sentinel which every torch surface emits.
        try:
            getattr(getattr(torch.ops, ns), _TORCH_SURFACE_SENTINEL)
        except AttributeError as e:
            raise RuntimeError(
                f"torch op {ns}.{_TORCH_SURFACE_SENTINEL} not found in {so_path}; was the .so "
                f"compiled with GRID_RBD_WITH_TORCH (torch installed at register time)? "
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


def _make_autograd(ns, nv, mujoco=False):
    import torch

    ops = getattr(torch.ops, ns)

    def _op(name):
        # In mjx mode every forward/backward op dispatches to its _mujoco
        # variant (kernel launched with MUJOCO_OUTPUT=true): mjx-convention
        # forward + mjx-convention analytic Jacobian, so backward stays
        # self-consistent. mjx is FLOATING-base only (the _mujoco symbols are
        # #ifdef'd out of fixed .so). A CORE op absent on a SUBSET build maps to
        # the clean "not built — add to algorithm_list" error (not a bare
        # AttributeError) via _resolve_core_op.
        return _resolve_core_op(ops, (name + "_mujoco") if mujoco else name)

    # nq↔nv bridge for the backward VJPs. The dynamics VALUE outputs (c / qdd)
    # are nj-wide (so grad_c / grad_qdd are nj-wide), but the analytic Jacobians
    # / Minv / regressor rows are nv-wide (tangent space). For a FLOATING base
    # nv < nj: slice the leading nv of the value cotangent (the meaningful
    # tangent rows; the trailing nj-vs-nv slot is the quaternion-padding of the
    # nj-wide value buffer) before bmm, and pad the nv-wide input cotangent back
    # to nj for the nj-wide q/qd/u inputs. FIXED base nv == nj → both no-ops.
    def _slice_nv(ct, nj, nv):
        return ct if nv == nj else ct[:, :nv]

    def _pad_nj(g, nj, nv):
        if nv == nj:
            return g
        return torch.nn.functional.pad(g, (0, nj - nv))

    class InverseDynamicsFn(torch.autograd.Function):
        # forward args mirror the op schema order (q, qd, gravity, qdd, f_ext);
        # qdd/gravity/f_ext are non-differentiated (backward returns None for them).
        @staticmethod
        def forward(ctx, q, qd, gravity, qdd, f_ext):
            ctx.save_for_backward(q, qd)
            ctx.gravity = gravity
            ctx.qdd = qdd
            ctx.f_ext = f_ext
            ctx.nv = nv
            return _op("inverse_dynamics")(q, qd, gravity, qdd, f_ext)

        @staticmethod
        def backward(ctx, grad_c):
            q, qd = ctx.saved_tensors
            nj = q.shape[1]
            nv = ctx.nv
            # f_ext is affine in RNEA → ∂c/∂(q,qd) is unchanged by a constant
            # f_ext; we pass it through for bias consistency only. qdd shifts the
            # value (M·qdd); ∂/∂(q,qd) at fixed qdd is the bias gradient plus
            # ∂(M·qdd)/∂q — included by threading the saved ctx.qdd into the grad
            # op (USE_QDD overload). A None/zero qdd reduces to the bias Jacobian.
            raw = _op("inverse_dynamics_gradient")(q, qd, ctx.gravity, ctx.qdd, ctx.f_ext)  # (B, 2*NV*NV) col-major
            B = raw.shape[0]
            blocks = raw.reshape(B, 2, nv, nv).transpose(2, 3)  # row-major (B,2,NV,NV)
            dc_dq, dc_dqd = blocks[:, 0], blocks[:, 1]  # (B, NV, NV): rows=out, cols=in
            # VJP: grad_in = grad_c · J → (B,1,NV) bmm (B,NV,NV) = (B,1,NV); the
            # torque cotangent is nj-wide so slice leading nv, then pad result to nj.
            gc = _slice_nv(grad_c, nj, nv).unsqueeze(1)
            grad_q = _pad_nj(torch.bmm(gc, dc_dq).squeeze(1), nj, nv)
            grad_qd = _pad_nj(torch.bmm(gc, dc_dqd).squeeze(1), nj, nv)
            # grads for (q, qd, gravity, qdd, f_ext)
            return grad_q, grad_qd, None, None, None

    def _make_fd_like(fwd_op):
        # forward_dynamics & aba share the qdd output and the fd-grad backward
        # (∂qdd/∂(q,qd) from fd_grad; ∂qdd/∂u = M⁻¹). One factory, two ops.
        class FDLikeFn(torch.autograd.Function):
            @staticmethod
            def forward(ctx, q, qd, u, gravity, f_ext):
                ctx.save_for_backward(q, qd, u)
                ctx.gravity = gravity
                ctx.f_ext = f_ext
                return fwd_op(q, qd, u, gravity, f_ext)

            @staticmethod
            def backward(ctx, grad_qdd):
                q, qd, u = ctx.saved_tensors
                nj = q.shape[1]
                raw = _op("forward_dynamics_gradient")(q, qd, u, ctx.gravity, ctx.f_ext)
                B = raw.shape[0]
                blocks = raw.reshape(B, 2, nv, nv).transpose(2, 3)
                df_dq, df_dqd = blocks[:, 0], blocks[:, 1]
                # ∂qdd/∂u = M⁻¹ (NV x NV). The pin minv kernel writes only the
                # lower triangle → symmetrize; the mjx minv_mujoco kernel writes a
                # FULL DENSE symmetric matrix (the G^-T Minv G^-1 congruence baked
                # in) so it is used as-is.
                m = _op("minv")(q).reshape(B, nv, nv)
                if mujoco:
                    minv = m
                else:
                    eye = torch.eye(nv, dtype=m.dtype, device=m.device)
                    minv = m + m.transpose(1, 2) - m * eye
                # qdd cotangent is nj-wide → slice leading nv, bmm, pad back to nj.
                g = _slice_nv(grad_qdd, nj, nv).unsqueeze(1)
                grad_q = _pad_nj(torch.bmm(g, df_dq).squeeze(1), nj, nv)
                grad_qd = _pad_nj(torch.bmm(g, df_dqd).squeeze(1), nj, nv)
                grad_u = _pad_nj(torch.bmm(g, minv).squeeze(1), nj, nv)
                return grad_q, grad_qd, grad_u, None, None
        return FDLikeFn

    FDFn = _make_fd_like(lambda q, qd, u, g, fe: _op("forward_dynamics")(q, qd, u, g, fe))
    AbaFn = _make_fd_like(lambda q, qd, u, g, fe: _op("aba")(q, qd, u, g, fe))

    # ── inertial-parameter (sysID) VJPs ──
    # The forward op is independent of the `params` (π) VALUE (the compiled .so
    # carries the baked-in inertia); π exists so autograd can flow the analytic
    # ∂(·)/∂π to it — the linearization of the bias / qdd around the compiled
    # model (mirrors the JAX idyn_pi / fd_pi custom_vjp). q/qd[/u] cotangents
    # flow exactly as the plain id / fd VJPs above.

    class IDWrtParamsFn(torch.autograd.Function):
        @staticmethod
        def forward(ctx, q, qd, params, gravity, f_ext):
            # forward value ignores `params`; baked-in inertia → c = ID(q, qd).
            ctx.save_for_backward(q, qd)
            ctx.gravity = gravity
            ctx.f_ext = f_ext
            return ops.inverse_dynamics(q, qd, gravity, None, f_ext)

        @staticmethod
        def backward(ctx, grad_c):
            q, qd = ctx.saved_tensors
            nj = q.shape[1]
            # sysID is the bias gradient (qdd=0) → pass None for the qdd slot.
            raw = ops.inverse_dynamics_gradient(q, qd, ctx.gravity, None, ctx.f_ext)
            B = raw.shape[0]
            blocks = raw.reshape(B, 2, nv, nv).transpose(2, 3)
            dc_dq, dc_dqd = blocks[:, 0], blocks[:, 1]
            # torque cotangent nj-wide → slice leading nv, bmm, pad inputs to nj.
            gc = _slice_nv(grad_c, nj, nv).unsqueeze(1)
            grad_q = _pad_nj(torch.bmm(gc, dc_dq).squeeze(1), nj, nv)
            grad_qd = _pad_nj(torch.bmm(gc, dc_dqd).squeeze(1), nj, nv)
            # π cotangent: gc · Y, Y = ∂c/∂π (NV x 10*NB) at qdd=0 (no input-pad: π is npar-wide).
            qdd0 = torch.zeros_like(q)
            Y = ops.inverse_dynamics_regressor(q, qd, qdd0, ctx.gravity).reshape(B, nv, -1)
            grad_pi = torch.bmm(gc, Y).squeeze(1)
            return grad_q, grad_qd, grad_pi, None, None

    class FDWrtParamsFn(torch.autograd.Function):
        @staticmethod
        def forward(ctx, q, qd, u, params, gravity, f_ext):
            ctx.save_for_backward(q, qd, u)
            ctx.gravity = gravity
            ctx.f_ext = f_ext
            return ops.forward_dynamics(q, qd, u, gravity, f_ext)

        @staticmethod
        def backward(ctx, grad_qdd):
            q, qd, u = ctx.saved_tensors
            nj = q.shape[1]
            raw = ops.forward_dynamics_gradient(q, qd, u, ctx.gravity, ctx.f_ext)
            B = raw.shape[0]
            blocks = raw.reshape(B, 2, nv, nv).transpose(2, 3)
            df_dq, df_dqd = blocks[:, 0], blocks[:, 1]
            m = ops.minv(q).reshape(B, nv, nv)
            eye = torch.eye(nv, dtype=m.dtype, device=m.device)
            minv = m + m.transpose(1, 2) - m * eye
            # qdd cotangent nj-wide → slice leading nv, bmm, pad q/qd/u inputs to nj.
            g = _slice_nv(grad_qdd, nj, nv).unsqueeze(1)
            grad_q = _pad_nj(torch.bmm(g, df_dq).squeeze(1), nj, nv)
            grad_qd = _pad_nj(torch.bmm(g, df_dqd).squeeze(1), nj, nv)
            grad_u = _pad_nj(torch.bmm(g, minv).squeeze(1), nj, nv)
            # π cotangent: g · (∂qdd/∂π), ∂qdd/∂π = -Minv·Y (NV x 10*NB); π is npar-wide.
            G = ops.forward_dynamics_parameter_gradient(q, qd, u, ctx.gravity).reshape(B, nv, -1)
            grad_pi = torch.bmm(g, G).squeeze(1)
            return grad_q, grad_qd, grad_u, grad_pi, None, None

    class IntegratorFn(torch.autograd.Function):
        @staticmethod
        def forward(ctx, q, qd, u, dt, it, gravity):
            ctx.save_for_backward(q, qd, u)
            ctx.dt, ctx.it, ctx.gravity = dt, it, gravity
            return _op("integrator")(q, qd, u, dt, it, gravity)

        @staticmethod
        def backward(ctx, grad_x):
            q, qd, u = ctx.saved_tensors
            nv = q.shape[1]
            raw = _op("integrator_gradient")(q, qd, u, ctx.dt, ctx.it, ctx.gravity)
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

    fns = {"inverse_dynamics": InverseDynamicsFn, "fd": FDFn, "aba": AbaFn,
           "integrator": IntegratorFn}
    if not mujoco:
        # sysID (inverse/forward_dynamics_wrt_params) has NO _mujoco kernel; omit
        # in mjx mode so a caller hitting it gets a clean KeyError, not a
        # missing-symbol crash.
        fns["id_wrt_params"] = IDWrtParamsFn
        fns["fd_wrt_params"] = FDWrtParamsFn
    return fns


# ─── CUDA-Graphs callable ───────────────────────────────────────────────────


class GraphCallable:
    """A CUDA-Graphs-captured op for fixed-batch replay.

    ``static_in`` are the captured input tensors (``.copy_()`` new data in);
    ``static_out`` is the captured output; ``replay()`` re-runs the graph.

    Two usage modes, with different costs:

    * ``replay()`` — re-runs the captured graph on the CURRENT contents of
      ``static_in`` (write new data into those buffers in place, or reuse the
      captured inputs). One ``cudaGraphLaunch``: CPU submission cost collapses
      to ~2 us regardless of how many kernels/memcpys were captured.
    * ``__call__(*inputs)`` — convenience: copies each input D->D into
      ``static_in`` first, then replays. The copy-in adds one fused foreach
      launch (or one per tensor on older torch) of REAL GPU + CPU work per
      call, so for identical repeated inputs prefer ``replay()``.

    Performance note: a CUDA graph removes CPU launch overhead; it does not
    speed up the kernels themselves. If the captured op is GPU-execution-bound
    (large batch and/or a heavy kernel), replay wall time ~= eager wall time
    and the win is the freed CPU time (submission drops from ~10-15 us of
    per-op dispatch to ~2 us), which matters when the python thread has other
    work (RL/training loops) or when many small ops are captured together.
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
        # Fused copy-in: one dispatcher hop for all inputs (falls back to a
        # per-tensor loop on torch builds without _foreach_copy_).
        foreach = getattr(self._torch, "_foreach_copy_", None)
        if foreach is not None:
            foreach(self.static_in, list(inputs))
        else:
            for dst, src in zip(self.static_in, inputs):
                dst.copy_(src)
        return self.replay()


# ─── mjx view ────────────────────────────────────────────────────────────────


class _TorchMujocoView:
    """MuJoCo-native, autograd-aware view over a :class:`TorchRobotHandle`
    (``handle.mujoco``). Mirrors the jax handle's ``.mujoco`` view: MuJoCo
    parameter names, the mjx output convention applied PER CALL (forwards an
    explicit ``_convention="mujoco"``, never mutating the shared default), so it
    is safe alongside pinocchio-convention calls on the same handle. The
    differentiable methods (inverse_dynamics/forward_dynamics/aba/integrator)
    stay autograd-aware (the backward uses the mjx-convention analytic Jacobian)."""

    __slots__ = ("_h",)

    def __init__(self, handle: "TorchRobotHandle") -> None:
        self._h = handle

    # ── value / dynamics ──────────────────────────────────────────────────
    def inverse_dynamics(self, qpos, qvel, qacc=None, *, gravity: float = -9.81, f_ext=None):
        """RNEA in MuJoCo convention: τ = id(qpos, qvel, qacc). Returns mjx-frame τ."""
        return self._h.inverse_dynamics(qpos, qvel, qacc, gravity=gravity, f_ext=f_ext,
                                        _convention="mujoco")

    def forward_dynamics(self, qpos, qvel, qfrc, *, gravity: float = -9.81, f_ext=None):
        """Forward dynamics in MuJoCo convention: qacc = fd(qpos, qvel, qfrc)."""
        return self._h.forward_dynamics(qpos, qvel, qfrc, gravity=gravity, f_ext=f_ext,
                                        _convention="mujoco")

    def aba(self, qpos, qvel, qfrc, *, gravity: float = -9.81, f_ext=None):
        """Articulated-body forward dynamics in MuJoCo convention."""
        return self._h.aba(qpos, qvel, qfrc, gravity=gravity, f_ext=f_ext, _convention="mujoco")

    def crba(self, qpos, *, gravity: float = -9.81):
        """Mass matrix M(qpos) in the mjx frame (G M G^T)."""
        return self._h.crba(qpos, gravity=gravity, _convention="mujoco")

    def minv(self, qpos):
        """Inverse mass matrix Minv(qpos) in the mjx frame (G^-T Minv G^-1)."""
        return self._h.minv(qpos, _convention="mujoco")

    # ── kinematics / regressor ────────────────────────────────────────────
    def end_effector_pose(self, qpos):
        """End-effector pose from mjx-convention qpos."""
        return self._h.end_effector_pose(qpos, _convention="mujoco")

    def end_effector_pose_gradient(self, qpos):
        """EE pose Jacobian reframed to the mjx free-joint tangent (J·G^-1)."""
        return self._h.end_effector_pose_gradient(qpos, _convention="mujoco")

    def end_effector_pose_hessian(self, qpos):
        """EE pose Hessian in the mjx convention."""
        return self._h.end_effector_pose_hessian(qpos, _convention="mujoco")

    def inverse_dynamics_regressor(self, qpos, qvel, qacc=None, *, gravity: float = -9.81):
        """Joint-torque regressor with base-linear rows in the mjx frame."""
        return self._h.inverse_dynamics_regressor(qpos, qvel, qacc, gravity=gravity,
                                                  _convention="mujoco")

    # ── first / second-order derivatives ──────────────────────────────────
    def inverse_dynamics_gradient(self, qpos, qvel, qacc=None, *, gravity: float = -9.81):
        """∂τ/∂(q,qd) in the mjx convention."""
        return self._h.inverse_dynamics_gradient(qpos, qvel, qacc, gravity=gravity,
                                                 _convention="mujoco")

    def forward_dynamics_gradient(self, qpos, qvel, qfrc, *, gravity: float = -9.81):
        """∂qacc/∂(q,qd) in the mjx convention."""
        return self._h.forward_dynamics_gradient(qpos, qvel, qfrc, gravity=gravity,
                                                 _convention="mujoco")

    def idsva_so(self, qpos, qvel, qacc=None, *, gravity: float = -9.81):
        """Second-order inverse dynamics (4 tensors) in the mjx convention."""
        return self._h.idsva_so(qpos, qvel, qacc, gravity=gravity, _convention="mujoco")

    def fdsva_so(self, qpos, qvel, qfrc, *, gravity: float = -9.81):
        """Second-order forward dynamics (4 tensors) in the mjx convention."""
        return self._h.fdsva_so(qpos, qvel, qfrc, gravity=gravity, _convention="mujoco")

    # ── integrator / plant ────────────────────────────────────────────────
    def integrator(self, qpos, qvel, qfrc, dt, *, integrator_type: str = "euler",
                   gravity: float = -9.81):
        """One integration step in the mjx convention (global-additive retract)."""
        return self._h.integrator(qpos, qvel, qfrc, dt, integrator_type=integrator_type,
                                  gravity=gravity, _convention="mujoco")

    def integrator_gradient(self, qpos, qvel, qfrc, dt, *, integrator_type: str = "euler",
                            gravity: float = -9.81):
        """Integrator state-transition Jacobian in the mjx convention."""
        return self._h.integrator_gradient(qpos, qvel, qfrc, dt, integrator_type=integrator_type,
                                           gravity=gravity, _convention="mujoco")

    def plant_step(self, x, u, dt, *, integrator_type: str = "euler", gravity: float = -9.81):
        """Plant step x_{k+1} in the mjx convention."""
        return self._h.plant_step(x, u, dt, integrator_type=integrator_type, gravity=gravity,
                                  _convention="mujoco")

    def plant_step_gradient(self, x, u, dt, *, integrator_type: str = "euler", gravity: float = -9.81):
        """Plant-step state-transition Jacobian in the mjx convention."""
        return self._h.plant_step_gradient(x, u, dt, integrator_type=integrator_type,
                                           gravity=gravity, _convention="mujoco")

    def quadratic_state_cost(self, x, x_des, Q):
        """Quadratic state cost (value/grad/GN-hess) in the mjx convention."""
        return self._h.quadratic_state_cost(x, x_des, Q, _convention="mujoco")

    def ee_pos_cost(self, qpos, p_des, W):
        """End-effector position tracking cost in the mjx convention."""
        return self._h.ee_pos_cost(qpos, p_des, W, _convention="mujoco")

    def com_cost(self, qpos, p_des, W):
        """Center-of-mass tracking cost in the mjx convention."""
        return self._h.com_cost(qpos, p_des, W, _convention="mujoco")

    def momentum_cost(self, qpos, qvel, h_des, W):
        """Centroidal-momentum tracking cost in the mjx convention."""
        return self._h.momentum_cost(qpos, qvel, h_des, W, _convention="mujoco")

    def __repr__(self) -> str:
        return f"<mujoco view of {self._h!r}>"


# ─── TorchRobotHandle ───────────────────────────────────────────────────────


class TorchRobotHandle:
    """Torch-flavored wrapper. Methods return ``torch.Tensor`` (autograd-aware
    for inverse_dynamics / forward_dynamics / aba / integrator)."""

    def __init__(self, base: RobotHandle, cache_key: str, so_path: str,
                 output_convention: str = "pinocchio"):
        self._base = base
        self._cache_key = cache_key
        self._so_path = Path(so_path)
        self._ns = _load_ops(self._so_path, cache_key)
        import torch
        self._ops = getattr(torch.ops, self._ns)
        # Per-convention registry of autograd Functions, built lazily. mjx mode
        # binds every op to its _mujoco variant (floating-base only).
        self._fns_cache: dict[str, dict] = {}
        if output_convention not in ("pinocchio", "mujoco"):
            raise ValueError(
                f"output_convention must be 'pinocchio' or 'mujoco'; got {output_convention!r}")
        self._output_convention = output_convention
        self._mjx_view = None

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
    def num_bodies(self) -> int:  return self._base.num_bodies
    @property
    def floating_base(self) -> bool: return self._base.floating_base
    @property
    def max_batch(self) -> int:   return self._base.max_batch
    @property
    def dtype(self) -> str:
        """Compute precision of the underlying .so (``"float32"`` / ``"float64"``).
        Tensors passed to the ops must match this dtype (checked in the op)."""
        return self._base.dtype

    # ─── runtime-mutable inertia (D.4 / sysID) ───────────────────────────
    @property
    def runtime_inertia(self) -> bool:
        """True if this robot was registered with ``runtime_inertia=True`` (the
        .so carries a mutable inertia table + :py:meth:`set_inertia_params`)."""
        return self._base.runtime_inertia

    @property
    def inertia_params(self):
        """The BAKED 10-param-per-body inertia table, shape ``(num_bodies, 10)``
        (``[m, h(3), I_O(6)]`` per body). Fetch, mutate, and pass to
        :py:meth:`set_inertia_params`. Only on a ``runtime_inertia`` build."""
        return self._base.inertia_params

    def set_inertia_params(self, params) -> None:
        """Update the device-resident inertia table at runtime (no recompile).
        ``params`` is the ``(num_bodies, 10)`` (or flat ``10*num_bodies``) table in
        the same basis as :py:attr:`inertia_params`. Only valid on a robot
        registered with ``runtime_inertia=True``."""
        self._base.set_inertia_params(params)

    # ─── welded tool / payload (attach_tool) ─────────────────────────────
    def attach_tool(self, joint, *, mass, com=(0.0, 0.0, 0.0), inertia=None,
                    tip_transform=None):
        """Weld a rigid tool/payload at runtime (no recompile). Delegates to the base
        handle: composes the payload inertia into ``joint``'s child link and stores an
        optional SE(3) ``tip_transform``. See :py:meth:`grid_rbd.RobotHandle.attach_tool`.
        The inertia change is seen by every surface (numpy/jax/torch share the .so); for
        the tool-tip frame, pass ``ee_offsets=[tip_transform]`` to
        :py:meth:`end_effector_pose_runtime`."""
        return self._base.attach_tool(joint, mass=mass, com=com, inertia=inertia,
                                      tip_transform=tip_transform)

    def detach_tool(self):
        """Remove the attached tool (restore baked inertia). See
        :py:meth:`grid_rbd.RobotHandle.detach_tool`."""
        self._base.detach_tool()

    @property
    def tool(self):
        """The currently attached tool dict or ``None``."""
        return self._base.tool

    def tool_fext(self, q, wrench, *, joint=None, offset=None):
        """World-aligned tool-tip wrench -> joint-local f_ext ``(B, 6*num_bodies)``,
        ready to pass as ``f_ext=`` to the dynamics. Delegates to the base handle
        (returns a numpy array). See :py:meth:`grid_rbd.RobotHandle.tool_fext`."""
        return self._base.tool_fext(q, wrench, joint=joint, offset=offset)

    # ─── runtime-mutable joint-frame transform (runtime_transform) ───────
    @property
    def runtime_transform(self) -> bool:
        """True if registered with ``runtime_transform=True`` (mutable joint-origin
        table + :py:meth:`set_transform_params`). The torch-op kernels read the same
        device-resident table, so a poke through any surface is seen here."""
        return self._base.runtime_transform

    @property
    def transform_params(self):
        """The BAKED per-joint origin table, shape ``(num_joints, 6)``
        (``[x, y, z, r, p, y]`` per joint). Fetch, mutate, and pass to
        :py:meth:`set_transform_params`. Only on a ``runtime_transform`` build."""
        return self._base.transform_params

    def set_transform_params(self, params) -> None:
        """Update the device-resident joint-origin table at runtime (no recompile).
        ``params`` is the ``(num_joints, 6)`` (or flat ``6*num_joints``) table in the
        same basis as :py:attr:`transform_params`. Only valid on a robot registered
        with ``runtime_transform=True``."""
        self._base.set_transform_params(params)

    # ─── runtime-mutable joint dynamics (C5 / sysID) ─────────────────────
    @property
    def runtime_joint_dynamics(self) -> bool:
        """True if registered with ``runtime_joint_dynamics=True`` (mutable
        damping/friction table + :py:meth:`set_joint_dynamics`). The torch-op kernels
        read the same device-resident table, so a poke through any surface is seen
        here."""
        return self._base.runtime_joint_dynamics

    @property
    def joint_damping(self):
        """Baked per-v-slot viscous damping (length nv). Only on a
        ``runtime_joint_dynamics`` build."""
        return self._base.joint_damping

    @property
    def joint_friction(self):
        """Baked per-v-slot Coulomb friction (length nv). Only on a
        ``runtime_joint_dynamics`` build."""
        return self._base.joint_friction

    def set_joint_dynamics(self, damping=None, friction=None) -> None:
        """Update the device-resident damping/friction table at runtime (no
        recompile). ``damping``/``friction`` are length-nv (v-slot indexed); an
        omitted side keeps its baked value. Only on a ``runtime_joint_dynamics``
        build."""
        self._base.set_joint_dynamics(damping=damping, friction=friction)

    # ─── kernel launch configuration ─────────────────────────────────────
    @property
    def max_perf_level_threads(self) -> int:
        """Codegen-time thread-count hint (DOF-aware, warp-rounded); the default
        per-block thread count for kernel launches."""
        return self._base.max_perf_level_threads

    @property
    def threads_per_block(self) -> int:
        """Active global threads-per-block override (``-1`` = per-algo autotuned
        default baked into ``grid.cuh``; ``>= 1`` = a forced global override)."""
        return self._base.threads_per_block

    def set_threads_per_block(self, n: int) -> None:
        """Force a single global per-block thread count for all subsequent kernel
        launches issued through the underlying .so. By default each algorithm uses
        its own autotuned per-algo thread count; ``n >= 1`` overrides that."""
        self._base.set_threads_per_block(n)

    def kernel_max_threads(self, algo: str) -> int:
        """Real compiled ``__launch_bounds__`` ceiling for the short autotune key
        (E1 tier-contract introspection); ``-1`` when unknown/not-built. Delegates
        to the base handle — see ``RobotHandle.kernel_max_threads``."""
        return self._base.kernel_max_threads(algo)

    # ─── output convention (mjx parity) ──────────────────────────────────
    @property
    def output_convention(self) -> str:
        """Default IO convention for this handle: ``"pinocchio"`` or ``"mujoco"``.
        Settable. ``"mujoco"`` requires a floating base (mjx and pinocchio coincide
        on a fixed base). Per-call overrides use the thread-safe ``.mujoco`` view."""
        return self._output_convention

    @output_convention.setter
    def output_convention(self, value: str) -> None:
        if value not in ("pinocchio", "mujoco"):
            raise ValueError(
                f"output_convention must be 'pinocchio' or 'mujoco'; got {value!r}")
        if value == "mujoco" and not self.floating_base:
            raise ValueError(
                "output_convention='mujoco' requires a floating-base robot "
                f"({self.name} is fixed-base)")
        self._output_convention = value

    def _resolve_convention(self, convention):
        """None → the handle default; else the explicit per-call convention."""
        return self._output_convention if convention is None else convention

    def _fns_for(self, convention):
        """Per-convention autograd Function registry (lazily built + cached).
        ``convention="mujoco"`` builds the SAME closures but every op is the
        ``_mujoco`` variant; mjx is floating-base only."""
        conv = self._resolve_convention(convention)
        if conv not in ("pinocchio", "mujoco"):
            raise ValueError(
                f"output_convention must be 'pinocchio' or 'mujoco'; got {conv!r}")
        if conv == "mujoco" and not self.floating_base:
            raise ValueError(
                "output_convention='mujoco' requires a floating-base robot "
                f"({self.name} is fixed-base; mjx and pinocchio coincide there)")
        cache = self._fns_cache
        if conv not in cache:
            cache[conv] = _make_autograd(self._ns, self._base.num_vel,
                                         mujoco=(conv == "mujoco"))
        return cache[conv]

    @property
    def _fns(self):
        """The pinocchio-convention autograd registry (existing pin call sites)."""
        return self._fns_for("pinocchio")

    def _op(self, conv, name):
        """Resolve a DIRECT (non-autograd) CORE op, dispatching to the ``_mujoco``
        variant when the resolved convention is mujoco. mjx requires a floating
        base (the _mujoco symbol is #ifdef'd out of fixed .so). A core op missing
        on a SUBSET build maps to the clean "not built — add to algorithm_list"
        error (via _resolve_core_op)."""
        c = self._resolve_convention(conv)
        if c == "mujoco":
            if not self.floating_base:
                raise ValueError(
                    "output_convention='mujoco' requires a floating-base robot "
                    f"({self.name} is fixed-base)")
            return _resolve_core_op(self._ops, name + "_mujoco")
        return _resolve_core_op(self._ops, name)

    def _gated_op(self, conv, name):
        """Like :py:meth:`_op` but for the GATED centroidal / kinematics methods
        (com / ccrba / energy / dccrba / cmm_time_variation / frame_jacobian[_dot]
        / osc_inertia): if the op isn't in the .so (the kernel wasn't generated
        for this robot) re-raise with an actionable message mirroring the numpy
        handle. These are #ifdef-gated (emitted-when-present), distinct from the
        core subset gating, so resolve the raw op directly and keep the dedicated
        'not generated' message."""
        c = self._resolve_convention(conv)
        if c == "mujoco" and not self.floating_base:
            raise ValueError(
                "output_convention='mujoco' requires a floating-base robot "
                f"({self.name} is fixed-base)")
        resolved = (name + "_mujoco") if c == "mujoco" else name
        try:
            return getattr(self._ops, resolved)
        except AttributeError as e:
            raise AttributeError(
                f"{name} not generated for this robot .so; re-register with "
                f"force_rebuild=True (and ensure the kernel is enabled in codegen)."
            ) from e

    @property
    def mujoco(self) -> "_TorchMujocoView":
        """MuJoCo-native view (``handle.mujoco.inverse_dynamics(qpos, qvel, qacc)``):
        forwards a per-call ``_convention="mujoco"`` WITHOUT mutating the shared
        ``output_convention`` default, so it is safe alongside pinocchio calls."""
        v = self._mjx_view
        if v is None:
            v = self._mjx_view = _TorchMujocoView(self)
        return v

    # ─── differentiable algorithms ───────────────────────────────────────

    def inverse_dynamics(self, q, qd, qdd=None, *, gravity: float = -9.81, f_ext=None,
                         _convention=None):
        """Inverse dynamics (RNEA) τ = M·qdd + h − g (B, NJ). Autograd-aware wrt (q, qd).

        ``qdd`` (optional): joint acceleration, CUDA float32 ``(B, NJ)``. With
        ``qdd=None`` (default) returns the bias c = h − g; a nonzero ``qdd`` adds
        the M·qdd inertial term (USE_QDD overload). The autograd backward threads
        the saved qdd through, so the q/qd Jacobian includes ∂(M·qdd)/∂q for a
        nonzero-qdd call; qdd itself is not differentiated.

        ``f_ext`` (optional): per-body external forces, a CUDA float32 tensor
        ``(B, 6*num_bodies)``, body-major, each ``[angular; linear]`` in the
        body's local frame (subtracted from the per-body force; matches the
        numpy handle and ``RBDReference.inverse_dynamics(..., f_ext=...)``).

        With ``output_convention="mujoco"`` (floating base) inputs/outputs are
        MuJoCo-convention and the autograd VJP uses the mjx-convention Jacobian."""
        return self._fns_for(_convention)["inverse_dynamics"].apply(
            q, qd, float(gravity), qdd, f_ext)

    def forward_dynamics(self, q, qd, u, *, gravity: float = -9.81, f_ext=None,
                         _convention=None):
        """qdd = M⁻¹(τ − c) (B, NJ). Autograd-aware wrt (q, qd, u).

        ``f_ext`` (optional): per-body external forces ``(B, 6*num_bodies)``
        CUDA float32 (see :py:meth:`inverse_dynamics`).

        With ``output_convention="mujoco"`` (floating base) inputs/outputs are
        MuJoCo-convention and the autograd VJP uses the mjx-convention Jacobian."""
        return self._fns_for(_convention)["fd"].apply(q, qd, u, float(gravity), f_ext)

    def aba(self, q, qd, u, *, gravity: float = -9.81, f_ext=None, _convention=None):
        """qdd via ABA (B, NJ). Autograd-aware wrt (q, qd, u).

        ``f_ext`` (optional): per-body external forces ``(B, 6*num_bodies)``
        CUDA float32 (see :py:meth:`inverse_dynamics`).

        With ``output_convention="mujoco"`` (floating base) inputs/outputs are
        MuJoCo-convention."""
        return self._fns_for(_convention)["aba"].apply(q, qd, u, float(gravity), f_ext)

    def inverse_dynamics_wrt_params(self, q, qd, params, *, gravity: float = -9.81, f_ext=None):
        """Inverse-dynamics bias c = ID(q, qd, qdd=0) (B, NJ), differentiable wrt
        the per-link inertial parameters ``params`` (π) AND ``q``/``qd``.

        ``params``: (B, 10*num_bodies) — per-link [m, m*c(3), I_O(6)] in the
        parser's origin-frame basis (same as ``inverse_dynamics_regressor`` Y and
        ``RBDReference._regressor``). The FORWARD value is independent of
        ``params`` (the compiled ``.so`` carries the baked-in inertia); the op
        exists so ``torch.autograd`` flows the analytic ``∂c/∂π = Y(q,qd,qdd=0)``
        (regressor) to ``params`` — the outer-loop system-ID gradient.
        ``q``/``qd`` gradients are unchanged. Mirrors the JAX
        ``inverse_dynamics_wrt_params`` custom_vjp."""
        return self._fns["id_wrt_params"].apply(q, qd, params, float(gravity), f_ext)

    def forward_dynamics_wrt_params(self, q, qd, u, params, *, gravity: float = -9.81, f_ext=None):
        """Forward dynamics qdd = FD(q, qd, u) (B, NJ), differentiable wrt the
        per-link inertial parameters ``params`` (π) AND ``q``/``qd``/``u``.

        ``params``: (B, 10*num_bodies) — see :py:meth:`inverse_dynamics_wrt_params`.
        The forward value is independent of ``params`` (baked-in inertia); the VJP
        flows the analytic ``∂qdd/∂π = -M⁻¹·Y`` (the
        ``forward_dynamics_parameter_gradient`` kernel) to ``params``.
        ``q``/``qd``/``u`` gradients are unchanged. Mirrors the JAX
        ``forward_dynamics_wrt_params`` custom_vjp."""
        return self._fns["fd_wrt_params"].apply(q, qd, u, params, float(gravity), f_ext)

    def integrator(self, q, qd, u, dt, *, integrator_type: str = "euler", gravity: float = -9.81,
                   _convention=None):
        """x_{k+1} (B, NP+NV). Autograd-aware wrt (q, qd, u).

        With ``output_convention="mujoco"`` (floating base) inputs/outputs are
        MuJoCo-convention (global-additive base retract + reframed base velocity)."""
        it = _integrator_code(integrator_type)
        return self._fns_for(_convention)["integrator"].apply(
            q, qd, u, float(dt), it, float(gravity))

    # ─── forward-only algorithms (raw kernel ops; reshapes mirror _handle) ──

    def minv(self, q, *, _convention=None):
        """Minv(q) (B, NV, NV), symmetrized (kernel writes lower triangle).

        Tangent-space (pinocchio) inverse mass matrix. FIXED base: NV == NJ
        (shape unchanged); FLOATING base: NV < NJ.

        With ``output_convention="mujoco"`` (floating base) the returned Minv is the
        mjx-frame inverse mass matrix (G^-T Minv G^-1); the ``minv_mujoco`` kernel
        writes it FULL DENSE so no host symmetrize is applied."""
        import torch
        conv = self._resolve_convention(_convention)
        nv = self.num_vel
        m = self._op(conv, "minv")(q).reshape(-1, nv, nv)
        if conv == "mujoco":
            return m  # mjx kernel writes a full dense symmetric matrix
        eye = torch.eye(nv, dtype=m.dtype, device=m.device)
        return m + m.transpose(1, 2) - m * eye

    def crba(self, q, *, gravity: float = -9.81, _convention=None):
        """Mass matrix M(q) (B, NV, NV), tangent-space (pinocchio) convention.
        FIXED base: NV == NJ (unchanged); FLOATING base: NV < NJ.

        With ``output_convention="mujoco"`` (floating base) the returned M is the
        mjx-frame mass matrix (G M G^T congruence, written full dense)."""
        nv = self.num_vel
        return self._op(_convention, "crba")(q, float(gravity)).reshape(-1, nv, nv)

    # ─── centroidal / energy / kinematics value methods (numpy-handle parity) ──
    #
    # Mirror the numpy RobotHandle's centroidal / energy / regressor / frame
    # methods exactly (same torch op names, same gravity/int args, same output
    # reshape copied VERBATIM from `_handle.py`). Forward-only (no autograd).
    # The GATED ops (com / ccrba / energy / dccrba / cmm_time_variation /
    # frame_jacobian[_dot] / osc_inertia) route through `_gated_op` for an
    # actionable "not generated for this robot" message; the always-emitted ones
    # (generalized_gravity / nonlinear_effects / coriolis_matrix / *_regressor)
    # use `_op`.

    def generalized_gravity(self, q, *, gravity: float = -9.81, _convention=None):
        """Generalized gravity torque g(q) = RNEA(q, 0, 0). Returns (B, NV).

        With ``output_convention="mujoco"`` (floating base) ``q`` is MuJoCo-convention
        and the returned g is in the mjx frame (base rows rotated in-kernel)."""
        return self._op(_convention, "generalized_gravity")(q, float(gravity))

    def nonlinear_effects(self, q, qd, *, gravity: float = -9.81, _convention=None):
        """Nonlinear (bias) effects c(q,qd) = RNEA(q, qd, 0). Returns (B, NV).

        With ``output_convention="mujoco"`` (floating base) ``q``/``qd`` are
        MuJoCo-convention and the returned bias matches MuJoCo's ``qfrc_bias``."""
        return self._op(_convention, "nonlinear_effects")(q, qd, float(gravity))

    def energy(self, q, qd, *, gravity: float = -9.81, _convention=None):
        """Kinetic / potential / mechanical energy. Returns (B, 3) = [KE, PE, KE+PE].

        With ``output_convention="mujoco"`` (floating base) ``q``/``qd`` are
        MuJoCo-convention; the energies are frame-INVARIANT (inputs converted in-kernel)."""
        return self._gated_op(_convention, "energy")(q, qd, float(gravity))

    def com(self, q, *, _convention=None):
        """Center-of-mass world position p_com and CoM Jacobian J_com.

        Returns ``(p_com, J_com)`` where ``p_com`` is ``(B, 3)`` and ``J_com`` is
        ``(B, 3, NV)``. The op returns one flat ``(B, 3 + 3*NV)`` buffer
        ``[p_com(3); J_com(3 x NV col-major)]`` (split mirrors the numpy handle).

        With ``output_convention="mujoco"`` (floating base) ``p_com`` is invariant
        and the ``J_com`` columns are reframed (computed in-kernel)."""
        nv = self.num_vel
        raw = self._gated_op(_convention, "com")(q)
        B = raw.shape[0]
        p_com = raw[:, :3]
        # J_com stored column-major (3 x NV): J[r + 3*c]; recover (B, 3, NV).
        j_com = raw[:, 3:].reshape(B, nv, 3).permute(0, 2, 1)
        return p_com, j_com

    def ccrba(self, q, qd, *, _convention=None):
        """Centroidal momentum matrix A (6 x NV) and momentum h = A·qd (6,).

        Returns ``(A, h)`` where ``A`` is ``(B, 6, NV)`` and ``h`` is ``(B, 6)``.
        The op returns one flat ``(B, 6*NV + 6)`` buffer ``[A(6 x NV col-major); h(6)]``.

        With ``output_convention="mujoco"`` (floating base) ``h`` is invariant and
        the ``A`` columns are reframed (computed in-kernel)."""
        nv = self.num_vel
        raw = self._gated_op(_convention, "ccrba")(q, qd)
        B = raw.shape[0]
        A = raw[:, : 6 * nv].reshape(B, nv, 6).permute(0, 2, 1)
        h = raw[:, 6 * nv:]
        return A, h

    def dccrba(self, q, *, _convention=None):
        """dCCRBA tensor ∂A/∂q, shape ``(B, 6, NV, NV)`` indexed
        ``[:, :, k, i] = ∂A[:, k]/∂q_i`` (Pinocchio centroidal convention).

        With ``output_convention="mujoco"`` (floating base) ``q`` is MuJoCo-convention
        and the returned tensor is the dA/dq of the mjx CMM (same layout)."""
        nv = self.num_vel
        raw = self._gated_op(_convention, "dccrba")(q)
        B = raw.shape[0]
        # flat layout dA[row + 6*k + 6*NV*m] -> (B, m, k, row) then -> (B, row, k, m).
        return raw.reshape(B, nv, nv, 6).permute(0, 3, 2, 1)

    def cmm_time_variation(self, q, qd, *, _convention=None):
        """Centroidal-momentum-matrix time variation Ȧ = dA(q(t))/dt, shape
        ``(B, 6, NV)`` = ``Σ_i (∂A/∂q_i)·qd_i``.

        With ``output_convention="mujoco"`` (floating base) ``q``/``qd`` are
        MuJoCo-convention and Ȧ has its columns reframed (computed in-kernel)."""
        nv = self.num_vel
        raw = self._gated_op(_convention, "cmm_time_variation")(q, qd)
        B = raw.shape[0]
        # (B, 6*NV) col-major A[r + 6*c] -> (B, NV, 6) -> (B, 6, NV).
        return raw.reshape(B, nv, 6).permute(0, 2, 1)

    def coriolis_matrix(self, q, qd, *, gravity: float = -9.81, _convention=None):
        """Coriolis matrix C(q,qd). Returns ``(B, NV, NV)`` row-major, with
        ``C·qd + g(q) = nonlinear_effects(q, qd)`` (gravity unused by C; the
        kwarg mirrors the host wrapper signature).

        With ``output_convention="mujoco"`` (floating base) ``q``/``qd`` are
        MuJoCo-convention and the returned ``C`` is the mjx-frame Coriolis matrix."""
        nv = self.num_vel
        raw = self._op(_convention, "coriolis_matrix")(q, qd, float(gravity))
        return raw.reshape(-1, nv, nv)  # row-major

    def kinetic_energy_regressor(self, q, qd, *, gravity: float = -9.81, _convention=None):
        """Kinetic-energy regressor y_KE, length ``10*num_bodies``, with
        ``KE = y_KE · π``. Returns ``(B, 10*num_bodies)`` (gravity unused).

        With ``output_convention="mujoco"`` (floating base) ``q``/``qd`` are
        MuJoCo-convention; the regressor is frame-INVARIANT (inputs converted in-kernel)."""
        return self._op(_convention, "kinetic_energy_regressor")(q, qd, float(gravity))

    def potential_energy_regressor(self, q, *, gravity: float = -9.81, _convention=None):
        """Potential-energy regressor y_PE, length ``10*num_bodies``, with
        ``PE = y_PE · π`` (PE uses ``gravity``). Returns ``(B, 10*num_bodies)``.

        With ``output_convention="mujoco"`` (floating base) ``q`` is MuJoCo-convention;
        the regressor is frame-INVARIANT (input converted in-kernel)."""
        return self._op(_convention, "potential_energy_regressor")(q, float(gravity))

    def frame_jacobian(self, q, *, target_jid=None, reference_frame=None, _convention=None):
        """Geometric Jacobian (6 x NV, ``[linear; angular]``) of a frame.
        Returns ``(B, 6, NV)``.

        ``target_jid`` selects the frame's joint id (default: the leaf
        end-effector joint baked at codegen time). ``reference_frame`` is
        ``'LOCAL'`` (0), ``'WORLD'`` (1), or ``'LOCAL_WORLD_ALIGNED'`` (2, the
        default), or the equivalent int — passed as int op args.

        With ``output_convention="mujoco"`` (floating base) ``q`` is MuJoCo-convention
        and the returned Jacobian is column-reframed ``J G⁻¹`` (mjx frame)."""
        from .._handle import _resolve_frame_args
        nv = self.num_vel
        tj, rf = _resolve_frame_args(self._base._meta, target_jid, reference_frame)
        raw = self._gated_op(_convention, "frame_jacobian")(q, int(tj), int(rf))
        B = raw.shape[0]
        # (B, 6*NV) col-major J[r + 6*c] -> (B, NV, 6) -> (B, 6, NV).
        return raw.reshape(B, nv, 6).permute(0, 2, 1)

    def frame_jacobian_dot(self, q, qd, *, target_jid=None, reference_frame=None, _convention=None):
        """Time derivative Jdot of :py:meth:`frame_jacobian` along v = qd
        (6 x NV, ``[linear; angular]``). Returns ``(B, 6, NV)``.

        ``target_jid`` / ``reference_frame`` are runtime int op args (default:
        leaf-EE joint / ``LOCAL_WORLD_ALIGNED``); see :py:meth:`frame_jacobian`.

        With ``output_convention="mujoco"`` (floating base) ``q``/``qd`` are
        MuJoCo-convention and Jdot is column-reframed (mjx frame)."""
        from .._handle import _resolve_frame_args
        nv = self.num_vel
        tj, rf = _resolve_frame_args(self._base._meta, target_jid, reference_frame)
        raw = self._gated_op(_convention, "frame_jacobian_dot")(q, qd, int(tj), int(rf))
        B = raw.shape[0]
        return raw.reshape(B, nv, 6).permute(0, 2, 1)

    def osc_inertia(self, q, *, _convention=None):
        """Operational-space (task) inertia Lambda = (J·M⁻¹·Jᵀ)⁻¹ (6 x 6) for
        the leaf-EE frame (LWA). Returns ``(B, 6, 6)``.

        With ``output_convention="mujoco"`` the MuJoCo ``q`` is reordered before the
        kinematics build (Lambda is otherwise frame-INVARIANT)."""
        return self._gated_op(_convention, "osc_inertia")(q).reshape(-1, 6, 6)

    def end_effector_pose(self, q, *, _convention=None):
        """EE pose [xyz, rpy] per EE (B, 6*NUM_EES).

        With ``output_convention="mujoco"`` (floating base) ``q`` is MuJoCo-convention."""
        return self._op(_convention, "end_effector_pose")(q)

    def end_effector_pose_gradient(self, q, *, _convention=None):
        """EE pose Jacobian d/dv (B, 6*NEE, NV), pinocchio tangent convention.

        With ``output_convention="mujoco"`` (floating base) the base-velocity
        Jacobian columns are reframed to the mjx free-joint tangent (J·G^-1)."""
        nee, nv = self.num_ees, self.num_vel
        raw = self._op(_convention, "end_effector_pose_gradient")(q)
        B = raw.shape[0]
        return raw.reshape(B, nee, nv, 6).permute(0, 1, 3, 2).reshape(B, 6 * nee, nv)

    def end_effector_pose_hessian(self, q, *, _convention=None):
        """EE pose Hessian d²/dv² (B, 6*NEE, NV, NV).

        With ``output_convention="mujoco"`` (floating base) the base-tangent indices
        are reframed to the mjx free-joint convention."""
        nee, nv = self.num_ees, self.num_vel
        return self._op(_convention, "end_effector_pose_hessian")(q).reshape(-1, 6 * nee, nv, nv)

    def end_effector_pose_runtime(self, q, ee_joint_names=None, ee_offsets=None,
                                  *, _convention=None):
        """Runtime-target multi-EE pose ``[xyz; rpy]`` at an offset point.

        Mirrors :py:meth:`grid_rbd.RobotHandle.end_effector_pose_runtime`:
        ``ee_joint_names`` (None => all leaf joints, or a str / list of joint
        names) selects the EE frames; ``ee_offsets`` (None => frame origin, or one
        ``[x,y,z]`` / ``[x,y,z,1]`` per EE) shifts the measurement point. The
        single-target op is looped over the resolved jid list and stacked.

        Returns ``(B, NUM_EE, 6)``. Forward-only (no autograd).

        With ``output_convention="mujoco"`` (floating base) ``q`` is MuJoCo-convention;
        the world pose VALUE is frame-invariant."""
        import torch
        jids = self._base._resolve_ee_jids(ee_joint_names)
        offsets = self._base._normalize_ee_offsets(ee_offsets, len(jids))
        op = self._gated_op(_convention, "end_effector_pose_runtime")
        per_ee = []
        for jid, off in zip(jids, offsets):
            # off is the 16-float col-major X_tool from _normalize_ee_offsets;
            # the op wants ALL 16 (translation lives at [12..14], not [0..2]).
            off_t = torch.as_tensor(off, dtype=q.dtype, device=q.device).reshape(-1).contiguous()
            per_ee.append(op(q, int(jid), off_t))  # (B, 6)
        return torch.stack(per_ee, dim=1)  # (B, NUM_EE, 6)

    def end_effector_pose_gradient_runtime(self, q, ee_joint_names=None, ee_offsets=None,
                                           *, _convention=None):
        """Runtime-target multi-EE pose gradient ``d[xyz; rpy]/dv`` (6 x NV) at an
        offset point. Same ``ee_joint_names`` / ``ee_offsets`` semantics as
        :py:meth:`end_effector_pose_runtime`; mirrors
        :py:meth:`grid_rbd.RobotHandle.end_effector_pose_gradient_runtime`.

        The kernel writes column-major ``(B, 6*NV)`` per target; we reshape
        ``(B, NV, 6)`` then transpose to ``(B, 6, NV)`` and stack.

        Returns ``(B, NUM_EE, 6, NV)``. Forward-only (no autograd).

        With ``output_convention="mujoco"`` (floating base) the base-linear columns
        are reframed by R^T in-kernel (column-reframe class)."""
        import torch
        nv = self.num_vel
        jids = self._base._resolve_ee_jids(ee_joint_names)
        offsets = self._base._normalize_ee_offsets(ee_offsets, len(jids))
        op = self._gated_op(_convention, "end_effector_pose_gradient_runtime")
        per_ee = []
        for jid, off in zip(jids, offsets):
            off_t = torch.as_tensor(off, dtype=q.dtype, device=q.device).reshape(-1).contiguous()
            raw = op(q, int(jid), off_t)  # (B, 6*NV) col-major
            B = raw.shape[0]
            per_ee.append(raw.reshape(B, nv, 6).permute(0, 2, 1))  # (B, 6, NV)
        return torch.stack(per_ee, dim=1)  # (B, NUM_EE, 6, NV)

    def inverse_dynamics_gradient(self, q, qd, qdd=None, *, gravity: float = -9.81, f_ext=None,
                                  _convention=None):
        """∂c/∂(q,qd) (B, NV, 2*NV) = [dc_dq | dc_dqd], tangent-space (pinocchio)
        convention. FIXED base: NV == NJ (unchanged); FLOATING base: NV < NJ.

        ``qdd`` (optional): joint acceleration ``(B, NJ)``. With ``qdd=None``
        (default) this is the bias gradient ∂(h−g)/∂(q,qd); a nonzero ``qdd``
        adds ∂(M·qdd)/∂q (USE_QDD overload).

        ``f_ext`` (optional): per-body external forces ``(B, 6*num_bodies)``;
        affine in f_ext so a constant f_ext leaves this Jacobian unchanged.

        With ``output_convention="mujoco"`` (floating base) the gradient is the
        mjx-convention Jacobian (rows base-rotated, columns base-reframed)."""
        nv = self.num_vel
        raw = self._op(_convention, "inverse_dynamics_gradient")(q, qd, float(gravity), qdd, f_ext)
        B = raw.shape[0]
        blocks = raw.reshape(B, 2, nv, nv).transpose(2, 3)
        return _concat_blocks(blocks)

    def forward_dynamics_gradient(self, q, qd, u, *, gravity: float = -9.81, f_ext=None,
                                  _convention=None):
        """∂qdd/∂(q,qd) (B, NV, 2*NV), tangent-space (pinocchio) convention.
        FIXED base: NV == NJ (unchanged); FLOATING base: NV < NJ.

        ``f_ext`` (optional): per-body external forces ``(B, 6*num_bodies)``;
        affine in f_ext so a constant f_ext leaves this Jacobian unchanged.

        With ``output_convention="mujoco"`` (floating base) the gradient is the
        mjx-convention Jacobian."""
        nv = self.num_vel
        raw = self._op(_convention, "forward_dynamics_gradient")(q, qd, u, float(gravity), f_ext)
        B = raw.shape[0]
        blocks = raw.reshape(B, 2, nv, nv).transpose(2, 3)
        return _concat_blocks(blocks)

    def inverse_dynamics_regressor(self, q, qd, qdd=None, *, gravity: float = -9.81,
                                   _convention=None):
        """Joint-torque regressor Y with τ = Y·π (∂τ/∂π). Returns
        (B, NV, 10*num_bodies), row-major (NV, 10*NB) per sample. ``qdd=None`` ⇒
        zeros (the bias regressor used by :py:meth:`inverse_dynamics_wrt_params`).
        Per-link basis [m, m*c(3), I_O(6)].

        With ``output_convention="mujoco"`` (floating base) the base-linear rows are
        rotated to the mjx frame (same covector transform as the τ value)."""
        import torch
        nv, npar = self.num_vel, 10 * self.num_bodies
        if qdd is None:
            q = torch.as_tensor(q)
            qdd = torch.zeros_like(q)
        return self._op(_convention, "inverse_dynamics_regressor")(
            q, qd, qdd, float(gravity)).reshape(-1, nv, npar)

    def forward_dynamics_parameter_gradient(self, q, qd, u, *, gravity: float = -9.81):
        """FD inertial-parameter gradient ∂qdd/∂π = -M⁻¹·Y. Returns
        (B, NV, 10*num_bodies), row-major (NV, 10*NB) per sample."""
        nv, npar = self.num_vel, 10 * self.num_bodies
        return self._ops.forward_dynamics_parameter_gradient(q, qd, u, float(gravity)).reshape(-1, nv, npar)

    def idsva_so(self, q, qd, qdd=None, *, gravity: float = -9.81, _convention=None):
        """Second-order ID at joint acceleration ``qdd``. Returns a
        :class:`grid_rbd.SecondOrderID` NamedTuple of 4 tensors each
        (B, NV, NV, NV) (a plain tuple — positional unpacking / indexing work).

        ``qdd=None`` ⇒ zero acceleration (explicit zeros are passed so the
        result never depends on a stale device buffer from a prior call).

        With ``output_convention="mujoco"`` (floating base) all four tensors are in
        the mjx convention (the kernel transforms every slab)."""
        import torch
        nv = self.num_vel
        if qdd is None:
            q = torch.as_tensor(q)
            qdd = torch.zeros_like(q)
        flat = self._op(_convention, "idsva_so")(q, qd, qdd, float(gravity))
        B = flat.shape[0]
        return SecondOrderID(*(flat[:, i*nv**3:(i+1)*nv**3].reshape(B, nv, nv, nv) for i in range(4)))

    def fdsva_so(self, q, qd, u, *, gravity: float = -9.81, _convention=None):
        """Second-order FD. Returns a :class:`grid_rbd.SecondOrderFD` NamedTuple
        of 4 tensors each (B, NV, NV, NV) (a plain tuple, positional-compatible).

        With ``output_convention="mujoco"`` (floating base) all four tensors are in
        the mjx convention."""
        nv = self.num_vel
        # mjx fdsva_so LANDED: _op routes to fdsva_so_mujoco (MUJOCO_OUTPUT=true kernel;
        # epilogue recomputes Minv/qdd/dqdd_du fresh into a disjoint scratch band — the
        # §1g/§1h liveness bug is fixed).
        flat = self._op(_convention, "fdsva_so")(q, qd, u, float(gravity))
        B = flat.shape[0]
        return SecondOrderFD(*(flat[:, i*nv**3:(i+1)*nv**3].reshape(B, nv, nv, nv) for i in range(4)))

    def integrator_gradient(self, q, qd, u, dt, *, integrator_type: str = "euler", gravity: float = -9.81,
                            _convention=None):
        """dAB (B, 2*NV, 3*NV) = [d/dq | d/dqd | d/du] tangent.

        With ``output_convention="mujoco"`` (floating base) the state-transition
        Jacobian is in the mjx convention (retract tangent + reframed velocity)."""
        nv = self.num_vel
        it = _integrator_code(integrator_type)
        raw = self._op(_convention, "integrator_gradient")(q, qd, u, float(dt), it, float(gravity))
        B = raw.shape[0]
        return raw.reshape(B, 3 * nv, 2 * nv).transpose(1, 2)

    # ─── field-standard short aliases ────────────────────────────────────────
    # `rnea`/`fd` are the names roboticists reach for (pinocchio / frax / bard);
    # bind them to the long-named methods (aba / crba / minv already match).
    rnea = inverse_dynamics
    fd = forward_dynamics

    # ─── grid_plant surface (cost / barrier / plant-step) ────────────────────
    #
    # Mirror the numpy RobotHandle plant methods (shapes / fields / gravity /
    # integrator_type). Cost methods return (value, grad, hess); barriers return
    # (value, grad, hess_diag). value is squeezed to (B,) to match numpy.

    def quadratic_state_cost(self, x, x_des, Q, *, _convention=None):
        """1/2 sum_i Q_i (x_i - x_des_i)^2 over x=[q;qd]. Returns
        (value (B,), grad (B, NX), hess=diag(Q) (B, NX, NX)).

        With ``output_convention="mujoco"`` (floating base) ``x`` is MuJoCo-convention;
        the value is invariant, the grad base-rotates (covector) and the GN hess is
        the mjx congruence."""
        nx = self.num_joints + self.num_vel
        out, grad, hess = self._op(_convention, "quadratic_state_cost")(x, x_des, Q)
        return out[:, 0], grad, hess.reshape(-1, nx, nx)

    def quadratic_input_cost(self, u, u_des, R):
        """1/2 sum_i R_i (u_i - u_des_i)^2 over u (NV). Returns
        (value (B,), grad (B, NV), hess=diag(R) (B, NV, NV))."""
        nv = self.num_vel
        out, grad, hess = self._ops.quadratic_input_cost(u, u_des, R)
        return out[:, 0], grad, hess.reshape(-1, nv, nv)

    def joint_position_barrier(self, var, lower, upper, mu):
        """Log-barrier over NUM_POS positions. Returns
        (value (B,), grad (B, NUM_POS), hess_diag (B, NUM_POS))."""
        out, grad, hdiag = self._ops.joint_position_barrier(var, lower, upper, float(mu))
        return out[:, 0], grad, hdiag

    def joint_velocity_barrier(self, var, lower, upper, mu):
        """Log-barrier over NUM_VEL velocities. See joint_position_barrier."""
        out, grad, hdiag = self._ops.joint_velocity_barrier(var, lower, upper, float(mu))
        return out[:, 0], grad, hdiag

    def joint_torque_barrier(self, var, lower, upper, mu):
        """Log-barrier over NUM_VEL torques. See joint_position_barrier."""
        out, grad, hdiag = self._ops.joint_torque_barrier(var, lower, upper, float(mu))
        return out[:, 0], grad, hdiag

    def plant_step(self, x, u, dt, *, integrator_type: str = "euler", gravity: float = -9.81,
                   _convention=None):
        """x_{k+1} = integrator(x_k, u_k, dt). x (B, NX); u (B, NV). Returns (B, NX).

        With ``output_convention="mujoco"`` (floating base) ``x`` is MuJoCo-convention
        and the returned next state is mjx-convention (global-additive base retract)."""
        it = _integrator_code(integrator_type)
        return self._op(_convention, "plant_step")(x, u, float(dt), it, float(gravity))

    def plant_step_gradient(self, x, u, dt, *, integrator_type: str = "euler", gravity: float = -9.81,
                            _convention=None):
        """[A|B] = d x_{k+1}/d(x,u). x (B, NX); u (B, NV). Returns (B, 2*NV, 3*NV)
        with column blocks [d/dq | d/dqd | d/du] (tangent space).

        With ``output_convention="mujoco"`` (floating base) the state-transition
        Jacobian is in the mjx convention."""
        nv = self.num_vel
        it = _integrator_code(integrator_type)
        raw = self._op(_convention, "plant_step_gradient")(x, u, float(dt), it, float(gravity))
        B = raw.shape[0]
        return raw.reshape(B, 3 * nv, 2 * nv).transpose(1, 2)

    def ee_pos_cost(self, q, p_des, W, *, _convention=None):
        """End-effector position cost (EE 0). q (B, NQ); p_des/W (B, 3). Returns
        (value (B,), grad_x (B, NX), GN hess_x (B, NX, NX)).

        With ``output_convention="mujoco"`` (floating base) ``q`` is MuJoCo-convention;
        value invariant, grad covector-rotated, GN hess the mjx congruence."""
        nx = self.num_joints + self.num_vel
        out, grad, hess = self._op(_convention, "ee_pos_cost")(q, p_des, W)
        return out[:, 0], grad, hess.reshape(-1, nx, nx)

    def com_cost(self, q, p_des, W, *, _convention=None):
        """Center-of-mass tracking cost. q (B, NQ); p_des/W (B, 3). Returns
        (value (B,), grad_x (B, NX), GN hess_x (B, NX, NX)).

        With ``output_convention="mujoco"`` (floating base) ``q`` is MuJoCo-convention;
        value invariant, grad covector-rotated, GN hess the mjx congruence."""
        nx = self.num_joints + self.num_vel
        out, grad, hess = self._op(_convention, "com_cost")(q, p_des, W)
        return out[:, 0], grad, hess.reshape(-1, nx, nx)

    def momentum_cost(self, q, qd, h_des, W, *, _convention=None):
        """Centroidal-momentum tracking cost. q (B, NQ); qd (B, NV); h_des/W (B, 6).
        Returns (value (B,), grad_x (B, NX), GN hess_x (B, NX, NX)).

        With ``output_convention="mujoco"`` (floating base) ``q``/``qd`` are
        MuJoCo-convention; value invariant, grad covector-rotated, GN hess congruence."""
        nx = self.num_joints + self.num_vel
        out, grad, hess = self._op(_convention, "momentum_cost")(q, qd, h_des, W)
        return out[:, 0], grad, hess.reshape(-1, nx, nx)

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

    # ─── lifecycle ───────────────────────────────────────────────────────────

    def close(self) -> None:
        """Release the underlying .so handle (delegates to the base RobotHandle).
        Idempotent. The process-global torch op-library registration is unaffected."""
        self._base.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc) -> None:
        self.close()

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
    output_convention: str = "pinocchio",
    algorithm_list: list[str] | str | None = None,
    use_joint_dynamics: bool = False,
    runtime_joint_dynamics: bool = False,
    runtime_inertia: bool = False,
    runtime_transform: bool = False,
    enable_tool: bool = False,
    enable_mujoco_kernels: bool = True,
    dtype: str = "float32",
) -> TorchRobotHandle:
    """Register a robot for the torch backend (same cache as the plain/JAX
    surfaces). Returns a :py:class:`TorchRobotHandle`.

    ``dtype="float64"`` (Wave 2a) builds/loads the fp64 .so — the torch ops then
    take and return float64 CUDA tensors (fp32 tensors are rejected by the op's
    dtype check, and vice versa for the default fp32 build).

    ``algorithm_list`` (subset build) is supported: only the requested cores + their
    transitive deps are compiled into the torch op surface; calling a method that
    was excluded raises a clean "not built into this robot .so — add to
    algorithm_list and rebuild" error. ``None`` ⇒ the full default profile."""
    _require_torch()  # fail early with install guidance if torch is missing
    base = _grid_rbd.register_robot(
        name=name, urdf_path=urdf_path, urdf_string=urdf_string,
        floating_base=floating_base, ee_joint_names=ee_joint_names,
        max_batch_size=max_batch_size, cache_dir=cache_dir,
        force_rebuild=force_rebuild, cuda_arch=cuda_arch,
        algorithm_list=algorithm_list, use_joint_dynamics=use_joint_dynamics,  # C5
        runtime_joint_dynamics=runtime_joint_dynamics,  # C5 mutable damping/friction table
        runtime_inertia=runtime_inertia,  # D.4: mutable inertia table (torch op reads same device global)
        runtime_transform=runtime_transform,  # mutable joint-origin table (shared device global)
        enable_tool=enable_tool,  # tool welding: attach_tool/detach_tool/tool_fext surface
        enable_mujoco_kernels=enable_mujoco_kernels,  # pin-only builds (RAM/compile-time)
        dtype=dtype,  # Wave 2a: fp64 .so carries an fp64 torch surface
        _profile_overlay="torch",  # E6 torch threads overlay
    )
    cache_key, so_path = _lookup(name, cache_dir)
    return TorchRobotHandle(base, cache_key, so_path,
                            output_convention=output_convention)


def get_robot(name: str, cache_dir: str | Path | None = None, *,
              output_convention: str = "pinocchio") -> TorchRobotHandle:
    """Look up a previously-registered robot (same cache as grid_rbd.get_robot).
    ``output_convention`` ('pinocchio' or 'mujoco') mirrors register_robot and
    can also be set later via the handle's ``output_convention`` property."""
    _require_torch()  # fail early with install guidance if torch is missing
    base = _grid_rbd.get_robot(name, cache_dir=cache_dir, _profile_overlay="torch")  # E6
    cache_key, so_path = _lookup(name, cache_dir)
    return TorchRobotHandle(base, cache_key, so_path,
                            output_convention=output_convention)


def _lookup(name, cache_dir):
    from grid_rbd._cache import default_cache_dir, manifest_lookup, store_dir
    cd = Path(cache_dir).expanduser() if cache_dir else default_cache_dir()
    entry = manifest_lookup(cd, name)
    if entry is None:
        raise RuntimeError(f"{name!r} isn't in manifest; cache may be corrupted")
    so_path = store_dir(cd, entry["cache_key"]) / "robot.so"
    return entry["cache_key"], str(so_path)


__all__ = ["TorchRobotHandle", "GraphCallable", "register_robot", "get_robot"]
