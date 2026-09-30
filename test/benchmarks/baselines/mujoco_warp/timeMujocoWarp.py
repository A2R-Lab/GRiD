#!/usr/bin/env python3
"""Time MuJoCo Warp (MJWarp) algorithms for one robot.

Mirrors timeMJX.py's structure and emits the SAME label format so
`parse_grid_output` can be reused. See the module docstring of `run.py` and
`docs/open-tasks/mujoco_warp_baseline_plan.md` for the validation checklist.

MJWarp (https://github.com/google-deepmind/mujoco_warp) is the NVIDIA-Warp-based
GPU successor to MJX. It loads a standard MuJoCo MJCF (same models as MJX), and
batches across `nworld` parallel environments. Confirmed top-level API names
(from the repo __init__.py / mjwarp API docs, June 2026):
    mjw.put_model(mjm) -> Model
    mjw.make_data(mjm, nworld=N) -> Data          # batched, fields are wp arrays
    mjw.put_data(mjm, mjd, nworld=N) -> Data
    mjw.forward(m, d)        # forward dynamics -> d.qacc
    mjw.inverse(m, d)        # inverse dynamics -> d.qfrc_inverse
    mjw.kinematics(m, d)     # forward kinematics -> d.xpos / d.xquat
    mjw.crb(m, d), mjw.factor_m(m, d)   # composite-rigid-body + mass-matrix factor

Algorithm coverage:
    inverse_dynamics   -> mjw.inverse      (CONFIRMED export)
    forward_dynamics   -> mjw.forward      (CONFIRMED export)
    end_effector_pose  -> mjw.kinematics   (CONFIRMED export)
    crba (mass matrix) -> mjw.crb + mjw.factor_m  (UNCERTAIN: factor_m yields an
                          LTDL/LDL factorization in d.qLD, not a dense M like
                          GRiD/Frax CRBA. Timed as the closest analog; verify the
                          output semantics before trusting the comparison.)
Derivatives (inverse_dynamics_gradient / forward_dynamics_gradient): wired via the
    GPU FINITE-DIFFERENCE Jacobian wp.autograd.jacobian_fd (the mjd_transitionFD GPU
    analogue). Warp ITSELF has autodiff (wp.Tape / wp.autograd.jacobian), but
    mujoco_warp ships every module with wp.set_module_options(enable_backward=False)
    (forward.py/inverse.py/support.py/... ~20 modules), so its kernels emit NO adjoint
    code and the AUTODIFF path wp.autograd.jacobian(mjw.forward, ...) raises "Kernel
    must have backward pass enabled". jacobian_fd does NOT need backward — it just
    relaunches the forward kernel with central-difference-perturbed inputs — so it
    works against the vendored package unpatched. jacobian_fd does not accept
    @wp.struct args, so we wrap mjw.inverse/forward in a plain Python function whose
    differentiable inputs are bare warp arrays (qpos/qvel/qacc for id_du,
    qpos/qvel/qfrc_applied for fd_du) that we assign into the captured Data struct
    before each relaunch and whose output (qfrc_inverse / qacc) is returned as a
    fresh requires_grad array. Capped at DERIV_BATCH_SIZES (default <=256, like the
    MJX adapter) because FD does O(nworld*nv) forward launches per input.
NOT available (null): the SO (second-order) algorithms — no FD path wired for those.

Usage:
    python timeMujocoWarp.py <mjcf_path> [T/F] [ee_body_name]
"""
from __future__ import annotations

import os
import sys
import time

import numpy as np

# Configurable via env var (set by mujoco_warp/run.py's --test-iters flag).
TEST_ITERS  = int(os.environ.get("BENCH_TEST_ITERS", "500"))
BATCH_SIZES = [16, 32, 64, 128, 256, 1024]
# Finite-difference Jacobian timing does O(nworld*nv) forward launches per input,
# so N=1024 is intractable; the competitive gradient figures only need up to
# batch_256. Cap here (override via MUJOCO_WARP_DERIV_BATCH_SIZES="16,256"),
# mirroring the MJX adapter's DERIV_BATCH_SIZES.
DERIV_BATCH_SIZES = [int(x) for x in
                     os.environ.get("MUJOCO_WARP_DERIV_BATCH_SIZES", "16,32,64,128,256").split(",") if x.strip()]
N_WARMUP_PASSES = 3


# ---------------------------------------------------------------------------
# Output helpers (same format as timeGRiD / timeMJX)
# ---------------------------------------------------------------------------

def _print_stats(label: str, n: int, times: np.ndarray) -> None:
    print(
        f"[N:{n}]: {label}: "
        f"Average[{np.mean(times):.4f}us] "
        f"Std Dev [{np.std(times):.4f}us] "
        f"Min [{np.min(times):.4f}us] "
        f"Max [{np.max(times):.4f}us]"
    )


# ---------------------------------------------------------------------------
# Warp timing primitives
#
# Discipline (mirrors timeGRiD on the C++ side + timeMJX on the JAX side):
#   1. Warmup: a few launches of the kernel to trigger Warp kernel codegen/JIT
#      (module load + ptx) and stabilize the GPU clock. Discarded.
#   2. Time the next N calls. Every timed launch is bracketed by wp.synchronize()
#      so the host clock measures real device completion, not just enqueue.
#
# COMPUTE ONLY: state already resident in device `Data`; just relaunch the kernel.
# WITH MEMORY: regenerate host numpy state per-iter and copy it into the device
#   Data arrays (wp.copy / .assign) inside the timed region, simulating the
#   host->device transfer that the GRiD "with mem" timing includes.
# ---------------------------------------------------------------------------

def _sync():
    import warp as wp
    wp.synchronize()


def _warmup(launch_fn, n_warmup: int = N_WARMUP_PASSES) -> None:
    """Trigger Warp kernel compile + warmup passes. Discards results."""
    for _ in range(n_warmup + 1):
        launch_fn()
    _sync()


def _time_compute(launch_fn, n_iters: int = TEST_ITERS) -> np.ndarray:
    """Time a warmed-up Warp launch with state already on device."""
    times = []
    for _ in range(n_iters):
        _sync()
        t0 = time.perf_counter()
        launch_fn()
        _sync()
        times.append((time.perf_counter() - t0) * 1e6)
    return np.array(times)


def _time_with_mem(launch_fn, upload_fn, n_iters: int) -> np.ndarray:
    """Time launch_fn where upload_fn() copies fresh host state into device Data
    inside the timed region (host->device transfer included)."""
    times = []
    for _ in range(n_iters):
        host_state = _make_host_state()      # outside the timer
        _sync()
        t0 = time.perf_counter()
        upload_fn(host_state)
        launch_fn()
        _sync()
        times.append((time.perf_counter() - t0) * 1e6)
    return np.array(times)


# Module-level holders so the with-mem helpers can rebuild host state cheaply.
_NQ = 0
_NV = 0
_NWORLD = 1
_FLOATING = False


def _make_host_state() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Fresh (qpos, qvel, qacc) host arrays of shape (nworld, n*)."""
    qpos = np.random.randn(_NWORLD, _NQ).astype(np.float32)
    qvel = np.random.randn(_NWORLD, _NV).astype(np.float32)
    qacc = np.random.randn(_NWORLD, _NV).astype(np.float32)
    if _FLOATING and _NQ >= 7:
        quat = qpos[:, 3:7]
        norm = np.linalg.norm(quat, axis=1, keepdims=True) + 1e-8
        qpos[:, 3:7] = quat / norm
    return qpos, qvel, qacc


def _upload_state(d, host_state) -> None:
    """Copy host (qpos, qvel, qacc) into the batched device Data warp arrays.

    UNCERTAIN: the exact warp-array assignment API. `wp.array.assign(np_array)`
    works for matching shape/dtype; if MJWarp stores these as 2D (nworld, n)
    arrays this is a direct assign. Verify field shapes on first run.
    """
    import warp as wp
    qpos, qvel, qacc = host_state
    d.qpos.assign(wp.array(qpos, dtype=wp.float32))
    d.qvel.assign(wp.array(qvel, dtype=wp.float32))
    d.qacc.assign(wp.array(qacc, dtype=wp.float32))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    global _NQ, _NV, _NWORLD, _FLOATING

    if len(sys.argv) < 2:
        print(f"Usage: {sys.argv[0]} <mjcf_path> [T/F] [ee_body_name]", file=sys.stderr)
        sys.exit(1)

    mjcf_path     = sys.argv[1]
    floating_base = len(sys.argv) > 2 and sys.argv[2].upper() == "T"
    ee_body_name  = sys.argv[3] if len(sys.argv) > 3 else None
    _FLOATING     = floating_base

    try:
        import warp as wp
        import mujoco
        import mujoco_warp as mjw
    except ImportError as e:
        print(f"MuJoCo Warp import failed: {e}", file=sys.stderr)
        sys.exit(1)

    wp.config.quiet = True
    wp.init()

    # ------------------------------------------------------------------
    # Load model. Mirror timeMJX's collision/constraint disabling so the
    # model loads on every robot MJCF and we time the unconstrained
    # articulated-body dynamics (apples-to-apples with GRiD).
    # ------------------------------------------------------------------
    model = mujoco.MjModel.from_xml_path(mjcf_path)
    model.geom_contype[:]     = 0
    model.geom_conaffinity[:] = 0
    model.opt.disableflags |= int(mujoco.mjtDisableBit.mjDSBL_CONSTRAINT)

    nq, nv = model.nq, model.nv
    _NQ, _NV = nq, nv

    mjd = mujoco.MjData(model)
    m   = mjw.put_model(model)

    # EE body id (kinematics writes all body poses; we don't slice here, but
    # validate the name resolves so the harness EE-frame plumbing is exercised).
    if ee_body_name:
        _id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, ee_body_name)
        if _id < 0:
            print(f"  [mujoco_warp] warning: body '{ee_body_name}' not found",
                  file=sys.stderr)

    # ------------------------------------------------------------------
    # Metadata block (parsed by run.py::_parse_mujoco_warp_metadata)
    # ------------------------------------------------------------------
    try:
        wp_version = wp.__version__
    except Exception:
        wp_version = "unknown"
    try:
        device = str(wp.get_device())
    except Exception:
        device = "unknown"

    print("=== BEGIN MUJOCO_WARP METADATA ===")
    print(f"mujoco_version: {mujoco.__version__}")
    print(f"warp_version: {wp_version}")
    print(f"warp_device: {device}")
    try:
        print(f"mujoco_warp_version: {mjw.__version__}")
    except Exception:
        print("mujoco_warp_version: unknown")
    print("=== END MUJOCO_WARP METADATA ===")

    # ------------------------------------------------------------------
    # Helper: build a batched device Data with randomized state.
    # ------------------------------------------------------------------
    def _make_device_data(nworld: int):
        d = mjw.make_data(model, nworld=nworld)
        qpos = np.random.randn(nworld, nq).astype(np.float32)
        qvel = np.random.randn(nworld, nv).astype(np.float32)
        qacc = np.random.randn(nworld, nv).astype(np.float32)
        if floating_base and nq >= 7:
            quat = qpos[:, 3:7]
            norm = np.linalg.norm(quat, axis=1, keepdims=True) + 1e-8
            qpos[:, 3:7] = quat / norm
        d.qpos.assign(wp.array(qpos, dtype=wp.float32))
        d.qvel.assign(wp.array(qvel, dtype=wp.float32))
        d.qacc.assign(wp.array(qacc, dtype=wp.float32))
        return d

    # ------------------------------------------------------------------
    # GPU warmup — a few forward passes on a single world, discarded.
    # ------------------------------------------------------------------
    d_warm = _make_device_data(1)
    _warmup(lambda: mjw.forward(m, d_warm))

    # ------------------------------------------------------------------
    # Single-call timing (nworld = 1)
    # ------------------------------------------------------------------
    _NWORLD = 1
    d1 = _make_device_data(1)

    _ALGOS_SINGLE = [
        ("INVERSE_DYNAMICS",  lambda d=d1: mjw.inverse(m, d)),
        ("FORWARD_DYNAMICS",  lambda d=d1: mjw.forward(m, d)),
        ("END_EFFECTOR_POSE", lambda d=d1: mjw.kinematics(m, d)),
        # UNCERTAIN: crb + factor_m is the mass-matrix analog (factorized, not
        # dense). Time it but flag the semantic difference in the report.
        ("CRBA",              lambda d=d1: (mjw.crb(m, d), mjw.factor_m(m, d))),
    ]
    for label, fn in _ALGOS_SINGLE:
        try:
            _warmup(fn)
            t = _time_compute(fn)
            print(f"Single Call {label} {np.median(t):.4f}us")
        except Exception as e:
            print(f"# Single Call {label} skipped: {e}", file=sys.stderr)

    # Single-call FD-Jacobian timing (nworld = 1), mirroring timeMJX's single-call
    # gradient bars. Same wp.autograd.jacobian_fd wrapper as the batch path below.
    import warp.autograd as _wa_single
    _SINGLE_DERIVS = [
        ("INVERSE_DYNAMICS_GRADIENT", "qfrc_inverse", "qacc",         lambda dd: mjw.inverse(m, dd)),
        ("FORWARD_DYNAMICS_GRADIENT", "qacc",         "qfrc_applied", lambda dd: mjw.forward(m, dd)),
    ]
    for label, out_field, third_field, kernel in _SINGLE_DERIVS:
        try:
            _qp = wp.array(np.random.randn(1, nq).astype(np.float32), dtype=wp.float32, requires_grad=True)
            _qv = wp.array(np.random.randn(1, nv).astype(np.float32), dtype=wp.float32, requires_grad=True)
            _th = wp.array(np.random.randn(1, nv).astype(np.float32), dtype=wp.float32, requires_grad=True)

            def _du1(qp, qv, th, kf=kernel, of=out_field, tf=third_field):
                d1.qpos.assign(qp); d1.qvel.assign(qv); getattr(d1, tf).assign(th)
                kf(d1)
                out = wp.zeros((1, nv), dtype=wp.float32, requires_grad=True)
                wp.copy(out, getattr(d1, of))
                return out

            def _launch1(qp=_qp, qv=_qv, th=_th, f=_du1):
                _wa_single.jacobian_fd(f, inputs=[qp, qv, th])

            _warmup(_launch1)
            t = _time_compute(_launch1, n_iters=max(1, TEST_ITERS // 20))
            print(f"Single Call {label} {np.median(t):.4f}us")
        except Exception as e:
            print(f"# Single Call {label} skipped: {e}", file=sys.stderr)

    # ------------------------------------------------------------------
    # Batch timing via nworld (Warp's native batch axis — no vmap needed).
    # ------------------------------------------------------------------
    for N in BATCH_SIZES:
        _NWORLD = N
        try:
            d = _make_device_data(N)
        except Exception as e:
            print(f"# [N:{N}] make_data skipped: {e}", file=sys.stderr)
            continue

        n_batch_iters = max(10, TEST_ITERS // 5)

        _ALGOS_BATCH = [
            ("INVERSE_DYNAMICS",  lambda dd=d: mjw.inverse(m, dd)),
            ("FORWARD_DYNAMICS",  lambda dd=d: mjw.forward(m, dd)),
            ("END_EFFECTOR_POSE", lambda dd=d: mjw.kinematics(m, dd)),
            ("CRBA",              lambda dd=d: (mjw.crb(m, dd), mjw.factor_m(m, dd))),
        ]

        for label, fn in _ALGOS_BATCH:
            try:
                _warmup(fn)
            except Exception as e:
                print(f"# [N:{N}] {label} warmup skipped: {e}", file=sys.stderr)
                continue

            # WITH MEMORY: fresh host state copied to device inside the timer.
            try:
                wm = _time_with_mem(
                    fn,
                    lambda hs, dd=d: _upload_state(dd, hs),
                    n_iters=n_batch_iters,
                )
                _print_stats(f"{label} WITH MEMORY", N, wm)
            except Exception as e:
                print(f"# [N:{N}] {label} WITH MEMORY skipped: {e}", file=sys.stderr)

            # COMPUTE ONLY: state already resident on device.
            try:
                co = _time_compute(fn, n_iters=n_batch_iters)
                _print_stats(f"{label} COMPUTE ONLY", N, co)
            except Exception as e:
                print(f"# [N:{N}] {label} COMPUTE ONLY skipped: {e}", file=sys.stderr)

    # ------------------------------------------------------------------
    # Derivative (gradient) timing: id_du + fd_du via the GPU FINITE-DIFFERENCE
    # Jacobian wp.autograd.jacobian_fd — the mujoco_warp analogue of GRiD's
    # analytic gradients and of MJX's jax.jacobian path. mujoco_warp ships
    # enable_backward=False so the AUTODIFF Jacobian is unavailable, but jacobian_fd
    # only needs the forward kernel (central-difference relaunches), so it works
    # unpatched. jacobian_fd rejects @wp.struct args, so each derivative wraps
    # mjw.inverse/forward in a plain Python fn whose differentiable inputs are bare
    # requires_grad warp arrays assigned into the captured Data before the launch.
    # id_du differentiates qfrc_inverse w.r.t. (qpos, qvel, qacc); fd_du
    # differentiates qacc w.r.t. (qpos, qvel, qfrc_applied) (qfrc_applied = GRiD's
    # tau). Capped at DERIV_BATCH_SIZES because FD does O(nworld*nv) launches/input.
    # ------------------------------------------------------------------
    import warp.autograd as wa

    def _rand_grad_array(rows: int, cols: int):
        return wp.array(np.random.randn(rows, cols).astype(np.float32),
                        dtype=wp.float32, requires_grad=True)

    # (label, output-Data field, third differentiable input field). id_du perturbs
    # qacc, fd_du perturbs qfrc_applied; both also perturb qpos/qvel.
    _DERIVS = [
        ("INVERSE_DYNAMICS_GRADIENT", "qfrc_inverse", "qacc",         lambda d: mjw.inverse(m, d)),
        ("FORWARD_DYNAMICS_GRADIENT", "qacc",         "qfrc_applied", lambda d: mjw.forward(m, d)),
    ]

    for N in DERIV_BATCH_SIZES:
        try:
            dd = _make_device_data(N)
        except Exception as e:
            print(f"# [N:{N}] deriv make_data skipped: {e}", file=sys.stderr)
            continue
        n_du_iters = max(1, TEST_ITERS // 20)

        for label, out_field, third_field, kernel in _DERIVS:
            try:
                # Differentiable inputs (requires_grad) assigned into the Data
                # struct each launch; the captured `dd` provides every other field.
                qpos_in = _rand_grad_array(N, nq)
                qvel_in = _rand_grad_array(N, nv)
                third_in = _rand_grad_array(N, nv)

                def _du_fn(qp, qv, th, kf=kernel, of=out_field, tf=third_field, _dd=dd):
                    _dd.qpos.assign(qp)
                    _dd.qvel.assign(qv)
                    getattr(_dd, tf).assign(th)
                    kf(_dd)
                    out = wp.zeros((N, nv), dtype=wp.float32, requires_grad=True)
                    wp.copy(out, getattr(_dd, of))
                    return out

                def _launch_du(qp=qpos_in, qv=qvel_in, th=third_in, f=_du_fn):
                    wa.jacobian_fd(f, inputs=[qp, qv, th])

                _warmup(_launch_du)

                # WITH MEMORY: upload fresh host state into the FD input arrays
                # inside the timed region (host->device transfer included).
                def _upload_du(host_state, qp=qpos_in, qv=qvel_in, th=third_in):
                    hqpos, hqvel, hthird = host_state
                    qp.assign(wp.array(hqpos, dtype=wp.float32))
                    qv.assign(wp.array(hqvel, dtype=wp.float32))
                    th.assign(wp.array(hthird, dtype=wp.float32))

                def _make_du_host():
                    hq = np.random.randn(N, nq).astype(np.float32)
                    if floating_base and nq >= 7:
                        quat = hq[:, 3:7]
                        hq[:, 3:7] = quat / (np.linalg.norm(quat, axis=1, keepdims=True) + 1e-8)
                    return (hq,
                            np.random.randn(N, nv).astype(np.float32),
                            np.random.randn(N, nv).astype(np.float32))

                wm_times = []
                for _ in range(n_du_iters):
                    hs = _make_du_host()          # outside the timer
                    _sync()
                    t0 = time.perf_counter()
                    _upload_du(hs)
                    _launch_du()
                    _sync()
                    wm_times.append((time.perf_counter() - t0) * 1e6)
                _print_stats(f"{label} WITH MEMORY", N, np.array(wm_times))

                # COMPUTE ONLY: inputs already resident on device.
                co = _time_compute(_launch_du, n_iters=n_du_iters)
                _print_stats(f"{label} COMPUTE ONLY", N, co)
            except Exception as e:
                print(f"# [N:{N}] {label} skipped: {e}", file=sys.stderr)


if __name__ == "__main__":
    main()
