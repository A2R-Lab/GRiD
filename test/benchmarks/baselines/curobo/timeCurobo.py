#!/usr/bin/env python3
"""Time cuRobo dynamics CUDA kernels for one robot, in OUR label format.

UNTESTED: no `curobo` package is installed on this machine yet. This script
mirrors the cuRobo author's own benchmark
    https://github.com/NVlabs/curobo/blob/main/benchmark/inverse_dynamics_kernel_benchmark.py
for the API (imports, Kinematics/Dynamics construction, setup_batch_size,
compute_inverse_dynamics, the RNEA-backward via torch.autograd.backward) but
SWAPS its measurement layer + batch sweep for ours:
  * cuRobo's benchmark uses torch.profiler and reads `self_cuda_time_total` /
    `self_device_time_total` per "rnea_forward" / "rnea_backward" kernel,
    batches [1, 64, 256, 1024], and prints a tabulate grid.
  * WE need µs/iter in the timeGRiD/timeMJX label format (parse_grid_output)
    across OUR batch sweep 16/32/64/128/256 plus a single-call (batch=1).
So this script wall-clocks each launch with torch.cuda.synchronize() bracketing
(the same device-completion discipline as timeMJX's _sync()), which is the
robust apples-to-apples measure against GRiD/MJX/MuJoCo-Warp (all of which we
time the same way). The torch.profiler self_cuda_time path can be added later if
we want pure-kernel (launch-overhead-excluded) numbers; for parity with the
other GPU baselines we use synchronized wall-clock here.

API (confirmed from the cuRobo benchmark, June 2026):
    from curobo._src.robot.dynamics.dynamics import Dynamics
    from curobo._src.robot.dynamics.dynamics_cfg import DynamicsCfg
    from curobo._src.robot.kinematics.kinematics import Kinematics, KinematicsCfg
    from curobo._src.state.state_joint import JointState
    from curobo._src.types.device_cfg import DeviceCfg
    from curobo._src.util.config_io import join_path, load_yaml
    from curobo.content import get_robot_configs_path

    robot_file = load_yaml(join_path(get_robot_configs_path(), <yml>))
    if "robot_cfg" in robot_file: robot_file = robot_file["robot_cfg"]
    robot_file["kinematics"]["collision_link_names"] = None
    robot_file["kinematics"]["lock_joints"] = {}
    kin  = Kinematics(KinematicsCfg.from_data_dict(robot_file, device_cfg=device_cfg))
    dyn  = Dynamics(DynamicsCfg(kinematics_config=kin.kinematics_config, device_cfg=device_cfg))
    dyn.setup_batch_size(batch_size=N)
    js   = JointState(position=q, velocity=qd, acceleration=qdd)   # (N, dof) cuda
    tau  = dyn.compute_inverse_dynamics(js)                        # RNEA forward
    torch.autograd.backward(tau, grad_tensors=torch.ones_like(tau))# RNEA backward

Algorithm coverage (see run.py docstring):
    INVERSE_DYNAMICS           -> compute_inverse_dynamics       (RNEA forward)
    INVERSE_DYNAMICS_GRADIENT  -> autograd.backward on tau       (RNEA backward)
    END_EFFECTOR_POSE          -> forward kinematics             (Kinematics FK)

Usage:
    python timeCurobo.py <curobo_robot_yml> [T/F]
    (T/F is the floating flag for harness parity; cuRobo is fixed-base only, so
     run.py never passes T here — it nulls floating before shelling out.)
"""
from __future__ import annotations

import os
import sys
import time

import numpy as np

# Configurable via env var (set by curobo/run.py's --test-iters flag).
TEST_ITERS  = int(os.environ.get("BENCH_TEST_ITERS", "500"))
BATCH_SIZES = [16, 32, 64, 128, 256, 1024]
N_WARMUP_PASSES = 5   # matches cuRobo benchmark default warmup_iters


# ---------------------------------------------------------------------------
# Output helpers (same format as timeGRiD / timeMJX / timeMujocoWarp)
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
# Timing primitives (torch.cuda.synchronize-bracketed wall clock).
#
#   COMPUTE ONLY: inputs already resident on device; just relaunch the kernel.
#   WITH MEMORY:  regenerate host numpy state per-iter and copy it into the
#                 device tensors inside the timed region (host->device transfer
#                 included, mirroring GRiD's "with mem" timing).
# ---------------------------------------------------------------------------
def _sync(device) -> None:
    import torch
    torch.cuda.synchronize(device)


def _warmup(launch_fn, device, n_warmup: int = N_WARMUP_PASSES) -> None:
    for _ in range(n_warmup + 1):
        launch_fn()
    _sync(device)


def _time_compute(launch_fn, device, n_iters: int = TEST_ITERS) -> np.ndarray:
    times = []
    for _ in range(n_iters):
        _sync(device)
        t0 = time.perf_counter()
        launch_fn()
        _sync(device)
        times.append((time.perf_counter() - t0) * 1e6)
    return np.array(times)


def _time_with_mem(launch_fn, upload_fn, make_host_fn, device,
                   n_iters: int) -> np.ndarray:
    times = []
    for _ in range(n_iters):
        host_state = make_host_fn()           # outside the timer
        _sync(device)
        t0 = time.perf_counter()
        upload_fn(host_state)
        launch_fn()
        _sync(device)
        times.append((time.perf_counter() - t0) * 1e6)
    return np.array(times)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    if len(sys.argv) < 2:
        print(f"Usage: {sys.argv[0]} <curobo_robot_yml> [T/F]", file=sys.stderr)
        sys.exit(1)

    curobo_yml    = sys.argv[1]
    floating_base = len(sys.argv) > 2 and sys.argv[2].upper() == "T"
    if floating_base:
        # run.py should already have nulled floating; guard anyway.
        print("# cuRobo Dynamics is fixed-base only; floating not supported.",
              file=sys.stderr)
        sys.exit(2)

    try:
        import torch
        from curobo._src.robot.dynamics.dynamics import Dynamics
        from curobo._src.robot.dynamics.dynamics_cfg import DynamicsCfg
        from curobo._src.robot.kinematics.kinematics import Kinematics, KinematicsCfg
        from curobo._src.state.state_joint import JointState
        from curobo._src.types.device_cfg import DeviceCfg
        from curobo._src.util.config_io import join_path, load_yaml
        from curobo.content import get_robot_configs_path
    except ImportError as e:
        print(f"cuRobo import failed: {e}", file=sys.stderr)
        sys.exit(1)

    if not torch.cuda.is_available():
        print("cuRobo timing requires a CUDA device.", file=sys.stderr)
        sys.exit(1)

    device     = torch.device("cuda:0")
    device_cfg = DeviceCfg(device=device)
    torch.manual_seed(2)
    _ = torch.zeros((1,), device=device)   # init context

    # ------------------------------------------------------------------
    # Load robot config + build Kinematics/Dynamics (per cuRobo benchmark).
    # ------------------------------------------------------------------
    robot_file = load_yaml(join_path(get_robot_configs_path(), curobo_yml))
    if "robot_cfg" in robot_file:
        robot_file = robot_file["robot_cfg"]
    # Disable collision/lock so it loads + times pure articulated-body dynamics.
    try:
        robot_file["kinematics"]["collision_link_names"] = None
        robot_file["kinematics"]["lock_joints"] = {}
    except (KeyError, TypeError):
        pass

    kin = Kinematics(KinematicsCfg.from_data_dict(robot_file, device_cfg=device_cfg))
    dynamics = Dynamics(DynamicsCfg(kinematics_config=kin.kinematics_config,
                                    device_cfg=device_cfg))
    dof = kin.get_dof()

    # ------------------------------------------------------------------
    # Metadata block (parsed by run.py::_parse_curobo_metadata)
    # ------------------------------------------------------------------
    try:
        import curobo
        curobo_version = getattr(curobo, "__version__", "unknown")
    except Exception:
        curobo_version = "unknown"
    print("=== BEGIN CUROBO METADATA ===")
    print(f"curobo_version: {curobo_version}")
    print(f"torch_version: {torch.__version__}")
    print(f"torch_device: {torch.cuda.get_device_name(device)}")
    print(f"robot_yml: {curobo_yml}")
    print(f"dof: {dof}")
    print("=== END CUROBO METADATA ===")

    # ------------------------------------------------------------------
    # State builders
    # ------------------------------------------------------------------
    def _make_state(n: int, requires_grad: bool = False):
        q   = torch.rand((n, dof), device=device, requires_grad=requires_grad)
        qd  = torch.randn((n, dof), device=device, requires_grad=requires_grad)
        qdd = torch.randn((n, dof), device=device, requires_grad=requires_grad)
        return q.contiguous(), qd.contiguous(), qdd.contiguous()

    def _make_host(n: int):
        return (
            np.random.rand(n, dof).astype(np.float32),
            np.random.randn(n, dof).astype(np.float32),
            np.random.randn(n, dof).astype(np.float32),
        )

    # ------------------------------------------------------------------
    # Per-algorithm launch closures bound to a given (n, state).
    # ------------------------------------------------------------------
    def _fwd_launch(q, qd, qdd):
        js = JointState(position=q, velocity=qd, acceleration=qdd)
        return dynamics.compute_inverse_dynamics(js)

    def _make_grad_launch(q, qd, qdd):
        grad_tau = torch.ones((q.shape[0], dof), device=device)
        def _launch():
            for t in (q, qd, qdd):
                if t.grad is not None:
                    t.grad.zero_()
            tau = _fwd_launch(q, qd, qdd)
            torch.autograd.backward(tau, grad_tensors=grad_tau)
        return _launch

    def _fk_launch(q):
        # cuRobo forward kinematics. Method name may be get_state / forward;
        # try the common ones and fall back. (Validate on first run.)
        for meth in ("get_state", "forward", "compute_kinematics"):
            fn = getattr(kin, meth, None)
            if fn is not None:
                return fn(q)
        raise AttributeError("no Kinematics FK method (get_state/forward) found")

    # ==================================================================
    # SINGLE-CALL timing (batch = 1)
    # ==================================================================
    dynamics.setup_batch_size(batch_size=1)

    # INVERSE_DYNAMICS (RNEA forward)
    try:
        q, qd, qdd = _make_state(1)
        fn = lambda: _fwd_launch(q, qd, qdd)
        _warmup(fn, device)
        t = _time_compute(fn, device)
        print(f"Single Call INVERSE_DYNAMICS {np.median(t):.4f}us")
    except Exception as e:
        print(f"# Single Call INVERSE_DYNAMICS skipped: {e}", file=sys.stderr)

    # INVERSE_DYNAMICS_GRADIENT (RNEA backward)
    try:
        q, qd, qdd = _make_state(1, requires_grad=True)
        fn = _make_grad_launch(q, qd, qdd)
        _warmup(fn, device)
        t = _time_compute(fn, device)
        print(f"Single Call INVERSE_DYNAMICS_GRADIENT {np.median(t):.4f}us")
    except Exception as e:
        print(f"# Single Call INVERSE_DYNAMICS_GRADIENT skipped: {e}", file=sys.stderr)

    # END_EFFECTOR_POSE (forward kinematics)
    try:
        q, _, _ = _make_state(1)
        fn = lambda: _fk_launch(q)
        _warmup(fn, device)
        t = _time_compute(fn, device)
        print(f"Single Call END_EFFECTOR_POSE {np.median(t):.4f}us")
    except Exception as e:
        print(f"# Single Call END_EFFECTOR_POSE skipped: {e}", file=sys.stderr)

    # ==================================================================
    # BATCH timing across OUR sweep (16..256), NOT cuRobo's [1,64,256,1024].
    # ==================================================================
    for N in BATCH_SIZES:
        try:
            dynamics.setup_batch_size(batch_size=N)
        except Exception as e:
            print(f"# [N:{N}] setup_batch_size skipped: {e}", file=sys.stderr)
            continue

        n_batch_iters = max(10, TEST_ITERS // 5)

        # ---- INVERSE_DYNAMICS ----
        try:
            q, qd, qdd = _make_state(N)
            fn = lambda: _fwd_launch(q, qd, qdd)
            _warmup(fn, device)

            def _upload(hs, q=q, qd=qd, qdd=qdd):
                hq, hqd, hqdd = hs
                q.copy_(torch.from_numpy(hq).to(device))
                qd.copy_(torch.from_numpy(hqd).to(device))
                qdd.copy_(torch.from_numpy(hqdd).to(device))
            wm = _time_with_mem(fn, _upload, lambda: _make_host(N), device, n_batch_iters)
            _print_stats("INVERSE_DYNAMICS WITH MEMORY", N, wm)
            co = _time_compute(fn, device, n_iters=n_batch_iters)
            _print_stats("INVERSE_DYNAMICS COMPUTE ONLY", N, co)
        except Exception as e:
            print(f"# [N:{N}] INVERSE_DYNAMICS skipped: {e}", file=sys.stderr)

        # ---- INVERSE_DYNAMICS_GRADIENT ----
        try:
            q, qd, qdd = _make_state(N, requires_grad=True)
            fn = _make_grad_launch(q, qd, qdd)
            _warmup(fn, device)

            def _upload_g(hs, q=q, qd=qd, qdd=qdd):
                hq, hqd, hqdd = hs
                with torch.no_grad():
                    q.copy_(torch.from_numpy(hq).to(device))
                    qd.copy_(torch.from_numpy(hqd).to(device))
                    qdd.copy_(torch.from_numpy(hqdd).to(device))
            wm = _time_with_mem(fn, _upload_g, lambda: _make_host(N), device, n_batch_iters)
            _print_stats("INVERSE_DYNAMICS_GRADIENT WITH MEMORY", N, wm)
            co = _time_compute(fn, device, n_iters=n_batch_iters)
            _print_stats("INVERSE_DYNAMICS_GRADIENT COMPUTE ONLY", N, co)
        except Exception as e:
            print(f"# [N:{N}] INVERSE_DYNAMICS_GRADIENT skipped: {e}", file=sys.stderr)

        # ---- END_EFFECTOR_POSE (forward kinematics) ----
        try:
            q, _, _ = _make_state(N)
            fn = lambda: _fk_launch(q)
            _warmup(fn, device)

            def _upload_fk(hs, q=q):
                hq, _, _ = hs
                q.copy_(torch.from_numpy(hq).to(device))
            wm = _time_with_mem(fn, _upload_fk, lambda: _make_host(N), device, n_batch_iters)
            _print_stats("END_EFFECTOR_POSE WITH MEMORY", N, wm)
            co = _time_compute(fn, device, n_iters=n_batch_iters)
            _print_stats("END_EFFECTOR_POSE COMPUTE ONLY", N, co)
        except Exception as e:
            print(f"# [N:{N}] END_EFFECTOR_POSE skipped: {e}", file=sys.stderr)


if __name__ == "__main__":
    main()
