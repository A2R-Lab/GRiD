#!/usr/bin/env python3
"""Time Frax algorithms for one robot.

Prints results in the same format as timeGRiD so parse_grid_output can reuse it.
Available algorithms: id (rnea), fd (forward_dynamics), crba, minv. Others
(aba, id_du, fd_du, ee_pose, ee_pose_gradient, idsva_so, fdsva_so) are null.

Frax represents floating-base as 6 extra angle-axis DOF appended to the joint
vector (NOT quaternion), so for go2_floating: num_joints == 18 (12 actuated + 6).

Discipline (matches timeMJX.py): JIT once with a sample input, run N_WARMUP_PASSES
warmup calls, then time. With-memory variants take raw numpy arrays so host->device
transfer is included in the timed region.

Usage:
    python timeFrax.py <urdf_path> [T/F]
"""

from __future__ import annotations

import os
import sys
import time

import numpy as np

TEST_ITERS  = int(os.environ.get("BENCH_TEST_ITERS", "500"))
BATCH_SIZES = [16, 32, 64, 128, 256]
N_WARMUP_PASSES = 3

# Allow overriding the JAX backend before any jax import. Frax advertises
# fast performance on BOTH CPU and GPU; the multi-version harness runs
# this script twice (once per device) to capture both columns. Default:
# whatever JAX picks (CUDA if jax[cuda12] is installed; CPU otherwise).
_FRAX_DEVICE = os.environ.get("FRAX_DEVICE", "").strip().lower()
if _FRAX_DEVICE in ("cpu", "gpu"):
    # Must be set BEFORE the first `import jax` anywhere — JAX picks the
    # backend at first import and won't switch later. Two equivalent env
    # vars; set both so JAX picks regardless of version.
    _JAX_PLATFORM = "cpu" if _FRAX_DEVICE == "cpu" else "cuda"
    os.environ["JAX_PLATFORMS"] = _JAX_PLATFORM
    os.environ["JAX_PLATFORM_NAME"] = _JAX_PLATFORM


# ---------------------------------------------------------------------------
# Output helpers (same format as timeGRiD)
# ---------------------------------------------------------------------------

def _print_stats(label: str, n: int, times: np.ndarray) -> None:
    print(
        f"[N:{n}]: {label}: "
        f"Average[{np.mean(times):.4f}us] "
        f"Std Dev [{np.std(times):.4f}us] "
        f"Min [{np.min(times):.4f}us] "
        f"Max [{np.max(times):.4f}us]"
    )


def _jit_and_warmup(fn, sample_args, n_warmup: int = N_WARMUP_PASSES) -> None:
    """Trigger JIT compile + run warmup passes. Discards results."""
    import jax
    jax.block_until_ready(fn(*sample_args))
    for _ in range(n_warmup):
        jax.block_until_ready(fn(*sample_args))


def _time_device(fn, *args, n_iters: int = TEST_ITERS) -> np.ndarray:
    """Time a JIT-compiled, warmed-up JAX function with args already on device."""
    import jax
    times = []
    for _ in range(n_iters):
        t0 = time.perf_counter()
        jax.block_until_ready(fn(*args))
        times.append((time.perf_counter() - t0) * 1e6)
    return np.array(times)


def _time_with_mem(fn, make_args_fn, n_iters: int) -> np.ndarray:
    """Time fn where make_args_fn() produces numpy arrays transferred each call."""
    import jax
    times = []
    for _ in range(n_iters):
        np_args = make_args_fn()    # outside the timer
        t0 = time.perf_counter()
        jax.block_until_ready(fn(*np_args))
        times.append((time.perf_counter() - t0) * 1e6)
    return np.array(times)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    if len(sys.argv) < 2:
        print(f"Usage: {sys.argv[0]} <urdf_path> [T/F]", file=sys.stderr)
        sys.exit(1)

    urdf_path     = sys.argv[1]
    floating_base = len(sys.argv) > 2 and sys.argv[2].upper() == "T"

    try:
        import jax
        import jax.numpy as jnp
        import frax
        from frax.core.robot import Robot
    except ImportError as e:
        print(f"Frax import failed: {e}", file=sys.stderr)
        sys.exit(1)

    # ------------------------------------------------------------------
    # Load model
    # ------------------------------------------------------------------
    robot = Robot(str(urdf_path), add_floating_base=floating_base)

    # Frax has nq == nv: floating-base appends 6 angle-axis DOF, no quaternion.
    n = int(robot.num_joints)
    print(f"# frax: num_joints={n}, num_actuated={robot.num_actuated_joints}, "
          f"includes_floating_dof={robot.includes_floating_dof}")

    # ------------------------------------------------------------------
    # Metadata block (parsed by timing_parser)
    # ------------------------------------------------------------------
    try:
        backend = jax.default_backend()
    except Exception:
        backend = "unknown"
    frax_version = getattr(frax, "__version__", "unknown")

    print("=== BEGIN FRAX METADATA ===")
    print(f"frax_version: {frax_version}")
    print(f"jax_version: {jax.__version__}")
    print(f"jax_backend: {backend}")
    print(f"num_joints: {n}")
    print("ID codegen: false")
    print("FD codegen: false")
    print("CRBA codegen: false")
    print("MINV codegen: false")
    print("=== END FRAX METADATA ===")

    # ------------------------------------------------------------------
    # State factory
    # ------------------------------------------------------------------
    rng = jax.random.PRNGKey(0)

    def _make_state_jnp(key):
        k1, k2, k3, k4 = jax.random.split(key, 4)
        q   = jax.random.normal(k1, (n,))
        qd  = jax.random.normal(k2, (n,))
        qdd = jax.random.normal(k3, (n,))
        tau = jax.random.normal(k4, (n,))
        return q, qd, qdd, tau

    q_single, qd_single, qdd_single, tau_single = _make_state_jnp(rng)

    # ------------------------------------------------------------------
    # Pre-build JIT functions (each algorithm has its own)
    # ------------------------------------------------------------------
    _id_jit   = jax.jit(lambda q, qd, qdd: robot.rnea(q, qd, qdd, None, None))
    _fd_jit   = jax.jit(lambda q, qd, tau: robot.forward_dynamics(q, qd, tau, None))
    _crba_jit = jax.jit(lambda q: robot.crba(q))
    _minv_jit = jax.jit(lambda q: robot.mass_matrix_inverse(robot.mass_matrix(q)))

    # Warmup GPU (a few fd calls)
    for _ in range(5):
        jax.block_until_ready(_fd_jit(q_single, qd_single, tau_single))

    # ------------------------------------------------------------------
    # Single-call timing
    # ------------------------------------------------------------------
    try:
        _jit_and_warmup(_id_jit, (q_single, qd_single, qdd_single))
        t = _time_device(_id_jit, q_single, qd_single, qdd_single)
        print(f"Single Call ID {np.median(t):.4f}us")
    except Exception as e:
        print(f"# Single Call ID skipped: {e}", file=sys.stderr)

    try:
        _jit_and_warmup(_fd_jit, (q_single, qd_single, tau_single))
        t = _time_device(_fd_jit, q_single, qd_single, tau_single)
        print(f"Single Call FD {np.median(t):.4f}us")
    except Exception as e:
        print(f"# Single Call FD skipped: {e}", file=sys.stderr)

    try:
        _jit_and_warmup(_crba_jit, (q_single,))
        t = _time_device(_crba_jit, q_single)
        print(f"Single Call CRBA {np.median(t):.4f}us")
    except Exception as e:
        print(f"# Single Call CRBA skipped: {e}", file=sys.stderr)

    try:
        _jit_and_warmup(_minv_jit, (q_single,))
        t = _time_device(_minv_jit, q_single)
        print(f"Single Call MINV {np.median(t):.4f}us")
    except Exception as e:
        print(f"# Single Call MINV skipped: {e}", file=sys.stderr)

    # ------------------------------------------------------------------
    # Batch timing via vmap
    # ------------------------------------------------------------------
    for N in BATCH_SIZES:
        keys = jax.random.split(rng, N)
        states = jax.vmap(_make_state_jnp)(keys)   # tuple of (q, qd, qdd, tau)
        q_batch, qd_batch, qdd_batch, tau_batch = states

        # Vmapped JITs
        _batch_id_fn   = jax.jit(jax.vmap(_id_jit))
        _batch_fd_fn   = jax.jit(jax.vmap(_fd_jit))
        _batch_crba_fn = jax.jit(jax.vmap(_crba_jit))
        _batch_minv_fn = jax.jit(jax.vmap(_minv_jit))

        # Compile + warmup each batched algo
        _jit_and_warmup(_batch_id_fn,   (q_batch, qd_batch, qdd_batch))
        _jit_and_warmup(_batch_fd_fn,   (q_batch, qd_batch, tau_batch))
        _jit_and_warmup(_batch_crba_fn, (q_batch,))
        _jit_and_warmup(_batch_minv_fn, (q_batch,))

        n_batch_iters = max(10, TEST_ITERS // 5)

        # WITH MEMORY: pre-build a wrapper that takes numpy args (transfer in timed region).
        def _make_np_batch():
            return (
                np.random.randn(N, n).astype(np.float32),
                np.random.randn(N, n).astype(np.float32),
                np.random.randn(N, n).astype(np.float32),
            )
        def _make_np_q():
            return (np.random.randn(N, n).astype(np.float32),)

        # warmup the with-mem wrappers (jit happens on first call with the numpy input shapes)
        sample_np = _make_np_batch()
        sample_np_q = _make_np_q()
        _jit_and_warmup(lambda qs, vs, ds: _batch_id_fn(jnp.array(qs), jnp.array(vs), jnp.array(ds)),
                        sample_np)
        _jit_and_warmup(lambda qs, vs, ts: _batch_fd_fn(jnp.array(qs), jnp.array(vs), jnp.array(ts)),
                        sample_np)
        _jit_and_warmup(lambda qs: _batch_crba_fn(jnp.array(qs)), sample_np_q)
        _jit_and_warmup(lambda qs: _batch_minv_fn(jnp.array(qs)), sample_np_q)

        for label, _fn, make_args in [
            ("ID",   lambda qs, vs, ds: _batch_id_fn(jnp.array(qs), jnp.array(vs), jnp.array(ds)),
                _make_np_batch),
            ("FD",   lambda qs, vs, ts: _batch_fd_fn(jnp.array(qs), jnp.array(vs), jnp.array(ts)),
                _make_np_batch),
            ("CRBA", lambda qs: _batch_crba_fn(jnp.array(qs)), _make_np_q),
            ("MINV", lambda qs: _batch_minv_fn(jnp.array(qs)), _make_np_q),
        ]:
            try:
                wm = _time_with_mem(_fn, make_args, n_iters=n_batch_iters)
                _print_stats(f"{label} WITH MEMORY", N, wm)
            except Exception as e:
                print(f"# [N:{N}] {label} WITH MEMORY skipped: {e}", file=sys.stderr)

        # COMPUTE ONLY: args already on device
        for label, _fn, args in [
            ("ID",   _batch_id_fn,   (q_batch, qd_batch, qdd_batch)),
            ("FD",   _batch_fd_fn,   (q_batch, qd_batch, tau_batch)),
            ("CRBA", _batch_crba_fn, (q_batch,)),
            ("MINV", _batch_minv_fn, (q_batch,)),
        ]:
            try:
                co = _time_device(_fn, *args, n_iters=n_batch_iters)
                _print_stats(f"{label} COMPUTE ONLY", N, co)
            except Exception as e:
                print(f"# [N:{N}] {label} COMPUTE ONLY skipped: {e}", file=sys.stderr)


if __name__ == "__main__":
    main()
