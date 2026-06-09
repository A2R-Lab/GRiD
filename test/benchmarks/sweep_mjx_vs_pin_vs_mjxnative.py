"""Perf sweep: GRiD-mjx (A) vs GRiD-pin (B) vs MJX-native (C), all via the JAX surface.

Contestants (per the agreed design):
  A) GRiD mjx-fused   : handle.mujoco.<fn>   (MUJOCO_OUTPUT=true kernel, one launch)
  B) GRiD pin         : handle.<fn>          (MUJOCO_OUTPUT=false; "what if we stayed pin in jax"
                                              -- isolates the mjx epilogue cost vs A)
  C) MJX native       : baselines/mjx/timeMJX.py (subprocess; MuJoCo MJX)

Functions (the MJX-comparable set): inverse_dynamics, forward_dynamics, end_effector_pose,
inverse_dynamics_gradient (= MJX id_du).  Robots: go2 + a humanoid (g1).  Batch: 1..4096.

ISOLATED timing: one (contestant, robot, fn, batch) at a time, GPU otherwise quiet, jit-compiled,
warmed up, median of N reps each block_until_ready'd. Run this alone (no other GPU work).

Usage:
  python test/benchmarks/sweep_mjx_vs_pin_vs_mjxnative.py --robots go2 g1 \
      --batches 1 16 64 256 1024 4096 --reps 50 --out test/benchmarks/results/mjx_sweep_<ts>.json
"""
from __future__ import annotations
import argparse, json, statistics, subprocess, sys, time
from pathlib import Path

_REPO = Path(__file__).resolve().parents[2]
_ASSETS = _REPO / "robot_assets"

# (label, A-callable-name on handle.mujoco, B-callable-name on handle, builds-inputs key)
FUNCS = [
    ("inverse_dynamics",          "inverse_dynamics",          "id"),
    ("forward_dynamics",          "forward_dynamics",          "fd"),
    ("end_effector_pose",         "end_effector_pose",         "q"),
    ("inverse_dynamics_gradient", "inverse_dynamics_gradient", "id"),
]


def _bench_jax(fn, *args, reps, warmup=5):
    """Median wall-clock (ms) of fn(*args), block_until_ready each rep. fn returns jax array(s)."""
    import jax
    for _ in range(warmup):
        jax.block_until_ready(fn(*args))
    ts = []
    for _ in range(reps):
        t0 = time.perf_counter()
        jax.block_until_ready(fn(*args))
        ts.append((time.perf_counter() - t0) * 1e3)
    return statistics.median(ts), min(ts)


def _make_inputs(handle, B, key):
    import jax.numpy as jnp
    import numpy as np
    nq, nv = handle.num_joints, handle.num_vel
    rng = np.random.default_rng(0)
    q = rng.standard_normal((B, nq)).astype(np.float32)
    if handle.floating_base:
        q[:, 3:7] /= np.linalg.norm(q[:, 3:7], axis=1, keepdims=True)
    qd = rng.standard_normal((B, nq)).astype(np.float32); qd[:, nv:] = 0.0
    qdd = rng.standard_normal((B, nq)).astype(np.float32); qdd[:, nv:] = 0.0
    u = rng.standard_normal((B, nq)).astype(np.float32); u[:, nv:] = 0.0
    return {k: jnp.asarray(v) for k, v in dict(q=q, qd=qd, qdd=qdd, u=u).items()}


def _call(handle_method, key, inp):
    q, qd, qdd, u = inp["q"], inp["qd"], inp["qdd"], inp["u"]
    if key == "id":   return lambda: handle_method(q, qd, qdd)
    if key == "fd":   return lambda: handle_method(q, qd, u)
    if key == "q":    return lambda: handle_method(q)
    raise KeyError(key)


def run_grid(robot, batches, reps):
    import grid_rbd.jax as gjax
    urdf = _ASSETS / f"{robot}.urdf"
    # default convention = pinocchio so jh.<fn> is contestant B (pin); jh.mujoco.<fn>
    # is contestant A (forces mjx per-call). Keeps A/B cleanly separated.
    jh = gjax.register_robot(f"{robot}_mjx_sweep", str(urdf), floating_base=True,
                             max_batch_size=max(batches))
    rows = []
    for B in batches:
        inp = _make_inputs(jh, B, None)
        for label, mname, key in FUNCS:
            a_fn = _call(getattr(jh.mujoco, mname), key, inp)   # A: mjx
            b_fn = _call(getattr(jh, mname), key, inp)          # B: pin
            a_med, a_min = _bench_jax(a_fn, reps=reps)
            b_med, b_min = _bench_jax(b_fn, reps=reps)
            rows.append(dict(robot=robot, fn=label, batch=B,
                             A_grid_mjx_ms=a_med, A_min=a_min,
                             B_grid_pin_ms=b_med, B_min=b_min,
                             mjx_epilogue_overhead_pct=100.0 * (a_med - b_med) / b_med if b_med else None))
            print(f"  [{robot:4} {label:26} B={B:5}] A(mjx)={a_med:8.4f}ms  B(pin)={b_med:8.4f}ms  "
                  f"epilogue +{rows[-1]['mjx_epilogue_overhead_pct']:.1f}%")
    return rows


def run_mjx_native(robot, base="floating"):
    """Invoke the existing MJX baseline; returns its parsed per-(fn,batch) us (or None)."""
    runner = _REPO / "test" / "benchmarks" / "baselines" / "mjx" / "run.py"
    try:
        out = subprocess.run([sys.executable, str(runner), "--robot", robot, "--base", base],
                             capture_output=True, text=True, timeout=900)
        return out.stdout
    except Exception as e:
        return f"MJX run failed: {e}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--robots", nargs="+", default=["go2", "g1"])
    ap.add_argument("--batches", nargs="+", type=int, default=[1, 16, 64, 256, 1024, 4096])
    ap.add_argument("--reps", type=int, default=50)
    ap.add_argument("--skip-mjx", action="store_true", help="skip contestant C (MJX native)")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    all_rows = []
    for robot in args.robots:
        print(f"=== GRiD A/B (jax): {robot} ===")
        all_rows += run_grid(robot, args.batches, args.reps)
    mjx_raw = {}
    if not args.skip_mjx:
        for robot in args.robots:
            print(f"=== MJX native (C): {robot} ===")
            mjx_raw[robot] = run_mjx_native(robot)
            print(mjx_raw[robot])

    result = dict(grid=all_rows, mjx_native_raw=mjx_raw,
                  config=dict(batches=args.batches, reps=args.reps, robots=args.robots))
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(result, indent=2))
        print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
