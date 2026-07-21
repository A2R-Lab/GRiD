"""grid-rbd quickstart: register an iiwa14 and exercise every method.

Demonstrates the register-then-run UX:

    1. register_robot() generates grid.cuh, compiles to .so, caches under
       ~/.cache/grid-rbd/. First run takes ~30-60s for iiwa14; subsequent
       runs hit the cache and start in <1s.

    2. handle.<method>(q, qd, ...) runs on the GPU and returns numpy
       arrays. Every method is batched on axis 0.

Run with:
    python bindings/examples/quickstart_iiwa14.py

Requires:
    pip install -e .   # in your virtualenv
    nvcc on PATH             # CUDA Toolkit installed
    An iiwa14 URDF           # see _URDF below; or pass --urdf <path>
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np


_DEFAULT_URDF = (
    Path.home()
    / ".cache/robot_descriptions/drake/manipulation/models/iiwa_description/urdf/iiwa14_primitive_collision.urdf"
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--urdf", default=str(_DEFAULT_URDF),
                        help=f"Path to URDF (default: {_DEFAULT_URDF})")
    parser.add_argument("--batch", type=int, default=4,
                        help="Batch size for the demo calls")
    parser.add_argument("--force-rebuild", action="store_true",
                        help="Ignore the cache and regenerate + recompile")
    args = parser.parse_args()

    urdf = Path(args.urdf).expanduser()
    if not urdf.exists():
        sys.exit(f"URDF not found: {urdf}")

    import grid_rbd  # noqa: F401 (deferred until URDF check)

    print(f"grid_rbd v{grid_rbd.__version__}")
    print(f"Cache dir: {grid_rbd.default_cache_dir()}")

    # ─── 1. Register ────────────────────────────────────────────────────
    t0 = time.time()
    handle = grid_rbd.register_robot(
        name="iiwa14_quickstart",
        urdf_path=str(urdf),
        floating_base=False,
        max_batch_size=max(args.batch, 32),
        force_rebuild=args.force_rebuild,
    )
    print(f"register_robot: {time.time() - t0:.1f}s")
    print(f"  {handle}")

    # ─── 2. Run ─────────────────────────────────────────────────────────
    rng = np.random.default_rng(0)
    NJ = handle.num_joints
    B = args.batch
    q  = rng.standard_normal((B, NJ)).astype(np.float32)
    qd = rng.standard_normal((B, NJ)).astype(np.float32)
    u  = rng.standard_normal((B, NJ)).astype(np.float32)

    print("\nCalling every method on a random batch:")
    def show(name, out, *more):
        if isinstance(out, tuple):
            shapes = ", ".join(f"{o.shape}" for o in out)
            print(f"  {name:30s} tuple of {len(out)} arrays, shapes: ({shapes})")
        else:
            print(f"  {name:30s} {out.shape}  first row [:5]: {out.reshape(B, -1)[0, :5]}")

    show("inverse_dynamics(q, qd)",                       handle.inverse_dynamics(q, qd))
    show("minv(q)",                           handle.minv(q))
    show("forward_dynamics(q, qd, u)",        handle.forward_dynamics(q, qd, u))
    show("aba(q, qd, u)",                     handle.aba(q, qd, u))
    show("crba(q)",                           handle.crba(q))
    show("end_effector_pose(q)",              handle.end_effector_pose(q))
    show("end_effector_pose_gradient(q)",     handle.end_effector_pose_gradient(q))
    show("end_effector_pose_hessian(q)",      handle.end_effector_pose_hessian(q))
    show("inverse_dynamics_gradient(q, qd)",                  handle.inverse_dynamics_gradient(q, qd))
    show("forward_dynamics_gradient(q, qd, u)",   handle.forward_dynamics_gradient(q, qd, u))
    show("idsva_so(q, qd)",                   handle.idsva_so(q, qd))
    show("fdsva_so(q, qd, u)",                handle.fdsva_so(q, qd, u))

    # ─── 3. Demonstrate the fast path ────────────────────────────────────
    print("\nHot-path timing (median of 100 calls on the registered handle):")
    for name, call in [
        ("inverse_dynamics",             lambda: handle.inverse_dynamics(q, qd)),
        ("forward_dynamics", lambda: handle.forward_dynamics(q, qd, u)),
        ("crba",             lambda: handle.crba(q)),
    ]:
        # Warmup
        for _ in range(5): call()
        times = []
        for _ in range(100):
            t = time.perf_counter()
            call()
            times.append((time.perf_counter() - t) * 1e6)
        print(f"  {name:20s} median={np.median(times):6.1f} us  (B={B})")


if __name__ == "__main__":
    main()
