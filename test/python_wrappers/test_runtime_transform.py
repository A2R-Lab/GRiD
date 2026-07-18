"""Validation for the runtime_transform feature (mutable joint-frame <origin>).

runtime_transform lets a caller change link origin (xyz+rpy) parameters at
runtime WITHOUT recompiling the generated CUDA, by mirroring the runtime_inertia
path: the raw [x,y,z,r,p,y] per joint live in a mutable d_transform_params table,
each joint's constant Xfixed 6x6 is rebuilt on-device once per launch, and the
hot sin/cos(q) loop reads that scratch instead of inline origin literals.

Three checks (run as a script: exits 0 on success, 1 on failure):
  CHECK1  default == baked      runtime_transform(original URDF params) ~= plain baked .so.
          Tolerance is RELATIVE (~float32): the on-device sincos(rpy) rebuild does
          NOT reproduce the baked sympy-FOLDED origin constants bit-for-bit, but the
          relative error is float32 noise (~1e-7). Not bit-identical; float-identical.
  CHECK2  perturb == recodegen  set_transform_params(perturbed) == a freshly codegen'd
          robot whose URDF origins ARE those perturbed values (proves mutation == recompile).
  CHECK3  thread-invariance     bit-identical across {1,32,256} threads (single-block core).

Requires a CUDA GPU + the grid_rbd binding build toolchain (nvcc). iiwa14 fixed base.
"""
import contextlib
import io
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "bindings"))
import grid_rbd  # noqa: E402
from config import robot_urdf, ROBOT_ASSETS_DIR  # noqa: E402

IIWA = str(robot_urdf("iiwa14"))
CHECK1_REL_TOL = 1e-4   # float32 rebuild-vs-folded-constant noise (observed ~1e-7)
CHECK2_REL_TOL = 5e-3
CHECK3_ABS_TOL = 1e-5


def _reg(name, **kw):
    with contextlib.redirect_stdout(io.StringIO()):
        return grid_rbd.register_robot(name=name, urdf_path=IIWA, floating_base=False,
                                       max_batch_size=8, **kw)


def _relerr(a, b):
    a, b = np.asarray(a), np.asarray(b)
    return float(np.max(np.abs(a - b)) / max(1.0, float(np.max(np.abs(b)))))


def main():
    rt = _reg("iiwa_rt_xform_test", runtime_transform=True, force_rebuild=True)
    baked = _reg("iiwa_baked_xform_test")
    nj = rt.num_joints
    rng = np.random.default_rng(1)
    q = rng.standard_normal((4, nj)).astype(np.float32)
    qd = rng.standard_normal((4, nj)).astype(np.float32)
    u = rng.standard_normal((4, nj)).astype(np.float32)

    # CHECK1 — runtime_transform with original params == baked (relative/float32).
    rt.set_transform_params(rt.transform_params)
    ok1 = True
    print("=== CHECK1: runtime_transform(original params) ~= baked (relative) ===")
    for name, a, b in [("inverse_dynamics", rt.inverse_dynamics(q, qd), baked.inverse_dynamics(q, qd)),
                       ("crba", rt.crba(q), baked.crba(q)),
                       ("minv", rt.minv(q), baked.minv(q)),
                       ("forward_dynamics", rt.forward_dynamics(q, qd, u), baked.forward_dynamics(q, qd, u))]:
        e = _relerr(a, b)
        ok1 &= e < CHECK1_REL_TOL
        print(f"  {name:18s} relerr={e:.3e}  {'PASS' if e < CHECK1_REL_TOL else 'FAIL'}")

    # CHECK2 — perturbed params == a recodegen'd robot baked with those origins.
    print("=== CHECK2: set_transform_params(perturbed) == recodegen(perturbed URDF) ===")
    baked_params = rt.transform_params.astype(np.float64)
    rng2 = np.random.default_rng(7)
    perturbed = baked_params.copy()
    perturbed[:, 0:3] += 0.03 * rng2.standard_normal((nj, 3))
    perturbed[:, 3:6] += 0.05 * rng2.standard_normal((nj, 3))
    with contextlib.redirect_stdout(io.StringIO()):
        robot = __import__("URDFParser", fromlist=["URDFParser"]).URDFParser().parse(IIWA, floating_base=False)
    names = [robot.get_joint_by_id(j).get_name() for j in range(robot.get_num_joints())]
    tree = ET.parse(IIWA)
    pert_by_name = {names[j]: perturbed[j] for j in range(nj)}
    for joint in tree.getroot().findall("joint"):
        if joint.get("name") in pert_by_name:
            x, y, z, r, p, yw = pert_by_name[joint.get("name")]
            origin = joint.find("origin") or ET.SubElement(joint, "origin")
            origin.set("xyz", f"{x} {y} {z}")
            origin.set("rpy", f"{r} {p} {yw}")
    pert_urdf = str(ROBOT_ASSETS_DIR / ".iiwa14_perturbed_rt_test.urdf")
    tree.write(pert_urdf)
    with contextlib.redirect_stdout(io.StringIO()):
        recodegen = grid_rbd.register_robot(name="iiwa_recodegen_pert_test", urdf_path=pert_urdf,
                                            floating_base=False, max_batch_size=8, force_rebuild=True)
    rt.set_transform_params(perturbed.astype(np.float32))
    ok2 = True
    for name, a, b in [("inverse_dynamics", rt.inverse_dynamics(q, qd), recodegen.inverse_dynamics(q, qd)),
                       ("crba", rt.crba(q), recodegen.crba(q)),
                       ("forward_dynamics", rt.forward_dynamics(q, qd, u), recodegen.forward_dynamics(q, qd, u))]:
        e = _relerr(a, b)
        ok2 &= e < CHECK2_REL_TOL
        print(f"  {name:18s} relerr={e:.3e}  {'PASS' if e < CHECK2_REL_TOL else 'FAIL'}")
    Path(pert_urdf).unlink(missing_ok=True)

    # CHECK3 — thread-invariance on the runtime_transform path.
    print("=== CHECK3: thread-invariance {1,32,256} ===")
    rt.set_transform_params(perturbed.astype(np.float32))
    ref, ok3 = None, True
    for nthreads in (1, 32, 256):
        rt.set_threads_per_block(nthreads)
        out = np.asarray(rt.inverse_dynamics(q, qd))
        if ref is None:
            ref = out
        else:
            d = float(np.max(np.abs(out - ref)))
            ok3 &= d < CHECK3_ABS_TOL
            print(f"  threads={nthreads:4d} vs 1  maxabsdiff={d:.3e}  {'PASS' if d < CHECK3_ABS_TOL else 'FAIL'}")

    print(f"\nSUMMARY  CHECK1={'PASS' if ok1 else 'FAIL'}  CHECK2={'PASS' if ok2 else 'FAIL'}  CHECK3={'PASS' if ok3 else 'FAIL'}")
    sys.exit(0 if (ok1 and ok2 and ok3) else 1)


if __name__ == "__main__":
    main()
