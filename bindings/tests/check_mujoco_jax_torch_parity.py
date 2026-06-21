"""Comprehensive jax + torch mjx parity: every wired mjx method's jax.mujoco.X and
torch.mujoco.X must match the proven numpy ref.mujoco.X oracle (fp32 tolerance).
Builds go2-floating ONCE (force_rebuild) so the .so carries all new mjx handlers."""
import numpy as np
from pathlib import Path

_REPO = Path(__file__).resolve().parents[2]
_GO2 = _REPO / "robot_assets" / "go2.urdf"
TOL = 2e-3


def _arr(x):
    import numpy as _np
    try:
        import torch
        if isinstance(x, torch.Tensor):
            return x.detach().cpu().numpy()
    except Exception:
        pass
    return _np.asarray(x)


def _cmp(name, ref, got, results):
    ref = _arr(ref).astype(np.float64)
    got = _arr(got).astype(np.float64)
    if ref.shape != got.shape:
        results.append((name, "SHAPE", f"{ref.shape} vs {got.shape}"))
        return
    d = np.max(np.abs(ref - got)) if ref.size else 0.0
    results.append((name, "OK" if d < TOL else "FAIL", f"{d:.2e}"))


def main():
    import grid_rbd.jax as gjax
    import grid_rbd.torch as gtorch
    import torch
    import jax.numpy as jnp

    jh = gjax.register_robot("go2_mjx_all_val", str(_GO2), floating_base=True,
                             force_rebuild=True, output_convention="mujoco")
    # NOTE: nothing is skipped here. fdsva_so mjx was un-skipped 2026-06-21 (jax+torch match
    # the numpy mujoco oracle on go2-floating, max |Δ| ~9e-5 fp32). The old ">48KB opt-in gap"
    # reason was stale (go2-floating fdsva_so uses 85KB smem and launches fine via
    # init_grid_kernel_attrs). The LITE/MINIMAL spilled-epilogue concern is a separate
    # perf-phase item; the default tier exercised here does not spill.
    th = gtorch.get_robot("go2_mjx_all_val")
    ref = jh._base  # numpy handle (proven mjx oracle)
    nq, nv, nb = jh.num_joints, jh.num_vel, jh.num_bodies
    print(f"built go2-floating: nq={nq} nv={nv} nb={nb}")

    rng = np.random.default_rng(7)
    B = 4
    def mk(width_nv=False):
        a = rng.standard_normal((B, nq)).astype(np.float32)
        a[:, 3:7] /= np.linalg.norm(a[:, 3:7], axis=1, keepdims=True)
        return a
    q = mk(); q[:, 3:7] /= np.linalg.norm(q[:, 3:7], axis=1, keepdims=True)
    qd = rng.standard_normal((B, nq)).astype(np.float32); qd[:, nv:] = 0.0
    qdd = rng.standard_normal((B, nq)).astype(np.float32); qdd[:, nv:] = 0.0
    u = rng.standard_normal((B, nq)).astype(np.float32); u[:, nv:] = 0.0
    jq, jqd, jqdd, ju = map(jnp.asarray, (q, qd, qdd, u))
    tq, tqd, tqdd, tu = (torch.tensor(z, device="cuda") for z in (q, qd, qdd, u))

    R = []
    # ── q-only kinematics ──
    for nm, rf, jf, tf in [
        ("crba",                 lambda: ref.mujoco.crba(q),                 lambda: jh.mujoco.crba(jq),                 lambda: th.mujoco.crba(tq)),
        ("minv",                 lambda: ref.mujoco.minv(q),                 lambda: jh.mujoco.minv(jq),                 lambda: th.mujoco.minv(tq)),
        ("end_effector_pose",    lambda: ref.mujoco.end_effector_pose(q) if hasattr(ref.mujoco,'end_effector_pose') else ref.end_effector_pose(q,_convention='mujoco'),
                                  lambda: jh.mujoco.end_effector_pose(jq),    lambda: th.mujoco.end_effector_pose(tq)),
        ("ee_pose_gradient",     lambda: ref.end_effector_pose_gradient(q, _convention='mujoco'), lambda: jh.mujoco.end_effector_pose_gradient(jq), lambda: th.mujoco.end_effector_pose_gradient(tq)),
        ("ee_pose_hessian",      lambda: ref.end_effector_pose_hessian(q, _convention='mujoco'),  lambda: jh.mujoco.end_effector_pose_hessian(jq),  lambda: th.mujoco.end_effector_pose_hessian(tq)),
    ]:
        try:
            r = rf()
            _cmp(f"jax.{nm}", r, jf(), R)
            _cmp(f"torch.{nm}", r, tf(), R)
        except Exception as e:
            R.append((nm, "ERR", str(e)[:80]))

    # ── dynamics (q, qd, qdd/u) ──
    for nm, rf, jf, tf in [
        ("inverse_dynamics",      lambda: ref.mujoco.inverse_dynamics(q, qd, qdd),  lambda: jh.mujoco.inverse_dynamics(jq, jqd, jqdd),  lambda: th.mujoco.inverse_dynamics(tq, tqd, tqdd)),
        ("forward_dynamics",      lambda: ref.mujoco.forward_dynamics(q, qd, u),    lambda: jh.mujoco.forward_dynamics(jq, jqd, ju),    lambda: th.mujoco.forward_dynamics(tq, tqd, tu)),
        ("aba",                   lambda: ref.mujoco.aba(q, qd, u),                 lambda: jh.mujoco.aba(jq, jqd, ju),                 lambda: th.mujoco.aba(tq, tqd, tu)),
        ("id_gradient",           lambda: ref.inverse_dynamics_gradient(q, qd, qdd, _convention='mujoco'), lambda: jh.mujoco.inverse_dynamics_gradient(jq, jqd, jqdd), lambda: th.mujoco.inverse_dynamics_gradient(tq, tqd, tqdd)),
        ("fd_gradient",           lambda: ref.forward_dynamics_gradient(q, qd, u, _convention='mujoco'),   lambda: jh.mujoco.forward_dynamics_gradient(jq, jqd, ju),   lambda: th.mujoco.forward_dynamics_gradient(tq, tqd, tu)),
        ("idsva_so",              lambda: ref.idsva_so(q, qd, qdd, _convention='mujoco'),  lambda: jh.mujoco.idsva_so(jq, jqd, jqdd),  lambda: th.mujoco.idsva_so(tq, tqd, tqdd)),
        ("fdsva_so",              lambda: ref.fdsva_so(q, qd, u, _convention='mujoco'),    lambda: jh.mujoco.fdsva_so(jq, jqd, ju),    lambda: th.mujoco.fdsva_so(tq, tqd, tu)),
    ]:
        try:
            r = rf()
            # idsva_so/fdsva_so return tuples of 4 tensors
            if isinstance(r, (tuple, list)) or hasattr(r, "_fields"):
                jr, tr = jf(), tf()
                for i in range(len(r)):
                    _cmp(f"jax.{nm}[{i}]", r[i], jr[i], R)
                    _cmp(f"torch.{nm}[{i}]", r[i], tr[i], R)
            else:
                _cmp(f"jax.{nm}", r, jf(), R)
                _cmp(f"torch.{nm}", r, tf(), R)
        except Exception as e:
            R.append((nm, "ERR", str(e)[:80]))

    # ── integrator ──
    for nm, rf, jf, tf in [
        ("integrator",          lambda: ref.integrator(q, qd, u, 0.01, _convention='mujoco'),          lambda: jh.mujoco.integrator(jq, jqd, ju, 0.01),          lambda: th.mujoco.integrator(tq, tqd, tu, 0.01)),
        ("integrator_gradient", lambda: ref.integrator_gradient(q, qd, u, 0.01, _convention='mujoco'), lambda: jh.mujoco.integrator_gradient(jq, jqd, ju, 0.01), lambda: th.mujoco.integrator_gradient(tq, tqd, tu, 0.01)),
    ]:
        try:
            r = rf()
            _cmp(f"jax.{nm}", r, jf(), R)
            _cmp(f"torch.{nm}", r, tf(), R)
        except Exception as e:
            R.append((nm, "ERR", str(e)[:80]))

    print("\n=== RESULTS (ref = numpy mjx oracle) ===")
    nfail = 0
    for name, status, detail in R:
        flag = "" if status == "OK" else "  <<<"
        if status != "OK":
            nfail += 1
        print(f"  {status:5} {name:28} {detail}{flag}")
    print(f"\n{len(R)} checks, {nfail} non-OK")
    if nfail:
        raise SystemExit(1)
    print("ALL JAX + TORCH MJX METHODS MATCH NUMPY ORACLE")


if __name__ == "__main__":
    main()
