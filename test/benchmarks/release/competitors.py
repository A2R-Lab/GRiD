"""Release adapters: selected outputs, matched URDF, explicit full-call copies.

The installed simulator is loaded from the same stripped URDF, not a similarly
named Menagerie robot. Collision geometry is excluded from this dynamics study.
"""
from __future__ import annotations
from pathlib import Path
import xml.etree.ElementTree as ET
import numpy as np
from .protocol import digest


def mujoco_model(fixture, directory):
    import mujoco
    tree = ET.parse(fixture.urdf)
    root = tree.getroot()
    for link in root.findall("link"):
        for tag in ("visual", "collision"):
            for element in list(link.findall(tag)):
                link.remove(element)
    mj = root.find("mujoco")
    if mj is None:
        mj = ET.SubElement(root, "mujoco")
    compiler = mj.find("compiler")
    if compiler is None:
        compiler = ET.SubElement(mj, "compiler")
    compiler.set("fusestatic", "false")
    compiler.set("discardvisual", "true")
    model = mujoco.MjModel.from_xml_string(ET.tostring(root, encoding="unicode"))
    path = Path(directory) / f"{fixture.spec.robot_id}.resolved.xml"
    mujoco.mj_saveLastXML(str(path), model)
    resolved = ET.parse(path)
    if fixture.base == "floating":
        body = resolved.getroot().find("worldbody/body")
        if body is None:
            raise ValueError("URDF import produced no root body")
        ET.SubElement(body, "freejoint", name="release_floating_root")
    resolved.write(path)
    model = mujoco.MjModel.from_xml_path(str(path))
    # Unconstrained rigid dynamics: no contacts, limits, springs or friction.
    model.opt.disableflags |= int(mujoco.mjtDisableBit.mjDSBL_CONSTRAINT)
    model.geom_contype[:] = 0
    model.geom_conaffinity[:] = 0
    model.dof_damping[:] = 0
    model.dof_frictionloss[:] = 0
    model.dof_armature[:] = 0
    model.jnt_stiffness[:] = 0
    model.opt.gravity[:] = [0, 0, -9.81]
    if (model.nq, model.nv) != (fixture.nq, fixture.nv):
        raise ValueError(f"Model mismatch: MuJoCo {(model.nq, model.nv)} vs GRiD {(fixture.nq, fixture.nv)}")
    target_joint = root.find(f"joint[@name='{fixture.target}']")
    if target_joint is None:
        raise ValueError(f"Unknown target joint {fixture.target}")
    body_name = target_joint.find("child").get("link")
    body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, body_name)
    if body_id < 0:
        raise ValueError(f"Target body missing after import: {body_name}")
    names = fixture.project.actuated_joint_names
    offset_q, offset_v = (7, 6) if fixture.base == "floating" else (0, 0)
    qindex, vindex = list(range(offset_q)), list(range(offset_v))
    for name in names:
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        if jid < 0:
            raise ValueError(f"Model mismatch: joint {name} absent")
        qindex.append(int(model.jnt_qposadr[jid]))
        vindex.append(int(model.jnt_dofadr[jid]))
    return model, body_id, np.array(qindex), np.array(vindex), path


# Bias/gravity are inverse dynamics at zero acceleration (and zero velocity) in
# the SIMULATOR's convention; the oracle transport carries the pin-frame
# "qacc = 0 is not frame-invariant" correction (mujoco_convention).
FAMILY = {"nonlinear_effects": "inverse_dynamics", "generalized_gravity": "inverse_dynamics"}


class SimulatorAdapter:
    def __init__(self, backend, operation, fixture, directory):
        import mujoco
        self.backend, self.op, self.f = backend, operation, fixture
        self.family = FAMILY.get(operation, operation)
        self.m, self.body, self.qindex, self.vindex, path = mujoco_model(fixture, directory)
        self.metadata = {"backend": backend, "dtype": "float64" if backend == "mujoco_cpu" else "float32",
            "method": "autodiff" if operation.endswith("_gradient") else
                      "simulator inverse dynamics at zero acceleration" if operation == "nonlinear_effects" else
                      "simulator inverse dynamics at zero velocity and acceleration" if operation == "generalized_gravity" else
                      "simulator CRB + dense mass matrix" if operation == "crba" else "simulator",
            "model_xml_sha256": digest(path), "source_urdf_sha256": digest(fixture.urdf),
            "convention": "MuJoCo tangent space; oracle transported outside timing",
            "constraints": "disabled", "contacts": "disabled", "passive_forces": "disabled",
            "output": "selected generalized vector, tangent Jacobian, or endpoint xyz+RPY pose",
            "precision_note": "MuJoCo CPU installed binary uses mjtNum double" if backend == "mujoco_cpu" else "fp32"}
        self.qs, self.vs, self.ts = self.inputs()
        if backend == "mjx":
            import jax
            import mujoco.mjx as mjx
            if jax.default_backend() != "gpu":
                raise RuntimeError("MJX did not select a GPU")
            self.mx = mjx.put_model(self.m)
            self.dx = mjx.put_data(self.m, mujoco.MjData(self.m))
        elif backend == "mujoco_warp":
            import mujoco_warp as mjw
            self.mx = mjw.put_model(self.m)

    def inputs(self):
        from RBDReference.equivalents import mujoco_convention as c
        qs, vs, ts = [], [], []
        fd = self.op in {"forward_dynamics", "forward_dynamics_gradient"}
        zero_v = self.op == "generalized_gravity"
        zero_a = self.op in {"nonlinear_effects", "generalized_gravity"}
        for q, v, a, u in zip(self.f.q, self.f.v, self.f.a, self.f.u):
            v = np.zeros_like(v) if zero_v else v
            q1, v1, t1 = q.copy(), v.copy(), (u if fd else a).copy()
            if self.f.base == "floating":
                R = c.base_rotation(q)
                q1 = c.q_pin_to_mjx(q)
                v1 = c.v_pin_to_mjx(v, R)
                t1 = c.id_tau_pin_to_mjx(u, R) if fd else c.accel_pin_to_mjx(a, v, R)
            if zero_a:
                t1 = np.zeros_like(t1)   # the simulator's own bias: qacc = 0 in ITS convention
            qm, vm, tm = np.empty(self.f.nq), np.empty(self.f.nv), np.empty(self.f.nv)
            qm[self.qindex], vm[self.vindex], tm[self.vindex] = q1, v1, t1
            qs.append(qm); vs.append(vm); ts.append(tm)
        dtype = np.float64 if self.backend == "mujoco_cpu" else np.float32
        return tuple(np.ascontiguousarray(x, dtype=dtype) for x in (qs, vs, ts))

    def expected(self, batch):
        from RBDReference.equivalents import mujoco_convention as c
        f, op = self.f, self.op
        out = f.expected(op, batch)
        if f.base != "floating" or op == "end_effector_pose":
            return out
        converted = []
        for i in range(batch):
            q, v, a, u = [np.asarray(x[i], np.float64) for x in (f.q, f.v, f.a, f.u)]
            R = c.base_rotation(q)
            if op == "inverse_dynamics":
                value = c.id_tau_pin_to_mjx(out[i], R)
            elif op == "nonlinear_effects":
                value = c.nonlinear_effects_pin_to_mjx(out[i], f.oracle.crba(q), v, R)
            elif op == "generalized_gravity":
                value = c.nonlinear_effects_pin_to_mjx(out[i], f.oracle.crba(q), np.zeros_like(v), R)
            elif op == "crba":
                value = c.mass_matrix_pin_to_mjx(out[i], R)
            elif op == "forward_dynamics":
                value = c.accel_pin_to_mjx(out[i], v, R)
            elif op == "minv":
                value = c.minv_pin_to_mjx(out[i], R)
            elif op == "inverse_dynamics_gradient":
                M = f.oracle.crba(q)
                tau = f.oracle.inverse_dynamics(q, v, a)
                value = np.concatenate(c.id_gradient_pin_to_mjx(out[i, :, :f.nv], out[i, :, f.nv:], M, tau, v, a, R), axis=-1)
            else:
                mi = f.oracle.minv(q)
                qdd = f.oracle.forward_dynamics(q, v, u)
                value = np.concatenate(c.fd_gradient_pin_to_mjx(out[i, :, :f.nv], out[i, :, f.nv:], mi, qdd, v, u, R), axis=-1)
            converted.append(value)
        return np.stack(converted)

    def normalize(self, result):
        a = np.asarray(self.download(result))
        if self.op == "end_effector_pose":
            return a
        if self.op.endswith("_gradient"):
            cols = np.r_[self.vindex, self.vindex + self.f.nv]
            return a[:, self.vindex][:, :, cols]
        if self.op in {"minv", "crba"}:
            return a[:, self.vindex][:, :, self.vindex]
        return a[:, self.vindex]

    def prepare(self, batch):
        import mujoco
        host = tuple(a[:batch] for a in (self.qs, self.vs, self.ts))
        op = self.op
        if self.backend == "mujoco_cpu":
            datas = [mujoco.MjData(self.m) for _ in range(batch)]
            def call():
                values = []
                for i, d in enumerate(datas):
                    d.qpos[:], d.qvel[:] = host[0][i], host[1][i]
                    if op in {"inverse_dynamics", "nonlinear_effects", "generalized_gravity"}:
                        d.qacc[:] = host[2][i]
                        mujoco.mj_inverse(self.m, d)
                        out = d.qfrc_inverse
                    elif op == "crba":
                        # MuJoCo >= 3.3: mj_crb fills the CSR M; mj_makeM builds the
                        # legacy sparse qM that mj_fullM expands (mj_forward does both).
                        mujoco.mj_kinematics(self.m, d)
                        mujoco.mj_comPos(self.m, d)
                        mujoco.mj_crb(self.m, d)
                        mujoco.mj_makeM(self.m, d)
                        dense = np.empty((self.m.nv, self.m.nv))
                        mujoco.mj_fullM(self.m, dense, d.qM)
                        out = dense
                    elif op == "forward_dynamics":
                        d.qfrc_applied[:] = host[2][i]
                        mujoco.mj_forward(self.m, d)
                        out = d.qacc
                    elif op == "minv":
                        mujoco.mj_forward(self.m, d)
                        dense = np.empty((self.m.nv, self.m.nv))
                        mujoco.mj_fullM(self.m, dense, d.qM)
                        out = np.linalg.inv(dense)
                    elif op == "end_effector_pose":
                        mujoco.mj_kinematics(self.m, d)
                        r = d.xmat[self.body].reshape(3,3)
                        out = np.r_[d.xpos[self.body], np.arctan2(r[2,1], r[2,2]),
                                    np.arctan2(-r[2,0], np.hypot(r[2,2], r[2,1])), np.arctan2(r[1,0], r[0,0])]
                    else:
                        raise NotImplementedError(op)
                    values.append(out.copy())
                return np.stack(values)
            self.host, self.resident = call, None
            self.sync, self.download = lambda x: None, lambda x: x
        elif self.backend == "mjx":
            import jax
            import jax.numpy as jnp
            import mujoco.mjx as mjx
            def value(q, v, third):
                if op == "end_effector_pose":
                    d = mjx.kinematics(self.mx, self.dx.replace(qpos=q))
                    r = d.xmat[self.body].reshape(3,3)
                    return jnp.concatenate((d.xpos[self.body], jnp.array([jnp.arctan2(r[2,1], r[2,2]),
                        jnp.arctan2(-r[2,0], jnp.hypot(r[2,2], r[2,1])), jnp.arctan2(r[1,0], r[0,0])])))
                if op in {"inverse_dynamics", "inverse_dynamics_gradient", "nonlinear_effects", "generalized_gravity"}:
                    return mjx.inverse(self.mx, self.dx.replace(qpos=q, qvel=v, qacc=third)).qfrc_inverse
                if op == "crba":
                    d = mjx.kinematics(self.mx, self.dx.replace(qpos=q))
                    d = mjx.com_pos(self.mx, d)
                    d = mjx.crb(self.mx, d)
                    return mjx.full_m(self.mx, d)
                return mjx.forward(self.mx, self.dx.replace(qpos=q, qvel=v, qfrc_applied=third)).qacc
            def retract(q, delta):
                if self.f.base == "fixed":
                    return q + delta
                # Exact first derivative of the free-joint MuJoCo retract at 0;
                # normalized quaternion product avoids an axis-angle 0/0 in AD.
                w, xyz = q[3], q[4:7]
                d = delta[3:6] * .5
                quat = jnp.concatenate((jnp.array([w-jnp.dot(xyz, d)]), xyz+w*d+jnp.cross(xyz, d)))
                quat = quat / jnp.linalg.norm(quat)
                return jnp.concatenate((q[:3]+delta[:3], quat, q[7:]+delta[6:]))
            def grad(q, v, third):
                dq = jax.jacfwd(lambda delta: value(retract(q, delta), v, third))(jnp.zeros(self.m.nv, q.dtype))
                dv = jax.jacfwd(lambda vel: value(q, vel, third))(v)
                return jnp.concatenate((dq, dv), axis=-1)
            fn = jax.jit(jax.vmap(grad if op.endswith("_gradient") else value))
            dev = tuple(jax.device_put(a) for a in host)
            jax.block_until_ready(dev)
            self.resident = lambda: fn(*dev)
            self.host = lambda: jax.device_get(fn(*(jax.device_put(a.copy()) for a in host)))
            self.sync, self.download = jax.block_until_ready, jax.device_get
        else:
            import warp as wp
            import mujoco_warp as mjw
            d = mjw.make_data(self.m, nworld=batch)
            if op == "end_effector_pose":
                from .warp_kernels import endpoint_pose
                pose = wp.empty((batch,6), dtype=wp.float32)
            inverse_like = op in {"inverse_dynamics", "nonlinear_effects", "generalized_gravity"}
            field = "qfrc_inverse" if inverse_like else "qacc"
            if op == "crba" and getattr(self.mx, "is_sparse", False):
                raise NotImplementedError("Warp model is sparse; the dense qM path needs a dense model")
            def upload():
                d.qpos.assign(host[0]); d.qvel.assign(host[1])
                (d.qacc if inverse_like else d.qfrc_applied).assign(host[2])
            def launch():
                if op == "end_effector_pose":
                    mjw.kinematics(self.mx, d)
                    wp.launch(endpoint_pose, dim=batch, inputs=[d.xpos, d.xmat, self.body, pose])
                    return pose
                if op == "crba":
                    # dense CRB: kinematics + com_pos + crb fill d.qM (nworld, nv, nv)
                    mjw.kinematics(self.mx, d); mjw.com_pos(self.mx, d); mjw.crb(self.mx, d)
                    return d.qM
                (mjw.inverse if inverse_like else mjw.forward)(self.mx, d)
                return getattr(d, field)
            # One eager call loads/compiles every kernel, then the call is
            # captured into a CUDA graph and replayed — mujoco_warp's own
            # benchmark path. Eager launches are Python-launch-bound (the
            # first collector saw a flat 1.2 ms for g1 at every batch size)
            # and are kept only as a disclosed secondary series.
            upload(); launch(); wp.synchronize()
            with wp.ScopedCapture() as capture:
                launch()
            graph = capture.graph
            def replay():
                wp.capture_launch(graph)
                return pose if op == "end_effector_pose" else getattr(d, field)
            def full():
                upload()
                return replay().numpy()
            self.resident, self.host = replay, full
            self.resident_eager = launch
            self.sync = lambda result: wp.synchronize()
            self.download = lambda a: a if isinstance(a, np.ndarray) else a.numpy()
            self.metadata["launch_policy"] = "captured CUDA graph replay (resident and full-call); eager launches recorded as resident_eager"

    def extra_timings(self, warmups, iterations, warm_seconds=0.0):
        if self.backend != "mujoco_warp":
            return {}
        from .protocol import timed
        return {"resident_eager": timed(self.resident_eager, self.sync, warmups, iterations, warm_seconds)}


class TensorAdapter:
    def __init__(self, backend, operation, fixture, directory):
        self.backend, self.op, self.f = backend, operation, fixture
        self.metadata = {"backend": backend, "dtype": "float32", "method": "library dynamics",
                         "source_urdf_sha256": digest(fixture.urdf)}
        if backend == "bard":
            import bard
            import torch
            torch.backends.cuda.matmul.allow_tf32 = False
            if not torch.cuda.is_available():
                raise RuntimeError("BARD CUDA unavailable")
            self.model = bard.build_model_from_urdf(fixture.urdf, floating_base=fixture.base == "floating", dtype=torch.float32, device="cuda")
            if (self.model.nq, self.model.nv) != (fixture.nq, fixture.nv):
                raise ValueError("BARD model dimensions differ")
        else:
            import jax
            from frax.core.robot import Robot
            if jax.default_backend() != "gpu":
                raise RuntimeError("Frax JAX CUDA unavailable")
            self.model = Robot(fixture.urdf, add_floating_base=False)

    def prepare(self, batch):
        f = self.f
        zero_v = self.op == "generalized_gravity"
        third = f.u[:batch] if self.op == "forward_dynamics" else np.zeros_like(f.a[:batch]) if self.op in {"nonlinear_effects", "generalized_gravity"} else f.a[:batch]
        args = (f.q[:batch], np.zeros_like(f.v[:batch]) if zero_v else f.v[:batch], third)
        if self.backend == "bard":
            import bard
            import torch
            # BARD uses scalar-first quaternion, but the same local linear /
            # angular velocity, acceleration and force convention as Pinocchio.
            if self.f.base == "floating":
                q = args[0].copy()
                q[:, 3:7] = q[:, [6,3,4,5]]
                args = (q,*args[1:])
            data = bard.create_data(self.model, max_batch_size=batch)
            gravity = torch.tensor([0, 0, -9.81], dtype=torch.float32, device="cuda")
            def fn(q, v, t):
                with torch.no_grad():
                    bard.update_kinematics(self.model, data, q, v)
                    if self.op == "crba":
                        return bard.crba(self.model, data)
                    if self.op == "forward_dynamics":
                        return bard.aba(self.model, data, t, gravity=gravity)
                    return bard.rnea(self.model, data, t, gravity=gravity)
            dev = tuple(torch.from_numpy(a).to("cuda") for a in args)
            torch.cuda.synchronize()
            self.resident = lambda: fn(*dev)
            self.host = lambda: fn(*(torch.from_numpy(a).to("cuda") for a in args)).cpu().numpy()
            self.sync = lambda out: torch.cuda.synchronize()
            self.download = lambda out: out if isinstance(out, np.ndarray) else out.detach().cpu().numpy()
        else:
            import jax
            import jax.numpy as jnp
            model = self.model
            if self.op in {"inverse_dynamics", "nonlinear_effects", "generalized_gravity"}:
                # Unlike forward_dynamics, rnea(None gravity) means ZERO gravity.
                gravity_accel = jnp.array([0,0,9.81,0,0,0], dtype=jnp.float32)
                fn = jax.jit(jax.vmap(lambda q, v, a: model.rnea(q, v, a, gravity_accel, None)))
            elif self.op == "forward_dynamics":
                fn = jax.jit(jax.vmap(lambda q, v, u: model.forward_dynamics(q, v, u, None)))
            elif self.op == "crba":
                fn = jax.jit(jax.vmap(lambda q, v, a: model.mass_matrix(q)))
            else:
                fn = jax.jit(jax.vmap(lambda q, v, a: model.mass_matrix_inverse(model.mass_matrix(q))))
            dev = tuple(jax.device_put(a) for a in args)
            jax.block_until_ready(dev)
            self.resident = lambda: fn(*dev)
            self.host = lambda: jax.device_get(fn(*(jax.device_put(a.copy()) for a in args)))
            self.sync, self.download = jax.block_until_ready, jax.device_get

    def normalize(self, out):
        return self.download(out)


def make_adapter(backend, operation, fixture, directory, cpu_threads=1):
    if backend in {"mjx", "mujoco_warp", "mujoco_cpu"}:
        return SimulatorAdapter(backend, operation, fixture, directory)
    if backend in {"bard", "frax"}:
        return TensorAdapter(backend, operation, fixture, directory)
    from .pin_adapter import PinAdapter
    return PinAdapter(operation, fixture, directory, cpu_threads=cpu_threads, plain=(backend == "pinocchio_plain"))
