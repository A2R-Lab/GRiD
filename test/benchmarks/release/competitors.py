"""Release adapters: selected outputs, matched URDF, explicit full-call copies.

The installed simulator is loaded from the same stripped URDF, not a similarly
named Menagerie robot. Collision geometry is excluded from this dynamics study.
"""
from __future__ import annotations
from pathlib import Path
import xml.etree.ElementTree as ET
import numpy as np
from .protocol import digest


def mujoco_model(fixture, directory, dense_jacobian=False):
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
    if dense_jacobian:
        # The dense mass-matrix study needs the dense formulation (auto picks
        # sparse for the humanoid); only the crba worker asks for it.
        model.opt.jacobian = int(mujoco.mjtJacobian.mjJAC_DENSE)
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


def rpy_rate_map_inverse(R):
    """E(rpy)^{-1} with omega_world = E(rpy) d[roll, pitch, yaw]/dt for
    R = Rz(yaw) Ry(pitch) Rx(roll): maps a world angular velocity to RPY rates,
    i.e. the rotational rows of d[xyz; rpy]/dv are E^{-1} J_rot(world)."""
    roll = np.arctan2(R[2,1], R[2,2])
    pitch = np.arctan2(-R[2,0], np.hypot(R[2,2], R[2,1]))
    yaw = np.arctan2(R[1,0], R[0,0])
    cp, sp, cy, sy = np.cos(pitch), np.sin(pitch), np.cos(yaw), np.sin(yaw)
    E = np.array([[cp*cy, -sy, 0.], [cp*sy, cy, 0.], [-sp, 0., 1.]])
    return np.linalg.inv(E)


# Bias/gravity are inverse dynamics at zero acceleration (and zero velocity) in
# the SIMULATOR's convention; the oracle transport carries the pin-frame
# "qacc = 0 is not frame-invariant" correction (mujoco_convention).
FAMILY = {"nonlinear_effects": "inverse_dynamics", "generalized_gravity": "inverse_dynamics"}


class SimulatorAdapter:
    def __init__(self, backend, operation, fixture, directory):
        import mujoco
        self.backend, self.op, self.f = backend, operation, fixture
        self.family = FAMILY.get(operation, operation)
        self.m, self.body, self.qindex, self.vindex, path = mujoco_model(
            fixture, directory, dense_jacobian=(backend == "mujoco_warp" and operation == "crba"))
        self.metadata = {"backend": backend, "dtype": "float64" if backend == "mujoco_cpu" else "float32",
            "method": ("autodiff" if backend == "mjx" else "analytic body Jacobian (mj_jac) mapped to xyz+RPY rates")
                      if operation == "end_effector_pose_gradient" else
                      "autodiff" if operation.endswith("_gradient") else
                      "simulator inverse dynamics at zero acceleration" if operation == "nonlinear_effects" else
                      "simulator inverse dynamics at zero velocity and acceleration" if operation == "generalized_gravity" else
                      "simulator CRB + dense mass matrix" if operation == "crba" else "simulator",
            "model_xml_sha256": digest(path), "source_urdf_sha256": digest(fixture.urdf),
            "convention": "MuJoCo tangent space; oracle transported outside timing",
            "constraints": "disabled", "contacts": "disabled", "passive_forces": "disabled",
            "output": "selected generalized vector, tangent Jacobian, or endpoint xyz+RPY pose",
            "precision_note": "MuJoCo CPU installed binary uses mjtNum double" if backend == "mujoco_cpu" else "fp32",
            "jacobian_option": "dense (forced for the mass-matrix study)" if (backend == "mujoco_warp" and operation == "crba") else "model default"}
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
            elif op == "end_effector_pose_gradient":
                value = c.jacobian_pin_to_mjx(out[i], R)
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
        if self.op == "end_effector_pose_gradient":
            return a[:, :, self.vindex]
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
            jacp, jacr = np.empty((3, self.m.nv)), np.empty((3, self.m.nv))
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
                    elif op == "end_effector_pose_gradient":
                        mujoco.mj_kinematics(self.m, d)
                        mujoco.mj_comPos(self.m, d)
                        mujoco.mj_jacBody(self.m, d, jacp, jacr, self.body)
                        out = np.concatenate((jacp, rpy_rate_map_inverse(d.xmat[self.body].reshape(3,3)) @ jacr))
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
                if op in {"end_effector_pose", "end_effector_pose_gradient"}:
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
                if op == "end_effector_pose_gradient":
                    return dq
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
            if op == "end_effector_pose_gradient":
                from .warp_kernels import endpoint_point, endpoint_pose_jacobian
                from mujoco_warp._src import support as mjw_support
                nv = self.m.nv
                point = wp.empty(batch, dtype=wp.vec3)
                bodies = wp.array(np.full(batch, self.body, dtype=np.int32), dtype=wp.int32)
                jacp = wp.zeros((batch, 3, nv), dtype=wp.float32)
                jacr = wp.zeros((batch, 3, nv), dtype=wp.float32)
                pose_jac = wp.empty((batch, 6, nv), dtype=wp.float32)
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
                if op == "end_effector_pose_gradient":
                    mjw.kinematics(self.mx, d); mjw.com_pos(self.mx, d)
                    wp.launch(endpoint_point, dim=batch, inputs=[d.xpos, self.body, point])
                    mjw_support.jac(self.mx, d, jacp, jacr, point, bodies)
                    wp.launch(endpoint_pose_jacobian, dim=(batch, nv), inputs=[d.xmat, self.body, jacp, jacr, pose_jac])
                    return pose_jac
                if op == "crba":
                    # dense CRB: kinematics + com_pos + crb fill d.M (nworld, nv_pad, nv_pad)
                    # on a dense model; normalize() selects the nv x nv block by dof index.
                    mjw.kinematics(self.mx, d); mjw.com_pos(self.mx, d); mjw.crb(self.mx, d)
                    return d.M
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
                return (pose if op == "end_effector_pose" else pose_jac if op == "end_effector_pose_gradient"
                        else d.M if op == "crba" else getattr(d, field))
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
            # Frax is matmul-based (4x4 transform chains, ancestor-mask products).
            # JAX's GPU default runs fp32 matmuls in TF32 (10-bit mantissa), which
            # misses the oracle gate at the 1e-3 level on the floating robots and the
            # gradients. Full fp32 is requested, as the BARD adapter disables TF32.
            jax.config.update("jax_default_matmul_precision", "highest")
            self.metadata["matmul_precision"] = "highest (TF32 disabled; full fp32, the collector worker policy)"
            self.floating = fixture.base == "floating"
            self.model = Robot(fixture.urdf, add_floating_base=self.floating)
            if self.model.num_joints != fixture.nv:
                raise ValueError(f"Frax model dimensions differ: {self.model.num_joints} joints vs nv {fixture.nv}")
            # Frax orders a tree breadth-first (all hips, then thighs, ...); the fixture is
            # depth-first project order. perm[k] = project velocity index of Frax coordinate k.
            names = list(fixture.project.joint_names)
            base = 6 if self.floating else 0
            actuated = names[1:] if self.floating else names
            self.perm = np.array(list(range(base)) + [base + actuated.index(n) for n in self.model.joint_names[base:]])
            if sorted(self.perm) != list(range(fixture.nv)):
                raise ValueError("Frax joint names do not match the fixture")
            self.inv_perm = np.argsort(self.perm)
            self.metadata["joint_order"] = "library breadth-first order; inputs permuted in, outputs permuted back to project order"
            if operation.endswith("_gradient"):
                self.metadata["method"] = "autodiff (jax.jacfwd of the library function)"
            if self.floating:
                self.metadata["convention"] = ("Frax floating base = x/y/z prismatic + intrinsic XYZ Euler revolute joints "
                    "(nq == nv); fixture mapped through v_pin = G(theta) qdot_frax in fp64 outside timing; "
                    "oracle transported by the same map (frax_floating.py)")
            if operation.startswith("end_effector_pose"):
                # The target is a fixed joint the library folds into its parent body;
                # compose the parent joint's world transform with the fixed offset.
                model = fixture.oracle.model
                frame = model.frames[fixture.oracle._resolve_frame_id(fixture.target)]
                parent = getattr(frame, "parentJoint", None)
                parent = frame.parent if parent is None else parent
                joint_name = model.names[parent]
                if joint_name not in self.model.joint_names:
                    raise ValueError(f"Frax has no joint {joint_name!r} for the end-effector frame")
                self.ee_joint = self.model.joint_names.index(joint_name)
                self.ee_offset = np.asarray(frame.placement.homogeneous, np.float64)
                self.metadata["end_effector"] = {"parent_joint": joint_name, "fixed_offset_from_pin_frame": True}

    def _frames(self, batch):
        from .frax_floating import FloatingFrame
        return [FloatingFrame(q, self.f.nv) for q in self.f.q[:batch]]

    def _frax_inputs(self, batch):
        """Frax-native (q, qd, third) arrays for this operation."""
        f, op = self.f, self.op
        zero_v = op == "generalized_gravity"
        zero_a = op in {"nonlinear_effects", "generalized_gravity"}
        fd = op in {"forward_dynamics", "forward_dynamics_gradient"}
        v = np.zeros_like(f.v[:batch]) if zero_v else f.v[:batch]
        third = f.u[:batch] if fd else np.zeros_like(f.a[:batch]) if zero_a else f.a[:batch]
        if not self.floating:
            return tuple(np.ascontiguousarray(x[:, self.perm], dtype=np.float32) for x in (f.q[:batch], v, third))
        qs, vs, ts = [], [], []
        for i, fr in enumerate(self._frames(batch)):
            qs.append(fr.q_frax())
            vs.append(fr.qdot_frax(v[i]))
            if fd:
                ts.append(fr.force_frax(third[i]))
            elif zero_a:
                ts.append(np.zeros(f.nv))
            else:
                ts.append(fr.qddot_frax(v[i], third[i]))
        return tuple(np.ascontiguousarray(np.asarray(x)[:, self.perm], dtype=np.float32) for x in (qs, vs, ts))

    def expected(self, batch):
        f, op = self.f, self.op
        out = f.expected(op, batch)
        if self.backend != "frax" or not self.floating or op == "end_effector_pose":
            return out
        converted = []
        for i, fr in enumerate(self._frames(batch)):
            q, v = np.asarray(f.q[i], np.float64), np.asarray(f.v[i], np.float64)
            if op in {"inverse_dynamics", "generalized_gravity"}:
                value = fr.force_frax(out[i]) if op == "inverse_dynamics" else fr.bias_frax(out[i], f.oracle.crba(q), np.zeros_like(v))
            elif op == "nonlinear_effects":
                value = fr.bias_frax(out[i], f.oracle.crba(q), v)
            elif op == "forward_dynamics":
                value = fr.qddot_frax(v, out[i])
            elif op == "crba":
                value = fr.mass_matrix_frax(out[i])
            elif op == "minv":
                value = fr.minv_frax(out[i])
            elif op == "end_effector_pose_gradient":
                value = fr.pose_jacobian_frax(out[i])
            else:
                raise NotImplementedError(f"no Frax floating-base transport for {op}")
            converted.append(value)
        return np.stack(converted)

    def prepare(self, batch):
        f = self.f
        if self.backend == "bard":
            import bard
            import torch
            zero_v = self.op == "generalized_gravity"
            third = f.u[:batch] if self.op == "forward_dynamics" else np.zeros_like(f.a[:batch]) if self.op in {"nonlinear_effects", "generalized_gravity"} else f.a[:batch]
            args = (f.q[:batch], np.zeros_like(f.v[:batch]) if zero_v else f.v[:batch], third)
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
            args = self._frax_inputs(batch)
            # Unlike forward_dynamics, rnea(None gravity) means ZERO gravity.
            gravity_accel = jnp.array([0,0,9.81,0,0,0], dtype=jnp.float32)
            rnea = lambda q, v, a: model.rnea(q, v, a, gravity_accel, None)
            fdyn = lambda q, v, u: model.forward_dynamics(q, v, u, None)
            if self.op.startswith("end_effector_pose"):
                offset = jnp.asarray(self.ee_offset, dtype=jnp.float32)
                def pose(q):
                    T = model.joint_to_world_transforms(q)[self.ee_joint] @ offset
                    r = T[:3, :3]
                    return jnp.concatenate((T[:3, 3], jnp.array([jnp.arctan2(r[2,1], r[2,2]),
                        jnp.arctan2(-r[2,0], jnp.hypot(r[2,2], r[2,1])), jnp.arctan2(r[1,0], r[0,0])])))
            if self.op in {"inverse_dynamics", "nonlinear_effects", "generalized_gravity"}:
                per_sample = rnea
            elif self.op == "forward_dynamics":
                per_sample = fdyn
            elif self.op == "crba":
                per_sample = lambda q, v, a: model.mass_matrix(q)
            elif self.op == "minv":
                per_sample = lambda q, v, a: model.mass_matrix_inverse(model.mass_matrix(q))
            elif self.op == "end_effector_pose":
                per_sample = lambda q, v, a: pose(q)
            elif self.op == "end_effector_pose_gradient":
                per_sample = lambda q, v, a: jax.jacfwd(pose)(q)
            elif self.op in {"inverse_dynamics_gradient", "forward_dynamics_gradient"}:
                base = rnea if self.op == "inverse_dynamics_gradient" else fdyn
                def per_sample(q, v, t):
                    dq = jax.jacfwd(lambda x: base(x, v, t))(q)
                    dv = jax.jacfwd(lambda x: base(q, x, t))(v)
                    return jnp.concatenate((dq, dv), axis=-1)
            else:
                raise NotImplementedError(self.op)
            fn = jax.jit(jax.vmap(per_sample))
            dev = tuple(jax.device_put(a) for a in args)
            jax.block_until_ready(dev)
            self.resident = lambda: fn(*dev)
            self.host = lambda: jax.device_get(fn(*(jax.device_put(a.copy()) for a in args)))
            self.sync, self.download = jax.block_until_ready, jax.device_get

    def normalize(self, out):
        a = np.asarray(self.download(out))
        if self.backend != "frax":
            return a
        inv, n = self.inv_perm, self.f.nv
        if self.op == "end_effector_pose":
            return a
        if self.op == "end_effector_pose_gradient":
            return a[:, :, inv]
        if self.op in {"inverse_dynamics_gradient", "forward_dynamics_gradient"}:
            return a[:, inv][:, :, np.r_[inv, inv + n]]
        if self.op in {"crba", "minv"}:
            return a[:, inv][:, :, inv]
        return a[:, inv]


class CuroboAdapter:
    """cuRobo's native CUDA RNEA (`curobo._src.robot.dynamics`) on the SAME URDF via
    `KinematicsCfg.from_basic_urdf`; fixed-base only, inverse dynamics only (its
    autograd backward is a VJP, not the full Jacobian this study compares)."""

    def __init__(self, backend, operation, fixture, directory):
        import xml.etree.ElementTree as ET
        import torch
        from curobo._src.robot.kinematics.kinematics_cfg import KinematicsCfg
        from curobo._src.robot.dynamics.dynamics import Dynamics
        from curobo._src.robot.dynamics.dynamics_cfg import DynamicsCfg
        from curobo._src.types.device_cfg import DeviceCfg
        import curobo
        self.backend, self.op, self.f = backend, operation, fixture
        if fixture.base != "fixed" or operation != "inverse_dynamics":
            raise NotImplementedError("cuRobo adapter covers fixed-base inverse dynamics")
        if not torch.cuda.is_available():
            raise RuntimeError("cuRobo CUDA unavailable")
        root = ET.parse(fixture.urdf).getroot()
        children = {j.find("child").get("link") for j in root.findall("joint")}
        base = [l.get("name") for l in root.findall("link") if l.get("name") not in children]
        ee = root.find(f"joint[@name='{fixture.target}']").find("child").get("link")
        device = DeviceCfg()
        cfg = KinematicsCfg.from_basic_urdf(fixture.urdf, base[0], [ee], device_cfg=device)
        names = list(cfg.kinematics_config.joint_names)
        project = list(fixture.project.joint_names)
        if sorted(names) != sorted(project):
            raise ValueError("cuRobo joint set differs from the fixture")
        self.perm = np.array([project.index(n) for n in names])     # cuRobo coordinate k <- project index
        self.inv_perm = np.argsort(self.perm)
        self.dyn = Dynamics(DynamicsCfg(kinematics_config=cfg.kinematics_config, device_cfg=device, gravity=[0., 0., -9.81]))
        self.metadata = {"backend": backend, "dtype": "float32", "method": "library native CUDA RNEA (compute_inverse_dynamics)",
                         "source_urdf_sha256": digest(fixture.urdf), "library_version": getattr(curobo, "__version__", "unknown"),
                         "model": "KinematicsCfg.from_basic_urdf on the shared URDF; base link " + base[0],
                         "joint_order": "library order; inputs permuted in, outputs permuted back to project order"}

    def prepare(self, batch):
        import torch
        from curobo._src.state.state_joint import JointState
        f = self.f
        self.dyn.setup_batch_size(batch_size=batch)
        args = tuple(np.ascontiguousarray(x[:batch][:, self.perm], dtype=np.float32) for x in (f.q, f.v, f.a))
        def fn(q, v, a):
            with torch.no_grad():
                return self.dyn.compute_inverse_dynamics(JointState(position=q, velocity=v, acceleration=a))
        dev = tuple(torch.from_numpy(a).to("cuda") for a in args)
        torch.cuda.synchronize()
        self.resident = lambda: fn(*dev)
        self.host = lambda: fn(*(torch.from_numpy(a).to("cuda") for a in args)).cpu().numpy()
        self.sync = lambda out: torch.cuda.synchronize()
        self.download = lambda out: out if isinstance(out, np.ndarray) else out.detach().cpu().numpy()

    def normalize(self, out):
        return np.asarray(self.download(out))[:, self.inv_perm]


def make_adapter(backend, operation, fixture, directory, cpu_threads=1):
    if backend in {"mjx", "mujoco_warp", "mujoco_cpu"}:
        return SimulatorAdapter(backend, operation, fixture, directory)
    if backend in {"bard", "frax"}:
        return TensorAdapter(backend, operation, fixture, directory)
    if backend == "curobo":
        return CuroboAdapter(backend, operation, fixture, directory)
    from .pin_adapter import PinAdapter
    return PinAdapter(operation, fixture, directory, cpu_threads=cpu_threads, plain=(backend == "pinocchio_plain"))
