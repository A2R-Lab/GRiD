"""Identical project-order fp32 states and a double-precision independent oracle."""
from __future__ import annotations
import numpy as np
from .protocol import ROBOTS, Q_ONLY_OPS, Q_QD_OPS, digest

TARGETS = {"iiwa14": "iiwa_joint_ee", "go2": "FR_foot_joint", "g1": "right_hand_palm_joint"}


class Fixture:
    def __init__(self, robot, count):
        from RBDReference.tests import MANIFEST_PATH
        from RBDReference.tests.model_sources import iter_robot_cases, resolve_robot_spec
        from RBDReference.equivalents.reference_backend import build_project_adapter
        from RBDReference.equivalents import build_adapter
        from test.cuda_equivalents.cuda_harness import _build_cuda_samples
        self.base = ROBOTS[robot]
        self.spec = next(c["spec"] for c in iter_robot_cases(MANIFEST_PATH, base_mode=self.base)
                         if c["spec"].robot_id == robot)
        self.resolved = resolve_robot_spec(self.spec)
        self.project = build_project_adapter(self.spec, self.resolved, base_mode=self.base)
        self.oracle = build_adapter(self.spec, self.resolved, base_mode=self.base, backend="pinocchio")
        self.nq, self.nv = self.project.nq, self.project.nv
        self.urdf = self.resolved.urdf_path
        self.target = TARGETS[robot]
        samples = _build_cuda_samples(self.project, random_count=count, include_corner_samples=False)[:count]
        self.q = np.ascontiguousarray([s.q for s in samples], dtype=np.float32)
        self.v = np.ascontiguousarray([s.qd for s in samples], dtype=np.float32)
        self.a = np.ascontiguousarray([s.qdd for s in samples], dtype=np.float32)
        self.u = np.random.default_rng(723).uniform(-1, 1, (count, self.nv)).astype(np.float32)
        self.metadata = {"urdf": self.urdf, "urdf_sha256": digest(self.urdf),
            "nq": self.nq, "nv": self.nv, "target": self.target,
            "base": self.base, "sample_names": [s.name for s in samples],
            "sampling": "cuda_harness stable robot seed; zero/conservative + bounded random; torque seed 723",
            "gravity": -9.81, "dtype": "float32", "oracle_dtype": "float64",
            "joint_names": self.project.joint_names,
            "convention": "project order, Pinocchio tangent convention, quaternion xyzw"}

    def args(self, op, batch, padded=True):
        def width(a):
            return np.pad(a[:batch], ((0, 0), (0, self.nq-self.nv))) if padded else a[:batch]
        if op in Q_ONLY_OPS:
            return (self.q[:batch],)
        if op in Q_QD_OPS:
            return self.q[:batch], width(self.v)
        third = self.u if op in {"forward_dynamics", "forward_dynamics_gradient", "fdsva_so"} else self.a
        return self.q[:batch], width(self.v), width(third)

    def expected(self, op, batch):
        args = self.args(op, batch, padded=False)
        method = getattr(self.oracle, "idsva_so_body_frame" if op == "idsva_so" else op)
        outs = []
        for i in range(batch):
            inputs = [a[i].astype(np.float64) for a in args]
            if op.startswith("end_effector_pose"):
                inputs.append(self.target)
            out = method(*inputs)
            # The oracle may hand back views of its Pinocchio data buffers (e.g.
            # ccrba's Ag / hg); copy before the next sample overwrites them.
            out = tuple(np.array(o, copy=True) for o in out) if isinstance(out, tuple) else np.array(out, copy=True)
            if op in {"inverse_dynamics_gradient", "forward_dynamics_gradient"}:
                out = np.concatenate(out[:2], axis=-1)
            outs.append(out)
        if isinstance(outs[0], tuple):
            return tuple(np.stack([o[k] for o in outs]) for k in range(len(outs[0])))
        return np.stack(outs)
