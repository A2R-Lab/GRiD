"""Independent Pinocchio energy oracle for baked and mutable rotor inertia."""
import numpy as np
import pytest

import grid_rbd
from grid_assets import robot_urdf
from grid_codegen.armature import supported_slots
from RBDReference.tests import MANIFEST_PATH
from RBDReference.tests.model_sources import iter_robot_cases, resolve_robot_spec
from RBDReference.tests.state_sampling import build_dynamics_samples
from RBDReference.tests.comparators import assert_close
from RBDReference.equivalents.pinocchio_backend import build_pinocchio_adapter
from RBDReference.equivalents.reference_backend import build_project_adapter

pytestmark = pytest.mark.python_wrappers
CASES = [("iiwa14", "fixed"), ("iiwa14", "floating"), ("go2", "floating"),
         ("gen3", "fixed"), ("fr3", "fixed")]


@pytest.mark.parametrize("robot_id,base_mode", CASES,
                         ids=[f"{r}-{b}" for r, b in CASES])
def test_energy_with_armature_matches_pinocchio(robot_id, base_mode):
    spec = next(c["spec"] for c in iter_robot_cases(MANIFEST_PATH, base_mode=base_mode)
                if c["spec"].robot_id == robot_id)
    resolved = resolve_robot_spec(spec)
    project = build_project_adapter(spec, resolved, base_mode=base_mode)
    pin = build_pinocchio_adapter(spec, resolved, base_mode=base_mode)
    robot = project.reference.robot
    slots = supported_slots(robot)
    armature = np.zeros(robot.get_num_vel(), dtype=np.float32)
    armature[slots] = np.linspace(.15, .45, len(slots))
    samples = build_dynamics_samples(project)
    q = np.asarray([s.q for s in samples], dtype=np.float32)
    v = np.asarray([s.qd for s in samples], dtype=np.float32)
    plain = np.asarray([pin.kinetic_energy(qr.astype(float), vr.astype(float))
                        for qr, vr in zip(q, v)])
    reference_plain = np.asarray([project.reference.kinetic_energy(qr.astype(float), vr.astype(float))
                                  for qr, vr in zip(q, v)])
    for runtime in (False, True):
        handle = grid_rbd.register_robot(
            f"energy_pin_armature_{robot_id}_{base_mode}_{runtime}",
            str(robot_urdf(robot_id)), floating_base=base_mode == "floating",
            armature=armature, runtime_armature=runtime,
            algorithm_list=["energy"], enable_mujoco_kernels=False,
            max_batch_size=len(samples))
        try:
            for scale in ((1., 2., 1.) if runtime else (1.,)):
                current = armature * scale
                if runtime:
                    handle.set_armature(current)
                pin.set_armature(current.astype(float))
                project.reference.set_armature(current.astype(float))
                expected = []
                for row, (qr, vr) in enumerate(zip(q.astype(float), v.astype(float))):
                    ke, pe = pin.kinetic_energy(qr, vr), pin.potential_energy(qr)
                    expected.append([ke, pe, ke + pe])
                    ref_ke = project.reference.kinetic_energy(qr, vr)
                    # Use the established cross-library energy class (Gen3's
                    # continuous-joint discrepancy also exists with zero armature).
                    assert_close(ref_ke, ke, algorithm="energy", robot_id=robot_id)
                    # The armature addition itself must still agree tightly.
                    np.testing.assert_allclose(ref_ke - reference_plain[row], ke - plain[row],
                                               rtol=1e-10, atol=1e-10)
                expected = np.asarray(expected)
                # Independently confirm the oracle includes rotor energy.
                np.testing.assert_allclose(expected[:, 0] - plain,
                    .5 * np.sum(current.astype(float) * v.astype(float)**2, axis=1),
                    rtol=1e-10, atol=1e-10)
                np.testing.assert_allclose(handle.energy(q, v), expected,
                                           rtol=2e-4, atol=2e-4)
        finally:
            handle.close()
