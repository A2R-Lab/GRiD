"""Skew revolute mimic dynamics on the GPU against closed-form mechanics.

The revolute twin of ``test_prismatic_dynamics_against_independent_mechanics``:
two bodies on one skew axis, the follower a revolute mimic with multiplier -0.5
and offset 0.17. Expectations come from ``test/revolute_mimic_mechanics.py``,
which uses neither generator nor RBDReference; the numpy oracle is pinned to
the same closed form by ``test/test_revolute_mimic_mechanics.py``. Outputs must
be thread-count invariant.
"""
import numpy as np
import pytest
from grid_assets import robot_urdf
from grid_codegen import GRiDCodeGenerator
from URDFParser import URDFParser
from test import revolute_mimic_mechanics as mech

pytestmark = [pytest.mark.cuda_equivalence, pytest.mark.robot_smoke]

ALGOS = ["inverse_dynamics", "forward_dynamics", "aba", "crba", "minv"]


def test_revolute_mimic_dynamics_against_independent_mechanics(tmp_path, monkeypatch):
    from RBDReference import RBDReference
    from test.cuda_equivalents.cuda_harness import _compile_runner, _parse_runner_output, _run_runner

    robot = URDFParser().parse(str(robot_urdf("revolute_mimic_fixed_tool")))
    gen = GRiDCodeGenerator(robot, False, True, "grid")
    gen.gen_all_code(output_path=str(tmp_path / "grid.cuh"), algorithm_list=ALGOS,
                     enable_mujoco_kernels=False)
    monkeypatch.setenv("GRID_CUDA_CACHE_DIR", str(tmp_path / "executables"))
    exe, command = _compile_runner(tmp_path, run_tokens={"RUN_" + a.upper() for a in ALGOS},
                                   skip_gradients=True, skip_eepose_gradients=True)
    mass = mech.generalized_mass()
    reference = RBDReference(robot)
    for q, qd, torque in ((0., 0., 0.), (.13, -.4, 2.), (.5, .7, -3.), (-1.2, .25, .4)):
        bias = mech.gravity_torque(q)
        expected = {"inverse_dynamics": bias, "forward_dynamics": (torque - bias) / mass,
                    "aba": (torque - bias) / mass, "crba": mass, "minv": 1. / mass}
        np.testing.assert_allclose(reference.crba(np.array([q])), [[mass]], atol=1e-12, rtol=1e-12)
        np.testing.assert_allclose(reference.inverse_dynamics(np.array([q]), np.array([qd]), np.zeros(1))[0],
                                   [bias], atol=1e-12, rtol=1e-12)
        baseline = None
        for threads in (1, 32, 64, 128):
            stdout = _run_runner(exe, f"{q}\n{qd}\n{torque}\n", command, num_threads=threads)
            outputs = _parse_runner_output(stdout)
            actual = []
            for name, value in expected.items():
                assert name in outputs, (name, stdout)
                result = np.asarray(outputs[name]).reshape(-1)
                assert result.size == 1, (name, result)
                np.testing.assert_allclose(result, [value], atol=2e-4, rtol=2e-4, err_msg=name)
                actual.extend(result)
            if baseline is not None:
                np.testing.assert_array_equal(actual, baseline)
            baseline = actual
