"""Polynomial followers do not introduce independent rotor parameters."""
import contextlib
import io

import numpy as np
import pytest

from grid_codegen import GRiDCodeGenerator
from grid_codegen.armature import supported_slots
from URDFParser import URDFParser
from test.cuda_equivalents.polynomial_coupling_model import write_polynomial_chain


@pytest.mark.parametrize("floating", [False, True])
def test_polynomial_armature_uses_only_reduced_independent_slots(tmp_path, floating):
    with contextlib.redirect_stdout(io.StringIO()):
        robot = URDFParser().parse(str(write_polynomial_chain(tmp_path)), floating_base=floating)
    offset = 6 if floating else 0
    assert supported_slots(robot) == [offset, offset + 1]
    # Joint/body indexing is not accepted as a separate follower-rotor API.
    body_values = np.zeros(robot.get_num_joints())
    body_values[-1] = .2
    with pytest.raises(ValueError, match="shape"):
        GRiDCodeGenerator(robot, armature=body_values)
    values = np.zeros(robot.get_num_vel())
    values[supported_slots(robot)] = [.2, .4]
    generator = GRiDCodeGenerator(robot, armature=values)
    assert generator.initial_armature == values.astype(np.float32).tolist()


def test_polynomial_boundary_distinguishes_coriolis_from_second_order(tmp_path):
    with contextlib.redirect_stdout(io.StringIO()):
        robot = URDFParser().parse(str(write_polynomial_chain(tmp_path)))
        GRiDCodeGenerator(robot).gen_all_code(output_path=str(tmp_path / "coriolis.cuh"),
            algorithm_list=["coriolis_matrix"], enable_mujoco_kernels=False)
        with pytest.raises(NotImplementedError, match="polynomial couplings.*fdsva_so"):
            GRiDCodeGenerator(robot).gen_all_code(output_path=str(tmp_path / "refused.cuh"),
                algorithm_list=["fdsva_so"], enable_mujoco_kernels=False)
