"""CPU ownership/fold guards complement the CUDA empty-buffer regressions."""
import itertools
import numpy as np
import pytest
from URDFParser import URDFParser
from grid_codegen import GRiDCodeGenerator
from grid_codegen.algorithms._idsva_so import _idsva_so_fold_jobs
from grid_codegen.algorithms._frame_jacobian import frame_jacobian_jobs
from grid_codegen.algorithms._coriolis import _coriolis_metadata
from grid_codegen.helpers._topology_helpers import (
    _global_hom_derivative_matrices_by_q, _global_hom_second_derivative_matrices,
)
from grid_codegen.algorithms._eepose_gradient_hessian import (
    _eepose_grad_chain_metadata, _eepose_hessian_chain_metadata, group_jacobian_jobs,
)
from test.cuda_equivalents.constant_model import constant_model


def test_constant_tensor_fold_never_reads_zero_weight_cells():
    # Slot 1 belongs to a constant body. Its placeholder is slot zero, but the
    # projection must not load its tensor cells even with a zero multiplier.
    starts, sources, weights, values = _idsva_so_fold_jobs({0: [0, 1], 1: [2]}, [1., 0., -.5], 2, 3)
    expected_sources = [i*9+j*3+k for i, j, k in itertools.product((0, 2), repeat=3)]
    assert sources == expected_sources
    assert starts == list(range(9))
    internal = np.full(27, np.nan)
    internal[sources] = np.arange(8)+1
    projected = np.array([values[weights[j]] * internal[sources[j]] for j in range(8)])
    expected = np.array([a*b*c for a, b, c in itertools.product((1., -.5), repeat=3)]) * (np.arange(8)+1)
    np.testing.assert_array_equal(projected, expected)


def test_empty_tensor_fold_has_no_jobs():
    assert _idsva_so_fold_jobs({}, [0., 0.], 0, 2) == ([0], [], [], [])


@pytest.mark.parametrize('alpha', [1., -.5, 2.])
def test_lone_mimic_column_keeps_its_scale(alpha):
    from types import SimpleNamespace
    gen = SimpleNamespace(robot_has_mimic_joints=lambda: True,
                          _alpha_for_jid=lambda jid: alpha)
    job = {'j': 1, 'vi': 0}
    direct, weighted, mimic = group_jacobian_jobs(gen, [[job]], [1])
    assert mimic
    if alpha == 1.:
        assert direct == [(0, 1, job)] and weighted == []
    else:
        assert direct == [] and weighted == [[(0, 1, job)]]


@pytest.mark.parametrize('zero', [False, True])
def test_constant_columns_are_not_coordinate_owners(tmp_path, zero):
    robot = constant_model(tmp_path, zero=zero)
    gen = GRiDCodeGenerator(robot)
    constant = robot.get_joint_by_name('constant').jid
    jobs = frame_jacobian_jobs(gen)
    assert all(row[1] != constant and row[-1] != 0 for row in jobs)
    md = _coriolis_metadata(gen)
    assert constant not in md['col_body']
    assert md['body_col_start'][constant] == md['body_col_start'][constant+1]
    chains, _, ee_jobs = _eepose_grad_chain_metadata(gen, [constant])
    assert constant in chains[0]
    assert all(job['j'] != constant for job in ee_jobs[0])
    chains, _, ee_dofs, _ = _eepose_hessian_chain_metadata(gen, [constant])
    assert constant in chains[0]
    assert all(job['joint_jid'] != constant for job in ee_dofs[0])
    gen.gen_all_code(output_path=str(tmp_path / 'grid.cuh'), emit_plant=False,
                     algorithm_list=['inverse_dynamics_gradient', 'frame_jacobian_dot', 'crba'],
                     include_homogenous_transforms=True, enable_mujoco_kernels=False)
    code = (tmp_path / 'grid.cuh').read_text()
    body = code.split(f'// --- body {constant} (v-slot', 1)[1].split('// --- bpass', 1)[0]
    assert 'T qd_a =' not in body
    assert '// dv_dqd[:,idx] += alpha*S' not in body
    assert 'if (fjd_alpha[t] == static_cast<T>(0)) continue;' in code
    assert f'// body {constant} -> v-slot 0 (alpha=0.0)' not in code
    if zero:
        assert jobs == []


@pytest.mark.parametrize('zero', [False, True])
def test_constant_hom_derivative_does_not_own_placeholder(tmp_path, zero):
    robot = constant_model(tmp_path, zero=zero)
    gen = GRiDCodeGenerator(robot)
    matrices, owners = _global_hom_derivative_matrices_by_q(gen)
    if zero:
        assert matrices == owners == []
    else:
        drive = robot.get_joint_by_name('drive').jid
        assert owners == [drive]
        assert matrices == [robot.get_dXmat_hom_local_by_id(drive, 0)]


def test_constant_hom_derivative_preserves_floating_base_owner(tmp_path):
    constant_model(tmp_path)
    robot = URDFParser().parse(str(tmp_path / 'model-False-False-False.urdf'),
                               floating_base=True)
    gen = GRiDCodeGenerator(robot)
    first, owners = _global_hom_derivative_matrices_by_q(gen)
    second, second_owners = _global_hom_second_derivative_matrices(gen)
    nq = robot.get_num_pos()
    constant = robot.get_joint_by_name('constant').jid
    assert constant not in owners and constant not in second_owners
    for i in range(7):
        assert owners[i] == 0
        assert first[i] == robot.get_dXmat_hom_local_by_id(0, i)
        for j in range(7):
            assert second_owners[i*nq+j] == 0
            assert second[i*nq+j] == robot.get_d2Xmat_hom_local_by_id(0, i, j)


def test_floating_constant_crba_has_no_placeholder_projection(tmp_path):
    from test.cuda_equivalents.test_cuda_constant_dynamics_audit import audit_model
    robot = audit_model(tmp_path, floating=True, descendant=True)
    gen = GRiDCodeGenerator(robot)
    gen.gen_all_code(output_path=str(tmp_path / 'grid.cuh'), emit_plant=False,
                     algorithm_list=['crba'], enable_mujoco_kernels=False)
    code = (tmp_path / 'grid.cuh').read_text()
    phase = code.split('// === mimic-aware floating CRBA Phase 2', 1)[1]
    phase = phase.split('// floating-base root block', 1)[0]
    constant = robot.get_joint_by_name('constant').jid
    assert f'// body {constant} ->' not in phase
    assert 'static_cast<T>(0.0) *' not in phase
    # The constant intermediary's transform must still carry a descendant's
    # inertia to the active ancestor and then to the floating base.
    assert f's_XImats[{36*constant} + p + 6*r]' in phase


@pytest.mark.parametrize('algorithm', [
    'inverse_dynamics', 'minv', 'forward_dynamics', 'aba', 'crba',
    'forward_dynamics_gradient', 'integrator_gradient', 'fdsva_so',
    'end_effector_pose_gradient', 'end_effector_pose_hessian',
])
@pytest.mark.parametrize('plant', [False, True])
def test_zero_dof_generation(tmp_path, algorithm, plant):
    """Generation is necessary, not sufficient, for zero-DoF device support."""
    robot = constant_model(tmp_path, zero=True)
    gen = GRiDCodeGenerator(robot)
    gen.gen_all_code(output_path=str(tmp_path / 'grid.cuh'),
                     algorithm_list=[algorithm], emit_plant=plant,
                     enable_mujoco_kernels=False)
