"""CPU ownership/fold guards complement the CUDA empty-buffer regressions."""
import itertools
import numpy as np
import pytest
from grid_codegen import GRiDCodeGenerator
from grid_codegen.algorithms._idsva_so import _idsva_so_fold_jobs
from grid_codegen.algorithms._frame_jacobian import frame_jacobian_jobs
from grid_codegen.algorithms._coriolis import _coriolis_metadata
from test.cuda_equivalents.test_cuda_constant_derivatives import constant_model


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
    gen.gen_all_code(output_path=str(tmp_path / 'grid.cuh'), emit_plant=False,
                     algorithm_list=['inverse_dynamics_gradient', 'frame_jacobian_dot'],
                     include_homogenous_transforms=True, enable_mujoco_kernels=False)
    code = (tmp_path / 'grid.cuh').read_text()
    body = code.split(f'// --- body {constant} (v-slot', 1)[1].split('// --- bpass', 1)[0]
    assert 'T qd_a =' not in body
    assert '// dv_dqd[:,idx] += alpha*S' not in body
    assert 'if (fjd_alpha[t] == static_cast<T>(0)) continue;' in code
    if zero:
        assert jobs == []
