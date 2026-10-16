"""The flagship CUDA-vs-oracle equivalence matrix in double precision: the fp64
alpha lane (2026-10-15).

GRiD runs in fp32 on the GPU. The fp64 tier (the codegen ``dtype="double"``
knob, the runners' ``GRID_EQUIV_T=double`` switch) is kept in full as an alpha
feature, and this module is its CUDA half: every flagship split cell
(``FLAGSHIP_SPLIT_CELLS``) for every manifest robot and both bases, at one
thread count (the robot's suggested launch, ``threadssuggested``) to bound the
matrix, with

  * the header generated with ``GRiDCodeGenerator(dtype="double")`` under its
    own header cache key,
  * the runner compiled under its own executable cache key and launched with
    ``GRID_EQUIV_T=double``,
  * the samples written to stdin at full double precision, and
  * the comparison at the fp64 tolerance class (``CUDA_FP64_TOLERANCE``: rtol
    1e-7 with a scale-relative atol; an fp64 build of the floating
    forward_dynamics_gradient cell reached norm_rel <= 8e-8).

Every item carries ``fp64_alpha`` (module ``pytestmark``), so test/conftest.py
leaves ``gpu_proof`` off and the signed receipt excludes the lane by collection.
Run it with ``make test-fp64``. The cells still carry ``cuda_equivalence`` (the
nvcc marker the CPU-only filter keys on). The two launch points that cannot run
on this hardware in float (``FLAGSHIP_DECOLLECTED_CELLS``, a register budget
exceeded at the suggested thread count) are left out here too.
"""
import pytest

from test.cuda_equivalents.cuda_harness import (
    FLAGSHIP_DECOLLECTED_CELLS,
    FLAGSHIP_SPLIT_CELLS,
    _run_flagship_split_cell,
    build_cuda_case_params,
)

pytestmark = pytest.mark.fp64_alpha

# The suggested launch (num_threads 0 = the robot's MAX_PERF_LEVEL_THREADS,
# resolved in the runner from the generated header), the one thread count of
# this lane. It is also the receipt's run-to-run determinism point.
_SUGGESTED = 0


def build_fp64_flagship_cell_params(base_mode: str):
    """One pytest.param per flagship cell at the suggested thread count:
    (spec, base_mode, cell, num_threads), id ``<robot>-<base>-<cell_id>-threadssuggested``."""
    params = []
    for case in build_cuda_case_params(base_mode):
        spec = case.values[0]
        for cell in FLAGSHIP_SPLIT_CELLS:
            if (spec.robot_id, base_mode, cell.cell_id, _SUGGESTED) in FLAGSHIP_DECOLLECTED_CELLS:
                continue
            params.append(
                pytest.param(
                    spec, base_mode, cell, _SUGGESTED,
                    id=f"{spec.robot_id}-{base_mode}-{cell.cell_id}-threadssuggested",
                    marks=case.marks,
                )
            )
    return params


@pytest.mark.parametrize(("spec", "base_mode", "cell", "num_threads"), build_fp64_flagship_cell_params("fixed"))
def test_fixed_base_generated_cuda_matches_python_reference_fp64(spec, base_mode, cell, num_threads, tmp_path, request):
    _run_flagship_split_cell(spec, base_mode, cell, num_threads, tmp_path, request, dtype="double")


@pytest.mark.parametrize(("spec", "base_mode", "cell", "num_threads"), build_fp64_flagship_cell_params("floating"))
def test_floating_base_generated_cuda_matches_python_reference_fp64(spec, base_mode, cell, num_threads, tmp_path, request):
    _run_flagship_split_cell(spec, base_mode, cell, num_threads, tmp_path, request, dtype="double")
