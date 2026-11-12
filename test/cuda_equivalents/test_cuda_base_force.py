"""Native full-force plant values and complete first derivatives, not timing."""
from pathlib import Path
import shutil
import subprocess
import numpy as np
import pytest
from URDFParser import URDFParser
from grid_codegen import GRiDCodeGenerator
from test.cuda_equivalents.free_body_contract import URDF, cases, step, step_jacobian, rotor_map
from test.cuda_equivalents.cuda_harness import _detect_cuda_arch
from test.cuda_equivalents.executable_cache import cached_nvcc_executable

pytestmark = [pytest.mark.cuda_equivalence, pytest.mark.floating_base, pytest.mark.robot_smoke]


@pytest.fixture(scope='module', params=['float', pytest.param('double',marks=pytest.mark.fp64_alpha)])
def free_body(request, tmp_path_factory):
    scalar = request.param
    path = tmp_path_factory.mktemp('base_force_' + scalar)
    robot = URDFParser().parse(str(URDF), floating_base=True)
    gen = GRiDCodeGenerator(robot, dtype=scalar)
    gen.gen_all_code(algorithm_list=['integrator','integrator_gradient'], emit_plant=True,
                     enable_mujoco_kernels=False, output_path=str(path/'grid.cuh'))
    exe, _ = cached_nvcc_executable(
        [Path(__file__).with_name('cuda_base_force_runner.cu')],
        ['-std=c++17','-O2','-lineinfo','-DGRID_CUDA_DEBUG_LAYOUT',
         '-DTEST_SCALAR='+scalar,'-arch=sm_'+_detect_cuda_arch(),'-Xptxas=-v'],
        exe_name='base_force', fallback_dir=path, include_dirs=[path], what='base-force contract')
    dtype = np.float32 if scalar=='float' else np.float64
    inputs = np.array([np.r_[q,v,f] for q,v,f in cases()], dtype=dtype)
    inputs.tofile(path/'input.bin')
    return path, exe, dtype, inputs


@pytest.mark.parametrize('semi', [False, True], ids=['euler','semi-implicit'])
def test_base_force_native_first_order(free_body, semi):
    path, exe, dtype, inputs = free_body
    baseline = None
    for dt0 in (.003,.02):
        dt = float(dtype(dt0))
        expected = np.array([step(row[:7],row[7:13],row[13:],dt,semi) for row in inputs.astype(float)])
        jac = np.array([step_jacobian(row[:7],row[7:13],row[13:],dt,semi) for row in inputs.astype(float)])
        baseline = None
        for threads in (1,17,32):
            output = path/f'out-{semi}-{dt0}-{threads}.bin'
            args = [str(exe),str(path/'input.bin'),str(output),str(threads),str(int(semi)),repr(dt)]
            result = subprocess.run(args, capture_output=True, text=True)
            (path/f'run-{semi}-{dt0}-{threads}.log').write_text(result.stdout+result.stderr)
            assert result.returncode==0, result.stdout+result.stderr+f' exit={result.returncode}'
            actual = np.fromfile(output,dtype)
            values = actual[:8*13].reshape(8,13)
            gradients = actual[8*13:].reshape(8,18,12).transpose(0,2,1)
            np.testing.assert_allclose(values,expected,rtol=3e-5 if dtype==np.float32 else 1e-11,
                                       atol=3e-6 if dtype==np.float32 else 1e-11)
            np.testing.assert_allclose(gradients,jac,rtol=3e-4 if dtype==np.float32 else 2e-7,
                                       atol=3e-6 if dtype==np.float32 else 2e-9)
            np.testing.assert_allclose(gradients[:,:,12:]@rotor_map(),jac[:,:,12:]@rotor_map(),
                                       rtol=3e-4 if dtype==np.float32 else 2e-7,
                                       atol=3e-6 if dtype==np.float32 else 2e-9)
            if baseline is not None:
                np.testing.assert_array_equal(actual.view(np.uint8),baseline.view(np.uint8))
            baseline = actual
        sanitizer = shutil.which('compute-sanitizer')
        assert sanitizer, 'strict CUDA acceptance requires compute-sanitizer'
        for tool in ('memcheck','synccheck','racecheck'):
            output = path/f'{tool}-{semi}-{dt0}.bin'
            result = subprocess.run([sanitizer,'--tool',tool,'--error-exitcode','86',str(exe),
                str(path/'input.bin'),str(output),'17',str(int(semi)),repr(dt)],capture_output=True,text=True)
            log = result.stdout+result.stderr
            (path/f'{tool}-{semi}-{dt0}.log').write_text(log)
            assert result.returncode==0,log
            summary = 'RACECHECK SUMMARY: 0 hazards displayed (0 errors, 0 warnings)' if tool=='racecheck' else 'ERROR SUMMARY: 0 errors'
            assert summary in log,log
            np.testing.assert_array_equal(np.fromfile(output,dtype).view(np.uint8),baseline.view(np.uint8))
