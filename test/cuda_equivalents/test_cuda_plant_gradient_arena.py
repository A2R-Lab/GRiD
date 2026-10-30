"""Full-shared plant allocation must not borrow a spilled integrator sizer."""
import json
import shutil
import subprocess

import numpy as np
import pytest
from URDFParser import URDFParser
from grid_assets import robot_urdf
from grid_codegen import GRiDCodeGenerator
from grid_codegen.algorithms._plant import _plant_step_gradient_extra_t_buffers
from test.cuda_equivalents.cuda_harness import _detect_cuda_arch

pytestmark = [pytest.mark.cuda_equivalence, pytest.mark.robot_smoke]


@pytest.mark.parametrize('name,floating,dtype,runtime', [
    ('iiwa14', False, 'double', False),
    ('go2', True, 'float', False),
    ('go2', True, 'float', True),
    ('g1', False, 'float', False),
])
def test_full_shared_plant_gradient_bounds(tmp_path, name, floating, dtype, runtime):
    robot = URDFParser().parse(str(robot_urdf(name)), floating_base=floating)
    gen = GRiDCodeGenerator(robot, dtype=dtype)
    gen.gen_all_code(algorithm_list=['integrator_gradient'], emit_plant=True,
                     enable_mujoco_kernels=False, runtime_transform=runtime,
                     output_path=str(tmp_path / 'grid.cuh'))
    # The real kernel-emission helper, not the binder's own size expression.
    layout = gen.gen_XImats_helpers_temp_shared_memory_code(
        gen.gen_integrator_gradient_inner_temp_mem_size(),
        extra_t_buffers=_plant_step_gradient_extra_t_buffers(gen))
    fields = list(layout.t_buffers) + [('s_XImats', layout.ximat_size),
                                      ('s_temp', layout.temp_mem_size)]
    if layout.topology_count:
        fields.append(('s_topology_helpers', layout.topology_count))
    writes = '\n'.join(
        f'ends[{i}] = reinterpret_cast<unsigned char *>(a.{field} + {count}) - base;\n'
        f'if ({count}) a.{field}[{count}-1] = 0;'
        for i, (field, count) in enumerate(fields))
    source = r'''
#include "grid.cuh"
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <vector>
using T = DTYPE;
#define CK(x) do { auto e=(x); if(e!=cudaSuccess) {fprintf(stderr,"%s: %s\n",#x,cudaGetErrorString(e));exit(2);} } while(0)
__global__ void inspect(unsigned char *base, size_t *ends) {
    auto a = grid_plant::plant_step_gradient_arena<T>::carve(base);
    WRITES
}
int main() {
    const size_t bytes = grid_plant::PLANT_STEP_GRADIENT_DYNAMIC_SHARED_MEM_BYTES<T>();
    const size_t legacy = grid::INTEGRATOR_DU_DYNAMIC_SHARED_MEM_BYTES<T,grid::TIER_SHARED>();
    unsigned char *base; size_t *ends, host[NFIELDS];
    CK(cudaMalloc(&base, bytes + 32)); CK(cudaMemset(base, 0xA5, bytes + 32));
    CK(cudaMalloc(&ends, sizeof(host)));
    inspect<<<1,1>>>(base, ends); CK(cudaGetLastError()); CK(cudaDeviceSynchronize());
    CK(cudaMemcpy(host, ends, sizeof(host), cudaMemcpyDeviceToHost));
    unsigned char guard[32]; CK(cudaMemcpy(guard, base+bytes, 32, cudaMemcpyDeviceToHost));
    for(auto x:guard) if(x!=0xA5) return 3;
    size_t end=0; for(auto x:host) end=std::max(end,x);
    if(end>bytes || bytes-end>=16) return 4;
    size_t cap; CK(grid::grid_get_max_dynamic_shared_memory_bytes(&cap));
    auto admission=grid_plant::plant_step_gradient_check_shared_memory<T>();
    if(admission!=(bytes>cap ? cudaErrorInvalidConfiguration : cudaSuccess)) return 5;
    printf("{\"bytes\":%zu,\"legacy\":%zu,\"end\":%zu,\"capacity\":%zu,\"admitted\":%s}\n",
           bytes, legacy, end, cap, admission==cudaSuccess ? "true":"false");
    CK(cudaFree(ends)); CK(cudaFree(base));
}
'''.replace('DTYPE', dtype).replace('WRITES', writes).replace('NFIELDS', str(len(fields)))
    runner, exe = tmp_path / 'probe.cu', tmp_path / 'probe'
    runner.write_text(source)
    nvcc = shutil.which('nvcc')
    assert nvcc, 'CUDA toolkit required'
    compiled = subprocess.run([nvcc, '-std=c++17', '-O2', '-DGRID_CUDA_DEBUG_LAYOUT',
                               f'-arch=sm_{_detect_cuda_arch()}', str(runner), '-o', str(exe)],
                              capture_output=True, text=True)
    (tmp_path / 'compile.log').write_text(compiled.stdout + compiled.stderr)
    assert compiled.returncode == 0, compiled.stderr
    ran = subprocess.run([str(exe)], capture_output=True, text=True)
    (tmp_path / 'run.log').write_text(ran.stdout + ran.stderr)
    assert ran.returncode == 0, ran.stdout + ran.stderr
    measured = json.loads(ran.stdout.strip().splitlines()[-1])
    if name == 'g1':
        # Negative control: the old allocation really was smaller than the carve.
        assert measured['end'] > measured['legacy']
    assert measured['admitted'] == (measured['bytes'] <= measured['capacity'])
    if name in ('iiwa14', 'g1'):
        import grid_rbd
        from RBDReference import RBDReference
        handle = grid_rbd.register_robot(
            name=f'plant_arena_{name}', urdf_path=str(robot_urdf(name)),
            cache_dir=tmp_path / 'cache', max_batch_size=2,
            algorithm_list=['integrator_gradient'], enable_mujoco_kernels=False,
            dtype='float64' if dtype == 'double' else 'float32')
        scalar = np.float64 if dtype == 'double' else np.float32
        rng = np.random.default_rng(91)
        q = rng.uniform(-.1, .1, (2, robot.get_num_pos())).astype(scalar)
        qd = rng.uniform(-.1, .1, (2, robot.get_num_vel())).astype(scalar)
        u = rng.uniform(-.1, .1, qd.shape).astype(scalar)
        x = np.concatenate([q, qd], axis=1)
        # Rejection must not poison subsequent use of the supported tiered path.
        if not measured['admitted']:
            with pytest.raises(RuntimeError):
                handle.plant_step_gradient(x, u, .01)
        tiered = handle.integrator_gradient(q, qd, u, .01)
        reference = RBDReference(robot)
        for b in range(2):
            expected = reference.plant_step_gradient(q[b], qd[b], u[b], .01)
            error = np.max(np.abs(tiered[b] - expected) / (np.abs(expected) + 1e-3))
            assert error < (5e-8 if dtype == 'double' else 5e-3)
        if measured['admitted']:
            actual = handle.plant_step_gradient(x, u, .01)
            np.testing.assert_array_equal(actual, tiered)
            # Step and gradient share the precision-dependent C ABI scalar
            # contract. Non-default gravity and dt catch accidental constants.
            for dt, gravity in ((.01, -9.81), (.025, -3.7)):
                step = handle.plant_step(x, u, dt, gravity=gravity)
                for b in range(2):
                    bias = reference.inverse_dynamics(q[b], qd[b], GRAVITY=gravity)[0]
                    acceleration = np.linalg.solve(reference.crba(q[b]), u[b] - bias)
                    expected = np.concatenate([q[b] + dt * qd[b], qd[b] + dt * acceleration])
                    error = np.max(np.abs(step[b] - expected) / (np.abs(expected) + 1e-3))
                    assert error < (5e-8 if dtype == 'double' else 5e-3)
