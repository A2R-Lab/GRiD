"""Full base-force rows survive wrapper dispatch and compiler-free reload."""
import subprocess
import sys

import numpy as np
import pytest
import grid_rbd
from test.cuda_equivalents.free_body_contract import URDF, cases, step, step_jacobian

pytestmark = [pytest.mark.python_wrappers, pytest.mark.floating_base]


@pytest.mark.parametrize('dtype', ['float32', pytest.param('float64', marks=pytest.mark.fp64_alpha)])
def test_base_force_prepared_dispatch(dtype, tmp_path):
    h = grid_rbd.register_robot('base_force_contract_' + dtype, str(URDF),
        dtype=dtype, floating_base=True, enable_mujoco_kernels=False,
        max_batch_size=8, algorithm_list=['integrator', 'integrator_gradient'])
    inputs = np.asarray([np.r_[q,v,f] for q,v,f in cases()], dtype=dtype)
    x, u = np.ascontiguousarray(inputs[:,:13]), np.ascontiguousarray(inputs[:,13:])
    saved = dict(x=x,u=u)
    try:
        for semi in (False, True):
            kind = 'semi_implicit_euler' if semi else 'euler'
            dt = float(np.dtype(dtype).type(.003))
            values = h.plant_step(x,u,dt,integrator_type=kind)
            jac = h.plant_step_gradient(x,u,dt,integrator_type=kind)
            expected = np.asarray([step(r[:7],r[7:13],r[13:],dt,semi) for r in inputs.astype(float)])
            expected_jac = np.asarray([step_jacobian(r[:7],r[7:13],r[13:],dt,semi) for r in inputs.astype(float)])
            np.testing.assert_allclose(values,expected,rtol=3e-5 if dtype=='float32' else 1e-11,
                                       atol=3e-6 if dtype=='float32' else 1e-11)
            np.testing.assert_allclose(jac,expected_jac,rtol=3e-4 if dtype=='float32' else 2e-7,
                                       atol=3e-6 if dtype=='float32' else 2e-9)
            saved[kind], saved[kind+'_jac'] = values, jac
        data = tmp_path/'expected.npz'
        np.savez(data, **saved)
        result = subprocess.run([sys.executable,'-c', '''
import sys
import numpy as np
import grid_rbd
import grid_rbd._compile as compiler
import grid_codegen
import URDFParser
def forbidden(*a, **kw):
    raise AssertionError('prepared execution attempted compilation')
compiler.find_nvcc = forbidden
compiler.compile_so = forbidden
compiler.compile_sources = forbidden
grid_rbd.compile_sources = forbidden
grid_rbd.warm_robot = forbidden
grid_rbd.generate_sources = forbidden
grid_codegen.GRiDCodeGenerator.gen_all_code = forbidden
URDFParser.URDFParser.parse = forbidden
h = grid_rbd.get_robot(sys.argv[1])
d = np.load(sys.argv[2])
dt = float(d['x'].dtype.type(.003))
for kind in ('euler','semi_implicit_euler'):
    np.testing.assert_array_equal(h.plant_step(d['x'],d['u'],dt,integrator_type=kind),d[kind])
    np.testing.assert_array_equal(h.plant_step_gradient(d['x'],d['u'],dt,integrator_type=kind),d[kind+'_jac'])
h.close()
''',h.name,str(data)],capture_output=True,text=True)
        (tmp_path/'prepared.log').write_text(result.stdout+result.stderr)
        assert result.returncode == 0, result.stdout+result.stderr
    finally:
        h.close()
