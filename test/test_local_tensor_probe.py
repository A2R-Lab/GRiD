"""CPU-only checks for the isolated GPU emitter's expressions and boundaries."""
import ast
from pathlib import Path

import pytest

from tools.run_local_tensor_probe import expression, literal


def test_only_float_expressions_and_explicit_names():
    source = expression(ast.parse('2*x**2 + sin(x)/3.0',mode='eval').body, {'x':'q[0]'})
    assert source == '((2.0f*powf(q[0],2.0f))+(sinf(q[0])/3.0f))'
    assert literal(-.25) == '-0.25f'
    with pytest.raises(ValueError):
        expression(ast.parse('f(x)',mode='eval').body, {'x':'q[0]'})
    with pytest.raises(KeyError):
        expression(ast.parse('other',mode='eval').body, {'x':'q[0]'})


def test_private_probe_has_no_runtime_timing_or_global_scratch():
    source=(Path(__file__).parents[1]/'tools/local_tensor_probe.cuh').read_text()
    assert 'cudaEvent' not in source
    assert 'atomicAdd' not in source
    assert 'double' not in source.replace('// timing events, global scratch arena, or public GRiD layout changes.', '').replace('// One block, one directional pair at a time. No atomics, double arithmetic,','')
    assert '<<<1,threads>>>' in source
    assert '__shared__ float v[NB*24], a[NB*24], force[NB*24]' in source
    assert 'cudaMemset(dout,255' in source
    assert 'cudaGetLastError' in source and 'cudaDeviceSynchronize' in source


def test_full_gpu_probe_computes_its_own_acceleration_and_solve():
    source=(Path(__file__).parents[1]/'tools/local_chain_probe.cuh').read_text()
    assert 'qdd[i]=torque[i]-bias[i]' in source
    assert 'solve_column(qdd,1,0,lower,scale)' in source
    assert 'grad[row*(2*NV)+pair]=-value' in source
    assert 'ARMATURE[row]' in source
    assert 'cudaEvent' not in source and 'atomicAdd' not in source
    assert '<<<1,threads>>>' in source
