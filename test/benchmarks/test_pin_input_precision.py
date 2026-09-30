"""CPU-only tests: exercise the real prepare closure with a fake native ABI.

No native build, GPU initialization or benchmark timing is involved.
"""
import ctypes
from types import SimpleNamespace

import numpy as np
import pytest

from test.benchmarks.release.pin_adapter import FP32_OPS, OPS, PinAdapter


@pytest.mark.parametrize("operation", OPS)
def test_pin_prepare_preserves_algorithm_input_precision(operation):
    # A quaternion whose fp64 normalization cannot survive an fp32 round-trip.
    q = np.array([[0, 0, 0, .1, .2, .3, .9],
                  [1, 2, 3, .4, -.1, .7, .2]], dtype=np.float32)
    def normalize(x):
        out = x.copy()
        out[3:7] /= np.linalg.norm(out[3:7])
        return out
    v = np.arange(12, dtype=np.float32).reshape(2, 6) / 10
    fixture = SimpleNamespace(q=q, v=v, a=v + 1, u=v + 2, nv=6,
        oracle=SimpleNamespace(_to_pin_q=normalize, _expand_project_v_to_pin=lambda x: x))
    single = OPS.index(operation) in FP32_OPS
    dtype, ctype = (np.float32, ctypes.c_float) if single else (np.float64, ctypes.c_double)
    seen = []
    def evaluate(pool, qp, vp, tp, batch, output, active):
        assert isinstance(qp, ctypes.POINTER(ctype))
        assert isinstance(vp, ctypes.POINTER(ctype))
        assert isinstance(tp, ctypes.POINTER(ctype))
        assert batch == 2 and active == 1
        mapped = np.ctypeslib.as_array(qp, shape=(batch * 7,)).reshape(batch, 7)
        np.testing.assert_array_equal(mapped, np.array([normalize(x.astype(float)) for x in q], dtype=dtype))
        for ptr, expected in ((vp, v), (tp, fixture.u if operation in {
                "forward_dynamics", "forward_dynamics_gradient", "fdsva_so"} else fixture.a)):
            np.testing.assert_array_equal(np.ctypeslib.as_array(ptr, shape=(12,)).reshape(2, 6), expected)
        if not single:
            assert not np.array_equal(mapped, mapped.astype(np.float32).astype(np.float64))
        seen.append(True)
        return 0  # output is unused: this test is of input lifetime/ABI selection
    def wrong_abi(*args):
        pytest.fail("selected the wrong native input precision")
    adapter = PinAdapter.__new__(PinAdapter)
    adapter.f, adapter.op, adapter.pool, adapter.metadata = fixture, operation, object(), {}
    adapter.lib = SimpleNamespace(
        pin_release_pool_eval=evaluate if single else wrong_abi,
        pin_release_pool_eval_f64=wrong_abi if single else evaluate)
    adapter.prepare(2)
    adapter.host()  # arrays must survive after prepare returns
    assert seen == [True]
