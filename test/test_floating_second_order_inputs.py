"""The two meanings of the SO runner's third input must not alias test state."""
import numpy as np
import pytest

from RBDReference.tests.state_sampling import DynamicsSample
from test.cuda_equivalents.test_cuda_second_order_fallback import (
    _unactuated_floating_fd_sample, _floating_fdsva_enabled)


@pytest.mark.parametrize('value,enabled', [(None, False), ('0', False), ('1', True)])
def test_floating_fdsva_is_uniformly_opt_in(monkeypatch, value, enabled):
    key = 'GRID_CUDA_FLOATING_SECOND_ORDER_ENABLE_FDSVA'
    if value is None:
        monkeypatch.delenv(key, raising=False)
    else:
        monkeypatch.setenv(key, value)
    assert _floating_fdsva_enabled() is enabled


@pytest.mark.parametrize('nv', [7, 13, 35])
def test_floating_fd_input_preserves_id_sample(nv):
    sample = DynamicsSample(name='nonzero', q=np.arange(nv + 1, dtype=float),
                            qd=np.arange(nv, dtype=float), qdd=np.arange(nv, dtype=float) + 1)
    original = sample.qdd.copy()
    fd = _unactuated_floating_fd_sample(sample)
    np.testing.assert_array_equal(fd.qdd[:6], np.zeros(6))
    np.testing.assert_array_equal(fd.qdd[6:], original[6:])
    np.testing.assert_array_equal(sample.qdd, original)
    assert not np.shares_memory(fd.qdd, sample.qdd)
    assert fd.q is sample.q and fd.qd is sample.qd
    assert fd.name == sample.name
