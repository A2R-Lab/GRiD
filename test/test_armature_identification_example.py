"""Excitation and fitting failure controls for the public synthetic example."""
import importlib.util
from pathlib import Path
import numpy as np
import pytest

spec = importlib.util.spec_from_file_location("armature_example", Path(__file__).resolve().parents[1]/"examples/armature_identification.py")
example = importlib.util.module_from_spec(spec)
spec.loader.exec_module(example)


def test_known_fit_and_held_out_prediction():
    rng = np.random.default_rng(44)
    y = rng.normal(size=(20,4,3))
    alpha = np.array([.1,.3,.8])
    residual = y @ alpha
    got, singular = example.fit_armature(y[:15],residual[:15])
    np.testing.assert_allclose(got,alpha,rtol=1e-12,atol=1e-12)
    np.testing.assert_allclose(y[15:]@got,residual[15:],rtol=1e-12,atol=1e-12)
    assert singular[-1] > 0


def test_rank_deficiency_is_not_success():
    with pytest.raises(ValueError, match="rank-deficient"):
        example.fit_armature(np.zeros((10,4,3)),np.zeros((10,4)))


def test_negative_fit_is_not_silently_clipped():
    with pytest.raises(ValueError, match="negative fitted"):
        example.fit_armature(np.ones((10,1,1)),-np.ones((10,1)))
