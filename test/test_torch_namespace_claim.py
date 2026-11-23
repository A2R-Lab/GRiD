"""One process cannot load two artifacts that register one PyTorch namespace.

The operator namespace is named after the model key. Two builds of one model
in two cache directories carry the same name, and PyTorch aborts the process
when the second registers. The handle must raise before the load.
"""
import pytest

from grid_rbd import _handle


@pytest.fixture(autouse=True)
def fresh_owners(monkeypatch):
    monkeypatch.setattr(_handle, "_TORCH_NAMESPACE_OWNERS", {})


def _artifact(path, namespace):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"\x7fELF" + b"\0" * 64 + b"_ZN" + namespace + b"ERN5torch7LibraryE" + b"\0" * 64)
    return str(path)


def _load(path):
    _handle._record_torch_namespace(_handle._check_torch_namespace(path))


def test_same_model_from_a_second_cache_is_refused_before_loading(tmp_path):
    a = _artifact(tmp_path / "a/robot.so", b"grid_rbd_torch_k94404b389d0a")
    b = _artifact(tmp_path / "b/robot.so", b"grid_rbd_torch_k94404b389d0a")
    _load(a)
    _load(a)                                   # the same file shares one image
    link = tmp_path / "link.so"
    link.symlink_to(a)
    _load(str(link))                           # and so does another name for it
    with pytest.raises(RuntimeError) as err:
        _load(b)
    message = str(err.value)
    assert a in message and b in message and "grid_rbd_torch_k94404b389d0a" in message
    _load(a)                                   # the refusal claimed nothing


def test_different_models_and_artifacts_without_torch_load_side_by_side(tmp_path):
    _load(_artifact(tmp_path / "a/robot.so", b"grid_rbd_torch_k000000000001"))
    _load(_artifact(tmp_path / "b/robot.so", b"grid_rbd_torch_k000000000002"))
    plain_a = tmp_path / "c/robot.so"
    plain_b = tmp_path / "d/robot.so"
    for plain in (plain_a, plain_b):
        plain.parent.mkdir()
        plain.write_bytes(b"\x7fELF" + b"grid_rbd_torch_def_kinematics" + b"\0" * 32)
        _load(str(plain))                      # no namespace, nothing to collide


def test_a_failed_load_claims_nothing(tmp_path):
    a = _artifact(tmp_path / "a/robot.so", b"grid_rbd_torch_k94404b389d0a")
    b = _artifact(tmp_path / "b/robot.so", b"grid_rbd_torch_k94404b389d0a")
    assert _handle._check_torch_namespace(a) is not None   # checked, never recorded
    _load(b)
    assert _handle._check_torch_namespace(str(tmp_path / "missing.so")) is None
