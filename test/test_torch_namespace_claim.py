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


class _Loader:
    """Stand-in for the native runner: registers the artifact's namespaces the
    way the image does when it is opened, and fails on a second registration."""

    def __init__(self, fail=None, on_open=None):
        self.registered, self.opened, self.fail, self.on_open = set(), [], fail, on_open

    def __call__(self, path):
        if self.fail == "open":
            raise RuntimeError(f"dlopen failed for {path}: no such file")
        self.opened.append(path)
        if self.on_open:
            self.on_open()
        with open(path, "rb") as f:
            names = set(_handle._TORCH_NAMESPACE_RE.findall(f.read()))
        assert not names & self.registered, "duplicate registration reached the loader"
        self.registered |= names
        if self.fail == "symbol":
            raise RuntimeError("missing symbol grid_rbd_num_joints")
        return object()


def test_two_threads_cannot_both_pass_the_check(tmp_path):
    import threading
    a = _artifact(tmp_path / "a/robot.so", b"grid_rbd_torch_k94404b389d0a")
    b = _artifact(tmp_path / "b/robot.so", b"grid_rbd_torch_k94404b389d0a")
    inside, release = threading.Event(), threading.Event()
    loader = _Loader(on_open=lambda: (inside.set(), release.wait(5)))
    outcomes = {}

    def load(path):
        try:
            _handle._load_artifact(path, loader)
            outcomes[path] = "loaded"
        except RuntimeError as exc:
            outcomes[path] = str(exc)

    first = threading.Thread(target=load, args=(a,))
    first.start()
    assert inside.wait(5)                      # the first load is inside the loader
    second = threading.Thread(target=load, args=(b,))
    second.start()
    second.join(0.3)
    assert second.is_alive() and loader.opened == [a]   # the second waits its turn
    release.set()
    first.join(5), second.join(5)
    assert outcomes[a] == "loaded" and "another cache" in outcomes[b]
    assert loader.opened == [a]                # the second never reached the loader


def test_only_a_failed_open_leaves_the_namespace_free(tmp_path):
    a = _artifact(tmp_path / "a/robot.so", b"grid_rbd_torch_k94404b389d0a")
    b = _artifact(tmp_path / "b/robot.so", b"grid_rbd_torch_k94404b389d0a")
    with pytest.raises(RuntimeError, match="dlopen failed"):
        _handle._load_artifact(a, _Loader(fail="open"))
    assert _handle._TORCH_NAMESPACE_OWNERS == {}
    # The image opened and registered before the lookup failed: it holds the name.
    with pytest.raises(RuntimeError, match="missing symbol"):
        _handle._load_artifact(a, _Loader(fail="symbol"))
    with pytest.raises(RuntimeError, match="another cache"):
        _handle._load_artifact(b, _Loader())


def test_the_handle_constructor_loads_through_the_guard():
    import inspect
    source = inspect.getsource(_handle.RobotHandle.__init__)
    assert source.count("_load_artifact(") == 2 and "_core.Runner(" not in source \
        and "_core.RunnerF64(" not in source


@pytest.mark.skipif(not hasattr(__import__("os"), "fork"), reason="needs fork")
def test_a_forked_child_gets_a_free_lock_and_keeps_the_owners(tmp_path):
    import os
    a = _artifact(tmp_path / "a/robot.so", b"grid_rbd_torch_k94404b389d0a")
    b = _artifact(tmp_path / "b/robot.so", b"grid_rbd_torch_k94404b389d0a")
    _handle._load_artifact(a, _Loader())
    _handle._ARTIFACT_LOAD_LOCK.acquire()      # as if another thread were mid-load
    try:
        pid = os.fork()
        if pid == 0:
            code = 1
            try:
                _handle._load_artifact(a, _Loader())          # not blocked
                try:
                    _handle._load_artifact(b, _Loader())
                except RuntimeError:
                    code = 0                                  # the parent's image is known
            finally:
                os._exit(code)
        assert os.waitpid(pid, 0)[1] == 0
    finally:
        _handle._ARTIFACT_LOAD_LOCK.release()
