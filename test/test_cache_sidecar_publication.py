"""Shared pointer/identity publication must also be safe between threads."""
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import threading

import pytest

from grid_rbd import _cache


@pytest.mark.parametrize("kind", ["pointer", "identity"])
def test_same_process_publication_uses_private_temporary(tmp_path, monkeypatch, kind):
    target = tmp_path / "bykey" / "key" if kind == "pointer" else tmp_path / "build_inputs.json"
    original = Path.replace
    barrier = threading.Barrier(2)
    temporary_paths = []

    def replace(path, destination):
        if Path(destination) == target:
            temporary_paths.append(path)
            barrier.wait(timeout=10)
        return original(path, destination)

    monkeypatch.setattr(Path, "replace", replace)

    def publish(_):
        if kind == "pointer":
            _cache.keymap_record(tmp_path, "key", "content")
        else:
            _cache.write_build_inputs(tmp_path, {"key_schema": 2})

    with ThreadPoolExecutor(2) as pool:
        list(pool.map(publish, range(2)))
    assert len(set(temporary_paths)) == 2
    assert target.read_text().strip() == ("content" if kind == "pointer" else '{\n  "key_schema": 2\n}')
    assert not list(target.parent.glob("*.tmp"))


@pytest.mark.parametrize("value", [[], ["key_schema"], 1, "identity", None])
def test_identity_record_requires_an_object(tmp_path, value):
    (tmp_path / "build_inputs.json").write_text(json.dumps(value))
    assert _cache.read_build_inputs(tmp_path) is None
    assert _cache.stale_hit_reasons(tmp_path, {"key_schema": 2})
    assert _cache.load_incompat_reasons(tmp_path, {"key_schema": 2})


@pytest.mark.parametrize("kind", ["pointer", "identity"])
def test_permission_failure_is_not_treated_as_a_missing_record(tmp_path, monkeypatch, kind):
    def fail(*args, **kwargs):
        raise PermissionError("injected read refusal")

    monkeypatch.setattr(Path, "read_text", fail)
    with pytest.raises(PermissionError, match="injected"):
        if kind == "identity":
            _cache.read_build_inputs(tmp_path)
        else:
            _cache.keymap_lookup(tmp_path, "key")
