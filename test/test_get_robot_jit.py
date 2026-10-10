"""get_robot's just-in-time path + the one-time build notice (CPU-only: the
compile halves are the test_rbd_cache_identity stubs)."""
from __future__ import annotations
import logging
import sys
from pathlib import Path
import pytest
import grid_rbd
sys.path.insert(0, str(Path(__file__).resolve().parent))
from test_rbd_cache_identity import stubs, _warm  # noqa: E402,F401  (fixture + helper)


class _Handle:
    def __init__(self, name, so, entry):
        self.name = name

    def apply_profile_overlay(self, _):
        pass



def test_miss_without_urdf_raises_with_the_three_ways_out(tmp_path):
    with pytest.raises(grid_rbd.RobotNotRegisteredError) as ei:
        grid_rbd.get_robot("nope", cache_dir=tmp_path)
    msg = str(ei.value)
    assert "register_robot" in msg and "get_robot('nope', urdf_path=...)" in msg and "load_robot" in msg


def test_register_opts_without_urdf_is_an_error(tmp_path):
    with pytest.raises(TypeError):
        grid_rbd.get_robot("nope", cache_dir=tmp_path, floating_base=True)


def test_miss_with_urdf_registers_then_hit_ignores_it(stubs, monkeypatch):
    calls = []

    def rec(name, urdf_path=None, **kw):
        calls.append((name, urdf_path, kw))
        return _warm(stubs, name=name)
    monkeypatch.setattr(grid_rbd, "register_robot", rec)
    monkeypatch.setattr(grid_rbd, "RobotHandle", _Handle)
    h = grid_rbd.get_robot("jit", cache_dir=stubs.cache_dir, urdf_path="/robots/jit.urdf", floating_base=True)
    assert calls == [("jit", "/robots/jit.urdf", {"cache_dir": stubs.cache_dir, "floating_base": True})]
    assert h.name == "jit"
    # a hit never re-registers, even with a urdf_path
    grid_rbd.get_robot("jit", cache_dir=stubs.cache_dir, urdf_path="/elsewhere.urdf")
    assert len(calls) == 1


def test_build_notice_is_a_warning_only_when_compiling(stubs, caplog, monkeypatch):
    caplog.set_level(logging.WARNING, logger="grid_rbd")
    _warm(stubs)
    msgs = [r.message for r in caplog.records if r.levelno == logging.WARNING]
    assert any("compiling the dynamics of" in m and "one-time" in m for m in msgs)
    assert any(m.startswith("grid_rbd: built") for m in msgs)
    caplog.clear()
    _warm(stubs)          # warm hit: silent
    assert not [r for r in caplog.records if r.levelno == logging.WARNING]
    caplog.clear()
    monkeypatch.setenv("GRID_RBD_QUIET", "1")
    stubs.emitted = "// grid.cuh v2\n"
    monkeypatch.setattr(grid_rbd._cache, "_codegen_source_hash", lambda: "edited")
    _warm(stubs)          # recompiles, but quiet
    assert not [r for r in caplog.records if r.levelno == logging.WARNING]
