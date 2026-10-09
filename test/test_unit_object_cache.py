"""bindings/grid_rbd/_compile._build_units: the per-unit object cache (M4).

CPU-only: nvcc is replaced by a fake that writes the requested object / .so
and records every `-c` compile; the assertions are WHICH units got compiled.
"""
from __future__ import annotations
from pathlib import Path
import pytest
from grid_rbd import _compile, _cache


class FakeNvcc:
    def __init__(self):
        self.compiled: list[str] = []   # unit stems, in call order
        self.links = 0

    def __call__(self, cmd, *a, **k):
        if "-c" in cmd:
            unit = Path(cmd[cmd.index("-c") + 1])
            Path(cmd[cmd.index("-o") + 1]).write_bytes(b"OBJ:" + unit.read_bytes())
            self.compiled.append(unit.stem)
        else:
            Path(cmd[cmd.index("-o") + 1]).write_bytes(b"SO")
            self.links += 1

        class R:
            returncode = 0
        return R()


@pytest.fixture
def env(monkeypatch, tmp_path):
    fake = FakeNvcc()
    monkeypatch.setattr(_compile.subprocess, "run", fake)
    for fn in ("_nvcc_identity", "_host_cxx_tag", "_glass_content_hash", "_torch_abi_tag", "_jax_ffi_tag"):
        monkeypatch.setattr(_cache, fn, lambda fn=fn: fn)
    monkeypatch.setenv("GRID_RBD_BUILD_JOBS", "2")
    src = tmp_path / "store" / ".gen_1"
    src.mkdir(parents=True)
    (src / "grid.cuh").write_text("// header v1\n")
    (src / "wrapper_common.cuh").write_text("// common v1\n")
    (src / "wrapper_core.cu").write_text("// core v1\n")
    (src / "wrapper_kinematics.cu").write_text("// kinematics v1\n")
    (src / "wrapper_dyn_values.cu").write_text("// dyn_values v1\n")
    return fake, src, tmp_path / "objects"


def _build(src, objects, *, reuse=True, core_flags=None, flags=()):
    cmd = ["/fake/nvcc", "-O3", "-I" + str(src), "-ltorch", "-Xlinker", "-rpath,/x", *flags]
    _compile._build_units(cmd, src, src / "robot.so", "sm_120", core_flags=core_flags,
                          object_cache_dir=objects, reuse_objects=reuse)


def test_cold_build_compiles_every_unit_and_fills_the_cache(env):
    fake, src, objects = env
    _build(src, objects)
    assert sorted(fake.compiled) == ["wrapper_core", "wrapper_dyn_values", "wrapper_kinematics"]
    assert fake.links == 1
    assert len(list(objects.glob("*.o"))) == 3
    assert not list((src / "build").glob("*.o")), "cached objects are moved out of the staging dir"
    log = (src / "robot.build.log").read_text()
    assert "[wrapper_core] rc=0" in log


def test_warm_build_links_without_compiling(env):
    fake, src, objects = env
    _build(src, objects)
    fake.compiled.clear()
    _build(src, objects)
    assert fake.compiled == [] and fake.links == 2
    assert "[wrapper_kinematics] cached wrapper_kinematics-" in (src / "robot.build.log").read_text()


def test_unit_edit_recompiles_only_that_unit(env):
    fake, src, objects = env
    _build(src, objects)
    fake.compiled.clear()
    (src / "wrapper_kinematics.cu").write_text("// kinematics v2\n")
    _build(src, objects)
    assert fake.compiled == ["wrapper_kinematics"]
    assert len(list(objects.glob("*.o"))) == 4


@pytest.mark.parametrize("shared", ["grid.cuh", "wrapper_common.cuh"])
def test_shared_header_edit_recompiles_every_unit(env, shared):
    fake, src, objects = env
    _build(src, objects)
    fake.compiled.clear()
    (src / shared).write_text(f"// {shared} v2\n")
    _build(src, objects)
    assert len(fake.compiled) == 3


def test_flag_change_recompiles_every_unit(env):
    fake, src, objects = env
    _build(src, objects)
    fake.compiled.clear()
    _build(src, objects, flags=("-DGRID_RBD_MAX_BATCH=8",))
    assert len(fake.compiled) == 3


def test_torch_namespace_key_touches_only_the_core_object(env):
    """The store key (torch op namespace) is a core-only flag: two store
    entries differing only in that key share every family object."""
    fake, src, objects = env
    _build(src, objects, core_flags=["-DGRID_RBD_TORCH_KEY=kaaa"])
    fake.compiled.clear()
    _build(src, objects, core_flags=["-DGRID_RBD_TORCH_KEY=kbbb"])
    assert fake.compiled == ["wrapper_core"]


def test_staging_dir_path_is_not_part_of_the_key(env, tmp_path):
    """Every build generates into a fresh .gen_<pid> dir; the same sources
    elsewhere must hit."""
    fake, src, objects = env
    _build(src, objects)
    fake.compiled.clear()
    other = tmp_path / "store" / ".gen_2"
    other.mkdir()
    for f in src.glob("*.cu*"):
        (other / f.name).write_bytes(f.read_bytes())
    _build(other, objects)
    assert fake.compiled == []


def test_force_rebuild_recompiles_but_refreshes_the_cache(env):
    fake, src, objects = env
    _build(src, objects)
    fake.compiled.clear()
    _build(src, objects, reuse=False)
    assert len(fake.compiled) == 3
    assert len(list(objects.glob("*.o"))) == 3, "same keys, overwritten in place"


def test_no_cache_dir_deletes_objects_after_link(env):
    fake, src, objects = env
    _build(src, None)
    assert len(fake.compiled) == 3
    assert not list((src / "build").glob("*.o")) and (src / "robot.so").exists()


def test_stale_objects_are_pruned_and_hits_refresh_mtime(env, monkeypatch):
    import os, time
    fake, src, objects = env
    _build(src, objects)
    objs = sorted(objects.glob("*.o"))
    old = time.time() - 30 * 86400
    for o in objs:
        os.utime(o, (old, old))
    # a hit refreshes the object it links from: the three live objects survive the prune
    fake.compiled.clear()
    _build(src, objects)
    assert fake.compiled == [] and sorted(objects.glob("*.o")) == objs
    # an object nothing links from any more ages out
    stale = objects / "wrapper_kinematics-deadbeef.o"
    stale.write_bytes(b"x"); os.utime(stale, (old, old))
    _build(src, objects)
    assert not stale.exists() and len(list(objects.glob("*.o"))) == 3
    # the knob: 0 disables pruning
    stale.write_bytes(b"x"); os.utime(stale, (old, old))
    monkeypatch.setenv("GRID_RBD_OBJECT_CACHE_DAYS", "0")
    _build(src, objects)
    assert stale.exists()
