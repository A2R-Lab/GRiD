"""Gate for the consumer artifact build (HJCD-IK ask G9, with G8b folded in).

`grid_rbd.build_artifact` compiles a caller's translation unit against the `grid.cuh` it
generates for a robot, under the store's rules, in a namespace of the cache directory that
belongs to the consumer. This suite builds `consumer_artifact_kernel.cu` for iiwa14 in a fresh
cache directory and checks, in order: the artifact loads through ctypes and its end-effector
pose equals the RBDReference oracle; the second build of the same request is a pointer hit
(no generation, no nvcc); a changed define re-keys the artifact and recompiles; a second unit
that does not change reuses its object; a tampered identity record is refused and the
content key re-hits without nvcc; `load_artifact` works with nvcc off the PATH and refuses an
artifact recorded for another device; the collision options emit the `grid_collision`
namespace and the sphere-model validator runs; a kinematics-only artifact carries no plant
block unless asked.

Run with:
    pytest test/python_wrappers/test_consumer_artifact.py -m python_wrappers -v
"""
from __future__ import annotations

import ctypes
from concurrent.futures import ThreadPoolExecutor
import json
import shutil
import sys
import sysconfig
import importlib.util
import subprocess
from pathlib import Path

import numpy as np
import pytest

_REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO_ROOT))
from grid_assets import robot_urdf  # noqa: E402

_grid_rbd = pytest.importorskip("grid_rbd", reason="grid-rbd not installed (pip install bindings/)")
if shutil.which("nvcc") is None:
    pytest.skip("nvcc not on PATH; build_artifact requires it", allow_module_level=True)

pytestmark = pytest.mark.python_wrappers
_URDF = robot_urdf("iiwa14")
_KERNEL = Path(__file__).with_name("consumer_artifact_kernel.cu")
_KINEMATICS = ["end_effector_pose"]


def test_consumer_migration_example(tmp_path):
    from test.cuda_equivalents.cuda_harness import _detect_cuda_arch
    script = _REPO_ROOT / "bindings/examples/consumer_artifact_migration.py"
    result = subprocess.run([sys.executable, str(script), "--urdf", str(_URDF),
                             "--work", str(tmp_path / "migration"),
                             "--cuda-arch", str(_detect_cuda_arch())],
                            cwd=tmp_path, capture_output=True, text=True)
    assert result.returncode == 0, result.stdout + result.stderr
    report = json.loads(result.stdout.strip().splitlines()[-1])
    assert report["num_pos"] == 7
    assert report["artifacts"]["native"]["python_extension"] is False
    assert report["artifacts"]["python"]["python_extension"] is True


def test_concurrent_consumer_builds_publish_loadable_winner(tmp_path):
    """Real separately imported builders share a private store, not a mock lock."""
    from test.cuda_equivalents.cuda_harness import _detect_cuda_arch
    source = tmp_path / "shared.cu"
    source.write_text('#include "grid.cuh"\n'
                      'extern "C" __attribute__((visibility("default")))\n'
                      'int concurrent_num_pos() { return grid::NUM_POS; }\n')
    cache = tmp_path / "cache"
    script = '''import ctypes, json, sys
from grid_rbd import build_artifact, load_artifact
art = build_artifact("concurrent", "robot", sys.argv[1], sources=[sys.argv[2]],
                     cache_dir=sys.argv[3], cuda_arch=int(sys.argv[4]),
                     algorithm_list=["end_effector_pose"], enable_mujoco_kernels=False)
loaded = load_artifact("concurrent", "robot", cache_dir=sys.argv[3], cuda_arch=int(sys.argv[4]))
assert loaded.content_key == art.content_key
lib = ctypes.CDLL(str(loaded.path))
assert lib.concurrent_num_pos() == 7
print(json.dumps({"key": loaded.content_key, "path": str(loaded.path)}))
'''
    argv = [sys.executable, "-c", script, str(_URDF), str(source), str(cache), str(_detect_cuda_arch())]
    def build(_):
        result = subprocess.run(argv, cwd=tmp_path, text=True, capture_output=True, timeout=120)
        assert result.returncode == 0, result.stdout + result.stderr
        return json.loads(result.stdout.strip().splitlines()[-1])
    with ThreadPoolExecutor(max_workers=2) as workers:
        first, second = list(workers.map(build, range(2)))
    assert first == second
    assert Path(first["path"]).is_file()


def _build(cache_dir, name="iiwa14", tag=1, **kw):
    opts = dict(sources=[_KERNEL], defines={"CONSUMER_BUILD_TAG": tag}, cache_dir=cache_dir,
                algorithm_list=_KINEMATICS, enable_mujoco_kernels=False)
    opts.update(kw)
    return _grid_rbd.build_artifact("gate", name, _URDF, **opts)


def _pose(artifact, q):
    lib = ctypes.CDLL(str(artifact.path))
    n = lib.consumer_num_pos()
    assert n == len(q)
    pose = np.zeros(6 * lib.consumer_num_ees(), dtype=np.float32)
    free = ctypes.c_int(-7)
    rc = lib.consumer_pose(np.ascontiguousarray(q, dtype=np.float32).ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                           pose.ctypes.data_as(ctypes.POINTER(ctypes.c_float)), ctypes.byref(free))
    assert rc == 0
    return pose, free.value, lib


def _oracle_pose(q):
    from RBDReference import RBDReference
    from URDFParser import URDFParser
    robot = URDFParser().parse(str(_URDF))
    ref = RBDReference(robot)
    return np.asarray(ref.end_effector_pose(np.asarray(q, dtype=np.float64))).reshape(-1)


def test_artifact_builds_loads_and_matches_the_oracle(tmp_path):
    cache = tmp_path / "cache"
    art = _build(cache)
    assert art.compiled and art.path.exists() and art.header.exists()
    assert art.store_dir == cache / "consumers" / "gate" / "store" / art.content_key
    assert (art.store_dir / "build_inputs.json").exists() and (art.store_dir / "build.log").exists()
    meta = json.loads((art.store_dir / "meta.json").read_text())
    assert meta["sources"] == {_KERNEL.name: meta["sources"][_KERNEL.name]} and meta["defines"] == ["CONSUMER_BUILD_TAG=1"]
    rng = np.random.default_rng(3)
    q = rng.uniform(-1.5, 1.5, size=7)
    pose, free, lib = _pose(art, q)
    assert lib.consumer_build_tag() == 1
    assert free == -1          # no collision namespace in this header
    ref = _oracle_pose(q)
    assert pose.shape == ref.shape
    np.testing.assert_allclose(pose, ref, atol=2e-4, rtol=1e-4)
    listed = _grid_rbd.list_artifacts("gate", cache_dir=cache)
    assert [e["name"] for e in listed] == ["iiwa14"] and listed[0]["content_key"] == art.content_key


def test_second_build_is_a_pointer_hit_and_a_define_rekeys(tmp_path):
    cache = tmp_path / "cache"
    first = _build(cache)
    plan = _build(cache, plan_only=True)
    assert plan["cached"] and plan["content_key"] == first.content_key and plan["would"] == "load"
    again = _build(cache)
    assert not again.compiled and again.content_key == first.content_key
    changed = _build(cache, tag=2)
    assert changed.compiled and changed.content_key != first.content_key
    assert _pose(changed, np.zeros(7))[2].consumer_build_tag() == 2
    assert json.loads((changed.store_dir / "meta.json").read_text())["build"]["units"][_KERNEL.name]["compiled"]


def test_unchanged_unit_reuses_its_object(tmp_path):
    cache = tmp_path / "cache"
    other = tmp_path / "other_unit.cu"
    other.write_text('#include "grid.cuh"\nextern "C" __attribute__((visibility("default"))) int consumer_other() { return grid::NUM_POS * 10; }\n')
    first = _build(cache, sources=[_KERNEL, other])
    units = json.loads((first.store_dir / "meta.json").read_text())["build"]["units"]
    assert units[_KERNEL.name]["compiled"] and units[other.name]["compiled"]
    other.write_text('#include "grid.cuh"\nextern "C" __attribute__((visibility("default"))) int consumer_other() { return grid::NUM_POS * 20; }\n')
    second = _build(cache, sources=[_KERNEL, other])
    units = json.loads((second.store_dir / "meta.json").read_text())["build"]["units"]
    assert second.content_key != first.content_key
    assert not units[_KERNEL.name]["compiled"] and units[other.name]["compiled"]
    assert ctypes.CDLL(str(second.path)).consumer_other() == 140


def test_tampered_identity_is_refused_and_the_content_key_rehits(tmp_path):
    cache = tmp_path / "cache"
    first = _build(cache)
    inputs = first.store_dir / "build_inputs.json"
    recorded = json.loads(inputs.read_text())
    recorded["nvcc"] = "another toolchain"
    inputs.write_text(json.dumps(recorded))
    plan = _build(cache, plan_only=True)
    assert not plan["cached"] and any("nvcc" in r for r in plan["stale_reasons"])
    rebuilt = _build(cache)
    assert not rebuilt.compiled and rebuilt.content_key == first.content_key   # same bytes: no nvcc
    assert json.loads(inputs.read_text())["nvcc"] == "another toolchain"       # the record is the entry's; the pointer re-validates on the next call


def test_load_without_a_toolchain_and_device_refusal(tmp_path, monkeypatch):
    cache = tmp_path / "cache"
    first = _build(cache)
    monkeypatch.setenv("PATH", "/nonexistent")
    assert shutil.which("nvcc") is None
    loaded = _grid_rbd.load_artifact("gate", "iiwa14", cache_dir=cache)
    assert loaded.path == first.path and not loaded.compiled and loaded.cuda_arch == first.cuda_arch
    with pytest.raises(_grid_rbd.ArtifactNotFoundError):
        _grid_rbd.load_artifact("gate", "missing", cache_dir=cache)
    with pytest.raises(_grid_rbd.ArtifactIncompatibleError, match="cuda_arch"):
        _grid_rbd.load_artifact("gate", "iiwa14", cache_dir=cache, cuda_arch=first.cuda_arch + 1000)


def test_collision_options_emit_the_namespace_and_validate(tmp_path):
    cache = tmp_path / "cache"
    art = _build(cache, name="iiwa14_collision", collision=True, collision_res=0.08,
                 defines={"CONSUMER_BUILD_TAG": 3, "CONSUMER_HAS_COLLISION": None})
    header = art.header.read_text()
    assert "namespace grid_collision {" in header and "[grid_plant] namespace not emitted" in header
    pose, free, _ = _pose(art, np.zeros(7))
    assert free == 1
    np.testing.assert_allclose(pose, _oracle_pose(np.zeros(7)), atol=2e-4, rtol=1e-4)
    bad = tmp_path / "bad_spheres.urdf"
    bad.write_text('<robot name="x"><link name="only"><collision><geometry><sphere radius="0.1"/></geometry></collision></link></robot>')
    with pytest.raises(ValueError, match="kinematic frames|link set"):
        _build(cache, name="iiwa14_bad", spherized_urdf=bad)
    with pytest.raises(ValueError, match="one of"):
        _build(cache, name="iiwa14_two", collision=True, collision_native=True)


def test_plant_block_follows_the_request(tmp_path):
    cache = tmp_path / "cache"
    without = _build(cache)
    assert "namespace grid_plant {" not in without.header.read_text()
    forced = _build(cache, name="iiwa14_plant", emit_plant=True)
    assert "namespace grid_plant {" in forced.header.read_text() and forced.content_key != without.content_key


def test_declared_python_extension_import_and_prepared_load(tmp_path, monkeypatch):
    import pybind11
    source = tmp_path / "abi_probe.cpp"
    source.write_text('#include <pybind11/pybind11.h>\nextern "C" int abi_num_pos();\n'
                      'PYBIND11_MODULE(grid_abi_probe, m) { m.def("num_pos", &abi_num_pos); }\n')
    cuda_source = tmp_path / "abi_probe.cu"
    cuda_source.write_text('#include "grid.cuh"\nextern "C" int abi_num_pos() { return grid::NUM_POS; }\n')
    opts = dict(sources=[source, cuda_source], include_dirs=[pybind11.get_include(), sysconfig.get_path("include")],
                artifact_name="grid_abi_probe.so", python_extension=True)
    art = _build(tmp_path, name="extension", **opts)
    assert art.meta["python_abi"]["soabi"] == sysconfig.get_config_var("SOABI")
    spec = importlib.util.spec_from_file_location("grid_abi_probe", art.path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    assert module.num_pos() == 7
    assert not _build(tmp_path, name="extension", **opts).compiled
    monkeypatch.setenv("PATH", "/nonexistent")
    loaded = _grid_rbd.load_artifact("gate", "extension", cache_dir=tmp_path, cuda_arch=art.cuda_arch)
    assert loaded.path == art.path


def test_undeclared_python_extension_is_not_published(tmp_path):
    source = tmp_path / "undeclared.cpp"
    source.write_text('extern "C" __attribute__((visibility("default"))) void* PyInit_undeclared() { return nullptr; }\n')
    with pytest.raises(ValueError, match="python_extension"):
        _build(tmp_path, name="undeclared", sources=[source])
    assert not _grid_rbd.list_artifacts("gate", cache_dir=tmp_path)


def test_source_path_macro_survives_cache_reuse(tmp_path):
    keys = []
    for folder in ("first", "second"):
        source = tmp_path / folder / "same.cpp"
        source.parent.mkdir()
        source.write_text('extern "C" __attribute__((visibility("default"))) const char* source_path() { return __FILE__; }\n')
        art = _build(tmp_path / "cache", name=folder, sources=[source])
        lib = ctypes.CDLL(str(art.path))
        lib.source_path.restype = ctypes.c_char_p
        assert lib.source_path().decode() == str(source)
        keys.append(art.content_key)
    assert len(set(keys)) == 2


def test_two_robot_consumer_statics_are_isolated_in_both_load_orders(tmp_path):
    source = tmp_path / "state.cu"
    source.write_text('''#include "grid.cuh"
inline int& consumer_state() { static int value = grid::NUM_POS; return value; }
extern "C" __attribute__((visibility("default"))) int state_get() { return consumer_state(); }
extern "C" __attribute__((visibility("default"))) void state_set(int n) { consumer_state() = n; }
extern "C" __attribute__((visibility("default"))) void* state_address() { return &consumer_state(); }
''')
    artifacts = []
    for name, urdf, n in (("arm", _URDF, 7), ("slider", robot_urdf("slider_fixed_tool"), 1)):
        art = _grid_rbd.build_artifact("isolation", name, urdf, sources=[source], cache_dir=tmp_path,
                                      algorithm_list=["end_effector_pose"], enable_mujoco_kernels=False)
        symbols = subprocess.run(["nm", "-D", "--defined-only", str(art.path)],
                                 capture_output=True, text=True, check=True).stdout
        assert not any(line.split()[1] == "u" for line in symbols.splitlines() if len(line.split()) >= 3)
        assert "consumer_state" not in symbols
        for export in ("state_get", "state_set", "state_address"):
            assert any(line.split()[-1] == export for line in symbols.splitlines())
        artifacts.append((str(art.path), n))
    # New interpreter for each order: unloading a CUDA shared library is not a
    # supported reset mechanism, and previous imports would mask order dependence.
    program = '''import ctypes, json, sys
pairs = json.loads(sys.argv[1])
libs = [ctypes.CDLL(path) for path, n in pairs]
for lib, (_, n) in zip(libs, pairs):
    lib.state_address.restype = ctypes.c_void_p
    assert lib.state_get() == n
assert libs[0].state_address() != libs[1].state_address()
libs[0].state_set(101)
assert libs[0].state_get() == 101 and libs[1].state_get() == pairs[1][1]
libs[1].state_set(202)
assert libs[0].state_get() == 101 and libs[1].state_get() == 202
'''
    for ordered in (artifacts, artifacts[::-1]):
        subprocess.run([sys.executable, "-c", program, json.dumps(ordered)], check=True)
