"""The error messages a new user meets, read as product guidance.

Five situations and the grid-generate console script, every one exercised on
the CPU: nvcc off PATH, a rejected URDF, a missing framework extra, a stale or
damaged cache entry, and the CLI's input errors. No GPU, no nvcc, no codegen
beyond the URDF parse: where a build would otherwise start, the generation and
compile halves are replaced by stubs that write the files the store expects.
Each assertion pins the wording that tells the user the cause and the next
action (the call, flag or environment variable that fixes it)."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

import grid_rbd
from grid_rbd import _cache, _compile

ARCH = 120

URDF_OK = """<robot name="okbot">
  <link name="base"/>
  <link name="l1">
    <inertial><mass value="1"/><origin xyz="0 0 0.1"/>
      <inertia ixx="0.1" iyy="0.1" izz="0.1" ixy="0" ixz="0" iyz="0"/></inertial>
  </link>
  <joint name="j1" type="revolute">
    <parent link="base"/><child link="l1"/>
    <axis xyz="0 0 1"/><limit lower="-1" upper="1" effort="10" velocity="1"/>
  </joint>
</robot>
"""

URDF_NO_ROBOT = """<?xml version="1.0"?>
<model name="x"><link name="base"/></model>
"""

URDF_BAD_JOINT = """<robot name="badjoint">
  <link name="base"/>
  <link name="l1">
    <inertial><mass value="1"/>
      <inertia ixx="1" iyy="1" izz="1" ixy="0" ixz="0" iyz="0"/></inertial>
  </link>
  <joint name="j1" type="ball">
    <parent link="base"/><child link="l1"/>
    <axis xyz="0 0 1"/>
  </joint>
</robot>
"""

URDF_MESH = URDF_OK.replace(
    "</inertial>\n  </link>",
    "</inertial>\n    <collision><geometry>"
    "<mesh filename=\"package://nowhere/meshes/l1.stl\"/></geometry></collision>\n  </link>")


# ─── stubs: the build halves never run ───────────────────────────────────────

class _Stubs:
    def __init__(self):
        self.generate_calls = 0
        self.compile_calls = 0

    def generate_sources(self, urdf_path, options, out_dir):
        self.generate_calls += 1
        out_dir = Path(out_dir)
        (out_dir / "grid.cuh").write_text("// grid.cuh stub\n")
        (out_dir / "wrapper.cu").write_text("// wrapper\n")
        (out_dir / "wrapper_common.cuh").write_text("// common\n")
        return {"NUM_JOINTS": 1, "NUM_VEL": 1, "NUM_EES": 0}

    def compile_sources(self, gen_dir, meta, options, *, cuda_arch, max_batch,
                        torch_op_key, object_cache_dir=None, reuse_objects=True):
        self.compile_calls += 1
        gen_dir = Path(gen_dir)
        (gen_dir / "robot.so").write_bytes(b"\x7fELF-stub")
        meta = dict(meta, cuda_arch=cuda_arch, max_batch=max_batch)
        (gen_dir / "meta.json").write_text(json.dumps(meta))
        return meta


class _HandleStub:
    def __init__(self, name, so_path, entry):
        self.name, self.so_path, self.entry = name, so_path, entry

    def apply_profile_overlay(self, *_):
        pass


@pytest.fixture
def cache_dir(tmp_path):
    return tmp_path / "grid-rbd"


@pytest.fixture
def stubs(monkeypatch):
    st = _Stubs()
    monkeypatch.setattr(grid_rbd, "generate_sources", st.generate_sources)
    monkeypatch.setattr(grid_rbd, "compile_sources", st.compile_sources)
    return st


@pytest.fixture
def no_toolchain(monkeypatch, tmp_path):
    """A PATH with neither nvcc nor nvidia-smi on it."""
    empty = tmp_path / "emptybin"
    empty.mkdir()
    monkeypatch.setenv("PATH", str(empty))


@pytest.fixture
def loader_stubbed(monkeypatch):
    monkeypatch.setattr(grid_rbd, "RobotHandle", _HandleStub)
    monkeypatch.setattr(grid_rbd, "detect_cuda_arch", lambda: ARCH)


def _fake_entry(cache_dir, name, key="deadbeefcafe", *, with_so=True, identity=None):
    """A manifest binding plus a store entry, as a finished build leaves them."""
    entry = _cache.store_dir(cache_dir, key)
    entry.mkdir(parents=True, exist_ok=True)
    meta = {"NUM_JOINTS": 1, "NUM_VEL": 1, "NUM_EES": 0, "cuda_arch": ARCH, "max_batch": 8}
    (entry / "meta.json").write_text(json.dumps(meta))
    if with_so:
        (entry / "robot.so").write_bytes(b"\x7fELF-stub")
    if identity is not None:
        _cache.write_build_inputs(entry, identity)
    _cache.manifest_register(cache_dir, name, key, meta)
    return entry


# ─── 1. no nvcc on PATH ───────────────────────────────────────────────────────

def test_find_nvcc_names_the_toolkit_and_the_call_that_needs_it(no_toolchain):
    with pytest.raises(RuntimeError) as ei:
        _compile.find_nvcc()
    msg = str(ei.value)
    assert "nvcc not found" in msg and "CUDA Toolkit" in msg and "register_robot" in msg


def test_register_without_nvcc_fails_before_any_build(no_toolchain, stubs, monkeypatch, cache_dir):
    # Real compile half: it must stop at the nvcc lookup, with the toolkit message.
    monkeypatch.setattr(grid_rbd, "compile_sources", _compile.compile_sources)
    with pytest.raises(RuntimeError, match="nvcc not found") as ei:
        grid_rbd.register_robot("probe", urdf_string=URDF_OK, cache_dir=cache_dir, cuda_arch=ARCH)
    assert "CUDA Toolkit" in str(ei.value)
    assert stubs.generate_calls == 1
    assert not list((cache_dir / "store").glob("*/robot.so")), "nothing was published"
    with pytest.raises(RuntimeError, match="nvcc not found"):
        grid_rbd.precompile("probe", urdf_string=URDF_OK, cache_dir=cache_dir, cuda_arch=ARCH)
    with pytest.raises(RuntimeError, match="nvcc not found"):
        grid_rbd.load_robot(urdf_string=URDF_OK, cache_dir=cache_dir, cuda_arch=ARCH)


def test_no_gpu_visible_says_how_to_pass_the_arch(no_toolchain, stubs, cache_dir):
    with pytest.raises(RuntimeError) as ei:
        grid_rbd.register_robot("probe", urdf_string=URDF_OK, cache_dir=cache_dir)
    msg = str(ei.value)
    assert "nvidia-smi" in msg and "cuda_arch=" in msg and "cuda_arch=120" in msg
    assert stubs.generate_calls == 0


# ─── 3. rejected URDFs (the parser runs for real; no codegen follows) ─────────

def _register(path, cache_dir, **kw):
    return grid_rbd.register_robot("rejected", urdf_path=str(path), cache_dir=cache_dir,
                                   cuda_arch=ARCH, **kw)


def test_missing_urdf_file_names_the_argument(cache_dir, tmp_path):
    missing = tmp_path / "nowhere" / "robot.urdf"
    with pytest.raises(FileNotFoundError) as ei:
        _register(missing, cache_dir)
    msg = str(ei.value)
    assert "URDF file not found" in msg and str(missing) in msg
    assert "urdf_path=" in msg and "urdf_string=" in msg


def test_both_or_neither_urdf_source_is_refused(cache_dir):
    with pytest.raises(ValueError, match="exactly one of urdf_path="):
        grid_rbd.register_robot("x", urdf_path="a.urdf", urdf_string=URDF_OK,
                                cache_dir=cache_dir, cuda_arch=ARCH)
    with pytest.raises(ValueError, match="exactly one of urdf_path="):
        grid_rbd.load_robot(cache_dir=cache_dir, cuda_arch=ARCH)


def test_urdf_without_a_robot_element_is_rejected_by_name(cache_dir, tmp_path):
    from URDFParser import URDFParseError
    f = tmp_path / "no_robot.urdf"
    f.write_text(URDF_NO_ROBOT)
    with pytest.raises(URDFParseError, match="no <robot> element") as ei:
        _register(f, cache_dir)
    assert str(f) in str(ei.value)
    assert not (cache_dir / "store").exists() or not list((cache_dir / "store").glob("*/robot.so"))


def test_unsupported_joint_type_lists_the_supported_ones(cache_dir, tmp_path):
    from URDFParser import URDFParseError, UnsupportedJointTypeError
    f = tmp_path / "bad_joint.urdf"
    f.write_text(URDF_BAD_JOINT)
    with pytest.raises(URDFParseError) as ei:
        _register(f, cache_dir)
    assert isinstance(ei.value, UnsupportedJointTypeError)
    msg = str(ei.value)
    assert "Unsupported joint type 'ball'" in msg and "joint 'j1'" in msg
    assert "Supported joint types are" in msg and "revolute" in msg and "spherical" in msg


def test_unreadable_urdf_bytes_name_the_file(cache_dir, tmp_path):
    from URDFParser import URDFParseError
    f = tmp_path / "binary.urdf"
    f.write_bytes(b"\xff\xfe\x00garbage")
    with pytest.raises(URDFParseError, match="Failed to parse URDF") as ei:
        _register(f, cache_dir)
    assert str(f) in str(ei.value)


# ─── 4. missing framework extra ──────────────────────────────────────────────

def _assert_install_hint(msg, framework, extra):
    assert framework in msg and "not installed" in msg
    assert f'pip install "grid-rbd[{extra}]"' in msg
    assert "numpy" in msg, "says which backends keep working"


def test_torch_surface_without_torch_gives_the_install_hint(monkeypatch, cache_dir):
    import grid_rbd.torch as gt
    monkeypatch.setitem(sys.modules, "torch", None)
    with pytest.raises(ImportError) as ei:
        gt.get_robot("x", cache_dir=cache_dir)
    _assert_install_hint(str(ei.value), "PyTorch", "torch")
    assert "cu128" in str(ei.value), "names the wheel family an sm_120 card needs"
    with pytest.raises(ImportError) as ei:
        grid_rbd.register_robot("x", urdf_string=URDF_OK, backend="torch",
                                cache_dir=cache_dir, cuda_arch=ARCH)
    _assert_install_hint(str(ei.value), "PyTorch", "torch")
    with pytest.raises(ImportError):
        gt.pinned_host_like([])


def test_jax_surface_without_jax_gives_the_install_hint(monkeypatch, cache_dir):
    import grid_rbd.jax as gj
    monkeypatch.setitem(sys.modules, "jax", None)
    with pytest.raises(ImportError) as ei:
        gj.register_robot("x", urdf_string=URDF_OK, cache_dir=cache_dir, cuda_arch=ARCH)
    _assert_install_hint(str(ei.value), "JAX", "jax")
    assert 'jax[cuda12]' in str(ei.value)
    with pytest.raises(ImportError) as ei:
        grid_rbd.get_robot("x", backend="jax", cache_dir=cache_dir)
    _assert_install_hint(str(ei.value), "JAX", "jax")


def test_cpu_only_jax_names_the_cuda_wheel(monkeypatch):
    jax = pytest.importorskip("jax")
    from grid_rbd.jax import _require_gpu_backend
    monkeypatch.setattr(jax, "default_backend", lambda: "cpu")
    with pytest.raises(RuntimeError, match=r"jax\[cuda12\]") as ei:
        _require_gpu_backend()
    assert "'cpu'" in str(ei.value)


# ─── 5. stale or damaged cache entries ───────────────────────────────────────

def test_unregistered_name_offers_the_three_ways_in(cache_dir):
    with pytest.raises(grid_rbd.RobotNotRegisteredError) as ei:
        grid_rbd.get_robot("nope", cache_dir=cache_dir)
    assert isinstance(ei.value, KeyError)
    msg = str(ei.value)
    assert "No robot registered under name 'nope'" in msg
    assert "register_robot" in msg and "get_robot('nope', urdf_path=...)" in msg and "load_robot" in msg


def test_build_options_without_a_urdf_are_refused(cache_dir):
    with pytest.raises(TypeError) as ei:
        grid_rbd.get_robot("nope", cache_dir=cache_dir, floating_base=True)
    msg = str(ei.value)
    assert "['floating_base']" in msg and "urdf_path=" in msg and "looked up by name" in msg


def test_missing_library_says_rebuild_and_gc(loader_stubbed, cache_dir):
    entry = _fake_entry(cache_dir, "probe", with_so=False, identity=_cache.build_identity(ARCH))
    with pytest.raises(RuntimeError) as ei:
        grid_rbd.get_robot("probe", cache_dir=cache_dir)
    msg = str(ei.value)
    assert "compiled library is missing" in msg and str(entry / "robot.so") in msg
    assert "force_rebuild=True" in msg and "gc_cache" in msg


def test_entry_from_before_the_identity_record_is_stale(loader_stubbed, cache_dir):
    _fake_entry(cache_dir, "probe", identity=None)
    with pytest.raises(grid_rbd.StaleRobotError, match="predates") as ei:
        grid_rbd.get_robot("probe", cache_dir=cache_dir)
    msg = str(ei.value)
    assert isinstance(ei.value, RuntimeError)
    assert "cannot be loaded" in msg and "register_robot(name='probe'" in msg and "precompile" in msg
    assert ei.value.reasons and "predates" in ei.value.reasons[0]


def test_entry_from_another_wrapper_abi_is_stale_with_the_reason(loader_stubbed, cache_dir):
    ident = _cache.build_identity(ARCH)
    _fake_entry(cache_dir, "probe", identity=dict(ident, wrapper_template="deadbeef"))
    with pytest.raises(grid_rbd.StaleRobotError) as ei:
        grid_rbd.get_robot("probe", cache_dir=cache_dir)
    assert ei.value.reasons == [
        f"wrapper_template: recorded 'deadbeef' != current {ident['wrapper_template']!r}"]
    assert "different grid_rbd setup" in str(ei.value) and "register_robot" in str(ei.value)
    # provenance alone (which nvcc built it) is not a load blocker
    _fake_entry(cache_dir, "probe2", key="feedfacefeed", identity=dict(ident, nvcc="/old/nvcc:11.0"))
    assert grid_rbd.get_robot("probe2", cache_dir=cache_dir).name == "probe2"


def test_dangling_pointer_is_rebuilt_not_raised(stubs, cache_dir, caplog):
    import logging
    caplog.set_level(logging.INFO, logger="grid_rbd")
    key, so, _ = grid_rbd.warm_robot("probe", urdf_string=URDF_OK, cache_dir=cache_dir, cuda_arch=ARCH)
    assert stubs.compile_calls == 1
    # The pointer now names an entry that is gone (a corrupt or half-cleaned cache).
    pointer = next(p for p in (cache_dir / "bykey").iterdir() if not p.name.startswith("."))
    pointer.write_text("0000000000000000000000000000000000000000000000000000000000000000\n")
    key2, so2, _ = grid_rbd.warm_robot("probe", urdf_string=URDF_OK, cache_dir=cache_dir, cuda_arch=ARCH)
    assert key2 == key and so2 == so, "identical sources re-hit the content store"
    assert stubs.generate_calls == 2 and stubs.compile_calls == 1, "regenerated, not recompiled"
    assert any("rejected" in r.message and "incomplete" in r.message for r in caplog.records)


def test_manifest_from_another_schema_says_which_cache_to_clear(cache_dir):
    cache_dir.mkdir(parents=True)
    path = _cache.manifest_path(cache_dir)
    path.write_text(json.dumps({"schema_version": 999, "robots": {}}))
    with pytest.raises(RuntimeError) as ei:
        grid_rbd.get_robot("probe", cache_dir=cache_dir)
    msg = str(ei.value)
    assert "schema_version" in msg and str(path) in msg and "999" in msg


def test_old_artifact_without_joint_layout_says_re_register():
    from grid_rbd._configuration import configuration_layout_from_meta
    with pytest.raises(ValueError, match="re-register") as ei:
        configuration_layout_from_meta({"floating_base": False}, nq=5, nv=4)
    assert "force_rebuild=True" in str(ei.value)
    with pytest.raises(ValueError, match="re-register"):
        configuration_layout_from_meta({"configuration_layout": [("bogus", 0, 0, 2, 1)]}, nq=2, nv=1)


# ─── grid-generate (grid_codegen.cli) ────────────────────────────────────────

def _grid_generate(monkeypatch, capsys, *argv):
    from grid_codegen import cli
    monkeypatch.setattr(sys, "argv", ["grid-generate", *map(str, argv)])
    with pytest.raises(SystemExit) as ei:
        cli.main()
    captured = capsys.readouterr()
    return ei.value.code, captured.out + captured.err


def test_cli_without_a_urdf_argument_exits_2(monkeypatch, capsys):
    code, out = _grid_generate(monkeypatch, capsys)
    assert code == 2 and "urdf_path" in out


def test_cli_missing_file_exits_1_with_the_path(monkeypatch, capsys, tmp_path):
    missing = tmp_path / "nope.urdf"
    code, out = _grid_generate(monkeypatch, capsys, missing)
    assert code == 1
    assert "[!Error] URDF file not found: %s" % missing in out
    assert "run grid-generate again" in out


def test_cli_parse_failure_exits_1_with_the_parser_reason(monkeypatch, capsys, tmp_path):
    f = tmp_path / "no_robot.urdf"
    f.write_text(URDF_NO_ROBOT)
    code, out = _grid_generate(monkeypatch, capsys, f)
    assert code == 1
    assert "[!Error] URDF parsing failed:" in out and "no <robot> element" in out
    assert "Traceback" not in out and "Fix the model file" in out
    f = tmp_path / "bad_joint.urdf"
    f.write_text(URDF_BAD_JOINT)
    code, out = _grid_generate(monkeypatch, capsys, f)
    assert code == 1
    assert "[!Error] URDF parsing failed:" in out and "Unsupported joint type 'ball'" in out
    assert "Supported joint types are" in out and "Traceback" not in out


def test_cli_unknown_algorithm_exits_1_with_examples(monkeypatch, capsys, tmp_path):
    f = tmp_path / "ok.urdf"
    f.write_text(URDF_OK)
    code, out = _grid_generate(monkeypatch, capsys, f, "--algorithm-list", "inverse_dynamics,bogus_algo")
    assert code == 1
    assert "[!Error] --algorithm-list was not understood" in out and "bogus_algo" in out
    assert "inverse_dynamics, minv" in out and "dynamics" in out
    assert "Traceback" not in out


def test_cli_bad_collision_res_exits_1(monkeypatch, capsys, tmp_path):
    f = tmp_path / "ok.urdf"
    f.write_text(URDF_OK)
    code, out = _grid_generate(monkeypatch, capsys, f, "--collision-res", "abc")
    assert code == 1
    assert "[!Error] --collision-res expects" in out and "'abc'" in out and "0.10,0.05" in out


def test_cli_conflicting_geometry_flags_exit_2(monkeypatch, capsys, tmp_path):
    f = tmp_path / "ok.urdf"
    f.write_text(URDF_OK)
    code, out = _grid_generate(monkeypatch, capsys, f, "--spherized-urdf", "foo", "--collision-native")
    assert code == 2 and "different geometry models" in out


def test_cli_unresolvable_mesh_exits_1_with_the_asset_hint(monkeypatch, capsys, tmp_path):
    f = tmp_path / "mesh.urdf"
    f.write_text(URDF_MESH)
    code, out = _grid_generate(monkeypatch, capsys, f, "-c")
    assert code == 1
    assert "[!Error] Collision geometry could not be built" in out
    assert "package://nowhere/meshes/l1.stl" in out and "could not be resolved" in out
    assert "ROS_PACKAGE_PATH" in out and "Traceback" not in out
