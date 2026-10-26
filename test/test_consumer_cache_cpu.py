"""Consumer artifact cache recovery and identity without CUDA or a compiler."""
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from grid_rbd import _artifacts as artifacts


@pytest.fixture
def consumer(tmp_path, monkeypatch):
    urdf = tmp_path / "robot.urdf"
    urdf.write_text('<robot name="probe"><link name="base"/></robot>')
    source = tmp_path / "kernel.cu"
    source.write_text('// consumer\n')
    state = SimpleNamespace(generated=0, compiled=0, urdf=urdf, source=source)
    monkeypatch.setattr(artifacts, "artifact_identity", lambda arch: {"key_schema": 1, "cuda_arch": arch})
    monkeypatch.setattr(artifacts, "_compile_command", lambda gen, *a: ["fake-nvcc", f"-I{gen}"])
    monkeypatch.setattr(artifacts, "_toolchain_tag", lambda: "fake-toolchain")
    monkeypatch.setattr(artifacts, "_validate_python_exports", lambda *a: None)
    monkeypatch.setattr(artifacts, "resolve_consumer_toolchain", lambda *a: None)

    def generate(urdf, opts, out):
        state.generated += 1
        out.write_text('// header\n')

    def build(gen, sources, headers, cmd, flags, out, objects, reuse_objects, validate_inputs=None,
              python_abi=None, toolchain=None):
        state.compiled += 1
        out.write_bytes(b"artifact")
        return {"compiled_units": 1, "units": {}}

    monkeypatch.setattr(artifacts, "generate_grid_cuh", generate)
    monkeypatch.setattr(artifacts, "_build", build)
    state.build = lambda **kw: artifacts.build_artifact(
        "cpu-probe", "robot", urdf, sources=[source], cuda_arch=120,
        cache_dir=tmp_path / "cache", **kw)
    return state


@pytest.mark.parametrize("missing", ["artifact.so", "meta.json", "grid.cuh"])
def test_missing_entry_file_is_repaired(consumer, missing):
    first = consumer.build()
    (first.store_dir / missing).unlink()
    second = consumer.build()
    assert second.content_key == first.content_key
    assert (consumer.generated, consumer.compiled) == (2, 2)
    assert (second.store_dir / missing).is_file()
    consumer.build()
    assert (consumer.generated, consumer.compiled) == (2, 2)


def test_bad_metadata_is_repaired(consumer):
    first = consumer.build()
    (first.store_dir / "meta.json").write_text("not json")
    second = consumer.build()
    assert second.content_key == first.content_key and consumer.compiled == 2
    assert json.loads((second.store_dir / "meta.json").read_text())["consumer"] == "cpu-probe"


def test_same_stem_sources_produce_distinct_objects_and_links(tmp_path, monkeypatch):
    gen = tmp_path / "generated"
    gen.mkdir()
    (gen / "grid.cuh").write_text("// header")
    sources = [tmp_path / "kernel.cu", tmp_path / "kernel.cpp"]
    for p in sources:
        p.write_text("// identical bytes, different source languages")
    monkeypatch.setattr(artifacts, "_toolchain_tag", lambda: "fake")
    monkeypatch.setattr(artifacts, "_build_jobs", lambda: 2)
    paths = []
    linked = []

    def nvcc(cmd, **kw):
        out = Path(cmd[cmd.index("-o") + 1])
        if "-c" in cmd:
            source = Path(cmd[cmd.index("-c") + 1])
            paths.append(out)
            out.write_bytes(source.name.encode())
        else:
            linked.extend(Path(a).read_bytes() for a in cmd if a.endswith(".o"))
            out.write_bytes(b"shared")
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(artifacts.subprocess, "run", nvcc)
    artifacts._build(gen, sources, [], ["fake"], [], gen / "artifact.so", tmp_path / "objects", True)
    assert len(set(paths)) == 2
    assert linked == [b"kernel.cu", b"kernel.cpp"]


def test_source_language_participates_in_object_key(tmp_path, monkeypatch):
    (tmp_path / "grid.cuh").write_text("// header")
    sources = [tmp_path / "kernel.cu", tmp_path / "kernel.cpp"]
    for source in sources:
        source.write_text("same")
    monkeypatch.setattr(artifacts, "_toolchain_tag", lambda: "fake")
    keys = [artifacts._unit_object_key(p, tmp_path, [], ["nvcc"]) for p in sources]
    assert keys[0] != keys[1]


def test_source_location_participates_in_all_reuse_keys(tmp_path, monkeypatch):
    # nvcc sees an absolute source path, which is observable through __FILE__.
    (tmp_path / "grid.cuh").write_text("header")
    sources = []
    for folder in ("first", "second"):
        source = tmp_path / folder / "same.cpp"
        source.parent.mkdir()
        source.write_text('const char *file = __FILE__;')
        sources.append(source)
    monkeypatch.setattr(artifacts, "_toolchain_tag", lambda: "fake")
    inputs = [artifacts._input_key(b"urdf", {}, [p], [], [], [], [], [], "artifact.so", {}) for p in sources]
    contents = [artifacts._content_key(tmp_path, [p], [], ["nvcc"], [], "artifact.so") for p in sources]
    units = [artifacts._unit_object_key(p, tmp_path, [], ["nvcc"]) for p in sources]
    for keys in (inputs, contents, units):
        assert len(set(keys)) == 2


def test_collision_mesh_change_invalidates_fast_pointer(consumer):
    mesh = consumer.urdf.parent / "body.stl"
    mesh.write_bytes(b"mesh-one")
    consumer.urdf.write_text('<robot name="probe"><link name="base"><collision><geometry>'
                             '<mesh filename="body.stl"/></geometry></collision></link></robot>')
    first = consumer.build(collision=True, plan_only=True)
    consumer.build(collision=True)
    assert consumer.build(collision=True, plan_only=True)["cached"]
    mesh.write_bytes(b"mesh-two")
    second = consumer.build(collision=True, plan_only=True)
    assert first["input_key"] != second["input_key"] and not second["cached"]
    mesh.unlink()
    with pytest.raises(ValueError, match="could not be resolved"):
        consumer.build(collision=True)


def test_package_resolution_participates_in_identity(tmp_path, monkeypatch):
    urdf_dir = tmp_path / "robot"
    urdf_dir.mkdir()
    urdf = urdf_dir / "robot.urdf"
    urdf.write_text('<robot><link name="base"><collision><geometry>'
                    '<mesh filename="package://pkg/body.stl"/></geometry></collision></link></robot>')
    records = []
    for name in ("first", "second"):
        root = tmp_path / name / "pkg"
        root.mkdir(parents=True)
        (root / "body.stl").write_bytes(b"same geometry")
        monkeypatch.setenv("ROS_PACKAGE_PATH", str(root.parent))
        records.append(artifacts._collision_mesh_identity(urdf))
    assert records[0][0]["sha256"] == records[1][0]["sha256"]
    assert records[0][0]["path"] != records[1][0]["path"]


@pytest.mark.parametrize("phase", ["generate", "compile"])
@pytest.mark.parametrize("target", ["urdf", "source"])
def test_mutation_does_not_publish_under_earlier_key(consumer, monkeypatch, phase, target):
    path = getattr(consumer, target)
    original_bytes = path.read_bytes()
    name = "generate_grid_cuh" if phase == "generate" else "_build"
    original = getattr(artifacts, name)

    def mutate(*a, **kw):
        value = original(*a, **kw)
        path.write_bytes(original_bytes + b"\n")
        return value

    monkeypatch.setattr(artifacts, name, mutate)
    with pytest.raises(RuntimeError, match="inputs changed"):
        consumer.build()
    path.write_bytes(original_bytes)
    monkeypatch.setattr(artifacts, name, original)
    assert not consumer.build(plan_only=True)["cached"]
    assert consumer.build().compiled


def test_changed_source_is_rejected_before_object_publication(tmp_path, monkeypatch):
    gen = tmp_path / "gen"
    gen.mkdir()
    (gen / "grid.cuh").write_text("header")
    source = tmp_path / "kernel.cu"
    source.write_text("original")
    objects = tmp_path / "objects"
    monkeypatch.setattr(artifacts, "_toolchain_tag", lambda: "fake")

    def validate():
        if source.read_text() != "original":
            raise RuntimeError("changed source")

    def nvcc(cmd, **kw):
        source.write_text("changed")
        Path(cmd[cmd.index("-o") + 1]).write_bytes(b"object from changed source")
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(artifacts.subprocess, "run", nvcc)
    with pytest.raises(RuntimeError, match="changed source"):
        artifacts._build(gen, [source], [], ["fake"], [], gen / "artifact.so", objects,
                         True, validate_inputs=validate)
    assert not list(objects.glob("*.o"))
