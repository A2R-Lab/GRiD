"""Python ABI must guard every consumer reuse layer, not only prepared loads."""
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from grid_rbd import _artifacts as artifacts
from test._consumer_model_fixture import FakeToolchain, generate_metadata_header


@pytest.fixture
def build_case(tmp_path, monkeypatch):
    urdf = tmp_path / "robot.urdf"
    urdf.write_text('<robot name="probe"><link name="base"/></robot>')
    source = tmp_path / "extension.cpp"
    source.write_text("// unchanged consumer source")
    state = SimpleNamespace(builds=0, abi={"soabi": "cpython-test-a"})
    monkeypatch.setattr(artifacts, "_python_abi_identity", lambda: dict(state.abi), raising=False)
    monkeypatch.setattr(artifacts, "artifact_identity", lambda arch: {
        "key_schema": artifacts._KEY_SCHEMA, "artifact_schema": artifacts.ARTIFACT_SCHEMA,
        "cuda_arch": arch})
    monkeypatch.setattr(artifacts, "_compile_command", lambda gen, *a: ["fake", f"-I{gen}"])
    monkeypatch.setattr(artifacts, "_toolchain_tag", lambda: "fake")
    monkeypatch.setattr(artifacts, "generate_grid_cuh", generate_metadata_header)
    monkeypatch.setattr(artifacts, "_validate_python_exports", lambda *a: None, raising=False)
    monkeypatch.setattr(artifacts, "resolve_consumer_toolchain", lambda *a: FakeToolchain())
    monkeypatch.setattr(artifacts, "discover_dependencies", lambda *a: {
        "schema": 1, "mode": "nvcc-MM", "units": {str(source): []}})

    def compile(gen, srcs, hdrs, cmd, flags, out, objects, **kw):
        state.builds += 1
        out.write_bytes(b"artifact")
        return {"compiled_units": 1, "units": {}}

    monkeypatch.setattr(artifacts, "_build", compile)
    state.build = lambda **kw: artifacts.build_artifact(
        "abi", "robot", urdf, sources=[source], cuda_arch=120, cache_dir=tmp_path, **kw)
    state.load = lambda: artifacts.load_artifact("abi", "robot", cache_dir=tmp_path, cuda_arch=120)
    return state


def test_python_abi_guards_pointer_content_and_load(build_case):
    first = build_case.build(python_extension=True)
    assert not build_case.build(python_extension=True).compiled
    assert build_case.load().path == first.path
    build_case.abi = {"soabi": "cpython-test-b"}
    assert not build_case.build(python_extension=True, plan_only=True)["cached"]
    with pytest.raises(artifacts.ArtifactIncompatibleError, match="python_abi"):
        build_case.load()
    second = build_case.build(python_extension=True)
    assert second.content_key != first.content_key
    assert build_case.builds == 2
    assert build_case.load().path == second.path


def test_native_artifact_does_not_depend_on_python_abi(build_case, monkeypatch):
    first = build_case.build()
    monkeypatch.setattr(artifacts, "_python_abi_identity", lambda: pytest.fail("native used Python ABI"))
    assert not build_case.build().compiled
    assert build_case.load().path == first.path


def test_extension_declaration_cannot_reuse_native_content(build_case):
    native = build_case.build()
    extension = build_case.build(python_extension=True)
    assert extension.content_key != native.content_key
    assert build_case.builds == 2


def test_python_abi_participates_in_unit_key(tmp_path, monkeypatch):
    (tmp_path / "grid.cuh").write_text("header")
    unit = tmp_path / "module.cpp"
    unit.write_text("// identical source")
    monkeypatch.setattr(artifacts, "_toolchain_tag", lambda: "fake")
    keys = [artifacts._unit_object_key(unit, tmp_path, [], ["nvcc"], python_abi=abi, toolchain=None,
                                       dependencies=None)
            for abi in (None, {"soabi": "a"}, {"soabi": "b"})]
    assert len(set(keys)) == 3


def test_ambiguous_legacy_prepared_artifact_is_refused(build_case):
    first = build_case.build()
    record = first.store_dir / "build_inputs.json"
    data = json.loads(record.read_text())
    data.pop("python_abi", None)
    record.write_text(json.dumps(data))
    with pytest.raises(artifacts.ArtifactIncompatibleError, match="python_abi"):
        build_case.load()
    assert not build_case.build(plan_only=True)["cached"]
    assert build_case.build().compiled
    assert build_case.load().path == first.path


def test_incompatible_abi_sidecar_cannot_be_used_on_a_content_hit(build_case):
    first = build_case.build(python_extension=True)
    record = first.store_dir / "build_inputs.json"
    data = json.loads(record.read_text())
    data["python_abi"] = {"soabi": "incompatible"}
    record.write_text(json.dumps(data))
    assert build_case.build(python_extension=True).compiled
    assert build_case.load().path == first.path


def test_extension_cannot_be_relabeled_native_by_a_damaged_sidecar(build_case):
    first = build_case.build(python_extension=True)
    record = first.store_dir / "build_inputs.json"
    data = json.loads(record.read_text())
    data["python_abi"] = None
    record.write_text(json.dumps(data))
    with pytest.raises(artifacts.ArtifactIncompatibleError, match="python_abi"):
        build_case.load()


@pytest.mark.parametrize("field,value", [("python_abi", None), ("python_extension", False)])
def test_pointer_hit_must_not_trust_damaged_metadata(build_case, field, value):
    first = build_case.build(python_extension=True)
    path = first.store_dir / "meta.json"
    meta = json.loads(path.read_text())
    meta[field] = value
    path.write_text(json.dumps(meta))
    with pytest.raises(artifacts.ArtifactIncompatibleError, match="python_abi"):
        build_case.load()
    assert not build_case.build(python_extension=True, plan_only=True)["cached"]
    assert build_case.build(python_extension=True).compiled
    assert build_case.load().path == first.path


@pytest.mark.parametrize("exported,declared", [(True, False), (False, True)])
def test_python_export_declaration_mismatch_is_rejected(monkeypatch, exported, declared):
    monkeypatch.setattr(artifacts.subprocess, "run", lambda *a, **kw: SimpleNamespace(
        returncode=0, stdout="000001 T PyInit_probe\n" if exported else "000001 T c_entry\n", stderr=""))
    with pytest.raises(ValueError, match="python_extension"):
        artifacts._validate_python_exports(Path("probe.so"), declared)
