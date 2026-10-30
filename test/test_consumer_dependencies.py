"""Compiler dependency capture and lookup selection without a CUDA runtime."""
import json
import os
from pathlib import Path
import re
from types import SimpleNamespace

import pytest

from grid_rbd import _artifacts as artifacts
from test._consumer_model_fixture import generate_metadata_header
from grid_rbd._consumer_dependencies import parse_dependencies, discover_dependencies


def test_make_dependency_escaping(tmp_path):
    paths = [tmp_path / name for name in ["space name.h", "hash#name.h", "dollar$name.h"]]
    encoded = [str(p).replace(" ", r"\ ") for p in paths]
    output = "grid_dependency : " + " \\\n ".join(encoded) + "\n"
    assert parse_dependencies(output) == sorted(paths)


@pytest.mark.parametrize("output", ["other: a", "grid_dependency:",
    "grid_dependency: a\nother: b", "grid_dependency: a\\"])
def test_invalid_make_dependencies_fail_closed(output):
    with pytest.raises(RuntimeError):
        parse_dependencies(output)


def test_dependency_parser_keeps_selected_symlink_spelling(tmp_path):
    target = tmp_path / "target.h"
    alias = tmp_path / "alias.h"
    target.write_text("// header\n")
    alias.symlink_to(target)
    assert parse_dependencies(f"grid_dependency: {alias}\n") == [alias]


@pytest.mark.parametrize("flag", ["-E", "-c", "--ptx", "-M", "-MF=dep", "-MTother", "--output-file=x"])
def test_phase_overrides_rejected_before_preprocessing(tmp_path, monkeypatch, flag):
    monkeypatch.setattr(artifacts.subprocess, "run", lambda *a, **k: pytest.fail("must not invoke nvcc"))
    with pytest.raises(ValueError, match="phase/output"):
        discover_dependencies(tmp_path, [tmp_path / "unit.cu"], ["nvcc", flag])


@pytest.fixture
def case(tmp_path, monkeypatch):
    urdf = tmp_path / "robot.urdf"
    urdf.write_text('<robot name="probe"><link name="base"/></robot>')
    early, late = tmp_path / "early", tmp_path / "late"
    early.mkdir(); late.mkdir()
    header = late / "config.h"
    header.write_text("#define VALUE 1\n")
    source = tmp_path / "kernel.cu"
    source.write_text('#include "grid.cuh"\n#include "config.h"\n')
    state = SimpleNamespace(compiles=0, scans=0, early=early, late=late,
                            header=header, source=source, mutate=None, fail_scan=False)
    monkeypatch.setattr(artifacts, "artifact_identity", lambda arch: {
        "key_schema": artifacts._KEY_SCHEMA, "artifact_schema": artifacts.ARTIFACT_SCHEMA,
        "cuda_arch": arch})
    monkeypatch.setattr(artifacts, "resolve_consumer_toolchain", lambda *a: None)
    monkeypatch.setattr(artifacts, "_toolchain_tag", lambda: "fake")
    monkeypatch.setattr(artifacts, "_build_jobs", lambda: 1)
    monkeypatch.setattr(artifacts, "_validate_python_exports", lambda *a: None)
    monkeypatch.setattr(artifacts, "generate_grid_cuh", generate_metadata_header)
    monkeypatch.setattr(artifacts, "_compile_command", lambda gen, arch, incs, *a:
                        ["fake-nvcc", f"-I{gen}", *[f"-I{p}" for p in incs]])

    def dependencies(unit, includes, found):
        unit = unit.resolve()
        if unit in found:
            return
        found.append(unit)
        for name in re.findall(r'#include "([^"]+)"', unit.read_text()):
            chosen = next((p / name for p in [unit.parent, *includes] if (p / name).is_file()), None)
            if chosen is None:
                raise FileNotFoundError(name)
            dependencies(chosen, includes, found)

    def nvcc(cmd, **kwargs):
        if "-MM" in cmd:
            state.scans += 1
            if state.fail_scan:
                return SimpleNamespace(returncode=1, stdout="", stderr="preprocessor failed")
            includes = [Path(a[2:]) for a in cmd if a.startswith("-I")]
            found = []
            try:
                dependencies(Path(cmd[-1]), includes, found)
            except FileNotFoundError as exc:
                return SimpleNamespace(returncode=1, stdout="", stderr=str(exc))
            escaped = [str(p).replace(" ", "\\ ") for p in found]
            return SimpleNamespace(returncode=0, stdout="grid_dependency: " + " ".join(escaped) + "\n", stderr="")
        out = Path(cmd[cmd.index("-o") + 1])
        if "-c" in cmd:
            state.compiles += 1
            if state.mutate:
                state.mutate()
        out.write_bytes(b"compiled")
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(artifacts.subprocess, "run", nvcc)
    state.build = lambda **kw: artifacts.build_artifact(
        "deps", "robot", urdf, sources=kw.pop("sources", [source]),
        include_dirs=[early, late], cache_dir=tmp_path / "cache", cuda_arch=120, **kw)
    state.load = lambda: artifacts.load_artifact("deps", "robot", tmp_path / "cache", cuda_arch=120)
    return state


@pytest.mark.parametrize("change", ["direct", "transitive", "shadow", "delete_selected"])
def test_dependency_selection_and_bytes_rekey(case, change):
    if change == "transitive":
        case.header.write_text('#include "detail.h"\n')
        (case.late / "detail.h").write_text("one")
    if change == "delete_selected":
        (case.early / "config.h").write_text("early")
    first = case.build()
    if change == "direct":
        case.header.write_text("two")
    elif change == "transitive":
        (case.late / "detail.h").write_text("two")
    elif change == "shadow":
        (case.early / "config.h").write_text("new earlier include")
    else:
        (case.early / "config.h").unlink()
    assert not case.build(plan_only=True)["cached"]
    second = case.build()
    assert second.content_key != first.content_key
    assert second.compiled and case.compiles == 2


def test_touch_hits_and_generated_staging_paths_do_not_rekey(case):
    first = case.build()
    os.utime(case.header, None)
    assert case.build(plan_only=True)["cached"]
    assert not case.build().compiled
    # A fresh generation directory must not change content identity.
    second = case.build(force_rebuild=True)
    assert first.content_key == second.content_key


def test_unrelated_unit_keeps_object(case):
    other = case.source.parent / "other.cu"
    other.write_text('// no config dependency\n')
    first = case.build(sources=[case.source, other])
    case.header.write_text("two")
    second = case.build(sources=[case.source, other])
    assert first.content_key != second.content_key
    assert second.meta["build"]["units"][case.source.name]["compiled"]
    assert not second.meta["build"]["units"][other.name]["compiled"]


def test_dependency_preprocessing_scales_linearly_with_units(case):
    sources = [case.source]
    for i in range(3):
        source = case.source.parent / f"other{i}.cu"
        source.write_text("// independent unit\n")
        sources.append(source)
    case.build(sources=sources)
    # Initial, pre-key, pre/post unit, post-build and pre-publication scans.
    assert case.scans == 6 * len(sources)


def test_discovery_failure_does_not_return_old_hit(case):
    case.build()
    case.fail_scan = True
    with pytest.raises(RuntimeError, match="dependenc"):
        case.build()


def test_mutated_discovered_header_is_rejected_before_object_publication(case):
    case.mutate = lambda: case.header.write_text("changed during compile")
    with pytest.raises(RuntimeError, match="changed"):
        case.build()
    assert not list((case.source.parent / "cache").rglob("*.o"))


def test_prepared_load_does_not_discover_or_read_source(case):
    first = case.build()
    case.header.unlink(); case.source.unlink()
    case.fail_scan = True
    scans = case.scans
    assert case.load().content_key == first.content_key
    assert case.scans == scans


def test_dependency_manifest_records_paths_and_hashes(case):
    first = case.build()
    record = json.loads((first.store_dir / "meta.json").read_text())["dependencies"]
    assert record["schema"] == 1
    deps = record["units"][str(case.source)]
    assert any(row["path"] == str(case.header) and len(row["sha256"]) == 64 for row in deps)
    assert any(row["path"] == "@generated/grid.cuh" for row in deps)


@pytest.mark.parametrize("damage", [None, {"schema": 999}])
def test_dependency_metadata_damage_is_repaired(case, damage):
    first = case.build()
    path = first.store_dir / "meta.json"
    meta = json.loads(path.read_text())
    if damage is None:
        del meta["dependencies"]
    else:
        meta["dependencies"] = damage
    path.write_text(json.dumps(meta))
    assert not case.build(plan_only=True)["cached"]
    repaired = case.build()
    assert repaired.content_key == first.content_key
    assert repaired.meta["dependencies"]["schema"] == 1
    assert case.build(plan_only=True)["cached"]
