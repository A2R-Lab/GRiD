"""CPU-only gates for the split driver's granular cuda partition (no GPU, no
nvcc): atom-key parsing, bin-pack invariants (completeness, budget, unsplit
atoms), the --resume ledger state machine, and one live-collection integration
gate that proves the real partition is set-equal to the real collection."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_split_suite as rss  # noqa: E402

FLAGSHIP = "test_cuda_executable_equivalence"
_P = "test/cuda_equivalents"


def _fid(robot, base, cell, threads):
    return (f"{_P}/{FLAGSHIP}.py::test_x[{robot}-{base}-{cell}-threads{threads}]")


def test_atom_key_shapes():
    # flagship: (module, robot, base, cell) — thread token excluded
    assert rss._atom_key(_fid("iiwa14", "fixed", "crba", 96)) == (
        FLAGSHIP, "iiwa14", "fixed", "crba")
    assert rss._atom_key(_fid("h1_2", "floating", "end_effector_pose_hessian",
                              "suggested")) == (
        FLAGSHIP, "h1_2", "floating", "end_effector_pose_hessian")
    # second_order_fallback: (module, robot, base)
    nid = f"{_P}/test_cuda_second_order_fallback.py::test_y[g1-floating-extra]"
    assert rss._atom_key(nid) == ("test_cuda_second_order_fallback", "g1",
                                  "floating")
    # unlisted module: whole-module atom even when parameterized — including
    # the codegen_layout ids that embed raw CUDA source (never param-parsed)
    raw = (f"{_P}/test_cuda_codegen_layout.py::test_z"
           '[__global__ void k() { T *p; }\\n#define X "a-b" ]')
    assert rss._atom_key(raw) == ("test_cuda_codegen_layout",)
    # unparameterized test in a listed module: falls back to module atom
    bare = f"{_P}/{FLAGSHIP}.py::test_unparameterized"
    assert rss._atom_key(bare) == (FLAGSHIP,)


def _synthetic_ids():
    ids = []
    for robot in ("iiwa14", "g1"):
        for base in ("fixed", "floating"):
            for cell in ("inverse_dynamics", "crba"):
                for th in (32, 96, 100, "suggested"):
                    ids.append(_fid(robot, base, cell, th))
    ids += [f"{_P}/test_cuda_spherical_integrator_gradient_equivalence.py"
            f"::test_g[q{i}]" for i in range(6)]
    ids += [f"{_P}/test_cuda_dccrba.py::test_d[{r}]"
            for r in ("iiwa14", "go2", "g1")]
    ids.append(f"{_P}/test_cuda_codegen_layout.py::test_z"
               '[__global__ void k() { T *p; }\\n ]')
    return ids


def test_binpack_completeness_budget_and_unsplit_atoms():
    ids = _synthetic_ids()
    budget = 20 * 60.0  # small budget to force many shards
    shards = rss.pack_cuda_shards(ids, {}, budget)

    # completeness: exact set-equality, no duplicates
    packed = [i for s in shards for i in s.targets]
    assert sorted(packed) == sorted(ids)
    assert len(packed) == len(set(packed))

    by_id = {i: s.name for s in shards for i in s.targets}
    # flagship thread quads never split across shards (they share one compile)
    for robot in ("iiwa14", "g1"):
        for base in ("fixed", "floating"):
            for cell in ("inverse_dynamics", "crba"):
                homes = {by_id[_fid(robot, base, cell, th)]
                         for th in (32, 96, 100, "suggested")}
                assert len(homes) == 1, (robot, base, cell, homes)
    # the only module-scoped-fixture module stays whole
    spherical = {n for i, n in by_id.items()
                 if "spherical_integrator_gradient" in i}
    assert len(spherical) == 1
    # budget respected except single-atom shards (which may legally exceed it)
    atom_names = {}
    for i in ids:
        atom_names.setdefault(rss._atom_key(i), set()).add(by_id[i])
    for s in shards:
        n_atoms = sum(1 for homes in atom_names.values() if s.name in homes)
        if s.est_secs > budget:
            assert n_atoms == 1, f"{s.name} over budget with {n_atoms} atoms"


def test_binpack_uses_measured_durations():
    ids = _synthetic_ids()
    fast = {i: 1.0 for i in ids}  # everything measured trivially fast
    shards = rss.pack_cuda_shards(ids, fast, 120 * 60.0)
    assert len(shards) == 1  # collapses to one shard once calibrated
    assert sorted(shards[0].targets) == sorted(ids)


def test_ledger_roundtrip_and_resume_filter(tmp_path):
    rows = [
        dict(shard="cuda_00_a", domain="cuda", kind="OK", rc=0),
        dict(shard="cuda_01_b", domain="cuda", kind="FAILURES", rc=1),
        dict(shard="cuda_02_c", domain="cuda", kind="CASUALTY", rc="STALL"),
        dict(shard="test_tool", domain="wrappers", kind="FLOOR-SKIP", rc=1),
        dict(shard="gone_shard", domain="cuda", kind="OK", rc=0),
    ]
    rss._write_ledger(tmp_path, rows)
    assert json.loads((tmp_path / "results.json").read_text()) == rows

    names = {"cuda_00_a", "cuda_01_b", "cuda_02_c", "test_tool"}
    prior = rss.load_prior_results(tmp_path, names)
    kept = {r["shard"] for r in prior}
    # clean rows for live shards kept; failures/casualties/vanished re-run
    assert kept == {"cuda_00_a", "test_tool"}
    assert all(r["fresh"] is False for r in prior)
    # corrupt ledger → resume from scratch, never a crash
    (tmp_path / "results.json").write_text("{not json")
    assert rss.load_prior_results(tmp_path, names) == []


def test_partition_persistence_roundtrip(tmp_path):
    shards = rss.pack_cuda_shards(_synthetic_ids(), {}, 20 * 60.0)
    rss.save_partition(tmp_path, shards)
    loaded = rss.load_partition(tmp_path)
    assert loaded == shards
    assert rss.load_partition(tmp_path / "nope") is None


def test_live_collection_partition_is_complete():
    """Integration: the REAL collection partitions completely (the builder
    asserts set-equality internally) and node-id targets survive as argv
    elements (raw-CUDA-source ids included, by construction of the id list)."""
    shards, ids = rss.build_cuda_shards(None, 120 * 60.0)
    assert len(ids) > 900, f"suspiciously small collection: {len(ids)}"
    assert all(i.startswith("test/cuda_equivalents/") and "::" in i for i in ids)
    assert sum(len(s.targets) for s in shards) == len(ids)
    assert all(s.apply_marker and s.domain == "cuda" for s in shards)
    for s in shards:
        assert s.fingerprint_paths, s.name
        assert all("," not in p for p in s.fingerprint_paths), s.name


# ─── RAM-aware compile pool gates (2026-08-18) ───────────────────────────────
import compile_sched  # noqa: E402
import prewarm_cuda_flagship as pw  # noqa: E402


def test_prewarm_atom_parse():
    """Node-id -> (robot, base, cell) atoms; non-flagship ids ignored; the
    flagship test-name prefix pair maps both bases; the threads token drops."""
    ids = [
        _fid("iiwa14", "fixed", "crba", 32).replace(
            "test_x", "test_fixed_base_generated_cuda_matches_python_reference"),
        _fid("go2", "floating", "end_effector_pose_hessian", "suggested").replace(
            "test_x", "test_floating_base_generated_cuda_matches_python_reference"),
        f"{_P}/test_cuda_dccrba.py::test_cuda_dccrba_matches_reference[go2-floating]",
    ]
    atoms = pw._parse_atoms(ids)
    assert atoms == {
        ("iiwa14", "fixed", "crba"),
        ("go2", "floating", "end_effector_pose_hessian"),
    }


def test_scheduler_admission_and_groups(tmp_path):
    """max_jobs + group serialization + ledger roundtrip on dummy commands."""
    ledger = compile_sched.Ledger(tmp_path / "rss.json")
    ledger.record("k1", 123456)
    # reload roundtrip + max-over-history (smaller peak never lowers)
    ledger2 = compile_sched.Ledger(tmp_path / "rss.json")
    assert ledger2.data == {"k1": 123456}
    ledger2.record("k1", 5)
    assert ledger2.data["k1"] == 123456
    assert ledger2.predict_kb("k1", 999, 2.0) == 246912
    assert ledger2.predict_kb("unknown", 1000, 1.5) == 1500

    marker = tmp_path / "serial_marker"
    script = (
        "import sys, time, pathlib\n"
        "m = pathlib.Path(sys.argv[1])\n"
        "assert not m.exists(), 'group serialization violated'\n"
        "m.touch(); time.sleep(0.4); m.unlink()\n"
    )
    jobs = []
    for i in range(3):
        jobs.append(compile_sched.Job(
            name=f"g{i}",
            argv=[sys.executable, "-c", script, str(marker)],
            ledger_key=f"g{i}",
            group="serial-group",
            log_path=str(tmp_path / f"g{i}.log")))
    jobs.append(compile_sched.Job(
        name="free", argv=[sys.executable, "-c", "print('ok')"],
        ledger_key="free", log_path=str(tmp_path / "free.log")))
    sched = compile_sched.RamScheduler(
        ledger, max_jobs=4, floor_kb=0, default_peak_kb=1, label="test")
    results = sched.run(jobs)
    assert {r.rc for r in results.values()} == {0}, {
        n: (r.rc, r.stdout_tail) for n, r in results.items()}
    # peaks parsed from /usr/bin/time for every job
    assert all(r.peak_kb > 0 for r in results.values())


def test_scheduler_stop_resolves_pending(tmp_path):
    """stop() drains: unlaunched jobs resolve rc=-1 so waiters never hang."""
    ledger = compile_sched.Ledger(tmp_path / "rss.json")
    sched = compile_sched.RamScheduler(
        ledger, max_jobs=1, floor_kb=0, default_peak_kb=1, label="test")
    jobs = [
        compile_sched.Job(name="slow",
                          argv=[sys.executable, "-c", "import time; time.sleep(1.5)"],
                          ledger_key="slow", log_path=str(tmp_path / "slow.log")),
        compile_sched.Job(name="never",
                          argv=[sys.executable, "-c", "print('no')"],
                          ledger_key="never", log_path=str(tmp_path / "never.log")),
    ]
    t = sched.run_async(jobs)
    import time as _t
    _t.sleep(0.3)
    sched.stop()
    t.join(timeout=10)
    assert not t.is_alive()
    assert sched.results["slow"].rc == 0
    assert sched.results["never"].rc == -1


def test_prewarm_plan_iiwa14_live():
    """Integration (CPU, no nvcc): the plan for iiwa14 flagship ids builds via
    the test module's own machinery — selection filtering applies, atoms map to
    jobs, and job names are filesystem-safe."""
    ids = []
    for base in ("fixed", "floating"):
        tname = (f"test_{base}_base_generated_cuda_matches_python_reference")
        for cell in ("crba", "end_effector_pose_hessian"):
            ids.append(f"{_P}/{FLAGSHIP}.py::{tname}[iiwa14-{base}-{cell}-threads32]")
    plan = pw.build_plan(ids)
    assert plan["dropped"] == []
    names = set(plan["jobs"])
    # fixed crba + fixed ee-hessian always in selection; floating ee-hessian is
    # NOT in the default floating flagship selection (benign always-skip) so it
    # must NOT be pre-warmed; floating crba is.
    assert "prewarm_iiwa14_fixed_crba" in names
    assert "prewarm_iiwa14_fixed_end_effector_pose_hessian" in names
    assert "prewarm_iiwa14_floating_crba" in names
    assert "prewarm_iiwa14_floating_end_effector_pose_hessian" not in names
    for name, job in plan["jobs"].items():
        assert name == name.strip() and "/" not in name and " " not in name
        assert job["cells"]
    assert all(v in names for v in plan["atom_to_job"].values())
