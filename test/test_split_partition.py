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


# ─── shard-level refresh gates (2026-08-20, user-ratified carry policy) ──────


def _mk_shard(name, ids, paths, digest):
    return {"name": name, "node_ids": ids,
            "fingerprint": {"included_paths": paths, "digest": digest}}


def test_plan_refresh_stale_detection_and_repack(monkeypatch):
    import codegen_neutrality
    monkeypatch.setattr(codegen_neutrality, "cuda_carry_soundness",
                        lambda r: (True, "stubbed neutral"))
    # wrapper shards fingerprint the bindings source too (2026-09-09 schema);
    # the carried/stale wrapper fixtures record the NEW-style path lists.
    _W1 = ["test/python_wrappers/test_w1.py", "bindings/grid_rbd", "bindings/src"]
    _W2 = ["test/python_wrappers/test_w2.py", "bindings/grid_rbd", "bindings/src"]
    now = {("a.py",): "d1", ("b.py",): "d2", tuple(_W1): "w1", tuple(_W2): "NEW"}
    clean_ids = [_fid("iiwa14", "fixed", "crba", t) for t in (1, 32)]
    stale_ids = [_fid("go2", "floating", "aba", t) for t in (1, 32)]
    old = {"shards": [
        _mk_shard("cuda_00_clean", clean_ids, ["a.py"], "d1"),
        _mk_shard("cuda_01_stale", stale_ids, ["b.py"], "CHANGED"),
        _mk_shard("test_w1", ["test/python_wrappers/test_w1.py::t"], _W1, "w1"),
        _mk_shard("test_w2", ["test/python_wrappers/test_w2.py::t"], _W2, "OLD"),
    ]}
    fn = lambda paths: now[tuple(paths)]
    current_cuda = clean_ids + stale_ids  # stale shard's tests still exist
    stale, carried, wrap_run, cuda_fresh = rss.plan_refresh(
        old, current_cuda, ["test_w1", "test_w2", "test_w3_new"], {}, 7200.0, fn)
    assert set(stale) == {"cuda_01_stale", "test_w2"}
    assert set(carried) == {"cuda_00_clean", "test_w1"}
    # stale wrapper re-runs; NEW module runs; clean wrapper carried
    assert set(wrap_run) == {"test_w2", "test_w3_new"}
    # fresh cuda covers exactly the non-carried ids
    fresh_ids = sorted(i for s in cuda_fresh for i in s.targets)
    assert fresh_ids == sorted(stale_ids)
    # NAME RECYCLING: every stale cuda name is shadowed by a fresh shard
    # (carry_forward has no "superseded" state — an unshadowed stale name
    # would be refused at merge time; the 2026-08-21 first-run lesson)
    assert {s.name for s in cuda_fresh} == {"cuda_01_stale"}


def test_plan_refresh_edge_cases(monkeypatch):
    import codegen_neutrality
    monkeypatch.setattr(codegen_neutrality, "cuda_carry_soundness",
                        lambda r: (True, "stubbed neutral"))
    ids = [_fid("iiwa14", "fixed", "crba", 1)]
    fn = {"a.py": "d1", "b.py": "CHANGED"}.__getitem__
    fn1 = lambda paths: fn(paths[0])
    # stale cuda shard with ZERO fresh atoms to fill its name -> refuse
    old = {"shards": [_mk_shard("cuda_00_crba", ids, ["a.py"], "d1"),
                      _mk_shard("cuda_00_misc", ["x::y"], ["b.py"], "d2")]}
    with pytest.raises(RuntimeError, match="cannot shadow"):
        rss.plan_refresh(old, ids, [], {}, 7200.0, fn1)
    # CARRIED shard's id missing from collection -> loud refusal
    with pytest.raises(RuntimeError, match="no longer collectable"):
        rss.plan_refresh(old, [], [], {}, 7200.0, fn1)
    # stale WRAPPER shard whose module was deleted -> refuse (no namesake)
    old_w = {"shards": [_mk_shard("test_gone",
                                  ["test/python_wrappers/test_gone.py::t"],
                                  ["b.py"], "OLD")]}
    with pytest.raises(RuntimeError, match="deleted module|no current module"):
        rss.plan_refresh(old_w, [], ["test_other"], {}, 7200.0, fn1)
    # multiple stale cuda names: fresh partition recycles ALL of them, exactly
    stale_a = [_fid("go2", "floating", "aba", t) for t in (1, 32)]
    stale_b = [_fid("g1", "fixed", "minv", t) for t in (1, 32)]
    old3 = {"shards": [
        _mk_shard("cuda_05_one", stale_a, ["b.py"], "OLD"),
        _mk_shard("cuda_09_two", stale_b, ["b.py"], "OLD"),
    ]}
    _, carried3, _, fresh3 = rss.plan_refresh(
        old3, stale_a + stale_b, [], {}, 7200.0, fn1)
    assert carried3 == []
    assert {s.name for s in fresh3} == {"cuda_05_one", "cuda_09_two"}
    assert sorted(i for s in fresh3 for i in s.targets) == sorted(stale_a + stale_b)
    assert all(s.targets for s in fresh3)


def test_plan_refresh_refuses_monolithic_receipt():
    with pytest.raises(RuntimeError, match="shards"):
        rss.plan_refresh({"tests": []}, [], [], {}, 7200.0, lambda p: "")


def test_plan_refresh_bindings_fingerprint_schema_upgrade():
    """A wrapper shard recorded under the OLD test-file-only fingerprint paths
    is stale BY DEFINITION (2026-09-09: wrapper fingerprints must include the
    bindings source, so a bindings-only change stales the tests importing it).
    Its digest is never even recomputed — one re-run installs the new paths."""
    old_paths = ["test/python_wrappers/test_w_old.py"]
    old = {"shards": [
        _mk_shard("test_w_old", ["test/python_wrappers/test_w_old.py::t"],
                  old_paths, "SAME"),
    ]}
    # digest fn says "unchanged" — the schema rule must stale it anyway
    stale, carried, wrap_run, cuda_fresh = rss.plan_refresh(
        old, [], ["test_w_old"], {}, 7200.0, lambda paths: "SAME")
    assert set(stale) == {"test_w_old"} and not carried
    assert set(wrap_run) == {"test_w_old"} and not cuda_fresh


def test_plan_refresh_cuda_carry_gate(monkeypatch):
    """2026-09-11 byte-neutrality gate: fingerprint-clean cuda shards are
    carried ONLY when codegen_neutrality declares the carry sound; otherwise
    the whole cuda domain is demoted to stale (and the fresh partition
    shadows every demoted name). Wrapper carries are never affected."""
    import codegen_neutrality
    ids = [_fid("iiwa14", "fixed", "crba", 1), _fid("go2", "fixed", "crba", 1)]
    _W = ["test/python_wrappers/test_w.py", "bindings/grid_rbd", "bindings/src"]
    old = {"repo": {"commit_sha": "deadbeef", "dirty": False}, "shards": [
        _mk_shard("cuda_00_x", [ids[0]], ["test/cuda_equivalents/a.py"], "SAME"),
        _mk_shard("cuda_01_y", [ids[1]], ["test/cuda_equivalents/b.py"], "SAME"),
        _mk_shard("test_w", ["test/python_wrappers/test_w.py::t"], _W, "SAME"),
    ]}
    same = lambda paths: "SAME"

    monkeypatch.setattr(codegen_neutrality, "cuda_carry_soundness",
                        lambda r: (True, "proven"))
    stale, carried, wrap_run, cuda_fresh = rss.plan_refresh(
        old, ids, ["test_w"], {}, 7200.0, same)
    assert set(carried) == {"cuda_00_x", "cuda_01_y", "test_w"} and not stale

    monkeypatch.setattr(codegen_neutrality, "cuda_carry_soundness",
                        lambda r: (False, "not neutral"))
    stale, carried, wrap_run, cuda_fresh = rss.plan_refresh(
        old, ids, ["test_w"], {}, 7200.0, same)
    assert set(stale) == {"cuda_00_x", "cuda_01_y"}
    assert carried == ["test_w"]          # wrapper carry untouched
    assert {s.name for s in cuda_fresh} == {"cuda_00_x", "cuda_01_y"}
    assert sorted(i for s in cuda_fresh for i in s.targets) == sorted(ids)


def test_cuda_carry_soundness_paths(monkeypatch):
    """Unit-level decision table for codegen_neutrality.cuda_carry_soundness:
    unchanged inputs -> carry; changed+proven -> carry; changed+dirty old
    receipt -> refuse; changed+not-neutral -> refuse; assume-neutral env ->
    carry without proof."""
    import codegen_neutrality as cn
    clean = {"repo": {"commit_sha": "abc", "dirty": False}}

    monkeypatch.setattr(cn, "codegen_inputs_changed", lambda sha: False)
    ok, why = cn.cuda_carry_soundness(clean)
    assert ok and "unchanged" in why

    monkeypatch.setattr(cn, "codegen_inputs_changed", lambda sha: True)
    monkeypatch.setattr(cn, "prove_byte_neutrality",
                        lambda sha: (True, "6 matrix rows byte-identical"))
    ok, why = cn.cuda_carry_soundness(clean)
    assert ok and "PROVEN" in why

    monkeypatch.setattr(cn, "prove_byte_neutrality",
                        lambda sha: (False, "matrix rows differ: x"))
    ok, why = cn.cuda_carry_soundness(clean)
    assert not ok and "NOT byte-neutral" in why

    ok, why = cn.cuda_carry_soundness(
        {"repo": {"commit_sha": "abc", "dirty": True}})
    assert not ok and "DIRTY" in why

    assert not cn.cuda_carry_soundness({})[0]

    monkeypatch.setenv("GRID_REFRESH_ASSUME_NEUTRAL", "1")
    ok, why = cn.cuda_carry_soundness(clean)
    assert ok and "WITHOUT proof" in why


def test_resume_prunes_sha_divergent_clean_rows(tmp_path, monkeypatch):
    """2026-09-11: a clean resume row whose per-shard receipt was recorded at
    a different commit than HEAD re-runs (mixed-sha receipts cannot merge);
    same-sha and receipt-less rows are kept as before."""
    import json as _json
    import subprocess as _sp
    head = _sp.run(["git", "rev-parse", "HEAD"], cwd=rss.REPO_ROOT,
                   capture_output=True, text=True, check=True).stdout.strip()
    (tmp_path / "receipts").mkdir()
    rows = [
        {"shard": "s_same", "kind": "OK"},
        {"shard": "s_divergent", "kind": "OK"},
        {"shard": "s_noreceipt", "kind": "OK"},
        {"shard": "s_failed", "kind": "FAILURES"},
    ]
    (tmp_path / "results.json").write_text(_json.dumps(rows))
    (tmp_path / "receipts" / "s_same.json").write_text(
        _json.dumps({"repo": {"commit_sha": head}}))
    (tmp_path / "receipts" / "s_divergent.json").write_text(
        _json.dumps({"repo": {"commit_sha": "0" * 40}}))
    kept = rss.load_prior_results(
        tmp_path, {"s_same", "s_divergent", "s_noreceipt", "s_failed"})
    assert {r["shard"] for r in kept} == {"s_same", "s_noreceipt"}


def test_plan_refresh_never_carries_recorded_failures(monkeypatch):
    """2026-09-11: a fingerprint-clean shard whose recorded results contain a
    non-passing test is stale anyway — carrying it would keep attesting the
    old failure after a codegen-side fix that never touched its test files
    (the cuda_04 MT case, live)."""
    import codegen_neutrality
    monkeypatch.setattr(codegen_neutrality, "cuda_carry_soundness",
                        lambda r: (True, "stubbed neutral"))
    ids = [_fid("iiwa14", "fixed", "crba", 1), _fid("go2", "fixed", "crba", 1)]
    old = {"repo": {"commit_sha": "abc", "dirty": False},
           "tests": [{"node_id": ids[0], "outcome": "failed"},
                     {"node_id": ids[1], "outcome": "passed"}],
           "shards": [
        _mk_shard("cuda_00_bad", [ids[0]], ["test/cuda_equivalents/a.py"], "SAME"),
        _mk_shard("cuda_01_good", [ids[1]], ["test/cuda_equivalents/b.py"], "SAME"),
    ]}
    stale, carried, wrap_run, cuda_fresh = rss.plan_refresh(
        old, ids, [], {}, 7200.0, lambda paths: "SAME")
    assert "cuda_00_bad" in stale and carried == ["cuda_01_good"]
    assert {s.name for s in cuda_fresh} == {"cuda_00_bad"}
    assert sorted(i for s in cuda_fresh for i in s.targets) == [ids[0]]
