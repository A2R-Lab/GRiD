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


@pytest.fixture(autouse=True)
def unchanged_oracle_for_synthetic_receipts(monkeypatch):
    # Synthetic receipts below use fictitious commit SHAs; isolate their
    # partition/codegen cases from the independently tested oracle-pin guard.
    import codegen_neutrality
    monkeypatch.setattr(codegen_neutrality, "reference_inputs_changed", lambda sha: False)


def test_oracle_pin_change_demotes_cuda_and_wrapper_even_with_identical_headers(monkeypatch):
    import codegen_neutrality
    monkeypatch.setattr(codegen_neutrality, "reference_inputs_changed", lambda sha: True)
    monkeypatch.setenv("GRID_REFRESH_ASSUME_NEUTRAL", "1")
    ids = [_fid("iiwa14", "fixed", "end_effector_pose", 32)]
    old = {"repo": {"commit_sha": "old"}, "shards": [
        _mk_shard("cuda_00_pose", ids, ["test.py"], "same"),
        _mk_shard("test_pose", ["test/python_wrappers/test_pose.py::t"],
                  ["bindings/grid_rbd"], "same"),
    ]}
    stale, carried, wrappers, fresh = rss.plan_refresh(
        old, ids, ["test_pose"], {}, 7200, _plan(old, lambda paths: "same"))
    assert set(stale) == {"cuda_00_pose", "test_pose"}
    assert not carried and wrappers == ["test_pose"]
    assert [n for s in fresh for n in s.targets] == ids
    assert "external/RBDReference" in fresh[0].fingerprint_paths


def _fid(robot, base, cell, threads):
    return (f"{_P}/{FLAGSHIP}.py::test_x[{robot}-{base}-{cell}-threads{threads}]")


def _dispatch_spec(name):
    return rss.ShardSpec(name, "cuda", [f"test/{name}.py::test_one"],
                         [f"test/{name}.py"], apply_marker=True)


def test_cuda_prewarm_and_test_environment_share_defaults(monkeypatch):
    monkeypatch.delenv("GRID_CUDA_CACHE_DIR", raising=False)
    monkeypatch.delenv("GRID_ENABLE_MUJOCO_KERNELS", raising=False)
    assert rss.cuda_worker_env() == {
        "GRID_CUDA_CACHE_DIR": rss.CUDA_CACHE_DIR_DEFAULT,
        "GRID_ENABLE_MUJOCO_KERNELS": "0",
    }
    monkeypatch.setenv("GRID_CUDA_CACHE_DIR", "/tmp/explicit-cache")
    monkeypatch.setenv("GRID_ENABLE_MUJOCO_KERNELS", "1")
    assert rss.cuda_worker_env() == {
        "GRID_CUDA_CACHE_DIR": "/tmp/explicit-cache",
        "GRID_ENABLE_MUJOCO_KERNELS": "1",
    }


def test_compile_priority_follows_first_consumer():
    from types import SimpleNamespace
    jobs = [SimpleNamespace(name=n) for n in ("h1", "iiwa", "shared", "unused")]
    ordered = rss.prioritize_compile_jobs(jobs, ["first", "second"],
        {"first": ["iiwa", "shared"], "second": ["h1", "shared"]})
    assert [j.name for j in ordered] == ["iiwa", "shared", "h1", "unused"]
    assert {id(j) for j in ordered} == {id(j) for j in jobs}


def test_compile_priority_heavy_first_within_a_consumer(tmp_path):
    """R3 (2026-10-10): within one consumer rank the longest recorded job
    launches first; unrecorded jobs rank at DEFAULT_SECS (after a known-heavy
    job, before a known-light one); consumer order still wins across ranks."""
    from types import SimpleNamespace
    import compile_sched
    ledger = compile_sched.Ledger(tmp_path / "rss.json")
    ledger.record("prewarm_light", 1000, 30.0)
    ledger.record("prewarm_heavy", 1000, 16000.0)
    ledger.record("prewarm_second_rank", 1000, 99999.0)
    jobs = [SimpleNamespace(name=n, ledger_key=n) for n in
            ("prewarm_light", "prewarm_unknown", "prewarm_heavy", "prewarm_second_rank")]
    ordered = rss.prioritize_compile_jobs(jobs, ["s1", "s2"],
        {"s1": ["prewarm_light", "prewarm_unknown", "prewarm_heavy"],
         "s2": ["prewarm_second_rank"]}, ledger)
    assert [j.name for j in ordered] == [
        "prewarm_heavy", "prewarm_unknown", "prewarm_light", "prewarm_second_rank"]


def test_ready_shard_waits_for_all_writers_and_keeps_failed_jobs():
    from threading import Event
    from types import SimpleNamespace
    a, b = Event(), Event()
    pool = SimpleNamespace(done_events={"a": a, "b": b})
    first, later = _dispatch_spec("first"), _dispatch_spec("later")
    prereqs = {"first": ["a", "b"]}
    assert rss.next_ready_shard([first, later], pool, prereqs) is later
    a.set()
    assert rss.next_ready_shard([first], pool, prereqs) is None
    b.set()  # also what a failed prewarm does: allow the real test to retry
    assert rss.next_ready_shard([first, later], pool, prereqs) is first
    assert rss.next_ready_shard([first, later], None, prereqs) is first


def test_phase_run_dispatches_ready_shards_once_without_gpu(tmp_path, monkeypatch):
    from threading import Event
    from types import SimpleNamespace
    first, second, third = map(_dispatch_spec, ("first", "second", "third"))
    a, b = Event(), Event()
    pool = SimpleNamespace(done_events={"a": a, "b": b}, raise_floor=lambda _: None)
    launched = []
    def launch(command, **kwargs):
        # Only one shard can be active; its compile writer must have finished.
        name = next(s.name for s in (first, second, third) if s.targets[0] in command)
        assert name != "first" or a.is_set()
        assert name != "second" or b.is_set()
        launched.append(name)
        return SimpleNamespace(name=name)
    def finish(proc, *args):
        if proc.name == "third":
            b.set()  # second becomes ready before first
        elif proc.name == "second":
            a.set()
        return 0
    monkeypatch.setattr(rss.subprocess, "Popen", launch)
    monkeypatch.setattr(rss.subprocess, "run", lambda *a, **k: SimpleNamespace(stdout="0\n"))
    monkeypatch.setattr(rss._ShardRun, "poll", lambda self: finish(self.proc, self.spec.name, self.log_path))
    monkeypatch.setattr(rss, "parse_junit", lambda _: (1, 0, 0, 0, []))
    results, paused = rss.phase_run([first, second, third], tmp_path, True, [], [],
                                  pool, {"first": ["a"], "second": ["b"]})
    assert not paused and launched == ["third", "second", "first"]
    assert [r["shard"] for r in results] == launched
    assert all(r['rc'] == 0 for r in results)
    assert len(json.loads((tmp_path / "results.json").read_text())) == 3
    assert [s.name for s in (first, second, third)] == ["first", "second", "third"]


def test_phase_run_pause_is_checked_while_all_shards_wait(tmp_path, monkeypatch):
    from types import SimpleNamespace
    class PausingEvent:
        def is_set(self):
            return False
        def wait(self, timeout):
            assert 0 < timeout <= 1
            (tmp_path / "PAUSE").touch()
    pool = SimpleNamespace(done_events={"a": PausingEvent()}, raise_floor=lambda _: None)
    def no_launch(*args, **kwargs):
        raise AssertionError("must not start a blocked shard")
    monkeypatch.setattr(rss.subprocess, "Popen", no_launch)
    results, paused = rss.phase_run([_dispatch_spec("first")], tmp_path, False, [], [],
                                  pool, {"first": ["a"]})
    assert paused and results == []


def test_phase_run_rescans_all_shards_after_bounded_wait(tmp_path, monkeypatch):
    from threading import Event
    from types import SimpleNamespace
    later_done = Event()
    class EarlierEvent:
        done = False
        def is_set(self):
            return self.done
        def wait(self, timeout):
            assert 0 < timeout <= 1
            later_done.set()  # later compile completes while earlier is blocked
    earlier = EarlierEvent()
    pool = SimpleNamespace(done_events={"a": earlier, "b": later_done}, raise_floor=lambda _: None)
    launched = []
    def launch(command, **kwargs):
        name = "first" if "test/first.py::test_one" in command else "second"
        assert (earlier.is_set() if name == "first" else later_done.is_set())
        launched.append(name)
        return SimpleNamespace()
    def finish(*args):
        earlier.done = True
        return 0
    monkeypatch.setattr(rss.subprocess, "Popen", launch)
    monkeypatch.setattr(rss.subprocess, "run", lambda *a, **k: SimpleNamespace(stdout="0\n"))
    monkeypatch.setattr(rss._ShardRun, "poll", lambda self: finish(self.proc, self.spec.name, self.log_path))
    monkeypatch.setattr(rss, "parse_junit", lambda _: (1, 0, 0, 0, []))
    results, paused = rss.phase_run([_dispatch_spec("first"), _dispatch_spec("second")],
        tmp_path, False, [], [], pool, {"first": ["a"], "second": ["b"]})
    assert not paused and launched == ["second", "first"]
    assert len(results) == 2


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


def test_receipt_resume_requires_success_and_explicit_clean_receipt(tmp_path):
    import subprocess
    head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=rss.REPO_ROOT,
                          capture_output=True, text=True, check=True).stdout.strip()
    cases = {
        "good": ("OK", 0, {"repo": {"commit_sha": head, "dirty": False}}),
        "floor": ("FLOOR-SKIP", 1, {"repo": {"commit_sha": head, "dirty": False}}),
        "deselected": ("DESELECTED", 0, {"repo": {"commit_sha": head, "dirty": False}}),
        "bad_rc": ("OK", 1, {"repo": {"commit_sha": head, "dirty": False}}),
        "missing_rc": ("OK", None, {"repo": {"commit_sha": head, "dirty": False}}),
        "missing": ("OK", 0, None),
        "empty_repo": ("OK", 0, {"repo": {}}),
        "unknown_dirty": ("OK", 0, {"repo": {"commit_sha": head}}),
        "dirty": ("OK", 0, {"repo": {"commit_sha": head, "dirty": True}}),
        "old_sha": ("OK", 0, {"repo": {"commit_sha": "0" * 40, "dirty": False}}),
        "bad_repo": ("OK", 0, {"repo": "invalid"}),
        "bad_document": ("OK", 0, []),
    }
    (tmp_path / "receipts").mkdir()
    rows = []
    for name, (kind, rc, receipt) in cases.items():
        rows.append(dict(shard=name, kind=kind, rc=rc))
        if receipt is not None:
            (tmp_path / "receipts" / f"{name}.json").write_text(json.dumps(receipt))
    (tmp_path / "receipts" / "corrupt.json").write_text("{invalid")
    rows.append(dict(shard="corrupt", kind="OK", rc=0))
    rss._write_ledger(tmp_path, rows)
    kept = rss.load_prior_results(tmp_path, {r["shard"] for r in rows}, require_receipts=True)
    assert [r["shard"] for r in kept] == ["good"]
    assert kept[0]["fresh"] is False


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
    ledger.record("k1", 123456, 40.0)
    # reload roundtrip + max-over-history (smaller peak / shorter run never lowers)
    ledger2 = compile_sched.Ledger(tmp_path / "rss.json")
    assert ledger2.data == {"k1": {"peak_kb": 123456, "secs": 40.0}}
    ledger2.record("k1", 5, 1.0)
    assert ledger2.data["k1"] == {"peak_kb": 123456, "secs": 40.0}
    assert ledger2.predict_kb("k1", 999, 2.0) == 246912
    assert ledger2.predict_kb("unknown", 1000, 1.5) == 1500
    assert ledger2.predict_secs("k1") == 40.0
    assert ledger2.predict_secs("unknown", 7.0) == 7.0

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
    # peaks (sampled tree or /usr/bin/time floor) for every job
    assert all(r.peak_kb > 0 for r in results.values())


def test_ledger_drops_legacy_ru_maxrss_schema(tmp_path):
    """S1 (2026-10-10): a schema-1 flat ru_maxrss map under-predicts parallel
    builds ~4x; it is dropped (counted) rather than trusted."""
    path = tmp_path / "rss.json"
    path.write_text(json.dumps({"warm:x": 4800000, "prewarm_y": 9000000}))
    ledger = compile_sched.Ledger(path)
    assert ledger.data == {} and ledger.dropped_legacy == 2
    assert ledger.predict_kb("warm:x", 1000, 1.0) == 1000
    ledger.record("warm:x", 15000000, 120.0)
    assert json.loads(path.read_text()) == {
        "schema": 2, "jobs": {"warm:x": {"peak_kb": 15000000, "secs": 120.0}}}


def test_scheduler_peak_is_the_process_tree_sum(tmp_path):
    """S1: two concurrent 200 MB grandchildren → ru_maxrss sees ~200 MB (the
    largest single descendant) but the sampled session tree sees ~400 MB."""
    child = ("import time, sys; b = bytearray(200 * 1024 * 1024); "
             "b[::4096] = b'x' * len(b[::4096]); time.sleep(2.5)")
    parent = ("import subprocess, sys; ps = [subprocess.Popen([sys.executable, '-c', %r]) "
              "for _ in range(2)]; [p.wait() for p in ps]" % child)
    ledger = compile_sched.Ledger(tmp_path / "rss.json")
    sched = compile_sched.RamScheduler(
        ledger, max_jobs=1, floor_kb=0, default_peak_kb=1, label="test")
    job = compile_sched.Job(name="tree", argv=[sys.executable, "-c", parent],
                            ledger_key="tree", log_path=str(tmp_path / "tree.log"))
    res = sched.run([job])["tree"]
    assert res.rc == 0, res.stdout_tail
    assert res.peak_kb >= 350 * 1024, res.peak_kb


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


def test_prewarm_plan_iiwa14_live(monkeypatch):
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
    # Both bases validate every cell by default since 2026-10-15, so the floating
    # ee-hessian is pre-warmed like the fixed one.
    assert "prewarm_iiwa14_fixed_crba" in names
    assert "prewarm_iiwa14_fixed_end_effector_pose_hessian" in names
    assert "prewarm_iiwa14_floating_crba" in names
    assert "prewarm_iiwa14_floating_end_effector_pose_hessian" in names
    # The narrowing override still drops the unselected floating cells.
    monkeypatch.setenv("GRID_CUDA_FLOATING_ALGORITHMS", "crba")
    narrowed = set(pw.build_plan(ids)["jobs"])
    assert "prewarm_iiwa14_floating_crba" in narrowed
    assert "prewarm_iiwa14_floating_end_effector_pose_hessian" not in narrowed
    for name, job in plan["jobs"].items():
        assert name == name.strip() and "/" not in name and " " not in name
        assert job["cells"]
    assert all(v in names for v in plan["atom_to_job"].values())


# ─── shard-level refresh gates (2026-08-20, user-ratified carry policy) ──────


def _mk_shard(name, ids, paths, digest):
    return {"name": name, "node_ids": ids,
            "fingerprint": {"included_paths": paths, "digest": digest}}



def _plan(old, digest_fn):
    """Emulate pytest-gpu-proof's `plan_carry` report for the SYNTHETIC receipts
    below (fictitious SHAs, fake digests): the receipt is carryable unless it
    was recorded dirty; a shard is carryable iff it is clean and `digest_fn`
    reproduces its recorded narrow digest. The real tool is exercised against a
    real repo in test_plan_refresh_uses_the_real_plan_carry."""
    dirty = bool((old.get("repo") or {}).get("dirty"))
    rec = {"commit_sha": (old.get("repo") or {}).get("commit_sha"), "head": "HEAD",
           "carryable": not dirty,
           "reason": "recorded with a dirty working tree" if dirty else "ok"}
    shards = []
    for sh in old.get("shards") or []:
        fp = sh.get("fingerprint") or {}
        if dirty:
            v, why = "stale", rec["reason"]
        elif sh.get("dirty"):
            v, why = "stale", "recorded with a dirty working tree"
        elif digest_fn(fp.get("included_paths") or []) == fp.get("digest"):
            v, why = "carryable", "fingerprint unchanged"
        else:
            v, why = "stale", "fingerprint changed"
        shards.append({"name": sh.get("name"), "verdict": v, "reason": why,
                       "node_count": len(sh.get("node_ids") or [])})
    return {"receipt": rec, "shards": shards}

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
        old, current_cuda, ["test_w1", "test_w2", "test_w3_new"], {}, 7200.0, _plan(old, fn))
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
        rss.plan_refresh(old, ids, [], {}, 7200.0, _plan(old, fn1))
    # CARRIED shard's id missing from collection -> loud refusal
    with pytest.raises(RuntimeError, match="no longer collectable"):
        rss.plan_refresh(old, [], [], {}, 7200.0, _plan(old, fn1))
    # stale WRAPPER shard whose module was deleted -> refuse (no namesake)
    old_w = {"shards": [_mk_shard("test_gone",
                                  ["test/python_wrappers/test_gone.py::t"],
                                  ["b.py"], "OLD")]}
    with pytest.raises(RuntimeError, match="deleted module|no current module"):
        rss.plan_refresh(old_w, [], ["test_other"], {}, 7200.0, _plan(old_w, fn1))
    # multiple stale cuda names: fresh partition recycles ALL of them, exactly
    stale_a = [_fid("go2", "floating", "aba", t) for t in (1, 32)]
    stale_b = [_fid("g1", "fixed", "minv", t) for t in (1, 32)]
    old3 = {"shards": [
        _mk_shard("cuda_05_one", stale_a, ["b.py"], "OLD"),
        _mk_shard("cuda_09_two", stale_b, ["b.py"], "OLD"),
    ]}
    _, carried3, _, fresh3 = rss.plan_refresh(
        old3, stale_a + stale_b, [], {}, 7200.0, _plan(old3, fn1))
    assert carried3 == []
    assert {s.name for s in fresh3} == {"cuda_05_one", "cuda_09_two"}
    assert sorted(i for s in fresh3 for i in s.targets) == sorted(stale_a + stale_b)
    assert all(s.targets for s in fresh3)


def test_plan_refresh_asset_change_demotes_covering_shards(monkeypatch):
    """2026-09-15 rizon4 lessons: a per-robot URDF edit is invisible to the
    covering-matrix fallback (its rows are six fixed robots), and an
    ORACLE-side asset edit (RBDReference submodule copy) never rotates the
    emitted headers — so the asset gate must demote BEFORE header-key
    replay: even a replay verdict of True must not keep a shard covering a
    changed robot. Shards of unchanged robots keep their replay/fallback
    carry. Also: asset changes ALONE (no codegen-input diff) must open the
    soundness block."""
    import codegen_neutrality
    import header_key_replay
    monkeypatch.setattr(codegen_neutrality, "codegen_inputs_changed",
                        lambda sha: False)  # assets alone open the block
    monkeypatch.setattr(codegen_neutrality, "cuda_carry_soundness",
                        lambda r: (True, "stubbed neutral"))
    monkeypatch.setattr(codegen_neutrality, "changed_robot_assets",
                        lambda sha: {"rizon4"})
    monkeypatch.setattr(header_key_replay, "shard_replay_verdict",
                        lambda recs, cache: (True, "stubbed byte-identical"))
    hit_ids = [_fid("rizon4", "fixed", "inverse_dynamics", t) for t in (1, 32)]
    safe_ids = [_fid("iiwa14", "fixed", "crba", t) for t in (1, 32)]
    old = {"repo": {"commit_sha": "0" * 40},
           "shards": [_mk_shard("cuda_00_rizon4", hit_ids, ["a.py"], "d1"),
                      _mk_shard("cuda_01_iiwa14", safe_ids, ["a.py"], "d1")]}
    stale, carried, _, fresh = rss.plan_refresh(
        old, hit_ids + safe_ids, [], {}, 7200.0, _plan(old, lambda paths: "d1"))
    assert set(stale) == {"cuda_00_rizon4"}
    assert set(carried) == {"cuda_01_iiwa14"}
    assert {s.name for s in fresh} == {"cuda_00_rizon4"}
    assert sorted(i for s in fresh for i in s.targets) == sorted(hit_ids)


def test_plan_refresh_refuses_monolithic_receipt():
    with pytest.raises(RuntimeError, match="shards"):
        rss.plan_refresh({"tests": []}, [], [], {}, 7200.0, _plan({"tests": []}, lambda p: ""))


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
        old, [], ["test_w_old"], {}, 7200.0, _plan(old, lambda paths: "SAME"))
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
        old, ids, ["test_w"], {}, 7200.0, _plan(old, same))
    assert set(carried) == {"cuda_00_x", "cuda_01_y", "test_w"} and not stale

    monkeypatch.setattr(codegen_neutrality, "cuda_carry_soundness",
                        lambda r: (False, "not neutral"))
    stale, carried, wrap_run, cuda_fresh = rss.plan_refresh(
        old, ids, ["test_w"], {}, 7200.0, _plan(old, same))
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
        old, ids, [], {}, 7200.0, _plan(old, lambda paths: "SAME"))
    assert "cuda_00_bad" in stale and carried == ["cuda_01_good"]
    assert {s.name for s in cuda_fresh} == {"cuda_00_bad"}
    assert sorted(i for s in cuda_fresh for i in s.targets) == [ids[0]]


def test_aggregate_header_keys_folds_and_carries(tmp_path, monkeypatch):
    """A4: fresh sidecars are deduped + folded; carried shards (in the ledger,
    no fresh sidecar) keep their committed rows; shards gone from the ledger
    drop out."""
    import run_split_suite as rss

    committed = tmp_path / "gpu-proof-header-keys.json"
    monkeypatch.setattr(rss, "HEADER_KEYS_PATH", committed)
    committed.write_text(json.dumps({"schema": 1, "shards": {
        "cuda_00": [{"kind": "flagship", "robot": "iiwa14",
                     "content_sha256": "old00"}],
        "cuda_01": [{"kind": "direct", "robot": "go2",
                     "content_sha256": "old01"}],
        "cuda_gone": [{"kind": "direct", "robot": "baxter",
                       "content_sha256": "oldgone"}],
    }}))
    rdir = tmp_path / "receipts"
    rdir.mkdir()
    row = json.dumps({"kind": "flagship", "robot": "iiwa14",
                      "content_sha256": "new00"}, sort_keys=True)
    (rdir / "cuda_00.header_keys.jsonl").write_text(
        row + "\n" + row + "\nnot-json\n")
    results = [{"shard": "cuda_00", "kind": "OK"},
               {"shard": "cuda_01", "kind": "OK"}]

    rss.aggregate_header_keys(rdir, results)

    data = json.loads(committed.read_text())["shards"]
    # fresh sidecar replaces (and dedupes; junk line dropped)
    assert data["cuda_00"] == [{"kind": "flagship", "robot": "iiwa14",
                                "content_sha256": "new00"}]
    # carried shard keeps committed rows
    assert data["cuda_01"] == [{"kind": "direct", "robot": "go2",
                                "content_sha256": "old01"}]
    # shard absent from the ledger is dropped
    assert "cuda_gone" not in data


def test_aggregate_header_keys_keeps_rows_for_receipt_carried_shards(tmp_path, monkeypatch):
    """Guide §7.z38 (2026-10-02): on a wrapper-only refresh the executed ledger
    holds wrapper modules only — the carried cuda shards enter the merged
    receipt via `gpu-proof merge --carry-from` and are never in `results`.
    Their committed rows must survive (the next refresh replays them); a
    shard the merged receipt no longer attests is still dropped."""
    import run_split_suite as rss

    committed = tmp_path / "gpu-proof-header-keys.json"
    monkeypatch.setattr(rss, "HEADER_KEYS_PATH", committed)
    committed.write_text(json.dumps({"schema": 1, "shards": {
        "cuda_00": [{"kind": "flagship", "robot": "iiwa14",
                     "content_sha256": "old00"}],
        "cuda_01": [{"kind": "direct", "robot": "go2",
                     "content_sha256": "old01"}],
        "cuda_gone": [{"kind": "direct", "robot": "baxter",
                       "content_sha256": "oldgone"}],
    }}))
    rdir = tmp_path / "receipts"
    rdir.mkdir()
    merged = tmp_path / "gpu-proof.json"
    merged.write_text(json.dumps({"shards": [
        {"name": "test_wrapper_a", "carried": False},
        {"name": "cuda_00", "carried": True},
        {"name": "cuda_01", "carried": True},
    ]}))
    results = [{"shard": "test_wrapper_a", "kind": "OK", "domain": "wrappers"}]

    rss.aggregate_header_keys(rdir, results, merged)

    data = json.loads(committed.read_text())["shards"]
    assert data["cuda_00"] == [{"kind": "flagship", "robot": "iiwa14",
                                "content_sha256": "old00"}]
    assert data["cuda_01"] == [{"kind": "direct", "robot": "go2",
                                "content_sha256": "old01"}]
    assert "cuda_gone" not in data
    assert "test_wrapper_a" not in data   # no sidecar, no committed rows


def test_fingerprint_follows_named_runners_and_local_helpers():
    """Runner sources outside the same-stem convention and local helper modules
    must stale the shards that use them (2026-09-28: the integrator/plant/
    second-order smoke runners and executable_cache.py were unfingerprinted)."""
    from test import run_split_suite as R

    paths = R._shard_fingerprint_paths({"test_cuda_integrator_equivalence"})
    assert "test/cuda_equivalents/cuda_integrator_smoke_runner.cu" in paths
    assert "test/cuda_equivalents/executable_cache.py" in paths
    assert "test/cuda_equivalents/cuda_harness.py" in paths
    assert len(paths) == len(set(paths))
    # A module that uses neither keeps its narrow fingerprint.
    narrow = R._shard_fingerprint_paths({"test_cuda_kinematics_thread_invariance"})
    assert "test/cuda_equivalents/executable_cache.py" not in narrow


def test_resume_prunes_dirty_recorded_clean_rows(tmp_path):
    """2026-10-10: a clean resume row whose per-shard receipt recorded
    repo.dirty re-runs (the merge would fold the dirty flag into the merged
    receipt and the release verify would refuse it); a same-sha clean-tree
    row is kept."""
    import subprocess as _sp
    head = _sp.run(["git", "rev-parse", "HEAD"], cwd=rss.REPO_ROOT,
                   capture_output=True, text=True, check=True).stdout.strip()
    (tmp_path / "receipts").mkdir()
    rows = [{"shard": "s_clean", "kind": "OK"}, {"shard": "s_dirty", "kind": "OK"}]
    (tmp_path / "results.json").write_text(json.dumps(rows))
    (tmp_path / "receipts" / "s_clean.json").write_text(
        json.dumps({"repo": {"commit_sha": head, "dirty": False}}))
    (tmp_path / "receipts" / "s_dirty.json").write_text(
        json.dumps({"repo": {"commit_sha": head, "dirty": True}}))
    kept = rss.load_prior_results(tmp_path, {"s_clean", "s_dirty"})
    assert {r["shard"] for r in kept} == {"s_clean"}


def test_plan_refresh_dirty_old_receipt_carries_nothing(monkeypatch):
    """2026-10-10: `gpu-proof merge --carry-from` takes /repo/dirty from the
    FRESH shards only, so carrying from a receipt recorded with a dirty tree
    would launder it. Fingerprint-identical shards are still ALL stale."""
    import codegen_neutrality
    monkeypatch.setattr(codegen_neutrality, "cuda_carry_soundness",
                        lambda r: (True, "stubbed neutral"))
    _W = ["test/python_wrappers/test_w.py", "bindings/grid_rbd", "bindings/src"]
    ids = [_fid("iiwa14", "fixed", "crba", t) for t in (1, 32)]
    shards = [_mk_shard("cuda_00_x", ids, ["a.py"], "d1"),
              _mk_shard("test_w", ["test/python_wrappers/test_w.py::t"], _W, "w")]
    now = {("a.py",): "d1", tuple(_W): "w"}
    fn = lambda paths: now[tuple(paths)]
    clean = {"repo": {"commit_sha": "abc", "dirty": False}, "shards": shards}
    stale, carried, wrap_run, cuda_fresh = rss.plan_refresh(
        clean, ids, ["test_w"], {}, 7200.0, _plan(clean, fn))
    assert set(carried) == {"cuda_00_x", "test_w"} and not stale
    dirty = {"repo": {"commit_sha": "abc", "dirty": True}, "shards": shards}
    stale, carried, wrap_run, cuda_fresh = rss.plan_refresh(
        dirty, ids, ["test_w"], {}, 7200.0, _plan(dirty, fn))
    assert not carried
    assert set(stale) == {"cuda_00_x", "test_w"}
    assert wrap_run == ["test_w"]
    assert {s.name for s in cuda_fresh} == {"cuda_00_x"}
    assert sorted(i for s in cuda_fresh for i in s.targets) == sorted(ids)


def test_driver_import_puts_repo_root_ahead_of_stdlib_test_package(tmp_path):
    """2026-10-10: the no-op --resume crashed with ModuleNotFoundError on
    `test.receipt_integrity` (the stdlib `test` package shadowed the repo's)
    because the sys.path insert lived only in the shard/prewarm helpers.
    Importing the driver from an unrelated cwd must be enough."""
    import subprocess as _sp
    code = ("import sys; sys.path.insert(0, %r); import run_split_suite; "
            "from test.receipt_integrity import verify_header_keys; print('ok')"
            % str(rss.REPO_ROOT / "test"))
    r = _sp.run([sys.executable, "-c", code], cwd=tmp_path,
                capture_output=True, text=True)
    assert r.returncode == 0 and r.stdout.strip() == "ok", r.stderr


def test_plan_refresh_uses_the_real_plan_carry(tmp_path, monkeypatch):
    """Integration with pytest-gpu-proof's OWN `plan_carry` (0.4.1): the base
    verdicts plan_refresh consumes come from the tool against a real git repo,
    so a clean shard carries, a touched test file stales exactly that shard,
    and a dirty recording tree stales everything."""
    import subprocess
    pytest.importorskip("pytest_gpu_proof")  # needs CPython 3.11 or newer
    from pytest_gpu_proof.fingerprint import compute_fingerprint
    from pytest_gpu_proof.merge import plan_carry

    repo = tmp_path / "repo"
    (repo / "test/python_wrappers").mkdir(parents=True)
    (repo / "bindings/grid_rbd").mkdir(parents=True)
    (repo / "test/python_wrappers/test_a.py").write_text("def test_a(): pass\n")
    (repo / "test/python_wrappers/test_b.py").write_text("def test_b(): pass\n")
    (repo / "bindings/grid_rbd/__init__.py").write_text("x = 1\n")

    def git(*a):
        return subprocess.check_output(["git", *a], cwd=repo, text=True).strip()
    git("init", "-q"); git("config", "user.email", "t@example.invalid")
    git("config", "user.name", "t"); git("add", "."); git("commit", "-qm", "base")
    sha = git("rev-parse", "HEAD")

    def shard(mod):
        paths = [f"test/python_wrappers/{mod}.py", "bindings/grid_rbd"]
        return {"name": mod, "node_ids": [f"test/python_wrappers/{mod}.py::t"],
                "fingerprint": compute_fingerprint(paths, root=str(repo)),
                "carried": None, "dirty": False}
    old = {"schema_version": "3", "repo": {"commit_sha": sha, "dirty": False},
           "shards": [shard("test_a"), shard("test_b")]}

    plan = plan_carry(old, repo_root=str(repo))
    assert plan["receipt"]["carryable"] and \
        {s["name"]: s["verdict"] for s in plan["shards"]} == {"test_a": "carryable", "test_b": "carryable"}
    stale, carried, wrap_run, fresh = rss.plan_refresh(old, [], ["test_a", "test_b"], {}, 7200.0, plan)
    assert not stale and set(carried) == {"test_a", "test_b"} and wrap_run == [] and fresh == []

    (repo / "test/python_wrappers/test_a.py").write_text("def test_a(): pass  # touched\n")
    git("commit", "-qam", "touch a")
    plan = plan_carry(old, repo_root=str(repo))
    assert {s["name"]: s["verdict"] for s in plan["shards"]} == {"test_a": "stale", "test_b": "carryable"}
    stale, carried, wrap_run, fresh = rss.plan_refresh(old, [], ["test_a", "test_b"], {}, 7200.0, plan)
    assert stale == ["test_a"] and carried == ["test_b"] and wrap_run == ["test_a"]

    dirty = dict(old, repo={"commit_sha": sha, "dirty": True})
    plan = plan_carry(dirty, repo_root=str(repo))
    assert not plan["receipt"]["carryable"]
    stale, carried, wrap_run, fresh = rss.plan_refresh(dirty, [], ["test_a", "test_b"], {}, 7200.0, plan)
    assert set(stale) == {"test_a", "test_b"} and not carried and set(wrap_run) == {"test_a", "test_b"}


def test_resign_module_when_receipt_level_fingerprint_moved():
    """No stale shard + moved receipt-level fingerprint → re-run the cheapest
    carried wrapper module so the merge can re-sign at HEAD; unmoved → None."""
    old = {"fingerprint": {"digest": "OLD"}, "shards": [
        _mk_shard("cuda_00", ["test/cuda_equivalents/a.py::t"], ["a"], "x"),
        _mk_shard("test_big", ["test/python_wrappers/test_big.py::t1",
                               "test/python_wrappers/test_big.py::t2"], ["b"], "x"),
        _mk_shard("test_small", ["test/python_wrappers/test_small.py::t"], ["c"], "x"),
    ]}
    dur = {"test/python_wrappers/test_big.py::t1": 5.0,
           "test/python_wrappers/test_big.py::t2": 5.0,
           "test/python_wrappers/test_small.py::t": 1.0}
    carried = ["cuda_00", "test_big", "test_small"]
    assert rss.resign_module_if_fingerprint_moved(old, carried, dur, "OLD") is None
    assert rss.resign_module_if_fingerprint_moved(old, carried, dur, None) is None
    assert rss.resign_module_if_fingerprint_moved(old, carried, dur, "NEW") == "test_small"
    # only cuda carried → nothing to re-sign with (caller falls through to verify)
    assert rss.resign_module_if_fingerprint_moved(old, ["cuda_00"], dur, "NEW") is None


def test_warm_entries_restricted_to_running_modules():
    """A refresh that re-runs ONE module warms only that module's robots."""
    everything = {d for d, _ in rss.warm_entries()}
    mod = next(m for m in rss.WARM_MANIFEST if rss.WARM_MANIFEST[m])
    only = {d for d, _ in rss.warm_entries([mod])}
    assert only == {rss._warm_desc(e) for e in rss.WARM_MANIFEST[mod]}
    assert only <= everything and len(only) < len(everything)
    assert rss.warm_entries([]) == []
    assert rss.warm_entries(None) == rss.warm_entries()


def test_report_pool_labels_drained_jobs_cancelled_not_failed(tmp_path, capsys):
    from types import SimpleNamespace
    import compile_sched
    (tmp_path / "compile").mkdir()
    results = {
        "warm00_ok": compile_sched.JobResult("warm00_ok", 0, 1.0, 0),
        "warm01_killed": compile_sched.JobResult("warm01_killed", -15, 1.0, 0),
        "warm02_never": compile_sched.JobResult("warm02_never", -1, 0.0, 0, "not attempted (stopped)"),
        "warm03_real_fail": compile_sched.JobResult("warm03_real_fail", 2, 1.0, 0),
    }
    pool = SimpleNamespace(results=results, stopped=True, ledger=None)
    assert rss.report_pool(pool, tmp_path) == 1          # only the rc=2 job is a failure
    out = capsys.readouterr().out
    assert out.count("POOL-CANCELLED") == 2 and "POOL-FAIL warm03_real_fail" in out
    # an UNstopped pool reports every non-zero rc as a failure (no silent masking)
    pool = SimpleNamespace(results=results, stopped=False, ledger=None)
    assert rss.report_pool(pool, tmp_path) == 3
