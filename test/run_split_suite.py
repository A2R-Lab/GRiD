#!/usr/bin/env python3
"""Split-compile / split-run driver for the python_wrappers + cuda_equivalents suites.

Why this exists: the monolithic ``pytest -m python_wrappers`` run dlopens every
robot ``.so`` (plus torch AND jax) into one interpreter, and the generated host
wrappers ``exit(1)``/abort on GPU errors — one C++ death loses the names of
every failure after it (three times in the week of 2026-08-03). This driver:

  Phase A (warm): compiles every cache-missing robot ``.so`` via
      ``grid_rbd.warm_robot`` — codegen + nvcc only, NO handle, NO CUDA context,
      so a compile failure is its own named row, never a mid-suite surprise.
  Phase B (run):  runs each test module in its OWN pytest subprocess (serial on
      the GPU) with ``--junitxml`` and an explicitly captured exit code. A module
      that dies (SIGABRT etc.) is a named CASUALTY row; the driver continues.
  Aggregate:      merges the per-module JUnit XMLs + rcs into one table and one
      overall exit code (nonzero iff any failure/casualty).

Usage (from the repo root):
  .venv/bin/python test/run_split_suite.py                 # full wrappers split suite
  .venv/bin/python test/run_split_suite.py --modules test_tool test_iiwa14_smoke
  .venv/bin/python test/run_split_suite.py --skip-warm     # straight to Phase B
  .venv/bin/python test/run_split_suite.py --receipts      # + per-shard gpu-proof
                                                           #   receipts
  .venv/bin/python test/run_split_suite.py --domains wrappers,cuda --receipts
                                                           # + granular cuda shards
  .venv/bin/python test/run_split_suite.py --resume test/.split_suite/<dir>
                                                           # continue an interrupted run

Granular cuda shards (--domains ...,cuda): test/cuda_equivalents used to run as
ONE monolithic pytest (~15h cold, 2026-08-16). The driver now partitions its
gpu_proof tests into shards bounded by --shard-budget-mins (default 120) using
EXPLICIT node-id lists (never -k), runs each as its own crash-isolated pytest,
and bounds pause latency to one shard: touch <out>/PAUSE to stop cleanly
between shards, Ctrl-C/SIGTERM to stop within one, then --resume to continue —
completed shards are never re-run (ledger: <out>/results.json) and their shard
receipts merge together at the end.

Notes:
- Modules are DISCOVERED by glob (``test/python_wrappers/test_*.py``); the warm
  manifest below is only an optimization + drift-guard, so a brand-new module is
  never silently skipped — it just isn't pre-warmed.
- The conftest SUITE FLOOR turns env-guard skips into rc=1 per invocation; the
  aggregator reconciles "rc!=0 but zero XML failures" as FLOOR rows rather than
  casualties.
- Modules with ``force_rebuild=True`` registrations recompile in Phase B by
  design (that is what those tests exercise); they are warm-exempt here.
"""
from __future__ import annotations

import argparse
import ast
import glob
import json
import os
import subprocess
import sys
import time
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
WRAPPERS_DIR = REPO_ROOT / "test" / "python_wrappers"
CUDA_DIR = REPO_ROOT / "test" / "cuda_equivalents"
PYTHON = str(REPO_ROOT / ".venv" / "bin" / "python")
DURATIONS_PATH = REPO_ROOT / "test" / ".split_suite" / "durations.json"
# Result kinds the summary treats as clean (and --resume will not re-run).
CLEAN_KINDS = ("OK", "FLOOR-SKIP", "DESELECTED")


@dataclass
class ShardSpec:
    """One crash-isolated pytest invocation.

    wrappers domain: one module file per shard (name = module stem — keeps
    shard names, receipts, and the --changed-only state file byte-compatible
    with the module-based driver).
    cuda domain: an explicit node-id list (targets), bounded by the shard
    budget. Targets are passed as argv elements, NEVER through a shell string
    — two codegen_layout node ids embed raw CUDA source (spaces/quotes/escapes).
    """
    name: str
    domain: str                 # "wrappers" | "cuda"
    targets: list               # pytest targets: [module path] or [node ids...]
    fingerprint_paths: list     # narrow receipt fingerprint paths (comma-free)
    est_secs: float = 0.0
    apply_marker: bool = False  # add -m gpu_proof (cuda shards)

# ─── warm manifest ───────────────────────────────────────────────────────────
# module -> list of warm_robot kwarg dicts (cache-key-relevant kwargs only).
# URDFs resolve through config.robot_urdf at runtime. Modules NOT listed (or
# listed with []) still run in Phase B — they are just not pre-warmed:
#   * force_rebuild=True modules rebuild in-test regardless;
#   * test_subset_build* use isolated tmp cache_dirs;
#   * test_any_thread_count / test_ee_named_target_* use robot_descriptions /
#     per-test inline registrations.
def _r(robot, **kw):
    return {"robot": robot, **kw}

WARM_MANIFEST: dict[str, list[dict]] = {
    "test_centroidal_energy_frame": [_r("iiwa14", max_batch_size=8),
                                     _r("go2", max_batch_size=8)],
    # 08-13: was warm-exempt ("inline registrations") — the 08-12 night pass cold-
    # built all 5 robots IN-MODULE and blew the 7200s cap mid-test-11 (10/12
    # already passed; solo rerun green). The registrations resolve through the
    # same config URDFs, so they ARE warmable; mb values are the module's exact
    # len(samples) (floating=21, fixed=18) — cache-key material, keep in sync.
    "test_ee_named_target_floating_multileaf": [
        _r("iiwa14", max_batch_size=21, floating_base=True,
           ee_joint_names=["iiwa_joint_ee"], enable_mujoco_kernels=False),
        _r("go2", max_batch_size=21, floating_base=True,
           ee_joint_names=["imu_joint"], enable_mujoco_kernels=False),
        _r("baxter", max_batch_size=21, floating_base=True,
           ee_joint_names=["right_hand_camera_axis"], enable_mujoco_kernels=False),
        _r("go2", max_batch_size=21, floating_base=True,
           enable_mujoco_kernels=False),
        _r("baxter", max_batch_size=21, floating_base=True,
           enable_mujoco_kernels=False),
        _r("iiwa14", max_batch_size=18,
           ee_joint_names=["iiwa_joint_ee"], enable_mujoco_kernels=False),
    ],
    "test_fk_batched": [_r("iiwa14", max_batch_size=64),
                        _r("gen3", max_batch_size=64),
                        _r("go2", max_batch_size=64),
                        _r("fr3", max_batch_size=64),
                        _r("iiwa14", max_batch_size=64, floating_base=True,
                           enable_mujoco_kernels=False)],
    "test_fp64_parity": [_r("iiwa14", max_batch_size=8, dtype="float64"),
                         _r("iiwa14", max_batch_size=8)],
    "test_g1_plant_hessian_smoke": [_r("g1", max_batch_size=8,
                                       algorithm_list=["integrator_hessian"])],
    "test_iiwa14_f_ext": [_r("iiwa14", max_batch_size=8)],
    "test_iiwa14_jax_smoke": [_r("iiwa14", max_batch_size=8)],
    "test_iiwa14_plant_smoke": [_r("iiwa14", max_batch_size=8)],
    "test_iiwa14_smoke": [_r("iiwa14", max_batch_size=8)],
    "test_iiwa14_torch_smoke": [_r("iiwa14", max_batch_size=64)],
    "test_kinematics_thread_batch_matrix": [
        _r("iiwa14", max_batch_size=256, enable_mujoco_kernels=False),
        _r("go2", max_batch_size=256, floating_base=True,
           enable_mujoco_kernels=False)],
    "test_runtime_inertia": [_r("iiwa14", runtime_inertia=True),
                             _r("iiwa14", runtime_inertia=True, dtype="float64"),
                             _r("iiwa14"),
                             _r("go2", floating_base=True, runtime_inertia=True,
                                enable_mujoco_kernels=False)],
    "test_runtime_transform": [_r("iiwa14", max_batch_size=8,
                                  runtime_transform=True)],
    "test_tool": [_r("iiwa14", max_batch_size=8, enable_tool=True)],
}

# 08-13: wall-clock caps RETIRED (they killed a healthy cold-building module on
# the 08-12 night pass). Phase B now uses PROGRESS-AWARE hang detection — see
# _wait_progress_aware: kill only after GRID_SPLIT_STALL_SECS (default 900) with
# neither log growth nor a live compiler child; an absolute cap is opt-in via
# GRID_SPLIT_HARD_TIMEOUT. Historical expected COLD durations, for triage only:
# g1_plant_hessian / joint_dynamics / runtime_joint_dynamics ~1-2h (multi-build,
# incl. float64); ee_named_target_floating_multileaf ~2h (5 in-module builds).


def discover_modules() -> list[str]:
    mods = sorted(Path(p).stem for p in glob.glob(str(WRAPPERS_DIR / "test_*.py")))
    return mods


# ─── cuda_equivalents granular partition ─────────────────────────────────────
# Atom = the smallest node-id group never split across shards:
#   * test_cuda_executable_equivalence (720 tests = 72% of the suite):
#     (robot, base, cell) — its 4 thread-count variants share exactly ONE
#     header+exe compile (neither cache key folds num_threads), so splitting
#     them would pay the compile twice;
#   * test_cuda_second_order_fallback: (robot, base) — SO codegen dominates
#     (g1/h1_2 are hours cold);
#   * integrator / kinematics_thread_invariance: (robot);
#   * every other module: whole-module. That automatically keeps the tree's
#     only module-scoped compile fixture (spherical_integrator_gradient) whole,
#     and never param-parses the two codegen_layout ids that embed raw CUDA
#     source (their module isn't in ATOM_PARAM_TOKENS).
# Robot is always the first '-'-token inside [...] (no robot id contains '-').
_FLAGSHIP = "test_cuda_executable_equivalence"
ATOM_PARAM_TOKENS = {
    _FLAGSHIP: 3,
    "test_cuda_second_order_fallback": 2,
    "test_cuda_integrator_equivalence": 1,
    "test_cuda_kinematics_thread_invariance": 1,
}

# Conservative COLD estimates (seconds) — only used for node ids that have no
# measured entry in test/.split_suite/durations.json (rolling, written after
# every green --receipts run). Over-estimating just splits shards finer; an
# atom estimated over budget gets its OWN shard and is logged — never split
# (would duplicate its compile), never silently capped.
DEFAULT_MODULE_EST_SECS = {
    "test_cuda_codegen_layout": 5400.0,
    "test_cuda_plant_equivalence": 3600.0,
    "test_cuda_spherical_fdsva_so_equivalence": 2700.0,
    "test_cuda_spherical_equivalence": 2400.0,
    "test_cuda_spherical_so_equivalence": 1800.0,
    "test_cuda_idsva_so_world_frame": 1800.0,
    "test_smem_poison": 1800.0,  # nested pytest over ~20 flagship iiwa14 cells
}


def _default_atom_est(key: tuple, n_ids: int) -> float:
    mod = key[0]
    if mod == _FLAGSHIP:
        robot = key[1] if len(key) > 1 else ""
        base = 480.0
        if robot in ("g1", "h1_2"):
            base *= 2.5
        elif robot in ("baxter", "fetch"):
            base *= 1.5
        return base
    if mod == "test_cuda_second_order_fallback":
        robot = key[1] if len(key) > 1 else ""
        return 5400.0 if robot in ("g1", "h1_2") else 1800.0
    if mod in ATOM_PARAM_TOKENS:
        return 150.0 * n_ids
    return DEFAULT_MODULE_EST_SECS.get(mod, 240.0 + 90.0 * n_ids)


def load_durations() -> dict:
    if DURATIONS_PATH.exists():
        try:
            return json.loads(DURATIONS_PATH.read_text())
        except json.JSONDecodeError:
            pass
    return {}


def _atom_est(members: list[str], durations: dict, key: tuple) -> float:
    known = [durations.get(i) for i in members]
    if known and all(isinstance(v, (int, float)) for v in known):
        # Measured (warm-run) durations + headroom; +10s covers per-test
        # cache-stat / process overhead the receipt's duration_s misses.
        return sum(known) * 1.15 + 10.0
    return _default_atom_est(key, len(members))


def _atom_key(nid: str) -> tuple:
    mod = Path(nid.split("::", 1)[0]).stem
    ntok = ATOM_PARAM_TOKENS.get(mod, 0)
    if not ntok or "[" not in nid:
        return (mod,)
    params = nid.split("[", 1)[1].rstrip("]")
    return (mod, *params.split("-")[:ntok])


def collect_cuda_node_ids(cuda_k: str | None) -> list[str]:
    """Full gpu_proof collection (optionally SCOPE-narrowed at COLLECTION time,
    so shard receipts attest exactly the narrowed scope — a valid narrower
    receipt, with no per-shard -k deselection noise at run time)."""
    cmd = [PYTHON, "-m", "pytest", "test/cuda_equivalents", "-m", "gpu_proof",
           "--collect-only", "-q", "-p", "no:cacheprovider"]
    if cuda_k:
        cmd += ["-k", cuda_k]
    r = subprocess.run(cmd, cwd=REPO_ROOT, capture_output=True, text=True,
                       timeout=900)
    ids = [ln for ln in r.stdout.splitlines()
           if ln.startswith("test/cuda_equivalents/") and "::" in ln]
    if r.returncode not in (0, 5) or (r.returncode == 0 and not ids):
        raise RuntimeError(
            f"cuda collect-only failed (rc={r.returncode}):\n"
            f"{r.stdout[-2000:]}\n{r.stderr[-2000:]}")
    return ids


def _shard_fingerprint_paths(member_mods: set[str]) -> list[str]:
    """Member module files + their same-stem runner .cu (test_cuda_X.py ↔
    cuda_X_runner.cu convention) — finer than the whole-directory fingerprint
    the monolithic shard used. Paths must stay comma-free (plugin flag is
    comma-joined)."""
    paths = []
    for mod in sorted(member_mods):
        paths.append(f"test/cuda_equivalents/{mod}.py")
        runner = mod.removeprefix("test_") + "_runner.cu"
        if (CUDA_DIR / runner).exists():
            paths.append(f"test/cuda_equivalents/{runner}")
    if _FLAGSHIP in member_mods:
        for extra in ("cuda_equivalence_runner.cu", "grid_runner_select.cuh"):
            if (CUDA_DIR / extra).exists():
                paths.append(f"test/cuda_equivalents/{extra}")
    return paths


def pack_cuda_shards(ids: list[str], durations: dict,
                     budget_secs: float) -> list[ShardSpec]:
    atoms: dict[tuple, list[str]] = {}
    for nid in ids:  # collection order → same-module atoms stay adjacent
        atoms.setdefault(_atom_key(nid), []).append(nid)

    shards: list[ShardSpec] = []
    cur_ids: list[str] = []
    cur_mods: set[str] = set()
    cur_tags: list[str] = []
    cur_est = 0.0

    def _close() -> None:
        nonlocal cur_ids, cur_mods, cur_tags, cur_est
        if not cur_ids:
            return
        hint = "_".join(dict.fromkeys(cur_tags))[:32] or "misc"
        shards.append(ShardSpec(
            name=f"cuda_{len(shards):02d}_{hint}", domain="cuda",
            targets=list(cur_ids),
            fingerprint_paths=_shard_fingerprint_paths(cur_mods),
            est_secs=cur_est, apply_marker=True))
        cur_ids, cur_mods, cur_tags, cur_est = [], set(), [], 0.0

    for key, members in atoms.items():
        est = _atom_est(members, durations, key)
        if cur_ids and cur_est + est > budget_secs:
            _close()
        cur_ids += members
        cur_mods.add(key[0])
        cur_tags.append(key[1] if key[0] == _FLAGSHIP and len(key) > 1
                        else key[0].removeprefix("test_cuda_")
                        .removeprefix("test_")[:18])
        cur_est += est
        if cur_est > budget_secs:  # single over-budget atom → own shard
            _close()
    _close()

    # COMPLETENESS GATE: the packed union must be set-equal to collection.
    # (pytest-gpu-proof's verifier re-checks exact partition downstream, but
    # failing here beats failing after hours of GPU time.)
    packed = sorted(i for s in shards for i in s.targets)
    if packed != sorted(ids):
        raise RuntimeError(
            f"cuda partition not set-equal to collection "
            f"({len(packed)} packed vs {len(ids)} collected)")
    return shards


def build_cuda_shards(cuda_k: str | None,
                      budget_secs: float) -> tuple[list[ShardSpec], list[str]]:
    ids = collect_cuda_node_ids(cuda_k)
    if not ids:
        return [], []
    return pack_cuda_shards(ids, load_durations(), budget_secs), ids


def plan_refresh(old_receipt: dict, cuda_ids_now: list[str],
                 wrapper_mods_now: list[str], durations: dict,
                 budget_secs: float, fingerprint_digest_fn):
    """Shard-level receipt refresh (user decision 2026-08-20): re-run ONLY the
    shards whose narrow fingerprints changed vs `old_receipt`; the rest are
    carried by `gpu-proof merge --carry-from` (policy allow_carried gates
    acceptance — the everyday policy allows it, the release policy refuses it).

    Returns (stale_names, carried_names, wrapper_mods_to_run, cuda_fresh_shards).
    `fingerprint_digest_fn(paths) -> digest` must be pytest-gpu-proof's OWN
    compute_fingerprint (imported, not reimplemented) so the staleness decision
    can never drift from what carry_forward/verify recompute downstream.

    Safety shape: fresh cuda shards are packed from {current collection} minus
    {carryable shards' node ids}; a carryable shard whose ids vanished from the
    collection is impossible via file edits (removing a test edits its module,
    which is in the fingerprint) — if it happens anyway (env drift), we refuse
    loudly rather than emit an incomplete receipt.

    NAME RECYCLING (the 2026-08-21 first-run lesson): carry_forward walks EVERY
    old shard — an old shard is either freshly re-run (same name; fresh wins)
    or it must fingerprint-match to carry. There is no "superseded" state, so a
    stale shard's name MUST reappear among the fresh shards or the merge
    refuses. Fresh cuda coverage is therefore partitioned into EXACTLY the
    stale cuda shards' names; wholly-new tests (no stale name to recycle) get
    new names, de-conflicted from carried names (a same-named fresh shard would
    silently shadow a carried one, orphaning its node ids). A stale WRAPPER
    shard whose module was deleted has no fresh run to shadow it → refuse
    (full pass; deleting a test module is release-grade anyway)."""
    from dataclasses import replace as _dc_replace  # noqa: PLC0415

    shards = old_receipt.get("shards") or []
    if not shards:
        raise RuntimeError(
            "--refresh-from: receipt has no schema-2 shards[] — cannot "
            "shard-refresh a monolithic receipt; run a full SPLIT=1 pass")

    def _is_wrapper(s: dict) -> bool:
        ids = s.get("node_ids") or []
        return bool(ids) and str(ids[0]).startswith("test/python_wrappers/")

    carried, stale = [], []
    for s in shards:
        fp = s.get("fingerprint") or {}
        paths = fp.get("included_paths") or []
        # 2026-09-09 fingerprint-schema upgrade: wrapper shards fingerprint the
        # bindings SOURCE too (_BINDINGS_FP at shard construction — a bindings-
        # only change must stale the tests that import it). A wrapper shard
        # recorded under the old test-file-only paths is stale BY DEFINITION;
        # one re-run installs the new paths, then digest comparison resumes.
        if _is_wrapper(s) and "bindings/grid_rbd" not in paths:
            stale.append(s)
            continue
        now = fingerprint_digest_fn(paths)
        (carried if now == fp.get("digest") else stale).append(s)

    old_wrapper_names = {s["name"] for s in shards if _is_wrapper(s)}
    stale_wrapper_names = {s["name"] for s in stale if _is_wrapper(s)}
    # Stale wrapper shard whose module still exists → re-run it; a module NEW
    # since the old receipt runs too. A stale wrapper shard with NO current
    # module has no fresh namesake to shadow it in carry_forward → refuse.
    deleted = stale_wrapper_names - set(wrapper_mods_now)
    if deleted:
        raise RuntimeError(
            f"--refresh-from: stale wrapper shard(s) {sorted(deleted)} have no "
            f"current module — carry_forward has no 'superseded' state, so a "
            f"deleted module needs a full SPLIT=1 pass.")
    wrapper_to_run = [m for m in wrapper_mods_now
                      if m in stale_wrapper_names or m not in old_wrapper_names]

    carried_cuda_ids = {i for s in carried if not _is_wrapper(s)
                        for i in (s.get("node_ids") or [])}
    missing = carried_cuda_ids - set(cuda_ids_now)
    if missing:
        raise RuntimeError(
            f"--refresh-from: {len(missing)} node id(s) in fingerprint-clean "
            f"shards are no longer collectable (e.g. {sorted(missing)[0]!r}) — "
            f"collection changed without a fingerprint change; a refresh "
            f"cannot preserve completeness. Run a full SPLIT=1 pass.")
    fresh_ids = [i for i in cuda_ids_now if i not in carried_cuda_ids]
    stale_cuda_names = [s["name"] for s in stale if not _is_wrapper(s)]

    cuda_fresh: list[ShardSpec] = []
    if stale_cuda_names:
        # Every stale cuda name must be shadowed by a fresh shard: partition
        # the fresh ids into EXACTLY those names (contiguous atom groups,
        # est-balanced — same-module atoms stay adjacent like pack_cuda_shards).
        atoms: dict[tuple, list[str]] = {}
        for nid in fresh_ids:
            atoms.setdefault(_atom_key(nid), []).append(nid)
        atom_rows = [(k, m, _atom_est(m, durations, k)) for k, m in atoms.items()]
        n_groups = len(stale_cuda_names)
        if len(atom_rows) < n_groups:
            raise RuntimeError(
                f"--refresh-from: only {len(atom_rows)} fresh atom(s) for "
                f"{n_groups} stale shard name(s) — cannot shadow every stale "
                f"shard. Run a full SPLIT=1 pass.")
        total_est = sum(e for _, _, e in atom_rows) or 1.0
        gi, cur_ids, cur_mods, cur_est = 0, [], set(), 0.0
        for idx, (key, members, est) in enumerate(atom_rows):
            cur_ids += members
            cur_mods.add(key[0])
            cur_est += est
            atoms_left = len(atom_rows) - idx - 1
            groups_left = n_groups - gi - 1
            if groups_left > 0 and (
                    cur_est >= total_est / n_groups
                    or atoms_left == groups_left):
                cuda_fresh.append(ShardSpec(
                    name=stale_cuda_names[gi], domain="cuda",
                    targets=list(cur_ids),
                    fingerprint_paths=_shard_fingerprint_paths(cur_mods),
                    est_secs=cur_est, apply_marker=True))
                gi, cur_ids, cur_mods, cur_est = gi + 1, [], set(), 0.0
        cuda_fresh.append(ShardSpec(
            name=stale_cuda_names[gi], domain="cuda", targets=list(cur_ids),
            fingerprint_paths=_shard_fingerprint_paths(cur_mods),
            est_secs=cur_est, apply_marker=True))
        packed = sorted(i for s in cuda_fresh for i in s.targets)
        if packed != sorted(fresh_ids) or len(cuda_fresh) != n_groups:
            raise RuntimeError("refresh partition failed its completeness gate")
    elif fresh_ids:
        # Wholly-new tests with no stale name to recycle: new shards, names
        # de-conflicted from carried names (a same-named fresh shard would
        # shadow a carried one and orphan its node ids).
        cuda_fresh = pack_cuda_shards(fresh_ids, durations, budget_secs)
        taken = {s["name"] for s in carried} | set(wrapper_mods_now)
        deconflicted = []
        for sp in cuda_fresh:
            nm = sp.name
            while nm in taken:
                nm += "_r"
            taken.add(nm)
            deconflicted.append(sp if nm == sp.name else _dc_replace(sp, name=nm))
        cuda_fresh = deconflicted
    return ([s["name"] for s in stale],
            [s["name"] for s in carried],
            wrapper_to_run, cuda_fresh)


def load_prior_results(out_dir: Path, spec_names: set[str]) -> list[dict]:
    """--resume: keep only CLEAN rows for shards that still exist; everything
    else (failures, casualties, vanished shards) re-runs."""
    path = out_dir / "results.json"
    if not path.exists():
        return []
    try:
        prior = json.loads(path.read_text())
    except json.JSONDecodeError:
        return []
    return [dict(r, fresh=False) for r in prior
            if r.get("kind") in CLEAN_KINDS and r.get("shard") in spec_names]


def save_partition(out_dir: Path, shards: list[ShardSpec]) -> None:
    """Persist the cuda partition so --resume re-runs the SAME shards even if
    durations/estimates have changed since the run started."""
    data = [dict(name=s.name, domain=s.domain, targets=s.targets,
                 fingerprint_paths=s.fingerprint_paths, est_secs=s.est_secs,
                 apply_marker=s.apply_marker) for s in shards]
    (out_dir / "partition.json").write_text(json.dumps(data, indent=2) + "\n")


def load_partition(out_dir: Path) -> list[ShardSpec] | None:
    p = out_dir / "partition.json"
    if not p.exists():
        return None
    return [ShardSpec(**d) for d in json.loads(p.read_text())]


def update_durations(merged_receipt: Path) -> None:
    """Roll per-node duration_s from a successfully merged receipt into the
    durations map — the bin-packer's calibration data for the NEXT run."""
    try:
        data = json.loads(merged_receipt.read_text())
    except (OSError, json.JSONDecodeError):
        return
    durations = load_durations()
    n = 0
    for t in data.get("tests", []):
        nid, dur = t.get("node_id"), t.get("duration_s")
        if isinstance(nid, str) and isinstance(dur, (int, float)):
            durations[nid] = dur
            n += 1
    if n:
        DURATIONS_PATH.parent.mkdir(parents=True, exist_ok=True)
        DURATIONS_PATH.write_text(
            json.dumps(durations, indent=2, sort_keys=True) + "\n")
        print(f"  durations: recorded {n} node duration(s) -> "
              f"{DURATIONS_PATH.relative_to(REPO_ROOT)}")


# ─── change-aware selection ──────────────────────────────────────────────────
# A module's input fingerprint covers everything that can change its outcome:
# the module file itself (registration kwargs live in it), the shared conftest,
# the manifest robots' URDF bytes, the codegen-source + wrapper-template hashes
# the bindings cache key uses, the grid_rbd version, the CUDA arch, and the
# nvcc version (the one input the bindings cache key does NOT fold). Skip
# decisions come from THIS fingerprint (strictly stronger than the receipt's
# narrow shard fingerprint, which only covers git-tracked test files — GRiD
# pins codegen/submodules by commit SHA at the receipt level instead).
STATE_PATH = REPO_ROOT / "test" / ".split-suite-state.json"


def _nvcc_version() -> str:
    try:
        out = subprocess.run(["nvcc", "--version"], capture_output=True, text=True,
                             timeout=30).stdout
        return out.strip().splitlines()[-1] if out else "unknown"
    except Exception:
        return "unknown"


def module_fingerprint(mod: str) -> str:
    import hashlib
    sys.path.insert(0, str(REPO_ROOT))
    from config import robot_urdf  # noqa: PLC0415
    from grid_rbd._cache import (  # noqa: PLC0415
        _codegen_source_hash, _wrapper_template_hash, detect_cuda_arch,
        package_version)

    h = hashlib.sha256()
    h.update((WRAPPERS_DIR / f"{mod}.py").read_bytes())
    h.update((REPO_ROOT / "test" / "conftest.py").read_bytes())
    for e in WARM_MANIFEST.get(mod, []):
        try:
            urdf = robot_urdf(e["robot"])
            if urdf.exists():
                h.update(urdf.read_bytes())
        except Exception:
            h.update(f"unresolved:{e['robot']}".encode())
    h.update(_codegen_source_hash().encode())
    h.update(_wrapper_template_hash().encode())
    h.update(package_version().encode())
    h.update(str(detect_cuda_arch()).encode())
    h.update(_nvcc_version().encode())
    return h.hexdigest()


def load_state() -> dict:
    if STATE_PATH.exists():
        try:
            return json.loads(STATE_PATH.read_text())
        except json.JSONDecodeError:
            pass
    return {}


def save_state(state: dict) -> None:
    STATE_PATH.write_text(json.dumps(state, indent=2, sort_keys=True) + "\n")


def drift_guard(modules: list[str]) -> list[str]:
    """Soft check: modules whose register_robot call-site count is not covered by
    the warm manifest get a warning row (never an error — Phase B still runs
    them; this only flags that the WARM table has drifted from the code)."""
    warnings = []
    for mod in modules:
        src = (WRAPPERS_DIR / f"{mod}.py").read_text()
        try:
            tree = ast.parse(src)
        except SyntaxError as e:  # a broken module is Phase B's problem to report
            warnings.append(f"{mod}: unparseable ({e})")
            continue
        calls = sum(
            isinstance(n, ast.Call)
            and (
                (isinstance(n.func, ast.Attribute) and n.func.attr == "register_robot")
                or (isinstance(n.func, ast.Name) and n.func.id == "register_robot")
            )
            for n in ast.walk(tree)
        )
        warmed = len(WARM_MANIFEST.get(mod, []))
        force_rebuild = "force_rebuild=True" in src
        if warmed == 0 and calls > 0 and not force_rebuild and mod not in (
                "test_any_thread_count",
                "test_subset_build", "test_subset_build_jax", "test_subset_build_torch"):
            warnings.append(
                f"{mod}: {calls} register_robot site(s), none warmed and not "
                f"force_rebuild/exempt — extend WARM_MANIFEST?")
    return warnings


def _warm_desc(entry: dict) -> str:
    kw = dict(entry)
    robot = kw.pop("robot")
    return robot + "".join(
        f",{k}={v}" for k, v in sorted(kw.items()) if k != "max_batch_size"
    ) + f",mb={kw.get('max_batch_size', 256)}"


def warm_entries() -> list[tuple[str, dict]]:
    """Deduped Phase A warm targets as (desc, warm_robot-kwargs)."""
    seen: set[str] = set()
    out: list[tuple[str, dict]] = []
    for _mod, entries in sorted(WARM_MANIFEST.items()):
        for e in entries:
            desc = _warm_desc(e)
            if desc in seen:
                continue
            seen.add(desc)
            out.append((desc, dict(e)))
    return out


def _warm_one_entry(desc: str, entry: dict) -> tuple[str, float, str]:
    """One warm_robot call. Returns (status, secs, detail)."""
    sys.path.insert(0, str(REPO_ROOT))
    from config import robot_urdf  # noqa: PLC0415
    import grid_rbd  # noqa: PLC0415

    kw = dict(entry)
    robot = kw.pop("robot")
    t0 = time.monotonic()
    try:
        urdf = robot_urdf(robot)
        if not urdf.exists():
            return "SKIP", 0.0, f"no urdf: {urdf}"
        key, _so_path, _meta = grid_rbd.warm_robot(
            name=f"__split_warm_{robot}", urdf_path=str(urdf), **kw)
        dt = time.monotonic() - t0
        return ("HIT" if dt < 5.0 else "BUILT"), dt, key[:12]
    except Exception as exc:  # a warm failure must never stop Phase B
        return "COMPILE-FAIL", time.monotonic() - t0, repr(exc)[:200]


def warm_one_worker(spec_path: str) -> int:
    """``--warm-one`` subprocess mode: run ONE warm entry under the RAM-aware
    pool (test/compile_sched.py wraps this process with /usr/bin/time -v) and
    write its outcome JSON where the parent expects it."""
    spec = json.loads(Path(spec_path).read_text())
    status, secs, detail = _warm_one_entry(spec["desc"], spec["entry"])
    Path(spec["result_path"]).write_text(json.dumps(
        {"desc": spec["desc"], "status": status, "secs": secs, "detail": detail}))
    print(f"{status} {spec['desc']} {detail}", flush=True)
    return 0 if status in ("HIT", "BUILT", "SKIP") else 1


def phase_warm(out_dir: Path) -> list[tuple[str, str, float, str]]:
    """Serial fallback (GRID_SPLIT_COMPILE_JOBS=0). Returns
    (robot-desc, status, seconds, detail) rows."""
    rows = []
    for desc, entry in warm_entries():
        status, secs, detail = _warm_one_entry(desc, entry)
        rows.append((desc, status, secs, detail))
    return rows


def parse_junit(xml_path: Path):
    """Returns (tests, failures, errors, skipped, failed_names) or None."""
    if not xml_path.exists():
        return None
    try:
        root = ET.parse(xml_path).getroot()
    except ET.ParseError:
        return None
    suites = root.iter("testsuite")
    t = f = e = s = 0
    failed = []
    for su in suites:
        t += int(su.get("tests", 0))
        f += int(su.get("failures", 0))
        e += int(su.get("errors", 0))
        s += int(su.get("skipped", 0))
        for case in su.iter("testcase"):
            if case.find("failure") is not None or case.find("error") is not None:
                failed.append(f"{case.get('classname')}::{case.get('name')}")
    return t, f, e, s, failed


def _compiler_child_alive(pgid: int) -> bool:
    """True if the module's process group has a live compiler child. Cold .so
    builds are stdout-silent for 30-60+ min while nvcc/cicc/ptxas grind — that
    IS progress; a GPU-hung test has neither output growth nor compiler kids."""
    try:
        out = subprocess.run(["ps", "-eo", "pgid=,comm="], capture_output=True,
                             text=True, timeout=10).stdout
    except Exception:
        return False
    tgt = str(pgid)
    names = ("nvcc", "cicc", "ptxas", "fatbinary", "cudafe", "nvlink")
    for line in out.splitlines():
        parts = line.split(None, 1)
        if len(parts) == 2 and parts[0].strip() == tgt and any(n in parts[1] for n in names):
            return True
    return False


def _kill_group(proc: subprocess.Popen) -> None:
    import signal as _signal
    try:
        os.killpg(proc.pid, _signal.SIGKILL)
    except ProcessLookupError:
        pass
    proc.wait()


def _wait_progress_aware(proc: subprocess.Popen, mod: str, log_path: Path):
    """Hang detection by PROGRESS, not wall clock (the no-leg-timeouts rule —
    the 08-12 night pass killed a healthy module mid-test at a 7200s cap while
    it was legitimately cold-building 5 robots).

    Progress = the module's log grew OR its process group has a live compiler
    child. Kill (rc="STALL") only after GRID_SPLIT_STALL_SECS (default 900)
    with NEITHER. An absolute wall-clock cap is OPT-IN via
    GRID_SPLIT_HARD_TIMEOUT seconds (unset/0 = none)."""
    stall_limit = int(os.environ.get("GRID_SPLIT_STALL_SECS", "900"))
    hard = int(os.environ.get("GRID_SPLIT_HARD_TIMEOUT", "0"))
    t0 = time.monotonic()
    last_progress = t0
    last_size = -1
    while True:
        try:
            return proc.wait(timeout=15)
        except subprocess.TimeoutExpired:
            pass
        now = time.monotonic()
        try:
            size = log_path.stat().st_size
        except OSError:
            size = -1
        if size != last_size:
            last_size = size
            last_progress = now
        elif _compiler_child_alive(proc.pid):
            last_progress = now
        if (now - last_progress) > stall_limit:
            _kill_group(proc)
            return "STALL"
        if hard and (now - t0) > hard:
            _kill_group(proc)
            return "TIMEOUT"


def _write_ledger(out_dir: Path, results: list[dict]) -> None:
    """Incremental per-shard ledger — the --resume source of truth. Rewritten
    (atomic replace) after EVERY shard so an interrupt/pause never loses
    completed work; an in-flight shard is simply absent (= re-runs)."""
    tmp = out_dir / "results.json.tmp"
    tmp.write_text(json.dumps(results, indent=2) + "\n")
    os.replace(tmp, out_dir / "results.json")


CUDA_CACHE_DIR_DEFAULT = str(REPO_ROOT / ".pytest_cache" / "grid_cuda")
# Host-RAM floor once GPU shards are running (jax/torch shards need real RAM
# on top of the compile pool's own floor).
GPU_PHASE_FLOOR_KB = 14 * 1024 * 1024


def build_compile_pool(out_dir: Path, *, warm: bool, cuda_shards: list,
                       modules: list[str], max_jobs: int):
    """Build the RAM-aware compile pool covering BOTH compile populations
    (Phase A wrapper .so warm + cuda flagship header/exe pre-warm) and the
    shard-name -> prerequisite-job-names map that lets GPU shard execution
    OVERLAP the remaining compiles: a shard only waits for ITS OWN compile
    jobs, and the pool is the ONLY writer of a cache key until that wait
    completes (the unlocked cache writers are never raced).

    Returns (pool, jobs, prereqs) — pool/jobs empty-safe (None, [], {}).
    """
    import compile_sched  # noqa: PLC0415  (test/ is on sys.path for scripts)

    cdir = out_dir / "compile"
    cdir.mkdir(exist_ok=True)
    jobs: list = []
    prereqs: dict[str, list[str]] = {}

    desc_to_job: dict[str, str] = {}
    if warm:
        for i, (desc, entry) in enumerate(warm_entries()):
            name = f"warm{i:02d}_{entry['robot']}"
            spec_path = cdir / f"{name}.spec.json"
            spec_path.write_text(json.dumps({
                "desc": desc, "entry": entry,
                "result_path": str(cdir / f"{name}.result.json")}))
            jobs.append(compile_sched.Job(
                name=name,
                argv=[PYTHON, str(REPO_ROOT / "test" / "run_split_suite.py"),
                      "--warm-one", str(spec_path)],
                ledger_key=f"warm:{desc}",
                log_path=str(cdir / f"{name}.log")))
            desc_to_job[desc] = name
        for mod in modules:
            names = []
            for e in WARM_MANIFEST.get(mod, []):
                jn = desc_to_job.get(_warm_desc(e))
                if jn:
                    names.append(jn)
            if names:
                prereqs[mod] = sorted(set(names))

    if cuda_shards and os.environ.get("GRID_SPLIT_PREWARM", "1") != "0":
        import prewarm_cuda_flagship as pw  # noqa: PLC0415
        flagship_ids = [t for s in cuda_shards for t in s.targets
                        if "test_cuda_executable_equivalence" in t]
        if flagship_ids:
            cache_env = {"GRID_CUDA_CACHE_DIR":
                         os.environ.get("GRID_CUDA_CACHE_DIR", CUDA_CACHE_DIR_DEFAULT)}
            ids_file = cdir / "flagship_ids.txt"
            ids_file.write_text("\n".join(flagship_ids))
            plan_path = cdir / "prewarm_plan.json"
            env = os.environ.copy()
            env.update(cache_env)
            prc = subprocess.run(
                [PYTHON, str(REPO_ROOT / "test" / "prewarm_cuda_flagship.py"),
                 "--plan", "--node-ids", str(ids_file), "--out", str(plan_path)],
                cwd=REPO_ROOT, env=env).returncode
            if prc != 0:
                print(f"  prewarm: --plan failed rc={prc}; cuda shards will "
                      f"compile inline (slower, still correct)", flush=True)
            else:
                plan = json.loads(plan_path.read_text())
                for jname in sorted(plan["jobs"]):
                    jobs.append(compile_sched.Job(
                        name=jname,
                        argv=[PYTHON,
                              str(REPO_ROOT / "test" / "prewarm_cuda_flagship.py"),
                              "--worker", str(plan_path), "--job", jname],
                        ledger_key=jname,
                        env=cache_env,
                        log_path=str(cdir / f"{jname}.log")))
                atom_to_job = plan.get("atom_to_job", {})
                for s in cuda_shards:
                    names = set()
                    for robot, base, cell in pw._parse_atoms(list(s.targets)):
                        jn = atom_to_job.get(f"{robot}|{base}|{cell}")
                        if jn:
                            names.add(jn)
                    if names:
                        prereqs[s.name] = sorted(
                            set(prereqs.get(s.name, [])) | names)

    if not jobs:
        return None, [], {}
    ledger = compile_sched.Ledger(
        REPO_ROOT / "test" / ".split_suite" / "compile_rss.json")
    pool = compile_sched.RamScheduler(ledger, max_jobs=max_jobs, label="pool")
    return pool, jobs, prereqs


def report_pool(pool, out_dir: Path) -> int:
    """Print pool outcomes; feed the peak-RSS ledger from REAL builds only
    (cache-HIT invocations must not shrink future predictions). Returns the
    count of failed compile jobs (informational — a failed pre-warm's shard
    still runs and reports properly via pytest)."""
    if pool is None:
        return 0
    cdir = out_dir / "compile"
    failed = 0
    for name, res in sorted(pool.results.items()):
        detail = ""
        rp = cdir / f"{name}.result.json"
        if rp.exists():  # wrapper warm job: result JSON carries HIT/BUILT
            try:
                r = json.loads(rp.read_text())
                detail = f"{r['status']} {r['detail']}"
                real_build = r["status"] == "BUILT"
            except (ValueError, KeyError):
                real_build = False
        else:  # prewarm job: a real build leaves the module's compile line in the log
            try:
                log_text = (cdir / f"{name}.log").read_text(errors="replace")
            except OSError:
                log_text = res.stdout_tail
            real_build = "compiling runner" in log_text or "cache miss" in log_text
        if res.rc not in (0,):
            failed += 1
            print(f"  POOL-FAIL {name}: rc={res.rc} {detail} "
                  f"(log: {cdir / (name + '.log')})")
        if real_build and res.peak_kb > 0:
            job_key = (f"warm:{json.loads((cdir / f'{name}.spec.json').read_text())['desc']}"
                       if (cdir / f"{name}.spec.json").exists() else name)
            pool.ledger.record(job_key, res.peak_kb)
    return failed


def phase_run(shards: list[ShardSpec], out_dir: Path, receipts: bool,
              extra_args: list[str],
              prior_results: list[dict],
              pool=None, prereqs: dict | None = None) -> tuple[list[dict], bool]:
    """Returns (results incl. prior clean rows, paused)."""
    results = list(prior_results)
    if pool is not None:
        # GPU shards need host RAM too — tighten the pool's live floor for the
        # rest of the run (admissions only; running compiles finish).
        pool.raise_floor(GPU_PHASE_FLOOR_KB)
    for spec in shards:
        if pool is not None and prereqs:
            need = [n for n in prereqs.get(spec.name, [])
                    if n in pool.done_events and not pool.done_events[n].is_set()]
            if need:
                print(f"[{datetime.now():%H:%M:%S}] {spec.name}: waiting on "
                      f"{len(need)} compile job(s) …", flush=True)
                pool.wait(need)
        if (out_dir / "PAUSE").exists():
            print(f"[{datetime.now():%H:%M:%S}] PAUSE file present — stopping "
                  f"cleanly before {spec.name} (rm it, then --resume {out_dir})",
                  flush=True)
            return results, True
        xml_path = out_dir / f"{spec.name}.xml"
        log_path = out_dir / f"{spec.name}.log"
        cmd = [PYTHON, "-m", "pytest", *spec.targets,
               "-q", "-rf", f"--junitxml={xml_path}"]
        if spec.apply_marker:
            cmd += ["-m", "gpu_proof"]
        cmd += extra_args
        if receipts:
            rdir = out_dir / "receipts"
            rdir.mkdir(exist_ok=True)
            cmd += ["--gpu-proof-enable",
                    f"--gpu-proof-out={rdir / (spec.name + '.json')}",
                    f"--gpu-proof-shard={spec.name}",
                    "--gpu-proof-shard-fingerprint-paths="
                    + ",".join(spec.fingerprint_paths)]
        env = os.environ.copy()
        if spec.domain == "cuda":
            # The cuda suite's artifact-cache default is CWD-relative — pin it
            # absolute so every shard (incl. nested pytest processes) shares
            # ONE warm cache. NOTE the cache writers have no file locking:
            # shards must stay SERIAL (they are — one GPU, one at a time).
            env.setdefault("GRID_CUDA_CACHE_DIR", CUDA_CACHE_DIR_DEFAULT)
        t0 = time.monotonic()
        with open(log_path, "w") as log:
            # start_new_session so a kill takes the WHOLE process group —
            # otherwise pytest dies but its nvcc/cicc grandchildren survive as
            # orphans and poison the next shard's run (bench-orchestration
            # trap: "pkill orphans, GPU/CPU EMPTY before the next leg").
            proc = subprocess.Popen(cmd, cwd=REPO_ROOT, stdout=log,
                                    stderr=subprocess.STDOUT,
                                    start_new_session=True, env=env)
            try:
                rc: int | str = _wait_progress_aware(proc, spec.name, log_path)
            except (KeyboardInterrupt, SystemExit):
                # In-flight shard = incomplete: kill its whole group and leave
                # it OUT of the ledger so --resume re-runs it from scratch.
                _kill_group(proc)
                raise
        dt = time.monotonic() - t0
        # VRAM watermark AFTER the shard's process exits: per-shard isolation
        # means memory.used should return to the desktop baseline every time.
        # A rising floor here = leaked device allocations surviving process
        # exit (the driver-wedge signature) — the accumulation-probe dataset.
        try:
            vram = subprocess.run(
                ["nvidia-smi", "--query-gpu=memory.used",
                 "--format=csv,noheader,nounits"],
                capture_output=True, text=True, timeout=15).stdout.strip().split("\n")[0]
        except Exception:
            vram = "?"
        junit = parse_junit(xml_path)
        if junit is None:
            kind = "CASUALTY" if rc != 0 else "NO-XML"
            row = dict(shard=spec.name, domain=spec.domain, rc=rc, secs=dt,
                       kind=kind, vram=vram, tests=0, failures=0, errors=0,
                       skipped=0, failed=[], fresh=True)
        else:
            t, f, e, s, failed = junit
            if rc == 0:
                kind = "OK"
            elif rc == 5 and t == 0 and f + e == 0:
                # pytest exit 5 = "no tests collected". For a wrappers module
                # that is a benign -k deselection (SCOPE narrowing). For a
                # cuda shard the targets are EXPLICIT node ids — all of them
                # vanishing means the partition is stale vs the tree (e.g. a
                # --resume across test edits): loud, never clean...unless the
                # caller really did pass a -k through the pytest passthrough.
                benign = spec.domain == "wrappers" or "-k" in extra_args
                kind = "DESELECTED" if benign else "STALE-IDS"
            elif f + e > 0:
                kind = "FAILURES"
            elif s > 0:
                # SUITE FLOOR: conftest forces rc=1 on env-guard skips on a
                # capable box; the XML itself is clean.
                kind = "FLOOR-SKIP"
            else:
                kind = "RC!=0"
            row = dict(shard=spec.name, domain=spec.domain, rc=rc, secs=dt,
                       kind=kind, vram=vram, tests=t, failures=f, errors=e,
                       skipped=s, failed=failed, fresh=True)
        results.append(row)
        _write_ledger(out_dir, results)
        est = (f", est {spec.est_secs / 60:.0f}min vs {dt / 60:.0f}min"
               if spec.domain == "cuda" and spec.est_secs else "")
        print(f"[{datetime.now():%H:%M:%S}] {spec.name}: {row['kind']} "
              f"({row['tests']} tests, {row['failures']}F/{row['errors']}E/"
              f"{row['skipped']}S, rc={rc}, {dt:.0f}s{est}, "
              f"vram={row['vram']}MiB)", flush=True)
    return results, False


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--modules", nargs="*", help="subset of wrapper module stems to run")
    ap.add_argument("--skip-warm", action="store_true")
    ap.add_argument("--receipts", action="store_true",
                    help="write per-shard schema-2 receipts and merge them "
                         "into <out>/gpu-proof.json at the end")
    ap.add_argument("--receipts-carry-from", default=None, metavar="RECEIPT",
                    help="with --receipts: carry still-valid shards from this "
                         "older merged receipt (gpu-proof merge --carry-from)")
    ap.add_argument("--changed-only", action="store_true",
                    help="skip WRAPPER modules whose input fingerprint (module "
                         "file + conftest + URDFs + codegen/template hashes + "
                         "toolchain) matches the last GREEN run in "
                         "test/.split-suite-state.json (cuda shards always run "
                         "— their composition is duration-dependent)")
    ap.add_argument("--all", action="store_true",
                    help="force a full run (explicitly overrides --changed-only)")
    ap.add_argument("--domains", default="wrappers",
                    help="comma list of test domains to run: wrappers,cuda "
                         "(default wrappers — byte-compatible with the "
                         "module-based driver)")
    ap.add_argument("--cuda-k", default=None, metavar="EXPR",
                    help="-k narrowing applied to the cuda COLLECTION (shards "
                         "then attest exactly the narrowed scope; wrapper "
                         "shards never take -k — see run_gpu_proof.sh)")
    ap.add_argument("--shard-budget-mins", type=float,
                    default=float(os.environ.get(
                        "GRID_SPLIT_SHARD_BUDGET_MINS", "120")),
                    help="target max ESTIMATED minutes per cuda shard "
                         "(default 120; a single atom over budget gets its "
                         "own shard)")
    ap.add_argument("--resume", default=None, metavar="OUTDIR",
                    help="resume an interrupted/paused run: reuse OUTDIR's "
                         "partition.json, skip shards its results.json records "
                         "as clean, re-run the rest, merge everything")
    ap.add_argument("--refresh-from", default=None, metavar="RECEIPT",
                    help="shard-level receipt refresh: re-run ONLY the shards "
                         "whose narrow fingerprints changed vs RECEIPT and "
                         "carry the rest (gpu-proof merge --carry-from; the "
                         "everyday policy allows carried shards, the release "
                         "policy refuses them). Implies --receipts, both "
                         "domains, full scope.")
    ap.add_argument("--out", default=None, help="output dir (default test/.split_suite/<stamp>)")
    ap.add_argument("--prune-receipts", type=int, default=None, metavar="N",
                    help="delete all but the newest N timestamped run dirs under "
                         "test/.split_suite/ (receipt_* and bare <stamp> dirs; the "
                         "rolling durations/RSS ledgers are never touched), then exit")
    ap.add_argument("--warm-one", default=None, metavar="SPEC_JSON",
                    help=argparse.SUPPRESS)  # internal pool-worker mode
    ap.add_argument("pytest_args", nargs="*", default=[],
                    help="extra args passed to every pytest invocation (after --)")
    args = ap.parse_args()

    if args.warm_one:
        return warm_one_worker(args.warm_one)

    if args.prune_receipts is not None:
        # Retention housekeeping (N3, 2026-09-09): run dirs are append-only
        # otherwise and quietly accumulate. Keep the newest N; a dir named
        # in a live SPLIT_RESUME should simply not be pruned mid-run — this
        # is an explicit, standalone invocation, not part of a pass.
        import shutil as _shutil
        root = Path(__file__).resolve().parent / ".split_suite"
        runs = sorted(p for p in root.iterdir()
                      if p.is_dir() and (p.name.startswith("receipt_")
                                         or p.name[:8].isdigit()))
        doomed = runs[:-args.prune_receipts] if args.prune_receipts > 0 else runs
        for p in doomed:
            print(f"prune: {p}")
            _shutil.rmtree(p)
        print(f"kept {len(runs) - len(doomed)} run dir(s), pruned {len(doomed)}")
        return 0

    domains = [d.strip() for d in args.domains.split(",") if d.strip()]
    bad_domains = [d for d in domains if d not in ("wrappers", "cuda")]
    if bad_domains:
        print(f"FATAL: unknown domain(s): {bad_domains}", file=sys.stderr)
        return 2
    run_wrappers = "wrappers" in domains
    run_cuda = "cuda" in domains

    if args.refresh_from:
        # Refresh is a FULL-receipt operation: any narrowing would make the
        # fresh side attest less than the shards it replaces.
        if args.cuda_k or args.modules or args.changed_only or \
                args.receipts_carry_from:
            print("FATAL: --refresh-from is incompatible with --cuda-k / "
                  "--modules / --changed-only / --receipts-carry-from",
                  file=sys.stderr)
            return 2
        if not Path(args.refresh_from).exists():
            print(f"FATAL: --refresh-from {args.refresh_from}: no such receipt",
                  file=sys.stderr)
            return 2
        args.receipts = True
        args.receipts_carry_from = args.refresh_from
        run_wrappers = run_cuda = True

    if run_cuda and "random" in os.environ.get("GRID_CUDA_THREAD_COUNTS", ""):
        print("FATAL: GRID_CUDA_THREAD_COUNTS contains 'random' — node ids "
              "would be nondeterministic and the cuda partition unsound",
              file=sys.stderr)
        return 2

    if args.resume:
        out_dir = Path(args.resume)
        if not (out_dir / "results.json").exists() and \
                not (out_dir / "partition.json").exists():
            print(f"FATAL: --resume {out_dir}: no results.json/partition.json "
                  f"to resume from", file=sys.stderr)
            return 2
    else:
        out_dir = Path(args.out) if args.out else (
            REPO_ROOT / "test" / ".split_suite"
            / datetime.now().strftime("%Y%m%d_%H%M%S"))
    out_dir.mkdir(parents=True, exist_ok=True)
    # A stale PAUSE sentinel must not instantly stop the (re)start.
    (out_dir / "PAUSE").unlink(missing_ok=True)

    # Resuming a refresh run: reuse the SAVED plan verbatim (re-planning could
    # disagree with the persisted partition if the tree moved; receipts refuse
    # cross-commit merges anyway, so the saved plan is the only sound one).
    refresh_plan_path = out_dir / "refresh_plan.json"
    refresh_resume_mods: list[str] | None = None
    if args.resume and refresh_plan_path.exists() and not args.refresh_from:
        _rp = json.loads(refresh_plan_path.read_text())
        args.refresh_from = _rp["refresh_from"]
        args.receipts = True
        args.receipts_carry_from = _rp["refresh_from"]
        refresh_resume_mods = list(_rp["wrapper_mods"])
        run_wrappers = run_cuda = True

    refresh_cuda_shards: list[ShardSpec] | None = None
    if args.refresh_from:
        if refresh_resume_mods is not None:
            modules = refresh_resume_mods
            refresh_cuda_shards = load_partition(out_dir) or []
            print(f"=== refresh resume: {len(modules)} wrapper module(s) + "
                  f"{len(refresh_cuda_shards)} cuda shard(s) from the saved plan ===")
        else:
            from pytest_gpu_proof.fingerprint import compute_fingerprint  # noqa: PLC0415
            old = json.loads(Path(args.refresh_from).read_text())
            stale_names, carried_names, modules, refresh_cuda_shards = \
                plan_refresh(
                    old, collect_cuda_node_ids(None), discover_modules(),
                    load_durations(), args.shard_budget_mins * 60.0,
                    lambda paths: compute_fingerprint(
                        paths, root=str(REPO_ROOT))["digest"])
            print(f"=== refresh vs {args.refresh_from}: {len(stale_names)} "
                  f"stale / {len(carried_names)} carried shard(s); re-running "
                  f"{len(modules)} wrapper module(s) + "
                  f"{len(refresh_cuda_shards)} fresh cuda shard(s) ===")
            for n in stale_names:
                print(f"  STALE {n}")
            if not modules and not refresh_cuda_shards:
                print("nothing stale — the receipt already covers this tree")
                return 0
            if os.environ.get("GRID_SPLIT_REFRESH_DRY") == "1":
                print("GRID_SPLIT_REFRESH_DRY=1 — plan printed, not running")
                return 0
            refresh_plan_path.write_text(json.dumps(
                {"refresh_from": args.refresh_from, "wrapper_mods": modules,
                 "carried": carried_names, "stale": stale_names}, indent=1))
    else:
        modules = (args.modules or discover_modules()) if run_wrappers else []
    unknown = [m for m in (args.modules or []) if not (WRAPPERS_DIR / f"{m}.py").exists()]
    if unknown:
        print(f"FATAL: unknown module(s): {unknown}", file=sys.stderr)
        return 2

    fingerprints = {}
    skipped_unchanged: list[str] = []
    if run_wrappers and args.changed_only and not args.all:
        state = load_state()
        for mod in modules:
            fingerprints[mod] = module_fingerprint(mod)
        skipped_unchanged = [m for m in modules
                             if state.get(m) == fingerprints[m]]
        modules = [m for m in modules if m not in skipped_unchanged]
        print(f"=== --changed-only: {len(skipped_unchanged)} unchanged module(s) "
              f"skipped, {len(modules)} to run ===")
        for m in skipped_unchanged:
            print(f"  UNCHANGED {m}")
        if not modules and not run_cuda and \
                not (args.receipts and args.receipts_carry_from):
            print("nothing to run — all module fingerprints match the last green run")
            return 0

    # RAM-aware parallel compile pool (2026-08-18): Phase A wrapper warm +
    # cuda flagship pre-warm run as admission-controlled parallel workers that
    # OVERLAP GPU shard execution — a shard waits only for its own compile
    # jobs. GRID_SPLIT_COMPILE_JOBS=0 restores the serial inline path.
    pool_jobs_n = int(os.environ.get("GRID_SPLIT_COMPILE_JOBS", "5") or "0")

    specs: list[ShardSpec] = []
    if run_wrappers:
        for w in drift_guard(modules):
            print(f"WARM-MANIFEST DRIFT: {w}")
        if not args.skip_warm and pool_jobs_n == 0:
            print(f"=== Phase A: compile-warm ({len(WARM_MANIFEST)} modules' robots, serial) ===")
            for desc, status, secs, detail in phase_warm(out_dir):
                print(f"  {status:12s} {secs:7.1f}s  {desc}  {detail}")
        # Wrapper shards fingerprint the BINDINGS SOURCE alongside their test
        # module (user decision 2026-09-09): the wrapper tests import
        # bindings/grid_rbd, so a bindings-only change must stale them — before
        # this, an entire bindings refactor shipped under carry because only
        # test-file edits were fingerprinted. Deliberately NOT extended to the
        # cuda domain (grid_codegen inputs there would turn every codegen edit
        # into a multi-day full-domain refresh; that class stays covered by the
        # byte-gates + equivalence smokes + release-policy full passes).
        _BINDINGS_FP = ["bindings/grid_rbd", "bindings/src"]
        specs += [ShardSpec(name=mod, domain="wrappers",
                            targets=[f"test/python_wrappers/{mod}.py"],
                            fingerprint_paths=[f"test/python_wrappers/{mod}.py"]
                            + _BINDINGS_FP)
                  for mod in modules]

    if run_cuda:
        if refresh_cuda_shards is not None:
            cuda_shards = refresh_cuda_shards
            save_partition(out_dir, cuda_shards)
            for s in cuda_shards:
                print(f"  {s.name:42s} {len(s.targets):4d} ids  "
                      f"est {s.est_secs / 60:6.1f} min")
            specs += cuda_shards
        else:
            cuda_shards = load_partition(out_dir) if args.resume else None
            if cuda_shards is None:
                budget = args.shard_budget_mins * 60.0
                cuda_shards, cuda_ids = build_cuda_shards(args.cuda_k, budget)
                save_partition(out_dir, cuda_shards)
                print(f"=== cuda partition: {len(cuda_ids)} node ids -> "
                      f"{len(cuda_shards)} shard(s) "
                      f"(budget {args.shard_budget_mins:.0f} min) ===")
                for s in cuda_shards:
                    over = ("  <-- over budget (single unsplittable atom)"
                            if s.est_secs > budget else "")
                    print(f"  {s.name:42s} {len(s.targets):4d} ids  "
                          f"est {s.est_secs / 60:6.1f} min{over}")
            else:
                print(f"=== cuda partition: reusing {len(cuda_shards)} shard(s) "
                      f"from {out_dir / 'partition.json'} ===")
            specs += cuda_shards

    prior: list[dict] = []
    if args.resume:
        prior = load_prior_results(out_dir, {s.name for s in specs})
        if prior:
            print(f"=== --resume: {len(prior)} clean shard(s) kept, "
                  f"{len(specs) - len(prior)} to run ===")
    prior_clean = {r["shard"] for r in prior}
    to_run = [s for s in specs if s.name not in prior_clean]

    pool = None
    prereqs: dict[str, list[str]] = {}
    if pool_jobs_n > 0:
        cuda_specs = [s for s in to_run if s.domain == "cuda"]
        pool, pool_jobs, prereqs = build_compile_pool(
            out_dir,
            warm=run_wrappers and not args.skip_warm,
            cuda_shards=cuda_specs,
            modules=[s.name for s in to_run if s.domain == "wrappers"],
            max_jobs=pool_jobs_n)
        if pool is not None:
            print(f"=== compile pool: {len(pool_jobs)} job(s), "
                  f"{pool_jobs_n} slots, budget "
                  f"{pool.budget_kb // (1024 * 1024)} GiB "
                  f"(overlaps GPU shard execution) ===")
            pool.run_async(pool_jobs)

    import signal

    def _sigterm(_sig, _frm):
        raise KeyboardInterrupt

    signal.signal(signal.SIGTERM, _sigterm)

    print(f"=== Phase B: per-shard runs -> {out_dir} ===")
    try:
        results, paused = phase_run(to_run, out_dir, args.receipts,
                                    args.pytest_args, prior,
                                    pool=pool, prereqs=prereqs)
    except KeyboardInterrupt:
        if pool is not None:
            pool.kill_all()
        print(f"\nINTERRUPTED — completed shards are in {out_dir}/results.json; "
              f"continue with --resume {out_dir}", flush=True)
        return 130
    finally:
        # Normal completion: every job was some shard's prereq, so the pool is
        # drained. PAUSE / resume-with-clean-shards can leave queued or running
        # workers — kill them (their caches re-warm on the next run; PAUSE must
        # actually free the box, and orphaned compilers must never outlive us).
        if pool is not None:
            pool.kill_all()

    if pool is not None:
        pool_failed = report_pool(pool, out_dir)
        if pool_failed:
            print(f"  compile pool: {pool_failed} job(s) failed — affected "
                  f"shards compiled inline or reported the failure themselves")

    print("\n=== SPLIT SUITE SUMMARY ===")
    tot = dict(tests=0, failures=0, errors=0, skipped=0)
    bad = 0
    for r in results:
        for k in tot:
            tot[k] += r[k]
        clean = r["kind"] in CLEAN_KINDS
        flag = "" if clean else "  <-- "
        if not clean:
            bad += 1
        carried = "" if r.get("fresh", True) else " (prior run)"
        print(f"  {r['shard']:42s} {r['kind']:10s} rc={r['rc']!s:>7} "
              f"{r['tests']:4d}T {r['failures']}F {r['errors']}E "
              f"{r['skipped']}S{flag}{carried}")
        for name in r["failed"]:
            print(f"      FAILED {name}")
    print(f"\n  TOTAL: {tot['tests']} tests, {tot['failures']}F "
          f"{tot['errors']}E {tot['skipped']}S across {len(results)} shards"
          + (f" (+{len(skipped_unchanged)} unchanged-skipped)" if skipped_unchanged else "")
          + f"; {bad} shard(s) not clean")

    # Update the green-state fingerprints for WRAPPER modules that ended clean
    # THIS run (--changed-only skip data; never for failures, never for prior
    # rows — their fingerprints belong to the run that produced them).
    state = load_state()
    for r in results:
        if r["kind"] == "OK" and r.get("domain") == "wrappers" \
                and r.get("fresh", True):
            mod = r["shard"]
            state[mod] = fingerprints.get(mod) or module_fingerprint(mod)
    save_state(state)

    if paused:
        print(f"PAUSED — continue with --resume {out_dir} "
              f"(receipts merge deferred to the completing run)")
        return 3

    # Merge shard receipts into one verifiable artifact. On --resume this
    # re-globs receipts/, folding prior clean shards' receipts back in;
    # cross-commit staleness is machine-guarded (shards disagreeing on the
    # commit SHA refuse to merge).
    if args.receipts:
        rdir = out_dir / "receipts"
        shards = sorted(str(p) for p in rdir.glob("*.json")) if rdir.exists() else []
        if shards:
            merged = out_dir / "gpu-proof.json"
            cmd = [str(REPO_ROOT / ".venv" / "bin" / "gpu-proof"), "merge",
                   "--out", str(merged), "--repo", str(REPO_ROOT)]
            if args.receipts_carry_from:
                cmd += ["--carry-from", args.receipts_carry_from]
            cmd += shards
            mrc = subprocess.run(cmd, cwd=REPO_ROOT).returncode
            print(f"  receipts: merged {len(shards)} shard(s) -> {merged} (rc={mrc})"
                  + (f" carrying from {args.receipts_carry_from}"
                     if args.receipts_carry_from else ""))
            if mrc != 0:
                bad += 1
            elif bad == 0:
                update_durations(merged)
        else:
            print("  receipts: no shard receipts were produced")

    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
