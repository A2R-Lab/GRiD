"""Wave A' (P2/P3, 2026-09-11): replay recorded header content keys CPU-side.

test/gpu-proof-header-keys.json (aggregated by run_split_suite from the A4
sidecars) records, per cuda shard, every grid.cuh the shard generated: the
recipe evidence (flagship robot/base/algorithm-list, or a direct
gen_all_code call's bound kwargs + generator-ctor state + env snapshot) and
the emitted header's CONTENT sha256. At SPLIT_REFRESH time, when the
generator inputs changed vs the old receipt's commit, this module
regenerates each recorded header from the CURRENT tree and compares content
hashes:

  - every record of a shard reproduces byte-identically -> the shard's
    carried proof still covers the CURRENT tree's emitted code (the nvcc
    content keys would not even rotate) -> sound to carry;
  - any record's bytes rotate -> the shard is honestly stale;
  - a record that CANNOT be replayed (opaque kwargs with no HEADER_RECIPES
    entry, unknown robot, missing identity) -> the shard falls back to the
    caller's conservative path (the 6-row covering-matrix prover in
    codegen_neutrality, until every shard carries replayable records).

Replays run in-process under the record's env snapshot and use the harness
header cache where applicable, so a refresh pays seconds per unique cell.
"""
from __future__ import annotations

import contextlib
import hashlib
import json
import os
import sys
import tempfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

# ── P3: recipes for opaque kwargs ────────────────────────────────────────────
# A direct record whose `opaque` dict is non-empty replays ONLY if every
# opaque kwarg name has a builder here: builder(record) -> the kwarg value.
# Collision/multi-target/contact constructions are the known citizens; until
# their builders land, those cells stay conservatively matrix-/fingerprint-
# ruled (never silently carried). Keyed by kwarg name.
HEADER_RECIPES: dict = {}


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


@contextlib.contextmanager
def _apply_env(env: dict | None):
    if not env:
        yield
        return
    old = {k: os.environ.get(k) for k in env}
    try:
        for k, v in env.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v
        yield
    finally:
        for k, v in old.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


def _flagship_content(record: dict) -> tuple[str | None, str]:
    """Regenerate a flagship cell's header (robot_id x base_mode x optional
    subset list) through the harness path — header cache included, so an
    unchanged input key costs a file copy. Returns (sha or None, detail)."""
    if str(REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT))
    from RBDReference.tests import MANIFEST_PATH
    from RBDReference.tests.model_sources import (
        load_manifest, resolve_robot_spec, select_robot_specs)
    from RBDReference.equivalents.reference_backend import build_project_adapter
    from test.cuda_equivalents import cuda_harness

    specs = [s for s in select_robot_specs(load_manifest(MANIFEST_PATH))
             if s.robot_id == record.get("robot")]
    if not specs:
        return None, f"unknown flagship robot {record.get('robot')!r}"
    try:
        resolved = resolve_robot_spec(specs[0])
        with open(os.devnull, "w") as devnull, \
                contextlib.redirect_stdout(devnull):
            model = build_project_adapter(
                specs[0], resolved, base_mode=record.get("base_mode"))
            with tempfile.TemporaryDirectory(prefix="hk_replay_") as td:
                header, _key = cuda_harness._generate_grid_header(
                    model, resolved, Path(td), None,
                    codegen_algorithm_list=record.get("algorithm_list"))
                return _sha(header), "flagship regenerated"
    except Exception as exc:  # resolution/codegen failure = cannot replay
        return None, f"flagship replay failed: {exc!r}"


def _direct_content(record: dict) -> tuple[str | None, str]:
    """Regenerate a direct gen_all_code record from its bound kwargs +
    generator-ctor snapshot. Returns (sha or None, detail)."""
    opaque = record.get("opaque") or {}
    missing = [k for k in opaque if k not in HEADER_RECIPES]
    if missing:
        return None, f"opaque kwargs without recipes: {missing}"
    name = record.get("robot")
    if not name:
        return None, "record carries no robot identity"
    ctor = record.get("codegen") or {}
    if ctor.get("launch_config_robot"):
        return None, "launch_config_robot is opaque"
    if str(REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT))
    try:
        from config import robot_urdf
        urdf = Path(robot_urdf(name))
        if not urdf.exists():
            return None, f"no URDF for robot {name!r}"
        if record.get("urdf_sha256") and _sha(urdf) != record["urdf_sha256"]:
            # the URDF asset itself changed — a genuine codegen-input rotation
            return "__urdf_rotated__", "urdf bytes differ from record"
        from URDFParser import URDFParser
        from grid_codegen.GRiDCodeGenerator import GRiDCodeGenerator
        kwargs = dict(record.get("kwargs") or {})
        kwargs.pop("output_path", None)
        for k in opaque:
            kwargs[k] = HEADER_RECIPES[k](record)
        with open(os.devnull, "w") as devnull, \
                contextlib.redirect_stdout(devnull), \
                contextlib.redirect_stderr(devnull):
            robot = URDFParser().parse(
                str(urdf), floating_base=bool(record.get("floating")))
            gen = GRiDCodeGenerator(
                robot,
                DEBUG_MODE=ctor.get("DEBUG_MODE", False),
                NEED_PRINT_MAT=ctor.get("gen_print_mat", False),
                FILE_NAMESPACE=ctor.get("file_namespace", "grid"),
                USE_JOINT_DYNAMICS=ctor.get("USE_JOINT_DYNAMICS", False),
                MUJOCO_OUTPUT=ctor.get("MUJOCO_OUTPUT", False),
                LAUNCH_CONFIG_PROFILE=ctor.get("launch_config_profile", "host"),
                runtime_joint_dynamics=ctor.get("runtime_joint_dynamics", False),
                # fp64: reconstruct the ctor dtype from the recorded resolved
                # T byte size (old records lack the key -> 4 -> "float", which
                # matches how every pre-audit record was generated). When the
                # record's env carries GRID_CUDA_SHARED_MEM_TYPE_SIZE_BYTES,
                # _apply_env re-applies it and the env override wins, exactly
                # as it did at record time.
                dtype=("double" if int(ctor.get("t_bytes", 4)) == 8 else "float"),
            )
            with tempfile.TemporaryDirectory(prefix="hk_replay_") as td:
                out = Path(td) / "grid.cuh"
                gen.gen_all_code(output_path=str(out), **kwargs)
                return _sha(out), "direct regenerated"
    except Exception as exc:
        return None, f"direct replay failed: {exc!r}"


def replay_record(record: dict, cache: dict | None = None) -> tuple[bool | None, str]:
    """(True, why) = record reproduces byte-identically from the current tree;
    (False, why) = the emitted bytes rotated (honest staleness);
    (None, why)  = cannot replay (caller must stay conservative).
    ``cache`` (keyed by the record's identity WITHOUT its content hash) lets a
    refresh replay each unique cell once across shards."""
    want = record.get("content_sha256")
    if not want:
        return None, "record carries no content hash"
    ident = {k: v for k, v in record.items() if k != "content_sha256"}
    key = json.dumps(ident, sort_keys=True)
    if cache is not None and key in cache:
        got, detail = cache[key]
    else:
        with _apply_env(record.get("env")):
            if record.get("kind") == "flagship":
                got, detail = _flagship_content(record)
            elif record.get("kind") == "direct":
                got, detail = _direct_content(record)
            else:
                got, detail = None, f"unknown record kind {record.get('kind')!r}"
        if cache is not None:
            cache[key] = (got, detail)
    if got is None:
        return None, detail
    if got == "__urdf_rotated__":
        return False, detail
    return got == want, detail


def shard_replay_verdict(rows: list[dict], cache: dict | None = None
                         ) -> tuple[bool | None, str]:
    """Fold one shard's records: True = ALL reproduce (carry is sound);
    False = at least one rotated (stale); None = no rows, or some row could
    not be replayed and none proved rotation (fall back conservatively)."""
    if not rows:
        return None, "no header-key records for shard"
    unknown = 0
    for r in rows:
        ok, detail = replay_record(r, cache)
        if ok is False:
            return False, f"header bytes rotated: {detail} ({r.get('robot')})"
        if ok is None:
            unknown += 1
    if unknown:
        return None, f"{unknown}/{len(rows)} record(s) not replayable"
    return True, f"all {len(rows)} recorded header(s) byte-identical"
