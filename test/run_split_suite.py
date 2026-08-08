#!/usr/bin/env python3
"""Split-compile / split-run driver for the python_wrappers suite.

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
  .venv/bin/python test/run_split_suite.py                 # full split suite
  .venv/bin/python test/run_split_suite.py --modules test_tool test_iiwa14_smoke
  .venv/bin/python test/run_split_suite.py --skip-warm     # straight to Phase B
  .venv/bin/python test/run_split_suite.py --receipts      # + per-module gpu-proof
                                                           #   shard receipts

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
from datetime import datetime
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
WRAPPERS_DIR = REPO_ROOT / "test" / "python_wrappers"
PYTHON = str(REPO_ROOT / ".venv" / "bin" / "python")

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

# Per-module Phase-B timeout (s). Cold-cache g1 codegen+compile dominates; the
# force_rebuild modules recompile in-test every run — test_joint_dynamics does
# FOUR builds including two float64 ones (slowest compiles in the suite).
DEFAULT_TIMEOUT = 3600
TIMEOUTS = {
    "test_g1_plant_hessian_smoke": 7200,
    "test_joint_dynamics": 7200,
    "test_runtime_joint_dynamics": 7200,
}


def discover_modules() -> list[str]:
    mods = sorted(Path(p).stem for p in glob.glob(str(WRAPPERS_DIR / "test_*.py")))
    return mods


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
                "test_any_thread_count", "test_ee_named_target_floating_multileaf",
                "test_subset_build", "test_subset_build_jax", "test_subset_build_torch"):
            warnings.append(
                f"{mod}: {calls} register_robot site(s), none warmed and not "
                f"force_rebuild/exempt — extend WARM_MANIFEST?")
    return warnings


def phase_warm(out_dir: Path) -> list[tuple[str, str, float, str]]:
    """Returns rows (robot-desc, status, seconds, detail)."""
    sys.path.insert(0, str(REPO_ROOT))
    from config import robot_urdf  # noqa: PLC0415
    import grid_rbd  # noqa: PLC0415

    rows = []
    seen: set[str] = set()
    for mod, entries in sorted(WARM_MANIFEST.items()):
        for e in entries:
            kw = dict(e)
            robot = kw.pop("robot")
            desc = robot + "".join(
                f",{k}={v}" for k, v in sorted(kw.items()) if k != "max_batch_size"
            ) + f",mb={kw.get('max_batch_size', 256)}"
            if desc in seen:
                continue
            seen.add(desc)
            t0 = time.monotonic()
            try:
                urdf = robot_urdf(robot)
                if not urdf.exists():
                    rows.append((desc, "SKIP", 0.0, f"no urdf: {urdf}"))
                    continue
                key, so_path, _meta = grid_rbd.warm_robot(
                    name=f"__split_warm_{robot}", urdf_path=str(urdf), **kw)
                dt = time.monotonic() - t0
                status = "HIT" if dt < 5.0 else "BUILT"
                rows.append((desc, status, dt, key[:12]))
            except Exception as exc:  # a warm failure must never stop Phase B
                rows.append((desc, "COMPILE-FAIL", time.monotonic() - t0, repr(exc)[:200]))
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


def phase_run(modules: list[str], out_dir: Path, receipts: bool,
              extra_args: list[str]) -> list[dict]:
    results = []
    for mod in modules:
        xml_path = out_dir / f"{mod}.xml"
        log_path = out_dir / f"{mod}.log"
        cmd = [PYTHON, "-m", "pytest", f"test/python_wrappers/{mod}.py",
               "-q", "-rf", f"--junitxml={xml_path}"] + extra_args
        if receipts:
            rdir = out_dir / "receipts"
            rdir.mkdir(exist_ok=True)
            cmd += ["--gpu-proof-enable", f"--gpu-proof-out={rdir / (mod + '.json')}",
                    f"--gpu-proof-shard={mod}",
                    f"--gpu-proof-shard-fingerprint-paths=test/python_wrappers/{mod}.py"]
        t0 = time.monotonic()
        timeout = TIMEOUTS.get(mod, DEFAULT_TIMEOUT)
        with open(log_path, "w") as log:
            # start_new_session so a timeout kills the WHOLE process group —
            # otherwise pytest dies but its nvcc/cicc grandchildren survive as
            # orphans and poison the next module's run (bench-orchestration
            # trap: "pkill orphans, GPU/CPU EMPTY before the next leg").
            proc = subprocess.Popen(cmd, cwd=REPO_ROOT, stdout=log,
                                    stderr=subprocess.STDOUT,
                                    start_new_session=True)
            try:
                rc: int | str = proc.wait(timeout=timeout)
            except subprocess.TimeoutExpired:
                import signal as _signal
                try:
                    os.killpg(proc.pid, _signal.SIGKILL)
                except ProcessLookupError:
                    pass
                proc.wait()
                rc = "TIMEOUT"
        dt = time.monotonic() - t0
        junit = parse_junit(xml_path)
        if junit is None:
            kind = "CASUALTY" if rc != 0 else "NO-XML"
            results.append(dict(module=mod, rc=rc, secs=dt, kind=kind,
                                tests=0, failures=0, errors=0, skipped=0, failed=[]))
        else:
            t, f, e, s, failed = junit
            if rc == 0:
                kind = "OK"
            elif f + e > 0:
                kind = "FAILURES"
            elif s > 0:
                # SUITE FLOOR: conftest forces rc=1 on env-guard skips on a
                # capable box; the XML itself is clean.
                kind = "FLOOR-SKIP"
            else:
                kind = "RC!=0"
            results.append(dict(module=mod, rc=rc, secs=dt, kind=kind,
                                tests=t, failures=f, errors=e, skipped=s,
                                failed=failed))
        r = results[-1]
        print(f"[{datetime.now():%H:%M:%S}] {mod}: {r['kind']} "
              f"({r['tests']} tests, {r['failures']}F/{r['errors']}E/"
              f"{r['skipped']}S, rc={rc}, {dt:.0f}s)", flush=True)
    return results


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--modules", nargs="*", help="subset of module stems to run")
    ap.add_argument("--skip-warm", action="store_true")
    ap.add_argument("--receipts", action="store_true",
                    help="write per-module schema-2 shard receipts and merge them "
                         "into <out>/gpu-proof.json at the end")
    ap.add_argument("--receipts-carry-from", default=None, metavar="RECEIPT",
                    help="with --receipts: carry still-valid shards from this "
                         "older merged receipt (gpu-proof merge --carry-from)")
    ap.add_argument("--changed-only", action="store_true",
                    help="skip modules whose input fingerprint (module file + "
                         "conftest + URDFs + codegen/template hashes + toolchain) "
                         "matches the last GREEN run in test/.split-suite-state.json")
    ap.add_argument("--all", action="store_true",
                    help="force a full run (explicitly overrides --changed-only)")
    ap.add_argument("--out", default=None, help="output dir (default test/.split_suite/<stamp>)")
    ap.add_argument("pytest_args", nargs="*", default=[],
                    help="extra args passed to every pytest invocation (after --)")
    args = ap.parse_args()

    out_dir = Path(args.out) if args.out else (
        REPO_ROOT / "test" / ".split_suite" / datetime.now().strftime("%Y%m%d_%H%M%S"))
    out_dir.mkdir(parents=True, exist_ok=True)

    modules = args.modules or discover_modules()
    unknown = [m for m in (args.modules or []) if not (WRAPPERS_DIR / f"{m}.py").exists()]
    if unknown:
        print(f"FATAL: unknown module(s): {unknown}", file=sys.stderr)
        return 2

    fingerprints = {}
    skipped_unchanged: list[str] = []
    if args.changed_only and not args.all:
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
        if not modules and not (args.receipts and args.receipts_carry_from):
            print("nothing to run — all module fingerprints match the last green run")
            return 0

    for w in drift_guard(modules):
        print(f"WARM-MANIFEST DRIFT: {w}")

    if not args.skip_warm:
        print(f"=== Phase A: compile-warm ({len(WARM_MANIFEST)} modules' robots) ===")
        for desc, status, secs, detail in phase_warm(out_dir):
            print(f"  {status:12s} {secs:7.1f}s  {desc}  {detail}")

    print(f"=== Phase B: per-module runs -> {out_dir} ===")
    results = phase_run(modules, out_dir, args.receipts, args.pytest_args)

    print("\n=== SPLIT SUITE SUMMARY ===")
    tot = dict(tests=0, failures=0, errors=0, skipped=0)
    bad = 0
    for r in results:
        for k in tot:
            tot[k] += r[k]
        flag = "" if r["kind"] in ("OK", "FLOOR-SKIP") else "  <-- "
        if r["kind"] not in ("OK", "FLOOR-SKIP"):
            bad += 1
        print(f"  {r['module']:42s} {r['kind']:10s} rc={r['rc']!s:>7} "
              f"{r['tests']:4d}T {r['failures']}F {r['errors']}E {r['skipped']}S{flag}")
        for name in r["failed"]:
            print(f"      FAILED {name}")
    print(f"\n  TOTAL: {tot['tests']} tests, {tot['failures']}F "
          f"{tot['errors']}E {tot['skipped']}S across {len(results)} modules"
          + (f" (+{len(skipped_unchanged)} unchanged-skipped)" if skipped_unchanged else "")
          + f"; {bad} module(s) not clean")

    # Update the green-state fingerprints for modules that ended clean this run
    # (so --changed-only can skip them next time). Never recorded for failures.
    state = load_state()
    for r in results:
        if r["kind"] == "OK":
            mod = r["module"]
            state[mod] = fingerprints.get(mod) or module_fingerprint(mod)
    save_state(state)

    # Merge shard receipts into one verifiable artifact.
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
        else:
            print("  receipts: no shard receipts were produced")

    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
