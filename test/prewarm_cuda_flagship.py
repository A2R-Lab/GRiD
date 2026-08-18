"""Pre-warm the cuda flagship module's header + runner-exe caches (2026-08-18).

The flagship module (test_cuda_executable_equivalence) is the only cached cuda
compile path; cold, its ~180 (robot, base, cell) header+exe compiles dominate a
full gpu-proof pass. This tool compiles them OUTSIDE pytest — in parallel under
test/compile_sched.py's RAM-aware scheduler — by importing the test module and
calling ITS OWN model/codegen/compile functions, so every cache key is equal by
construction (zero drift risk vs. what the tests will look up).

Two modes (driven by test/run_split_suite.py):

  --plan  --node-ids FILE --out PLAN.json
      Map flagship node ids -> deduped compile jobs. Grouping mirrors the
      tests' own cache topology: non-mimic robots get one job per
      (robot, base, cell) — each cell has its own subset header + exe — while
      MIMIC robots share ONE header across all cells (the per-cell list is
      nulled), so all their cells fold into a single serial job (the unlocked
      header-cache write must not race). Cells outside a base's algorithm
      selection and zero-inertia robots are dropped exactly like the tests
      skip them.

  --worker PLAN.json --job NAME
      Execute one job: for each cell, generate/fetch the header and compile
      the runner exe into the shared cache. Exit 0 even when every cell was a
      cache hit; nonzero only on a real failure.

Safety: workers only ever WRITE distinct cache dirs (distinct keys) except
within a mimic job, which is serial by construction. Test shards later only
READ these entries (or compile inline if a pre-warm job failed — the scheduler
protocol guarantees the pool has finished with a key before its shard starts,
so the unlocked cache is never written concurrently).
"""
from __future__ import annotations

import argparse
import contextlib
import json
import os
import re
import sys
import tempfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]

_FLAGSHIP_TESTS = (
    "test_fixed_base_generated_cuda_matches_python_reference",
    "test_floating_base_generated_cuda_matches_python_reference",
)
# node id param block: [<robot>-<base>-<cell_id>-threads<N>]
_PARAM_RE = re.compile(r"\[([^\]]+)\]$")


def _import_flagship():
    sys.path.insert(0, str(REPO_ROOT))
    from test.cuda_equivalents import test_cuda_executable_equivalence as mod  # noqa: PLC0415
    return mod


def _iter_specs(mod):
    from RBDReference.tests import MANIFEST_PATH  # noqa: PLC0415
    from RBDReference.tests.model_sources import iter_robot_cases  # noqa: PLC0415
    for base_mode in ("fixed", "floating"):
        for case in iter_robot_cases(MANIFEST_PATH, base_mode=base_mode):
            yield case["spec"], base_mode


def _parse_atoms(node_ids: list[str]) -> set[tuple[str, str, str]]:
    """(robot, base, cell_id) atoms from flagship node ids; ignores others."""
    atoms = set()
    for nid in node_ids:
        test_name = nid.split("::")[-1]
        if not any(test_name.startswith(t) for t in _FLAGSHIP_TESTS):
            continue
        m = _PARAM_RE.search(test_name)
        if not m:
            continue
        tokens = m.group(1).split("-")
        # robot ids contain no '-'; base is 'fixed'|'floating'; last token is
        # the threads param; the cell id is everything between.
        if len(tokens) < 4 or tokens[1] not in ("fixed", "floating"):
            continue
        robot, base = tokens[0], tokens[1]
        cell_id = "-".join(tokens[2:-1])
        atoms.add((robot, base, cell_id))
    return atoms


def build_plan(node_ids: list[str]) -> dict:
    mod = _import_flagship()
    from RBDReference.tests.model_sources import resolve_robot_spec  # noqa: PLC0415
    from RBDReference.equivalents.reference_backend import build_project_adapter  # noqa: PLC0415

    atoms = _parse_atoms(node_ids)
    cells_by_id = {c.cell_id: c for c in mod.FLAGSHIP_SPLIT_CELLS}
    specs = {}
    for spec, base_mode in _iter_specs(mod):
        specs[(spec.robot_id, base_mode)] = spec

    jobs: dict[str, dict] = {}
    atom_to_job: dict[str, str] = {}   # "robot|base|cell" -> job name
    dropped: list[str] = []
    model_cache: dict[tuple[str, str], tuple] = {}
    for robot, base, cell_id in sorted(atoms):
        spec = specs.get((robot, base))
        cell = cells_by_id.get(cell_id)
        if spec is None or cell is None:
            dropped.append(f"{robot}-{base}-{cell_id}: unknown spec/cell")
            continue
        selected = mod._flagship_selected_algorithms(base)
        if not any(a in selected for a in cell.compare_algorithms):
            continue  # the test skips this cell — nothing ever compiles
        if (robot, base) not in model_cache:
            resolved = resolve_robot_spec(spec)
            with open(os.devnull, "w") as dn, contextlib.redirect_stdout(dn):
                pm = build_project_adapter(spec, resolved, base_mode=base)
            degenerate = mod._model_inertia_is_degenerate(pm, pm.nv)
            mimic = mod._robot_has_mimic_joints(pm)
            model_cache[(robot, base)] = (degenerate, mimic)
        degenerate, mimic = model_cache[(robot, base)]
        if degenerate:
            continue  # the test skips pre-compile (zero-inertia asset)
        if mimic:
            # One shared header for ALL of a mimic robot's cells -> one serial job.
            name = f"prewarm_{robot}_{base}"
            job = jobs.setdefault(
                name,
                {"robot": robot, "base": base, "cells": [], "mimic": True},
            )
            job["cells"].append(cell_id)
        else:
            name = f"prewarm_{robot}_{base}_{cell_id}"
            jobs[name] = {"robot": robot, "base": base, "cells": [cell_id], "mimic": False}
        atom_to_job[f"{robot}|{base}|{cell_id}"] = name
    return {"jobs": jobs, "atom_to_job": atom_to_job, "dropped": dropped}


def run_job(plan: dict, name: str) -> int:
    import pytest  # noqa: PLC0415

    job = plan["jobs"][name]
    mod = _import_flagship()
    from RBDReference.tests.model_sources import resolve_robot_spec  # noqa: PLC0415
    from RBDReference.equivalents.reference_backend import build_project_adapter  # noqa: PLC0415

    robot, base = job["robot"], job["base"]
    spec = None
    for s, base_mode in _iter_specs(mod):
        if s.robot_id == robot and base_mode == base:
            spec = s
            break
    if spec is None:
        print(f"FATAL: no spec for {robot}-{base}")
        return 2
    resolved = resolve_robot_spec(spec)
    with open(os.devnull, "w") as dn, contextlib.redirect_stdout(dn):
        project_model = build_project_adapter(spec, resolved, base_mode=base)
    cells_by_id = {c.cell_id: c for c in mod.FLAGSHIP_SPLIT_CELLS}
    mimic = mod._robot_has_mimic_joints(project_model)

    failures = 0
    import time  # noqa: PLC0415
    for cell_id in job["cells"]:
        t_cell = time.monotonic()
        cell = cells_by_id[cell_id]
        codegen_list = list(cell.codegen_algorithm_list)
        subset, _ = mod._codegen_subset_from_env()
        if subset is not None:
            codegen_list = sorted(set(subset) | set(cell.codegen_algorithm_list))
        # Mirrors _run_cuda_equivalence_case: mimic robots null the per-cell
        # list so all cells share one forced-mimic header.
        if mimic:
            codegen_list = None
        build_dir = Path(tempfile.mkdtemp(prefix=f"prewarm_{robot}_{base}_"))
        try:
            _, header_key = mod._generate_grid_header(
                project_model, resolved, build_dir, None,
                codegen_algorithm_list=codegen_list,
            )
            mod._compile_runner(
                build_dir,
                floating_base=(base == "floating"),
                header_key=header_key,
                run_tokens=cell.run_tokens,
                skip_gradients=False,
                skip_eepose_gradients=False,
                config=None,
            )
            cell_secs = time.monotonic() - t_cell
            print(f"WARMED {robot}-{base}-{cell_id} header={header_key[:12]} "
                  f"({cell_secs:.0f}s)", flush=True)
        except pytest.skip.Exception as exc:
            print(f"SKIP {robot}-{base}-{cell_id}: {exc}", flush=True)
        except Exception as exc:  # a pre-warm failure must not sink the pool
            print(f"FAIL {robot}-{base}-{cell_id}: {exc!r}", flush=True)
            failures += 1
        finally:
            import shutil  # noqa: PLC0415
            shutil.rmtree(build_dir, ignore_errors=True)
    # (The driver decides real-build vs all-HIT from this log's "cache miss"/
    # "compiling runner" lines when feeding the peak-RSS ledger.)
    return 1 if failures else 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    mode = ap.add_mutually_exclusive_group(required=True)
    mode.add_argument("--plan", action="store_true")
    mode.add_argument("--worker", metavar="PLAN_JSON")
    ap.add_argument("--node-ids", metavar="FILE", help="plan: newline-separated node ids")
    ap.add_argument("--out", metavar="FILE", help="plan: output plan JSON")
    ap.add_argument("--job", metavar="NAME", help="worker: job name to run")
    args = ap.parse_args()

    if args.plan:
        node_ids = Path(args.node_ids).read_text().splitlines()
        plan = build_plan([n for n in node_ids if n.strip()])
        Path(args.out).write_text(json.dumps(plan, indent=1, sort_keys=True))
        print(f"plan: {len(plan['jobs'])} job(s), {len(plan['dropped'])} dropped")
        for d in plan["dropped"]:
            print(f"  dropped: {d}")
        return 0

    plan = json.loads(Path(args.worker).read_text())
    return run_job(plan, args.job)


if __name__ == "__main__":
    sys.exit(main())
