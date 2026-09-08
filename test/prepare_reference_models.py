#!/usr/bin/env python3
import argparse
import json
import sys
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from RBDReference.tests.model_sources import (
    load_manifest,
    resolve_robot_spec,
    select_robot_specs,
)
from RBDReference.tests.source_lock import build_lock_entry
from RBDReference.tests import MANIFEST_PATH, SOURCE_LOCK_PATH


SUITE_ROOT = REPO_ROOT / "external" / "RBDReference" / "tests"
DEFAULT_TARGET = SUITE_ROOT / "test_all.py"


def list_tests(manifest_path: Path, tier: str | None) -> int:
    manifest = load_manifest(manifest_path)
    resolved_tier = tier or manifest["default_tier"]
    print(f"Manifest: {manifest_path}")
    print(f"Tier: {resolved_tier}")
    for robot in select_robot_specs(manifest, tier=resolved_tier):
        base_modes = ", ".join(robot.base_modes)
        source_chain = " -> ".join(
            f"{candidate.source_kind}:{candidate.description_name}"
            for candidate in robot.source_candidates
        )
        print(
            f"- {robot.robot_id}: {robot.embodiment} via "
            f"{source_chain}; base_modes={base_modes}"
        )
    return 0


def prepare_models(manifest_path: Path, tier: str | None, update_lock: bool) -> int:
    manifest = load_manifest(manifest_path)
    specs = select_robot_specs(manifest, tier=tier)
    entries = []
    failures = []
    for spec in specs:
        try:
            resolved = resolve_robot_spec(spec)
            entry = build_lock_entry(spec, resolved)
            print(
                f"resolved {spec.robot_id}: urdf={entry['resolved_urdf_path']} "
                f"package_root={entry['resolved_package_root']}"
            )
        except Exception as exc:
            resolved = None
            entry = build_lock_entry(spec, resolved, resolution_error=str(exc))
            failures.append((spec.robot_id, str(exc)))
            print(f"failed {spec.robot_id}: {exc}")
        entries.append(entry)

    generated_lock = {
        "schema_version": 1,
        "tier": tier or manifest["default_tier"],
        "generated_entries": entries,
    }

    generated_lock_path = REPO_ROOT / ".external_test_assets" / "robot_source_lock.generated.json"
    generated_lock_path.parent.mkdir(parents=True, exist_ok=True)
    generated_lock_path.write_text(
        json.dumps(generated_lock, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    if update_lock:
        checked_in_lock_path = SOURCE_LOCK_PATH
        checked_in_lock_path.write_text(
            json.dumps(generated_lock, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    return 1 if failures else 0


def run_pytest(target: Path, pytest_args: list[str]) -> int:
    args = [str(target)] + pytest_args
    return pytest.main(args)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run and inspect GRiD's test suites.",
    )
    parser.add_argument(
        "--list-tests",
        action="store_true",
        help="List the robots in the Pinocchio equivalence manifest.",
    )
    parser.add_argument(
        "--prepare-models",
        action="store_true",
        help="Resolve robot assets for the Pinocchio equivalence suite.",
    )
    parser.add_argument(
        "--update-lock",
        action="store_true",
        help="When preparing models, also refresh the checked-in lock file.",
    )
    parser.add_argument(
        "--tier",
        default=None,
        help="Optional tier override for listing or preparing models.",
    )
    parser.add_argument(
        "--target",
        default=str(DEFAULT_TARGET),
        help="Pytest target to run. Defaults to the Pinocchio equivalence test_all.py entrypoint.",
    )
    parser.add_argument(
        "pytest_args",
        nargs=argparse.REMAINDER,
        help="Additional arguments forwarded to pytest after '--'.",
    )
    args = parser.parse_args()

    manifest_path = MANIFEST_PATH
    if args.list_tests:
        return list_tests(manifest_path, args.tier)
    if args.prepare_models:
        return prepare_models(manifest_path, args.tier, args.update_lock)

    pytest_args = list(args.pytest_args)
    if pytest_args and pytest_args[0] == "--":
        pytest_args = pytest_args[1:]
    return run_pytest(Path(args.target), pytest_args)


if __name__ == "__main__":
    raise SystemExit(main())
