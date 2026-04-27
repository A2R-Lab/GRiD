#!/usr/bin/env python3
import argparse
import json
from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from test.pinocchio_equivalents.model_sources import (
    load_manifest,
    resolve_robot_spec,
    select_robot_specs,
)
from test.pinocchio_equivalents.source_lock import build_lock_entry


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Resolve and optionally cache robot assets for the Pinocchio equivalence suite."
    )
    parser.add_argument(
        "--manifest",
        default="test/pinocchio_equivalents/robot_manifest.json",
        help="Path to the robot manifest JSON file.",
    )
    parser.add_argument(
        "--tier",
        default=None,
        help="Tier to resolve. Defaults to the manifest default tier.",
    )
    parser.add_argument(
        "--cache-only",
        action="store_true",
        help="Resolve and print robots without updating the checked-in lock file.",
    )
    parser.add_argument(
        "--generated-lock",
        default=".external_test_assets/robot_source_lock.generated.json",
        help="Path for the generated machine-readable lock output.",
    )
    parser.add_argument(
        "--checked-in-lock",
        default="test/pinocchio_equivalents/ROBOT_SOURCE_LOCK.json",
        help="Path to the checked-in lock file.",
    )
    parser.add_argument(
        "--update-checked-in-lock",
        action="store_true",
        help="Overwrite the checked-in lock file with the newly resolved results.",
    )
    args = parser.parse_args()

    manifest = load_manifest(Path(args.manifest))
    specs = select_robot_specs(manifest, tier=args.tier)
    lock_entries = []

    for spec in specs:
        resolved = resolve_robot_spec(spec)
        entry = build_lock_entry(spec, resolved)
        lock_entries.append(entry)
        print(
            f"resolved {spec.robot_id}: urdf={entry['resolved_urdf_path']} "
            f"package_root={entry['resolved_package_root']}"
        )

    generated_lock = {
        "schema_version": 1,
        "tier": args.tier or manifest["default_tier"],
        "generated_entries": lock_entries,
    }

    generated_lock_path = Path(args.generated_lock)
    generated_lock_path.parent.mkdir(parents=True, exist_ok=True)
    generated_lock_path.write_text(
        json.dumps(generated_lock, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    if args.update_checked_in_lock and not args.cache_only:
        checked_in_lock_path = Path(args.checked_in_lock)
        checked_in_lock_path.write_text(
            json.dumps(generated_lock, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
