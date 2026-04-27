#!/usr/bin/env python3
import argparse
import json
from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def load_manifest(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="List manifest-controlled Pinocchio equivalence robots."
    )
    parser.add_argument(
        "--manifest",
        default="test/pinocchio_equivalents/robot_manifest.json",
        help="Path to the manifest JSON file.",
    )
    parser.add_argument(
        "--tier",
        default=None,
        help="Optional tier filter. Defaults to the manifest default tier.",
    )
    args = parser.parse_args()

    manifest_path = Path(args.manifest).resolve()
    manifest = load_manifest(manifest_path)
    tier = args.tier or manifest["default_tier"]

    print(f"Manifest: {manifest_path}")
    print(f"Tier: {tier}")
    for robot in manifest["robots"]:
        if robot["tier"] != tier:
            continue
        base_modes = ", ".join(robot["base_modes"])
        print(
            f"- {robot['robot_id']}: {robot['embodiment']} via "
            f"{robot['source_kind']} ({robot['description_name']}); base_modes={base_modes}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
