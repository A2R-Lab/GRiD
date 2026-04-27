import importlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional


@dataclass(frozen=True)
class RobotSpec:
    robot_id: str
    tier: str
    embodiment: str
    source_kind: str
    description_name: str
    base_modes: List[str]
    preferred_variant: str
    notes: str


@dataclass(frozen=True)
class ResolvedRobotModel:
    robot_id: str
    source_kind: str
    description_name: str
    urdf_path: str
    package_path: Optional[str]
    repository_path: Optional[str]
    repository_url: Optional[str]
    revision: Optional[str]
    notes: str


def load_manifest(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def select_robot_specs(manifest: Dict[str, Any], tier: Optional[str] = None) -> List[RobotSpec]:
    resolved_tier = tier or manifest["default_tier"]
    specs = []
    for robot in manifest["robots"]:
        if robot["tier"] != resolved_tier:
            continue
        specs.append(
            RobotSpec(
                robot_id=robot["robot_id"],
                tier=robot["tier"],
                embodiment=robot["embodiment"],
                source_kind=robot["source_kind"],
                description_name=robot["description_name"],
                base_modes=list(robot["base_modes"]),
                preferred_variant=robot.get("preferred_variant", "default"),
                notes=robot.get("notes", ""),
            )
        )
    return specs


def resolve_robot_descriptions(spec: RobotSpec) -> ResolvedRobotModel:
    try:
        module = importlib.import_module(spec.description_name)
    except ModuleNotFoundError:
        module = importlib.import_module(f"robot_descriptions.{spec.description_name}")
    urdf_path = getattr(module, "URDF_PATH", None)
    if not urdf_path:
        raise RuntimeError(
            f"{spec.description_name} did not expose URDF_PATH for {spec.robot_id}"
        )

    package_path = getattr(module, "PACKAGE_PATH", None)
    repository_path = getattr(module, "REPOSITORY_PATH", None)
    repository_url = getattr(module, "REPOSITORY_URL", None)
    revision = getattr(module, "COMMIT", None) or getattr(module, "REVISION", None)

    return ResolvedRobotModel(
        robot_id=spec.robot_id,
        source_kind=spec.source_kind,
        description_name=spec.description_name,
        urdf_path=str(urdf_path),
        package_path=str(package_path) if package_path else None,
        repository_path=str(repository_path) if repository_path else None,
        repository_url=str(repository_url) if repository_url else None,
        revision=str(revision) if revision else None,
        notes=spec.notes,
    )


def resolve_robot_spec(spec: RobotSpec) -> ResolvedRobotModel:
    if spec.source_kind == "robot_descriptions":
        return resolve_robot_descriptions(spec)
    if spec.source_kind == "example_robot_data":
        raise NotImplementedError(
            "example_robot_data resolution is planned but not used in the smoke manifest"
        )
    if spec.source_kind == "direct_git":
        raise NotImplementedError(
            "direct_git resolution is planned but not used in the smoke manifest"
        )
    if spec.source_kind == "local_path":
        raise NotImplementedError(
            "local_path resolution is reserved for debugging and is not part of the smoke manifest"
        )
    raise ValueError(f"Unsupported source_kind: {spec.source_kind}")


def iter_robot_cases(
    manifest_path: Path, tier: Optional[str] = None, base_mode: Optional[str] = None
) -> Iterable[Dict[str, Any]]:
    manifest = load_manifest(manifest_path)
    for spec in select_robot_specs(manifest, tier=tier):
        for mode in spec.base_modes:
            if base_mode and mode != base_mode:
                continue
            yield {"spec": spec, "base_mode": mode}
