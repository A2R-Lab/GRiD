import hashlib
from pathlib import Path
from typing import Any, Dict, Optional


def sha256_file(path: Optional[str]) -> Optional[str]:
    if not path:
        return None
    file_path = Path(path)
    if not file_path.is_file():
        return None

    digest = hashlib.sha256()
    with file_path.open("rb") as handle:
        while True:
            chunk = handle.read(1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def build_lock_entry(spec, resolved: Optional[Any]) -> Dict[str, Any]:
    if resolved is None:
        return {
            "robot_id": spec.robot_id,
            "tier": spec.tier,
            "source_kind": spec.source_kind,
            "description_name": spec.description_name,
            "preferred_variant": spec.preferred_variant,
            "resolution_status": "unresolved",
            "resolved_urdf_path": None,
            "resolved_package_root": None,
            "resolved_repository_root": None,
            "upstream_repository_url": None,
            "upstream_revision": None,
            "urdf_sha256": None,
            "parse_status": {"fixed": "not_attempted", "floating": "not_attempted"},
            "notes": spec.notes,
        }

    return {
        "robot_id": spec.robot_id,
        "tier": spec.tier,
        "source_kind": spec.source_kind,
        "description_name": spec.description_name,
        "preferred_variant": spec.preferred_variant,
        "resolution_status": "resolved",
        "resolved_urdf_path": resolved.urdf_path,
        "resolved_package_root": resolved.package_path,
        "resolved_repository_root": resolved.repository_path,
        "upstream_repository_url": resolved.repository_url,
        "upstream_revision": resolved.revision,
        "urdf_sha256": sha256_file(resolved.urdf_path),
        "parse_status": {"fixed": "not_attempted", "floating": "not_attempted"},
        "notes": spec.notes,
    }
