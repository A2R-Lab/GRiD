"""Central config / asset path helpers for GRiD.

All robot URDFs live in ``config/robot_assets/`` and all baked launch-config
overrides in ``config/launch_configs/``. Import these helpers instead of
hard-coding the ``"config/robot_assets"`` string literal so that a future
relocation of the assets is a one-file change (this module) rather than a
sweep across every test / example / binding.

    from config import robot_urdf, ROBOT_ASSETS_DIR, LAUNCH_CONFIGS_DIR

    urdf = robot_urdf("iiwa14")     # -> <repo>/config/robot_assets/iiwa14.urdf

The paths are self-locating (relative to THIS file), so callers do not need to
compute their own ``parents[N]`` repo-root depth to reach the assets — they
only need the repo root on ``sys.path`` to ``import config``.

Note: ``GRiDCodeGenerator`` deliberately does NOT import this module for its own
launch-config lookup (``_launch_configs_dir``) — it resolves the directory
structurally to avoid a codegen-package -> top-level-package import edge. This
module is for the test / tooling / example readers.
"""
from __future__ import annotations

from pathlib import Path

CONFIG_DIR = Path(__file__).resolve().parent
REPO_ROOT = CONFIG_DIR.parent
ROBOT_ASSETS_DIR = CONFIG_DIR / "robot_assets"
LAUNCH_CONFIGS_DIR = CONFIG_DIR / "launch_configs"


def robot_urdf(robot_id: str) -> Path:
    """Path to ``config/robot_assets/<robot_id>.urdf``.

    Accepts a bare robot id (``"iiwa14"``) or a name already carrying the
    ``.urdf`` suffix. The returned path is NOT existence-checked — callers that
    want a graceful skip should test ``.exists()`` themselves (many URDFs are
    only vendored for robots absent from the ``robot_descriptions`` pip pkg).
    """
    name = robot_id if robot_id.endswith(".urdf") else f"{robot_id}.urdf"
    return ROBOT_ASSETS_DIR / name


__all__ = [
    "CONFIG_DIR",
    "REPO_ROOT",
    "ROBOT_ASSETS_DIR",
    "LAUNCH_CONFIGS_DIR",
    "robot_urdf",
]
