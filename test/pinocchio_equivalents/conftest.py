import importlib.util
from functools import lru_cache
from pathlib import Path

import pytest

from test.pinocchio_equivalents.utils.pinocchio_adapter import build_pinocchio_adapter
from test.pinocchio_equivalents.utils.project_adapter import (
    ProjectParseError,
    build_project_adapter,
)
from test.pinocchio_equivalents.utils.model_sources import (
    iter_robot_cases,
    load_manifest,
    resolve_robot_spec,
)
from test.pinocchio_equivalents.utils.source_lock import build_lock_entry


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SUITE_ROOT = Path(__file__).resolve().parent
MANIFEST_PATH = SUITE_ROOT / "robot_manifest.json"


def pytest_configure(config):
    config.addinivalue_line("markers", "pinocchio_equivalence")
    config.addinivalue_line("markers", "developer_only")
    config.addinivalue_line("markers", "floating_base")
    config.addinivalue_line("markers", "robot_smoke")
    config.addinivalue_line("markers", "robot_curated")
    config.addinivalue_line("markers", "robot_nightly")


def require_developer_dependencies():
    missing = []
    for module_name in ("pinocchio", "robot_descriptions"):
        if importlib.util.find_spec(module_name) is None:
            missing.append(module_name)
    if missing:
        pytest.fail(
            "Missing developer-only dependencies for the Pinocchio equivalence suite: "
            f"{', '.join(missing)}. Run ./developer_install.sh before executing these tests."
        )


def build_case_params(base_mode=None):
    params = []
    for case in iter_robot_cases(MANIFEST_PATH, base_mode=base_mode):
        spec = case["spec"]
        mode = case["base_mode"]
        marks = [
            pytest.mark.pinocchio_equivalence,
            pytest.mark.developer_only,
            getattr(pytest.mark, f"robot_{spec.tier}"),
        ]
        if mode == "floating":
            marks.append(pytest.mark.floating_base)
        params.append(
            pytest.param(
                spec,
                mode,
                id=f"{spec.robot_id}-{mode}",
                marks=marks,
            )
        )
    return params


@pytest.fixture(scope="session")
def developer_environment():
    require_developer_dependencies()
    return True


@pytest.fixture(scope="session")
def manifest():
    return load_manifest(MANIFEST_PATH)


@pytest.fixture
def resolved_robot_spec(developer_environment, spec):
    return resolve_robot_spec(spec)


@lru_cache(maxsize=None)
def _project_model_attempt(robot_id, embodiment, source_kind, urdf_path, base_mode):
    from test.pinocchio_equivalents.utils.model_sources import (
        ResolvedRobotModel,
        RobotSpec,
        SourceCandidate,
    )

    spec = RobotSpec(
        robot_id=robot_id,
        tier="smoke",
        embodiment=embodiment,
        source_kind=source_kind,
        description_name="",
        base_modes=[base_mode],
        preferred_variant="default",
        notes="",
        source_candidates=[SourceCandidate(source_kind=source_kind, description_name="")],
    )
    resolved = ResolvedRobotModel(
        robot_id=robot_id,
        source_kind=source_kind,
        description_name="",
        urdf_path=urdf_path,
        package_path=None,
        repository_path=None,
        repository_url=None,
        revision=None,
        notes="",
    )
    try:
        return build_project_adapter(spec, resolved, base_mode=base_mode), None
    except ProjectParseError as exc:
        return None, str(exc)


@pytest.fixture
def project_model_attempt(spec, base_mode, resolved_robot_spec):
    return _project_model_attempt(
        spec.robot_id,
        spec.embodiment,
        spec.source_kind,
        resolved_robot_spec.urdf_path,
        base_mode,
    )


@pytest.fixture
def project_model(spec, base_mode, resolved_robot_spec, project_model_attempt):
    model, error = project_model_attempt
    if error is not None:
        pytest.skip(
            "Skipping downstream comparison because GRiD strict parse already failed for "
            f"robot_id={spec.robot_id}, embodiment={spec.embodiment}, "
            f"source_kind={spec.source_kind}, urdf={resolved_robot_spec.urdf_path}, "
            f"base_mode={base_mode}: {error}"
        )
    return model


@pytest.fixture
def pinocchio_model(spec, base_mode, resolved_robot_spec):
    try:
        return build_pinocchio_adapter(spec, resolved_robot_spec, base_mode=base_mode)
    except Exception as exc:
        pytest.fail(
            "Pinocchio load failed for "
            f"robot_id={spec.robot_id}, embodiment={spec.embodiment}, "
            f"source_kind={spec.source_kind}, urdf={resolved_robot_spec.urdf_path}, "
            f"base_mode={base_mode}: {exc}"
        )


@pytest.fixture
def source_lock_entry(spec, resolved_robot_spec):
    return build_lock_entry(spec, resolved_robot_spec)
