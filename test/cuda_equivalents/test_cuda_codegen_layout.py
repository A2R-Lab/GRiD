import contextlib
import os
import re
from pathlib import Path

import pytest

from GRiDCodeGenerator import GRiDCodeGenerator
from test.pinocchio_equivalents.conftest import MANIFEST_PATH
from test.pinocchio_equivalents.utils.model_sources import (
    iter_robot_cases,
    resolve_robot_spec,
)
from test.pinocchio_equivalents.utils.project_adapter import build_project_adapter


CONST_RE = re.compile(r"const int (?P<name>[A-Z0-9_]+) = (?P<value>-?[0-9]+);")


@contextlib.contextmanager
def _temporary_env(updates):
    previous = {key: os.environ.get(key) for key in updates}
    try:
        for key, value in updates.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = str(value)
        yield
    finally:
        for key, value in previous.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


def _robot_spec(robot_id: str, base_mode: str):
    for case in iter_robot_cases(MANIFEST_PATH, base_mode=base_mode):
        if case["spec"].robot_id == robot_id:
            return case["spec"]
    pytest.skip(f"{robot_id}-{base_mode} was not found in the robot manifest.")


def _generate_header(
    tmp_path: Path,
    robot_id: str,
    base_mode: str,
    target_shared_bytes=None,
    codegen_profile="all",
    algorithm_list=None,
) -> str:
    spec = _robot_spec(robot_id, base_mode)
    try:
        resolved = resolve_robot_spec(spec)
    except RuntimeError as exc:
        pytest.skip(
            f"Could not resolve manifest {spec.robot_id}. Run ./developer_install.sh "
            f"before executing CUDA codegen tests. Resolution error: {exc}"
        )
    project_model = build_project_adapter(spec, resolved, base_mode=base_mode)
    header_path = tmp_path / f"{robot_id}_{base_mode}_{target_shared_bytes or 'default'}.cuh"
    with _temporary_env({"GRID_CUDA_TARGET_SHARED_MEM_BYTES": target_shared_bytes}):
        codegen = GRiDCodeGenerator(
            project_model.robot,
            DEBUG_MODE=False,
            NEED_PRINT_MAT=False,
            FILE_NAMESPACE="grid",
        )
        with open(os.devnull, "w") as devnull, contextlib.redirect_stdout(devnull):
            codegen.gen_all_code(
                include_homogenous_transforms=base_mode == "fixed",
                codegen_profile=codegen_profile,
                algorithm_list=algorithm_list,
                output_path=str(header_path),
            )
    return header_path.read_text()


def _constants(header: str) -> dict[str, int]:
    return {match.group("name"): int(match.group("value")) for match in CONST_RE.finditer(header)}


@pytest.mark.cuda_equivalence
@pytest.mark.developer_only
def test_fixed_default_header_keeps_gradient_paths_all_shared(tmp_path):
    header = _generate_header(tmp_path, "fr3", "fixed")
    constants = _constants(header)

    assert "__shared__ T" not in header
    assert constants["GRID_ID_DU_USES_GLOBAL_TEMP"] == 0
    assert constants["GRID_FD_DU_USES_GLOBAL_TEMP"] == 0
    assert constants["GRID_ID_DU_USES_DA_DF_SPILL"] == 0
    assert constants["GRID_FD_DU_USES_DA_DF_SPILL"] == 0
    assert constants["GRID_GENERATES_IDSVA_SO"] == 1
    assert constants["GRID_GENERATES_FDSVA_SO"] == 1


@pytest.mark.cuda_equivalence
@pytest.mark.developer_only
def test_fixed_forced_low_shared_header_selects_fallbacks(tmp_path):
    header = _generate_header(tmp_path, "fr3", "fixed", target_shared_bytes=10000)
    constants = _constants(header)

    assert "__shared__ T" not in header
    assert constants["GRID_FD_DU_USES_DA_DF_SPILL"] == 1
    assert constants["GRID_IDSVA_SO_USES_GLOBAL_OUTPUT"] == 1
    assert constants["GRID_FDSVA_SO_USES_GLOBAL_TENSORS"] == 1
    assert constants["GRID_FDSVA_SO_USES_WORKSPACE_TEMP"] == 1
    assert "grid_begin_l2_persisting" in header
    assert "grid_end_l2_persisting" in header


@pytest.mark.cuda_equivalence
@pytest.mark.developer_only
@pytest.mark.floating_base
def test_floating_header_does_not_require_second_order_kernels(tmp_path):
    header = _generate_header(tmp_path, "iiwa14", "floating")
    constants = _constants(header)

    assert "__shared__ T" not in header
    assert constants["GRID_GENERATES_IDSVA_SO"] == 0
    assert constants["GRID_GENERATES_FDSVA_SO"] == 0
    assert "!GRID_GENERATES_IDSVA_SO" in header
    assert "!GRID_GENERATES_FDSVA_SO" in header
    assert "void kinematics_only(gridData<T, KIND> *hd_data" not in header
    assert "aba<T,KIND>(hd_data" not in header
    assert "crba<T,false,KIND>(hd_data" not in header


@pytest.mark.cuda_equivalence
@pytest.mark.developer_only
def test_generated_header_includes_grid_data_variants_and_rnea_aliases(tmp_path):
    header = _generate_header(tmp_path, "fr3", "fixed")

    assert "enum gridDataKind { GRID_DATA_ALL = 0, GRID_DATA_DYNAMICS = 1, GRID_DATA_KINEMATICS = 2 };" in header
    assert "template <typename T, gridDataKind KIND = GRID_DATA_ALL>" in header
    assert "gridData<T, KIND> *init_gridData" in header
    assert "void close_grid(cudaStream_t *streams, robotModel<T> *d_robotModel, gridData<T, KIND> *hd_data)" in header
    assert "void rnea(gridData<T, KIND> *hd_data" in header
    assert "void rnea_single_timing(gridData<T, KIND> *hd_data" in header
    assert "void rnea_compute_only(gridData<T, KIND> *hd_data" in header
    assert "inverse_dynamics<T,USE_QDD_FLAG,USE_COMPRESSED_MEM,KIND>" in header


@pytest.mark.cuda_equivalence
@pytest.mark.developer_only
def test_dynamics_core_profile_generates_only_core_dynamics_hosts(tmp_path):
    header = _generate_header(tmp_path, "fr3", "fixed", codegen_profile="dynamics-core")

    assert "Codegen profile: dynamics-core" in header
    assert "void inverse_dynamics(gridData<T, KIND> *hd_data" in header
    assert "void direct_minv(gridData<T, KIND> *hd_data" in header
    assert "void forward_dynamics(gridData<T, KIND> *hd_data" in header
    assert "void dynamics_core(gridData<T, KIND> *hd_data" in header
    assert "void id_minv_fd(gridData<T, KIND> *hd_data" in header
    assert "void inverse_dynamics_gradient(gridData<T, KIND> *hd_data" not in header
    assert "void forward_dynamics_gradient(gridData<T, KIND> *hd_data" not in header
    assert "void all_dynamics(gridData<T, KIND> *hd_data" not in header
    assert "void end_effector_pose(gridData<T, KIND> *hd_data" not in header
    assert "void idsva_so_host(gridData<T, KIND> *hd_data" not in header
    assert "void fdsva_so(gridData<T, KIND> *hd_data" not in header


@pytest.mark.cuda_equivalence
@pytest.mark.developer_only
def test_kinematics_profile_generates_kinematics_hosts_only(tmp_path):
    header = _generate_header(tmp_path, "fr3", "fixed", codegen_profile="kinematics")

    assert "Codegen profile: kinematics" in header
    assert "void end_effector_pose(gridData<T, KIND> *hd_data" in header
    assert "void kinematics_only(gridData<T, KIND> *hd_data" in header
    assert "static_assert(KIND == GRID_DATA_ALL || KIND == GRID_DATA_KINEMATICS" in header
    assert "void inverse_dynamics(gridData<T, KIND> *hd_data" not in header
    assert "void direct_minv(gridData<T, KIND> *hd_data" not in header
    assert "void forward_dynamics(gridData<T, KIND> *hd_data" not in header


@pytest.mark.cuda_equivalence
@pytest.mark.developer_only
def test_algorithm_list_override_expands_dependencies(tmp_path):
    header = _generate_header(tmp_path, "fr3", "fixed", algorithm_list="fd-gradient")

    assert "Generated algorithms:" in header
    assert "void inverse_dynamics(gridData<T, KIND> *hd_data" in header
    assert "void direct_minv(gridData<T, KIND> *hd_data" in header
    assert "void forward_dynamics(gridData<T, KIND> *hd_data" in header
    assert "void inverse_dynamics_gradient(gridData<T, KIND> *hd_data" in header
    assert "void forward_dynamics_gradient(gridData<T, KIND> *hd_data" in header
    assert "void dynamics_core(gridData<T, KIND> *hd_data" in header
    assert "void dynamics_gradients(gridData<T, KIND> *hd_data" in header
    assert "void all_dynamics(gridData<T, KIND> *hd_data" in header
    assert "void end_effector_pose(gridData<T, KIND> *hd_data" not in header
