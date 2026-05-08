import ast
import contextlib
import os
import re
import shutil
import subprocess
from pathlib import Path

import pytest

from GRiDCodeGenerator import GRiDCodeGenerator
from GRiDCodeGenerator.algorithms._idsva_so import (
    idsva_so_parent_topology_needs_reference_order_output_repair,
)
from test.cuda_equivalents.test_cuda_executable_equivalence import _detect_cuda_arch
from test.pinocchio_equivalents.conftest import MANIFEST_PATH
from test.pinocchio_equivalents.utils.model_sources import (
    iter_robot_cases,
    resolve_robot_spec,
)
from test.pinocchio_equivalents.utils.project_adapter import build_project_adapter


CONST_RE = re.compile(r"const int (?P<name>[A-Z0-9_]+) = (?P<value>-?[0-9]+);")

CODEGEN_ROOT = Path(__file__).resolve().parents[2] / "GRiDCodeGenerator"


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


def _codegen_for_robot(robot_id: str, base_mode: str):
    spec = _robot_spec(robot_id, base_mode)
    try:
        resolved = resolve_robot_spec(spec)
    except RuntimeError as exc:
        pytest.skip(
            f"Could not resolve manifest {spec.robot_id}. Run ./developer_install.sh "
            f"before executing CUDA codegen tests. Resolution error: {exc}"
        )
    project_model = build_project_adapter(spec, resolved, base_mode=base_mode)
    return GRiDCodeGenerator(
        project_model.robot,
        DEBUG_MODE=False,
        NEED_PRINT_MAT=False,
        FILE_NAMESPACE="grid",
    )


def _constants(header: str) -> dict[str, int]:
    return {match.group("name"): int(match.group("value")) for match in CONST_RE.finditer(header)}


def _ast_name_text(node) -> str:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        base = _ast_name_text(node.value)
        return f"{base}.{node.attr}" if base else node.attr
    if isinstance(node, ast.Subscript):
        return _ast_name_text(node.value)
    if isinstance(node, ast.Call):
        return _ast_name_text(node.func)
    return ""


def _compile_header_consumer(
    tmp_path: Path,
    header: str,
    source: str,
    label: str,
    *,
    expect_success: bool = True,
    expected_error: str | None = None,
):
    nvcc = shutil.which("nvcc")
    if nvcc is None:
        pytest.skip("nvcc was not found; install CUDA Toolkit to run CUDA compile-only tests.")

    build_dir = tmp_path / label
    build_dir.mkdir()
    header_path = build_dir / "grid.cuh"
    source_path = build_dir / f"{label}.cu"
    object_path = build_dir / f"{label}.o"
    header_path.write_text(header)
    source_path.write_text(source)
    arch = _detect_cuda_arch()
    cmd = [
        nvcc,
        "-std=c++11",
        "-c",
        "-gencode",
        f"arch=compute_{arch},code=sm_{arch}",
        "-gencode",
        f"arch=compute_{arch},code=compute_{arch}",
        "-o",
        str(object_path),
        str(source_path),
    ]
    result = subprocess.run(cmd, cwd=build_dir, capture_output=True, text=True)
    if expect_success and result.returncode != 0:
        pytest.fail(
            f"CUDA compile-only check failed for {label}.\n"
            f"Command: {' '.join(cmd)}\n"
            f"stdout:\n{result.stdout}\n"
            f"stderr:\n{result.stderr}"
        )
    if not expect_success:
        if result.returncode == 0:
            pytest.fail(
                f"CUDA compile-only check unexpectedly passed for {label}.\n"
                f"Command: {' '.join(cmd)}"
            )
        if expected_error is not None:
            combined = f"{result.stdout}\n{result.stderr}"
            if expected_error not in combined:
                pytest.fail(
                    f"CUDA compile-only check failed for {label}, but did not "
                    f"include expected error text {expected_error!r}.\n"
                    f"Command: {' '.join(cmd)}\n"
                    f"stdout:\n{result.stdout}\n"
                    f"stderr:\n{result.stderr}"
                )


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
@pytest.mark.parametrize(
    ("parent_ids", "needs_repair"),
    [
        ([-1, 0, 1, 2], False),
        ([-1, -1, -1], False),
        ([-1, 0, -1, 2, -1, 4], False),
        ([-1, 0, 0], True),
        ([-1, 0, 1, 1, 3], True),
    ],
    ids=[
        "serial_chain",
        "base_fanout",
        "base_rooted_independent_chains",
        "moving_joint_fanout",
        "deep_moving_joint_fanout",
    ],
)
def test_idsva_so_reference_order_repair_uses_parent_topology(parent_ids, needs_repair):
    assert (
        idsva_so_parent_topology_needs_reference_order_output_repair(parent_ids)
        is needs_repair
    )


@pytest.mark.cuda_equivalence
@pytest.mark.developer_only
@pytest.mark.parametrize(
    ("robot_id", "needs_repair"),
    [
        ("iiwa14", False),
        ("go2", False),
        ("gen3", False),
        ("rizon4", False),
        ("fr3", True),
        ("fetch", True),
    ],
    ids=lambda case: str(case),
)
def test_idsva_so_reference_order_repair_is_topology_gated(robot_id, needs_repair):
    codegen = _codegen_for_robot(robot_id, "fixed")

    assert codegen.idsva_so_needs_reference_order_output_repair() is needs_repair


@pytest.mark.cuda_equivalence
@pytest.mark.developer_only
def test_codegen_does_not_select_algorithms_by_fixture_name_or_filename():
    fixture_names = {
        "iiwa",
        "iiwa14",
        "go2",
        "g1",
        "fr3",
        "fetch",
        "rizon",
        "rizon4",
        "gen3",
        "baxter",
        "hyq",
    }
    fixture_patterns = [
        re.compile(r"(?<![A-Za-z0-9_])" + re.escape(name) + r"(?![A-Za-z0-9_])")
        for name in fixture_names
    ]
    filename_branch_names = {
        "robot_id",
        "robot_name",
        "urdf_name",
        "urdf_filename",
        "urdf_path",
        "filename",
        "file_name",
        "basename",
        "robot.name",
        "self.robot.name",
    }

    fixture_literals = []
    filename_conditionals = []
    for path in sorted(CODEGEN_ROOT.rglob("*.py")):
        source = path.read_text()
        tree = ast.parse(source, filename=str(path))
        relpath = path.relative_to(CODEGEN_ROOT)
        for node in ast.walk(tree):
            if isinstance(node, ast.Constant) and isinstance(node.value, str):
                value = node.value.lower()
                if any(pattern.search(value) for pattern in fixture_patterns):
                    fixture_literals.append((relpath, node.lineno, node.value))
            if isinstance(node, ast.If):
                test_nodes = list(ast.walk(node.test))
                names = {_ast_name_text(child).lower() for child in test_nodes}
                names = {name for name in names if name}
                if names & filename_branch_names:
                    filename_conditionals.append((relpath, node.lineno, sorted(names & filename_branch_names)))

    assert fixture_literals == []
    assert filename_conditionals == []


@pytest.mark.cuda_equivalence
@pytest.mark.developer_only
@pytest.mark.floating_base
@pytest.mark.parametrize("robot_id", ["iiwa14", "go2"], ids=lambda robot_id: f"{robot_id}-floating")
def test_floating_header_does_not_require_second_order_kernels(tmp_path, robot_id):
    header = _generate_header(tmp_path, robot_id, "floating")
    constants = _constants(header)

    assert "__shared__ T" not in header
    assert constants["GRID_GENERATES_IDSVA_SO"] == 0
    assert constants["GRID_GENERATES_FDSVA_SO"] == 0
    assert "!GRID_GENERATES_IDSVA_SO" in header
    assert "!GRID_GENERATES_FDSVA_SO" in header
    assert "void end_effector_pose(gridData<T, KIND> *hd_data" in header
    assert "void end_effector_pose_gradient(gridData<T, KIND> *hd_data" in header
    assert "void end_effector_pose_gradient_hessian(gridData<T, KIND> *hd_data" in header
    assert "void kinematics_only(gridData<T, KIND> *hd_data" in header
    assert "void aba(gridData<T, KIND> *hd_data" in header
    assert "void crba(gridData<T, KIND> *hd_data" in header


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
def test_dynamics_grid_data_variant_wrappers_compile(tmp_path):
    header = _generate_header(tmp_path, "fr3", "fixed", codegen_profile="dynamics")
    source = r'''
#include "grid.cuh"

int main() {
    using T = float;
    grid::gridData<T, grid::GRID_DATA_DYNAMICS> *data = nullptr;
    grid::robotModel<T> *model = nullptr;
    cudaStream_t *streams = nullptr;
    dim3 blocks(1, 1, 1);
    dim3 threads(32, 1, 1);
    grid::inverse_dynamics<T, false, false, grid::GRID_DATA_DYNAMICS>(
        data, model, static_cast<T>(9.81), 1, blocks, threads, streams);
    grid::rnea<T, false, false, grid::GRID_DATA_DYNAMICS>(
        data, model, static_cast<T>(9.81), 1, blocks, threads, streams);
    grid::direct_minv<T, false, grid::GRID_DATA_DYNAMICS>(
        data, model, 1, blocks, threads, streams);
    grid::forward_dynamics<T, grid::GRID_DATA_DYNAMICS>(
        data, model, static_cast<T>(9.81), 1, blocks, threads, streams);
    grid::inverse_dynamics_gradient<T, false, false, grid::GRID_DATA_DYNAMICS>(
        data, model, static_cast<T>(9.81), 1, blocks, threads, streams);
    grid::forward_dynamics_gradient<T, false, grid::GRID_DATA_DYNAMICS>(
        data, model, static_cast<T>(9.81), 1, blocks, threads, streams);
    grid::dynamics_only<T, grid::GRID_DATA_DYNAMICS>(
        data, model, static_cast<T>(9.81), 1, blocks, threads, streams);
    return 0;
}
'''
    _compile_header_consumer(tmp_path, header, source, "dynamics_grid_data_variant")


@pytest.mark.cuda_equivalence
@pytest.mark.developer_only
def test_kinematics_grid_data_variant_wrappers_compile(tmp_path):
    header = _generate_header(tmp_path, "fr3", "fixed", codegen_profile="kinematics")
    source = r'''
#include "grid.cuh"

int main() {
    using T = float;
    grid::gridData<T, grid::GRID_DATA_KINEMATICS> *data = nullptr;
    grid::robotModel<T> *model = nullptr;
    cudaStream_t *streams = nullptr;
    dim3 blocks(1, 1, 1);
    dim3 threads(32, 1, 1);
    grid::end_effector_pose<T, false, grid::GRID_DATA_KINEMATICS>(
        data, model, 1, blocks, threads, streams);
    grid::kinematics_only<T, grid::GRID_DATA_KINEMATICS>(
        data, model, 1, blocks, threads, streams);
    return 0;
}
'''
    _compile_header_consumer(tmp_path, header, source, "kinematics_grid_data_variant")


@pytest.mark.cuda_equivalence
@pytest.mark.developer_only
def test_fixed_kinematics_derivative_wrappers_compile(tmp_path):
    header = _generate_header(tmp_path, "fr3", "fixed", codegen_profile="kinematics-derivatives")
    source = r'''
#include "grid.cuh"

int main() {
    using T = float;
    grid::gridData<T, grid::GRID_DATA_KINEMATICS> *data = nullptr;
    grid::robotModel<T> *model = nullptr;
    cudaStream_t *streams = nullptr;
    dim3 blocks(1, 1, 1);
    dim3 threads(32, 1, 1);
    grid::end_effector_pose<T, false, grid::GRID_DATA_KINEMATICS>(
        data, model, 1, blocks, threads, streams);
    grid::end_effector_pose_gradient<T, false, grid::GRID_DATA_KINEMATICS>(
        data, model, 1, blocks, threads, streams);
    grid::end_effector_pose_gradient_hessian<T, false, grid::GRID_DATA_KINEMATICS>(
        data, model, 1, blocks, threads, streams);
    grid::kinematics_only<T, grid::GRID_DATA_KINEMATICS>(
        data, model, 1, blocks, threads, streams);
    return 0;
}
'''
    _compile_header_consumer(tmp_path, header, source, "fixed_kinematics_derivative_wrappers")


@pytest.mark.cuda_equivalence
@pytest.mark.developer_only
@pytest.mark.floating_base
def test_floating_kinematics_derivative_wrappers_compile(tmp_path):
    header = _generate_header(tmp_path, "iiwa14", "floating", codegen_profile="kinematics-derivatives")
    source = r'''
#include "grid.cuh"

int main() {
    using T = float;
    grid::gridData<T, grid::GRID_DATA_KINEMATICS> *data = nullptr;
    grid::robotModel<T> *model = nullptr;
    cudaStream_t *streams = nullptr;
    dim3 blocks(1, 1, 1);
    dim3 threads(32, 1, 1);
    grid::end_effector_pose<T, false, grid::GRID_DATA_KINEMATICS>(
        data, model, 1, blocks, threads, streams);
    grid::end_effector_pose_gradient<T, false, grid::GRID_DATA_KINEMATICS>(
        data, model, 1, blocks, threads, streams);
    grid::end_effector_pose_gradient_hessian<T, false, grid::GRID_DATA_KINEMATICS>(
        data, model, 1, blocks, threads, streams);
    grid::kinematics_only<T, grid::GRID_DATA_KINEMATICS>(
        data, model, 1, blocks, threads, streams);
    return 0;
}
'''
    _compile_header_consumer(tmp_path, header, source, "floating_kinematics_derivative_wrappers")


@pytest.mark.cuda_equivalence
@pytest.mark.developer_only
@pytest.mark.parametrize(
    ("label", "source", "expected_error"),
    [
        (
            "dynamics_api_rejects_kinematics_grid_data",
            r'''
#include "grid.cuh"

int main() {
    using T = float;
    grid::gridData<T, grid::GRID_DATA_KINEMATICS> *data = nullptr;
    grid::robotModel<T> *model = nullptr;
    cudaStream_t *streams = nullptr;
    dim3 blocks(1, 1, 1);
    dim3 threads(32, 1, 1);
    grid::inverse_dynamics<T, false, false, grid::GRID_DATA_KINEMATICS>(
        data, model, static_cast<T>(9.81), 1, blocks, threads, streams);
    return 0;
}
''',
            "inverse_dynamics requires all-data or dynamics gridData",
        ),
        (
            "kinematics_api_rejects_dynamics_grid_data",
            r'''
#include "grid.cuh"

int main() {
    using T = float;
    grid::gridData<T, grid::GRID_DATA_DYNAMICS> *data = nullptr;
    grid::robotModel<T> *model = nullptr;
    cudaStream_t *streams = nullptr;
    dim3 blocks(1, 1, 1);
    dim3 threads(32, 1, 1);
    grid::end_effector_pose<T, false, grid::GRID_DATA_DYNAMICS>(
        data, model, 1, blocks, threads, streams);
    return 0;
}
''',
            "end_effector_pose requires all-data or kinematics gridData",
        ),
    ],
)
def test_grid_data_variant_invalid_wrappers_fail_to_compile(
    tmp_path,
    label,
    source,
    expected_error,
):
    header = _generate_header(tmp_path, "fr3", "fixed")
    _compile_header_consumer(
        tmp_path,
        header,
        source,
        label,
        expect_success=False,
        expected_error=expected_error,
    )


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
