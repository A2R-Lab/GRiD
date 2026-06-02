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
from RBDReference.tests import MANIFEST_PATH
from RBDReference.tests.model_sources import (
    iter_robot_cases,
    resolve_robot_spec,
)
from RBDReference.equivalents.reference_backend import build_project_adapter


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
    enable_floating_second_order=False,
    enable_idsva_so_world_frame=False,
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
                enable_floating_second_order=enable_floating_second_order,
                enable_idsva_so_world_frame=enable_idsva_so_world_frame,
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
    cxx_standard: str = "-std=c++11",
    extra_flags: list[str] | None = None,
    expect_success: bool = True,
    expected_error: str | None = None,
    expected_output: str | None = None,
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
        cxx_standard,
        "-c",
        "-gencode",
        f"arch=compute_{arch},code=sm_{arch}",
        "-gencode",
        f"arch=compute_{arch},code=compute_{arch}",
        "-o",
        str(object_path),
    ]
    if extra_flags:
        cmd.extend(extra_flags)
    cmd.append(str(source_path))
    result = subprocess.run(cmd, cwd=build_dir, capture_output=True, text=True)
    combined = f"{result.stdout}\n{result.stderr}"
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
            if expected_error not in combined:
                pytest.fail(
                    f"CUDA compile-only check failed for {label}, but did not "
                    f"include expected error text {expected_error!r}.\n"
                    f"Command: {' '.join(cmd)}\n"
                    f"stdout:\n{result.stdout}\n"
                    f"stderr:\n{result.stderr}"
                )
    if expected_output is not None and expected_output not in combined:
        pytest.fail(
            f"CUDA compile-only check for {label} did not include expected output "
            f"{expected_output!r}.\nCommand: {' '.join(cmd)}\n"
            f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
        )


@pytest.mark.cuda_equivalence
@pytest.mark.developer_only
def test_fixed_default_header_keeps_gradient_paths_all_shared(tmp_path):
    # iiwa14 (non-mimic fixed-base) exercises the full gradient/SO surface. fr3
    # is a MIMIC robot whose gradient codegen is now refused (G0 footgun guard;
    # mimic gradients deferred to T3-finisher), so it can no longer emit the
    # gradient spill-tier constants this test asserts.
    header = _generate_header(tmp_path, "iiwa14", "fixed")
    constants = _constants(header)

    assert "__shared__ T" not in header
    assert constants["GRID_INVERSE_DYNAMICS_GRADIENT_USES_GLOBAL_TEMP"] == 0
    assert constants["GRID_FORWARD_DYNAMICS_GRADIENT_USES_GLOBAL_TEMP"] == 0
    assert constants["GRID_INVERSE_DYNAMICS_GRADIENT_USES_DA_DF_SPILL"] == 0
    assert constants["GRID_FORWARD_DYNAMICS_GRADIENT_USES_DA_DF_SPILL"] == 0
    assert constants["GRID_GENERATES_IDSVA_SO_BODY_FRAME"] == 1
    assert constants["GRID_GENERATES_FDSVA_SO"] == 1


@pytest.mark.cuda_equivalence
@pytest.mark.developer_only
def test_fixed_forced_low_shared_header_selects_fallbacks(tmp_path):
    # iiwa14 (non-mimic) — fr3's gradient/SO codegen is refused under the G0
    # mimic-gradient guard, so it can't exercise the forced-low-shared fallbacks.
    header = _generate_header(tmp_path, "iiwa14", "fixed", target_shared_bytes=10000)
    constants = _constants(header)

    assert "__shared__ T" not in header
    assert constants["GRID_FORWARD_DYNAMICS_GRADIENT_USES_DA_DF_SPILL"] == 1
    assert constants["GRID_IDSVA_SO_USES_GLOBAL_OUTPUT"] == 1
    assert constants["GRID_FDSVA_SO_USES_GLOBAL_TENSORS"] == 1
    assert constants["GRID_FDSVA_SO_USES_WORKSPACE_TEMP"] == 1
    assert "grid_begin_l2_persisting" in header
    assert "grid_end_l2_persisting" in header


# The G0 footgun guard now refuses exactly ONE remaining mimic gradient:
# floating-base mimic INTEGRATOR gradients (B3 — multi-stage RK stage-projection,
# deferred). Everything else that was once refused is now SUPPORTED + emits:
# fixed-base mimic id_du/fd_du + ee grad/hessian (P3/P4) and floating-base mimic
# id_du/fd_du (B1) + second-order idsva_so/fdsva_so (B2/B4).
@pytest.mark.cuda_equivalence
@pytest.mark.developer_only
@pytest.mark.parametrize("base,profile", [("floating", "integrators"), ("floating", "all")])
def test_mimic_integrator_gradient_codegen_refused_not_zeroed(tmp_path, base, profile):
    """G0 footgun guard for the ONE remaining unsupported mimic gradient
    (floating-base mimic integrator gradients, B3): codegen must RAISE a clear
    NotImplementedError — NOT silently emit zeroed output."""
    with pytest.raises(NotImplementedError, match="mimic gradients not yet supported"):
        _generate_header(tmp_path, "fr3", base, codegen_profile=profile)


@pytest.mark.cuda_equivalence
@pytest.mark.developer_only
@pytest.mark.parametrize(
    "base,profile",
    [("fixed", "dynamics-gradients"), ("fixed", "second-order"),
     ("floating", "dynamics-gradients"), ("floating", "second-order")],
)
def test_mimic_gradient_codegen_now_emits(tmp_path, base, profile):
    """Fixed-base mimic gradients (P3/P4) + floating-base mimic id_du/fd_du (B1)
    and second-order (B2/B4) are SUPPORTED now — they emit, no longer refused."""
    header = _generate_header(tmp_path, "fr3", base, codegen_profile=profile)
    assert "Generated algorithms:" in header


@pytest.mark.cuda_equivalence
@pytest.mark.developer_only
@pytest.mark.parametrize("profile", ["dynamics-core", "kinematics"])
def test_mimic_nongradient_codegen_still_works(tmp_path, profile):
    """The G0 guard is scoped to GRADIENT algorithms: a mimic robot still
    codegens its non-gradient surface (id/fd/aba/crba/minv/ee_pose) normally."""
    header = _generate_header(tmp_path, "fr3", "fixed", codegen_profile=profile)
    assert "Generated algorithms:" in header


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
    expected_d2ee_workspace = 1 if robot_id == "go2" else 0

    assert "__shared__ T" not in header
    assert constants["NUM_POS"] == constants["NUM_JOINTS"]
    assert constants["SECOND_ORDER_COORDS"] == constants["NUM_VEL"]
    assert constants["SECOND_ORDER_TENSOR_SIZE"] == 4 * constants["NUM_VEL"]**3
    assert constants["Q_QD_U_STRIDE"] == constants["NUM_POS"] + 2 * constants["NUM_VEL"]
    assert constants["GRID_GENERATES_IDSVA_SO_BODY_FRAME"] == 0
    assert constants["GRID_GENERATES_FDSVA_SO"] == 0
    assert constants["GRID_GENERATES_D2EE"] == 1
    assert constants["GRID_END_EFFECTOR_POSE_HESSIAN_USES_WORKSPACE_TEMP"] == expected_d2ee_workspace
    assert constants["GRID_END_EFFECTOR_POSE_HESSIAN_USES_WORKSPACE_D2XHOM"] == 0
    assert constants["GRID_END_EFFECTOR_POSE_HESSIAN_SHARED_TIER_VALUE"] == expected_d2ee_workspace
    assert "!GRID_GENERATES_IDSVA_SO_BODY_FRAME" in header
    assert "!GRID_GENERATES_FDSVA_SO" in header
    assert "void end_effector_pose(gridData<T, KIND> *hd_data" in header
    assert "void end_effector_pose_gradient(gridData<T, KIND> *hd_data" in header
    assert "void end_effector_pose_hessian(gridData<T, KIND> *hd_data" in header
    assert "void kinematics_only(gridData<T, KIND> *hd_data" in header
    assert "void aba(gridData<T, KIND> *hd_data" in header
    assert "void crba(gridData<T, KIND> *hd_data" in header


@pytest.mark.cuda_equivalence
@pytest.mark.developer_only
@pytest.mark.floating_base
@pytest.mark.parametrize(
    ("robot_id", "algorithm_list", "generates_fdsva", "enable_world_frame"),
    [
        pytest.param("iiwa14", "idsva_so_body_frame", 0, False, id="iiwa14-idsva-body-frame-only"),
        pytest.param("iiwa14", "idsva_so_body_frame,fdsva_so", 1, False, id="iiwa14-idsva-body-frame-fdsva"),
        pytest.param("go2", "idsva_so_body_frame", 0, False, id="go2-idsva-body-frame-only"),
        pytest.param("iiwa14", "idsva_so_body_frame", 0, True, id="iiwa14-idsva-body-frame-world-frame"),
    ],
)
def test_floating_second_order_opt_in_header_compiles(
    tmp_path,
    robot_id,
    algorithm_list,
    generates_fdsva,
    enable_world_frame,
):
    header = _generate_header(
        tmp_path,
        robot_id,
        "floating",
        algorithm_list=algorithm_list,
        enable_floating_second_order=True,
        enable_idsva_so_world_frame=enable_world_frame,
    )
    constants = _constants(header)

    assert constants["GRID_GENERATES_IDSVA_SO_BODY_FRAME"] == 1
    assert constants["GRID_GENERATES_FDSVA_SO"] == generates_fdsva
    assert constants["SECOND_ORDER_COORDS"] == constants["NUM_VEL"]
    assert constants["SECOND_ORDER_TENSOR_SIZE"] == 4 * constants["NUM_VEL"]**3
    assert constants["Q_QD_U_STRIDE"] == constants["NUM_POS"] + 2 * constants["NUM_VEL"]
    assert "void idsva_so_body_frame(gridData<T, KIND> *hd_data" in header
    if generates_fdsva:
        assert "void fdsva_so(gridData<T, KIND> *hd_data" in header
    else:
        assert "void fdsva_so(gridData<T, KIND> *hd_data" not in header
    if enable_world_frame:
        assert "void idsva_so_world_frame(gridData<T, KIND> *hd_data" in header
        assert "void idsva_so_world_frame_inner(" in header
        assert "void idsva_so_world_frame_kernel(" in header
    else:
        assert "idsva_so_world_frame" not in header

    _compile_header_consumer(
        tmp_path,
        header,
        """
        #include "grid.cuh"
        int main() {
            static_assert(grid::GRID_GENERATES_IDSVA_SO_BODY_FRAME == 1, "IDSVA-SO must be generated");
            static_assert(grid::GRID_GENERATES_FDSVA_SO == EXPECTED_FDSVA, "FDSVA-SO flag mismatch");
            static_assert(grid::SECOND_ORDER_COORDS == grid::NUM_VEL, "second-order tensor must be velocity-sized");
            static_assert(grid::SECOND_ORDER_TENSOR_SIZE == 4 * grid::NUM_VEL * grid::NUM_VEL * grid::NUM_VEL, "tensor size mismatch");
            static_assert(grid::Q_QD_U_STRIDE == grid::NUM_POS + 2 * grid::NUM_VEL, "q/qd/u stride mismatch");
            return 0;
        }
        """.replace("EXPECTED_FDSVA", str(generates_fdsva)),
        f"{robot_id}_floating_so_{algorithm_list.replace(',', '_')}",
    )


@pytest.mark.cuda_equivalence
@pytest.mark.developer_only
@pytest.mark.parametrize(
    ("robot_id", "base_mode", "expected_tier"),
    [
        pytest.param("iiwa14", "fixed", 0, id="iiwa14-fixed"),
        pytest.param("go2", "fixed", 0, id="go2-fixed"),
        pytest.param("g1", "fixed", 1, id="g1-fixed"),
        pytest.param("fetch", "fixed", 1, id="fetch-fixed"),
        pytest.param("iiwa14", "floating", 0, id="iiwa14-floating"),
        pytest.param("go2", "floating", 1, id="go2-floating"),
        pytest.param("g1", "floating", 2, id="g1-floating"),
    ],
)
def test_d2ee_spill_tiers_are_size_and_base_selected(robot_id, base_mode, expected_tier):
    codegen = _codegen_for_robot(robot_id, base_mode)
    codegen.generated_algorithms = {"end_effector_pose", "end_effector_pose_gradient", "end_effector_pose_hessian"}
    codegen.generate_id_du = False
    codegen.generate_fd_du = False
    codegen.generate_end_effector_pose_hessian = True
    codegen.generate_idsva_so_body_frame = False
    codegen.generate_fdsva_so = False
    codegen.include_fixed_kinematic_targets = False

    codegen.gen_add_constants_helpers(include_homogenous_transforms=True)
    constants = _constants(codegen.code_str)

    assert constants["GRID_END_EFFECTOR_POSE_HESSIAN_SHARED_TIER_VALUE"] == expected_tier
    assert constants["GRID_END_EFFECTOR_POSE_HESSIAN_USES_WORKSPACE_TEMP"] == int(expected_tier >= 1)
    assert constants["GRID_END_EFFECTOR_POSE_HESSIAN_USES_WORKSPACE_D2XHOM"] == int(expected_tier >= 2)


@pytest.mark.cuda_equivalence
@pytest.mark.developer_only
def test_generated_header_includes_grid_data_variants_and_no_rnea_alias(tmp_path):
    # iiwa14 (non-mimic): the default "all" profile emits gradients, which is
    # refused for mimic fr3 under the G0 guard. The gridData surface asserted
    # here is robot-agnostic.
    header = _generate_header(tmp_path, "iiwa14", "fixed")

    assert "enum gridDataKind { GRID_DATA_ALL = 0, GRID_DATA_DYNAMICS = 1, GRID_DATA_KINEMATICS = 2 };" in header
    assert "template <typename T, gridDataKind KIND = GRID_DATA_ALL>" in header
    assert "gridData<T, KIND> *init_gridData" in header
    assert "void close_grid(cudaStream_t *streams, robotModel<T> *d_robotModel, gridData<T, KIND> *hd_data)" in header
    # Clean-break: there is NO grid::inverse_dynamics alias — inverse_dynamics is the single
    # canonical name (RNEA stays greppable via docstrings/comments only).
    assert "void inverse_dynamics(gridData<T, KIND> *hd_data" not in header
    assert "void rnea_single_timing(gridData<T, KIND> *hd_data" not in header
    assert "void rnea_compute_only(gridData<T, KIND> *hd_data" not in header
    assert "void inverse_dynamics(gridData<T, KIND> *hd_data" in header


@pytest.mark.cuda_equivalence
@pytest.mark.developer_only
def test_linalg_backend_controls_and_helpers_are_generated(tmp_path):
    header = _generate_header(tmp_path, "fr3", "fixed", codegen_profile="dynamics-core")

    assert "#define GRID_LINALG_GLASS 0" in header
    assert "#define GRID_LINALG_GLASS_NVIDIA 1" in header
    assert "#ifndef GRID_CUDA_LINALG_BACKEND" in header
    assert "GRID_LINALG_AUTO resolves to GLASS simple helpers" not in header
    assert "namespace glass" in header
    assert "Vendored from GLASS at codegen time" in header
    assert "BEGIN GLASS src/base/L1/dot_strided.cuh" in header
    assert "BEGIN GLASS src/base/L2/gemv_strided.cuh" in header
    assert "BEGIN GLASS src/base/L3/gemm_strided.cuh" in header
    assert "glass::dot_strided" in header
    assert "glass::row_strided_gemv" in header
    assert "glass::row_strided_gemm" in header
    assert "glass::gemm_ex" in header
    assert "namespace nvidia" in header
    assert "GRID_LINALG_NVIDIA_MAX_HELPER_BYTES" in header
    assert "grid_linalg_gemm_glass" in header
    assert "grid_linalg_gemm" in header
    assert "grid_linalg_gemv" in header
    assert "grid_linalg_row_strided_gemv" in header
    assert "grid_linalg_row_strided_gemm" in header
    assert "grid_linalg_nvidia_row_strided_gemv_smem_bytes" in header
    assert "grid_linalg_nvidia_row_strided_gemm_smem_bytes" in header
    # GLASS round-2 + Gap D unlock: the internal `_nvidia` helpers
    # (grid_linalg_packed_gemm_nvidia_colmajor, _transb, row_strided_*_nvidia)
    # were collapsed into the public wrappers above, which now call
    # ::glass::nvidia::* directly. The internal helpers must NOT regrow.
    assert "grid_linalg_packed_gemm_nvidia_colmajor" not in header
    assert "grid_linalg_packed_gemm_nvidia_transb" not in header
    assert "grid_linalg_row_strided_gemv_nvidia" not in header
    assert "grid_linalg_row_strided_gemm_nvidia" not in header


@pytest.mark.cuda_equivalence
@pytest.mark.developer_only
def test_linalg_backend_default_cxx11_compiles_without_mathdx(tmp_path):
    header = _generate_header(tmp_path, "fr3", "fixed", codegen_profile="dynamics-core")
    source = r'''
#include "grid.cuh"

int main() {
    using T = float;
    T *A = nullptr;
    T *B = nullptr;
    T *C = nullptr;
    (void)A;
    (void)B;
    (void)C;
    return grid::GRID_CUDA_USE_GLASS_NVIDIA_VALUE;
}
'''
    _compile_header_consumer(
        tmp_path,
        header,
        source,
        "linalg_backend_default_cxx11",
    )


@pytest.mark.cuda_equivalence
@pytest.mark.developer_only
def test_linalg_base_strided_helpers_compile(tmp_path):
    header = _generate_header(tmp_path, "fr3", "fixed", codegen_profile="dynamics-core")
    source = r'''
#include "grid.cuh"

__global__ void smoke(float *A, float *B, float *C) {
    C[0] = grid::dot_prod<float, 4, 4, 1>(A, B);
    C[1] = grid::dot_prod<float, 6, 1, 1>(A, B);
    C[2] = grid::dot_prod<float, 6, 6, 1>(A, B);
    C[3] = grid::dot_prod<float, 6, 6, 6>(A, B);
    grid::grid_linalg_row_strided_gemv<float, 6, 6, 8>(A, B, C, 1.0f, 0.0f);
    grid::grid_linalg_row_strided_gemm<float, 6, 6, 6, 8, 8>(A, B, C, 1.0f, 0.0f);
}

int main() { return 0; }
'''
    _compile_header_consumer(
        tmp_path,
        header,
        source,
        "linalg_base_strided_helpers",
    )


@pytest.mark.cuda_equivalence
@pytest.mark.developer_only
def test_dynamics_grid_data_variant_wrappers_compile(tmp_path):
    # iiwa14 (non-mimic): the "dynamics" profile includes gradient algos, which
    # are refused for mimic fr3 under the G0 guard. The DYNAMICS gridData-variant
    # wrapper surface asserted below is robot-agnostic.
    header = _generate_header(tmp_path, "iiwa14", "fixed", codegen_profile="dynamics")
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
        data, model, static_cast<T>(-9.81), 1, blocks, threads, streams);
    grid::minv<T, false, grid::GRID_DATA_DYNAMICS>(
        data, model, 1, blocks, threads, streams);
    grid::forward_dynamics<T, grid::GRID_DATA_DYNAMICS>(
        data, model, static_cast<T>(-9.81), 1, blocks, threads, streams);
    grid::inverse_dynamics_gradient<T, false, false, grid::GRID_DATA_DYNAMICS>(
        data, model, static_cast<T>(-9.81), 1, blocks, threads, streams);
    grid::forward_dynamics_gradient<T, false, grid::GRID_DATA_DYNAMICS>(
        data, model, static_cast<T>(-9.81), 1, blocks, threads, streams);
    grid::dynamics_only<T, grid::GRID_DATA_DYNAMICS>(
        data, model, static_cast<T>(-9.81), 1, blocks, threads, streams);
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
    # iiwa14 (non-mimic): kinematics-derivatives includes ee_pose_gradient/hessian,
    # which the G0 guard refuses for mimic fr3. The wrapper surface is robot-agnostic
    # (the floating variant below already uses iiwa14).
    header = _generate_header(tmp_path, "iiwa14", "fixed", codegen_profile="kinematics-derivatives")
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
    grid::end_effector_pose_hessian<T, false, grid::GRID_DATA_KINEMATICS>(
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
    grid::end_effector_pose_hessian<T, false, grid::GRID_DATA_KINEMATICS>(
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
        data, model, static_cast<T>(-9.81), 1, blocks, threads, streams);
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
    # iiwa14 (non-mimic): needs the full "all" header (gridData variant wrappers),
    # which the G0 guard refuses for mimic fr3. The KIND-mismatch negative-compile
    # checks are robot-agnostic.
    header = _generate_header(tmp_path, "iiwa14", "fixed")
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
    assert "void minv(gridData<T, KIND> *hd_data" in header
    assert "void forward_dynamics(gridData<T, KIND> *hd_data" in header
    assert "void dynamics_core(gridData<T, KIND> *hd_data" in header
    assert "void id_minv_fd(gridData<T, KIND> *hd_data" in header
    assert "void inverse_dynamics_gradient(gridData<T, KIND> *hd_data" not in header
    assert "void forward_dynamics_gradient(gridData<T, KIND> *hd_data" not in header
    assert "void all_dynamics(gridData<T, KIND> *hd_data" not in header
    assert "void end_effector_pose(gridData<T, KIND> *hd_data" not in header
    assert "void idsva_so_body_frame(gridData<T, KIND> *hd_data" not in header
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
    assert "void minv(gridData<T, KIND> *hd_data" not in header
    assert "void forward_dynamics(gridData<T, KIND> *hd_data" not in header


@pytest.mark.cuda_equivalence
@pytest.mark.developer_only
def test_algorithm_list_override_expands_dependencies(tmp_path):
    # iiwa14 (non-mimic): this asserts the fd-gradient dependency expansion emits
    # gradient wrappers, which the G0 guard refuses for mimic fr3. The
    # algorithm_list expansion logic under test is robot-agnostic.
    header = _generate_header(tmp_path, "iiwa14", "fixed", algorithm_list="forward_dynamics_gradient")

    assert "Generated algorithms:" in header
    assert "void inverse_dynamics(gridData<T, KIND> *hd_data" in header
    assert "void minv(gridData<T, KIND> *hd_data" in header
    assert "void forward_dynamics(gridData<T, KIND> *hd_data" in header
    assert "void inverse_dynamics_gradient(gridData<T, KIND> *hd_data" in header
    assert "void forward_dynamics_gradient(gridData<T, KIND> *hd_data" in header
    assert "void dynamics_core(gridData<T, KIND> *hd_data" in header
    assert "void dynamics_gradients(gridData<T, KIND> *hd_data" in header
    assert "void all_dynamics(gridData<T, KIND> *hd_data" in header
    assert "void end_effector_pose(gridData<T, KIND> *hd_data" not in header
