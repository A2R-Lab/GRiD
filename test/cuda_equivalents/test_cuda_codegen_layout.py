import ast
import contextlib
import os
import re
import shutil
import subprocess
from pathlib import Path

import pytest

from grid_codegen import GRiDCodeGenerator
from grid_codegen.algorithms._idsva_so import (
    idsva_so_parent_topology_needs_reference_order_output_repair,
)
from test.cuda_equivalents.cuda_harness import _detect_cuda_arch
from RBDReference.tests import MANIFEST_PATH
from RBDReference.tests.model_sources import (
    iter_robot_cases,
    resolve_robot_spec,
)
from RBDReference.equivalents.reference_backend import build_project_adapter


CONST_RE = re.compile(r"const int (?P<name>[A-Z0-9_]+) = (?P<value>-?[0-9]+);")

REPO_ROOT = Path(__file__).resolve().parents[2]
CODEGEN_ROOT = REPO_ROOT / "grid_codegen"


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
            f"Could not resolve manifest {spec.robot_id}. Run ./install/developer_install.sh "
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
            f"Could not resolve manifest {spec.robot_id}. Run ./install/developer_install.sh "
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


def _compares_against_string_literal(test_node) -> bool:
    """True if the expression compares (== != in not-in) against a string literal.

    That is the syntactic shape of identity DISPATCH (`robot_id == "go2"`,
    `"iiwa14" in urdf_name`). It deliberately does NOT match a value-existence
    guard (`if not robot_id`) or a path/URI FORM predicate
    (`filename.startswith("package://")`) — neither selects anything per robot.
    """
    for node in ast.walk(test_node):
        if not isinstance(node, ast.Compare):
            continue
        if not any(isinstance(op, (ast.Eq, ast.NotEq, ast.In, ast.NotIn)) for op in node.ops):
            continue
        for operand in [node.left, *node.comparators]:
            for child in ast.walk(operand):
                if isinstance(child, ast.Constant) and isinstance(child.value, str):
                    return True
    return False


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
    # iiwa14 (non-mimic fixed-base) exercises the full gradient/SO surface.
    # (Historically fr3 was used here, but mimic gradient codegen was refused at
    # the time; mimic gradients are fully supported now — iiwa14 simply remains
    # the sentinel for these gradient spill-tier constants.)
    header = _generate_header(tmp_path, "iiwa14", "fixed")
    constants = _constants(header)

    # NOTE: the gradient/SO spill-tier constants below are the real signal that the
    # dynamics/gradient paths keep their workspace in shared (tier 0). A blanket
    # `"__shared__ T" not in header` is no longer valid: the batched FK / quadratic
    # cost helper kernels legitimately declare small fixed-size __shared__ T scratch
    # of their own, unrelated to the gradient spill arenas this test asserts about.
    assert constants["GRID_INVERSE_DYNAMICS_GRADIENT_USES_GLOBAL_TEMP"] == 0
    assert constants["GRID_FORWARD_DYNAMICS_GRADIENT_USES_GLOBAL_TEMP"] == 0
    assert constants["GRID_INVERSE_DYNAMICS_GRADIENT_USES_DA_DF_SPILL"] == 0
    assert constants["GRID_FORWARD_DYNAMICS_GRADIENT_USES_DA_DF_SPILL"] == 0
    assert constants["GRID_GENERATES_IDSVA_SO_BODY_FRAME"] == 1
    assert constants["GRID_GENERATES_FDSVA_SO"] == 1


@pytest.mark.cuda_equivalence
@pytest.mark.developer_only
def test_fixed_forced_low_shared_header_selects_fallbacks(tmp_path):
    # iiwa14 (non-mimic) — kept as the sentinel from when fr3's gradient/SO
    # codegen was refused; mimic gradients are fully supported now.
    header = _generate_header(tmp_path, "iiwa14", "fixed", target_shared_bytes=10000)
    constants = _constants(header)

    # See test_fixed_default_header_keeps_gradient_paths_all_shared: the spill-tier
    # constants are the real signal. A blanket `"__shared__ T" not in header` would
    # now trip on the batched FK / cost helper kernels' own small __shared__ scratch.
    assert constants["GRID_FORWARD_DYNAMICS_GRADIENT_USES_DA_DF_SPILL"] == 1
    assert constants["GRID_IDSVA_SO_USES_GLOBAL_OUTPUT"] == 1
    assert constants["GRID_FDSVA_SO_USES_GLOBAL_TENSORS"] == 1
    assert constants["GRID_FDSVA_SO_USES_WORKSPACE_TEMP"] == 1
    assert "grid_begin_l2_persisting" in header
    assert "grid_end_l2_persisting" in header


# There are NO remaining refused mimic gradients: the last one — floating-base
# mimic INTEGRATOR gradients — is now SUPPORTED + emits (B3, reduced-tangent-space
# stage projection, Euler..RK4 validated conditioning-scoped). Everything once
# refused now emits: fixed-base mimic id_du/fd_du + ee grad/hessian (P3/P4),
# floating-base mimic id_du/fd_du (B1), second-order idsva_so/fdsva_so (B2/B4),
# and floating-base mimic integrator gradients (B3).
@pytest.mark.cuda_equivalence
@pytest.mark.developer_only
@pytest.mark.parametrize("base,profile", [("floating", "integrators"), ("floating", "all")])
def test_mimic_integrator_gradient_codegen_now_emits(tmp_path, base, profile):
    """The previously-refused floating-base mimic integrator gradient is SUPPORTED
    now (B3): codegen must EMIT it, not raise — and must NOT silently zero it (the
    integrator gradient equivalence is validated separately in
    test_cuda_integrator_equivalence.py)."""
    header = _generate_header(tmp_path, "fr3", base, codegen_profile=profile)
    assert "Generated algorithms:" in header


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
                # A robot name embedded in PROSE (docstrings, C++ comment lines, doc
                # notes — anything containing whitespace) is documentation, never a
                # selector. The concern this test guards against is codegen BRANCHING
                # on robot identity, which would use a bare token like "iiwa14" as a
                # comparison RHS (no whitespace). Only flag bare-token literals; prose
                # mentioning robot names (perf tables, validation notes) is allowed.
                if re.search(r"\s", node.value.strip()):
                    continue
                if any(pattern.search(value) for pattern in fixture_patterns):
                    fixture_literals.append((relpath, node.lineno, node.value))
            if isinstance(node, ast.If):
                test_nodes = list(ast.walk(node.test))
                names = {_ast_name_text(child).lower() for child in test_nodes}
                names = {name for name in names if name}
                # Only identity DISPATCH breaks robot-agnostic codegen: a branch that
                # compares the identity against literal value(s). Merely testing that an
                # optional value EXISTS (`if not robot_id: return {}` for the per-robot
                # launch-config lookup) or resolving a mesh path's URI FORM
                # (`filename.startswith("package://")`) emits the same code for every
                # robot, so a literal comparison is required before flagging. A per-robot
                # lookup TABLE stays covered by the fixture_literals scan above, which
                # flags bare robot-name tokens anywhere in codegen.
                if (names & filename_branch_names) and _compares_against_string_literal(node.test):
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
    # v2.0: the d2ee inner no longer carries the dXhom/d2Xhom slabs (geometric-
    # Jacobian gradient computes them internally), so the d2ee smem arena shrank.
    # iiwa14 + go2 floating now both fit the d2ee output in shared at tier 0
    # (USES_WORKSPACE_TEMP == 0). See test_d2ee_spill_tiers_are_size_and_base_selected.
    expected_d2ee_workspace = 0

    # A blanket `"__shared__ T" not in header` is no longer valid: the batched FK /
    # cost helper kernels legitimately declare their own small __shared__ T scratch,
    # unrelated to the second-order / d2ee spill arenas this test asserts about.
    assert constants["NUM_POS"] == constants["NUM_JOINTS"]
    assert constants["SECOND_ORDER_COORDS"] == constants["NUM_VEL"]
    assert constants["SECOND_ORDER_TENSOR_SIZE"] == 4 * constants["NUM_VEL"]**3
    assert constants["Q_QD_U_STRIDE"] == 3 * constants["NUM_POS"]
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
    ("robot_id", "algorithm_list", "generates_fdsva", "enable_world_frame", "generates_body"),
    [
        pytest.param("iiwa14", "idsva_so_body_frame", 0, False, 1, id="iiwa14-idsva-body-frame-only"),
        # fdsva_so on floating-base calls idsva_so_world_frame_inner, so codegen now
        # (correctly) requires enable_idsva_so_world_frame=True alongside fdsva_so on
        # floating-base — otherwise it raises ValueError to avoid a link-time
        # undefined symbol. Enable world frame for this case.
        # A6 (a40d06e): on a world-dispatching robot (floating here) the body-frame
        # family is DEFAULT-DROPPED whenever world-frame emission is on — the
        # dispatcher routes to world and floating fdsva_so composes the WORLD inner,
        # so generates_body == int(not enable_world_frame) for these floating cases.
        pytest.param("iiwa14", "idsva_so_body_frame,fdsva_so", 1, True, 0, id="iiwa14-idsva-body-frame-fdsva"),
        pytest.param("go2", "idsva_so_body_frame", 0, False, 1, id="go2-idsva-body-frame-only"),
        pytest.param("iiwa14", "idsva_so_body_frame", 0, True, 0, id="iiwa14-idsva-body-frame-world-frame"),
    ],
)
def test_floating_second_order_opt_in_header_compiles(
    tmp_path,
    robot_id,
    algorithm_list,
    generates_fdsva,
    enable_world_frame,
    generates_body,
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

    assert constants["GRID_GENERATES_IDSVA_SO_BODY_FRAME"] == generates_body
    assert constants["GRID_GENERATES_FDSVA_SO"] == generates_fdsva
    assert constants["SECOND_ORDER_COORDS"] == constants["NUM_VEL"]
    assert constants["SECOND_ORDER_TENSOR_SIZE"] == 4 * constants["NUM_VEL"]**3
    # The per-timestep input block is THREE NUM_POS-WIDE SLOTS, not a tight packing:
    # every kernel reads s_q = base, s_qd = &base[NUM_POS], s_u = &base[2*NUM_POS]. On a
    # quaternion floating base qd/u carry nv meaningful values in their LEADING slots
    # plus one trailing pad each (fixed base: NUM_POS == NUM_VEL, so tight and slotted
    # coincide — which is why a tight expectation went unnoticed here). A consumer that
    # packs q|qd|u tightly would write u at NUM_POS+NUM_VEL while the kernels read it at
    # 2*NUM_POS — silent corruption, so this constant is a real ABI and is pinned here,
    # in test_cuda_input_abi.py, and by the device allocation / host memcpy / stride arg.
    assert constants["Q_QD_U_STRIDE"] == 3 * constants["NUM_POS"]
    if generates_body:
        assert "void idsva_so_body_frame(gridData<T, KIND> *hd_data" in header
    else:
        assert "void idsva_so_body_frame(gridData<T, KIND> *hd_data" not in header
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
            static_assert(grid::GRID_GENERATES_IDSVA_SO_BODY_FRAME == EXPECTED_BODY, "IDSVA-SO body-frame flag mismatch");
            static_assert(grid::GRID_GENERATES_FDSVA_SO == EXPECTED_FDSVA, "FDSVA-SO flag mismatch");
            static_assert(grid::SECOND_ORDER_COORDS == grid::NUM_VEL, "second-order tensor must be velocity-sized");
            static_assert(grid::SECOND_ORDER_TENSOR_SIZE == 4 * grid::NUM_VEL * grid::NUM_VEL * grid::NUM_VEL, "tensor size mismatch");
            static_assert(grid::Q_QD_U_STRIDE == 3 * grid::NUM_POS, "q/qd/u stride mismatch");
            static_assert(grid::GRID_Q_OFFSET == 0 && grid::GRID_QD_OFFSET == grid::NUM_POS
                          && grid::GRID_U_OFFSET == 2 * grid::NUM_POS
                          && grid::GRID_QDD_OFFSET == grid::GRID_U_OFFSET,
                          "published input-slot offset constants mismatch");
            return 0;
        }
        """.replace("EXPECTED_FDSVA", str(generates_fdsva))
           .replace("EXPECTED_BODY", str(generates_body)),
        f"{robot_id}_floating_so_{algorithm_list.replace(',', '_')}",
    )


@pytest.mark.cuda_equivalence
@pytest.mark.developer_only
@pytest.mark.parametrize(
    ("robot_id", "base_mode", "expected_tier"),
    [
        # v2.0: the d2ee inner dropped the dXhom/d2Xhom slabs (the geometric-
        # Jacobian gradient computes them internally), so the d2ee smem arena
        # shrank substantially. Robots that previously had to spill the d2ee output
        # now fit it in shared at a lower tier. The d2xhom (tier-2) flag is also
        # permanently 0 now — the FD inner never touches d2Xhom — so the max
        # meaningful d2ee tier is 1 (output -> workspace). Expected tiers updated:
        #   g1-fixed 1->0, fetch-fixed 1->0, go2-floating 1->0, g1-floating 2->1.
        pytest.param("iiwa14", "fixed", 0, id="iiwa14-fixed"),
        pytest.param("go2", "fixed", 0, id="go2-fixed"),
        pytest.param("g1", "fixed", 0, id="g1-fixed"),
        pytest.param("fetch", "fixed", 0, id="fetch-fixed"),
        pytest.param("iiwa14", "floating", 0, id="iiwa14-floating"),
        pytest.param("go2", "floating", 0, id="go2-floating"),
        pytest.param("g1", "floating", 1, id="g1-floating"),
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
    # iiwa14: kept as the sentinel from when mimic gradient codegen was refused
    # (mimic gradients are fully supported now). The gridData surface asserted
    # here is robot-agnostic.
    header = _generate_header(tmp_path, "iiwa14", "fixed")

    assert "enum gridDataKind { GRID_DATA_ALL = 0, GRID_DATA_DYNAMICS = 1, GRID_DATA_KINEMATICS = 2 };" in header
    assert "template <typename T, gridDataKind KIND = GRID_DATA_ALL>" in header
    assert "gridData<T, KIND> *init_gridData" in header
    assert "void close_grid(cudaStream_t *streams, robotModel<T> *d_robotModel, gridData<T, KIND> *hd_data)" in header
    # Clean-break: inverse_dynamics is the single canonical RNEA host (RNEA stays
    # greppable via docstrings/comments only). There is NO rnea_* alias host. The
    # canonical inverse_dynamics host legitimately exists and must be present.
    assert "void rnea_single_timing(gridData<T, KIND> *hd_data" not in header
    assert "void rnea_compute_only(gridData<T, KIND> *hd_data" not in header
    assert "void inverse_dynamics(gridData<T, KIND> *hd_data" in header


@pytest.mark.cuda_equivalence
@pytest.mark.developer_only
def test_linalg_backend_controls_and_helpers_are_generated(tmp_path):
    header = _generate_header(tmp_path, "fr3", "fixed", codegen_profile="dynamics-core")

    # v2.0 clean-break: cuBLASDx / GLASS-NVIDIA backend was REMOVED (it was a
    # no-op on sm_120 / RTX 5090 per the dispatcher findings). The linalg layer
    # is now SIMT-only vendored GLASS. The NVIDIA macros / namespace / helper
    # functions and the GRID_CUDA_LINALG_BACKEND/GRID_LINALG_GLASS* selector
    # macros no longer exist. The `GRID_LINALG_NVIDIA_MAX_HELPER_BYTES<T>()`
    # symbol is retained ONLY as a 0-returning stub so the shared-memory arena
    # macros keep compiling unchanged.
    assert "#define GRID_LINALG_GLASS_NVIDIA" not in header
    assert "#ifndef GRID_CUDA_LINALG_BACKEND" not in header
    assert "namespace nvidia" not in header
    assert "GRID_CUDA_USE_GLASS_NVIDIA_VALUE" not in header

    # Vendored SIMT GLASS primitives (still present).
    assert "namespace glass" in header
    assert "Vendored from GLASS at codegen time" in header
    assert "BEGIN GLASS src/base/L1/dot_strided.cuh" in header
    assert "BEGIN GLASS src/base/L2/gemv_strided.cuh" in header
    assert "BEGIN GLASS src/base/L3/gemm_strided.cuh" in header
    assert "glass::dot_strided" in header
    # GLASS v2 BLAS-convention names (row_strided_* -> *_strided; gemm_ex removed).
    assert "glass::gemv_strided" in header
    assert "glass::gemm_strided" in header
    assert "glass::gemm_ex" not in header
    assert "glass::gemv_ex" not in header

    # The arena-sizing stub must remain (returns 0 now, but the macros call it).
    assert "GRID_LINALG_NVIDIA_MAX_HELPER_BYTES" in header

    # Public grid_linalg_* SIMT wrappers (the surface kernels call).
    assert "grid_linalg_gemm" in header
    assert "grid_linalg_gemv" in header
    assert "grid_linalg_row_strided_gemv" in header
    assert "grid_linalg_row_strided_gemm" in header
    assert "grid_linalg_dot_strided" in header
    assert "grid_linalg_segmented_row_strided_gemv" in header

    # The removed NVIDIA-backed helpers must NOT regrow.
    assert "grid_linalg_gemm_glass" not in header
    assert "grid_linalg_nvidia_row_strided_gemv_smem_bytes" not in header
    assert "grid_linalg_nvidia_row_strided_gemm_smem_bytes" not in header
    assert "grid_linalg_packed_gemm_nvidia_colmajor" not in header
    assert "grid_linalg_packed_gemm_nvidia_transb" not in header
    assert "grid_linalg_row_strided_gemv_nvidia" not in header
    assert "grid_linalg_row_strided_gemm_nvidia" not in header


@pytest.mark.cuda_equivalence
@pytest.mark.developer_only
def test_linalg_backend_default_cxx11_compiles_without_mathdx(tmp_path):
    header = _generate_header(tmp_path, "fr3", "fixed", codegen_profile="dynamics-core")
    # v2.0: cuBLASDx/mathDx was removed; the default header is SIMT-only GLASS and
    # must compile under -std=c++11 with no mathDx headers present. The retained
    # GRID_LINALG_NVIDIA_MAX_HELPER_BYTES<T>() stub returns 0 (no NVIDIA helper
    # smem) — referencing it proves the SIMT-only arena path compiles clean.
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
    return static_cast<int>(grid::GRID_LINALG_NVIDIA_MAX_HELPER_BYTES<T>());
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
    # iiwa14: kept as the sentinel from when mimic gradient codegen was refused
    # (mimic gradients are fully supported now). The DYNAMICS gridData-variant
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
    # iiwa14: kept as the sentinel from when mimic ee_pose_gradient/hessian codegen
    # was refused (mimic gradients are fully supported now). The wrapper surface is
    # robot-agnostic (the floating variant below already uses iiwa14).
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
    # iiwa14: kept as the sentinel from when the full "all" header was refused for
    # mimic fr3 (mimic gradients are fully supported now). The KIND-mismatch
    # negative-compile checks are robot-agnostic.
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
    # iiwa14: kept as the sentinel from when mimic gradient wrapper codegen was
    # refused (mimic gradients are fully supported now). The algorithm_list
    # expansion logic under test is robot-agnostic.
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


@pytest.mark.cuda_equivalence
def test_joint_limits_land_at_true_q_offsets(tmp_path):
    """BUG 5 (GATO, 2026-08-09): gen_init_joint_limits kept only revolute joints
    and wrote each limit at the FILTERED index, not the joint's q-offset — on a
    floating robot the leg limits landed on the base q-slots (0..6) and the tail
    stayed uninitialized malloc. Pin the emitted rows against the URDF's own
    <limit> tags at the TRUE q layout for a fixed arm and a floating quadruped."""
    import re
    import xml.etree.ElementTree as _ET

    def emitted_rows(header, n):
        rows = {}
        for m in re.finditer(r"h_joint_limits\[(\d+)\] = (.+?);", header):
            rows[int(m.group(1))] = m.group(2).strip()
        assert set(rows) == set(range(2 * n)), "every q slot must be initialized (lower AND upper)"
        return rows

    def urdf_limits_in_doc_order(urdf_path):
        root = _ET.parse(urdf_path).getroot()
        out = []
        for j in root.iter("joint"):
            jt = j.get("type")
            if jt in ("fixed", None):
                continue
            if j.find("mimic") is not None:
                continue  # no own q slot
            lim = j.find("limit")
            if jt in ("revolute", "prismatic") and lim is not None:
                out.append((float(lim.get("lower")), float(lim.get("upper"))))
            else:
                out.append(None)  # continuous / unlimited -> +/-inf slot
        return out

    def num(s):
        assert "INFINITY" not in s.upper(), f"expected a finite limit, got {s}"
        return float(re.search(r"static_cast<T>\((.+)\)", s).group(1))

    # iiwa14 fixed: 7 revolute limits at slots 0..6 exactly (regression guard for
    # the fixed-base case, which the old code got right).
    header = _generate_header(tmp_path, "iiwa14", "fixed", algorithm_list="inverse_dynamics")
    limits = urdf_limits_in_doc_order(REPO_ROOT / "config" / "robot_assets" / "iiwa14.urdf")
    n = 7
    rows = emitted_rows(header, n)
    for i, lim in enumerate(limits):
        assert lim is not None
        assert abs(num(rows[i]) - lim[0]) < 1e-9, f"slot {i} lower"
        assert abs(num(rows[i + n]) - lim[1]) < 1e-9, f"slot {i} upper"

    # go2 floating: slots 0..6 (base pose incl quaternion) must be +/-inf, and the
    # 12 leg limits land at slots 7..18 in joint order.
    header = _generate_header(tmp_path, "go2", "floating", algorithm_list="inverse_dynamics")
    limits = urdf_limits_in_doc_order(REPO_ROOT / "config" / "robot_assets" / "go2.urdf")
    assert len(limits) == 12
    n = 19
    rows = emitted_rows(header, n)
    for i in range(7):
        assert "INFINITY" in rows[i] and "INFINITY" in rows[i + n], f"base slot {i} must be unlimited"
    for k, lim in enumerate(limits):
        i = 7 + k
        assert lim is not None
        assert abs(num(rows[i]) - lim[0]) < 1e-9, f"slot {i} lower"
        assert abs(num(rows[i + n]) - lim[1]) < 1e-9, f"slot {i} upper"

    # COMPILE the go2 restricted header: the +/-inf rows only ever EMIT on robots
    # with unlimited slots, so a text-only check let an uncompilable spelling
    # (std::numeric_limits with no <limits> include) reach the suite once already.
    nvcc = shutil.which("nvcc")
    if nvcc is None:
        pytest.skip("nvcc not on PATH")
    tu = tmp_path / "limits_tu.cu"
    tu.write_text('#include "go2_floating_default.cuh"\n'
                  'template float* grid::init_joint_limits<float>();\n'
                  'int main() { return 0; }\n')
    arch = os.environ.get("GRID_CUDA_ARCH", "sm_120")
    proc = subprocess.run([nvcc, "-std=c++17", f"-arch={arch}", f"-I{tmp_path}",
                           "-c", str(tu), "-o", str(tu.with_suffix(".o"))],
                          capture_output=True, text=True, timeout=1200)
    assert proc.returncode == 0, f"go2 limits TU failed to compile:\n{proc.stderr[-3000:]}"


@pytest.mark.cuda_equivalence
def test_tracking_cost_fc_overloads_emit_and_compile(tmp_path):
    """GATO ASK 1: with contact frames baked, grid_plant emits CONTROL_SIZE-wide
    fc overloads of the tracking-cost preset (value + gradient + hessian) with
    runtime fc_cost + nullable fc_ref; without contact frames the header is
    bitwise free of them (the FC_SIZE==0 contract). Compile-checks a TU that
    instantiates all three against the generated header."""
    import subprocess as _sp
    import shutil as _sh
    from URDFParser import URDFParser as _P
    from grid_codegen.GRiDCodeGenerator import GRiDCodeGenerator as _G
    from grid_codegen.algorithms._f_ext_contact import contact_frames_from_urdf as _cf

    urdf = str(REPO_ROOT / "config" / "robot_assets" / "iiwa14.urdf")
    algos = ["end_effector_pose", "end_effector_pose_gradient", "f_ext_gradient"]

    with open(os.devnull, "w") as devnull, contextlib.redirect_stdout(devnull):
        robot = _P().parse(urdf, floating_base=False)
        frames = _cf(robot, ["iiwa_joint_ee"])
        fc_header = tmp_path / "fc" / "grid.cuh"
        fc_header.parent.mkdir()
        _G(robot, FILE_NAMESPACE="grid").gen_all_code(
            algorithm_list=algos, output_path=str(fc_header), contact_frames=frames)
        robot2 = _P().parse(urdf, floating_base=False)
        plain_header = tmp_path / "plain" / "grid.cuh"
        plain_header.parent.mkdir()
        _G(robot2, FILE_NAMESPACE="grid").gen_all_code(
            algorithm_list=algos, output_path=str(plain_header))

    fc_text = fc_header.read_text()
    assert "GRID_PLANT_HAS_TRACKING_COST_FC" in fc_text
    assert "GRID_PLANT_CONTROL_SIZE = 13;" in fc_text  # NU=7 + 6*1 frame
    for fn in ("tracking_cost_fc", "tracking_cost_gradient_fc", "tracking_cost_hessian_fc"):
        assert f"void {fn}(" in fc_text, fn
        assert fn not in plain_header.read_text(), f"{fn} leaked into a no-contact build"

    nvcc = _sh.which("nvcc")
    if nvcc is None:
        pytest.skip("nvcc not on PATH")
    tu = fc_header.parent / "fc_tu.cu"
    tu.write_text(r'''
#include "grid.cuh"
using T = float;
__global__ void fc_probe(T *out, const T *x, const T *u, const grid::robotModel<T> *m) {
    __shared__ T s_scratch[4096]; __shared__ T s_ee[6]; __shared__ T s_eeg[6*7];
    __shared__ T s_qk[14]; __shared__ T s_rk[13]; __shared__ T s_Qk[14*14]; __shared__ T s_Rk[13*13];
    __shared__ T s_Rnu[7*7]; __shared__ T buf[14];
    grid_plant::tracking_cost_fc<T, 0, false>(out, x, u, buf, buf, buf, buf, buf, buf,
        buf, buf, (T)1, buf, buf, (T)1, buf, buf, (T)1, (T)0.5, nullptr, s_ee, s_scratch, m);
    grid_plant::tracking_cost_gradient_fc<T, 0, true>(s_qk, s_rk, x, u, buf, buf, buf, buf, buf, buf,
        buf, buf, (T)1, buf, buf, (T)1, buf, buf, (T)1, (T)0.5, nullptr, s_ee, s_eeg, s_scratch, m);
    grid_plant::tracking_cost_hessian_fc<T, 0, false>(s_Qk, s_Rk, x, u, buf, buf, buf,
        buf, buf, (T)1, buf, buf, (T)1, buf, buf, (T)1, (T)0.5, s_Rnu, s_eeg, s_scratch, m);
}
int main() { return 0; }
''')
    arch = os.environ.get("GRID_CUDA_ARCH", "sm_120")
    proc = _sp.run([nvcc, "-std=c++17", f"-arch={arch}", f"-I{fc_header.parent}",
                    "-c", str(tu), "-o", str(tu.with_suffix(".o"))],
                   capture_output=True, text=True, timeout=1200)
    assert proc.returncode == 0, f"fc TU failed to compile:\n{proc.stderr[-3000:]}"


@pytest.mark.cuda_equivalence
@pytest.mark.developer_only
def test_arena_carve_structs_compile(tmp_path):
    """GATO ASK6: the namespace-scope carve structs (grid::integrator_arena /
    grid::integrator_du_arena / grid_plant::plant_step_gradient_arena) are
    emitted, carve() compiles as device code, and members carry the
    kernel-layout types (drift in a member name/type fails this compile).

    The EXACT layout tie — carve walk == the sizer's baked t-count — is
    enforced at EMISSION time (gen_arena_carve_struct's expected_t_count
    raises during codegen on any drift), so this gate's job is the C++
    surface, not the arithmetic. The GRID_EE_FIXED_TARGET_NAME stamp
    (ASK6 step 4) is asserted both in the header text and via #ifndef."""
    # end_effector_pose pulls in the EE target-alias surface, whose emission
    # carries the GRID_EE_FIXED_TARGET_NAME stamp (the stamp is deliberately
    # absent from EE-less headers — it describes the alias family).
    header = _generate_header(tmp_path, "iiwa14", "fixed",
                              algorithm_list="integrator,integrator_with_gradient,end_effector_pose")
    assert "struct integrator_arena {" in header
    assert "struct integrator_du_arena {" in header
    assert "struct plant_step_gradient_arena {" in header
    assert '#define GRID_EE_FIXED_TARGET_NAME ""' in header
    assert "void end_effector_pose_target_inner(" in header
    source = r'''
#include "grid.cuh"

#ifndef GRID_EE_FIXED_TARGET_NAME
#error "GRID_EE_FIXED_TARGET_NAME must be stamped by gen_ee_target_aliases"
#endif

__global__ void carve_probe(unsigned char *out) {
    extern __shared__ __align__(16) unsigned char smem[];
    auto ia = grid::integrator_arena<float>::carve(smem);
    auto da = grid::integrator_du_arena<float>::carve(smem);
    auto pa = grid_plant::plant_step_gradient_arena<float>::carve(smem);
    // Type-checked member access: name/type drift fails to compile.
    float *f = ia.s_q_qd_u; f = ia.s_qdd; f = ia.s_stage_qdd; f = ia.s_stage_point;
    f = ia.s_x_kp1; f = ia.s_XImats; f = ia.s_temp;
    int *ti = ia.s_topology_helpers;
    unsigned char *lb = ia.s_linalg_smem;
    f = da.s_q_qd_u; f = da.s_dAB; f = da.s_df_du; f = da.s_dc_du; f = da.s_vaf;
    f = da.s_Minv; f = da.s_qdd; f = da.s_q_orig; f = da.s_qd_orig;
    f = da.s_stage_grad_qdd; f = da.s_D_qdd_stage;
    f = da.s_dInt_q_6x6; f = da.s_dInt_v_6x6; f = da.s_x_kp1;
    f = pa.s_x; f = pa.s_u; f = pa.s_dAB; f = pa.s_D_qdd_stage; f = pa.s_temp;
    out[0] = static_cast<unsigned char>((f != nullptr) + (ti != nullptr) + (lb != nullptr));
}

int main() {
    // The launch reservations the carve contract pairs with, host-visible.
    size_t a = grid::INTEGRATOR_DYNAMIC_SHARED_MEM_BYTES<float, grid::TIER_SHARED>();
    size_t b = grid::INTEGRATOR_DU_DYNAMIC_SHARED_MEM_BYTES<float, grid::TIER_SHARED>();
    return (a > 0 && b > 0) ? 0 : 1;
}
'''
    _compile_header_consumer(tmp_path, header, source, "arena_carve_structs",
                             cxx_standard="-std=c++17")
