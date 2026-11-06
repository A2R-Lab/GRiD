"""Construction, metadata and default-byte contracts for baked armature."""
import contextlib
import io

import numpy as np
import pytest

from grid_assets import robot_urdf
from grid_codegen import GRiDCodeGenerator
from grid_codegen.armature import option_vector, supported_slots
from grid_rbd._compile import generate_grid_cuh
from URDFParser import URDFParser


@pytest.fixture(scope="module")
def robot():
    with contextlib.redirect_stdout(io.StringIO()):
        return URDFParser().parse(str(robot_urdf("iiwa14")))


@pytest.mark.parametrize("runtime", [False, True])
def test_initializer_is_copied_and_validation_precedes_narrowing(robot, runtime):
    a = np.full(robot.get_num_vel(), .25)
    cg = GRiDCodeGenerator(robot, runtime_armature=runtime, armature=a)
    a[:] = 8
    assert cg.initial_armature == [.25] * robot.get_num_vel()
    for value in (-1e-100, np.inf, np.nan, 1e100):
        bad = np.zeros(robot.get_num_vel())
        bad[0] = value
        with pytest.raises(ValueError):
            GRiDCodeGenerator(robot, runtime_armature=runtime, armature=bad)


@pytest.mark.parametrize("runtime", [False, True])
def test_floating_root_checked_before_float32_conversion(runtime):
    with contextlib.redirect_stdout(io.StringIO()):
        robot = URDFParser().parse(str(robot_urdf("go2")), floating_base=True)
    a = np.zeros(robot.get_num_vel())
    a[0] = 1e-100
    with pytest.raises(ValueError, match="zero"):
        GRiDCodeGenerator(robot, runtime_armature=runtime, armature=a)


def test_options_are_json_safe_and_do_not_hide_invalid_values():
    assert option_vector(np.array([-.0, .25])) == [0., .25]
    for a in ([-1e-100], [np.nan], [[0]], [1j]):
        with pytest.raises(ValueError):
            option_vector(a)


@pytest.mark.parametrize("runtime", [False, True])
def test_generated_storage_and_metadata(tmp_path, runtime):
    out = tmp_path / "grid.cuh"
    a = [.25] * 7
    options = dict(armature=a, runtime_armature=runtime,
                   algorithm_list=["inverse_dynamics", "crba", "minv", "aba", "energy"],
                   enable_mujoco_kernels=False)
    with contextlib.redirect_stdout(io.StringIO()):
        meta = generate_grid_cuh(robot_urdf("iiwa14"), options, out)
    code = out.read_text()
    assert meta["initial_armature"] == a
    assert meta["armature_slots"] == list(range(7))
    assert meta["armature_units"] == ["kg m^2"] * 7
    if runtime:
        assert "h_armature_params[0] = static_cast<T>(0.25)" in code
        assert "d_armature_params" in code
        assert "armature_baked" not in code
    else:
        assert "constexpr T armature_baked(int slot)" in code
        assert "d_armature_params" not in code
        assert "s_c[vs] += armature_baked<T>(vs) * s_qdd[vs]" in code
        assert "s_M[vs + 7*vs] += armature_baked<T>(vs)" in code


def test_explicit_baked_zero_has_plain_header_bytes(robot, tmp_path):
    headers = []
    for index, a in enumerate((None, [0.] * robot.get_num_vel())):
        out = tmp_path / f"{index}.cuh"
        with contextlib.redirect_stdout(io.StringIO()):
            cg = GRiDCodeGenerator(robot, armature=a)
            cg.gen_all_code(output_path=str(out), algorithm_list=["inverse_dynamics", "minv"])
        headers.append(out.read_bytes())
        assert supported_slots(robot) == list(range(7))
    assert headers[0] == headers[1]


def test_prismatic_parameter_units_and_mimic_ownership(tmp_path):
    with contextlib.redirect_stdout(io.StringIO()):
        meta = generate_grid_cuh(robot_urdf("slider_mimic_fixed_tool"),
                                 {"algorithm_list": ["inverse_dynamics"]},
                                 tmp_path / "slider.cuh")
    assert meta["armature_slots"] == [0]
    assert meta["armature_units"] == ["kg"]
    assert meta["initial_armature"] == [0.]
