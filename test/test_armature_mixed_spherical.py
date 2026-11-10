"""A scalar joint after a ball joint owns a velocity slot, not a body ID."""
import contextlib
import io
from pathlib import Path

import pytest

from grid_codegen import GRiDCodeGenerator
from URDFParser import URDFParser


def mixed_skew_path(tmp_path):
    source = Path(__file__).parents[1] / "external/URDFParser/tests/fixtures/mixed_spherical_arm.urdf"
    path = tmp_path / "mixed_skew.urdf"
    path.write_text(source.read_text().replace('<axis xyz="1 0 0"/>', '<axis xyz="1 2 3"/>'))
    return path


@pytest.mark.parametrize("runtime", [False, True])
def test_mixed_spherical_skew_aba_uses_dense_axis_and_velocity_slot(tmp_path, runtime):
    with contextlib.redirect_stdout(io.StringIO()):
        robot = URDFParser().parse(str(mixed_skew_path(tmp_path)))
        assert robot.get_joint_index_v(2) == 4
        assert not robot.S_is_cardinal_by_id(2)
        cg = GRiDCodeGenerator(robot, runtime_armature=runtime,
                              armature=[.1, 0, 0, 0, .4])
        cg.gen_aba_inner()
    code = cg.code_str
    assert "mxS_general_peq_scaled<T>" in code
    assert "S_skew[r] * s_qdd[4]" in code
    assert "s_tau[4]" in code
    assert cg._armature_term_cpp(4) in code
    assert "None" not in code
