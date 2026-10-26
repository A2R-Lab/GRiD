"""CPU code-generation contract for feasibility-only separate margins."""
import pytest

from test.cuda_equivalents.test_cuda_collision_config_free import _robot, _gen_header
from test.cuda_equivalents.test_cuda_collision_warp import _mixed_spec, _two_tier_spec
from test.cuda_equivalents.test_cuda_collision_native import _two_row_spec, _iiwa_native_spec


@pytest.mark.parametrize("kind", ["sphere", "sphere_two", "capsule", "capsule_two"])
def test_separate_margins_emission(tmp_path, kind):
    robot = _robot()
    specs = {"sphere": lambda: _mixed_spec(robot), "sphere_two": lambda: _two_tier_spec(robot),
             "capsule": lambda: _two_row_spec(robot, 0, 2, 0.05),
             "capsule_two": lambda: _iiwa_native_spec()[1]}
    code = _gen_header(robot, tmp_path / kind, specs[kind]()).read_text()
    assert "typename CollisionMargins<T>::type margins = {}, bool check_self = true" in code
    assert "T margin = static_cast<T>(0), bool check_self" not in code
    if kind == "capsule_two":
        assert "dbg_fine_rechecked, margins, check_self" in code
    else:
        assert "margins.environment_margin" in code
        assert "margins.self_margin" in code
    if kind.startswith("sphere"):
        assert "CollisionMargins<float> margins = {}, bool check_self = true" in code
    # Feasibility inputs do not replace the independent environment hinge margin.
    assert "const Environment<T> &env, T margin, T weight," in code
    assert "T viol = margin -" in code
