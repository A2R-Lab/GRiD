"""Separate capsule margins through both generated signatures and all thread counts."""
from pathlib import Path
import pytest

from test.cuda_equivalents.test_cuda_collision_config_free import _robot, _gen_header, _compile_and_run
from test.cuda_equivalents.test_cuda_collision_warp import _mixed_spec, _two_tier_spec


@pytest.mark.cuda_equivalence
@pytest.mark.developer_only
@pytest.mark.robot_smoke
@pytest.mark.parametrize("two_tier", [False, True])
def test_native_separate_margins(tmp_path, two_tier):
    robot = _robot()
    if two_tier:
        spec = _two_tier_spec(robot)
        fine = spec["tiers"][-1]
        # Original covering spheres enclose the old endpoints; +3cm also encloses
        # the new endpoints shifted 2cm along z, with the original safety slack.
        spec["tiers"][0]["radius"] = [r + .03 for r in spec["tiers"][0]["radius"]]
    else:
        fine = spec = _mixed_spec(robot)
    fine["pb"] = [v + (.02 if i % 3 == 2 else 0) for i, v in enumerate(fine["offset"])]
    _gen_header(robot, tmp_path, spec)
    (tmp_path / "margin_config.h").write_text(f"#define TEST_NATIVE_TWO_TIERS {int(two_tier)}\n")
    print(_compile_and_run(tmp_path, Path(__file__).with_name("cuda_collision_native_margins_runner.cu")))
