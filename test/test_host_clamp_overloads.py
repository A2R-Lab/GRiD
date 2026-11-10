"""Native overloaded launches must clamp the actual selected kernel."""
import contextlib
import io
import re

import pytest

from grid_assets import robot_urdf
from grid_codegen import GRiDCodeGenerator
from URDFParser import URDFParser


@pytest.mark.parametrize("floating", [False, True])
def test_overloaded_host_launches_are_clamped(tmp_path, floating):
    path = tmp_path / "grid.cuh"
    with contextlib.redirect_stdout(io.StringIO()):
        robot = URDFParser().parse(str(robot_urdf("iiwa14")), floating_base=floating)
        GRiDCodeGenerator(robot).gen_all_code(output_path=str(path), algorithm_list=[
            "inverse_dynamics", "inverse_dynamics_gradient", "forward_dynamics_gradient"])
    text = path.read_text()
    assert not re.search(r"dim3 _grid_thr_clamped_[^\n]*\n\s*else", text)
    for name in ("inverse_dynamics", "inverse_dynamics_gradient", "forward_dynamics_gradient"):
        launches = [line for line in text.splitlines()
                    if name + "_kernel<" in line and "<<<" in line]
        assert launches
        assert all("_grid_thr_clamped_" in line for line in launches)
        assert re.search(r"static_cast<void\(\*\)\([^\n]+\)>\(&" + name + "_kernel<", text)
