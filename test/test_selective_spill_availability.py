"""The dense inverse_dynamics_gradient inner refuses the selective spill flag.

The sparse inner routes its da_df band through grid_id_du_temp_ptr, so a caller
may shrink s_temp by ID_DU_TEMP_SPILL_COUNT and pass d_temp_spill. The dense
fold (mimic, skew axis, spherical) writes its whole pool through s_temp. A
consumer that composes the inner must be told which one it has, and the dense
one must fail to compile with the flag instead of writing out of bounds.
"""
import contextlib
import os
import re

import pytest
from grid_assets import robot_urdf
from URDFParser import URDFParser
from grid_codegen import GRiDCodeGenerator

ALGORITHMS = ["inverse_dynamics_gradient", "forward_dynamics_gradient"]
ASSERT = "static_assert(!USE_DA_DF_SPILL"


def _header(tmp_path, robot):
    header = tmp_path / "grid.cuh"
    with open(os.devnull, "w") as sink, contextlib.redirect_stdout(sink):
        parsed = URDFParser().parse(str(robot_urdf(robot)), floating_base=False)
        GRiDCodeGenerator(parsed, FILE_NAMESPACE="grid").gen_all_code(
            output_path=str(header), algorithm_list=ALGORITHMS)
    return header


def _inner_body(text):
    start = text.index("void inverse_dynamics_gradient_inner(")
    return text[start:text.index("\n}", start)]


@pytest.mark.parametrize("robot,available", [("iiwa14", 1), ("fr3", 0)])
def test_header_states_whether_the_selective_band_exists(tmp_path, robot, available):
    text = _header(tmp_path, robot).read_text()
    assert re.search(rf"const int GRID_ID_DU_SELECTIVE_SPILL_AVAILABLE = {available};", text)
    body = _inner_body(text)
    assert ("grid_id_du_temp_ptr" in body) == bool(available)
    assert (ASSERT in body) == (not available)

