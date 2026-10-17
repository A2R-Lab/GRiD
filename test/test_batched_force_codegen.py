"""All batched force consumers select by sample k, not block or joint width."""
import contextlib
import os
import re

import pytest
from grid_assets import robot_urdf
from URDFParser import URDFParser
from grid_codegen import GRiDCodeGenerator

FAMILIES = ("inverse_dynamics", "forward_dynamics", "aba",
            "inverse_dynamics_gradient", "forward_dynamics_gradient",
            "integrator", "integrator_gradient", "integrator_with_gradient")
OFFSET = "T *d_f_ext_k = d_f_ext ? d_f_ext + static_cast<size_t>(k) * 6 * NUM_BODIES : nullptr;"


@pytest.mark.parametrize("robot,floating", [("iiwa14", False), ("go2", True), ("fr3", True)])
def test_every_batched_force_consumer_uses_sample_row(tmp_path, robot, floating):
    header = tmp_path / "grid.cuh"
    with open(os.devnull, "w") as sink, contextlib.redirect_stdout(sink):
        parsed = URDFParser().parse(str(robot_urdf(robot)), floating_base=floating)
        gen = GRiDCodeGenerator(parsed, FILE_NAMESPACE="grid")
        gen.gen_all_code(output_path=str(header), algorithm_list=list(FAMILIES))
    text = header.read_text()
    found = set()
    for match in re.finditer(r"void (\w+_kernel(?:_single_timing)?(?:_mjx)?)\(", text):
        name = match[1]
        family = next((f for f in FAMILIES if name.startswith(f + "_kernel")), None)
        if family is None:
            continue
        start = text.index("{", match.end())
        depth, end = 1, start + 1
        while depth:
            depth += (text[end] == "{") - (text[end] == "}")
            end += 1
        body = text[start:end]
        if "single_timing" in name:
            assert "d_f_ext_k" not in body, name
        else:
            found.add(family)
            # Some kernels emit a separate body for each resource tier.
            assert body.count(OFFSET) >= 1, name
            assert body.count("d_f_ext_k") >= 2, name
            assert not re.search(r"\bd_f_ext\b", body.replace(OFFSET, "")), name
    assert found == set(FAMILIES)
