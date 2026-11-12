"""The dense inversion fallback must return the documented triangular buffer."""
import contextlib
import io
from pathlib import Path

import pytest

from grid_assets import robot_urdf
from grid_rbd._compile import generate_grid_cuh


@pytest.mark.parametrize("model", ["fr3", "mixed_spherical_arm"])
def test_crba_inverse_clears_lower_triangle(tmp_path, model):
    path = (robot_urdf(model) if model == "fr3" else
            Path(__file__).resolve().parents[1] / "external/URDFParser/tests/fixtures/mixed_spherical_arm.urdf")
    out = tmp_path / "grid.cuh"
    with contextlib.redirect_stdout(io.StringIO()):
        meta = generate_grid_cuh(path, {"algorithm_list": ["minv"],
                                      "enable_mujoco_kernels": False}, out)
    code = out.read_text()
    nv = meta["num_vel"]
    inversion = code.index(f"invert_matrix({nv}, s_M_reduced, s_Minv, s_invert_temp);")
    clear = code.index(f"if (ind % {nv} > ind / {nv}) s_Minv[ind] = static_cast<T>(0);", inversion)
    assert "__syncthreads();" in code[inversion:clear]
    assert f"ind < {nv * nv}" in code[inversion:clear]
