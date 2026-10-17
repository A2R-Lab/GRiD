"""The installed release acceptance matrix also belongs to the signed scope."""
from pathlib import Path

import pytest
import grid_rbd
from tools.check_installed_numerics import run

pytestmark = pytest.mark.python_wrappers


@pytest.mark.parametrize('robot,floating', [('iiwa14', False), ('go2', True)])
def test_release_numerical_matrix(robot, floating, tmp_path):
    run(tmp_path, Path(__file__).resolve().parents[2] / 'grid_assets/robots',
        [(robot, floating)], require_installed=False,
        cache_dir=grid_rbd.default_cache_dir())
