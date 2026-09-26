"""CPU-only checks for documented interfaces and runnable input examples."""
import ast
import csv
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from examples.integrator_semantics import position_error
from examples.model_inputs import inertial_parameters, model_inputs, regressor_residual

ROOT = Path(__file__).resolve().parents[1]


def test_backend_inventory_matches_class_definitions():
    sources = (
        ("bindings/grid_rbd/_handle.py", "RobotHandle"),
        ("bindings/grid_rbd/jax/__init__.py", "JaxRobotHandle"),
        ("bindings/grid_rbd/torch/__init__.py", "TorchRobotHandle"),
    )
    methods = []
    for path, name in sources:
        tree = ast.parse((ROOT / path).read_text())
        cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == name)
        methods.append({n.name for n in cls.body if isinstance(n, ast.FunctionDef)})
    page = ROOT / "docs/source/user_guide/tutorials/backend_coverage.rst"
    table = page.read_text().split("   :widths: 55, 15, 15, 15\n\n", 1)[1].split("\n\n", 1)[0]
    rows = list(csv.reader(table.splitlines(), skipinitialspace=True))
    assert len(rows) == 35  # Guard against accidentally testing an empty/truncated table.
    assert len({row[0] for row in rows}) == len(rows)
    for method, *flags in rows:
        assert flags == ["Yes" if method in m else "No" for m in methods], method


@pytest.mark.parametrize("name,nq,nv", [("iiwa14", 7, 7), ("go2", 19, 18)])
def test_documented_input_packing(name, nq, nv):
    robot, q, qd, u_padded, x, u = model_inputs(name)
    assert (robot.get_num_pos(), robot.get_num_vel()) == (nq, nv)
    assert q.shape == qd.shape == u_padded.shape == (2, nq)
    assert x.shape == (2, nq + nv)
    assert u.shape == (2, nv)
    for array in (q, qd, u_padded, x, u):
        assert array.dtype == np.float32
        assert array.flags.c_contiguous
    np.testing.assert_array_equal(x[:, :nq], q)
    np.testing.assert_array_equal(x[:, nq:], qd[:, :nv])
    np.testing.assert_array_equal(u, u_padded[:, :nv])
    assert not np.any(qd[:, nv:])
    assert not np.any(u_padded[:, nv:])
    if name == "go2":
        np.testing.assert_array_equal(q[:, 3:7], [[0, 0, 0, 1]] * 2)


def test_documented_inertia_basis_matches_reference_regressor():
    robot, *_ = model_inputs("iiwa14")
    assert inertial_parameters(robot).shape == (10 * robot.get_num_bodies(),)
    assert regressor_residual(robot) < 1e-11


def test_runtime_tool_transform_normalization_cpu_only():
    from grid_rbd._handle import RobotHandle

    # Exercise the real pure-NumPy helper without constructing a GPU handle.
    context = SimpleNamespace(_dt=np.float32)
    normalize = lambda offsets, count: RobotHandle._normalize_ee_offsets(context, offsets, count)
    X = np.eye(4, dtype=np.float32)
    X[:3, :3] = [[0, -1, 0], [1, 0, 0], [0, 0, 1]]
    X[:3, 3] = [0.1, 0.2, 0.3]
    for normalized in normalize(X, 2):
        np.testing.assert_array_equal(normalized.reshape(4, 4, order="F"), X)
    for normalized in normalize([[0, 0, 0.1]], 2):
        actual = normalized.reshape(4, 4, order="F")
        np.testing.assert_array_equal(actual[:3, :3], np.eye(3))
        np.testing.assert_allclose(actual[:3, 3], [0, 0, 0.1])
    np.testing.assert_array_equal(normalize(None, 1)[0].reshape(4, 4), np.eye(4))
    with pytest.raises(ValueError, match="length"):
        normalize([[0, 0, 0], [0, 0, 1]], 3)


def test_current_integrator_characterization():
    # This documents current semantics, NOT desired classical RK4 behavior.
    # Update deliberately together with the docs if that contract changes.
    for steps in (10, 20, 40, 80):
        assert position_error(steps) == pytest.approx(0.5 / steps, abs=1e-14)
        assert position_error(steps, "trapezoidal") < 1e-14
