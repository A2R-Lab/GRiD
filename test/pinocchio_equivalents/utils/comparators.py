import numpy as np

from test.pinocchio_equivalents.utils.tolerances import get_tolerance


def assert_close(actual, expected, algorithm: str, robot_id: str | None = None) -> None:
    tol = get_tolerance(algorithm, robot_id=robot_id)
    np.testing.assert_allclose(
        actual,
        expected,
        rtol=tol.rtol,
        atol=tol.atol,
        err_msg=f"{algorithm} mismatch with tolerances rtol={tol.rtol}, atol={tol.atol}",
    )
