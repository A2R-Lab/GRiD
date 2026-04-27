import numpy as np

from test.pinocchio_equivalents.tolerances import get_tolerance


def assert_close(actual, expected, algorithm: str) -> None:
    tol = get_tolerance(algorithm)
    np.testing.assert_allclose(
        actual,
        expected,
        rtol=tol.rtol,
        atol=tol.atol,
        err_msg=f"{algorithm} mismatch with tolerances rtol={tol.rtol}, atol={tol.atol}",
    )
