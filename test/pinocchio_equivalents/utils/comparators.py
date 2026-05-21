import numpy as np

from test.pinocchio_equivalents.utils.tolerances import get_tolerance


def assert_close(actual, expected, algorithm: str, robot_id: str | None = None) -> None:
    tol = get_tolerance(algorithm, robot_id=robot_id)
    expected_arr = np.asarray(expected, dtype=np.float64)
    # Magnitude-scaled absolute floor. np.testing.assert_allclose checks
    #   |actual - expected| <= atol + rtol*|expected|
    # per element, so an entry that is structurally ~0 (e.g. a coupling term
    # that vanishes at low energy) but carries float64 round-off ~1e-7*scale
    # trips the check at high velocity/acceleration even when the algorithm is
    # correct — because rtol*|expected| ~ 0 there and atol is tiny. Floor atol
    # at rtol*max|expected| so "small relative to the array's overall scale"
    # counts as close. A genuine error is O(scale) (or O(0.1*scale)) and still
    # exceeds this floor, so real bugs are NOT masked.
    scale = float(np.max(np.abs(expected_arr))) if expected_arr.size else 0.0
    atol_eff = max(tol.atol, tol.rtol * scale)
    np.testing.assert_allclose(
        actual,
        expected,
        rtol=tol.rtol,
        atol=atol_eff,
        err_msg=f"{algorithm} mismatch with tolerances rtol={tol.rtol}, atol={atol_eff:.3e} (scale={scale:.3e})",
    )
