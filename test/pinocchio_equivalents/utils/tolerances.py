from dataclasses import dataclass


@dataclass(frozen=True)
class Tolerance:
    rtol: float
    atol: float
    note: str = ""


DEFAULT_TOLERANCE = Tolerance(
    rtol=1e-7,
    atol=1e-9,
    note="Default tolerance for CPU-side reference comparisons against Pinocchio.",
)


ALGORITHM_TOLERANCES = {
    "rnea": DEFAULT_TOLERANCE,
    "minv": DEFAULT_TOLERANCE,
}


def get_tolerance(algorithm: str) -> Tolerance:
    return ALGORITHM_TOLERANCES.get(algorithm, DEFAULT_TOLERANCE)
