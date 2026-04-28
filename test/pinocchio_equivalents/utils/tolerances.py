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
    "aba": DEFAULT_TOLERANCE,
}

ROBOT_ALGORITHM_TOLERANCES = {
    ("g1", "rnea"): Tolerance(
        rtol=1e-6,
        atol=5e-7,
        note="G1 fixed-base dynamics, ABA, and derivative comparisons show stable agreement against Pinocchio at the sub-micro scale.",
    ),
    ("g1", "minv"): Tolerance(
        rtol=1e-6,
        atol=1e-7,
        note="G1 inverse-mass and CRBA comparisons need a slightly wider absolute tolerance than the smaller smoke robots.",
    ),
    ("g1", "aba"): Tolerance(
        rtol=1e-6,
        atol=5e-7,
        note="G1 fixed-base ABA comparisons stay within the same sub-micro residual scale as the existing G1 dynamics checks.",
    ),
    ("baxter", "aba"): Tolerance(
        rtol=1e-7,
        atol=5e-9,
        note="Baxter fixed-base ABA agrees with Pinocchio to within a few nanounits; this narrowly scoped absolute tolerance avoids failing on near-zero residuals.",
    ),
    ("rizon4", "rnea"): Tolerance(
        rtol=1e-7,
        atol=2e-9,
        note="Rizon4 fixed-base pose and dynamics checks stay at nanounit residual scale; this narrow absolute tolerance covers tiny frame-placement differences.",
    ),
}

def get_tolerance(algorithm: str, robot_id: str | None = None) -> Tolerance:
    if robot_id is not None:
        override = ROBOT_ALGORITHM_TOLERANCES.get((robot_id, algorithm))
        if override is not None:
            return override
    return ALGORITHM_TOLERANCES.get(algorithm, DEFAULT_TOLERANCE)
