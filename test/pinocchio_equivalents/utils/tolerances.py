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
    "pose_gradient": Tolerance(
        rtol=1e-6,
        atol=1e-8,
        note="End-effector pose gradients are compared against Pinocchio using a mix of analytic and finite-difference paths, so they use a slightly wider absolute tolerance than the primary dynamics checks.",
    ),
    "pose_hessian": Tolerance(
        rtol=1e-5,
        atol=5e-4,
        note="End-effector pose Hessians compare the analytic GRiD path against a finite-difference Pinocchio reference, so they use a wider tolerance than the first-order dynamics checks.",
    ),
}

ROBOT_ALGORITHM_TOLERANCES = {
    ("iiwa14", "pose_hessian"): Tolerance(
        rtol=1e-5,
        atol=3e-2,
        note="Floating-base iiwa14 pose Hessians now use the analytic free-flyer path and agree with the Pinocchio reference finite-difference check to within a few hundredths on the root-root block.",
    ),
    ("g1", "rnea"): Tolerance(
        rtol=1e-6,
        atol=3e-6,
        note="G1 fixed-base and floating-base dynamics, ABA, and derivative comparisons show stable agreement against Pinocchio at the low-micro scale, with floating gradients needing a slightly wider absolute tolerance.",
    ),
    ("g1", "minv"): Tolerance(
        rtol=1e-6,
        atol=1e-7,
        note="G1 inverse-mass and CRBA comparisons need a slightly wider absolute tolerance than the smaller smoke robots.",
    ),
    ("g1", "aba"): Tolerance(
        rtol=1e-6,
        atol=2e-6,
        note="G1 ABA comparisons stay within a low-micro residual scale against Pinocchio, so they use a narrowly widened absolute tolerance.",
    ),
    ("baxter", "aba"): Tolerance(
        rtol=1e-7,
        atol=5e-9,
        note="Baxter fixed-base ABA agrees with Pinocchio to within a few nanounits; this narrowly scoped absolute tolerance avoids failing on near-zero residuals.",
    ),
    ("gen3", "rnea"): Tolerance(
        rtol=1e-7,
        atol=1e-8,
        note="Gen3 floating-base inverse-dynamics gradients match Pinocchio up to a few nanounits on the current reference suite.",
    ),
    ("fetch", "rnea"): Tolerance(
        rtol=1e-6,
        atol=1e-8,
        note="Fetch floating-base inverse dynamics reaches single-digit nanounit residuals on near-zero entries; this narrow override avoids spurious failures without loosening the suite globally.",
    ),
    ("fetch", "aba"): Tolerance(
        rtol=1e-7,
        atol=1e-8,
        note="Fetch floating-base ABA reaches single-digit nanounit residuals on near-zero entries; this narrow override avoids spurious failures without loosening the suite globally.",
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
