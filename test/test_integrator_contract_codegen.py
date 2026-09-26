"""CPU-only guards for the shared integrator dispatch and stage contract."""
from grid_codegen.algorithms._integrator import _INTEGRATOR_TYPES, _STAGE_COUNT
from grid_codegen.algorithms._integrator_gradient import _INTEGRATOR_BUTCHER
from grid_codegen.kernel_attrs import KERNEL_OVERLOADS


def test_enum_order_and_stage_counts():
    assert _INTEGRATOR_TYPES == (
        "EULER", "SEMI_IMPLICIT_EULER", "MIDPOINT", "RK4", "TRAPEZOIDAL", "CONSTANT_ACCELERATION")
    assert _STAGE_COUNT == dict(EULER=1, SEMI_IMPLICIT_EULER=1, MIDPOINT=2,
                               RK4=4, TRAPEZOIDAL=2, CONSTANT_ACCELERATION=1)


def test_multistage_coefficients():
    assert _INTEGRATOR_BUTCHER == {
        "MIDPOINT": (2, [0.5], [0., 1.]),
        "TRAPEZOIDAL": (2, [1.], [0.5, 0.5]),
        "RK4": (4, [0.5, 0.5, 1.], [1/6, 1/3, 1/3, 1/6]),
    }


def test_kernel_attribute_manifest_covers_canonical_schemes_only():
    for operation in ("integrator", "integrator_gradient", "integrator_with_gradient"):
        kernels = [name for name, _ in KERNEL_OVERLOADS[operation]]
        assert len(kernels) == 12  # regular + timing for each of six schemes
        assert not any("RK3" in name for name in kernels)
        for scheme in _INTEGRATOR_TYPES:
            assert sum(f"IntegratorType::{scheme}>" in name for name in kernels) == 2
