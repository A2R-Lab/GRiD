"""Structural guard: KERNEL_ATTR_MANIFEST must register EVERY integrator-type
kernel the codegen actually emits.

WHY THIS EXISTS (the TRAPEZOIDAL launch-cliff, 2026-06-18). The integrator
kernels are templated on a non-type `IntegratorType IT` param, so each IT is a
distinct `__global__` instantiation. `init_grid` calls `cudaFuncSetAttribute(...,
cudaFuncAttributeMaxDynamicSharedMemorySize, ...)` once per manifest entry to
raise a kernel's dynamic-shared-memory ceiling above the 48 KB device default.
If an emitted IT is MISSING from the manifest, its attribute is never raised and
launching it with >48 KB of dynamic shared memory fails at runtime with
`cudaErrorInvalidValue` ("invalid argument") — but ONLY at an unspilled tier
(TIER_SHARED) on a robot whose arena exceeds 48 KB, so it slips through the
small/spilled cases. That is exactly how the floating TRAPEZOIDAL gradient kernel
crashed go2-floating at TIER_SHARED while passing at TIER_LITE.

The invariant: for every NON-mjx integrator family in `KERNEL_ATTR_MANIFEST`, the
set of registered `IntegratorType::X` equals the set the codegen emits
(`_INTEGRATOR_TYPES`). This is a pure-Python introspection test (no codegen run,
no nvcc, no GPU) so it runs in ordinary CI and fails the instant someone adds an
integrator type to the codegen without registering its kernel attribute.

(The mjx `mujoco_manifest` is built locally in `gen_init_gridData` with a
DELIBERATELY restricted IT policy — integrator_gradient(mjx) is single-stage
EULER/SI only — so it is intentionally NOT covered by this parity check.)
"""

from __future__ import annotations

import re

from GRiDCodeGenerator.GRiDCodeGenerator import GRiDCodeGenerator
from GRiDCodeGenerator.algorithms._integrator import _INTEGRATOR_TYPES

# The non-mjx integrator kernel families that fan out over IntegratorType and
# whose every emitted IT must carry a registered cudaFuncSetAttribute entry.
_INTEGRATOR_FAMILIES = ("integrator", "integrator_gradient", "integrator_with_gradient")


def _registered_integrator_types():
    """Map each integrator family (by algo_short) to the set of IntegratorType
    names registered for it in the class-level KERNEL_ATTR_MANIFEST."""
    by_family: dict[str, set[str]] = {}
    for entry in GRiDCodeGenerator.KERNEL_ATTR_MANIFEST:
        # entry = (algo_label, algo_short, gate_attr, bytes_macro, [(kernel_name, sig), ...])
        algo_short, kernels = entry[1], entry[4]
        if algo_short not in _INTEGRATOR_FAMILIES:
            continue
        its = by_family.setdefault(algo_short, set())
        for kernel_name, _sig in kernels:
            its.update(re.findall(r"IntegratorType::(\w+)", kernel_name))
    return by_family


def test_manifest_registers_every_emitted_integrator_type():
    expected = set(_INTEGRATOR_TYPES)
    by_family = _registered_integrator_types()
    for family in _INTEGRATOR_FAMILIES:
        assert family in by_family, (
            f"{family!r} integrator family is absent from KERNEL_ATTR_MANIFEST"
        )
        assert by_family[family] == expected, (
            f"{family} registers IntegratorTypes {sorted(by_family[family])} but the "
            f"codegen emits {sorted(expected)}. A kernel whose IT is emitted but NOT "
            f"registered launches with cudaErrorInvalidValue once its dynamic shared "
            f"memory exceeds 48 KB at an unspilled tier (the TRAPEZOIDAL launch-cliff)."
        )


def test_trapezoidal_is_registered():
    """Direct regression pin for the 2026-06-18 bug: TRAPEZOIDAL must be present
    in every non-mjx integrator family (it was historically omitted because the
    floating-base trapezoidal gradient was static_assert-refused)."""
    by_family = _registered_integrator_types()
    for family in _INTEGRATOR_FAMILIES:
        assert "TRAPEZOIDAL" in by_family.get(family, set()), (
            f"TRAPEZOIDAL missing from {family} KERNEL_ATTR_MANIFEST registration"
        )
