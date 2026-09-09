The MuJoCo (mjx) output convention
==================================

**The canonical page** for GRiD's MuJoCo-convention support: what differs,
what transforms, where it applies, and what it costs. The mathematical
derivation with its machine-precision validation lives in
``external/RBDReference/equivalents/mujoco_convention.md`` (the transforms
themselves in ``mujoco_convention.py``); the per-method API tour is in
:doc:`../tutorials/python_wrappers`.

The two raw differences
-----------------------

GRiD is natively **Pinocchio**-convention. MuJoCo differs for the
free-floating base in exactly two ways:

1. **Quaternion order.** MuJoCo ``qpos`` stores the free-flyer quaternion
   **wxyz** (scalar first); pin/GRiD use **xyzw**. A pure relabel on ``q``
   and integrator outputs — never touches velocities, forces, or gradients.
2. **Free-joint velocity frame.** MuJoCo ``qvel`` is
   ``[v_lin GLOBAL ; omega LOCAL]``; the pin spatial twist is
   ``[v_lin LOCAL ; omega LOCAL]``. That is ONE root-block basis change
   ``G(q) = blockdiag(R, I3)`` on the leading 6 tangent DOF (``R`` = base
   orientation). ``G`` is orthogonal: ``G^{-1} = G^T``.

Everything else — every joint DOF past the root, every fixed-base robot — is
IDENTICAL between the conventions. That is why ``output_convention="mujoco"``
is accepted (as a no-op) on fixed-base robots: generic code can set it
everywhere.

Value and derivative transforms
-------------------------------

Velocities transform by ``G``, forces (covectors) by ``G^{-T} = G``, the mass
matrix by congruence ``G M G^T``, and each derivative index contracts with
the matching ``G`` factor (see the derivation doc for the full table,
including the subtle dR/dq chart-slope terms in the second-order tensors).

**Do not "debug" the reframe.** If an mjx output looks wrong, check the pin
baseline FIRST — pin↔mjx is a KNOWN, validated transform, and every mjx bug
so far was actually a pin bug or a caller-side convention mixup
(``docs/agent_debugging_guide.md`` §1k).

How GRiD serves it: native kernel twins
---------------------------------------

For a **floating-base robot without mimic joints or skew axes**, codegen
emits an mjx twin of each kernel (the ``MUJOCO_OUTPUT=true`` template
instantiation) with the input conversion + output reframe fused in-kernel —
no host-side transform, no extra copies. Values AND the derivative /
second-order surfaces are served natively this way.

Twins are **never emitted** for fixed-base or mimic/skew robots (fixed base:
the conventions coincide; mimic/skew: the reduced-coordinate reframe is not
implemented). On such a ``.so`` the mjx-specific methods raise a clear error
naming the cause.

Three ways to ask for it
------------------------

- ``handle.mujoco.<method>(...)`` — per-call view, thread-safe, never mutates
  shared state (recommended).
- ``handle.output_convention = "mujoco"`` — the handle-wide default.
- ``register_robot(..., output_convention="mujoco")`` — the same default at
  registration. Runtime-only: NOT in the ``.so`` cache key.

What it costs (and how to opt out)
----------------------------------

The mjx twins — not second-order kernel size — dominate humanoid build cost:
the second-order twins are the largest kernels in a big floating-base build
(``idsva_so_world_frame``'s twin was 28× its pin kernel raw; block-
parallelizing the epilogue cut it to ~2.4×). If you only need
Pinocchio-convention outputs, build pin-only with
``enable_mujoco_kernels=False`` (or ``grid-generate --no-mujoco-kernels``) —
on a large robot this is the difference between building and running out of
memory. A pin-only floating build then refuses ``output_convention="mujoco"``
with a clear error at registration time.

The known-parity contract
-------------------------

``mujoco_convention.py``'s transforms are validated to machine precision
against a real MuJoCo (``mj_fullM`` / ``mj_inverse`` / ``mj_forward``) and by
finite-difference self-consistency; the CUDA twins are validated against that
reference in the equivalence suites. The single source of truth for the
convention mapping is ``mujoco_convention.py`` — both the tests and the
codegen input-conversion mirror it.
