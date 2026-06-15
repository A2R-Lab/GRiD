CUDA Support Status
===================

This page summarizes the generated CUDA paths that are currently exercised by
the GRiD developer test suite. For the commands that run these checks, see
:doc:`cuda_validation`.

Development testing currently targets **sm_120 (RTX 5090)** for correctness
and performance validation. Earlier compute capabilities (sm_8x) remain
supported but are not the active development target.

Fixed-Base Robots
-----------------

Fixed-base CUDA coverage includes the core dynamics and kinematics paths:

* Inverse dynamics / RNEA, direct Minv, forward dynamics, ABA, and CRBA.
* Inverse- and forward-dynamics gradients.
* End-effector pose, gradient, and Hessian.
* Fixed-base forced-fallback coverage for oversized gradient kernels.
* IDSVA-SO (body-frame and world-frame variants) and FDSVA-SO. Body-frame
  is selected by the dispatcher for fixed-base because it wins by a wide
  margin (multi-pass amortizes when the tree is fixed).
* Optional per-body external forces (``d_f_ext``) on RNEA, forward
  dynamics, ABA, and the inverse-/forward-dynamics gradients (opt-in;
  ``nullptr`` reproduces the no-force path).
* The ``grid_plant`` layer (``plant_step``, quadratic state/input costs,
  end-effector position cost, and joint position/velocity/torque
  log-barriers), emitted as a sibling ``grid_plant`` namespace.
* The centroidal family and its derivatives: ``com``, ``ccrba``, ``energy``,
  ``dccrba`` (∂A/∂q tensor), and ``cmm_time_variation`` (Ȧ).
* The ``coriolis_matrix`` ``C(q,q̇)`` and the kinetic / potential
  inertial-parameter energy regressors.
* Runtime arbitrary multi-EE: ``end_effector_pose_runtime`` and
  ``end_effector_pose_gradient_runtime`` (runtime target joint id + per-target
  offset).
* Optional runtime-mutable inertial parameters (flag-gated ``d_inertia_params``
  + ``set_inertia_params``; the baked default is byte-identical).
* Arbitrary/skew joint ``<axis>`` (dense 6-vector ``S``) for
  ``inverse_dynamics`` and ``crba`` (stage 1; cardinal axes byte-identical).

Second-order fixed-base diagnostics are still developer-only. The current
green zero-sample set includes ``iiwa14``, ``go2``, ``gen3``, ``fr3``, and
``fetch``. ``rizon4`` still needs diagnosis for a non-finite IDSVA-SO output.

Floating-Base Robots
--------------------

Floating-base CUDA coverage includes:

* Inverse dynamics / RNEA.
* Direct Minv, forward dynamics, ABA, and CRBA.
* Inverse- and forward-dynamics gradients.
* End-effector pose.
* Opt-in end-effector pose gradient and Hessian checks.
* IDSVA-SO (world-frame variant, dispatcher-selected) and FDSVA-SO.
  World-frame wins by 2–4× on floating-base because the single-pass
  formulation avoids the body-frame gravity shim and the floating chain
  is deep enough that the body-frame subtree-broadcast no longer
  amortizes.
* The centroidal family ``com`` / ``ccrba`` / ``energy`` and the centroidal
  derivatives ``dccrba`` / ``cmm_time_variation``. The two derivatives also
  emit on big floating-base robots (e.g. ``g1`` / ``h1_2``-floating) via the
  sweep-pool spill path, so their prior big-floating gap is eliminated.

Floating end-effector Hessian generation uses target-aware spill tiers when
needed:

* Tier 0 keeps all Hessian scratch in shared memory.
* Tier 1 spills chained ``d2eeTemp`` scratch to ``d_workspace``.
* Tier 2 spills both d2XHom and ``d2eeTemp`` to ``d_workspace``.

Tier selection is generated from robot dimensions, base mode, topology-derived
transform counts, and ``GRID_CUDA_TARGET_SHARED_MEM_BYTES``. It is not keyed on
robot fixture names.

Shared-Memory Fallbacks
-----------------------

Generated kernels prefer the all-shared path when it fits the target shared
memory budget. The default target is 96 KiB:

.. code-block:: bash

   GRID_CUDA_TARGET_SHARED_MEM_BYTES=98304

Set a lower target to test fallback paths, or a higher target only when the
deployment GPU supports the requested dynamic shared memory. Runtime checks
compare generated requests against the actual device limit before launch.

Gradient fallback tiers keep the hot ``dv/*`` derivative intermediates in
shared memory first, then spill larger ``da/*`` and ``df/*`` buffers when the
full shared-memory path is oversized. The emergency tier spills more temporary
state to the generated workspace.

Known Caveats
-------------

* Floating IDSVA-SO/FDSVA-SO CUDA generation is now enabled (dispatcher
  picks ``idsva_so_world_frame`` for floating-base). FDSVA-SO on
  ``g1_floating`` requires the selective-spill tier under sm_120's
  ~100 KiB per-block dynamic shared-memory cap; the tier selector picks
  it automatically.
* Broad nonzero/random floating Hessian coverage is slower than the default
  smoke suite and remains opt-in.
* Compute Sanitizer should be run on a supported GPU/driver setup before
  treating fallback paths as fully sanitizer-clean.
* Performance tier choices can depend on register pressure and occupancy; use
  ptxas output and timing kernels on the target GPU before saving local
  baselines. Tiers are named ``TIER_SHARED`` (default) / ``TIER_LITE`` /
  ``TIER_MINIMAL`` (the old ``TIER_PERF`` alias has been removed — use
  ``TIER_SHARED``). See :doc:`../concepts/resource_tier_system`.
* Robots with **mimic joints**: non-gradient algorithms are supported, and
  **every** gradient now emits a correct mimic-reduced result on both bases —
  ``inverse_dynamics_gradient`` / ``forward_dynamics_gradient``,
  ``end_effector_pose_gradient`` / ``end_effector_pose_hessian``, the second-order
  ``idsva_so`` / ``fdsva_so``, the external-force gradients (``f_ext_gradient``),
  and the integrator gradients. No mimic gradient raises ``NotImplementedError``
  anymore. The centroidal kinematics family
  (``com`` / ``ccrba`` / ``energy``) and the centroidal derivatives
  (``dccrba`` / ``cmm_time_variation``) are now mimic-reduced as well: the
  per-body world Jacobian and per-unit motion columns carry the mimic
  multiplier (α), so all five emit + validate against the mimic-aware
  RBDReference oracle on fixed-base mimic robots.
* External-force **gradients**: ``f_ext_gradient``
  (∂τ/∂f_ext = −Jᵀ, ∂q̈/∂f_ext = M⁻¹Jᵀ) and the fixed-base ``f_ext_gradient_dq``
  (−∂Jᵀ/∂q), both with CUDA equivalence tests, and both now fold correctly to the
  reduced coordinates on mimic robots as well.
