CUDA Support Status
===================

This page summarizes the generated CUDA paths that are currently exercised by
the GRiD developer test suite. For the commands that run these checks, see
:doc:`cuda_validation`.

Fixed-Base Robots
-----------------

Fixed-base CUDA coverage includes the core dynamics and kinematics paths:

* Inverse dynamics / RNEA, direct Minv, forward dynamics, ABA, and CRBA.
* Inverse- and forward-dynamics gradients.
* End-effector pose, gradient, and Hessian.
* Fixed-base forced-fallback coverage for oversized gradient kernels.
* IDSVA-SO/FDSVA-SO zero-sample diagnostics for selected fixed-base robots.

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

* Floating IDSVA-SO/FDSVA-SO CUDA generation is intentionally disabled while
  fixed-base second-order coverage is still being broadened.
* Broad nonzero/random floating Hessian coverage is slower than the default
  smoke suite and remains opt-in.
* Compute Sanitizer should be run on a supported GPU/driver setup before
  treating fallback paths as fully sanitizer-clean.
* Performance tier choices can depend on register pressure and occupancy; use
  ptxas output and timing kernels on the target GPU before saving local
  baselines.
