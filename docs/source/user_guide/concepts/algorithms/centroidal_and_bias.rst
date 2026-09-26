Bias terms, centroidal quantities and energy
============================================

.. currentmodule:: RBDReference.RBDReference

Overview
--------
Several quantities that controllers and identification pipelines need are
special cases or by-products of the recursive algorithms. GRiD exposes them as
first-class operations rather than asking users to assemble them from RNEA
calls:

* the **bias** (nonlinear effects) ``c(q, qd) = C(q, qd)·qd + g(q)`` and the
  **generalized gravity** ``g(q)``, both evaluated by RNEA with zero
  acceleration (and zero velocity for gravity);
* the **Coriolis matrix** ``C(q, qd)`` itself, with
  ``C·qd + g = nonlinear_effects``;
* the **centroidal** quantities: centre of mass and its Jacobian, the
  centroidal momentum matrix ``A(q)`` with the momentum ``h = A·qd``, the
  tensor ``∂A/∂q`` and the time variation ``Ȧ``;
* the **energies** (kinetic, potential, mechanical) and their
  inertial-parameter **regressors**, plus the inverse-dynamics regressor
  ``Y(q, qd, qdd)`` with ``tau = Y·π``.

Signature
---------
.. code-block:: python

   c    = h.nonlinear_effects(q, qd)       # (B, NV)
   g    = h.generalized_gravity(q)         # (B, NV)
   C    = h.coriolis_matrix(q, qd)         # (B, NV, NV)
   p, J = h.com(q)                         # (B, 3), (B, 3, NV)
   A, hm = h.ccrba(q, qd)                  # (B, 6, NV), (B, 6)
   dA   = h.dccrba(q)                      # (B, 6, NV, NV)
   Adot = h.cmm_time_variation(q, qd)      # (B, 6, NV)
   E    = h.energy(q, qd)                  # (B, 3): kinetic, potential, mechanical
   yKE  = h.kinetic_energy_regressor(q, qd)    # (B, 10*num_bodies)
   yPE  = h.potential_energy_regressor(q)      # (B, 10*num_bodies)
   Y    = h.inverse_dynamics_regressor(q, qd, qdd)   # (B, NV, 10*num_bodies)

The centroidal quantities follow the Pinocchio convention: ``[linear;
angular]`` at the centre of mass, world aligned. The ten inertial parameters
per body use GRiD's order
``[m, hx, hy, hz, Ixx, Ixy, Ixz, Iyy, Iyz, Izz]``: ``h = m*c`` and the
inertia is about the body-frame origin, not the centre of mass. This differs
from Pinocchio's ``toDynamicParameters`` (which places ``Iyy`` before
``Ixz``); do not multiply a GRiD regressor by an unconverted Pinocchio vector.
See :doc:`../../tutorials/verified_inputs` for a CPU-tested parameter-vector
construction and regressor identity check.

These examples use the NumPy handle and batched inputs. ``q`` is
``(B, h.nq)``; the default Pinocchio-convention dynamics calls shown here
expect ``qd``/``qdd`` padded to that width, with ``h.num_vel`` tangent entries
first. ``dccrba`` is indexed ``dA[b, row, column, tangent_direction]``.
Select the named operations in ``algorithm_list`` when loading the model;
the :doc:`Python wrapper guide <../../tutorials/python_wrappers>` describes
registration and operation discovery.

Implementation
--------------
The Python references are the ``_energy.py`` and ``_centroidal.py`` mixins of
RBDReference (``generalized_gravity``, ``nonlinear_effects``,
``coriolis_matrix``, ``com``, ``ccrba``, ``dccrba``, ``cmm_time_variation``,
the energies and regressors), each validated against Pinocchio. The CUDA
generators are the matching modules under ``grid_codegen/algorithms/``.

In GRiD
-------
The bias and gravity vectors have dedicated kernels that reuse the
inverse-dynamics inner recursion with acceleration (and velocity for gravity)
held at zero. They are not timings of the full inverse-dynamics API. The
Coriolis matrix and the centroidal quantities are their own kernels, and the
centroidal derivatives (``dccrba``, ``cmm_time_variation``) run on the big
floating-base humanoids through the sweep-pool spill path of the
:doc:`resource tier system <../resource_tier_system>`. All of them fold to the
reduced coordinates on mimic robots. Spill tiers extend coverage, but
generation and launch remain subject to the target GPU's resource limits.

The CUDA host entries carry the same names (``grid::nonlinear_effects``,
``grid::coriolis_matrix``, ``grid::ccrba`` and so on), each with a
``_compute_only`` variant. With ``output_convention="mujoco"`` the bias
matches MuJoCo's ``qfrc_bias`` (the floating-root acceleration couple is
injected and the base rows rotated in the kernel), the Coriolis matrix is the
congruence-transformed MuJoCo-frame matrix, and the centroidal momentum is
invariant while the matrix columns are reframed.

In the release benchmarks the bias and gravity vectors are where GRiD's
advantage over Pinocchio's code-generated C++ is smallest: they are the
cheapest operations, so the host↔device copies are a large share of the GPU
time. The centroidal momentum matrix is one operation where Pinocchio's
standard API beats GRiD's host call on the floating-base robots at every
measured batch size. That timing alone does not isolate the cause to output
size or transfer cost.

See Also
--------
* :doc:`inverse_dynamics` — the recursion these quantities come from.
* :doc:`integrators_and_plant` — the centre-of-mass and momentum tracking
  costs that consume ``com`` and ``ccrba``.
* :doc:`../mjx_convention` — the MuJoCo-convention views.
