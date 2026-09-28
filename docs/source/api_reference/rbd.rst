RBDReference
============

RBDReference provides CPU implementations of dynamics, kinematics, derivatives
and integration for prototyping and checking generated CUDA. It lives in
``external/RBDReference``; it is not the GPU runtime wrapper.

Quick start
-----------

After the :doc:`source installation <../user_guide/getting_started/installation>`,
run from the GRiD repository root:

.. code-block:: python

   import numpy as np
   from URDFParser import URDFParser
   from RBDReference import RBDReference

   robot = URDFParser().parse("config/robot_assets/iiwa14.urdf")
   rbd = RBDReference(robot)
   q = np.zeros(robot.get_num_pos())  # this example is fixed-base
   qd = np.zeros(robot.get_num_vel())
   tau, v, a, f = rbd.inverse_dynamics(q, qd)
   M = rbd.crba(q)
   qdd = rbd.forward_dynamics(q, qd, tau)

Configurations have width ``NQ``; velocities, forces and tangent perturbations
have width ``NV``. Floating and spherical joints use quaternions, so an
all-zero configuration is not valid for those models. Use ``integrate`` and
``difference`` for configuration perturbations and errors.

Dynamics and derivatives
------------------------

.. list-table::
   :header-rows: 1
   :widths: 40 60

   * - Operation
     - Call / result
   * - Inverse dynamics (RNEA)
     - ``tau, v, a, f = rbd.inverse_dynamics(q, qd, qdd=None)``
   * - Forward dynamics
     - ``rbd.forward_dynamics(q, qd, u)`` or ``rbd.aba(q, qd, u)``
   * - Mass matrix and inverse
     - ``rbd.crba(q)``, ``rbd.minv(q, output_dense=True)``
   * - Inverse-dynamics gradient
     - ``rbd.inverse_dynamics_gradient(q, qd, qdd=None)``; concatenated configuration-tangent and velocity blocks
   * - Forward-dynamics gradient
     - ``dqdd_dq, dqdd_dqd = rbd.forward_dynamics_gradient(q, qd, u)``
   * - Second-order inverse dynamics
     - ``rbd.idsva_so(q, qd, qdd)``; returns four tensors, including ``dM_dq``
   * - Second-order forward dynamics
     - ``rbd.fdsva_so(q, qd, u)``

``idsva_so`` selects the body-frame implementation for fixed-base models
and the world-frame implementation for floating-base models. This dispatch
is not a universal performance ranking. See
:doc:`../user_guide/concepts/algorithms/idsva`.

Kinematics, centroidal quantities and energy
------------------------------------------------

* ``rbd.end_effector_pose(q)`` returns end-effector poses;
  ``end_effector_pose_gradient(q)`` returns geometric Jacobians with tangent
  columns. ``end_effector_pose_hessian(q)`` is a finite-difference reference
  with tangent-space shape ``(6, NV, NV)`` per end effector; the separate
  ``end_effector_pose_hessian_analytic`` supplies the analytical path.
* ``rbd.frame_jacobian(q, frame_name, reference_frame)`` and
  ``frame_jacobian_dot(q, qd, ...)`` support general frames.
* ``rbd.com(q)`` returns a position of shape ``(3,)``;
  ``rbd.jacobian_com(q)`` returns a ``(3, NV)`` Jacobian.
* ``A, h = rbd.ccrba(q, qd)`` returns the centroidal momentum matrix and
  momentum. ``dccrba(q)`` differentiates the matrix;
  ``cmm_time_variation(q, qd)`` returns its time derivative.
* Gravity, nonlinear effects, Coriolis matrix, energy and inertial-parameter
  regressors are described in :doc:`../user_guide/concepts/algorithms/centroidal_and_bias`.

State operations and plant
--------------------------

``integrate(q, delta)`` retracts a tangent perturbation; ``difference(q_from,
q_to)`` returns a tangent error. ``integrator(q, qd, u, dt, integrator_type=...)``
supports Euler, semi-implicit Euler, constant acceleration, midpoint, Heun
(``trapezoidal``), and RK4. See
:doc:`../user_guide/concepts/algorithms/integrators_and_plant` for orders, supported
derivatives and plant costs.

Source and validation
---------------------

The `RBDReference README <https://github.com/A2R-Lab/RBDReference#readme>`_
lists the full method families and standalone installation instructions.
Signatures and per-pass helpers are in ``RBDReference.py`` and its topic
mixins. Reference availability does not imply support on every GPU surface;
check :doc:`../user_guide/tutorials/backend_coverage`.

For tests and numerical checks, see :doc:`../user_guide/tutorials/cuda_validation`
and ``external/RBDReference/tests/README.md``.
