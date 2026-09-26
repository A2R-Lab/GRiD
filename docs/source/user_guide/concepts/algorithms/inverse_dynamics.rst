inverse_dynamics (RNEA / Recursive Newton-Euler Algorithm)
==========================================================

.. currentmodule:: RBDReference.RBDReference

Overview
--------
``inverse_dynamics`` computes the inverse dynamics of a robot using the
Recursive Newton-Euler Algorithm (RNEA) — given joint positions,
velocities, and accelerations, it returns the joint torques required to
produce them. GRiD also exposes the per-pass helpers
(``inverse_dynamics_fpass`` / ``inverse_dynamics_bpass``) for downstream
accelerator pieces that need access to the spatial velocity /
acceleration / force intermediates.

Signature
---------
.. code-block:: python

   (c, v, a, f) = rbd.inverse_dynamics(q, qd, qdd=None, GRAVITY=-9.81)

``GRAVITY`` is the signed gravitational acceleration; the default
``-9.81`` is standard downward gravity (matching pinocchio /
RBDReference). If ``qdd`` is omitted, ``inverse_dynamics`` returns the
bias term (Coriolis + gravity) used by the forward dynamics composition
``qdd = Minv·(τ − c)``.

The gradient
~~~~~~~~~~~~
First-order gradients are available as
``rbd.inverse_dynamics_gradient(q, qd, qdd, GRAVITY=-9.81)``, returning
``np.hstack((dc_dq, dc_dqd))``. The second-order tensors are exposed
through :doc:`idsva` (``idsva_so_body_frame`` / ``idsva_so_world_frame``
/ the ``idsva_so`` dispatcher).

Implementation
--------------
The Python reference is ``RBDReference.inverse_dynamics`` in
`RBDReference/RBDReference.py
<https://github.com/A2R-Lab/RBDReference>`__. CUDA codegen lives in
`grid_codegen/algorithms/_inverse_dynamics.py
<https://github.com/A2R-Lab/GRiDCodeGenerator>`__.

In GRiD
-------
On the NumPy handle, ``tau = h.inverse_dynamics(q, qd, qdd)`` returns
``(B, h.nq)``, with physical torques in the first ``NV = h.num_vel`` slots
and padding in the remaining slots. On the default Pinocchio-convention
path, ``q``, ``qd`` and ``qdd`` all use ``(B, h.nq)`` storage; velocity and
acceleration have tangent entries first and zero padding. Matrix and
derivative outputs are instead ``NV``-wide. See :doc:`../input_output_abi`
and :doc:`../../tutorials/python_wrappers`. With ``qdd``
omitted it returns the bias ``c(q, qd) = C(q, qd)·qd + g(q)``, which is also
available as ``h.nonlinear_effects``; ``h.generalized_gravity`` is the
``qd = 0`` special case. Optional per-body external forces (``f_ext``,
``(B, 6*num_bodies)``, body-major, ``[angular; linear]`` in the body frame)
are subtracted from the per-body force, and the signed gravity defaults to
``-9.81``.

Derivatives: ``h.inverse_dynamics_gradient`` returns ``∂τ/∂(q, qd)`` as
``(B, NV, 2*NV)`` in the tangent space, and :doc:`idsva` provides the
second-order tensors. Inverse dynamics also anchors the inertial-parameter
regressor ``Y(q, qd, qdd)`` with ``tau = Y·π`` (``h.inverse_dynamics_regressor``)
and a generated CUDA regressor-gradient operation. The latter is not a
method on the NumPy ``RobotHandle``.

The CUDA host entries are ``grid::inverse_dynamics`` and
``inverse_dynamics_compute_only``. RNEA is a short-running operation in the
release collection, so dispatch and transfers can be substantial relative
to compute time. The choice of C++ host call, NumPy, PyTorch or JAX surface
is therefore part of the performance comparison, not just a syntax choice.

See Also
--------
* :doc:`crba` — composite-rigid-body mass matrix.
* :doc:`aba` — recursive forward dynamics counterpart.
* :doc:`minv` — direct mass-matrix inverse (the FD composition partner).
* :doc:`idsva` — second-order inverse dynamics (IDSVA-SO).
