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

See Also
--------
* :doc:`crba` — composite-rigid-body mass matrix.
* :doc:`aba` — recursive forward dynamics counterpart.
* :doc:`minv` — direct mass-matrix inverse (the FD composition partner).
* :doc:`idsva` — second-order inverse dynamics (IDSVA-SO).
