RNEA (Recursive Newton-Euler Algorithm)
=======================================

.. currentmodule:: RBDReference.RBDReference

Overview
--------
The Recursive Newton-Euler Algorithm (RNEA) computes the inverse
dynamics of a robot — given joint positions, velocities, and
accelerations, it returns the joint torques required to produce them.
GRiD also exposes the per-pass helpers (``rnea_fpass`` / ``rnea_bpass``)
for downstream accelerator pieces that need access to the spatial
velocity / acceleration / force intermediates.

Signature
---------
.. code-block:: python

   (c, v, a, f) = rbd.rnea(q, qd, qdd=None, GRAVITY=-9.81)

If ``qdd`` is omitted, RNEA returns the bias term (Coriolis + gravity)
used by the Forward Dynamics composition ``qdd = Minv·(τ − c)``.

The gradient
~~~~~~~~~~~~
First-order gradients of RNEA are available as
``rbd.rnea_grad(q, qd, qdd, GRAVITY=-9.81)``, returning
``np.hstack((dc_dq, dc_dqd))``. The second-order tensors are exposed
through :doc:`idsva` (``idsva_so_body_frame`` / ``idsva_so_world_frame``
/ the ``idsva_so`` dispatcher).

Implementation
--------------
The Python reference is ``RBDReference.rnea`` in
`RBDReference/RBDReference.py
<https://github.com/A2R-Lab/RBDReference>`__. CUDA codegen lives in
`GRiDCodeGenerator/algorithms/_inverse_dynamics.py
<https://github.com/A2R-Lab/GRiDCodeGenerator>`__.

See Also
--------
* :doc:`crba` — composite-rigid-body mass matrix.
* :doc:`aba` — recursive forward dynamics counterpart.
* :doc:`minv` — direct mass-matrix inverse (the FD composition partner).
* :doc:`idsva` — second-order RNEA.
