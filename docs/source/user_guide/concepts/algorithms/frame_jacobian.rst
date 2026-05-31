Frame Jacobian (general-frame geometric Jacobian)
=================================================

.. currentmodule:: RBDReference.RBDReference

Overview
--------
The frame-Jacobian family computes the :math:`6 \times n_v` geometric
Jacobian :math:`J` of an arbitrary target frame — typically an
end-effector or any named joint frame — relating generalized velocity
:math:`\dot q` to the spatial velocity (twist) of that frame. Rows are
ordered ``[linear(3); angular(3)]`` to match Pinocchio's
``getFrameJacobian`` / ``getJointJacobian``.

The same family also provides two derived quantities in the numpy
reference:

* :math:`\dot J`, the Jacobian time-variation (so that the frame
  acceleration is :math:`J\,\ddot q + \dot J\,\dot q`).
* :math:`\Lambda = (J\,M^{-1}\,J^{\top})^{-1}`, the :math:`6 \times 6`
  operational-space (OSC) inertia of the task frame. :math:`\Lambda`
  is only defined when the frame is reachable by at least six
  independent DOFs (the task matrix :math:`J\,M^{-1}\,J^{\top}` must be
  well-conditioned).

Reference frames
----------------
:math:`J` (and :math:`\dot J`, :math:`\Lambda`) can be expressed in any
of the three Pinocchio reference-frame conventions, selected by a
``reference_frame`` argument:

* ``LOCAL`` (``0``) — twist in the frame's own body axes.
* ``WORLD`` (``1``) — spatial Jacobian at the world origin.
* ``LOCAL_WORLD_ALIGNED`` (``2``) — at the frame origin, with
  world-aligned axes (the default).

Signature
---------
.. code-block:: python

   # general-frame geometric Jacobian (6 x nv)
   J      = rbd.frame_jacobian(q, frame_name, reference_frame="LOCAL_WORLD_ALIGNED")
   # Jacobian time-variation (6 x nv)
   Jdot   = rbd.frame_jacobian_dot(q, qd, frame_name, reference_frame="LOCAL_WORLD_ALIGNED")
   # operational-space (OSC) inertia (6 x 6)
   Lambda = rbd.osc_inertia(q, frame_name, reference_frame="LOCAL_WORLD_ALIGNED")

Validation (Pinocchio oracle)
-----------------------------
The numpy reference is validated against Pinocchio across all three
reference frames on both a fixed-base (iiwa14) and a floating-base
(go2) robot:

* :math:`J` vs ``getFrameJacobian`` / ``getJointJacobian``.
* :math:`\dot J` vs ``getFrameJacobianTimeVariation`` /
  ``getJointJacobianTimeVariation`` (via
  ``computeJointJacobiansTimeVariation``).
* :math:`\Lambda` vs ``(J\,M^{-1}\,J^{\top})^{-1}`` built from
  Pinocchio's ``computeMinverse``.

See ``RBDReference/tests/test_frame_jacobian_equivalence.py``.

Implementation
--------------
The Python reference is ``RBDReference.frame_jacobian`` /
``.frame_jacobian_dot`` / ``.osc_inertia`` in
`RBDReference/RBDReference.py
<https://github.com/A2R-Lab/RBDReference>`__.

CUDA codegen lives in
`GRiDCodeGenerator/algorithms/_frame_jacobian.py
<https://github.com/A2R-Lab/GRiDCodeGenerator>`__. It is an
**opt-in, additive** family: the kernels are only emitted when the
``frame_jacobian`` key is explicitly selected (it pulls in the
``ee_pose`` world-transform machinery), so existing profiles' headers
are byte-identical. The CUDA path emits the geometric Jacobian
:math:`J` (all three reference frames) via ``frame_jacobian_inner`` /
``frame_jacobian_device``, the Jacobian time-variation :math:`\dot J`
via ``frame_jacobian_dot_device`` (opt-in ``frame_jacobian_dot`` key),
and the OSC inertia :math:`\Lambda` via ``osc_inertia_device`` (opt-in
``osc_inertia`` key). All three are validated on-device against the numpy
reference across the three reference frames. The :math:`\Lambda` kernel
currently takes a precomputed :math:`M^{-1}` as input (the smoke runner
feeds it from ``direct_minv_device``); folding ``direct_minv_inner`` into
``osc_inertia_device`` so it composes :math:`M^{-1}` on-device is a
follow-up. Mimic-joint robots are not yet supported on the CUDA
frame-Jacobian path.

See Also
--------
* :doc:`crba` — joint-space mass matrix :math:`M` (used to form
  :math:`\Lambda`).
* :doc:`minv` — direct :math:`M^{-1}` (the inverse used inside
  :math:`J\,M^{-1}\,J^{\top}`).
* :doc:`rnea` — inverse dynamics.
