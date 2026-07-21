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

See ``external/RBDReference/tests/test_frame_jacobian_equivalence.py``.

Implementation
--------------
The Python reference is ``RBDReference.frame_jacobian`` /
``.frame_jacobian_dot`` / ``.osc_inertia`` in
`RBDReference/RBDReference.py
<https://github.com/A2R-Lab/RBDReference>`__.

CUDA codegen lives in
`grid_codegen/algorithms/_frame_jacobian.py
<https://github.com/A2R-Lab/GRiDCodeGenerator>`__. It is an
**opt-in, additive** family: the surfaces are only emitted when the
``frame_jacobian`` key is explicitly selected (it pulls in the
``end_effector_pose`` world-transform machinery), so existing profiles' headers
are byte-identical. The CUDA path emits the geometric Jacobian
:math:`J` (all three reference frames) via ``frame_jacobian_inner`` /
``frame_jacobian_device``, the Jacobian time-variation :math:`\dot J`
via ``frame_jacobian_dot_device`` (opt-in ``frame_jacobian_dot`` key),
and the OSC inertia :math:`\Lambda` via ``osc_inertia_device`` (opt-in
``osc_inertia`` key). The :math:`\Lambda` device is **self-contained**: it
composes :math:`M^{-1}` on-device via ``minv_inner`` (no external
:math:`M^{-1}` feed). Mimic-joint robots ARE supported (the geometric-Jacobian
column fold is alpha-weighted onto the shared velocity slot; :math:`\Lambda`
routes the mimic :math:`M^{-1}` through ``crba_inner``).

**Full launchable surface (S1).** ``frame_jacobian`` now has the same surface
set as the other benchmarkable kinematics algorithms: a batched ``__global__``
``frame_jacobian_kernel`` (plus a ``_single_timing`` variant) and a 3-mode
``__host__`` launcher ``frame_jacobian`` / ``frame_jacobian_single_timing`` /
``frame_jacobian_compute_only`` that reads/writes the ``gridData`` output buffer
``hd_data->d_frame_jacobian`` (6 × NUM_VEL, copied back into
``h_frame_jacobian``). The launchable surface bakes a fixed frame target (the
leaf-EE joint) and the ``LOCAL_WORLD_ALIGNED`` reference frame (mirroring the
fixed-target ``end_effector_pose`` pattern); use ``frame_jacobian_device``
directly for an arbitrary ``target_jid`` / ``reference_frame`` at runtime. The
host surface is validated end-to-end against the numpy oracle in
``test/cuda_equivalents/test_cuda_frame_jacobian_host.py`` (the device functions
are covered by ``test_cuda_frame_jacobian.py``).

``frame_jacobian_dot`` (opt-in ``frame_jacobian_dot`` key) now has the same
launchable set: ``frame_jacobian_dot_kernel`` (+ ``_single_timing``) and the
3-mode host ``frame_jacobian_dot`` / ``_single_timing`` / ``_compute_only``,
reading the packed ``[q; qd]`` input and writing
``hd_data->d_frame_jacobian_dot`` (6 × NUM_VEL). The kernel keeps its input /
output in static ``__shared__`` because ``frame_jacobian_dot_device`` owns the
whole dynamic-smem arena; the same fixed leaf-EE / ``LOCAL_WORLD_ALIGNED``
default applies. It is covered by the same host-surface test (validated for
iiwa14-fixed + go2-floating).

``osc_inertia`` (opt-in ``osc_inertia`` key) likewise gains
``osc_inertia_kernel`` (+ ``_single_timing``) and the 3-mode host
``osc_inertia`` / ``_single_timing`` / ``_compute_only``, writing
``hd_data->d_osc_inertia`` (6 × 6). It is **self-contained** (q-only input;
composes :math:`M^{-1}` on-device), keeps its q input + Λ output in static
``__shared__``, and carries ``__launch_bounds__`` so nvcc fits the heavy
``minv``/``J``/``invert`` register footprint to the tier thread cap (the
un-annotated device smoke runner instead clamps threads manually). Same
host-surface test coverage (Λ checked at non-singular configs).

See Also
--------
* :doc:`crba` — joint-space mass matrix :math:`M` (used to form
  :math:`\Lambda`).
* :doc:`minv` — direct :math:`M^{-1}` (the inverse used inside
  :math:`J\,M^{-1}\,J^{\top}`).
* :doc:`inverse_dynamics` — inverse dynamics (RNEA).
