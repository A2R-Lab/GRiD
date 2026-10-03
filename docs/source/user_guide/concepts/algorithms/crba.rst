CRBA (Composite Rigid Body Algorithm)
=====================================

.. currentmodule:: RBDReference.RBDReference

Overview
--------
The Composite Rigid Body Algorithm (CRBA) computes the joint-space
mass matrix :math:`M(q)`. It does so by recursively combining
body inertias along the kinematic tree.

CRBA is one of two GRiD paths for getting mass-matrix information:

* **CRBA** produces the full :math:`M`, useful when downstream
  algorithms need the dense matrix (e.g. operational-space inverse
  dynamics, Cholesky-based forward-dynamics solves).
* :doc:`minv` produces :math:`M^{-1}`, normally without forming :math:`M`
  first. Mimic and spherical models instead use a dense CRBA-based inverse.
  Forward dynamics can use Minv/RNEA composition or the independent :doc:`aba`.

Signature
---------
.. code-block:: python

   M = rbd.crba(q)

Implementation
--------------
The Python reference is ``RBDReference.crba`` in
`RBDReference/RBDReference.py
<https://github.com/A2R-Lab/RBDReference>`__. CUDA codegen lives in
`grid_codegen/algorithms/_crba.py
<https://github.com/A2R-Lab/GRiD/tree/main/grid_codegen>`__.

In GRiD
-------
From the Python handles, ``M = h.crba(q)`` returns ``(B, NV, NV)``, the
tangent-space mass matrix in the Pinocchio convention. For a fixed base
with independent scalar joints, ``NV`` equals the joint count. A floating
root contributes six tangent and seven position coordinates; spherical and
mimic joints require the model's coordinate maps rather than joint counts.
Pass ``q`` as ``(B, h.nq)`` and obtain ``NV`` from ``h.num_vel``.
The result does not depend on ``gravity=-9.81``; the keyword only mirrors
the host signature. With ``output_convention="mujoco"`` the input is
MuJoCo-convention and the matrix comes back in the MuJoCo frame, computed in
the kernel.

The generated CUDA host entry is ``grid::crba`` (host arrays in, host arrays
out, copies included) with a ``crba_compute_only`` variant that runs the kernel
alone on data already resident on the GPU; see
:doc:`../../tutorials/codegen` for the host-call pattern. The dense matrix is
what operational-space formulations and Cholesky-based solves consume.
Mimic and spherical joints, arbitrary axes and the floating
base are supported; per-robot caveats are listed on the
:doc:`support matrix <../../tutorials/cuda_support_status>`.

See Also
--------
* :doc:`minv` — direct mass-matrix inverse (skip ``crba`` if you
  only need :math:`M^{-1}` for forward dynamics).
* :doc:`inverse_dynamics` — inverse dynamics (RNEA).
* :doc:`aba` — recursive forward dynamics (no explicit mass matrix).
