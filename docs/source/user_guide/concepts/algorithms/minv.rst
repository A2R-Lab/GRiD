Minv (Direct Mass-Matrix Inverse)
=================================

.. currentmodule:: RBDReference.RBDReference

Overview
--------
The Direct Inverse of the Mass Matrix from
`Carpentier <https://hal.science/hal-01790934>`__
computes :math:`M(q)^{-1}` directly without first forming :math:`M` on
the scalar-joint non-mimic path. Mimic and spherical models instead form
the reduced :doc:`crba` matrix and invert it densely.

GRiD's standard forward-dynamics path composes ``minv`` with
:doc:`inverse_dynamics`:

.. math::

   \ddot{q} = M^{-1}(q) \cdot (\tau - c(q, \dot{q}))

The independent ABA path is exposed via :doc:`aba`.

Signature
---------
.. code-block:: python

   Minv = rbd.minv(q, output_dense=True)

On the direct scalar-joint path, ``output_dense=False`` leaves only the
upper triangle of the inverse populated; it does not return an LTL
factorization. Dense fallback paths can return the full inverse regardless
of this option.

Implementation
--------------
The Python reference is ``RBDReference.minv`` in
`RBDReference/RBDReference.py
<https://github.com/A2R-Lab/RBDReference>`__. CUDA codegen lives in
`grid_codegen/algorithms/_minv.py
<https://github.com/A2R-Lab/GRiD/tree/main/grid_codegen>`__.

In GRiD
-------
``Minv = h.minv(q)`` takes ``q`` of shape ``(B, h.nq)`` and returns
``(B, h.num_vel, h.num_vel)``. On the default Pinocchio-convention path the
kernel writes the upper triangle; the NumPy handle symmetrises it before returning, so the
result matches ``RBDReference.minv(..., output_dense=True)``. It is the
building block of ``h.forward_dynamics`` and of the forward-dynamics
gradients, and it is the operation to compare against libraries that expose
an inverse-inertia product directly (Frax in the release benchmarks). On a
robot with mimic or spherical joints the CUDA path falls back to a dense inverse of
the composite-rigid-body matrix, matching Pinocchio's reduced model.

The CUDA host entries are ``grid::minv`` and ``minv_compute_only``. Because
``minv`` is used in the forward-dynamics derivative composition, its rounding
can contribute to fp32 discrepancies. The release collection retains an
entrywise-failing fp32 forward-dynamics-family cell only if every checked
block satisfies the additional 0.1 % relative-L2 gate. Other failures remain
excluded. The policy does not establish a universal entrywise error bound.

See Also
--------
* :doc:`crba` — full mass matrix (use this if you need :math:`M`
  itself, not just :math:`M^{-1}`).
* :doc:`inverse_dynamics` — inverse dynamics (RNEA); the ``c`` term in
  the FD composition.
* :doc:`aba` — recursive forward dynamics (independent of
  :math:`M^{-1}`).
