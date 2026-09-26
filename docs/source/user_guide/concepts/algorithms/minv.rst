Minv (Direct Mass-Matrix Inverse)
=================================

.. currentmodule:: RBDReference.RBDReference

Overview
--------
The Direct Inverse of the Mass Matrix from
`Carpentier <https://www.researchgate.net/publication/343098270_Analytical_Inverse_of_the_Joint_Space_Inertia_Matrix>`__
computes :math:`M(q)^{-1}` directly without first forming :math:`M`.
This skips the :doc:`crba` step entirely and is the right choice when
forward dynamics is the only downstream consumer.

GRiD's standard forward-dynamics path composes ``minv`` with
:doc:`inverse_dynamics`:

.. math::

   \ddot{q} = M^{-1}(q) \cdot (\tau - c(q, \dot{q}))

The independent ABA path is exposed via :doc:`aba`.

Signature
---------
.. code-block:: python

   Minv = rbd.minv(q, output_dense=True)

Setting ``output_dense=False`` returns the algorithm's natural
LTL-factored output for callers who can consume it without
materializing the full dense matrix.

Implementation
--------------
The Python reference is ``RBDReference.minv`` in
`RBDReference/RBDReference.py
<https://github.com/A2R-Lab/RBDReference>`__. CUDA codegen lives in
`grid_codegen/algorithms/_minv.py
<https://github.com/A2R-Lab/GRiDCodeGenerator>`__.

In GRiD
-------
``Minv = h.minv(q)`` returns ``(B, NV, NV)``. The kernel writes the upper
triangle only; the handle symmetrises it on the host before returning, so the
result matches ``RBDReference.minv(..., output_dense=True)``. It is the
building block of ``h.forward_dynamics`` and of the forward-dynamics
gradients, and it is the operation to compare against libraries that expose
an inverse-inertia product directly (Frax in the release benchmarks). On a
robot with a spherical joint the CUDA path falls back to a dense inverse of
the composite-rigid-body matrix, matching Pinocchio's reduced model.

The CUDA host entries are ``grid::minv`` and ``minv_compute_only``. Because
``minv`` feeds every forward-dynamics derivative, its fp32 rounding is what
the release collection's accuracy policy is about: the forward-dynamics
family is validated against the fp64 oracle and cells that exceed the strict
entrywise gate are kept with their error reported.

See Also
--------
* :doc:`crba` — full mass matrix (use this if you need :math:`M`
  itself, not just :math:`M^{-1}`).
* :doc:`inverse_dynamics` — inverse dynamics (RNEA); the ``c`` term in
  the FD composition.
* :doc:`aba` — recursive forward dynamics (independent of
  :math:`M^{-1}`).
