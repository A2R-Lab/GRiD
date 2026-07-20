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
* :doc:`minv` produces :math:`M^{-1}` directly without forming
  :math:`M` first — the right choice when forward dynamics is the
  only downstream consumer.

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
<https://github.com/A2R-Lab/GRiDCodeGenerator>`__.

See Also
--------
* :doc:`minv` — direct mass-matrix inverse (skip ``crba`` if you
  only need :math:`M^{-1}` for forward dynamics).
* :doc:`inverse_dynamics` — inverse dynamics (RNEA).
* :doc:`aba` — recursive forward dynamics (no explicit mass matrix).
