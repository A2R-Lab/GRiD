ABA (Articulated Body Algorithm)
================================

.. currentmodule:: RBDReference.RBDReference

Overview
--------
The Articulated Body Algorithm (ABA) is Featherstone's recursive
forward-dynamics algorithm: given joint positions, velocities, and
applied torques, compute joint accelerations directly without forming
or inverting the mass matrix.

GRiD also provides a "FD" variant that composes
:doc:`minv` ∘ :doc:`rnea` (i.e. ``qdd = M⁻¹·(τ − c)``). The two
forward-dynamics paths are independent implementations; the
benchmark suite reports both so users can pick by their downstream
workload.

Signature
---------
.. code-block:: python

   qdd = rbd.aba(q, qd, tau, f_ext=[], GRAVITY=-9.81)

Implementation
--------------
The Python reference is ``RBDReference.aba`` in
`RBDReference/RBDReference.py
<https://github.com/A2R-Lab/RBDReference>`__. CUDA codegen lives in
`GRiDCodeGenerator/algorithms/_aba.py
<https://github.com/A2R-Lab/GRiDCodeGenerator>`__.

See Also
--------
* :doc:`rnea` — inverse dynamics counterpart.
* :doc:`minv` — direct mass-matrix inverse (used by the FD variant).
* :doc:`crba` — composite-rigid-body mass matrix.
