ABA (Articulated Body Algorithm)
================================

.. currentmodule:: RBDReference.RBDReference

Overview
--------
The Articulated Body Algorithm (ABA) is Featherstone's recursive
forward-dynamics algorithm: given joint positions, velocities, and
applied torques, compute joint accelerations directly without forming
or inverting the mass matrix.

GRiD also provides a ``forward_dynamics`` variant that composes
:doc:`minv` ∘ :doc:`inverse_dynamics` (i.e. ``qdd = M⁻¹·(τ − c)``). The two
forward-dynamics paths are independent implementations; the
library exposes both so users can choose by their downstream workload.

Signature
---------
.. code-block:: python

   qdd = rbd.aba(q, qd, tau, f_ext=[], GRAVITY=-9.81)

Implementation
--------------
The Python reference is ``RBDReference.aba`` in
`RBDReference/RBDReference.py
<https://github.com/A2R-Lab/RBDReference>`__. CUDA codegen lives in
`grid_codegen/algorithms/_aba.py
<https://github.com/A2R-Lab/GRiD/tree/modernizing-tests/grid_codegen>`__.

In GRiD
-------
On every handle, ``qdd = h.aba(q, qd, u)`` takes ``q`` at ``(B, h.nq)`` and
``qd``, ``u`` at ``(B, h.nv)``, and returns the accelerations at
``(B, h.nv)``, the tangent width. See :doc:`../input_output_abi` and
:doc:`../../tutorials/python_wrappers`.
It takes the same optional
per-body external forces as inverse dynamics (``f_ext``, shape
``(B, 6*num_bodies)``, body-major, ``[angular; linear]`` in each body's local
frame) and the signed gravity (default ``-9.81``). ``h.forward_dynamics`` gives
the same accelerations through the mass-matrix-inverse path,
``qdd = M⁻¹·(u − c)``; the two are independent implementations. The current
release table collects ``forward_dynamics``, not a separate GRiD ``aba`` row.
The forward-dynamics gradient and the
second-order :doc:`fdsva_so` are built on the ``forward_dynamics`` path, so a
workload that needs derivatives usually calls that one for the value as
well.

The CUDA host entries are ``grid::aba`` and ``grid::forward_dynamics``, each
with a ``_compute_only`` variant. In fp32 the forward-dynamics family can
amplify rounding on high-velocity states; the release collection retains
eligible cells under the explicit relative-L2 warning gate, not every
entrywise failure (see the
:doc:`release measurements <../../../release_measurements>`).

See Also
--------
* :doc:`inverse_dynamics` — inverse dynamics counterpart (RNEA).
* :doc:`minv` — direct mass-matrix inverse (used by the FD variant).
* :doc:`crba` — composite-rigid-body mass matrix.
