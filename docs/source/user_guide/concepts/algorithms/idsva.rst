IDSVA / IDSVA-SO (Inverse Dynamics, Second-Order)
=================================================

.. currentmodule:: RBDReference.RBDReference

Overview
--------
GRiD implements the **second-order** inverse dynamics algorithm (IDSVA-SO)
from Singh, Russell, & Wensing (`arXiv:2302.06001
<https://arxiv.org/abs/2302.06001>`_), which produces the rank-3 tensors
:math:`\partial^2 \tau / \partial q^2`, :math:`\partial^2 \tau / \partial \dot q^2`,
the mixed :math:`\partial^2 \tau / \partial q \partial \dot q`, and the
mass-matrix gradient :math:`\partial M / \partial q`.

Two mathematically equivalent variants are implemented; they differ only
in **reference frame**, and the codegen picks between them per robot
configuration:

* ``idsva_so_body_frame`` — body-frame propagation, body-frame inertia,
  body-frame motion subspace. Multi-pass forward/backward sweeps.
* ``idsva_so_world_frame`` — world-frame propagation, world-frame motion
  subspace, gravity baked into the main sweep. Single-pass reference
  (closer to the textbook spatial-vector-algebra derivation).

Dispatcher
----------
Both variants are correct on either base type. The choice is a pure
performance optimization, baked in at codegen time:

* **Fixed-base** → ``idsva_so_body_frame``: multi-pass subtree-broadcast
  amortizes when the chain is short and the tree is fixed (e.g. iiwa14
  fixed: ~27 µs vs ~830 µs on GPU).
* **Floating-base** → ``idsva_so_world_frame``: single-pass cost is flat
  in chain depth and the formulation avoids the body-frame gravity shim,
  winning 2–4× as DOF grows under floating base (e.g. iiwa14_floating
  ~1.7×, g1_floating ~3.6× on GPU).

In Python, both variants stay publicly callable; the meta dispatcher
``rbd.idsva_so(q, qd, qdd)`` forwards based on ``robot.floating_base``.
The C++ codegen emits both kernels (so the benchmark can compare them)
and a ``grid::idsva_so<T>()`` host wrapper that hard-routes to the
chosen variant.

Implementation
--------------
The reference implementations live in
`RBDReference/RBDReference.py
<https://github.com/A2R-Lab/RBDReference>`__ as
``idsva_so_body_frame``, ``idsva_so_world_frame``, and the meta
``idsva_so`` dispatcher. The codegen for the corresponding CUDA kernels
lives in
`GRiDCodeGenerator/algorithms/_idsva_so.py
<https://github.com/A2R-Lab/GRiDCodeGenerator>`__.

Example Usage
-------------
.. code-block:: python

   from RBDReference import RBDReference
   rbd = RBDReference(robot)

   # Auto-dispatched (recommended):
   d2tau_dq, d2tau_dqd, d2tau_cross, dM_dq = rbd.idsva_so(q, qd, qdd)

   # Or pick a variant explicitly (both are correct on either base):
   out_body  = rbd.idsva_so_body_frame(q, qd, qdd)
   out_world = rbd.idsva_so_world_frame(q, qd, qdd)

Performance Characteristics
---------------------------
GPU benchmarks above are from ``test/benchmarks/run_multi_version.py``
on sm_120 (RTX 5090). See the benchmark report for current numbers across
the full robot manifest.

See Also
--------
* :doc:`fdsva_so` — second-order forward dynamics
* :doc:`rnea`
* :doc:`crba`
* :doc:`aba`
* :doc:`minv`
