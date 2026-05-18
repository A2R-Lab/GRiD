FDSVA-SO (Forward Dynamics, Second-Order)
=========================================

.. currentmodule:: RBDReference.RBDReference

Overview
--------
FDSVA-SO computes the second-order partials of forward dynamics
:math:`\partial^2 \ddot q / \partial \cdot \partial \cdot` by combining
IDSVA-SO with first-order forward-dynamics gradients, following Singh,
Russell, & Wensing (`arXiv:2302.06001
<https://arxiv.org/abs/2302.06001>`_).

The inner pass reuses the IDSVA-SO variant selected by the dispatcher
(body-frame inner for fixed-base, world-frame inner for floating-base),
so FDSVA-SO inherits the same per-base-type performance crossover.

Shared-Memory Tier Selector
---------------------------
On sm_120 (RTX 5090) the per-block dynamic shared-memory cap is roughly
100 KiB. FDSVA-SO is the most shared-memory-intensive algorithm in GRiD;
the codegen automatically picks one of four tiers per (robot, base)
combination based on the arena size needed:

#. ``full`` — all scratch in shared memory.
#. ``global_tensors`` — the four rank-3 output tensors live in
   ``d_workspace``.
#. ``workspace_temp`` — output tensors plus the IDSVA-SO inner temp
   buffer move to ``d_workspace``.
#. ``workspace_temp_spill`` — additionally spills the selective
   ``da_dq..fxvi`` band into ``d_workspace`` using the same spill
   mechanism the first-order ID gradient uses.

The tier is determined at codegen time from
``GRID_CUDA_TARGET_SHARED_MEM_BYTES`` and the robot's NB / NV; no runtime
selection is required. The selector lives next to the other generated-size
helpers in
`GRiDCodeGenerator.py <https://github.com/A2R-Lab/GRiDCodeGenerator>`_.

Implementation
--------------
The reference implementation is ``RBDReference.fdsva_so`` in
`RBDReference/RBDReference.py
<https://github.com/A2R-Lab/RBDReference>`__. The CUDA kernel codegen
lives in
`GRiDCodeGenerator/algorithms/_fdsva_so.py
<https://github.com/A2R-Lab/GRiDCodeGenerator>`__.

Example Usage
-------------
.. code-block:: python

   from RBDReference import RBDReference
   rbd = RBDReference(robot)
   out = rbd.fdsva_so(q, qd, u)

Performance Characteristics
---------------------------
On sm_120 (RTX 5090) ``g1_floating`` requires the
``workspace_temp_spill`` tier to fit under the 100 KiB per-block cap;
all smaller robots and base configurations fit in lower tiers. See the
generated benchmark report for current per-cell numbers.

See Also
--------
* :doc:`idsva` — second-order inverse dynamics (the inner pass)
* :doc:`rnea`
* :doc:`aba`
