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

Shared-Memory Spill Ladder
--------------------------
On sm_120 (RTX 5090) the per-block dynamic shared-memory cap is roughly
100 KiB. FDSVA-SO is the most shared-memory-intensive algorithm in GRiD, so
its kernel body is emitted from a spill ladder: an ordered list of rungs,
least spill first, each a tuple of flags saying which buffers move from the
shared-memory arena to the global ``d_workspace``
(``_FDSVA_SO_PICK_FLAGS`` in ``grid_codegen/algorithms/_fdsva_so.py``; the
per-rung arena sizes are composed in ``grid_codegen/_constants_arena.py``).
The rungs, in order:

#. ``full``: everything in shared memory.
#. ``global_tensors``: the four rank-3 output tensors live in
   ``d_workspace``.
#. ``idsva_cold``: additionally the embedded world-frame IDSVA-SO inner's
   cold buffers (its ``Xdown`` / ``v`` / ``a`` trio, dead before the hot
   triple walk) go to ``d_workspace`` while its hot pool stays in shared
   memory (a no-op for the body-frame inner).
#. ``workspace_temp``: the inner temp buffer moves to ``d_workspace``.
#. ``workspace_temp_spill``: additionally the first-order
   forward-dynamics-gradient ``da/df`` band spills, using the same
   mechanism the first-order ID gradient uses.
#. ``spill_df_du``: additionally the ``s_df_du`` block.
#. ``spill_Minv``: additionally ``s_Minv``.
#. ``pool_global``: the whole IDSVA-SO scratch pool goes to
   ``d_workspace`` (``SCRATCH_IN_SMEM=false``); ``s_df_du`` and ``s_Minv``
   stay in shared memory.
#. ``pool_global_spill_df_du``: ``pool_global`` plus ``s_df_du`` in
   ``d_workspace``; appended only for robots whose ``pool_global`` arena
   still overflows the target (nv ≈ 80 humanoids), so smaller robots keep a
   byte-identical ladder.

The selection is made at codegen time, per robot and base, for each of the
three :doc:`resource tiers <../resource_tier_system>`: the performance tier
takes the first rung whose arena fits ``GRID_CUDA_TARGET_SHARED_MEM_BYTES``,
the lite tier the first rung that fits the lite target (never less spill than
the performance pick), and the minimal tier always the last rung. No runtime
selection is involved; every rung computes the same numbers and only relocates
buffers.

Implementation
--------------
The reference implementation is ``RBDReference.fdsva_so`` in
`RBDReference/RBDReference.py
<https://github.com/A2R-Lab/RBDReference>`__. The CUDA kernel codegen
lives in
`grid_codegen/algorithms/_fdsva_so.py
<https://github.com/A2R-Lab/GRiD/tree/main/grid_codegen>`__.

Example Usage
-------------
.. code-block:: python

   from RBDReference import RBDReference
   rbd = RBDReference(robot)
   out = rbd.fdsva_so(q, qd, u)

Performance Characteristics
---------------------------
On sm_120 (RTX 5090) large floating-base humanoids such as ``g1_floating``
need one of the spilling rungs to fit under the 100 KiB per-block cap; arms
and quadrupeds fit lower on the ladder. The generated header's
``FDSVA_SO_*`` constants record the picks for a robot; see the benchmark
report for current per-cell numbers.

See Also
--------
* :doc:`idsva`: second-order inverse dynamics (the inner pass)
* :doc:`inverse_dynamics`
* :doc:`aba`
