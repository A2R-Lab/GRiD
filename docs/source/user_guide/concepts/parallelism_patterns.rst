Parallelism patterns
====================

.. note::

   On a modern GPU (for example an RTX 5090) a single GRiD kernel is a single
   block (see :doc:`design_principles`), and the SMs are not the bottleneck;
   nvcc compile time and latency-bound serial chains are. The goal of every
   algorithm is therefore to fan independent work across the threads within
   the block and to shrink the critical path (the serial chain of dependent
   operations), not to conserve occupancy. Leaving work serial to save threads
   is almost always wrong.

GRiD exposes four recurring levers. P1 to P3 are sources of in-block
parallelism: they decide what work to fan across threads. P4 is the offline
memory layout that makes those parallel reads coalesced, so that the fan-out
is not bandwidth- or indirection-bound. All four are design requirements:
every algorithm is reviewed against them, and every remaining serial block,
strided access or uncoalesced access must justify why none of P1, P2, P3 or
P4 applies. This document is the canonical reference; a condensed checklist
lives in the `debugging guide <https://github.com/A2R-Lab/GRiD/blob/main/docs/agent_debugging_guide.md>`_ §4.

----

P1: level batching of tree recursions
-------------------------------------

Rigid-body algorithms walk the kinematic tree, forward from root to leaf or
backward from leaf to root. Bodies at the same tree depth are data-independent
of each other: a sibling never reads a sibling. The recursion is therefore
not emitted as a serial loop over bodies (``for jid in range(NJ)``) but as a
serial loop over levels (BFS depth). Within each level all of that level's
bodies are fanned across threads, with one ``__syncthreads()`` per level. The
serial cross-level dependency, a parent accumulating its children's
contributions, is handled with ``atomicAdd`` or a segmented reduction.

A serial per-body walk is ``O(NB)`` latency-bound steps, each a ≤36-thread
6×6 operation followed by a full-block sync. The level form is ``O(depth)``
steps. On a branched robot (a humanoid: ~30+ joints but depth ~7) that is a
~4× shorter critical path; on a chain (depth == NB) it is neutral, so P1
never hurts.

In-tree exemplars:

- ``_aba.py`` forward pass: same-BFS-level bodies batched via
  ``segmented_row_strided_gemv``.
- ``_crba.py`` floating composite-inertia backward pass: per-level fan with a
  parent ``atomicAdd``.

Review cue: any ``for jid in range(...)`` (or reversed) that emits a per-body
6×6 operation ending in a block sync is a P1 candidate. Reuse the BFS-level
and branch grouping the codegen already computes for the forward and
segmented paths for the backward pass.

.. warning::

   P1 is not automatically a win: A/B-time it, and preserve the GLASS
   operations. The serial per-body loop already calls tuned GLASS
   ``gemm``/``gemv`` that parallelize each 6×6's inner reduction across
   threads. Replacing them with a hand-rolled per-output-element ``dot_prod``
   (a serial 6-element reduction per thread) to level-batch trades GLASS's
   intra-op parallelism for a shorter sync chain, and for the modest level
   widths of most branched robots that loses (measured: ABA on g1 ~2 % slower,
   reverted). The right P1 keeps the GLASS operations and interleaves them
   across a level without per-op block syncs, a harder rewrite that may still
   not beat serial GLASS for narrow levels. A/B time at N=256 and revert a
   regression. Verify that the benchmarked robot compiles the path being
   optimized: mimic robots (h1_2) route fixed-base ABA through the
   ``Minv·(τ−rnea)`` compose path, not the ABA backward recursion, so that
   cost is really CRBA/Minv.

----

P2: parallel independent columns (gradients, Jacobians, Hessians)
-----------------------------------------------------------------

A gradient ``∂f/∂q`` is a matrix whose columns are independent, one per input
DoF; a Hessian is a tensor whose (j,k) cells are independent. Compute the
single forward or recursion pass once, then fan the per-column or per-cell
work across threads, for example ``2·n²`` threads for an ``n×n`` gradient pair
with one thread-group per output cell, rather than looping column by column.
The shared recursion result is read by every column; the columns diverge in
parallel.

Gradient and Hessian work is ``O(n)`` or ``O(n²)`` columns each doing
independent algebra; serializing them multiplies the critical path by the
column count for no reason. This is the dominant lever for the ``*_du`` /
``*_so`` / ``*_hessian`` families.

In-tree exemplars:

- ``_inverse_dynamics_gradient.py`` (id_du): per-output-element fan (from
  2·NJ to 2·n² threads).
- ``_eepose_gradient_hessian.py`` (d2ee): per-cell Step-5b fan.
- ``_idsva_so.py``: per-column forward sweep.

Review cue: any gradient or Hessian emit with an outer ``for col in range(n)``
(or per-(j,k)) wrapping otherwise-independent algebra is a P2 candidate.

----

P3: loop-invariant hoisting with stored temporaries
---------------------------------------------------

When a computation inside a serial recursion does not depend on the loop's
serial carry (it needs only per-iteration local data, not the running
accumulator), it must not be paid for inside the latency-bound serial loop.
Instead, during the recursion, store the per-iteration inputs or temporaries
to a scratch band; after the loop, do the hoisted computation as one fully
parallel pass over all stored temporaries.

P1 parallelizes the within-a-level body work but keeps it inside the
recursion. P3 removes loop-independent work from the recursion entirely,
converting ``O(loop-len × op-latency)`` serial work into ``O(op-latency)``
parallel work. The two compose: hoist what is independent (P3), level-batch
what is not (P1).

P3 costs scratch, the per-iteration temporaries that are stored, so it
interacts with the resource tiers (:doc:`resource_tier_system`). The stored
band is a candidate to keep in smem when it fits and to spill to
``d_workspace`` when it does not; it is cold, written once and read once,
which makes it the ideal spill candidate. Size it, and pick its tier,
explicitly.

Review cue: inside a serial body walk, look for any sub-expression whose
inputs are all per-iteration-local (transforms, per-body inertias, per-stage
forces) and whose output is consumed after the walk. That is a P3 hoist.

----

P4: offline memory layout (sparse compaction, coalesced distribution, topology helpers)
---------------------------------------------------------------------------------------

GRiD is a code generator: it knows the robot's topology and the structural
sparsity of its spatial transforms, inertias and Jacobians offline. That
offline knowledge is spent to make the online reads fast, in three ways.

The first is sparse compaction. Each robot's matrices have a known per-robot
zero pattern (for example the ``Xmat`` blocks, or the support/ancestor
structure of the Jacobian). Store and touch only the structurally nonzero
entries in a compact layout: fewer bytes (less smem pressure, so the kernel
fits a higher tier, :doc:`resource_tier_system`) and fewer operations. Do not
emit dense loops over known zeros.

The second is coalesced distribution. Choose the data layout
(struct-of-arrays versus array-of-structs, per-body stride, padding and
alignment) offline so that the kernel's thread-to-data mapping reads
contiguous, aligned addresses within a warp. The fan-out of P1 and P2 only
pays off if its reads coalesce; an uncoalesced strided access can erase the
parallelism win. Match the layout to how the threads are assigned: lay out
per-body data so a level's bodies (P1) are contiguous, and per-column data so
a gradient's columns (P2) are contiguous.

The third is topology-helper arrays. Bake precomputed index arrays into the
``robotModel`` and generated constants (``parent[]``, BFS-level and branch
group offsets, sparsity-pattern offsets, column and support maps) so the
online kernel does a coalesced array lookup instead of branchy per-thread
index arithmetic. These helpers are also what make P1 (level and branch
grouping) and P2 (column maps) expressible without per-thread divergence.

P1 to P3 shorten the compute critical path; P4 ensures the memory path keeps
up with coalesced, minimal-byte, branch-free indexing. On a single-block
kernel the warp's read efficiency and the smem footprint (which sets the
tier) are first-order; an algorithm can be fully parallel on paper and still
be slow if its accesses are strided or it carries known zeros.

Compaction and custom layouts make the emit more robot-specialized; that is
the GRiD bet, and power users get hand-tuned per-robot code in return. Keep
the index math in offline-baked helper arrays, not online branches. Topology
helpers cost a little constant memory but remove online indirection and
divergence.

Review cue: look for dense loops or stores over structurally zero entries;
per-thread index arithmetic that could be a baked array lookup; a
thread-to-data mapping whose warp reads are strided or uncoalesced; and
per-body or per-column data interleaved against the access order.

----

The parallelism review
----------------------

Every algorithm carries a parallelism-review line: for each of P1, P2, P3
and P4 it states whether the lever applies and is done, applies and is a
TODO, or does not apply, with the reason. A serial block, a strided or
uncoalesced access, or a dense pass over known zeros with no such
justification is a bug to file, not a style choice. When adding or
refactoring an algorithm, the review is part of the change.

.. seealso::

   :doc:`design_principles` (single-block, inner-owns-placement),
   :doc:`resource_tier_system` (where the P3 stored-temp bands spill).
