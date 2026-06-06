Parallelism Patterns (the in-block fan-out toolkit)
===================================================

.. note::

   **Mental model.** On a modern GPU (e.g. RTX 5090) a single GRiD kernel is a
   **single block** (see :doc:`design_principles`), and the SMs are *not* the
   bottleneck — **nvcc compile time + latency-bound serial chains** are. So the
   goal of every algorithm is to **fan independent work across threads within the
   block** and shrink the *critical path* (the serial chain of dependent ops),
   not to conserve occupancy. Leaving work serial "to save threads" is almost
   always wrong.

GRiD exposes four recurring levers. **P1–P3 are sources of in-block
parallelism** (what work to fan across threads); **P4 is the offline-memory
layout that makes those parallel reads cheap and coalesced** (how the data is
laid out so the fan-out is fast, not bandwidth- or indirection-bound). They are
**first-class design requirements**: every algorithm should be audited against
all four, and **every remaining serial block (or strided/uncoalesced access)
must justify why none of P1/P2/P3/P4 apply**. This document is the canonical
reference; the agent-facing condensed version lives in
``docs/agent_debugging_guide.md`` §4.

----

P1 — Depth / BFS-level batching of tree recursions
--------------------------------------------------

**The pattern.** Rigid-body algorithms walk the kinematic tree — forward
(root→leaf) or backward (leaf→root). **Bodies at the same tree depth are
data-independent of each other** (a sibling never reads a sibling). So do not
emit the recursion as a serial loop over *bodies* (``for jid in range(NJ)``):
emit it as a serial loop over **levels** (BFS depth), and **within each level fan
all that level's bodies across threads**, with **one** ``__syncthreads()`` per
level. The genuinely-serial cross-level dependency (a parent accumulating its
children's contributions) is handled with ``atomicAdd`` or a segmented
reduction.

**Why it wins.** A serial per-body walk is ``O(NB)`` latency-bound steps (each a
≤36-thread 6×6 op + a full-block sync). The level form is ``O(depth)`` steps. On
a *branched* robot (a humanoid: ~30+ joints but depth ~7) that is a ~4× shorter
critical path; on a chain (depth == NB) it is neutral, so P1 never hurts.

**In-tree exemplars (use as templates):**

- ``_aba.py`` forward pass — same-BFS-level bodies batched via
  ``segmented_row_strided_gemv``.
- ``_crba.py`` *floating* composite-inertia backward — per-level fan + parent
  ``atomicAdd``.

**Audit cue.** Any ``for jid in range(...)`` (or reversed) that emits a per-body
6×6 op ending in a block sync is a P1 candidate. Look for the BFS-level / branch
grouping the codegen already computes for the forward/segmented paths and reuse
it for the backward pass.

.. warning::

   **P1 is not automatically a win — A/B-time it, and PRESERVE the GLASS ops.**
   The serial per-body loop already calls *tuned* GLASS ``gemm``/``gemv`` that
   parallelize each 6×6's inner reduction across threads. Replacing them with a
   hand-rolled per-output-element ``dot_prod`` (a serial 6-element reduction per
   thread) to "level-batch" trades GLASS's intra-op parallelism for a shorter
   sync chain — and for the *modest level widths* of most branched robots that
   **loses** (measured: ABA on g1 ~2 % slower, reverted). The right P1 keeps the
   GLASS ops and interleaves them across a level *without* per-op block syncs (a
   harder rewrite that may still not beat serial-GLASS for narrow levels). Always
   A/B time at N=256 and revert a regression. And **verify the benchmarked robot
   actually compiles the path you're optimizing** — e.g. mimic robots (h1_2)
   route fixed-base ABA through the ``Minv·(τ−rnea)`` compose path, not the ABA
   backward recursion, so that cost is really CRBA/Minv.

----

P2 — Parallel independent columns (gradients / Jacobians / Hessians)
-------------------------------------------------------------------

**The pattern.** A gradient ``∂f/∂q`` is a matrix whose **columns are
independent** (one per input DoF); a Hessian is a tensor whose **(j,k) cells are
independent**. Compute the *single forward/recursion pass once*, then **fan the
per-column / per-cell work across threads** — e.g. ``2·n²`` threads for an
``n×n`` gradient pair, one thread-group per output cell — rather than looping
column-by-column. The shared recursion result is read by every column; the
columns diverge in parallel.

**Why it wins.** Gradient/Hessian work is ``O(n)`` or ``O(n²)`` columns each
doing independent algebra; serializing them multiplies the critical path by the
column count for zero reason. This is the dominant lever for the ``*_du`` /
``*_so`` / ``*_hessian`` families.

**In-tree exemplars:**

- ``_inverse_dynamics_gradient.py`` (id_du) — per-output-element fan (2·NJ →
  2·n² threads).
- ``_eepose_gradient_hessian.py`` (d2ee) — per-cell Step-5b fan.
- ``_idsva_so.py`` — per-column forward sweep.

**Audit cue.** Any gradient/Hessian emit with an outer ``for col in range(n)``
(or per-(j,k)) wrapping otherwise-independent algebra is a P2 candidate.

----

P3 — Loop-invariant hoist → store temps → batch-parallel after
--------------------------------------------------------------

**The pattern.** When a computation inside a serial recursion **does not depend
on the loop's serial carry** (it needs only per-iteration *local* data, not the
running accumulator), it must **not** be paid for inside the latency-bound serial
loop. Instead: during the recursion, **store the per-iteration inputs/temps** to
a scratch band; **after** the loop, do the hoisted computation as **one
fully-parallel pass** over all stored temps.

**Why it wins (and how it differs from P1).** P1 parallelizes the
*within-a-level* body work but keeps it *inside* the recursion. P3 removes
genuinely loop-independent work from the recursion **entirely** — converting
``O(loop-len × op-latency)`` serial into ``O(op-latency)`` parallel. The two
compose: hoist what's independent (P3), level-batch what isn't (P1).

**The trade-off.** P3 costs scratch — the per-iteration temps you stash — so it
**interacts with the resource tiers** (:doc:`resource_tier_system`): the stored
band is a candidate to keep in smem when it fits and spill to ``d_workspace``
when it doesn't (it is cold / write-once-then-read-once, the ideal spill
candidate). Size it, and pick its tier, deliberately.

**Audit cue.** Inside a serial body-walk, look for any sub-expression whose
inputs are all per-iteration-local (transforms, per-body inertias, per-stage
forces) and whose output is consumed *after* the walk — that is a P3 hoist.

----

P4 — Offline memory layout: sparse compaction, coalesced distribution, topology-helper indirection
--------------------------------------------------------------------------------------------------

**The pattern.** GRiD is a **code generator**: it knows the robot's topology and
the *structural sparsity* of its spatial transforms / inertias / Jacobians
**offline**. Spend that offline knowledge to make the online reads cheap:

- **Sparse compaction.** Each robot's matrices have a known per-robot zero
  pattern (e.g. ``Xmat`` blocks, the support/ancestor structure of the Jacobian).
  Store and touch **only the structurally-nonzero entries** in a compact layout —
  fewer bytes (less smem pressure → fits a higher tier, :doc:`resource_tier_system`)
  and fewer ops. Do NOT emit dense loops over known zeros.
- **Coalesced distribution.** Choose the data layout (struct-of-arrays vs
  array-of-structs, per-body stride, padding/alignment) **offline** so that the
  kernel's thread→data mapping reads **contiguous, aligned addresses** within a
  warp. The fan-out of P1/P2 only pays off if its reads coalesce; an
  uncoalesced strided access can erase the parallelism win. Match the layout to
  how the threads are assigned (e.g. lay out per-body data so a level's bodies —
  P1 — are contiguous; lay out per-column data so a gradient's columns — P2 —
  are contiguous).
- **Topology-helper arrays.** Bake **precomputed index arrays** into the
  ``robotModel`` / generated constants — ``parent[]``, BFS-level/branch group
  offsets, sparsity-pattern offsets, column/support maps — so the online kernel
  does a **cheap coalesced array lookup** instead of **branchy per-thread index
  arithmetic**. These helpers are also what make P1 (level/branch grouping) and
  P2 (column maps) expressible without per-thread divergence.

**Why it wins.** P1–P3 shorten the *compute* critical path; P4 ensures the
*memory* path keeps up — coalesced, minimal-byte, branch-free indexing. On a
single-block kernel the warp's read efficiency and the smem footprint (which
sets the tier) are first-order; an algorithm can be "fully parallel" on paper
and still be slow if its accesses are strided or it carries known zeros.

**The trade-off / cautions.** Compaction + custom layouts make the emit more
robot-specialized (that is the GRiD bet — power users get hand-tuned per-robot
code). Keep the *index math* in offline-baked helper arrays, not online
branches. Topology helpers cost a little constant memory but remove online
indirection and divergence.

**Audit cue.** Look for: dense loops/stores over structurally-zero entries;
per-thread index arithmetic that could be a baked array lookup; a thread→data
mapping whose warp reads are strided/uncoalesced; per-body/per-column data
interleaved against the access order.

----

The audit (a standing requirement)
----------------------------------

Every algorithm carries a **parallelism-audit line**: for each of P1/P2/P3/P4,
state whether it *applies-and-is-done*, *applies-and-is-a-TODO*, or
*does-not-apply (why)*. A serial block (or a strided/uncoalesced access, or a
dense pass over known zeros) with no such justification is a bug to file, not a
style choice. When adding or refactoring an algorithm, the audit is part of the
change. See the per-algorithm audit table in
``docs/open-tasks/parallelism_audit.md`` (working doc) for current status.

.. seealso::

   :doc:`design_principles` (single-block, inner-owns-placement),
   :doc:`resource_tier_system` (where the P3 stored-temp bands spill).
