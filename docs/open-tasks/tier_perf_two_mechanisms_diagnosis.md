# Tier perf: two independent mechanisms (light diagnosis + Phase-2 measurement design)

T5 Part-1 investigation. **Code-level diagnosis only** — the heavy N=256 timing
sweep is deferred to the main agent's final serialized perf run. A *cheap*
occupancy/regcount probe is in scope; the full sweep is not.

## Why a tier can change perf — two distinct knobs

A `RESOURCE_TIER` selects a `(launch_bounds, smem-spill-placement)` profile. Those
two are **independent** levers and they move perf through **different**
mechanisms. Conflating them is what produced the A.7 LITE mis-tune.

### Mechanism 1 — smem-occupancy (the spill placement)

Higher tiers (LITE/MINIMAL) move inner scratch from `s_temp` (shared) to
`d_workspace` (L2-pinned global). Effect on perf:

- **Up:** less dynamic smem per block → more resident blocks per SM → higher
  occupancy → better latency hiding. This is the *intended* win and the only
  reason a humanoid kernel fits the ~100 KB cap at all.
- **Down:** a spilled hot buffer now costs an L2 hit instead of a shared-mem
  access (bounded by L2 pinning to ~smem→L2 latency, not smem→HBM).

This mechanism only exists when the algo actually has a spill rung that differs
between tiers (`select_shared_tier_3way` returns divergent picks). When the picks
collapse, mechanism 1 is inert.

### Mechanism 2 — launch_bounds-register (the register cap)

Every tier carries `__launch_bounds__(tier_max_threads<TIER>())` regardless of
spill: SHARED=`MAX_PERF_LEVEL_THREADS`, LITE=`min(2×SUGGESTED,768)`,
MINIMAL=`1024`. On sm_120 `regs_per_thread × max_threads ≤ 65536`, so a *looser*
bound forces ptxas to budget for more threads and therefore allocate **fewer
registers per thread**. Effect on perf:

- **Down:** fewer regs/thread → more spills to local memory / recompute →
  slower, *even with identical code and identical smem*. This is "register
  starvation."
- **Up:** the looser bound *permits* larger blocks; if the kernel actually
  benefits from >MAX_PERF_LEVEL_THREADS threads, only LITE/MINIMAL can launch
  them.

This mechanism is **always live** — it differs across tiers even when the spill
placement is byte-identical. (This is the rst "byte-identical" correction landed
in T5: collapsed-placement tiers are *not* byte-identical SASS; their
launch_bounds differ.)

### The A.7 cell, explained by the two mechanisms

`h1_2.fixed inverse_dynamics`: **no smem spill** at any tier → mechanism 1 inert.
So SHARED and LITE run the *same body*; the *only* difference is mechanism 2.
LITE's looser bound (`min(2×SUGGESTED,768)` > `MAX_PERF_LEVEL_THREADS`) starves
registers → LITE is *slower* than SHARED. MINIMAL (1024) is looser still but, on
this cell, the block-size flexibility apparently outweighs the regcount loss (or
ties), so the autotune's joint argmin lands on SHARED or MINIMAL, never LITE.

## Cheap Phase-2 probe (in scope; no heavy timing)

Two cheap, non-timing signals isolate which mechanism is in play per cell:

1. **`ptxas -v` register counts per tier.** Compile each kernel at SHARED / LITE /
   MINIMAL with `-Xptxas -v` and capture `registers`, `smem`, `spill stores/loads`.
   - Mechanism-2 fingerprint: regs/thread *falls* as the tier loosens, smem
     unchanged. (A.7 cells show exactly this.)
   - Mechanism-1 fingerprint: smem *falls* across tiers (spill placement changed).
2. **`cudaOccupancyMaxActiveBlocksPerMultiprocessor`** at each tier's
   `(block_size, dynamic_smem, regcount)`. Gives predicted active blocks/SM
   *without running the kernel* — a static occupancy delta that explains a
   measured perf delta.

Recommended artifact: extend `test/diagnostics/tier_baseline.py` (already collects
per-tier regs/smem) to emit, per `(algo, robot, base)`, a row tagging the dominant
mechanism: `spill` (smem differs), `regcap` (only regs differ), or `none`
(identical). That row tells the autotune *why* a tier won and flags the
no-spill-regcap-only cells (the A.7 class) for the deeper codegen fix (open-task
`tier_autotune_followups.md` §2).

## How the autotune (T5 Part 2) consumes this

The autotune does the *empirical* version of the above: it sweeps `(tier ×
threads)` and takes the global argmin µs at batch N, so it captures both
mechanisms jointly without needing to attribute the win. The probe above is the
*explanatory* companion — it says which knob moved, which is what you need to
decide whether to (a) bake the winner tier as the codegen default (followups §1)
or (b) fix the launch_bounds aliasing in codegen (followups §2).

## Measurement design for the deferred heavy sweep (for the main agent)

When the serialized perf sweep runs:

- Sweep `(tier ∈ {shared,lite,minimal}) × (threads ∈ grid) × (base) × (robot) ×
  (algo)` at the bench's batch N (256), compute-only path; also single-call when
  `--autotune-mode {single,both}`.
- Clip the thread grid per `(tier, algo)` to
  `min(cudaFuncAttributes.maxThreadsPerBlock, tier_max_threads<TIER>())` —
  probing above the launch_bounds cap just fails/clamps.
- Cross-reference each winning cell against the `tier_baseline.py` mechanism tag
  to confirm the win is explained (sanity check against noise).
