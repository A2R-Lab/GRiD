# IDSVA-SO inner-layout refactor — deferred task notes

**Status:** DEFERRED. Filed 2026-05-22 during the idsva_so per-tier spill work.
Revisit **after a benchmark sweep** quantifies how much LITE/MINIMAL perf we
actually lose with the current "whole-inner-to-global" fallback. Only invest in
this refactor if the sweep shows the whole-arena spill is a real bottleneck.

## Why this exists

The per-tier spill we shipped can keep idsva_so under the smem budgets, but the
*surgical* part (spill only cold buffers, keep hot buffers in smem) turned out to
be far less effective than hoped, because of how the inner's shared arena is laid
out. These notes record the analysis and concrete refactor ideas so a future
session can pick it up without re-deriving everything.

## The measured problem (sm_120, PERF smem budget = 98304 B, LITE = 49152 B)

Fixed-base **body** inner (`gen_idsva_so_body_frame_inner`, `_idsva_so.py:~1300`):

- g1_fixed: NV=29, NB=29. `inner_temp = 16572 floats (~66 KB)`,
  XI=2088, s_q_qd_u=87, output 4·NV³=97556 floats.
  - full arena (+output in smem) = 465212 B → output MUST spill.
  - base (output→global) = 74988 B → fits PERF (98304), NOT LITE (49152).
  - base − BC = 70812; base − BC − IC = 66636 → still way over LITE.
  - whole s_temp→global (only XI + s_q_qd_u in smem) = 8700 B.
- h1_2_fixed: base arena ≈ 146368 B (from the tier baseline) → **overflows even
  PERF**; ~48 KB must be shed.

**Key fact:** the only cleanly-separable *cold* buffer in the fixed body inner is
**BC** (`36*NB`, ~1044 floats / ~4 KB for g1). IC (`36*NB`) is also cold-ish but
is **live through the serial `reference_order_output_repair`** on branched robots
(h1_2/g1 are branched). Everything else in the ~66 KB inner is either hot in the
t1–t9 / p1–p6 ancestor loops or **memory-aliased** (the layout deliberately reuses
slabs: `f=vJ=Xdown`, `T1=IC_S`, `D3=B_IC_S`, `crm_v=crm_S`, etc., see
`_idsva_so.py:~1366-1428`). Aliasing already minimizes the smem high-water mark,
so you cannot independently relocate an aliased buffer without moving its alias
partner. Net: surgical cold-spill ceiling ≈ BC ≈ a few KB — nowhere near the
~18 KB (g1) / ~48 KB (h1_2) needed to hit LITE/PERF budgets. Hence the
whole-inner→global fallback is what actually makes them fit.

## Where the bytes are (fixed body inner, g1)

`inner_temp = 36*NV*10 + 30*NV + 6 + len(jids_a)*36`. For g1 (NV=29):
- `36*NV*10 = 10440` floats — ten 36×NV matrix slabs (IC, BC, Xup/Xdown, the
  crm_*/crf_*/D1–D4/B_IC_S family, aliased per phase).
- `len(jids_a)*36 = 5256` floats — the **ancestor-pair scratch** (`t`, `t1..t9`,
  `p1..p6`); this term aliases `Xup` and is the **hot t-loop working set**. It is
  also the term that **grows fastest** with robot size/branching, so for h1_2 it
  likely dominates.
- `30*NV = 870` floats — five 6×NV vectors (S/psid/psidd/psid_Sd/…).

## Refactor ideas (ranked)

1. **De-alias + independently place the ancestor-pair scratch (`t`/`t1..t9`/
   `p1..p6`).** This is the single biggest growable region and the best partial-
   spill target *if* its access pattern is coalesced/parallel (the t-loop
   distributes ancestor pairs across threads, each thread owning a disjoint
   `jid_a` slice — looks coalescible). Today it aliases `Xup` (layout line
   ~1362), so it can't be placed separately. Refactor: give the ancestor scratch
   its own arena region + a placement template bool so a tier can put just this
   region in global while the small hot D-matrices (D1–D4, 4·36·NV) stay in smem.
   Benchmark the global-access cost first — if the per-thread slices coalesce
   well, this could hit LITE while keeping most hot state fast. **Highest payoff,
   medium risk.**

2. **Split the inner into explicit phase-scoped sub-arenas with lifetime-aware
   placement.** Forward sweep (v/a/f/vJ/Sd/psid/psidd) → IC/BC build → D-matrix
   build → t-loop. Some early-phase buffers are dead by the t-loop but currently
   alias later hot buffers (which is *why* the peak is already small). To make a
   partial spill reduce the *peak*, you must move a peak-resident buffer to
   global — i.e. this only helps in combination with idea 1 (the ancestor
   scratch is the peak resident worth moving). Lower marginal value on its own.

3. **IC spill (cheap, but gated).** IC (`36*NB`) is dead after the D-loop EXCEPT
   the serial `reference_order_output_repair` reads it (via IC_S/crm_S
   derivations) on branched robots. Safe to spill only when
   `not idsva_so_needs_reference_order_output_repair(self)` (serial chains /
   base-rooted forests), or accept a slow serial global-read pass. Small payoff
   (~4 KB); only worth it bundled with idea 1.

## World frame

The **world** inner (`gen_idsva_so_world_frame_inner`, `_idsva_so.py:~2492`,
explicit layout `:~2548-2599`) is NOT aliased the same way — `f_w`, `IC`, `BC`,
`v_w`, `a_w`, `Xup`, `Xdown` are separate regions. So surgical partial spill is
*more* tractable there (f_w `6*NB` coldest; IC/BC `36*NB` warm, not in the hottest
(k,rr) loop; v_w/a_w/Xup/Xdown dead after Phase 4). The world inner only overflows
LITE for the biggest floating humanoids (h1_2_floating_W ~58 KB > 48 KB LITE),
and PERF fits. So world-frame surgical spill could plausibly hit LITE without the
whole-arena hammer — worth trying world-frame surgical first if the sweep says
LITE perf matters. (Production floating path is world frame; fixed is body.)

## Pointers

- Arena helper + the existing whole-`s_temp` routing: `gen_declare_shared_arena`
  / `tier_workspace_expr` in `helpers/_code_generation_helpers.py:543`.
- Pick selection: `select_shared_tier_3way` in `GRiDCodeGenerator.py:~257`.
- Mirror kernel-body pattern: `_emit_fdsva_so_kernel_body_for_flags` /
  `gen_fdsva_so_kernel` in `algorithms/_fdsva_so.py`.
- Introspection script used to get the numbers above: rebuildable — load robot
  via `URDFParser`, `cg.gen_idsva_so_body_frame_inner_temp_mem_size()`,
  `cg.py_arena_bytes(t_count)`, `cg.cuda_target_shared_mem_bytes` /
  `cuda_target_lite_shared_mem_bytes`.
