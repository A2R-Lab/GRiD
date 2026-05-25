# IDSVA-SO inner-layout refactor — deferred task notes

**Status:** DEFERRED. Filed 2026-05-22 during the idsva_so per-tier spill work.
Revisit **after a benchmark sweep** quantifies how much LITE/MINIMAL perf we
actually lose with the current "whole-inner-to-global" fallback. Only invest in
this refactor if the sweep shows the whole-arena spill is a real bottleneck.

## Core design principle: the INNER owns scratch placement (read this first)

When a kernel's scratch arena does not fit the device shared-memory cap, the
spill decision belongs to the **inner device function**, not the caller. Every
`*_inner` is templated on a placement flag (`SCRATCH_IN_SMEM` — or, equivalently,
the `RESOURCE_TIER`) and takes **both** pointers: `s_temp` (shared) and
`d_workspace` (global). At the very top of the inner it selects where its scratch
lives via `if constexpr`:

```cpp
template <typename T, bool SCRATCH_IN_SMEM = true>
__device__ void foo_inner(..., T *s_temp, T *d_workspace, ...) {
    if constexpr (!SCRATCH_IN_SMEM) { s_temp = d_workspace; } else { (void)d_workspace; }
    // ... rest of the body is unchanged; it just uses s_temp ...
}
```

The **caller** (a standalone kernel, or a composing kernel such as `fdsva_so`
which embeds the `idsva_so` inner) only (a) sizes both arenas from the inner's
exposed `*_SMEM_BYTES` / `*_WORKSPACE_BYTES` constants, and (b) threads the
per-tier flag down. It must **never** hard-repoint or alias the inner's scratch
from the outside.

Why this is the rule, not a stylistic preference:

* **Single source of truth.** The inner sizes *and* places its own scratch, so
  the shared-mem-bytes macro, the workspace-bytes macro, and the actual pointer
  arithmetic can never drift out of sync across callers.
* **Surgical improvements propagate for free.** If a future change teaches an
  inner to spill only its *cold* buffers (keeping the hot loop in smem), every
  caller — the standalone kernel *and* every composing kernel — inherits that
  improvement just by passing the flag. No caller edits, no re-derived offsets.
* **Backward-compatible by default.** `SCRATCH_IN_SMEM = true` keeps small robots
  byte-identical; only robots that overflow flip it to `false`.

Reference implementations that already follow this: `aba_inner`,
`forward_dynamics_inner`, `fdsva_so_inner`. The `idsva_so` body- and world-frame
inners adopt the same `SCRATCH_IN_SMEM` template; `fdsva_so` then spills the
embedded idsva_so scratch purely by passing `SCRATCH_IN_SMEM=false` to it (its
dominant cost), instead of the caller aliasing pointers.

### Project-wide propagation status (this is the standard for ALL algorithms)

This is not an SO-specific pattern — every algorithm's inner should own its
scratch placement. Conformance audit (2026-05-24):

| Inner | Placement template | Status |
|-------|--------------------|--------|
| `aba_inner` | `TEMP_IN_SMEM` (whole arena) | conforms |
| `forward_dynamics_inner` | `MINV_F_IN_SMEM` (F region) | conforms (surgical-F) |
| `direct_minv_inner` | `F_IN_SMEM` (F region) | conforms (surgical-F) |
| `integrator_inner` | `MINV_F_IN_SMEM` | conforms |
| `fdsva_so_inner` | `SCRATCH_IN_SMEM` (4·NV³) | conforms |
| `end_effector_pose_gradient_inner` | `TEMP_IN_SMEM` | conforms |
| `idsva_so_body_frame_inner` | `BC_IN_SMEM` (surgical BC only) | **partial** — needs whole-arena `SCRATCH_IN_SMEM` so callers stop repointing `s_temp` from outside |
| `idsva_so_world_frame_inner` | none | **needs migration** — add `SCRATCH_IN_SMEM` |
| `inverse_dynamics_gradient_inner` (id_du) | none | **needs migration** — kernel repoints `s_temp` from outside (`_inverse_dynamics_gradient.py:1015,1034`) |
| `forward_dynamics_gradient_inner` (fd_du) | none | **needs migration** — kernel repoints `s_temp` (`_forward_dynamics_gradient.py:159,182`) |
| `integrator_gradient` inner | none | **needs migration** — kernel `_emit_spill_pointers` repoints `s_temp` (`_integrator_gradient.py:744`) |

**Caveat — the XImats/XmatsHom helper is a separate caller-level scratch user.**
Even once the inner owns its arena, the kernel still calls
`load_update_X*mats_helpers(..., s_temp)` *outside* the inner, and that helper
dereferences `s_temp` for its sincos scratch (`2*num_pos` floats). When the inner
arena is spilled and the smem `s_temp` slot is `nullptr`, the helper segfaults
(this was the 2026-05-24 null-`s_temp` crash in `aba` + `ee_pose_gradient`). Two
acceptable resolutions, pick one and apply uniformly: (a) the kernel repoints
`s_temp` at the spilled workspace before the helper call (current fix), or
(b) always reserve the tiny `2*num_pos` helper scratch in smem regardless of
inner spill (keeps sincos fast). (b) is the cleaner long-term target.

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

## Implementation log — 2026-05-25 (UNVALIDATED working tree; commits need approval)

Done (no-compile probe shows it generates + fits; NOT yet compiled/equivalence-tested):

1. **idsva_so world inner** owns placement: `SCRATCH_IN_SMEM` template +
   top-of-body `if constexpr(!SCRATCH_IN_SMEM){s_temp=d_workspace;}`;
   `gen_idsva_so_world_frame_inner_function_call` gained `scratch_in_smem_expr`
   (default `"true"` → standalone idsva world kernel byte-identical).
2. **`fdsva_so_full_inner`** (new): wraps the whole fdsva orchestration
   (XImats-helper → minv → fd → fd-grad-inline → idsva → contraction) as one
   inner templated `<T, SCRATCH_IN_SMEM, FD_GRAD_USE_SPILL, CONTRACT_IN_SMEM>`.
   The `s_temp` repoint at the top covers EVERY consumer incl. the helper sincos,
   so the kernel never repoints. `gen_fdsva_so_full_inner_function_call` mirrors
   the def. Both fdsva kernel paths now call it (rungs 0–5 behavior-preserving;
   new rung 6 = pool→global). Registered in the class import list.
3. **fdsva tier level-6 (both bases)**: `("pool_global", base_t_count, T,T,F,F,F,T)`.
   The full inner hands the placed pool to the idsva inner (world OR body), so it
   works for fixed too without touching the aliased body inner.
   Probe: h1_2_floating fdsva 198→**53.8 KB**, fits; (re-probe fixed pending).

REMAINING for full unity (these kernels currently WORK via kernel-side repoint —
deferred, not broken):
- `gen_fdsva_so_device` (inline API) still has a duplicate inline orchestration →
  rewire to call `fdsva_so_full_inner`.
- `id_du`, `fd_du`, `integrator_gradient`: migrate kernel-side `s_temp` repoints
  into `*_full_inner` orchestration inners (same pattern as fdsva).
- standalone idsva body kernel `output_temp` rung: migrate its kernel repoint to a
  body-inner `SCRATCH_IN_SMEM` (or leave — it works).

VALIDATION OWED before commit: regen all robots; compile iiwa14 + g1 + h1_2;
idsva_so / fdsva_so / world-frame CUDA equivalence at PERF and MINIMAL, fixed +
floating. Rungs 0–5 MUST be numerically identical (pure relocation); rung 6 must
match too. Sanitizer on h1_2_floating MINIMAL.
