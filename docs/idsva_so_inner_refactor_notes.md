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
scratch placement. Conformance audit (2026-05-29; refreshed after the
id_du/fd_du/integrator_gradient `_device`-orchestrator landing):

| Inner | Placement template | Status |
|-------|--------------------|--------|
| `aba_inner` | `TEMP_IN_SMEM` (whole arena) + `COLD_IN_SMEM` (surgical cold sub-band) | conforms |
| `forward_dynamics_inner` | `MINV_F_IN_SMEM` (F region) | conforms (surgical-F) |
| `direct_minv_inner` | `F_IN_SMEM` (F region) | conforms (surgical-F) |
| `integrator_inner` | `MINV_F_IN_SMEM` | conforms |
| `fdsva_so_contract` | `SCRATCH_IN_SMEM` (4·NV³) | conforms |
| `fdsva_so_device` (orchestrator) | `SCRATCH_IN_SMEM × FD_GRAD_USE_SPILL × CONTRACT_IN_SMEM` | conforms (canonical 3-lever pattern) |
| `crba_inner` | `TEMP_IN_SMEM` | conforms |
| `end_effector_pose_gradient_inner` | `TEMP_IN_SMEM` | conforms |
| `idsva_so_body_frame_inner` | `SCRATCH_IN_SMEM × BC_IN_SMEM` (whole-arena + surgical BC; mutually exclusive per the body tier table) | conforms |
| `idsva_so_world_frame_inner` | `SCRATCH_IN_SMEM × COLD_IN_SMEM` (whole-arena + surgical cold trio Xdown/v_w/a_w; mutually exclusive) | conforms |
| `inverse_dynamics_gradient_device` (id_du) | `SCRATCH_IN_SMEM` (whole-arena via the `_device` orchestrator) | conforms |
| `forward_dynamics_gradient_device` (fd_du) | `SCRATCH_IN_SMEM` (whole-arena via the `_device` orchestrator) | conforms |
| `integrator_gradient_device` | per-tier rung (Dqdd / dAB / id_du level) via `_device` orchestrator | conforms |

Every emitted kernel now passes its per-tier placement flag through to the
inner / `_device` and lets `if constexpr (!FLAG) { s_temp = d_workspace; ... }` at
the top of the callee do the repoint. For idsva_so / fdsva_so the kernel does
**no** `s_temp` surgery at all — the inner / `_device` owns the placement and
the XImats helper is called *inside* the inner after the repoint, so its sincos
scratch follows the placement too (this is the canonical pattern).

`aba_kernel`, `crba_kernel`, and `end_effector_pose_gradient_kernel` still do a
kernel-side `s_temp = <ws>;` before calling `load_update_XImats_helpers(..., s_temp)`
on the whole-arena rung, because in those algorithms the helper is called from the
kernel (not from inside the inner). The inner still owns its own placement via its
template flag; this kernel repoint is purely to give the helper's sincos slot a
valid backing pointer instead of `nullptr` (see the "null `s_temp` to the load
helper" anti-pattern note). Moving those helper calls *inside* the inner (as
idsva_so / fdsva_so already do) is the cleaner long-term target — tracked but not
urgent.

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

## Implementation log

### 2026-05-25 — inner-owns-placement migration (LANDED)

1. **idsva_so world inner** owns placement: `SCRATCH_IN_SMEM` template +
   top-of-body `if constexpr(!SCRATCH_IN_SMEM){s_temp=d_workspace;}`;
   `gen_idsva_so_world_frame_inner_function_call` gained `scratch_in_smem_expr`
   (default `"true"` → standalone idsva world kernel byte-identical).
2. **`fdsva_so_device`** (canonical orchestration; originally landed as
   `fdsva_so_full_inner` before the 2026-05-28 rename): wraps the whole fdsva
   orchestration (XImats-helper → minv → fd → fd-grad-inline → idsva → contraction)
   as one device function templated
   `<T, SCRATCH_IN_SMEM, FD_GRAD_USE_SPILL, CONTRACT_IN_SMEM>`. The `s_temp`
   repoint at the top covers EVERY consumer incl. the helper sincos, so the kernel
   never repoints. `gen_fdsva_so_device_function_call` mirrors the def. Both fdsva
   kernel paths call it (rungs 0–5 behavior-preserving; rung 6 = pool→global).
3. **fdsva tier level-6 (both bases)**: `("pool_global", base_t_count, T,T,F,F,F,T)`.
   The `_device` hands the placed pool to the idsva inner (world OR body), so it
   works for fixed too without touching the aliased body inner.
   Probe: h1_2_floating fdsva 198→**53.8 KB**, fits.

### 2026-05-28 — orchestrator training-wheels drop

The auto-allocating ``_device`` training-wheels wrapper that previously sat
alongside ``_full_inner`` for all 4 orchestrators (fdsva_so / id_du / fd_du /
integrator_gradient) has been dropped — the equivalence runner's only consumers
(`floating_inverse_dynamics_gradient_runner` and `floating_forward_dynamics_gradient_runner`)
were dead code (the actual floating id_du/fd_du tests use the regular kernel) and
were removed too. After the rename + drop the orchestrators are a clean 3 layers:
``_host`` / ``_kernel`` / ``_device``. Simple algorithms (id, minv, fd, aba, crba,
ee_pose*, integrator, idsva_so_*) still ship their auto-allocating ``_device``
because the equivalence runner's simple-algo test kernels still call them; a
future cleanup can collapse those too.

### 2026-05-29 — B.1 + B.3 audit (no code change; doc-clarify only)

Audited the idsva_so body / world inner + the fdsva_so device for the
inner-owns-placement contract. Findings:

* **idsva_so_body_frame_inner** already exposes the 2-lever pattern
  `<T, SCRATCH_IN_SMEM, BC_IN_SMEM>`. The body's `if constexpr (!SCRATCH_IN_SMEM)
  { s_temp = d_workspace; }` (line ~1372) and `if constexpr (!BC_IN_SMEM) { BC =
  d_workspace; }` (line ~1500) are real and operative. Mutually exclusive per the
  body tier table (rung 2 = BC=false+SCRATCH=true; rung 3 = SCRATCH=false+BC=true).
* **idsva_so_world_frame_inner** already exposes
  `<T, SCRATCH_IN_SMEM, COLD_IN_SMEM>` with the surgical cold trio
  (Xdown/v_w/a_w) repoint at the end-of-layout. Mutually exclusive per the
  world tier table.
* **fdsva_so_device** already exposes the canonical 3-lever pattern
  `<T, SCRATCH_IN_SMEM, FD_GRAD_USE_SPILL, CONTRACT_IN_SMEM>` and is called
  from both kernels for all 7 tier rungs (rung 6 = pool→global routes the whole
  s_temp through the inner's SCRATCH_IN_SMEM=false).
* **Body / world kernels' `output_temp` rung** correctly emits a per-rung body
  with `smem_temp = 0`, calls the inner with `SCRATCH_IN_SMEM = false`, and hands
  the per-timestep d_workspace sub-region in via `d_temp_spill`. The kernel does
  NOT hand-repoint `s_temp`; the inner does the repoint via `if constexpr`.
* Confirmed by inspecting emitted h1_2 fixed body kernel: the `output_temp` rung
  declares `T *s_temp = nullptr;` and calls
  `idsva_so_body_frame_inner<T, false, true>(...)` directly. No
  caller-side surgery.

**Conclusion:** the surgical de-alias work this doc lists as a follow-up is in
fact implemented at the inner-template level for both idsva_so frames + the
fdsva_so orchestrator. The remaining tracked refinement is the deeper de-alias
of the body inner's ancestor-pair scratch (idea 1 above), which would let
surgical rungs (BC-only) close a bigger gap on humanoid-scale robots — that work
is still deferred pending a perf sweep that quantifies the win from the
whole-arena rung vs. a surgical rung at LITE/MINIMAL.

Validation (2026-05-29, SUGGESTED threads on RTX 5090 sm_120):
* `iiwa14-fixed-threadssuggested` + `iiwa14-floating-threadssuggested` CUDA
  equivalence — confirms the byte-identical PERF path is intact.
* `g1-fixed-threadssuggested` CUDA equivalence — exercises rung 1 (PERF) +
  rung 3 (LITE/MINIMAL = output_temp via inner SCRATCH_IN_SMEM=false) on
  idsva_so body and the equivalent fdsva_so spilled rungs.
* See the commit message for the test pass counts.

### 2026-05-31 — ancestor-pair (`t`/`p`) de-alias RE-EXAMINATION (verdict: NOW VIABLE)

Re-examined the "no clean surgical de-alias" stance (the old idea-1 blocker was
that the ancestor-pair scratch aliased `Xup`). **That blocker is GONE.** A prior
refactor (the BC-de-alias landing) already moved the `t`/`p1..p6` ancestor-pair
scratch into its **own** arena region: in the current layout
(`algorithms/_idsva_so.py`, fixed body inner var block) it is anchored
`T *t = D2 + 36*NUM_BODIES;` followed by `p1=t`, `p2..p6` each `+6*var_offset`,
then `BC = p6 + 6*var_offset`. It is **no longer aliased to `Xup`**, so it can be
relocated independently.

Size + lifetime (measured 2026-05-31, sm_120):

| robot (base)  | NV | NB | jids_a | t/p floats | t/p KB | body arena KB | t/p frac |
|---------------|----|----|--------|------------|--------|---------------|----------|
| iiwa14 fixed  |  7 |  7 |  28    | 1008       |  3     | 14            | 27%      |
| g1 fixed      | 29 | 29 | 146    | 5256       | 20     | 64            | 32%      |
| g1 floating   | 35 | 30 | 176    | 6336       | 24     | 78            | 32%      |
| h1_2 fixed    | 39 | 51 | 349    | 12564      | 49     | 108           | 45%      |

* **Lifetime:** `t`/`p` are written+read ONLY in the final block-parallel output
  assembly (the t-loop / p-phase, `_idsva_so.py:~2056-2358`). They are **dead**
  through the entire recursion-hot forward sweep (Xup → IC → v/a/Sd/psid/psidd →
  IC/BC backward propagation) and the D-matrix build. So spilling `t`/`p` to
  `d_workspace` keeps every recursion-hot buffer in smem — the exact "spill cold,
  keep hot in smem" target the whole-arena rung fails to achieve.
* **Access pattern:** the t-loop distributes ancestor-pairs across the block,
  each thread owning a disjoint `t_index_map[jid][anc]*36` slice → coalescible;
  with L2-pinning a spilled access is ~L2 latency, not HBM.
* **Payoff:** at 30–45% of the body arena, spilling `t`/`p` alone could let
  LITE/MINIMAL (and PERF on h1_2) avoid the whole-arena hammer while keeping the
  forward recursion fast. This is the single highest-payoff cold sub-band.

**Verdict: implement-able, not implemented this session.** The surgical de-alias
is now a clean, well-scoped change: add a 3rd placement lever to the body inner
(`TP_IN_SMEM`, mirroring `BC_IN_SMEM`), repoint `t`/`p1..p6` to `d_workspace` at
the top when false (BC/internal anchors must then re-base off the in-smem end of
the hot chain, NOT off `p6`, since `p6` would move to global), wire a new tier
rung into `select_shared_tier_3way` for the body table, and validate per-tier
equivalence (fixed+floating, small iiwa14 + big g1/h1_2) + report smem deltas.
It was NOT landed here to avoid stacking a new spill rung on top of the deferred
mimic-SO work in one session (value-path-stability mandate); it is the clear next
step and no longer blocked.

### 2026-05-31 — ancestor-pair (`t`/`p`) de-alias surgical spill — LANDED (body frame)

Implemented exactly as the re-examination above scoped it. The body inner now
carries a 3rd placement lever `TP_IN_SMEM` (4th template param, default `true`):

```cpp
template <typename T, bool SCRATCH_IN_SMEM = true, bool BC_IN_SMEM = true, bool TP_IN_SMEM = true>
```

Mechanism (`algorithms/_idsva_so.py`, fixed body inner var block):

* A FIXED in-smem anchor `T *tp_anchor = D2 + 36*NUM_BODIES;` marks the end of the
  recursion-hot chain. `t` (and `p1..p6`, which overlay it) anchors here when in smem.
* `if constexpr (!TP_IN_SMEM) { t = d_workspace; }` repoints the whole ancestor-pair
  scratch (t/p1..p6 = `36*len(jids_a)` floats) to the L2-pinned global workspace. p1=t
  and p2..p6 derive off t, so they follow automatically.
* `T *BC = tp_anchor + (TP_IN_SMEM ? 36*var_offset : 0);` — BC re-bases off the SAME
  in-smem anchor (NOT off `p6`, which would move to global). When t/p is in smem BC sits
  exactly where the legacy `p6 + 6*var_offset` put it (byte-identical); when t/p spills,
  BC slides DOWN to `tp_anchor`, reclaiming the vacated `36*len(jids_a)` smem so the
  arena shrinks by exactly the t/p span. The existing `BC_IN_SMEM=false` repoint still
  overrides BC for the BC rung; the two surgical levers are mutually exclusive rungs.

Why it's correct: `t`/`p` is written/read ONLY in the final block-parallel output
assembly (t1-t9 / p-phase, `:~1942-2322`) and is DEAD before that — the entire
recursion-hot forward sweep + D-matrix build never touch it, and the
`reference_order_output_repair` (branched robots) recomputes its own PRIVATE
`rt*/rp*` register scratch rather than reading shared t/p. So spilling t/p keeps every
recursion-hot buffer in smem. Block-stride writes-then-reads across p1..p6 are ordered
by the existing `__syncthreads()` (valid for global mem too).

Tier wiring (`GRiDCodeGenerator.py`, body ladder only — I-regressor owns its additive
rows elsewhere): new rung `output_tp` inserted between `output_bc` (rung2) and
`output_temp` (now rung4): `("output_tp", _idsva_bf_out - _idsva_bf_TP, True, False, False, True)`
with `_idsva_bf_TP = 36*len(jids_a)`. All body tuples gained a 4th `tp_in_global` flag;
`_idsva_body_ws_floats` updated for the index shift (4=whole inner, 3=t/p, 2=BC).
`select_shared_tier_3way` auto-picks it (MINIMAL stays the deepest = whole-arena
guaranteed-fit fallback). The kernel threads `tp_in_smem_expr` to the inner only when
t/p actually spills, so every non-tp rung emits the same `<T,SCRATCH,BC>` instantiation
as before (Gate A byte-identical default).

Measured smem (sm_120, RTX 5090), body fixed:

| robot       | NV | jids_a | t/p   | rung1 global_output | rung2 output_bc | **rung3 output_tp** | rung4 output_temp |
|-------------|----|--------|-------|---------------------|-----------------|---------------------|-------------------|
| iiwa14 fix  |  7 |  28    | 3.9KB | 17088 B             | 16080 B         | **13056 B**         | 2112 B            |
| g1 fix      | 29 | 146    | 20.5KB| 74992 B             | 70816 B         | **53968 B**         | 8704 B            |

At default tier targets (PERF 96KB / LITE 48KB) iiwa14 fits PERF so output_tp isn't
picked there; g1 PERF picks global_output, LITE/MIN pick output_temp — so the new rung
sits between BC and the whole-arena hammer. It becomes the PERF+LITE pick whenever the
smem target lands in (output_temp, output_bc]: e.g. at a 60KB target g1-fixed picks
`(3,3,4) = (output_tp, output_tp, output_temp)`, keeping the full recursion-hot chain in
smem at ~54KB instead of dropping to the 8.7KB whole-arena rung. This is the intermediate
the LITE tier was missing for g1-class fixed robots.

Validation (vs pin_so_ext oracle, RTX 5090, suggested threads):
* iiwa14-fixed: GREEN at default PERF AND forced output_tp tier (target=14000B → picks (3,3,4)).
* g1-fixed: GREEN at default PERF AND forced output_tp tier (target=60000B → picks (3,3,4)).
* iiwa14-floating + g1-floating: GREEN (world-frame path, unchanged — confirms no
  floating regression).
* Gate A: iiwa14-fixed + g1-fixed default `grid.cuh` byte-identical to the
  `modernizing-tests` (25c00f2) baseline (TP spill is fully opt-in/tier-gated).

Scope note: h1_2 fixed body SO is mimic-refused (idsva_so_body_frame is on the G0
gradient-refusal set), so the body-frame t/p rung can't be exercised there; h1_2's
production SO path is world frame (unchanged) and the h1_2 body-frame smem-cap skip
stays. The body t/p de-alias closes the g1-class fixed-base gap; the world-frame surgical
cold trio (already landed) covers the floating humanoids.

### 2026-05-31 — fixed-base mimic SO (idsva_so / fdsva_so) — LANDED (root-caused + fixed)

**Status: FIXED.** The J-idsva internal-NB sweep strategy (described in the
SUPERSEDED section below) was correct; the value bug was a single shared-helper
modulus, not anything in the SO assembly, fold, arena, or inputs (all of which the
prior session had already proven correct).

**Root cause — the `matmul` helper's `% NUM_JOINTS` block wrap.**
`gen_matmul` (`helpers/_lin_alg_helpers.py`) computed the per-block offset as
`int cur = 36*((index/num)%NUM_JOINTS);`. The fixed-body idsva_so inner is the
ONLY caller of `matmul`, and its IC forward build calls it over `36*NUM_BODIES`
elements (`matmul<T>(i, Xup, I, I_Xup, 36, false)`). For a mimic robot
`NUM_BODIES > NUM_JOINTS` (the extra mimic-sibling bodies), so the LAST mimic body
(`index/36 == NUM_BODIES-1`) wrapped `% NUM_JOINTS` back to block 0 and computed
`Xup[last] @ I[0]` — reading body 0's inertia instead of its own. That corrupted
the mimic body's composite inertia `IC`, which then propagated up the entire
backward IC accumulation (`IC[parent] += IC[child]`), corrupting EVERY body's
`IC` → every D-matrix → all four output tensors GLOBALLY (exactly why the symptom
looked un-localized: every `[0:7]^3` cell was wrong, not just body-8 rows). For
fr3: `NUM_JOINTS=8`, `NUM_BODIES=9`, so body 8 (the finger mimic) wrapped to 0.

**Fix:** `% NUM_JOINTS` → `% NUM_BODIES` in `gen_matmul`. `matmul` is used only by
`_idsva_so.py` (the two Xup call sites pass `index/num == 0`, so the modulus is a
no-op there; only the IC build exercises the wrap). `NUM_BODIES == NUM_JOINTS` for
every non-mimic fixed robot, so the constant is numerically identical there
(non-mimic SO output byte-for-byte unchanged; the generated text differs only by
the macro token on that one line). MAIN-OWNED shared helper — **FLAG for reconcile.**

How it was found (per the resume hint's cell-by-cell dump): the CUDA internal
`4*NB^3` buffer diffed GLOBALLY vs the oracle internal ⇒ not an output-write bug.
Forward-sweep dumps showed per-body `Xup[8]`/`I[8]`/`S[8]` all CORRECT but per-body
LOCAL `IC[8]` (pre-accumulation) already wrong by ~160× ⇒ the `Xup[8] @ I[8]`
matmul with correct inputs ⇒ the block index. Dumping `I_Xup[8]` confirmed it
equalled `Xup[8] @ I[0]`, exposing the `% NUM_JOINTS` wrap.

**Validation (RTX 5090 sm_120, vs `RBDReference.idsva_so_body_frame` =
pin_so_ext-backed oracle, fresh-compiled clean cache):**
* fr3-fixed idsva_so: GREEN at PERF (smem 43168 B, relmax 7.9e-7) AND a forced
  spilled tier (`GRID_CUDA_TARGET_SHARED_MEM_BYTES=20000` → use_global_output,
  smem 2912 B, relmax 7.9e-7), seeds 7/13/99, all 4 tensors at the fp32 noise floor.
* fr3-fixed fdsva_so (composes the same inner): GREEN (relmax ~5e-6).
* Gate-A: iiwa14-fixed + go2-fixed identical to the ec00b71 baseline except the
  single `matmul` macro-token line (numerically identical; `NUM_BODIES==NUM_JOINTS`).
* Non-mimic branched `fetch` (NB=NV=14, shared repair/D-matrix machinery): GREEN
  (relmax 9e-7) — no regression.
* Floating-base mimic SO stays REFUSED (floating root needs a per-root-DoF 6-DoF
  subspace fold, not the scalar v-slot/alpha fold).

**Mechanism (the strategy that was always correct):** inner temp sized by NB for
mimic + a `4*NB^3` internal output slab; a function-local
`const int SECOND_ORDER_COORDS = NUM_BODIES;` shadow retargets every output-stride
site to the NB stride; output pointers repoint at the internal slab; qd/qdd reads
fold `body_alpha[jid] * s_qd[body_vslot[jid]]` (fixes the legacy OOB
`s_qd[body_id]`); after assembly a scatter-accumulate fold
`public[v(i),v(j),v(k)] += a_i*a_j*a_k * internal[i,j,k]` (atomicAdd; mimic
siblings collide on a public cell) reduces 4*NB^3 → 4*NV^3. `idsva_so_body_frame`/
`fdsva_so` removed from the G0 mimic refusal set for FIXED-base (kept for floating).
h1_2-fixed body SO is impractically large (4*NB^3, NB=51 ≈ 2.1M floats) so its
production SO path stays world-frame; fr3 is the landed fixed-mimic case.

---

### 2026-05-31 — FLOATING-base mimic SO (idsva_so / fdsva_so) — LANDED (world-frame)

**Status: LANDED.** The last mimic-SO refusal (floating-base) is closed. The
production floating SO path is the **WORLD frame** (the dispatcher routes all
floating SO to `idsva_so_world_frame`), so the fold lives in the world-frame
inner — NOT the fixed-base body inner's internal-NB sweep. The body-frame
floating *reference shim* (`gen_idsva_so_body_frame_floating_reference_inner`)
is deliberately left un-mimic'd: it is the documented NON-production diagnostic
path (SO-AUDIT FLAG ~L1077) and is already numerically red for ALL floating
robots incl. non-mimic iiwa14 (verified on the baseline branch), so it is not a
valid oracle to fold against.

**Mechanism (world-frame inner, `gen_idsva_so_world_frame_inner`):** the world
inner already walks the triple ancestor recursion PER-VELOCITY-COLUMN of each
body (so the floating root's 6 DoF are native — 6 columns), but keyed on the
SHARED reduced v-slot, so mimic siblings clobber each other in the S/psid bands
and collide on output cells. Fix mirrors `RBDReference.idsva_so_world_frame`'s
`has_mimic` path EXACTLY:

* `_idsva_so_floating_velocity_metadata` now also emits INTERNAL-coordinate
  tables: `body_vint_index` (a UNIQUE internal slot 0..n_int-1 per body-velocity
  column; n_int = total column count >= NV), `int_true_vel` (internal -> reduced
  slot), `int_alpha` (internal -> mimic multiplier), `int_s_index/sign`.
* In the inner, gated on `robot_has_mimic_joints()`: `SO_N` stride = `SO_N_INT`
  (= n_int) instead of `NUM_VEL`; the S/Sd/psid/psidd bands size by n_int;
  `wf_body_v_index` holds internal slots; `wf_vel_s_*` index by internal slot.
  The forward sweep reads `alpha*qd[true]`, `alpha*qdd[true]`. The triple-walk
  output writes + the final dvdq transpose use the `SO_N_INT` stride into a
  `4*n_int^3` INTERNAL slab (anchored in the always-hot region, BEFORE the cold
  trio, so the surgical COLD spill never disturbs it). After the transpose a
  scatter-accumulate fold `public[true_i,true_j,true_k] += a_i*a_j*a_k *
  internal[i,j,k]` (atomicAdd; siblings collide) reduces 4*n_int^3 -> 4*NV^3 —
  exactly the oracle's `einsum('ia,ijk,jb,kc->abc', R, T, R, R)` with
  `R[i, true(i)] = alpha_i`.
* **The per-root-DoF fold is EMERGENT, not separate.** The floating root's 6
  columns get 6 DISTINCT internal slots that fold identity (alpha=1) to reduced
  slots 0..5. So the "per-root-DoF loop" the B1 id_du needs is automatically
  achieved by the per-column internal slotting — no special 6-DoF root code.
* Arena (`gen_idsva_so_world_frame_temp_mem_size`) grows by `4*n_int^3` + the
  n_int-vs-NV band delta for mimic only.

Non-mimic stays CHARACTER-IDENTICAL: every mimic branch is gated on
`robot_has_mimic_joints()`; for non-mimic `SO_N == "NUM_VEL"`, no slab, no fold,
no `SO_N_INT`/`wf_int_*` tables. Confirmed byte-identical headers for
iiwa14/go2/g1 floating + iiwa14/go2 fixed; fr3-fixed differs only in the
(forced-on, non-production) world-frame section, body-frame production untouched.

**fdsva_so floating-mimic:** composes the (now-correct) world inner, so it needed
no fdsva edit. It DID surface a pre-existing dependency bug: the mimic Minv path
(`_direct_minv.py`) forward-declares + calls `crba_inner<T,true>` but
`_normalize_codegen_algorithms` didn't pull in `crba` for `minv` on mimic robots
(non-mimic Minv never touches crba), so fdsva_so floating-mimic hit `nvlink:
unresolved extern crba_inner`. Fixed with a 1-line additive dep:
`if "minv" in algorithms and self.robot_has_mimic_joints(): algorithms.add("crba")`
(GCG.py, FLAGGED — non-mimic byte-identical). fdsva_so floating has NO
independent equivalence oracle in the harness (the floating diagnostic only
checks the idsva block via the broken body-shim; fixed-base does check fdsva),
so its floating value-correctness rests on (a) the GREEN world idsva_so inner it
consumes, (b) GREEN mimic minv/crba, (c) the GREEN fixed-base fr3 fdsva landing.

**Validation (RTX 5090 sm_120, vs pin_so_ext oracle via
`reference_model.idsva_so_body_frame`, fresh-compiled clean cache):**
* fr3-floating world-frame idsva_so: GREEN at default PERF (smem ~52KB internal
  slab fits) AND a forced spilled tier (`...TARGET_SHARED_BYTES=20000` ->
  whole-arena -> d_workspace), samples zero/conservative + 3 random.
* fr3-floating fdsva_so: compiles clean (crba dep), composes the GREEN inner.
* Gate-A byte-identity: iiwa14/go2/g1 floating + iiwa14/go2 fixed idsva_so
  headers + iiwa14-floating fdsva header IDENTICAL to the 15374aa baseline; fr3
  body-frame production untouched. iiwa14/go2 floating world-frame GREEN;
  fr3-fixed matmul-blockwrap (body-frame mimic SO sentinel) GREEN.
* fr3-floating added to the world-frame test's default MIMIC SO coverage.
* `idsva_so_body_frame`/`fdsva_so` removed from the FLOATING branch of
  `_MIMIC_GRADIENT_ALGORITHMS` (GCG.py, FLAGGED additive ungate).

**h1_2-floating feasibility:** NB=52, NV=45, n_int=57 -> internal slab alone =
4*57^3 ≈ 741K floats (~2.9 MB), far beyond any smem budget (PERF=96KB) and the
public 4*NV^3 (~1.4 MB). It can ONLY run via the whole-arena spill to global
d_workspace, and the n_int^3 sweep + fold is impractically large/slow — same
verdict as h1_2-fixed body SO. Not added to default coverage; production h1_2
floating SO uses the (non-mimic-shaped... it IS mimic) world spill rung if ever
needed, but is practically a non-target. fr3 is the landed floating-mimic case.

---

### 2026-05-31 — fixed-base mimic SO (idsva_so / fdsva_so) — DEFERRED (bug localized) [SUPERSEDED — root-caused + fixed above]

Attempted fixed-base mimic support for the body-frame SO inner via the
oracle-proven strategy (mirrors `RBDReference.idsva_so_body_frame`'s `has_mimic`
path): run the whole per-body sweep in **unique-per-body internal coordinates**
(n_int = NUM_BODIES; internal slot == body id on a fixed base) into an internal
`4*NB^3` buffer, then **fold** to the reduced `4*NV^3` public output with the
alpha reduction `R[i, v_slot(i)] += alpha_i` along all three axes. The mechanics
that were built and individually VERIFIED correct:

* **Arena:** body-inner temp sized by NB (not NV) for mimic + a `4*NB^3` internal
  output slab on top (`gen_idsva_so_body_frame_inner_temp_mem_size`). Offsets
  computed: internal slab `[5100, 8016)` for fr3, non-overlapping with every
  hot buffer and within the grown arena. Gate A confirmed byte-identical
  (iiwa14/go2 fixed + iiwa14 floating-world).
* **Stride shadow:** a function-local `const int SECOND_ORDER_COORDS = NUM_BODIES;`
  retargets all ~106 output-stride/zeroing sites (incl. the inline
  reference-order repair) to the internal NB stride for free; output pointers
  repoint at the internal slab; qd/qdd reads fold `alpha * s_qd[v_slot]` (fixes a
  real OOB: the legacy `s_qd[body_id]` reads index NB-1 > NV-1 for mimic).
* **Fold:** PROVEN correct — `R-fold(oracle_internal) == oracle_public` to 0.0,
  and the emitted gather-fold tables (`so_fold_start={...,7,9}` summing internal
  rows 7,8→reduced 7 for fr3) are correct.

**Why deferred — the bug:** the CUDA **internal NB^3 sweep itself** produces wrong
values for fr3, even though (a) the fold is proven correct, (b) the effective
internal qd vector matches the oracle (`[...,qd7,qd7]` for bodies 7,8), and (c)
the IDENTICAL machinery (repair path, D-matrices, IC/BC propagation) is GREEN for
the branched **non-mimic** robot `fetch` (NB=NV=14) to 1e-7. Decisive isolation
(q≠0, qd=qdd=0, gravity=0): `d2tau_dvdq` is correct (0), but `dM_dq` and
`d2tau_dqd2` are numerically wrong (rel ~20) and `d2tau_dq2` carries garbage
(~40 where the zero-force oracle is ~0). So the defect is in the internal sweep's
**q+inertia-dependent** path for the mimic body — NOT the fold, NOT the inputs,
NOT the arena offsets (all verified). Since CRBA (composite IC) is GREEN for fr3,
the composite inertia is right; the discrepancy is SO-specific (likely the
D-matrix / B(IC,S) / repair use of the extra mimic body's row, or a subtle
interaction of the shadowed stride with a buffer the sweep reads). Root-cause
was localized but not found within the session; landing a numerically-wrong SO
path was rejected per the value-path-stability mandate, so the change was REVERTED
and `idsva_so_body_frame`/`fdsva_so` remain in the G0 mimic refusal set.
Floating-base mimic SO stays deferred regardless (the floating root's 6-DoF
subspace needs a loop over root DoFs, not the scalar v-slot/alpha fold).

**Resume hint for next session:** dump the CUDA internal `4*NB^3` buffer in full
(the truncation-runner trick: patch the fold to `s_idsva_so[...] = src[(a*SO_INT+
b)*SO_INT+c]`) and diff cell-by-cell vs the oracle internal captured by hooking
`np.einsum('ia,ijk,jb,kc->abc', ...)` in `RBDReference.idsva_so_body_frame`. The
zero-force `d2tau_dq2` garbage is the highest-signal lead (a block that should be
~0 but isn't ⇒ a stale/mis-strided read in the internal layout, despite the
offset math checking out).
