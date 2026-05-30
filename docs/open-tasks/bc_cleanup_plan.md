# B+C Cleanup Phase — Plan

**One combined phase, runs AFTER the in-flight F-batch lands** (T2 inner-loop
restructure, T3 mimic CUDA, T4 `d_f_ext`, T5 `TIER_PERF→TIER_SHARED` + autotune,
T6 `grid_plant` — T6 already merged at `4c91305`). This plan is written against
the **post-F** tree and flags every place an item is already resolved or
reshaped by F. Read-only audit; file:line anchors are from the current
`modernizing-tests` tip (`65e5cde`) — re-anchor after F merges.

Scope: (B) architecture cleanup + warnings sweep; (C) naming/uniformity
residuals + deferred D.5 RBDReference split + URDF feature adds.

---

## F-batch reconciliation (read first)

What F changes that this plan must NOT redo:

- **T5 `TIER_PERF→TIER_SHARED` rename + autotune.** F renames the tier-0 constant
  and (per the naming-audit backlog) wires `performance_threads` autotune. After
  F, the literal string `"TIER_PERF"` is gone. Every architecture item below that
  emits `"TIER_PERF"`/`"TIER_LITE"`/`"TIER_MINIMAL"` tuples (the dispatch ladders,
  the `if constexpr (RESOURCE_TIER == TIER_PERF)` in
  `helpers/_code_generation_helpers.py:538,571`, and the per-tier
  `*_DYNAMIC_SHARED_MEM_BYTES` ladders in `GRiDCodeGenerator.py:815-895`) must use
  **whatever symbol T5 lands**. The B+C "tier-name macros" item is therefore
  *largely subsumed by T5* — B+C only needs to (a) verify T5 touched all ladder
  sites, and (b) do the *table-driven dispatch consolidation* (§1.2), which is
  orthogonal to the rename.
- **T2 inner-loop restructure** of `idsva_so` / `ee_pose_grad` / `crba`. These are
  the three biggest emitters and the three open Pinocchio gaps. Do NOT plan
  emitter-dedup edits inside their inner-loop bodies until T2 lands — the
  `_emit_*_kernel_body_for_flags` shells are stable API, but their interiors move.
- **T3 mimic CUDA codegen** (tracked in `docs/d2_codegen_mimic_plan.md`). This
  resolves the URDF matrix's "mimic — CUDA pending" row. B+C URDF work (§5)
  **excludes mimic**; only planar / skew-axis / damping-friction / limits /
  degenerate-inertial remain.
- **T4 `d_f_ext` trailing param** on dynamics signatures. Not yet in tree
  (`grep d_f_ext` is empty in `GRiDCodeGenerator/`). This *adds a param to every
  dynamics `_device`/kernel signature*, which directly collides with the §1.1
  `_device` wrapper-collapse refactor. **Sequence §1.1 strictly after T4** so the
  collapsed wrapper carries the final signature shape. Also resolves the
  RBDReference `aba` `f_ext` TODO (`notes.md` → `RBDReference.py:2140-2144`,
  "T4 owns the fix"); the D.5 split (§4) must not re-touch that.
- **T6 `grid_plant`** (merged). Sibling-namespace block in `_plant.py`. Stable;
  no B+C interaction except the §4 split and §3 naming leave it alone.

---

## 1. Architecture consolidation (B)

### 1.1 `_device` wrapper collapse — HIGH value, blocked-on-T4

**What's one-off now:** 11 `gen_*_device` emitters
(`_inverse_dynamics.py:347`, `_forward_dynamics.py:136`, `_aba.py:724`,
`_crba.py:430`, `_direct_minv.py:527`, `_inverse_dynamics_gradient.py:960`,
`_forward_dynamics_gradient.py:95`, `_fdsva_so.py:259`, `_idsva_so.py:3382`,
`_integrator.py:632`, `_integrator_gradient.py:589`, plus the eepose trio at
`_eepose_gradient_hessian.py:174,1066,2174`) each repeat the identical skeleton:

```
gen_add_func_doc(...)
gen_add_code_line("template <typename T[, int RESOURCE_TIER = TIER_PERF]>")
gen_add_code_line("__device__")
gen_add_code_line(func_def, True)
gen_XImats_helpers_temp_shared_memory_code(shared_mem_size, extra_t_buffers=..., include_linalg_scratch=True[, tier_workspace_expr=...])
gen_load_update_XImats_helpers_function_call()
gen_<algo>_inner_function_call(...)
gen_add_end_function()
```

(confirmed: all 11 call `gen_XImats_helpers_temp_shared_memory_code` — grep hit.)

**Target uniform shape:** a single helper
`gen_device_wrapper(self, func_def, shared_mem_size, inner_call_fn, *,
extra_t_buffers=(), tier_workspace_expr=None, template_line=..., func_doc=...)`
in `helpers/_code_generation_helpers.py`. Each `gen_*_device` shrinks to:
build its `func_def`/params (the genuinely per-algo part), then one call to the
shared wrapper with an `inner_call_fn` closure.

**Risk:** MEDIUM. The wrappers diverge in three real ways that the helper must
parametrize, not erase: (a) the `template` line (`<typename T>` vs
`<typename T, int RESOURCE_TIER = TIER_PERF>`); (b) `tier_workspace_expr`
present only on the 5 tier-aware device paths (FD `_forward_dynamics.py:168`,
id_du, fd_du, d2ee, idsva_so); (c) `extra_t_buffers` content. Keep these as
params; do NOT try to unify the `func_def` string-building (that's irreducibly
per-algo and trying to table it would *increase* one-off-ness).

**F dependency:** **T4** adds `d_f_ext` to the dynamics `func_def` strings →
collapse AFTER T4 so the wrapper sees final signatures. **T5** rename touches the
`template` line default → use T5's symbol. **T2** does not touch device wrappers
(it's inner-loop bodies), so no block there.

### 1.2 Table-driven tier-dispatch dedup — HIGH value, mostly F-independent

**What's duplicated now:** the *identical* tier-dispatch ladder is copy-pasted in
**12 kernel emitters**. Each has the shape (verbatim across files):

```
picks = getattr(self, "<algo>_spill_tier_3way", (0,0,0))
if picks[0] == picks[1] == picks[2]:
    <flags> = _<ALGO>_PICK_FLAGS[picks[0]]
    _emit_<algo>_kernel_body_for_flags(self, ..., <flags>, ...)
else:
    tier_names = ("TIER_PERF", "TIER_LITE", "TIER_MINIMAL")
    for tier_idx, (tier_name, pick) in enumerate(zip(tier_names, picks)):
        head = "if constexpr (RESOURCE_TIER == "+tier_name+") {" if tier_idx==0 else "else if constexpr ..."
        self.gen_add_code_line(head, True)
        _emit_<algo>_kernel_body_for_flags(self, ..., _<ALGO>_PICK_FLAGS[pick], ...)
        self.gen_add_end_control_flow()
```

Sites: `_crba.py:557`, `_forward_dynamics.py:251`, `_forward_dynamics_gradient.py:306`,
`_aba.py:862`, `_fdsva_so.py:484`, `_integrator.py:752`, `_direct_minv.py:616`,
`_inverse_dynamics_gradient.py:1127`, `_eepose_gradient_hessian.py:1208,2321`,
`_idsva_so.py:2425,3290`. (Note `_integrator_gradient.py:864` and the two
`_idsva_so` sites have a hand-rolled variant — the idsva one nests an extra loop
dim, integrator_gradient uses a local `_emit_body` closure with 3 flags.)

**Target uniform shape:** one helper
`gen_tier_dispatch(self, picks, pick_flags_table, emit_body_fn)` where
`emit_body_fn(flags)` closes over the per-algo `_emit_*_for_flags` call. The 9
simple sites collapse to a single line. The 3 irregular sites
(`_idsva_so` ×2, `_integrator_gradient`) keep bespoke bodies but should still
route their `if-constexpr` ladder text through the same helper (pass a richer
`emit_body_fn`) so the **dispatch scaffolding** is single-sourced — the bloat
audit goal of "one unified pattern, not N one-offs."

**Risk:** LOW for the 9 simple sites (pure extraction, byte-identical output).
MEDIUM for the 3 irregular sites — verify generated text is unchanged.

**F dependency:** **T5** owns the `tier_names` tuple strings → the helper reads
T5's tier-symbol list (single source). **T2** rewrites the *interiors* of the
`_idsva_so`/`crba`/`ee_pose` `_emit_*_for_flags` bodies but NOT the dispatch
shell → do the dispatch-helper extraction AFTER T2 to avoid churn on those two
idsva sites, but the 9 clean sites can go anytime.

### 1.3 `*_DYNAMIC_SHARED_MEM_BYTES` ladder dedup — MEDIUM

**What's duplicated:** `GRiDCodeGenerator.py:815-895` emits ~12 near-identical
per-tier `if constexpr (TIER==TIER_PERF) return grid_shared_arena_bytes<T>(...)`
ladders (one per algo: MINV, FD, ID_DU, FD_DU, INTEGRATOR, INTEGRATOR_DU, ...),
differing only in the `*_t_count_per_tier[0..2]` tuple. 18 distinct
`*_DYNAMIC_SHARED_MEM_BYTES` symbols enumerated.

**Target:** a Python helper `_emit_tier_smem_bytes(name, t_count_per_tier)` that
prints the 3-branch ladder from a `(name, tuple)` table. Pure codegen-Python
dedup; generated C++ unchanged.

**Risk:** LOW. **F dependency:** T5 rename touches the `TIER_PERF` literal here;
fold into T5's symbol or sequence after. T4's `d_f_ext` does not change smem
sizing (external force is an input, not scratch) — verify.

### 1.4 `s_temp`-as-workspace misnomer — LOW (and partially MOOT)

The naming-audit flagged `s_temp_spill` as an `s_`-named device pointer.
**Already resolved:** grep shows zero `s_temp_spill`; it's now `d_temp_spill`
everywhere (`_fdsva_so.py:206,311`, `_forward_dynamics_gradient.py`, `_plant.py:124`).
**Residual:** `s_temp` is still the *name* of a pointer that is reassigned to
`d_workspace` (global) under non-PERF tiers — e.g. `_fdsva_so.py:310`
`if constexpr (!SCRATCH_IN_SMEM) { s_temp = d_workspace; }` and the
`tier_workspace_expr` repoint in `helpers/_code_generation_helpers.py:545`. The
`s_` prefix then lies about storage class at LITE/MINIMAL. **Recommendation:**
LEAVE AS-IS this phase — `s_temp` is a load-bearing public inline-CUDA symbol
(appears in emitted `_device` signatures and user docs); renaming it is a
breaking API change disproportionate to the clarity gain, and the
`tier_workspace_expr` comment already documents the repoint. Catalog only.

---

## 2. Warnings sweep (B)

### 2.1 RBDReference `mxS` / `mx1`-`mx6` DeprecationWarning — mostly DONE, verify

Memory and `notes.md` (`RBDReference.py:655-658`) flagged ~177k NumPy
`"Conversion of an array with ndim>0 to a scalar"` warnings. **Current state:**
the mitigation is **already in place** — `mxS` flattens with
`np.asarray(S).reshape(-1)` (`RBDReference.py:659`) and `mx1` flattens `vec`
(`:690`). The `notes.md` entry is explicitly "**catalog only**."

**Remaining B+C action:** the notes say *"Other `mx1`..`mx6` call sites that pass
raw array slices as `alpha` should be audited in that sweep."* Audit the
`_mxS`/`mxS`/`mx1-6` callers at `RBDReference.py:1761, 2289, 2541, 2550, 2553,
2556, 2562, 2619, 2621, 2624` — several pass `qd[ind]` / `alpha * qd[idx]`
(scalar, safe) but `_mxS(S, np.matmul(Xmat, v[:,parent]))` (`:2541,2556`) and the
`S[ii]` column slices (`:2550,2619`) route through `_mxS` (`:620`,
`np.dot(cross_operator(vec), S)`) which can still emit the warning when `S` is a
column. **Method:** run the `RBDReference/tests/` suite under
`python -W error::DeprecationWarning -m pytest` and fix each site that trips
(extract scalar via `.item()` / index, or flatten the column), keeping pass/fail
counts unchanged. Target: **zero** DeprecationWarnings from the suite.

**F dependency:** T3 (mimic) and T4 (`f_ext`) both add RBDReference call sites →
run this audit AFTER T3/T4 RBDReference work merges so new call sites are covered.
This also overlaps the D.5 split (§4) — **run the warnings audit BEFORE the split
move** so warning fixes land on the monolith and the split stays a pure move.

### 2.2 nvcc / ptxas warnings on big robots (g1 / h1_2) — NEW capability needed

**Current state:** the CUDA equivalence compiles use `-O0` with **no** `-Wall`,
no `-Xptxas -v`, no `-Xptxas -warn-spills` (`test_cuda_integrator_equivalence.py:111-119`
and the sibling `test_cuda_*_equivalence.py` runners). Warnings are invisible
today.

**Method to surface + triage:**
1. Add an opt-in env flag (e.g. `GRID_CUDA_WARN=1`) to the shared nvcc command
   builder in the equivalence runners that appends
   `-Xcompiler -Wall -Xptxas -v,-warn-spills,-warn-lmem-usage`. Keep it OFF by
   default so the parallel equivalence sweep (per
   `feedback_parallel_equivalence_testing`) stays fast and noise-free.
2. Run a **one-off triage build** of the generated headers for the warning-prone
   robots — g1, h1_2 (both bases) — capture ptxas register/spill/lmem reports and
   any `-Wall` diagnostics into a scratch log. These are the robots HANDOFF flags
   for smem-cap pressure (`HANDOFF.md:709,829`).
3. Triage into: (a) real register spills / local-memory usage (perf, may
   intersect T5 autotune), (b) benign unused-param/shadow `-Wall` noise from the
   emitted boilerplate (fixable in codegen), (c) ignore. Fix (b) in the emitters;
   record (a) for the perf re-sweep (§6).

**F dependency:** T2 changes idsva/crba/ee_pose register pressure; T4 adds a
param (possible new unused-param warnings on qdd=0 paths). Run AFTER T2+T4 so the
warning set reflects final code.

---

## 3. Naming / uniformity residuals (C)

From `project_grid_naming_audit_backlog`. Much of the sweep is already DONE
(`SUGGESTED_THREADS→MAX_PERF_LEVEL_THREADS`, `s_workspace→d_workspace`,
always-present `s_topology_helpers`, `s_temp_spill→d_temp_spill`). Residuals:

1. **Tier names** — **OWNED BY T5** (`TIER_PERF→TIER_SHARED`). B+C only verifies
   T5 covered all sites: the constant defs (`GRiDCodeGenerator.py:784-786`), the
   `tier_max_threads` ladder (`:804-807`), the dispatch tuples (§1.2), the
   `*_DYNAMIC_SHARED_MEM_BYTES` ladders (§1.3), the `if constexpr` in
   `helpers/_code_generation_helpers.py:538,571`, the docs/rst, and the
   `GRID_CUDA_INTEGRATOR_TIER` test guard
   (`test_cuda_integrator_equivalence.py:125`). **Blast radius if T5 missed any:**
   compile error (constant undefined) — easy to catch. Do NOT re-rename; only
   close gaps.
2. **`*_DYNAMIC_SHARED_MEM_BYTES`** — naming was flagged as a candidate. After
   audit it is **consistent** (18 symbols, uniform suffix). No rename needed;
   only the §1.3 *emission* dedup. Catalog as "reviewed, keep."
3. **`s_temp` storage-class misnomer** — see §1.4; recommend keep (breaking API).
4. **Autotune `performance_threads`** — the backlog's autotune script
   (binary-search best batched thread count ≤ `MAX_PERF_LEVEL_THREADS`) is
   **OWNED BY T5**. B+C does not write it; just confirm the emitted
   `performance_threads` value + API property exist post-F and are documented.
5. **Signature uniformity** — backlog wanted uniform inner/device/helper
   signatures. The §1.1 wrapper collapse is the natural place to enforce a
   uniform `_device` shape; while there, audit that every `_device` takes params
   in the same order (result ptr, inputs, `d_robotModel`, `gravity`,
   `[d_workspace]`, `[d_f_ext from T4]`). Flag any straggler that still omits
   `d_workspace`/`s_topology_helpers` — but per the backlog this is back-burner,
   not blocking.

---

## 4. D.5 RBDReference file-split execution (C)

**Reference T1's contract — do NOT redo it:**
`docs/open-tasks/rbdreference_split_plan.md` already has the full method→file map
(4 mixins: `_helpers` / `_kinematics` / `_dynamics` / `_gradients_and_hessians`,
MRO-composed, public import unchanged). Execute that map verbatim.

**Sequencing (critical):**
- **Run STRICTLY AFTER all F RBDReference work merges** — T3 (mimic), T4
  (`f_ext` in `aba`), and any T1 follow-ups all edit `RBDReference.py`. The split
  is a pure *move*; doing it before F means F agents rewrite lines in files that
  no longer exist. The split plan itself says it was "deliberately deferred so T3/T4
  do not have to rewrite every line they touch."
- **Run §2.1 warnings audit BEFORE the move** (so fixes land on the monolith).
- **Pure move, zero behavior change.** Validation = the `RBDReference/tests/`
  suite with **identical pass/fail/skip counts** before and after (the plan's
  acceptance criterion).
- Honor the plan's boundary notes: `crm`, `_spatial_xmat_*derivative_func`,
  `_floating_lie_perturbed_q` stay in `_HelpersMixin` (cross-used);
  `_HelpersMixin` first in MRO; `__init__` stays on the concrete class.
- **Update the `notes.md` line anchors** that point into `RBDReference.py`
  (`:655, :936, :987, :1907, :2140, :2158`) since the split renumbers everything —
  re-target them to the new module files as part of the move.

**Risk:** LOW mechanically (Python binds by name), but the file is 3717 lines and
the mixin boundaries touch `self.robot` shared state — the test-count gate is the
safety net. **F dependency:** hard ordering after T3/T4.

**`grid_plant` numpy reference (additive follow-on, lands with the split):** add a
fifth `_plant.py` `_PlantMixin` (plant_step + quadratic/EE costs + joint barriers)
as the CPU oracle for the CUDA `grid_plant` surface, and rewire
`test_cuda_plant_equivalence.py` from FD-only self-validation to a real
CUDA-vs-RBDReference equivalence test (+ a new `RBDReference/tests/test_plant_equivalence.py`).
Pure additive after the move (counted separately from the split's test-count gate).
Full spec in `rbdreference_split_plan.md` → "grid_plant reference module". Converges
with D.4's regressor reference (`d4_runtime_inertia_params_plan.md §G.5`) and the
`grid_plant` handle surface (`notebook_examples_plan.md`).

---

## 5. URDF feature additions (C)

**Reference T1's matrix — do NOT redo it:**
`docs/open-tasks/urdf_feature_matrix.md` has status + Python-first/CUDA proposals
for each gap. **Mimic CUDA is T3/F** — exclude it. Remaining, in recommended
order (cheapest-and-safest → largest):

1. **Velocity / effort `<limit>` parse** (matrix §4) — SMALL, metadata only, no
   dynamics/CUDA change. Read `velocity`/`effort` in `parse_joints`, store as
   `joint.velocity_limit`/`effort_limit`, add accessors + a parse test. Lowest
   risk, do first.
2. **Degenerate / missing `<inertial>` detection** (matrix §5;
   `URDFParser.py:74`) — SMALL, parse-time validation. Distinguish "declared root"
   from "non-root link missing inertia"; raise/warn on degenerate (rizon4-class)
   instead of silently zeroing. Pure parse-time; no CUDA. Add a rizon4-like
   regression test. (Resolves the `broad_coverage_findings` rizon4 skip.)
3. **Arbitrary / skew `<axis>`** (matrix §1; `Joint._axis_scale`) — MEDIUM.
   Replace the 3-way cardinal cascade in `set_type` with a single general-axis
   Rodrigues path (cardinal falls out as special case); `S=[axis_xyz,0,0,0]`
   (revolute) / `[0,0,0,axis_xyz]` (prismatic). Python-first + parse test; then
   **verify** the CUDA emitters use the full 6-vector `S` (not a single-axis
   index assumption) in transform/crba/rnea — the matrix flags this as the CUDA
   risk. T2 touches crba inner loops → verify-S step runs AFTER T2.
4. **`<dynamics damping>` / `friction`** (matrix §3) — MEDIUM, behind an explicit
   flag so existing equivalence tests (which assume no damping) stay green. Thread
   stored `damping` into RNEA/ABA bias (`tau += damping*qd`); parse + apply
   `friction`. Python-first + flag-gated tests; CUDA propagation = per-joint
   coeffs as robot constants + gated bias accumulation. Interacts with T4's
   `f_ext` bias plumbing → do AFTER T4.
5. **Planar joint** (matrix §2; `set_type` else-branch `exit()`) — LARGEST. New
   3-dof non-floating joint shape; mirror the `floating` branch
   (`position_symbols`, `local_q_dim`, `dof`, 6x3 `S`). Python-first +
   RNEA/CRBA equivalence tests; CUDA = the multi-dof-non-floating emitter path is
   genuinely new (matrix: "likely the larger half"). Do LAST; consider its own
   sub-phase if scope balloons.

**Each feature: Python-first (land+test in URDFParser/RBDReference), then CUDA
propagation.** The matrix has per-feature CUDA notes — follow them.

---

## 6. Recommended whole-phase ordering + perf re-sweep

Gate everything on **F fully merged** (T2-T5; T6 already in). Then:

**Stage 0 — reconcile F.** Confirm T5's tier symbol, T4's `d_f_ext` final
signature, T2's `_emit_*_for_flags` interiors, T3's mimic emit. Re-anchor this
plan's line numbers.

**Stage 1 — warnings (§2), can start immediately, low-risk.**
  1a. RBDReference `mx*` audit under `-W error` (BEFORE D.5 split).
  1b. Add `GRID_CUDA_WARN` flag + one-off g1/h1_2 ptxas triage.

**Stage 2 — architecture dedup (§1), F-ordered.**
  2a. §1.3 smem-bytes ladder dedup (after T5).
  2b. §1.2 tier-dispatch helper — 9 clean sites first, then the 3 irregular
      (after T2).
  2c. §1.1 `_device` wrapper collapse (after T4) — the highest-value, highest-care
      item; do it once 2a/2b have proven the helper-extraction pattern.

**Stage 3 — naming verify (§3).** Mostly verification of T5; close any gap.
Catalog the keep-as-is items (`s_temp`, `*_DYNAMIC_SHARED_MEM_BYTES`).

**Stage 4 — D.5 split (§4).** AFTER §2.1 fixes land on the monolith; pure move;
test-count gate.

**Stage 5 — URDF features (§5).** In the §5 order (limits → inertial → axis →
damping → planar); each Python-first + tested before CUDA.

**Stage 6 — bloat audit + comprehensive perf re-sweep.**
Per `feedback_refactor_bloat_audit`: after the multi-file dedup, run a dead-code /
dup-logic pass over the touched emitters (the wrapper collapse + dispatch helper
should have *removed* one-offs — verify no near-dup helper crept back in).
Then the **full perf re-sweep**: all 9 robots × {fixed,floating} × {SHARED,LITE,
MINIMAL} tiers, N=256, vs the current baseline
`test/benchmarks/results/tier_sweep_20260525_002438/` (and the C.7 Pinocchio
column). Run the perf sweep **isolated** (no concurrent CPU work, per
`feedback_no_cpu_during_bench`). Acceptance: no perf regression from the
codegen-only refactors (output should be byte-identical for §1.2/§1.3; §1.1 and
the URDF/damping adds need numeric+perf validation). Equivalence tests
(correctness-only, per-(robot,base) parallel, sized to cores+RAM) run separately
and must stay green; clear the header cache after codegen changes.

---

## Top risks / F-dependencies to watch

- **§1.1 ↔ T4:** wrapper collapse must carry T4's final `d_f_ext` signature — do
  not start §1.1 until T4 is merged, or the collapsed wrapper will need a second
  signature edit.
- **§1.2/§1.3/§3 ↔ T5:** all consume the tier symbol T5 renames — single-source
  it from T5's output, don't hardcode `TIER_PERF`.
- **§4 (D.5 split) ↔ T3/T4:** strictly after both finish editing `RBDReference.py`.
- **§5.3 skew-axis & §2.2 ptxas ↔ T2:** verify generated `S`/register pressure
  after T2's inner-loop restructure.
