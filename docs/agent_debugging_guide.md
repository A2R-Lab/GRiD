# GRiD agent guide — debugging, patterns, pitfalls (what works, what doesn't)

Hard-won institutional knowledge from the multi-agent campaigns on `modernizing-tests`
(F/G/H/I/J/K + the mimic-completion + perf/SO-audit rounds). **Read this before debugging a
GRiD codegen/CUDA issue or doing a refactor.** GRiD = Python codegen (`GRiDCodeGenerator`)
emitting CUDA C++ from URDFs; numpy/pinocchio reference oracle lives in `RBDReference`.

---

## 0. The validation checklist (do these EVERY time — they each caught a real bug)
1. **Clean the generated-header cache before re-validating.** A stale `grid.cuh` gives phantom
   pass/fail. The CUDA equivalence harness keys its cache on a hash of the whole
   `GRiDCodeGenerator/*.py` tree (`_header_cache_key`), so codegen edits self-invalidate — but
   manual/ad-hoc `gen_all_code` runs into temp dirs do not. When in doubt, clear it.
2. **Gate-A byte-identical** for any refactor or opt-in algorithm: capture the generated `grid.cuh`
   for representative robots (iiwa14-fixed + a floating + a big robot) BEFORE your change, regen
   AFTER, `diff`. Must be empty (refactor) or confined to your new opt-in kernel (additive).
3. **Floating + fixed codegen smoke, not just `py_compile`.** A floating-codegen regression (the
   single-axis-S guard hitting the 6-DoF floating root) slipped past `py_compile` — it only
   surfaced when actually generating a floating header. Use
   `gen_all_code(algorithm_list=[...])` per robot/base.
4. **AST dup-key check `RBDReference/tests/tolerances.py`** after any RBDReference merge — multiple
   agents add the same `(robot, algo)` key → silent duplicate dict keys.
5. **For refactors: compare the full before/after test SET, not the count.** (See §3, F1.)
6. **Confirm your merge touched ONLY the files you expect** (`git diff --stat HEAD~1 HEAD`).
7. **Propagate to docs + READMEs (main + ALL submodules) + examples** for any rename / new feature /
   convention change — grep them for the OLD names/values too. Code-only changes leave docs stale (§8).

---

## 1. Recurring bug classes (these bit us 3+ times each — check them first)

### 1a. Per-body scratch sized by NV/`num_pos`, must be NB/`num_joints` (MIMIC overflow)
**The single most common bug this session** (h1_2 RNEA `s_vaf`, B2-SO body scratch, integrator
`s_vaf`; `_centroidal.py:64,88` `s_vaf=18*n` is a latent suspect). Mimic joints carry **0 DoF**,
so for mimic robots `NUM_BODIES (NB) > NUM_VEL (NV) = NUM_POS`. Any scratch buffer that an inner
writes **body-indexed** (stride over NB) MUST be sized by `get_num_joints()`/NB, not
`get_num_pos()`/`get_num_vel()`/NV. Undersizing overflows into the adjacent buffer (e.g. `s_vaf`
→ `s_Minv`), corrupting downstream silently.
- **Symptom:** a mimic robot's output is wrong in a way that looks "random" / global, while the
  non-mimic version is exact. Often the corruption is in a DIFFERENT tensor than where the
  undersized buffer lives (it overflows into a neighbor).
- **Fix:** size by NB when `robot_has_mimic_joints()`; mirror how `_forward_dynamics_gradient.py`
  sizes its fd_du kernel. Also remember `alpha*s_sign` (the mimic multiplier + motion-subspace
  sign) folds — dropping it is the OTHER recurring mimic gradient bug.
- **Grep:** `s_vaf|s_XImats`-adjacent temps, `18*n`, `6*n` per-body buffers across `algorithms/*.py`.

### 1b. Shared linalg-helper bugs (one bug, fleet-wide blast radius)
`gen_matmul` in `helpers/_lin_alg_helpers.py` used `36*((index/num)%NUM_JOINTS)` — for mimic
(NB>NJ) the last mimic body wrapped `%NUM_JOINTS` back to block 0 and read body-0's inertia,
corrupting the entire composite-inertia chain. **No-op for non-mimic (NB==NJ), so it hid for ages**
and only surfaced via fr3 second-order equivalence.
- **Lesson:** when a mimic algorithm is globally wrong but CRBA/Minv are green, suspect a SHARED
  helper with an NB-vs-NJ index, not the algorithm itself. Shared helpers (`_lin_alg_helpers.py`,
  `_code_generation_helpers.py`) are high-blast-radius — validate non-mimic byte-identical AND a
  mimic robot after touching them.

### 1c. Silent CUDA launch failures (zeros masquerading as results)
A heavy kernel with **no `__launch_bounds__`** (osc_inertia, ~100+ regs) fails to launch at high
thread counts ("too many resources requested for launch"). If the runner only `cudaDeviceSynchronize`s
and never checks `cudaGetLastError`, the **zeroed output looks like a real (wrong) answer**, and a
PERF harness records a **bogus-fast timing** the autotune argmin then wrongly picks as "best."
- **This was MISDIAGNOSED TWICE** as a "broken mimic-Minv-compose gap" before the real cause (a
  512-thread launch failure) was found. The mimic Minv was always correct.
- **Always** `cudaGetLastError()` + `cudaDeviceSynchronize()` after launches and FAIL LOUDLY.
  Clamp launches to `cudaFuncGetAttributes().maxThreadsPerBlock` (the register cap, which can be
  BELOW the `__launch_bounds__` thread cap). The benchmarked kernels carry launch_bounds (compiler
  fits registers) so they're safer; un-annotated opt-in kernels are the risk.
- Audit any opt-in runners that still omit this error check and add it.

### 1d. Cross-cutting convention flips miss non-uniform encodings (sign/unit changes)
Flipping a convention (R5: gravity `+9.81` → `-9.81`) by grepping ONE pattern (`*gravity`) negated
every multiply-form but MISSED the vector-assignment forms — `a_world[5] = gravity`,
`gravity_vec[]={...,gravity}`, `S_agrav[5] = -gravity` (3 idsva_so sites + the fixed-base aba
`gravity_vec`). Same physical constant, different syntax.
- **Lesson:** for any sign/unit/convention flip, enumerate EVERY encoding form: `*x`, `= x`,
  `vec[i]=x`, `{...,x}`, and existing `= -x` (which may need to become `= +x`, not a double-flip).
  Grep `\bx\b` broadly, reason per-site, never sed.
- **Validate fixed AND floating AND mimic.** The missed aba site was FIXED-base-only (floating used a
  different code path), so a floating-only validation shipped the bug. A green floating run is NOT
  evidence the fixed path is correct — different branches. (Mirror of §0/§5.)
- A pattern-based sub-agent reliably misses the non-uniform forms; reconcile its diff by grepping ALL
  forms yourself before trusting it — and never trust an agent that returns without a validation result.

### 1e. Per-timestep INPUT buffer slot sized by NV, must be NUM_JOINTS=nq (FLOATING-base stride bug)
**Found in 6 algorithms in one sweep** (crba, aba, forward_dynamics, fd_gradient, integrator,
integrator_gradient, idsva_so, fdsva_so, regressor, id_du). The canonical per-timestep input buffer
gives each field (q, qd, u/tau, qdd) a `NUM_JOINTS`(=`get_num_pos()`=nq)-wide slot: q@0, qd@nq,
u/tau@2·nq, qdd@3·nq; per-timestep stride `3*NUM_JOINTS`. The binding packs exactly this
(`pack_q_qd_u`, stride `3*num_joints`). Any field offset / load-count / host-stride / smem-arena term
built from `get_num_vel()`(nv) — `NUM_POS+nv`, `2*nv+fb`, `Q_QD_U_STRIDE=nq+2nv`, a `nv*nv` host
DtoH copy of an `nv*nv`-written matrix — is the bug.
- **Why it hid:** for FIXED base nq==nv so every nv-form coincides with the nq-form → **byte-identical,
  invisible**. Only FLOATING base (nq>nv: go2 19/18, quaternion root) diverges. AND the CUDA equivalence
  harness only exercises a SINGLE timestep via the *device function*, so the host/kernel BATCH path
  (what the binding uses) shipped wrong on floating base undetected. A whole class lurked for ages.
- **Symptom:** floating-base output wrong; the offset half breaks at **B=1** (qdd/u read from the wrong
  intra-slot offset), the stride half breaks only at **batch>1** (timestep k≥1 reads `k*(nq+2nv)`
  instead of `k*3nq`). Fixed-base identical.
- **Decisive oracle-free catch:** feed an IDENTICAL-input batch (B≥4) → every slot MUST be identical;
  and batched[b] MUST == standalone(input[b]). Then vs the oracle at **B=1** to catch the offset half
  (self-consistency alone misses it). Always confirm fixed-base BYTE-IDENTICAL (no-regression).
- **Distinguish value vs tangent outputs (don't over-flag):** VALUE outputs (qdd, coriolis vector, M,
  Minv-as-a-matrix on the host path) live in nq-wide slots; TANGENT outputs (gradients dtau/dq, SO
  tensors nv³, Jacobians 6×nv, regressors nv×params) are genuinely nv-strided and the binding reads
  them tangent-strided — those nv strides are CORRECT. Only INPUT offsets + intermediate value buffers
  flip to nq.
- **Grep:** `get_num_vel()`/`nv`/`NUM_VEL` in per-timestep INPUT offsets, `Q_QD_U_STRIDE`,
  `NUM_POS + nv`, `2*nv + ` in load-counts/slot-widths/host-strides across `algorithms/*.py` +
  the `*_DYNAMIC_SHARED_MEM_BYTES` input-slot terms in `GRiDCodeGenerator.py`.

---

## 2. Debugging methodology (what actually localizes a bug fast)

- **Validate the DEPENDENCY standalone first.** Before assuming "Λ = J·M⁻¹·Jᵀ is broken for mimic,"
  check: is mimic `direct_minv` itself correct vs the oracle? (It was — the bug was elsewhere.) A
  wrong composite output is often a correct-component + a wiring/infrastructure artifact.
- **Decisive isolation inputs.** Set `q≠0, qd=qdd=0, gravity=0` (or similar) to zero out whole
  terms and see WHICH output tensor is wrong. "d2tau_dvdq correct(0) but dM_dq wrong" instantly
  narrows from "the whole algorithm" to "the q+inertia-dependent path."
- **Dump the internal buffer cell-by-cell vs the oracle einsum.** For SO/Hessian tensors, patch
  the fold to emit the raw internal `4*NB^3` slab and diff against the oracle's
  `np.einsum('ia,ijk,jb,kc->abc', ...)` captured by hooking the numpy reference. The first
  divergent `(a,b,c)` cell points at the emitter.
- **A block that should be ~0 but isn't is the highest-signal lead** (stale/mis-strided read).
- **Cross-check a fold in numpy before trusting CUDA.** e.g. `R-fold(oracle_internal) == oracle_public`
  to 0.0 proves the fold table is right, so the bug must be in the CUDA sweep, not the fold.

---

## 3. Refactor traps (looks-fine-but-isn't)

- **An identical public surface does NOT mean identical behavior.** The F1 RBDReference split had
  all 126 public methods present (126/126) and 824 tests passing — but **131 failed vs the main
  baseline's 38** (~93 regressions from mis-wired mixin MRO / cross-`self.` helper references).
  ALWAYS diff the full before/after failing-test SET (same failures, not same count), not just the
  surface or the pass count. For a big class→mixin split, do it **incrementally** (one mixin →
  full suite green → repeat), never all-at-once.
- **`py_compile` ≠ it codegens.** (See §0.3.)
- **Underscore-prefixed names are skipped by `from x import *`.** A shared helper
  (`_emit_fb_bfs_level_indexing`) caused an ImportError until explicitly exported. If you add a
  `_`-prefixed function others import, add it to `__all__` or import it explicitly.
- **Removing an input alias needs EVERY caller form found first.** Dropping the short-form
  `algorithm_list` aliases (`id`→`inverse_dynamics`) broke callers a token-pattern grep missed:
  list literals `["id"]`, comma-strings `"id,minv,..."`, dash-forms `"fd-gradient"`, and module-level
  vars (`_MIMIC_SAFE_ALGORITHMS`) — each surfaced as a separate `ValueError` only when that one test
  ran (whack-a-mole). Find them definitively with `grep -rn "algorithm_list\s*="` (catches list AND
  string) and by codegen-ing EVERY distinct value, not by grepping for a token.

---

## 4. Optimization patterns that worked (and the mental model)

**Mental model:** on a 5090 the GPU SMs are NOT the bottleneck — **nvcc compile time + host RAM**
are. So prefer MORE parallelism (more threads/blocks) even for single-block accuracy; don't leave
work serial to "save" SM occupancy. Justify every serial block.

**The three structural parallelism levers — AUDIT every algorithm against all three (canonical doc:
`docs/source/user_guide/concepts/parallelism_patterns.rst`; status table: `docs/open-tasks/parallelism_audit.md`).
A serial block with no P1/P2/P3 justification is a bug to file, not a style choice:**
- **P1 — Depth/BFS-level batching of tree recursions.** Bodies at the same tree DEPTH are independent;
  emit the recursion as a serial loop over LEVELS (O(depth)) and fan all bodies in a level across
  threads (one sync/level), parent←children via atomicAdd/segmented reduce. Turns O(NB) serial steps
  into O(depth) — big on branched humanoids (depth≪NB), neutral on chains (never hurts). Templates:
  `_aba.py` forward (`segmented_row_strided_gemv`), floating `_crba.py` backward (per-level fan + atomicAdd).
  Cue: any `for jid in range(...)` emitting a per-body 6×6 + block sync.
- **P2 — Parallel independent columns** (gradients/Jacobians/Hessians have independent columns/(j,k) cells):
  compute the recursion ONCE, then fan per-column/per-cell work across threads (e.g. 2·n² for an n×n grad
  pair). Templates: id_du per-element fan, d2ee per-cell, idsva_so per-column. Cue: an outer `for col`/`for (j,k)`
  wrapping otherwise-independent algebra.
- **P3 — Loop-invariant hoist → store temps → batch-parallel after.** Work inside a serial recursion that
  does NOT depend on the loop's serial carry: stash its per-iteration inputs during the walk, then do it as
  ONE parallel pass AFTER. Differs from P1 (which keeps work in the recursion) — P3 removes loop-independent
  work entirely. Costs a (cold, write-once) scratch band → interacts with the spill tiers (keep in smem when
  it fits, spill to d_workspace when not). Cue: a sub-expr inside the body-walk whose inputs are all
  per-iteration-local and whose output is consumed after the walk.
- **P4 — Offline memory layout: sparse compaction + coalesced distribution + topology-helper indirection.**
  GRiD is a code GENERATOR that knows the robot's topology + matrix sparsity OFFLINE — spend that to make
  online reads cheap: (a) **compact** to only structurally-nonzero entries (fewer bytes → higher tier fits;
  don't loop/store over known zeros); (b) **lay out** data (SoA/stride/padding) offline so the thread→data
  map reads CONTIGUOUS aligned addresses per warp — an uncoalesced strided access can erase a P1/P2 fan-out
  win, so lay per-level bodies / per-column data contiguous to the access order; (c) **bake topology-helper
  arrays** (`parent[]`, BFS-level/branch offsets, sparsity offsets, column/support maps) into the robotModel
  so the kernel does cheap coalesced array LOOKUPS, not branchy per-thread index math (these helpers are also
  what make P1/P2 expressible without divergence). P4 keeps the MEMORY path up with P1–P3's shortened compute
  path. Cue: dense passes over known zeros; per-thread index arithmetic that could be a baked lookup; strided
  warp reads; data interleaved against access order.
- **CAVEAT (learned the hard way, 2026-06-06): P1 level-batching is NOT automatically a win — A/B-TIME it, and
  PRESERVE the GLASS ops.** A serial per-body backward loop already calls tuned GLASS `gemm`/`gemv` that
  parallelize each 6×6's inner reduction across threads. If you "level-batch" by replacing those with hand-rolled
  per-output-element `dot_prod` (a serial 6-elem reduction per thread), you TRADE GLASS's intra-op parallelism
  for a shorter sync chain — and for *modest level widths* (most branched robots) that LOSES (measured ABA g1
  ~2% SLOWER → reverted). Sync-count reduction only helps when per-op thread-utilization isn't already the
  bottleneck. Correct P1: keep the GLASS ops and batch by interleaving them across a level WITHOUT per-op block
  syncs (harder; may still not beat serial-GLASS for narrow levels) — and ALWAYS A/B time at N=256 before
  keeping the refactor (per perf-cleanup discipline: revert a regression). Also: **verify the benchmarked robot
  actually COMPILES the path you're optimizing** — h1_2 is MIMIC (12 mimic joints, `robot_has_mimic_joints()==True`
  on the loaded URDF), so its fixed-base ABA routes through the compose path `qdd=Minv·(τ−rnea)` (= CRBA/Minv),
  NOT the ABA backward recursion. Check `robot_has_mimic_joints()` for the EXACT URDF the sweep loads; don't
  trust in-code "all non-mimic" comments (`baselines/grid/run.py:455` is wrong for h1_2).
  - **UPDATE (2026-06-06, ISOLATED micro-bench — the premise above FLIPS in isolation):** a clean standalone
    micro-bench (`/tmp/perf_microbench/`) of the GEMV-replacement tradeoff — GLASS-serial-per-body `gemv` vs a
    `dot_prod`-batched level fan — at PRODUCTION thread counts (32–352) shows **`dot_prod`-batched WINS from level
    width L≥4 (1.3× at L=4 up to ~11–13× at L=32); GLASS-serial only ties at L=2.** Reason: GLASS's intra-op
    parallelism on a single 6×6 is only **6 lanes wide**, so the **serial-over-bodies loop is the real cost**, not
    the inner reduction; removing the per-body sync barely moved variant A (compute/serialization-bound, not
    sync-bound). So the "GLASS intra-op parallelism beats a shorter sync chain for modest widths" premise is FALSE
    in isolation. **BUT this does NOT by itself greenlight #2/#6** — the full-kernel ABA/CRBA reverts almost
    certainly regressed on **occupancy/register pressure** (the exact mechanism that sank #3 frame_jacobian: a
    local parallel fan raised whole-kernel register use → spill → regression at high thread counts), and the
    h1_2/CRBA reverts were partly MIS-TARGETED (h1_2 mimic routes through the serial mimic fold, not the GLASS
    per-jid path). **Net: #2 (RNEA-backward, compounds across fd/fd_du/idsva_so/fdsva_so) + #6 (minv-backward)
    RE-OPEN as MEASURE-FIRST FULL-KERNEL experiments** — gate on a full-kernel A/B at N=256 + production threads
    watching occupancy/spill, and use `segmented_row_strided_gemv<TRANSPOSE,ATOMIC_Y>` (already in GLASS). The
    micro-bench can't rule out the occupancy regression — only the full-kernel A/B can.
  - **RESOLVED (2026-06-06, full-kernel A/B measured): #2 is NEUTRAL → reverted; #6 not pursued.** Implemented
    #2 (RNEA-backward → segmented per-level GEMV), fully correctness-validated across ALL composing algos
    (id/fd/fd_du/aba/crba/idsva_so/fdsva_so/centroidal/regressor, fixed+floating+mimic, thread-invariant),
    then isolated A/B at N=256 + autotuned production threads on iiwa14-fixed (canary) AND go2-floating (the
    level-width-4 "win region"): **B/A = 0.99–1.01 on EVERY algo** — no regression (occupancy fear didn't
    materialize; the descriptors are cheap `static const int[]`), but **NO WIN either**, including the direct
    RNEA path (inverse_dynamics B/A=0.99). **The isolated micro-bench win is real but does NOT translate
    because the RNEA backward is a tiny FRACTION of the full kernels** (fdsva_so is 800µs on go2; its backward
    pass is single-digit %). Classic Amdahl. **#6 (minv-backward) inherits the same structure (the backward
    is a small fraction of the minv kernel) → not pursued** (predictable neutral; not worth the hours).
    **THE LESSON: a micro-bench win on a kernel STAGE only moves the needle if that stage is a meaningful
    fraction of the full-kernel runtime. Always (a) measure the FULL kernel, and (b) weight the isolated win
    by the stage's fraction of total before investing.** The #2 diff is preserved at
    `/tmp/perf_wip/rnea_backward_2.diff` and the GLASS wrapper TRANSPOSE/ATOMIC_Y extension it added is reverted
    too (unused → bloat); reapply both if a future use makes the backward a larger fraction.
- **CAVEAT 2 (P2 column-fan, 2026-06-06): fanning a thread-0-serial assembly over threads can REGRESS when the
  serial body materializes large baked `const T[]` job-tables — REVERTED frame_jacobian #3.** Audit item #3
  (`_frame_jacobian.py` Step 3+4, the "adds parallelism where there was NONE / Effort M, risk Low, upside High"
  target) was implemented exactly per spec (P2 column fan + P4 per-target `[start,len)` support map, eepose Step-3b
  template for the mimic v-slot fold), passed full equivalence (iiwa14-fixed, g1-floating, fr3-fixed J/Jdot;
  h1_2-fixed J/Jdot too — the lone h1_2 Lambda-WORLD fail is a PRE-EXISTING osc_inertia float32-inverse flake,
  byte-identical on clean HEAD) AND thread-invariance (6/6, threads 1/2/16/32/256). But A/B at N=256 was a **mixed
  net loss → reverted**: g1-floating (nv=36, deep chains) **1.29× FASTER** (1189→922 us), but **iiwa14-fixed
  (nv=7) up to 7× SLOWER (7→53 us)** — and the slowdown SCALES WITH THREAD COUNT (t=8 ≈parity 8.6 vs 8.1 us;
  t=64 1.6×; t=352 7×) while the serial baseline is FLAT (~7 us at every thread count, since only thread 0 works).
  Root cause: Step 3 bakes `const int fj_jj[njobs]` + `const T fj_ang[3·njobs]`/`fj_lin[3·njobs]` (+`fj_off_*`) as
  FUNCTION-LOCAL arrays. In the serial version one thread instantiates them; the parallel `for(job_idx)` makes ALL
  block threads instantiate them (nvcc spills the big ones to local memory) → local-mem traffic ∝ thread count
  swamps the fan-out gain except where the serial column-walk is genuinely long (big floating). The autotuner picks
  high thread counts for kinematics, so production hits the slow side. **Lesson:** a thread-0-serial block isn't
  automatically a free parallelization target — if its body declares big baked `const[]` tables (P4 "bake into
  const arrays" pattern), parallelizing multiplies that materialization cost across the block. Either hoist the
  tables to `__shared__`/`__constant__` (one materialization, all threads read), or cap the parallel loop's active
  lanes, or just leave tiny-fan-out assemblies serial. A/B at MULTIPLE thread counts (not just MAX) — a
  thread-count-SCALING regression is the tell. (#3 stays open in the audit with this caveat; the win is real only
  for big floating robots and would need the const-table-hoist redesign to not regress the common small case.)
- **GLASS is VENDORED (inlined) into every generated `grid.cuh` at codegen time — a GLASS change does NOT reach
  GRiD's emit until you also do the GRiD-side plumbing.** Mechanism (`GRiDCodeGenerator/helpers/_lin_alg_helpers.py`):
  `gen_grid_linalg_backend_helpers` reads each file in the curated list `_GLASS_BASE_FILES` *fresh from the GLASS
  submodule* and inlines it into the header (`// BEGIN/END GLASS ...`), pinning the GLASS commit in a comment.
  GRiD code never `#include`s GLASS — it's embedded, so the generated header is self-contained. Three consequences
  any GLASS-touching agent MUST handle: **(1)** the vendoring is automatic *on regen*, so editing an
  already-listed file (e.g. `src/base/L2/gemv_segmented.cuh`) reaches GRiD on the next `gen_all_code` — but
  **(2)** a NEW GLASS file is invisible to GRiD until you add it to `_GLASS_BASE_FILES` (so keep additions inside
  an already-vendored file when you can); and **(3)** GRiD calls GLASS ONLY through the `grid_linalg_*` wrappers
  in the same file (e.g. `grid_linalg_gemm → glass::gemm`) — a new GLASS *capability/flag* (e.g. the L2
  `TRANSPOSE`/`ATOMIC_Y` flags) is present-but-uncallable until you EXTEND the wrapper to pass it. So "land a GLASS
  feature for GRiD" = GLASS change + (file in `_GLASS_BASE_FILES`) + wrapper exposing it + regen to verify it
  vendored. The committed example headers (`./grid.cuh`, `examples/cuda/grid.cuh`) are stale snapshots — regen them
  if they must track GLASS. (Caller compat: don't reorder GLASS template params *before* a param a `grid_linalg_*`
  wrapper or emitter passes positionally; GRiD callers pass `<T,M,N,ROW_STRIDE,FUSE>` and no `IDX_T`, so appending
  flags before `IDX_T` was safe — verify this when generalizing a vendored signature.)

- **Parallelize independent COLUMNS in gradients/hessians.** d2ee (per-slot Step-2 + per-cell
  Step-5b), id_du branched-fixed (per-output-element fan, 2·NJ → 2·n² threads), idsva_so world
  forward-sweep. Keep the recursion (body-walk) serial with a per-body `__syncthreads`; fan the
  inner per-column work.
- **Surgical spill, not whole-arena.** Keep HOT / serially-built / randomly-accessed buffers in
  smem; spill only COLD / write-once / dead-before-hot / large-returned-output buffers to L2-pinned
  `d_workspace`. De-alias buffers so a cold one can be repointed independently. Whole-arena spill
  stays only as the guaranteed-fit MINIMAL fallback. (g1-spill: 135→94 KB, 99→75 KB, PERF arena
  byte-identical.)
- **Internal-coordinate sweep + alpha-fold for mimic.** Run the per-body sweep in unique-per-body
  INTERNAL coords (n_int = NB or total S-column count) into a `4*NB^3`/`4*n_int^3` slab, then
  scatter-fold `public[v(i),v(j),v(k)] += α_iα_jα_k · internal[i,j,k]` to the reduced `4*NV^3`
  output (atomicAdd). The per-root-DoF treatment for the floating root EMERGES from per-column
  internal slotting (its 6 columns get 6 distinct slots, alpha=1) — no separate 6-DoF root code.
- **Reuse existing infrastructure before writing new code.** Repeatedly, the "missing" piece was
  already there: the v-slot reduction already sums shared columns; the mimic effective-angle q-fold
  is already baked into `s_XmatsHom` upstream; the floating-mimic ee fold needed NO new code (6
  independent root v-slots flow through the existing per-column path). **Check what the existing
  emit already does before adding a fold.**
- **THE mimic column-fold template** is `_eepose_gradient_hessian.py` Step 3b (alpha-weighted
  geometric-Jacobian column accumulate onto the shared reduced v-slot). It was reused verbatim for
  f_ext_gradient and frame_jacobian mimic. When you need a mimic-reduced gradient fold, start there.
- **Dedup byte-identically.** Two near-identical emitters (timed/untimed, Xdown fixed/floating) →
  one helper parameterized on the differing token. Prove byte-identical generated output.
- **Mixed-precision device sub-blocks: do a tiny FD in `double` inside a float32 kernel to match a
  float64 oracle.** The floating `plant_step_hessian` needs `d2Integrate` = FD of the 6×6 SE(3)
  `dIntegrate` blocks. A float32 FD there is too noisy to hit the equivalence bucket, but the blocks
  are tiny (6×6×6), so compute the FD by calling the `T`-templated helper as `grid_dIntegrate_*_block<double>`
  (h=1e-3, 4th-order), then cast the result to `T`. The float32 kernel then matches the float64 oracle
  to ~1e-6 while paying double only on a negligible sub-computation.
- **Reuse the freed inner pool for follow-on scratch (inner-owns-placement).** After a composed
  `*_device` call returns and you `__syncthreads()`, its `s_temp` pool is dead — a follow-on stage can
  carve small block-shared scratch from its front (`s_se3 = SCRATCH_IN_SMEM ? s_temp : d_workspace`),
  with NO new smem allocation. It works whether the pool is in smem or routed to `d_workspace` (the
  spilled tier). Just floor the kernel's pool sizing at `max(inner_pool, new_scratch)`.
- **Gate a new emit path with a numpy mirror BEFORE the nvcc loop.** A floating-header `-O0` compile is
  ~5 min; mirroring the exact CUDA index arithmetic (block lookups, transposes, the t1/t2/t3 contractions)
  in numpy and diffing vs the oracle catches index/transpose/sign bugs in *seconds*. Only spend the
  compile once the mirror is 0-error. (Caught the velocity-row `(a,b)` transpose + the un-symmetrized
  Euler position fill before any GPU build.)
- **An `if(loop_var==k){…}` ladder over many cells with IDENTICAL arithmetic = a P4 baked-table win
  (compile-time + code-size, byte-identical numerics).** When a per-cell parallel body differs ONLY in a
  handful of integer offsets (not its op SHAPE), the legacy "inline the body once per cell behind
  `if(d2m_cell==k)`" emits O(n_cells) copies → the nvcc compile-time / Python-codegen / header-size
  blow-up. Replace with a baked `static const int tab[6*n] = {…}` of per-cell offsets and ONE shared body
  that reads `&tab[6*loop_var]` (pattern templates: `_inverse_dynamics.py` seg-offset tables,
  `_f_ext_gradient.py feg_*`). Keep cells whose op SHAPE varies per cell (axis literals, variable-length
  sums — e.g. eepose same-joint rev/pris and mimic block-pair) in the residual ladder; index the table
  cells `[0,n_cross)` first, the shape-varying cells `[n_cross,n_cells)` after, so the launch geometry /
  total cell count / values are unchanged. Numerics are byte-identical (same scalar ops, offsets just
  sourced from the table instead of being compile-time constants) — the win is purely emit-shape.
  **Measured (2026-06-06, eepose `ee_pose_hessian` Step-5b, the documented h1_2/big-floating gap):** the
  win SCALES with cell count and is huge on the gap target. h1_2-floating (2600 cells, 1780 cross):
  Python codegen 1607s→412s (−74%), generated header 16.8MB→8.9MB (−47%), `*_inner` region
  254.9k→96.6k lines, nvcc(`-O3`, sm_120, single-kernel TU) 73.4s→21.3s (−71%), obj 5.48MB→2.27MB
  (−59%). Smaller robots scale down (iiwa14-fixed 49 cells: nvcc 1.3s→1.0s). Equivalence stayed green at
  identical tolerances on iiwa14-fixed, go2-floating, AND h1_2-fixed (mimic — proves the table coexists
  with the residual mimic-block-pair ladder). NB the table is `static const` declared inside the parallel
  `for` loop body but BEFORE the `if(loop_var<n_cross)` guard — fine (compiler hoists to one static
  instance; threads with `loop_var>=n_cross` skip the body so no OOB on `&tab[6*loop_var]`).
- **PERF TIMING — measure in ISOLATION; concurrently-measured verdicts are PROVISIONAL (2026-06-07).**
  Correctness (equivalence + thread-invariance) runs in per-test `tmp_path` dirs + a content-hashed
  header cache → contention changes how LONG a run takes, not pass/fail, so **parallelize correctness
  freely**. TIMING is the opposite: several agents timing on the SAME GPU contend for SMs / memory
  bandwidth / host-CPU-during-compile → µs numbers inflate and destabilize. So **decouple the two**:
  implementation agents do codegen + equivalence + thread-invariance and DEFER timing; **serialize ALL
  A/B timing into one isolated pass** (GPU quiet, one measurement at a time) that commits-win / reverts-
  no-win off clean numbers. Treat any win/no-win verdict measured under concurrency as PROVISIONAL until
  re-timed isolated — the 4 reverts above (CRBA, ABA, #3, #10) + the "hand-rolled `dot_prod` < GLASS"
  claim were concurrent-measured, so the META-finding (don't parallelize cheap serial work) is robust
  across 4 mechanisms but the individual MAGNITUDES (incl. CAVEAT/CAVEAT 2's 1.29×/7×) await isolated
  re-timing. **(SETTLED for the `dot_prod`<GLASS claim, 2026-06-06: an isolated micro-bench FLIPPED it —
  `dot_prod`-batched wins the GEMV tradeoff from L≥4; the full-kernel reverts were occupancy/register +
  h1_2-mimic mis-targeting, not the inner-reduction tradeoff. #2/#6 re-open as measure-first full-kernel.
  See the §4 P1-CAVEAT UPDATE.)** The standing META-finding still holds for genuinely-cheap serial work
  (tiny folds with zero sync cost); the nuance is that a SERIAL-OVER-MANY-BODIES loop calling a narrow
  (6-lane) GLASS op is NOT "cheap serial work" — it's under-parallelized, and batching it can win IF the
  whole-kernel occupancy survives. **A/B at PRODUCTION thread counts** (autotuner-picked, HIGH) not th=1 — a th=1-only or
  isolated-microbench measure falsely greenlit #3 and #10 (both won only at th=1). And **never use a
  long serial float32 accumulation as a thread-invariance oracle** — it amplifies the benign tree-sum
  reassociation into a false FAIL; use the float64-oracle equivalence harness + a single-call checksum.

---

## 5. Merge discipline (multi-agent, file-isolated clones)

- Agents clone off varying bases → expect **3-way merges**. File-isolated agents merge clean; the
  ONE collision zone is `GRiDCodeGenerator.py`'s **`_MIMIC_GRADIENT_ALGORITHMS`** set (every mimic
  ungate touches it). Hand-reconcile to the **UNION** of removals.
- **`set()` not `{}`** for an empty refusal set — `{}` is a dict and `dict |= set` raises.
- **API 529 / an agent that dies MID-process** leaves UNVALIDATED partial edits in its file. PRESERVE
  the diff (`/tmp`), REVERT the file to clean, and re-dispatch when the API is stable — never build the
  next step on a half-applied edit.
- **Bring new parent-level test files by COPYING from the clone**, then bump submodule pointers
  yourself — do NOT merge the clone's parent commit (its submodule pointers reference the clone's
  local SHAs).
- **Main owns `HANDOFF.md` + the memory system** — discard agents' edits to them.
- Per merge: no conflict markers, `py_compile`, targeted codegen smoke, confirm only-expected-files.
- **Agents that end mid-turn without a complete report** (the heavy-iteration "d2ee class") leave
  work UNCOMMITTED in their clone. Inspect the clone's working tree, validate yourself, commit for
  reproducibility, THEN merge — don't trust a truncated report.

---

## 6. Oracle / reference gotchas

- **Pinocchio's reduced model OMITS mimic bodies**, so its column/output dims differ from GRiD's
  complete (NB-wide) outputs (e.g. f_ext_gradient: pin gives nv×6·(NB−1), GRiD/RBDReference give
  nv×6·NB). For mimic robots, **skip the pinocchio cross-check and treat RBDReference as
  authoritative** (CUDA matches it exactly). The RBDReference numpy ref IS mimic-aware.
- **The RBDReference numpy suite has ~38 PRE-EXISTING failures** (h1_2 minv, plant-floating,
  floating-quaternion d2tau, rk4 floating) — pinocchio-alignment gaps, not your regression. Always
  diff against a fresh main baseline, don't assume green.
- Per-robot float32 conditioning floors are real (e.g. g1 fd: `|Minv|≈3.2e3` round-off ⇒ ~0.2
  absolute residual at a static sample where the float64 ref cancels to ~0). Gate via the per-robot
  tolerance bucket, NEVER loosen a global tolerance. Confirm it's conditioning (high-energy samples
  agree to ~1e-6 RELATIVE) not a structural bug (which is O(magnitude)).
- **The RBDReference numpy oracle can be PATHOLOGICALLY SLOW for big mimic robots — it looks like a
  hang, not a bug.** `crba`/`minv`/`fd` rebuild a fresh `sympy.lambdify` of each joint's transform
  PER BODY PER CALL (`URDFParser.Joint.get_transformation_matrix_function`), and `fd_grad_at`
  finite-differences the whole chain ~2·nv × {euler,midpoint,rk3,rk4} × samples — so h1_2 triggered
  ~1e4–1e5 lambdify builds and the reference took >25 min (SIGABRT'd mid-`lambdify`). FIX: memoize the
  pure lambdify getters (per-instance `_lambdify_cache`; the mimic multiplier is applied to numeric q
  BEFORE the call, so the lambda is constant) → 25 min → **0.16 s**, value-identical. Lesson: a
  "hanging" big-robot equivalence test is often the slow PYTHON oracle, not the CUDA side — `faulthandler`
  the stack first (see §7). RBDReference already had the pattern (`_spatial_xmat_*_func_cache`); finish
  it (backlog PS3) and watch for the SAME trap in any per-call pure-sympy rebuild.
- **Mimic reduction commutes — the fold IS exact (corrected via PS4).** URDF `<mimic>` is ALWAYS linear
  (`q_m = mult·q_t + offset`), so the coupling Jacobian **G is CONSTANT**. Therefore reduction commutes with
  BOTH inversion and differentiation: the existing fold path computes `minv = inv(GᵀMG)` (the correct reduced
  inverse, NOT `Gᵀ M_full⁻¹ G`) and the reduced GRADIENTS are likewise exact (validated to ~1e-13 vs both a
  native-reduced-RNEA finite-difference and the numpy oracle on fr3). So the earlier "reduction⊥inversion don't
  commute" worry was WRONG for linear mimic. **Pinocchio 3.9 native mimic** (`pin.transformJointIntoMimic` —
  NOT `buildReducedModel`, which *locks* DoF instead of coupling) gives an INDEPENDENT reduced-space oracle,
  but pin 3.9 only supports `crba`/`rnea`/`generalizedGravity` on a mimic model — `computeMinverse`/`aba`/
  `compute{RNEA,ABA}Derivatives` RAISE "does not support Joint Mimic". So native mimic = a clean independent
  oracle for **crba/minv only**; gradients/2nd-order stay on the (provably-exact) fold. (RBDReference
  `pinocchio_backend.py` now routes mimic minv/crba through the native model; PS4 commit `c9883da`.)
- **FD-of-Jacobian oracles near the Lie-group identity: a *bigger* step is better, not smaller.** When you
  finite-difference a Lie-group derivative (`d2Integrate` = ∂/∂v of `dIntegrate`) and validate at `v_dt→0`,
  a "precise" tiny step (`h=1e-6`) is WORSE: the perturbed increments land in the ill-conditioned small-angle
  regime of the *exact* closed forms (`(1-cos θ)/θ²`, `(θ-sin θ)/θ³`, the SE(3) Q-block — same cancellation the
  `_se3_Q_block` `θ<1e-4` Taylor guard exists for), so `1-cos(1e-6)≈5e-13` carries ~1e-4 relative error and the
  derivative is wrong at the 5th digit. A **4th-order central stencil** (`(8(f(h)-f(-h)) - (f(2h)-f(-2h)))/(12h)`)
  with `h≈1e-3` (≈ `eps**(1/5)`, the roundoff/truncation optimum) clears the cliff AND minimizes error → ~1e-8.
  Rule: pick the FD step to dodge the conditioning cliff of the function you're differencing, not to be
  "small." (RBDReference `d2Integrate`, default `fd_step=1e-3`.)
- **Anchor an FD oracle with a closed form somewhere, or the test is tautological.** `d2Integrate`(FD-of-our-
  `dIntegrate`) vs `pin.d2Integrate`(FD-of-`pin.dIntegrate`) only re-confirms `dIntegrate≈pin.dIntegrate` (already
  known) — it can't catch a conceptually-wrong second-order object. Add an independent closed-form anchor at a
  special point: at `v_dt=0` the leading-order SE(3) expansions give exact block structure — `∂J_r^SE3/∂v` has
  `-½[e_k]_x` in the top-right (ρ-dir) and the two diagonal blocks (φ-dir); `∂Ad(exp(-v))/∂v` is the same with
  factor `-1`. Those pin the signs and the ½-vs-1 factor with zero dependence on pin or on our own FD.
- **The pinocchio adapter returns VIEWS into reused `self.data` buffers.** Any FD loop that calls
  `forward_dynamics_gradient` / `integrator_gradient` / similar repeatedly MUST `np.array(..., copy=True)` each
  result — successive calls alias the same `pin.Data` storage and silently overwrite your stencil's earlier
  evaluations. The tell: a derivative that comes out *constant across genuinely-different configurations* (a
  phantom "connection term" ≈ a fixed value like 9.81 was chased for several iterations before this was the cause).
- **A finite-difference "Hessian" of a vector-valued gradient is NOT (a,b)-symmetric on a Lie group.** For a
  free-flyer, `dIntegrate(ARG0)` (the q-side Jacobian) is q-INDEPENDENT, so the q-perturbation row of
  `∂(plant gradient)/∂z` is exactly 0 while the qd-perturbation row carries the whole cross term. Fill the
  second-order tensor **un-symmetrized** (do NOT average `H[o,a,b]` with `H[o,b,a]`); the FD-of-pin ground truth
  is itself asymmetric. (Only the genuinely-symmetric fixed-base case lets you get away with symmetrizing.)
- **Mind the axis order when an assembly helper and its consumer disagree.** `_d2qdd_tangent` returns
  `[out, column, perturb]` (b before a); `plant_step_hessian` needs `[out, perturb, column]` → `transpose(0,2,1)`.
  Invisible on a fixed base (D2qdd is a true Hessian → (a,b)-symmetric, transpose is a no-op) but **load-bearing**
  on the floating q-q block, which is genuinely asymmetric. When the symmetric case hides a transpose bug, the
  asymmetric (floating / off-diagonal) case is where it bites — test there.
- **Floating-base world-position trap: reuse the CoM/FK path, don't re-derive R,p from the spatial 6×6.**
  Reconstructing a body's world position by inverting the *spatial* `get_Xmat_Func_by_id` transform gave a wrong
  (≈2× on one term) floating-base potential energy. For any quantity that must match a CoM/PE oracle (PE
  regressor, etc.), share the **homogeneous, mimic-aware `_world_transforms`** the CoM path already uses rather
  than rebuilding (R,p) from the spatial transform.
- **rpy pose-gradient gimbal lock is a recurring MULTI-TARGET hazard.** A test that sweeps ALL leaf EEs over ALL
  samples (vs one curated leaf) WILL eventually hit pitch=±π/2 and blow up the `[xyz;rpy]` Jacobian's `E⁻¹` on
  BOTH the analytic oracle and pin-FD. Guard with a pitch-band skip; the pose position + rotation matrix stay
  valid and should still be checked. (Codegen should prefer a quaternion / rotation-matrix pose output, or
  document the rpy limitation.) Also: `pin.dccrba(model,data,q,v)` is the exact analytic `Adot` oracle (= ∂A/∂t,
  NOT the ∂A/∂q tensor — they differ; ∂A/∂q contracts to Adot over the DOF axis and to dh_dq over the column axis).
- **MuJoCo/mjx free-joint convention: ACCELERATION is not a frame rotation (cost us a wrong gradient model).**
  Converting GRiD↔MuJoCo for a floating base, the velocity is a simple root-block rotation `G=blockdiag(R,I)`
  (mjx base-linear vel is GLOBAL `ṗ=R·v_local`), BUT acceleration carries an extra `ω×v` term:
  `a_mjx_lin = R(a_pin_lin + ω×v_local)`. Skipping it matches gravity + the `M·a` inertial term but is O(1) wrong
  on the **Coriolis** term — invisible to FD-self-consistency (which differentiates your *own* value def), caught
  ONLY by cross-checking real MuJoCo (`mj_inverse`/`mj_fullM`/`mj_forward`; pip-installable, load a matched
  hand-MJCF + URDF for the SAME tiny model → machine-precision diff). Lesson: when matching an EXTERNAL convention,
  an internal FD round-trip is necessary but NOT sufficient — cross-check the real external tool. MuJoCo's `d/dqpos`
  also holds qvel/qacc FIXED IN MJX FRAME → base-rotation gradient columns pick up Coriolis/inertial/`ω×v`
  couplings. Full derivation + validated oracle: `RBDReference/equivalents/mujoco_convention.{py,md}`.
- **mjx oracle is COMPLETE (2026-06-08) — mirror it, don't re-derive.** `mujoco_convention.py` now has
  every floating output transform, each FD/MuJoCo-validated (`tests/test_mujoco_convention.py`, 12 passing).
  Three recurring shapes cover almost everything: `base_rotate` (covector rows `G·`), `jacobian` (column
  reframe `J·G⁻¹`), `congruence` (`G·X·Gᵀ`). The `ω×v` trap recurs anywhere a quantity is "evaluated at
  qacc_mjx=0" or reads qd: `nonlinear_effects` (=`G·ID(q,qd,−ω×v)`, naive covector off by 32.7),
  `dccrba` dh/dq (needs `+A·_cross_cols(v_lin,−1)`, naive off by 23.6), the hdot/gradient families. The
  full convention map (formula + class + verified error per function) is `docs/open-tasks/mjx_codegen_fusion_master_plan.md` §2/§0b.
- **EE-Hessian (and any coordinate-Hessian) validation trap (cost a cycle, 2026-06-08).** GRiD's analytic
  `end_effector_pose_hessian` is the SYMMETRIC coordinate Hessian (2nd derivative of the scalar value along
  the retract), NOT `d/dξ[gradient]`. To FD-validate it, use the symmetric 2nd central-difference of the
  VALUE along the retract — `d/dξ_mjx[gradient_mjx]` carries retract-connection curvature, is non-symmetric
  (asym≈17 on go2), and a symmetric analytic Hessian can never match it (you'll see ~asym/2 residual and
  chase a phantom bug). The mjx transform is a double column-reframe + a ½-symmetrized frame correction.
- **`RBDReference.inverse_dynamics_gradient` floating-root `da_dq` term (FIXED 2026-06-08, f22b391).**
  `inverse_dynamics_gradient_fpass_dq` indexed the BODY axis with a floating-base velocity-DOF index
  (`da_dq[:,c,ii]`, ii∈0..5) — crashed on NB≤6 models and only avoided it on NB≥7 (e.g. go2) because the
  floating root's own `dv_dq` is identically zero so the term was structurally a no-op. Fixed: the floating
  root contributes nothing to that term (validated byte-identical on go2/other floating robots vs pin).

---

## 7. Test-infra gotchas

- **pytest-xdist needs deterministic collection.** Per-process randomness (a random default thread
  count) → "different tests collected between workers." Make defaults deterministic.
- The CUDA equivalence harness has graceful-skip idioms: `GRID_SKIP_IF_KERNEL_TOO_BIG` (smem cap)
  prints a parseable `SKIPPED` line the parser nulls. The timing parser overwrites with the LAST
  numeric `Single Call X` line and ignores non-numeric ones.
- **Static `__shared__` in a smoke-runner kernel is capped at 48 KB (0xC000) even on sm_120** — a
  `ptxas error: uses too much shared data` at COMPILE time, distinct from the launch-time *dynamic*
  smem opt-in cap (~99–101 KB). A large output band staged in static smem (e.g. a 2nv×3nv×3nv plant
  Hessian = 6174 floats ≈ 24 KB for nv=7, *plus* the fdsva_so scratch) blows past it. Fix in the
  smoke runner: pass the **global output pointer straight in as the device fn's output arg** (the fold
  scatters into it; no smem staging) and size the device fn's `s_temp` to the actual inner-pool need,
  not a round over-allocation. (The *generated* kernel stages its output in DYNAMIC smem under the
  ~99 KB cap, which is fine for small robots; big robots need the global workspace band — deferred.)
- **The plant smoke runner's `plant_step_kernel` / `plant_kernel` use big STATIC `__shared__` arrays
  (`s_dAB`, `s_D_qdd_stage`, `s_XImats`, `s_temp[4096]`…) and DO NOT compile for big robots** — g1
  (nv=29) overflows the 48 KB static cap (`ptxas: plant_step_kernel uses too much shared data
  0xe1d0`). This is pre-existing and independent of any one algorithm: `GRID_CUDA_PLANT_ROBOTS=g1`
  fails at *compile* on the sibling kernel before the kernel-under-test even builds. So **validate
  big-robot plant surfaces (e.g. `plant_step_hessian`) via the BINDINGS** (`grid_rbd`, a separate TU
  whose kernels are all dynamic-smem + `cudaFuncSetAttribute`), not the cuda_equivalents smoke runner.
  Making the smoke runner's plant kernels dynamic-smem (mirroring the hessian kernel, which already is)
  is the proper infra fix to unblock big-robot plant smoke — a separate, bounded task.
- **Force a spill tier on a SMALL robot to validate the spill *code path* without a big-robot compile.**
  `GRID_CUDA_TARGET_SHARED_MEM_BYTES=10000` makes `select_shared_tier_3way` pick the deep-spill tier
  even for iiwa14, so the equivalence test exercises the exact `d_workspace`-band / pool-aliasing kernel
  body (the one g1 would use) in a ~3-min iiwa14 compile instead of a ~17-min g1 build. Pair it with a
  codegen-only header regen of the big robot to confirm its per-tier smem macro fits the ~99 KB cap.
- **Plant kernels are NOT in `KERNEL_ATTR_MANIFEST`, so they get no automatic `cudaFuncSetAttribute`.**
  `init_grid_kernel_attrs` raises `MaxDynamicSharedMemorySize` only for the manifest's kernels; the
  plant kernels (`plant_step_gradient_kernel`, `plant_step_hessian_kernel`) aren't listed, so when a
  plant kernel's dynamic-smem arena exceeds the 48 KB default (the hessian's 18·nv³ `s_d2AB` band pushes
  iiwa14 to ~53 KB) the **binding launcher must call `cudaFuncSetAttribute(kernel<T,IT>, MaxDynamicSharedMemorySize, bytes)`
  itself per template instantiation**, or the launch fails with `cudaErrorInvalidValue` (the C-ABI rc=100+e).
  The gradient launcher dodges this only because its arena fits 48 KB. To size the launch, emit a
  `*_DYNAMIC_SHARED_MEM_BYTES<T>()` helper next to the kernel whose `t_count` mirrors the kernel arena
  exactly (extra_t_buffers + XI_size + inner pool); a wrong count silently under/over-allocates.
- **Binding a kernel templated on an enum some of whose values `static_assert` out: the C-ABI dispatch
  switch must NOT name the unsupported cases.** Even an unreached `case FN<RK4>(...)` *instantiates* the
  template and trips the device `static_assert` at compile time. Use a restricted dispatch macro
  (e.g. `GRID_RBD_IT_DISPATCH_HESSIAN` lists only EULER/SI-EULER) and return rc=3 for the rest.
- `run_parallel.sh` auto-sizes xdist by free RAM (~5 GB/compile). Equivalence tests are
  correctness-only and safe to run concurrent; the PERF sweep must run ISOLATED (no other GPU/CPU,
  it skews timing).
- **The editable `grid_rbd` install can point at a STALE sibling worktree.** `.venv` is under main
  but `pip install -e bindings` may have last run from another worktree (e.g. `GRiD-H-roadmap`), so a
  bare `import grid_rbd` silently loads OLD bindings — `AttributeError: 'RobotHandle' has no
  attribute 'inverse_dynamics_gradient'` for a method that exists in main. pytest under the repo tree
  passes (repo-relative `sys.path`) while a notebook/example fails. Confirm with
  `python -c "import grid_rbd, inspect; print(inspect.getfile(grid_rbd))"`; the editable `.pth`
  merely *appends*, so `sys.path.insert(0, '<main>/bindings')` (or `PYTHONPATH`) reliably overrides it
  for validation. The real fix is `pip install -e bindings` from the intended tree. (Sibling of the
  §0/B5 stale-compiled-binary class: always verify you imported the tree you think you did.)
- **A bare detached `pytest -n6` over the CUDA-equiv suite HANGS and NEVER notifies completion.** The
  xdist controller wedges (`Sl`, 0 %CPU) after its workers drain on a slow big-robot compile; workers
  vanish, no progress, no summary. Seen ≥2×. NOT OOM. `pytest-timeout`/`pytest-forked` are NOT
  installed. FIX: run validation as **sequential coreutils-`timeout`-bounded chunks**, verbose
  (`for cell: timeout <s> pytest -k $cell -n3 -v -rfE`): no long-lived controller to wedge, a hang is
  force-killed and the loop continues, `-v` captures each PASS incrementally even if a later chunk
  times out (EXIT 124 = timeout, NOT a failure — grep `FAILED` to confirm 0 real fails).
- **Distinguish "slow compile" from "hung" by process inspection, not the dot count.** Slow =
  `cicc`/`ptxas` at 99 %CPU (R) + staggered `nvcc` ages (youngest started recently) + load avg ≈ workers;
  the xdist controller being `Sl 0%` is NORMAL (it idles while workers compile). HUNG = zero
  `nvcc`/`cicc`/`ptxas`, GPU 0 %, controller `Sl 0%`, **log mtime frozen for many minutes**, workers gone.
- **`faulthandler` pinpoints WHERE a process is stuck.** Run `timeout --signal=ABRT <s> python -X
  faulthandler -m pytest ...`; on timeout the SIGABRT dumps the live Python stack — instantly tells you
  codegen-loop vs `subprocess.wait`(nvcc) vs the slow numpy reference (§6) vs `cudaDeviceSynchronize`.
  This is how the "h1_2 hang" was traced to sympy, not CUDA.
- **Big-robot (g1/h1_2) second-order kernels compile 20–40 min EACH** (`idsva_so`/`fdsva_so`/
  `ee_hessian`); a per-robot gate/sweep chunk of ~30 such cells at -n4 will EXIT 124 on a 60 min budget.
  Budget big-robot chunks generously, or isolate the slow 2nd-order algos into their own long-budget chunk.
  **It is COMPILATION, not code-generation, that is slow** — Python codegen emits the full `grid.cuh` in
  ~20 s (fast `ccode` string-printing); the 20–40 min is one `cicc` process (nvcc's NVVM/LLVM optimizer)
  at 99.9 %CPU, SINGLE-THREADED, on the enormous single-block fully-unrolled kernel (cicc's reg-alloc/
  sched/LICM scale superlinearly with function size; ptxas is a smaller tail). So: a "stuck" big-robot
  build = check `cicc` age (a single 60+ min cicc is the pathological threshold; 30–40 min is normal).
  For DEV iteration on big robots, trade compile-time via the sweep's `--split-compile` /
  `--ofast-compile {min,mid,max}` / `--ptxas-opt-level` knobs; a MEASURING sweep wants full opt.
- **Header cache key does NOT hash codegen source** (only schema/robot/nq/nv/urdf) → a comment-only or
  internal codegen change is cache-invisible. `rm -rf .pytest_cache/grid_cuda` to force a fresh emit when
  you NEED to test new codegen; conversely, a proven comment-only change reuses the cache validly (the
  compiled binary is identical) — no wipe needed.

---

## 8. Process meta-lessons

- **The backlog drifts — verify "is this still open?" before dispatching.** This session found 5
  "open" items already done (C1-float, C3, mxS-ndim warning, mimic regressor-Y, FD param-grad). A
  cheap grep/check beats an agent redoing landed work.
- **A sharp deferral beats a wrong value path.** When a fix can't be made bit-exact, REVERT and
  document a precise resume hint (which tensor/cell/stride, what was ruled out) rather than ship
  silently-wrong numbers. Several "deferred" items came back and landed fast because the prior
  agent left a cell-level diagnosis.
- **Scope-discipline for big features:** land the smallest fully-validated slice FIRST (E2: frame
  Jacobian J solid before J̇/Λ; partial-but-green beats broad-but-unvalidated).
- **Docs + READMEs are part of "done" — propagate EVERY change to keep the project UNIFIED.** A code
  change (rename, new feature, convention shift, API tweak) that doesn't also update the user-facing
  docs + ALL relevant READMEs (top-level AND every submodule: RBDReference / URDFParser /
  GRiDCodeGenerator / python) + examples/notebooks leaves the project inconsistent. The verbose rename
  touched code but left `docs/source/**`, the `RBDReference/README.md` submodule README, and a `rnea.rst`
  page stale (audit A3). For any change: in the SAME pass update its doc page, the relevant main+submodule
  README(s), examples/notebooks, and do the cleanup (names/tokens/dead code/comments). After a rename/
  convention change, grep docs/READMEs/submodule-READMEs for the OLD names/values too — code-only greps
  miss them. Keep it consistent + clean + COMPACT.
- **Internal renames: prove safety with a before/after HEADER BYTE-DIFF.** Regen the same robots
  pre- and post-rename and diff the emitted `grid.cuh`. If every delta is a `//`-comment line, the
  emitted CUDA is functionally IDENTICAL (compiled-identical) → no equivalence re-run needed for those
  cells. Classify each token: **Tier-1** pure-Python identifiers (byte-identical output) / **Tier-2**
  abbreviations inside emitted COMMENT strings (`gen_add_func_doc`) (equiv-safe, comment-only diff) /
  **Tier-3** emitted SYMBOLS — struct fields/buffers (`d_eePos`, `d_did_du_dfext`), printf labels —
  which are the C-ABI surface referenced by bindings (`wrapper_template.cu`/`_handle.py`) + `printGRiD.cu`
  + example notebooks; renaming those is high-blast-radius and MUST be a single lockstep change across
  emit+bindings+printGRiD+examples with full re-validation (NEVER piecemeal). Before a blind substring
  replace, SENTINEL-PROTECT the Tier-3 emitted symbols so the rename can't corrupt the ABI.
- **`pkill -f '<pattern>'` can self-terminate the job** if `<pattern>` appears in your OWN command line
  (the script that runs the pkill). It silently kills the wrapper before the real work runs (empty log,
  exit 1). Kill by explicit PID (from `ps -C python`/`ps -o pid`), or use `TaskStop` for harness tasks —
  never a pattern that matches yourself.
- **Parallel doc/rename agents race on code STATE.** A docs agent that reads the codegen attr names
  *before* a sibling rename-agent commits will "correctly" leave a doc referencing the OLD names — which
  the rename then makes stale. After any parallel batch where one agent renames symbols another agent
  documents, the orchestrator must reconcile the docs against the post-rename state.
- **An agent that dies before committing still leaves its work in the shared tree.** Verify the diff and
  RE-RUN its validation yourself (don't trust its claimed numbers), then commit sole-committer,
  path-scoped. Background agents/jobs survive `/compact`; but a SILENT hang (e.g. wedged xdist, §7) never
  notifies — put a watchdog on every long job (poll for a done-marker / process-death / a timeout, and
  re-snapshot).

---

*Linked from HANDOFF.md. Companion: `docs/idsva_so_inner_refactor_notes.md` (SO internals + resume hints).*
