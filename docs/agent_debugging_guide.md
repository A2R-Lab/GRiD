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

---

## 5. Merge discipline (multi-agent, file-isolated clones)

- Agents clone off varying bases → expect **3-way merges**. File-isolated agents merge clean; the
  ONE collision zone is `GRiDCodeGenerator.py`'s **`_MIMIC_GRADIENT_ALGORITHMS`** set (every mimic
  ungate touches it). Hand-reconcile to the **UNION** of removals.
- **`set()` not `{}`** for an empty refusal set — `{}` is a dict and `dict |= set` raises.
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

---

## 7. Test-infra gotchas

- **pytest-xdist needs deterministic collection.** Per-process randomness (a random default thread
  count) → "different tests collected between workers." Make defaults deterministic.
- The CUDA equivalence harness has graceful-skip idioms: `GRID_SKIP_IF_KERNEL_TOO_BIG` (smem cap)
  prints a parseable `SKIPPED` line the parser nulls. The timing parser overwrites with the LAST
  numeric `Single Call X` line and ignores non-numeric ones.
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

---

*Linked from HANDOFF.md. Companion: `docs/idsva_so_inner_refactor_notes.md` (SO internals + resume hints).*
