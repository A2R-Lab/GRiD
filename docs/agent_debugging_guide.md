# GRiD agent guide — debugging, patterns, pitfalls (what works, what doesn't)

Hard-won institutional knowledge from the multi-agent campaigns on `modernizing-tests`
(F/G/H/I/J/K + the mimic-completion + perf/SO-audit rounds). **Read this before debugging a
GRiD codegen/CUDA issue or doing a refactor.** GRiD = Python codegen (`GRiDCodeGenerator`)
emitting CUDA C++ from URDFs; numpy/pinocchio reference oracle lives in `RBDReference`.

> **APPEND-ONLY: never renumber or retitle existing §IDs.** A dozen source files cite
> sections by ID (grep `§1j` / `agent_debugging_guide` to see); three legacy tail sections
> are all literally titled "7.x" — leave them, and give NEW tail sections fresh unique IDs
> (`7.z4`, `7.z5`, …). New bug classes go at the END of their numbered family.

## Symptom → section index (start here)

| Symptom you're staring at | Go to |
|---|---|
| Mimic robot: buffer overflow / garbage past `nv` entries | §1a |
| One bug appearing across MANY algorithms at once | §1b (shared linalg helper), §1j (beta=0 GEMM), §1p (stale GLASS pin) |
| All-zeros output that "ran fine" | §1c (silent launch fail), §1l (missing barrier before epilogue) |
| Floating base: wrong dynamics, no error, off-by-one-slot flavor | §1e (nq vs nv input stride) |
| mjx/`_mujoco` output "wrong" vs pin | §1k (it's a KNOWN frame transform — check the pin baseline first) |
| Kernel won't launch >48KB smem / `cudaErrorInvalidValue` on a variant | §1f (unregistered variant), §1z (probe checked the wrong kernel) |
| Results differ run-to-run at last ULP (floating base) | §1q (atomicAdd fold order) |
| "undefined `*_inner`" at nvcc time on a subset/high-DOF build | §1m (dispatcher vs emission gate), §1aa (helper gated behind unrequested algo) |
| NaNs from a direct `*_inner` call in YOUR kernel | §1n (usually caller wiring, not codegen) |
| "out of memory" on a nearly-empty GPU (big robot) | §1y (int overflow in size arithmetic), §1v (stack `const T[]`) |
| Kernel got SLOWER after adding per-thread arrays | §1u (register-array spill — block-share it) |
| Spill tier still over budget after "reducing" | §1t (reduction must apply inside the max()) |
| Sanitizer (racecheck/initcheck) findings | §1r (two known non-bugs) FIRST |
| Fixed-target EE chain silently wrong/dead | §1s (fixed jid has no link) |
| Spherical fixed-base buffer too small | §1ab (`n+fb` is not a position-space width) |
| Equivalence test flakes / stale results after codegen edit | §0 (cache discipline), §7 (test-infra gotchas) |
| FFI/jax/torch "kernel launch failed" | §7.z3 (THREE distinct causes) |
| Wrapper builds broken only on fixed-base or only floating | §1i (non-uniform signatures), §1x (backend delegation kwargs) |
| A `.so` behaves like OLD code after an edit | §7.z (bindings cache poisoning), and rebuild `_core` via `make build` |
| Driver wedged, every GPU run hangs | §7.x "SIGKILLing a running big-SO exe…" (never SIGKILL GPU exes) |
| Second derivatives wrong only at the floating root | §7.z2 (dropped dN(0) chart-slope term) |

---

## 0. The validation checklist (do these EVERY time — they each caught a real bug)
1. **Clean the generated-header cache before re-validating.** A stale `grid.cuh` gives phantom
   pass/fail. The CUDA equivalence harness keys its cache on a hash of the whole
   `grid_codegen/*.py` tree (`_header_cache_key`), so codegen edits self-invalidate — but
   manual/ad-hoc `gen_all_code` runs into temp dirs do not. When in doubt, clear it.
2. **Gate-A byte-identical** for any refactor or opt-in algorithm: capture the generated `grid.cuh`
   for representative robots (iiwa14-fixed + a floating + a big robot) BEFORE your change, regen
   AFTER, `diff`. Must be empty (refactor) or confined to your new opt-in kernel (additive).
3. **Floating + fixed codegen smoke, not just `py_compile`.** A floating-codegen regression (the
   single-axis-S guard hitting the 6-DoF floating root) slipped past `py_compile` — it only
   surfaced when actually generating a floating header. Use
   `gen_all_code(algorithm_list=[...])` per robot/base.
4. **AST dup-key check `external/RBDReference/tests/tolerances.py`** after any RBDReference merge — multiple
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
  the `*_DYNAMIC_SHARED_MEM_BYTES` input-slot terms in `_constants_arena.py` (post monolith-split).
- **Same family in DEBUG_MODE printf loops (2026-06-10):** a `for ind in range(n=NUM_VEL)` debug loop that
  calls `get_*_by_id(ind)` / indexes per-jid structures (`running_sum_*_per_jid[ind]`) crashes on floating
  (nv-index isn't a body id → `get_bfs_level_by_id` returns None) — invisible on fixed base. Iterate
  `range(NUM_JOINTS)` and map vel-col→body-id the way the non-debug emit does. Debug-only, but `debug_mode=True`
  codegen is how you dump kernel scratch, so it must work on floating too.

### 1f. New compile-time kernel VARIANT must be registered for `cudaFuncSetAttribute` (mjx >48KB launch fail)
**Found adding the `MUJOCO_OUTPUT=true` kernel variants (G-cross, 2026-06-09).** Adding a new compile-time
template instantiation (here `kernel<T, TIER, MUJOCO_OUTPUT=true>`) creates a DISTINCT `__global__` function
with its OWN attributes. `KERNEL_ATTR_MANIFEST`/`init_grid_kernel_attrs` registered
`cudaFuncSetAttribute(MaxDynamicSharedMemorySize)` ONLY for the pin (`false`) instantiation. The mjx twins
whose dynamic smem exceeds the 48 KB device default (fdsva_so, idsva_so-world, integrator*, id-grad on
humanoids) launched with `cudaErrorInvalidValue` ("invalid argument") while their pin twins succeeded.
- **Symptom:** `GPUassert: invalid argument` at the kernel-launch line for the mjx variant only; pin works.
- **Fix:** register the new instantiation too, up to the DEVICE max (`grid_get_max_dynamic_shared_memory_bytes`,
  ~96 KB on sm_120 — NOT a hardcoded 48 KB). >48 KB is fine once registered.
- **TRAP:** gate the registration on the ACTUAL emission condition. The mjx kernels are emitted for any
  `self.robot.floating_base` robot (the template param is added there); they are NOT gated on the
  `MUJOCO_OUTPUT` constructor arg, which the per-robot `.so` build (`_compile.py`) never sets. Gating the
  registration on `self.MUJOCO_OUTPUT` silently emitted nothing → no-op fix. Gate on `self.robot.floating_base`.
- **2nd instance — a non-type-template (IntegratorType) value, not just MJX (TRAPEZOIDAL, 2026-06-18).**
  Ungating the floating TRAPEZOIDAL integrator gradient made `integrator_gradient_kernel<T,TRAPEZOIDAL,...>`
  a NEW distinct `__global__`. The `KERNEL_ATTR_MANIFEST` integrator entries enumerated ITs as
  `EULER/SI/MIDPOINT/RK3/RK4` — TRAPEZOIDAL omitted (it had been fixed-base-only / floating-refused, so
  address-taking it tripped the old static_assert). go2-floating crashed `GPUassert: invalid argument` at
  the gradient launch at TIER_SHARED but PASSED at TIER_LITE — the spilled tier's dynamic smem is ≤48 KB so
  it needs no opt-in, which is exactly what makes this class hide on small/spilled cases. Same fix: add the IT
  to the 3 non-mjx manifest tuples (leave mjx euler/si-only). **GUARD ADDED:**
  `test/test_kernel_attr_manifest_consistency.py` asserts every non-mjx integrator family registers exactly
  `_INTEGRATOR_TYPES` — pure-Python, catches any future emitted-but-unregistered IT before a GPU run.
- **General rule:** ANY new fully-instantiated kernel (new template TYPE, new non-type VALUE like an IT, new
  flag) is a new `__global__` needing its own `cudaFuncSetAttribute`. Tier-spill HIDES the omission (spilled
  ≤48 KB launches fine), so test the UNSPILLED tier on a robot whose arena exceeds 48 KB, and prefer a static
  manifest-parity test over relying on a GPU run to surface it.

### 1g. In-kernel mjx OUTPUT-BAND scratch must be spill-aware (aliases spilled buffers → state-dependent garbage)
**Found in fdsva_so mjx (2026-06-09, still open).** An mjx epilogue that writes a large output band into
"dead" scratch (`s_temp`) is WRONG on robots where the algorithm SPILLS: `s_temp`==`d_workspace`, and the
spilled live buffers (`s_df_du`, `s_Minv`) ALSO live in `d_workspace` → the band overlaps them →
state-dependent garbage (the result depends on the PREVIOUS kernel call's `d_workspace` contents; jax≠torch
in an interleaved test but deterministic in isolation — the tell-tale signature).
- **Diagnostic:** run the op 3× in isolation (deterministic? → not uninitialized-per-launch) AND interleaved
  after a different op (changes? → reads cross-call global state = a spilled-buffer alias).
- **Rule:** an in-kernel scratch band must be provably DEAD *and* DISJOINT in BOTH the smem and the spilled
  layouts. A buffer that's disjoint in smem can alias in `d_workspace`. (Reusing `s_idsva_so` made fdsva_so
  deterministic but still wrong — `s_df_du`/`s_Minv` are likely clobbered before the epilogue reads them, a
  second liveness bug. Open.) This is the general **mjx-vs-tier-spill** hazard: the spill classifier never saw
  the mjx epilogue's scratch usage.

### 1h. Templating a wrapper on a flag that only ONE kernel overload carries
**Found in torch_inverse_dynamics/_gradient (2026-06-09).** A kernel with an optional-input overload set
(qdd present vs absent) may carry the new template flag (MUJOCO_OUTPUT) on only the qdd-input overload (the
bias/no-qdd overload is `<T, TIER>`, no MUJOCO). Templating the wrapper `<bool MUJOCO>` and passing
`<T, TIER, MUJOCO>` to BOTH branches fails to compile the no-qdd branch ("no instance matches"). Route the
mjx path through the flag-carrying overload — for ID-grad, the bias path zeros `d_qdd` and uses the qdd
overload (matches the jax handler, which always passes qdd) under `if constexpr(MUJOCO)`.

**Sibling (2026-06-10): wrapper gated on the wrong capability macro → mimic/skew robots fail to build.**
The binding's mjx (`*_mujoco`) C-ABI handlers instantiate `grid::*<...,MUJOCO_OUTPUT=true>`, but codegen EMITS
those template overloads only for `floating && !mimic && !skew` (the `mjx_inner`/`mjx_device` gates). They were
`#ifdef GRID_FLOATING_BASE` — defined for ANY floating robot — so a floating+mimic robot (h1_2, 12 mimic joints)
compiled the wrapper against overloads codegen never emitted → `grid::fdsva_so<T,GRID_DATA_ALL,true>` "no matching
function." Fix: emit a DEDICATED capability macro whose condition mirrors the codegen emission EXACTLY
(`GRID_RBD_WITH_MUJOCO`, defined iff `floating && !mimic && !skew`) and gate the wrapper + pybind on it; the
pybind side already used `opt_sym` (nullptr-tolerant) so only the wrapper's compile-time instantiation was the
hard failure. RULE: a wrapper that references a conditionally-emitted kernel variant must gate on a macro that
tracks the EMISSION condition, not a looser proxy (floating ⊋ mjx-capable).

### 1i. Non-uniform kernel SIGNATURES break the whole fixed-base binding build (and torch's optional-dep masks it in CI)
**Found during P-tier1 (2026-06-10).** The §1f-1h fixes had codegen DROP the `bool MUJOCO_OUTPUT` template
param from kernels for non-mjx robots (`if mjx_kernel: <T,TIER,MUJOCO> else: <T,TIER>`), so fixed/mimic/skew
robots emitted a 2-param `*_kernel`, while the wrapper unconditionally launches `*_kernel<T,TIER,MUJOCO>`
(3 args). Result: EVERY fixed-base / floating+mimic binding build fails to compile (`inverse_dynamics_kernel`,
then `momentum_cost_kernel`, then `plant_step_kernel`, … — a CHAIN, since nvcc stops after a few errors). It
went unnoticed because the binding build only compiles the torch op block when torch is installed, and CI
**skips torch** (optional dep) — so `test_iiwa14_torch_smoke` / even `_jax_smoke` for a FIXED robot never
exercised this path. Detection: build the bindings for a FIXED-base robot in a torch-installed env.
**Fix = UNIFORMITY, not a per-call workaround:** make the codegen emit the SAME kernel template signature for
every robot class (`template <typename T, int RESOURCE_TIER = ..., bool MUJOCO_OUTPUT = false>` always — the
plant cost/step kernels likewise; integrator kernels carry `IntegratorType IT` too). Keep the mjx BODY
(`if constexpr(MUJOCO_OUTPUT){...}`) gated on `mjx_kernel` so non-mjx robots emit NO mjx body (it would
reference floating-only constructs) — only the SIGNATURE is unconditional. The `false` instantiation is
byte-identical PTX (unused defaulted param, mjx body absent), so the pinocchio path is preserved and floating
non-mimic codegen is unchanged (it already took the 3-param branch). RULE: a kernel the wrapper calls with N
template args must emit N params on ALL robot classes — prefer making the DEFINITION uniform over branching
every call site. (Validated bit-exact on iiwa14-fixed + go2-floating + fr3-mimic, jax+torch.)

### 1j. `beta=0` GEMM still READS C → uninitialized-scratch `0*NaN` poisoning (load-dependent thread-inv flake)
**Found in spherical CRBA (2026-06-11).** A composite-inertia fold emitted `grid_linalg_gemm<...,false,true>(.., &s_temp[off], 1, 0, ..)` — alpha=1, **beta=0**, into a scratch slot. GLASS's with-beta gemm kernel computes `C[i] = alpha*res + beta*C[i]`, i.e. it **reads C even when beta=0**. On the slot's COLD first use that scratch is uninitialized; whenever the leftover bit-pattern happened to be NaN/Inf, `0*NaN == NaN` poisoned the whole fold (and M, minv, fd downstream). It presented as a *thread-invariance flake on `mixed_spherical_arm` under heavy concurrent build load*: at `threads=1` the work serializes and the slot is effectively always overwritten cleanly; at `threads>1` it intermittently surfaced (slot contents are nondeterministic across launches). Equivalence-vs-oracle (single isolated run) almost always passed — so it hid as "1×/15 under load." Fix: **zero the gemm temp slot once before the first beta=0 write** (a tiny `parallel_loop` + `sync`). RULES: (1) `beta=0` is NOT "write-only" in GLASS — a destination that a beta gemm writes must be initialized (or use a beta-less / overwrite kernel variant). (2) A *thread-count-dependent* discrepancy that vanishes when isolated is almost always an **uninitialized/under-initialized shared-scratch read** (or a missing sync), not a hardware blip — hunt the cold scratch slot. (3) Reproduce flakes by running the thread-inv check 20–30× **under concurrent GPU load**, not isolated. (Gate-A byte-identical for cardinal robots — the fix is in the spherical-only emit path.)

### 1k. pin↔mjx is a KNOWN frame transform — don't "debug" the reframe; check the pin baseline FIRST
**Cost a long session 2026-06-19.** GRiD exposes BOTH pinocchio (default) and mujoco/mjx output conventions
(intentional, user directive — keep both; flag = handle `output_convention` / the per-call `handle.mujoco`
view / C-ABI `*_mujoco` twins). They differ by a **documented, validated** base-frame transform, NOT a bug:
pinocchio = quat **xyzw** + free-joint velocity `[v_lin LOCAL; ω LOCAL]`; mujoco = quat **wxyz** + `qvel
[v_lin GLOBAL; ω LOCAL]`. Transform `G(q)=blockdiag(R, I_3)` on the leading 6 tangent DOF (R = base rotation);
G orthogonal ⇒ `G^{-1}=G^T`. Gradient base-linear maps `grad_mjx = R·grad_pin`; GN-hessian by congruence
`G X Gᵀ`; velocity inputs `v_pin = Rᵀ v_mjx`. **SSOT: `RBDReference/equivalents/mujoco_convention.py`** +
`docs/open-tasks/archive/mjx_output_convention_flag.md`. CUDA emit: `_code_generation_helpers.py`
(`gen_mjx_base_rotate`/`gen_mjx_congruence`/`_gen_mjx_build_R_lines`) + `_plant.py` (`_gen_cost_mjx_kernel_input`).
- **THE TRAP:** `test_mujoco_kernel` (and friends) are a CONSISTENCY check `mjx_kernel(q) == G·pin_kernel(q_pin)`.
  When one fails, the reflex "the reframe is broken / centroidal base columns are stale" is almost always WRONG.
- **DO THIS FIRST:** dump the mjx kernel's INTERNAL pre-reframe value (a one-line `printf` in `gen_mjx_base_rotate`
  before the `R*b` write) and compare to **RBDReference** (pinocchio). In the com_cost case the pre-reframe
  base-linear gradient `b=[-0.3356,-0.1297,0.0157]` bit-matched RBDReference exactly, and `R·b=[0.2405,…]` was the
  correct mjx value — i.e. **the reframe was perfect**. If `b` matches the pin oracle, the reframe is fine and the
  discrepancy is in the PIN BASELINE the oracle reframes (or a stale/version-rotated build), not the mjx path.
- **CHECK THE CORE IS VALIDATED:** the CUDA `com_cost` gradient (incl. floating base block) is already verified vs
  RBDReference by `test_cuda_plant_centroidal_costs_match_reference` (go2:floating, via the standalone `.cu` harness).
  A passing plant-equiv ⇒ the centroidal/cost CORE is correct ⇒ a binding-layer mjx test failure is in the bindings
  wiring or build staleness, not the kernel math. (Don't re-derive a "kernel bug" the .cu harness already disproved.)
- **STALE-BUILD GOTCHA (updated 2026-06-21):** the grid_rbd compile cache key hashes URDF + options + arch +
  package version + `_wrapper_template_hash()` + `_codegen_source_hash()` (the latter hashes all `*.py` under
  `grid_codegen/` + `URDFParser/`, AND now `bindings/grid_rbd/_compile.py`). So a codegen-SOURCE edit DOES
  rotate the key (no `force_rebuild` needed). The historical trap was narrower: editing the codegen INVOCATION in
  `_compile.py` (algorithm_list / `enable_*` flags) was NOT hashed → a flag change silently reused an old .so. That
  gap is now closed (`_compile.py` hashed). Still verify with `stat` if suspicious. NOTE: re-keying invalidates ALL
  cached robots → every robot's next build is a fresh (slow) compile; expected, not a failure.

### 1l. In-device mjx epilogue reading parallel-written scratch needs a barrier FIRST (race → zeros, printf masks it)
**Found in integrator_gradient mjx (2026-06-19).** The `_emit_integrator_gradient_mjx_output` epilogue runs
IN the device function right after `gen_integrator_gradient_dAB_assembly`, whose parallel loop writes `s_dAB`
with **NO trailing `__syncthreads()`**. The epilogue's Phase 1 reads `s_dAB` across all threads — so the
**high-column entries (the du-block), written by high-index threads, are read before they land → zeros** in
exactly the bottom-half base rows. The pin path is safe because the kernel-level output copy supplies a
barrier; the in-device mjx epilogue had none. Symptom: `integrator_gradient mjx != oracle max|d|=0.5`, with
CUDA **pin** dAB matching RBDReference to 1.5e-5 (so the algorithm is correct — bug is in the mjx epilogue).
**HEISENBUG TELL:** adding a `printf` reading `s_dAB` at the epilogue top made the test PASS (the read/serialize
perturbed scheduling enough to hide the race). RULES: (1) any in-device epilogue (mjx or otherwise) that READS
a buffer a prior **parallel loop WROTE must `__syncthreads()` first** — don't assume the writer synced.
(2) "a `printf` makes the failure disappear" ≈ race / missing sync / uninitialized read — never ship the
printf; find the barrier. (3) localize value bugs by comparing the **pin** path to RBDReference first: if pin
matches, the bug is in the mjx transform/epilogue, not the algorithm. Fix: `gen_add_sync()` at the epilogue
start. (Pairs with §1j's "thread-dependent discrepancy = scratch race".)

### 1m. Codegen-time DISPATCHER predicate must match the EMISSION gate (else "undefined inner" at compile)
**Cost a build cycle 2026-06-21; broke ALL high-DOF fixed-base robots (g1/h1_2/h2_plus fixed) on pushed
modernizing-tests.** `idsva_so` picks body- vs world-frame at codegen time via `_idsva_so_use_world_frame(self)`
(= floating OR spherical OR NV≥`NV_FIXED_WORLD_THRESHOLD`-fixed; `_idsva_so.py:20`). The dispatcher + `idsva_so_device`
EMIT a call to `idsva_so_world_frame_inner` whenever that predicate is true — but the world-frame *emission*
(`gen_idsva_so_world_frame()`, which DEFINES the inner) was gated on `floating_base` only, in TWO places
(`GRiDCodeGenerator.py` default + the `_compile.py` binding kwarg). For a high-DOF FIXED robot the predicate routes
to world, the inner is CALLED, but never DEFINED → `error: identifier "idsva_so_world_frame_inner" is undefined`.
- **THE RULE:** any "pick variant X at codegen time" predicate used by a dispatcher/device wrapper MUST be the SAME
  predicate that gates EMISSION of X. Don't write the selection logic twice. Fix here: emission default now reuses
  `_idsva_so_use_world_frame` (`b4719cc`); binding defers to it via `enable_*=None` (`b75eefa`).
- **WHY THE GATE MISSED IT:** the pre-push gate built only low-DOF (iiwa14/go2, NV<threshold) + floating, where the
  predicate and the floating-only gate happen to agree. **Always compile at least ONE high-DOF FIXED robot
  (g1-fixed) when touching idsva_so/fdsva_so frame selection** — that's where dispatch and emission diverge.
- Relevant to the perf-cleanup idsva_so agents (11a/11b): they rework exactly this body/world emission.
- **2026-09-08 recurrence, WRAPPER flavor (the "h1_2 idsva_so unlaunchable" myth):** the jax FFI + torch
  handlers for `idsva_so` picked body- vs world-frame via `#ifdef GRID_RBD_WITH_MUJOCO` — the mjx-TWINS
  gate (floating AND non-mimic AND non-skew), NOT the dispatch predicate. A floating MIMIC robot (h1_2)
  has no `WITH_MUJOCO`, so the handlers launched the 3.0MB body-frame no-ladder diagnostic → launch fail
  at every thread count → the 09-05 autotune verdict "genuinely unlaunchable" (the DISPATCHED world frame
  needs 15KB and runs fine — proven by the numpy path, which goes through `grid::idsva_so` itself). The
  same wrong gate silently ran body-frame on high-DOF FIXED robots (g1-fixed: 98KB body vs the intended
  33KB world). Fix: both handlers now fork on `GRID_IDSVA_SO_DISPATCHES_WORLD_FRAME` (the macro codegen
  emits FROM the dispatch predicate), with the MUJOCO_OUTPUT template param forked separately on
  `GRID_RBD_SIG_MJX_IDSVA_SO` (floating-mimic world kernels carry the param; high-DOF-fixed ones don't).
  **Tell:** numpy path works, FFI/torch path "launch failed", error names a kernel the dispatcher
  wouldn't pick. **Rule addendum:** `WITH_MUJOCO` gates TWIN existence, never frame/variant selection.

### 1n. Consumer NaN from direct `*_inner` calls is USUALLY caller wiring, not codegen (triage before "fixing")
**A consumer (GATO) filed "iiwa14 7-DoF `forward_dynamics` → NaN for every input, indy7 6-DoF fine, DoF-specific,
initcheck→finite, racecheck→NaN+0 hazards" (2026-06-21).** Signature screams "real uninitialized-shared codegen bug."
It was NOT. The `*_device` wrappers (`forward_dynamics_device`, `minv_device`, …) do two things for you that the
`*_inner` functions deliberately push to the caller; consumers who call the `_inner` directly must replicate BOTH:
1. **Load XImats first.** `*_inner` READS `s_XImats` but never writes it. The wrapper calls
   `load_update_XImats_helpers(s_XImats, s_q, s_topology_helpers, d_robotModel, s_temp)` + `__syncthreads()` before
   the inner. Skip it → inner reads uninitialized shared → NaN, race-clean, initcheck-fixable.
2. **Size `s_temp` to `FD_INNER_SMEM_BYTES<T, MINV_F_IN_SMEM>()` (resp. `MINV_INNER_SMEM_BYTES`).** At
   `MINV_F_IN_SMEM=true` the `6*NV*NV` Minv-F band lives in the **TAIL of `s_temp`**; `d_workspace`/workspace-bytes is
   0 but that does NOT mean the band is free — it moved into `s_temp`. A caller who sizes `s_temp` short (e.g. reuses a
   smaller-DoF constant) makes the inner read its own never-written band → NaN. **DoF-specific because the band scales
   as `6*NV*NV`** (indy7 216 fits the slack; iiwa14 294 overflows). This is the "why only 7-DoF" tell.
- **THE RULE / triage order:** before touching codegen for a consumer-reported NaN, reproduce **correct GRiD usage** in
  a standalone harness — `*_device<T, TIER_SHARED>` (and the correctly-wired `*_inner`) on zero input. If that's finite
  (it was: float+double, 1+32 threads, sensible gravity qdd), the codegen is fine and the bug is the call site:
  unloaded/partial XImats, under-sized `s_temp` (missing the band), wrong `MINV_F_IN_SMEM`/`nullptr` pairing, or scratch
  overwritten mid-call. Harnesses kept at `/tmp/fdbug/{fd_repro,fd_repro2}.cu` (device path + inner-direct misuse modes).
- **Doc hardening (so the next consumer doesn't trip):** the `forward_dynamics_inner`/`minv_inner` emitted docstrings now
  carry an explicit "CALLER CONTRACT" block (`_forward_dynamics.py` func_notes, `_minv.py` func_notes); the recommended
  consumer path is always `*_device` (sizes + loads everything). `nq=NUM_JOINTS` vs `nv=NUM_VEL` arena confusion is the
  same family as §1a/§1e.

### 1o. Frame-index convention mismatch — anchoring a target/sphere to the WRONG frame (ANTICIPATED — W3 collision)
Not yet encountered, but flagged in the collision design (`docs/open-tasks/design_W3_collision_2026-07-07.md`)
as the #1 silent-wrong-answer risk, and it is the same *index-convention* family as §1a/§1e. A "target" (named EE
point, or a foam collision sphere) is a fixed offset off a link; its world position/gradient reads
`s_Xworld[16*anchor_jid]`. Different tools index links DIFFERENTLY: **foam's `sphere_to_joint` uses an
actuated-joint COUNT** (base=0, +1 per revolute/prismatic/continuous, fixed joints don't advance), while HJCD's
`utils.cuh` *also* carries a rival URDF-link-ORDINAL table (hand=9 vs 7). **GRiD must map each target's link to
ITS OWN frame/joint id (the `s_Xhom`/`s_Xworld` slot) via `URDFParser`, NOT copy either external table.** A
mismatch silently checks collision / places the target on the wrong link with NO error — positions look
plausible. Guard: assert the mapping against a base-0-monotone-down-chain property (port foam's
`test_foam_spheres.py` UR10e assertion) and cross-check one sphere's world position vs an independent numpy FK.

### 1p. Consumer "uninitialized-`s_vaf` read" is USUALLY a STALE GLASS pin (§1j), not a GRiD read-before-write — the NaN-poison harness settles it
**PDDP filed (2026-07-09):** `grid_plant::plant_step_gradient` → the emitted `[A|B]` B-block (the
`d(qd_next)/dx` rows) goes NaN whenever garbage/NaN is resident in shared memory (a diverged rollout);
poison-bisect showed zeroing **only** the `s_vaf` slice restores immunity → looks like a genuine
`s_vaf` read-before-write in the du-gradient chain. **It is not.** It is §1j (a `beta==0` GLASS gemm
that still READS its destination `C`, `0*NaN=NaN`) landing on an `s_vaf` slot — and it was **already
fixed upstream** by GLASS PR#19 (`beta_blend`, pinned `08b98a7`). A consumer only still hits it if its
vendored GLASS predates PR#19 (PDDP's checked-in header was GLASS `5caa6d0`).
- **THE TRIAGE (do this before touching any emitter):** generate the header at *current* GRiD HEAD and
  diff the whole chain function-by-function against the consumer's header
  (`plant_step_gradient` → `integrator_gradient_device` → `forward_dynamics_gradient_device` →
  `inverse_dynamics_gradient_inner` / `inverse_dynamics_inner` / `minv_inner`). If **every
  GRiD-generated function is byte-identical** and only the vendored GLASS block differs, the fix is a
  **GLASS pin bump + REGEN**, not a codegen change. (Verified: current HEAD, GLASS `08b98a7`, is
  poison-immune with no zeroing; PDDP's `5caa6d0` header reproduces 98 NaN, deterministic.)
- **THE TOOL — NaN-poison harness** (reusable; the arbiter for every "is this arena slot read before
  written?" question, since **initcheck is blind to shared memory** and **racecheck doesn't flag a
  never-written read**): (1) a `poison_kernel` fills the block's dynamic smem with `0xFF` bytes (`=NaN`
  for float AND double) over `8*numSMs` blocks + `cudaDeviceSynchronize`; (2) launch the target kernel
  with **finite** inputs; (3) assert the output is finite. A genuine read-before-write then fails
  **first-iteration, every run** (deterministic — no "1×/15 under load" flake). **MANDATORY positive
  control:** a twin kernel that carves the same arena and reads the suspect slice *without writing it*
  must come back NaN — otherwise an `ALL_FINITE` just means the poison didn't land (wrong smem size /
  scheduling), not that the path is clean. Reference harness + scrub-bisect (`zero=none|vaf|df_du|…`)
  in this session's scratch `poison/` + `poison_pddp/`; the poison recipe mirrors PDDP
  `docs/agent_debugging_guide.md` "Bug class 8".
- **RESOLUTION for consumers (PDDP et al.):** bump the GLASS pin to `≥08b98a7` (via a GRiD submodule
  bump), then **regenerate `grid.cuh`** — a *gitignored/per-robot* header does NOT auto-regen when you
  bump the submodule pointer (PDDP's did not, which is why they believed "08b98a7 still reproduces").
  Then drop any caller-side `s_vaf` zero-fill workaround. GRiD itself needs no change: the `beta==0`
  contract is GLASS's (`beta_blend`), and current HEAD already pins the fix.
- Same family as §1j (root), §1n (consumer-NaN triage-before-fixing), §1a (`s_vaf` sizing).

### 1q. Floating-base shared-parent `atomicAdd` folds are run-to-run NON-DETERMINISTIC (last-ULP) — replace with a parent-major fixed-order sum (also FASTER)

**Found 2026-07-09 (Inc6), fixed GCG `bc6c75a`.** Branched floating-base robots (quadruped legs, e.g.
go2-floating) fold each child link's spatial contribution onto the SHARED floating root (parent 0) via
`atomicAdd` into a shared-memory cell. `atomicAdd` sums in **warp-scheduling order**, which varies
launch-to-launch → the reduction's last **1–2 ULP** drift run-to-run on the SAME binary + SAME input.
It hit `crba` (composite-inertia IC fold), `minv` (IA fold), and `inverse_dynamics_gradient` (df/du
floating path); crba's jitter propagated into `forward_dynamics_gradient`. **Deterministic at 1 thread**
(serial), non-deterministic only at >1 thread — the tell for a scheduling-order reduction (vs a
compile-time FMA-contraction difference, which is stable run-to-run). NOT an init/poison defect: 0 NaN
under the §1p 0xFF sweep — the slots ARE written, only their *summation order* varies.
- **THE FIX (root cause, not a guard):** replace the slot-major `atomicAdd` with a **parent-major
  fixed-order sum** — iterate over the UNIQUE-parent cells (one thread owns each `(parent,row,col)`),
  sum the child slots in FIXED ascending slot order with a plain `+=` (single writer per cell → no
  atomics, no race). Wrap each per-BFS-level emission in its own `{}` scope so multi-branch humanoids
  don't redeclare the `s_upar_lvl` table. This is deterministic BY CONSTRUCTION and thread-count-invariant.
- **IT'S ALSO FASTER (measured, not assumed).** A/B on go2-floating (quiet GPU, cudaEvent, batch=2000×15,
  min-of-reps): crba **−8.0%** @288 / −3.7% @448, minv −3.6/−1.2%, id_grad −3.3/−1.5%, fd_grad
  −1.9/−1.2%; an untouched control kernel (aba) matched to 0.01% (noise floor). Removing shared-memory
  atomic contention (all leg threads racing into the SAME root cells) BOTH kills the nondeterminism AND
  cuts latency. **So convert these folds unconditionally — no opt-in perf flag.** Harness:
  `scratchpad/eqaudit/abtiming.cu`.
- **VERIFY: correctness, not the symptom.** The nondeterminism is INTERMITTENT (a warp-race needs
  scheduling contention — on a quiet box even the pre-fix binary is often bit-stable across dozens of
  trials), so don't try to reproduce the jitter as your gate. Instead verify the conversion is
  numerically correct: equivalence-vs-oracle still passes + `grid.cuh` byte-identical for robots the
  fold doesn't exercise (fixed-base) → identical-in-exact-arithmetic by construction (pure summation
  reorder). The new values land inside the old atomicAdd jitter band (old = oracle-passing).
- **DURABLE GATE (§0-style):** `test_cuda_executable_equivalence.py` now runs the equivalence runner
  TWICE at `num_threads==0` (MAX_PERF) on floating robots and asserts byte-identical stdout
  (`_first_differing_block` names the culprit). Non-vacuous because the runner prints float32 at
  `setprecision(10)` (2–3 digits past float precision). It's a PROBABILISTIC catch-net (can't force an
  intermittent race), zero-cost when deterministic.
- **TAIL (same class, not yet converted — do the same parent-major treatment when you touch them):**
  `aba` fixed-base fold (`_aba.py`, unexercised by the current matrix — go2-fixed's only repeated-parent
  BFS level is 0, skipped by the `bfs_level!=0` guard; fr3 mimic collapses; iiwa14 is a chain),
  `idsva_so` SO sibling folds (`_idsva_so.py:2780,3849`), `dccrba`/cmm (`_dccrba.py:345-350`),
  `coriolis` mimic (`_coriolis.py:448`), and the `inverse_dynamics_gradient` FIXED-base sparsity path
  (`_inverse_dynamics_gradient.py:1010-1012`, high-risk compressed indexing — only the floating path was
  converted). Same family as §1b (a shared-reduce fix has fleet-wide blast radius — prefer the narrowest
  per-fold change, Gate-A byte-identical on fixed-base).

### 1r. Two sanitizer findings that are NOT bugs — do not "fix" them (2026-07-11)

Both were investigated to the bottom during Inc4b (multi_target bench registration) and cost real time.
**Triage them with the tests below before touching any code** — the "fix" in both cases would perturb
oracle-validated codegen (and break header byte-identity) for zero correctness gain.

**(a) `racecheck` "Potential WAR hazard (Warp Level Programming)" in `end_effector_pose_kernel`.**
Reported as **3 hazards / 0 errors / 3 WARNINGS** (racecheck classifies genuine races as *errors*;
"Potential ..." is advisory). It is a FALSE POSITIVE on a **structurally-constant** shared slot:
`s_XmatsHom` is re-written each timestep from the FIXED `d_XImats` model data, and a homogeneous
transform's `[3][3]` element is **always exactly 1.0** (products of homogeneous transforms preserve the
bottom row `[0,0,0,1]`). So iteration *k*'s read and iteration *k+1*'s rewrite touch a slot whose value
never changes.
  * **The tell:** racecheck prints `Current Value : X, Incoming Value : X` — *identical* — on every
    flagged access. A WAR can only corrupt anything if the write changes what the reader observes; when
    incoming == current, no execution order can produce a different result. **If Current == Incoming on
    every access, stop — it is benign.**
  * Source check confirms it: every access pair is separated by `__syncthreads()` (`load_update_XmatsHom`
    ends with one; every serial-chain ping-pong level ends with one), and within a level the reads/writes
    hit **disjoint halves** of `s_temp` (`[0..15]` vs `[16..31]`).
  * Empirical confirmation (do this, it's cheap): ee_pose is **bit-identical across 8 runs** and agrees
    with the INDEPENDENT `multi_target_position` FK path to float32 eps — plus it already passes CUDA
    equivalence vs the numpy oracle across the robot matrix.

**(b) `initcheck` "Uninitialized __global__ memory read" in EVERY `*_kernel_single_timing`.**
This is the **anti-LICM feedback by design**, and it is FLEET-WIDE, not specific to any one algo.
`gen_anti_licm_input_reload` injects a loop-carried dependency by reading a PRIOR rep's output slot —
`d_<out>[(rep + 0x3FF) & 0x3FF]` — which on rep 0 has never been written (the buffer is `cudaMalloc`'d,
not zeroed). The value is **deliberately garbage-tolerant**: it only perturbs the input so ptxas cannot
hoist the rep loop. Timing-only path; it never reaches a correctness output.
  * **Control that proves it:** build a binary calling ONLY an existing single_timing algo (e.g.
    `end_effector_pose_single_timing`) and run initcheck — it reports the same **6 errors per kernel**.
    Any new algo will show 6×(number of its single_timing kernels).
  * Do NOT "fix" by zeroing the buffer or changing `gen_anti_licm_*` — that is a SHARED primitive
    (fleet-wide blast radius, §1b family) and zeroing would weaken the LICM barrier it exists to provide.

**⚠ But `memcheck` on the SAME path DID find a real bug — don't let (b) desensitize you.** The anti-LICM
feedback indexes up to slot **1023**, so `gen_anti_licm_*` silently REQUIRE the output buffer to have
**≥ 1024 elements**. Every legacy algo satisfies this by accident (`ee_pose` = `6*NUM_EES*256` = 1536),
but any algo whose output scales with a *user-supplied batch* can under-allocate: a 1-target
`multi_target` batch is only `3*1*256` = 768 < 1024 → **out-of-bounds read**, surfacing as
`cudaErrorLaunchFailure (719)` under memcheck. Fixed by flooring the DEVICE buffer at 1024 elements in
`gen_init_gridData` (the D2H copy still moves only the natural size). **If you add an algo whose output
size depends on a batch/config count, check this floor.**

---

### 1s. A FIXED-target jid has NO link — resolving its parent via `get_link_by_id` silently kills the whole chain-up (2026-07-11, GATO Ask-4)

**Symptom.** On a **BRANCHED** robot (go2), `end_effector_pose_inner_<target>` for a named fixed
kinematic target returned a pose that *looked* plausible — it matched the parent joint's world frame
exactly — instead of the target frame (go2 `FR_foot_joint`: the 0.213 m foot origin was missing).
On a **SERIAL** robot (iiwa14) the same feature worked perfectly, which is why it survived so long.

**Root cause.** Moving joints and fixed joints live in **separate tables** with **disjoint id spaces**
(go2: moving 0-11, fixed 12-40). The branched chain-up resolved each level's parent with
`get_link_by_id(jid).get_parent_id()` — but a *fixed-target* jid (32) has **no link**, so that returned
`None` → the `-1` root sentinel. The emitted code therefore read
`int parent_jid = (ind < 16) * -1;` at **every** level, and the guard `if(parent_jid == -1){continue;}`
skipped **every single compose**. The chain-up never ran. The serial path was immune because it
resolves the first hop through the fixed-joint table
(`get_fixed_joint_by_id(jid).get_parent()` → `get_joint_by_name(...).get_id()`).

**Why it looked right (the dangerous part).** With no level ever writing, the extract read a
**never-written half of `s_temp`** — an uninitialized shared-memory read (§1a/§1p family). It happened
to return the parent joint's world transform *only because a prior `end_effector_pose` call in the same
block had left one there*. Call it in isolation and you get garbage; call it after the generic EE fn
and you get a confidently wrong answer. **A plausible-looking value is not evidence the chain ran.**

**Fix.** Resolve a fixed-target jid through the fixed-joint table before falling back to the link table
(`_parent_or_root` in `_eepose_gradient_hessian.py`). Moving jids are unaffected
(`get_fixed_joint_by_id` → `None`), so generic/all-leaf emission stays **byte-identical** on every robot.

**Lessons.**
- **Two id spaces ⇒ two lookups.** Any helper that walks a topology by jid must say what it does when
  handed a *fixed* jid. Returning the root sentinel is the worst option: it fails **silently**.
- **A `continue` guard is a chain-up kill switch.** If a sentinel can be produced by a *lookup failure*
  rather than by genuinely reaching the root, the guard turns a bug into a no-op.
- **Test the feature on a BRANCHED robot.** Serial chains take a completely separate emission path here
  (`robot.is_serial_chain()`), so an iiwa-only test proves nothing about go2/h1.
- The gradient/hessian inners were **already correct** — they chain through the shared world-FK pass
  (`s_Xworld`, which bakes fixed targets), not this per-level parent walk. Verified vs pinocchio
  `getFrameJacobian(LOCAL_WORLD_ALIGNED)` at 2.8e-17. Don't assume a whole family shares a bug.

---

### 1t. A spill rung must apply its reduction INSIDE the `max()`, not subtract it from the total (2026-07-13)

**Symptom.** `fdsva_so_kernel` threw **"an illegal memory access"** on **go2-floating @ TIER_SHARED at
EVERY thread count** (32..1024) and every batch size (died on the first launch, N=16). memcheck:
`Invalid __shared__ write of size 4` at `0x190fc` = **102652 B — 252 B past the 100 KiB HW cap**.

**Blast radius (why this cost us a whole robot's autotune).** The batch binary runs every algo in ONE
process and `gpuErrchk` does `exit(code)`. So one kernel's death **killed the entire shared-tier
binary**, and *every* algo on go2-floating lost its shared-tier probes (`shared=0 / lite=240 /
minimal=192`). SHARED is the no-spill tier and usually the fastest ⇒ the autotune silently fell back to
lite/minimal and **GRiD under-reported its own performance on that robot**. The picks were not *wrong*,
they were *pessimistic* — a far quieter failure than a crash.

**Root cause.** The spill pool is a **`max` over three INDEPENDENT consumers**:

    fdsva_temp_full = max(idsva_inner, contraction 4·nv³, fd_grad_inline) + rt

The `idsva_cold` rung spills the idsva world inner's *cold quad* to global — which shrinks **only the
idsva term**. But the rung formula subtracted it from the **total**:

    base + fdsva_temp_full - cold_floats          # WRONG

On go2-floating the max is dominated by the **contraction**, not the idsva inner
(`idsva=3030, contraction=23328, fdg=10494`), so shrinking idsva `3030 → 1938` changes the max by
**nothing** — yet the formula still cut 1092 elements off the arena. Correct:

    base + max(idsva_inner - cold_floats, 4·nv³, fd_grad_inline) + rt     # RIGHT

**The kernel was correct all along; the ARENA FORMULA lied.** The kernel carved the full pool (25311
elems) while the launch reserved the fraudulent 24234 → **short by 1077 elems (4308 B)** → OOB write.

**Why the "does it fit" guard didn't catch it.** `grid_check_dynamic_shared_memory_bytes` compares the
*computed* arena against `GRID_CUDA_TARGET_SHARED_MEM_BYTES` (98304). The fraudulent 24234 (= 97396 B)
**passed** the check; the honest 25326 (= 101764 B) does **not** — so with the fix the picker correctly
rejects the rung and falls to `workspace_temp` (spilling the contraction to global). **An under-counted
arena doesn't just under-reserve — it defeats the fits-check that exists to prevent exactly this.**

**Lessons.**
- **A reduction that targets ONE term of a `max` must be applied to that term, inside the max.**
  Subtracting it from the total is only valid if that term *is* the max — which is a robot-dependent
  fact, so it is never safe to assume. This is a whole *class*: audit every rung whose formula does
  `pool - something`.
- **Invariant worth asserting in codegen:** for every algo/tier, `DYNAMIC_SHARED_MEM_BYTES` must be
  **≥ the sum of the regions the kernel actually carves**. Here they disagreed by 1077 elems and nothing
  caught it. (Cross-check: `inverse_dynamics` 1468 == 1468 and `idsva_so_world_frame` 4023 == 4023 —
  fdsva_so was the ONLY algo where macro ≠ carve, which is how it was localized.)
  **★ NOW ENFORCED — `test/test_shared_arena_covers_carve.py` (2026-07-13).** Purely static on the
  GENERATED header (no compile, no GPU, zero blast radius on codegen): `gen_declare_shared_arena` already
  emits every carve as a `// GRID shared arena layout` comment block, so the test sums each `__global__`
  kernel's regions per tier branch and asserts its launch-sizing macro covers them. **Positive control
  run:** re-introducing the bad rung makes it fail with
  `fdsva_so_kernel: tier 0 macro reports 24234 but the kernel CARVES 25311 (short by 1077)` — i.e. it
  names the algo, the tier, and the exact shortfall. Matrix: iiwa14-fixed, go2-floating, go2-fixed,
  fr3(mimic); 48 kernel/tier pairs checked. The assert is `>=`, not `==`: a spill ladder legitimately
  leaves slack (the arena is a max over rungs and the picked rung may not be the argmax). **Slack is
  waste; under-count is corruption.**
  Note the pre-existing `#ifdef GRID_CUDA_DEBUG_LAYOUT` assert in `gen_declare_shared_arena` does NOT
  cover this — it checks the carve against the SAME t_buffers list it was built from (self-consistent by
  construction) and never against the macro the HOST uses to size the launch. That was the whole gap.
- **Sanitizers find this instantly, tier sweeps don't.** The bug needs (floating base) × (TIER_SHARED) ×
  (contraction-dominated pool) — a corner no default run hits. It sat here from before the Inc3 arena
  fold (verified: pre-Inc3 `4381096` emits the identical wrong 24234).
- **A dead value must still be CORRECT.** The in-gen 9-tuple's arena column is unused since Step 3.4
  (the composer supplies arenas) — but it carried the same wrong formula. Leaving a stale-but-wrong
  number next to the right one is a trap; fix both or delete one.

### 1u. A large per-thread REGISTER ARRAY in a single-block kernel spills → huge SASS AND slow; block-share it (2026-07-25)

**The mjx second-order epilogues (idsva_so / fdsva_so) assembled each output slab "thread-per-k": every
thread owned some k-slabs and built them in PER-THREAD register arrays** — idsva `T work1[nv^2],
work2[nv^2]` (648 floats @ nv=18), fdsva SIX nv^2 arrays (`work1/work2/inner_u` + sensitivities
`d_dq/d_dqd/d_Mi` = 1944 floats). That far exceeds the register budget → ptxas **spills to local memory**,
and the spill/reload code is the DOMINANT SASS (it *looks* like "irreducible dense math" in a nulling
experiment, but it is spill traffic). It is also slow: only ~nv of the ~448 launched threads are active,
and every op hits local memory.

**Fix = BLOCK-PARALLEL: move the work matrices to block-shared scratch (here `d_mjx_scratch`, global,
tier-safe) and spread each nv^2 op across the whole block (block-strided loop + a `__syncthreads()`
between producer and consumer). Keep the cheap per-k SCALARS (R, the j-vectors, Rd, the 3 base-linear
sensitivities) per-thread register-local — every thread recomputes them deterministically, so per-element
results stay BIT-IDENTICAL.** Measured go2-floating: idsva mjx SASS 5.53x→2.42x + runtime 1.44x; fdsva
2.97x→1.41x + runtime 2.44x. Block-strided loops also carry a runtime bound (`blockDim`) so nvcc cannot
unroll them → they roll for free (no `#pragma unroll 1` needed).

- **Tell:** a nulling experiment shows a slab "costs" a lot of SASS, but the slab's actual FLOPs are
  O(nv) (only 3 rotated cols/rows) — the size is spill, not compute. Check the kernel's local-memory
  frame (`nvcc -Xptxas -v` → "stack frame" / "spill stores"); a big frame on a single-block kernel is the
  smell.
- **Gotcha:** each block-parallel op needs a `__syncthreads()` before any read of an element a DIFFERENT
  thread wrote (rot_rows/gd/reframe cross rows or cols). Sync-after-every-op is correct; racecheck
  validates. `+=` accumulate blocks that write a fixed 3×3 base sub-block must be parallelized over the
  column index `a∈[0,3)`, NOT left to all threads (all-threads += is a WAW corruption).
- **Validate fast:** golden-compare the new kernel's output to the OLD (committed) kernel bit-for-bit
  (per output tensor) — a refactor that preserves per-element math is bit-identical; no numpy oracle
  needed for the inner loop. Harness pattern in `scratchpad/glass_slabs/` (git-stash the one edited file
  to regen the baseline header, compile both, diff device output). Then run the real oracle + 4 sanitizers.
- Same "single-block perf comes from in-block parallelism" mental model as §4; the trap is that the
  register-array form looked already-parallel (over k) but under-utilized threads and spilled.

### 1v. A NON-static local `const T[]` array lands on the per-thread STACK → `cudaLaunchKernel` OOMs on a big robot (2026-07-26)

**A kernel that bakes robot topology as function-local `const T foo[] = {...}` (NOT `static`) puts the
array on the per-thread STACK.** For a small robot this is invisible; for a big one it is fatal. The
analytic `f_ext_gradient_dq` kernel baked its sub-job 6-vecs as `const T fegdq_Sj[]`/`fegdq_Sm[]`; on
H2 (nv=81, nsub=8022) each is 48 k floats = 192 KB, so the kernel's **stack frame hit 385 KB**
(`cuobjdump -res-usage` → `STACK:385088`). The driver reserves local memory = stack_frame ×
max_resident_threads across the WHOLE device (≈ threads/SM × #SMs), so 385 KB × ~270 k threads ≈ 100 GB →
`cudaLaunchKernel` returns **`cudaErrorMemoryAllocation` "out of memory"** (error 2). Small robots
launch; only the largest one fails.

- **Fix = `static const`.** A function-local `static const` array with constant initializers goes to
  constant/global memory (read-only, shared by all threads), NOT the stack. Stack frame drops to ~0.
  Values are unchanged → **bit-identical output** (a storage-class change only; prior equivalence still
  holds). Integer topology arrays (`fegdq_pre[]` etc.) were already `static const`; the float ones were
  the leak.
- **Tell:** `cudaErrorMemoryAllocation` on the *launch* (not a `cudaMalloc`) of ONLY the biggest robot,
  q-independent (baked arrays don't depend on inputs). Confirm with `cuobjdump -res-usage <exe>` — a
  large `STACK:` on a single-block kernel is the smell (same probe as §1u's register spill, different
  cause: DATA on the stack vs WORKING arrays spilling).
- **Gotcha:** the error surfaces at the *next* `gpuErrchkKernel`/sync after the failed launch, so the
  reported `grid.cuh` line points at the sync, not the array. Distinct from §1u (that was per-thread
  working arrays spilling to local; this is per-thread const DATA sitting on the stack). Sweep for the
  same pattern in sibling emitters (the value −Jᵀ path's `feg_job_S[]` had the same latent leak).

---

### 1w. A CUDA-graph "replay slower than eager" is usually the TIMING LOOP, and the wall clock is the wrong axis when GPU-bound (2026-08-01)

**The overnight torch cell showed graph replay at 0.71–0.94× vs eager — a graph "losing" to
eager.** Two stacked causes, neither a kernel bug:

- **Harness artifact (the headline number):** the timed loop called `GraphCallable.__call__`,
  which D→D-copies every input into `static_in` before `graph.replay()` — 3 extra copy launches
  (~2.5 us GPU + ~8 us CPU dispatch) per iteration that the eager loop never pays, on inputs that
  never changed. For repeated identical launches the apples-to-apples graph number is
  **`replay()` only** (update `static_in` in place outside the timed region, symmetric with eager
  reading the same tensors).
- **Genuine structural fact:** at B ∈ {64…1024} a GRiD fd kernel is 14–80 us of GPU work vs
  ~13 us of CPU submission — the loop is **GPU-execution-bound**, so collapsing launch overhead
  cannot move the wall clock (replay ≈ eager, 0.9–1.0×; even a 140-node 20-step captured rollout
  is 1.00–1.02×). The graph win is real but lives on the **CPU-submission axis**: enqueue drops
  ~13 us → ~1.9 us (≈7×), i.e. a freed python thread, not a faster GPU. A single-op capture is
  also only ~5 graph nodes (pack-memcpys + kernel + copy-out) — `cudaGraphLaunch` fixed cost eats
  most of what 5 pipelined async enqueues cost anyway.
- **Tells:** replay-vs-eager ratio ~1.0 that *degrades* when input copies are inside the loop;
  `torch.profiler` showing kernel self-CUDA time ≈ wall/iter (GPU-bound); submit-only timing
  (no sync inside the timer) collapsing under the graph while wall does not.
- **Measure both axes:** wall time (sync at end) AND submission time (sync outside the timer);
  report the copy-in variant separately. Always assert `torch.equal(eager, replay)` — bit-equal
  is the expectation, drift means a capture bug.

---

### 1x. Pin-only FLOATING bindings builds broke twice: a dropped kwarg in backend delegation + a signature switch keyed on the wrong macro (2026-08-01)

Found building the first `floating_base=True, enable_mujoco_kernels=False` robot through the
jax/torch BINDINGS (go2 gpu-resident examples). Two independent bugs:

- **Dropped kwarg in backend delegation:** top-level `grid_rbd.register_robot(backend="jax"/"torch")`
  forwards to `grid_rbd.{jax,torch}.register_robot(...)` with an EXPLICIT kwarg list — and
  `enable_mujoco_kernels` wasn't in it, so the flag silently reverted to True and the build got the
  full mjx twins (different cache key, double compile time; a humanoid would OOM). The numpy backend
  honored it. **Tell:** two cache entries whose grid.cuh differ by `#define GRID_RBD_WITH_MUJOCO`.
  When a backend wrapper mirrors a long kwarg list, every new register_robot option must be added in
  THREE places (top-level → backend fn signature → backend's base call) — grep all delegation sites.
- **Wrapper signature switch keyed on the mjx-KERNELS gate instead of the emitted SIGNATURE:**
  grid.cuh emits host launchers as `<..., KIND, MUJOCO_OUTPUT, RESOURCE_TIER>` on FLOATING robots
  (per-algo exceptions: fdsva_so/fd_gradient drop it on mimic/skew; id_gradient also on spherical) —
  INDEPENDENT of `enable_mujoco_kernels`. The wrapper's pin call sites chose the 3-vs-4-template-arg
  form via `#if defined(GRID_RBD_WITH_MUJOCO)` (= enable && floating && !mimic && !skew). On any
  floating build where those diverge (pin-only floating; floating mimic/skew), the 3-arg form binds
  the explicit TIER into the `bool MUJOCO_OUTPUT` slot: **TIER=2 → hard nvcc error** ("narrowing
  conversion of '2' to bool", seen on fdsva_so), **TIER=1 → silently compiles with
  MUJOCO_OUTPUT=true + default RESOURCE_TIER** (mjx-convention output from the pin entry point).
  **Fix (no rule duplication — §1m):** `_compile._mjx_signature_flags()` scans the JUST-GENERATED
  grid.cuh for each host fn's template line and passes `-DGRID_RBD_SIG_MJX_<FN>` iff it carries
  MUJOCO_OUTPUT; the wrapper's 13 signature switches key on those per-fn flags. Ground truth = the
  emitted header, so the switch can never drift from codegen's per-algo rules.
- **Still latent (out of scope 2026-08-01):** a floating SPHERICAL non-mimic robot with mjx enabled
  — `GRID_RBD_WITH_MUJOCO` is defined but id_gradient's mjx overload/signature is not emitted, so
  the wrapper's mjx ENTRY POINT for it should fail to compile. Same class, needs the entry-point
  gates audited against the per-algo `mjx_host` rules.

### 1y. Host-side alloc/memcpy SIZE arithmetic overflows int on big robots — "out of memory" on a nearly-empty card (2026-08-01)

Found smoking the chunked-workspace seam on h2_plus (nv=81) at N=1024: `cudaMalloc` for
`d_f_ext_gradient_dq` reported OOM with the card nearly empty. The emitted size was
`NUM_VEL*6*NUM_BODIES*NUM_VEL*NUM_TIMESTEPS*sizeof(T)` — every factor an `int`, and C++ evaluates
left-to-right, so the ELEMENT COUNT (81·6·76·81·1024 ≈ 3.06e9 > INT_MAX) wrapped NEGATIVE **before**
`sizeof(T)` entered and promoted it: the negative int converts to a ~1.8e19 `size_t` → instant OOM.
Same class in `d_idsva_so`/`d_df2` (`SECOND_ORDER_TENSOR_SIZE*NUM_TIMESTEPS` = 2.18e9) and the SO
wrappers' D2H `cudaMemcpy` sizes. Latent since those emissions existed — every prior h2_plus SO
"device OOM at init" partially had THIS cause, mislabeled as genuine footprint.

- **Fix pattern: `sizeof(T)` LEADS the product** (`sizeof(T)*A*B*N`) so the arithmetic is `size_t`
  from the first multiply. One-token reorder, no casts.
- **Tell:** OOM on an alloc whose hand-computed size fits comfortably; or element count within ~2×
  of 2^31. Grep audit: any emitted `malloc/cudaMalloc/cudaMemcpy` whose size expression ENDS with
  `*sizeof(T)` and can exceed 2^31 elements on nv≈80+ robots.
- **Related, currently safe by construction:** per-timestep grid-stride offsets inside kernels
  (`k*stride`) stay under INT_MAX because the chunked launches cap k<C (C=512·2.99e6 ≈ 1.53e9);
  an UNCHUNKED N=1024 launch on an nv=81 robot would overflow there too — if such a config ever
  becomes reachable, the kernel-side index arithmetic needs the same size_t promotion.

### 1z. A launchability PROBE must check the kernel the DISPATCHER actually launches, not a fixed variant (2026-08-02)

h2_plus `idsva_so` reported "SKIPPED (242,816 B shared mem exceeds device cap)" at EVERY tier —
looked like a hard smem wall. It was the bench's skip probe: `PER_ALGO_SPECS["idsva_so"]`
checked `IDSVA_SO_BODY_FRAME_DYNAMIC_SHARED_MEM_BYTES` (the floating no-ladder DIAGNOSTIC path),
but `grid::idsva_so` forwards AT CODEGEN TIME to the WORLD frame on floating/spherical/high-DOF
robots — which fits (84 KB shared / 23 KB lite+minimal). The cell was never actually blocked.
- **Tell:** "skipped for resource X" where the probed constant belongs to a DIFFERENT emitted
  variant than the dispatcher's forward target; the per-tier constants in the generated header
  (grep `*_DYNAMIC_SHARED_MEM_BYTES`) disagree with the skip message's number.
- **Fix pattern:** emit the dispatch decision as a header macro
  (`GRID_IDSVA_SO_DISPATCHES_WORLD_FRAME`, from the same predicate the dispatcher emit uses) and
  make the probe `#if` on it (undefined → old behavior, so stale headers keep working).
- **General rule:** any consumer-side gate keyed to "algorithm X" must resolve X the way the
  PUBLIC entry point does. Same class as §1x (wrapper switch keyed on the wrong macro).

### 1aa. Restricted `algorithm_list` emits a CALLER whose shared HELPER is gated behind an unrequested algorithm (2026-08-02)

`algorithm_list=[idsva_so_body_frame,fdsva_so]` + `enable_floating_second_order=True` emitted the
floating integrator-hessian SE(3) block (calls `grid_dIntegrate_{q,v}_block` /
`grid_d2Integrate_block`) but NOT `gen_lie_group_helpers` — every full-profile header gets those
helpers from ANOTHER consumer (integrator / f_ext_gradient / d2ee / frame_jacobian_dot), so the
gap only reproduces under a restricted list → nvcc "identifier undefined" deep in grid.cuh.
- **Fix pattern:** the EMITTER that emits the caller calls the (idempotent) helper emitter itself
  — `gen_lie_group_helpers` guards with `_lie_helpers_emitted`, so a duplicate call is a no-op
  and full-profile headers stay byte-identical.
- **Audit rule:** a helper emitted at ORCHESTRATION level (gen_all_code deciding "algorithm A is
  in the list so emit helper H") is a latent restricted-list bug for every OTHER emitter that
  uses H. Idempotent helper emitters called at the point of use are the robust shape.

### 1ab. `n+fb` (or `2n+fb`) as a stand-in for a POSITION-space width silently under-sizes fixed-base spherical buffers (2026-08-18)

Caught by the first full granular gpu-proof pass: `gen_arena_carve_struct(integrator_du_arena)`
failed its emission-time exactness check (resolved t-count 1047 != sizer 1049) on the spherical
fixtures only. Two buffers in `_integrator_du_extra_t_buffers` sized position-space storage with
velocity-space arithmetic: `s_q_orig` as `n+fb` and `s_x_kp1` as `2n+fb`. Those equal `nq` /
`nq+nv` for plain fixed base (nq==nv) AND floating (nq-nv==1==fb) — the only two shapes the
9-robot flagship set exercises — but a FIXED-base spherical robot has nq-nv = #spherical-joints
with fb==0, so each buffer came up short by nq-nv and the q-retract tail would overrun the next
arena buffer (the kernel emitter shares the same list). Sibling of the §7 "nq==nv makes
tight==slotted, fixed-base proves nothing" input-ABI trap, now on the SMEM-arena side.
- **Fix pattern:** size q-shaped storage with `nq` (and next-state with `nq+nv`) directly; never
  reconstruct it from nv+fb. Byte-identical for all non-spherical robots by the identities above.
- **Audit rule:** grep emitters for `+ fb` / `+ NUM_POS - NUM_VEL`-flavored arithmetic feeding a
  buffer that holds q or [q; qd]; each is a latent fixed-base-spherical bug.
- **Meta:** the carve struct's `expected_t_count` emission-time equality check (GATO ASK6) is what
  surfaced this — a `total <= sizer` slack check would have hidden the kernel-side under-sizing
  forever. Prefer exact-count invariants over upper bounds when two walks must agree.

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
`docs/source/user_guide/concepts/parallelism_patterns.rst`; status table: `docs/open-tasks/archive/parallelism_audit.md`).
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
  GRiD's emit until you also do the GRiD-side plumbing.** Mechanism (`grid_codegen/helpers/_lin_alg_helpers.py`):
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
- **SHAPE-VARYING cells collapse too — via SHAPE-BUCKETED tables (2026-07-07, W1a, extends the above).**
  The prior guidance ("keep shape-varying same-joint rev/pris + mimic cells in the residual `if==k`
  ladder") is superseded for the same-joint family. When a residual cell's op SHAPE varies over a SMALL
  fixed set (eepose same-joint = {rev-rev, mixed lin-ang, pris-pris}), bake a `shape` code + the offsets
  + the shape's constants into a per-cell table and emit ONE shared body with an internal
  `if(shape==0)…else if(shape==1)…` switch — each shape's arm appears ONCE, not once-per-cell. This
  collapses the residual ladder with the same emit-shape win, O(1) bodies instead of O(cells). (mimic
  block-pair, whose SUM length varies, stays in the small ladder — rare, mimic-robots-only.)
  - **Bit-identity trick for baked axis coefficients**: a world-axis emit that inline-DROPS near-zero
    coefficient terms becomes, in the table body, all-3-terms `X0*t0 + X1*t1 + X2*t2` with the near-zero
    coeff baked as exact `0.0`. This is BIT-identical (`X*0.0==0.0`, `sum+0.0==sum` in IEEE; no
    signed-zero/NaN in play) — same argument the gradient inner's `eeg_job_ax` table already relies on.
    Snap `|c|<1e-15→0.0` when baking and format `{:.17g}` so the loaded `const T` equals the old inline
    literal exactly. Gate = CUDA numerical equivalence (grid.cuh TEXT changes — that's the win — so it
    is NOT a byte-diff gate). Validated 2026-07-07 (GCG b8bc31e): CUDA equivalence PASSED on iiwa14-fixed
    (rev-rev + tail) AND go2-floating (mixed + off-diagonal + pris-pris, all three shape arms).
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

- **NEW robot or NEW GPU → run the launch-config autotune so it defaults to a FAST launch (A1).** The
  per-`(robot, base, algo)` optimal `(tier, threads)` is device- AND robot-specific (register-clamped big
  kernels want LOW threads; it is NOT "bigger → more threads"). Codegen bakes
  `config/launch_configs/<robot>/<gpu>.json` into `grid_launch_config.cuh` so the host launchers (and every
  python/jax/torch binding) default to it — fixing the FFI thread-default pathology at the C++ root. If a
  `(robot, GPU)` pair has no entry, GRiD falls back to a conservative (slow) default. To generate one:
  `bash config/autotune_robot.sh <robot> [fixed floating]` (RAM-safe serial build; single-call timing OFF —
  it needs the `-rdc` shim; tunes on batch N=256). It auto-detects the GPU key `<model>_sm<arch>` (override
  with `GPU_KEY=`), writes the override JSON via `config/autotune_to_launch_config.py`, then you re-codegen +
  rebuild to pick it up, and optionally PR the JSON (`config/launch_configs/README.md`) to crowdsource the matrix.
  Run it on a QUIET GPU (timing must be isolated). Full workflow:
  `docs/source/user_guide/tutorials/benchmarks.rst` ("Autotune launch config for your robot / GPU").

---

## 5. Merge discipline (multi-agent, file-isolated clones)

- Agents clone off varying bases → expect **3-way merges**. File-isolated agents merge clean; the
  ONE collision zone is `GRiDCodeGenerator.py`'s **`(historical: `_MIMIC_GRADIENT_ALGORITHMS`, removed 2026-08-27 — mimic refusals fully ungated)`** set (every mimic
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
- **Test the HOST-WRAPPER BATCH path, not just the single-timestep device fn (2026-06-08).** The
  floating nq-vs-nv matrix-buffer stride bug (Minv/M/dc_du/df_du host malloc+copy at nq² while the
  kernel writes nv²) corrupted only BATCHED floating (`init_gridData<T,B>`, B>1, slot k>0) — silent
  on fixed base (nq==nv) AND on batch=1. It survived because every CUDA equivalence test drove either
  the `*_device` functions or kernels with single-timestep buffers it owned, never the gridData `h_*`
  host-wrapper copy at B>1. New regression guard: `test/cuda_equivalents/test_cuda_batched_host_wrapper.py`
  (batched `grid::minv`/`grid::crba` host wrappers, every slot vs oracle, floating + fixed control).
  **Rule:** any new output buffer needs a batch>1 FLOATING host-wrapper equivalence test; a
  single-timestep or device-fn check cannot see a per-timestep stride bug.
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
- **Spatial-vs-hom transform CONVENTION DRIFT for multi-DOF joints (FIXED 2026-08-02, spherical ee
  grad/hess arc).** `URDFParser.Joint` composed the multi-DOF (floating/planar/spherical) HOMOGENEOUS
  transform as `hom_free * origin` while the SPATIAL `Xmat_sp = X_free * X_origin` (Featherstone
  parent→child) corresponds to hom = `origin ∘ free` (pinocchio `M_placement * exp(q)`). Net effect: a
  mid-chain ball joint's hom ROTATED ABOUT ITS PARENT'S ORIGIN, not its own anchor — EE world POSITION
  diverged from pinocchio whenever the ball's `<origin xyz>` ≠ 0 while ROTATIONS matched exactly, and
  the whole DYNAMICS suite stayed green (it consumes only the spatial X). Detection recipe that caught
  it: FD the pose ON THE MANIFOLD (`integrate` retraction) against the geometric-Jacobian gradient —
  the position rows disagreed by a constant lever-arm RATIO (1.75×) with perfect angular rows, i.e.
  "right axis, wrong fixed point". Also beware `origin.Xmat_sp_hom_fixed`: it is a MIXED object
  (TRANSPOSED frame rotation E + forward translation), not a composable forward hom — build
  `[[E.T, p],[0,1]]` explicitly before composing. Lesson: any joint type whose hom and spatial
  transforms are built by separate code paths needs BOTH a dynamics-vs-pin AND a hom-FK-vs-pin gate;
  dynamics green proves nothing about the hom chain.

---

## 7. Test-infra gotchas

- **A runner's support-header include silently kills every test that compiles it from an
  isolated dir (2026-08-02, `grid_runner_select.cuh`).** The 07-28 split scaffold added an
  UNGUARDED `#include "grid_runner_select.cuh"` to `cuda_equivalence_runner.cu`; the flagship
  harness copies the selector next to its runner copy, but three OTHER tests (fext,
  continuous_joint, fd_du_output_spill) copy only the runner into a tmp build dir → quote-include
  searches the includer's dir (NOT cwd, NOT the source tree) → fatal missing-include → those
  tests were compile-dead for five days and nobody noticed because no FULL `-m cuda_equivalence`
  pass ran in the window (night scripts run benches; day gates ran targeted tests). Lessons:
  (1) when adding a support header to a shared runner, grep for EVERY test that copies that
  runner and give each the copy (or add `-I` to the source dir); (2) a suite that hasn't run
  END-TO-END since a change to shared test infra has an unbounded blind spot — after touching
  shared runners/conftest, run the full marker suite once, not just the neighbor test.

- **NUM_JOINTS-framed dumps of NUM_VEL-strided buffers read unwritten memory on floating robots
  — and pass while fresh allocations happen to be zero (2026-08-02, fd_du go2 flake).** The
  monolith runner's FIXED-base section printed id/fd gradients as `NUM_JOINTS^2` blocks with the
  qd half at offset `NJ*NJ`, over a buffer the kernels write `2*NV*NV`-strided. Fixed non-mimic
  robots (nq==nv) are exactly right; but `test_cuda_fd_du_output_spill` compiled the runner
  WITHOUT `-DGRID_CUDA_FLOATING_BASE=1` for its go2-floating case, so the floating header ran
  the fixed section: the "qd" block (offset 361 > written extent 324) read allocation tail —
  zeros on a fresh GPU (weeks of green), garbage under memory pressure (today's 63/361 flake,
  run-to-run NONdeterministic in BOTH arms). Diagnostics that localized it fast: (a) run each
  exe TWICE — uninit reads are run-to-run nonidentical, real rung divergence is deterministic;
  (b) look at the VALUES — denormals/1e8 garbage vs the reference's exact ±0.0 means uninit
  read, not arithmetic; (c) map flat mismatch indices to (row, col) — a clean "last rows only"
  pattern is a framing/stride bug, not math. Fixes: the test passes the floating define, and the
  fixed-path prints are now NV-framed (byte-identical for every robot that legitimately reaches
  them). The bug predated 07-20 — a bisect chasing "what broke it" was chasing memory-pressure
  weather; characterize the failure mode BEFORE bisecting.

- **Zero-only sample coverage hides input-LAYOUT bugs (2026-07-30, the floating q_qd_u shear).**
  The equivalence runner packed the floating `q_qd_u` buffer TIGHT (`nq + 2*nv`, stride 40) while
  every generated kernel unpacks canonical nq-wide slots (`s_qd = &buf[nq]`, `s_tau = &buf[2*nq]`,
  stride `3*nq`) — qd/u sheared one slot, ~100% wrong output on any energetic state. It survived
  for weeks because the floating suite's default sample set was `{zero}` (a shear of zeros is
  zeros), and fixed-base coincides (`nq == nv` → tight == slot layout). Lessons: (1) any harness
  that PACKS a multi-segment input buffer must be validated against a sample where EVERY segment
  is nonzero and distinct — zero/identity states satisfy any layout; (2) when a suite gates a
  whole base-mode to one sample "for time", the un-run samples are an ACTIVE blind spot — record
  it as debt (this one is now fixed: default floating set = zero + velocity_only + mixed_sign +
  floating_quat_mixed); (3) a kernel that is right on `zero` and ~100% wrong (not noise-wrong) on
  energetic states with error scaling by input magnitude is an input-layout/stride suspect BEFORE
  it is a math suspect — diff the harness's pack offsets against the kernel's unpack offsets first.

- **GPU is SHARED-OK for CORRECTNESS, ISOLATED-only for TIMING (orchestration rule).** Equivalence /
  Gate-A / thread-invariance / batch runs are correctness checks — run MANY concurrently (sized to
  cores + RAM; an nvcc compile peaks ~5 GB, so cap concurrency by free RAM, not just core count). Only
  **performance timing** (single-call µs sweeps, tier sweeps, A/B) must run one-at-a-time on a quiet GPU
  — contention skews the numbers, and that is the ONLY reason to serialize. So the right pattern for a
  feature run is: fan out file-isolated correctness agents in parallel (each does its own Gate-A +
  equivalence), and quarantine the perf sweep to its own isolated phase at the end. "No concurrent heavy
  GPU builds" applies to TIMING, not to correctness builds. See [[feedback_parallel_equivalence_testing]],
  [[feedback_safe_dev_and_timing_methodology]].
- **A test that validates a CODEGEN EMIT must `force_rebuild=True`.** grid-rbd's build cache is
  content-addressed on build INPUTS (urdf + flags + arch + grid_rbd_version + wrapper_template_hash) —
  NOT on the generated CUDA source, and NOT on the GCG codegen version. So after a codegen change, a
  `register_robot(...)` with the same inputs silently returns a STALE `.so` built by the OLD codegen.
  This burned a damping-gradient test: the robots were cached before the gradient emit existed, so
  `inverse_dynamics_gradient` came out identical on-vs-off (the new term was in the source but not the
  cached binary) — looking exactly like a "missing emit" bug when the emit was correct. Fix: any pytest
  asserting on generated-code behavior must pass `force_rebuild=True` to `register_robot` (or clear
  `~/.cache/grid-rbd`). Same root cause as "clear GCG `__pycache__` after codegen edits" — the cache
  key doesn't track the codegen, so the human/test must force regeneration.
- **VENDORED dependency content is a codegen INPUT — the cache key must track it (GLASS bump, 2026-06-18).**
  The CUDA-equivalence header cache (`test_cuda_executable_equivalence._header_cache_key`) hashed
  `_hash_tree(GRiDCodeGenerator, ".py")` but NOT the GLASS submodule commit — yet GLASS sources are vendored
  VERBATIM into every generated `grid.cuh` (`helpers/_lin_alg_helpers.py::_emit_glass_source_file`). So after
  a GLASS bump the cache would FALSELY HIT headers vendored from the OLD GLASS — a silent stale-codegen
  validation, the same family as the grid-rbd build-cache trap above. Fix: fold `_glass_commit()` into the
  cache-key payload (git HEAD of the GLASS submodule, with a hash-of-`src/base` fallback for exported trees).
  **General rule:** anything copied INTO generated output (vendored headers, baked tables, template files) is a
  codegen input; if the cache key only tracks the generator's own source, a dependency bump goes undetected.
  When in doubt after bumping a vendored dep, clear `.pytest_cache/grid_cuda` to force regeneration.
- **The `grid::grid::` per-tier macro trap.** `GRID_DEFAULT_RESOURCE_TIER` is `#define`d BARE (`TIER_SHARED`).
  Algos emitting INSIDE `namespace grid` (11 of 12) reference it bare; `_plant` emits its kernel +
  `*_DYNAMIC_SHARED_MEM_BYTES` OUTSIDE the namespace so it correctly qualifies `grid::GRID_DEFAULT_RESOURCE_TIER`
  (bare would not resolve there — NOT a uniformity wart, do not "fix" it). The trap: a `-D` tier override must
  MATCH the bare `#define` style — passing `-DGRID_DEFAULT_RESOURCE_TIER=grid::TIER_*` makes `_plant` expand to the
  illegal `grid::grid::TIER_*` and breaks EVERY per-tier build. Pass bare `-D...=TIER_*`. (Cost a per-tier build
  break this session; fixed in `baselines/grid/run.py`.)
- **The bench harness `timeGRiD_{batch,single}.cu` + `timeGRiD_common.h` are NOT subset-aware.** Only the
  SO/integrator measure block is `#if GRID_HAS_*`-gated; the 10 CORE measures (id/minv/fd/aba/crba/id_du/fd_du/
  ee_pose{,_gradient,_hessian}) — both their CALLS and their `measure_*` / `*_single_timing` DEFINITIONS — reference
  `grid::<algo>` unconditionally. So a `GRID_BENCH_ALGORITHM_LIST` subset build fails to link on every omitted core
  algo. Gating the CALLS is necessary but NOT sufficient (the DEFINITIONS in `timeGRiD_common.h` + the
  `_single_timing`/`_batch_timing` wrappers must also be `#if GRID_HAS_*`-wrapped — bench analogue of C1, still TODO).
  Gating CALLS only is timing-neutral for FULL builds (`#if 1`), a safe partial step. After any gating edit, verify a
  full build still TIMES all 10 core algos (a wrong macro name silently drops an algo from the sweep).
- **Single-CALL timing is OPT-IN / DEFAULT-OFF (`run.py --single-timing` / env `GRID_BENCH_SINGLE_TIMING=1`; B8,
  2026-06-12).** The bench builds TWO timing binaries per cell: `timeGRiD_single.cu` (single-call latency,
  `-rdc=true`) and `timeGRiD_batch.cu` (batch throughput, `-rdc=false`). The single binary NEEDS `-rdc=true` for its
  anti-LICM shim, but under `-rdc` nvcc does NOT inline `inverse_dynamics_inner_vaf` (140 regs) into the
  `__launch_bounds__(128)` kernels (fdsva_so / integrator(_with)_gradient / id-gradient), so ptxas FATALLY errors on
  BIG FLOATING robots (g1/h1_2): `Entry function <kernel> with max regcount of 128 calls <inner> with regcount of 140`.
  There is no way to satisfy that under `-rdc`, and the doomed compile still burns ~50 min/tier before giving up — an
  overnight g1-floating autotune ran 7+ hours. The BATCH binary (`-rdc=false` → inner inlined → no regcount error) is
  the ONLY thing the autotune MATRIX uses. So `compile_binaries(build_single=...)` defaults the single build OFF; the
  flag/env opt back in. When OFF, NO single TU is compiled + NO single run happens in BOTH the standard path AND the
  `--autotune-threads` path (`build_tier_binaries` passes `build_single=(mode=='single')`, so batch autotune never
  touches it), with no "single-call build failed" / "skipping single run" churn. `run_multi_version.py` threads
  `--single-timing` through too (so `run_a1b_*.sh` are default-OFF). Opt in only when you actually need single-call
  latency on small/fixed robots; never expect it to build on g1/h1_2 floating.
- **Two DISTINCT caches in the bench path — a subset request must be in BOTH or a stale full-set artifact is served.**
  `run.py generate_header` keys a `codegen_hash` cache (hashes the GCG `.py` tree, so codegen edits self-invalidate);
  it now ALSO includes `GRID_BENCH_ALGORITHM_LIST` (else a cached full-set header is reused and the subset silently
  ignored). The binary `runner_key` includes `header_hash` so it follows. SEPARATE from the bindings `grid-rbd`
  content-cache above (which is NOT codegen-keyed). Don't conflate them.
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
- Parallel equivalence runs go through `test/run_split_suite.py` (RAM-aware compile pool in `test/compile_sched.py`); the old `run_parallel.sh` xdist wrapper was retired 2026-09-08.
  correctness-only and safe to run concurrent; the PERF sweep must run ISOLATED (no other GPU/CPU,
  it skews timing).
- **The editable `grid_rbd` install can point at a STALE sibling worktree.** `.venv` is under main
  but `pip install -e .` may have last run from another worktree (e.g. `GRiD-H-roadmap`), so a
  bare `import grid_rbd` silently loads OLD bindings — `AttributeError: 'RobotHandle' has no
  attribute 'inverse_dynamics_gradient'` for a method that exists in main. pytest under the repo tree
  passes (repo-relative `sys.path`) while a notebook/example fails. Confirm with
  `python -c "import grid_rbd, inspect; print(inspect.getfile(grid_rbd))"`; the editable `.pth`
  merely *appends*, so `sys.path.insert(0, '<main>/bindings')` (or `PYTHONPATH`) reliably overrides it
  for validation. The real fix is `pip install -e .` from the intended tree. (Sibling of the
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
- **A STRUCTURALLY ZERO reference makes a norm guard vacuous — and only an absolute floor is
  meaningful** (2026-08-03, fetch fd-gradient-q). The `norm_rtol` escape hatch computes
  `||diff|| / max(||expected||, 1e-12)`; when the reference matrix is identically zero that ratio
  explodes (observed `norm_rel = 9.9e8`) and the guard can NEVER bind, so the entrywise `atol` is
  the only thing holding the test. Fetch's fixed-base model carries a TRANSLATIONAL BASE DOF, so
  every zero-torque sample free-falls — `qdd = (0,0,-9.81,0...)`, no internal relative
  acceleration — and that is configuration-INDEPENDENT, making `dqdd/dq` structurally zero. The
  CUDA float32 residual (5.3e-4) was pure cancellation amplified by `||Minv||_inf ~9e2`
  (cond(M) ~2e4), and it drifted just past a 5e-4 floor. Diagnostics, in order: (1) `scale=2.1e-14`
  in the failure message already says the reference is zero — READ IT before assuming a numeric
  regression; (2) compare the quantity across ALL robots at the same sample (every other robot was
  O(40-250); fetch alone ~1e-12 — that isolation IS the finding); (3) finite-difference the
  oracle's OWN forward dynamics to prove the analytic zero is right, not a dropped term;
  (4) bound the float32 floor as `||Minv||_inf * eps32 * max|dID/dq|` and set `atol` just above it.
  Because `atol_eff = max(atol, rtol*scale)`, raising `atol` this way canNOT loosen the
  informative samples — anything with `scale >= atol/rtol` is still governed by `rtol`. Beware the
  mirror-image trap: a vacuously-zero comparison also PASSES vacuously, so a test that only ever
  sees a free-fall config validates nothing.

### 7.x SIGKILLing a running big-SO exe can WEDGE the NVIDIA driver — and the wedge is silent, sticky, and poisons every later leg (2026-08-09)

The 20260808 night queue: leg A1's `timeout` killed `per_algo_bench` mid-run while a
`solo_batch_f_ext_gradient*` exe (h2_plus, nv=81, ~31 GB device arena) was inside a
kernel/teardown. The exe became a ZOMBIE whose driver-side context cleanup never
completed: 31 GB stayed allocated, and from then on ANY process touching the driver
(`nvidia-smi`, the next leg's mjx/XLA init, even gnome-shell/VSCode's GPU process)
entered uninterruptible D-state. Symptoms to recognize:
- a timing leg with a **0-byte log for hours** (it's blocked in CUDA init, not slow);
- `gpu_guard`-style `nvidia-smi` probes reporting **0% util but ~full memory.used**
  right after a killed leg — that line IS the alarm; do not start the next leg on it;
- `nvidia-smi` itself hanging (D-state) — at that point `timeout`/`kill -9` cannot
  help (D-state ignores signals) and every further GPU probe adds another unkillable
  process. **Stop touching the GPU; only a reboot (user's call) clears it.**
Prevention: (1) don't wall-clock-kill timing legs in a dedicated window
(user rule, 2026-08-09 — one finished leg beats none); (2) if a leg must be stopped,
signal the PYTHON orchestrator with SIGINT/SIGTERM and give the exe time to exit its
sync + frees — never SIGKILL a process with tens of GB of live device allocations;
(3) inter-leg guards must VERIFY memory returned to baseline and FAIL LOUDLY instead
of proceeding (the A2 leg "passed" in 60 s on a card with 31 GB held — its numbers
are suspect, not bankable); (4) treat "util 0% + memory high + no owning process" as
a wedged-teardown signature, not as idle.

**ROOT CAUSE FOUND 2026-08-23 (second box freeze, NO kill involved): the
driver's LAZY vidmem free races the next launch.** With the timeout removed,
the same h2_plus f_ext autotune arm froze the box again — and the kernel log
shows the true first event: a NULL-pointer Oops in the NVIDIA open kernel
module's OWN kthread (`vidmem lazy fre`, nvidia_uvm `free_chunk` →
`uvm_pmm_gpu_process_lazy_free`) at 05:38:11, five seconds BEFORE the Xid
109/31 — those and the soft lockup are wreckage, not cause. Mechanism: exe #N
exits and its ~30 GB frees LAZILY in a background driver thread; the sweep
immediately launches exe #N+1 which allocates into the still-freeing memory —
serial processes, but DRIVER-INTERNAL concurrency — and 610.57.04 (open
kmod) races to the NULL deref. Only h2_plus triggers it: ~30 GB/invocation on
a 32 GB card × dozens of rapid autotune invocations. This also reframes
08-22's zombie (same teardown subsystem) and largely EXONERATES the GRiD
kernels (memcheck-clean arms stand; the Xid-31 "OOB write" was post-Oops).
- **FIX: the settle gate** (`gridrun.settle_gpu_before_launch`, wired into
  every bench exe launch path): before each launch, wait until memory.used is
  back at the process-start baseline (+1 GB margin, 120 s cap); refuse loudly
  on non-settle AND on a dirty-GPU process start (a fresh process must never
  adopt a wedged residue as baseline).
- **Audit rule:** any harness that launches GPU exes back-to-back needs the
  gate; "serial" at the process level is NOT serial in the driver.
- **System-level:** the NULL deref is an upstream driver bug (open kmod
  610.57.04) — a driver update / proprietary kmod may remove the race
  entirely; worth reporting upstream.

**RECURRED 2026-08-22 (campaign-1 night-1) — a PYTHON-LEVEL timeout is the same
SIGKILL.** `per_algo_bench._run_one` used `subprocess.run(timeout=900)`;
Python's `TimeoutExpired` path SIGKILLs the child. h2_plus's f_ext cell blew
the 900s default mid-op (~30 GB live) → zombie whose driver context cleanup
never completed → GPU 0% util with 29.9 GB held by a DEFUNCT pid for 9+ hours,
the whole night stalled silently behind it. New variant notes: `nvidia-smi`
stayed RESPONSIVE (softer than the full 2026-08-09 wedge) and the zombie was
UNREAPABLE even after its parents exited (kernel thread stuck in the driver
exit path) — reboot required all the same. FIX: per-exe timeout now defaults
0 = DISABLED (hang detection is output-progress, per the standing rule); a
positive timeout escalates SIGTERM → 120s grace → mark "hung" and LEAVE the
process. AUDIT RULE: grep every bench/orchestration harness for
`subprocess.run(...timeout=` / `communicate(timeout=` around GPU exes — each
is a latent SIGKILL of a process with live device allocations.

**CLOSED 2026-08-21 — h2_plus DE-QUARANTINED.** Root cause was the init-copy
async race fixed @660e626 (garbage q under concurrent load → wild indices →
OOB → Xid-31). Supervised repro at the fixed tree (fe55531, exes REBUILT —
the earlier sanitizer arms had run pre-fix binaries): 20× TWO concurrent plain
`solo_batch_f_ext_gradient.exe floating` (10 simultaneous + 10 staggered
starts, covering init-vs-init and init-vs-kernel interleavings) — every
iteration rc=0/rc=0, memory back to baseline, zero kernel-log events,
nvidia-smi responsive. f_ext_gradient_dq: solo clean at the fixed tree;
concurrent dq+fegrad is VRAM-INFEASIBLE at N=1024 on 32 GB (outputs are not
slot-clamped; the exe refuses with a clean cudaMalloc OOM exit(2) — run dq
arms SERIALLY, as the night scripts always did). h2_plus timing cells may
run again; the SIGKILL prevention rules above remain in force regardless
(they are about teardown, not this bug). Repro protocol + logs:
test/benchmarks/results/wedge_20260821/run_wedge_repro.sh. Verify Xid claims
via `journalctl -k` (dmesg is permission-blocked for this user — and beware
`cmd 2>/dev/null | tail` readability probes: the pipe eats the failure).

### 7.x Compile-RAM crash discipline for big-robot bench builds (salvaged from the 2026-07-23 one-off scripts before deletion)

Two desktop-crashing OOMs taught the g1-floating build schedule; the durable rules
(the dated `run_g1_*`/`probe_*` scripts that carried them are deleted — this entry
is their tombstone):
- `--ram-per-compile-gb` is only a PRE-ADMISSION check — it cannot throttle a
  compile that balloons after admission. **Both crashes came from trusting it.**
  The real guard is a hard cgroup cap applied by the caller
  (`systemd-run --user -p MemoryMax=...`), which lets the kernel kill a ballooning
  cicc instead of taking the desktop down.
- `run_multi_version --build-jobs 1` SKIPS the RAM-managed build phase entirely
  (that exact combination crashed the box at 01:59) — to serialize big compiles,
  drive `per_algo_bench --compile-jobs 1` directly instead.
- Schedule shape that works: non-SO algos parallel (small TUs), SO monsters
  strictly serial (24-36 GB each observed on g1-floating; worst non-SO single
  compile 17 GB), and no `set -e` in the night script so a late failure can't
  discard earlier banked exes (everything is content-cached and resumable).
- 2026-08-08 addendum: `GRID_ENABLE_MUJOCO_KERNELS=0` (pin-only) shrinks the same
  h2_plus SO-family compiles to ~1 min each — always build pin-only for timing
  cells; the mjx twins are what made these "monster" compiles.

---

## 8. Process meta-lessons

- **The backlog drifts — verify "is this still open?" before dispatching.** This session found 5
  "open" items already done (C1-float, C3, mxS-ndim warning, mimic regressor-Y, FD param-grad). A
  cheap grep/check beats an agent redoing landed work.
- **Gate/overflow/"not-emitted" claims in WRAPPER COMMENTS and dated MEMORIES go stale as ladders/folds
  land — re-derive from code before trusting them.** The 2026-06-10 audit re-flagged ~6 items that were
  already resolved: "h1_2 integrator/idsva_so OOB" (the per-tier spill ladder now fits — verify by reading
  `*_t_count_per_tier` × `cuda_shared_mem_type_size_bytes` vs the 98KB cap, no nvcc needed); "mimic centroidal
  NOT emitted" (it IS — verify by calling `_normalize_codegen_algorithms("all")` and checking the algo set for
  a mimic robot like fr3/h1_2, + the validating runner); "prismatic `theta` undeclared" (the substitution was
  already in `_eepose_gradient_hessian.py`); "eePos/deePos rename pending" (grep finds only the clean current
  name). CHEAP CHECKS THAT SETTLE IT WITHOUT A BUILD: (a) instantiate `GRiDCodeGenerator(robot)` +
  `gen_add_constants_helpers()` and print per-tier arena bytes vs cap; (b) `_normalize_codegen_algorithms`
  to see what's actually emitted; (c) grep the validating smoke-runner/test for the robot. A wrapper comment
  that says "NOT emitted for mimic / returns rc=3" is describing the `#else` branch, NOT proof the `#ifdef`
  is off — confirm which branch a real mimic robot takes. When you find a stale comment, FIX IT in the same pass.
  Also: grep for the WRAPPING MACRO, not just the literal API — "smoke runners miss `cudaGetLastError`" was a false
  positive; every runner checks launches via the `gpuErrchkKernel()` macro (count `gpuErrchk` ≥ `<<<`, not `cudaGetLastError`).
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
- **Keep the ON-DISK todo / backlog current as work lands — not just the ephemeral in-session list (USER
  PREFERENCE).** The harness TodoWrite list is per-session and disappears at `/compact`; the durable state
  lives in `docs/open-tasks/session_progress_*.md` (live commit log) + the authoritative forward backlog
  (`backlog_*_post_features.md`) + the `feedback_*`/`project_*` memories + `MEMORY.md` pointers. Update
  these AS each slice commits (mark done, append the commit SHA, move/refresh outstanding items, supersede
  stale lists) so a fresh context can resume losslessly. When a doc goes stale (an item it lists is now
  done), fix it IN THE SAME PASS and add a "SUPERSEDED -> <new doc>" header rather than leaving two
  conflicting lists. Treat the on-disk todo/backlog as a first-class deliverable of every work session.
- **GPU sharing: PARALLEL for correctness, SERIAL only for performance timing (USER PREFERENCE; see §7).**
  Equivalence / Gate-A / thread-invariance / batch builds are correctness checks — fan them out
  concurrently (file-isolated agents, each doing its own Gate-A + equivalence), sized to cores + free RAM
  (an nvcc compile peaks ~5 GB). The ONLY thing that must run one-at-a-time on a quiet GPU is PERFORMANCE
  TIMING (single-call us sweeps, tier sweeps, A/B) — contention skews the numbers. So the default shape of
  a feature run is: many concurrent correctness agents (scope -> verify -> merge), then a quarantined
  isolated perf-timing phase at the end. "No concurrent heavy GPU builds" applies to TIMING, never to
  correctness. [[feedback_parallel_equivalence_testing]] [[feedback_safe_dev_and_timing_methodology]]

## 9. Lessons (2026-06-15 — runtime_transform + autonomous-run session)

- **Shared-helper scratch must be reserved in EVERY per-algo arena `t_count` (a SILENT OOB class).** When a SHARED device
  helper (e.g. `load_update_XImats_helpers`) writes a new block into `s_temp` (runtime_transform appended a 36·NB `Xfixed`
  block at offset 2·num_pos), growing the helper's OWN declared temp size is NOT enough — every algorithm's arena `t_count`
  (grid_codegen/_constants_arena.py (arena/tier section; moved in the 2026-08-27 monolith split), feeding `grid_shared_arena_bytes(t_count,…)`) must reserve it too, or the helper writes past
  the kernel's dynamic-shared allocation. NO compile error, and the functional test can pass on small data — only
  **`compute-sanitizer --tool memcheck`** catches the "Invalid __shared__ write … out of bounds". A purely-additive `+= reserve`
  per arena is safe when the block is consumed inside the helper (dead after). The M descriptor table kills this class via an
  auto-injected reservation region (`design_descriptor_table_spec.md`).
- **Don't trust a subagent's "done" — capture the verdict yourself** (recurred 3× this session). Codegen agents end their turn
  with "waiting for the Monitor event" while their OWN detached validation (nvcc + `/tmp/validate_*.py`) is still building, so
  they report nothing and commit nothing. After an agent returns: `git diff --stat`, `ps` for a detached `nvcc`/`validate_*`,
  wait on the PID, then RE-RUN the validation yourself — definitive gate = compute-sanitizer + a committed equivalence test.
  Memory `feedback_capture_subagent_verdict_yourself`. (Also: pass `isolation: worktree` so WIP isn't left on the main tree.)
- **Editing codegen invalidates the .so cache → every robot is a FRESH 28-57min rebuild.** The cache key hashes the
  GRiDCodeGenerator+URDFParser source (the J fix), so after any codegen commit ALL robots cache-miss. This paces GPU
  validation; keep it serial, prefer light robots (iiwa14/fr3) for correctness, reserve g1/h2_plus SO builds for when needed.
- **"Already built but unvalidated" is the dominant backlog state.** This session confirmed mimic, damping/friction,
  install-extras, runtime_inertia were ALL already implemented — the work was VALIDATION (run the test) + closing narrow gaps,
  not building. Always grep/run-the-test before authoring a "missing" feature.

### 7.x Cross-surface ABI drift: widen a kernel input on ONE binding surface, silently break the twins (2026-08-04)
The tool-welding feature widened the runtime-EE offset from a 3-float point to a 16-float
col-major SE(3) `X_tool` — kernel + numpy C-ABI were upgraded, but the parallel jax-FFI and
torch-op surfaces kept 3-float semantics. Result: jax sent `off.reshape(-1)[:3]` of the NEW
16-float layout = **R_tool column 0** (`[1,0,0]` — not the translation, which lives at
`[12..14]`), staged onto a shared device buffer whose other 13 entries held whatever the
numpy surface last wrote; torch handed a 3-element tensor's pointer straight to the kernel's
16-float parameter = device OOB read (and a process-killing abort mid-suite). Lessons:
- **An input-layout change must be swept across EVERY surface that packs it** (numpy C-ABI,
  jax FFI impl + handler-symbol attrs, torch op + TORCH_CHECK, and BOTH Python wrappers) —
  grep the buffer/parameter name repo-wide before calling the feature done.
- **The failure signature identifies the mechanism**: position diff bounded by `2|t|` with
  rotation diff EXACTLY zero = the offset was dropped (stale identity), not misapplied in the
  wrong frame. Frame bugs perturb rotation too.
- **Default-value tests cannot catch it**: with `X_tool = I`, writing `[1,0,0]` over the
  first 3 entries of an identity is a no-op — only a NON-default offset exposes the drift
  (same shape as the fixed-base-hides-floating-slot-padding class: the degenerate case
  coincides bit-exactly with the wrong layout).
- Shared per-.so staging buffers (`d_eepose_runtime_offset`) make the wrong path's output
  depend on the OTHER surface's last call → order-dependent test outcomes; suspect a shared
  buffer whenever a failure appears/disappears with test ordering.

### 7.x Worktree A/B arms fail SILENTLY three ways (2026-07-31, the night-2 prebuild)
Building an old-commit arm for an interleaved A/B via `git worktree` has three traps that
compose into a "clean" prebuild with ZERO binaries in one arm (rcA=0, no error lines):
1. **`git worktree add` does not populate submodules** → codegen dies on missing GLASS.
   Always `git -C $WT submodule update --init external/...` after creating one.
2. **Relative paths resolve against the WORKTREE root** in the worktree's harness (its
   `run.py` prepends its own repo root): a relative `--build-dir` bakes a relative
   `GRID_HEADER_FILE` into nvcc, which then can't find the header. Pass ABSOLUTE dirs.
3. **`per_algo_bench --compile-only` (pre-af27345) exited 0 on failed compiles** — the
   BUILD-phase philosophy ("failures surface in the measure phase") is wrong for A/B
   prebuilds, where the measure phase would time a one-armed pair and produce
   plausible-looking garbage. Fixed at HEAD (af27345: nonzero exit + missing list), but
   OLD worktree arms keep the silent behavior forever → orchestration scripts must ALSO
   verify expected exe COUNTS per build dir and hard-abort the timing leg on a shortfall.
Detection heuristic: a build phase that "succeeds" suspiciously fast + an empty
`ls build_dir/*.exe`. The exe count is the ground truth, not the exit code.

### 7.y Post-emission transform passes: "must have matched" invariants break restricted profiles (2026-08-10)
A codegen pass that rewrites emitted text (e.g. the host-thread-clamp pass) needs a
drift tripwire — but `raise if rewrites == 0` is the WRONG invariant. Restricted
`algorithm_list` regens (the 13 aux equivalence tests, plant-only headers) legitimately
emit ZERO of the pattern, so the global invariant turned every narrow-profile test into
a hard failure while all full-profile gates stayed green. Correct postcondition is
per-site accounting: every matched site is either rewritten or CONSCIOUSLY skipped
(`rewritten + skipped == total`), and "pattern token present in the file but zero
launch lines matched" is the real drift signal. Corollary: gate transform passes on at
least one RESTRICTED-profile regen, not just full-profile byte-identity — the failure
mode lives exactly where the full-profile gate can't see. (Mapped from a dots-only
suite log via the `--collect-only` order-replay trick, §triage.)
Second escape from the SAME wave, same lesson inverted: an emission that only fires
on SOME robots (±inf limit defaults — only robots with unlimited slots) compiled on
the one robot the gate compiled (iiwa14, full limit tags → zero inf rows) and broke
everywhere else (`std::numeric_limits` with no `<limits>` in grid.cuh). Text-grep
gates don't catch uncompilable spellings; compile gates must cover a robot that
actually EMITS the new lines. grid.cuh deliberately has C-header includes only —
prefer `INFINITY`/math.h forms over `std::` in emitted code.
Third escape, same family: emitting a CALL to a new `glass::` primitive without
adding its source file to the `_lin_alg_helpers.py` vendoring list — the name
resolves in the GLASS checkout but not in the emitted, self-contained grid.cuh.
Any new glass:: reference in an emitter needs (a) the vendor-list entry (ordered
after its dependencies) and (b) a compile gate on a header that emits the call.

### 7.y2 Copy-on-streams[0] + launch-on-default-stream = input race under NONBLOCKING streams (2026-08-11)
Root cause of the ps5 ordering-dependent failure (fr3 dccrba: module-run fails
1.33 / solo passes / EVERY sanitizer passes with 0 findings). Every generated
host wrapper does `cudaMemcpyAsync(d_q, h_q, ..., streams[0])` then launches the
kernel on the DEFAULT stream — an ordering that is only safe via legacy
default-stream implicit sync, which `cudaStreamNonBlocking` (in init_grid since
the original codebase) explicitly disables. The race hid for years because:
(a) large pageable copies take the driver's synchronous path, (b) a process's
FIRST launch is slow (module load) so the copy wins, (c) all sanitizers
serialize. A WARM process making the FIRST call on a SECOND robot's handle with
a TINY copy (144 B) loses the race deterministically → kernel computes a
structurally-valid answer for a GARBAGE q. Diagnostic signature that finally
cracked it: wrong values that are (1) no rearrangement/permutation of the
correct tensor, (2) zero exactly in q-independent slices, (3) "cured" by every
sanitizer, (4) dependent on prior in-process GPU activity. Fix: create streams
with `cudaStreamDefault` (blocking) — one emitter line; the whole
copy->launch->copy pattern then inherits legacy ordering. ⚠Bench numbers are
not silently comparable across this change (stricter sync). ⚠Suspected same
mechanism behind the h2_plus f_ext Xid-31 OOB-write wedge under concurrent
load (garbage inputs -> wild indices); the supervised 2-process repro is the
confirmation experiment. Rule: any async producer op must share a stream (or
an event) with its consumer — grep new wrappers for `<<<` launches whose
inputs were last touched on a different stream.

### 7.z Bindings .so cache poisoning: disk-hash key, loaded-module emission (2026-08-11)
`register_robot` keys the compile cache by hashing the ON-DISK `grid_codegen/`
tree (`_cache.py`), but the code it compiles is emitted by the ALREADY-IMPORTED
modules. A long-running pytest session (15h equivalence pass) imported codegen at
00:13; a codegen commit landed at 12:38; wrapper tests registering robots after
that wrote store entries whose KEY says post-commit but whose CONTENT is
pre-commit emission. A later "verification" run at the new tip then cache-hits a
poisoned entry and proves nothing — detectable only because an 11.7s wall time
was too fast to contain the expected rebuild. Rules: (a) treat a suspiciously
fast post-codegen-change wrapper run as a cache hit to INVESTIGATE, not a pass
(wall-clock is the tell, same spirit as §7.x's exe-count rule); (b) after
committing codegen changes while any long pytest session is alive, purge store
entries newer than the commit (`stat -c %Y` vs `git show -s --format=%ct`).
Root fix (backlogged): key the cache on the EMITTED bytes (generate first,
hash the generated header + template + flags) — content-addressed emission is
immune to the loaded-vs-disk skew by construction.

---

*Linked from HANDOFF.md. Companion: `docs/idsva_so_inner_refactor_notes.md` (SO internals + resume hints).*

### 7.z2 Lie-chart second derivatives: the dN(0) chart-slope term is easy to drop (2026-08-11)

Building the tangent-space Newton hessian of a manifold cost
`g(delta) = cost(integrate(q, delta))` by chain-ruling through
`e(delta) = difference(q_des, integrate(q, delta))`:
the FIRST derivative at delta=0 is just `J_diff = dIntegrate(q_des, e, 'v')^-1`
(the chart factor `N(delta) = dIntegrate(q, delta, 'v')` is identity at 0), so
any gradient FD gate passes. But the SECOND derivative differentiates BOTH
factors of `de/ddelta = M(e)^-1 N(delta)`: the `-J (dM) J` term through e AND
`J^T`-composed `dN(0)` — and `dN(0) != 0` even though `N(0) = I` (it is the
right-Jacobian slope, `dJ_r/dphi_k|_0 = -1/2 skew(e_k)`; in RBDReference,
`d2Integrate(q, 0, 'v', 'v')`). Dropping it produced a hessian wrong by O(1)
(max err 1.48 on go2) while the gradient was FD-perfect.
Rules: (a) a gradient-level FD gate proves NOTHING about a chart hessian —
gate the hessian against 4-point second-order VALUE differences taken in the
SAME chart (perturb delta at the evaluation point, never FD the gradient
across charts); (b) any identity of the form `M(e(delta)) J(delta) = I` you
derived at a point should be re-checked away from that point before
differentiating it — here the correct identity is `M J = N(delta)`.
Landed use: `quadratic_state_cost_tangent` (RBDReference/_plant.py), gates in
tests/test_tangent_state_cost.py. The CUDA ASK3 preset must carry both terms.

### 7.z3 FFI "kernel launch failed" masks THREE distinct causes (2026-09-05, h1_2 autotune)

The jax/torch FFI handlers report every post-launch `cudaGetLastError() != cudaSuccess`
as `"<name>_kernel launch failed"` — but that sticky error can be (a) a real launch-config
error, (b) an error from the handler's OWN pre-launch async memcpys, or (c) **launch-time
`cudaErrorMemoryAllocation`**: the runtime could not allocate the kernel's local-memory
pool because VRAM was exhausted. Cause (c) was the h1_2 story: XLA preallocates 75% of
VRAM at backend init, and `test/conftest.py`'s `XLA_PYTHON_CLIENT_PREALLOCATE=false`
guard only covers pytest — every STANDALONE driver ran unguarded, so heavy kernels
(fd/minv/fdgrad on a 46-joint humanoid) failed at launch while light ones (id) passed,
mimicking a per-kernel smem bug. Now guarded at the top of
`test/benchmarks/baselines/grid/timeGRiD_bindings.py` (imported by the drivers before jax).

Triage recipe for FFI "launch failed":
1. Run the SAME call on the numpy/pybind surface in the SAME process — it goes through the
   generated host wrapper with `gpuErrchk` file:line + rc=100+cudaError, naming the real
   error (rc=102 = memory allocation). Fastest isolator.
2. If pybind passes standalone but fails alongside jax → VRAM pressure (prealloc class).
3. If ALL surfaces fail at 32 threads → check `<ALGO>_DYNAMIC_SHARED_MEM_BYTES` in the
   generated header vs `sharedMemPerBlockOptin` (99KB on sm_120). h1_2's
   IDSVA_SO_BODY_FRAME bakes 782,584 floats ≈ 3.0MB at EVERY tier — unlaunchable on any
   GPU, the legitimate hardware-limit skip class (real fix = SO memory wave-2).
