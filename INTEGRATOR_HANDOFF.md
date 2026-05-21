# Time-Integrator Codegen — Handoff & Open Questions

Status as of 2026-05-21 (branch `modernizing-tests`). This documents the
time-integrator work (Euler / Semi-Implicit Euler / Midpoint / RK3 / RK4),
what is validated, what is gated, and — importantly — a **contradiction in the
floating-base gradient story that is currently UNRESOLVED** and must be settled
before the floating gradient is trusted or its xfail removed.

If you are a future agent picking this up: read the "CRITICAL OPEN QUESTION"
section first. The xfail rationale shipped in the code comments may be wrong.

---

## 1. What was built

GRiD now emits time-integrator kernels alongside the existing dynamics
algorithms, in the usual inner / device / kernel / host layering, dispatched at
compile time on `template <typename T, IntegratorType IT>`.

Per integrator, three entry points:
- `integrator<T,IT>`                       — value only, `x_{k+1}`
- `integrator_gradient<T,IT>`              — gradient only, `dAB = [A|B]`
- `integrator_gradient_with_x_kp1<T,IT>`   — both at once

State / shape conventions:
- `q` size `nq` (= `nv` fixed-base; `= nv+1` floating-base, xyzw quaternion).
- `v` size `nv`, GRiD internal order `[ω(3), v_lin(3), joint_v…]` (angular first).
- `x_{k+1}` size `nq + nv` (NOT `2·nv`); host wrappers truncate the
  over-allocated `d_x_kp1` tail.
- `dAB` shape `(2·nv) × (3·nv)`, column-major, columns `[∂/∂q | ∂/∂v | ∂/∂u]`
  all in **tangent space** (so the shape is identical fixed vs floating).

Canonical Python lives in the `RBDReference` submodule (not the adapter):
- `RBDReference.integrator(q,qd,u,dt,integrator_type)` — [RBDReference/RBDReference.py:333](RBDReference/RBDReference.py#L333)
- `RBDReference.integrator_grad(...)`                  — [RBDReference/RBDReference.py:372](RBDReference/RBDReference.py#L372)
- SE(3)/SO(3) Lie-group helpers (`_quat_mul_xyzw`, `_quat_exp_from_half_omega`,
  `_so3_right_jacobian`, `_se3_Q_block`, `integrate`, `dIntegrate`) —
  [RBDReference/RBDReference.py:98-330](RBDReference/RBDReference.py#L98).

`ProjectModelAdapter.integrator / integrator_gradient` are thin pass-throughs
that wrap RBDReference and apply `normalize_vector` / `normalize_matrix`
([project_adapter.py:112-119](test/pinocchio_equivalents/utils/project_adapter.py#L112)).
**Note:** `normalize_matrix` is the IDENTITY — `np.atleast_2d(as_float64(...))`
([normalization.py:166](test/pinocchio_equivalents/utils/normalization.py#L166)).
It does NOT reorder ω/v_lin. RBDReference already emits in GRiD internal order,
so CUDA-vs-ProjectModelAdapter is an apples-to-apples internal-order comparison.

CUDA codegen:
- `GRiDCodeGenerator/algorithms/_integrator.py` — value path incl. floating
  Lie-group retract helpers for the q-update (`integrate_q`).
- `GRiDCodeGenerator/algorithms/_integrator_gradient.py` — gradient assembly.
  Floating Euler reads precomputed `s_dInt_q_6x6` / `s_dInt_v_6x6` for the top
  `nv` rows; bottom rows are the fixed-base code (`dt·J_qq | I+dt·J_qv | dt·Minv`).
  Floating SI-Euler and all floating multi-stage gradients are behind
  template-dependent `static_assert`s (not yet wired) —
  [_integrator_gradient.py:133,173,212](GRiDCodeGenerator/algorithms/_integrator_gradient.py#L133).
- `GRID_HAS_INTEGRATOR` / `GRID_HAS_INTEGRATOR_GRADIENT` macros gate consumer
  code so a value-only build still compiles
  ([GRiDCodeGenerator.py:1233](GRiDCodeGenerator/GRiDCodeGenerator.py#L1233)).

---

## 2. What is validated (green, trusted)

- **Fixed-base, all 5 integrators, value + gradient + both-at-once**: CUDA ↔
  ProjectModelAdapter(RBDReference) passes at rtol=atol=5e-4 on iiwa14-fixed
  and go2-fixed across dt ∈ {1e-3, 1e-2, 1e-1} and corner + random samples.
  Last verified 2026-05-21: `2 passed`.
- **Floating-base, all 5 integrators, VALUE path**: passes (~5e-7) on
  go2-floating. The q-update IS the Lie-group retract; cross-checked against
  Pinocchio `pin.integrate` (see `test_integrator_pinocchio_equivalence.py`,
  20/20).
- **Floating-base Euler GRADIENT**: emitted and exercised, but currently
  **xfail** — see below.

Test files:
- CUDA equivalence: `test/cuda_equivalents/test_cuda_integrator_equivalence.py`
  (smoke runner `cuda_integrator_smoke_runner.cu`).
- Pinocchio equivalence: `test/pinocchio_equivalents/tests/test_integrator_pinocchio_equivalence.py`.
- FD sanity (analytical ↔ finite diff): `test/pinocchio_equivalents/tests/test_integrator_gradient_fd_sanity.py`
  (fixed-base only; floating FD-sanity deferred — needs SE(3) log for the
  tangent-space perturbation).

---

## 3. The floating Euler gradient xfail — how the test is wired

`test_cuda_integrator_matches_python_reference[go2-integrator-floating]`:
- checks VALUE for all 5 integrators (must pass),
- checks the Euler GRADIENT, and if it mismatches, calls `pytest.xfail(...)`.

Design intent: the day the floating Euler gradient matches, the test passes
with **no xfail** — a fully-green run (no xfailed integrator line) is the signal
that the floating gradient is correct. SI-Euler / Midpoint / RK3 / RK4 floating
gradients are not emitted (kernel static_assert), so only Euler is checked for
floating; the smoke runner guards them with
`if constexpr (!FLOATING || IT==EULER)` so the un-emitted kernels are never
instantiated.

Last verified 2026-05-21: go2-floating → `1 xfailed` (values pass, Euler
gradient mismatches → xfail).

---

## 4. CRITICAL OPEN QUESTION — the xfail's stated cause is contradicted

**The code comments and commit messages attribute the floating Euler gradient
mismatch to a structural bug in CUDA `forward_dynamics_gradient` (dropping
linear↔angular velocity-coupling in the dqdd/dqd spatial 6×6 block for
floating-base). New evidence contradicts this. Treat the root cause as UNKNOWN.**

Evidence AGAINST the "FD-grad structural bug" story:
1. `test/cuda_equivalents/test_cuda_executable_equivalence.py` validates CUDA
   `forward_dynamics_gradient_qd` (= dqdd/dqd, = J_qv) against
   ProjectModelAdapter(RBDReference) for go2-floating, and there is **no
   tolerance override** for `("go2","forward_dynamics_gradient_qd")` (overrides
   exist only for iiwa14/gen3/baxter qd) — i.e. go2-floating dqdd/dqd passes at
   STRICT tolerance. See the overrides table
   [test_cuda_executable_equivalence.py:90-189](test/cuda_equivalents/test_cuda_executable_equivalence.py#L90).
2. `normalize_matrix` is the identity, so that strict comparison is
   apples-to-apples in GRiD internal order — the same order the integrator
   gradient is compared in.
3. The integrator-gradient bottom `nv` rows are literally
   `dt·J_qq | I + dt·J_qv | dt·Minv`, built from that same (validated) CUDA
   `forward_dynamics_gradient` + `direct_minv`
   ([RBDReference.py:404](RBDReference/RBDReference.py#L404) mirrors the CUDA
   assembly). If CUDA J_qv == RBDReference J_qv strictly, the integrator-gradient
   bottom rows MUST match too.

So if the bottom rows match, any real integrator-gradient mismatch must live in
the **top `nv` rows** (the SE(3) `dIntegrate` blocks `s_dInt_q_6x6` /
`s_dInt_v_6x6`) — the OPPOSITE of what the shipped comments claim (they say top
dIntegrate rows match ~1e-7 and the bottom rows inherit the bug).

Likely explanation for the original "0.48 error" finding: the standalone debug
driver (`/tmp/go2_minv_hypothesis.py`, `/tmp/grid_go2_grad_check.cu`) compared
CUDA against **raw RBDReference** without the convention/operating-point
handling the adapter path uses (ω-first vs v_lin-first ordering in the spatial
6×6, and/or a different qdd operating point in `inverse_dynamics_gradient`). A
6×6 ordering mismatch produces exactly a "dropped velocity-coupling" signature.
That artifact, not a kernel bug, may be what was measured.

### What a future agent must do before trusting/removing the xfail
1. **Measure the real mismatch.** Instrument
   `test_cuda_integrator_matches_python_reference` (or a one-off driver) to print
   `max|dAB_cuda − dAB_ref|` split into top `nv` rows vs bottom `nv` rows, per
   column-block `[∂/∂q | ∂/∂v | ∂/∂u]`, for go2-floating Euler. ~4 min (nvcc).
2. **If the error is in the bottom rows**: reconcile with the strict
   `forward_dynamics_gradient_qd` pass — likely the integrator kernel computes
   its internal qdd / FD-grad at a different operating point than the standalone
   `forward_dynamics_gradient` host wrapper. Find where they diverge.
3. **If the error is in the top rows**: the `s_dInt_q_6x6` / `s_dInt_v_6x6`
   emission in `_integrator_gradient.py` is wrong for the CUDA path even though
   `RBDReference.dIntegrate` matches Pinocchio. Compare the emitted SO(3)
   right-Jacobian against `RBDReference.dIntegrate(q, dt·qd, 'q'/'v')`.
4. **If there is no large error (just marginally over 5e-4)**: it may be float32
   noise on a stiff Minv — then the fix is a per-entry tolerance override (like
   the other floating FD-grad entries), not a code change, and the xfail should
   become a normal pass with an override.
5. Update the comments in `GRiDCodeGenerator.py` (`_normalize_codegen_algorithms`,
   ~[line 163](GRiDCodeGenerator/GRiDCodeGenerator.py#L163)) and this doc once
   the real cause is known. Do NOT propagate the "structural FD-grad bug"
   narrative further until it is independently re-confirmed.

### MEASURED 2026-05-21 (step 1 done) — error localized to the J_qv block

Ran step 1 (split `max|dAB_cuda − dAB_ref|` top `nv` rows vs bottom `nv` rows,
per column-block) for **go2-floating Euler, all 15 samples, dt=0.01**, on the
`humanoid-tier-spill` branch (after merging modernizing-tests). Result, every
sample:

- **Top `nv` rows** (SE(3) `dIntegrate` blocks `s_dInt_q_6x6`/`s_dInt_v_6x6`):
  `[dq|dv|du] = [0, 0, 0]` — **exact**. The integrator's own SE(3) Jacobian
  assembly is correct; it is NOT the source.
- **Bottom `nv` rows** (`dt·J_qq | I+dt·J_qv | dt·Minv`): error is **isolated to
  the ∂/∂v column block** = `I + dt·J_qv`, magnitude **~0.002–0.010**. The
  ∂/∂q (`dt·J_qq`) and ∂/∂u (`dt·Minv`) blocks are `~0` (≤1e-6).

So the mismatch is **purely in `J_qv = ∂qdd/∂qd` velocity-coupling**, surfaced
through the bottom rows. This *confirms the location* the shipped comment
claimed (bottom/FD-grad rows) and *refutes* the §4 worry that it might be the
top dIntegrate rows.

Magnitude argues **structural, not float32 noise**: at dt=0.01 a 0.002–0.010
block error ⇒ `J_qv` error ~0.2–1.0 (far above float32 noise on a stiff Minv).

**Remaining contradiction to resolve (step 2):** standalone
`forward_dynamics_gradient_qd` (= J_qv) passes STRICT for go2-floating with no
override, yet the integrator's J_qv is off by ~0.2–1.0. Since the integrator
builds its bottom rows from the **inlined** FD-grad
(`gen_forward_dynamics_gradient_inner_python`) rather than the standalone
`forward_dynamics_gradient_kernel`, the prime suspects are now:
  (a) the **inlined FD-grad path differs from the standalone kernel** on
      floating-base velocity coupling, or
  (b) an **operating-point difference** — the integrator computes its own qdd
      (FD value step) and evaluates FD-grad there, while the standalone test
      supplies qdd; J_qv depends on the qdd operating point.
Next: dump the integrator's internal J_qv vs the standalone kernel's J_qv at the
**same** (q, qd, u, qdd) and diff — that isolates (a) vs (b). (Not a
forward_dynamics_gradient *structural* bug in the standalone sense — that path
is independently validated, incl. by the humanoid-tier-spill rollout's
`forward_dynamics_gradient_qd` fixed+floating equivalence pass.)

---

## 5. Known limitations (independent of the open question)

- Floating SI-Euler / Midpoint / RK3 / RK4 **gradients** are not implemented
  (kernel `static_assert`). They need the per-stage `dIntegrate` chain-rule
  wiring (the value path already does the per-stage Lie retract). The Python
  `RBDReference.integrator_grad` DOES implement all five for floating
  ([RBDReference.py:427-462](RBDReference/RBDReference.py#L427)), so the CUDA
  side is the only gap.
- Floating gradient shared-memory budget: go2 floating (nv=18) fits in float
  (~73 KB). g1 floating (nv=35) would likely need selective spill (out of scope).
- Floating FD-sanity test is fixed-base only (needs SE(3) log for tangent-space
  finite differencing of the q columns).

---

## 6. Commits

GRiDCodeGenerator submodule (branch `modernizing-tests`):
- `f24ce27` Ungate floating-base integrator gradient (Euler).
- `7a08a0a` (prior) Root-cause + gate-off — **its root-cause claim is the one now
  in doubt; see §4.**

Parent GRiD (branch `modernizing-tests`):
- `f2d6826` Enable floating Euler integrator gradient in CUDA equivalence test
  (xfail) + submodule bump.
- `299bef8` (prior) Floating value path for all 5 integrators.
