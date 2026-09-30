# GRiD equivalence-testing strategy

This document explains **how** GRiD's correctness tests are built and, more
importantly, **why** — so that future contributors (human or agent) who add
tests for new kernels follow the same approach instead of writing checks that
pass while real bugs hide. If you are adding a kernel or a test, read this
first.

GRiD generates CUDA rigid-body-dynamics kernels. We trust them only as far as we
can show they match an independent reference. There are two equivalence layers,
and the order matters:

1. **Python reference ↔ Pinocchio** (`RBDReference/`), float64 vs
   float64. Pinocchio is the canonical, independently-implemented authority.
   This validates `RBDReference` (the hand-written Python reference).
2. **Generated CUDA ↔ Python reference** (`test/cuda_equivalents/`), float32 vs
   float64. This validates the codegen against the now-trusted Python reference.

Layer 2 is only meaningful if layer 1 is green, because layer 2 compares against
the Python reference, not Pinocchio. **Always fix the Python↔Pinocchio layer
first.**

---

## Principle 1 — Exercise high velocity and acceleration (varying magnitudes)

A correct algorithm and a subtly-wrong one **agree at rest and at low energy**.
Most RBD bugs live in velocity- and acceleration-coupling terms — the spatial
cross products (`crm`/`crf`), their gradients, and the SE(3)/SO(3) Jacobians —
whose contribution is proportional to `|qd|`, `|qd|²`, or `|qdd|`. At a near-zero
state these terms are ~0, so a test that only samples small states is
**structurally blind** to the most common class of bug.

Therefore the shared sampler (`RBDReference/equivalents/state_sampling.py`,
`build_dynamics_samples`) deliberately includes high-energy states:

- `zero`, `conservative` (|qd|≤1, |qdd|≤2) — sanity / low energy.
- `high_velocity` (|qd|≤10), `high_acceleration` (|qdd|≤50),
  `high_velocity_accel`, and randomized energetic samples.
- Floating-base **base velocity** (`qd[0:6]`) is scaled too, so free-flyer
  gyroscopic / linear↔angular coupling is exercised.

Why this works as a bug detector: in the Python↔Pinocchio layer the comparison
is **float64 vs float64**, so a correct algorithm matches to ~1e-12 *regardless*
of magnitude. Any mismatch that grows with energy is therefore **structural, not
numerical** — a real bug, not noise.

Real bugs this caught: an SE(3) right-Jacobian `Q`-block sign error (invisible
at tiny `dIntegrate` increments, O(1) at unit increments); and it was the
high-energy CUDA samples that surfaced the floating-base gradient race below.

**When adding a test:** feed it `build_dynamics_samples` (plus the CUDA corner
samples on the CUDA side). Do not invent a new low-energy-only sample set.

## Principle 2 — Sweep block thread counts, including a random non-multiple of 32

CUDA kernels are written as `for (i = threadIdx.x; i < N; i += blockDim.x)`
phases separated by `__syncthreads()`. A whole class of bug — a **missing
barrier between a write phase and a later read/accumulate phase**, or a
multi-thread `+=` into a shared destination — is **invisible at 32 threads**,
because a single warp executes in lockstep (warp-synchronous) and "accidentally"
behaves as if synchronized. The same kernel races at 2+ warps.

GRiD launches real workloads at `MAX_PERF_LEVEL_THREADS` (e.g. 448), which is many
warps. So a test that launches at a fixed 32 threads validates a configuration
**nobody runs in production** and passes while production silently corrupts
results.

Therefore the CUDA equivalence tests sweep block thread counts:

- `1` warp (32) — the warp-synchronous baseline.
- Multi-warp counts (e.g. 96, 448=`MAX_PERF_LEVEL_THREADS`).
- A **session-random count that is not a multiple of 32** (`_random_thread_count`
  in `cuda_harness.py`, the shared harness behind `test_cuda_executable_equivalence.py`), so a trailing partial warp is always
  present and, over many runs, many distinct counts are probed. The chosen value
  appears in the test id / error message for reproducibility; override with the
  relevant `GRID_CUDA_*_THREAD(S)*` env var to reproduce a specific failure.

The runners take the thread count as `argv[1]` (or a `-D…_TEST_THREADS` macro for
the second-order runners) so the sweep does not require recompiling per count
where avoidable.

Real bug this caught: the floating-base `inverse_dynamics_gradient` `da/du`
accumulation (a) zeroed then `+=`-accumulated with no `__syncthreads` between,
and (b) had 6 floating-root axes `+=` into the same destination via a
single-writer helper. Both were correct at 32 threads and produced
non-deterministic wrong gradients (and wrong integrator gradients) at 448. See
the retired `docs/HANDOFF.md` §3 ("RESOLVED 2026-05-22"; in git history).

**When adding a CUDA test:** route kernel launches through the runner's thread
count (`g_num_threads` / argv, or the `_TEST_THREADS` macro) — never hard-code
`<<<…, 32, …>>>`. Make the test exercise more than one warp count, including a
random non-multiple of 32.

## Principle 3 — Tolerances that scale with the array, not fixed floors

Two pitfalls when comparing arrays:

- **Float32 vs float64 (CUDA layer):** float32 carries ~1e-7 relative error.
  A per-element `atol` that is tiny will trip on an entry that is *structurally
  ~0* but carries `~rtol·scale` round-off — even though the kernel is correct —
  because `rtol·|expected|` ≈ 0 there. Fix: floor `atol` at `rtol · max|expected|`
  (the array's overall scale). "Small relative to the matrix" then counts as
  close, while a genuine error is `O(scale)` and still fails. See
  `_assert_close_scaled` (CUDA) and `assert_close` (Pinocchio comparator).
- **Singular / near-singular configurations:** forward dynamics / `Minv` /
  their derivatives are ill-defined when the mass matrix is near-singular (its
  inverse amplifies enormously, and float32 overflows to NaN). Skip such configs
  via the invertibility gate rather than loosening tolerance
  (`has_invertible_mass_matrix`, `min_singular_value`).

A specific instance: the **`aba` (Articulated-Body) kernel** can return
non-finite values in float32 on moderately ill-conditioned floating-base configs
(e.g. the quaternion corner samples, condition number ~6e3) where the robust
Minv-based `forward_dynamics` (CRBA + solve) still computes the correct result.
ABA is correct in float64 (it matches Pinocchio); the recursion is just more
float32-fragile than the CRBA+solve path. The test excuses a non-finite `aba`
**only** when the float64 reference is finite *and* the CUDA `forward_dynamics`
matched its reference for that sample (`_forward_dynamics_float32_matches`) — so
a non-finite ABA at a well-conditioned config, or one coinciding with a broken
FD path, still fails. Prefer `forward_dynamics` over `aba` for float32 on stiff
configs.

**Never loosen a tolerance to hide a real, magnitude-scaling discrepancy.** If
the error grows with energy or exceeds `O(rtol·scale)`, it is a bug — find the
root cause. Tolerance widening is only legitimate for documented float32 noise
or finite-difference references, and each override should carry a note saying
why.

## Principle 4 — One sampler feeds every layer

`build_dynamics_samples` is consumed by the Pinocchio tests, the CUDA executable
tests, and the CUDA integrator tests. Fixing conservatism in one place fixes
coverage everywhere. Do not fork sample generation per test.

---

## Checklist for adding a kernel + its equivalence test

1. Add a Python reference in `RBDReference` and validate it against Pinocchio
   (float64) over `build_dynamics_samples`, **including the high-energy samples**.
   Green here first.
2. Add the CUDA↔Python check in `test/cuda_equivalents/`:
   - Feed it `build_dynamics_samples` (+ CUDA corner samples).
   - Launch through the runner's configurable thread count; **sweep ≥1 warp,
     a multi-warp count, and a random non-multiple of 32**.
   - Compare with a scale-aware tolerance (`_assert_close_scaled`); gate
     singular configs instead of loosening.
3. If a discrepancy grows with velocity/acceleration → structural bug, not
   noise. If it appears only above one warp → a `__syncthreads`/accumulation
   race. Fix the kernel; do not mask it.
