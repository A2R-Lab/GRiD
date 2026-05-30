# Tier autotune — propose-only follow-ups (T5)

Filed by T5 (PERF→SHARED rename + dynamic autotune). These are **proposals**, not
implemented in T5. The main agent decides whether/when to land them.

## 1. Feed autotune's best tier back into codegen as the per-robot default

**What.** Today codegen's `select_shared_tier_3way` assigns the *SHARED* (ex-PERF)
pick as the kernel's generation default (the body emitted when no
`-DGRID_DEFAULT_RESOURCE_TIER` override is passed, and the tier the
Python/JAX handles always launch). The autotune (T5, Part 2) now measures, per
`(robot, base, algo)`, which `(tier, threads)` pair is actually fastest at
batch N — and that winning tier is **not always SHARED**:

- For a no-smem-spill algo like `inverse_dynamics`, SHARED and MINIMAL share the
  same body and differ only in `__launch_bounds__`; the looser-bounds tier can
  win or lose purely on register pressure (see A.7 below). The empirically best
  tier may be MINIMAL.
- For robots where SHARED/LITE/MINIMAL collapse to one body, the tier choice is
  moot, but the *launch_bounds* still differs, so "best tier" still means
  something.

**Proposal.** Add an optional codegen input (e.g. a per-`(robot, base, algo)` map,
or a path to an `autotune_best_<host>.json`) that overrides the per-algo
generation-default tier with the autotuned winner. Concretely:
`GRiDCodeGenerator` would read the winner tier for each algo and emit that tier's
launch_bounds + spill body as the default specialization (the one
`GRID_DEFAULT_RESOURCE_TIER` resolves to when unset).

**Why deferred.** This is a **codegen behavior change** — it changes the bytes of
`grid.cuh` and the default the sealed Python/JAX surface launches. It is exactly
the kind of change that must not ride along inside a mechanical rename + a
measurement-tool addition. It also wants the *final serialized perf sweep* data
(deferred to the main agent) to choose winners from, not synthetic data.

**Risks / open questions.**
- Per-host: the winner is GPU-specific (sm_120/RTX 5090 here). The override map
  should be host-keyed, or codegen should refuse to bake a winner from a
  different GPU than the target.
- Determinism: codegen output would depend on a measured artifact. Need a
  pinned/committed `autotune_best` and a clean "no artifact → today's SHARED
  default" fallback so a fresh checkout still generates.
- The Python/JAX persona is documented as "always SHARED tier". If the default
  becomes MINIMAL for some algos, that doc + the "Why is JAX/Python locked to
  TIER_SHARED?" section need updating.

## 2. Deeper A.7 fix: LITE aliases SHARED launch_bounds for no-smem-spill algos

**Symptom (root-caused in T5).** `h1_2.fixed` `inverse_dynamics` (`id`) mis-tunes
at LITE. `id` has **no smem spill**, so SHARED and LITE emit the *same* body;
the only difference is `__launch_bounds__`: LITE uses `min(2×SUGGESTED, 768)`,
which is *looser* than SHARED's `MAX_PERF_LEVEL_THREADS`. A looser bound lets
ptxas budget for more threads/block and therefore allocate **fewer registers per
thread** → register starvation → LITE is *slower* than SHARED on this cell.

**T5's mitigation (implemented).** The autotune naturally never picks LITE for
this cell (it picks SHARED or MINIMAL on the joint tier×threads argmin), and a
unit assertion locks that property in (`tier ∈ {shared, minimal}` for
`h1_2.fixed id`). This fixes the *symptom* at the measurement layer.

**Proposal (NOT implemented — overlaps T2's perf work).** Fix the *cause* in
codegen: when an algorithm has **no smem spill at any tier** (SHARED rung ==
LITE rung == MINIMAL rung in `select_shared_tier_3way`), LITE should **alias
SHARED's launch_bounds** instead of using the generic `min(2×SUGGESTED, 768)`
formula. I.e. `tier_max_threads<TIER_LITE>()` for a no-spill algo should return
`MAX_PERF_LEVEL_THREADS`, not the looser mid-budget cap — there is no smem to
buy back by loosening the bound, so the looser bound is pure downside.

**Why deferred.** This changes `tier_max_threads` / the emitted launch_bounds for
LITE — a register/occupancy tuning change that lives squarely in **T2's perf
cleanup** scope. Landing it here would collide with T2 on the same emitted
machinery. Propose; let T2 own it.

**Note for whoever implements.** Generalize beyond `id`: the alias should apply to
*every* algo whose 3-way pick collapses with no workspace bytes at LITE (today
that's `inverse_dynamics`, `crba`, `end_effector_pose`, and any robot/algo cell
where the LITE rung == SHARED rung). The cleanest hook is in the
`tier_max_threads<>()` emission or a per-algo `LITE_LAUNCH_BOUNDS_ALIASES_SHARED`
constexpr keyed on whether `*_WORKSPACE_BYTES<T, TIER_LITE>() == 0`.
