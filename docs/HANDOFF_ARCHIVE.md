# HANDOFF archive — historical "Done" log

Completed-work log moved out of `HANDOFF.md` to keep the live tracker lean
(2026-05-31 consolidation). The current state + active backlog live in
`HANDOFF.md`; the per-merge history is in `git log`. This file preserves the
narrative rationale for older batches.

### Done (since this backlog was last refactored 2026-05-28)
- **2026-05-29 EVENING-2 batch (h1_2 MINIMAL bug verification + rpy-snap fix + polish/cleanup):**
  - **Bug 1 (h1_2 MINIMAL CRBA `M[0,13]≈0`) — STALE, RETRACTED.** The earlier
    HANDOFF entry (lines 1195+ in the previous revision) flagged this as a
    pre-existing tier-emit bug. An empirical verification run (compiled MINIMAL
    runner direct-probe on h1_2-floating-zero) shows the current codebase
    returns `M[0,13] = -3.856656` (matches expected ~-3.86) **deterministically
    across threads ∈ {32, 64, 128, 256, 384, 512}**, with M symmetric to ~1e-7.
    No memory-ordering hazard. No codegen edit needed. The BFS-parallel CRBA
    refactor (`809b145`) + per-jid chain-walk refactor (`9eb51a6`) +
    URDFParser FK + rpy-snap fixes closed the gap silently. **The h1_2-fixed
    `M[0,13]=0` value is analytically correct** (jid 0 = `left_hip_yaw`,
    jid 13 = `left_shoulder_pitch`, disjoint subtrees → zero cross-inertia);
    the old HANDOFF leaked the floating-base "expected -3.86" into the
    fixed-base narrative incorrectly.
  - **Bug 2 (h1_2 ee_pose ±π) — FIXED in URDFParser `ce01c53`** (perf-cleanup).
    Root cause: `sp.nsimplify(tolerance=1e-6)` in `Joint.py` leaves a 3.67e-6
    residual on URDF rpy=`"-1.5708"` (which means -π/2 truncated). The residual
    cascades through float32 kinematic chains and the GPU `atan2` in ee_pose
    rpy extraction flips a yaw component by exactly π (e.g. h1_2 L_thumb_distal
    yaw -4.71 vs CPU-ref -1.57). Fix: rpy-grid snap before nsimplify — snap to
    exact `N*π/2` for `|N| ≤ 4` when input is within 1e-5 of the grid point.
    Bounded N keeps the snap targeted. **Byte-identical for iiwa14/go2/g1**
    (their URDF rpy values are either exact 0 or full-precision π/2, already
    captured by existing nsimplify). **Snaps 6 joints on h1_2** (thumb proximal
    yaw/pitch L+R, R_middle_proximal, etc.). Pure URDFParser-side fix; no codegen
    edit. Equivalence validation in flight at HANDOFF time.
  - **Polish/cleanup A-batch — codegen `bd1bdd3`.** Dropped ~110 lines of pure
    dead Python (zero callers verified across `.py`/`.cu`/`.cuh`): the singular
    `select_shared_tier` (shadowed by the 3-way variant), `gen_add_debug_print_code_line`
    (singular; only the `_lines` plural is used), `_any_algo_uses_workspace_spill`
    (planned L2-persistence hook that landed elsewhere), `_gravity_shim_full_spill_count`
    (replaced by `_gravity_shim_use_full_spill`), `gen_idsva_so_body_frame_device`
    + `gen_idsva_so_body_frame_device_temp_mem_size` (no callers — body_frame is
    dispatched through `gen_idsva_so_device`), and the two
    `gen_end_effector_pose_gradient{,_hessian}_device_temp_mem_size` shells.
    Also removed the stale `# self.gen_idsva_so_body_frame_device(False) TODO`
    commented call. **Codegen output unchanged** (Python-only dead-code).
  - **REAL h1_2-floating MINIMAL equivalence failure (the one the morning
    C.1 MINIMAL run actually surfaced):** a **shape mismatch** — CUDA M is
    (57, 57) while the mimic-aware RBDReference/`pinocchio_backend` reference
    is (45, 45). h1_2 has 12 mimic joints (51 raw → 39 reduced) + 6 base =
    45. This is the D.2 CUDA codegen-mimic gap, planned at
    `docs/d2_codegen_mimic_plan.md` (4-phase, ~3 focused days). Affects every
    algo whose output dimension scales with NV (crba, minv, rnea, fd,
    ee_pose_gradient, ee_pose_hessian). Does NOT affect `ee_pose` (the rpy
    extraction output is `6 * N_ee`, independent of NV).
    **For the C.7 perf sweep:** timing is unaffected by the shape mismatch
    (the sweep measures kernel runtime, not equivalence). h1_2 floating timing
    will be reported on size-57 not size-45 matrices, same as the prior
    `tier_sweep_20260525_002438` baseline (apples-to-apples comparison).
  - **Sweep scoping decision** (resolved this session): **launch the full
    sweep, all 4 robots × {fixed, floating} × {PERF, LITE, MINIMAL}**. No
    h1_2 MINIMAL skip needed — the equivalence shape-mismatch is irrelevant
    for timing. Plan: `python test/benchmarks/run_multi_version.py
    --robots iiwa14 go2 g1 h1_2 --bases fixed floating --columns glass
    --tiers perf lite minimal --autotune-threads --output-dir
    test/benchmarks/results/perf_cleanup_<ts>`. Gated on (a) the rpy-snap
    equivalence validation clearing, (b) the user's personal review of D.2
    RBDReference (`0b1a89d` → `d0e552a`). **2026-05-30 UPDATE:** sweep ran
    to completion green; D.2 review verdict = "fine for now, backlog captures
    the rest" — gate is closed, perf-cleanup is cleared to merge into
    `modernizing-tests`.
- **2026-05-29 EVENING evening batch (8-agent + 2-direct landings):**
  - **D.2 FK orientation fix** — URDFParser `acbdab8` (perf-cleanup branch) +
    `8182770` (modernizing-tests branch). `Joint.set_type` now rotates `t_free`
    through the origin's rotation before adding `t_origin`. Invisible when
    origin rpy=0 (every iiwa/go2/g1 revolute, fr3_finger_joint1, etc.); fixes
    fr3_finger_joint2 (origin rpy=π around Z). fr3 world_T_finger_joint2 chain
    composition now matches pinocchio `oMi` to ~9e-13 (was 1.25e-01); iiwa14
    + go2 + g1 + h1_2 + baxter unchanged at machine precision. Parent
    `675249d` bumps URDFParser to perf-cleanup tip. **CORRECTION:** earlier
    HANDOFF claimed gen3 + fetch had PRE-EXISTING URDFParser composition bugs
    based on chain diffs ~7.9e-1 and ~1.0e0 vs pinocchio. The follow-up
    rotation-block audit (2026-05-29 PM) found this was a **validation
    harness artifact**, not a URDFParser bug: gen3 + fetch have URDF
    `continuous` joints which pinocchio's `buildModelFromUrdf` encodes as
    `JointModelRUBZ` (2-D `(cos, sin)` q encoding, so `pin.nq > grid.nq`).
    Filling `pin_q[idx_q] = theta` puts the raw angle in the cos slot and
    0 in the sin slot. The `pinocchio_backend.py` adapter via
    `expand_continuous_joint_positions_for_pin` (`conventions.py:51-73`)
    already handles RUBZ correctly. With proper pin_q construction, gen3
    max diff is 5.9e-11 and fetch is 2.1e-12 — both machine precision. **No
    URDFParser rotation block bug exists.** Original "side finding" left
    in the historical record for cross-reference; superseded by this audit.
  - **A.3 surgical floating XImats lever** — codegen `b140319`. Added
    `SKIP_FLOATING_BASE_X` template flag to `gen_load_update_XImats_helpers`;
    CRBA caller passes `skip=True` to elide the per-call recomputation of
    `s_XImats[0..35]` (the heavy floating-root quat→rotation expansion). CRBA
    never reads `s_XImats[0..35]` — Phase 1 BFS starts at level 1 and
    Phase 2's chain walk only dereferences `X[X_id != 0]`. iiwa14 + go2 + g1
    floating equivalence GREEN.
  - **B.4 forward_dynamics_device tier-aware** — codegen `8f39604` + parent
    `27c2f0b` runner threading. `forward_dynamics_device<T, RESOURCE_TIER>`
    now accepts `d_workspace`; at `TIER_LITE`/`TIER_MINIMAL` the FD inner
    arena (incl. the Minv-F band) routes to L2-pinned `d_workspace`, freeing
    ~120 KB smem on humanoid-scale robots. Mirrors `idsva_so_device` /
    `d2ee_device` / `id_du_device` tier pattern. Equivalence runner now
    calls `forward_dynamics_device<T, TIER_MINIMAL>` with the existing
    `hd_data->d_workspace` so the inline device path can fit h1_2-floating.
    Added `FD_DEVICE_INLINE_{SMEM,WORKSPACE}_BYTES<T,TIER>` macros.
    iiwa14 fixed+floating + g1 fixed equivalence GREEN. **Not end-to-end
    validated on h1_2-floating yet** (blocked by `idsva_so_body_frame_inner`
    pre-existing bug — see below).
  - **D.2 dynamics ripple (CRBA/MINV/RNEA-grad/FD-grad)** — RBDReference
    `8c351ad` (perf-cleanup). All four functions now mimic-aware with the
    canonical `+= mimic_scale *` accumulation pattern; `_qd[idx]` /
    `gravity-transport` reads scale by `α`; indexing uses
    `get_joint_index_v(ind)` throughout. `minv()` detects mimic robots and
    falls back to `np.linalg.inv(crba(q))` (pinocchio's reduced-model
    `M^{-1} = (G^T M_full G)^{-1}` is not equal to `G^T M_full^{-1} G` so
    the ABA-style recursion can't be `+=`-patched). Side fix in
    `pinocchio_backend.py`: reordered `rnea_grad` / `forward_dynamics_grad`
    so mimic-axis reduction runs BEFORE the q-Jacobian chain reducer.
    **Validated:** 5 target tests GREEN (`crba[fr3-fixed]`,
    `crba[h1_2-fixed]`, `minv[fr3-fixed]`, `rnea_grad[fr3-fixed]`,
    `rnea_grad[h1_2-fixed]`). Full RBDReference suite:
    707 PASSED / 50 failed / 68 skipped; all 50 failures are
    ABA-bound on fr3/h1_2 (next task — see D.2 OPEN ISSUE below). Zero
    non-mimic regressions.
  - **d2Xhom_owners defensive fix** — codegen `9c61e37`.
    `_global_hom_second_derivative_matrices` fixed-base branch now builds
    NJ-length owners list (was n_pos-length, IndexError when NJ != n_pos).
    Surfaces only with mimic-aware URDFParser on h1_2-fixed (51 joints / 39
    DoFs). Defensive: doesn't change current sweep behavior (parent's
    URDFParser pointer is mimic-unaware).
  - **D.2 ABA mimic** — RBDReference `bea0ac1` (perf-cleanup, parked).
    Algebraic-decomposition strategy: `qdd = M_reduced^{-1} * (tau -
    rnea(q, qd, 0))`. Sidesteps the per-body `(S, U, d)` recursion (whose
    α/α² scaling diverges from slot-accumulated `+=`) by reusing
    already-mimic-aware CRBA + RNEA. Bonus fixes: `has_invertible_mass_
    matrix` checks reduced M (pin's unreduced M is structurally singular
    for mimic models); `_pin_dIntegrate` size handling for integrator;
    h1_2 aba/rnea/minv tolerance overrides (cond(M_reduced) ~7e5 fixed /
    ~5e6 floating amplifies cross-library float64-eps to ~1e-3). Target
    tests GREEN: `aba[fr3-fixed]`, `aba[fr3-floating]`, `aba[h1_2-fixed]`,
    `aba[h1_2-floating]`, `forward_dynamics[*-fr3-*]`/`[*-h1_2-*]`,
    integrator state for fr3/h1_2.
  - **D.2 FD-grad mimic** — RBDReference `aa3eaa1` (perf-cleanup, parked).
    Pin backend bypass: pin's `computeABADerivatives` per-body recursion
    on unreduced model doesn't commute with `+= alpha *` fold (same root
    cause as ABA). Use implicit-function identity: `dqdd/dq = -M^{-1} ·
    drnea/dq`, all three components already mimic-aware. fr3 + h1_2
    fixed/floating GREEN; non-mimic 16/18 unchanged (2 rizon4 pre-existing
    skip).
  - **idsva_so_body_frame_inner t_index_map fix** — codegen `b573849`.
    Pre-existing bug: `t_index_map` sized `NV × NV` but indexed by `jid`
    (assumes `NJ == NV`). Now sized `NJ × NJ`; S/psid downstream stays jid-
    indexed. iiwa14 + g1 floating equivalence GREEN (regression check).
    h1_2-fixed regen smoke (with mimic-aware URDFParser, 51 joints / 39
    DoFs) now succeeds where it previously raised IndexError.
  - **d2Xhom_owners defensive fix** — codegen `9c61e37`.
    `_global_hom_second_derivative_matrices` fixed-base branch now builds
    NJ-length owners list (was n_pos-length, IndexError when NJ != n_pos).
    Defensive: doesn't change current sweep behavior.
  - **D.2 CUDA codegen mimic SCOPING DOC** — `docs/d2_codegen_mimic_plan.md`
    (parent, uncommitted at HANDOFF time). 4-phase plan ~3 focused days
    total: (P1, 1d) foundation helpers + ID + CRBA `+= α_i α_j` fold;
    (P2, 0.5d) `direct_minv` for mimic via CRBA-then-invert, `aba_kernel`
    for mimic via Minv·(u−c) algebraic decomposition; (P3, 0.75d)
    gradients (ID-du + FD-du + integrator); (P4, 1d) ee_pose + idsva_so
    + fdsva_so. Critical insight: per-body ABA / direct-Minv recursion
    fundamentally builds `M_full^{-1}`, not `M_red^{-1}`, so those algos
    can't be `+= α`-patched in codegen either; must fall back to CRBA +
    invert. Also flagged: `NUM_JOINTS` C++ macro is actually `nq`
    (reduced), not raw NJ — naming mismatch dangerous, P1 renames to
    `NUM_POS` + adds `NUM_LINKS` for un-reduced count.
  - **C.4 autotune `performance_threads`** — parent `32ac963`. New
    `--autotune-threads` flag (+ `--autotune-thread-grid`,
    `--autotune-N`). Coarse sweep + midpoint refinement, persists picks
    under `algo_picks` in benchmark JSON. ~14 sec wall per (robot, base).
    iiwa14-fixed example picks: aba 128 thr→14.26 µs, crba 224 thr→10.89
    µs, fdsva_so 192 thr→81.22 µs. Backward-compatible (no flag = old
    schema). Follow-ups: codegen-cap-aware grid clipping (currently wastes
    384/512 probes on small robots), floating + branched untested,
    single-call path not yet tuned.
  - **RUBZ audit on pinocchio_backend continuous joints** — CLOSED, NO
    BUG. `pinocchio_backend.py:_to_pin_q → _expand_project_q_to_pin_full
    → normalize_project_q_for_pin → expand_continuous_joint_positions_for
    _pin` (in `conventions.py:51-73`) already handles continuous-joint
    expansion to `(cos, sin)` correctly. gen3 (4 continuous joints) and
    fetch (5 cont) are the only RUBZ robots in the manifest; both match
    pinocchio at ~5e-11 / ~2e-12 via the adapter. The FK rotation agent's
    earlier failure mode was test-harness-specific.
  - **Open issues after this evening evening batch:**
    - **D.2 CUDA codegen mimic-awareness NOT done.** Bumping
      URDFParser+RBDReference into parent would now break fr3/h1_2 CUDA
      equivalence (Python correct, CUDA still naive). URDFParser
      perf-cleanup (`acbdab8`) is FK-only; mimic commits live on
      `modernizing-tests` (`511e398`, `8182770`). RBDReference
      perf-cleanup at `aa3eaa1` has D.2 morning + dynamics + ABA + FD-grad;
      parent still pins `990786a`. Net: D.2 parked. Plan in
      `docs/d2_codegen_mimic_plan.md` ready to pick up.
    - ✅ **D.2 idsva_so / fdsva_so mimic DONE** (Python side) —
      RBDReference `d0e552a`. Per-body unique internal slot indexing +
      R-matrix axis fold to project layout (elegant solve for "last-
      write-wins" issue with mimic-sharing v-slots). Pin backend bypass:
      3-axis reduce of unreduced pin SO tensor for `idsva_so`;
      `pin.aba`/`pin.computeMinverse` swap for mimic-aware
      `self.aba`/`self.minv` in fdsva_so. **All 52 SO equivalence tests
      GREEN** (idsva_so body+world + fdsva_so on fr3 + h1_2 fixed/floating).
      Full RBDReference suite: **765 passed / 26 failed / 34 skipped**
      (was 707/50/68 yesterday — net +58 passes). All 26 remaining
      failures verified pre-existing, none from D.2 work. Tolerance entry
      added for `(h1_2, second_order_fdsva)` (rtol=1e-4, atol=1e-1) —
      cond(M_reduced)~4.4e6 amplifies cross-impl 1e-7 to 1e-3 absolute
      through `Minv @ ... @ Minv` composition; relative residual stays
      ~1e-7. Same precedent as existing g1/h1_2 entries.
    - **D.2 ABA external-forces** — `f_ext` not threaded through mimic
      fast path. Niche feature, no failing tests. Defer.
    - **NEW finding: h1_2 MINIMAL-tier CRBA + ee_pose bugs surfaced by
      C.1 MINIMAL run** — **RETRACTED / SUPERSEDED** by the EVENING-2
      batch above. CRBA `M[0,13]≈0.001` claim was stale (current codebase
      returns -3.856656 deterministically; HANDOFF entry pre-dated the
      BFS-parallel CRBA refactor `809b145` + chain-walk refactor `9eb51a6`
      + URDFParser FK fix `acbdab8` + rpy-snap `ce01c53`). For h1_2-fixed
      `M[0,13]=0` is analytically correct (disjoint subtrees).
      `end_effector_pose ±π` was real but is fixed by the rpy-grid snap
      in URDFParser `ce01c53`. The actual remaining h1_2-floating MINIMAL
      equivalence failure is the D.2 codegen-mimic shape mismatch (57 vs
      45), tracked in `docs/d2_codegen_mimic_plan.md`.
  - **User-flagged review obligation**: user wants to personally review
    all 2026-05-29 D.2 RBDReference changes (`0b1a89d`, `8c351ad`,
    `bea0ac1`, `aa3eaa1`, + the idsva_so/fdsva_so commit when it lands)
    before E merge. Memory note set; do NOT auto-bump RBDReference into
    parent without flagging this.
- **2026-05-29 PM batch (parallel sub-agent sweep + investigation):**
  - **A.1 GPU d2ee analytic port — VERIFIED ALREADY DONE in codegen `2bf6d53`**
    (yesterday's analytic landing was sibling Python + GPU, not Python-then-GPU
    as HANDOFF previously suggested). Re-validated iiwa14 fixed + iiwa14
    floating (with `GRID_CUDA_FLOATING_ALGORITHMS=…hessian`) GREEN.
  - **A.3 BFS-parallel CRBA floating body** — codegen `809b145`: fused per-
    level forward+backward over `36*k` siblings at each BFS level; iiwa14
    stays byte-identical (1-wide BFS); go2 Phase 1 collapses ~24 syncs → ~6;
    g1 has partial-shared-parent siblings handled via atomic accumulation.
    iiwa14 floating + go2 floating + g1 floating equivalence GREEN. Estimated
    saves go2 ~1.8 µs/launch, g1 ~3.8 µs/launch (C.7 sweep will measure).
  - **A.4 pinocchio fdsva_so synthesis baseline** — parent `c6ab64c`: chain
    `ComputeRNEASecondOrderDerivatives` + `computeABADerivatives` + `Minv`
    (mirrors `pinocchio_backend.fdsva_so`). Wired into `timing_parser.py` +
    `run.py` + `generate_report.py`. iiwa14 fixed single 22.4 µs/iter,
    floating 67.4 µs/iter. Numerical cross-check vs GRiD output deferred
    (formula matches the Python backend so high confidence).
  - **B.3 idsva body `output_temp` → SCRATCH_IN_SMEM + B.1 idsva/fdsva
    cold-only de-alias** — VERIFIED ALREADY IMPLEMENTED in prior commits.
    Body kernel at `_idsva_so.py:2390` maps `s_temp_in_global=True` →
    `SCRATCH_IN_SMEM=false`; inner does the repoint at line 1372; the
    world inner already has `SCRATCH_IN_SMEM × COLD_IN_SMEM` (surgical cold
    trio Xdown/v_w/a_w). Docs landed in codegen `124be77` make the contract
    explicit; conformance audit in `docs/idsva_so_inner_refactor_notes.md`
    refreshed.
  - **C.1 LITE on g1+h1_2** — 3/4 GREEN (g1-fixed + g1-floating + h1_2-fixed).
    h1_2-floating SKIPPED on the pre-existing `forward_dynamics` smem cap
    (120 KB > 101 KB; matches B.4 backlog item, ancestor-scratch de-alias).
    NOT a regression.
  - **D.2 mimic-joint support — PARTIAL LANDING (parked on submodule branches,
    NOT bumped into parent yet).**
    - URDFParser `511e398` (on `modernizing-tests`): parse `<mimic>`,
      `Joint.get_num_dof()` returns 0 for mimic joints, `Robot._refresh_
      mimic_index_maps` builds dense q/v slot maps, `Robot.q_for_joint`
      exposes `mult * q[target] + offset`.
    - RBDReference `0b1a89d` (on `perf-cleanup`): kinematic Jacobian +
      analytic Hessian accumulate per-chain contributions and scale by
      mimic multiplier; pinocchio backend expands project→pin and reduces
      pin→project with mimic folding; new `test_pose_hessian_analytic_
      matches_fd[fr3-fixed]` smoke. Test suites GREEN for kinematics +
      RNEA + parse/metadata (173/173).
    - **Why parked (not yet bumped into parent):**
      1. Dynamics ripple is OPEN: `aba`, `crba`, `minv`, RNEA-grad, fd-grad
         still ASSIGN to the mimicked column instead of `+= mimic_scale * …`;
         the Robot.py nq/nv reduction now exposes this. Tests
         `test_aba_equivalence[fr3]`, `test_crba_equivalence[fr3,h1_2]`,
         `test_minv_equivalence[fr3]`, `test_integrator_pinocchio_
         equivalence[fr3]` now fail (they were passing by coincidence with
         naive sides on both).
      2. **CUDA codegen has no mimic awareness.** Bumping submodules into
         parent would make CUDA-equivalence-vs-RBDReference mismatch on
         fr3/h1_2 across multiple algos.
      3. Pre-existing FK orientation bug in `URDFParser/Joint.set_type`
         (sums `t_free + t_origin` without rotating `t_free` through origin
         rpy) surfaces only on `fr3_finger_joint2` (origin rpy = π around Z).
         Independent of mimic but in the way.
    - **fr3 analytic d²(pose)/dv² now matches FD oracle ~2e-11 and matches
      pinocchio's `getJointKinematicHessian(LOCAL_WORLD_ALIGNED)`**. h1_2
      (12 mimics, multipliers 1.0/1.6/2.4) kinematics + RNEA GREEN.
    - **Pickup recipe:** finish dynamics ripple in RBDReference (per the
      RNEA pattern, every `… = …` becomes `+= mimic_scale * …`); propagate
      mimic awareness into CUDA codegen; then bump submodules into parent.
      Or: gate fr3/h1_2 mimic-touching tests as xfail with link to this entry.
- **B.6.a + B.6.b (2026-05-29):** dropped the dead `use_thread_group`
  parameter from helper signatures + 1079 call sites across 16 codegen files
  (production always passes False; the conditional branches had been stripped
  earlier in b228756, leaving only signatures + plumbing); consolidated
  `gen_kernel_load_inputs` with `_single_timing` (same for save) into one fn
  each via optional `stride` kwarg (74 call sites rewritten). iiwa14
  fixed+floating + go2 floating CUDA equivalence GREEN. codegen `75089f8`,
  parent `1ff8d54`.
- **A.1 + A.3 scoping (2026-05-28 PM):** d2ee analytic derivation captured in
  `docs/d2ee_analytic_derivation.md` (fixed-base machine-precision validated in
  Python; floating intra-joint open gap); core-dynamics floating loss audit in
  `docs/a3_core_dynamics_floating_loss_audit.md` (top targets: ABA + CRBA
  floating root). C.3 partial: `-Wall` warnings probe on iiwa14-fixed runner
  found one class (42× `d_temp_spill` #177-D), silenced via `(void)` cast.
- **Cleanup batch** (2026-05-28): collapsed orchestrators to 3 layers
  (`_host` / `_kernel` / `_device`); per-tier L2 gates for fdsva_so;
  LITE/MINIMAL tier columns in `generate_report.py`; analytic d2ee in
  pinocchio_backend (joint targets + default offset); design docs + emitter
  README updated to the 3-layer convention; macro/name audit pass.
  iiwa14 fixed + floating CUDA equivalence GREEN.
- Floating `ee_pose_gradient` gap **CLOSED** (Step C, 2026-05-28: pin 7-16× → GRiD
  wins/parity on g1/h1_2 floating, pin only 1.18-1.34× on iiwa14/go2 floating).
- `ee_pose_hessian` (d2ee) Python d/dv rewrite (RBDReference + pinocchio_backend).
- d2ee GPU codegen rewrite (FD-on-Jacobian, d/dv tangent, pinocchio convention);
  iiwa14/go2/g1/h1_2 fixed + iiwa14/go2/g1 floating CUDA equivalence GREEN.
- d2ee timing wired into the bench (`timeGRiD_batch.cu` +
  `pinocchio/timePinocchio.cpp` via `computeJointKinematicHessians`) and measured
  end-to-end vs pinocchio.
- FIXED-path kinematics wired in CUDA equivalence runner (iiwa14 validated).
- `crba` regression fix (2026-05-27, 34edcfc) — recovered to baseline + beats
  pinocchio at batch on all robots.
- Pinocchio-as-fast-reference (equivalence-test oracle now defaults to pinocchio
  via `RBDReference/equivalents/pinocchio_backend.py`; first + second-order +
  d2ee all covered; pure-Python `RBDReference` is the fallback).

