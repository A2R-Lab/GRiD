# Centroidal + Energy Quick-Wins — implementation plan (R1–R3, refs R4/R5)

**Status:** planning (read-only exploration done 2026-05-30, branch `modernizing-tests`).
**Scope:** Implementer-ready plan for the "quick win" capability additions from
`docs/open-tasks/library_capability_roadmap.md` §Tier-1 (R1–R3 in depth; R4/R5 as
forward references). These reuse existing GRiD machinery (CRBA composite inertias,
RNEA sweeps, the EE geometric-Jacobian fill) and close the biggest *value-per-effort*
gaps vs Pinocchio / frax. GRiD's niche stays: **GPU-batched, codegen'd, differentiable
RBD primitives for control / learning / MPC**.

This doc is the sibling of `differentiability_extensions_plan.md` (the `du × {π,f_ext}`
matrix) and `d4_runtime_inertia_params_plan.md` (the π column). R5 gates on the f_ext
`M⁻¹Jᵀ` product from `differentiability_extensions_plan.md §A.2`.

---

## 0. The load-bearing facts that make these cheap

1. **CRBA already computes every composite spatial inertia GRiD needs for centroidal,
   KE and CoM.** Fixed-base IC init at `GRiDCodeGenerator/algorithms/_crba.py:232-237`
   (`s_temp[ICOffset..] = s_XImats[36*NJ..]`, i.e. `IC = I`), the BFS-level upward
   composite accumulation `IC[parent] += Xⱼᵀ ICⱼ Xⱼ` at `_crba.py:251-341` (fused
   forward `alphaⱼ = Xⱼᵀ ICⱼ` + backward `IC[par] += alphaⱼ Xⱼ`), and the floating
   root block `H[:6,:6] = Sᵀ IC[0] S` at `_crba.py:411-418`. **`IC[0]` after Phase 1
   is the whole-robot composite spatial inertia at the root frame** — the single
   object centroidal/CoM/KE all derive from. The numpy mirror is `RBDReference.crba`
   at `RBDReference.py:2350` (composite recursion `:2379-2425`).

2. **RNEA's inner already parameterizes the arg-pinning R1 wants.**
   `gen_inverse_dynamics_inner(self, compute_c, use_qdd_input)` at
   `_inverse_dynamics.py:31`: `use_qdd_input=False` runs the *whole RNEA with qdd≡0*
   (note `"optimized for qdd = 0"` at `:56`), and `compute_c=True` emits the final
   `S^T f` projection at `:322-339` into `s_c`. So:
   - `c(q,q̇) = RNEA(q, q̇, 0)` is **already exactly the `compute_c=True,
     use_qdd_input=False` emit** (the existing `s_c` is the nonlinear-effects bias).
   - `g(q) = RNEA(q, 0, 0)` is the same emit *with q̇ also zeroed* — i.e. a thin
     wrapper that passes a zero `s_qd`, or a `qd≡0` template specialization.
   The gravity column seeding `a_base = X·gravity_vec` lives at
   `_inverse_dynamics.py:103-125` (`s_vaf[6n + …] = X[...]·gravity`), the velocity-
   product bias `fx(v)·I·v` (`vxIv`) at `:281` (device `fx_times_v_peq`), numpy mirror
   `RBDReference.rnea` at `RBDReference.py:1814`, fpass `:1709`, bpass `:1774`.

3. **The EE geometric-Jacobian fill is the per-link spatial Jacobian R3/R4 need.**
   `gen_end_effector_pose_gradient_inner` at `_eepose_gradient_hessian.py:406`:
   Step-1 BFS world transforms (`s_Xworld`, `:480-508`), Step-3 per-(ee,chain-joint,
   S-col) column fill of `J_v`/`J_w` (`:519-577`) computing `axis_world = R_j·S_local`,
   `J_w = axis_world`, `J_v = axis_world × (p_ee − p_j)`. The per-job chain metadata
   bake is `_eepose_grad_chain_metadata` at `:364`. numpy mirror `RBDReference.py:1073`.

4. **`grid_plant` has a ready cost template** (`ee_pos_cost` value/grad/GN-hess) at
   `_plant.py:259-376` — `r = p−p_des`, `grad = Jᵀ W r`, `GN-hess = Jᵀ W J`. A
   CoM-tracking cost (R3) is this pattern with `p→p_com`, `J→J_com`; a momentum cost
   (R2) is the same with `h→A q̇`. Gating logic for plant costs is `_plant.py:544-548`.

5. **Registration:** each new family gets an `AlgoEntry` in
   `GRiDCodeGenerator/algo_registry.py:41-89` (e.g. a new `"Energy"` /
   `"Centroidal"` / `"Kinematics"` section), and a `gen_<name>()` dispatcher mirroring
   `gen_crba` (`_crba.py:638`): inner → device → kernel(timing+batch) → host(0/1/2).

---

## R1 — Energy, gravity, Coriolis/centrifugal (`do first` — near-free)

### Math
- `KE = ½ q̇ᵀ M q̇` (M from CRBA). Equivalently `KE = ½ Σᵢ vᵢᵀ Iᵢ vᵢ` (per-link
  spatial KE from the RNEA velocity sweep — avoids forming M; see GPU note).
- `PE = −Σᵢ mᵢ gᵀ p_comᵢ = M_total · gᵀ p_com` (g the gravity vector); reuses R3 CoM.
- `mechanicalEnergy = KE + PE`.
- `g(q) = RNEA(q, 0, 0)` (generalized gravity).
- `c(q,q̇) = RNEA(q, q̇, 0)` (nonLinearEffects = `C q̇ + g`).
- `C(q,q̇)` (explicit Coriolis matrix — **optional/heavier**, Tier-2): column `k` is
  `∂(c)/∂q̇ₖ`-style; practically `C` is read out of the `id_du` ∂τ/∂q̇ structure
  (see "Coriolis matrix" below).

### GRiD machinery reused (file:line)
- `g`, `c`: `gen_inverse_dynamics_inner(compute_c=True, use_qdd_input=…)`,
  `_inverse_dynamics.py:31`. `c` ≡ the **current** `use_qdd_input=False` emit
  (`s_c` at `:336-337`). `g` ≡ same emit with q̇ pinned to 0.
- `KE`: M from `crba_inner` (`_crba.py:38`) **or** the per-link `Iᵢ vᵢ` from the RNEA
  velocity/force sweep (`s_vaf` v-block `_inverse_dynamics.py:103-260`, `I·v` at
  `:265-267`). PE: R3 CoM + `M_total` (CRBA `IC[0]` mass = `IC[0][5,5]` region).

### Codegen hook
- **`c(q,q̇)`:** essentially *already shipped* inside ID. Promote it to a first-class
  `nonlinear_effects` (a.k.a. `bias`) family: a thin `gen_bias_*` that calls
  `gen_inverse_dynamics_inner_function_call(compute_c=True, use_qdd_input=False)`
  (`_inverse_dynamics.py:5`) and saves `s_c`. New device+kernel+host trio cloning
  `gen_inverse_dynamics_device/kernel/host` with `compute_c=True, use_qdd_input=False`.
- **`g(q)`:** same trio, additionally **arg-pin q̇≡0**. Cleanest: a `template<bool
  GRAVITY_ONLY>` on the inner that, when true, skips the `S·qd` v-seed (`:128-135`)
  and the `vxIv` bias (`:281`) — both vanish at q̇=0 — leaving only the gravity-column
  forward sweep. This makes `g` strictly cheaper than `c`, not just a wrapper.
- **`KE`/`PE`/`mechanicalEnergy`:** one new `gen_energy_*` family. Device computes M
  (or reuses the RNEA v-sweep) then a **single block reduction** `½ q̇ᵀMq̇`; PE adds
  `M_total gᵀ p_com`. Emit value-only (scalar out). Host trio + `AlgoEntry("energy")`.
- **`C(q,q̇)` (optional):** defer to Tier-2; emit by repacking `id_du`'s ∂τ/∂q̇ block
  (`_inverse_dynamics_gradient.py`) — mark NOT on the critical path.

### GPU-batch parallelism angle
- `g`/`c` are RNEA — already fully batched per-timestep block. No new parallelism.
- `KE` from per-link `Iᵢvᵢ`: each link's `vᵢᵀ(Iᵢvᵢ)` is an independent 6-dot; the
  sum is one warp/block reduction → avoids materializing M entirely (cheaper than
  `½q̇ᵀMq̇` when only the scalar is wanted). Ideal warp work; trivially batched.

### Pinocchio validation oracle
`pin.computeGeneralizedGravity(model,data,q)` → `g`; `pin.nonLinearEffects(model,
data,q,v)` → `c`; `pin.computeKineticEnergy` / `computePotentialEnergy` /
`computeMechanicalEnergy`; (optional) `pin.computeCoriolisMatrix` → `C`.

### RBDReference numpy reference to add
- `nonlinear_effects(q, qd, GRAVITY)` := `rnea(q, qd, np.zeros, GRAVITY)[0]` (the τ
  return of `RBDReference.rnea`, `RBDReference.py:1814`). `c` already obtainable;
  give it a named method for the oracle.
- `generalized_gravity(q, GRAVITY)` := `rnea(q, 0, 0, GRAVITY)`.
- `kinetic_energy(q, qd)` := `0.5 * qd @ crba(q) @ qd` (reuse `crba`, `:2350`).
- `potential_energy(q, GRAVITY)` := `M_total * g · com(q)` (reuse R3 `com`).
- (optional) `coriolis_matrix(q, qd)` for the Tier-2 `C` oracle.
- File placement per `rbdreference_split_plan.md`: lands in `_dynamics` mixin.

**Effort: S.** `c` is a rename/promote; `g` is one template bool; energy is M-reuse +
a reduction. This is the highest value-per-effort item — do first.

---

## R2 — Centroidal (the flagship): CMM `A(q)`, momentum `h`, rate `ḣ`/`Ȧq̇`, + derivatives

### Math
Let `IC[0]` = whole-robot composite spatial inertia at the root frame (CRBA Phase-1
output). The CoM position `c = p_com(q)` (R3). Define the shift transform from root to
a frame located at the CoM but axis-aligned with the world: `ᶜXₒ` = the 6×6 spatial
transform `[[I₃, 0],[ -[c]ₓ? ...]]` (standard Featherstone `Xtrans(c)` dual). Then:
- **Centroidal composite inertia** `I_G = ᶜXₒᵀ · IC[0] · ᶜXₒ` (project the composite
  inertia to the CoM frame). Pinocchio's `data.Ig`.
- **CMM** `A(q)` (6×NV): column `vᵢ` = `ᶜXₒᵀ · (Xₒ→bodyⱼ composite) · S_localⱼ`, i.e.
  the per-link composite-inertia × motion subspace, mapped to the CoM frame. Built
  from the **same per-body partial composites CRBA already forms on its upward sweep**
  (`IC` slabs in `s_temp`, `_crba.py:290`) shifted by `ᶜXₒ`.
- **Centroidal momentum** `h = A(q) q̇` (6-vector: angular about CoM ; linear =
  `M_total · ċ`).
- **Rate** `ḣ = A q̈ + Ȧ q̇`; the **bias** `Ȧ q̇` is `ḣ` evaluated at `q̈=0` — i.e.
  `ḣ|_{q̈=0}` = the spatial force at the CoM from an RNEA-style velocity sweep with
  zero accel. (Pinocchio computes `Ȧq̇` exactly this way: it is the centroidal-frame
  bias wrench, the analogue of `c` for the floating base 6-row.)
- **Derivatives** (`∂h/∂q`, `∂h/∂q̇=A`, `∂(Ȧq̇)/∂{q,q̇}`): GRiD's differentiator.
  Pinocchio's trick (`getCentroidalDynamicsDerivatives`) is that the centroidal
  derivatives are a **repack of the RNEA/`id_du` derivatives** projected to the CoM
  frame — reuse `_inverse_dynamics_gradient.py` (`id_du`), do NOT re-derive.

### GRiD machinery reused (file:line)
- Composite inertias: CRBA Phase-1, `_crba.py:232-341`; root composite `IC[0]`
  (`_crba.py:411-418` reads it). The per-body partial composites needed for `A`'s
  columns are the `IC[jid]` slabs at `s_temp[ICOffset + 36*jid]` (`_crba.py:290`)
  **before** they are folded into the parent — emit `A`'s columns during/just-after
  the same BFS sweep so the partial composites are still live.
- CoM-frame shift `ᶜXₒ`: needs `c = p_com` from R3.
- Floating root projection `Sᵀ IC S` pattern already at `_crba.py:411-418`.
- `Ȧq̇` bias: the RNEA velocity/force sweep `s_vaf` (`_inverse_dynamics.py:103-281`),
  read out at the root (the floating-base 6-row of `f`), analogous to `c`.
- Derivatives: `id_du` machinery `_inverse_dynamics_gradient.py` (the ∂f/∂{q,q̇}
  spatial-force derivatives), projected to CoM frame.

### Codegen hook
- New `gen_ccrba_*` family (inner+device+kernel+host), `AlgoEntry("ccrba",
  "Centroidal (A, h, ḣ)", "Centroidal")`. Inner reuses `crba_inner`'s Phase-1
  composite sweep, then a new pass: for each dof column, `A[:,vᵢ] = ᶜXₒᵀ · ICⱼ ·
  Sⱼ`; then `h = A·q̇` (one 6×NV gemv). Emit a `COMPUTE_HDOT` template bool that
  additionally runs the zero-accel velocity sweep for `Ȧq̇`.
- Derivatives: a follow-on `gen_ccrba_gradient_*` that consumes `id_du`'s outputs —
  land **after** R2 value (and after the f_ext/π SO work matures the repack path).
- Plant cost: a `momentum_cost` (`grid_plant`) — `r = h − h_des`, `Jₕ = A` (already
  the velocity-Jacobian of momentum), `grad = Aᵀ W r`, `GN-hess = Aᵀ W A`. Drop next
  to `ee_pos_cost` via the same gating pattern (`_plant.py:544-548`).

### GPU-batch parallelism angle
- `A` = **one extra 6×6-per-body projection** on top of CRBA's existing per-body
  composite work (the `IC[jid]·Sⱼ` then `ᶜXₒᵀ·(…)`), parallel over bodies → reduce
  is trivial because each column `vᵢ` is independent. `h = A q̇` is a single 6×NV
  gemv (warp-friendly). This is the **batched-centroidal niche**: neither frax (value
  only, no analytic deriv) nor Brax ships batched centroidal *derivatives*.

### Pinocchio validation oracle
`pin.ccrba(model,data,q,v)` → `A` (= `data.Ag`), `h` (= `data.hg`); `data.Ig` →
`I_G`. `pin.computeCentroidalMap(model,data,q)` → `A`. `pin.computeCentroidalMomentum`
→ `h`. `pin.computeCentroidalMomentumTimeVariation` → `ḣ`. For the gradient variant
`pin.computeCentroidalDynamicsDerivatives` / `getCentroidalDynamicsDerivatives`.

### RBDReference numpy reference to add
- `ccrba(q, qd)` → `(A, h)`: build `IC` via the existing `crba` composite recursion
  (`RBDReference.py:2379-2425`), shift each body composite by `ᶜXₒ`, stack columns.
- `centroidal_momentum(q, qd)` := `A @ qd`.
- `centroidal_momentum_time_variation(q, qd, qdd)` for `ḣ`; bias `Ȧq̇` := value at
  `qdd=0`.
- (follow-on) `centroidal_dynamics_derivatives` repacking `rnea_grad`
  (`RBDReference.py:2728`). Lands in a new `_centroidal` mixin per
  `rbdreference_split_plan.md`.

**Effort: M** (value); derivatives **M-L** as a follow-on. The centerpiece.

---

## R3 — CoM position + CoM Jacobian (`do before R2` — centroidal prereq)

### Math
- `p_com(q) = (Σᵢ mᵢ pᵢ) / M_total` (mass-weighted body-origin sum; in practice the
  CoM is read straight off the **root composite inertia** `IC[0]`: its first-moment
  block `m·c` divided by the mass `m` — i.e. `c = IC[0]` first-moment / mass, no
  separate accumulation needed once CRBA's composite sweep has run).
- `J_com(q)` (3×NV) `= (Σᵢ mᵢ J_v,ᵢ) / M_total` — mass-weighted average of per-body
  *linear* velocity Jacobians. Equivalently `J_com = (1/M_total)·[linear block of
  A(q)]` (the linear rows of the CMM are exactly `M_total·J_com`), tying R3 to R2.

### GRiD machinery reused (file:line)
- Mass / first-moment: CRBA root composite `IC[0]` (`_crba.py:411-418`; the
  `s_temp[ICOffset..]` slab holds `IC[0]` after Phase-1). `M_total` and `m·c` are
  fixed entries of that 6×6.
- `J_com`: the per-body linear Jacobian columns `J_v` already emitted by the EE
  geometric-Jacobian fill, `_eepose_gradient_hessian.py:519-577` (the `J_v[ee,vi,:]`
  rows). Reuse that column-fill but (a) target **every body** weighted by `mᵢ`, or
  (b) read the linear rows of `A` from R2 and divide by `M_total`.

### Codegen hook
- New `gen_com_*` family (inner+device+kernel+host), `AlgoEntry("com", "CoM + CoM
  Jacobian", "Kinematics")`. Inner: run CRBA Phase-1 composite sweep (or a trimmed
  mass-only sweep), read `c` from `IC[0]`; for `J_com`, reuse the `J_v` fill
  weighting each body column by `mᵢ/M_total` and accumulating.
- Plant cost: `com_cost` in `grid_plant`, a *direct clone* of `ee_pos_cost`
  (`_plant.py:259-376`) with `p→p_com`, `J_p→J_com`. Same value/grad/GN-hess
  (`grad = J_comᵀ W r`, `GN-hess = J_comᵀ W J_com`). Gate it next to `ee_pos_cost`
  (`_plant.py:544-548`) requiring the `com` algorithm.

### GPU-batch parallelism angle
- A **mass-weighted reduction over the same per-body Jacobian columns** the EE path
  already builds — per-body work is independent, the `Σ mᵢ J_v,ᵢ` is one block
  reduction. `p_com` from `IC[0]` is free once CRBA ran. Fully batched per-timestep.

### Pinocchio validation oracle
`pin.centerOfMass(model,data,q)` → `p_com` (`data.com[0]`); `pin.jacobianCenterOfMass(
model,data,q)` → `J_com` (`data.Jcom`).

### RBDReference numpy reference to add
- `com(q)` → `p_com` (mass-weighted via the same `Imats`/composite the `crba`
  reference already loads, `RBDReference.py:2370`).
- `jacobian_com(q)` → `J_com` (`Σ mᵢ J_v,ᵢ / M_total`, reusing the body-Jacobian path
  behind `RBDReference.py:1073`). Lands in the `_kinematics` mixin.

**Effort: S-M.** Implement **before R2** (R2's `ᶜXₒ` needs `p_com`) but **after R1**
(R1 is near-free).

---

## R4 / R5 — forward references (do not fully expand here)

### R4 — General frame Jacobian (any link, all 3 conventions)
The EE-anchored geometric Jacobian (`gen_end_effector_pose_gradient_inner`,
`_eepose_gradient_hessian.py:406`, with the chain-job bake `:364`) is already ~80% of
this: it builds `s_Xworld` for every joint and fills `J_v`/`J_w` per chain. Generalize
by (a) accepting an **arbitrary requested link** as the anchor (today `anchors = leaf
nodes`, `:433`/`:529`), and (b) emitting LOCAL / WORLD / LOCAL_WORLD_ALIGNED instead
of only the LWA `E(rpy)⁻¹` mapping (Step-4, `:579+`). Mostly indexing + a rotation
map. Oracle `pin.getFrameJacobian(..., pin.LOCAL/WORLD/LOCAL_WORLD_ALIGNED)`;
`frameJacobianTimeVariation` as a `J̇` follow-on. Effort S-M. **Full plan deferred.**

### R5 — Operational-space inertia `Λ = (J M⁻¹ Jᵀ)⁻¹` / OSC / Delassus
Gates on the f_ext `M⁻¹Jᵀ` product: `differentiability_extensions_plan.md §A.2`
emits `∂q̈/∂f_ext = M⁻¹Jᵀ` as `−s_Minv·(∂τ/∂f_ext)` (`§A.2`, `:120-140`) using
`direct_minv` (`_direct_minv.py`) and the §A.1 stacked-Jacobian transpose. Once that
lands, `Λ = (J·(M⁻¹Jᵀ))⁻¹` is a small dense 3×3/6×6 factor+solve per task on top —
warp-friendly. R4's general frame Jacobian supplies the `J`. Oracle: Pinocchio
Delassus / `J M⁻¹ Jᵀ` from `crba`+`getFrameJacobian`. Effort M, **gated on Tier-0
f_ext landing — see `differentiability_extensions_plan.md §A`.**

---

## Recommended implementation ORDER

1. **R1 — `c(q,q̇)` then `g(q)` then energy** (near-free; `c` is a promote of the
   existing `compute_c` ID emit, `g` is one template bool, energy is M-reuse + a
   reduction). *Land first.*
2. **R3 — CoM + `J_com`** (cheap; **prerequisite for R2's `ᶜXₒ`**; clones `ee_pos_cost`
   into a `com_cost`).
3. **R2 — CCRBA `A`, `h`, `ḣ`/`Ȧq̇`** (the centerpiece; reuses CRBA composite sweep +
   R3 CoM; adds a `momentum_cost`). Then **R2-derivatives** as a follow-on once the
   `id_du` repack path is mature.
4. *(forward)* **R4** (generalize the EE Jacobian) → unblocks **R5** (OSC/Λ), which
   additionally gates on the f_ext `M⁻¹Jᵀ` work.

**Each becomes a `grid_plant` cost** (`_plant.py`, gated at `:544-548`): R3 → CoM-
tracking cost (clone of `ee_pos_cost`); R2 → centroidal-momentum-tracking cost
(`Jₕ = A`, GN-hess `AᵀWA`); R1 energy → a scalar reward/Lyapunov term. These are the
control/learning/MPC hooks that justify the niche.
