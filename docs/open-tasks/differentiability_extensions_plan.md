# Differentiability extensions — beyond state (f_ext column + du×π cell)

**Status:** planning (read-only exploration 2026-05-30, branch `modernizing-tests`).
**Goal:** consolidate GRiD's differentiability roadmap into a single matrix and spec
the two not-yet-planned regions: (A) gradients **w.r.t. external forces `f_ext`**, and
(B) the **gradient-of-gradient w.r.t. inertial params π** (the `du × π` cell), for
sysID of sensitivities.

This doc is the umbrella for the differentiability surface. It **builds on**
two siblings and cross-references them rather than restating:
- `docs/open-tasks/d4_runtime_inertia_params_plan.md` — the **π column** (runtime
  inertia + the joint-torque regressor `Y = ∂tau/∂π`, §G.0-G.10 there).
- The in-flight **T4 external-forces** work (branch `external-forces`, F-batch). At
  the time of writing T4 has **not landed on this tree** (`grep f_ext
  GRiDCodeGenerator/algorithms/` → 0 hits; only the RBDReference forward path knows
  `f_ext`). The f_ext **convention** below is locked to that forward path (§A.0).

---

## 1. The differentiability matrix

Rows = output order. Columns = differentiation variable.
`x = [q; q̇]` (state), `u` = control (`tau` for FD / `q̈` for ID), `π` = per-link
10-vector inertial params, `f_ext` = external spatial force per link (local frame).

| output \ var            | state `x=[q;q̇]`                          | control `u`                              | params `π`                                   | external force `f_ext`                                  |
|-------------------------|-------------------------------------------|------------------------------------------|----------------------------------------------|--------------------------------------------------------|
| **value**               | `tau=ID(x,q̈)` / `q̈=FD(x,u)` **DONE**     | (same fwd kernels) **DONE**              | `tau=Y·π` (runtime fwd) **PLANNED-D.4**      | `ID(x,q̈,f_ext)` / `FD(x,u,f_ext)` **PLANNED-T4**       |
| **1st-order grad `du`** | `id_du` / `fd_du` **DONE**                | `∂tau/∂q̈=M` (CRBA) / `∂q̈/∂tau=M⁻¹` **DONE** | `∂tau/∂π=Y` ; `∂q̈/∂π=−M⁻¹Y` **PLANNED-D.4** | `∂tau/∂f_ext=−Jᵀ` ; `∂q̈/∂f_ext=M⁻¹Jᵀ`  **NEW (§A)**     |
| **2nd-order `SO`**      | `idsva_so` / `fdsva_so` **DONE**          | mixed `x×u` folded in SO/`M` **DONE**    | `∂(id_du)/∂π=∂Y/∂x` **NEW (§B)** ; `∂(fd_du)/∂π` **DEFERRED (§B.3)** | `∂(id_du)/∂f_ext=−∂Jᵀ/∂q` **NEW (§A.3)** ; `∂²/∂f_ext²=0` **OUT (linear)** |

Column-level notes:
- **state / control columns are fully DONE**: `inverse_dynamics_gradient` (id_du),
  `forward_dynamics_gradient` (fd_du), `idsva_so` (body+world), `fdsva_so`. Control
  first-order is the mass matrix and its inverse, already emitted by `crba` /
  `direct_minv`; control enters ID/FD linearly so higher control derivatives vanish
  or fold into the state SO kernels.
- **π column = D.4** (`Y`, `−M⁻¹Y`); see that doc §G. This doc only adds the **`du×π`
  second-order cell** (§B), which D.4 explicitly listed OUT-OF-SCOPE-for-v1 (its §G.6).
- **f_ext column = NEW here** (§A). Value lands with T4; the gradients are this doc.

OUT-OF-SCOPE cells (with reason):
- `∂(ee_pose)/∂π = 0` identically (kinematics carries no inertia; D.4 §G.6).
- `∂²(anything)/∂f_ext² = 0` — `f_ext` enters RNEA's force sweep **additively &
  linearly** (§A.0), so all second and higher `f_ext`-derivatives vanish. The only
  non-trivial f_ext second-order term is the **mixed** `∂(du)/∂f_ext = −∂Jᵀ/∂q`
  (§A.3), which is purely kinematic.
- `∂(fd_du)/∂π` (`du×π` for forward dynamics) — derivable but heavy; **DEFERRED**
  (§B.3).
- mixed `π×f_ext`: `∂(∂tau/∂f_ext)/∂π = ∂(−Jᵀ)/∂π = 0` (Jacobian is q-only,
  inertia-free) and `∂(∂q̈/∂f_ext)/∂π = ∂(M⁻¹Jᵀ)/∂π = −M⁻¹(∂M/∂π)M⁻¹Jᵀ` is a niche
  cross-term — **OUT-OF-SCOPE** (no use-case; falls out of D.4's `−M⁻¹Y` + this
  doc's `Jᵀ` by chain rule if ever needed).

---

## A. Gradients w.r.t. external forces `f_ext`

### A.0 Convention (locked to the RBDReference forward path / pin.rnea(...,fext))

The golden forward semantics already in the tree
(`RBDReference/RBDReference.py:1659-1685`, `apply_external_forces`, and the `rnea`
`f_ext=` arg at `:1820`/`:2199` for ABA):

```
f_out[i]  -=  inv(Xa_i.T) @ f_ext[i]          # RBDReference.py:1684
```

i.e. each link `i` carries a 6-vector external **wrench `f_ext[i]` expressed in link
`i`'s local frame**, and it is **SUBTRACTED** from the RNEA body force before the
back-propagation. `Xa_i` is the **root→i** world transform, so `inv(Xa_i.T)` maps the
local wrench into the propagation frame. This matches `pin.rnea(model,data,q,v,a,fext)`
(fext = `pin.StdVec_Force`, local joint frame, **subtracted** in pinocchio's bias) and
the GATO `iiwa14_fext.cuh` convention (local link frame, negative sign). **The sign
is `−`; the frame is LOCAL.** Lock all gradients below to this sign+frame; flag at
emit time if T4 lands with a different (e.g. world-frame, or additive) convention and
flip a single `SIGN`/`FRAME` constant.

Because `f_ext` enters **only** through that single additive `f_out -= … f_ext`, RNEA
(hence ID and, through `M⁻¹`, FD) is **affine in `f_ext`** with a `q`-only Jacobian.
This is the whole reason the f_ext column is cheap — exactly mirroring why the π
column is cheap (affine in π, D.4 §G.0).

### A.1 `∂tau/∂f_ext = −J(q)ᵀ`  (stacked body-Jacobian transpose; q-only)

A unit wrench on link `i` (local frame) contributes to joint `j`'s torque iff `j` is
on the path **root→i**, through the geometric mapping

```
∂tau_j / ∂f_ext[i]  =  −( Sⱼᵀ · X_{i→j}ᵀ )            for j ∈ path(root→i),  else 0
```

where `Sⱼ` is joint `j`'s motion subspace (6×dofⱼ) and `X_{i→j} = Π_{m on j→i} X_m`
is the composed spatial transform from `i` up to `j`. Stacking over all links gives
the block-lower-triangular **stacked body-Jacobian transpose**

```
∂tau/∂f_ext  =  −J(q)ᵀ ,   J = [J_1; …; J_N]  (6N × n_v),  J_i = body Jacobian of link i.
```

`J_i[:, vj] = X_{i→j}ᵀ-mapped Sⱼ` for `j` on root→i, zero otherwise — i.e. the
**same per-link spatial Jacobian** that `gen_end_effector_pose_gradient_inner`
already builds for the leaf end-effectors, generalized to **every** link and kept in
the **local link frame** (no `E(rpy)⁻¹` rpy map — we want the spatial Jacobian, not
the pose-tangent Jacobian).

**Reuse map:**
- The BFS-level chain-up of world transforms `s_Xworld` and the per-(ee, chain-joint,
  subspace-col) Jacobian fill in `_eepose_gradient_hessian.py:406-445`
  (`gen_end_effector_pose_gradient_inner`, the geometric-Jacobian rewrite). Generalize
  "ee ∈ leaf_nodes" → "link i ∈ all bodies" and drop the rpy-row `E⁻¹` map (steps 3-4
  there); keep step 2's `J_w = R_world·ang_local`, `J_v = J_w × (p_i − p_j)`.
- Equivalently, the **RNEA backward force sweep itself** (`_inverse_dynamics.py`
  `Xᵀ f` back-prop at `:312`, subspace projection `s_c[dof]=±s_vaf[…]` at `:336-337`)
  IS `Jᵀ·(force)`: feeding a unit local wrench at link `i` and running the backward
  sweep yields column `i` of `−Jᵀ`. So `∂tau/∂f_ext` can be emitted **either** as the
  explicit stacked Jacobian (eepose-grad machinery) **or** as `N` RNEA-backward
  applications with unit RHS — pick the explicit-Jacobian emit (denser, one pass).

### A.2 `∂q̈/∂f_ext = M⁻¹ J(q)ᵀ`  (operational-space / contact inverse-inertia map)

From `M q̈ + c = tau + Jᵀ f_ext` (f_ext enters FD through the same `−Jᵀ` term, sign
flips because it moves to the RHS of `M q̈ = tau − c + Jᵀf_ext`):

```
∂q̈/∂f_ext  =  M⁻¹ Jᵀ        (n_v × 6N)
```

This is exactly the **contact-space / operational-space inverse-inertia map** Λ⁻¹
building block (`J M⁻¹ Jᵀ` is the operational-space inertia inverse; this is the
half-product `M⁻¹Jᵀ`).

**Reuse map:**
- `s_Minv` from `direct_minv` (`_direct_minv.py`, `direct_minv_inner` → explicit M⁻¹,
  the same buffer fd_du and D.4's `−M⁻¹Y` reuse). One `n_v×n_v · n_v×6N` GEMM
  (`s_Minv · Jᵀ`) via the existing `grid_linalg` GEMM primitives.
- `Jᵀ` is the §A.1 output. So `∂q̈/∂f_ext` is literally `−s_Minv · (∂tau/∂f_ext)` —
  it reuses **both** new emits chained, no new math.

### A.3 `∂(id_du)/∂f_ext = −∂Jᵀ/∂q`  (q-derivative of the body Jacobian)

Since `∂tau/∂f_ext = −Jᵀ(q)` is **velocity- and accel-independent** (q-only):

```
∂/∂q̇ (∂tau/∂f_ext) = 0,      ∂/∂q (∂tau/∂f_ext) = −∂Jᵀ/∂q
```

`∂Jᵀ/∂q` is the **q-derivative of the stacked body Jacobian** — the same kinematic
2nd-derivative object the **end-effector pose Hessian** already computes.

**Reuse map:**
- `gen_end_effector_pose_*hessian*` chain (`_eepose_gradient_hessian.py:884-1018`, the
  "compacted hessian chain" / `s_d2eeTemp`) builds `dJ/dq` for the leaf ees via the
  GLASS indexed-batched 4×4 GEMM chain-up. Generalize to all links + local frame, same
  as §A.1 generalizes the gradient. This is the heaviest of the three f_ext emits and
  is the only one with a non-trivial second-order structure (it slots as the f_ext
  entry of the SO row).
- Mixed `∂(id_du)/∂f_ext` w.r.t. q̇ is identically 0 (no q̇ dependence) — only the
  q-block is emitted.

---

## B. Gradient-of-gradient w.r.t. params π  (the `du × π` cell — sysID of sensitivities)

### B.0 The load-bearing fact: `id_du` is **also** linear in π

D.4 §G.0 establishes `tau = Y(q,q̇,q̈)·π` (RNEA affine in each link's spatial
inertia). Differentiating that identity in the **state** commutes with the
**π-linearity**: the id_du blocks are themselves regressors.

```
∂tau/∂q  = (∂Y/∂q)·π ,      ∂tau/∂q̇ = (∂Y/∂q̇)·π
⇒  ∂(∂tau/∂q)/∂π = ∂Y/∂q   (n_v × n_v × 10n, INDEPENDENT of π)
   ∂(∂tau/∂q̇)/∂π = ∂Y/∂q̇  (likewise π-independent)
```

So the `du × π` cell for **inverse dynamics** is the **state-derivative of the
joint-torque regressor**, `∂Y/∂x`, and like `Y` itself it does **not** depend on the
param values — pure `(q,q̇,q̈)` kinematics/velocity products.

**Reuse map:**
- `id_du` already computes `dv/dq`, `dv/dq̇`, `da/dq`, `da/dq̇` (the partials staged in
  `_inverse_dynamics_gradient.py` temp layout: `offset_dv_dq`, `offset_dv_dqd`,
  `offset_da_dq`, `offset_da_dqd` at `:7-11`, filled in
  `gen_inverse_dynamics_gradient_inner` `:139-211`). The body regressor `Y_body,i` is
  assembled from `(v_i, a_i)` via `crm/crf/icrf` (D.4 §G.1). Therefore `∂Y_body,i/∂x`
  is assembled from `(dv_i/dx, da_i/dx)` through the **same** `crm/crf/icrf` operators
  (they are bilinear/linear in their motion arg, so the derivative just substitutes
  `dv,da` for `v,a`). The tree back-prop to joint columns is the same RNEA-backward
  sweep with a 6×10 RHS (D.4 §G.2), now differentiated — reuses id_du's `df/dq`
  back-prop ordering (`offset_df_dq` at `:11`).
- **Net:** `∂Y/∂x` reuses D.4's regressor emit + id_du's `dv/dq,da/dq` staging. No new
  spatial-algebra primitive (same conclusion as D.4 §G.1 for `Y`).

This is the headline `du × π` deliverable: it is **exact, analytic, π-independent**,
and is precisely what gradient-based sysID with **sensitivity / gradient matching**
(match `∂tau/∂x` across params, not just `tau`) and **bilevel / optimal-control sysID**
(differentiate an OC solution's KKT/sensitivity w.r.t. π) consume.

### B.1 Output shape & scope

- `∂Y/∂q`, `∂Y/∂q̇` : each `n_v × n_v × 10n` (for fixed q̈; if q̈ is also a variable,
  `∂Y/∂q̈ = ∂Y/∂a`-block is the body-regressor's own a-columns, trivially available).
- **In scope (NEW):** `∂(id_du)/∂π = ∂Y/∂x` for inverse dynamics (B.0).
- **Stretch:** a mass-matrix-regressor state-derivative `∂(∂M/∂q)/∂π` (CRBA is linear
  in π too, D.4 §G.6 stretch) — only if a use-case appears.

### B.3 `∂(fd_du)/∂π` — derivable but DEFERRED

For forward dynamics, `q̈ = M(π)⁻¹(tau − c(x,π))`, and `fd_du = −M⁻¹ (∂c/∂x)` (the
emitted `−M⁻¹·dc/du`, `_forward_dynamics_gradient.py:57-64`). Differentiating in π
hits **both** the explicit `M⁻¹(π)` and the regressor inside `dc/dx`:

```
∂(fd_du)/∂π  =  M⁻¹ [ (∂M/∂π) M⁻¹ (∂c/∂x) − ∂(∂c/∂x)/∂π ]
            =  M⁻¹ [ (∂M/∂π)·(−fd_du) − ∂Y_c/∂x ]      (rough form)
```

i.e. it needs `∂M/∂π` (the CRBA/mass regressor, D.4 §G.6 stretch) **and** `∂Y/∂x`
(B.0) **and** two `M⁻¹` applications. Derivable from pieces this doc + D.4 already
spec, but a heavier multi-GEMM kernel with a new `∂M/∂π` dependency. **Mark
derivable-but-DEFERRED**; the inverse-dynamics `∂Y/∂x` (B.0) covers the primary sysID
sensitivity-matching use-case without it.

---

## 2. Codegen hooks (new emits)

Mirror the existing `*_gradient` 7-function family pattern
(`_inverse_dynamics_gradient.py`) and register in `GRiDCodeGenerator.py:24-37`'s
import/register block (the same block D.4 §G.7 extends for the regressor).

**f_ext column (§A):**
- `*_dfext` — emit the stacked body-Jacobian transpose `−Jᵀ` (§A.1) by generalizing
  `gen_end_effector_pose_gradient_inner` to all links / local frame. Output `s_dtau_dfext`
  size `n_v × 6N`.
- `Minv·Jᵀ` — `∂q̈/∂f_ext` (§A.2): chain `s_Minv` (direct_minv) with the §A.1 `Jᵀ` via
  a `grid_linalg` GEMM. Output `s_dqdd_dfext` size `n_v × 6N`.
- `*_dfext_dq` — `−∂Jᵀ/∂q` (§A.3): generalize the eepose **Hessian** chain
  (`_eepose_gradient_hessian.py:884-1018`) to all links / local frame.
- **Signature convention:** add the f_ext output as a trailing caller-placed pointer
  (like `s_dc_du` at `_inverse_dynamics_gradient.py:997`); thread T4's `d_f_ext` only
  where the **value** is needed (the gradients `−Jᵀ`, `M⁻¹Jᵀ`, `−∂Jᵀ/∂q` are f_ext-VALUE
  independent — q-only — so they need q, not f_ext, as input).

**du × π cell (§B):**
- `∂Y/∂x` — the **regressor-state-derivative** emit: extend D.4's
  `gen_inverse_dynamics_regressor_inner` (D.4 §G.7) to also walk id_du's
  `dv/dq,dv/dq̇,da/dq,da/dq̇` staging and produce `∂Y/∂q`, `∂Y/∂q̇`. Reuses id_du temp
  layout (`_inverse_dynamics_gradient.py:7-11`) + the regressor back-prop. Output
  `s_dY_dx` size `2 · n_v · n_v · 10n` (q and q̇ blocks).

All new emits inherit the D.4 `RUNTIME_INERTIA` template wiring automatically via
`gen_load_update_XImats_helpers_function_call` (D.4 §3) — though `∂Y/∂x` and the
f_ext Jacobians don't read inertia at all (only `tau`'s value does), so they're
inertia-buffer-free like `Y`.

---

## 3. Validation oracles

| quantity                         | exact oracle                                                                 | else |
|----------------------------------|------------------------------------------------------------------------------|------|
| `∂tau/∂f_ext = −Jᵀ` (§A.1)       | `pin.computeJointJacobians` + `pin.getJointJacobian(LOCAL)` per link, stacked | —    |
| `∂q̈/∂f_ext = M⁻¹Jᵀ` (§A.2)      | `M⁻¹` (`pin.computeMinverse`) · the above `Jᵀ` (composed exact)              | FD vs T4 fwd: perturb `f_ext`, re-run T4 `forward_dynamics`, `(q̈(f+δ)−q̈(f))/δ` |
| `∂(id_du)/∂f_ext = −∂Jᵀ/∂q` (§A.3)| `pin.getJointJacobianTimeVariation`/`pin` kinematic-Hessian (frame Hessian)  | FD vs T4 fwd id_du across `f_ext` |
| `∂Y/∂x = ∂(id_du)/∂π` (§B.0)     | none direct (pin has no regressor-Jacobian) → **FD of `pin.computeJointTorqueRegressor`** across q, q̇ | FD vs D.4 runtime-param path: perturb π via `set_inertia_params`, re-run runtime `id_du`, `(id_du(π+δ)−id_du(π))/δ` |
| `∂(fd_du)/∂π` (§B.3)             | (deferred)                                                                    | FD vs D.4 runtime `fd_du` |

Notes on what's exact vs FD-only:
- **`Jᵀ` and `M⁻¹Jᵀ` are EXACT** against pinocchio's joint-Jacobian (LOCAL frame) and
  `computeMinverse` — these are first-class pin outputs. **NEW oracle methods needed**
  in `RBDReference/equivalents/pinocchio_backend.py`: `joint_jacobian(q, link)` /
  stacked `body_jacobian_stacked(q)` (via `pin.computeJointJacobians` +
  `getJointJacobian(LOCAL)`), reusing the existing `_to_pin_q` / link-id reindex
  (`pinocchio_backend.py:473-570`). None exists today (grep `Jacobian` →
  reindex-helper imports only).
- **`∂Jᵀ/∂q` and `∂Y/∂x` have NO closed-form pin oracle** → validate by **finite
  differences of an exact first-order pin oracle** (FD of `getJointJacobian` for §A.3;
  FD of `computeJointTorqueRegressor` for §B.0). These are exact-first-order-FD, much
  tighter than full forward FD.
- **Cross-checks against the sibling forward paths** are the integration tests:
  §A.* FD against the **T4 f_ext forward path**, §B.* FD against the **D.4
  runtime-param forward path** — both pin the gradient to the SAME convention/basis as
  the forward kernel (the f_ext sign+frame for A, the π basis for B; D.4 §G.5).
- **Numpy reference** (`reference_backend.py`): add a self-contained
  `body_jacobian_stacked` from the existing reference FK transforms (so f_ext
  validation needs no pinocchio), matching the dual-oracle layer. `∂Y/∂x` reuses the
  numpy regressor reference D.4 §G.4(2) adds, FD-differenced.

---

## 4. Use cases

- **f_ext gradients → differentiable contact / force-aware trajectory optimization.**
  `∂q̈/∂f_ext = M⁻¹Jᵀ` is the operational-space inverse-inertia block needed to
  differentiate contact-force trajectory-optimization (contact-implicit TO,
  force-at-contact decision variables) and to back-prop through a contact model.
  `−∂Jᵀ/∂q` supplies the second-order term for SQP/DDP Hessians of contact-rich OC.
- **`∂(id_du)/∂π = ∂Y/∂x` → sysID with sensitivity / gradient matching.** Beyond
  torque-matching (`tau = Yπ`), match the **sensitivities** `∂tau/∂x` across params —
  more identifiable / better-conditioned for excitation-poor data; and **bilevel /
  optimal-control sysID** (differentiate an OC/MPC solution w.r.t. π through its
  first-order optimality, which needs `∂(dynamics-Jacobian)/∂π`).

---

## 5. Sequencing

1. **f_ext gradients (§A): after T4 lands** (need the f_ext **convention** — sign,
   local-vs-world frame — locked to the shipped forward path; §A.0). The three emits
   are convention-agnostic in structure (single `SIGN`/`FRAME` constant), so the
   Jacobian machinery (generalize eepose grad/hessian to all-links/local) can be
   prototyped against the RBDReference forward path **now** and re-pinned when T4
   merges. Add the pin joint-Jacobian + numpy stacked-Jacobian oracles first.
2. **`∂Y/∂x` (§B): after D.4** (needs the regressor `Y` emit + the shared π basis +
   the `set_inertia_params` FD cross-check path; D.4 §G.5, §G.10). Extends D.4's
   `inverse_dynamics_regressor` emit with the id_du `dv/dq,da/dq` walk.
3. `∂(fd_du)/∂π` (§B.3): deferred backlog — needs the `∂M/∂π` mass-regressor (D.4 §G.6
   stretch) on top of §B.

**Cross-references:** π column + regressor `Y`, basis, identifiability →
`d4_runtime_inertia_params_plan.md` §G. f_ext forward value + convention → T4
`external-forces` (branch `external-forces`; RBDReference `apply_external_forces`
`RBDReference.py:1659-1685` is the current golden semantics until T4 emits CUDA).
