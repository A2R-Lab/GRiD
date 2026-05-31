# d²(pose)/dv² analytic derivation — closed-form algorithm

**Status (2026-05-31):** Math complete and validated FLEET-WIDE. The analytic
d²(pose)/dv² is implemented as `RBDReference.end_effector_pose_hessian_analytic`
and matches BOTH the FD oracle (`end_effector_pose_hessian`) AND pinocchio's
analytic `getJointKinematicHessian(LOCAL_WORLD_ALIGNED)` to the FD-noise floor on
EVERY manifest robot × base (iiwa14/go2/g1/h1_2/fr3/rizon4/gen3/fetch/baxter,
fixed + floating, incl. mimic fr3/h1_2) — pin-parity widened accordingly (HANDOFF
A2 resolved). The CUDA codegen mirrors this analytic path and is GREEN vs the
pinocchio oracle. (The earlier "orientation-hessian bug" was in the now-retired
analytic d²/dq² path; this d²/dv² derivation has no such bug.) Original 2026-05-29
validation note: matched the FD oracle to ~1e-9 on iiwa14-fixed,
iiwa14-floating, and go2-floating across 3 non-degenerate samples each. The
single near-gimbal-lock sample (iiwa14 fixed sample 1, pitch ~87°) reduces from
1.8e-5 to 1.6e-7 when the FD oracle drops h from 1e-5 to 1e-6, confirming the
analytic is correct and the previous 1.8e-5 was FD noise from rpy losing
precision near pitch = π/2 (E^{-1} becomes ill-conditioned, but the analytic
result is exact).

Goal: replace the GPU FD-on-Jacobian d2ee (`2·nv + 1` gradient calls) with a
closed-form pass (~O(N·nv²) at most). This closes the 3–36× pin gap on big
floating-base robots (HANDOFF A.1).

## Convention recap

- Pose := `[xyz_world; rpy_world]`, RPY for `R = Rz(yaw) Ry(pitch) Rx(roll)`.
- d/dv (TANGENT) — per pinocchio LWA.
- `ω = E(rpy) · drpy/dt`, so `drpy/dv_i = E⁻¹ · J_w[:, i]`.
- For floating base: v-indices 0..2 are body-frame linear DOFs, 3..5 are
  body-frame angular DOFs (Pinocchio order [v_lin; ω]; see `Joint.py:210` for
  the 6×6 S matrix).
- `J_w[:, i]` is the WORLD angular velocity per unit v_i; `J_v[:, i]` is the
  WORLD linear velocity of the EE point per unit v_i.

## High-level algorithm

For each EE chain `[j_0, j_1, ..., j_k]`, parametrize the chain world
transform `M(v) = X_0 Δ_0(v_0) X_1 Δ_1(v_1) ... X_k Δ_k(v_k)` and compute the
exact first- and second-order Taylor coefficients of `M(v)` at `v = 0`. Here:

- `X_a` = the q-fixed local transform of chain joint `a` (already cached during
  the forward-kinematics pass).
- `Δ_a(v_a)` = the per-joint perturbation, with `Δ_a(0) = I`. For revolute and
  prismatic joints, `Δ_a` is a single matrix exponential along one body-frame
  axis. For the SE(3) free-flyer at jid=0, `Δ_a(v_lin, ω) = SE(3)_exp(v_lin, ω)`
  in body-frame coordinates.

The chain composition has clean closed-form first and second derivatives:

```
∂M/∂v_i = L_a · A_i^local · R_a       (i is DOF c of joint a; A_i^local = ∂Δ_a/∂v_i at v=0)

∂²M/∂v_i ∂v_j = L_a · B_{c_i,c_j}^local · R_a            if i, j in SAME joint a
              = L_{a} · A_i^local · P_{a→b} · A_j^local · R_b   if i in joint a, j in joint b > a
```

where:
- `L_a = X_0 X_1 ... X_a` (prefix to and including joint a, with all Δ=I)
- `R_a = X_{a+1} X_{a+2} ... X_k` (suffix after joint a)
- `P_{a→b} = X_{a+1} ... X_b = L_a^{-1} L_b` (between-product)
- `B_{c, c'}^local = ∂²Δ_a / (∂v_c ∂v_{c'})` at v=0 — the joint's intrinsic
  Lie-group second-order term.

## Per-joint Δ and its first/second derivatives at v=0

Let `e_x = (1,0,0)`, etc. `[a]_×` denotes the 3×3 skew-symmetric matrix.

**Revolute (single DOF, body axis a):**
```
Δ_rev(v) = [[ exp_SO(3)([a]_x · v),  0 ], [ 0, 1 ]]
∂Δ/∂v|_0  = [[ [a]_x,  0 ], [ 0, 0 ]]
∂²Δ/∂v²|_0 = [[ [a]_x · [a]_x,  0 ], [ 0, 0 ]]
```

**Prismatic (single DOF, body axis a):**
```
Δ_pris(v) = [[ I, a · v ], [ 0, 1 ]]
∂Δ/∂v|_0  = [[ 0, a ], [ 0, 0 ]]
∂²Δ/∂v²|_0 = 0
```

**Floating-base (6 DOFs, body frame):** Let v = (v_lin, ω) ∈ R³ × R³.
```
Δ_fb(v_lin, ω) = [[ exp_SO(3)([ω]_x),  V(ω) · v_lin ], [ 0, 1 ]]
```
where `V(ω) = I + ½[ω]_x + (1/6)[ω]_x² + ...` (the SE(3)-exp "V" coupling).

First derivatives at v=0:
- For each linear DOF `a ∈ {0..2}` (body axis `e_a`):
  `∂Δ/∂v_lin_a = [[0, e_a], [0, 0]]`
- For each angular DOF `b ∈ {3..5}` (body axis `e_{b-3}`):
  `∂Δ/∂ω_b = [[ [e_{b-3}]_x, 0 ], [ 0, 0 ]]`

Second derivatives at v=0 (the key intra-joint piece for floating base):

```
lin-lin (a, a' ∈ {0..2}, a ≤ a'):    ∂²Δ/(∂v_lin_a ∂v_lin_{a'}) = 0
                                      (translations in body frame commute)

ang-ang (b, b' ∈ {3..5}, any):       ∂²Δ/(∂ω_b ∂ω_{b'}) =
                                       [[ ½ ([e_{b-3}]_x [e_{b'-3}]_x
                                            + [e_{b'-3}]_x [e_{b-3}]_x),
                                          0 ],
                                        [ 0, 0 ]]
                                      (symmetric anticommutator from SO(3) exp;
                                       reduces to [a]_x² for b = b')

lin-ang (a ∈ {0..2}, b ∈ {3..5}):    ∂²Δ/(∂v_lin_a ∂ω_b) =
                                       [[ 0, ½ (e_{b-3} × e_a) ],
                                        [ 0, 0 ]]
                                      (from V(ω)·v_lin: ∂V/∂ω_b|_0 = ½ [e_{b-3}]_x)
```

These come from Taylor-expanding the SE(3) exp:
```
exp(α A + β B) = I + (αA + βB) + ½(αA + βB)² + O(3)
             = I + αA + βB + ½α²A² + ½β²B² + ½αβ(AB+BA) + O(3)
```
so `∂²/(∂α ∂β)|_0 exp(αA + βB) = ½(AB + BA)` (symmetric, NO BCH correction since
it's a single `exp` of a sum, not a product of `exp`s).

## Extracting xyz Hessian

```
p_ee(v) = (M(v) · ee_offset)[:3]
H_xyz[:, i, j] = (∂²M/(∂v_i ∂v_j) · ee_offset)[:3]
```

where `ee_offset` is the constant 4-vector `[x_off, y_off, z_off, 1]` of the
EE point in the last chain joint's frame. For fixed-joint EEs, the caller
pre-applies `X_fixed`: pass the chain ending at the fixed joint's parent `pid`,
`X_ee = Xw[pid] · X_fixed` (for rpy extraction), and
`ee_offset_new = X_fixed · user_offset` (for the position derivative).

## Extracting rpy Hessian

Define `R(v) = M(v)[:3, :3]` and compute its first/second Taylor coefficients
at v=0. Then build the world-angular Jacobian and its kinematic Hessian:

```
J_w[:, i]      = skew⁻¹( ∂R/∂v_i · R(0)^T )
H_w[:, i, j]   = ∂J_w[:, i] / ∂v_j
               = skew⁻¹( ∂²R/(∂v_i ∂v_j) · R(0)^T - [J_w[:, i]]_x · [J_w[:, j]]_x )
```

The H_w formula comes from differentiating `∂R/∂v_i = [J_w_i]_x · R(0)` and
using `∂R/∂v_j = [J_w_j]_x · R(0)`:
```
∂²R/(∂v_i ∂v_j) = [∂J_w_i/∂v_j]_x · R(0) + [J_w_i]_x · ∂R/∂v_j
                = [∂J_w_i/∂v_j]_x · R(0) + [J_w_i]_x · [J_w_j]_x · R(0)
⇒ [∂J_w_i/∂v_j]_x = ∂²R/(∂v_i ∂v_j) · R(0)^T - [J_w_i]_x · [J_w_j]_x
```

(Note: when the EE has a fixed-joint offset, R(0) here is the *chain* rotation
`L[k][:3, :3]` = world rotation of the last chain joint, NOT `X_ee[:3, :3]`,
because the rigid offset doesn't contribute to angular velocity. The rpy
extraction for E(rpy) uses R_ee0 = X_ee[:3, :3] (the EE-frame rotation) since
rpy is a frame-dependent angle convention.)

rpy chain rule:
```
∂rpy/∂v_i = E⁻¹(rpy) · J_w[:, i]
∂²rpy/(∂v_i ∂v_j) = (∂E⁻¹/∂v_j) · J_w[:, i] + E⁻¹ · H_w[:, i, j]
                  = -E⁻¹ · (Σ_k (∂E/∂rpy_k) (E⁻¹ J_w[:, j])_k) · E⁻¹ · J_w[:, i]
                    + E⁻¹ · H_w[:, i, j]
```

with the closed forms:
```
E(roll, pitch, yaw) = [[cy·cp, -sy, 0],
                       [sy·cp,  cy, 0],
                       [-sp,    0,   1]]

∂E/∂roll  = 0           (E does not depend on roll)
∂E/∂pitch = [[-cy·sp, 0, 0],
             [-sy·sp, 0, 0],
             [-cp,    0, 0]]
∂E/∂yaw   = [[-sy·cp, -cy, 0],
             [ cy·cp, -sy, 0],
             [ 0,      0,  0]]
```

The rpy formula is not symmetric in (i, j) by construction (the chain rule
expansion picks one direction), so we symmetrize at the end:
`H_rpy[:, i, j] ← ½(H_rpy[:, i, j] + H_rpy[:, j, i])`.

## Why the previous single-DOF-per-joint formula failed

The previous attempt (see git history of this doc on 2026-05-28) used a
chain-order proximal/distal classification: for a pair (a, b) with both DOFs in
the chain to ee, `proximal = min(a, b)`, `distal = max(a, b)`, then
`H_xyz[:, a, b] = ω_proximal × J_v[:, distal]`. This works for any pair where
the DOFs come from DIFFERENT joints (the proximal joint's rotation transports
the distal joint's spatial Jacobian column). It fails for pairs in the SAME
multi-DOF joint (the floating base at jid=0), specifically:

- **lin-ang intra-base** (e.g., DOFs 0 and 3): the chain formula gives 0 because
  the proximal-by-min-index is linear (`ω = 0`), but the truth is
  `½ ω_b × u_a` from the `V(ω)·v_lin` coupling in the SE(3) exp.
- **ang-ang intra-base**: the chain formula gives `ω_proximal × (ω_distal × X)`,
  but the truth is `½(ω_a × (ω_b × X) + ω_b × (ω_a × X))`. The difference is
  `½(ω_a × ω_b) × X` (Jacobi identity).

The current algorithm sidesteps this by working at the chain composition level,
where each joint contributes its INTRINSIC Lie-group second-order term via
`B_{c, c'}^local` (the SE(3) anticommutator/V-coupling), and cross-joint pairs
come naturally from the matrix-product chain rule. The same code path handles
all of cases A (separate joints), B (same multi-DOF joint), and C (proximal
multi-DOF + distal single-DOF). 

## Complexity

For a chain of length k with the EE chain having `n_chain` joints and m total
DOFs in the chain (m = nv typically):

- Precompute L, R_post, L_inv arrays: O(k · 64) flops
- A_i_local for each DOF: O(m)
- B_{c, c'}^local for each intra-joint pair (only base for floating-base
  robots): O(36) per pair, O(36 · 6 · 6) at most for the base
- d2M loop: O(m² · matmul_cost) — dominant; ~4×4 matmul per pair, so O(m² · 64)
- Skew-inv and rpy chain rule: O(m² · 9)

Total: O(m² · 64 + k · 64) per ee. For m=18 (go2 floating), that's ~20k flops
per ee — easily under a microsecond on CPU and well-suited to GPU codegen.

## GPU port guidance

The same shared FK pass (`Xw[j]`) used by `end_effector_pose_gradient` already
exists in the GPU codegen (`_eepose_gradient_hessian.py`). The new analytic
Hessian only needs:

1. Per-joint `A_local` (already implicit in the gradient code's S_local).
2. Per-joint `B_local` for multi-DOF joints (a small 6×6 tensor of 4×4
   matrices; for floating-base this is the SE(3) anticommutator table above).
3. Chain prefix/suffix products `L_a, R_post_a` (one register tile per chain).
4. Two-level loop over DOF pairs computing `d2M = L_a · (A_i or B) · (P · A_j)
   · R_post` and `(d2M · ee_offset)[:3]` for xyz, then `d2M[:3, :3]` for rpy.
5. The skew_inv + Einv/dE chain rule (closed form).

The whole Hessian is O(nv² · constant) per EE — comparable to existing kernels
like `IDSVA_so`.

## Useful references

- Featherstone, *Rigid Body Dynamics Algorithms* (Ch. 7–8) — spatial cross
  product, ad operator.
- Pinocchio `computeJointKinematicHessians` / `getJointKinematicHessian` —
  the canonical analytic Hessian; its LWA output is the joint kinematic Hessian
  tensor, related to but not exactly d²pose/dv² (it does not include the rpy
  chain rule). Conversion: linear rows become xyz rows after the EE-offset
  shift, and the angular rows feed through the rpy chain rule above.
- Sola, Deray & Atchuthan, *A micro Lie theory for state estimation in
  robotics* (2018) — closed forms for V(ω), J_r(ω) used in the SE(3) exp.
- `RBDReference.end_effector_pose_gradient` — the shared FK pass + per-DOF
  chain walk that this Hessian piggybacks on.
- `RBDReference.end_effector_pose_hessian` — the FD oracle this analytic was
  validated against.

## Test harness

`/tmp/test_d2ee_analytic.py` — standalone validation against the FD oracle on
iiwa14 fixed, iiwa14 floating, and go2 floating. Auto-retries with smaller FD
step (1e-6) when the default 1e-5 step lands near gimbal lock.
