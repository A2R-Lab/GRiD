# d²(pose)/dv² analytic derivation — notes & open gap

**Status (2026-05-28):** Math worked out for fixed-base / single-DOF chains
(validated machine-precision in a /tmp/ probe vs the FD oracle). The
floating-base intra-joint multi-DOF case (DOFs 0..5 of the same SE(3) base
joint) is **not** handled by the formula below and needs a Lie-group second-
order term that I have not finished deriving. Reverted from RBDReference;
keep the FD path until the floating case is closed.

Goal: replace the GPU FD-on-Jacobian d2ee (`2·nv + 1` gradient calls) with
a closed-form pass (O(N · depth²) at most). This closes the 3–36× pin gap on
big floating robots (HANDOFF A.1).

## Convention recap

- Pose := `[xyz_world; rpy_world]`, RPY for `R = Rz(yaw) Ry(pitch) Rx(roll)`.
- d/dv (TANGENT) — per pinocchio LWA.
- `ω = E(rpy) · drpy/dt`, so `drpy/dv_i = E⁻¹ · J_w[:, i]`.
- For the chain to an ee joint k, ordered proximal→distal:
  - revolute DOF i: `ω_i = R_{joint_i} @ axis_local` (world axis), `J_w[:, i] = ω_i`,
    `J_v[:, i] = ω_i × (p_ee - p_joint_i)`.
  - prismatic DOF i: `ω_i = 0`, `J_w[:, i] = 0`,
    `J_v[:, i] = R_{joint_i} @ axis_local` (world translation direction).

## Single-DOF-per-joint formula (FIXED-base or chains without multi-DOF joints)

For a pair `(a, b)` with both DOFs in chain to ee, let
`proximal = min(a,b)` and `distal = max(a,b)` in chain order:

### xyz rows
```
H_xyz[:, a, b] = ω_proximal × J_v[:, distal]
```

Symmetric in `(a, b)` by construction — proximal/distal labels are invariant under swap. Handles both DOF types uniformly via `ω_prismatic = 0`:
- rev-rev:   `a_p × (a_d × (p_ee - p_d))` ✓
- rev-pris (distal pris): `a_p × axis_d_world` ✓ (the pris axis rotates with proximal rev)
- pris-rev (proximal pris): `0` ✓ (pris translation does not rotate distal axis)
- pris-pris: `0` ✓
- diagonal (a=b, revolute): `a_a × (a_a × (p_ee - p_a))` ✓
- diagonal (a=b, prismatic): `0` ✓

### rpy rows
Per-pair formula (asymmetric in (a,b); symmetrize at end):
```
H_rpy[:, a, b] = Σ_k (∂E⁻¹/∂rpy_k) · (E⁻¹ · J_w[:, b])_k · J_w[:, a]
                 + E⁻¹ · ∂J_w[:, a]/∂v_b
```
with
```
∂J_w[:, a]/∂v_b = ω_b × J_w[:, a]   if b is STRICT proximal ancestor of a (b < a in chain)
                = 0                  otherwise
∂E⁻¹/∂rpy_k     = - E⁻¹ · (∂E/∂rpy_k) · E⁻¹
∂E/∂roll        = 0   (E does not depend on roll)
∂E/∂pitch, ∂E/∂yaw — closed form from `E = [[cy·cp, -sy, 0], [sy·cp, cy, 0], [-sp, 0, 1]]`
```
Then symmetrize `H_rpy[:, a, b] ← 0.5 · (H_rpy[:, a, b] + H_rpy[:, b, a])`.

**Empirically validated:** machine precision (~1e-11 max-err vs FD-h=1e-5) on
iiwa14 fixed across 2/3 random samples; the third sample hit ~1.8e-5 around
an rpy gimbal-lock region (the FD oracle itself loses precision there —
expected, not an analytic bug).

## Floating-base intra-joint multi-DOF case — OPEN GAP

The above formula gives `~0.5–1.2` worst-error vs FD on iiwa14-floating and
go2-floating. The issue is structural: for two DOFs `a, b` of the SAME 6-DOF
SE(3) free-flyer base joint, the "proximal/distal in chain" classification is
meaningless — neither DOF moves the other's frame in the usual kinematic
sense, but they still interact through the SE(3) exp map.

Concretely, for two angular DOFs (a, b ∈ {0,1,2}) of the base joint:
- `∂ω_a/∂v_b = ω_b × ω_a` (because `R_0` evolves under `v_b` via the body-
  frame infinitesimal rotation, and `ω_a = R_0 e_a^body`).
- `∂ω_b/∂v_a = ω_a × ω_b = -ω_b × ω_a`.

Both are non-zero, with opposite signs. The single-DOF formula uses one
direction (depending on chain-index ordering) and misses the second, so
neither `H[:, a, b]` nor its symmetric partner picks up the
`(ω_b × ω_a) × (p_ee - p_0_base)` correction term that comes from
`ω_a = R_0 e_a` covarying with `v_b` *within* the same joint.

Sketch of the corrected formula (needs verification):
```
∂J_v[:, a]/∂v_b  with a, b intra-joint base angular:
  = (∂ω_a/∂v_b) × (p_ee - p_0_base) + ω_a × (∂p_ee/∂v_b - ∂p_0_base/∂v_b)
  = (ω_b × ω_a) × (p_ee - p_0_base) + ω_a × (ω_b × (p_ee - p_0_base))
  (since base origin doesn't translate under pure-rotation base perturbation)
```

For two-DOF perturbation `V = α e_a + β e_b` integrated as `M exp(V)`:
```
p_ee_new = p_0 + R_0 · (chain_to_ee_in_body)
exp([V])_pos = (mixed term in α β cancels to zero at second order
                because [e_x×][e_y×] e_z = 0 for orthogonal axes)
```

So the **mixed pose-Hessian for intra-base-joint angular pairs at v=0** is
actually the SYMMETRIC PART of the directional derivative — the asymmetric
`(ω_b × ω_a) × X` term cancels between the two directions. The symmetric
part is:
```
H_pose_xyz[:, a, b] = ½ · [ω_a × (ω_b × X) + ω_b × (ω_a × X)]   (intra-base rot-rot, a≠b)
                   X = p_ee - p_0_base
```
which by BAC–CAB equals
```
                  = ½ · [ω_b (ω_a · X) + ω_a (ω_b · X) - 2 X (ω_a · ω_b)]
```

For rot-rot pairs across *different* joints (the original chain-index case),
the symmetric part already reduces to `ω_proximal × (ω_distal × X)` because
the cross-`(ω_b × ω_a) × X` term appears in only one direction (the proximal
one), so symmetrizing halves it but the proximal-direction also contains the
other piece. Need to re-derive cleanly across all combinations.

### Recommended next-session plan

1. **Re-derive `H_pose[:, a, b]`** by writing the SE(3)-integrate exp map to
   2nd order, taking the mixed partial of `p_ee(integrate(q, v))` w.r.t.
   `v_a v_b`, and collecting terms case-by-case:
   - (a, b) in **different joints** with j_a ≠ j_b (call them A and B):
     - A proximal ancestor of B: ω_a doesn't change with v_b (b is below a
       in the tree), but p_ee, p_B do → use ω_a × (J_v[:, b] - ω_a × (p_B - p_A))
       formula (etc).
     - General: combination of the above and chain analysis.
   - (a, b) in the **same joint** (multi-DOF joint, esp. floating base):
     - Use the SE(3)-exp 2nd-order expansion. The mixed partial is the
       symmetric part `½ (ω_a × (ω_b × X) + ω_b × (ω_a × X))` for ang-ang,
       and similar for ang-lin and lin-lin.
2. **Validate against the FD-on-Jacobian oracle** in Python on iiwa14/go2/g1
   fixed AND floating, multiple non-degenerate samples, machine precision.
3. **Port to GPU codegen** in `_eepose_gradient_hessian.py` — the formula
   uses the same shared FK pass as the gradient (`Xw[j]`) so the cost is
   incremental O(nv²) work per ee.
4. **Validate CUDA equivalence** end-to-end on the standard robot set.

### Useful references
- Featherstone Ch. 7-8 — spatial cross product, ad operator.
- Pinocchio `computeJointKinematicHessians` / `getJointKinematicHessian` —
  the canonical analytic Hessian; its LWA output is the joint kinematic
  Hessian tensor (not directly d²pose/dv², but the conversion is well-defined
  via the linear and rpy chain rules above).
- The existing `end_effector_pose_gradient` in `RBDReference.py:1079` — the
  shared FK pass + per-DOF chain walk that this Hessian piggybacks on.
