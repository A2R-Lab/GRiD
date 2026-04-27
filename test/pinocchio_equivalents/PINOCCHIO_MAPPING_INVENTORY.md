# Pinocchio Mapping Inventory

This inventory was created from the checked-out `URDFParser` and `RBDReference`
code in `research/GRiD-A2R`. It is intentionally conservative: only mappings that
appear clear from the current source are considered safe for enforcing tests.

## RBDReference Functions Found In This Checkout

Clear or mostly clear dynamics-facing surface:

- `rnea(q, qd, qdd=None, GRAVITY=-9.81, f_ext=None)`
- `minv(q, output_dense=True)`
- `crba(q)`
- `aba(q, qd, tau, f_ext=[], GRAVITY=-9.81)`
- `forward_dynamics(q, qd, u)`
- `rnea_grad(q, qd, qdd=None, GRAVITY=-9.81, USE_VELOCITY_DAMPING=False)`
- `forward_dynamics_grad(q, qd, u)`

Kinematics-facing surface:

- `end_effector_pose(...)`
- `end_effector_pose_gradient(...)`
- `end_effector_pose_hessian(...)`

Pass-level helpers and implementation internals:

- `rnea_fpass(...)`, `rnea_bpass(...)`
- `minv_bpass(...)`, `minv_fpass(...)`
- `rnea_grad_fpass_dq(...)`, `rnea_grad_fpass_dqd(...)`
- `rnea_grad_bpass_dq(...)`, `rnea_grad_bpass_dqd(...)`
- several spatial algebra helpers used by the algorithms above

## Clear Pinocchio Mappings Used In V1

- `RBDReference.rnea(...)` maps to `pinocchio.rnea(...)`
  Note: GRiD returns `(c, v, a, f)` while Pinocchio returns the generalized
  torque result directly, so the suite compares `c` to Pinocchio's `tau`.
- `RBDReference.minv(...)` maps to a Pinocchio mass-matrix path using
  `pinocchio.crba(...)` followed by matrix inversion.
  This is treated as the stable Python-side comparison target for v1.
- `RBDReference.crba(q)` maps to `pinocchio.crba(...)`
  This is currently enforced for fixed-base `iiwa14`.
- `RBDReference.rnea_grad(...)` maps to `pinocchio.computeRNEADerivatives(...)`
  for the `dtau_dq` and `dtau_dv` blocks. This is currently enforced for
  fixed-base `iiwa14`.
- `RBDReference.end_effector_pose(...)` maps to Pinocchio frame placements plus
  local-point offsets. The suite compares translation directly and compares
  orientation through reconstructed rotation matrices to avoid Euler-angle
  singularity artifacts. This is currently enforced for fixed-base `iiwa14`
  targets `iiwa_joint_7`, `iiwa_joint_ee`, and `tool0_joint`.

## Ambiguous Or Deferred Mappings

- `aba(q, qd, tau, ...)`
  Likely maps to `pinocchio.aba(...)`, but the current checkout and docs still
  describe floating-base ABA as under development.
- `forward_dynamics(q, qd, u)`
  This appears to be a composed quantity rather than a direct Pinocchio primitive.
  It is deferred until the exact semantics are documented relative to `aba`.
- `forward_dynamics_grad(...)`
  Pinocchio has related derivative APIs, but shape, semantics, and state ordering
  still need careful validation before turning this into an enforcing test.
- End-effector helpers beyond pose
  The current source includes a TODO that floating-base support is not fully added
  and tested for end-effector derivatives and Hessians, so those remain deferred.

## Normalization Steps Required Today

- GRiD floating-base quaternion order is `wxyz`.
- Pinocchio free-flyer quaternion order is `xyzw`.
- GRiD `URDFParser` uses parser-defined DFS joint ordering with optional sibling
  tie-breaking, so joint-name alignment must be explicit.
- GRiD merges fixed joints into retained custom structures, while Pinocchio keeps a
  different model/data view for those URDF elements.
- `RBDReference.rnea(...)` returns more than the primary generalized torque result,
  so tests extract and compare only the torque-like output for v1.

## Relevant Pinocchio Capabilities Not Yet Implemented In RBDReference

The list below is intentionally restricted to capabilities that look relevant from
the current repo layout and documentation, not from guesswork about hidden APIs.

- Pinocchio-style model/data separation as a first-class public Python interface
- A mass-matrix-first public API surface analogous to `crba` plus helper utilities
- More Pinocchio-shaped user-facing wrappers for bias, gravity, and composed
  dynamics quantities
- Clearly documented free-flyer conventions that match Pinocchio terminology
- Broader, clearly exposed kinematics and frame-placement helpers comparable to
  Pinocchio's frame APIs

## V1 Deferrals

- Additional fixed-base algorithms beyond `rnea`, `minv`, `crba`, `rnea_grad`,
  and selected pose targets on `iiwa14`
- All floating-base numerical enforcement until free-flyer conventions are
  confirmed trustworthy
- Forward-dynamics derivative equivalence
- End-effector derivative and Hessian equivalence
