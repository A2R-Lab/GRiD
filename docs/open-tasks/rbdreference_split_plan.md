# RBDReference File-Split Plan (D.5)

**Status: design only. NO code is moved in this batch (T1).**

The split was deliberately deferred out of the T1 alignment-audit batch so that
T3/T4 do not have to rewrite every line they touch. This document is the contract
for the future B+C pass that actually performs the move.

## Goal

Break the single `RBDReference/RBDReference.py` (~3700 lines, one class) into
topical mixin modules, while preserving the public import contract:

```python
from RBDReference import RBDReference
```

## Mechanism

Each topical module defines one mixin class holding the relevant methods. The
public class is a thin shim that composes them via MRO:

```python
# RBDReference/RBDReference.py  (post-split shim)
from ._helpers import _HelpersMixin
from ._kinematics import _KinematicsMixin
from ._dynamics import _DynamicsMixin
from ._gradients_and_hessians import _GradientsAndHessiansMixin


class RBDReference(
    _HelpersMixin,
    _KinematicsMixin,
    _DynamicsMixin,
    _GradientsAndHessiansMixin,
):
    def __init__(self, robotObj):
        ...  # stays on the concrete class
```

`RBDReference/__init__.py` keeps `from .RBDReference import RBDReference`
unchanged, so all callers (`from RBDReference import RBDReference`) are untouched.

Notes for the implementer:
- All mixins share `self.robot` and the normalize/denormalize helpers, so
  `_HelpersMixin` must be first in MRO (and the others may call into it).
- `__init__` stays on the concrete `RBDReference` class, not a mixin.
- `@staticmethod` methods move with their topic unchanged.
- No behavior change: this is a pure move + recompose. Validate by the same
  `tests/` suite with unchanged pass/fail/skip counts.

## Method → file map

Line numbers are pre-split anchors (current file) for locating each method.

### `_helpers.py` — `_HelpersMixin`
Normalization, quaternion/SO3/SE3 math, spatial-algebra primitives, EE-joint
selection. (Shared by everything; first in MRO.)

- `_normalize_q_input`, `_normalize_v_input`, `_denormalize_v_output`
- `_permute_matrix_prefix`, `_denormalize_qv_matrix_output`,
  `_denormalize_reduced_q_matrix_output`, `_denormalize_rnea_grad_output`
- `_normalize_xyzw_quaternion`, `_quat_mul_xyzw`, `_quat_exp_from_half_omega`,
  `_rotation_from_quat_xyzw`, `_quat_xyzw_from_rotation_matrix`
- `_so3_skew`, `_so3_V_matrix`, `_so3_right_jacobian`, `_se3_Q_block`,
  `_so3_exp`
- `cross_operator`, `dual_cross_operator`, `dot_matrix`, `icrf`,
  `factor_functions`
- `_mxS`, `mxS`, `mx1`, `mx2`, `mx3`, `mx4`, `mx5`, `mx6`
- `fxv`, `fxS`, `vxIv`, `crm`
- `select_end_effector_joints`, `_normalize_ee_offsets`, `equals_or_hstack`
- `_mimic_multiplier`, `_mimic_offset`, `_has_mimic_joints`,
  `_vinds_for_subtree`
- skew/rotation helpers used by the floating-base Lie paths:
  `_skew_from_vector`, `_vector_from_skew`, `_rotation_from_grid_rotvec`,
  `_spatial_transform_from_motion`, `_floating_root_q_from_spatial_transform`,
  `_floating_lie_perturbed_q`, `_as_index_list`,
  `_spatial_xmat_derivative_func`, `_spatial_xmat_second_derivative_func`

### `_kinematics.py` — `_KinematicsMixin`
Integrators, end-effector pose and its derivatives.

- `integrate`, `dIntegrate`, `_integrator_butcher`, `integrator`,
  `integrator_grad`
- `_normalize_kinematics_q`
- `end_effector_pose`, `end_effector_pose_gradient`,
  `end_effector_pose_hessian`, `end_effector_pose_hessian_analytic`

### `_dynamics.py` — `_DynamicsMixin`
RNEA, mass-matrix inverse, CRBA, ABA / forward dynamics, external forces.

- `apply_external_forces`
- `rnea_fpass`, `rnea_bpass`, `rnea`
- `minv_bpass`, `minv_fpass`, `minv`
- `aba`, `crba`
- `forward_dynamics`

### `_gradients_and_hessians.py` — `_GradientsAndHessiansMixin`
First-order RNEA gradients, forward-dynamics gradient, second-order (IDSVA-SO /
FDSVA-SO) including the floating-base Lie-derivative paths.

- `rnea_grad_fpass_dq`, `rnea_grad_fpass_dqd`, `rnea_grad_bpass_dq`,
  `rnea_grad_bpass_dqd`, `rnea_grad`
- `forward_dynamics_grad`
- `_floating_idsva_d2tau_dq_lie_finite_diff`,
  `_floating_gravity_d2tau_dq_lie_direct`
- `idsva_so_body_frame`, `idsva_so_world_frame`, `idsva_so`, `fdsva_so`

## Boundary risks to watch in the B+C pass

- A few small spatial/Lie helpers (e.g. `_spatial_xmat_*derivative_func`,
  `_floating_lie_perturbed_q`) are used only by the gradients module but are
  classified as helpers above; either is fine as long as MRO resolves them.
  Keep them in `_HelpersMixin` to avoid cross-mixin coupling.
- `crm` is used by both dynamics and gradients → keep in `_HelpersMixin`.
- Confirm no method relies on textual ordering within the file (none should;
  Python binds by name at call time).
