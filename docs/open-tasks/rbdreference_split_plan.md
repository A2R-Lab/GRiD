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

## grid_plant reference module (add during the split)

**New requirement (filed 2026-05-30):** add an RBDReference (numpy) reference
implementation of the CUDA `grid_plant` surface — plant step + costs + barriers —
so the `grid_plant::` namespace gets a *real* CPU equivalence oracle. Today
`grid_plant` only self-validates (analytic + finite-difference recompute inside
`test/cuda_equivalents/cuda_plant_smoke_runner.cu` /
`test_cuda_plant_equivalence.py`); it never compares against a shared numpy
reference the way every other algorithm does (e.g. the integrator test calls
`project_model.integrator(...)` / `.integrator_gradient(...)` directly as the
oracle — `test_cuda_integrator_equivalence.py:223,238`). This closes that gap.

The CUDA spec to mirror **verbatim in convention** is the already-merged
`GRiDCodeGenerator/algorithms/_plant.py` (T6, merged at `4c91305`). Reuse it as
the spec — do not re-derive the math.

### Module home — recommendation: a new `_plant.py` `_PlantMixin`

Add a **fifth mixin module** `RBDReference/_plant.py` defining `_PlantMixin`, and
compose it into the shim:

```python
# RBDReference/RBDReference.py  (post-split shim, augmented)
from ._helpers import _HelpersMixin
from ._kinematics import _KinematicsMixin
from ._dynamics import _DynamicsMixin
from ._gradients_and_hessians import _GradientsAndHessiansMixin
from ._plant import _PlantMixin


class RBDReference(
    _HelpersMixin,
    _KinematicsMixin,
    _DynamicsMixin,
    _GradientsAndHessiansMixin,
    _PlantMixin,
):
    ...
```

Rationale for a *separate* module rather than folding the plant methods into
`_kinematics.py` (where the integrators live):
- The plant surface is a distinct *composition layer* (it consumes the
  integrator + EE-pose references, it doesn't implement RBD), so it reads
  cleanest as its own topic — same separation the CUDA side keeps (`_plant.py`
  is a sibling namespace, not part of `grid::`).
- It keeps the user's stated preference satisfied: the plant reference lives
  **with the integrators conceptually** (both are the "step/trajectory" layer)
  while staying decoupled, and `_PlantMixin` simply calls `self.integrator(...)`
  / `self.end_effector_pose[_gradient](...)` across the mixin boundary via MRO —
  no code duplication.
- Last in MRO is fine: it only *calls into* the other mixins, nothing calls into
  it. `_HelpersMixin` stays first (shared state), unchanged.

(If the implementer finds the extra file not worth it, folding `_PlantMixin`'s
methods into `_kinematics.py` is acceptable — they're pure compositions of
methods already in that module — but the separate module is the recommendation.)

Public surface is preserved: `from RBDReference import RBDReference` still yields
one class exposing `.plant_step`, `.quadratic_state_cost`, etc.

### Reference functions to implement (mirror `_plant.py` verbatim in convention)

State convention (matches the integrator + the CUDA plant): `x = [q (nq); qd
(nv)]`, `u = control torques (nv)`. Cost = `½·rᵀ diag(W) r`. Hessian = the
**ratified Gauss-Newton** outer product (diag for quadratic, `Jₚᵀ W Jₚ` for
EE-pos) — the true 2nd-order term is intentionally dropped, exactly as the CUDA
`// TODO(plant-2nd-order)` / `// TODO(ee-2nd-order)` sites note. Barriers are the
log-barrier `−μ·Σ(log(x−lo)+log(hi−x))` with an `isfinite` guard per side so an
`±inf` bound contributes exactly zero.

1. **`plant_step(q, qd, u, dt, integrator_type="euler")` → `x_kp1` (nq+nv,)** —
   thin wrapper: `return self.integrator(q, qd, u, dt, integrator_type)`
   (`RBDReference.py:338`). Mirrors `gen_plant_step` (`_plant.py:42`, which wraps
   `grid::integrator_device`). The `[A|B]` plant-step gradient is likewise a
   pass-through of `self.integrator_grad(q, qd, u, dt, integrator_type)`
   (`RBDReference.py:377`) — expose it as `plant_step_gradient(...)` returning the
   same `(2·nv, 3·nv)` `[d/dq | d/dqd | d/du]` matrix the CUDA `s_dAB` carries
   (`gen_plant_step_gradient`, `_plant.py:69`). No true 2nd-order plant Hessian
   (grid has none; the cost layer supplies the GN Hessian instead).

2. **`quadratic_state_cost(x, x_des, Q)` → (value, grad, hess)** — mirrors
   `_gen_quadratic_cost_family(self, "state")` (`_plant.py:146`, emitted via
   `gen_quadratic_state_cost`, `:247`). `r = x − x_des` (size `nx = nq+nv`);
   `value = ½·Σ Qᵢ rᵢ²`; `grad = Q ∘ r`; `hess = diag(Q)` (column-major `nx×nx`).

3. **`quadratic_input_cost(u, u_des, R)` → (value, grad, hess)** — mirrors
   `_gen_quadratic_cost_family(self, "input")` (`gen_quadratic_input_cost`,
   `_plant.py:251`). Same as above with `r = u − u_des` (size `nu = nv`),
   `hess = diag(R)`.

4. **`ee_pos_cost(q, p_des, W, ee=0)` → (value, grad_x, hess_x)** — mirrors
   `gen_ee_pos_cost` (`_plant.py:259`). `p(q)` = position rows (0..2) of
   `self.end_effector_pose(q, <ee target>)` (`RBDReference.py:941`); `Jₚ` = rows
   0..2 of `self.end_effector_pose_gradient(q, <ee target>)` (`RBDReference.py:1073`,
   shape `6×nv`, the **d/dv tangent** Jacobian — matches the CUDA `s_deePos`
   layout). With `r = p(q) − p_des` (3-vector) and `W` a 3-vector of per-axis
   weights:
   - `value = ½·Σ_r W[r]·r[r]²`
   - `grad_q = Jₚᵀ (W ∘ r)` (size `nv`); `grad_x = [grad_q ; 0]` (the qd-block is
     **exactly** zero, as the CUDA path asserts).
   - `hess_x` = `nx×nx` with the top-left `nv×nv` q-block = `Jₚᵀ diag(W) Jₚ`,
     everything else zero (GN; the `W∘r`-weighted EE-Hessian term is dropped, per
     the ratified decision and the `// TODO(ee-2nd-order)` site).
   - Use the **same EE selection** the CUDA test uses: `leaf =
     robot.get_leaf_nodes()[ee]`, `target = robot.get_joint_by_id(leaf).get_name()`
     (cf. `test_cuda_plant_equivalence.py:124-127`). Reshape the pose list →
     6-vector before slicing rows 0..2.

5. **`joint_position_barrier` / `joint_velocity_barrier` / `joint_torque_barrier`
   `(vals, lower, upper, mu)` → (value, grad, hess_diag)** — mirror
   `_gen_one_barrier` + `_gen_barrier_helpers` (`_plant.py:383,420`, emitted by
   `gen_plant_barriers`, `:493`). Per DOF `i`, isfinite-guarded each side:
   - value += `−μ·(log(vᵢ−loᵢ) + log(hiᵢ−vᵢ))`
   - grad[i] += `−μ·(1/(vᵢ−loᵢ) − 1/(hiᵢ−vᵢ))`
   - hess_diag[i] += `μ·(1/(vᵢ−loᵢ)² + 1/(hiᵢ−vᵢ)²)`

   An `±inf` bound on either side contributes exactly zero to all three (skip via
   `np.isfinite`). Position reads the q-block, velocity the qd-block, torque a
   standalone u-buffer — the three differ only in which slice they read (mirror
   the CUDA `slice_offset` 0 / nq / 0). A shared scalar helper trio
   (`_plant_log_barrier{,_grad,_hess}`) keeps it single-sourced, matching the
   CUDA `grid_plant_log_barrier*` helpers. (The CUDA path floors the interior
   margin at 1e-10/1e-6 to stay finite at the boundary; the numpy oracle should
   match that flooring so the two agree at/near the boundary — the existing test's
   `barrier_terms` already uses the `max(d, 1e-10)` floor,
   `test_cuda_plant_equivalence.py:217-228`, which is the spec to copy.)

These are pure compositions of methods that already exist in the post-split
`_kinematics.py` (integrator, EE-pose[_gradient]); `_PlantMixin` adds **no** new
RBD math.

### Rewiring `grid_plant` validation

Today `test_cuda_plant_equivalence.py` recomputes every expected value inline
(analytic diag(Q)/diag(R), `JₚᵀWJₚ` from the Python EE Jacobian, a central-diff
FD check, and a numpy `barrier_terms`). Upgrade it to a true
**CUDA-vs-RBDReference** equivalence test, same shape as the other algos:

- Replace the inline analytic recomputes with calls into the new `_PlantMixin`
  methods on the `project_model` adapter (`build_project_adapter` already returns
  an `RBDReference`-backed model — `reference_backend.py`). I.e. compare
  `out["state_cost_*"]` against `project_model.quadratic_state_cost(x, x_des, Q)`,
  `out["ee_cost_*"]` against `project_model.ee_pos_cost(q, p_des, W)`,
  `out["pos_barrier_*"]` against `project_model.joint_position_barrier(...)`, etc.
  This makes the **CUDA and numpy surfaces validate each other**, instead of the
  test re-deriving both.
- Keep the existing FD/self-consistency checks as a secondary guard (they catch a
  shared-bug-in-both case the way the integrator test's FD-sanity does), but the
  primary assertion becomes CUDA == RBDReference. The pass-through checks
  (`plant_step == grid::integrator`, `plant_step_gradient == grid::integrator_gradient`)
  already are equivalence checks — leave them.
- **Add the numpy reference to the RBDReference pytest suite too.** New
  `RBDReference/tests/test_plant_equivalence.py` that exercises `_PlantMixin`
  against a closed-form/FD oracle (value vs `½ rᵀWr`; cost grad vs central-diff;
  barrier vs hand-rolled log-barrier; `plant_step` vs `integrator`) — so the
  reference is covered even when nvcc is absent (the CUDA test skips without
  nvcc). This lands under the **same test-count gate** the D.5 split uses
  (`§4`/`bc_cleanup_plan.md`): the split itself stays a pure move; the plant
  reference + its test are an **additive** follow-on, counted separately.

### Cross-references (this converges with two other plant-surface additions)

All three of these are RBDReference / binding additions clustered around the
`grid_plant` surface — coordinate them:

- **(a) D.4 sysID regressor reference.** `d4_runtime_inertia_params_plan.md` §G.5
  notes "RBDReference has **NO regressor** today" and adds
  `joint_torque_regressor(q, q̇, q̈)` to both the pinocchio backend and the numpy
  reference (`d4_..._plan.md:443-464`). That is the *other* new numpy reference
  method landing in this same window — keep both in mind when touching the
  reference backend / `equivalents/` so the two oracles share plumbing.
- **(b) `grid_plant` Python/handle surface.** `notebook_examples_plan.md:142-168,
  268-269` flags that `grid_plant` has **no `grid_rbd` handle method** yet
  (CUDA-only), which blocks the plant notebook (e). That follow-on exposes
  `plant_step`/costs/barriers through the C-ABI + pybind/JAX handle. The numpy
  `_PlantMixin` here is the natural **CPU oracle** for that handle surface once it
  exists (same role the rest of RBDReference plays for `grid_rbd`).

A one-line pointer is added under `bc_cleanup_plan.md §4`.

### Sequencing

Lands **with / just after the D.5 split (post-F)**, reusing the merged
`_plant.py` as the spec. Concretely: do the pure move first (the split's
test-count gate stays clean), then add `_plant.py`/`_PlantMixin` + its test as an
additive commit, then rewire `test_cuda_plant_equivalence.py` to call the new
reference. No F dependency beyond the split's existing post-T3/T4 ordering (the
plant reference only consumes integrator + EE-pose, which F does not reshape).

## Boundary risks to watch in the B+C pass

- A few small spatial/Lie helpers (e.g. `_spatial_xmat_*derivative_func`,
  `_floating_lie_perturbed_q`) are used only by the gradients module but are
  classified as helpers above; either is fine as long as MRO resolves them.
  Keep them in `_HelpersMixin` to avoid cross-mixin coupling.
- `crm` is used by both dynamics and gradients → keep in `_HelpersMixin`.
- Confirm no method relies on textual ordering within the file (none should;
  Python binds by name at call time).
