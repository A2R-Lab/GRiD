# Clean-break rename mapping (verbose canonical) — RULINGS LOCKED 2026-06-01

**Resolved (user):** verbosify true abbreviations only; KEEP proper/standard names. Final:
- Verbosify: `id`→`inverse_dynamics`, `fd`→`forward_dynamics`, `id_du`→`inverse_dynamics_gradient`,
  `fd_du`→`forward_dynamics_gradient`, `fd_parameter_gradient`→`forward_dynamics_parameter_gradient`,
  `regressor`→`inverse_dynamics_regressor`, `ee_pose`→`end_effector_pose` (+`_gradient`/`_hessian`).
- KEEP: `aba`, `crba`, `ccrba`, `minv` (crba=M, minv=M⁻¹ — paired family; no `mass_matrix` fn exists),
  `idsva_so`, `fdsva_so`, `f_ext_gradient`, `com`, `osc_inertia`, `integrator`, `energy`,
  `generalized_gravity`, `nonlinear_effects`.
- The ⚠️ rows below are SUPERSEDED by this ruling.
- **`rnea`:** NO alias (it would be a 2nd name for the single `inverse_dynamics` fn, unlike aba/crba
  which are distinct functions). Instead surface "RNEA / Recursive Newton–Euler Algorithm"
  prominently in the `inverse_dynamics` docstring + docs so it's greppable. (user, 2026-06-01)



One verbose, self-documenting name per algo, used identically as **registry key == emitted `grid::`
symbol == bench key == algorithm_list token == printf label**. No short aliases anywhere. No
back-compat. Drop stray aliases (`rnea*`, `eepos`/`deepos` legacy labels, `direct_minv`, the
`_with_x_kp1` suffix, the `_host` suffix on SO symbols, the `SUGGESTED_THREADS` alias).

Legend: ✅ = unambiguous rename; ⚠️ = JUDGMENT CALL (needs your ruling — usually a
literature-standard algorithm name vs a fully-descriptive name).

## Core dynamics
| today (key → emitted symbol) | proposed canonical | note |
|---|---|---|
| `id` → `inverse_dynamics` | **`inverse_dynamics`** ✅ | drop `rnea`/`rnea_single_timing`/`rnea_compute_only` aliases |
| `fd` → `forward_dynamics` | **`forward_dynamics`** ✅ | |
| `aba` → `aba` | **`aba`** ⚠️ | proper name (Articulated Body Algorithm). Keep `aba`, or `forward_dynamics_aba`? |
| `crba` → `crba` | **`crba`** ⚠️ | proper name (Composite Rigid Body Algorithm). Keep `crba`, or `mass_matrix_crba`? |
| `minv` → `direct_minv` | **`mass_matrix_inverse`** ⚠️ | computes M⁻¹ via the "direct" algorithm. `mass_matrix_inverse`? or keep the algo hint `direct_mass_matrix_inverse`? |
| `integrator` → `integrator` | **`integrator`** ✅ | |
| `integrator_gradient` → `integrator_gradient` | **`integrator_gradient`** ✅ | give it a uniform `_inner` (today: `_multistage`) |
| `integrator_with_gradient` → `integrator_gradient_with_x_kp1` | **`integrator_with_gradient`** ✅ | rename the emitted symbol to match the key; add a numpy test |

## Gradients / second-order
| today (key → emitted symbol) | proposed canonical | note |
|---|---|---|
| `id_du` → `inverse_dynamics_gradient` | **`inverse_dynamics_gradient`** ✅ | |
| `fd_du` → `forward_dynamics_gradient` | **`forward_dynamics_gradient`** ✅ | add a standalone `_inner` (today reuses id_du band) |
| `idsva_so` → `idsva_so` | **`inverse_dynamics_hessian`** ⚠️ | IDSVA-SO is the literature name; descriptive = "inverse_dynamics_hessian"/"..._second_order". Keep `idsva_so` or go descriptive? |
| `idsva_so_body_frame` → `idsva_so_body_frame_host` | **`<idsva_so>_body_frame`** ✅ | drop `_host`; frame suffix follows whatever we pick for idsva_so |
| `idsva_so_world_frame` → `idsva_so_world_frame_host` | **`<idsva_so>_world_frame`** ✅ | drop `_host`; ALSO make it a first-class algorithm_list key (today flag-gated only) |
| `fdsva_so` → `fdsva_so` | **`forward_dynamics_hessian`** ⚠️ | same call as idsva_so: keep `fdsva_so` or descriptive? ✅ R3 done: `fdsva_so_kernel` args reordered (`d_workspace` → 2nd) — kernel-internal only, host/Python surface unchanged |
| `f_ext_grad`(token)/`f_ext_gradient` | **`f_ext_gradient`** ⚠️ | fix the truncated token. `f_ext` vs spelled-out `external_force`? `f_ext` is standard notation |
| `f_ext_gradient_dq` | **`f_ext_gradient_dq`** ✅ | (kernel-only today; decide if it needs a `_device`) |
| `regressor` → `inverse_dynamics_regressor` | **`inverse_dynamics_regressor`** ✅ | move output `d_Y` into gridData (drop caller-owned buffer) |
| `fd_parameter_gradient` | **`forward_dynamics_parameter_gradient`** ⚠️ | verbose form; or keep `fd_parameter_gradient`? move `d_dqdd_dpi` into gridData |

## Kinematics / centroidal
| today (key → emitted symbol) | proposed canonical | note |
|---|---|---|
| `ee_pose` → `end_effector_pose` | **`end_effector_pose`** ✅ | drop `eepos` legacy label |
| `ee_pose_gradient` → `end_effector_pose_gradient` | **`end_effector_pose_gradient`** ✅ | drop `deepos` |
| `ee_pose_hessian` → `end_effector_pose_gradient_hessian` | **`end_effector_pose_hessian`** ✅ | drop the stray `_gradient_` in the symbol |
| `frame_jacobian` | **`frame_jacobian`** ✅ | + add kernel/host/timing surfaces (device-only today) |
| `frame_jacobian_dot` | **`frame_jacobian_dot`** ✅ | + add surfaces; add to PER_ALGO_SPECS |
| `osc_inertia` | **`operational_space_inertia`** ⚠️ | osc = operational-space control. Descriptive, or keep `osc_inertia`? + add surfaces |
| `com` | **`center_of_mass`** ⚠️ | keep `com` (standard) or spell out? emit on its own key, not gated on `ee_pose` |
| `ccrba` | **`ccrba`** ⚠️ | proper name (Centroidal Composite Rigid Body Algorithm) → or `centroidal_momentum_matrix`? add `gravity` param for signature uniformity |
| `energy` | **`energy`** ✅ | |
| `generalized_gravity` | **`generalized_gravity`** ✅ | |
| `nonlinear_effects` | **`nonlinear_effects`** ✅ | |

## Cross-cutting (no name choice, just cleanup)
- Outputs into `gridData` for `regressor`/`fd_parameter_gradient` (uniform `(hd_data, model, …)` shape).
- Uniform surface set per algo: `_inner` → `_device` → `_kernel` → `_host` → `_single_timing`/`_compute_only` → batch.
- Uniform host signature block: `(hd_data, d_robotModel, [gravity], [dt], …, d_workspace, …)`.
- Resolve the gravity-sign convention split (GRiD `+9.81` vs RBDReference `-9.81`) to ONE convention.
- Remove the `SUGGESTED_THREADS` back-compat alias (keep `MAX_PERF_LEVEL_THREADS`).
- `com`/`ccrba`/`energy`/`generalized_gravity`/`nonlinear_effects` emit on their OWN keys, not gated on a sibling.

## The ⚠️ judgment calls that need your ruling
1. **Proper literature names** — `aba`, `crba`, `ccrba`, `idsva_so`, `fdsva_so`: keep the standard
   short names (they ARE the correct names in the field), or force-descriptive
   (`forward_dynamics_aba`, `inverse_dynamics_hessian`, …)? My lean: **keep the proper algorithm
   names** (aba/crba/ccrba/idsva_so/fdsva_so) — they're correct, not cryptic abbreviations — while
   verbosifying the true abbreviations (id, fd, minv, id_du, fd_du, ee_pose).
2. **`minv`** → `mass_matrix_inverse` (drop the `direct_` algo hint) — agree?
3. **`osc_inertia`** → `operational_space_inertia` or keep `osc_inertia`?
4. **`f_ext`** and **`com`** — keep the standard short notation, or spell out (`external_force`,
   `center_of_mass`)? My lean: keep `f_ext` and `com` (standard).
