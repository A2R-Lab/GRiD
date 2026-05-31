# Coverage & Parity Matrix: codegen ⟺ RBDReference ⟺ pinocchio

Audit date: 2026-05-30. Branch: `modernizing-tests`. Read-only audit; no code changed.

**Standard:** every function the codegen emits should have (a) an RBDReference
numpy reference twin, (b) a pinocchio oracle test, (c) a CUDA-equivalence test.
The chain is `codegen ⟺ RBDReference numpy ⟺ pinocchio`.

Authoritative sources verified for this audit:
- Emit surface: `GRiDCodeGenerator/algo_registry.py` + `GRiDCodeGenerator/algorithms/_*.py`,
  driven by `GRiDCodeGenerator/GRiDCodeGenerator.py:1689-2021` (`gen_all_code`).
- (a) numpy refs: `RBDReference/RBDReference.py`, mixins `_centroidal.py`, `_energy.py`,
  `_plant.py`, `_regressor.py`.
- (b) pin oracle: `RBDReference/equivalents/pinocchio_backend.py` + `RBDReference/tests/test_*.py`.
- (c) CUDA tests: `test/cuda_equivalents/*`, `test/python_wrappers/*`.

Legend — pin oracle column: **EXACT** = bound `pin.*` C++ call; **FD-of-pin** =
finite-difference of a `pin.*` primitive; **pin_so_ext** = the bound C++
`ComputeRNEASecondOrderDerivatives` extension; **NONE** = no pin oracle (GRiD-defined
quantity, FD-checked or analytic-only). CUDA column: **HAS** = dedicated CUDA-equivalence
test diffs against the (pinocchio-default) reference; **PARTIAL** = shape/smoke or
subset only; **MISSING** = no CUDA test.

---

## 1. Coverage matrix

### Core Dynamics

| Emitted fn (device) | numpy ref | pin oracle | CUDA-equiv | Notes |
|---|---|---|---|---|
| `inverse_dynamics_device` (id / RNEA) | HAS `RBDReference.rnea` (`RBDReference.py:1847`) | EXACT `pin.rnea` (`pinocchio_backend.py:263`) | HAS `test_cuda_executable_equivalence.py:47` | f_ext threaded as param (see f_ext section) |
| `direct_minv_device` (minv) | HAS `RBDReference.minv` (`:2135`) | EXACT `pin.crba`→inv (`:318`) | HAS (`:48`) | singular-dependent (skip on singular q) |
| `forward_dynamics_device` (fd) | HAS `RBDReference.forward_dynamics` (`:2919`) | EXACT via crba/rnea (`:315`) | HAS (`:49`) | f_ext param supported |
| `aba_device` | HAS `RBDReference.aba` (`:2206`) | EXACT `pin.aba` (`:276`) | HAS (`:54`) | f_ext param supported |
| `crba_device` | HAS `RBDReference.crba` (`:2474`) | EXACT `pin.crba` (`:336`) | HAS (`:55`) | known gap: big-floating perf only, parity OK |

Pin tests: `test_rnea_equivalence.py`, `test_minv_equivalence.py`,
`test_forward_dynamics_equivalence.py`, `test_aba_equivalence.py`, `test_crba_equivalence.py`.

### Gradients

| Emitted fn | numpy ref | pin oracle | CUDA-equiv | Notes |
|---|---|---|---|---|
| `inverse_dynamics_gradient` (id_du, ∂ID/∂q,∂v) | HAS `rnea_grad` (`:2852`) | EXACT `pin.computeRNEADerivatives` (`:549`) | HAS (`:50-51`) | f_ext param supported |
| `forward_dynamics_gradient` (fd_du) | HAS `forward_dynamics_grad` (`:2947`) | EXACT `pin.computeABADerivatives` (`:721`) | HAS (`:52-53`) | f_ext param supported |
| **`f_ext_gradient`** (dtau_dfext=−Jᵀ, dqdd_dfext=M⁻¹Jᵀ, −∂Jᵀ/∂q) | HAS `f_ext_gradient` + `f_ext_jacobian_transpose` (`:1888`,`:1920`) | **PARTIAL**: only `dtau_dfext`=−Jᵀ via FD-of-pin rnea (`:349`); `dqdd_dfext` and `−∂Jᵀ/∂q` have NONE | **MISSING** | Python test `test_f_ext_gradient_equivalence.py`. The CUDA `test_cuda_fext_equivalence.py` tests f_ext as a *parameter*, NOT this dedicated 3-output algorithm. See GAP #2. |

Pin tests: `test_rnea_grad_equivalence.py`, `test_forward_dynamics_grad_equivalence.py`,
`test_f_ext_gradient_equivalence.py`.

### f_ext PARAMETER threading (rnea/fd/aba/id_du/fd_du accept f_ext)

| Threaded path | numpy ref | pin oracle | CUDA-equiv | Notes |
|---|---|---|---|---|
| rnea(...,f_ext), fd, aba, rnea_grad(q-block), rnea_grad(qd-block) | HAS (f_ext kwarg on each, `:1847/:2206/:2919/:2852`) | EXACT (rnea/fd/aba/rnea_grad with `fext`) | HAS `test_cuda_fext_equivalence.py:177-185` | 3-way (CUDA/ref/pin) for id/fd/aba/id_du; fd_du grads CUDA-vs-ref only (`:173`) |

### Integrators

| Emitted fn | numpy ref | pin oracle | CUDA-equiv | Notes |
|---|---|---|---|---|
| `integrator_device` (x_{k+1}) | HAS `integrator` (`:355`) | EXACT `pin.integrate`+`pin.aba` (`:952`) | HAS `test_cuda_integrator_equivalence.py` (euler, si-euler, midpoint, rk3, rk4) | |
| `integrator_gradient` (∂x_{k+1}/∂x,u) | HAS `integrator_grad` (`:394`) | EXACT `pin.dIntegrate`+`pin.computeABADerivatives` (`:984`) + FD sanity | HAS (same runner) | floating-base grad pin-checked; FD sanity `test_integrator_gradient_fd_sanity.py` |
| `integrator_with_gradient` (x_{k+1}+grad) | HAS (composed) | EXACT (composed) | HAS (with_x_kp1 host wrapper) | |

Pin tests: `test_integrator_pinocchio_equivalence.py`, `test_integrator_gradient_fd_sanity.py`.

### Kinematics

| Emitted fn | numpy ref | pin oracle | CUDA-equiv | Notes |
|---|---|---|---|---|
| `end_effector_pose_device` (ee_pose) | HAS `end_effector_pose` (`:958`) | EXACT `pin.forwardKinematics`/`updateFramePlacements` (`:1051`) | HAS (`:56`) | always compared even for mimic |
| `end_effector_pose_gradient` (Jacobian) | HAS `end_effector_pose_gradient` (`:1090`) | FD-of-pin (`pin.integrate` perturb, step) (`:1092`) | HAS fixed (`:57`); floating = candidate-only (`:74`) | floating mimic gradient skipped (P4 pending) |
| `end_effector_pose_hessian` / d2ee | HAS `end_effector_pose_hessian` (FD, `:1223`) + `_analytic` (`:1293`) | FD-of-pin + analytic via `getJointKinematicHessian` (`:1128`) | HAS fixed (`:58`); candidate floating | **KNOWN BUG**: orientation-hessian. Pin test restricted to `{iiwa14, fr3}` (`test_kinematics_derivatives_equivalence.py:93`). Analytic-vs-FD self-check at `:129`. |

Pin tests: `test_kinematics_equivalence.py`, `test_kinematics_derivatives_equivalence.py`.

### Kinematics — warp/thread batched FK (NEW)

| Emitted fn | numpy ref | pin oracle | CUDA-equiv | Notes |
|---|---|---|---|---|
| `ee_pose_inner_thread` (`_eepose_gradient_hessian.py:2510`) | (shares `end_effector_pose`) | (shares ee_pose EXACT) | **MISSING** | no test references it (verified: 0 hits in `test/`) |
| `ee_pose_inner_warp` (`:2602`) | (shares `end_effector_pose`) | (shares ee_pose) | **MISSING** | DANGEROUS: emitted, untested |
| `ee_pose_fk_batched_kernel` / `ee_pose_fk_batched` host (`:2702`,`:2782`) | (shares `end_effector_pose`) | (shares ee_pose) | **MISSING** | DANGEROUS: emitted, untested. See GAP #6 |

### Second-Order

| Emitted fn | numpy ref | pin oracle | CUDA-equiv | Notes |
|---|---|---|---|---|
| `idsva_so_device` (dispatcher: body for fixed, world for floating) | HAS `idsva_so` (`:3804`) | (dispatches to body/world) | via the body/world CUDA tests | dispatcher itself = thin selector |
| `idsva_so_body_frame` | HAS `idsva_so_body_frame` (`:3255`) | **pin_so_ext** EXACT (`pinocchio_backend.py:584`) | HAS `test_cuda_second_order_fallback.py` (fixed forced-fallback) | |
| `idsva_so_world_frame` | HAS `idsva_so_world_frame` (`:3557`) | pin_so_ext (cross-checked vs body-frame tensor) | HAS `test_cuda_idsva_so_world_frame.py` | floating path |
| `fdsva_so` (2nd-order FD) | HAS `fdsva_so` (`:3825`) | pin_so_ext composition (`:640`) | HAS (`test_cuda_second_order_fallback.py`, gated on invertible M) | floating = diagnostic (not hard-fail) |

Pin tests: `test_second_order_pinocchio_equivalence.py` (idsva_so body+world, fdsva_so).
Smoke: `test_iiwa14_smoke.py` idsva_so/fdsva_so shape checks.

### Centroidal / Energy / CoM (NEW quickwins)

| Emitted fn | numpy ref | pin oracle | CUDA-equiv | Notes |
|---|---|---|---|---|
| `generalized_gravity` (g(q)=RNEA(q,0,0)) via `gen_id_bias(gravity_only=True)` | HAS `generalized_gravity` (`_energy.py:24`) | EXACT `pin.computeGeneralizedGravity` (`:415`) | **MISSING** | no CUDA test (verified 0 hits) |
| `nonlinear_effects` (c(q,qd)) via `gen_id_bias(gravity_only=False)` | HAS `nonlinear_effects` (`_energy.py:31`) | EXACT `pin.nonLinearEffects` (`:422`) | **MISSING** | see GAP #4 |
| `com_device` (CoM + CoM Jacobian) | HAS `com` (`_centroidal.py:158`) + `jacobian_com` (`:163`) | EXACT `pin.centerOfMass` / `pin.jacobianCenterOfMass` (`:463`,`:470`) | **MISSING** | jacobian_com is the J_com sub-output of `com_device` |
| `ccrba_device` (A, h) | HAS `ccrba` (`:207`) + `centroidal_momentum` (`:215`) | EXACT `pin.ccrba` + `pin.computeCentroidalMomentum` (`:480`,`:494`) | **MISSING** | centroidal_momentum = h sub-output |
| `energy_device` (KE/PE/mechanical) | HAS `kinetic/potential/mechanical_energy` (`_energy.py:38-53`) | EXACT `pin.compute{Kinetic,Potential,Mechanical}Energy` (`:430-448`) | **MISSING** | gravity-sign convention note in codegen `_centroidal.py:571` |

Pin tests: `test_centroidal_equivalence.py` (com/jacobian_com/ccrba/centroidal_momentum),
`test_energy_equivalence.py` (gen-gravity/nonlinear/energy + coriolis_matrix).
**All five emitted centroidal device kernels lack a CUDA-equivalence test.**

### Regressor (numpy + pin only; not a benchmarked GRiD kernel)

| Item | numpy ref | pin oracle | CUDA-equiv | Notes |
|---|---|---|---|---|
| `joint_torque_regressor` | HAS (`_regressor.py:101`) | EXACT `pin.computeJointTorqueRegressor` (`:512`) | N/A | sysID reference; no codegen kernel emitted, so no CUDA twin expected |
| `coriolis_matrix` | HAS (`_energy.py:57`, FD) | EXACT `pin.computeCoriolisMatrix` (`:450`) | N/A | reference-only helper |

### Plant / Cost (grid_plant namespace — GRiD-defined, FD is the oracle)

| Emitted fn | numpy ref | pin oracle | CUDA-equiv | Notes |
|---|---|---|---|---|
| `plant_step` / `plant_step_gradient` | HAS (`_plant.py:104`,`:114`) | EXACT via integrator pass-through (delegates to `integrator`) | HAS `test_cuda_plant_equivalence.py` (== grid::integrator) | |
| `quadratic_state_cost` / `quadratic_input_cost` | HAS (`_plant.py:138`,`:145`) | NONE (GRiD-defined; FD-checked) | HAS (value/grad/GN-hess vs NumPy recompute) | |
| `ee_pos_cost` | HAS (`_plant.py:160`) | NONE (FD-of-EE-pose oracle) | HAS (vs double-precision Python EE Jacobian) | |
| `joint_position/velocity/torque_barrier` | HAS (`_plant.py:216-224`) | NONE (hand-rolled log-barrier oracle) | HAS (value/grad/hess vs NumPy log-barrier) | |
| `com_cost` (`_plant.py:389`) | **MISSING** (no `com_cost` in `RBDReference/_plant.py`) | NONE | **MISSING** | emitted kernel, no numpy ref, no test. See GAP #3 |
| `momentum_cost` (`_plant.py:474`) | **MISSING** (no `momentum_cost` in `RBDReference/_plant.py`) | NONE | **MISSING** | emitted kernel, no numpy ref, no test. See GAP #3 |

Pin/FD tests: `test_plant_equivalence.py` (state/input/ee_pos costs + barriers + plant_step).

---

## 2. Prioritized GAP LIST

Ranked: core dynamics > derivatives > kinematics > plant/cost > FK-batched.
(All core/first-order dynamics + integrators + SO are fully covered — no gaps there.)

1. **[Derivatives] `f_ext_gradient` `dqdd_dfext`=M⁻¹Jᵀ and `−∂Jᵀ/∂q` have NO pinocchio oracle.**
   Only `dtau_dfext`=−Jᵀ is pin-checked (FD-of-pin, `pinocchio_backend.py:349`); the
   other two blocks are internal-FD/analytic only. Minimal fix: add
   `PinocchioModelAdapter.f_ext_gradient` blocks for `dqdd_dfext` (= `pin.computeMinverse @ (-dtau)`)
   and FD-of-pin for `−∂Jᵀ/∂q`, and extend `test_f_ext_gradient_equivalence.py` to assert them.

2. **[Derivatives] `f_ext_gradient` algorithm has NO CUDA-equivalence test.**
   The emitted `f_ext_gradient_device`/`_dq_device` (`_f_ext_gradient.py:261,351`) is never
   diffed in CUDA. (`test_cuda_fext_equivalence.py` only covers f_ext as a *parameter*, not
   this 3-output algorithm.) Minimal fix: add a CUDA-equivalence case (runner emits
   `f_ext_gradient`, diff dtau/dqdd/−∂Jᵀ vs `RBDReference.f_ext_gradient`).

3. **[Plant/Cost] `com_cost` and `momentum_cost` are emitted but have NO numpy ref and NO test (DANGEROUS).**
   `_plant.py:389,474` emit kernels; `RBDReference/_plant.py` has no twin. Minimal fix: add
   `RBDReference._plant.com_cost` / `momentum_cost` (built on existing `com`/`centroidal_momentum`),
   then add CUDA cases to `test_cuda_plant_equivalence.py` (FD/NumPy oracle, like ee_pos_cost).

4. **[Centroidal] All five centroidal device kernels (generalized_gravity, nonlinear_effects, com, ccrba, energy) have NO CUDA-equivalence test.**
   numpy refs + EXACT pin oracle exist and are tested in Python, but the CUDA emission is
   unverified end-to-end. Minimal fix: one CUDA smoke/equivalence runner (mirror
   `test_cuda_plant_equivalence.py`) codegen'ing the centroidal profile and diffing
   com/ccrba/energy/gen-gravity/nonlinear vs the pinocchio-default reference.

5. **[Kinematics] `end_effector_pose_hessian` (d2ee) orientation block KNOWN BUG; pin parity restricted to {iiwa14, fr3}.**
   `test_kinematics_derivatives_equivalence.py:93` only runs hessian-vs-pin on those two
   robots; CUDA hessian is in `KNOWN_FAILING`-style restricted comparison. Not a missing
   test — a tracked correctness bug (CUDA + analytic share it). Fix = land the d2ee
   orientation-hessian correction (`docs/d2ee_analytic_derivation.md`), then widen the robot set.

6. **[FK-batched] `ee_pose_inner_thread` / `ee_pose_inner_warp` / `ee_pose_fk_batched` have NO test at all (DANGEROUS).**
   Emitted at `_eepose_gradient_hessian.py:2510/2602/2702/2782`; zero references in `test/`.
   They reuse the validated `end_effector_pose` math, so correctness risk is the *batched
   warp/thread plumbing*, not the kinematics. Minimal fix: a CUDA smoke that runs
   `ee_pose_fk_batched` over B configs and diffs the 7-vector pose vs `end_effector_pose`
   (and vs the per-call `ee_pose` kernel).

7. **[Plant/Cost — floating mimic] ee_pose_gradient / hessian CUDA comparison skipped for floating mimic (P4 pending).**
   `MIMIC_SUPPORTED_ALGORITHMS` (`test_cuda_executable_equivalence.py:121`) and
   `MIMIC_FLOATING_UNSUPPORTED_GRADIENTS` gate these off. Expected (phased rollout), not a
   silent gap — tracked in T3-finisher.

---

## 3. Known-correct EXCEPTIONS (pin oracle impossible / awkward — by design)

- **Plant costs** (`quadratic_state/input_cost`, `ee_pos_cost`, barriers): GRiD-defined
  objectives. The oracle is a hand-rolled NumPy / FD recompute, run in double precision
  (`test_plant_equivalence.py`). A `pin.*` oracle does not exist for these.
- **d2ee orientation-hessian**: KNOWN BUG shared by CUDA + analytic; pin test scoped to
  `{iiwa14, fr3}` and an analytic-vs-FD self-check guards the rest.
- **idsva_so / fdsva_so**: use the bound `pin_so_ext` C++ extension
  (`RBDReference/equivalents/pin_so_ext/`) as the EXACT oracle — pinocchio's Python API does
  not expose second-order RNEA directly. fdsva_so is a pinocchio-grounded *composition*, not
  a single bound call. Floating fdsva_so is diagnostic (not hard-fail) by design.
- **warp/thread/batched FK**: validate vs `end_effector_pose` (same math), NOT a new pin
  primitive — but see GAP #6 (currently NO test exists yet).
- **fd_du with f_ext**: pinocchio has no direct fd-grad-with-fext entry, so the CUDA fext
  test checks it CUDA-vs-RBDReference only; the RBDReference fd-grad is itself pin-checked
  in the Python suite (`test_cuda_fext_equivalence.py:169-173`).
- **regressor / coriolis_matrix**: reference-only helpers (no emitted GRiD kernel), fully
  pin-checked in Python; no CUDA twin is expected.

---

## 4. Emitted-but-UNTESTED (the dangerous gaps)

Functions the codegen emits with **NO test exercising them at all**:

1. `ee_pose_inner_thread`, `ee_pose_inner_warp`, `ee_pose_fk_batched_kernel`,
   `ee_pose_fk_batched` host (warp/thread batched FK) — zero test references.
2. `com_cost`, `momentum_cost` plant kernels — no numpy ref, no Python test, no CUDA test.

Functions emitted with refs+Python-pin coverage but **NO CUDA-equivalence test**
(lower risk — Python side is verified, CUDA emission unproven):

3. `f_ext_gradient` (dedicated 3-output algorithm).
4. `generalized_gravity`, `nonlinear_effects`, `com_device`, `ccrba_device`, `energy_device`
   (all five centroidal quickwins).
