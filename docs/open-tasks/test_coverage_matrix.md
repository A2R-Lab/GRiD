# Test Coverage Matrix — per-algo Python (numpy↔pinocchio) AND CUDA, with ROBOT coverage

Audit date: 2026-05-31. Branch: `coverage-audit` (off `modernizing-tests` @ `dca2f39`).
Submodules: GRiDCodeGenerator @ `7ed550c`, RBDReference @ `5232f78`, URDFParser @ `b9ab62f`.
Read-only analysis (this doc + one new isolated regression test); no shared file edited.

**Scope/axis (distinct from `coverage_parity_matrix.md`).** The existing
`coverage_parity_matrix.md` (2026-05-30) maps the *codegen-emit surface ⟺
RBDReference numpy ⟺ pinocchio* chain. THIS doc adds the missing axis the parent
asked for: for every `algo_registry.py` key, **which test exercises it, over
WHICH ROBOTS, and whether MIMIC + floating + the other corner cases are covered**
— then a prioritized gap list and a precise cheap-fill list.

## Robot manifest (the parametrization universe)
`RBDReference/tests/robot_manifest.json` → 9 robots, each `base_modes:["fixed","floating"]`:

| robot | embodiment | DoF class | MIMIC? | continuous joints? | notes |
|---|---|---|---|---|---|
| iiwa14 | manipulator | 7 | no | no | smoke gate-first robot everywhere |
| go2 | quadruped | 12+fb | no | no | small floating, compiles fast |
| g1 | humanoid | ~37 | no | no | large floating |
| h1_2 | humanoid 51DoF | large | **YES** | no | high-DoF mimic stress; spill paths |
| fr3 | manipulator | 7+1mimic | **YES** | no | single mimic (s_sign=+1); NUM_BODIES=9 > NUM_JOINTS=8 |
| rizon4 | manipulator | 7 | no | no | broken zero-inertia URDF (per memory: skip-prone) |
| gen3 | manipulator | 7 | no | **YES** (4 cont + 3 rev) | the continuous-joint robot |
| fetch | mobile_manip | ~10 | no | no | negative gripper axis |
| baxter | dual_arm | ~15 | no | no | dual-arm |

`build_case_params()` (conftest) / `iter_robot_cases()` (model_sources) iterate ALL 9
robots × {base_mode filter}. A test "parametrized over the manifest" therefore hits all
9 robots incl. both mimic robots; a test with a hand-rolled `_CASES`/env-default list does NOT.

Two test layers:
- **Python layer** (`RBDReference/tests/test_*.py`): pure-numpy `RBDReference` vs a pinocchio
  oracle (float64). Driven by `build_case_params` → full manifest.
- **CUDA layer** (`test/cuda_equivalents/test_*.py`): generated CUDA kernels (float32) vs a
  reference. Split into (1) the big `test_cuda_executable_equivalence.py` (full manifest, gated
  algorithm subset) and (2) per-family *smoke runners* with HAND-ROLLED robot lists (mostly
  iiwa14 + 1 floating; env-overridable).

---

## 1. Coverage matrix (per algo_registry key)

Legend — **Python**: `M`=parametrized over full manifest (all 9, both mimic);
`subset`=hand-rolled robot list. **oracle**: `pin`=exact pinocchio call,
`FD`=finite-difference-of-pin or analytic-FD, `pin_so_ext`=bound C++ SO extension,
`xconv`=cross-convention self-consistency (no pin), `NONE`=GRiD-defined (FD/analytic only).
**CUDA**: `EXE`=in the full-manifest executable test, `smoke:<robots>`=dedicated smoke runner
with that robot list, `—`=none. **MIMIC-CUDA**: does a mimic robot (fr3/h1_2) actually get
compared on the CUDA side.

### Core Dynamics
| key | Python test (file) | Py robots | oracle | CUDA test | CUDA robots | MIMIC-CUDA |
|---|---|---|---|---|---|---|
| id | test_rnea_equivalence.py | M (fixed+float) | pin.rnea | EXE | all 9 ×2 | YES (fr3+h1_2; A1 fix un-gated) |
| minv | test_minv_equivalence.py | M | pin.crba→inv | EXE (`direct_minv`) | all 9 | YES |
| fd | test_forward_dynamics_equivalence.py | M | pin.aba + project-aba | EXE | all 9 | YES (norm-rtol for h1_2) |
| aba | test_aba_equivalence.py | M | pin.aba | EXE | all 9 | YES (norm-rtol h1_2/fr3) |
| crba | test_crba_equivalence.py | M | pin.crba | EXE | all 9 | YES |

### Gradients
| key | Python test | Py robots | oracle | CUDA test | CUDA robots | MIMIC-CUDA |
|---|---|---|---|---|---|---|
| id_du | test_rnea_grad_equivalence.py | M (fixed+float) | pin.computeRNEADerivatives | EXE (`inverse_dynamics_gradient_q/qd`) | all 9 ×2 | YES (fr3+h1_2; both bases, B1) |
| fd_du | test_forward_dynamics_grad_equivalence.py | M (fixed+float) | pin ABA derivatives | EXE (`forward_dynamics_gradient_q/qd`) | all 9 ×2 | YES (fixed+floating, B1) |
| f_ext_gradient | test_f_ext_gradient_equivalence.py | M (fixed+float) | pin (−Jᵀ, M⁻¹Jᵀ) | smoke (test_cuda_f_ext_gradient_equivalence.py) | **iiwa14-fixed, go2-float, g1-float** | NO |
| f_ext_gradient_dq | (covered inside f_ext_gradient test, A.3 −∂Jᵀ/∂q) | iiwa14+go2+g1 | FD-of-exact | smoke (same file) | iiwa14, go2, g1 | NO |
| regressor | test_regressor_equivalence.py | M (fixed+float) | pin.jointTorqueRegressor | smoke (test_cuda_regressor.py) | **iiwa14-fixed, fr3-fixed, g1-float** | **YES (fr3-fixed only)** |
| fd_parameter_gradient | (numpy ref ported; checked via regressor + −Minv·Y) | M (via regressor) | −Minv·Y vs pin | smoke (test_cuda_fd_parameter_gradient.py) | **iiwa14 fixed+float only** | NO |

### Integrators
| key | Python test | Py robots | oracle | CUDA test | CUDA robots | MIMIC-CUDA |
|---|---|---|---|---|---|---|
| integrator | test_integrator_pinocchio_equivalence.py | M (fixed+float) × {euler,rk*,...} | pin integrate | smoke (test_cuda_integrator_equivalence.py) | **iiwa14,go2 × fixed+float** | NO |
| integrator_gradient | test_integrator_pinocchio_equivalence.py (+ _fd_sanity) | M (fixed) + FD sanity | pin / FD | smoke (same file) | iiwa14, go2 | NO |
| integrator_with_gradient | (composite of above) | M | pin/FD | smoke (same file) | iiwa14, go2 | NO |

### Kinematics
| key | Python test | Py robots | oracle | CUDA test | CUDA robots | MIMIC-CUDA |
|---|---|---|---|---|---|---|
| ee_pose | test_kinematics_equivalence.py | M (fixed+float) | pin.framePlacement | EXE (`end_effector_pose`) | all 9 ×2 | YES (value mimic-unaffected) |
| ee_pose_gradient | test_kinematics_derivatives_equivalence.py | M (fixed+float) | pin frame Jacobian | EXE (fixed all 9; floating candidate) | all 9 fixed; floating non-mimic + mimic (B2-ee) | YES (fixed+floating, B2-ee) |
| ee_pose_hessian | test_kinematics_derivatives_equivalence.py | M (fixed+float) | pin getJointKinematicHessian | EXE | all 9 fixed; floating | YES (B2-ee) |
| frame_jacobian | test_frame_jacobian_equivalence.py (+ _dot, osc_inertia) | M (fixed+float) | pin getFrameJacobian; FD for Jdot | smoke (test_cuda_frame_jacobian.py) | **iiwa14-fixed, go2-float** | NO |

### Second-Order
| key | Python test | Py robots | oracle | CUDA test | CUDA robots | MIMIC-CUDA |
|---|---|---|---|---|---|---|
| idsva_so (dispatcher) | (dispatches to body/world; both tested below) | — | — | EXE compares body (fixed) | — | — |
| idsva_so_body_frame | test_second_order_pinocchio_equivalence.py | M (fixed+float) | pin_so_ext | EXE (fixed mimic, B2-SO) + smoke fallback (test_cuda_second_order_fallback.py) | EXE: all 9 fixed incl. fr3+h1_2; smoke default **iiwa14** | **YES (fr3+h1_2 fixed via EXE)** |
| idsva_so_world_frame | test_second_order_pinocchio_equivalence.py | M (fixed+float) | pin_so_ext | smoke (test_cuda_idsva_so_world_frame.py) | default **iiwa14,go2,g1,fr3**-floating (env-widenable); **fr3 = floating-MIMIC sentinel, PASSES** | **YES (fr3 floating via world-frame smoke)** |
| fdsva_so | test_second_order_pinocchio_equivalence.py | M (fixed+float) | pin_so_ext compose | EXE (fixed mimic, B2-SO) + smoke fallback | EXE all 9 fixed incl. mimic; smoke default iiwa14 | **YES (fr3+h1_2 FIXED via EXE); FLOATING BROKEN (see C6b)** |

> NOTE (C6 desync resolved 2026-06-01): floating-base mimic SECOND ORDER is *emitted*
> (codegen ungated; idsva_so dispatches to the WORLD-frame inner on floating). Verified
> vs pin_so_ext on fr3-floating (+ go2-floating non-mimic control):
> - **idsva_so (world-frame): CORRECT** (rel ~5e-7). NO LONGER refused; `idsva_so_body_frame`
>   removed from `MIMIC_FLOATING_UNSUPPORTED_GRADIENTS`. Covered by
>   `test_cuda_idsva_so_world_frame.py` (fr3 is its floating-mimic sentinel, default-run).
> - **fdsva_so: REAL floating-base bug (C6b)** the stale gate was hiding — `daba_dqdq` block
>   wrong, rel ~6e-2 fr3-floating AND ~4.4e-2 go2-floating NON-mimic (so a floating-base bug,
>   not mimic-specific; fixed-base PASSES). Root cause lead: `_fdsva_so.py:325-329` — fixed runs
>   `gen_idsva_so_body_frame_public_dvdq_layout_repair` after its inner, floating (world inner)
>   applies none, and the inline FD-gradient shares `s_temp` with the world inner before the
>   contract. `fdsva_so` stays in `MIMIC_FLOATING_UNSUPPORTED_GRADIENTS` until fixed.

### Centroidal / Energy / CoM
| key | Python test | Py robots | oracle | CUDA test | CUDA robots | MIMIC-CUDA |
|---|---|---|---|---|---|---|
| generalized_gravity | test_energy_equivalence.py | M (fixed+float) | pin.computeGeneralizedGravity | smoke (test_cuda_centroidal_equivalence.py) | **iiwa14-fixed, go2-float** | NO |
| nonlinear_effects | test_energy_equivalence.py | M (fixed+float) | pin.nonLinearEffects | smoke (centroidal) | iiwa14, go2 | NO |
| energy | test_energy_equivalence.py | M (fixed+float) | pin KE/PE/mech + coriolis | smoke (centroidal) | iiwa14, go2 | NO |
| com | test_centroidal_equivalence.py | M (fixed+float) | pin.centerOfMass + Jcom | smoke (centroidal) | iiwa14, go2 | NO |
| ccrba | test_centroidal_equivalence.py | M (fixed+float) | pin.ccrba + dh | smoke (centroidal) | iiwa14, go2 | NO |

### Plant
| key | Python test | Py robots | oracle | CUDA test | CUDA robots | MIMIC-CUDA |
|---|---|---|---|---|---|---|
| plant | test_plant_equivalence.py | M (fixed+float) | mixed (FD for GRiD-defined cost; pin for dyn) | smoke (test_cuda_plant_equivalence.py) | **iiwa14-fixed only** (env-widenable) | NO |

### Corner-case coverage (cross-cutting)
| corner case | where covered | robots | notes |
|---|---|---|---|
| floating base | every Python `*_floating_*` test + EXE floating params | all 9 floating | strongest coverage |
| MIMIC joints (Python) | ALL manifest-parametrized Python tests | fr3 + h1_2 | full numpy↔pin coverage incl. floating SO |
| MIMIC joints (CUDA) | EXE only (gated subset) | fr3 + h1_2 | id/crba/minv/fd/aba/ee + fixed gradients/SO; smoke runners SKIP mimic |
| continuous joints | test_continuous_joint_equivalence.py | gen3-fixed ONLY | wrapped-angle invariance; **no floating, no CUDA** |
| cross-convention floating | test_floating_base_conventions.py + _quaternion_derivatives | all 9 floating | pinocchio-order vs legacy-order self-consistency |
| metadata/parse | test_model_metadata.py, test_parse_models.py | all 9 ×2 | nq/nv, mimic actuated-set, strict parse |

---

## 2. Algos tested on only 1–2 robots (FLAGGED)

These have FULL Python manifest coverage but the CUDA side runs a tiny hand-rolled list.
Risk = a CUDA codegen bug that only manifests on a robot shape not in the list (esp.
floating, big-DoF, or mimic) ships silently.

| algo | CUDA robots actually compiled | missing shapes (highest risk) |
|---|---|---|
| **plant** | iiwa14-fixed ONLY | NO floating, NO mimic, NO big-DoF on CUDA |
| **fd_parameter_gradient** | iiwa14 fixed+float | only iiwa14; no quadruped/humanoid/mimic |
| **idsva_so_world_frame** (CUDA) | iiwa14-floating ONLY | no go2/g1/h1_2 floating world-frame on CUDA |
| **frame_jacobian** | iiwa14-fixed, go2-float | no big humanoid, no mimic |
| **centroidal family** (gg/nle/energy/com/ccrba) | iiwa14-fixed, go2-float | no g1/h1_2, no mimic |
| **integrator family** | iiwa14, go2 (×fixed+float) | no big-DoF, no mimic |
| **f_ext_gradient / _dq** | iiwa14, go2, g1 | no mimic, no fetch/baxter shapes |
| **regressor** | iiwa14-f, fr3-f, g1-fl | (best of the smoke set; has 1 mimic) |

---

## 3. PRIORITIZED GAP LIST (ranked by risk)

**G1 (HIGH) — `plant` CUDA tested on iiwa14-fixed only.** The whole plant
cost/constraint/step surface (T6) has zero CUDA coverage for floating base, mimic,
or any non-iiwa shape. A floating- or mimic-specific plant codegen bug ships silently.
Python layer is full-manifest, so the oracle exists — only the CUDA breadth is missing.

**G2 (HIGH) — continuous joints have NO CUDA test and NO floating Python test.**
`test_continuous_joint_equivalence.py` is gen3-fixed-only, numpy-vs-pin. The CUDA
codegen path for continuous (unbounded-revolute, NQ>NV via cos/sin) joints is
**entirely unverified end-to-end**. gen3-floating continuous is also untested.

**G3 (MED) — `idsva_so_world_frame` CUDA on iiwa14-floating only.** World-frame SO is
the dispatched path for FLOATING robots, yet only iiwa14 (smallest floating) is
compiled. go2/g1/h1_2 floating world-frame SO get no CUDA check (the spill/tier paths
that big floating robots hit are exactly where SO bugs live — cf. the matmul blockwrap bug).

**G4 (MED) — centroidal family (gg/nle/energy/com/ccrba) CUDA on 2 small robots.**
No humanoid (g1/h1_2) and no mimic on the CUDA side. Floating CoM/CCRBA on a big
humanoid is the realistic use case and is uncompiled.

**G5 (MED) — `fd_parameter_gradient` CUDA on iiwa14 only.** Param-gradient (−Minv·Y)
on CUDA never sees a quadruped/humanoid/mimic. Python regressor twin is full-manifest.

**G6 (LOW-MED) — `frame_jacobian` CUDA missing big/mimic; `f_ext_gradient` CUDA no mimic.**
Smoke lists omit humanoids and mimic robots.

**G7 (REVISED 2026-06-01) — floating-base mimic SO: idsva_so now CUDA-CORRECT, fdsva_so BROKEN (C6b).**
The C6 desync was investigated. Floating-mimic `idsva_so` (world-frame) is emitted AND verified
correct vs pin_so_ext (fr3-floating rel ~5e-7) — removed from `MIMIC_FLOATING_UNSUPPORTED_GRADIENTS`,
covered by `test_cuda_idsva_so_world_frame.py` (fr3 default-run). `fdsva_so` is a REAL floating-base
bug (not mimic-specific — go2-floating non-mimic fails too): `daba_dqdq` block wrong (rel ~6e-2),
in-kernel scratch corrupted on the fused floating path (`_fdsva_so.py:325-329`, no layout-repair on the
world-inner branch + inline FD-gradient/s_temp sharing). Stays refused for floating-mimic until fixed.

**G8 (LOW) — `idsva_so` dispatcher key has no direct test** — only its two targets
(body/world) are tested. The dispatch *selection* logic (body-for-fixed,
world-for-floating) is asserted in `test_cuda_codegen_layout.py`, so acceptable.

---

## 4. CHEAP both-layer fill list (precise; for main to dispatch — DO NOT edit shared files here)

Each item is a follow-up that fills a gap by WIDENING an existing smoke runner's robot
list (these runners already env-override, so most are a one-line default change OR a CI
env var — no new oracle code needed). Listed file + exact change.

1. **`test/cuda_equivalents/test_cuda_centroidal_equivalence.py`** — default robot list is
   `iiwa14:fixed,go2:floating` (`GRID_CUDA_CENTROIDAL_ROBOTS`). Add `g1:floating` (humanoid)
   and `fr3:fixed` (mimic) to the default, or set the env in the sweep. Closes G4.

2. **`test/cuda_equivalents/test_cuda_idsva_so_world_frame.py`** — default
   `GRID_CUDA_IDSVA_SO_WORLD_FRAME_ROBOTS="iiwa14"`. Add `go2,g1` (and optionally h1_2)
   to default. Closes G3.

3. **`test/cuda_equivalents/test_cuda_plant_equivalence.py`** — `_robot_ids()` default
   `iiwa14` and `base_mode="fixed"` hard-coded. Add a `go2:floating` case (and an `fr3`
   mimic case) via `GRID_CUDA_PLANT_ROBOTS` + a floating parametrization. Closes G1.
   (Higher effort than the others: the test hard-codes `base_mode="fixed"`.)

4. **`test/cuda_equivalents/test_cuda_fd_parameter_gradient.py`** — `_CASES=[("iiwa14","fixed"),
   ("iiwa14","floating")]`. Add `("g1","floating")` and `("fr3","fixed")`. Closes G5.

5. **`test/cuda_equivalents/test_cuda_frame_jacobian.py`** — default
   `GRID_CUDA_FRAME_JAC_ROBOTS="iiwa14:fixed,go2:floating"`. Add `g1:floating`. Closes G6 (frame_jac).

6. **`test/cuda_equivalents/test_cuda_f_ext_gradient_equivalence.py`** &
   **`test_cuda_fext_equivalence.py`** — `_CASES` lack a mimic robot. Add `("fr3","fixed")`.
   Closes G6 (f_ext).

7. **`test/cuda_equivalents/test_cuda_integrator_equivalence.py`** — default
   `GRID_CUDA_INTEGRATOR_ROBOTS="iiwa14,go2"`. Add `g1` (and a mimic) for big-DoF + mimic
   integrator coverage.

8. **NEW (no shared edit): continuous-joint CUDA test (G2).** There is NO CUDA test for
   gen3 continuous joints. A follow-up should add a new isolated
   `test/cuda_equivalents/test_cuda_continuous_joint_equivalence.py` (gen3-fixed; assert the
   generated kernel's id/crba/ee_pose match the RBDReference numpy reference at large wrapped
   angles, mirroring `RBDReference/tests/test_continuous_joint_equivalence.py`). Also add a
   gen3-floating case to that Python test. Highest-value of the list.

9. **`test/cuda_equivalents/test_cuda_second_order_fallback.py`** — default smoke robot
   `iiwa14`. Widen `GRID_CUDA_SECOND_ORDER_SMOKE_ROBOTS` / `..._FLOATING_..` defaults to
   include a quadruped + humanoid so the fallback (non-spill) SO path is checked on big robots.

All of 1,2,4,5,6,7,9 are effectively env-default one-liners with the oracle already in place;
3 and 8 need a little structural work. None require new physics/oracle code.

---

## 5. Regression pin added by this audit (Task 2)
New isolated test: `test/cuda_equivalents/test_cuda_matmul_blockwrap_regression.py`.
Codegen-string sentinel (no nvcc, ~3s): codegens fr3-fixed (`idsva_so_body_frame`,`fdsva_so`
in the list) and asserts (a) fr3 is a meaningful mimic sentinel (`NUM_BODIES`=9 > `NUM_JOINTS`=8),
(b) the emitted `matmul` helper wraps with `%NUM_BODIES`, NOT `%NUM_JOINTS`. Would FAIL if
`gen_matmul` (`GRiDCodeGenerator/helpers/_lin_alg_helpers.py:266`) regressed to the old
`%NUM_JOINTS` form that corrupted the mimic composite-inertia chain. Verified: 2/2 pass on the
fix; the modulus assertion FAILS when reverted to `%NUM_JOINTS`.
