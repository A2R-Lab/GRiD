The MuJoCo (mjx) output convention
==================================

**The canonical page** for GRiD's MuJoCo-convention support: what differs,
what transforms, where it applies, and what it costs. The mathematical
derivation with its machine-precision validation lives in
``external/RBDReference/equivalents/mujoco_convention.md`` (the transforms
themselves in ``mujoco_convention.py``); the per-method API tour is in
:doc:`../tutorials/python_wrappers`.

The two raw differences
-----------------------

GRiD is natively **Pinocchio**-convention. MuJoCo differs for the
free-floating base in exactly two ways:

1. **Quaternion order.** MuJoCo ``qpos`` stores the free-flyer quaternion
   **wxyz** (scalar first); pin/GRiD use **xyzw**. A pure relabel on ``q``
   and integrator outputs — never touches velocities, forces, or gradients.
2. **Free-joint velocity frame.** MuJoCo ``qvel`` is
   ``[v_lin GLOBAL ; omega LOCAL]``; the pin spatial twist is
   ``[v_lin LOCAL ; omega LOCAL]``. That is ONE root-block basis change
   ``G(q) = blockdiag(R, I3)`` on the leading 6 tangent DOF (``R`` = base
   orientation). ``G`` is orthogonal: ``G^{-1} = G^T``.

Everything else — every joint DOF past the root, every fixed-base robot — is
IDENTICAL between the conventions. That is why ``output_convention="mujoco"``
is accepted (as a no-op) on fixed-base robots: generic code can set it
everywhere.

Value and derivative transforms
-------------------------------

Velocities transform by ``G``, forces (covectors) by ``G^{-T} = G``, the mass
matrix by congruence ``G M G^T``, and each derivative index contracts with
the matching ``G`` factor (see the derivation doc for the full table,
including the subtle dR/dq chart-slope terms in the second-order tensors).

**Do not "debug" the reframe.** If an mjx output looks wrong, check the pin
baseline FIRST — pin↔mjx is a KNOWN, validated transform, and every mjx bug
so far was actually a pin bug or a caller-side convention mixup
(`agent debugging guide <https://github.com/A2R-Lab/GRiD/blob/main/docs/agent_debugging_guide.md>`_ §1k).

How GRiD serves it: native kernel twins
---------------------------------------

For a **floating-base robot without mimic joints or skew axes**, codegen
emits an mjx twin of each kernel (the ``MUJOCO_OUTPUT=true`` template
instantiation) with the input conversion + output reframe fused in-kernel —
no host-side transform, no extra copies. Values AND the derivative /
second-order surfaces are served natively this way.

Twins are **never emitted** for fixed-base or mimic/skew robots (fixed base:
the conventions coincide; mimic/skew: the reduced-coordinate reframe is not
implemented). On such a ``.so`` the mjx-specific methods raise a clear error
naming the cause.

Three ways to ask for it
------------------------

- ``handle.mujoco.<method>(...)`` — per-call view, thread-safe, never mutates
  shared state (recommended).
- ``handle.output_convention = "mujoco"`` — the handle-wide default.
- ``register_robot(..., output_convention="mujoco")`` — the same default at
  registration. Runtime-only: NOT in the ``.so`` cache key.

What it costs (and how to opt out)
----------------------------------

The mjx twins — not second-order kernel size — dominate humanoid build cost:
the second-order twins are the largest kernels in a big floating-base build
(``idsva_so_world_frame``'s twin was 28× its pin kernel raw; block-
parallelizing the epilogue cut it to ~2.4×). If you only need
Pinocchio-convention outputs, build pin-only with
``enable_mujoco_kernels=False`` (or ``grid-generate --no-mujoco-kernels``) —
on a large robot this is the difference between building and running out of
memory. A pin-only floating build then refuses ``output_convention="mujoco"``
with a clear error at registration time.

The known-parity contract
-------------------------

``mujoco_convention.py``'s transforms are validated to machine precision
against a real MuJoCo (``mj_fullM`` / ``mj_inverse`` / ``mj_forward``) and by
finite-difference self-consistency; the CUDA twins are validated against that
reference in the equivalence suites. The single source of truth for the
convention mapping is ``mujoco_convention.py`` — both the tests and the
codegen input-conversion mirror it.

Convention map (per function, empirically confirmed)
----------------------------------------------------

Promoted from the 2026-06 fusion ledger (previously only in a gitignored
planning file) so the tracked docs are self-contained. ``G = blockdiag(R, I3)``
on the tangent ``[v_lin(0:3); omega(3:6); joints]`` is orthogonal
(``G^{-1} = G^T``); ``q = [pos(3), quat_xyzw(4), joints]``; ``R`` is the base
rotation; every transform is a no-op on a fixed base. Verified against real
MuJoCo (``mj_jacBody`` / ``mj_fullM`` / ``mj_inverse`` / ``mj_energy*`` /
``mj_integratePos`` / ``mj_step``) to ~1e-15, or by finite differences along
the mjx retraction where MuJoCo has no equivalent. The implementation is
``bindings/grid_rbd/_mujoco.py`` (values) and the generated mjx kernel twins
(derivatives / second order).

.. code-block:: text

   G = blockdiag(R, I3) on tangent [v_lin(0:3); omega(3:6); joints]; ORTHOGONAL so G^{-1}=G^T,
   G^{-T}=G. q=[pos(3),quat_xyzw(4),joints]. R = base_rotation(q). All transforms are NO-OP on fixed
   base. Verified vs real MuJoCo (mj_jacBody/mj_fullM/mj_inverse/mj_energy*/mj_integratePos/mj_step)
   to ~1e-15, or FD-along-mjx-retract / defining-relationship where no MuJoCo equivalent.

   ### Helper primitives (emit these once, reuse everywhere) — all act on the base block only:
   - `input_convert`: quat reorder wxyz->xyzw on q[3:7]; qd[0:3] = R^T·qd[0:3] (=G^{-1}); qdd =
     accel_mjx_to_pin = R^T·qdd[0:3] − omega×v_local (omega=qd[3:6], v_local=qd[0:3] AFTER qd convert);
     u(force) = R^T·u[0:3] (=G^{-T}=... covector, R^T).
   - `base_rotate` (covector/contravector OUT): out[0:3] = R·out[0:3].
   - `accel_out` (fd qdd OUT): qdd[0:3] = R·(qdd[0:3] + omega×v_local).
   - `congruence`: rows0:3 ← R·rows0:3, then cols0:3 ← cols0:3·R^T (corner gets R·M00·R^T).
   - `column_reframe` (Jacobian J·G^{-1}): base-linear COLUMNS 0:3 ← cols0:3·R^T (right-mult).
   - `retract`: floating base pos += dt·qvel[:3] (GLOBAL add) instead of pin R·V(phi)·rho; quat & joints
     unchanged. `mjx_dIntegrate`: base-linear Jacobian block → I, zero lin↔ang cross-coupling.

   ### Per-function table (formula + class + verified error):
   | Function | Transform | Class | Verified |
   |---|---|---|---|
   | energy (KE/PE/mech) | INVARIANT; input-convert qd only | invariant | mj_energy* 3.6e-15 |
   | KE/PE regressor | INVARIANT (Y in param space); input qd | invariant | Y·π=energy 1e-15 |
   | com (position) | INVARIANT; quat reorder only | invariant | subtree_com 1.1e-16 |
   | end_effector_pose | INVARIANT (out=rpy); quat reorder | invariant | xpos 0 |
   | osc_inertia Λ | INVARIANT (G's cancel in J Minv J^T); quat | invariant | mj 2.7e-15 |
   | centroidal_momentum h | INVARIANT; input qd | invariant | 7.1e-15 |
   | fk_batched | INVARIANT (already wxyz out!); quat in | invariant | — |
   | generalized_gravity | base_rotate (G·out) [immune to ω×v: qd=0] | base-row | mj_inverse 3.6e-15 |
   | inverse_dynamics_regressor | base_rotate rows (G·Y) + input qd,qdd | base-row | Y·π=Gτ 8.9e-16 |
   | forward_dynamics_parameter_gradient | base_rotate rows (G·out), **NO ω×v** (π-indep, drops) | base-row | 9.6e-8 FD floor |
   | **nonlinear_effects** ⚠REVISED | G·ID(q, qd_pin, accel_mjx_to_pin(0,v_pin)) i.e. a_pin=−ω×v, NOT plain covector | base-row + **accel-couple** | naive 1.68→**5e-16** vs qfrc_bias |
   | crba M | congruence G·M·G^T | congruence | done |
   | minv | congruence (SYMMETRIC_UPPER — re-mirror base block) | congruence | done |
   | coriolis_matrix C | similarity G·C·G^{-1} (=G·C·G^T, G orthog) + input qd | congruence | C·qd covector 1.4e-14 |
   | frame_jacobian | column_reframe J·G^{-1} | column-reframe | mj_jacBody **4.4e-16** |
   | frame_jacobian_dot | column_reframe + input qd | column-reframe | 8.9e-16 |
   | jacobian_com (J_com) | column_reframe | column-reframe | mj_jacSubtreeCom 5.6e-16 |
   | ccrba A | column_reframe A·G^{-1} (h=A·qd invariant) | column-reframe | mj subtreeVel 4.4e-15 |
   | cmm_time_variation Ȧ | column_reframe (hdot invariant) | column-reframe | 8.9e-15 |
   | end_effector_pose_gradient | column_reframe dpose·G^{-1} | column-reframe | FD-mjx 1.9e-10 |
   | **dccrba** dh_dq ⚠REVISED | dh_dq·G^{-1} + A_pin·Jv_q on base-ROT cols (Jv_q=_cross_cols(v_lin,−1)); dhdot_dq/dv/da same family | column-reframe + **vel-couple** | naive off 25→**1.5e-8** |
   | id/fd gradient, idsva_so, fdsva_so | full (rows+cols+couplings) | gradient/SO | done in oracle |
   | **end_effector_pose_hessian** ⚠REVISED | H_mjx[i,a,k]=Σ H_pin[i,b,c]G^{-1}[b,a]G^{-1}[c,k] + sym(dpose_pin·∂(G^{-1})/∂θ_k); ∂(G^{-1})/∂θ_a base-lin block=−[e_a]×R^T; **symmetrize** in the 2 tangent idx (GRiD analytic hess IS symmetric; raw Lie 2nd-deriv isn't) | tensor-slab + frame | FD-mjx 2.1e-10 |
   | **integrator** (euler+rk) | mjx retract: floating pos += dt·qvel[:3] GLOBAL (vs pin SE(3) V(φ), O(dt²) wrong) | retract | mj_integratePos **0.0** |
   | integrator_gradient | mjx_dIntegrate (base-lin→I, no lin↔ang) + fd value transform + minv_pin_to_mjx velocity rows | retract | FD-mjx 2.7e-10 |
   | plant_step | composes fd qdd transform (ω×v) + mjx retract | retract | mj_step (si_euler) 1e-16 |
   | plant_step_gradient | == integrator_gradient (byte-identical) | retract | diff 0.0 |
   | plant_step_hessian | retract Jacobian + fdsva_so SO transform, base-localized | retract | ~loc base-rot |
   | quadratic_state_cost | VALUE qd-convention-DEPENDENT (user supplies mjx x_des/Q); grad/hess qd-block reframe G | composite | 6.42 vs 3.92 |
   | ee_pos_cost / com_cost | VALUE invariant (geometric); grad base-translation COLUMN reframes by R | composite | ee grad 0.80→1.2e-10 |
   | momentum_cost | VALUE needs qd input-convert (geometric h); grad/hess qd reframe | composite | 75.55 vs 76.04 |
   | joint_*_barrier | pass-through; convention-free on JOINT DOF; base-reframe only if on base vel/torque/quat | pass-through | joint-only diff 0.0 |

   NOTE the recurring trap: any quantity evaluated at "mjx qacc=0" or reading qd has the ω×v
   acceleration coupling (`a_pin=−ω×v` when mjx accel=0). nonlinear_effects, dccrba, the gradients
   all carry it; generalized_gravity & forward_dynamics_parameter_gradient provably do NOT.
