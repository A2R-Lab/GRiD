"""Verbatim doc comments for the generated mjx twin bodies (P1 incr-4b).

Preserved from the hand-written originals when the twins were folded into
the generated region (wrapper_body_gen.gen_mjx_block). Keys are abi stems;
twins with no doc comment in the original (frame_jacobian_dot, osc_inertia
— covered by the frame_jacobian family note) simply have no entry.
"""

MJX_DOC: dict[str, str] = {
    "inverse_dynamics": "\n".join([
        '// MuJoCo-convention inverse dynamics (floating base only). Identical signature to',
        '// grid_rbd_inverse_dynamics, but q/qd/qdd are MuJoCo-native (quat wxyz, free-joint',
        '// velocity [v_lin GLOBAL; omega LOCAL]) and the returned tau is in the mjx frame.',
        '//',
        '// The mjx output convention is baked into the KERNEL via the compile-time',
        '// MUJOCO_OUTPUT=true template arg: the kernel converts the inputs mjx->pin on load',
        '// (quat reorder + base velocity/acceleration reframe) and rotates the base-linear',
        '// tau rows back to the mjx frame before saving — so NO host-side pre/post-process is',
        '// needed (this is the fast path that replaces pin-kernel + _mujoco.py rotation).',
        '//',
        '// qdd is REQUIRED: the qdd=0 "bias" path cannot represent mjx (mjx qacc=0 implies a',
        '// nonzero pin acceleration -omega x v — the nonlinear_effects accel-coupling), so a',
        '// null qdd returns rc=4. Callers wanting the mjx bias use nonlinear_effects instead.',
    ]),
    "minv": "\n".join([
        '// MuJoCo-convention direct mass-matrix inverse (floating base only): the kernel',
        '// reorders the quaternion and applies the congruence Minv_mjx = G^-T Minv_pin G^-1',
        '// on the base block (MUJOCO_OUTPUT=true). The native kernel writes a FULL DENSE',
        '// SYMMETRIC mjx Minv (both triangles), so NO host symmetrize and NO host',
        '// minv_pin_to_mjx post-process are needed.',
    ]),
    "forward_dynamics": "\n".join([
        '// MuJoCo-convention forward dynamics (floating base only). q/qd/u are MuJoCo-native;',
        '// the kernel converts inputs mjx->pin on load and maps the output acceleration',
        '// qdd[0:3] = R(qdd_pin + omega x v) back to the mjx frame (MUJOCO_OUTPUT=true) — no',
        '// host pre/post-process. f_ext is not reframed by the kernel input-convert, so the',
        '// _handle dispatch only takes this path when f_ext is null.',
    ]),
    "aba": "\n".join([
        '// MuJoCo-convention ABA (floating base only). Same accel_out convention as',
        '// forward_dynamics: q/qd/u raw mjx in, mjx-frame qdd out (MUJOCO_OUTPUT=true). f_ext',
        '// not reframed -> the _handle dispatch only uses this path when f_ext is null.',
    ]),
    "crba": "\n".join([
        '// MuJoCo-convention mass matrix (floating base only): M_mjx = G M_pin G^T, the',
        '// congruence baked into the kernel (MUJOCO_OUTPUT=true). q is MuJoCo-native (quat',
        '// wxyz); the kernel reorders the quaternion and applies the congruence on the base',
        '// block before saving, so NO host pre/post-process is needed.',
    ]),
    "end_effector_pose": "\n".join([
        '// MuJoCo-convention end_effector_pose(q) -> 6*NUM_EES per timestep. The pose is',
        '// frame-INVARIANT; the kernel (MUJOCO_OUTPUT=true) only converts the mjx-native q',
        '// (quaternion reorder, like osc_inertia). Output byte-equal to feeding the pin',
        '// kernel the pin-converted q.',
    ]),
    "end_effector_pose_gradient": "\n".join([
        '// MuJoCo-convention end_effector_pose Jacobian (q) -> 6*NUM_EES*NUM_VEL per',
        '// timestep. Column-reframe: q raw mjx in (kernel reorders the quaternion) and the',
        '// kernel (MUJOCO_OUTPUT=true) reframes the base-linear Jacobian columns before',
        '// saving (the column reframe acts on the NV axis cols 0:3).',
    ]),
    "inverse_dynamics_gradient": "\n".join([
        '// MuJoCo-convention inverse-dynamics gradient (floating base only). q/qd/qdd are',
        '// MuJoCo-native; the kernel converts inputs mjx->pin on load and applies the full',
        '// gradient convention transform (reframe + base-row rotate + ω×v couplings, with M',
        '// from an in-kernel crba reuse) so the returned dc/d(q,qd) is the mjx-frame gradient.',
        '// REQUIRES qdd (the mjx gradient is the with-qdd surface; a null qdd returns rc=4).',
    ]),
    "forward_dynamics_gradient": "\n".join([
        '// MuJoCo-convention forward-dynamics gradient (floating base only). q/qd/u are',
        '// MuJoCo-native; qdd is computed internally. The kernel converts inputs mjx->pin on',
        '// load and applies the full gradient convention transform (reframe + base-row rotate',
        '// + ω×v couplings) so the returned df/d(q,qd) is the mjx-frame gradient.',
    ]),
    "end_effector_pose_hessian": "\n".join([
        '// MuJoCo-convention end_effector_pose Hessian (q) -> 6*NUM_EES*NV*NV per timestep.',
        '// q is raw mjx (kernel reorders the quaternion); the kernel (MUJOCO_OUTPUT=true)',
        '// double-column-reframes the Hessian (J·G^{-1} on both tangent indices) and adds the',
        '// symmetrized base-rotation frame term before saving. Output is invariant-shaped.',
    ]),
    "idsva_so": "\n".join([
        '// MuJoCo-convention idsva_so(q, qd, qdd) -> 4*NV^3 (floating base only). q/qd/qdd are',
        '// raw mjx (kernel input-converts); the kernel (MUJOCO_OUTPUT=true) transforms all four',
        '// 2nd-order tensors to the mjx frame (explicit-analytic SO transform + dM_dq closed',
        '// form), reusing the id/crba/id-grad inners. The mjx kernel is register-heavy; the',
        '// post-launch error check surfaces a silent launch-config failure as rc!=0.',
    ]),
    "inverse_dynamics_regressor": "\n".join([
        '// MuJoCo-convention inverse_dynamics_regressor (floating base only). q/qd/qdd are raw',
        '// mjx (kernel input-converts); the regressor ROWS are tangent-indexed generalized',
        '// forces, so the base-LINEAR rows (0:3) rotate by R (Y_mjx[0:3] = R Y_pin[0:3]) -- the',
        '// same base-row rotate as id_tau. Baked via the MUJOCO_OUTPUT=true template flag.',
    ]),
    "fdsva_so": "\n".join([
        '// MuJoCo-convention fdsva_so(q, qd, u) -> 4*NV^3 (floating base only). q/qd/u raw mjx',
        '// (kernel input-converts); the kernel (MUJOCO_OUTPUT=true) transforms all four',
        '// 2nd-order forward-dynamics tensors to the mjx frame (explicit-analytic SO transform,',
        '// contravector output-map). Register-heavy; post-launch error check surfaces a silent',
        '// launch-config failure as rc!=0.',
    ]),
    "com": "\n".join([
        '// MuJoCo-convention com(q) -> [p_com(3); J_com(3 x NV)] per timestep. p_com is',
        '// INVARIANT; the J_com columns are reframed by the kernel (MUJOCO_OUTPUT=true): q',
        '// is MuJoCo-native (quat wxyz) and the kernel reorders the quaternion + applies the',
        '// column reframe before saving, so NO host pre/post-process is needed.',
    ]),
    "ccrba": "\n".join([
        '// MuJoCo-convention ccrba(q, qd) -> [A(6 x NV); h(6)] per timestep. The centroidal',
        '// momentum h is INVARIANT; the A columns are reframed by the kernel',
        '// (MUJOCO_OUTPUT=true). q/qd raw mjx in (kernel reorders the quaternion + reframes',
        '// qd), so NO host pre/post-process is needed.',
    ]),
    "energy": "\n".join([
        '// MuJoCo-convention energy(q, qd) -> [KE, PE, KE+PE] per timestep. The energies are',
        '// frame-INVARIANT; the kernel (MUJOCO_OUTPUT=true) just converts the mjx-native',
        '// inputs (quaternion reorder + qd reframe) so the energy is built correctly. Output',
        '// is byte-equal to feeding the pin kernel the pin-converted q/qd.',
    ]),
    "generalized_gravity": "\n".join([
        '// MuJoCo-convention generalized_gravity(q) -> g(q) (floating base only). q is raw',
        '// mjx (kernel reorders the quaternion); the kernel (MUJOCO_OUTPUT=true) base-rotates',
        '// the gravity output so the returned g is mjx-frame. Output is NUM_VEL invariant-shaped.',
    ]),
    "nonlinear_effects": "\n".join([
        '// MuJoCo-convention nonlinear_effects(q, qd) -> c(q,qd) (floating base only). q is',
        '// raw mjx (kernel reorders the quaternion). The kernel (MUJOCO_OUTPUT=true) injects',
        '// the accel-couple delta_a (base-linear = -(omega x v_lin)) via a zeroed s_qdd then',
        '// base-rotates the bias output, so the returned c is the mjx-frame qfrc_bias.',
    ]),
    "coriolis_matrix": "\n".join([
        '// MuJoCo-convention Coriolis matrix (floating base only): C_mjx = G C_pin G^T, a',
        '// congruence baked into the kernel (MUJOCO_OUTPUT=true). q/qd raw mjx in (kernel',
        '// reorders the quaternion + reframes qd), mjx-frame C out (nv x nv row-major).',
    ]),
    "kinetic_energy_regressor": "\n".join([
        '// MuJoCo-convention kinetic_energy_regressor(q, qd) -> length 10*NUM_BODIES y_KE.',
        '// The regressor is frame-INVARIANT; the kernel (MUJOCO_OUTPUT=true) only converts',
        '// the mjx-native inputs (quaternion reorder + qd reframe). Output is byte-equal to',
        '// feeding the pin kernel the pin-converted q/qd. (The MUJOCO_OUTPUT instantiation',
        '// only exists for floating, hence the GRID_RBD_WITH_MUJOCO gate.)',
    ]),
    "potential_energy_regressor": "\n".join([
        '// MuJoCo-convention potential_energy_regressor(q) -> length 10*NUM_BODIES y_PE.',
        '// Frame-INVARIANT; the kernel (MUJOCO_OUTPUT=true) only converts the mjx-native q',
        '// (quaternion reorder). Output byte-equal to feeding the pin kernel pin-converted q.',
    ]),
    "dccrba": "\n".join([
        '// MuJoCo-convention dccrba(q) -> 6*NV*NV dA/dq tensor (floating base only). q is raw',
        '// mjx (kernel reorders the quaternion); the kernel (MUJOCO_OUTPUT=true) double-reframes',
        '// the qd-column and q-tangent indices by G^{-1} and adds the base-rotation frame term',
        '// (using the in-kernel CMM value) before saving.',
    ]),
    "cmm_time_variation": "\n".join([
        '// MuJoCo-convention cmm_time_variation(q, qd) -> 6*NUM_VEL Adot (per timestep).',
        '// Column-reframe: q/qd raw mjx in (kernel reorders the quaternion + reframes qd)',
        '// and the kernel (MUJOCO_OUTPUT=true) reframes the Adot columns before saving.',
    ]),
    "frame_jacobian": "\n".join([
        '// MuJoCo-convention frame_jacobian family (floating base only). The geometric',
        '// Jacobian / its time-derivative are column-reframed J_mjx = J_pin G^{-1} in the',
        "// kernel (MUJOCO_OUTPUT=true). osc_inertia's value Lambda is frame-INVARIANT, but",
        '// its q input still needs the quaternion reordered (wxyz->xyzw) so the internal',
        '// J/Minv build correctly — the mjx kernel does that, so a raw-mjx q is handled here',
        '// rather than silently mis-built by the pin kernel. All take raw mjx inputs.',
    ]),
    "end_effector_pose_runtime": "\n".join([
        '// MuJoCo-convention end_effector_pose_runtime (floating base only). The kernel',
        '// input-converts q (quat reorder) on load; the pose VALUE is frame-INVARIANT (the',
        '// 6-vector [xyz; rpy] is the same world frame), so this matches the pin pose with',
        '// the mjx-reordered quaternion. Baked via the MUJOCO_OUTPUT=true host/kernel flag.',
    ]),
    "end_effector_pose_gradient_runtime": "\n".join([
        '// MuJoCo-convention end_effector_pose_gradient_runtime (floating base only). The',
        '// pose value is invariant, but the gradient is COLUMN-reframed: the base-linear',
        '// columns reframe by R^T (mjx base-linear velocity is global). Baked via',
        '// MUJOCO_OUTPUT=true. Output 6 x NUM_VEL col-major, like the pin variant.',
    ]),
    "integrator": "\n".join([
        '// MuJoCo-convention integrator (floating base only): the free-joint base POSITION',
        "// takes a GLOBAL additive step (mjx retract) instead of pin's SE(3) V(phi); the",
        '// base quaternion + joints integrate normally. q/qd raw mjx in (q wxyz, qd global),',
        '// x_kp1 raw mjx out (q wxyz). Baked into the kernel (MUJOCO_OUTPUT=true).',
    ]),
    "integrator_gradient": "\n".join([
        '// MuJoCo-convention integrator_gradient(q, qd, u, dt, it) -> dAB (2NV x 3NV) (floating',
        '// base only). q/qd/u raw mjx (kernel input-converts); the kernel (MUJOCO_OUTPUT=true)',
        '// transforms the discrete state-transition Jacobian to the mjx tangent (global-add',
        '// retract rows + G velocity reframe + input-conversion column couplings). EULER/SI-EULER',
        '// only (multistage static_asserts out). Register-heavy; post-launch error check.',
    ]),
}
