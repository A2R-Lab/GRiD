"""Verbatim per-op doc comments for the generated torch/jax surface regions.

Extracted 1:1 from the hand-written wrapper_template.cu units when REGIONS
#4-6 landed (A2, 2026-09-11) — the emitters in wrapper_body_gen.py splice
these above each generated impl so the op documentation survives the
table-driven collapse. Editing THESE strings (then regenerating) is how the
docs change; the region text itself is never hand-edited.
"""
from __future__ import annotations

JAX_OP_DOCS: dict[str, str] = {
    'aba': (
        '// aba(q, qd, u, f_ext) → qdd  — same kernel signature shape as forward_dynamics\n'
    ),
    'ccrba': (
        '// ccrba(q, qd) → flat [A(6*NV); h(6)], 6*NV + 6  [no workspace, no gravity]\n'
        '// Single flat buffer; the Python layer splits the (A, h) tuple.\n'
    ),
    'cmm_time_variation': (
        '// cmm_time_variation(q, qd) → Adot, 6*NV  [d_workspace, no gravity]\n'
    ),
    'com': (
        '// com(q) → flat [p_com(3); J_com(3*NV)], 3 + 3*NV  [no workspace, no gravity]\n'
        '// Single flat buffer; the Python layer splits the (p_com, J_com) tuple.\n'
    ),
    'coriolis_matrix': (
        '// coriolis_matrix(q, qd) → C(q,qd), NV*NV row-major  [d_workspace, gravity]\n'
    ),
    'crba': (
        '// crba(q) → M  (kernel writes the full mass matrix; no symmetrize needed)\n'
    ),
    'dccrba': (
        '// dccrba(q) → dA/dq, 6*NV*NV  [d_workspace, no gravity]  (compressed d_q kernel)\n'
    ),
    'end_effector_pose': (
        '// end_effector_pose(q) → end_effector_pose  flat (B, 6*NUM_EES)\n'
    ),
    'end_effector_pose_gradient': (
        '// end_effector_pose_gradient(q) → end_effector_pose_gradient d/dv flat (B, 6*NUM_EES*NV).\n'
        '// Output convention: d/dv tangent (pinocchio); floating-base shape uses NV\n'
        '// (= 6 + n_joints) NOT NJ. Python side reshapes/transposes to the\n'
        '// (B, 6*NUM_EES, NV) row-major convention (see _handle.py).\n'
    ),
    'end_effector_pose_hessian': (
        '// end_effector_pose_hessian(q) → end_effector_pose_hessian  flat (B, 6*NUM_EES*NV*NV)\n'
        '// The kernel also writes d_end_effector_pose_gradient as a byproduct; we only return d2.\n'
    ),
    'energy': (
        '// energy(q, qd) → [KE, PE, KE+PE], 3  [gravity, no workspace]\n'
    ),
    'fdsva_so': (
        '// fdsva_so(q, qd, u) → packed (B, SECOND_ORDER_TENSOR_SIZE)\n'
        '// Uses d_idsva_so as scratch — must not run concurrently with idsva_so.\n'
    ),
    'forward_dynamics': (
        '// forward_dynamics(q, qd, u, f_ext) → qdd  (f_ext always passed; zeros if omitted)\n'
    ),
    'forward_dynamics_gradient': (
        '// forward_dynamics_gradient(q, qd, u) → df_du  flat (B, 2*NV*NV)\n'
        '// Python reshapes/transposes to (B, NV, 2*NV) (tangent-space; FIXED base\n'
        '// NV == NJ, FLOATING base NV < NJ).\n'
    ),
    'forward_dynamics_parameter_gradient': (
        '// forward_dynamics_parameter_gradient(q, qd, u) → dqdd/dpi = -Minv . Y\n'
        '// flat (B, NV*10*NUM_BODIES). Internally runs FD at (q,qd,u) and the regressor\n'
        '// at the resulting qdd, then applies -Minv (mirrors RBDReference). The kernel\n'
        '// takes d_workspace (the s_Y regressor scratch spills there at LITE/MINIMAL).\n'
    ),
    'frame_jacobian': (
        '// frame_jacobian(q) → 6*NV geometric Jacobian  [no workspace, no gravity, int attrs]\n'
    ),
    'frame_jacobian_dot': (
        '// frame_jacobian_dot(q, qd) → 6*NV d/dt Jacobian  [no workspace, no gravity, int attrs]\n'
    ),
    'generalized_gravity': (
        '// generalized_gravity(q) → g(q), NV  [d_workspace, gravity]\n'
    ),
    'inverse_dynamics': (
        '// inverse_dynamics(q, qd, qdd, f_ext) → c   — fully device-resident path.\n'
        '//\n'
        '// qdd and f_ext are ALWAYS passed as explicit device buffers from the Python\n'
        '// surface (JAX FFI has no optional-buffer support, so the wrapper passes zeros\n'
        '// when the caller omits them — mirroring idsva_so). qdd flows through the\n'
        '// separate d_qdd buffer + the USE_QDD overload of the kernel (signature\n'
        '// (d_c, d_q_qd, stride, d_qdd, d_f_ext, ...)); f_ext is copied D→D into d_f_ext.\n'
        "// MUJOCO templates the kernel's compile-time output-convention flag: MUJOCO=false\n"
        '// is the pinocchio path (byte-identical to the legacy handler); MUJOCO=true launches\n'
        '// the same kernel with MUJOCO_OUTPUT=true so the device code converts q/qd mjx->pin on\n'
        '// load and rotates the base-linear tau rows back to the mjx frame -- no host pre/post.\n'
        '// The mjx instantiation is FLOATING-base only (gated where the handler is defined).\n'
    ),
    'inverse_dynamics_gradient': (
        '// inverse_dynamics_gradient(q, qd, qdd) → dc_du  flat (B, 2*NV*NV)\n'
        '// Python reshapes/transposes to (B, NV, 2*NV) [dc_dq | dc_dqd] (tangent-space;\n'
        '// FIXED base NV == NJ, FLOATING base NV < NJ).\n'
        '//\n'
        '// qdd is ALWAYS passed as an explicit device buffer from the Python surface\n'
        '// (JAX FFI has no optional-buffer support, so the wrapper passes zeros when the\n'
        '// caller omits it — mirroring the VALUE inverse_dynamics FFI). ∂c/∂(q,qd)\n'
        "// depends on qdd via the M·qdd term's derivatives, so qdd flows through the\n"
        '// separate d_qdd buffer + the USE_QDD overload of the gradient kernel\n'
        '// (signature adds d_qdd after stride). A zero qdd is byte-identical to the old\n'
        '// no-qdd behaviour.\n'
    ),
    'inverse_dynamics_regressor': (
        '// inverse_dynamics_regressor(q, qd, qdd) → Y  flat (B, NV*10*NUM_BODIES).\n'
        "// qdd is passed explicitly (the bias regressor used by inverse_dynamics's VJP\n"
        '// passes zeros). The regressor kernel reads q|qd|qdd from d_q_qd_u (stride\n'
        '// Q_QD_U_STRIDE), the qdd occupying the u-slot — mirroring idsva_so.\n'
    ),
    'kinetic_energy_regressor': (
        '// kinetic_energy_regressor(q, qd) → y_KE, 10*NUM_BODIES  [gravity, no workspace]\n'
    ),
    'minv': (
        '// minv(q) → Minv  (kernel writes lower triangle only; symmetrize Python-side)\n'
    ),
    'nonlinear_effects': (
        '// nonlinear_effects(q, qd) → c(q,qd), NV  [d_workspace, gravity]\n'
    ),
    'osc_inertia': (
        '// osc_inertia(q) → 6x6 task inertia Lambda, 36  [d_workspace (tier-spill scratch), no gravity, frame baked at codegen]\n'
    ),
    'potential_energy_regressor': (
        '// potential_energy_regressor(q) → y_PE, 10*NUM_BODIES  [gravity, no workspace]\n'
    ),
}

TORCH_OP_DOCS: dict[str, str] = {
    'ccrba': (
        '// ccrba → single flat (B, 6*NV + 6); the Python layer splits the (A, h) tuple.\n'
    ),
    'com': (
        '// com → single flat (B, 3 + 3*NV); the Python layer splits the (p_com, J_com) tuple.\n'
    ),
    'frame_jacobian': (
        '// frame_jacobian / frame_jacobian_dot: target_jid + reference_frame trail the\n'
        '// schema as ints, passed straight through to the kernel (NO host -1 default\n'
        '// resolution — the Python torch surface passes resolved non-negative values).\n'
    ),
}

# Ops whose mujoco twin BIND carries the one-line convention doc
# (formulaic; text derived by the emitter).
JAX_TWIN_BIND_DOC_OPS: frozenset[str] = frozenset({
    'aba',
    'crba',
    'end_effector_pose',
    'end_effector_pose_gradient',
    'end_effector_pose_hessian',
    'fdsva_so',
    'forward_dynamics',
    'forward_dynamics_gradient',
    'inverse_dynamics',
    'inverse_dynamics_gradient',
    'inverse_dynamics_regressor',
    'minv',
})


# Section banners moved INTO the generated regions (verbatim from the
# hand-written layout); keyed by the op they precede.
SECTION_BANNERS: dict[str, str] = {
    'jax_regressor': (
        '// ────────────────────────────────────────────────────────────────────────────\n'
        '// Inertial-parameter regressor surface (PS2a — sysID autodiff)\n'
        '// ────────────────────────────────────────────────────────────────────────────\n'
        '//\n'
        '// Both outputs are row-major (NV x 10*NUM_BODIES) per timestep, where the\n'
        "// per-link 10-parameter basis is pi_i = [m, m*c(3), I_O(6)] (the parser's\n"
        '// origin-frame inertia; matches RBDReference._regressor). These back the\n'
        '// inertial-parameter VJP: tau = Y . pi so dtau/dpi = Y, and\n'
        '// dqdd/dpi = -Minv . Y. The Python custom_vjp contracts a cotangent ct (NV)\n'
        '// with these (NV x 10NB) Jacobians to produce the pi-cotangent (10NB).\n'
    ),
    'jax_ptier1': (
        '// ─── P-tier1: centroidal / energy / kinematics family (jax FFI) ──────────────\n'
        '//\n'
        '// These 13 handlers mirror the crba template EXACTLY: stage the device-resident\n'
        "// input(s) into the singleton's d_q_qd_u buffer (q at offset 0, qd at offset nj)\n"
        "// via cudaMemcpy2DAsync, launch the per-robot __global__ kernel DIRECTLY on JAX's\n"
        "// stream (no host round-trip), then D→D copy the flat result into JAX's output.\n"
        '//\n'
        '// Input layout (UNIFIED with crba / end_effector_pose): every kernel reads its\n'
        '// inputs from d_q_qd_u with stride = 3*NUM_JOINTS. The compressed-d_q kernels\n'
        '// (com/dccrba/pe_regressor/frame_jacobian/osc_inertia) only touch the first\n'
        '// NUM_JOINTS elements of each strided block, so packing q into the q-slot of\n'
        '// d_q_qd_u and passing stride=3*NJ feeds them the correct q — identical to how\n'
        '// the end_effector_pose handler launches the compressed-d_q ee kernel.\n'
        '//\n'
        '// d_workspace: passed ONLY for gravity / nle / dccrba / cmm / coriolis (per the\n'
        '// contract); OMITTED for the others (their kernels have no workspace arg).\n'
        '//\n'
        '// R2 (>48KB dynamic smem opt-in): NOT handled per-handler. grid_rbd_init() calls\n'
        '// grid::init_grid<T>() → init_grid_kernel_attrs<T>(), which issues the\n'
        '// cudaFuncSetAttribute(MaxDynamicSharedMemorySize) opt-in for EVERY emitted\n'
        '// algorithm kernel (coriolis/cmm/dccrba/com/ccrba/energy/... all enumerated\n'
        '// there). The crba/idsva_so handlers rely on the same warmup; these do too.\n'
    ),
    'jax_wave2': (
        '// ─── Wave 2: gated value ops (energy/com/ccrba/cmm/dccrba) ───────────────────\n'
    ),
    'jax_wave3': (
        '// ─── Wave 3: int-attr kinematics (frame_jacobian/dot, osc_inertia) ───────────\n'
        '//\n'
        '// frame_jacobian / frame_jacobian_dot take target_jid + reference_frame as\n'
        '// runtime int64 FFI attrs (cast to int for the kernel). UNLIKE the C-ABI host\n'
        '// wrapper, these handlers do NOT resolve a negative "use default" sentinel: the\n'
        '// kernel itself does not interpret target_jid<0 / reference_frame<0, so the\n'
        '// Python jax surface must pass already-resolved (non-negative) values (it reads\n'
        '// the leaf-EE / LOCAL_WORLD_ALIGNED defaults from the numpy handle). reference_frame:\n'
        '// 0=LOCAL, 1=WORLD, 2=LOCAL_WORLD_ALIGNED.\n'
    ),
    'torch_sysid': (
        '// ── inertial-parameter (sysID) regressor + FD parameter gradient ──\n'
        '// Mirror the JAX grid_rbd_jax_inverse_dynamics_regressor /\n'
        '// grid_rbd_jax_forward_dynamics_parameter_gradient handlers: same kernels, same\n'
        '// d_Y / d_dqdd_dpi / d_workspace scratch, same (B, NV*10*NUM_BODIES) row-major\n'
        '// output. These back the torch inertial-parameter VJP (tau = Y . pi so\n'
        '// dtau/dpi = Y, dqdd/dpi = -Minv . Y).\n'
    ),
    'torch_ptier1': (
        '// ─── P-tier1: centroidal / energy / kinematics family (torch ops) ────────────\n'
        '//\n'
        '// Mirror torch_crba EXACTLY: grid_torch_pack stages q (+qd) into d_q_qd_u, launch\n'
        '// the per-robot kernel directly on the current torch CUDA stream, D→D copy the\n'
        '// flat result into the empty output. Stride is always 3*NUM_JOINTS (the\n'
        '// compressed-d_q kernels only read the first NUM_JOINTS of each strided block).\n'
        '// d_workspace passed ONLY where the contract row says YES; gravity only where YES.\n'
        '// R2 smem opt-in: covered by grid_rbd_init → init_grid_kernel_attrs (see jax note).\n'
    ),
}
