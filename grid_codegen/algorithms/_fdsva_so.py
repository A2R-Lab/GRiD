"""Second-order forward-dynamics derivatives (∂²qdd/∂x²).

NOTE: joint damping/friction contribute NOTHING to second-order derivatives, so
this file emits no damping term. The damping bias τ_damp = b·qd is linear in qd
(∂²/∂*∂* = 0); Coulomb friction f·sign(qd) is non-smooth (subgradient 0 a.e.).
They appear only in the FIRST-order inverse_dynamics_gradient output, which
fdsva_so already composes — never here.
"""

# Shared block-parallel emit primitives (also used by _idsva_so). See _mjx_blockpar.
from ._mjx_blockpar import bpfor as _bpfor, bpctrl as _bpctrl, stride_rc as _fbp

MEMORY_THRESHOLD = 8 # Max num joints for shared mem allocation of result


def _fdsva_so_use_world_idsva(self):
    """Route the composed idsva_so inner to the WORLD frame for floating-base OR
    spherical-joint robots (body-frame single-DoF contractions are wrong for a
    6×3 spherical motion subspace). Mirrors `_idsva_so_use_world_frame`. Byte-
    identical for cardinal fixed-base robots (predicate false)."""
    return self.robot.floating_base or self.robot.robot_has_spherical()


def gen_fdsva_so_contract(self):
	# construct the boilerplate and function definition
    n = self.robot.get_num_vel()
    inner_arena_size = 4 * n**3
    func_params = ["s_df2 are the second derivatives of forward dynamics WRT q,qd,tau", \
                "s_idsva_so are the second derivative tensors of inverse dynamics", \
                "s_Minv is the inverse mass matrix", \
                "s_df_du is the gradient of the forward dynamics", \
                "s_temp is the (shared) scratch buffer; size FDSVA_SO_INNER_SMEM_BYTES<T, SCRATCH_IN_SMEM>() bytes (= " + str(inner_arena_size) + "*sizeof(T) when SCRATCH_IN_SMEM, else 0)", \
                "d_workspace is the global scratch buffer; size FDSVA_SO_INNER_WORKSPACE_BYTES<T, SCRATCH_IN_SMEM>() bytes (= " + str(inner_arena_size) + "*sizeof(T) when !SCRATCH_IN_SMEM, else 0). Pass nullptr when SCRATCH_IN_SMEM", \
                "gravity is the gravity constant"]
    func_def_start = "void fdsva_so_contract("
    func_def_middle = "T *s_df2, T *s_idsva_so, T *s_Minv, T *s_df_du, "
    func_def_end = "T *s_temp, T *d_workspace, const T gravity) {"
    func_notes = ["Assumes works with IDSVA",
                  "Inline-CUDA users: SCRATCH_IN_SMEM selects where the 4*NV^3 scratch arena lives.",
                  "  true  -> s_temp (shared memory; fastest, current default).",
                  "  false -> d_workspace (global; frees shared memory for the caller's outer kernel).",
                  "The placement is the INNER's choice (made at the top of this function) so the",
                  "kernel/device caller just sizes both arenas from the exposed *_BYTES constants",
                  "and hands both pointers in. Codegen maps each RESOURCE_TIER to a SCRATCH_IN_SMEM",
                  "value per robot (small robots keep all tiers in smem; large robots spill)."]
    func_def_middle, func_params = self.gen_insert_helpers_func_def_params(func_def_middle, func_params, -3)
    func_def = func_def_start + func_def_middle + func_def_end
    self.gen_add_func_doc("Second Order of Forward Dynamics with Spatial Vector Algebra", func_notes, func_params, None)
    self.gen_add_code_line("template <typename T, bool SCRATCH_IN_SMEM = true>")
    self.gen_add_code_line("__device__")
    self.gen_add_code_line(func_def, True)

    n = self.robot.get_num_vel()
    NV = self.robot.get_num_vel()

    self.gen_add_code_line('// Second Derivatives of Inverse Dynamics')
    self.gen_add_code_line("T *d2tau_dqdq = &s_idsva_so[" + str(n*n*n*0) + "];" )
    self.gen_add_code_line("T *d2tau_dvdv = &s_idsva_so[" + str(n*n*n*1) + "];" )
    self.gen_add_code_line("T *d2tau_dvdq = &s_idsva_so[" + str(n*n*n*2) + "];" )
    self.gen_add_code_line("T *dM_dq = &s_idsva_so[" + str(n*n*n*3) + "];" )
    self.gen_add_code_line('\n\n')
    self.gen_add_code_line('// First Derivatives of Forward Dynamics')
    self.gen_add_code_line("T *s_df_dq = s_df_du; T *s_df_dqd = &s_df_du[" + str(n*n) + "];")
    self.gen_add_code_line('\n\n')
    self.gen_add_code_line('// Second Derivatives of Forward Dynamics')
    self.gen_add_code_line('T *d2a_dqdq = s_df2;')
    self.gen_add_code_line('T *d2a_dvdq = &s_df2[' + str(n*n*n) + '];')
    self.gen_add_code_line('T *d2a_dvdv = &s_df2[' + str(2*n*n*n) + '];')
    self.gen_add_code_line('T *d2a_dtdq = &s_df2[' + str(3*n*n*n) + '];')
    self.gen_add_code_line('\n\n')
    self.gen_add_code_line('// Temporary Variables. The 4*n^3 scratch arena placement is the INNER\'s')
    self.gen_add_code_line('// choice, keyed on SCRATCH_IN_SMEM: s_temp (shared) when true, d_workspace')
    self.gen_add_code_line('// (global) when false. A surgical-spill change is local here: repoint a')
    self.gen_add_code_line('// sub-buffer + update FDSVA_SO_INNER_{SMEM,WORKSPACE}_BYTES.')
    self.gen_add_code_line('T *inner_arena;')
    self.gen_add_code_line('if constexpr (SCRATCH_IN_SMEM) {', True)
    self.gen_add_code_line('(void)d_workspace;')
    self.gen_add_code_line('inner_arena = s_temp;')
    self.gen_add_end_control_flow()
    self.gen_add_code_line('else {', True)
    self.gen_add_code_line('(void)s_temp;')
    self.gen_add_code_line('inner_arena = d_workspace;')
    self.gen_add_end_control_flow()
    self.gen_add_code_line(f'T *inner_dq = inner_arena; // Inner term for d2a_dqdq')
    self.gen_add_code_line(f'T *inner_cross = inner_dq + {n**3}; // Inner term for d2a_dvdq (dM_dq*Minv)')
    self.gen_add_code_line(f'T *inner_tau = inner_cross + {n**3}; // Inner term for d2a_dtdq (d2tau_dvdq + dM_dq*da_dv)')
    self.gen_add_code_line(f'T *rot_dq = inner_tau + {n**3}; // Rotated (dM_dq*da_dq)^R term used to compute inner_dq')
    self.gen_add_code_line(f'\n\n')

    # Start inner term for d2a_dqdq
    self.gen_add_code_line('// Start inner term for d2a_dqdq & Fill out Minv')
    self.gen_add_parallel_loop("ind",str(n**3 + n*n))
    self.gen_add_code_line(f'int i = ind / {n*n} % {n}; int j = ind / {n} % {n}; int k = ind % {n};')
    self.gen_add_code_line(f'if (ind < {n**3}) {{', True)
    self.gen_add_code_line(f'inner_dq[ind] = dot_prod<T, {n}, {n}, 1>(&dM_dq[{n*n}*i + k], &s_df_dq[{n}*j]);')
    self.gen_add_code_line(f'rot_dq[i*{n*n} + k*{n} + j] = inner_dq[ind];')
    self.gen_add_end_control_flow()
    self.gen_add_code_line(f'else if (k > j) s_Minv[j*{n} + k] = s_Minv[k*{n} + j];')
    self.gen_add_end_control_flow()
    self.gen_add_sync()
    
    # 3Dx2D Tensor Computation defined as iLk,Lj->ijk
    self.gen_add_code_line('// Compute relevant inner subterms in parallel')
    self.gen_add_parallel_loop("ind",str(3*n**3))
    self.gen_add_code_line(f'int i = ind / {n*n} % {n}; int j = ind / {n} % {n}; int k = ind % {n};')
    # inner_dq is assembled in natural [i,j,k] layout but the final -Minv reduction
    # reads it jk-transposed (inner_dq[j + k*n], i.e. [L,k,j]); the dM_dq*da_dq
    # (inner_dq + rot_dq) part is jk-symmetric so the swap is a no-op there, but
    # d2tau_dqdq is NOT jk-symmetric for the 6-DoF FLOATING ROOT (its q-q columns
    # 3..5 carry a genuine asymmetry that the world-frame inner stores un-symmetrized).
    # So on floating base the d2tau_dqdq term must be added jk-transposed to land in
    # the layout the final reduction consumes. Fixed-base d2tau_dqdq is jk-symmetric
    # (1-DoF joints), so the natural index is kept there -> byte-identical fixed output.
    # Spherical (ball) joints route the composed idsva_so inner through the WORLD
    # frame (the body-frame inner's single-DoF S contractions are wrong for a 3-DoF
    # joint — see the idsva_so spherical world-routing precedent). The world inner
    # stores d2tau_dqdq un-symmetrized, so spherical must use the same jk-transposed
    # index as the floating base to land in the layout the final -Minv reduction reads.
    d2tau_dqdq_term = (f'd2tau_dqdq[i*{n*n} + k*{n} + j]'
                       if _fdsva_so_use_world_idsva(self)
                       else 'd2tau_dqdq[ind]')
    self.gen_add_code_line(f'if (ind < {n**3}) inner_dq[ind] += rot_dq[ind] + {d2tau_dqdq_term}; // Started with dM_dq*da_dq')
    self.gen_add_code_line(f'else if (ind < {2*n**3}) inner_cross[i*{n*n} + k*{n} + j] = dot_prod<T, {n}, {n}, 1>(&dM_dq[{n*n}*i + k], &s_df_dqd[{n}*j]) + d2tau_dvdq[i*{n*n} + j*{n} + k];')
    self.gen_add_code_line(f'else inner_tau[i*{n*n} + k*{n} + j] = dot_prod<T, {n}, {n}, 1>(&dM_dq[{n*n}*i + k], &s_Minv[{n}*j]);')
    self.gen_add_end_control_flow()
    self.gen_add_sync()

    # 2Dx3D tensor computation defined as iL,Ljk->ijk
    self.gen_add_code_line('// Multiply by -Minv to finish algorithm')
    # PERF NOTE: this 4*n^3 iL,Ljk->ijk contraction (each thread a serial n-element
    # dot, ~4*n^4 FMAs) is the hottest loop in fdsva_so_contract. Two rewrites were
    # evaluated and REJECTED:
    #  - cuBLASDx gemm: standalone-gemm wins (2.4-5.2x) don't survive in-kernel
    #    (register pressure with SO state live, non-gemm-friendly iL,Ljk strides,
    #    tier/smem pressure on already-spilled g1_floating). Profile in-context
    #    before retrying — may be bandwidth/sync-bound, not FMA-bound.
    #  - grid_linalg_dot_strided_coalesced: the contracted axis L is the buffer's
    #    stride-n^2 slowest index, so coalescing along L issues WORSE stride-n^2
    #    warp loads; and the block-cooperative primitive would serialize 4*n^3
    #    block-reductions, collapsing the current 4*n^3-way thread parallelism.
    # Left AS-IS; don't re-attempt without a layout that makes the contracted axis
    # contiguous AND avoids one-dot-per-block serialization.
    self.gen_add_parallel_loop("ind",str(4*n**3))
    self.gen_add_code_line(f'int i = ind / {n*n} % {n}; int j = ind / {n} % {n}; int k = ind % {n};')
    self.gen_add_code_line(f'if (ind < {n**3}) d2a_dqdq[i*{n*n} + j*{n} + k] = -dot_prod<T, {n}, {n}, {n*n}>(&s_Minv[i], &inner_dq[j + k*{n}]);')
    self.gen_add_code_line(f'else if (ind < {2*n**3}) d2a_dvdq[i*{n*n} + j*{n} + k] = -dot_prod<T, {n}, {n}, {n*n}>(&s_Minv[i], &inner_cross[j + k*{n}]);')
    self.gen_add_code_line(f'else if (ind < {3*n**3}) d2a_dvdv[i*{n*n} + j*{n} + k] = -dot_prod<T, {n}, {n}, {n*n}>(&s_Minv[i], &d2tau_dvdv[j + k*{n}]);')
    self.gen_add_code_line(f'else d2a_dtdq[i*{n*n} + j*{n} + k] = -dot_prod<T, {n}, {n}, {n*n}>(&s_Minv[i], &inner_tau[j + k*{n}]);')
    self.gen_add_end_control_flow()
    self.gen_add_sync()

    self.gen_add_end_function()

def gen_fdsva_so_contract_temp_mem_size(self):
    n = self.robot.get_num_vel()
    return 4*n**3

def gen_fdsva_so_fd_gradient_inline_temp_mem_size(self):
    n = self.robot.get_num_vel()
    return 18*self.robot.get_num_joints() + 2*n*n + self.gen_inverse_dynamics_gradient_inner_temp_mem_size()


def gen_fdsva_so_fd_gradient_inline_temp_mem_size_spilled(self):
    """MEM1 variant: shared-mem footprint when inverse_dynamics_gradient_gradient_inner uses spill.

    Same prefix layout (s_fd_vaf + s_fd_dc_du) but s_fd_temp is sized for
    `selective_shared_count` (= full minus the spilled da_dq..fxvi band)
    instead of full. inverse_dynamics_inner_vaf also needs s_fd_temp but
    its temp size (6*NUM_POS) is tiny — max() handles it.
    """
    n = self.robot.get_num_vel()
    layout = self.gen_inverse_dynamics_gradient_temp_layout()
    id_inner_vaf_temp = self.gen_inverse_dynamics_inner_temp_mem_size()
    s_fd_temp_size = max(id_inner_vaf_temp, layout["selective_shared_count"])
    return 18*self.robot.get_num_joints() + 2*n*n + s_fd_temp_size

def gen_fdsva_so_fd_gradient_inline(self, use_spill = False, spill_ptr_expr = "nullptr"):
    """Emit the inline FD-gradient computation.

    MEM1 (use_spill=True path): when the kernel is shared-mem-pressured (big
    floating-base robots like g1), have the inner inverse_dynamics_gradient_gradient_inner spill
    its da_dq..fxvi block to ``spill_ptr_expr`` (typically a slice of
    d_workspace). This matches the well-tested spill pattern used by
    inverse_dynamics_gradient_kernel — the WHOLE temp is NOT pushed to global; only the
    overflow band. s_fd_vaf/s_fd_dc_du/s_fd_temp still live in shared
    `s_temp`, but s_fd_temp is sized for ``selective_shared_count``, not
    ``full_count``. Saves the bulk of fd_grad_inline's ~40k-float footprint
    for NV=35 robots.
    """
    n = self.robot.get_num_vel()
    self.gen_add_code_line("// Compute FD gradient inline so nested device wrappers do not carve a second shared arena")
    self.gen_add_code_line("T *s_fd_vaf = s_temp;")
    self.gen_add_code_line(f"T *s_fd_dc_du = s_fd_vaf + {18*self.robot.get_num_joints()};")
    self.gen_add_code_line(f"T *s_fd_temp = s_fd_dc_du + {2*n*n};")
    self.gen_inverse_dynamics_inner_function_call(
        compute_c = False,
        use_qdd_input = True,
        updated_var_names = dict(
            s_vaf_name = "s_fd_vaf",
            s_q_name = "s_q",
            s_qd_name = "s_qd",
            s_qdd_name = "s_qdd",
            s_temp_name = "s_fd_temp",
            d_f_ext_name = "nullptr",  # fdsva_so does not support external forces
            gravity_name = "gravity",
        ),
    )
    self.gen_inverse_dynamics_gradient_inner_function_call(
        updated_var_names = dict(
            s_dc_du_name = "s_fd_dc_du",
            s_vaf_name = "s_fd_vaf",
            s_q_name = "s_q",
            s_qd_name = "s_qd",
            s_temp_name = "s_fd_temp",
            d_temp_spill_name = spill_ptr_expr if use_spill else "nullptr",
            temp_spill_flag_name = "true" if use_spill else "false",
            gravity_name = "gravity",
        ),
    )
    self.gen_add_parallel_loop("ind", str(2*n*n))
    self.gen_add_code_line(f"int row = ind % {n}; int dc_col_offset = ind - row;")
    self.gen_add_code_line("T val = static_cast<T>(0);")
    self.gen_add_code_line(f"for(int col = 0; col < {n}; col++) {{", True)
    self.gen_add_code_line(f"int index = (row <= col) * (col * {n} + row) + (row > col) * (row * {n} + col);")
    self.gen_add_code_line("val += s_Minv[index] * s_fd_dc_du[dc_col_offset + col];")
    self.gen_add_end_control_flow()
    self.gen_add_code_line("s_df_du[ind] = -val;")
    self.gen_add_end_control_flow()
    self.gen_add_sync()
    
def gen_fdsva_so_contract_function_call(self, updated_var_names = None,
                                     scratch_in_smem_expr = "true"):
    var_names = dict( \
        s_df2_name = "s_df2", \
        s_idsva_so_name = "s_idsva_so", \
        s_Minv_name = "s_Minv", \
        s_df_du_name = "s_df_du", \
        s_temp_name = "s_temp", \
        d_workspace_name = "nullptr", \
        gravity_name = "gravity"
    )
    if updated_var_names is not None:
        for key,value in updated_var_names.items():
            var_names[key] = value
    fdsva_so_code_start = "fdsva_so_contract<T, " + scratch_in_smem_expr + ">(" + var_names["s_df2_name"] + ", " + var_names["s_idsva_so_name"] + ", " + var_names["s_Minv_name"] + ", " + var_names["s_df_du_name"] + ", "
    fdsva_so_code_end = var_names["s_temp_name"] + ", " + var_names["d_workspace_name"] + ", " + var_names["gravity_name"] + ");"
    fdsva_so_code_middle = self.gen_insert_helpers_function_call()
    fdsva_so_code = fdsva_so_code_start + fdsva_so_code_middle + fdsva_so_code_end
    self.gen_add_code_line(fdsva_so_code)

def gen_fdsva_so_device_function_call(self,
                                          scratch_in_smem_expr = "true",
                                          fd_grad_use_spill_expr = "false",
                                          contract_in_smem_expr = "true",
                                          d_workspace_pool_name = "nullptr",
                                          d_fd_grad_spill_name = "nullptr",
                                          s_fdsva_temp_name = "nullptr",
                                          mujoco_output_expr = None,
                                          d_mjx_scratch_name = "nullptr",
                                          idsva_cold_in_smem_expr = "true",
                                          d_idsva_cold_spill_name = "nullptr"):
    """Emit the call to `fdsva_so_device`. Arg order MUST match the def in
    gen_fdsva_so_device. Pool/spill regions default to nullptr (unused under
    the matching if-constexpr); the kernel passes real pointers per tier.
    mujoco_output_expr (floating non-mimic/skew): the trailing MUJOCO_OUTPUT
    template arg; None -> 4-arg template (byte-identical for non-mjx kernels).
    d_mjx_scratch_name: the mjx-epilogue scratch arena (only forwarded when the
    mjx template arg is present so the non-mjx signature stays byte-identical).
    A4 — idsva_cold_in_smem_expr: the trailing IDSVA_COLD_IN_SMEM template arg (the
    WORLD idsva inner's cold-trio placement). Appended LAST (after MUJOCO_OUTPUT when
    present); only emitted when != "true" so the legacy template stays byte-identical.
    d_idsva_cold_spill_name: the SO-temp d_workspace region backing the cold trio (only
    appended when idsva_cold_in_smem_expr != "true", to keep the legacy signature)."""
    cold = (idsva_cold_in_smem_expr != "true")
    mjx_tmpl = ("" if mujoco_output_expr is None else ", " + mujoco_output_expr)
    # IDSVA_COLD_IN_SMEM is the last template param (after MUJOCO_OUTPUT). When the cold
    # rung is active but the kernel is NON-mjx, the template skips the (absent) MUJOCO
    # slot entirely and IDSVA_COLD is the 5th arg; when mjx, it is the 6th — either way
    # it is appended after whatever precedes it, matching the def's positional ordering.
    cold_tmpl = (", " + idsva_cold_in_smem_expr) if cold else ""
    tmpl = "<T, " + scratch_in_smem_expr + ", " + fd_grad_use_spill_expr + ", " + contract_in_smem_expr + mjx_tmpl + cold_tmpl + ">"
    start = "fdsva_so_device" + tmpl + "(s_df2, s_idsva_so, s_Minv, s_df_du, s_qdd, s_q, s_qd, s_u, "
    middle = self.gen_insert_helpers_function_call()
    mjx_arg = (", " + d_mjx_scratch_name) if mujoco_output_expr is not None else ""
    # The cold-spill pointer is the last call arg (after d_mjx_scratch when present).
    # When the cold rung is active on a NON-mjx kernel, d_mjx_scratch is absent from the
    # def, so d_idsva_cold_spill is the FIRST defaulted trailing pointer -> pass it
    # directly (positional). When mjx, it follows d_mjx_scratch (always passed by the
    # mjx kernel). Only emit when active, so the legacy call stays byte-identical.
    cold_arg = (", " + d_idsva_cold_spill_name) if cold else ""
    end = ("s_temp, " + d_workspace_pool_name + ", " + d_fd_grad_spill_name + ", "
           + s_fdsva_temp_name + ", d_robotModel, gravity" + mjx_arg + cold_arg + ");")
    self.gen_add_code_line(start + middle + end)

def gen_fdsva_so_device(self):
    """Emit `fdsva_so_device` — the whole fdsva_so orchestration as ONE
    inner that OWNS its scratch (s_temp) placement (inner-owns-placement; see
    docs/idsva_so_inner_refactor_notes.md). It wraps, in order:
      load_update_XImats -> minv_inner -> forward_dynamics_inner ->
      fd-gradient-inline -> idsva_so_{world,body}_inner -> fdsva_so_contract.
    Because the s_temp repoint happens at the very top, EVERY consumer below —
    including the XImats helper's sincos scratch — follows the placement, so the
    kernel never repoints s_temp from the outside.

    Template flags:
      SCRATCH_IN_SMEM   : the shared s_temp pool lives in smem (true) or routes
                          to d_workspace (false). This is the dominant lever — on
                          big floating humanoids the fd-gradient pool (~180 KB)
                          is what overflows, so false makes it fit.
      FD_GRAD_USE_SPILL : the fd-gradient inline spills its da_dq..fxvi band to
                          d_fd_grad_spill (only meaningful when the pool is smem).
      CONTRACT_IN_SMEM  : the fdsva_so_contract 4*NV^3 contraction scratch placement.

    Pointer params are caller-supplied (the kernel decides where the OUTPUTS and
    df_du/Minv live — smem, device arrays, or workspace bands — and hands in the
    spill regions): only the s_temp POOL placement is the inner's call."""
    n = self.robot.get_num_vel()
    func_params = [
        "s_df2/s_idsva_so/s_Minv/s_df_du/s_qdd/s_q/s_qd/s_u are the in/out buffers (caller places)",
        "s_temp is the shared scratch pool (used when SCRATCH_IN_SMEM)",
        "d_workspace is the global scratch pool (used when !SCRATCH_IN_SMEM)",
        "d_fd_grad_spill is the fd-gradient da_df spill band (used when FD_GRAD_USE_SPILL)",
        "s_fdsva_temp is the 4*NV^3 contraction scratch (smem or workspace per CONTRACT_IN_SMEM)",
        "d_robotModel holds XImats/topology; gravity is the gravity constant",
    ]
    func_def_start = "void fdsva_so_device("
    func_def_middle = ("T *s_df2, T *s_idsva_so, T *s_Minv, T *s_df_du, T *s_qdd, "
                       "const T *s_q, const T *s_qd, const T *s_u, ")
    # MUJOCO_OUTPUT (floating non-mimic/skew): the mjx epilogue recomputes Minv /
    # qdd / the fd gradient FRESH into d_mjx_scratch (a carve of the SO-temp region
    # of d_workspace, dead post-contract) so it NEVER reads the possibly-spilled
    # s_Minv / s_df_du / s_qdd (the §1g/§1h liveness bug). Defaulted nullptr; only
    # read under if constexpr(MUJOCO_OUTPUT). Mirrors idsva_so's d_mjx_scratch.
    # A4: d_idsva_cold_spill is the SO-temp d_workspace sub-region that backs the
    # composed WORLD idsva_so inner's cold trio (Xdown/v_w/a_w) when IDSVA_COLD_IN_SMEM
    # is false (the surgical cold rung). Appended LAST + defaulted nullptr so every
    # existing positional call site is unaffected. Only read when !IDSVA_COLD_IN_SMEM.
    mjx_inner_sig = self.robot.floating_base and not (self.robot_has_mimic_joints() or self.robot.robot_has_skew_axis())
    if mjx_inner_sig:
        func_def_end = ("T *s_temp, T *d_workspace, T *d_fd_grad_spill, T *s_fdsva_temp, "
                        "const robotModel<T> *d_robotModel, const T gravity, T *d_mjx_scratch = nullptr, "
                        "T *d_idsva_cold_spill = nullptr) {")
        func_params.append("d_mjx_scratch is the mjx-epilogue scratch arena (SO-temp region of d_workspace; only read when MUJOCO_OUTPUT)")
    else:
        func_def_end = ("T *s_temp, T *d_workspace, T *d_fd_grad_spill, T *s_fdsva_temp, "
                        "const robotModel<T> *d_robotModel, const T gravity, "
                        "T *d_idsva_cold_spill = nullptr) {")
    func_params.append("d_idsva_cold_spill is the SO-temp d_workspace region backing the WORLD idsva_so inner's cold trio (only read when !IDSVA_COLD_IN_SMEM)")
    func_def_middle, func_params = self.gen_insert_helpers_func_def_params(func_def_middle, func_params, -2)
    func_def = func_def_start + func_def_middle + func_def_end
    self.gen_add_func_doc("fdsva_so orchestration as a single inner-owns-placement device function",
                          [], func_params, None)
    # MUJOCO_OUTPUT (floating non-mimic/skew): compile-time mjx output-convention
    # flag, appended LAST so existing positional <T,SCRATCH,SPILL,CONTRACT> call
    # sites are unaffected; default false -> the epilogue if-constexpr-elides to
    # byte-identical PTX. Mirrors the idsva_so flag exactly.
    # A4: IDSVA_COLD_IN_SMEM (default true) — when false, the composed WORLD idsva_so
    # inner spills its cold trio (Xdown/v_w/a_w) to d_idsva_cold_spill (COLD_IN_SMEM=false)
    # while the hot pool stays in smem. Appended LAST (after MUJOCO_OUTPUT when present)
    # so existing positional <T,SCRATCH,SPILL,CONTRACT[,MUJOCO]> call sites are unaffected;
    # default true -> the world inner gets COLD_IN_SMEM=true (byte-identical to today).
    mjx_inner = self.robot.floating_base and not (self.robot_has_mimic_joints() or self.robot.robot_has_skew_axis())
    if mjx_inner:
        self.gen_add_code_line("template <typename T, bool SCRATCH_IN_SMEM = true, bool FD_GRAD_USE_SPILL = false, bool CONTRACT_IN_SMEM = true, bool MUJOCO_OUTPUT = false, bool IDSVA_COLD_IN_SMEM = true>")
    else:
        self.gen_add_code_line("template <typename T, bool SCRATCH_IN_SMEM = true, bool FD_GRAD_USE_SPILL = false, bool CONTRACT_IN_SMEM = true, bool IDSVA_COLD_IN_SMEM = true>")
    # __forceinline__ so the whole orchestration inlines into the calling kernel.
    # Under -rdc (single-call/anti-LICM build) a separate __device__ wrapper keeps
    # its callees (e.g. minv_inner, ~108 regs) as distinct functions whose
    # regcount must fit the kernel's launch_bounds budget (80 at LITE / 64 at
    # MINIMAL) -> ptxas regcount error. Inlining folds them into the kernel (as the
    # pre-refactor inline orchestration did). See HANDOFF.md / "Problem 1".
    self.gen_add_code_line("__device__ __forceinline__")
    self.gen_add_code_line(func_def, True)
    # Inner owns the pool placement; the repoint covers every consumer below
    # (incl. the XImats helper's sincos scratch), so no caller-side repoint.
    self.gen_add_code_line("if constexpr (!SCRATCH_IN_SMEM) { s_temp = d_workspace; } else { (void)d_workspace; }")
    # A4: d_temp_spill is the WORLD idsva inner's d_workspace param for its cold trio.
    # When IDSVA_COLD_IN_SMEM (the default + pool-in-smem rungs), the inner keeps the
    # cold trio in smem and never touches d_temp_spill (it stays nullptr). When the
    # surgical cold rung sets IDSVA_COLD_IN_SMEM=false, the inner repoints Xdown/v_w/a_w
    # to d_idsva_cold_spill (a SO-temp d_workspace carve, disjoint from the contract /
    # df_du / Minv bands — see §1g argument in the kernel body).
    self.gen_add_code_line("T *d_temp_spill = nullptr;")
    self.gen_add_code_line("if constexpr (!IDSVA_COLD_IN_SMEM) { d_temp_spill = d_idsva_cold_spill; } else { (void)d_idsva_cold_spill; }")
    self.gen_load_update_XImats_helpers_function_call()
    self.gen_minv_inner_function_call(f_in_smem_expr = "true")
    self.gen_add_code_line("forward_dynamics_inner<T, true>(s_qdd, s_q, s_qd, s_u, " + self.gen_insert_helpers_function_call() + "s_temp, nullptr, nullptr, gravity);")
    self.gen_add_sync()
    # fd-gradient inline; the band-spill variant is a compile-time choice.
    self.gen_add_code_line("if constexpr (FD_GRAD_USE_SPILL) {", True)
    self.gen_fdsva_so_fd_gradient_inline(use_spill=True, spill_ptr_expr="d_fd_grad_spill")
    self.gen_add_end_control_flow()
    self.gen_add_code_line("else {", True)
    self.gen_add_code_line("(void)d_fd_grad_spill;")
    self.gen_fdsva_so_fd_gradient_inline(use_spill=False, spill_ptr_expr="nullptr")
    self.gen_add_end_control_flow()
    # Spherical (ball) joints route the composed idsva_so inner to the WORLD frame
    # (the body-frame inner's single-DoF S contractions are wrong for a 3-DoF joint),
    # exactly as the idsva_so dispatcher does for spherical. The world inner needs no
    # dvdq layout repair (that is a body-frame-only fixup).
    if _fdsva_so_use_world_idsva(self):
        # A4: thread the surgical cold-trio placement to the WORLD inner. Default true
        # (every legacy/non-cold rung) -> COLD_IN_SMEM=true (byte-identical). The cold
        # rung passes IDSVA_COLD_IN_SMEM=false + d_temp_spill (= d_idsva_cold_spill).
        self.gen_idsva_so_world_frame_inner_function_call(cold_in_smem_expr="IDSVA_COLD_IN_SMEM")
    else:
        # Body-frame (fixed-base non-spherical): keep existing defaults — the body inner
        # has no exposed cold trio (its surgical levers are BC/TP). A4 leaves it untouched.
        self.gen_idsva_so_body_frame_inner_function_call()
        self.gen_idsva_so_body_frame_public_dvdq_layout_repair()
    self.gen_fdsva_so_contract_function_call(
        updated_var_names = dict(d_workspace_name = "s_fdsva_temp"),
        scratch_in_smem_expr = "CONTRACT_IN_SMEM")
    # ---- mjx output-convention epilogue (floating non-mimic/skew only) ----
    # Runs at the very END, where the contract scratch (s_temp pool when
    # CONTRACT_IN_SMEM, else s_fdsva_temp) is DEAD and s_df2 holds the finalized
    # pin SO tensors. The epilogue must NOT read s_Minv / s_df_du / s_qdd directly:
    # on spill tiers those live in d_workspace and may be CLOBBERED before the
    # epilogue runs (the §1g/§1h liveness bug — fdsva_so was guarded for it). Mirror
    # idsva_so EXACTLY: recompute Minv / qdd / dqdd_dq|dqdd_dqd FRESH into a disjoint
    # d_mjx_scratch band (carve of d_workspace's SO-temp region, dead post-contract),
    # then run the per-k assembly register-locally (R + base vectors rebuilt per loop
    # body, nothing staged to shared scratch) parallel over k, and copy the mjx band
    # back over s_df2 block-parallel.
    if mjx_inner:
        self.gen_add_code_line("if constexpr (MUJOCO_OUTPUT) {", True)
        _emit_fdsva_so_mjx_output(self)
        self.gen_add_end_control_flow()
    self.gen_add_end_function()


def _emit_fdsva_so_mjx_locals_lines(self, nv, nv3):
    """Register-/stack-local recompute of the loop-invariant helpers, emitted at
    the TOP of every parallel-loop body (mirrors _emit_idsva_so_mjx_locals_lines).
    SAFE by construction: R is rebuilt from the base quaternion in s_q; the base
    source vectors from s_qd/s_u and the FRESH fd value qdd in s_mjx_qdd; the
    tensor + matrix pointers are plain offsets into s_df2 / s_mjx_* — NOTHING is
    staged into shared scratch (no aliasing risk vs the spilled buffers). A few
    dozen flops per thread. The R-build math is copied verbatim from the idsva_so
    epilogue / the id-gradient reference."""
    nv2 = nv * nv
    return [
        # R (row-major R[3*i+j]) from the xyzw base quaternion s_q[3..6] — matches
        # the idsva_so epilogue / mujoco_convention.rotation_from_quat_xyzw exactly.
        "T qx = s_q[3], qy = s_q[4], qz = s_q[5], qw = s_q[6];",
        "T xx = qx*qx, yy = qy*qy, zz = qz*qz;",
        "T xy = qx*qy, xz = qx*qz, yz = qy*qz, wx = qw*qx, wy = qw*qy, wz = qw*qz;",
        "T R[9];",
        "R[0] = static_cast<T>(1) - static_cast<T>(2)*(yy+zz); R[1] = static_cast<T>(2)*(xy-wz);                    R[2] = static_cast<T>(2)*(xz+wy);",
        "R[3] = static_cast<T>(2)*(xy+wz);                    R[4] = static_cast<T>(1) - static_cast<T>(2)*(xx+zz); R[5] = static_cast<T>(2)*(yz-wx);",
        "R[6] = static_cast<T>(2)*(xz-wy);                    R[7] = static_cast<T>(2)*(yz+wx);                    R[8] = static_cast<T>(1) - static_cast<T>(2)*(xx+yy);",
        "T v_lin[3]   = {s_qd[0], s_qd[1], s_qd[2]};",
        "T omega[3]   = {s_qd[3], s_qd[4], s_qd[5]};",
        "T u_lin[3]   = {s_u[0], s_u[1], s_u[2]};",
        # FRESH fd value qdd (recomputed into s_mjx_qdd; NOT the possibly-spilled s_qdd).
        "T qdd_lin[3] = {s_mjx_qdd[0], s_mjx_qdd[1], s_mjx_qdd[2]};",
        # FRESH fd gradient (col-major) + FRESH dense symmetric Minv — all in the
        # disjoint d_mjx_scratch band, never the spilled s_df_du / s_Minv. The slab
        # formulas read these by the names s_dqdd_dq / s_dqdd_dqd / s_Minv, so alias.
        "T *s_dqdd_dq  = s_mjx_dfdu;                 // col-major dqdd_dq  [c*NV+r]",
        "T *s_dqdd_dqd = s_mjx_dfdu + " + str(nv2) + ";   // col-major dqdd_dqd [c*NV+r]",
        "T *s_Minv     = s_mjx_Minv;                 // dense symmetric Minv (fresh)",
        "// pin tensor blocks in s_df2 (row-major [(i*NV+j)*NV+k]):",
        "T *T_d2q   = s_df2 + " + str(0 * nv3) + ";   // daba_dqdq",
        "T *T_cross = s_df2 + " + str(1 * nv3) + ";   // daba_dvdq [i,qd,q]",
        "T *T_d2qd  = s_df2 + " + str(2 * nv3) + ";   // daba_dvdv",
        "T *T_dtdq  = s_df2 + " + str(3 * nv3) + ";   // daba_dtdq = dMinv/dq [i,u,q]",
        "// mjx output blocks (same layout/order):",
        "T *O_d2q   = s_mjx_out + " + str(0 * nv3) + ";",
        "T *O_cross = s_mjx_out + " + str(1 * nv3) + ";",
        "T *O_d2qd  = s_mjx_out + " + str(2 * nv3) + ";",
        "T *O_dtdq  = s_mjx_out + " + str(3 * nv3) + ";",
    ]


def _emit_fdsva_so_mjx_output(self):
    """Emit the MuJoCo (mjx) output-convention epilogue for fdsva_so, transforming
    the 4 pin second-order tensors held in ``s_df2`` to the mjx convention IN PLACE.
    Floating-base (non-mimic/skew) only; runs at the END of ``fdsva_so_device``
    where the contract scratch is dead.

    ``s_df2`` is 4 contiguous NV^3 ROW-major blocks ``[i*NV*NV + j*NV + k]``:
      [0] daba_dqdq[i,j,k]  [1] daba_dvdq[i,qd,q] (=cross)  [2] daba_dvdv[i,j,k]
      [3] daba_dtdq[i,u,q]  (= dMinv[i,u]/dq, since dqdd/du = Minv).
    Matches the RBDReference.fdsva_so tuple order EXACTLY.

    LIVENESS (the §1g/§1h fix — mirrors idsva_so EXACTLY): the assembly needs the
    fd value qdd, the dense symmetric Minv, and the fd gradient dqdd_dq|dqdd_dqd.
    On spill tiers the in-flight s_qdd / s_Minv / s_df_du live in d_workspace and
    may be CLOBBERED before this runs, so we RECOMPUTE all three FRESH into a
    disjoint ``d_mjx_scratch`` band (a carve of d_workspace's SO-temp region, dead
    after the contract) and read ONLY from there. s_XImats/s_q/s_qd/s_u are live.

    d_mjx_scratch layout (all dead post-assembly):
      [0,         NV*NV)        s_M       (dense mass matrix, crba_inner)
      [NV*NV,   3*NV*NV)        s_dc_du   (pin dtau_dq | dtau_dqd, col-major)
      [3*NV*NV, 4*NV*NV)        s_mjx_Minv(dense symmetric Minv, inverse of s_M)
      [4*NV*NV, 6*NV*NV)        s_mjx_dfdu(fd gradient dqdd_dq | dqdd_dqd, col-major)
      [6*NV*NV, 6*NV*NV+NV)     s_mjx_qdd (fd value qdd)
      [VAFOFF,  VAFOFF+18*NJ)   s_vaf     (id-value intermediate band)
      [INNERTMP,INNERTMP+IDG)   reused-inner scratch (id/crba/id-grad/invert temps)
      [OUT,     OUT+4*NV^3)     mjx output band (copied back over s_df2 at the end)

    The per-k assembly (explicit form, forward-dynamics analog of idsva_so's) is
    transcribed verbatim from docs/open-tasks/mjx_proto/proto_fdsva_so_emit_spec.py
    (validated <1e-13 vs second_order_fd_pin_to_mjx)."""
    nv = self.robot.get_num_vel()
    NJ = self.robot.get_num_joints()
    nv2 = nv * nv
    nv3 = nv * nv * nv
    M_off       = 0
    DCDU_off    = nv2            # dtau_dq at [DCDU_off + c*nv + r]; dtau_dqd at [DCDU_off + nv2 + ...]
    MINV_off    = 3 * nv2
    DFDU_off    = 4 * nv2        # fd gradient dqdd_dq | dqdd_dqd (col-major)
    QDD_off     = 6 * nv2        # fd value qdd (nv)
    VAF_off     = 6 * nv2 + nv
    vaf_band    = 18 * NJ
    INNERTMP_off = VAF_off + vaf_band
    idgrad_temp = self.gen_inverse_dynamics_gradient_inner_temp_mem_size()
    OUT_off     = INNERTMP_off + idgrad_temp
    # Block-parallel per-k assembly: the 6 per-k nv^2 matrices (work1/work2/inner_u +
    # the sensitivities d_dq/d_dqd/d_Mi) are BLOCK-SHARED (whole block cooperates on
    # one k), carved after the 4*NV^3 output band. Per-k scalars (R, jqk/jvk/juk/jvvk/
    # juuk, Rd, d_qd[0:3]) stay per-thread register-local. This removes the 6*nv^2
    # (=1944 float @ nv=18) register-array footprint that spilled to local memory.
    BP_off      = OUT_off + 4 * nv3
    BP_W1   = BP_off
    BP_W2   = BP_W1 + nv2
    BP_IU   = BP_W2 + nv2
    BP_DDQ  = BP_IU + nv2
    BP_DDQD = BP_DDQ + nv2
    BP_DMI  = BP_DDQD + nv2
    BP_END  = BP_DMI + nv2
    assert BP_END <= 8 * nv3, \
        f"fdsva_so mjx scratch overflow: need {BP_END} > 8*nv^3={8*nv3} (nv={nv})"

    self.gen_add_code_line("// === mjx output convention (floating-base fdsva_so) ===")
    self.gen_add_code_lines([
        "T *s_M        = d_mjx_scratch + " + str(M_off) + ";       // dense mass matrix (crba_inner)",
        "T *s_dc_du    = d_mjx_scratch + " + str(DCDU_off) + ";    // pin dtau_dq | dtau_dqd",
        "T *s_mjx_Minv = d_mjx_scratch + " + str(MINV_off) + ";    // dense symmetric Minv = inv(M)",
        "T *s_mjx_dfdu = d_mjx_scratch + " + str(DFDU_off) + ";    // fd gradient dqdd_dq | dqdd_dqd (col-major)",
        "T *s_mjx_qdd  = d_mjx_scratch + " + str(QDD_off) + ";     // fd value qdd",
        "T *s_vaf      = d_mjx_scratch + " + str(VAF_off) + ";     // id-value band",
        "T *s_mjx_tmp  = d_mjx_scratch + " + str(INNERTMP_off) + ";    // reused-inner scratch (one at a time)",
        "T *s_mjx_out  = d_mjx_scratch + " + str(OUT_off) + ";     // 4*NV^3 mjx output band",
        "T *s_work1 = d_mjx_scratch + " + str(BP_W1) + "; T *s_work2 = d_mjx_scratch + " + str(BP_W2) + "; T *s_inner_u = d_mjx_scratch + " + str(BP_IU) + ";  // block-shared per-k work",
        "T *s_d_dq = d_mjx_scratch + " + str(BP_DDQ) + "; T *s_d_dqd = d_mjx_scratch + " + str(BP_DDQD) + "; T *s_d_Mi = d_mjx_scratch + " + str(BP_DMI) + ";  // block-shared per-k sensitivities",
    ])
    # ---- 1) recompute the fd VALUE qdd FRESH (forward_dynamics_inner) ----
    # Uses the live s_XImats (built for the converted q). s_mjx_tmp is its scratch.
    self.gen_add_code_line("// recompute Minv / qdd / dqdd_dq|dqdd_dqd FRESH (the spilled in-flight copies may be clobbered)")
    self.gen_forward_dynamics_inner_function_call(
        updated_var_names=dict(s_qdd_name="s_mjx_qdd", s_temp_name="s_mjx_tmp",
                               d_workspace_name="nullptr", d_f_ext_name="nullptr"))
    self.gen_add_sync()
    # ---- 2) recompute dense M (crba_inner), then Minv = inv(M) (dense symmetric) ----
    self.gen_crba_inner_function_call(
        updated_var_names=dict(s_M_name="s_M", s_temp_name="s_mjx_tmp", d_workspace_name="nullptr"),
        temp_in_smem_expr="true")
    self.gen_add_sync()
    # invert_matrix(n, in, out, scratch) -> dense symmetric Minv. Scratch >= NV in
    # s_mjx_tmp (disjoint from s_M / s_mjx_Minv).
    self.gen_add_code_line("invert_matrix(" + str(nv) + ", s_M, s_mjx_Minv, s_mjx_tmp);")
    self.gen_add_sync()
    # ---- 3) recompute pin id-value band (vaf) then dtau_dq|dtau_dqd (id-gradient) ----
    self.gen_inverse_dynamics_inner_function_call(
        compute_c=False, use_qdd_input=True,
        updated_var_names=dict(s_vaf_name="s_vaf", s_qdd_name="s_mjx_qdd",
                               s_temp_name="s_mjx_tmp", d_f_ext_name="nullptr"))
    self.gen_add_sync()
    self.gen_inverse_dynamics_gradient_inner_function_call(
        dict(s_dc_du_name="s_dc_du", s_vaf_name="s_vaf", s_temp_name="s_mjx_tmp",
             d_temp_spill_name="nullptr", temp_spill_flag_name="false"))
    self.gen_add_sync()
    # ---- 4) fd gradient s_mjx_dfdu = -Minv @ dc_du (col-major), block-parallel ----
    # Mirrors gen_fdsva_so_fd_gradient_inline's final reduction but reads the FRESH
    # s_mjx_Minv / s_dc_du and writes the FRESH s_mjx_dfdu.
    self.gen_add_parallel_loop("ind", str(2 * nv2))
    self.gen_add_code_line(f"int row = ind % {nv}; int dc_col_offset = ind - row;")
    self.gen_add_code_line("T val = static_cast<T>(0);")
    self.gen_add_code_line(f"for(int col = 0; col < {nv}; col++) {{", True)
    self.gen_add_code_line(f"int index = (row <= col) * (col * {nv} + row) + (row > col) * (row * {nv} + col);")
    self.gen_add_code_line("val += s_mjx_Minv[index] * s_dc_du[dc_col_offset + col];")
    self.gen_add_end_control_flow()
    self.gen_add_code_line("s_mjx_dfdu[ind] = -val;")
    self.gen_add_end_control_flow()
    self.gen_add_sync()
    # ---- 5) BLOCK-PARALLEL per-k assembly ----
    # The whole block cooperates on one k at a time: each dense nv^2 op is spread
    # across all threads (block-strided) over the block-shared s_work1/s_work2/
    # s_inner_u + s_d_dq/s_d_dqd/s_d_Mi, instead of one-thread-per-k with 6 nv^2
    # register arrays (which spilled to local memory -> huge SASS). Block-strided
    # loops carry a runtime bound so nvcc cannot unroll them -> rolled with NO _ur1.
    # Numerically identical (per-element math preserved; scalars stay register-local).
    _emit_fdsva_so_mjx_blockpar_assembly(self, nv)
    self.gen_add_sync()
    # ---- 6) copy the mjx output band back over s_df2 (block-parallel) ----
    self.gen_add_parallel_loop("ci", str(4 * nv3))
    self.gen_add_code_line("s_df2[ci] = s_mjx_out[ci];")
    self.gen_add_end_control_flow()
    self.gen_add_sync()

_FDSVA_SO_PICK_FLAGS = [
    # (use_global_tensors, use_workspace_temp, fd_grad_use_spill, use_workspace_df_du, use_workspace_Minv, use_workspace_idsva_temp, idsva_cold_in_global)
    (False, False, False, False, False, False, False),   # pick 0: full smem
    (True,  False, False, False, False, False, False),   # pick 1: outputs to global
    # A4: idsva_cold — outputs to global (like pick 1) AND the embedded idsva_so
    # WORLD inner's surgical cold trio (Xdown 36*NB + v_w/a_w 12*NB, dead before the
    # hot triple-walk) routes to d_workspace via the inner's COLD_IN_SMEM=false. The
    # whole hot pool stays in smem (SCRATCH_IN_SMEM=true). This shaves the cold trio
    # off the smem high-water mark BEFORE the all-or-nothing pool->global rung, so the
    # heavy mid-chain-ball world inner (the ~20-min/7GB cicc compile) keeps its hot
    # band in smem at a spill tier instead of dumping the whole pool to global. Only
    # meaningful when the composed idsva inner is the WORLD frame (floating OR spherical);
    # the body-frame inner has no exposed cold trio so the flag is a no-op there.
    (True,  False, False, False, False, False, True),    # pick 2 (A4): + idsva cold trio to global (surgical)
    (True,  True,  False, False, False, False, False),   # pick 3: + inner temp to global
    (True,  True,  True,  False, False, False, False),   # pick 4: + fd_grad da_df band to global
    (True,  True,  True,  True,  False, False, False),   # pick 5 (Phase 3e): + s_df_du to global
    (True,  True,  True,  True,  True,  False, False),   # pick 6 (Phase 3e): + s_Minv to global
    (True,  True,  False, False, False, True,  False),   # pick 7: pool->global (whole s_temp via full inner SCRATCH_IN_SMEM=false); df_du/Minv stay in smem (small). Works for BOTH bases because fdsva_so_device hands the placed pool to whichever idsva inner it composes (world for floating, body for fixed) and the inner does the repoint via its own SCRATCH_IN_SMEM=false.
    # pick 8 (S1): pool->global AND s_df_du->workspace. Only present in a robot's
    # ladder when even pick 7's BASE overflows the device cap (nv~81 humanoids:
    # s_df_du 2nv² = 52 KB pushes the base past ~99 KB) — the generator appends
    # this rung conditionally (see _fdsva_so_tiers in GRiDCodeGenerator.py).
    # Layout-safe by construction: the pool lives in the SO-temp region
    # (GRID_SO_WORKSPACE_TEMP_OFFSET_BYTES) while s_df_du sits in the dedicated
    # FDSVA_SO spill section (GRID_FDSVA_SO_SPILL_OFFSET_BYTES), disjoint bands.
    (True,  True,  False, True,  False, True,  False),   # pick 8: pool->global + s_df_du->workspace
]

def _emit_fdsva_so_kernel_body_for_flags(self, n, NUM_POS, use_global_tensors, use_workspace_temp,
                                         fd_grad_use_spill, use_workspace_df_du, use_workspace_Minv,
                                         single_call_timing, use_workspace_idsva_temp=False,
                                         idsva_cold_in_global=False):
    """Emit fdsva_so kernel body for one tier's spill flags.
    use_workspace_idsva_temp: route the embedded idsva_so inner's scratch to
    d_workspace (via the inner's SCRATCH_IN_SMEM=false). The idsva inner is the
    dominant s_temp consumer on big robots; spilling it drops the smem arena to
    the next-largest sub-inner (fd_grad / minv / fd). Works for BOTH bases —
    both the world inner (floating) and the body inner (fixed) now own their
    placement and accept SCRATCH_IN_SMEM=false, so the whole-arena pool->global
    fallback composes uniformly. (Earlier revisions of this comment said
    floating-only because the body inner had not yet migrated; that has since
    landed — see docs/idsva_so_inner_refactor_notes.md.)

    idsva_cold_in_global (A4): SURGICAL — route ONLY the embedded WORLD-frame
    idsva_so inner's cold trio (Xdown 36*NB + v_w/a_w 12*NB, dead before the hot
    triple-walk) to d_workspace via the inner's COLD_IN_SMEM=false, while the whole
    hot pool stays in smem (SCRATCH_IN_SMEM=true). Mutually exclusive with
    use_workspace_idsva_temp (which spills the WHOLE pool). Shrinks the smem arena
    by the cold-trio span (48*NB floats) so a spilling robot keeps its hot band in
    smem at a tier between 'outputs->global' and the all-or-nothing pool->global rung.
    Only meaningful when the composed idsva inner is the WORLD frame (floating OR
    spherical); for the body-frame inner there is no exposed cold trio so the inner's
    COLD_IN_SMEM template arg is inert there (the body inner's signature carries its
    own BC/TP levers, not COLD). The fdsva_so_device threads this through as the
    IDSVA_COLD_IN_SMEM template flag — see gen_fdsva_so_device."""
    # MUJOCO_OUTPUT (floating non-mimic/skew): the kernel template flag in scope;
    # gates the mjx input-convert + the trailing device-template arg.
    mjx_kernel = self.robot.floating_base and not (self.robot_has_mimic_joints() or self.robot.robot_has_skew_axis())
    # Spherical robots compose the WORLD-frame idsva_so inner (see the inner function
    # call routing above), so they must reserve the world-frame inner's temp memory.
    inner_idsva_so_temp_size = (
        self.gen_idsva_so_world_frame_temp_mem_size()
        if _fdsva_so_use_world_idsva(self)
        else self.gen_idsva_so_body_frame_inner_temp_mem_size()
    )
    fd_grad_temp_size = (
        self.gen_fdsva_so_fd_gradient_inline_temp_mem_size_spilled() if fd_grad_use_spill
        else self.gen_fdsva_so_fd_gradient_inline_temp_mem_size()
    )
    # A4: cold-quad span (Xup 36*NB + Xdown 36*NB + v_w/a_w 12*NB = 84*NB floats).
    # Subtracted from the world-inner footprint when idsva_cold_in_global routes it to
    # global. Single source of truth = gen_idsva_so_world_cold_floats (mirrors the
    # standalone idsva_so world cold rung _idsva_wf_cold in GRiDCodeGenerator.py).
    cold_floats = self.gen_idsva_so_world_cold_floats()
    if use_workspace_idsva_temp:
        # Pool -> global: fdsva_so_device runs with SCRATCH_IN_SMEM=false, so the
        # WHOLE shared s_temp pool (helper sincos + minv + fd + fd_grad + idsva) lives
        # in d_workspace. The smem s_temp slot is unused -> size 0.
        shared_temp_size = 0
    else:
        idsva_temp_size = inner_idsva_so_temp_size
        if idsva_cold_in_global:
            # Surgical: the world inner keeps its hot band in smem but repoints the
            # cold quad to d_workspace (COLD_IN_SMEM=false), so its smem footprint
            # shrinks by exactly the cold-quad span. (No-op for the body inner — but
            # this rung is only ever PICKED when the world inner is composed, see the
            # GRiDCodeGenerator.py tier table which gates the rung's arena on world.)
            idsva_temp_size = inner_idsva_so_temp_size - cold_floats
        shared_temp_size = max(idsva_temp_size, fd_grad_temp_size)
        if not use_workspace_temp:
            shared_temp_size = max(shared_temp_size, self.gen_fdsva_so_contract_temp_mem_size())
    # Phase 3e: s_df_du and s_Minv can now be in workspace too. Drop them from
    # extra_t_buffers when spilled; declare workspace pointers in the body.
    extra_t_buffers = [("s_q_qd_u", 3*NUM_POS), ("s_qdd", n)]
    if not use_workspace_Minv:
        extra_t_buffers.append(("s_Minv", n*n))
    if not use_workspace_df_du:
        extra_t_buffers.append(("s_df_du", 2*n*n))
    if not use_global_tensors:
        extra_t_buffers.append(("s_idsva_so", n*n*n*4))
        extra_t_buffers.append(("s_df2", 4*n*n*n))
    self.gen_XImats_helpers_temp_shared_memory_code(shared_temp_size, extra_t_buffers = extra_t_buffers)
    needs_d_workspace = (use_workspace_temp or fd_grad_use_spill or use_workspace_df_du
                         or use_workspace_Minv or idsva_cold_in_global)
    if not needs_d_workspace:
        self.gen_add_code_line("(void)d_workspace;")
    if not use_global_tensors:
        self.gen_add_code_line("(void)d_idsva_so;")
    self.gen_add_code_line("T *s_q = s_q_qd_u; T *s_qd = &s_q_qd_u[" + str(NUM_POS) + "]; T *s_u = &s_q_qd_u[" + str(2*NUM_POS) + "];")
    self.gen_add_code_line("T *d_temp_spill = nullptr; (void)d_temp_spill;")
    # Inner-controlled placement: forward_dynamics_inner slices its own Minv-F
    # from s_temp (smem; this kernel does not surgically spill Minv/FD-F — its
    # tiers spill the SO outputs / df_du / Minv instead).
    # Canonical: build the shared helper ARGS via gen_insert_helpers_function_call
    # (was a bespoke "make a def-params string then .replace() the types out" hack,
    # which silently dropped/duplicated args when the helper signature changed).
    fd_start = "forward_dynamics_inner<T, true>(s_qdd, s_q, s_qd, s_u, " + self.gen_insert_helpers_function_call()
    fd_end = "s_temp, nullptr, nullptr, gravity);"  # trailing nullptr = no external forces

    # B3 dedup (so_audit_plan): the timed (single_call_timing) and untimed kernel
    # bodies differed ONLY by the per-timestep `k*...PER_TIMESTEP +` offset prefix
    # on the global/workspace pointers (the untimed body indexes per-timestep k;
    # the timed body reuses slot 0 across NUM_TIMESTEPS reps). The output-tensor
    # pointers, the four workspace spill pointers, and the device call were
    # otherwise verbatim. Factored into _emit_fdsva_so_compute_pointers_and_call,
    # parameterized by `timing` (drops the k-offset + the d_df2/d_idsva_so k-slice
    # and uses the bare-`d_workspace` fd_grad_spill form the timed path used).
    def _emit_fdsva_so_compute_pointers_and_call(timing):
        # per-timestep slot offset prefix into d_workspace (empty for timing reps)
        ws_k = "" if timing else "k*GRID_WORKSPACE_BYTES_PER_TIMESTEP<T>() + "
        if use_global_tensors:
            if timing:
                self.gen_add_code_line('T *s_df2 = d_df2;')
                self.gen_add_code_line('T *s_idsva_so = d_idsva_so;')
            else:
                self.gen_add_code_line(f'T *s_df2 = &d_df2[k*{4*n**3}];')
                self.gen_add_code_line(f'T *s_idsva_so = &d_idsva_so[k*{4*n**3}];')
        if use_workspace_temp:
            self.gen_add_code_line('T *s_fdsva_temp = reinterpret_cast<T *>(&d_workspace[' + ws_k + 'GRID_SO_WORKSPACE_TEMP_OFFSET_BYTES<T>()]);')
        if fd_grad_use_spill:
            # spill band sits at offset 0 of this timestep's slot; timed reps reuse
            # slot 0 so the index collapses to the bare base pointer.
            if timing:
                self.gen_add_code_line('T *d_fd_grad_spill = reinterpret_cast<T *>(d_workspace);')
            else:
                self.gen_add_code_line('T *d_fd_grad_spill = reinterpret_cast<T *>(&d_workspace[k*GRID_WORKSPACE_BYTES_PER_TIMESTEP<T>()]);')
        if use_workspace_df_du:
            # Phase 3e: s_df_du in L2-pinned workspace, in its own dedicated section
            # past grad + SO (avoids conflict with fd_grad_spill which is at offset 0).
            self.gen_add_code_line('T *s_df_du = reinterpret_cast<T *>(&d_workspace[' + ws_k + 'GRID_FDSVA_SO_SPILL_OFFSET_BYTES<T>()]);')
        if use_workspace_Minv:
            # Phase 3e: s_Minv lives just past s_df_du in the FDSVA_SO spill section.
            self.gen_add_code_line('T *s_Minv = reinterpret_cast<T *>(&d_workspace[' + ws_k + 'GRID_FDSVA_SO_SPILL_OFFSET_BYTES<T>() + ' + str(2*n*n) + '*sizeof(T)]);')
        # MUJOCO_OUTPUT: carve d_mjx_scratch from the SO-temp region of d_workspace.
        # That region holds the contract scratch (s_fdsva_temp), which is DEAD by the
        # time the epilogue runs, so the reuse is safe and DISJOINT from the fd_grad /
        # s_df_du / s_Minv spill bands the epilogue must not read. Mirrors idsva_so.
        if mjx_kernel:
            self.gen_add_code_line("T *d_mjx_scratch = nullptr; (void)d_mjx_scratch;")
            self.gen_add_code_line("if constexpr (MUJOCO_OUTPUT) { d_mjx_scratch = reinterpret_cast<T *>(&d_workspace[" + ws_k + "GRID_SO_WORKSPACE_TEMP_OFFSET_BYTES<T>()]); }")
        # A4: surgical cold rung — the WORLD idsva inner's cold trio (48*NB floats) spills
        # to the SO-temp region of d_workspace (GRID_SO_WORKSPACE_TEMP_OFFSET_BYTES). DISJOINT
        # by construction at THIS rung: use_workspace_temp is False (contract stays in smem -> no
        # s_fdsva_temp carved from this region), fd_grad/df_du/Minv are NOT spilled (all in smem),
        # so nothing else touches the SO-temp band. The cold trio is dead before the hot triple
        # walk AND the contract runs entirely in smem after — no aliasing. (Mirrors the standalone
        # idsva_so world cold rung, which carves the same region; the two kernels never co-run.)
        if idsva_cold_in_global:
            self.gen_add_code_line('T *d_idsva_cold_spill = reinterpret_cast<T *>(&d_workspace[' + ws_k + 'GRID_SO_WORKSPACE_TEMP_OFFSET_BYTES<T>()]);')
        if not timing:
            self.gen_add_code_line("// compute — the orchestration inner owns its s_temp pool placement")
            # Pool->global reuses the (non-concurrent) fdsva SO-temp region; the
            # contraction uses the same region in its later phase. See full inner.
        self.gen_fdsva_so_device_function_call(
            scratch_in_smem_expr = "false" if use_workspace_idsva_temp else "true",
            fd_grad_use_spill_expr = "true" if fd_grad_use_spill else "false",
            contract_in_smem_expr = "false" if use_workspace_temp else "true",
            d_workspace_pool_name = "s_fdsva_temp" if use_workspace_idsva_temp else "nullptr",
            d_fd_grad_spill_name = "d_fd_grad_spill" if fd_grad_use_spill else "nullptr",
            s_fdsva_temp_name = "s_fdsva_temp" if use_workspace_temp else "nullptr",
            mujoco_output_expr = "MUJOCO_OUTPUT" if mjx_kernel else None,
            d_mjx_scratch_name = "d_mjx_scratch" if mjx_kernel else "nullptr",
            idsva_cold_in_smem_expr = "false" if idsva_cold_in_global else "true",
            d_idsva_cold_spill_name = "d_idsva_cold_spill" if idsva_cold_in_global else "nullptr")

    # MUJOCO_OUTPUT: convert the mjx-frame inputs (quat wxyz->xyzw, base-linear
    # qd/u -> pin frame) in place BEFORE the device inner runs (XImats build from
    # s_q). qdd is NOT converted (it is computed internally by forward_dynamics).
    def _emit_mjx_input_convert():
        if mjx_kernel:
            self.gen_add_code_line("if constexpr (MUJOCO_OUTPUT) {", True)
            self.gen_mjx_input_convert(q_name="s_q", qd_name="s_qd", u_name="s_u")
            self.gen_add_end_control_flow()

    if not single_call_timing:
        self.gen_add_parallel_loop("k","NUM_TIMESTEPS",block_level = True)
        self.gen_kernel_load_inputs("q_qd_u",str(3*NUM_POS),stride="stride_q_qd_u")
        _emit_mjx_input_convert()
        _emit_fdsva_so_compute_pointers_and_call(timing=False)
        self.gen_add_sync()
        if not use_global_tensors: self.gen_kernel_save_result("df2",str(4*n*n*n),stride=f"{4*n**3}")
        self.gen_add_end_control_flow()
    else:
        self.gen_kernel_load_inputs("q_qd_u",str(3*NUM_POS))
        self.gen_add_code_line("// compute with NUM_TIMESTEPS as NUM_REPS for timing")
        self.gen_add_code_line("for (int rep = 0; rep < NUM_TIMESTEPS; rep++){", True)
        self.gen_anti_licm_input_reload("q_qd_u",str(3*NUM_POS))
        _emit_mjx_input_convert()
        _emit_fdsva_so_compute_pointers_and_call(timing=True)
        self.gen_add_end_control_flow()
        if not use_global_tensors: self.gen_kernel_save_result("df2",str(4*n*n*n))


def gen_fdsva_so_kernel(self, single_call_timing = False):
    # NUM_VEL is the SO tensor dimension (rank-3 nv*nv*nv); NUM_POS is q-vector size.
    n = self.robot.get_num_vel()
    NUM_POS = self.robot.get_num_pos()
    func_params = ["d_df2 is the second derivatives of forward dynamics WRT q,qd,tau", \
                    "d_workspace is the generated global spill workspace", \
                    "d_q_qd_u is the vector of joint positions, velocities, torques", \
                    "stride_q_qd_u is the stride between each q, qd, qdd", \
                    "d_idsva_so is the pointer to the idsva_so output tensor in global memory", \
                    "d_robotModel is the pointer to the initialized model specific helpers on the GPU (XImats, topology_helpers, etc.)", \
                    "gravity is the gravity constant", \
                    "num_timesteps is the length of the trajectory points we need to compute over (or overloaded as test_iters for timing)"]
    func_notes = []
    func_def_start = "void fdsva_so_kernel(T *d_df2, unsigned char *d_workspace, const T *d_q_qd_u, const int stride_q_qd_u, T *d_idsva_so, "
    func_def_end = "const robotModel<T> *d_robotModel, const T gravity, const int NUM_TIMESTEPS) {"
    func_def = func_def_start + func_def_end
    if single_call_timing:
        func_def = func_def.replace("kernel(", "kernel_single_timing(")
    self.gen_add_func_doc("Compute the FDSVA_SO (Second Order of Forward Dynamics with Spacial Vector Algebra)", func_notes, func_params, None)
    # MUJOCO_OUTPUT (floating non-mimic/skew): kernel template flag appended LAST
    # after RESOURCE_TIER so the host forwards it positionally; default false ->
    # if-constexpr-elided to byte-identical PTX. Fixed-base keeps the 2-arg template.
    mjx_kernel = self.robot.floating_base and not (self.robot_has_mimic_joints() or self.robot.robot_has_skew_axis())
    self.gen_add_code_line("template <typename T, int RESOURCE_TIER = GRID_DEFAULT_RESOURCE_TIER, bool MUJOCO_OUTPUT = false>")
    self.gen_add_code_line("__global__")
    self.gen_add_code_line("__launch_bounds__(tier_max_threads<RESOURCE_TIER>())")
    self.gen_add_code_line(func_def, True)
    picks = getattr(self, "fdsva_so_spill_tier_3way", (5, 5, 5))
    def _emit_fdsva_so_body(pick):
        ugt, uwt, fgs, uwdfdu, uwminv, uwit, uict = _FDSVA_SO_PICK_FLAGS[pick]
        _emit_fdsva_so_kernel_body_for_flags(self, n, NUM_POS, ugt, uwt, fgs, uwdfdu, uwminv, single_call_timing, uwit, uict)
    self.gen_tier_dispatch(picks, _emit_fdsva_so_body)
    self.gen_add_end_function()

def gen_fdsva_so_host(self, mode = 0):
    # NUM_VEL is the SO tensor dimension. See gen_fdsva_so_kernel for details.
    n = self.robot.get_num_vel()
    # default is to do the full kernel call -- options are for single timing or compute only kernel wrapper
    single_call_timing = True if mode == 1 else False
    compute_only = True if mode == 2 else False

    # define function def and params
    func_params = ["hd_data is the packaged input and output pointers", \
                   "d_robotModel is the pointer to the initialized model specific helpers on the GPU (XImats, topology_helpers, etc.)", \
                   "gravity is the gravity constant,", \
                   "num_timesteps is the length of the trajectory points we need to compute over (or overloaded as test_iters for timing)", \
                   "streams are pointers to CUDA streams for async memory transfers (if needed)"]
    func_notes = []
    func_def_start = "void fdsva_so(gridData<T, KIND> *hd_data, const robotModel<T> *d_robotModel, const T gravity, const int num_timesteps,"
    func_def_end =   "                      const dim3 block_dimms, const dim3 thread_dimms, cudaStream_t *streams) {"
    if single_call_timing:
        func_def_start = func_def_start.replace("(", "_single_timing(")
        func_def_end = "              " + func_def_end
    if compute_only:
        func_def_start = func_def_start.replace("(", "_compute_only(")
        func_def_end = "             " + func_def_end.replace(", cudaStream_t *streams", "")
    # then generate the code
    self.gen_add_func_doc("Compute the FDSVA_SO (Second Order of Forward Dynamics with Spacial Vector Algebra)",\
                          func_notes,func_params,None)
    # MUJOCO_OUTPUT (floating non-mimic/skew): host template flag appended LAST;
    # forwarded positionally to fdsva_so_kernel<T, TIER, MUJOCO_OUTPUT>. Binding
    # calls grid::fdsva_so<T, KIND, /*MUJOCO_OUTPUT=*/true>. Fixed-base unchanged.
    mjx_host = self.robot.floating_base and not (self.robot_has_mimic_joints() or self.robot.robot_has_skew_axis())
    if mjx_host:
        self.gen_add_code_line("template <typename T, gridDataKind KIND = GRID_DATA_ALL, bool MUJOCO_OUTPUT = false, int RESOURCE_TIER = GRID_DEFAULT_RESOURCE_TIER>")
    else:
        self.gen_add_code_line("template <typename T, gridDataKind KIND = GRID_DATA_ALL, int RESOURCE_TIER = GRID_DEFAULT_RESOURCE_TIER>")
    self.gen_add_code_line("__host__")
    self.gen_add_code_line(func_def_start)
    self.gen_add_code_line(func_def_end, True)
    self.gen_add_code_line("static_assert(KIND == GRID_DATA_ALL || KIND == GRID_DATA_DYNAMICS, \"fdsva_so requires all-data or dynamics gridData\");")

    kernel_tmpl = "fdsva_so_kernel<T, RESOURCE_TIER, MUJOCO_OUTPUT>" if mjx_host else "fdsva_so_kernel<T, RESOURCE_TIER>"
    func_call_start = kernel_tmpl + "<<<block_dimms,thread_dimms,FDSVA_SO_DYNAMIC_SHARED_MEM_BYTES<T, RESOURCE_TIER>()>>>(hd_data->d_df2,hd_data->d_workspace,hd_data->d_q_qd_u,stride_q_qd_qdd,hd_data->d_idsva_so,"
    func_call_end = "d_robotModel,gravity,num_timesteps);"
    self.gen_add_code_line("int stride_q_qd_qdd = Q_QD_U_STRIDE;")
    if single_call_timing:
        func_call_start = func_call_start.replace("fdsva_so_kernel<T, RESOURCE_TIER, MUJOCO_OUTPUT>","fdsva_so_kernel_single_timing<T, RESOURCE_TIER, MUJOCO_OUTPUT>").replace("fdsva_so_kernel<T, RESOURCE_TIER>","fdsva_so_kernel_single_timing<T, RESOURCE_TIER>")
    if not compute_only:
        # start code with memory transfer
        self.gen_add_code_lines(["// start code with memory transfer", \
                                 "gpuErrchk(cudaMemcpyAsync(hd_data->d_q_qd_u,hd_data->h_q_qd_u,stride_q_qd_qdd" + \
                                    ("*num_timesteps" if not single_call_timing else "") + "*sizeof(T),cudaMemcpyHostToDevice,streams[0]));", \
                                 "gpuErrchkKernel();"])    
    
    # then compute:
    self.gen_add_code_line("// call the kernel")
    func_call = func_call_start + func_call_end
    func_call_code = [func_call, "gpuErrchkKernel();"]
    # wrap function call in timing (if needed)
    if single_call_timing:
        func_call_code.insert(0,"struct timespec start, end; clock_gettime(CLOCK_MONOTONIC,&start);")
        func_call_code.append("clock_gettime(CLOCK_MONOTONIC,&end);")
    self.gen_add_code_line("gpuErrchk(grid_check_dynamic_shared_memory_bytes(\"fdsva_so\", FDSVA_SO_DYNAMIC_SHARED_MEM_BYTES<T, RESOURCE_TIER>()));")
    chunked = getattr(self, "emit_workspace_chunking", False) and not single_call_timing
    ws_ts = "grid_workspace_chunk(num_timesteps)" if chunked else "num_timesteps"
    workspace_bytes = "GRID_WORKSPACE_BYTES_PER_TIMESTEP<T>()*GRID_WORKSPACE_SLOTS*" + ("1" if single_call_timing else ws_ts)
    self.gen_add_code_line("if (GRID_FDSVA_SO_USES_WORKSPACE_ANY_TIER) {gpuErrchk(grid_begin_l2_persisting(0, hd_data->d_workspace, " + workspace_bytes + "));}")
    if single_call_timing:
        self.gen_add_code_lines(func_call_code)
    else:
        # chunked-workspace seam (modes 0/2): d_df2 OUT and d_idsva_so INPUT both
        # advance by SECOND_ORDER_TENSOR_SIZE per timestep; d_workspace stays at
        # base (arena reused). The per-launch gpuErrchkKernel stays OUTSIDE the
        # chunk loop (same-stream in-order; one final sync is honest wall time).
        self.gen_add_workspace_chunked_launch([func_call],
            [("hd_data->d_df2", "SECOND_ORDER_TENSOR_SIZE"),
             ("hd_data->d_idsva_so", "SECOND_ORDER_TENSOR_SIZE"),
             ("hd_data->d_q_qd_u", "stride_q_qd_qdd")])
        self.gen_add_code_line("gpuErrchkKernel();")
    self.gen_add_code_line("if (GRID_FDSVA_SO_USES_WORKSPACE_ANY_TIER) {gpuErrchk(grid_end_l2_persisting(0));}")
    if not compute_only:
        # then transfer memory back
        # sizeof(T) leads: 4*nv^3*num_timesteps overflows int on big robots
        self.gen_add_code_lines(["// finally transfer the result back", \
                                "gpuErrchk(cudaMemcpy(hd_data->h_df2,hd_data->d_df2,sizeof(T)*" + \
                                ("num_timesteps*" if not single_call_timing else "") + str(4*n**3) + ",cudaMemcpyDeviceToHost));",
                                "gpuErrchkKernel();"])
    # finally report out timing if requested
    if single_call_timing:
        from ..algo_registry import single_call_printf_line
        self.gen_add_code_line(single_call_printf_line("fdsva_so"))
    self.gen_add_end_function()

def _emit_fdsva_so_mjx_blockpar_assembly(self, n):
    """BLOCK-PARALLEL per-k assembly of all 4 fdsva_so mjx tensors. Same math as
    the (removed) thread-per-k form (proto_fdsva_so_emit_spec.py)
    but the whole block cooperates on one k at a time: the 6 per-k nv^2 matrices
    (s_work1/s_work2/s_inner_u + sensitivities s_d_dq/s_d_dqd/s_d_Mi) are BLOCK-SHARED
    and every dense op is spread across the block (block-strided) with a sync between
    producers and consumers. Per-k scalars (R, jqk/jvk/juk/jvvk/juuk, Rd, d_qd[0:3])
    stay per-thread register-local, so per-element results are bit-identical to the
    thread-per-k form; only WHICH thread computes each nv^2 element changes."""
    nv3 = n * n * n
    nv2 = n * n
    N = str(n)
    def t3(i, j, k):
        return "((" + i + ")*" + N + " + (" + j + "))*" + N + " + (" + k + ")"
    self.gen_add_code_line("// === block-parallel per-k assembly (whole block cooperates per k) ===")
    self.gen_add_code_line("#pragma unroll 1")
    self.gen_add_code_line("for (int k = 0; k < " + N + "; k++) {", True)
    self.gen_add_code_lines(_emit_fdsva_so_mjx_locals_lines(self, n, nv3))
    # jqk/jvk/juk (base-block sparse) + jvvk/juuk (== jqk-form) — register per thread.
    self.gen_add_code_lines([
        "T jqk[" + str(n) + "], jvk[" + str(n) + "], juk[" + str(n) + "], jvvk[" + str(n) + "], juuk[" + str(n) + "];",
        "for (int q_ = 0; q_ < " + N + "; q_++) { jqk[q_] = static_cast<T>(0); jvk[q_] = static_cast<T>(0); juk[q_] = static_cast<T>(0); jvvk[q_] = static_cast<T>(0); juuk[q_] = static_cast<T>(0); }",
        "bool is_rot = (k >= 3 && k < 6);",
        "int a_rot = k - 3;",
        "if (k < 3) { jqk[0] = R[3*k+0]; jqk[1] = R[3*k+1]; jqk[2] = R[3*k+2]; }",
        "else { jqk[k] = static_cast<T>(1); }",
        "for (int q_ = 0; q_ < " + N + "; q_++) { jvvk[q_] = jqk[q_]; juuk[q_] = jqk[q_]; }",
        "if (is_rot) {",
        "  T evx = (a_rot==1)*( v_lin[2]) + (a_rot==2)*(-v_lin[1]);",
        "  T evy = (a_rot==0)*(-v_lin[2]) + (a_rot==2)*( v_lin[0]);",
        "  T evz = (a_rot==0)*( v_lin[1]) + (a_rot==1)*(-v_lin[0]);",
        "  jvk[0] = -evx; jvk[1] = -evy; jvk[2] = -evz;",
        "  T eux = (a_rot==1)*( u_lin[2]) + (a_rot==2)*(-u_lin[1]);",
        "  T euy = (a_rot==0)*(-u_lin[2]) + (a_rot==2)*( u_lin[0]);",
        "  T euz = (a_rot==0)*( u_lin[1]) + (a_rot==1)*(-u_lin[0]);",
        "  juk[0] = -eux; juk[1] = -euy; juk[2] = -euz;",
        "}",
    ])
    # Rd = R @ skew(e_{a_rot}) — register per thread.
    self.gen_add_code_lines([
        "T Rd[9];",
        "for (int ii = 0; ii < 9; ii++) Rd[ii] = static_cast<T>(0);",
        "if (is_rot) {",
        "  T sk[9]; for (int ii=0; ii<9; ii++) sk[ii]=static_cast<T>(0);",
        "  if (a_rot==0){ sk[1*3+2] = static_cast<T>(-1); sk[2*3+1] = static_cast<T>(1); }",
        "  if (a_rot==1){ sk[2*3+0] = static_cast<T>(-1); sk[0*3+2] = static_cast<T>(1); }",
        "  if (a_rot==2){ sk[0*3+1] = static_cast<T>(-1); sk[1*3+0] = static_cast<T>(1); }",
        "  for (int r = 0; r < 3; r++) for (int c = 0; c < 3; c++) {",
        "    T acc = static_cast<T>(0);",
        "    for (int p = 0; p < 3; p++) acc += R[3*r+p]*sk[3*p+c];",
        "    Rd[3*r+c] = acc;",
        "  }",
        "}",
    ])
    # Sensitivities d_dq/d_dqd/d_Mi -> BLOCK-SHARED (block-strided over rows i).
    self.gen_add_code_lines([
        _bpfor("i", n),
        "  for (int j = 0; j < " + N + "; j++) {",
        "    T s = static_cast<T>(0);",
        "    for (int m = 0; m < " + N + "; m++) s += T_d2q[" + t3("i", "j", "m") + "]*jqk[m];",
        "    for (int nn = 0; nn < " + N + "; nn++) s += T_cross[" + t3("i", "nn", "j") + "]*jvk[nn];",
        "    for (int l = 0; l < " + N + "; l++) s += T_dtdq[" + t3("i", "l", "j") + "]*juk[l];",
        "    s_d_dq[j*" + N + " + i] = s;",
        "    T s2 = static_cast<T>(0);",
        "    for (int m = 0; m < " + N + "; m++) s2 += T_cross[" + t3("i", "j", "m") + "]*jqk[m];",
        "    for (int nn = 0; nn < " + N + "; nn++) s2 += T_d2qd[" + t3("i", "j", "nn") + "]*jvk[nn];",
        "    s_d_dqd[j*" + N + " + i] = s2;",
        "    T s3 = static_cast<T>(0);",
        "    for (int m = 0; m < " + N + "; m++) s3 += T_dtdq[" + t3("i", "j", "m") + "]*jqk[m];",
        "    s_d_Mi[j*" + N + " + i] = s3;",
        "  }",
        "}",
    ]); self.gen_add_sync()
    # d_qd[0:3] only (out_R base needs the base-linear 3) — register per thread.
    self.gen_add_code_lines([
        "T d_qd[3];",
        "for (int i = 0; i < 3; i++) {",
        "  T st = static_cast<T>(0);",
        "  for (int j = 0; j < " + N + "; j++) {",
        "    T mij = s_Minv[(i<=j) ? (j*" + N + "+i) : (i*" + N + "+j)];",
        "    st += s_dqdd_dq[j*" + N + "+i]*jqk[j] + s_dqdd_dqd[j*" + N + "+i]*jvk[j] + mij*juk[j];",
        "  }",
        "  d_qd[i] = st;",
        "}",
    ])
    # each slab in a fresh brace scope (its d(out_*) setup scalars must not collide
    # across slabs in the shared per-k loop body).
    for _slab in (_emit_fbp_slab_d2q, _emit_fbp_slab_cross, _emit_fbp_slab_d2qd, _emit_fbp_slab_dtdq):
        self.gen_add_code_line("{", True)
        _slab(self, n)
        self.gen_add_end_control_flow()
    self.gen_add_end_control_flow()   # k


def _fbp_writeback(self, n, src, Oname):
    """Block-strided transpose writeback: O[i,j,k] = src[j*nv+i], over rows i."""
    N = str(n)
    self.gen_add_code_lines([
        _bpfor("i", n),
        "  for (int j = 0; j < " + N + "; j++) " + Oname + "[(i*" + N + " + j)*" + N + " + k] = " + src + "[j*" + N + " + i];",
        "}",
    ]); self.gen_add_sync()


def _emit_fbp_slab_d2q(self, n):
    """d2qdd/dq2 slab, block-parallel. Transcribed from proto_fdsva_so_emit_spec.py."""
    N = str(n)
    self.gen_add_code_line("// --- d2qdd/dq2 slab[:,:,k] (block-parallel) ---")
    # P (value inner pin-accel gradient) into s_work1
    self.gen_add_code_lines(_fbp(_fdsva_reframe_cols(n, "s_dqdd_dq", "s_work1"), n)); self.gen_add_sync()
    self.gen_add_code_lines(_fbp(_fdsva_add_XJvq(n, "s_dqdd_dqd", "v_lin", "s_work1"), n)); self.gen_add_sync()
    self.gen_add_code_lines(_fbp(_fdsva_add_MinvJvq(n, "u_lin", "s_work1"), n)); self.gen_add_sync()
    # dP into s_work2
    self.gen_add_code_lines(_fbp(_fdsva_reframe_cols(n, "s_d_dq", "s_work2"), n)); self.gen_add_sync()
    self.gen_add_code_lines(_fbp(_fdsva_add_Rd_reframe(n, "s_dqdd_dq", "s_work2"), n)); self.gen_add_sync()
    self.gen_add_code_lines(_fbp(_fdsva_add_XJvq(n, "s_d_dqd", "v_lin", "s_work2"), n)); self.gen_add_sync()
    self.gen_add_code_lines(_fbp(_fdsva_add_XJvq(n, "s_dqdd_dqd", "jvk", "s_work2"), n)); self.gen_add_sync()
    self.gen_add_code_lines(_fbp(_fdsva_add_XJvq(n, "s_d_Mi", "u_lin", "s_work2"), n)); self.gen_add_sync()
    self.gen_add_code_lines(_fbp(_fdsva_add_MinvJvq(n, "juk", "s_work2"), n)); self.gen_add_sync()
    # g0 = rot_rows(dP=s_work2) -> s_inner_u ; + Gd @ P(s_work1)
    self.gen_add_code_lines(_fbp(_fdsva_rot_rows(n, "s_work2", "s_inner_u"), n)); self.gen_add_sync()
    self.gen_add_code_lines(_fbp(_fdsva_add_Gd(n, "s_work1", "s_inner_u"), n)); self.gen_add_sync()
    # d(out_R) + d(out_vd): setup (register, all threads) then += over a in [0,3).
    self.gen_add_code_lines([
        "T ov0 = omega[1]*v_lin[2] - omega[2]*v_lin[1];",
        "T ov1 = omega[2]*v_lin[0] - omega[0]*v_lin[2];",
        "T ov2 = omega[0]*v_lin[1] - omega[1]*v_lin[0];",
        "T base0 = qdd_lin[0]+ov0, base1 = qdd_lin[1]+ov1, base2 = qdd_lin[2]+ov2;",
        "T dvx = jvk[0], dvy = jvk[1], dvz = jvk[2];",
        "T domx = jvk[3], domy = jvk[4], domz = jvk[5];",
        "T dob0 = (domy*v_lin[2]-domz*v_lin[1]) + (omega[1]*dvz-omega[2]*dvy);",
        "T dob1 = (domz*v_lin[0]-domx*v_lin[2]) + (omega[2]*dvx-omega[0]*dvz);",
        "T dob2 = (domx*v_lin[1]-domy*v_lin[0]) + (omega[0]*dvy-omega[1]*dvx);",
        "T dbase0 = d_qd[0]+dob0, dbase1 = d_qd[1]+dob1, dbase2 = d_qd[2]+dob2;",
        _bpfor("a", 3),
        "  T wb0 = (a==1)*( dbase2) + (a==2)*(-dbase1);",
        "  T wb1 = (a==0)*(-dbase2) + (a==2)*( dbase0);",
        "  T wb2 = (a==0)*( dbase1) + (a==1)*(-dbase0);",
        "  s_inner_u[(3+a)*" + N + "+0] += R[0]*wb0 + R[1]*wb1 + R[2]*wb2;",
        "  s_inner_u[(3+a)*" + N + "+1] += R[3]*wb0 + R[4]*wb1 + R[5]*wb2;",
        "  s_inner_u[(3+a)*" + N + "+2] += R[6]*wb0 + R[7]*wb1 + R[8]*wb2;",
        "  T rb0 = (a==1)*( base2) + (a==2)*(-base1);",
        "  T rb1 = (a==0)*(-base2) + (a==2)*( base0);",
        "  T rb2 = (a==0)*( base1) + (a==1)*(-base0);",
        "  s_inner_u[(3+a)*" + N + "+0] += Rd[0]*rb0 + Rd[1]*rb1 + Rd[2]*rb2;",
        "  s_inner_u[(3+a)*" + N + "+1] += Rd[3]*rb0 + Rd[4]*rb1 + Rd[5]*rb2;",
        "  s_inner_u[(3+a)*" + N + "+2] += Rd[6]*rb0 + Rd[7]*rb1 + Rd[8]*rb2;",
        "  T evx = (a==1)*( v_lin[2]) + (a==2)*(-v_lin[1]);",
        "  T evy = (a==0)*(-v_lin[2]) + (a==2)*( v_lin[0]);",
        "  T evz = (a==0)*( v_lin[1]) + (a==1)*(-v_lin[0]);",
        "  T dvqx = -evx, dvqy = -evy, dvqz = -evz;",
        "  T edvx = (a==1)*( dvz) + (a==2)*(-dvy);",
        "  T edvy = (a==0)*(-dvz) + (a==2)*( dvx);",
        "  T edvz = (a==0)*( dvy) + (a==1)*(-dvx);",
        "  T ddvqx = -edvx, ddvqy = -edvy, ddvqz = -edvz;",
        "  T w0 = (domy*dvqz-domz*dvqy) + (omega[1]*ddvqz-omega[2]*ddvqy);",
        "  T w1 = (domz*dvqx-domx*dvqz) + (omega[2]*ddvqx-omega[0]*ddvqz);",
        "  T w2 = (domx*dvqy-domy*dvqx) + (omega[0]*ddvqy-omega[1]*ddvqx);",
        "  s_inner_u[(3+a)*" + N + "+0] += R[0]*w0 + R[1]*w1 + R[2]*w2;",
        "  s_inner_u[(3+a)*" + N + "+1] += R[3]*w0 + R[4]*w1 + R[5]*w2;",
        "  s_inner_u[(3+a)*" + N + "+2] += R[6]*w0 + R[7]*w1 + R[8]*w2;",
        "  T wr0 = omega[1]*dvqz - omega[2]*dvqy;",
        "  T wr1 = omega[2]*dvqx - omega[0]*dvqz;",
        "  T wr2 = omega[0]*dvqy - omega[1]*dvqx;",
        "  s_inner_u[(3+a)*" + N + "+0] += Rd[0]*wr0 + Rd[1]*wr1 + Rd[2]*wr2;",
        "  s_inner_u[(3+a)*" + N + "+1] += Rd[3]*wr0 + Rd[4]*wr1 + Rd[5]*wr2;",
        "  s_inner_u[(3+a)*" + N + "+2] += Rd[6]*wr0 + Rd[7]*wr1 + Rd[8]*wr2;",
        "}",
    ]); self.gen_add_sync()
    _fbp_writeback(self, n, "s_inner_u", "O_d2q")


def _emit_fbp_slab_cross(self, n):
    """cross slab, block-parallel. Transcribed from proto_fdsva_so_emit_spec.py."""
    N = str(n)
    self.gen_add_code_line("// --- cross (d2qdd/dqd dq) slab[:,:,k] (block-parallel) ---")
    self.gen_add_code_lines(_fbp(_fdsva_reframe_cols(n, "s_dqdd_dqd", "s_work1"), n)); self.gen_add_sync()
    self.gen_add_code_lines(_fbp(_fdsva_reframe_cols(n, "s_d_dqd", "s_work2"), n)); self.gen_add_sync()
    self.gen_add_code_lines(_fbp(_fdsva_add_Rd_reframe(n, "s_dqdd_dqd", "s_work2"), n)); self.gen_add_sync()
    self.gen_add_code_lines(_fbp(_fdsva_rot_rows(n, "s_work2", "s_inner_u"), n)); self.gen_add_sync()
    self.gen_add_code_lines(_fbp(_fdsva_add_Gd(n, "s_work1", "s_inner_u"), n)); self.gen_add_sync()
    self.gen_add_code_lines([
        "T dvx = jvk[0], dvy = jvk[1], dvz = jvk[2];",
        "T domx = jvk[3], domy = jvk[4], domz = jvk[5];",
        _bpfor("a", 3),
        "  T rte0 = R[3*a+0], rte1 = R[3*a+1], rte2 = R[3*a+2];",
        "  T rdte0 = Rd[3*a+0], rdte1 = Rd[3*a+1], rdte2 = Rd[3*a+2];",
        "  T wl0 = (domy*rte2-domz*rte1) + (omega[1]*rdte2-omega[2]*rdte1);",
        "  T wl1 = (domz*rte0-domx*rte2) + (omega[2]*rdte0-omega[0]*rdte2);",
        "  T wl2 = (domx*rte1-domy*rte0) + (omega[0]*rdte1-omega[1]*rdte0);",
        "  s_inner_u[(0+a)*" + N + "+0] += R[0]*wl0 + R[1]*wl1 + R[2]*wl2;",
        "  s_inner_u[(0+a)*" + N + "+1] += R[3]*wl0 + R[4]*wl1 + R[5]*wl2;",
        "  s_inner_u[(0+a)*" + N + "+2] += R[6]*wl0 + R[7]*wl1 + R[8]*wl2;",
        "  T wlr0 = omega[1]*rte2 - omega[2]*rte1;",
        "  T wlr1 = omega[2]*rte0 - omega[0]*rte2;",
        "  T wlr2 = omega[0]*rte1 - omega[1]*rte0;",
        "  s_inner_u[(0+a)*" + N + "+0] += Rd[0]*wlr0 + Rd[1]*wlr1 + Rd[2]*wlr2;",
        "  s_inner_u[(0+a)*" + N + "+1] += Rd[3]*wlr0 + Rd[4]*wlr1 + Rd[5]*wlr2;",
        "  s_inner_u[(0+a)*" + N + "+2] += Rd[6]*wlr0 + Rd[7]*wlr1 + Rd[8]*wlr2;",
        "  T wn0 = (a==1)*( dvz) + (a==2)*(-dvy);",
        "  T wn1 = (a==0)*(-dvz) + (a==2)*( dvx);",
        "  T wn2 = (a==0)*( dvy) + (a==1)*(-dvx);",
        "  s_inner_u[(3+a)*" + N + "+0] += R[0]*wn0 + R[1]*wn1 + R[2]*wn2;",
        "  s_inner_u[(3+a)*" + N + "+1] += R[3]*wn0 + R[4]*wn1 + R[5]*wn2;",
        "  s_inner_u[(3+a)*" + N + "+2] += R[6]*wn0 + R[7]*wn1 + R[8]*wn2;",
        "  T wnr0 = (a==1)*( v_lin[2]) + (a==2)*(-v_lin[1]);",
        "  T wnr1 = (a==0)*(-v_lin[2]) + (a==2)*( v_lin[0]);",
        "  T wnr2 = (a==0)*( v_lin[1]) + (a==1)*(-v_lin[0]);",
        "  s_inner_u[(3+a)*" + N + "+0] += Rd[0]*wnr0 + Rd[1]*wnr1 + Rd[2]*wnr2;",
        "  s_inner_u[(3+a)*" + N + "+1] += Rd[3]*wnr0 + Rd[4]*wnr1 + Rd[5]*wnr2;",
        "  s_inner_u[(3+a)*" + N + "+2] += Rd[6]*wnr0 + Rd[7]*wnr1 + Rd[8]*wnr2;",
        "}",
    ]); self.gen_add_sync()
    _fbp_writeback(self, n, "s_inner_u", "O_cross")


def _emit_fbp_slab_d2qd(self, n):
    """d2qdd/dqd2 slab, block-parallel. Transcribed from proto_fdsva_so_emit_spec.py."""
    N = str(n)
    self.gen_add_code_line("// --- d2qdd/dqd2 slab[:,:,k] (qvel perturb, R fixed) (block-parallel) ---")
    # dvQ[i,j] = sum_n d2qd[i,j,n]*jvvk[n] -> s_work1 (over rows i)
    self.gen_add_code_lines([
        _bpfor("i", n),
        "  for (int j = 0; j < " + N + "; j++) {",
        "    T s = static_cast<T>(0);",
        "    for (int nn = 0; nn < " + N + "; nn++) s += T_d2qd[(i*" + N + " + j)*" + N + " + nn]*jvvk[nn];",
        "    s_work1[j*" + N + " + i] = s;",
        "  }",
        "}",
    ]); self.gen_add_sync()
    self.gen_add_code_lines(_fbp(_fdsva_reframe_cols(n, "s_work1", "s_work2"), n)); self.gen_add_sync()
    self.gen_add_code_lines(_fbp(_fdsva_rot_rows(n, "s_work2", "s_inner_u"), n)); self.gen_add_sync()
    self.gen_add_code_lines([
        "T dvvv0 = jvvk[0], dvvv1 = jvvk[1], dvvv2 = jvvk[2];",
        "T dvvo0 = jvvk[3], dvvo1 = jvvk[4], dvvo2 = jvvk[5];",
        _bpfor("a", 3),
        "  T rte0 = R[3*a+0], rte1 = R[3*a+1], rte2 = R[3*a+2];",
        "  T wl0 = dvvo1*rte2 - dvvo2*rte1;",
        "  T wl1 = dvvo2*rte0 - dvvo0*rte2;",
        "  T wl2 = dvvo0*rte1 - dvvo1*rte0;",
        "  s_inner_u[(0+a)*" + N + "+0] += R[0]*wl0 + R[1]*wl1 + R[2]*wl2;",
        "  s_inner_u[(0+a)*" + N + "+1] += R[3]*wl0 + R[4]*wl1 + R[5]*wl2;",
        "  s_inner_u[(0+a)*" + N + "+2] += R[6]*wl0 + R[7]*wl1 + R[8]*wl2;",
        "  T wn0 = (a==1)*( dvvv2) + (a==2)*(-dvvv1);",
        "  T wn1 = (a==0)*(-dvvv2) + (a==2)*( dvvv0);",
        "  T wn2 = (a==0)*( dvvv1) + (a==1)*(-dvvv0);",
        "  s_inner_u[(3+a)*" + N + "+0] += R[0]*wn0 + R[1]*wn1 + R[2]*wn2;",
        "  s_inner_u[(3+a)*" + N + "+1] += R[3]*wn0 + R[4]*wn1 + R[5]*wn2;",
        "  s_inner_u[(3+a)*" + N + "+2] += R[6]*wn0 + R[7]*wn1 + R[8]*wn2;",
        "}",
    ]); self.gen_add_sync()
    _fbp_writeback(self, n, "s_inner_u", "O_d2qd")


def _emit_fbp_slab_dtdq(self, n):
    """dtdq slab, block-parallel. Transcribed from proto_fdsva_so_emit_spec.py. Final in s_work2."""
    N = str(n)
    self.gen_add_code_line("// --- d2qdd/du dq slab[:,:,k] (force perturb, R fixed) (block-parallel) ---")
    # du_dq[i,j] = sum_l T_dtdq[i,l,j]*juuk[l] -> s_work1 (over rows i)
    self.gen_add_code_lines([
        _bpfor("i", n),
        "  for (int j = 0; j < " + N + "; j++) {",
        "    T s = static_cast<T>(0);",
        "    for (int l = 0; l < " + N + "; l++) s += T_dtdq[(i*" + N + " + l)*" + N + " + j]*juuk[l];",
        "    s_work1[j*" + N + " + i] = s;",
        "  }",
        "}",
    ]); self.gen_add_sync()
    self.gen_add_code_lines(_fbp(_fdsva_reframe_cols(n, "s_work1", "s_inner_u"), n)); self.gen_add_sync()
    self.gen_add_code_lines(_fbp(_fdsva_add_MinvJvq(n, "juuk", "s_inner_u"), n)); self.gen_add_sync()
    self.gen_add_code_lines(_fbp(_fdsva_rot_rows(n, "s_inner_u", "s_work2"), n)); self.gen_add_sync()
    # du_qdd[0:3] = Minv[0:3,:] @ juuk (register per thread), then d(out_R)/du over a.
    self.gen_add_code_lines([
        "T duq0 = static_cast<T>(0), duq1 = static_cast<T>(0), duq2 = static_cast<T>(0);",
        "for (int j = 0; j < " + N + "; j++) {",
        "  duq0 += s_Minv[(0<=j) ? (j*" + N + "+0) : (0*" + N + "+j)]*juuk[j];",
        "  duq1 += s_Minv[(1<=j) ? (j*" + N + "+1) : (1*" + N + "+j)]*juuk[j];",
        "  duq2 += s_Minv[(2<=j) ? (j*" + N + "+2) : (2*" + N + "+j)]*juuk[j];",
        "}",
        _bpfor("a", 3),
        "  T w0 = (a==1)*( duq2) + (a==2)*(-duq1);",
        "  T w1 = (a==0)*(-duq2) + (a==2)*( duq0);",
        "  T w2 = (a==0)*( duq1) + (a==1)*(-duq0);",
        "  s_work2[(3+a)*" + N + "+0] += R[0]*w0 + R[1]*w1 + R[2]*w2;",
        "  s_work2[(3+a)*" + N + "+1] += R[3]*w0 + R[4]*w1 + R[5]*w2;",
        "  s_work2[(3+a)*" + N + "+2] += R[6]*w0 + R[7]*w1 + R[8]*w2;",
        "}",
    ]); self.gen_add_sync()
    _fbp_writeback(self, n, "s_work2", "O_dtdq")


def _fdsva_reframe_cols(n, src, dst):
    """dst = src then cols c<3 <- sum_{cp<3} src[:,cp]*R[c,cp] (R[3c+cp]).  (X@Ginv)"""
    N = str(n)
    return [
        "for (int r = 0; r < " + N + "; r++) {",
        "  T c0 = " + src + "[0*" + N + "+r], c1 = " + src + "[1*" + N + "+r], c2 = " + src + "[2*" + N + "+r];",
        "  " + dst + "[0*" + N + "+r] = c0*R[0] + c1*R[1] + c2*R[2];",
        "  " + dst + "[1*" + N + "+r] = c0*R[3] + c1*R[4] + c2*R[5];",
        "  " + dst + "[2*" + N + "+r] = c0*R[6] + c1*R[7] + c2*R[8];",
        "  for (int c = 3; c < " + N + "; c++) " + dst + "[c*" + N + "+r] = " + src + "[c*" + N + "+r];",
        "}",
    ]


def _fdsva_add_Rd_reframe(n, src, dst):
    """dst[0:3,:] += src[:,0:3] @ Rd^T  (Rd^T[cp,c]=Rd[c,cp]=Rd[3c+cp]). Rd=0 unless is_rot."""
    N = str(n)
    return [
        "for (int r = 0; r < " + N + "; r++) {",
        "  T c0 = " + src + "[0*" + N + "+r], c1 = " + src + "[1*" + N + "+r], c2 = " + src + "[2*" + N + "+r];",
        "  " + dst + "[0*" + N + "+r] += c0*Rd[0] + c1*Rd[1] + c2*Rd[2];",
        "  " + dst + "[1*" + N + "+r] += c0*Rd[3] + c1*Rd[4] + c2*Rd[5];",
        "  " + dst + "[2*" + N + "+r] += c0*Rd[6] + c1*Rd[7] + c2*Rd[8];",
        "}",
    ]


def _fdsva_add_XJvq(n, X, vecname, dst):
    """dst[:,3+a] += X[:,0:3] @ (-(e_a x vecname[3]))   (Jvq / Juq column-couple)."""
    N = str(n)
    Xc = (lambda c: X + "[" + str(c) + "*" + N + "+r]")
    return [
        "for (int a = 0; a < 3; a++) {",
        "  T evx = (a==1)*( " + vecname + "[2]) + (a==2)*(-" + vecname + "[1]);",
        "  T evy = (a==0)*(-" + vecname + "[2]) + (a==2)*( " + vecname + "[0]);",
        "  T evz = (a==0)*( " + vecname + "[1]) + (a==1)*(-" + vecname + "[0]);",
        "  T jc0 = -evx, jc1 = -evy, jc2 = -evz;",
        "  for (int r = 0; r < " + N + "; r++) " + dst + "[(3+a)*" + N + "+r] += "
        + Xc(0) + "*jc0 + " + Xc(1) + "*jc1 + " + Xc(2) + "*jc2;",
        "}",
    ]


def _fdsva_add_MinvJvq(n, vecname, dst):
    """dst[:,3+a] += Minv[:,0:3] @ (-(e_a x vecname[3]))  (Minv symmetric read)."""
    N = str(n)
    mc = (lambda c: "s_Minv[(r<=" + str(c) + ") ? (" + str(c) + "*" + N + "+r) : (r*" + N + "+" + str(c) + ")]")
    return [
        "for (int a = 0; a < 3; a++) {",
        "  T evx = (a==1)*( " + vecname + "[2]) + (a==2)*(-" + vecname + "[1]);",
        "  T evy = (a==0)*(-" + vecname + "[2]) + (a==2)*( " + vecname + "[0]);",
        "  T evz = (a==0)*( " + vecname + "[1]) + (a==1)*(-" + vecname + "[0]);",
        "  T jc0 = -evx, jc1 = -evy, jc2 = -evz;",
        "  for (int r = 0; r < " + N + "; r++) {",
        "    T m0 = " + mc(0) + ", m1 = " + mc(1) + ", m2 = " + mc(2) + ";",
        "    " + dst + "[(3+a)*" + N + "+r] += m0*jc0 + m1*jc1 + m2*jc2;",
        "  }",
        "}",
    ]


def _fdsva_rot_rows(n, src, dst):
    """dst = src then rows r<3 <- R @ rows0:3  (G @ X)."""
    N = str(n)
    return [
        "for (int c = 0; c < " + N + "; c++) {",
        "  T m0 = " + src + "[c*" + N + "+0], m1 = " + src + "[c*" + N + "+1], m2 = " + src + "[c*" + N + "+2];",
        "  " + dst + "[c*" + N + "+0] = R[0]*m0 + R[1]*m1 + R[2]*m2;",
        "  " + dst + "[c*" + N + "+1] = R[3]*m0 + R[4]*m1 + R[5]*m2;",
        "  " + dst + "[c*" + N + "+2] = R[6]*m0 + R[7]*m1 + R[8]*m2;",
        "  for (int r = 3; r < " + N + "; r++) " + dst + "[c*" + N + "+r] = " + src + "[c*" + N + "+r];",
        "}",
    ]


def _fdsva_add_Gd(n, src, dst):
    """dst += Gd @ src : rows0:3 += Rd @ src[0:3,:]."""
    N = str(n)
    return [
        "for (int c = 0; c < " + N + "; c++) {",
        "  T r0 = " + src + "[c*" + N + "+0], r1 = " + src + "[c*" + N + "+1], r2 = " + src + "[c*" + N + "+2];",
        "  " + dst + "[c*" + N + "+0] += Rd[0]*r0 + Rd[1]*r1 + Rd[2]*r2;",
        "  " + dst + "[c*" + N + "+1] += Rd[3]*r0 + Rd[4]*r1 + Rd[5]*r2;",
        "  " + dst + "[c*" + N + "+2] += Rd[6]*r0 + Rd[7]*r1 + Rd[8]*r2;",
        "}",
    ]


def gen_fdsva_so(self):
    # first the contraction sub-inner used by the orchestration _device
    self.gen_fdsva_so_contract()
    # then the canonical _device (orchestrator: owns s_temp placement; called from kernel)
    self.gen_fdsva_so_device()
    # then the kernels (call _device with caller-allocated smem)
    self.gen_fdsva_so_kernel(True)
    self.gen_fdsva_so_kernel(False)
    # then the host wrappers
    self.gen_fdsva_so_host(0)
    self.gen_fdsva_so_host(1)
    self.gen_fdsva_so_host(2)
    
