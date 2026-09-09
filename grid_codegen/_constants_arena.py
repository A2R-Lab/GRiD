"""Arena/constants emission: gen_add_constants_helpers (the shared-memory tier
and spill-rung composer + every *_DYNAMIC_SHARED_MEM_BYTES helper + the
robotModel/gridData structs) and gen_init_gridData. H4 move from
GRiDCodeGenerator.py (2026-08-27, verbatim — the ~80-attribute self-state
contract with the algorithm emitters is unchanged; see the audit before
restructuring internals)."""
import numpy as np

from .algo_registry import (ALGO_DESCRIPTORS, arena_ctx_from_codegen, compose_arena_full,
                            compose_arena_rungs, ARENA_COMPOSED_KEYS, ARENA_RUNG_KEYS)


def _tier_bytes_lines(macro, counts, linalg_arg=", GRID_LINALG_NVIDIA_MAX_HELPER_BYTES<T>()"):
    """The tier-templated *_DYNAMIC_SHARED_MEM_BYTES helper as its ONE emitted
    source line: three if-constexpr branches over the per-tier t_counts.
    String-for-string identical to the hand-written wall it replaced (C3
    table-drive, 2026-09-08). `linalg_arg` is the third grid_shared_arena_bytes
    argument including its leading ", " ("" for the SO arenas)."""
    return ["template <typename T, int TIER = GRID_DEFAULT_RESOURCE_TIER> __host__ __device__ inline size_t " + macro + "() { "
            "if constexpr (TIER == TIER_SHARED)    return grid_shared_arena_bytes<T>(" + str(counts[0]) + ", TOPOLOGY_HELPERS_COUNT" + linalg_arg + "); "
            "else if constexpr (TIER == TIER_LITE) return grid_shared_arena_bytes<T>(" + str(counts[1]) + ", TOPOLOGY_HELPERS_COUNT" + linalg_arg + "); "
            "else                                 return grid_shared_arena_bytes<T>(" + str(counts[2]) + ", TOPOLOGY_HELPERS_COUNT" + linalg_arg + "); }"]


def _tier_ternary_line(name, ret_type, vals):
    """A per-tier constexpr ternary helper line: `template <int TIER> ... NAME()
    { return (TIER == TIER_SHARED) ? v0 : (TIER == TIER_LITE) ? v1 : v2; }`.
    `vals` are the three already-formatted C literals (use _b for bools)."""
    return ("template <int TIER> __host__ __device__ constexpr " + ret_type + " " + name +
            "() { return (TIER == TIER_SHARED) ? " + vals[0] + " : (TIER == TIER_LITE) ? " +
            vals[1] + " : " + vals[2] + "; }")



def gen_add_constants_helpers(self, include_base_inertia = False, include_homogenous_transforms = False):
    # first add constants
    n = self.robot.get_num_pos()
    nv = self.robot.get_num_vel()
    NJ = self.robot.get_num_joints()
    # Dynamics kernels only need the spatial X/I storage. Homogeneous transforms
    # are accounted separately for kinematics kernels.
    XI_size = self.gen_get_XI_size(include_base_inertia,include_homogenous_transforms=False)
    # runtime_transform: the load_update_XImats helper rebuilds each joint's
    # constant 6x6 Xfixed into s_temp at offset _runtime_transform_xfixed_offset
    # (= 2*num_pos non-mimic / 3*NB mimic), occupying 36*NB extra floats. That
    # block is consumed ENTIRELY within the helper (the hot loop reads it to
    # build s_XImats) and is DEAD after the helper returns, so each algorithm's
    # inner-temp scratch may reuse the region afterward. A purely ADDITIVE
    # reservation of 36*NB floats in every XImats-domain (s_temp-backed) arena
    # t_count is therefore sufficient to stop the OOB AND keep correctness; the
    # baked path keeps rt_xfixed_reserve == 0 so its arena/header stays
    # byte-identical. Only the s_temp-backed (dynamics, XImats) arenas need it;
    # the XmatsHom/kinematics arenas don't invoke the Xfixed rebuild. NOT added
    # to XI_size itself (that sizes the s_XImats array + DYNAMICS_XI_T_COUNT and
    # would corrupt the XImats layout) — it is appended to the temp region.
    rt_xfixed_reserve = (36 * NJ) if getattr(self, "runtime_transform", False) else 0
    XHom_size, dXhom_size, d2Xhom_size = self.gen_get_Xhom_size()
    # Descriptor-table Step 3: build the ArenaCtx snapshot from the exact sizing
    # locals so the per-algo `arena_full_fn` closures (algo_registry) can drive the
    # FULL/rung-0 arena t_counts. Each fold keeps the hand-written legacy expression
    # behind `assert composed == legacy` (the parity shim, deleted in 3.6). The
    # inner-temp helpers ArenaCtx reads are pure robot-shape functions, so building
    # the snapshot here (before the arena math) matches their mid-function values —
    # proven by test/test_algo_descriptor_arena_parity.py.
    self._arena_ctx = arena_ctx_from_codegen(self, xi=XI_size, xhom=XHom_size, rt=rt_xfixed_reserve)
    dva_cols_per_partial = self.robot.get_total_ancestor_count() + self.robot.get_num_joints()
    max_threads_in_comp_loop = 6*2*dva_cols_per_partial
    max_perf_level_threads = 32 * int(np.ceil(max_threads_in_comp_loop/32.0))
    # cap to 512 mirrors the constant we emit further down; exposed on self
    # for helpers that need the same per-robot thread cap.
    self.max_perf_level_threads = min(max_perf_level_threads, 512)
    topology_count = self.gen_topology_helpers_size()
    def py_align_up(offset, alignment):
        return ((offset + alignment - 1) // alignment) * alignment

    def py_arena_bytes(t_count, int_count = topology_count):
        offset = 0
        t_align = max(1, min(self.cuda_shared_mem_type_size_bytes, 8))
        offset = py_align_up(offset, t_align)
        offset += self.cuda_shared_mem_type_size_bytes * int(t_count)
        if int_count > 0:
            offset = py_align_up(offset, 4)
            offset += 4 * int(int_count)
        return py_align_up(offset, 16)

    def select_shared_tier_3way(*t_counts):
        """Pick the (perf, lite, minimal) spill-level indices for one algo.
        `t_counts` is the ordered list of arena t_counts at each spill level,
        least-spill first. PERF picks the lowest index whose arena fits
        cuda_target_shared_mem_bytes; LITE picks the lowest index whose
        arena fits cuda_target_lite_shared_mem_bytes (clamped to be ≥ PERF
        pick — LITE can't be less spill than PERF); MINIMAL is always the
        last (most-spill) index."""
        last = len(t_counts) - 1
        perf = next((i for i, t in enumerate(t_counts)
                     if py_arena_bytes(t) <= self.cuda_target_shared_mem_bytes), last)
        lite = next((i for i, t in enumerate(t_counts)
                     if py_arena_bytes(t) <= self.cuda_target_lite_shared_mem_bytes), last)
        lite = max(perf, lite)
        return (perf, lite, last)

    # The ID kernel's s_vaf band is body-indexed (18*NJ); for mimic robots
    # (NJ > n) size it 18*NJ so the inner's body f-writes don't overflow into
    # the XImats region. Non-mimic keeps the legacy 18*n byte-identical.
    id_t_count = compose_arena_full("inverse_dynamics", self._arena_ctx)   # Step 3.1 fold
    # joint-torque regressor (E1): kernel smem = XI + s_q_qd_qdd(NUM_POS+2nv)
    # + s_Y (nv x 10*NUM_BODIES) + s_vaf(18*NUM_POS) + RNEA forward scratch.
    # n == get_num_pos() here. Additive.
    # g1-spill: 2-rung s_Y output-spill ladder (mirror fdpg :795-802). The s_Y output
    # (nv*10*NB, ~277KB on h2_plus -> unlaunchable) routes to the L2-pinned d_workspace
    # SO section at any tier whose full arena overflows the target, keeping s_vaf +
    # inputs + XImats + RNEA scratch in smem. Small robots stay rung 0 (s_Y in smem).
    _idr_Y_count          = nv*10*self.robot.get_num_bodies()
    (_idr_t_count_full, _idr_t_count_surgical) = compose_arena_rungs("inverse_dynamics_regressor", self._arena_ctx)   # Step 3.5d fold
    self.inverse_dynamics_regressor_spill_tier_3way = select_shared_tier_3way(_idr_t_count_full, _idr_t_count_surgical)
    self.inverse_dynamics_regressor_t_count_per_tier = tuple(
        (_idr_t_count_full, _idr_t_count_surgical)[i] for i in self.inverse_dynamics_regressor_spill_tier_3way)
    self.inverse_dynamics_regressor_spill_Y_ws_count = _idr_Y_count
    # PS5 energy regressors. Both outputs are 10*NUM_BODIES (no DoF sweep).
    # KE (spatial / XImats domain): s_q_qd(n+nv) + s_y_ke(10NB) + s_vaf(18n)
    #   + RNEA forward scratch + XI_size.
    self.kinetic_energy_regressor_t_count = compose_arena_full("kinetic_energy_regressor", self._arena_ctx)   # Step 3.1 fold
    # PE (kinematics / XmatsHom domain): s_q(n) + s_y_pe(10NB)
    #   + world-transform BFS scratch (16*NUM_JOINTS) + XHom_size.
    self.potential_energy_regressor_t_count = compose_arena_full("potential_energy_regressor", self._arena_ctx)   # Step 3.1 fold
    # PS5 Coriolis matrix C(q,qd): kernel smem = XI + s_q_qd(NUM_POS+nv) + s_coriolis(nv*nv)
    #   + the inner spatial-recursion scratch (per-body NB bands + per-column n_int bands).
    # 3-rung surgical ladder (mirror crba): full | s_coriolis(nv*nv output) -> L2-pinned SO
    # band, hot inner band stays smem | whole-band (inner scratch -> d_workspace GRAD section,
    # output -> SO band). On h2_plus the inner band (18330 t ~73KB) dominates and the full
    # arena is ~121KB (UNLAUNCHABLE); the output-spill rung is ~95KB (PERF-launchable) and the
    # whole-band rung ~22KB (LITE/MINIMAL).
    (_coriolis_t_full, _coriolis_t_output_spill, _coriolis_t_workspace) = \
        compose_arena_rungs("coriolis_matrix", self._arena_ctx)   # Step 3.3 fold
    self.coriolis_matrix_spill_tier_3way = select_shared_tier_3way(_coriolis_t_full, _coriolis_t_output_spill, _coriolis_t_workspace)
    # whole-band spill (inner scratch -> d_workspace) fires only at the deepest rung (index 2).
    self.coriolis_matrix_t_count_per_tier = tuple(
        (_coriolis_t_full, _coriolis_t_output_spill, _coriolis_t_workspace)[i] for i in self.coriolis_matrix_spill_tier_3way)
    # PS5 dCCRBA (kinematics / XmatsHom domain). The shared inner pool is the
    # SHRUNK (no-J) centroidal_inner pool + 6*n_int per-unit phi band
    # (== _dccrba_inner_temp_mem_size). The Jw sweep band (6*nv*NB) is carved as a
    # SEPARATE tier-routed buffer s_J (in-smem at L0/L1, d_workspace at the
    # J-spilled tier) -- DE-GATE #2: this is the cold/large quadratic buffer whose
    # spill de-gates big floating robots (g1/h1_2-floating).
    _dccrba_sJ = self._dccrba_sweep_J_count()   # 6*nv*NB
    # cmm_time_variation (Adot, 6*nv output). 2-rung ladder (the only lever is the
    # Jw band; its tiny 6*nv output never spills): L0 keeps s_J in smem, L1 spills
    # it to the d_workspace SO band.
    #   base = s_q_qd(2n) + s_out(6nv) + s_A(6nv) + s_com(3) + s_extra(4) + inner + XHom.
    (_cmm_t_count_full, _cmm_t_count_Jspill) = compose_arena_rungs("cmm_time_variation", self._arena_ctx)   # Step 3.3 fold
    self.cmm_time_variation_spill_tier_3way = select_shared_tier_3way(_cmm_t_count_full, _cmm_t_count_Jspill)
    self.cmm_time_variation_t_count_per_tier = tuple(
        (_cmm_t_count_full, _cmm_t_count_Jspill)[i] for i in self.cmm_time_variation_spill_tier_3way
    )
    self.cmm_time_variation_spill_J_ws_count = _dccrba_sJ
    # dccrba (full 6*nv*nv tensor). 3-rung ladder: L0 keeps the s_dccrba output
    # (6*nv*nv) AND s_J in smem; L1 spills the output to d_workspace but keeps s_J
    # in smem (today's surgical rung on robots that fit it); L2 (NEW) spills BOTH
    # output and s_J -> d_workspace at distinct SO sub-offsets (the de-gating rung).
    #   base = s_q(n) + s_A(6nv) + s_com(3) + s_extra(4) + inner + XHom (no out, no s_J).
    _dccrba_out = 6 * nv * nv
    (_dccrba_L0, _dccrba_L1, _dccrba_L2) = compose_arena_rungs("dccrba", self._arena_ctx)   # Step 3.3 fold
    self.dccrba_spill_tier_3way = select_shared_tier_3way(_dccrba_L0, _dccrba_L1, _dccrba_L2)
    self.dccrba_t_count_per_tier = tuple(
        (_dccrba_L0, _dccrba_L1, _dccrba_L2)[i] for i in self.dccrba_spill_tier_3way
    )
    self.dccrba_spill_out_ws_count = _dccrba_out
    self.dccrba_spill_J_ws_count = _dccrba_sJ
    # FD param gradient: kernel smem = XI + s_q_qd_u(NUM_POS+2nv) + s_dqdd_dpi
    # + s_Minv(nv*nv) + s_Y(nv x 10*NB) + s_qdd(nv) + s_vaf(18*NUM_POS) + s_c(nv)
    # + the (max) inner forward scratch. n == get_num_pos() here. Additive.
    # FD-param-gradient g1-spill: 2-level surgical ladder. Level 0 keeps every
    # buffer in smem (current behavior on robots that fit). Level 1 spills the
    # s_Y regressor scratch (nv*10*NB, write-once / consumed-once in the final
    # -Minv.Y GEMM) to the L2-pinned d_workspace SO section -- the hot Minv +
    # vaf + inner-RNEA path stays in smem. On g1-floating this drops the arena
    # from ~135 KB to ~94 KB, under the sm_120 ~99 KB cap. The picker selects
    # level 1 for any tier whose level-0 arena overflows the smem target.
    _fpg_Y_count = nv * 10 * self.robot.get_num_bodies()
    (_fpg_t_count_full, _fpg_t_count_surgical) = compose_arena_rungs("forward_dynamics_parameter_gradient", self._arena_ctx)   # Step 3.5d fold
    self.forward_dynamics_parameter_gradient_spill_tier_3way = select_shared_tier_3way(_fpg_t_count_full, _fpg_t_count_surgical)
    self.forward_dynamics_parameter_gradient_t_count_per_tier = tuple(
        (_fpg_t_count_full, _fpg_t_count_surgical)[i] for i in self.forward_dynamics_parameter_gradient_spill_tier_3way
    )
    self.forward_dynamics_parameter_gradient_spill_Y_ws_count = _fpg_Y_count
    # f_ext gradient (section A): kernel smem = XI + s_q + the two nv x (6*NB)
    # outputs + temp (nv*nv s_Minv + max(J^T-inner, minv-inner) scratch).
    _n_pos = self.robot.get_num_pos()
    _NB = self.robot.get_num_bodies()
    _feg_out = nv * 6 * _NB
    # f_ext-gradient (first-order) g1/h2_plus-spill: 3-rung surgical ladder.
    #   rung 0 (full): both outputs (s_dtau_dfext, s_dqdd_dfext) + minv-F in smem.
    #   rung 1 (out-spill): spill s_dqdd_dfext (the SECOND output, written write-once
    #     by the final -Minv@s_dtau GEMM) to the L2-pinned d_workspace SO section;
    #     s_dtau_dfext (read by that GEMM) + s_Minv + minv-F stay in smem. On
    #     g1-floating the full arena is ~99.3 KB (272 B over the sm_120 cap), so this
    #     ~74.6 KB rung is what lets it run; mid robots stop here.
    #   rung 2 (deep): ALSO spill s_dtau_dfext (-> SO section) and route minv's
    #     6*nv*nv F-region to the GRAD-section minv-F workspace offset (F_IN_SMEM=
    #     false). On h2_plus (nv=81) rung 0/1 are ~506/361 KB (UNLAUNCHABLE); the deep
    #     rung is ~68 KB. s_Minv (nv*nv) + the J^T-inner + minv no-F scratch stay hot
    #     in smem. select picks the least-spill rung that fits, so small robots keep 0.
    (_feg_t_count_full, _feg_t_count_out_spill, _feg_t_count_deep) = compose_arena_rungs("f_ext_gradient", self._arena_ctx)   # Step 3.5d fold
    self.f_ext_gradient_spill_tier_3way = select_shared_tier_3way(
        _feg_t_count_full, _feg_t_count_out_spill, _feg_t_count_deep)
    self.f_ext_gradient_t_count_per_tier = tuple(
        (_feg_t_count_full, _feg_t_count_out_spill, _feg_t_count_deep)[i] for i in self.f_ext_gradient_spill_tier_3way
    )
    # SO-section reservation: rung 1 holds 1 output, rung 2 holds BOTH (s_dqdd at
    # SO base, s_dtau at SO base + _feg_out). Reserve 2x when any tier reaches deep.
    self.f_ext_gradient_spill_out_ws_count = (2*_feg_out if any(p >= 2 for p in self.f_ext_gradient_spill_tier_3way) else _feg_out)
    # rung 2 routes minv's 6*nv*nv F-region to the GRAD-section minv-F offset.
    self.f_ext_gradient_spill_minv_F_count = self.gen_minv_inner_F_size() if any(p >= 2 for p in self.f_ext_gradient_spill_tier_3way) else 0
    # A.3 (-dJ^T/dq) ANALYTIC kernel (both base modes): arena = XI + s_q + the
    # load_update_XImats reload scratch (loaded ONCE for the current q) + a
    # MIMIC-ONLY per-sub-job slab (6*nsub) folded in a deterministic serial reduce.
    # Non-mimic robots write each sub-job directly to its unique output cell (no slab).
    # 2-rung ladder: rung 0 keeps the mimic slab in smem; rung 1 spills it to the
    # L2-pinned d_workspace SO section. Non-mimic robots have no slab, so both rungs
    # collapse (no spill). rt: the single load_update_XImats reserves rt_xfixed under
    # runtime_transform, matching the kernel carve.
    (_feg_dq_t_count_full, _feg_dq_t_count_spill) = compose_arena_rungs("f_ext_gradient_dq", self._arena_ctx)   # Step 3.5d fold
    self.f_ext_gradient_dq_spill_tier_3way = select_shared_tier_3way(_feg_dq_t_count_full, _feg_dq_t_count_spill)
    self.f_ext_gradient_dq_t_count_per_tier = tuple(
        (_feg_dq_t_count_full, _feg_dq_t_count_spill)[i] for i in self.f_ext_gradient_dq_spill_tier_3way
    )
    # MIMIC slab spill count (6*nsub floats into the SO band); 0 for non-mimic (no slab).
    self.f_ext_gradient_dq_spill_slab_ws_count = (6*self._arena_ctx.feg_dq_jobs if self._arena_ctx.has_mimic else 0)
    # Minv Phase 3a: per-tier spill picks. Level 0 = F in smem (6*NV*NV
    # bytes); Level 1 = surgical F to L2-pinned workspace.
    (_minv_t_count_full, _minv_t_count_surgical) = compose_arena_rungs("minv", self._arena_ctx)   # Step 3.2 fold
    self.minv_spill_tier_3way = select_shared_tier_3way(_minv_t_count_full, _minv_t_count_surgical)
    self.minv_use_workspace_F = self.minv_spill_tier_3way[0] == 1
    minv_t_count = _minv_t_count_full if not self.minv_use_workspace_F else _minv_t_count_surgical
    self.minv_t_count_per_tier = tuple(
        (_minv_t_count_full, _minv_t_count_surgical)[i] for i in self.minv_spill_tier_3way
    )
    # FD inner-controlled placement: forward_dynamics_inner slices its own
    # Minv-F. The inner-temp size now bundles F (or not) per MINV_F_IN_SMEM,
    # so the full vs surgical kernel arenas come straight from the sized
    # helper (no separate F term — avoids double-counting). Level 0 = F in
    # smem; Level 1 = F in L2-pinned workspace.
    # Canonical input slot: q/qd/u each NUM_JOINTS(=nq=n here)-wide -> s_q_qd_u
    # is 3*n (matches _emit_fd_kernel_body_for_flags' ("s_q_qd_u", 3*nq)); qdd is
    # nv. For a FIXED base n==nv so 3*n == old 3*nv+fb byte-identical; FLOATING
    # n>nv so the arena must reserve the wider 3*n slot the kernel slices (the
    # old 3*nv+fb under-reserved by 3*(n-nv)-fb floats -> smem overrun).
    (_fd_t_count_full, _fd_t_count_surgical) = compose_arena_rungs("forward_dynamics", self._arena_ctx)   # Step 3.2 fold
    self.fd_spill_tier_3way = select_shared_tier_3way(_fd_t_count_full, _fd_t_count_surgical)
    self.fd_use_workspace_F = self.fd_spill_tier_3way[0] == 1
    fd_t_count = _fd_t_count_full if not self.fd_use_workspace_F else _fd_t_count_surgical
    self.fd_t_count_per_tier = tuple(
        (_fd_t_count_full, _fd_t_count_surgical)[i] for i in self.fd_spill_tier_3way
    )
    # Integrator: kernel-shared t-count layout is
    #   s_q_qd_u (3*nq) + s_qdd (nv) + s_stage_qdd ((max_stages-1)*nv)
    #   + s_stage_point ((max_stages-1)*(nq+nv)) + s_x_kp1 (nq+nv)
    #   + s_temp (= FD inner)
    # Canonical input slot: q/qd/u each NUM_JOINTS(=nq=n here)-wide -> 3*n; the
    # per-stage point + next state are [q (nq); qd (nv)] = n+nv. Must match
    # _emit_integrator_kernel_body_for_flags exactly. For a FIXED base n==nv so
    # 3*n == old 3*nv+fb and n+nv == old 2*nv+fb (byte-identical); FLOATING n>nv
    # so this reserves the wider input slot (old 3*nv+fb under-reserved -> overrun).
    # max_stages = 4 (RK4) — see _integrator._max_stages_in_use().
    _max_stages = 4
    _integrator_base = ((3*n) + nv
                        + (_max_stages - 1) * nv
                        + (_max_stages - 1) * (n + nv)
                        + (n + nv) + XI_size + rt_xfixed_reserve)
    integrator_t_count = _integrator_base + self.gen_forward_dynamics_inner_temp_mem_size()
    # Integrator VALUE surgical spill. The dominant inner buffer is the FD
    # inner's Minv F-region (6*NV*NV). Level 0 keeps it in smem; level 1
    # spills ONLY F to d_workspace (the hot FD path stays in smem), mirroring
    # the standalone forward_dynamics kernel's MINV_F_IN_SMEM lever. For
    # h1_2 the value arena overflows by only a few KB, so the surgical F
    # spill is enough — no whole-arena dump.
    (_integrator_t_count_full, _integrator_t_count_Fspill) = compose_arena_rungs("integrator", self._arena_ctx)   # Step 3.3 fold
    self.integrator_spill_tier_3way = select_shared_tier_3way(_integrator_t_count_full, _integrator_t_count_Fspill)
    self.integrator_t_count_per_tier = tuple(
        (_integrator_t_count_full, _integrator_t_count_Fspill)[i] for i in self.integrator_spill_tier_3way)
    # F float-count the value path spills (for grad-section sizing); 0 if no tier spills.
    self.integrator_minv_F_workspace_count = (self.gen_minv_inner_F_size()
                                              if any(p == 1 for p in self.integrator_spill_tier_3way) else 0)
    # Integrator gradient: kernel-shared t-count layout is
    #   s_q_qd_u (3nv+fb) + s_dAB (2nv*3nv) + s_df_du (nv*2nv) + s_dc_du (nv*2nv) +
    #   s_vaf (18nv) + s_Minv (nv*nv) + s_qdd (nv)
    #   + multi-stage scratch: s_q_orig (nv+fb) + s_qd_orig (nv)
    #     + s_stage_grad_qdd (max_stages*nv) + s_D_qdd_stage (max_stages*nv*3nv)
    #   + s_temp (= FD-grad inner)
    # s_q_orig holds the FULL nq pose (floating-base adds the quaternion slot),
    # so it is nv+fb — must match _emit_body's ("s_q_orig", n+fb) exactly, else
    # the launched dynamic-smem (this t_count) is fb floats short of the arena
    # the kernel slices and the tail buffer overruns shared memory (floating only).
    # max_stages = 4 (RK4) — see _integrator._max_stages_in_use().
    # The multi-stage scratch is always allocated even for single-stage IT;
    # cost is small relative to total (~12*nv² for iiwa14 ≈ 588 floats).
    _max_stages = 4
    # s_vaf is body-indexed (stride 6 over NB bodies). For a MIMIC robot
    # (fixed base) NB > nv, so the composed FD-grad inner writes 18*NB — size
    # the arena's s_vaf term 18*NB to match _emit_body's ("s_vaf", 18*NB)
    # exactly (else the launched dynamic-smem is short and the kernel overruns
    # smem). Non-mimic keeps 18*nv (byte-identical; floating nv > NB).
    _vaf_count = 18 * (self.robot.get_num_joints() if self.robot_has_mimic_joints() else nv)
    # +72 for the two 6x6 SE(3) dIntegrate blocks (floating-base gradient;
    # allocated for fixed-base too but unused there).
    # Canonical input slot: q/qd/u each NUM_JOINTS(=nq=n here)-wide -> s_q_qd_u
    # is 3*n (matches _emit_body's ("s_q_qd_u", 3*nq)). The (n+nv) term is
    # s_q_orig(nq) + s_qd_orig(nv). For a FIXED base n==nv so 3*n == old 3*nv+fb
    # and n+nv == old 2*nv+fb (byte-identical); FLOATING n>nv so this reserves
    # the wider input slot (old 3*nv+fb under-reserved -> smem overrun).
    integrator_gradient_t_count = ((3*n) + 2*nv*3*nv + 2*(nv*2*nv)
                             + _vaf_count + nv*nv + nv
                             + (n + nv) + _max_stages * nv + _max_stages * nv * 3*nv
                             + 72
                             + self.gen_forward_dynamics_gradient_inner_temp_mem_size() + XI_size + rt_xfixed_reserve)
    # The "with x_kp1" variant adds s_x_kp1 ([q (nq); qd (nv)] = n+nv) on top.
    integrator_gradient_with_x_kp1_t_count = integrator_gradient_t_count + (n + nv)
    # Integrator-gradient surgical spill ladder (4 rungs, least-spill first).
    # Each rung spills only cold / output / coalesced matrices to d_workspace,
    # keeping the hot path (s_vaf + the FD-grad scaffold) in smem as long as
    # it fits. The integrator-gradient kernel never runs concurrently with
    # inverse_dynamics_gradient/forward_dynamics_gradient/fdsva_so, so its spilled buffers safely reuse those sections.
    #   rung 0: everything in smem.
    #   rung 1: s_D_qdd_stage (max_stages*nv*3nv) -> d_workspace. (g1_fixed)
    #   rung 2: + s_dAB output (2nv*3nv) -> d_workspace, + inverse_dynamics_gradient da_df band
    #           SELECTIVE spill (the FD-grad inner shrinks to the selective
    #           shared count; only the da_df band leaves smem). (g1_floating)
    #   rung 3: + the WHOLE FD-grad inner s_temp -> d_workspace (inverse_dynamics_gradient
    #           global_temp; the inner can't fit a 100 KB box on h1_2). The
    #           gradient scaffold (s_dc_du / s_vaf / s_Minv) stays in smem.
    # PERF picks the lowest fitting rung; MINIMAL is the last (always fits).
    _integrator_gradient_D_qdd_count = _max_stages * nv * 3 * nv
    _integrator_gradient_dAB_count = 2 * nv * 3 * nv
    _integrator_gradient_inner_full = self.gen_forward_dynamics_gradient_inner_temp_mem_size()
    # The FD-grad inner's da_df-band SELECTIVE spill only exists on the SPARSE
    # (non-mimic) inverse_dynamics_gradient layout. The MIMIC inner is a DENSE
    # serial fold that ignores USE_DA_DF_SPILL and always writes its full pool,
    # so it cannot shrink — size the "selective" rung to the FULL inner there.
    # (Otherwise rung 2 claims a false shrink: the arena is sized for the sparse
    # selective_shared_count but the dense inner writes its full pool -> smem OOB.
    # This is the design_principles §7 "arena sized for one rung, inner flag for
    # another" trap.) Mimic robots therefore only ever use rung 0 (full smem) or
    # rung 3 (whole pool -> d_workspace); the selective rungs 1/2 collapse onto
    # the full-inner size so the picker never lands a mimic robot on a rung that
    # under-sizes the dense pool.
    _integrator_gradient_full = max(integrator_gradient_t_count, integrator_gradient_with_x_kp1_t_count)
    _integrator_gradient_arenas = compose_arena_rungs("integrator_gradient", self._arena_ctx)   # Step 3.5b fold
    self.integrator_gradient_spill_tier_3way = select_shared_tier_3way(*_integrator_gradient_arenas)
    self.integrator_gradient_t_count_per_tier = tuple(_integrator_gradient_arenas[i] for i in self.integrator_gradient_spill_tier_3way)
    _picks = self.integrator_gradient_spill_tier_3way
    # Placement booleans per rung index: which buffers leave smem.
    self.integrator_gradient_dqdd_in_smem_per_tier  = tuple(p < 1 for p in _picks)  # spilled at rungs >=1
    self.integrator_gradient_dab_in_smem_per_tier    = tuple(p < 2 for p in _picks)  # spilled at rungs >=2
    # FD-grad inner level per tier: 0 full smem, 1 selective (da_df band), 2 global_temp (whole inner).
    # The selective level (1) only exists for the SPARSE (non-mimic) inner; the
    # MIMIC dense inner ignores USE_DA_DF_SPILL, so it must never be emitted at
    # level 1 (that would set USE_DA_DF_SPILL=true against a dense pool that does
    # not honour it). For mimic, rung 2 collapses to level 0 (full smem; its arena
    # equals rung-1's-minus-dAB because the inner can't shrink) and only rung 3
    # routes the whole pool to global (level 2).
    if self.robot_has_mimic_joints():
        self.integrator_gradient_inner_level_per_tier = tuple((0 if p < 3 else 2) for p in _picks)
    else:
        self.integrator_gradient_inner_level_per_tier = tuple((0 if p < 2 else (1 if p == 2 else 2)) for p in _picks)
    # d_workspace floats the gradient needs when ANY tier spills: Dqdd + dAB +
    # the whole inner (rung-3 worst case; the rungs reuse the same regions).
    self.integrator_gradient_workspace_count = (
        (_integrator_gradient_D_qdd_count + _integrator_gradient_dAB_count + _integrator_gradient_inner_full)
        if any(p >= 1 for p in _picks) else 0)
    # da_df selective spill is a sparse-inner-only mechanism (level 1); mimic
    # never uses it (it has no selective level).
    self.integrator_gradient_uses_da_df_spill = (
        (not self.robot_has_mimic_joints()) and any(p == 2 for p in _picks))
    self._integrator_gradient_dqdd_count = _integrator_gradient_D_qdd_count
    self._integrator_gradient_dAB_count = _integrator_gradient_dAB_count
    inverse_dynamics_gradient_temp_layout = self.gen_inverse_dynamics_gradient_temp_layout()
    # Mimic robots emit a DENSE serial inverse_dynamics_gradient inner (no sparse-band spill), so
    # its inner scratch is the dense count; the selective-spill tier doesn't
    # apply (its band offsets are meaningless for the dense layout). Use the
    # dense count for full AND selective so the kernel always sizes smem for
    # the dense buffers and never selects a too-small selective arena.
    _inverse_dynamics_gradient_has_mimic = self.robot_has_mimic_joints()
    inverse_dynamics_gradient_temp_count = self.gen_inverse_dynamics_gradient_inner_temp_mem_size()
    inverse_dynamics_gradient_selective_temp_count = (
        inverse_dynamics_gradient_temp_count if _inverse_dynamics_gradient_has_mimic
        else inverse_dynamics_gradient_temp_layout["selective_shared_count"]
    )
    forward_dynamics_gradient_temp_count = self.gen_forward_dynamics_gradient_inner_temp_mem_size()
    forward_dynamics_gradient_selective_temp_count = max(self.gen_minv_inner_temp_mem_size(), inverse_dynamics_gradient_selective_temp_count)
    # s_vaf is body-indexed (NB bodies). For a MIMIC robot NB > nv so size
    # 18*NB; non-mimic keeps 18*nv/18*n (byte-identical; floating non-mimic has
    # nv > NB so 18*nv already covers the body writes). The id_device path uses
    # the get_num_pos() (==n) flavour for non-mimic to stay byte-identical with
    # the legacy 18*n; mimic robots route through 18*NB so the inner's
    # body-indexed f writes never overflow s_vaf into the XImats region.
    _vaf_cnt = 18 * (self.robot.get_num_joints() if self.robot_has_mimic_joints() else nv)
    _vaf_cnt_id = 18 * (self.robot.get_num_joints() if self.robot_has_mimic_joints() else n)
    id_device_t_count = _vaf_cnt_id + self.gen_inverse_dynamics_inner_temp_mem_size() + XI_size + rt_xfixed_reserve
    minv_device_t_count = self.gen_minv_inner_temp_mem_size() + XI_size + rt_xfixed_reserve
    fd_device_t_count = self.gen_forward_dynamics_inner_temp_mem_size() + XI_size + rt_xfixed_reserve
    inverse_dynamics_gradient_device_t_count = _vaf_cnt + inverse_dynamics_gradient_temp_count + XI_size + rt_xfixed_reserve
    forward_dynamics_gradient_device_t_count = (2*nv*nv) + (_vaf_cnt) + nv + (nv*nv) + forward_dynamics_gradient_temp_count + XI_size + rt_xfixed_reserve
    inverse_dynamics_gradient_t_count_full = (nv + n) + (2*nv*nv) + (_vaf_cnt) + nv + inverse_dynamics_gradient_temp_count + XI_size + rt_xfixed_reserve
    # Canonical input slot: q/qd/u each NUM_JOINTS(=nq=n here)-wide -> s_q_qd_u
    # is 3*n (matches _emit_forward_dynamics_gradient_kernel_body_for_flags'
    # ("s_q_qd_u", 3*nq)). For a FIXED base n==nv so 3*n == old 3*nv+fb
    # byte-identical; FLOATING n>nv reserves the wider input slot (old 3*nv+fb
    # under-reserved -> smem overrun on the s_q_qd_u load).
    forward_dynamics_gradient_t_count_full = (3*n) + (2*nv*nv) + (_vaf_cnt) + nv + (nv*nv) + forward_dynamics_gradient_temp_count + XI_size + rt_xfixed_reserve
    inverse_dynamics_gradient_t_count_selective = inverse_dynamics_gradient_t_count_full - inverse_dynamics_gradient_temp_count + inverse_dynamics_gradient_selective_temp_count
    forward_dynamics_gradient_t_count_selective = forward_dynamics_gradient_t_count_full - forward_dynamics_gradient_temp_count + forward_dynamics_gradient_selective_temp_count
    inverse_dynamics_gradient_t_count_emergency = inverse_dynamics_gradient_t_count_full - inverse_dynamics_gradient_temp_count
    forward_dynamics_gradient_t_count_emergency = forward_dynamics_gradient_t_count_full - forward_dynamics_gradient_temp_count
    # fd_du OUTPUT-spill rung (h2_plus de-gate): the emergency rung routes the
    # whole inner s_temp pool to d_workspace but keeps the OUTPUT bands s_dc_du
    # (2*nv*nv id-gradient band) + s_Minv (nv*nv mass matrix) in smem -> on a big
    # floating robot (h2_plus) those 3*nv*nv floats leave the arena at ~105KB,
    # still over sm_120's ~99KB. This rung additionally repoints s_dc_du/s_Minv
    # to the L2-pinned SO band (crba/fdsva_so output-spill style), shrinking the
    # arena by 3*nv*nv (~77KB) so h2_plus launches (~28KB). Inner pool stays in
    # the GRAD section (offset 0); the outputs sit in the disjoint SO section.
    forward_dynamics_gradient_t_count_output_spill = forward_dynamics_gradient_t_count_emergency - (2*nv*nv + nv*nv)
    # Per-tier picks (perf, lite, minimal). The existing single-pick flags
    # (inverse_dynamics_gradient_spill_tier etc.) are kept = perf pick so today's emit paths
    # are byte-for-byte unchanged; the lite/minimal indices are exposed
    # only as metadata until the per-tier emit work lands.
    _inverse_dynamics_gradient_arenas = compose_arena_rungs("inverse_dynamics_gradient", self._arena_ctx)   # Step 3.5c fold
    _forward_dynamics_gradient_arenas = compose_arena_rungs("forward_dynamics_gradient", self._arena_ctx)   # Step 3.5c fold
    self.inverse_dynamics_gradient_spill_tier_3way = select_shared_tier_3way(*_inverse_dynamics_gradient_arenas)
    self.forward_dynamics_gradient_spill_tier_3way = select_shared_tier_3way(*_forward_dynamics_gradient_arenas)
    self.inverse_dynamics_gradient_spill_tier = self.inverse_dynamics_gradient_spill_tier_3way[0]
    self.forward_dynamics_gradient_spill_tier = self.forward_dynamics_gradient_spill_tier_3way[0]
    self.inverse_dynamics_gradient_use_selective_spill = self.inverse_dynamics_gradient_spill_tier == 1
    self.forward_dynamics_gradient_use_selective_spill = self.forward_dynamics_gradient_spill_tier == 1
    self.inverse_dynamics_gradient_use_global_temp = self.inverse_dynamics_gradient_spill_tier == 2
    # global-temp (whole inner pool -> d_workspace) is used at BOTH the emergency
    # rung (2) and the output-spill rung (3, which also spills s_dc_du/s_Minv).
    self.forward_dynamics_gradient_use_global_temp = self.forward_dynamics_gradient_spill_tier >= 2
    inverse_dynamics_gradient_t_count = _inverse_dynamics_gradient_arenas[self.inverse_dynamics_gradient_spill_tier]
    forward_dynamics_gradient_t_count = _forward_dynamics_gradient_arenas[self.forward_dynamics_gradient_spill_tier]
    # Per-tier t_counts exposed for tier-aware constexpr metadata.
    self.inverse_dynamics_gradient_t_count_per_tier = tuple(_inverse_dynamics_gradient_arenas[i] for i in self.inverse_dynamics_gradient_spill_tier_3way)
    self.forward_dynamics_gradient_t_count_per_tier = tuple(_forward_dynamics_gradient_arenas[i] for i in self.forward_dynamics_gradient_spill_tier_3way)
    # §1e: the aba kernel body reserves a 3*nq-wide per-timestep input slot
    # ("s_q_qd_tau", 3*nq in _emit_aba_kernel_body_for_flags; n == get_num_pos()
    # == nq here); the matching ABA_DYNAMIC_SHARED_MEM_BYTES arena count (now
    # composed in algo_registry's arena ctx) must use the SAME 3*n, not n+2*nv:
    # for a fixed-base cardinal robot nq==nv==n so 3*n == n+2*nv, but on a
    # spherical/floating (nq>nv) base the n+2*nv form under-sizes the launch smem
    # by 3*(nq-nv) floats -> the aba batch kernel OOBs in
    # load_update_XImats_helpers. Mirrors the fd/minv 3*n input slots.
    crba_input_t_count = n + nv
    # ABA surgical-spill ladder, 3 rungs. The 140*NJ+138 inner scratch band
    # keeps its hot recursion in smem and spills only the cold sub-band when
    # possible:
    #   level 0 (full)     : whole inner arena in smem (PERF, byte-identical).
    #   level 1 (surgical) : hot band in smem, cold sub-band -> d_cold. The
    #                        smem arena shrinks to the hot region only.
    #   level 2 (workspace): whole inner arena -> L2-pinned workspace (blunt
    #                        MINIMAL fallback).
    _aba_inner_temp_count = self.gen_aba_inner_temp_mem_size()
    _aba_inner_cold_count = self.gen_aba_inner_cold_mem_size()
    # Hot smem arena at the surgical rung: FIXED reclaims the whole 42*n cold
    # tail (hot ends at 98*n); FLOATING reclaims only the 138-float fb* tail
    # above tempVec (the interior vcross slot still relocates to d_cold but
    # cannot be byte-identically compacted out of smem).
    _aba_arenas = compose_arena_rungs("aba", self._arena_ctx)   # Step 3.5c fold
    self.aba_spill_tier_3way = select_shared_tier_3way(*_aba_arenas)
    aba_t_count = _aba_arenas[self.aba_spill_tier_3way[0]]
    self.aba_t_count_per_tier = tuple(_aba_arenas[i] for i in self.aba_spill_tier_3way)
    self._aba_inner_cold_count = _aba_inner_cold_count
    # CRBA: the inner scratch band is spilled as one band to L2-pinned
    # workspace at LITE/MINIMAL. Level 0 = scratch in smem (current);
    # Level 1 = scratch redirected to workspace.
    # An intermediate surgical-spill rung (keep hot band in smem, spill only a
    # cold sub-band) was investigated for K-crbarung and DEFERRED: post-I-crba
    # the whole inner band (42*NJ / 36*NJ+slab) is hot with no cold sub-band,
    # and the full arena (<=~19 KB) already fits smem at every default tier.
    # See _crba.py header (gen_crba_inner_temp_mem_size) for the full rationale.
    # 3-rung ladder: full | s_M->d_workspace (surgical OUTPUT_SPILL of the nv*nv mass
    # matrix, the dominant write-once buffer, read only by the optional mjx congruence;
    # the HOT inner band + XI stay in smem) | whole-band-to-workspace (MINIMAL fallback).
    # s_M routes to the L2-pinned SO band exactly like dccrba's output. The surgical
    # rung keeps the hot band resident at LITE (vs the blunt whole-band spill it replaces),
    # raising occupancy on big floating robots (h2_plus crba LITE ~47.6KB -> ~34KB).
    (_crba_t_count_full, _crba_t_count_output_spill, _crba_t_count_workspace) = \
        compose_arena_rungs("crba", self._arena_ctx)   # Step 3.2 fold
    self.crba_spill_tier_3way = select_shared_tier_3way(_crba_t_count_full, _crba_t_count_output_spill, _crba_t_count_workspace)
    # whole-band spill (inner band -> d_workspace) fires only at the deepest rung (index 2).
    self.crba_t_count_per_tier = tuple(
        (_crba_t_count_full, _crba_t_count_output_spill, _crba_t_count_workspace)[i] for i in self.crba_spill_tier_3way
    )
    crba_t_count = self.crba_t_count_per_tier[0]
    # osc_inertia (Lambda = (J Minv J^T)^-1): SELF-CONTAINED, composes Minv on
    # device via minv_inner with the heavy F-region kept in a dedicated SHARED
    # s_F buffer (6*nv*nv). On h2_plus that F-region is ~153KB and the full arena
    # ~228KB (UNLAUNCHABLE). 2-rung ladder: full (s_F in smem) | spill-F (s_F ->
    # the L2-pinned minv-F workspace offset, keeping s_Minv + everything else in
    # smem). h2_plus picks spill-F (~74KB) -> LAUNCHES; all other robots fit full.
    _osc_Xhom, _, _ = self.gen_get_Xhom_size()
    (_osc_t_full, _osc_t_spill_F) = compose_arena_rungs("osc_inertia", self._arena_ctx)   # Step 3.5e fold
    self.osc_inertia_spill_tier_3way = select_shared_tier_3way(_osc_t_full, _osc_t_spill_F)
    self.osc_inertia_t_count_per_tier = tuple(
        (_osc_t_full, _osc_t_spill_F)[i] for i in self.osc_inertia_spill_tier_3way)
    ee_t_count = compose_arena_full("end_effector_pose", self._arena_ctx)   # Step 3.1 fold
    # Phase 3d (EE_POSE_GRAD): two-tier spill, mirrors D2EE's (full, spill, spill).
    # Level 0 = full smem (inner_temp + s_end_effector_pose_gradient). Level 1 =
    # inner_temp + s_end_effector_pose_gradient -> workspace/global. The old dXmatsHom
    # spill rung is RETIRED: the geometric-Jacobian gradient inner reads only s_Xhom
    # and (void)s_dXhom, so dXmatsHom was never allocated in smem -> dropping the dead
    # + dXhom_size reservation shrinks the PERF/LITE arenas fleet-wide and pushes the
    # spill threshold outward. inner_temp is recursion-hot but L2-pinned at the host
    # wrapper for spill tiers; s_end_effector_pose_gradient is write-once output.
    _end_effector_pose_gradient_num_ees = self.robot.get_total_leaf_nodes()
    _end_effector_pose_gradient_inner_temp_count = self.gen_end_effector_pose_gradient_inner_temp_mem_size()
    _end_effector_pose_gradient_full_t_count        = n + 6*n*_end_effector_pose_gradient_num_ees + _end_effector_pose_gradient_inner_temp_count + XHom_size
    _end_effector_pose_gradient_spill_temp_t_count  = n                                                    + XHom_size
    # Degenerate 3rd arena == 2nd (MINIMAL = last index always; collapses to spill).
    _end_effector_pose_gradient_arenas = compose_arena_rungs("end_effector_pose_gradient", self._arena_ctx)   # Step 3.5e fold
    self.end_effector_pose_gradient_spill_tier_3way = select_shared_tier_3way(*_end_effector_pose_gradient_arenas)
    self.end_effector_pose_gradient_spill_tier = self.end_effector_pose_gradient_spill_tier_3way[0]
    self.end_effector_pose_gradient_use_workspace_temp = self.end_effector_pose_gradient_spill_tier >= 1
    self.end_effector_pose_gradient_use_workspace_dxhom = False  # dXhom never in smem (the FD inner never uses dXhom)
    dee_t_count = _end_effector_pose_gradient_arenas[self.end_effector_pose_gradient_spill_tier]
    self.end_effector_pose_gradient_t_count_per_tier = tuple(_end_effector_pose_gradient_arenas[i] for i in self.end_effector_pose_gradient_spill_tier_3way)
    # D2EE (FD-on-d/dv-Jacobian): two spill levels (the nv^2 output is the only
    # large buffer that can move out of smem). dXhom/d2Xhom are no longer used
    # by the geometric-Jacobian gradient inner the d2ee inner runs internally.
    _d2ee_num_ees = self.robot.get_total_leaf_nodes()
    d2ee_inner_temp_count = self.gen_end_effector_pose_hessian_inner_temp_mem_size()
    d2ee_output_count = self.gen_end_effector_pose_hessian_output_count()
    d2ee_grad_count = 6 * nv * _d2ee_num_ees
    # full smem: q + grad + d2ee_output + inner_temp + Xhom
    d2ee_full_t_count   = n + d2ee_grad_count + d2ee_output_count + d2ee_inner_temp_count + XHom_size
    _d2ee_arenas = compose_arena_rungs("end_effector_pose_hessian", self._arena_ctx)   # Step 3.5e fold
    if "end_effector_pose_hessian" in getattr(self, "generated_algorithms", set()):
        self.d2ee_spill_tier_3way = select_shared_tier_3way(*_d2ee_arenas)
    else:
        self.d2ee_spill_tier_3way = (0, 0, 0)
    self.d2ee_spill_tier = self.d2ee_spill_tier_3way[0]
    self.d2ee_use_workspace_output = self.d2ee_spill_tier >= 1
    d2ee_t_count = _d2ee_arenas[self.d2ee_spill_tier]
    self.d2ee_t_count_per_tier = tuple(_d2ee_arenas[i] for i in self.d2ee_spill_tier_3way)
    # G2 centroidal quick-wins smem t-counts (no tier spill — new, low perf
    # priority families use the full smem arena).
    NB = self.robot.get_num_bodies()
    # generalized_gravity (the larger of the two ID-bias kernels: + s_qd0):
    #   s_q_qd(2n) + s_out(nv) + s_vaf(18*nb_vaf) + s_qd0(nv) + inner_temp(6n) + XI
    # s_vaf is body-indexed by RAW body id inside inverse_dynamics_inner; for mimic
    # robots get_num_joints() (NB) > get_num_pos() (NV) so it MUST be 18*NB here to
    # match the device wrapper's arena (_centroidal.py:68/96) — else the host arena
    # macro under-budgets and the high-body f-writes overflow shared mem (h1_2:fixed
    # NB=51>NV=39 crashed). Non-mimic (nb_vaf==n) is byte-identical to the old 18*n.
    nb_vaf = self.robot.get_num_joints() if self.robot_has_mimic_joints() else n
    # generalized_gravity / nonlinear_effects share the INVERSE_DYNAMICS_BIAS arena.
    self.id_bias_t_count = compose_arena_full("generalized_gravity", self._arena_ctx)   # Step 3.1 fold
    # com/ccrba/energy (DE-GATE #2): the Jw band (6*nv*NB) is carved as a SEPARATE
    # tier-routed buffer s_J (in-smem tail at the J-in-smem tier, d_workspace at the
    # J-spilled tier), mirroring cmm/dccrba. Each family gets a 2-rung ladder:
    #   L0 keeps s_J in smem (arena byte-equivalent to the old single rung),
    #   L1 spills it -> d_workspace at the shared GRID_DCCRBA_J_OFFSET_BYTES.
    # base = s_in(<=2n) + s_out(<=6nv+6) + s_A(6nv) + s_com(3) + s_extra(4)
    #        + centroidal_inner(no-J) + XHom_size  (the Jw band is added per-tier).
    _centroidal_inner_noJ = 16*self.robot.get_num_joints() + 36*NB + 6*nv + 36
    _centroidal_sJ = 6*nv*NB
    _com_base    = n     + (3 + 3*nv) + 6*nv + 3 + 4 + _centroidal_inner_noJ + XHom_size
    _ccrba_base  = 2*n   + (6*nv + 6) + 6*nv + 3 + 4 + _centroidal_inner_noJ + XHom_size
    _energy_base = 2*n   + 3          + 6*nv + 3 + 4 + _centroidal_inner_noJ + XHom_size
    _com_rungs    = compose_arena_rungs("com", self._arena_ctx)      # Step 3.3 fold
    _ccrba_rungs  = compose_arena_rungs("ccrba", self._arena_ctx)    # Step 3.3 fold
    _energy_rungs = compose_arena_rungs("energy", self._arena_ctx)   # Step 3.3 fold
    self.com_spill_tier_3way    = select_shared_tier_3way(*_com_rungs)
    self.ccrba_spill_tier_3way  = select_shared_tier_3way(*_ccrba_rungs)
    self.energy_spill_tier_3way = select_shared_tier_3way(*_energy_rungs)
    self.com_t_count_per_tier    = tuple(_com_rungs[i] for i in self.com_spill_tier_3way)
    self.ccrba_t_count_per_tier  = tuple(_ccrba_rungs[i] for i in self.ccrba_spill_tier_3way)
    self.energy_t_count_per_tier = tuple(_energy_rungs[i] for i in self.energy_spill_tier_3way)
    self.centroidal_spill_J_ws_count = _centroidal_sJ
    # Size-triggered gravity-shim full-spill. Default OFF; if shim total shared
    # would exceed the target, set self.idsva_so_body_frame_grav_full_spill and
    # let gen_idsva_so_body_frame_inner_temp_mem_size() return the smaller value
    # (the gravity shim's dX/a/da/f/df spill into d_workspace alongside
    # d2X/d2a/d2f). Saves 50-60 KB on g1-class robots without changing
    # iiwa14/go2 behavior.
    def _compute_idsva_body_t_count():
        inner = self.gen_idsva_so_body_frame_inner_temp_mem_size()
        base = (3*n) + inner + XI_size + rt_xfixed_reserve
        full = base + 4*nv**3
        use_global_output = py_arena_bytes(full) > self.cuda_target_shared_mem_bytes
        return base if use_global_output else full, use_global_output

    self.idsva_so_body_frame_grav_full_spill = False
    idsva_so_body_frame_t_count, self.idsva_so_body_frame_use_global_output = _compute_idsva_body_t_count()
    if self.robot.floating_base and py_arena_bytes(idsva_so_body_frame_t_count) > self.cuda_target_shared_mem_bytes:
        self.idsva_so_body_frame_grav_full_spill = True
        idsva_so_body_frame_t_count, self.idsva_so_body_frame_use_global_output = _compute_idsva_body_t_count()
    # Capture the inner temp count for use downstream (world-frame fallback for
    # fixed-base + FDSVA-SO inner sizing) and for the per-tier spill ladder below.
    idsva_so_body_frame_inner_temp_count = self.gen_idsva_so_body_frame_inner_temp_mem_size()

    # ----- idsva_so BODY-frame per-tier spill ladder -----
    # Rungs least->most spill. Flags = (use_global_output, s_temp_in_global, bc_in_global, tp_in_global).
    #   rung0 full:          output + s_temp + BC all in smem
    #   rung1 global_output: 4*NV^3 output tensor -> d_workspace (cheap; coalesced one-shot)
    #   rung2 output_bc:     + BC (36*NB cold buffer, dead before hot loops) -> d_workspace (surgical)
    #   rung3 output_tp:     + ancestor-pair scratch t/p1..p6 (36*len(jids_a), 30-45% of the
    #                          body arena; DEAD through the whole recursion-hot forward sweep,
    #                          live only in the final block-parallel output assembly) ->
    #                          d_workspace. BC stays in smem (slides down to fill the vacated
    #                          t/p span). This surgical rung keeps the entire recursion-hot
    #                          chain in smem and is the highest-payoff cold sub-band on
    #                          humanoid-scale robots.
    #   rung4 output_temp:   + whole s_temp inner arena -> d_workspace (guaranteed-fit fallback)
    # rungs 2 (BC) and 3 (t/p) are mutually exclusive surgical levers (inner enforces it).
    # Fixed-base is the production overflow case. Floating-base BODY is diagnostic
    # (the dispatcher routes floating to the WORLD frame) so it keeps the legacy
    # single-body emit with the gravity shim; its picks are (0,0,0) and unused.
    _idsva_bf_BC = 36 * self.robot.get_num_bodies()
    _idsva_bf_jids_a = len(self.robot.get_jid_ancestor_ids(include_joint=True)[0])
    _idsva_bf_TP = 36 * _idsva_bf_jids_a
    _idsva_bf_base_smem = (3*n) + XI_size                                  # whole s_temp -> global
    _idsva_bf_full     = (3*n) + idsva_so_body_frame_inner_temp_count + XI_size + 4*nv**3 + rt_xfixed_reserve
    _idsva_bf_out      = (3*n) + idsva_so_body_frame_inner_temp_count + XI_size + rt_xfixed_reserve
    _idsva_so_body_tiers = [
        ("full",          _idsva_bf_full,                False, False, False, False),
        ("global_output", _idsva_bf_out,                 True,  False, False, False),
        ("output_bc",     _idsva_bf_out - _idsva_bf_BC,  True,  False, True,  False),
        ("output_tp",     _idsva_bf_out - _idsva_bf_TP,  True,  False, False, True),
        ("output_temp",   _idsva_bf_base_smem,           True,  True,  False, False),
    ]
    self._idsva_so_body_tier_table = _idsva_so_body_tiers
    if self.robot.floating_base:
        # Diagnostic path: keep legacy single-body emit + grav shim (no ladder).
        # The body ladder is NOT composed here: floating body inner is grav_full_spill-
        # dependent (picker-driven), so it is not ctx-pure — the single-value override
        # stays in-gen and the body rungs are asserted/captured only on fixed base.
        self.idsva_so_body_frame_spill_tier_3way = (0, 0, 0)
        self.idsva_so_body_frame_t_count_per_tier = (idsva_so_body_frame_t_count,) * 3
        self.idsva_so_body_frame_use_ladder = False
    else:
        _idsva_so_body_arenas = compose_arena_rungs("idsva_so_body_frame", self._arena_ctx)   # Step 3.5 fold (fixed-base ladder)
        self.idsva_so_body_frame_spill_tier_3way = select_shared_tier_3way(*_idsva_so_body_arenas)
        self.idsva_so_body_frame_t_count_per_tier = tuple(_idsva_so_body_arenas[i] for i in self.idsva_so_body_frame_spill_tier_3way)
        self.idsva_so_body_frame_use_ladder = True
        # Keep the const flag accurate to the PERF-tier pick.
        self.idsva_so_body_frame_use_global_output = _idsva_so_body_tiers[self.idsva_so_body_frame_spill_tier_3way[0]][2]

    # world-frame path has its own (smaller) scratch — no gravity-shim shared, no
    # main-sweep extras. Sized via gen_idsva_so_world_frame_temp_mem_size.
    idsva_so_world_frame_inner_temp_count = self.gen_idsva_so_world_frame_temp_mem_size() if self.robot.floating_base else idsva_so_body_frame_inner_temp_count
    idsva_so_world_frame_base_t_count = (3*n) + idsva_so_world_frame_inner_temp_count + XI_size + rt_xfixed_reserve
    idsva_so_world_frame_full_t_count = idsva_so_world_frame_base_t_count + 4*nv**3
    # ----- idsva_so WORLD-frame per-tier spill ladder -----
    # Flags = (use_global_output, s_temp_in_global, cold_in_global). The world inner
    # is UN-aliased, so a surgical rung is landed: the cold QUAD Xup (36*NB, dead after
    # the Step-4 IC build) + Xdown (36*NB, dead after Step 3) + v_w/a_w (6*NB each, dead
    # after Step 4's f_w build) = 84*NB can move to d_workspace while the hot arena stays
    # in smem (inner COLD_IN_SMEM=false).
    # Rungs least->most spill:
    #   full:               output + whole s_temp arena in smem
    #   global_output:      4*NV^3 output tensor -> d_idsva_so global (coalesced one-shot)
    #   output_cold:        + surgical cold quad (Xup 36*NB + Xdown 36*NB + v_w/a_w 12*NB = 84*NB) -> d_workspace
    #   output_temp:        + whole s_temp inner arena -> d_workspace (guaranteed-fit fallback)
    _idsva_wf_cold = self.gen_idsva_so_world_cold_floats()
    _idsva_wf_base_smem = (3*n) + XI_size
    _idsva_so_world_tiers = [
        ("full",          idsva_so_world_frame_full_t_count,                  False, False, False),
        ("global_output", idsva_so_world_frame_base_t_count,                  True,  False, False),
        ("output_cold",   idsva_so_world_frame_base_t_count - _idsva_wf_cold, True,  False, True),
        ("output_temp",   _idsva_wf_base_smem,                                True,  True,  False),
    ]
    self._idsva_so_world_tier_table = _idsva_so_world_tiers
    _idsva_so_world_arenas = compose_arena_rungs("idsva_so_world_frame", self._arena_ctx)   # Step 3.5 fold
    self.idsva_so_world_frame_spill_tier_3way = select_shared_tier_3way(*_idsva_so_world_arenas)
    self.idsva_so_world_frame_t_count_per_tier = tuple(_idsva_so_world_arenas[i] for i in self.idsva_so_world_frame_spill_tier_3way)
    # d_workspace floats needed per timestep by the idsva_so spill rungs (for so_workspace sizing).
    # Body rungs: 4=output_temp (whole inner arena), 3=output_tp (36*len(jids_a) ancestor-pair
    # scratch), 2=output_bc (36*NB cold slab); 0/1 spill nothing into d_workspace.
    def _idsva_body_ws_floats(pick):
        if pick == 4:
            # whole s_temp routed to workspace; the XImats helper still rebuilds
            # Xfixed into it (offset 2*num_pos) so the workspace must reserve it.
            return idsva_so_body_frame_inner_temp_count + rt_xfixed_reserve
        if pick == 3:
            return _idsva_bf_TP
        if pick == 2:
            return _idsva_bf_BC
        return 0
    def _idsva_world_ws_floats(pick):
        # pick 3 (output_temp) spills the whole inner arena; pick 2 (output_cold)
        # spills just the surgical cold quad (Xup 36*NB + Xdown 36*NB + v_w/a_w 12*NB).
        if pick == 3:
            # whole s_temp routed to workspace; XImats helper rebuilds Xfixed
            # into it (offset 2*num_pos) so the workspace must reserve it.
            return idsva_so_world_frame_inner_temp_count + rt_xfixed_reserve
        if pick == 2:
            return self.gen_idsva_so_world_cold_floats()
        return 0
    idsva_so_spill_ws_t_count = max(
        [_idsva_body_ws_floats(p) for p in self.idsva_so_body_frame_spill_tier_3way] +
        [_idsva_world_ws_floats(p) for p in self.idsva_so_world_frame_spill_tier_3way] + [0])

    # ----- FDSVA_SO shared-mem tier selection -----
    # Four nested tiers, ordered from least-spill to most-spill. Pick the
    # lowest-spill tier whose shared-arena bytes fit cuda_target_shared_mem.
    # Each tier sets three orthogonal state flags read by gen_fdsva_so_*:
    #   - use_global_tensors:  s_idsva_so + s_df2 (8*nv³ outputs) -> d_workspace
    #   - use_workspace_temp:  s_fdsva_temp (4*nv³ inner) -> d_workspace
    #   - fd_grad_use_spill:   fd_grad_inline's da_dq..fxvi band -> d_workspace grad section
    fdsva_so_base_t_count = 4*nv + nv*nv + nv + 2*nv*nv + XI_size
    fdsva_so_contract_temp_count = 4*nv**3
    fdsva_so_fd_gradient_inline_temp_count = self.gen_fdsva_so_fd_gradient_inline_temp_mem_size()
    fdsva_so_fd_gradient_inline_spilled_count = self.gen_fdsva_so_fd_gradient_inline_temp_mem_size_spilled()
    # fdsva_so dispatches to world_frame_inner for floating-base (smaller
    # footprint + no grav-shim spill) and body_frame_inner for fixed-base.
    fdsva_so_inner_idsva_so_temp_count = (
        idsva_so_world_frame_inner_temp_count if self.robot.floating_base
        else idsva_so_body_frame_inner_temp_count
    )
    # runtime_transform: the composed XImats helper rebuilds Xfixed into this
    # shared s_temp pool (offset 2*num_pos), so every pool variant must reserve
    # the 36*NB Xfixed band on top of its inner peak (dead after the helper).
    _temp_full     = max(fdsva_so_inner_idsva_so_temp_count, fdsva_so_contract_temp_count, fdsva_so_fd_gradient_inline_temp_count) + rt_xfixed_reserve
    _temp_no_contract = max(fdsva_so_inner_idsva_so_temp_count, fdsva_so_fd_gradient_inline_temp_count) + rt_xfixed_reserve
    _temp_spilled  = max(fdsva_so_inner_idsva_so_temp_count, fdsva_so_fd_gradient_inline_spilled_count) + rt_xfixed_reserve
    # Phase 3e: extend to 6 levels. Each level pushes an additional buffer
    # to L2-pinned workspace. Tuple is
    # (name, shared_count, use_global_tensors, use_workspace_temp,
    #  fd_grad_use_spill, use_workspace_df_du, use_workspace_Minv).
    # Levels 0-3 unchanged from pre-Phase-3e. Level 4 pushes s_df_du
    # (2*NV²); Level 5 also pushes s_Minv (NV²).
    fdsva_so_base_no_df_du = fdsva_so_base_t_count - 2*nv*nv
    fdsva_so_base_no_df_du_no_Minv = fdsva_so_base_no_df_du - nv*nv
    # Level 6: pool -> global. fdsva_so_device runs with SCRATCH_IN_SMEM=false,
    # routing the WHOLE shared s_temp pool (helper sincos + minv + fd + fd_grad +
    # idsva) to d_workspace (reusing the non-concurrent contraction SO-temp region).
    # Smem then holds only the base: inputs + s_qdd + s_Minv + s_df_du + XI
    # (~46-54 KB on h1_2 -> fits the ~99 KB cap). Outputs->device arrays and
    # contraction->global as in levels >=2. Works for BOTH bases because the full
    # inner repoints s_temp and hands the placed pool to the idsva inner (body or
    # world) — the sub-inner just uses the pointer it is given (inner-owns-placement).
    # A4: idsva_cold rung (between global_tensors and workspace_temp). Keeps outputs
    # in global (like global_tensors) but additionally spills the embedded WORLD
    # idsva_so inner's cold quad (Xup 36*NB + Xdown 36*NB + v_w/a_w 12*NB = 84*NB floats,
    # dead before the hot triple-walk) to the SO-temp d_workspace region via the inner's
    # COLD_IN_SMEM=false, while the hot pool stays in smem. Its smem arena = the
    # global_tensors arena minus the cold-quad span. Only EFFECTIVE when the composed
    # idsva inner is the WORLD frame (floating OR spherical) — the body-frame inner
    # has no exposed cold quad, so for body-frame robots this rung's arena is set
    # EQUAL to global_tensors (no fit advantage -> the picker never distinguishes it,
    # and the kernel's COLD flag is inert there). The reduction matches the standalone
    # idsva_so world cold rung (_idsva_wf_cold).
    _fdsva_so_uses_world_idsva = self.robot.floating_base or self.robot.robot_has_spherical()
    _fdsva_so_cold_floats = self.gen_idsva_so_world_cold_floats() if _fdsva_so_uses_world_idsva else 0
    # idsva_cold rung pool: the idsva world inner spills its cold quad to GLOBAL, so ONLY the
    # idsva term shrinks -- the reduction goes INSIDE the max. The pool is a max over three
    # INDEPENDENT consumers and the CONTRACTION (4nv^3) dominates on a floating quadruped
    # (go2-floating: idsva=3030, contraction=23328, fdg=10494), so shrinking idsva changes the
    # max by NOTHING. The old `_temp_full - _fdsva_so_cold_floats` cut the TOTAL instead,
    # under-reserving the launch by 1077 elems (4308 B) while the kernel still carved the full
    # pool -> fdsva_so_kernel wrote past shared memory ("illegal memory access", go2-floating @
    # TIER_SHARED, every thread count). See docs/agent_debugging_guide.md §1t.
    # NOTE the arena values in the 9-tuple below are DEAD (the composer has supplied arenas
    # since Step 3.4) -- but they must stay CORRECT so they can't mislead a future reader.
    _temp_idsva_cold = max(fdsva_so_inner_idsva_so_temp_count - _fdsva_so_cold_floats,
                           fdsva_so_contract_temp_count,
                           fdsva_so_fd_gradient_inline_temp_count) + rt_xfixed_reserve
    # 9-tuple: (..., use_workspace_idsva_temp == pool->global, idsva_cold_in_global). Levels keep pool in smem except pool_global.
    _fdsva_so_tiers = [
        ("full",                 fdsva_so_base_t_count + 8*nv**3 + _temp_full,  False, False, False, False, False, False, False),
        ("global_tensors",       fdsva_so_base_t_count + _temp_full,            True,  False, False, False, False, False, False),
        ("idsva_cold",           fdsva_so_base_t_count + _temp_idsva_cold,      True,  False, False, False, False, False, True),
        ("workspace_temp",       fdsva_so_base_t_count + _temp_no_contract,        True,  True,  False, False, False, False, False),
        ("workspace_temp_spill", fdsva_so_base_t_count + _temp_spilled,         True,  True,  True,  False, False, False, False),
        ("spill_df_du",          fdsva_so_base_no_df_du + _temp_spilled,        True,  True,  True,  True,  False, False, False),
        ("spill_Minv",           fdsva_so_base_no_df_du_no_Minv + _temp_spilled,True,  True,  True,  True,  True,  False, False),
        # pool->global: smem = base (inputs + qdd + Minv + df_du + XI), no pool/outputs/contraction.
        ("pool_global",          fdsva_so_base_t_count,                         True,  True,  False, False, False, True,  False),
    ]
    _fdsva_so_arenas = compose_arena_rungs("fdsva_so", self._arena_ctx)   # Step 3.4 fold
    # S1 (h2_plus smem triage): on nv~81 humanoids even the guaranteed-fit
    # pool_global rung overflows the device target — its BASE (s_Minv nv² +
    # s_df_du 2nv² + inputs + XI) alone is ~104 KB vs the ~99 KB cap, so every
    # tier clamps to an unlaunchable arena. Extend the ladder with a
    # pool_global + df_du->workspace rung (s_df_du rides the already-reserved
    # GRID_FDSVA_SO_SPILL section, pick>=5). In-gen override, NOT in the
    # registry ladder: the fit predicate needs py_arena_bytes + the device
    # target, which are not ctx-pure (same reasoning as the floating
    # body-frame idsva single-value override above). Conditional so every
    # robot whose pool_global fits keeps a byte-identical ladder.
    if py_arena_bytes(_fdsva_so_arenas[-1]) > self.cuda_target_shared_mem_bytes:
        _fdsva_so_tiers.append(
            ("pool_global_spill_df_du", fdsva_so_base_no_df_du,
             True, True, False, True, False, True, False))
        _fdsva_so_arenas = _fdsva_so_arenas + (fdsva_so_base_no_df_du,)
    self.fdsva_so_spill_tier_3way = select_shared_tier_3way(*_fdsva_so_arenas)
    # arena counts now come from the composer; the 9-tuple STATE FLAGS stay in-gen.
    _chosen = _fdsva_so_tiers[self.fdsva_so_spill_tier_3way[0]]
    (_, _, self.fdsva_so_use_global_tensors,
     self.fdsva_so_use_workspace_temp, self.fdsva_so_fd_grad_use_spill,
     self.fdsva_so_use_workspace_df_du, self.fdsva_so_use_workspace_Minv,
     self.fdsva_so_use_workspace_idsva_temp, self.fdsva_so_idsva_cold_in_global) = _chosen
    fdsva_so_t_count = _fdsva_so_arenas[self.fdsva_so_spill_tier_3way[0]]
    self.fdsva_so_t_count_per_tier = tuple(_fdsva_so_arenas[i] for i in self.fdsva_so_spill_tier_3way)

    # ----- F1: plant_step_hessian shared-mem tier selection (both bases) -----
    # The hessian kernel composes fdsva_so_device and stages its 18*nv^3 output
    # band s_d2AB. Two tiers (mirrors _PLANT_HESSIAN_PICK_FLAGS in _plant.py):
    #   tier 0 (full smem): base + s_d2AB(18nv^3) + s_df2+s_idsva_so(8nv^3) + pool
    #   tier 1 (deep spill): base only in smem; d2AB + fdsva tensors + pool -> global
    # PERF picks the lowest tier whose arena fits cuda_target_shared_mem; LITE
    # clamps >= PERF; MINIMAL is always the deep-spill tier. Floating base IS
    # supported (routed to gen_integrator_hessian_device_floating); only
    # multi-stage RK static_asserts out. Computed unconditionally.
    (_psh_t_full, _psh_t_spill) = compose_arena_rungs("integrator_hessian", self._arena_ctx)   # Step 3.5e fold
    self.plant_step_hessian_spill_tier_3way = select_shared_tier_3way(_psh_t_full, _psh_t_spill)

    # ── Descriptor-table Step 3 parity net (item M) ──────────────────────
    # Ground-truth capture of each algo's FULL (least-spill / rung-0) arena
    # t_count, keyed by ALGO_DESCRIPTORS key. This EMITS NOTHING — it snapshots
    # the hand-written arena math above so test/test_algo_descriptor_arena_parity.py
    # can assert the descriptor `arena_regions` composer reproduces it exactly on
    # every matrix robot (the Step-0-style safety net that de-risks driving the
    # arena sites from the table). See docs/open-tasks/design_descriptor_table_spec.md.
    self._arena_full_t_counts = {
        "inverse_dynamics":                    id_t_count,
        "inverse_dynamics_regressor":          _idr_t_count_full,
        "kinetic_energy_regressor":            self.kinetic_energy_regressor_t_count,
        "potential_energy_regressor":          self.potential_energy_regressor_t_count,
        "coriolis_matrix":                     _coriolis_t_full,
        "cmm_time_variation":                  _cmm_t_count_full,
        "dccrba":                              _dccrba_L0,
        "forward_dynamics_parameter_gradient": _fpg_t_count_full,
        "f_ext_gradient":                      _feg_t_count_full,
        "f_ext_gradient_dq":                   _feg_dq_t_count_full,
        "minv":                                _minv_t_count_full,
        "forward_dynamics":                    _fd_t_count_full,
        "integrator":                          _integrator_t_count_full,
        "integrator_gradient":                 _integrator_gradient_full,
        "integrator_with_gradient":            _integrator_gradient_full,
        "inverse_dynamics_gradient":           inverse_dynamics_gradient_t_count_full,
        "forward_dynamics_gradient":           forward_dynamics_gradient_t_count_full,
        "aba":                                 _aba_arenas[0],
        "crba":                                _crba_t_count_full,
        "osc_inertia":                         _osc_t_full,
        "end_effector_pose":                   ee_t_count,
        "end_effector_pose_gradient":          _end_effector_pose_gradient_full_t_count,
        "end_effector_pose_hessian":           d2ee_full_t_count,
        "generalized_gravity":                 self.id_bias_t_count,
        "nonlinear_effects":                   self.id_bias_t_count,
        "com":                                 _com_base + _centroidal_sJ,
        "ccrba":                               _ccrba_base + _centroidal_sJ,
        "energy":                              _energy_base + _centroidal_sJ,
        # SO-dispatch monsters (per_base_override / workspace / dispatch aliases):
        # captured here for the 3.4/3.5 fold commits, NOT composed in Step 3.0.
        "idsva_so_body_frame":                 (_idsva_bf_full if not self.robot.floating_base
                                                else idsva_so_body_frame_t_count),
        "idsva_so_world_frame":                _idsva_so_world_arenas[0],
        "fdsva_so":                            _fdsva_so_arenas[0],
        "integrator_hessian":                  _psh_t_full,
    }
    # Spill-ladder rung arenas (least-spill first). The generator DRIVES every
    # `select_shared_tier_3way` call above from `compose_arena_rungs`; this snapshot
    # is the descriptor table's canonical arena record + what the invariant parity
    # test reads (test/test_algo_descriptor_arena_parity.py). The body ladder is
    # fixed-base only (floating body uses the grav_full_spill picker override, which
    # is NOT ctx-pure), so it is captured only there.
    self._arena_rung_t_counts = {
        k: compose_arena_rungs(k, self._arena_ctx)
        for k in sorted(ARENA_RUNG_KEYS)
        if not (k == "idsva_so_body_frame" and self.robot.floating_base)
    }
    # In-generation descriptor-arena invariants (runs on EVERY robot generated): every
    # composed rung is positive, and — for keys whose FULL is composed — rung[0] equals
    # the full arena. Cheap structural guard against a future closure edit producing a
    # negative/absurd/inconsistent arena (the §2 under-size bug class). rt-reservation
    # correctness is checked in the parity test (needs a no-rt ctx variant).
    for _k, _rungs in self._arena_rung_t_counts.items():
        assert all(r > 0 for r in _rungs), f"descriptor arena {_k}: non-positive rung {_rungs}"
        if _k in ARENA_COMPOSED_KEYS:
            _full = compose_arena_full(_k, self._arena_ctx)
            assert _rungs[0] == _full, f"descriptor arena {_k}: rung[0] {_rungs[0]} != full {_full}"

    # Phase 3a: include Minv-F count if Minv is spilling (collisions are OK
    # because Minv runs before inverse_dynamics_gradient / forward_dynamics_gradient in any kernel that
    # composes both — they sequentially reuse the same workspace bytes).
    _minv_F_workspace_count = self.gen_minv_inner_F_size() if any(p == 1 for p in self.minv_spill_tier_3way) else 0
    # osc_inertia spill-F rung routes its 6*nv*nv minv F-region (s_F) to the same
    # minv-F workspace offset (offset 0 of the GRAD section); cover it in the max.
    _osc_F_workspace_count = self.gen_minv_inner_F_size() if any(p >= 1 for p in self.osc_inertia_spill_tier_3way) else 0
    # f_ext_gradient deep rung (2) routes minv's 6*nv*nv F-region to the SAME
    # GRAD-section minv-F offset (offset 0); never co-runs with minv/osc, so the
    # max-fold is free. (self.f_ext_gradient_spill_minv_F_count is 0 unless deep.)
    _feg_minv_F_workspace_count = self.f_ext_gradient_spill_minv_F_count
    # CRBA whole-arena spill: when crba_inner's scratch band is redirected to
    # d_workspace (LITE/MINIMAL, or a forced deep-spill tier), the per-timestep
    # workspace must be able to back the full 140*NJ-class band. Include it in
    # the grad-section max so the allocation always covers it regardless of the
    # tier the kernel template is instantiated with.
    _crba_inner_temp_count = self.gen_crba_inner_temp_mem_size()
    # coriolis whole-band spill: at the deepest rung the inner spatial-recursion
    # scratch redirects to the GRAD section, so it must back the full inner band.
    _coriolis_inner_temp_count = self.gen_coriolis_matrix_inner_temp_mem_size() if any(p >= 2 for p in self.coriolis_matrix_spill_tier_3way) else 0
    # SO-region band gating locals (see the GRID_WS_SO_REGION_LIVE emission
    # below): only bench headers (emit_alloc_gating=True) get the gated
    # form; default emission stays byte-identical.
    _ws_gating = getattr(self, "emit_alloc_gating", False)

    def _ag_expr(keys):
        return " || ".join(["!defined(GRID_ALLOC_GATE)"]
                           + ["GRID_ALLOC_" + k.upper() for k in keys])

    grad_spill_workspace_t_count = max(inverse_dynamics_gradient_temp_layout["spill_count"],
                                       inverse_dynamics_gradient_temp_count,
                                       forward_dynamics_gradient_temp_count,
                                       2*nv*nv,
                                       _crba_inner_temp_count,
                                       _coriolis_inner_temp_count,
                                       _minv_F_workspace_count,
                                       _osc_F_workspace_count,
                                       _feg_minv_F_workspace_count,
                                       self.integrator_minv_F_workspace_count,
                                       self.integrator_gradient_workspace_count)
    # D2EE needs no d_workspace: under the FD-on-Jacobian inner the only large
    # buffer is the nv^2 output, and when spilled the inner writes directly into
    # d_end_effector_pose_hessian (the persistent output) -- not a per-timestep workspace slice.
    d2ee_workspace_t_count = 0
    # Phase 3d: max workspace required by EE_POSE_GRAD across any tier (PERF
    # may pick 0, but the workspace allocation must cover what LITE/MINIMAL
    # need at runtime when the user switches tier via the kernel template).
    end_effector_pose_gradient_workspace_t_count = 0
    if any(p >= 1 for p in self.end_effector_pose_gradient_spill_tier_3way):
        end_effector_pose_gradient_workspace_t_count += _end_effector_pose_gradient_inner_temp_count
    # dXhom workspace clause RETIRED: the dxhom-spill rung is gone (pick 2 == pick 1),
    # so no tier materializes s_dXmatsHom -> the phantom dXhom reservation is dropped.
    # Include the floating-base gravity-shim spill (Phase D): the d2X/d2a/d2f
    # tensors live in d_workspace instead of shared memory for larger robots.
    idsva_so_body_frame_grav_spill_t_count = self.gen_floating_gravity_d2tau_dq_spill_count() if self.robot.floating_base else 0
    # g1-spill: fd_parameter_gradient (s_Y) and f_ext_gradient (s_dqdd_dfext)
    # surgically spill into this same SO workspace section when their tier picks
    # level >= 1. They never run concurrently with the SO/d2ee/end_effector_pose_gradient kernels,
    # so reuse is safe and costs no new allocation. Fold their spill counts into
    # the max so the per-timestep workspace always covers them.
    _fpg_spill_ws = self.forward_dynamics_parameter_gradient_spill_Y_ws_count if any(p >= 1 for p in self.forward_dynamics_parameter_gradient_spill_tier_3way) else 0
    _idr_spill_ws = self.inverse_dynamics_regressor_spill_Y_ws_count if any(p >= 1 for p in self.inverse_dynamics_regressor_spill_tier_3way) else 0
    _feg_spill_ws = self.f_ext_gradient_spill_out_ws_count if any(p >= 1 for p in self.f_ext_gradient_spill_tier_3way) else 0
    # f_ext_gradient_dq spills its MIMIC per-sub slab (6*nsub) into the same SO band.
    _feg_dq_spill_ws = self.f_ext_gradient_dq_spill_slab_ws_count if any(p >= 1 for p in self.f_ext_gradient_dq_spill_tier_3way) else 0
    # PS5 dccrba: its 6*nv*nv output (L1+) and the Jw sweep band (L2, DE-GATE #2)
    # surgically spill into this same SO band at DISTINCT sub-offsets, so at L2 the
    # band must hold BOTH simultaneously (out at the base, s_J at base+6*nv*nv).
    # cmm spills only s_J (L1). Never runs concurrently with the SO kernels, so the
    # max-fold is free (8*nv^3 dwarfs out+s_J on the big robots this de-gates).
    _dccrba_spill_ws = 0
    if any(p >= 2 for p in self.dccrba_spill_tier_3way):
        _dccrba_spill_ws = self.dccrba_spill_out_ws_count + self.dccrba_spill_J_ws_count
    elif any(p >= 1 for p in self.dccrba_spill_tier_3way):
        _dccrba_spill_ws = self.dccrba_spill_out_ws_count
    # cmm ALWAYS needs the leading 6*nv*nv region: the two-stage deterministic
    # contraction stages its qd-scaled partials there (the dccrba OUTPUT
    # sub-region at SO_TEMP_OFFSET, which cmm never otherwise writes; kernels
    # never run concurrently so the overlay is free). At the J-spilled tier the
    # Jw band additionally sits after it at GRID_DCCRBA_J_OFFSET_BYTES =
    # SO_TEMP_OFFSET + 6*nv*nv*sizeof(T), so the term also spans the band.
    _cmm_spill_ws = 6 * nv * nv + (self.cmm_time_variation_spill_J_ws_count if any(p >= 1 for p in self.cmm_time_variation_spill_tier_3way) else 0)
    # com/ccrba/energy place their Jw band at the SAME GRID_DCCRBA_J_OFFSET_BYTES
    # sub-offset (SO_TEMP_OFFSET + 6*nv*nv), so the band must span that offset region
    # plus the Jw band itself when any of them spills (same robots cmm spills).
    _centroidal_spill_ws = (6 * nv * nv + self.centroidal_spill_J_ws_count) if any(
        p >= 1 for p in (self.com_spill_tier_3way + self.ccrba_spill_tier_3way + self.energy_spill_tier_3way)) else 0
    so_workspace_t_count = max(8*max(nv**3, 1), d2ee_workspace_t_count, end_effector_pose_gradient_workspace_t_count, idsva_so_body_frame_grav_spill_t_count, idsva_so_spill_ws_t_count, _fpg_spill_ws, _idr_spill_ws, _feg_spill_ws, _feg_dq_spill_ws, _dccrba_spill_ws, _cmm_spill_ws, _centroidal_spill_ws)
    # Deprecated launch-count constants remain for external callers that still
    # pass COUNT*sizeof(T).  Make them conservative aliases for the byte arena
    # layouts so those callers do not under-allocate int topology helpers or
    # 16-byte alignment padding.
    legacy_count_pad = topology_count + 8
    legacy_arena_count = lambda t_count: int(t_count + legacy_count_pad)
    _b = lambda flag: "true" if flag else "false"
    # GRID_LINALG_NVIDIA_MAX_HELPER_BYTES is a stub returning 0 (v2.0+).
    # Forward-declared here so the *_DYNAMIC_SHARED_MEM_BYTES helpers
    # below compile before _lin_alg_helpers emits the definition.
    self.gen_add_code_line("template <typename T> __host__ __device__ constexpr size_t GRID_LINALG_NVIDIA_MAX_HELPER_BYTES();")
    self.gen_add_code_lines(["const int NUM_JOINTS = " + str(self.robot.get_num_pos()) + ";", \
                             "const int NUM_POS = " + str(self.robot.get_num_pos()) + ";", \
                             "const int NUM_VEL = " + str(self.robot.get_num_vel()) + ";", \
                             "const int NUM_BODIES = " + str(self.robot.get_num_bodies()) + ";", \
                             "const int SECOND_ORDER_COORDS = " + str(self.robot.get_num_vel()) + ";", \
                             "const int SECOND_ORDER_TENSOR_SIZE = " + str(4 * self.robot.get_num_vel()**3) + ";", \
                             "const int Q_QD_U_STRIDE = " + str(3 * self.robot.get_num_pos()) + ";", \
                             "// h_q_qd_u / h_q_qd_qdd input ABI (PUBLISHED — docs: user_guide/concepts/input_output_abi):", \
                             "// three NUM_POS-wide slots per timestep (stride Q_QD_U_STRIDE = 3*NUM_POS):", \
                             "//   q at +GRID_Q_OFFSET | qd at +GRID_QD_OFFSET | u (or qdd) at +GRID_U_OFFSET.", \
                             "// qd/u/qdd are passed at nq width; floating base: nv live values in the LEADING", \
                             "// slots + one trailing pad each. Matrix/gradient OUTPUTS are nv-wide. Do NOT pack", \
                             "// tightly: on a floating base nq > nv, so a tight u lands at nq+nv while kernels", \
                             "// read 2*nq — in-bounds and silently wrong. Fixed base (nq == nv) cannot expose this.", \
                             "const int GRID_Q_OFFSET = 0;", \
                             "const int GRID_QD_OFFSET = " + str(self.robot.get_num_pos()) + ";", \
                             "const int GRID_U_OFFSET = " + str(2 * self.robot.get_num_pos()) + ";", \
                             "const int GRID_QDD_OFFSET = " + str(2 * self.robot.get_num_pos()) + ";", \
                             "const int NUM_EES = " + str(self.robot.get_total_leaf_nodes()) + ";", \
                             "const int TOPOLOGY_HELPERS_COUNT = " + str(topology_count) + ";", \
                             "const int DYNAMICS_XI_T_COUNT = " + str(XI_size) + ";", \
                             "const int XHOM_T_COUNT = " + str(XHom_size) + ";", \
                             "const int DXHOM_T_COUNT = " + str(dXhom_size) + ";", \
                             "const int D2XHOM_T_COUNT = " + str(d2Xhom_size) + ";", \
                             "const int GRID_INVERSE_DYNAMICS_GRADIENT_USES_GLOBAL_TEMP = " + str(int(self.inverse_dynamics_gradient_use_global_temp)) + ";", \
                             "const int GRID_INVERSE_DYNAMICS_GRADIENT_USES_WORKSPACE_ANY_TIER = " + str(1 if any(p >= 1 for p in self.inverse_dynamics_gradient_spill_tier_3way) else 0) + ";", \
                             "const int GRID_FORWARD_DYNAMICS_GRADIENT_USES_GLOBAL_TEMP = " + str(int(self.forward_dynamics_gradient_use_global_temp)) + ";", \
                             "const int GRID_FORWARD_DYNAMICS_GRADIENT_USES_WORKSPACE_ANY_TIER = " + str(1 if any(p >= 1 for p in self.forward_dynamics_gradient_spill_tier_3way) else 0) + ";", \
                             "const int GRID_INVERSE_DYNAMICS_GRADIENT_USES_DA_DF_SPILL = " + str(int(self.inverse_dynamics_gradient_use_selective_spill)) + ";", \
                             "const int GRID_FORWARD_DYNAMICS_GRADIENT_USES_DA_DF_SPILL = " + str(int(self.forward_dynamics_gradient_use_selective_spill)) + ";", \
                             "const int GRID_INTEGRATOR_USES_WORKSPACE = " + str(int(any(p == 1 for p in self.integrator_spill_tier_3way))) + ";", \
                             "const int GRID_INTEGRATOR_GRADIENT_USES_WORKSPACE = " + str(int(any(p >= 1 for p in self.integrator_gradient_spill_tier_3way))) + ";", \
                             "const int GRID_INTEGRATOR_GRADIENT_USES_DA_DF_SPILL = " + str(int(self.integrator_gradient_uses_da_df_spill)) + ";", \
                             "const int GRID_GENERATES_IDSVA_SO_BODY_FRAME = " + str(int(getattr(self, "generate_idsva_so_body_frame", True))) + ";", \
                             "const int GRID_GENERATES_FDSVA_SO = " + str(int(getattr(self, "generate_fdsva_so", True))) + ";", \
                             "const int GRID_GENERATES_D2EE = " + str(int(getattr(self, "generate_end_effector_pose_hessian", True))) + ";", \
                             "const int GRID_IDSVA_SO_USES_GLOBAL_OUTPUT = " + str(int(self.idsva_so_body_frame_use_global_output)) + ";", \
                             "const int GRID_FDSVA_SO_USES_GLOBAL_TENSORS = " + str(int(self.fdsva_so_use_global_tensors)) + ";", \
                             # Single-bool kept for inline-CUDA back-compat (reflects PERF-pick only).
                             "const int GRID_FDSVA_SO_USES_WORKSPACE_TEMP = " + str(int(self.fdsva_so_use_workspace_temp)) + ";", \
                             # Per-tier-aware gate (true if ANY of PERF/LITE/MINIMAL spills any band
                             # of the s_temp pool to d_workspace; picks >= 2 cover spill_temp,
                             # spill_fd_grad_band, spill_df_du, spill_Minv, pool_global). The host
                             # uses this to pin L2 persistence on d_workspace for the per-tier path.
                             "const int GRID_FDSVA_SO_USES_WORKSPACE_ANY_TIER = " + str(1 if any(p >= 2 for p in self.fdsva_so_spill_tier_3way) else 0) + ";", \
                             # GRID_END_EFFECTOR_POSE_HESSIAN_USES_WORKSPACE_TEMP: 1 if the PERF tier spills the d2ee output
                             # (the only large buffer in the new FD-on-Jacobian path) to global memory.
                             # When spilled, the inner writes directly into d_end_effector_pose_hessian (the persistent
                             # output buffer) -- no extra per-timestep workspace slice is used. The old
                             # d2xhom-spill bit is permanently 0 (the FD inner never touches d2Xhom).
                             "const int GRID_END_EFFECTOR_POSE_HESSIAN_USES_WORKSPACE_TEMP = " + str(int(self.d2ee_use_workspace_output)) + ";", \
                             "const int GRID_END_EFFECTOR_POSE_HESSIAN_USES_WORKSPACE_D2XHOM = 0;", \
                             "const int GRID_END_EFFECTOR_POSE_HESSIAN_USES_WORKSPACE_TEMP_ANY = " + str(1 if any(p >= 1 for p in self.d2ee_spill_tier_3way) else 0) + ";", \
                             "const int GRID_END_EFFECTOR_POSE_HESSIAN_SHARED_TIER_VALUE = " + str(self.d2ee_spill_tier) + ";", \
                             "const int GRID_END_EFFECTOR_POSE_GRADIENT_USES_WORKSPACE_TEMP = " + str(int(self.end_effector_pose_gradient_use_workspace_temp)) + ";", \
                             # Same per-tier gate for the EE_POSE_GRAD chain workspace.
                             "const int GRID_END_EFFECTOR_POSE_GRADIENT_USES_WORKSPACE_TEMP_ANY = " + str(1 if any(p >= 1 for p in self.end_effector_pose_gradient_spill_tier_3way) else 0) + ";", \
                             "const int GRID_END_EFFECTOR_POSE_GRADIENT_USES_WORKSPACE_DXHOM = " + str(int(self.end_effector_pose_gradient_use_workspace_dxhom)) + ";", \
                             "const int GRID_END_EFFECTOR_POSE_GRADIENT_SHARED_TIER_VALUE = " + str(self.end_effector_pose_gradient_spill_tier) + ";", \
                             # DE-GATE #2: 1 if the DEFAULT tier spills the dccrba output or Jw band / cmm Jw band
                             # into d_workspace (so init_gridData must allocate d_workspace in the kinematics path).
                             "const int GRID_DCCRBA_USES_WORKSPACE_TEMP = " + str(1 if (self.dccrba_spill_tier_3way[0] >= 1 or self.cmm_time_variation_spill_tier_3way[0] >= 1 or self.com_spill_tier_3way[0] >= 1 or self.ccrba_spill_tier_3way[0] >= 1 or self.energy_spill_tier_3way[0] >= 1) else 0) + ";", \
                             # osc_inertia (kinematics-path) spill-F rung needs d_workspace; the dynamics
                             # block does not allocate it for a kinematics-only codegen -> signal the lazy-alloc.
                             "const int GRID_OSC_INERTIA_USES_WORKSPACE = " + str(1 if self.osc_inertia_spill_tier_3way[0] >= 1 else 0) + ";", \
                             "const int GRID_INVERSE_DYNAMICS_GRADIENT_SHARED_TIER_VALUE = " + str(self.inverse_dynamics_gradient_spill_tier) + ";", \
                             "const int GRID_FORWARD_DYNAMICS_GRADIENT_SHARED_TIER_VALUE = " + str(self.forward_dynamics_gradient_spill_tier) + ";", \
                             "const int ID_DU_TEMP_SPILL_START = " + str(inverse_dynamics_gradient_temp_layout["spill_start"]) + ";", \
                             "const int ID_DU_TEMP_SPILL_END = " + str(inverse_dynamics_gradient_temp_layout["spill_end"]) + ";", \
                             "const int ID_DU_TEMP_SPILL_COUNT = " + str(inverse_dynamics_gradient_temp_layout["spill_count"]) + ";", \
                             "const int INVERSE_DYNAMICS_DYNAMIC_SHARED_MEM_COUNT = " + str(legacy_arena_count(id_t_count)) + ";", \
                             "const int MINV_DYNAMIC_SHARED_MEM_COUNT = " + str(legacy_arena_count(minv_t_count)) + ";", \
                             "const int FORWARD_DYNAMICS_DYNAMIC_SHARED_MEM_COUNT = " + str(legacy_arena_count(fd_t_count)) + ";", \
                             "const int INVERSE_DYNAMICS_GRADIENT_DYNAMIC_SHARED_MEM_COUNT = " + str(legacy_arena_count(inverse_dynamics_gradient_t_count)) + ";", \
                             "const int FORWARD_DYNAMICS_GRADIENT_DYNAMIC_SHARED_MEM_COUNT = " + str(legacy_arena_count(forward_dynamics_gradient_t_count)) + ";", \
                             "const int INTEGRATOR_DYNAMIC_SHARED_MEM_COUNT = " + str(legacy_arena_count(integrator_t_count)) + ";", \
                             "const int INTEGRATOR_DU_DYNAMIC_SHARED_MEM_COUNT = " + str(legacy_arena_count(max(integrator_gradient_t_count, integrator_gradient_with_x_kp1_t_count))) + ";", \
                             "const int ABA_DYNAMIC_SHARED_MEM_COUNT = " + str(legacy_arena_count(aba_t_count)) + ";", \
                             "const int CRBA_SHARED_MEM_COUNT = " + str(legacy_arena_count(crba_t_count)) + ";", \
                             "const int ID_DU_MAX_SHARED_MEM_COUNT = " + str(legacy_arena_count(inverse_dynamics_gradient_t_count_full)) + ";", \
                             "const int FD_DU_MAX_SHARED_MEM_COUNT = " + str(legacy_arena_count(forward_dynamics_gradient_t_count_full)) + ";", \
                             "const int END_EFFECTOR_POSE_DYNAMIC_SHARED_MEM_COUNT = " + str(legacy_arena_count(ee_t_count)) + ";", \
                             "const int END_EFFECTOR_POSE_GRADIENT_DYNAMIC_SHARED_MEM_COUNT = " + str(legacy_arena_count(dee_t_count)) + ";", \
                             "const int END_EFFECTOR_POSE_HESSIAN_DYNAMIC_SHARED_MEM_COUNT = " + str(legacy_arena_count(d2ee_t_count)) + ";", \
                             f"const int IDSVA_SO_DYNAMIC_SHARED_MEM_COUNT = {legacy_arena_count(idsva_so_body_frame_t_count)};", \
                             f"const int FDSVA_SO_DYNAMIC_SHARED_MEM_COUNT = {legacy_arena_count(fdsva_so_t_count)};", \
                             "const int MAX_PERF_LEVEL_THREADS = " + str(self.max_perf_level_threads) + ";", \
                             "",
                             "// Resource-tier API (v2.0): each emitted kernel/_device/_inner takes a",
                             "// `RESOURCE_TIER` template parameter that picks the (launch_bounds, smem,",
                             "// register-footprint) profile. TIER_SHARED is the default and is the",
                             "// current-best perf; TIER_LITE keeps the same launch_bounds but reduces",
                             "// smem footprint (some intermediates moved to workspace global mem);",
                             "// TIER_MINIMAL drops launch_bounds to 1024 for maximum block-size flexibility",
                             "// at the cost of register slack. Inline-CUDA power users with tight outer",
                             "// kernels pick LITE/MINIMAL to fit GRiD primitives in their resource budget.",
                             "constexpr int TIER_SHARED    = 0;",
                             "constexpr int TIER_LITE    = 1;",
                             "constexpr int TIER_MINIMAL = 2;",
                             "",
                             "// Compile-time override for the default RESOURCE_TIER baked into every",
                             "// emitted kernel template. Defaults to TIER_SHARED so existing callsites",
                             "// (kernel<T>, host wrappers that call kernel<T><<<...>>>) keep their",
                             "// current best-perf semantics. Bench harness sets this via",
                             "// -DGRID_DEFAULT_RESOURCE_TIER=TIER_LITE (or TIER_MINIMAL) to sweep",
                             "// per-tier perf without modifying host-wrapper template signatures.",
                             "#ifndef GRID_DEFAULT_RESOURCE_TIER",
                             "#define GRID_DEFAULT_RESOURCE_TIER TIER_SHARED",
                             "#endif",
                             "",
                             "// Per-tier launch_bounds upper-bound (= max threads per block nvcc must",
                             "// budget registers for). sm_120 has 65536 regs/block; nvcc enforces",
                             "// regs_per_thread * max_threads <= regs_per_block, so a larger max_threads",
                             "// directly caps regs_per_thread. PERF=SUGGESTED keeps current best perf;",
                             "// LITE=min(2*SUGGESTED, 768) gives ~85 regs/thread cap (mid-budget);",
                             "// MINIMAL=1024 gives ~64 regs/thread cap (maximum block-size flexibility).",
                             "template <int TIER> __host__ __device__ constexpr int tier_max_threads() {",
                             "    return (TIER == TIER_MINIMAL) ? 1024",
                             "         : (TIER == TIER_LITE)    ? ((MAX_PERF_LEVEL_THREADS * 2 < 768) ? MAX_PERF_LEVEL_THREADS * 2 : 768)",
                             "         :                          MAX_PERF_LEVEL_THREADS;",
                             "}"])
    # A1b: baked autotuned per-algo launch config (single source of truth).
    self.gen_add_launch_config_helpers()
    self.gen_add_code_lines([
                             "#define GRID_GENERATED_NUM_JOINTS " + str(n),
                             "#define GRID_GENERATED_NUM_EES " + str(self.robot.get_total_leaf_nodes()),
                             ""])
    self.gen_add_code_lines([
                             "template <typename T> __host__ __device__ inline size_t INVERSE_DYNAMICS_DYNAMIC_SHARED_MEM_BYTES() { return grid_shared_arena_bytes<T>(" + str(id_t_count) + ", TOPOLOGY_HELPERS_COUNT, GRID_LINALG_NVIDIA_MAX_HELPER_BYTES<T>()); }",
                             *_tier_bytes_lines("INVERSE_DYNAMICS_REGRESSOR_DYNAMIC_SHARED_MEM_BYTES", self.inverse_dynamics_regressor_t_count_per_tier),
                             # g1-spill: per-tier placement of s_Y -- true => smem, false => d_workspace.
                             _tier_ternary_line("INVERSE_DYNAMICS_REGRESSOR_Y_IN_SMEM", "bool", (("true" if self.inverse_dynamics_regressor_spill_tier_3way[0] == 0 else "false"), ("true" if self.inverse_dynamics_regressor_spill_tier_3way[1] == 0 else "false"), ("true" if self.inverse_dynamics_regressor_spill_tier_3way[2] == 0 else "false"))),
                             # PS5 energy regressors (each output 10*NUM_BODIES, fits every tier -> no spill).
                             "template <typename T> __host__ __device__ inline size_t KINETIC_ENERGY_REGRESSOR_DYNAMIC_SHARED_MEM_BYTES() { return grid_shared_arena_bytes<T>(" + str(self.kinetic_energy_regressor_t_count) + ", TOPOLOGY_HELPERS_COUNT, GRID_LINALG_NVIDIA_MAX_HELPER_BYTES<T>()); }",
                             # PS5 Coriolis matrix C(q,qd) (nv x nv; fits smem at FULL -> no spill).
                             *_tier_bytes_lines("CORIOLIS_MATRIX_DYNAMIC_SHARED_MEM_BYTES", self.coriolis_matrix_t_count_per_tier),
                             # g1-spill: tier-aware. At a spilled tier the s_Y regressor
                             # scratch moves to d_workspace, shrinking the smem arena. Default
                             # TIER = TIER_SHARED keeps every existing single-arg call site working.
                             *_tier_bytes_lines("FORWARD_DYNAMICS_PARAMETER_GRADIENT_DYNAMIC_SHARED_MEM_BYTES", self.forward_dynamics_parameter_gradient_t_count_per_tier),
                             # g1-spill: per-tier placement of s_Y -- true => smem, false => d_workspace.
                             _tier_ternary_line("FD_PARAMETER_GRADIENT_Y_IN_SMEM", "bool", (("true" if self.forward_dynamics_parameter_gradient_spill_tier_3way[0] == 0 else "false"), ("true" if self.forward_dynamics_parameter_gradient_spill_tier_3way[1] == 0 else "false"), ("true" if self.forward_dynamics_parameter_gradient_spill_tier_3way[2] == 0 else "false"))),
                             # g1-spill: tier-aware. At a spilled tier s_dqdd_dfext (2nd output) moves to d_workspace.
                             *_tier_bytes_lines("F_EXT_GRADIENT_DYNAMIC_SHARED_MEM_BYTES", self.f_ext_gradient_t_count_per_tier),
                             # g1/h2_plus-spill: per-tier placement. DQDD in smem only at rung 0 (full);
                             # DTAU in smem at rungs 0-1 (spilled only at the deep rung 2, which also
                             # routes minv-F to the GRAD workspace, F_IN_SMEM == DTAU_IN_SMEM).
                             _tier_ternary_line("F_EXT_GRADIENT_DQDD_IN_SMEM", "bool", (("true" if self.f_ext_gradient_spill_tier_3way[0] == 0 else "false"), ("true" if self.f_ext_gradient_spill_tier_3way[1] == 0 else "false"), ("true" if self.f_ext_gradient_spill_tier_3way[2] == 0 else "false"))),
                             _tier_ternary_line("F_EXT_GRADIENT_DTAU_IN_SMEM", "bool", (("true" if self.f_ext_gradient_spill_tier_3way[0] <= 1 else "false"), ("true" if self.f_ext_gradient_spill_tier_3way[1] <= 1 else "false"), ("true" if self.f_ext_gradient_spill_tier_3way[2] <= 1 else "false"))),
                             # mimic-spill: dq kernel arena is tier-aware; the mimic per-sub slab spills at rung 1.
                             *_tier_bytes_lines("F_EXT_GRADIENT_DQ_DYNAMIC_SHARED_MEM_BYTES", self.f_ext_gradient_dq_t_count_per_tier),
                             # mimic-spill: per-tier placement of the mimic per-sub slab -- true => smem, false => d_workspace.
                             "template <int TIER> __host__ __device__ constexpr bool F_EXT_GRADIENT_DQ_SLAB_IN_SMEM() { return (TIER == TIER_SHARED) ? " + ("true" if self.f_ext_gradient_dq_spill_tier_3way[0] == 0 else "false") + " : (TIER == TIER_LITE) ? " + ("true" if self.f_ext_gradient_dq_spill_tier_3way[1] == 0 else "false") + " : " + ("true" if self.f_ext_gradient_dq_spill_tier_3way[2] == 0 else "false") + "; }"] + [
                             *_tier_bytes_lines("MINV_DYNAMIC_SHARED_MEM_BYTES", self.minv_t_count_per_tier),
                             *_tier_bytes_lines("FORWARD_DYNAMICS_DYNAMIC_SHARED_MEM_BYTES", self.fd_t_count_per_tier),
                             *_tier_bytes_lines("INVERSE_DYNAMICS_GRADIENT_DYNAMIC_SHARED_MEM_BYTES", self.inverse_dynamics_gradient_t_count_per_tier),
                             *_tier_bytes_lines("FORWARD_DYNAMICS_GRADIENT_DYNAMIC_SHARED_MEM_BYTES", self.forward_dynamics_gradient_t_count_per_tier),
                             # Tier-aware: at LITE/MINIMAL the FD inner's Minv F-region (6*nv*nv)
                             # spills to d_workspace, so the smem arena shrinks. Default TIER keeps
                             # the existing single-arg call sites working.
                             *_tier_bytes_lines("INTEGRATOR_DYNAMIC_SHARED_MEM_BYTES", self.integrator_t_count_per_tier),
                             # Per-robot tier->placement map for the integrator VALUE path's Minv F-region:
                             # in smem at spill level 0, in d_workspace (grad section) at level 1.
                             _tier_ternary_line("INTEGRATOR_MINV_F_IN_SMEM", "bool", (("true" if self.integrator_spill_tier_3way[0] == 0 else "false"), ("true" if self.integrator_spill_tier_3way[1] == 0 else "false"), ("true" if self.integrator_spill_tier_3way[2] == 0 else "false"))),
                             # Tier-aware: at LITE/MINIMAL the s_D_qdd_stage buffer (max_stages*nv*3nv)
                             # spills to d_workspace, so the smem arena shrinks. Default TIER keeps the
                             # existing single-arg call sites working.
                             *_tier_bytes_lines("INTEGRATOR_DU_DYNAMIC_SHARED_MEM_BYTES", self.integrator_gradient_t_count_per_tier),
                             # Per-robot tier->placement map for the integrator gradient's s_D_qdd_stage
                             # buffer: in smem at spill level 0, in d_workspace (grad section) at level 1.
                             # Per-robot tier->placement maps for the integrator gradient's surgical
                             # spill ladder. Each buffer's IN_SMEM bool is keyed on RESOURCE_TIER;
                             # INNER_LEVEL gives the FD-grad inner spill (0 full smem, 1 da_df-band
                             # selective, 2 whole inner -> d_workspace).
                             _tier_ternary_line("INTEGRATOR_DU_D_QDD_IN_SMEM", "bool", (_b(self.integrator_gradient_dqdd_in_smem_per_tier[0]), _b(self.integrator_gradient_dqdd_in_smem_per_tier[1]), _b(self.integrator_gradient_dqdd_in_smem_per_tier[2]))),
                             _tier_ternary_line("INTEGRATOR_DU_DAB_IN_SMEM", "bool", (_b(self.integrator_gradient_dab_in_smem_per_tier[0]), _b(self.integrator_gradient_dab_in_smem_per_tier[1]), _b(self.integrator_gradient_dab_in_smem_per_tier[2]))),
                             _tier_ternary_line("INTEGRATOR_DU_INNER_LEVEL", "int", (str(self.integrator_gradient_inner_level_per_tier[0]), str(self.integrator_gradient_inner_level_per_tier[1]), str(self.integrator_gradient_inner_level_per_tier[2]))),
                             # d_workspace sub-offsets (within the per-timestep slot): Dqdd at 0, then dAB, then the inner-spill region.
                             "template <typename T> __host__ __device__ inline size_t GRID_INTEGRATOR_GRADIENT_DAB_OFFSET_BYTES() { return sizeof(T) * static_cast<size_t>(" + str(self._integrator_gradient_dqdd_count) + "); }",
                             "template <typename T> __host__ __device__ inline size_t GRID_INTEGRATOR_GRADIENT_INNER_OFFSET_BYTES() { return sizeof(T) * static_cast<size_t>(" + str(self._integrator_gradient_dqdd_count + self._integrator_gradient_dAB_count) + "); }",
                             "template <typename T> __host__ __device__ inline size_t INVERSE_DYNAMICS_DEVICE_DYNAMIC_SHARED_MEM_BYTES() { return grid_shared_arena_bytes<T>(" + str(id_device_t_count) + ", TOPOLOGY_HELPERS_COUNT, GRID_LINALG_NVIDIA_MAX_HELPER_BYTES<T>()); }",
                             "template <typename T> __host__ __device__ inline size_t MINV_DEVICE_DYNAMIC_SHARED_MEM_BYTES() { return grid_shared_arena_bytes<T>(" + str(minv_device_t_count) + ", TOPOLOGY_HELPERS_COUNT, GRID_LINALG_NVIDIA_MAX_HELPER_BYTES<T>()); }",
                             "template <typename T> __host__ __device__ inline size_t FORWARD_DYNAMICS_DEVICE_DYNAMIC_SHARED_MEM_BYTES() { return grid_shared_arena_bytes<T>(" + str(fd_device_t_count) + ", TOPOLOGY_HELPERS_COUNT, GRID_LINALG_NVIDIA_MAX_HELPER_BYTES<T>()); }",
                             # Per-tier sizes for forward_dynamics_device (inline-CUDA users only).
                             # At TIER_SHARED the FD inner s_temp lives in the smem arena; at
                             # TIER_LITE/MINIMAL the whole arena moves to d_workspace (this is the
                             # device-path analog of the FD kernel's MINV_F_IN_SMEM lever, which
                             # surgically spills only the F tail; the device path takes the
                             # whole-arena route to keep the inline call's smem footprint minimal).
                             "template <typename T, int TIER = GRID_DEFAULT_RESOURCE_TIER> __host__ __device__ constexpr size_t FORWARD_DYNAMICS_DEVICE_INLINE_SMEM_BYTES() {",
                             "    return (TIER == TIER_SHARED)",
                             "        ? grid_shared_arena_bytes<T>(" + str(fd_device_t_count) + ", TOPOLOGY_HELPERS_COUNT, GRID_LINALG_NVIDIA_MAX_HELPER_BYTES<T>())",
                             "        : grid_shared_arena_bytes<T>(" + str(fd_device_t_count - self.gen_forward_dynamics_inner_temp_mem_size(minv_f_in_smem=True)) + ", TOPOLOGY_HELPERS_COUNT, GRID_LINALG_NVIDIA_MAX_HELPER_BYTES<T>());",
                             "}",
                             "template <typename T, int TIER = GRID_DEFAULT_RESOURCE_TIER> __host__ __device__ constexpr size_t FORWARD_DYNAMICS_DEVICE_INLINE_WORKSPACE_BYTES() { return (TIER == TIER_SHARED) ? static_cast<size_t>(0) : sizeof(T) * static_cast<size_t>(" + str(self.gen_forward_dynamics_inner_temp_mem_size(minv_f_in_smem=True)) + "); }",
                             *_tier_bytes_lines("ABA_DYNAMIC_SHARED_MEM_BYTES", self.aba_t_count_per_tier),
                             *_tier_bytes_lines("CRBA_DYNAMIC_SHARED_MEM_BYTES", self.crba_t_count_per_tier),
                             "template <typename T> __host__ __device__ constexpr size_t GRID_EE_LINALG_SHARED_BYTES() { return static_cast<size_t>(0); }",
                             # PS5 potential-energy regressor (kinematics / XmatsHom domain; uses the ee linalg helper bytes).
                             "template <typename T> __host__ __device__ inline size_t POTENTIAL_ENERGY_REGRESSOR_DYNAMIC_SHARED_MEM_BYTES() { return grid_shared_arena_bytes<T>(" + str(self.potential_energy_regressor_t_count) + ", TOPOLOGY_HELPERS_COUNT, GRID_EE_LINALG_SHARED_BYTES<T>()); }",
                             "template <typename T> __host__ __device__ inline size_t END_EFFECTOR_POSE_DYNAMIC_SHARED_MEM_BYTES() { return grid_shared_arena_bytes<T>(" + str(ee_t_count) + ", TOPOLOGY_HELPERS_COUNT, GRID_EE_LINALG_SHARED_BYTES<T>()); }",
                             # Phase 3d: tier-aware. PERF/LITE/MINIMAL each report the smem
                             # bytes their picked spill level needs. Collapsed picks (small
                             # robots) return identical values across branches.
                             *_tier_bytes_lines("END_EFFECTOR_POSE_GRADIENT_DYNAMIC_SHARED_MEM_BYTES", self.end_effector_pose_gradient_t_count_per_tier, linalg_arg=", GRID_EE_LINALG_SHARED_BYTES<T>()"),
                             # Tier-aware: TIER_SHARED/LITE/MINIMAL each report the smem bytes their
                             # picked spill level needs. When the picks collapse (small robots) the
                             # three branches return identical values. Default TIER = TIER_SHARED
                             # preserves all existing single-arg call sites.
                             *_tier_bytes_lines("END_EFFECTOR_POSE_HESSIAN_DYNAMIC_SHARED_MEM_BYTES", self.d2ee_t_count_per_tier, linalg_arg=", GRID_EE_LINALG_SHARED_BYTES<T>()"),
                             # G2 centroidal quick-wins shared-mem macros (no tier spill).
                             "template <typename T> __host__ __device__ inline size_t INVERSE_DYNAMICS_BIAS_DYNAMIC_SHARED_MEM_BYTES() { return grid_shared_arena_bytes<T>(" + str(self.id_bias_t_count) + ", TOPOLOGY_HELPERS_COUNT, GRID_LINALG_NVIDIA_MAX_HELPER_BYTES<T>()); }",
                             # com/ccrba/energy: 2-rung J-spill ladder (DE-GATE #2). L0 keeps the Jw
                             # band in smem (== old single rung), L1 spills it -> d_workspace.
                             *_tier_bytes_lines("COM_DYNAMIC_SHARED_MEM_BYTES", self.com_t_count_per_tier, linalg_arg=", GRID_EE_LINALG_SHARED_BYTES<T>()"),
                             *_tier_bytes_lines("CCRBA_DYNAMIC_SHARED_MEM_BYTES", self.ccrba_t_count_per_tier, linalg_arg=", GRID_EE_LINALG_SHARED_BYTES<T>()"),
                             *_tier_bytes_lines("ENERGY_DYNAMIC_SHARED_MEM_BYTES", self.energy_t_count_per_tier, linalg_arg=", GRID_EE_LINALG_SHARED_BYTES<T>()"),
                             _tier_ternary_line("COM_J_IN_SMEM", "bool", (("true" if self.com_spill_tier_3way[0] == 0 else "false"), ("true" if self.com_spill_tier_3way[1] == 0 else "false"), ("true" if self.com_spill_tier_3way[2] == 0 else "false"))),
                             _tier_ternary_line("CCRBA_J_IN_SMEM", "bool", (("true" if self.ccrba_spill_tier_3way[0] == 0 else "false"), ("true" if self.ccrba_spill_tier_3way[1] == 0 else "false"), ("true" if self.ccrba_spill_tier_3way[2] == 0 else "false"))),
                             _tier_ternary_line("ENERGY_J_IN_SMEM", "bool", (("true" if self.energy_spill_tier_3way[0] == 0 else "false"), ("true" if self.energy_spill_tier_3way[1] == 0 else "false"), ("true" if self.energy_spill_tier_3way[2] == 0 else "false"))),
                             # PS5 dCCRBA (kinematics domain, uses the EE linalg scratch like ccrba):
                             # cmm_time_variation (Adot, 6*nv; no spill) + dccrba (6*nv*nv; per-tier
                             # surgical spill of the output to the d_workspace SO band).
                             *_tier_bytes_lines("CMM_TIME_VARIATION_DYNAMIC_SHARED_MEM_BYTES", self.cmm_time_variation_t_count_per_tier, linalg_arg=", GRID_EE_LINALG_SHARED_BYTES<T>()"),
                             # DE-GATE #2: per-tier placement of the cmm Jw sweep band -- true => smem, false => d_workspace.
                             _tier_ternary_line("CMM_J_IN_SMEM", "bool", (("true" if self.cmm_time_variation_spill_tier_3way[0] == 0 else "false"), ("true" if self.cmm_time_variation_spill_tier_3way[1] == 0 else "false"), ("true" if self.cmm_time_variation_spill_tier_3way[2] == 0 else "false"))),
                             *_tier_bytes_lines("DCCRBA_DYNAMIC_SHARED_MEM_BYTES", self.dccrba_t_count_per_tier, linalg_arg=", GRID_EE_LINALG_SHARED_BYTES<T>()"),
                             # per-tier placement of the s_dccrba output -- true => smem, false => d_workspace.
                             _tier_ternary_line("DCCRBA_OUTPUT_IN_SMEM", "bool", (("true" if self.dccrba_spill_tier_3way[0] == 0 else "false"), ("true" if self.dccrba_spill_tier_3way[1] == 0 else "false"), ("true" if self.dccrba_spill_tier_3way[2] == 0 else "false"))),
                             # DE-GATE #2: per-tier placement of the dccrba Jw sweep band -- in smem at L0/L1 (pick<=1), spilled at L2.
                             _tier_ternary_line("DCCRBA_J_IN_SMEM", "bool", (("true" if self.dccrba_spill_tier_3way[0] <= 1 else "false"), ("true" if self.dccrba_spill_tier_3way[1] <= 1 else "false"), ("true" if self.dccrba_spill_tier_3way[2] <= 1 else "false"))),
                             *_tier_bytes_lines("IDSVA_SO_BODY_FRAME_DYNAMIC_SHARED_MEM_BYTES", self.idsva_so_body_frame_t_count_per_tier, linalg_arg=""),
                             *_tier_bytes_lines("IDSVA_SO_WORLD_FRAME_DYNAMIC_SHARED_MEM_BYTES", self.idsva_so_world_frame_t_count_per_tier, linalg_arg=""),
                             *_tier_bytes_lines("FDSVA_SO_DYNAMIC_SHARED_MEM_BYTES", self.fdsva_so_t_count_per_tier, linalg_arg=""),
                             "// Per-tier scratch sizes for fdsva_so_contract (inline-CUDA users only — the host launchers always use TIER_SHARED).",
                             "// At TIER_SHARED the 4*NV^3 inner scratch lives in s_temp; at TIER_LITE/MINIMAL it moves to d_workspace, freeing shared memory for the caller's outer kernel.",
                             "// fdsva_so_contract scratch sizing, keyed on the INNER's placement choice",
                             "// (SCRATCH_IN_SMEM) rather than a tier — the inner decides placement, the",
                             "// caller sizes both arenas from these. FDSVA_SO_SCRATCH_IN_SMEM<TIER>()",
                             "// gives the placement codegen assigned to each tier for THIS robot.",
                             "template <typename T, bool SCRATCH_IN_SMEM = true> __host__ __device__ constexpr size_t FDSVA_SO_INNER_SMEM_BYTES() { return SCRATCH_IN_SMEM ? sizeof(T) * static_cast<size_t>(" + str(4*nv**3) + ") : static_cast<size_t>(0); }",
                             "template <typename T, bool SCRATCH_IN_SMEM = true> __host__ __device__ constexpr size_t FDSVA_SO_INNER_WORKSPACE_BYTES() { return SCRATCH_IN_SMEM ? static_cast<size_t>(0) : sizeof(T) * static_cast<size_t>(" + str(4*nv**3) + "); }",
                             # Per-robot tier->placement map for the fdsva_so_contract 4*NV^3 scratch:
                             # it stays in smem (CONTRACT_IN_SMEM=true) at every rung that does NOT
                             # set use_workspace_temp (the contract-spill flag). Derived from the per-rung
                             # use_workspace_temp flag (tuple field index 3) so the A4 idsva_cold rung
                             # (which keeps the contract in smem) is classified correctly without a
                             # hardcoded index that the rung insertion would have shifted.
                             "// Per-robot tier->placement map: contraction scratch stays in smem at any rung that doesn't set use_workspace_temp.",
                             _tier_ternary_line("FDSVA_SO_SCRATCH_IN_SMEM", "bool", (("true" if not _fdsva_so_tiers[self.fdsva_so_spill_tier_3way[0]][3] else "false"), ("true" if not _fdsva_so_tiers[self.fdsva_so_spill_tier_3way[1]][3] else "false"), ("true" if not _fdsva_so_tiers[self.fdsva_so_spill_tier_3way[2]][3] else "false"))),
                             "// Inner-controlled placement API (design rollout): each inline inner is keyed on a",
                             "// placement bool and decides arena pointers itself. *_INNER_{SMEM,WORKSPACE}_BYTES<T, IN_SMEM>",
                             "// give the two arena sizes; *_<...>_IN_SMEM<TIER>() give the per-robot tier->placement",
                             "// map codegen assigned (multiple tiers may share a placement on small robots).",
                             "// --- minv_inner (F-region) ---",
                             "template <typename T, bool F_IN_SMEM = true> __host__ __device__ constexpr size_t MINV_INNER_SMEM_BYTES() { return sizeof(T) * static_cast<size_t>(" + str(self.gen_minv_inner_no_F_size()) + (" + " + str(6*nv*nv) + " * (F_IN_SMEM ? 1 : 0)") + "); }",
                             "template <typename T, bool F_IN_SMEM = true> __host__ __device__ constexpr size_t MINV_INNER_WORKSPACE_BYTES() { return F_IN_SMEM ? static_cast<size_t>(0) : sizeof(T) * static_cast<size_t>(" + str(6*nv*nv) + "); }",
                             _tier_ternary_line("MINV_F_IN_SMEM", "bool", (("true" if self.minv_spill_tier_3way[0] == 0 else "false"), ("true" if self.minv_spill_tier_3way[1] == 0 else "false"), ("true" if self.minv_spill_tier_3way[2] == 0 else "false"))),
                             "// --- forward_dynamics_inner (internal Minv F-region) ---",
                             "template <typename T, bool MINV_F_IN_SMEM = true> __host__ __device__ constexpr size_t FD_INNER_SMEM_BYTES() { return MINV_F_IN_SMEM ? sizeof(T) * static_cast<size_t>(" + str(self.gen_forward_dynamics_inner_temp_mem_size(minv_f_in_smem=True)) + ") : sizeof(T) * static_cast<size_t>(" + str(self.gen_forward_dynamics_inner_temp_mem_size(minv_f_in_smem=False)) + "); }",
                             "template <typename T, bool MINV_F_IN_SMEM = true> __host__ __device__ constexpr size_t FD_INNER_WORKSPACE_BYTES() { return MINV_F_IN_SMEM ? static_cast<size_t>(0) : sizeof(T) * static_cast<size_t>(" + str(6*nv*nv) + "); }",
                             _tier_ternary_line("FD_MINV_F_IN_SMEM", "bool", (("true" if self.fd_spill_tier_3way[0] == 0 else "false"), ("true" if self.fd_spill_tier_3way[1] == 0 else "false"), ("true" if self.fd_spill_tier_3way[2] == 0 else "false"))),
                             # The integrator value path's only inner scratch is the FD inner itself,
                             # so its arena sizes mirror FD_INNER_* exactly: when MINV_F_IN_SMEM the
                             # 6*NV*NV F-region is in s_temp, else it spills to d_workspace. The
                             # per-robot tier->placement map is INTEGRATOR_MINV_F_IN_SMEM<TIER> (above).
                             "// --- integrator_inner (forwards the FD inner's Minv F-region lever) ---",
                             "template <typename T, bool MINV_F_IN_SMEM = true> __host__ __device__ constexpr size_t INTEGRATOR_INNER_SMEM_BYTES() { return MINV_F_IN_SMEM ? sizeof(T) * static_cast<size_t>(" + str(self.gen_integrator_inner_temp_mem_size(minv_f_in_smem=True)) + ") : sizeof(T) * static_cast<size_t>(" + str(self.gen_integrator_inner_temp_mem_size(minv_f_in_smem=False)) + "); }",
                             "template <typename T, bool MINV_F_IN_SMEM = true> __host__ __device__ constexpr size_t INTEGRATOR_INNER_WORKSPACE_BYTES() { return MINV_F_IN_SMEM ? static_cast<size_t>(0) : sizeof(T) * static_cast<size_t>(" + str(6*nv*nv) + "); }",
                             "// --- aba_inner (scratch band, surgical-spill ladder) ---",
                             "// Levels: 0=full (smem), 1=surgical (hot smem + cold d_cold), 2=workspace (whole band global).",
                             "// TEMP_IN_SMEM is false only at the level-2 (workspace) rung; COLD_IN_SMEM is false only at the level-1 (surgical) rung.",
                             "template <typename T, bool TEMP_IN_SMEM = true> __host__ __device__ constexpr size_t ABA_INNER_SMEM_BYTES() { return TEMP_IN_SMEM ? sizeof(T) * static_cast<size_t>(" + str(self.gen_aba_inner_temp_mem_size()) + ") : static_cast<size_t>(0); }",
                             "template <typename T, bool TEMP_IN_SMEM = true> __host__ __device__ constexpr size_t ABA_INNER_WORKSPACE_BYTES() { return TEMP_IN_SMEM ? static_cast<size_t>(0) : sizeof(T) * static_cast<size_t>(" + str(self.gen_aba_inner_temp_mem_size()) + "); }",
                             "template <typename T> __host__ __device__ constexpr size_t ABA_INNER_COLD_BYTES() { return sizeof(T) * static_cast<size_t>(" + str(self._aba_inner_cold_count) + "); }",
                             _tier_ternary_line("ABA_TEMP_IN_SMEM", "bool", (("false" if self.aba_spill_tier_3way[0] == 2 else "true"), ("false" if self.aba_spill_tier_3way[1] == 2 else "true"), ("false" if self.aba_spill_tier_3way[2] == 2 else "true"))),
                             _tier_ternary_line("ABA_COLD_IN_SMEM", "bool", (("false" if self.aba_spill_tier_3way[0] == 1 else "true"), ("false" if self.aba_spill_tier_3way[1] == 1 else "true"), ("false" if self.aba_spill_tier_3way[2] == 1 else "true"))),
                             "// --- crba_inner (scratch band) ---",
                             "template <typename T, bool TEMP_IN_SMEM = true> __host__ __device__ constexpr size_t CRBA_INNER_SMEM_BYTES() { return TEMP_IN_SMEM ? sizeof(T) * static_cast<size_t>(" + str(self.gen_crba_inner_temp_mem_size()) + ") : static_cast<size_t>(0); }",
                             "template <typename T, bool TEMP_IN_SMEM = true> __host__ __device__ constexpr size_t CRBA_INNER_WORKSPACE_BYTES() { return TEMP_IN_SMEM ? static_cast<size_t>(0) : sizeof(T) * static_cast<size_t>(" + str(self.gen_crba_inner_temp_mem_size()) + "); }",
                             # Inner band stays in smem at rung0 AND rung1 (surgical output-spill); spills only at rung2 (whole-band). Mirror DCCRBA_J_IN_SMEM's <=1.
                             _tier_ternary_line("CRBA_TEMP_IN_SMEM", "bool", (("true" if self.crba_spill_tier_3way[0] <= 1 else "false"), ("true" if self.crba_spill_tier_3way[1] <= 1 else "false"), ("true" if self.crba_spill_tier_3way[2] <= 1 else "false"))),
                             # s_M output in smem ONLY at rung0; spilled to the SO band at rung1/rung2. Mirror DCCRBA_OUTPUT_IN_SMEM's ==0.
                             _tier_ternary_line("CRBA_M_IN_SMEM", "bool", (("true" if self.crba_spill_tier_3way[0] == 0 else "false"), ("true" if self.crba_spill_tier_3way[1] == 0 else "false"), ("true" if self.crba_spill_tier_3way[2] == 0 else "false"))),
                             "// --- end_effector_pose_gradient_inner (chain workspace) ---",
                             "template <typename T, bool TEMP_IN_SMEM = true> __host__ __device__ constexpr size_t EE_GRAD_INNER_SMEM_BYTES() { return TEMP_IN_SMEM ? sizeof(T) * static_cast<size_t>(" + str(self.gen_end_effector_pose_gradient_inner_temp_mem_size()) + ") : static_cast<size_t>(0); }",
                             "template <typename T, bool TEMP_IN_SMEM = true> __host__ __device__ constexpr size_t EE_GRAD_INNER_WORKSPACE_BYTES() { return TEMP_IN_SMEM ? static_cast<size_t>(0) : sizeof(T) * static_cast<size_t>(" + str(self.gen_end_effector_pose_gradient_inner_temp_mem_size()) + "); }",
                             _tier_ternary_line("EE_GRAD_TEMP_IN_SMEM", "bool", (("true" if self.end_effector_pose_gradient_spill_tier_3way[0] == 0 else "false"), ("true" if self.end_effector_pose_gradient_spill_tier_3way[1] == 0 else "false"), ("true" if self.end_effector_pose_gradient_spill_tier_3way[2] == 0 else "false"))),
                             "// --- end_effector_pose_hessian_inner (large nv^2 end_effector_pose_hessian output) ---",
                             "// Per-tier placement of the d2ee inner's OUTPUT s_end_effector_pose_hessian: true => smem, false => d_workspace (which the kernel sets to d_end_effector_pose_hessian directly).",
                             _tier_ternary_line("D2EE_OUT_IN_SMEM", "bool", (("true" if self.d2ee_spill_tier_3way[0] == 0 else "false"), ("true" if self.d2ee_spill_tier_3way[1] == 0 else "false"), ("true" if self.d2ee_spill_tier_3way[2] == 0 else "false"))),
                             "// Per-tier sizes for forward_dynamics_gradient_device (inline-CUDA users only). At TIER_SHARED the temp scratch arena lives in s_temp; at TIER_LITE/MINIMAL it moves to d_workspace, freeing roughly " + str(forward_dynamics_gradient_temp_count) + "*sizeof(T) bytes of smem.",
                             "template <typename T, int TIER = GRID_DEFAULT_RESOURCE_TIER> __host__ __device__ constexpr size_t FORWARD_DYNAMICS_GRADIENT_DEVICE_INLINE_SMEM_BYTES() {",
                             "    return (TIER == TIER_SHARED)",
                             "        ? grid_shared_arena_bytes<T>(" + str(forward_dynamics_gradient_device_t_count) + ", TOPOLOGY_HELPERS_COUNT, GRID_LINALG_NVIDIA_MAX_HELPER_BYTES<T>())",
                             "        : grid_shared_arena_bytes<T>(" + str(forward_dynamics_gradient_device_t_count - forward_dynamics_gradient_temp_count) + ", TOPOLOGY_HELPERS_COUNT, GRID_LINALG_NVIDIA_MAX_HELPER_BYTES<T>());",
                             "}",
                             "template <typename T, int TIER = GRID_DEFAULT_RESOURCE_TIER> __host__ __device__ constexpr size_t FORWARD_DYNAMICS_GRADIENT_DEVICE_INLINE_WORKSPACE_BYTES() { return (TIER == TIER_SHARED) ? static_cast<size_t>(0) : sizeof(T) * static_cast<size_t>(" + str(forward_dynamics_gradient_temp_count) + "); }",
                             "// Per-tier sizes for end_effector_pose_hessian_device (inline-CUDA users only). At TIER_SHARED the smem arena keeps only the FD scratch + s_Xhom; at TIER_LITE/MINIMAL the device contract is unchanged (smem arena is the same -- the caller-provided s_end_effector_pose_hessian is what shifts), and the inner writes its " + str(d2ee_output_count) + "*sizeof(T) output bytes to d_workspace instead.",
                             "template <typename T, int TIER = GRID_DEFAULT_RESOURCE_TIER> __host__ __device__ constexpr size_t END_EFFECTOR_POSE_HESSIAN_DEVICE_INLINE_SMEM_BYTES() {",
                             "    return grid_shared_arena_bytes<T>(" + str(d2ee_inner_temp_count + XHom_size) + ", TOPOLOGY_HELPERS_COUNT, GRID_EE_LINALG_SHARED_BYTES<T>());",
                             "}",
                             "template <typename T, int TIER = GRID_DEFAULT_RESOURCE_TIER> __host__ __device__ constexpr size_t END_EFFECTOR_POSE_HESSIAN_DEVICE_INLINE_WORKSPACE_BYTES() { return (TIER == TIER_SHARED) ? static_cast<size_t>(0) : sizeof(T) * static_cast<size_t>(" + str(d2ee_output_count) + "); }",
                             "// Per-tier sizes for inverse_dynamics_gradient_device (inline-CUDA users only). At TIER_SHARED temp lives in s_temp; at TIER_LITE/MINIMAL it moves to d_workspace, freeing " + str(inverse_dynamics_gradient_temp_count) + "*sizeof(T) bytes of smem.",
                             "template <typename T, int TIER = GRID_DEFAULT_RESOURCE_TIER> __host__ __device__ constexpr size_t INVERSE_DYNAMICS_GRADIENT_DEVICE_INLINE_SMEM_BYTES() {",
                             "    return (TIER == TIER_SHARED)",
                             "        ? grid_shared_arena_bytes<T>(" + str(inverse_dynamics_gradient_device_t_count) + ", TOPOLOGY_HELPERS_COUNT, GRID_LINALG_NVIDIA_MAX_HELPER_BYTES<T>())",
                             "        : grid_shared_arena_bytes<T>(" + str(inverse_dynamics_gradient_device_t_count - inverse_dynamics_gradient_temp_count) + ", TOPOLOGY_HELPERS_COUNT, GRID_LINALG_NVIDIA_MAX_HELPER_BYTES<T>());",
                             "}",
                             "template <typename T, int TIER = GRID_DEFAULT_RESOURCE_TIER> __host__ __device__ constexpr size_t INVERSE_DYNAMICS_GRADIENT_DEVICE_INLINE_WORKSPACE_BYTES() { return (TIER == TIER_SHARED) ? static_cast<size_t>(0) : sizeof(T) * static_cast<size_t>(" + str(inverse_dynamics_gradient_temp_count) + "); }",
                             "// Per-tier sizes for idsva_so_device (inline-CUDA users only). At TIER_SHARED temp lives in s_temp; at TIER_LITE/MINIMAL it moves to d_workspace, freeing " + str(idsva_so_world_frame_inner_temp_count if self.robot.floating_base else idsva_so_body_frame_inner_temp_count) + "*sizeof(T) bytes of smem. Frame picked at codegen time: " + ("world_frame" if self.robot.floating_base else "body_frame") + ".",
                             "template <typename T, int TIER = GRID_DEFAULT_RESOURCE_TIER> __host__ __device__ constexpr size_t IDSVA_SO_DEVICE_INLINE_SMEM_BYTES() {",
                             "    return (TIER == TIER_SHARED)",
                             "        ? grid_shared_arena_bytes<T>(" + str((idsva_so_world_frame_inner_temp_count if self.robot.floating_base else idsva_so_body_frame_inner_temp_count) + XI_size) + ", TOPOLOGY_HELPERS_COUNT)",
                             "        : grid_shared_arena_bytes<T>(" + str(XI_size) + ", TOPOLOGY_HELPERS_COUNT);",
                             "}",
                             "template <typename T, int TIER = GRID_DEFAULT_RESOURCE_TIER> __host__ __device__ constexpr size_t IDSVA_SO_DEVICE_INLINE_WORKSPACE_BYTES() { return (TIER == TIER_SHARED) ? static_cast<size_t>(0) : sizeof(T) * static_cast<size_t>(" + str(idsva_so_world_frame_inner_temp_count if self.robot.floating_base else idsva_so_body_frame_inner_temp_count) + "); }",
                             "template <typename T> __host__ __device__ inline size_t GRID_GRAD_WORKSPACE_BYTES_PER_TIMESTEP() { return sizeof(T) * static_cast<size_t>(" + str(grad_spill_workspace_t_count) + "); }"] + (
                             # SO-REGION band gating v2 (2026-08-23, the "why 30 GB" fix): the
                             # SO section is a UNION region — its size is the max over the
                             # per-algo spill terms that overlay it. v1 gated the whole region
                             # on a user list and (a) MISSED users whose spill bands live here
                             # (f_ext_gradient's out band → OOB write, clean rc=188 crash) and
                             # (b) granted any single user the FULL 25.2 MB/ts. v2 emits the
                             # max PER TERM, each under its own GRID_ALLOC_ gate, so a gated
                             # exe is charged exactly the largest band ITS kernels overlay
                             # (h2_plus f_ext_gradient: 0.15 MB/ts, not 25.2). All terms
                             # overlay at the same base (offsets unchanged; the kernels never
                             # run concurrently). Ungated emission stays byte-identical; a
                             # gated header with NO -D flags evaluates every #if true and
                             # reproduces the ungated max exactly.
                             ["template <typename T> __host__ __device__ inline size_t GRID_SO_WORKSPACE_BYTES_PER_TIMESTEP() {",
                              "    size_t _m = 0, _t = 0; (void)_t;"]
                             + [line
                                for term_count, term_keys in (
                                    (8 * max(nv**3, 1), ("fdsva_so", "fdsva_so_mjx")),
                                    (d2ee_workspace_t_count, ("end_effector_pose_hessian",)),
                                    (end_effector_pose_gradient_workspace_t_count, ("end_effector_pose_gradient",)),
                                    (idsva_so_body_frame_grav_spill_t_count,
                                     ("idsva_so", "idsva_so_body_frame", "idsva_so_world_frame", "idsva_so_world_frame_mjx")),
                                    (idsva_so_spill_ws_t_count,
                                     ("idsva_so", "idsva_so_body_frame", "idsva_so_world_frame", "idsva_so_world_frame_mjx")),
                                    (_fpg_spill_ws, ("forward_dynamics_parameter_gradient",)),
                                    (_idr_spill_ws, ("inverse_dynamics_regressor",)),
                                    (_feg_spill_ws, ("f_ext_gradient",)),
                                    (_feg_dq_spill_ws, ("f_ext_gradient_dq",)),
                                    (_dccrba_spill_ws, ("dccrba",)),
                                    (_cmm_spill_ws, ("cmm_time_variation",)),
                                    (_centroidal_spill_ws, ("com", "ccrba", "energy")),
                                ) if term_count > 0
                                for line in (
                                    "#if " + _ag_expr(term_keys),
                                    "    _t = sizeof(T) * static_cast<size_t>(" + str(term_count) + "); if (_t > _m) { _m = _t; }",
                                    "#endif")]
                             + ["    return _m;",
                                "}"]
                             if _ws_gating else
                             ["template <typename T> __host__ __device__ inline size_t GRID_SO_WORKSPACE_BYTES_PER_TIMESTEP() { return sizeof(T) * static_cast<size_t>(" + str(so_workspace_t_count) + "); }"]) + [
                             # Phase 3e: sized for MINIMAL tier's spill (max across PERF/LITE/MINIMAL).
                             # Even if PERF doesn't spill df_du/Minv, MINIMAL might — the workspace
                             # allocation has to cover MINIMAL's needs at all times.
                             # A4: the idsva_cold rung insertion shifted spill_df_du/spill_Minv/pool_global
                             # from old indices 4/5/6 to 5/6/7, so the legacy `p >= 4` threshold (= "any rung
                             # at or past spill_df_du") is now `p >= 5` to reproduce the EXACT same per-robot
                             # value (Gate-A byte-identical for cardinals). The 3*NV^2 span backs
                             # s_df_du(2*NV^2) + s_Minv(NV^2) at GRID_FDSVA_SO_SPILL_OFFSET_BYTES.
                             ] + (lambda _fdsva_spill_line=(
                                 "template <typename T> __host__ __device__ inline size_t GRID_FDSVA_SO_SPILL_BYTES_PER_TIMESTEP() { return sizeof(T) * static_cast<size_t>("
                                 + str(3*nv*nv if any(p >= 5 for p in getattr(self, 'fdsva_so_spill_tier_3way', (0, 0, 0))) else 0) + "); }"):
                                 (["#if " + _ag_expr(("fdsva_so", "fdsva_so_mjx")),
                                   _fdsva_spill_line,
                                   "#else",
                                   "template <typename T> __host__ __device__ inline size_t GRID_FDSVA_SO_SPILL_BYTES_PER_TIMESTEP() { return static_cast<size_t>(0); }",
                                   "#endif"]
                                  if _ws_gating else [_fdsva_spill_line]))() + [
                             "template <typename T> __host__ __device__ inline size_t GRID_FDSVA_SO_SPILL_OFFSET_BYTES() { return GRID_GRAD_WORKSPACE_BYTES_PER_TIMESTEP<T>() + GRID_SO_WORKSPACE_BYTES_PER_TIMESTEP<T>(); }",
                             "template <typename T> __host__ __device__ inline size_t GRID_WORKSPACE_BYTES_PER_TIMESTEP() { return GRID_GRAD_WORKSPACE_BYTES_PER_TIMESTEP<T>() + GRID_SO_WORKSPACE_BYTES_PER_TIMESTEP<T>() + GRID_FDSVA_SO_SPILL_BYTES_PER_TIMESTEP<T>(); }",
                             "template <typename T> __host__ __device__ inline gridSharedTier GRID_INVERSE_DYNAMICS_GRADIENT_SHARED_TIER() { return static_cast<gridSharedTier>(GRID_INVERSE_DYNAMICS_GRADIENT_SHARED_TIER_VALUE); }",
                             "template <typename T> __host__ __device__ inline gridSharedTier GRID_FORWARD_DYNAMICS_GRADIENT_SHARED_TIER() { return static_cast<gridSharedTier>(GRID_FORWARD_DYNAMICS_GRADIENT_SHARED_TIER_VALUE); }",
                             "template <typename T> __host__ __device__ inline size_t GRID_SO_WORKSPACE_TEMP_OFFSET_BYTES() { return GRID_GRAD_WORKSPACE_BYTES_PER_TIMESTEP<T>(); }",
                             # DE-GATE #2: the dccrba Jw sweep band (and the cmm Jw band) spill to the SO
                             # band at a DISTINCT sub-offset so they never alias the dccrba output (which
                             # sits at GRID_SO_WORKSPACE_TEMP_OFFSET_BYTES, size 6*nv*nv*sizeof(T)). For cmm
                             # the output region is unused so the overlap is harmless.
                             "template <typename T> __host__ __device__ inline size_t GRID_DCCRBA_J_OFFSET_BYTES() { return GRID_SO_WORKSPACE_TEMP_OFFSET_BYTES<T>() + sizeof(T) * static_cast<size_t>(" + str(6 * nv * nv) + "); }",
                             # Phase 3a: Minv-F lives at offset 0 of the grad section when spilled.
                             # Safe to overlap with inverse_dynamics_gradient spill region because Minv finishes before
                             # inverse_dynamics_gradient starts in any kernel that composes both.
                             "template <typename T> __host__ __device__ inline size_t GRID_MINV_F_WORKSPACE_OFFSET_BYTES() { return static_cast<size_t>(0); }",
                             # ABA surgical cold sub-buffer reuses the SO/grad workspace band base (ABA
                             # never runs concurrently with SO/grad). The cold band (ABA_INNER_COLD_BYTES)
                             # is far smaller than GRID_GRAD_WORKSPACE_BYTES_PER_TIMESTEP, so it fits at
                             # offset 0 without growing GRID_WORKSPACE_BYTES_PER_TIMESTEP.
                             "template <typename T> __host__ __device__ inline size_t GRID_ABA_COLD_OFFSET_BYTES() { return static_cast<size_t>(0); }",
                             # D2EE no longer uses a per-timestep d_workspace slice (the spilled
                             # s_end_effector_pose_hessian is written directly into d_end_effector_pose_hessian); these offset macros are
                             # retained as 0 for backward compatibility with any inline-CUDA caller
                             # pattern that still references them. New code should not use them.
                             "template <typename T> __host__ __device__ inline size_t GRID_END_EFFECTOR_POSE_HESSIAN_WORKSPACE_TEMP_OFFSET_BYTES() { return static_cast<size_t>(0); }",
                             "template <typename T> __host__ __device__ inline size_t GRID_END_EFFECTOR_POSE_HESSIAN_WORKSPACE_D2XHOM_OFFSET_BYTES() { return static_cast<size_t>(0); }",
                             "template <typename T> __host__ __device__ inline size_t GRID_END_EFFECTOR_POSE_HESSIAN_WORKSPACE_D2EETEMP_OFFSET_BYTES() { return static_cast<size_t>(0); }",
                             # Phase 3d: EE_POSE_GRAD reuses the SO section (the kernels don't
                             # run concurrently — d_workspace bytes are safely repurposed). When
                             # the MINIMAL tier spills dXmatsHom, it sits before the temp arena.
                             "template <typename T> __host__ __device__ inline size_t GRID_END_EFFECTOR_POSE_GRADIENT_WORKSPACE_DXHOM_OFFSET_BYTES() { return GRID_SO_WORKSPACE_TEMP_OFFSET_BYTES<T>(); }",
                             "template <typename T> __host__ __device__ inline size_t GRID_END_EFFECTOR_POSE_GRADIENT_WORKSPACE_TEMP_OFFSET_BYTES() { return GRID_END_EFFECTOR_POSE_GRADIENT_WORKSPACE_DXHOM_OFFSET_BYTES<T>() + (GRID_END_EFFECTOR_POSE_GRADIENT_USES_WORKSPACE_DXHOM ? sizeof(T) * static_cast<size_t>(DXHOM_T_COUNT) : 0); }",
                             "template <typename T> __host__ __device__ inline bool grid_selected_shared_memory_fits() { return INVERSE_DYNAMICS_GRADIENT_DYNAMIC_SHARED_MEM_BYTES<T>() <= GRID_CUDA_TARGET_SHARED_MEM_BYTES && FORWARD_DYNAMICS_GRADIENT_DYNAMIC_SHARED_MEM_BYTES<T>() <= GRID_CUDA_TARGET_SHARED_MEM_BYTES && (!GRID_GENERATES_D2EE || END_EFFECTOR_POSE_HESSIAN_DYNAMIC_SHARED_MEM_BYTES<T>() <= GRID_CUDA_TARGET_SHARED_MEM_BYTES) && (!GRID_GENERATES_IDSVA_SO_BODY_FRAME || IDSVA_SO_BODY_FRAME_DYNAMIC_SHARED_MEM_BYTES<T>() <= GRID_CUDA_TARGET_SHARED_MEM_BYTES) && (!GRID_GENERATES_FDSVA_SO || FDSVA_SO_DYNAMIC_SHARED_MEM_BYTES<T>() <= GRID_CUDA_TARGET_SHARED_MEM_BYTES); }",
                             "// __forceinline__ used throughout the xhom helper chain so ptxas folds these into the",
                             "// inner kernels at all opt levels. For fixed-base the body of grid_q_index_affects_joint is",
                             "// the trivial `q_index == joint_id` check that pre-GLASS callsites used directly.",
                             "__host__ __device__ __forceinline__ bool grid_q_index_affects_joint(const int q_index, const int joint_id) {",
                             ("    if (joint_id == 0) { return q_index >= 0 && q_index < 7; } return q_index == joint_id + 6;" if self.robot.floating_base else "    return q_index == joint_id;"),
                             "}",
                             "__host__ __device__ __forceinline__ int grid_d2xhom_offset(const int q_index_i, [[maybe_unused]] const int q_index_j) {",
                             ("    return 16 * (q_index_i * NUM_JOINTS + q_index_j);" if self.robot.floating_base else "    return 16 * q_index_i;"),
                             "}",
                             "template <typename T>",
                             "__device__ __forceinline__ const T *grid_xhom_or_dxhom_ptr(const T *s_Xhom, const T *s_dXhom, const int q_index, const int joint_id) {",
                             "    return grid_q_index_affects_joint(q_index, joint_id) ? &s_dXhom[16 * q_index] : &s_Xhom[16 * joint_id];",
                             "}",
                             "template <typename T>",
                             "__device__ __forceinline__ const T *grid_xhom_or_dxhom_or_d2xhom_ptr(const T *s_Xhom, const T *s_dXhom, const T *s_d2Xhom, const int q_index_i, const int q_index_j, const int joint_id) {",
                             "    const bool i_affects = grid_q_index_affects_joint(q_index_i, joint_id);",
                             "    const bool j_affects = grid_q_index_affects_joint(q_index_j, joint_id);",
                             "    if (i_affects && j_affects) { return &s_d2Xhom[grid_d2xhom_offset(q_index_i, q_index_j)]; }",
                             "    if (i_affects) { return &s_dXhom[16 * q_index_i]; }",
                             "    if (j_affects) { return &s_dXhom[16 * q_index_j]; }",
                             "    return &s_Xhom[16 * joint_id];",
                             "}",
                             "template <typename T, bool USE_DA_DF_SPILL>",
                             "__device__ inline T *grid_id_du_temp_ptr(T *s_temp, T *d_temp_spill, int index) {",
                             "    if (!USE_DA_DF_SPILL) { return &s_temp[index]; }",
                             "    if (index >= ID_DU_TEMP_SPILL_START && index < ID_DU_TEMP_SPILL_END) {",
                             "        return &d_temp_spill[index - ID_DU_TEMP_SPILL_START];",
                             "    }",
                             "    if (index >= ID_DU_TEMP_SPILL_END) {",
                             "        return &s_temp[index - ID_DU_TEMP_SPILL_COUNT];",
                             "    }",
                             "    return &s_temp[index];",
                             "}",
                             ""])
    # then the structs
    # first add the struct
    self.gen_add_code_line("// Define custom structs")
    struct_lines = ["template <typename T>", \
                    "struct robotModel {", \
                    "    T *d_XImats;", \
                    "    int *d_topology_helpers;"]
    if getattr(self, "runtime_inertia", False):
        # D.4 / Phase 5: flag-gated mutable inertia table (10*NB or 10*(NB+1)
        # floats, body-indexed, in the frozen regressor basis). Emitted ONLY
        # under runtime_inertia so the baked struct stays byte-identical.
        struct_lines.append("    T *d_inertia_params;")
    if getattr(self, "runtime_transform", False):
        # runtime_transform: flag-gated mutable joint-origin table (6*NB
        # floats, joint-indexed, raw [x,y,z,r,p,y] basis). Emitted ONLY under
        # runtime_transform so the baked struct stays byte-identical.
        struct_lines.append("    T *d_transform_params;")
    if getattr(self, "runtime_joint_dynamics", False):
        # runtime_joint_dynamics: flag-gated mutable joint-dynamics table
        # (2*nv floats, v-slot indexed, [damping||friction], alpha-folded).
        # Emitted ONLY under runtime_joint_dynamics so the baked struct stays
        # byte-identical.
        struct_lines.append("    T *d_joint_dynamics_params;")
    struct_lines.append("};")
    self.gen_add_code_lines(struct_lines)
    self.gen_add_code_lines(["template <typename T, gridDataKind KIND = GRID_DATA_ALL>", \
                             "struct gridData {", \
                             "    // GPU INPUTS", \
                             "    T *d_q_qd_u;", \
                             "    T *d_q_qd;", \
                             "    T *d_q;", \
                             # external forces: body-major 6*NUM_BODIES local-frame, per timestep (zeroed by default)
                             "    T *d_f_ext;", \
                             "    // CPU INPUTS", \
                             "    T *h_q_qd_u;", \
                             "    T *h_q_qd;", \
                             "    T *h_q;", \
                             "    T *h_f_ext;", \
                             "    // GPU OUTPUTS", \
                             "    T *d_c;", \
                             "    T *d_Minv;", \
                             "    T *d_qdd;", \
                             "    T *d_M;", \
                             "    T *d_dc_du;", \
                             "    T *d_df_du;", \
                             # f_ext gradient column (section A): dtau/dfext = -J^T,
                             # dqdd/dfext = M^-1 J^T; each nv x (6*NB), body-major.
                             "    T *d_dtau_dfext;",
                             "    T *d_dqdd_dfext;",
                             "    T *d_f_ext_gradient_dq;  // -dJ^T/dq = d(inverse_dynamics_gradient)/dfext, nv*6NB*nv (both base modes)",
                             # R2: regressor + FD param-gradient outputs (each nv x 10*NUM_BODIES)
                             "    T *d_Y;          // inverse_dynamics_regressor (tau = Y . pi), nv*10NB",
                             "    T *d_dqdd_dpi;   // forward_dynamics_parameter_gradient (-Minv . Y), nv*10NB",
                             # PS5 energy regressors (each 10*NUM_BODIES; KE=y_KE.pi, PE=y_PE.pi)
                             "    T *d_ke_regressor;   // kinetic_energy_regressor (KE = y_KE . pi), 10NB",
                             "    T *d_pe_regressor;   // potential_energy_regressor (PE = y_PE . pi), 10NB",
                             # PS5 Coriolis matrix C(q,qd), row-major nv x nv (C qd+g = nonlinear_effects)
                             "    T *d_coriolis;       // coriolis_matrix C(q,qd), nv*nv",
                             # PS5 dCCRBA: dccrba tensor (6*nv*nv) + cmm_time_variation Adot (6*nv)
                             "    T *d_dccrba;             // dccrba dA_dq[:,k,m], 6*nv*nv",
                             "    T *d_cmm_time_variation; // cmm_time_variation Adot, 6*nv"]
                             + [
                             "    T *d_end_effector_pose;", \
                             "    T *d_end_effector_pose_gradient;", \
                             "    T *d_end_effector_pose_hessian;", \
                             # E2/S1: general-frame Jacobian outputs (frame_jacobian / frame_jacobian_dot:
                             # 6 x NUM_VEL each; osc_inertia: 6 x 6 task inertia Lambda)
                             "    T *d_frame_jacobian;       // frame_jacobian (6 x NUM_VEL)", \
                             "    T *d_frame_jacobian_dot;   // frame_jacobian_dot (6 x NUM_VEL)", \
                             "    T *d_osc_inertia;          // osc_inertia Lambda (6 x 6)", \
                             # runtime-target pose/pose-gradient (additive, opt-in). The 3-vector
                             # runtime tool/tip transform is a single device buffer, host-init to identity.
                             "    T *d_eePose;               // end_effector_pose_runtime (6 = [xyz;rpy])", \
                             "    T *d_eePoseGrad;           // end_effector_pose_gradient_runtime (6 x NUM_VEL)", \
                             "    T *d_eepose_runtime_offset; // runtime 4x4 col-major SE(3) tool/tip transform (target frame)"] \
                             # W1b.3 batched multi-target world positions / position-gradient (opt-in via
                             # multi_target_batch). Emitted ONLY for an MT robot -- a Python-side condition,
                             # not a #if, so a non-MT header stays BYTE-IDENTICAL (no inert preprocessor
                             # text). NUM_MULTI_TARGETS (the malloc size) is emitted before gen_init_gridData.
                             + ([
                             "    T *d_multi_target_position;          // multi_target_position (3 x NUM_MULTI_TARGETS)",
                             "    T *d_multi_target_position_gradient; // multi_target_position_gradient (3 x NUM_VEL x NUM_MULTI_TARGETS)",
                             ] if getattr(self, "_has_multi_target_position", False) else []) \
                             + [
                             "    unsigned char *d_workspace;", \
                             # workspace arena slot count, set by init_gridData (auto-fit from
                             # cudaMemGetInfo, GRID_WORKSPACE_TIMESTEP_SLOTS env override). Kernels
                             # index the arena per-BLOCK slot; host wrappers clamp their grid to it.
                             # 0 (calloc default, arena never allocated) => wrappers do not clamp.
                             "    int workspace_timestep_slots;", \
                             # idsva_so - d2tau_dq2, d2tau_dqd2, d2tau_dvdq, dM_dq
                             "    T *d_idsva_so;", \
                             # fdsva_so - d2a_dq2, d2a_dv2, d2a_dvdq, d2a_dtdq
                             "    T *d_df2;", \
                             # integrator outputs
                             "    T *d_x_kp1;", \
                             "    T *d_dAB;", \
                             # G2 centroidal quick-wins outputs
                             "    T *d_com;", \
                             "    T *d_ccrba;", \
                             "    T *d_energy;", \
                             "    // CPU OUTPUTS", \
                             "    T *h_c;", \
                             "    T *h_Minv;", \
                             "    T *h_qdd;", \
                             "    T *h_M;", \
                             "    T *h_dc_du;", \
                             "    T *h_df_du;", \
                             "    T *h_dtau_dfext;",
                             "    T *h_dqdd_dfext;",
                             "    T *h_f_ext_gradient_dq;  // -dJ^T/dq, nv*6NB*nv (both base modes)",
                             # R2: regressor + FD param-gradient outputs (each nv x 10*NUM_BODIES)
                             "    T *h_Y;",
                             "    T *h_dqdd_dpi;",
                             # PS5 energy regressors
                             "    T *h_ke_regressor;",
                             "    T *h_pe_regressor;",
                             # PS5 Coriolis matrix C(q,qd)
                             "    T *h_coriolis;",
                             # PS5 dCCRBA host buffers
                             "    T *h_dccrba;",
                             "    T *h_cmm_time_variation;"]
                             + [
                             "    T *h_end_effector_pose;", \
                             "    T *h_end_effector_pose_gradient;", \
                             "    T *h_end_effector_pose_hessian;", \
                             # E2/S1: general-frame Jacobian host buffers
                             "    T *h_frame_jacobian;", \
                             "    T *h_frame_jacobian_dot;", \
                             "    T *h_osc_inertia;", \
                             "    T *h_eePose;", \
                             "    T *h_eePoseGrad;"] \
                             # W1b.3 batched multi-target host buffers (opt-in; Python-conditional like d_ above)
                             + ([
                             "    T *h_multi_target_position;",
                             "    T *h_multi_target_position_gradient;",
                             ] if getattr(self, "_has_multi_target_position", False) else []) \
                             + [
                             # idsva_so - d2tau_dq2, d2tau_dqd2, d2tau_dvdq, dM_dq
                             "    T *h_idsva_so;", \
                             # fdsva_so - d2a_dq2, d2a_dv2, d2a_dvdq, d2a_dtdq
                             "    T *h_df2;", \
                             # integrator outputs
                             "    T *h_x_kp1;", \
                             "    T *h_dAB;", \
                             # G2 centroidal quick-wins outputs
                             "    T *h_com;", \
                             "    T *h_ccrba;", \
                             "    T *h_energy;", \
                             "};"])

def _derive_device_bytes_lines(code_lines):
    """Derive the body of gridData_device_bytes from THE SAME init_gridData
    line list, so the byte count and the carve can never drift: identical
    #if GRID_HAS_* / if (needs_*) structure, one grid_pool_align(size) add per
    cudaMalloc, and the workspace block mirrored on the ws_slots parameter
    (env/auto-fit lines dropped — the CALLER of the bytes fn decides slots;
    the pool-mode init branch consumes the same declared count). Any
    unclassified cudaMalloc raises at codegen time; the runtime referee is
    pool.used == gridData_device_bytes(slots) after a pool-mode init
    (asserted by the bindings' install path)."""
    import re as _re
    alloc = _re.compile(r"^(\s*)gpuErrchk\(cudaMalloc\(\(void\*\*\)&hd_data->\w+, (.+)\)\);$")
    ws_i = next(i for i, l in enumerate(code_lines) if "workspace arena LAST" in l)
    out = ["size_t _total = 0;"]
    for line in code_lines[:ws_i]:
        s = line.strip()
        m = alloc.match(line)
        if m:
            out.append(m.group(1) + "_total += grid_pool_align(" + m.group(2) + ");")
        elif (s.startswith("#if") or s.startswith("#endif")
              or s.startswith("const bool needs_") or s.startswith("if (needs_")
              or s == "}"):
            out.append(line)
        elif "cudaMalloc" in line:
            raise RuntimeError("unclassified cudaMalloc in init_gridData: " + line)
    ws_tail = code_lines[ws_i:]
    cond = next(l for l in ws_tail if l.strip().startswith("if (needs_dynamics ||"))
    gate_open = [l for l in ws_tail if l.strip().startswith("#if")]
    gate_close = [l for l in ws_tail if l.strip().startswith("#endif")]
    out += gate_open + [
        cond,
        "        const size_t _ws_per_ts = GRID_WORKSPACE_BYTES_PER_TIMESTEP<T>()*GRID_WORKSPACE_SLOTS;",
        "        int _ws_slots = ws_slots < 1 ? 1 : (ws_slots < NUM_TIMESTEPS ? ws_slots : NUM_TIMESTEPS);",
        "        _total += grid_pool_align(_ws_per_ts*(size_t)_ws_slots);",
        "    }"] + gate_close + ["return _total;"]
    return out


def gen_init_gridData(self):
    # 2a (h2_plus OOM): opt-in per-algo ALLOC GATING for the bench's solo exes.
    # When gen_all_code(emit_alloc_gating=True), every LARGE per-algo output
    # buffer's alloc gains an extra preprocessor guard
    #     (!defined(GRID_ALLOC_GATE) || GRID_ALLOC_<ALGO> || ...)
    # so a consumer compiled WITHOUT -DGRID_ALLOC_GATE still allocates
    # everything (behavior-identical to the ungated header), while a solo
    # bench exe passes -DGRID_ALLOC_GATE=1 plus -DGRID_ALLOC_<its algo>=1 and
    # allocates ONLY the buffers its algo's host wrappers actually touch
    # (h2_plus nv=81 x 1024 timesteps: d_f_ext_gradient_dq 12.2 GB + the two
    # SO tensors 8.7 GB each + the 26.6 MB/timestep workspace arena sum past
    # the card, killing EVERY solo exe at init_gridData). Guard key = the
    # per_algo_bench PER_ALGO_SPECS key, uppercased, prefixed GRID_ALLOC_.
    # Small shared buffers (inputs, d_c/d_qdd/d_Minv/d_M, kinematics /
    # centroidal vectors) stay unconditional: the warmup path runs
    # inverse_dynamics in every solo exe and cross-wrapper reads touch them.
    # Default emit_alloc_gating=False emits the header BYTE-IDENTICAL.
    gating = getattr(self, "emit_alloc_gating", False)

    def _ag_expr(keys):
        return " || ".join(["!defined(GRID_ALLOC_GATE)"]
                           + ["GRID_ALLOC_" + k.upper() for k in keys])

    def ag(*keys):
        # appended to an existing "#if GRID_HAS_X" alloc guard
        return (" && (" + _ag_expr(keys) + ")") if gating else ""

    def ag_open(*keys):
        # wraps a currently-unconditional alloc in its own #if block
        return ["    #if " + _ag_expr(keys)] if gating else []

    def ag_close():
        return ["    #endif"] if gating else []

    # Algos whose HOST WRAPPERS pass hd_data->d_workspace to a kernel (scan of
    # the generated wrappers, 2026-08-01). Everything NOT here (inverse_dynamics,
    # end_effector_pose, frame_jacobian(_dot), ke/pe regressors, the runtime-EE
    # pair, multi_target pair) never touches the arena, so its solo exe skips
    # the (potentially multi-GB) workspace alloc entirely.
    _ws_keys = ("minv", "forward_dynamics", "aba", "crba",
                "inverse_dynamics_gradient", "inverse_dynamics_gradient_mjx",
                "forward_dynamics_gradient", "forward_dynamics_gradient_mjx",
                "f_ext_gradient", "f_ext_gradient_dq",
                "inverse_dynamics_regressor", "forward_dynamics_parameter_gradient",
                "integrator", "integrator_gradient", "integrator_with_gradient",
                "generalized_gravity", "nonlinear_effects",
                "com", "ccrba", "energy", "cmm_time_variation", "dccrba",
                "coriolis_matrix", "osc_inertia",
                "end_effector_pose_gradient", "end_effector_pose_hessian",
                "idsva_so", "idsva_so_body_frame", "idsva_so_world_frame",
                "idsva_so_world_frame_mjx", "fdsva_so", "fdsva_so_mjx")
    code_lines = (["gridData<T, KIND> *hd_data = (gridData<T, KIND> *)calloc(1, sizeof(gridData<T, KIND>));",
                  "const bool needs_dynamics = KIND == GRID_DATA_ALL || KIND == GRID_DATA_DYNAMICS;",
                  "const bool needs_kinematics = KIND == GRID_DATA_ALL || KIND == GRID_DATA_KINEMATICS;",
                  "// input variables used by dynamics and/or kinematics",
                  "if (needs_dynamics || needs_kinematics) {", \
                  "    gpuErrchk(cudaMalloc((void**)&hd_data->d_q_qd_u, 3*NUM_JOINTS*NUM_TIMESTEPS*sizeof(T)));", \
                  "    gpuErrchk(cudaMalloc((void**)&hd_data->d_q, NUM_JOINTS*NUM_TIMESTEPS*sizeof(T)));", \
                  "    hd_data->h_q_qd_u = grid_host_alloc<T>(3*NUM_JOINTS*NUM_TIMESTEPS*sizeof(T));", \
                  "    hd_data->h_q = grid_host_alloc<T>(NUM_JOINTS*NUM_TIMESTEPS*sizeof(T));", \
                  "    // external forces (body-major 6*NUM_BODIES local-frame); zeroed so the", \
                  "    // default (no-fext) path subtracts nothing. Users overwrite h_f_ext and", \
                  "    // copy to d_f_ext to apply external forces.", \
                  "    gpuErrchk(cudaMalloc((void**)&hd_data->d_f_ext, 6*NUM_BODIES*NUM_TIMESTEPS*sizeof(T)));", \
                  "    gpuErrchk(cudaMemset(hd_data->d_f_ext, 0, 6*NUM_BODIES*NUM_TIMESTEPS*sizeof(T)));", \
                  "    hd_data->h_f_ext = (T *)calloc(6*NUM_BODIES*NUM_TIMESTEPS, sizeof(T));", \
                  "}", \
                  "if (needs_dynamics) {", \
                  "    gpuErrchk(cudaMalloc((void**)&hd_data->d_q_qd, 2*NUM_JOINTS*NUM_TIMESTEPS*sizeof(T)));", \
                  "    hd_data->h_q_qd = grid_host_alloc<T>(2*NUM_JOINTS*NUM_TIMESTEPS*sizeof(T));", \
                  "}", \
                  "// dynamics outputs and fallback workspace", \
                  "if (needs_dynamics) {", \
                  "    gpuErrchk(cudaMalloc((void**)&hd_data->d_c, NUM_JOINTS*NUM_TIMESTEPS*sizeof(T)));", \
                  "    gpuErrchk(cudaMalloc((void**)&hd_data->d_Minv, NUM_VEL*NUM_VEL*NUM_TIMESTEPS*sizeof(T)));", \
                  "    gpuErrchk(cudaMalloc((void**)&hd_data->d_qdd, NUM_JOINTS*NUM_TIMESTEPS*sizeof(T)));", \
                  "    gpuErrchk(cudaMalloc((void**)&hd_data->d_M, NUM_VEL*NUM_VEL*NUM_TIMESTEPS*sizeof(T)));", \
                  "    #if GRID_HAS_INVERSE_DYNAMICS_GRADIENT" + ag("inverse_dynamics_gradient", "inverse_dynamics_gradient_mjx"), \
                  "    gpuErrchk(cudaMalloc((void**)&hd_data->d_dc_du, 2*NUM_VEL*NUM_VEL*NUM_TIMESTEPS*sizeof(T)));", \
                  "    #endif", \
                  "    #if GRID_HAS_FORWARD_DYNAMICS_GRADIENT" + ag("forward_dynamics_gradient", "forward_dynamics_gradient_mjx"), \
                  "    gpuErrchk(cudaMalloc((void**)&hd_data->d_df_du, 2*NUM_VEL*NUM_VEL*NUM_TIMESTEPS*sizeof(T)));", \
                  "    #endif", \
                  "    // f_ext gradient column (section A): dtau/dfext, dqdd/dfext are each nv x (6*NB)", \
                  "    #if GRID_HAS_F_EXT_GRADIENT" + ag("f_ext_gradient"), \
                  "    gpuErrchk(cudaMalloc((void**)&hd_data->d_dtau_dfext, NUM_VEL*6*NUM_BODIES*NUM_TIMESTEPS*sizeof(T)));", \
                  "    gpuErrchk(cudaMalloc((void**)&hd_data->d_dqdd_dfext, NUM_VEL*6*NUM_BODIES*NUM_TIMESTEPS*sizeof(T)));", \
                  "    #endif"]
                  + ag_open("f_ext_gradient") + [
                  "    hd_data->h_dtau_dfext = grid_host_alloc<T>(NUM_VEL*6*NUM_BODIES*NUM_TIMESTEPS*sizeof(T));",
                  "    hd_data->h_dqdd_dfext = grid_host_alloc<T>(NUM_VEL*6*NUM_BODIES*NUM_TIMESTEPS*sizeof(T));"]
                  + ag_close() + [
                  "    // f_ext A.3: -dJ^T/dq = d(inverse_dynamics_gradient)/dfext, nv*6NB*nv (fixed base only; the largest per-timestep buffer)",
                  "    // sizeof(T) leads so the byte count is size_t throughout: the element count",
                  "    // alone overflows int on big robots (h2_plus nv=81 @N=1024: 3.06e9 > INT_MAX)",
                  "    #if GRID_HAS_F_EXT_GRADIENT_DQ" + ag("f_ext_gradient_dq"),
                  "    gpuErrchk(cudaMalloc((void**)&hd_data->d_f_ext_gradient_dq, sizeof(T)*NUM_VEL*6*NUM_BODIES*NUM_VEL*NUM_TIMESTEPS));",
                  "    hd_data->h_f_ext_gradient_dq = grid_host_alloc<T>(sizeof(T)*NUM_VEL*6*NUM_BODIES*NUM_VEL*NUM_TIMESTEPS);",
                  "    #endif",
                  "    // R2: regressor Y and FD param-gradient dqdd/dpi (each nv x 10*NUM_BODIES)",
                  "    #if GRID_HAS_INVERSE_DYNAMICS_REGRESSOR" + ag("inverse_dynamics_regressor"),
                  "    gpuErrchk(cudaMalloc((void**)&hd_data->d_Y, NUM_VEL*10*NUM_BODIES*NUM_TIMESTEPS*sizeof(T)));",
                  "    hd_data->h_Y = grid_host_alloc<T>(NUM_VEL*10*NUM_BODIES*NUM_TIMESTEPS*sizeof(T));",
                  "    #endif",
                  "    #if GRID_HAS_FORWARD_DYNAMICS_PARAMETER_GRADIENT" + ag("forward_dynamics_parameter_gradient"),
                  "    gpuErrchk(cudaMalloc((void**)&hd_data->d_dqdd_dpi, NUM_VEL*10*NUM_BODIES*NUM_TIMESTEPS*sizeof(T)));",
                  "    hd_data->h_dqdd_dpi = grid_host_alloc<T>(NUM_VEL*10*NUM_BODIES*NUM_TIMESTEPS*sizeof(T));",
                  "    #endif"]
                  + [
                  # d_idsva_so is ALSO a kernel input of fdsva_so (the fdsva_so kernel takes
                  # hd_data->d_idsva_so), so the fdsva keys must keep it allocated too.
                  # sizeof(T) leads: SECOND_ORDER_TENSOR_SIZE*NUM_TIMESTEPS alone overflows
                  # int on big robots (h2_plus 2,125,764*1024 = 2.18e9 > INT_MAX).
                  "    #if GRID_HAS_IDSVA_SO" + ag("idsva_so", "idsva_so_body_frame", "idsva_so_world_frame", "idsva_so_world_frame_mjx", "fdsva_so", "fdsva_so_mjx"), \
                  "    gpuErrchk(cudaMalloc((void**)&hd_data->d_idsva_so, sizeof(T)*SECOND_ORDER_TENSOR_SIZE*NUM_TIMESTEPS));", \
                  "    #endif", \
                  "    #if GRID_HAS_FDSVA_SO" + ag("fdsva_so", "fdsva_so_mjx"), \
                  "    gpuErrchk(cudaMalloc((void**)&hd_data->d_df2, sizeof(T)*SECOND_ORDER_TENSOR_SIZE*NUM_TIMESTEPS));", \
                  "    #endif"]
                  + [
                  "    hd_data->h_c = grid_host_alloc<T>(NUM_JOINTS*NUM_TIMESTEPS*sizeof(T));", \
                  "    hd_data->h_Minv = grid_host_alloc<T>(NUM_VEL*NUM_VEL*NUM_TIMESTEPS*sizeof(T));", \
                  "    hd_data->h_M = grid_host_alloc<T>(NUM_VEL*NUM_VEL*NUM_TIMESTEPS*sizeof(T));", \
                  "    hd_data->h_qdd = grid_host_alloc<T>(NUM_JOINTS*NUM_TIMESTEPS*sizeof(T));", \
                  "    #if GRID_HAS_INVERSE_DYNAMICS_GRADIENT" + ag("inverse_dynamics_gradient", "inverse_dynamics_gradient_mjx"), \
                  "    hd_data->h_dc_du = grid_host_alloc<T>(2*NUM_VEL*NUM_VEL*NUM_TIMESTEPS*sizeof(T));", \
                  "    #endif", \
                  "    #if GRID_HAS_FORWARD_DYNAMICS_GRADIENT" + ag("forward_dynamics_gradient", "forward_dynamics_gradient_mjx"), \
                  "    hd_data->h_df_du = grid_host_alloc<T>(2*NUM_VEL*NUM_VEL*NUM_TIMESTEPS*sizeof(T));", \
                  "    #endif", \
                  "    #if GRID_HAS_IDSVA_SO" + ag("idsva_so", "idsva_so_body_frame", "idsva_so_world_frame", "idsva_so_world_frame_mjx"), \
                  "    hd_data->h_idsva_so = grid_host_alloc<T>(sizeof(T)*SECOND_ORDER_TENSOR_SIZE*NUM_TIMESTEPS);", \
                  "    #endif", \
                  "    #if GRID_HAS_FDSVA_SO" + ag("fdsva_so", "fdsva_so_mjx"), \
                  "    hd_data->h_df2 = grid_host_alloc<T>(sizeof(T)*SECOND_ORDER_TENSOR_SIZE*NUM_TIMESTEPS);", \
                  "    #endif", \
                  "    #if GRID_HAS_INTEGRATOR", \
                  "    gpuErrchk(cudaMalloc((void**)&hd_data->d_x_kp1, 2*NUM_JOINTS*NUM_TIMESTEPS*sizeof(T)));", \
                  "    hd_data->h_x_kp1 = grid_host_alloc<T>(2*NUM_JOINTS*NUM_TIMESTEPS*sizeof(T));", \
                  "    #endif", \
                  "    #if GRID_HAS_INTEGRATOR_GRADIENT" + ag("integrator_gradient", "integrator_with_gradient"), \
                  "    gpuErrchk(cudaMalloc((void**)&hd_data->d_dAB, 2*NUM_JOINTS*3*NUM_JOINTS*NUM_TIMESTEPS*sizeof(T)));", \
                  "    hd_data->h_dAB = grid_host_alloc<T>(2*NUM_JOINTS*3*NUM_JOINTS*NUM_TIMESTEPS*sizeof(T));", \
                  "    #endif", \
                  "}", \
                  "// kinematics outputs", \
                  "if (needs_kinematics) {", \
                  "    gpuErrchk(cudaMalloc((void**)&hd_data->d_end_effector_pose, 6*NUM_EES*NUM_TIMESTEPS*sizeof(T)));", \
                  "    gpuErrchk(cudaMalloc((void**)&hd_data->d_end_effector_pose_gradient, 6*NUM_EES*NUM_VEL*NUM_TIMESTEPS*sizeof(T)));"]
                  + ag_open("end_effector_pose_hessian") + [
                  "    gpuErrchk(cudaMalloc((void**)&hd_data->d_end_effector_pose_hessian, 6*NUM_EES*NUM_VEL*NUM_VEL*NUM_TIMESTEPS*sizeof(T)));"]
                  + ag_close() + [
                  "    hd_data->h_end_effector_pose = grid_host_alloc<T>(6*NUM_EES*NUM_TIMESTEPS*sizeof(T));", \
                  "    hd_data->h_end_effector_pose_gradient = grid_host_alloc<T>(6*NUM_EES*NUM_VEL*NUM_TIMESTEPS*sizeof(T));"]
                  + ag_open("end_effector_pose_hessian") + [
                  "    hd_data->h_end_effector_pose_hessian = grid_host_alloc<T>(6*NUM_EES*NUM_VEL*NUM_VEL*NUM_TIMESTEPS*sizeof(T));"]
                  + ag_close() + [
                  # E2/S1: general-frame Jacobian outputs (J / Jdot: 6*NV each ; Lambda: 36)
                  "    gpuErrchk(cudaMalloc((void**)&hd_data->d_frame_jacobian, 6*NUM_VEL*NUM_TIMESTEPS*sizeof(T)));", \
                  "    gpuErrchk(cudaMalloc((void**)&hd_data->d_frame_jacobian_dot, 6*NUM_VEL*NUM_TIMESTEPS*sizeof(T)));", \
                  "    gpuErrchk(cudaMalloc((void**)&hd_data->d_osc_inertia, 36*NUM_TIMESTEPS*sizeof(T)));", \
                  "    hd_data->h_frame_jacobian = grid_host_alloc<T>(6*NUM_VEL*NUM_TIMESTEPS*sizeof(T));", \
                  "    hd_data->h_frame_jacobian_dot = grid_host_alloc<T>(6*NUM_VEL*NUM_TIMESTEPS*sizeof(T));", \
                  "    hd_data->h_osc_inertia = grid_host_alloc<T>(36*NUM_TIMESTEPS*sizeof(T));", \
                  # runtime-target pose / pose-gradient (additive, opt-in). The runtime
                  # 3-vector offset is a single device buffer init to {0,0,0} (frame origin);
                  # a binding overwrites it before the call to request a nonzero tool transform.
                  "    gpuErrchk(cudaMalloc((void**)&hd_data->d_eePose, 6*NUM_TIMESTEPS*sizeof(T)));", \
                  "    gpuErrchk(cudaMalloc((void**)&hd_data->d_eePoseGrad, 6*NUM_VEL*NUM_TIMESTEPS*sizeof(T)));", \
                  "    gpuErrchk(cudaMalloc((void**)&hd_data->d_eepose_runtime_offset, 16*sizeof(T)));", \
                  "    { T h_Xtool_identity[16] = {1,0,0,0, 0,1,0,0, 0,0,1,0, 0,0,0,1};", \
                  "      gpuErrchk(cudaMemcpy(hd_data->d_eepose_runtime_offset, h_Xtool_identity, 16*sizeof(T), cudaMemcpyHostToDevice)); }", \
                  "    hd_data->h_eePose = grid_host_alloc<T>(6*NUM_TIMESTEPS*sizeof(T));", \
                  "    hd_data->h_eePoseGrad = grid_host_alloc<T>(6*NUM_VEL*NUM_TIMESTEPS*sizeof(T));"] \
                  # W1b.3 batched multi-target. Emitted ONLY for an MT robot (Python-conditional,
                  # not a #if) so a non-MT header is byte-identical AND never references
                  # NUM_MULTI_TARGETS, which only exists when the batch is emitted.
                  #
                  # The DEVICE buffers carry a 1024-element FLOOR. The single_timing kernels'
                  # anti-LICM feedback indexes d_<out>[(rep + 0x3FF) & 0x3FF] -> up to slot 1023,
                  # so gen_anti_licm_{input_reload,output_write} REQUIRE >= 1024 output slots.
                  # Every other algo meets that naturally (ee_pose: 6*NUM_EES*256 = 1536), but
                  # multi_target scales with the BATCH: a small batch under-allocates (a 1-target
                  # batch is only 3*1*256 = 768 < 1024) and the timing kernel reads OOB (caught by
                  # compute-sanitizer memcheck). The floor costs a few KB and makes it unbreakable;
                  # the D2H copy still moves only the natural 3*NUM_MULTI_TARGETS*n elements.
                  + ([
                  "    const int MT_POS_SLOTS  = 3*NUM_MULTI_TARGETS*NUM_TIMESTEPS;",
                  "    const int MT_GRAD_SLOTS = 3*NUM_VEL*NUM_MULTI_TARGETS*NUM_TIMESTEPS;",
                  "    gpuErrchk(cudaMalloc((void**)&hd_data->d_multi_target_position, (MT_POS_SLOTS > 1024 ? MT_POS_SLOTS : 1024)*sizeof(T)));",
                  "    gpuErrchk(cudaMalloc((void**)&hd_data->d_multi_target_position_gradient, (MT_GRAD_SLOTS > 1024 ? MT_GRAD_SLOTS : 1024)*sizeof(T)));",
                  "    hd_data->h_multi_target_position = grid_host_alloc<T>(MT_POS_SLOTS*sizeof(T));",
                  "    hd_data->h_multi_target_position_gradient = grid_host_alloc<T>(MT_GRAD_SLOTS*sizeof(T));",
                  ] if getattr(self, "_has_multi_target_position", False) else []) \
                  + [
                  "}", \
                  "// G2 centroidal quick-wins outputs (com: 3+3*NV ; ccrba: 6*NV+6 ; energy: 3)", \
                  "if (needs_dynamics || needs_kinematics) {", \
                  "    gpuErrchk(cudaMalloc((void**)&hd_data->d_com, (3+3*NUM_VEL)*NUM_TIMESTEPS*sizeof(T)));", \
                  "    gpuErrchk(cudaMalloc((void**)&hd_data->d_ccrba, (6*NUM_VEL+6)*NUM_TIMESTEPS*sizeof(T)));", \
                  "    gpuErrchk(cudaMalloc((void**)&hd_data->d_energy, 3*NUM_TIMESTEPS*sizeof(T)));", \
                  "    hd_data->h_com = grid_host_alloc<T>((3+3*NUM_VEL)*NUM_TIMESTEPS*sizeof(T));", \
                  "    hd_data->h_ccrba = grid_host_alloc<T>((6*NUM_VEL+6)*NUM_TIMESTEPS*sizeof(T));", \
                  "    hd_data->h_energy = grid_host_alloc<T>(3*NUM_TIMESTEPS*sizeof(T));", \
                  "    // PS5 energy regressors (each 10*NUM_BODIES): KE (dynamics) + PE (kinematics)", \
                  "    gpuErrchk(cudaMalloc((void**)&hd_data->d_ke_regressor, 10*NUM_BODIES*NUM_TIMESTEPS*sizeof(T)));", \
                  "    gpuErrchk(cudaMalloc((void**)&hd_data->d_pe_regressor, 10*NUM_BODIES*NUM_TIMESTEPS*sizeof(T)));", \
                  "    hd_data->h_ke_regressor = grid_host_alloc<T>(10*NUM_BODIES*NUM_TIMESTEPS*sizeof(T));", \
                  "    hd_data->h_pe_regressor = grid_host_alloc<T>(10*NUM_BODIES*NUM_TIMESTEPS*sizeof(T));", \
                  "    // PS5 Coriolis matrix C(q,qd) (nv x nv)", \
                  "    gpuErrchk(cudaMalloc((void**)&hd_data->d_coriolis, NUM_VEL*NUM_VEL*NUM_TIMESTEPS*sizeof(T)));", \
                  "    hd_data->h_coriolis = grid_host_alloc<T>(NUM_VEL*NUM_VEL*NUM_TIMESTEPS*sizeof(T));", \
                  "    // PS5 dCCRBA: dccrba tensor (6*nv*nv) + cmm_time_variation Adot (6*nv)"]
                  + ag_open("dccrba") + [
                  "    gpuErrchk(cudaMalloc((void**)&hd_data->d_dccrba, 6*NUM_VEL*NUM_VEL*NUM_TIMESTEPS*sizeof(T)));"]
                  + ag_close() + [
                  "    gpuErrchk(cudaMalloc((void**)&hd_data->d_cmm_time_variation, 6*NUM_VEL*NUM_TIMESTEPS*sizeof(T)));"]
                  + ag_open("dccrba") + [
                  "    hd_data->h_dccrba = grid_host_alloc<T>(6*NUM_VEL*NUM_VEL*NUM_TIMESTEPS*sizeof(T));"]
                  + ag_close() + [
                  "    hd_data->h_cmm_time_variation = grid_host_alloc<T>(6*NUM_VEL*NUM_TIMESTEPS*sizeof(T));", \
                  "}", \
                  # Workspace arena LAST (after every other device alloc, so cudaMemGetInfo sees
                  # true remaining memory): auto-fit the slot count. The per-timestep workspace is
                  # the large-batch RAM hog on big robots (h2_plus nv=81 @N=1024: 27.3 GB); when
                  # the full-N arena does not fit, allocate fewer slots — kernels index the arena
                  # per-BLOCK (grid_workspace_slot) and host wrappers clamp their launch grid to
                  # the slot count, so any slot count >= 1 is correct (just fewer concurrent
                  # blocks). Outputs stay full-N. GRID_WORKSPACE_TIMESTEP_SLOTS (env) forces a
                  # slot count (tests + bench A/B comparability).
                  "// workspace arena LAST: auto-fit slots to remaining device memory (see struct field)."]
                  + ag_open(*_ws_keys) + [
                  "    if (needs_dynamics || (needs_kinematics && (GRID_END_EFFECTOR_POSE_HESSIAN_USES_WORKSPACE_TEMP || GRID_END_EFFECTOR_POSE_GRADIENT_USES_WORKSPACE_TEMP || GRID_DCCRBA_USES_WORKSPACE_TEMP || GRID_OSC_INERTIA_USES_WORKSPACE))) {", \
                  "        const size_t _ws_per_ts = GRID_WORKSPACE_BYTES_PER_TIMESTEP<T>()*GRID_WORKSPACE_SLOTS;", \
                  "        int _ws_slots = NUM_TIMESTEPS;", \
                  "        const char *_ws_env = getenv(\"GRID_WORKSPACE_TIMESTEP_SLOTS\");", \
                  "        if (_ws_env != nullptr && atoi(_ws_env) > 0) { _ws_slots = atoi(_ws_env) < NUM_TIMESTEPS ? atoi(_ws_env) : NUM_TIMESTEPS; }", \
                  "        else if (_ws_per_ts > 0) {", \
                  "            size_t _ws_free = 0, _ws_total = 0;", \
                  "            gpuErrchk(cudaMemGetInfo(&_ws_free, &_ws_total));", \
                  ] + (
                  # GRID_WORKSPACE_RESERVE_MB (gated emission only): an ABSOLUTE
                  # device-memory reserve on top of the 10% — near-capacity
                  # allocation is what tickles the open-kmod nvidia_uvm chunk
                  # crashes (guide §7.x), so bench runs can keep e.g. 2-4 GB free.
                  # Default/unset = 0 -> identical arithmetic; ungated emission
                  # keeps the original line byte-identically.
                  ["            size_t _ws_reserve = 0; { const char *_r = getenv(\"GRID_WORKSPACE_RESERVE_MB\"); if (_r != nullptr && atoll(_r) > 0) { _ws_reserve = (size_t)atoll(_r) * 1048576ULL; } }",
                   "            const size_t _ws_budget = (_ws_free > _ws_free/10 + _ws_reserve) ? (_ws_free - _ws_free/10 - _ws_reserve) : 0;  // 10% + absolute reserve headroom"]
                  if gating else
                  ["            const size_t _ws_budget = _ws_free - _ws_free/10;  // 10% headroom"]) + [ \
                  "            if (_ws_per_ts*(size_t)NUM_TIMESTEPS > _ws_budget) {", \
                  "                _ws_slots = (int)(_ws_budget/_ws_per_ts);", \
                  "                if (_ws_slots < 1) { _ws_slots = 1; }  // one slot must fit; else the malloc below fails loudly", \
                  "            }", \
                  "        }", \
                  "        hd_data->workspace_timestep_slots = _ws_slots;", \
                  "        gpuErrchk(cudaMalloc((void**)&hd_data->d_workspace, _ws_per_ts*(size_t)_ws_slots));", \
                  "        // Phase 3a/b/c/e: L2-pin d_workspace for its lifetime. Spilled buffers", \
                  "        // (Minv-F, FD's Minv-F, ABA's inner scratch, FDSVA_SO's df_du/Minv) are", \
                  "        // recursion-hot — L2 pinning narrows the smem→HBM gap to smem→L2.", \
                  "        gpuErrchk(grid_begin_l2_persisting(0, hd_data->d_workspace, _ws_per_ts*(size_t)_ws_slots));", \
                  "    }"]
                  + ag_close() + [
                  "return hd_data;"])
    # Pinned host staging: page-locked pages let the DMA engine run at full
    # PCIe rate (measured 13.3 GB/s pageable vs ~50 GB/s pinned on gen5) and
    # are a prerequisite for async D2H overlap. cudaMallocHost can exhaust the
    # page-locked pool, so fall back to plain malloc; the free side asks the
    # driver which allocator owned the pointer instead of tracking a flag
    # (cudaPointerGetAttributes reports cudaMemoryTypeHost ONLY for pinned —
    # a plain-malloc pointer comes back cudaMemoryTypeUnregistered).
    self.gen_add_code_lines([
        "template <typename T>",
        "__host__",
        "T *grid_host_alloc(size_t bytes) {",
        "    void *p = nullptr;",
        "    if (cudaMallocHost(&p, bytes) == cudaSuccess) { return (T *)p; }",
        "    cudaGetLastError(); // consume the failed pinned alloc",
        "    return (T *)malloc(bytes);",
        "}",
        "",
        "template <typename T>",
        "__host__",
        "void grid_host_free(T *p) {",
        "    if (p == nullptr) { return; }",
        "    cudaPointerAttributes _attr;",
        "    if (cudaPointerGetAttributes(&_attr, p) == cudaSuccess && _attr.type == cudaMemoryTypeHost) {",
        "        cudaFreeHost(p);",
        "        return;",
        "    }",
        "    cudaGetLastError();",
        "    free(p);",
        "}",
        ""])
    # Device-pool (slab) mode (2026-09-09): an embedding framework (jax/torch)
    # installs a caller-owned device slab BEFORE init_gridData; every gridData
    # device allocation then carves from it (256-aligned bump) instead of
    # cudaMalloc, so GRiD's VRAM lives INSIDE the framework allocator's pool
    # (jax: an XLA-pool jnp buffer; torch: a caching-allocator tensor) rather
    # than fighting it — the root cause of the XLA-75%-prealloc "launch failed"
    # starvation class. gridData_device_bytes (emitted below, DERIVED from the
    # same alloc lines) tells the caller how big a slab to hand over; the
    # bindings assert used == bytes after a pool-mode init as the runtime
    # referee. The slab is caller-owned: close_grid's grid_device_free is a
    # no-op for carved pointers and the caller frees the slab by releasing its
    # framework buffer.
    self.gen_add_code_lines([
        "struct grid_device_pool_t { void *base; size_t bytes; size_t used; int ws_slots; };",
        "// ⚠hidden visibility is LOAD-BEARING: without it the dynamic linker",
        "// unifies this inline function's static (weak symbol) across every",
        "// dlopened robot .so, so a second robot's init would carve from the",
        "// FIRST robot's (already exhausted) slab and 'OOM' on an empty GPU",
        "// (observed 2026-09-09, jax-then-torch two-robot process).",
        "__host__ inline __attribute__((visibility(\"hidden\"))) grid_device_pool_t &grid_device_pool() {",
        "    static grid_device_pool_t p = {nullptr, 0, 0, 0};",
        "    return p;",
        "}",
        "__host__ __device__ constexpr size_t grid_pool_align(size_t b) { return (b + 255) & ~(size_t)255; }",
        "__host__ inline cudaError_t grid_device_alloc(void **p, size_t bytes) {",
        "    grid_device_pool_t &pool = grid_device_pool();",
        "    if (pool.base != nullptr) {",
        "        const size_t need = grid_pool_align(bytes);",
        "        if (pool.used + need > pool.bytes) { *p = nullptr; return cudaErrorMemoryAllocation; }",
        "        *p = (void *)((char *)pool.base + pool.used);",
        "        pool.used += need;",
        "        return cudaSuccess;",
        "    }",
        "    return cudaMalloc(p, bytes);",
        "}",
        "template <typename T>",
        "__host__ inline cudaError_t grid_device_free(T *p) {",
        "    grid_device_pool_t &pool = grid_device_pool();",
        "    if (pool.base != nullptr && (void *)p >= pool.base && (char *)p < (char *)pool.base + pool.bytes) {",
        "        return cudaSuccess;  // carved from the caller-owned slab: nothing to free",
        "    }",
        "    return cudaFree((void *)p);",
        "}",
        ""])
    # Device-pool mode (2026-09-09): (a) the workspace slot count honors an
    # installed pool's declared ws_slots (env override still wins; the
    # cudaMemGetInfo auto-fit stays the cudaMalloc-path fallback); (b) every
    # gridData cudaMalloc goes through grid_device_alloc (carve-or-malloc);
    # (c) gridData_device_bytes is DERIVED from the same lines so the slab
    # size and the carve can never drift.
    _ws_env_i = next(i for i, l in enumerate(code_lines) if "_ws_env != nullptr" in l)
    code_lines.insert(_ws_env_i + 1,
        "        else if (grid_device_pool().base != nullptr && grid_device_pool().ws_slots > 0) "
        "{ _ws_slots = grid_device_pool().ws_slots < NUM_TIMESTEPS ? grid_device_pool().ws_slots : NUM_TIMESTEPS; }")
    bytes_lines = _derive_device_bytes_lines(code_lines)
    code_lines = [l.replace("gpuErrchk(cudaMalloc((void**)&hd_data->",
                            "gpuErrchk(grid_device_alloc((void**)&hd_data->")
                  for l in code_lines]
    # generate as templated or not function
    self.gen_add_func_doc("Allocated device and host memory for all computations",
                          [], [], "A pointer to the gridData struct of pointers")
    self.gen_add_code_line("template <typename T, int NUM_TIMESTEPS, gridDataKind KIND = GRID_DATA_ALL>")
    self.gen_add_code_line("__host__")
    self.gen_add_code_line("gridData<T, KIND> *init_gridData(){", True)
    self.gen_add_code_lines(code_lines)
    self.gen_add_end_function()
    self.gen_add_func_doc("Allocated device and host memory for all computations",
                          [], ["Max number of timesteps in the trajectory"], "A pointer to the gridData struct of pointers")
    self.gen_add_code_line("template <typename T, gridDataKind KIND = GRID_DATA_ALL>")
    self.gen_add_code_line("__host__")
    self.gen_add_code_line("gridData<T, KIND> *init_gridData(int NUM_TIMESTEPS){", True)
    self.gen_add_code_lines(code_lines)
    self.gen_add_end_function()
    self.gen_add_func_doc("Device bytes a pool-mode init_gridData will carve for "
                          "this KIND at the given workspace slot count — size the "
                          "slab handed to grid_device_pool() with this (derived "
                          "from the SAME allocation list as init_gridData).",
                          [], ["workspace timestep slots (clamped to [1, NUM_TIMESTEPS])"],
                          "total device bytes (256-aligned per allocation)")
    self.gen_add_code_line("template <typename T, int NUM_TIMESTEPS, gridDataKind KIND = GRID_DATA_ALL>")
    self.gen_add_code_line("__host__")
    self.gen_add_code_line("size_t gridData_device_bytes(int ws_slots = NUM_TIMESTEPS){", True)
    self.gen_add_code_lines(bytes_lines)
    self.gen_add_end_function()
