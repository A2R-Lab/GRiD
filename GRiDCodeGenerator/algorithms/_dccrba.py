"""dCCRBA CUDA emit (PS5 oracle 1 — the hardest PS5 item).

Two analytic centroidal-derivative surfaces sharing ONE world-frame sweep:

  cmm_time_variation  Adot = dA(q(t))/dt = sum_m (dA/dq_m) qd_m   (6 x NV, no spill)
  dccrba              dA_dq[:, k, m] = d A[:, k]/d q_m            (6 x NV x NV, spill)

Both are direct transcriptions of the verified numpy oracle
`RBDReference._CentroidalMixin._dccrba_analytic` / `cmm_time_variation`
(validated vs pin.dccrba / computeCentroidalDynamicsDerivatives ~1e-12), so the
TIGHT value-tolerance bucket applies (NOT the FD bucket).

The shared world sweep is `centroidal_inner` (already byte-checked vs this oracle):
it leaves, in the caller's s_temp,
  s_Xworld (16*NJ), s_J = Jw (6*NV per body, angular-first), s_Iw (36 per body),
  s_A0 (6*NV world-ORIGIN momentum map [ang;lin]), plus s_com / s_extra[0]=mass.
On top of that we build the analytic tensor with the spatial cross operators
crm/crf (the Coriolis `_emit_crm_cm`/`_emit_crf_cm` builders;
dot_matrix(I,phi) = crf(phi)@I - I@crm(phi)).

Derivation (world-ORIGIN [ang;lin], then CoM-shifted + reordered):
  A0 = sum_i Iw_i Jw_i.  d/dq_m :
    * BASE dof m (floating root): dA0/dq_m = crf(phi_m) @ A0
    * JOINT dof m: for each body i,
        if owner(m) is ancestor-or-self of i:  dIw_i = dot_matrix(Iw_i, phi_m)
            contributes dIw_i @ Jw_i
        dJw_i col c += crm(phi_m) @ phi_c   for units c whose owner is STRICTLY
            below owner(m) in i's chain (owner(m) a strict ancestor of owner(c)),
            contributing Iw_i @ dJw_i.
  Then CoM dual-shift Xstar[:3,3:] = -skew(com); its q-derivative adds the
  CoM-motion term dXstar[:3,3:] = -skew(Jcom[:,m]) (Jcom = A0[lin]/mass); finally
  reorder rows [ang;lin] -> [lin;ang].

Per-(body,DOF) world "motion units" phi: baked owner body + local S axis; phi is
the world screw of that DOF at the world origin, recomputed here from s_Xworld
EXACTLY as centroidal_inner builds its Jw columns (so phi sums to Jw).

PARALLELISM: the NV output columns m of dA/dq are INDEPENDENT -> P2 fan over m
(each thread-group owns one m). The single world sweep is computed once by
centroidal_inner. cmm_time_variation contracts each column with qd[m] on the fly
(NO full-tensor materialization).

MIMIC: SUPPORTED. centroidal_inner's Jw is mimic-alpha-folded (s_J columns carry
the multiplier), and the per-unit phi here is likewise alpha-scaled (dc_unit_alpha,
mirroring the RBDReference _dccrba_world_sweep oracle). The downstream tensor math
is bilinear in phi so no further change is needed. Non-mimic emit stays byte-
identical (the alpha table + multiply live entirely inside HAS_MIMIC branches).

SPILL: dccrba's output s_dccrba (6*NV*NV) is the cold/large write-once output ->
repointed to the L2-pinned d_workspace SO band when DCCRBA_OUTPUT_IN_SMEM<TIER>()
is false (mirrors the regressor s_Y spill). cmm_time_variation (6*NV) never spills.
"""

import numpy as np

from ._coriolis import _emit_crm_cm, _emit_crf_cm, _coriolis_int_array as _dccrba_int_array
from ._centroidal import _centroidal_inner_temp_mem_size


def _dccrba_metadata(self):
    """Per-(body,local-DOF) world motion-unit tables + per-column job topology.

    One UNIT per S-column (n_int). For NON-mimic robots n_int == nv and vi is a
    bijection; the mimic gate keeps us here. Tables (all int, baked static const):
      unit_body[u]   owning body id (for the world (R,p) and the chain tests)
      unit_vi[u]     reduced v-slot the unit writes
      unit_ax[3u..]  local angular S axis (3 floats) — usually a unit axis
      unit_lin[3u..] local linear  S axis (3 floats)
      is_root_v[m]   1 if reduced slot m is a floating-base root slot
      unit_anc_self[u*NB + i]  1 if owner(u) is ancestor-or-self of body i
      unit_anc_strict[u*NB+ j] 1 if owner(u) is a STRICT ancestor of body j
    """
    robot = self.robot
    NB = robot.get_num_bodies()
    nv = robot.get_num_vel()
    HAS_MIMIC = self.robot_has_mimic_joints()

    def ancestors(i):
        res = set()
        j = robot.get_parent_id(i)
        while j != -1:
            res.add(j)
            j = robot.get_parent_id(j)
        return res

    anc = [ancestors(i) for i in range(NB)]
    anc_self = [anc[i] | {i} for i in range(NB)]

    unit_body, unit_vi, unit_ax, unit_lin, unit_alpha = [], [], [], [], []
    for j in range(NB):
        S = np.asarray(robot.get_S_by_id(j), dtype=np.float64)
        if S.ndim == 1:
            S = S.reshape(-1, 1)
        vinds = robot.get_joint_index_v(j)
        if not isinstance(vinds, (list, tuple, np.ndarray)):
            vinds = [vinds]
        vinds = [int(v) for v in np.asarray(vinds).reshape(-1)]
        # MIMIC: the per-unit world motion column phi is alpha-scaled exactly as
        # the RBDReference oracle (_dccrba_world_sweep / _body_spatial_jacobian_world)
        # and centroidal_inner's s_J. Non-mimic alpha == 1.0 (unused, gated off).
        alpha = self._alpha_for_jid(j) if HAS_MIMIC else 1.0
        for c in range(S.shape[1]):
            vi = vinds[c] if c < len(vinds) else vinds[-1]
            unit_body.append(j)
            unit_vi.append(vi)
            unit_alpha.append(alpha)
            unit_ax += [float(S[0, c]), float(S[1, c]), float(S[2, c])]
            unit_lin += [float(S[3, c]), float(S[4, c]), float(S[5, c])]
    n_int = len(unit_body)

    # floating-base root reduced slots (the 6 root columns)
    root_v = set()
    if robot.floating_base:
        rv = robot.get_joint_index_v(0)
        if not isinstance(rv, (list, tuple, np.ndarray)):
            rv = [rv]
        root_v = {int(v) for v in np.asarray(rv).reshape(-1)}
    is_root_v = [1 if m in root_v else 0 for m in range(nv)]

    # per-unit ancestor membership flags (flattened u*NB + body)
    unit_anc_self = []
    unit_anc_strict = []
    for u in range(n_int):
        jb = unit_body[u]
        for i in range(NB):
            unit_anc_self.append(1 if jb in anc_self[i] else 0)
        for jj in range(NB):
            unit_anc_strict.append(1 if jb in anc[jj] else 0)

    return {
        "NB": NB, "nv": nv, "n_int": n_int, "HAS_MIMIC": HAS_MIMIC,
        "unit_body": unit_body, "unit_vi": unit_vi, "unit_alpha": unit_alpha,
        "unit_ax": unit_ax, "unit_lin": unit_lin,
        "is_root_v": is_root_v,
        "unit_anc_self": unit_anc_self, "unit_anc_strict": unit_anc_strict,
    }


# ===========================================================================
# Shared inner: build per-unit world motion columns phi (s_phi), then assemble
# the dA0/dq tensor column-by-column. Parameterized on whether we materialize the
# full 6*NV*NV tensor (dccrba) or contract on the fly with qd into 6*NV (Adot).
# ===========================================================================

def _dccrba_sweep_J_count(self):
    """Size (t) of the Jw sweep band (6*nv*NB) that dccrba/cmm hold externally in
    s_J_ext (in-smem tier-routed slot at L0/L1, d_workspace at the J-spilled tier)."""
    NB = self.robot.get_num_bodies()
    nv = self.robot.get_num_vel()
    return 6 * nv * NB


def _dccrba_inner_temp_mem_size(self):
    # The s_temp pool the dccrba/cmm inners use: the SHRUNK (no-J) centroidal pool
    # (s_Xworld/s_Iw/s_A0/s_IW) + s_phi (6*n_int world motion columns). The Jw band
    # (6*nv*NB) is held EXTERNALLY in s_J_ext, so it is NOT part of s_temp. Each
    # (m,k) cell thread writes the output directly, so s_phi is the only shared inner
    # scratch beyond the centroidal pool.
    md = _dccrba_metadata(self)
    return _centroidal_inner_temp_mem_size(self, j_in_smem=False) + 6 * md["n_int"]


def _emit_dccrba_assembly(self, out_name, contract_qd):
    """Emit the per-column assembly. `out_name` is the output buffer name.
    If contract_qd: out is 6*nv (Adot), each m-column scaled by qd[m] and atomic-
    added. Else: out is 6*nv*nv, column m written at out[row + 6*k + 6*nv*m]."""
    md = _dccrba_metadata(self)
    NB = md["NB"]
    nv = md["nv"]
    n_int = md["n_int"]
    HAS_MIMIC = md["HAS_MIMIC"]
    NJ = self.robot.get_num_joints()

    # scratch pointers. dccrba/cmm always run centroidal_inner with the s_J band
    # held EXTERNALLY (s_J_ext): at L0/L1 s_J_ext is an in-smem tier-routed buffer,
    # at the J-spilled tier (DE-GATE #2) it is the L2-pinned d_workspace SO band.
    # Either way the s_temp pool is the shrunk (no-J) centroidal pool, so the
    # trailing s_Iw/s_A0 and s_phi sit right after s_Xworld with no s_J hole.
    off_J = 16 * NJ                       # s_Xworld occupies [0, 16*NJ)
    off_Iw = off_J                        # s_J lives in s_J_ext, not s_temp
    off_A0 = off_Iw + 36 * NB
    off_phi = _centroidal_inner_temp_mem_size(self, False)   # shrunk (no-J) pool

    self.gen_add_code_lines([
        "// dccrba scratch: centroidal_inner left s_Xworld(0)/s_Iw/s_A0 in s_temp and",
        "// the Jw band (6*NV per body) in s_J_ext (in-smem at L0/L1, d_workspace at L2).",
        f"T *dc_Xworld = &s_temp[0];",
        f"T *dc_J  = s_J_ext;            // Jw, 6*NV per body (angular-first)",
        f"T *dc_Iw = &s_temp[{off_Iw}];  // 36 per body world inertia",
        f"T *dc_A0 = &s_temp[{off_A0}];  // 6*NV world-origin momentum map [ang;lin]",
        f"T *s_phi = &s_temp[{off_phi}]; // 6*n_int per-unit world motion columns",
    ])

    # baked topology tables
    self.gen_add_code_lines([
        f"static const int dc_unit_body[] = {{ {_dccrba_int_array(md['unit_body'])} }};",
        f"static const int dc_unit_vi[] = {{ {_dccrba_int_array(md['unit_vi'])} }};",
        "static const T dc_unit_ax[] = { " + ", ".join(
            "static_cast<T>({:.17g})".format(v) for v in md["unit_ax"]) + " };",
        "static const T dc_unit_lin[] = { " + ", ".join(
            "static_cast<T>({:.17g})".format(v) for v in md["unit_lin"]) + " };",
        f"static const int dc_is_root_v[] = {{ {_dccrba_int_array(md['is_root_v'])} }};",
        f"static const int dc_unit_anc_self[] = {{ {_dccrba_int_array(md['unit_anc_self'])} }};",
        f"static const int dc_unit_anc_strict[] = {{ {_dccrba_int_array(md['unit_anc_strict'])} }};",
    ])
    if HAS_MIMIC:
        # MIMIC: per-unit mimic multiplier; folds into phi (the alpha-scaled world
        # motion column), mirroring the RBDReference _dccrba_world_sweep oracle.
        self.gen_add_code_line(
            "static const T dc_unit_alpha[] = { " + ", ".join(
                "static_cast<T>({:.17g})".format(v) for v in md["unit_alpha"]) + " };")

    # ---- Step A: per-unit world motion column phi (angular-first), EXACTLY as
    #      centroidal_inner builds Jw columns: aw = R_j*ang ; lw = R_j*lin ;
    #      phi = [aw ; lw + p_j x aw].  P2 fan over units. ----
    self.gen_add_code_line("// per-unit world motion column phi (angular-first)")
    self.gen_add_parallel_loop("u", str(n_int))
    self.gen_add_code_line("int jb = dc_unit_body[u]; const T *Xj = &dc_Xworld[16*jb];")
    self.gen_add_code_line("T a0=dc_unit_ax[3*u], a1=dc_unit_ax[3*u+1], a2=dc_unit_ax[3*u+2];")
    self.gen_add_code_line("T l0=dc_unit_lin[3*u], l1=dc_unit_lin[3*u+1], l2=dc_unit_lin[3*u+2];")
    self.gen_add_code_line("T aw0 = Xj[0]*a0 + Xj[4]*a1 + Xj[8]*a2;")
    self.gen_add_code_line("T aw1 = Xj[1]*a0 + Xj[5]*a1 + Xj[9]*a2;")
    self.gen_add_code_line("T aw2 = Xj[2]*a0 + Xj[6]*a1 + Xj[10]*a2;")
    self.gen_add_code_line("T lw0 = Xj[0]*l0 + Xj[4]*l1 + Xj[8]*l2;")
    self.gen_add_code_line("T lw1 = Xj[1]*l0 + Xj[5]*l1 + Xj[9]*l2;")
    self.gen_add_code_line("T lw2 = Xj[2]*l0 + Xj[6]*l1 + Xj[10]*l2;")
    self.gen_add_code_line("T pjx = Xj[12], pjy = Xj[13], pjz = Xj[14];")
    # MIMIC: scale the world motion column by the unit's mimic multiplier alpha
    # (phi[:3]=alpha*aw; phi[3:]=alpha*(lw + p x aw)); non-mimic alpha==1.0 and the
    # multiply is gated off so non-mimic emit is byte-identical. Downstream tensor
    # math (crm/crf/dot_matrix) is bilinear in phi -> no other change needed.
    if HAS_MIMIC:
        self.gen_add_code_line("T al = dc_unit_alpha[u];")
        self.gen_add_code_line("s_phi[6*u+0] = al*aw0; s_phi[6*u+1] = al*aw1; s_phi[6*u+2] = al*aw2;")
        self.gen_add_code_line("s_phi[6*u+3] = al*(lw0 + (pjy*aw2 - pjz*aw1));")
        self.gen_add_code_line("s_phi[6*u+4] = al*(lw1 + (pjz*aw0 - pjx*aw2));")
        self.gen_add_code_line("s_phi[6*u+5] = al*(lw2 + (pjx*aw1 - pjy*aw0));")
    else:
        self.gen_add_code_line("s_phi[6*u+0] = aw0; s_phi[6*u+1] = aw1; s_phi[6*u+2] = aw2;")
        self.gen_add_code_line("s_phi[6*u+3] = lw0 + (pjy*aw2 - pjz*aw1);")
        self.gen_add_code_line("s_phi[6*u+4] = lw1 + (pjz*aw0 - pjx*aw2);")
        self.gen_add_code_line("s_phi[6*u+5] = lw2 + (pjx*aw1 - pjy*aw0);")
    self.gen_add_end_control_flow()
    self.gen_add_sync()

    # ---- com + Jcom (Jcom[:,m] = dc_A0[lin row, m]/mass) ----
    self.gen_add_code_line("T dc_cx = s_com[0], dc_cy = s_com[1], dc_cz = s_com[2];")
    self.gen_add_code_line("T dc_inv_m = static_cast<T>(1)/s_extra[0];")

    if contract_qd:
        # zero the Adot output (6*nv) once
        self.gen_add_code_line("// zero Adot output")
        self.gen_add_parallel_loop("ind", str(6 * nv))
        self.gen_add_code_line(out_name + "[ind] = static_cast<T>(0);")
        self.gen_add_end_control_flow()
        self.gen_add_sync()

    # ---- Step B: P2 fan over columns m. Each thread owns ONE m, builds the full
    #      6*nv column dA[:, :, m] into a per-thread accumulator? 6*nv can be large
    #      (g1: 216). Per-thread stack arrays of that size blow registers. Instead
    #      each (m) thread loops k over nv and writes directly: for the on-the-fly
    #      Adot we accumulate the contracted column; for the full tensor we write
    #      each (row,k) cell. To keep per-thread memory bounded we recompute the
    #      6-vector dA0[:,k,m] per (m,k) cell.
    #
    #      dA0[:,k,m] (world-origin [ang;lin], 6-vector) =
    #        base m:  (crf(phi_m) @ A0)[:,k]
    #        joint m: sum_i [ (owner(m) anc-self i) * dot_matrix(Iw_i, phi_m) @ Jw_i[:,k] ]
    #                 + sum_i Iw_i @ ( sum_{units c: c anc-self i & owner(m) strict-anc owner(c) & vi(c)==k} crm(phi_m) @ phi_c )
    #      We fold over the units owning slot m (mimic gated -> exactly one unit). ----
    #
    # Fan over (m, k) cells = nv*nv threads. Each computes the 6-vector dA0col then
    # applies CoM-shift + reorder, then writes (contract or full).
    self.gen_add_code_line("// P2 fan: one thread per (m, k) output cell")
    self.gen_add_parallel_loop("cell", str(nv * nv))
    self.gen_add_code_line(f"int m = cell / {nv}; int k = cell % {nv};")
    self.gen_add_code_line("T dA0col[6]; for (int r=0;r<6;++r) dA0col[r] = static_cast<T>(0);")

    # find the unit(s) owning slot m. Non-mimic: exactly one. We loop all units and
    # branch on vi==m to stay mimic-structurally-correct (gate keeps it 1-unit).
    self.gen_add_code_line("bool m_is_root = dc_is_root_v[m] != 0;")
    self.gen_add_code_line(f"for (int um = 0; um < {n_int}; ++um) {{", True)
    self.gen_add_code_line("if (dc_unit_vi[um] != m) continue;")
    self.gen_add_code_line("const T *phim = &s_phi[6*um]; int jm = dc_unit_body[um];")
    # crm(phi_m), crf(phi_m) as 6x6 col-major register builds
    _emit_crm_cm(self, "CrmM", "phim")
    _emit_crf_cm(self, "CrfM", "phim")
    self.gen_add_code_line("if (m_is_root) {", True)
    # base: dA0col = (crf(phi_m) @ A0)[:, k] = CrfM @ A0[:,k]
    self.gen_add_code_line("const T *A0k = &dc_A0[6*k];")
    self.gen_add_code_line("for (int r=0;r<6;++r){ T s=0; for(int c=0;c<6;++c) s += CrfM[r+6*c]*A0k[c]; dA0col[r] += s; }")
    self.gen_add_code_line("continue;")
    self.gen_add_end_control_flow()
    # joint m: loop bodies i
    self.gen_add_code_line(f"for (int i = 0; i < {NB}; ++i) {{", True)
    self.gen_add_code_line(f"const T *Iw_i = &dc_Iw[36*i];")
    self.gen_add_code_line(f"const T *Jw_i = &dc_J[{6*nv}*i];")
    # term1: if owner(m) anc-self i -> dot_matrix(Iw_i, phi_m) @ Jw_i[:,k]
    #   dot_matrix(I,phi) = crf(phi)@I - I@crm(phi). Apply to vector Jw_i[:,k]:
    #   (crf(phi)@I - I@crm(phi)) @ x  with x = Jw_i[:,k]
    self.gen_add_code_line(f"if (dc_unit_anc_self[um*{NB} + i]) {{", True)
    self.gen_add_code_line("const T *xk = &Jw_i[6*k];")
    # tmp1 = I @ x ; t1 = crf(phi) @ tmp1
    self.gen_add_code_line("T Ix[6]; for(int r=0;r<6;++r){ T s=0; for(int c=0;c<6;++c) s += Iw_i[r+6*c]*xk[c]; Ix[r]=s; }")
    self.gen_add_code_line("T cmx[6]; for(int r=0;r<6;++r){ T s=0; for(int c=0;c<6;++c) s += CrmM[r+6*c]*xk[c]; cmx[r]=s; }")
    self.gen_add_code_line("T Icmx[6]; for(int r=0;r<6;++r){ T s=0; for(int c=0;c<6;++c) s += Iw_i[r+6*c]*cmx[c]; Icmx[r]=s; }")
    self.gen_add_code_line("for(int r=0;r<6;++r){ T s=0; for(int c=0;c<6;++c) s += CrfM[r+6*c]*Ix[c]; dA0col[r] += s - Icmx[r]; }")
    self.gen_add_end_control_flow()
    # term2: Iw_i @ dJi[:,k] where dJi[:,k] = sum_{units c: c anc-self i, owner(m) strict-anc owner(c), vi(c)==k} crm(phi_m) @ phi_c
    self.gen_add_code_line("T dJk[6]; for(int r=0;r<6;++r) dJk[r] = static_cast<T>(0);")
    self.gen_add_code_line(f"for (int uc = 0; uc < {n_int}; ++uc) {{", True)
    self.gen_add_code_line("if (dc_unit_vi[uc] != k) continue;")
    self.gen_add_code_line(f"if (!dc_unit_anc_self[uc*{NB} + i]) continue;")
    self.gen_add_code_line(f"if (!dc_unit_anc_strict[um*{NB} + dc_unit_body[uc]]) continue;")
    self.gen_add_code_line("const T *phic = &s_phi[6*uc];")
    self.gen_add_code_line("for(int r=0;r<6;++r){ T s=0; for(int c=0;c<6;++c) s += CrmM[r+6*c]*phic[c]; dJk[r] += s; }")
    self.gen_add_end_control_flow()
    # dA0col += Iw_i @ dJk
    self.gen_add_code_line("for(int r=0;r<6;++r){ T s=0; for(int c=0;c<6;++c) s += Iw_i[r+6*c]*dJk[c]; dA0col[r] += s; }")
    self.gen_add_end_control_flow()  # body i loop
    self.gen_add_end_control_flow()  # unit um loop

    # ---- CoM-shift + reorder for cell (k is the CMM column index) ----
    # dA_fs = dXstar @ A0[:,k] + Xstar @ dA0col, where
    #   Xstar[:3,3:] = -skew(com); dXstar[:3,3:] = -skew(Jcom[:,m]); Jcom[:,m]=A0[lin,m]/mass.
    # Xstar @ x : ang' = x_ang - com x x_lin ; lin' = x_lin
    # dXstar @ A0[:,k] : ang' += -Jcom[:,m] x A0_lin[:,k] ; lin' += 0
    self.gen_add_code_line("// CoM-shift (Xstar) + CoM-motion (dXstar) + [ang;lin]->[lin;ang] reorder")
    self.gen_add_code_line("const T *A0k2 = &dc_A0[6*k];")
    self.gen_add_code_line("T jcx = dc_A0[3 + 6*m]*dc_inv_m, jcy = dc_A0[4 + 6*m]*dc_inv_m, jcz = dc_A0[5 + 6*m]*dc_inv_m;")
    # ang_out (Featherstone angular rows 0..2)
    self.gen_add_code_line("T fa0 = A0k2[3], fa1 = A0k2[4], fa2 = A0k2[5];")  # A0 linear part of column k
    self.gen_add_code_line("T xa0 = dA0col[0] - (dc_cy*dA0col[5] - dc_cz*dA0col[4]);")
    self.gen_add_code_line("T xa1 = dA0col[1] - (dc_cz*dA0col[3] - dc_cx*dA0col[5]);")
    self.gen_add_code_line("T xa2 = dA0col[2] - (dc_cx*dA0col[4] - dc_cy*dA0col[3]);")
    # add dXstar term: -Jcom[:,m] x A0_lin[:,k]
    self.gen_add_code_line("xa0 += -(jcy*fa2 - jcz*fa1);")
    self.gen_add_code_line("xa1 += -(jcz*fa0 - jcx*fa2);")
    self.gen_add_code_line("xa2 += -(jcx*fa1 - jcy*fa0);")
    # lin_out = dA0col linear part (unchanged by Xstar/dXstar)
    self.gen_add_code_line("T xl0 = dA0col[3], xl1 = dA0col[4], xl2 = dA0col[5];")
    # reorder [ang;lin] -> [lin;ang]: out rows [0..2]=lin, [3..5]=ang
    if contract_qd:
        self.gen_add_code_line("T qm = s_qd[m];")
        self.gen_add_code_line("atomicAdd(&" + out_name + "[0 + 6*k], xl0*qm);")
        self.gen_add_code_line("atomicAdd(&" + out_name + "[1 + 6*k], xl1*qm);")
        self.gen_add_code_line("atomicAdd(&" + out_name + "[2 + 6*k], xl2*qm);")
        self.gen_add_code_line("atomicAdd(&" + out_name + "[3 + 6*k], xa0*qm);")
        self.gen_add_code_line("atomicAdd(&" + out_name + "[4 + 6*k], xa1*qm);")
        self.gen_add_code_line("atomicAdd(&" + out_name + "[5 + 6*k], xa2*qm);")
    else:
        # full tensor: out[row + 6*k + 6*nv*m]
        base = f"{out_name}[6*k + {6*nv}*m"
        self.gen_add_code_line(base + " + 0] = xl0;")
        self.gen_add_code_line(base + " + 1] = xl1;")
        self.gen_add_code_line(base + " + 2] = xl2;")
        self.gen_add_code_line(base + " + 3] = xa0;")
        self.gen_add_code_line(base + " + 4] = xa1;")
        self.gen_add_code_line(base + " + 5] = xa2;")
    self.gen_add_end_control_flow()  # cell loop
    self.gen_add_sync()


# ===========================================================================
# mjx output-convention epilogue for the dccrba TENSOR dA_dq[:,k,m]
# ===========================================================================

def _emit_dccrba_mjx_output(self, out_name):
    """Emit the MuJoCo/mjx output-convention transform of the dccrba tensor, in
    place on ``out_name`` (the buffer holding s_dccrba, whichever tier routed it).

    Tensor layout (kernel): ``out_name[i + 6*k + 6*NV*m] = dA[i,k]/dq_m`` with
    i:momentum-row(0..5), k:qd-COLUMN(NV), m:q-TANGENT(NV). This is the oracle
    ``T[i,l,m]`` (l==k). The centroidal momentum lives in the world-aligned CoM
    frame, so the momentum-row index i is INVARIANT; only the velocity/config
    tangent reparameterization G = blockdiag(R, I) acts. Transform (validated vs
    `RBDReference.equivalents.mujoco_convention.dccrba_dA_dq_pin_to_mjx` to 4e-16):

        out[i,a,k] = sum_{l,m} T[i,l,m] Ginv[l,a] Ginv[m,k]              (double reframe)
        out[i,:,k=3+c] += (A_pin @ g_dot(R,c)^T)[i,:]   c=0,1,2          (frame term)

    Ginv = G^T => base-linear 3x3 block = R^T (Ginv[l,a]=R[a][l] for l,a in 0..2;
    identity elsewhere), so each reframe MIXES ONLY the base-linear indices {0,1,2}.
    g_dot(R,c) base-linear block = R @ skew(e_c); A_pin is read live from s_A (the
    centroidal_inner CMM value, col-major s_A[i+6*l], NOT overwritten by the tensor
    assembly).

    PARALLELIZED across the block (was single-thread; ~50% runtime overhead at large
    batch — mirrors the validated id-gradient mjx epilogue). The two reframe steps
    fan over the 6*NV (i,k)/(i,m) output cells and the frame term fans over the 6
    momentum rows. R is recomputed REGISTER-LOCAL at the top of every parallel-loop
    body from the (already xyzw-reordered) base quaternion s_q[3..6] — a few dozen
    flops, zero aliasing risk (no shared dead-scratch staging). s_q and s_A are
    block-shared read-only inputs (s_A is NOT overwritten by the tensor assembly), so
    every thread reads them safely. Phase boundaries carry read-after-write deps:
    step 2 reads step-1 outputs, step 3 += into step-1/2 outputs -> a sync after each.
    Fixed base never reaches here (the call site gates on floating_base)."""
    nv = self.robot.get_num_vel()
    # R-build lines (3x3 row-major R[3*i+j]) from the xyzw base quaternion at s_q[3..6]
    # -- EXACTLY mirroring helpers._gen_mjx_build_R_lines / mujoco_convention.
    # rotation_from_quat_xyzw. Re-materialized register-local at the top of each
    # parallel-loop body (loop-invariant; recompute is cheaper than any shared stage
    # and carries no aliasing risk -- see the fdsva_so dead-scratch hazard).
    _BUILD_R = [
        "T qx = s_q[3], qy = s_q[4], qz = s_q[5], qw = s_q[6];",
        "T xx = qx*qx, yy = qy*qy, zz = qz*qz;",
        "T xy = qx*qy, xz = qx*qz, yz = qy*qz, wx = qw*qx, wy = qw*qy, wz = qw*qz;",
        "T R[9];",
        "R[0] = static_cast<T>(1) - static_cast<T>(2)*(yy+zz); R[1] = static_cast<T>(2)*(xy-wz);                    R[2] = static_cast<T>(2)*(xz+wy);",
        "R[3] = static_cast<T>(2)*(xy+wz);                    R[4] = static_cast<T>(1) - static_cast<T>(2)*(xx+zz); R[5] = static_cast<T>(2)*(yz-wx);",
        "R[6] = static_cast<T>(2)*(xz-wy);                    R[7] = static_cast<T>(2)*(yz+wx);                    R[8] = static_cast<T>(1) - static_cast<T>(2)*(xx+yy);",
    ]
    self.gen_add_code_lines([
        "// mjx output-convention transform of the dccrba tensor dA_dq[i + 6*k + 6*NV*m]",
        "// (double G^{-1} reframe of the qd-col k and q-tangent m base-linear indices",
        "//  + base-rotation frame term A_pin @ g_dot(R,c)^T). See _emit_dccrba_mjx_output.",
    ])
    # ---- Step 1: reframe the q-tangent index m on base cols m=0,1,2, for every
    #      (i,k) cell. tmp[i,k,a] = sum_{m=0..2} T[i,k,m] R[a][m] (Ginv[m,a]=R[a][m]).
    #      Read the 3 originals into registers, then overwrite. P2 fan over the
    #      6*NV (i,k) cells (the ik index) -- each cell touches only its own col triple.
    self.gen_add_code_line("// step 1: reframe the q-tangent index m (base-linear cols 0,1,2), per (i,k) cell")
    self.gen_add_parallel_loop("ik", str(6 * nv))
    self.gen_add_code_lines(_BUILD_R)
    self.gen_add_code_line(f"T t0 = {out_name}[ik + {6*nv}*0], t1 = {out_name}[ik + {6*nv}*1], t2 = {out_name}[ik + {6*nv}*2];")
    self.gen_add_code_line(f"{out_name}[ik + {6*nv}*0] = R[0]*t0 + R[1]*t1 + R[2]*t2;")
    self.gen_add_code_line(f"{out_name}[ik + {6*nv}*1] = R[3]*t0 + R[4]*t1 + R[5]*t2;")
    self.gen_add_code_line(f"{out_name}[ik + {6*nv}*2] = R[6]*t0 + R[7]*t1 + R[8]*t2;")
    self.gen_add_end_control_flow()
    self.gen_add_sync()
    # ---- Step 2: reframe the qd-col index k on base cols k=0,1,2, for every (i,m)
    #      cell. out[i,a,m] = sum_{k=0..2} R[a][k] tmp[i,k,m]. Reads step-1 values
    #      (sync above). P2 fan over the 6*NV (i,m) cells; cell -> m = cell/6, i = cell%6
    #      (preserves the original i-inner, m-outer iteration -> identical accumulation). ----
    self.gen_add_code_line("// step 2: reframe the qd-col index k (base-linear cols 0,1,2), per (i,m) cell")
    self.gen_add_parallel_loop("cell", str(6 * nv))
    self.gen_add_code_lines(_BUILD_R)
    self.gen_add_code_line(f"int m = cell / 6; int i = cell % 6;")
    self.gen_add_code_line(f"int b = i + {6*nv}*m;")
    self.gen_add_code_line(f"T t0 = {out_name}[b + 6*0], t1 = {out_name}[b + 6*1], t2 = {out_name}[b + 6*2];")
    self.gen_add_code_line(f"{out_name}[b + 6*0] = R[0]*t0 + R[1]*t1 + R[2]*t2;")
    self.gen_add_code_line(f"{out_name}[b + 6*1] = R[3]*t0 + R[4]*t1 + R[5]*t2;")
    self.gen_add_code_line(f"{out_name}[b + 6*2] = R[6]*t0 + R[7]*t1 + R[8]*t2;")
    self.gen_add_end_control_flow()
    self.gen_add_sync()
    # ---- Step 3: base-rotation frame term. For c=0,1,2 (q-tangent col k=3+c) and
    #      qd-col a in {0,1,2}, ALL rows i:
    #        out[i, a, 3+c] += sum_{b=0..2} A_pin[i,b] * Gd_c[a][b],  Gd_c = R @ skew(e_c).
    #      Kernel index out_name[i + 6*a + 6*NV*(3+c)]; A_pin col-major s_A[i + 6*b].
    #      P2 fan over the 6 momentum rows i (each i writes its own disjoint cells in
    #      cols 3+c, a in 0..2; reads the live read-only s_A + step-1/2 outputs, sync above).
    #      Gd_c[a][b] = sum_p R[a][p] skew(e_c)[p][b], R row-major (R[3*a+p]):
    #        skew(e0)=[[0,0,0],[0,0,-1],[0,1,0]] -> Gd0[a]=[0, R[3a+2], -R[3a+1]]
    #        skew(e1)=[[0,0,1],[0,0,0],[-1,0,0]] -> Gd1[a]=[-R[3a+2], 0, R[3a+0]]
    #        skew(e2)=[[0,-1,0],[1,0,0],[0,0,0]] -> Gd2[a]=[R[3a+1], -R[3a+0], 0]
    #      The (e_c x .) signs are BAKED in the per-(c,a) rhs below (validated vs the
    #      oracle by transcribing this exact encoding -> 4e-16). ----
    self.gen_add_code_line("// step 3: base-rotation frame term  out[i,a,3+c] += A_pin[i,:] . (R skew(e_c))[a][:], per row i")
    self.gen_add_parallel_loop("i", "6")
    self.gen_add_code_lines(_BUILD_R)
    self.gen_add_code_line("T a0 = s_A[i + 6*0], a1 = s_A[i + 6*1], a2 = s_A[i + 6*2];")
    # gd[c][a] = list over b of (R-index or None, negate?) for Gd_c[a][b]
    gd = {
        0: lambda a: [(None, False), (3 * a + 2, False), (3 * a + 1, True)],
        1: lambda a: [(3 * a + 2, True), (None, False), (3 * a + 0, False)],
        2: lambda a: [(3 * a + 1, False), (3 * a + 0, True), (None, False)],
    }
    for c in range(3):
        k = 3 + c
        for a in range(3):
            terms = []
            for b, (ridx, neg) in enumerate(gd[c](a)):
                if ridx is None:
                    continue
                terms.append(f"{'- ' if neg else '+ '}a{b}*R[{ridx}]")
            rhs = " ".join(terms)
            rhs = rhs[2:] if rhs.startswith("+ ") else "-" + rhs[2:]
            self.gen_add_code_line(f"{out_name}[i + 6*{a} + {6*nv}*{k}] += {rhs};")
    self.gen_add_end_control_flow()
    self.gen_add_sync()


# ===========================================================================
# cmm_time_variation (Adot, 6*NV, NO spill)
# ===========================================================================

def _cmm_time_variation_inner(self):
    nv = self.robot.get_num_vel()
    func_params = [
        "s_adot is the output Adot = dA/dt, 6 x NUM_VEL (column-major, [linear;angular] @ CoM) = " + str(6 * nv),
        "s_q is the joint positions (unused; q is baked into s_Xhom)",
        "s_qd is the joint velocities (contracted: Adot = sum_m (dA/dq_m) qd_m)",
        "s_Xhom is the per-joint LOCAL homogeneous transforms",
        "d_robotModel is the GPU model helpers (constant body inertias)",
        "s_temp is scratch of size " + str(_dccrba_inner_temp_mem_size(self)),
        "s_linalg_smem is reserved (unused)",
    ]
    func_def = ("void cmm_time_variation_inner(T *s_adot, T *s_A, T *s_com, T *s_extra, const T *s_q, const T *s_qd, const T *s_Xhom, "
                "const robotModel<T> *d_robotModel, T *s_temp, T *s_J_ext, unsigned char *s_linalg_smem) {")
    self.gen_add_func_doc("Compute Adot = dA(q(t))/dt = sum_m (dA/dq_m) qd_m (analytic dCCRBA contraction)",
                          [], func_params, None)
    self.gen_add_code_line("template <typename T>")
    self.gen_add_code_line("__device__")
    self.gen_add_code_line(func_def, True)
    self.gen_add_code_line("(void)s_q;")
    # The Jw sweep band lives EXTERNALLY in s_J_ext (in-smem at L0/L1, d_workspace
    # at the J-spilled tier), so centroidal_inner runs with J_IN_SMEM=false and the
    # shrunk (no-J) s_temp pool. The CoM/CMM math is a pure pointer move (DE-GATE #2).
    self.gen_add_code_line("centroidal_inner<T, false>(s_A, s_com, s_extra, s_q, s_Xhom, d_robotModel, s_temp, s_J_ext, s_linalg_smem);")
    self.gen_add_sync()
    _emit_dccrba_assembly(self, "s_adot", contract_qd=True)
    self.gen_add_end_function()


def _dccrba_full_inner(self):
    nv = self.robot.get_num_vel()
    func_params = [
        "s_dccrba is the output tensor dA_dq[:,k,m], 6*NUM_VEL*NUM_VEL = " + str(6 * nv * nv) +
        " (layout dA[row + 6*k + 6*NV*m], [linear;angular] @ CoM)",
        "s_q is the joint positions (unused; baked into s_Xhom)",
        "s_Xhom is the per-joint LOCAL homogeneous transforms",
        "d_robotModel is the GPU model helpers (constant body inertias)",
        "s_temp is scratch of size " + str(_dccrba_inner_temp_mem_size(self)),
        "s_linalg_smem is reserved (unused)",
    ]
    func_def = ("void dccrba_inner(T *s_dccrba, T *s_A, T *s_com, T *s_extra, const T *s_q, const T *s_Xhom, "
                "const robotModel<T> *d_robotModel, T *s_temp, T *s_J_ext, unsigned char *s_linalg_smem) {")
    self.gen_add_func_doc("Compute the analytic dCCRBA tensor dA_dq[:,k,m] (6 x NV x NV)",
                          [], func_params, None)
    self.gen_add_code_line("template <typename T>")
    self.gen_add_code_line("__device__")
    self.gen_add_code_line(func_def, True)
    self.gen_add_code_line("(void)s_q;")
    # The Jw sweep band lives EXTERNALLY in s_J_ext (in-smem at L0/L1, d_workspace
    # at the J-spilled tier), so centroidal_inner runs with J_IN_SMEM=false and the
    # shrunk (no-J) s_temp pool. Pure pointer move (DE-GATE #2).
    self.gen_add_code_line("centroidal_inner<T, false>(s_A, s_com, s_extra, s_q, s_Xhom, d_robotModel, s_temp, s_J_ext, s_linalg_smem);")
    self.gen_add_sync()
    _emit_dccrba_assembly(self, "s_dccrba", contract_qd=False)
    self.gen_add_end_function()


# ----- device wrappers (kinematics / XmatsHom domain, like com/ccrba) -----

def _centroidal_out_extra(self):
    nv = self.robot.get_num_vel()
    return [("s_A", 6 * nv), ("s_com", 3), ("s_extra", 4)]


def gen_cmm_time_variation_device(self):
    nv = self.robot.get_num_vel()
    func_def = ("void cmm_time_variation_device(T *s_adot, const T *s_q, const T *s_qd, const robotModel<T> *d_robotModel) {")
    func_params = ["s_adot holds Adot (6 x NUM_VEL, column-major [linear;angular] @ CoM)",
                   "s_q / s_qd are joint position / velocity", "d_robotModel is the GPU model helpers"]
    self.gen_add_func_doc("Compute Adot = dA/dt (analytic dCCRBA contraction)", [], func_params, None)
    self.gen_add_code_line("template <typename T>")
    self.gen_add_code_line("__device__")
    self.gen_add_code_line(func_def, True)
    # s_J holds the externalized Jw sweep band in smem (device wrappers don't spill).
    extra = [("s_A", 6 * nv), ("s_com", 3), ("s_extra", 4), ("s_J", _dccrba_sweep_J_count(self))]
    self.gen_XmatsHom_helpers_temp_shared_memory_code(
        _dccrba_inner_temp_mem_size(self), extra_t_buffers=extra, include_linalg_scratch=True,
        linalg_scratch_bytes="GRID_EE_LINALG_SHARED_BYTES<T>()")
    self.gen_load_update_XmatsHom_helpers_function_call()
    self.gen_add_code_line("cmm_time_variation_inner<T>(s_adot, s_A, s_com, s_extra, s_q, s_qd, s_XmatsHom, d_robotModel, s_temp, s_J, s_linalg_smem);")
    self.gen_add_sync()
    self.gen_add_end_function()


def gen_dccrba_device(self):
    nv = self.robot.get_num_vel()
    func_def = ("void dccrba_device(T *s_dccrba, const T *s_q, const robotModel<T> *d_robotModel) {")
    func_params = ["s_dccrba holds the tensor dA_dq[:,k,m] (6*NV*NV, dA[row + 6*k + 6*NV*m])",
                   "s_q is joint position", "d_robotModel is the GPU model helpers"]
    self.gen_add_func_doc("Compute the analytic dCCRBA tensor dA_dq[:,k,m]", [], func_params, None)
    self.gen_add_code_line("template <typename T>")
    self.gen_add_code_line("__device__")
    self.gen_add_code_line(func_def, True)
    # s_J holds the externalized Jw sweep band in smem (device wrappers don't spill).
    extra = [("s_A", 6 * nv), ("s_com", 3), ("s_extra", 4), ("s_J", _dccrba_sweep_J_count(self))]
    self.gen_XmatsHom_helpers_temp_shared_memory_code(
        _dccrba_inner_temp_mem_size(self), extra_t_buffers=extra, include_linalg_scratch=True,
        linalg_scratch_bytes="GRID_EE_LINALG_SHARED_BYTES<T>()")
    self.gen_load_update_XmatsHom_helpers_function_call()
    self.gen_add_code_line("dccrba_inner<T>(s_dccrba, s_A, s_com, s_extra, s_q, s_XmatsHom, d_robotModel, s_temp, s_J, s_linalg_smem);")
    self.gen_add_sync()
    self.gen_add_end_function()


# ----- cmm_time_variation kernel/host (no spill; reuse the generic kin helper) -----

def gen_cmm_time_variation_kernel(self, single_call_timing=False):
    n = self.robot.get_num_pos()
    nv = self.robot.get_num_vel()
    out_size = 6 * nv
    in_size = 2 * n
    sJ = _dccrba_sweep_J_count(self)
    func_def = ("void cmm_time_variation_kernel(T *d_out, unsigned char *d_workspace, const T *d_q_qd, const int stride_q_qd, "
                "const robotModel<T> *d_robotModel, const int NUM_TIMESTEPS) {")
    if single_call_timing:
        func_def = func_def.replace("kernel(", "kernel_single_timing(")
    self.gen_add_func_doc("Compute cmm_time_variation (Adot) per timestep", [], [], None)
    # MUJOCO_OUTPUT (floating only): compile-time mjx output-convention flag, LAST
    # after RESOURCE_TIER so existing positional <T,TIER> call sites are unaffected;
    # default false if-constexpr-elides the epilogues -> byte-identical pin PTX.
    self.gen_add_code_line("template <typename T, int RESOURCE_TIER = GRID_DEFAULT_RESOURCE_TIER, bool MUJOCO_OUTPUT = false>")
    self.gen_add_code_line("__global__")
    self.gen_add_code_line("__launch_bounds__(tier_max_threads<RESOURCE_TIER>())")
    self.gen_add_code_line(func_def, True)
    # DE-GATE #2: the Jw sweep band (6*nv*NB) is the LAST t_buffer; its smem slot is
    # sized sJ at TIER_SHARED/LITE (CMM_J_IN_SMEM<TIER>()==true) and 0 at the
    # J-spilled tier (then repointed to the L2-pinned d_workspace SO band below).
    self.gen_add_code_line("constexpr bool CMM_J_SMEM = CMM_J_IN_SMEM<RESOURCE_TIER>();")
    self.gen_add_code_line("constexpr int CMM_J_SLOT = CMM_J_SMEM ? " + str(sJ) + " : 0;")
    extra = [("s_q_qd", in_size), ("s_out", out_size), ("s_A", 6 * nv), ("s_com", 3), ("s_extra", 4), ("s_J", "CMM_J_SLOT")]
    self.gen_XmatsHom_helpers_temp_shared_memory_code(
        _dccrba_inner_temp_mem_size(self), extra_t_buffers=extra, include_linalg_scratch=True,
        linalg_scratch_bytes="GRID_EE_LINALG_SHARED_BYTES<T>()")
    self.gen_add_code_line("T *s_q = s_q_qd; T *s_qd = &s_q_qd[" + str(n) + "];")
    self.gen_add_code_line("if constexpr (CMM_J_SMEM) { (void)d_workspace; }")

    def _repoint(in_loop):
        self.gen_add_code_line("if constexpr (!CMM_J_SMEM) {", True)
        if in_loop:
            self.gen_add_code_line("s_J = reinterpret_cast<T *>(&d_workspace[k*GRID_WORKSPACE_BYTES_PER_TIMESTEP<T>() + GRID_DCCRBA_J_OFFSET_BYTES<T>()]);")
        else:
            self.gen_add_code_line("s_J = reinterpret_cast<T *>(&d_workspace[GRID_DCCRBA_J_OFFSET_BYTES<T>()]);")
        self.gen_add_end_control_flow()

    def _compute():
        # mjx INPUT convert (floating only): reorder the base quaternion wxyz->xyzw
        # (so XmatsHom builds X[0] correctly + the output R reads xyzw) and convert
        # the base-linear velocity to the pin frame (Adot = dA/dt depends on qd via
        # the contraction sum_m (dA/dq_m) qd_m, so qd must be pin-framed before the
        # inner). Emitted BEFORE the XmatsHom build.
        if self.robot.floating_base:
            self.gen_add_code_line("if constexpr (MUJOCO_OUTPUT) {", True)
            self.gen_mjx_input_convert(q_name="s_q", qd_name="s_qd")
            self.gen_add_end_control_flow()
        self.gen_load_update_XmatsHom_helpers_function_call()
        self.gen_add_code_line("cmm_time_variation_inner<T>(s_out, s_A, s_com, s_extra, s_q, s_qd, s_XmatsHom, d_robotModel, s_temp, s_J, s_linalg_smem);")
        self.gen_add_sync()
        # mjx OUTPUT: Adot (6 x NV col-major at s_out) column-reframes Adot . G^{-1}
        # (base-linear cols . R^T); hdot = Adot qd + A qddot is invariant (the qd
        # input was already pin-converted above so the contraction is correct). R
        # reads the already-reordered xyzw quaternion in s_q.
        if self.robot.floating_base:
            self.gen_add_code_line("if constexpr (MUJOCO_OUTPUT) {", True)
            self.gen_mjx_column_reframe("s_out", 6, nv)
            self.gen_add_end_control_flow()

    if not single_call_timing:
        self.gen_add_parallel_loop("k", "NUM_TIMESTEPS", block_level=True)
        self.gen_kernel_load_inputs("q_qd", str(in_size), stride="stride_q_qd")
        _repoint(in_loop=True)
        self.gen_add_code_line("// compute")
        _compute()
        self.gen_kernel_save_result("out", str(out_size), stride=str(out_size))
        self.gen_add_end_control_flow()
    else:
        self.gen_kernel_load_inputs("q_qd", str(in_size))
        _repoint(in_loop=False)
        self.gen_add_code_line("for (int rep = 0; rep < NUM_TIMESTEPS; rep++){", True)
        self.gen_anti_licm_input_reload("q_qd", str(in_size), feedback_from="out")
        _compute()
        self.gen_anti_licm_output_write("out")
        self.gen_add_end_control_flow()
        self.gen_kernel_save_result("out", str(out_size))
    self.gen_add_end_function()


def gen_cmm_time_variation_host(self, mode=0):
    n = self.robot.get_num_pos()
    nv = self.robot.get_num_vel()
    out_size = 6 * nv
    macro = "CMM_TIME_VARIATION_DYNAMIC_SHARED_MEM_BYTES<T, RESOURCE_TIER>()"
    single_call_timing = (mode == 1)
    compute_only = (mode == 2)
    func_def_start = ("void cmm_time_variation(gridData<T, KIND> *hd_data, const robotModel<T> *d_robotModel, "
                      "const int num_timesteps,")
    func_def_end = "                      const dim3 block_dimms, const dim3 thread_dimms, cudaStream_t *streams) {"
    if single_call_timing:
        func_def_start = func_def_start.replace("(", "_single_timing(")
        func_def_end = "              " + func_def_end
    if compute_only:
        func_def_start = func_def_start.replace("(", "_compute_only(")
        func_def_end = "             " + func_def_end.replace(", cudaStream_t *streams", "")
    self.gen_add_func_doc("Compute cmm_time_variation (Adot)", [], [], None)
    # MUJOCO_OUTPUT (floating only) host flag, LAST: forwarded to the kernel launch
    # naming the tier positionally (<T, GRID_DEFAULT_RESOURCE_TIER, MUJOCO_OUTPUT>) to
    # reach the trailing flag. Default false -> byte-identical pin codegen.
    mjx_host = self.robot.floating_base
    if mjx_host:
        self.gen_add_code_line("template <typename T, bool USE_COMPRESSED_MEM = false, gridDataKind KIND = GRID_DATA_ALL, bool MUJOCO_OUTPUT = false, int RESOURCE_TIER = GRID_DEFAULT_RESOURCE_TIER>")
    else:
        self.gen_add_code_line("template <typename T, bool USE_COMPRESSED_MEM = false, gridDataKind KIND = GRID_DATA_ALL, int RESOURCE_TIER = GRID_DEFAULT_RESOURCE_TIER>")
    self.gen_add_code_line("__host__")
    self.gen_add_code_line(func_def_start)
    self.gen_add_code_line(func_def_end, True)
    self.gen_add_code_line("static_assert(KIND == GRID_DATA_ALL || KIND == GRID_DATA_KINEMATICS, \"cmm_time_variation requires all-data or kinematics gridData\");")
    if mjx_host:
        ktmpl = "<T, RESOURCE_TIER, MUJOCO_OUTPUT>"
        kname = "cmm_time_variation_kernel" + ("_single_timing" if single_call_timing else "") + ktmpl
    else:
        kname = "cmm_time_variation_kernel" + ("_single_timing<T, RESOURCE_TIER>" if single_call_timing else "<T, RESOURCE_TIER>")
    func_call = (kname + "<<<block_dimms,thread_dimms," + macro + ">>>(hd_data->d_cmm_time_variation,hd_data->d_workspace,hd_data->d_q_qd,stride_q_qd,d_robotModel,num_timesteps);")
    if not compute_only:
        self.gen_add_code_lines([
            "// start code with memory transfer", "int stride_q_qd;",
            "if (USE_COMPRESSED_MEM) {stride_q_qd = 2*NUM_JOINTS; gpuErrchk(cudaMemcpyAsync(hd_data->d_q_qd,hd_data->h_q_qd,stride_q_qd*" +
            ("num_timesteps*" if not single_call_timing else "") + "sizeof(T),cudaMemcpyHostToDevice,streams[0]));}",
            "else {stride_q_qd = 3*NUM_JOINTS; gpuErrchk(cudaMemcpyAsync(hd_data->d_q_qd_u,hd_data->h_q_qd_u,stride_q_qd*" +
            ("num_timesteps*" if not single_call_timing else "") + "sizeof(T),cudaMemcpyHostToDevice,streams[0]));}"])
    else:
        self.gen_add_code_line("int stride_q_qd = USE_COMPRESSED_MEM ? 2*NUM_JOINTS : 3*NUM_JOINTS;")
    self.gen_add_code_line("// then call the kernel")
    func_call_mem = "if (USE_COMPRESSED_MEM) {" + func_call + "}"
    func_call_mem2 = "else                    {" + func_call.replace("hd_data->d_q_qd", "hd_data->d_q_qd_u") + "}"
    func_call_code = [func_call_mem, func_call_mem2, "gpuErrchkKernel();"]
    if single_call_timing:
        func_call_code.insert(0, "struct timespec start, end; clock_gettime(CLOCK_MONOTONIC,&start);")
        func_call_code.append("clock_gettime(CLOCK_MONOTONIC,&end);")
    # DE-GATE #2: L2-pin d_workspace when the default tier spills the Jw band into it.
    ws_bytes = ("GRID_WORKSPACE_BYTES_PER_TIMESTEP<T>()" if single_call_timing
                else "GRID_WORKSPACE_BYTES_PER_TIMESTEP<T>()*static_cast<size_t>(num_timesteps)")
    self.gen_add_code_line("if (!CMM_J_IN_SMEM<RESOURCE_TIER>() && hd_data->d_workspace != nullptr) {gpuErrchk(grid_begin_l2_persisting(0, hd_data->d_workspace, " + ws_bytes + "));}")
    self.gen_add_code_line("gpuErrchk(grid_check_dynamic_shared_memory_bytes(\"cmm_time_variation\", " + macro + "));")
    self.gen_add_code_lines(func_call_code)
    if not compute_only:
        self.gen_add_code_lines([
            "// finally transfer the result back",
            "gpuErrchk(cudaMemcpy(hd_data->h_cmm_time_variation,hd_data->d_cmm_time_variation," + str(out_size) + "*" +
            ("num_timesteps*" if not single_call_timing else "") + "sizeof(T),cudaMemcpyDeviceToHost));",
            "gpuErrchkKernel();"])
    if single_call_timing:
        from ..algo_registry import single_call_printf_line
        self.gen_add_code_line(single_call_printf_line("cmm_time_variation"))
    self.gen_add_end_function()


# ----- dccrba kernel/host (WITH spill: s_dccrba -> d_workspace SO band) -----

def gen_dccrba_kernel(self, single_call_timing=False):
    n = self.robot.get_num_pos()
    nv = self.robot.get_num_vel()
    out_size = 6 * nv * nv
    in_size = n
    func_def = ("void dccrba_kernel(T *d_dccrba, unsigned char *d_workspace, const T *d_q, const int stride_q, "
                "const robotModel<T> *d_robotModel, const int NUM_TIMESTEPS) {")
    if single_call_timing:
        func_def = func_def.replace("kernel(", "kernel_single_timing(")
    self.gen_add_func_doc("Compute the dCCRBA tensor dA_dq[:,k,m] per timestep", [], [], None)
    # MUJOCO_OUTPUT (floating only): compile-time mjx output-convention flag, LAST
    # after RESOURCE_TIER so existing positional <T,TIER> call sites are unaffected;
    # default false if-constexpr-elides the epilogues -> byte-identical pin PTX.
    self.gen_add_code_line("template <typename T, int RESOURCE_TIER = GRID_DEFAULT_RESOURCE_TIER, bool MUJOCO_OUTPUT = false>")
    self.gen_add_code_line("__global__")
    self.gen_add_code_line("__launch_bounds__(tier_max_threads<RESOURCE_TIER>())")
    self.gen_add_code_line(func_def, True)
    # spill: s_dccrba (output) and s_J (the Jw sweep band) are the last t_buffers;
    # each is sized in smem at the rungs that keep it (DCCRBA_OUTPUT_IN_SMEM /
    # DCCRBA_J_IN_SMEM) and 0 at the rung that spills it (then repointed to the
    # L2-pinned d_workspace SO band at distinct sub-offsets). L0: both in smem;
    # L1: output spills, s_J in smem; L2 (DE-GATE #2): both spill.
    sJ = _dccrba_sweep_J_count(self)
    self.gen_add_code_line("constexpr bool DCCRBA_OUT_IN_SMEM = DCCRBA_OUTPUT_IN_SMEM<RESOURCE_TIER>();")
    self.gen_add_code_line("constexpr int DCCRBA_OUT_SLOT = DCCRBA_OUT_IN_SMEM ? " + str(out_size) + " : 0;")
    self.gen_add_code_line("constexpr bool DCCRBA_J_SMEM = DCCRBA_J_IN_SMEM<RESOURCE_TIER>();")
    self.gen_add_code_line("constexpr int DCCRBA_J_SLOT = DCCRBA_J_SMEM ? " + str(sJ) + " : 0;")
    extra = [("s_q", in_size), ("s_A", 6 * nv), ("s_com", 3), ("s_extra", 4),
             ("s_dccrba", "DCCRBA_OUT_SLOT"), ("s_J", "DCCRBA_J_SLOT")]
    self.gen_XmatsHom_helpers_temp_shared_memory_code(
        _dccrba_inner_temp_mem_size(self), extra_t_buffers=extra, include_linalg_scratch=True,
        linalg_scratch_bytes="GRID_EE_LINALG_SHARED_BYTES<T>()")
    self.gen_add_code_line("if constexpr (DCCRBA_OUT_IN_SMEM && DCCRBA_J_SMEM) { (void)d_workspace; }")

    def _repoint(in_loop):
        if in_loop:
            base = "&d_workspace[k*GRID_WORKSPACE_BYTES_PER_TIMESTEP<T>() + "
        else:
            base = "&d_workspace["
        self.gen_add_code_line("if constexpr (!DCCRBA_OUT_IN_SMEM) {", True)
        self.gen_add_code_line("s_dccrba = reinterpret_cast<T *>(" + base + "GRID_SO_WORKSPACE_TEMP_OFFSET_BYTES<T>()]);")
        self.gen_add_end_control_flow()
        self.gen_add_code_line("if constexpr (!DCCRBA_J_SMEM) {", True)
        self.gen_add_code_line("s_J = reinterpret_cast<T *>(" + base + "GRID_DCCRBA_J_OFFSET_BYTES<T>()]);")
        self.gen_add_end_control_flow()

    if not single_call_timing:
        self.gen_add_parallel_loop("k", "NUM_TIMESTEPS", block_level=True)
        self.gen_kernel_load_inputs("q", str(in_size), stride="stride_q")
        _repoint(in_loop=True)
        self.gen_add_code_line("// compute")
        # mjx INPUT (floating only): dccrba is q-only and the CMM tensor is base-
        # orientation-independent in body frame, so only the base quaternion needs
        # reordering wxyz->xyzw (so XmatsHom builds X[0] correctly + the output
        # epilogue's R reads xyzw). Emitted BEFORE the XmatsHom build.
        if self.robot.floating_base:
            self.gen_add_code_line("if constexpr (MUJOCO_OUTPUT) {", True)
            self.gen_mjx_quat_reorder("s_q")
            self.gen_add_end_control_flow()
        self.gen_load_update_XmatsHom_helpers_function_call()
        self.gen_add_code_line("dccrba_inner<T>(s_dccrba, s_A, s_com, s_extra, s_q, s_XmatsHom, d_robotModel, s_temp, s_J, s_linalg_smem);")
        self.gen_add_sync()
        # mjx OUTPUT (floating only): transform the dccrba tensor (double G^{-1}
        # reframe + base-rotation frame term) in place on s_dccrba (whichever tier
        # buffer holds it). A_pin is read live from s_A (not overwritten). R is built
        # from the already-reordered xyzw quaternion in s_q.
        if self.robot.floating_base:
            self.gen_add_code_line("if constexpr (MUJOCO_OUTPUT) {", True)
            _emit_dccrba_mjx_output(self, "s_dccrba")
            self.gen_add_end_control_flow()
        self.gen_kernel_save_result("dccrba", str(out_size), stride=str(out_size))
        self.gen_add_end_control_flow()
    else:
        self.gen_kernel_load_inputs("q", str(in_size))
        _repoint(in_loop=False)
        self.gen_add_code_line("for (int rep = 0; rep < NUM_TIMESTEPS; rep++){", True)
        self.gen_load_update_XmatsHom_helpers_function_call()
        self.gen_add_code_line("dccrba_inner<T>(s_dccrba, s_A, s_com, s_extra, s_q, s_XmatsHom, d_robotModel, s_temp, s_J, s_linalg_smem);")
        self.gen_add_sync()
        self.gen_add_end_control_flow()
        self.gen_kernel_save_result("dccrba", str(out_size))
    self.gen_add_end_function()


def gen_dccrba_host(self, mode=0):
    n = self.robot.get_num_pos()
    nv = self.robot.get_num_vel()
    out_size = 6 * nv * nv
    macro = "DCCRBA_DYNAMIC_SHARED_MEM_BYTES<T, RESOURCE_TIER>()"
    single_call_timing = (mode == 1)
    compute_only = (mode == 2)
    func_def_start = ("void dccrba(gridData<T, KIND> *hd_data, const robotModel<T> *d_robotModel, "
                      "const int num_timesteps,")
    func_def_end = "                      const dim3 block_dimms, const dim3 thread_dimms, cudaStream_t *streams) {"
    if single_call_timing:
        func_def_start = func_def_start.replace("(", "_single_timing(")
        func_def_end = "              " + func_def_end
    if compute_only:
        func_def_start = func_def_start.replace("(", "_compute_only(")
        func_def_end = "             " + func_def_end.replace(", cudaStream_t *streams", "")
    self.gen_add_func_doc("Compute the dCCRBA tensor dA_dq[:,k,m]", [], [], None)
    # MUJOCO_OUTPUT (floating only) host flag, LAST: forwarded to the kernel launch
    # naming the tier positionally (<T, GRID_DEFAULT_RESOURCE_TIER, MUJOCO_OUTPUT>) to
    # reach the trailing flag. Default false -> byte-identical pin codegen.
    mjx_host = self.robot.floating_base
    if mjx_host:
        self.gen_add_code_line("template <typename T, bool USE_COMPRESSED_MEM = false, gridDataKind KIND = GRID_DATA_ALL, bool MUJOCO_OUTPUT = false, int RESOURCE_TIER = GRID_DEFAULT_RESOURCE_TIER>")
    else:
        self.gen_add_code_line("template <typename T, bool USE_COMPRESSED_MEM = false, gridDataKind KIND = GRID_DATA_ALL, int RESOURCE_TIER = GRID_DEFAULT_RESOURCE_TIER>")
    self.gen_add_code_line("__host__")
    self.gen_add_code_line(func_def_start)
    self.gen_add_code_line(func_def_end, True)
    self.gen_add_code_line("static_assert(KIND == GRID_DATA_ALL || KIND == GRID_DATA_KINEMATICS, \"dccrba requires all-data or kinematics gridData\");")
    # The non-timing kernel carries the MUJOCO_OUTPUT epilogue; name the tier
    # positionally so the trailing flag binds. The single-timing kernel takes the
    # flag too (host forwarding) but elides the epilogue (perf-phase follow-up).
    if mjx_host:
        ktmpl = ("_single_timing<T, RESOURCE_TIER, MUJOCO_OUTPUT>"
                 if single_call_timing else "<T, RESOURCE_TIER, MUJOCO_OUTPUT>")
        kname = "dccrba_kernel" + ktmpl
    else:
        kname = "dccrba_kernel" + ("_single_timing<T, RESOURCE_TIER>" if single_call_timing else "<T, RESOURCE_TIER>")
    func_call = (kname + "<<<block_dimms,thread_dimms," + macro + ">>>(hd_data->d_dccrba,hd_data->d_workspace,hd_data->d_q,stride_q,d_robotModel,num_timesteps);")
    if not compute_only:
        self.gen_add_code_lines([
            "// start code with memory transfer", "int stride_q = NUM_JOINTS;",
            "gpuErrchk(cudaMemcpyAsync(hd_data->d_q,hd_data->h_q,stride_q*" +
            ("num_timesteps*" if not single_call_timing else "") + "sizeof(T),cudaMemcpyHostToDevice,streams[0]));"])
    else:
        self.gen_add_code_line("int stride_q = NUM_JOINTS;")
    self.gen_add_code_line("// then call the kernel")
    # L2-pin d_workspace when the default tier spills s_dccrba into it.
    ws_bytes = ("GRID_WORKSPACE_BYTES_PER_TIMESTEP<T>()" if single_call_timing
                else "GRID_WORKSPACE_BYTES_PER_TIMESTEP<T>()*static_cast<size_t>(num_timesteps)")
    self.gen_add_code_line("if ((!DCCRBA_OUTPUT_IN_SMEM<RESOURCE_TIER>() || !DCCRBA_J_IN_SMEM<RESOURCE_TIER>()) && hd_data->d_workspace != nullptr) {gpuErrchk(grid_begin_l2_persisting(0, hd_data->d_workspace, " + ws_bytes + "));}")
    func_call_code = [func_call, "gpuErrchkKernel();"]
    if single_call_timing:
        func_call_code.insert(0, "struct timespec start, end; clock_gettime(CLOCK_MONOTONIC,&start);")
        func_call_code.append("clock_gettime(CLOCK_MONOTONIC,&end);")
    self.gen_add_code_line("gpuErrchk(grid_check_dynamic_shared_memory_bytes(\"dccrba\", " + macro + "));")
    self.gen_add_code_lines(func_call_code)
    if not compute_only:
        self.gen_add_code_lines([
            "// finally transfer the result back",
            "gpuErrchk(cudaMemcpy(hd_data->h_dccrba,hd_data->d_dccrba," + str(out_size) + "*" +
            ("num_timesteps*" if not single_call_timing else "") + "sizeof(T),cudaMemcpyDeviceToHost));",
            "gpuErrchkKernel();"])
    if single_call_timing:
        from ..algo_registry import single_call_printf_line
        self.gen_add_code_line(single_call_printf_line("dccrba"))
    self.gen_add_end_function()


# ----- top-level emit -----

def gen_cmm_time_variation(self):
    _cmm_time_variation_inner(self)
    gen_cmm_time_variation_device(self)
    gen_cmm_time_variation_kernel(self, single_call_timing=True)
    gen_cmm_time_variation_kernel(self, single_call_timing=False)
    gen_cmm_time_variation_host(self, 0)
    gen_cmm_time_variation_host(self, 1)
    gen_cmm_time_variation_host(self, 2)


def gen_dccrba(self):
    _dccrba_full_inner(self)
    gen_dccrba_device(self)
    gen_dccrba_kernel(self, single_call_timing=True)
    gen_dccrba_kernel(self, single_call_timing=False)
    gen_dccrba_host(self, 0)
    gen_dccrba_host(self, 1)
    gen_dccrba_host(self, 2)
