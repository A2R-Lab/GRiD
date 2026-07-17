"""Contact-FRAME -> joint-local f_ext map + derivatives (GATO ask 1, part C.2).

SSOT: docs/open-tasks/c2_contact_frame_map_design_2026-07-13.md

WHAT THIS CLOSES. GRiD already had the whole f_ext DERIVATIVE stack (_f_ext_gradient.py):
    dtau/dfext = -J^T,   dqdd/dfext = M^-1 J^T,   d(id_gradient)/dfext = -dJ^T/dq
...but all of it speaks the f_ext SLOT convention: a per-body wrench in that body's JOINT-LOCAL frame,
Featherstone-ordered [angular; linear]. A solver's decision variable is not that -- it is a contact force
at a DESIGNATED FRAME (foot, gripper) with WORLD-ALIGNED axes (the frame friction cones and gravity live
in). The conversion is CONFIGURATION-DEPENDENT, so it also contributes a term to d/dq that is easy to
drop. This module emits the map and both derivatives so the chain rule closes.

CONVENTION (= pinocchio LOCAL_WORLD_ALIGNED):
    f_c = [n_w ; f_w]   6-vec, [angular; linear] (Featherstone order), axes WORLD-ALIGNED,
                        moment taken about the CONTACT FRAME ORIGIN.
A pure linear point force (a foot pushing on the ground) is just n_w = 0 -- the 6D form SUBSUMES the 3D
case, so there is one entry point, not two.
★ Only the contact frame's OFFSET matters, never its ORIENTATION -- a consequence of world-aligned axes.

THE MAP. Frame c is rigidly attached to body b at BAKED local offset r_c. With R = R_b(q) the world
rotation of body b, g = R^T n_w, h = R^T f_w:

    f_ext[b] += [ g + r_c x h ;  h ]

(rotate both halves into the body frame, then TRANSPORT the moment from the contact origin to the joint
origin -- the r_c x h term).

DERIVATIVES.
  d(f_ext)/d(f_c): the map is LINEAR in f_c, so this is f_c-INDEPENDENT (a pure function of q):
        [ R^T   skew(r_c) R^T ]
        [  0          R^T     ]      6x6 for the (b, c) pair; zero for every other body.

  d(f_ext)/dq at FIXED f_c: q enters ONLY through R_b. With dR/dq_v = R [w_v]x, where w_v is the ANGULAR
  half of body b's body-Jacobian column v in b's LOCAL frame:
        d(f_ext[b])/dq_v = [ -w_v x g - r_c x (w_v x h) ;  -w_v x h ]

  ★★ w_v NEEDS NO NEW MACHINERY. `f_ext_gradient_jacobianT_inner` already builds exactly it:
        s_dtau_dfext[v + nv*(6*i + k)] == -(body i's LOCAL body-Jacobian column v)[k]
     so w_v = -s_dtau_dfext[v + nv*(6*b + k)] for k in 0..2. The dq emitter therefore takes s_dtau_dfext
     as an INPUT (the caller already computes it for the chain rule) rather than duplicating the chain-up.

COMPOSING (every factor already exists):
    dqdd/df_c    = (dqdd/dfext)[nv x 6NB] @ (dfext/df_c)[6NB x 6NC]
    dqdd/dq|f_c  = (the usual dqdd/dq) + (dqdd/dfext) @ (dfext/dq)   <-- the term you'd otherwise DROP
"""
import numpy as _np


def contact_frames_from_urdf(robot, names):
    """Resolve named URDF frames -> [{name, jid, offset[3]}] contact-frame specs.

    `names` are URDF *fixed*-joint names (a contact frame is the same kind of thing as a kinematic
    target: a foot frame, a gripper frame). Each resolves to the MOVING body it is rigidly attached to
    plus the frame origin in that body's joint frame -- exactly the baked r_c the map needs.

    ⚠ §1s: a fixed-joint jid has NO link, so the parent MUST be resolved through the FIXED-JOINT table
    (get_fixed_joint_by_name -> get_parent -> get_joint_by_name), never via get_link_by_id. Getting this
    wrong silently no-ops the chain-up on BRANCHED robots (go2!) and reads uninitialized shared memory.
    That bug already happened once this month; see debugging-guide §1s.

    A caller whose contact point has no URDF frame can hand-build the spec list; the emitters only ever
    read {jid, offset}.
    """
    specs = []
    for nm in names:
        fixed = robot.get_fixed_joint_by_name(nm)
        if fixed is None:
            raise ValueError(
                f"contact frame '{nm}' is not a fixed joint in this URDF. Pass an explicit "
                f"{{'name','jid','offset'}} spec if the contact point has no URDF frame.")
        parent_name = fixed.get_parent()
        parent = robot.get_joint_by_name(parent_name) if parent_name else None
        if parent is None:
            raise ValueError(f"contact frame '{nm}' hangs off the world/root, not a moving body -- "
                             f"its wrench would apply to no joint.")
        Xh = _np.asarray(fixed.get_transformation_matrix_hom(), dtype=_np.float64)
        specs.append({"name": nm, "jid": int(parent.get_id()),
                      "offset": [float(Xh[0, 3]), float(Xh[1, 3]), float(Xh[2, 3])]})
    return specs


def build_contact_set(self, contacts):
    """Normalize + bake the body grouping. Returns
        {"n", "jid"[n], "offset"[3n], "uniq"[m], "start"[m], "count"[m], "ids"[n]}

    Grouping BY BODY is load-bearing for DETERMINISM: it lets every (body, component) output slot be
    owned by exactly ONE thread that sums its body's contacts in a fixed baked order. The obvious
    alternative -- a thread per contact atomicAdd-ing into f_ext -- both races and makes the sum order
    warp-dependent, which is the exact non-determinism class Inc6 removed from the floating-base
    reductions. Two contacts on one body is normal (a foot and a shin), so this is not hypothetical.
    """
    def _snap(v):
        return [float(c) if abs(c) >= 1e-15 else 0.0 for c in v]
    n = len(contacts)
    jid = [int(c["jid"]) for c in contacts]
    offset = [v for c in contacts for v in _snap(c["offset"])]
    groups = {}
    for ci, b in enumerate(jid):
        groups.setdefault(b, []).append(ci)
    uniq, start, count, ids = [], [], [], []
    for b in sorted(groups):
        uniq.append(b)
        start.append(len(ids))
        count.append(len(groups[b]))
        ids.extend(groups[b])
    return {"n": n, "jid": jid, "offset": offset,
            "uniq": uniq, "start": start, "count": count, "ids": ids}


def gen_f_ext_contact_inner_temp_mem_size(self):
    """Helper scratch = the s_Xworld world-transform arena (16 per joint) -- IDENTICAL to the
    multi_target position inner, because it is the same shared FK prefix."""
    from ._eepose_gradient_hessian import _eepose_xworld_slot_count
    return 16 * _eepose_xworld_slot_count(self)


def _emit_contact_tables(self, cs):
    n, m = cs["n"], len(cs["uniq"])
    self.gen_add_code_lines([
        "// baked contact set: body id + LOCAL frame origin offset per contact",
        "static const int fc_body[" + str(n) + "] = {" + ", ".join(str(v) for v in cs["jid"]) + "};",
        "const T fc_offset[" + str(3 * n) + "] = {" +
        ", ".join("static_cast<T>({:.17g})".format(v) for v in cs["offset"]) + "};",
        "// body-grouped: one writer per output slot, fixed-order sums, NO atomics (determinism)",
        "static const int fc_uniq[" + str(m) + "] = {" + ", ".join(str(v) for v in cs["uniq"]) + "};",
        "static const int fc_start[" + str(m) + "] = {" + ", ".join(str(v) for v in cs["start"]) + "};",
        "static const int fc_count[" + str(m) + "] = {" + ", ".join(str(v) for v in cs["count"]) + "};",
        "static const int fc_ids[" + str(n) + "] = {" + ", ".join(str(v) for v in cs["ids"]) + "};",
    ])


def _emit_gh(self, fc_expr="s_f_c"):
    """Emit `g` and `h` (= R^T n_w and R^T f_w) for contact `c` on body `b`, from s_Xworld.

    R(row,col) = s_Xworld[16*b + 4*col + row]  (4x4 COLUMN-major), so R^T v contracts over the ROW
    index: (R^T v)[a] = sum_r R[r][a] v[r] = sum_r s_Xworld[16b + 4a + r] * v[r].
    """
    self.gen_add_code_lines([
        "const T *nw = &" + fc_expr + "[6*c];       // world angular (moment about the CONTACT origin)",
        "const T *fw = &" + fc_expr + "[6*c + 3];   // world linear",
        "T g[3], h[3];",
        "for (int a = 0; a < 3; ++a) {", True,
        "const T *Ra = &s_Xworld[16*b + 4*a];   // column a of R  ->  row a of R^T",
        "g[a] = Ra[0]*nw[0] + Ra[1]*nw[1] + Ra[2]*nw[2];",
        "h[a] = Ra[0]*fw[0] + Ra[1]*fw[1] + Ra[2]*fw[2];",
    ])
    self.gen_add_end_control_flow()
    self.gen_add_code_lines([
        "const T r0 = fc_offset[3*c+0], r1 = fc_offset[3*c+1], r2 = fc_offset[3*c+2];",
    ])


def gen_f_ext_body_inner(self, contacts):
    """Emit `f_ext_body_inner<T, TEMP_IN_SMEM>`: contact wrenches -> joint-local f_ext (6*NUM_BODIES)."""
    cs = build_contact_set(self, contacts)
    NB = self.robot.get_num_bodies()
    m = len(cs["uniq"])
    n_joints = self.robot.get_num_joints()

    func_params = [
        "s_f_ext is the output, size 6*NUM_BODIES (joint-local Featherstone [angular;linear]); ZEROED then scattered",
        "s_f_c is the contact wrench input, size 6*NUM_CONTACT_FRAMES = " + str(6 * cs["n"]) +
        " ([n_w; f_w] per frame, WORLD-ALIGNED axes, moment about the contact origin)",
        "s_q is the vector of joint positions",
        "s_Xhom is the per-joint local homogeneous transforms (already updated for q)",
        "s_temp is helper shared memory (holds s_Xworld = 16*NUM_JOINTS)",
        "d_workspace is the global-memory scratch used when !TEMP_IN_SMEM",
    ]
    func_notes = [
        "f_ext[b] += [ R^T n_w + r_c x (R^T f_w) ; R^T f_w ] -- rotate into the body frame, then "
        "TRANSPORT the moment from the contact origin to the joint origin.",
        "Output is fully zeroed first, so bodies with no contact stay 0 and the array can be handed "
        "straight to grid_plant::plant_step(..., d_f_ext).",
        "A pure point force is n_w = 0; the 6D form subsumes the 3D case.",
    ]
    func_def_start = "void f_ext_body_inner("
    func_def_middle = "T *s_f_ext, const T *s_f_c, const T *s_q, const T *s_Xhom, "
    func_def_end = "T *s_temp, T *d_workspace, unsigned char *s_linalg_smem) {"
    func_def_middle, func_params = self.gen_insert_helpers_func_def_params(
        func_def_middle, func_params, -1, NO_XI_FLAG=True)
    self.gen_add_func_doc("Contact-frame wrenches -> joint-local f_ext", func_notes, func_params, None)
    self.gen_add_code_line("template <typename T, bool TEMP_IN_SMEM = true>")
    self.gen_add_code_line("__device__")
    self.gen_add_code_line(func_def_start + func_def_middle + func_def_end, True)
    self.gen_add_code_line("if constexpr (!TEMP_IN_SMEM) { s_temp = d_workspace; } else { (void)d_workspace; }")
    self.gen_add_code_line("(void)s_q; (void)s_linalg_smem;")
    self.gen_add_code_line("T *s_Xworld = s_temp;   // 16 * " + str(n_joints))

    from ._eepose_gradient_hessian import emit_world_fk_chainup
    emit_world_fk_chainup(
        self,
        header_lines=["//", "// Build world transforms for every joint via BFS-level chain-up", "//"],
        fixed_anchors=None)

    _emit_contact_tables(self, cs)

    # zero the WHOLE output first: bodies with no contact must read 0.
    self.gen_add_code_line("// zero every slot: bodies with no contact contribute nothing")
    self.gen_add_parallel_loop("ind", str(6 * NB))
    self.gen_add_code_line("s_f_ext[ind] = static_cast<T>(0);")
    self.gen_add_end_control_flow()
    self.gen_add_sync()

    # one thread per (contacted body, component) -- exactly one writer per slot, fixed-order sum.
    self.gen_add_code_line("// one thread per (contacted body, component): single writer, fixed-order sum")
    self.gen_add_parallel_loop("ind", str(6 * m))
    self.gen_add_code_lines([
        "int k = ind % 6; int u = ind / 6;",
        "const int b = fc_uniq[u];",
        "T acc = static_cast<T>(0);",
        "for (int t = 0; t < fc_count[u]; ++t) {", True,
        "const int c = fc_ids[fc_start[u] + t];",
    ])
    _emit_gh(self)
    self.gen_add_code_lines([
        "if (k >= 3) { acc += h[k-3]; }",
        "else {", True,
        "// angular = g + r_c x h",
        "const T rxh0 = r1*h[2] - r2*h[1];",
        "const T rxh1 = r2*h[0] - r0*h[2];",
        "const T rxh2 = r0*h[1] - r1*h[0];",
        "acc += g[k] + ((k == 0) ? rxh0 : ((k == 1) ? rxh1 : rxh2));",
    ])
    self.gen_add_end_control_flow()   # else
    self.gen_add_end_control_flow()   # for t
    self.gen_add_code_line("s_f_ext[6*b + k] = acc;")
    self.gen_add_end_control_flow()   # parallel loop
    self.gen_add_sync()
    self.gen_add_end_function()


def gen_f_ext_body_jacobian_dq_inner(self, contacts):
    """Emit `f_ext_body_jacobian_dq_inner`: d(f_ext)/dq at FIXED f_c, size 6*NUM_BODIES x NUM_VEL.

    Takes s_dtau_dfext (= -J^T, from grid::f_ext_gradient_device) as an INPUT rather than rebuilding the
    body-Jacobian chain: its columns ARE the local body Jacobian, and the caller already computes it for
    the chain rule. w_v = -s_dtau_dfext[v + nv*(6*b + k)], k in 0..2.
    """
    cs = build_contact_set(self, contacts)
    nv = self.robot.get_num_vel()
    NB = self.robot.get_num_bodies()
    m = len(cs["uniq"])
    n_joints = self.robot.get_num_joints()

    func_params = [
        "s_dfext_dq is the output, size 6*NUM_BODIES*NUM_VEL, column-major [row + 6*NUM_BODIES*v]",
        "s_f_c is the contact wrench input, size 6*NUM_CONTACT_FRAMES (the Jacobian is LINEAR in it)",
        "s_dtau_dfext is -J^T from grid::f_ext_gradient_device, size NUM_VEL*6*NUM_BODIES "
        "(column-major [v + NUM_VEL*(6*i+k)]); its columns ARE the local body Jacobian",
        "s_q is the vector of joint positions",
        "s_Xhom is the per-joint local homogeneous transforms (already updated for q)",
        "s_temp is helper shared memory (holds s_Xworld = 16*NUM_JOINTS)",
        "d_workspace is the global-memory scratch used when !TEMP_IN_SMEM",
    ]
    func_notes = [
        "d(f_ext[b])/dq_v = [ -w_v x g - r_c x (w_v x h) ; -w_v x h ], with w_v the ANGULAR half of "
        "body b's LOCAL body-Jacobian column v (= -s_dtau_dfext[v + nv*(6b+k)], k in 0..2).",
        "This is the chain-rule term a solver DROPS if it treats the applied wrench as q-independent.",
    ]
    func_def_start = "void f_ext_body_jacobian_dq_inner("
    func_def_middle = ("T *s_dfext_dq, const T *s_f_c, const T *s_dtau_dfext, const T *s_q, const T *s_Xhom, ")
    func_def_end = "T *s_temp, T *d_workspace, unsigned char *s_linalg_smem) {"
    func_def_middle, func_params = self.gen_insert_helpers_func_def_params(
        func_def_middle, func_params, -1, NO_XI_FLAG=True)
    self.gen_add_func_doc("d(f_ext)/dq at fixed contact wrench", func_notes, func_params, None)
    self.gen_add_code_line("template <typename T, bool TEMP_IN_SMEM = true>")
    self.gen_add_code_line("__device__")
    self.gen_add_code_line(func_def_start + func_def_middle + func_def_end, True)
    self.gen_add_code_line("if constexpr (!TEMP_IN_SMEM) { s_temp = d_workspace; } else { (void)d_workspace; }")
    self.gen_add_code_line("(void)s_q; (void)s_linalg_smem;")
    self.gen_add_code_line("T *s_Xworld = s_temp;   // 16 * " + str(n_joints))

    from ._eepose_gradient_hessian import emit_world_fk_chainup
    emit_world_fk_chainup(
        self,
        header_lines=["//", "// Build world transforms for every joint via BFS-level chain-up", "//"],
        fixed_anchors=None)

    _emit_contact_tables(self, cs)

    self.gen_add_code_line("// zero every slot: bodies with no contact have zero sensitivity")
    self.gen_add_parallel_loop("ind", str(6 * NB * nv))
    self.gen_add_code_line("s_dfext_dq[ind] = static_cast<T>(0);")
    self.gen_add_end_control_flow()
    self.gen_add_sync()

    self.gen_add_code_line("// one thread per (contacted body, component, velocity): single writer")
    self.gen_add_parallel_loop("ind", str(6 * m * nv))
    self.gen_add_code_lines([
        "int k = ind % 6; int rest = ind / 6; int u = rest % " + str(m) + "; int v = rest / " + str(m) + ";",
        "const int b = fc_uniq[u];",
        "// w_v = angular half of body b's LOCAL body-Jacobian column v = -s_dtau_dfext[v + nv*(6b+kk)]",
        "const T w0 = -s_dtau_dfext[v + " + str(nv) + "*(6*b + 0)];",
        "const T w1 = -s_dtau_dfext[v + " + str(nv) + "*(6*b + 1)];",
        "const T w2 = -s_dtau_dfext[v + " + str(nv) + "*(6*b + 2)];",
        "T acc = static_cast<T>(0);",
        "for (int t = 0; t < fc_count[u]; ++t) {", True,
        "const int c = fc_ids[fc_start[u] + t];",
    ])
    _emit_gh(self)
    self.gen_add_code_lines([
        "// dh = -w x h ;  dg = -w x g",
        "const T dh0 = -(w1*h[2] - w2*h[1]), dh1 = -(w2*h[0] - w0*h[2]), dh2 = -(w0*h[1] - w1*h[0]);",
        "if (k >= 3) { acc += (k == 3) ? dh0 : ((k == 4) ? dh1 : dh2); }",
        "else {", True,
        "const T dg0 = -(w1*g[2] - w2*g[1]), dg1 = -(w2*g[0] - w0*g[2]), dg2 = -(w0*g[1] - w1*g[0]);",
        "// angular sensitivity = dg + r_c x dh",
        "const T rxd0 = r1*dh2 - r2*dh1;",
        "const T rxd1 = r2*dh0 - r0*dh2;",
        "const T rxd2 = r0*dh1 - r1*dh0;",
        "acc += ((k == 0) ? (dg0 + rxd0) : ((k == 1) ? (dg1 + rxd1) : (dg2 + rxd2)));",
    ])
    self.gen_add_end_control_flow()   # else
    self.gen_add_end_control_flow()   # for t
    self.gen_add_code_line("s_dfext_dq[(6*b + k) + " + str(6 * NB) + "*v] = acc;")
    self.gen_add_end_control_flow()   # parallel loop
    self.gen_add_sync()
    self.gen_add_end_function()


# ---------------------------------------------------------------------------
# Device wrappers: own the tier-aware smem arena + build s_XmatsHom, then call the inner.
# Modeled verbatim on gen_multi_target_position_device (same FK, same arena shape) -- deliberately
# NOT hand-rolled: this arena/tier surface is where both §1s and §1t lived.
# ---------------------------------------------------------------------------
def _emit_device(self, contacts, which):
    """which in {"value", "dq", "dfc"}. Each owns the tier-aware smem arena + s_XmatsHom, then calls
    its inner -- modeled verbatim on gen_multi_target_position_device (same FK, same arena shape)."""
    scratch = gen_f_ext_contact_inner_temp_mem_size(self)
    nv = self.robot.get_num_vel()
    NB = self.robot.get_num_bodies()
    NC = len(contacts)
    NAME = {"value": "f_ext_body_device",
            "dq":    "f_ext_body_jacobian_dq_device",
            "dfc":   "f_ext_body_jacobian_dfc_device"}[which]
    OUT = {"value": ("T *s_f_ext, ",     "s_f_ext is the output joint-local wrench array, size " + str(6 * NB)),
           "dq":    ("T *s_dfext_dq, ",  "s_dfext_dq is the output d(f_ext)/dq, size " + str(6 * NB * nv) +
                                         " (column-major [row + " + str(6 * NB) + "*v])"),
           "dfc":   ("T *s_dfext_dfc, ", "s_dfext_dfc is the output d(f_ext)/d(f_c), size " + str(6 * NB * 6 * NC) +
                                         " (column-major [row + " + str(6 * NB) + "*(6*c+j)])")}[which]
    func_params = [OUT[1]]
    if which != "dfc":   # the dfc block is f_c-INDEPENDENT -> takes no wrench
        func_params.append("s_f_c is the contact wrench input, size " + str(6 * NC) +
                           " ([n_w; f_w] per frame, WORLD-ALIGNED, moment about the contact origin)")
    if which == "dq":
        func_params.append("s_dtau_dfext is -J^T from grid::f_ext_gradient_device, size " + str(nv * 6 * NB))
    func_params += [
        "s_q is the vector of joint positions",
        "d_robotModel is the initialized model-specific helpers on the GPU",
        "d_workspace is the global scratch (= 0 bytes at TIER_SHARED, " + str(scratch) +
        "*sizeof(T) at TIER_LITE+); pass nullptr at TIER_SHARED"]
    doc = {"value": "Contact-frame wrenches -> the joint-local f_ext array plant_step consumes",
           "dq":    "d(f_ext)/dq at fixed contact wrench (the chain-rule term solvers drop)",
           "dfc":   "d(f_ext)/d(contact wrench) -- compose with dqdd/dfext to get dqdd/df_c"}[which]
    self.gen_add_func_doc(doc, [], func_params, None)
    self.gen_add_code_line("template <typename T, int RESOURCE_TIER = GRID_DEFAULT_RESOURCE_TIER>")
    self.gen_add_code_line("__device__")
    sig = "void " + NAME + "(" + OUT[0]
    if which != "dfc":
        sig += "const T *s_f_c, "
    if which == "dq":
        sig += "const T *s_dtau_dfext, "
    sig += "const T *s_q, const robotModel<T> *d_robotModel, T *d_workspace = nullptr) {"
    self.gen_add_code_line(sig, True)
    self.gen_XmatsHom_helpers_temp_shared_memory_code(scratch, include_linalg_scratch=True,
                                                      linalg_scratch_bytes="GRID_EE_LINALG_SHARED_BYTES<T>()",
                                                      tier_workspace_expr="d_workspace")
    self.gen_load_update_XmatsHom_helpers_function_call()
    inner = {"value": "f_ext_body_inner<T>(s_f_ext, s_f_c, s_q, s_XmatsHom, ",
             "dq":    "f_ext_body_jacobian_dq_inner<T>(s_dfext_dq, s_f_c, s_dtau_dfext, s_q, s_XmatsHom, ",
             "dfc":   "f_ext_body_jacobian_dfc_inner<T>(s_dfext_dfc, s_q, s_XmatsHom, "}[which]
    inner += self.gen_insert_helpers_function_call(NO_XI_FLAG=True)
    inner += "s_temp, nullptr, s_linalg_smem);"
    self.gen_add_code_line(inner)
    self.gen_add_end_function()


def gen_f_ext_body_jacobian_dfc_inner(self, contacts):
    """Emit `f_ext_body_jacobian_dfc_inner`: d(f_ext)/d(f_c), size 6*NUM_BODIES x 6*NUM_CONTACT_FRAMES.

    THE quantity a solver differentiates its decision variable through:
        dqdd/df_c = (dqdd/dfext)[nv x 6NB] @ (dfext/df_c)[6NB x 6NC]
    The map is LINEAR in f_c, so this block is f_c-INDEPENDENT -- a pure function of q. Per (body b,
    contact c) pair it is the 6x6

        [ R^T   skew(r_c) R^T ]
        [  0          R^T     ]

    and ZERO for every other body. Column-major [row + 6*NUM_BODIES*col], col = 6*c + j.
    """
    cs = build_contact_set(self, contacts)
    NB = self.robot.get_num_bodies()
    n = cs["n"]
    n_joints = self.robot.get_num_joints()
    NR = 6 * NB

    func_params = [
        "s_dfext_dfc is the output, size " + str(NR * 6 * n) + " (column-major [row + " + str(NR) + "*col], col = 6*c + j)",
        "s_q is the vector of joint positions",
        "s_Xhom is the per-joint local homogeneous transforms (already updated for q)",
        "s_temp is helper shared memory (holds s_Xworld = 16*NUM_JOINTS)",
        "d_workspace is the global-memory scratch used when !TEMP_IN_SMEM",
    ]
    func_notes = [
        "f_c-INDEPENDENT by construction (the map is linear in f_c) -- takes no f_c argument.",
        "Compose with grid::f_ext_gradient_device's dqdd/dfext to get dqdd/df_c.",
    ]
    func_def_start = "void f_ext_body_jacobian_dfc_inner("
    func_def_middle = "T *s_dfext_dfc, const T *s_q, const T *s_Xhom, "
    func_def_end = "T *s_temp, T *d_workspace, unsigned char *s_linalg_smem) {"
    func_def_middle, func_params = self.gen_insert_helpers_func_def_params(
        func_def_middle, func_params, -1, NO_XI_FLAG=True)
    self.gen_add_func_doc("d(f_ext)/d(contact wrench) -- f_c-independent", func_notes, func_params, None)
    self.gen_add_code_line("template <typename T, bool TEMP_IN_SMEM = true>")
    self.gen_add_code_line("__device__")
    self.gen_add_code_line(func_def_start + func_def_middle + func_def_end, True)
    self.gen_add_code_line("if constexpr (!TEMP_IN_SMEM) { s_temp = d_workspace; } else { (void)d_workspace; }")
    self.gen_add_code_line("(void)s_q; (void)s_linalg_smem;")
    self.gen_add_code_line("T *s_Xworld = s_temp;   // 16 * " + str(n_joints))

    from ._eepose_gradient_hessian import emit_world_fk_chainup
    emit_world_fk_chainup(
        self,
        header_lines=["//", "// Build world transforms for every joint via BFS-level chain-up", "//"],
        fixed_anchors=None)

    _emit_contact_tables(self, cs)

    self.gen_add_code_line("// zero: only the (body of contact c) rows are nonzero for column block c")
    self.gen_add_parallel_loop("ind", str(NR * 6 * n))
    self.gen_add_code_line("s_dfext_dfc[ind] = static_cast<T>(0);")
    self.gen_add_end_control_flow()
    self.gen_add_sync()

    # one thread per (contact, output-row k, input-col j): every slot has a single writer.
    self.gen_add_code_line("// one thread per (contact, out-component k, in-component j)")
    self.gen_add_parallel_loop("ind", str(36 * n))
    self.gen_add_code_lines([
        "int j = ind % 6; int rest = ind / 6; int k = rest % 6; int c = rest / 6;",
        "const int b = fc_body[c];",
        "const T r0 = fc_offset[3*c+0], r1 = fc_offset[3*c+1], r2 = fc_offset[3*c+2];",
        "// Rt(a, r) = R^T[a][r] = R[r][a] = s_Xworld[16*b + 4*a + r]",
        "T val = static_cast<T>(0);",
        "if (k >= 3) {", True,
        "// linear rows: [ 0 | R^T ] -> only the linear half of f_c contributes",
        "if (j >= 3) { val = s_Xworld[16*b + 4*(k-3) + (j-3)]; }",
    ])
    self.gen_add_end_control_flow()
    self.gen_add_code_lines([
        "else {", True,
        "// angular rows: [ R^T | skew(r_c) R^T ]",
        "if (j < 3) { val = s_Xworld[16*b + 4*k + j]; }",
        "else {", True,
        "// (skew(r) R^T)[k][j-3] = sum_a skew(r)[k][a] * R^T[a][j-3]",
        "const int jj = j - 3;",
        "const T Rt0 = s_Xworld[16*b + 4*0 + jj];",
        "const T Rt1 = s_Xworld[16*b + 4*1 + jj];",
        "const T Rt2 = s_Xworld[16*b + 4*2 + jj];",
        "// skew(r) rows: [0,-r2,r1], [r2,0,-r0], [-r1,r0,0]",
        "val = (k == 0) ? (-r2*Rt1 + r1*Rt2)",
        "    : ((k == 1) ? ( r2*Rt0 - r0*Rt2)",
        "                : (-r1*Rt0 + r0*Rt1));",
    ])
    self.gen_add_end_control_flow()   # else j>=3
    self.gen_add_end_control_flow()   # else k<3
    self.gen_add_code_line("s_dfext_dfc[(6*b + k) + " + str(NR) + "*(6*c + j)] = val;")
    self.gen_add_end_control_flow()   # parallel loop
    self.gen_add_sync()
    self.gen_add_end_function()


def gen_f_ext_contact(self, contacts):
    """Dispatcher: emit the whole contact-frame f_ext surface (constants + inners + devices)."""
    if not contacts:
        return
    cs = build_contact_set(self, contacts)
    self.gen_add_code_line("")
    self.gen_add_code_line("// ---- contact frames (GATO ask 1 C.2): world-aligned contact wrench -> joint-local f_ext")
    for c in contacts:
        self.gen_add_code_line("//   '" + c["name"] + "' -> body " + str(c["jid"]) +
                               " @ local offset (" + ", ".join("%.6g" % v for v in c["offset"]) + ")")
    self.gen_add_code_line("#define GRID_HAS_CONTACT_FRAMES 1")
    self.gen_add_code_line("const int NUM_CONTACT_FRAMES = " + str(cs["n"]) + ";")
    gen_f_ext_body_inner(self, contacts)
    gen_f_ext_body_jacobian_dfc_inner(self, contacts)
    gen_f_ext_body_jacobian_dq_inner(self, contacts)
    _emit_device(self, contacts, "value")
    _emit_device(self, contacts, "dfc")
    _emit_device(self, contacts, "dq")
