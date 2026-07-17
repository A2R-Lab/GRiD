import numpy as np
import sympy as sp

def gen_get_XI_size(self, include_base_inertia = False, include_homogenous_transforms = False):
    n = self.robot.get_num_joints()
    Xhom_size, dXhom_size, d2Xhom_size = self.gen_get_Xhom_size()
    base_size = 36*2*n + (36 if include_base_inertia else 0)
    return base_size + (Xhom_size + dXhom_size + d2Xhom_size if include_homogenous_transforms else 0)

def gen_get_Xhom_size(self):
    n = self.robot.get_num_pos()
    NJ = self.robot.get_num_joints()
    nfj = self.robot.get_num_fixed_joints() if self.include_fixed_kinematic_targets else 0
    Xhom_size = 16*(NJ+nfj) # one homogeneous transform per joint plus optional fixed kinematic targets
    # The fixed-base dXhom/d2Xhom are stored PER-BODY (one 4x4 per joint id):
    # both gen_init_XImats (host) and gen_load_update_XImats_helpers (device)
    # write len(get_d2Xmats_hom_ordered_by_id()) == NUM_BODIES matrices. Budget
    # by NUM_BODIES so a mimic robot (NB > nq) doesn't overflow h_XImats. For
    # non-mimic robots NB == nq, so this is byte-identical to the legacy 16*n.
    NB = self.robot.get_num_bodies()
    body_count = NB if not self.robot.floating_base else n
    dXhom_size = 16*body_count # kinematic targets are fixed so don't include (gradient is 0)
    d2Xhom_size = 16*(n*n if self.robot.floating_base else NB) # floating root has dense local quaternion second derivatives
    return Xhom_size, dXhom_size, d2Xhom_size

def _qinds_to_list(self, inds):
    if isinstance(inds, (list, tuple, np.ndarray)):
        return list(inds)
    return [inds]

def _global_hom_derivative_matrices_by_q(self):
    n = self.robot.get_num_pos()
    mats = [sp.zeros(4, 4) for _ in range(n)]
    owners = [None for _ in range(n)]
    for jid in range(self.robot.get_num_joints()):
        qinds = _qinds_to_list(self, self.robot.get_joint_index_q(jid))
        for local_ind, qind in enumerate(qinds):
            mats[qind] = self.robot.get_dXmat_hom_local_by_id(jid, local_ind)
            owners[qind] = jid
    return mats, owners

def _global_hom_second_derivative_matrices(self):
    n = self.robot.get_num_pos()
    if not self.robot.floating_base:
        # mats has one entry per JOINT (length NJ); owners maps each entry to
        # its joint's q-index. NJ != n_pos when fixed sub-joints are present
        # (e.g. h1_2 has 51 joints but 39 DoF positions); the prior `list(
        # range(n))` form length-mismatched, raising IndexError on the
        # zero-d2Xhom fixed-joint entries. The None sentinel for non-DoF
        # joints is handled by the consumer's `owner_jid if owner_jid is not
        # None else ind` fallback (the d2Xhom for fixed joints is the zero
        # matrix, so the inner code is dead anyway — owner is unread).
        mats = self.robot.get_d2Xmats_hom_ordered_by_id()
        owners = [None] * len(mats)
        for jid in range(self.robot.get_num_joints()):
            qinds = _qinds_to_list(self, self.robot.get_joint_index_q(jid))
            if qinds:
                owners[jid] = qinds[0]
        return mats, owners

    mats = [sp.zeros(4, 4) for _ in range(n*n)]
    owners = [None for _ in range(n*n)]
    for jid in range(self.robot.get_num_joints()):
        qinds = _qinds_to_list(self, self.robot.get_joint_index_q(jid))
        for local_i, qind_i in enumerate(qinds):
            for local_j, qind_j in enumerate(qinds):
                pair_ind = qind_i*n + qind_j
                mats[pair_ind] = self.robot.get_d2Xmat_hom_local_by_id(jid, local_i, local_j)
                owners[pair_ind] = jid
    return mats, owners

def custom_is_constant(self, val):
    # Memoize per codegen instance. sympy `is_constant()` is expensive (it
    # runs simplify → cancel → factor_terms internally) and gets called on
    # every cell of every Xmat / Xhom / dXhom / d2Xhom matrix — many cells
    # are duplicate expressions (literal 0, 1, sin(q_k), etc.) so caching
    # collapses tens of thousands of calls to a few hundred unique ones.
    # On g1_floating: dropped ~22min codegen by ~Nx (see Phase 7a profile).
    if not hasattr(val, 'is_constant'):
        return isinstance(val, (int, float, complex, np.number))
    cache = getattr(self, '_is_constant_cache', None)
    if cache is None:
        cache = {}
        self._is_constant_cache = cache
    try:
        if val in cache:
            return cache[val]
        result = val.is_constant()
        cache[val] = result
        return result
    except TypeError:
        # Unhashable expression — fall back to uncached call.
        return val.is_constant()

def gen_init_XImats(self, include_base_inertia = False, include_homogenous_transforms = False):
    # add function description
    if include_base_inertia:
        desc = "Memory order is X[0...N], Ibase, I[0...N]"
    else:
        desc = "Memory order is X[0...N], I[0...N]"
    if include_homogenous_transforms:
        desc += ", Xhom[0...N]"
    self.gen_add_func_doc("Initializes the Xmats and Imats in GPU memory", \
            [desc], \
            [],"A pointer to the XI memory in the GPU")
    # add the function start boilerplate
    self.gen_add_code_line("template <typename T>")
    self.gen_add_code_line("__host__")
    self.gen_add_code_line("T* init_XImats() {", True)
    # allocate CPU memory
    n = self.robot.get_num_pos()
    XI_size = self.gen_get_XI_size(include_base_inertia,include_homogenous_transforms)
    baseXI_size = self.gen_get_XI_size(include_base_inertia,include_homogenous_transforms = False) #just base XI_size to know where Xhom starts in XI (if needed)
    self.gen_add_code_line("T *h_XImats = (T *)calloc(" + str(XI_size) + ",sizeof(T));")
    # loop through Xmats and add all constant values from the sp matrix (initialize non-constant to 0)
    Xmats = self.robot.get_Xmats_ordered_by_id()
    for ind in range(len(Xmats)):
        self.gen_add_code_line("// X[" + str(ind) + "]")
        for col in range(6):
            for row in range(6):
                val = Xmats[ind][row,col]
                if not self.custom_is_constant(val): # initialize to 0
                    val = 0
                str_val = str(val)
                cpp_ind = self.gen_static_array_ind_3d(ind,col,row)
                self.gen_add_code_line("h_XImats[" + str(cpp_ind) + "] = static_cast<T>(" + str_val + ");")
    # loop through Imats and add all values (inertias are always constant and stored as np arrays)
    Imats = self.robot.get_Imats_ordered_by_id()
    if not include_base_inertia:
        Imats = Imats[1:]
    mem_offset = len(Xmats)
    for ind in range(len(Imats)):
        if include_base_inertia and ind == 0:
            self.gen_add_code_line("// Base Inertia")
        else:
            self.gen_add_code_line("// I[" + str(ind-int(include_base_inertia)) + "]")
        for col in range(6):
            for row in range(6):
                str_val = str(Imats[ind][row,col])
                cpp_ind = str(self.gen_static_array_ind_3d(ind + mem_offset,col,row))
                self.gen_add_code_line("h_XImats[" + cpp_ind + "] = static_cast<T>(" + str_val + ");")
    # add the X_hom if asked (follow the method from Xmats)
    if (include_homogenous_transforms):
        Xmats_hom = self.robot.get_Xmats_hom_ordered_by_id(include_fixed_joints = self.include_fixed_kinematic_targets)
        generated_algorithms = getattr(self, "generated_algorithms", set())
        include_hom_gradients = ("end_effector_pose_gradient" in generated_algorithms) or ("end_effector_pose_hessian" in generated_algorithms)
        include_hom_hessians = "end_effector_pose_hessian" in generated_algorithms
        dXmats_hom, _ = _global_hom_derivative_matrices_by_q(self) if include_hom_gradients else ([], [])
        d2Xmats_hom, _ = _global_hom_second_derivative_matrices(self) if include_hom_hessians else ([], [])
        Xhom_size, dXhom_size, d2Xhom_size = self.gen_get_Xhom_size()
        for ind in range(len(Xmats_hom)):
            self.gen_add_code_line("// Xhom[" + str(ind) + "]")
            for col in range(4):
                for row in range(4):
                    val = Xmats_hom[ind][row,col]
                    if not self.custom_is_constant(val): # initialize to 0
                        val = 0
                    str_val = str(val)
                    cpp_ind = baseXI_size + self.gen_static_array_ind_3d(ind,col,row,ind_stride=16,col_stride=4)
                    self.gen_add_code_line("h_XImats[" + str(cpp_ind) + "] = static_cast<T>(" + str_val + ");")
        # and the gradients
        if include_hom_gradients:
            for ind in range(len(dXmats_hom)):
                self.gen_add_code_line("// dXhom[" + str(ind) + "]")
                for col in range(4):
                    for row in range(4):
                        val = dXmats_hom[ind][row,col]
                        if not self.custom_is_constant(val): # initialize to 0
                            val = 0
                        str_val = str(val)
                        cpp_ind = baseXI_size + Xhom_size + self.gen_static_array_ind_3d(ind,col,row,ind_stride=16,col_stride=4)
                        self.gen_add_code_line("h_XImats[" + str(cpp_ind) + "] = static_cast<T>(" + str_val + ");")
        # and the 2nd derivatives
        if include_hom_hessians:
            for ind in range(len(d2Xmats_hom)):
                self.gen_add_code_line("// d2Xhom[" + str(ind) + "]")
                for col in range(4):
                    for row in range(4):
                        val = d2Xmats_hom[ind][row,col]
                        if not self.custom_is_constant(val): # initialize to 0
                            val = 0
                        str_val = str(val)
                        cpp_ind = baseXI_size + Xhom_size + dXhom_size + self.gen_static_array_ind_3d(ind,col,row,ind_stride=16,col_stride=4)
                        self.gen_add_code_line("h_XImats[" + str(cpp_ind) + "] = static_cast<T>(" + str_val + ");")
    # allocate and transfer data to the GPU, free CPU memory and return the pointer to the memory
    self.gen_add_code_line("T *d_XImats; gpuErrchk(cudaMalloc((void**)&d_XImats," + str(XI_size) + "*sizeof(T)));")
    self.gen_add_code_line("gpuErrchk(cudaMemcpy(d_XImats,h_XImats," + str(XI_size) + "*sizeof(T),cudaMemcpyHostToDevice));")
    self.gen_add_code_line("free(h_XImats);")
    self.gen_add_code_line("return d_XImats;")
    # add the function end
    self.gen_add_end_function()

def gen_get_inertia_params_size(self, include_base_inertia = False):
    # 10 standard inertial parameters per body, body-indexed, mirroring the
    # I-region layout of gen_init_XImats: bodies 1..N by default (Imats[1:]),
    # or all N+1 bodies when include_base_inertia.
    n = self.robot.get_num_joints()
    return 10 * (n + (1 if include_base_inertia else 0))

def gen_init_inertia_params(self, include_base_inertia = False):
    # D.4 / Phase 5: host-side init of the flag-gated mutable inertia table.
    # Fills d_inertia_params from the SAME URDF (frozen regressor basis), so
    # init_robotModel reproduces the baked answer until set_inertia_params is
    # called. Layout mirrors the I-region of gen_init_XImats exactly: bodies
    # 1..N by default (the base inertia is dropped just like Imats[1:]).
    self.gen_add_func_doc("Initializes the mutable inertia parameter table in GPU memory",
            ["Memory order is pi[0...N-1], each pi_i = [m, h(3)=m*c, I_O(6)=[Ixx,Ixy,Ixz,Iyy,Iyz,Izz]]"],
            [], "A pointer to the inertia-params memory in the GPU")
    self.gen_add_code_line("template <typename T>")
    self.gen_add_code_line("__host__")
    self.gen_add_code_line("T* init_inertia_params() {", True)
    size = self.gen_get_inertia_params_size(include_base_inertia)
    self.gen_add_code_line("T *h_inertia_params = (T *)calloc(" + str(size) + ",sizeof(T));")
    params = self.robot.get_inertia_params_ordered_by_id()
    if not include_base_inertia:
        params = params[1:]
    for ind in range(len(params)):
        self.gen_add_code_line("// pi[" + str(ind) + "]")
        for k in range(10):
            self.gen_add_code_line("h_inertia_params[" + str(10*ind + k) + "] = static_cast<T>(" + str(params[ind][k]) + ");")
    self.gen_add_code_line("T *d_inertia_params; gpuErrchk(cudaMalloc((void**)&d_inertia_params," + str(size) + "*sizeof(T)));")
    self.gen_add_code_line("gpuErrchk(cudaMemcpy(d_inertia_params,h_inertia_params," + str(size) + "*sizeof(T),cudaMemcpyHostToDevice));")
    self.gen_add_code_line("free(h_inertia_params);")
    self.gen_add_code_line("return d_inertia_params;")
    self.gen_add_end_function()

def gen_set_inertia_params(self, include_base_inertia = False):
    # D.4 / Phase 5: public mutator. Thin cudaMemcpy of the 10*NB (or 10*(NB+1))
    # param table into d_inertia_params. The sysID / domain-randomization /
    # payload entry point. h_params must be in the frozen regressor basis
    # (see Robot.get_inertia_params), body-indexed, bodies 1..N (or 0..N when
    # include_base_inertia).
    size = self.gen_get_inertia_params_size(include_base_inertia)
    self.gen_add_func_doc("Updates the mutable inertia parameter table on the GPU at runtime (no recompile)",
            ["h_params is the host array of " + str(size) + " floats, body-indexed 10-vectors [m, h(3), I_O(6)]"],
            [], None)
    self.gen_add_code_line("template <typename T>")
    self.gen_add_code_line("__host__")
    self.gen_add_code_line("void set_inertia_params(robotModel<T> *d_robotModel, const T *h_params) {", True)
    # d_inertia_params is an inner pointer inside the device-resident struct; read
    # it back to host so we can memcpy into the buffer it points at.
    self.gen_add_code_line("robotModel<T> h_robotModel;")
    self.gen_add_code_line("gpuErrchk(cudaMemcpy(&h_robotModel,d_robotModel,sizeof(robotModel<T>),cudaMemcpyDeviceToHost));")
    self.gen_add_code_line("gpuErrchk(cudaMemcpy(h_robotModel.d_inertia_params,h_params," + str(size) + "*sizeof(T),cudaMemcpyHostToDevice));")
    self.gen_add_end_function()

def gen_get_transform_params_size(self):
    # runtime_transform: 6 raw URDF origin scalars [x,y,z,roll,pitch,yaw] per
    # JOINT (joint-indexed, ALL joints), mirroring the X-region body order.
    return 6 * self.robot.get_num_joints()

def gen_init_transform_params(self):
    # runtime_transform (mirror of gen_init_inertia_params): host-side init of
    # the flag-gated mutable origin-param table. Fills d_transform_params from
    # the SAME URDF (frozen [x,y,z,r,p,y] basis) so init_robotModel reproduces
    # the baked Xfixed bit-for-bit until set_transform_params is called.
    self.gen_add_func_doc("Initializes the mutable joint-origin transform parameter table in GPU memory",
            ["Memory order is op[0...NB-1], each op_i = [x, y, z, roll, pitch, yaw] (raw URDF <origin>)"],
            [], "A pointer to the transform-params memory in the GPU")
    self.gen_add_code_line("template <typename T>")
    self.gen_add_code_line("__host__")
    self.gen_add_code_line("T* init_transform_params() {", True)
    size = self.gen_get_transform_params_size()
    self.gen_add_code_line("T *h_transform_params = (T *)calloc(" + str(size) + ",sizeof(T));")
    params = self.robot.get_origin_params_ordered_by_id()
    for ind in range(len(params)):
        self.gen_add_code_line("// op[" + str(ind) + "]")
        for k in range(6):
            self.gen_add_code_line("h_transform_params[" + str(6*ind + k) + "] = static_cast<T>(" + repr(float(params[ind][k])) + ");")
    self.gen_add_code_line("T *d_transform_params; gpuErrchk(cudaMalloc((void**)&d_transform_params," + str(size) + "*sizeof(T)));")
    self.gen_add_code_line("gpuErrchk(cudaMemcpy(d_transform_params,h_transform_params," + str(size) + "*sizeof(T),cudaMemcpyHostToDevice));")
    self.gen_add_code_line("free(h_transform_params);")
    self.gen_add_code_line("return d_transform_params;")
    self.gen_add_end_function()

def gen_set_transform_params(self):
    # runtime_transform (mirror of gen_set_inertia_params): public mutator. Thin
    # cudaMemcpy of the 6*NB origin table into d_transform_params. h_params must
    # be joint-indexed 6-vectors [x,y,z,roll,pitch,yaw] (raw URDF origin basis).
    size = self.gen_get_transform_params_size()
    self.gen_add_func_doc("Updates the mutable joint-origin transform table on the GPU at runtime (no recompile)",
            ["h_params is the host array of " + str(size) + " floats, joint-indexed 6-vectors [x,y,z,roll,pitch,yaw]"],
            [], None)
    self.gen_add_code_line("template <typename T>")
    self.gen_add_code_line("__host__")
    self.gen_add_code_line("void set_transform_params(robotModel<T> *d_robotModel, const T *h_params) {", True)
    self.gen_add_code_line("robotModel<T> h_robotModel;")
    self.gen_add_code_line("gpuErrchk(cudaMemcpy(&h_robotModel,d_robotModel,sizeof(robotModel<T>),cudaMemcpyDeviceToHost));")
    self.gen_add_code_line("gpuErrchk(cudaMemcpy(h_robotModel.d_transform_params,h_params," + str(size) + "*sizeof(T),cudaMemcpyHostToDevice));")
    self.gen_add_end_function()

def _joint_dynamics_folded_by_vslot(self):
    """runtime_joint_dynamics: the SINGLE source of truth for the alpha-FOLDED
    per-v-slot damping/friction table (length nv each).

    Reuses the EXACT jid->v-slot fold the baked bias emitters use
    (gen_inverse_dynamics_joint_dynamics_bias / _idg_damping_diag_cpp): iterate
    joints, skip the floating root, map jid->v-slot via _v_slot_cpp /
    get_joint_index_v, and ACCUMULATE alpha*b / alpha*fr into the shared slot so
    several mimic joints feeding one reduced coordinate fold to a single fused
    coefficient. Returning the folded vectors here (rather than re-deriving the
    fold in init_joint_dynamics_params, _compile meta, and the bias read) keeps
    the device-table init, the persisted meta, and the bias reads agreeing BY
    CONSTRUCTION — the bit-identity invariant in C5 plan §3.4.

    Returns (damping_by_vslot, friction_by_vslot): two length-nv python float
    lists, v-slot indexed (slot k == s_qd[k]/s_c[k]). Zero in every slot the
    URDF leaves undamped (incl. the floating root's slots 0..5).
    """
    nv = self.robot.get_num_vel()
    b_vslot  = [0.0] * nv
    f_vslot  = [0.0] * nv
    HAS_DAMP = self.robot.robot_has_joint_damping()
    HAS_FRIC = self.robot.robot_has_joint_friction()
    HAS_MIMIC = self.robot_has_mimic_joints()
    fb = self.robot.floating_base
    for jid in range(self.robot.get_num_joints()):
        if fb and jid == 0:
            continue  # floating root carries no damping/friction
        b  = float(self.robot.get_damping_by_id(jid))  if HAS_DAMP else 0.0
        fr = float(self.robot.get_friction_by_id(jid)) if HAS_FRIC else 0.0
        if b == 0.0 and fr == 0.0:
            continue
        if HAS_MIMIC:
            vs = self._v_slot_cpp(jid)
            alpha = float(self._alpha_for_jid(jid))
        else:
            vs = self.robot.get_joint_index_v(jid)
            alpha = 1.0
        b_vslot[vs] += alpha * b
        f_vslot[vs] += alpha * fr
    return b_vslot, f_vslot

def gen_get_joint_dynamics_params_size(self):
    # runtime_joint_dynamics: nv damping + nv friction = 2*nv scalars, v-slot
    # indexed (damping in [0,nv), friction in [nv,2nv)). Damping/friction are
    # pure per-DOF scalars (NOT in any sparsity pattern), so this is the
    # simplest of the three runtime tables.
    return 2 * self.robot.get_num_vel()

def gen_init_joint_dynamics_params(self):
    # runtime_joint_dynamics (mirror of gen_init_inertia_params): host-side init
    # of the flag-gated mutable [damping||friction] table. Fills
    # d_joint_dynamics_params from the SAME URDF (the alpha-FOLDED per-v-slot
    # coefficient, _joint_dynamics_folded_by_vslot) so init_robotModel reproduces
    # the baked bias BIT-FOR-BIT until set_joint_dynamics_params is called. Unlike
    # runtime_transform there is no on-device sin/cos rebuild and no sparsity
    # change, so the runtime path is bit-identical (not merely float-identical).
    self.gen_add_func_doc("Initializes the mutable joint-dynamics (damping/friction) table in GPU memory",
            ["Memory order is jd[0..nv-1]=damping[v], jd[nv..2nv-1]=friction[v] (v-slot indexed, alpha-folded)"],
            [], "A pointer to the joint-dynamics-params memory in the GPU")
    self.gen_add_code_line("template <typename T>")
    self.gen_add_code_line("__host__")
    self.gen_add_code_line("T* init_joint_dynamics_params() {", True)
    nv   = self.robot.get_num_vel()
    size = self.gen_get_joint_dynamics_params_size()
    self.gen_add_code_line("T *h_joint_dynamics_params = (T *)calloc(" + str(size) + ",sizeof(T));")
    b_vslot, f_vslot = self._joint_dynamics_folded_by_vslot()
    for vs in range(nv):
        b = b_vslot[vs]; fr = f_vslot[vs]
        if b == 0.0 and fr == 0.0:
            continue  # leave calloc'd zero (incl. the floating root's slots 0..5)
        self.gen_add_code_line("// v-slot " + str(vs))
        if b != 0.0:
            self.gen_add_code_line("h_joint_dynamics_params[" + str(vs) + "] = static_cast<T>(" + repr(float(b)) + ");")
        if fr != 0.0:
            self.gen_add_code_line("h_joint_dynamics_params[" + str(nv + vs) + "] = static_cast<T>(" + repr(float(fr)) + ");")
    self.gen_add_code_line("T *d_joint_dynamics_params; gpuErrchk(cudaMalloc((void**)&d_joint_dynamics_params," + str(size) + "*sizeof(T)));")
    self.gen_add_code_line("gpuErrchk(cudaMemcpy(d_joint_dynamics_params,h_joint_dynamics_params," + str(size) + "*sizeof(T),cudaMemcpyHostToDevice));")
    self.gen_add_code_line("free(h_joint_dynamics_params);")
    self.gen_add_code_line("return d_joint_dynamics_params;")
    self.gen_add_end_function()

def gen_set_joint_dynamics_params(self):
    # runtime_joint_dynamics (mirror of gen_set_inertia_params): public mutator.
    # Thin cudaMemcpy of the 2*nv [damping||friction] table into
    # d_joint_dynamics_params. The sysID / domain-randomization entry point for
    # joint dynamics. h_params must be the alpha-FOLDED per-v-slot coefficients
    # (same basis as init_joint_dynamics_params); passing the baked values back
    # reproduces the baked result bit-for-bit.
    size = self.gen_get_joint_dynamics_params_size()
    self.gen_add_func_doc("Updates the mutable joint-dynamics table on the GPU at runtime (no recompile)",
            ["h_params is the host array of " + str(size) + " floats: [damping(nv) || friction(nv)], v-slot indexed"],
            [], None)
    self.gen_add_code_line("template <typename T>")
    self.gen_add_code_line("__host__")
    self.gen_add_code_line("void set_joint_dynamics_params(robotModel<T> *d_robotModel, const T *h_params) {", True)
    # d_joint_dynamics_params is an inner pointer inside the device-resident
    # struct; read it back to host so we can memcpy into the buffer it points at.
    self.gen_add_code_line("robotModel<T> h_robotModel;")
    self.gen_add_code_line("gpuErrchk(cudaMemcpy(&h_robotModel,d_robotModel,sizeof(robotModel<T>),cudaMemcpyDeviceToHost));")
    self.gen_add_code_line("gpuErrchk(cudaMemcpy(h_robotModel.d_joint_dynamics_params,h_params," + str(size) + "*sizeof(T),cudaMemcpyHostToDevice));")
    self.gen_add_end_function()

def _emit_runtime_transform_rebuild(self, n):
    """runtime_transform: rebuild each joint's constant 6x6 Xfixed scratch from
    d_transform_params, once per launch (mirror of _emit_runtime_inertia_rebuild).

    Emitted only inside the `if constexpr (RUNTIME_TRANSFORM)` path. For joint
    `ind` the 6 raw params [x,y,z,r,p,y] recompute Xfixed = rot(E(rpy)) *
    xlt(skew(xyz)) into s_temp[XFIXED_OFF + 36*ind + 6*col + row] (X-region
    col-major 6x6). Xfixed has the block structure
        [[ E,        0 ],
         [ -E*skew(t), E ]]
    so TL==BR==E, TR==0, BL==-E*skew(t). One thread per joint writes its 36
    entries; n is tiny and this runs once per kernel on the cold XImats load.
    The hot loop then loads these xf_* cells instead of inline origin literals,
    so the rebuilt Xfixed (numerically == the baked origin) makes the runtime
    path BIT-IDENTICAL to the baked path until set_transform_params mutates it.
    """
    xoff = _runtime_transform_xfixed_offset(self)
    def off(row, col):
        return 6 * col + row
    self.gen_add_code_line("if constexpr (RUNTIME_TRANSFORM) {", True)
    self.gen_add_sync()
    self.gen_add_parallel_loop("rtj", str(n))
    self.gen_add_code_line("const T *op = &d_robotModel->d_transform_params[6*rtj];")
    self.gen_add_code_line("T tx = op[0]; T ty = op[1]; T tz = op[2];")
    self.gen_add_code_line("T cr = static_cast<T>(cos(op[3])); T sr = static_cast<T>(sin(op[3]));")
    self.gen_add_code_line("T cp = static_cast<T>(cos(op[4])); T sp = static_cast<T>(sin(op[4]));")
    self.gen_add_code_line("T cy = static_cast<T>(cos(op[5])); T sy = static_cast<T>(sin(op[5]));")
    # E = rx(r)*ry(p)*rz(y), GRiD frame-rotation convention (see SpatialAlgebra
    # Rotation.{rx,ry,rz} and the closed form derived from build_fixed_transform).
    E = {
        (0, 0): "cp*cy", (0, 1): "sy*cp", (0, 2): "-sp",
        (1, 0): "sp*sr*cy - sy*cr", (1, 1): "sp*sr*sy + cr*cy", (1, 2): "sr*cp",
        (2, 0): "sp*cr*cy + sr*sy", (2, 1): "sp*sy*cr - sr*cy", (2, 2): "cp*cr",
    }
    for (i, j), e in E.items():
        self.gen_add_code_line("T E" + str(i) + str(j) + " = " + e + ";")
    self.gen_add_code_line("int b = " + str(xoff) + " + 36*rtj;")
    # TL = E (rows 0-2, cols 0-2), BR = E (rows 3-5, cols 3-5)
    for i in range(3):
        for j in range(3):
            self.gen_add_code_line("s_temp[b + " + str(off(i, j)) + "] = E" + str(i) + str(j) + ";")          # TL
            self.gen_add_code_line("s_temp[b + " + str(off(i + 3, j + 3)) + "] = E" + str(i) + str(j) + ";")  # BR
    # TR = 0 (rows 0-2, cols 3-5)
    for i in range(3):
        for j in range(3):
            self.gen_add_code_line("s_temp[b + " + str(off(i, j + 3)) + "] = static_cast<T>(0);")
    # BL = -E*skew(t) (rows 3-5, cols 0-2); skew(t)=[[0,-tz,ty],[tz,0,-tx],[-ty,tx,0]]
    # BL[i,j] = -sum_k E[i,k]*skew[k,j]
    skew = [["0", "-tz", "ty"], ["tz", "0", "-tx"], ["-ty", "tx", "0"]]
    for i in range(3):
        for j in range(3):
            terms = []
            for k in range(3):
                sk = skew[k][j]
                if sk == "0":
                    continue
                terms.append("E" + str(i) + str(k) + "*(" + sk + ")")
            expr = " + ".join(terms) if terms else "static_cast<T>(0)"
            self.gen_add_code_line("s_temp[b + " + str(off(i + 3, j)) + "] = -(" + expr + ");")
    self.gen_add_end_control_flow()
    self.gen_add_sync()
    self.gen_add_end_control_flow()

def _runtime_transform_xfixed_offset(self):
    # Base offset into s_temp where the per-joint Xfixed scratch (s_Xfixed)
    # begins, i.e. the size of the existing sin/cos region. Must match the
    # region the non-runtime path uses so the hot-loop sin/cos reads are
    # byte-identical; s_Xfixed is appended AFTER it under runtime_transform.
    if self.robot_has_mimic_joints():
        return 3 * self.robot.get_num_joints()
    return 2 * self.robot.get_num_pos()

def gen_load_update_XImats_helpers_temp_mem_size(self):
    # runtime_transform (mirror of runtime_inertia): append a per-joint Xfixed
    # scratch (36*NB floats, X-region 6x6 col-major layout) AFTER the existing
    # sin/cos region. The on-device prologue rebuilds each joint's constant
    # origin transform here once per launch; the hot loop then loads origin
    # coefficients from s_Xfixed instead of inline literals. Gated on the flag so
    # the baked arena/header stays byte-identical.
    if self.robot_has_mimic_joints():
        # Mimic path needs per-BODY scratch: s_q_eff[NB] (the folded angle
        # alpha*q[target]+offset) plus per-body sin/cos (2*NB). Non-mimic
        # robots keep the legacy 2*nq so their arena/header stays byte-identical.
        NB = self.robot.get_num_joints()
        base = 3*NB
    else:
        n = self.robot.get_num_pos()
        base = 2*n
    if getattr(self, "runtime_transform", False):
        base += 36 * self.robot.get_num_joints()
    return base

def gen_load_update_XImats_helpers_function_call(self, updated_var_names = None,
                                                 skip_floating_base_X = False):
    """Emit a call to load_update_XImats_helpers.

    skip_floating_base_X (CRBA-only surgical lever, A.3): when True AND the robot
    has a floating base, the called specialization elides the per-call
    recomputation of the floating root spatial transform X[0] (the heavy
    quaternion->rotation block emitted on the single-thread serial path).
    CRBA never dereferences s_XImats[0..35] (Phase-1 BFS starts at level 1, and
    Phase-2's chain walk only reads X[X_id] for X_id in {jid} ∪ ancestors[:-1]
    — the root 0 only appears as `anc`, never as `X_id`), so the work is dead
    for CRBA. Other algorithms (ID/FD/Minv/ABA/integrator/IDSVA-SO/EE-pose)
    keep the default False — they walk the root X. Has no effect on fixed-base
    robots.
    """
    var_names = dict( \
        s_XImats_name = "s_XImats", \
        d_robotModel_name = "d_robotModel", \
        s_q_name = "s_q", \
        s_temp_name = "s_temp", \
        s_topology_helpers_name = "s_topology_helpers", \
    )
    if updated_var_names is not None:
        for key,value in updated_var_names.items():
            var_names[key] = value
    # D.4 / Phase 5: thread RUNTIME_INERTIA (3rd tparam) through every call-site
    # when self.runtime_inertia. SKIP_FLOATING_BASE_X (2nd tparam) must then be
    # spelled explicitly so the 3rd can be set. Baked default keeps the legacy
    # <T> / <T,true> tparams byte-identical (the 3rd param doesn't exist).
    # runtime_transform threads RUNTIME_TRANSFORM (4th tparam). When set it also
    # forces RUNTIME_INERTIA (3rd) to be spelled (its actual value follows the
    # inertia flag). All explicit-tparam spellings degrade to the legacy
    # <T>/<T,true> when no runtime flag is active (byte-identical baked header).
    skip_val = "true" if (skip_floating_base_X and self.robot.floating_base) else "false"
    runtime_inertia = getattr(self, "runtime_inertia", False)
    runtime_transform = getattr(self, "runtime_transform", False)
    if runtime_transform:
        ri_val = "true" if runtime_inertia else "false"
        tparams = "<T, " + skip_val + ", " + ri_val + ", true>"
    elif runtime_inertia:
        tparams = "<T, " + skip_val + ", true>"
    else:
        tparams = "<T, true>" if (skip_floating_base_X and self.robot.floating_base) else "<T>"
    code_start = "load_update_XImats_helpers" + tparams + "(" + var_names["s_XImats_name"] + ", " + var_names["s_q_name"] + ", "
    code_end = var_names["d_robotModel_name"] + ", " + var_names["s_temp_name"] + ");"
    n = self.robot.get_num_pos()
    # Always pass s_topology_helpers (uniform signature; nullptr for serial chains).
    code_start += var_names["s_topology_helpers_name"] + ", "
    self.gen_add_code_line(code_start + code_end)

def gen_XImats_helpers_temp_shared_memory_code(self, temp_mem_size = 0, include_base_inertia = False,
                                               include_homogenous_transforms = False, extra_t_buffers = None,
                                               include_linalg_scratch = False,
                                               linalg_scratch_bytes = "GRID_LINALG_NVIDIA_MAX_HELPER_BYTES<T>()",
                                               tier_workspace_expr = None):
    n = self.robot.get_num_pos()
    XI_size = self.gen_get_XI_size(include_base_inertia,include_homogenous_transforms)
    if extra_t_buffers is None:
        extra_t_buffers = []
    # runtime_transform: the XImats helper rebuilds each joint's constant 6x6
    # Xfixed into s_temp at offset _runtime_transform_xfixed_offset (= 2*num_pos
    # non-mimic / 3*NB mimic), occupying 36*NB extra floats. That block is DEAD
    # after the helper returns (the hot loop consumes it to build s_XImats), so
    # the algorithm's inner scratch may reuse [0, inner) afterward. The s_temp
    # region must hold max(inner, xfixed_offset + 36*NB) during the helper. We
    # reserve a purely-additive 36*NB band ON TOP of the algorithm's inner scratch
    # — [inner, inner+36*NB) — which is always >= the helper's xfixed peak because
    # every XImats kernel's inner scratch is >= xfixed_offset (the RNEA/ABA/CRBA
    # inners stash >= 6 floats per body >= 2*num_pos). This matches the per-algo
    # arena t_count reserve (also +36*NB) byte-for-byte so the layout assert and
    # the launched smem agree. Baked path keeps temp_mem_size byte-identical (no
    # growth, no Xfixed block). The Xfixed write at offset 2*num_pos still lands
    # inside [0, inner+36*NB) since inner >= 2*num_pos.
    if getattr(self, "runtime_transform", False) and not include_homogenous_transforms:
        temp_mem_size = int(temp_mem_size or 0) + 36 * self.robot.get_num_joints()
    self.gen_declare_shared_arena(extra_t_buffers, temp_mem_size,
                                  include_topology_helpers = (not self.robot.is_serial_chain() or not self.robot.are_Ss_identical(list(range(n)))),
                                  ximat_name = "s_XImats",
                                  ximat_size = XI_size,
                                  temp_name = "s_temp",
                                  topology_name = "s_topology_helpers",
                                  extra_byte_regions = [("s_linalg_smem", linalg_scratch_bytes)] if include_linalg_scratch else None,
                                  tier_workspace_expr = tier_workspace_expr)

def _emit_mimic_q_fold(self):
    """Emit the per-BODY effective-angle q-fold scratch used by every
    mimic-aware transform path. Layout in s_temp:
        [0, NB)   s_q_eff   [NB, 2NB) sin   [2NB, 3NB) cos
    For body `ind` (single-DoF revolute/prismatic) the effective angle is
    s_q_eff[ind] = alpha_ind * s_q[q_slot(ind)] + offset_ind, mirroring
    RBDReference.q_for_joint. A mimic body and its target both evaluate their
    transform at the prescribed scaled+offset coordinate. For NON-mimic bodies
    alpha==1, offset==0, and q_slot(ind) is the body's own dense q-offset
    (== ind on a fixed base, == ind+6 on a floating base), so this reduces to a
    plain copy. The floating root (ind 0, a multi-DoF joint with no `theta`) is
    skipped — its quaternion transform reads the raw s_q[0..6] directly. This
    one emit supports BOTH fixed-base and floating-base mimic models."""
    NB = self.robot.get_num_joints()
    self.gen_add_serial_ops()
    for ind in range(NB):
        qslot = self.robot.get_joint_index_q(ind)
        if isinstance(qslot, (list, tuple)):
            if len(qslot) != 1:
                # Multi-DoF root (floating base): no scalar theta to fold; the
                # quaternion substitution path reads s_q[0..6] directly. Still
                # initialize the scratch slot so a stray read is well-defined.
                self.gen_add_code_line("s_temp[" + str(ind) + "] = static_cast<T>(0);")
                continue
            qslot = qslot[0]
        j = self.robot.get_joint_by_id(ind)
        if getattr(j, "is_mimic", False):
            mult = j.get_mimic_multiplier()
            off = j.get_mimic_offset()
            expr = "static_cast<T>(" + repr(mult) + ") * s_q[" + str(qslot) + "]"
            if off != 0.0:
                expr += " + static_cast<T>(" + repr(off) + ")"
            self.gen_add_code_line("s_temp[" + str(ind) + "] = " + expr + ";")
        else:
            self.gen_add_code_line("s_temp[" + str(ind) + "] = s_q[" + str(qslot) + "];")
    self.gen_add_end_control_flow()
    self.gen_add_sync()
    self.gen_add_parallel_loop("k", str(NB))
    self.gen_add_code_line("s_temp[k+" + str(NB) + "] = static_cast<T>(sin(s_temp[k]));")
    self.gen_add_code_line("s_temp[k+" + str(2*NB) + "] = static_cast<T>(cos(s_temp[k]));")
    self.gen_add_end_control_flow()
    self.gen_add_sync()

def _xi_fixed_sincos_subst(self, str_val, ind):
    """Substitute sin/cos/theta for fixed-base body `ind` into a transform
    cell. Mimic-aware: on a model with mimic joints, body `ind` reads its
    folded angle and per-body sin/cos from the s_q_eff layout
    (s_q_eff[NB] | sin[NB] | cos[NB]); otherwise it reads the legacy
    s_temp[ind]/s_temp[ind+nq]/s_q[ind] (byte-identical to pre-mimic).

    Tier-C spherical bodies carry quaternion symbols (q1_sph..q4_sph) instead of
    a single `theta`; they're substituted via _xi_spherical_quat_subst (ccode form
    for the homogeneous-transform consumer) BEFORE the theta passes, which then
    no-op (no `theta` token remains). Byte-identical for non-spherical robots."""
    if self.robot.joint_is_spherical(ind):
        return _xi_spherical_quat_subst(self, str_val, ind, ccode=True)
    if self.robot_has_mimic_joints():
        NB = self.robot.get_num_joints()
        str_val = str_val.replace("sin(theta)", "s_temp[" + str(ind + NB) + "]")
        str_val = str_val.replace("cos(theta)", "s_temp[" + str(ind + 2*NB) + "]")
        str_val = str_val.replace("theta", "s_temp[" + str(ind) + "]")
        return str_val
    # sincos fill is over q in [0,nq); read this body's OWN q-slot (== ind on an
    # all-cardinal fixed robot, where nq==NB -> byte-identical; shifted when a
    # multi-DoF spherical joint precedes this body).
    nq = self.robot.get_num_pos()
    qslot = self.robot.get_joint_index_q(ind)
    if isinstance(qslot, (list, tuple)):
        qslot = qslot[0]
    str_val = str_val.replace("sin(theta)", "s_temp[" + str(qslot) + "]")
    str_val = str_val.replace("cos(theta)", "s_temp[" + str(qslot + nq) + "]")
    str_val = str_val.replace("theta", "s_q[" + str(qslot) + "]")
    return str_val

def _xi_spherical_quat_subst(self, str_val, ind, ccode=False):
    """Substitute a SPHERICAL (ball) joint's unit-quaternion symbols
    (q1_sph..q4_sph = x,y,z,w) into a transform cell for body `ind`.

    A spherical joint owns a CONTIGUOUS 4-wide q-block at `get_joint_index_q(ind)`
    (e.g. spherical_arm's joint_1 owns s_q[0..3]); a mid-chain spherical shifts
    every downstream joint's q-offset by +1 vs its v-offset (nq>nv), so we read
    the offset from the parser's dense q-map rather than from `ind`. This mirrors
    the floating-root q1_fb..q4_fb substitution but parameterized on the joint's
    own q-block start (the root is hardcoded to s_q[3..6]).

    `ccode=True` selects the C `pow(q, 2)` squared form emitted by `sp.ccode`
    (homogeneous transforms); the default handles the Python `q**2` repr used by
    the spatial X block (`str(val)`). The substitution is q-normalized in the
    symbolic transform itself, so no separate normalization is needed here."""
    qoff = self.robot.get_joint_index_q(ind)
    if isinstance(qoff, (list, tuple)):
        qoff = qoff[0]
    syms = ["q1_sph", "q2_sph", "q3_sph", "q4_sph"]
    if ccode:
        # sp.ccode emits pow(qk_sph, 2); replace the squared form first so the
        # bare-symbol pass below doesn't corrupt the `pow(` argument.
        for k, s in enumerate(syms):
            slot = "s_q[" + str(qoff + k) + "]"
            str_val = str_val.replace("pow(" + s + ", 2)", slot + "*" + slot)
    else:
        for k, s in enumerate(syms):
            slot = "s_q[" + str(qoff + k) + "]"
            str_val = str_val.replace(s + "**2", slot + "*" + slot)
    for k, s in enumerate(syms):
        str_val = str_val.replace(s, "s_q[" + str(qoff + k) + "]")
    return str_val

def _emit_runtime_inertia_rebuild(self, n):
    """D.4 / Phase 5: rebuild the I-region of s_XImats from d_inertia_params.

    Emitted only inside the `if constexpr (RUNTIME_INERTIA)` path. The I-region
    holds bodies 1..N at static slot `ind` in [0,n): base offset 36*(n+ind),
    element (row,col) at 36*(n+ind)+6*col+row (column-major 6x6). Each body's
    10-vector p = [m, hx,hy,hz, Ixx,Ixy,Ixz,Iyy,Iyz,Izz] is scattered into the
    6x6 as
        I = [[ I_O,        skew(h) ],
             [ skew(h)^T,  m*I3    ]]
    a pure divide-free scatter of the SAME numbers the baked path stored (read
    out verbatim by Robot.get_inertia_params) -> BIT-IDENTICAL to the baked path.
    One thread per body writes its 36 entries; n is tiny and this runs once per
    kernel on the cold XImats load.
    """
    # (within-6x6 col-major offset, expr-in-terms-of m,h,IO) for all 36 entries.
    # Zeros are written explicitly because the bulk copy skipped the I-region.
    def off(row, col):
        return 6 * col + row
    entries = {}
    for r in range(6):
        for c in range(6):
            entries[off(r, c)] = "static_cast<T>(0)"
    IO_names = {  # (row,col within top-left 3x3) -> param index for symmetric I_O
        (0, 0): 4, (1, 1): 7, (2, 2): 9,
        (0, 1): 5, (1, 0): 5, (0, 2): 6, (2, 0): 6, (1, 2): 8, (2, 1): 8,
    }
    for (r, c), k in IO_names.items():
        entries[off(r, c)] = "IO" + str(k - 4)  # IO0..IO5
    # TR = skew(h) at rows0-2, cols3-5 ; BL = skew(h)^T at rows3-5, cols0-2
    # skew(h) = [[0,-hz,hy],[hz,0,-hx],[-hy,hx,0]]
    skew = {(0, 1): "-hz", (0, 2): "hy", (1, 0): "hz",
            (1, 2): "-hx", (2, 0): "-hy", (2, 1): "hx"}
    for (r, c), e in skew.items():
        entries[off(r, c + 3)] = e          # TR
        entries[off(r + 3, c)] = e          # BL = skew^T -> transpose (r<->c) but skew^T[i,j]=skew[j,i]
    # Correct BL = skew(h)^T: BL[i,j] = skew(h)[j,i]
    for (r, c), e in skew.items():
        entries[off(c + 3, r)] = e
    # BR = m*I3
    entries[off(3, 3)] = "m"
    entries[off(4, 4)] = "m"
    entries[off(5, 5)] = "m"
    # NB: the syncs and the loop all live INSIDE the if-constexpr so the baked
    # path (RUNTIME_INERTIA=false) emits NOTHING here -> byte-identical.
    self.gen_add_code_line("if constexpr (RUNTIME_INERTIA) {", True)
    self.gen_add_sync()
    self.gen_add_parallel_loop("rib", str(n))
    self.gen_add_code_line("const T *p = &d_robotModel->d_inertia_params[10*rib];")
    self.gen_add_code_line("T m = p[0]; T hx = p[1]; T hy = p[2]; T hz = p[3];")
    self.gen_add_code_line("T IO0 = p[4]; T IO1 = p[5]; T IO2 = p[6]; T IO3 = p[7]; T IO4 = p[8]; T IO5 = p[9];")
    self.gen_add_code_line("int base = 36*(" + str(n) + " + rib);")
    for o in range(36):
        self.gen_add_code_line("s_XImats[base + " + str(o) + "] = " + entries[o] + ";")
    self.gen_add_end_control_flow()
    self.gen_add_sync()
    self.gen_add_end_control_flow()

def gen_load_update_XImats_helpers(self, include_base_inertia = False, include_homogenous_transforms = False):
    n = self.robot.get_num_joints()
    XI_size = self.gen_get_XI_size(include_base_inertia,include_homogenous_transforms)
    baseXI_size = self.gen_get_XI_size(include_base_inertia,include_homogenous_transforms=False) # just base XI size (for homogenous if needed)
    # add function description
    func_def_start = "void load_update_XImats_helpers("
    func_def_middle = "T *s_XImats, const T *s_q, "
    func_def_end = "const robotModel<T> *d_robotModel, T *s_temp) {"
    func_params = ["s_XImats is the (shared) memory destination location for the XImats",\
        "s_q is the (shared) memory location of the current configuration",\
        "d_robotModel is the pointer to the initialized model specific helpers (XImats, mxfuncs, topology_helpers, etc.)", \
        "s_temp is temporary (shared) memory used to compute sin and cos if needed of size: " + \
                str(self.gen_load_update_XImats_helpers_temp_mem_size())]
    # Always emit s_topology_helpers for a uniform signature; serial chains with
    # identical Ss don't read it (they pass nullptr and skip the topology-copy body
    # below). -Wunused-parameter is off in our builds.
    func_def_middle += "int *s_topology_helpers, "
    func_params.insert(-2,"s_topology_helpers is the (shared) memory location for the topology_helpers (nullptr/unused for serial chains with identical Ss)")
    func_def = func_def_start + func_def_middle + func_def_end
    # then genearte the code
    self.gen_add_func_doc("Updates the Xmats in (shared) GPU memory acording to the configuration",[],func_params,None)
    # SKIP_FLOATING_BASE_X: A.3 surgical lever. When true AND the robot has a
    # floating base, elide the per-call recomputation of X[0] (the floating
    # root spatial transform's heavy quaternion->rotation block). CRBA never
    # reads s_XImats[0..35] (the BFS body recursion starts at level 1 and the
    # M-fill chain walk only uses X[X_id] for X_id != 0). Fixed-base robots
    # ignore the flag (X[0] is a regular joint). Default false keeps every
    # other algorithm (ID/FD/Minv/ABA/IDSVA-SO/integrator/EE-pose) unchanged.
    # RUNTIME_INERTIA (D.4 / Phase 5): when true, the I-region of s_XImats is
    # reconstructed on-device from the mutable d_inertia_params table instead of
    # streamed verbatim from the baked d_XImats. The rebuild is a divide-free
    # scatter of the SAME 10 numbers the baked path stored (Robot.get_inertia_params
    # reads them out of the baked 6x6), so the runtime path is BIT-IDENTICAL to the
    # baked path until set_inertia_params mutates the table. The whole template
    # param + branch is emitted ONLY under self.runtime_inertia so a baked header
    # stays byte-identical (the param doesn't even appear in the signature).
    # RUNTIME_TRANSFORM (runtime_transform): when true the joint-origin block of
    # each X[ind] is rebuilt on-device from the mutable d_transform_params table
    # in a once-per-launch prologue (Xfixed scratch in s_temp) and the hot
    # sin/cos(q) loop loads the origin coefficients from that scratch instead of
    # inline literals (with the GENERAL-rpy DENSE pattern baked so rpy can move).
    # Bit-identical to the baked path until set_transform_params mutates it. The
    # template param + branches are emitted ONLY under self.runtime_transform.
    runtime_inertia = getattr(self, "runtime_inertia", False)
    runtime_transform = getattr(self, "runtime_transform", False)
    tparam_line = "template <typename T, bool SKIP_FLOATING_BASE_X = false"
    if runtime_inertia or runtime_transform:
        # RUNTIME_INERTIA must exist (3rd) so RUNTIME_TRANSFORM (4th) is reachable.
        tparam_line += ", bool RUNTIME_INERTIA = false"
    if runtime_transform:
        tparam_line += ", bool RUNTIME_TRANSFORM = false"
    tparam_line += ">"
    self.gen_add_code_line(tparam_line)
    self.gen_add_code_line("__device__ __forceinline__")
    self.gen_add_code_line(func_def, True)
    # test to see if we need to compute any trig functions
    Xmats = self.robot.get_Xmats_ordered_by_id()
    use_trig = False
    for mat in Xmats:
        if len(mat.atoms(sp.sin, sp.cos)) > 0:
            use_trig = True
            break
    # if trig is needed, compute sin/cos while loading XImats from global to shared
    # D.4 / Phase 5: the I-region of s_XImats lives at [36*n, 36*2*n). Under
    # RUNTIME_INERTIA it is NOT streamed from the baked d_XImats (it is rebuilt
    # below from d_inertia_params); under the default it is copied verbatim.
    i_region_lo = 36 * n
    i_region_hi = 36 * 2 * n
    if use_trig:
        self.gen_add_parallel_loop("ind",str(XI_size))
        if runtime_inertia:
            self.gen_add_code_line("if constexpr (RUNTIME_INERTIA) { if (ind >= " + str(i_region_lo) + " && ind < " + str(i_region_hi) + ") { continue; } }")
        self.gen_add_code_line("s_XImats[ind] = d_robotModel->d_XImats[ind];")
        self.gen_add_end_control_flow()
        if runtime_inertia:
            _emit_runtime_inertia_rebuild(self, n)
        if runtime_transform:
            _emit_runtime_transform_rebuild(self, n)
        if not self.robot.is_serial_chain() or not self.robot.are_Ss_identical(list(range(n))):
            self.gen_add_parallel_loop("ind",str(self.gen_topology_helpers_size()))
            self.gen_add_code_line("s_topology_helpers[ind] = d_robotModel->d_topology_helpers[ind];")
            self.gen_add_end_control_flow()
        if self.robot_has_mimic_joints():
            # Mimic q-fold: build per-BODY effective angle s_q_eff[ind] =
            # alpha_ind * s_q[q_slot(ind)] + offset_ind (mirrors RBDReference's
            # q_for_joint), then compute per-body sin/cos against it. Supports
            # both fixed-base and floating-base (the floating root's quaternion
            # transform reads raw s_q[0..6]; only single-DoF bodies are folded).
            _emit_mimic_q_fold(self)
        else:
            self.gen_add_parallel_loop("k",str(self.robot.get_num_pos()))
            self.gen_add_code_line("s_temp[k] = static_cast<T>(sin(s_q[k]));")
            self.gen_add_code_line("s_temp[k+" + str(self.robot.get_num_pos()) + "] = static_cast<T>(cos(s_q[k]));")
            self.gen_add_end_control_flow()
            self.gen_add_sync()
    # else just load in XI from global to shared efficiently
    else:
        if runtime_inertia:
            # RUNTIME_INERTIA: copy only the X-region [0,36*n) and any Xhom tail
            # [36*2*n, XI_size); rebuild the I-region [36*n,36*2*n) from the
            # param table. Default: one contiguous memcpy_async of the whole XI.
            self.gen_add_code_line("if constexpr (RUNTIME_INERTIA) {", True)
            self.gen_add_code_line("cgrps::memcpy_async(tgrp,s_XImats,d_robotModel->d_XImats," + str(i_region_lo) + ");")
            if XI_size > i_region_hi:
                self.gen_add_code_line("cgrps::memcpy_async(tgrp,s_XImats+" + str(i_region_hi) + ",d_robotModel->d_XImats+" + str(i_region_hi) + "," + str(XI_size - i_region_hi) + ");")
            self.gen_add_end_control_flow()
            self.gen_add_code_line("else {", True)
            self.gen_add_code_line("cgrps::memcpy_async(tgrp,s_XImats,d_robotModel->d_XImats," + str(XI_size) + ");")
            self.gen_add_end_control_flow()
        else:
            self.gen_add_code_line("cgrps::memcpy_async(tgrp,s_XImats,d_robotModel->d_XImats," + str(XI_size) + ");")
        if not self.robot.is_serial_chain() or not self.robot.are_Ss_identical(list(range(n))):
            self.gen_add_code_line("cgrps::memcpy_async(tgrp,s_topology_helpers,d_robotModel->d_topology_helpers," + str(self.gen_topology_helpers_size()) + "*sizeof(int));")
        self.gen_add_code_line("cgrps::wait(tgrp);")
        if runtime_inertia:
            _emit_runtime_inertia_rebuild(self, n)
        if runtime_transform:
            _emit_runtime_transform_rebuild(self, n)
    # loop through Xmats and update all non-constant values serially
    # runtime_transform: source the X cells from the symbolic-Xfixed transforms
    # (dense rpy pattern, origin carried as xf_* symbols) so the origin literals
    # become s_Xfixed scratch loads below; the baked path keeps the folded Xmats.
    if runtime_transform:
        Xmats_serial = self.robot.get_runtime_transform_mats_ordered_by_id()
        xfixed_off = _runtime_transform_xfixed_offset(self)
        xf_cells = self.robot.get_joint_by_id(0)._runtime_xfixed_symbol_cells() \
            if hasattr(self.robot.get_joint_by_id(0), "_runtime_xfixed_symbol_cells") else {}
    else:
        Xmats_serial = Xmats
    self.gen_add_serial_ops()
    for ind in range(n):
        # A.3 lever: skip the floating-base root X[0] block under
        # SKIP_FLOATING_BASE_X. The quat->rot expansion below is the bulk of
        # this serial section; it's dead for CRBA (see function docstring).
        # Includes the X_hom / dX_hom / d2X_hom emits below for ind==0 since
        # CRBA doesn't request hom transforms anyway, and skipping them all
        # together keeps the elision a single contiguous if-constexpr block.
        wrap_skip = self.robot.floating_base and ind == 0
        if wrap_skip:
            self.gen_add_code_line("if constexpr (!SKIP_FLOATING_BASE_X) {", True)
        self.gen_add_code_line("// X[" + str(ind) + "]")
        for col in range(3): # TL and BR are identical so only update TL and BL serially
            for row in range(6):
                val = Xmats_serial[ind][row,col]
                if not self.custom_is_constant(val):
                    # parse the symbolic value into the appropriate array access
                    str_val = str(val)

                    _jt = getattr(self.robot.get_joint_by_id(ind), "jtype", None)
                    if _jt == "spherical":
                        # Tier-C spherical (ball) joint: its transform cell holds
                        # the q1_sph..q4_sph unit-quaternion symbols; substitute the
                        # joint's own 4-wide q-block (NOT the floating-root s_q[3..6]
                        # and NOT a sin/cos fold). A mid-chain spherical shifts every
                        # downstream q-offset, handled inside the helper.
                        str_val = _xi_spherical_quat_subst(self, str_val, ind)
                    elif self.robot.floating_base: # extra dof offset due to floating base
                        num_dof = self.robot.get_num_pos()
                        if self.robot_has_mimic_joints():
                            # Floating + mimic: body `ind`'s joint angle is the
                            # FOLDED per-body effective angle (alpha*q[target]+off)
                            # stored in the s_q_eff scratch (NB | sin | cos), NOT
                            # the legacy s_q[ind+6] (which assumes NJ==nq and is
                            # wrong/OOB once mimic bodies share a q-slot).
                            NB = self.robot.get_num_joints()
                            str_val = str_val.replace("sin(theta)","s_temp[" + str(ind + NB) + "]")
                            str_val = str_val.replace("cos(theta)","s_temp[" + str(ind + 2*NB) + "]")
                            str_val = str_val.replace("theta","s_temp[" + str(ind) + "]")
                        else:
                            # revolute
                            str_val = str_val.replace("sin(theta)","s_temp[" + str(ind + 6) + "]")
                            str_val = str_val.replace("cos(theta)","s_temp[" + str(ind + num_dof + 6) + "]")
                            # then just the variable (prismatic)
                            str_val = str_val.replace("theta","s_q[" + str(ind + 6) + "]")

                        # replace floating base linear & quaternion values (s_q[1..6]: [x,y,z,qx,qy,qz,qw])
                        # replace square with self-multiply
                        str_val = str_val.replace("x_fb**2", "s_q[0]*s_q[0]")
                        str_val = str_val.replace("y_fb**2", "s_q[1]*s_q[1]")
                        str_val = str_val.replace("z_fb**2", "s_q[2]*s_q[2]")
                        str_val = str_val.replace("q1_fb**2", "s_q[3]*s_q[3]")
                        str_val = str_val.replace("q2_fb**2", "s_q[4]*s_q[4]")
                        str_val = str_val.replace("q3_fb**2", "s_q[5]*s_q[5]")
                        str_val = str_val.replace("q4_fb**2", "s_q[6]*s_q[6]")
                        # replace any remaining ones
                        str_val = str_val.replace("x_fb", "s_q[0]")
                        str_val = str_val.replace("y_fb", "s_q[1]")
                        str_val = str_val.replace("z_fb", "s_q[2]")
                        str_val = str_val.replace("q1_fb", "s_q[3]")
                        str_val = str_val.replace("q2_fb", "s_q[4]")
                        str_val = str_val.replace("q3_fb", "s_q[5]")
                        str_val = str_val.replace("q4_fb", "s_q[6]")
                    
                    elif self.robot_has_mimic_joints():
                        # Mimic path: body `ind`'s transform reads the per-body
                        # FOLDED angle and its sin/cos from the s_q_eff layout
                        # (s_q_eff[NB] | sin[NB] | cos[NB]). For the prismatic
                        # ("theta" bare) substitution we use the effective angle
                        # s_temp[ind] (= alpha*q[target]+offset) directly.
                        NB = self.robot.get_num_joints()
                        str_val = str_val.replace("sin(theta)","s_temp[" + str(ind + NB) + "]")
                        str_val = str_val.replace("cos(theta)","s_temp[" + str(ind + 2*NB) + "]")
                        str_val = str_val.replace("theta","s_temp[" + str(ind) + "]")
                    else:
                        # Single-DoF revolute/prismatic. The sincos fill loop
                        # builds s_temp[q]=sin(s_q[q]), s_temp[q+nq]=cos(s_q[q])
                        # over q in [0,nq) (nq = get_num_pos()), so a joint reads
                        # its sin/cos at its OWN q-slot, NOT at `ind`. On an
                        # all-cardinal fixed robot q-slot==ind and nq==n, so this
                        # is byte-identical to the legacy s_temp[ind]/s_temp[ind+n].
                        # On a robot with a multi-DoF (spherical) joint, downstream
                        # single-DoF joints are shifted (nq>NB), so reading `ind`
                        # would grab the wrong angle's sin/cos (the §1e q-slot bug).
                        nq = self.robot.get_num_pos()
                        qslot = self.robot.get_joint_index_q(ind)
                        if isinstance(qslot, (list, tuple)):
                            qslot = qslot[0]
                        str_val = str_val.replace("sin(theta)","s_temp[" + str(qslot) + "]")
                        str_val = str_val.replace("cos(theta)","s_temp[" + str(qslot + nq) + "]")
                        # then just the variable (prismatic)
                        str_val = str_val.replace("theta","s_q[" + str(qslot) + "]")
                    # runtime_transform: replace the joint-origin xf_* symbols with
                    # s_Xfixed scratch loads (rebuilt in the prologue). Only the TL
                    # (xf_TL_*) and BL (xf_BL_*) symbols appear in cols 0-2; BR is the
                    # TL copy below. No-op for the floating/planar/spherical root
                    # (its runtime mat is the baked Xmat_sp, no xf_* symbols).
                    if runtime_transform:
                        for sym, (sr, sc) in xf_cells.items():
                            slot = "s_temp[" + str(xfixed_off + 36*ind + 6*sc + sr) + "]"
                            str_val = str_val.replace(sym, slot)
                    # then output the code
                    cpp_ind = str(self.gen_static_array_ind_3d(ind,col,row))
                    self.gen_add_code_line("s_XImats[" + cpp_ind + "] = static_cast<T>(" + str_val + ");")
        # also replace in homogenous ones
        if (include_homogenous_transforms):
            Xmats_hom = self.robot.get_Xmats_hom_ordered_by_id(include_fixed_joints = self.include_fixed_kinematic_targets)
            dXmats_hom = self.robot.get_dXmats_hom_ordered_by_id()
            d2Xmats_hom = self.robot.get_d2Xmats_hom_ordered_by_id()
            Xhom_size, dXhom_size, d2Xhom_size = self.gen_get_Xhom_size()
            self.gen_add_code_line("// X_hom[" + str(ind) + "]")
            for col in range(4):
                for row in range(4):
                    val = Xmats_hom[ind][row,col]
                    if not self.custom_is_constant(val):
                        # parse the symbolic value into the appropriate array access
                        str_val = sp.ccode(val)
                        # sin/cos (revolute) + theta (prismatic), mimic-aware
                        str_val = _xi_fixed_sincos_subst(self, str_val, ind)
                        # then output the code
                        cpp_ind = str(baseXI_size + self.gen_static_array_ind_3d(ind,col,row,ind_stride=16,col_stride=4))
                        self.gen_add_code_line("s_XImats[" + cpp_ind + "] = static_cast<T>(" + str_val + ");")
            # and gradients
            self.gen_add_code_line("// dX_hom[" + str(ind) + "]")
            for col in range(4):
                for row in range(4):
                    val = dXmats_hom[ind][row,col]
                    if not self.custom_is_constant(val):
                        # parse the symbolic value into the appropriate array access
                        str_val = sp.ccode(val)
                        # sin/cos (revolute) + theta (prismatic), mimic-aware
                        str_val = _xi_fixed_sincos_subst(self, str_val, ind)
                        # then output the code
                        cpp_ind = str(baseXI_size + Xhom_size + self.gen_static_array_ind_3d(ind,col,row,ind_stride=16,col_stride=4))
                        self.gen_add_code_line("s_XImats[" + cpp_ind + "] = static_cast<T>(" + str_val + ");")
            # and 2nd derivatives
            self.gen_add_code_line("// d2X_hom[" + str(ind) + "]")
            for col in range(4):
                for row in range(4):
                    val = d2Xmats_hom[ind][row,col]
                    if not self.custom_is_constant(val):
                        # parse the symbolic value into the appropriate array access
                        str_val = sp.ccode(val)
                        # sin/cos (revolute) + theta (prismatic), mimic-aware
                        str_val = _xi_fixed_sincos_subst(self, str_val, ind)
                        # then output the code
                        cpp_ind = str(baseXI_size + Xhom_size + dXhom_size + self.gen_static_array_ind_3d(ind,col,row,ind_stride=16,col_stride=4))
                        self.gen_add_code_line("s_XImats[" + cpp_ind + "] = static_cast<T>(" + str_val + ");")
        if wrap_skip:
            # close A.3 SKIP_FLOATING_BASE_X if-constexpr block (opened above)
            self.gen_add_end_control_flow()

    # end the serial section
    self.gen_add_end_control_flow()
    self.gen_add_sync()
    # then copy the TL to BR in parallel across all 6x6 X.
    # CRBA also never reads s_XImats[21..35] (X[0] BR block) but the parallel
    # loop is a thin coalesced shared->shared copy (negligible cost) and
    # keeping it joint-uniform avoids per-tier branching; intentionally not
    # gated by SKIP_FLOATING_BASE_X.
    self.gen_add_parallel_loop("kcr",str(9*self.robot.get_num_joints()))
    self.gen_add_code_line("int k = kcr / 9; int cr = kcr % 9; int c = cr / 3; int r = cr % 3;")
    self.gen_add_code_line("int srcInd = k*36 + c*6 + r; int dstInd = srcInd + 21; // 3 more rows and cols")
    self.gen_add_code_line("s_XImats[dstInd] = s_XImats[srcInd];")
    self.gen_add_end_control_flow()
    self.gen_add_sync()
    # add the function end
    self.gen_add_end_function()

def gen_load_update_XmatsHom_helpers_function_call(self, updated_var_names = None, include_gradients = False, include_hessians = False):
    var_names = dict( \
        s_XmatsHom_name = "s_XmatsHom", \
        s_dXmatsHom_name = "s_dXmatsHom", \
        s_d2XmatsHom_name = "s_d2XmatsHom", \
        d_robotModel_name = "d_robotModel", \
        s_q_name = "s_q", \
        s_temp_name = "s_temp", \
        s_topology_helpers_name = "s_topology_helpers", \
    )
    if updated_var_names is not None:
        for key,value in updated_var_names.items():
            var_names[key] = value
    code_start = "load_update_XmatsHom_helpers<T>(" + var_names["s_XmatsHom_name"] + ", "
    code_end = var_names["s_q_name"] + ", " + var_names["d_robotModel_name"] + ", " + var_names["s_temp_name"] + ");"
    n = self.robot.get_num_pos()
    if include_gradients:
        code_start += var_names["s_dXmatsHom_name"] + ", "
    if include_hessians:
        code_start += var_names["s_d2XmatsHom_name"] + ", "
    # Always pass s_topology_helpers (uniform signature; nullptr for serial chains
    # with identical Ss — see load_update_XImats_helpers / gen_declare_shared_arena).
    code_start += var_names["s_topology_helpers_name"] + ", "
    self.gen_add_code_line(code_start + code_end)

def gen_XmatsHom_helpers_temp_shared_memory_code(self, temp_mem_size = 0, include_gradients = False,
                                                 include_hessians = False, extra_t_buffers = None,
                                                 include_dxhom_shared = True,
                                                 include_d2xhom_shared = True,
                                                 include_linalg_scratch = False,
                                                 linalg_scratch_bytes = "GRID_LINALG_NVIDIA_MAX_HELPER_BYTES<T>()",
                                                 arena_base_expr = None,
                                                 tier_workspace_expr = None):
    n = self.robot.get_num_pos()
    Xhom_size, dXhom_size, d2Xhom_size = self.gen_get_Xhom_size()
    if extra_t_buffers is None:
        extra_t_buffers = []
    hom_buffers = [("s_XmatsHom", Xhom_size)]
    if include_gradients and include_dxhom_shared:
        hom_buffers.append(("s_dXmatsHom", dXhom_size))
    if include_hessians and include_d2xhom_shared:
        hom_buffers.append(("s_d2XmatsHom", d2Xhom_size))
    self.gen_declare_shared_arena(extra_t_buffers + hom_buffers, temp_mem_size,
                                  # Always declare s_topology_helpers (nullptr for serial chains with
                                  # identical Ss) so the uniform load_update_XmatsHom_helpers signature
                                  # always has the arg in scope. Mirrors load_update_XImats_helpers.
                                  include_topology_helpers = True,
                                  ximat_name = "",
                                  ximat_size = 0,
                                  temp_name = "s_temp",
                                  topology_name = "s_topology_helpers",
                                  extra_byte_regions = [("s_linalg_smem", linalg_scratch_bytes)] if include_linalg_scratch else None,
                                  arena_base_expr = arena_base_expr,
                                  tier_workspace_expr = tier_workspace_expr)

def gen_load_topology_helpers(self):
    """Emit a standalone ``load_topology_helpers(s_topology_helpers, d_robotModel)``
    device helper that cooperatively fills a caller-provided shared topology buffer
    from the model (then syncs). For external / inline-CUDA callers that drive the
    generated ``*_inner`` functions directly (e.g. a generic multi-robot solver):
    those inners TAKE a pre-filled ``s_topology_helpers``, so the caller needs to
    surface it from ``d_robotModel->d_topology_helpers`` first. Serial chains with
    identical Ss (TOPOLOGY_HELPERS_COUNT == 0) make this a no-op (the loop is empty),
    so it is safe to call unconditionally with a nullptr buffer. Emitted for EVERY
    robot for a uniform interface."""
    count = self.gen_topology_helpers_size()
    self.gen_add_func_doc(
        "Cooperatively fill s_topology_helpers from the model (no-op when "
        "TOPOLOGY_HELPERS_COUNT == 0); for external callers of the *_inner functions",
        [],
        ["s_topology_helpers is the (shared) int buffer to fill (size TOPOLOGY_HELPERS_COUNT; nullptr/unused when 0)",
         "d_robotModel holds d_topology_helpers"],
        None)
    self.gen_add_code_line("template <typename T>")
    self.gen_add_code_line("__device__ __forceinline__")
    self.gen_add_code_line("void load_topology_helpers(int *s_topology_helpers, const robotModel<T> *d_robotModel) {", True)
    self.gen_add_parallel_loop("ind", str(count))
    self.gen_add_code_line("s_topology_helpers[ind] = d_robotModel->d_topology_helpers[ind];")
    self.gen_add_end_control_flow()
    self.gen_add_sync()
    self.gen_add_end_function()

def gen_load_update_XmatsHom_helpers(self, include_base_inertia = False, include_gradients = False, include_hessians = False):
    n = self.robot.get_num_pos()
    NJ = self.robot.get_num_joints()
    Xhom_size, dXhom_size, d2Xhom_size = self.gen_get_Xhom_size()
    baseXI_size = self.gen_get_XI_size(include_base_inertia,include_homogenous_transforms=False) # need to know where the Xhom starts in XI to load from global to shared
    # add function description
    func_def_start = "void load_update_XmatsHom_helpers("
    func_def_middle = "T *s_XmatsHom, "
    func_def_middle2 = "const T *s_q, "
    func_def_end = "const robotModel<T> *d_robotModel, T *s_temp) {"
    func_params = ["s_XmatsHom is the (shared) memory destination location for the XmatsHom",\
        "s_q is the (shared) memory location of the current configuration",\
        "d_robotModel is the pointer to the initialized model specific helpers (XImats, mxfuncs, topology_helpers, etc.)", \
        "s_temp is temporary (shared) memory used to compute sin and cos if needed of size: " + \
                str(self.gen_load_update_XImats_helpers_temp_mem_size())]
    if include_gradients:
        func_params.insert(1,"s_dXmatsHom is the (shared) memory destination location for the dXmatsHom")
        func_def_middle += "T *s_dXmatsHom, "
    if include_hessians:
        func_params.insert(1,"s_d2XmatsHom is the (shared) memory destination location for the d2XmatsHom")
        func_def_middle += "T *s_d2XmatsHom, "
    # Always emit s_topology_helpers for a uniform signature across ALL robots
    # (mirrors load_update_XImats_helpers). Serial chains with identical Ss pass
    # nullptr and skip the topology-copy body below; a generic caller (e.g. a
    # Parallel-DDP-style solver driving many robots through one entry) no longer
    # hits per-robot overload drift. -Wunused-parameter is off in our builds.
    func_def_middle += "int *s_topology_helpers, "
    func_params.insert(-2,"s_topology_helpers is the (shared) memory location for the topology_helpers (nullptr/unused for serial chains with identical Ss)")
    func_def = func_def_start + func_def_middle + func_def_middle2 + func_def_end
    # then genearte the code
    self.gen_add_func_doc("Updates the (d)XmatsHom in (shared) GPU memory acording to the configuration",[],func_params,None)
    self.gen_add_code_line("template <typename T>")
    self.gen_add_code_line("__device__ __forceinline__")
    self.gen_add_code_line(func_def, True)
    # test to see if we need to compute any trig functions
    Xmats_hom = self.robot.get_Xmats_hom_ordered_by_id(include_fixed_joints = self.include_fixed_kinematic_targets)
    use_trig = False
    for mat in Xmats_hom:
        if len(mat.atoms(sp.sin, sp.cos)) > 0:
            use_trig = True
            break
    # if trig is needed, compute sin/cos while loading XImats from global to shared
    if use_trig:
        self.gen_add_parallel_loop("ind",str(Xhom_size))
        self.gen_add_code_line("s_XmatsHom[ind] = d_robotModel->d_XImats[ind+" + str(baseXI_size) + "];")
        self.gen_add_end_control_flow()
        if include_gradients:
            self.gen_add_parallel_loop("ind",str(dXhom_size))
            self.gen_add_code_line("s_dXmatsHom[ind] = d_robotModel->d_XImats[ind+" + str(baseXI_size + Xhom_size) + "];")
            self.gen_add_end_control_flow()
        if include_hessians:
            self.gen_add_parallel_loop("ind",str(d2Xhom_size))
            self.gen_add_code_line("s_d2XmatsHom[ind] = d_robotModel->d_XImats[ind+" + str(baseXI_size + Xhom_size + dXhom_size) + "];")
            self.gen_add_end_control_flow()
        if not self.robot.is_serial_chain() or not self.robot.are_Ss_identical(list(range(n))):
            self.gen_add_parallel_loop("ind",str(self.gen_topology_helpers_size()))
            self.gen_add_code_line("s_topology_helpers[ind] = d_robotModel->d_topology_helpers[ind];")
            self.gen_add_end_control_flow()
        if self.robot_has_mimic_joints():
            # Mimic q-fold (same scheme as gen_load_update_XImats_helpers):
            # s_temp = [s_q_eff(NB) | sin(NB) | cos(NB)] so body `ind`'s Xhom
            # reads its folded angle/sincos. Avoids the OOB s_q[ind>=nq] the
            # legacy per-joint substitution would emit for NB > nq. Supports
            # both fixed-base and floating-base mimic models.
            _emit_mimic_q_fold(self)
        else:
            self.gen_add_parallel_loop("k",str(self.robot.get_num_pos()))
            self.gen_add_code_line("s_temp[k] = static_cast<T>(sin(s_q[k]));")
            self.gen_add_code_line("s_temp[k+" + str(self.robot.get_num_pos()) + "] = static_cast<T>(cos(s_q[k]));")
            self.gen_add_end_control_flow()
            self.gen_add_sync()
    # else just load in XI from global to shared efficiently
    else:
        self.gen_add_parallel_loop("ind",str(Xhom_size))
        self.gen_add_code_line("s_XmatsHom[ind] = d_robotModel->d_XImats[ind+" + str(baseXI_size) + "];")
        self.gen_add_end_control_flow()
        if include_gradients:
            self.gen_add_parallel_loop("ind",str(dXhom_size))
            self.gen_add_code_line("s_dXmatsHom[ind] = d_robotModel->d_XImats[ind+" + str(baseXI_size + Xhom_size) + "];")
            self.gen_add_end_control_flow()
        if include_hessians:
            self.gen_add_parallel_loop("ind",str(d2Xhom_size))
            self.gen_add_code_line("s_d2XmatsHom[ind] = d_robotModel->d_XImats[ind+" + str(baseXI_size + Xhom_size + dXhom_size) + "];")
            self.gen_add_end_control_flow()
        if not self.robot.is_serial_chain() or not self.robot.are_Ss_identical(list(range(n))):
            self.gen_add_parallel_loop("ind",str(self.gen_topology_helpers_size()))
            self.gen_add_code_line("s_topology_helpers[ind] = d_robotModel->d_topology_helpers[ind];")
            self.gen_add_end_control_flow()
        self.gen_add_sync()
    # loop through Xmats and update all non-constant values serially
    def replace_hom_config_symbols(str_val, ind):
        # Tier-C spherical (ball) joint: its homogeneous-transform cell holds the
        # q1_sph..q4_sph unit-quaternion symbols (sp.ccode `pow(qk_sph,2)` form).
        # Substitute the joint's OWN 4-wide q-block (a mid-chain spherical shifts
        # downstream q-offsets, handled inside the helper) -- NOT a sin/cos fold.
        # Byte-identical for non-spherical robots (branch never taken).
        if self.robot.joint_is_spherical(ind):
            return _xi_spherical_quat_subst(self, str_val, ind, ccode=True)
        if self.robot.floating_base:
            if self.robot_has_mimic_joints():
                # Floating + mimic: read the folded effective angle/sincos from
                # the s_q_eff scratch (NB | sin | cos) instead of s_q[ind+6].
                NB = self.robot.get_num_joints()
                str_val = str_val.replace("sin(theta)","s_temp[" + str(ind + NB) + "]")
                str_val = str_val.replace("cos(theta)","s_temp[" + str(ind + 2*NB) + "]")
                str_val = str_val.replace("theta","s_temp[" + str(ind) + "]")
            else:
                str_val = str_val.replace("sin(theta)","s_temp[" + str(ind + 6) + "]")
                str_val = str_val.replace("cos(theta)","s_temp[" + str(ind + n + 6) + "]")
                str_val = str_val.replace("theta","s_q[" + str(ind + 6) + "]")
            str_val = str_val.replace("x_fb**2", "s_q[0]*s_q[0]")
            str_val = str_val.replace("y_fb**2", "s_q[1]*s_q[1]")
            str_val = str_val.replace("z_fb**2", "s_q[2]*s_q[2]")
            str_val = str_val.replace("q1_fb**2", "s_q[3]*s_q[3]")
            str_val = str_val.replace("q2_fb**2", "s_q[4]*s_q[4]")
            str_val = str_val.replace("q3_fb**2", "s_q[5]*s_q[5]")
            str_val = str_val.replace("q4_fb**2", "s_q[6]*s_q[6]")
            str_val = str_val.replace("x_fb", "s_q[0]")
            str_val = str_val.replace("y_fb", "s_q[1]")
            str_val = str_val.replace("z_fb", "s_q[2]")
            str_val = str_val.replace("q1_fb", "s_q[3]")
            str_val = str_val.replace("q2_fb", "s_q[4]")
            str_val = str_val.replace("q3_fb", "s_q[5]")
            str_val = str_val.replace("q4_fb", "s_q[6]")
        elif self.robot_has_mimic_joints():
            # body `ind` reads its folded angle / per-body sincos from the
            # s_q_eff layout (s_q_eff[NB] | sin[NB] | cos[NB]); mirrors the
            # XImats mimic q-fold above.
            NB = self.robot.get_num_joints()
            str_val = str_val.replace("sin(theta)","s_temp[" + str(ind + NB) + "]")
            str_val = str_val.replace("cos(theta)","s_temp[" + str(ind + 2*NB) + "]")
            str_val = str_val.replace("theta","s_temp[" + str(ind) + "]")
        else:
            # Single-DoF revolute/prismatic. The hom sincos fill loop builds
            # s_temp[q]=sin(s_q[q]), s_temp[q+nq]=cos(s_q[q]) over q in [0,nq)
            # (nq = get_num_pos()), so a joint reads its sin/cos at its OWN
            # q-slot, NOT at `ind`. On an all-cardinal fixed robot q-slot==ind
            # and nq==n -> byte-identical to the legacy s_temp[ind]/s_temp[ind+n].
            # On a robot with a multi-DoF (spherical) joint, downstream single-DoF
            # joints are shifted (nq>NJ), so reading `ind` would grab the wrong
            # angle's sin/cos (the §1e q-slot bug; mirrors the XImats fix).
            nq = self.robot.get_num_pos()
            qslot = self.robot.get_joint_index_q(ind)
            if isinstance(qslot, (list, tuple)):
                qslot = qslot[0]
            str_val = str_val.replace("sin(theta)","s_temp[" + str(qslot) + "]")
            str_val = str_val.replace("cos(theta)","s_temp[" + str(qslot + nq) + "]")
            str_val = str_val.replace("theta","s_q[" + str(qslot) + "]")
        return str_val

    # Split the per-matrix updates into separate serial sections (one each
    # for X_hom, dX_hom, d2X_hom) with a __syncthreads() between each.
    # WHY: nvcc allocates registers per-region; one giant `if(tid==0) {...}`
    # containing hundreds of `static_cast<T>(expr)` writes (esp. when the
    # Hessian is included — O(NJ × n²) entries × 16 cells each) pushes
    # peak per-thread register usage above the __launch_bounds__ cap that
    # bigger robots impose (140+ regs vs cap of 128 at MAX_PERF_LEVEL_THREADS=512).
    # Three smaller serial sections drop peak per-thread reg usage to a
    # function of the largest individual matrix group, not the sum.
    # FUTURE: this serial section could be parallelized by fanning each per-matrix
    # block across threads via parallel_loop (let any thread count consume it, not
    # just thread 0).
    self.gen_add_serial_ops()
    for ind in range(NJ):
        self.gen_add_code_line("// X_hom[" + str(ind) + "]")
        for col in range(4):
            for row in range(4):
                val = Xmats_hom[ind][row,col]
                if not self.custom_is_constant(val):
                    # parse the symbolic value into the appropriate array access
                    str_val = replace_hom_config_symbols(sp.ccode(val), ind)
                    # then output the code
                    cpp_ind = str(self.gen_static_array_ind_3d(ind,col,row,ind_stride=16,col_stride=4))
                    self.gen_add_code_line("s_XmatsHom[" + cpp_ind + "] = static_cast<T>(" + str_val + ");")
    self.gen_add_end_control_flow()
    self.gen_add_sync()
    if include_gradients:
        self.gen_add_serial_ops()
        dXmats_hom, dXhom_owners = _global_hom_derivative_matrices_by_q(self)
        for ind in range(n):
            owner_jid = dXhom_owners[ind]
            self.gen_add_code_line("// dX_hom[" + str(ind) + "]")
            for col in range(4):
                for row in range(4):
                    val = dXmats_hom[ind][row,col]
                    if not self.custom_is_constant(val):
                        # parse the symbolic value into the appropriate array access
                        str_val = replace_hom_config_symbols(sp.ccode(val), owner_jid if owner_jid is not None else ind)
                        # then output the code
                        cpp_ind = str(self.gen_static_array_ind_3d(ind,col,row,ind_stride=16,col_stride=4))
                        self.gen_add_code_line("s_dXmatsHom[" + cpp_ind + "] = static_cast<T>(" + str_val + ");")
        self.gen_add_end_control_flow()
        self.gen_add_sync()
    if include_hessians:
        self.gen_add_serial_ops()
        d2Xmats_hom, d2Xhom_owners = _global_hom_second_derivative_matrices(self)
        for ind in range(len(d2Xmats_hom)):
            owner_jid = d2Xhom_owners[ind]
            self.gen_add_code_line("// d2X_hom[" + str(ind) + "]")
            for col in range(4):
                for row in range(4):
                    val = d2Xmats_hom[ind][row,col]
                    if not self.custom_is_constant(val):
                        # parse the symbolic value into the appropriate array access
                        str_val = replace_hom_config_symbols(sp.ccode(val), owner_jid if owner_jid is not None else ind)
                        # then output the code
                        cpp_ind = str(self.gen_static_array_ind_3d(ind,col,row,ind_stride=16,col_stride=4))
                        self.gen_add_code_line("s_d2XmatsHom[" + cpp_ind + "] = static_cast<T>(" + str_val + ");")
        self.gen_add_end_control_flow()
        self.gen_add_sync()
    self.gen_add_end_function()

def gen_topology_helpers_size(self):
    # The topology-helper sections (parent_inds, num_ancestors, num_subtree,
    # running sums, S_inds) are BUILT NJ-wide in gen_init_topology_helpers
    # (n = get_num_joints()), but this size historically used get_num_pos().
    # For a non-mimic fixed-base chain NJ == nq so they agree. For floating
    # non-mimic nq > NJ (the floating root's 7 quat coords inflate nq), so the
    # array was over-allocated-but-correct. For a MIMIC model NJ > nq, so the
    # old nq sizing UNDER-allocates and the NJ-wide build overflows / the device
    # reads the wrong parent/S-index at multi-joint BFS levels (h1_2 fixed AND
    # floating). Size on max(nq, NJ): byte-identical for every non-mimic robot
    # (nq >= NJ there), and never shrinks (so Gate A is preserved), while
    # covering the NJ-wide build for mimic robots. The floating read path is
    # already NJ-consistent (go2/g1 floating build NJ-wide and pass), so a
    # not-under-allocated array is sufficient for the ID value path.
    n = max(self.robot.get_num_pos(), self.robot.get_num_joints())
    size = 0
    if not self.robot.is_serial_chain():
        size += 5*n + 1
    # Preserve the legacy are_Ss_identical query set (get_num_pos()) so the
    # boolean — and thus whether the S_inds section exists at all — is unchanged
    # for non-mimic robots (Gate A byte-identical). Only the allocated WIDTH (n)
    # grows to cover the NJ-wide build for mimic models.
    if not self.robot.are_Ss_identical(list(range(self.robot.get_num_pos()))):
        size += n
    return size

def gen_topology_sparsity_helpers_python(self, INIT_MODE = False):
    NJ = self.robot.get_num_joints()
    n = self.robot.get_num_vel()
    num_ancestors = [len(self.robot.get_ancestors_by_id(jid)) for jid in range(NJ)]
    num_subtree = [len(self.robot.get_subtree_by_id(jid)) for jid in range(NJ)]
    running_sum_num_ancestors = [sum(num_ancestors[0:jid]) for jid in range(NJ+1)] # for the loops that check < jid+1
    running_sum_num_subtree = [sum(num_subtree[0:jid]) for jid in range(NJ)]

    if self.robot.floating_base:
        dva_cols_per_partial = n*NJ
        df_cols_per_partial = n*NJ
    else:
        dva_cols_per_partial = self.robot.get_total_ancestor_count() + NJ
        df_cols_per_partial = self.robot.get_total_ancestor_count() + self.robot.get_total_subtree_count()

    dva_cols_per_jid = [num_ancestors[jid] + 1 for jid in range(NJ)]
    df_cols_per_jid = [num_ancestors[jid] + num_subtree[jid] for jid in range(NJ)]
    df_col_that_is_jid = num_ancestors
    
    running_sum_dva_cols_per_jid = [running_sum_num_ancestors[jid] + jid for jid in range(NJ+1)] # for the loops that check < jid+1
    running_sum_df_cols_per_jid = [running_sum_num_ancestors[jid] + running_sum_num_subtree[jid] for jid in range(NJ)]

    if INIT_MODE:
        return [str(val) for val in num_ancestors], [str(val) for val in num_subtree], \
               [str(val) for val in running_sum_num_ancestors], [str(val) for val in running_sum_num_subtree]
    else:
        return dva_cols_per_partial, dva_cols_per_jid, running_sum_dva_cols_per_jid, \
                df_cols_per_partial,  df_cols_per_jid,  running_sum_df_cols_per_jid,  df_col_that_is_jid

def gen_init_topology_helpers(self):
    """
    S_ind is either the actual index of the 1 in the S matrix, or points to the index in cpp memory.
    It accounts for the floating base offset, so any operations with the floating base S indices will
    require an offset elsewhere in the code
    """
    n = self.robot.get_num_joints()
    if self.robot.is_serial_chain() and self.robot.are_Ss_identical(list(range(n))):
        self.gen_add_code_lines(["//", \
                                 "// Topology Helpers not needed!", \
                                 "//", \
                                 "template <typename T>", \
                                 "__host__", \
                                 "int *init_topology_helpers(){return nullptr;}"])
        return
    # add function description
    self.gen_add_func_doc("Initializes the topology_helpers in GPU memory", \
        [], [],"A pointer to the topology_helpers memory in the GPU")
    # add the function start boilerplate
    self.gen_add_code_line("template <typename T>")
    self.gen_add_code_line("__host__")
    self.gen_add_code_line("int *init_topology_helpers() {", True)
    # add the helpers needed
    code = []
    if not self.robot.is_serial_chain():
        parent_inds = [str(self.robot.get_parent_id(jid)) for jid in range(n)]
        # generate sparsity helpers
        num_ancestors, num_subtree, running_sum_num_ancestors, running_sum_num_subtree = self.gen_topology_sparsity_helpers_python(True)
        _, _, running_sum_dva_cols_per_jid, _, _, running_sum_df_cols_per_jid, _ = self.gen_topology_sparsity_helpers_python()
        code.extend(["int h_topology_helpers[] = {" + ",".join(parent_inds) + ", // parent_inds",
                     "                            " + ",".join(num_ancestors) + ", // num_ancestors",
                     "                            " + ",".join(num_subtree) + ", // num_subtree",
                     "                            " + ",".join(running_sum_num_ancestors) + ", // running_sum_num_ancestors",
                     "                            " + ",".join(running_sum_num_subtree) + "}; // running_sum_num_subtree"])
        if not self.robot.are_Ss_identical(list(range(n))):
            S_inds = self.robot.get_S_inds(n)
            code.insert(-4,"                            " + ",".join(S_inds) + ", // S_inds")
    elif not self.robot.are_Ss_identical(list(range(n))):
            S_inds = self.robot.get_S_inds(n)
            code.append("int h_topology_helpers[] = {" + ",".join(S_inds) + "}; // S_inds")
    self.gen_add_code_lines(code)
    
    # allocate and transfer data to the GPU and return the pointer to the memory
    self.gen_add_code_line("int *d_topology_helpers; gpuErrchk(cudaMalloc((void**)&d_topology_helpers," + str(self.gen_topology_helpers_size()) + "*sizeof(int)));")
    self.gen_add_code_line("gpuErrchk(cudaMemcpy(d_topology_helpers,h_topology_helpers," + str(self.gen_topology_helpers_size()) + "*sizeof(int),cudaMemcpyHostToDevice));")
    self.gen_add_code_line("return d_topology_helpers;")
    self.gen_add_end_function()

def gen_topology_helpers_pointers_for_cpp(self, inds = None, updated_var_names = None, NO_GRAD_FLAG = False, OFFSET = True):
    """
    This function needs to be rewritten for floating base --- Repurcussions extend to all algorithms that support
    floating base, so those algorithms must be modified to support edits. 'OFFSET' input added for now
    """
    var_names = dict(jid_name = "jid", s_topology_helpers_name = "s_topology_helpers")
    if updated_var_names is not None:
        for key,value in updated_var_names.items():
            var_names[key] = value
    n = self.robot.get_num_vel()
    NJ = self.robot.get_num_joints()
    # FIXED-BASE MIMIC: the topology-helper sections are built NJ-wide, but the
    # OFFSET=True section strides below use `n` (= get_num_vel()). For a mimic
    # model NJ > nv, so the device would read parent/S-index at the wrong stride
    # at multi-joint BFS levels. Use NJ as the stride here (the algorithms also
    # iterate over NJ bodies for mimic-fixed). Floating-base keeps the legacy
    # nv stride (its sections + reads are already NJ-consistent via the
    # floating-specific path, and a swap breaks the byte-identical Gate A).
    if self.robot_has_mimic_joints() and not self.robot.floating_base:
        n = NJ
    if inds == None:
        inds = list(range(n))
    IDENTICAL_S_FLAG_INDS = self.robot.are_Ss_identical(inds)
    IDENTICAL_S_FLAG_GLOBAL = self.robot.are_Ss_identical(list(range(n)))

    # check for one ind
    if len(inds) == 1:
        parent_ind = str(self.robot.get_parent_id(inds[0]))
        dva_cols_per_partial, _, running_sum_dva_cols_per_jid, _, _, running_sum_df_cols_per_jid, df_col_that_is_jid = self.gen_topology_sparsity_helpers_python()
        dva_col_offset_for_jid = str(running_sum_dva_cols_per_jid[inds[0]])
        df_col_offset_for_jid = str(running_sum_df_cols_per_jid[inds[0]])
        dva_col_offset_for_parent = str(running_sum_dva_cols_per_jid[self.robot.get_parent_id(inds[0])])
        df_col_offset_for_parent = str(running_sum_df_cols_per_jid[self.robot.get_parent_id(inds[0])])
        dva_col_offset_for_jid_p1 = str(running_sum_dva_cols_per_jid[inds[0] + 1])
        df_col_that_is_jid = str(df_col_that_is_jid[inds[0]])

        if self.robot.floating_base:
            if 0 in inds: S_ind = '-1'
            else: S_ind = str(self.robot.get_S_index_by_id(inds[0]))
    
    # else branch based on type of robot
    else:
    
        # special case for serial chain
        if self.robot.is_serial_chain():
            parent_ind = "(" + var_names["jid_name"] + "-1" + ")"
            dva_col_offset_for_jid = var_names["jid_name"] + "*(" + var_names["jid_name"] + "+1)/2"
            df_col_offset_for_jid = str(n) + "*" + var_names["jid_name"]
            dva_col_offset_for_parent = var_names["jid_name"] + "*(" + var_names["jid_name"] + "-1)/2"
            df_col_offset_for_parent = str(n) + "*(" + var_names["jid_name"] + "-1)"
            dva_col_offset_for_jid_p1 = "(" + var_names["jid_name"] + "+1)*(" + var_names["jid_name"] + "+2)/2"
            df_col_that_is_jid = var_names["jid_name"]
            if not IDENTICAL_S_FLAG_INDS:
                S_id = var_names["s_topology_helpers_name"] + "[" + var_names["jid_name"] + "]"
                S_ind = "((" + S_id + ") > 0 ? (" + S_id + ") - 1 : -(" + S_id + ") - 1)"
    
        # generic robot
        else:
            parent_ind = var_names["s_topology_helpers_name"] + "[" + var_names["jid_name"] + "]"
            if not IDENTICAL_S_FLAG_INDS: # this set of inds can be optimized if all S are the same
                if OFFSET:
                    S_id = var_names["s_topology_helpers_name"] + "[" + str(n) + " + " + var_names["jid_name"] +  "]"
                else: S_id = var_names["s_topology_helpers_name"] + "[" + str(NJ) + " + " + var_names["jid_name"] +  "]"
                S_ind = "((" + S_id + ") > 0 ? (" + S_id + ") - 1 : -(" + S_id + ") - 1)"
            if not IDENTICAL_S_FLAG_GLOBAL: # ofset is based on any S different at all
                if OFFSET: ancestor_offset = 2*n
                else: ancestor_offset = NJ+n
            else:
                ancestor_offset = NJ
            
            if OFFSET:
                subtree_offset = ancestor_offset + n
                running_sum_ancestor_offset = subtree_offset + n
                running_sum_subtree_offset = running_sum_ancestor_offset + n + 1
            else:
                subtree_offset = ancestor_offset + NJ
                running_sum_ancestor_offset = subtree_offset + NJ
                running_sum_subtree_offset = running_sum_ancestor_offset + NJ + 1

            dva_col_offset_for_jid = "(" + var_names["s_topology_helpers_name"] + "[" + str(running_sum_ancestor_offset) + " + " + var_names["jid_name"] + "]" + \
                                     " + " + var_names["jid_name"] + ")"
            df_col_offset_for_jid = "(" + var_names["s_topology_helpers_name"] + "[" + str(running_sum_ancestor_offset) + " + " + var_names["jid_name"] + "]" + \
                                    " + " + var_names["s_topology_helpers_name"] + "[" + str(running_sum_subtree_offset) + " + " + var_names["jid_name"] + "])"

            dva_col_offset_for_parent = "(" + var_names["s_topology_helpers_name"] + "[" + str(running_sum_ancestor_offset) + " + " + parent_ind + "]" + \
                                        " + " + parent_ind + ")"
            df_col_offset_for_parent = "(" + var_names["s_topology_helpers_name"] + "[" + str(running_sum_ancestor_offset) + " + " + parent_ind + "]" + \
                                       " + " + var_names["s_topology_helpers_name"] + "[" + str(running_sum_subtree_offset) + " + " + parent_ind + "])"

            dva_col_offset_for_jid_p1 = "(" + var_names["s_topology_helpers_name"] + "[" + str(running_sum_ancestor_offset) + " + " + var_names["jid_name"] + " + 1]" + \
                                        " + " + var_names["jid_name"] + " + 1)"

            df_col_that_is_jid = var_names["s_topology_helpers_name"] + "[" + str(ancestor_offset) + " + " + var_names["jid_name"] + "]"

    if IDENTICAL_S_FLAG_INDS: # always true for one ind
        # Tier-B (skew) joints have no signed unit index. Callers that need S
        # for a skew joint (RNEA forward/c-extract) take a Tier-B branch BEFORE
        # consuming this; the S-independent callers (backward f-update) just
        # discard S_ind. Emit a "0" placeholder so those S-free uses don't trip
        # the cardinal guard. Cardinal robots are unaffected (byte-identical).
        if self.robot.S_is_cardinal_by_id(inds[0]):
            S_ind = str(self.robot.get_S_index_by_id(inds[0]))
        else:
            S_ind = "0"

    if NO_GRAD_FLAG:
        return parent_ind, S_ind
    else:
        return parent_ind, S_ind, dva_col_offset_for_jid, df_col_offset_for_jid, dva_col_offset_for_parent, df_col_offset_for_parent, dva_col_offset_for_jid_p1, df_col_that_is_jid

def gen_topology_S_sign_for_cpp(self, inds = None, updated_var_names = None, OFFSET = True):
    var_names = dict(jid_name = "jid", s_topology_helpers_name = "s_topology_helpers")
    if updated_var_names is not None:
        for key,value in updated_var_names.items():
            var_names[key] = value
    n = self.robot.get_num_vel()
    NJ = self.robot.get_num_joints()
    # FIXED-BASE MIMIC: S_inds section is NJ-wide; use NJ stride (see the matching
    # note in gen_topology_helpers_pointers_for_cpp).
    if self.robot_has_mimic_joints() and not self.robot.floating_base:
        n = NJ
    if inds == None:
        inds = list(range(n))

    if self.robot.floating_base and len(inds) == 1 and inds[0] != 0:
        return str(self.robot.get_S_sign_by_id(inds[0]))

    if self.robot.are_Ss_identical(inds):
        return str(self.robot.get_S_sign_by_id(inds[0]))

    if self.robot.is_serial_chain():
        S_id = var_names["s_topology_helpers_name"] + "[" + var_names["jid_name"] + "]"
        return "((" + S_id + ") > 0 ? 1 : -1)"

    if OFFSET:
        S_id = var_names["s_topology_helpers_name"] + "[" + str(n) + " + " + var_names["jid_name"] + "]"
    else:
        S_id = var_names["s_topology_helpers_name"] + "[" + str(NJ) + " + " + var_names["jid_name"] + "]"
    return "((" + S_id + ") > 0 ? 1 : -1)"

def gen_insert_helpers_function_call(self, updated_var_names = None, NO_XI_FLAG = False):
    # Canonical builder for the shared helper ARGS of an inner-function call. Mirror
    # of gen_insert_helpers_func_def_params so def + call can never drift: pass
    # NO_XI_FLAG=True for inners that take s_Xhom (homogeneous transforms) instead of
    # s_XImats (e.g. the end_effector_pose family) — same as the def helper.
    var_names = dict( \
        s_XImats_name = "s_XImats", \
        s_topology_helpers_name = "s_topology_helpers", \
    )
    if updated_var_names is not None:
        for key,value in updated_var_names.items():
            var_names[key] = value
    func_call = ""
    if not NO_XI_FLAG:
        func_call += var_names["s_XImats_name"] + ", "
    # Always pass s_topology_helpers for a uniform inner-function signature across
    # robots; it is nullptr (and unused) for serial chains with identical Ss, where
    # the topology is hardcoded into the generated indices.
    func_call += var_names["s_topology_helpers_name"] + ", "
    return func_call

def gen_insert_helpers_func_def_params(self, func_def, func_params, param_insert_position = -1, updated_var_names = None, NO_XI_FLAG = False):
    n = self.robot.get_num_pos()
    var_names = dict( \
        s_XImats_name = "s_XImats", \
        s_topology_helpers_name = "s_topology_helpers", \
    )
    if updated_var_names is not None:
        for key,value in updated_var_names.items():
            var_names[key] = value
    n = self.robot.get_num_pos()
    if not NO_XI_FLAG:
        func_def += "T *" + var_names["s_XImats_name"] + ", "
        func_params.insert(param_insert_position,"s_XImats is the (shared) memory holding the updated XI matricies for the given s_q")
    # Always emit s_topology_helpers for a uniform signature across robots. It is
    # nullptr and unused for serial chains with identical Ss (they hardcode the
    # topology into the generated indices); -Wunused-parameter is off in our builds.
    func_def += "int *" + var_names["s_topology_helpers_name"] + ", "
    func_params.insert(param_insert_position,"s_topology_helpers is the (shared) memory location for the topology_helpers (nullptr/unused for serial chains with identical Ss)")
    return func_def, func_params

def gen_init_robotModel(self):
    self.gen_add_func_doc("Initializes the robotModel helpers in GPU memory", \
                           [], [],"A pointer to the robotModel struct")
    # add the function start boilerplate
    self.gen_add_code_line("template <typename T>")
    self.gen_add_code_line("__host__")
    self.gen_add_code_line("robotModel<T>* init_robotModel() {", True)
    # then construct the host side struct
    init_lines = ["robotModel<T> h_robotModel;", \
                  "h_robotModel.d_XImats = init_XImats<T>();", \
                  "h_robotModel.d_topology_helpers = init_topology_helpers<T>();"]
    if getattr(self, "runtime_inertia", False):
        # D.4 / Phase 5: flag-gated mutable inertia table init.
        init_lines.append("h_robotModel.d_inertia_params = init_inertia_params<T>();")
    if getattr(self, "runtime_transform", False):
        # runtime_transform: flag-gated mutable joint-origin table init.
        init_lines.append("h_robotModel.d_transform_params = init_transform_params<T>();")
    if getattr(self, "runtime_joint_dynamics", False):
        # runtime_joint_dynamics: flag-gated mutable damping/friction table init.
        init_lines.append("h_robotModel.d_joint_dynamics_params = init_joint_dynamics_params<T>();")
    self.gen_add_code_lines(init_lines)
    # then allocate memeory and copy to device
    self.gen_add_code_lines(["robotModel<T> *d_robotModel; gpuErrchk(cudaMalloc((void**)&d_robotModel,sizeof(robotModel<T>)));",
                             "gpuErrchk(cudaMemcpy(d_robotModel,&h_robotModel,sizeof(robotModel<T>),cudaMemcpyHostToDevice));"])
    self.gen_add_code_line("return d_robotModel;")
    self.gen_add_end_function()

def gen_free_robotModel(self):
    self.gen_add_func_doc(
        "Frees a robotModel allocated by init_robotModel: the NESTED device arrays "
        "(d_XImats / d_topology_helpers [+ any flag-gated runtime parameter tables]) AND the "
        "struct itself. A bare cudaFree(d_robotModel) frees ONLY the struct and leaks the nested "
        "arrays; this recovers them by copying the struct back to host first.",
        [], ["d_robotModel is a pointer returned by init_robotModel"], None)
    self.gen_add_code_line("template <typename T>")
    self.gen_add_code_line("__host__")
    self.gen_add_code_line("void free_robotModel(robotModel<T> *d_robotModel) {", True)
    # Copy the struct back to host to recover the nested device pointers, then free each.
    self.gen_add_code_line("robotModel<T> h_robotModel;")
    self.gen_add_code_line("gpuErrchk(cudaMemcpy(&h_robotModel, d_robotModel, sizeof(robotModel<T>), cudaMemcpyDeviceToHost));")
    free_lines = ["gpuErrchk(cudaFree(h_robotModel.d_XImats));",
                  "gpuErrchk(cudaFree(h_robotModel.d_topology_helpers));"]
    if getattr(self, "runtime_inertia", False):
        free_lines.append("gpuErrchk(cudaFree(h_robotModel.d_inertia_params));")
    if getattr(self, "runtime_transform", False):
        free_lines.append("gpuErrchk(cudaFree(h_robotModel.d_transform_params));")
    if getattr(self, "runtime_joint_dynamics", False):
        free_lines.append("gpuErrchk(cudaFree(h_robotModel.d_joint_dynamics_params));")
    self.gen_add_code_lines(free_lines)
    self.gen_add_code_line("gpuErrchk(cudaFree(d_robotModel));")
    self.gen_add_end_function()

def gen_joint_limits_size(self):
    n = self.robot.get_num_pos()
    return 2 * n

def gen_init_joint_limits(self):
    n = self.robot.get_num_pos()

    limits_q_order = []
    for j in self.robot.get_joints_ordered_by_id():
        jt = j.get_type() if hasattr(j, "get_type") else j.jtype
        if jt == "revolute":
            lims = j.get_joint_limits() if hasattr(j, "get_joint_limits") else getattr(j, "joint_limits", [])
            if lims and len(lims) >= 2:
                lo, hi = lims[0], lims[1]
            else:
                lo, hi = -float("inf"), float("inf")
            if lo is None: lo = -float("inf")
            if hi is None: hi =  float("inf")
            limits_q_order.append((lo, hi))

    self.gen_add_func_doc(
        "Initializes joint limits (lower/upper) in GPU memory",
        ["Memory order is lower[0..n-1], upper[0..n-1]"],
        [],
        "A device pointer to the joint limits array"
    )
    self.gen_add_code_line("template <typename T>")
    self.gen_add_code_line("__host__")
    self.gen_add_code_line("T* init_joint_limits() {", True)

    total_size = self.gen_joint_limits_size()
    self.gen_add_code_line(f"T *h_joint_limits = (T*)malloc({total_size}*sizeof(T));")

    for i, (lo, hi) in enumerate(limits_q_order):
        lo_str = ("-std::numeric_limits<T>::infinity()" if (lo is None or lo == -float('inf'))
                  else f"static_cast<T>({lo})")
        hi_str = (" std::numeric_limits<T>::infinity()" if (hi is None or hi ==  float('inf'))
                  else f"static_cast<T>({hi})")
        self.gen_add_code_line(f"h_joint_limits[{i}] = {lo_str};")
        self.gen_add_code_line(f"h_joint_limits[{i + n}] = {hi_str};")

    self.gen_add_code_line("T *d_joint_limits;")
    self.gen_add_code_line(f"gpuErrchk(cudaMalloc((void**)&d_joint_limits, {total_size}*sizeof(T)));")
    self.gen_add_code_line(f"gpuErrchk(cudaMemcpy(d_joint_limits, h_joint_limits, {total_size}*sizeof(T), cudaMemcpyHostToDevice));")
    self.gen_add_code_line("free(h_joint_limits);")
    self.gen_add_code_line("return d_joint_limits;")
    self.gen_add_end_function()
