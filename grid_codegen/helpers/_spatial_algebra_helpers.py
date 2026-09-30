def gen_mx_func_call_for_cpp(self, inds = None, PEQ_FLAG = False, SCALE_FLAG = False, updated_var_names = None):
    var_names = dict(S_ind_name = "S_ind", s_dst_name = "s_dst", s_src_name = "s_src", s_scale_name = "s_scale")
    if updated_var_names is not None:
        for key,value in updated_var_names.items():
            var_names[key] = value
    n = self.robot.get_num_pos()
    if inds == None:
        inds = list(range(n))
    IDENTICAL_S_FLAG_INDS = self.robot.are_Ss_identical(inds)

    # check for all the same mxFunc
    if IDENTICAL_S_FLAG_INDS:
        S_ind = str(self.robot.get_S_index_by_id(inds[0]))
    else:
        S_ind = "X"
    # find which function type
    if not SCALE_FLAG and not PEQ_FLAG:
        func_name = "mx" + S_ind + "<T>"
    elif not SCALE_FLAG:
        func_name = "mx" + S_ind + "_peq<T>"
    elif not PEQ_FLAG:
        func_name = "mx" + S_ind + "_scaled<T>"
    else:
        func_name = "mx" + S_ind + "_peq_scaled<T>"
    # then make the code line
    code_start = func_name + "(" + var_names["s_dst_name"] + ", " + var_names["s_src_name"]
    code_middle = ""
    if SCALE_FLAG:
        code_middle += ", " + var_names["s_scale_name"]
    if not IDENTICAL_S_FLAG_INDS:
        code_middle += ", " + var_names["S_ind_name"]
    code_end = ");"
    self.gen_add_code_line(code_start + code_middle + code_end)

def gen_crm_mul(self):
    """
    This function generates the code for the 
    motion cross product matrix multiplication function.
    It multiplies a cross product matrix by a vector.
    It returns the result of the entry at the index.
    """
    self.gen_add_func_doc("Compute the motion cross product multiplication of a 6-vector, v_crm, with a second 6-vector v", [], \
                          ['index is the index of the result vector to compute', \
                            'v_crm is the 6-vector to take the cross product matrix of', \
                           'v is the 6-vector to multiply with v_crm'])
    self.gen_add_code_line("template <typename T>")
    self.gen_add_code_line("__device__")
    self.gen_add_code_line("T crm_mul(int index, T *v_crm, T *v) {", True)
    # Row `index` of motion_cross(v_crm)*v -- the GLASS spatial primitive (formulas
    # promoted verbatim from this emitter). Scalar, single-thread, no barrier.
    self.gen_add_code_line("return glass::spatial_detail::motion_cross_mul_row<T>((uint32_t)index, v_crm, v);")
    self.gen_add_end_function()


def gen_mxS_general(self):
    """Tier-B (skew/general axis) motion cross product against a DENSE S column.

    For a cardinal axis the codegen uses the specialized mx0..mx5 columns; a
    skew joint's motion subspace S is a dense 6-vector, so mxS(v) = crm(v) * S
    cannot pick a precomputed column. This generic helper computes
        s_vecX += alpha * (crm(s_vec) * S)
    reusing crm_mul (row r of crm(v)*x). Emitted ONLY for models with a skew
    axis (gated by robot_has_skew_axis), so all-cardinal headers are unchanged.
    """
    self.gen_add_func_doc(
        "Adds alpha*(crm(s_vec)*S) into s_vecX for a DENSE motion subspace column S",
        ["Assumes only one thread is running each function call", "Tier-B skew-axis helper"],
        ["s_vecX is the destination 6-vector", "s_vec is the source 6-vector",
         "S is the dense 6-vector motion subspace column", "alpha is the scaling factor"],
        None)
    self.gen_add_code_line("template <typename T>")
    self.gen_add_code_line("__device__")
    self.gen_add_code_line("void mxS_general_peq_scaled(T *s_vecX, const T *s_vec, const T *S, const T alpha) {", True)
    # s_vecX += alpha*(motion_cross(s_vec)*S) via the fused GLASS thread op
    # (dense column S => AXIS=-1, HAS_BETA=true with beta=1 gives the += accumulate).
    self.gen_add_code_line("glass::thread::motion_cross_mul<T, -1, true>(alpha, s_vec, S, static_cast<T>(1), s_vecX);")
    self.gen_add_end_function()


def gen_crm(self):
    """Entry `index` (flat column-major) of the motion cross product matrix
    motion_cross(v). Delegated to the GLASS spatial primitive (verbatim formulas);
    consumed element-wise by the block-parallel crm/crf matrix builds in idsva_so."""
    self.gen_add_func_doc("Compute the motion cross product matrix of a 6-vector, v Returns the entry at the index.", \
                          ['The force cross product matrix is just the negative transpose of this matrix'], \
                          ['index is the index of the result matirx to compute', \
                           'v is the 6-vector to take the cross product matrix of'])
    self.gen_add_code_line("template <typename T>")
    self.gen_add_code_line("__device__")
    self.gen_add_code_line("T crm(int index, T *v) {", True)
    self.gen_add_code_line("return glass::spatial_detail::motion_cross_entry<T>((uint32_t)(index % 6), (uint32_t)(index / 6), v);")
    self.gen_add_end_function()


def gen_spatial_algebra_helpers(self):
    # First function -- add dot product code with and without const
    for i in range(4):
        self.gen_add_func_doc("Compute the dot product between two vectors", ["Assumes computed by a single thread"], \
                              ["vec1 is the first vector of length N with stride S1", "vec2 is the second vector of length N with stride S2"], \
                              "the resulting final value")
        self.gen_add_code_line("template <typename T, int N, int S1, int S2>")
        self.gen_add_code_line("__device__")
        if i == 0:
            self.gen_add_code_line("T dot_prod(const T *vec1, const T *vec2) {", True)
        elif i == 1:
            self.gen_add_code_line("T dot_prod(T *vec1, const T *vec2) {", True)
        elif i == 2:
            self.gen_add_code_line("T dot_prod(const T *vec1, T *vec2) {", True)
        else:
            self.gen_add_code_line("T dot_prod(T *vec1, T *vec2) {", True)
        self.gen_add_code_line("return glass::dot_strided<T, N, S1, S2>(vec1, vec2);")
        self.gen_add_end_function()

    # Then the motion vector matrix cross product operations: each mx{k} is the
    # cardinal-axis column k of motion_cross(s_vec) (i.e. motion_cross(s_vec)*e_k).
    # Delegated to the GLASS thread op (single-thread, no barrier; formulas promoted
    # verbatim from this emitter). The (alpha, beta, HAS_BETA) triple encodes the
    # four variants: plain=(1,0,false), _peq=(1,1,true), _scaled=(a,0,false),
    # _peq_scaled=(a,1,true). x is ignored for AXIS>=0, so pass nullptr.
    def _mx_body(k, has_beta, alpha, beta):
        return ("glass::thread::motion_cross_mul<T, " + str(k) + ", "
                + ("true" if has_beta else "false") + ">(" + alpha
                + ", s_vec, nullptr, " + beta + ", s_vecX);")
    for ind in range(6):
        # without alpha
        self.gen_add_func_doc("Generates the motion vector cross product matrix column " + str(ind),\
                             ["Assumes only one thread is running each function call"],\
                             ["s_vecX is the destination vector","s_vec is the source vector"],None)
        self.gen_add_code_line("template <typename T>")
        self.gen_add_code_line("__device__")
        self.gen_add_code_line("void mx" + str(ind) + "(T *s_vecX, const T *s_vec) {", True)
        self.gen_add_code_line(_mx_body(ind, False, "static_cast<T>(1)", "static_cast<T>(0)"))
        self.gen_add_end_function()
        # without alpha PEQ
        self.gen_add_func_doc("Adds the motion vector cross product matrix column " + str(ind),\
                             ["Assumes only one thread is running each function call"],\
                             ["s_vecX is the destination vector","s_vec is the source vector"],None)
        self.gen_add_code_line("template <typename T>")
        self.gen_add_code_line("__device__")
        self.gen_add_code_line("void mx" + str(ind) + "_peq(T *s_vecX, const T *s_vec) {", True)
        self.gen_add_code_line(_mx_body(ind, True, "static_cast<T>(1)", "static_cast<T>(1)"))
        self.gen_add_end_function()
        # with alpha
        self.gen_add_func_doc("Generates the motion vector cross product matrix column " + str(ind),\
                             ["Assumes only one thread is running each function call"],\
                             ["s_vecX is the destination vector","s_vec is the source vector","alpha is the scaling factor"],None)
        self.gen_add_code_line("template <typename T>")
        self.gen_add_code_line("__device__")
        self.gen_add_code_line("void mx" + str(ind) + "_scaled(T *s_vecX, const T *s_vec, const T alpha) {", True)
        self.gen_add_code_line(_mx_body(ind, False, "alpha", "static_cast<T>(0)"))
        self.gen_add_end_function()
        # with alpha PEQ
        self.gen_add_func_doc("Adds the motion vector cross product matrix column " + str(ind),\
                             ["Assumes only one thread is running each function call"],\
                             ["s_vecX is the destination vector","s_vec is the source vector","alpha is the scaling factor"],None)
        self.gen_add_code_line("template <typename T>")
        self.gen_add_code_line("__device__")
        self.gen_add_code_line("void mx" + str(ind) + "_peq_scaled(T *s_vecX, const T *s_vec, const T alpha) {", True)
        self.gen_add_code_line(_mx_body(ind, True, "alpha", "static_cast<T>(1)"))
        self.gen_add_end_function()
    # then the generics with a switch statement
    # without alpha
    func_params = ["s_vecX is the destination vector","s_vec is the source vector"]
    func_call_params = ["T *s_vecX", "const T *s_vec", "const int S_ind"]
    subFunc_call_params = ["s_vecX", "s_vec"]
    for i in range(4):
        func_name_suffix = ("_peq" if i % 2 else "") + ("_scaled" if i > 1 else "")
        if i == 2: # add the alpha in once we start doing the scaled ones
            func_params.append("alpha is the scaling factor")
            func_call_params.insert(-1,"const T alpha")
            subFunc_call_params.append("alpha")
        self.gen_add_func_doc("Generates the motion vector cross product matrix for a runtime selected column",\
                             ["Assumes only one thread is running each function call"],func_params,None)
        self.gen_add_code_line("template <typename T>")
        self.gen_add_code_line("__device__")
        self.gen_add_code_line("void mxX" + func_name_suffix + "(" + ", ".join(func_call_params) + ") {", True)
        self.gen_add_code_line("switch(S_ind){", True)
        for ind in range(6):
            self.gen_add_code_line("case " + str(ind) + ": mx" + str(ind) + func_name_suffix + "<T>(" + ", ".join(subFunc_call_params) + "); break;")
        self.gen_add_end_control_flow()
        self.gen_add_end_function()

    # Force cross apply: fx_times_v(r, fxVec, timesVec) = force_cross(fxVec)*timesVec.
    # Delegated to the GLASS thread op (single-thread, no barrier; force_cross_mul
    # rows promoted verbatim from this emitter). HAS_BETA=false overwrites; the _peq
    # sibling uses HAS_BETA=true, beta=1 for the += accumulate.
    self.gen_add_func_doc("Generates the force cross product matrix and multiplies by the input vector",\
                         ["Assumes only one thread is running each function call"],\
                         ["s_result is the result vector","s_fxVec is the fx vector","s_timesVec is the multipled vector"],None)
    self.gen_add_code_line("template <typename T>")
    self.gen_add_code_line("__device__")
    self.gen_add_code_line("void fx_times_v(T *s_result, const T *s_fxVec, const T *s_timesVec) {", True)
    self.gen_add_code_line("glass::thread::force_cross_mul<T, false>(static_cast<T>(1), s_fxVec, s_timesVec, static_cast<T>(0), s_result);")
    self.gen_add_end_function()
    self.gen_add_func_doc("Adds the force cross product matrix multiplied by the input vector",\
                         ["Assumes only one thread is running each function call"],\
                         ["s_result is the result vector","s_fxVec is the fx vector","s_timesVec is the multipled vector"],None)
    self.gen_add_code_line("template <typename T>")
    self.gen_add_code_line("__device__")
    self.gen_add_code_line("void fx_times_v_peq(T *s_result, const T *s_fxVec, const T *s_timesVec) {", True)
    self.gen_add_code_line("glass::thread::force_cross_mul<T, true>(static_cast<T>(1), s_fxVec, s_timesVec, static_cast<T>(1), s_result);")
    self.gen_add_end_function()

    # vcross: full 6x6 motion cross matrix (== crm(v)), consumed by coriolis' bias
    # build. Delegated to the GLASS thread op (single-thread, all 36 entries).
    self.gen_add_func_doc("Generates the full motion vector cross product matrix (6x6, column-major)",\
                         ["Assumes only one thread is running each function call"],\
                         ["dest is the destination 6x6 matrix","v is the source 6-vector"],None)
    self.gen_add_code_line("template <typename T>")
    self.gen_add_code_line("__device__")
    self.gen_add_code_line("void vcross(T *dest, T *v){", True)
    self.gen_add_code_line("glass::thread::motion_cross(v, dest);")
    self.gen_add_end_function()

    # icrf(index, v): entry `index` (flat column-major) of the inverse force cross
    # product matrix (v crf f == f icrf v). GLASS force_cross_dual bakes in the
    # global negative, so return its entry directly. Consumed element-wise by the
    # block-parallel BC/T3 builds in idsva_so.
    self.gen_add_func_doc("Compute the inverse force cross product matrix of a 6-vector, v Returns the entry at the index.", \
                          ['ICRF is the operation defined such that v crf f = f icrf v'], \
                          ['index is the index of the result matirx to compute', \
                           'v is the 6-vector to take the cross product matrix of'])
    self.gen_add_code_line("template <typename T>")
    self.gen_add_code_line("__device__")
    self.gen_add_code_line("T icrf(int index, T *v) {", True)
    self.gen_add_code_line("return glass::spatial_detail::force_cross_dual_entry<T>((uint32_t)(index % 6), (uint32_t)(index / 6), v);")
    self.gen_add_end_function()

    
                            
