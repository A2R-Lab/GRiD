"""gpuErrchk / sticky-error emission (H4 move from GRiDCodeGenerator.py,
2026-08-27, verbatim)."""


def gen_add_gpu_err(self):
    # add the GPU error check code
    self.gen_add_func_doc("Check for runtime errors using the CUDA API", \
            ["Adapted from https://stackoverflow.com/questions/14038589/what-is-the-canonical-way-to-check-for-errors-using-the-cuda-runtime-api"], \
            [],None)
    # Sticky first-error slot + accessors. Emitted in BOTH modes so embedders can
    # always compile against grid_last_error()/grid_consume_last_error(); in the
    # default (fail-fast) mode the slot simply never becomes nonzero because
    # gpuAssert exits first. `inline` + function-local static = one process-wide
    # copy across the per-algo TU split (ODR), same reason gpuAssert is inline.
    self.gen_add_code_line("__host__ inline cudaError_t* grid_last_error_slot(){ static cudaError_t e = cudaSuccess; return &e; }")
    self.gen_add_code_line("__host__ inline cudaError_t grid_last_error(){ return *grid_last_error_slot(); }")
    self.gen_add_code_line("__host__ inline cudaError_t grid_consume_last_error(){ cudaError_t e = *grid_last_error_slot(); *grid_last_error_slot() = cudaSuccess; return e; }")
    # Default mode is FAIL-FAST (exit) — every direct consumer (bench runners,
    # examples, GATO/MPCGPU-style embedders) relies on it. Library embedders that
    # must survive a CUDA error (the Python bindings: exit(code) inside a dlopened
    # .so kills the host interpreter with no traceback) opt in with
    # -DGRID_GPUERRCHK_NO_EXIT: gpuAssert then records the FIRST error in the
    # sticky slot and returns; the caller checks grid_consume_last_error() at its
    # ABI boundary. Not atomic — concurrent host threads may race the first-error
    # pick, but never lose error-ness (slot only moves away from cudaSuccess).
    self.gen_add_code_line("__host__")
    # `inline` is required so the per-algo TU split (multiple .o files all
    # including grid.cuh) doesn't trip ODR multiple-definition errors at link.
    self.gen_add_code_line("inline void gpuAssert(cudaError_t code, const char *file, const int line, bool abort=true){", True)
    self.gen_add_code_line("if (code != cudaSuccess){", True)
    # note that below we need to escape the \n and "" to get it to print to a string or file correctly
    self.gen_add_code_line("fprintf(stderr,\"GPUassert: %s %s %d\\n\", cudaGetErrorString(code), file, line);")
    self.gen_add_code_line("#ifdef GRID_GPUERRCHK_NO_EXIT")
    self.gen_add_code_line("if (abort && *grid_last_error_slot() == cudaSuccess){ *grid_last_error_slot() = code; }")
    self.gen_add_code_line("#else")
    self.gen_add_code_line("if (abort){cudaDeviceReset(); exit(code);}")
    self.gen_add_code_line("#endif")
    self.gen_add_end_control_flow()
    self.gen_add_end_control_flow() # end of function but don't want spacing
    # #ifndef-guarded so a consumer that defines its own gpuErrchk BEFORE including
    # grid.cuh keeps theirs (previously: silent redefinition collision).
    self.gen_add_code_line("#ifndef gpuErrchk")
    self.gen_add_code_line("#define gpuErrchk(err) {gpuAssert(err, __FILE__, __LINE__);}")
    self.gen_add_code_line("#endif")
    # gpuErrchkKernel catches BOTH (a) synchronous launch-time errors via
    # cudaPeekAtLastError — e.g. cudaErrorLaunchOutOfResources (code 701)
    # when the kernel asks for more registers than the SM can give — and
    # (b) asynchronous execution-time errors via cudaDeviceSynchronize.
    # Use this after every <<<>>> kernel launch. Plain cudaDeviceSynchronize
    # alone does NOT propagate launch-time errors: a launch can fail before
    # work is queued, leaving the stream empty, so sync returns success and
    # the next call clears the error. That's why the overnight bench was
    # silently reporting failed launches as ~2us "compute time."
    self.gen_add_code_line("#ifndef gpuErrchkKernel")
    self.gen_add_code_line("#define gpuErrchkKernel() {gpuErrchk(cudaPeekAtLastError()); gpuErrchk(cudaDeviceSynchronize());}")
    self.gen_add_code_line("#endif")
    self.gen_add_code_line("")
    self.gen_library_safe_init_contract()

def gen_library_safe_init_contract(self):
    """Library-safe (nonterminating) initialization contract (2026-09-22,
    HJCD ask): the `*_checked` initializers/destructor emitted by
    _topology_helpers return cudaError_t, name the failed operation, publish
    ownership only on complete success and roll back everything acquired by a
    failed attempt without exit/abort/cudaDeviceReset. The legacy
    `init_*()` / `free_robotModel()` spellings are thin wrappers that apply the
    historical policy (fail-fast, or sticky slot under GRID_GPUERRCHK_NO_EXIT).
    GRID_CUDA_CALL / GRID_HOST_ALLOC are HOST-ONLY fault-injection seams: a
    test TU defines them before including grid.cuh; the default expands to the
    bare expression (no runtime cost, never used in device code)."""
    self.gen_add_code_line("// ─── library-safe initialization contract (init_*_checked / free_robotModel_checked) ───")
    self.gen_add_code_line("// Host-only fault-injection seams: define BEFORE including this header to intercept")
    self.gen_add_code_line("// every allocation/copy the checked initializers make (tests); default = the bare call.")
    self.gen_add_code_line("#ifndef GRID_CUDA_CALL")
    self.gen_add_code_line("#define GRID_CUDA_CALL(expr) (expr)")
    self.gen_add_code_line("#endif")
    self.gen_add_code_line("#ifndef GRID_HOST_ALLOC")
    self.gen_add_code_line("#define GRID_HOST_ALLOC(expr) (expr)")
    self.gen_add_code_line("#endif")
    # Record the FIRST failed operation (primary error wins; cleanup never overwrites it).
    self.gen_add_code_line("__host__ inline cudaError_t grid_fail(const char **failed_op, const char *op, cudaError_t code){")
    self.gen_add_code_line("    if (failed_op != nullptr && *failed_op == nullptr) { *failed_op = op; }")
    self.gen_add_code_line("    return code;")
    self.gen_add_code_line("}")
    # Best-effort free of an owned device pointer during rollback/destruction: a
    # null pointer is a no-op; the first cleanup error is recorded separately so
    # the caller's primary error is never hidden.
    self.gen_add_code_line("__host__ inline void grid_cleanup_free(void *p, const char *op, cudaError_t *first_cleanup_code, const char **first_cleanup_op){")
    self.gen_add_code_line("    if (p == nullptr) { return; }")
    self.gen_add_code_line("    cudaError_t e = GRID_CUDA_CALL(cudaFree(p));")
    self.gen_add_code_line("    if (e != cudaSuccess && first_cleanup_code != nullptr && *first_cleanup_code == cudaSuccess) {")
    self.gen_add_code_line("        *first_cleanup_code = e; if (first_cleanup_op != nullptr) { *first_cleanup_op = op; }")
    self.gen_add_code_line("    }")
    self.gen_add_code_line("}")
    # Legacy policy: report the op, then the historical gpuAssert behaviour
    # (exit / sticky first error). Used ONLY by the un-suffixed wrappers.
    self.gen_add_code_line("__host__ inline void grid_legacy_check(cudaError_t e, const char *op, const char *file, const int line){")
    self.gen_add_code_line("    if (e != cudaSuccess) { fprintf(stderr, \"GRiD: %s failed: \", op ? op : \"initialization\"); gpuAssert(e, file, line); }")
    self.gen_add_code_line("}")
    self.gen_add_code_line("")

    # also add printMat for debug if requested
    if self.gen_print_mat:
        self.gen_add_code_line("template <typename T, int M, int N>")
        self.gen_add_code_line("__host__ __device__")
        self.gen_add_code_line("void printMat(T *A, int lda){", True)
        self.gen_add_code_line("for(int i=0; i<M; i++){", True)
        self.gen_add_code_line("for(int j=0; j<N; j++){printf(\"%.4f \",A[i + lda*j]);}")
        self.gen_add_code_line("printf(\"\\n\");")
        self.gen_add_end_control_flow()
        self.gen_add_end_function()
        self.gen_add_code_line("template <typename T, int M, int N>")
        self.gen_add_code_line("__host__ __device__")
        self.gen_add_code_line("void printMat(const T *A, int lda){", True)
        self.gen_add_code_line("for(int i=0; i<M; i++){", True)
        self.gen_add_code_line("for(int j=0; j<N; j++){printf(\"%.4f \",A[i + lda*j]);}")
        self.gen_add_code_line("printf(\"\\n\");")
        self.gen_add_end_control_flow()
        self.gen_add_end_function()
