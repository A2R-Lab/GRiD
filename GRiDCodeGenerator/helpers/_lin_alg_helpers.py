from pathlib import Path
import re
import subprocess


# SIMT-only GLASS sources vendored into every generated grid.cuh. The
# cuBLASDx-backed `glass::nvidia` namespace was removed in v2.0; see
# docs/source/user_guide/concepts/cublasdx_removal_design.rst for the
# rationale and the `archive/last-cublasdx` git tag for the historical
# vendoring list.
_GLASS_BASE_FILES = [
    # Barrier policy for the shared *_impl bodies (GLASS v2 cgrps dedup + TRAILING_SYNC);
    # defines BlockBarrier, which every L1/L2/L3 op body now references. MUST be first.
    "src/base/barrier.cuh",
    "src/base/L1/reduce.cuh",
    "src/base/L1/dot.cuh",
    "src/base/L1/dot_strided.cuh",
    "src/base/L1/dot_strided_coalesced.cuh",
    "src/base/L2/gemv.cuh",
    "src/base/L2/gemv_strided.cuh",
    "src/base/L2/gemv_segmented.cuh",
    "src/base/L3/gemm.cuh",
    "src/base/L3/gemm_strided.cuh",
    "src/base/L3/gemm_batched_indexed.cuh",
    "src/base/L3/inv.cuh",            # used by invert_matrix (floating-base 6x6 root invert)
    "src/base/L3/syev.cuh",           # symmetric eigensolve + eig_clamp (PSD projection of the Newton ee cost hessian)
]


def _grid_repo_root():
    return Path(__file__).resolve().parents[2]


def _glass_root():
    root = _grid_repo_root() / "GLASS"
    if not root.exists():
        raise FileNotFoundError(
            "GLASS submodule is missing. Run `git submodule update --init GLASS` "
            "from the GRiD-A2R repository root."
        )
    return root


def _glass_commit():
    try:
        return subprocess.check_output(
            ["git", "-C", str(_glass_root()), "rev-parse", "HEAD"],
            text=True,
        ).strip()
    except Exception:
        return "unknown"


# File-level `GLASS_*` preprocessor guards (e.g. gemm.cuh's
# `GLASS_TILE4_HELPERS_DEFINED`, which dedups the tile4 helper block across the
# gemm/gemm_strided/gemm_batched includes) are GLOBAL macros — the preprocessor
# does not respect the `grid::glass` namespace we vendor into. If a consumer
# includes BOTH its own global GLASS and this generated header, whichever defines
# the guard first suppresses the OTHER copy's helper definitions, leaving e.g.
# `tile4_has_vec`/`tile4_profitable` undefined in that namespace. Rename every
# vendored `GLASS_*` macro token to a `GRID_VENDORED_`-prefixed name so the
# vendored snapshot's guards can never collide with the consumer's GLASS — which
# is the whole point of nesting the vendored copy (see gen_grid_linalg_backend_helpers).
_GLASS_MACRO_TOKEN = re.compile(r"\bGLASS_[A-Z0-9_]+\b")


def _emit_glass_source_file(self, relative_path):
    source_path = _glass_root() / relative_path
    if not source_path.exists():
        raise FileNotFoundError("Required GLASS source file is missing: " + str(source_path))
    self.gen_add_code_line("// BEGIN GLASS " + relative_path)
    for line in source_path.read_text().splitlines():
        stripped = line.strip()
        if stripped.startswith("#pragma once"):
            continue
        if stripped.startswith("#include"):
            continue
        line = _GLASS_MACRO_TOKEN.sub(lambda m: "GRID_VENDORED_" + m.group(0), line)
        self.gen_add_code_line(line)
    self.gen_add_code_line("// END GLASS " + relative_path)
    self.gen_add_code_line("")


def gen_linalg_smem_setup(self, temp_size):
    """Emit ``unsigned char *s_linalg_smem = nullptr;``.

    Historically set up an aligned smem pointer for cuBLASDx scratch.
    cuBLASDx was removed in v2.0 (see
    ``docs/source/user_guide/concepts/cublasdx_removal_design.rst``);
    the SIMT linalg path needs no scratch. The function (and the
    ``s_linalg_smem`` local) are kept so the ~100 callsites that pass
    it as the last ``grid_linalg_*`` argument continue to compile
    without per-callsite edits.
    """
    del temp_size  # unused; kept in signature for caller compatibility
    self.gen_add_code_line("unsigned char *s_linalg_smem = nullptr;")


def gen_grid_linalg_backend_helpers(self):
    """Emit the GRiD linear-algebra adapter.

    SIMT GLASS is vendored at codegen time, plus thin
    ``grid_linalg_*`` wrappers that delegate to it. The wrappers
    accept (and ignore) a trailing ``glass_nvidia_smem`` argument so
    pre-v2.0 callsites continue to compile without edits.
    """
    glass_commit = _glass_commit()
    self.gen_add_func_doc("Vendored GLASS linear algebra helpers (SIMT only)")
    # Vendor GLASS NESTED inside the generated namespace (e.g. `grid::glass`) rather
    # than at global `glass::`. This keeps GRiD hermetic: a consumer can include its
    # own (possibly newer) GLASS at the global `glass::` without an ODR clash against
    # this pinned snapshot. Generated code refers to it as bare `glass::...`, which
    # resolves to the nested namespace by ordinary lookup, so call sites need no
    # qualification (and must NOT be globally qualified, which would escape to the
    # consumer's global GLASS).
    self.gen_add_code_lines([
        "",
        "// Vendored from GLASS at codegen time (nested in this namespace).",
        "// Source repository: git@github.com:A2R-Lab/GLASS.git",
        "// Pinned commit: " + glass_commit,
        "namespace glass {",
        "",
    ])
    for relative_path in _GLASS_BASE_FILES:
        _emit_glass_source_file(self, relative_path)
    self.gen_add_code_line("} // namespace glass")
    self.gen_add_code_line("")

    self.gen_add_func_doc("Linear algebra wrappers (SIMT GLASS)")
    self.gen_add_code_lines([
        "// SIMT-only linalg. cuBLASDx was removed in v2.0; the",
        "// `glass_nvidia_smem` parameter on each wrapper is retained for",
        "// caller compatibility and is always ignored. The stub",
        "// `GRID_LINALG_NVIDIA_MAX_HELPER_BYTES<T>()` below returns 0 so",
        "// shared-memory arena calculations continue to compile unchanged.",
        "",
        "template <typename T>",
        "__host__ __device__ constexpr size_t GRID_LINALG_NVIDIA_MAX_HELPER_BYTES() {",
        "    return static_cast<size_t>(0);",
        "}",
        "",
        "template <typename T, int M, int N, int K, bool TRANSPOSE_B = false, bool ROW_MAJOR_A = false, bool ROW_MAJOR_B = false, bool ROW_MAJOR_C = false>",
        "__device__ void grid_linalg_gemm(const T *A, const T *B, T *C, T alpha, T beta, unsigned char *glass_nvidia_smem = nullptr) {",
        "    (void)glass_nvidia_smem;",
        "    T *A_mut = const_cast<T *>(A);",
        "    T *B_mut = const_cast<T *>(B);",
        "    // GLASS v2 gemm: contraction is the LAST template dim, so the old (M,N,K)",
        "    // contraction-in-the-middle maps to <M,K,N>. A row-major operand equals its",
        "    // col-major transpose (ROW_MAJOR_A -> TRANSPOSE_A; ROW_MAJOR_B XORs into",
        "    // TRANSPOSE_B); ROW_MAJOR_C is the only surviving per-operand layout flag.",
        "    glass::gemm<T, M, K, N, ROW_MAJOR_A, (TRANSPOSE_B != ROW_MAJOR_B), ROW_MAJOR_C>(alpha, A_mut, B_mut, beta, C);",
        "    __syncthreads();",
        "}",
        "",
        "template <typename T, int M, int N, bool TRANSPOSE = false, bool ROW_MAJOR_A = false>",
        "__device__ void grid_linalg_gemv(const T *A, const T *x, T *y, T alpha, T beta, unsigned char *glass_nvidia_smem = nullptr) {",
        "    (void)glass_nvidia_smem;",
        "    T *A_mut = const_cast<T *>(A);",
        "    T *x_mut = const_cast<T *>(x);",
        "    // GLASS v2: gemv keeps its per-operand ROW_MAJOR flag (TRANSPOSE selects the math",
        "    // op A*x vs A^T*x, independent of storage); the redundant gemv_ex was removed.",
        "    glass::gemv<T, M, N, TRANSPOSE, ROW_MAJOR_A>(alpha, A_mut, x_mut, beta, y);",
        "    __syncthreads();",
        "}",
        "",
        "template <typename T, int M, int N, int ROW_STRIDE>",
        "__device__ void grid_linalg_row_strided_gemv(const T *A, const T *x, T *y, T alpha, T beta, unsigned char *glass_nvidia_smem = nullptr) {",
        "    (void)glass_nvidia_smem;",
        "    glass::gemv_strided<T, M, N, ROW_STRIDE>(alpha, A, x, beta, y);",
        "    __syncthreads();",
        "}",
        "",
        "template <typename T, int M, int N, int K, int A_RS, int B_RS>",
        "__device__ void grid_linalg_row_strided_gemm(const T *A, const T *B, T *C, T alpha, T beta, unsigned char *glass_nvidia_smem = nullptr) {",
        "    (void)glass_nvidia_smem;",
        "    glass::gemm_strided<T, M, K, N, A_RS, B_RS>(alpha, A, B, beta, C);",
        "    __syncthreads();",
        "}",
        "",
        "template <typename T, int N, int S1, int S2>",
        "__device__ T grid_linalg_dot_strided(const T *vec1, const T *vec2) {",
        "    return glass::dot_strided<T, N, S1, S2>(vec1, vec2);",
        "}",
        "",
        "// Segmented (batched) row-strided GEMV: `segments` independent M x N GEMVs in one",
        "// block-cooperative pass, base offsets per segment via the descriptor arrays. With",
        "// FUSE_SCALED_ADD, folds a per-segment y += S*scalar add into the single y store.",
        "template <typename T, int M, int N, int ROW_STRIDE = M, bool FUSE_SCALED_ADD = false>",
        "__device__ void grid_linalg_segmented_row_strided_gemv(unsigned int segments, const int *seg_a_off, const int *seg_x_off, const int *seg_y_off, const T *A, const T *x, T *y, T alpha, T beta, const int *seg_s_off = nullptr, const T *S = nullptr, const T *scalar = nullptr, unsigned char *glass_nvidia_smem = nullptr) {",
        "    (void)glass_nvidia_smem;",
        "    glass::gemv_segmented<T, M, N, ROW_STRIDE, FUSE_SCALED_ADD>(segments, seg_a_off, seg_x_off, seg_y_off, A, x, y, alpha, beta, seg_s_off, S, scalar);",
        "    __syncthreads();",
        "}",
        "",
        "// Indexed batched DIMxDIM (col-major) GEMM: C[c_idx[p]] = A[a_idx[p]] * B[b_idx[p]]",
        "// over a flat in-chain index list (e.g. compacted (ancestor,ee) pairs).",
        "template <typename T, int DIM = 4>",
        "__device__ void grid_linalg_indexed_batched_gemm(unsigned int pairs, const int *a_idx, const int *b_idx, const int *c_idx, const T *A_base, const T *B_base, T *C_base, unsigned char *glass_nvidia_smem = nullptr) {",
        "    (void)glass_nvidia_smem;",
        "    glass::gemm_batched_indexed<T, DIM>(pairs, a_idx, b_idx, c_idx, A_base, B_base, C_base);",
        "    __syncthreads();",
        "}",
        "",
        "// Coalesced block-cooperative strided dot: same value as grid_linalg_dot_strided but",
        "// the whole block cooperates on ONE dot with transposed/tiled iteration so consecutive",
        "// threads hit consecutive global addresses (fast when the operand is L2-pinned global).",
        "// Writes the scalar to *out (valid after the trailing barrier); needs ceil(blockDim/32)",
        "// T of s_scratch. NOT a drop-in for grid_linalg_dot_strided (that one is per-thread).",
        "template <typename T, int N, int SX = 1, int SY = 1>",
        "__device__ void grid_linalg_dot_strided_coalesced(const T *x, const T *y, T *out, T *s_scratch) {",
        "    glass::dot_strided_coalesced<T, N, SX, SY>(x, y, out, s_scratch);",
        "    __syncthreads();",
        "}",
        ""
    ])


def gen_invert_matrix(self):
    """Emits a thin wrapper around `glass::inv_dense` (block-
    cooperative Gauss-Jordan; `GLASS/src/base/L3/inv.cuh`).

    Why a wrapper rather than re-implementing here: GLASS is the first-party
    linalg layer (memory `project_grid_glass_first_party.md`); pinning the
    primitive there means future GLASS improvements (e.g. swapping to
    Cholesky/LDLT for SPD inputs, vectorizing the save loop, lifting the
    pivot loop) auto-propagate on the next vendor without re-touching the
    emitter.

    The base `glass::inv` (also embedded) takes the classic augmented
    `[A | I]` n×(2n) layout; that doesn't fit the GRiD callers which
    pre-allocate separate A and Ainv buffers, so we use the dense
    in-place variant `glass::inv_dense(dimA, A, Ainv, s_temp)`
    (3*dimA scratch, A → A^-1 AND Ainv → A^-1; named
    invertMatrix_dense before the GLASS r2 rename).

    Signature preserved for caller compatibility:
        invert_matrix(dimA, A, Ainv, s_temp)
    On return: A := A^-1 (in-place — old code also overwrote A to identity,
    which no caller depended on); Ainv := A^-1 (alias of A's inverse).
    Callers that wrote a pre-init to Ainv = I before calling are now
    paying redundant work; clean those up in a follow-up commit.
    s_temp must hold at least (3*dimA) elements; the legacy callers reserve
    4*dimA so there is headroom.
    """
    self.gen_add_func_doc(
        "Compute the inverse of a matrix (wraps glass::inv_dense)",
        ["Block-cooperative Gauss-Jordan via GLASS.",
         "Both A and Ainv hold A^-1 on return (dual-output for caller compat).",
         "s_temp must hold at least 3*dimA elements."],
        ['dimA is the matrix dimension',
         'A is the original invertible matrix (overwritten with A^-1 on return)',
         'Ainv is workspace; on return it also holds A^-1',
         's_temp is shared scratch of size >= 3*dimA'])
    self.gen_add_code_line("template <typename T>")
    self.gen_add_code_line("__device__")
    self.gen_add_code_line("void invert_matrix(uint32_t dimA, T *A, T *Ainv, T *s_temp) {", True)
    self.gen_add_code_line("glass::inv_dense<T>(dimA, A, Ainv, s_temp);")
    self.gen_add_end_function()
    return


def gen_matmul(self):
    """
    Generates the matrix multiplication helper function.
    This function allows for a transpose of B
    """
    self.gen_add_func_doc("Matrix multiplication helper function of AB", [], \
                          ['index - the index of the result vector', \
                           'A - pointer to the first matrix', \
                           'B - pointer to the second matrix', \
                           'dest - pointer to the destination matrix', \
                           'num - 36 or 6 depending on the indexing scheme', \
                           't - true => multiply with the transpose of B'])
    self.gen_add_code_line("template <typename T>")
    self.gen_add_code_line("__device__")
    self.gen_add_code_line("void matmul(int index, T *A, T *B, T *dest, int num, bool t) {", True)
    # B2-SO FIX (FLAG for main reconcile): the per-block modulus must wrap by the
    # number of BODY blocks, not NUM_JOINTS. For mimic robots NUM_BODIES > NUM_JOINTS
    # (extra mimic-sibling bodies), and this helper is called over 36*NUM_BODIES
    # elements by the idsva_so IC build (I @ Xup). With the old NUM_JOINTS modulus the
    # last mimic body wrapped to block 0 and read body 0's inertia -> corrupted IC for
    # that body -> propagated up the whole composite-inertia chain -> wrong SO output.
    # NUM_BODIES == NUM_JOINTS for every non-mimic fixed robot, so this is byte-identical
    # there. matmul is used ONLY by _idsva_so.py (3 call sites; the two Xup sites pass
    # index/num == 0 so the modulus is a no-op for them), so this change is self-contained.
    self.gen_add_code_line("int cur = 36*((index/num)%NUM_BODIES);")
    self.gen_add_code_line("T *vec1 = &B[cur + (t*5+1)*(index%6)];")
    self.gen_add_code_line("T *vec2 = &A[6*(index/6)];")
    self.gen_add_code_line("dest[index] = dot_prod<T,6, 6, 1>(vec1, vec2);")
    self.gen_add_end_function()


def gen_matmul_trans(self):
    """
    Generates the matrix multiplication helper function where one of the
    matrices is transposed. Both A and B are 6x6 matrices.
    """
    self.gen_add_func_doc("Matrix multiplication helper function where one of the matrices is tranposed.", [], \
                          ['index - the index of the result vector', \
                           'A - pointer to the first 6x6 matrix', \
                           'B - pointer to the second 6x6 matrix', \
                           'dest - pointer to the destination matrix', \
                           'char trans_mat - a for A^TB, b for AB^T'])
    self.gen_add_code_line("template <typename T>")
    self.gen_add_code_line("__device__")
    self.gen_add_code_line("void matmul_trans(int index, T *A, T *B, T *dest, char trans_mat) {", True)
    self.gen_add_code_line("T *vec1;")
    self.gen_add_code_line("T *vec2;")
    self.gen_add_code_line("if (trans_mat == 'a'){", True)
    self.gen_add_code_line("vec1 = &A[6*(index%6)];")
    self.gen_add_code_line("vec2 = &B[6*(index/6)];")
    self.gen_add_code_line("dest[index] = dot_prod<T,6,1,1>(vec1, vec2);")
    self.gen_add_end_control_flow()
    self.gen_add_code_line("if (trans_mat == 'b'){", True)
    self.gen_add_code_line("vec1 = &A[index%6];")
    self.gen_add_code_line("vec2 = &B[index/6];")
    self.gen_add_code_line("dest[index] = dot_prod<T,6,6,6>(vec1, vec2);")
    self.gen_add_end_control_flow()
    self.gen_add_end_function()


def gen_outer_product(self):
    """
    This function generates the cuda for the outerProduct
    function.
    """
    self.gen_add_func_doc("Compute the outer product between two vectors: dest = ab^T", \
                          ["Function assumes it is called by a single thread."], \
                          ['a - first vector', \
                           'b - second vector', \
                           'dest - destination matrix', \
                           'aLength - length of a', \
                           'bLength - length of b', \
                           'idx - index of resulting matrix to be computed by this thread'])
    self.gen_add_code_line("template <typename T>")
    self.gen_add_code_line("__device__")
    self.gen_add_code_line("void outerProduct(T *a, T *b, T *dest, int aLength, int bLength, int idx) {", True)
    self.gen_add_code_line("int row = idx / bLength;")
    self.gen_add_code_line("int col = idx % bLength;")
    self.gen_add_code_line("if (row < aLength && col < bLength) dest[col * aLength + row] = a[row] * b[col];")
    self.gen_add_end_function()
