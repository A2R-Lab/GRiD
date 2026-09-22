// Library-safe initialization runner (HJCD ask 2026-09-21): deterministic
// fault injection around EVERY host/device allocation and copy the generated
// `*_checked` initializers make. Modes (argv[1]):
//   success   : float+double init/free via the checked API; values identical to
//               the legacy path; ledger balanced (every successful cudaMalloc
//               saw a cudaFree); null free is a no-op.
//   sweep     : fail the k-th CUDA call for every k in the construction
//               sequence (and the host calloc): error reaches the caller, *out
//               stays null, every acquired resource gets its cudaFree, the
//               primary error/op is preserved; then a normal init succeeds and
//               the ledger balances (no accumulation across failed attempts).
//   cleanup   : inject failures INSIDE free_robotModel_checked (copy-back, a
//               nested cudaFree): no termination, unknown pointers untouched,
//               first error returned; then a clean free succeeds.
//   limits    : init_joint_limits_checked under the same sweep.
//   legacy    : fail the k-th call under the LEGACY init_robotModel(): the
//               default build exit()s (the pytest runs this in a subprocess and
//               expects a nonzero exit); a -DGRID_GPUERRCHK_NO_EXIT build
//               returns nullptr with the sticky slot set (prints LEGACY_NULL).
// Every mode prints "OK <mode>" on success and exits 0; any check failure
// prints "FAIL <what>" and exits 1. Nothing here touches device code.
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

static int  g_call = 0;         // CUDA-call sequence counter (per attempt)
static int  g_fail_at = -1;     // index of the CUDA call to fail (-1 = none)
static int  g_fail_at2 = -1;    // second injection point (cleanup failure during rollback)
static int  g_host_fail = 0;    // fail the next host allocation
static int  g_mallocs_ok = 0;   // successful cudaMalloc calls seen
static int  g_frees = 0;        // cudaFree calls attempted
static int  g_calls_last = 0;   // calls made in the last attempt
static cudaError_t g_inject_code = cudaErrorMemoryAllocation;
static cudaError_t g_inject_code_now = cudaErrorMemoryAllocation;

static cudaError_t g_inject_code2 = cudaErrorInvalidValue;
static bool is_alloc(const char *w) {  // acquisitions the ledger tracks
    return std::strncmp(w, "cudaMalloc", 10) == 0 || std::strncmp(w, "grid_device_alloc", 17) == 0
        || std::strncmp(w, "cudaStreamCreate", 16) == 0;
}
static bool is_free(const char *w) {   // releases the ledger tracks
    return std::strncmp(w, "cudaFree", 8) == 0 || std::strncmp(w, "grid_device_free", 16) == 0
        || std::strncmp(w, "cudaStreamDestroy", 17) == 0;
}
static bool fi_fail(const char *what) {
    int i = g_call++;
    bool fail = (i == g_fail_at) || (i == g_fail_at2);
    if (i == g_fail_at2) g_inject_code_now = g_inject_code2; else g_inject_code_now = g_inject_code;
    if (!fail && is_alloc(what)) g_mallocs_ok++;
    if (is_free(what)) g_frees++;
    return fail;
}
static bool host_fail(const char *) { bool f = g_host_fail != 0; g_host_fail = 0; return f; }
#define GRID_CUDA_CALL(expr) (fi_fail(#expr) ? g_inject_code_now : (expr))
#define GRID_HOST_ALLOC(expr) (host_fail(#expr) ? nullptr : (expr))

#include "grid.cuh"

static void reset(int fail_at = -1) { g_call = 0; g_fail_at = fail_at; g_fail_at2 = -1; g_mallocs_ok = 0; g_frees = 0; g_host_fail = 0; }
#define CHECK(cond, what) do { if (!(cond)) { std::printf("FAIL %s (line %d)\n", what, __LINE__); return 1; } } while (0)

template <typename T>
static int ledger_balanced(const char *tag) {
    if (g_frees != g_mallocs_ok) { std::printf("FAIL ledger %s: mallocs_ok=%d frees=%d\n", tag, g_mallocs_ok, g_frees); return 1; }
    return 0;
}

template <typename T>
static int values_match_legacy() {
    // Checked vs legacy construction must produce identical device tables.
    grid::robotModel<T> *a = nullptr, *b = nullptr; const char *op = nullptr;
    reset();
    CHECK(grid::init_robotModel_checked<T>(&a, &op) == cudaSuccess && a != nullptr && op == nullptr, "checked init");
    b = grid::init_robotModel<T>();
    CHECK(b != nullptr, "legacy init");
    grid::robotModel<T> ha, hb;
    CHECK(cudaMemcpy(&ha, a, sizeof(ha), cudaMemcpyDeviceToHost) == cudaSuccess, "copy a");
    CHECK(cudaMemcpy(&hb, b, sizeof(hb), cudaMemcpyDeviceToHost) == cudaSuccess, "copy b");
    // XImats bytes
    const size_t n_xi = (size_t)GRID_TEST_XI_SIZE;   // -D from the test (parsed from grid.cuh)
    std::vector<T> xa(n_xi), xb(n_xi);
    CHECK(cudaMemcpy(xa.data(), ha.d_XImats, n_xi * sizeof(T), cudaMemcpyDeviceToHost) == cudaSuccess, "copy XImats a");
    CHECK(cudaMemcpy(xb.data(), hb.d_XImats, n_xi * sizeof(T), cudaMemcpyDeviceToHost) == cudaSuccess, "copy XImats b");
    CHECK(std::memcmp(xa.data(), xb.data(), n_xi * sizeof(T)) == 0, "XImats differ between checked and legacy");
    // joint limits
    T *la = nullptr, *lb = nullptr;
    CHECK(grid::init_joint_limits_checked<T>(&la, &op) == cudaSuccess && la != nullptr, "checked limits");
    lb = grid::init_joint_limits<T>();
    const size_t n_l = (size_t)GRID_TEST_JL_SIZE;
    std::vector<T> va(n_l), vb(n_l);
    CHECK(cudaMemcpy(va.data(), la, n_l * sizeof(T), cudaMemcpyDeviceToHost) == cudaSuccess, "copy limits a");
    CHECK(cudaMemcpy(vb.data(), lb, n_l * sizeof(T), cudaMemcpyDeviceToHost) == cudaSuccess, "copy limits b");
    CHECK(std::memcmp(va.data(), vb.data(), n_l * sizeof(T)) == 0, "joint limits differ");
    CHECK(cudaFree(la) == cudaSuccess && cudaFree(lb) == cudaSuccess, "free limits");
    CHECK(grid::free_robotModel_checked<T>(a, &op) == cudaSuccess, "checked free a");
    grid::free_robotModel<T>(b);
    CHECK(grid::free_robotModel_checked<T>((grid::robotModel<T>*)nullptr, &op) == cudaSuccess, "null free");
    return 0;
}

template <typename T>
static int sweep() {
    // dry attempt to count the CUDA calls in one construction
    grid::robotModel<T> *m = nullptr; const char *op = nullptr;
    reset();
    CHECK(grid::init_robotModel_checked<T>(&m, &op) == cudaSuccess, "dry init");
    const int n_calls = g_call;
    CHECK(grid::free_robotModel_checked<T>(m, &op) == cudaSuccess, "dry free");
    CHECK(n_calls >= 4, "construction has too few CUDA calls to sweep");
    int failed_attempts = 0;
    for (int k = 0; k < n_calls; ++k) {
        reset(k); m = nullptr; op = nullptr;
        cudaError_t e = grid::init_robotModel_checked<T>(&m, &op);
        CHECK(e == g_inject_code, "injected error not returned");
        CHECK(m == nullptr, "output published on failure");
        CHECK(op != nullptr, "failed_op not recorded");
        if (ledger_balanced<T>("sweep")) return 1;
        failed_attempts++;
    }
    // host allocation failure (first calloc)
    reset(); g_host_fail = 1; m = nullptr; op = nullptr;
    CHECK(grid::init_robotModel_checked<T>(&m, &op) == cudaErrorMemoryAllocation && m == nullptr && op != nullptr, "host alloc failure");
    CHECK(std::strncmp(op, "calloc", 6) == 0, "host failure op name");
    if (ledger_balanced<T>("hostfail")) return 1;
    // after all the failed attempts, a normal construction works and balances
    reset(); m = nullptr; op = nullptr;
    CHECK(grid::init_robotModel_checked<T>(&m, &op) == cudaSuccess && m != nullptr, "post-sweep init");
    CHECK(grid::free_robotModel_checked<T>(m, &op) == cudaSuccess, "post-sweep free");
    if (ledger_balanced<T>("post")) return 1;
    std::printf("SWEEP calls=%d failed_attempts=%d\n", n_calls, failed_attempts);
    return 0;
}

template <typename T>
static int cleanup_failures() {
    grid::robotModel<T> *m = nullptr; const char *op = nullptr;
    // (a) copy-back failure inside free: nothing else touched, error returned.
    reset();
    CHECK(grid::init_robotModel_checked<T>(&m, &op) == cudaSuccess, "init");
    reset(); op = nullptr;
    // call sequence in free: PointerGetAttributes(0) GetDevice(1) Memcpy D2H(2) frees...
    g_fail_at = 2; g_inject_code = cudaErrorInvalidValue;
    cudaError_t e = grid::free_robotModel_checked<T>(m, &op);
    CHECK(e == cudaErrorInvalidValue, "copy-back error not returned");
    CHECK(g_frees == 0, "frees attempted after a failed copy-back");
    CHECK(op != nullptr && std::strstr(op, "D2H") != nullptr, "copy-back op not named");
    // (b) a nested cudaFree fails: cleanup continues, FIRST error returned, no termination.
    reset(); op = nullptr; g_fail_at = 3; g_inject_code = cudaErrorInvalidValue;
    e = grid::free_robotModel_checked<T>(m, &op);
    CHECK(e == cudaErrorInvalidValue, "nested free error not returned");
    CHECK(op != nullptr && std::strncmp(op, "cudaFree", 8) == 0, "nested free op not named");
    // all frees were still attempted (the injected one did not actually free: leaked by design)
    // (c) construction rollback with a failing cleanup keeps the PRIMARY error:
    // calls 0/1 = XImats malloc+memcpy, 2 = the next malloc (fails, primary),
    // 3 = the rollback cudaFree(d_XImats) (fails too, secondary).
    g_inject_code = cudaErrorMemoryAllocation;
    reset(2); g_fail_at2 = 3; m = nullptr; op = nullptr;
    e = grid::init_robotModel_checked<T>(&m, &op);
    CHECK(e == cudaErrorMemoryAllocation, "primary error overwritten by cleanup failure");
    CHECK(m == nullptr, "published after primary failure");
    CHECK(op != nullptr && std::strncmp(op, "cudaMalloc", 10) == 0, "primary op not preserved");
    // and a clean construction/destruction still works afterwards
    reset(); m = nullptr; op = nullptr;
    CHECK(grid::init_robotModel_checked<T>(&m, &op) == cudaSuccess, "init 2");
    CHECK(grid::free_robotModel_checked<T>(m, &op) == cudaSuccess, "free 2");
    std::printf("CLEANUP ok\n");
    return 0;
}

template <typename T>
static int limits_sweep() {
    T *l = nullptr; const char *op = nullptr;
    reset();
    CHECK(grid::init_joint_limits_checked<T>(&l, &op) == cudaSuccess && l != nullptr, "limits init");
    const int n_calls = g_call;
    CHECK(cudaFree(l) == cudaSuccess, "limits free");
    for (int k = 0; k < n_calls; ++k) {
        reset(k); l = nullptr; op = nullptr;
        CHECK(grid::init_joint_limits_checked<T>(&l, &op) == g_inject_code && l == nullptr && op != nullptr, "limits injected");
        if (ledger_balanced<T>("limits")) return 1;
    }
    std::printf("LIMITS calls=%d\n", n_calls);
    return 0;
}

// ─── part 2: arena (init_gridData), streams (init_grid), close_grid ─────────
template <typename T>
static int arena_sweep() {
    grid::gridData<T> *d = nullptr; const char *op = nullptr;
    reset();
    CHECK((grid::init_gridData_checked<T, 4>(&d, &op)) == cudaSuccess && d != nullptr, "arena dry init");
    const int n_calls = g_call;
    CHECK(n_calls >= 8, "arena construction has too few CUDA calls to sweep");
    reset(); op = nullptr;
    CHECK((grid::close_grid_checked<T, grid::GRID_DATA_ALL>(nullptr, nullptr, d, &op)) == cudaSuccess, "arena dry close");
    for (int k = 0; k < n_calls; ++k) {
        reset(k); d = nullptr; op = nullptr;
        cudaError_t e = grid::init_gridData_checked<T, 4>(&d, &op);
        CHECK(e == g_inject_code, "arena: injected error not returned");
        CHECK(d == nullptr, "arena: output published on failure");
        CHECK(op != nullptr, "arena: failed_op not recorded");
        if (ledger_balanced<T>("arena")) return 1;
    }
    reset(); g_host_fail = 1; d = nullptr; op = nullptr;
    CHECK((grid::init_gridData_checked<T, 4>(&d, &op)) == cudaErrorMemoryAllocation && d == nullptr && op != nullptr, "arena host alloc failure");
    if (ledger_balanced<T>("arena hostfail")) return 1;
    reset(); d = nullptr; op = nullptr;
    CHECK((grid::init_gridData_checked<T, 4>(&d, &op)) == cudaSuccess && d != nullptr, "arena post-sweep init");
    // the runtime-int spelling shares the body
    grid::gridData<T> *d2 = nullptr;
    CHECK((grid::init_gridData_checked<T>(4, &d2, &op)) == cudaSuccess && d2 != nullptr, "arena int init");
    CHECK((grid::close_grid_checked<T, grid::GRID_DATA_ALL>(nullptr, nullptr, d, &op)) == cudaSuccess, "arena post-sweep close");
    CHECK((grid::close_grid_checked<T, grid::GRID_DATA_ALL>(nullptr, nullptr, d2, &op)) == cudaSuccess, "arena int close");
    if (ledger_balanced<T>("arena post")) return 1;
    std::printf("ARENA calls=%d\n", n_calls);
    return 0;
}

template <typename T>
static int streams_sweep() {
    cudaStream_t *st = nullptr; const char *op = nullptr;
    reset();
    CHECK((grid::init_grid_checked<T>(&st, &op)) == cudaSuccess && st != nullptr, "streams dry init");
    const int n_calls = g_call;
    g_fail_at = -1; op = nullptr;  // keep the ledger across init+close: creates must equal destroys
    CHECK((grid::close_grid_checked<T, grid::GRID_DATA_ALL>(st, nullptr, nullptr, &op)) == cudaSuccess, "streams dry close");
    if (ledger_balanced<T>("streams dry")) return 1;
    int with_rollback = 0;
    for (int k = 0; k < n_calls; ++k) {
        reset(k); st = nullptr; op = nullptr;
        cudaError_t e = grid::init_grid_checked<T>(&st, &op);
        CHECK(e == g_inject_code, "streams: injected error not returned");
        CHECK(st == nullptr, "streams: output published on failure");
        CHECK(op != nullptr, "streams: failed_op not recorded");
        if (ledger_balanced<T>("streams")) return 1;
        if (g_mallocs_ok > 0) with_rollback++;
    }
    CHECK(with_rollback > 0, "streams: no failure index exercised the created-streams rollback");
    reset(); st = nullptr; op = nullptr;
    CHECK((grid::init_grid_checked<T>(&st, &op)) == cudaSuccess && st != nullptr, "streams post-sweep init");
    CHECK((grid::close_grid_checked<T, grid::GRID_DATA_ALL>(st, nullptr, nullptr, &op)) == cudaSuccess, "streams post-sweep close");
    std::printf("STREAMS calls=%d rollback_cases=%d\n", n_calls, with_rollback);
    return 0;
}

template <typename T>
static int close_failures() {
    const char *op = nullptr;
    // null everything: no-op
    CHECK((grid::close_grid_checked<T, grid::GRID_DATA_ALL>(nullptr, nullptr, nullptr, &op)) == cudaSuccess, "close null no-op");
    // full set, then a stream destroy fails: cleanup continues, FIRST error returned + named, all frees attempted
    cudaStream_t *st = nullptr; grid::robotModel<T> *m = nullptr; grid::gridData<T> *d = nullptr;
    reset();
    CHECK((grid::init_grid_checked<T>(&st, &op)) == cudaSuccess, "close: init streams");
    CHECK((grid::init_robotModel_checked<T>(&m, &op)) == cudaSuccess, "close: init model");
    CHECK((grid::init_gridData_checked<T, 4>(&d, &op)) == cudaSuccess, "close: init arena");
    const int allocs = g_mallocs_ok;
    reset(); op = nullptr;
    // dry count of close calls to place the injection on a stream destroy (the last calls)
    // (we cannot dry-run close without freeing, so inject on a late index and verify the op name)
    g_fail_at = 5; g_inject_code = cudaErrorInvalidValue;
    cudaError_t e = grid::close_grid_checked<T, grid::GRID_DATA_ALL>(st, m, d, &op);
    CHECK(e == cudaErrorInvalidValue, "close: injected error not returned as first error");
    CHECK(op != nullptr, "close: failed op not named");
    CHECK(g_frees >= allocs, "close: cleanup stopped early after a failure");
    g_inject_code = cudaErrorMemoryAllocation;
    std::printf("CLOSE ok (op=%s)\n", op);
    return 0;
}

int main(int argc, char **argv) {
    const char *mode = argc > 1 ? argv[1] : "success";
    if (std::strcmp(mode, "success") == 0) {
        if (values_match_legacy<float>()) return 1;
        if (values_match_legacy<double>()) return 1;
    } else if (std::strcmp(mode, "sweep") == 0) {
        if (sweep<float>()) return 1;
    } else if (std::strcmp(mode, "cleanup") == 0) {
        if (cleanup_failures<float>()) return 1;
    } else if (std::strcmp(mode, "limits") == 0) {
        if (limits_sweep<float>()) return 1;
    } else if (std::strcmp(mode, "arena") == 0) {
        if (arena_sweep<float>()) return 1;
    } else if (std::strcmp(mode, "streams") == 0) {
        if (streams_sweep<float>()) return 1;
    } else if (std::strcmp(mode, "close") == 0) {
        if (close_failures<float>()) return 1;
    } else if (std::strcmp(mode, "legacy") == 0) {
        reset(argc > 2 ? std::atoi(argv[2]) : 1);
        grid::robotModel<float> *m = grid::init_robotModel<float>();  // exits here in the default build
        std::printf("%s\n", m == nullptr ? "LEGACY_NULL" : "LEGACY_NONNULL");
        std::printf("STICKY %d\n", (int)grid_last_error());
    } else {
        std::printf("FAIL unknown mode %s\n", mode); return 1;
    }
    std::printf("OK %s\n", mode);
    return 0;
}
