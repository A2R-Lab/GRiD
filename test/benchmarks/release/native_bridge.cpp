// Native host-to-host loop for the SAME fp32 artifact used by Python wrappers.
// No CUDA headers required. The already-loaded Python handle owns the context.
#include <chrono>
#include <cstring>
#include <dlfcn.h>
#include <memory>

extern "C" int grid_release_time(const char *library, const char *symbol,
    long long context, const float *q, const float *v, const float *a,
    int batch, int output_count, int warmups, int iterations, double warm_seconds,
    double *times_us, float *last_output) {
  if (!library || !symbol || !q || !v || !a || !times_us || !last_output ||
      batch <= 0 || output_count <= 0 || warmups < 1 || iterations < 1 || warm_seconds < 0) return -1;
  if (std::strcmp(symbol, "grid_rbd_inverse_dynamics") &&
      std::strcmp(symbol, "grid_rbd_inverse_dynamics_gradient")) return -2;
  void *lib = dlopen(library, RTLD_NOW | RTLD_LOCAL);
  if (!lib) return -3;
  using Fn = int (*)(long long, const float*, const float*, const float*,
                     float*, int, float, const float*);
  auto fn = reinterpret_cast<Fn>(dlsym(lib, symbol));
  if (!fn) { dlclose(lib); return -4; }
  using clock = std::chrono::steady_clock;
  // Time-based warm-up so the device reaches its steady clock before the
  // samples (a handful of microsecond calls does not leave the idle clock).
  const auto warm_start = clock::now();
  int warmed = 0;
  do {
    std::unique_ptr<float[]> out(new float[output_count]);
    int rc = fn(context, q, v, a, out.get(), batch, -9.81f, nullptr);
    if (rc) { dlclose(lib); return rc; }
    ++warmed;
  } while (warmed < warmups ||
           std::chrono::duration<double>(clock::now() - warm_start).count() < warm_seconds);
  for (int i = 0; i < iterations; ++i) {
    auto start = clock::now();
    // Include host output allocation, like the public NumPy surface (np.empty:
    // an uninitialized buffer, so no zero fill is timed). The C ABI
    // synchronously returns host output; no Python call or timer is in this loop.
    std::unique_ptr<float[]> out(new float[output_count]);
    int rc = fn(context, q, v, a, out.get(), batch, -9.81f, nullptr);
    auto end = clock::now();
    if (rc) { dlclose(lib); return rc; }
    times_us[i] = std::chrono::duration<double, std::micro>(end-start).count();
    if (i == iterations-1) std::memcpy(last_output, out.get(), output_count*sizeof(float));
  }
  dlclose(lib);
  return 0;
}
