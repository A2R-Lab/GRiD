#pragma once
#include <cuda_runtime_api.h>

// GRiD's floating ID arena uses nq-wide rows, while the common force output
// has nv entries. Copy rows explicitly; a flat batch copy leaks padding and
// can overrun the caller's dense output allocation.
inline cudaError_t consumer_copy_dense(float *out, int width, const float *in,
                                       int source_width, int batch, cudaMemcpyKind kind) {
    return cudaMemcpy2D(out, size_t(width)*sizeof(float), in,
                        size_t(source_width)*sizeof(float), size_t(width)*sizeof(float), batch, kind);
}
