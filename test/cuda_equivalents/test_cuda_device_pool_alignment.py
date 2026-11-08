"""Caller-owned slabs must preserve cudaMalloc's allocation alignment.

This exercises the generated allocator, not a Python reimplementation. Invalid
requests must neither advance the cursor nor return a usable pointer.
"""
from pathlib import Path
import subprocess

import pytest
from grid_assets import robot_urdf
from grid_codegen import GRiDCodeGenerator
from URDFParser import URDFParser
from test.cuda_equivalents.cuda_harness import _detect_cuda_arch
from test.cuda_equivalents.executable_cache import cached_nvcc_executable

pytestmark = [pytest.mark.cuda_equivalence, pytest.mark.robot_smoke]


def test_device_pool_alignment_and_capacity(tmp_path):
    robot = URDFParser().parse(str(robot_urdf('iiwa14')))
    GRiDCodeGenerator(robot).gen_all_code(
        algorithm_list=['forward_dynamics'], enable_mujoco_kernels=False,
        output_path=str(tmp_path / 'grid.cuh'))
    source = tmp_path / 'pool.cu'
    source.write_text(r'''
#include "grid.cuh"
#include <cstdio>
#include <limits>
#define CHECK(x) do { if (!(x)) { fprintf(stderr, "line %d: %s\n", __LINE__, #x); return 1; } } while (0)
int main() {
    unsigned char *base = nullptr;
    CHECK(cudaMalloc(&base, 2048) == cudaSuccess);
    // Every nonzero residue violates the slab allocator's 256-byte contract.
    for (size_t residue = 1; residue < 256; ++residue) {
        grid::grid_device_pool_t p{base + residue, 1024, 0, 1};
        void *out = base;
        CHECK(grid::grid_device_alloc(&p, &out, 8) == cudaErrorInvalidValue);
        CHECK(out == nullptr && p.used == 0);
    }
    for (size_t residue = 1; residue < 256; ++residue) {
        grid::grid_device_pool_t p{base, 1024, residue, 1};
        void *out = base;
        CHECK(grid::grid_device_alloc(&p, &out, 8) == cudaErrorInvalidValue);
        CHECK(out == nullptr && p.used == residue);
    }
    grid::grid_device_pool_t p{base, 1024, 0, 1};
    void *a = nullptr, *b = nullptr;
    CHECK(grid::grid_device_alloc(&p, &a, 257) == cudaSuccess);
    CHECK(grid::grid_device_alloc(&p, &b, 512) == cudaSuccess);
    CHECK(a == base && b == base + 512 && p.used == 1024);
    CHECK(grid::grid_device_alloc(&p, &b, 1) == cudaErrorMemoryAllocation);
    CHECK(b == nullptr && p.used == 1024);
    // Overflow must not round an enormous request down to a zero-sized success.
    p.used = 0;
    CHECK(grid::grid_device_alloc(&p, &b, std::numeric_limits<size_t>::max()) == cudaErrorMemoryAllocation);
    CHECK(b == nullptr && p.used == 0);
    CHECK(cudaFree(base) == cudaSuccess);
    puts("pool alignment, exact capacity, atomic refusal and overflow passed");
}
''')
    exe, _ = cached_nvcc_executable(
        [source], ['-std=c++17', '-O2', '-arch=sm_' + _detect_cuda_arch()],
        exe_name='pool', fallback_dir=tmp_path, include_dirs=[tmp_path],
        what='device pool alignment contract')
    result = subprocess.run([str(exe)], capture_output=True, text=True)
    (tmp_path / 'run.log').write_text(result.stdout + result.stderr)
    assert result.returncode == 0, result.stdout + result.stderr
