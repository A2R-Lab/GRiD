"""Trace the production completion rule with a CPU CUDA double.

A mirrored launch is fenced by the bridge. The generated host function fences itself, so
the native layout adds nothing after it. The dense contract fences once after its copy.
"""
from pathlib import Path
import subprocess

import pytest

PROGRAM = r'''
#include "consumer_completion.h"
#include <cstdio>
#include <cstdlib>
int main(int argc, char **argv) {
    fail_at = std::atoi(argv[2]);
    auto mirrored = [] { return step('L'); };                       // enqueue only
    auto generated = [] { if(int rc=step('G')) return rc; return step('s'); };   // launch and its own fence
    auto check = [] { return step('K'); };
    auto copy = [] { return cudaError_t(step('C') ? 7 : 0); };
    int rc = -1;
    switch(argv[1][0]) {
        case 'a': rc = consumer_native_call(false, mirrored, check); break;
        case 'b': rc = consumer_native_call(true, generated, check); break;
        case 'c': rc = consumer_dense_call(mirrored, check, copy); break;
        case 'd': rc = consumer_dense_call(generated, check, copy); break;
    }
    std::printf("%s %d\n", events.c_str(), rc);
}
'''
DOUBLE = r'''
#include <string>
using cudaError_t=int;
const int cudaSuccess=0;
static std::string events; static int fail_at=0, calls=0;
static int step(char c) { events+=c; return ++calls==fail_at ? 7 : 0; }
static int cudaDeviceSynchronize() { return step('S'); }
'''
# case: (trace, bridge fences). 's' is the generated function's fence, 'S' the bridge's.
EXPECTED = {'a': 'LKS', 'b': 'GsK', 'c': 'LKCS', 'd': 'GsKCS'}


@pytest.fixture(scope='module')
def program(tmp_path_factory):
    root = tmp_path_factory.mktemp('consumer-completion')
    (root/'cuda_runtime_api.h').write_text(DOUBLE)
    (root/'test.cpp').write_text(PROGRAM)
    subprocess.run(['g++', '-std=c++17', '-I'+str(root), '-I'+str(Path(__file__).parent/'release'),
                    str(root/'test.cpp'), '-o', str(root/'test')], check=True)
    return root/'test'


def run(program, case, fail=0):
    trace, rc = subprocess.run([str(program), case, str(fail)], check=True, capture_output=True, text=True).stdout.split()
    return trace, int(rc)


@pytest.mark.parametrize('case', sorted(EXPECTED))
def test_each_path_has_one_fence_after_its_last_device_work(program, case):
    trace, rc = run(program, case)
    assert (trace, rc) == (EXPECTED[case], 0)
    assert trace.count('S') + trace.count('s') == (2 if case == 'd' else 1)
    assert trace[-1] in 'SK' and (trace[-1] == 'K') == (case == 'b')


@pytest.mark.parametrize('case', sorted(EXPECTED))
def test_every_step_error_stops_the_call_and_is_returned(program, case):
    for fail in range(1, len(EXPECTED[case]) + 1):
        trace, rc = run(program, case, fail)
        assert trace == EXPECTED[case][:fail] and rc != 0
