"""Per-algorithm suggested thread counts (grid_codegen/suggested_threads.py).

The header ends with one ``<ALGO>_SUGGESTED_THREADS`` constant per launch-config
algorithm present in it (plus ``TRACKING_COST_SUGGESTED_THREADS`` when the plant
preset is emitted): the smallest multiple of 32 covering the widest block-parallel
compute loop reachable from the algorithm's kernel, read back from the emitted
text. These referees pin the derivation on iiwa14 (fixed) and go2 (floating): the
constants exist for every algorithm in the header, are warp multiples in
[32, 1024], agree with a fresh derivation on the written header, carry no
unreadable loop bound, and respect composition (a composed algorithm is at least
as wide as what it composes). The loop classifier is checked on small strings.
"""
from __future__ import annotations

import contextlib
import os
import re

import pytest

from grid_codegen import GRiDCodeGenerator
from grid_codegen import suggested_threads as st
from grid_codegen.algo_registry import launch_config_descriptors
from grid_assets import robot_urdf
from URDFParser import URDFParser

_CONST_RE = re.compile(r"^\s*const int (\w+)_SUGGESTED_THREADS = (\d+);\s*$", re.M)
_KEYS = [d.key for d in launch_config_descriptors()]


def _generate(tmp_path, name, floating):
    with open(os.devnull, "w") as devnull, contextlib.redirect_stdout(devnull), \
            contextlib.redirect_stderr(devnull):
        robot = URDFParser().parse(str(robot_urdf(name)), floating_base=floating)
        gen = GRiDCodeGenerator(robot, False, False, "grid")
        out = tmp_path / f"{name}.cuh"
        gen.gen_all_code(output_path=str(out), enable_mujoco_kernels=False)
    return gen, out.read_text()


@pytest.fixture(scope="module", params=[("iiwa14", False), ("go2", True)], ids=["iiwa14", "go2"])
def header(request, tmp_path_factory):
    name, floating = request.param
    return _generate(tmp_path_factory.mktemp(name), name, floating)


def test_constants_present_warp_sized_and_reproducible(header):
    gen, text = header
    emitted = {m.group(1).lower(): int(m.group(2)) for m in _CONST_RE.finditer(text)}
    assert emitted, "no suggested thread constants in the header"
    for key, val in emitted.items():
        assert val % 32 == 0 and st.MIN_THREADS <= val <= st.MAX_THREADS, (key, val)
    # one per launch-config algorithm whose kernel is in the header
    for key in _KEYS:
        roots = st._kernel_roots(key)
        in_header = any(re.search(r"\bvoid " + r + r"\(", text) for r in roots)
        assert (key in emitted) == in_header, (key, in_header)
    assert ("tracking_cost" in emitted) == ("void tracking_cost(" in text)
    # the generator's own record and a fresh derivation on the written text agree
    assert gen._suggested_threads == emitted
    fresh, unknown = st.derive(text, _KEYS)
    assert fresh == emitted
    assert unknown == {}, f"unreadable loop bounds: {unknown}"
    assert gen._suggested_threads_unknown == {}


def test_composition_order(header):
    _, text = header
    v = {m.group(1).lower(): int(m.group(2)) for m in _CONST_RE.finditer(text)}
    pairs = [("forward_dynamics", "inverse_dynamics"),
             ("forward_dynamics_gradient", "inverse_dynamics_gradient"),
             ("forward_dynamics_gradient", "forward_dynamics"),
             ("integrator_gradient", "integrator"),
             ("integrator_with_gradient", "integrator"),
             ("nonlinear_effects", "inverse_dynamics"),
             ("fdsva_so", "idsva_so_body_frame"), ("fdsva_so", "idsva_so_world_frame")]
    for wide, narrow in pairs:
        if wide in v and narrow in v:
            assert v[wide] >= v[narrow], (wide, v[wide], narrow, v[narrow])
    # the second-order kernels sweep NV^3 cells: with NV >= 7 that is above 32
    for key in ("idsva_so_body_frame", "idsva_so_world_frame", "fdsva_so"):
        if key in v:
            assert v[key] >= 352


def test_block_is_last_and_reopens_the_namespace(header):
    _, text = header
    i = text.index("// Suggested thread counts")
    tail = text[i:]
    assert "namespace grid {" in tail
    assert tail.rstrip().endswith("}")
    assert "__global__" not in tail and "__device__" not in tail


@pytest.mark.parametrize("body,expected", [
    ("a[ind] = b[ind];", True),
    ("s[ind] = static_cast<T>(0);", True),
    ("s[ind] = d_robotModel->d_XImats[ind + 616];", True),
    ("if constexpr (RUNTIME) { if (ind >= 4 && ind < 8) { continue; } } s[ind] = d[ind];", True),
    ("if (ind < 360) { s[ind] = x[360 + ind]; } else { m[ind - 360] = static_cast<T>(0); }", True),
    ("s[ind] += a[ind] * b[ind];", False),
    ("s[ind] = a[ind] + b[ind];", False),
    ("T th = tanh(s_qd[ind] / w); s[ind] = th;", False),
    ("int row = ind % 6; s[ind] = dot_prod<T,6,1,1>(&a[row], &b[0]);", False),
])
def test_copy_loop_classifier(body, expected):
    assert st.is_copy_loop(body) is expected


def test_bound_evaluation_and_rounding():
    consts = {"NUM_VEL": 7, "XIMAT_SIZE": 36}
    assert st._eval_bound("4*NUM_VEL*NUM_VEL", consts) == 196
    assert st._eval_bound("XIMAT_SIZE*2", consts) == 72
    assert st._eval_bound("vc_count", consts) is None
    assert st._round_up(0, 128) == 128
    assert st._round_up(1, 128) == 32
    assert st._round_up(33, 128) == 64
    assert st._round_up(5000, 128) == 1024


def test_call_tree_follows_calls_and_skips_copies():
    text = """
    const int NUM_VEL = 4;
    template <typename T>
    __device__
    void helper_inner(T *s) {
        for(int ind = threadIdx.x + threadIdx.y*blockDim.x; ind < 100; ind += blockDim.x*blockDim.y){
            s[ind] = static_cast<T>(0);
        }
        for(int ind = threadIdx.x + threadIdx.y*blockDim.x; ind < 40; ind += blockDim.x*blockDim.y){
            s[ind] += s[ind + 1] * static_cast<T>(2);
        }
    }
    template <typename T>
    __global__
    void thing_kernel(T *d) {
        for(int ind = threadIdx.x + threadIdx.y*blockDim.x; ind < NUM_VEL*NUM_VEL; ind += blockDim.x*blockDim.y){
            d[ind] = d[ind] + static_cast<T>(1);
        }
        helper_inner<T>(d);
    }
    """
    bodies = st.index_functions(text)
    assert set(bodies) == {"helper_inner", "thing_kernel"}
    width, unknown, found = st.call_tree_width(["thing_kernel"], bodies, {"NUM_VEL": 4})
    assert found and unknown == set() and width == 40
    width_all, _, _ = st.call_tree_width(["thing_kernel"], bodies, {"NUM_VEL": 4}, compute_only=False)
    assert width_all == 100
