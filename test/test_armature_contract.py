"""CPU armature boundary regressions; fake runners do not prove CUDA ownership."""
import contextlib
import io
from pathlib import Path
import subprocess

import numpy as np
import pytest

from grid_rbd._handle import RobotHandle
from grid_codegen import GRiDCodeGenerator
from grid_assets import robot_urdf
from URDFParser import URDFParser
from RBDReference import RBDReference


class Runner:
    num_vel = 2

    def __init__(self):
        self.values = np.zeros(2, dtype=np.float32)
        self.version = 0

    def set_armature_params(self, values):
        self.values = values.copy()
        self.version += 1

    def get_armature_params(self):
        return self.values.copy()


def handle(name="shared_name", runner=None, slots=(0, 1)):
    h = RobotHandle.__new__(RobotHandle)
    h._name = name
    h._dt = np.float32
    h._meta = {"runtime_armature": True, "armature_slots": list(slots)}
    h._runner = runner if runner is not None else Runner()
    h._owns_context = False
    return h


def test_readback_follows_owner_not_name():
    one, two = handle(), handle()
    alias = handle("different_name", one._runner)
    one.set_armature([.1, .2])
    np.testing.assert_array_equal(two.armature, [0, 0])
    np.testing.assert_array_equal(alias.armature, one.armature)
    # Direct native mutation, not a call through this Python handle.
    one._runner.set_armature_params(np.array([.3, .4], dtype=np.float32))
    np.testing.assert_array_equal(alias.armature, one._runner.values)
    copy = one.armature
    copy[:] = 0
    np.testing.assert_array_equal(one.armature, one._runner.values)


def test_closed_handle_cannot_read_or_mutate():
    h = handle()
    h.close()
    with pytest.raises(RuntimeError, match="closed"):
        _ = h.armature
    with pytest.raises(RuntimeError, match="closed"):
        h.set_armature([0, 0])


@pytest.mark.parametrize("values", [
    [-.1, 0], [-1e-100, 0], [np.nan, 0], [np.inf, 0], [1e100, 0],
    [[0, 0]], [0], [0j, 0],
])
def test_python_invalid_update_preserves_values_and_version(values):
    h = handle()
    h.set_armature([.2, .3])
    before, version = h.armature, h._runner.version
    with pytest.raises(ValueError):
        h.set_armature(values)
    np.testing.assert_array_equal(h.armature, before)
    assert h._runner.version == version


def test_unsupported_slot_checked_before_precision_conversion():
    h = handle(slots=(1,))
    with pytest.raises(ValueError):
        h.set_armature([1e-100, 0])
    assert h._runner.version == 0


def test_missing_slot_metadata_fails_closed():
    h = handle()
    del h._meta["armature_slots"]
    with pytest.raises(RuntimeError, match="metadata"):
        h.set_armature([.1, .2])
    assert h._runner.version == 0


@pytest.fixture(scope="module")
def robot():
    with contextlib.redirect_stdout(io.StringIO()):
        return URDFParser().parse(str(robot_urdf("iiwa14")))


@pytest.mark.parametrize("kind", ["negative", "nan", "inf", "matrix", "complex"])
def test_reference_invalid_update_is_atomic(robot, kind):
    ref = RBDReference(robot)
    good = np.full(robot.get_num_vel(), .2)
    ref.set_armature(good)
    bad = good.copy()
    if kind == "matrix":
        bad = bad[None]
    elif kind == "complex":
        bad = bad.astype(complex)
    else:
        bad[0] = {"negative": -.1, "nan": np.nan, "inf": np.inf}[kind]
    with pytest.raises(ValueError):
        ref.set_armature(bad)
    np.testing.assert_array_equal(ref.armature, good)


@pytest.mark.parametrize("floating", [False, True])
def test_constant_body_never_owns_armature_slot(tmp_path, floating):
    path = tmp_path / "constant.urdf"
    path.write_text('''<robot name="constant">
      <link name="base"><inertial><mass value="1"/><inertia ixx=".3" iyy=".4" izz=".5" ixy="0" ixz="0" iyz="0"/></inertial></link>
      <link name="mount"><inertial><mass value="1"/><inertia ixx=".3" iyy=".4" izz=".5" ixy="0" ixz="0" iyz="0"/></inertial></link>
      <link name="follower"><inertial><mass value="1"/><inertia ixx=".3" iyy=".4" izz=".5" ixy="0" ixz="0" iyz="0"/></inertial></link>
      <joint name="fixed" type="fixed"><parent link="base"/><child link="mount"/></joint>
      <joint name="constant" type="revolute"><parent link="mount"/><child link="follower"/><axis xyz="0 0 1"/><limit lower="-3" upper="3" effort="10" velocity="10"/><mimic joint="fixed" multiplier="2" offset=".17"/></joint>
    </robot>''')
    with contextlib.redirect_stdout(io.StringIO()):
        r = URDFParser().parse(str(path), floating_base=floating)
        cg = GRiDCodeGenerator(r, runtime_armature=True)
    assert r.get_num_vel() == (6 if floating else 0)
    assert cg._armature_vslots() == []
    ref = RBDReference(r)
    ref.set_armature(np.zeros(r.get_num_vel()))
    assert ref.armature_slots == ()
    assert ref.armature_regressor(np.zeros(r.get_num_vel())).shape == (r.get_num_vel(), 0)


def test_generated_armature_has_checked_native_boundaries(robot):
    cg = GRiDCodeGenerator(robot, runtime_armature=True)
    cg.gen_set_armature_params()
    code = cg.code_str
    assert "validate_armature_params" in code
    assert "set_armature_params_checked" in code
    assert "get_armature_params_checked" in code
    assert "cudaErrorInvalidValue" in code
    assert "count != 7" in code


def test_wrapper_validation_precedes_epoch_and_getter_has_admission():
    text = (Path(__file__).parents[1] / "bindings/grid_rbd/wrapper_template.cu").read_text()
    start = text.index('extern "C" int grid_rbd_set_armature_params(')
    setter = text[start:text.index('extern "C" int grid_rbd_get_armature_params(', start)]
    assert setter.index("validate_armature_params") < setter.index("++g_model_epoch")
    getter = text[text.index('extern "C" int grid_rbd_get_armature_params('):]
    getter = getter[:getter.index("\n}")]
    assert "GRID_RBD_CTX_OR_RETURN(ctx_id)" in getter


def test_checked_native_helpers_reject_before_copy(robot, tmp_path):
    """Compile the real emitted host helpers with instrumented memcpy, no CUDA."""
    cg = GRiDCodeGenerator(robot, runtime_armature=True)
    cg.gen_set_armature_params()
    source = tmp_path / "armature_host.cpp"
    source.write_text(r'''
#include <cmath>
#include <cstring>
#include <cassert>
#include <limits>
using std::isfinite;
#define __host__
enum cudaError_t { cudaSuccess, cudaErrorInvalidValue, cudaErrorUnknown };
enum cudaMemcpyKind { cudaMemcpyDeviceToHost, cudaMemcpyHostToDevice };
int copies = 0;
bool fail_copy = false;
cudaError_t cudaMemcpy(void* out, const void* in, size_t n, cudaMemcpyKind) {
    ++copies;
    if (fail_copy) return cudaErrorUnknown;
    std::memcpy(out, in, n);
    return cudaSuccess;
}
#define gpuErrchk(expr) assert((expr) == cudaSuccess)
template <typename T> struct robotModel { T* d_armature_params; };
''' + cg.code_str + r'''
int main() {
    float table[7] = {}, input[7] = {}, output[7] = {};
    robotModel<float> model{table};
    assert(set_armature_params_checked(&model, input, 6) == cudaErrorInvalidValue);
    assert(set_armature_params_checked(&model, (const float*)nullptr, 7) == cudaErrorInvalidValue);
    input[0] = -.1f;
    assert(set_armature_params_checked(&model, input, 7) == cudaErrorInvalidValue);
    input[0] = std::numeric_limits<float>::quiet_NaN();
    assert(set_armature_params_checked(&model, input, 7) == cudaErrorInvalidValue);
    assert(copies == 0 && table[0] == 0);
    input[0] = .25f;
    assert(set_armature_params_checked(&model, input, 7) == cudaSuccess);
    assert(table[0] == .25f);
    assert(get_armature_params_checked(&model, output, 7) == cudaSuccess);
    assert(output[0] == .25f);
    int before = copies;
    assert(get_armature_params_checked(&model, output, 8) == cudaErrorInvalidValue);
    assert(copies == before);
    fail_copy = true;
    input[0] = .5f;
    assert(set_armature_params_checked(&model, input, 7) == cudaErrorUnknown);
    assert(table[0] == .25f);
    assert(get_armature_params_checked(&model, output, 7) == cudaErrorUnknown);
}
''')
    exe = tmp_path / "armature_host"
    subprocess.run(["c++", "-std=c++17", str(source), "-o", str(exe)], check=True,
                   capture_output=True, text=True)
    subprocess.run([str(exe)], check=True, capture_output=True, text=True)


@pytest.mark.parametrize("runtime", [False, True])
def test_energy_armature_reaches_device_and_both_kernel_bodies(robot, tmp_path, runtime):
    path = tmp_path / "grid.cuh"
    with contextlib.redirect_stdout(io.StringIO()):
        cg = GRiDCodeGenerator(robot, runtime_armature=runtime)
        cg.gen_all_code(output_path=str(path), algorithm_list=["energy"],
                        enable_mujoco_kernels=False)
    text = path.read_text()
    for function in ("energy_device", "energy_kernel", "energy_kernel_single_timing"):
        start = text.index("{", text.index(f"void {function}("))
        depth, end = 1, start + 1
        while depth:
            depth += (text[end] == "{") - (text[end] == "}")
            end += 1
        body = text[start:end]
        if runtime:
            for slot in range(robot.get_num_vel()):
                assert f"ke += static_cast<T>(0.5) * d_robotModel->d_armature_params[{slot}]" in body
        else:
            assert "d_armature_params" not in body


@pytest.mark.parametrize("runtime", [False, True])
def test_centroidal_inertia_source_matches_storage_mode(robot, runtime):
    cg=GRiDCodeGenerator(robot)
    cg.runtime_inertia=runtime
    cg.gen_centroidal_inner()
    text=cg.code_str
    assert ("d_inertia_params[10*jid]" in text) == runtime
    assert ("const T *Ib = &d_robotModel->d_XImats" in text) == (not runtime)
