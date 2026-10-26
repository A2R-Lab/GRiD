"""Build, reuse and load a native C library and a declared Python extension.

Run with an installed candidate, nvcc, a supported GCC host and pybind11:
  python consumer_artifact_migration.py --urdf robot.urdf --work /tmp/new-probe --cuda-arch 120

The work directory must be new. Inputs stay at stable paths for both builds;
the example does not modify a shared cache or pin a consumer dependency.
This tests the build/load contract, not GPU numerical behavior or performance.
"""
from __future__ import annotations

import argparse
import ctypes
import importlib.util
import json
from pathlib import Path
import sysconfig

import pybind11
from grid_rbd import build_artifact, load_artifact


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--urdf", type=Path, required=True)
    parser.add_argument("--work", type=Path, required=True)
    parser.add_argument("--cuda-arch", type=int, required=True)
    args = parser.parse_args()
    work = args.work.resolve()
    work.mkdir(parents=True, exist_ok=False)
    native = work / "native.cu"
    native.write_text('#include "grid.cuh"\n'
                      'extern "C" __attribute__((visibility("default")))\n'
                      'int consumer_num_pos() { return grid::NUM_POS; }\n')
    extension = work / "python.cu"
    extension.write_text('#include <pybind11/pybind11.h>\n#include "grid.cuh"\n'
                         'PYBIND11_MODULE(grid_migration_probe, m) {\n'
                         '  m.def("num_pos", []() { return grid::NUM_POS; });\n}\n')
    common = dict(cache_dir=work / "cache", cuda_arch=args.cuda_arch,
                  algorithm_list=["end_effector_pose"], enable_mujoco_kernels=False)
    requests = {
        "native": dict(sources=[native]),
        "python": dict(sources=[extension], python_extension=True,
                       include_dirs=[pybind11.get_include(), sysconfig.get_path("include")],
                       artifact_name="grid_migration_probe" + sysconfig.get_config_var("EXT_SUFFIX")),
    }
    records = {}
    for name, options in requests.items():
        first = build_artifact("migration", name, args.urdf, **common, **options)
        again = build_artifact("migration", name, args.urdf, **common, **options)
        prepared = load_artifact("migration", name, cache_dir=common["cache_dir"], cuda_arch=args.cuda_arch)
        assert first.compiled and not again.compiled and not prepared.compiled
        assert first.content_key == again.content_key == prepared.content_key
        records[name] = {"path": str(prepared.path), "content_key": prepared.content_key,
                         "python_extension": prepared.meta["python_extension"]}
    lib = ctypes.CDLL(records["native"]["path"])
    lib.consumer_num_pos.argtypes = []
    lib.consumer_num_pos.restype = ctypes.c_int
    spec = importlib.util.spec_from_file_location("grid_migration_probe", records["python"]["path"])
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    assert lib.consumer_num_pos() == module.num_pos() > 0
    print(json.dumps({"num_pos": module.num_pos(), "artifacts": records}, sort_keys=True))


if __name__ == "__main__":
    main()
