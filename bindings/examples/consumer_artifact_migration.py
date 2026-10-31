"""Build, reuse and load a native C library and a declared Python extension.

Run with an installed candidate, nvcc, a supported GCC host and pybind11:
  python consumer_artifact_migration.py --urdf robot.urdf --work /tmp/new-probe --cuda-arch 120

The work directory must be new. Inputs stay at stable paths for both builds;
the example does not modify a shared cache or pin a consumer dependency.
This tests the build/load contract, not GPU numerical behavior or performance.
The last phase re-runs in a fresh process that cannot see the URDF or nvcc and
loads both artifacts through the prepared path with the expected descriptor
identity, which is how a deployed application without a toolchain should start.
"""
from __future__ import annotations

import argparse
import ctypes
import importlib.util
import json
from pathlib import Path
import sysconfig

import os
import shutil
import subprocess
import sys

import pybind11
from grid_rbd import build_artifact, describe_model, load_artifact

PREPARED_PHASE = "GRID_MIGRATION_EXAMPLE_PREPARED"


def prepared_phase():
    """Fresh process: no URDF, no compiler on PATH, prepared load with the expected identity."""
    spec = json.loads(os.environ[PREPARED_PHASE])
    assert shutil.which("nvcc") is None, "the prepared phase must not see a compiler"
    assert not Path(spec["urdf"]).exists() or spec["urdf_hidden"], "the prepared phase must not need the URDF"
    native = load_artifact("migration", "native", cache_dir=spec["cache_dir"], cuda_arch=spec["cuda_arch"],
                           expected_descriptor_id=spec["descriptor_id"])
    python = load_artifact("migration", "python", cache_dir=spec["cache_dir"], cuda_arch=spec["cuda_arch"],
                           expected_descriptor_id=spec["descriptor_id"])
    assert not native.compiled and not python.compiled
    assert native.descriptor["descriptor_id"] == python.descriptor["descriptor_id"] == spec["descriptor_id"]
    lib = ctypes.CDLL(str(native.path))
    lib.consumer_num_pos.restype = ctypes.c_int
    module_spec = importlib.util.spec_from_file_location("grid_migration_probe", str(python.path))
    module = importlib.util.module_from_spec(module_spec)
    module_spec.loader.exec_module(module)
    assert lib.consumer_num_pos() == module.num_pos() == spec["num_pos"]
    print(json.dumps({"prepared_phase": "ok", "num_pos": module.num_pos()}, sort_keys=True))


def main():
    if PREPARED_PHASE in os.environ:
        return prepared_phase()
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
    model = describe_model(args.urdf)
    common = dict(cache_dir=work / "cache", cuda_arch=args.cuda_arch,
                  algorithm_list=["end_effector_pose"], enable_mujoco_kernels=False,
                  expected_descriptor_id=model["descriptor_id"])
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

    # Deployment check: a fresh interpreter with no compiler on PATH and a URDF path that
    # does not exist loads both artifacts through the prepared path and gets the same value.
    toolchain_free = os.pathsep.join(p for p in os.environ.get("PATH", "").split(os.pathsep)
                                     if not (Path(p) / "nvcc").exists())
    spec = {"cache_dir": str(common["cache_dir"]), "cuda_arch": args.cuda_arch,
            "descriptor_id": model["descriptor_id"], "num_pos": module.num_pos(),
            "urdf": str(work / "missing.urdf"), "urdf_hidden": True}
    env = {**os.environ, "PATH": toolchain_free, PREPARED_PHASE: json.dumps(spec)}
    result = subprocess.run([sys.executable, "-I", __file__], env=env, cwd=work,
                            capture_output=True, text=True)
    if result.returncode != 0:
        raise SystemExit("prepared phase failed:\n" + result.stdout + result.stderr)
    prepared = json.loads(result.stdout.strip().splitlines()[-1])
    print(json.dumps({"num_pos": module.num_pos(), "artifacts": records,
                      "prepared_phase": prepared["prepared_phase"]}, sort_keys=True))


if __name__ == "__main__":
    main()
