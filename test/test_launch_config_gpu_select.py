"""W15 (2026-09-22): the baked launch-config profile is keyed on the device the
build targets. CPU-only."""
from __future__ import annotations

import warnings

from grid_codegen.launch_config import (LAUNCH_CONFIG_DEFAULT_GPU, _GPU_SELECT_WARNED,
                                        load_launch_config, select_launch_config_gpu)


def test_exact_arch_profile_is_selected():
    assert select_launch_config_gpu("iiwa14", 120) == "rtx5090_sm120"
    assert load_launch_config("iiwa14", False, gpu="rtx5090_sm120", profile="ffi")


def test_other_arch_falls_back_to_default_with_one_warning():
    _GPU_SELECT_WARNED.discard(("iiwa14", 89))
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        assert select_launch_config_gpu("iiwa14", 89) == LAUNCH_CONFIG_DEFAULT_GPU
        assert select_launch_config_gpu("iiwa14", 89) == LAUNCH_CONFIG_DEFAULT_GPU
    msgs = [str(x.message) for x in w if "no profile tuned for sm_89" in str(x.message)]
    assert len(msgs) == 1, msgs


def test_unknown_robot_or_no_arch_uses_default_silently():
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        assert select_launch_config_gpu("no_such_robot", 120) == LAUNCH_CONFIG_DEFAULT_GPU
        assert select_launch_config_gpu("iiwa14", None) == LAUNCH_CONFIG_DEFAULT_GPU
        assert select_launch_config_gpu(None, 120) == LAUNCH_CONFIG_DEFAULT_GPU
    assert not [x for x in w if "no profile tuned" in str(x.message)]


def test_generator_bakes_the_selected_profile_name(tmp_path):
    from grid_codegen import GRiDCodeGenerator
    from URDFParser import URDFParser
    robot = URDFParser().parse("config/robot_assets/iiwa14.urdf", floating_base=False)
    gen = GRiDCodeGenerator(robot, DEBUG_MODE=False, NEED_PRINT_MAT=False, FILE_NAMESPACE="grid",
                            LAUNCH_CONFIG_ROBOT="iiwa14", LAUNCH_CONFIG_GPU="rtx5090_sm120")
    gen.gen_all_code(algorithm_list=["inverse_dynamics"], output_path=str(tmp_path / "g.cuh"),
                     enable_mujoco_kernels=False)
    assert "config/launch_configs/iiwa14/rtx5090_sm120.json" in (tmp_path / "g.cuh").read_text()
