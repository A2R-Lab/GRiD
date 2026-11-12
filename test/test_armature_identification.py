"""Opt-in identification contract and fused-operation structure (CPU lane)."""
import contextlib
import io
from pathlib import Path
import re

from grid_assets import robot_urdf
from grid_rbd._compile import generate_grid_cuh
from grid_codegen.abi_specs import ABI_SPECS, expand_py_out_dims


def test_identification_reuses_one_fd_and_no_regressor_scratch(tmp_path):
    with contextlib.redirect_stdout(io.StringIO()):
        generate_grid_cuh(robot_urdf("iiwa14"),
            dict(algorithm_list=["forward_dynamics_armature_gradient"],
                 runtime_armature=True, enable_mujoco_kernels=False), tmp_path / "grid.cuh")
    source = (tmp_path / "grid.cuh").read_text()
    assert "constexpr int NUM_ARMATURE_PARAMS = 7;" in source
    assert "armature_gradient_finish<T>(out" in source
    assert "out[i] = -Minv[index] * qdd[slot]" in source
    assert "inverse_dynamics_regressor_inner(" not in source
    assert 'grid_cleanup_device_free(hd_data->pool, hd_data->d_dqdd_da, "grid_device_free(d_dqdd_da)"' in source
    assert expand_py_out_dims(ABI_SPECS["forward_dynamics_armature_gradient"], 7, 7, 1, 7, 7) == (7, 7)


def test_ordinary_build_has_no_identification_allocation(tmp_path):
    with contextlib.redirect_stdout(io.StringIO()):
        generate_grid_cuh(robot_urdf("iiwa14"),
            dict(algorithm_list=["forward_dynamics"], enable_mujoco_kernels=False), tmp_path / "grid.cuh")
    source = (tmp_path / "grid.cuh").read_text()
    assert "d_dqdd_da" not in source
    assert "armature_gradient_kernel" not in source


def test_spherical_empty_parameter_block_is_supported(tmp_path):
    import xml.etree.ElementTree as ET
    fixture=Path(__file__).resolve().parents[1]/"external/URDFParser/tests/fixtures/spherical_arm.urdf"
    tree=ET.parse(fixture)
    tree.find(".//joint[@name='joint_2']").set("type","fixed")
    source=tmp_path/"only_spherical.urdf"
    tree.write(source)
    with contextlib.redirect_stdout(io.StringIO()):
        generate_grid_cuh(source,dict(algorithm_list=["forward_dynamics_armature_gradient"],
            enable_mujoco_kernels=False),tmp_path/"grid.cuh")
    text=(tmp_path/"grid.cuh").read_text()
    assert "constexpr int NUM_ARMATURE_PARAMS = 0;" in text
    assert "armature_gradient_kernel" in text


def test_sensitivity_is_one_admitted_native_operation():
    from grid_codegen import wrapper_body_gen
    source = wrapper_body_gen.gen_body(ABI_SPECS["forward_dynamics_armature_gradient"])
    assert source.count("GRID_RBD_CTX_OR_RETURN(ctx_id)") == 1
    assert source.count("grid::forward_dynamics_armature_gradient<T") == 1
    assert "grid::minv<" not in source and "grid::forward_dynamics<" not in source
    assert "apply_f_ext(g_ctx, f_ext, batch)" in source
    assert "reset_f_ext(g_ctx, f_ext, batch)" in source


def test_collective_size_does_not_shrink_with_kernel_spill(tmp_path):
    from URDFParser import URDFParser
    from grid_codegen import GRiDCodeGenerator
    with contextlib.redirect_stdout(io.StringIO()):
        robot=URDFParser().parse(str(robot_urdf("iiwa14")))
        cg=GRiDCodeGenerator(robot)
        cg.cuda_target_shared_mem_bytes=4096
        cg.gen_all_code(algorithm_list=["forward_dynamics_armature_gradient"],
            enable_mujoco_kernels=False,emit_alloc_gating=True,output_path=str(tmp_path/"grid.cuh"))
    assert cg.fd_spill_tier_3way[0]==1
    def count(name):
        return int(re.search(name+r"\(\) \{ return grid_shared_arena_bytes<T>\((\d+),",cg.code_str)[1])
    assert count("FORWARD_DYNAMICS_ARMATURE_GRADIENT_DEVICE_SHARED_MEM_BYTES")==count("FORWARD_DYNAMICS_DEVICE_DYNAMIC_SHARED_MEM_BYTES")+robot.get_num_vel()
    workspace_guards=[line for line in cg.code_str.splitlines() if line.lstrip().startswith("#if") and "GRID_ALLOC_MINV" in line]
    assert workspace_guards
    assert all("GRID_ALLOC_FORWARD_DYNAMICS_ARMATURE_GRADIENT" in line for line in workspace_guards)


def test_direct_backend_launch_requests_are_clamped():
    source = (Path(__file__).resolve().parents[1]/"bindings/grid_rbd/wrapper_template.cu").read_text()
    sites = re.findall(r"<<<(.*?)>>>", source, re.S)
    requested = [site for site in sites if "grid_rbd_launch_threads_n<" in site]
    assert len(requested) > 20
    for site in requested:
        assert "grid_clamp_threads_for(" in site, site
