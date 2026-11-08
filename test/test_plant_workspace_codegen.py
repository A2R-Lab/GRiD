"""Public composition views must share the production arena's authorities."""
from test.test_plant_menu_codegen import _gen, _robot
from grid_codegen.algorithms._plant import _plant_hessian_shared_slots, _plant_hessian_pool_size
import pytest
import re


def test_workspace_surfaces_and_scope(tmp_path):
    gen, text = _gen(tmp_path, _robot('iiwa14'),
                     algorithm_list=['integrator_hessian', 'frame_jacobian'])
    assert '#define GRID_PLANT_HAS_HESSIAN_ARENA 1' in text
    assert 'struct tracking_cost_menu_arena' in text
    assert 'plant_hessian_layout::full_shared' in text
    assert 'plant_hessian_layout::deep_spill' in text
    assert 'workspace_capacity < workspace_bytes()' in text
    assert 'reinterpret_cast<uintptr_t>(workspace) % 16' in text
    assert 'FRAME_EVAL_DYNAMIC_SHARED_MEM_BYTES<T>()' in text
    assert 'rows > 93 ? rows : 93' in text
    assert '#ifdef GRID_CUDA_DEBUG_LAYOUT\n' in text
    assert 's_arena_offset = grid_align_up(s_arena_offset, 16);' in text
    slots = _plant_hessian_shared_slots(gen, False, False)
    assert dict(slots)['s_d2AB'] == 18 * 7 ** 3
    assert 's_d2AB' not in dict(_plant_hessian_shared_slots(gen, True, True))
    assert _plant_hessian_pool_size(gen) > 0


def test_floating_hessian_composition_is_not_silently_advertised(tmp_path):
    _, text = _gen(tmp_path, _robot('go2', True), algorithm_list=['integrator_hessian'])
    assert '#define GRID_PLANT_HAS_HESSIAN_ARENA 0' in text
    assert 'void plant_step_hessian_kernel(' in text
    assert 'enum class plant_hessian_layout' not in text


@pytest.mark.parametrize('name,target,leaves', [('iiwa14','iiwa_joint_ee',1),('go2','FR_foot_joint',4),('go2','',4)])
def test_exact_inner_pose_queries_follow_selected_target(tmp_path,name,target,leaves):
    gen,text=_gen(tmp_path,_robot(name,name=='go2'),fixed_target_name=target,
                 algorithm_list=['end_effector_pose','end_effector_pose_gradient'])
    pose_query=re.search(r'EE_TARGET_POSE_INNER_SCRATCH_BYTES\(\).*?size_t\((\d+)\)',text)
    assert pose_query and int(pose_query[1])==32*(1 if target else leaves)+3
    gradient_query=re.search(r'EE_TARGET_GRAD_INNER_SCRATCH_BYTES\(\).*?size_t\((\d+)\)',text)
    assert gradient_query and int(gradient_query[1])==gen.gen_end_effector_pose_gradient_inner_temp_mem_size(target)
    assert f'static_cast<size_t>({32*leaves+3})' in text.split('EE_POSE_INNER_SCRATCH_BYTES()',1)[1].split('\n',1)[0]
    assert 's_temp += ((size_t(0) - reinterpret_cast<size_t>(s_temp)) & size_t(15)) / sizeof(T);' in text
    assert 's_temp = reinterpret_cast<T *>(grid_align_up(reinterpret_cast<size_t>(s_temp), 16));' not in text
    # Independent bound: every legal caller residue fits both matrix banks,
    # without assuming the caller's pointer or enclosing arena is 16-aligned.
    for scalar_bytes in (4,8):
        for start in range(0,16,scalar_bytes):
            aligned=(start+15)//16*16
            padding=(-start)&15
            assert padding % scalar_bytes == 0
            assert start+(padding//scalar_bytes)*scalar_bytes == aligned
            logical=32*(1 if target else leaves)*scalar_bytes
            capacity=int(pose_query[1])*scalar_bytes
            assert aligned+logical <= start+capacity
            assert aligned % 16 == 0


def test_runtime_hessian_query_matches_resolved_kernel_layout(tmp_path):
    from grid_codegen.algorithms._plant import _plant_hessian_layout
    gen,text=_gen(tmp_path,_robot('iiwa14'),runtime_transform=True,algorithm_list=['integrator_hessian'])
    query=text.split('INTEGRATOR_HESSIAN_DYNAMIC_SHARED_MEM_BYTES()',1)[1].split('\n',1)[0]
    counts=[int(n) for n in re.findall(r'grid_shared_arena_bytes<T>\((\d+),',query)]
    expected=[]
    for pick in gen.plant_step_hessian_spill_tier_3way:
        layout=_plant_hessian_layout(gen,bool(pick),bool(pick),bool(pick))
        expected.append(sum(c for _,c in layout.t_buffers)+layout.ximat_size+layout.temp_mem_size)
    assert counts==expected


def test_empty_coordinates_do_not_advertise_hessian_composition(tmp_path):
    from test.cuda_equivalents.constant_model import constant_model
    _,text=_gen(tmp_path,constant_model(tmp_path,zero=True),algorithm_list=['integrator_hessian'])
    assert '#define GRID_PLANT_HAS_HESSIAN_ARENA 0' in text


def _constant_slot_report(case):
    from test.ee_constant_slot_warnings import CASES
    target, accesses, groups = CASES[case]
    detail = ("========= Warning: (Warp Level Programming) Potential WAR hazard detected at __shared__ 0x10\n"
              "========= Read Thread at grid::glass::dot_strided<float,4>\n"
              f"========= Write Thread at grid::{target}<float,true>\n"
              "========= Current Value : 63, Incoming Value : 63\n")
    group = ("========= Warning: Race reported between Read access at grid::glass::dot_strided<float,4>\n"
             f"========= and Write access at grid::{target}<float,true> [8 hazards]\n")
    total = accesses + groups
    return detail * accesses + group * groups + f'========= RACECHECK SUMMARY: {total} hazards displayed (0 errors, {total} warnings)\n'


@pytest.mark.parametrize('case', ['test_exact_target_scratch[float-iiwa14-iiwa_joint_ee]',
                                  'test_exact_target_scratch[float-go2-FR_foot_joint]', 'test_exact_target_scratch[float-go2-]'])
def test_known_ee_warning_classification_is_narrow(case):
    from test.ee_constant_slot_warnings import known_constant_slot_warning
    report = _constant_slot_report(case)
    assert known_constant_slot_warning(report, case)
    for damaged in (
        report.replace('Incoming Value : 63', 'Incoming Value : 62', 1),
        report.replace('Potential WAR', 'Potential RAW', 1),
        report.replace('grid::glass::dot_strided', 'other_read', 1),
        report.replace('0 errors', '1 errors'),
        report.replace('Incoming Value : 63', 'Incoming Value : unknown', 1),
        report.replace('Current Value : 63, Incoming Value : 63', 'Current Value : 42, Incoming Value : 42', 1),
        report + '========= Error: invalid memory access\n',
        report + '========= Warning: unexpected finding\n',
        report.replace('[8 hazards]', '[9 hazards]', 1),
        report.replace('Read Thread', 'Missing detail', 1),
    ):
        assert not known_constant_slot_warning(damaged, case)
    assert not known_constant_slot_warning(report, 'unreviewed_fixture')
