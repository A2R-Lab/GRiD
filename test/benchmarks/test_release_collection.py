"""CPU-only regression tests for the release measurement/reporting contract."""
import ctypes
import json
from pathlib import Path
import subprocess
import sys
import numpy as np
import pytest

from test.benchmarks.release import protocol as p
from test.benchmarks.release.collect import command_output, run_job
from test.benchmarks.release.report import aggregate, records


def test_stage_sizes_and_exact_batches():
    assert p.BATCHES == (16,32,64,128,256,1024)
    assert len(list(p.jobs("core", p.ROBOTS))) == 45
    assert len(list(p.jobs("wrappers", p.ROBOTS))) == 48      # 3 robots x 2 ops x (5 surfaces + 3 allocate-once)
    assert len(list(p.jobs("table", p.ROBOTS))) == 3*len(p.OPERATIONS)*len(p.TABLE_BACKENDS)
    assert all(job["backend"] == "grid_cuda" for job in list(p.jobs("core", ["iiwa14"]))[::len(p.PRIMARY["inverse_dynamics"])][:1])


@pytest.mark.parametrize("args", [("core",["fake"]), ("core",["iiwa14"],["fake"]),
    ("core",["iiwa14"],None,["fake"]), ("core",["iiwa14"],None,["minv"])])
def test_invalid_selection(args):
    with pytest.raises(ValueError): list(p.jobs(*args))


def test_capability_gaps_are_not_library_claims():
    assert p.capability("pinocchio","idsva_so","g1") is None
    assert p.capability("mjx","idsva_so","iiwa14").startswith("excluded_method:")
    assert p.capability("frax","inverse_dynamics","g1").startswith("model_mismatch:")
    assert p.capability("pinocchio","end_effector_pose_hessian","iiwa14").startswith("adapter_pending:")
    assert all(p.capability("grid_cuda",op,"g1") is None for op in p.CORE)
    assert p.capability("grid_cuda","minv","g1") is None and p.capability("grid_cuda","end_effector_pose_hessian","g1").startswith("adapter_pending:")
    assert p.capability("grid_native","idsva_so","g1").startswith("adapter_pending:")
    assert p.capability("pinocchio_plain","ccrba","g1") is None and p.capability("pinocchio","ccrba","g1").startswith("adapter_pending:")
    assert p.capability("mjx","crba","go2") is None and p.capability("mujoco_warp","crba","go2") is None
    assert p.capability("bard","generalized_gravity","g1") is None and p.capability("frax","crba","iiwa14") is None
    assert p.capability("mjx","ccrba","iiwa14").startswith("adapter_pending:")
    # allocate-once companions: NumPy only where the API has out=; torch/JAX everywhere
    assert p.capability("grid_numpy_prealloc","inverse_dynamics","g1").startswith("not_applicable:")
    assert all(p.capability("grid_numpy_prealloc",op,"g1") is None for op in p.NUMPY_OUT_OPS)
    assert all(p.capability(b,op,"g1") is None for b in ("grid_torch_prealloc","grid_jax_prealloc") for op in p.CORE)
    assert set(p.PREALLOC) <= set(p.WRAPPERS) and set(p.PREALLOC.values()) <= set(p.WRAPPERS)


def test_jax_pinned_floor_matches_the_bindings_and_selects_cells():
    """The figure draws a JAX allocate-once bar only where to_host took the pinned route."""
    import re
    source = (p.ROOT / "bindings/grid_rbd/jax/__init__.py").read_text()
    floor = re.search(r"^_PINNED_MIN_BYTES = (.+)$", source, re.M).group(1)
    assert eval(floor, {"__builtins__": {}}) == p.JAX_PINNED_MIN_BYTES
    row = lambda op, entries: {"operation": op, "entries": entries}
    assert not p.jax_pinned_route(row("inverse_dynamics", 1024 * 35))                 # 140 KiB
    assert p.jax_pinned_route(row("inverse_dynamics_gradient", 1024 * 2 * 7 * 7))     # 392 KiB
    assert not p.jax_pinned_route(row("idsva_so", 128 * 4 * 7 ** 3))                  # 4 x 171.5 KiB
    assert p.jax_pinned_route(row("idsva_so", 256 * 4 * 7 ** 3))                      # 4 x 343 KiB
    assert not p.jax_pinned_route({"operation": "idsva_so", "entries": None})


def test_numpy_out_ops_match_the_bindings_table():
    """The benchmark's notion of "NumPy has out=" is the bindings' py_out_param flag."""
    from grid_codegen.abi_specs import ABI_SPECS
    assert p.NUMPY_OUT_OPS == {k for k, s in ABI_SPECS.items() if s.py_out_param}


def test_accuracy_checks_shapes_blocks_finiteness_and_near_zero():
    assert p.agreement(np.zeros(4),np.zeros(4))["passed"]
    for a,b in [(np.zeros(4),np.zeros(3)), ([np.zeros(4)], [np.zeros(4),np.zeros(4)]),
                ([np.nan],[0]), ([np.inf],[np.inf]), ([.1],[0])]:
        assert not p.agreement(a,b)["passed"]
    assert not p.agreement([1.00001],[1],rtol=0,atol=0)["passed"]


def test_explicit_fp32_fd_warning_policy_preserves_hard_failures():
    check = p.agreement(np.array([1000., .01]), np.array([1000., 0.]))
    assert not check['passed']
    assert p.accuracy_status(check, 'strict', 'fdsva_so', 'float32') == 'validation_failed'
    assert p.accuracy_status(check, 'fp32-fd-warnings', 'fdsva_so', 'float32') == 'accuracy_warning'
    for op,dtype in [('idsva_so','float32'), ('fdsva_so','float64')]:
        assert p.accuracy_status(check, 'fp32-fd-warnings', op, dtype) == 'validation_failed'
    for actual,expected in [([1.],[0.]), ([np.nan],[0.]), ([np.inf],[1.]), ([1.,2.],[1.])]:
        assert p.accuracy_status(p.agreement(actual,expected), 'fp32-fd-warnings', 'fdsva_so', 'float32') == 'validation_failed'
    assert check['blocks'][0]['entries']==2
    assert check['blocks'][0]['max_abs_reference_below_atol']==.01


def test_warning_repeat_is_retained_but_never_relabeled_validated():
    a,b=row(0),row(1)
    for r in (a,b):r.update(expected_repeats=2,accuracy_policy='fp32-fd-warnings')
    b.update(status='accuracy_warning',bad_entries=3,entries=100)
    out=aggregate([a,b])[0]
    assert out['status']=='accuracy_warning' and out['host_us']==10.
    assert out['max_bad_entries']==3


@pytest.mark.parametrize('tamper', ['none', 'strict', 'nonfinite', 'unstable'])
def test_warning_export_requires_policy_and_safety_checks(tmp_path,tamper):
    job=list(p.jobs('table',['iiwa14'],['grid_jax'],['forward_dynamics_gradient']))[0]
    check=p.agreement(np.array([1000.,.01]),np.array([1000.,0.]))
    capture=dict(accuracy_policy='fp32-fd-warnings',adapter=dict(dtype='float32'),cells=[dict(
        batch=16,status='accuracy_warning',comparison_eligible=True,oracle_agreement=check,
        post_timing_agreement=check,repeatability_agreement=dict(passed=True),host_to_host=dict(mean_us=10.))])
    if tamper=='nonfinite':capture['cells'][0]['oracle_agreement']=dict(passed=False,reason='non-finite output')
    if tamper=='unstable':capture['cells'][0]['repeatability_agreement']=dict(passed=False)
    p.write_json(tmp_path/'plan.json',dict(jobs=[job],batches=[16],repeats=1,purpose='smoke',iterations=2,warmups=2,
        accuracy_policy='strict' if tamper=='strict' else 'fp32-fd-warnings'))
    p.write_json(tmp_path/'capture.json',capture)
    p.write_json(tmp_path/'results.json',dict(jobs=[dict(job,repeat=0,capture='capture.json',sha256=p.digest(tmp_path/'capture.json'))]))
    r=list(records(tmp_path))[0]
    assert r['host_us']==(10. if tamper=='none' else None)


def variation_cell():
    good=p.agreement(np.array([1000.,0.]),np.array([1000.,0.]))
    small=p.agreement(np.array([1000.,.01]),np.array([1000.,0.]))
    return dict(oracle_agreement=good,post_timing_agreement=good,
        resident_oracle_agreement=good,resident_post_timing_agreement=good,
        boundary_agreement=small,post_boundary_agreement=small,
        repeatability_agreement=small,resident_repeatability_agreement=small)


def test_bounded_variation_requires_v2_policy_and_every_oracle_check():
    cell=variation_cell()
    decide=lambda c,**kw:p.cell_accuracy_status(c,'fp32-fd-warnings','forward_dynamics','float32',**kw)
    assert decide(cell)=='accuracy_warning'
    assert decide(cell,version=1)=='validation_failed'
    assert decide(cell,version=999)=='validation_failed'
    assert p.cell_accuracy_status(cell,'strict','forward_dynamics','float32')=='validation_failed'
    for key in cell:
        missing=cell.copy();missing.pop(key)
        assert decide(missing)=='validation_failed'
        for bad in (p.agreement(np.array([1.]),np.array([0.])),
                    p.agreement(np.array([np.nan]),np.array([0.])),
                    p.agreement(np.zeros(2),np.zeros(3))):
            changed={**cell,key:bad}
            assert decide(changed)=='validation_failed'
    assert decide({**cell,'native_wrapper_agreement':dict(passed=False)})=='validation_failed'
    for op,dtype in [('inverse_dynamics','float32'),('forward_dynamics','float64')]:
        assert p.cell_accuracy_status(cell,'fp32-fd-warnings',op,dtype)=='validation_failed'


@pytest.mark.parametrize('tamper',['none','missing_resident_oracle','bad_post_oracle','wrong_version'])
def test_v2_variation_export_requires_both_paths_oracle_validation(tmp_path,tamper):
    job=list(p.jobs('table',['g1'],['mjx'],['forward_dynamics']))[0]
    cell=dict(variation_cell(),batch=16,status='accuracy_warning',comparison_eligible=True,
        host_to_host=dict(mean_us=10.),resident=dict(mean_us=5.))
    if tamper=='missing_resident_oracle':cell.pop('resident_oracle_agreement')
    if tamper=='bad_post_oracle':cell['resident_post_timing_agreement']=p.agreement(np.ones(2),np.zeros(2))
    capture=dict(accuracy_policy='fp32-fd-warnings',accuracy_policy_version=1 if tamper=='wrong_version' else 2,
        adapter=dict(dtype='float32'),cells=[cell])
    p.write_json(tmp_path/'plan.json',dict(jobs=[job],batches=[16],repeats=1,purpose='smoke',iterations=2,warmups=2,
        accuracy_policy='fp32-fd-warnings',accuracy_policy_version=2))
    p.write_json(tmp_path/'capture.json',capture)
    p.write_json(tmp_path/'results.json',dict(jobs=[dict(job,repeat=0,capture='capture.json',sha256=p.digest(tmp_path/'capture.json'))]))
    r=list(records(tmp_path))[0]
    assert r['host_us']==(10. if tamper=='none' else None)
    assert r['variation_max_abs_error']==.01


def test_preparation_plan_and_report_rejection(tmp_path):
    result=subprocess.run([sys.executable,'-m','test.benchmarks.release.collect','--stage','core','--prepare-only'],
        cwd=p.ROOT,text=True,capture_output=True,check=True)
    plan=json.loads(result.stdout)
    assert plan['purpose']=='preparation' and plan['repeats']==1 and plan['batches']==list(p.BATCHES)
    p.write_json(tmp_path/'plan.json',plan)
    with pytest.raises(ValueError,match='Preparation'):list(records(tmp_path))


def test_jax_cache_environment_is_persistent_and_owner_controlled(tmp_path,monkeypatch):
    from test.benchmarks.release import collect
    cache=tmp_path/'private-cache'
    monkeypatch.setattr(collect,'JAX_CACHE',cache)
    env=collect.worker_environment()
    assert env['JAX_COMPILATION_CACHE_DIR']==str(cache)
    assert env['JAX_PERSISTENT_CACHE_MIN_COMPILE_TIME_SECS']=='0'
    assert cache.stat().st_mode & 0o777 == 0o700
    cache.chmod(0o777)
    with pytest.raises(PermissionError):collect.worker_environment()


@pytest.mark.parametrize('finite',[True,False])
def test_preparation_worker_never_times_or_claims_accuracy(tmp_path,monkeypatch,finite):
    from test.benchmarks.release import worker, fixtures, grid_adapter
    calls=[]
    class FakeFixture:
        def __init__(self,robot,count):
            self.q=self.v=self.a=self.u=np.zeros((count,2),np.float32)
            self.metadata={}
        def expected(self,*args):raise AssertionError('Preparation must not run an oracle')
    class FakeAdapter:
        metadata=dict(dtype='float32')
        def __init__(self,*args):pass
        def prepare(self,batch):
            def call(mode):
                calls.append((batch,mode))
                return np.array([1. if finite else np.nan])
            self.host=lambda:call('host')
            self.resident=lambda:call('resident')
        def normalize(self,x):return x
    def no_timer(*args):raise AssertionError('Preparation must not collect timings')
    monkeypatch.setattr(fixtures,'Fixture',FakeFixture)
    monkeypatch.setattr(grid_adapter,'GridAdapter',FakeAdapter)
    monkeypatch.setattr(worker,'timed',no_timer)
    path=tmp_path/'prepared.json'
    monkeypatch.setattr(sys,'argv',['worker','--robot','g1','--backend','grid_jax','--operation','inverse_dynamics',
        '--batches','16','32','--warmups','1','--iterations','1','--prepare-only','--output',str(path)])
    if finite:worker.main()
    else:
        with pytest.raises(SystemExit):worker.main()
    capture=json.loads(path.read_text())
    assert capture['purpose']=='preparation'
    assert len(capture['cells'])==2
    for c in capture['cells']:
        assert c['status']==('prepared' if finite else 'error')
        assert not c['comparison_eligible'] and 'host_to_host' not in c and 'resident' not in c
    if finite:assert calls==[(16,'host'),(16,'resident'),(32,'host'),(32,'resident')]


def test_codegen_constants_use_round_trip_precision():
    # Pinocchio's default CppADCodeGen digits10=6 loses fp32 constants.
    value = np.float32(0.123456789)
    assert np.float32(format(value, '.6g')) != value
    assert np.float32(format(value, '.9g')) == value
    source = (p.ROOT / 'test/benchmarks/release/pin_codegen_init.h').read_text()
    assert 'setParameterPrecision(std::numeric_limits<float>::max_digits10)' in source
    bridge = (p.ROOT / 'test/benchmarks/release/pin_bridge.cpp').read_text()
    assert '#include "pin_codegen_init.h"' in bridge
    for name in ('rnea', 'grad', 'minv'):
        assert f'init_release_codegen(*{name});' in bridge


def oracle_loader():
    import importlib.util
    path = p.ROOT / 'external/RBDReference/equivalents/pin_so_ext/__init__.py'
    spec = importlib.util.spec_from_file_location('test_oracle_loader', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_oracle_loader_builds_once_and_rejects_stale_result(tmp_path, monkeypatch):
    loader = oracle_loader()
    monkeypatch.setattr(loader, '_DIR', tmp_path)
    monkeypatch.setattr(sys, 'path', list(sys.path))
    so = tmp_path / 'pin_so_ext.test.so'
    source = tmp_path / 'setup.py'
    source.touch()
    built = []
    def build():
        import os
        built.append(True)
        so.touch()
        os.utime(so, (source.stat().st_mtime + 1, source.stat().st_mtime + 1))
    monkeypatch.setattr(loader, '_build', build)
    marker = object()
    monkeypatch.setattr(loader.importlib, 'import_module', lambda name: marker)
    assert loader.load() is marker
    assert loader.load() is marker
    assert len(built) == 1
    monkeypatch.setattr(loader, '_is_stale', lambda path: True)
    with pytest.raises(RuntimeError, match='current pin_so_ext'):
        loader.load()


def test_oracle_rebuild_forces_relink(monkeypatch):
    loader = oracle_loader()
    commands = []
    monkeypatch.setattr(loader.subprocess, 'check_call', lambda cmd, **kw: commands.append(cmd))
    monkeypatch.setattr(loader, '_build_env', lambda: {})
    loader._build()
    assert commands[0][-1] == '--force'


def test_timer_syncs_every_call_and_preserves_samples():
    calls, syncs = [], []
    def call():
        calls.append(len(calls)); return calls[-1]
    out = p.timed(call,syncs.append,2,3)
    assert calls == syncs == list(range(5))
    assert len(out["samples_us"]) == 3 and min(out["samples_us"]) >= 0
    with pytest.raises(ValueError): p.timed(call,syncs.append,0,3)
    with pytest.raises(ValueError): p.timed(call,syncs.append,1,1,-1)


def test_timer_sustains_warmup_for_the_requested_wall_time():
    import time
    calls=[]
    def call():
        calls.append(time.perf_counter()); return len(calls)
    start=time.perf_counter()
    out=p.timed(call,lambda out:None,1,2,warm_seconds=.05)
    warm_calls=len(calls)-2
    # The last warm-up may START before the deadline and finish after it.
    # The first measured call must start after the requested warm-up duration.
    # A preempted warm-up can consume the whole interval in one call.
    assert warm_calls >= 1 and calls[warm_calls]-start >= .05
    assert len(out["samples_us"]) == 2


def test_gradient_blocks_split_position_and_velocity_columns():
    from test.benchmarks.release.worker import gradient_blocks
    j=np.arange(2*3*6,dtype=np.float64).reshape(2,3,6)
    dq,dv=gradient_blocks(j,3)
    np.testing.assert_array_equal(dq,j[...,:3]); np.testing.assert_array_equal(dv,j[...,3:])
    # A wholly wrong velocity block must fail the gross-error backstop when checked per block.
    wrong=j.copy(); wrong[...,3:]=0
    whole=p.agreement(wrong*1e-3,j*1e-3)
    split=p.agreement(gradient_blocks(wrong*1e-3,3),gradient_blocks(j*1e-3,3))
    assert whole["blocks"][0]["relative_l2"] < 1 and split["blocks"][1]["relative_l2"] == 1.


def test_negative_missing_overhead_never_becomes_zero():
    assert p.overhead(8,3) == 5
    for pair in [(None,3),(8,None),(2,3),(float("nan"),3),(float("inf"),3)]:
        assert p.overhead(*pair) is None


def test_atomic_json_rejects_nan_without_overwriting(tmp_path):
    path = tmp_path/"a.json"
    p.write_json(path,{"old":True})
    with pytest.raises(ValueError): p.write_json(path,{"value":float("nan")})
    assert json.loads(path.read_text()) == {"old":True}


def row(repeat=0, **kw):
    return dict(robot="iiwa14", operation="inverse_dynamics", backend="grid_jax", batch=16,
        repeat=repeat, expected_repeats=1, purpose="smoke", dtype="float32", method="analytical",
        status="validated", reason="", host_us=10., resident_us=4., contract="x",
        urdf_sha256="u",input_values_sha256="i",max_abs_error=0.,relative_l2_error=0.,**kw)


def test_aggregation_uses_median_of_means_not_median_samples():
    rows=[row(i) for i in range(3)]
    for r,t in zip(rows,[8.,20.,12.]): r.update(expected_repeats=3,host_us=t)
    a=aggregate(rows)[0]
    assert (a["host_us"],a["host_min_us"],a["host_max_us"],a["overhead_us"]) == (12,8,20,8)


def test_failed_or_incomplete_repeats_produce_no_timing():
    a=row(); a.update(status="validation_failed",host_us=None)
    assert aggregate([a])[0]["host_us"] is None
    a=row(); a["expected_repeats"]=3
    assert aggregate([a])[0]["status"] == "incomplete"
    with pytest.raises(ValueError): aggregate([row(),row()])


def test_changed_input_or_contract_refuses_aggregation():
    for field in ("input_values_sha256","urdf_sha256","contract","dtype","input_storage_dtype"):
        aa,bb=row(0),row(1)
        aa["expected_repeats"]=bb["expected_repeats"]=2
        bb[field]="changed"
        assert aggregate([aa,bb])[0]["status"] == "contract_mismatch"


def test_report_contract_retains_sustained_warmup_and_input_dtype(tmp_path):
    job=list(p.jobs('core',['iiwa14'],['pinocchio'],['inverse_dynamics']))[0]
    capture=dict(warm_seconds=1.5, adapter=dict(dtype='float32', input_storage_dtype='float32'),
        cells=[dict(batch=16,status='validated',comparison_eligible=True,
                    oracle_agreement=dict(passed=True),host_to_host=dict(mean_us=10.))])
    p.write_json(tmp_path/'plan.json',dict(jobs=[job],batches=[16],repeats=1,
        purpose='smoke',iterations=2,warmups=2,warm_seconds=1.5))
    p.write_json(tmp_path/'capture.json',capture)
    def reread():
        p.write_json(tmp_path/'results.json',dict(jobs=[dict(job,repeat=0,
            capture='capture.json',sha256=p.digest(tmp_path/'capture.json'))]))
        return list(records(tmp_path))[0]
    first=reread()
    assert first['input_storage_dtype']=='float32'
    assert json.loads(first['contract'])['warm_seconds']==1.5
    capture['warm_seconds']=0.
    p.write_json(tmp_path/'capture.json',capture)
    assert reread()['contract']!=first['contract']


def test_negative_delta_is_flagged_not_clipped():
    a=row(); a["resident_us"]=11.
    out=aggregate([a])[0]
    assert out["overhead_us"] is None and out["boundary_flag"]


def test_positive_aggregate_delta_does_not_erase_bad_repeat():
    from test.benchmarks.release.report import decompose, grid_stack
    rows = [dict(robot="iiwa14", operation="inverse_dynamics", batch=16,
                 backend="grid_cuda", host_us=12., resident_us=4.,
                 boundary_flag="negative total-minus-resident in one repeat"),
            dict(robot="iiwa14", operation="inverse_dynamics", batch=16,
                 backend="grid_jax", host_us=20., resident_us=10., boundary_flag="")]
    d = decompose(rows)[0]
    assert d["memory_traffic_us"] is None
    assert "inconsistent per-repeat boundaries" in d["flags"]
    lookup = {(r["robot"], r["operation"], r["backend"], r["batch"]): r for r in rows}
    assert grid_stack(lookup, "iiwa14", "inverse_dynamics", 16, "grid_jax") == (4., None, 8., ["memory"])


def test_incomplete_stack_draws_actual_total_not_partial_sum():
    import matplotlib.pyplot as plt
    from test.benchmarks.release.report import draw_grid_stack
    fig, ax = plt.subplots()
    try:
        total = draw_grid_stack(ax, 0., .5, (4., 8., None, ["wrapper"]),
                                dict(host_us=10., host_min_us=9., host_max_us=11.))
        assert total == 10.
        assert [p.get_height() for p in ax.patches] == [10.]
        assert ax.lines[0].get_ydata()[0] == 10.
    finally:
        plt.close(fig)


def test_speedup_palette_white_at_parity_red_below_blue_above():
    import math
    from matplotlib.colors import TwoSlopeNorm
    from test.benchmarks.release.report import SPEEDUP_CMAP
    norm = TwoSlopeNorm(vmin=-2., vcenter=0., vmax=2.)
    assert SPEEDUP_CMAP(norm(0.))[:3] == (1., 1., 1.)
    for ratio in (.01, .1, .5, .9):
        red, _, blue, _ = SPEEDUP_CMAP(norm(math.log10(ratio)))
        assert red > blue
    for ratio in (1.1, 2., 10., 100.):
        red, _, blue, _ = SPEEDUP_CMAP(norm(math.log10(ratio)))
        assert blue > red


def test_heatmap_annotation_contrast_tracks_actual_palette():
    import matplotlib.pyplot as plt
    from test.benchmarks.release.report import _ratio_heatmap
    fig, ax = plt.subplots()
    try:
        _ratio_heatmap(ax, [[.5, 1., 10., 100.]], ["cell"], [1, 2, 3, 4], "")
        assert [t.get_color() for t in ax.texts] == ["#0b0b0b", "#0b0b0b", "#0b0b0b", "white"]
    finally:
        plt.close(fig)


def test_website_api_plot_uses_requested_boundaries_and_palette(tmp_path, monkeypatch):
    def reject_whiskers(*args, **kwargs):
        raise AssertionError("The homepage API figure must not draw whiskers")
    from matplotlib.axes import Axes
    monkeypatch.setattr(Axes, "errorbar", reject_whiskers)
    from docs.plot_release_figures import API_BOUNDARIES, api_boundaries
    # a resident (no-I/O) group — CUDA Device, PyTorch resident, JAX resident — then the host-call group
    assert [x[0] for x in API_BOUNDARIES] == ["CUDA Device", "PyTorch resident", "JAX resident", "C++ Host", "NumPy", "PyTorch", "JAX"]
    assert [x[3] for x in API_BOUNDARIES] == ["#00693e", "#f0a58a", "#c3adcb", "#c4dd88", "#267aba", "#d94415", "#8a6996"]
    rows = []
    for op in p.WRAPPER_OPS:
        for backend, host in (("grid_cuda", 20.), ("grid_native", 25.), ("grid_numpy", 30.),
                              ("grid_torch", 40.), ("grid_jax", 50.)):
            rows.append(dict(robot="iiwa14", operation=op, backend=backend, batch=16,
                             host_us=host, host_min_us=host-1, host_max_us=host+1,
                             resident_us=10., resident_min_us=9., resident_max_us=11.))
    original = Axes.bar
    heights = []
    def capture(ax, x, height, *args, **kwargs):
        heights.append(height)
        assert "bottom" not in kwargs  # these are measured totals, not stacks
        return original(ax, x, height, *args, **kwargs)
    monkeypatch.setattr(Axes, "bar", capture)
    path = api_boundaries(rows, tmp_path)
    assert heights == [10., 10., 10., 20., 30., 40., 50.]*2   # resident group (all 10.) then the host calls
    svg = path.read_text()
    assert "C ABI" not in svg and "DRAFT" not in svg


def test_different_backend_capture_contracts_do_not_make_a_comparison():
    a,b=row(),row(); b.update(backend="mjx",contract="different machine")
    assert all(r["host_us"] is None and r["status"] == "contract_mismatch" for r in aggregate([a,b]))


def test_missing_jobs_expand_all_batches_and_repeats(tmp_path):
    p.write_json(tmp_path/"plan.json",dict(jobs=list(p.jobs("core",["iiwa14"],["mjx"],["idsva_so"])),
        batches=[16,32],repeats=2,purpose="smoke",iterations=2,warmups=2))
    out=list(records(tmp_path))
    assert len(out)==4 and all(r["status"]=="excluded_method" for r in out)
    assert all(r["host_us"] is None for r in out)


def test_corrupt_capture_rejected(tmp_path):
    job=list(p.jobs("core",["iiwa14"],["mjx"],["inverse_dynamics"]))[0]
    p.write_json(tmp_path/"plan.json",dict(jobs=[job],batches=[16],repeats=1,purpose="smoke",iterations=2,warmups=2))
    p.write_json(tmp_path/"a.json",{})
    p.write_json(tmp_path/"results.json",dict(jobs=[dict(job,repeat=0,capture="a.json",sha256="wrong")]))
    with pytest.raises(ValueError,match="hash mismatch"): list(records(tmp_path))


def test_cli_dry_run_and_bad_tool_no_gpu_needed():
    result=subprocess.run([sys.executable,"-m","test.benchmarks.release.collect","--stage","wrappers","--smoke"],cwd=p.ROOT,text=True,capture_output=True,check=True)
    plan=json.loads(result.stdout)
    assert plan["batches"]==[16] and len(plan["jobs"])==48 and plan["warm_seconds"]==p.WARM_SECONDS
    full=json.loads(subprocess.run([sys.executable,"-m","test.benchmarks.release.collect","--stage","core"],cwd=p.ROOT,text=True,capture_output=True,check=True).stdout)
    assert full["batches"]==list(p.BATCHES) and full["batches"][-1]==1024
    assert command_output(["/definitely/no/such/binary"]).startswith("unavailable:")


def test_timeout_terminates_worker(tmp_path):
    rc,expired=run_job([sys.executable,"-c","import time; time.sleep(30)"],tmp_path/"worker.log",.1)
    assert expired and rc < 0


def test_native_bridge_cpu_fake_abi(tmp_path):
    """Compile and exercise the real native timer without loading CUDA."""
    source=tmp_path/"fake.cpp"
    source.write_text('extern "C" int grid_rbd_inverse_dynamics(long long c,const float*q,const float*v,const float*a,float*out,int b,float g,const float*f){for(int i=0;i<b;++i)out[i]=q[i]+v[i]+a[i];return c==99?-9:0;}')
    fake=tmp_path/"fake.so"; bridge=tmp_path/"bridge.so"
    for src,out in [(source,fake),(Path(p.__file__).with_name("native_bridge.cpp"),bridge)]:
        subprocess.run(["g++","-std=c++17","-shared","-fPIC",str(src),"-ldl","-o",str(out)],check=True)
    lib=ctypes.CDLL(str(bridge)); fn=lib.grid_release_time
    fp=ctypes.POINTER(ctypes.c_float); dp=ctypes.POINTER(ctypes.c_double)
    fn.argtypes=[ctypes.c_char_p,ctypes.c_char_p,ctypes.c_longlong,fp,fp,fp,ctypes.c_int,ctypes.c_int,ctypes.c_int,ctypes.c_int,ctypes.c_double,dp,fp]
    inp=np.arange(4,dtype=np.float32); out=np.empty_like(inp); times=np.empty(3)
    args=[str(fake).encode(),b"grid_rbd_inverse_dynamics",0,*([inp.ctypes.data_as(fp)]*3),4,4,2,3,0.02,times.ctypes.data_as(dp),out.ctypes.data_as(fp)]
    assert fn(*args)==0
    np.testing.assert_array_equal(out,3*inp)
    assert np.isfinite(times).all() and (times>=0).all()
    args[2]=99; assert fn(*args)==-9
    args[1]=b"unapproved_symbol"; assert fn(*args)==-2
    args[1]=b"grid_rbd_inverse_dynamics"; args[2]=0; args[10]=-1.0; assert fn(*args)==-1


def test_release_pool_runs_every_slice_once_on_persistent_threads(tmp_path):
    """The C++ pool behind the CPU baselines, compiled into a fake TU without Pinocchio."""
    source=tmp_path/"pool.cpp"
    source.write_text('#include "release_pool.h"\n#include <atomic>\n#include <thread>\n'
        'extern "C" int pool_probe(int helpers,int active,int rounds,int*hits,int*distinct_threads){\n'
        '  ReleasePool pool((std::size_t)helpers); std::atomic<int> bad{0};\n'
        '  std::mutex m; std::vector<std::thread::id> seen;\n'
        # run() executes slot 0 on the CALLING thread (helpers take 1..active-1), so
        # the contract is checked against the caller's id captured up front. (It used
        # to compare against seen.front(), i.e. whichever slot won the mutex first in
        # round 0 — a coin flip that CI lost on 2026-10-01.)
        '  const std::thread::id caller = std::this_thread::get_id();\n'
        '  for(int r=0;r<rounds;++r) pool.run((std::size_t)active,[&](std::size_t slot){\n'
        '    hits[slot]++; std::lock_guard<std::mutex> g(m); seen.push_back(std::this_thread::get_id());\n'
        '    if(slot==0 && std::this_thread::get_id()!=caller) bad++; });\n'
        '  std::sort(seen.begin(),seen.end()); *distinct_threads=(int)(std::unique(seen.begin(),seen.end())-seen.begin());\n'
        '  return bad? -1 : (int)pool.capacity(); }\n')
    lib=tmp_path/"pool.so"
    subprocess.run(["g++","-std=c++17","-O2","-shared","-fPIC","-pthread",f"-I{Path(p.__file__).parent}",str(source),"-o",str(lib)],check=True)
    probe=ctypes.CDLL(str(lib)).pool_probe
    ip=ctypes.POINTER(ctypes.c_int)
    probe.argtypes=[ctypes.c_int,ctypes.c_int,ctypes.c_int,ip,ip]
    hits=np.zeros(8,np.int32); distinct=ctypes.c_int(0)
    assert probe(3,4,25,hits.ctypes.data_as(ip),ctypes.byref(distinct))==4
    assert hits.tolist()==[25,25,25,25,0,0,0,0] and distinct.value==4
    hits[:]=0
    assert probe(3,1,5,hits.ctypes.data_as(ip),ctypes.byref(distinct))==4
    assert hits.tolist()==[5,0,0,0,0,0,0,0] and distinct.value==1
    hits[:]=0  # active above capacity is clamped, never over-subscribed
    assert probe(1,6,3,hits.ctypes.data_as(ip),ctypes.byref(distinct))==2
    assert hits.tolist()==[3,3,0,0,0,0,0,0]


def test_overhead_decomposition_differences_and_negative_flags():
    from test.benchmarks.release.report import decompose
    cells={"grid_cuda":(12.,4.),"grid_native":(15.,None),"grid_numpy":(22.,None),"grid_jax":(400.,9.),"grid_torch":(300.,3.),
           "pinocchio":(30.,None),"pinocchio_plain":(70.,None)}
    rows=[dict(robot="iiwa14",operation="inverse_dynamics",batch=16,backend=b,host_us=h,resident_us=r) for b,(h,r) in cells.items()]
    d=decompose(rows)[0]
    assert (d["kernel_compute_us"],d["memory_traffic_us"],d["c_abi_staging_us"],d["numpy_python_us"]) == (4.,8.,3.,7.)
    assert (d["jax_dispatch_us"],d["jax_round_trip_us"]) == (5.,391.)
    assert (d["pinocchio_codegen_us"],d["pinocchio_standard_api_overhead_us"]) == (30.,40.)
    assert d["torch_dispatch_us"] is None and "torch_dispatch_us: negative" in d["flags"]
    assert decompose([r for r in rows if r["backend"] not in {"grid_cuda","pinocchio_plain"}]) == []


def test_plot_handles_single_robot_and_missing_backends(tmp_path):
    import warnings
    from test.benchmarks.release.report import plot
    with warnings.catch_warnings():
        warnings.filterwarnings("error", message="Tight layout not applied.*")
        plot(aggregate([row()]),tmp_path,"wrappers","smoke")
    assert (tmp_path/"wrappers.png").stat().st_size > 1000


def test_stacked_and_composition_figures_from_synthetic_rows(tmp_path):
    from test.benchmarks.release.report import plot_stacked_comparison, plot_grid_composition, grid_stack
    cells={"grid_cuda":(12.,4.),"grid_native":(15.,None),"grid_numpy":(22.,None),"grid_jax":(400.,9.),"grid_torch":(10.,3.),
           "pinocchio":(30.,None),"pinocchio_plain":(70.,None),"mjx":(900.,300.),"mujoco_warp":(500.,600.)}
    rows=[]
    for op in ("inverse_dynamics","inverse_dynamics_gradient","idsva_so"):
        for batch in (16,256):
            rows+=[dict(robot="iiwa14",operation=op,backend=b,batch=batch,host_us=h,resident_us=r,dtype="float32",status="validated")
                   for b,(h,r) in cells.items()]
    lookup={(r["robot"],r["operation"],r["backend"],r["batch"]):r for r in rows}
    assert grid_stack(lookup,"iiwa14","inverse_dynamics",16,"grid_jax")==(4.,8.,388.,[])
    compute,memory,wrapper,flags=grid_stack(lookup,"iiwa14","inverse_dynamics",16,"grid_torch")
    assert (compute,memory,wrapper,flags)==(4.,8.,None,["wrapper"])
    assert plot_stacked_comparison(rows,tmp_path,"smoke").exists()
    assert plot_grid_composition(rows,tmp_path,"smoke").exists()
    assert (tmp_path/"comparison_stacked.png").stat().st_size>1000 and (tmp_path/"grid_composition.png").stat().st_size>1000
    assert plot_stacked_comparison([r for r in rows if r["backend"]!="grid_cuda" and r["operation"]=="minv"],tmp_path,"smoke") is None


def test_homepage_core_palette_and_hessian_baseline_selection(tmp_path, monkeypatch):
    from collections import defaultdict
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure
    from test.benchmarks.release.report import plot_stacked_comparison
    def reject_whiskers(*args, **kwargs):
        raise AssertionError("The homepage core figure must not draw whiskers")
    monkeypatch.setattr(Axes, "errorbar", reject_whiskers)
    rows=[]
    for op in ("inverse_dynamics", "idsva_so"):
        for b in ("grid_cuda", "grid_jax", "pinocchio", "pinocchio_plain", "mujoco_cpu", "mjx", "mujoco_warp", "bard", "frax"):
            if op == "idsva_so" and b in ("mujoco_cpu", "mjx", "mujoco_warp"):
                continue
            h = 12. if b == "grid_cuda" else 40.
            rows.append(dict(robot="iiwa14", operation=op, backend=b, batch=16,
                             host_us=h, resident_us=4. if b in ("grid_cuda", "grid_jax", "mjx", "mujoco_warp") else None,
                             host_min_us=h-1, host_max_us=h+1, status="validated", dtype="float32"))
    colors = defaultdict(list)
    original = Axes.bar
    def capture(ax, *args, **kwargs):
        assert kwargs["linewidth"] == .4
        if kwargs.get("color"):
            assert kwargs["edgecolor"] == kwargs["color"]
        if kwargs.get("facecolor") == "white":
            assert kwargs["hatch"] == "...." and kwargs["edgecolor"] == "#707070"
        colors[ax.get_subplotspec().rowspan.start].append(kwargs.get("color", kwargs.get("facecolor")))
        return original(ax, *args, **kwargs)
    monkeypatch.setattr(Axes, "bar", capture)
    legends = []
    original_legend = Figure.legend
    def capture_legend(fig, *args, **kwargs):
        legends.append(kwargs)
        return original_legend(fig, *args, **kwargs)
    monkeypatch.setattr(Figure, "legend", capture_legend)
    path = plot_stacked_comparison(rows, tmp_path, "collection", ops=("inverse_dynamics", "idsva_so"),
                                   show_title=False, homepage_style=True)
    assert set(colors[0]) == {"#00693e", "#e2e2e2", "white", "#d94415", "#9d162e", "#a1d6ff", "#267aba", "#003c73"}
    assert set(colors[1]) == {"#00693e", "#e2e2e2", "white", "#9d162e"}
    assert colors[0].index("#003c73") < colors[0].index("#267aba")
    assert legends[0]["ncol"] == 4
    labels = [h.get_label() for h in legends[0]["handles"]]
    assert labels == ["GRiD CUDA Device - GPU", " ", " ",
                      "Pinocchio Codegen - CPU", "Pinocchio Standard API - CPU", " ",
                      "Mujoco - CPU", "Mujoco Warp - GPU", "Mujoco XLA (MJX) - GPU",
                      "GPU-CPU I/O Overhead", "GRiD Jax Wrapper Overhead", " "]
    svg=path.read_text()
    assert all(label not in svg for label in ("BARD", "Frax", "N/C", "DRAFT", "Whiskers"))
    assert "GPU-CPU I/O Overhead" in svg and "GRiD Jax Wrapper Overhead" in svg
    assert "fp64 required by the evaluated library path/build" in svg
    assert "I/O overhead not resolved reliably" in svg and "▼" in svg


def test_report_earlier_capture_supersedes_same_cell(tmp_path):
    """core and wrappers both plan the GRiD CUDA/JAX RNEA cells; the first capture
    listed wins, later duplicates are recorded as superseded, never as extra repeats."""
    from test.benchmarks.release.report import main as report_main
    import sys as _sys
    plan=json.loads(subprocess.run([_sys.executable,"-m","test.benchmarks.release.collect","--stage","wrappers","--smoke",
        "--robots","iiwa14","--backends","grid_cuda","--operations","inverse_dynamics"],cwd=p.ROOT,text=True,capture_output=True,check=True).stdout)
    for name in ("first","second"):
        (tmp_path/name).mkdir(); p.write_json(tmp_path/name/"plan.json",plan)
    out=tmp_path/"report"
    _sys.argv=["report",str(tmp_path/"first"),str(tmp_path/"second"),"--output",str(out)]
    report_main()
    table=json.loads((out/"table.json").read_text())
    assert len(table["raw_records"])==1 and len(table["superseded_cells"])==1
    assert table["superseded_cells"][0]["capture"].endswith("second")
    assert all(r["capture"].startswith(str(tmp_path/"first")) for r in table["raw_records"])


def test_contract_ignores_momentary_clock_fields():
    from test.benchmarks.release.report import stable_provenance
    a = {"gpu": "NVIDIA GeForce RTX 5090, 615.71.09, 32607 MiB, 292 MHz, 575.00 W", "cpu": "Model name: X\nCPU(s) scaling MHz: 67%\nCPU max MHz: 6500.0000"}
    b = {"gpu": "NVIDIA GeForce RTX 5090, 615.71.09, 32607 MiB, 2407 MHz, 575.00 W", "cpu": "Model name: X\nCPU(s) scaling MHz: 43%\nCPU max MHz: 6500.0000"}
    c = {"gpu": "NVIDIA GeForce RTX 5090, 615.71.09, 32607 MiB, 292 MHz, 450.00 W", "cpu": a["cpu"]}
    assert stable_provenance(a) == stable_provenance(b)
    assert stable_provenance(a) != stable_provenance(c)
    assert "max MHz" in stable_provenance(a)[1] and "scaling" not in stable_provenance(a)[1]


def test_speedup_best_and_throughput_figures(tmp_path):
    from test.benchmarks.release.report import plot_speedup, plot_best_competitor, plot_throughput
    cells={"grid_cuda":(12.,4.),"grid_jax":(400.,9.),"pinocchio":(30.,None),"pinocchio_plain":(70.,None),"mjx":(900.,300.),"mujoco_warp":(500.,600.)}
    rows=[dict(robot="iiwa14",operation=op,backend=b,batch=batch,host_us=h,resident_us=r,dtype="float32",status="validated")
          for op in ("inverse_dynamics","minv") for batch in (16,256) for b,(h,r) in cells.items()]
    assert plot_speedup(rows,tmp_path,"smoke","grid_jax","host_us","host_us","t","speedup_full").exists()
    assert plot_speedup(rows,tmp_path,"smoke","grid_jax","resident_us","resident_us","t","speedup_resident").exists()
    assert plot_best_competitor(rows,tmp_path,"smoke").exists() and plot_throughput(rows,tmp_path,"smoke").exists()
    assert plot_speedup([r for r in rows if r["backend"].startswith("grid_")],tmp_path,"smoke","grid_jax","host_us","host_us","t","none") is None


def test_report_uncollected_placeholder_never_shadows_a_later_collected_cell(tmp_path):
    """A capture whose chain died leaves planned cells with no job; listing it first
    must not hide the same cell collected by a later capture (2026-09-26: the g1
    grid_cuda remainder was reported not_collected behind the dead chain's plan)."""
    from test.benchmarks.release.report import main as report_main
    import sys as _sys
    job=list(p.jobs('table',['iiwa14'],['grid_cuda'],['crba']))[0]
    plan=dict(jobs=[job],batches=[16],repeats=1,purpose='smoke',iterations=2,warmups=2,accuracy_policy='strict')
    (tmp_path/'dead').mkdir(); p.write_json(tmp_path/'dead'/'plan.json',plan)
    later=tmp_path/'later'; later.mkdir(); p.write_json(later/'plan.json',plan)
    good=p.agreement(np.array([1000.,0.]),np.array([1000.,0.]))
    p.write_json(later/'capture.json',dict(adapter=dict(dtype='float32'),cells=[dict(batch=16,status='validated',
        comparison_eligible=True,oracle_agreement=good,post_timing_agreement=good,host_to_host=dict(mean_us=10.))]))
    p.write_json(later/'results.json',dict(jobs=[dict(job,repeat=0,capture='capture.json',sha256=p.digest(later/'capture.json'))]))
    out=tmp_path/'report'
    _sys.argv=['report',str(tmp_path/'dead'),str(later),'--output',str(out)]
    report_main()
    table=json.loads((out/'table.json').read_text())
    assert [r['status'] for r in table['raw_records']]==['validated']
    assert table['raw_records'][0]['capture']==str(later/'capture.json')
    assert len(table['superseded_cells'])==1 and table['superseded_cells'][0]['capture'].endswith('dead/plan.json')
    assert 'never collected' in table['superseded_cells'][0]['reason']
    assert table['cells'][0]['status']=='validated' and table['cells'][0]['host_us']==10.


def test_cpu_power_is_recorded_and_part_of_the_contract(tmp_path):
    """Timings taken under different CPU power settings (governor / energy preference /
    affinity) must not be compared silently: the report flags the cross-backend cells."""
    from test.benchmarks.release.collect import cpu_power
    from test.benchmarks.release.report import main as report_main
    import sys as _sys
    keys = set(cpu_power())
    assert {"governor", "energy_performance_preference", "affinity", "scaling_max_khz"} <= keys
    jobs = list(p.jobs('table', ['iiwa14'], ['grid_cuda', 'pinocchio'], ['crba']))
    good = p.agreement(np.array([1000., 0.]), np.array([1000., 0.]))
    for name, job, governor in (("a", jobs[0], "powersave"), ("b", jobs[1], "performance")):
        d = tmp_path / name; d.mkdir()
        p.write_json(d / 'plan.json', dict(jobs=[job], batches=[16], repeats=1, purpose='smoke', iterations=2, warmups=2,
            accuracy_policy='strict', provenance=dict(gpu="G", cpu="C", cpu_power=dict(governor=governor, affinity=[0]))))
        p.write_json(d / 'capture.json', dict(adapter=dict(dtype='float32'), cells=[dict(batch=16, status='validated',
            comparison_eligible=True, oracle_agreement=good, post_timing_agreement=good, host_to_host=dict(mean_us=10.))]))
        p.write_json(d / 'results.json', dict(jobs=[dict(job, repeat=0, capture='capture.json', sha256=p.digest(d / 'capture.json'))]))
    out = tmp_path / 'report'
    _sys.argv = ['report', str(tmp_path / 'a'), str(tmp_path / 'b'), '--output', str(out)]
    report_main()
    table = json.loads((out / 'table.json').read_text())
    assert {c['status'] for c in table['cells']} == {'contract_mismatch'}
    assert all('cpu_power' in r['contract'] for r in table['raw_records'])


def test_forward_dynamics_rows_are_labelled_fd_not_aba():
    # GRiD's timed forward_dynamics is the mass-matrix-inverse path, and MJX / Warp solve with
    # their own factorisations; "ABA" named an algorithm none of the timed rows necessarily run.
    from test.benchmarks.release.report import OP_LABELS, SHORT_OP
    assert (OP_LABELS["forward_dynamics"], OP_LABELS["forward_dynamics_gradient"], OP_LABELS["fdsva_so"]) == ("FD", "grad FD", "Hessian FD")
    assert (SHORT_OP["forward_dynamics"], SHORT_OP["forward_dynamics_gradient"], SHORT_OP["fdsva_so"]) == ("FD", "∇FD", "∇²FD")
    assert not any("ABA" in v for v in (*OP_LABELS.values(), *SHORT_OP.values()))


@pytest.mark.parametrize('batch,expected', [
    (16, [1, 2, 4, 8, 16]),
    (32, [1, 2, 4, 8, 16, 24]),
    (64, [1, 2, 4, 8, 16, 24]),
    (128, [1, 2, 4, 8, 16, 24]),
    (256, [1, 2, 4, 8, 16, 24]),
    (1024, [1, 2, 4, 8, 16, 24]),
])
def test_pinocchio_full_machine_thread_candidates(batch, expected):
    from test.benchmarks.release.pin_adapter import PinAdapter
    adapter = PinAdapter.__new__(PinAdapter)  # policy test: no native build
    adapter.ceiling = 24
    assert adapter.candidates(batch) == expected
    # The expanded sweep still evaluates every formerly tested count.
    old = {n for n in (1, max(1, batch // 16), 8) if n <= min(8, batch)}
    assert old <= set(expected)


def test_pinocchio_candidates_respect_small_ceiling():
    from test.benchmarks.release.pin_adapter import PinAdapter
    adapter = PinAdapter.__new__(PinAdapter)
    adapter.ceiling = 1
    assert adapter.candidates(1024) == [1]
    adapter.ceiling = 6
    assert adapter.candidates(128) == [1, 2, 4, 6]
