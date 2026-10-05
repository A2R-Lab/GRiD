"""CPU-only admission and builder-contract tests for long-build prewarming."""
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).parent))
import prewarm_cuda_flagship as pw
import run_split_suite as rss

I = 'test/cuda_equivalents/test_cuda_integrator_equivalence.py::'
S = 'test/cuda_equivalents/test_cuda_second_order_fallback.py::'
INTEGRATOR = I + 'test_cuda_integrator_matches_python_reference[iiwa14-integrator-fixed-TIER_SHARED]'
FEXT = I + 'test_cuda_integrator_fext_matches_python_reference'
SO = S + 'test_floating_second_order_diagnostic_matches_python_reference[h1_2-floating]'


def test_smoke_plan_deduplicates_exact_supported_ids():
    jobs, nodes = pw.smoke_jobs([INTEGRATOR, INTEGRATOR, FEXT, SO, SO + '-wrong', 'other::' + SO])
    assert len(jobs) == 2 and len(nodes) == 3
    assert nodes[INTEGRATOR] == nodes[FEXT]
    assert jobs[nodes[SO]] == dict(family='second_order', robot='h1_2', base='floating', tier=None)


@pytest.mark.parametrize('nid,fdsva', [(INTEGRATOR, False), (SO, False), (SO, True)])
def test_smoke_worker_uses_actual_builder_options(tmp_path, monkeypatch, nid, fdsva):
    from test.cuda_equivalents import test_cuda_integrator_equivalence as integ
    from test.cuda_equivalents import test_cuda_second_order_fallback as so
    jobs, nodes = pw.smoke_jobs([nid])
    job = jobs[nodes[nid]]
    mod = integ if job['family'] == 'integrator' else so
    monkeypatch.setattr(mod, '_robot_spec', lambda robot, base: (robot, base))
    monkeypatch.setattr(mod, 'resolve_robot_spec', lambda spec: spec)
    monkeypatch.setattr(mod, 'build_project_adapter', lambda *args, **kwargs: 'model')
    monkeypatch.setenv('GRID_CUDA_FLOATING_SECOND_ORDER_ENABLE_FDSVA', str(int(fdsva)))
    calls = []
    def build(*args, **kwargs):
        calls.append((args, kwargs))
        assert Path(args[1]).is_dir()
        return {'key': 'test', 'status': 'miss'}
    monkeypatch.setattr(mod, '_build_case' if mod is integ else '_build_second_order_case', build)
    pw.run_smoke_job(job, inspect_only=True)
    args, kwargs = calls[0]
    assert args[0] == 'model' and not Path(args[1]).exists()
    assert kwargs['inspect_only'] is True
    if mod is integ:
        assert kwargs['tier'] == 'TIER_SHARED'
    else:
        assert args[3] == mod._second_order_target_shared_bytes()
        assert kwargs['enable_floating_second_order'] and kwargs['enable_idsva_so_body_frame']
        assert kwargs['enable_fdsva'] is fdsva
        assert kwargs['algorithm_list'] == ('idsva_so_body_frame,fdsva_so' if fdsva else 'idsva_so_body_frame')


def test_smoke_jobs_use_existing_pool_and_every_consumer_waits(tmp_path, monkeypatch):
    monkeypatch.setenv('GRID_SPLIT_PREWARM', '1')
    def plan(command, **kwargs):
        ids = Path(command[command.index('--node-ids') + 1]).read_text().splitlines()
        jobs, nodes = pw.smoke_jobs(ids)
        Path(command[-1]).write_text(json.dumps(dict(jobs=jobs, node_to_job=nodes)))
        return SimpleNamespace(returncode=0)
    monkeypatch.setattr(rss.subprocess, 'run', plan)
    shards = [rss.ShardSpec('one', 'cuda', [INTEGRATOR], []),
              rss.ShardSpec('two', 'cuda', [FEXT, SO], [])]
    pool, jobs, prereqs = rss.build_compile_pool(tmp_path, warm=False, cuda_shards=shards,
                                                modules=[], max_jobs=1)
    assert pool is not None and len(jobs) == 2
    assert len(prereqs['one']) == 1 and set(prereqs['one']) < set(prereqs['two'])
    assert {j.name for j in jobs} == set(prereqs['two'])
    assert all(j.env == rss.cuda_worker_env() for j in jobs)
    assert not pool.running_pids()  # planning did not launch compilers


def test_scheduler_shutdown_reaps_workers_and_blocks_new_admissions(tmp_path):
    import time
    import compile_sched
    sched = compile_sched.RamScheduler(compile_sched.Ledger(tmp_path / 'rss.json'),
                                       max_jobs=1, floor_kb=0, default_peak_kb=1)
    job = compile_sched.Job(name='owned', argv=[sys.executable, '-c', 'import time; time.sleep(60)'],
                            ledger_key='owned', log_path=str(tmp_path / 'owned.log'))
    thread = sched.run_async([job])
    deadline = time.monotonic() + 10
    while not sched.running_pids() and time.monotonic() < deadline:
        time.sleep(.02)
    assert sched.running_pids()
    sched.kill_all()
    assert not thread.is_alive() and not sched.running_pids()
    assert not sched._launch(job, 1, {}, set())
    assert sched.results['owned'].rc != 0
