"""Admission must see the constrained job's capacity, not only host RAM."""
import pytest
from test.host_memory import effective_available_kb

GIB = 1024**3


@pytest.fixture
def memory(tmp_path):
    proc = tmp_path / 'proc'
    (proc / 'self').mkdir(parents=True)
    (proc / 'meminfo').write_text(f'MemAvailable: {64 * GIB // 1024} kB\n')
    mount = tmp_path / 'cgroup'
    leaf = mount / 'parent/worker'
    leaf.mkdir(parents=True)
    (proc / 'self/cgroup').write_text('0::/parent/worker\n')
    (proc / 'self/mountinfo').write_text(f'1 0 0:1 / {mount} rw - cgroup2 cgroup rw\n')
    for directory in (mount, leaf.parent, leaf):
        (directory / 'memory.current').write_text(str(GIB))
        (directory / 'memory.max').write_text('max')
        (directory / 'memory.high').write_text('max')
    return proc, mount, leaf


def test_host_unlimited_and_leaf_limit(memory):
    proc, mount, leaf = memory
    assert effective_available_kb(proc) == 64 * GIB // 1024
    (leaf / 'memory.max').write_text(str(8 * GIB))
    assert effective_available_kb(proc) == 7 * GIB // 1024
    (proc / 'meminfo').write_text(f'MemAvailable: {3 * GIB // 1024} kB\n')
    assert effective_available_kb(proc) == 3 * GIB // 1024


def test_parent_limit_and_pressure_are_visible(memory):
    proc, mount, leaf = memory
    (leaf / 'memory.max').write_text(str(16 * GIB))
    (leaf.parent / 'memory.max').write_text(str(8 * GIB))
    (leaf.parent / 'memory.current').write_text(str(6 * GIB))
    assert effective_available_kb(proc) == 2 * GIB // 1024
    (leaf / 'memory.high').write_text(str(GIB // 2))
    assert effective_available_kb(proc) == 0


@pytest.mark.parametrize('problem', ['unreadable', 'malformed', 'unmapped', 'traversal'])
def test_bad_accounting_fails_closed(memory, problem):
    proc, mount, leaf = memory
    if problem == 'unreadable':
        (leaf / 'memory.current').unlink()
    elif problem == 'malformed':
        (leaf / 'memory.max').write_text('invalid')
    elif problem == 'unmapped':
        (proc / 'self/mountinfo').write_text('')
    else:
        (proc / 'self/cgroup').write_text('0::/../escape\n')
    assert effective_available_kb(proc) == 0


def test_namespaced_mount_root(memory):
    proc, mount, leaf = memory
    (proc / 'self/cgroup').write_text('0::/outer/parent/worker\n')
    (proc / 'self/mountinfo').write_text(f'1 0 0:1 /outer {mount} rw - cgroup2 cgroup rw\n')
    (mount / 'memory.max').write_text(str(4 * GIB))
    assert effective_available_kb(proc) == 3 * GIB // 1024


def test_v1_memory_controller(memory):
    proc, mount, leaf = memory
    (proc / 'self/cgroup').write_text('3:cpu:/unrelated\n4:memory:/parent/worker\n')
    (proc / 'self/mountinfo').write_text(f'1 0 0:1 / {mount} rw - cgroup cgroup rw,memory\n')
    for directory in (mount, leaf.parent, leaf):
        (directory / 'memory.usage_in_bytes').write_text(str(GIB))
        (directory / 'memory.limit_in_bytes').write_text(str(8 * GIB))
    assert effective_available_kb(proc) == 7 * GIB // 1024


def test_hybrid_uses_memory_hierarchy(memory):
    proc, mount, leaf = memory
    test_v1_memory_controller(memory)
    unified = mount.parent / 'unified'
    (unified / 'worker').mkdir(parents=True)
    with (proc / 'self/cgroup').open('a') as f:
        f.write('0::/worker\n')
    with (proc / 'self/mountinfo').open('a') as f:
        f.write(f'2 0 0:2 / {unified} rw - cgroup2 cgroup rw\n')
    assert effective_available_kb(proc) == 7 * GIB // 1024
    (leaf / 'memory.usage_in_bytes').write_text('broken')
    assert effective_available_kb(proc) == 0  # cannot fall back to unified


def test_undelegated_leaf_keeps_ancestor_limit(memory):
    proc, mount, leaf = memory
    for name in ('memory.current', 'memory.max', 'memory.high'):
        (leaf / name).unlink()
    (leaf.parent / 'cgroup.subtree_control').write_text('cpu io')
    (leaf.parent / 'memory.max').write_text(str(8 * GIB))
    assert effective_available_kb(proc) == 7 * GIB // 1024
    (leaf.parent / 'cgroup.subtree_control').write_text('cpu memory')
    assert effective_available_kb(proc) == 0


def test_inactive_v2_memory(memory):
    proc, mount, leaf = memory
    for directory in (mount, leaf.parent, leaf):
        for name in ('memory.current', 'memory.max', 'memory.high'):
            (directory / name).unlink()
        (directory / 'cgroup.controllers').write_text('cpu io')
        (directory / 'cgroup.subtree_control').write_text('cpu')
    assert effective_available_kb(proc) == 64 * GIB // 1024
    (mount / 'cgroup.controllers').unlink()
    assert effective_available_kb(proc) == 0


@pytest.mark.parametrize('name', ['memory.current', 'memory.max', 'memory.high'])
def test_negative_accounting_refused(memory, name):
    proc, mount, leaf = memory
    (leaf / name).write_text('-1')
    assert effective_available_kb(proc) == 0
