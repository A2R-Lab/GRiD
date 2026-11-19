"""Conservative Linux headroom for the contributor lane, including cgroup limits.

Use the smaller of host MemAvailable and remaining controller headroom at each
visible ancestor. This deliberately counts cgroup cache as used: no reclaim
prediction, swap allowance or promise against later external pressure. The lab
driver keeps its existing policy. Unreadable/malformed accounting refuses work.
"""
from pathlib import Path
import re


def _unescape(value):
    return re.sub(r'\\([0-7]{3})', lambda m: chr(int(m[1], 8)), value)


def _headroom(current, mount, unified):
    """None means demonstrably inactive, not missing active accounting."""
    names = (('memory.current', 'memory.max', 'memory.high') if unified else
             ('memory.usage_in_bytes', 'memory.limit_in_bytes'))
    present = [(current / name).exists() for name in names]
    if not any(present):
        if not unified:
            raise ValueError('missing v1 memory accounting')
        # A unified hierarchy can exist with memory entirely on v1, or with
        # memory disabled below an accounted ancestor. A missing file alone
        # cannot distinguish that from damaged active accounting.
        if current != mount:
            enabled = (current.parent / 'cgroup.subtree_control').read_text().split()
            if 'memory' in enabled:
                raise ValueError('missing delegated memory accounting')
        else:
            # The real v2 root has no memory accounting; require the control
            # files so an inaccessible/malformed mount cannot look unlimited.
            (current / 'cgroup.controllers').read_text()
            (current / 'cgroup.subtree_control').read_text()
        return None
    if not all(present):
        raise ValueError('partial memory accounting')
    used = int((current / names[0]).read_text())
    if used < 0:
        raise ValueError('negative usage')
    remaining = []
    for name in names[1:]:
        raw = (current / name).read_text().strip()
        if unified and raw == 'max':
            continue
        limit = int(raw)
        if limit < 0:
            raise ValueError('negative limit')
        # high is a conservative admission ceiling to avoid reclaim and
        # throttling, not the kernel's hard memory limit (max).
        remaining.append(max(0, limit - used) // 1024)
    return min(remaining) if remaining else None


def effective_available_kb(proc=Path('/proc')):
    try:
        host = next(int(line.split()[1]) for line in
                    (proc / 'meminfo').read_text().splitlines()
                    if line.startswith('MemAvailable:'))
        memberships = []
        for line in (proc / 'self/cgroup').read_text().splitlines():
            _, controllers, member = line.split(':', 2)
            if not controllers or 'memory' in controllers.split(','):
                memberships.append((not controllers, Path(member)))
        # A controller belongs to one hierarchy. On hybrid hosts the unified
        # hierarchy is irrelevant when memory has an explicit v1 membership.
        if any(not unified for unified, _ in memberships):
            memberships = [(u, m) for u, m in memberships if not u]
        for unified, member in memberships:
            matched = False
            for line in (proc / 'self/mountinfo').read_text().splitlines():
                left, right = line.split(' - ', 1)
                fields, fs = left.split(), right.split()
                if fs[0] != ('cgroup2' if unified else 'cgroup'):
                    continue
                if not unified and 'memory' not in fs[2].split(','):
                    continue
                root, mount = Path(_unescape(fields[3])), Path(_unescape(fields[4]))
                if not member.is_absolute() or '..' in member.parts:
                    return 0
                if not member.is_relative_to(root):
                    continue
                current = mount / member.relative_to(root)
                matched = True
                while True:
                    remaining = _headroom(current, mount, unified)
                    if remaining is not None:
                        host = min(host, remaining)
                    if current == mount:
                        break
                    current = current.parent
            if not matched:
                return 0
        return max(0, host)
    except (OSError, ValueError, IndexError, StopIteration):
        return 0
