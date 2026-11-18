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
                if not member.is_relative_to(root) or '..' in member.parts:
                    continue
                current = mount / member.relative_to(root)
                matched = True
                while True:
                    usage_name = 'memory.current' if unified else 'memory.usage_in_bytes'
                    limits = ('memory.max', 'memory.high') if unified else ('memory.limit_in_bytes',)
                    # The controller root may have no memory files at all.
                    if (current / usage_name).exists():
                        used = int((current / usage_name).read_text())
                        if used < 0:
                            return 0
                        for name in limits:
                            path = current / name
                            if not path.exists():
                                if current != mount:
                                    return 0
                                continue
                            raw = path.read_text().strip()
                            if raw != 'max':
                                host = min(host, max(0, int(raw) - used) // 1024)
                    elif current != mount:
                        return 0
                    if current == mount:
                        break
                    current = current.parent
            if not matched:
                return 0
        return max(0, host)
    except (OSError, ValueError, IndexError, StopIteration):
        return 0
