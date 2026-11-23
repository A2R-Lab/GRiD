"""Content-qualified resource predictions, separate from test/receipt evidence.

No result is reused here. Cold-build floors keep cache-hit samples from promising
unsafe concurrency; a source/toolchain/options change gives a new ledger key.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

GIB_KB = 1024 * 1024


def content_digest(root: Path, paths: list[str]) -> str:
    """Hash actual bytes (including dirty/new sources), not a checkout path/SHA."""
    h = hashlib.sha256()
    for rel in sorted(set(paths)):
        path = root / rel
        files = sorted(p for p in path.rglob('*') if p.is_file()
                       and '__pycache__' not in p.parts and p.suffix != '.pyc') if path.is_dir() else [path]
        for p in files:
            h.update(str(p.relative_to(root)).encode() + b'\0')
            h.update(p.read_bytes() if p.is_file() else b'<missing>')
    return h.hexdigest()


def identity(root: Path, paths: list[str], recipe: object, environment: dict) -> str:
    material = dict(source=content_digest(root, paths), recipe=recipe, environment=environment)
    return 'resource-v1:' + hashlib.sha256(json.dumps(material, sort_keys=True).encode()).hexdigest()


def build_environment(nvcc_version: str, arch: str) -> dict:
    # Explicit values distinguish historical four/eight-unit builds from two-unit
    # receipts and include toolchain flags that can change compiler memory.
    keys = ('GRID_RBD_BUILD_JOBS', 'GRID_ENABLE_MUJOCO_KERNELS',
            'GRID_CUDA_FLOATING_SECOND_ORDER_ENABLE_FDSVA',
            'CXX', 'CC', 'CXXFLAGS', 'CFLAGS', 'NVCC_PREPEND_FLAGS', 'NVCC_APPEND_FLAGS')
    defaults = {'GRID_RBD_BUILD_JOBS': '2', 'GRID_ENABLE_MUJOCO_KERNELS': '0',
                'GRID_CUDA_FLOATING_SECOND_ORDER_ENABLE_FDSVA': '0'}
    return dict(nvcc=nvcc_version, arch=arch,
                **{k: os.environ.get(k, defaults.get(k, '')) for k in keys})


def compile_floor_kb(kind: str, robot: str, units: int, recipe: dict | None = None) -> int:
    # November10 sampled process trees, units=2: pool max4.73GiB, wrappers4.19.
    # Preserve the historical large-wrapper allowance for unpiloted humanoids;
    # do not extrapolate two-unit measurements to wider compiler parallelism.
    measured_robots = {'iiwa14', 'go2', 'g1', 'h1_2', 'fr3', 'rizon4', 'gen3', 'fetch', 'baxter'}
    if units > 2 or robot not in measured_robots or robot in {'g1', 'h1_2'} and kind == 'wrapper':
        return 16 * GIB_KB
    if kind == 'wrapper':
        recipe = recipe or {}
        if recipe.get('floating_base') and recipe.get('enable_mujoco_kernels', True) and not recipe.get('algorithm_list'):
            return 12 * GIB_KB  # Go2 default/MJX shard measured9.66GiB.
        return 8 * GIB_KB
    if kind == 'native':
        return 6 * GIB_KB
    return 16 * GIB_KB


def shard_floor_kb(targets: list[str]) -> int:
    # Whole shard trees include inline compiles. Default-registration measured
    # 9.66GiB; all other measured shards <=5.13GiB. Unpiloted large builds keep16.
    text = ' '.join(targets)
    if 'h1_2' in text or 'h2_plus' in text:
        return 16 * GIB_KB
    if 'test_armature_default_registration' in text:
        return 12 * GIB_KB
    return 8 * GIB_KB


# --- Measured classes (adaptive profile) ------------------------------------
# The identity above changes with every source edit, so its history is lost at
# each commit. A class names what is built (kind, robot, recipe) and how wide
# (compiler units), not the bytes, so a measurement survives a code change.
# Only real compiles are recorded: a cache hit measures a relink.
CLASS_SCHEMA = 1
BUILD_MARKERS = ('cache miss', 'compiling runner', 'compiling the dynamics', 'BUILT ')


def log_shows_build(text: str) -> bool:
    return any(marker in text for marker in BUILD_MARKERS)


def recipe_name(recipe: object) -> str:
    return hashlib.sha256(json.dumps(recipe, sort_keys=True).encode()).hexdigest()[:16]


def adaptive_reserve_kb(total_kb: int) -> int:
    """Memory left to the rest of the machine: 15% of RAM, at least 4 GiB."""
    return max(4 * GIB_KB, int(0.15 * total_kb))


class ClassHistory:
    """Max-over-history cold peaks: {kind|robot|name: {units: {tree_kb, proc_kb, secs}}}.

    tree_kb is the peak of the whole process tree, proc_kb the largest single
    process in it. A build at u units runs one driver and at most u compilers,
    so (u + 1) * proc_kb bounds its tree and gives a figure for a width that
    was never measured."""

    shows_build = staticmethod(log_shows_build)

    def __init__(self, path: Path):
        self.path = Path(path)
        self.rows: dict[str, dict[str, dict]] = {}
        try:
            raw = json.loads(self.path.read_text())
        except (OSError, ValueError):
            raw = {}
        if isinstance(raw, dict) and raw.get('schema') == CLASS_SCHEMA:
            self.rows = raw.get('classes') or {}

    @staticmethod
    def key(kind: str, robot: str, name: str) -> str:
        return f'{kind}|{robot}|{name}'

    def record(self, kind, robot, name, units, tree_kb, proc_kb=0, secs=0.0) -> None:
        if tree_kb <= 0:
            return
        row = self.rows.setdefault(self.key(kind, robot, name), {})
        cur = row.get(str(units), dict(tree_kb=0, proc_kb=0, secs=0.0))
        new = dict(tree_kb=max(cur['tree_kb'], int(tree_kb)),
                   proc_kb=max(cur['proc_kb'], int(proc_kb)),
                   secs=max(cur['secs'], float(secs)))
        if new != cur:
            row[str(units)] = new
            self.path.parent.mkdir(parents=True, exist_ok=True)
            tmp = self.path.with_suffix(f'.{os.getpid()}.tmp')
            tmp.write_text(json.dumps(dict(schema=CLASS_SCHEMA, classes=self.rows),
                                      indent=0, sort_keys=True))
            tmp.replace(self.path)

    @staticmethod
    def _at(row: dict, units: int):
        """Peak of one class at this width, or None when nothing bounds it."""
        if str(units) in row:
            return row[str(units)]['tree_kb']
        wider = [v['tree_kb'] for u, v in row.items() if int(u) > units]
        if wider:
            return min(wider)   # fewer compilers at once cannot need more
        bounds = [max(v['tree_kb'], (units + 1) * v['proc_kb'])
                  for v in row.values() if v['proc_kb'] > 0]
        return min(bounds) if bounds else None

    def predict_kb(self, kind, robot, name, units, floor_kb: int) -> int:
        """Charge before margin. The class itself when it was measured, else the
        largest measured class of the same kind and robot, else the class floor.
        A bound from another width never exceeds the floor it replaces."""
        row = self.rows.get(self.key(kind, robot, name))
        if row:
            peak = self._at(row, units)
            if peak is not None:
                # Small jobs vary by more than the margin: never charge under 1 GiB.
                return max(GIB_KB, peak if str(units) in row else min(peak, floor_kb))
        prefix = f'{kind}|{robot}|'
        peers = [self._at(r, units) for k, r in self.rows.items() if k.startswith(prefix)]
        if peers and all(p is not None for p in peers):
            return max(GIB_KB, min(max(peers), floor_kb))
        return floor_kb

    def largest_kb(self, units: int, floor_kb: int) -> int:
        """Largest single charge at this width over every recorded class."""
        peaks = [self._at(row, units) for row in self.rows.values()]
        if not peaks or any(p is None for p in peaks):
            return floor_kb
        return min(max(peaks), floor_kb)


def plan_width(*, budget_kb: int, cores: int, history: 'ClassHistory', margin: float,
               units: int | None = None, slots: int | None = None) -> tuple[int, int, int]:
    """(slots, units, largest allowance) for a budget. Units halve until the
    largest recorded build fits. Slots default to one per core: admission against
    live memory and free cores decides how many are in use at any moment."""
    floor = 16 * GIB_KB
    units = units or max(1, min(4, cores))
    while units > 1 and int(history.largest_kb(units, floor) * margin) > budget_kb:
        units //= 2
    largest = int(history.largest_kb(units, floor) * margin)
    return slots or max(1, cores), units, largest


def import_run(history: 'ClassHistory', compile_dir: Path, ledger: dict) -> int:
    """Seed class history from a finished serial or bounded-overlap run: its
    resource-identities.json, the identity ledger and the per-job logs."""
    import re
    identities = json.loads((compile_dir / 'resource-identities.json').read_text())
    count = 0
    for name, ident in identities.items():
        log = compile_dir / f'{name}.log'
        entry = ledger.get(ident['key'])
        if not entry or not log.exists() or not log_shows_build(log.read_text(errors='replace')):
            continue
        time_file = compile_dir / f'{name}.time'
        m = re.search(r'Maximum resident set size \(kbytes\):\s*(\d+)',
                      time_file.read_text() if time_file.exists() else '')
        recipe = ident['recipe']
        history.record(ident['kind'], recipe.get('robot', ''), recipe_name(recipe),
                       # A native job is one compiler at a time at any width.
                       1 if ident['kind'] == 'native' else
                       int(ident['environment'].get('GRID_RBD_BUILD_JOBS', 2)),
                       entry['peak_kb'], int(m.group(1)) if m else 0, entry.get('secs', 0.0))
        count += 1
    return count


if __name__ == '__main__':
    import argparse
    ap = argparse.ArgumentParser(description='Seed or print the measured class history.')
    ap.add_argument('--history', type=Path,
                    default=Path(__file__).parent / '.split_suite' / 'compile_class.json')
    ap.add_argument('--import-run', type=Path, metavar='COMPILE_DIR', action='append', default=[])
    ap.add_argument('--ledger', type=Path,
                    default=Path(__file__).parent / '.split_suite' / 'compile_rss.json')
    args = ap.parse_args()
    hist = ClassHistory(args.history)
    for run in args.import_run:
        jobs = json.loads(args.ledger.read_text()).get('jobs', {})
        print(f'{run}: {import_run(hist, run, jobs)} cold builds imported')
    for u in (1, 2, 4, 8):
        print(f'units {u}: largest charge {hist.largest_kb(u, 16 * GIB_KB) / GIB_KB:.1f} GiB '
              f'({len(hist.rows)} classes)')
