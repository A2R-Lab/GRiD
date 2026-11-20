"""Structural evidence checks, complementary to gpu-proof signature verification.

The receipt scope itself is enforced by pytest-gpu-proof (policy
required_test_manifest = test/gpu-proof-tests.txt, an EXACT node-ID set); this
module serializes that manifest from a receipt or authoritative driver collection. Header-key checks
distinguish carried evidence from freshly recorded sidecars and never restore
an old header-key file over a fresh recording.

Receipts by architecture (test/gpu-proof-receipts.yaml): the primary receipt,
gpu-proof.json, gates CI and defines the manifest; secondary receipts
(receipts/gpu-proof-sm<arch>.json) document other GPUs. This module reads the
table, picks the row for a detected compute capability, verifies every receipt
on disk (``--verify-all``) and prints their rows for the compatibility page
(``--table``). The manifest is defined by the primary GPU's scope only.
"""
from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
import sys
import tempfile
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
RECEIPTS_TABLE = REPO_ROOT / 'test' / 'gpu-proof-receipts.yaml'
EVERYDAY_POLICY = 'test/gpu-proof-policy.yaml'
DEFAULT_EXPECTED_SKIPS = 'test/gpu-proof-expected-skips.txt'
DEFAULT_HEADER_KEYS = 'test/gpu-proof-header-keys.json'
SECONDARY_DIR = 'receipts'


def node_ids(receipt):
    ids = [t['node_id'] for t in receipt['tests']]
    if len(ids) != len(set(ids)):
        raise ValueError('Duplicate test node IDs in receipt')
    return set(ids)


def verify_manifest_scope(receipt, path):
    """Reject missing, extra or duplicate outcomes before evidence promotion.

    A newly planned manifest deliberately makes old evidence stale. Compare
    it to the completed recording here, not to last release's receipt during
    CPU-only pre-freeze tests. The external policy verifier remains mandatory.
    """
    expected = [line for line in path.read_text().splitlines()
                if line and not line.startswith('#')]
    if not expected or expected != sorted(set(expected)):
        raise ValueError('Manifest must be nonempty, unique and canonically sorted')
    declared = receipt['session']['node_ids']
    if len(declared) != len(set(declared)):
        raise ValueError('Duplicate collected node IDs in receipt')
    wanted = set(expected)
    if node_ids(receipt) != wanted or set(declared) != wanted:
        raise ValueError('Completed receipt scope differs from the frozen manifest')


def write_manifest(receipt, path):
    """Rewrite the reviewed scope manifest from a receipt's collection, keeping
    the file's leading comment block. Explicit maintainer action: the diff
    (added AND removed node IDs) is reviewed in the PR, and the file is a
    receipt fingerprint path, so it must be committed BEFORE recording."""
    ids = node_ids(receipt)
    return write_manifest_ids(ids, path)


def write_manifest_ids(node_ids, path):
    """Canonical manifest serialization shared by collection and receipts."""
    ids = list(node_ids)
    if not ids or len(ids) != len(set(ids)):
        raise ValueError('Manifest requires nonempty, unique test node IDs')
    header = [line for line in path.read_text().splitlines()
              if line.startswith('#')] if path.exists() else []
    path.write_text('\n'.join(header + sorted(ids)) + '\n')
    return set(ids)


def read_sidecar(path):
    """Deduplicate complete records; malformed evidence is an error."""
    unique = {}
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if not isinstance(row, dict) or not row.get('content_sha256'):
            raise ValueError(f'Invalid header-key record in {path}')
        unique[json.dumps(row, sort_keys=True)] = row
    return list(unique.values())


def verify_header_keys(before, after, receipt, sidecar_dir):
    old, new = before.get('shards', {}), after.get('shards', {})
    canonical = lambda rows: {json.dumps(r, sort_keys=True) for r in rows}
    attested = {s['name'] for s in receipt['shards']}
    if set(new) - attested:
        raise ValueError('Header keys include unattested shards')
    for shard in receipt['shards']:
        name = shard['name']
        if shard.get('carried'):
            if canonical(old.get(name, [])) != canonical(new.get(name, [])):
                raise ValueError(f'Carried header evidence changed: {name}')
        else:
            path = Path(sidecar_dir) / f'{name}.header_keys.jsonl'
            rows = read_sidecar(path) if path.exists() else []
            if canonical(rows) != canonical(new.get(name, [])):
                raise ValueError(f'Fresh header evidence differs from sidecar: {name}')
            if name in old and old[name] and not rows:
                raise ValueError(f'Fresh shard lost its header-key recording: {name}')


# ─── receipts by architecture ────────────────────────────────────────────────

@dataclass(frozen=True)
class ReceiptRow:
    """One row of test/gpu-proof-receipts.yaml. Paths are repository-relative."""
    arch: int
    label: str
    receipt: str
    expected_skips: str
    header_keys: str
    primary: bool
    tier: str = 'minor'
    waivers: tuple = ()
    retired_in: str | None = None

    @property
    def sm(self) -> str:
        return f'sm_{self.arch}'


def _row(entry, *, primary, table_path) -> ReceiptRow:
    if not isinstance(entry, dict):
        raise ValueError(f'{table_path}: each row must be a mapping, got {entry!r}')
    missing = [k for k in ('arch', 'label', 'receipt') if k not in entry]
    if missing:
        raise ValueError(f'{table_path}: row {entry!r} lacks {missing}')
    unknown = sorted(set(entry) - {'arch', 'label', 'receipt', 'expected_skips', 'header_keys',
                                  'tier', 'waivers', 'retired_in'})
    if unknown:
        raise ValueError(f'{table_path}: row {entry!r} has unknown key(s) {unknown}')
    arch = entry['arch']
    if not isinstance(arch, int) or isinstance(arch, bool) or arch <= 0:
        raise ValueError(f'{table_path}: arch must be a positive integer '
                         f'(compute capability times ten), got {arch!r}')
    receipt = str(entry['receipt'])
    if Path(receipt).is_absolute() or '..' in Path(receipt).parts:
        raise ValueError(f'{table_path}: receipt paths are repository-relative: {receipt}')
    if not primary and not receipt.startswith(SECONDARY_DIR + '/'):
        raise ValueError(f'{table_path}: secondary receipt {receipt} must live under '
                         f'{SECONDARY_DIR}/ (gpu-proof.json is the primary receipt)')
    if primary:
        header_keys = entry.get('header_keys', DEFAULT_HEADER_KEYS)
    else:
        header_keys = entry.get('header_keys', receipt[:-len('.json')] + '-header-keys.json'
                                if receipt.endswith('.json') else receipt + '-header-keys.json')
    tier = entry.get('tier', 'primary' if primary else 'minor')
    if tier not in {'primary', 'minor', 'major', 'retired'} or (tier == 'primary') != primary:
        raise ValueError(f'{table_path}: invalid tier {tier!r} for this row')
    waivers = entry.get('waivers', [])
    if not isinstance(waivers, list) or (primary and waivers):
        raise ValueError(f'{table_path}: waivers must be a list; primary waivers are forbidden')
    versions = set()
    for waiver in waivers:
        if not isinstance(waiver, dict) or set(waiver) != {'version', 'reason'}:
            raise ValueError(f'{table_path}: waiver needs exactly version and reason')
        release_version(waiver['version'])
        if not isinstance(waiver['reason'], str) or not waiver['reason'].strip():
            raise ValueError(f'{table_path}: waiver reason must be nonempty')
        if waiver['version'] in versions:
            raise ValueError(f'{table_path}: duplicate waiver version')
        versions.add(waiver['version'])
    retired = entry.get('retired_in')
    if tier == 'retired':
        if not retired or release_version(retired)[2] != 0:
            raise ValueError(f'{table_path}: retirement needs an X.Y.0 version')
    elif retired is not None:
        raise ValueError(f'{table_path}: retired_in requires retired tier')
    for value in (header_keys, entry.get('expected_skips', DEFAULT_EXPECTED_SKIPS)):
        if not isinstance(value, str) or Path(value).is_absolute() or '..' in Path(value).parts:
            raise ValueError(f'{table_path}: evidence paths must be repository-relative')
    return ReceiptRow(arch=arch, label=str(entry['label']), receipt=receipt,
                      expected_skips=str(entry.get('expected_skips', DEFAULT_EXPECTED_SKIPS)),
                      header_keys=str(header_keys), primary=primary, tier=tier,
                      waivers=tuple((w['version'], w['reason']) for w in waivers), retired_in=retired)


def release_version(version: str) -> tuple[int, int, int]:
    """Only explicit stable X.Y.Z releases; reject ambiguous/pre-release tags."""
    if not isinstance(version, str) or not re.fullmatch(r'(0|[1-9][0-9]*)\.(0|[1-9][0-9]*)\.(0|[1-9][0-9]*)', version):
        raise ValueError(f'expected stable X.Y.Z release version, got {version!r}')
    return tuple(map(int, version.split('.')))


def requires_fresh(row: ReceiptRow, version: str) -> bool:
    major, minor, patch = release_version(version)
    if row.tier == 'retired':
        if release_version(version) < release_version(row.retired_in):
            raise ValueError('cannot apply retirement to an earlier release')
        return False
    return row.primary or (row.tier == 'minor' and patch == 0) or (
        row.tier == 'major' and minor == 0 and patch == 0)


def provisional_row(arch: int) -> ReceiptRow:
    """Evidence routing only: an unknown architecture gains no support tier."""
    if type(arch) is not int or arch <= 0:
        raise ValueError('no valid GPU architecture')
    return ReceiptRow(arch, 'Contributor GPU (not a support claim)',
                      'receipts/provisional.json', 'test/gpu-proof-provisional-skips.txt',
                      'receipts/provisional-header-keys.json', False, 'provisional')


def load_receipt_table(path: Path = RECEIPTS_TABLE) -> list[ReceiptRow]:
    """Parse the table; the primary row comes first. Every arch and every
    receipt path is unique, and secondary receipts never alias the primary."""
    import yaml  # noqa: PLC0415  (pyyaml, a verifier dependency already)
    data = yaml.safe_load(Path(path).read_text())
    if not isinstance(data, dict) or 'primary' not in data:
        raise ValueError(f'{path}: expected a mapping with a primary row')
    unknown = sorted(set(data) - {'primary', 'secondary'})
    if unknown:
        raise ValueError(f'{path}: unknown top-level key(s) {unknown}')
    rows = [_row(data['primary'], primary=True, table_path=path)]
    secondary = data.get('secondary') or []
    if not isinstance(secondary, list):
        raise ValueError(f'{path}: secondary must be a list of rows')
    rows += [_row(e, primary=False, table_path=path) for e in secondary]
    for key in ('arch', 'receipt', 'header_keys'):
        values = [getattr(r, key) for r in rows]
        dupes = sorted({v for v in values if values.count(v) > 1})
        if dupes:
            raise ValueError(f'{path}: duplicate {key} {dupes}; one row per architecture')
    return rows


def primary_row(table: list[ReceiptRow] | None = None) -> ReceiptRow:
    return (table or load_receipt_table())[0]


def receipt_row_for_arch(arch: int, table: list[ReceiptRow] | None = None) -> ReceiptRow:
    """The row recording on a GPU of compute capability ``arch`` writes to.
    ``arch`` 0 means no GPU was detected; an arch without a row is refused."""
    table = table or load_receipt_table()
    if not arch:
        raise ValueError('no GPU detected (nvidia-smi reported no compute capability); '
                         'a receipt can only be recorded on a GPU box')
    for row in table:
        if row.arch == arch:
            return row
    known = ', '.join(f'{r.sm} ({r.label})' for r in table)
    raise ValueError(f'no receipt row for sm_{arch}: add a row for this architecture to '
                     f'{RECEIPTS_TABLE.relative_to(REPO_ROOT)} (known: {known})')


def receipt_arch(receipt: dict) -> int | None:
    """Compute capability recorded in a receipt's environment.gpu_info, as an
    integer (120 for "12.0"); None when the receipt carries no GPU record."""
    devices = ((receipt.get('environment') or {}).get('gpu_info') or {}).get('devices') or []
    cap = devices[0].get('compute_capability') if devices and isinstance(devices[0], dict) else None
    if not cap:
        return None
    try:
        major, minor = str(cap).split('.')
        return int(major) * 10 + int(minor)
    except ValueError:
        return None


def require_primary_receipt(receipt: dict, table: list[ReceiptRow] | None = None) -> ReceiptRow:
    """The collection manifest is defined by the primary receipt. Refuse any
    receipt recorded on another architecture, or one with no GPU record."""
    primary = primary_row(table)
    arch = receipt_arch(receipt)
    if arch is None:
        raise ValueError('refusing to write the manifest: the receipt records no GPU '
                         '(environment.gpu_info), so it cannot be the primary receipt '
                         f'({primary.sm}, {primary.receipt})')
    if arch != primary.arch:
        row = next((r for r in (table or load_receipt_table()) if r.arch == arch), None)
        where = f'{row.label}, {row.receipt}' if row else 'no row in the receipt table'
        raise ValueError(f'refusing to write the manifest from a secondary receipt: '
                         f'recorded on sm_{arch} ({where}); the collection manifest '
                         f'test/gpu-proof-tests.txt is defined by the primary receipt '
                         f'({primary.sm} {primary.label}, {primary.receipt})')
    return primary


# ─── reporting ───────────────────────────────────────────────────────────────

NOT_RECORDED = 'not in the receipt'


def _gpu_device(receipt: dict) -> dict:
    devices = ((receipt.get('environment') or {}).get('gpu_info') or {}).get('devices') or []
    return devices[0] if devices and isinstance(devices[0], dict) else {}


def _gpu_name(receipt: dict) -> str:
    dev = _gpu_device(receipt)
    name = dev.get('name') or NOT_RECORDED
    mem = str(dev.get('memory') or '')
    if mem.endswith('MiB'):
        try:
            name += f', {round(int(mem[:-3].strip()) / 1024)} GB'
        except ValueError:
            pass
    return name


def _ended_at(receipt: dict) -> datetime | None:
    ended = (receipt.get('session') or {}).get('ended_at')
    if not isinstance(ended, str):
        return None
    try:
        return datetime.fromisoformat(ended.replace('Z', '+00:00'))
    except ValueError:
        return None


def _counts(receipt: dict) -> dict:
    counts: dict = {}
    for t in receipt.get('tests') or []:
        counts[t.get('outcome', 'unknown')] = counts.get(t.get('outcome', 'unknown'), 0) + 1
    return counts


def receipt_summary(receipt: dict) -> dict:
    """The fields the compatibility page and verify-all summaries show."""
    arch = receipt_arch(receipt)
    dev = _gpu_device(receipt)
    env = receipt.get('environment') or {}
    repo = receipt.get('repo') or {}
    counts = _counts(receipt)
    ended = _ended_at(receipt)
    shards = receipt.get('shards') or []
    return {
        'arch': arch,
        'sm': f'sm_{arch}' if arch is not None else NOT_RECORDED,
        'compute_capability': dev.get('compute_capability'),
        'gpu': _gpu_name(receipt),
        'driver': dev.get('driver_version') or NOT_RECORDED,
        'python': (f'CPython {env["python_version"]}' if env.get('python_version')
                   else NOT_RECORDED),
        'platform': env.get('platform') or NOT_RECORDED,
        'commit': (repo.get('commit_sha') or '')[:7] or NOT_RECORDED,
        'branch': repo.get('branch'),
        'date': ended.date().isoformat() if ended else NOT_RECORDED,
        'ended_at': ended,
        'tests': len(receipt.get('tests') or []),
        'passed': counts.get('passed', 0),
        'skipped': counts.get('skipped', 0),
        'failed': counts.get('failed', 0) + counts.get('error', 0),
        'carried': sum(1 for s in shards if s.get('carried')),
        'shards': len(shards),
    }


def rst_table_rows(receipts: list[dict]) -> str:
    """List-table rows in the column order of the "What we ran" table of
    docs/source/user_guide/getting_started/compatibility.rst: GPU
    architecture, GPU, CUDA toolkit (nvcc), Driver, Python, torch wheel, jax
    plugin, OS, What ran, Date. The receipt does not record the toolkit or the
    framework wheels; those cells say so and are filled in by hand."""
    out = []
    for receipt in receipts:
        s = receipt_summary(receipt)
        cap = s['compute_capability']
        arch_cell = f'{s["sm"]} (compute capability {cap})' if cap else s['sm']
        ran = f'receipt ({s["tests"]} tests: {s["passed"]} passed, {s["skipped"]} skipped'
        if s['failed']:
            ran += f', {s["failed"]} failed'
        ran += ')'
        if s['carried']:
            ran += f', {s["carried"]} of {s["shards"]} shards carried'
        ran += f'; commit {s["commit"]}'
        cells = [arch_cell, s['gpu'], NOT_RECORDED, s['driver'], s['python'],
                 NOT_RECORDED, NOT_RECORDED, s['platform'], ran, f'receipt {s["date"]}']
        out.append('   * - ' + '\n     - '.join(cells))
    return '\n'.join(out) + ('\n' if out else '')


def _gpu_proof_bin() -> str:
    sibling = Path(sys.executable).with_name('gpu-proof')
    if sibling.exists():
        return str(sibling)
    venv = REPO_ROOT / '.venv' / 'bin' / 'gpu-proof'
    return str(venv) if venv.exists() else (shutil.which('gpu-proof') or 'gpu-proof')


@contextmanager
def recorded_tree(root: Path, commit: str):
    """Never execute historical code. Only give the installed verifier its data."""
    if not re.fullmatch(r'[0-9a-f]{40}', commit):
        raise ValueError('recorded commit must be a full SHA-1')
    subprocess.run(['git', 'merge-base', '--is-ancestor', commit, 'HEAD'],
                   cwd=root, check=True, capture_output=True)
    directory = Path(tempfile.mkdtemp(prefix='grid-receipt-validity-'))
    tree = directory / 'tree'
    added = False
    try:
        subprocess.run(['git', 'worktree', 'add', '--detach', str(tree), commit],
                       cwd=root, check=True, capture_output=True)
        added = True
        try:
            yield tree, directory
        finally:
            # No force: unexpected edits must not be silently discarded.
            # If removal fails, preserve the private directory for diagnosis.
            subprocess.run(['git', 'worktree', 'remove', str(tree)],
                           cwd=root, check=True, capture_output=True)
        shutil.rmtree(directory)  # only our scratch, after verified worktree removal
    except BaseException:
        if not added and not tree.exists():
            directory.rmdir()
        raise


def historical_policy(policy: dict, receipt: dict, now: datetime) -> dict:
    """Evaluate age at recording, not today; retain the original carry-age bound.

    gpu-proof has no as-of option. Shift both age ceilings by the elapsed whole
    days plus one (rounding allowance). Signature, timestamps, dirty flags,
    ancestry, fingerprint, results and exact historical scope still verify.
    This policy is NEVER used for current merge or release freshness.
    """
    ended = _ended_at(receipt)
    if ended is None or ended.tzinfo is None:
        raise ValueError('historical receipt needs a timezone-aware end timestamp')
    elapsed = max(0, (now - ended).days + 1)
    return dict(policy, max_age_days=policy.get('max_age_days', 30) + elapsed,
                carried_max_age_days=policy.get('carried_max_age_days', 30) + elapsed)


def verify_recorded(row: ReceiptRow, *, root: Path, policy: str, now: datetime,
                    run=subprocess.run):
    import yaml
    path = (root / row.receipt).resolve()
    path.relative_to(root.resolve())
    receipt = json.loads(path.read_text())
    if receipt_arch(receipt) != row.arch:
        raise ValueError('receipt GPU architecture differs from table row')
    with recorded_tree(root, receipt['repo']['commit_sha']) as (tree, scratch):
        # Policy comes from the trusted caller, not the recorded checkout.
        rules = historical_policy(yaml.safe_load((root / policy).read_text()), receipt, now)
        policy_path = scratch / 'historical-policy.json'
        policy_path.write_text(json.dumps(rules))
        skips = tree / row.expected_skips
        skips.resolve().relative_to(tree.resolve())
        if not skips.is_file():
            raise ValueError('expected-skip baseline missing at recorded commit')
        return run([_gpu_proof_bin(), 'verify', '--receipt', str(path), '--repo', str(tree),
                    '--policy', str(policy_path), '--expected-skips', str(skips)],
                   cwd=root, capture_output=True, text=True)


def verify_all(table: list[ReceiptRow] | None = None, *, root: Path = REPO_ROOT,
               policy: str = EVERYDAY_POLICY, run=subprocess.run,
               now: datetime | None = None, historical: bool = True,
               required_receipts: set[str] | None = None) -> int:
    """Verify every receipt at its recorded ancestor (historical validity).
    The trusted everyday policy and historical skips apply; one summary per
    architecture. A missing secondary receipt is reported, not an error; a
    missing primary receipt, or any failed verification, returns 1."""
    now = now or datetime.now(timezone.utc)
    rc = 0
    for row in table or load_receipt_table():
        path = root / row.receipt
        head = f'{row.sm:7s} {row.label:14s} {row.receipt}'
        if not path.exists():
            kind = 'primary' if row.primary else 'secondary'
            print(f'{head}: not recorded ({kind} receipt missing)')
            if row.primary or row.receipt in (required_receipts or set()):
                rc = 1
            continue
        try:
            summary = receipt_summary(json.loads(path.read_text()))
        except (ValueError, OSError) as exc:
            print(f'{head}: unreadable ({exc}); FAIL')
            rc = 1
            continue
        age = (f'{(now - summary["ended_at"]).days} d old' if summary['ended_at']
               else 'age unknown')
        try:
            if historical:
                proc = verify_recorded(row, root=root, policy=policy, now=now, run=run)
            else:
                proc = run([_gpu_proof_bin(), 'verify', '--receipt', str(path), '--repo', str(root),
                            '--policy', str(root / policy),
                            '--expected-skips', str(root / row.expected_skips)],
                           cwd=root, capture_output=True, text=True)
        except (ValueError, KeyError, OSError, subprocess.CalledProcessError) as exc:
            print(f'{head}: FAIL ({exc})')
            rc = 1
            continue
        ok = proc.returncode == 0
        if not ok:
            rc = 1
            sys.stdout.write(proc.stdout or '')
            sys.stdout.write(proc.stderr or '')
        print(f'{head}: {summary["gpu"]}; commit {summary["commit"]}; {age}; '
              f'{summary["passed"]} passed, {summary["skipped"]} skipped; '
              f'{"ok" if ok else f"FAIL (rc={proc.returncode})"}')
    return rc


def verify_release(version: str, table=None, *, root: Path = REPO_ROOT,
                   run=subprocess.run) -> int:
    """Historical validity for all evidence, fresh full evidence by release tier."""
    release_version(version)
    table = table or load_receipt_table(root / 'test/gpu-proof-receipts.yaml')
    rc = verify_all(table, root=root, run=run)
    for row in table:
        if not requires_fresh(row, version):
            continue
        waiver = dict(row.waivers).get(version)
        if waiver and not row.primary:
            print(f'{row.sm}: WAIVED for {version}: {waiver}')
            continue
        path = root / row.receipt
        if not path.is_file():
            print(f'{row.sm}: required {row.tier} receipt missing for {version}; FAIL')
            rc = 1
            continue
        if receipt_arch(json.loads(path.read_text())) != row.arch:
            print(f'{row.sm}: receipt architecture mismatch; FAIL')
            rc = 1
            continue
        proc = run([_gpu_proof_bin(), 'verify', '--receipt', str(path), '--repo', str(root),
                    '--policy', str(root / 'test/gpu-proof-policy-release.yaml'),
                    '--expected-skips', str(root / row.expected_skips)],
                   cwd=root, capture_output=True, text=True)
        sys.stdout.write(proc.stdout or '')
        sys.stdout.write(proc.stderr or '')
        rc |= proc.returncode != 0
    return int(rc)


def tier_table(table=None, *, root: Path = REPO_ROOT) -> str:
    """Generated policy inventory, distinct from manually curated toolchain facts."""
    lines = ['.. list-table:: Hardware receipt policy (generated)',
             '   :header-rows: 1', '',
             '   * - Architecture', '     - Tier', '     - Recorded source version',
             '     - Receipt date', '     - Waivers / retirement']
    for row in table or load_receipt_table(root / 'test/gpu-proof-receipts.yaml'):
        version = date = 'not recorded'
        path = root / row.receipt
        if path.is_file():
            receipt = json.loads(path.read_text())
            date = receipt_summary(receipt)['date']
            commit = receipt['repo']['commit_sha']
            if not re.fullmatch('[0-9a-f]{40}', commit):
                raise ValueError('invalid receipt commit')
            result = subprocess.run(['git', 'show', f'{commit}:pyproject.toml'], cwd=root,
                                    check=True, capture_output=True, text=True)
            match = re.search(r'^version\s*=\s*"([^"]+)"', result.stdout, re.M)
            version = (match[1] if match else 'unknown') + f' ({commit[:7]})'
        note = '; '.join(f'{v}: {reason}' for v, reason in row.waivers) or 'none'
        if row.retired_in:
            note += f'; retired in {row.retired_in}'
        cells = [f'{row.sm} ({row.label})', row.tier, version, date, note]
        lines.append('   * - ' + '\n     - '.join(c.replace('\n', ' ') for c in cells))
    return '\n'.join(lines) + '\n'


def receipts_on_disk(table: list[ReceiptRow] | None = None, *, root: Path = REPO_ROOT) -> list[dict]:
    out = []
    for row in table or load_receipt_table():
        path = root / row.receipt
        if path.exists():
            out.append(json.loads(path.read_text()))
    return out


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--receipt', type=Path, metavar='PATH',
                        help='receipt to act on (with --write-manifest or --table)')
    parser.add_argument('--write-manifest', type=Path, metavar='PATH',
                        help='Explicit maintainer action: rewrite the reviewed scope manifest '
                             '(test/gpu-proof-tests.txt) from a verified full-scope PRIMARY '
                             'receipt (--receipt); a secondary receipt is refused')
    parser.add_argument('--verify-all', action='store_true',
                        help='verify every receipt of test/gpu-proof-receipts.yaml that exists '
                             'on disk (everyday policy, per-arch expected skips); one line per arch')
    parser.add_argument('--table', action='store_true',
                        help='print reStructuredText list-table rows for the compatibility page '
                             'from every receipt on disk (or from --receipt alone)')
    parser.add_argument('--release', metavar='X.Y.Z', help='verify validity and release-tier freshness')
    parser.add_argument('--tier-table', action='store_true', help='generate policy inventory RST')
    parser.add_argument('--repo', type=Path, default=REPO_ROOT,
                        help='candidate repository; no code from it is executed')
    parser.add_argument('--policy-root', type=Path, default=REPO_ROOT,
                        help='trusted checkout owning table and policy (PR base in CI)')
    args = parser.parse_args(argv)
    chosen = [n for n, v in (('--write-manifest', args.write_manifest),
                             ('--verify-all', args.verify_all), ('--table', args.table),
                             ('--release', args.release), ('--tier-table', args.tier_table)) if v]
    if len(chosen) != 1:
        parser.error('choose exactly one of --write-manifest PATH, --verify-all, --table')
    if args.write_manifest:
        if not args.receipt:
            parser.error('--write-manifest needs --receipt')
        receipt = json.loads(args.receipt.read_text())
        require_primary_receipt(receipt)
        ids = write_manifest(receipt, args.write_manifest)
        print(f'Wrote {len(ids)} node IDs -> {args.write_manifest}')
        return 0
    if args.verify_all:
        table = load_receipt_table(args.policy_root / 'test/gpu-proof-receipts.yaml')
        required = {row.receipt for row in table if (args.policy_root / row.receipt).is_file()}
        return verify_all(table, root=args.repo, policy=str(args.policy_root / EVERYDAY_POLICY),
                          required_receipts=required)
    if args.release:
        return verify_release(args.release)
    if args.tier_table:
        sys.stdout.write(tier_table())
        return 0
    receipts = ([json.loads(args.receipt.read_text())] if args.receipt
                else receipts_on_disk())
    sys.stdout.write(rst_table_rows(receipts))
    return 0


if __name__ == '__main__':
    sys.exit(main())
