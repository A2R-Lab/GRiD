"""CPU-only packaging hooks. Bundle pinned peers/resources; never invoke CUDA."""
import hashlib
import json
from pathlib import Path
import shutil
import subprocess

from setuptools.command.build_py import build_py
from setuptools.command.sdist import sdist

ROOT = Path(__file__).resolve().parent
PEERS = ('GLASS', 'URDFParser', 'RBDReference')


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def source_bundle(root):
    """Return validated (provenance, data sources); no mutation of the checkout."""
    data = root / 'grid_codegen/_data'
    saved = data / 'provenance.json'
    if saved.is_file():
        info = json.loads(saved.read_text())
        files = {name: data / name for name in info['resources']}
    else:
        def git(*args):
            return subprocess.check_output(['git', *args], cwd=root, text=True).strip()
        info = {'schema': 1, 'peers': {}, 'resources': {}, 'python_sources': {}}
        for name in PEERS:
            path = root / 'external' / name
            pin = git('rev-parse', f'HEAD:external/{name}')
            actual = git('-C', str(path), 'rev-parse', 'HEAD')
            if actual != pin or git('-C', str(path), 'status', '--porcelain', '--untracked-files=no'):
                raise RuntimeError(f'{name} must be clean at its committed GRiD gitlink before packaging')
            info['peers'][name] = {'commit': pin, 'repository': f'https://github.com/A2R-Lab/{name}'}
        glass = root / 'external/GLASS'
        tracked = git('-C', str(glass), 'ls-tree', '-r', '--name-only',
                      info['peers']['GLASS']['commit']).splitlines()
        files = {f'GLASS/{name}': glass / name for name in tracked
                 if name.endswith('.cuh') and ('/' not in name or name.startswith('src/'))}
        profiles = root / 'config/launch_configs'
        files.update({f'launch_configs/{p.relative_to(profiles).as_posix()}': p for p in profiles.rglob('*.json')})
        for name in PEERS:
            licenses = sorted((root / 'external' / name).glob('LICENSE*'))
            if not licenses:
                raise RuntimeError(f'Missing license for {name}')
            files.update({f'licenses/{name}/{p.name}': p for p in licenses if p.is_file()})
        info['resources'] = {name: digest(path) for name, path in sorted(files.items())}
        for name in ('URDFParser', 'RBDReference'):
            for path in sorted((root / 'external' / name).glob('*.py')):
                pinned = subprocess.check_output(['git', '-C', str(path.parent),
                    'show', f"{info['peers'][name]['commit']}:{path.name}"], cwd=root)
                if path.read_bytes() != pinned:
                    raise RuntimeError(f'Peer source differs from pinned commit: {name}/{path.name}')
                info['python_sources'][f'{name}/{path.name}'] = digest(path)
    # Rebuilding an exported sdist validates its bundled bytes too.
    for name, expected in info['resources'].items():
        if digest(files[name]) != expected:
            raise RuntimeError(f'Bundled resource hash mismatch: {name}')
    for name, expected in info['python_sources'].items():
        if digest(root / 'external' / name) != expected:
            raise RuntimeError(f'Bundled peer source hash mismatch: {name}')
    return info, files


def stage_resources(destination):
    info, files = source_bundle(ROOT)
    destination.mkdir(parents=True, exist_ok=True)
    for name, source in files.items():
        target = destination / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)
    (destination / 'provenance.json').write_text(json.dumps(info, indent=2, sort_keys=True) + '\n')


class BuildPy(build_py):
    def run(self):
        super().run()
        if not self.editable_mode:
            stage_resources(Path(self.build_lib) / 'grid_codegen/_data')

    def get_outputs(self, include_bytecode=1):
        outputs = super().get_outputs(include_bytecode)
        if not self.editable_mode:
            info, _ = source_bundle(ROOT)
            outputs += [str(Path(self.build_lib) / 'grid_codegen/_data' / name)
                        for name in [*info['resources'], 'provenance.json']]
        return outputs


class Sdist(sdist):
    def make_release_tree(self, base_dir, files):
        super().make_release_tree(base_dir, files)
        stage_resources(Path(base_dir) / 'grid_codegen/_data')
