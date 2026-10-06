"""Runtime resources for editable checkouts and installed wheels.

Wheels contain ordinary unpacked package files. No Git executable, network,
repository-shaped site-packages layout, or GPU is required for resolution.
"""
from importlib.util import find_spec
import json
from pathlib import Path


def package_dir(name):
    spec = find_spec(name)
    if spec is None or not spec.submodule_search_locations:
        raise RuntimeError(f'Required bundled package {name!r} is missing; reinstall grid-rbd')
    return Path(next(iter(spec.submodule_search_locations)))


def data_dir():
    return Path(__file__).resolve().parent / '_data'


def checkout_root():
    if (data_dir() / 'provenance.json').is_file():
        return None
    root = Path(__file__).resolve().parent.parent
    return root if (root / 'pyproject.toml').is_file() and (root / 'external/URDFParser').is_dir() else None


def resource_path(name):
    roots = {'GLASS': ('external', 'GLASS'), 'launch_configs': ('config', 'launch_configs')}
    if name not in roots:
        raise ValueError(f'Unknown GRiD resource: {name}')
    root = checkout_root()
    path = root.joinpath(*roots[name]) if root else data_dir() / name
    if not path.is_dir():
        raise FileNotFoundError(f'Missing GRiD resource {name}: {path}. Initialize submodules in a checkout or reinstall grid-rbd.')
    return path


def bundled_provenance():
    path = data_dir() / 'provenance.json'
    return json.loads(path.read_text()) if path.is_file() else None


def collision_include_dir():
    return Path(__file__).resolve().parent / 'collision'
