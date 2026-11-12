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
