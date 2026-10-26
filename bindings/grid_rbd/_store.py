"""Shared publication rules for wrapper and consumer content-addressed entries."""
from contextlib import contextmanager
import fcntl
import json
import os
from pathlib import Path
import shutil
import tempfile
import uuid


def staging_dir(store: Path, input_key: str) -> Path:
    store.mkdir(parents=True, exist_ok=True)
    return Path(tempfile.mkdtemp(prefix=f".gen_{os.getpid()}_{input_key[:8]}_", dir=store))


def atomic_write_text(path: Path, text: str) -> None:
    """Publish a small sidecar with a private temporary, including across threads."""
    fd, name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    temporary = Path(name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            stream.write(text)
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def entry_metadata(entry: Path, artifact: str) -> dict | None:
    """A readable manifest and all runtime files are required for reuse.

    Missing/corrupt entries are cache misses. Permission and other I/O errors
    are not misses and must reach the caller.
    """
    try:
        for name in (artifact, "grid.cuh"):
            if not (entry / name).is_file() or (entry / name).stat().st_size == 0:
                return None
        meta = json.loads((entry / "meta.json").read_text())
    except (FileNotFoundError, json.JSONDecodeError, UnicodeDecodeError):
        return None
    return meta if isinstance(meta, dict) else None


@contextmanager
def entry_lock(entry: Path):
    # Lock inode lives outside the replaceable entry and is never unlinked.
    locks = entry.parent / ".locks"
    locks.mkdir(parents=True, exist_ok=True)
    with (locks / (entry.name + ".lock")).open("a+") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(lock.fileno(), fcntl.LOCK_UN)


def publish_entry(staged: Path, entry: Path, artifact: str, *, force: bool = False) -> dict:
    """Adopt a complete concurrent winner, or replace an incomplete entry.

    Only the owner of the per-content lock may displace an entry. Keep the
    previous directory until publication succeeds, restoring it on failure.
    Do not interpret arbitrary rename errors as a successful concurrent build.
    """
    meta = entry_metadata(staged, artifact)
    if meta is None:
        raise RuntimeError(f"cannot publish incomplete artifact: {staged}")
    with entry_lock(entry):
        winner = entry_metadata(entry, artifact)
        if winner is not None and not force:
            return winner
        backup = entry.with_name(f".replaced_{entry.name}_{uuid.uuid4().hex}")
        displaced = False
        try:
            if entry.exists():
                os.replace(entry, backup)
                displaced = True
            os.replace(staged, entry)
        except OSError:
            if displaced:
                os.replace(backup, entry)
            raise
        if displaced:
            shutil.rmtree(backup)
    return meta


def publish_object(obj: Path, cached: Path) -> None:
    """Unique temporary even for threads in the same process and object key."""
    fd, name = tempfile.mkstemp(prefix=cached.name + ".", suffix=".tmp", dir=cached.parent)
    os.close(fd)
    tmp = Path(name)
    try:
        shutil.move(str(obj), str(tmp))
        os.replace(tmp, cached)
    finally:
        tmp.unlink(missing_ok=True)
