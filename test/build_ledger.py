"""A record of the robot libraries each wrapper test module asks for.

Wrapper test modules register their robots with options computed in the test
(parametrized dtypes, armature vectors, models written to a temporary file), so
a list kept by hand in the driver drifts from the code. A module outside that
list compiles its robots inside its own GPU shard, one after the other, while
the compile pool sits idle.

This module records the requests instead. While a wrapper shard runs, every
build request that reaches the shared cache is appended to a file: the model
text and the options. After a clean shard the driver stores the list under the
module's name and the hash of its source. On the next run the driver hands the
stored requests to the compile pool, so the shard finds its libraries built.

The record only moves compiles from a shard to the pool. It does not choose
which tests run or what they check. A request that no longer matches (the
module was edited) is dropped with the module's whole record, and a replay that
resolves to another key than the one recorded builds a library nobody loads,
which costs time and nothing else.
"""
from __future__ import annotations

import hashlib
import inspect
import json
import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
LEDGER_PATH = REPO_ROOT / "test" / ".split_suite" / "wrapper_builds.json"
RECORD_ENV = "GRID_SPLIT_BUILD_RECORD"
SCHEMA = 1
# Arguments that do not describe the library: where it is stored, whether to
# force or only plan the build, and the model text (kept beside the options).
_NOT_OPTIONS = ("cache_dir", "force_rebuild", "plan_only", "urdf_string")


def _plain(value):
    """``value`` as JSON data; TypeError for anything a replay could not rebuild."""
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, (list, tuple)):
        return [_plain(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _plain(item) for key, item in value.items()}
    if isinstance(value, Path):
        return str(value)
    if hasattr(value, "tolist"):  # numpy arrays and scalars
        return _plain(value.tolist())
    raise TypeError(f"not replayable: {type(value).__name__}")


def request_record(urdf_path, kwargs: dict) -> dict | None:
    """The replayable form of one build request, or None when it is not one.

    A request with its own ``cache_dir`` builds into a private cache the pool
    cannot fill, and a forced or planned request is not a plain build.
    """
    if kwargs.get("cache_dir") is not None or kwargs.get("force_rebuild") or kwargs.get("plan_only"):
        return None
    try:
        text = kwargs.get("urdf_string")
        if text is None:
            text = Path(urdf_path).expanduser().read_text(encoding="utf-8")
        options = _plain({key: value for key, value in kwargs.items() if key not in _NOT_OPTIONS})
    except (TypeError, OSError, UnicodeDecodeError):
        return None
    return {"urdf": text, "options": options}


def install_recorder(path: str | os.PathLike) -> None:
    """Wrap ``grid_rbd.warm_robot`` so each successful shared-cache build request
    of this process is appended to ``path``, one JSON object per line."""
    import grid_rbd  # noqa: PLC0415

    real = grid_rbd.warm_robot
    if getattr(real, "_records_builds", False):
        return
    target = Path(path)

    def warm_robot(name, urdf_path=None, **kwargs):
        result = real(name, urdf_path, **kwargs)
        record = request_record(urdf_path, kwargs)
        if record is not None:
            record["key"] = result[0]
            target.parent.mkdir(parents=True, exist_ok=True)
            with target.open("a", encoding="utf-8") as out:
                out.write(json.dumps(record, sort_keys=True) + "\n")
        return result

    warm_robot._records_builds = True
    grid_rbd.warm_robot = warm_robot


def module_hash(module_path: Path) -> str:
    return hashlib.sha256(Path(module_path).read_bytes()).hexdigest()


def _load(path: Path | None) -> dict:
    path = path or LEDGER_PATH
    try:
        data = json.loads(path.read_text())
    except (OSError, ValueError):
        return {}
    modules = data.get("modules") if isinstance(data, dict) and data.get("schema") == SCHEMA else None
    return modules if isinstance(modules, dict) else {}


def read_records(record_path: Path) -> list[dict]:
    """The distinct requests one shard appended, in first-seen order."""
    records, seen = [], set()
    try:
        lines = Path(record_path).read_text(encoding="utf-8").splitlines()
    except OSError:
        return []
    for line in lines:
        try:
            record = json.loads(line)
        except ValueError:
            continue
        if not (isinstance(record, dict) and isinstance(record.get("urdf"), str)
                and isinstance(record.get("options"), dict)):
            continue
        identity = json.dumps([record["urdf"], record["options"]], sort_keys=True)
        if identity not in seen:
            seen.add(identity)
            records.append(record)
    return records


def store(module: str, module_path: Path, records: list[dict], path: Path | None = None) -> None:
    """Replace ``module``'s stored requests with those of its last clean run."""
    path = path or LEDGER_PATH
    modules = _load(path)
    modules[module] = {"module_sha256": module_hash(module_path), "builds": records}
    path.parent.mkdir(parents=True, exist_ok=True)
    scratch = path.with_name(f"{path.name}.{os.getpid()}.tmp")
    scratch.write_text(json.dumps({"schema": SCHEMA, "modules": modules}, indent=1, sort_keys=True) + "\n")
    os.replace(scratch, path)


def _asset_names() -> dict[str, str]:
    """sha256 of each packaged robot model -> its name, for readable job labels."""
    names = {}
    for urdf in sorted((REPO_ROOT / "grid_assets" / "robots").glob("*.urdf")):
        names.setdefault(hashlib.sha256(urdf.read_bytes()).hexdigest(), urdf.stem)
    return names


def _option_defaults() -> dict:
    import grid_rbd  # noqa: PLC0415

    real = inspect.unwrap(grid_rbd.warm_robot)
    return {name: parameter.default for name, parameter in inspect.signature(real).parameters.items()
            if parameter.default is not inspect.Parameter.empty}


def entries(module: str, module_path: Path, path: Path | None = None) -> list[dict]:
    """Pool entries for ``module``'s stored requests; none when the module changed.

    An entry is the driver's warm form: ``robot`` (a label), ``urdf_string`` and
    the options that differ from the defaults, so an entry equal to one the
    driver lists by hand has the same description and is built once. The recorded cache key stays in the
    file for a reader; it is not part of the entry, which must not change when
    the generator does.
    """
    stored = _load(path).get(module)
    if not (isinstance(stored, dict) and stored.get("module_sha256") == module_hash(module_path)
            and isinstance(stored.get("builds"), list)):
        return []
    assets, defaults, out = _asset_names(), _option_defaults(), []
    for record in stored["builds"]:
        if not (isinstance(record, dict) and isinstance(record.get("urdf"), str)
                and isinstance(record.get("options"), dict)):
            continue
        digest = hashlib.sha256(record["urdf"].encode("utf-8")).hexdigest()
        options = {key: value for key, value in record["options"].items()
                   if key in defaults and _plain(defaults[key]) != value}
        entry = {"robot": assets.get(digest, f"model_{digest[:10]}"), **options}
        if digest not in assets:
            entry["urdf_string"] = record["urdf"]
        out.append(entry)
    return out
