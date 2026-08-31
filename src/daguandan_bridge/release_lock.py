from __future__ import annotations

"""Validation for the offline, hash-locked Windows release inputs."""

import hashlib
import importlib.metadata
import json
import os
import platform
from pathlib import Path
from pathlib import PurePosixPath
import re
import stat
import subprocess
import sys
from typing import Iterable, Mapping, Sequence


RELEASE_INPUT_AUDIT_SCHEMA = "guandan.release-input-audit/1"
PYTHON_RUNTIME_LOCK_SCHEMA = "guandan.python-runtime-lock/1"
PYTHON_RUNTIME_POLICY = {
    "schema": "guandan.python-runtime-inventory-policy/1",
    "root_files": [
        "LICENSE.txt",
        "python.exe",
        "pythonw.exe",
        "python3.dll",
        "python312.dll",
        "vcruntime140.dll",
        "vcruntime140_1.dll",
    ],
    "recursive_roots": ["DLLs", "Lib", "libs"],
    "excluded_directories": ["Lib/site-packages", "__pycache__"],
    "excluded_suffixes": [".pyc", ".pyo"],
}
_HEX = re.compile(r"^[0-9a-f]{64}$")


class ReleaseLockError(RuntimeError):
    pass


def verify_release_inputs(
    *,
    project_root: Path | str,
    wheelhouse_root: Path | str,
    python_executable: Path | str | None = None,
    verify_installed: bool = False,
    installed_distribution_paths: Sequence[Path | str] | None = None,
) -> dict[str, object]:
    project = Path(project_root).resolve()
    wheelhouse = Path(wheelhouse_root).resolve()
    python_path = Path(python_executable or sys.executable).resolve()
    errors: list[dict[str, object]] = []
    if not wheelhouse.is_dir() or _is_link_or_reparse(wheelhouse):
        raise ReleaseLockError("wheelhouse root is unavailable or a reparse point")
    toolchain = _json(project / "release_toolchain.lock.json")
    wheel_lock = _json(project / "wheelhouse.lock.json")
    requirements_lock = project / "requirements-release.lock"
    requirements_input = project / "requirements-release.in"
    expected_schema = "guandan.release-toolchain-lock/3"
    if toolchain.get("schema") != expected_schema:
        errors.append(_error("LOCK-TOOLCHAIN-SCHEMA", {"expected": expected_schema}))
    acquisition = toolchain.get("acquisition")
    if not isinstance(acquisition, Mapping) or (
        acquisition.get("provider") != "python.org"
        or acquisition.get("artifact") != "python-3.12.0-amd64.exe"
        or acquisition.get("url")
        != "https://www.python.org/ftp/python/3.12.0/python-3.12.0-amd64.exe"
        or _HEX.fullmatch(str(acquisition.get("sha256") or "")) is None
        or not isinstance(acquisition.get("bytes"), int)
        or isinstance(acquisition.get("bytes"), bool)
        or int(acquisition.get("bytes")) <= 0
    ):
        errors.append(_error("LOCK-PYTHON-ACQUISITION", {}))
    if wheel_lock.get("schema") != "guandan.wheelhouse-lock/1":
        errors.append(_error("LOCK-WHEELHOUSE-SCHEMA", {}))
    locks = toolchain.get("locks") if isinstance(toolchain.get("locks"), Mapping) else {}
    for label, path in (
        ("requirements_input", requirements_input),
        ("requirements_lock", requirements_lock),
        ("python_runtime_lock", project / "python_runtime.lock.json"),
        ("wheelhouse_lock", project / "wheelhouse.lock.json"),
    ):
        expected = str(locks.get(f"{label}_sha256", ""))
        actual = _sha256_file(path) if path.is_file() else None
        if expected != actual:
            errors.append(
                _error(
                    "LOCK-FILE-HASH",
                    {"file": path.name, "expected": expected, "actual": actual},
                )
            )
    expected_python = toolchain.get("python_executable")
    expected_python_hash = (
        str(expected_python.get("sha256", ""))
        if isinstance(expected_python, Mapping)
        else ""
    )
    actual_python_hash = _sha256_file(python_path) if python_path.is_file() else None
    if expected_python_hash != actual_python_hash:
        errors.append(
            _error(
                "LOCK-PYTHON-HASH",
                {"expected": expected_python_hash, "actual": actual_python_hash},
            )
        )
    python_base_root = _interpreter_base_prefix(python_path)
    runtime_lock = _load_python_runtime_inventory(project, toolchain, errors)
    _verify_python_runtime_lock(
        toolchain,
        python_base_root,
        errors,
        runtime_inventory=runtime_lock,
    )
    platform_lock = toolchain.get("platform")
    if isinstance(platform_lock, Mapping):
        checks = {
            "system": platform.system(),
            "python_version": platform.python_version(),
            "architecture": platform.machine(),
            "python_cache_tag": sys.implementation.cache_tag,
        }
        for key, actual in checks.items():
            if str(platform_lock.get(key, "")) != str(actual):
                errors.append(
                    _error(
                        "LOCK-PLATFORM-MISMATCH",
                        {"field": key, "expected": platform_lock.get(key), "actual": actual},
                    )
                )
    records = wheel_lock.get("files")
    if not isinstance(records, list) or not records:
        errors.append(_error("LOCK-WHEELHOUSE-FILES", {}))
        records = []
    expected_names: set[str] = set()
    verified: list[dict[str, object]] = []
    for raw in records:
        if not isinstance(raw, Mapping):
            errors.append(_error("LOCK-WHEEL-RECORD", {}))
            continue
        filename = str(raw.get("filename", ""))
        if Path(filename).name != filename or not filename.casefold().endswith(".whl"):
            errors.append(_error("LOCK-WHEEL-NAME", {"filename": filename}))
            continue
        identity = filename.casefold()
        if identity in expected_names:
            errors.append(_error("LOCK-WHEEL-DUPLICATE", {"filename": filename}))
            continue
        expected_names.add(identity)
        candidate = wheelhouse / filename
        if _is_link_or_reparse(candidate) or not candidate.is_file():
            errors.append(_error("LOCK-WHEEL-MISSING", {"filename": filename}))
            continue
        actual_hash = _sha256_file(candidate)
        actual_size = candidate.stat().st_size
        if actual_hash != raw.get("sha256") or actual_size != raw.get("bytes"):
            errors.append(
                _error(
                    "LOCK-WHEEL-INTEGRITY",
                    {
                        "filename": filename,
                        "expected_sha256": raw.get("sha256"),
                        "actual_sha256": actual_hash,
                        "expected_bytes": raw.get("bytes"),
                        "actual_bytes": actual_size,
                    },
                )
            )
            continue
        verified.append(
            {"filename": filename, "bytes": actual_size, "sha256": actual_hash}
        )
    actual_names = {
        path.name.casefold()
        for path in wheelhouse.glob("*.whl")
        if path.is_file() and not _is_link_or_reparse(path)
    }
    unexpected = sorted(actual_names - expected_names)
    if unexpected:
        errors.append(_error("LOCK-WHEEL-UNEXPECTED", {"filenames": unexpected}))
    _verify_requirements_hashes(requirements_lock, records, errors)
    installed: dict[str, str] | None = None
    if verify_installed:
        installed = (
            collect_installed_distributions()
            if installed_distribution_paths is None
            else collect_installed_distributions(installed_distribution_paths)
        )
        expected = {
            _canonical_name(str(item.get("distribution", ""))): str(
                item.get("version", "")
            )
            for item in records
            if isinstance(item, Mapping)
        }
        tools = toolchain.get("tools")
        if isinstance(tools, Mapping):
            expected["pip"] = str(tools.get("pip", ""))
        missing = sorted(set(expected) - set(installed))
        unexpected = sorted(set(installed) - set(expected))
        mismatched = {
            name: {"expected": expected[name], "actual": installed[name]}
            for name in sorted(set(expected) & set(installed))
            if expected[name] != installed[name]
        }
        if missing or unexpected or mismatched:
            errors.append(
                _error(
                    "LOCK-INSTALLED-DISTRIBUTIONS",
                    {
                        "missing": missing,
                        "unexpected": unexpected,
                        "mismatched": mismatched,
                    },
                )
            )
    return {
        "schema": RELEASE_INPUT_AUDIT_SCHEMA,
        "status": "PASS" if not errors else "FAIL",
        "python": {
            "version": platform.python_version(),
            "architecture": platform.machine(),
            "cache_tag": sys.implementation.cache_tag,
            "sha256": actual_python_hash,
            "base_runtime": {
                "root_name": python_base_root.name,
                "aggregate_sha256": (
                    toolchain.get("python_runtime", {}).get("aggregate_sha256")
                    if isinstance(toolchain.get("python_runtime"), Mapping)
                    else None
                ),
            },
        },
        "wheelhouse": {
            "file_count": len(verified),
            "aggregate_sha256": wheel_lock.get("aggregate_sha256"),
            "files": verified,
        },
        "installed_distributions": installed,
        "errors": errors,
    }


def interpreter_distribution_paths(
    prefix: Path | str | None = None,
) -> tuple[Path, ...]:
    """Return only this interpreter's own site-packages directories.

    Deliberately do not consult ``sys.path``, the current directory, user site,
    or ``sys.base_prefix``.  Release verification must describe the fresh venv
    itself, not editable metadata from the repository running the verifier.
    """

    selected = Path(prefix or sys.prefix).resolve()
    candidates = (
        selected / "Lib" / "site-packages",
        selected
        / "lib"
        / f"python{sys.version_info.major}.{sys.version_info.minor}"
        / "site-packages",
    )
    return tuple(
        path
        for path in dict.fromkeys(candidate.resolve() for candidate in candidates)
        if path.is_dir()
    )


def collect_installed_distributions(
    paths: Sequence[Path | str] | None = None,
) -> dict[str, str]:
    """Return distributions installed in explicit fresh-interpreter roots."""

    selected = (
        interpreter_distribution_paths()
        if paths is None
        else tuple(Path(path).resolve() for path in paths)
    )
    if not selected:
        raise ReleaseLockError("fresh interpreter has no site-packages directory")

    inventory: dict[str, str] = {}
    for distribution in importlib.metadata.distributions(
        path=[str(path) for path in selected]
    ):
        raw_name = distribution.metadata.get("Name")
        if not raw_name:
            continue
        name = _canonical_name(str(raw_name))
        version = str(distribution.version)
        previous = inventory.get(name)
        if previous is not None and previous != version:
            raise ReleaseLockError(
                f"multiple installed versions found for distribution: {name}"
            )
        inventory[name] = version
    return dict(sorted(inventory.items()))


def runtime_inventory_sha256(records: Sequence[Mapping[str, object]]) -> str:
    normalized = sorted(
        (
            {
                "path": str(item.get("path", "")),
                "bytes": int(item.get("bytes", -1)),
                "sha256": str(item.get("sha256", "")),
            }
            for item in records
        ),
        key=lambda item: item["path"].casefold(),
    )
    return hashlib.sha256(
        json.dumps(normalized, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def collect_python_runtime_inventory(base_root: Path | str) -> list[dict[str, object]]:
    """Hash every base-runtime input that can affect venv/PyInstaller output."""

    base = Path(base_root).resolve()
    if not base.is_dir() or _is_link_or_reparse(base):
        raise ReleaseLockError("Python base runtime is unavailable or unsafe")
    selected: list[Path] = []
    for name in PYTHON_RUNTIME_POLICY["root_files"]:
        candidate = base / str(name)
        if candidate.is_file() and not _is_link_or_reparse(candidate):
            selected.append(candidate)
    for name in PYTHON_RUNTIME_POLICY["recursive_roots"]:
        root = base / str(name)
        if not root.is_dir() or _is_link_or_reparse(root):
            raise ReleaseLockError(f"Python runtime inventory root is missing: {name}")
        selected.extend(_runtime_tree_files(base, root))
    records: list[dict[str, object]] = []
    seen: set[str] = set()
    for path in sorted(selected, key=lambda item: item.relative_to(base).as_posix().casefold()):
        relative = path.relative_to(base).as_posix()
        identity = relative.casefold()
        if identity in seen:
            raise ReleaseLockError(f"case-colliding Python runtime path: {relative}")
        seen.add(identity)
        before = path.stat()
        digest = _sha256_file(path)
        after = path.stat()
        if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
            raise ReleaseLockError(f"Python runtime file changed while hashing: {relative}")
        records.append(
            {"path": relative, "bytes": after.st_size, "sha256": digest}
        )
    return records


def create_python_runtime_lock(base_root: Path | str) -> dict[str, object]:
    records = collect_python_runtime_inventory(base_root)
    required = {
        "python.exe",
        "pythonw.exe",
        "python3.dll",
        "python312.dll",
        "Lib/encodings/__init__.py",
        "Lib/encodings/aliases.py",
        "Lib/venv/__init__.py",
    }
    present = {str(item["path"]) for item in records}
    missing = sorted(required - present)
    if missing:
        raise ReleaseLockError(
            "Python runtime inventory is incomplete: " + ", ".join(missing)
        )
    ensurepip_wheels = [
        item for item in records if str(item["path"]).startswith("Lib/ensurepip/_bundled/")
        and str(item["path"]).casefold().endswith(".whl")
    ]
    if not ensurepip_wheels:
        raise ReleaseLockError("Python runtime inventory has no ensurepip wheels")
    return {
        "schema": PYTHON_RUNTIME_LOCK_SCHEMA,
        "policy": PYTHON_RUNTIME_POLICY,
        "file_count": len(records),
        "bytes": sum(int(item["bytes"]) for item in records),
        "aggregate_sha256": runtime_inventory_sha256(records),
        "ensurepip_wheels": ensurepip_wheels,
        "files": records,
    }


def _verify_python_runtime_lock(
    toolchain: Mapping[str, object],
    base_root: Path,
    errors: list[dict[str, object]],
    *,
    runtime_inventory: Mapping[str, object] | None = None,
) -> None:
    raw_runtime = toolchain.get("python_runtime")
    if not isinstance(raw_runtime, Mapping):
        errors.append(_error("LOCK-PYTHON-RUNTIME-MISSING", {}))
        return
    inventory = runtime_inventory if runtime_inventory is not None else raw_runtime
    raw_files = inventory.get("files") if isinstance(inventory, Mapping) else None
    if not isinstance(raw_files, list) or not raw_files:
        errors.append(_error("LOCK-PYTHON-RUNTIME-FILES", {}))
        return
    records: list[dict[str, object]] = []
    seen: set[str] = set()
    for raw in raw_files:
        if not isinstance(raw, Mapping):
            errors.append(_error("LOCK-PYTHON-RUNTIME-RECORD", {}))
            continue
        relative = _safe_runtime_relative(raw.get("path"))
        identity = relative.casefold()
        if identity in seen:
            errors.append(_error("LOCK-PYTHON-RUNTIME-DUPLICATE", {"path": relative}))
            continue
        seen.add(identity)
        expected_size = raw.get("bytes")
        expected_hash = str(raw.get("sha256") or "")
        source = base_root.joinpath(*PurePosixPath(relative).parts)
        actual_size = source.stat().st_size if source.is_file() and not _is_link_or_reparse(source) else None
        actual_hash = _sha256_file(source) if actual_size is not None else None
        if actual_size != expected_size or actual_hash != expected_hash:
            errors.append(
                _error(
                    "LOCK-PYTHON-RUNTIME-INTEGRITY",
                    {
                        "path": relative,
                        "expected_bytes": expected_size,
                        "actual_bytes": actual_size,
                        "expected_sha256": expected_hash,
                        "actual_sha256": actual_hash,
                    },
                )
            )
        records.append(
            {"path": relative, "bytes": expected_size, "sha256": expected_hash}
        )
    if runtime_inventory is not None:
        if runtime_inventory.get("schema") != PYTHON_RUNTIME_LOCK_SCHEMA:
            errors.append(_error("LOCK-PYTHON-RUNTIME-SCHEMA", {}))
        if runtime_inventory.get("policy") != PYTHON_RUNTIME_POLICY:
            errors.append(_error("LOCK-PYTHON-RUNTIME-POLICY", {}))
        try:
            actual_records = collect_python_runtime_inventory(base_root)
        except ReleaseLockError as exc:
            errors.append(
                _error("LOCK-PYTHON-RUNTIME-ENUMERATION", {"message": str(exc)})
            )
            actual_records = []
        declared_paths = {str(item["path"]).casefold(): str(item["path"]) for item in records}
        actual_paths = {
            str(item["path"]).casefold(): str(item["path"]) for item in actual_records
        }
        missing = sorted(
            (declared_paths[key] for key in set(declared_paths) - set(actual_paths)),
            key=str.casefold,
        )
        unexpected = sorted(
            (actual_paths[key] for key in set(actual_paths) - set(declared_paths)),
            key=str.casefold,
        )
        if missing or unexpected:
            errors.append(
                _error(
                    "LOCK-PYTHON-RUNTIME-FILESET",
                    {
                        "missing": missing[:200],
                        "unexpected": unexpected[:200],
                        "missing_count": len(missing),
                        "unexpected_count": len(unexpected),
                    },
                )
            )
    aggregate = runtime_inventory_sha256(records)
    expected_aggregate = (
        inventory.get("aggregate_sha256") if isinstance(inventory, Mapping) else None
    )
    if aggregate != expected_aggregate:
        errors.append(
            _error(
                "LOCK-PYTHON-RUNTIME-AGGREGATE",
                {"expected": expected_aggregate, "actual": aggregate},
            )
        )
    platform_lock = toolchain.get("platform")
    platform_values = platform_lock if isinstance(platform_lock, Mapping) else {}
    version = str(platform_values.get("python_version") or "")
    match = re.fullmatch(r"(\d+)\.(\d+)(?:\.\d+)?", version)
    expected_dll = f"python{match.group(1)}{match.group(2)}.dll" if match else ""
    cache_tag = str(platform_values.get("python_cache_tag") or "")
    if match and cache_tag != f"cpython-{match.group(1)}{match.group(2)}":
        errors.append(_error("LOCK-PYTHON-CACHE-TAG-DERIVED", {"version": version, "cache_tag": cache_tag}))
    python_dll = raw_runtime.get("python_dll")
    dll_record = python_dll if isinstance(python_dll, Mapping) else {}
    matching = next((item for item in records if item["path"].casefold() == expected_dll.casefold()), None)
    if not expected_dll or dict(dll_record) != matching:
        errors.append(
            _error(
                "LOCK-PYTHON-DLL-DERIVED",
                {"expected_path": expected_dll, "declared": dict(dll_record), "inventory": matching},
            )
        )


def _load_python_runtime_inventory(
    project: Path,
    toolchain: Mapping[str, object],
    errors: list[dict[str, object]],
) -> dict[str, object] | None:
    raw_runtime = toolchain.get("python_runtime")
    runtime = raw_runtime if isinstance(raw_runtime, Mapping) else {}
    raw_lock = runtime.get("inventory_lock")
    lock = raw_lock if isinstance(raw_lock, Mapping) else {}
    try:
        relative = _safe_runtime_relative(lock.get("path"))
    except ReleaseLockError:
        errors.append(_error("LOCK-PYTHON-RUNTIME-LOCK-PATH", {}))
        return None
    path = project.joinpath(*PurePosixPath(relative).parts)
    try:
        path.resolve().relative_to(project)
    except ValueError:
        errors.append(_error("LOCK-PYTHON-RUNTIME-LOCK-PATH", {}))
        return None
    actual_hash = _sha256_file(path) if path.is_file() and not _is_link_or_reparse(path) else None
    if actual_hash != lock.get("sha256"):
        errors.append(
            _error(
                "LOCK-PYTHON-RUNTIME-LOCK-HASH",
                {"expected": lock.get("sha256"), "actual": actual_hash},
            )
        )
        return None
    try:
        value = _json(path)
    except ReleaseLockError as exc:
        errors.append(_error("LOCK-PYTHON-RUNTIME-LOCK-READ", {"message": str(exc)}))
        return None
    return value


def _runtime_tree_files(base: Path, root: Path) -> Iterable[Path]:
    excluded_directories = {
        str(value).casefold() for value in PYTHON_RUNTIME_POLICY["excluded_directories"]
    }
    excluded_suffixes = {
        str(value).casefold() for value in PYTHON_RUNTIME_POLICY["excluded_suffixes"]
    }
    for current, directory_names, file_names in os.walk(root, followlinks=False):
        current_path = Path(current)
        if _is_link_or_reparse(current_path):
            raise ReleaseLockError("Python runtime inventory traversed a reparse point")
        kept: list[str] = []
        for name in directory_names:
            candidate = current_path / name
            relative = candidate.relative_to(base).as_posix().casefold()
            if name.casefold() == "__pycache__" or relative in excluded_directories:
                continue
            if _is_link_or_reparse(candidate):
                raise ReleaseLockError(
                    f"Python runtime inventory contains a reparse directory: {relative}"
                )
            kept.append(name)
        directory_names[:] = kept
        for name in file_names:
            candidate = current_path / name
            if candidate.suffix.casefold() in excluded_suffixes:
                continue
            if _is_link_or_reparse(candidate) or not candidate.is_file():
                raise ReleaseLockError(
                    f"Python runtime inventory contains an unsafe file: {name}"
                )
            yield candidate


def _safe_runtime_relative(value: object) -> str:
    if not isinstance(value, str):
        raise ReleaseLockError("unsafe Python runtime lock path")
    raw = value.replace("\\", "/")
    path = PurePosixPath(raw)
    if (
        not raw
        or path.is_absolute()
        or any(part in {"", ".", ".."} for part in path.parts)
        or (path.parts and ":" in path.parts[0])
    ):
        raise ReleaseLockError("unsafe Python runtime lock path")
    return path.as_posix()


def _interpreter_base_prefix(python_executable: Path) -> Path:
    if os.path.normcase(str(python_executable.resolve())) == os.path.normcase(
        str(Path(sys.executable).resolve())
    ):
        return Path(sys.base_prefix).resolve()
    completed = subprocess.run(
        [str(python_executable), "-I", "-c", "import sys; print(sys.base_prefix)"],
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=30,
        check=False,
        text=True,
    )
    if completed.returncode != 0 or not completed.stdout.strip():
        raise ReleaseLockError("could not identify locked Python base runtime")
    return Path(completed.stdout.strip()).resolve()


def _verify_requirements_hashes(
    path: Path,
    records: list[object],
    errors: list[dict[str, object]],
) -> None:
    if not path.is_file():
        return
    text = path.read_text(encoding="utf-8")
    declared_hashes = set(re.findall(r"--hash=sha256:([0-9a-f]{64})", text))
    wheel_hashes = {
        str(item.get("sha256"))
        for item in records
        if isinstance(item, Mapping) and _HEX.fullmatch(str(item.get("sha256", "")))
    }
    if declared_hashes != wheel_hashes:
        errors.append(
            _error(
                "LOCK-REQUIREMENT-HASH-SET",
                {
                    "missing": sorted(wheel_hashes - declared_hashes),
                    "unexpected": sorted(declared_hashes - wheel_hashes),
                },
            )
        )
    requirement_lines = [
        line.strip()
        for line in text.splitlines()
        if line.strip() and not line.lstrip().startswith("#") and not line.startswith(" ")
    ]
    if any("==" not in line or "--hash=sha256:" not in line for line in requirement_lines):
        errors.append(_error("LOCK-UNPINNED-REQUIREMENT", {}))


def _json(path: Path) -> dict[str, object]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ReleaseLockError(f"release lock is unreadable: {path.name}") from exc
    if not isinstance(value, dict):
        raise ReleaseLockError(f"release lock must be a JSON object: {path.name}")
    return value


def _canonical_name(value: str) -> str:
    return re.sub(r"[-_.]+", "-", value).casefold()


def _is_link_or_reparse(path: Path) -> bool:
    try:
        if path.is_symlink():
            return True
        is_junction = getattr(os.path, "isjunction", None)
        if callable(is_junction) and is_junction(path):
            return True
        return bool(
            getattr(path.lstat(), "st_file_attributes", 0)
            & getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400)
        )
    except OSError:
        return False


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _error(code: str, evidence: Mapping[str, object]) -> dict[str, object]:
    return {"code": code, "evidence": dict(evidence)}


__all__ = [
    "RELEASE_INPUT_AUDIT_SCHEMA",
    "ReleaseLockError",
    "collect_installed_distributions",
    "collect_python_runtime_inventory",
    "create_python_runtime_lock",
    "interpreter_distribution_paths",
    "runtime_inventory_sha256",
    "verify_release_inputs",
]
