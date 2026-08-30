from __future__ import annotations

"""Validation for the offline, hash-locked Windows release inputs."""

import hashlib
import importlib.metadata
import json
import os
import platform
from pathlib import Path
import re
import stat
import sys
from typing import Mapping, Sequence


RELEASE_INPUT_AUDIT_SCHEMA = "guandan.release-input-audit/1"
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
    expected_schema = "guandan.release-toolchain-lock/1"
    if toolchain.get("schema") != expected_schema:
        errors.append(_error("LOCK-TOOLCHAIN-SCHEMA", {"expected": expected_schema}))
    if wheel_lock.get("schema") != "guandan.wheelhouse-lock/1":
        errors.append(_error("LOCK-WHEELHOUSE-SCHEMA", {}))
    locks = toolchain.get("locks") if isinstance(toolchain.get("locks"), Mapping) else {}
    for label, path in (
        ("requirements_input", requirements_input),
        ("requirements_lock", requirements_lock),
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
    "interpreter_distribution_paths",
    "verify_release_inputs",
]
