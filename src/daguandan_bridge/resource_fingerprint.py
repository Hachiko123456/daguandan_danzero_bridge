from __future__ import annotations

"""Canonical recognition-resource identity shared by capture and replay."""

import hashlib
import json
from pathlib import Path


_CONFIG_FILES = frozenset(
    {"profile.json", "regions_config.json", "templates_config.json"}
)
_RESOURCE_DIRS = frozenset({"templates", "models"})
_FINGERPRINT_ALGORITHM = "recognition-resources-v2"


def recognition_resource_identity(
    profiles_root: Path,
    profile_name: str,
) -> dict[str, object]:
    """Hash the exact recognition configuration with order-independent discovery."""

    root = Path(profiles_root) / str(profile_name)
    try:
        files = sorted(
            (
                path
                for path in root.rglob("*")
                if path.is_file()
                and (
                    path.name in _CONFIG_FILES
                    or bool(_RESOURCE_DIRS.intersection(path.relative_to(root).parts))
                )
            ),
            key=lambda path: path.relative_to(root).as_posix(),
        )
        records = [
            {
                "path": path.relative_to(root).as_posix(),
                "bytes": path.stat().st_size,
                "sha256": _sha256_file(path),
            }
            for path in files
        ]
        if not records:
            return {
                "status": "unavailable",
                "algorithm": _FINGERPRINT_ALGORITHM,
                "profile_name": str(profile_name),
                "files": [],
            }
        payload = json.dumps(
            records,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
        return {
            "status": "identified",
            "algorithm": _FINGERPRINT_ALGORITHM,
            "profile_name": str(profile_name),
            "sha256": hashlib.sha256(payload).hexdigest(),
            "files": records,
        }
    except (OSError, RuntimeError, ValueError) as exc:
        return {
            "status": "unavailable",
            "algorithm": _FINGERPRINT_ALGORITHM,
            "profile_name": str(profile_name),
            "error_type": type(exc).__name__,
            "files": [],
        }


def compare_resource_identities(
    expected: object,
    actual: object,
) -> dict[str, object]:
    """Compare capture/replay resource identities with explainable details."""

    expected_map = expected if isinstance(expected, dict) else {}
    actual_map = actual if isinstance(actual, dict) else {}
    expected_sha = expected_map.get("sha256")
    actual_sha = actual_map.get("sha256")
    if isinstance(expected_sha, str) and isinstance(actual_sha, str):
        matched = expected_sha == actual_sha
        status = "match" if matched else "mismatch"
    else:
        matched = None
        status = "unavailable"

    def file_map(value: dict[str, object]) -> dict[str, str]:
        result: dict[str, str] = {}
        files = value.get("files")
        if isinstance(files, list):
            for item in files:
                if isinstance(item, dict):
                    path = item.get("path")
                    sha = item.get("sha256")
                    if isinstance(path, str) and isinstance(sha, str):
                        result[path] = sha
        return result

    expected_files = file_map(expected_map)
    actual_files = file_map(actual_map)
    return {
        "status": status,
        "matched": matched,
        "expected_sha256": expected_sha,
        "actual_sha256": actual_sha,
        "missing_files": sorted(set(expected_files) - set(actual_files)),
        "unexpected_files": sorted(set(actual_files) - set(expected_files)),
        "changed_files": sorted(
            path
            for path in set(expected_files).intersection(actual_files)
            if expected_files[path] != actual_files[path]
        ),
        "expected_algorithm": expected_map.get("algorithm"),
        "actual_algorithm": actual_map.get("algorithm"),
    }


def resource_identities_match(expected: object, actual: object) -> bool | None:
    """Return True/False when comparable, otherwise None for legacy data."""

    result = compare_resource_identities(expected, actual)
    value = result["matched"]
    return value if isinstance(value, bool) else None


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


__all__ = [
    "compare_resource_identities",
    "recognition_resource_identity",
    "resource_identities_match",
]
