from __future__ import annotations

"""Canonical recognition-resource identity shared by capture and replay."""

import hashlib
import json
from pathlib import Path


_CONFIG_FILES = frozenset(
    {"profile.json", "regions_config.json", "templates_config.json"}
)


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
                    or "templates" in path.relative_to(root).parts
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
            return {"status": "unavailable", "files": []}
        payload = json.dumps(
            records,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
        return {
            "status": "identified",
            "sha256": hashlib.sha256(payload).hexdigest(),
            "files": records,
        }
    except (OSError, RuntimeError, ValueError) as exc:
        return {
            "status": "unavailable",
            "error_type": type(exc).__name__,
            "files": [],
        }


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


__all__ = ["recognition_resource_identity"]
