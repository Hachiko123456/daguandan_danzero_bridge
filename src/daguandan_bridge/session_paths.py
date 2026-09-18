from pathlib import Path
from typing import Mapping
import os
import json

from .storage import atomic_write_json

SESSIONS_ROOT_ENV = "DAGUANDAN_SESSIONS_ROOT"
SESSIONS_ROOT_OVERRIDE_FILENAME = ".sessions_root.json"
SESSIONS_ROOT_SCHEMA = "guandan.sessions-root/1"


def _is_reparse(path: Path) -> bool:
    checker = getattr(path, "is_junction", None)
    return path.is_symlink() or bool(checker()) if callable(checker) else path.is_symlink()


def _assert_safe_path(path: Path) -> Path:
    value = path.expanduser()
    if not value.is_absolute():
        raise ValueError("sessions 根目录必须是绝对路径")
    current = value
    while True:
        if current.exists() and _is_reparse(current):
            raise ValueError(f"sessions 根目录不能经过符号链接或 junction：{current}")
        parent = current.parent
        if parent == current:
            break
        current = parent
    if value.exists() and not value.is_dir():
        raise ValueError(f"sessions 根目录不是目录：{value}")
    return value.resolve(strict=False)


def default_sessions_root(profiles_root: Path | str, profile_name: str) -> Path:
    return Path(profiles_root).expanduser().resolve() / str(profile_name) / "sessions"


def sessions_root_override_path(profiles_root: Path | str, profile_name: str) -> Path:
    return (
        Path(profiles_root).expanduser().resolve()
        / str(profile_name)
        / SESSIONS_ROOT_OVERRIDE_FILENAME
    )


def resolve_sessions_root(
    profiles_root: Path | str,
    profile_name: str,
    *,
    environ: Mapping[str, str] | None = None,
) -> Path:
    """Resolve one profile's session directory without changing old defaults.

    Environment override wins over the persisted pointer. The pointer lives
    beside profile.json, so both source and frozen writable generations can use
    the same contract while templates/models stay in their existing profile.
    """
    values = os.environ if environ is None else environ
    configured = str(values.get(SESSIONS_ROOT_ENV, "") or "").strip()
    if configured:
        return _assert_safe_path(Path(configured))
    pointer = sessions_root_override_path(profiles_root, profile_name)
    if pointer.is_file():
        try:
            payload = json.loads(pointer.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError) as exc:
            raise ValueError(f"sessions 根目录配置不可读：{pointer}") from exc
        if not isinstance(payload, dict) or payload.get("schema") != SESSIONS_ROOT_SCHEMA:
            raise ValueError(f"sessions 根目录配置格式无效：{pointer}")
        root = payload.get("root")
        if not isinstance(root, str) or not root.strip():
            raise ValueError(f"sessions 根目录配置缺少 root：{pointer}")
        return _assert_safe_path(Path(root))
    return default_sessions_root(profiles_root, profile_name)


def write_sessions_root_override(
    profiles_root: Path | str,
    profile_name: str,
    root: Path | str,
    *,
    source_root: Path | str | None = None,
    receipt_path: Path | str | None = None,
) -> Path:
    target = _assert_safe_path(Path(root))
    pointer = sessions_root_override_path(profiles_root, profile_name)
    pointer.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, object] = {
        "schema": SESSIONS_ROOT_SCHEMA,
        "profile": str(profile_name),
        "root": str(target),
        "source_root": str(source_root) if source_root is not None else None,
        "receipt_path": str(receipt_path) if receipt_path is not None else None,
    }
    atomic_write_json(pointer, payload)
    return pointer


def clear_sessions_root_override(profiles_root: Path | str, profile_name: str) -> None:
    sessions_root_override_path(profiles_root, profile_name).unlink(missing_ok=True)


def sessions_root_info(
    profiles_root: Path | str, profile_name: str, *, environ: Mapping[str, str] | None = None
) -> dict[str, object]:
    values = os.environ if environ is None else environ
    configured = str(values.get(SESSIONS_ROOT_ENV, "") or "").strip()
    pointer = sessions_root_override_path(profiles_root, profile_name)
    root = resolve_sessions_root(profiles_root, profile_name, environ=environ)
    return {
        "root": str(root),
        "source": "environment" if configured else "pointer" if pointer.is_file() else "profile_default",
        "profile": str(profile_name),
        "override_path": str(pointer),
    }


__all__ = [
    "SESSIONS_ROOT_ENV",
    "SESSIONS_ROOT_OVERRIDE_FILENAME",
    "SESSIONS_ROOT_SCHEMA",
    "clear_sessions_root_override",
    "default_sessions_root",
    "resolve_sessions_root",
    "sessions_root_info",
    "sessions_root_override_path",
    "write_sessions_root_override",
]
