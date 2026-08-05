from __future__ import annotations

import json
from pathlib import Path
from threading import Lock
from typing import Any

from .config import DEFAULT_BASE_SIZE
from .models import Box


class ConfigFileError(RuntimeError):
    """配置文件无法安全读取或写入时抛出的错误。"""


FORBIDDEN_WINDOWS_FILENAME_CHARS: set[str] = set('<>:"/\\|?*')
RESERVED_WINDOWS_FILENAMES: set[str] = {
    "CON",
    "PRN",
    "AUX",
    "NUL",
    *(f"COM{i}" for i in range(1, 10)),
    *(f"LPT{i}" for i in range(1, 10)),
}

_JSON_LINE_LOCKS: dict[Path, Lock] = {}
_JSON_LINE_LOCKS_GUARD = Lock()


def _json_line_lock(path: Path) -> Lock:
    key = path.resolve()
    with _JSON_LINE_LOCKS_GUARD:
        return _JSON_LINE_LOCKS.setdefault(key, Lock())


def calc_ratio_box(box: Box, base_size: tuple[int, int] = DEFAULT_BASE_SIZE) -> list[float]:
    """把标准化画面中的绝对 ROI 坐标转换为百分比坐标。"""
    base_width, base_height = base_size
    if base_width <= 0 or base_height <= 0:
        raise ValueError("base_size 必须是正数尺寸")
    return [
        round(box.x / base_width, 6),
        round(box.y / base_height, 6),
        round(box.w / base_width, 6),
        round(box.h / base_height, 6),
    ]


def validate_label_name(raw_name: str) -> str:
    """校验模板名/区域名，避免生成 Windows 非法文件名。"""
    name = raw_name.strip()
    if not name:
        raise ValueError("名称不能为空")
    if any(char in FORBIDDEN_WINDOWS_FILENAME_CHARS for char in name):
        raise ValueError('名称不能包含这些字符：< > : " / \\ | ? *')
    if any(ord(char) < 32 for char in name):
        raise ValueError("名称不能包含控制字符")
    if name.endswith("."):
        raise ValueError("名称不能以英文句号结尾")
    if name.upper() in RESERVED_WINDOWS_FILENAMES:
        raise ValueError("名称不能使用 Windows 保留文件名")
    if len(name) > 120:
        raise ValueError("名称过长，请控制在 120 个字符以内")
    return name


def load_json_list(config_path: Path) -> list[dict[str, Any]]:
    """读取 JSON 列表；文件不存在或为空时返回空列表。"""
    if not config_path.exists():
        return []

    text = config_path.read_text(encoding="utf-8")
    if not text.strip():
        return []

    try:
        data = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ConfigFileError(
            f"{config_path.name} 已损坏，已停止写入以避免覆盖原文件：{exc}"
        ) from exc

    if not isinstance(data, list):
        raise ConfigFileError(f"{config_path.name} 的顶层结构必须是 JSON 数组")
    for index, item in enumerate(data):
        if not isinstance(item, dict):
            raise ConfigFileError(f"{config_path.name} 第 {index + 1} 项必须是对象")
    return data


def assert_json_list_readable(config_path: Path) -> None:
    """预检查 JSON，避免先保存图片后才发现配置文件损坏。"""
    load_json_list(config_path)


def append_json_record(config_path: Path, record: dict[str, Any]) -> None:
    """安全追加一条 JSON 记录；如果原文件损坏，绝不覆盖原内容。"""
    config_path.parent.mkdir(parents=True, exist_ok=True)
    # Recognition logs are append-heavy.  Keep their JSON-array contract but
    # avoid decoding and re-encoding the entire history on every frame.
    if config_path.name == "recognition.json" and config_path.exists():
        raw = config_path.read_bytes()
        stripped = raw.rstrip()
        # Generated recognition logs always begin with typed records.  Keep
        # malformed/arbitrary arrays on the safe JSON-decoding path so an
        # invalid file is still rejected instead of being silently appended.
        if (
            stripped.startswith(b"[")
            and stripped.endswith(b"]")
            and b'"type"' in stripped[:4096]
        ):
            body = stripped[:-1].rstrip()
            payload = json.dumps(record, ensure_ascii=False).encode("utf-8")
            separator = b"" if body.endswith(b"[") else bytes((44,))
            temp_path = config_path.with_name(f"{config_path.name}.tmp")
            temp_path.write_bytes(body + separator + b"\n  " + payload + b"\n]\n")
            temp_path.replace(config_path)
            return
    data = load_json_list(config_path)
    data.append(record)
    temp_path = config_path.with_name(f"{config_path.name}.tmp")
    temp_path.write_text(
        json.dumps(data, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temp_path.replace(config_path)


def append_json_line(path: Path, record: dict[str, Any]) -> None:
    """Append one complete UTF-8 JSON object without cross-thread interleaving."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(record, ensure_ascii=False) + "\n"
    with _json_line_lock(path):
        with path.open("a", encoding="utf-8", newline="\n") as handle:
            handle.write(payload)


def load_json_document(config_path: Path, default: Any) -> Any:
    """读取任意 JSON 文档；文件不存在或为空时返回调用方提供的默认值。"""
    if not config_path.exists():
        return default
    text = config_path.read_text(encoding="utf-8")
    if not text.strip():
        return default
    try:
        return json.loads(text)
    except json.JSONDecodeError as exc:
        raise ConfigFileError(
            f"{config_path.name} 已损坏，已停止写入以避免覆盖原文件：{exc}"
        ) from exc


def atomic_write_json(config_path: Path, data: Any) -> None:
    """在同目录写临时文件并原子替换 JSON。"""
    config_path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = config_path.with_name(f".{config_path.name}.tmp")
    temp_path.write_text(
        json.dumps(data, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temp_path.replace(config_path)


def make_template_record(
    name: str,
    box: Box,
    base_size: tuple[int, int] = DEFAULT_BASE_SIZE,
) -> dict[str, Any]:
    """生成模板配置记录。"""
    return {
        "name": name,
        "abs_box": box.to_list(),
        "ratio_box": calc_ratio_box(box, base_size),
    }


def make_region_record(
    name: str,
    source_image: str,
    box: Box,
    base_size: tuple[int, int] = DEFAULT_BASE_SIZE,
) -> dict[str, Any]:
    """生成区域配置记录。"""
    record = make_template_record(name, box, base_size)
    record["source_image"] = source_image
    return record
