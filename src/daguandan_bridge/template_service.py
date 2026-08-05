from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Iterable

import cv2
import numpy as np

from .config import PROFILES_ROOT
from .image_io import save_image_unicode
from .models import Box
from .storage import atomic_write_json


TEMPLATE_KINDS = {"rank", "suit", "anchor", "button", "status", "effect", "timer"}
SOURCE_ROLES = {"hand", "hand_partial", "play", "generic", "level"}
RANK_LABELS = {"A", "2", "3", "4", "5", "6", "7", "8", "9", "10", "J", "Q", "K", "small_joker", "big_joker"}
RESERVED_LABELS = {"CON", "PRN", "AUX", "NUL", *(f"COM{index}" for index in range(1, 10)), *(f"LPT{index}" for index in range(1, 10))}
SUIT_ALIASES = {
    "s": "spade", "spade": "spade", "黑桃": "spade",
    "h": "heart", "heart": "heart", "红桃": "heart",
    "c": "club", "club": "club", "梅花": "club",
    "d": "diamond", "diamond": "diamond", "方块": "diamond",
}


@dataclass(frozen=True)
class TemplateSample:
    sample_id: str
    kind: str
    label: str
    file: str
    source_image: str
    source_role: str
    abs_box: Box
    ratio_box: tuple[float, float, float, float]
    base_size: tuple[int, int]


class TemplateService:
    def __init__(self, profiles_root: Path = PROFILES_ROOT, profile_name: str = "tencent_daguandan"):
        self.profiles_root = Path(profiles_root)
        self.profile_name = profile_name
        self.profile_root = self.profiles_root / profile_name
        self.templates_root = self.profile_root / "templates"
        self.templates_path = self.profile_root / "templates_config.json"
        self.base_size = (1280, 720)

    def list_templates(self) -> tuple[dict[str, Any], ...]:
        if not self.templates_path.is_file():
            return ()
        document = json.loads(self.templates_path.read_text(encoding="utf-8"))
        raw_templates = document.get("templates", [])
        if not isinstance(raw_templates, list):
            raise ValueError("templates_config.json 缺少 templates 数组")
        return tuple(item for item in raw_templates if isinstance(item, dict))

    def delete_templates(self, sample_ids: Iterable[str]) -> tuple[dict[str, Any], ...]:
        requested = tuple(str(sample_id).strip() for sample_id in sample_ids)
        if not requested or any(not sample_id for sample_id in requested):
            raise ValueError("请至少选择一个模板")
        if len(set(requested)) != len(requested):
            raise ValueError("删除列表中包含重复模板")
        records = list(self.list_templates())
        by_id = {str(record.get("sample_id", "")): record for record in records}
        missing = next((sample_id for sample_id in requested if sample_id not in by_id), None)
        if missing is not None:
            raise KeyError(f"找不到模板样本：{missing}")
        root = self.templates_root.resolve()
        selected = tuple(by_id[sample_id] for sample_id in requested)
        paths: list[Path] = []
        for record in selected:
            relative = Path(str(record.get("file", "")))
            path = (self.profile_root / relative).resolve()
            if path != root and root not in path.parents:
                raise ValueError("模板文件路径超出 templates 目录")
            if not path.is_file():
                raise FileNotFoundError(f"模板图片不存在：{relative}")
            paths.append(path)
        remaining = [record for record in records if str(record.get("sample_id", "")) not in requested]
        moved: list[tuple[Path, Path]] = []
        try:
            for path in paths:
                tombstone = path.with_name(f".{path.name}.deleting")
                path.replace(tombstone)
                moved.append((path, tombstone))
            atomic_write_json(self.templates_path, {"schema_version": 2, "templates": remaining})
        except Exception:
            for original, tombstone in reversed(moved):
                if tombstone.exists() and not original.exists():
                    tombstone.replace(original)
            raise
        for _original, tombstone in moved:
            tombstone.unlink(missing_ok=True)
        return selected

    def delete_template(self, sample_id: str) -> dict[str, Any]:
        return self.delete_templates((sample_id,))[0]

    @staticmethod
    def _identity(kind: str, label: str) -> tuple[str, str]:
        normalized_kind = str(kind).strip().lower()
        if normalized_kind not in TEMPLATE_KINDS:
            raise ValueError("模板类型只能是 rank、suit、anchor、button、status、effect 或 timer")
        raw = str(label).strip()
        if not raw or any(char in raw for char in "\\/:*?\"<>|"):
            raise ValueError("模板标签不能为空或包含路径字符")
        if any(ord(char) < 32 for char in raw) or raw.endswith(".") or len(raw) > 120 or raw.upper() in RESERVED_LABELS:
            raise ValueError("模板标签包含非法字符或长度超限")
        if normalized_kind == "rank":
            value = raw.upper()
            if value in {"SMALL_JOKER", "BIG_JOKER"}:
                value = value.lower()
            if value not in RANK_LABELS:
                raise ValueError("点数只能是 A、2-10、J、Q、K、small_joker 或 big_joker")
            return normalized_kind, value
        if normalized_kind == "suit":
            value = SUIT_ALIASES.get(raw.lower())
            if value is None:
                raise ValueError("花色只能是 spade、heart、club 或 diamond")
            return normalized_kind, value
        if normalized_kind == "timer" and raw.lower() != "active":
            raise ValueError("timer 模板标签必须是 active")
        return normalized_kind, raw

    @staticmethod
    def _select_components(labels: np.ndarray, stats: np.ndarray, indexes: list[int], kind: str, label: str) -> np.ndarray:
        normalized_label = str(label).strip().lower()
        ordered = sorted(indexes, key=lambda index: int(stats[index, cv2.CC_STAT_AREA]), reverse=True)
        if kind == "rank" and normalized_label in {"small_joker", "big_joker"}:
            selected_indexes = ordered
        elif kind == "rank" and normalized_label == "10":
            roi_height, roi_width = labels.shape[:2]
            interior = []
            for index in ordered:
                x, y, width, height = (int(stats[index, offset]) for offset in range(4))
                touched_edges = sum((x <= 0, y <= 0, x + width >= roi_width, y + height >= roi_height))
                if touched_edges < 3:
                    interior.append(index)
            candidates = interior or ordered
            base = max(candidates, key=lambda index: int(stats[index, cv2.CC_STAT_AREA]))
            base_y, base_height = int(stats[base, 1]), int(stats[base, 3])
            selected_indexes = []
            for index in candidates:
                y, height = int(stats[index, 1]), int(stats[index, 3])
                overlap = max(0, min(base_y + base_height, y + height) - max(base_y, y))
                if height >= max(3, int(round(base_height * 0.65))) and overlap >= max(2, int(round(min(base_height, height) * 0.55))):
                    selected_indexes.append(index)
            if not selected_indexes:
                selected_indexes = [base]
        else:
            selected_indexes = ordered[:1]
        selected = np.zeros(labels.shape, dtype=np.uint8)
        for index in selected_indexes:
            selected[labels == index] = 1
        return selected

    @staticmethod
    def _prepare_crop(image: np.ndarray, kind: str, label: str, box: Box) -> tuple[Box, np.ndarray]:
        image_height, image_width = image.shape[:2]
        if not box.fits_within((int(image_width), int(image_height))):
            raise ValueError("模板坐标超出实际图片")
        crop = image[box.y : box.y + box.h, box.x : box.x + box.w]
        if crop.size == 0 or min(crop.shape[:2]) < 5:
            raise ValueError("模板框太小，请完整框住数字或花色")
        if kind in {"anchor", "button", "status", "effect", "timer"} or (kind == "rank" and label in {"small_joker", "big_joker"}):
            return box, crop.copy()
        if kind not in {"rank", "suit"}:
            raise ValueError("template kind must be rank, suit, anchor, button, status, effect, or timer")
        gray = crop.astype(np.uint8) if crop.ndim == 2 else cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        border_width = max(1, min(3, min(gray.shape[:2]) // 8))
        border = np.concatenate((gray[:border_width, :].reshape(-1), gray[-border_width:, :].reshape(-1), gray[:, :border_width].reshape(-1), gray[:, -border_width:].reshape(-1)))
        background_level = float(np.percentile(border, 75))
        if background_level < 150:
            background_level = float(np.percentile(gray, 90))
        threshold = max(22.0, background_level * 0.11)
        foreground = gray.astype(np.float32) < background_level - threshold
        count, component_labels, stats, _ = cv2.connectedComponentsWithStats(foreground.astype(np.uint8), connectivity=8)
        minimum_component_area = max(3, int(round(box.w * box.h * 0.0008)))
        component_indexes = [index for index in range(1, count) if int(stats[index, cv2.CC_STAT_AREA]) >= minimum_component_area]
        if not component_indexes:
            raise ValueError("框内没有检测到清晰的数字或花色")
        selected = TemplateService._select_components(component_labels, stats, component_indexes, kind, label)
        if float(selected.mean()) > 0.65:
            raise ValueError("框内非背景内容过多，请只框住一个数字或一个花色")
        ys, xs = np.where(selected > 0)
        left, top, right, bottom = int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1
        content_width, content_height = right - left, bottom - top
        if content_width < 2 or content_height < 3:
            raise ValueError("检测到的模板内容太小，请重新框选")
        padding = max(3, int(round(max(content_width, content_height) * 0.10)))
        content_left, content_top = box.x + left, box.y + top
        content_right, content_bottom = box.x + right, box.y + bottom
        if content_left <= 0 or content_top <= 0 or content_right >= image_width or content_bottom >= image_height:
            raise ValueError("检测到的数字或花色触碰截图边缘，可能已经被截断")
        template_width, template_height = content_width + 2 * padding, content_height + 2 * padding
        absolute_left = min(max(0, content_left - padding), image_width - template_width)
        absolute_top = min(max(0, content_top - padding), image_height - template_height)
        normalized = Box(absolute_left, absolute_top, template_width, template_height)
        prepared = np.full((normalized.h, normalized.w, 3), 255, dtype=np.uint8)
        content_mask = selected[top:bottom, left:right] > 0
        prepared[padding : padding + content_height, padding : padding + content_width][content_mask] = 0
        return normalized, prepared

    def save_template(self, image: np.ndarray, *, kind: str, label: str, box: Box, source_image: str = "", source_role: str = "generic") -> TemplateSample:
        normalized_kind, normalized_label = self._identity(kind, label)
        source_role = str(source_role).strip().lower()
        if source_role not in SOURCE_ROLES:
            raise ValueError("模板来源只能是 hand、hand_partial、play、generic 或 level")
        if not box.fits_within(self.base_size):
            raise ValueError("模板坐标超出 1280×720 标准化画面")
        normalized_box, prepared = self._prepare_crop(image, normalized_kind, normalized_label, box)
        records = list(self.list_templates())
        existing_ids = {str(record.get("sample_id", "")) for record in records}
        stem = normalized_label if normalized_kind not in {"rank", "suit"} else f"{normalized_label}_{source_role}"
        index = 1
        while f"{normalized_kind}_{normalized_label.lower()}_{index:03d}" in existing_ids:
            index += 1
        sample_id = f"{normalized_kind}_{normalized_label.lower()}_{index:03d}"
        destination = self.templates_root / normalized_kind / f"{stem}.png"
        filename_index = 1
        while destination.exists():
            filename_index += 1
            destination = self.templates_root / normalized_kind / f"{stem}_{filename_index:03d}.png"
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = destination.with_name(f".{sample_id}.tmp.png")
        ratio = tuple(round(value, 6) for value in (
            normalized_box.x / 1280,
            normalized_box.y / 720,
            normalized_box.w / 1280,
            normalized_box.h / 720,
        ))
        record = {
            "sample_id": sample_id, "kind": normalized_kind, "label": normalized_label,
            "file": destination.relative_to(self.profile_root).as_posix(), "source_image": str(source_image).replace("\\", "/"),
            "source_role": source_role, "abs_box": normalized_box.to_list(), "ratio_box": list(ratio), "base_size": [1280, 720],
        }
        records.append(record)
        try:
            save_image_unicode(temporary, prepared)
            temporary.replace(destination)
            atomic_write_json(self.templates_path, {"schema_version": 2, "templates": records})
        except Exception:
            temporary.unlink(missing_ok=True)
            destination.unlink(missing_ok=True)
            raise
        return TemplateSample(sample_id, normalized_kind, normalized_label, record["file"], record["source_image"], source_role, normalized_box, ratio, self.base_size)
