from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np

from .config import PROFILES_ROOT
from .models import Box
from .storage import atomic_write_json


REGION_DISPLAY_NAMES = {
    "first_play_left": "左侧首出牌提示",
    "first_play_opposite": "对侧首出牌提示",
    "first_play_right": "右侧首出牌提示",
    "first_play_self": "己方首出牌提示",
    "left_play": "左侧出牌区",
    "level_rank": "当前级牌",
    "my_hand": "我的手牌",
    "my_play": "我的出牌区",
    "opposite_play": "对侧出牌区",
    "passed_left": "左侧过牌提示",
    "passed_opposite": "对侧过牌提示",
    "passed_right": "右侧过牌提示",
    "passed_self": "己方过牌提示",
    "right_play": "右侧出牌区",
    "table_anchor_1": "牌桌锚点一",
    "table_anchor_2": "牌桌锚点二",
    "timer_left": "左侧计时器",
    "timer_opposite": "对侧计时器",
    "timer_right": "右侧计时器",
    "timer_self": "己方计时器",
    "button_actions": "按钮区域",
}
ROLE_DISPLAY_NAMES = {
    "hand": "手牌",
    "play": "出牌",
    "anchor": "锚点",
    "generic": "通用区域",
}
ROLE_VALUES = set(ROLE_DISPLAY_NAMES)
IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp"}
BASE_SIZE = (1280, 720)
OVERLAY_COLORS = (
    (40, 220, 255),
    (80, 230, 120),
    (255, 160, 50),
    (220, 80, 220),
    (80, 180, 255),
    (120, 255, 180),
    (255, 100, 100),
    (180, 180, 60),
)


@dataclass(frozen=True)
class RegionRecord:
    name: str
    role: str
    abs_box: Box
    ratio_box: tuple[float, float, float, float]


def display_region_name(name: str) -> str:
    return REGION_DISPLAY_NAMES.get(name, name)


def display_role(role: str) -> str:
    return ROLE_DISPLAY_NAMES.get(role, role)


class AnnotationService:
    """Persistence and overlay operations for configured screenshot regions."""

    def __init__(self, profiles_root: Path = PROFILES_ROOT, profile_name: str = "tencent_daguandan"):
        self.profiles_root = Path(profiles_root)
        self.profile_name = profile_name
        self.profile_root = self.profiles_root / profile_name
        self.regions_path = self.profile_root / "regions_config.json"
        self.screenshots_root = (self.profile_root / "screenshots").resolve()

    @staticmethod
    def _ratio_for_box(box: Box) -> tuple[float, float, float, float]:
        return (
            box.x / BASE_SIZE[0],
            box.y / BASE_SIZE[1],
            box.w / BASE_SIZE[0],
            box.h / BASE_SIZE[1],
        )

    @staticmethod
    def _record_from_json(raw: dict) -> RegionRecord:
        name = str(raw.get("name", "")).strip()
        role = str(raw.get("role", "generic")).strip().lower()
        if not name:
            raise ValueError("区域名称不能为空")
        if name not in REGION_DISPLAY_NAMES:
            raise ValueError(f"区域名称无效：{name}")
        if role not in ROLE_VALUES:
            raise ValueError(f"区域角色无效：{role}")
        box = Box.from_value(raw.get("abs_box"))
        ratio_raw = raw.get("ratio_box")
        ratio = (
            tuple(float(value) for value in ratio_raw)
            if isinstance(ratio_raw, (list, tuple)) and len(ratio_raw) == 4
            else AnnotationService._ratio_for_box(box)
        )
        return RegionRecord(
            name=name,
            role=role,
            abs_box=box,
            ratio_box=ratio,
        )

    def list_regions(self) -> tuple[RegionRecord, ...]:
        if not self.regions_path.is_file():
            return ()
        import json

        document = json.loads(self.regions_path.read_text(encoding="utf-8"))
        if int(document.get("schema_version", 0)) != 2:
            raise ValueError("regions_config.json 版本无效")
        raw_regions = document.get("regions")
        if not isinstance(raw_regions, list):
            raise ValueError("regions_config.json 缺少 regions 数组")
        regions = tuple(self._record_from_json(item) for item in raw_regions)
        if len({region.name for region in regions}) != len(regions):
            raise ValueError("区域名称不能重复")
        return regions

    def list_recorded_images(self) -> tuple[Path, ...]:
        if not self.screenshots_root.is_dir():
            return ()
        paths = (
            path.resolve()
            for path in self.screenshots_root.rglob("*")
            if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
        )
        return tuple(sorted(paths, key=lambda path: path.relative_to(self.screenshots_root).as_posix().lower()))

    def update_region(
        self,
        old_name: str,
        *,
        name: str,
        role: str,
        box: Box,
    ) -> RegionRecord:
        name = str(name).strip()
        role = str(role).strip().lower()
        if name not in REGION_DISPLAY_NAMES:
            raise ValueError(f"区域名称无效：{name}")
        if role not in ROLE_VALUES:
            raise ValueError(f"区域角色无效：{role}")
        if box.x < 0 or box.y < 0 or box.w <= 0 or box.h <= 0:
            raise ValueError("区域坐标必须为非负且宽高大于 0")
        if box.x + box.w > BASE_SIZE[0] or box.y + box.h > BASE_SIZE[1]:
            raise ValueError("区域坐标超出 1280×720 标准化画面")
        regions = list(self.list_regions())
        try:
            index = next(index for index, item in enumerate(regions) if item.name == old_name)
        except StopIteration as exc:
            raise KeyError(f"未找到区域：{old_name}") from exc
        if any(item.name == name and item.name != old_name for item in regions):
            raise ValueError(f"区域名称不能重复：{name}")
        updated = RegionRecord(
            name=name,
            role=role,
            abs_box=box,
            ratio_box=self._ratio_for_box(box),
        )
        regions[index] = updated
        atomic_write_json(
            self.regions_path,
            {
                "schema_version": 2,
                "regions": [
                    {
                        "name": item.name,
                        "role": item.role,
                        "abs_box": item.abs_box.to_list(),
                        "ratio_box": list(item.ratio_box),
                    }
                    for item in regions
                ],
            },
        )
        return updated

    @staticmethod
    def _box_for_image(region: RegionRecord, image: np.ndarray) -> Box:
        height, width = image.shape[:2]
        rx, ry, rw, rh = region.ratio_box
        if rw > 0 and rh > 0:
            return Box(
                round(rx * width),
                round(ry * height),
                max(1, round(rw * width)),
                max(1, round(rh * height)),
            )
        return region.abs_box

    @classmethod
    def overlay_regions(cls, image: np.ndarray, regions: Iterable[RegionRecord]) -> np.ndarray:
        overlay = image.copy()
        for index, region in enumerate(regions):
            box = cls._box_for_image(region, overlay)
            color = OVERLAY_COLORS[index % len(OVERLAY_COLORS)]
            start = (box.x, box.y)
            end = (box.x + box.w, box.y + box.h)
            cv2.rectangle(overlay, start, end, color, 3)
            cv2.putText(
                overlay,
                display_region_name(region.name),
                (box.x, max(20, box.y - 6)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                color,
                2,
                cv2.LINE_AA,
            )
        return overlay
