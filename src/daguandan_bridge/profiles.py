from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .config import (
    DEFAULT_AUTO_CAPTURE_INTERVAL_SEC,
    DEFAULT_BASE_SIZE,
    PICS_DIR_NAME,
    PROFILE_CONFIG_FILE_NAME,
    REGIONS_CONFIG_FILE_NAME,
    SCREENSHOTS_DIR_NAME,
    TEMPLATES_CONFIG_FILE_NAME,
)
from .storage import atomic_write_json


class ProfileNameError(ValueError):
    """profile 名称无法作为安全目录名时抛出的错误。"""


class ProfileConfigError(RuntimeError):
    """profile.json 结构无效时抛出的错误。"""


KNOWN_CHINESE_PROFILE_NAMES: dict[str, str] = {
    "斗地主": "doudizhu",
    "欢乐斗地主": "doudizhu",
    "掼蛋": "guandan",
    "惯蛋": "guandan",
    "腾讯欢乐掼蛋": "guandan",
    "跑得快": "paodekuai",
    "德州扑克": "texas_holdem",
    "德扑": "texas_holdem",
}


@dataclass(frozen=True)
class MatchSettings:
    """模板匹配与兼容性检查参数。"""

    scales: tuple[float, ...] = (0.9, 0.95, 1.0, 1.05, 1.1)
    min_confidence: float = 0.78
    min_margin: float = 0.06
    channel_weights: tuple[float, float, float] = (0.45, 0.30, 0.25)
    stable_frames: int = 3
    anchor_required: bool = True
    anchor_search_padding_ratio: float = 0.08

    def normalized(self) -> "MatchSettings":
        scales = tuple(float(value) for value in self.scales)
        if not scales or any(
            not math.isfinite(value) or value <= 0 or value > 4
            for value in scales
        ):
            raise ProfileConfigError("match_settings.scales 必须是非空的正数列表")
        min_confidence = float(self.min_confidence)
        min_margin = float(self.min_margin)
        if not math.isfinite(min_confidence) or not 0 <= min_confidence <= 1:
            raise ProfileConfigError("match_settings.min_confidence 必须在 0 到 1 之间")
        if not math.isfinite(min_margin) or not 0 <= min_margin <= 1:
            raise ProfileConfigError("match_settings.min_margin 必须在 0 到 1 之间")
        if len(self.channel_weights) != 3:
            raise ProfileConfigError("match_settings.channel_weights 必须包含 3 个权重")
        weights = tuple(float(value) for value in self.channel_weights)
        if (
            any(not math.isfinite(value) or value < 0 for value in weights)
            or sum(weights) <= 0
        ):
            raise ProfileConfigError("match_settings.channel_weights 必须是非负数且总和大于 0")
        if int(self.stable_frames) < 1:
            raise ProfileConfigError("match_settings.stable_frames 必须大于等于 1")
        padding_ratio = float(self.anchor_search_padding_ratio)
        if not math.isfinite(padding_ratio) or not 0 <= padding_ratio <= 1:
            raise ProfileConfigError(
                "match_settings.anchor_search_padding_ratio 必须在 0 到 1 之间"
            )

        weight_sum = sum(weights)
        return MatchSettings(
            scales=scales,
            min_confidence=min_confidence,
            min_margin=min_margin,
            channel_weights=tuple(value / weight_sum for value in weights),
            stable_frames=int(self.stable_frames),
            anchor_required=bool(self.anchor_required),
            anchor_search_padding_ratio=padding_ratio,
        )

    def to_json_dict(self) -> dict[str, Any]:
        normalized = self.normalized()
        return {
            "scales": list(normalized.scales),
            "min_confidence": normalized.min_confidence,
            "min_margin": normalized.min_margin,
            "channel_weights": list(normalized.channel_weights),
            "stable_frames": normalized.stable_frames,
            "anchor_required": normalized.anchor_required,
            "anchor_search_padding_ratio": normalized.anchor_search_padding_ratio,
        }


@dataclass(frozen=True)
class CounterSettings:
    """牌组数量和需要计数的区域类型。"""

    deck_copies: int = 1
    include_jokers: bool = True
    count_region_roles: tuple[str, ...] = ("play",)

    def normalized(self) -> "CounterSettings":
        deck_copies = int(self.deck_copies)
        if deck_copies < 1 or deck_copies > 16:
            raise ProfileConfigError("counter_settings.deck_copies 必须在 1 到 16 之间")
        roles = tuple(str(role).strip().lower() for role in self.count_region_roles if str(role).strip())
        allowed_roles = {"hand", "play", "generic"}
        if not roles or any(role not in allowed_roles for role in roles):
            raise ProfileConfigError(
                "counter_settings.count_region_roles 只能包含 hand、play、generic"
            )
        return CounterSettings(
            deck_copies=deck_copies,
            include_jokers=bool(self.include_jokers),
            count_region_roles=roles,
        )

    def to_json_dict(self) -> dict[str, Any]:
        normalized = self.normalized()
        return {
            "deck_copies": normalized.deck_copies,
            "include_jokers": normalized.include_jokers,
            "count_region_roles": list(normalized.count_region_roles),
        }


@dataclass(frozen=True)
class ProfileConfig:
    """一个卡牌游戏或窗口环境的独立配置。"""

    name: str
    display_name: str
    window_title_keywords: tuple[str, ...]
    base_size: tuple[int, int] = DEFAULT_BASE_SIZE
    auto_capture_interval_sec: float = DEFAULT_AUTO_CAPTURE_INTERVAL_SEC
    schema_version: int = 2
    aspect_ratio_tolerance: float = 0.03
    capture_backend: str = "auto"
    allow_screen_fallback: bool = True
    detect_black_bars: bool = True
    viewport_mode: str = "full"
    viewport_aspect_ratio: float = 16 / 9
    target_client_size: tuple[int, int] | None = None
    match_settings: MatchSettings = field(default_factory=MatchSettings)
    counter_settings: CounterSettings = field(default_factory=CounterSettings)
    advisor_strategy: str = "fabledan"

    def normalized(self) -> "ProfileConfig":
        base_size = (int(self.base_size[0]), int(self.base_size[1]))
        if base_size[0] <= 0 or base_size[1] <= 0:
            raise ProfileConfigError("base_size 必须包含两个正整数")
        interval = float(self.auto_capture_interval_sec)
        if not math.isfinite(interval) or interval <= 0:
            raise ProfileConfigError("auto_capture_interval_sec 必须大于 0")
        tolerance = float(self.aspect_ratio_tolerance)
        if not math.isfinite(tolerance) or tolerance < 0 or tolerance > 0.25:
            raise ProfileConfigError("aspect_ratio_tolerance 必须在 0 到 0.25 之间")
        backend = str(self.capture_backend).strip().lower()
        if backend not in {"auto", "printwindow", "screen", "gdi_screen"}:
            raise ProfileConfigError(
                "capture_backend 只能是 auto、printwindow、screen 或 gdi_screen"
            )
        viewport_mode = str(self.viewport_mode).strip().lower()
        if viewport_mode not in {"full", "bottom_aspect"}:
            raise ProfileConfigError(
                "viewport_mode must be full or bottom_aspect"
            )
        viewport_aspect_ratio = float(self.viewport_aspect_ratio)
        if (
            not math.isfinite(viewport_aspect_ratio)
            or viewport_aspect_ratio <= 0
            or viewport_aspect_ratio > 10
        ):
            raise ProfileConfigError(
                "viewport_aspect_ratio must be a finite positive number"
            )
        target_client_size = None
        if self.target_client_size is not None:
            target_client_size = (
                int(self.target_client_size[0]),
                int(self.target_client_size[1]),
            )
            if target_client_size[0] <= 0 or target_client_size[1] <= 0:
                raise ProfileConfigError("target_client_size 必须包含两个正整数")
        keywords = tuple(
            keyword.strip() for keyword in self.window_title_keywords if keyword.strip()
        )
        if not keywords:
            raise ProfileConfigError("window_title_keywords 至少需要一个窗口标题关键字")
        advisor_strategy = str(self.advisor_strategy).strip().lower()
        if advisor_strategy not in {"danzero", "fabledan"}:
            raise ProfileConfigError("advisor_strategy 只能是 danzero 或 fabledan")

        return ProfileConfig(
            name=normalize_profile_name(self.name),
            display_name=self.display_name.strip() or normalize_profile_name(self.name),
            window_title_keywords=keywords,
            base_size=base_size,
            auto_capture_interval_sec=interval,
            schema_version=2,
            aspect_ratio_tolerance=tolerance,
            capture_backend=backend,
            allow_screen_fallback=bool(self.allow_screen_fallback),
            detect_black_bars=bool(self.detect_black_bars),
            viewport_mode=viewport_mode,
            viewport_aspect_ratio=viewport_aspect_ratio,
            target_client_size=target_client_size,
            match_settings=self.match_settings.normalized(),
            counter_settings=self.counter_settings.normalized(),
            advisor_strategy=advisor_strategy,
        )

    def to_json_dict(self) -> dict[str, Any]:
        normalized = self.normalized()
        return {
            "name": normalized.name,
            "display_name": normalized.display_name,
            "window_title_keywords": list(normalized.window_title_keywords),
            "base_size": list(normalized.base_size),
            "auto_capture_interval_sec": normalized.auto_capture_interval_sec,
            "schema_version": normalized.schema_version,
            "aspect_ratio_tolerance": normalized.aspect_ratio_tolerance,
            "capture_backend": normalized.capture_backend,
            "allow_screen_fallback": normalized.allow_screen_fallback,
            "detect_black_bars": normalized.detect_black_bars,
            "viewport_mode": normalized.viewport_mode,
            "viewport_aspect_ratio": normalized.viewport_aspect_ratio,
            "target_client_size": (
                list(normalized.target_client_size)
                if normalized.target_client_size is not None
                else None
            ),
            "match_settings": normalized.match_settings.to_json_dict(),
            "counter_settings": normalized.counter_settings.to_json_dict(),
            "advisor_strategy": normalized.advisor_strategy,
        }


@dataclass(frozen=True)
class ProfilePaths:
    """一个 profile 下所有托管路径。"""

    name: str
    root: Path
    screenshots_dir: Path
    pics_dir: Path
    templates_dir: Path
    profile_config_path: Path
    templates_config_path: Path
    regions_config_path: Path

    def ensure_dirs(self) -> None:
        self.root.mkdir(parents=True, exist_ok=True)
        self.screenshots_dir.mkdir(parents=True, exist_ok=True)
        self.pics_dir.mkdir(parents=True, exist_ok=True)
        self.templates_dir.mkdir(parents=True, exist_ok=True)


def normalize_profile_name(raw_name: str) -> str:
    """把用户输入转换为安全、稳定的 profile 目录名。"""
    stripped = raw_name.strip()
    if not stripped:
        raise ProfileNameError("profile 名称不能为空")

    if stripped in KNOWN_CHINESE_PROFILE_NAMES:
        return KNOWN_CHINESE_PROFILE_NAMES[stripped]

    lowered = stripped.lower().replace("-", "_").replace(" ", "_")
    lowered = re.sub(r"_+", "_", lowered)
    if not re.fullmatch(r"[a-z0-9_]+", lowered):
        raise ProfileNameError(
            "profile 名称请使用英文、数字、下划线，常见中文名如“斗地主”“掼蛋”会自动转换"
        )
    if lowered in {".", ".."} or lowered.startswith("_") or lowered.endswith("_"):
        raise ProfileNameError("profile 名称不能以特殊分隔符开头或结尾")
    return lowered


def get_profile_paths(data_root: Path, raw_name: str) -> ProfilePaths:
    """返回托管式 profile 路径，不允许任意目录逃逸。"""
    name = normalize_profile_name(raw_name)
    root = data_root / name
    return ProfilePaths(
        name=name,
        root=root,
        screenshots_dir=root / SCREENSHOTS_DIR_NAME,
        pics_dir=root / PICS_DIR_NAME,
        templates_dir=root / "templates",
        profile_config_path=root / PROFILE_CONFIG_FILE_NAME,
        templates_config_path=root / TEMPLATES_CONFIG_FILE_NAME,
        regions_config_path=root / REGIONS_CONFIG_FILE_NAME,
    )


def create_profile(data_root: Path, config: ProfileConfig) -> ProfilePaths:
    """创建或更新 profile.json，并确保目录结构存在。"""
    normalized = config.normalized()
    if not normalized.window_title_keywords:
        raise ProfileConfigError("window_title_keywords 至少需要一个窗口标题关键字")

    paths = get_profile_paths(data_root, normalized.name)
    paths.ensure_dirs()
    atomic_write_json(paths.profile_config_path, normalized.to_json_dict())
    return paths


def load_profile_config(paths: ProfilePaths) -> ProfileConfig:
    """读取 profile.json。"""
    if not paths.profile_config_path.exists():
        raise ProfileConfigError(f"缺少配置文件：{paths.profile_config_path}")

    try:
        data = json.loads(paths.profile_config_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ProfileConfigError(f"profile.json 已损坏：{exc}") from exc

    if not isinstance(data, dict):
        raise ProfileConfigError("profile.json 顶层结构必须是 JSON 对象")

    try:
        schema_version = int(data.get("schema_version", 1))
        if schema_version not in {1, 2}:
            raise ProfileConfigError(
                f"不支持的 profile schema_version：{schema_version}"
            )
        base_size_raw = data.get("base_size", list(DEFAULT_BASE_SIZE))
        target_client_size_raw = data.get("target_client_size")
        match_raw = data.get("match_settings", {})
        counter_raw = data.get("counter_settings", {})
        if not isinstance(match_raw, dict) or not isinstance(counter_raw, dict):
            raise TypeError("settings 必须是 JSON 对象")
        return ProfileConfig(
            name=str(data["name"]),
            display_name=str(data.get("display_name") or data["name"]),
            window_title_keywords=tuple(str(item) for item in data["window_title_keywords"]),
            base_size=(int(base_size_raw[0]), int(base_size_raw[1])),
            auto_capture_interval_sec=float(
                data.get("auto_capture_interval_sec", DEFAULT_AUTO_CAPTURE_INTERVAL_SEC)
            ),
            schema_version=schema_version,
            aspect_ratio_tolerance=float(data.get("aspect_ratio_tolerance", 0.03)),
            capture_backend=str(data.get("capture_backend", "auto")),
            allow_screen_fallback=bool(data.get("allow_screen_fallback", True)),
            detect_black_bars=bool(data.get("detect_black_bars", True)),
            viewport_mode=str(data.get("viewport_mode", "full")),
            viewport_aspect_ratio=float(
                data.get("viewport_aspect_ratio", 16 / 9)
            ),
            target_client_size=(
                int(target_client_size_raw[0]),
                int(target_client_size_raw[1]),
            )
            if target_client_size_raw is not None
            else None,
            match_settings=MatchSettings(
                scales=tuple(float(item) for item in match_raw.get("scales", (0.9, 0.95, 1.0, 1.05, 1.1))),
                min_confidence=float(match_raw.get("min_confidence", 0.78)),
                min_margin=float(match_raw.get("min_margin", 0.06)),
                channel_weights=tuple(
                    float(item)
                    for item in match_raw.get("channel_weights", (0.45, 0.30, 0.25))
                ),
                stable_frames=int(match_raw.get("stable_frames", 3)),
                anchor_required=bool(match_raw.get("anchor_required", True)),
                anchor_search_padding_ratio=float(
                    match_raw.get("anchor_search_padding_ratio", 0.08)
                ),
            ),
            counter_settings=CounterSettings(
                deck_copies=int(counter_raw.get("deck_copies", 1)),
                include_jokers=bool(counter_raw.get("include_jokers", True)),
                count_region_roles=tuple(
                    str(item) for item in counter_raw.get("count_region_roles", ("play",))
                ),
            ),
            advisor_strategy=str(data.get("advisor_strategy", "danzero")),
        ).normalized()
    except (KeyError, TypeError, ValueError, IndexError) as exc:
        raise ProfileConfigError("profile.json 字段无效") from exc


def list_profiles(data_root: Path) -> list[str]:
    """列出已创建的 profile 名称。"""
    if not data_root.exists():
        return []
    return sorted(
        path.name
        for path in data_root.iterdir()
        if path.is_dir() and (path / PROFILE_CONFIG_FILE_NAME).exists()
    )


def ensure_default_profile(data_root: Path) -> ProfilePaths:
    """首次运行时创建一个通用 default profile。"""
    existing = list_profiles(data_root)
    if existing:
        return get_profile_paths(data_root, existing[0])

    return create_profile(
        data_root,
        ProfileConfig(
            name="default",
            display_name="通用扑克游戏",
            window_title_keywords=("微信", "腾讯欢乐掼蛋", "斗地主"),
            base_size=DEFAULT_BASE_SIZE,
            auto_capture_interval_sec=DEFAULT_AUTO_CAPTURE_INTERVAL_SEC,
        ),
    )
