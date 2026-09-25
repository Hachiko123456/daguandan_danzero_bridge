from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping

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


@dataclass(frozen=True)
class ProfileValidationIssue:
    """一个可序列化、可展示的 profile/ROI 验证问题。"""

    code: str
    severity: str
    message: str
    region: str | None = None
    related_region: str | None = None
    details: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "code": self.code,
            "severity": self.severity,
            "message": self.message,
            "region": self.region,
            "related_region": self.related_region,
            "details": dict(self.details),
        }


# These are the only regions whose overlap can misattribute one player's
# action to another player.  Other ROI overlaps are allowed because status,
# animation and decorative regions often intentionally touch play areas.
KEY_PLAY_REGION_NAMES = ("left_play", "opposite_play", "right_play", "my_play")
CRITICAL_PLAY_OVERLAP_RATIO = 0.15


def _region_value(region: object, name: str, default: object = None) -> object:
    if isinstance(region, Mapping):
        return region.get(name, default)
    return getattr(region, name, default)


def _box_value(region: object) -> tuple[int, int, int, int] | None:
    value = _region_value(region, "abs_box")
    if value is None:
        return None
    if hasattr(value, "x") and hasattr(value, "y"):
        try:
            return (int(value.x), int(value.y), int(value.w), int(value.h))
        except (TypeError, ValueError):
            return None
    if isinstance(value, (list, tuple)) and len(value) == 4:
        try:
            return tuple(int(item) for item in value)  # type: ignore[return-value]
        except (TypeError, ValueError):
            return None
    return None


def _ratio_box_value(region: object) -> tuple[float, float, float, float] | None:
    value = _region_value(region, "ratio_box")
    if not isinstance(value, (list, tuple)) or len(value) != 4:
        return None
    try:
        return tuple(float(item) for item in value)  # type: ignore[return-value]
    except (TypeError, ValueError):
        return None


def _intersection_area(
    first: tuple[int, int, int, int],
    second: tuple[int, int, int, int],
) -> int:
    first_x, first_y, first_w, first_h = first
    second_x, second_y, second_w, second_h = second
    width = max(
        0,
        min(first_x + first_w, second_x + second_w) - max(first_x, second_x),
    )
    height = max(
        0,
        min(first_y + first_h, second_y + second_h) - max(first_y, second_y),
    )
    return width * height


def validate_profile_regions(
    regions: Iterable[object],
    *,
    base_size: tuple[int, int] = DEFAULT_BASE_SIZE,
    critical_overlap_ratio: float = CRITICAL_PLAY_OVERLAP_RATIO,
) -> tuple[ProfileValidationIssue, ...]:
    """Validate configured ROI geometry without changing recognition behavior.

    ``regions`` may contain ``RegionRecord`` instances or JSON-like mappings.
    The function is intentionally pure so startup, diagnostics and tooling can
    use the same checks.  Large overlaps between the four player play
    regions are warnings that block action use, while malformed or incomplete
    ROI geometry remains fatal.
    """

    width, height = (int(base_size[0]), int(base_size[1]))
    issues: list[ProfileValidationIssue] = []
    by_name: dict[str, object] = {}
    if width <= 0 or height <= 0:
        return (
            ProfileValidationIssue(
                code="profile.invalid_base_size",
                severity="fatal",
                message="profile 基准画面尺寸必须为正数",
                details={"base_size": [width, height]},
            ),
        )

    for region in regions:
        name = str(_region_value(region, "name", "")).strip()
        if not name:
            issues.append(
                ProfileValidationIssue(
                    code="roi.missing_name",
                    severity="fatal",
                    message="ROI 缺少名称",
                )
            )
            continue
        if name in by_name:
            issues.append(
                ProfileValidationIssue(
                    code="roi.duplicate_name",
                    severity="fatal",
                    message=f"ROI 名称重复：{name}",
                    region=name,
                )
            )
            continue
        by_name[name] = region
        box = _box_value(region)
        if box is None:
            issues.append(
                ProfileValidationIssue(
                    code="roi.invalid_box",
                    severity="fatal",
                    message=f"ROI 坐标无效：{name}",
                    region=name,
                )
            )
        else:
            x, y, box_width, box_height = box
            if box_width <= 0 or box_height <= 0:
                issues.append(
                    ProfileValidationIssue(
                        code="roi.non_positive_size",
                        severity="fatal",
                        message=f"ROI 宽高必须大于 0：{name}",
                        region=name,
                        details={"abs_box": list(box)},
                    )
                )
            elif x < 0 or y < 0 or x + box_width > width or y + box_height > height:
                issues.append(
                    ProfileValidationIssue(
                        code="roi.out_of_bounds",
                        severity="fatal",
                        message=f"ROI 超出 {width}×{height} 基准画面：{name}",
                        region=name,
                        details={"abs_box": list(box), "base_size": [width, height]},
                    )
                )

        ratio = _ratio_box_value(region)
        if ratio is None:
            issues.append(
                ProfileValidationIssue(
                    code="roi.invalid_ratio_box",
                    severity="fatal",
                    message=f"ROI ratio_box 无效：{name}",
                    region=name,
                )
            )
        else:
            rx, ry, rw, rh = ratio
            if (
                any(not math.isfinite(value) for value in ratio)
                or rw <= 0
                or rh <= 0
                or rx < 0
                or ry < 0
                or rx + rw > 1
                or ry + rh > 1
            ):
                issues.append(
                    ProfileValidationIssue(
                        code="roi.ratio_out_of_bounds",
                        severity="fatal",
                        message=f"ROI ratio_box 超出 0 到 1 范围：{name}",
                        region=name,
                        details={"ratio_box": list(ratio)},
                    )
                )

    missing = [name for name in KEY_PLAY_REGION_NAMES if name not in by_name]
    if missing:
        issues.append(
            ProfileValidationIssue(
                code="roi.missing_key_play_regions",
                severity="fatal",
                message="缺少关键出牌 ROI",
                details={"regions": missing},
            )
        )

    try:
        threshold = float(critical_overlap_ratio)
    except (TypeError, ValueError):
        threshold = CRITICAL_PLAY_OVERLAP_RATIO
    if not math.isfinite(threshold) or threshold <= 0 or threshold > 1:
        threshold = CRITICAL_PLAY_OVERLAP_RATIO

    for index, first_name in enumerate(KEY_PLAY_REGION_NAMES):
        first = _box_value(by_name[first_name]) if first_name in by_name else None
        if first is None or first[2] <= 0 or first[3] <= 0:
            continue
        for second_name in KEY_PLAY_REGION_NAMES[index + 1 :]:
            second = _box_value(by_name[second_name]) if second_name in by_name else None
            if second is None or second[2] <= 0 or second[3] <= 0:
                continue
            overlap = _intersection_area(first, second)
            smaller_area = min(first[2] * first[3], second[2] * second[3])
            ratio = overlap / smaller_area if smaller_area else 0.0
            if ratio >= threshold:
                issues.append(
                    ProfileValidationIssue(
                        code="roi.critical_play_overlap",
                        severity="warning",
                        message=(
                            f"关键出牌 ROI 重叠过大，可能导致动作归属错误："
                            f"{first_name} 与 {second_name}"
                        ),
                        region=first_name,
                        related_region=second_name,
                        details={
                            "overlap_area": overlap,
                            "overlap_ratio_of_smaller": round(ratio, 6),
                            "threshold": threshold,
                            "first_box": list(first),
                            "second_box": list(second),
                        },
                    )
                )
    return tuple(issues)


def profile_validation_report(
    regions: Iterable[object],
    *,
    base_size: tuple[int, int] = DEFAULT_BASE_SIZE,
    critical_overlap_ratio: float = CRITICAL_PLAY_OVERLAP_RATIO,
) -> dict[str, Any]:
    """Return a stable report suitable for startup and diagnostic manifests."""

    issues = validate_profile_regions(
        regions,
        base_size=base_size,
        critical_overlap_ratio=critical_overlap_ratio,
    )
    has_fatal_issue = any(item.severity == "fatal" for item in issues)
    has_action_boundary_warning = any(
        item.code == "roi.critical_play_overlap" for item in issues
    )
    return {
        "status": "fail" if has_fatal_issue else "pass",
        # Opening is blocked only by structural/configuration errors.  A play
        # ROI overlap remains visible as a warning and blocks action use,
        # because it can misattribute a player's move, without making the
        # profile itself fail validation.
        "opening_blocking": has_fatal_issue,
        "action_blocking": has_fatal_issue or has_action_boundary_warning,
        "base_size": [int(base_size[0]), int(base_size[1])],
        "critical_overlap_ratio": float(critical_overlap_ratio),
        "issues": [item.to_dict() for item in issues],
    }


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
    allow_resize: bool = True

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
            allow_resize=bool(self.allow_resize),
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
            "allow_resize": normalized.allow_resize,
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
            allow_resize=bool(data.get("allow_resize", True)),
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


def validate_profile_directory(
    profile_root: Path | ProfilePaths,
    *,
    base_size: tuple[int, int] | None = None,
    critical_overlap_ratio: float = CRITICAL_PLAY_OVERLAP_RATIO,
) -> dict[str, Any]:
    """Validate a profile directory before capture or replay starts.

    This is deliberately non-mutating and never changes recognition results.
    Callers can fail fast with the returned, human-readable issue list instead
    of letting an invalid ROI silently produce empty recognition results.
    """

    root = Path(profile_root.root if isinstance(profile_root, ProfilePaths) else profile_root)
    issues: list[ProfileValidationIssue] = []
    selected_size = tuple(base_size or DEFAULT_BASE_SIZE)
    profile_path = root / PROFILE_CONFIG_FILE_NAME
    regions_path = root / REGIONS_CONFIG_FILE_NAME
    if base_size is None and profile_path.is_file():
        try:
            raw_profile = json.loads(profile_path.read_text(encoding="utf-8"))
            raw_size = raw_profile.get("base_size") if isinstance(raw_profile, dict) else None
            if isinstance(raw_size, (list, tuple)) and len(raw_size) == 2:
                selected_size = (int(raw_size[0]), int(raw_size[1]))
        except (OSError, json.JSONDecodeError, TypeError, ValueError, AttributeError):
            issues.append(
                ProfileValidationIssue(
                    code="profile.invalid_json",
                    severity="fatal",
                    message=f"无法读取 profile.json：{profile_path}",
                    details={"path": str(profile_path)},
                )
            )
    if not regions_path.is_file():
        issues.append(
            ProfileValidationIssue(
                code="profile.missing_regions_config",
                severity="fatal",
                message=f"缺少 regions_config.json：{regions_path}",
                details={"path": str(regions_path)},
            )
        )
        report = profile_validation_report((), base_size=selected_size, critical_overlap_ratio=critical_overlap_ratio)
        report["issues"] = [item.to_dict() for item in issues] + list(report["issues"])
        report["status"] = "fail"
        report["opening_blocking"] = True
        report["action_blocking"] = True
        return report
    try:
        document = json.loads(regions_path.read_text(encoding="utf-8"))
        raw_regions = document.get("regions") if isinstance(document, dict) else None
        if not isinstance(raw_regions, list):
            raise ValueError("regions 必须是数组")
    except (OSError, json.JSONDecodeError, TypeError, ValueError, AttributeError) as exc:
        issues.append(
            ProfileValidationIssue(
                code="profile.invalid_regions_config",
                severity="fatal",
                message=f"无法读取 regions_config.json：{exc}",
                details={"path": str(regions_path)},
            )
        )
        report = profile_validation_report((), base_size=selected_size, critical_overlap_ratio=critical_overlap_ratio)
        report["issues"] = [item.to_dict() for item in issues] + list(report["issues"])
        report["status"] = "fail"
        report["opening_blocking"] = True
        report["action_blocking"] = True
        return report

    report = profile_validation_report(
        raw_regions,
        base_size=selected_size,
        critical_overlap_ratio=critical_overlap_ratio,
    )
    if issues:
        report["issues"] = [item.to_dict() for item in issues] + list(report["issues"])
        has_fatal_issue = any(
            isinstance(item, dict) and item.get("severity") == "fatal"
            for item in report["issues"]
        )
        has_action_boundary_warning = any(
            isinstance(item, dict) and item.get("code") == "roi.critical_play_overlap"
            for item in report["issues"]
        )
        report["status"] = "fail" if has_fatal_issue else "pass"
        report["opening_blocking"] = has_fatal_issue
        report["action_blocking"] = has_fatal_issue or has_action_boundary_warning
    report["profile_root"] = str(root)
    report["profile_name"] = root.name
    return report


def require_valid_profile_directory(
    profile_root: Path | ProfilePaths,
    *,
    base_size: tuple[int, int] | None = None,
    critical_overlap_ratio: float = CRITICAL_PLAY_OVERLAP_RATIO,
) -> dict[str, Any]:
    """Raise a concise, explainable error for a profile that cannot start."""

    report = validate_profile_directory(
        profile_root,
        base_size=base_size,
        critical_overlap_ratio=critical_overlap_ratio,
    )
    if report["status"] == "fail":
        messages = [str(item.get("message", "未知配置错误")) for item in report["issues"]]
        raise ProfileConfigError("profile 配置验证失败：" + "；".join(messages))
    return report


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
