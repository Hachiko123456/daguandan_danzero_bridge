from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from .config import PROFILES_ROOT
from .storage import atomic_write_json


ADVISOR_OPTIONS: tuple[tuple[str, str], ...] = (
    ("fabledan", "FableDan"),
    ("danzero", "DanZero"),
)
EVALUATION_STRATEGY_OPTIONS: tuple[tuple[str, str], ...] = (
    ("fabledan_model", "FableDan 模型"),
    ("danzero_model", "DanZero"),
)
DEFAULT_ADVISOR_STRATEGY = "fabledan"
DEFAULT_FABLEDAN_DEBUG = False
DEFAULT_FABLEDAN_DIAGNOSTICS = "off"
DEFAULT_SESSION_DATA_RECORDING_ENABLED = True
RECORDING_MODE_OPTIONS: tuple[tuple[str, str], ...] = (
    ("none", "不保存"),
    ("game", "对局录制"),
    ("all", "完整牌桌录制"),
)
DEFAULT_RECORDING_MODE = "game"
_VALID_ADVISORS = {value for value, _label in ADVISOR_OPTIONS}
_VALID_EVALUATION_STRATEGIES = {
    value for value, _label in EVALUATION_STRATEGY_OPTIONS
} | {"fabledan_rule"}
_VALID_RECORDING_MODES = {value for value, _label in RECORDING_MODE_OPTIONS}


def normalize_advisor_strategy(value: object) -> str:
    strategy = str(value or DEFAULT_ADVISOR_STRATEGY).strip().lower()
    if strategy not in _VALID_ADVISORS:
        raise ValueError(f"不支持的建议模型：{strategy}")
    return strategy


def advisor_strategy_id(advisor: object) -> str:
    """Return the explicit production backend advertised by an advisor."""

    value = getattr(advisor, "strategy_id", None)
    if not isinstance(value, str) or not value.strip():
        raise ValueError("建议模型缺少明确的 strategy_id")
    return normalize_advisor_strategy(value)


def load_profile_advisor_strategy(
    profiles_root: Path | str = PROFILES_ROOT,
    profile_name: str = "tencent_daguandan",
) -> str:
    path = Path(profiles_root) / profile_name / "profile.json"
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(raw, dict):
            return DEFAULT_ADVISOR_STRATEGY
        return normalize_advisor_strategy(raw.get("advisor_strategy"))
    except (OSError, json.JSONDecodeError, ValueError):
        return DEFAULT_ADVISOR_STRATEGY


def load_profile_fabledan_debug(
    profiles_root: Path | str = PROFILES_ROOT,
    profile_name: str = "tencent_daguandan",
) -> bool:
    path = Path(profiles_root) / profile_name / "profile.json"
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return DEFAULT_FABLEDAN_DEBUG
    if not isinstance(raw, dict):
        return DEFAULT_FABLEDAN_DEBUG
    value = raw.get("fabledan_debug", DEFAULT_FABLEDAN_DEBUG)
    return value if isinstance(value, bool) else DEFAULT_FABLEDAN_DEBUG


def load_profile_session_data_recording_enabled(
    profiles_root: Path | str = PROFILES_ROOT,
    profile_name: str = "tencent_daguandan",
) -> bool:
    """Compatibility wrapper for callers that only need an on/off answer."""

    return load_profile_recording_mode(profiles_root, profile_name) != "none"


def normalize_recording_mode(value: object) -> str:
    normalized = str(value or DEFAULT_RECORDING_MODE).strip().lower()
    if normalized not in _VALID_RECORDING_MODES:
        raise ValueError(f"不支持的保存方式：{value}")
    return normalized


def load_profile_recording_mode(
    profiles_root: Path | str = PROFILES_ROOT,
    profile_name: str = "tencent_daguandan",
) -> str:
    """Load the persisted replay policy, upgrading the former boolean setting."""

    path = Path(profiles_root) / profile_name / "profile.json"
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return DEFAULT_RECORDING_MODE
    if not isinstance(raw, dict):
        return DEFAULT_RECORDING_MODE
    if "recording_mode" in raw:
        try:
            return normalize_recording_mode(raw["recording_mode"])
        except ValueError:
            pass
    legacy = raw.get(
        "save_session_data", DEFAULT_SESSION_DATA_RECORDING_ENABLED
    )
    return "game" if legacy is not False else "none"


def normalize_fabledan_diagnostics(value: object) -> str:
    normalized = str(value or DEFAULT_FABLEDAN_DIAGNOSTICS).strip().lower()
    if normalized not in {"off", "basic", "full"}:
        raise ValueError(f"不支持的 FableDan 诊断级别：{value}")
    return normalized


def load_profile_fabledan_diagnostics(
    profiles_root: Path | str = PROFILES_ROOT,
    profile_name: str = "tencent_daguandan",
) -> str:
    environment = os.getenv("FABLEDAN_DIAGNOSTICS")
    if environment is not None:
        try:
            return normalize_fabledan_diagnostics(environment)
        except ValueError:
            return DEFAULT_FABLEDAN_DIAGNOSTICS
    path = Path(profiles_root) / profile_name / "profile.json"
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return DEFAULT_FABLEDAN_DIAGNOSTICS
    if not isinstance(raw, dict):
        return DEFAULT_FABLEDAN_DIAGNOSTICS
    if "fabledan_diagnostics" in raw:
        try:
            return normalize_fabledan_diagnostics(raw["fabledan_diagnostics"])
        except ValueError:
            return DEFAULT_FABLEDAN_DIAGNOSTICS
    return "full" if raw.get("fabledan_debug") is True else DEFAULT_FABLEDAN_DIAGNOSTICS


def save_profile_advisor_strategy(
    profiles_root: Path | str,
    profile_name: str,
    strategy: object,
) -> str:
    normalized = normalize_advisor_strategy(strategy)
    path = Path(profiles_root) / profile_name / "profile.json"
    try:
        raw: Any = json.loads(path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise ValueError(f"缺少 profile 配置：{path}") from exc
    except json.JSONDecodeError as exc:
        raise ValueError(f"profile 配置已损坏：{path}") from exc
    if not isinstance(raw, dict):
        raise ValueError("profile.json 顶层结构必须是 JSON 对象")
    raw["advisor_strategy"] = normalized
    atomic_write_json(path, raw)
    return normalized


def save_profile_session_data_recording_enabled(
    profiles_root: Path | str,
    profile_name: str,
    enabled: object,
) -> bool:
    """Persist the next-live-game recording preference in ``profile.json``."""

    if not isinstance(enabled, bool):
        raise ValueError("保存对局数据开关必须为布尔值")
    path = Path(profiles_root) / profile_name / "profile.json"
    try:
        raw: Any = json.loads(path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise ValueError(f"缺少 profile 配置：{path}") from exc
    except json.JSONDecodeError as exc:
        raise ValueError(f"profile 配置已损坏：{path}") from exc
    if not isinstance(raw, dict):
        raise ValueError("profile.json 顶层结构必须是 JSON 对象")
    raw["save_session_data"] = enabled
    raw["recording_mode"] = "game" if enabled else "none"
    atomic_write_json(path, raw)
    return enabled


def save_profile_recording_mode(
    profiles_root: Path | str,
    profile_name: str,
    mode: object,
) -> str:
    """Persist one of ``none``, ``game``, or ``all`` recording modes."""

    normalized = normalize_recording_mode(mode)
    path = Path(profiles_root) / profile_name / "profile.json"
    try:
        raw: Any = json.loads(path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise ValueError(f"缺少 profile 配置：{path}") from exc
    except json.JSONDecodeError as exc:
        raise ValueError(f"profile 配置已损坏：{path}") from exc
    if not isinstance(raw, dict):
        raise ValueError("profile.json 顶层结构必须是 JSON 对象")
    raw["recording_mode"] = normalized
    # Keep this key for existing scripts and older app versions.
    raw["save_session_data"] = normalized != "none"
    atomic_write_json(path, raw)
    return normalized


def build_advisor(
    strategy: object,
    *,
    profiles_root: Path | str = PROFILES_ROOT,
    profile_name: str = "tencent_daguandan",
    fabledan_debug: bool | None = None,
    fabledan_diagnostics: str | None = None,
):
    normalized = normalize_advisor_strategy(strategy)
    if normalized == "fabledan":
        from .fabledan import FableDanAdvisor

        diagnostics = (
            normalize_fabledan_diagnostics(fabledan_diagnostics)
            if fabledan_diagnostics is not None
            else "full"
            if fabledan_debug is True
            else "off"
            if fabledan_debug is False
            else load_profile_fabledan_diagnostics(profiles_root, profile_name)
        )
        return FableDanAdvisor(
            profiles_root,
            profile_name,
            runtime_policy="model_required",
            diagnostics=diagnostics,
        )
    from .danzero import DanzeroAdvisor

    return DanzeroAdvisor(profiles_root, profile_name)


def normalize_evaluation_strategy(value: object) -> str:
    strategy = str(value).strip().lower()
    if strategy not in _VALID_EVALUATION_STRATEGIES:
        raise ValueError(f"不支持的评测策略：{strategy}")
    return strategy


def build_evaluation_advisor(
    strategy: object,
    *,
    profiles_root: Path | str = PROFILES_ROOT,
    profile_name: str = "tencent_daguandan",
):
    """构建不读取也不改写配置默认值的评测专用策略。"""

    normalized = normalize_evaluation_strategy(strategy)
    if normalized.startswith("fabledan_"):
        from .fabledan import FableDanAdvisor

        policy = "model_required" if normalized == "fabledan_model" else "rule_only"
        return FableDanAdvisor(
            profiles_root,
            profile_name,
            runtime_policy=policy,
            diagnostics="full",
            write_decision_log=False,
        )
    from .danzero import DanzeroAdvisor

    return DanzeroAdvisor(profiles_root, profile_name)
