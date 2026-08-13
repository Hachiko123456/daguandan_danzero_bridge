from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .config import PROFILES_ROOT
from .storage import atomic_write_json


ADVISOR_OPTIONS: tuple[tuple[str, str], ...] = (
    ("danzero", "DanZero"),
    ("fabledan", "FableDan"),
)
EVALUATION_STRATEGY_OPTIONS: tuple[tuple[str, str], ...] = (
    ("danzero_model", "DanZero"),
    ("fabledan_model", "FableDan 模型"),
    ("fabledan_rule", "FableDan RuleAgent 规则基线"),
)
DEFAULT_ADVISOR_STRATEGY = "danzero"
_VALID_ADVISORS = {value for value, _label in ADVISOR_OPTIONS}
_VALID_EVALUATION_STRATEGIES = {
    value for value, _label in EVALUATION_STRATEGY_OPTIONS
}


def normalize_advisor_strategy(value: object) -> str:
    strategy = str(value or DEFAULT_ADVISOR_STRATEGY).strip().lower()
    if strategy not in _VALID_ADVISORS:
        raise ValueError(f"不支持的建议模型：{strategy}")
    return strategy


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


def build_advisor(
    strategy: object,
    *,
    profiles_root: Path | str = PROFILES_ROOT,
    profile_name: str = "tencent_daguandan",
):
    normalized = normalize_advisor_strategy(strategy)
    if normalized == "fabledan":
        from .fabledan import FableDanAdvisor

        return FableDanAdvisor(profiles_root, profile_name)
    from .danzero import DanzeroAdvisor

    return DanzeroAdvisor()


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
    """Build an evaluation-only advisor without reading or writing profile defaults."""

    normalized = normalize_evaluation_strategy(strategy)
    if normalized.startswith("fabledan_"):
        from .fabledan import FableDanAdvisor

        policy = "model_required" if normalized == "fabledan_model" else "rule_only"
        return FableDanAdvisor(
            profiles_root,
            profile_name,
            runtime_policy=policy,
        )
    from .danzero import DanzeroAdvisor

    return DanzeroAdvisor()
