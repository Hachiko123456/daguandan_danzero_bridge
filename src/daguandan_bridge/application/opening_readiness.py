from __future__ import annotations

"""Unified, UI-facing readiness reports for automatic opening detection.

The live controller has several independent sources of "not ready" evidence:
window discovery, capture, ROI validation, page classification, opening-gate
consensus, and worker lifecycle.  This module gives those sources one small
contract so callers can make the same safe decision without matching ad-hoc
messages.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Mapping


SCHEMA = "guandan.opening-readiness/v1"


class OpeningReadinessStatus(str, Enum):
    PASS = "PASS"
    WAIT = "WAIT"
    FAIL = "FAIL"


# Short aliases are intentionally public: they make policy checks read like
# the contract (status is PASS/WAIT/FAIL) while retaining a typed enum API.
PASS = OpeningReadinessStatus.PASS
WAIT = OpeningReadinessStatus.WAIT
FAIL = OpeningReadinessStatus.FAIL


class OpeningReadinessCode(str, Enum):
    READY = "READY"
    READY_WAITING_FIRST_ACTION = "READY_WAITING_FIRST_ACTION"
    LISTENING = "LISTENING"
    WINDOW_NOT_FOUND = "WINDOW_NOT_FOUND"
    MULTIPLE_WINDOWS = "MULTIPLE_WINDOWS"
    WINDOW_MINIMIZED = "WINDOW_MINIMIZED"
    CAPTURE_FAILED = "CAPTURE_FAILED"
    ROI_FATAL = "ROI_FATAL"
    LOBBY = "LOBBY"
    DEAL_IN_PROGRESS = "DEAL_IN_PROGRESS"
    MID_GAME_HAND_COUNT = "MID_GAME_HAND_COUNT"
    HAND_UNSTABLE = "HAND_UNSTABLE"
    OPENING_UNRESOLVED = "OPENING_UNRESOLVED"
    WORKER_FAULT = "WORKER_FAULT"


# Naming used by callers that think of the primary reason as an error code.
OpeningReadinessReason = OpeningReadinessCode


@dataclass(frozen=True)
class OpeningReadinessReport:
    """One safe decision for the opening/listening boundary.

    ``primary_reason`` is deliberately a stable code, not localized prose.
    ``message`` and ``suggested_action`` may be localized by the producer and
    are for presentation only.  WAIT is safe to show in the compact window
    with an explanatory state, while only PASS may establish a session.
    """

    status: OpeningReadinessStatus
    primary_reason: OpeningReadinessCode
    message: str
    recoverable: bool
    suggested_action: str
    details: Mapping[str, object] = field(default_factory=dict)

    @property
    def error_code(self) -> str:
        """Compatibility spelling for integrations that call this an error."""

        return self.primary_reason.value

    @property
    def code(self) -> str:
        return self.primary_reason.value

    @property
    def compact_allowed(self) -> bool:
        """Whether the recommendation compact window may be requested.

        This is intentionally strict and remains true only for PASS.  WAIT
        has its own diagnostic-compact flag so the two contracts cannot be
        confused by a caller that requests actual recommendations.
        """

        return self.status is OpeningReadinessStatus.PASS

    @property
    def diagnostic_compact_allowed(self) -> bool:
        """Whether a future waiting/diagnostic compact surface may be shown."""

        return self.status in {OpeningReadinessStatus.PASS, OpeningReadinessStatus.WAIT}

    @property
    def session_allowed(self) -> bool:
        """Whether the opening is strong enough to establish a live session."""

        return self.status is OpeningReadinessStatus.PASS

    @property
    def can_request_compact(self) -> bool:
        """Explicit naming for callers guarding recommendation compact mode."""

        return self.compact_allowed

    @property
    def can_show_waiting_compact(self) -> bool:
        """Independent diagnostic-compact policy for WAIT states."""

        return self.diagnostic_compact_allowed

    @property
    def hard_error(self) -> bool:
        return self.status is OpeningReadinessStatus.FAIL

    def to_dict(self) -> dict[str, object]:
        payload: dict[str, object] = {
            "schema": SCHEMA,
            "status": self.status.value,
            "primary_reason": self.primary_reason.value,
            "error_code": self.primary_reason.value,
            "message": self.message,
            "recoverable": self.recoverable,
            "suggested_action": self.suggested_action,
            "compact_allowed": self.compact_allowed,
            "can_request_compact": self.can_request_compact,
            "diagnostic_compact_allowed": self.diagnostic_compact_allowed,
            "can_show_waiting_compact": self.can_show_waiting_compact,
            "session_allowed": self.session_allowed,
            "can_start_session": self.session_allowed,
            "hard_error": self.hard_error,
        }
        payload.update(dict(self.details))
        return payload


def _report(
    status: OpeningReadinessStatus,
    reason: OpeningReadinessCode,
    message: str,
    suggested_action: str,
    *,
    recoverable: bool,
    details: Mapping[str, object] | None = None,
) -> OpeningReadinessReport:
    return OpeningReadinessReport(
        status=status,
        primary_reason=reason,
        message=message,
        recoverable=bool(recoverable),
        suggested_action=suggested_action,
        details=dict(details or {}),
    )


def listening_report(*, message: str = "持续监听页面中") -> OpeningReadinessReport:
    return _report(
        WAIT,
        OpeningReadinessCode.LISTENING,
        message,
        "请保持牌桌窗口可见，等待进入牌桌",
        recoverable=True,
    )


def ready_report(
    *,
    message: str = "完整开局已确认，正在建立对局",
    details: Mapping[str, object] | None = None,
) -> OpeningReadinessReport:
    return _report(
        PASS,
        OpeningReadinessCode.READY,
        message,
        "等待推荐窗口显示当前回合建议",
        recoverable=True,
        details=details,
    )


def report_for_page(
    stage: object,
    *,
    message: str | None = None,
    details: Mapping[str, object] | None = None,
) -> OpeningReadinessReport:
    """Convert cheap listening-page evidence into the shared report."""

    value = str(stage or "unknown").strip().lower()
    if value in {"lobby", "settlement", "unknown", "waiting_table"}:
        return _report(
            WAIT,
            OpeningReadinessCode.LOBBY,
            message or "已连接，尚未进入可识别的牌桌开局画面",
            "打开或返回牌桌，等待新一局发牌",
            recoverable=True,
            details=details,
        )
    if value in {"deal_in_progress", "mid_game", "already_started"}:
        return _report(
            FAIL,
            OpeningReadinessCode.DEAL_IN_PROGRESS,
            message or "当前对局已经进行，未捕获完整开局",
            "等待下一局开局后再开始监听",
            recoverable=True,
            details=details,
        )
    if value == "table":
        return _report(
            WAIT,
            OpeningReadinessCode.OPENING_UNRESOLVED,
            message or "已进入牌桌，正在收集完整开局证据",
            "保持牌桌清晰可见，等待起手牌和首出稳定",
            recoverable=True,
            details=details,
        )
    return _report(
        WAIT,
        OpeningReadinessCode.OPENING_UNRESOLVED,
        message or "正在确认牌桌开局状态",
        "保持牌桌窗口可见，等待稳定画面",
        recoverable=True,
        details=details,
    )


def report_for_phase(
    phase: object,
    *,
    hand_count: int | None = None,
    message: str | None = None,
    details: Mapping[str, object] | None = None,
) -> OpeningReadinessReport:
    """Map opening-gate phases to stable readiness codes."""

    value = str(phase or "opening_unresolved").strip().lower()
    count = int(hand_count or 0)
    extra = {"phase": value, "hand_count": count}
    extra.update(dict(details or {}))

    if value == "doubling":
        return _report(
            WAIT,
            OpeningReadinessCode.OPENING_UNRESOLVED,
            message or "已进入牌桌，等待加倍结束",
            "保持监听，加倍结束后继续确认完整开局",
            recoverable=True,
            details=extra,
        )
    if value == "page_recovering":
        return _report(WAIT, OpeningReadinessCode.LISTENING,
                       message or "页面暂时无法确认，继续等待恢复", "保持窗口可见；无需退出监听",
                       recoverable=True, details=extra)
    if value in {"ready_waiting_first_action", "ready_waiting_lead"}:
        lead = extra.get("lead_player")
        lead = getattr(lead, "value", lead)
        lead_label = {
            "self": "自己", "right": "下家", "opposite": "对家", "left": "上家",
        }.get(str(lead), "")
        waiting = f"等待{lead_label}首出"
        return _report(
            PASS,
            OpeningReadinessCode.READY_WAITING_FIRST_ACTION,
            message or f"已进入牌桌，{waiting}",
            f"{waiting}；首出后继续识别出牌",
            recoverable=True,
            details=extra,
        )
    if value == "ready":
        return ready_report(message=message or "完整开局已确认，正在建立对局", details=extra)
    if value in {"unknown", "lobby", "settlement", "settlement_screen", "waiting_table", "table_anchor_unresolved"}:
        settlement = value in {"settlement", "settlement_screen"}
        return _report(
            WAIT,
            OpeningReadinessCode.LOBBY,
            message or (
                "本局已结束，当前处于结算页面，等待下一局"
                if settlement
                else "已连接，等待进入牌桌"
            ),
            "点击继续游戏或换桌，等待新一局发牌" if settlement else "打开或返回牌桌，等待新一局发牌",
            recoverable=True,
            details=extra,
        )
    if value in {"missed_opening", "deal_in_progress", "already_started"}:
        return _report(
            WAIT,
            OpeningReadinessCode.DEAL_IN_PROGRESS,
            message or "当前未确认完整开局，等待新局",
            "保持监听，等待新局完整27张起手牌",
            recoverable=True,
            details=extra,
        )
    if value in {"hand_count_mismatch", "mid_game_hand_count"}:
        return _report(
            WAIT,
            OpeningReadinessCode.MID_GAME_HAND_COUNT,
            message or (
                "当前未确认完整开局，等待新局"
                if 0 < count < 27
                else f"起手牌数量尚未稳定，当前识别到{count}张"
            ),
            (
                "保持监听，等待新局完整27张起手牌"
                if 0 < count < 27
                else "保持牌桌清晰，等待完整27张起手牌"
            ),
            recoverable=True,
            details=extra,
        )
    if value in {"hand_invalid", "hand_unstable", "confirming_hand"}:
        return _report(
            WAIT,
            OpeningReadinessCode.HAND_UNSTABLE,
            message or f"起手牌识别仍在变化，当前识别到{count}张",
            "保持牌桌清晰，等待连续一致的起手牌识别",
            recoverable=True,
            details=extra,
        )
    if value in {"round_level_unresolved", "hand_unresolved", "opening_seed_invalid", "confirming_opening", "opening_unresolved"}:
        return _report(
            WAIT,
            OpeningReadinessCode.OPENING_UNRESOLVED,
            message or "开局证据尚未完整确认",
            "保持牌桌清晰，等待级牌、起手牌和首出证据稳定",
            recoverable=True,
            details=extra,
        )
    return _report(
        WAIT,
        OpeningReadinessCode.OPENING_UNRESOLVED,
        message or "正在确认完整开局",
        "保持牌桌窗口可见，等待稳定开局证据",
        recoverable=True,
        details=extra,
    )


def _normalized_error_code(error: object) -> str:
    raw = getattr(error, "code", "")
    if not raw:
        raw = getattr(error, "error_code", "")
    return str(raw or "").strip().upper().replace("-", "_")


def report_for_error(
    error: object,
    *,
    stage: str = "capture",
    details: Mapping[str, object] | None = None,
    recovering: bool = False,
) -> OpeningReadinessReport:
    """Normalize capture/window/ROI/worker exceptions without string matching in UI."""

    code = _normalized_error_code(error)
    text = str(error or "操作失败")
    lower = text.lower()
    supplied = dict(details or {})
    if isinstance(getattr(error, "details", None), Mapping):
        supplied = {**dict(getattr(error, "details")), **supplied}

    if code in {"WINDOW_NOT_FOUND", "TARGET_WINDOW_NOT_FOUND"} or "没有找到" in text or "window not found" in lower:
        return _report(
            FAIL,
            OpeningReadinessCode.WINDOW_NOT_FOUND,
            text or "未找到牌桌窗口",
            "打开牌桌并确认窗口标题匹配配置",
            recoverable=True,
            details=supplied,
        )
    if code in {"WINDOW_AMBIGUOUS", "WINDOW_MULTIPLE", "MULTIPLE_WINDOWS"} or "多个" in text or "multiple window" in lower:
        return _report(
            FAIL,
            OpeningReadinessCode.MULTIPLE_WINDOWS,
            text or "找到多个可能的牌桌窗口",
            "关闭重复牌桌窗口，或收窄窗口标题关键字配置",
            recoverable=True,
            details=supplied,
        )
    if code in {"WINDOW_MINIMIZED", "WINDOW_ICONIC"} or "最小化" in text or "minimized" in lower:
        status = WAIT if recovering else FAIL
        return _report(
            status,
            OpeningReadinessCode.WINDOW_MINIMIZED,
            text or "牌桌窗口处于最小化状态",
            "还原牌桌窗口并保持其可见",
            recoverable=True,
            details=supplied,
        )
    if (
        code in {"ROI_FATAL", "ROI_FATAL_ERROR", "PROFILE_CONFIG_FATAL"}
        or "roi" in lower and ("fatal" in lower or "致命" in text)
        or type(error).__name__ == "ProfileConfigError"
    ):
        return _report(
            FAIL,
            OpeningReadinessCode.ROI_FATAL,
            text or "识别区域配置存在致命错误",
            "打开完整助手修复 ROI 配置后重新连接",
            recoverable=False,
            details=supplied,
        )
    if code.startswith("WORKER_") or code in {"WORKER_FAULT", "ANALYSIS_FAILED", "RECOGNITION_FAILED"} or stage in {"worker", "analysis"}:
        return _report(
            FAIL,
            OpeningReadinessCode.WORKER_FAULT,
            text or "开局识别后台线程失败",
            "打开完整助手重新连接牌桌",
            recoverable=True,
            details=supplied,
        )
    if code.startswith("CAPTURE_") or code in {"CAPTURE_FAILED", "CAPTURE_BACKEND_FAILED", "CAPTURE_OCCLUDED"} or stage in {"capture", "window"}:
        action = "移开遮挡后点击继续" if code == "CAPTURE_OCCLUDED" else "确认牌桌窗口可见后重新连接"
        return _report(
            FAIL,
            OpeningReadinessCode.CAPTURE_FAILED,
            text or "牌桌画面捕获失败",
            action,
            recoverable=True,
            details=supplied,
        )
    return _report(
        FAIL,
        OpeningReadinessCode.WORKER_FAULT if stage in {"worker", "analysis", "recognition"} else OpeningReadinessCode.CAPTURE_FAILED,
        text or "监听发生未知错误",
        "打开完整助手重新连接牌桌",
        recoverable=True,
        details=supplied,
    )


def coerce_report(value: object) -> OpeningReadinessReport | None:
    """Read a report from a signal payload while accepting old payloads."""

    if isinstance(value, OpeningReadinessReport):
        return value
    if not isinstance(value, Mapping):
        return None
    for key in ("report", "readiness"):
        nested = value.get(key)
        if nested is not value:
            result = coerce_report(nested)
            if result is not None:
                return result
    status_value = str(value.get("status", "") or "").upper()
    reason_value = str(value.get("primary_reason", value.get("error_code", "")) or "").upper()
    if status_value not in {item.value for item in OpeningReadinessStatus} or not reason_value:
        return None
    try:
        status = OpeningReadinessStatus(status_value)
        reason = OpeningReadinessCode(reason_value)
    except ValueError:
        return None
    return OpeningReadinessReport(
        status=status,
        primary_reason=reason,
        message=str(value.get("message", "") or ""),
        recoverable=bool(value.get("recoverable", False)),
        suggested_action=str(value.get("suggested_action", "") or ""),
        details={
            key: item
            for key, item in value.items()
            if key not in {
                "schema", "status", "primary_reason", "error_code", "message",
                "recoverable", "suggested_action", "compact_allowed",
                "can_request_compact", "diagnostic_compact_allowed",
                "can_show_waiting_compact", "session_allowed", "can_start_session",
                "hard_error",
            }
        },
    )


__all__ = [
    "SCHEMA",
    "OpeningReadinessStatus",
    "OpeningReadinessCode",
    "OpeningReadinessReason",
    "OpeningReadinessReport",
    "PASS",
    "WAIT",
    "FAIL",
    "listening_report",
    "ready_report",
    "report_for_page",
    "report_for_phase",
    "report_for_error",
    "coerce_report",
]
