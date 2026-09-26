"""Read-only GUI page for inspecting target windows and one-frame reports.

The page intentionally depends on a narrow report-service protocol.  It does
not import Win32 APIs and does not try to repair, focus, move, resize, or
otherwise control the selected window.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping, Protocol, Sequence

from PySide6.QtCore import Qt
from PySide6.QtGui import QPixmap
from PySide6.QtWidgets import (
    QApplication,
    QFileDialog,
    QFrame,
    QComboBox,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QPlainTextEdit,
    QPushButton,
    QScrollArea,
    QSplitter,
    QSizePolicy,
    QToolButton,
    QVBoxLayout,
    QWidget,
)
from qfluentwidgets import (
    BodyLabel,
    CaptionLabel,
    CardWidget,
    PrimaryPushButton,
    PushButton,
    StrongBodyLabel,
    SubtitleLabel,
)

from ..application.window_debug_report import WindowDebugReportService
from ..application.diagnostic_presentation import _roi_opening_issues

try:
    from ..application.session_diagnostic_frames import SessionDiagnosticFrameStore
except Exception:  # optional while the runtime API is being rolled out
    SessionDiagnosticFrameStore = None  # type: ignore[assignment,misc]


WindowRecord = Mapping[str, object]
WindowReport = Mapping[str, object]


class ProblemExportUiPort(Protocol):
    """UI presenter injected by the composition root, not a runtime dependency."""

    def bind_button(self, button: QWidget) -> None: ...

    def request(self, *, parent: QWidget, case_directory: Path | None = None) -> None: ...


class WindowDebugReportPort(Protocol):
    """The read-only service surface needed by :class:`WindowDebugPage`."""

    def list_windows(self) -> Sequence[WindowRecord]: ...

    def probe(self, hwnd: int) -> WindowRecord: ...

    def build(
        self,
        *,
        hwnd: int,
        capture: bool = False,
        recognize: bool = False,
        include_media: bool = False,
    ) -> WindowReport: ...

    def build_from_report_file(self, source_path: str | Path, *, recognize: bool = True) -> WindowReport: ...


def _pretty_json(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, indent=2, default=str)


def _window_label(window: WindowRecord) -> str:
    hwnd = window.get("hwnd", "?")
    title = str(window.get("title") or "（无标题）").strip()
    process = str(window.get("process_name") or "未知进程").strip()
    class_name = str(window.get("class_name") or "未知类").strip()
    if process.casefold() in {"wechatex.exe", "weixin.exe"} or "微信" in title:
        return f"{title} · 微信小程序窗口"
    return f"{title} · {process}"


def _readiness(report: WindowReport) -> Mapping[str, object] | None:
    inputs = report.get("opening_readiness_inputs")
    if not isinstance(inputs, Mapping):
        return None
    value = inputs.get("readiness")
    return value if isinstance(value, Mapping) else None


def _key_blockers(report: WindowReport) -> list[str]:
    """Extract presentation hints without inventing a second diagnostic policy."""

    blockers: list[str] = []

    errors = report.get("errors")
    if isinstance(errors, Sequence) and not isinstance(errors, (str, bytes)):
        for error in errors:
            if isinstance(error, Mapping):
                code = str(error.get("code") or error.get("type") or "ERROR")
                message = str(error.get("message") or "未提供错误信息")
                blockers.append(f"{code}: {message}")
            elif error:
                blockers.append(str(error))

    window = report.get("window")
    if isinstance(window, Mapping):
        if window.get("visible") is False:
            blockers.append("目标窗口当前不可见。")
        if window.get("iconic") is True:
            blockers.append("目标窗口当前处于最小化状态。")
        if not window.get("client_rect"):
            blockers.append("未取得有效客户区矩形，无法确认单帧尺寸。")

    capture = report.get("capture")
    if isinstance(capture, Mapping):
        standardization = capture.get("standardization")
        if isinstance(standardization, Mapping):
            issues = standardization.get("issues")
            if isinstance(issues, Sequence) and not isinstance(issues, (str, bytes)):
                blockers.extend(str(item) for item in issues if item)

    roi_validation = report.get("roi_validation")
    if isinstance(roi_validation, Mapping):
        status = str(roi_validation.get("status") or "").lower()
        all_issues = roi_validation.get("issues")
        issues = _roi_opening_issues(roi_validation)
        status_blocks = status not in {"", "pass", "ok", "valid"}
        if all_issues and not issues:
            status_blocks = False
        if status_blocks:
            blockers.append(f"ROI 配置校验：{status}")
        blockers.extend(str(item) for item in issues if item)

    readiness = _readiness(report)
    if readiness is not None:
        status = str(readiness.get("status") or "").upper()
        reason = str(
            readiness.get("primary_reason")
            or readiness.get("error_code")
            or "未提供原因码"
        )
        message = str(readiness.get("message") or "")
        if status and status != "PASS":
            blockers.append(
                f"开局就绪度 {status} / {reason}"
                + (f"：{message}" if message else "")
            )

    # Preserve order while avoiding repeated errors from layered service data.
    unique: list[str] = []
    seen: set[str] = set()
    for blocker in blockers:
        text = blocker.strip()
        if text and text not in seen:
            seen.add(text)
            unique.append(text)
    return unique


_UNKNOWN = "未识别/尚未提供（报告中没有该字段）"
_SEAT_LABELS = {
    "self": "自己",
    "right": "右家",
    "opposite": "对家",
    "left": "左家",
}


def _mapping(value: object) -> Mapping[str, object]:
    return value if isinstance(value, Mapping) else {}


def _sequence(value: object) -> list[object]:
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return list(value)
    return []


def _first_value(sources: Sequence[Mapping[str, object]], *keys: str) -> object:
    for source in sources:
        for key in keys:
            value = source.get(key)
            if value is not None and value != "":
                return value
    return None


def _text(value: object, *, unknown: str = _UNKNOWN) -> str:
    if value is None or value == "":
        return unknown
    if isinstance(value, bool):
        return "是" if value else "否"
    return str(value)


def _player_text(value: object) -> str:
    if value is None or value == "":
        return _UNKNOWN
    raw = str(value)
    return f"{_SEAT_LABELS.get(raw, raw)}（{raw}）" if raw in _SEAT_LABELS else raw


def _confidence_text(value: object) -> str:
    if value is None or value == "":
        return "未提供"
    try:
        score = float(value)
    except (TypeError, ValueError):
        return str(value)
    return f"{score:.3f}（{score:.1%}）"


def _cards_text(value: object) -> str:
    if value is None or value == "":
        return "未识别/尚未提供"
    if isinstance(value, Mapping):
        return _pretty_json(value)
    values = _sequence(value)
    if values:
        return " ".join(str(item) for item in values)
    return str(value)


def _detail_lines(value: object, *, indent: str = "  ") -> list[str]:
    """Render structured diagnostic fragments without inventing missing data."""

    if isinstance(value, Mapping):
        lines: list[str] = []
        for key, item in value.items():
            if isinstance(item, (Mapping, list, tuple)):
                rendered = (
                    _confidence_text(item)
                    if str(key).casefold() in {"score", "confidence", "threshold", "min_score", "max_score"}
                    else _text(item)
                )
                lines.append(f"{indent}{key}\uFF1A{rendered}")
                lines.extend(_detail_lines(item, indent=indent + "  "))
            else:
                rendered = (
                    _confidence_text(item)
                    if str(key).casefold() in {"score", "confidence", "threshold", "min_score", "max_score"}
                    else _text(item)
                )
                lines.append(f"{indent}{key}\uFF1A{rendered}")
        return lines or [f"{indent}{_UNKNOWN}"]
    values = _sequence(value)
    if values:
        lines = []
        for item in values:
            if isinstance(item, (Mapping, list, tuple)):
                lines.extend(_detail_lines(item, indent=indent))
            else:
                lines.append(f"{indent}- {_text(item)}")
        return lines or [f"{indent}{_UNKNOWN}"]
    return [f"{indent}{_text(value)}"]


def _structured_detail_text(report: WindowReport) -> list[str]:
    texts: list[str] = []
    candidates: list[object] = [report]
    recognition = _mapping(report.get("recognition"))
    user_view = _mapping(report.get("user_view"))
    for container in (recognition, user_view):
        candidates.append(container)
    keys = (
        "详细识别结果", "详细诊断", "识别详情", "详细文本", "详细说明",
        "可复制详细诊断", "recognition_detail_text", "detailed_text",
    )
    for container in candidates:
        if not isinstance(container, Mapping):
            continue
        for key in keys:
            value = container.get(key)
            if isinstance(value, str) and value.strip() and value.strip() not in texts:
                texts.append(value.strip())
    return texts


def _render_detailed_recognition(report: WindowReport, source_label: str) -> str:
    """Build a defensive, Chinese, report-only recognition presentation."""

    recognition = _mapping(report.get("recognition"))
    result = _mapping(recognition.get("result"))
    details = _mapping(_first_value(
        [report, recognition, result],
        "recognition_details", "detailed_recognition", "diagnostic_details", "details",
    ))
    summary = _mapping(_first_value([report, details], "recognition_summary", "summary"))
    readiness = _mapping(_readiness(report) or {})
    sources = _mapping(_first_value([details, summary, result, report], "sources"))
    confidences = _mapping(_first_value([details, summary, result, report], "field_confidences"))
    assessments = _first_value(
        [report, details, recognition, result], "field_assessments", "field_results"
    )

    lines = [
        "【详细识别结果】",
        f"来源：{source_label}",
        f"诊断模式：{_text(report.get('diagnosis_mode'))}",
        "",
        "【牌局概览】",
        f"牌局阶段：{_text(_first_value([summary, details, result, report], 'game_phase', 'phase', 'game_stage', 'stage'))}",
        f"开局状态：{_text(_first_value([summary, details, readiness, report], 'opening_status', 'status'))}",
        f"级牌：{_text(_first_value([summary, details, result, report], 'round_level', 'level'))}",
        f"百变牌：{_text(_first_value([summary, details, result, report], 'wild_rank', 'wildcard', 'wild_card'))}",
        f"首出玩家：{_player_text(_first_value([summary, details, result, report], 'lead_player', 'first_player'))}",
        f"当前行动者：{_player_text(_first_value([summary, details, result, report], 'current_player', 'active_player'))}",
        f"首出置信度：{_confidence_text(_first_value([summary, details], 'lead_confidence'))}",
        f"当前行动者置信度：{_confidence_text(_first_value([summary, details], 'current_player_confidence', 'active_player_confidence'))}",
    ]

    hand = _first_value([summary, details, result], "my_hand", "hand", "hand_cards")
    hand_count = _first_value([summary, details, result], "hand_count", "my_hand_count")
    if hand_count is None and _sequence(hand):
        hand_count = len(_sequence(hand))
    target_count = _first_value(
        [summary, details, result, report],
        "target_hand_count", "expected_hand_count", "hand_target_count",
    )
    unconfirmed = _first_value(
        [summary, details, result],
        "unconfirmed_cards", "unconfirmed_hand", "uncertain_cards", "unknown_cards",
    )
    lines.extend([
        "",
        "【手牌】",
        f"手牌数量：{_text(hand_count, unknown='未识别/尚未提供')} 张",
        f"\u76ee\u6807\u6570\u91cf\uff1a{_text(target_count, unknown='\u5c1a\u672a\u63d0\u4f9b')}",
        f"完整手牌：{_cards_text(hand)}",
        f"未确认牌：{_cards_text(unconfirmed)}",
    ])

    events = _first_value([report, details, result], "recognized_events", "events", "play_events")
    event_values = _sequence(events)
    lines.extend(["", "【各家出牌/不出事件】"])
    if not event_values:
        lines.append(f"  {_UNKNOWN}")
    else:
        for index, event in enumerate(event_values, 1):
            item = _mapping(event)
            if not item:
                lines.append(f"  {index}. {_text(event)}")
                continue
            action = item.get("action")
            if action is None and "is_pass" in item:
                action = "不出" if item.get("is_pass") is True else "出牌"
            lines.extend([
                f"  {index}. 玩家：{_player_text(_first_value([item], 'player', 'seat', 'actor'))}",
                f"     动作：{_text(action, unknown='未提供')}",
                f"     牌面：{_cards_text(_first_value([item], 'cards', 'card_list', 'play'))}",
                f"     置信度：{_confidence_text(_first_value([item], 'confidence', 'score'))}",
                f"     来源/ROI：{_text(_first_value([item], 'source', 'roi', 'evidence_source'), unknown='未提供')}",
            ])
            reason = _first_value([item], "reason", "diagnostics", "rejection_reason")
            if reason is not None:
                lines.append(f"     原因：{_text(reason)}")

    evidence = _first_value(
        [report, details, result, _mapping(_mapping(report.get("opening_readiness_inputs")).get("opening_signal"))],
        "lead_evidence", "candidates", "first_play_evidence",
    )
    lines.extend(["", "【首出证据候选】"])
    evidence_values = _sequence(evidence)
    if not evidence_values:
        lines.append(f"  {_UNKNOWN}")
    else:
        for index, candidate in enumerate(evidence_values, 1):
            item = _mapping(candidate)
            if not item:
                lines.append(f"  {index}. {_text(candidate)}")
                continue
            lines.append(f"  {index}. 候选玩家：{_player_text(_first_value([item], 'candidate_seat', 'seat', 'player'))}")
            lines.append(f"     首出文字证据：{_confidence_text(_first_value([item], 'first_play_score', 'first_play_confidence'))}")
            lines.append(f"     牌面动作证据：{_confidence_text(_first_value([item], 'card_action_score', 'card_confidence'))}")
            lines.append(f"     计时/状态证据：{_confidence_text(_first_value([item], 'timer_score', 'timer_confidence'))}")
            lines.append(f"     状态：{_text(_first_value([item], 'status'), unknown='未提供')}")
            lines.append(f"     拒绝原因：{_text(_first_value([item], 'rejection_reason'), unknown='无/未提供')}")

    lines.extend(["", "【字段置信度、来源与未解决字段】"])
    unresolved = _first_value([details, summary, result, report], "unresolved_fields", "unresolved")
    if isinstance(assessments, Mapping):
        assessments = list(assessments.values())
    assessment_values = _sequence(assessments)
    if assessment_values:
        for index, assessment in enumerate(assessment_values, 1):
            item = _mapping(assessment)
            if not item:
                lines.append(f"  {index}. {_text(assessment)}")
                continue
            lines.append(
                f"  {index}. {_text(_first_value([item], 'label', 'field'), unknown='未提供字段')}："
                f"{_text(_first_value([item], 'value'), unknown='未识别')}；"
                f"状态={_text(_first_value([item], 'status'), unknown='未提供')}；"
                f"置信度={_confidence_text(_first_value([item], 'confidence', 'score'))}；"
                f"来源={_text(_first_value([item], 'source'), unknown='未提供')}；"
                f"阈值={_text(_first_value([item], 'threshold'), unknown='未提供')}；"
                f"原因={_text(_first_value([item], 'reason', 'rejection_reason'), unknown='未提供')}"
            )
    elif confidences or sources:
        for field in sorted(set(confidences) | set(sources), key=str):
            lines.append(
                f"  {field}：置信度={_confidence_text(confidences.get(field))}；"
                f"来源={_text(sources.get(field), unknown='未提供')}"
            )
    else:
        lines.append(f"  {_UNKNOWN}")
    lines.append(f"  未解决字段：{_text(unresolved, unknown='无/未提供')}")

    threshold_parts = []
    for container in (report, details, recognition, result):
        for key in ("threshold_analysis", "candidate_rejections", "rejection_reasons", "candidate_match_diagnostics"):
            value = container.get(key)
            if value not in (None, "", [], {}):
                threshold_parts.append((key, value))
    trace = _mapping(recognition.get("trace"))
    for key in ("threshold_analysis", "candidate_rejections", "rejection_reasons", "candidate_match_diagnostics"):
        value = trace.get(key)
        if value not in (None, "", [], {}):
            threshold_parts.append((key, value))
    lines.extend(["", "【阈值、候选拒绝与技术原因】"])
    if threshold_parts:
        for key, value in threshold_parts:
            lines.append(f"  {key}：")
            lines.extend(_detail_lines(value, indent="    "))
    else:
        lines.append(f"  {_UNKNOWN}")

    diagnostics: list[object] = []
    for container in (report, details, result):
        for key in ("diagnostics", "blockers", "issues", "errors"):
            value = container.get(key)
            if value not in (None, "", [], {}):
                diagnostics.append(value)
    roi = _mapping(report.get("roi_validation"))
    capture = _mapping(report.get("capture"))
    standardization = _mapping(capture.get("standardization"))
    for value in (roi.get("issues"), standardization.get("issues")):
        if value not in (None, "", [], {}):
            diagnostics.append(value)
    lines.extend(["", "\u3010识别诊断与阻塞原因\u3011"])
    if diagnostics:
        for value in diagnostics:
            lines.extend(_detail_lines(value, indent="  "))
    else:
        lines.append(f"  {_UNKNOWN}")

    suggestions = _first_value(
        [report, details, summary, _mapping(report.get("user_view"))],
        "recommendations", "repair_suggestions", "suggested_actions", "next_actions",
    )
    lines.extend(["", "【可执行修复建议】"])
    if suggestions not in (None, "", [], {}):
        lines.extend(_detail_lines(suggestions, indent="  "))
    else:
        lines.append("  尚未提供可执行建议；请结合上面的阈值、ROI、模板和截图质量原因处理。")

    detail_texts = _structured_detail_text(report)
    if detail_texts:
        lines.extend(["", "【报告提供的中文详细文本】"])
        lines.extend(detail_texts)
    return "\n".join(lines)


class WindowDebugPage(QWidget):
    """A read-only window picker and one-frame diagnostic report page."""

    def __init__(
        self,
        report_service: WindowDebugReportPort | None = None,
        parent: QWidget | None = None,
        *,
        service: WindowDebugReportPort | None = None,
        problem_export_ui: ProblemExportUiPort | None = None,
    ) -> None:
        super().__init__(parent)
        self.setObjectName("windowDebugPage")
        self.report_service = report_service or service or WindowDebugReportService()
        self.problem_export_ui = problem_export_ui
        self._window_records: dict[int, WindowRecord] = {}
        self._last_report: WindowReport | None = None
        self._session_directory: Path | None = None
        self._session_frames: list[dict[str, object]] = []
        self._current_session_frame_index = -1
        self._current_preview_pixmap = QPixmap()
        self._build_ui()
        if problem_export_ui is not None:
            problem_export_ui.bind_button(self.problem_export_button)
        self._set_idle_state()

    def _build_ui(self) -> None:
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Ignored)
        self.setMinimumHeight(0)

        page_layout = QVBoxLayout(self)
        page_layout.setContentsMargins(0, 0, 0, 0)
        page_layout.setSpacing(0)
        scroll = QScrollArea(self)
        scroll.setObjectName("windowDebugScrollArea")
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        scroll.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        content = QWidget(scroll)
        content.setObjectName("windowDebugContent")
        content.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred)
        root = QVBoxLayout(content)
        root.setContentsMargins(16, 12, 16, 16)
        root.setSpacing(8)

        header = CardWidget(self)
        header_layout = QVBoxLayout(header)
        header_layout.setContentsMargins(20, 16, 20, 16)
        header_layout.setSpacing(4)
        title_row = QHBoxLayout()
        title_row.addWidget(SubtitleLabel("窗口与牌局诊断"), 1)
        self.problem_export_button = PrimaryPushButton("导出问题包", header)
        self.problem_export_button.setObjectName("windowDebugProblemExportButton")
        self.problem_export_button.setToolTip("导出所选对局；未选择时导出当前或最近对局，不自动上传")
        self.problem_export_button.clicked.connect(self.request_problem_export)
        title_row.addWidget(self.problem_export_button)
        header_layout.addLayout(title_row)
        header_layout.addWidget(
            CaptionLabel(
                "刷新可见窗口，选择 HWND 后执行一次只读单帧诊断。"
                "页面不会自动移动、缩放、聚焦、恢复或点击目标窗口。"
            )
        )
        root.addWidget(header)

        controls = CardWidget(self)
        controls_layout = QHBoxLayout(controls)
        controls_layout.setContentsMargins(16, 12, 16, 12)
        controls_layout.setSpacing(8)

        self.refresh_button = PrimaryPushButton("刷新窗口列表", self)
        self.refresh_button.setObjectName("refreshWindowsButton")
        self.refresh_button.clicked.connect(self.refresh_windows)
        controls_layout.addWidget(self.refresh_button)

        controls_layout.addWidget(StrongBodyLabel("选择窗口："))
        self.window_selector = QComboBox(self)
        self.window_selector.setObjectName("windowSelector")
        self.window_selector.setMinimumHeight(34)
        self.window_selector.setSizeAdjustPolicy(
            QComboBox.SizeAdjustPolicy.AdjustToMinimumContentsLengthWithIcon
        )
        self.window_selector.currentIndexChanged.connect(self._window_selection_changed)
        controls_layout.addWidget(self.window_selector, 1)

        self.diagnose_button = PrimaryPushButton("诊断当前单帧", self)
        self.diagnose_button.setObjectName("diagnoseWindowButton")
        self.diagnose_button.setToolTip(
            "调用窗口调试报告服务捕获一帧并运行识别；截图只保留在内存中。"
        )
        self.diagnose_button.clicked.connect(self.diagnose_current_window)
        controls_layout.addWidget(self.diagnose_button)

        self.import_button = PushButton("导入诊断文件", self)
        self.import_button.setObjectName("importWindowDebugButton")
        self.import_button.setToolTip("选择之前保存的 report.json，重新诊断其中的截图")
        self.import_button.clicked.connect(self.import_report)
        self.export_button = PushButton("导出诊断文件", self)
        self.export_button.setObjectName("exportWindowReportButton")
        self.export_button.setToolTip("仅在你选择保存路径后导出当前 JSON 报告；默认不保存截图。")
        self.export_button.clicked.connect(self.export_report)
        root.addWidget(controls)

        policy = CardWidget(self)
        policy_layout = QHBoxLayout(policy)
        policy_layout.setContentsMargins(16, 10, 16, 10)
        self.policy_label = BodyLabel(
            "只读诊断不控制目标窗口；问题包仅存本机，可选包含截图或仅日志，不自动上传。"
        )
        self.policy_label.setWordWrap(True)
        policy_layout.addWidget(self.policy_label)
        root.addWidget(policy)
        root.addWidget(self._build_session_frames_panel())

        root.addWidget(self._build_compact_result_panel())

        splitter = QSplitter(Qt.Orientation.Horizontal, self)
        splitter.setChildrenCollapsible(False)
        splitter.addWidget(self._build_status_panel())
        splitter.addWidget(self._build_report_panel())
        splitter.setStretchFactor(0, 1)
        splitter.setStretchFactor(1, 2)
        self.details_toggle = QToolButton(content)
        self.details_toggle.setObjectName("windowDebugDetailsToggle")
        self.details_toggle.setText("查看完整诊断详情")
        self.details_toggle.setCheckable(True)
        self.details_toggle.setChecked(False)
        self.details_toggle.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextOnly)
        self.details_toggle.toggled.connect(self._set_details_visible)
        root.addWidget(self.details_toggle)

        self.details_container = QWidget(content)
        self.details_container.setObjectName("windowDebugDetailsContainer")
        details_layout = QVBoxLayout(self.details_container)
        details_layout.setContentsMargins(0, 0, 0, 0)
        details_layout.setSpacing(8)
        advanced_actions = QHBoxLayout()
        advanced_actions.addWidget(self.import_button)
        advanced_actions.addWidget(self.export_button)
        advanced_actions.addStretch(1)
        details_layout.addLayout(advanced_actions)
        details_layout.addWidget(splitter, 1)
        self.details_container.setVisible(False)
        root.addWidget(self.details_container)

        scroll.setWidget(content)
        page_layout.addWidget(scroll)

    def _build_compact_result_panel(self) -> QWidget:
        panel = CardWidget(self)
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(14, 10, 14, 10)
        layout.setSpacing(4)
        layout.addWidget(StrongBodyLabel("诊断结论", panel))
        self.compact_status_label = SubtitleLabel("尚未诊断", panel)
        self.compact_status_label.setObjectName("compactDiagnosticStatusLabel")
        layout.addWidget(self.compact_status_label)
        self.compact_facts_label = BodyLabel("请选择一张截图，点击“诊断当前截图”。", panel)
        self.compact_facts_label.setObjectName("compactDiagnosticFactsLabel")
        self.compact_facts_label.setWordWrap(True)
        layout.addWidget(self.compact_facts_label)
        self.compact_issue_label = BodyLabel("", panel)
        self.compact_issue_label.setObjectName("compactDiagnosticIssueLabel")
        self.compact_issue_label.setWordWrap(True)
        layout.addWidget(self.compact_issue_label)
        self.compact_next_label = CaptionLabel("", panel)
        self.compact_next_label.setObjectName("compactDiagnosticNextLabel")
        self.compact_next_label.setWordWrap(True)
        layout.addWidget(self.compact_next_label)
        return panel

    def _set_details_visible(self, visible: bool) -> None:
        self.details_container.setVisible(bool(visible))
        self.details_toggle.setText(
            "收起完整诊断详情" if visible else "查看完整诊断详情"
        )

    def _build_session_frames_panel(self) -> QWidget:
        panel = CardWidget(self)
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(16, 12, 16, 12)
        layout.setSpacing(8)

        layout.addWidget(StrongBodyLabel("对局截图诊断"))
        directory_row = QHBoxLayout()
        self.choose_session_button = PushButton("选择对局目录", panel)
        self.choose_session_button.setToolTip("可选择 case 目录、frames 目录或旧对局的 diagnostic_frames 目录")
        self.choose_session_button.setObjectName("chooseSessionDirectoryButton")
        self.choose_session_button.clicked.connect(self.choose_session_directory)
        directory_row.addWidget(self.choose_session_button)
        self.refresh_frames_button = PushButton("刷新截图", panel)
        self.refresh_frames_button.setObjectName("refreshSessionFramesButton")
        self.refresh_frames_button.clicked.connect(self.refresh_session_frames)
        directory_row.addWidget(self.refresh_frames_button)
        self.session_directory_label = CaptionLabel("尚未选择对局目录", panel)
        self.session_directory_label.setWordWrap(True)
        directory_row.addWidget(self.session_directory_label, 1)
        layout.addLayout(directory_row)

        frame_row = QHBoxLayout()
        frame_row.addWidget(StrongBodyLabel("截图："))
        self.frame_selector = QComboBox(panel)
        self.frame_selector.setObjectName("sessionFrameSelector")
        self.frame_selector.currentIndexChanged.connect(self._session_frame_changed)
        frame_row.addWidget(self.frame_selector, 1)
        self.frame_index_label = CaptionLabel("0 / 0", panel)
        frame_row.addWidget(self.frame_index_label)
        self.previous_frame_button = PushButton("上一张", panel)
        self.previous_frame_button.setObjectName("previousSessionFrameButton")
        self.previous_frame_button.clicked.connect(self.previous_session_frame)
        frame_row.addWidget(self.previous_frame_button)
        self.next_frame_button = PushButton("下一张", panel)
        self.next_frame_button.setObjectName("nextSessionFrameButton")
        self.next_frame_button.clicked.connect(self.next_session_frame)
        frame_row.addWidget(self.next_frame_button)
        self.recognize_frame_button = PrimaryPushButton("识别当前截图", panel)
        self.recognize_frame_button.setObjectName("recognizeSessionFrameButton")
        self.recognize_frame_button.clicked.connect(self.recognize_current_session_frame)
        frame_row.addWidget(self.recognize_frame_button)
        layout.addLayout(frame_row)

        preview_row = QHBoxLayout()
        self.frame_preview = QLabel("选择对局或 frames 目录后查看诊断截图", panel)
        self.frame_preview.setObjectName("sessionFramePreview")
        self.frame_preview.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.frame_preview.setMinimumHeight(120)
        self.frame_preview.setStyleSheet("border:1px solid palette(mid);")
        preview_row.addWidget(self.frame_preview, 2)
        self.frame_metadata_view = QPlainTextEdit(panel)
        self.frame_metadata_view.setObjectName("sessionFrameMetadataView")
        self.frame_metadata_view.setReadOnly(True)
        self.frame_metadata_view.setLineWrapMode(QPlainTextEdit.LineWrapMode.NoWrap)
        preview_row.addWidget(self.frame_metadata_view, 1)
        layout.addLayout(preview_row)

        # Stable aliases make the page easy to drive from compatibility tests
        # and from the future session-diagnostic controller.
        self.session_path_label = self.session_directory_label
        self.session_frame_combo = self.frame_selector
        self.prev_frame_button = self.previous_frame_button
        self.next_frame_button = self.next_frame_button
        self.diagnose_session_frame_button = self.recognize_frame_button
        return panel

    def _build_status_panel(self) -> QWidget:
        panel = QWidget(self)
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(10)

        status_group = QGroupBox("窗口状态", panel)
        status_layout = QVBoxLayout(status_group)
        self.status_label = QLabel("尚未选择窗口", status_group)
        self.status_label.setWordWrap(True)
        status_layout.addWidget(self.status_label)
        self.status_view = QPlainTextEdit(status_group)
        self.status_view.setObjectName("windowStatusView")
        self.status_view.setReadOnly(True)
        self.status_view.setLineWrapMode(QPlainTextEdit.LineWrapMode.NoWrap)
        status_layout.addWidget(self.status_view, 1)
        layout.addWidget(status_group, 3)

        blockers_group = QGroupBox("关键阻塞原因", panel)
        blockers_layout = QVBoxLayout(blockers_group)
        self.blockers_view = QListWidget(blockers_group)
        self.blockers_view.setObjectName("keyBlockersView")
        self.blockers_view.setWordWrap(True)
        self.blockers_view.setSpacing(4)
        blockers_layout.addWidget(self.blockers_view)
        layout.addWidget(blockers_group, 2)
        return panel

    def _build_report_panel(self) -> QWidget:
        panel = QWidget(self)
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(10)
        layout.addWidget(StrongBodyLabel("中文诊断摘要"))
        self.summary_view = QPlainTextEdit(panel)
        self.summary_view.setObjectName("windowDebugSummaryView")
        self.summary_view.setReadOnly(True)
        self.summary_view.setLineWrapMode(QPlainTextEdit.LineWrapMode.WidgetWidth)
        layout.addWidget(self.summary_view, 2)

        layout.addWidget(StrongBodyLabel("详细识别结果"))
        self.detailed_view = QPlainTextEdit(panel)
        self.detailed_view.setObjectName("windowDebugDetailedRecognitionView")
        self.detailed_view.setReadOnly(True)
        self.detailed_view.setLineWrapMode(QPlainTextEdit.LineWrapMode.WidgetWidth)
        layout.addWidget(self.detailed_view, 4)
        # Compatibility alias for callers/tests that use the longer name.
        self.detailed_recognition_view = self.detailed_view

        copy_row = QHBoxLayout()
        self.copy_issue_button = PushButton("复制当前问题", panel)
        self.copy_issue_button.setObjectName("copyDiagnosticIssueButton")
        self.copy_issue_button.clicked.connect(lambda: self._copy_user_view("可复制错误说明", "当前没有可复制的问题"))
        copy_row.addWidget(self.copy_issue_button)
        self.copy_summary_button = PushButton("复制诊断摘要", panel)
        self.copy_summary_button.setObjectName("copyDiagnosticSummaryButton")
        self.copy_summary_button.clicked.connect(lambda: self._copy_user_view("可复制诊断摘要", "当前没有可复制的摘要"))
        copy_row.addWidget(self.copy_summary_button)
        self.copy_technical_button = PushButton("复制技术详情", panel)
        self.copy_technical_button.setObjectName("copyDiagnosticTechnicalButton")
        self.copy_technical_button.clicked.connect(lambda: self._copy_user_view("技术详情", "当前没有技术详情"))
        copy_row.addWidget(self.copy_technical_button)
        layout.addLayout(copy_row)

        layout.addWidget(StrongBodyLabel("技术详情（原始 JSON）"))
        self.json_view = QPlainTextEdit(panel)
        self.json_view.setObjectName("windowDebugJsonView")
        self.json_view.setReadOnly(True)
        self.json_view.setLineWrapMode(QPlainTextEdit.LineWrapMode.NoWrap)
        layout.addWidget(self.json_view, 3)
        return panel

    @staticmethod
    def _frame_value(value: object, *names: str, default: object = None) -> object:
        if isinstance(value, Mapping):
            for name in names:
                if name in value and value[name] is not None:
                    return value[name]
        for name in names:
            candidate = getattr(value, name, None)
            if candidate is not None:
                return candidate
        return default

    @classmethod
    def _normalize_session_frame(cls, value: object, root: Path, ordinal: int) -> dict[str, object] | None:
        if isinstance(value, (str, Path)):
            image_value = value
            raw: object = {"image_path": str(value)}
        else:
            raw = value
            image_value = cls._frame_value(
                value, "image_path", "image", "path", "frame_path", "png_path"
            )
        if image_value is None:
            return None
        image_path = Path(str(image_value)).expanduser()
        if not image_path.is_absolute():
            image_path = (root / image_path).resolve(strict=False)
        metadata_value = cls._frame_value(
            value, "metadata_path", "meta_path", "json_path", "sidecar_path"
        )
        metadata_path = (
            Path(str(metadata_value)).expanduser()
            if metadata_value is not None
            else image_path.with_suffix(".json")
        )
        if not metadata_path.is_absolute():
            metadata_path = (root / metadata_path).resolve(strict=False)
        if not image_path.is_file():
            return None
        sequence = cls._frame_value(
            value, "sequence", "capture_seq", "frame_index", "index", default=ordinal
        )
        captured_at = cls._frame_value(
            value, "captured_at", "timestamp", "created_at", default=""
        )
        return {
            "image_path": image_path,
            "metadata_path": metadata_path,
            "sequence": sequence,
            "captured_at": str(captured_at or ""),
            "raw": raw,
        }

    def _list_session_frames(self, session_directory: Path) -> list[dict[str, object]]:
        """Load frame records from the shared store with a filesystem fallback.

        The fallback only keeps the UI usable with older runtimes. Production
        uses ``SessionDiagnosticFrameStore.list_frames`` so its ordering and
        integrity rules remain authoritative.
        """
        values: object = None
        store_cls = SessionDiagnosticFrameStore
        if store_cls is not None:
            stores: list[object] = []
            for args, kwargs in (
                ((), {}),
                ((session_directory,), {}),
                ((), {"session_directory": session_directory}),
                ((), {"root": session_directory}),
            ):
                try:
                    stores.append(store_cls(*args, **kwargs))
                    break
                except TypeError:
                    continue
                except Exception:
                    break
            stores.append(store_cls)
            for store in stores:
                method = getattr(store, "list_frames", None)
                if not callable(method):
                    continue
                for args in ((), (session_directory,), (session_directory / "diagnostic_frames",)):
                    try:
                        values = method(*args)
                        break
                    except TypeError:
                        continue
                if values is not None:
                    break
        if values is None:
            frames_dir = session_directory / "diagnostic_frames"
            if (session_directory / "case.json").is_file():
                frames_dir = session_directory / "frames"
            elif session_directory.name.casefold() == "diagnostic_frames":
                frames_dir = session_directory
            values = sorted(frames_dir.glob("*.png")) if frames_dir.is_dir() else []

        if isinstance(values, Mapping):
            values = values.get("frames", values.get("items", []))
        if values is None or isinstance(values, (str, bytes)):
            values = []
        result: list[dict[str, object]] = []
        for ordinal, value in enumerate(values, 1):
            normalized = self._normalize_session_frame(value, session_directory, ordinal)
            if normalized is not None:
                result.append(normalized)
        def _sequence_key(item: dict[str, object]) -> tuple[int, object]:
            try:
                return (0, int(item.get("sequence", 0)))
            except (TypeError, ValueError):
                return (1, str(item.get("sequence", "")))

        result.sort(key=lambda item: (*_sequence_key(item), str(item["image_path"])))
        return result

    def choose_session_directory(self) -> Path | None:
        selected = QFileDialog.getExistingDirectory(self, "选择对局或 frames 目录", "")
        if not selected:
            return None
        try:
            return self.set_session_directory(Path(selected))
        except (OSError, ValueError) as exc:
            self.status_label.setText(f"无法选择截图目录：{exc}")
            return None

    def set_session_directory(self, directory: Path | str) -> Path:
        selected = Path(directory)
        if not selected.is_absolute() or selected == Path(selected.anchor) or ".." in selected.parts or "\x00" in str(selected):
            raise ValueError("请选择本机绝对目录，不支持 URL 或相对路径")
        if selected.name.casefold() == "frames" and not (selected.parent / "case.json").is_file():
            raise ValueError("frames 目录缺少所属对局的 case.json 标识")
        if selected.name.casefold() in {"diagnostic_frames", "frames"}:
            session_directory = selected.parent
        else:
            session_directory = selected
        self._session_directory = session_directory
        self.session_directory_label.setText(str(selected))
        self.refresh_session_frames()
        return session_directory

    def request_problem_export(self) -> None:
        # Pass the explicitly selected historical case even when it has no
        # frames. Only the controller may resolve current/recent when None.
        if self.problem_export_ui is None:
            self.policy_label.setText("当前运行时不支持问题包导出；不影响已有诊断功能。")
            return
        self.problem_export_ui.request(parent=self, case_directory=self._session_directory)

    def refresh_session_frames(self) -> None:
        load_error = ""
        self.frame_selector.blockSignals(True)
        try:
            self.frame_selector.clear()
            try:
                self._session_frames = (
                    self._list_session_frames(self._session_directory)
                    if self._session_directory is not None
                    else []
                )
            except Exception as exc:
                # Keep the selected historical case bound to export, but never
                # bypass a store integrity error with the legacy file scan.
                self._session_frames = []
                load_error = str(exc) or type(exc).__name__
            for index, frame in enumerate(self._session_frames, 1):
                sequence = frame.get("sequence", index)
                captured_at = str(frame.get("captured_at") or "").replace("T", " ")
                suffix = f" · {captured_at}" if captured_at else ""
                self.frame_selector.addItem(f"第{index}张 · {sequence}{suffix}", userData=index - 1)
        finally:
            self.frame_selector.blockSignals(False)
        if not self._session_frames:
            self._current_session_frame_index = -1
            self.frame_index_label.setText("0 / 0")
            self.frame_preview.setPixmap(QPixmap())
            self.frame_preview.setText("当前对局没有可用的诊断截图")
            self.frame_metadata_view.clear()
            self.previous_frame_button.setEnabled(False)
            self.next_frame_button.setEnabled(False)
            self.recognize_frame_button.setEnabled(False)
            self._clear_report_views("请选择一张实时监听帧后显式点击识别。")
            self.status_label.setText("无法加载截图：" + load_error if load_error else "截图为空或没有有效的 PNG 帧")
            return
        self.frame_selector.setCurrentIndex(0)
        self._session_frame_changed(0)

    def _session_frame_changed(self, index: int) -> None:
        if not self._session_frames:
            return
        data_index = self.frame_selector.itemData(index)
        try:
            frame_index = int(data_index if data_index is not None else index)
        except (TypeError, ValueError):
            frame_index = index
        if not 0 <= frame_index < len(self._session_frames):
            return
        self._current_session_frame_index = frame_index
        frame = self._session_frames[frame_index]
        image_path = Path(frame["image_path"])
        pixmap = QPixmap(str(image_path))
        self._current_preview_pixmap = pixmap
        if pixmap.isNull():
            self.frame_preview.setPixmap(QPixmap())
            self.frame_preview.setText("截图损坏或无法读取")
        else:
            self.frame_preview.setText("")
            self.frame_preview.setPixmap(pixmap.scaled(
                self.frame_preview.size(),
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation,
            ))
        metadata = {
            "来源": "诊断截图",
            "图片": str(image_path),
            "元数据": str(frame["metadata_path"]),
            "序号": frame.get("sequence"),
            "采集时间": frame.get("captured_at"),
        }
        metadata_path = Path(frame["metadata_path"])
        if metadata_path.is_file():
            try:
                metadata["metadata_json"] = json.loads(metadata_path.read_text(encoding="utf-8"))
            except Exception as exc:
                metadata["metadata_error"] = str(exc)
        saved_metadata = _mapping(metadata.get("metadata_json"))
        source = str(saved_metadata.get("source") or "")
        phase = str(saved_metadata.get("source_phase") or "")
        source_label = "手动窗口截图" if source == "manual_window_capture" else "实时监听帧"
        if phase in {"last_listener_frame", "failed_listener_frame", "waiting_capture_failed", "geometry_recovery_failed"}:
            source_label = "故障恢复截图"
        metadata["来源"] = source_label
        self.frame_metadata_view.setPlainText(_pretty_json(metadata))
        self.frame_index_label.setText(f"{frame_index + 1} / {len(self._session_frames)}")
        self.previous_frame_button.setEnabled(frame_index > 0)
        self.next_frame_button.setEnabled(frame_index + 1 < len(self._session_frames))
        self.recognize_frame_button.setEnabled(not pixmap.isNull())
        self._clear_report_views("当前截图尚未识别；点击“识别当前截图”开始。")
        self.status_label.setText(f"来源：{source_label} · 第 {frame_index + 1} 张")

    def previous_session_frame(self) -> None:
        if self._current_session_frame_index > 0:
            self.frame_selector.setCurrentIndex(self._current_session_frame_index - 1)

    def next_session_frame(self) -> None:
        if self._current_session_frame_index + 1 < len(self._session_frames):
            self.frame_selector.setCurrentIndex(self._current_session_frame_index + 1)

    def _clear_report_views(self, message: str = "尚未执行单帧诊断。") -> None:
        self._last_report = None
        self.summary_view.setPlainText(message + ("\n" if not message.endswith("\n") else ""))
        self.detailed_view.setPlainText(message + ("\n" if not message.endswith("\n") else ""))
        self.json_view.setPlainText("尚未执行单帧诊断。\n")
        self.compact_status_label.setText("尚未诊断")
        self.compact_facts_label.setText("请选择一张截图，点击“诊断当前截图”。")
        self.compact_issue_label.clear()
        self.compact_next_label.clear()
        self.details_toggle.setChecked(False)
        self.export_button.setEnabled(False)
        self.copy_issue_button.setEnabled(False)
        self.copy_summary_button.setEnabled(False)
        self.copy_technical_button.setEnabled(False)

    @staticmethod
    def _compact_display(value: object, default: str = "未识别") -> str:
        if value is None or value == "" or value == [] or value == ():
            return default
        if isinstance(value, (list, tuple)):
            return "、".join(str(item) for item in value) if value else default
        return str(value)

    def _update_compact_result(self, report: WindowReport, *, source_label: str) -> None:
        detailed = report.get("detailed_diagnostic") if isinstance(report, Mapping) else None
        detailed = detailed if isinstance(detailed, Mapping) else {}
        summary = detailed.get("summary")
        summary = summary if isinstance(summary, Mapping) else {}
        recognition = report.get("recognition") if isinstance(report, Mapping) else None
        result = recognition.get("result") if isinstance(recognition, Mapping) else None
        result = result if isinstance(result, Mapping) else {}
        blockers = detailed.get("blockers")
        if not isinstance(blockers, Sequence) or isinstance(blockers, (str, bytes)):
            blockers = _key_blockers(report)
        blockers = [str(item.get("message") or item) if isinstance(item, Mapping) else str(item) for item in blockers]
        recommendations = detailed.get("recommendations")
        if not isinstance(recommendations, Sequence) or isinstance(recommendations, (str, bytes)):
            recommendations = ()
        recommendations = [str(item) for item in recommendations if item]

        if report.get("errors") or not result:
            state = "无法识别"
        elif blockers:
            state = "部分识别"
        else:
            state = "识别完成"
        self.compact_status_label.setText(state)

        hand_count = summary.get("hand_count", len(result.get("my_hand", ()) or ()))
        expected = summary.get("expected_hand_count", 27)
        facts = " · ".join((
            f"级牌：{self._compact_display(summary.get('round_level', result.get('round_level')))}",
            f"手牌：{hand_count}/{expected} 张",
            f"首出：{self._compact_display(summary.get('lead_player', result.get('lead_player')))}",
            f"当前行动：{self._compact_display(summary.get('current_player', result.get('current_player')))}",
        ))
        self.compact_facts_label.setText(f"{facts}\n来源：{source_label}")

        if blockers:
            self.compact_issue_label.setText(f"主要问题：{blockers[0]}")
        else:
            self.compact_issue_label.setText("主要问题：未发现关键阻塞原因")
        self.compact_next_label.setText(
            f"下一步：{recommendations[0]}" if recommendations else "下一步：展开详情查看识别证据"
        )

    def _apply_report(self, report: WindowReport, *, source_label: str) -> None:
        self._last_report = report
        blockers = _key_blockers(report)
        self._update_compact_result(report, source_label=source_label)
        user_view = report.get("user_view") if isinstance(report, Mapping) else None
        if isinstance(user_view, Mapping):
            summary = str(user_view.get("可复制诊断摘要") or "")
            self.summary_view.setPlainText(summary or "已完成诊断，但没有中文摘要。")
            self.copy_issue_button.setEnabled(bool(user_view.get("可复制错误说明") or blockers))
            self.copy_summary_button.setEnabled(bool(user_view.get("可复制诊断摘要") or summary))
            self.copy_technical_button.setEnabled(bool(user_view.get("技术详情") or report))
        else:
            self.summary_view.setPlainText("已完成诊断，但当前报告没有中文摘要。")
            self.copy_issue_button.setEnabled(bool(blockers))
            self.copy_summary_button.setEnabled(True)
            self.copy_technical_button.setEnabled(True)
        self.detailed_view.setPlainText(_render_detailed_recognition(report, source_label))
        self.json_view.setPlainText(_pretty_json(report))
        self.blockers_view.clear()
        for blocker in blockers or ["未返回关键阻塞原因。"]:
            self.blockers_view.addItem(QListWidgetItem(blocker))
        self.export_button.setEnabled(True)
        self.policy_label.setText(f"当前诊断来源：{source_label} · 目标窗口未被控制")

    def recognize_current_session_frame(self) -> None:
        if not 0 <= self._current_session_frame_index < len(self._session_frames):
            self._show_error("无法识别", ValueError("请先选择一张实时监听帧"))
            return
        frame = self._session_frames[self._current_session_frame_index]
        builder = getattr(self.report_service, "build_from_listener_frame", None)
        if not callable(builder):
            self._show_error("识别实时监听帧失败", RuntimeError("当前诊断服务不支持实时监听帧重诊"))
            return
        self.recognize_frame_button.setEnabled(False)
        try:
            report = builder(
                Path(frame["image_path"]),
                metadata_path=Path(frame["metadata_path"]),
                recognize=True,
            )
            self._apply_report(report, source_label="实时监听帧（listener）")
            self.status_label.setText(f"来源：实时监听帧 · 第 {self._current_session_frame_index + 1} 张 · 识别完成")
        except Exception as exc:
            self._show_error("识别实时监听帧失败", exc)
        finally:
            self.recognize_frame_button.setEnabled(True)

    def _set_idle_state(self) -> None:
        self.window_selector.clear()
        self.status_label.setText("尚未选择窗口")
        self.status_view.clear()
        self.blockers_view.clear()
        self.blockers_view.addItem(QListWidgetItem("刷新窗口列表后选择一个 HWND。"))
        self._clear_report_views()
        self.diagnose_button.setEnabled(False)

    @staticmethod
    def _coerce_hwnd(value: object) -> int | None:
        try:
            return int(value) if value is not None else None
        except (TypeError, ValueError):
            return None

    def refresh_windows(self) -> None:
        """Refresh only the read-only window list exposed by the service."""

        self.refresh_button.setEnabled(False)
        try:
            records = list(self.report_service.list_windows())
        except Exception as exc:
            self._show_error("刷新窗口列表失败", exc)
            self._window_records.clear()
            self._set_idle_state()
            return
        finally:
            self.refresh_button.setEnabled(True)

        self._window_records.clear()
        self.window_selector.blockSignals(True)
        try:
            self.window_selector.clear()
            for record in records:
                hwnd = self._coerce_hwnd(record.get("hwnd"))
                if hwnd is None:
                    continue
                self._window_records[hwnd] = record
                self.window_selector.addItem(_window_label(record), userData=hwnd)
        finally:
            self.window_selector.blockSignals(False)

        self._clear_report_views()
        if self.window_selector.count() == 0:
            self.status_label.setText("未发现可见顶层窗口。")
            self.status_view.clear()
            self.blockers_view.clear()
            self.blockers_view.addItem(QListWidgetItem("没有可供选择的 HWND。"))
            self.diagnose_button.setEnabled(False)
            return

        self.window_selector.setCurrentIndex(0)
        self._window_selection_changed(0)

    def _window_selection_changed(self, index: int) -> None:
        hwnd = self._coerce_hwnd(self.window_selector.itemData(index))
        self._clear_report_views()
        self.diagnose_button.setEnabled(hwnd is not None)
        if hwnd is None:
            self.status_label.setText("尚未选择窗口")
            self.status_view.clear()
            return

        record = self._window_records.get(hwnd, {"hwnd": hwnd})
        try:
            status = self.report_service.probe(hwnd)
        except Exception as exc:
            status = {**dict(record), "probe_error": str(exc)}
            self.status_label.setText(f"HWND {hwnd} · 状态读取失败")
        else:
            self.status_label.setText(self._status_summary(status))
        self.status_view.setPlainText(_pretty_json(status))
        self.blockers_view.clear()
        self.blockers_view.addItem(QListWidgetItem("选择“诊断当前单帧”查看牌局识别阻塞原因。"))

    @staticmethod
    def _status_summary(status: WindowRecord) -> str:
        title = str(status.get("title") or "（无标题）")
        visible = "可见" if status.get("visible") is not False else "不可见"
        iconic = "，已最小化" if status.get("iconic") is True else ""
        return f"{title} · {visible}{iconic} · 窗口编号 {status.get('hwnd', '?')}"

    def selected_hwnd(self) -> int | None:
        return self._coerce_hwnd(self.window_selector.currentData())

    def diagnose_current_window(self, *, include_media: bool = False) -> None:
        """Run exactly one read-only report for the selected HWND."""

        hwnd = self.selected_hwnd()
        if hwnd is None:
            self._show_error("无法诊断", ValueError("请先选择 HWND"))
            return

        self.diagnose_button.setEnabled(False)
        try:
            if include_media:
                report = self.report_service.build(
                    hwnd=hwnd, capture=True, recognize=True, include_media=True
                )
                writer = getattr(self.report_service, "write", None)
                if callable(writer):
                    writer(report)
            else:
                # Preserve compatibility with injected legacy test/services
                # that do not yet expose the optional include_media keyword.
                report = self.report_service.build(
                    hwnd=hwnd, capture=True, recognize=True
                )
            self._apply_report(report, source_label="独立窗口截图（manual capture）")
            if include_media:
                self.status_label.setText("已截取当前画面并保存诊断文件")
        except Exception as exc:
            self._last_report = None
            self._show_error("单帧诊断失败", exc)
        finally:
            self.diagnose_button.setEnabled(hwnd is not None)

    def _copy_user_view(self, field: str, empty_message: str) -> None:
        report = self._last_report
        user_view = report.get("user_view") if isinstance(report, Mapping) else None
        value = user_view.get(field) if isinstance(user_view, Mapping) else None
        text = str(value or "")
        if not text and field == "可复制诊断摘要":
            text = self.summary_view.toPlainText().strip() + "\n\n" + self.detailed_view.toPlainText().strip()
        elif not text and field == "可复制错误说明":
            text = "\n".join(
                self.blockers_view.item(index).text()
                for index in range(self.blockers_view.count())
            ).strip()
        elif not text and field == "技术详情" and report is not None:
            text = _pretty_json(report)
        if not text:
            self.status_label.setText(empty_message)
            return
        QApplication.clipboard().setText(text)
        self.status_label.setText("已复制到剪贴板")

    def import_report(self) -> Path | None:
        """Re-diagnose an embedded screenshot from one saved report.json."""

        filename, _ = QFileDialog.getOpenFileName(
            self, "导入诊断文件", "", "诊断 JSON (*.json);;所有文件 (*)"
        )
        if not filename:
            return None
        try:
            builder = getattr(self.report_service, "build_from_report_file", None)
            if not callable(builder):
                raise RuntimeError("当前诊断服务不支持导入保存的诊断文件")
            report = builder(Path(filename), recognize=True)
            self._apply_report(report, source_label="保存报告截图（manual capture）")
            self.status_label.setText("已导入诊断文件并完成截图重诊")
            return Path(filename)
        except Exception as exc:
            self._show_error("导入诊断文件失败", exc)
            return None

    def export_report(self) -> Path | None:
        """Persist JSON only after the user explicitly selects a destination."""

        if self._last_report is None:
            return None
        filename, _ = QFileDialog.getSaveFileName(
            self,
            "导出窗口诊断文件",
            "window-debug-report.json",
            "JSON 文件 (*.json);;所有文件 (*)",
        )
        if not filename:
            return None
        path = Path(filename)
        try:
            path.write_text(_pretty_json(self._last_report) + "\n", encoding="utf-8")
        except Exception as exc:
            self._show_error("导出诊断报告失败", exc)
            return None
        self.policy_label.setText(
            f"已导出诊断文件：{path} · 截图仍未保存；目标窗口未被控制"
        )
        return path

    def _show_error(self, title: str, error: Exception) -> None:
        message = str(error) or type(error).__name__
        self.compact_status_label.setText("无法识别")
        self.compact_facts_label.setText("当前截图没有生成可用识别结果")
        self.compact_issue_label.setText(f"主要问题：{title}：{message}")
        self.compact_next_label.setText("下一步：检查截图来源、窗口状态或展开详情")
        self.status_label.setText(f"{title}：{message}")
        self.status_view.setPlainText(
            _pretty_json({"error": type(error).__name__, "message": message})
        )
        self.blockers_view.clear()
        self.blockers_view.addItem(QListWidgetItem(message))
        # Keep the page usable in headless tests and avoid modal dialogs for
        # service failures; the status and blocker panels are the entry point.


__all__ = ["WindowDebugPage", "WindowDebugReportPort"]

