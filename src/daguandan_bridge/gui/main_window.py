from __future__ import annotations

from PySide6.QtCore import QRect, Slot
from PySide6.QtWidgets import QApplication, QMessageBox, QSizePolicy
from collections.abc import Mapping
from qfluentwidgets import FluentIcon, FluentWindow

from .annotation_page import AnnotationPage
from .live_assistant_page import LiveAssistantPage
from .recommendation_window import RecommendationFloatWindow
from .replay_page import ReplayPage
from .window_debug_page import WindowDebugPage


def _readiness_field(report: object, name: str, default: object = None) -> object:
    if isinstance(report, Mapping):
        return report.get(name, default)
    return getattr(report, name, default)


def _enum_value(value: object) -> str:
    return str(getattr(value, "value", value) or "")


def _readiness_payload(report: object) -> dict[str, object]:
    to_dict = getattr(report, "to_dict", None)
    if callable(to_dict):
        try:
            payload = to_dict()
        except Exception:
            payload = None
        if isinstance(payload, Mapping):
            return dict(payload)
    return {
        "status": _enum_value(_readiness_field(report, "status", "")),
        "primary_reason": _enum_value(
            _readiness_field(
                report,
                "primary_reason",
                _readiness_field(report, "error_code", ""),
            )
        ),
        "message": str(_readiness_field(report, "message", "") or ""),
        "suggested_action": str(
            _readiness_field(report, "suggested_action", "") or ""
        ),
        "hard_error": bool(_readiness_field(report, "hard_error", False)),
    }


class DaguandanBridgeWindow(FluentWindow):
    def __init__(
        self,
        *,
        dependencies=None,
        live_runtime=None,
        window_debug_service=None,
    ) -> None:
        super().__init__()
        if dependencies is None:
            from ..bootstrap import build_application_dependencies

            dependencies = build_application_dependencies(live_runtime=live_runtime)
        self.live_runtime = live_runtime or dependencies.live_runtime
        self.annotation_page = AnnotationPage(dependencies.annotation_service)
        self.annotation_page.setObjectName("annotationPage")
        self.live_assistant_page = LiveAssistantPage(self.live_runtime)
        self.replay_page = ReplayPage(dependencies.sessions_root)
        self.window_debug_page = WindowDebugPage(
            report_service=window_debug_service,
        )
        self.recommendation_window = RecommendationFloatWindow(self.live_runtime)
        self.live_assistant_page.compact_mode_requested.connect(
            self.show_compact_recommendation
        )
        self.recommendation_window.open_full_assistant_requested.connect(
            self.show_full_assistant
        )
        self.recommendation_window.open_diagnostic_requested.connect(
            self.open_window_debug
        )
        self.recommendation_window.capture_diagnostic_requested.connect(
            self.capture_diagnostic_from_compact
        )
        self.recommendation_window.copy_issue_requested.connect(
            self.copy_compact_issue
        )
        self.recommendation_window.copy_summary_requested.connect(
            self.copy_compact_summary
        )
        self.recommendation_window.stop_listening_requested.connect(
            self.stop_live_listening
        )

        self.addSubInterface(
            self.live_assistant_page,
            FluentIcon.ROBOT,
            "实时助手",
        )
        self.addSubInterface(
            self.annotation_page,
            FluentIcon.EDIT,
            "标记与模板",
        )
        self.addSubInterface(
            self.replay_page,
            FluentIcon.VIDEO,
            "对局回放",
        )
        self.addSubInterface(
            self.window_debug_page,
            FluentIcon.SEARCH,
            "窗口诊断",
        )
        # The full assistant remains the startup surface. Compact mode is still
        # an explicit action from the live-assistant page.
        self._interfaces = (
            self.live_assistant_page,
            self.annotation_page,
            self.replay_page,
            self.window_debug_page,
        )
        self.stackedWidget.currentChanged.connect(
            self._sync_interface_size_policies
        )
        self._sync_interface_size_policies()
        self.setWindowTitle("大掼蛋智能助手")
        self.resize(1220, 820)
        self.setMinimumSize(980, 700)

    @Slot(int)
    def _sync_interface_size_policies(self, _index: int = -1) -> None:
        """Keep hidden navigation pages from imposing their full height.

        ``QStackedWidget`` calculates its size hint from all child pages.  The
        annotation page contains a tall editor surface, so it used to force
        the whole frameless window to a height of about 1385px even while the
        replay page was active.  On a 1080px display, maximizing then laid out
        the replay page outside the available client area and made its header
        appear clipped.  Only the current page should contribute a vertical
        size constraint; inactive pages still expand normally when selected.
        """
        current = self.stackedWidget.currentWidget()
        for interface in self._interfaces:
            policy = interface.sizePolicy()
            policy.setVerticalPolicy(
                QSizePolicy.Policy.Preferred
                if interface is current
                else QSizePolicy.Policy.Ignored
            )
            interface.setSizePolicy(policy)
        self.stackedWidget.updateGeometry()
        self.updateGeometry()

    def show_compact_recommendation(self) -> None:
        # Readiness is the safety gate for the explicit compact-window action.
        # Older runtimes may not expose it, so absence preserves compatibility.
        readiness = getattr(self.live_runtime, "opening_readiness", None)
        if readiness is not None:
            hard_error = bool(_readiness_field(readiness, "hard_error", False))
            if hard_error:
                self.show_full_assistant()
                message = str(_readiness_field(readiness, "message", "") or "")
                suggested_action = str(
                    _readiness_field(readiness, "suggested_action", "") or ""
                )
                reason = _enum_value(
                    _readiness_field(
                        readiness,
                        "primary_reason",
                        _readiness_field(readiness, "error_code", ""),
                    )
                )
                details = [part for part in (message, suggested_action) if part]
                if reason:
                    details.append(f"原因码：{reason}")
                QMessageBox.warning(
                    self,
                    "暂不显示极简窗口",
                    "\n\n".join(details) or "当前监听状态不允许显示极简窗口。",
                )
                return

            status = _enum_value(_readiness_field(readiness, "status", ""))
            if status == "WAIT":
                # Keep the structured WAIT reason visible in the diagnostic
                # compact surface instead of reducing it to a generic title.
                self._last_compact_readiness = readiness
                apply_status = getattr(
                    self.recommendation_window, "apply_listening_status", None
                )
                if callable(apply_status):
                    apply_status(_readiness_payload(readiness))

        try:
            rect = self.live_runtime.target_client_rect()
            # WAIT is not safe for recommendations, but it is safe for the
            # explanatory diagnostic compact surface (for example, settlement
            # page / waiting for the next deal). Use a separate geometry API so
            # this cannot bypass the recommendation safety gate.
            if rect is None:
                status = _enum_value(_readiness_field(readiness, "status", "")) if readiness is not None else ""
                diagnostic_rect = getattr(self.live_runtime, "diagnostic_target_client_rect", None)
                if status == "WAIT" and callable(diagnostic_rect):
                    rect = diagnostic_rect()
        except Exception:
            rect = None
        if rect is None:
            self.show_full_assistant()
            QMessageBox.information(
                self, "无法进入极简窗口",
                "未能获得掼蛋窗口坐标，已保留完整助手窗口。",
            )
            return
        placed = self.recommendation_window.place_beside(
            QRect(rect.left, rect.top, rect.width, rect.height)
        )
        if not placed:
            self.show_full_assistant()
            QMessageBox.information(
                self, "无法进入极简窗口",
                "当前屏幕工作区没有不遮挡掼蛋窗口的安全位置，已保留完整助手窗口。",
            )
            return
        self.recommendation_window.show()
        self.recommendation_window.raise_()
        self.showMinimized()

    def open_window_debug(self) -> None:
        """Show the Chinese diagnostic page from the compact icon bar."""

        self.show_full_assistant()
        self.switchTo(self.window_debug_page)

    def capture_diagnostic_from_compact(self) -> None:
        """Save the exact latest live-listener frame without leaving compact mode."""

        apply_status = getattr(self.recommendation_window, "apply_diagnostic_frame_status", None)
        if callable(apply_status):
            apply_status({"status": "SAVING"})
        background_saver = getattr(
            self.live_runtime,
            "save_latest_live_frame_to_session_background",
            None,
        )
        if callable(background_saver):
            try:
                background_saver()
            except Exception as exc:
                if callable(apply_status):
                    apply_status({"status": "FAILURE", "message": str(exc) or type(exc).__name__})
            return

        saver = getattr(self.live_runtime, "save_latest_live_frame_to_session", None)
        if not callable(saver):
            if callable(apply_status):
                apply_status({"status": "FAILURE", "message": "当前运行时不支持保存监听截图"})
            return

        try:
            result = saver()
        except Exception as exc:
            if callable(apply_status):
                apply_status({"status": "FAILURE", "message": str(exc) or type(exc).__name__})
            return
        # Production runtimes emit diagnostic_frame_status asynchronously. A
        # returned structured result is also accepted for small/fake runtimes.
        if isinstance(result, Mapping) and callable(apply_status):
            payload = dict(result)
            payload.setdefault("status", "SUCCESS")
            apply_status(payload)

    def _compact_copy_text(self, *, summary: bool) -> str:
        report = getattr(self.live_runtime, "opening_readiness", None)
        if callable(report):
            try:
                report = report()
            except Exception:
                report = None
        message = str(_readiness_field(report, "message", "当前没有可复制的诊断问题") or "当前没有可复制的诊断问题")
        action = str(_readiness_field(report, "suggested_action", "") or "")
        reason = _enum_value(_readiness_field(report, "primary_reason", _readiness_field(report, "error_code", "")))
        if summary:
            return "\n".join((
                "大掼蛋窗口与牌局诊断摘要",
                f"状态：{_enum_value(_readiness_field(report, 'status', '')) or '未知'}",
                f"问题：{message}",
                f"下一步：{action}" if action else "",
                f"原因码：{reason}" if reason else "",
            )).strip()
        return "\n".join((f"问题：{message}", f"下一步：{action}" if action else "", f"原因码：{reason}" if reason else "")).strip()

    def copy_compact_issue(self) -> None:
        QApplication.clipboard().setText(self._compact_copy_text(summary=False))
        apply_status = getattr(self.recommendation_window, "apply_diagnostic_frame_status", None)
        if callable(apply_status):
            apply_status({"status": "SUCCESS", "message": "当前问题已复制到剪贴板"})

    def copy_compact_summary(self) -> None:
        QApplication.clipboard().setText(self._compact_copy_text(summary=True))
        apply_status = getattr(self.recommendation_window, "apply_diagnostic_frame_status", None)
        if callable(apply_status):
            apply_status({"status": "SUCCESS", "message": "诊断摘要已复制到剪贴板"})

    def stop_live_listening(self) -> None:
        stopper = getattr(self.live_runtime, "stop_listening", None)
        if callable(stopper):
            stopper()
        self.show_full_assistant()

    def show_full_assistant(self) -> None:
        self.recommendation_window.hide()
        self.showNormal()
        self.raise_()
        self.activateWindow()

    def closeEvent(self, event) -> None:
        self.recommendation_window.hide()
        self.replay_page.shutdown()
        self.live_assistant_page.shutdown()
        super().closeEvent(event)
