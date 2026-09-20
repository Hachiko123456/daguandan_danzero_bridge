from __future__ import annotations

import inspect
from pathlib import Path
from threading import Event
from typing import Callable

from PySide6.QtCore import QEvent, QThread, Qt, QUrl, Signal, Slot
from PySide6.QtGui import QDesktopServices, QGuiApplication
from PySide6.QtWidgets import QFileDialog, QHBoxLayout, QProgressBar, QVBoxLayout, QWidget
from qfluentwidgets import BodyLabel, CardWidget, PrimaryPushButton, PushButton, StrongBodyLabel

from ..application.offline_diagnostic_replay import OfflineDiagnosticReplayService
from ..config import PROFILES_ROOT


# A running QThread must outlive its widget, even if a host deletes the page
# rather than delivering a close event. No GUI object is retained by the worker.
_ACTIVE_THREADS: set[OfflineDiagnosticReplayThread] = set()


class OfflineDiagnosticReplayThread(QThread):
    """Call the service once; preserve cancellation and report-bearing failures."""

    completed = Signal(object)
    failed = Signal(object)
    progress = Signal(object)

    def __init__(self, service: object, input_path: Path, *, environment: dict) -> None:
        super().__init__()  # Do not destroy a running worker with its QWidget parent.
        self.service = service
        self.input_path = Path(input_path)
        self.environment = environment
        self.stop_event = Event()
        try:
            self._parameters = inspect.signature(service.run).parameters
        except (TypeError, ValueError):
            self._parameters = None
        self.supports_cancel = self._accepts("stop_requested")

    def _accepts(self, name: str) -> bool:
        return (
            self._parameters is None
            or name in self._parameters
            or any(p.kind is inspect.Parameter.VAR_KEYWORD for p in self._parameters.values())
        )

    @staticmethod
    def _progress_values(*args: object, **kwargs: object) -> dict:
        if len(args) == 1 and isinstance(args[0], dict):
            return dict(args[0])
        if isinstance(kwargs.get("progress"), dict):
            return dict(kwargs["progress"])
        data = dict(kwargs)
        data.update(zip(("processed", "total", "frame_index"), args))
        return data

    def run(self) -> None:
        try:
            values = {
                "on_progress": lambda *args, **kwargs: self.progress.emit(
                    self._progress_values(*args, **kwargs)
                ),
                "stop_requested": self.stop_event.is_set,
                "environment": self.environment,
            }
            positional = [self.input_path]
            kwargs = {}
            for name, value in values.items():
                if not self._accepts(name):
                    continue  # Only explicitly old stub signatures omit options.
                parameter = (self._parameters or {}).get(name)
                if parameter is not None and parameter.kind is inspect.Parameter.POSITIONAL_ONLY:
                    positional.append(value)
                else:
                    kwargs[name] = value
            # Never retry TypeError: it can originate inside an already running
            # service. A real interface always receives all three options.
            self.completed.emit(self.service.run(*positional, **kwargs))
        except Exception as exc:
            self.failed.emit(exc)

    @Slot()
    def release(self) -> None:
        _ACTIVE_THREADS.discard(self)
        self.deleteLater()


def _safe_int(value: object, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError, OverflowError):
        return default


def _path(value: object) -> Path | None:
    if value is None or value == "":
        return None
    try:
        return Path(str(value)).expanduser().resolve()
    except (OSError, ValueError):
        return None


def _data(value: object) -> dict:
    if isinstance(value, dict):
        return value
    try:
        data = value.to_dict() if hasattr(value, "to_dict") else {}
    except Exception:
        data = {}
    data = dict(data) if isinstance(data, dict) else {}
    for key in ("status", "report_path", "output_directory", "modes"):
        if key not in data and hasattr(value, key):
            data[key] = getattr(value, key)
    return data


def _mode_label(mode: object) -> str:
    name = str(mode).lower()
    if "async" in name or "latest" in name or "realtime" in name:
        return "原速复测（1×）"
    if "sync" in name or "frame" in name:
        return "逐帧对照"
    return "离线复测"


def _status_label(status: object) -> str:
    return {
        "complete": "已完成", "completed": "已完成", "passed": "已完成",
        "success": "已完成", "failed": "有问题", "error": "有问题",
        "incomplete": "未通过（未完成）", "cancelled": "已取消",
        "canceled": "已取消", "skipped": "未执行", "not_run": "未执行",
        "not_started": "未执行", "running": "进行中",
    }.get(str(status).lower(), "状态待确认")


def _modes(data: dict) -> list[tuple[str, dict]]:
    modes = data.get("modes")
    if isinstance(modes, dict):
        return [(str(name), _data(result)) for name, result in modes.items()]
    if isinstance(modes, list):
        return [(str(item.get("mode", "")), item) for item in modes if isinstance(item, dict)]
    return []


class OfflineDiagnosticPanel(CardWidget):
    """Two-mode diagnostic UI; application service owns replay policy and files."""

    busy_changed = Signal(bool)
    SHUTDOWN_WAIT_MS = 150

    def __init__(
        self, *, profiles_root: Path | str = PROFILES_ROOT,
        profile_name: str = "tencent_daguandan",
        service_factory: Callable[..., object] | None = None,
        can_start: Callable[[], bool] | None = None,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.setObjectName("offlineDiagnosticPanel")
        self.profiles_root = Path(profiles_root).expanduser().resolve()
        self.profile_name = str(profile_name)
        self._service_factory = service_factory or OfflineDiagnosticReplayService
        self._can_start = can_start or (lambda: True)
        self._thread: OfflineDiagnosticReplayThread | None = None
        self._cancel_requested = False
        self._report_path: Path | None = None
        self._output_directory: Path | None = None
        self._close_target: QWidget | None = None
        self._build_ui()

    @property
    def is_running(self) -> bool:
        # Keep ownership until queued completion/finished signals are consumed.
        return self._thread is not None

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(16, 12, 16, 12)
        layout.setSpacing(8)
        header = QHBoxLayout()
        header.addWidget(StrongBodyLabel("离线诊断"))
        self.hint_label = BodyLabel("按原速复测，再逐帧对照；无需新开局")
        self.hint_label.setWordWrap(True)
        header.addWidget(self.hint_label, 1)
        layout.addLayout(header)

        buttons = QHBoxLayout()
        self.import_button = PrimaryPushButton("导入诊断并测试")
        self.import_button.setToolTip("选择完整诊断 ZIP 后开始复测；输出另存，不改原对局")
        self.directory_button = PushButton("导入目录…")
        self.cancel_button = PushButton("停止")
        self.cancel_button.setEnabled(False)
        self.cancel_button.setVisible(False)
        self.report_button = PushButton("打开报告")
        self.report_button.setVisible(False)
        self.directory_open_button = PushButton("打开报告目录")
        self.directory_open_button.setVisible(False)
        buttons.addWidget(self.import_button)
        buttons.addWidget(self.directory_button)
        buttons.addStretch(1)
        buttons.addWidget(self.cancel_button)
        buttons.addWidget(self.report_button)
        buttons.addWidget(self.directory_open_button)
        layout.addLayout(buttons)

        self.status_label = BodyLabel("导入完整诊断 ZIP 或对局目录即可开始")
        self.status_label.setWordWrap(True)
        self.status_label.setTextFormat(Qt.TextFormat.PlainText)
        layout.addWidget(self.status_label)
        self.progress_bar = QProgressBar()
        self.progress_bar.setObjectName("offlineDiagnosticProgress")
        self.progress_bar.setRange(0, 100)
        self.progress_bar.hide()
        layout.addWidget(self.progress_bar)
        self.summary_label = BodyLabel("")
        self.summary_label.setWordWrap(True)
        self.summary_label.setTextFormat(Qt.TextFormat.PlainText)
        self.summary_label.hide()
        layout.addWidget(self.summary_label)

        self.import_button.clicked.connect(self._choose_zip)
        self.directory_button.clicked.connect(self._choose_directory)
        self.cancel_button.clicked.connect(self.request_cancel)
        self.report_button.clicked.connect(self.open_report)
        self.directory_open_button.clicked.connect(self.open_report_directory)

    def _choose_zip(self) -> None:
        if self.is_running:
            return
        path, _ = QFileDialog.getOpenFileName(self, "选择完整诊断 ZIP", "", "诊断 ZIP (*.zip)")
        if path:
            self.start(path)

    def _choose_directory(self) -> None:
        if self.is_running:
            return
        path = QFileDialog.getExistingDirectory(self, "选择对局目录")
        if path:
            self.start(path)

    def start(self, input_path: Path | str) -> bool:
        if self.is_running:
            return False
        if not self._can_start():
            self.status_label.setText("请先停止实时监听，或等待本页其他扫描完成。")
            return False
        self._report_path = None
        self._output_directory = None
        self.report_button.hide()
        self.directory_open_button.hide()
        self.summary_label.setToolTip("")
        try:
            service = self._service_factory(
                profiles_root=self.profiles_root, profile_name=self.profile_name,
            )
            thread = OfflineDiagnosticReplayThread(
                service, Path(input_path).expanduser().resolve(),
                environment=self._host_environment(),
            )
        except Exception as exc:
            self._on_failed(exc)
            return False

        self._thread = thread
        self._cancel_requested = False
        self._report_path = None
        self._output_directory = None
        self._close_target = None
        # Install on the containing window too: child widgets do not receive
        # their parent's close event. Veto destruction until cooperation ends.
        self.window().installEventFilter(self)
        self.summary_label.clear()
        self.summary_label.hide()
        self.report_button.hide()
        self.progress_bar.setRange(0, 0)
        self.progress_bar.show()
        self.status_label.setText("正在准备原速复测，随后逐帧对照…")
        self._set_running(True)
        queued = Qt.ConnectionType.QueuedConnection
        thread.progress.connect(self._on_progress, queued)
        thread.completed.connect(self._on_completed, queued)
        thread.failed.connect(self._on_failed, queued)
        thread.finished.connect(self._on_thread_finished, queued)
        thread.finished.connect(thread.release, queued)
        _ACTIVE_THREADS.add(thread)
        self.busy_changed.emit(True)
        thread.start()
        return True

    @staticmethod
    def _host_environment() -> dict:
        """All Qt display facts are host metadata, never screenshot validation."""
        def rect(value) -> dict:
            return {"x": value.x(), "y": value.y(), "width": value.width(), "height": value.height()}

        primary = QGuiApplication.primaryScreen()
        return {
            "source": "host_environment",
            "note": "宿主 Qt 显示环境，不是验证截图，也不证明录像采集环境。",
            "screens": [
                {
                    "name": screen.name(), "primary": screen == primary,
                    "geometry": rect(screen.geometry()),
                    "availableGeometry": rect(screen.availableGeometry()),
                    "dpr": float(screen.devicePixelRatio()),
                }
                for screen in QGuiApplication.screens()
            ],
        }

    def _remember_paths(self, data: dict) -> None:
        self._report_path = _path(data.get("report_path")) or self._report_path
        self._output_directory = _path(data.get("output_directory")) or self._output_directory
        self.report_button.setVisible(self._report_target() is not None)
        self.directory_open_button.setVisible(self._directory_target() is not None)

    @Slot(object)
    def _on_progress(self, data: dict) -> None:
        self._remember_paths(data)
        processed = max(0, _safe_int(data.get("processed")))
        total = max(0, _safe_int(data.get("total")))
        if total:
            self.progress_bar.setRange(0, 100)
            self.progress_bar.setValue(min(100, processed * 100 // total))
            detail = f"{processed}/{total} 帧"
        else:
            self.progress_bar.setRange(0, 0)
            detail = "准备中"
        frame = _safe_int(data.get("frame_index"), -1)
        if frame >= 0:
            detail += f" · 当前帧 {frame}"
        prefix = "正在停止，等待报告保存" if self._cancel_requested else "正在复测"
        # Do not expose internal phase names or raw production backend messages.
        self.status_label.setText(f"{prefix} · {_mode_label(data.get('mode'))} · {detail}")

    @Slot(object)
    def _on_completed(self, result: object) -> None:
        data = _data(result)
        self._remember_paths(data)
        status = str(data.get("status", "unknown")).lower()
        mode_statuses = [str(item.get("status", "unknown")).lower() for _, item in _modes(data)]
        if status in {"failed", "error"} or any(s in {"failed", "error"} for s in mode_statuses):
            text = "复测有问题，未通过；请查看报告"
        elif status == "incomplete" or "incomplete" in mode_statuses:
            text = "复测未通过：结果不完整；请查看报告"
        elif status in {"cancelled", "canceled"} or any(s in {"cancelled", "canceled"} for s in mode_statuses):
            text = "复测已取消；可查看已生成的报告"
        elif status in {"complete", "completed", "passed", "success"}:
            text = "复测已完成；各轮结果如下，请查看报告确认差异"
        else:
            text = "复测结果待确认；请查看报告"
        self.status_label.setText(text)
        self.summary_label.setText(self._summary_text(data))
        self.summary_label.show()

    @Slot(object)
    def _on_failed(self, error: object) -> None:
        self._remember_paths(_data(error))
        self.status_label.setText("复测有问题，未通过；请查看报告")
        # Full exception details belong in the readable service report.
        self.summary_label.setText(
            "服务异常退出；已保留报告入口。" if self._report_target() else
            "服务异常退出且未返回报告路径，无法打开报告。"
        )
        self.summary_label.setToolTip(str(error))
        self.summary_label.show()

    @staticmethod
    def _summary_text(data: dict) -> str:
        lines = []
        for mode, item in _modes(data):
            line = f"{_mode_label(mode)}：{_status_label(item.get('status'))}"
            if "frame_count" in item:
                line += f" · {item['frame_count']} 帧"
            if "processed_turn_count" in item:
                line += f" · {item['processed_turn_count']} 个回合"
            # Never combine advice counts across modes or turn process completion
            # into a PASS claim. Failure of either mode stays visible.
            lines.append(line)
        return "\n".join(lines) or "服务未提供分轮结果，请查看报告。"

    def request_cancel(self) -> None:
        if self._thread is None:
            return
        self._thread.stop_event.set()
        self._cancel_requested = True
        self.cancel_button.setEnabled(False)
        if self._thread.supports_cancel:
            self.status_label.setText("正在停止，等待保存已完成部分的报告…")
        else:
            self.status_label.setText("旧接口不支持中途停止；窗口保持打开以保护任务。")

    def _report_target(self) -> Path | None:
        path = self._report_path
        if path and path.is_dir():
            return path
        if path and path.is_file():
            if path.suffix.lower() in {".md", ".html", ".htm", ".txt"}:
                return path
            readable = path.parent / "report.md"
            return readable if readable.is_file() else path.parent
        output = self._output_directory
        if output and output.is_dir():
            readable = output / "report.md"
            return readable if readable.is_file() else output
        return None

    def open_report(self) -> None:
        target = self._report_target()
        if target is not None and not QDesktopServices.openUrl(QUrl.fromLocalFile(str(target))):
            self.status_label.setText("无法打开报告，请按提示路径手动打开。")
            self.status_label.setToolTip(str(target))

    def _directory_target(self) -> Path | None:
        if self._output_directory and self._output_directory.is_dir():
            return self._output_directory
        if self._report_path:
            if self._report_path.is_dir():
                return self._report_path
            if self._report_path.parent.is_dir():
                return self._report_path.parent
        return None

    def open_report_directory(self) -> None:
        target = self._directory_target()
        if target is not None and not QDesktopServices.openUrl(QUrl.fromLocalFile(str(target))):
            self.status_label.setText("无法打开报告目录，请按提示路径手动打开。")
            self.status_label.setToolTip(str(target))

    def _set_running(self, running: bool) -> None:
        self.import_button.setEnabled(not running)
        self.directory_button.setEnabled(not running)
        self.cancel_button.setVisible(running)
        supported = self._thread is not None and self._thread.supports_cancel
        self.cancel_button.setEnabled(running and supported)
        self.cancel_button.setToolTip("停止并保存部分报告" if supported else "旧接口不支持中途停止")
        if running:
            self.report_button.hide()
            self.directory_open_button.hide()

    @Slot()
    def _on_thread_finished(self) -> None:
        if self.sender() is not self._thread:
            return
        self._thread = None
        self.window().removeEventFilter(self)
        self._set_running(False)
        self.progress_bar.hide()
        self.busy_changed.emit(False)
        target, self._close_target = self._close_target, None
        if target is not None:
            target.close()

    def shutdown(self, timeout_ms: int = SHUTDOWN_WAIT_MS) -> bool:
        """Bound GUI waiting; a slow worker stays owned until its finished signal."""
        thread = self._thread
        if thread is None:
            return True
        self.request_cancel()
        finished = thread.wait(max(0, min(int(timeout_ms), self.SHUTDOWN_WAIT_MS)))
        if not finished:
            self.status_label.setText("正在停止并保存报告；尚未结束，窗口保持可用，请稍候。")
        return bool(finished)

    def request_close(self, target: QWidget) -> bool:
        if not self.is_running:
            return True
        self._close_target = target
        self.shutdown()
        return False

    def closeEvent(self, event) -> None:
        if not self.request_close(self):
            event.ignore()
            return
        super().closeEvent(event)

    def eventFilter(self, watched, event) -> bool:
        if event.type() == QEvent.Type.Close and not self.request_close(watched):
            event.ignore()
            return True
        return super().eventFilter(watched, event)


__all__ = ["OfflineDiagnosticPanel", "OfflineDiagnosticReplayThread"]
