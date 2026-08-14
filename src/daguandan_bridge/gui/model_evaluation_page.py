from __future__ import annotations

from pathlib import Path
from threading import Event

from PySide6.QtCore import QThread, Signal
from PySide6.QtWidgets import (
    QComboBox,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QPlainTextEdit,
    QProgressBar,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from ..advisor_strategy import EVALUATION_STRATEGY_OPTIONS
from ..application.model_evaluation import (
    EvaluationInputValidation,
    EvaluationProgress,
    EvaluationRunResult,
    ModelEvaluationService,
    PreparedEvaluation,
)
from ..application.timeline_truth_migration import (
    ExistingTruthRepairInspection,
    ExistingTruthRepairResult,
    TimelineTruthMigrationService,
)


class _EvaluationWorker(QThread):
    prepared = Signal(int, object)
    completed = Signal(int, object)
    failed = Signal(int, str)
    progress_changed = Signal(int, object)

    def __init__(
        self,
        *,
        token: int,
        service: ModelEvaluationService,
        operation: str,
        session: Path | None = None,
        strategy_id: str = "danzero_model",
        prepared: PreparedEvaluation | None = None,
        parent=None,
    ) -> None:
        super().__init__(parent)
        self.token = token
        self.service = service
        self.operation = operation
        self.session = session
        self.strategy_id = strategy_id
        self.prepared_input = prepared
        self.cancel_event = Event()

    def cancel(self) -> None:
        self.cancel_event.set()
        self.requestInterruption()

    def run(self) -> None:
        try:
            if self.operation == "prepare":
                if self.session is None:
                    raise ValueError("尚未选择会话")
                value = self.service.prepare(
                    self.session,
                    strategy_id=self.strategy_id,
                )
                self.prepared.emit(self.token, value)
                return
            if self.prepared_input is None:
                raise ValueError("评测尚未完成预检")
            value = self.service.run(
                self.prepared_input,
                progress=lambda update: self.progress_changed.emit(self.token, update),
                is_cancelled=lambda: self.cancel_event.is_set()
                or self.isInterruptionRequested(),
            )
            self.completed.emit(self.token, value)
        except Exception as exc:
            self.failed.emit(self.token, str(exc))


class _RepairWorker(QThread):
    completed = Signal(int, object)
    failed = Signal(int, str)

    def __init__(
        self,
        *,
        token: int,
        service: TimelineTruthMigrationService,
        session: Path,
        operation: str,
        parent=None,
    ) -> None:
        super().__init__(parent)
        self.token = token
        self.service = service
        self.session = session
        self.operation = operation

    def cancel(self) -> None:
        self.requestInterruption()

    def run(self) -> None:
        try:
            if self.operation == "inspect":
                result = self.service.inspect_existing_truth_repair(self.session)
            else:
                result = self.service.repair_existing_truth(self.session)
            self.completed.emit(self.token, result)
        except Exception as exc:
            self.failed.emit(self.token, str(exc))


class ModelEvaluationPanel(QWidget):
    """Whole-game evaluation controls bound to one TruthLog editor session."""

    result_ready = Signal(object)
    result_invalidated = Signal()
    truth_repaired = Signal(object)

    def __init__(
        self,
        session: Path | str,
        *,
        service: ModelEvaluationService | None = None,
        repair_service: TimelineTruthMigrationService | None = None,
        parent=None,
    ) -> None:
        super().__init__(parent)
        self.setObjectName("modelEvaluationPanel")
        self.setMaximumWidth(720)
        self.session = Path(session).resolve()
        profile_root = self.session.parents[1]
        self.service = service or ModelEvaluationService(
            profiles_root=profile_root.parent,
            profile_name=profile_root.name,
        )
        self.repair_service = repair_service or TimelineTruthMigrationService()
        self._token = 0
        self._validation: EvaluationInputValidation | None = None
        self._prepared: PreparedEvaluation | None = None
        self._result: EvaluationRunResult | None = None
        self._input_stale = False
        self._saved_input_dirty = False
        self._cancel_requested = False
        self._workers: set[QThread] = set()
        self._retired_workers: list[QThread] = []
        self._run_worker: _EvaluationWorker | None = None
        self._repair_worker: _RepairWorker | None = None
        self._repair_inspection: ExistingTruthRepairInspection | None = None
        self._detail_sections: dict[str, str] = {}
        self._shutting_down = False
        self._build_ui()
        self.refresh_eligibility()

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 8, 0, 0)
        layout.setSpacing(8)
        title = QLabel("模型整局评测（teacher-forced，不是闭环胜率）")
        title.setStyleSheet("font-size: 16px; font-weight: 600;")
        layout.addWidget(title)

        form = QFormLayout()
        self.session_label = QLabel(self.session.name)
        form.addRow("对局", self.session_label)
        self.input_label = QLabel("已保存的 truth_log.json")
        form.addRow("输入", self.input_label)
        self.strategy_combo = QComboBox()
        for value, label in EVALUATION_STRATEGY_OPTIONS:
            self.strategy_combo.addItem(label, value)
        form.addRow("策略", self.strategy_combo)
        self.model_info_label = QLabel("模型将在点击开始后于后台加载")
        self.model_info_label.setWordWrap(True)
        self.model_info_label.setMaximumHeight(44)
        form.addRow("运行身份", self.model_info_label)
        eligibility_row = QWidget()
        eligibility_layout = QHBoxLayout(eligibility_row)
        eligibility_layout.setContentsMargins(0, 0, 0, 0)
        eligibility_layout.setSpacing(8)
        self.eligibility_label = QLabel()
        self.eligibility_label.setWordWrap(True)
        self.eligibility_label.setMaximumHeight(44)
        eligibility_layout.addWidget(self.eligibility_label, 1)
        self.details_toggle = QPushButton("显示详情")
        self.details_toggle.setCheckable(True)
        eligibility_layout.addWidget(self.details_toggle)
        form.addRow("资格", eligibility_row)
        layout.addLayout(form)
        self.details_container = QWidget()
        details_layout = QVBoxLayout(self.details_container)
        details_layout.setContentsMargins(0, 0, 0, 0)
        details_layout.setSpacing(6)
        self.details_view = QPlainTextEdit()
        self.details_view.setReadOnly(True)
        self.details_view.setFixedHeight(96)
        details_layout.addWidget(self.details_view)

        repair_row = QHBoxLayout()
        self.inspect_repair_button = QPushButton("检查旧日志修复资格")
        self.repair_button = QPushButton("确认修复旧日志")
        self.repair_button.setEnabled(False)
        self.repair_status_label = QLabel("修复只用于可确定重建的旧版日志")
        self.repair_status_label.setMaximumHeight(44)
        repair_row.addWidget(self.inspect_repair_button)
        repair_row.addWidget(self.repair_button)
        repair_row.addWidget(self.repair_status_label, 1)
        details_layout.addLayout(repair_row)
        self.details_container.setVisible(False)
        layout.addWidget(self.details_container)
        self._set_repair_status(self.repair_status_label.text())

        controls = QHBoxLayout()
        self.start_button = QPushButton("开始整局评测")
        self.cancel_button = QPushButton("取消")
        self.cancel_button.setEnabled(False)
        self.status_label = QLabel("就绪")
        controls.addWidget(self.start_button)
        controls.addWidget(self.cancel_button)
        controls.addWidget(self.status_label, 1)
        layout.addLayout(controls)
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 1)
        self.progress_bar.setValue(0)
        layout.addWidget(self.progress_bar)

        self.metrics_label = QLabel("尚无评测汇总")
        self.metrics_label.setWordWrap(True)
        self.metrics_label.setMaximumHeight(44)
        layout.addWidget(self.metrics_label)

        self.details_toggle.toggled.connect(self._toggle_details)
        self.strategy_combo.currentIndexChanged.connect(self._configuration_changed)
        self.start_button.clicked.connect(self.start_evaluation)
        self.cancel_button.clicked.connect(self.cancel_evaluation)
        self.inspect_repair_button.clicked.connect(self.inspect_repair)
        self.repair_button.clicked.connect(self.repair_existing_truth)

    def refresh_eligibility(self) -> None:
        if self._shutting_down or self._run_worker is not None:
            return
        if self._saved_input_dirty:
            self._validation = None
            self._set_eligibility_text(
                "等待保存 truth_log.json",
                "编辑内容尚未保存；整局评测只读取已保存的 truth_log.json",
            )
            self.start_button.setEnabled(False)
            self.status_label.setText("等待保存")
            return
        try:
            validation = self.service.validate_input(self.session)
        except Exception as exc:
            self._validation = None
            self._set_eligibility_text("输入检查失败", f"阻断：{exc}")
            self.start_button.setEnabled(False)
            return
        self._validation = validation
        if validation.issues:
            details = "阻断：" + "；".join(
                f"{item.code} {item.message}" for item in validation.issues
            )
            self._set_eligibility_text(
                f"阻断：{len(validation.issues)} 个输入问题",
                details,
            )
            self.start_button.setEnabled(False)
            self.status_label.setText("blocked")
        else:
            self._set_eligibility_text(
                f"可运行：{validation.eligible_self_decisions} 个我方决策点",
                "输入为已保存的 truth_log.json；严格游戏合法性检查通过。",
            )
            self.start_button.setEnabled(True)
            self.status_label.setText("输入检查通过")

    def _set_eligibility_text(self, summary: str, details: str | None = None) -> None:
        full_text = details or summary
        self.eligibility_label.setText(summary)
        self.eligibility_label.setToolTip(full_text)
        self._set_detail_section("输入资格", full_text)

    def _set_repair_status(self, text: str) -> None:
        self.repair_status_label.setText(text)
        self._set_detail_section("旧日志修复", text)

    def _set_detail_section(self, title: str, text: str) -> None:
        self._detail_sections[title] = text
        self.details_view.setPlainText(
            "\n\n".join(
                f"【{name}】\n{value}"
                for name, value in self._detail_sections.items()
                if value
            )
        )

    def _toggle_details(self, visible: bool) -> None:
        self.details_container.setVisible(visible)
        self.details_toggle.setText("隐藏详情" if visible else "显示详情")

    def invalidate_input(self) -> None:
        """Mark editor contents dirty until they are saved to truth_log.json."""

        self._prepared = None
        self._result = None
        self._saved_input_dirty = True
        self._input_stale = self._run_worker is not None
        self.metrics_label.setText("编辑内容已变化，旧推荐已失效；请先保存")
        self.result_invalidated.emit()
        if self._run_worker is None:
            self.refresh_eligibility()
        else:
            self._set_eligibility_text(
                "评测使用已冻结输入",
                "编辑内容已变化；当前任务仍使用启动时冻结的已保存日志",
            )

    def saved_input_updated(self) -> None:
        """Refresh eligibility after the editor successfully saves its log."""

        self._prepared = None
        self._result = None
        self._saved_input_dirty = False
        self.metrics_label.setText("已保存，尚无评测汇总")
        self.result_invalidated.emit()
        if self._run_worker is None:
            self.refresh_eligibility()

    def _configuration_changed(self, *_args) -> None:
        self.model_info_label.setText("模型将在点击开始后于后台加载")
        self.model_info_label.setToolTip("")
        self._set_detail_section("运行身份", "模型将在点击开始后于后台加载")
        self._prepared = None
        self._result = None
        self.metrics_label.setText("策略已变化，旧推荐已失效")
        self.result_invalidated.emit()
        self.refresh_eligibility()

    def start_evaluation(self) -> None:
        if (
            self._validation is None
            or not self._validation.ready
            or self._saved_input_dirty
            or self._run_worker is not None
            or self._repair_worker is not None
        ):
            return
        self._token += 1
        token = self._token
        self._input_stale = False
        self._cancel_requested = False
        self._set_running(True)
        self.status_label.setText("正在后台加载并校验模型…")
        worker = _EvaluationWorker(
            token=token,
            service=self.service,
            operation="prepare",
            session=self.session,
            strategy_id=str(self.strategy_combo.currentData()),
            parent=self,
        )
        self._run_worker = worker
        worker.prepared.connect(self._prepare_completed)
        worker.failed.connect(self._worker_failed)
        self._track_worker(worker)
        worker.start()

    def _prepare_completed(self, token: int, prepared: PreparedEvaluation) -> None:
        if self._shutting_down or token != self._token:
            return
        if self._cancel_requested:
            self.status_label.setText("cancelled")
            self._set_running(False)
            return
        self._prepared = prepared
        audit = prepared.strategy_audit
        prefix = "规则基线，不使用 npz | " if prepared.strategy_id == "fabledan_rule" else ""
        identity = (
            f"{prefix}{prepared.strategy_id} | backend={audit.get('backend')} | "
            f"status={audit.get('status', 'ready')} | path={audit.get('path') or '-'} | "
            f"digest={audit.get('digest') or '-'}"
        )
        self.model_info_label.setText(
            f"{prepared.strategy_id} | backend={audit.get('backend')} | "
            f"status={audit.get('status', 'ready')}"
        )
        self.model_info_label.setToolTip(identity)
        self._set_detail_section("运行身份", identity)
        if not prepared.ready:
            details = "阻断：" + "；".join(
                f"{item.code} {item.message}" for item in prepared.issues
            )
            self._set_eligibility_text(
                f"阻断：{len(prepared.issues)} 个预检问题",
                details,
            )
            self.status_label.setText("blocked")
            self._set_running(False)
            return
        worker = _EvaluationWorker(
            token=token,
            service=self.service,
            operation="run",
            prepared=prepared,
            parent=self,
        )
        self._run_worker = worker
        worker.progress_changed.connect(self._progress_changed)
        worker.completed.connect(self._run_completed)
        worker.failed.connect(self._worker_failed)
        self._track_worker(worker)
        worker.start()

    def cancel_evaluation(self) -> None:
        if self._run_worker is not None:
            self._cancel_requested = True
            self._run_worker.cancel()
            self.status_label.setText("正在取消…")

    def _progress_changed(self, token: int, update: EvaluationProgress) -> None:
        if token != self._token or self._shutting_down:
            return
        self.progress_bar.setRange(0, max(1, update.total))
        self.progress_bar.setValue(update.completed)
        self.status_label.setText(
            f"{update.phase} {update.completed}/{update.total} {update.message}".strip()
        )

    def _run_completed(self, token: int, result: EvaluationRunResult) -> None:
        if token != self._token or self._shutting_down:
            return
        self._result = result
        self._cancel_requested = False
        self._set_running(False)
        stale = self._input_stale
        self.status_label.setText(
            f"{result.status}（编辑内容与评测输入不一致，推荐不再适用）"
            if stale
            else str(result.status)
        )
        self._render_result(result)
        if not stale:
            self.result_ready.emit(result)
        else:
            self.refresh_eligibility()

    def _render_result(self, result: EvaluationRunResult) -> None:
        metrics = result.summary.get("metrics", {})
        coverage = metrics.get("coverage", {})
        exact = metrics.get("exact_action_accuracy", {})
        binary = metrics.get("pass_play_accuracy", {})
        latency = metrics.get("latency_ms", {})
        self.metrics_label.setText(
            "coverage {}/{} ({}) | exact {} | pass/play {} | p50/p95 {} / {} ms".format(
                coverage.get("numerator"),
                coverage.get("denominator"),
                _rate_text(coverage.get("rate")),
                _rate_text(exact.get("rate")),
                _rate_text(binary.get("rate")),
                latency.get("p50"),
                latency.get("p95"),
            )
        )
        decisions_name = str(result.summary.get("decisions_path", "decisions.jsonl"))
        report_name = str(result.summary.get("report_path", "report.md"))
        self._set_detail_section(
            "评测产物",
            "决策日志：{}\n评测报告：{}".format(
                result.report_directory / decisions_name,
                result.report_directory / report_name,
            ),
        )

    def inspect_repair(self) -> None:
        self._start_repair_worker("inspect", "正在检查旧日志修复资格…")

    def repair_existing_truth(self) -> None:
        if (
            self._repair_inspection is None
            or self._repair_inspection.status != "candidate"
        ):
            return
        self._start_repair_worker("repair", "正在备份并修复旧日志…")

    def _start_repair_worker(self, operation: str, message: str) -> None:
        if self._repair_worker is not None or self._run_worker is not None:
            return
        self._token += 1
        token = self._token
        self._set_repair_status(message)
        self.inspect_repair_button.setEnabled(False)
        self.repair_button.setEnabled(False)
        worker = _RepairWorker(
            token=token,
            service=self.repair_service,
            session=self.session,
            operation=operation,
            parent=self,
        )
        self._repair_worker = worker
        worker.completed.connect(self._repair_completed)
        worker.failed.connect(self._repair_failed)
        self._track_worker(worker)
        worker.start()

    def _repair_completed(
        self,
        token: int,
        result: ExistingTruthRepairInspection | ExistingTruthRepairResult,
    ) -> None:
        if token != self._token or self._shutting_down:
            return
        if isinstance(result, ExistingTruthRepairInspection):
            self._repair_inspection = result
            self._set_repair_status(f"{result.code}：{result.message}")
            self.repair_button.setEnabled(result.status == "candidate")
            return
        self._repair_inspection = None
        self._set_repair_status(f"{result.code}：{result.message}")
        if result.status == "repaired":
            self.saved_input_updated()
            self.truth_repaired.emit(result)

    def _repair_failed(self, token: int, message: str) -> None:
        if token == self._token and not self._shutting_down:
            self._set_repair_status(f"修复失败：{message}")

    def _track_worker(self, worker: QThread) -> None:
        self._workers.add(worker)
        worker.finished.connect(lambda: self._worker_finished(worker))

    def _worker_finished(self, worker: QThread) -> None:
        self._workers.discard(worker)
        evaluation_finished = self._run_worker is worker
        if self._run_worker is worker:
            self._run_worker = None
        if self._repair_worker is worker:
            self._repair_worker = None
            if not self._shutting_down:
                self.inspect_repair_button.setEnabled(True)
        self._retired_workers.append(worker)
        if evaluation_finished and self._input_stale and not self._shutting_down:
            stale_status = self.status_label.text()
            self._input_stale = False
            self.refresh_eligibility()
            self.status_label.setText(stale_status)

    def _set_running(self, running: bool) -> None:
        self.strategy_combo.setEnabled(not running)
        self.start_button.setEnabled(
            not running
            and not self._saved_input_dirty
            and self._validation is not None
            and self._validation.ready
        )
        self.cancel_button.setEnabled(running)
        self.inspect_repair_button.setEnabled(not running and self._repair_worker is None)
        self.repair_button.setEnabled(
            not running
            and self._repair_inspection is not None
            and self._repair_inspection.status == "candidate"
        )

    def _worker_failed(self, token: int, message: str) -> None:
        if token != self._token or self._shutting_down:
            return
        self._set_running(False)
        self.status_label.setText(f"failed：{message}")

    def shutdown(self) -> None:
        self._shutting_down = True
        self._token += 1
        for worker in tuple(self._workers):
            worker.cancel()
        for worker in tuple(self._workers):
            worker.wait()
        self._workers.clear()
        self._run_worker = None
        self._repair_worker = None


# Kept as an import compatibility name; the feature is no longer a navigation page.
ModelEvaluationPage = ModelEvaluationPanel


def _rate_text(value: object) -> str:
    return "-" if value is None else f"{float(value):.1%}"
