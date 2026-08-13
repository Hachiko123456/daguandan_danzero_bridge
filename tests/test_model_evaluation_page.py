from __future__ import annotations

import json
import os
from pathlib import Path
from threading import Event
import time

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QApplication

from daguandan_bridge.application.model_evaluation import ModelEvaluationService
from daguandan_bridge.application.timeline_truth_migration import (
    TimelineTruthMigrationService,
)
from daguandan_bridge.domain.advice import AdviceResult
from daguandan_bridge.gui.model_evaluation_page import ModelEvaluationPanel
from daguandan_bridge.gui.truth_log_editor import TruthLogEditor
from daguandan_bridge.live.reducer import LiveReducer
from daguandan_bridge.live.truth_log import (
    TruthInitialState,
    TruthLog,
    TruthTurn,
    save_truth_log,
)


HAND = tuple(
    f"{rank}{suit}"
    for rank in ("2", "3", "4", "5", "6", "7")
    for suit in "SHCD"
) + ("8S", "8H", "8C")


class _Advisor:
    def __init__(self) -> None:
        self.initialize_count = 0

    def initialize(self):
        self.initialize_count += 1

    def audit_info(self):
        return {"backend": "danzero", "status": "loaded", "digest": "a" * 64}

    def recommend(self, state, *, request_id="", trace=None):
        del trace
        return AdviceResult(
            strategy="danzero",
            cards=("2S",),
            play_type="SINGLE",
            is_pass=False,
            state_revision=state.revision,
            elapsed_ms=0.1,
            request_id=request_id,
            engine_input={"backend": "danzero"},
        )


class _SlowAdvisor(_Advisor):
    def __init__(self) -> None:
        super().__init__()
        self.started = Event()
        self.release = Event()

    def recommend(self, state, *, request_id="", trace=None):
        self.started.set()
        self.release.wait(2.0)
        return super().recommend(state, request_id=request_id, trace=trace)


def _app():
    return QApplication.instance() or QApplication([])


def _wait(app, predicate, timeout=5.0):
    deadline = time.time() + timeout
    while not predicate() and time.time() < deadline:
        app.processEvents()
        time.sleep(0.01)
    assert predicate()


def _session(tmp_path: Path, *, legacy: bool = False) -> tuple[Path, TruthLog]:
    session = tmp_path / "profiles" / "profile" / "sessions" / "game"
    session.mkdir(parents=True)
    (session / "manifest.json").write_text(
        json.dumps({"session_id": "game", "status": "sealed"}), encoding="utf-8"
    )
    truth = TruthLog(
        "game",
        TruthInitialState("8", "self", HAND),
        (
            TruthTurn(1, "self", False, ("2S",), trick_id=1),
            TruthTurn(2, "right", False, ("3S",), trick_id=1),
        ),
    )
    if legacy:
        raw = truth.to_dict()
        raw.pop("schema")
        raw["schema_version"] = 2
        (session / "truth_log.json").write_text(
            json.dumps(raw, ensure_ascii=False), encoding="utf-8"
        )
    else:
        save_truth_log(session / "truth_log.json", truth)
    return session, truth


def test_fixed_session_panel_initializes_model_only_after_explicit_start(tmp_path):
    app = _app()
    session, truth = _session(tmp_path)
    advisor = _Advisor()
    panel = ModelEvaluationPanel(
        session,
        service=ModelEvaluationService(advisor_factory=lambda _strategy: advisor),
    )

    assert panel.objectName() == "modelEvaluationPanel"
    assert panel.maximumWidth() == 720
    assert panel.eligibility_label.maximumHeight() == 44
    assert panel.metrics_label.maximumHeight() == 44
    assert panel.details_view.height() == 96
    assert panel.details_view.isReadOnly()
    assert panel.details_container.isHidden()
    assert panel.details_container.isAncestorOf(panel.inspect_repair_button)
    assert panel.details_container.isAncestorOf(panel.repair_button)
    assert panel.details_container.isAncestorOf(panel.repair_status_label)
    panel.details_toggle.click()
    assert not panel.details_container.isHidden()
    assert "输入资格" in panel.details_view.toPlainText()
    panel.details_toggle.click()
    assert panel.details_container.isHidden()
    assert not hasattr(panel, "mode_combo")
    assert not hasattr(panel, "decisions_table")
    assert not hasattr(panel, "open_report_button")
    assert panel.start_button.isEnabled()
    assert advisor.initialize_count == 0
    panel.start_button.click()
    _wait(app, lambda: panel._result is not None)

    assert advisor.initialize_count == 1
    assert panel._result.status == "completed"
    assert "coverage 1/1" in panel.metrics_label.text()
    assert (panel._result.report_directory / "decisions.jsonl").is_file()
    assert (panel._result.report_directory / "report.md").is_file()
    panel.shutdown()
    panel.close()


def test_editor_shows_recommendations_only_on_self_rows_and_invalidates_on_edit(
    tmp_path,
):
    app = _app()
    session, truth = _session(tmp_path)
    editor = TruthLogEditor(
        session,
        truth,
        evaluation_service=ModelEvaluationService(
            advisor_factory=lambda _strategy: _Advisor()
        ),
    )

    panel_item = editor.layout().itemAt(
        editor.layout().indexOf(editor.evaluation_panel)
    )
    alignment = panel_item.alignment()
    assert alignment & Qt.AlignmentFlag.AlignLeft
    assert alignment & Qt.AlignmentFlag.AlignTop

    editor.evaluation_panel.start_button.click()
    _wait(app, lambda: editor.evaluation_panel._result is not None)

    assert editor.table.item(0, 6).text() == "2S"
    assert editor.table.item(1, 6) is None
    editor.table.cellWidget(0, 1).setCurrentIndex(
        editor.table.cellWidget(0, 1).findData("right")
    )
    assert editor.table.item(0, 6) is None
    assert "失效" in editor.evaluation_panel.metrics_label.text()
    assert not editor.evaluation_panel.start_button.isEnabled()
    assert "保存" in editor.evaluation_panel.eligibility_label.text()
    editor.shutdown()
    editor.close()


def test_editor_mutation_during_worker_keeps_frozen_run_but_rejects_stale_ui(
    tmp_path,
):
    app = _app()
    session, truth = _session(tmp_path)
    advisor = _SlowAdvisor()
    editor = TruthLogEditor(
        session,
        truth,
        evaluation_service=ModelEvaluationService(
            advisor_factory=lambda _strategy: advisor
        ),
    )

    editor.evaluation_panel.start_button.click()
    _wait(app, advisor.started.is_set)
    editor.table.cellWidget(0, 1).setCurrentIndex(
        editor.table.cellWidget(0, 1).findData("right")
    )
    advisor.release.set()
    _wait(app, lambda: editor.evaluation_panel._result is not None)

    assert editor.evaluation_panel._result.summary["input_sha256"] is not None
    assert editor.table.item(0, 6) is None
    assert "不一致" in editor.evaluation_panel.status_label.text()
    editor.shutdown()
    editor.close()


def test_panel_exposes_explicit_guarded_existing_log_repair(tmp_path):
    app = _app()
    session, _truth = _session(tmp_path, legacy=True)
    reducer = LiveReducer("game")
    events = [
        reducer.confirm_initial_state(
            round_level="8", hand=HAND, lead_player="self"
        ),
        reducer.record_play("self", ("2S",)),
        reducer.record_play("right", ("3S",)),
    ]
    (session / "timeline.jsonl").write_text(
        "".join(
            json.dumps({**event.to_dict(), "schema_version": 1}) + "\n"
            for event in events
        ),
        encoding="utf-8",
    )
    panel = ModelEvaluationPanel(
        session,
        service=ModelEvaluationService(advisor_factory=lambda _strategy: _Advisor()),
        repair_service=TimelineTruthMigrationService(),
    )
    original = (session / "truth_log.json").read_bytes()

    assert panel.details_container.isHidden()
    panel.details_toggle.click()
    assert not panel.details_container.isHidden()
    panel.inspect_repair_button.click()
    _wait(app, lambda: panel._repair_worker is None)
    assert panel.repair_button.isEnabled()
    panel.repair_button.click()
    _wait(app, lambda: panel._repair_worker is None)

    receipt = json.loads(
        (session / "truth_log.repair.json").read_text(encoding="utf-8")
    )
    backup = session / receipt["previous_truth_log"]["path"]
    assert backup.read_bytes() == original
    assert receipt["validation"]["strict_evaluation_input"] is True
    assert "repaired" in panel.repair_status_label.text()
    assert "repaired" in panel.details_view.toPlainText()
    panel.shutdown()
    panel.close()
