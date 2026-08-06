from __future__ import annotations

import json
import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
from PySide6.QtWidgets import QApplication

from daguandan_bridge.gui.main_window import DaguandanBridgeWindow
from daguandan_bridge.gui.replay_page import ReplayPage
from daguandan_bridge.live.recorder import SessionRecorder


def _app():
    return QApplication.instance() or QApplication([])


def _recorded_session(tmp_path):
    session = tmp_path / "sessions" / "game-test"
    session.mkdir(parents=True)
    (session / "manifest.json").write_text(
        json.dumps(
            {
                "session_id": "game-test",
                "status": "sealed",
                "frame_count": 3,
                "dropped_frames": 0,
            }
        ),
        encoding="utf-8",
    )
    (session / "timeline.jsonl").touch()
    recorder = SessionRecorder(session, size=(64, 32), fps=10)
    for index in range(3):
        recorder.write_frame(
            np.full((32, 64, 3), index * 50, np.uint8),
            captured_monotonic_ms=index * 100,
            wall_time=f"t{index}",
        )
    recorder.close()
    return session


def test_replay_page_can_load_recorded_session(tmp_path):
    app = _app()
    session = _recorded_session(tmp_path)
    page = ReplayPage(session.parent)

    page.select_session(session)
    app.processEvents()

    assert page.play_button.isEnabled()
    assert "game-test" in page.session_summary.text()
    assert "3" in page.session_summary.text()
    page.shutdown()
    page.close()


def test_main_window_registers_replay_navigation():
    app = _app()
    window = DaguandanBridgeWindow()

    assert window.replay_page.objectName() == "replayPage"

    window.close()
    app.processEvents()
