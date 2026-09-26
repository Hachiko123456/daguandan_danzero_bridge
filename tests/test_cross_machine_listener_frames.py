"""Optional real-image acceptance: independent snapshots, never a fake timeline.

Set DAGUANDAN_CROSS_MACHINE_FRAMES to a directory of the seven PNG/JSON pairs.
Without the external corpus these tests are explicitly skipped, not a replay PASS.
All writes use tmp_path. The originals and profile template configuration are read-only.
"""
from __future__ import annotations

from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
from types import SimpleNamespace

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
import pytest
from PySide6.QtWidgets import QApplication

from daguandan_bridge.annotation_service import AnnotationService
from daguandan_bridge.application.session_diagnostic_frames import SessionDiagnosticFrameStore
from daguandan_bridge.application.window_debug_report import WindowDebugReportService
from daguandan_bridge.capture_service import FrameSnapshot
from daguandan_bridge.gui.live_controller import LiveAssistantController
from daguandan_bridge.image_io import StandardizationResult
from daguandan_bridge.models import Box, ClientRect
from daguandan_bridge.opening_gate import OpeningTracker
from daguandan_bridge.recognition_service import ScreenshotRecognitionService
from daguandan_bridge.window_capture import CapturedStandardizedFrame

pytestmark = [pytest.mark.visual_fixture, pytest.mark.integration]
ROOT = Path(__file__).resolve().parents[1]
PROFILE = ROOT / "data/profiles/tencent_daguandan"


class ReadOnlyTemplates:
    profile_root = PROFILE

    def list_templates(self):
        return tuple(json.loads((PROFILE / "templates_config.json").read_text(encoding="utf8"))["templates"])


@pytest.fixture(scope="module")
def corpus():
    configured = os.environ.get("DAGUANDAN_CROSS_MACHINE_FRAMES")
    if not configured:
        pytest.skip("external seven-frame corpus not configured")
    root = Path(configured)
    assert all((root / f"{i:06}.{suffix}").is_file() for i in range(1, 8) for suffix in ("png", "json"))
    return root


@pytest.fixture(scope="module")
def recognizer():
    return ScreenshotRecognitionService(AnnotationService(PROFILE.parent, PROFILE.name), ReadOnlyTemplates())


@pytest.mark.parametrize("number", range(1, 8))
def test_random_snapshots_are_tables_and_exact_listener_save_roundtrips(number, corpus, recognizer, tmp_path):
    app = QApplication.instance() or QApplication([])
    store = SessionDiagnosticFrameStore()
    path = corpus / f"{number:06}.png"
    record = store.read_record(path)
    image = store.load_image(record)
    page = recognizer.recognize_listening_page(image)
    result = recognizer.recognize(image, allow_unknown_suit=True)
    assert page.stage == "table" and page.anchor_score > .85
    assert result.round_level == "Q"
    # Each random snapshot is judged separately; never imply consecutive live evidence.
    gate = OpeningTracker().observe(result, anchor_score=page.anchor_score,
                                    generation=1, monotonic_ms=100, observation_id=number)
    assert not gate.ready and gate.seed is None
    if number <= 3:
        assert gate.reason == "doubling"
    else:
        # These are independent mid-game snapshots. Their exact opening-gate
        # reason may be a candidate conflict or reduced-hand/missed-opening
        # diagnosis; the invariant is that they are table evidence, never page_unknown.
        assert gate.reason in {"candidate_conflict", "hand_count_mismatch", "missed_opening", "opening_seed_invalid"}
    height, width = image.shape[:2]
    geometry = StandardizationResult(image=image, source_size=(width, height),
        source_viewport=Box(0, 0, width, height), content_box=Box(0, 0, width, height),
        scale=1.0, padding=(0, 0, 0, 0), aspect_error=0.0, aspect_compatible=True)
    snapshot = FrameSnapshot(CapturedStandardizedFrame(
        standardization=geometry, rect=ClientRect(*record.metadata["rect"]),
        backend=record.metadata["backend"], dpi=record.metadata["dpi"], window_title="test",
        raw_image=None), captured_at=datetime.fromisoformat(record.metadata["captured_at"]),
        captured_monotonic_ms=record.metadata["captured_monotonic_ms"],
        evidence_frame_id=record.metadata["evidence_frame_id"])
    def forbidden_capture(*_a, **_k):
        pytest.fail("listener screenshot must not recapture the window")
    controller = LiveAssistantController(
        SimpleNamespace(profiles_root=tmp_path, open_live_source=forbidden_capture),
        recognition_service=recognizer, advisor=object(), session_factory=SimpleNamespace())
    controller._listening_enabled = True
    generation = controller._waiting_generation
    context = controller._retain_listener_snapshot(snapshot, generation, number)
    controller._listener_evidence.annotate(context, page=controller._page_evidence(page))
    try:
        saved = controller.save_latest_live_frame_to_session()
        copied = store.read_record(saved["image_path"])
        assert copied.metadata["source"] == "live_listener_frame"
        assert copied.metadata["raw_sha256"] == record.metadata["raw_sha256"]
        assert copied.metadata["evidence_frame_id"] == record.metadata["evidence_frame_id"]
        assert copied.metadata["diagnostic_context"]["page"]["stage"] == "table"
        np.testing.assert_array_equal(store.load_image(copied), image)
        report = WindowDebugReportService(profiles_root=PROFILE.parent, profile_name=PROFILE.name,
                                        recognizer=recognizer).build_from_listener_frame(copied.image_path)
        assert report["recognition"]["input_sha256"] == copied.metadata["raw_sha256"]
        assert report["recognition"]["trace"]["input_sha256"] == copied.metadata["raw_sha256"]
        assert report["opening_readiness_inputs"]["gate"]["reason"] == (
            "doubling" if number <= 3 else "hand_count_mismatch")
        assert hashlib.sha256(path.read_bytes()).hexdigest() == record.metadata["png_sha256"]
    finally:
        controller._listening_enabled = False
        controller._listener_evidence.writer.close()
        controller.opening_evidence.close()
        app.processEvents()


@pytest.fixture(autouse=True)
def _isolate_case_root(tmp_path, monkeypatch):
    monkeypatch.setenv("DAGUANDAN_DIAGNOSTICS_ROOT", str(tmp_path / "diagnostics"))
