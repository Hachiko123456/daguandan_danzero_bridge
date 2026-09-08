from datetime import datetime
import importlib.util
import json
import os
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtWidgets import QApplication
from daguandan_bridge.annotation_service import AnnotationService
from daguandan_bridge.config import PROFILES_ROOT
from daguandan_bridge.gui.live_controller import LiveAssistantController, _WaitingAnalysisTask
from daguandan_bridge.opening_gate import ListeningPageSignal


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "verify_first_play_regions.py"
SPEC = importlib.util.spec_from_file_location("verify_first_play_regions", SCRIPT)
verifier = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(verifier)


def test_timeline_first_lead_is_declared_evidence_not_claimed_visual_truth(tmp_path):
    events = [
        {"event_type": "initial_state_confirmed", "payload": {"lead_player": None}},
        {"event_type": "lead_player_confirmed", "payload": {"lead_player": "opposite"}, "monotonic_ms": 5000, "source": "manual_or_auto_lead_confirmation"},
    ]
    timeline = tmp_path / "timeline.jsonl"
    timeline.write_text("\n".join(json.dumps(item) for item in events), encoding="utf-8")
    result = verifier.declared_lead(tmp_path)
    assert result["expected_seat"] == "opposite"
    assert result["evidence_line"] == 2
    assert "not_visual_truth" in result["evidence_kind"]
    assert "modern_layout_verified" not in result


def test_event_frame_lookup_is_bounded_and_returns_only_three_nearby_frames(tmp_path):
    video = tmp_path / "video"
    video.mkdir()
    (video / "frame_index.jsonl").write_text("\n".join(
        json.dumps({"frame_index": i, "monotonic_ms": i * 100}) for i in range(500)
    ), encoding="utf-8")
    candidates = verifier.event_frame_candidates(tmp_path, {"monotonic_ms": 50000})
    assert candidates == (298, 300)
    assert len(candidates) <= 3


def test_read_only_template_service_does_not_register_new_files(tmp_path):
    profile = tmp_path / "test"
    templates = profile / "templates" / "rank"
    templates.mkdir(parents=True)
    (templates / "2_hand.png").write_bytes(b"not-decoded-by-list")
    manifest = profile / "templates_config.json"
    manifest.write_text('{"templates": []}', encoding="utf-8")
    before = manifest.read_bytes()
    service = verifier.ReadOnlyTemplates(tmp_path, "test")
    assert service.list_templates() == ()
    assert manifest.read_bytes() == before


@pytest.mark.parametrize("seat", verifier.SEATS)
@pytest.mark.parametrize("size", [(1280, 720), (1600, 900), (1920, 1080)])
def test_latest_user_four_first_play_regions_stay_within_table(seat, size):
    regions = {item.name: item for item in AnnotationService(PROFILES_ROOT).list_regions()}
    region = regions["first_play_" + seat]
    assert region.abs_box.fits_within((1280, 720))
    canvas = np.zeros((size[1], size[0], 3), dtype=np.uint8)
    assert AnnotationService._box_for_image(region, canvas).fits_within(size)


def test_report_output_cannot_overwrite_read_only_profile_tree(tmp_path):
    with pytest.raises(ValueError, match="outside read-only"):
        verifier.verify(tmp_path, tmp_path / "generated", max_videos=1, remote_left=None)


@pytest.mark.parametrize("mode", ["all", "game"])
def test_full_scene_delivery_settlement_30_minutes_stays_listening_without_new_media(tmp_path, mode):
    app = QApplication.instance() or QApplication([])

    class Recorder:
        frames = 0
        closed = 0
        def record_frame(self, image, **kwargs):
            assert not self.closed
            self.frames += 1
        def record_recognition(self, result):
            pass
        def close(self, **kwargs):
            self.closed += 1

    recording = Recorder()

    class Factory:
        def start_listener_recording(self, **kwargs):
            return recording

    class Recognition:
        full_calls = 0
        def recognize(self, image, **kwargs):
            self.full_calls += 1
            return SimpleNamespace(my_hand=(), round_level=None, lead_player=None, current_player=None, events=(), buttons=())

    recognition = Recognition()
    controller = LiveAssistantController(
        SimpleNamespace(profiles_root=tmp_path), recognition_service=recognition,
        advisor=object(), session_factory=Factory(), opening_evidence_monitor=object(),
    )
    controller.recording_mode = mode
    controller._listening_enabled = True
    initial = SimpleNamespace(image=np.zeros((1, 1, 3), dtype=np.uint8), captured_monotonic_ms=1000, captured_at=datetime.now().astimezone())
    value, envelope = controller._recognize_waiting_frame(_WaitingAnalysisTask(initial, 0, ListeningPageSignal("table", .95)))
    controller._consume_waiting_recognition(value, envelope)
    initial_frames = recording.frames
    assert initial_frames == (1 if mode == "all" else 0)
    for seconds in range(1801):
        snapshot = SimpleNamespace(image=initial.image, captured_monotonic_ms=2000 + seconds * 1000, captured_at=initial.captured_at)
        task = _WaitingAnalysisTask(snapshot, 0, ListeningPageSignal("settlement", .1, ("continue_game", "change_table")))
        value, envelope = controller._recognize_waiting_frame(task)
        controller._consume_waiting_recognition(value, envelope)
        controller._record_listener_frame(snapshot)
    assert recording.frames == initial_frames
    assert recording.closed == (1 if mode == "all" else 0)
    assert recognition.full_calls == 1
    assert controller._listening_enabled
    assert controller.orchestrator is None
    controller.stop_listening()
    controller.opening_evidence.close(.5)
    app.processEvents()
