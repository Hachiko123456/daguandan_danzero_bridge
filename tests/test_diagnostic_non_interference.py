from __future__ import annotations

from dataclasses import replace
import hashlib
import os
from types import SimpleNamespace

import cv2
import numpy as np

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from daguandan_bridge.annotation_service import AnnotationService
from daguandan_bridge.capture_service import FrameSnapshot
from daguandan_bridge.config import PROFILES_ROOT
from daguandan_bridge.gui.live_controller import LiveAssistantController
from daguandan_bridge.image_io import read_image_unicode, standardize_to_base
from daguandan_bridge.models import ClientRect
from daguandan_bridge.opening_evidence import (
    NullOpeningEvidenceSink,
    build_opening_evidence_monitor,
)
from daguandan_bridge.recognition_service import ScreenshotRecognitionService
from daguandan_bridge.template_service import TemplateService
from daguandan_bridge.window_capture import CapturedStandardizedFrame


PROFILE_ROOT = PROFILES_ROOT / "tencent_daguandan"


def _paste(canvas: np.ndarray, relative: str, x: int, y: int) -> None:
    template = read_image_unicode(PROFILE_ROOT / relative)
    height, width = template.shape[:2]
    canvas[y : y + height, x : x + width] = template


def _normalized_result(value):
    return replace(value, elapsed_ms=0.0)


def test_trace_toggle_preserves_output_input_and_match_call_count(monkeypatch):
    image = np.full((720, 1280, 3), 255, dtype=np.uint8)
    _paste(image, "templates/rank/2_level.png", 80, 35)
    _paste(image, "templates/rank/2_hand.png", 40, 510)
    _paste(image, "templates/suit/spade_hand.png", 43, 550)
    before_hash = hashlib.sha256(memoryview(image)).hexdigest()
    service = ScreenshotRecognitionService(
        AnnotationService(PROFILES_ROOT),
        TemplateService(PROFILES_ROOT),
        diagnostic_tracing=False,
    )
    original = cv2.matchTemplate
    calls: list[str] = []

    def counted(*args, **kwargs):
        calls.append("matchTemplate")
        return original(*args, **kwargs)

    monkeypatch.setattr(cv2, "matchTemplate", counted)
    disabled = service.recognize(image, allow_unknown_suit=True)
    disabled_calls = tuple(calls)
    calls.clear()

    service.set_diagnostic_tracing_enabled(True)
    enabled = service.recognize(image, allow_unknown_suit=True)
    enabled_calls = tuple(calls)
    trace = service.get_last_diagnostic_trace()

    assert _normalized_result(enabled) == _normalized_result(disabled)
    assert enabled_calls == disabled_calls
    assert hashlib.sha256(memoryview(image)).hexdigest() == before_hash
    assert trace is not None
    assert trace["input_sha256"] == before_hash
    assert trace["threshold_policy"] == "production-unchanged"
    assert trace["candidates"]
    assert all(
        {
            "peak_index",
            "search_box",
            "roi_box",
            "peak_location",
            "match_box",
            "threshold",
            "accepted",
            "rejection_reason",
        }.issubset(candidate)
        for candidate in trace["candidates"]
    )


class _Capture:
    def __init__(self, root):
        self.profiles_root = root


class _Advisor:
    def initialize(self):
        return None


class _Factory:
    def with_advisor(self, advisor):
        del advisor
        return self


class _Recognizer:
    def __init__(self):
        self.calls = 0

    def recognize(self, image, *, allow_unknown_suit=False):
        del image, allow_unknown_suit
        self.calls += 1
        return SimpleNamespace(
            round_level=None,
            my_hand=(),
            lead_player=None,
            current_player=None,
        )


class _ThrowingSink:
    def observe_recognition(self, *_args, **_kwargs):
        raise OSError("diagnostic sink failed")


def _snapshot() -> FrameSnapshot:
    raw = np.full((72, 128, 3), 127, dtype=np.uint8)
    return FrameSnapshot(
        CapturedStandardizedFrame(
            standardization=standardize_to_base(raw, (128, 72), detect_black_bars=False),
            rect=ClientRect(0, 0, 128, 72),
            backend="test",
            dpi=96,
            window_title="test",
            raw_image=raw,
        )
    )


def test_throwing_diagnostic_sink_does_not_add_recognition_calls(tmp_path):
    recognizer = _Recognizer()
    controller = LiveAssistantController(
        _Capture(tmp_path),
        recognition_service=recognizer,
        advisor=_Advisor(),
        session_factory=_Factory(),
        opening_evidence_monitor=_ThrowingSink(),  # type: ignore[arg-type]
    )

    result, returned = controller._recognize_waiting_frame(_snapshot())
    assert controller.opening_evidence.flush(2)

    assert recognizer.calls == 1
    assert result.round_level is None
    assert returned.image.shape == (72, 128, 3)
    assert controller.opening_evidence.failed_calls == 1
    controller.shutdown()


def test_monitor_constructor_failure_falls_back_to_null(monkeypatch):
    monkeypatch.setattr(
        "daguandan_bridge.opening_evidence.OpeningEvidenceMonitor",
        lambda **_kwargs: (_ for _ in ()).throw(OSError("unwritable")),
    )

    assert isinstance(build_opening_evidence_monitor(), NullOpeningEvidenceSink)
