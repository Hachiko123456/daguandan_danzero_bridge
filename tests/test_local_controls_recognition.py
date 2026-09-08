"""Narrow controls use actual template matching, no card/PASS/placement scan."""
from __future__ import annotations

from types import SimpleNamespace
import threading

import numpy as np
import pytest

from daguandan_bridge.annotation_service import RegionRecord
from daguandan_bridge.domain.live_runtime import LiveUpdate
from daguandan_bridge.domain.recognition import RecognitionAnnotation
from daguandan_bridge.live.local_rule_hint import LocalRuleHintTracker
from daguandan_bridge.models import Box
from daguandan_bridge.recognition_service import ScreenshotRecognitionService


def local_controls_fixture():
    """Small deterministic image/templates usable by tests and microbenchmark."""
    width, height = 320, 180
    boxes = {
        "timer_self": (10, 10, 40, 30), "button_actions": (80, 100, 130, 40),
        "game_end_controls": (230, 135, 70, 40), "my_play": (60, 40, 100, 45),
        "timer_right": (260, 20, 40, 30), "timer_opposite": (130, 5, 40, 30),
        "timer_left": (10, 60, 40, 30), "right_play": (230, 40, 65, 50),
        "opposite_play": (130, 30, 65, 50), "left_play": (10, 60, 65, 50),
    }
    for index, seat in enumerate(("self", "right", "opposite", "left")):
        boxes[f"passed_{seat}"] = (index * 70, 145, 50, 25)
        boxes[f"placement_{seat}"] = (index * 70, 145, 50, 25)
    regions = tuple(RegionRecord(name, "generic", Box(*box),
                                tuple(value / (width if index % 2 == 0 else height) for index, value in enumerate(box)))
                    for name, box in boxes.items())
    rng = np.random.default_rng(317)
    definitions = (("active", "timer"), ("cannot_beat", "button"),
                   ("continue_game", "button"), ("double", "button"),
                   ("bomb", "effect"), ("passed", "status"), ("head", "status"),
                   ("second", "status"), ("third", "status"), ("last", "status"),
                   ("3", "rank"))
    templates = tuple((dict(label=label, kind=kind, file=f"synthetic-{label}", source_role="generic"),
                       rng.integers(0, 256, (10, 16, 3), dtype=np.uint8)) for label, kind in definitions)
    service = ScreenshotRecognitionService(
        SimpleNamespace(list_regions=lambda: regions), SimpleNamespace(), diagnostic_tracing=False,
    )
    service._template_cache = templates
    image = np.full((height, width, 3), 31, dtype=np.uint8)
    patches = {raw["label"]: template for raw, template in templates}
    image[15:25, 15:31] = patches["active"]
    image[110:120, 90:106] = patches["cannot_beat"]
    return service, image, patches


def test_local_controls_real_match_only_touches_allowed_regions_and_templates(monkeypatch):
    service, image, _ = local_controls_fixture()
    original = service._matches_for_region
    calls = []
    def counted(frame, region, templates, **kwargs):
        selected = [raw for raw, _ in templates if kwargs["predicate"](raw)]
        calls.append((region.name, {raw["kind"] for raw in selected}, kwargs["threshold"]))
        return original(frame, region, templates, **kwargs)
    monkeypatch.setattr(service, "_matches_for_region", counted)
    monkeypatch.setattr(service, "_recognize_cards", lambda *a, **k: pytest.fail("card scan forbidden"))
    monkeypatch.setattr(service, "_recognize_placements", lambda *a, **k: pytest.fail("placement scan forbidden"))
    result = service.recognize_local_controls(image)
    assert result.active_player == "self" and result.cannot_beat_visible
    assert result.cannot_beat_box == (90, 110, 16, 10)
    assert result.cannot_beat_confidence > .99
    assert not result.pass_visible and result.pass_marker_players == () and result.placements == ()
    assert {name for name, _, _ in calls} == {"timer_self", "button_actions", "game_end_controls", "my_play"}
    assert all(kinds <= {"timer", "button", "effect"} for _, kinds, _ in calls)
    assert next(threshold for name, _, threshold in calls if name == "timer_self") == .62
    assert all(threshold == .72 for _, kinds, threshold in calls if kinds == {"button"})


@pytest.mark.parametrize("condition", ["timer_missing", "button_missing", "terminal", "effect", "double"])
def test_unsafe_controls_never_confirm_a_hint(condition):
    service, image, patches = local_controls_fixture()
    if condition == "timer_missing":
        image[15:25, 15:31] = 31
    elif condition == "button_missing":
        image[110:120, 90:106] = 31
    elif condition == "terminal":
        image[145:155, 240:256] = patches["continue_game"]
    elif condition == "effect":
        image[50:60, 80:96] = patches["bomb"]
    elif condition == "double":
        image[110:120, 130:146] = patches["double"]
    result = service.recognize_local_controls(image)
    tracker = LocalRuleHintTracker()
    assert tracker.observe(result, session_id="test", capture_generation=1, captured_ms=1000, now_ms=1000,
                           frame_size=(320, 180)) is None
    assert tracker.observe(result, session_id="test", capture_generation=1, captured_ms=1100, now_ms=1100,
                           frame_size=(320, 180)) is None


@pytest.mark.parametrize("box", [(315, 110, 16, 10), (-1, 2, 16, 10), (10, 10, 0, 1), (True, 0, 2, 2)])
def test_out_of_frame_or_invalid_button_box_is_rejected_without_score_fallback(monkeypatch, box):
    service, image, _ = local_controls_fixture()
    original = service._recognize_buttons
    def fake(frame, region, templates):
        if any(raw.get("label") == "cannot_beat" for raw, _ in templates):
            return ("cannot_beat",), .99, "test", (RecognitionAnnotation("cannot_beat", box, .99, "button"),)
        return original(frame, region, templates)
    monkeypatch.setattr(service, "_recognize_buttons", fake)
    result = service.recognize_local_controls(image)
    assert not result.cannot_beat_visible
    assert result.cannot_beat_box is None


def test_narrow_probe_does_not_poison_formal_expected_other_result():
    service, image, _ = local_controls_fixture()
    service.recognize_local_controls(image)
    formal = service.recognize_fast_signals(image, "right", allow_pass=True)
    assert formal.expected_player == "right"


@pytest.mark.parametrize("missing", ["timer_self", "button_actions", "game_end_controls", "my_play"])
def test_missing_required_controls_or_exclusion_region_abstains(missing):
    service, image, _ = local_controls_fixture()
    regions = service.annotation_service.list_regions()
    service.annotation_service.list_regions = lambda: tuple(region for region in regions if region.name != missing)
    assert not service.recognize_local_controls(image).cannot_beat_visible


def test_ctrl_probe_missing_api_or_failure_keeps_main_analysis(tmp_path):
    from test_live_controller import _app, _CaptureServiceStub, _TokenAnalysisOrchestrator, _preselection_frame
    from daguandan_bridge.gui.live_controller import LiveAssistantController, _AnalysisFrameTask
    _app()
    controller = LiveAssistantController(_CaptureServiceStub(tmp_path))
    core = _TokenAnalysisOrchestrator("test")
    controller.orchestrator = core
    token = controller._activate_live_token(core)
    task = _AnalysisFrameTask(token, _preselection_frame(), 1, 100)
    assert controller._analyze_live_frame(token, task) is not None
    core.preview_controls = lambda *a, **k: pytest.fail("failed scan cannot publish")
    def fail(frame):
        raise RuntimeError("optional controls template error")
    core.recognition_service = SimpleNamespace(recognize_local_controls=fail)
    assert controller._analyze_live_frame(token, task) is not None
    assert core.calls == 2
    assert token.pipeline_timing.snapshot()["counters"]["controls_preview_failed"] == 1
    controller._invalidate_live_token()


def test_real_controller_core_preview_and_analysis_are_not_recording_gated(tmp_path):
    from test_live_controller import _app, _wait_until, _isolated_controller, _preselection_frame
    from daguandan_bridge.gui.live_controller import LiveAssistantController, _AnalysisFrameTask
    from daguandan_bridge.gui.recommendation_window import RecommendationFloatWindow
    from daguandan_bridge.live.orchestrator import LiveOrchestrator
    from daguandan_bridge.live.recorder import InMemorySessionRecorder
    from daguandan_bridge.live.reducer import LiveReducer
    from daguandan_bridge.live.session_store import InMemoryLiveSessionStore
    _app()
    recognition, image, _ = local_controls_fixture()
    store = InMemoryLiveSessionStore(tmp_path, "test")
    store.directory.mkdir()
    recorder = InMemorySessionRecorder(store.directory)
    clock = [1000]
    controller = _isolated_controller(tmp_path)
    core = LiveOrchestrator(
        reducer=LiveReducer(store.session_id), store=store, recorder=recorder,
        recognition_service=recognition, minimum_free_bytes=0,
        processing_clock_ms=lambda: clock[0], on_update=controller._queue_orchestrator_update,
    )
    hand = tuple(f"{rank}{suit}" for rank in ("2", "3", "4", "5", "6", "7") for suit in "SHCD") + ("8S", "8H", "8C")
    core.start(round_level="2", hand=hand, lead_player="right", monotonic_ms=900)
    controller.orchestrator = core
    token = controller._activate_live_token(core)
    before = core.snapshot
    frame = SimpleNamespace(image=image)
    captured = []
    controller.update_ready.connect(captured.append)
    first = _AnalysisFrameTask(token, frame, 1, 1000)
    controller._preview_local_controls(token, first)
    assert not captured
    clock[0] = 1100
    second = _AnalysisFrameTask(token, frame, 2, 1100)
    actual_analysis_calls = []
    def counted(*args, **kwargs):
        actual_analysis_calls.append(1)
        return LiveUpdate(
            status="running",
            snapshot=core.snapshot,
            capture_generation=token.generation,
            update_sequence=1,
        )
    core.analyze_frame = counted
    try:
        result = controller._analyze_live_frame(token, second)
        assert result is not None
        assert _wait_until(lambda: any(update.local_rule_hint is not None for update in captured))
        hint = captured[-1].local_rule_hint
        assert hint.confirmation_frames == 2 and hint.capture_generation == token.generation
        assert actual_analysis_calls == [1]
        assert core.snapshot == before and recorder.frame_count == 0
        assert not core._requested_advice and not core._pending_model_advice
        # Revocation also uses the same read-only core API and generation.
        image[110:120, 90:106] = 31
        clock[0] = 1200
        controller._preview_local_controls(token, _AnalysisFrameTask(token, frame, 3, 1200))
        assert _wait_until(lambda: captured[-1].local_rule_hint is None)
        assert core.snapshot == before and actual_analysis_calls == [1]
    finally:
        controller._invalidate_live_token()
        core.finish()


def test_preview_result_from_replaced_generation_is_not_published(tmp_path):
    from test_live_controller import _app, _CaptureServiceStub, _TokenAnalysisOrchestrator, _preselection_frame
    from daguandan_bridge.gui.live_controller import LiveAssistantController, _AnalysisFrameTask
    _app()
    controller = LiveAssistantController(_CaptureServiceStub(tmp_path))
    core = _TokenAnalysisOrchestrator("test")
    controller.orchestrator = core
    token = controller._activate_live_token(core)
    entered, release = threading.Event(), threading.Event()
    published = []
    def narrow(frame):
        entered.set()
        assert release.wait(1)
        return object()
    core.recognition_service = SimpleNamespace(recognize_local_controls=narrow)
    core.preview_controls = lambda *a, **k: published.append(1)
    thread = threading.Thread(target=lambda: controller._preview_local_controls(
        token, _AnalysisFrameTask(token, _preselection_frame(), 1, 100)))
    thread.start()
    try:
        assert entered.wait(1)
        replacement = controller._activate_live_token(core)
        release.set()
        thread.join(1)
        assert published == [] and controller._active_live_token is replacement
    finally:
        release.set()
        thread.join(1)
        controller._invalidate_live_token()
