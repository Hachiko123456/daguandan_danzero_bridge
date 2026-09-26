from __future__ import annotations

"""Acceptance tests for the simplified opening/session architecture.

These tests intentionally exercise public-ish seams already used by the live
controller/runtime.  They are black-box contracts for the five user-facing
requirements, not implementation tests.  In particular, they must fail if a
single missing opening action, one transient ``page_unknown``, or a duplicated
pre-opening directory can still destroy the current episode.
"""

import hashlib
import os
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtWidgets import QApplication

from daguandan_bridge.application.live_v2_session_runtime import LiveV2SessionRuntime
from daguandan_bridge.capture_service import FrameSnapshot
from daguandan_bridge.domain.recording import RecordingResult
from daguandan_bridge.infrastructure.live_v2_rule_session import ProductionRuleSession
from daguandan_bridge.live.session_store import LiveSessionStore
from daguandan_bridge.opening_gate import ListeningPageSignal, OpeningTracker, serialized_result
from daguandan_bridge.gui.live_controller import LiveAssistantController
from daguandan_bridge.image_io import StandardizationResult
from daguandan_bridge.models import Box, ClientRect
from daguandan_bridge.window_capture import CapturedStandardizedFrame


HAND = tuple(
    f"{rank}{suit}"
    for suit in "SHC"
    for rank in ("2", "3", "4", "5", "6", "7", "8", "9", "10", "J", "Q", "K", "A")
)[:27]


# ---------------------------------------------------------------------------
# Opening gate contract


def _opening_observation(*, events=(), lead="self", current="self"):
    return serialized_result(
        round_level="2",
        hand=HAND,
        lead_player=lead,
        current_player=current,
        events=events,
    )


def test_stable_initial_observations_enter_waiting_first_action_without_faking_one():
    """Two stable opening frames create readiness, not a synthetic play event."""

    tracker = OpeningTracker()
    evaluations = [
        tracker.observe(
            _opening_observation(),
            anchor_score=0.95,
            generation=1,
            monotonic_ms=tick,
            observation_id=tick,
        )
        for tick in (100, 200)
    ]

    final = evaluations[-1]
    assert final.ready, (
        "两帧已经提供 27 张合法手牌、级牌、首出和当前行动者；"
        f"不应继续卡在 {final.reason!r}"
    )
    assert final.reason == "ready_waiting_first_action", (
        "开局确认不应要求真实首出动作；应进入 READY_WAITING_FIRST_ACTION，"
        f"实际 reason={final.reason!r}"
    )
    assert final.seed is not None
    assert final.seed.opening_action is None, (
        "没有真实 events 时不得伪造 OpeningActionSeed"
    )


def test_missing_lead_action_is_not_treated_as_a_completed_action():
    """A static opening frame may establish context but never invent an event."""

    tracker = OpeningTracker()
    result = _opening_observation(events=())
    first = tracker.observe(result, anchor_score=0.95, generation=1, monotonic_ms=100, observation_id=100)
    second = tracker.observe(result, anchor_score=0.95, generation=1, monotonic_ms=200, observation_id=200)

    assert second.seed is not None
    assert second.seed.opening_action is None
    assert not getattr(result, "events", ())
    assert getattr(tracker, "synthetic_action_count", 0) == 0
    assert first.reason in {"confirming_hand", "confirming_opening", "ready_waiting_first_action"}


# ---------------------------------------------------------------------------
# Formal runtime contract


class _MemoryStore:
    session_id = "acceptance-session"
    persistence_enabled = True
    automatic_log_delivery_enabled = False

    def __init__(self, root: Path) -> None:
        self.directory = root
        self.events: list[object] = []
        self.batches: list[tuple[object, ...]] = []
        self.advice: list[dict[str, object]] = []
        self.started: list[dict[str, object]] = []
        self.identities: list[dict[str, object]] = []
        self.seals: list[dict[str, object]] = []

    def start(self, manifest):
        self.started.append(dict(manifest))

    def update_runtime_identity(self, identity):
        self.identities.append(dict(identity))

    def append_event(self, event):
        self.events.append(event)

    def append_event_batch(self, events):
        self.batches.append(tuple(events))
        self.events.extend(events)

    def append_advice(self, record):
        self.advice.append(dict(record))

    def append_observation(self, _record):
        pass

    def append_recognition_trace(self, _record):
        pass

    def upsert_decision(self, _record):
        pass

    def create_incident(self, **_kwargs):
        return self.directory

    def append_incident_occurrence(self, *_args, **_kwargs):
        pass

    def seal(self, **kwargs):
        self.seals.append(dict(kwargs))

    def append_post_seal_health_audit(self, *_args, **_kwargs):
        pass

    def record_automatic_log_delivery(self, _result):
        pass


class _MemoryRecorder:
    frame_count = 0

    def write_frame(self, *_args, **_kwargs):
        self.frame_count += 1

    def close(self):
        return RecordingResult(Path("video.avi"), Path("frames.jsonl"), self.frame_count, 0)


class _NoOpVision:
    def start(self):
        pass

    def close(self):
        pass


class _NoOpAdvice:
    def __init__(self):
        self.submissions: list[object] = []

    def start(self, *, timeout=10.0):
        del timeout

    def submit(self, *args, **kwargs):
        self.submissions.append((args, kwargs))
        return ()

    def drain_results(self):
        return ()

    def close(self, *, timeout=5.0):
        del timeout


def _runtime_without_opening_action(tmp_path: Path):
    store = _MemoryStore(tmp_path / "formal-session")
    store.directory.mkdir()
    store.start({"schema": "acceptance.live-session/1"})
    recorder = _MemoryRecorder()
    advisers: list[_NoOpAdvice] = []

    def make_advice(_version):
        value = _NoOpAdvice()
        advisers.append(value)
        return value

    runtime = LiveV2SessionRuntime(
        rule_session=ProductionRuleSession(store),
        store=store,
        recorder=recorder,
        recognition_service=SimpleNamespace(),
        vision_factory=lambda _version: _NoOpVision(),
        advice_runtime_factory=make_advice,
    )
    return runtime, store, advisers


def test_formal_session_can_start_without_opening_action_and_wait_safely(tmp_path):
    """Session context is usable before the first real action is observed."""

    runtime, store, advisers = _runtime_without_opening_action(tmp_path)
    try:
        update = runtime.start(
            round_level="2",
            hand=HAND,
            lead_player="self",
            opening_action=None,
            monotonic_ms=0,
        )

        action_types = {"player_played", "player_passed", "manual_confirmed_event"}
        assert runtime.session_state in {"waiting_first_action", "WAITING_FIRST_ACTION"}, (
            "opening_action=None 时应进入等待首出状态，而不是直接进入普通推荐流程"
        )
        assert not any(getattr(event, "event_type", None) in action_types for event in update.events)
        assert not any(getattr(event, "event_type", None) in action_types for event in store.events)
        assert not any(getattr(advice, "visible", False) for advice in [runtime.latest_advice])
        assert all(not value.submissions for value in advisers), (
            "第一手动作未确认前不得请求正式建议"
        )
    finally:
        if getattr(runtime, "_initialized", False) and runtime.status != "sealed":
            runtime.finish()


def test_real_legal_first_play_advances_waiting_session(tmp_path):
    """The first legal trusted play is the only thing that exits the wait state."""

    runtime, _store, _advisers = _runtime_without_opening_action(tmp_path)
    try:
        runtime.start(
            round_level="2",
            hand=HAND,
            lead_player="self",
            opening_action=None,
            monotonic_ms=0,
        )
        update = runtime.commit_trusted_action(
            actor="self",
            cards=(HAND[0],),
            is_pass=False,
            monotonic_ms=100,
            source="acceptance_test",
        )
        assert any(
            getattr(event, "event_type", None) == "player_played"
            for event in update.events
        ), "真实合法首出后必须产生 player_played 事件"
        assert runtime.status in {"running", "RUNNING"}
    finally:
        if getattr(runtime, "_initialized", False) and runtime.status != "sealed":
            runtime.finish()


# ---------------------------------------------------------------------------
# Recoverable page-unknown contract


class _CaptureStub:
    def __init__(self, root: Path) -> None:
        self.profiles_root = root


class _Recording:
    def __init__(self) -> None:
        self.closed_with: list[str] = []

    def close(self, *, reason):
        self.closed_with.append(str(reason))


def _app():
    return QApplication.instance() or QApplication([])


def test_transient_and_persistent_page_unknown_keep_bounded_recovery_without_stopping_recording(tmp_path):
    _app()
    controller = LiveAssistantController(_CaptureStub(tmp_path))
    controller._listening_enabled = True
    controller._listening_page = ListeningPageSignal("table", 0.95)
    recording = _Recording()
    controller._listener_recording = recording
    snapshot = SimpleNamespace(captured_monotonic_ms=100)

    controller._apply_listening_page(ListeningPageSignal("unknown", 0.0), snapshot)
    assert not recording.closed_with, (
        "一次 page_unknown 只代表当前帧不确定，不应立即封存或终止 episode"
    )
    assert controller._listening_enabled is True

    controller._apply_listening_page(ListeningPageSignal("table", 0.95), snapshot)
    assert not recording.closed_with, "短暂 unknown 后回到 table 应保持同一监听回合"

    # A finite fast budget transitions to low-frequency recovery. The
    # listener remains enabled without fabricating actions from unknown frames.
    for _ in range(32):
        controller._apply_listening_page(ListeningPageSignal("unknown", 0.0), snapshot)
        if recording.closed_with:
            break
    assert not recording.closed_with, (
        "page_unknown 超过快速预算后进入低频恢复，不应永久封存监听；"
        "只有硬故障或用户停止才能关闭 recording"
    )
    assert controller._page_recovery_slow is True


# ---------------------------------------------------------------------------
# Episode directory and exact-frame capture contracts


def test_one_episode_owns_manifest_trace_and_diagnostic_frames_without_parallel_lifecycles(tmp_path):
    store = LiveSessionStore.for_episode(tmp_path, "tencent_daguandan")
    store.start_episode({"schema": "acceptance.episode/1"})

    assert store.directory.parent.name == ".preopening"
    assert (store.directory / "manifest.json").is_file()
    assert (store.directory / "recognition_trace.jsonl").is_file()
    assert (store.directory / "diagnostic_frames").is_dir()
    siblings = list(store.directory.parent.iterdir())
    assert siblings == [store.directory], (
        "一个 episode 必须统一承载 manifest/trace/diagnostic_frames，"
        f"不应额外创建并列 opening/diagnostic 目录：{siblings}"
    )
    assert not (store.directory / "opening").exists()
    assert not (store.directory / "diagnostic").exists()


def _snapshot(value: int) -> FrameSnapshot:
    image = np.array(
        [[[value, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]],
        dtype=np.uint8,
    )
    standardization = StandardizationResult(
        image=image,
        source_size=(2, 2),
        source_viewport=Box(0, 0, 2, 2),
        content_box=Box(0, 0, 2, 2),
        scale=1.0,
        padding=(0, 0, 0, 0),
        aspect_error=0.0,
        aspect_compatible=True,
    )
    frame = CapturedStandardizedFrame(
        standardization=standardization,
        rect=ClientRect(10, 20, 2, 2),
        backend="screen",
        dpi=96,
        window_title="牌桌",
        raw_image=image.copy(),
    )
    return FrameSnapshot(
        frame=frame,
        captured_at=None,
        captured_monotonic_ms=1234,
        evidence_frame_id=f"acceptance-{value}",
    )


def test_saved_listener_frame_is_exact_and_never_recaptured(tmp_path, monkeypatch):
    """The diagnostic image must be the same cached frame the listener saw."""

    from daguandan_bridge.application import session_diagnostic_frames as frames_module
    from daguandan_bridge.gui.live_controller import LiveAssistantController

    saved: list[dict[str, object]] = []

    class _Store:
        def save_snapshot(self, directory, snapshot, **kwargs):
            saved.append({"directory": directory, "snapshot": snapshot, **kwargs})
            return {
                "sequence": 1,
                "count": 1,
                "image_path": directory / "diagnostic_frames/000001.png",
                "metadata_path": directory / "diagnostic_frames/000001.json",
            }
    monkeypatch.setattr(frames_module, "SessionDiagnosticFrameStore", _Store)
    capture = _CaptureStub(tmp_path)
    controller = LiveAssistantController(capture)
    controller.orchestrator = SimpleNamespace(
        store=SimpleNamespace(directory=tmp_path / "session-1", session_id="session-1"),
        snapshot=SimpleNamespace(session_id="session-1"),
    )
    from daguandan_bridge.gui.live_controller import _LiveRunToken
    controller._active_live_token = _LiveRunToken(
        orchestrator=controller.orchestrator,
        session_id="session-1",
        nonce=1,
        generation=3,
    )
    controller._capture_generation = 3
    snapshot = _snapshot(91)
    controller._latest_live_frame = snapshot
    controller._latest_live_frame_generation = 3
    controller._latest_live_frame_capture_seq = 17

    def forbidden_recapture(*_args, **_kwargs):
        raise AssertionError("保存监听截图不允许重新捕获窗口")

    capture.capture_frame = forbidden_recapture
    result = controller.save_latest_live_frame_to_session()

    assert saved and saved[0]["snapshot"].evidence_frame_id == snapshot.evidence_frame_id
    assert saved[0]["capture_seq"] == 17
    assert saved[0]["capture_generation"] == 3
    assert result["evidence_frame_id"] == snapshot.evidence_frame_id
    assert result["raw_sha256"] == hashlib.sha256(snapshot.image.tobytes(order="C")).hexdigest()




def test_manual_fallback_uses_the_same_live_capture_service_chain(tmp_path, monkeypatch):
    """Without a cached listener frame, manual capture still uses open_live_source()."""

    from daguandan_bridge.application import session_diagnostic_frames as frames_module

    snapshot = _snapshot(92)
    saved: list[dict[str, object]] = []

    class _Source:
        def __init__(self):
            self.capture_calls = 0
            self.closed = False

        def capture(self):
            self.capture_calls += 1
            return snapshot

        def close(self):
            self.closed = True

    class _CaptureWithSource(_CaptureStub):
        def __init__(self, root):
            super().__init__(root)
            self.open_calls: list[str] = []
            self.sources: list[_Source] = []

        def open_live_source(self, profile_name):
            self.open_calls.append(profile_name)
            source = _Source()
            self.sources.append(source)
            return source

    class _Store:
        def save_snapshot(self, directory, image, **kwargs):
            saved.append({"directory": directory, "snapshot": image, **kwargs})
            return {
                "sequence": 1,
                "count": 1,
                "image_path": directory / "diagnostic_frames/000001.png",
                "metadata_path": directory / "diagnostic_frames/000001.json",
            }

    monkeypatch.setattr(frames_module, "SessionDiagnosticFrameStore", _Store)
    capture = _CaptureWithSource(tmp_path)
    controller = LiveAssistantController(capture)
    controller.capture_frame = lambda *_args, **_kwargs: pytest.fail(
        "manual fallback must use open_live_source().capture(), not capture_frame()"
    )

    result = controller.save_latest_live_frame_to_session()

    assert capture.open_calls == ["tencent_daguandan"]
    assert len(capture.sources) == 1
    assert capture.sources[0].capture_calls == 1
    assert capture.sources[0].closed is True
    assert result["source"] == "manual_window_capture"
    assert result["source_phase"] == "manual_window_capture"
    assert saved[0]["snapshot"].evidence_frame_id == snapshot.evidence_frame_id
    assert result["raw_sha256"] == hashlib.sha256(snapshot.image.tobytes(order="C")).hexdigest()




@pytest.fixture(autouse=True)
def _isolate_case_root(tmp_path, monkeypatch):
    monkeypatch.setenv("DAGUANDAN_DIAGNOSTICS_ROOT", str(tmp_path / "diagnostics"))
