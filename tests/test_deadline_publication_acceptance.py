"""Independent bounded-thread deadline/publication acceptance; no GUI process."""
from __future__ import annotations

import ast
from dataclasses import dataclass
from pathlib import Path
import threading
from types import SimpleNamespace

import numpy as np
import pytest

import daguandan_bridge.live.orchestrator as core_module
from daguandan_bridge.domain.advice import LocalAdvice
from daguandan_bridge.domain.recognition import FastSignalResult, PlayRegionResult
from daguandan_bridge.gui.compact_view_state import CompactUpdateGate, CompactViewState, project_compact_view
from daguandan_bridge.live.orchestrator import AdviceRequestKey, LiveAdvice, LiveOrchestrator, LiveUpdate
from daguandan_bridge.live.recorder import InMemorySessionRecorder
from daguandan_bridge.live.reducer import LiveReducer
from daguandan_bridge.live.session_store import LiveSessionStore


HAND = tuple(f"{rank}{suit}" for rank in ("2", "3", "4", "5", "6", "7") for suit in "SHCD") + ("8S", "8H", "8C")


@dataclass
class Clock:
    now: int = 1_000

    def __call__(self):
        return self.now


class ManualTimer:
    """Use the production callback/interval, but deterministically fire it."""
    def __init__(self, interval, function):
        self.interval, self.function = interval, function
        self.daemon, self.started, self.cancelled = False, False, False

    def start(self):
        self.started = True

    def cancel(self):
        self.cancelled = True


class Recognition:
    def recognize_fast_signals(self, _image, expected_player, *, allow_pass=True):
        return FastSignalResult(expected_player, "self", False, True, False,
                                cannot_beat_visible=True, cannot_beat_confidence=.99,
                                cannot_beat_box=(10, 10, 20, 10))

    def recognize_play_region(self, _image, seat, *, wild_rank, allow_pass=True):
        return PlayRegionResult(seat, (), False, 0., ("no evidence",), (), source="acceptance")


@pytest.fixture
def deadline_rig(tmp_path, monkeypatch):
    monkeypatch.setattr(core_module, "Timer", ManualTimer)
    clock, notices, background = Clock(), [], []
    store = LiveSessionStore(tmp_path, "acceptance", session_id="deadline-case", automatic_log_delivery_enabled=False)
    store.start({"application_version": "deadline-acceptance"})
    core = LiveOrchestrator(
        reducer=LiveReducer(store.session_id), store=store,
        recorder=InMemorySessionRecorder(store.directory), recognition_service=Recognition(),
        advisor=None, minimum_free_bytes=0, settle_ms=0,
        processing_clock_ms=clock, on_update=notices.append,
    )
    core.start(round_level="2", hand=HAND, lead_player="right", monotonic_ms=900)
    core._capture_generation = 3
    rig = SimpleNamespace(core=core, clock=clock, notices=notices, background=background)
    yield rig
    for thread in background:
        thread.join(1)
    if not any(thread.is_alive() for thread in background):
        core.finish()
    else:
        pytest.fail("bounded concurrency case left a deadlocked daemon; skipped blocking finish")


def arm_recovery(rig):
    core = rig.core
    window = core._ensure_turn_ownership_window()
    fast = FastSignalResult("right", "self", False, True, False)
    core._begin_turn_recovery(window, fast, 1_000, reason="acceptance_missing_right_lead")
    core._withhold_advice_for_turn_recovery(window, 1_000)
    timer = core._deadline_timer
    assert isinstance(timer, ManualTimer) and timer.interval == 2.0 and timer.started
    return window, timer


def launch(rig, target):
    done, errors = threading.Event(), []

    def run():
        try:
            target()
        except BaseException as exc:
            errors.append(exc)
        finally:
            done.set()

    thread = threading.Thread(target=run, daemon=True)
    rig.background.append(thread)
    thread.start()
    return thread, done, errors


def ready_self_update(core):
    core.commit_trusted_action(actor="right", cards=("9C",), is_pass=False, monotonic_ms=1_100)
    core.commit_trusted_action(actor="opposite", is_pass=True, monotonic_ms=1_150)
    core.commit_trusted_action(actor="left", is_pass=True, monotonic_ms=1_200)
    snapshot = core.snapshot
    key = AdviceRequestKey(snapshot.session_id, snapshot.turn_id, snapshot.revision)
    core.latest_advice = LiveAdvice(key, "ready", LocalAdvice(
        strategy="acceptance", cards=(), play_type="PASS", is_pass=True,
        state_revision=snapshot.revision, elapsed_ms=0, request_id=key.request_id,
    ), visible=True)
    return core._update()


def test_two_second_notice_publishes_identity_and_blocked_view_while_state_lock_busy(deadline_rig):
    rig, noticed = deadline_rig, threading.Event()
    window, timer = arm_recovery(rig)
    baseline = rig.core._update()

    def listener(update):
        rig.notices.append(update)
        if update.block_reason == "turn_recovery_budget_exceeded":
            noticed.set()

    rig.core._update_listener = listener
    rig.clock.now += 2_000
    with rig.core._state_lock:
        thread, done, errors = launch(rig, timer.function)
        assert noticed.wait(1), "budget notice incorrectly waited for expensive state lock"
        notice = rig.notices[-1]
        assert isinstance(notice, LiveUpdate)
        assert notice.capture_generation == 3
        assert notice.update_sequence > baseline.update_sequence
        assert notice.snapshot.session_id == "deadline-case"
        assert notice.missing_player == "right" and notice.missing_action_kind == "lead"
        state = project_compact_view(notice, now_ms=rig.clock.now)
        assert state.kind == "blocked" and state.title == "暂无法推荐"
        assert "右家" in state.detail and "手动出牌" in state.detail
        assert not state.cards
        assert window.turn_recovery_deadline_expired
        assert not done.is_set()  # Publication completed; canonical change waits safely.
    assert done.wait(1)
    thread.join(1)
    assert not errors and window.turn_recovery_failed


def test_old_timer_cannot_overwrite_ready_after_window_revision_or_generation_changes(deadline_rig):
    rig = deadline_rig
    _window, timer = arm_recovery(rig)
    current = ready_self_update(rig.core)
    rig.core._capture_generation = 4
    current = rig.core._update()
    assert project_compact_view(current, now_ms=rig.clock.now).kind == "ready"
    gate = CompactUpdateGate()
    assert gate.accept(current)
    before, count = rig.core.snapshot, len(rig.notices)
    rig.clock.now += 2_000
    timer.function()  # Already queued callbacks must self-reject despite cancel().
    assert len(rig.notices) == count
    assert rig.core.snapshot == before
    assert rig.core.latest_advice.status == "ready"
    assert not rig.core._turn_ownership_window.turn_recovery_failed


def test_timer_rechecks_generation_after_publication_before_canonical_failure(deadline_rig):
    """Catch a source replacement in the publication->state-lock race window."""
    rig, notice_seen, release_listener = deadline_rig, threading.Event(), threading.Event()
    window, timer = arm_recovery(rig)
    rig.clock.now += 2_000

    def listener(update):
        rig.notices.append(update)
        if update.block_reason == "turn_recovery_budget_exceeded":
            notice_seen.set()
            assert release_listener.wait(1)

    rig.core._update_listener = listener
    thread = None
    try:
        with rig.core._state_lock:
            thread, done, errors = launch(rig, timer.function)
            assert notice_seen.wait(1)
            # Same owner/revision, but the capture source was replaced while
            # the old timer was outside the publication lock. New-generation
            # state must not inherit the expired old transport's failure.
            rig.core._capture_generation = 4
            fresh = rig.core._update()
            gate = CompactUpdateGate()
            assert gate.accept(fresh)
            assert not gate.accept(rig.notices[0])
            release_listener.set()
        assert done.wait(1)
        assert not errors
        assert not window.turn_recovery_failed, "old-generation timer committed failure after source replacement"
    finally:
        release_listener.set()
        if thread is not None:
            thread.join(1)


def test_listener_snapshot_pause_reentrancy_and_state_to_publication_do_not_deadlock(deadline_rig):
    rig = deadline_rig
    _window, timer = arm_recovery(rig)
    rig.clock.now += 2_000
    holder_ready, listener_entered = threading.Event(), threading.Event()

    def listener(update):
        if update.block_reason == "turn_recovery_budget_exceeded":
            listener_entered.set()
            assert rig.core.snapshot.session_id == "deadline-case"
            rig.core.pause()

    rig.core._update_listener = listener

    def expensive_state_owner():
        with rig.core._state_lock:
            holder_ready.set()
            assert listener_entered.wait(1)
            rig.core._update()  # Competing state -> publication path.

    state_thread, state_done, state_errors = launch(rig, expensive_state_owner)
    assert holder_ready.wait(1)
    timer_thread, timer_done, timer_errors = launch(rig, timer.function)
    assert state_done.wait(1), "publication lock retained across reentrant listener"
    assert timer_done.wait(1), "listener snapshot/pause deadlocked with state owner"
    state_thread.join(1)
    timer_thread.join(1)
    assert not state_errors and not timer_errors
    assert rig.core.status == "paused"


def test_real_hint_expires_without_new_capture_and_actual_ui_expiry_slot_clears_it(deadline_rig):
    rig = deadline_rig
    _window, _timer = arm_recovery(rig)
    frame = np.zeros((32, 64, 3), dtype=np.uint8)
    for timestamp in (1_000, 1_100):
        rig.clock.now = timestamp
        update = rig.core.analyze_frame(frame, monotonic_ms=timestamp, trace_context={"capture_generation": 3, "capture_seq": timestamp})
    assert update.local_rule_hint is not None
    shown = project_compact_view(update, now_ms=rig.clock.now)
    assert shown.kind == "local_rule_hint"
    rig.clock.now = update.local_rule_hint.expires_ms + 1
    assert project_compact_view(update, now_ms=rig.clock.now).kind != "local_rule_hint"

    # Exercise the unchanged real timeout slot with an empty renderer. No Qt
    # import/QApplication/native window is required for this no-new-frame path.
    source = Path(core_module.__file__).parents[1] / "gui" / "recommendation_window.py"
    module = ast.parse(source.read_text(encoding="utf-8-sig"))
    window_class = next(node for node in module.body if isinstance(node, ast.ClassDef) and node.name == "RecommendationFloatWindow")
    method = next(node for node in window_class.body if isinstance(node, ast.FunctionDef) and node.name == "_expire_local_hint")
    namespace = {"CompactViewState": CompactViewState}
    exec(compile(ast.Module(body=[method], type_ignores=[]), str(source), "exec"), namespace)
    rendered = []
    empty_window = SimpleNamespace(_view_state=shown, _render_view=rendered.append)
    namespace["_expire_local_hint"](empty_window)
    assert rendered[-1].kind == "waiting" and rendered[-1].cards == ()
    assert rendered[-1].title != "不出"


def test_late_high_risk_1200ms_verification_callback_cannot_change_executed_history(deadline_rig):
    rig = deadline_rig
    core = rig.core
    # A low-confidence legal play is explicitly high risk. Its next actor's
    # PASS opens the production 1.2s adjacent-action verification window.
    played = core.commit_trusted_action(actor="right", cards=("9C",), is_pass=False,
                                       confidence=.75, monotonic_ms=1_000)
    core.commit_trusted_action(actor="opposite", is_pass=True, monotonic_ms=1_100)
    target = core._previous_action_verification_target(core.snapshot)
    assert target is not None and target.target.event_id == played.event.event_id
    timer = core._verification_timer
    assert isinstance(timer, ManualTimer) and timer.interval == 1.2
    # The following actor already executed another move before the callback
    # gets the state lock; the old adjacent target is no longer authoritative.
    core.commit_trusted_action(actor="left", is_pass=True, monotonic_ms=1_150)
    before = core.snapshot
    original_history = before.play_history
    rig.clock.now += 1_200
    timer.function()
    delayed_cards = PlayRegionResult("right", ("9C", "9D"), False, .99, (), (), source="late-acceptance")
    assert core._apply_previous_action_correction(target, delayed_cards, monotonic_ms=2_300) is None
    assert core.snapshot == before and core.snapshot.play_history == original_history
    assert core.snapshot.revision == before.revision
