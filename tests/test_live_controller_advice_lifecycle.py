from __future__ import annotations

import os
from types import SimpleNamespace

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtWidgets import QApplication

from daguandan_bridge.danzero.advisor import LocalAdvice
from daguandan_bridge.domain.live_runtime import AdviceRequestKey, LiveAdvice, LiveUpdate
from daguandan_bridge.gui.live_controller import LiveAssistantController


def _app():
    return QApplication.instance() or QApplication([])


class _CaptureServiceStub:
    def __init__(self, root):
        self.profiles_root = root


def _controller(tmp_path):
    _app()
    return LiveAssistantController(
        _CaptureServiceStub(tmp_path),
        recognition_service=SimpleNamespace(),
        advisor=object(),
        session_factory=SimpleNamespace(),
    )


def _snapshot(*, session="session", generation=1, turn=1, revision=1, player="self"):
    return SimpleNamespace(
        session_id=session,
        current_player=player,
        turn_id=turn,
        revision=revision,
        my_hand=("QS", "QH", "QD"),
    )


def _advice(key, *, cards=("QS", "QH", "QD"), status="ready", visible=True):
    local = LocalAdvice(
        strategy="fabledan",
        cards=tuple(cards),
        play_type="Trips",
        is_pass=False,
        state_revision=key.state_revision,
        elapsed_ms=1.0,
        request_id=key.request_id,
        engine_input={},
        timings={},
    )
    return LiveAdvice(
        key=key,
        status=status,
        advice=local if status == "ready" else None,
        visible=visible,
    )


def _update(*, session="session", generation=1, turn=1, revision=1, player="self", advice=None, **kwargs):
    snapshot = _snapshot(
        session=session,
        generation=generation,
        turn=turn,
        revision=revision,
        player=player,
    )
    return LiveUpdate(
        status=kwargs.pop("status", "running"),
        snapshot=snapshot,
        advice=advice,
        capture_generation=generation,
        update_sequence=revision,
        **kwargs,
    )


def test_authoritative_update_clears_advice_when_turn_changes(tmp_path):
    controller = _controller(tmp_path)
    key = AdviceRequestKey("session", 1, 1)
    ready = _update(advice=_advice(key))
    assert controller._authoritative_update_for_ui(ready) is ready

    stale = _update(turn=2, revision=2, player="right", advice=ready.advice)
    cleared = controller._authoritative_update_for_ui(stale)

    assert isinstance(cleared.advice, LiveAdvice)
    assert cleared.advice.status == "withheld"
    assert cleared.advice.visible is False
    assert cleared.advice.advice is None
    assert cleared.advice.withhold_reason == "not_local_turn"
    assert controller._authoritative_advice_key is None


def test_action_recovery_and_terminal_updates_invalidate_old_advice(tmp_path):
    controller = _controller(tmp_path)
    key = AdviceRequestKey("session", 1, 1)
    ready = _update(advice=_advice(key))
    controller._authoritative_update_for_ui(ready)

    action = SimpleNamespace(event_type="player_played")
    action_update = _update(advice=ready.advice, event=action)
    cleared_action = controller._authoritative_update_for_ui(action_update)
    assert cleared_action.advice.status == "withheld"
    assert cleared_action.advice.withhold_reason == "action_committed"

    controller._authoritative_update_for_ui(ready)
    recovery = _update(advice=ready.advice, block_reason="turn_recovery_pending")
    cleared_recovery = controller._authoritative_update_for_ui(recovery)
    assert cleared_recovery.advice.status == "withheld"
    assert cleared_recovery.advice.withhold_reason == "turn_recovery_pending"

    controller._authoritative_update_for_ui(ready)
    terminal = _update(status="sealed", advice=ready.advice)
    cleared_terminal = controller._authoritative_update_for_ui(terminal)
    assert cleared_terminal.advice.status == "withheld"
    assert cleared_terminal.advice.withhold_reason == "terminal"


def test_new_authoritative_advice_replaces_old_atomically(tmp_path):
    controller = _controller(tmp_path)
    old_key = AdviceRequestKey("old-session", 1, 1)
    new_key = AdviceRequestKey("new-session", 1, 1)
    old = _update(session="old-session", advice=_advice(old_key))
    new = _update(session="new-session", generation=2, advice=_advice(new_key))

    assert controller._authoritative_update_for_ui(old).advice is old.advice
    accepted = controller._authoritative_update_for_ui(new)

    assert accepted.advice is new.advice
    assert accepted.advice.key == new_key
    assert controller._authoritative_advice_key == new_key
    assert controller._authoritative_advice_generation == 2


def test_preselection_authority_is_controller_update_not_runtime_sidecar(tmp_path):
    controller = _controller(tmp_path)
    key = AdviceRequestKey("session", 1, 1)
    ready = _update(advice=_advice(key))
    controller._authoritative_update_for_ui(ready)

    controller._capture_generation = 1
    controller.orchestrator = SimpleNamespace(
        status="running",
        snapshot=ready.snapshot,
        latest_advice=None,
    )
    task = SimpleNamespace(
        key=key,
        capture_generation=1,
        frame=SimpleNamespace(captured_at=__import__("datetime").datetime.now().astimezone()),
    )
    assert controller._preselection_task_is_current(task) is True

    stale = _update(turn=2, revision=2, player="right")
    controller._authoritative_update_for_ui(stale)
    assert controller._preselection_task_is_current(task) is False


def test_update_signal_never_publishes_stale_advice_after_state_change(tmp_path):
    controller = _controller(tmp_path)
    updates = []
    controller.update_ready.connect(updates.append)
    key = AdviceRequestKey("session", 1, 1)
    ready = _update(advice=_advice(key))
    controller._emit_authoritative_update(ready)

    changed = _update(turn=2, revision=2, player="right", advice=ready.advice)
    controller._emit_authoritative_update(changed)

    assert updates[-1].advice.status == "withheld"
    assert updates[-1].advice.advice is None
    assert updates[-1].advice.withhold_reason == "not_local_turn"
