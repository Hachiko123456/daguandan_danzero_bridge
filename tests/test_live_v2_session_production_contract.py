from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from threading import enumerate as live_threads
from time import sleep

import pytest

from daguandan_bridge.application.live_v2_advice_protocol import (
    AdviceRequestIdentity,
    AdviceRuntimeResult,
    AdviceRuntimeStatus,
)
from daguandan_bridge.application.live_v2_frame_types import FramePipelineResult
from daguandan_bridge.application.live_v2_session_runtime import LiveV2SessionRuntime
from daguandan_bridge.application.ports import LiveRuntimePort
from daguandan_bridge.domain.advice import AdviceResult
from daguandan_bridge.domain.recognition import FastSignalResult
from daguandan_bridge.domain.recording import RecordingResult
from daguandan_bridge.infrastructure.live_v2_rule_session import ProductionRuleSession
from daguandan_bridge.live_v2.candidates import (
    ActionCandidate, ActionKind, CandidateReason, EvidenceOrigin,
)
from daguandan_bridge.live_v2.identity import FrameIdentity, Seat


HAND = tuple(
    f"{rank}{suit}"
    for suit in "SHC"
    for rank in ("2", "3", "4", "5", "6", "7", "8", "9", "10", "J", "Q", "K", "A")
)[:27]


class StubStore:
    session_id = "production-contract"
    directory = Path("memory-production-contract")
    persistence_enabled = True
    automatic_log_delivery_enabled = False

    def __init__(self) -> None:
        self.events: list[object] = []
        self.batches: list[tuple[object, ...]] = []
        self.traces: list[dict[str, object]] = []
        self.advice: list[dict[str, object]] = []
        self.health: list[dict[str, object]] = []
        self.seals = 0
        self.fail_batches = False

    def start(self, manifest): self.manifest = manifest
    def append_event(self, event): self.events.append(event)
    def append_event_batch(self, events):
        if self.fail_batches:
            raise OSError("disk full")
        self.batches.append(tuple(events))
    def append_advice(self, record): self.advice.append(record)
    def append_observation(self, record): pass
    def append_recognition_trace(self, record): self.traces.append(record)
    def update_runtime_identity(self, identity): pass
    def upsert_decision(self, record): pass
    def create_incident(self, **kwargs): return self.directory
    def append_incident_occurrence(self, *args, **kwargs): pass
    def seal(self, **kwargs): self.seals += 1
    def append_post_seal_health_audit(self, report, **kwargs): self.health.append(report)
    def record_automatic_log_delivery(self, result): pass


class StubRecorder:
    frame_count = 0

    def __init__(self) -> None:
        self.fail = False
        self.closed = 0

    def write_frame(self, frame, monotonic_ms, wall_time):
        if self.fail:
            raise OSError("codec failed")
        self.frame_count += 1
        return None

    def close(self):
        self.closed += 1
        return RecordingResult(Path("game.avi"), Path("frames.jsonl"), self.frame_count, 0)


class StubRecognition:
    pass


class StubRuleSession(ProductionRuleSession):
    """Spyable rule-session boundary; runtime behavior must not inspect internals."""

    def __init__(self, store: StubStore) -> None:
        super().__init__(store)
        self.corrections = []
        self.health_override = None

    def correct_latest(self, command):
        self.corrections.append(command)
        return super().correct_latest(command)

    def health(self, **kwargs):
        return self.health_override or super().health(**kwargs)


def _fast(expected: str = "right") -> FastSignalResult:
    return FastSignalResult(expected, None, False, False, False)


class StubVision:
    def __init__(self) -> None:
        self.started = 0
        self.closed = 0
        self.calls = 0
        self.frames: list[FrameIdentity] = []
        self.opening_leads: list[Seat | None] = []
        self.batches: list[tuple[dict[str, object], ...]] = []

    def start(self, **kwargs): self.started += 1
    def close(self, **kwargs): self.closed += 1
    def queue(self, *specs: dict[str, object]) -> None: self.batches.append(tuple(specs))

    def process_frame(
        self, image, *, frame, version, wild_rank, expected_seat=None,
        now_ms=None, formal_action_boundary=None, opening_lead_seat=None,
    ):
        del formal_action_boundary
        del image, wild_rank
        self.calls += 1
        self.frames.append(frame)
        self.opening_leads.append(opening_lead_seat)
        specs = self.batches.pop(0) if self.batches else ()
        candidates = []
        for offset, spec in enumerate(specs, 1):
            first = FrameIdentity(
                frame.session_id, frame.capture_generation, frame.frame_sequence * 10 + offset,
                max(0, frame.captured_ms - 10), frame.roi_version, frame.source_id,
            )
            last = FrameIdentity(
                frame.session_id, frame.capture_generation, frame.frame_sequence * 10 + offset + 4,
                frame.captured_ms, frame.roi_version, frame.source_id,
            )
            cards = tuple(spec.get("cards", ()))
            options = tuple(spec.get("suit_options", tuple((card,) for card in cards)))
            candidates.append(ActionCandidate(
                str(spec["id"]), version, Seat(str(spec["seat"])),
                ActionKind(str(spec.get("kind", "play"))), cards, options,
                tuple(spec.get("evidence_ids", (f"obs-{offset}-a", f"obs-{offset}-b"))),
                int(spec.get("action_epoch", frame.frame_sequence)), first, last, int(now_ms),
                float(spec.get("confidence", 0.96)),
                CandidateReason.STABLE_PLAY if cards else CandidateReason.FRESH_PASS_EDGE,
            ))
        return FramePipelineResult(
            frame, _fast(expected_seat.value if expected_seat else "right"),
            (), (), tuple(candidates), (), (), 0,
        )


class StubAdvice:
    def __init__(self) -> None:
        self.started = 0
        self.closed = 0
        self.submissions: list[tuple[object, object, int]] = []
        self.pending: list[AdviceRuntimeResult] = []
        self.release = False
        self.next_status = AdviceRuntimeStatus.ADVICE
        self.next_diagnostic: dict[str, object] = {}

    def start(self, **kwargs): self.started += 1
    def close(self, **kwargs): self.closed += 1
    def submit(self, snapshot, opportunity, *, request_sequence, timeout_ms=3000):
        self.submissions.append((snapshot, opportunity, request_sequence))
        status = self.next_status
        result = AdviceRuntimeResult(
            AdviceRequestIdentity(opportunity.version, request_sequence, opportunity.opportunity_id),
            status, 1, 101, 1.0,
            None if status is not AdviceRuntimeStatus.ADVICE else AdviceResult(
                "stub", (snapshot.my_hand[0],), "Single", False,
                snapshot.version.state_revision, 1.0,
                engine_input={
                    "unknown_suit_resolution": dict(self.next_diagnostic)
                } if self.next_diagnostic else None,
            ),
            failure_code=("suit_pending" if status is AdviceRuntimeStatus.BLOCKED else ""),
            diagnostic=dict(self.next_diagnostic),
        )
        self.pending.append(result)
        return ()

    def drain_results(self):
        if not self.release:
            return ()
        values, self.pending = tuple(self.pending), []
        return values


class Rig:
    def __init__(self, *, lead: str | None = "right", on_update=None) -> None:
        self.store, self.recorder = StubStore(), StubRecorder()
        self.store.start({"schema": "test.live-v2/1"})
        self.rules = StubRuleSession(self.store)
        self.visions: list[StubVision] = []
        self.advisers: list[StubAdvice] = []
        self.live = LiveV2SessionRuntime(
            rule_session=self.rules, store=self.store, recorder=self.recorder,
            recognition_service=StubRecognition(),
            vision_factory=self._vision, advice_runtime_factory=self._advice,
            on_update=on_update, processing_clock_ms=lambda: 0,
            local_hint_window_ms=0,
        )
        self.first = self.live.start(
            round_level="2", hand=HAND, lead_player=lead, monotonic_ms=0,
            wall_time="2026-09-06T00:00:00+08:00",
        )

    def _vision(self, version):
        value = StubVision(); self.visions.append(value); return value

    def _advice(self, version):
        value = StubAdvice(); self.advisers.append(value); return value

    def bind(self, generation: int = 1):
        return self.live.bind_capture_generation(generation)


@pytest.fixture
def rigs():
    values: list[Rig] = []
    yield values
    for rig in values:
        rig.live.finish()


def _rig(rigs, **kwargs) -> Rig:
    value = Rig(**kwargs)
    rigs.append(value)
    assert isinstance(value.live, LiveRuntimePort)
    return value


def _play(candidate_id: str, seat: str, cards=("3D",), **extra):
    return {"id": candidate_id, "seat": seat, "cards": cards, **extra}


def test_start_is_inert_until_first_positive_generation_bind(rigs) -> None:
    rig = _rig(rigs)
    assert rig.first.capture_generation == 0
    assert rig.visions == [] and rig.advisers == []
    bound = rig.bind(7)
    assert bound.capture_generation == 7
    assert len(rig.visions) == len(rig.advisers) == 1
    assert rig.visions[0].started == rig.advisers[0].started == 1
    rig.bind(7)
    assert len(rig.visions) == len(rig.advisers) == 1


def test_application_runtime_has_no_legacy_or_infrastructure_imports() -> None:
    source = Path("src/daguandan_bridge/application/live_v2_session_runtime.py").read_text(
        encoding="utf-8"
    )
    assert "..live.reducer" not in source
    assert "..infrastructure" not in source
    assert "daguandan_bridge.infrastructure" not in source


def test_trace_context_preserves_capture_identity_and_rejects_old_generation(rigs) -> None:
    rig = _rig(rigs); rig.bind(3)
    vision = rig.visions[-1]
    rig.live.analyze_frame(
        object(), monotonic_ms=999,
        trace_context={
            "capture_generation": 3, "capture_seq": 41, "captured_ms": 123,
            "roi_version": "roi-from-capture", "source_id": "window:real-hwnd",
        },
    )
    identity = vision.frames[-1]
    assert (
        identity.capture_generation, identity.frame_sequence, identity.captured_ms,
        identity.roi_version, identity.source_id,
    ) == (3, 41, 123, "roi-from-capture", "window:real-hwnd")
    calls = vision.calls
    stale = rig.live.analyze_frame(
        object(), monotonic_ms=1_000,
        trace_context={"capture_generation": 2, "capture_seq": 42, "captured_ms": 124},
    )
    assert stale.block_reason == "stale_capture_identity"
    assert vision.calls == calls


def test_missing_trace_fields_use_safe_monotonic_defaults(rigs) -> None:
    rig = _rig(rigs); rig.bind(1)
    vision = rig.visions[-1]
    rig.live.analyze_frame(object(), monotonic_ms=250, trace_context={})
    identity = vision.frames[-1]
    assert identity.capture_generation == 1
    assert identity.frame_sequence == 1
    assert identity.captured_ms == 250
    assert identity.roi_version == "live-v2"
    assert identity.source_id == "live-v2-capture"


def test_conflicting_current_seat_candidates_are_not_retained_for_manual_guessing(rigs) -> None:
    rig = _rig(rigs); rig.bind()
    vision = rig.visions[-1]
    vision.queue(
        _play("chosen", "right", ("3?",), suit_options=(("3?", "3S", "3D"),),
              action_epoch=47, evidence_ids=("visual-a", "visual-b")),
        _play("other", "right", ("4D",), action_epoch=47),
    )
    review = rig.live.analyze_frame(object(), monotonic_ms=100)
    assert review.snapshot.revision == 1
    confirmed = rig.live.confirm_candidate("chosen")
    assert confirmed.block_reason == "unknown_candidate"
    assert not confirmed.snapshot.play_history
    assert len(rig.store.batches) == 1


def test_opening_action_uses_one_audit_evidence_and_the_unified_commit(rigs) -> None:
    rig = _rig(rigs, lead=None); rig.bind()
    update = rig.live.bootstrap_opening_action(
        actor="left", cards=("7?",), expected_next_player="self",
        suit_options=(("7C", "7D"),),
        monotonic_ms=100, confidence=0.99, source="opening-marker",
    )
    assert update.snapshot.current_player == "self"
    assert update.snapshot.play_history[-1].cards == ("7?",)
    assert update.snapshot.play_history[-1].suit_options == (("7C", "7D"),)
    assert [batch[0].event_type for batch in rig.store.batches] == [
        "initial_state_confirmed", "lead_player_confirmed", "player_played",
    ]
    event = rig.store.batches[-1][0]
    assert event.event_type == "player_played"
    assert len(event.evidence_refs) == 1
    assert event.evidence_refs[0].startswith("opening-")


def test_seeded_lead_priority_and_partial_suit_reach_production_pipeline(rigs) -> None:
    rig = _rig(rigs, lead="left"); rig.bind(); vision = rig.visions[-1]
    vision.queue(_play(
        "partial-left-opening", "left", ("7?",),
        suit_options=(("7C", "7D"),),
    ))
    update = rig.live.analyze_frame(object(), monotonic_ms=100)
    assert vision.opening_leads[0] is Seat.LEFT
    assert update.snapshot.current_player == "self"
    assert update.snapshot.play_history[-1].cards == ("7?",)
    assert update.snapshot.play_history[-1].suit_options == (("7C", "7D"),)


def test_provisional_consensus_and_suit_pending_diagnostics_reach_ui_and_audit(rigs) -> None:
    rig = _rig(rigs, lead="right"); rig.bind(); advice = rig.advisers[-1]
    advice.next_status = AdviceRuntimeStatus.ADVICE
    advice.release = True
    advice.next_diagnostic = {
        "status": "consensus",
        "recommendation_status": "provisional_consensus",
        "world_set_fingerprint": "world-set-1",
        "world_count": 2,
    }
    rig.live.commit_trusted_action(
        actor="right", cards=("3S",), is_pass=False, monotonic_ms=100,
    )
    rig.live.commit_trusted_action(
        actor="opposite", is_pass=True, monotonic_ms=110,
    )
    rig.live.commit_trusted_action(
        actor="left", is_pass=True, monotonic_ms=120,
    )
    ready = rig.live.poll_deadlines()
    assert ready.advice is not None and ready.advice.visible
    assert ready.advice.suit_uncertain is True
    assert ready.advice.variant_count == 2

    rig2 = _rig(rigs, lead="right"); rig2.bind(); blocked = rig2.advisers[-1]
    blocked.next_status = AdviceRuntimeStatus.BLOCKED
    blocked.release = True
    blocked.next_diagnostic = {
        "code": "unknown_suit_recommendation_disagreement",
        "status": "suit_pending",
        "world_set_fingerprint": "world-set-2",
        "world_count": 2,
        "divergence": [{"world_fingerprint": "a"}, {"world_fingerprint": "b"}],
    }
    rig2.live.commit_trusted_action(
        actor="right", cards=("3S",), is_pass=False, monotonic_ms=100,
    )
    rig2.live.commit_trusted_action(
        actor="opposite", is_pass=True, monotonic_ms=110,
    )
    rig2.live.commit_trusted_action(
        actor="left", is_pass=True, monotonic_ms=120,
    )
    pending = rig2.live.poll_deadlines()
    assert pending.snapshot.current_player == "self"
    assert pending.advice is not None
    assert pending.advice.status == "withheld"
    assert pending.advice.suit_uncertain is True
    audit = [row for row in rig2.store.advice if row.get("status") == "withheld"]
    assert audit and audit[-1]["diagnostic"]["world_set_fingerprint"] == "world-set-2"


def test_manual_correction_invalidates_old_advice_and_recomputes_new_revision(rigs) -> None:
    rig = _rig(rigs, lead="right"); rig.bind(); advice = rig.advisers[-1]
    advice.release = True
    rig.live.commit_trusted_action(
        actor="right", cards=("3S",), is_pass=False, monotonic_ms=100,
    )
    rig.live.commit_trusted_action(
        actor="opposite", is_pass=True, monotonic_ms=110,
    )
    rig.live.commit_trusted_action(
        actor="left", is_pass=True, monotonic_ms=120,
    )
    before = rig.live.snapshot
    corrected = rig.live.correct_latest(cards=("4S",), is_pass=False)
    assert corrected.snapshot.revision == before.revision + 1
    assert rig.advisers[-1] is not advice
    assert rig.advisers[-1].submissions[-1][0].version.state_revision == corrected.snapshot.revision
    assert any(row.get("status") == "cancelled" for row in rig.store.advice)


def test_visual_correction_rejected_by_downstream_state_enters_review(rigs, monkeypatch) -> None:
    from daguandan_bridge.application.live_v2_rule_session_protocol import RuleSessionRejected
    rig = _rig(rigs, lead="right"); rig.bind()
    rig.live.commit_trusted_action(
        actor="right", cards=("3S",), is_pass=False, monotonic_ms=100,
    )
    action = SimpleNamespace(
        action_id="visual-action", kind=ActionKind.PLAY,
        evidence_origin=EvidenceOrigin.VISUAL, seat=Seat.RIGHT,
        cards=("3S",), suit_options=(("3S",),), confidence=0.9,
        source_candidate=SimpleNamespace(confidence=0.9, diagnostics=()),
    )
    rig.live._register_visual_correction(action)
    monkeypatch.setattr(
        rig.live.rule_session, "correct_latest",
        lambda command: (_ for _ in ()).throw(RuleSessionRejected("downstream conflict")),
    )
    from daguandan_bridge.live_v2.identity import FrameIdentity
    observation = SimpleNamespace(
        confidence=0.99,
        frame=FrameIdentity(rig.store.session_id, 1, 99, 200, "roi", "source"),
    )
    pending = next(iter(rig.live._visual_corrections.values()))
    result = rig.live._commit_visual_correction(
        pending, ("4S",), observation,
    )
    assert result.status == "review_required"
    assert result.block_reason == "visual_correction_requires_review"


def test_waiting_lead_visual_play_atomically_confirms_lead_and_first_action(rigs) -> None:
    rig = _rig(rigs, lead=None); rig.bind(); vision = rig.visions[-1]
    vision.queue(_play(
        "historical-left-opening", "left", ("5H", "5C"),
        action_epoch=84, evidence_ids=("frame-83", "frame-86"),
    ))
    opened = rig.live.analyze_frame(object(), monotonic_ms=860)
    assert opened.status == "running" and not opened.block_reason
    assert opened.snapshot.lead_player == "left"
    assert opened.snapshot.current_player == "self"
    assert opened.snapshot.play_history[-1].cards == ("5H", "5C")
    assert [event.event_type for event in opened.events] == [
        "lead_player_confirmed", "player_played",
    ]
    play = opened.snapshot.play_history[-1]
    assert play.action_metadata["action_epoch"] == 84
    assert play.action_metadata["evidence_ids"] == ["frame-83", "frame-86"]
    assert rig.store.batches[-1] == opened.events

    vision.queue(_play("self-followup", "self", ("6S", "6H")))
    followed = rig.live.analyze_frame(object(), monotonic_ms=960)
    assert followed.snapshot.current_player == "right"
    assert len(followed.snapshot.play_history) == 2


def test_waiting_lead_pass_and_multi_seat_conflict_cannot_choose_lead(rigs) -> None:
    rig = _rig(rigs, lead=None); rig.bind(); vision = rig.visions[-1]
    vision.queue(_play("pass-only", "left", (), kind="pass"))
    passed = rig.live.analyze_frame(object(), monotonic_ms=100)
    assert passed.snapshot.lead_player is None
    assert not passed.snapshot.play_history

    vision.queue(
        _play("left-option", "left", ("5H", "5C")),
        _play("right-conflict", "right", ("6H",)),
    )
    conflicted = rig.live.analyze_frame(object(), monotonic_ms=200)
    assert conflicted.snapshot.lead_player is None
    assert not conflicted.snapshot.play_history
    assert all(batch[0].event_type == "initial_state_confirmed"
               for batch in rig.store.batches)


def test_opening_batch_persistence_failure_leaves_no_confirmed_lead_or_action(rigs) -> None:
    rig = _rig(rigs, lead=None); rig.bind(); vision = rig.visions[-1]
    before = rig.live.snapshot
    rig.store.fail_batches = True
    vision.queue(_play(
        "opening-disk-full", "left", ("5H", "5C"),
        evidence_ids=("disk-a", "disk-b"),
    ))
    failed = rig.live.analyze_frame(object(), monotonic_ms=100)
    assert failed.status == "review_required"
    assert failed.block_reason == "rule_opening_action_failed"
    assert failed.snapshot.revision == before.revision
    assert failed.snapshot.lead_player is None
    assert not failed.snapshot.play_history
    assert len(rig.store.batches) == 1

    rig.store.fail_batches = False
    vision.queue(_play(
        "opening-retry", "left", ("5H", "5C"),
        evidence_ids=("retry-a", "retry-b"),
    ))
    recovered = rig.live.analyze_frame(object(), monotonic_ms=200)
    assert recovered.status == "running"
    assert recovered.snapshot.lead_player == "left"
    assert len(recovered.snapshot.play_history) == 1


def test_persistence_failure_is_visible_and_never_advances_history(rigs) -> None:
    rig = _rig(rigs); rig.bind(); rig.store.fail_batches = True
    before = rig.live.snapshot
    failed = rig.live.commit_trusted_action(
        actor="right", cards=("3D",), is_pass=False, monotonic_ms=100,
    )
    assert failed.snapshot.revision == before.revision
    assert failed.snapshot.turn_id == before.turn_id
    assert failed.snapshot.play_history == before.play_history
    assert failed.status == "review_required" and failed.block_reason


def test_foreign_candidate_is_ignored_without_entering_gap_recovery(rigs) -> None:
    rig = _rig(rigs); rig.bind(); vision = rig.visions[-1]
    vision.queue(_play("out-of-order", "opposite", (), kind="pass"))
    blocked = rig.live.analyze_frame(object(), monotonic_ms=100)
    assert blocked.status == "running" and vision.calls == 1
    assert not blocked.block_reason
    expired = rig.live.analyze_frame(object(), monotonic_ms=8_200)
    assert not expired.block_reason and vision.calls == 2
    vision.queue(_play("recovery", "right", ("3D",)))
    recovered = rig.live.analyze_frame(object(), monotonic_ms=8_300)
    assert vision.calls == 3
    assert recovered.snapshot.current_player == "opposite"
    assert recovered.status == "running" and not recovered.block_reason


def test_correct_latest_is_explicit_rule_correction_without_advancing_turn(rigs) -> None:
    delivered = []
    rig = _rig(rigs, lead="self", on_update=delivered.append); rig.bind()
    rig.live.commit_trusted_action(
        actor="self", cards=("3S",), is_pass=False, monotonic_ms=100,
    )
    before = rig.live.snapshot
    assert rig.advisers[-1].submissions
    old_vision, old_advice = rig.visions[-1], rig.advisers[-1]
    corrected = rig.live.correct_latest(
        cards=("4S",), is_pass=False, reason="operator-correction",
    )
    assert len(rig.rules.corrections) == 1
    assert corrected.event.event_type == "event_correction"
    assert corrected.snapshot.revision == before.revision + 1
    assert corrected.snapshot.turn_id == before.turn_id
    assert corrected.snapshot.play_history[-1].cards == ("4S",)
    assert not (corrected.advice and corrected.advice.visible)
    assert old_vision.closed == old_advice.closed == 1
    assert any(row.get("status") == "cancelled" for row in rig.store.advice)
    assert rig.visions[-1] is not old_vision and rig.advisers[-1] is not old_advice


def test_premature_terminal_health_is_fail(rigs) -> None:
    rig = _rig(rigs); rig.bind()
    rig.rules.health_override = {
        "schema": "guandan.session-health/1", "status": "FAIL",
        "issues": [{"code": "HEALTH-PREMATURE-GAME-END"}],
    }
    update = rig.live.finish()
    assert rig.store.health[-1]["status"] == "FAIL"
    assert any(item["code"] == "HEALTH-PREMATURE-GAME-END"
               for item in rig.store.health[-1]["issues"])
    assert update.block_reason == "HEALTH-PREMATURE-GAME-END"


def test_recording_failure_does_not_block_same_frame_visual_commit(rigs) -> None:
    rig = _rig(rigs); rig.bind(); rig.recorder.fail = True
    frame = object()
    warning = rig.live.record_frame(frame, monotonic_ms=100, wall_time="now")
    rig.visions[-1].queue(_play("same-frame", "right", ("3D",)))
    update = rig.live.analyze_frame(frame, monotonic_ms=100)
    assert warning and warning.reason == "recording_failed"
    assert update.snapshot.play_history[-1].cards == ("3D",)


def test_generation_rebind_closes_old_workers_preserves_history_and_drops_old_result(rigs) -> None:
    delivered = []
    rig = _rig(rigs, lead="self", on_update=delivered.append); rig.bind(1)
    rig.live.commit_trusted_action(
        actor="self", cards=("3S",), is_pass=False, monotonic_ms=100,
    )
    old_vision, old_advice = rig.visions[-1], rig.advisers[-1]
    assert old_advice.submissions
    rebound = rig.bind(2)
    assert old_vision.closed == old_advice.closed == 1
    assert len(rebound.snapshot.play_history) == 1
    old_advice.release = True
    rig.live.poll_deadlines(); sleep(0.03)
    assert not delivered
    assert not (rig.live.latest_advice and rig.live.latest_advice.visible)


def test_finish_closes_every_worker_and_pump_and_is_idempotent(rigs) -> None:
    rig = _rig(rigs); rig.bind()
    first = rig.live.finish(); second = rig.live.finish()
    assert first.status == second.status == "sealed"
    assert rig.recorder.closed == rig.store.seals == 1
    assert all(worker.closed == 1 for worker in (*rig.visions, *rig.advisers))
    sleep(0.03)
    assert not any(thread.name == "live-v2-advice-pump" for thread in live_threads())
