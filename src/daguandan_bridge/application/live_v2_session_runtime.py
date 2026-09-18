"""Application coordinator for one live-v2 rule session."""

from __future__ import annotations

from dataclasses import dataclass, replace
from datetime import datetime
from threading import RLock
from collections import Counter
from time import monotonic_ns
from typing import Any, Callable

from ..domain.live import LiveEvent, LiveSnapshot
from ..domain.live_runtime import LiveAdvice, LiveStatus, LiveUpdate
from ..live.local_rule_hint import LocalRuleHintTracker
from ..live_v2.action_semantics import ActionSemantics
from ..live_v2.candidates import ActionCandidate, ActionKind, CandidateReason, EvidenceOrigin
from ..live_v2.engine import EngineInput, EngineResult, LiveEngine
from ..live_v2.identity import FrameIdentity, Seat, VersionIdentity
from .live_v2_candidate_gate import gate_visual_candidates
from .live_v2_advice_protocol import AdviceRuntimeResult
from .live_v2_advice_pump import (
    AdviceRuntimeLike, LiveV2AdvicePump, result_matches_opportunity,
)
from .live_v2_rule_session_protocol import RuleBinding, RuleSession
from .live_v2_runtime_journal import LiveV2LifecycleMixin, LiveV2RuntimeJournal
from .live_v2_session_runtime_commands import LiveV2RuleCommandsMixin
from .live_v2_session_runtime_controls import LiveV2ControlMixin
from .live_v2_runtime_updates import (
    VisionRuntimeLike, consume_vision, correction_event, live_update,
    project_engine_result, runtime_result_to_advice, trusted_candidate,
    trusted_to_live_snapshot,
)
from .ports import RecognitionPort, RecordingPort, SessionPersistencePort


class _Clock:
    def __init__(self, provider: Callable[[], int]) -> None:
        self._provider, self._floor = provider, 0

    def advance(self, value: int) -> None:
        self._floor = max(self._floor, int(value))

    def processing_ms(self) -> int:
        return max(self._floor, int(self._provider()))


@dataclass
class _VisualCorrection:
    action_id: str
    seat: Seat
    cards: tuple[str, ...]
    suit_options: tuple[tuple[str, ...], ...]
    last_signature: tuple[object, ...] | None
    rejected_signature: tuple[object, ...] | None
    streak: int
    last_frame: FrameIdentity | None
    original_confidence: float
    allow_complete_rewrite: bool
    allow_expansion: bool


def _normalized_options(
    cards: tuple[str, ...],
    options: tuple[tuple[str, ...], ...],
) -> tuple[tuple[str, ...], ...]:
    if len(cards) != len(options):
        return tuple((card,) for card in cards)
    return tuple(
        tuple(sorted(dict.fromkeys(str(value) for value in choices if str(value))))
        or (card,)
        for card, choices in zip(cards, options, strict=True)
    )


def _same_cards(left: tuple[str, ...], right: tuple[str, ...]) -> bool:
    return sorted(str(card) for card in left) == sorted(str(card) for card in right)


def _repair_signature(
    cards: tuple[str, ...],
    options: tuple[tuple[str, ...], ...],
) -> tuple[object, ...]:
    normalized = _normalized_options(cards, options)
    return tuple(
        sorted(
            (card, choices)
            for card, choices in zip(cards, normalized, strict=True)
        )
    )


def _resolve_visual_correction(
    target_cards: tuple[str, ...], target_options: tuple[tuple[str, ...], ...],
    observed_cards: tuple[str, ...], observed_options: tuple[tuple[str, ...], ...],
    *, allow_complete_rewrite: bool = False, allow_expansion: bool = False,
) -> tuple[str, ...] | None:
    """Return only a reread that improves an uncertain formal action.

    Automatic visual repair is not a general history editor.  It may resolve
    unknown cards, add cards to an explicitly incomplete action, or replace a
    low-confidence complete action with an equally-sized stronger reread.  It
    must never shrink a complete surface or reinterpret settlement cards as an
    old play.
    """

    target = tuple(str(card) for card in target_cards)
    observed = tuple(str(card) for card in observed_cards)
    if not target or not observed or any(card.endswith("?") for card in observed):
        return None
    if _repair_signature(target, target_options) == _repair_signature(
        observed, observed_options
    ):
        return None
    target_partial = any(card.endswith("?") for card in target) or any(
        len(choices) != 1 for choices in _normalized_options(target, target_options)
    )
    if len(observed) < len(target):
        return None
    observed_counter = Counter(observed)
    known_target = Counter(card for card in target if not card.endswith("?"))
    if target_partial and known_target - observed_counter:
        return None
    if allow_expansion and len(observed) > len(target):
        # The old action must be a physical subset of the newer surface; this
        # accepts an early animation read such as 77 -> 777888, but rejects
        # settlement surfaces such as small_joker -> QQ88.
        exact_target = Counter(card for card in target if not card.endswith("?"))
        if not exact_target - observed_counter:
            return observed
    if not target_partial and not allow_complete_rewrite:
        return None
    if allow_complete_rewrite and not target_partial and len(observed) != len(target):
        return None
    return observed


class LiveV2SessionRuntime(
    LiveV2LifecycleMixin, LiveV2RuleCommandsMixin, LiveV2ControlMixin
):
    """Coordinate injected rules and capture-scoped worker runtimes."""

    def __init__(
        self, *, rule_session: RuleSession, store: SessionPersistencePort,
        recorder: RecordingPort, recognition_service: RecognitionPort,
        vision_factory: Callable[[VersionIdentity], VisionRuntimeLike],
        advice_runtime_factory: Callable[[VersionIdentity], AdviceRuntimeLike],
        on_update: Callable[[LiveUpdate], None] | None = None,
        processing_clock_ms: Callable[[], int] | None = None,
        roi_version: str = "live-v2", source_id: str = "live-v2-capture",
        local_hint_window_ms: int = 200,
        synchronous_vision: bool = False,
    ) -> None:
        if not isinstance(rule_session, RuleSession):
            raise TypeError("rule_session must implement RuleSession")
        self.rule_session, self.store, self.recorder = rule_session, store, recorder
        self.recognition_service = recognition_service
        self.status: LiveStatus = "initializing"
        self.latest_advice: LiveAdvice | None = None
        self.automatic_log_delivery_result: dict[str, object] | None = None
        self._vision_factory, self._advice_factory = vision_factory, advice_runtime_factory
        self._on_update = on_update
        self._clock = _Clock(processing_clock_ms or (lambda: monotonic_ns() // 1_000_000))
        self._roi_version, self._source_id = roi_version, source_id
        self._local_hint_window_ms = local_hint_window_ms
        self._synchronous_vision = bool(synchronous_vision)
        self._journal = LiveV2RuntimeJournal(store)
        self._lock = RLock()
        self._engine: LiveEngine | None = None
        self._vision: VisionRuntimeLike | None = None
        self._advice_pump: LiveV2AdvicePump | None = None
        self._pending: dict[str, ActionCandidate] = {}
        opportunity_keys = ("total", "valid", "late", "no_result", "unrecoverable")
        self._opportunity_metrics = {f"opportunity_{key}": 0 for key in opportunity_keys}
        self._initialized = False
        self._generation = self._sequence = self._frame_sequence = self._last_ms = 0
        self._hint = LocalRuleHintTracker()
        self._status_before_pause: LiveStatus | None = None
        self._opening_required = False
        self._visual_corrections: dict[str, _VisualCorrection] = {}
        self._last_visual_repair_seat: Seat | None = None
        self._suppressed_correction_surfaces: dict[Seat, tuple[str, ...]] = {}
        self._terminal_control: str | None = None
        self._terminal_streak = 0
        self._terminal_last_frame: FrameIdentity | None = None
        self._terminal_detected = False
        self._terminal_event: LiveEvent | None = None
        self._aux_event_sequence = 0

    @property
    def snapshot(self) -> LiveSnapshot:
        return trusted_to_live_snapshot(self._trusted_snapshot())

    @property
    def needs_first_action_frames(self) -> bool:
        return self.status == "waiting_lead"

    def start(
        self, *, round_level: str, hand: tuple[str, ...], lead_player: str | None,
        monotonic_ms: int, wall_time: str | None = None,
        historical_scan: bool = False,
    ) -> LiveUpdate:
        del historical_scan
        with self._lock:
            if self._initialized:
                raise RuntimeError("live-v2 session already started")
            self._clock.advance(monotonic_ms)
            self._last_ms = monotonic_ms
            self.rule_session.initialize(
                round_level=round_level, hand=hand,
                lead_player=None if lead_player is None else Seat(lead_player),
                monotonic_ms=monotonic_ms, wall_time=wall_time,
                capture_generation=0, evidence_id="initial-state",
            )
            self._initialized = True
            self._opening_required = lead_player is None
            self.status = "waiting_lead" if self._opening_required else "running"
            self.store.update_runtime_identity(self._identity())
            self._journal.lifecycle("session_started", lead_player=lead_player)
            return self._plain_update()

    def bind_capture_generation(self, generation: int) -> LiveUpdate:
        with self._lock:
            self._require_initialized()
            if isinstance(generation, bool) or generation < 0:
                raise ValueError("capture generation must be non-negative")
            if generation < self._generation:
                raise ValueError("capture generation cannot move backwards")
            if generation == 0 and self._engine is None:
                return self._plain_update()
            if generation == self._generation and self._engine is not None:
                return self._plain_update()
            detached = self._detach_workers()
        self._close_detached(detached)
        with self._lock:
            binding = self.rule_session.bind_generation(generation)
            self._generation = generation
            self._frame_sequence = 0
            self._hint.reset()
            self._visual_corrections.clear()
            self._last_visual_repair_seat = None
            self._suppressed_correction_surfaces.clear()
            self._install_workers(binding)
            self.store.update_runtime_identity(self._identity())
            self._journal.lifecycle("capture_generation_bound", generation=generation)
            return self._process(EngineInput(captured_watermark_ms=self._last_ms))

    def analyze_frame(
        self, frame: Any, *, monotonic_ms: int, metrics: Any | None = None,
        trace_context: dict[str, object] | None = None,
    ) -> LiveUpdate:
        del metrics
        # Snapshot the formal state under the session lock, but do not hold that
        # lock during synchronous image analysis.  Advice results can then be
        # accepted while the next frame is being recognized instead of waiting
        # behind a long sequence of frame calls and becoming artificially stale.
        with self._lock:
            self._require_engine()
            if self.status == "paused":
                return self._plain_update(block_reason="paused")
            if self.status in {"finalizing", "sealed"}:
                self._visual_corrections.clear()
                return self._plain_update(block_reason=self.status)
            self._ensure_engine_current()
            identity = self._capture_identity(trace_context, monotonic_ms)
            if identity is None:
                return self._plain_update(block_reason="stale_capture_identity")
            snapshot = self._engine.state.snapshot
            version = self._engine.state.version
            expected = snapshot.current_seat
            self._expire_visual_corrections(expected)
            formal_action_boundary = (
                snapshot.play_history[-1].last_frame
                if snapshot.play_history else None
            )
            repair_seats = self._ordered_visual_repair_seats()
            vision = self._vision
            processing_ms = self._clock.processing_ms()

        results, faults = consume_vision(
            vision, frame, frame=identity, version=version,
            wild_rank=snapshot.wild_rank,
            expected_seat=expected, processing_ms=processing_ms,
            formal_action_boundary=formal_action_boundary,
            repair_seats=repair_seats,
            synchronous=self._synchronous_vision,
        )

        with self._lock:
            self._require_engine()
            if self.status in {"finalizing", "sealed"}:
                self._visual_corrections.clear()
                return self._plain_update(block_reason=self.status)
            self._ensure_engine_current()
            current = self._engine.state.version
            if (
                current.session_id,
                current.capture_generation,
                current.state_revision,
                current.turn_index,
            ) != (
                version.session_id,
                version.capture_generation,
                version.state_revision,
                version.turn_index,
            ):
                return self._plain_update(block_reason="state_changed_during_frame_analysis")
            for fault in faults:
                self._safe_fault(
                    "vision_runtime", fault, monotonic_ms=identity.captured_ms
                )
            fast = results[-1].fast_signals if results else None
            terminal = self._observe_terminal_control(
                fast, frame=(results[-1].frame if results else None),
            )
            if terminal is not None:
                return terminal
            observations = tuple(
                item for result in results for item in result.observations
            )
            candidates = tuple(
                item for result in results for item in result.candidates
            )
            repaired = self._try_visual_corrections(
                observations, expected_seat=expected, fast=fast,
            )
            if repaired is not None:
                return repaired
            candidates = self._filter_suppressed_correction_surfaces(
                candidates, observations
            )
            gated = gate_visual_candidates(snapshot, candidates)
            if self._opening_required and not snapshot.play_history:
                opening = (
                    self._commit_visual_opening(gated.selected)
                    if len(gated.selected) == 1 and snapshot.lead_seat is None
                    else None
                )
                if opening is not None:
                    self._opening_required = not bool(opening.snapshot.play_history)
                    return opening
                return self._plain_update(
                    fast=fast,
                    block_reason="opening_waiting_for_unique_visual_action",
                )
            selected = gated.selected
            if selected:
                selected_seats = {item.seat for item in selected}
                for action_id, pending in tuple(self._visual_corrections.items()):
                    if pending.seat in selected_seats:
                        self._visual_corrections.pop(action_id, None)
            self._pending.update((item.candidate_id, item) for item in selected)
            return self._process(
                EngineInput(
                    observations=observations,
                    candidates=selected,
                    captured_watermark_ms=identity.captured_ms,
                ),
                fast=fast,
            )

    def _observe_terminal_control(
        self, fast: Any | None, *, frame: FrameIdentity | None,
    ) -> LiveUpdate | None:
        """Persist one terminal event after two independent settlement frames.

        Terminal evidence is intentionally independent of rule gaps: an
        incomplete action history must never prevent sealing reproducible
        evidence once Tencent's settlement controls are stably visible.
        """

        if fast is None or frame is None or self._terminal_detected:
            return None
        control = str(getattr(fast, "game_end_control", "") or "")
        if control not in {"continue_game", "change_table"}:
            self._terminal_control = None
            self._terminal_streak = 0
            self._terminal_last_frame = frame
            return None
        previous = self._terminal_last_frame
        newer = bool(
            previous is not None
            and frame.session_id == previous.session_id
            and frame.capture_generation == previous.capture_generation
            and frame.frame_sequence > previous.frame_sequence
            and frame.captured_ms > previous.captured_ms
        )
        if control == self._terminal_control and newer:
            self._terminal_streak += 1
        else:
            self._terminal_control = control
            self._terminal_streak = 1
        self._terminal_last_frame = frame
        if self._terminal_streak < 2:
            return None

        snapshot = self._trusted_snapshot()
        self._aux_event_sequence += 1
        event = LiveEvent(
            event_id=f"AUX-LIVEV2-{self._aux_event_sequence:06d}",
            event_type="game_end_detected",
            session_id=snapshot.version.session_id,
            seq=0,
            monotonic_ms=frame.captured_ms,
            wall_time=datetime.now().astimezone().isoformat(),
            trick_id=max(1, snapshot.trick_index),
            turn_id=max(1, snapshot.version.turn_index + 1),
            actor=None,
            payload={
                "control": control,
                "remaining_cards": {
                    item.seat.value: int(item.count) for item in snapshot.remaining
                },
                "finished_seats": [seat.value for seat in snapshot.finished],
            },
            confidence=1.0,
            source="live_v2_terminal_control",
            state_revision_before=snapshot.version.state_revision,
            state_revision_after=snapshot.version.state_revision,
            evidence_refs=(
                f"{frame.session_id}:{frame.capture_generation}:"
                f"{frame.frame_sequence}:{frame.source_id}:game-end",
            ),
        )
        try:
            self.store.append_event(event)
        except Exception as exc:
            self._safe_fault(
                "game_end_persistence_failed", str(exc), control=control,
                frame_sequence=frame.frame_sequence,
            )
            return self._plain_update(
                fast=fast, block_reason="game_end_persistence_failed"
            )
        if self._advice_pump is not None:
            self._advice_pump.cancel_pending(reason="game_end_detected")
        self._terminal_detected = True
        self._terminal_event = event
        self._visual_corrections.clear()
        self.latest_advice = None
        self.status = "finalizing"
        self._sequence += 1
        return live_update(
            status=self.status, snapshot=snapshot, sequence=self._sequence,
            advice=None, events=(event,), fast_signals=fast,
        )

    def _register_visual_correction(self, action: Any) -> None:
        if (
            getattr(action, "kind", None) is not ActionKind.PLAY
            or getattr(action, "evidence_origin", None) is not EvidenceOrigin.VISUAL
        ):
            return
        partial = bool(getattr(action, "partial_suits", False)) or any(
            str(card).endswith("?") for card in tuple(action.cards)
        )
        source_candidate = getattr(action, "source_candidate", action)
        confidence = float(getattr(source_candidate, "confidence", 1.0))
        diagnostics = set(getattr(source_candidate, "diagnostics", ()) or ())
        low_confidence = confidence < 0.80 or bool(diagnostics & {
            "play_confidence", "play_quality", "play_annotation_count",
            "play_annotation_min_confidence",
        })
        action_id = str(action.action_id)
        # At most one repair window per seat is useful.  A newer formal play
        # supersedes an older surface that was never repaired.
        for existing_id, pending in tuple(self._visual_corrections.items()):
            if pending.seat is action.seat:
                self._visual_corrections.pop(existing_id, None)
        self._visual_corrections[action_id] = _VisualCorrection(
            action_id=action_id,
            seat=action.seat,
            cards=tuple(action.cards),
            suit_options=tuple(tuple(item) for item in action.suit_options),
            last_signature=None,
            rejected_signature=None,
            streak=0,
            last_frame=None,
            original_confidence=confidence,
            allow_complete_rewrite=low_confidence and not partial,
            allow_expansion=True,
        )

    def _ordered_visual_repair_seats(self) -> tuple[Seat, ...]:
        """Rotate bounded repair priority so no pending seat is starved."""

        seats = tuple(dict.fromkeys(
            item.seat for item in self._visual_corrections.values()
        ))
        if not seats:
            self._last_visual_repair_seat = None
            return ()
        if self._last_visual_repair_seat in seats:
            start = seats.index(self._last_visual_repair_seat) + 1
            seats = seats[start:] + seats[:start]
        self._last_visual_repair_seat = seats[0]
        return seats

    def _expire_visual_corrections(self, expected_seat: Seat | None) -> None:
        """Close old repair windows at a seat boundary or terminal state."""

        if expected_seat is None:
            self._visual_corrections.clear()
            self._last_visual_repair_seat = None
            return
        for action_id, pending in tuple(self._visual_corrections.items()):
            if pending.seat is expected_seat:
                self._visual_corrections.pop(action_id, None)

    def _try_visual_corrections(
        self, observations: tuple[Any, ...], *,
        expected_seat: Seat | None, fast: Any = None
    ):
        if not self._visual_corrections:
            return None
        # Prefer the latest formal action while retaining older seats until
        # their displayed surface is explicitly cleared or replaced.
        for pending in reversed(tuple(self._visual_corrections.values())):
            if pending.seat is expected_seat:
                continue
            for observation in observations:
                if observation.seat is not pending.seat:
                    continue
                kind = str(getattr(observation.kind, "value", observation.kind))
                if kind in {"empty", "pass"}:
                    self._visual_corrections.pop(pending.action_id, None)
                    break
                if kind != "play":
                    continue
                observed_cards = tuple(str(card) for card in observation.cards)
                observed_options = tuple(
                    tuple(str(value) for value in choices)
                    for choices in observation.suit_options
                )
                corrected = _resolve_visual_correction(
                    pending.cards,
                    pending.suit_options,
                    observed_cards,
                    observed_options,
                    allow_complete_rewrite=pending.allow_complete_rewrite,
                    allow_expansion=pending.allow_expansion,
                )
                if (
                    corrected is not None
                    and pending.allow_complete_rewrite
                    and float(observation.confidence)
                    <= pending.original_confidence + 0.05
                ):
                    corrected = None
                if corrected is None:
                    pending.last_signature = None
                    pending.streak = 0
                    pending.last_frame = observation.frame
                    continue
                signature = _repair_signature(corrected, observed_options)
                if signature == pending.rejected_signature:
                    continue
                if (
                    pending.last_signature == signature
                    and pending.last_frame is not None
                    and observation.frame.frame_sequence
                    > pending.last_frame.frame_sequence
                ):
                    pending.streak += 1
                else:
                    pending.streak = 1
                pending.last_signature = signature
                pending.last_frame = observation.frame
                # All action-wide changes use the same two-frame rule.  The
                # rule backend then validates the corrected action plus every
                # downstream event before the replacement is adopted.
                if pending.streak >= 2:
                    repaired = self._commit_visual_correction(
                        pending, corrected, observation, fast=fast
                    )
                    if repaired is not None:
                        return repaired
        return None

    def _filter_suppressed_correction_surfaces(
        self, candidates: tuple[ActionCandidate, ...],
        observations: tuple[Any, ...],
    ) -> tuple[ActionCandidate, ...]:
        for seat, cards in tuple(self._suppressed_correction_surfaces.items()):
            for observation in observations:
                if observation.seat is not seat:
                    continue
                kind = str(getattr(observation.kind, "value", observation.kind))
                if kind == "empty":
                    self._suppressed_correction_surfaces.pop(seat, None)
                    break
                if kind == "play" and _same_cards(cards, tuple(observation.cards)):
                    break
                if kind == "play":
                    self._suppressed_correction_surfaces.pop(seat, None)
                    break
        return tuple(
            item for item in candidates
            if not (
                item.seat in self._suppressed_correction_surfaces
                and _same_cards(
                    self._suppressed_correction_surfaces[item.seat],
                    tuple(item.cards),
                )
            )
        )

    def _commit_visual_correction(
        self, pending: "_VisualCorrection", corrected: tuple[str, ...],
        observation: Any, *, fast: Any = None,
    ):
        from ..live_v2.corrections import CorrectionCommand, CorrectionReason
        evidence_id = (
            f"visual-correction:{pending.action_id}:"
            f"{observation.frame.frame_sequence}"
        )
        try:
            correction = self.rule_session.correct_latest(CorrectionCommand(
                correction_id=f"correction:{pending.action_id}:{observation.frame.frame_sequence}",
                expected_version=self.rule_session.version,
                target_action_id=pending.action_id,
                kind=ActionKind.PLAY,
                cards=corrected,
                suit_options=tuple((card,) for card in corrected),
                reason=CorrectionReason.VISUAL_REREAD,
                evidence_id=evidence_id,
                evidence_origin=EvidenceOrigin.VISUAL,
                confidence=float(observation.confidence),
                corrected_ms=int(observation.frame.captured_ms),
            ))
        except Exception as exc:
            pending.rejected_signature = pending.last_signature
            pending.streak = 0
            self._safe_fault(
                "visual_correction", str(exc),
                target_action_id=pending.action_id,
                old_cards=list(pending.cards),
                proposed_cards=list(corrected),
                observation_confidence=float(observation.confidence),
                rejection_type=type(exc).__name__,
            )
            # A rejected reread is only non-authoritative visual evidence.
            # Keep listening for a different stable reread instead of blocking
            # the whole session.
            return None
        # The correction is now durable. Terminalize only the old logical
        # opportunity; keep both prewarmed worker processes alive. Their hosts
        # bind the new revision on the next request and discard old completions.
        if self._advice_pump is not None:
            self._advice_pump.cancel_pending(
                reason="visual_correction_superseded", preserve_worker=True
            )
        self._visual_corrections.pop(pending.action_id, None)
        self._suppressed_correction_surfaces[pending.seat] = tuple(corrected)
        binding = self.rule_session.bind_generation(self._generation)
        self._reset_engine(binding)
        update = self._process(
            EngineInput(captured_watermark_ms=self._last_ms), fast=fast
        )
        event = correction_event(correction, snapshot=self._trusted_snapshot())
        target = next(
            item for item in self.rule_session.confirmed_actions
            if item.action_id == pending.action_id
        )
        target_event = self.rule_session.events_for_actions((target,))[0]
        event = replace(
            event, payload={**event.payload, "target_event_id": target_event.event_id},
        )
        return replace(update, event=event, events=(event,))

    def commit_trusted_action(
        self, *, actor: str, cards: tuple[str, ...] = (), is_pass: bool,
        monotonic_ms: int, evidence_refs: tuple[str, ...] = (),
        suit_options: tuple[tuple[str, ...], ...] = (),
        action_metadata: dict[str, object] | None = None,
        confidence: float = 1.0, source: str = "trusted_log_replay",
    ) -> LiveUpdate:
        if action_metadata is not None:
            self._safe_audit("trusted_action_metadata", source=source, metadata=action_metadata)
        origin = EvidenceOrigin.MANUAL if source == "manual" else EvidenceOrigin.TRUSTED
        return self._commit_trusted(
            Seat(actor), cards, is_pass, monotonic_ms, confidence, origin,
            CandidateReason.LOCAL_ACTION_CONFIRMED, evidence_refs, suit_options,
            ActionSemantics.requested_from_metadata(action_metadata),
        )

    def _install_workers(self, binding: RuleBinding) -> None:
        self.latest_advice = None
        self._pending.clear()
        vision = self._vision_factory(binding.version)
        runtime = self._advice_factory(binding.version)
        pump = LiveV2AdvicePump(
            runtime, snapshot_provider=lambda: self._engine.state.snapshot,
            on_result=self._accept_advice,
            on_local_pass=self._accept_local_pass,
            on_failure=lambda message: self._safe_fault("advice", message),
            store=self.store, metrics=self._opportunity_metrics,
            local_hint_window_ms=self._local_hint_window_ms,
            processing_clock_ms=self._clock.processing_ms,
        )
        engine = self._new_engine(binding, pump)
        self._engine, self._vision, self._advice_pump = engine, vision, pump
        try:
            vision.start()
            pump.start()
        except BaseException:
            self._close_detached(self._detach_workers())
            raise

    def _new_engine(self, binding: RuleBinding, pump: LiveV2AdvicePump) -> LiveEngine:
        return LiveEngine(
            initial_version=binding.version, projector=binding.adapter,
            committer=binding.adapter, state_provider=binding.adapter,
            clock=self._clock, journal=self._journal, advice_consumer=pump,
            initial_captured_ms=self._last_ms,
        )

    def _reset_engine(self, binding: RuleBinding) -> None:
        if self._advice_pump is None:
            raise RuntimeError("advice runtime is not active")
        self.latest_advice = None
        self._pending.clear()
        self._engine = self._new_engine(binding, self._advice_pump)

    def _process(self, incoming: EngineInput, *, fast: Any = None) -> LiveUpdate:
        self._require_engine()
        self._ensure_engine_current()
        return self._from_engine(self._engine.process(incoming), fast=fast)

    def _ensure_engine_current(self) -> None:
        if self._engine and self._engine.state.version.update_sequence < self._sequence:
            target = replace(self._engine.state.version, update_sequence=self._sequence)
            self._from_engine(self._engine.process(EngineInput(rebind_version=target)))

    def _from_engine(self, result: EngineResult, *, fast: Any = None) -> LiveUpdate:
        confirmed = result.update.confirmed_actions if result.update else ()
        events = self.rule_session.events_for_actions(confirmed) if confirmed else ()
        for action in confirmed:
            self._pending.pop(action.source_candidate.candidate_id, None)
        projected, self.status, self.latest_advice, self._sequence = project_engine_result(
            result=result, snapshot=self._engine.state.snapshot, action_events=events,
            status=self.status, latest_advice=self.latest_advice,
            sequence=self._sequence, fast_signals=fast,
        )
        if self._engine.state.snapshot.current_seat is None:
            self._visual_corrections.clear()
            self._last_visual_repair_seat = None
        else:
            for action in confirmed:
                self._register_visual_correction(action)
        return projected

    def _commit_trusted(
        self, seat: Seat, cards: tuple[str, ...], is_pass: bool, captured_ms: int,
        confidence: float, origin: EvidenceOrigin, reason: CandidateReason,
        evidence_refs: tuple[str, ...], suit_options: tuple[tuple[str, ...], ...],
        requested_semantics: ActionSemantics | None = None,
    ) -> LiveUpdate:
        with self._lock:
            self._require_engine()
            self._ensure_engine_current()
            self._clock.advance(captured_ms)
            self._last_ms = max(self._last_ms, captured_ms)
            candidate = trusted_candidate(
                version=self._engine.state.version, seat=seat, cards=cards,
                is_pass=is_pass, captured_ms=captured_ms,
                processing_ms=self._clock.processing_ms(), confidence=confidence,
                origin=origin, reason=reason, evidence_refs=evidence_refs,
                sequence=self._sequence + 1, suit_options=suit_options,
                requested_semantics=requested_semantics,
            )
            return self._process(EngineInput(
                candidates=(candidate,), captured_watermark_ms=captured_ms,
            ))

    def _accept_advice(self, result: AdviceRuntimeResult) -> bool | str:
        with self._lock:
            if self.status in {"finalizing", "sealed"}:
                return "session_terminal"
            if self._engine is None:
                return "advice_runtime_unavailable"
            opportunity = self._engine.state.opportunity.current
            current = self._engine.state.version
            if not result_matches_opportunity(result, current, opportunity):
                snapshot = self._engine.state.snapshot
                if (
                    current.state_revision > result.identity.version.state_revision
                    and snapshot.play_history
                    and snapshot.play_history[-1].seat is Seat.SELF
                ):
                    return "superseded_by_self_action"
                if current.state_revision != result.identity.version.state_revision:
                    return "superseded_by_state_change"
                if current.capture_generation != result.identity.version.capture_generation:
                    return "capture_generation_changed"
                return "opportunity_closed"
            self.latest_advice = runtime_result_to_advice(
                result, self._engine.state.snapshot
            )
            update = self._plain_update()
        if self._on_update:
            self._on_update(update)
        return True

    def _plain_update(
        self, *, block_reason: str = "", fast: Any = None,
        local_rule_hint: Any = None,
    ) -> LiveUpdate:
        self._require_initialized()
        self._sequence += 1
        return live_update(
            status=self.status, snapshot=self._trusted_snapshot(),
            sequence=self._sequence, advice=self.latest_advice,
            fast_signals=fast, local_rule_hint=local_rule_hint,
            block_reason=block_reason,
        )

    def _rule_failure(self, operation: str, exc: Exception) -> LiveUpdate:
        self.status = "review_required"
        if operation == "opening_action" and not self._trusted_snapshot().play_history:
            self._opening_required = True
        self._safe_fault("rule_session", str(exc), operation=operation)
        return self._plain_update(block_reason=f"rule_{operation}_failed")

    def _trusted_snapshot(self):
        self._require_initialized()
        if self._engine is not None:
            return self._engine.state.snapshot
        actions = self.rule_session.confirmed_actions
        corrections = self.rule_session.correction_history
        captured_ms = max(
            self._last_ms,
            actions[-1].captured_ms if actions else 0,
            corrections[-1].corrected_ms if corrections else 0,
        )
        return self.rule_session.snapshot(captured_ms=captured_ms)

    def _detach_workers(self) -> tuple[LiveV2AdvicePump | None, VisionRuntimeLike | None]:
        detached = self._advice_pump, self._vision
        self._advice_pump = self._vision = self._engine = None
        self.latest_advice = None
        return detached

    @staticmethod
    def _close_detached(
        detached: tuple[LiveV2AdvicePump | None, VisionRuntimeLike | None]
    ) -> None:
        pump, vision = detached
        if pump is not None:
            pump.close()
        if vision is not None:
            vision.close()

    def _safe_audit(self, kind: str, **context: Any) -> None:
        try:
            self._journal.lifecycle(kind, **context)
        except Exception:
            pass

    def _identity(self) -> dict[str, object]:
        return {"runtime": "live_v2", "capture_generation": self._generation,
                "update_sequence": self._sequence}

    def _require_initialized(self) -> None:
        if not self._initialized:
            raise RuntimeError("live-v2 session is not active")

    def _require_engine(self) -> None:
        self._require_initialized()
        if self._engine is None or self.status == "sealed":
            raise RuntimeError("capture generation is not bound")


__all__ = ["LiveV2SessionRuntime"]
