"""Application coordinator for one live-v2 rule session."""

from __future__ import annotations

from dataclasses import replace
from threading import RLock
from time import monotonic_ns
from typing import Any, Callable

from ..domain.live import LiveSnapshot
from ..domain.live_runtime import LiveAdvice, LiveStatus, LiveUpdate
from ..live.local_rule_hint import LocalRuleHintTracker
from ..live_v2.action_semantics import ActionSemantics
from ..live_v2.candidates import ActionCandidate, CandidateReason, EvidenceOrigin
from ..live_v2.engine import EngineInput, EngineResult, LiveEngine
from ..live_v2.identity import Seat, VersionIdentity
from .live_v2_advice_protocol import AdviceRuntimeResult
from .live_v2_advice_pump import (
    AdviceRuntimeLike, LiveV2AdvicePump, result_matches_opportunity,
)
from .live_v2_rule_session_protocol import RuleBinding, RuleSession
from .live_v2_runtime_journal import LiveV2LifecycleMixin, LiveV2RuntimeJournal
from .live_v2_session_runtime_commands import LiveV2RuleCommandsMixin
from .live_v2_session_runtime_controls import LiveV2ControlMixin
from .live_v2_runtime_updates import (
    VisionRuntimeLike, consume_vision, live_update, project_engine_result,
    runtime_result_to_advice, trusted_candidate, trusted_to_live_snapshot,
)
from .ports import RecognitionPort, RecordingPort, SessionPersistencePort


class _Clock:
    def __init__(self, provider: Callable[[], int]) -> None:
        self._provider, self._floor = provider, 0

    def advance(self, value: int) -> None:
        self._floor = max(self._floor, int(value))

    def processing_ms(self) -> int:
        return max(self._floor, int(self._provider()))


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
            self.status = "waiting_lead" if lead_player is None else "running"
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
            self._install_workers(binding)
            self.store.update_runtime_identity(self._identity())
            self._journal.lifecycle("capture_generation_bound", generation=generation)
            return self._process(EngineInput(captured_watermark_ms=self._last_ms))

    def analyze_frame(
        self, frame: Any, *, monotonic_ms: int, metrics: Any | None = None,
        trace_context: dict[str, object] | None = None,
    ) -> LiveUpdate:
        del metrics
        with self._lock:
            self._require_engine()
            if self.status == "paused":
                return self._plain_update(block_reason="paused")
            if self.status in {"finalizing", "sealed"}:
                return self._plain_update(block_reason=self.status)
            self._ensure_engine_current()
            identity = self._capture_identity(trace_context, monotonic_ms)
            if identity is None:
                return self._plain_update(block_reason="stale_capture_identity")
            snapshot = self._engine.state.snapshot
            expected = snapshot.current_seat
            formal_action_boundary = (
                snapshot.play_history[-1].last_frame
                if snapshot.play_history else None
            )
            results, faults = consume_vision(
                self._vision, frame, frame=identity, version=self._engine.state.version,
                wild_rank=snapshot.wild_rank,
                expected_seat=expected, processing_ms=self._clock.processing_ms(),
                formal_action_boundary=formal_action_boundary,
            )
            for fault in faults:
                self._safe_fault("vision_runtime", fault, monotonic_ms=identity.captured_ms)
            observations = tuple(item for result in results for item in result.observations)
            candidates = tuple(item for result in results for item in result.candidates)
            opening = self._commit_visual_opening(candidates)
            if opening is not None:
                return opening
            self._pending.update((item.candidate_id, item) for item in candidates)
            return self._process(
                EngineInput(observations=observations, candidates=candidates,
                            captured_watermark_ms=identity.captured_ms),
                fast=results[-1].fast_signals if results else None,
            )

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

    def _accept_advice(self, result: AdviceRuntimeResult) -> bool:
        with self._lock:
            if self.status in {"finalizing", "sealed"} or self._engine is None:
                return False
            opportunity = self._engine.state.opportunity.current
            current = self._engine.state.version
            if not result_matches_opportunity(result, current, opportunity):
                return False
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
