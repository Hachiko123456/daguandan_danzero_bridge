"""Auditable operator commands for the live-v2 session coordinator."""

from dataclasses import replace

from ..live_v2.action_semantics import ActionSemantics
from ..live_v2.candidates import ActionKind, CandidateReason, EvidenceOrigin
from ..live_v2.corrections import CorrectionCommand, CorrectionReason
from ..live_v2.engine import EngineInput
from ..live_v2.identity import Seat
from .live_v2_rule_session_protocol import RuleSessionError
from .live_v2_runtime_updates import correction_event, manual_confirmation_candidate


class LiveV2RuleCommandsMixin:
    """Manual/opening commands routed through the injected RuleSession."""

    def _commit_visual_opening(self, candidates: tuple):
        snapshot = self.snapshot
        if snapshot.lead_player is not None or snapshot.play_history:
            return None
        if len(candidates) != 1:
            return None
        candidate = candidates[0]
        if (
            candidate.kind is not ActionKind.PLAY
            or candidate.reason is not CandidateReason.STABLE_PLAY
            or candidate.evidence_origin is not EvidenceOrigin.VISUAL
        ):
            return None
        try:
            committed = self.rule_session.confirm_opening_action(
                candidate, processing_ms=self._clock.processing_ms()
            )
        except RuleSessionError as exc:
            return self._rule_failure("opening_action", exc)
        self.status = "running"
        self._reset_engine(committed.binding)
        update = self._process(EngineInput(
            captured_watermark_ms=candidate.last_captured_ms
        ))
        register = getattr(self, "_register_visual_correction", None)
        if callable(register):
            register(committed.action)
        return replace(
            update, event=committed.events[-1], events=committed.events
        )

    def bootstrap_opening_action(
        self, *, actor: str, cards: tuple[str, ...], expected_next_player: str,
        monotonic_ms: int, confidence: float, source: str,
    ):
        seat, order = Seat(actor), tuple(Seat)
        if Seat(expected_next_player) is not order[(order.index(seat) + 1) % len(order)]:
            return self._plain_update(block_reason="opening_next_player_mismatch")
        if self.snapshot.lead_player is None:
            self.confirm_lead_player(actor)
        update = self._commit_trusted(
            seat, cards, False, monotonic_ms, confidence, EvidenceOrigin.OPENING,
            CandidateReason.OPENING_ACTION_CONFIRMED,
            (f"opening-{source}-{seat.value}-{monotonic_ms}",), (),
        )
        if update.snapshot.current_player != expected_next_player:
            return self._plain_update(block_reason="opening_next_player_mismatch")
        return update

    def confirm_candidate(self, candidate_id: str):
        with self._lock:
            self._require_engine()
            item = self._pending.get(candidate_id)
            if item is None:
                return self._plain_update(block_reason="unknown_candidate")
            self._ensure_engine_current()
            self._reset_engine(self.rule_session.bind_generation(self._generation))
            captured_ms = self._clock.processing_ms()
            candidate = manual_confirmation_candidate(
                item, version=self._engine.state.version, captured_ms=captured_ms,
                processing_ms=captured_ms, sequence=self._sequence + 1,
            )
            return self._process(EngineInput(
                candidates=(candidate,), captured_watermark_ms=captured_ms,
            ))

    def confirm_manual_action(
        self, *, cards: tuple[str, ...] = (), is_pass: bool,
        action_metadata: dict[str, object] | None = None,
    ):
        current = self.snapshot.current_player
        if current is None:
            return self._plain_update(block_reason="no_current_player")
        return self._commit_trusted(
            Seat(current), cards, is_pass, self._clock.processing_ms(), 1.0,
            EvidenceOrigin.MANUAL, CandidateReason.LOCAL_ACTION_CONFIRMED, (), (),
            ActionSemantics.requested_from_metadata(action_metadata),
        )

    def correct_latest(
        self, *, cards: tuple[str, ...] = (), is_pass: bool,
        reason: str = "one_click_correction",
        action_metadata: dict[str, object] | None = None,
    ):
        with self._lock:
            self._require_initialized()
            actions = self.rule_session.confirmed_actions
            if not actions:
                return self._plain_update(block_reason="no_action_to_correct")
            corrected_ms, target = self._clock.processing_ms(), actions[-1]
            evidence_id = f"manual:correction:{target.action_id}:{self._sequence + 1}"
            self._safe_audit("correction_requested", requested_reason=reason)
            try:
                correction = self.rule_session.correct_latest(CorrectionCommand(
                    correction_id=f"correction:{target.action_id}:{self._sequence + 1}",
                    expected_version=self.rule_session.version,
                    target_action_id=target.action_id,
                    kind=ActionKind.PASS if is_pass else ActionKind.PLAY,
                    cards=() if is_pass else cards,
                    suit_options=() if is_pass else tuple((card,) for card in cards),
                    reason=CorrectionReason.MANUAL_REVIEW,
                    evidence_id=evidence_id, evidence_origin=EvidenceOrigin.MANUAL,
                    confidence=1.0, corrected_ms=corrected_ms,
                    requested_semantics=ActionSemantics.requested_from_metadata(
                        action_metadata
                    ),
                ))
            except RuleSessionError as exc:
                return self._rule_failure("correction", exc)
            if self._advice_pump is not None:
                self._advice_pump.cancel_pending(
                    reason="manual_correction_superseded", preserve_worker=True
                )
            binding = self.rule_session.bind_generation(self._generation)
            if self._generation > 0:
                self._reset_engine(binding)
                update = self._process(
                    EngineInput(captured_watermark_ms=self._last_ms)
                )
            else:
                update = self._plain_update()
            event = correction_event(correction, snapshot=self._trusted_snapshot())
            return replace(update, event=event, events=(event,))

    def confirm_lead_player(self, lead_player: str):
        with self._lock:
            self._require_initialized()
            try:
                binding = self.rule_session.confirm_lead(
                    Seat(lead_player), monotonic_ms=self._clock.processing_ms(),
                    evidence_id=f"opening:lead:{lead_player}:{self._sequence + 1}",
                )
            except RuleSessionError as exc:
                return self._rule_failure("confirm_lead", exc)
            self.status = "running"
            detached = self._detach_workers() if self._engine is not None else (None, None)
        self._close_detached(detached)
        with self._lock:
            if self._generation > 0:
                self._install_workers(binding)
                return self._process(EngineInput(captured_watermark_ms=self._last_ms))
            return self._plain_update()


__all__ = ["LiveV2RuleCommandsMixin"]
