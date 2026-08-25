from __future__ import annotations

import shutil
import logging
import hashlib
import sys
from collections import Counter, deque
from dataclasses import dataclass, field, replace
from datetime import datetime
from functools import wraps
from inspect import signature
from pathlib import Path
from threading import Event, RLock
from typing import Any, Callable, Literal

import cv2
import numpy as np

from ..action_semantics import project_play_type
from ..application.ports import (
    AdvicePort,
    RecognitionPort,
    RecordingPort,
    SessionPersistencePort,
)
from ..domain.advice import LocalAdvice, StrategyExecutionTrace
from ..domain.recognition import (
    PLAY_REGION_TO_SEAT,
    FastSignalResult,
    OpeningSignal,
    PlayRegionResult,
)
from ..domain.recording import RecorderWarning
from ..danzero.rules import (
    infer_best_action,
    logical_action_label,
    wildcard_substitutions,
)
from ..danzero.state import GameStateError, GuanDanState, Seat
from .action_uncertainty import state_variants_for_action_semantics
from .consensus import (
    BurstConsensus,
    canonical_candidate,
    ConsensusCandidate,
    ConsensusContext,
    ConsensusResult,
    RecognitionSample,
)
from .card_uncertainty import (
    is_unknown_suit_card,
    normalized_suit_options,
    state_variants_for_unknown_suits_detailed,
)
from .display_text import reasons_text
from .recognition_strategy import (
    RecognitionStrategy,
    coerce_recognition_strategy,
    decide_best_effort_candidate,
    decide_recognition_strategy,
    has_exhausted_valid_candidates,
    strategy_spec,
)
from .models import LiveEvent, LiveSnapshot
from .latest_worker import LatestOnlyWorker
from .reducer import LiveReducer
from .suit_correction import SuitCorrectionObservation, SuitCorrectionTracker
from .turns import TURN_ORDER, next_active_seat, project_trick_turn
from .zone_lifecycle import ZoneDecision, ZoneFrameMetrics, ZoneLifecycle, ZonePhase


LiveStatus = Literal[
    "initializing",
    "waiting_lead",
    "running",
    "review_required",
    "paused",
    "finalizing",
    "sealed",
]

_MAX_ADVICE_STATE_VARIANTS = 32
_MAX_SUIT_STATE_VARIANTS = 256
_VISUAL_FINISH_WITHHOLD_REASON = "visual_finish_without_complete_history"
_VISUAL_FINISH_WITHHOLD_TEXT = "牌局历史不完整，暂停推荐"
_TURN_DESYNCHRONIZED_REASON = "turn_desynchronized"
_TURN_DESYNCHRONIZED_TEXT = "牌局历史存在缺口，暂停推荐和预选"
_TURN_RECOVERY_WITHHOLD_REASON = "turn_recovery_pending"
_TURN_RECOVERY_WITHHOLD_TEXT = "牌局历史待恢复，暂停推荐"
_PREVIOUS_ACTION_VERIFICATION_WITHHOLD_REASON = "previous_action_reread_pending"
_PREVIOUS_ACTION_VERIFICATION_WITHHOLD_TEXT = "上一手牌面待复核，暂停推荐"
_PREVIOUS_ACTION_VERIFICATION_TIMEOUT_MS = 1_200
_HANDOFF_GLOBAL_GRACE_MS = 3_000
_HANDOFF_READABLE_CONFIRMATION_MS = 1_000
# A local turn normally has about 15 seconds.  A delayed foreign action must
# be given a substantial part of that budget before this turn is deferred, but
# it must never turn a recoverable visual transition into a session-wide stop.
_ADVICE_RECOVERY_TARGET_MS = 8_000
_LOGGER = logging.getLogger(__name__)


def _state_synchronized(method):
    @wraps(method)
    def synchronized(self, *args, **kwargs):
        with self._state_lock:
            return method(self, *args, **kwargs)

    return synchronized


@dataclass(frozen=True)
class ReviewCandidate:
    candidate_id: str
    cards: tuple[str, ...]
    is_pass: bool
    votes: int
    confidence: float
    valid: bool
    rejected_reason: str = ""


@dataclass(frozen=True)
class ReviewRequest:
    reason: str
    player: Seat
    candidates: tuple[ReviewCandidate, ...]
    evidence_refs: tuple[str, ...] = ()


@dataclass(frozen=True)
class LiveUpdate:
    status: LiveStatus
    snapshot: LiveSnapshot
    event: LiveEvent | None = None
    events: tuple[LiveEvent, ...] = ()
    advice: object | None = None
    review: ReviewRequest | None = None
    fast_signals: FastSignalResult | None = None


@dataclass(frozen=True)
class AdviceRequestKey:
    session_id: str
    turn_id: int
    state_revision: int

    @property
    def request_id(self) -> str:
        return f"ADV-{self.turn_id:04d}-{self.state_revision:04d}"

    @property
    def decision_id(self) -> str:
        return f"{self.session_id}:turn_{self.turn_id}:revision_{self.state_revision}"


@dataclass(frozen=True)
class LiveAdvice:
    key: AdviceRequestKey
    status: Literal["requested", "ready", "stale", "failed", "withheld"]
    advice: LocalAdvice | None = None
    visible: bool = False
    error: str = ""
    withhold_reason: str = ""
    suit_uncertain: bool = False
    variant_count: int = 1
    advice_agrees_across_variants: bool = True
    semantic_uncertain: bool = False
    semantic_source_history_indices: tuple[int, ...] = ()
    suit_variant_count: int = 1
    suit_equivalence_class_count: int = 1
    semantic_variant_count: int = 1


@dataclass
class _TurnOwnershipWindow:
    """Evidence ownership for one immutable expected-turn snapshot.

    Fast active-player recognition is intentionally treated as a guard around
    the slower play-region recognizer.  Candidate cards observed before the
    owner is authenticated remain provisional.  Once a delayed-action window
    is open, that same owner's ROI can be reread until its next action would
    overwrite the visual evidence.
    """

    key: tuple[str, int, int, Seat]
    expected_player: Seat
    active_player: Seat | None = None
    owner_active_streak: int = 0
    handoff_active_player: Seat | None = None
    handoff_active_streak: int = 0
    handoff_detected_ms: int | None = None
    handoff_global_deadline_ms: int | None = None
    handoff_readable_since_ms: int | None = None
    handoff_local_deadline_ms: int | None = None
    handoff_unreadable_since_ms: int | None = None
    handoff_sample_count: int = 0
    handoff_block_reason: str = ""
    handoff_last_block_reason: str = ""
    handoff_deadline_kind: str | None = None
    # A visible expected play can survive just long enough for the following
    # player to PASS and the timer to reach the seat after that.  Keep this
    # bounded recovery separate from a normal direct handoff: it requires a
    # two-frame, seat-bound PASS marker before it can reconstruct history.
    crossed_handoff_recovery_pending: bool = False
    crossed_handoff_recovery_active_player: Seat | None = None
    crossed_handoff_recovery_pass_player: Seat | None = None
    crossed_handoff_recovery_detected_ms: int | None = None
    crossed_handoff_recovery_pass_marker_streak: int = 0
    # When an expected action has not reached consensus, retain its visual
    # evidence until that same player becomes active again.  The next action
    # from that player overwrites the only reliable recovery surface.
    turn_recovery_pending: bool = False
    turn_recovery_detected_ms: int | None = None
    turn_recovery_active_player: Seat | None = None
    turn_recovery_active_streak: int = 0
    turn_recovery_owner_return_streak: int = 0
    turn_recovery_pass_marker_streaks: dict[Seat, int] = field(
        default_factory=dict
    )
    turn_recovery_advice_withheld: bool = False
    turn_recovery_advice_event_id: str | None = None
    # The physical UI may expose local controls before the reducer can safely
    # replay all foreign actions.  This is an advice deadline, not an action
    # recognition deadline: recovery continues after it and can restore a
    # later FableDan request as soon as the canonical state is complete.
    turn_recovery_local_started_ms: int | None = None
    turn_recovery_local_deadline_ms: int | None = None
    turn_recovery_target_exceeded: bool = False
    sample_allowed: bool = False
    crossing_active_player: Seat | None = None
    crossing_active_streak: int = 0
    crossing_non_owner_streak: int = 0
    authenticated: bool = False
    disposition: str = "unseen"
    provisional_samples: list[RecognitionSample] = field(default_factory=list)
    handoff_samples: list[RecognitionSample] = field(default_factory=list)
    # This is deliberately a separate, pass-only recovery state.  Unlike an
    # authenticated owner handoff it never makes a card ROI sample eligible
    # for consensus.
    unseen_direct_next_pass_pending: bool = False
    unseen_direct_next_pass_detected_ms: int | None = None
    unseen_direct_next_pass_deadline_ms: int | None = None
    unseen_direct_next_pass_marker_visible: bool = False
    unseen_direct_next_pass_edge_seen: bool = False
    unseen_direct_next_pass_marker_streak: int = 0
    unseen_direct_next_pass_marker_player: Seat | None = None
    # The active-seat template can disappear for a transition frame.  Retain
    # only a *new* expected-seat PASS edge seen in that unknown interval so it
    # can be confirmed once the direct next seat becomes readable.
    expected_pass_marker_edge_while_active_unknown: bool = False
    # When a finishing leader's partner has already received the timer, the
    # final active opponent's PASS can still be proved from a two-frame,
    # seat-bound marker.  This is narrower than generic crossed handoff
    # recovery: the reducer must prove that this exact PASS closes a wind
    # catch, and no card ROI is ever admitted.
    wind_catch_pass_recovery_pending: bool = False
    wind_catch_pass_recovery_receiver: Seat | None = None
    wind_catch_pass_recovery_detected_ms: int | None = None
    wind_catch_pass_recovery_deadline_ms: int | None = None
    wind_catch_pass_recovery_marker_streak: int = 0
    # The PASS surface may flash for one frame while the wind receiver's next
    # play is already visible but the active-seat decoration still lags.  This
    # recovery is intentionally stricter than timer handoff: it needs that
    # prior seat-bound PASS edge and two identical, legal receiver reads.
    wind_catch_receiver_play: RecognitionSample | None = None
    wind_catch_receiver_play_streak: int = 0
    # Kept independently of the pending state so one owner-provisional frame
    # can establish the false baseline for the immediately following timer
    # handoff without making any card sample eligible.
    last_expected_pass_marker_visible: bool = False
    last_expected_pass_marker_player: Seat | None = None
    accepted_sample_count: int = 0
    isolated_sample_count: int = 0
    desynchronized: bool = False


@dataclass(frozen=True)
class _OwnershipResolution:
    event: LiveEvent
    events: tuple[LiveEvent, ...]


@dataclass
class _PreviousActionVerification:
    """One constrained reread opened by the immediately following action.

    An action is armed as soon as it is recorded.  It is intentionally not
    reread while the next player is still acting: the target play can still be
    covered by its own animation then.  Only the next legal action opens its
    short verification window, so the target is still the penultimate formal
    action when a correction is considered.
    """

    target: LiveEvent
    expected_followup_actor: Seat
    followup_event_id: str | None = None
    state: Literal["armed", "open", "confirmed", "expired"] = "armed"
    opened_monotonic_ms: int | None = None
    last_probe_monotonic_ms: int | None = None
    advice_withheld: bool = False


@dataclass(frozen=True)
class LiveMetrics:
    frame_count: int
    confirmed_action_count: int
    review_count: int
    recognition_sample_count: int
    advice_visible_latency_ms: int | None

    def to_dict(self) -> dict[str, object]:
        return {
            "frame_count": self.frame_count,
            "confirmed_action_count": self.confirmed_action_count,
            "review_count": self.review_count,
            "recognition_sample_count": self.recognition_sample_count,
            "advice_visible_latency_ms": self.advice_visible_latency_ms,
        }


@dataclass(frozen=True)
class _AdviceJob:
    key: AdviceRequestKey
    state: GuanDanState


@dataclass(frozen=True)
class _AdviceCompletion:
    key: AdviceRequestKey
    advice: LocalAdvice | None = None
    error: str = ""
    engine_input: dict[str, object] | None = None
    trace: dict[str, object] | None = None
    suit_uncertain: bool = False
    variant_count: int = 1
    advice_agrees_across_variants: bool = True
    semantic_uncertain: bool = False
    semantic_source_history_indices: tuple[int, ...] = ()
    suit_variant_count: int = 1
    suit_equivalence_class_count: int = 1
    semantic_variant_count: int = 1
    uncertainty_diagnostics: dict[str, object] | None = None


def _state_semantic_choices(
    state: GuanDanState,
    source_history_indices: tuple[int, ...],
) -> list[dict[str, object]]:
    choices: list[dict[str, object]] = []
    for history_index in source_history_indices:
        if not 1 <= history_index <= len(state.play_history):
            continue
        event = state.play_history[history_index - 1]
        metadata = event.action_metadata or {}
        selected = metadata.get("selected_interpretation")
        choices.append(
            {
                "source_history_index": history_index,
                "physical_cards": list(event.cards),
                "selected_interpretation": (
                    dict(selected) if isinstance(selected, dict) else None
                ),
            }
        )
    return choices


def _readable_advice(advice: LocalAdvice) -> str:
    if advice.is_pass:
        return "PASS（不出）"
    cards = " ".join(advice.cards) or "无牌面"
    return f"{advice.play_type}：{cards}"


def _recognition_retry_message(
    reason: str,
    observations: list[dict[str, object]],
) -> str:
    candidates: list[str] = []
    for item in reversed(observations):
        cards = tuple(str(card) for card in item.get("cards", ()))
        if bool(item.get("is_pass", False)):
            readable = "PASS（不出）"
        elif cards:
            readable = " ".join(cards)
        else:
            readable = "未识别到牌面"
        if readable not in candidates:
            candidates.append(readable)
        if len(candidates) >= 3:
            break
    candidate_text = "、".join(candidates) if candidates else "没有可用候选"
    return (
        f"{reasons_text(reason)}；最近读取到：{candidate_text}。"
        "这些结果未通过确认，未写入正式牌局历史"
    )


def _deduplicate_advisor_input_variants(
    states: tuple[GuanDanState, ...],
    advisor: AdvicePort,
) -> tuple[GuanDanState, ...]:
    fingerprint = getattr(advisor, "decision_input_fingerprint", None)
    if not callable(fingerprint):
        return states
    unique: dict[object, GuanDanState] = {}
    for index, state in enumerate(states, start=1):
        try:
            key = fingerprint(
                state,
                request_id=f"suit-equivalence-{index}",
            )
        except Exception:
            # Keep unmappable states separate so normal inference records the
            # exact branch-specific error instead of hiding it during dedup.
            key = ("unmappable", index)
        unique.setdefault(key, state)
    return tuple(unique.values())


_RULE_CONTRADICTION_CODES = {
    "history_move_does_not_beat_lead",
    "observed_move_invalid",
}


def _validate_and_deduplicate_encoded_advice_variants(
    variants: list[tuple[GuanDanState, int, int]],
    advisor: AdvicePort,
    source_indices: tuple[int, ...],
) -> tuple[
    list[tuple[GuanDanState, int, int]],
    list[dict[str, object]],
    list[dict[str, object]],
]:
    fingerprint = getattr(advisor, "decision_input_fingerprint", None)
    if not callable(fingerprint):
        return variants, [], []
    unique: dict[object, tuple[GuanDanState, int, int]] = {}
    eliminated: list[dict[str, object]] = []
    validation_errors: list[dict[str, object]] = []
    for index, variant in enumerate(variants, start=1):
        try:
            key = fingerprint(
                variant[0],
                request_id=f"complete-equivalence-{index}",
            )
        except Exception as exc:
            raw_diagnostic = getattr(exc, "diagnostic", None)
            diagnostic = (
                dict(raw_diagnostic) if isinstance(raw_diagnostic, dict) else {}
            )
            item = {
                "input_variant_index": index,
                "suit_variant_index": variant[1],
                "semantic_variant_index": variant[2],
                "semantic_choices": _state_semantic_choices(
                    variant[0],
                    source_indices,
                ),
                "error_type": type(exc).__name__,
                "error": str(exc),
                "diagnostic": diagnostic,
            }
            if diagnostic.get("code") in _RULE_CONTRADICTION_CODES:
                item["status"] = "eliminated_by_complete_history"
                eliminated.append(item)
            else:
                item["status"] = "input_validation_failed"
                validation_errors.append(item)
            continue
        unique.setdefault(key, variant)
    return list(unique.values()), eliminated, validation_errors


def _semantic_choice_text(item: dict[str, object]) -> str:
    choices = item.get("semantic_choices", ())
    labels: list[str] = []
    if isinstance(choices, (list, tuple)):
        for choice in choices:
            if not isinstance(choice, dict):
                continue
            selected = choice.get("selected_interpretation")
            if not isinstance(selected, dict):
                continue
            history_index = choice.get("source_history_index", "?")
            label = str(selected.get("logical_label", "") or "").strip()
            if not label:
                label = (
                    f"{selected.get('move_type', '未知牌型')}"
                    f"(key={selected.get('key', '?')})"
                )
            labels.append(f"历史第 {history_index} 条={label}")
    return "、".join(labels) or "无显式语义选择"


def _compact_input_failure_text(
    failures: list[dict[str, object]],
) -> tuple[str, str]:
    unique: list[dict[str, object]] = []
    seen: set[tuple[str, str]] = set()
    for item in failures:
        signature = (_semantic_choice_text(item), str(item.get("error", "")))
        if signature in seen:
            continue
        seen.add(signature)
        unique.append(item)

    def history_index(item: dict[str, object]) -> int:
        diagnostic = item.get("diagnostic")
        if not isinstance(diagnostic, dict):
            return 0
        try:
            return int(diagnostic.get("source_turn_id", 0) or 0)
        except (TypeError, ValueError):
            return 0

    primary = max(unique, key=history_index, default={})
    primary_text = str(primary.get("error", "无法还原合法模型输入"))
    details = "；".join(
        f"{_semantic_choice_text(item)}：{item.get('error', '未知错误')}"
        for item in unique
    )
    return primary_text, details


class LiveOrchestrator:
    """Qt-free coordinator for recording, gating, recognition, and reduction."""

    def __init__(
        self,
        *,
        reducer: LiveReducer,
        store: SessionPersistencePort,
        recorder: RecordingPort,
        recognition_service: RecognitionPort,
        advisor: AdvicePort | None = None,
        settle_ms: int = 0,
        action_timeout_ms: int = 28_000,
        burst_sample_limit: int = 5,
        burst_sample_interval_ms: int = 100,
        minimum_free_bytes: int = 512 * 1024 * 1024,
        lead_wait_timeout_ms: int = 30_000,
        lead_stable_frames: int = 3,
        recognition_strategy: str | RecognitionStrategy = RecognitionStrategy.TWO_VALID_STREAK,
        on_update: Callable[[LiveUpdate], None] | None = None,
    ) -> None:
        if burst_sample_limit < 3:
            raise ValueError("突发读取至少需要 3 帧")
        self.reducer = reducer
        self.store = store
        self.recorder = recorder
        self.recognition_service = recognition_service
        self.advisor = advisor
        self.settle_ms = int(settle_ms)
        self.action_timeout_ms = int(action_timeout_ms)
        self.burst_sample_limit = int(burst_sample_limit)
        self.burst_sample_interval_ms = int(burst_sample_interval_ms)
        self.minimum_free_bytes = int(minimum_free_bytes)
        self.lead_wait_timeout_ms = int(lead_wait_timeout_ms)
        self.lead_stable_frames = max(2, int(lead_stable_frames))
        self.recognition_strategy = coerce_recognition_strategy(recognition_strategy)
        self._update_listener = on_update
        self.consensus = BurstConsensus(min_votes=3)
        self._state_lock = RLock()
        self.status: LiveStatus = "initializing"
        self.latest_review: ReviewRequest | None = None
        self._zone: ZoneLifecycle | None = None
        self._samples: list[RecognitionSample] = []
        self._observations: list[dict[str, object]] = []
        self._last_sample_ms: int | None = None
        self._observation_sequence = 0
        self._last_monotonic_ms = 0
        self._recognition_trace_context: dict[str, object] = {}
        self._recognition_trace_pre_job_key: tuple[object, ...] = ()
        self._baseline_by_seat: dict[Seat, np.ndarray] = {}
        self._previous_by_seat: dict[Seat, np.ndarray] = {}
        self._content_prev_by_seat: dict[Seat, np.ndarray] = {}
        self._all_events: list[LiveEvent] = []
        self._aux_event_sequence = 0
        self._published_sequence = 0
        self._advice_lock = RLock()
        self._requested_advice: set[AdviceRequestKey] = set()
        self._decision_id_by_revision: dict[int, str] = {}
        self._advice_completion_events: dict[AdviceRequestKey, Event] = {}
        self._self_turn_corroborated = False
        self.latest_advice: LiveAdvice | None = None
        self._advice_worker: LatestOnlyWorker | None = None
        self._review_count = 0
        self._advice_requested_at_ms: dict[AdviceRequestKey, int] = {}
        self._advice_visible_latency_ms: int | None = None
        self._advice_withhold_reason: str | None = None
        self._advice_withhold_revision: int | None = None
        self._advice_withhold_finish_event_id: str | None = None
        self._advice_suspended_reason: str | None = None
        self._advice_suspended_turn_key: tuple[str, int, int, Seat] | None = None
        self._accept_advice_results = True
        self._status_before_pause: LiveStatus | None = None
        self._analysis_epoch = 0
        self._recent_incidents: dict[str, tuple[int, Path]] = {}
        self._lead_wait_started_ms: int | None = None
        self._deal_complete_recorded = False
        self._opening_controls_seen = False
        self._lead_candidate: Seat | None = None
        self._lead_candidate_frames = 0
        self._lead_auto_confirmed_from_marker = False
        self._lead_confirmation_frame: tuple[int, np.ndarray] | None = None
        self._lead_stability_frames: deque[tuple[int, np.ndarray]] = deque(maxlen=3)
        self._first_action_pending = False
        # The normal recognition burst is deliberately short-lived: it is
        # discarded whenever the UI effect or action-zone lifecycle changes.
        # Keep a small, independent evidence window for the opening play so a
        # queued live frame cannot split its two confirmation reads across a
        # burst reset.  It is never used after the first action is committed.
        self._first_action_samples: list[RecognitionSample] = []
        self._self_lead_controls_seen = False
        self._self_lead_controls_cleared = False
        self._first_action_gate_reason = "not_started"
        self._turn_ownership_window: _TurnOwnershipWindow | None = None
        # Pass labels persist for a short time after an action.  Keep their
        # previous-frame ownership independently from an expected-turn window
        # so a just-created turn can tell a new direct-next PASS from a stale
        # decoration, regardless of which seat owns that turn.
        self._last_pass_marker_players: frozenset[Seat] = frozenset()
        self._desynchronized_turn_key: tuple[str, int, int, Seat] | None = None
        self._game_end_detected = False
        self._finish_order: list[Seat] = []
        self._placement_streaks: dict[Seat, tuple[str, int]] = {}
        # A left-side read may improve only a previously obscured left action.
        # It is deliberately kept outside the reducer history.  The same
        # result must be observed twice before it becomes a visual correction.
        self._suit_corrected_event_ids: set[str] = set()
        self._suit_correction_tracker = SuitCorrectionTracker()
        # Every non-pass action is armed for one constrained reread.  Its
        # window opens only once the next legal actor has completed an action;
        # this is when the previous play's animation is expected to be gone.
        self._previous_action_verifications: dict[str, _PreviousActionVerification] = {}
        # Formal actions are published immediately after reducer mutation.  A
        # verification retired by that mutation is queued briefly so its
        # lifecycle terminal is persisted *after* the triggering action.
        self._pending_previous_action_retirements: list[
            tuple[str, _PreviousActionVerification]
        ] = []
        if advisor is not None:
            self._advice_worker = LatestOnlyWorker(
                self._run_advice,
                on_result=self._complete_advice_job,
                on_discard=self._discard_advice_job,
            )
            self._advice_worker.start()

    @property
    def needs_first_action_frames(self) -> bool:
        with self._state_lock:
            snapshot = self.reducer.snapshot()
            return bool(
                self.status == "waiting_lead"
                or (
                    self.status == "running"
                    and self._first_action_pending
                    and snapshot.current_player == snapshot.lead_player
                )
            )

    @property
    def snapshot(self) -> LiveSnapshot:
        with self._state_lock:
            return self.reducer.snapshot()

    @property
    def events(self) -> tuple[LiveEvent, ...]:
        with self._advice_lock:
            return tuple(self._all_events)

    @property
    def metrics(self) -> LiveMetrics:
        with self._state_lock:
            action_types = {"player_played", "player_passed", "manual_confirmed_event"}
            return LiveMetrics(
                frame_count=self.recorder.frame_count,
                confirmed_action_count=sum(
                    event.event_type in action_types for event in self.reducer.events
                ),
                review_count=self._review_count,
                recognition_sample_count=self._observation_sequence,
                advice_visible_latency_ms=self._advice_visible_latency_ms,
            )

    @_state_synchronized
    def start(
        self,
        *,
        round_level: str,
        hand: tuple[str, ...],
        lead_player: Seat | None,
        monotonic_ms: int,
    ) -> LiveUpdate:
        if self.status != "initializing":
            raise RuntimeError("实时对局已经启动")
        free_bytes = shutil.disk_usage(self.store.directory).free
        if free_bytes < self.minimum_free_bytes:
            raise RuntimeError(
                f"可用磁盘空间不足：需要 {self.minimum_free_bytes}，实际 {free_bytes}"
            )
        self._last_monotonic_ms = int(monotonic_ms)
        self._recognition_trace_context = {}
        self.store.update_runtime_identity(self._runtime_identity())
        self._game_end_detected = False
        self._finish_order = []
        self._placement_streaks.clear()
        self._advice_withhold_reason = None
        self._advice_withhold_revision = None
        self._advice_withhold_finish_event_id = None
        self._advice_suspended_reason = None
        self._advice_suspended_turn_key = None
        self._turn_ownership_window = None
        self._last_pass_marker_players = frozenset()
        self._desynchronized_turn_key = None
        self._previous_action_verifications.clear()
        self._pending_previous_action_retirements.clear()
        self._first_action_gate_reason = "not_started"
        self._lead_auto_confirmed_from_marker = False
        self._clear_first_action_candidates()
        event = self.reducer.confirm_initial_state(
            round_level=round_level,
            hand=hand,
            lead_player=lead_player,
            source="manual_start_with_initial_metadata",
        )
        event = self._publish_event(event)
        if lead_player is None:
            self.status = "waiting_lead"
            self._lead_wait_started_ms = int(monotonic_ms)
            self._deal_complete_recorded = False
            self._opening_controls_seen = False
            self._lead_candidate = None
            self._lead_candidate_frames = 0
            waiting = self._append_lifecycle_event("waiting_for_lead", {})
            return self._update(event=event, events=(event, waiting))
        self.status = "running"
        self._first_action_pending = True
        self._self_lead_controls_seen = False
        self._self_lead_controls_cleared = lead_player != "self"
        self._activate_zone(int(monotonic_ms))
        turn_started = self._append_lifecycle_event(
            "turn_started",
            {"player": lead_player},
            actor=lead_player,
        )
        self._request_advice_if_needed()
        return self._update(event=event, events=(event, turn_started))

    def ingest_frame(
        self,
        frame: np.ndarray,
        *,
        monotonic_ms: int,
        wall_time: str,
        metrics: ZoneFrameMetrics | None = None,
    ) -> LiveUpdate:
        self.record_frame(
            frame,
            monotonic_ms=monotonic_ms,
            wall_time=wall_time,
        )
        return self.analyze_frame(
            frame,
            monotonic_ms=monotonic_ms,
            metrics=metrics,
        )

    def record_frame(
        self,
        frame: np.ndarray,
        *,
        monotonic_ms: int,
        wall_time: str,
    ) -> RecorderWarning | None:
        with self._state_lock:
            status = self.status
        if status == "sealed":
            raise RuntimeError("对局已经结束")
        if status == "finalizing":
            return None
        warning = self.recorder.write_frame(frame, monotonic_ms, wall_time)
        if warning is not None:
            with self._state_lock:
                self._record_recorder_warning(warning)
        return warning

    def analyze_frame(
        self,
        frame: np.ndarray,
        *,
        monotonic_ms: int,
        metrics: ZoneFrameMetrics | None = None,
        trace_context: dict[str, object] | None = None,
    ) -> LiveUpdate:
        round_finished = False
        terminal_expected: Seat = "self"
        with self._state_lock:
            if self.status == "sealed":
                raise RuntimeError("对局已经结束")
            self._last_monotonic_ms = int(monotonic_ms)
            self._recognition_trace_context = dict(trace_context or {})
            if self.status == "waiting_lead":
                job_key = self._analysis_job_key()
            elif self.status != "running":
                return self._update()
            else:
                snapshot = self.reducer.snapshot()
                expected = snapshot.current_player
                if expected is None:
                    # Three places are known as soon as the third player has
                    # gone out.  Keep polling only terminal controls so the
                    # controller can seal the recording; never create a
                    # fictional fourth-player action or a review request.
                    terminal_expected = snapshot.lead_player or "self"
                    job_key = self._analysis_job_key()
                    round_finished = True
                else:
                    round_finished = False
                    job_key = self._analysis_job_key()
            self._recognition_trace_pre_job_key = job_key

        if self.status == "waiting_lead":
            opening = self._recognize_opening_signal(frame)
            fast = FastSignalResult(
                expected_player="self",
                active_player=opening.active_player,
                pass_visible=False,
                self_action_buttons_visible=opening.self_action_buttons_visible,
                effect_visible=False,
                super_double_visible=opening.super_double_visible,
                game_end_control=opening.game_end_control,
            )
            lead = self._lead_candidate_from_opening(opening)
            with self._state_lock:
                if not self._analysis_job_is_current(job_key):
                    return self._update()
                lead_frame = (int(monotonic_ms), frame.copy())
                self._lead_confirmation_frame = lead_frame
                self._lead_stability_frames.append(lead_frame)
                return self._analyze_waiting_lead(fast, lead, monotonic_ms)

        if round_finished:
            fast = self._recognize_fast_signals(
                frame,
                terminal_expected,
                allow_pass=False,
            )
            with self._state_lock:
                if not self._analysis_job_is_current(job_key):
                    return self._update()
                game_end = self._handle_game_end_control(fast)
                if game_end is not None:
                    return game_end
                return self._update(fast_signals=fast)

        # Vision runs without the state lock so pause/correction/finalize stay instant.
        first_action = bool(
            self._first_action_pending
            and expected == self.reducer.snapshot().lead_player
        )
        # PASS badges are seat-bound visual evidence.  Read all of them in
        # every trick, including a new lead, so recovery can retain a badge's
        # disappearance/reappearance lifecycle.  ``pass_allowed`` remains a
        # formal action rule below: the leader still cannot commit PASS.
        pass_allowed = bool(snapshot.trick_plays)
        fast = self._recognize_fast_signals(
            frame,
            expected,
            allow_pass=True,
        )
        with self._state_lock:
            if not self._analysis_job_is_current(job_key):
                return self._update()
            placement_events = self._apply_visual_placements(
                fast,
                defer_player=expected,
            )
            if placement_events:
                self._analysis_epoch += 1
                self._clear_burst()
                self._activate_zone(monotonic_ms)
                self._request_advice_if_needed()
                return self._update(
                    event=placement_events[-1],
                    events=placement_events,
                    fast_signals=fast,
                )
            game_end = self._handle_game_end_control(fast)
            if game_end is not None:
                return game_end
            # A history gap makes further card/PASS reconstruction unsafe, but
            # terminal controls and visual placement badges remain read-only
            # evidence required to finish the recording cleanly.  Keep those
            # paths above this guard and suspend every action path below it.
            if self._advice_suspended_reason == _TURN_DESYNCHRONIZED_REASON:
                return self._update(fast_signals=fast)
            if fast.super_double_visible:
                self._clear_burst()
                return self._update(fast_signals=fast)
            self._apply_fast_signal(fast)
            if self._self_lead_waiting_for_action(expected, fast):
                self._reset_waiting_self_lead(
                    monotonic_ms,
                    frame=frame,
                    expected=expected,
                )
                return self._update(fast_signals=fast)
            if self._zone is None or self._zone.expected_player != expected:
                self._activate_zone(monotonic_ms)
            assert self._zone is not None
            if metrics is None:
                current_metrics = self._extract_metrics(
                    frame, expected, monotonic_ms, fast
                )
            else:
                current_metrics = ZoneFrameMetrics(
                    monotonic_ms=int(monotonic_ms),
                    occupied=metrics.occupied,
                    motion_score=metrics.motion_score,
                    pass_visible=(metrics.pass_visible or fast.pass_visible) and pass_allowed,
                    effect_visible=metrics.effect_visible or fast.effect_visible,
                    content_changed=getattr(metrics, "content_changed", False),
                )
            if (
                self._first_action_pending
                and expected == self.snapshot.lead_player
                and self._zone.phase == ZonePhase.WAIT_ACTION
                and current_metrics.occupied
                and (
                    expected != "self" or self._self_lead_controls_cleared
                )
            ):
                # 首出动作可能在首出标志消失前就已静止，首回合允许当前
                # 玩家区域的已有牌面直接打开动作窗口。
                current_metrics = ZoneFrameMetrics(
                    monotonic_ms=current_metrics.monotonic_ms,
                    occupied=current_metrics.occupied,
                    motion_score=current_metrics.motion_score,
                    pass_visible=current_metrics.pass_visible,
                    effect_visible=current_metrics.effect_visible,
                    content_changed=True,
                )
            decision = self._zone.observe(current_metrics)
            ownership_resolution = self._observe_turn_ownership(
                expected,
                fast,
                monotonic_ms,
                frame=frame,
                metrics=current_metrics,
                decision=decision,
            )
            if ownership_resolution is not None:
                return self._update(
                    event=ownership_resolution.event,
                    events=ownership_resolution.events,
                    fast_signals=fast,
                )
            if self._desynchronized_turn_key == self._turn_ownership_key():
                return self._update(fast_signals=fast)
            owner_window = self._ensure_turn_ownership_window()
            turn_recovery_pending = bool(
                owner_window is not None and owner_window.turn_recovery_pending
            )
            had_observations = bool(
                self._samples
                or (owner_window is not None and owner_window.provisional_samples)
            )
            if decision.timed_out:
                if turn_recovery_pending:
                    self._restart_turn_recovery_zone(monotonic_ms)
                    return self._update(fast_signals=fast)
                timeout_consensus = self._decide_timeout_candidate(
                    current_metrics,
                    fast,
                )
                if timeout_consensus is not None:
                    event, events = self._commit_consensus(
                        timeout_consensus,
                        monotonic_ms,
                        fast=fast,
                    )
                    return self._update(
                        event=event,
                        events=events,
                        fast_signals=fast,
                    )
                # Keep the rejected burst intact until _require_review has
                # written its diagnostics.  Clearing it first made a genuine
                # observed play look like "no available candidates".
                if not had_observations and not self._first_action_samples:
                    self._activate_zone(monotonic_ms)
                    return self._update(fast_signals=fast)
                return self._require_review(decision.reason, monotonic_ms, fast)
            if decision.discard_burst:
                if turn_recovery_pending:
                    self._restart_turn_recovery_zone(monotonic_ms)
                    return self._update(fast_signals=fast)
                self._clear_burst()
                # A direct-next fallback may replay only the current readable
                # expected-ROI burst.  An effect/motion reset invalidates any
                # handoff samples collected before it.
                if owner_window is not None:
                    owner_window.handoff_samples.clear()
            snapshot = self.reducer.snapshot()
            wild_rank = snapshot.wild_rank
            previous_action_target = self._previous_action_verification_target(snapshot)
            legacy_suit_target = (
                None
                if previous_action_target is not None
                else self._suit_correction_target(snapshot)
            )
            collect_expected_sample = bool(
                decision.collect_sample
                and self._ownership_allows_sample()
                and self._sample_due(monotonic_ms)
            )
            if (
                not collect_expected_sample
                and previous_action_target is None
                and legacy_suit_target is None
            ):
                return self._update(fast_signals=fast)

        # The expected player remains the sole action-commit path.  A previous
        # action is reread only after its next legal actor has committed, when
        # its own visual effect has had time to disappear.  The sidecar is
        # deliberately best effort; only two distinct, identical legal reads
        # can affect formal history.
        previous_action_result = self._probe_previous_action(
            frame,
            target=previous_action_target,
            wild_rank=wild_rank,
        ) if previous_action_target is not None else None
        legacy_suit_result = self._probe_suit_correction(
            frame,
            target=legacy_suit_target,
            wild_rank=wild_rank,
        ) if legacy_suit_target is not None else None
        result = (
            self._recognize_play_region(
                frame,
                expected,
                wild_rank=wild_rank,
                allow_pass=pass_allowed,
            )
            if collect_expected_sample
            else None
        )
        with self._state_lock:
            if not self._analysis_job_is_current(job_key):
                return self._update()
            if self._zone is None or self._zone.expected_player != expected:
                return self._update()
            previous_action_event = self._apply_previous_action_correction(
                previous_action_target,
                previous_action_result,
                monotonic_ms=monotonic_ms,
            )
            if previous_action_event is not None:
                return self._update(
                    event=previous_action_event,
                    events=(previous_action_event,),
                    fast_signals=fast,
                )
            legacy_suit_event = self._apply_suit_correction(
                legacy_suit_target,
                legacy_suit_result,
            )
            if legacy_suit_event is not None:
                return self._update(
                    event=legacy_suit_event,
                    events=(legacy_suit_event,),
                    fast_signals=fast,
                )
            if not collect_expected_sample:
                return self._update(fast_signals=fast)
            assert result is not None
            self._append_sample(
                result,
                monotonic_ms,
                fast=fast,
                metrics=current_metrics,
                decision=decision,
            )
            consensus = self._decide_if_ready(current_metrics, fast)
            handoff_window = self._turn_ownership_window
            if consensus is None:
                consensus = self._decide_opening_handoff_anchor(
                    handoff_window,
                    current_metrics,
                    fast,
                )
                if consensus is not None:
                    self._append_recognition_trace(
                        window=handoff_window,
                        outcome="opening_handoff_anchor_confirmed",
                        strategy_result=consensus,
                        fallback=True,
                        reason="two_distinct_valid_non_pass_reads",
                        commit_attempted=True,
                    )
            if handoff_window is not None and handoff_window.handoff_samples:
                self._append_recognition_trace(
                    window=handoff_window,
                    outcome="strategy_evaluated",
                    strategy_result=consensus,
                )
            if consensus is not None:
                if consensus.status == "confirmed":
                    if (
                        handoff_window is not None
                        and handoff_window.turn_recovery_pending
                    ):
                        if self._confirmed_handoff_precedes_turn_recovery(
                            handoff_window,
                            consensus,
                        ):
                            # The expected ROI already produced two matching,
                            # legal non-pass reads.  A fast active-seat jump
                            # between those reads describes the following
                            # action; it must not invalidate the action whose
                            # pixels are still present in the expected ROI.
                            event, events = self._commit_consensus(
                                consensus,
                                monotonic_ms,
                                fast=fast,
                            )
                            self._append_recognition_trace(
                                window=handoff_window,
                                outcome="confirmed_handoff_before_turn_recovery",
                                strategy_result=consensus,
                                fallback=True,
                                reason="two_matching_expected_roi_reads_precede_active_cross",
                                commit_attempted=True,
                                commit_event_id=event.event_id,
                            )
                            return self._update(
                                event=event,
                                events=events,
                                fast_signals=fast,
                            )
                        if self._turn_recovery_is_ready(
                            handoff_window,
                            consensus,
                            fast,
                        ):
                            event, events = self._commit_turn_recovery(
                                consensus,
                                monotonic_ms,
                                fast=fast,
                            )
                            return self._update(
                                event=event,
                                events=events,
                                fast_signals=fast,
                            )
                        return self._update(fast_signals=fast)
                    if (
                        handoff_window is not None
                        and handoff_window.crossed_handoff_recovery_pending
                    ):
                        if self._crossed_handoff_recovery_is_ready(
                            handoff_window,
                            consensus,
                        ):
                            event, events = self._commit_crossed_handoff_recovery(
                                consensus,
                                monotonic_ms,
                                fast=fast,
                            )
                            return self._update(
                                event=event,
                                events=events,
                                fast_signals=fast,
                            )
                        # A verified play alone cannot fill the missing
                        # action.  Keep sampling until the seat-bound PASS has
                        # its own two-frame confirmation or the guard expires.
                        return self._update(fast_signals=fast)
                    event, events = self._commit_consensus(
                        consensus,
                        monotonic_ms,
                        fast=fast,
                    )
                    if handoff_window is not None and handoff_window.handoff_samples:
                        self._append_recognition_trace(
                            window=handoff_window,
                            outcome="committed",
                            strategy_result=consensus,
                            commit_event_id=event.event_id,
                        )
                    return self._update(
                        event=event,
                        events=events,
                        fast_signals=fast,
                    )
                if consensus.status == "needs_confirmation":
                    if (
                        handoff_window is not None
                        and handoff_window.turn_recovery_pending
                    ):
                        return self._update(fast_signals=fast)
                    update = self._require_review(
                        ",".join(consensus.rejected_reasons) or consensus.status,
                        monotonic_ms,
                        fast,
                        consensus,
                    )
                    return update
            retry_reason = self._recognition_retry_reason(current_metrics, fast)
            if retry_reason is not None:
                if (
                    handoff_window is not None
                    and handoff_window.turn_recovery_pending
                ):
                    return self._update(fast_signals=fast)
                return self._require_review(
                    retry_reason,
                    monotonic_ms,
                    fast,
                )
            return self._update(fast_signals=fast)


    def _analysis_job_key(self) -> tuple[object, ...]:
        snapshot = self.reducer.snapshot()
        return (
            snapshot.session_id,
            snapshot.turn_id,
            snapshot.revision,
            snapshot.current_player,
            self._analysis_epoch,
        )

    def _analysis_job_is_current(self, key: tuple[object, ...]) -> bool:
        return (
            self.status in {"running", "waiting_lead"}
            and key == self._analysis_job_key()
        )

    @_state_synchronized
    def _analyze_waiting_lead(
        self,
        fast: FastSignalResult,
        lead: Seat | None,
        monotonic_ms: int,
    ) -> LiveUpdate:
        if self.status != "waiting_lead":
            return self._update()
        game_end = self._handle_game_end_control(fast)
        if game_end is not None:
            return game_end
        if fast.super_double_visible:
            self._opening_controls_seen = True
            self._deal_complete_recorded = False
            self._lead_candidate = None
            self._lead_candidate_frames = 0
            self._lead_stability_frames.clear()
            self._lead_wait_started_ms = int(monotonic_ms)
            return self._update(fast_signals=fast)
        if self._opening_controls_seen and self._deal_complete_recorded is False:
            self._deal_complete_recorded = True
            self._append_lifecycle_event(
                "deal_complete",
                {"signal": "opening_controls_absent"},
                actor="self",
            )
        if lead is not None:
            if self._lead_candidate == lead:
                self._lead_candidate_frames += 1
            else:
                self._lead_candidate = lead
                self._lead_candidate_frames = 1
                self._lead_stability_frames.clear()
                self._lead_stability_frames.append(self._lead_confirmation_frame)
            if self._lead_candidate_frames >= self.lead_stable_frames:
                self._lead_candidate = None
                self._lead_candidate_frames = 0
                return self._complete_lead(
                    lead,
                    monotonic_ms,
                    auto_confirmed_from_marker=True,
                )
        else:
            self._lead_candidate = None
            self._lead_candidate_frames = 0
            self._lead_stability_frames.clear()
        started = self._lead_wait_started_ms
        if started is not None and int(monotonic_ms) - started >= self.lead_wait_timeout_ms:
            return self._require_lead_review("lead_player_timeout", monotonic_ms, fast)
        return self._update(fast_signals=fast)

    def _complete_lead(
        self,
        lead: Seat,
        monotonic_ms: int,
        *,
        auto_confirmed_from_marker: bool = False,
    ) -> LiveUpdate:
        event = self.reducer.confirm_lead_player(lead)
        event = self._publish_event(event)
        self.status = "running"
        self._lead_auto_confirmed_from_marker = bool(auto_confirmed_from_marker)
        turn_started = self._append_lifecycle_event(
            "turn_started",
            {"player": lead},
            actor=lead,
        )
        self._clear_burst()
        self._clear_first_action_candidates()
        self._first_action_pending = True
        self._self_lead_controls_seen = False
        self._self_lead_controls_cleared = lead != "self"
        self._activate_zone(int(monotonic_ms))
        # Activate first (it clears old per-seat pixels), then retain the
        # pre-lead opening frames as the first action's empty/reference image.
        # Seeding before activation silently discarded the baseline and made a
        # visible first action look static in replay and live capture alike.
        self._seed_first_action_baseline(lead)
        self._request_advice_if_needed()
        return self._update(event=event, events=(event, turn_started))

    def _require_lead_review(
        self,
        reason: str,
        monotonic_ms: int,
        fast: FastSignalResult | None = None,
    ) -> LiveUpdate:
        self._review_count += 1
        self.latest_review = None
        self._lead_wait_started_ms = int(monotonic_ms)
        self._lead_candidate = None
        self._lead_candidate_frames = 0
        event = self._append_lifecycle_event(
            "recognition_retry",
            {"reason": str(reason), "stage": "waiting_lead"},
        )
        self._create_incident(reason, monotonic_ms)
        return self._update(event=event, fast_signals=fast)

    def _recognize_super_double_visible(self, frame: np.ndarray) -> bool:
        """Pre-lead phase only checks the deal completion control and lead mark."""
        method = getattr(self.recognition_service, "recognize_super_double_visible", None)
        if callable(method):
            return bool(method(frame))
        # Compatibility for older plug-ins and test doubles.  Production uses
        # the dedicated method above, which does not inspect a seat/pass ROI.
        return bool(self.recognition_service.recognize_fast_signals(frame, "self").super_double_visible)

    def _recognize_opening_signal(self, frame: np.ndarray) -> OpeningSignal:
        """Use one opening recognizer for live play and pipeline replay.

        The compatibility branch keeps external recognizer plug-ins usable,
        while the project recognizer returns all opening evidence from the
        same screenshot.
        """

        method = getattr(self.recognition_service, "recognize_opening_signal", None)
        if callable(method):
            return method(frame)
        super_double_visible = self._recognize_super_double_visible(frame)
        lead_method = getattr(self.recognition_service, "recognize_lead_player", None)
        marker_player = None if super_double_visible or not callable(lead_method) else lead_method(frame)
        fast = self._recognize_fast_signals(frame, "self", allow_pass=False)
        return OpeningSignal(
            super_double_visible=super_double_visible,
            marker_player=marker_player,
            active_player=fast.active_player,
            self_action_buttons_visible=fast.self_action_buttons_visible,
        )

    @staticmethod
    def _lead_candidate_from_opening(signal: OpeningSignal) -> Seat | None:
        """Resolve only non-conflicting raw opening evidence.

        A marker and a live timer naming different players is a transient
        screen state, not a valid lead.  A single source is allowed through
        the existing consecutive-frame stability gate.
        """

        if signal.super_double_visible:
            return None
        if (
            signal.marker_player is not None
            and signal.active_player is not None
            and signal.marker_player != signal.active_player
        ):
            return None
        if signal.marker_player is not None:
            return signal.marker_player
        if signal.active_player is not None:
            return signal.active_player
        # The local action buttons only say that this client may act.  They do
        # not identify the opening lead: during the transition out of the
        # doubling screen they can remain visible after another seat has
        # already been selected.  Do not manufacture a self lead from that
        # one-sided control; wait for a marker or an active-player signal.
        return None

    def _recognize_fast_signals(
        self,
        frame: np.ndarray,
        expected: Seat,
        *,
        allow_pass: bool,
    ) -> FastSignalResult:
        try:
            return self.recognition_service.recognize_fast_signals(
                frame, expected, allow_pass=allow_pass
            )
        except TypeError as exc:
            if "allow_pass" not in str(exc):
                raise
            return self.recognition_service.recognize_fast_signals(frame, expected)

    def _recognize_play_region(
        self,
        frame: np.ndarray,
        expected: Seat,
        *,
        wild_rank: str,
        allow_pass: bool,
    ) -> PlayRegionResult:
        # Keep integrations and test doubles written against older recognizer
        # signatures usable while preferring the full production call.
        attempts = (
            {"allow_pass": allow_pass, "allow_unknown_suit": True},
            {"allow_pass": allow_pass},
            {"allow_unknown_suit": True},
            {},
        )
        for extra_kwargs in attempts:
            try:
                return self.recognition_service.recognize_play_region(
                    frame,
                    expected,
                    wild_rank=wild_rank,
                    **extra_kwargs,
                )
            except TypeError as exc:
                message = str(exc)
                unsupported = {
                    name
                    for name in extra_kwargs
                    if f"unexpected keyword argument '{name}'" in message
                }
                if unsupported:
                    continue
                raise
        raise TypeError("识别服务不接受任何兼容的出牌识别调用签名")


    @_state_synchronized
    def confirm_lead_player(self, lead_player: Seat) -> LiveUpdate:
        if self.status not in {"waiting_lead", "review_required"}:
            raise RuntimeError("当前不在等待首发阶段")
        snapshot = self.reducer.snapshot()
        if snapshot.lead_player is not None:
            raise RuntimeError("首发座位已经确认")
        if lead_player not in TURN_ORDER:
            raise RuntimeError("首出座位无效")
        return self._complete_lead(lead_player, self._last_monotonic_ms)

    @_state_synchronized
    def confirm_candidate(self, candidate_id: str) -> LiveUpdate:
        review = self.latest_review
        if self.status != "review_required" or review is None:
            raise RuntimeError("当前没有待确认动作")
        candidate = next(
            (item for item in review.candidates if item.candidate_id == candidate_id),
            None,
        )
        if candidate is None:
            raise ValueError("待确认候选不存在")
        if not candidate.valid:
            raise ValueError(
                "候选未通过自动校验，请使用“不出”或“都不对”手动补录"
            )
        before = self.reducer.snapshot()
        event = self._record_action(
            review.player,
            candidate.cards,
            candidate.is_pass,
            confidence=1.0,
            source="manual_one_click_confirmation",
            evidence_refs=review.evidence_refs,
        )
        event, outcomes = self._publish_action_with_outcomes(event, before)
        self._first_action_pending = False
        self._clear_first_action_candidates()
        self._append_lifecycle_event(
            "review_resolved",
            {
                "resolution": "candidate_confirmed",
                "candidate_id": candidate.candidate_id,
                "action_event_id": event.event_id,
            },
            actor=review.player,
        )
        self.status = "running"
        self.latest_review = None
        self._clear_burst()
        self._activate_zone(self._last_monotonic_ms)
        turn_started = self._append_current_turn_started()
        self._request_advice_if_needed()
        events = (event, *outcomes) + ((turn_started,) if turn_started else ())
        return self._update(event=event, events=events)

    @_state_synchronized
    def commit_trusted_action(
        self,
        *,
        actor: Seat,
        cards: tuple[str, ...] = (),
        is_pass: bool,
        monotonic_ms: int,
        evidence_refs: tuple[str, ...] = (),
        suit_options: tuple[tuple[str, ...], ...] = (),
        action_metadata: dict[str, object] | None = None,
        confidence: float = 1.0,
        source: str = "trusted_log_replay",
    ) -> LiveUpdate:
        """Commit a trusted action through the same post-action live path.

        Trusted replay deliberately bypasses vision and consensus because the
        source event has already been confirmed.  State advancement, event
        publication, turn lifecycle, and selected-strategy scheduling remain the same as
        a live consensus commit.
        """

        if self.status != "running":
            raise RuntimeError("当前不在实时对局进行状态")
        expected = self.reducer.snapshot().current_player
        if expected != actor:
            raise ValueError(f"当前应由 {expected} 行动，不能提交 {actor} 的可信动作")
        self._last_monotonic_ms = int(monotonic_ms)
        before = self.reducer.snapshot()
        event = self._record_action(
            actor,
            cards,
            is_pass,
            confidence=float(confidence),
            source=str(source),
            evidence_refs=evidence_refs,
            suit_options=suit_options,
            action_metadata=action_metadata,
        )
        event, outcomes = self._publish_action_with_outcomes(event, before)
        self._first_action_pending = False
        self._clear_first_action_candidates()
        self._activate_zone(int(monotonic_ms))
        turn_started = self._append_current_turn_started()
        self._request_advice_if_needed()
        events = (event, *outcomes) + ((turn_started,) if turn_started else ())
        return self._update(event=event, events=events)

    @_state_synchronized
    def bootstrap_opening_action(
        self,
        *,
        actor: Seat,
        cards: tuple[str, ...],
        expected_next_player: Seat,
        monotonic_ms: int,
        confidence: float,
        source: str,
    ) -> LiveUpdate:
        """Commit a two-frame visual opening anchor before live capture starts.

        The listener may first observe the table after the leader has already
        played.  This narrow bootstrap accepts only that fully anchored first
        action; it validates both sides of the transition before applying the
        ordinary action/recommendation path.  It never consumes a saved
        timeline or advice record.
        """

        snapshot = self.reducer.snapshot()
        if (
            self.status != "running"
            or not self._first_action_pending
            or snapshot.lead_player != actor
            or snapshot.current_player != actor
        ):
            raise RuntimeError("首出动作锚点与当前开局状态不一致")
        if not cards:
            raise ValueError("首出动作锚点没有已识别的牌")
        next_player = next_active_seat(actor, snapshot.finished_seats)
        if next_player != expected_next_player:
            raise ValueError("首出动作锚点的下一行动座位不一致")
        return self.commit_trusted_action(
            actor=actor,
            cards=cards,
            is_pass=False,
            monotonic_ms=monotonic_ms,
            confidence=max(0.0, min(1.0, float(confidence))),
            source="visual_opening_anchor",
            action_metadata={
                "bootstrap": "listener_opening_anchor",
                "recognition_source": str(source),
                "expected_next_player": expected_next_player,
            },
        )

    @_state_synchronized
    def confirm_manual_action(
        self,
        *,
        cards: tuple[str, ...] = (),
        is_pass: bool,
    ) -> LiveUpdate:
        if self.status != "review_required" or self.latest_review is None:
            raise RuntimeError("当前没有待补录动作")
        player = self.latest_review.player
        before = self.reducer.snapshot()
        event = self._record_action(
            player,
            cards,
            is_pass,
            confidence=1.0,
            source="manual_minimal_editor",
        )
        event, outcomes = self._publish_action_with_outcomes(event, before)
        self._first_action_pending = False
        self._clear_first_action_candidates()
        self._append_lifecycle_event(
            "review_resolved",
            {
                "resolution": "manual_action",
                "action_event_id": event.event_id,
            },
            actor=player,
        )
        self.status = "running"
        self.latest_review = None
        self._clear_burst()
        self._activate_zone(self._last_monotonic_ms)
        turn_started = self._append_current_turn_started()
        self._request_advice_if_needed()
        events = (event, *outcomes) + ((turn_started,) if turn_started else ())
        return self._update(event=event, events=events)

    @_state_synchronized
    def correct_latest(
        self,
        *,
        cards: tuple[str, ...] = (),
        is_pass: bool,
        reason: str = "one_click_correction",
    ) -> LiveUpdate:
        actions = [
            event
            for event in self.reducer.events
            if event.event_type in {"player_played", "player_passed", "manual_confirmed_event"}
        ]
        if not actions:
            raise RuntimeError("没有可纠正的正式动作")
        event = self.reducer.correct_event(
            actions[-1].event_id,
            cards=cards,
            is_pass=is_pass,
            reason=reason,
        )
        event = self._publish_event(event)
        self._append_lifecycle_event(
            "correction_applied",
            {
                "correction_event_id": event.event_id,
                "target_event_id": event.payload.get("target_event_id"),
            },
            actor=event.actor,
        )
        self.status = "running"
        self.latest_review = None
        self._clear_burst()
        self._activate_zone(self._last_monotonic_ms)
        self._request_advice_if_needed()
        return self._update(event=event)

    @_state_synchronized
    def pause(self) -> LiveUpdate:
        if self.status in {"running", "review_required", "waiting_lead"}:
            self._status_before_pause = self.status
            self.status = "paused"
            self._analysis_epoch += 1
            self._clear_burst()
            self._clear_first_action_candidates()
            self._zone = None
            self._append_lifecycle_event("session_paused", {})
        return self._update()

    @_state_synchronized
    def resume(self, *, monotonic_ms: int) -> LiveUpdate:
        if self.status != "paused":
            raise RuntimeError("只有暂停状态可以继续")
        self.status = self._status_before_pause or "running"
        self._status_before_pause = None
        self._analysis_epoch += 1
        if self.status == "running":
            self._activate_zone(monotonic_ms)
        self._append_lifecycle_event("session_resumed", {})
        return self._update()

    @_state_synchronized
    def begin_finalizing(self) -> LiveUpdate:
        if self.status == "sealed":
            return self._update()
        self.status = "finalizing"
        self._analysis_epoch += 1
        self._accept_advice_results = False
        self._zone = None
        self._clear_burst()
        self._clear_first_action_candidates()
        return self._update()

    @_state_synchronized
    def capture_interrupted(self, reason: str, *, monotonic_ms: int) -> LiveUpdate:
        self._append_lifecycle_event(
            "capture_interrupted",
            {"reason": str(reason)},
        )
        self._create_incident("capture_interrupted:" + str(reason), monotonic_ms)
        return self.pause()

    @_state_synchronized
    def analysis_failed(self, reason: str, *, monotonic_ms: int) -> LiveUpdate:
        if self.status == "waiting_lead":
            return self._update()
        if self.status == "review_required":
            return self._update()
        return self._require_review(
            "recognition_failed:" + str(reason),
            monotonic_ms,
        )

    @_state_synchronized
    def ingest_fast_signal(
        self,
        *,
        active_player: Seat | None,
        self_action_buttons_visible: bool = False,
        game_end_control: str | None = None,
    ) -> LiveUpdate:
        expected = self.snapshot.current_player or "self"
        fast = FastSignalResult(
            expected_player=expected,
            active_player=active_player,
            pass_visible=False,
            self_action_buttons_visible=bool(self_action_buttons_visible),
            effect_visible=False,
            game_end_control=game_end_control,
        )
        game_end = self._handle_game_end_control(fast)
        if game_end is not None:
            return game_end
        self._apply_fast_signal(fast)
        return self._update(fast_signals=fast)

    def _handle_game_end_control(
        self,
        fast: FastSignalResult,
    ) -> LiveUpdate | None:
        """Publish one terminal-screen signal without sealing in the worker thread."""

        control = fast.game_end_control
        if control not in {"continue_game", "change_table"}:
            return None
        if self.status not in {"waiting_lead", "running", "review_required"}:
            return None
        if self._game_end_detected:
            return self._update(fast_signals=fast)
        self._game_end_detected = True
        self._clear_burst()
        self._clear_first_action_candidates()
        snapshot = self.reducer.snapshot()
        event = self._append_lifecycle_event(
            "game_end_detected",
            {
                "control": control,
                # Terminal UI evidence must carry a self-contained copy of
                # the final card counts.  The training sidecar may use it to
                # order the two players left after a double-down; it never
                # feeds back into the reducer.
                "remaining_cards": {
                    seat: int(snapshot.remaining_cards[seat])
                    for seat in TURN_ORDER
                },
                "finished_seats": [
                    seat for seat in TURN_ORDER if seat in snapshot.finished_seats
                ],
            },
        )
        return self._update(event=event, fast_signals=fast)

    @_state_synchronized
    def start_self_advice(self) -> AdviceRequestKey | None:
        return self._request_advice_if_needed()

    def wait_for_advice(
        self,
        key: AdviceRequestKey,
        *,
        timeout: float = 60.0,
    ) -> LiveAdvice | None:
        """Wait for one advisor job without blocking the GUI thread.

        The trusted replay runs inside its own worker thread.  The event is
        signalled by the normal advisor completion callback, so this method
        does not create a second recommendation path.
        """

        with self._advice_lock:
            completed = self._advice_completion_events.get(key)
        if completed is None:
            return None
        if not completed.wait(max(0.0, float(timeout))):
            return None
        with self._advice_lock:
            advice = self.latest_advice
            if advice is None or advice.key != key:
                return None
            return advice

    def wait_for_advice_idle(self, *, timeout: float = 60.0) -> bool:
        """Drain the production advice worker without stopping or bypassing it."""

        worker = self._advice_worker
        if worker is None:
            return True
        return worker.wait_idle(timeout=max(0.0, float(timeout)))

    def complete_advice(
        self,
        key: AdviceRequestKey,
        advice: LocalAdvice,
    ) -> None:
        self._complete_advice_job(_AdviceCompletion(key=key, advice=advice))

    def finish(self) -> LiveUpdate:
        with self._state_lock:
            if self.status == "sealed":
                return self._update()
            self.status = "finalizing"
            self._analysis_epoch += 1
            self._accept_advice_results = False
            self._append_lifecycle_event("session_finalizing", {})
        if self._advice_worker is not None:
            self._advice_worker.stop(timeout=5.0)
        with self._state_lock:
            recording = self.recorder.close()
            self.store.seal(
                frame_count=recording.frame_count,
                dropped_frames=recording.dropped_frames,
                metrics=self.metrics.to_dict(),
                incident_media_failures=(
                    failure.to_dict()
                    for failure in recording.incident_media_failures
                ),
            )
            self.status = "sealed"
            self._zone = None
            self._clear_burst()
            self._clear_first_action_candidates()
            update = self._update()
        # Creating a review is intentionally best-effort and happens only
        # after sealing.  A data-export problem must never make a real game
        # appear unsealed or disturb live-session finalisation.
        if bool(getattr(self.store, "persistence_enabled", True)):
            try:
                from ..application.fabledan_training_data import FableDanTrainingDataService

                FableDanTrainingDataService().initialize_review(self.store.directory)
            except Exception:
                _LOGGER.warning("FableDan 训练数据待确认文件创建失败", exc_info=True)
        return update

    def _self_lead_waiting_for_action(
        self,
        expected: Seat,
        fast: FastSignalResult,
    ) -> bool:
        if (
            not self._first_action_pending
            or expected != "self"
            or self.snapshot.lead_player != "self"
        ):
            self._first_action_gate_reason = "not_self_opening_turn"
            return False
        # A visible local action bar can be left over for a frame after a
        # successful first play.  A timer that has already moved to another
        # seat is stronger, affirmative evidence than that stale decoration.
        # Check it first so the baseline is retained and the opening action
        # can be recognised from the current frame.
        if fast.active_player not in (None, "self"):
            self._self_lead_controls_cleared = True
            self._first_action_gate_reason = "active_player_not_self"
            return False
        if fast.self_action_buttons_visible:
            self._self_lead_controls_seen = True
            self._first_action_gate_reason = "self_action_buttons_visible"
            return True
        if not self._self_lead_controls_seen:
            # The analysis worker can resume after all opening-control frames
            # have been dropped.  Without positive next-seat evidence, keep
            # waiting instead of treating a missing control as a submitted
            # opening action.
            self._first_action_gate_reason = "awaiting_self_action_evidence"
            return True
        self._self_lead_controls_cleared = True
        self._first_action_gate_reason = "self_action_controls_cleared"
        return False

    def _reset_waiting_self_lead(
        self,
        monotonic_ms: int,
        *,
        frame: np.ndarray | None = None,
        expected: Seat = "self",
    ) -> None:
        # Keep the pre-play reference while self is choosing a card.  Calling
        # ``_activate_zone`` without preservation on every buttons-visible
        # frame erased the opening baseline; the first static played card then
        # became the new baseline and the action window never opened.
        if frame is not None and expected not in self._baseline_by_seat:
            roi = self._play_roi(frame, expected)
            gray = cv2.GaussianBlur(
                cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY),
                (5, 5),
                0,
            )
            self._baseline_by_seat[expected] = gray.copy()
            self._previous_by_seat[expected] = gray
            self._content_prev_by_seat[expected] = self._content_fingerprint(
                frame,
                expected,
            )
        self._clear_burst()
        self._activate_zone(monotonic_ms, preserve_baseline=True)

    def _seed_first_action_baseline(self, player: Seat) -> None:
        frames = self._lead_stability_frames
        self._lead_confirmation_frame = None
        if not frames:
            return
        _, frame = frames[0]
        frames.clear()
        roi = self._play_roi(frame, player)
        gray = cv2.GaussianBlur(cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY), (5, 5), 0)
        self._baseline_by_seat[player] = gray.copy()
        self._previous_by_seat[player] = gray

    def _activate_zone(
        self,
        monotonic_ms: int,
        *,
        accept_initial_occupied: bool = False,
        preserve_baseline: bool = False,
    ) -> None:
        player = self.snapshot.current_player
        if player is None:
            self._zone = None
            self._turn_ownership_window = None
            return
        # Every action window starts from the expected player's current ROI.
        # Old cards and an earlier animation in this same seat must not count
        # as a new action after turn ownership changes.
        if not preserve_baseline:
            self._baseline_by_seat.pop(player, None)
            self._previous_by_seat.pop(player, None)
            self._content_prev_by_seat.pop(player, None)
        spec = strategy_spec(self.recognition_strategy)
        self._zone = ZoneLifecycle(
            expected_player=player,
            activated_at_ms=int(monotonic_ms),
            settle_ms=max(self.settle_ms, spec.settle_ms),
            stable_ms=spec.stable_ms,
            action_timeout_ms=self.action_timeout_ms,
            accept_initial_occupied=accept_initial_occupied,
        )
        self._self_turn_corroborated = False
        self._clear_burst()
        self._ensure_turn_ownership_window()

    def _turn_ownership_key(
        self,
        snapshot: LiveSnapshot | None = None,
    ) -> tuple[str, int, int, Seat] | None:
        current = self.snapshot if snapshot is None else snapshot
        if current.current_player is None:
            return None
        return (
            current.session_id,
            current.turn_id,
            current.revision,
            current.current_player,
        )

    def _ensure_turn_ownership_window(self) -> _TurnOwnershipWindow | None:
        snapshot = self.snapshot
        key = self._turn_ownership_key(snapshot)
        if key is None:
            self._turn_ownership_window = None
            return None
        window = self._turn_ownership_window
        if window is None or window.key != key:
            window = _TurnOwnershipWindow(key=key, expected_player=key[-1])
            self._turn_ownership_window = window
        return window

    def _is_self_lead_handoff(
        self,
        expected: Seat,
        active: Seat | None,
    ) -> bool:
        snapshot = self.snapshot
        return bool(
            self._first_action_pending
            and expected == "self"
            and snapshot.lead_player == "self"
            and active in TURN_ORDER
            and active != "self"
        )

    def _is_direct_next_active(
        self,
        expected: Seat,
        active: Seat | None,
    ) -> bool:
        if active not in TURN_ORDER:
            return False
        snapshot = self.snapshot
        return active == next_active_seat(expected, snapshot.finished_seats)

    def _begin_owner_handoff(
        self,
        window: _TurnOwnershipWindow,
        active: Seat,
        monotonic_ms: int,
    ) -> None:
        self._clear_unseen_direct_next_pass(window)
        self._clear_crossed_handoff_recovery(window)
        self._clear_wind_catch_pass_recovery(window)
        window.handoff_active_player = active
        window.handoff_active_streak = 1
        window.handoff_detected_ms = int(monotonic_ms)
        window.handoff_global_deadline_ms = self._handoff_global_deadline_ms(
            monotonic_ms
        )
        window.handoff_readable_since_ms = None
        window.handoff_local_deadline_ms = None
        window.handoff_unreadable_since_ms = None
        window.handoff_sample_count = 0
        window.handoff_block_reason = "waiting_for_readable_expected_roi"
        window.handoff_last_block_reason = ""
        window.handoff_deadline_kind = None
        # Do not let an earlier static owner ROI read combine with the first
        # post-turn-timer handoff read.  The readable handoff window itself
        # must provide all strategy evidence required for a commit.
        self._samples.clear()
        self._first_action_samples.clear()
        window.provisional_samples.clear()
        window.handoff_samples.clear()
        self._last_sample_ms = None

    @staticmethod
    def _matching_expected_pass_marker(
        fast: FastSignalResult,
        expected: Seat,
    ) -> bool:
        """Return only a seat-bound pass marker for this expected action."""

        return LiveOrchestrator._matching_pass_marker(fast, expected)

    @staticmethod
    def _matching_pass_marker(
        fast: FastSignalResult,
        player: Seat,
    ) -> bool:
        """Return a pass marker tied to one seat, including recovery scans."""

        return bool(
            player in getattr(fast, "pass_marker_players", ())
            or (
                fast.pass_visible
                and fast.pass_marker_player == player
            )
        )

    @staticmethod
    def _pass_marker_players(fast: FastSignalResult) -> frozenset[Seat]:
        """Normalize all seat-bound PASS markers from one fast observation."""

        players = {
            player
            for player in getattr(fast, "pass_marker_players", ())
            if player in TURN_ORDER
        }
        if fast.pass_visible and fast.pass_marker_player in TURN_ORDER:
            players.add(fast.pass_marker_player)
        return frozenset(players)

    @staticmethod
    def _unseen_direct_next_pass_telemetry(
        window: _TurnOwnershipWindow,
    ) -> dict[str, object]:
        return {
            "pending": window.unseen_direct_next_pass_pending,
            "detected_ms": window.unseen_direct_next_pass_detected_ms,
            "deadline_ms": window.unseen_direct_next_pass_deadline_ms,
            "marker_visible": window.unseen_direct_next_pass_marker_visible,
            "marker_player": window.unseen_direct_next_pass_marker_player,
            "fresh_edge_seen": window.unseen_direct_next_pass_edge_seen,
            "marker_streak": window.unseen_direct_next_pass_marker_streak,
            "edge_while_active_unknown": (
                window.expected_pass_marker_edge_while_active_unknown
            ),
            "last_expected_marker_visible": window.last_expected_pass_marker_visible,
            "last_expected_marker_player": window.last_expected_pass_marker_player,
        }

    def _clear_unseen_direct_next_pass(
        self,
        window: _TurnOwnershipWindow,
    ) -> None:
        window.unseen_direct_next_pass_pending = False
        window.unseen_direct_next_pass_detected_ms = None
        window.unseen_direct_next_pass_deadline_ms = None
        window.unseen_direct_next_pass_marker_visible = False
        window.unseen_direct_next_pass_edge_seen = False
        window.unseen_direct_next_pass_marker_streak = 0
        window.unseen_direct_next_pass_marker_player = None

    @staticmethod
    def _wind_catch_pass_recovery_telemetry(
        window: _TurnOwnershipWindow,
    ) -> dict[str, object]:
        return {
            "pending": window.wind_catch_pass_recovery_pending,
            "receiver": window.wind_catch_pass_recovery_receiver,
            "detected_ms": window.wind_catch_pass_recovery_detected_ms,
            "deadline_ms": window.wind_catch_pass_recovery_deadline_ms,
            "marker_streak": window.wind_catch_pass_recovery_marker_streak,
            "receiver_play_streak": window.wind_catch_receiver_play_streak,
            "receiver_play": (
                list(window.wind_catch_receiver_play.cards)
                if window.wind_catch_receiver_play is not None
                else []
            ),
        }

    @staticmethod
    def _clear_wind_catch_pass_recovery(
        window: _TurnOwnershipWindow,
    ) -> None:
        window.wind_catch_pass_recovery_pending = False
        window.wind_catch_pass_recovery_receiver = None
        window.wind_catch_pass_recovery_detected_ms = None
        window.wind_catch_pass_recovery_deadline_ms = None
        window.wind_catch_pass_recovery_marker_streak = 0
        window.wind_catch_receiver_play = None
        window.wind_catch_receiver_play_streak = 0

    @staticmethod
    def _crossed_handoff_recovery_telemetry(
        window: _TurnOwnershipWindow,
    ) -> dict[str, object]:
        return {
            "pending": window.crossed_handoff_recovery_pending,
            "detected_ms": window.crossed_handoff_recovery_detected_ms,
            "active_player": window.crossed_handoff_recovery_active_player,
            "pass_player": window.crossed_handoff_recovery_pass_player,
            "pass_marker_streak": window.crossed_handoff_recovery_pass_marker_streak,
        }

    @staticmethod
    def _clear_crossed_handoff_recovery(
        window: _TurnOwnershipWindow,
    ) -> None:
        window.crossed_handoff_recovery_pending = False
        window.crossed_handoff_recovery_active_player = None
        window.crossed_handoff_recovery_pass_player = None
        window.crossed_handoff_recovery_detected_ms = None
        window.crossed_handoff_recovery_pass_marker_streak = 0

    @staticmethod
    def _turn_recovery_telemetry(
        window: _TurnOwnershipWindow,
    ) -> dict[str, object]:
        return {
            "pending": window.turn_recovery_pending,
            "detected_ms": window.turn_recovery_detected_ms,
            "active_player": window.turn_recovery_active_player,
            "active_streak": window.turn_recovery_active_streak,
            "owner_return_streak": window.turn_recovery_owner_return_streak,
            "pass_marker_streaks": dict(window.turn_recovery_pass_marker_streaks),
            "advice_withheld": window.turn_recovery_advice_withheld,
            "advice_event_id": window.turn_recovery_advice_event_id,
            "local_started_ms": window.turn_recovery_local_started_ms,
            "local_deadline_ms": window.turn_recovery_local_deadline_ms,
            "target_exceeded": window.turn_recovery_target_exceeded,
        }

    def _begin_turn_recovery(
        self,
        window: _TurnOwnershipWindow,
        fast: FastSignalResult,
        monotonic_ms: int,
        *,
        reason: str,
    ) -> None:
        """Keep rereading one missed action until its owner acts again."""

        if window.turn_recovery_pending:
            return
        self._clear_crossed_handoff_recovery(window)
        self._clear_unseen_direct_next_pass(window)
        window.turn_recovery_pending = True
        window.turn_recovery_detected_ms = int(monotonic_ms)
        window.turn_recovery_active_player = fast.active_player
        window.turn_recovery_active_streak = 0
        window.turn_recovery_owner_return_streak = 0
        window.turn_recovery_pass_marker_streaks.clear()
        window.turn_recovery_advice_withheld = False
        window.turn_recovery_advice_event_id = None
        window.turn_recovery_local_started_ms = None
        window.turn_recovery_local_deadline_ms = None
        window.turn_recovery_target_exceeded = False
        window.disposition = "turn_recovery_pending"
        window.handoff_block_reason = reason
        window.handoff_last_block_reason = reason
        window.handoff_deadline_kind = None

    def _turn_recovery_intervening_passes(
        self,
        window: _TurnOwnershipWindow,
        active: Seat | None,
    ) -> tuple[Seat, ...] | None:
        """List the players that must have PASSed to reach ``active``."""

        if active not in TURN_ORDER or active == window.expected_player:
            return None
        snapshot = self.snapshot
        try:
            start = next_active_seat(
                window.expected_player,
                snapshot.finished_seats,
            )
        except GameStateError:
            return None
        return self._intervening_passes_from(
            start,
            active,
            snapshot.finished_seats,
        )

    @staticmethod
    def _intervening_passes_from(
        start: Seat | None,
        active: Seat | None,
        finished_seats: frozenset[Seat],
    ) -> tuple[Seat, ...] | None:
        if start not in TURN_ORDER or active not in TURN_ORDER:
            return None
        passes: list[Seat] = []
        player = start
        for _ in range(len(TURN_ORDER)):
            if player == active:
                return tuple(passes)
            passes.append(player)
            try:
                player = next_active_seat(player, finished_seats)
            except GameStateError:
                return None
        return None

    def _preview_turn_recovery_snapshot(
        self,
        result: ConsensusResult,
    ) -> LiveSnapshot | None:
        """Preview the expected action without mutating the live reducer."""

        preview = self._turn_recovery_preview_reducer(result)
        return preview.snapshot() if preview is not None else None

    def _turn_recovery_preview_reducer(
        self,
        result: ConsensusResult,
    ) -> LiveReducer | None:
        """Build a reducer containing the delayed action but no live mutation."""

        window = self._ensure_turn_ownership_window()
        if window is None:
            return None
        preview = self.reducer.clone_empty()
        try:
            for event in self.reducer.events:
                preview.apply(event)
            commit_cards = result.resolved_cards if not result.is_pass else result.cards
            if result.is_pass:
                preview.record_pass(
                    window.expected_player,
                    confidence=result.confidence,
                    source="turn_recovery_preview",
                )
            else:
                preview.record_play(
                    window.expected_player,
                    commit_cards,
                    confidence=result.confidence,
                    source="turn_recovery_preview",
                    suit_options=(
                        () if commit_cards != result.cards else result.suit_options
                    ),
                )
        except GameStateError:
            return None
        return preview

    def _turn_recovery_passes_after_action(
        self,
        result: ConsensusResult,
        active: Seat | None,
    ) -> tuple[Seat, ...] | None:
        preview = self._preview_turn_recovery_snapshot(result)
        if preview is None or preview.current_player is None:
            return None
        return self._intervening_passes_from(
            preview.current_player,
            active,
            preview.finished_seats,
        )

    def _withhold_advice_for_turn_recovery(
        self,
        window: _TurnOwnershipWindow,
        monotonic_ms: int,
    ) -> None:
        """Defer only the current local recommendation while history catches up."""

        now = int(monotonic_ms)
        if window.turn_recovery_local_started_ms is None:
            window.turn_recovery_local_started_ms = now
            window.turn_recovery_local_deadline_ms = now + _ADVICE_RECOVERY_TARGET_MS

        snapshot = self.snapshot
        key = AdviceRequestKey(
            snapshot.session_id,
            snapshot.turn_id,
            snapshot.revision,
        )
        if not window.turn_recovery_advice_withheld:
            window.turn_recovery_advice_withheld = True
            self.latest_advice = LiveAdvice(
                key=key,
                status="withheld",
                error=_TURN_RECOVERY_WITHHOLD_TEXT,
                withhold_reason=_TURN_RECOVERY_WITHHOLD_REASON,
            )
            self.store.append_advice(
                {
                    "request_id": key.request_id,
                    "status": "withheld",
                    "reason": _TURN_RECOVERY_WITHHOLD_REASON,
                    "expected_player": window.expected_player,
                    "active_player": window.active_player,
                    "turn_id": key.turn_id,
                    "state_revision": key.state_revision,
                    "recovery_target_ms": _ADVICE_RECOVERY_TARGET_MS,
                    **self._advisor_identity(),
                }
            )
            event = self._append_advice_event(
                "advice_withheld",
                {
                    "request_id": key.request_id,
                    "reason": _TURN_RECOVERY_WITHHOLD_REASON,
                    "expected_player": window.expected_player,
                    "active_player": window.active_player,
                    "state_revision": key.state_revision,
                    "recovery_target_ms": _ADVICE_RECOVERY_TARGET_MS,
                },
                confidence=0.0,
            )
            window.turn_recovery_advice_event_id = event.event_id

        deadline_ms = window.turn_recovery_local_deadline_ms
        if (
            deadline_ms is not None
            and now >= deadline_ms
            and not window.turn_recovery_target_exceeded
        ):
            # This is audit/UI information only.  Do not set the global
            # desynchronization latch: visual repair must continue, and a
            # later complete revision can still safely request FableDan.
            window.turn_recovery_target_exceeded = True
            self.store.append_advice(
                {
                    "request_id": key.request_id,
                    "status": "withheld",
                    "reason": _TURN_RECOVERY_WITHHOLD_REASON,
                    "outcome": "recovery_target_exceeded",
                    "elapsed_ms": now - window.turn_recovery_local_started_ms,
                    "target_ms": _ADVICE_RECOVERY_TARGET_MS,
                    "turn_id": key.turn_id,
                    "state_revision": key.state_revision,
                    **self._advisor_identity(),
                }
            )
            self._append_advice_event(
                "advice_recovery_target_exceeded",
                {
                    "request_id": key.request_id,
                    "expected_player": window.expected_player,
                    "active_player": window.active_player,
                    "elapsed_ms": now - window.turn_recovery_local_started_ms,
                    "target_ms": _ADVICE_RECOVERY_TARGET_MS,
                    "state_revision": key.state_revision,
                },
                confidence=0.0,
            )

    def _advance_turn_recovery(
        self,
        window: _TurnOwnershipWindow,
        fast: FastSignalResult,
        monotonic_ms: int,
        *,
        metrics: ZoneFrameMetrics,
        decision: ZoneDecision,
        frame: np.ndarray | None = None,
    ) -> _OwnershipResolution | None:
        """Collect delayed evidence until the missed owner's next action."""

        active = fast.active_player
        window.active_player = active
        window.sample_allowed = False
        if active == window.expected_player:
            window.turn_recovery_owner_return_streak += 1
            window.disposition = "turn_recovery_owner_returned"
            window.handoff_block_reason = "expected_player_acted_again_before_recovery"
            window.handoff_last_block_reason = window.handoff_block_reason
            window.handoff_deadline_kind = "owner_next_turn"
            if window.turn_recovery_owner_return_streak < 2:
                return None
            # Local controls mean every intervening seat still exposes its
            # latest card/PASS surface.  Reconstruct that cycle before
            # declaring a gap; no timer-skin classification chooses actions.
            if frame is not None:
                recovered = self._recover_turn_recovery_cycle(
                    window,
                    frame,
                    fast,
                    monotonic_ms,
                    metrics=metrics,
                )
                if recovered is not None:
                    event, events = recovered
                    return _OwnershipResolution(event, events)
            if (
                self.snapshot.current_player == "self"
                or fast.self_action_buttons_visible
            ):
                self._withhold_advice_for_turn_recovery(window, monotonic_ms)
            # Evidence may remain undecided, but it must not globally latch
            # the session while visual repair is still possible.
            return None

        window.turn_recovery_owner_return_streak = 0
        if active == window.turn_recovery_active_player:
            window.turn_recovery_active_streak += 1
        else:
            window.turn_recovery_active_player = active
            window.turn_recovery_active_streak = 1

        required_passes = self._turn_recovery_intervening_passes(window, active)
        readable = bool(
            decision.collect_sample
            and decision.phase == ZonePhase.BURST_READ
            and not fast.effect_visible
            and not metrics.effect_visible
        )
        if required_passes is not None and readable:
            for player in required_passes:
                if self._matching_pass_marker(fast, player):
                    window.turn_recovery_pass_marker_streaks[player] = (
                        window.turn_recovery_pass_marker_streaks.get(player, 0) + 1
                    )
                else:
                    window.turn_recovery_pass_marker_streaks[player] = 0

        if (
            self.snapshot.current_player == "self"
            or active == "self"
            or fast.self_action_buttons_visible
        ):
            self._withhold_advice_for_turn_recovery(window, monotonic_ms)

        window.sample_allowed = readable
        window.disposition = (
            "turn_recovery"
            if readable
            else "turn_recovery_waiting_readable"
        )
        window.handoff_block_reason = (
            "" if readable else self._handoff_block_reason(metrics, decision, fast)
        )
        window.handoff_last_block_reason = window.handoff_block_reason
        return None

    def _turn_recovery_is_ready(
        self,
        window: _TurnOwnershipWindow | None,
        result: ConsensusResult,
        fast: FastSignalResult,
    ) -> bool:
        if (
            window is None
            or not window.turn_recovery_pending
            or result.status != "confirmed"
            or window.turn_recovery_active_streak < 2
        ):
            return False
        required_passes = self._turn_recovery_passes_after_action(
            result,
            fast.active_player,
        )
        return bool(
            required_passes is not None
            and all(
                window.turn_recovery_pass_marker_streaks.get(player, 0) >= 2
                for player in required_passes
            )
        )

    @staticmethod
    def _confirmed_handoff_precedes_turn_recovery(
        window: _TurnOwnershipWindow | None,
        result: ConsensusResult,
    ) -> bool:
        """Keep a confirmed expected-ROI play across a fast active-seat jump."""

        if (
            window is None
            or not window.turn_recovery_pending
            or window.handoff_detected_ms is None
            or result.status != "confirmed"
            or result.is_pass
            or result.vote_count < 2
            or len(window.handoff_samples) < 2
        ):
            return False
        expected_cards, _options = canonical_candidate(
            result.cards,
            result.suit_options,
            is_pass=False,
        )
        for sample in window.handoff_samples[-2:]:
            cards, _sample_options = canonical_candidate(
                sample.cards,
                sample.suit_options,
                is_pass=sample.is_pass,
            )
            if sample.is_pass or cards != expected_cards:
                return False
        return True

    def _turn_recovery_cycle_candidate(
        self,
        preview: LiveReducer,
        observation: PlayRegionResult,
        *,
        metrics: ZoneFrameMetrics,
        fast: FastSignalResult,
    ) -> ConsensusResult | None:
        """Validate one static seat surface against the previewed next turn."""

        snapshot = preview.snapshot()
        player = snapshot.current_player
        if player is None or observation.player != player:
            return None
        if observation.confidence < 0.80:
            return None
        context = self._consensus_context_for_snapshot(snapshot, metrics, fast)
        cards = tuple(str(card) for card in observation.cards)
        suit_options = tuple(
            tuple(str(suit) for suit in options)
            for options in observation.suit_options
        )
        rejected = BurstConsensus.validate_candidate(
            bool(observation.is_pass),
            cards,
            context,
            suit_options=suit_options,
        )
        if rejected:
            return None
        resolved_cards = BurstConsensus.resolve_commit_cards(
            bool(observation.is_pass),
            cards,
            context,
            suit_options=suit_options,
        )
        return ConsensusResult(
            status="confirmed",
            cards=cards,
            is_pass=bool(observation.is_pass),
            confidence=float(observation.confidence),
            source=(
                "turn_recovery_cycle_pass_marker"
                if observation.is_pass
                else "turn_recovery_cycle_play_region"
            ),
            vote_count=1,
            candidates=(),
            resolved_cards=resolved_cards,
            suit_options=suit_options,
        )

    @staticmethod
    def _apply_preview_consensus(
        preview: LiveReducer,
        result: ConsensusResult,
    ) -> None:
        player = preview.snapshot().current_player
        if player is None:
            raise GameStateError("回合容错预览已结束")
        cards = result.resolved_cards if not result.is_pass else result.cards
        if result.is_pass:
            preview.record_pass(
                player,
                confidence=result.confidence,
                source=result.source,
            )
            return
        preview.record_play(
            player,
            cards,
            confidence=result.confidence,
            source=result.source,
            suit_options=(() if cards != result.cards else result.suit_options),
        )

    def _recover_turn_recovery_cycle(
        self,
        window: _TurnOwnershipWindow,
        frame: np.ndarray,
        fast: FastSignalResult,
        monotonic_ms: int,
        *,
        metrics: ZoneFrameMetrics,
    ) -> tuple[LiveEvent, tuple[LiveEvent, ...]] | None:
        """Repair a complete visible response cycle before local advice."""

        if (
            window.expected_player != "self"
            or not fast.self_action_buttons_visible
            or window.turn_recovery_detected_ms is None
            or int(monotonic_ms) - window.turn_recovery_detected_ms < 500
        ):
            return None
        expected_action = self._decide_timeout_candidate(metrics, fast)
        if expected_action is None or expected_action.status != "confirmed":
            return None
        preview = self._turn_recovery_preview_reducer(expected_action)
        if preview is None:
            return None

        recovered: list[ConsensusResult] = []
        try:
            for _ in range(len(TURN_ORDER) - 1):
                player = preview.snapshot().current_player
                if player == window.expected_player:
                    break
                if player is None:
                    return None
                observation = self._recognize_play_region(
                    frame,
                    player,
                    wild_rank=preview.snapshot().wild_rank,
                    allow_pass=True,
                )
                candidate = self._turn_recovery_cycle_candidate(
                    preview,
                    observation,
                    metrics=metrics,
                    fast=fast,
                )
                if candidate is None:
                    return None
                self._apply_preview_consensus(preview, candidate)
                recovered.append(candidate)
        except (cv2.error, GameStateError, OSError, RuntimeError, ValueError):
            return None

        if preview.snapshot().current_player != window.expected_player:
            return None

        first_event, first_events = self._commit_consensus(
            expected_action,
            monotonic_ms,
            fast=fast,
            suppress_turn_side_effects=True,
        )
        events: list[LiveEvent] = [*first_events]
        last_event = first_event
        for candidate in recovered:
            event, action_events = self._commit_consensus(
                candidate,
                monotonic_ms,
                fast=fast,
                suppress_turn_side_effects=True,
            )
            events.extend(action_events)
            last_event = event

        if self.snapshot.current_player != window.expected_player:
            raise RuntimeError("整轮容错恢复后的行动座位不匹配")
        self._activate_zone(monotonic_ms)
        recovery_event = self._append_lifecycle_event(
            "turn_recovery_cycle_recovered",
            {
                "action_event_id": first_event.event_id,
                "action_player": first_event.actor,
                "recovered_actions": [
                    {
                        "player": event.actor,
                        "cards": list(event.payload.get("cards", ())),
                        "is_pass": bool(event.payload.get("is_pass", False)),
                        "source": event.source,
                    }
                    for event in events
                    if event.event_type in {"player_played", "player_passed"}
                    and event.event_id != first_event.event_id
                ],
                "action_vote_count": expected_action.vote_count,
                "recovery_surface": "local_controls_visible",
            },
            actor=first_event.actor,
            confidence=expected_action.confidence,
            source="turn_recovery_cycle",
        )
        self._append_recognition_trace(
            window=window,
            outcome="turn_recovery_cycle_recovered",
            strategy_result=expected_action,
            fallback=True,
            reason="local_controls_visible_cycle_scan",
            commit_attempted=True,
            commit_event_id=last_event.event_id,
        )
        turn_started = self._append_current_turn_started()
        self._request_advice_if_needed()
        events.extend((recovery_event, *((turn_started,) if turn_started else ())))
        return last_event, tuple(events)

    def _restart_turn_recovery_zone(self, monotonic_ms: int) -> None:
        """Refresh UI timing without discarding delayed action evidence."""

        window = self._ensure_turn_ownership_window()
        if window is None or not window.turn_recovery_pending:
            return
        spec = strategy_spec(self.recognition_strategy)
        self._zone = ZoneLifecycle(
            expected_player=window.expected_player,
            activated_at_ms=int(monotonic_ms),
            settle_ms=max(self.settle_ms, spec.settle_ms),
            stable_ms=spec.stable_ms,
            action_timeout_ms=self.action_timeout_ms,
            accept_initial_occupied=True,
        )
        self._last_sample_ms = None

    def _begin_unseen_direct_next_pass(
        self,
        window: _TurnOwnershipWindow,
        fast: FastSignalResult,
        monotonic_ms: int,
        *,
        marker_baseline_visible: bool | None = None,
    ) -> None:
        """Open the one safe unauthenticated recovery: a new pass marker.

        A just-created turn can miss the outgoing player's timer entirely.  It
        is never safe to infer their cards from that condition, but a new
        seat-bound PASS marker can be confirmed independently.  Capture the
        first marker state as a baseline so an old, already-visible marker is
        not mistaken for the current action.
        """

        # The marker can be painted one frame after the direct successor's
        # timer.  If that first frame opened an empty normal handoff, discard
        # it before changing to the pass-only path; it contains no cards and
        # therefore cannot justify treating the delayed marker as a play.
        self._clear_owner_handoff(window)
        window.unseen_direct_next_pass_pending = True
        window.unseen_direct_next_pass_detected_ms = int(monotonic_ms)
        window.unseen_direct_next_pass_deadline_ms = self._handoff_global_deadline_ms(
            monotonic_ms
        )
        window.unseen_direct_next_pass_marker_visible = bool(
            self._matching_expected_pass_marker(fast, window.expected_player)
            if marker_baseline_visible is None
            else marker_baseline_visible
        )
        window.unseen_direct_next_pass_marker_player = fast.pass_marker_player
        # The edge is consumed as the baseline for this one recovery window;
        # it must not leak into a later action of the same owner.
        window.expected_pass_marker_edge_while_active_unknown = False
        window.disposition = "unseen_direct_next_pass_pending"
        window.sample_allowed = False
        # An unauthenticated handoff is never card evidence.  Keep any old ROI
        # pixels diagnostic-only and prevent them from reaching consensus.
        self._samples.clear()
        self._first_action_samples.clear()
        window.provisional_samples.clear()
        window.handoff_samples.clear()
        self._last_sample_ms = None

    def _clear_owner_handoff(self, window: _TurnOwnershipWindow) -> None:
        window.handoff_active_player = None
        window.handoff_active_streak = 0
        window.handoff_detected_ms = None
        window.handoff_global_deadline_ms = None
        window.handoff_readable_since_ms = None
        window.handoff_local_deadline_ms = None
        window.handoff_unreadable_since_ms = None
        window.handoff_sample_count = 0
        window.handoff_block_reason = ""
        window.handoff_last_block_reason = ""
        window.handoff_deadline_kind = None
        # A later owner-active read invalidates the previous next-seat
        # handoff.  Do not let its evidence be replayed or attributed to a
        # normal owner commit.
        window.handoff_samples.clear()
        self._clear_unseen_direct_next_pass(window)
        self._clear_crossed_handoff_recovery(window)
        self._clear_wind_catch_pass_recovery(window)
        window.expected_pass_marker_edge_while_active_unknown = False

    def _can_start_crossed_handoff_recovery(
        self,
        window: _TurnOwnershipWindow,
        fast: FastSignalResult,
    ) -> bool:
        """Allow only a visible play followed by its direct-next PASS.

        The expected action must already have one readable, non-PASS sample.
        A timer jump alone is never enough to manufacture the missing PASS.
        """

        snapshot = self.snapshot
        expected = window.expected_player
        if not window.handoff_samples or not any(
            not sample.is_pass and sample.cards for sample in window.handoff_samples
        ):
            return False
        try:
            pass_player = next_active_seat(expected, snapshot.finished_seats)
            recovery_active = next_active_seat(
                pass_player,
                snapshot.finished_seats,
            )
        except GameStateError:
            return False
        return bool(
            fast.active_player == recovery_active
            and self._matching_expected_pass_marker(fast, pass_player)
        )

    def _begin_crossed_handoff_recovery(
        self,
        window: _TurnOwnershipWindow,
        fast: FastSignalResult,
        monotonic_ms: int,
    ) -> None:
        """Retain the expected play while its direct follower PASSes quickly."""

        snapshot = self.snapshot
        pass_player = next_active_seat(
            window.expected_player,
            snapshot.finished_seats,
        )
        self._clear_crossed_handoff_recovery(window)
        window.crossed_handoff_recovery_pending = True
        window.crossed_handoff_recovery_active_player = fast.active_player
        window.crossed_handoff_recovery_pass_player = pass_player
        window.crossed_handoff_recovery_detected_ms = int(monotonic_ms)
        window.crossing_active_player = fast.active_player
        window.crossing_active_streak = 0
        window.crossing_non_owner_streak = 0
        window.disposition = "crossed_handoff_recovery"
        window.handoff_block_reason = "waiting_for_crossed_handoff_confirmation"
        window.handoff_last_block_reason = ""
        window.handoff_deadline_kind = None

    def _advance_crossed_handoff_recovery(
        self,
        window: _TurnOwnershipWindow,
        fast: FastSignalResult,
        monotonic_ms: int,
        *,
        metrics: ZoneFrameMetrics,
        decision: ZoneDecision,
    ) -> _OwnershipResolution | None:
        """Confirm a skipped direct-next PASS without relaxing normal votes."""

        now = int(monotonic_ms)
        global_deadline_ms = window.handoff_global_deadline_ms
        if global_deadline_ms is not None and now >= global_deadline_ms:
            window.disposition = "crossed_handoff_recovery_deadline_expired"
            window.handoff_block_reason = "crossed_handoff_recovery_deadline"
            window.handoff_last_block_reason = window.handoff_block_reason
            window.handoff_deadline_kind = "global"
            self._begin_turn_recovery(
                window,
                fast,
                monotonic_ms,
                reason=window.handoff_block_reason,
            )
            return self._advance_turn_recovery(
                window,
                fast,
                monotonic_ms,
                metrics=metrics,
                decision=decision,
            )

        if fast.active_player != window.crossed_handoff_recovery_active_player:
            window.disposition = "crossed_handoff_recovery_active_changed"
            window.handoff_block_reason = "crossed_handoff_recovery_active_changed"
            window.handoff_last_block_reason = window.handoff_block_reason
            window.handoff_deadline_kind = "seat_cross"
            self._begin_turn_recovery(
                window,
                fast,
                monotonic_ms,
                reason=window.handoff_block_reason,
            )
            return self._advance_turn_recovery(
                window,
                fast,
                monotonic_ms,
                metrics=metrics,
                decision=decision,
            )

        window.crossing_active_streak += 1
        window.crossing_non_owner_streak += 1
        readable = bool(
            decision.collect_sample
            and decision.phase == ZonePhase.BURST_READ
            and not fast.effect_visible
            and not metrics.effect_visible
        )
        if not readable:
            window.crossed_handoff_recovery_pass_marker_streak = 0
            window.sample_allowed = False
            window.disposition = "crossed_handoff_recovery_waiting_readable"
            window.handoff_block_reason = self._handoff_block_reason(
                metrics,
                decision,
                fast,
            )
            window.handoff_last_block_reason = window.handoff_block_reason
            return None

        pass_player = window.crossed_handoff_recovery_pass_player
        pass_marker_visible = bool(
            pass_player is not None
            and self._matching_expected_pass_marker(fast, pass_player)
        )
        if pass_marker_visible:
            window.crossed_handoff_recovery_pass_marker_streak += 1
        else:
            window.crossed_handoff_recovery_pass_marker_streak = 0
        window.sample_allowed = True
        window.disposition = "crossed_handoff_recovery"
        window.handoff_block_reason = ""
        return None

    def _crossed_handoff_recovery_is_ready(
        self,
        window: _TurnOwnershipWindow | None,
        result: ConsensusResult,
    ) -> bool:
        return bool(
            window is not None
            and window.crossed_handoff_recovery_pending
            and result.status == "confirmed"
            and not result.is_pass
            and bool(result.cards)
            and window.crossing_active_streak >= 2
            and window.crossed_handoff_recovery_pass_marker_streak >= 2
        )

    def _handoff_global_deadline_ms(self, monotonic_ms: int) -> int:
        """Bound recovery by both the handoff policy and this action window."""

        now = int(monotonic_ms)
        zone = self._zone
        if zone is None:
            return now
        zone_deadline_ms = zone.activated_at_ms + zone.action_timeout_ms
        return min(now + _HANDOFF_GLOBAL_GRACE_MS, zone_deadline_ms)

    def _ownership_allows_sample(self) -> bool:
        window = self._ensure_turn_ownership_window()
        return bool(window is not None and window.sample_allowed)

    @staticmethod
    def _handoff_block_reason(
        metrics: ZoneFrameMetrics,
        decision: ZoneDecision,
        fast: FastSignalResult,
    ) -> str:
        if fast.effect_visible or metrics.effect_visible:
            return "effect_settling"
        if decision.phase != ZonePhase.BURST_READ:
            return f"zone_{decision.phase.value}"
        if not decision.collect_sample:
            return decision.reason or "zone_not_collecting"
        return "expected_roi_not_readable"

    @staticmethod
    def _handoff_telemetry(window: _TurnOwnershipWindow) -> dict[str, object]:
        return {
            "detected_ms": window.handoff_detected_ms,
            "global_deadline_ms": window.handoff_global_deadline_ms,
            "readable_since_ms": window.handoff_readable_since_ms,
            "local_deadline_ms": window.handoff_local_deadline_ms,
            "sample_count": window.handoff_sample_count,
            "block_reason": window.handoff_block_reason,
            "last_block_reason": window.handoff_last_block_reason,
            "deadline_kind": window.handoff_deadline_kind,
        }

    @staticmethod
    def _zone_telemetry(
        metrics: ZoneFrameMetrics | None,
        decision: ZoneDecision | None,
        fast: FastSignalResult,
    ) -> dict[str, object]:
        return {
            "phase": decision.phase.value if decision is not None else None,
            "reason": decision.reason if decision is not None else "",
            "collect_sample": decision.collect_sample if decision is not None else False,
            "effect_visible": bool(
                fast.effect_visible or (metrics.effect_visible if metrics else False)
            ),
            "motion_score": metrics.motion_score if metrics is not None else None,
        }

    @staticmethod
    def _runtime_identity() -> dict[str, object]:
        try:
            fingerprint = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
        except OSError:
            fingerprint = "unavailable"
        return {
            "implementation_fingerprint": fingerprint,
            "executable_path": sys.executable,
        }

    @staticmethod
    def _strategy_trace(result: ConsensusResult | None) -> dict[str, object]:
        if result is None:
            return {"status": "none"}
        return {
            "status": result.status,
            "cards": list(result.cards),
            "is_pass": result.is_pass,
            "vote_count": result.vote_count,
            "rejected_reasons": list(result.rejected_reasons),
        }

    def _append_recognition_trace(
        self,
        *,
        window: _TurnOwnershipWindow | None,
        outcome: str,
        strategy_result: ConsensusResult | None = None,
        fallback: bool = False,
        deadline_kind: str | None = None,
        reason: str = "",
        commit_attempted: bool = False,
        commit_event_id: str = "",
    ) -> None:
        before_key = self._recognition_trace_pre_job_key or self._analysis_job_key()
        refs = (
            [sample.evidence_ref for sample in window.handoff_samples if sample.evidence_ref]
            if window is not None
            else []
        )
        self.store.append_recognition_trace(
            {
                **self._runtime_identity(),
                **self._recognition_trace_context,
                "monotonic_ms": self._last_monotonic_ms,
                "outcome": outcome,
                "reason": reason,
                "fallback": bool(fallback),
                "deadline_kind": deadline_kind,
                "pre_job_key": list(before_key),
                "post_job_key": list(self._analysis_job_key()),
                "observation_refs": refs,
                "strategy": self._strategy_trace(strategy_result),
                "commit_attempted": bool(commit_attempted or commit_event_id),
                "commit_event_id": commit_event_id,
                "handoff": self._handoff_telemetry(window) if window else {},
                "unseen_direct_next_pass": (
                    self._unseen_direct_next_pass_telemetry(window)
                    if window is not None
                    else {}
                ),
                "wind_catch_pass_recovery": (
                    self._wind_catch_pass_recovery_telemetry(window)
                    if window is not None
                    else {}
                ),
                "turn_recovery": (
                    self._turn_recovery_telemetry(window)
                    if window is not None
                    else {}
                ),
            }
        )

    def _can_start_unseen_direct_next_pass(
        self,
        window: _TurnOwnershipWindow,
        expected: Seat,
        active: Seat | None,
    ) -> bool:
        """Allow a fresh direct-next PASS on any new non-lead turn.

        Seats are rotationally symmetric.  The only invariant is that the
        expected player is not leading a new trick and the active timer has
        advanced to exactly its next active seat.  Freshness is checked from
        the immediately preceding fast frame below.
        """

        has_empty_direct_handoff = bool(
            window.handoff_detected_ms is not None
            and window.handoff_sample_count == 0
            and not window.handoff_samples
            and window.disposition
            in {
                "direct_next_handoff",
                "direct_next_handoff_waiting_readable",
            }
        )
        has_new_turn_surface = bool(
            window.handoff_detected_ms is None
            and (
                window.disposition
                in {"unseen", "owner_provisional", "owner_authenticated"}
                or (
                    window.disposition == "isolated_active_unknown"
                    and window.expected_pass_marker_edge_while_active_unknown
                )
            )
        )
        return bool(
            self._is_direct_next_active(expected, active)
            and window.last_expected_pass_marker_visible
            and not self._is_first_action_turn(expected)
            and bool(self.snapshot.trick_plays)
            # The owner may have stayed active long enough to authenticate
            # before clicking PASS.  A fresh seat-bound marker still wins
            # over generic direct-handoff card recovery; otherwise the
            # successor timer steals this PASS before its second read.
            and window.accepted_sample_count == 0
            and window.crossing_non_owner_streak == 0
            and (has_new_turn_surface or has_empty_direct_handoff)
        )

    def _advance_unseen_direct_next_pass(
        self,
        window: _TurnOwnershipWindow,
        fast: FastSignalResult,
        monotonic_ms: int,
        *,
        metrics: ZoneFrameMetrics,
        decision: ZoneDecision,
        active_is_direct_next: bool,
    ) -> _OwnershipResolution | None:
        """Confirm a fresh, expected-seat PASS marker without card recovery."""

        now = int(monotonic_ms)
        deadline_ms = window.unseen_direct_next_pass_deadline_ms
        if deadline_ms is not None and now >= deadline_ms:
            window.disposition = "unseen_direct_next_pass_deadline_expired"
            window.handoff_block_reason = "unseen_direct_next_pass_deadline"
            window.handoff_last_block_reason = window.handoff_block_reason
            window.handoff_deadline_kind = "unseen_direct_next_pass"
            self._begin_turn_recovery(
                window,
                fast,
                monotonic_ms,
                reason=window.handoff_block_reason,
            )
            return self._advance_turn_recovery(
                window,
                fast,
                monotonic_ms,
                metrics=metrics,
                decision=decision,
            )

        marker_visible = self._matching_expected_pass_marker(
            fast,
            window.expected_player,
        )
        window.unseen_direct_next_pass_marker_player = fast.pass_marker_player
        # Once a fresh direct-next edge opened this narrowly scoped recovery,
        # the timer is allowed to disappear while its PASS badge remains.  A
        # timer-only jump never starts the recovery; this only retains the
        # same expected-seat marker through a transitional ``active=None``
        # frame until two reads can commit it.
        readable = bool(
            (active_is_direct_next or fast.active_player is None)
            and not fast.effect_visible
            and not metrics.effect_visible
        )
        if not readable:
            # A marker observed while the ROI/effect is unsettled is baseline
            # evidence only.  It cannot later satisfy the required fresh edge.
            window.unseen_direct_next_pass_marker_visible = marker_visible
            window.unseen_direct_next_pass_marker_streak = 0
            window.disposition = (
                "unseen_direct_next_pass_pending"
                if active_is_direct_next
                else "unseen_direct_next_pass_active_unknown"
            )
            self._append_recognition_trace(
                window=window,
                outcome="unseen_direct_next_pass_pending",
                deadline_kind="unseen_direct_next_pass",
                reason=(
                    "unreadable_pass_marker"
                    if active_is_direct_next
                    else "active_player_unknown"
                ),
            )
            return None

        fresh_edge = (
            marker_visible
            and not window.unseen_direct_next_pass_marker_visible
        )
        window.unseen_direct_next_pass_marker_visible = marker_visible
        if fresh_edge:
            window.unseen_direct_next_pass_edge_seen = True
            window.unseen_direct_next_pass_marker_streak = 1
        elif marker_visible and window.unseen_direct_next_pass_edge_seen:
            window.unseen_direct_next_pass_marker_streak += 1
        elif not marker_visible:
            window.unseen_direct_next_pass_marker_streak = 0

        if (
            window.unseen_direct_next_pass_edge_seen
            and window.unseen_direct_next_pass_marker_streak >= 2
        ):
            marker_pass = ConsensusResult(
                status="confirmed",
                cards=(),
                is_pass=True,
                confidence=1.0,
                source="unseen_direct_next_pass_marker",
                vote_count=window.unseen_direct_next_pass_marker_streak,
                candidates=(),
            )
            event, events = self._commit_consensus(
                marker_pass,
                monotonic_ms,
                fast=fast,
            )
            self._append_recognition_trace(
                window=window,
                outcome="committed",
                strategy_result=marker_pass,
                fallback=True,
                deadline_kind="unseen_direct_next_pass",
                commit_attempted=True,
                commit_event_id=event.event_id,
            )
            return _OwnershipResolution(event, events)

        window.disposition = "unseen_direct_next_pass_pending"
        self._append_recognition_trace(
            window=window,
            outcome="unseen_direct_next_pass_pending",
            deadline_kind="unseen_direct_next_pass",
            reason=(
                "fresh_pass_marker_edge"
                if fresh_edge
                else "waiting_for_second_pass_marker"
            ),
        )
        return None

    def _can_start_wind_catch_pass_recovery(
        self,
        window: _TurnOwnershipWindow,
        fast: FastSignalResult,
    ) -> bool:
        """Recognise only the UI jump caused by the final PASS before 接风.

        The reducer proves that the expected seat is the one remaining active
        opponent and that its PASS would start a wind-catch trick.  A timer on
        the partner alone is never sufficient: the corresponding, seat-bound
        PASS marker must then stay readable for two frames before state moves.
        """

        expected = window.expected_player
        receiver = self.reducer.wind_receiver_after_current_pass(expected)
        return bool(receiver is not None and fast.active_player == receiver)

    def _begin_wind_catch_pass_recovery(
        self,
        window: _TurnOwnershipWindow,
        fast: FastSignalResult,
        monotonic_ms: int,
    ) -> None:
        receiver = self.reducer.wind_receiver_after_current_pass(
            window.expected_player
        )
        if receiver is None:
            return
        self._clear_wind_catch_pass_recovery(window)
        window.wind_catch_pass_recovery_pending = True
        window.wind_catch_pass_recovery_receiver = receiver
        window.wind_catch_pass_recovery_detected_ms = int(monotonic_ms)
        window.wind_catch_pass_recovery_deadline_ms = self._handoff_global_deadline_ms(
            monotonic_ms
        )
        window.disposition = "wind_catch_pass_recovery"
        window.handoff_block_reason = "waiting_for_wind_catch_pass_marker"
        window.handoff_last_block_reason = ""
        window.handoff_deadline_kind = None
        window.sample_allowed = False
        # The receiver's newly exposed play area belongs to the next trick,
        # not the missing PASS.  It must never become evidence for this turn.
        self._samples.clear()
        self._first_action_samples.clear()
        window.provisional_samples.clear()
        window.handoff_samples.clear()
        self._last_sample_ms = None

    def _advance_wind_catch_pass_recovery(
        self,
        window: _TurnOwnershipWindow,
        fast: FastSignalResult,
        monotonic_ms: int,
        *,
        metrics: ZoneFrameMetrics,
        decision: ZoneDecision,
    ) -> _OwnershipResolution | None:
        """Commit the final opponent PASS only after two readable markers."""

        now = int(monotonic_ms)
        deadline_ms = window.wind_catch_pass_recovery_deadline_ms
        if deadline_ms is not None and now >= deadline_ms:
            window.disposition = "wind_catch_pass_recovery_deadline_expired"
            window.handoff_block_reason = "wind_catch_pass_marker_deadline"
            window.handoff_last_block_reason = window.handoff_block_reason
            window.handoff_deadline_kind = "wind_catch_pass"
            self._begin_turn_recovery(
                window,
                fast,
                monotonic_ms,
                reason=window.handoff_block_reason,
            )
            return self._advance_turn_recovery(
                window,
                fast,
                monotonic_ms,
                metrics=metrics,
                decision=decision,
            )

        receiver = window.wind_catch_pass_recovery_receiver
        still_expected = self.reducer.wind_receiver_after_current_pass(
            window.expected_player
        )
        if receiver is None or fast.active_player != receiver or still_expected != receiver:
            window.disposition = "wind_catch_pass_recovery_active_changed"
            window.handoff_block_reason = "wind_catch_receiver_changed"
            window.handoff_last_block_reason = window.handoff_block_reason
            window.handoff_deadline_kind = "seat_cross"
            self._begin_turn_recovery(
                window,
                fast,
                monotonic_ms,
                reason=window.handoff_block_reason,
            )
            return self._advance_turn_recovery(
                window,
                fast,
                monotonic_ms,
                metrics=metrics,
                decision=decision,
            )

        readable = bool(
            decision.collect_sample
            and decision.phase == ZonePhase.BURST_READ
            and not fast.effect_visible
            and not metrics.effect_visible
        )
        if not readable:
            window.wind_catch_pass_recovery_marker_streak = 0
            window.disposition = "wind_catch_pass_recovery_waiting_readable"
            window.handoff_block_reason = self._handoff_block_reason(
                metrics,
                decision,
                fast,
            )
            window.handoff_last_block_reason = window.handoff_block_reason
            window.sample_allowed = False
            self._append_recognition_trace(
                window=window,
                outcome="wind_catch_pass_recovery_pending",
                deadline_kind="wind_catch_pass",
                reason=window.handoff_block_reason,
            )
            return None

        if self._matching_expected_pass_marker(fast, window.expected_player):
            window.wind_catch_pass_recovery_marker_streak += 1
        else:
            window.wind_catch_pass_recovery_marker_streak = 0
        window.disposition = "wind_catch_pass_recovery"
        window.handoff_block_reason = ""
        window.sample_allowed = False
        if window.wind_catch_pass_recovery_marker_streak >= 2:
            marker_pass = ConsensusResult(
                status="confirmed",
                cards=(),
                is_pass=True,
                confidence=1.0,
                source="wind_catch_pass_marker",
                vote_count=window.wind_catch_pass_recovery_marker_streak,
                candidates=(),
            )
            event, events = self._commit_consensus(
                marker_pass,
                monotonic_ms,
                fast=fast,
            )
            self._append_recognition_trace(
                window=window,
                outcome="committed",
                strategy_result=marker_pass,
                fallback=True,
                deadline_kind="wind_catch_pass",
                commit_attempted=True,
                commit_event_id=event.event_id,
            )
            return _OwnershipResolution(event, events)

        self._append_recognition_trace(
            window=window,
            outcome="wind_catch_pass_recovery_pending",
            deadline_kind="wind_catch_pass",
            reason="waiting_for_second_pass_marker",
        )
        return None

    def _advance_wind_catch_receiver_play_recovery(
        self,
        window: _TurnOwnershipWindow,
        frame: np.ndarray,
        fast: FastSignalResult,
        monotonic_ms: int,
        *,
        metrics: ZoneFrameMetrics,
    ) -> _OwnershipResolution | None:
        """Recover a one-frame wind PASS followed by the receiver's play.

        A pass badge can exist for exactly one decoded frame.  On some table
        skins the active-seat highlight lingers on that passer while the wind
        receiver's next cards are already fully visible.  A highlight is not
        proof of an action, so this path is intentionally available only after
        the preceding frame recorded the expected seat's own PASS edge.  It
        then needs two identical legal cards from the *proven next receiver*
        before atomically applying the PASS and that next play.
        """

        if not window.expected_pass_marker_edge_while_active_unknown:
            return None
        receiver = self.reducer.wind_receiver_after_current_pass(
            window.expected_player
        )
        if receiver is None:
            window.wind_catch_receiver_play = None
            window.wind_catch_receiver_play_streak = 0
            return None
        try:
            observation = self._recognize_play_region(
                frame,
                receiver,
                wild_rank=self.snapshot.wild_rank,
                allow_pass=False,
            )
        except (cv2.error, OSError, RuntimeError, ValueError):
            return None
        if (
            observation.player != receiver
            or observation.is_pass
            or not observation.cards
            or observation.confidence < 0.80
        ):
            window.wind_catch_receiver_play = None
            window.wind_catch_receiver_play_streak = 0
            return None

        marker_pass = ConsensusResult(
            status="confirmed",
            cards=(),
            is_pass=True,
            confidence=1.0,
            source="wind_catch_pass_then_receiver_play",
            vote_count=1,
            candidates=(),
        )
        preview = self._turn_recovery_preview_reducer(marker_pass)
        if preview is None:
            return None
        candidate = self._turn_recovery_cycle_candidate(
            preview,
            observation,
            metrics=metrics,
            fast=fast,
        )
        if candidate is None or candidate.is_pass or not candidate.cards:
            window.wind_catch_receiver_play = None
            window.wind_catch_receiver_play_streak = 0
            return None

        sample = RecognitionSample(
            cards=tuple(candidate.cards),
            is_pass=False,
            confidence=float(candidate.confidence),
            source=str(candidate.source),
            suit_options=tuple(candidate.suit_options),
        )
        previous = window.wind_catch_receiver_play
        if previous is not None and (
            previous.cards,
            previous.suit_options,
        ) == (sample.cards, sample.suit_options):
            window.wind_catch_receiver_play_streak += 1
        else:
            window.wind_catch_receiver_play = sample
            window.wind_catch_receiver_play_streak = 1
        if window.wind_catch_receiver_play_streak < 2:
            window.disposition = "wind_catch_receiver_play_pending"
            window.sample_allowed = False
            return None

        receiver_play = replace(
            candidate,
            confidence=(
                float(previous.confidence) + float(candidate.confidence)
                if previous is not None
                else float(candidate.confidence)
            ) / 2,
            source="wind_catch_pass_then_receiver_play",
            vote_count=window.wind_catch_receiver_play_streak,
        )
        self._append_recognition_trace(
            window=window,
            outcome="wind_catch_pass_then_receiver_play",
            strategy_result=receiver_play,
            fallback=True,
            reason="pass_edge_plus_two_receiver_plays",
            commit_attempted=True,
        )
        pass_event, pass_events = self._commit_consensus(
            marker_pass,
            monotonic_ms,
            fast=fast,
            suppress_turn_side_effects=True,
        )
        play_event, play_events = self._commit_consensus(
            receiver_play,
            monotonic_ms,
            fast=fast,
            suppress_turn_side_effects=True,
        )
        self._clear_wind_catch_pass_recovery(window)
        turn_started = self._append_current_turn_started()
        self._request_advice_if_needed()
        events = [*pass_events, *play_events]
        if turn_started is not None:
            events.append(turn_started)
        return _OwnershipResolution(play_event, tuple(events))

    def _advance_direct_handoff(
        self,
        window: _TurnOwnershipWindow,
        fast: FastSignalResult,
        monotonic_ms: int,
        *,
        metrics: ZoneFrameMetrics,
        decision: ZoneDecision,
        readable: bool,
        block_reason: str,
    ) -> _OwnershipResolution | None:
        """Advance a direct-next recovery without charging unreadable frames."""

        now = int(monotonic_ms)
        global_deadline_ms = window.handoff_global_deadline_ms
        if global_deadline_ms is not None and now >= global_deadline_ms:
            window.disposition = "handoff_global_deadline_expired"
            window.handoff_block_reason = (
                "global_deadline_before_readable"
                if window.handoff_readable_since_ms is None
                else "global_handoff_deadline"
            )
            window.handoff_last_block_reason = window.handoff_block_reason
            window.handoff_deadline_kind = "global"
            self._begin_turn_recovery(
                window,
                fast,
                monotonic_ms,
                reason=window.handoff_block_reason,
            )
            return self._advance_turn_recovery(
                window,
                fast,
                monotonic_ms,
                metrics=metrics,
                decision=decision,
            )
        if not readable:
            if (
                window.handoff_readable_since_ms is not None
                and window.handoff_unreadable_since_ms is None
            ):
                window.handoff_unreadable_since_ms = now
            window.handoff_block_reason = block_reason
            window.handoff_last_block_reason = block_reason
            self._append_recognition_trace(
                window=window,
                outcome="invalidated",
                reason=block_reason,
            )
            return None
        if window.handoff_unreadable_since_ms is not None:
            paused_ms = max(0, now - window.handoff_unreadable_since_ms)
            if window.handoff_local_deadline_ms is not None:
                window.handoff_local_deadline_ms += paused_ms
            window.handoff_unreadable_since_ms = None
        if window.handoff_readable_since_ms is None:
            window.handoff_readable_since_ms = now
            window.handoff_local_deadline_ms = now + _HANDOFF_READABLE_CONFIRMATION_MS
        elif (
            window.handoff_local_deadline_ms is not None
            and now >= window.handoff_local_deadline_ms
        ):
            fallback = decide_recognition_strategy(
                self.recognition_strategy,
                window.handoff_samples,
                context=self._consensus_context(metrics, fast),
            )
            self._append_recognition_trace(
                window=window,
                outcome="local_deadline_fallback",
                strategy_result=fallback,
                fallback=True,
                deadline_kind="local",
                commit_attempted=(
                    fallback is not None and fallback.status == "confirmed"
                ),
            )
            if fallback is not None and fallback.status == "confirmed":
                event, events = self._commit_consensus(
                    fallback,
                    monotonic_ms,
                    fast=fast,
                )
                self._append_recognition_trace(
                    window=window,
                    outcome="committed",
                    strategy_result=fallback,
                    fallback=True,
                    deadline_kind="local",
                    commit_event_id=event.event_id,
                )
                return _OwnershipResolution(event, events)
            window.disposition = "handoff_local_deadline_expired"
            window.handoff_block_reason = "readable_confirmation_deadline"
            window.handoff_last_block_reason = window.handoff_block_reason
            window.handoff_deadline_kind = "local"
            self._begin_turn_recovery(
                window,
                fast,
                monotonic_ms,
                reason=window.handoff_block_reason,
            )
            return self._advance_turn_recovery(
                window,
                fast,
                monotonic_ms,
                metrics=metrics,
                decision=decision,
            )
        window.handoff_block_reason = ""
        return None

    def _append_consensus_sample(
        self,
        sample: RecognitionSample,
        window: _TurnOwnershipWindow,
    ) -> None:
        self._samples.append(sample)
        window.accepted_sample_count += 1
        if window.turn_recovery_pending:
            recovery_limit = max(12, self.burst_sample_limit * 3)
            del self._samples[:-recovery_limit]
        if self._is_first_action_turn(window.expected_player):
            self._first_action_samples.append(sample)
            # This buffer only bridges a short async handoff.  Bound it so a
            # genuinely unresolved opening play cannot bias later retries.
            first_action_limit = max(12, self.burst_sample_limit * 3)
            del self._first_action_samples[:-first_action_limit]

    def _promote_provisional_owner_samples(
        self,
        window: _TurnOwnershipWindow,
    ) -> None:
        for sample in window.provisional_samples:
            self._append_consensus_sample(sample, window)
        window.provisional_samples.clear()

    def _suspend_advice_for_turn_desynchronization(
        self,
        snapshot: LiveSnapshot,
        turn_key: tuple[str, int, int, Seat],
        *,
        event_id: str,
    ) -> None:
        self._advice_suspended_reason = _TURN_DESYNCHRONIZED_REASON
        self._advice_suspended_turn_key = turn_key
        key = AdviceRequestKey(
            snapshot.session_id,
            snapshot.turn_id,
            snapshot.revision,
        )
        self.latest_advice = LiveAdvice(
            key=key,
            status="withheld",
            error=_TURN_DESYNCHRONIZED_TEXT,
            withhold_reason=_TURN_DESYNCHRONIZED_REASON,
        )
        self.store.append_advice(
            {
                "request_id": key.request_id,
                "status": "withheld",
                "reason": _TURN_DESYNCHRONIZED_REASON,
                "turn_desynchronized_event_id": event_id,
                "turn_id": key.turn_id,
                "state_revision": key.state_revision,
                **self._advisor_identity(),
            }
        )

    def _mark_turn_desynchronized(
        self,
        window: _TurnOwnershipWindow,
        fast: FastSignalResult,
        monotonic_ms: int,
        *,
        metrics: ZoneFrameMetrics | None = None,
        decision: ZoneDecision | None = None,
    ) -> LiveEvent | None:
        snapshot = self.snapshot
        turn_key = self._turn_ownership_key(snapshot)
        if turn_key is None or self._desynchronized_turn_key == turn_key:
            return None
        self._desynchronized_turn_key = turn_key
        window.desynchronized = True
        incident_observations = list(self._observations)
        self._analysis_epoch += 1
        self._clear_burst()
        self._zone = None
        self.latest_review = None
        event = self._append_lifecycle_event(
            _TURN_DESYNCHRONIZED_REASON,
            {
                "reason": (
                    window.handoff_block_reason
                    or "active_player_advanced_without_committable_expected_action"
                ),
                "history_gap": True,
                "expected_player": window.expected_player,
                "active_player": fast.active_player,
                "active_streak": window.crossing_non_owner_streak,
                "zone": self._zone_telemetry(metrics, decision, fast),
                "owner_window": {
                    "key": list(window.key),
                    "owner": window.expected_player,
                    "active": window.active_player,
                    "disposition": window.disposition,
                    "authenticated": window.authenticated,
                    "handoff_sample_count": window.handoff_sample_count,
                    "accepted_sample_count": window.accepted_sample_count,
                    "isolated_sample_count": window.isolated_sample_count,
                    "provisional_sample_count": len(window.provisional_samples),
                    "handoff": self._handoff_telemetry(window),
                    "crossed_handoff_recovery": self._crossed_handoff_recovery_telemetry(
                        window
                    ),
                    "turn_recovery": self._turn_recovery_telemetry(window),
                    "unseen_direct_next_pass": self._unseen_direct_next_pass_telemetry(
                        window
                    ),
                    "wind_catch_pass_recovery": self._wind_catch_pass_recovery_telemetry(
                        window
                    ),
                },
            },
            actor=window.expected_player,
            confidence=0.0,
            source="turn_ownership_guard",
        )
        self._append_recognition_trace(
            window=window,
            outcome="desynchronized",
            deadline_kind=window.handoff_deadline_kind,
            reason=window.handoff_block_reason,
            commit_event_id="",
        )
        self._suspend_advice_for_turn_desynchronization(
            snapshot,
            turn_key,
            event_id=event.event_id,
        )
        self._create_incident(
            _TURN_DESYNCHRONIZED_REASON,
            monotonic_ms,
            observations=incident_observations,
        )
        self._notify_update_listener()
        return event

    def _observe_turn_ownership(
        self,
        expected: Seat,
        fast: FastSignalResult,
        monotonic_ms: int,
        *,
        frame: np.ndarray,
        metrics: ZoneFrameMetrics,
        decision: ZoneDecision,
    ) -> _OwnershipResolution | None:
        """Classify evidence after the zone has decided whether the ROI is readable."""

        window = self._ensure_turn_ownership_window()
        if window is None or window.expected_player != expected:
            return None
        active = fast.active_player
        previous_pass_marker_players = self._last_pass_marker_players
        self._last_pass_marker_players = self._pass_marker_players(fast)
        previous_expected_pass_marker_visible = window.last_expected_pass_marker_visible
        expected_pass_marker_visible = self._matching_expected_pass_marker(
            fast,
            expected,
        )
        window.last_expected_pass_marker_visible = expected_pass_marker_visible
        window.last_expected_pass_marker_player = fast.pass_marker_player
        window.active_player = active
        window.sample_allowed = False
        direct_next = self._is_direct_next_active(expected, active)
        if window.turn_recovery_pending:
            return self._advance_turn_recovery(
                window,
                fast,
                monotonic_ms,
                metrics=metrics,
                decision=decision,
                frame=frame,
            )
        if expected != "self" and (
            (active == "self" and not direct_next)
            or fast.self_action_buttons_visible
        ):
            self._begin_turn_recovery(
                window,
                fast,
                monotonic_ms,
                reason="self_turn_before_expected_action_confirmed",
            )
            return self._advance_turn_recovery(
                window,
                fast,
                monotonic_ms,
                metrics=metrics,
                decision=decision,
            )
        # The visual PASS edge can precede a stale expected-player highlight.
        # Probe only the reducer-proven wind receiver before the normal
        # expected-owner branch clears that edge; no generic foreign card is
        # accepted here.
        receiver_play_recovery = self._advance_wind_catch_receiver_play_recovery(
            window,
            frame,
            fast,
            monotonic_ms,
            metrics=metrics,
        )
        if receiver_play_recovery is not None:
            return receiver_play_recovery
        if active == expected:
            if window.wind_catch_receiver_play_streak:
                # Do not mix a stale owner-highlight ROI into the receiver
                # recovery burst.  The next frame either confirms the same
                # receiver cards or clears this narrowly-scoped fallback.
                window.disposition = "wind_catch_receiver_play_pending"
                window.sample_allowed = False
                return None
            if not window.expected_pass_marker_edge_while_active_unknown:
                self._clear_wind_catch_pass_recovery(window)
            window.owner_active_streak += 1
            if not window.expected_pass_marker_edge_while_active_unknown:
                self._clear_owner_handoff(window)
            window.crossing_active_player = None
            window.crossing_active_streak = 0
            window.crossing_non_owner_streak = 0
            if window.owner_active_streak >= 2:
                window.authenticated = True
                window.disposition = "owner_authenticated"
                self._promote_provisional_owner_samples(window)
            else:
                window.disposition = "owner_provisional"
            window.sample_allowed = True
            return None
        window.owner_active_streak = 0
        self_lead_handoff = self._is_self_lead_handoff(expected, active)

        # A finished leader's partner becomes the visible active seat as soon
        # as the final opponent clicks PASS.  Recover that one action before
        # generic foreign-active handling; the reducer has already proved the
        # exact wind transition and this branch still needs two marker frames.
        if window.wind_catch_pass_recovery_pending:
            return self._advance_wind_catch_pass_recovery(
                window,
                fast,
                monotonic_ms,
                metrics=metrics,
                decision=decision,
            )
        if self._can_start_wind_catch_pass_recovery(window, fast):
            self._begin_wind_catch_pass_recovery(window, fast, monotonic_ms)
            return self._advance_wind_catch_pass_recovery(
                window,
                fast,
                monotonic_ms,
                metrics=metrics,
                decision=decision,
            )

        # A pass can be recovered before the new owner is ever observed, but
        # only on this initial direct-next frame.  This branch is intentionally
        # ahead of generic handoff/card logic and leaves sample_allowed false.
        if window.unseen_direct_next_pass_pending:
            if direct_next:
                return self._advance_unseen_direct_next_pass(
                    window,
                    fast,
                    monotonic_ms,
                    metrics=metrics,
                    decision=decision,
                    active_is_direct_next=True,
                )
            if active is None:
                return self._advance_unseen_direct_next_pass(
                    window,
                    fast,
                    monotonic_ms,
                    metrics=metrics,
                    decision=decision,
                    active_is_direct_next=False,
                )
            window.disposition = "unseen_direct_next_pass_active_crossed"
            window.handoff_block_reason = "unseen_direct_next_pass_active_crossed"
            window.handoff_last_block_reason = window.handoff_block_reason
            window.handoff_deadline_kind = "seat_cross"
            self._begin_turn_recovery(
                window,
                fast,
                monotonic_ms,
                reason=window.handoff_block_reason,
            )
            return self._advance_turn_recovery(
                window,
                fast,
                monotonic_ms,
                metrics=metrics,
                decision=decision,
            )

        if self._can_start_unseen_direct_next_pass(window, expected, active):
            # Only owner_provisional has an immediately preceding expected
            # turn frame.  Reuse its explicit false marker as the edge
            # baseline; an unseen first frame still treats a visible marker as
            # stale by capturing the current value below.
            marker_baseline_visible = (
                False
                if window.expected_pass_marker_edge_while_active_unknown
                else (
                    previous_expected_pass_marker_visible
                    if window.disposition == "owner_provisional"
                    else expected in previous_pass_marker_players
                )
            )
            self._begin_unseen_direct_next_pass(
                window,
                fast,
                monotonic_ms,
                marker_baseline_visible=marker_baseline_visible,
            )
            return self._advance_unseen_direct_next_pass(
                window,
                fast,
                monotonic_ms,
                metrics=metrics,
                decision=decision,
                active_is_direct_next=True,
            )

        # With no expected-seat PASS marker, the direct successor is evidence
        # of a possibly visible play, not evidence of PASS.  Keep the normal
        # handoff open even before the owner has two authenticated frames so
        # a fast play cannot be trapped in the pass-only recovery branch.
        can_handoff = direct_next and (
            window.authenticated
            or self_lead_handoff
            or not self._matching_expected_pass_marker(fast, expected)
        )
        if can_handoff:
            if active != window.handoff_active_player or window.handoff_detected_ms is None:
                self._begin_owner_handoff(window, active, monotonic_ms)
            else:
                window.handoff_active_streak += 1
            window.crossing_active_player = None
            window.crossing_active_streak = 0
            window.crossing_non_owner_streak = 0
            readable = bool(
                decision.collect_sample
                and decision.phase == ZonePhase.BURST_READ
                and not fast.effect_visible
            )
            ownership_event = self._advance_direct_handoff(
                window,
                fast,
                monotonic_ms,
                metrics=metrics,
                decision=decision,
                readable=readable,
                block_reason=self._handoff_block_reason(metrics, decision, fast),
            )
            if ownership_event is not None:
                return ownership_event
            # A direct-handoff deadline can deliberately hand control to the
            # broader recovery path.  Do not overwrite that recovery state
            # with a normal timer disposition merely because it has no event
            # to publish on the transition frame.
            if window.turn_recovery_pending:
                return None
            if self_lead_handoff and not window.authenticated:
                if window.handoff_active_streak < 2:
                    window.disposition = "self_lead_handoff_provisional"
                else:
                    window.authenticated = True
                    window.disposition = "self_lead_handoff_authenticated"
                    self._promote_provisional_owner_samples(window)
            elif readable:
                window.disposition = "direct_next_handoff"
            else:
                window.disposition = "direct_next_handoff_waiting_readable"
            window.sample_allowed = readable
            return None
        if window.crossed_handoff_recovery_pending:
            return self._advance_crossed_handoff_recovery(
                window,
                fast,
                monotonic_ms,
                metrics=metrics,
                decision=decision,
            )
        if window.handoff_detected_ms is not None:
            if active is None:
                window.crossing_active_player = None
                window.crossing_active_streak = 0
                window.crossing_non_owner_streak = 0
                ownership_event = self._advance_direct_handoff(
                    window,
                    fast,
                    monotonic_ms,
                    metrics=metrics,
                    decision=decision,
                    readable=False,
                    block_reason="active_player_unknown",
                )
                if ownership_event is not None:
                    return ownership_event
                window.disposition = "direct_next_handoff_active_unknown"
                return None
            if self._can_start_crossed_handoff_recovery(window, fast):
                self._begin_crossed_handoff_recovery(window, fast, monotonic_ms)
                return self._advance_crossed_handoff_recovery(
                    window,
                    fast,
                    monotonic_ms,
                    metrics=metrics,
                    decision=decision,
                )
            # The direct successor can PASS while the expected player's play
            # animation is still settling.  At that point the UI has already
            # reached the next seat, but the expected player's cards often
            # remain on the table and the direct successor's PASS marker is
            # still visible.  Do not turn that recoverable window into a
            # permanent gap merely because no pre-crossing ROI sample exists.
            # Keep advice withheld and reread exactly one expected action plus
            # the intervening seat-bound PASS chain; if the evidence cannot be
            # reconstructed before the expected player acts again, turn
            # recovery itself escalates to a real history gap.
            self._begin_turn_recovery(
                window,
                fast,
                monotonic_ms,
                reason="active_player_crossed_handoff",
            )
            return self._advance_turn_recovery(
                window,
                fast,
                monotonic_ms,
                metrics=metrics,
                decision=decision,
            )
        if active is None:
            if (
                expected_pass_marker_visible
                and not previous_expected_pass_marker_visible
            ):
                # The timer vanished first and the expected seat's marker
                # appeared in that gap.  Treat it as a fresh edge only if the
                # very next readable owner is the direct successor; otherwise
                # no PASS is inferred.
                window.expected_pass_marker_edge_while_active_unknown = True
            elif not expected_pass_marker_visible:
                window.expected_pass_marker_edge_while_active_unknown = False
            window.crossing_active_player = None
            window.crossing_active_streak = 0
            window.crossing_non_owner_streak = 0
            # The active-seat decoration can vanish during a card animation.
            # The expected player's own ROI still has normal two-frame
            # consensus and rule validation, so keep that direct evidence
            # eligible instead of discarding a visible terminal play merely
            # because the timer is temporarily unknown.
            readable_expected_roi = bool(
                decision.collect_sample
                and decision.phase == ZonePhase.BURST_READ
                and not fast.effect_visible
                and not metrics.effect_visible
                and not expected_pass_marker_visible
            )
            window.disposition = (
                "owner_active_unknown"
                if readable_expected_roi
                else "isolated_active_unknown"
            )
            window.sample_allowed = readable_expected_roi
            return None
        self._clear_owner_handoff(window)
        if active == window.crossing_active_player:
            window.crossing_active_streak += 1
        else:
            window.crossing_active_player = active
            window.crossing_active_streak = 1
        window.crossing_non_owner_streak += 1
        window.disposition = "isolated_active_mismatch"
        window.handoff_block_reason = "foreign_active_mismatch"
        window.handoff_last_block_reason = window.handoff_block_reason
        # Keep one expected-ROI read as replayable diagnostic evidence, but
        # _append_sample will quarantine it from all consensus paths.
        window.sample_allowed = True
        if window.crossing_non_owner_streak < 2:
            return None
        # A non-owner active template is not a history fact.  It can lead or
        # lag the card ROI and the seat-bound PASS label by several frames.
        # Keep rereading the expected player's surface until it is actually
        # overwritten, rather than globally stopping FableDan on this timer
        # observation alone.
        self._begin_turn_recovery(
            window,
            fast,
            monotonic_ms,
            reason=window.handoff_block_reason,
        )
        return self._advance_turn_recovery(
            window,
            fast,
            monotonic_ms,
            metrics=metrics,
            decision=decision,
        )

    def _advance_previous_action_verifications(
        self,
        event: LiveEvent,
        *,
        after: LiveSnapshot,
    ) -> None:
        """Arm a real play and open only its predecessor's safe reread window.

        A correction is never allowed to reach past the penultimate formal
        action.  When another action is recorded before an open reread gains
        two matching frames, that window simply expires and its original
        history remains intact.
        """

        action_types = {"player_played", "player_passed", "manual_confirmed_event"}
        if event.event_type not in action_types:
            return

        for verification in self._previous_action_verifications.values():
            if verification.state == "open" and verification.followup_event_id != event.event_id:
                verification.state = "expired"
                self._suit_correction_tracker.clear(verification.target.event_id)
                self._pending_previous_action_retirements.append(
                    (event.event_id, verification)
                )

        for verification in self._previous_action_verifications.values():
            if (
                verification.state == "armed"
                and verification.expected_followup_actor == event.actor
            ):
                verification.followup_event_id = event.event_id
                verification.state = "open"
                verification.opened_monotonic_ms = self._last_monotonic_ms

        if event.event_type == "player_passed" or bool(event.payload.get("is_pass", False)):
            return
        if after.current_player is None:
            return
        self._previous_action_verifications[event.event_id] = _PreviousActionVerification(
            target=event,
            expected_followup_actor=after.current_player,
        )

    def _previous_action_verification_target(
        self,
        snapshot: LiveSnapshot,
    ) -> _PreviousActionVerification | None:
        """Return the one reread target that is still penultimate and open."""

        actions = [
            event
            for event in self.reducer.events
            if event.event_type in {"player_played", "player_passed", "manual_confirmed_event"}
        ]
        if len(actions) < 2:
            return None
        penultimate, latest = actions[-2:]
        verification = self._previous_action_verifications.get(penultimate.event_id)
        if (
            verification is None
            or verification.state != "open"
            or verification.target.event_id != penultimate.event_id
            or verification.followup_event_id != latest.event_id
            or latest.actor != verification.expected_followup_actor
            or snapshot.current_player is None
        ):
            return None
        return verification

    def _probe_previous_action(
        self,
        frame: np.ndarray,
        *,
        target: _PreviousActionVerification,
        wild_rank: str,
    ) -> PlayRegionResult | None:
        """Read the target's own region without treating it as a new action."""

        try:
            return self._recognize_play_region(
                frame,
                target.target.actor,
                wild_rank=wild_rank,
                allow_pass=False,
            )
        except (cv2.error, OSError, RuntimeError, ValueError):
            return None

    def _apply_previous_action_correction(
        self,
        target: _PreviousActionVerification | None,
        result: PlayRegionResult | None,
        *,
        monotonic_ms: int,
    ) -> LiveEvent | None:
        """Confirm or correct an adjacent action from two distinct rereads."""

        if target is None or target.state != "open":
            return None
        valid_distinct_result = bool(
            result is not None
            and not result.is_pass
            and result.player == target.target.actor
            and result.cards
            and target.last_probe_monotonic_ms != int(monotonic_ms)
        )
        if valid_distinct_result:
            assert result is not None
            target.last_probe_monotonic_ms = int(monotonic_ms)
            original_cards = tuple(
                str(card) for card in target.target.payload.get("cards", ())
            )
            observation = self._suit_correction_tracker.observe_visual_action(
                target.target.event_id,
                original_cards,
                tuple(str(card) for card in result.cards),
                tuple(tuple(str(suit) for suit in options) for options in result.suit_options),
            )
            if observation.confirmed:
                return self._complete_previous_action_verification(
                    target,
                    result,
                    observation=observation,
                    original_cards=original_cards,
                    monotonic_ms=monotonic_ms,
                )

        return self._expire_previous_action_verification(
            target,
            monotonic_ms=monotonic_ms,
        )

    def _complete_previous_action_verification(
        self,
        target: _PreviousActionVerification,
        result: PlayRegionResult,
        *,
        observation: SuitCorrectionObservation,
        original_cards: tuple[str, ...],
        monotonic_ms: int,
    ) -> LiveEvent | None:
        """Persist one confirmed exact/compatible reread or safe correction."""

        observed_cards = tuple(str(card) for card in observation.cards)
        evidence_kind = str(observation.evidence_kind)

        target.state = "confirmed"
        self._suit_correction_tracker.clear(target.target.event_id)
        if observed_cards == tuple(sorted(original_cards)):
            compatible = evidence_kind == "compatible"
            reason = (
                "two_distinct_candidate_compatible_rereads"
                if compatible
                else "two_distinct_adjacent_action_rereads"
            )
            source = (
                "two_frame_candidate_compatible_reread"
                if compatible
                else "two_frame_adjacent_action_reread"
            )
            event = self._append_lifecycle_event(
                "previous_action_verified",
                {
                    "target_event_id": target.target.event_id,
                    "followup_event_id": target.followup_event_id,
                    "cards": list(observed_cards),
                    "reason": reason,
                    "verification_mode": evidence_kind or "exact",
                    "original_cards_preserved": True,
                    "observed_cards": list(result.cards),
                    "observed_suit_options": [
                        list(options) for options in result.suit_options
                    ],
                },
                actor=target.target.actor,
                confidence=result.confidence,
                source=source,
            )
            self._request_advice_if_needed()
            return event

        before = self.reducer.snapshot()
        try:
            correction = self.reducer.correct_previous_action_after_followup(
                target.target.event_id,
                expected_followup_actor=target.expected_followup_actor,
                followup_event_id=target.followup_event_id or "",
                cards=observed_cards,
                reason="two_distinct_adjacent_action_rereads",
                confidence=result.confidence,
                source="two_frame_adjacent_action_reread",
            )
        except GameStateError as exc:  # reducer rolls back invalid semantic rewrites
            target.state = "expired"
            _LOGGER.warning("相邻动作复核纠正被拒绝：%s", exc)
            return None
        after = self.reducer.snapshot()
        if after.current_player != before.current_player:
            # The reducer method already rolls back violations, but keep the
            # orchestration invariant explicit: rereading an older action must
            # never change whose turn it is after the following action.
            raise RuntimeError("相邻动作复核改变了当前行动者")
        published = self._publish_event(correction)
        self._analysis_epoch += 1
        self._clear_burst()
        self._activate_zone(monotonic_ms)
        self._request_advice_if_needed()
        return published

    def _expire_previous_action_verification(
        self,
        target: _PreviousActionVerification,
        *,
        monotonic_ms: int,
    ) -> LiveEvent | None:
        """Release advice after a bounded reread while preserving history."""

        opened_ms = target.opened_monotonic_ms
        if (
            target.state != "open"
            or opened_ms is None
            or int(monotonic_ms) - opened_ms
            < _PREVIOUS_ACTION_VERIFICATION_TIMEOUT_MS
        ):
            return None
        target.state = "expired"
        self._suit_correction_tracker.clear(target.target.event_id)
        event = self._append_lifecycle_event(
            "previous_action_verification_expired",
            {
                "target_event_id": target.target.event_id,
                "followup_event_id": target.followup_event_id,
                "cards": list(target.target.payload.get("cards", ())),
                "reason": "reread_timeout_original_preserved",
                "opened_monotonic_ms": opened_ms,
                "deadline_monotonic_ms": (
                    opened_ms + _PREVIOUS_ACTION_VERIFICATION_TIMEOUT_MS
                ),
                "expired_monotonic_ms": int(monotonic_ms),
                "timeout_ms": _PREVIOUS_ACTION_VERIFICATION_TIMEOUT_MS,
                "original_cards_preserved": True,
            },
            actor=target.target.actor,
            confidence=0.0,
            source="previous_action_verification_timeout",
        )
        self._request_advice_if_needed()
        return event

    def _withhold_advice_for_previous_action_verification(
        self,
        snapshot: LiveSnapshot,
        key: AdviceRequestKey,
    ) -> bool:
        """Do not send FableDan an action history waiting for its reread."""

        target = self._previous_action_verification_target(snapshot)
        if target is None:
            return False
        if not target.advice_withheld:
            target.advice_withheld = True
            self.store.append_advice(
                {
                    "request_id": key.request_id,
                    "status": "withheld",
                    "reason": _PREVIOUS_ACTION_VERIFICATION_WITHHOLD_REASON,
                    "target_event_id": target.target.event_id,
                    "followup_event_id": target.followup_event_id,
                    "turn_id": key.turn_id,
                    "state_revision": key.state_revision,
                    **self._advisor_identity(),
                }
            )
            self._append_advice_event(
                "advice_withheld",
                {
                    "request_id": key.request_id,
                    "reason": _PREVIOUS_ACTION_VERIFICATION_WITHHOLD_REASON,
                    "target_event_id": target.target.event_id,
                    "followup_event_id": target.followup_event_id,
                    "state_revision": key.state_revision,
                },
                confidence=0.0,
            )
        self.latest_advice = LiveAdvice(
            key=key,
            status="withheld",
            error=_PREVIOUS_ACTION_VERIFICATION_WITHHOLD_TEXT,
            withhold_reason=_PREVIOUS_ACTION_VERIFICATION_WITHHOLD_REASON,
        )
        return True

    def _suit_correction_target(
        self,
        snapshot: LiveSnapshot,
    ) -> LiveEvent | None:
        """Return the one safe visual-only correction target, if any.

        The normal case is the short ``previous active seat -> self`` handoff:
        once local play controls change, an obscured suit on that immediately
        preceding action can become readable again.  ``previous active`` is
        derived from the reducer history rather than being fixed as ``left``;
        when a player has finished, the preceding active seat can be right or
        opposite.  We retain the post-self-finish probe as a second,
        display-only recovery window.  The sidecar result can never enter the
        reducer as a new play or pass.
        """

        if snapshot.current_player == "self" and snapshot.play_history:
            latest_play = snapshot.play_history[-1]
            if (
                latest_play.player != "self"
                and not latest_play.is_pass
                and any(is_unknown_suit_card(card) for card in latest_play.cards)
            ):
                target = self._play_event_for_cards(
                    latest_play.player,
                    tuple(latest_play.cards),
                )
                if target is not None:
                    return target

        if "self" not in snapshot.finished_seats or snapshot.current_player == "self":
            return None
        latest_foreign_play = next(
            (
                play
                for play in reversed(snapshot.play_history)
                if play.player != "self" and not play.is_pass
            ),
            None,
        )
        if latest_foreign_play is None or not any(
            is_unknown_suit_card(card) for card in latest_foreign_play.cards
        ):
            return None
        return self._play_event_for_cards(
            latest_foreign_play.player,
            tuple(latest_foreign_play.cards),
        )

    def _play_event_for_cards(
        self,
        player: Seat,
        cards: tuple[str, ...],
    ) -> LiveEvent | None:
        for event in reversed(self._all_events):
            if (
                event.event_type == "player_played"
                and event.actor == player
                and tuple(str(card) for card in event.payload.get("cards", ())) == cards
                and event.event_id not in self._suit_corrected_event_ids
            ):
                return event
        return None

    def _probe_suit_correction(
        self,
        frame: np.ndarray,
        *,
        target: LiveEvent,
        wild_rank: str,
    ) -> PlayRegionResult | None:
        """Run an optional seat-bound probe without jeopardizing formal read."""

        try:
            return self._recognize_play_region(
                frame,
                target.actor,
                wild_rank=wild_rank,
                allow_pass=False,
            )
        except (cv2.error, OSError, RuntimeError, ValueError):
            return None

    def _apply_suit_correction(
        self,
        target: LiveEvent | None,
        result: PlayRegionResult | None,
    ) -> LiveEvent | None:
        """Publish a two-frame display correction without changing state."""

        if target is None or result is None or result.is_pass:
            return None
        target_cards = tuple(str(card) for card in target.payload.get("cards", ()))
        observation = self._suit_correction_tracker.observe(
            target.event_id,
            target_cards,
            tuple(str(card) for card in result.cards),
        )
        if not observation.confirmed:
            return None
        corrected_cards = observation.cards
        self._suit_corrected_event_ids.add(target.event_id)
        self._suit_correction_tracker.clear(target.event_id)
        return self._append_lifecycle_event(
            "suit_corrected",
            {
                "target_event_id": target.event_id,
                "cards": list(corrected_cards),
                "reason": "two_frame_sidecar_probe",
            },
            actor=target.actor,
            confidence=result.confidence,
            source="two_frame_sidecar_probe",
        )

    def _reconstructed_remaining_cards(
        self,
        snapshot: LiveSnapshot,
    ) -> dict[Seat, int]:
        """Return the remaining-card counts implied by trusted actions."""

        played_by_seat = {seat: 0 for seat in TURN_ORDER}
        for event in snapshot.play_history:
            if not event.is_pass:
                played_by_seat[event.player] += len(event.cards)
        return {
            seat: 27 - played_by_seat[seat]
            for seat in TURN_ORDER
        }

    def _advice_history_is_complete(self, snapshot: LiveSnapshot) -> bool:
        """Require every remaining-card count to be derivable from actions.

        A visual placement badge is useful to keep the live display moving,
        but it does not say which cards were played.  Advice is safe again
        only after a later trusted/manual revision makes all four counts
        reproducible from the immutable action history.
        """

        reconstructed = self._reconstructed_remaining_cards(snapshot)
        return all(
            snapshot.remaining_cards[seat] == reconstructed[seat]
            for seat in TURN_ORDER
        )

    def _record_terminal_history_gap(
        self,
        finish_event: LiveEvent,
    ) -> LiveEvent | None:
        """Keep a terminal reconstruction mismatch as diagnostics, not a stop."""

        snapshot = self.snapshot
        if self._advice_history_is_complete(snapshot):
            return None
        reconstructed = self._reconstructed_remaining_cards(snapshot)
        mismatches = {
            seat: {
                "state_remaining_cards": int(snapshot.remaining_cards[seat]),
                "reconstructed_remaining_cards": int(reconstructed[seat]),
            }
            for seat in TURN_ORDER
            if snapshot.remaining_cards[seat] != reconstructed[seat]
        }
        return self._append_lifecycle_event(
            "terminal_history_gap",
            {
                "finish_event_id": finish_event.event_id,
                "remaining_cards": {
                    seat: int(snapshot.remaining_cards[seat])
                    for seat in TURN_ORDER
                },
                "reconstructed_remaining_cards": reconstructed,
                "mismatches": mismatches,
            },
            source="live_orchestrator",
        )

    def _clear_visual_finish_withhold_for_terminal_round(
        self,
        snapshot: LiveSnapshot,
    ) -> LiveEvent | None:
        """Remove a mid-game visual-finish guard once the round is decided."""

        if self._advice_withhold_reason != _VISUAL_FINISH_WITHHOLD_REASON:
            return None
        event = self._append_advice_event(
            "advice_withhold_cleared",
            {
                "reason": self._advice_withhold_reason,
                "finish_event_id": self._advice_withhold_finish_event_id,
                "withheld_state_revision": self._advice_withhold_revision,
                "state_revision": snapshot.revision,
                "clear_reason": "round_decided",
            },
        )
        self._advice_withhold_reason = None
        self._advice_withhold_revision = None
        self._advice_withhold_finish_event_id = None
        if (
            self.latest_advice is not None
            and self.latest_advice.status == "withheld"
            and self.latest_advice.withhold_reason
            == _VISUAL_FINISH_WITHHOLD_REASON
        ):
            self.latest_advice = None
        return event

    def _activate_visual_finish_advice_withhold(
        self,
        finish_event: LiveEvent,
    ) -> tuple[LiveEvent, ...]:
        """Persist one conservative advice-stop when a badge fills a gap."""

        snapshot = self.snapshot
        # A visual finish normally protects a later self turn whose card
        # history now contains an unknown gap.  Once a pair of team-mates has
        # already decided the round, there can be no later FableDan request.
        # Preserve the gap as terminal diagnostics instead of leaving a stale
        # "暂停推荐" state over the settlement screen.
        if snapshot.current_player is None:
            terminal_events = [
                event
                for event in (
                    self._record_terminal_history_gap(finish_event),
                    self._clear_visual_finish_withhold_for_terminal_round(snapshot),
                )
                if event is not None
            ]
            return tuple(terminal_events)
        if self._advice_withhold_reason is not None:
            return ()
        key = AdviceRequestKey(
            snapshot.session_id,
            snapshot.turn_id,
            snapshot.revision,
        )
        self._advice_withhold_reason = _VISUAL_FINISH_WITHHOLD_REASON
        self._advice_withhold_revision = snapshot.revision
        self._advice_withhold_finish_event_id = finish_event.event_id
        self.latest_advice = LiveAdvice(
            key=key,
            status="withheld",
            error=_VISUAL_FINISH_WITHHOLD_TEXT,
            withhold_reason=_VISUAL_FINISH_WITHHOLD_REASON,
        )
        self.store.append_advice(
            {
                "request_id": key.request_id,
                "status": "withheld",
                "reason": _VISUAL_FINISH_WITHHOLD_REASON,
                "finish_event_id": finish_event.event_id,
                "turn_id": key.turn_id,
                "state_revision": key.state_revision,
                **self._advisor_identity(),
            }
        )
        self._append_advice_event(
            "advice_withheld",
            {
                "request_id": key.request_id,
                "reason": _VISUAL_FINISH_WITHHOLD_REASON,
                "finish_event_id": finish_event.event_id,
                "state_revision": key.state_revision,
            },
            confidence=0.0,
        )
        self._notify_update_listener()
        return ()

    def _clear_advice_withhold_if_reconstructed(
        self,
        snapshot: LiveSnapshot,
    ) -> bool:
        """Clear a visual-fallback stop only for a newer complete revision."""

        held_revision = self._advice_withhold_revision
        if self._advice_withhold_reason is None:
            return True
        if (
            held_revision is None
            or snapshot.revision <= held_revision
            or not self._advice_history_is_complete(snapshot)
        ):
            return False
        self._append_advice_event(
            "advice_withhold_cleared",
            {
                "reason": self._advice_withhold_reason,
                "finish_event_id": self._advice_withhold_finish_event_id,
                "withheld_state_revision": held_revision,
                "state_revision": snapshot.revision,
            },
        )
        self._advice_withhold_reason = None
        self._advice_withhold_revision = None
        self._advice_withhold_finish_event_id = None
        return True

    def _request_advice_if_needed(self) -> AdviceRequestKey | None:
        if self._advice_suspended_reason is not None:
            return None
        if self.advisor is None or self.status != "running":
            return None
        snapshot = self.snapshot
        if (
            snapshot.current_player != "self"
            or "self" in snapshot.finished_seats
            or not snapshot.my_hand
        ):
            return None
        key = AdviceRequestKey(
            snapshot.session_id,
            snapshot.turn_id,
            snapshot.revision,
        )
        with self._advice_lock:
            if self._withhold_advice_for_previous_action_verification(snapshot, key):
                return None
            if not self._clear_advice_withhold_if_reconstructed(snapshot):
                # The placement path emits the single audit record.  Repeated
                # frames must be inert: no AdviceJob, FableDan trace, or
                # duplicate withheld event is permitted for this revision.
                self.latest_advice = LiveAdvice(
                    key=key,
                    status="withheld",
                    error=_VISUAL_FINISH_WITHHOLD_TEXT,
                    withhold_reason=_VISUAL_FINISH_WITHHOLD_REASON,
                )
                return None
            if key in self._requested_advice:
                return key
            state = self.reducer.to_guandan_state()
            self._decision_id_by_revision[key.state_revision] = key.decision_id
            self._requested_advice.add(key)
            self._advice_completion_events[key] = Event()
            self._advice_requested_at_ms[key] = self._last_monotonic_ms
            self.latest_advice = LiveAdvice(key=key, status="requested")
            self.store.append_advice(
                {
                    "request_id": key.request_id,
                    "status": "requested",
                    "turn_id": key.turn_id,
                    "state_revision": key.state_revision,
                    **self._advisor_identity(),
                }
            )
            self.store.upsert_decision(
                {
                    "decision_id": key.decision_id,
                    "request_id": key.request_id,
                    "actor": "self",
                    "turn_id": key.turn_id,
                    "trick_id": snapshot.trick_id,
                    "state_revision": key.state_revision,
                    "state_before": self._decision_state(state),
                    "label_status": "draft",
                    "status": "requested",
                }
            )
            self._append_advice_event(
                "advice_requested",
                {
                    "request_id": key.request_id,
                    "turn_id": key.turn_id,
                    "state_revision": key.state_revision,
                },
            )
            worker = self._advice_worker
            assert worker is not None
            worker.submit(_AdviceJob(key, state))
            return key

    def _run_advice(self, job: _AdviceJob) -> _AdviceCompletion:
        self.store.append_advice(
            {
                "request_id": job.key.request_id,
                "status": "worker_started",
                "turn_id": job.key.turn_id,
                "state_revision": job.key.state_revision,
                **self._advisor_identity(),
            }
        )
        suit_expansion = state_variants_for_unknown_suits_detailed(
            job.state,
            limit=_MAX_SUIT_STATE_VARIANTS,
        )
        raw_suit_variants = suit_expansion.states
        has_unknown_suit = any(
            is_unknown_suit_card(card)
            for card in job.state.my_hand
        ) or any(
            is_unknown_suit_card(card)
            for event in job.state.play_history
            for card in event.cards
        )
        if not raw_suit_variants:
            return _AdviceCompletion(
                job.key,
                error=(
                    "牌局历史中的未知花色无法分配：候选花色与当前手牌及双副牌"
                    "每张实体牌最多两张的约束冲突"
                ),
                engine_input=self._fallback_engine_input(job),
                suit_uncertain=has_unknown_suit,
                variant_count=0,
                advice_agrees_across_variants=False,
                suit_variant_count=0,
                suit_equivalence_class_count=0,
            )
        if suit_expansion.truncated:
            uncertainty = {
                "max_suit_variant_count": _MAX_SUIT_STATE_VARIANTS,
                "suit_uncertain": has_unknown_suit,
                "generated_suit_variant_count": len(raw_suit_variants),
                "suit_variants_truncated": True,
                "used_relaxed_suits": suit_expansion.used_relaxed_suits,
            }
            engine_input = self._fallback_engine_input(job)
            engine_input["uncertainty_diagnostics"] = uncertainty
            return _AdviceCompletion(
                job.key,
                error=(
                    f"{self._advisor_display_name()} 计算已阻断：未知花色至少产生 "
                    f"{_MAX_SUIT_STATE_VARIANTS + 1} 个可行实体牌状态，超过安全上限 "
                    f"{_MAX_SUIT_STATE_VARIANTS}；为避免遗漏花色解释，本次未调用模型"
                ),
                engine_input=engine_input,
                suit_uncertain=has_unknown_suit,
                variant_count=0,
                advice_agrees_across_variants=False,
                suit_variant_count=len(raw_suit_variants),
                suit_equivalence_class_count=0,
                uncertainty_diagnostics=uncertainty,
            )

        equivalence_strategy = (
            "advisor_encoded_input_fingerprint"
            if callable(getattr(self.advisor, "decision_input_fingerprint", None))
            else "none"
        )
        suit_variants = _deduplicate_advisor_input_variants(
            raw_suit_variants,
            self.advisor,
        )
        suit_variant_count = len(raw_suit_variants)
        suit_equivalence_class_count = len(suit_variants)
        if (
            suit_equivalence_class_count > _MAX_ADVICE_STATE_VARIANTS
            and equivalence_strategy == "none"
        ):
            uncertainty = {
                "max_variant_count": _MAX_ADVICE_STATE_VARIANTS,
                "suit_uncertain": has_unknown_suit,
                "suit_variant_count": suit_variant_count,
                "suit_equivalence_class_count": suit_equivalence_class_count,
                "suit_equivalence_strategy": equivalence_strategy,
                "suit_variants_truncated": False,
                "used_relaxed_suits": suit_expansion.used_relaxed_suits,
            }
            engine_input = self._fallback_engine_input(job)
            engine_input["uncertainty_diagnostics"] = uncertainty
            return _AdviceCompletion(
                job.key,
                error=(
                    f"{self._advisor_display_name()} 计算已阻断：{suit_variant_count} 个"
                    f"花色状态仍形成 {suit_equivalence_class_count} 个模型非等价输入，"
                    f"超过推理上限 {_MAX_ADVICE_STATE_VARIANTS}；本次未执行部分评估"
                ),
                engine_input=engine_input,
                suit_uncertain=has_unknown_suit,
                variant_count=0,
                advice_agrees_across_variants=False,
                suit_variant_count=suit_variant_count,
                suit_equivalence_class_count=suit_equivalence_class_count,
                uncertainty_diagnostics=uncertainty,
            )

        variants: list[tuple[GuanDanState, int, int]] = []
        semantic_diagnostics: list[dict[str, object]] = []
        semantic_source_indices: set[int] = set()
        semantic_variant_count = 1
        for suit_index, suit_state in enumerate(suit_variants, start=1):
            semantic = state_variants_for_action_semantics(
                suit_state,
                limit=_MAX_ADVICE_STATE_VARIANTS,
            )
            diagnostic = semantic.to_diagnostic()
            diagnostic["suit_variant_index"] = suit_index
            semantic_diagnostics.append(diagnostic)
            semantic_source_indices.update(semantic.source_history_indices)
            semantic_variant_count = max(
                semantic_variant_count,
                semantic.total_variant_count,
            )
            if semantic.error:
                uncertainty = {
                    "max_variant_count": _MAX_ADVICE_STATE_VARIANTS,
                    "suit_uncertain": has_unknown_suit,
                    "suit_variant_count": suit_variant_count,
                    "suit_equivalence_class_count": suit_equivalence_class_count,
                    "suit_equivalence_strategy": equivalence_strategy,
                    "semantic_uncertain": semantic.is_uncertain,
                    "semantic_source_history_indices": list(
                        semantic.source_history_indices
                    ),
                    "semantic_expansions": semantic_diagnostics,
                }
                engine_input = self._fallback_engine_input(job)
                engine_input["uncertainty_diagnostics"] = uncertainty
                return _AdviceCompletion(
                    job.key,
                    error=f"{self._advisor_display_name()} 计算已阻断：{semantic.error}",
                    engine_input=engine_input,
                    suit_uncertain=has_unknown_suit,
                    variant_count=0,
                    advice_agrees_across_variants=False,
                    semantic_uncertain=semantic.is_uncertain,
                    semantic_source_history_indices=semantic.source_history_indices,
                    suit_variant_count=suit_variant_count,
                    suit_equivalence_class_count=suit_equivalence_class_count,
                    semantic_variant_count=semantic.total_variant_count,
                    uncertainty_diagnostics=uncertainty,
                )
            if (
                len(variants) + len(semantic.states) > _MAX_ADVICE_STATE_VARIANTS
                and equivalence_strategy == "none"
            ):
                sources = sorted(semantic_source_indices)
                source_text = "、".join(str(index) for index in sources) or "无"
                uncertainty = {
                    "max_variant_count": _MAX_ADVICE_STATE_VARIANTS,
                    "suit_uncertain": has_unknown_suit,
                    "suit_variant_count": suit_variant_count,
                    "suit_equivalence_class_count": suit_equivalence_class_count,
                    "suit_equivalence_strategy": equivalence_strategy,
                    "semantic_uncertain": bool(sources),
                    "semantic_source_history_indices": sources,
                    "semantic_expansions": semantic_diagnostics,
                }
                engine_input = self._fallback_engine_input(job)
                engine_input["uncertainty_diagnostics"] = uncertainty
                return _AdviceCompletion(
                    job.key,
                    error=(
                        f"{self._advisor_display_name()} 计算已阻断：花色与动作语义组合后的"
                        f"完整状态超过 {_MAX_ADVICE_STATE_VARIANTS} 个；涉及历史第 "
                        f"{source_text} 条。为避免只评估部分分支，本次未调用模型"
                    ),
                    engine_input=engine_input,
                    suit_uncertain=has_unknown_suit,
                    variant_count=0,
                    advice_agrees_across_variants=False,
                    semantic_uncertain=bool(sources),
                    semantic_source_history_indices=tuple(sources),
                    suit_variant_count=suit_variant_count,
                    suit_equivalence_class_count=suit_equivalence_class_count,
                    semantic_variant_count=semantic_variant_count,
                    uncertainty_diagnostics=uncertainty,
                )
            variants.extend(
                (state, suit_index, semantic_index)
                for semantic_index, state in enumerate(semantic.states, start=1)
            )

        semantic_uncertain = bool(semantic_source_indices)
        source_indices = tuple(sorted(semantic_source_indices))
        input_variant_count = len(variants)
        variants, eliminated_variants, input_validation_errors = (
            _validate_and_deduplicate_encoded_advice_variants(
                variants,
                self.advisor,
                source_indices,
            )
        )
        validation_uncertainty: dict[str, object] = {
            "max_variant_count": _MAX_ADVICE_STATE_VARIANTS,
            "suit_uncertain": has_unknown_suit,
            "suit_variant_count": suit_variant_count,
            "suit_equivalence_class_count": suit_equivalence_class_count,
            "suit_equivalence_strategy": equivalence_strategy,
            "semantic_uncertain": semantic_uncertain,
            "semantic_source_history_indices": list(source_indices),
            "semantic_variant_count": semantic_variant_count,
            "input_variant_count_before_validation": input_variant_count,
            "valid_encoded_input_count": len(variants),
            "eliminated_variant_count": len(eliminated_variants),
            "eliminated_variants": eliminated_variants,
            "input_validation_errors": input_validation_errors,
            "semantic_expansions": semantic_diagnostics,
        }
        if input_validation_errors:
            primary, details = _compact_input_failure_text(input_validation_errors)
            validation_uncertainty["branch_results"] = input_validation_errors
            engine_input = self._fallback_engine_input(job)
            engine_input["uncertainty_diagnostics"] = validation_uncertainty
            return _AdviceCompletion(
                job.key,
                error=(
                    f"{self._advisor_display_name()} 计算已阻断：模型输入预校验发生"
                    f"不能安全忽略的异常。首个错误：{primary}。"
                    f"分支详情：{details}。本次未调用模型，主牌局历史未被改写"
                ),
                engine_input=engine_input,
                suit_uncertain=has_unknown_suit,
                variant_count=0,
                advice_agrees_across_variants=False,
                semantic_uncertain=semantic_uncertain,
                semantic_source_history_indices=source_indices,
                suit_variant_count=suit_variant_count,
                suit_equivalence_class_count=suit_equivalence_class_count,
                semantic_variant_count=semantic_variant_count,
                uncertainty_diagnostics=validation_uncertainty,
            )
        if not variants and eliminated_variants:
            primary, details = _compact_input_failure_text(eliminated_variants)
            validation_uncertainty["branch_results"] = eliminated_variants
            validation_uncertainty["root_cause"] = {
                "kind": "no_legal_complete_state",
                "message": primary,
            }
            engine_input = self._fallback_engine_input(job)
            engine_input["uncertainty_diagnostics"] = validation_uncertainty
            return _AdviceCompletion(
                job.key,
                error=(
                    f"{self._advisor_display_name()} 计算已阻断：{input_variant_count} 个"
                    "候选状态全部被完整牌局历史排除，模型未被调用。"
                    f"最深可达根因：{primary}。分支消歧：{details}。"
                    "请先修正最深可达根因对应的原始出牌识别"
                ),
                engine_input=engine_input,
                suit_uncertain=has_unknown_suit,
                variant_count=0,
                advice_agrees_across_variants=False,
                semantic_uncertain=semantic_uncertain,
                semantic_source_history_indices=source_indices,
                suit_variant_count=suit_variant_count,
                suit_equivalence_class_count=suit_equivalence_class_count,
                semantic_variant_count=semantic_variant_count,
                uncertainty_diagnostics=validation_uncertainty,
            )
        if len(variants) > _MAX_ADVICE_STATE_VARIANTS:
            sources = list(source_indices)
            source_text = "、".join(str(index) for index in sources) or "无"
            uncertainty = {
                "max_variant_count": _MAX_ADVICE_STATE_VARIANTS,
                "suit_uncertain": has_unknown_suit,
                "suit_variant_count": suit_variant_count,
                "suit_equivalence_class_count": suit_equivalence_class_count,
                "suit_equivalence_strategy": equivalence_strategy,
                "semantic_uncertain": bool(sources),
                "semantic_source_history_indices": sources,
                "encoded_input_class_count": len(variants),
                "input_variant_count_before_validation": input_variant_count,
                "eliminated_variant_count": len(eliminated_variants),
                "eliminated_variants": eliminated_variants,
                "semantic_expansions": semantic_diagnostics,
            }
            engine_input = self._fallback_engine_input(job)
            engine_input["uncertainty_diagnostics"] = uncertainty
            return _AdviceCompletion(
                job.key,
                error=(
                    f"{self._advisor_display_name()} 计算已阻断：完整分支归并后仍有 "
                    f"{len(variants)} 个不同的真实编码输入，超过推理上限 "
                    f"{_MAX_ADVICE_STATE_VARIANTS}；涉及历史第 {source_text} 条，"
                    "本次未执行部分评估"
                ),
                engine_input=engine_input,
                suit_uncertain=has_unknown_suit,
                variant_count=0,
                advice_agrees_across_variants=False,
                semantic_uncertain=bool(sources),
                semantic_source_history_indices=tuple(sources),
                suit_variant_count=suit_variant_count,
                suit_equivalence_class_count=suit_equivalence_class_count,
                semantic_variant_count=semantic_variant_count,
                uncertainty_diagnostics=uncertainty,
            )

        uncertainty: dict[str, object] = {
            "max_variant_count": _MAX_ADVICE_STATE_VARIANTS,
            "suit_uncertain": has_unknown_suit,
            "suit_variant_count": suit_variant_count,
            "suit_equivalence_class_count": suit_equivalence_class_count,
            "suit_equivalence_strategy": equivalence_strategy,
            "semantic_uncertain": semantic_uncertain,
            "semantic_source_history_indices": list(source_indices),
            "semantic_variant_count": semantic_variant_count,
            "encoded_input_class_count": len(variants),
            "evaluated_variant_count": len(variants),
            "input_variant_count_before_validation": input_variant_count,
            "valid_encoded_input_count": len(variants),
            "eliminated_variant_count": len(eliminated_variants),
            "eliminated_variants": eliminated_variants,
            "input_validation_errors": input_validation_errors,
            "semantic_expansions": semantic_diagnostics,
            "branch_results": list(eliminated_variants),
        }
        advice_groups: dict[tuple[bool, tuple[str, ...], str], list[LocalAdvice]] = {}
        first_trace: dict[str, object] | None = None
        first_engine_input: dict[str, object] | None = None
        errors: list[str] = []
        branch_results: list[dict[str, object]] = list(eliminated_variants)
        parameters = signature(self.advisor.recommend).parameters
        for index, (state, suit_index, semantic_index) in enumerate(variants, start=1):
            request_id = (
                job.key.request_id
                if len(variants) == 1
                else f"{job.key.request_id}/variant-{index}"
            )
            trace = StrategyExecutionTrace(request_id)
            try:
                kwargs: dict[str, object] = {"request_id": request_id}
                if "trace" in parameters:
                    kwargs["trace"] = trace
                advice = self.advisor.recommend(state, **kwargs)
            except Exception as exc:
                error = f"{type(exc).__name__}: {exc}"
                errors.append(error)
                branch_results.append(
                    {
                        "variant_index": index,
                        "suit_variant_index": suit_index,
                        "semantic_variant_index": semantic_index,
                        "semantic_choices": _state_semantic_choices(
                            state,
                            source_indices,
                        ),
                        "status": "failed",
                        "error": error,
                    }
                )
                trace_snapshot = trace.snapshot()
                if first_trace is None:
                    first_trace = trace_snapshot
                    candidate_input = trace_snapshot.get("engine_input")
                    if isinstance(candidate_input, dict):
                        first_engine_input = candidate_input
                continue
            key = (advice.is_pass, tuple(advice.cards), advice.play_type)
            advice_groups.setdefault(key, []).append(advice)
            branch_results.append(
                {
                    "variant_index": index,
                    "suit_variant_index": suit_index,
                    "semantic_variant_index": semantic_index,
                    "semantic_choices": _state_semantic_choices(
                        state,
                        source_indices,
                    ),
                    "status": "ready",
                    "advice": {
                        "is_pass": advice.is_pass,
                        "cards": list(advice.cards),
                        "play_type": advice.play_type,
                        "readable": _readable_advice(advice),
                    },
                }
            )
            if first_trace is None:
                first_trace = trace.snapshot()
                candidate_input = first_trace.get("engine_input")
                if isinstance(candidate_input, dict):
                    first_engine_input = candidate_input

        uncertainty["branch_results"] = branch_results
        if semantic_uncertain and errors:
            failed = [
                item for item in branch_results if item.get("status") == "failed"
            ]
            failed_text = "；".join(
                f"分支 {item['variant_index']}：{item['error']}" for item in failed
            )
            engine_input = dict(
                first_engine_input or self._fallback_engine_input(job)
            )
            engine_input["uncertainty_diagnostics"] = uncertainty
            return _AdviceCompletion(
                job.key,
                error=(
                    f"{self._advisor_display_name()} 计算已阻断：历史第 "
                    f"{'、'.join(str(value) for value in source_indices)} 条动作存在多种语义，"
                    f"但 {len(failed)} 个模型分支执行失败，无法比较全部候选。{failed_text}"
                ),
                engine_input=engine_input,
                trace=first_trace,
                suit_uncertain=has_unknown_suit,
                variant_count=len(variants),
                advice_agrees_across_variants=False,
                semantic_uncertain=True,
                semantic_source_history_indices=source_indices,
                suit_variant_count=suit_variant_count,
                suit_equivalence_class_count=suit_equivalence_class_count,
                semantic_variant_count=semantic_variant_count,
                uncertainty_diagnostics=uncertainty,
            )
        if not advice_groups:
            engine_input = dict(
                first_engine_input or self._fallback_engine_input(job)
            )
            engine_input["uncertainty_diagnostics"] = uncertainty
            return _AdviceCompletion(
                job.key,
                error=(
                    errors[0]
                    if errors
                    else f"{self._advisor_display_name()} 未返回建议"
                ),
                engine_input=engine_input,
                trace=first_trace,
                suit_uncertain=has_unknown_suit,
                variant_count=len(variants),
                advice_agrees_across_variants=False,
                semantic_uncertain=semantic_uncertain,
                semantic_source_history_indices=source_indices,
                suit_variant_count=suit_variant_count,
                suit_equivalence_class_count=suit_equivalence_class_count,
                semantic_variant_count=semantic_variant_count,
                uncertainty_diagnostics=uncertainty,
            )
        if semantic_uncertain and len(advice_groups) > 1:
            ready_results = [
                item for item in branch_results if item.get("status") == "ready"
            ]
            result_text = "；".join(
                f"分支 {item['variant_index']}：{item['advice']['readable']}"
                for item in ready_results
            )
            engine_input = dict(
                first_engine_input or self._fallback_engine_input(job)
            )
            engine_input["uncertainty_diagnostics"] = uncertainty
            return _AdviceCompletion(
                job.key,
                error=(
                    f"{self._advisor_display_name()} 无法给出唯一建议：历史第 "
                    f"{'、'.join(str(value) for value in source_indices)} 条动作存在多种合法语义，"
                    f"{len(ready_results)} 个完整状态分支给出了不同建议（{result_text}）。"
                    "主牌局历史未被改写，请确认该历史动作的逢人配声明"
                ),
                engine_input=engine_input,
                trace=first_trace,
                suit_uncertain=has_unknown_suit,
                variant_count=len(variants),
                advice_agrees_across_variants=False,
                semantic_uncertain=True,
                semantic_source_history_indices=source_indices,
                suit_variant_count=suit_variant_count,
                suit_equivalence_class_count=suit_equivalence_class_count,
                semantic_variant_count=semantic_variant_count,
                uncertainty_diagnostics=uncertainty,
            )
        winner = max(advice_groups.values(), key=len)
        return _AdviceCompletion(
            job.key,
            advice=winner[0],
            suit_uncertain=has_unknown_suit,
            variant_count=len(variants),
            advice_agrees_across_variants=len(advice_groups) == 1,
            semantic_uncertain=semantic_uncertain,
            semantic_source_history_indices=source_indices,
            suit_variant_count=suit_variant_count,
            suit_equivalence_class_count=suit_equivalence_class_count,
            semantic_variant_count=semantic_variant_count,
            uncertainty_diagnostics=uncertainty,
        )

    @staticmethod
    def _fallback_engine_input(job: _AdviceJob) -> dict[str, object]:
        state = job.state
        return {
            "request_id": job.key.request_id,
            "project_snapshot": {
                "round_level": state.round_level,
                "wild_rank": state.wild_rank,
                "current_player": state.current_player,
                "lead_player": state.lead_player,
                "my_hand": list(state.my_hand),
                "trick_plays": [event.to_dict() for event in state.trick_plays],
                "play_history": [event.to_dict() for event in state.play_history],
                "revision": state.revision,
            },
        }

    @_state_synchronized
    def _complete_advice_job(self, completion: _AdviceCompletion) -> None:
        key = completion.key
        if not self._accept_advice_results:
            self._signal_advice_completion(key)
            return
        with self._advice_lock:
            snapshot = self.snapshot
            current_key = AdviceRequestKey(
                snapshot.session_id,
                snapshot.turn_id,
                snapshot.revision,
            )
            if self._advice_suspended_reason is not None:
                self._signal_advice_completion(key)
                return
            if not self._clear_advice_withhold_if_reconstructed(snapshot):
                # A worker may have started just before the visual fallback.
                # Its result is intentionally discarded without a trace so a
                # lossily reconstructed turn can never surface as advice.
                self._signal_advice_completion(key)
                return
            if snapshot.current_player != "self" or key != current_key:
                self.store.append_advice(
                    {
                        "request_id": key.request_id,
                        "status": "stale",
                        "turn_id": key.turn_id,
                        "state_revision": key.state_revision,
                        "error": completion.error,
                        "engine_input": completion.engine_input,
                        "trace": completion.trace,
                        "semantic_uncertain": completion.semantic_uncertain,
                        "semantic_source_history_indices": list(
                            completion.semantic_source_history_indices
                        ),
                        "evaluated_variant_count": completion.variant_count,
                    }
                )
                self._append_advice_event(
                    "advice_stale",
                    {
                        "request_id": key.request_id,
                        "turn_id": key.turn_id,
                        "state_revision": key.state_revision,
                        "current_state_revision": snapshot.revision,
                    },
                )
                if self.latest_advice is not None and self.latest_advice.key == key:
                    self.latest_advice = LiveAdvice(
                        key=key,
                        status="stale",
                        advice=completion.advice,
                        error=completion.error,
                        suit_uncertain=completion.suit_uncertain,
                        variant_count=completion.variant_count,
                        advice_agrees_across_variants=completion.advice_agrees_across_variants,
                        semantic_uncertain=completion.semantic_uncertain,
                        semantic_source_history_indices=completion.semantic_source_history_indices,
                        suit_variant_count=completion.suit_variant_count,
                        suit_equivalence_class_count=completion.suit_equivalence_class_count,
                        semantic_variant_count=completion.semantic_variant_count,
                    )
                self._signal_advice_completion(key)
                return
            if completion.error or completion.advice is None:
                error = (
                    completion.error
                    or f"{self._advisor_display_name()} 未返回建议"
                )
                # Publish the incident before exposing the failed advice state.
                # Consumers use the state transition as the readiness signal and
                # must never observe ``status=failed`` while its evidence bundle
                # is still being assembled.
                self._create_incident(
                    "advisor_failed",
                    self._last_monotonic_ms,
                    engine_input=completion.engine_input,
                )
                self.latest_advice = LiveAdvice(
                    key=key,
                    status="failed",
                    error=error,
                    suit_uncertain=completion.suit_uncertain,
                    variant_count=completion.variant_count,
                    advice_agrees_across_variants=completion.advice_agrees_across_variants,
                    semantic_uncertain=completion.semantic_uncertain,
                    semantic_source_history_indices=completion.semantic_source_history_indices,
                    suit_variant_count=completion.suit_variant_count,
                    suit_equivalence_class_count=completion.suit_equivalence_class_count,
                    semantic_variant_count=completion.semantic_variant_count,
                )
                self.store.append_advice(
                    {
                        "request_id": key.request_id,
                        "status": "failed",
                        "turn_id": key.turn_id,
                        "state_revision": key.state_revision,
                        "error": error,
                        "engine_input": completion.engine_input,
                        "trace": completion.trace,
                        "suit_uncertain": completion.suit_uncertain,
                        "suit_variant_count": completion.suit_variant_count,
                        "suit_equivalence_class_count": completion.suit_equivalence_class_count,
                        "semantic_uncertain": completion.semantic_uncertain,
                        "semantic_source_history_indices": list(
                            completion.semantic_source_history_indices
                        ),
                        "semantic_variant_count": completion.semantic_variant_count,
                        "evaluated_variant_count": completion.variant_count,
                        "advice_agrees_across_variants": completion.advice_agrees_across_variants,
                        "uncertainty_diagnostics": completion.uncertainty_diagnostics,
                        **self._advisor_identity(),
                    }
                )
                self._append_advice_event(
                    "advice_failed",
                    {
                        "request_id": key.request_id,
                        "error": error,
                        "semantic_uncertain": completion.semantic_uncertain,
                        "semantic_source_history_indices": list(
                            completion.semantic_source_history_indices
                        ),
                        "evaluated_variant_count": completion.variant_count,
                    },
                    confidence=0.0,
                )
                self._signal_advice_completion(key)
                self._notify_update_listener()
                return
            advice = completion.advice
            missing_cards = Counter(advice.cards) - Counter(snapshot.my_hand)
            if not advice.is_pass and missing_cards:
                missing_text = " ".join(sorted(missing_cards.elements()))
                error = (
                    f"{self._advisor_display_name()} 建议包含当前手牌中不存在的牌："
                    f"{missing_text}"
                )
                self.latest_advice = LiveAdvice(
                    key=key,
                    status="failed",
                    error=error,
                    suit_uncertain=completion.suit_uncertain,
                    variant_count=completion.variant_count,
                    advice_agrees_across_variants=completion.advice_agrees_across_variants,
                    semantic_uncertain=completion.semantic_uncertain,
                    semantic_source_history_indices=completion.semantic_source_history_indices,
                    suit_variant_count=completion.suit_variant_count,
                    suit_equivalence_class_count=completion.suit_equivalence_class_count,
                    semantic_variant_count=completion.semantic_variant_count,
                )
                self.store.append_advice(
                    {
                        "request_id": key.request_id,
                        "status": "failed",
                        "turn_id": key.turn_id,
                        "state_revision": key.state_revision,
                        "error": error,
                        **self._advisor_identity(),
                    }
                )
                self._append_advice_event(
                    "advice_failed",
                    {
                        "request_id": key.request_id,
                        "error": error,
                        "reason": "cards_not_in_current_hand",
                    },
                    confidence=0.0,
                )
                self._signal_advice_completion(key)
                self._notify_update_listener()
                return
            visible = self._self_turn_corroborated
            if visible:
                self._set_advice_visible_latency(key)
            self.latest_advice = LiveAdvice(
                key=key,
                status="ready",
                advice=advice,
                visible=visible,
                suit_uncertain=completion.suit_uncertain,
                variant_count=completion.variant_count,
                advice_agrees_across_variants=completion.advice_agrees_across_variants,
                semantic_uncertain=completion.semantic_uncertain,
                semantic_source_history_indices=completion.semantic_source_history_indices,
                suit_variant_count=completion.suit_variant_count,
                suit_equivalence_class_count=completion.suit_equivalence_class_count,
                semantic_variant_count=completion.semantic_variant_count,
            )
            self.store.append_advice(
                {
                    "request_id": key.request_id,
                    "status": "ready",
                    "turn_id": key.turn_id,
                    "state_revision": key.state_revision,
                    "cards": list(advice.cards),
                    "is_pass": advice.is_pass,
                    "play_type": advice.play_type,
                    "strategy": advice.strategy,
                    "engine_input": advice.engine_input,
                    "timings": advice.timings,
                    "elapsed_ms": advice.elapsed_ms,
                    "visible": visible,
                    "suit_uncertain": completion.suit_uncertain,
                    "suit_variant_count": completion.suit_variant_count,
                    "suit_equivalence_class_count": completion.suit_equivalence_class_count,
                    "semantic_uncertain": completion.semantic_uncertain,
                    "semantic_source_history_indices": list(
                        completion.semantic_source_history_indices
                    ),
                    "semantic_variant_count": completion.semantic_variant_count,
                    "evaluated_variant_count": completion.variant_count,
                    "advice_agrees_across_variants": completion.advice_agrees_across_variants,
                    "uncertainty_diagnostics": completion.uncertainty_diagnostics,
                    **self._advisor_identity(),
                }
            )
            engine_input = advice.engine_input or {}
            self.store.upsert_decision(
                {
                    "decision_id": key.decision_id,
                    "status": "ready",
                    "legal_actions": list(engine_input.get("legal_actions", ())),
                    "feature_schema": engine_input.get("feature_schema"),
                    "features_567": engine_input.get("features_567"),
                    "fabledan_training_input": engine_input.get(
                        "fabledan_training_input"
                    ),
                    "model_advice": {
                        "cards": list(advice.cards),
                        "is_pass": advice.is_pass,
                        "play_type": advice.play_type,
                        "strategy": advice.strategy,
                    },
                }
            )
            self._append_advice_event(
                "advice_ready",
                {
                    "request_id": key.request_id,
                    "cards": list(advice.cards),
                    "is_pass": advice.is_pass,
                    "play_type": advice.play_type,
                    "state_revision": key.state_revision,
                    "visible": visible,
                    "suit_uncertain": completion.suit_uncertain,
                    "suit_variant_count": completion.suit_variant_count,
                    "suit_equivalence_class_count": completion.suit_equivalence_class_count,
                    "semantic_uncertain": completion.semantic_uncertain,
                    "semantic_source_history_indices": list(
                        completion.semantic_source_history_indices
                    ),
                    "semantic_variant_count": completion.semantic_variant_count,
                    "evaluated_variant_count": completion.variant_count,
                    "advice_agrees_across_variants": completion.advice_agrees_across_variants,
                },
            )
            self._signal_advice_completion(key)
            self._notify_update_listener()

    def _discard_advice_job(self, value: object, reason: str) -> None:
        """Give every requested-but-unconsumed advice job an auditable terminal."""

        if not isinstance(value, _AdviceJob):
            return
        key = value.key
        status = "stale" if reason in {"latest_replaced", "preserved_evicted"} else "cancelled"
        error = (
            "advice request superseded by a newer state"
            if status == "stale"
            else "advice request cancelled while stopping"
        )
        with self._advice_lock:
            self.store.append_advice(
                {
                    "request_id": key.request_id,
                    "status": status,
                    "turn_id": key.turn_id,
                    "state_revision": key.state_revision,
                    "discard_reason": reason,
                    "error": error,
                    "engine_input": self._fallback_engine_input(value),
                    **self._advisor_identity(),
                }
            )
            self.store.upsert_decision(
                {
                    "decision_id": key.decision_id,
                    "request_id": key.request_id,
                    "turn_id": key.turn_id,
                    "state_revision": key.state_revision,
                    "status": status,
                    "error": error,
                    "discard_reason": reason,
                }
            )
            self._append_advice_event(
                "advice_stale" if status == "stale" else "advice_cancelled",
                {
                    "request_id": key.request_id,
                    "turn_id": key.turn_id,
                    "state_revision": key.state_revision,
                    "discard_reason": reason,
                },
            )
            if self.latest_advice is not None and self.latest_advice.key == key:
                self.latest_advice = LiveAdvice(
                    key=key,
                    status="stale",
                    error=error,
                )
        self._signal_advice_completion(key)

    def _notify_update_listener(self) -> None:
        """Expose an advice transition immediately instead of waiting for a frame."""

        listener = self._update_listener
        if listener is not None:
            listener(self._update())

    def _signal_advice_completion(self, key: AdviceRequestKey) -> None:
        with self._advice_lock:
            event = self._advice_completion_events.get(key)
        if event is not None:
            event.set()

    def _apply_fast_signal(self, fast: FastSignalResult) -> None:
        if self.snapshot.current_player != "self":
            return
        if fast.active_player not in (None, "self"):
            return
        corroborated = (
            fast.active_player == "self" or fast.self_action_buttons_visible
        )
        if not corroborated:
            return
        with self._advice_lock:
            self._self_turn_corroborated = True
            current = self.latest_advice
            if current is None or current.status != "ready" or current.visible:
                return
            self.latest_advice = LiveAdvice(
                key=current.key,
                status=current.status,
                advice=current.advice,
                visible=True,
                error=current.error,
                suit_uncertain=current.suit_uncertain,
                variant_count=current.variant_count,
                advice_agrees_across_variants=current.advice_agrees_across_variants,
                semantic_uncertain=current.semantic_uncertain,
                semantic_source_history_indices=current.semantic_source_history_indices,
                suit_variant_count=current.suit_variant_count,
                suit_equivalence_class_count=current.suit_equivalence_class_count,
                semantic_variant_count=current.semantic_variant_count,
            )
            self._set_advice_visible_latency(current.key)
            self._append_advice_event(
                "advice_visible",
                {"request_id": current.key.request_id},
            )

    def _append_advice_event(
        self,
        event_type: str,
        payload: dict[str, object],
        *,
        confidence: float = 1.0,
    ) -> LiveEvent:
        snapshot = self.snapshot
        event_payload = self._advisor_identity()
        event_payload.update(payload)
        self._aux_event_sequence += 1
        event = LiveEvent(
            event_id=f"AUX-{self._aux_event_sequence:06d}",
            event_type=event_type,
            session_id=snapshot.session_id,
            seq=len(self._all_events) + 1,
            monotonic_ms=self._last_monotonic_ms,
            wall_time=datetime.now().astimezone().isoformat(),
            trick_id=max(1, snapshot.trick_id),
            turn_id=max(1, snapshot.turn_id),
            actor="self",
            payload=event_payload,
            confidence=float(confidence),
            source="live_advice_coordinator",
            state_revision_before=snapshot.revision,
            state_revision_after=snapshot.revision,
        )
        return self._publish_event(event)

    def _advisor_identity(self) -> dict[str, object]:
        strategy = str(getattr(self.advisor, "strategy_id", "") or "").strip()
        display_name = str(
            getattr(self.advisor, "display_name", "") or ""
        ).strip()
        return {
            "advisor_strategy": strategy or "unknown",
            "advisor_name": display_name or "建议模型",
        }

    def _advisor_display_name(self) -> str:
        return str(self._advisor_identity()["advisor_name"])

    def _publish_event(self, event: LiveEvent) -> LiveEvent:
        with self._advice_lock:
            self._published_sequence += 1
            published = replace(
                event,
                seq=self._published_sequence,
                monotonic_ms=self._last_monotonic_ms,
            )
            self.store.append_event(published)
            self._all_events.append(published)
            return published

    def _publish_action_with_outcomes(
        self,
        event: LiveEvent,
        before: LiveSnapshot,
    ) -> tuple[LiveEvent, tuple[LiveEvent, ...]]:
        """Persist the action first, then append its non-semantic outcomes."""

        published = self._publish_event(event)
        verification_retirements = self._publish_previous_action_retirements(
            published
        )
        if published.actor == "self":
            decision_id = self._decision_id_by_revision.get(
                published.state_revision_before
            )
            if decision_id is not None:
                self.store.upsert_decision(
                    {
                        "decision_id": decision_id,
                        "actual_action_event_id": published.event_id,
                        "actual_turn_id": published.turn_id,
                        "actual_trick_id": published.trick_id,
                        "actual_action": {
                            "cards": list(published.payload.get("cards", ())),
                            "is_pass": published.event_type == "player_passed"
                            or bool(published.payload.get("is_pass", False)),
                        },
                    }
                )
        outcomes = self._append_action_outcomes(
            before,
            self.reducer.snapshot(),
            trigger_action_event_id=published.event_id,
            trigger_actor=published.actor,
        )
        return published, (*verification_retirements, *outcomes)

    def _publish_previous_action_retirements(
        self,
        trigger: LiveEvent,
    ) -> tuple[LiveEvent, ...]:
        """Make next-action retirement explicit without changing advancement."""

        matching: list[_PreviousActionVerification] = []
        retained: list[tuple[str, _PreviousActionVerification]] = []
        for trigger_event_id, verification in self._pending_previous_action_retirements:
            if trigger_event_id == trigger.event_id:
                matching.append(verification)
            else:
                retained.append((trigger_event_id, verification))
        self._pending_previous_action_retirements = retained

        published: list[LiveEvent] = []
        emitted_target_ids: set[str] = set()
        for verification in matching:
            target_id = verification.target.event_id
            if target_id in emitted_target_ids:
                continue
            emitted_target_ids.add(target_id)
            opened_ms = verification.opened_monotonic_ms
            published.append(
                self._append_lifecycle_event(
                    "previous_action_verification_expired",
                    {
                        "target_event_id": target_id,
                        "followup_event_id": verification.followup_event_id,
                        "retirement_action_event_id": trigger.event_id,
                        "cards": list(
                            verification.target.payload.get("cards", ())
                        ),
                        "reason": "next_formal_action_original_preserved",
                        "opened_monotonic_ms": opened_ms,
                        "retired_monotonic_ms": self._last_monotonic_ms,
                        "elapsed_ms": (
                            None
                            if opened_ms is None
                            else max(0, self._last_monotonic_ms - opened_ms)
                        ),
                        "original_cards_preserved": True,
                    },
                    actor=verification.target.actor,
                    confidence=0.0,
                    source="previous_action_verification_retirement",
                )
            )
        return tuple(published)

    @staticmethod
    def _decision_state(state: GuanDanState) -> dict[str, object]:
        return {
            "round_level": state.round_level,
            "wild_rank": state.wild_rank,
            "current_player": state.current_player,
            "lead_player": state.lead_player,
            "my_hand": list(state.my_hand),
            "trick": [event.to_dict() for event in state.trick_plays],
            "history": [event.to_dict() for event in state.play_history],
            "remaining_cards": dict(state.remaining_cards),
            "revision": state.revision,
        }

    def _append_action_outcomes(
        self,
        before: LiveSnapshot,
        after: LiveSnapshot,
        *,
        trigger_action_event_id: str = "",
        trigger_actor: Seat | None = None,
    ) -> tuple[LiveEvent, ...]:
        outcomes: list[LiveEvent] = []
        newly_finished = sorted(
            after.finished_seats - before.finished_seats,
            key=TURN_ORDER.index,
        )
        placement_names = ("head", "second", "third")
        for player in newly_finished:
            if player in self._finish_order:
                continue
            self._finish_order.append(player)
            position = len(self._finish_order) - 1
            if position >= len(placement_names):
                continue
            payload: dict[str, object] = {"placement": placement_names[position]}
            if player == trigger_actor and trigger_action_event_id:
                payload["trigger_action_event_id"] = trigger_action_event_id
            outcomes.append(
                self._append_lifecycle_event(
                    "player_finished",
                    payload,
                    actor=player,
                )
            )
            if position == 2:
                last_players = [
                    seat
                    for seat in TURN_ORDER
                    if seat not in after.finished_seats and seat not in self._finish_order
                ]
                if len(last_players) == 1:
                    last_player = last_players[0]
                    self._finish_order.append(last_player)
                    outcomes.append(
                        self._append_lifecycle_event(
                            "player_finished",
                            {"placement": "last"},
                            actor=last_player,
                        )
                    )

        finished_leader = next(
            (
                play.player
                for play in reversed(before.trick_plays)
                if not play.is_pass
                and play.player in after.finished_seats
                and project_trick_turn(
                    play.player,
                    after.finished_seats,
                ).wind_receiver
                == after.lead_player
            ),
            None,
        )
        # Lifecycle-only snapshots created by an older recorder did not keep
        # the closing trick plays.  Preserve their unambiguous lead-based
        # event while live transitions above use the actual finishing play.
        if (
            finished_leader is None
            and before.lead_player in before.finished_seats
            and project_trick_turn(
                before.lead_player,
                after.finished_seats,
            ).wind_receiver
            == after.lead_player
        ):
            finished_leader = before.lead_player
        if (
            before.trick_id != after.trick_id
            and finished_leader is not None
            and after.lead_player is not None
        ):
            outcomes.append(
                self._append_lifecycle_event(
                    "wind_caught",
                    {
                        "from_player": finished_leader,
                        "to_player": after.lead_player,
                    },
                    actor=after.lead_player,
                )
            )
        return tuple(outcomes)

    def _apply_visual_placements(
        self,
        fast: FastSignalResult,
        *,
        defer_player: Seat | None = None,
    ) -> tuple[LiveEvent, ...]:
        """Commit persistent placement badges after two matching frames.

        Opponent starting counts are not always inferable from our own 27-card
        hand.  Explicit placement badges therefore override only that
        player's remaining count and finished state; they never fabricate an
        action or card face.
        """

        by_player = {
            signal.player: signal
            for signal in getattr(fast, "placements", ())
            if signal.player in TURN_ORDER
        }
        for player in tuple(self._placement_streaks):
            if player not in by_player:
                self._placement_streaks.pop(player, None)

        if self._placement_is_deferred_for_expected_action(fast, defer_player):
            # A rank badge is not action evidence.  In particular, the second
            # teammate badge can arrive while the other opponent's PASS is
            # still visible, and marking that teammate as finished would
            # immediately decide the round before the PASS (and the following
            # expected play) can reach the normal reducer path.  Do not retain
            # a partial badge streak across an actionable turn either: the
            # fallback must be freshly stable once that action surface is gone.
            for player in by_player:
                self._placement_streaks.pop(player, None)
            return ()

        completed: list[LiveEvent] = []
        placement_order = {"head": 0, "second": 1, "third": 2, "last": 3}
        placement_sequence = ("head", "second", "third")
        signals = sorted(
            by_player.values(),
            key=lambda item: placement_order.get(str(item.placement), 99),
        )
        for signal in signals:
            player = signal.player
            if player in self.reducer.snapshot().finished_seats:
                self._placement_streaks.pop(player, None)
                continue
            if player == defer_player:
                # A placement badge frequently appears before the final cards
                # have stopped animating.  When it belongs to the player whose
                # action is currently awaited, treating it as completion would
                # erase a still-readable terminal play (and any following
                # PASS/接风 chain).  Let the ordinary action recognizer prove
                # the final play; once it commits, reducer lifecycle events
                # record the same placement without a history gap.
                self._placement_streaks.pop(player, None)
                continue
            placement = str(signal.placement).strip().lower()
            expected_placement = (
                placement_sequence[len(self._finish_order)]
                if len(self._finish_order) < len(placement_sequence)
                else None
            )
            # A visually similar status decoration must never invent an
            # impossible finish order.  Out-of-order labels are discarded,
            # including their accumulated streak, so a later valid label has
            # to become stable from scratch.
            if placement != expected_placement:
                self._placement_streaks.pop(player, None)
                continue
            previous = self._placement_streaks.get(player)
            streak = (
                previous[1] + 1
                if previous is not None and previous[0] == placement
                else 1
            )
            self._placement_streaks[player] = (placement, streak)
            # A finish badge often appears on the same frames as the final
            # card.  Give the normal two-valid-card path several frames to
            # submit that action first.  If animation/card recognition never
            # settles, the badge still acts as a bounded fallback instead of
            # leaving the whole game stuck on this player.
            required_streak = 6 if player == defer_player else 2
            if streak < required_streak:
                continue
            self._placement_streaks.pop(player, None)
            event = self.reducer.confirm_player_finished(
                player,
                placement=placement,
                confidence=float(signal.confidence),
                source=f"two_frame_placement:{signal.source}",
            )
            published = self._publish_event(event)
            completed.append(published)
            # This is a display-only fallback, not a reconstructed card
            # action.  In a live round it stops advice before the surrounding
            # frame can request a job that depends on unknown history.  If
            # the badge instead decides the round, preserve any mismatch only
            # as terminal diagnostics; there is no future recommendation to
            # protect.
            completed.extend(
                self._activate_visual_finish_advice_withhold(published)
            )
            if player not in self._finish_order:
                self._finish_order.append(player)

            if placement == "third":
                after = self.reducer.snapshot()
                remaining = [
                    seat
                    for seat in TURN_ORDER
                    if seat not in after.finished_seats
                    and seat not in self._finish_order
                ]
                if len(remaining) == 1:
                    last_player = remaining[0]
                    self._finish_order.append(last_player)
                    completed.append(
                        self._append_lifecycle_event(
                            "player_finished",
                            {"placement": "last"},
                            actor=last_player,
                            source="inferred_after_visual_third",
                        )
                    )
        return tuple(completed)

    def _placement_is_deferred_for_expected_action(
        self,
        fast: FastSignalResult,
        expected: Seat | None,
    ) -> bool:
        """Keep visual fallback behind an unresolved expected-seat PASS.

        An ordinary active timer, PASS marker or local action control defers
        the same frame.  More importantly, a fresh direct-next PASS recovery
        owns its complete two-marker decision window; a placement badge must
        not decide the round while that reducer-valid recovery remains
        pending.  No timer-only or fixed-frame extension is used.
        """

        if expected is None:
            return False
        window = self._turn_ownership_window
        if (
            window is not None
            and window.expected_player == expected
            and window.unseen_direct_next_pass_pending
        ):
            return True
        return bool(
            fast.active_player == expected
            or self._matching_expected_pass_marker(fast, expected)
            or (expected == "self" and fast.self_action_buttons_visible)
        )

    def _append_lifecycle_event(
        self,
        event_type: str,
        payload: dict[str, object],
        *,
        actor: Seat | None = None,
        confidence: float = 1.0,
        source: str = "live_orchestrator",
    ) -> LiveEvent:
        snapshot = self.reducer.snapshot()
        self._aux_event_sequence += 1
        event = LiveEvent(
            event_id=f"AUX-{self._aux_event_sequence:06d}",
            event_type=event_type,
            session_id=snapshot.session_id,
            seq=0,
            monotonic_ms=self._last_monotonic_ms,
            wall_time=datetime.now().astimezone().isoformat(),
            trick_id=max(1, snapshot.trick_id),
            turn_id=max(1, snapshot.turn_id),
            actor=actor,
            payload=dict(payload),
            confidence=float(confidence),
            source=source,
            state_revision_before=snapshot.revision,
            state_revision_after=snapshot.revision,
        )
        return self._publish_event(event)

    def _append_current_turn_started(self) -> LiveEvent | None:
        player = self.reducer.snapshot().current_player
        if player is not None:
            return self._append_lifecycle_event(
                "turn_started",
                {"player": player},
                actor=player,
            )
        return None

    def _set_advice_visible_latency(self, key: AdviceRequestKey) -> None:
        requested_at = self._advice_requested_at_ms.get(key)
        if requested_at is not None:
            self._advice_visible_latency_ms = max(
                0,
                self._last_monotonic_ms - requested_at,
            )

    def _append_sample(
        self,
        result: PlayRegionResult,
        monotonic_ms: int,
        *,
        fast: FastSignalResult,
        metrics: ZoneFrameMetrics,
        decision: ZoneDecision,
    ) -> None:
        self._observation_sequence += 1
        observation_id = f"OBS-{self._observation_sequence:06d}"
        sample = RecognitionSample(
            cards=result.cards,
            is_pass=result.is_pass,
            confidence=result.confidence,
            source=result.source,
            evidence_ref=observation_id,
            suit_options=result.suit_options,
            post_hand=result.post_hand,
        )
        owner_window = self._ensure_turn_ownership_window()
        owner_disposition = (
            owner_window.disposition if owner_window is not None else "isolated_no_window"
        )
        accepted_for_consensus = owner_disposition in {
            "owner_authenticated",
            "owner_active_unknown",
            "self_lead_handoff_authenticated",
            "direct_next_handoff",
            "crossed_handoff_recovery",
            "turn_recovery",
        } and bool(owner_window and owner_window.sample_allowed)
        provisional_owner_sample = owner_disposition in {
            "owner_provisional",
            "self_lead_handoff_provisional",
        }
        is_readable_handoff_sample = bool(
            owner_window is not None
            and owner_disposition
            in {
                "self_lead_handoff_authenticated",
                "self_lead_handoff_provisional",
                "direct_next_handoff",
                "crossed_handoff_recovery",
                "turn_recovery",
            }
            and owner_window.handoff_readable_since_ms is not None
        )
        if is_readable_handoff_sample:
            owner_window.handoff_sample_count += 1
        record: dict[str, object] = {
            "id": observation_id,
            "monotonic_ms": int(monotonic_ms),
            "player": result.player,
            "cards": list(result.cards),
            "suit_options": [list(options) for options in result.suit_options],
            "is_pass": result.is_pass,
            "confidence": result.confidence,
            "source": result.source,
            "post_hand": list(result.post_hand),
            "post_hand_confidence": result.post_hand_confidence,
            "diagnostics": list(result.diagnostics),
            "phase": "burst_read",
            "self_action_controls_seen": self._self_lead_controls_seen,
            "self_action_controls_cleared": self._self_lead_controls_cleared,
            "first_action_gate": {
                "reason": self._first_action_gate_reason,
                "pending": self._first_action_pending,
            },
            "analysis_window": {
                "epoch": self._analysis_epoch,
                "sample_count_before": len(self._samples),
                "first_action_sample_count_before": len(self._first_action_samples),
                "recognition_strategy": self.recognition_strategy.value,
            },
            "fast_signals": {
                "active_player": fast.active_player,
                "self_action_buttons_visible": fast.self_action_buttons_visible,
                "pass_visible": fast.pass_visible,
                "pass_marker_player": fast.pass_marker_player,
                "pass_marker_players": list(
                    getattr(fast, "pass_marker_players", ())
                ),
                "effect_visible": fast.effect_visible,
            },
            "zone": {
                "expected_player": self._zone.expected_player if self._zone else None,
                **self._zone_telemetry(metrics, decision, fast),
            },
            "ownership": {
                "owner": owner_window.expected_player if owner_window else None,
                "active": fast.active_player,
                "window_key": list(owner_window.key) if owner_window else None,
                "owner_active_streak": (
                    owner_window.owner_active_streak if owner_window else 0
                ),
                "crossing_active_streak": (
                    owner_window.crossing_active_streak if owner_window else 0
                ),
                "crossing_non_owner_streak": (
                    owner_window.crossing_non_owner_streak if owner_window else 0
                ),
                "authenticated": owner_window.authenticated if owner_window else False,
                "disposition": owner_disposition,
                "accepted_for_consensus": accepted_for_consensus,
                "sample_allowed": owner_window.sample_allowed if owner_window else False,
                "handoff": (
                    self._handoff_telemetry(owner_window)
                    if owner_window is not None
                    else {}
                ),
                "unseen_direct_next_pass": (
                    self._unseen_direct_next_pass_telemetry(owner_window)
                    if owner_window is not None
                    else {}
                ),
                "crossed_handoff_recovery": (
                    self._crossed_handoff_recovery_telemetry(owner_window)
                    if owner_window is not None
                    else {}
                ),
                "wind_catch_pass_recovery": (
                    self._wind_catch_pass_recovery_telemetry(owner_window)
                    if owner_window is not None
                    else {}
                ),
                "turn_recovery": (
                    self._turn_recovery_telemetry(owner_window)
                    if owner_window is not None
                    else {}
                ),
            },
            "hand_card_count_before": len(self.snapshot.my_hand)
            if result.player == "self"
            else None,
            "hand_card_count_after": len(result.post_hand)
            if result.player == "self" and result.post_hand
            else None,
        }
        self._observations.append(record)
        if owner_window is None:
            pass
        elif accepted_for_consensus:
            self._append_consensus_sample(sample, owner_window)
            if is_readable_handoff_sample and owner_disposition in {
                "direct_next_handoff",
                "crossed_handoff_recovery",
            }:
                owner_window.handoff_samples.append(sample)
            elif owner_disposition == "turn_recovery":
                owner_window.handoff_samples.append(sample)
        elif provisional_owner_sample:
            owner_window.provisional_samples.append(sample)
        else:
            owner_window.isolated_sample_count += 1
        self.store.append_observation(record)
        self._last_sample_ms = int(monotonic_ms)

    def _sample_due(self, monotonic_ms: int) -> bool:
        return self._last_sample_ms is None or (
            int(monotonic_ms) - self._last_sample_ms >= self.burst_sample_interval_ms
        )

    def _decide_if_ready(
        self,
        metrics: ZoneFrameMetrics,
        fast: FastSignalResult,
    ) -> ConsensusResult | None:
        context = self._consensus_context(metrics, fast)
        samples = self._samples
        if self._is_first_action_turn():
            samples = self._first_action_samples
        result = decide_recognition_strategy(
            self.recognition_strategy,
            samples,
            context=context,
        )
        if (
            result is not None
            or len(samples) < self.burst_sample_limit
            or not context.next_turn_evidence
        ):
            return result
        return decide_best_effort_candidate(
            samples,
            context=context,
        )

    def _recognition_retry_reason(
        self,
        metrics: ZoneFrameMetrics,
        fast: FastSignalResult,
    ) -> str | None:
        context = self._consensus_context(metrics, fast)
        samples = (
            self._first_action_samples
            if self._is_first_action_turn()
            else self._samples
        )
        if has_exhausted_valid_candidates(
            samples,
            context=context,
            limit=self.burst_sample_limit,
        ):
            return "conflicting_valid_candidates"
        return None

    def _decide_opening_handoff_anchor(
        self,
        window: _TurnOwnershipWindow | None,
        metrics: ZoneFrameMetrics,
        fast: FastSignalResult,
    ) -> ConsensusResult | None:
        """Recover a stable opening play after its timer has already advanced.

        This is deliberately not another general consensus strategy.  It only
        covers the narrow asynchronous opening handoff where the lead is
        already known, the table is still empty, and the timer is on that
        lead's direct successor.  The normal strategy always has first
        priority; this receives only its ``None`` result.
        """

        if (
            window is None
            or not self._lead_auto_confirmed_from_marker
            or not self._is_first_action_turn(window.expected_player)
            or bool(self.snapshot.trick_plays)
            or not self._is_direct_next_active(window.expected_player, fast.active_player)
            or fast.effect_visible
            or metrics.effect_visible
            or window.disposition != "direct_next_handoff"
        ):
            return None

        latest_samples = window.handoff_samples[-2:]
        if len(latest_samples) != 2:
            return None

        context = self._consensus_context(metrics, fast)
        valid: list[RecognitionSample] = []
        for sample in latest_samples:
            if sample.is_pass:
                return None
            cards, suit_options = canonical_candidate(
                sample.cards,
                sample.suit_options,
                is_pass=False,
            )
            if not cards or BurstConsensus.validate_candidate(
                False,
                cards,
                context,
                suit_options=suit_options,
            ):
                return None
            valid.append(RecognitionSample(
                cards=cards,
                is_pass=False,
                confidence=sample.confidence,
                source=sample.source,
                evidence_ref=sample.evidence_ref,
                suit_options=suit_options,
                post_hand=sample.post_hand,
            ))
        if (
            len({sample.evidence_ref for sample in valid if sample.evidence_ref}) != 2
            or (valid[0].cards, valid[0].suit_options)
            != (valid[1].cards, valid[1].suit_options)
        ):
            return None

        sample = valid[-1]
        confidence = sum(item.confidence for item in valid) / len(valid)
        candidate = ConsensusCandidate(
            cards=sample.cards,
            is_pass=False,
            votes=len(valid),
            mean_confidence=confidence,
            valid=True,
        )
        return ConsensusResult(
            status="confirmed",
            cards=sample.cards,
            is_pass=False,
            confidence=confidence,
            source="opening_handoff_two_valid_anchor",
            vote_count=len(valid),
            candidates=(candidate,),
            resolved_cards=BurstConsensus.resolve_commit_cards(
                False,
                sample.cards,
                context,
                suit_options=sample.suit_options,
            ),
            evidence_refs=tuple(
                item.evidence_ref for item in valid if item.evidence_ref
            ),
            suit_options=sample.suit_options,
            integrity_warnings=BurstConsensus.integrity_warnings(
                False,
                sample.cards,
                context,
                suit_options=sample.suit_options,
            ),
        )

    def _decide_timeout_candidate(
        self,
        metrics: ZoneFrameMetrics,
        fast: FastSignalResult,
    ) -> ConsensusResult | None:
        """Commit only corroborated evidence when an action window expires.

        A capture/analysis handoff can let samples reach the append-only log
        while the normal decision pass is invalidated by an intervening UI
        update.  Before emitting a timeout, retry the exact same rule-aware
        strategy against the retained first-action evidence.  The fallback is
        intentionally limited to two matching legal reads: it rescues a
        visible stable play such as ``7D 7S`` without promoting a lone
        animation fragment or pass marker into game state.
        """

        context = self._consensus_context(metrics, fast)
        samples = (
            self._first_action_samples
            if self._is_first_action_turn() and self._first_action_samples
            else self._samples
        )
        if not samples:
            return None
        strategy_result = decide_recognition_strategy(
            self.recognition_strategy,
            samples,
            context=context,
        )
        if strategy_result is not None and strategy_result.status == "confirmed":
            return strategy_result
        best_effort = decide_best_effort_candidate(samples, context=context)
        if best_effort is None or best_effort.vote_count < 2:
            return None
        return best_effort

    def _consensus_context(
        self,
        metrics: ZoneFrameMetrics,
        fast: FastSignalResult,
    ) -> ConsensusContext:
        return self._consensus_context_for_snapshot(self.snapshot, metrics, fast)

    @staticmethod
    def _consensus_context_for_snapshot(
        snapshot: LiveSnapshot,
        metrics: ZoneFrameMetrics,
        fast: FastSignalResult,
    ) -> ConsensusContext:
        player = snapshot.current_player
        assert player is not None
        table_event = next(
            (play for play in reversed(snapshot.trick_plays) if not play.is_pass),
            None,
        )
        historical_cards = tuple(
            card
            for event in snapshot.play_history
            if not event.is_pass
            for card in event.cards
        )
        historical_options = tuple(
            option
            for event in snapshot.play_history
            if not event.is_pass
            for option in normalized_suit_options(event.cards, event.suit_options)
        )
        return ConsensusContext(
            level_rank=snapshot.wild_rank,
            remaining_cards=snapshot.remaining_cards[player],
            allow_pass=bool(snapshot.trick_plays),
            known_hand=snapshot.my_hand if player == "self" else (),
            table_cards=table_event.cards if table_event is not None else (),
            table_suit_options=(
                normalized_suit_options(table_event.cards, table_event.suit_options)
                if table_event is not None
                else ()
            ),
            known_cards=tuple(snapshot.my_hand) + historical_cards,
            known_suit_options=(
                normalized_suit_options(snapshot.my_hand) + historical_options
            ),
            candidate_already_known=player == "self",
            region_empty=not metrics.occupied,
            next_turn_evidence=(
                fast.active_player is not None and fast.active_player != player
            ),
            # 当前手牌仍只用于校验出牌是否属于已知手牌；不再重识别整手牌。
            # 牌型和压牌规则仍然保留，防止视觉结果直接污染状态机。
            validate_rules=True,
        )

    def _commit_consensus(
        self,
        result: ConsensusResult,
        monotonic_ms: int,
        *,
        fast: FastSignalResult | None = None,
        suppress_turn_side_effects: bool = False,
    ) -> tuple[LiveEvent, tuple[LiveEvent, ...]]:
        player = self.snapshot.current_player
        assert player is not None
        before = self.reducer.snapshot()
        commit_cards = result.resolved_cards if not result.is_pass else result.cards
        action_metadata: dict[str, object] = {}
        integrity_warnings = list(result.integrity_warnings)
        if not result.is_pass:
            table_event = next(
                (play for play in reversed(before.trick_plays) if not play.is_pass),
                None,
            )
            table_cards = table_event.cards if table_event is not None else ()
            preferred_play_type = None
            current_advice = self.latest_advice
            if (
                player == "self"
                and current_advice is not None
                and current_advice.status == "ready"
                and current_advice.advice is not None
                and current_advice.key
                == AdviceRequestKey(before.session_id, before.turn_id, before.revision)
                and Counter(current_advice.advice.cards) == Counter(commit_cards)
            ):
                preferred_play_type = project_play_type(
                    current_advice.advice.play_type
                )
            try:
                inference = infer_best_action(
                    commit_cards,
                    table_cards,
                    before.wild_rank,
                    preferred_play_type=preferred_play_type,
                )
            except (ImportError, ModuleNotFoundError, ValueError):
                inference = None
            if inference is None or inference.action is None:
                integrity_warnings.append("observed_pattern_unresolved")
            else:
                action_metadata = {
                    "play_type": str(inference.action[0]),
                    "logical_rank": str(inference.action[1]),
                    "logical_label": inference.logical_label,
                    "beats_table": inference.beats_table,
                    "interpretation_ambiguous": inference.ambiguous,
                    "wildcard_substitutions": [
                        {"card": card, "as_rank": rank}
                        for card, rank in inference.wildcard_substitutions
                    ],
                    "candidate_interpretations": [
                        {
                            "move_type": str(action[0]),
                            "key": str(action[1]),
                            "logical_label": logical_action_label(
                                action, before.wild_rank
                            ),
                            "wildcard_assignments": [
                                {"physical_card": card, "as_rank": rank}
                                for card, rank in wildcard_substitutions(
                                    action, before.wild_rank
                                )
                            ],
                        }
                        for action in inference.candidate_actions
                    ],
                    "selected_interpretation": (
                        None
                        if inference.ambiguous
                        else {
                            "move_type": str(inference.action[0]),
                            "key": str(inference.action[1]),
                            "logical_label": inference.logical_label,
                            "wildcard_assignments": [
                                {"physical_card": card, "as_rank": rank}
                                for card, rank in inference.wildcard_substitutions
                            ],
                        }
                    ),
                    "selection_source": (
                        "unresolved" if inference.ambiguous else "realtime_semantics"
                    ),
                }
                if table_cards and not inference.beats_table:
                    integrity_warnings.append("observed_table_mismatch")
                if inference.ambiguous:
                    integrity_warnings.append("wildcard_interpretation_ambiguous")
        event = self._record_action(
            player,
            commit_cards,
            result.is_pass,
            confidence=result.confidence,
            source=result.source,
            evidence_refs=result.evidence_refs,
            # A reconciled self action is now an exact physical hand action;
            # do not retain stale visual suit alternatives on the event.
            suit_options=(
                () if commit_cards != result.cards else result.suit_options
            ),
            integrity_warnings=tuple(dict.fromkeys(integrity_warnings)),
            action_metadata=action_metadata,
        )
        event, outcomes = self._publish_action_with_outcomes(event, before)
        self._first_action_pending = False
        self._clear_first_action_candidates()
        after = self.reducer.snapshot()
        self._activate_zone(
            monotonic_ms,
            accept_initial_occupied=bool(
                fast is not None
                and after.current_player is not None
                and fast.active_player == after.current_player
                and after.trick_id == before.trick_id
            ),
        )
        if suppress_turn_side_effects:
            return event, (event, *outcomes)
        turn_started = self._append_current_turn_started()
        self._request_advice_if_needed()
        events = (event, *outcomes) + ((turn_started,) if turn_started else ())
        return event, events

    def _commit_crossed_handoff_recovery(
        self,
        result: ConsensusResult,
        monotonic_ms: int,
        *,
        fast: FastSignalResult,
    ) -> tuple[LiveEvent, tuple[LiveEvent, ...]]:
        """Commit a visible play and its verified direct-next PASS together."""

        window = self._ensure_turn_ownership_window()
        if not self._crossed_handoff_recovery_is_ready(window, result):
            raise RuntimeError("跨座位交接恢复尚未满足提交条件")
        assert window is not None
        pass_player = window.crossed_handoff_recovery_pass_player
        if pass_player is None:
            raise RuntimeError("跨座位交接恢复缺少不出玩家")

        # Do not emit a turn-start or request advice after the recovered play
        # alone.  The state lock keeps the immediate PASS reconstruction in
        # this same analysis step before the caller receives an update.
        play_event, play_events = self._commit_consensus(
            result,
            monotonic_ms,
            fast=fast,
            suppress_turn_side_effects=True,
        )
        before_pass = self.reducer.snapshot()
        if before_pass.current_player != pass_player:
            raise RuntimeError("跨座位交接恢复后的不出玩家不匹配")
        pass_event = self._record_action(
            pass_player,
            (),
            True,
            confidence=1.0,
            source="crossed_handoff_recovery_pass_marker",
            evidence_refs=result.evidence_refs,
        )
        pass_event, pass_outcomes = self._publish_action_with_outcomes(
            pass_event,
            before_pass,
        )
        after = self.reducer.snapshot()
        self._activate_zone(
            monotonic_ms,
            accept_initial_occupied=bool(
                after.current_player is not None
                and fast.active_player == after.current_player
                and after.trick_id == before_pass.trick_id
            ),
        )
        recovery_event = self._append_lifecycle_event(
            "crossed_handoff_recovered",
            {
                "play_event_id": play_event.event_id,
                "pass_event_id": pass_event.event_id,
                "play_player": play_event.actor,
                "pass_player": pass_player,
                "active_player": fast.active_player,
                "pass_marker_streak": window.crossed_handoff_recovery_pass_marker_streak,
                "play_vote_count": result.vote_count,
            },
            actor=play_event.actor,
            confidence=result.confidence,
            source="crossed_handoff_recovery",
        )
        self._append_recognition_trace(
            window=window,
            outcome="crossed_handoff_recovered",
            strategy_result=result,
            fallback=True,
            reason="two_frame_pass_marker_and_play_reread",
            commit_attempted=True,
            commit_event_id=pass_event.event_id,
        )
        turn_started = self._append_current_turn_started()
        self._request_advice_if_needed()
        events = (
            *play_events,
            pass_event,
            *pass_outcomes,
            recovery_event,
            *((turn_started,) if turn_started else ()),
        )
        return pass_event, events

    def _commit_turn_recovery(
        self,
        result: ConsensusResult,
        monotonic_ms: int,
        *,
        fast: FastSignalResult,
    ) -> tuple[LiveEvent, tuple[LiveEvent, ...]]:
        """Commit a delayed expected action and its proven intervening PASSes."""

        window = self._ensure_turn_ownership_window()
        if not self._turn_recovery_is_ready(window, result, fast):
            raise RuntimeError("回合容错恢复尚未满足提交条件")
        assert window is not None

        action_event, action_events = self._commit_consensus(
            result,
            monotonic_ms,
            fast=fast,
            suppress_turn_side_effects=True,
        )
        events: list[LiveEvent] = [*action_events]
        last_action_event = action_event
        after_action = self.reducer.snapshot()
        recovered_passes = self._intervening_passes_from(
            after_action.current_player,
            fast.active_player,
            after_action.finished_seats,
        )
        if recovered_passes is None:
            raise RuntimeError("回合容错恢复后的活动座位顺序不匹配")
        for pass_player in recovered_passes:
            before_pass = self.reducer.snapshot()
            if before_pass.current_player != pass_player:
                raise RuntimeError("回合容错恢复中的不出玩家不匹配")
            pass_event = self._record_action(
                pass_player,
                (),
                True,
                confidence=1.0,
                source="turn_recovery_pass_marker",
                evidence_refs=result.evidence_refs,
            )
            pass_event, pass_outcomes = self._publish_action_with_outcomes(
                pass_event,
                before_pass,
            )
            events.extend((pass_event, *pass_outcomes))
            last_action_event = pass_event

        after = self.reducer.snapshot()
        if after.current_player != fast.active_player:
            raise RuntimeError("回合容错恢复后的活动座位不匹配")
        self._activate_zone(
            monotonic_ms,
            accept_initial_occupied=bool(
                after.current_player is not None
                and fast.active_player == after.current_player
            ),
        )
        recovery_event = self._append_lifecycle_event(
            "turn_recovery_recovered",
            {
                "action_event_id": action_event.event_id,
                "action_player": action_event.actor,
                "recovered_pass_players": list(recovered_passes),
                "active_player": fast.active_player,
                "action_vote_count": result.vote_count,
                "pass_marker_streaks": {
                    player: window.turn_recovery_pass_marker_streaks.get(player, 0)
                    for player in recovered_passes
                },
            },
            actor=action_event.actor,
            confidence=result.confidence,
            source="turn_recovery",
        )
        self._append_recognition_trace(
            window=window,
            outcome="turn_recovery_recovered",
            strategy_result=result,
            fallback=True,
            reason="recovered_before_expected_player_next_turn",
            commit_attempted=True,
            commit_event_id=last_action_event.event_id,
        )
        turn_started = self._append_current_turn_started()
        self._request_advice_if_needed()
        events.extend((recovery_event, *((turn_started,) if turn_started else ())))
        return last_action_event, tuple(events)

    def _record_action(
        self,
        player: Seat,
        cards: tuple[str, ...],
        is_pass: bool,
        *,
        confidence: float,
        source: str,
        evidence_refs: tuple[str, ...] = (),
        suit_options: tuple[tuple[str, ...], ...] = (),
        integrity_warnings: tuple[str, ...] = (),
        action_metadata: dict[str, object] | None = None,
    ) -> LiveEvent:
        if is_pass:
            event = self.reducer.record_pass(
                player,
                confidence=confidence,
                source=source,
                evidence_refs=evidence_refs,
            )
        else:
            event = self.reducer.record_play(
                player,
                cards,
                confidence=confidence,
                source=source,
                evidence_refs=evidence_refs,
                suit_options=suit_options,
                integrity_warnings=integrity_warnings,
                action_metadata=action_metadata,
            )
        self._advance_previous_action_verifications(
            event,
            after=self.reducer.snapshot(),
        )
        return event

    def _require_review(
        self,
        reason: str,
        monotonic_ms: int,
        fast: FastSignalResult | None = None,
        consensus: ConsensusResult | None = None,
    ) -> LiveUpdate:
        player = self.snapshot.current_player
        if self._first_action_pending and player == self.snapshot.lead_player:
            if player == "self" and not self._self_lead_controls_cleared:
                self._reset_waiting_self_lead(monotonic_ms)
                return self._update(fast_signals=fast)
            # Recognition retries are not a completed first action.  Retain
            # the opening-play guard and continuity samples until a consensus
            # or a user-confirmed action formally advances the reducer.
        candidates = tuple(
            f"CAND-{index}"
            for index, _candidate in enumerate(
                consensus.candidates if consensus is not None else (),
                start=1,
            )
        )
        self._review_count += 1
        self.latest_review = None
        # Preserve the rejected burst before resetting the action window.
        # Otherwise a timeout incident says "no observations" precisely when
        # the user needs to inspect the cards that were seen under an effect.
        incident_observations = list(self._observations)
        message = _recognition_retry_message(reason, incident_observations)
        self._clear_burst()
        if player is not None and self.status == "running":
            self._activate_zone(monotonic_ms)
        event = self._append_lifecycle_event(
            "recognition_retry",
            {
                "reason": str(reason),
                "message": message,
                "candidate_ids": list(candidates),
            },
            actor=player,
        )
        self._create_incident(
            reason,
            monotonic_ms,
            observations=incident_observations,
        )
        return self._update(event=event, fast_signals=fast)


    @staticmethod
    def _review_candidate(index: int, candidate: ConsensusCandidate) -> ReviewCandidate:
        return ReviewCandidate(
            candidate_id=f"CAND-{index}",
            cards=candidate.cards,
            is_pass=candidate.is_pass,
            votes=candidate.votes,
            confidence=candidate.mean_confidence,
            valid=candidate.valid,
            rejected_reason=candidate.rejected_reason,
        )

    def _create_incident(
        self,
        reason: str,
        monotonic_ms: int,
        *,
        engine_input: dict[str, object] | None = None,
        observations: list[dict[str, object]] | None = None,
    ) -> Path:
        previous = self._recent_incidents.get(str(reason))
        if previous is not None:
            previous_ms, previous_path = previous
            elapsed = int(monotonic_ms) - previous_ms
            if 0 <= elapsed <= 5_000 and previous_path.is_dir():
                self.store.append_incident_occurrence(
                    previous_path,
                    monotonic_ms=int(monotonic_ms),
                    reason=str(reason),
                )
                self._recent_incidents[str(reason)] = (
                    int(monotonic_ms),
                    previous_path,
                )
                return previous_path
        state = self._snapshot_document()
        path = self.store.create_incident(
            reason=str(reason),
            state_before=state,
            state_after=state,
            observations=list(self._observations if observations is None else observations),
            trigger_ms=int(monotonic_ms),
            engine_input=engine_input,
        )
        try:
            self.recorder.schedule_incident_media(
                path,
                trigger_ms=int(monotonic_ms),
            )
        except RuntimeError:
            pass
        self._recent_incidents[str(reason)] = (int(monotonic_ms), path)
        return path

    def _record_recorder_warning(self, warning: RecorderWarning) -> None:
        self._append_lifecycle_event(
            "recording_frame_dropped",
            {"reason": warning.reason, "details": warning.details},
        )
        self._create_incident(
            f"recording_frame_dropped:{warning.reason}", warning.monotonic_ms
        )

    def _snapshot_document(self) -> dict[str, object]:
        snapshot = self.snapshot
        value = snapshot.semantic_dict()
        value.update(
            {
                "session_id": snapshot.session_id,
                "revision": snapshot.revision,
            }
        )
        return value

    _CONTENT_CHANGE_THRESHOLD = 0.02

    def _content_fingerprint(
        self,
        frame: np.ndarray,
        player: Seat,
    ) -> np.ndarray:
        """把该玩家的出牌区域降采样为灰度指纹，用于内容变化判定。"""
        roi = self._play_roi(frame, player)
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        height, width = gray.shape[:2]
        target_width = 32
        target_height = max(8, int(round(32 * height / max(1, width))))
        return cv2.resize(
            gray,
            (target_width, target_height),
            interpolation=cv2.INTER_AREA,
        )

    def _extract_metrics(
        self,
        frame: np.ndarray,
        player: Seat,
        monotonic_ms: int,
        fast: FastSignalResult,
    ) -> ZoneFrameMetrics:
        roi = self._play_roi(frame, player)
        gray = cv2.GaussianBlur(cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY), (5, 5), 0)
        previous = self._previous_by_seat.get(player)
        motion = (
            float(np.mean(cv2.absdiff(gray, previous))) / 255.0
            if previous is not None and previous.shape == gray.shape
            else 0.0
        )
        self._previous_by_seat[player] = gray
        baseline = self._baseline_by_seat.get(player)
        if baseline is None or baseline.shape != gray.shape:
            self._baseline_by_seat[player] = gray.copy()
            occupancy_score = 0.0
        else:
            occupancy_score = float(np.mean(cv2.absdiff(gray, baseline))) / 255.0
        occupied = fast.pass_visible or occupancy_score >= 0.035
        if not occupied and motion <= 0.01:
            self._baseline_by_seat[player] = gray.copy()

        fingerprint = self._content_fingerprint(frame, player)
        previous_fingerprint = self._content_prev_by_seat.get(player)
        content_changed = bool(
            previous_fingerprint is not None
            and previous_fingerprint.shape == fingerprint.shape
            and float(
                np.mean(
                    np.abs(
                        fingerprint.astype(np.int16)
                        - previous_fingerprint.astype(np.int16)
                    )
                )
                / 255.0
            )
            >= self._CONTENT_CHANGE_THRESHOLD
        )
        self._content_prev_by_seat[player] = fingerprint
        return ZoneFrameMetrics(
            monotonic_ms=int(monotonic_ms),
            occupied=occupied,
            motion_score=motion,
            pass_visible=fast.pass_visible,
            effect_visible=fast.effect_visible,
            content_changed=content_changed,
        )


    def _play_roi(self, frame: np.ndarray, player: Seat) -> np.ndarray:
        cropper = getattr(self.recognition_service, "play_roi", None)
        return cropper(frame, player) if callable(cropper) else frame

    def _clear_burst(self) -> None:
        self._samples.clear()
        self._observations.clear()
        self._last_sample_ms = None
        window = self._turn_ownership_window
        if window is not None and not self._is_first_action_turn(window.expected_player):
            window.provisional_samples.clear()

    def _is_first_action_turn(self, player: Seat | None = None) -> bool:
        snapshot = self.snapshot
        return bool(
            self._first_action_pending
            and snapshot.lead_player is not None
            and (player is None or player == snapshot.lead_player)
            and snapshot.current_player == snapshot.lead_player
        )

    def _clear_first_action_candidates(self) -> None:
        self._first_action_samples.clear()

    def _update(
        self,
        *,
        event: LiveEvent | None = None,
        events: tuple[LiveEvent, ...] = (),
        review: ReviewRequest | None = None,
        fast_signals: FastSignalResult | None = None,
    ) -> LiveUpdate:
        if event is not None and not events:
            events = (event,)
        return LiveUpdate(
            status=self.status,
            snapshot=self.snapshot,
            event=event,
            events=events,
            advice=self.latest_advice,
            review=review if review is not None else self.latest_review,
            fast_signals=fast_signals,
        )
