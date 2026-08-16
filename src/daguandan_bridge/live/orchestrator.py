from __future__ import annotations

import shutil
import logging
from collections import Counter, deque
from dataclasses import dataclass, replace
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
from ..danzero.state import GuanDanState, Seat
from .action_uncertainty import state_variants_for_action_semantics
from .consensus import (
    BurstConsensus,
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
from .suit_correction import SuitCorrectionTracker
from .turns import TURN_ORDER
from .zone_lifecycle import ZoneFrameMetrics, ZoneLifecycle, ZonePhase


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
    status: Literal["requested", "ready", "stale", "failed"]
    advice: LocalAdvice | None = None
    visible: bool = False
    error: str = ""
    suit_uncertain: bool = False
    variant_count: int = 1
    advice_agrees_across_variants: bool = True
    semantic_uncertain: bool = False
    semantic_source_history_indices: tuple[int, ...] = ()
    suit_variant_count: int = 1
    suit_equivalence_class_count: int = 1
    semantic_variant_count: int = 1


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
        self._accept_advice_results = True
        self._status_before_pause: LiveStatus | None = None
        self._analysis_epoch = 0
        self._recent_incidents: dict[str, tuple[int, Path]] = {}
        self._lead_wait_started_ms: int | None = None
        self._deal_complete_recorded = False
        self._opening_controls_seen = False
        self._lead_candidate: Seat | None = None
        self._lead_candidate_frames = 0
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
        self._game_end_detected = False
        self._finish_order: list[Seat] = []
        self._placement_streaks: dict[Seat, tuple[str, int]] = {}
        # A left-side read may improve only a previously obscured left action.
        # It is deliberately kept outside the reducer history.  The same
        # result must be observed twice before it becomes a visual correction.
        self._suit_corrected_event_ids: set[str] = set()
        self._suit_correction_tracker = SuitCorrectionTracker()
        if advisor is not None:
            self._advice_worker = LatestOnlyWorker(
                self._run_advice,
                on_result=self._complete_advice_job,
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
        self._game_end_detected = False
        self._finish_order = []
        self._placement_streaks.clear()
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
    ) -> LiveUpdate:
        round_finished = False
        terminal_expected: Seat = "self"
        with self._state_lock:
            if self.status == "sealed":
                raise RuntimeError("对局已经结束")
            self._last_monotonic_ms = int(monotonic_ms)
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
        fast = self._recognize_fast_signals(
            frame,
            expected,
            allow_pass=not first_action,
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
                    pass_visible=(metrics.pass_visible or fast.pass_visible) and not first_action,
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
            had_observations = bool(self._observations)
            if decision.timed_out:
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
                self._clear_burst()
            if not decision.collect_sample or not self._sample_due(monotonic_ms):
                return self._update(fast_signals=fast)
            snapshot = self.reducer.snapshot()
            wild_rank = snapshot.wild_rank
            left_correction_target = self._left_suit_correction_target(snapshot)

        # The expected player remains the sole action-commit path.  While
        # self is deciding what to play, an already submitted left action can
        # become visible again after its action controls disappear.  Read it
        # as a best-effort sidecar only; a sidecar failure must never block
        # the current player's recognition.
        left_correction_result = self._probe_left_suit_correction(
            frame,
            wild_rank=wild_rank,
        ) if left_correction_target is not None else None
        result = self._recognize_play_region(
            frame,
            expected,
            wild_rank=wild_rank,
            allow_pass=not first_action,
        )
        with self._state_lock:
            if not self._analysis_job_is_current(job_key):
                return self._update()
            if self._zone is None or self._zone.expected_player != expected:
                return self._update()
            suit_correction = self._apply_left_suit_correction(
                left_correction_target,
                left_correction_result,
            )
            self._append_sample(result, monotonic_ms)
            consensus = self._decide_if_ready(current_metrics, fast)
            if consensus is not None:
                if consensus.status == "confirmed":
                    event, events = self._commit_consensus(
                        consensus,
                        monotonic_ms,
                        fast=fast,
                    )
                    if suit_correction is not None:
                        events = (suit_correction, *events)
                    return self._update(
                        event=event,
                        events=events,
                        fast_signals=fast,
                    )
                if consensus.status == "needs_confirmation":
                    update = self._require_review(
                        ",".join(consensus.rejected_reasons) or consensus.status,
                        monotonic_ms,
                        fast,
                        consensus,
                    )
                    if suit_correction is not None:
                        return replace(
                            update,
                            events=(suit_correction, *update.events),
                        )
                    return update
            if suit_correction is not None:
                return self._update(
                    event=suit_correction,
                    events=(suit_correction,),
                    fast_signals=fast,
                )
            retry_reason = self._recognition_retry_reason(current_metrics, fast)
            if retry_reason is not None:
                return self._require_review(
                    retry_reason,
                    monotonic_ms,
                    fast,
                )
            if suit_correction is not None:
                return self._update(
                    event=suit_correction,
                    fast_signals=fast,
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
                return self._complete_lead(lead, monotonic_ms)
        else:
            self._lead_candidate = None
            self._lead_candidate_frames = 0
            self._lead_stability_frames.clear()
        started = self._lead_wait_started_ms
        if started is not None and int(monotonic_ms) - started >= self.lead_wait_timeout_ms:
            return self._require_lead_review("lead_player_timeout", monotonic_ms, fast)
        return self._update(fast_signals=fast)

    def _complete_lead(self, lead: Seat, monotonic_ms: int) -> LiveUpdate:
        event = self.reducer.confirm_lead_player(lead)
        event = self._publish_event(event)
        self.status = "running"
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
        try:
            return self.recognition_service.recognize_play_region(
                frame,
                expected,
                wild_rank=wild_rank,
                allow_pass=allow_pass,
                allow_unknown_suit=True,
            )
        except TypeError as exc:
            # Test doubles and external integrations built against the old
            # recognizer may not know the new kwarg.  They remain compatible;
            # the shipped service always preserves rank-only cards.
            if "allow_unknown_suit" not in str(exc):
                if "allow_pass" not in str(exc):
                    raise
            try:
                return self.recognition_service.recognize_play_region(
                    frame,
                    expected,
                    wild_rank=wild_rank,
                    allow_pass=allow_pass,
                )
            except TypeError as fallback_exc:
                if "allow_pass" not in str(fallback_exc):
                    raise
                return self.recognition_service.recognize_play_region(
                    frame,
                    expected,
                    wild_rank=wild_rank,
                )


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
            confidence=1.0,
            source="trusted_log_replay",
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
        event = self._append_lifecycle_event(
            "game_end_detected",
            {"control": control},
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
            return False
        if fast.self_action_buttons_visible:
            self._self_lead_controls_seen = True
            return True
        if not self._self_lead_controls_seen:
            # The analysis worker intentionally drops stale frames.  It can
            # therefore see lead confirmation and then resume only after the
            # local action controls have disappeared.  A timer already on the
            # next seat is positive evidence that self did act; do not wait
            # forever for a controls-visible frame which is no longer queued.
            if fast.active_player not in (None, "self"):
                self._self_lead_controls_cleared = True
                return False
            return True
        self._self_lead_controls_cleared = True
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

    def _left_suit_correction_target(
        self,
        snapshot: LiveSnapshot,
    ) -> LiveEvent | None:
        """Return the one safe visual-only correction target, if any.

        The normal case is the short ``left -> self`` handoff: once the local
        play controls change, an obscured suit on the immediately preceding
        left action can become readable again.  We retain the existing
        post-self-finish probe as a second, display-only recovery window.
        Exact left actions are never re-read and the sidecar result can never
        enter the reducer as a new play or pass.
        """

        if snapshot.current_player == "self" and snapshot.play_history:
            latest_play = snapshot.play_history[-1]
            if (
                latest_play.player == "left"
                and not latest_play.is_pass
                and any(is_unknown_suit_card(card) for card in latest_play.cards)
            ):
                target = self._left_play_event_for_cards(tuple(latest_play.cards))
                if target is not None:
                    return target

        if "self" not in snapshot.finished_seats or snapshot.current_player == "left":
            return None
        latest_left_play = next(
            (
                play
                for play in reversed(snapshot.play_history)
                if play.player == "left" and not play.is_pass
            ),
            None,
        )
        if latest_left_play is None or not any(
            is_unknown_suit_card(card) for card in latest_left_play.cards
        ):
            return None
        return self._left_play_event_for_cards(tuple(latest_left_play.cards))

    def _left_play_event_for_cards(self, cards: tuple[str, ...]) -> LiveEvent | None:
        for event in reversed(self._all_events):
            if (
                event.event_type == "player_played"
                and event.actor == "left"
                and tuple(str(card) for card in event.payload.get("cards", ())) == cards
                and event.event_id not in self._suit_corrected_event_ids
            ):
                return event
        return None

    def _probe_left_suit_correction(
        self,
        frame: np.ndarray,
        *,
        wild_rank: str,
    ) -> PlayRegionResult | None:
        """Run the optional left probe without jeopardizing the formal read."""

        try:
            return self._recognize_play_region(
                frame,
                "left",
                wild_rank=wild_rank,
                allow_pass=False,
            )
        except (cv2.error, OSError, RuntimeError, ValueError):
            return None

    def _apply_left_suit_correction(
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
                "reason": "two_frame_left_sidecar_probe",
            },
            actor="left",
            confidence=result.confidence,
            source="two_frame_left_sidecar_probe",
        )

    def _request_advice_if_needed(self) -> AdviceRequestKey | None:
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
        suit_expansion = state_variants_for_unknown_suits_detailed(
            job.state,
            limit=_MAX_SUIT_STATE_VARIANTS,
        )
        raw_suit_variants = suit_expansion.states
        has_unknown_suit = any(
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
        return published, outcomes

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

        if (
            before.trick_id != after.trick_id
            and before.lead_player in before.finished_seats
            and after.lead_player is not None
            and after.lead_player != before.lead_player
        ):
            outcomes.append(
                self._append_lifecycle_event(
                    "wind_caught",
                    {
                        "from_player": before.lead_player,
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
            completed.append(self._publish_event(event))
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

    def _append_sample(self, result: PlayRegionResult, monotonic_ms: int) -> None:
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
            "hand_card_count_before": len(self.snapshot.my_hand)
            if result.player == "self"
            else None,
            "hand_card_count_after": len(result.post_hand)
            if result.player == "self" and result.post_hand
            else None,
        }
        self._samples.append(sample)
        self._observations.append(record)
        if self._is_first_action_turn(result.player):
            self._first_action_samples.append(sample)
            # This buffer only bridges a short async handoff.  Bound it so a
            # genuinely unresolved opening play cannot bias later retries.
            first_action_limit = max(12, self.burst_sample_limit * 3)
            del self._first_action_samples[:-first_action_limit]
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
        snapshot = self.snapshot
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
        turn_started = self._append_current_turn_started()
        self._request_advice_if_needed()
        events = (event, *outcomes) + ((turn_started,) if turn_started else ())
        return event, events

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
            return self.reducer.record_pass(
                player,
                confidence=confidence,
                source=source,
                evidence_refs=evidence_refs,
            )
        return self.reducer.record_play(
            player,
            cards,
            confidence=confidence,
            source=source,
            evidence_refs=evidence_refs,
            suit_options=suit_options,
            integrity_warnings=integrity_warnings,
            action_metadata=action_metadata,
        )

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
