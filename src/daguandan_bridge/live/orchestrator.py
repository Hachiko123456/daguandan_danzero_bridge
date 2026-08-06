from __future__ import annotations

import shutil
from dataclasses import dataclass, replace
from datetime import datetime
from functools import wraps
from inspect import signature
from pathlib import Path
from threading import RLock
from typing import Any, Literal

import cv2
import numpy as np

from ..annotation_service import AnnotationService
from ..danzero.advisor import LocalAdvice, StrategyExecutionTrace
from ..danzero.state import GuanDanState, Seat
from ..recognition_service import (
    PLAY_REGION_TO_SEAT,
    FastSignalResult,
    PlayRegionResult,
    ScreenshotRecognitionService,
)
from .consensus import (
    BurstConsensus,
    ConsensusCandidate,
    ConsensusContext,
    ConsensusResult,
    RecognitionSample,
)
from .models import LiveEvent, LiveSnapshot
from .latest_worker import LatestOnlyWorker
from .recorder import RecorderWarning, SessionRecorder
from .reducer import LiveReducer
from .session_store import LiveSessionStore
from .zone_lifecycle import ZoneFrameMetrics, ZoneLifecycle, ZonePhase


LiveStatus = Literal[
    "initializing",
    "running",
    "review_required",
    "paused",
    "finalizing",
    "sealed",
]


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


@dataclass(frozen=True)
class LiveAdvice:
    key: AdviceRequestKey
    status: Literal["requested", "ready", "stale", "failed"]
    advice: LocalAdvice | None = None
    visible: bool = False
    error: str = ""


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


class LiveOrchestrator:
    """Qt-free coordinator for recording, gating, recognition, and reduction."""

    def __init__(
        self,
        *,
        reducer: LiveReducer,
        store: LiveSessionStore,
        recorder: SessionRecorder,
        recognition_service: ScreenshotRecognitionService | Any,
        advisor: Any | None = None,
        settle_ms: int = 400,
        action_timeout_ms: int = 15_000,
        burst_sample_limit: int = 5,
        burst_sample_interval_ms: int = 100,
        minimum_free_bytes: int = 512 * 1024 * 1024,
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
        self._all_events: list[LiveEvent] = []
        self._aux_event_sequence = 0
        self._published_sequence = 0
        self._advice_lock = RLock()
        self._requested_advice: set[AdviceRequestKey] = set()
        self._self_turn_corroborated = False
        self.latest_advice: LiveAdvice | None = None
        self._advice_worker: LatestOnlyWorker | None = None
        self._review_count = 0
        self._advice_requested_at_ms: dict[AdviceRequestKey, int] = {}
        self._advice_visible_latency_ms: int | None = None
        self._accept_advice_results = True
        self._status_before_pause: LiveStatus | None = None
        self._analysis_epoch = 0
        self._next_turn_evidence_started_ms: int | None = None
        self._recent_incidents: dict[str, tuple[int, Path]] = {}
        if advisor is not None:
            self._advice_worker = LatestOnlyWorker(
                self._run_advice,
                on_result=self._complete_advice_job,
            )
            self._advice_worker.start()

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
        lead_player: Seat,
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
        event = self.reducer.confirm_initial_state(
            round_level=round_level,
            hand=hand,
            lead_player=lead_player,
            source="manual_start_with_single_image_recognition",
        )
        event = self._publish_event(event)
        self.status = "running"
        self._activate_zone(int(monotonic_ms), started_with_clear_zone=True)
        self._append_lifecycle_event(
            "turn_started",
            {"player": lead_player},
            actor=lead_player,
        )
        self._request_advice_if_needed()
        return self._update(event=event)

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
        with self._state_lock:
            if self.status == "sealed":
                raise RuntimeError("对局已经结束")
            self._last_monotonic_ms = int(monotonic_ms)
            if self.status != "running":
                return self._update()
            expected = self.reducer.snapshot().current_player
            if expected is None:
                return self._require_review("missing_expected_player", monotonic_ms)
            job_key = self._analysis_job_key()

        # Vision runs without the state lock so pause/correction/finalize stay instant.
        fast = self.recognition_service.recognize_fast_signals(frame, expected)
        with self._state_lock:
            if not self._analysis_job_is_current(job_key):
                return self._update()
            self._apply_fast_signal(fast)
            if self._zone is None or self._zone.expected_player != expected:
                self._activate_zone(monotonic_ms, started_with_clear_zone=False)
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
                    pass_visible=metrics.pass_visible or fast.pass_visible,
                    effect_visible=metrics.effect_visible or fast.effect_visible,
                )
            decision = self._zone.observe(current_metrics)
            if decision.discard_burst:
                self._clear_burst()
            if decision.phase == ZonePhase.REVIEW_REQUIRED:
                return self._require_review(decision.reason, monotonic_ms, fast)
            inferred_pass = self._inferred_pass_if_ready(current_metrics, fast)
            if inferred_pass is not None:
                return self._require_review(
                    "pass_template_missing",
                    monotonic_ms,
                    fast,
                    inferred_pass,
                )
            if not decision.collect_sample or not self._sample_due(monotonic_ms):
                return self._update(fast_signals=fast)
            wild_rank = self.reducer.snapshot().wild_rank

        result = self.recognition_service.recognize_play_region(
            frame,
            expected,
            wild_rank=wild_rank,
        )
        with self._state_lock:
            if not self._analysis_job_is_current(job_key):
                return self._update()
            if self._zone is None or self._zone.expected_player != expected:
                return self._update()
            self._append_sample(result, monotonic_ms)
            consensus = self._decide_if_ready(current_metrics, fast)
            if consensus is not None:
                if consensus.status == "confirmed":
                    event = self._commit_consensus(consensus, monotonic_ms)
                    return self._update(event=event, fast_signals=fast)
                if (
                    consensus.status == "needs_confirmation"
                    or len(self._samples) >= self.burst_sample_limit
                ):
                    return self._require_review(
                        ",".join(consensus.rejected_reasons) or consensus.status,
                        monotonic_ms,
                        fast,
                        consensus,
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
        return self.status == "running" and key == self._analysis_job_key()

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
        event = self._record_action(
            review.player,
            candidate.cards,
            candidate.is_pass,
            confidence=1.0,
            source="manual_one_click_confirmation",
            evidence_refs=review.evidence_refs,
        )
        event = self._publish_event(event)
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
        self._activate_zone(self._last_monotonic_ms, started_with_clear_zone=False)
        self._append_current_turn_started()
        self._request_advice_if_needed()
        return self._update(event=event)

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
        event = self._record_action(
            player,
            cards,
            is_pass,
            confidence=1.0,
            source="manual_minimal_editor",
        )
        event = self._publish_event(event)
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
        self._activate_zone(self._last_monotonic_ms, started_with_clear_zone=False)
        self._append_current_turn_started()
        self._request_advice_if_needed()
        return self._update(event=event)

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
        self._activate_zone(self._last_monotonic_ms, started_with_clear_zone=False)
        self._request_advice_if_needed()
        return self._update(event=event)

    @_state_synchronized
    def pause(self) -> LiveUpdate:
        if self.status in {"running", "review_required"}:
            self._status_before_pause = self.status
            self.status = "paused"
            self._analysis_epoch += 1
            self._clear_burst()
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
            self._activate_zone(monotonic_ms, started_with_clear_zone=False)
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
    ) -> LiveUpdate:
        expected = self.snapshot.current_player or "self"
        fast = FastSignalResult(
            expected_player=expected,
            active_player=active_player,
            pass_visible=False,
            self_action_buttons_visible=bool(self_action_buttons_visible),
            effect_visible=False,
        )
        self._apply_fast_signal(fast)
        return self._update(fast_signals=fast)

    @_state_synchronized
    def start_self_advice(self) -> AdviceRequestKey | None:
        return self._request_advice_if_needed()

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
            return self._update()

    def _activate_zone(self, monotonic_ms: int, *, started_with_clear_zone: bool) -> None:
        player = self.snapshot.current_player
        if player is None:
            self._zone = None
            return
        self._zone = ZoneLifecycle(
            expected_player=player,
            started_with_clear_zone=started_with_clear_zone,
            activated_at_ms=int(monotonic_ms),
            settle_ms=self.settle_ms,
            action_timeout_ms=self.action_timeout_ms,
        )
        self._self_turn_corroborated = False
        self._clear_burst()

    def _request_advice_if_needed(self) -> AdviceRequestKey | None:
        if self.advisor is None or self.status != "running":
            return None
        snapshot = self.snapshot
        if snapshot.current_player != "self":
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
            self._requested_advice.add(key)
            self._advice_requested_at_ms[key] = self._last_monotonic_ms
            self.latest_advice = LiveAdvice(key=key, status="requested")
            self.store.append_advice(
                {
                    "request_id": key.request_id,
                    "status": "requested",
                    "turn_id": key.turn_id,
                    "state_revision": key.state_revision,
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
        trace = StrategyExecutionTrace(job.key.request_id)
        try:
            parameters = signature(self.advisor.recommend).parameters
            kwargs: dict[str, object] = {"request_id": job.key.request_id}
            if "trace" in parameters:
                kwargs["trace"] = trace
            advice = self.advisor.recommend(job.state, **kwargs)
        except Exception as exc:
            trace_snapshot = trace.snapshot()
            engine_input = trace_snapshot.get("engine_input")
            if not isinstance(engine_input, dict):
                engine_input = self._fallback_engine_input(job)
            return _AdviceCompletion(
                job.key,
                error=str(exc),
                engine_input=engine_input,
                trace=trace_snapshot,
            )
        return _AdviceCompletion(job.key, advice=advice)

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
        if not self._accept_advice_results:
            return
        key = completion.key
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
                    )
                return
            if completion.error or completion.advice is None:
                self.latest_advice = LiveAdvice(
                    key=key,
                    status="failed",
                    error=completion.error or "DanZero 未返回建议",
                )
                self.store.append_advice(
                    {
                        "request_id": key.request_id,
                        "status": "failed",
                        "turn_id": key.turn_id,
                        "state_revision": key.state_revision,
                        "error": self.latest_advice.error,
                        "engine_input": completion.engine_input,
                        "trace": completion.trace,
                    }
                )
                self._append_advice_event(
                    "advice_failed",
                    {
                        "request_id": key.request_id,
                        "error": self.latest_advice.error,
                    },
                    confidence=0.0,
                )
                self._create_incident(
                    "advisor_failed",
                    self._last_monotonic_ms,
                    engine_input=completion.engine_input,
                )
                return
            advice = completion.advice
            visible = self._self_turn_corroborated
            if visible:
                self._set_advice_visible_latency(key)
            self.latest_advice = LiveAdvice(
                key=key,
                status="ready",
                advice=advice,
                visible=visible,
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
                },
            )

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
            payload=dict(payload),
            confidence=float(confidence),
            source="live_advice_coordinator",
            state_revision_before=snapshot.revision,
            state_revision_after=snapshot.revision,
        )
        return self._publish_event(event)

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

    def _append_lifecycle_event(
        self,
        event_type: str,
        payload: dict[str, object],
        *,
        actor: Seat | None = None,
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
            confidence=1.0,
            source="live_orchestrator",
            state_revision_before=snapshot.revision,
            state_revision_after=snapshot.revision,
        )
        return self._publish_event(event)

    def _append_current_turn_started(self) -> None:
        player = self.reducer.snapshot().current_player
        if player is not None:
            self._append_lifecycle_event(
                "turn_started",
                {"player": player},
                actor=player,
            )

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
            post_hand=result.post_hand,
        )
        record: dict[str, object] = {
            "id": observation_id,
            "monotonic_ms": int(monotonic_ms),
            "player": result.player,
            "cards": list(result.cards),
            "is_pass": result.is_pass,
            "confidence": result.confidence,
            "source": result.source,
            "post_hand": list(result.post_hand),
            "post_hand_confidence": result.post_hand_confidence,
            "diagnostics": list(result.diagnostics),
            "phase": "burst_read",
        }
        self._samples.append(sample)
        self._observations.append(record)
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
        if len(self._samples) < 3:
            return None
        snapshot = self.snapshot
        player = snapshot.current_player
        assert player is not None
        context = ConsensusContext(
            level_rank=snapshot.wild_rank,
            remaining_cards=snapshot.remaining_cards[player],
            allow_pass=bool(snapshot.trick_plays),
            known_hand=snapshot.my_hand if player == "self" else (),
            table_cards=next(
                (
                    play.cards
                    for play in reversed(snapshot.trick_plays)
                    if not play.is_pass
                ),
                (),
            ),
            region_empty=not metrics.occupied,
            next_turn_evidence=(
                fast.active_player is not None and fast.active_player != player
            ),
        )
        return self.consensus.decide(self._samples, context=context)

    def _inferred_pass_if_ready(
        self,
        metrics: ZoneFrameMetrics,
        fast: FastSignalResult,
    ) -> ConsensusResult | None:
        snapshot = self.reducer.snapshot()
        player = snapshot.current_player
        independent_next_turn = (
            player is not None
            and fast.active_player is not None
            and fast.active_player != player
        )
        eligible = (
            self._zone is not None
            and self._zone.phase == ZonePhase.WAIT_ACTION
            and bool(snapshot.trick_plays)
            and not metrics.occupied
            and not metrics.pass_visible
            and not metrics.effect_visible
            and independent_next_turn
        )
        if not eligible:
            self._next_turn_evidence_started_ms = None
            return None
        now = int(metrics.monotonic_ms)
        if self._next_turn_evidence_started_ms is None:
            self._next_turn_evidence_started_ms = now
            return None
        if now - self._next_turn_evidence_started_ms < 300:
            return None
        context = ConsensusContext(
            level_rank=snapshot.wild_rank,
            remaining_cards=snapshot.remaining_cards[player],
            allow_pass=True,
            known_hand=snapshot.my_hand if player == "self" else (),
            region_empty=True,
            next_turn_evidence=True,
        )
        return self.consensus.decide((), context=context)

    def _commit_consensus(self, result: ConsensusResult, monotonic_ms: int) -> LiveEvent:
        player = self.snapshot.current_player
        assert player is not None
        event = self._record_action(
            player,
            result.cards,
            result.is_pass,
            confidence=result.confidence,
            source=result.source,
            evidence_refs=result.evidence_refs,
        )
        event = self._publish_event(event)
        self._activate_zone(monotonic_ms, started_with_clear_zone=False)
        self._append_current_turn_started()
        self._request_advice_if_needed()
        return event

    def _record_action(
        self,
        player: Seat,
        cards: tuple[str, ...],
        is_pass: bool,
        *,
        confidence: float,
        source: str,
        evidence_refs: tuple[str, ...] = (),
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
        )

    def _require_review(
        self,
        reason: str,
        monotonic_ms: int,
        fast: FastSignalResult | None = None,
        consensus: ConsensusResult | None = None,
    ) -> LiveUpdate:
        player = self.snapshot.current_player
        assert player is not None
        source_candidates = consensus.candidates if consensus is not None else ()
        candidates = tuple(
            self._review_candidate(index, candidate)
            for index, candidate in enumerate(source_candidates, start=1)
        )
        evidence = tuple(
            sample.evidence_ref for sample in self._samples if sample.evidence_ref
        )
        self.latest_review = ReviewRequest(
            reason=str(reason),
            player=player,
            candidates=candidates,
            evidence_refs=evidence,
        )
        self._review_count += 1
        self.status = "review_required"
        if self._zone is not None:
            self._zone.require_review(reason)
        self._append_lifecycle_event(
            "review_required",
            {
                "reason": str(reason),
                "candidate_ids": [item.candidate_id for item in candidates],
                "evidence_refs": list(evidence),
            },
            actor=player,
        )
        self._create_incident(reason, monotonic_ms)
        return self._update(review=self.latest_review, fast_signals=fast)

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
            observations=list(self._observations),
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
        return ZoneFrameMetrics(
            monotonic_ms=int(monotonic_ms),
            occupied=occupied,
            motion_score=motion,
            pass_visible=fast.pass_visible,
            effect_visible=fast.effect_visible,
        )

    def _play_roi(self, frame: np.ndarray, player: Seat) -> np.ndarray:
        annotation_service = getattr(self.recognition_service, "annotation_service", None)
        if annotation_service is None:
            return frame
        region_name = next(
            name for name, seat in PLAY_REGION_TO_SEAT.items() if seat == player
        )
        region = next(
            (item for item in annotation_service.list_regions() if item.name == region_name),
            None,
        )
        if region is None:
            return frame
        box = AnnotationService._box_for_image(region, frame)
        return frame[box.y : box.y + box.h, box.x : box.x + box.w]

    def _clear_burst(self) -> None:
        self._samples.clear()
        self._observations.clear()
        self._last_sample_ms = None
        self._next_turn_evidence_started_ms = None

    def _update(
        self,
        *,
        event: LiveEvent | None = None,
        review: ReviewRequest | None = None,
        fast_signals: FastSignalResult | None = None,
    ) -> LiveUpdate:
        return LiveUpdate(
            status=self.status,
            snapshot=self.snapshot,
            event=event,
            advice=self.latest_advice,
            review=review if review is not None else self.latest_review,
            fast_signals=fast_signals,
        )
