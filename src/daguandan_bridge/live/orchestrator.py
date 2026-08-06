from __future__ import annotations

import shutil
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import cv2
import numpy as np

from ..annotation_service import AnnotationService
from ..danzero.state import Seat
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
from .recorder import RecorderWarning, SessionRecorder
from .reducer import LiveReducer
from .session_store import LiveSessionStore
from .zone_lifecycle import ZoneFrameMetrics, ZoneLifecycle, ZonePhase


LiveStatus = Literal[
    "initializing",
    "running",
    "review_required",
    "paused",
    "sealed",
]


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


class LiveOrchestrator:
    """Qt-free coordinator for recording, gating, recognition, and reduction."""

    def __init__(
        self,
        *,
        reducer: LiveReducer,
        store: LiveSessionStore,
        recorder: SessionRecorder,
        recognition_service: ScreenshotRecognitionService | Any,
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
        self.settle_ms = int(settle_ms)
        self.action_timeout_ms = int(action_timeout_ms)
        self.burst_sample_limit = int(burst_sample_limit)
        self.burst_sample_interval_ms = int(burst_sample_interval_ms)
        self.minimum_free_bytes = int(minimum_free_bytes)
        self.consensus = BurstConsensus(min_votes=3)
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

    @property
    def snapshot(self) -> LiveSnapshot:
        return self.reducer.snapshot()

    @property
    def events(self) -> tuple[LiveEvent, ...]:
        return self.reducer.events

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
        event = self.reducer.confirm_initial_state(
            round_level=round_level,
            hand=hand,
            lead_player=lead_player,
            source="manual_start_with_single_image_recognition",
        )
        self.store.append_event(event)
        self.status = "running"
        self._last_monotonic_ms = int(monotonic_ms)
        self._activate_zone(int(monotonic_ms), started_with_clear_zone=True)
        return self._update(event=event)

    def ingest_frame(
        self,
        frame: np.ndarray,
        *,
        monotonic_ms: int,
        wall_time: str,
        metrics: ZoneFrameMetrics | None = None,
    ) -> LiveUpdate:
        if self.status == "sealed":
            raise RuntimeError("对局已经结束")
        self._last_monotonic_ms = int(monotonic_ms)
        warning = self.recorder.write_frame(frame, monotonic_ms, wall_time)
        if warning is not None:
            self._record_recorder_warning(warning)
        if self.status in {"paused", "initializing"}:
            return self._update()

        expected = self.snapshot.current_player
        if expected is None:
            return self._require_review("missing_expected_player", monotonic_ms)
        fast = self.recognition_service.recognize_fast_signals(frame, expected)
        if self.status == "review_required":
            return self._update(fast_signals=fast)
        if self._zone is None or self._zone.expected_player != expected:
            self._activate_zone(monotonic_ms, started_with_clear_zone=False)
        assert self._zone is not None
        if metrics is None:
            metrics = self._extract_metrics(frame, expected, monotonic_ms, fast)
        else:
            metrics = ZoneFrameMetrics(
                monotonic_ms=int(monotonic_ms),
                occupied=metrics.occupied,
                motion_score=metrics.motion_score,
                pass_visible=metrics.pass_visible or fast.pass_visible,
                effect_visible=metrics.effect_visible or fast.effect_visible,
            )
        decision = self._zone.observe(metrics)
        if decision.discard_burst:
            self._clear_burst()
        if decision.phase == ZonePhase.REVIEW_REQUIRED:
            return self._require_review(decision.reason, monotonic_ms, fast)
        if decision.collect_sample and self._sample_due(monotonic_ms):
            result = self.recognition_service.recognize_play_region(
                frame,
                expected,
                wild_rank=self.snapshot.wild_rank,
            )
            self._append_sample(result, monotonic_ms)
            consensus = self._decide_if_ready(metrics, fast)
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
        event = self._record_action(
            review.player,
            candidate.cards,
            candidate.is_pass,
            confidence=1.0,
            source="manual_one_click_confirmation",
            evidence_refs=review.evidence_refs,
        )
        self.store.append_event(event)
        self.status = "running"
        self.latest_review = None
        self._clear_burst()
        self._activate_zone(self._last_monotonic_ms, started_with_clear_zone=False)
        return self._update(event=event)

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
        self.store.append_event(event)
        self.status = "running"
        self.latest_review = None
        self._clear_burst()
        self._activate_zone(self._last_monotonic_ms, started_with_clear_zone=False)
        return self._update(event=event)

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
        self.store.append_event(event)
        self.status = "running"
        self.latest_review = None
        self._clear_burst()
        self._activate_zone(self._last_monotonic_ms, started_with_clear_zone=False)
        return self._update(event=event)

    def pause(self) -> LiveUpdate:
        if self.status == "running":
            self.status = "paused"
            self._clear_burst()
            self._zone = None
        return self._update()

    def resume(self, *, monotonic_ms: int) -> LiveUpdate:
        if self.status != "paused":
            raise RuntimeError("只有暂停状态可以继续")
        self.status = "running"
        self._activate_zone(monotonic_ms, started_with_clear_zone=False)
        return self._update()

    def capture_interrupted(self, reason: str, *, monotonic_ms: int) -> LiveUpdate:
        self._create_incident("capture_interrupted:" + str(reason), monotonic_ms)
        return self.pause()

    def finish(self) -> LiveUpdate:
        if self.status == "sealed":
            return self._update()
        recording = self.recorder.close()
        self.store.seal(
            frame_count=recording.frame_count,
            dropped_frames=recording.dropped_frames,
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
        self._clear_burst()

    def _append_sample(self, result: PlayRegionResult, monotonic_ms: int) -> None:
        self._observation_sequence += 1
        observation_id = f"OBS-{self._observation_sequence:06d}"
        sample = RecognitionSample(
            cards=result.cards,
            is_pass=result.is_pass,
            confidence=result.confidence,
            source=result.source,
            evidence_ref=observation_id,
        )
        record: dict[str, object] = {
            "id": observation_id,
            "monotonic_ms": int(monotonic_ms),
            "player": result.player,
            "cards": list(result.cards),
            "is_pass": result.is_pass,
            "confidence": result.confidence,
            "source": result.source,
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
            region_empty=not metrics.occupied,
            next_turn_evidence=(
                fast.active_player is not None and fast.active_player != player
            ),
        )
        return self.consensus.decide(self._samples, context=context)

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
        self.store.append_event(event)
        self._activate_zone(monotonic_ms, started_with_clear_zone=False)
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
        self.status = "review_required"
        if self._zone is not None:
            self._zone.require_review(reason)
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

    def _create_incident(self, reason: str, monotonic_ms: int) -> Path:
        state = self._snapshot_document()
        path = self.store.create_incident(
            reason=str(reason),
            state_before=state,
            state_after=state,
            observations=list(self._observations),
        )
        try:
            self.recorder.save_incident_media(path, trigger_ms=int(monotonic_ms))
        except RuntimeError:
            pass
        return path

    def _record_recorder_warning(self, warning: RecorderWarning) -> None:
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
            review=review if review is not None else self.latest_review,
            fast_signals=fast_signals,
        )
