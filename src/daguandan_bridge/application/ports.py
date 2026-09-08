from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, Protocol, runtime_checkable

from ..danzero.state import GuanDanState, Seat
from ..domain.advice import AdviceResult, StrategyExecutionTrace
from ..domain.live_runtime import LiveAdvice, LiveStatus, LiveUpdate
from ..domain.recognition import FastSignalResult, OpeningSignal, PlayRegionResult
from ..domain.recording import RecorderWarning, RecordingResult
from ..domain.live import LiveEvent, LiveSnapshot


@runtime_checkable
class RecognitionPort(Protocol):
    def recognize(self, image: Any) -> Any: ...

    def recognize_table_anchor(self, image: Any) -> float: ...

    def recognize_play_region(
        self,
        image: Any,
        seat: Seat,
        *,
        wild_rank: str,
        allow_unknown_suit: bool = False,
    ) -> PlayRegionResult | None: ...

    def recognize_fast_signals(
        self,
        image: Any,
        expected_player: Seat,
    ) -> FastSignalResult: ...

    def recognize_opening_signal(self, image: Any) -> OpeningSignal: ...
    def play_roi(self, image: Any, seat: Seat) -> Any: ...


@runtime_checkable
class AdvicePort(Protocol):
    def initialize(self) -> None: ...

    def recommend(
        self,
        state: GuanDanState,
        *,
        request_id: str = "",
        trace: StrategyExecutionTrace | None = None,
    ) -> AdviceResult: ...


@runtime_checkable
class SessionPersistencePort(Protocol):
    @property
    def is_started(self) -> bool: ...
    session_id: str
    directory: Path
    persistence_enabled: bool

    def start(self, manifest: dict[str, object]) -> None: ...
    def append_event(self, event: LiveEvent) -> None: ...
    def append_event_batch(self, events: Iterable[LiveEvent]) -> None: ...
    def append_advice(self, record: dict[str, object]) -> None: ...
    def append_observation(self, record: dict[str, object]) -> None: ...
    def append_recognition_trace(self, record: dict[str, object]) -> None: ...
    def update_runtime_identity(self, identity: dict[str, object]) -> None: ...
    def upsert_decision(self, record: dict[str, object]) -> None: ...
    def create_incident(
        self,
        *,
        reason: str,
        state_before: dict[str, object],
        state_after: dict[str, object],
        observations: list[dict[str, object]],
        frame_paths: Iterable[Path] = (),
        engine_input: dict[str, object] | None = None,
        trigger_ms: int | None = None,
    ) -> Path: ...
    def append_incident_occurrence(
        self,
        incident_directory: Path,
        *,
        monotonic_ms: int,
        reason: str,
    ) -> None: ...
    def seal(
        self,
        *,
        frame_count: int,
        dropped_frames: int,
        metrics: dict[str, object] | None = None,
        incident_media_failures: Iterable[dict[str, object]] = (),
    ) -> None: ...
    def append_post_seal_health_audit(
        self,
        report: dict[str, object],
        *,
        state: dict[str, object],
        monotonic_ms: int,
    ) -> None: ...
    def record_automatic_log_delivery(self, result: dict[str, object]) -> None: ...


@runtime_checkable
class RecordingPort(Protocol):
    @property
    def frame_count(self) -> int: ...
    @property
    def dropped_frames(self) -> int: ...
    def write_frame(
        self,
        frame: Any,
        monotonic_ms: int,
        wall_time: str,
    ) -> RecorderWarning | None: ...
    def save_evidence_frame(self, path: Path, frame: Any) -> Path: ...
    def schedule_incident_media(
        self,
        directory: Path,
        *,
        trigger_ms: int,
        before_ms: int = 5_000,
        after_ms: int = 5_000,
    ) -> None: ...
    def close(self) -> RecordingResult: ...


@runtime_checkable
class CaptureSourcePort(Protocol):
    def capture(self) -> Any: ...
    def close(self) -> None: ...


@runtime_checkable
class CapturePort(Protocol):
    profiles_root: Path
    def load_profile(self, name: str) -> Any: ...
    def capture_frame(self, profile_name: str) -> Any: ...
    def open_live_source(self, profile_name: str) -> CaptureSourcePort: ...
    def target_client_rect(self, profile_name: str) -> Any: ...
    def lock_target_client_size(self, profile_name: str) -> Any: ...


@runtime_checkable
class LiveRuntimePort(Protocol):
    """Narrow application contract implemented by the production live core."""

    status: LiveStatus
    store: SessionPersistencePort
    recorder: RecordingPort
    recognition_service: RecognitionPort
    latest_advice: LiveAdvice | None
    automatic_log_delivery_result: dict[str, object] | None

    @property
    def snapshot(self) -> LiveSnapshot: ...

    @property
    def needs_first_action_frames(self) -> bool: ...

    def start(
        self,
        *,
        round_level: str,
        hand: tuple[str, ...],
        lead_player: Seat | None,
        monotonic_ms: int,
        wall_time: str | None = None,
        historical_scan: bool = False,
    ) -> LiveUpdate: ...

    def bind_capture_generation(self, generation: int) -> LiveUpdate: ...

    def analyze_frame(
        self,
        frame: Any,
        *,
        monotonic_ms: int,
        metrics: Any | None = None,
        trace_context: dict[str, object] | None = None,
    ) -> LiveUpdate: ...

    def record_frame(
        self,
        frame: Any,
        *,
        monotonic_ms: int,
        wall_time: str,
    ) -> RecorderWarning | None: ...

    def preview_controls(
        self,
        fast: FastSignalResult,
        *,
        captured_ms: int,
        capture_generation: int,
        frame_size: tuple[int, int],
    ) -> LiveUpdate | None: ...

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
    ) -> LiveUpdate: ...

    def bootstrap_opening_action(
        self,
        *,
        actor: Seat,
        cards: tuple[str, ...],
        expected_next_player: Seat,
        monotonic_ms: int,
        confidence: float,
        source: str,
    ) -> LiveUpdate: ...

    def confirm_candidate(self, candidate_id: str) -> LiveUpdate: ...

    def confirm_manual_action(
        self,
        *,
        cards: tuple[str, ...] = (),
        is_pass: bool,
        action_metadata: dict[str, object] | None = None,
    ) -> LiveUpdate: ...

    def correct_latest(
        self,
        *,
        cards: tuple[str, ...] = (),
        is_pass: bool,
        reason: str = "one_click_correction",
        action_metadata: dict[str, object] | None = None,
    ) -> LiveUpdate: ...

    def confirm_lead_player(self, lead_player: Seat) -> LiveUpdate: ...

    def pause(self) -> LiveUpdate: ...

    def resume(self, *, monotonic_ms: int) -> LiveUpdate: ...

    def begin_finalizing(self) -> LiveUpdate: ...

    def capture_interrupted(self, reason: str, *, monotonic_ms: int) -> LiveUpdate: ...

    def analysis_failed(self, reason: str, *, monotonic_ms: int) -> LiveUpdate: ...

    def poll_deadlines(self) -> LiveUpdate: ...

    def wait_for_advice_idle(self, *, timeout: float = 60.0) -> bool: ...

    def finish(self) -> LiveUpdate: ...


@dataclass(frozen=True)
class LiveSessionConstruction:
    orchestrator: LiveRuntimePort
    source: CaptureSourcePort
    initial_update: LiveUpdate


@runtime_checkable
class SessionFactoryPort(Protocol):
    def with_advisor(self, advisor: AdvicePort) -> "SessionFactoryPort": ...

    def start_session(
        self,
        *,
        round_level: str,
        hand: tuple[str, ...],
        lead_player: str | None,
        recognition_strategy: str,
        on_update: Callable[[LiveUpdate], None] | None = None,
    ) -> LiveSessionConstruction: ...

    def start_listener_recording(
        self,
        *,
        recognition_strategy: str,
    ) -> Any | None: ...
