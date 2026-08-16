from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, Protocol, runtime_checkable

from ..danzero.state import GuanDanState, Seat
from ..domain.advice import AdviceResult, StrategyExecutionTrace
from ..domain.recognition import FastSignalResult, OpeningSignal, PlayRegionResult
from ..domain.recording import RecorderWarning, RecordingResult
from ..domain.live import LiveEvent


@runtime_checkable
class RecognitionPort(Protocol):
    def recognize(self, image: Any) -> Any: ...

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
    session_id: str
    directory: Path
    persistence_enabled: bool

    def start(self, manifest: dict[str, object]) -> None: ...
    def append_event(self, event: LiveEvent) -> None: ...
    def append_advice(self, record: dict[str, object]) -> None: ...
    def append_observation(self, record: dict[str, object]) -> None: ...
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


@runtime_checkable
class RecordingPort(Protocol):
    @property
    def frame_count(self) -> int: ...
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


@dataclass(frozen=True)
class LiveSessionConstruction:
    orchestrator: Any
    source: CaptureSourcePort
    initial_update: Any


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
        on_update: Callable[[Any], None] | None = None,
    ) -> LiveSessionConstruction: ...
