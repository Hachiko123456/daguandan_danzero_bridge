from __future__ import annotations

import hashlib
import os
from dataclasses import dataclass
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from time import monotonic_ns
from typing import Any, Callable

from ..application.ports import (
    AdvicePort,
    CapturePort,
    LiveSessionConstruction,
    RecognitionPort,
    RecordingPort,
    SessionPersistencePort,
)
from ..advisor_strategy import load_profile_recording_mode
from ..live.orchestrator import LiveOrchestrator
from ..live.recorder import InMemorySessionRecorder, SessionRecorder
from ..live.reducer import LiveReducer
from ..live.session_store import InMemoryLiveSessionStore, LiveSessionStore
from ..runtime_identity import get_runtime_identity


@dataclass
class LiveSessionRecording:
    """Own persistence for one already-confirmed live game."""

    store: SessionPersistencePort
    recorder: RecordingPort
    _closed: bool = False

    @property
    def closed(self) -> bool:
        return self._closed

    def mark_initial_state_confirmed(
        self,
        *,
        round_level: str,
        hand: tuple[str, ...],
    ) -> None:
        if self._closed:
            raise RuntimeError("实时对局记录已经封存")
        self.store.append_recognition_trace(
            {
                "phase": "initial_state_confirmed",
                "round_level": str(round_level),
                "hand_count": len(hand),
            }
        )
        update_metadata = getattr(self.store, "update_session_metadata", None)
        if callable(update_metadata):
            update_metadata(
                {
                    "recording_phase": "live",
                    "initial_state_status": "confirmed",
                }
            )

    def close_start_failed(self) -> None:
        if self._closed:
            return
        recording = self.recorder.close()
        self.store.seal(
            frame_count=recording.frame_count,
            dropped_frames=recording.dropped_frames,
            metrics={"recording_mode": "live_start_failed"},
            incident_media_failures=(
                failure.to_dict() for failure in recording.incident_media_failures
            ),
        )
        self._closed = True


@dataclass
class ListenerRecording:
    """Persist raw listener frames before a complete initial state exists."""

    store: SessionPersistencePort
    recorder: RecordingPort
    _recognition_sample_count: int = 0
    _closed: bool = False

    @property
    def closed(self) -> bool:
        return self._closed

    def record_frame(
        self,
        image: Any,
        *,
        monotonic_ms: int,
        wall_time: str,
    ) -> None:
        if self._closed:
            return
        warning = self.recorder.write_frame(image, monotonic_ms, wall_time)
        if warning is not None:
            self.store.append_recognition_trace(
                {
                    "phase": "recording_frame_dropped",
                    "reason": warning.reason,
                    "monotonic_ms": warning.monotonic_ms,
                    "details": warning.details,
                }
            )

    def record_recognition(self, result: object) -> None:
        if self._closed:
            return
        self._recognition_sample_count += 1
        self.store.append_recognition_trace(
            {
                "phase": "waiting_for_initial_state",
                "round_level": getattr(result, "round_level", None),
                "hand_count": len(tuple(getattr(result, "my_hand", ()) or ())),
                "diagnostics": list(getattr(result, "diagnostics", ()) or ()),
            }
        )

    def close(self, *, reason: str) -> None:
        if self._closed:
            return
        recording = self.recorder.close()
        update_metadata = getattr(self.store, "update_session_metadata", None)
        if callable(update_metadata):
            update_metadata(
                {
                    "recording_phase": "ended_without_initial_state",
                    "initial_state_status": "unconfirmed",
                    "termination_reason": str(reason),
                }
            )
        self.store.seal(
            frame_count=recording.frame_count,
            dropped_frames=recording.dropped_frames,
            metrics={
                "recording_mode": "listening_only",
                "recognition_sample_count": self._recognition_sample_count,
            },
            incident_media_failures=(
                failure.to_dict() for failure in recording.incident_media_failures
            ),
        )
        self._closed = True


class DefaultLiveSessionFactory:
    """Construct one live session from concrete infrastructure adapters."""

    def __init__(
        self,
        capture: CapturePort,
        recognizer: RecognitionPort,
        advisor: AdvicePort | None,
        *,
        profile_name: str,
    ) -> None:
        self.capture = capture
        self.recognizer = recognizer
        self.advisor = advisor
        self.profile_name = profile_name

    def with_advisor(self, advisor: AdvicePort) -> "DefaultLiveSessionFactory":
        return DefaultLiveSessionFactory(
            self.capture,
            self.recognizer,
            advisor,
            profile_name=self.profile_name,
        )

    def start_session(
        self,
        *,
        round_level: str,
        hand: tuple[str, ...],
        lead_player: str | None,
        recognition_strategy: str,
        on_update: Callable[[Any], None] | None = None,
    ) -> LiveSessionConstruction:
        recording = self._create_recording(recognition_strategy)
        assert recording is not None
        return self._start_session_with_recording(
            recording,
            round_level=round_level,
            hand=hand,
            lead_player=lead_player,
            recognition_strategy=recognition_strategy,
            on_update=on_update,
        )

    def start_listener_recording(
        self,
        *,
        recognition_strategy: str,
    ) -> ListenerRecording | None:
        """Start a replay that survives even when the deal is never recognized."""

        if (
            load_profile_recording_mode(
                self.capture.profiles_root,
                self.profile_name,
            )
            != "all"
        ):
            return None
        loaded = self.capture.load_profile(self.profile_name)
        store = LiveSessionStore(
            self.capture.profiles_root,
            self.profile_name,
        )
        manifest = build_session_manifest(
            loaded.paths.profile_config_path,
            loaded.paths.templates_config_path,
        )
        manifest.update(
            {
                "recognition_strategy": recognition_strategy,
                "recording_phase": "listening",
                "initial_state_status": "unconfirmed",
                "recording_mode": "all",
                "advisor": self._advisor_manifest(),
            }
        )
        store.start(manifest)
        try:
            recorder = SessionRecorder(
                store.directory,
                size=loaded.config.base_size,
                fps=10,
            )
        except Exception:
            store.seal(frame_count=0, dropped_frames=0)
            raise
        return ListenerRecording(store=store, recorder=recorder)

    def _start_session_with_recording(
        self,
        recording: LiveSessionRecording,
        *,
        round_level: str,
        hand: tuple[str, ...],
        lead_player: str | None,
        recognition_strategy: str,
        on_update: Callable[[Any], None] | None,
    ) -> LiveSessionConstruction:
        source = None
        try:
            orchestrator = LiveOrchestrator(
                reducer=LiveReducer(recording.store.session_id),
                store=recording.store,
                recorder=recording.recorder,
                recognition_service=self.recognizer,
                advisor=self.advisor,
                recognition_strategy=recognition_strategy,
                on_update=on_update,
            )
            source = self.capture.open_live_source(self.profile_name)
            update = orchestrator.start(
                round_level=round_level,
                hand=hand,
                lead_player=lead_player,
                monotonic_ms=monotonic_ns() // 1_000_000,
            )
            recording.mark_initial_state_confirmed(
                round_level=round_level,
                hand=hand,
            )
            return LiveSessionConstruction(orchestrator, source, update)
        except Exception:
            if source is not None:
                source.close()
            recording.close_start_failed()
            raise

    def _create_recording(
        self,
        recognition_strategy: str,
    ) -> LiveSessionRecording:
        loaded = self.capture.load_profile(self.profile_name)
        recording_mode = load_profile_recording_mode(
            self.capture.profiles_root,
            self.profile_name,
        )
        if recording_mode == "none":
            store = InMemoryLiveSessionStore(
                self.capture.profiles_root,
                self.profile_name,
            )
            store.start({})
            return LiveSessionRecording(
                store=store,
                recorder=InMemorySessionRecorder(store.directory),
            )

        store = LiveSessionStore(
            self.capture.profiles_root,
            self.profile_name,
        )
        manifest = build_session_manifest(
            loaded.paths.profile_config_path,
            loaded.paths.templates_config_path,
        )
        manifest.update(
            {
                "recognition_strategy": recognition_strategy,
                "recording_phase": "live",
                "initial_state_status": "pending",
                "recording_mode": recording_mode,
                "advisor": self._advisor_manifest(),
            }
        )
        store.start(manifest)
        try:
            recorder = SessionRecorder(
                store.directory,
                size=loaded.config.base_size,
                fps=10,
            )
        except Exception:
            store.seal(frame_count=0, dropped_frames=0)
            raise
        return LiveSessionRecording(
            store=store,
            recorder=recorder,
        )

    def _advisor_manifest(self) -> dict[str, object]:
        audit_info = getattr(self.advisor, "audit_info", None)
        if callable(audit_info):
            return dict(audit_info())
        return {
            "backend": type(self.advisor).__name__ if self.advisor else "none",
            "standard_no_tribute": True,
        }


def build_session_manifest(config_path: Path, templates_path: Path) -> dict[str, object]:
    try:
        application_version = version("daguandan-danzero-bridge")
    except PackageNotFoundError:
        application_version = "0.1.0"
    return {
        "application_version": application_version,
        "owner_pid": os.getpid(),
        "configuration_hash": _file_hash(config_path),
        "template_manifest_hash": _file_hash(templates_path),
        "runtime_identity": get_runtime_identity(),
        "target_fps": 10,
        "codec": "MJPG",
    }


def _file_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes() if path.is_file() else b"").hexdigest()
