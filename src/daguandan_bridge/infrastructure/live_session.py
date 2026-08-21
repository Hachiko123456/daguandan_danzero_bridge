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
from ..advisor_strategy import load_profile_session_data_recording_enabled
from ..live.orchestrator import LiveOrchestrator
from ..live.recorder import InMemorySessionRecorder, SessionRecorder
from ..live.reducer import LiveReducer
from ..live.session_store import InMemoryLiveSessionStore, LiveSessionStore


@dataclass
class ListeningSessionRecording:
    """Persist capture evidence before the live game state is trustworthy."""

    store: SessionPersistencePort
    recorder: RecordingPort
    listening_started: bool = True
    _closed: bool = False
    _recognition_count: int = 0

    @property
    def closed(self) -> bool:
        return self._closed

    def record_frame(
        self,
        frame: Any,
        *,
        monotonic_ms: int,
        wall_time: str,
    ) -> None:
        if self._closed:
            return
        warning = self.recorder.write_frame(frame, monotonic_ms, wall_time)
        if warning is not None:
            self.store.append_recognition_trace(
                {
                    "phase": "waiting_for_initial_state",
                    "kind": "recorder_warning",
                    "monotonic_ms": int(monotonic_ms),
                    "reason": warning.reason,
                    "details": warning.details,
                }
            )

    def record_recognition(
        self,
        result: object,
        *,
        captured_at: str,
        acceptance_reason: str | None = None,
    ) -> None:
        if self._closed:
            return
        self._recognition_count += 1
        events = tuple(getattr(result, "events", ()) or ())
        round_level = str(getattr(result, "round_level", "") or "")
        wild_rank = str(getattr(result, "wild_rank", "") or "")
        current_player = str(getattr(result, "current_player", "") or "")
        lead_player = str(getattr(result, "lead_player", "") or "")
        hand = [str(card) for card in getattr(result, "my_hand", ())]
        self.store.append_recognition_trace(
            {
                "phase": "waiting_for_initial_state",
                "kind": "initial_recognition",
                "sequence": self._recognition_count,
                "captured_at": str(captured_at),
                "round_level": round_level or None,
                "wild_rank": wild_rank or None,
                "current_player": current_player or None,
                "lead_player": lead_player or None,
                "my_hand": hand,
                "initial_state_summary": {
                    "recognized_round_level": round_level or "unrecognized",
                    "recognized_wild_rank": wild_rank or "unrecognized",
                    "hand_count": len(hand),
                    "current_player": current_player or "unrecognized",
                    "lead_player": lead_player or "unrecognized",
                    "acceptance_reason": acceptance_reason or "unclassified",
                },
                "field_confidences": dict(
                    getattr(result, "field_confidences", {}) or {}
                ),
                "diagnostics": [str(item) for item in getattr(result, "diagnostics", ())],
                "unresolved_fields": [
                    str(item) for item in getattr(result, "unresolved_fields", ())
                ],
                "events": [
                    {
                        "player": str(getattr(event, "player", "")),
                        "cards": [str(card) for card in getattr(event, "cards", ())],
                        "is_pass": bool(getattr(event, "is_pass", False)),
                        "confidence": float(getattr(event, "confidence", 0.0)),
                    }
                    for event in events
                ],
                "initial_state_acceptance": acceptance_reason,
            }
        )

    def mark_initial_state_confirmed(
        self,
        *,
        round_level: str,
        hand: tuple[str, ...],
    ) -> None:
        if self._closed:
            raise RuntimeError("监听期录制已经封存")
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

    def close_unconfirmed(self, reason: str) -> None:
        if self._closed:
            return
        self.store.append_recognition_trace(
            {
                "phase": "initial_state_unconfirmed",
                "reason": str(reason),
                "recognition_sample_count": self._recognition_count,
            }
        )
        update_metadata = getattr(self.store, "update_session_metadata", None)
        if callable(update_metadata):
            update_metadata(
                {
                    "recording_phase": "ended_without_initial_state",
                    "initial_state_status": "unconfirmed",
                    "termination_reason": str(reason),
                }
            )
        recording = self.recorder.close()
        self.store.seal(
            frame_count=recording.frame_count,
            dropped_frames=recording.dropped_frames,
            metrics={
                "recording_mode": "listening_only",
                "recognition_sample_count": self._recognition_count,
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
        recording = self._create_recording(
            recognition_strategy,
            listening_started=False,
        )
        assert recording is not None
        return self._start_session_with_recording(
            recording,
            round_level=round_level,
            hand=hand,
            lead_player=lead_player,
            recognition_strategy=recognition_strategy,
            on_update=on_update,
        )

    def begin_listening_recording(
        self,
        *,
        recognition_strategy: str,
    ) -> ListeningSessionRecording | None:
        """Open durable capture storage as soon as the listener is enabled."""

        recording = self._create_recording(
            recognition_strategy,
            require_persistence=True,
            listening_started=True,
        )
        if recording is None:
            return None
        recording.store.append_recognition_trace(
            {
                "phase": "listening_started",
                "recognition_strategy": recognition_strategy,
            }
        )
        return recording

    def start_session_from_listening_recording(
        self,
        recording: ListeningSessionRecording,
        *,
        round_level: str,
        hand: tuple[str, ...],
        lead_player: str | None,
        recognition_strategy: str,
        on_update: Callable[[Any], None] | None = None,
    ) -> LiveSessionConstruction:
        """Promote one waiting recording into the real-time game session."""

        if recording.closed:
            raise RuntimeError("监听期录制已经封存，无法启动实时对局")
        return self._start_session_with_recording(
            recording,
            round_level=round_level,
            hand=hand,
            lead_player=lead_player,
            recognition_strategy=recognition_strategy,
            on_update=on_update,
        )

    def _start_session_with_recording(
        self,
        recording: ListeningSessionRecording,
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
            recording.close_unconfirmed("live_session_start_failed")
            raise

    def _create_recording(
        self,
        recognition_strategy: str,
        *,
        require_persistence: bool = False,
        listening_started: bool,
    ) -> ListeningSessionRecording | None:
        loaded = self.capture.load_profile(self.profile_name)
        save_session_data = load_profile_session_data_recording_enabled(
            self.capture.profiles_root,
            self.profile_name,
        )
        if not save_session_data:
            if require_persistence:
                return None
            store = InMemoryLiveSessionStore(
                self.capture.profiles_root,
                self.profile_name,
            )
            store.start({})
            return ListeningSessionRecording(
                store=store,
                recorder=InMemorySessionRecorder(store.directory),
                listening_started=listening_started,
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
                "recording_phase": (
                    "waiting_for_initial_state" if listening_started else "live"
                ),
                "initial_state_status": "pending",
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
        return ListeningSessionRecording(
            store=store,
            recorder=recorder,
            listening_started=listening_started,
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
        "target_fps": 10,
        "codec": "MJPG",
    }


def _file_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes() if path.is_file() else b"").hexdigest()
