from __future__ import annotations

import hashlib
import json
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
from ..advisor_strategy import (
    advisor_strategy_id,
    load_profile_advisor_strategy,
    load_profile_automatic_log_include_media,
    load_profile_recording_max_total_bytes,
    load_profile_recording_mode,
    recording_media_usage_bytes,
)
from ..live.recorder import InMemorySessionRecorder
from ..live.session_store import LiveSessionStore
from ..runtime_identity import get_runtime_identity
from .live_v2_composition import build_production_live_v2_runtime
from .process_session_recorder import ProcessSessionRecorder


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
        update_metadata = getattr(self.store, "update_session_metadata", None)
        if callable(update_metadata):
            update_metadata({"recording_integrity": recording.integrity})
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
    ) -> object | None:
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
        return warning

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
                    "recording_integrity": recording.integrity,
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
        self.advisor_strategy = (
            advisor_strategy_id(advisor)
            if advisor is not None
            else load_profile_advisor_strategy(capture.profiles_root, profile_name)
        )

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
            automatic_log_delivery_enabled=True,
            automatic_log_include_media=load_profile_automatic_log_include_media(
                self.capture.profiles_root, self.profile_name
            ),
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
            recorder = ProcessSessionRecorder(
                store.directory,
                size=loaded.config.base_size,
                fps=10,
                max_video_bytes=self._remaining_video_allowance(loaded.paths.profile_config_path),
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
            orchestrator = build_production_live_v2_runtime(
                store=recording.store,
                recorder=recording.recorder,
                recognizer=self.recognizer,
                profiles_root=self.capture.profiles_root,
                profile_name=self.profile_name,
                advisor_backend=self.advisor_strategy,
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
        store = LiveSessionStore(
            self.capture.profiles_root,
            self.profile_name,
            automatic_log_delivery_enabled=True,
            automatic_log_include_media=load_profile_automatic_log_include_media(
                self.capture.profiles_root, self.profile_name
            ),
        )
        manifest = build_session_manifest(
            loaded.paths.profile_config_path,
            loaded.paths.templates_config_path,
        )
        manifest.update(
            {
                "recognition_strategy": recognition_strategy,
                "schema": "guandan.live-v2.session/1",
                "runtime": "live_v2",
                "recording_phase": "live",
                "initial_state_status": "pending",
                "recording_mode": recording_mode,
                "advisor": self._advisor_manifest(),
            }
        )
        store.start(manifest)
        if recording_mode == "none":
            return LiveSessionRecording(
                store=store,
                recorder=InMemorySessionRecorder(store.directory),
            )
        try:
            recorder = ProcessSessionRecorder(
                store.directory,
                size=loaded.config.base_size,
                fps=10,
                max_video_bytes=self._remaining_video_allowance(loaded.paths.profile_config_path),
            )
        except Exception:
            store.seal(frame_count=0, dropped_frames=0)
            raise
        return LiveSessionRecording(
            store=store,
            recorder=recorder,
        )

    def _remaining_video_allowance(self, profile_config_path: Path) -> int:
        """Budget video across this profile, without deleting any user recording.

        Scan only on recorder creation, never on the capture hot path. Reparse
        points are not followed. Existing generations/exports remain untouched.
        """
        config = json.loads(Path(profile_config_path).read_text(encoding="utf-8"))
        limit = config.get(
            "recording_max_total_bytes",
            load_profile_recording_max_total_bytes(
                self.capture.profiles_root, self.profile_name
            ),
        )
        if isinstance(limit, bool) or not isinstance(limit, int) or limit < 0:
            raise ValueError("recording_max_total_bytes must be a non-negative integer")
        used = recording_media_usage_bytes(
            self.capture.profiles_root, self.profile_name
        )
        return max(0, limit - used)

    def _advisor_manifest(self) -> dict[str, object]:
        audit_info = getattr(self.advisor, "audit_info", None)
        if callable(audit_info):
            result = dict(audit_info())
            result["strategy_id"] = self.advisor_strategy
            result["advisor_backend"] = self.advisor_strategy
            return result
        return {
            "backend": self.advisor_strategy,
            "strategy_id": self.advisor_strategy,
            "advisor_backend": self.advisor_strategy,
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
