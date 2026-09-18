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
from ..session_paths import resolve_sessions_root
from ..storage import atomic_write_json
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
    _confirmed_callback: Callable[["ListenerRecording"], None] | None = None
    _close_summary: dict[str, object] | None = None
    _archived_directory: Path | None = None

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
        confirmed = str(reason) == "initial_state_confirmed"
        update_metadata = getattr(self.store, "update_session_metadata", None)
        if callable(update_metadata):
            update_metadata(
                {
                    "recording_phase": (
                        "opening_confirmed"
                        if confirmed
                        else "ended_without_initial_state"
                    ),
                    "initial_state_status": (
                        "confirmed" if confirmed else "unconfirmed"
                    ),
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
        self._close_summary = {
            "opening_id": str(getattr(self.store, "session_id", "")),
            "frame_count": int(recording.frame_count),
            "dropped_frames": int(recording.dropped_frames),
            "recognition_sample_count": int(self._recognition_sample_count),
            "recording_integrity": recording.integrity,
            "termination_reason": str(reason),
        }
        self._closed = True
        if confirmed and self._confirmed_callback is not None:
            self._confirmed_callback(self)

    def archive_into(self, live_store: SessionPersistencePort) -> Path:
        """Atomically attach sealed opening evidence below one formal game."""

        if not self._closed or self._close_summary is None:
            raise RuntimeError("开局监听证据尚未封存")
        if self._close_summary.get("termination_reason") != "initial_state_confirmed":
            raise RuntimeError("未确认的开局证据不能归档到正式对局")
        source = Path(getattr(self.store, "directory"))
        live_directory = Path(getattr(live_store, "directory"))
        if source.parent.name != ".preopening":
            raise RuntimeError("开局证据不在受管预开局目录")
        if not source.is_dir() or not live_directory.is_dir():
            raise RuntimeError("开局证据或正式对局目录不存在")
        destination = live_directory / "opening"
        if destination.exists():
            raise FileExistsError(f"正式对局已包含开局证据：{destination}")
        receipt = {
            "schema": "guandan.opening-archive/1",
            "status": "archived",
            "opening_id": self._close_summary["opening_id"],
            "formal_session_id": str(getattr(live_store, "session_id", "")),
            "relative_path": "opening",
            "frame_count": self._close_summary["frame_count"],
            "dropped_frames": self._close_summary["dropped_frames"],
            "recognition_sample_count": self._close_summary[
                "recognition_sample_count"
            ],
            "recording_integrity": self._close_summary["recording_integrity"],
        }
        atomic_write_json(source / "archive_receipt.json", receipt)
        source.replace(destination)
        self._archived_directory = destination
        try:
            source.parent.rmdir()
        except OSError:
            pass
        update_metadata = getattr(live_store, "update_session_metadata", None)
        if callable(update_metadata):
            try:
                update_metadata({"opening_evidence": receipt})
            except Exception as exc:
                # The directory move is already the authoritative atomic
                # publication.  Never turn a completed live start into a
                # leaked half-started runtime only because manifest annotation
                # failed afterwards; leave a local receipt beside the evidence.
                try:
                    atomic_write_json(
                        destination / "archive_metadata_error.json",
                        {
                            "schema": "guandan.opening-archive-error/1",
                            "status": "archived_manifest_update_failed",
                            "error_type": type(exc).__name__,
                            "message": str(exc),
                        },
                    )
                except OSError:
                    pass
        return destination


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
        self._pending_opening_recording: ListenerRecording | None = None
        self._active_listener_recording: ListenerRecording | None = None
        self.sessions_root = resolve_sessions_root(capture.profiles_root, profile_name)

    def with_advisor(self, advisor: AdvicePort) -> "DefaultLiveSessionFactory":
        replacement = DefaultLiveSessionFactory(
            self.capture,
            self.recognizer,
            advisor,
            profile_name=self.profile_name,
        )
        replacement._pending_opening_recording = self._pending_opening_recording
        replacement._active_listener_recording = self._active_listener_recording
        if (
            replacement._active_listener_recording is not None
            and not replacement._active_listener_recording.closed
        ):
            replacement._active_listener_recording._confirmed_callback = (
                replacement._remember_confirmed_opening
            )
        self._pending_opening_recording = None
        self._active_listener_recording = None
        return replacement

    def start_session(
        self,
        *,
        round_level: str,
        hand: tuple[str, ...],
        lead_player: str | None,
        recognition_strategy: str,
        on_update: Callable[[Any], None] | None = None,
    ) -> LiveSessionConstruction:
        opening_recording, self._pending_opening_recording = (
            self._pending_opening_recording,
            None,
        )
        try:
            recording = self._create_recording(recognition_strategy)
        except Exception:
            if opening_recording is not None:
                self._mark_opening_promotion_failed(
                    opening_recording, reason="live_recording_create_failed"
                )
            raise
        assert recording is not None
        try:
            construction = self._start_session_with_recording(
                recording,
                round_level=round_level,
                hand=hand,
                lead_player=lead_player,
                recognition_strategy=recognition_strategy,
                on_update=on_update,
            )
        except Exception:
            if opening_recording is not None:
                self._mark_opening_promotion_failed(
                    opening_recording, reason="live_session_start_failed"
                )
            raise
        if opening_recording is not None:
            try:
                opening_recording.archive_into(recording.store)
            except Exception as exc:
                self._record_opening_archive_failure(
                    recording.store, opening_recording, exc
                )
        return construction

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
        store = LiveSessionStore.for_opening_evidence(
            self.capture.profiles_root,
            self.profile_name,
            sessions_root=self.sessions_root,
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
                "schema": "guandan.opening-evidence/1",
                "runtime": "opening_listener",
                "recording_phase": "listening",
                "initial_state_status": "unconfirmed",
                "recording_mode": "all",
                "sessions_root": str(self.sessions_root),
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
        recording = ListenerRecording(
            store=store,
            recorder=recorder,
            _confirmed_callback=self._remember_confirmed_opening,
        )
        self._active_listener_recording = recording
        return recording

    def _remember_confirmed_opening(self, recording: ListenerRecording) -> None:
        if self._active_listener_recording is recording:
            self._active_listener_recording = None
        previous = self._pending_opening_recording
        if previous is not None and previous is not recording:
            self._mark_opening_promotion_failed(
                previous, reason="superseded_opening"
            )
        self._pending_opening_recording = recording

    @staticmethod
    def _mark_opening_promotion_failed(
        recording: ListenerRecording, *, reason: str
    ) -> None:
        directory = Path(getattr(recording.store, "directory"))
        manifest_path = directory / "manifest.json"
        if not manifest_path.is_file():
            return
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            manifest.update(
                {
                    "opening_archive_status": "not_archived",
                    "opening_archive_reason": str(reason),
                }
            )
            atomic_write_json(manifest_path, manifest)
        except (OSError, ValueError, TypeError, json.JSONDecodeError):
            return

    @staticmethod
    def _record_opening_archive_failure(
        live_store: SessionPersistencePort,
        recording: ListenerRecording,
        error: Exception,
    ) -> None:
        metadata = {
            "opening_evidence": {
                "schema": "guandan.opening-archive/1",
                "status": "archive_failed",
                "opening_id": str(getattr(recording.store, "session_id", "")),
                "error_type": type(error).__name__,
                "message": str(error),
            }
        }
        update_metadata = getattr(live_store, "update_session_metadata", None)
        if callable(update_metadata):
            try:
                update_metadata(metadata)
            except Exception:
                pass
        DefaultLiveSessionFactory._mark_opening_promotion_failed(
            recording, reason="archive_failed"
        )

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
            sessions_root=self.sessions_root,
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
                "sessions_root": str(self.sessions_root),
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
            self.capture.profiles_root, self.profile_name,
            sessions_root=self.sessions_root,
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
