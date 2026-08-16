from __future__ import annotations

import hashlib
import os
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from time import monotonic_ns
from typing import Any, Callable

from ..application.ports import (
    AdvicePort,
    CapturePort,
    LiveSessionConstruction,
    RecognitionPort,
)
from ..advisor_strategy import load_profile_session_data_recording_enabled
from ..live.orchestrator import LiveOrchestrator
from ..live.recorder import InMemorySessionRecorder, SessionRecorder
from ..live.reducer import LiveReducer
from ..live.session_store import InMemoryLiveSessionStore, LiveSessionStore


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
        store: LiveSessionStore | InMemoryLiveSessionStore | None = None
        recorder: SessionRecorder | InMemorySessionRecorder | None = None
        source = None
        try:
            loaded = self.capture.load_profile(self.profile_name)
            save_session_data = load_profile_session_data_recording_enabled(
                self.capture.profiles_root,
                self.profile_name,
            )
            if save_session_data:
                store = LiveSessionStore(
                    self.capture.profiles_root,
                    self.profile_name,
                )
                manifest = build_session_manifest(
                    loaded.paths.profile_config_path,
                    loaded.paths.templates_config_path,
                )
                manifest["recognition_strategy"] = recognition_strategy
                audit_info = getattr(self.advisor, "audit_info", None)
                if callable(audit_info):
                    manifest["advisor"] = audit_info()
                else:
                    manifest["advisor"] = {
                        "backend": type(self.advisor).__name__ if self.advisor else "none",
                        "standard_no_tribute": True,
                    }
                store.start(manifest)
                recorder = SessionRecorder(
                    store.directory,
                    size=loaded.config.base_size,
                    fps=10,
                )
            else:
                store = InMemoryLiveSessionStore(
                    self.capture.profiles_root,
                    self.profile_name,
                )
                store.start({})
                recorder = InMemorySessionRecorder(store.directory)
            orchestrator = LiveOrchestrator(
                reducer=LiveReducer(store.session_id),
                store=store,
                recorder=recorder,
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
            return LiveSessionConstruction(orchestrator, source, update)
        except Exception:
            if source is not None:
                source.close()
            if recorder is not None:
                recording = recorder.close()
                if store is not None:
                    store.seal(
                        frame_count=recording.frame_count,
                        dropped_frames=recording.dropped_frames,
                        incident_media_failures=(
                            failure.to_dict()
                            for failure in recording.incident_media_failures
                        ),
                    )
            raise


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
