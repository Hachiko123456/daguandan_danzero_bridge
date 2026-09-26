from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .application.ports import AdvicePort, CapturePort, RecognitionPort, SessionFactoryPort
from .advisor_strategy import build_advisor, load_profile_advisor_strategy
from .config import PROFILES_ROOT
from .session_paths import resolve_sessions_root


@dataclass(frozen=True)
class LiveControllerDependencies:
    capture: CapturePort
    recognizer: RecognitionPort
    advisor: AdvicePort
    session_factory: SessionFactoryPort


@dataclass(frozen=True)
class ApplicationDependencies:
    live_runtime: Any
    annotation_service: Any
    sessions_root: Path


def build_live_controller_dependencies(
    *,
    profile_name: str = "tencent_daguandan",
    capture: CapturePort | None = None,
    advisor: AdvicePort | None = None,
    advisor_strategy: str | None = None,
) -> LiveControllerDependencies:
    """The single composition root for the live assistant adapters."""

    from .annotation_service import AnnotationService
    from .capture_service import CaptureService
    from .infrastructure.live_session import DefaultLiveSessionFactory
    from .live.session_store import LiveSessionStore
    from .recognition_service import ScreenshotRecognitionService
    from .template_service import TemplateService

    capture_adapter = capture or CaptureService()
    LiveSessionStore.recover_incomplete_sessions(
        capture_adapter.profiles_root,
        profile_name,
        sessions_root=resolve_sessions_root(capture_adapter.profiles_root, profile_name),
    )
    annotation = AnnotationService(capture_adapter.profiles_root, profile_name)
    templates = TemplateService(capture_adapter.profiles_root, profile_name)
    recognizer = ScreenshotRecognitionService(annotation, templates)
    selected_strategy = advisor_strategy or load_profile_advisor_strategy(
        capture_adapter.profiles_root,
        profile_name,
    )
    advice_adapter = advisor or build_advisor(
        selected_strategy,
        profiles_root=capture_adapter.profiles_root,
        profile_name=profile_name,
    )
    factory = DefaultLiveSessionFactory(
        capture_adapter,
        recognizer,
        advice_adapter,
        profile_name=profile_name,
    )
    # Bind the startup report to the *actual* selected profile. This is done
    # once at composition, never on the capture thread or for each frame.
    from .startup_diagnostics import (
        initialized_startup_diagnostics, record_startup_event, write_startup_report,
    )
    if initialized_startup_diagnostics() is not None:
        try:
            write_startup_report(
                profiles_root=Path(capture_adapter.profiles_root).resolve(),
                profile_name=profile_name,
                resource_identity=recognizer.recognition_resource_identity(),
            )
        except Exception as exc:
            record_startup_event("profile_evidence_failed", {
                "error": f"{type(exc).__name__}: {exc}", "profile_name": profile_name,
            })
    return LiveControllerDependencies(
        capture_adapter,
        recognizer,
        advice_adapter,
        factory,
    )


def build_application_dependencies(
    *,
    profile_name: str = "tencent_daguandan",
    live_runtime: Any | None = None,
) -> ApplicationDependencies:
    from .annotation_service import AnnotationService
    from .gui.live_controller import LiveAssistantController
    from .opening_evidence import build_opening_evidence_monitor

    if live_runtime is None:
        live = build_live_controller_dependencies(profile_name=profile_name)
        live_runtime = LiveAssistantController(
            live.capture,
            profile_name=profile_name,
            recognition_service=live.recognizer,
            advisor=live.advisor,
            session_factory=live.session_factory,
            opening_evidence_monitor=build_opening_evidence_monitor(
                profiles_root=live.capture.profiles_root,
                profile_name=profile_name,
            ),
        )
        profiles_root = Path(live.capture.profiles_root)
    else:
        profiles_root = Path(getattr(getattr(live_runtime, "capture_service", None), "profiles_root", PROFILES_ROOT))
    return ApplicationDependencies(
        live_runtime,
        AnnotationService(profiles_root, profile_name),
        resolve_sessions_root(profiles_root, profile_name),
    )
