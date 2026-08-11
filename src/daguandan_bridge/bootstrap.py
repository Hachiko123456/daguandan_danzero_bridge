from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .application.ports import AdvicePort, CapturePort, RecognitionPort, SessionFactoryPort
from .config import PROFILES_ROOT


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
) -> LiveControllerDependencies:
    """The single composition root for the live assistant adapters."""

    from .annotation_service import AnnotationService
    from .capture_service import CaptureService
    from .danzero import DanzeroAdvisor
    from .infrastructure.live_session import DefaultLiveSessionFactory
    from .live.session_store import LiveSessionStore
    from .recognition_service import ScreenshotRecognitionService
    from .template_service import TemplateService

    capture_adapter = capture or CaptureService()
    LiveSessionStore.recover_incomplete_sessions(
        capture_adapter.profiles_root,
        profile_name,
    )
    annotation = AnnotationService(capture_adapter.profiles_root, profile_name)
    templates = TemplateService(capture_adapter.profiles_root, profile_name)
    recognizer = ScreenshotRecognitionService(annotation, templates)
    advice_adapter = advisor or DanzeroAdvisor()
    factory = DefaultLiveSessionFactory(
        capture_adapter,
        recognizer,
        advice_adapter,
        profile_name=profile_name,
    )
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

    if live_runtime is None:
        live = build_live_controller_dependencies(profile_name=profile_name)
        live_runtime = LiveAssistantController(
            live.capture,
            profile_name=profile_name,
            recognition_service=live.recognizer,
            advisor=live.advisor,
            session_factory=live.session_factory,
        )
        profiles_root = Path(live.capture.profiles_root)
    else:
        profiles_root = Path(getattr(getattr(live_runtime, "capture_service", None), "profiles_root", PROFILES_ROOT))
    return ApplicationDependencies(
        live_runtime,
        AnnotationService(profiles_root, profile_name),
        profiles_root / profile_name / "sessions",
    )
