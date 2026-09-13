"""Shared session descriptor and profile context for the workbench.

``SessionDescriptor`` is defined once here and re-exported by
``application.session_locator`` for the validation runner.  The GUI adds only
``ProfileContext`` and never derives it from a session path.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from ..config import PROFILES_ROOT
from ..live.truth_log import TruthLog, load_truth_log


@dataclass(frozen=True)
class ProfileContext:
    """Recognition resources selected independently from a session path."""

    profiles_root: Path
    profile_name: str

    def __post_init__(self) -> None:
        root = Path(self.profiles_root).expanduser().resolve()
        name = str(self.profile_name).strip()
        if not name:
            raise ValueError("profile_name cannot be empty")
        object.__setattr__(self, "profiles_root", root)
        object.__setattr__(self, "profile_name", name)

    @property
    def profile_path(self) -> Path:
        return self.profiles_root / self.profile_name

    @property
    def exists(self) -> bool:
        return self.profile_path.is_dir()

    @property
    def display_name(self) -> str:
        return f"{self.profile_name}  ({self.profiles_root})"


@dataclass(frozen=True)
class SessionDescriptor:
    """Canonical read-only integrity and TruthLog state for one session."""

    root: Path
    session_id: str
    manifest_path: Path
    video_path: Path
    frame_index_path: Path
    timeline_path: Path
    truth_log_path: Path
    manifest_readable: bool
    has_video: bool
    has_frame_index: bool
    has_timeline: bool
    truth_status: str
    truth_error: str
    frame_count: int | None
    timeline_event_count: int | None

    @property
    def source(self) -> Path:
        return self.root

    @property
    def has_manifest(self) -> bool:
        return self.manifest_readable

    @property
    def has_truth_log(self) -> bool:
        return self.truth_status != "missing"

    @property
    def is_playable(self) -> bool:
        """A recording is scannable even when the optional frame index is absent."""

        # The AVI is the source material.  A frame index is useful capture
        # metadata, but the scanner can derive timestamps from the AVI FPS.
        return self.has_video

    @property
    def integrity_status(self) -> str:
        if not self.manifest_readable:
            return "invalid"
        if self.is_playable:
            return "ready"
        if self.has_video or self.has_frame_index or self.has_timeline:
            return "partial"
        return "empty"

    @property
    def integrity_issues(self) -> tuple[str, ...]:
        issues: list[str] = []
        if not self.manifest_readable:
            issues.append("manifest.json 不可读或缺失")
        if not self.has_video:
            issues.append("缺少 video/game.avi")
        # ``frame_index.jsonl`` is optional for video-first scanning.
        if not self.has_timeline:
            issues.append("缺少 timeline.jsonl")
        if self.truth_status == "invalid":
            issues.append(f"truth_log.json 不可读：{self.truth_error}")
        return tuple(issues)

    @property
    def truth_label(self) -> str:
        return {
            "verified": "可信 TruthLog",
            "draft": "草稿 TruthLog",
            "missing": "缺少 TruthLog",
            "invalid": "TruthLog 不可读",
        }.get(self.truth_status, self.truth_status)

    def to_dict(self, *, relative_to: Path | None = None) -> dict[str, object]:
        relative = None
        if relative_to is not None:
            try:
                relative = self.root.relative_to(relative_to).as_posix()
            except ValueError:
                pass
        return {
            "session_id": self.session_id,
            "source": str(self.root),
            "relative_source": relative,
            "has_manifest": self.has_manifest,
            "has_video": self.has_video,
            "has_frame_index": self.has_frame_index,
            "has_timeline": self.has_timeline,
            "has_truth_log": self.has_truth_log,
            "integrity_status": self.integrity_status,
            "integrity_issues": list(self.integrity_issues),
            "truth_status": self.truth_status,
            "truth_error": self.truth_error or None,
            "frame_count": self.frame_count,
            "timeline_event_count": self.timeline_event_count,
        }


def default_profile_context() -> ProfileContext:
    return ProfileContext(PROFILES_ROOT, "tencent_daguandan")


def discover_profiles(
    profiles_root: Path | str = PROFILES_ROOT,
) -> tuple[ProfileContext, ...]:
    """Discover profile directories without inspecting session parents."""

    root = Path(profiles_root).expanduser().resolve()
    if not root.is_dir():
        return ()
    result = []
    for path in sorted(
        (item for item in root.iterdir() if item.is_dir()),
        key=lambda item: item.name.lower(),
    ):
        if any(
            (path / marker).exists()
            for marker in ("profile.json", "regions_config.json", "templates")
        ):
            result.append(ProfileContext(root, path.name))
    return tuple(result)


def _is_session_directory(path: Path) -> bool:
    return path.is_dir() and any(
        (path / marker).exists()
        for marker in (
            "manifest.json",
            "timeline.jsonl",
            "truth_log.json",
            "video/game.avi",
            "video/frame_index.jsonl",
        )
    )


def discover_session_paths(root: Path | str) -> tuple[Path, ...]:
    selected = Path(root).expanduser().resolve()
    if _is_session_directory(selected):
        return (selected,)
    if not selected.is_dir():
        return ()
    return tuple(
        sorted(
            (path for path in selected.iterdir() if _is_session_directory(path)),
            key=lambda path: path.name.lower(),
        )
    )


def _read_json(path: Path) -> dict[str, object] | None:
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError):
        return None
    return raw if isinstance(raw, dict) else None


def _count_json_lines(path: Path) -> int | None:
    if not path.is_file():
        return None
    count = 0
    try:
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    count += 1
    except (OSError, UnicodeError):
        return None
    return count


def inspect_session(session: Path | str) -> SessionDescriptor:
    root = Path(session).expanduser().resolve()
    manifest_path = root / "manifest.json"
    video_path = root / "video" / "game.avi"
    frame_index_path = root / "video" / "frame_index.jsonl"
    timeline_path = root / "timeline.jsonl"
    truth_log_path = root / "truth_log.json"
    manifest = _read_json(manifest_path)
    session_id = str((manifest or {}).get("session_id") or root.name)
    truth_status = "missing"
    truth_error = ""
    if truth_log_path.is_file():
        try:
            truth = load_truth_log(truth_log_path, session_id=session_id)
        except Exception as exc:
            truth_status = "invalid"
            truth_error = f"{type(exc).__name__}: {exc}"
        else:
            truth_status = "verified" if truth.label_status == "verified" else "draft"
    raw_frame_count = (manifest or {}).get("frame_count")
    frame_count = (
        raw_frame_count
        if isinstance(raw_frame_count, int) and raw_frame_count >= 0
        else _count_json_lines(frame_index_path)
    )
    return SessionDescriptor(
        root=root,
        session_id=session_id,
        manifest_path=manifest_path,
        video_path=video_path,
        frame_index_path=frame_index_path,
        timeline_path=timeline_path,
        truth_log_path=truth_log_path,
        manifest_readable=manifest is not None,
        has_video=video_path.is_file(),
        has_frame_index=frame_index_path.is_file(),
        has_timeline=timeline_path.is_file(),
        truth_status=truth_status,
        truth_error=truth_error,
        frame_count=frame_count,
        timeline_event_count=_count_json_lines(timeline_path),
    )


def inspect_sessions(root: Path | str) -> tuple[SessionDescriptor, ...]:
    return tuple(inspect_session(path) for path in discover_session_paths(root))


__all__ = [
    "ProfileContext",
    "SessionDescriptor",
    "default_profile_context",
    "discover_profiles",
    "discover_session_paths",
    "inspect_session",
    "inspect_sessions",
]
