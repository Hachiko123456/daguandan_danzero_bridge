"""Public session-selection facade shared by GUI and validation runners.

``SessionDescriptor`` is defined once in ``session_workbench`` and re-exported
here so the runner's canonical import path remains stable; the GUI creates no
second session model.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from .session_workbench import SessionDescriptor, discover_session_paths, inspect_session


@dataclass(frozen=True)
class SessionSelection:
    target: Path
    sessions: tuple[SessionDescriptor, ...]

    @property
    def is_single_session(self) -> bool:
        return len(self.sessions) == 1 and self.sessions[0].root == self.target


class SessionLocator:
    """Locate one selected session or direct children of a selected root."""

    def discover(self, target: Path | str) -> SessionSelection:
        root = Path(target).expanduser().resolve()
        paths = discover_session_paths(root)
        return SessionSelection(root, tuple(inspect_session(path) for path in paths))

    @staticmethod
    def describe(session: Path | str) -> SessionDescriptor:
        return inspect_session(session)


def inspect_sessions(root: Path | str) -> tuple[SessionDescriptor, ...]:
    return tuple(inspect_session(path) for path in discover_session_paths(root))


__all__ = ["SessionDescriptor", "SessionLocator", "SessionSelection", "discover_session_paths", "inspect_session", "inspect_sessions"]
