"""Identity and version contracts for the live-v2 pipeline."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class Seat(str, Enum):
    SELF = "self"
    RIGHT = "right"
    OPPOSITE = "opposite"
    LEFT = "left"


def require_text(value: str, field_name: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field_name} must be a non-empty string")


def require_non_negative_int(value: int, field_name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{field_name} must be a non-negative integer")


def require_probability(value: float, field_name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{field_name} must be a number between 0 and 1")
    if not 0.0 <= float(value) <= 1.0:
        raise ValueError(f"{field_name} must be between 0 and 1")


def require_tuple(value: object, field_name: str) -> None:
    if not isinstance(value, tuple):
        raise TypeError(f"{field_name} must be a tuple")


def require_enum(value: object, enum_type: type[Enum], field_name: str) -> None:
    if not isinstance(value, enum_type):
        raise TypeError(f"{field_name} must be a {enum_type.__name__}")


def require_instance(value: object, expected_type: type[object], field_name: str) -> None:
    if not isinstance(value, expected_type):
        raise TypeError(f"{field_name} must be a {expected_type.__name__}")


def require_unique_text_tuple(values: tuple[str, ...], field_name: str) -> None:
    require_tuple(values, field_name)
    for value in values:
        require_text(value, field_name)
    if len(set(values)) != len(values):
        raise ValueError(f"{field_name} must not contain duplicates")


@dataclass(frozen=True, slots=True)
class FrameIdentity:
    """Stable identity of one capture and its ROI contract.

    ``captured_ms`` belongs to the capture time domain. ``source_id`` names
    the physical/logical capture source and defaults to ``roi_version`` for
    compatibility with sources that have only one identity string.
    """

    session_id: str
    capture_generation: int
    frame_sequence: int
    captured_ms: int
    roi_version: str
    source_id: str = ""

    def __post_init__(self) -> None:
        require_text(self.session_id, "session_id")
        require_non_negative_int(self.capture_generation, "capture_generation")
        require_non_negative_int(self.frame_sequence, "frame_sequence")
        require_non_negative_int(self.captured_ms, "captured_ms")
        require_text(self.roi_version, "roi_version")
        if self.source_id == "":
            object.__setattr__(self, "source_id", self.roi_version)
        else:
            require_text(self.source_id, "source_id")


@dataclass(frozen=True, slots=True)
class StateVersion:
    """Authoritative rule-state position, independent of capture restarts."""

    session_id: str
    state_revision: int
    turn_index: int

    def __post_init__(self) -> None:
        require_text(self.session_id, "session_id")
        require_non_negative_int(self.state_revision, "state_revision")
        require_non_negative_int(self.turn_index, "turn_index")


@dataclass(frozen=True, slots=True)
class VersionIdentity:
    """Current capture/update identity with a derived rule-state version.

    ``state_revision`` and ``turn_index`` remain constructor fields for source
    compatibility. ``state_version`` is derived on demand, so rule state has
    one source of truth rather than duplicated mutable values.
    """

    session_id: str
    capture_generation: int
    state_revision: int
    update_sequence: int
    turn_index: int

    def __post_init__(self) -> None:
        require_text(self.session_id, "session_id")
        require_non_negative_int(self.capture_generation, "capture_generation")
        require_non_negative_int(self.state_revision, "state_revision")
        require_non_negative_int(self.update_sequence, "update_sequence")
        require_non_negative_int(self.turn_index, "turn_index")

    def belongs_to(self, frame: FrameIdentity) -> bool:
        return (
            self.session_id == frame.session_id
            and self.capture_generation == frame.capture_generation
        )

    @property
    def state_version(self) -> StateVersion:
        return StateVersion(self.session_id, self.state_revision, self.turn_index)

    @classmethod
    def from_state(
        cls,
        state_version: StateVersion,
        *,
        capture_generation: int,
        update_sequence: int,
    ) -> VersionIdentity:
        require_instance(state_version, StateVersion, "state_version")
        return cls(
            state_version.session_id,
            capture_generation,
            state_version.state_revision,
            update_sequence,
            state_version.turn_index,
        )

    def with_state(
        self,
        state_version: StateVersion,
        *,
        update_sequence: int | None = None,
    ) -> VersionIdentity:
        """Keep the current capture generation while adopting rule state."""

        require_instance(state_version, StateVersion, "state_version")
        if state_version.session_id != self.session_id:
            raise ValueError("state_version belongs to another session")
        return VersionIdentity.from_state(
            state_version,
            capture_generation=self.capture_generation,
            update_sequence=(
                self.update_sequence if update_sequence is None else update_sequence
            ),
        )
