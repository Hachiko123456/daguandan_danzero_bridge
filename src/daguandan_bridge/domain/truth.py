from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from ..danzero.state import SEATS, Seat


LabelStatus = Literal["draft", "verified", "rejected"]
TeamResult = Literal["win", "loss", "unknown"]
_LABEL_STATUSES = frozenset({"draft", "verified", "rejected"})


def normalize_label_status(value: object) -> LabelStatus:
    status = str(value or "draft").strip().lower()
    if status not in _LABEL_STATUSES:
        raise ValueError(f"invalid label status: {status}")
    return status  # type: ignore[return-value]


@dataclass(frozen=True)
class LabelProvenance:
    source: str = "legacy_migration"
    annotator: str = ""
    annotated_at: str = ""
    confidence: float | None = None

    def to_dict(self) -> dict[str, object]:
        raw: dict[str, object] = {"source": self.source}
        if self.annotator:
            raw["annotator"] = self.annotator
        if self.annotated_at:
            raw["annotated_at"] = self.annotated_at
        if self.confidence is not None:
            if not 0 <= self.confidence <= 1:
                raise ValueError("label confidence must be between 0 and 1")
            raw["confidence"] = self.confidence
        return raw

    @classmethod
    def from_dict(
        cls,
        raw: object,
        *,
        default_source: str = "legacy_migration",
    ) -> "LabelProvenance":
        value = raw if isinstance(raw, dict) else {}
        confidence = value.get("confidence")
        return cls(
            source=str(value.get("source", default_source)),
            annotator=str(value.get("annotator", "")),
            annotated_at=str(value.get("annotated_at", "")),
            confidence=float(confidence) if confidence is not None else None,
        )


@dataclass(frozen=True)
class TruthEvidence:
    frame_indices: tuple[int, ...] = ()
    monotonic_ms: int | None = None
    roi_name: str = ""

    def __post_init__(self) -> None:
        if any(index < 0 for index in self.frame_indices):
            raise ValueError("evidence frame indices cannot be negative")
        if self.monotonic_ms is not None and self.monotonic_ms < 0:
            raise ValueError("evidence monotonic time cannot be negative")

    def to_dict(self) -> dict[str, object]:
        raw: dict[str, object] = {"frame_indices": list(self.frame_indices)}
        if self.monotonic_ms is not None:
            raw["monotonic_ms"] = self.monotonic_ms
        if self.roi_name:
            raw["roi_name"] = self.roi_name
        return raw

    @classmethod
    def from_dict(cls, raw: object) -> "TruthEvidence":
        value = raw if isinstance(raw, dict) else {}
        indices = value.get("frame_indices", ())
        if not isinstance(indices, (list, tuple)):
            raise ValueError("evidence frame_indices must be an array")
        monotonic = value.get("monotonic_ms")
        return cls(
            frame_indices=tuple(int(item) for item in indices),
            monotonic_ms=int(monotonic) if monotonic is not None else None,
            roi_name=str(value.get("roi_name", "")),
        )


@dataclass(frozen=True)
class TruthOutcome:
    complete: bool = False
    finish_order: tuple[Seat, ...] = ()
    team_result: TeamResult = "unknown"
    reward: float | None = None
    reward_scheme: str = ""

    def __post_init__(self) -> None:
        if any(seat not in SEATS for seat in self.finish_order):
            raise ValueError("outcome finish_order contains an invalid seat")
        if len(set(self.finish_order)) != len(self.finish_order):
            raise ValueError("outcome finish_order contains duplicate seats")
        if self.team_result not in {"win", "loss", "unknown"}:
            raise ValueError("outcome team_result is invalid")
        if self.complete:
            if len(self.finish_order) != 4 or set(self.finish_order) != set(SEATS):
                raise ValueError("complete outcome requires all four seats")
            expected = "win" if self.finish_order[0] in {"self", "opposite"} else "loss"
            if self.team_result != expected:
                raise ValueError("outcome team_result conflicts with finish_order")
            if self.reward is None or not self.reward_scheme:
                raise ValueError("complete outcome requires reward and reward_scheme")
            if (self.team_result == "win" and self.reward <= 0) or (
                self.team_result == "loss" and self.reward >= 0
            ):
                raise ValueError("outcome reward sign conflicts with team_result")
        elif self.team_result != "unknown" or self.reward is not None:
            raise ValueError("incomplete outcome cannot declare result or reward")

    def to_dict(self) -> dict[str, object]:
        return {
            "complete": self.complete,
            "finish_order": list(self.finish_order),
            "team_result": self.team_result,
            "reward": self.reward,
            "reward_scheme": self.reward_scheme,
        }

    @classmethod
    def from_dict(cls, raw: object) -> "TruthOutcome":
        value = raw if isinstance(raw, dict) else {}
        finish = value.get("finish_order", ())
        if not isinstance(finish, (list, tuple)):
            raise ValueError("outcome finish_order must be an array")
        reward = value.get("reward")
        return cls(
            complete=bool(value.get("complete", False)),
            finish_order=tuple(str(seat) for seat in finish),  # type: ignore[arg-type]
            team_result=str(value.get("team_result", "unknown")),  # type: ignore[arg-type]
            reward=float(reward) if reward is not None else None,
            reward_scheme=str(value.get("reward_scheme", "")),
        )
