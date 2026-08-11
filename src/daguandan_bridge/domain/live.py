from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

from ..danzero.state import PlayEvent, Seat


@dataclass(frozen=True)
class LiveEvent:
    event_id: str
    event_type: str
    session_id: str
    seq: int
    monotonic_ms: int
    wall_time: str
    trick_id: int
    turn_id: int
    actor: Seat | None
    payload: dict[str, object]
    confidence: float
    source: str
    state_revision_before: int
    state_revision_after: int
    evidence_refs: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, object]:
        raw = asdict(self)
        raw["evidence_refs"] = list(self.evidence_refs)
        return raw

    @classmethod
    def from_dict(cls, raw: dict[str, object]) -> "LiveEvent":
        values: dict[str, Any] = dict(raw)
        schema_version = values.pop("schema_version", 1)
        if schema_version != 1:
            raise ValueError(f"unsupported live event schema: {schema_version}")
        values["evidence_refs"] = tuple(values.get("evidence_refs", ()))
        return cls(**values)


@dataclass(frozen=True)
class LiveSnapshot:
    session_id: str
    round_level: str
    wild_rank: str
    current_player: Seat | None
    lead_player: Seat | None
    my_hand: tuple[str, ...]
    trick_plays: tuple[PlayEvent, ...]
    play_history: tuple[PlayEvent, ...]
    remaining_cards: dict[Seat, int]
    finished_seats: frozenset[Seat]
    trick_id: int
    turn_id: int
    revision: int
    initialized: bool

    def semantic_dict(self) -> dict[str, object]:
        def play_semantics(event: PlayEvent) -> dict[str, object]:
            return {
                "player": event.player,
                "cards": list(event.cards),
                "is_pass": event.is_pass,
            }

        return {
            "round_level": self.round_level,
            "wild_rank": self.wild_rank,
            "current_player": self.current_player,
            "lead_player": self.lead_player,
            "my_hand": list(self.my_hand),
            "trick_plays": [play_semantics(event) for event in self.trick_plays],
            "play_history": [play_semantics(event) for event in self.play_history],
            "remaining_cards": dict(self.remaining_cards),
            "finished_seats": sorted(self.finished_seats),
            "trick_id": self.trick_id,
            "turn_id": self.turn_id,
            "initialized": self.initialized,
        }
