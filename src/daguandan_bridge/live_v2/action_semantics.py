"""Pure immutable rule-confirmed action-semantics contracts."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping


def _text(value: object, name: str, *, allow_empty: bool = False) -> str:
    if not isinstance(value, str) or (not allow_empty and not value.strip()):
        raise ValueError(f"{name} must be text")
    return value


@dataclass(frozen=True, slots=True)
class ActionInterpretation:
    move_type: str
    key: str | int
    logical_label: str = ""
    wildcard_assignments: tuple[tuple[str, str], ...] = ()
    type_id: int | None = None
    claim_ranks: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not self.move_type and self.type_id is None:
            raise ValueError("interpretation requires move_type or type_id")
        if self.move_type:
            _text(self.move_type, "move_type")
        if not isinstance(self.key, (str, int)) or isinstance(self.key, bool):
            raise TypeError("key must be text or an integer")
        _text(self.logical_label, "logical_label", allow_empty=True)
        if not isinstance(self.wildcard_assignments, tuple):
            raise TypeError("wildcard_assignments must be a tuple")
        for physical_card, as_rank in self.wildcard_assignments:
            _text(physical_card, "physical_card")
            _text(as_rank, "as_rank")
        if self.type_id is not None and (
            isinstance(self.type_id, bool) or not isinstance(self.type_id, int)
        ):
            raise TypeError("type_id must be an integer or None")
        if not isinstance(self.claim_ranks, tuple):
            raise TypeError("claim_ranks must be a tuple")
        for rank in self.claim_ranks:
            _text(rank, "claim_rank")

    def to_dict(self) -> dict[str, object]:
        if self.type_id is not None and not self.move_type:
            return {
                "type_id": self.type_id,
                "key": self.key,
                "claim_ranks": list(self.claim_ranks),
            }
        return {
            "move_type": self.move_type,
            "key": self.key,
            "logical_label": self.logical_label,
            "wildcard_assignments": [
                {"physical_card": card, "as_rank": rank}
                for card, rank in self.wildcard_assignments
            ],
        }

    @classmethod
    def from_mapping(cls, raw: Mapping[str, object]) -> ActionInterpretation:
        assignments = raw.get(
            "wildcard_assignments", raw.get("wildcard_substitutions", ())
        )
        return cls(
            move_type=str(raw.get("move_type", raw.get("play_type", ""))),
            key=(
                raw.get("key", "")
                if isinstance(raw.get("key", ""), int)
                else str(raw.get("key", ""))
            ),
            logical_label=str(raw.get("logical_label", "")),
            wildcard_assignments=tuple(
                (str(item.get("physical_card", "")), str(item.get("as_rank", "")))
                for item in assignments
                if isinstance(item, Mapping)
            )
            if isinstance(assignments, (list, tuple))
            else (),
            type_id=(
                int(raw["type_id"])
                if "type_id" in raw and raw["type_id"] is not None
                else None
            ),
            claim_ranks=tuple(str(item) for item in raw.get("claim_ranks", ()))
            if isinstance(raw.get("claim_ranks", ()), (list, tuple))
            else (),
        )


@dataclass(frozen=True, slots=True)
class ActionSemantics:
    selected: ActionInterpretation
    candidates: tuple[ActionInterpretation, ...]
    selection_source: str

    def __post_init__(self) -> None:
        if not isinstance(self.selected, ActionInterpretation):
            raise TypeError("selected must be an ActionInterpretation")
        if not isinstance(self.candidates, tuple) or not self.candidates:
            raise ValueError("candidates must be a non-empty tuple")
        if any(not isinstance(item, ActionInterpretation) for item in self.candidates):
            raise TypeError("candidates must contain ActionInterpretation values")
        if len(set(self.candidates)) != len(self.candidates):
            raise ValueError("candidate interpretations must be unique")
        if self.selected not in self.candidates:
            raise ValueError("selected interpretation must be one of candidates")
        _text(self.selection_source, "selection_source")

    @property
    def ambiguous(self) -> bool:
        return len(self.candidates) > 1

    def to_metadata(self) -> dict[str, object]:
        return {
            "interpretation_ambiguous": self.ambiguous,
            "candidate_interpretations": [item.to_dict() for item in self.candidates],
            "selected_interpretation": self.selected.to_dict(),
            "selection_source": self.selection_source,
        }

    @classmethod
    def requested_from_metadata(
        cls, raw: Mapping[str, object] | None
    ) -> ActionSemantics | None:
        if not raw:
            return None
        selected_raw = raw.get("selected_interpretation")
        if not isinstance(selected_raw, Mapping):
            if not any(key in raw for key in ("move_type", "play_type")):
                return None
            selected_raw = raw
        selected = ActionInterpretation.from_mapping(selected_raw)
        candidates_raw = raw.get("candidate_interpretations", ())
        candidates = (
            tuple(
                ActionInterpretation.from_mapping(item)
                for item in candidates_raw
                if isinstance(item, Mapping)
            )
            if isinstance(candidates_raw, (list, tuple))
            else ()
        )
        if selected not in candidates:
            candidates = (selected, *candidates)
        return cls(
            selected=selected,
            candidates=tuple(dict.fromkeys(candidates)),
            selection_source=str(raw.get("selection_source", "explicit_manual")),
        )


__all__ = ["ActionInterpretation", "ActionSemantics"]
