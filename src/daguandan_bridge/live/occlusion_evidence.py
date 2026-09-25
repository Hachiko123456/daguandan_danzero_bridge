"""Serializable evidence and delayed decisions for visually occluded actions.

This module is deliberately independent from the reducer.  It describes what
was observed and whether that observation is safe to use; callers decide how a
confirmed action is persisted.  In particular, an empty recognition is never
considered a successful read and a disagreement between concrete candidates
can be surfaced as a strategy block instead of being guessed away.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Iterable, Literal

EvidenceState = Literal["known", "unknown", "occluded", "stale", "confirmed", "rejected"]
EVIDENCE_STATES: frozenset[str] = frozenset(
    {"known", "unknown", "occluded", "stale", "confirmed", "rejected"}
)


def _cards(values: Iterable[str] = ()) -> tuple[str, ...]:
    return tuple(str(value) for value in values)


def _options(values: Iterable[Iterable[str]] = ()) -> tuple[tuple[str, ...], ...]:
    return tuple(tuple(str(suit) for suit in choices) for choices in values)


@dataclass(frozen=True)
class OcclusionEvidence:
    """A transport-safe snapshot of recognition evidence and its disposition."""

    state: EvidenceState
    action_id: str = ""
    cards: tuple[str, ...] = ()
    suit_options: tuple[tuple[str, ...], ...] = ()
    candidates: tuple[tuple[str, ...], ...] = ()
    confirmations: int = 0
    required_confirmations: int = 2
    strategy_blocked: bool = False
    reason: str = ""
    observed_at: str | None = None
    expires_at: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.state not in EVIDENCE_STATES:
            raise ValueError(f"unsupported occlusion evidence state: {self.state!r}")
        if self.confirmations < 0 or self.required_confirmations < 1:
            raise ValueError("confirmation counts must be non-negative and required >= 1")

    @property
    def is_actionable(self) -> bool:
        return (
            self.state in {"known", "confirmed"}
            and not self.strategy_blocked
            and self.confirmations >= self.required_confirmations
        )

    @property
    def delayed(self) -> bool:
        return self.state in {"unknown", "occluded", "stale"} or (
            self.state == "confirmed" and self.confirmations < self.required_confirmations
        )

    def to_dict(self) -> dict[str, Any]:
        """Return JSON-compatible primitives without leaking tuple internals."""

        return {
            "state": self.state,
            "action_id": self.action_id,
            "cards": list(self.cards),
            "suit_options": [list(options) for options in self.suit_options],
            "candidates": [list(candidate) for candidate in self.candidates],
            "confirmations": self.confirmations,
            "required_confirmations": self.required_confirmations,
            "strategy_blocked": self.strategy_blocked,
            "reason": self.reason,
            "observed_at": self.observed_at,
            "expires_at": self.expires_at,
            "metadata": dict(self.metadata),
        }

    as_dict = to_dict

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "OcclusionEvidence":
        """Restore evidence produced by :meth:`to_dict`."""

        return cls(
            state=str(payload.get("state", "rejected")),  # type: ignore[arg-type]
            action_id=str(payload.get("action_id", "")),
            cards=_cards(payload.get("cards", ())),
            suit_options=_options(payload.get("suit_options", ())),
            candidates=tuple(_cards(candidate) for candidate in payload.get("candidates", ())),
            confirmations=int(payload.get("confirmations", 0)),
            required_confirmations=int(payload.get("required_confirmations", 2)),
            strategy_blocked=bool(payload.get("strategy_blocked", False)),
            reason=str(payload.get("reason", "")),
            observed_at=payload.get("observed_at"),
            expires_at=payload.get("expires_at"),
            metadata=dict(payload.get("metadata", {})),
        )


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def mark_stale(evidence: OcclusionEvidence, *, now: datetime | None = None) -> OcclusionEvidence:
    """Mark time-bounded evidence stale without changing its original payload."""

    if not evidence.expires_at or evidence.state in {"rejected", "stale"}:
        return evidence
    current = now or datetime.now(timezone.utc)
    try:
        expires = datetime.fromisoformat(evidence.expires_at)
    except ValueError:
        return evidence
    if expires.tzinfo is None:
        expires = expires.replace(tzinfo=timezone.utc)
    if current >= expires:
        return OcclusionEvidence(
            **{**evidence.__dict__, "state": "stale", "strategy_blocked": True,
               "reason": evidence.reason or "occlusion evidence expired"}
        )
    return evidence
