from __future__ import annotations

"""Conservative, evidence-only audit performed immediately before sealing."""

from dataclasses import dataclass
from typing import Iterable, Mapping

from .danzero.state import Seat


SESSION_HEALTH_SCHEMA = "guandan.session-health/1"
PREMATURE_GAME_END = "HEALTH-PREMATURE-GAME-END"
FINISHED_SEATS_INCONSISTENT = "HEALTH-FINISHED-SEATS-INCONSISTENT"
ACTION_CHAIN_INCONSISTENT = "HEALTH-ACTION-CHAIN-INCONSISTENT"
RECORDING_INCOMPLETE = "HEALTH-RECORDING-INCOMPLETE"

_SEATS: tuple[Seat, ...] = ("self", "right", "opposite", "left")


@dataclass(frozen=True)
class SessionHealthIssue:
    code: str
    summary: str
    evidence: dict[str, object]

    def to_dict(self) -> dict[str, object]:
        return {
            "code": self.code,
            "severity": "FAIL",
            "summary": self.summary,
            "evidence": dict(self.evidence),
        }


def audit_session_health(
    snapshot: object,
    events: Iterable[object],
    *,
    recording_integrity: Mapping[str, object] | None = None,
) -> dict[str, object]:
    """Report only contradictions that can be proven from the final facts.

    The audit never rewrites or suppresses a timeline event.  In particular,
    a visually observed ``game_end_detected`` stays in the append-only
    timeline even when the audit marks it as an obvious premature terminal.
    """

    event_list = tuple(events)
    terminal = next(
        (
            event
            for event in reversed(event_list)
            if getattr(event, "event_type", None) == "game_end_detected"
        ),
        None,
    )
    remaining = _remaining_cards(snapshot)
    finished = frozenset(str(value) for value in getattr(snapshot, "finished_seats", ()) or ())
    issues: list[SessionHealthIssue] = []

    recording_status = str((recording_integrity or {}).get("status", "")).upper()
    if recording_integrity and recording_status in {"FAIL", "PARTIAL"}:
        issues.append(
            SessionHealthIssue(
                RECORDING_INCOMPLETE,
                    "recorded frame index cannot be fully decoded",
                {
                    "writer_frame_count": recording_integrity.get("writer_frame_count"),
                    "indexed_frame_count": recording_integrity.get("indexed_frame_count"),
                    "decodable_frame_count": recording_integrity.get("decodable_frame_count"),
                    "last_decodable_frame_index": recording_integrity.get(
                        "last_decodable_frame_index"
                    ),
                    "issues": list(recording_integrity.get("issues", ()) or ()),
                    "status": recording_status,
                    "tail_recovery": recording_integrity.get("tail_recovery"),
                },
            )
        )

    if terminal is not None and remaining:
        zero_seats = sorted(seat for seat, count in remaining.items() if count == 0)
        credible_finished = sorted(set(zero_seats) & finished)
        # A legal round end has either two teammate finishers (double-down)
        # or three finishers.  Fewer than two seats with both a zero count and
        # a finished marker is a proven terminal/state contradiction.
        if len(credible_finished) < 2:
            issues.append(
                SessionHealthIssue(
                    PREMATURE_GAME_END,
                    "terminal UI was accepted while every player still had many cards",
                    {
                        "remaining_cards": remaining,
                        "finished_seats": sorted(finished),
                        "terminal_event_id": getattr(terminal, "event_id", None),
                        "minimum_remaining": min(remaining.values()),
                        "credible_finished_seats": credible_finished,
                        "required_credible_finished_count": 2,
                    },
                )
            )

        zero_set = frozenset(zero_seats)
        impossible_finished = sorted(
            seat for seat in finished if remaining.get(seat, 1) != 0
        )
        unmarked_zero = sorted(zero_set - finished)
        if impossible_finished or unmarked_zero:
            issues.append(
                SessionHealthIssue(
                    FINISHED_SEATS_INCONSISTENT,
                    "finished-seat facts disagree with final card counts",
                    {
                        "remaining_cards": remaining,
                        "finished_seats": sorted(finished),
                        "finished_with_cards": impossible_finished,
                        "zero_without_finished_marker": unmarked_zero,
                    },
                )
            )

    plays = tuple(getattr(snapshot, "play_history", ()) or ())
    if remaining and plays:
        cards_played = {seat: 0 for seat in _SEATS}
        invalid_actions: list[dict[str, object]] = []
        for index, play in enumerate(plays):
            player = str(getattr(play, "player", ""))
            cards = tuple(getattr(play, "cards", ()) or ())
            is_pass = bool(getattr(play, "is_pass", False))
            if player not in cards_played or (is_pass and cards) or (not is_pass and not cards):
                invalid_actions.append(
                    {
                        "history_index": index,
                        "player": player,
                        "card_count": len(cards),
                        "is_pass": is_pass,
                    }
                )
                continue
            if not is_pass:
                cards_played[player] += len(cards)
        count_mismatches = {
            seat: {
                "remaining": remaining[seat],
                "cards_played": cards_played[seat],
                "expected_total": remaining[seat] + cards_played[seat],
                "expected_initial_total": 27,
                "missing_cards": 27 - (remaining[seat] + cards_played[seat]),
            }
            for seat in _SEATS
            if seat in remaining and remaining[seat] + cards_played[seat] != 27
        }
        if invalid_actions or count_mismatches:
            issues.append(
                SessionHealthIssue(
                    ACTION_CHAIN_INCONSISTENT,
                    "action history cannot be reconciled with final card counts",
                    {
                        "invalid_actions": invalid_actions[:20],
                        "count_mismatches": count_mismatches,
                        "history_length": len(plays),
                    },
                )
            )

    return {
        "schema": SESSION_HEALTH_SCHEMA,
        "status": "FAIL" if issues else "PASS",
        "terminal_event_present": terminal is not None,
        "checked_event_count": len(event_list),
        "issues": [issue.to_dict() for issue in issues],
    }


def _remaining_cards(snapshot: object) -> dict[str, int]:
    raw = getattr(snapshot, "remaining_cards", None)
    if not isinstance(raw, Mapping):
        return {}
    result: dict[str, int] = {}
    for seat in _SEATS:
        try:
            count = int(raw[seat])
        except (KeyError, TypeError, ValueError):
            continue
        if count < 0 or count > 27:
            continue
        result[seat] = count
    return result


__all__ = [
    "ACTION_CHAIN_INCONSISTENT",
    "FINISHED_SEATS_INCONSISTENT",
    "PREMATURE_GAME_END",
    "RECORDING_INCOMPLETE",
    "SESSION_HEALTH_SCHEMA",
    "SessionHealthIssue",
    "audit_session_health",
]
