"""Pure validation and construction of trusted rule-state snapshots."""

from __future__ import annotations

from dataclasses import dataclass

from .corrections import ConfirmedCorrection
from .game_state import GameAction, SeatCardCount, TrustedGameSnapshot
from .identity import StateVersion
from .types import ActionKind, ConfirmedAction, Seat, VersionIdentity


@dataclass(frozen=True, slots=True)
class ReducedActionState:
    seat: Seat
    kind: ActionKind
    cards: tuple[str, ...]
    suit_options: tuple[tuple[str, ...], ...]


@dataclass(frozen=True, slots=True)
class ReducedGameState:
    session_id: str
    revision: int
    round_level: str
    wild_rank: str
    trick_index: int
    current_seat: Seat | None
    lead_seat: Seat | None
    my_hand: tuple[str, ...]
    play_history: tuple[ReducedActionState, ...]
    current_trick: tuple[ReducedActionState, ...]
    remaining: tuple[SeatCardCount, ...]
    finished: tuple[Seat, ...]
    initialized: bool


class TrustedSnapshotLedger:
    """Own the immutable committed-action chain independently of a reducer."""

    def __init__(
        self,
        version: VersionIdentity,
        seed_actions: tuple[ConfirmedAction, ...],
        seed_corrections: tuple[ConfirmedCorrection, ...] = (),
    ) -> None:
        self._version = version
        self._actions = list(seed_actions)
        self._corrections = list(seed_corrections)

    @property
    def version(self) -> VersionIdentity:
        return self._version

    @property
    def actions(self) -> tuple[ConfirmedAction, ...]:
        return tuple(self._actions)

    @property
    def corrections(self) -> tuple[ConfirmedCorrection, ...]:
        return tuple(self._corrections)

    def matches(self, version: VersionIdentity) -> bool:
        return _is_current_view(version, self._version)

    def snapshot(
        self,
        state: ReducedGameState,
        captured_ms: int,
        *,
        version: VersionIdentity | None = None,
    ) -> TrustedGameSnapshot:
        requested = self._version if version is None else version
        if not _is_current_view(requested, self._version):
            raise ValueError("snapshot version does not match committed rule state")
        return build_trusted_snapshot(
            state=state,
            actions=self.actions,
            corrections=self.corrections,
            version=requested,
            captured_ms=captured_ms,
        )

    def preview_append(
        self,
        state: ReducedGameState,
        actions: tuple[ConfirmedAction, ...],
        resulting_version: VersionIdentity,
        captured_ms: int,
    ) -> TrustedGameSnapshot:
        if not actions or actions[0].version_before != self._version.state_version:
            raise ValueError("transaction does not continue committed ledger")
        if resulting_version.state_version != actions[-1].version_after:
            raise ValueError("resulting capture version does not match transaction")
        return build_trusted_snapshot(
            state=state,
            actions=self.actions + actions,
            corrections=self.corrections,
            version=resulting_version,
            captured_ms=captured_ms,
        )

    def append(
        self, actions: tuple[ConfirmedAction, ...], resulting_version: VersionIdentity
    ) -> None:
        if not actions or actions[0].version_before != self._version.state_version:
            raise ValueError("transaction does not continue committed ledger")
        if resulting_version.state_version != actions[-1].version_after:
            raise ValueError("resulting capture version does not match transaction")
        self._actions.extend(actions)
        self._version = resulting_version

    def preview_correction(
        self,
        state: ReducedGameState,
        correction: ConfirmedCorrection,
        resulting_version: VersionIdentity,
        captured_ms: int,
    ) -> TrustedGameSnapshot:
        if correction.version_before != self._version.state_version:
            raise ValueError("correction does not continue committed ledger")
        if resulting_version.state_version != correction.version_after:
            raise ValueError("resulting capture version does not match correction")
        return build_trusted_snapshot(
            state=state,
            actions=self.actions,
            corrections=self.corrections + (correction,),
            version=resulting_version,
            captured_ms=captured_ms,
        )

    def correct(
        self,
        correction: ConfirmedCorrection,
        resulting_version: VersionIdentity,
    ) -> None:
        if correction.version_before != self._version.state_version:
            raise ValueError("correction does not continue committed ledger")
        if resulting_version.state_version != correction.version_after:
            raise ValueError("resulting capture version does not match correction")
        self._corrections.append(correction)
        self._version = resulting_version


def build_trusted_snapshot(
    *,
    state: ReducedGameState,
    actions: tuple[ConfirmedAction, ...],
    corrections: tuple[ConfirmedCorrection, ...] = (),
    version: VersionIdentity,
    captured_ms: int,
) -> TrustedGameSnapshot:
    """Cross-check the reducer projection and the committed action ledger."""

    if isinstance(captured_ms, bool) or not isinstance(captured_ms, int):
        raise TypeError("captured_ms must be an integer")
    if captured_ms < 0:
        raise ValueError("captured_ms must not be negative")
    if not state.initialized:
        raise ValueError("cannot trust an uninitialized reducer")
    if state.session_id != version.session_id or state.revision != version.state_revision:
        raise ValueError("reducer state does not match requested version")
    if actions and actions[0].version_before.session_id != version.session_id:
        raise ValueError("committed action ledger belongs to another session")
    _validate_unique(actions, corrections)
    latest = {item.target_action_id: item for item in corrections}
    history = tuple(
        GameAction.from_confirmed(action, latest.get(action.action_id))
        for action in actions
    )
    if len(history) != len(state.play_history) or any(
        not _same_action(formal, reduced)
        for formal, reduced in zip(history, state.play_history, strict=True)
    ):
        raise ValueError("reducer history and committed action ledger differ")
    if len(state.current_trick) > len(history):
        raise ValueError("reducer trick is longer than committed history")
    current_trick = history[-len(state.current_trick) :] if state.current_trick else ()
    if any(
        not _same_action(formal, reduced)
        for formal, reduced in zip(current_trick, state.current_trick, strict=True)
    ):
        raise ValueError("reducer current trick is not the exact history suffix")
    if history and captured_ms < history[-1].captured_ms:
        raise ValueError("capture watermark precedes committed evidence")
    if corrections and captured_ms < corrections[-1].corrected_ms:
        raise ValueError("capture watermark precedes committed correction")

    terminal = state.current_seat is None and len(state.finished) >= 2
    return TrustedGameSnapshot(
        version=version,
        round_level=state.round_level,
        wild_rank=state.wild_rank,
        trick_index=state.trick_index,
        current_seat=state.current_seat,
        lead_seat=state.lead_seat,
        my_hand=state.my_hand,
        play_history=history,
        current_trick=current_trick,
        remaining=state.remaining,
        finished=state.finished,
        trusted=True,
        terminal=terminal,
        captured_ms=captured_ms,
        correction_history=corrections,
    )


def _validate_unique(
    actions: tuple[ConfirmedAction, ...],
    corrections: tuple[ConfirmedCorrection, ...],
) -> None:
    identifiers: set[str] = set()
    for action in actions:
        if action.action_id in identifiers:
            raise ValueError("committed action ledger contains duplicate IDs")
        identifiers.add(action.action_id)
    correction_ids: set[str] = set()
    for correction in corrections:
        if correction.correction_id in correction_ids:
            raise ValueError("correction ledger contains duplicate IDs")
        correction_ids.add(correction.correction_id)
        if correction.target_action_id not in identifiers:
            raise ValueError("correction target is absent from action ledger")


def _is_current_view(
    requested: VersionIdentity, committed: VersionIdentity
) -> bool:
    return (
        requested.session_id == committed.session_id
        and requested.capture_generation == committed.capture_generation
        and requested.state_version == committed.state_version
        and requested.update_sequence >= committed.update_sequence
    )


def _same_action(formal: GameAction, reduced: ReducedActionState) -> bool:
    if formal.seat is not reduced.seat or formal.kind is not reduced.kind:
        return False
    return physical_action_entities(
        formal.cards, formal.suit_options
    ) == physical_action_entities(
        reduced.cards, reduced.suit_options
    )


def physical_action_entities(
    cards: tuple[str, ...], options: tuple[tuple[str, ...], ...]
) -> tuple[tuple[str, tuple[str, ...]], ...]:
    """Canonical physical entities shared by ledger and replay comparisons."""

    normalized: list[tuple[str, tuple[str, ...]]] = []
    for index, card in enumerate(cards):
        if card in {"small_joker", "big_joker"}:
            normalized.append((card, ()))
            continue
        supplied = options[index] if index < len(options) else ()
        physical = tuple(
            sorted(
                choice
                if choice[-1:] in "SHCD" and len(choice) > 1
                else f"{card[:-1]}{choice}"
                for choice in supplied
                if choice != card or not card.endswith("?")
            )
        )
        if not physical and card[-1:] in "SHCD":
            physical = (card,)
        normalized.append((card[:-1] if card[-1:] in "SHCD?" else card, physical))
    return tuple(sorted(normalized))


def _entities(
    cards: tuple[str, ...], options: tuple[tuple[str, ...], ...]
) -> tuple[tuple[str, tuple[str, ...]], ...]:
    """Compatibility alias for focused tests and older internal callers."""

    return physical_action_entities(cards, options)
