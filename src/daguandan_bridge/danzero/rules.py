from __future__ import annotations

from collections import Counter
from dataclasses import dataclass

_PROJECT_SUITS = frozenset(("S", "H", "C", "D"))
_PROJECT_RANKS = frozenset(("2", "3", "4", "5", "6", "7", "8", "9", "10", "J", "Q", "K", "A"))
_ENGINE_RANKS = frozenset(("2", "3", "4", "5", "6", "7", "8", "9", "T", "J", "Q", "K", "A"))


def to_engine_card(code: str) -> str:
    """Convert project card notation (``AS``/``10D``) to engine notation."""
    normalized = str(code)
    if normalized == "small_joker":
        return "SB"
    if normalized == "big_joker":
        return "HR"
    rank, suit = normalized[:-1], normalized[-1:]
    if rank not in _PROJECT_RANKS or suit not in _PROJECT_SUITS:
        raise ValueError(f"无效项目牌编码：{code}")
    return suit + ("T" if rank == "10" else rank)


def from_engine_card(code: str) -> str:
    """Convert engine card notation (``SA``/``DT``) to project notation."""
    normalized = str(code)
    if normalized == "SB":
        return "small_joker"
    if normalized == "HR":
        return "big_joker"
    suit, rank = normalized[:1], normalized[1:]
    if suit not in _PROJECT_SUITS or rank not in _ENGINE_RANKS:
        raise ValueError(f"无效引擎牌编码：{code}")
    return ("10" if rank == "T" else rank) + suit


@dataclass(frozen=True)
class ActionInference:
    """Best logical interpretation of one already observed physical play.

    ``cards`` always remains the physical card set seen by the recognizer.
    ``action`` is only an interpretation used by DanZero and comparison code;
    it must never be used to replace the physical cards in the reducer.
    """

    cards: tuple[str, ...]
    action: list[object] | None
    beats_table: bool
    ambiguous: bool = False
    logical_label: str = ""
    wildcard_substitutions: tuple[tuple[str, str], ...] = ()
    candidate_actions: tuple[list[object], ...] = ()


def actions_for_cards(cards: tuple[str, ...], level_rank: str) -> list[list[object]]:
    """Return every engine action consuming exactly ``cards``.

    A level wildcard can produce more than one action from the same physical
    cards (for example ``JJ9H22`` can be ``JJJ22`` or ``222JJ``).  The old
    implementation returned the first match, which made the result depend on
    generator order and could reject a play that actually beat the table.
    """
    if not cards:
        return []
    from daguandan_bridge.danzero._vendor.guandan_rlcard.constants import CARD_RANK
    from daguandan_bridge.danzero._vendor.guandan_rlcard.game.card_utils import card_from_str
    from daguandan_bridge.danzero._vendor.guandan_rlcard.game.judger import GuandanJudger

    level = "T" if level_rank == "10" else str(level_rank)
    if level not in CARD_RANK:
        raise ValueError(f"无效级牌：{level_rank}")
    level_index = CARD_RANK.index(level)
    engine_cards = [to_engine_card(card) for card in cards]
    candidates = GuandanJudger.playable_actions_from_hand(
        [card_from_str(card) for card in engine_cards],
        level_index,
    )
    expected = Counter(engine_cards)
    matches: list[list[object]] = []
    seen: set[tuple[object, object, tuple[str, ...]]] = set()
    for action in candidates:
        if Counter(str(card) for card in action[2]) != expected:
            continue
        key = (action[0], action[1], tuple(sorted(str(card) for card in action[2])))
        if key in seen:
            continue
        seen.add(key)
        matches.append(action)
    return matches


def _action_strength(action: list[object], level_rank: str) -> tuple[object, ...]:
    """Sort equivalent physical interpretations from weakest to strongest."""

    from daguandan_bridge.danzero._vendor.guandan_rlcard.constants import (
        CARD_RANK_INDEX,
        NATURAL_SEQ_VALUE,
        make_card_values,
    )

    type_order = {
        "PASS": 0,
        "Single": 1,
        "Pair": 2,
        "Trips": 3,
        "ThreePair": 4,
        "ThreeWithTwo": 5,
        "TwoTrips": 6,
        "Straight": 7,
        "Bomb": 8,
        "StraightFlush": 9,
    }
    key_rank = str(action[1])
    if action[0] in {"Straight", "StraightFlush", "ThreePair", "TwoTrips"}:
        rank_value = NATURAL_SEQ_VALUE.get(key_rank, -1)
    else:
        values = make_card_values("T" if level_rank == "10" else level_rank)
        rank_value = values.get(key_rank, CARD_RANK_INDEX.get(key_rank, -1))
    return (
        type_order.get(str(action[0]), -1),
        int(rank_value),
        len(action[2]),
        tuple(sorted(str(card) for card in action[2])),
    )


def _logical_ranks(action: list[object], level_rank: str) -> tuple[str, ...]:
    """Return the declared rank of every physical card in one engine action."""

    from daguandan_bridge.danzero._vendor.guandan_rlcard.constants import CARD_RANK

    play_type = str(action[0])
    key_rank = str(action[1])
    card_count = len(action[2])
    ranks = CARD_RANK[:13]

    if play_type == "ThreeWithTwo":
        wildcard = "H" + ("T" if level_rank == "10" else str(level_rank))
        pair_cards = tuple(str(card) for card in action[2][3:])
        pair_rank = next(
            (
                card[-1]
                for card in pair_cards
                if card != wildcard and card not in {"SB", "HR"}
            ),
            next(
                ("B" if card == "SB" else "R" for card in pair_cards if card in {"SB", "HR"}),
                "T" if level_rank == "10" else str(level_rank),
            ),
        )
        return (key_rank,) * 3 + (pair_rank,) * 2
    if play_type == "ThreePair":
        start = ranks.index(key_rank)
        sequence = tuple(ranks[(start + offset) % len(ranks)] for offset in range(3))
        return tuple(rank for rank in sequence for _ in range(2))
    if play_type == "TwoTrips":
        start = ranks.index(key_rank)
        sequence = (ranks[start], ranks[(start + 1) % len(ranks)])
        return tuple(rank for rank in sequence for _ in range(3))
    if play_type in {"Straight", "StraightFlush"}:
        if key_rank == "A":
            return ("A", "2", "3", "4", "5")
        start = ranks.index(key_rank)
        return tuple(ranks[start + offset] for offset in range(5))
    if play_type == "PASS":
        return ()
    return (key_rank,) * card_count


def _project_rank(rank: str) -> str:
    return {"T": "10", "B": "小王", "R": "大王"}.get(rank, rank)


def logical_action_label(action: list[object] | None, level_rank: str) -> str:
    """Format an engine declaration without changing its physical cards."""

    if action is None:
        return ""
    return "".join(_project_rank(rank) for rank in _logical_ranks(action, level_rank))


def wildcard_substitutions(
    action: list[object] | None,
    level_rank: str,
) -> tuple[tuple[str, str], ...]:
    """Describe only substitutions where the heart level card changes rank."""

    if action is None:
        return ()
    engine_level = "T" if level_rank == "10" else str(level_rank)
    wildcard = "H" + engine_level
    mappings: list[tuple[str, str]] = []
    for card, declared_rank in zip(action[2], _logical_ranks(action, level_rank)):
        if str(card) == wildcard and declared_rank != engine_level:
            mappings.append((from_engine_card(wildcard), _project_rank(declared_rank)))
    return tuple(mappings)


def action_for_cards(cards: tuple[str, ...], level_rank: str) -> list[object] | None:
    """Return the strongest deterministic interpretation of ``cards``."""

    actions = actions_for_cards(cards, level_rank)
    if not actions:
        return None
    return max(actions, key=lambda action: _action_strength(action, level_rank))


def _action_beats(
    candidate: list[object],
    current: list[object],
    level_rank: str,
) -> bool:
    from types import SimpleNamespace

    from daguandan_bridge.danzero._vendor.guandan_rlcard.constants import CARD_RANK
    from daguandan_bridge.danzero._vendor.guandan_rlcard.game.action_compare import (
        get_gt_actions,
    )

    level = "T" if level_rank == "10" else str(level_rank)
    greater_player = SimpleNamespace(played_action=current)
    greater = get_gt_actions(CARD_RANK.index(level), greater_player, [candidate])
    return candidate in greater


def infer_best_action(
    cards: tuple[str, ...],
    table_cards: tuple[str, ...],
    level_rank: str,
    *,
    preferred_play_type: str | None = None,
) -> ActionInference:
    """Infer the strongest interpretation without vetoing an observed play.

    The table comparison is a ranking signal.  A visual action that has
    already been observed remains committable even if every interpretation is
    weaker than the reconstructed table; that situation is returned as an
    audit warning instead of being converted into PASS.
    """

    actions = actions_for_cards(cards, level_rank)
    all_actions = tuple(actions)
    table_actions = actions_for_cards(table_cards, level_rank) if table_cards else []
    if not table_cards:
        beating = list(actions)
    elif table_actions:
        beating = [
            action
            for action in actions
            if any(_action_beats(action, current, level_rank) for current in table_actions)
        ]
    else:
        beating = []
    pool = beating or actions
    if preferred_play_type:
        preferred = [action for action in pool if action[0] == preferred_play_type]
        if preferred:
            pool = preferred
    best = max(pool, key=lambda action: _action_strength(action, level_rank)) if pool else None
    return ActionInference(
        cards=tuple(cards),
        action=best,
        beats_table=bool(best is not None and best in beating),
        ambiguous=len(all_actions) > 1,
        logical_label=logical_action_label(best, level_rank),
        wildcard_substitutions=wildcard_substitutions(best, level_rank),
        candidate_actions=all_actions,
    )


def play_beats_table(
    cards: tuple[str, ...],
    table_cards: tuple[str, ...],
    level_rank: str,
) -> bool:
    """Return whether one already-validated card set beats the table action."""

    if not table_cards:
        return True
    candidate_actions = actions_for_cards(cards, level_rank)
    current_actions = actions_for_cards(table_cards, level_rank)
    if not candidate_actions or not current_actions:
        return False
    return any(
        _action_beats(candidate, current, level_rank)
        for candidate in candidate_actions
        for current in current_actions
    )
