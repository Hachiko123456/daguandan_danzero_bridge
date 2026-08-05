from __future__ import annotations

from collections import Counter

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


def action_for_cards(cards: tuple[str, ...], level_rank: str) -> list[object] | None:
    """Return the engine action for exactly ``cards``, or ``None`` if invalid."""
    if not cards:
        return None
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
    return next(
        (
            action
            for action in candidates
            if Counter(str(card) for card in action[2]) == expected
        ),
        None,
    )
