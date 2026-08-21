from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime
from typing import Iterable, Literal


Seat = Literal["self", "left", "opposite", "right"]
Phase = Literal["playing", "tribute_give", "tribute_back", "round_over"]

SEATS: tuple[Seat, ...] = ("self", "left", "opposite", "right")
RANKS = ("2", "3", "4", "5", "6", "7", "8", "9", "10", "J", "Q", "K", "A")
SUITS = ("S", "H", "C", "D")
SPECIAL_CARDS = {"small_joker", "big_joker"}


class GameStateError(ValueError):
    """The observed game state cannot safely be sent to a strategy model."""


@dataclass(frozen=True)
class PlayEvent:
    player: Seat
    cards: tuple[str, ...]
    is_pass: bool
    observed_at: datetime
    source: str = "manual"
    # A rank-only observation such as ``8?`` retains colour/suit candidates
    # here.  It is deliberately not resolved in the canonical history.
    suit_options: tuple[tuple[str, ...], ...] = ()
    # 仅用于审计已经确认的动作语义，不参与状态机或规则判定。
    action_metadata: dict[str, object] | None = None

    def to_dict(self) -> dict[str, object]:
        return {
            "player": self.player,
            "cards": list(self.cards),
            "is_pass": self.is_pass,
            "observed_at": self.observed_at.isoformat(),
            "source": self.source,
            "suit_options": [list(options) for options in self.suit_options],
            "action_metadata": (
                dict(self.action_metadata) if self.action_metadata is not None else None
            ),
        }


@dataclass(frozen=True)
class LocalStrategySnapshot:
    """Immutable, confirmed state consumed by the in-process advisor."""

    round_level: str
    wild_rank: str
    phase: Phase
    current_player: Seat | None
    lead_player: Seat | None
    my_hand: tuple[str, ...]
    trick_plays: tuple[PlayEvent, ...]
    play_history: tuple[PlayEvent, ...]
    remaining_cards: dict[Seat, int] | None
    revision: int
    readiness_errors: tuple[str, ...]


@dataclass
class GuanDanState:
    """A manually confirmed public state for the local GuanDan advisor."""

    round_level: str = ""
    wild_rank: str = ""
    phase: Phase = "playing"
    current_player: Seat | None = None
    lead_player: Seat | None = None
    my_hand: tuple[str, ...] = ()
    trick_plays: list[PlayEvent] = field(default_factory=list)
    play_history: list[PlayEvent] = field(default_factory=list)
    remaining_cards: dict[Seat, int] | None = None
    revision: int = 0
    updated_at: datetime = field(
        default_factory=lambda: datetime.now().astimezone()
    )

    def _touch(self) -> None:
        self.revision += 1
        self.updated_at = datetime.now().astimezone()

    @staticmethod
    def _validate_seat(seat: str | None, field_name: str) -> Seat:
        if seat not in SEATS:
            raise GameStateError(f"{field_name} 必须是四个座位之一")
        return seat  # type: ignore[return-value]

    @staticmethod
    def _normalize_cards(cards: Iterable[str]) -> tuple[str, ...]:
        normalized = tuple(sorted(str(card) for card in cards))
        if not normalized:
            raise GameStateError("牌组不能为空")
        invalid = [card for card in normalized if not _is_card_code(card)]
        if invalid:
            raise GameStateError("存在无效牌编码：" + "、".join(invalid))
        return normalized

    @staticmethod
    def _validate_deck_limit(cards: Iterable[str]) -> None:
        values = tuple(str(card) for card in cards)
        exact_cards = Counter(card for card in values if not card.endswith("?"))
        overflow = [card for card, count in exact_cards.items() if count > 2]
        if overflow:
            raise GameStateError("单张牌在双副牌中不能超过两张：" + "、".join(overflow))

        rank_counts = Counter(
            card[:-1]
            for card in values
            if card not in SPECIAL_CARDS
        )
        rank_overflow = [rank for rank, count in rank_counts.items() if count > 8]
        if rank_overflow:
            raise GameStateError("同点数在双副牌中不能超过八张：" + "、".join(rank_overflow))

    def set_context(
        self,
        *,
        round_level: str,
        wild_rank: str,
        current_player: str | None,
        lead_player: str | None,
        phase: Phase = "playing",
    ) -> None:
        if round_level not in RANKS:
            raise GameStateError("请选择当前级牌")
        if wild_rank not in RANKS:
            raise GameStateError("请选择当前百搭牌级别")
        if phase not in {"playing", "tribute_give", "tribute_back", "round_over"}:
            raise GameStateError("未知牌局阶段")
        self.round_level = round_level
        self.wild_rank = wild_rank
        self.phase = phase
        self.current_player = self._validate_seat(
            current_player,
            "当前行动者",
        )
        self.lead_player = self._validate_seat(lead_player, "本轮首出者")
        self._touch()

    def confirm_hand(
        self,
        cards: Iterable[str],
        *,
        source: str = "recognition",
    ) -> None:
        del source
        normalized = self._normalize_cards(cards)
        self._validate_deck_limit(normalized)
        self.my_hand = normalized
        self._touch()

    def set_round_level(self, rank: str) -> None:
        """Set level/wild rank together when the fixed level display is read."""
        if rank not in RANKS:
            raise GameStateError("请选择当前级牌")
        if self.round_level == rank and self.wild_rank == rank:
            return
        self.round_level = rank
        self.wild_rank = rank
        self._touch()

    def set_current_player(self, player: str) -> None:
        seat = self._validate_seat(player, "当前行动者")
        if self.current_player == seat:
            return
        self.current_player = seat
        self._touch()

    def record_play(
        self,
        player: str,
        cards: Iterable[str],
        *,
        source: str = "recognition_confirmed",
        suit_options: Iterable[Iterable[str]] = (),
        action_metadata: dict[str, object] | None = None,
    ) -> PlayEvent:
        seat = self._validate_seat(player, "出牌座位")
        raw_cards = tuple(str(card) for card in cards)
        raw_options = tuple(
            tuple(str(suit) for suit in options) for options in suit_options
        )
        aligned = sorted(
            zip(raw_cards, raw_options + ((),) * max(0, len(raw_cards) - len(raw_options))),
            key=lambda item: item[0],
        )
        normalized = self._normalize_cards(card for card, _options in aligned)
        event = PlayEvent(
            player=seat,
            cards=normalized,
            is_pass=False,
            observed_at=datetime.now().astimezone(),
            source=source,
            suit_options=tuple(options for _card, options in aligned),
            action_metadata=(dict(action_metadata) if action_metadata else None),
        )
        self.trick_plays.append(event)
        self.play_history.append(event)
        self._touch()
        return event

    def record_pass(
        self,
        player: str,
        *,
        source: str = "manual",
    ) -> PlayEvent:
        seat = self._validate_seat(player, "不出座位")
        event = PlayEvent(
            player=seat,
            cards=(),
            is_pass=True,
            observed_at=datetime.now().astimezone(),
            source=source,
        )
        self.trick_plays.append(event)
        self.play_history.append(event)
        self._touch()
        return event

    def start_new_trick(self, leader: str) -> None:
        seat = self._validate_seat(leader, "新一轮首出者")
        self.trick_plays.clear()
        self.lead_player = seat
        self.current_player = seat
        self._touch()

    def reset(self) -> None:
        self.round_level = ""
        self.wild_rank = ""
        self.phase = "playing"
        self.current_player = None
        self.lead_player = None
        self.my_hand = ()
        self.trick_plays.clear()
        self.play_history.clear()
        self.remaining_cards = None
        self._touch()

    @property
    def readiness_errors(self) -> tuple[str, ...]:
        errors: list[str] = []
        if self.phase != "playing":
            errors.append("当前不是常规出牌阶段")
        if self.round_level not in RANKS:
            errors.append("未确认当前级牌")
        if self.wild_rank not in RANKS:
            errors.append("未确认百搭牌级别")
        if self.current_player is None:
            errors.append("未确认当前行动者")
        if self.lead_player is None:
            errors.append("未确认本轮首出者")
        if not self.my_hand:
            errors.append("未确认自己的手牌")
        elif len(self.my_hand) > 27:
            errors.append("自己的手牌超过单人最大数量")
        if self.current_player != "self":
            errors.append("当前不是自己行动")
        return tuple(errors)

    @property
    def is_ready_for_advice(self) -> bool:
        return not self.readiness_errors

    def local_snapshot(self) -> LocalStrategySnapshot:
        if self.readiness_errors:
            raise GameStateError("；".join(self.readiness_errors))
        return LocalStrategySnapshot(
            round_level=self.round_level,
            wild_rank=self.wild_rank,
            phase=self.phase,
            current_player=self.current_player,
            lead_player=self.lead_player,
            my_hand=self.my_hand,
            trick_plays=tuple(self.trick_plays),
            play_history=tuple(self.play_history),
            remaining_cards=(
                dict(self.remaining_cards)
                if self.remaining_cards is not None
                else None
            ),
            revision=self.revision,
            readiness_errors=self.readiness_errors,
        )

    def summary(self) -> str:
        context = (
            f"级牌 {self.round_level or '未设置'}，"
            f"百搭 {self.wild_rank or '未设置'}，"
            f"当前 {self.current_player or '未设置'}，"
            f"首出 {self.lead_player or '未设置'}"
        )
        return (
            f"{context}；手牌 {len(self.my_hand)} 张；"
            f"本轮事件 {len(self.trick_plays)}；"
            f"历史事件 {len(self.play_history)}；"
            f"状态版本 {self.revision}"
        )


def _is_card_code(card: str) -> bool:
    if card in SPECIAL_CARDS:
        return True
    if card.endswith("?") and card[:-1] in RANKS:
        return True
    return any(card == f"{rank}{suit}" for rank in RANKS for suit in SUITS)
