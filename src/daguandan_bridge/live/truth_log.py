from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable

from .models import LiveEvent
from .turns import project_trick_turn

from ..danzero.state import RANKS, SEATS, SUITS, Seat
from ..domain.truth import (
    LabelProvenance,
    LabelStatus,
    TruthEvidence,
    TruthOutcome,
    normalize_label_status,
)
from ..storage import atomic_write_json, load_json_document

_SCHEMA = "guandan.truth/4"
_SCHEMA_VERSION = 4
_SCHEMA_VERSIONS = {"guandan.truth/3": 3, _SCHEMA: _SCHEMA_VERSION}
_SEAT_LABELS = {"self": "自己", "right": "右家", "opposite": "对家", "left": "左家"}
_SUIT_LABELS = {"S": "黑桃", "H": "红桃", "C": "梅花", "D": "方块"}
_LABEL_TO_SUIT = {value: key for key, value in _SUIT_LABELS.items()}


class TruthLogCardInventoryError(ValueError):
    """Raised when a TruthLog cannot fit in one physical double deck."""


def card_code_to_text(card: str) -> str:
    if card == "small_joker":
        return "小王"
    if card == "big_joker":
        return "大王"
    if card.endswith("?"):
        return f"{card[:-1]}？"
    for suit, label in _SUIT_LABELS.items():
        if card.endswith(suit):
            return f"{label}{card[:-1]}"
    return card


def card_text_to_code(value: str) -> str:
    value = str(value).strip()
    if value in {"小王", "small_joker"}:
        return "small_joker"
    if value in {"大王", "big_joker"}:
        return "big_joker"
    for marker in ("?", "？"):
        if value.endswith(marker) and value[:-1] in RANKS:
            return f"{value[:-1]}?"
    for label, suit in _LABEL_TO_SUIT.items():
        if value.startswith(label):
            rank = value[len(label):]
            if rank in RANKS:
                return f"{rank}{suit}"
    if value and value[-1:] in SUITS and value[:-1] in RANKS:
        return value
    raise ValueError(f"无法识别牌面：{value}")


def cards_to_text(cards: Iterable[str]) -> str:
    return "、".join(card_code_to_text(card) for card in cards)


def _normalized_uncertainty(
    cards: tuple[str, ...],
    values: tuple[str, ...],
) -> tuple[str, ...]:
    result = [str(item) for item in values]
    if any(card.endswith("?") for card in cards) and "unknown_suit" not in result:
        result.append("unknown_suit")
    return tuple(result)


@dataclass(frozen=True)
class TruthInitialState:
    round_level: str
    lead_player: Seat
    my_hand: tuple[str, ...]
    # Other seats may start with 25–29 cards because of tribute.  Empty keeps
    # the legacy 27-card default; populated values make strict replay honest.
    seat_hand_sizes: tuple[tuple[str, int], ...] = ()

    def to_dict(self) -> dict[str, object]:
        return {
            "round_level": self.round_level,
            "wild_rank": self.round_level,
            "lead_player": self.lead_player,
            "my_hand": list(self.my_hand),
            **(
                {"seat_hand_sizes": {seat: int(size) for seat, size in self.seat_hand_sizes}}
                if self.seat_hand_sizes
                else {}
            ),
        }


@dataclass(frozen=True, init=False)
class TruthTurn:
    index: int
    actor: Seat
    is_pass: bool
    cards: tuple[str, ...]
    frame_index: int | None = None
    monotonic_ms: int | None = None
    trick_id: int = 0
    evidence: TruthEvidence = TruthEvidence()
    label_status: LabelStatus = "draft"
    provenance: LabelProvenance = LabelProvenance()
    uncertainty: tuple[str, ...] = ()
    move_semantics: dict[str, object] | None = None

    def __init__(
        self,
        index: int,
        actor: Seat | int,
        is_pass: bool | Seat,
        cards: tuple[str, ...] | bool,
        frame_index: int | None = None,
        monotonic_ms: int | None = None,
        legacy_frame_index: int | None = None,
        *,
        trick_id: int | None = None,
        evidence: TruthEvidence | None = None,
        label_status: LabelStatus = "draft",
        provenance: LabelProvenance | None = None,
        uncertainty: tuple[str, ...] = (),
        move_semantics: dict[str, object] | None = None,
    ) -> None:
        # 继续兼容 schema 1 的位置参数构造，同时统一写出 schema 4。
        if isinstance(actor, int) and isinstance(is_pass, str) and isinstance(cards, bool):
            legacy_turn_id = int(index)
            legacy_trick_id = actor
            legacy_actor = is_pass
            legacy_is_pass = cards
            legacy_cards = tuple(frame_index or ()) if isinstance(frame_index, tuple) else ()
            object.__setattr__(self, "index", legacy_turn_id)
            object.__setattr__(self, "actor", legacy_actor)  # type: ignore[arg-type]
            object.__setattr__(self, "is_pass", legacy_is_pass)
            object.__setattr__(self, "cards", legacy_cards)
            object.__setattr__(self, "frame_index", legacy_frame_index)
            object.__setattr__(self, "monotonic_ms", monotonic_ms)
            object.__setattr__(self, "trick_id", max(1, legacy_trick_id))
            object.__setattr__(
                self,
                "evidence",
                evidence
                or TruthEvidence(
                    frame_indices=(legacy_frame_index,) if legacy_frame_index is not None else (),
                    monotonic_ms=monotonic_ms,
                ),
            )
            object.__setattr__(self, "label_status", normalize_label_status(label_status))
            object.__setattr__(self, "provenance", provenance or LabelProvenance())
            object.__setattr__(self, "uncertainty", _normalized_uncertainty(legacy_cards, uncertainty))
            object.__setattr__(self, "move_semantics", _normalize_move_semantics(move_semantics))
            return
        object.__setattr__(self, "index", int(index))
        object.__setattr__(self, "actor", actor)
        object.__setattr__(self, "is_pass", bool(is_pass))
        object.__setattr__(self, "cards", tuple(cards))
        object.__setattr__(self, "frame_index", frame_index)
        object.__setattr__(self, "monotonic_ms", monotonic_ms)
        object.__setattr__(self, "trick_id", max(0, int(trick_id or 0)))
        object.__setattr__(
            self,
            "evidence",
            evidence
            or TruthEvidence(
                frame_indices=(frame_index,) if frame_index is not None else (),
                monotonic_ms=monotonic_ms,
            ),
        )
        object.__setattr__(self, "label_status", normalize_label_status(label_status))
        object.__setattr__(self, "provenance", provenance or LabelProvenance())
        object.__setattr__(self, "uncertainty", _normalized_uncertainty(tuple(cards), uncertainty))
        object.__setattr__(
            self,
            "move_semantics",
            None if bool(is_pass) else _normalize_move_semantics(move_semantics),
        )

    @property
    def turn_id(self) -> int:
        return self.index

    def to_dict(self) -> dict[str, object]:
        raw: dict[str, object] = {
            "turn_id": self.index,
            "trick_id": self.trick_id,
            "actor": self.actor,
            "is_pass": self.is_pass,
            "cards": list(self.cards),
            "evidence": self.evidence.to_dict(),
            "label_status": self.label_status,
            "provenance": self.provenance.to_dict(),
            "uncertainty": list(self.uncertainty),
        }
        if self.move_semantics is not None:
            raw["move_semantics"] = _json_safe_semantics(self.move_semantics)
        return raw


@dataclass(frozen=True)
class TruthLog:
    source_session_id: str
    initial_state: TruthInitialState
    turns: tuple[TruthTurn, ...]
    source_video: str = "video/game.avi"
    frame_index_path: str = "video/frame_index.jsonl"
    label_status: LabelStatus = "draft"
    provenance: LabelProvenance = LabelProvenance()
    outcome: TruthOutcome = TruthOutcome()

    def __post_init__(self) -> None:
        object.__setattr__(self, "label_status", normalize_label_status(self.label_status))
        object.__setattr__(
            self,
            "turns",
            _with_inferred_trick_ids(self.turns, len(self.initial_state.my_hand)),
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "schema": _SCHEMA,
            "schema_version": _SCHEMA_VERSION,
            "source_session_id": self.source_session_id,
            "source_video": {
                "path": self.source_video,
                "frame_index_path": self.frame_index_path,
            },
            "label_status": self.label_status,
            "provenance": self.provenance.to_dict(),
            "initial_state": self.initial_state.to_dict(),
            "outcome": self.outcome.to_dict(),
            "turns": [turn.to_dict() for turn in self.turns],
        }

    def to_events(self, *, session_id: str = "truth-log") -> tuple[LiveEvent, ...]:
        events: list[LiveEvent] = [
            LiveEvent(
                event_id="TRUTH-000000",
                event_type="initial_state_confirmed",
                session_id=session_id,
                seq=0,
                monotonic_ms=0,
                wall_time=datetime.now().astimezone().isoformat(),
                trick_id=1,
                turn_id=0,
                actor=self.initial_state.lead_player,
                payload={
                    "round_level": self.initial_state.round_level,
                    "wild_rank": self.initial_state.round_level,
                    "hand": list(self.initial_state.my_hand),
                    "lead_player": self.initial_state.lead_player,
                    **(
                        {"seat_hand_sizes": {seat: int(size) for seat, size in self.initial_state.seat_hand_sizes}}
                        if self.initial_state.seat_hand_sizes
                        else {}
                    ),
                },
                confidence=1.0,
                source="truth_log",
                state_revision_before=0,
                state_revision_after=1,
            )
        ]
        for turn in self.turns:
            events.append(
                LiveEvent(
                    event_id=f"TRUTH-{turn.index:06d}",
                    event_type="player_passed" if turn.is_pass else "player_played",
                    session_id=session_id,
                    seq=turn.index,
                    monotonic_ms=turn.monotonic_ms or turn.index,
                    wall_time=datetime.now().astimezone().isoformat(),
                    trick_id=turn.trick_id,
                    turn_id=turn.index,
                    actor=turn.actor,
                    payload={
                        "cards": list(turn.cards),
                        "physical_cards": list(turn.cards),
                        "is_pass": turn.is_pass,
                        **(
                            {"move_semantics": _json_safe_semantics(turn.move_semantics)}
                            if turn.move_semantics is not None
                            else {}
                        ),
                    },
                    confidence=1.0,
                    source="truth_log",
                    state_revision_before=turn.index,
                    state_revision_after=turn.index + 1,
                )
            )
        return tuple(events)


def _physical_rank(card: str) -> str:
    if card in {"small_joker", "big_joker"}:
        return card
    return card[:-1]


def _physical_suit(card: str) -> str | None:
    return card[-1] if card[-1:] in SUITS else None


def _is_exact_physical_card(card: str) -> bool:
    return card in {"small_joker", "big_joker"} or card[-1:] in SUITS


def _turn_references(
    turns: Iterable[TruthTurn],
    *,
    card: str | None = None,
    rank: str | None = None,
    suit: str | None = None,
) -> str:
    indices: list[int] = []
    for turn in turns:
        if turn.is_pass:
            continue
        matched = any(
            (card is not None and value == card)
            or (rank is not None and _physical_rank(value) == rank)
            or (suit is not None and _physical_suit(value) == suit)
            for value in turn.cards
        )
        if matched:
            indices.append(int(turn.index))
    if not indices:
        return ""
    rendered = "、".join(str(value) for value in indices[:12])
    if len(indices) > 12:
        rendered += f" 等 {len(indices)} 条"
    return f"；涉及第 {rendered} 条动作"


def _rank_label(rank: str) -> str:
    if rank == "small_joker":
        return "小王"
    if rank == "big_joker":
        return "大王"
    return rank


def validate_truth_log_card_inventory(log: TruthLog) -> None:
    """Reject a TruthLog that cannot fit in a physical double deck.

    ``my_hand`` is the complete initial hand, so self plays are validated as a
    subset of that hand instead of being counted a second time.  The global
    physical ledger is therefore ``my_hand + all opponents' played cards``.
    This catches impossible third copies even before the last copy is played by
    self, while avoiding false positives from counting self's cards twice.

    Double-deck limits are checked at all useful levels:

    * one exact rank+suit card (and each Joker): at most 2 copies;
    * one ordinary rank across four suits: at most 8 copies;
    * one suit across thirteen ranks: at most 26 copies.

    Unknown-suit cards still consume their rank allowance but do not consume an
    arbitrary exact-card or suit slot until their suit is confirmed.
    """

    if not isinstance(log, TruthLog):
        raise TypeError("log must be a TruthLog")

    hand = tuple(card_text_to_code(str(card)) for card in log.initial_state.my_hand)
    self_turns = tuple(
        turn for turn in log.turns if not turn.is_pass and turn.actor == "self"
    )
    opponent_turns = tuple(
        turn for turn in log.turns if not turn.is_pass and turn.actor != "self"
    )
    self_cards = tuple(
        card_text_to_code(str(card)) for turn in self_turns for card in turn.cards
    )
    opponent_cards = tuple(
        card_text_to_code(str(card))
        for turn in opponent_turns
        for card in turn.cards
    )

    hand_exact = Counter(card for card in hand if _is_exact_physical_card(card))
    self_exact = Counter(
        card for card in self_cards if _is_exact_physical_card(card)
    )
    opponent_exact = Counter(
        card for card in opponent_cards if _is_exact_physical_card(card)
    )
    observed_exact = hand_exact + opponent_exact

    hand_ranks = Counter(_physical_rank(card) for card in hand)
    self_ranks = Counter(_physical_rank(card) for card in self_cards)
    opponent_ranks = Counter(_physical_rank(card) for card in opponent_cards)
    observed_ranks = hand_ranks + opponent_ranks

    hand_suits = Counter(
        suit for card in hand if (suit := _physical_suit(card)) is not None
    )
    opponent_suits = Counter(
        suit
        for card in opponent_cards
        if (suit := _physical_suit(card)) is not None
    )
    observed_suits = hand_suits + opponent_suits
    hand_unknown_ranks = Counter(
        _physical_rank(card) for card in hand if card.endswith("?")
    )

    violations: list[str] = []

    for card, count in sorted(observed_exact.items()):
        if count <= 2:
            continue
        references = _turn_references(opponent_turns, card=card)
        violations.append(
            f"牌面 {card_code_to_text(card)}（{card}）共 {count} 张："
            f"我方初始手牌 {hand_exact[card]} 张，其他玩家历史出牌 "
            f"{opponent_exact[card]} 张；双副牌最多 2 张{references}"
        )

    ordered_ranks = (*RANKS, "small_joker", "big_joker")
    for rank in ordered_ranks:
        count = observed_ranks[rank]
        limit = 2 if rank in {"small_joker", "big_joker"} else 8
        if count <= limit:
            continue
        references = _turn_references(opponent_turns, rank=rank)
        violations.append(
            f"点数 {_rank_label(rank)} 共 {count} 张：我方初始手牌 "
            f"{hand_ranks[rank]} 张，其他玩家历史出牌 {opponent_ranks[rank]} 张；"
            f"双副牌最多 {limit} 张{references}"
        )

    for suit in SUITS:
        count = observed_suits[suit]
        if count <= 26:
            continue
        references = _turn_references(opponent_turns, suit=suit)
        violations.append(
            f"花色 {_SUIT_LABELS[suit]} 共 {count} 张：我方初始手牌 "
            f"{hand_suits[suit]} 张，其他玩家历史出牌 {opponent_suits[suit]} 张；"
            f"双副牌最多 26 张{references}"
        )

    for rank in ordered_ranks:
        if self_ranks[rank] <= hand_ranks[rank]:
            continue
        references = _turn_references(self_turns, rank=rank)
        violations.append(
            f"自己累计打出点数 {_rank_label(rank)} {self_ranks[rank]} 张，"
            f"但初始手牌只有 {hand_ranks[rank]} 张{references}"
        )

    for rank in RANKS:
        deficits = {
            card: max(0, self_exact[card] - hand_exact[card])
            for card in (f"{rank}{suit}" for suit in SUITS)
        }
        required_unknowns = sum(deficits.values())
        if required_unknowns <= hand_unknown_ranks[rank]:
            continue
        detail = "、".join(
            f"{card_code_to_text(card)}缺 {count} 张"
            for card, count in deficits.items()
            if count
        )
        references = _turn_references(self_turns, rank=rank)
        violations.append(
            f"自己打出的点数 {rank} 与初始手牌花色不一致：{detail}；"
            f"初始手牌只有 {hand_unknown_ranks[rank]} 张未知花色可用于匹配"
            f"{references}"
        )

    for joker in ("small_joker", "big_joker"):
        if self_exact[joker] <= hand_exact[joker]:
            continue
        references = _turn_references(self_turns, card=joker)
        violations.append(
            f"自己累计打出{card_code_to_text(joker)} {self_exact[joker]} 张，"
            f"但初始手牌只有 {hand_exact[joker]} 张{references}"
        )

    if violations:
        visible = violations[:10]
        if len(violations) > len(visible):
            visible.append(f"另有 {len(violations) - len(visible)} 项牌库数量冲突")
        raise TruthLogCardInventoryError(
            "TruthLog 牌库数量校验未通过：\n- " + "\n- ".join(visible)
        )


def truth_log_from_dict(raw: dict[str, Any]) -> TruthLog:
    schema = str(raw.get("schema", ""))
    if schema and schema not in _SCHEMA_VERSIONS:
        raise ValueError(f"unsupported truth schema: {schema}")
    if schema and int(raw.get("schema_version", _SCHEMA_VERSIONS[schema])) != _SCHEMA_VERSIONS[schema]:
        raise ValueError("truth schema and schema_version conflict")
    version = _SCHEMA_VERSIONS[schema] if schema else int(raw.get("schema_version", 1))
    if version not in {1, 2, 3, 4}:
        raise ValueError("不支持的标准日志版本")
    session_id = str(raw.get("source_session_id", raw.get("session_id", ""))).strip()
    if not session_id:
        raise ValueError("标准日志缺少源对局 ID")
    initial = raw.get("initial_state")
    if not isinstance(initial, dict):
        raise ValueError("标准日志缺少初始状态")
    lead = str(initial.get("lead_player", ""))
    if lead not in SEATS:
        raise ValueError("标准日志中的首出座位无效")
    hand = _normalize_cards(initial.get("my_hand"), "初始手牌")
    turns_raw = raw.get("turns", [])
    if not isinstance(turns_raw, list):
        raise ValueError("标准日志中的出牌链必须是数组")
    turns: list[TruthTurn] = []
    for position, item in enumerate(turns_raw, start=1):
        if not isinstance(item, dict):
            raise ValueError(f"第 {position} 条动作不是对象")
        index = int(item.get("index", item.get("turn_id", position)))
        if index != position:
            raise ValueError("出牌链序号必须从 1 连续编号")
        actor = str(item.get("actor", ""))
        if actor not in SEATS:
            raise ValueError(f"第 {index} 条动作的玩家无效")
        is_pass = bool(item.get("is_pass", False))
        cards = _normalize_cards(item.get("cards", ()), f"第 {index} 条牌面", allow_empty=True)
        if is_pass and cards:
            raise ValueError(f"第 {index} 条不出动作不能有牌面")
        if not is_pass and not cards:
            raise ValueError(f"第 {index} 条出牌动作必须填写牌面")
        evidence_raw = item.get("evidence")
        if isinstance(evidence_raw, dict):
            evidence = TruthEvidence.from_dict(evidence_raw)
        else:
            legacy_frame = _optional_int(item.get("frame_index"))
            legacy_ms = _optional_int(item.get("monotonic_ms"))
            evidence = TruthEvidence(
                frame_indices=(legacy_frame,) if legacy_frame is not None else (),
                monotonic_ms=legacy_ms,
            )
        uncertainty_raw = item.get("uncertainty", ())
        if not isinstance(uncertainty_raw, (list, tuple)):
            raise ValueError(f"turn {index} uncertainty must be an array")
        turns.append(
            TruthTurn(
                index=index,
                actor=actor,  # type: ignore[arg-type]
                is_pass=is_pass,
                cards=cards,
                frame_index=evidence.frame_indices[0] if evidence.frame_indices else None,
                monotonic_ms=evidence.monotonic_ms,
                trick_id=_optional_int(item.get("trick_id")),
                evidence=evidence,
                label_status=normalize_label_status(item.get("label_status", "draft")),
                provenance=LabelProvenance.from_dict(
                    item.get("provenance"),
                    default_source="" if version >= 3 else "legacy_migration",
                ),
                uncertainty=tuple(str(value) for value in uncertainty_raw),
                move_semantics=item.get("move_semantics"),  # type: ignore[arg-type]
            )
        )
    raw_sizes = initial.get("seat_hand_sizes", {})
    seat_hand_sizes: tuple[tuple[str, int], ...] = ()
    if isinstance(raw_sizes, dict):
        parsed_sizes = []
        for seat, value in raw_sizes.items():
            if str(seat) not in SEATS:
                raise ValueError("标准日志中的座位起手牌数无效")
            size = int(value)
            if not 0 <= size <= 54:
                raise ValueError("标准日志中的座位起手牌数超出范围")
            parsed_sizes.append((str(seat), size))
        seat_hand_sizes = tuple(sorted(parsed_sizes))
    source = raw.get("source_video") or {}
    if not isinstance(source, dict):
        raise ValueError("标准日志录像路径无效")
    return TruthLog(
        source_session_id=session_id,
        initial_state=TruthInitialState(
            round_level=str(initial.get("round_level", initial.get("wild_rank", ""))),
            lead_player=lead,  # type: ignore[arg-type]
            my_hand=hand,
            seat_hand_sizes=seat_hand_sizes,
        ),
        turns=tuple(turns),
        source_video=str(source.get("path", "video/game.avi")),
        frame_index_path=str(source.get("frame_index_path", "video/frame_index.jsonl")),
        label_status=normalize_label_status(raw.get("label_status", "draft")),
        provenance=LabelProvenance.from_dict(
            raw.get("provenance"),
            default_source="" if version >= 3 else "legacy_migration",
        ),
        outcome=TruthOutcome.from_dict(raw.get("outcome")),
    )


def load_truth_log(path: Path, *, session_id: str | None = None) -> TruthLog:
    raw = load_json_document(path, {})
    if (
        session_id
        and isinstance(raw, dict)
        and not str(raw.get("source_session_id") or "").strip()
        and not str(raw.get("session_id") or "").strip()
    ):
        raw = {**raw, "source_session_id": session_id}
    log = truth_log_from_dict(raw)
    if session_id is not None and log.source_session_id != session_id:
        raise ValueError("标准日志不属于当前选择的对局")
    return log


def save_truth_log(path: Path, log: TruthLog) -> None:
    normalized = truth_log_from_dict(log.to_dict())
    # Diagnostic scan drafts may intentionally preserve impossible visual
    # observations for later human repair.  Canonical verified writes may not.
    if normalized.label_status == "verified":
        validate_truth_log_card_inventory(normalized)
    atomic_write_json(path, normalized.to_dict())


def _with_inferred_trick_ids(
    turns: tuple[TruthTurn, ...],
    self_starting_cards: int,
) -> tuple[TruthTurn, ...]:
    """Fill absent trick ids with the same pass-cycle semantics as the reducer."""

    current_trick = 1
    leader: Seat | None = None
    passed: set[Seat] = set()
    played = {seat: 0 for seat in SEATS}
    starting = {seat: 27 for seat in SEATS}
    starting["self"] = self_starting_cards
    result: list[TruthTurn] = []
    for turn in turns:
        if turn.trick_id > 0:
            current_trick = turn.trick_id
        assigned = current_trick
        result.append(
            TruthTurn(
                turn.index,
                turn.actor,
                turn.is_pass,
                turn.cards,
                frame_index=turn.frame_index,
                monotonic_ms=turn.monotonic_ms,
                trick_id=assigned,
                evidence=turn.evidence,
                label_status=turn.label_status,
                provenance=turn.provenance,
                uncertainty=turn.uncertainty,
                move_semantics=turn.move_semantics,
            )
        )
        if turn.is_pass:
            if leader is not None:
                passed.add(turn.actor)
        else:
            leader = turn.actor
            passed.clear()
            played[turn.actor] += len(turn.cards)
        finished = {
            seat for seat in SEATS if played[seat] >= max(1, starting[seat])
        }
        if leader is not None:
            projection = project_trick_turn(leader, finished, passed)
            if projection.is_complete:
                current_trick = assigned + 1
                leader = None
                passed.clear()
    return tuple(result)


def _normalize_cards(value: object, label: str, *, allow_empty: bool = False) -> tuple[str, ...]:
    if not isinstance(value, Iterable) or isinstance(value, (str, bytes)):
        raise ValueError(f"{label}必须是牌面数组")
    cards = tuple(card_text_to_code(str(card)) for card in value)
    if not allow_empty and not cards:
        raise ValueError(f"{label}不能为空")
    overflow = [card for card, count in Counter(cards).items() if count > 2]
    if overflow:
        raise ValueError(f"{label}中单张牌不能超过两张")
    return cards


def _optional_int(value: object) -> int | None:
    if value is None or value == "":
        return None
    parsed = int(value)
    if parsed < 0:
        raise ValueError("定位信息不能为负数")
    return parsed


def _normalize_move_semantics(value: object) -> dict[str, object] | None:
    if value is None:
        return None
    if not isinstance(value, dict):
        raise ValueError("move_semantics 必须是 JSON 对象")
    normalized = _json_safe_semantics(value)
    source = normalized.get("selection_source")
    if source is not None and source not in {
        "realtime_semantics",
        "ui_detection",
        "exact_engine_state",
        "inferred_unique",
        "unresolved",
    }:
        raise ValueError(f"不支持的 wildcard 语义来源：{source}")
    assignments = normalized.get("wildcard_assignments")
    if assignments is not None and not isinstance(assignments, list):
        raise ValueError("wildcard_assignments 必须是数组")
    candidates = normalized.get("candidate_interpretations")
    if candidates is not None and not isinstance(candidates, list):
        raise ValueError("candidate_interpretations 必须是数组")
    selected = normalized.get("selected_interpretation")
    if selected is not None and not isinstance(selected, dict):
        raise ValueError("selected_interpretation 必须是对象或 null")
    return normalized


def _json_safe_semantics(value: object) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_safe_semantics(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe_semantics(item) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    raise ValueError(f"move_semantics 包含不可序列化值：{type(value).__name__}")
