from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable

from .models import LiveEvent

from ..danzero.state import RANKS, SEATS, SUITS, Seat
from ..domain.truth import (
    LabelProvenance,
    LabelStatus,
    TruthEvidence,
    TruthOutcome,
    normalize_label_status,
)
from ..storage import atomic_write_json, load_json_document

_SCHEMA = "guandan.truth/3"
_SCHEMA_VERSION = 3
_SEAT_LABELS = {"self": "自己", "right": "右家", "opposite": "对家", "left": "左家"}
_SUIT_LABELS = {"S": "黑桃", "H": "红桃", "C": "梅花", "D": "方块"}
_LABEL_TO_SUIT = {value: key for key, value in _SUIT_LABELS.items()}


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

    def to_dict(self) -> dict[str, object]:
        return {
            "round_level": self.round_level,
            "wild_rank": self.round_level,
            "lead_player": self.lead_player,
            "my_hand": list(self.my_hand),
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
    ) -> None:
        # Accept schema-1 positional construction while writing schema 3.
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

    @property
    def turn_id(self) -> int:
        return self.index

    def to_dict(self) -> dict[str, object]:
        return {
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
                    payload={"cards": list(turn.cards), "is_pass": turn.is_pass},
                    confidence=1.0,
                    source="truth_log",
                    state_revision_before=turn.index,
                    state_revision_after=turn.index + 1,
                )
            )
        return tuple(events)


def truth_log_from_dict(raw: dict[str, Any]) -> TruthLog:
    schema = str(raw.get("schema", ""))
    if schema and schema != _SCHEMA:
        raise ValueError(f"unsupported truth schema: {schema}")
    if schema == _SCHEMA and int(raw.get("schema_version", 3)) != 3:
        raise ValueError("truth schema and schema_version conflict")
    version = 3 if schema == _SCHEMA else int(raw.get("schema_version", 1))
    if version not in {1, 2, 3}:
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
                    default_source="" if version == 3 else "legacy_migration",
                ),
                uncertainty=tuple(str(value) for value in uncertainty_raw),
            )
        )
    source = raw.get("source_video") or {}
    if not isinstance(source, dict):
        raise ValueError("标准日志录像路径无效")
    return TruthLog(
        source_session_id=session_id,
        initial_state=TruthInitialState(
            round_level=str(initial.get("round_level", initial.get("wild_rank", ""))),
            lead_player=lead,  # type: ignore[arg-type]
            my_hand=hand,
        ),
        turns=tuple(turns),
        source_video=str(source.get("path", "video/game.avi")),
        frame_index_path=str(source.get("frame_index_path", "video/frame_index.jsonl")),
        label_status=normalize_label_status(raw.get("label_status", "draft")),
        provenance=LabelProvenance.from_dict(
            raw.get("provenance"),
            default_source="" if version == 3 else "legacy_migration",
        ),
        outcome=TruthOutcome.from_dict(raw.get("outcome")),
    )


def load_truth_log(path: Path, *, session_id: str | None = None) -> TruthLog:
    log = truth_log_from_dict(load_json_document(path, {}))
    if session_id is not None and log.source_session_id != session_id:
        raise ValueError("标准日志不属于当前选择的对局")
    return log


def save_truth_log(path: Path, log: TruthLog) -> None:
    atomic_write_json(path, truth_log_from_dict(log.to_dict()).to_dict())


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
        active = set(SEATS) - finished
        if leader is not None:
            required = active - {leader}
            if required and required.issubset(passed):
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
