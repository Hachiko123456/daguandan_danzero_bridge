from __future__ import annotations

from collections import Counter
from dataclasses import replace
from datetime import datetime
from time import monotonic_ns
from typing import Iterable

from ..danzero.state import (
    GameStateError,
    GuanDanState,
    PlayEvent,
    RANKS,
    Seat,
)
from .models import LiveEvent, LiveSnapshot
from .card_uncertainty import normalized_suit_options
from .turns import (
    TURN_ORDER,
    next_active_seat,
    project_trick_turn,
    round_is_decided,
)


_ACTION_EVENT_TYPES = {
    "player_played",
    "player_passed",
    "manual_confirmed_event",
}

class LiveReducer:
    """Deterministically reduce immutable live events into confirmed game state."""

    def __init__(
        self,
        session_id: str,
        *,
        wind_receiver_must_pass: bool = True,
    ) -> None:
        if not str(session_id).strip():
            raise ValueError("session_id 不能为空")
        self.session_id = str(session_id)
        self._wind_receiver_must_pass = wind_receiver_must_pass
        self._events: list[LiveEvent] = []
        self._reset_semantic_state()

    @property
    def events(self) -> tuple[LiveEvent, ...]:
        return tuple(self._events)

    def _reset_semantic_state(self) -> None:
        self._round_level = ""
        self._wild_rank = ""
        self._current_player: Seat | None = None
        self._lead_player: Seat | None = None
        self._my_hand: tuple[str, ...] = ()
        self._trick_plays: list[PlayEvent] = []
        self._play_history: list[PlayEvent] = []
        self._remaining_cards: dict[Seat, int] = {
            seat: 27 for seat in TURN_ORDER
        }
        self._finished_seats: set[Seat] = set()
        self._trick_id = 0
        self._turn_id = 0
        self._revision = 0
        self._initialized = False

    @staticmethod
    def _normalize_cards(cards: Iterable[str]) -> tuple[str, ...]:
        state = GuanDanState()
        state.confirm_hand(cards)
        return state.my_hand

    def _new_event(
        self,
        event_type: str,
        *,
        actor: Seat | None,
        payload: dict[str, object],
        confidence: float,
        source: str,
        evidence_refs: Iterable[str] = (),
    ) -> LiveEvent:
        seq = len(self._events) + 1
        before = self._revision
        return LiveEvent(
            event_id=f"EVT-{seq:06d}",
            event_type=event_type,
            session_id=self.session_id,
            seq=seq,
            monotonic_ms=monotonic_ns() // 1_000_000,
            wall_time=datetime.now().astimezone().isoformat(),
            trick_id=max(1, self._trick_id),
            turn_id=max(1, self._turn_id),
            actor=actor,
            payload=dict(payload),
            confidence=float(confidence),
            source=str(source),
            state_revision_before=before,
            state_revision_after=before + 1,
            evidence_refs=tuple(str(value) for value in evidence_refs),
        )

    def confirm_initial_state(
        self,
        *,
        round_level: str,
        hand: Iterable[str],
        lead_player: Seat | None = None,
        confidence: float = 1.0,
        source: str = "manual_start",
        evidence_refs: Iterable[str] = (),
    ) -> LiveEvent:
        if self._initialized or self._events:
            raise GameStateError("对局已经初始化")
        normalized = self._normalize_cards(hand)
        if len(normalized) != 27:
            raise GameStateError("开局时必须确认完整 27 张手牌")
        if round_level not in RANKS:
            raise GameStateError("请选择当前级牌")
        if lead_player is not None and lead_player not in TURN_ORDER:
            raise GameStateError("首出座位无效")
        event = self._new_event(
            "initial_state_confirmed",
            actor=lead_player,
            payload={
                "round_level": round_level,
                "wild_rank": round_level,
                "hand": list(normalized),
                "lead_player": lead_player,
            },
            confidence=confidence,
            source=source,
            evidence_refs=evidence_refs,
        )
        self.apply(event)
        return event

    def confirm_lead_player(self, lead_player: Seat) -> LiveEvent:
        """Confirm the lead player once the first-play marker appears.

        The doubling phase shows no first-play marker, so a session can start
        with an unresolved lead and be finalized automatically (or manually).
        """
        if not self._initialized:
            raise GameStateError("牌局尚未初始化")
        if self._lead_player is not None:
            raise GameStateError("首发座位已经确认")
        if lead_player not in TURN_ORDER:
            raise GameStateError("首出座位无效")
        event = self._new_event(
            "lead_player_confirmed",
            actor=lead_player,
            payload={"lead_player": lead_player},
            confidence=1.0,
            source="manual_or_auto_lead_confirmation",
        )
        self.apply(event)
        return event

    def record_play(
        self,
        player: Seat,
        cards: Iterable[str],
        *,
        confidence: float = 1.0,
        source: str = "multi_frame_consensus",
        evidence_refs: Iterable[str] = (),
        suit_options: Iterable[Iterable[str]] = (),
        integrity_warnings: Iterable[str] = (),
        action_metadata: dict[str, object] | None = None,
    ) -> LiveEvent:
        self._require_expected_player(player)
        raw_cards = tuple(str(card) for card in cards)
        aligned = sorted(
            zip(raw_cards, normalized_suit_options(raw_cards, suit_options)),
            key=lambda item: item[0],
        )
        normalized = self._normalize_cards(card for card, _options in aligned)
        payload: dict[str, object] = {
            "cards": list(normalized),
            "is_pass": False,
            "suit_options": [list(options) for _card, options in aligned],
        }
        warnings = tuple(dict.fromkeys(str(item) for item in integrity_warnings if str(item)))
        if warnings:
            payload["integrity_warnings"] = list(warnings)
        if action_metadata:
            payload.update(
                {
                    str(key): value
                    for key, value in action_metadata.items()
                    if str(key)
                }
            )
        event = self._new_event(
            "player_played",
            actor=player,
            payload=payload,
            confidence=confidence,
            source=source,
            evidence_refs=evidence_refs,
        )
        self.apply(event)
        return event

    def record_pass(
        self,
        player: Seat,
        *,
        confidence: float = 1.0,
        source: str = "pass_template",
        evidence_refs: Iterable[str] = (),
    ) -> LiveEvent:
        self._require_expected_player(player)
        event = self._new_event(
            "player_passed",
            actor=player,
            payload={"cards": [], "is_pass": True},
            confidence=confidence,
            source=source,
            evidence_refs=evidence_refs,
        )
        self.apply(event)
        return event

    def confirm_player_finished(
        self,
        player: Seat,
        *,
        placement: str,
        confidence: float = 1.0,
        source: str = "visual_placement",
    ) -> LiveEvent:
        """Confirm a persistent finish badge when card-count inference is wrong."""

        if player not in TURN_ORDER:
            raise GameStateError("出完牌玩家无效")
        if player in self._finished_seats:
            raise GameStateError("该玩家已经出完牌")
        normalized_placement = str(placement).strip().lower()
        if normalized_placement not in {"head", "second", "third", "last"}:
            raise GameStateError("玩家名次无效")
        event = self._new_event(
            "player_finished",
            actor=player,
            payload={"placement": normalized_placement},
            confidence=confidence,
            source=source,
        )
        self.apply(event)
        return event

    def correct_event(
        self,
        target_event_id: str,
        *,
        cards: Iterable[str] = (),
        is_pass: bool,
        reason: str,
        confidence: float = 1.0,
        source: str = "manual_correction",
    ) -> LiveEvent:
        actions = [event for event in self._events if event.event_type in _ACTION_EVENT_TYPES]
        if not actions or actions[-1].event_id != target_event_id:
            raise GameStateError("第一版只能纠正最近一次正式动作")
        target = actions[-1]
        normalized = () if is_pass else self._normalize_cards(cards)
        event = self._new_event(
            "event_correction",
            actor=target.actor,
            payload={
                "target_event_id": target_event_id,
                "cards": list(normalized),
                "is_pass": bool(is_pass),
                "reason": str(reason),
            },
            confidence=confidence,
            source=source,
        )
        self.apply(event)
        return event

    def correct_previous_action_after_followup(
        self,
        target_event_id: str,
        *,
        expected_followup_actor: Seat,
        followup_event_id: str,
        cards: Iterable[str],
        reason: str,
        confidence: float = 1.0,
        source: str = "two_frame_adjacent_action_reread",
        evidence_refs: Iterable[str] = (),
    ) -> LiveEvent:
        """Correct only the penultimate play after its legal follower acted.

        This is intentionally narrower than :meth:`correct_event`: a visual
        reread may add cards only while the originally observed action is still
        immediately before its follower.  Rebuilding validates all remaining
        card counts and must leave the current player unchanged; any failure
        rolls the correction back atomically.
        """

        actions = [event for event in self._events if event.event_type in _ACTION_EVENT_TYPES]
        if len(actions) < 2 or actions[-2].event_id != target_event_id:
            raise GameStateError("相邻复核只能纠正倒数第二个正式动作")
        target, followup = actions[-2:]
        if target.event_type == "player_passed" or bool(target.payload.get("is_pass", False)):
            raise GameStateError("相邻复核不能纠正不出动作")
        if (
            followup.event_id != str(followup_event_id)
            or followup.actor != expected_followup_actor
        ):
            raise GameStateError("相邻复核后的跟随动作不匹配")
        if any(
            event.event_type == "event_correction"
            and str(event.payload.get("target_event_id", "")) == target_event_id
            for event in self._events
        ):
            raise GameStateError("该动作已经被纠正")
        normalized = self._normalize_cards(cards)
        if not normalized:
            raise GameStateError("相邻复核必须保留出牌")
        current_before = self.snapshot().current_player
        event = self._new_event(
            "event_correction",
            actor=target.actor,
            payload={
                "target_event_id": target_event_id,
                "cards": list(normalized),
                "is_pass": False,
                "reason": str(reason),
                "correction_scope": "previous_action_after_followup",
                "followup_event_id": followup.event_id,
                "expected_followup_actor": expected_followup_actor,
            },
            confidence=confidence,
            source=source,
            evidence_refs=evidence_refs,
        )
        self._events.append(event)
        try:
            self._rebuild()
            if self.snapshot().current_player != current_before:
                raise GameStateError("相邻复核不能改变后续当前行动者")
        except Exception:
            self._events.pop()
            self._rebuild()
            raise
        return event

    def _require_expected_player(self, player: Seat) -> None:
        if not self._initialized:
            raise GameStateError("牌局尚未初始化")
        if player != self._current_player:
            raise GameStateError(
                f"当前应由 {self._current_player} 行动，不能记录 {player}"
            )

    def apply(self, event: LiveEvent) -> None:
        if event.session_id != self.session_id:
            raise GameStateError("事件不属于当前对局")
        if any(existing.event_id == event.event_id for existing in self._events):
            raise GameStateError("事件 ID 已存在")
        self._events.append(event)
        try:
            self._rebuild()
        except Exception:
            self._events.pop()
            self._rebuild()
            raise

    def _rebuild(self) -> None:
        events = tuple(self._events)
        corrections = {
            str(event.payload["target_event_id"]): event
            for event in events
            if event.event_type == "event_correction"
        }
        self._reset_semantic_state()
        for event in events:
            if event.event_type == "event_correction":
                self._revision += 1
                continue
            correction = corrections.get(event.event_id)
            effective = self._corrected_event(event, correction) if correction else event
            self._apply_semantic(effective)
            self._revision += 1

    @staticmethod
    def _corrected_event(original: LiveEvent, correction: LiveEvent) -> LiveEvent:
        is_pass = bool(correction.payload.get("is_pass", False))
        return replace(
            original,
            event_type="player_passed" if is_pass else "player_played",
            payload={
                "cards": list(correction.payload.get("cards", ())),
                "is_pass": is_pass,
                "suit_options": [],
            },
            confidence=correction.confidence,
            source=correction.source,
            evidence_refs=correction.evidence_refs or original.evidence_refs,
        )

    def _apply_semantic(self, event: LiveEvent) -> None:
        if event.event_type == "initial_state_confirmed":
            self._apply_initial(event)
            return
        if event.event_type == "lead_player_confirmed":
            self._apply_lead_confirmed(event)
            return
        if event.event_type in _ACTION_EVENT_TYPES:
            self._apply_action(event)
            return
        if event.event_type == "player_finished":
            self._apply_player_finished(event)
            return
        raise GameStateError(f"Reducer 不支持事件类型：{event.event_type}")

    def _apply_player_finished(self, event: LiveEvent) -> None:
        player = event.actor
        if player not in TURN_ORDER:
            raise GameStateError("出完牌玩家无效")
        placement = str(event.payload.get("placement", "")).strip().lower()
        if placement not in {"head", "second", "third", "last"}:
            raise GameStateError("玩家名次无效")
        if player in self._finished_seats:
            return
        self._remaining_cards[player] = 0
        self._finished_seats.add(player)
        if self._round_is_decided():
            self._close_round()
            return
        if self._current_player == player:
            self._current_player = next_active_seat(
                player,
                frozenset(self._finished_seats),
            )

    def _apply_initial(self, event: LiveEvent) -> None:
        hand = self._normalize_cards(event.payload.get("hand", ()))
        if len(hand) != 27:
            raise GameStateError("开局时必须确认完整 27 张手牌")
        round_level = str(event.payload.get("round_level", ""))
        wild_rank = str(event.payload.get("wild_rank", round_level))
        if round_level not in RANKS or wild_rank not in RANKS:
            raise GameStateError("开局级牌无效")
        lead = event.payload.get("lead_player")
        if lead is not None and lead not in TURN_ORDER:
            raise GameStateError("开局首出座位无效")
        self._round_level = round_level
        self._wild_rank = wild_rank
        self._my_hand = hand
        self._lead_player = lead
        self._current_player = lead
        self._trick_id = 1
        self._turn_id = 1
        self._initialized = True

    def _apply_lead_confirmed(self, event: LiveEvent) -> None:
        if not self._initialized:
            raise GameStateError("牌局尚未初始化")
        if self._lead_player is not None:
            raise GameStateError("首发座位已经确认")
        lead = event.payload.get("lead_player")
        if lead not in TURN_ORDER:
            raise GameStateError("首发座位无效")
        self._lead_player = lead
        self._current_player = lead
        self._trick_id = 1
        self._turn_id = 1

    def _apply_action(self, event: LiveEvent) -> None:
        if not self._initialized:
            raise GameStateError("牌局尚未初始化")
        player = event.actor
        if player not in TURN_ORDER:
            raise GameStateError("动作玩家无效")
        if player != self._current_player:
            raise GameStateError(
                f"当前应由 {self._current_player} 行动，事件却来自 {player}"
            )
        is_pass = event.event_type == "player_passed" or bool(
            event.payload.get("is_pass", False)
        )
        cards = () if is_pass else self._normalize_cards(
            event.payload.get("cards", ())
        )
        if is_pass and not self._trick_plays:
            raise GameStateError("首出玩家不能不出")
        if len(cards) > self._remaining_cards[player]:
            raise GameStateError("出牌数量超过该玩家剩余牌数")
        if player == "self" and not is_pass:
            available = Counter(self._my_hand)
            requested = Counter(cards)
            if requested - available:
                raise GameStateError("实际出牌不在已确认手牌中")
            available.subtract(requested)
            self._my_hand = tuple(sorted((+available).elements()))

        observed_at = datetime.fromisoformat(event.wall_time)
        play = PlayEvent(
            player=player,
            cards=cards,
            is_pass=is_pass,
            observed_at=observed_at,
            source=event.source,
            suit_options=tuple(
                tuple(str(suit) for suit in options)
                for options in event.payload.get("suit_options", ())
                if isinstance(options, (list, tuple))
            ),
            action_metadata=_action_metadata_from_payload(event.payload),
        )
        self._trick_plays.append(play)
        self._play_history.append(play)
        if not is_pass:
            self._remaining_cards[player] -= len(cards)
            if self._remaining_cards[player] == 0:
                self._finished_seats.add(player)

        # 头游和二游为队友时已经形成双下；否则产生第三名后也能确定末游。
        # 两种情况都不能再伪造后续行动者、PASS 或模型请求。
        if self._round_is_decided():
            self._close_round()
            self._turn_id += 1
            return

        self._current_player = self._next_player_after_action(
            player,
        )
        self._turn_id += 1
        self._finish_trick_if_all_others_passed()

    def _next_player_after_action(self, player: Seat) -> Seat:
        """Advance by the common trick projection without synthetic PASSes."""

        projection = self._current_trick_projection()
        if projection is None:
            return next_active_seat(player, frozenset(self._finished_seats))
        expected = projection.expected_after(player)
        if expected is None:
            raise GameStateError("当前墩没有仍在对局中的下一位玩家")
        return expected

    def _current_trick_projection(self):
        last_non_pass_index = next(
            (
                index
                for index in range(len(self._trick_plays) - 1, -1, -1)
                if not self._trick_plays[index].is_pass
            ),
            None,
        )
        if last_non_pass_index is None:
            return None
        leader = self._trick_plays[last_non_pass_index].player
        passed = {
            event.player
            for event in self._trick_plays[last_non_pass_index + 1 :]
            if event.is_pass
        }
        return project_trick_turn(
            leader,
            frozenset(self._finished_seats),
            passed,
            wind_receiver_must_pass=self._wind_receiver_must_pass,
        )

    def _pending_wind_receiver(self) -> Seat | None:
        """Return the partner that will lead after a finished leader's trick."""

        projection = self._current_trick_projection()
        return projection.wind_receiver if projection is not None else None

    def pending_wind_receiver(self) -> Seat | None:
        """Return the partner that is waiting to catch wind, if any.

        This is intentionally read-only so the visual ownership guard can
        recognise the real UI handoff without manufacturing a player action.
        """

        return self._pending_wind_receiver()

    def wind_receiver_after_current_pass(self, player: Seat) -> Seat | None:
        """Return the wind receiver only when ``player`` can close this trick.

        The result is deliberately narrower than :meth:`pending_wind_receiver`:
        it proves that the current expected player's PASS is the one remaining
        legal action before the new trick starts.  It is safe for a visual
        recovery path to use only together with a seat-bound PASS marker.
        """

        if player != self._current_player:
            return None
        projection = self._current_trick_projection()
        if projection is None or projection.wind_receiver is None:
            return None
        if player not in projection.required_passers:
            return None
        return projection.next_leader if projection.closes_if(player) else None

    def _finish_trick_if_all_others_passed(self) -> None:
        projection = self._current_trick_projection()
        if projection is None or not projection.is_complete:
            return
        next_leader = projection.next_leader
        if next_leader is None:
            return
        self._trick_plays.clear()
        self._lead_player = next_leader
        self._current_player = next_leader
        self._trick_id += 1

    def _round_is_decided(self) -> bool:
        return round_is_decided(self._finished_seats)

    def _close_round(self) -> None:
        self._trick_plays.clear()
        self._current_player = None

    def snapshot(self) -> LiveSnapshot:
        return LiveSnapshot(
            session_id=self.session_id,
            round_level=self._round_level,
            wild_rank=self._wild_rank,
            current_player=self._current_player,
            lead_player=self._lead_player,
            my_hand=self._my_hand,
            trick_plays=tuple(self._trick_plays),
            play_history=tuple(self._play_history),
            remaining_cards=dict(self._remaining_cards),
            finished_seats=frozenset(self._finished_seats),
            trick_id=self._trick_id,
            turn_id=self._turn_id,
            revision=self._revision,
            initialized=self._initialized,
        )

    def to_guandan_state(self) -> GuanDanState:
        snapshot = self.snapshot()
        if not snapshot.initialized:
            raise GameStateError("牌局尚未初始化")
        state = GuanDanState(
            round_level=snapshot.round_level,
            wild_rank=snapshot.wild_rank,
            current_player=snapshot.current_player,
            lead_player=snapshot.lead_player,
            my_hand=snapshot.my_hand,
            trick_plays=list(snapshot.trick_plays),
            play_history=list(snapshot.play_history),
            remaining_cards=dict(snapshot.remaining_cards),
            revision=snapshot.revision,
        )
        return state

    def clone_empty(self) -> "LiveReducer":
        return LiveReducer(
            self.session_id,
            wind_receiver_must_pass=self._wind_receiver_must_pass,
        )


def _action_metadata_from_payload(
    payload: dict[str, object],
) -> dict[str, object] | None:
    nested = payload.get("move_semantics")
    if isinstance(nested, dict):
        return dict(nested)
    ignored = {"cards", "is_pass", "suit_options", "integrity_warnings"}
    metadata = {str(key): value for key, value in payload.items() if key not in ignored}
    return metadata or None
