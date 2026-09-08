from __future__ import annotations

from dataclasses import replace

import pytest

from daguandan_bridge.live.reducer import LiveReducer
from daguandan_bridge.live_v2.protocols import (
    RuleProjector,
    RuleStateProvider,
    TransactionalActionCommitter,
)
from daguandan_bridge.infrastructure.live_v2_rule_backend import create_reducer_backend
from daguandan_bridge.live_v2.rules_adapter import LiveReducerRuleAdapter
from daguandan_bridge.live_v2.types import (
    ActionCandidate,
    ActionKind,
    CandidateReason,
    CommitReason,
    FrameIdentity,
    ProjectionReason,
    Seat,
    VersionIdentity,
)


def _hand() -> tuple[str, ...]:
    ranks = ("2", "3", "4", "5", "6", "7", "8", "9", "10", "J", "Q", "K", "A")
    return tuple(f"{rank}{suit}" for suit in ("S", "H", "C") for rank in ranks)[:27]


def _adapter(
    *, lead: Seat = Seat.RIGHT, level: str = "2"
) -> tuple[LiveReducerRuleAdapter, LiveReducer]:
    reducer = LiveReducer("s")
    reducer.confirm_initial_state(
        round_level=level,
        hand=_hand(),
        lead_player=lead.value,
        monotonic_ms=1,
    )
    version = VersionIdentity("s", 1, reducer.snapshot().revision, 0, 0)
    backend = create_reducer_backend(reducer, version=version, in_memory=True)
    return LiveReducerRuleAdapter(backend, version), reducer


def _candidate(
    identity: int,
    seat: Seat,
    cards: tuple[str, ...] = (),
    *,
    kind: ActionKind = ActionKind.PLAY,
    first_seq: int | None = None,
    processing_ms: int | None = None,
    version: VersionIdentity | None = None,
    evidence_prefix: str = "e",
    generation: int = 1,
) -> ActionCandidate:
    start = identity * 10 if first_seq is None else first_seq
    first = FrameIdentity("s", generation, start, start * 10, "roi", "window")
    last = FrameIdentity("s", generation, start + 1, start * 10 + 10, "roi", "window")
    return ActionCandidate(
        candidate_id=f"c-{identity}",
        version=version or VersionIdentity("s", 1, 1, 0, 0),
        seat=seat,
        kind=kind,
        cards=cards,
        suit_options=(
            tuple((card,) for card in cards)
            if kind is ActionKind.PLAY
            else ()
        ),
        evidence_ids=(
            f"{evidence_prefix}-{identity}-1",
            f"{evidence_prefix}-{identity}-2",
        ),
        action_epoch=identity,
        first_frame=first,
        last_frame=last,
        processing_ms=processing_ms or last.captured_ms + 1,
        confidence=0.95,
        reason=(
            CandidateReason.STABLE_PLAY
            if kind is ActionKind.PLAY
            else CandidateReason.FRESH_PASS_EDGE
        ),
    )


def test_adapter_satisfies_ports_and_commits_one_legal_action() -> None:
    adapter, reducer = _adapter()
    assert isinstance(adapter, RuleProjector)
    assert isinstance(adapter, RuleStateProvider)
    assert isinstance(adapter, TransactionalActionCommitter)
    candidate = _candidate(1, Seat.RIGHT, ("3D",))
    projected = adapter.project(
        base_version=adapter.version,
        candidates=(candidate,),
        processing_ms=500,
    )
    assert projected.reason is ProjectionReason.ACCEPTED
    action = projected.actions[0]
    assert action.source_candidate is candidate
    assert action.action_epoch == candidate.action_epoch
    assert action.suit_options == candidate.suit_options
    assert action.first_frame == candidate.first_frame
    assert action.last_frame == candidate.last_frame
    committed = adapter.commit(
        expected_version=adapter.version,
        actions=projected.actions,
    )
    assert committed.reason is CommitReason.COMMITTED
    snapshot = reducer.snapshot()
    assert snapshot.current_player == Seat.OPPOSITE.value
    assert snapshot.play_history[-1].cards == ("3D",)


@pytest.mark.parametrize("reverse_input", [False, True])
def test_orders_by_capture_interval_not_input_or_processing_time(reverse_input: bool) -> None:
    adapter, _reducer = _adapter()
    right = _candidate(1, Seat.RIGHT, ("3D",), processing_ms=9_000)
    opposite = _candidate(2, Seat.OPPOSITE, ("4D",), processing_ms=500)
    candidates = (opposite, right) if reverse_input else (right, opposite)
    projected = adapter.project(
        base_version=adapter.version,
        candidates=candidates,
        processing_ms=10_000,
    )
    assert projected.reason is ProjectionReason.ACCEPTED
    assert [action.seat for action in projected.actions] == [Seat.RIGHT, Seat.OPPOSITE]


def test_pass_without_a_lead_is_rejected_but_illegal_future_does_not_block_current() -> None:
    adapter, reducer = _adapter()
    before = reducer.snapshot()
    leading_pass = _candidate(1, Seat.RIGHT, kind=ActionKind.PASS)
    rejected = adapter.project(
        base_version=adapter.version,
        candidates=(leading_pass,),
        processing_ms=500,
    )
    assert rejected.reason is ProjectionReason.RULE_REJECTED
    assert reducer.snapshot() == before

    legal = _candidate(2, Seat.RIGHT, ("8D",))
    weaker = _candidate(3, Seat.OPPOSITE, ("7D",))
    projected = adapter.project(
        base_version=adapter.version,
        candidates=(legal, weaker),
        processing_ms=1_000,
    )
    assert projected.reason is ProjectionReason.ACCEPTED
    assert tuple(action.source_candidate for action in projected.actions) == (legal,)
    committed = adapter.commit(
        expected_version=adapter.version,
        actions=projected.actions,
    )
    assert committed.reason is CommitReason.COMMITTED
    assert reducer.snapshot().play_history[-1].cards == ("8D",)


def test_maximal_valid_chain_wins_over_its_valid_prefixes() -> None:
    adapter, _reducer = _adapter()
    right = _candidate(1, Seat.RIGHT, ("3D",))
    opposite = _candidate(2, Seat.OPPOSITE, ("4D",))

    projected = adapter.project(
        base_version=adapter.version,
        candidates=(right, opposite),
        processing_ms=500,
    )

    assert projected.reason is ProjectionReason.ACCEPTED
    assert tuple(action.source_candidate for action in projected.actions) == (
        right,
        opposite,
    )


def test_distinct_maximal_single_action_solutions_are_not_guessed() -> None:
    adapter, reducer = _adapter()
    first = _candidate(1, Seat.RIGHT, ("3D",))
    alternative = _candidate(2, Seat.RIGHT, ("4D",))

    projected = adapter.project(
        base_version=adapter.version,
        candidates=(first, alternative),
        processing_ms=500,
    )

    assert projected.reason is ProjectionReason.OUT_OF_ORDER
    assert reducer.snapshot().play_history == ()


def test_stale_future_surface_does_not_block_current_self_bomb() -> None:
    hand = ("JS", "JH", "JC", "JD") + tuple(
        card for card in _hand() if card not in {"JS", "JH"}
    )[:23]
    reducer = LiveReducer("s")
    reducer.confirm_initial_state(
        round_level="2",
        hand=hand,
        lead_player=Seat.SELF.value,
        monotonic_ms=1,
    )
    version = VersionIdentity("s", 1, reducer.snapshot().revision, 0, 0)
    adapter = LiveReducerRuleAdapter(
        create_reducer_backend(reducer, version=version, in_memory=True),
        version,
    )
    stale_right = replace(
        _candidate(1, Seat.RIGHT, ("A?",), version=version),
        suit_options=(("A?", "AS", "AH"),),
    )
    self_bomb = _candidate(
        2,
        Seat.SELF,
        ("JS", "JH", "JC", "JD"),
        version=version,
    )

    projected = adapter.project(
        base_version=version,
        candidates=(stale_right, self_bomb),
        processing_ms=500,
    )

    assert projected.reason is ProjectionReason.ACCEPTED
    assert tuple(action.source_candidate for action in projected.actions) == (
        self_bomb,
    )
    committed = adapter.commit(expected_version=version, actions=projected.actions)
    assert committed.reason is CommitReason.COMMITTED
    assert reducer.snapshot().play_history[-1].cards == ("JC", "JD", "JH", "JS")


def test_duplicate_evidence_rejects_whole_batch_before_subset_selection() -> None:
    adapter, reducer = _adapter()
    current = _candidate(1, Seat.RIGHT, ("3D",))
    future = replace(
        _candidate(2, Seat.OPPOSITE, ("4D",)),
        evidence_ids=(current.evidence_ids[-1], "shared-tail"),
    )

    projected = adapter.project(
        base_version=adapter.version,
        candidates=(current, future),
        processing_ms=500,
    )

    assert projected.reason is ProjectionReason.RULE_REJECTED
    assert reducer.snapshot().play_history == ()


def test_overlapping_alternatives_reject_whole_batch_before_subset_selection() -> None:
    adapter, reducer = _adapter()
    current = _candidate(1, Seat.RIGHT, ("3D",))
    alternative = replace(
        _candidate(2, Seat.RIGHT, ("4D",)),
        action_epoch=current.action_epoch,
        first_frame=current.first_frame,
        last_frame=current.last_frame,
    )

    projected = adapter.project(
        base_version=adapter.version,
        candidates=(current, alternative),
        processing_ms=500,
    )

    assert projected.reason is ProjectionReason.OUT_OF_ORDER
    assert reducer.snapshot().play_history == ()


def test_candidate_and_evidence_are_consumed_at_most_once() -> None:
    adapter, _reducer = _adapter()
    candidate = _candidate(1, Seat.RIGHT, ("3D",))
    projected = adapter.project(
        base_version=adapter.version,
        candidates=(candidate,),
        processing_ms=500,
    )
    adapter.commit(expected_version=adapter.version, actions=projected.actions)
    replayed = replace(candidate, version=adapter.version)
    rejected = adapter.project(
        base_version=adapter.version,
        candidates=(replayed,),
        processing_ms=600,
    )
    assert rejected.reason is ProjectionReason.RULE_REJECTED


def test_same_cards_can_reappear_after_a_complete_action_cycle() -> None:
    adapter, _reducer = _adapter()
    chain = (
        _candidate(1, Seat.RIGHT, ("3D",)),
        _candidate(2, Seat.OPPOSITE, kind=ActionKind.PASS),
        _candidate(3, Seat.LEFT, kind=ActionKind.PASS),
        _candidate(4, Seat.SELF, kind=ActionKind.PASS),
    )
    projected = adapter.project(
        base_version=adapter.version,
        candidates=chain,
        processing_ms=1_000,
    )
    assert projected.reason is ProjectionReason.ACCEPTED
    adapter.commit(expected_version=adapter.version, actions=projected.actions)
    repeated = _candidate(
        5,
        Seat.RIGHT,
        ("3D",),
        version=adapter.version,
        evidence_prefix="new-cycle",
    )
    assert adapter.project(
        base_version=adapter.version,
        candidates=(repeated,),
        processing_ms=2_000,
    ).reason is ProjectionReason.ACCEPTED


def test_duplicate_physical_cards_are_not_collapsed() -> None:
    adapter, reducer = _adapter()
    pair = _candidate(1, Seat.RIGHT, ("5D", "5D"))
    projected = adapter.project(
        base_version=adapter.version,
        candidates=(pair,),
        processing_ms=500,
    )
    assert projected.reason is ProjectionReason.ACCEPTED
    adapter.commit(expected_version=adapter.version, actions=projected.actions)
    assert reducer.snapshot().play_history[-1].cards == ("5D", "5D")


def test_confirmed_card_entities_keep_candidate_suit_option_alignment() -> None:
    adapter, reducer = _adapter()
    pair = replace(
        _candidate(1, Seat.RIGHT, ("5H", "5D")),
        suit_options=(("5H",), ("5D",)),
    )
    projected = adapter.project(
        base_version=adapter.version,
        candidates=(pair,),
        processing_ms=500,
    )
    action = projected.actions[0]
    assert action.cards == ("5H", "5D")
    assert action.suit_options == (("5H",), ("5D",))
    adapter.commit(expected_version=adapter.version, actions=projected.actions)
    assert reducer.snapshot().play_history[-1].cards == ("5D", "5H")


def test_projection_cannot_be_committed_after_version_changes() -> None:
    adapter, reducer = _adapter()
    candidate = _candidate(1, Seat.RIGHT, ("3D",))
    projected = adapter.project(
        base_version=adapter.version,
        candidates=(candidate,),
        processing_ms=500,
    )
    wrong = replace(adapter.version, update_sequence=9)
    result = adapter.commit(expected_version=wrong, actions=projected.actions)
    assert result.reason is CommitReason.VERSION_CONFLICT
    assert not reducer.snapshot().play_history


def test_direct_reducer_mutation_cannot_be_overwritten_by_staged_commit() -> None:
    adapter, reducer = _adapter()
    candidate = _candidate(1, Seat.RIGHT, ("3D",))
    projected = adapter.project(
        base_version=adapter.version,
        candidates=(candidate,),
        processing_ms=500,
    )
    reducer.record_play("right", ("4D",), monotonic_ms=400)
    result = adapter.commit(
        expected_version=adapter.version,
        actions=projected.actions,
    )
    assert result.reason is CommitReason.VERSION_CONFLICT
    assert reducer.snapshot().play_history[-1].cards == ("4D",)


def test_full_candidate_version_must_match_not_only_revision() -> None:
    adapter, _reducer = _adapter()
    stale = _candidate(
        1,
        Seat.RIGHT,
        ("3D",),
        version=replace(adapter.version, update_sequence=99),
    )
    result = adapter.project(
        base_version=adapter.version,
        candidates=(stale,),
        processing_ms=500,
    )
    assert result.reason is ProjectionReason.VERSION_MISMATCH


def test_unknown_suit_options_survive_candidate_boundary() -> None:
    adapter, reducer = _adapter()
    candidate = replace(
        _candidate(1, Seat.RIGHT, ("7?",)),
        suit_options=(("7?", "7S", "7C"),),
    )
    projected = adapter.project(
        base_version=adapter.version,
        candidates=(candidate,),
        processing_ms=500,
    )
    assert projected.reason is ProjectionReason.ACCEPTED
    adapter.commit(expected_version=adapter.version, actions=projected.actions)
    event = reducer.snapshot().play_history[-1]
    assert event.cards == ("7?",)
    assert event.suit_options == (("S", "C"),)


def test_exact_visual_label_with_multiple_suit_options_stays_uncertain() -> None:
    adapter, reducer = _adapter()
    candidate = replace(
        _candidate(1, Seat.RIGHT, ("7D",)),
        suit_options=(("7D", "7S"),),
    )
    projected = adapter.project(
        base_version=adapter.version,
        candidates=(candidate,),
        processing_ms=500,
    )
    assert projected.reason is ProjectionReason.ACCEPTED
    adapter.commit(expected_version=adapter.version, actions=projected.actions)
    event = reducer.snapshot().play_history[-1]
    assert event.cards == ("7?",)
    assert event.suit_options == (("D", "S"),)


def test_level_heart_wildcard_is_validated_by_existing_rule_engine() -> None:
    adapter, _reducer = _adapter(level="5")
    wildcard_pair = _candidate(1, Seat.RIGHT, ("5H", "7S"))
    result = adapter.project(
        base_version=adapter.version,
        candidates=(wildcard_pair,),
        processing_ms=500,
    )
    assert result.reason is ProjectionReason.ACCEPTED


def test_conflicting_future_capture_is_excluded_without_blocking_current() -> None:
    adapter, reducer = _adapter()
    right = _candidate(1, Seat.RIGHT, ("3D",))
    opposite = _candidate(2, Seat.OPPOSITE, ("4D",))
    opposite = replace(
        opposite,
        first_frame=FrameIdentity("s", 1, 20, 50, "roi", "window"),
        last_frame=FrameIdentity("s", 1, 21, 60, "roi", "window"),
    )
    result = adapter.project(
        base_version=adapter.version,
        candidates=(right, opposite),
        processing_ms=500,
    )
    assert result.reason is ProjectionReason.ACCEPTED
    assert tuple(action.source_candidate for action in result.actions) == (right,)
    committed = adapter.commit(expected_version=adapter.version, actions=result.actions)
    assert committed.reason is CommitReason.COMMITTED
    assert reducer.snapshot().play_history[-1].cards == ("3D",)


def test_ambiguous_local_suit_is_not_guessed_from_known_hand() -> None:
    adapter, reducer = _adapter(lead=Seat.SELF)
    candidate = replace(
        _candidate(1, Seat.SELF, ("7?",)),
        suit_options=(("7?", "7S", "7H"),),
    )
    result = adapter.project(
        base_version=adapter.version,
        candidates=(candidate,),
        processing_ms=500,
    )
    assert result.reason is ProjectionReason.NEED_MORE_EVIDENCE
    assert len(reducer.snapshot().my_hand) == 27


def test_initial_snapshot_contains_complete_trusted_rule_state() -> None:
    adapter, _reducer = _adapter()
    state = adapter.snapshot_for(version=adapter.version, captured_ms=0)
    assert state.round_level == state.wild_rank == "2"
    assert state.current_seat is Seat.RIGHT
    assert state.lead_seat is Seat.RIGHT
    assert len(state.my_hand) == 27
    assert state.play_history == state.current_trick == ()
    assert tuple(item.count for item in state.remaining) == (27, 27, 27, 27)
    assert state.finished == ()
    assert state.trusted and not state.terminal


def test_snapshot_updates_immediately_after_single_committed_action() -> None:
    adapter, _reducer = _adapter()
    candidate = _candidate(1, Seat.RIGHT, ("3D",))
    projection = adapter.project(
        base_version=adapter.version,
        candidates=(candidate,),
        processing_ms=500,
    )
    commit = adapter.commit(
        expected_version=adapter.version, actions=projection.actions
    )
    state = adapter.snapshot_for(
        version=commit.resulting_version,
        captured_ms=candidate.last_captured_ms,
    )
    assert state.play_history[0].action_id == projection.actions[0].action_id
    assert state.current_trick == state.play_history
    assert state.remaining_for(Seat.RIGHT) == 26
    assert state.current_seat is Seat.OPPOSITE


def test_pass_chain_closes_trick_and_snapshot_uses_exact_empty_suffix() -> None:
    adapter, _reducer = _adapter()
    chain = (
        _candidate(1, Seat.RIGHT, ("3D",)),
        _candidate(2, Seat.OPPOSITE, kind=ActionKind.PASS),
        _candidate(3, Seat.LEFT, kind=ActionKind.PASS),
        _candidate(4, Seat.SELF, kind=ActionKind.PASS),
    )
    projection = adapter.project(
        base_version=adapter.version,
        candidates=chain,
        processing_ms=1_000,
    )
    commit = adapter.commit(
        expected_version=adapter.version, actions=projection.actions
    )
    state = adapter.snapshot_for(
        version=commit.resulting_version,
        captured_ms=chain[-1].last_captured_ms,
    )
    assert len(state.play_history) == 4
    assert state.current_trick == ()
    assert state.current_seat is Seat.RIGHT


def test_snapshot_advances_trick_and_uses_the_second_trick_leader() -> None:
    adapter, _reducer = _adapter()
    specs = (
        (Seat.RIGHT, ActionKind.PLAY, ("3D",)),
        (Seat.OPPOSITE, ActionKind.PLAY, ("4D",)),
        (Seat.LEFT, ActionKind.PASS, ()),
        (Seat.SELF, ActionKind.PASS, ()),
        (Seat.RIGHT, ActionKind.PASS, ()),
        (Seat.OPPOSITE, ActionKind.PLAY, ("5D",)),
    )
    last = None
    for index, (seat, kind, cards) in enumerate(specs, start=1):
        last = _candidate(
            index, seat, cards, kind=kind, version=adapter.version,
            processing_ms=1_000 + index,
        )
        projection = adapter.project(
            base_version=adapter.version,
            candidates=(last,),
            processing_ms=1_000 + index,
        )
        assert projection.reason is ProjectionReason.ACCEPTED
        commit = adapter.commit(
            expected_version=adapter.version, actions=projection.actions
        )
        assert commit.reason is CommitReason.COMMITTED
    assert last is not None
    state = adapter.snapshot_for(
        version=adapter.version,
        captured_ms=last.last_captured_ms,
    )
    assert state.trick_index == 2
    assert state.opening_seat is Seat.RIGHT
    assert state.lead_seat is Seat.OPPOSITE
    assert state.current_seat is Seat.LEFT
    assert state.current_trick == (state.play_history[-1],)


def test_snapshot_preserves_duplicate_card_entities() -> None:
    adapter, _reducer = _adapter()
    pair = _candidate(1, Seat.RIGHT, ("5D", "5D"))
    projection = adapter.project(
        base_version=adapter.version,
        candidates=(pair,),
        processing_ms=500,
    )
    commit = adapter.commit(
        expected_version=adapter.version, actions=projection.actions
    )
    state = adapter.snapshot_for(
        version=commit.resulting_version, captured_ms=pair.last_captured_ms
    )
    assert state.play_history[0].cards == ("5D", "5D")
    assert state.play_history[0].suit_options == (("5D",), ("5D",))
    assert state.remaining_for(Seat.RIGHT) == 25


def test_validated_seed_reconstructs_existing_opening_action() -> None:
    adapter, reducer = _adapter()
    candidate = _candidate(1, Seat.RIGHT, ("3D",))
    projection = adapter.project(
        base_version=adapter.version,
        candidates=(candidate,),
        processing_ms=500,
    )
    commit = adapter.commit(
        expected_version=adapter.version, actions=projection.actions
    )
    seeded_backend = create_reducer_backend(
        reducer,
        version=commit.resulting_version,
        seed_actions=commit.committed_actions,
        in_memory=True,
    )
    seeded = LiveReducerRuleAdapter(seeded_backend, commit.resulting_version)
    state = seeded.snapshot_for(
        version=seeded.version, captured_ms=candidate.last_captured_ms
    )
    assert state.play_history[0].cards == ("3D",)
    assert state.current_seat is Seat.OPPOSITE


def test_seed_must_replay_and_match_existing_reducer() -> None:
    adapter, _reducer = _adapter()
    candidate = _candidate(1, Seat.RIGHT, ("3D",))
    projection = adapter.project(
        base_version=adapter.version,
        candidates=(candidate,),
        processing_ms=500,
    )
    commit = adapter.commit(
        expected_version=adapter.version, actions=projection.actions
    )
    _other_adapter, other = _adapter()
    other.record_play("right", ("4D",), monotonic_ms=200)
    with pytest.raises(ValueError, match="history and committed"):
        create_reducer_backend(
            other,
            version=commit.resulting_version,
            seed_actions=commit.committed_actions,
            in_memory=True,
        )


def test_snapshot_accepts_newer_view_sequence_but_rejects_wrong_rule_state() -> None:
    adapter, _reducer = _adapter()
    candidate = _candidate(1, Seat.RIGHT, ("3D",))
    projection = adapter.project(
        base_version=adapter.version,
        candidates=(candidate,),
        processing_ms=500,
    )
    commit = adapter.commit(
        expected_version=adapter.version, actions=projection.actions
    )
    view_version = replace(commit.resulting_version, update_sequence=99)
    view = adapter.snapshot_for(
        version=view_version,
        captured_ms=candidate.last_captured_ms,
    )
    assert view.version == view_version
    assert view.play_history[0].version_after == commit.resulting_version.state_version
    with pytest.raises(ValueError, match="snapshot version"):
        adapter.snapshot_for(
            version=replace(commit.resulting_version, update_sequence=0),
            captured_ms=candidate.last_captured_ms,
        )
    with pytest.raises(ValueError, match="snapshot version"):
        adapter.snapshot_for(
            version=replace(commit.resulting_version, turn_index=99),
            captured_ms=candidate.last_captured_ms,
        )
    with pytest.raises(ValueError, match="precedes committed"):
        adapter.snapshot_for(
            version=commit.resulting_version,
            captured_ms=candidate.first_captured_ms - 1,
        )
    with pytest.raises(ValueError, match="must not be negative"):
        adapter.snapshot_for(version=commit.resulting_version, captured_ms=-1)


def test_failed_commit_does_not_append_to_snapshot_ledger() -> None:
    adapter, _reducer = _adapter()
    candidate = _candidate(1, Seat.RIGHT, ("3D",))
    projection = adapter.project(
        base_version=adapter.version,
        candidates=(candidate,),
        processing_ms=500,
    )
    initial = adapter.version
    failed = adapter.commit(
        expected_version=replace(initial, update_sequence=99),
        actions=projection.actions,
    )
    assert failed.reason is CommitReason.VERSION_CONFLICT
    assert adapter.snapshot_for(version=initial, captured_ms=500).play_history == ()


def test_reconnect_reuses_prior_generation_seed_and_continues_current_stream() -> None:
    first_adapter, reducer = _adapter()
    first_candidate = _candidate(1, Seat.RIGHT, ("3D",))
    first_projection = first_adapter.project(
        base_version=first_adapter.version,
        candidates=(first_candidate,),
        processing_ms=500,
    )
    first_commit = first_adapter.commit(
        expected_version=first_adapter.version,
        actions=first_projection.actions,
    )
    reconnected_version = VersionIdentity.from_state(
        first_commit.resulting_version.state_version,
        capture_generation=2,
        update_sequence=0,
    )
    second_backend = create_reducer_backend(
        reducer,
        version=reconnected_version,
        seed_actions=first_commit.committed_actions,
        in_memory=True,
    )
    second_adapter = LiveReducerRuleAdapter(second_backend, reconnected_version)
    before = second_adapter.snapshot_for(
        version=reconnected_version,
        captured_ms=first_candidate.last_captured_ms,
    )
    assert before.play_history[0].first_frame.capture_generation == 1

    second_candidate = _candidate(
        2,
        Seat.OPPOSITE,
        ("4D",),
        version=reconnected_version,
        generation=2,
    )
    second_projection = second_adapter.project(
        base_version=second_adapter.version,
        candidates=(second_candidate,),
        processing_ms=600,
    )
    second_commit = second_adapter.commit(
        expected_version=second_adapter.version,
        actions=second_projection.actions,
    )
    after = second_adapter.snapshot_for(
        version=second_commit.resulting_version,
        captured_ms=second_candidate.last_captured_ms,
    )
    assert after.version.capture_generation == 2
    assert [action.first_frame.capture_generation for action in after.play_history] == [1, 2]
    assert after.play_history[0].first_frame == first_candidate.first_frame
    assert after.play_history[0].version_after == after.play_history[1].version_before

    third_version = VersionIdentity.from_state(
        second_commit.resulting_version.state_version,
        capture_generation=3,
        update_sequence=0,
    )
    combined_seed = first_commit.committed_actions + second_commit.committed_actions
    third_backend = create_reducer_backend(
        reducer,
        version=third_version,
        seed_actions=combined_seed,
        in_memory=True,
    )
    third_adapter = LiveReducerRuleAdapter(third_backend, third_version)
    third = third_adapter.snapshot_for(
        version=third_version,
        captured_ms=second_candidate.last_captured_ms,
    )
    assert [action.first_frame.capture_generation for action in third.play_history] == [1, 2]


def test_current_candidate_batch_cannot_mix_capture_generations() -> None:
    adapter, _reducer = _adapter()
    current = _candidate(1, Seat.RIGHT, ("3D",))
    newer_version = replace(adapter.version, capture_generation=2)
    newer = _candidate(
        2,
        Seat.OPPOSITE,
        ("4D",),
        version=newer_version,
        generation=2,
    )
    result = adapter.project(
        base_version=adapter.version,
        candidates=(current, newer),
        processing_ms=500,
    )
    assert result.reason is ProjectionReason.VERSION_MISMATCH
