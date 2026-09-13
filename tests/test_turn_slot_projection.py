from __future__ import annotations

from daguandan_bridge.application.turn_slot_projection import project_turn_slots


def _row(frame, current, regions):
    return {
        "frame_index": frame,
        "timestamp_ms": frame * 100,
        "decode_ok": True,
        "opening": {"current_player_signal": current},
        "regions": regions,
    }


def test_slots_keep_missing_actor_visible_for_review_instead_of_skipping_seat():
    observations = [
        _row(0, "self", {}),
        _row(1, "self", {}),
        _row(2, "right", {"self": {"cards": [], "is_pass": True, "confidence": .98}}),
        _row(3, "right", {}),
        # Right's action surface is fully hidden. The next timer advances.
        _row(4, "opposite", {}),
        _row(5, "opposite", {}),
        _row(6, "left", {"opposite": {"cards": [], "is_pass": True, "confidence": .98}}),
    ]
    actions = [
        {"action_id": 1, "actor": "self", "is_pass": True, "cards": [], "frame_start": 2, "frame_end": 2}
    ]

    ledger = project_turn_slots(observations, actions)
    slots = [slot for slot in ledger["slots"] if slot["kind"] == "turn"]

    assert [slot["actor"] for slot in slots] == ["self", "right", "opposite"]
    assert slots[0]["status"] == "resolved"
    assert slots[1]["status"] == "needs_review"
    assert slots[2]["status"] == "recovered"
    assert slots[2]["action"]["is_pass"] is True


def test_initial_existing_table_surface_is_context_not_fabricated_turn():
    observations = [
        _row(0, "self", {"opposite": {"cards": ["5C"], "is_pass": False, "confidence": .9}}),
        _row(1, "right", {"self": {"cards": [], "is_pass": True, "confidence": .98}}),
    ]
    ledger = project_turn_slots(observations, [])

    assert ledger["slots"][0]["kind"] == "initial_context"
    assert ledger["slots"][0]["status"] == "needs_review"
    assert ledger["slots"][1]["actor"] == "self"
