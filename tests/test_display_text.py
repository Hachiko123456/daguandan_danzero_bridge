from __future__ import annotations

from daguandan_bridge.live.display_text import compact_cards_text


def test_compact_cards_text_handles_missing_suit_options_for_unknown_card():
    assert compact_cards_text(("2?", "2C", "2C", "2D", "6H")) == "2? 2♣ 2♣ 2♦ 6♥"
