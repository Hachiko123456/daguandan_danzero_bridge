from __future__ import annotations

import pytest

from daguandan_bridge.live_v2.observations import require_suit_options


def test_rank_only_card_accepts_concrete_same_rank_entities() -> None:
    require_suit_options(("7?",), (("7C", "7D"),))


def test_rank_only_card_rejects_wrong_rank_or_suit_only_options() -> None:
    with pytest.raises(ValueError, match="preserve the observed rank"):
        require_suit_options(("7?",), (("8C",),))
    with pytest.raises(ValueError, match="concrete physical cards"):
        require_suit_options(("7?",), (("C", "D"),))


def test_exact_card_still_requires_itself_and_concrete_options() -> None:
    with pytest.raises(ValueError, match="must occur"):
        require_suit_options(("7C",), (("7D",),))
    with pytest.raises(ValueError, match="must occur"):
        require_suit_options(("7C",), (("C",),))


def test_rank_only_options_remain_one_to_one() -> None:
    with pytest.raises(ValueError, match="align"):
        require_suit_options(("7?", "8?"), (("7C", "7D"),))
