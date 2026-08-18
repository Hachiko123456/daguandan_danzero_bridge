from __future__ import annotations

import pytest

from daguandan_bridge.gui.hand_preselection import PreselectionPlan
from daguandan_bridge.infrastructure.win32_hand_preselector import (
    Win32HandPreselector,
    _INTER_CARD_DELAY_SECONDS,
    _is_inside_client,
    _move_flags,
    _send_left_clicks,
)
from daguandan_bridge.models import ClientRect


def test_preselector_rejects_without_attempting_input_off_windows(monkeypatch):
    plan = PreselectionPlan(
        request_id="ADV-0001-0002",
        expected_client_rect=ClientRect(10, 20, 100, 50),
        points=((20, 30),),
    )
    monkeypatch.setattr(
        "daguandan_bridge.infrastructure.win32_hand_preselector.sys.platform",
        "linux",
    )

    result = Win32HandPreselector(("game",)).preselect_hand_cards(plan)

    assert result.status == "rejected"
    assert "Windows" in result.detail


def test_point_validation_accepts_only_client_area():
    rect = ClientRect(10, 20, 100, 50)

    assert _is_inside_client((10, 20), rect)
    assert _is_inside_client((109, 69), rect)
    assert not _is_inside_client((110, 69), rect)
    assert not _is_inside_client((109, 70), rect)


def test_absolute_pointer_motion_uses_the_entire_virtual_desktop():
    assert _move_flags() & 0x4000  # MOUSEEVENTF_VIRTUALDESK


class _User32Stub:
    def __init__(self, sends):
        self.sends = iter(sends)
        self.calls = []

    def GetSystemMetrics(self, index):
        return {76: -1000, 77: 0, 78: 3000, 79: 1200}[index]

    def SendInput(self, count, _buffer, _size):
        self.calls.append(count)
        return next(self.sends)


def test_each_recommended_card_is_injected_and_waited_for_independently():
    user32 = _User32Stub([3, 3, 3])
    waits = []

    _send_left_clicks(
        ((10, 20), (30, 40), (50, 60)),
        user32=user32,
        wait=waits.append,
    )

    assert user32.calls == [3, 3, 3]
    assert waits == [_INTER_CARD_DELAY_SECONDS, _INTER_CARD_DELAY_SECONDS]


def test_failed_card_input_stops_before_later_recommended_cards():
    user32 = _User32Stub([3, 2, 3])
    waits = []

    with pytest.raises(RuntimeError, match="2/3"):
        _send_left_clicks(
            ((10, 20), (30, 40), (50, 60)),
            user32=user32,
            wait=waits.append,
        )

    assert user32.calls == [3, 3]
    assert waits == [_INTER_CARD_DELAY_SECONDS]
