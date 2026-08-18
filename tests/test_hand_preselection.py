from __future__ import annotations

from datetime import datetime

import numpy as np

from daguandan_bridge.capture_service import FrameSnapshot
from daguandan_bridge.domain.recognition import RecognitionAnnotation, RecognitionResult
from daguandan_bridge.gui.hand_preselection import HandPreselectionPlanner
from daguandan_bridge.image_io import StandardizationResult
from daguandan_bridge.models import Box, ClientRect
from daguandan_bridge.window_capture import CapturedStandardizedFrame


def _frame() -> FrameSnapshot:
    standardization = StandardizationResult(
        image=np.zeros((900, 1700, 3), dtype=np.uint8),
        source_size=(1000, 600),
        source_viewport=Box(100, 50, 800, 400),
        content_box=Box(50, 40, 1600, 800),
        scale=2.0,
        padding=(50, 40, 50, 60),
        aspect_error=0.0,
        aspect_compatible=True,
    )
    return FrameSnapshot(
        CapturedStandardizedFrame(
            standardization=standardization,
            rect=ClientRect(left=1000, top=200, width=1000, height=600),
            backend="printwindow",
            dpi=96,
            window_title="game",
        ),
        captured_at=datetime.now().astimezone(),
    )


def _recognition(*, hand=("3S", "3S", "4H"), annotations=None):
    if annotations is None:
        annotations = (
            RecognitionAnnotation("3S", (210, 320, 20, 30), 0.99, "hand"),
            RecognitionAnnotation("3S", (250, 320, 20, 30), 0.99, "hand"),
            RecognitionAnnotation("4H", (290, 320, 20, 30), 0.99, "hand"),
        )
    return RecognitionResult(
        round_level="2",
        wild_rank="2",
        current_player="self",
        lead_player="self",
        my_hand=tuple(hand),
        events=(),
        field_confidences={},
        sources={},
        unresolved_fields=(),
        diagnostics=(),
        annotations=tuple(annotations),
    )


def test_planner_maps_each_recommended_duplicate_to_a_distinct_client_point():
    result = HandPreselectionPlanner().plan(
        request_id="ADV-0001-0002",
        advice_cards=("3S", "3S"),
        expected_hand=("3S", "3S", "4H"),
        recognition=_recognition(),
        frame=_frame(),
    )

    assert result.status == "planned"
    assert result.plan is not None
    assert result.plan.points == ((1185, 398), (1205, 398))
    assert result.plan.expected_client_rect == ClientRect(1000, 200, 1000, 600)


def test_planner_rejects_when_fresh_hand_counter_does_not_match_live_state():
    result = HandPreselectionPlanner().plan(
        request_id="ADV-0001-0002",
        advice_cards=("3S",),
        expected_hand=("3S", "3S", "4H"),
        recognition=_recognition(hand=("3S", "4H", "4H")),
        frame=_frame(),
    )

    assert result.status == "rejected"
    assert "不一致" in result.detail


def test_planner_rejects_a_hand_annotation_outside_standardized_content():
    result = HandPreselectionPlanner().plan(
        request_id="ADV-0001-0002",
        advice_cards=("3S",),
        expected_hand=("3S", "3S", "4H"),
        recognition=_recognition(
            annotations=(
                RecognitionAnnotation("3S", (0, 0, 20, 30), 0.99, "hand"),
                RecognitionAnnotation("3S", (250, 320, 20, 30), 0.99, "hand"),
                RecognitionAnnotation("4H", (290, 320, 20, 30), 0.99, "hand"),
            )
        ),
        frame=_frame(),
    )

    assert result.status == "rejected"
    assert "坐标" in result.detail
