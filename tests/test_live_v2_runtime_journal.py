from __future__ import annotations

import json
import gzip
import zipfile

from daguandan_bridge.automatic_log_delivery import AutomaticLogDeliveryService
from daguandan_bridge.application.live_v2_diagnostic_projection import MAX_PAYLOAD_BYTES
from daguandan_bridge.application.live_v2_runtime_journal import LiveV2RuntimeJournal
from daguandan_bridge.live.session_store import LiveSessionStore
from daguandan_bridge.live_v2.candidates import (
    ActionCandidate, ActionKind, CandidateReason, ConfirmationReason,
    ConfirmedAction,
)
from daguandan_bridge.live_v2.identity import FrameIdentity, Seat, StateVersion, VersionIdentity
from daguandan_bridge.live_v2.observations import (
    ObservationKind, ObservationReason, SeatObservation,
)
from daguandan_bridge.live_v2.results import (
    EngineUpdate, EngineUpdateReason, GapPhase, GapReason, GapState,
)


class Store:
    def __init__(self): self.rows = []
    def append_observation(self, value): self.rows.append(value)


def _fixtures(count: int = 1):
    version = VersionIdentity("journal", 3, 1, 9, 1)
    before = StateVersion("journal", 0, 0)
    after = StateVersion("journal", 1, 1)
    observations = []
    candidates = []
    for index in range(count):
        first = FrameIdentity("journal", 3, 100 + index * 2, 10_000 + index * 20, "roi", "source")
        last = FrameIdentity("journal", 3, 101 + index * 2, 10_010 + index * 20, "roi", "source")
        observations.append(SeatObservation(
            f"obs-{index}", last, Seat.RIGHT, ObservationKind.PLAY, ("3D",), .93456,
            ObservationReason.CARDS_RECOGNIZED, last.captured_ms,
            (("3D",),),
            ("cards_recognized=1", "path=C:/Users/private/secret.png", "X" * 500),
        ))
        candidates.append(ActionCandidate(
            f"candidate-{index}", version, Seat.RIGHT, ActionKind.PLAY,
            ("3D",), (("3D",),), (f"e-{index}-1", f"e-{index}-2"),
            7, first, last, last.captured_ms, .93456, CandidateReason.STABLE_PLAY,
            diagnostics=("play_quality=stable", "template_dump=C:/secret/template.png"),
        ))
    confirmed = ConfirmedAction.from_candidate(
        action_id="action-1", candidate=candidates[0], version_before=before,
        version_after=after, processing_ms=candidates[0].processing_ms,
        reason=ConfirmationReason.RULE_VALIDATED,
    )
    gap = GapState(
        version, GapPhase.BLOCKING, GapReason.RULE_REJECTION,
        (Seat.RIGHT, Seat.OPPOSITE), ("e-1", "e-2"), 10_000, 10_500,
    )
    return EngineUpdate(
        version, EngineUpdateReason.GAP_CHANGED, 10_500, 10_500,
        tuple(observations), tuple(candidates), (confirmed,), gap,
    )


def _stress_fixtures(
    hostile: str, diagnostics: tuple[str, ...],
) -> EngineUpdate:
    huge = 10 ** 3_000
    version = VersionIdentity(hostile, huge, huge, huge, huge)
    before = StateVersion(hostile, 0, 0)
    after = StateVersion(hostile, 1, 1)
    first = FrameIdentity(hostile, huge, 1, huge, "roi", "source")
    last = FrameIdentity(hostile, huge, 2, huge + 1, "roi", "source")
    cards = (hostile,) * 27
    observation = SeatObservation(
        "observation", last, Seat.RIGHT, ObservationKind.PLAY, cards, .9,
        ObservationReason.CARDS_RECOGNIZED, huge + 1,
        tuple((card,) for card in cards),
        diagnostics,
    )
    candidate = ActionCandidate(
        hostile, version, Seat.RIGHT, ActionKind.PLAY, cards,
        tuple((card,) for card in cards),
        tuple(f"evidence-{index}" for index in range(100)),
        huge, first, last, huge + 1, .9, CandidateReason.STABLE_PLAY,
        diagnostics=diagnostics,
    )
    confirmed = ConfirmedAction.from_candidate(
        action_id=hostile, candidate=candidate, version_before=before,
        version_after=after, processing_ms=huge + 1,
        reason=ConfirmationReason.RULE_VALIDATED,
    )
    gap = GapState(
        version, GapPhase.BLOCKING, GapReason.RULE_REJECTION,
        (Seat.RIGHT, Seat.OPPOSITE),
        tuple(f"gap-evidence-{index}" for index in range(100)), huge, huge + 1,
    )
    return EngineUpdate(
        version, EngineUpdateReason.GAP_CHANGED, huge + 1, huge + 1,
        (observation,) * 20, (candidate,) * 20, (confirmed,) * 20, gap,
    )


def test_engine_journal_records_bounded_actionable_evidence() -> None:
    store = Store()
    LiveV2RuntimeJournal(store).append(_fixtures())
    row = store.rows[0]

    assert row["schema"] == "guandan.live-v2.engine-update/2"
    assert row["observations"][0] == {
        "seat": "right", "kind": "play", "cards": ["3D"],
        "confidence": .9346, "reason": "cards_recognized",
        "frame_sequence": 101, "captured_ms": 10_010,
        "diagnostic_codes": ["cards_recognized", "path", "X" * 48],
    }
    candidate = row["candidates"][0]
    assert candidate["action_epoch"] == 7
    assert candidate["first_frame"] == {"sequence": 100, "captured_ms": 10_000}
    assert candidate["last_frame"] == {"sequence": 101, "captured_ms": 10_010}
    assert candidate["evidence_count"] == 2
    assert candidate["diagnostic_codes"] == ["play_quality", "template_dump"]
    assert row["gap"] == {
        "phase": "blocking", "reason": "rule_rejection",
        "expected_seats": ["right", "opposite"], "evidence_count": 2,
    }
    assert row["confirmed_actions"][0]["cards"] == ["3D"]


def test_projection_has_hard_size_limits_and_does_not_leak_paths_or_images() -> None:
    path = "C:/Users/private/secret-frame.png"
    hostile = path + ("😀" * 10_000)
    store = Store()
    LiveV2RuntimeJournal(store).append(_stress_fixtures(
        hostile,
        (hostile, "/home/private/template.png", "trace=" + hostile),
    ))
    row = store.rows[0]
    payload = json.dumps(row, ensure_ascii=False)

    assert 1 <= len(row["observations"]) <= 8
    assert 1 <= len(row["candidates"]) <= 8
    assert 1 <= len(row["confirmed_actions"]) <= 8
    assert len(row["observations"]) + row["truncated"]["observations"] == 20
    assert len(row["candidates"]) + row["truncated"]["candidates"] == 20
    assert (
        len(row["confirmed_actions"])
        + row["truncated"]["confirmed_actions"]
        == 20
    )
    assert len(payload.encode("utf-8")) < 16_384
    assert "C:/Users/private" not in payload
    assert "/home/private" not in payload
    assert "secret-frame.png" not in payload
    assert "template.png" not in payload
    assert "private" not in payload
    assert "image" not in row
    assert "annotations" not in payload


def test_projection_compacts_maximum_safe_text_without_losing_summary_types() -> None:
    diagnostics = tuple(f"diagnostic_{index}_" + ("X" * 10_000) for index in range(8))
    store = Store()
    LiveV2RuntimeJournal(store).append(_stress_fixtures("A" * 10_000, diagnostics))
    row = store.rows[0]

    size = len(json.dumps(row, ensure_ascii=False).encode("utf-8"))
    assert size <= MAX_PAYLOAD_BYTES < 16_384
    assert row["observations"]
    assert row["candidates"]
    assert row["confirmed_actions"]
    assert row["gap"] is not None
    assert sum(row["truncated"].values()) > 0


def test_engine_projection_round_trips_through_session_log_and_automatic_bundle(
    tmp_path,
) -> None:
    store = LiveSessionStore(
        tmp_path / "profiles", "profile",
        session_id="game_20260907_120000_journal",
    )
    store.start({"recording_mode": "game"})
    LiveV2RuntimeJournal(store).append(_fixtures())
    store.seal(frame_count=0, dropped_frames=0)

    with gzip.open(store.observations_gzip_path, "rt", encoding="utf-8") as handle:
        canonical = json.loads(handle.readline())
    assert canonical["schema"] == "guandan.live-v2.engine-update/2"

    result = AutomaticLogDeliveryService(
        documents_root=tmp_path / "Documents",
        fallback_root=tmp_path / "fallback",
    ).export(store.directory)
    assert result.diagnostic_zip_path is not None
    with zipfile.ZipFile(result.diagnostic_zip_path) as archive:
        bundled = gzip.decompress(
            archive.read("session/observations.jsonl.gz")
        ).decode("utf-8")
    assert json.loads(bundled)["schema"] == "guandan.live-v2.engine-update/2"
