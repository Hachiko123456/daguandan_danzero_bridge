"""Build a read-only index of recorded sessions for replay and regression tests.

The source session tree is deliberately treated as an immutable evidence store.  This
module only reads it and writes a small, deterministic catalog to a caller-selected
directory; it never creates ``derived`` files below a session and never copies video.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Literal

from ..danzero.state import RANKS, SEATS
from ..live.truth_log import TruthLog, TruthTurn, load_truth_log


CatalogClass = Literal["gold", "silver", "raw"]
Split = Literal["development", "regression", "release_gate", "unassigned"]

# Keep this explicit and reviewable.  A session must not silently move groups because
# its name happened to hash differently after a refactor.
FIXED_SPLITS: dict[str, tuple[str, ...]] = {
    "development": (
        "game_20260814_004447_aab3dc",
        "game_20260815_003925_66b329",
        "game_20260815_014247_14ec80",
        "game_20260822_142942_e356d4",
        "game_20260816_154444_392ea8",
    ),
    "regression": (
        "game_20260816_125402_1687ea",
        "game_20260822_130309_407888",
    ),
    "release_gate": (
        "game_20260822_002135_9c2328",
        "game_20260825_155452_7d2eb3",
    ),
}
_SPLIT_BY_SESSION = {
    session_id: split for split, session_ids in FIXED_SPLITS.items() for session_id in session_ids
}
_RANK_INDEX = {rank: index for index, rank in enumerate(RANKS)}


@dataclass(frozen=True)
class CatalogBuildResult:
    output_dir: Path
    catalog_path: Path
    cases_path: Path
    gaps_path: Path
    session_count: int
    case_count: int
    truth_session_count: int
    draft_truth_session_count: int = 0
    verified_truth_session_count: int = 0


class SessionDatasetCatalogBuilder:
    """Discover sessions and emit a compact replay catalog without mutating sources."""

    schema = "guandan.session-dataset-catalog/1"
    case_schema = "guandan.session-dataset-case/1"
    gaps_schema = "guandan.session-dataset-gaps/1"

    def build(
        self,
        sessions_root: Path,
        output_dir: Path,
        *,
        hash_video: bool = False,
    ) -> CatalogBuildResult:
        root = Path(sessions_root).resolve()
        output = Path(output_dir).resolve()
        if not root.is_dir():
            raise ValueError(f"sessions root does not exist: {root}")
        sessions = tuple(sorted((path for path in root.iterdir() if path.is_dir() and not path.name.startswith(".")), key=lambda p: p.name))
        records: list[dict[str, Any]] = []
        cases: list[dict[str, Any]] = []
        draft_truth_ids: set[str] = set()
        verified_truth_ids: set[str] = set()
        for session in sessions:
            record, session_cases = self._index_session(root, session, hash_video=hash_video)
            records.append(record)
            cases.extend(session_cases)
            if record.get("truth_qualification") == "gold":
                verified_truth_ids.add(session.name)
            elif record.get("truth_qualification") == "draft_truth":
                draft_truth_ids.add(session.name)

        gaps = self._coverage_gaps(records, cases, verified_truth_ids)
        catalog = {
            "schema": self.schema,
            "schema_version": 1,
            "sessions_root_name": root.name,
            "session_count": len(records),
            "truth_session_count": len(draft_truth_ids) + len(verified_truth_ids),
            "draft_truth_session_count": len(draft_truth_ids),
            "verified_truth_session_count": len(verified_truth_ids),
            "case_count": len(cases),
            "fixed_splits": {key: list(value) for key, value in FIXED_SPLITS.items()},
            "sessions": records,
        }
        output.mkdir(parents=True, exist_ok=True)
        catalog_path = output / "catalog.json"
        cases_path = output / "cases.jsonl"
        gaps_path = output / "coverage_gaps.json"
        _write_json(catalog_path, catalog)
        _write_jsonl(cases_path, cases)
        _write_json(gaps_path, gaps)
        return CatalogBuildResult(
            output_dir=output,
            catalog_path=catalog_path,
            cases_path=cases_path,
            gaps_path=gaps_path,
            session_count=len(records),
            case_count=len(cases),
            truth_session_count=len(draft_truth_ids) + len(verified_truth_ids),
            draft_truth_session_count=len(draft_truth_ids),
            verified_truth_session_count=len(verified_truth_ids),
        )

    def _index_session(
        self,
        root: Path,
        session: Path,
        *,
        hash_video: bool,
    ) -> tuple[dict[str, Any], list[dict[str, Any]]]:
        session_id = session.name
        truth_path = session / "truth_log.json"
        timeline_path = session / "timeline.jsonl"
        manifest_path = session / "manifest.json"
        video_path = session / "video" / "game.avi"
        frame_index_path = session / "video" / "frame_index.jsonl"
        has_truth = truth_path.is_file()
        has_timeline = timeline_path.is_file()
        has_video = video_path.is_file()
        has_frame_index = frame_index_path.is_file()
        classification: CatalogClass = "silver" if has_truth or has_timeline else "raw"
        split: Split = _SPLIT_BY_SESSION.get(session_id, "unassigned")  # type: ignore[assignment]
        truth: TruthLog | None = None
        truth_error: str | None = None
        if has_truth:
            try:
                truth = load_truth_log(truth_path, session_id=session_id)
            except Exception as exc:  # catalog must still expose a broken evidence item
                truth_error = f"{type(exc).__name__}: {exc}"
        if truth is not None and _is_verified_truth(truth):
            classification = "gold"
        truth_comparison = (
            _compare_truth_timeline(truth, _read_jsonl(timeline_path))
            if truth is not None and has_timeline
            else {
                "status": "not_comparable",
                "mismatch_count": 0,
                "mismatches": [],
            }
        )

        frame_count = _count_jsonl(frame_index_path)
        files = {
            "manifest": _file_info(root, manifest_path),
            "truth_log": _file_info(root, truth_path),
            "timeline": _file_info(root, timeline_path),
            "video": _file_info(root, video_path, hash_file=hash_video),
            "frame_index": _file_info(root, frame_index_path),
        }
        record: dict[str, Any] = {
            "session_id": session_id,
            "relative_session": session.relative_to(root).as_posix(),
            "classification": classification,
            "split": split,
            "has_truth_log": has_truth,
            "truth_log_valid": truth is not None,
            "truth_log_error": truth_error,
            "truth_vs_timeline": truth_comparison,
            "has_timeline": has_timeline,
            "has_video": has_video,
            "has_frame_index": has_frame_index,
            "frame_count": frame_count,
            "files": files,
        }
        session_cases: list[dict[str, Any]] = []
        if truth is not None:
            frame_values = [frame for turn in truth.turns for frame in turn.evidence.frame_indices]
            record.update(
                {
                    "lead_seat": truth.initial_state.lead_player,
                    "round_level": truth.initial_state.round_level,
                    "turn_count": len(truth.turns),
                    "truth_frame_range": _frame_range(frame_values),
                    "label_status": truth.label_status,
                    "provenance_source": truth.provenance.source,
                    "truth_qualification": (
                        "gold" if _is_verified_truth(truth) else "draft_truth"
                    ),
                    "outcome": truth.outcome.to_dict(),
                }
            )
            session_cases = self._truth_cases(root, session, truth, split)
        return record, session_cases

    def _truth_cases(self, root: Path, session: Path, truth: TruthLog, split: Split) -> list[dict[str, Any]]:
        cases: list[dict[str, Any]] = []
        previous: TruthTurn | None = None
        for turn in truth.turns:
            cases.append(self._turn_case(root, session, truth, turn, previous, split))
            previous = turn
        return cases

    def _turn_case(
        self,
        root: Path,
        session: Path,
        truth: TruthLog,
        turn: TruthTurn,
        previous: TruthTurn | None,
        split: Split,
    ) -> dict[str, Any]:
        frame_indices = tuple(int(value) for value in turn.evidence.frame_indices)
        tags = _scenario_tags(truth, turn, previous)
        return {
            "schema": self.case_schema,
            "case_id": f"{truth.source_session_id}:turn:{turn.index:04d}",
            "session_id": truth.source_session_id,
            "relative_session": session.relative_to(root).as_posix(),
            "split": split,
            "truth_status": (
                "human_confirmed" if _is_verified_truth(truth) else "draft_truth"
            ),
            "label_status": truth.label_status,
            "provenance_source": truth.provenance.source,
            "turn_index": turn.index,
            "trick_id": turn.trick_id,
            "actor": turn.actor,
            "is_pass": turn.is_pass,
            "cards": list(turn.cards),
            "card_count": len(turn.cards),
            "frame_range": _frame_range(frame_indices),
            "frame_indices": list(frame_indices),
            "monotonic_ms": turn.monotonic_ms,
            "lead_seat": truth.initial_state.lead_player,
            "round_level": truth.initial_state.round_level,
            "video_path": _relative_path(root, session / truth.source_video),
            "frame_index_path": _relative_path(root, session / truth.frame_index_path),
            "scenario_tags": tags,
            "uncertainty": list(turn.uncertainty),
            "move_semantics": turn.move_semantics,
        }

    @staticmethod
    def _coverage_gaps(
        records: list[dict[str, Any]],
        cases: list[dict[str, Any]],
        truth_ids: set[str],
    ) -> dict[str, Any]:
        gold = [record for record in records if record["classification"] == "gold"]
        gold_cases = [case for case in cases if case["truth_status"] == "human_confirmed"]
        tags = sorted({tag for case in gold_cases for tag in case["scenario_tags"]})
        counts = {tag: sum(tag in case["scenario_tags"] for case in gold_cases) for tag in tags}
        required_tags = (
            "opening",
            "pass",
            "single",
            "pair",
            "triple",
            "long_play",
            "wildcard",
            "joker",
            "pass_chain",
            "trick_boundary",
            "terminal",
        )
        missing_tags = [tag for tag in required_tags if counts.get(tag, 0) == 0]
        fixed_ids = set(_SPLIT_BY_SESSION)
        missing_fixed = sorted(fixed_ids - truth_ids)
        unexpected_gold = sorted(truth_ids - fixed_ids)
        split_counts = {split: sum(record["split"] == split for record in gold) for split in FIXED_SPLITS}
        return {
            "schema": SessionDatasetCatalogBuilder.gaps_schema,
            "schema_version": 1,
            "gold_session_count": len(gold),
            "gold_case_count": sum(record.get("turn_count", 0) for record in gold),
            "split_counts": split_counts,
            "missing_fixed_split_sessions": missing_fixed,
            "unexpected_gold_sessions": unexpected_gold,
            "scenario_counts": counts,
            "required_scenario_tags": list(required_tags),
            "missing_scenario_tags": missing_tags,
            "silver_session_count": sum(record["classification"] == "silver" for record in records),
            "raw_session_count": sum(record["classification"] == "raw" for record in records),
            "notes": [
                "Only TruthLog label_status=verified is treated as gold truth.",
                "Draft TruthLogs remain indexed as silver and never become gold without explicit review.",
                "A case references source frames; no AVI or other media is copied into the catalog.",
            ],
        }


def _scenario_tags(truth: TruthLog, turn: TruthTurn, previous: TruthTurn | None) -> list[str]:
    tags: set[str] = {"truth_log", f"actor_{turn.actor}"}
    if turn.index == 1:
        tags.add("opening")
    if turn.is_pass:
        tags.add("pass")
        if previous is not None and previous.is_pass:
            tags.add("pass_chain")
        if _is_third_pass_after_play(truth.turns, turn.index):
            tags.add("three_pass_reset")
    else:
        count = len(turn.cards)
        if count == 1:
            tags.add("single")
        elif count == 2:
            tags.add("pair")
        elif count == 3:
            tags.add("triple")
        else:
            tags.add("multi_card")
        if count >= 5:
            tags.add("long_play")
        if count > 5:
            tags.add("large_play")
        ranks = [card[:-1] if card.endswith(("S", "H", "C", "D")) else card for card in turn.cards]
        if any(rank in {"small_joker", "big_joker"} for rank in ranks):
            tags.add("joker")
        if any(rank == truth.initial_state.round_level for rank in ranks):
            tags.add("wildcard")
        if len(set(ranks)) == 1:
            tags.add("same_rank")
        suits = [card[-1] for card in turn.cards if card[-1:] in {"S", "H", "C", "D"}]
        if len(suits) == count and len(set(suits)) == 1 and count > 1:
            tags.add("same_suit")
        if _are_consecutive(ranks):
            tags.add("consecutive_ranks")
    if previous is not None and previous.trick_id != turn.trick_id:
        tags.add("trick_boundary")
    if turn.index == len(truth.turns):
        tags.add("terminal")
    return sorted(tags)


def _is_verified_truth(truth: TruthLog) -> bool:
    """Require an explicit verified label; file presence is not ground truth."""

    return truth.label_status == "verified"


def _compare_truth_timeline(
    truth: TruthLog, timeline: list[dict[str, Any]]
) -> dict[str, Any]:
    """Flag disagreements without treating either side as an oracle."""

    actual: list[dict[str, Any]] = []
    for event in timeline:
        if event.get("event_type") not in {"player_played", "player_passed"}:
            continue
        payload = event.get("payload")
        payload = payload if isinstance(payload, dict) else {}
        actual.append(
            {
                "actor": str(event.get("actor") or ""),
                "cards": sorted(str(card) for card in payload.get("cards", ())),
                "is_pass": event.get("event_type") == "player_passed"
                or bool(payload.get("is_pass")),
                "event_id": event.get("event_id"),
                "turn_id": event.get("turn_id"),
            }
        )
    mismatches: list[dict[str, Any]] = []
    for index, expected in enumerate(truth.turns):
        expected_actor = getattr(expected.actor, "value", str(expected.actor))
        expected_row = {
            "actor": str(expected_actor),
            "cards": sorted(str(card) for card in expected.cards),
            "is_pass": bool(expected.is_pass),
        }
        actual_row = actual[index] if index < len(actual) else None
        actual_compare = (
            None
            if actual_row is None
            else {
                "actor": actual_row["actor"],
                "cards": actual_row["cards"],
                "is_pass": actual_row["is_pass"],
            }
        )
        if actual_compare != expected_row:
            mismatches.append(
                {
                    "turn_index": index + 1,
                    "expected": expected_row,
                    "actual": actual_compare,
                    "event_id": actual_row.get("event_id") if actual_row else None,
                    "timeline_turn_id": actual_row.get("turn_id") if actual_row else None,
                }
            )
    if len(actual) > len(truth.turns):
        for index, row in enumerate(actual[len(truth.turns) :], start=len(truth.turns) + 1):
            mismatches.append(
                {
                    "turn_index": index,
                    "expected": None,
                    "actual": {
                        "actor": row["actor"],
                        "cards": row["cards"],
                        "is_pass": row["is_pass"],
                    },
                    "event_id": row.get("event_id"),
                    "timeline_turn_id": row.get("turn_id"),
                }
            )
    return {
        "status": "match" if not mismatches else "conflict",
        "mismatch_count": len(mismatches),
        "first_mismatch_turn": mismatches[0]["turn_index"] if mismatches else None,
        "mismatches": mismatches[:64],
    }


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    rows: list[dict[str, Any]] = []
    try:
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    value = json.loads(line)
                    if isinstance(value, dict):
                        rows.append(value)
    except (OSError, json.JSONDecodeError):
        return []
    return rows


def _is_third_pass_after_play(turns: Iterable[TruthTurn], index: int) -> bool:
    sequence = list(turns)
    position = index - 1
    if position < 0 or not sequence[position].is_pass:
        return False
    passes = 0
    for item in reversed(sequence[:position]):
        if item.is_pass:
            passes += 1
        else:
            break
    return passes == 2


def _are_consecutive(ranks: list[str]) -> bool:
    numeric = sorted({_RANK_INDEX[rank] for rank in ranks if rank in _RANK_INDEX})
    return len(numeric) == len(ranks) and len(numeric) > 1 and numeric[-1] - numeric[0] + 1 == len(numeric)


def _frame_range(frames: Iterable[int]) -> dict[str, int | None]:
    values = tuple(sorted(set(int(frame) for frame in frames)))
    return {"start": values[0] if values else None, "end": values[-1] if values else None}


def _relative_path(root: Path, path: Path) -> str:
    try:
        return path.resolve().relative_to(root).as_posix()
    except ValueError:
        return path.as_posix()


def _file_info(root: Path, path: Path, *, hash_file: bool = True) -> dict[str, Any]:
    if not path.is_file():
        return {"relative_path": _relative_path(root, path), "exists": False}
    info: dict[str, Any] = {
        "relative_path": _relative_path(root, path),
        "exists": True,
        "bytes": path.stat().st_size,
    }
    if hash_file:
        info["sha256"] = _sha256(path)
    return info


def _count_jsonl(path: Path) -> int:
    if not path.is_file():
        return 0
    count = 0
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                count += 1
    return count


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_jsonl(path: Path, records: Iterable[dict[str, Any]]) -> None:
    path.write_text(
        "".join(json.dumps(record, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n" for record in records),
        encoding="utf-8",
    )


__all__ = ["FIXED_SPLITS", "CatalogBuildResult", "SessionDatasetCatalogBuilder"]
