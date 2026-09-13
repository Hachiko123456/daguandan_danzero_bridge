"""Formal V2 acceptance audit for an unverified batch scan.

This audit combines the repository's detailed structural/turn checks with
rule-engine checks for legal play patterns and whether each non-lead play beats
the current table. It never modifies scan outputs and never opens video files.
"""
from __future__ import annotations

import argparse
import gzip
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from daguandan_bridge.application.session_workbench import inspect_sessions  # noqa: E402
from daguandan_bridge.danzero.rules import actions_for_cards, play_beats_table, to_engine_card  # noqa: E402
from daguandan_bridge.live.turns import (  # noqa: E402
    TURN_ORDER,
    next_active_seat,
    project_trick_turn,
    round_is_decided,
)
from verify_batch_scan_detailed import verify_session  # noqa: E402

SEATS = list(TURN_ORDER)
REQUIRED_FILES = (
    "scan_manifest.json",
    "frame_observations.jsonl.gz",
    "raw_action_trace.jsonl",
    "action_trace.jsonl",
    "opening_candidates.json",
    "scan_summary.json",
    "turn_slots.json",
)
PARTIAL_FRAME_THRESHOLD = 400


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def read_observations(path: Path) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    with gzip.open(path, "rt", encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                result.append(json.loads(line))
    return result


def action_label(action: dict[str, Any], idx: int) -> str:
    cards = "PASS" if action.get("is_pass") else " ".join(str(c) for c in action.get("cards") or ())
    return (
        f"#{idx} {action.get('actor', '?')} {cards} "
        f"(帧 {action.get('frame_start', '?')}-{action.get('frame_end', '?')})"
    )


def violation(
    severity: str,
    code: str,
    idx: int | None,
    action: dict[str, Any] | None,
    detail: str,
) -> dict[str, Any]:
    return {
        "severity": severity,
        "code": code,
        "action_idx": idx,
        "actor": (action or {}).get("actor", "-"),
        "frame_start": (action or {}).get("frame_start"),
        "frame_end": (action or {}).get("frame_end"),
        "cards": list((action or {}).get("cards") or ()),
        "label": action_label(action or {}, idx or 0) if action is not None else f"#{idx or '-'}",
        "detail": detail,
    }


def _self_hand_size(observations: list[dict[str, Any]]) -> int:
    sizes = []
    for row in observations:
        opening = row.get("opening") or {}
        hand = opening.get("my_hand") or ()
        if hand:
            sizes.append(len(hand))
    return max(sizes) if sizes else 27


def _levels(observations: list[dict[str, Any]]) -> Counter[str]:
    counts: Counter[str] = Counter()
    for row in observations:
        level = (row.get("opening") or {}).get("round_level")
        if isinstance(level, str) and level:
            counts[level] += 1
    return counts


def level_context(session_id: str, observations: list[dict[str, Any]]) -> dict[str, Any]:
    source_truth = ROOT / "data" / "profiles" / "tencent_daguandan" / "sessions" / session_id / "truth_log.json"
    truth_level: str | None = None
    if source_truth.is_file():
        try:
            raw = read_json(source_truth)
            initial = raw.get("initial_state") or {}
            if isinstance(initial.get("round_level"), str) and initial.get("round_level"):
                truth_level = str(initial["round_level"])
        except Exception:
            truth_level = None
    observed = _levels(observations)
    if truth_level:
        return {
            "level": truth_level,
            "source": "source_truth_log",
            "confidence": "authoritative_for_level",
            "observed_levels": dict(observed),
            "ambiguous_observation": len(observed) > 1,
        }
    if len(observed) == 1:
        level = next(iter(observed))
        return {
            "level": level,
            "source": "video_observation_consensus",
            "confidence": "manual_confirmation_required",
            "observed_levels": dict(observed),
            "ambiguous_observation": False,
        }
    if observed:
        mode = observed.most_common(1)[0][0]
        return {
            "level": mode,
            "source": "ambiguous_video_observation_mode",
            "confidence": "manual_confirmation_required",
            "observed_levels": dict(observed),
            "ambiguous_observation": True,
        }
    return {
        "level": None,
        "source": "unavailable",
        "confidence": "manual_confirmation_required",
        "observed_levels": {},
        "ambiguous_observation": False,
    }


def _advance_finished(
    actor: str,
    cards: list[str],
    is_pass: bool,
    remaining: dict[str, int],
    finished: set[str],
) -> None:
    if is_pass or actor not in remaining:
        return
    remaining[actor] -= len(cards)
    if remaining[actor] <= 0:
        remaining[actor] = 0
        finished.add(actor)


def rule_checks(
    session_id: str,
    actions: list[dict[str, Any]],
    observations: list[dict[str, Any]],
    *,
    partial: bool,
) -> dict[str, Any]:
    context = level_context(session_id, observations)
    level = context.get("level")
    self_hand = _self_hand_size(observations)
    # Do not use the modal level for strict semantic failure when the video
    # itself reports conflicting levels. Such cases remain manual-review
    # items; only authoritative truth or a single observed level is used for
    # deterministic engine checks.
    semantic_level = (
        None
        if context.get("ambiguous_observation") and context.get("source") != "source_truth_log"
        else level
    )
    remaining = {seat: 27 for seat in SEATS}
    remaining["self"] = self_hand
    finished: set[str] = set()
    table_cards: tuple[str, ...] = ()
    current_leader: str | None = None
    table_semantically_valid: bool | None = None
    passed: set[str] = set()
    legal_violations: list[dict[str, Any]] = []
    beat_violations: list[dict[str, Any]] = []
    physical_violations: list[dict[str, Any]] = []
    manual_context: list[dict[str, Any]] = []
    manual_indeterminate: list[dict[str, Any]] = []
    global_cards: Counter[str] = Counter()

    for idx, action in enumerate(actions, 1):
        cards = [str(card) for card in (action.get("cards") or ())]
        actor = str(action.get("actor", ""))
        is_pass = bool(action.get("is_pass", False))
        if is_pass:
            if cards:
                legal_violations.append(violation("P0", "V2", idx, action, "PASS 动作携带牌面"))
        else:
            if not cards:
                legal_violations.append(violation("P0", "V2", idx, action, "非 PASS 动作没有牌"))
            unknown = [card for card in cards if "?" in card]
            if unknown:
                physical_violations.append(
                    violation("P0", "V2", idx, action, f"最终动作仍含未知牌面: {' '.join(unknown)}")
                )
            for card in cards:
                try:
                    to_engine_card(card)
                except Exception:
                    physical_violations.append(violation("P0", "V2", idx, action, f"非法牌编码: {card}"))
                else:
                    global_cards[card] += 1
            current_action_legal: bool | None = None
            if semantic_level is not None and not unknown:
                try:
                    candidates = actions_for_cards(tuple(cards), str(semantic_level))
                    current_action_legal = bool(candidates)
                except Exception as exc:
                    candidates = []
                    current_action_legal = False
                    legal_violations.append(
                        violation("P0", "V2", idx, action, f"牌型判定异常（级牌 {semantic_level}）: {type(exc).__name__}: {exc}")
                    )
                if not candidates:
                    legal_violations.append(
                        violation("P0", "V2", idx, action, f"不是规则引擎支持的合法牌型（级牌 {semantic_level}）")
                    )
            # A bad/unknown current table cannot support a reliable comparison
            # for the next play. Flag the dependent action for human review,
            # but do not manufacture a second "cannot beat" P0.
            if table_cards and table_semantically_valid is True and semantic_level is not None and not unknown and current_action_legal is not False:
                try:
                    beats = play_beats_table(tuple(cards), table_cards, str(semantic_level))
                except Exception as exc:
                    beats = False
                    beat_violations.append(
                        violation("P0", "V3", idx, action, f"压牌判定异常（级牌 {semantic_level}）: {type(exc).__name__}: {exc}")
                    )
                if not beats:
                    beat_violations.append(
                        violation(
                            "P0",
                            "V3",
                            idx,
                            action,
                            f"当前出牌不能压过桌面上一手 {' '.join(table_cards)}（级牌 {semantic_level}），不应记录为出牌",
                        )
                    )
            elif table_cards and table_semantically_valid is not True and not is_pass:
                manual_indeterminate.append(
                    violation("P2", "V3", idx, action, "前一手桌面牌面/牌型无法确定，当前动作的压牌关系需人工复核")
                )
            table_cards = tuple(cards)
            table_semantically_valid = current_action_legal
            current_leader = actor
            passed = set()

        _advance_finished(actor, cards, is_pass, remaining, finished)
        if round_is_decided(finished):
            current_leader = None
            table_cards = ()
            table_semantically_valid = None
            passed = set()
            continue

        if current_leader is None:
            # PASS without a table is handled by the existing V1/V9 checks;
            # keep the semantic state empty so subsequent actions are not
            # incorrectly compared against a phantom table.
            table_cards = ()
            table_semantically_valid = None
            passed = set()
            continue
        if is_pass:
            passed.add(actor)
        projection = project_trick_turn(current_leader, frozenset(finished), frozenset(passed))
        if projection.is_complete:
            current_leader = None
            table_cards = ()
            table_semantically_valid = None
            passed = set()

    for card, count in sorted(global_cards.items()):
        if count > 2:
            # Point at the last occurrence; report also includes total count.
            last_idx = next(
                (i for i in range(len(actions), 0, -1) if card in (actions[i - 1].get("cards") or ())),
                None,
            )
            last_action = actions[last_idx - 1] if last_idx else None
            physical_violations.append(
                violation("P0", "V2", last_idx, last_action, f"整局物理牌超出双副上限: {card} 共 {count} 张")
            )

    # A missing/ambiguous level prevents a definitive semantic PASS on a
    # complete clip. A unique level observed from video is still usable for
    # provisional engine checks, but must be manually confirmed.
    if not partial and context.get("confidence") != "authoritative_for_level":
        manual_context.append(
            {
                "code": "CTX_LEVEL",
                "severity": "P1" if context.get("ambiguous_observation") or not level else "P2",
                "detail": "级牌来自视频观测而非会话真值，需人工确认"
                if level and not context.get("ambiguous_observation")
                else "级牌观测存在冲突或缺失，无法直接完成确定性牌型/压牌验收",
            }
        )

    return {
        "level_context": context,
        "self_hand_size": self_hand,
        "legal_violations": legal_violations,
        "beat_violations": beat_violations,
        "physical_violations": physical_violations,
        "manual_context": manual_context,
        "manual_indeterminate": manual_indeterminate,
        "global_card_counts_over_two": {card: n for card, n in global_cards.items() if n > 2},
        "checked_nonpass_with_engine": sum(
            1 for a in actions if not a.get("is_pass") and "?" not in " ".join(map(str, a.get("cards") or ())) and semantic_level is not None
        ),
    }


def structure_precheck(batch: Path, summary: dict[str, Any]) -> dict[str, Any]:
    root = ROOT / "data" / "profiles" / "tencent_daguandan" / "sessions"
    descriptors = inspect_sessions(root)
    eligible = {
        item.session_id: item
        for item in descriptors
        if item.truth_status in {"draft", "missing", "invalid"} and item.has_video
    }
    rows = summary.get("sessions") or []
    ids = [str(row.get("session_id")) for row in rows]
    dirs = {d.name: d for d in batch.iterdir() if d.is_dir()}
    errors: list[str] = []
    per: dict[str, Any] = {}
    for sid in sorted(set(eligible) | set(ids) | set(dirs)):
        d = dirs.get(sid)
        row = next((r for r in rows if str(r.get("session_id")) == sid), None)
        errs: list[str] = []
        if sid in eligible and row is None:
            errs.append("应扫描但 summary 无条目")
        if sid in ids and sid not in eligible:
            errs.append("summary 含非应扫描会话")
        if sid in eligible and d is None:
            errs.append("缺少输出目录")
        if sid in dirs and sid not in eligible:
            errs.append("存在非应扫描输出目录")
        if d is not None and sid in eligible:
            for name in REQUIRED_FILES:
                p = d / name
                if not p.is_file() or p.stat().st_size <= 0:
                    errs.append(f"缺少或空产物: {name}")
            if not errs:
                try:
                    actions = read_jsonl(d / "action_trace.jsonl")
                    raw = read_jsonl(d / "raw_action_trace.jsonl")
                    observations = read_observations(d / "frame_observations.jsonl.gz")
                    sm = read_json(d / "scan_summary.json")
                    manifest = read_json(d / "scan_manifest.json")
                    slots = read_json(d / "turn_slots.json")
                    read_json(d / "opening_candidates.json")
                    if len(actions) != int(sm.get("action_count", -1)):
                        errs.append(f"action_count 摘要/实际不一致: {sm.get('action_count')} / {len(actions)}")
                    if len(raw) != int(sm.get("raw_action_count", -1)):
                        errs.append(f"raw_action_count 摘要/实际不一致: {sm.get('raw_action_count')} / {len(raw)}")
                    if len(observations) != int(sm.get("observation_count", -1)):
                        errs.append(f"observation_count 摘要/实际不一致: {sm.get('observation_count')} / {len(observations)}")
                    if int(sm.get("decoded_frames", -1)) != len(observations):
                        errs.append(f"decoded_frames/观测实际不一致: {sm.get('decoded_frames')} / {len(observations)}")
                    if sm.get("status") != "complete" or manifest.get("status") != "complete":
                        errs.append("扫描状态不是 complete")
                    if int(sm.get("failed_frames", -1)) != 0:
                        errs.append(f"failed_frames={sm.get('failed_frames')}")
                    counts = slots.get("counts") or {}
                    turn_slots = [s for s in slots.get("slots", []) if s.get("kind") == "turn"]
                    if len(turn_slots) != int(counts.get("turn_slots", -1)):
                        errs.append("turn_slots 数量不一致")
                except Exception as exc:
                    errs.append(f"产物解析失败: {type(exc).__name__}: {exc}")
        per[sid] = errs
        errors.extend(f"{sid}: {e}" for e in errs)
    return {
        "corpus_count": len(descriptors),
        "eligible_count": len(eligible),
        "eligible_ids": sorted(eligible),
        "verified_skipped_count": sum(1 for item in descriptors if item.truth_status == "verified" and item.has_video),
        "summary_ids": sorted(ids),
        "summary_unique_ids": len(set(ids)),
        "output_dir_count": len(dirs),
        "missing_from_summary": sorted(set(eligible) - set(ids)),
        "unexpected_in_summary": sorted(set(ids) - set(eligible)),
        "missing_output_dirs": sorted(set(eligible) - set(dirs)),
        "unexpected_output_dirs": sorted(set(dirs) - set(eligible)),
        "errors": errors,
        "per_session_errors": per,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("batch_dir", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    batch = args.batch_dir.resolve()
    summary = read_json(batch / "summary.json")
    structure = structure_precheck(batch, summary)
    results: list[dict[str, Any]] = []
    severity_counts: Counter[str] = Counter()
    hotspot_rows: list[dict[str, Any]] = []
    for d in sorted(x for x in batch.iterdir() if x.is_dir()):
        actions = read_jsonl(d / "action_trace.jsonl")
        observations = read_observations(d / "frame_observations.jsonl.gz")
        sm = read_json(d / "scan_summary.json")
        detailed = verify_session(d)
        rules = rule_checks(d.name, actions, observations, partial=int(sm.get("decoded_frames", 0)) < PARTIAL_FRAME_THRESHOLD)
        detail_violations = [
            v for vs in detailed.get("checks", {}).values() for v in vs
            if not (v.get("severity") == "P2" and "可豁免" in str(v.get("detail")))
        ]
        custom_violations = rules["legal_violations"] + rules["beat_violations"] + rules["physical_violations"]
        manual_flags = []
        for idx, action in enumerate(actions, 1):
            if action.get("review_status") == "needs_review" or action.get("repair_status") == "unresolved":
                manual_flags.append(
                    {
                        "action_idx": idx,
                        "label": action_label(action, idx),
                        "review_status": action.get("review_status"),
                        "repair_status": action.get("repair_status"),
                        "repair_reason": action.get("repair_reason"),
                    }
                )
        all_blocking = detail_violations + custom_violations
        for v in all_blocking:
            severity_counts[str(v.get("severity", "P?"))] += 1
        severity_counts.update(str(v["severity"]) for v in rules["manual_context"])
        fail = any(str(v.get("severity")) in {"P0", "P1"} for v in all_blocking)
        # Manual context is not silently accepted for complete clips. It is
        # surfaced separately as conditional/manual rather than converted to
        # an invented card error.
        if not fail and rules["manual_context"] and not detailed.get("partial"):
            verdict = "CONDITIONAL_MANUAL_REVIEW"
        else:
            verdict = "FAIL" if fail else "PASS"
        row = {
            "session_id": d.name,
            "partial": bool(detailed.get("partial")),
            "initial_context_present": bool(detailed.get("init_ctx")),
            "action_count": len(actions),
            "decoded_frames": int(sm.get("decoded_frames", 0)),
            "verdict": verdict,
            "detail_checks": detailed.get("checks", {}),
            "rule_checks": rules,
            "manual_flags": manual_flags,
        }
        results.append(row)
        if fail or manual_flags or rules["manual_context"] or rules["manual_indeterminate"]:
            hotspot_rows.append(
                {
                    "session_id": d.name,
                    "verdict": verdict,
                    "blocking": [
                        {
                            "severity": v.get("severity"),
                            "code": v.get("code"),
                            "action_idx": v.get("action_idx"),
                            "label": v.get("label"),
                            "detail": v.get("detail"),
                        }
                        for v in all_blocking
                    ],
                    "manual_context": rules["manual_context"],
                    "manual_indeterminate": rules["manual_indeterminate"],
                    "manual_flags": manual_flags,
                }
            )
    pass_count = sum(r["verdict"] == "PASS" for r in results)
    fail_count = sum(r["verdict"] == "FAIL" for r in results)
    conditional_count = sum(r["verdict"] == "CONDITIONAL_MANUAL_REVIEW" for r in results)
    report = {
        "schema": "guandan.unverified-batch-acceptance/2",
        "batch_directory": str(batch),
        "standard": "docs/unverified_batch_scan_acceptance_standard_v2.md",
        "scope_note": "自动化数据验收；不打开视频。hotspots 是人工视频复核清单。",
        "batch_summary": {
            key: summary.get(key)
            for key in ("selected_count", "completed_count", "failed_count", "cancelled", "max_workers")
        },
        "structure_precheck": structure,
        "session_counts": {
            "total_in_batch": len(results),
            "pass": pass_count,
            "fail": fail_count,
            "conditional_manual_review": conditional_count,
        },
        "severity_counts": dict(severity_counts),
        "results": results,
        "hotspots": hotspot_rows,
        "overall": {
            "generation_completeness": "PASS" if not structure["errors"] else "FAIL",
            "automatic_correctness": "PASS" if fail_count == 0 else "FAIL",
            "manual_review_required": bool(hotspot_rows),
            "final_acceptance": "PENDING_MANUAL_VIDEO_REVIEW" if fail_count == 0 and hotspot_rows else ("FAIL" if fail_count else "PASS"),
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({
        "batch": str(batch),
        "structure_errors": len(structure["errors"]),
        "sessions": len(results),
        "pass": pass_count,
        "fail": fail_count,
        "conditional_manual_review": conditional_count,
        "severity_counts": dict(severity_counts),
        "overall": report["overall"],
        "report": str(args.output.resolve()),
    }, ensure_ascii=False, indent=2))
    return 1 if report["overall"]["final_acceptance"] == "FAIL" else 0


if __name__ == "__main__":
    raise SystemExit(main())
