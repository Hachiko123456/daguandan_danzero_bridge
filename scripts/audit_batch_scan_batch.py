"""批量扫描结果 — 详细问题定位报告（可执行验收工具）。

对一批「扫描未验证对局」的输出目录做逐局详细体检，输出：
  * 每局的结构完整性、自报告问题、位次/牌面违规明细
  * 每个问题精确定位到：会话 ID、动作序号、帧区间、涉及座位、原因
  * 汇总统计与「可疑局清单」

设计目标：验收者不需要看视频、不需要读源码，只靠本脚本的输出即可定位问题。

用法：
    python scripts/audit_batch_scan_batch.py <batch_dir> [--json <out.json>] [--quiet]
退出码：0 = 全部通过；1 = 存在不可豁免问题。
"""

from __future__ import annotations

import argparse
import gzip
import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from daguandan_bridge.live.turns import (  # noqa: E402
    TURN_ORDER,
    next_active_seat,
    project_trick_turn,
    round_is_decided,
)

# 少于该帧数视为「非完整对局片段」，其开局违规豁免
MIN_FULL_GAME_FRAMES = 400
# 开局若干手在缺少初始上下文时可豁免
OPENING_EXEMPT_HANDS = 3

REQUIRED_FILES = (
    "scan_manifest.json",
    "frame_observations.jsonl.gz",
    "action_trace.jsonl",
    "raw_action_trace.jsonl",
    "opening_candidates.json",
    "scan_summary.json",
    "turn_slots.json",
)


def _read_jsonl(path: Path) -> list[dict]:
    if not path.is_file():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _read_json(path: Path) -> dict:
    if not path.is_file():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _count_gz_lines(path: Path) -> int:
    if not path.is_file():
        return 0
    n = 0
    with gzip.open(path, "rt", encoding="utf-8") as fh:
        for _ in fh:
            n += 1
    return n


def _self_hand_size(report_dir: Path) -> int:
    obs = report_dir / "frame_observations.jsonl.gz"
    if not obs.is_file():
        return 27
    best = 0
    with gzip.open(obs, "rt", encoding="utf-8") as fh:
        for line in fh:
            opening = json.loads(line).get("opening")
            if isinstance(opening, dict):
                best = max(best, len(opening.get("my_hand") or ()))
    return best or 27


def check_structure(report: Path) -> list[str]:
    """S1：产物完整性与跨文件一致性。"""
    problems: list[str] = []
    missing = [name for name in REQUIRED_FILES if not (report / name).is_file()]
    if missing:
        problems.append(f"S1 缺产物文件: {', '.join(missing)}")
        return problems

    actions = _read_jsonl(report / "action_trace.jsonl")
    raw = _read_jsonl(report / "raw_action_trace.jsonl")
    summary = _read_json(report / "scan_summary.json")
    slots = _read_json(report / "turn_slots.json")
    obs_lines = _count_gz_lines(report / "frame_observations.jsonl.gz")

    if len(actions) != summary.get("action_count"):
        problems.append(
            f"S1 动作数不符: action_trace={len(actions)} vs scan_summary.action_count={summary.get('action_count')}"
        )
    if len(raw) < len(actions):
        problems.append(f"S1 原始轨迹少于规约轨迹: raw={len(raw)} < final={len(actions)}")
    if obs_lines != summary.get("observation_count"):
        problems.append(
            f"S1 观测数不符: frame_observations={obs_lines} vs scan_summary.observation_count={summary.get('observation_count')}"
        )
    counts = slots.get("counts") or {}
    turns = [s for s in slots.get("slots", []) if s.get("kind") == "turn"]
    if len(turns) != counts.get("turn_slots"):
        problems.append(f"S1 槽位数不符: turn槽位={len(turns)} vs counts.turn_slots={counts.get('turn_slots')}")
    if len(slots.get("signal_runs", [])) != counts.get("signal_runs"):
        problems.append(
            f"S1 信号段数不符: signal_runs={len(slots.get('signal_runs', []))} vs counts.signal_runs={counts.get('signal_runs')}"
        )
    unresolved = sum(
        s.get("status") == "needs_review" and s.get("kind") in {"turn", "missing_turn"}
        for s in slots.get("slots", [])
    )
    if unresolved != counts.get("needs_review"):
        problems.append(f"S1 needs_review 计数不符: 实算={unresolved} vs counts={counts.get('needs_review')}")
    for idx, action in enumerate(actions, start=1):
        if action.get("frame_start", 0) > action.get("frame_end", 0):
            problems.append(f"S1 #{idx} {action.get('actor')}: frame_start > frame_end")
    for idx in range(len(actions) - 1):
        if actions[idx].get("action_id") >= actions[idx + 1].get("action_id"):
            problems.append(f"S1 #{idx + 1}: action_id 非严格递增")
            break
    return problems


def check_self_reported(report: Path) -> list[dict]:
    """S2：解析数据自身的 needs_review / missing_turn 自报告问题（最精确的定位来源）。"""
    slots = _read_json(report / "turn_slots.json")
    actions = _read_jsonl(report / "action_trace.jsonl")
    by_id = {a.get("action_id"): a for a in actions}
    findings: list[dict] = []
    for slot in slots.get("slots", []):
        if slot.get("status") != "needs_review" or slot.get("kind") not in {"turn", "missing_turn"}:
            continue
        if slot.get("kind") == "missing_turn":
            findings.append({
                "kind": "missing_turn",
                "slot_id": slot.get("slot_id"),
                "actor": slot.get("actor"),
                "reason": slot.get("reason"),
                "start_frame": slot.get("start_frame"),
                "close_frame": slot.get("close_frame"),
                "remaining_cards_before": slot.get("remaining_cards_before"),
                "action_id": None,
                "cards": None,
            })
        else:
            action = by_id.get(slot.get("action_id")) or slot.get("action") or {}
            findings.append({
                "kind": "turn",
                "slot_id": slot.get("slot_id"),
                "actor": slot.get("actor"),
                "reason": slot.get("reason"),
                "start_frame": slot.get("start_frame"),
                "close_frame": slot.get("close_frame"),
                "remaining_cards_before": slot.get("remaining_cards_before"),
                "action_id": slot.get("action_id"),
                "cards": action.get("cards"),
            })
    return findings


def check_turn_order(actions: list[dict], self_hand: int) -> list[dict]:
    """S3：位次合法性，返回带定位信息的违规列表。"""
    violations: list[dict] = []
    remaining = {seat: 27 for seat in TURN_ORDER}
    remaining["self"] = self_hand
    finished: set[str] = set()
    trick_leader: str | None = None
    passed: set[str] = set()
    expected: str | None = None
    for index, action in enumerate(actions, start=1):
        actor = str(action.get("actor", ""))
        is_pass = bool(action.get("is_pass", False))
        cards = [str(c) for c in (action.get("cards") or ())]
        label = f"#{index} {actor} {'PASS' if is_pass else ' '.join(cards)}"
        base = {
            "index": index,
            "action_id": action.get("action_id"),
            "actor": actor,
            "cards": cards,
            "is_pass": is_pass,
            "frame_start": action.get("frame_start"),
            "frame_end": action.get("frame_end"),
            "best_frame": action.get("best_frame"),
            "label": label,
        }
        if actor not in TURN_ORDER:
            violations.append({**base, "rule": "illegal_seat", "detail": f"{label}: 非法座位"})
            continue
        if expected is not None and actor != expected:
            violations.append({
                **base, "rule": "out_of_turn",
                "detail": f"{label}: 位次错误，期望 {expected}",
                "expected": expected,
            })
        if expected is None and index > 1 and trick_leader is None:
            violations.append({**base, "rule": "missing_leader", "detail": f"{label}: 新墩领出者缺失"})
        if not is_pass:
            remaining[actor] -= len(cards)
            if remaining[actor] <= 0:
                remaining[actor] = 0
                finished.add(actor)
            trick_leader = actor
            passed = set()
        else:
            if trick_leader is None:
                violations.append({**base, "rule": "pass_without_leader", "detail": f"{label}: 没有领出者时不能不出"})
            passed.add(actor)
        if round_is_decided(finished):
            if index < len(actions):
                violations.append({**base, "rule": "action_after_end", "detail": f"{label}: 牌局已结束但仍有后续动作"})
            break
        if trick_leader is None:
            trick_leader = actor if not is_pass else None
            expected = next_active_seat(actor, frozenset(finished)) if not is_pass else None
            if is_pass:
                violations.append({**base, "rule": "pass_as_new_leader", "detail": f"{label}: 新墩首手不能是不出"})
            continue
        projection = project_trick_turn(trick_leader, frozenset(finished), frozenset(passed))
        if projection.is_complete:
            trick_leader = None
            passed = set()
            expected = projection.next_leader
        else:
            try:
                expected = projection.expected_after(actor)
            except Exception:
                expected = None
    return violations


def check_cards(actions: list[dict]) -> list[dict]:
    """S4：牌面完整性，返回带定位信息的违规列表。"""
    violations: list[dict] = []
    for index, action in enumerate(actions, start=1):
        cards = [str(c) for c in (action.get("cards") or ())]
        if bool(action.get("is_pass", False)):
            continue
        base = {
            "index": index,
            "action_id": action.get("action_id"),
            "actor": action.get("actor"),
            "cards": cards,
            "frame_start": action.get("frame_start"),
            "frame_end": action.get("frame_end"),
            "best_frame": action.get("best_frame"),
        }
        if not cards:
            violations.append({**base, "rule": "empty_cards", "detail": f"#{index} {action.get('actor')}: 非 PASS 动作没有牌"})
            continue
        unknown = [c for c in cards if c.endswith("?")]
        if unknown:
            violations.append({
                **base, "rule": "unknown_suit",
                "detail": f"#{index} {action.get('actor')}: 存在未知花色 {' '.join(cards)}",
                "offending": unknown,
            })
        counts = Counter(c for c in cards if not c.endswith("?"))
        dup = [c for c, n in counts.items() if n > 2]
        if dup:
            violations.append({
                **base, "rule": "over_two_copies",
                "detail": f"#{index} {action.get('actor')}: 单张牌超过两张 {' '.join(dup)}",
                "offending": dup,
            })
    return violations


def check_slots_alignment(actions: list[dict], slots_doc: dict) -> list[dict]:
    """S5：槽位与动作按序对齐。"""
    violations: list[dict] = []
    slots = [s for s in slots_doc.get("slots", []) if s.get("kind") == "turn"]
    cursor = 0
    for slot in slots:
        actor = slot.get("actor")
        status = slot.get("status")
        action = slot.get("action") or {}
        if status not in {"resolved", "recovered"}:
            violations.append({
                "rule": "bad_slot_status", "slot_id": slot.get("slot_id"), "actor": actor,
                "detail": f"槽位#{slot.get('slot_id')} {actor}: 状态 {status}",
            })
            continue
        if action.get("actor") != actor:
            violations.append({
                "rule": "slot_actor_mismatch", "slot_id": slot.get("slot_id"), "actor": actor,
                "detail": f"槽位#{slot.get('slot_id')} {actor}: 动作演员 {action.get('actor')}",
            })
        found = None
        for i in range(cursor, len(actions)):
            if actions[i].get("actor") == actor:
                found = i
                break
        if found is None:
            violations.append({
                "rule": "slot_action_not_found", "slot_id": slot.get("slot_id"), "actor": actor,
                "detail": f"槽位#{slot.get('slot_id')} {actor}: 动作序列中找不到对应演员",
            })
        else:
            cursor = found + 1
    return violations


def check_reconciliation(report: Path, actions: list[dict]) -> dict:
    """S6：统计规约诊断事件（用于判断问题性质，非违规本身）。"""
    event_types: Counter = Counter()
    drop_types: Counter = Counter()
    repair_reasons: Counter = Counter()
    unknown_repair_fail: list[dict] = []
    for action in actions:
        recon = action.get("reconciliation") or {}
        for event in recon.get("events") or []:
            event_types[f"{event.get('type')}/{event.get('reason')}"] += 1
        for event in recon.get("drop_events") or []:
            drop_types[f"{event.get('type')}/{event.get('reason')}"] += 1
        reason = action.get("repair_reason")
        if reason:
            repair_reasons[str(reason)] += 1
        if action.get("repair_status") == "needs_review":
            unknown_repair_fail.append({
                "action_id": action.get("action_id"),
                "actor": action.get("actor"),
                "cards": action.get("cards"),
                "repair_reason": reason,
                "frame_start": action.get("frame_start"),
            })
    return {
        "events": dict(event_types),
        "drop_events": dict(drop_types),
        "repair_reasons": dict(repair_reasons),
        "unresolved_repairs": unknown_repair_fail,
    }


def audit_session(report: Path) -> dict:
    actions = _read_jsonl(report / "action_trace.jsonl")
    slots_doc = _read_json(report / "turn_slots.json")
    summary = _read_json(report / "scan_summary.json")
    opening = _read_json(report / "opening_candidates.json")
    self_hand = _self_hand_size(report)

    structure = check_structure(report)
    self_reported = check_self_reported(report)
    turn = check_turn_order(actions, self_hand)
    cards = check_cards(actions)
    alignment = check_slots_alignment(actions, slots_doc)
    recon = check_reconciliation(report, actions)

    frames = summary.get("decoded_frames", 0)
    is_partial = frames < MIN_FULL_GAME_FRAMES
    has_initial_context = bool(slots_doc.get("initial_context_present"))
    # 豁免判定：片段局、或缺少初始上下文时的开局若干手
    def exempt(item: dict) -> bool:
        if is_partial:
            return True
        if has_initial_context and item.get("rule") in {
            "pass_without_leader", "pass_as_new_leader", "missing_leader", "out_of_turn"
        } and (item.get("index") or 99) <= OPENING_EXEMPT_HANDS:
            return True
        return False

    for item in turn:
        item["exempt"] = exempt(item)
    blocking = [i for i in turn if not i["exempt"]]

    return {
        "session_id": report.name,
        "report_dir": str(report),
        "frames": frames,
        "is_partial": is_partial,
        "has_initial_context": has_initial_context,
        "action_count": len(actions),
        "raw_action_count": len(_read_jsonl(report / "raw_action_trace.jsonl")),
        "opening_status": opening.get("status"),
        "opening_lead": opening.get("lead_player"),
        "self_hand": self_hand,
        "slot_counts": slots_doc.get("counts") or {},
        "structure_problems": structure,
        "self_reported": self_reported,
        "turn_violations": turn,
        "card_violations": cards,
        "alignment_violations": alignment,
        "reconciliation": recon,
        "blocking_count": len(blocking) + len(cards) + len(alignment) + len(structure),
        "verdict": "FAIL" if (blocking or cards or alignment or structure) else "PASS",
    }


def _fmt_loc(item: dict) -> str:
    act = f"#{item.get('index')}" if item.get("index") else f"动作{item.get('action_id')}"
    fr = f"帧{item.get('frame_start')}-{item.get('frame_end')}" if item.get("frame_start") is not None else ""
    return f"{act} {item.get('actor', '?')} {fr}".strip()


def main() -> int:
    parser = argparse.ArgumentParser(description="批量扫描结果详细问题定位")
    parser.add_argument("batch_dir", type=Path)
    parser.add_argument("--json", type=Path, default=None, help="把完整结构化结果写入该文件")
    parser.add_argument("--quiet", action="store_true", help="只输出汇总，不逐局展开")
    args = parser.parse_args()

    batch = args.batch_dir
    if not batch.is_dir():
        print(f"批次目录不存在: {batch}")
        return 2
    dirs = sorted(d for d in batch.iterdir() if d.is_dir())
    if not dirs:
        print(f"批次目录下没有会话子目录: {batch}")
        return 2

    results = [audit_session(d) for d in dirs]
    failed = [r for r in results if r["verdict"] == "FAIL"]

    print("=" * 100)
    print(f"批次目录: {batch}")
    print(f"会话数: {len(results)}   通过: {len(results) - len(failed)}   不通过: {len(failed)}")
    print("=" * 100)

    if not args.quiet:
        for r in results:
            print()
            print("-" * 100)
            tag = "PASS" if r["verdict"] == "PASS" else "FAIL"
            partial = " [片段局]" if r["is_partial"] else ""
            ctx = " [缺初始上下文]" if r["has_initial_context"] else ""
            print(f"[{tag}] {r['session_id']}{partial}{ctx}")
            print(f"      帧数={r['frames']} 动作={r['action_count']}(原始{r['raw_action_count']}) "
                  f"槽位={r['slot_counts']} 自己手牌={r['self_hand']} 开局={r['opening_status']}/{r['opening_lead']}")
            if r["structure_problems"]:
                print(f"  ▸ 结构问题 ({len(r['structure_problems'])}):")
                for p in r["structure_problems"]:
                    print(f"      - {p}")
            if r["self_reported"]:
                print(f"  ▸ 数据自报告问题 ({len(r['self_reported'])}):")
                for s in r["self_reported"]:
                    loc = f"帧{s.get('start_frame')}-{s.get('close_frame')}"
                    extra = f" 剩余牌={s.get('remaining_cards_before')}" if s.get("remaining_cards_before") is not None else ""
                    print(f"      - [{s['kind']}] 座位={s['actor']} 原因={s['reason']} {loc}{extra} slot={s['slot_id']}")
            if r["turn_violations"]:
                blk = [v for v in r["turn_violations"] if not v.get("exempt")]
                exh = [v for v in r["turn_violations"] if v.get("exempt")]
                print(f"  ▸ 位次违规 ({len(r['turn_violations'])}，其中不可豁免 {len(blk)}):")
                for v in r["turn_violations"]:
                    mark = "（豁免）" if v.get("exempt") else ""
                    print(f"      - {_fmt_loc(v)} {v['detail']}{mark}")
            if r["card_violations"]:
                print(f"  ▸ 牌面违规 ({len(r['card_violations'])}):")
                for v in r["card_violations"]:
                    print(f"      - {_fmt_loc(v)} {v['detail']}  [规则={v['rule']}]")
            if r["alignment_violations"]:
                print(f"  ▸ 槽位对齐违规 ({len(r['alignment_violations'])}):")
                for v in r["alignment_violations"]:
                    print(f"      - {v['detail']}")

    print()
    print("=" * 100)
    print("汇总")
    print("=" * 100)
    print(f"通过局数: {len(results) - len(failed)} / {len(results)}")
    if failed:
        print(f"\n需要修的局（{len(failed)}）:")
        for r in failed:
            reasons = []
            if r["structure_problems"]:
                reasons.append(f"结构{len(r['structure_problems'])}")
            if r["card_violations"]:
                reasons.append(f"牌面{len(r['card_violations'])}")
            if r["alignment_violations"]:
                reasons.append(f"对齐{len(r['alignment_violations'])}")
            nblk = len([v for v in r["turn_violations"] if not v.get("exempt")])
            if nblk:
                reasons.append(f"位次{nblk}")
            print(f"  - {r['session_id']}: {', '.join(reasons)}")
    total_self = sum(len(r["self_reported"]) for r in results)
    print(f"\n数据自报告需复核项合计: {total_self}")

    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"结构化结果已写入: {args.json}")

    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
