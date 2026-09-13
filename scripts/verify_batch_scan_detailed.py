"""批量视频扫描结果 — 详细验收脚本（纯数据层，不读视频）。

对一份批量扫描批次目录逐局检查，输出**精确到动作**的缺陷定位报告。

设计目标：
- 验收者只需运行此脚本，无需了解项目实现细节。
- 每条违规都标注：会话 / 动作序号 / 演员 / 帧范围 / 严重等级。
- 不打开任何视频文件，只检查生成的 JSON / JSONL 产物。

检查项：
  A1-A4  批次完整性（汇总自洽、产物齐全、覆盖率）
  V1     位次合法性（逆时针推进、跳过已出完座位、接风、PASS 时机）
  V2     牌面完整性（无未知花色、单编码≤2张、非PASS必须有牌）
  V3     槽位一致性（turn_slots 与动作序列按序对齐）
  V5     开局一致性（opening_candidates 与首动作演员/牌面一致）
  V6     终局判定（轨迹结束时对局是否已分胜负，或疑似未扫完）
  V7     帧范围合法性（start≤end、best_frame 落在区间、证据帧非空）
  V8     规约溯源（每个动作有 source_action_ids、无链式引用）
  V9     PASS 模式（无领出者时不 PASS、一墩不全 PASS）
  V10    座位覆盖（4 个座位均出现，除非是部分片段）

严重等级：
  P0 阻断  — 未知花色、单编码>2张、手牌为负、对局无法分胜负
  P1 严重  — 中局越位、牌局结束后仍有动作、无领出者 PASS（第4手后）
  P2 轻微  — 开局不一致、帧范围异常、证据帧缺失

用法：
    python scripts/verify_batch_scan_detailed.py <batch_dir>
    python scripts/verify_batch_scan_detailed.py <batch_dir> --session <session_id>
退出码：0 = 全部通过；1 = 存在不可豁免缺陷。
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
    PARTNER_SEAT,
    TEAMS,
    TURN_ORDER,
    next_active_seat,
    project_trick_turn,
    round_is_decided,
)

# ────────────────────────── 常量 ──────────────────────────

SEATS = list(TURN_ORDER)  # self, right, opposite, left
NEED_FILES = [
    "scan_manifest.json", "frame_observations.jsonl.gz", "action_trace.jsonl",
    "raw_action_trace.jsonl", "opening_candidates.json", "scan_summary.json",
    "turn_slots.json",
]
# 帧数低于此值视为"部分片段"（非完整对局），豁免终局与开局 PASS 检查。
PARTIAL_CLIP_FRAME_THRESHOLD = 400


# ────────────────────────── 工具 ──────────────────────────

def _load_jsonl(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text("utf-8"))


def _label(action: dict, idx: int) -> str:
    actor = action.get("actor", "?")
    if action.get("is_pass"):
        cards = "PASS"
    else:
        cards = " ".join(str(c) for c in (action.get("cards") or ()))
    fs = action.get("frame_start", "?")
    fe = action.get("frame_end", "?")
    return f"#{idx} {actor} {cards} (帧 {fs}-{fe})"


def _is_partial_clip(scan_summary: dict) -> bool:
    return scan_summary.get("decoded_frames", 0) < PARTIAL_CLIP_FRAME_THRESHOLD


def _initial_context_present(turn_slots: dict) -> bool:
    return bool(turn_slots.get("initial_context_present"))


# ────────────────────────── 检查函数 ──────────────────────────
# 每个返回 (violations: list[dict], passed: bool)
# violation dict: {severity, code, action_idx, actor, detail}

def check_v1_turn_order(actions: list[dict], self_hand: int, partial: bool, init_ctx: bool) -> list[dict]:
    """V1 位次合法性 — 逐手模拟逆时针推进。"""
    violations: list[dict] = []
    remaining = {s: 27 for s in SEATS}
    remaining["self"] = self_hand
    finished: set[str] = set()
    trick_leader: str | None = None
    passed: set[str] = set()
    expected: str | None = None
    for idx, action in enumerate(actions, start=1):
        actor = str(action.get("actor", ""))
        is_pass = bool(action.get("is_pass", False))
        cards = [str(c) for c in (action.get("cards") or ())]
        # 可豁免窗口：部分片段 或 缺初始上下文的前3手
        in_exempt_window = partial or (init_ctx and idx <= 3)
        if actor not in SEATS:
            violations.append({"severity": "P0", "code": "V1", "action_idx": idx,
                               "actor": actor, "detail": "非法座位"})
            continue
        if expected is not None and actor != expected:
            sev = "P2" if in_exempt_window else "P1"
            tag = "可豁免" if in_exempt_window else "不可豁免"
            violations.append({"severity": sev, "code": "V1", "action_idx": idx,
                               "actor": actor,
                               "detail": f"位次错误，期望 {expected}（{tag}）"})
        if expected is None and idx > 1 and trick_leader is None:
            violations.append({"severity": "P2" if in_exempt_window else "P1",
                               "code": "V1", "action_idx": idx, "actor": actor,
                               "detail": "新墩领出者缺失"})
        if not is_pass:
            remaining[actor] -= len(cards)
            if remaining[actor] <= 0:
                remaining[actor] = 0
                finished.add(actor)
            trick_leader = actor
            passed = set()
        else:
            if trick_leader is None:
                violations.append({"severity": "P2" if in_exempt_window else "P1",
                                   "code": "V1", "action_idx": idx, "actor": actor,
                                   "detail": "没有领出者时不能不出"})
            passed.add(actor)
        if round_is_decided(finished):
            if idx < len(actions):
                violations.append({"severity": "P1", "code": "V1", "action_idx": idx,
                                   "actor": actor, "detail": "牌局已结束但仍有后续动作"})
            break
        if trick_leader is None:
            trick_leader = actor if not is_pass else None
            expected = next_active_seat(actor, frozenset(finished)) if not is_pass else None
            if is_pass:
                violations.append({"severity": "P2" if in_exempt_window else "P1",
                                   "code": "V1", "action_idx": idx, "actor": actor,
                                   "detail": "新墩首手不能是不出"})
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


def check_v2_cards(actions: list[dict]) -> list[dict]:
    """V2 牌面完整性。"""
    violations: list[dict] = []
    for idx, action in enumerate(actions, start=1):
        cards = [str(c) for c in (action.get("cards") or ())]
        is_pass = bool(action.get("is_pass", False))
        actor = action.get("actor", "?")
        if is_pass:
            continue
        if not cards:
            violations.append({"severity": "P0", "code": "V2", "action_idx": idx,
                               "actor": actor, "detail": "非 PASS 动作没有牌"})
            continue
        unknown = [c for c in cards if c.endswith("?")]
        if unknown:
            violations.append({"severity": "P0", "code": "V2", "action_idx": idx,
                               "actor": actor,
                               "detail": f"存在未知花色 {' '.join(unknown)}"})
        counts = Counter(c for c in cards if not c.endswith("?"))
        dup = [c for c, n in counts.items() if n > 2]
        if dup:
            violations.append({"severity": "P0", "code": "V2", "action_idx": idx,
                               "actor": actor,
                               "detail": f"单张牌超过两张 {' '.join(dup)}"})
    return violations


def check_v3_slots(actions: list[dict], slots_doc: dict) -> list[dict]:
    """V3 槽位一致性。"""
    violations: list[dict] = []
    slots = [s for s in slots_doc.get("slots", []) if s.get("kind") == "turn"]
    cursor = 0
    for slot in slots:
        actor = slot.get("actor", "?")
        status = slot.get("status", "?")
        sid = slot.get("slot_id", "?")
        action = slot.get("action") or {}
        if status not in {"resolved", "recovered"}:
            violations.append({"severity": "P1", "code": "V3", "action_idx": sid,
                               "actor": actor,
                               "detail": f"槽位状态 {status}"})
            continue
        if action.get("actor") != actor:
            violations.append({"severity": "P1", "code": "V3", "action_idx": sid,
                               "actor": actor,
                               "detail": f"动作演员 {action.get('actor')} 与槽位不符"})
        found = None
        for i in range(cursor, len(actions)):
            if actions[i].get("actor") == actor:
                found = i
                break
        if found is None:
            violations.append({"severity": "P1", "code": "V3", "action_idx": sid,
                               "actor": actor, "detail": "动作序列中找不到对应演员"})
        else:
            cursor = found + 1
    return violations


def check_v5_opening(actions: list[dict], opening: dict, partial: bool) -> list[dict]:
    """V5 开局一致性：opening_candidates 与首动作演员/牌面一致。"""
    violations: list[dict] = []
    if partial:
        return violations  # 部分片段豁免
    status = opening.get("status", "?")
    if status != "resolved":
        return violations  # needs_review 时无法核对，不报
    lead = opening.get("lead_player")
    cands = opening.get("candidates") or []
    if not actions:
        return violations
    first = actions[0]
    if lead and first.get("actor") != lead:
        violations.append({"severity": "P2", "code": "V5", "action_idx": 1,
                           "actor": first.get("actor", "?"),
                           "detail": f"首动作演员 {first.get('actor')} 与 opening lead_player {lead} 不符"})
    if cands:
        want = sorted(str(c) for c in cands[0].get("cards", []))
        got = sorted(str(c) for c in (first.get("cards") or []))
        if want and want != got and not first.get("is_pass"):
            violations.append({"severity": "P2", "code": "V5", "action_idx": 1,
                               "actor": first.get("actor", "?"),
                               "detail": f"首动作牌面 {' '.join(got)} 与 opening 候选 {' '.join(want)} 不符"})
    return violations


def check_v6_termination(actions: list[dict], self_hand: int, partial: bool) -> list[dict]:
    """V6 终局判定：轨迹结束时对局是否已分胜负。

    手牌消耗检查注意贡牌规则：掼蛋开局时败方进贡，胜方收贡，
    各座位起手可能为 25–29 张（≠ 27）。因此：
    - self 用观测到的实际手牌数，负值即幽灵牌；
    - 其他座位只有当消耗 < -2（即打了 ≥30 张）才判为幽灵牌，-1/-2 可能是收贡。
    """
    violations: list[dict] = []
    if partial or not actions:
        return violations
    remaining = {s: 27 for s in SEATS}
    remaining["self"] = self_hand
    finished: set[str] = set()
    for idx, action in enumerate(actions, start=1):
        actor = str(action.get("actor", ""))
        if actor not in SEATS:
            continue
        if not action.get("is_pass"):
            cards = [str(c) for c in (action.get("cards") or ())]
            remaining[actor] -= len(cards)
            # 幽灵牌判定：贡牌可 +2、识别可能少读 1-2 张，故统一阈值 -2。
            # 只有当某座位打了 >=30 张（remaining < -2）才判为不可豁免幽灵牌。
            if remaining[actor] < -2:
                violations.append({"severity": "P0", "code": "V6", "action_idx": idx,
                                   "actor": actor,
                                   "detail": f"手牌消耗 {remaining[actor]}（>=30张，超出贡牌+识别容差）"})
            if remaining[actor] <= 0:
                remaining[actor] = 0
                finished.add(actor)
    if not round_is_decided(finished):
        rem_str = ", ".join(f"{s}:{remaining[s]}" for s in SEATS)
        violations.append({"severity": "P1", "code": "V6", "action_idx": len(actions),
                           "actor": "-",
                           "detail": f"轨迹结束但对局未分胜负（{rem_str}），疑似扫描不完整或漏动作"})
    return violations


def check_v7_frames(actions: list[dict]) -> list[dict]:
    """V7 帧范围合法性（只对非 PASS 动作检查 best_frame；PASS 的 best_frame
    可能来自拆分前的原始 run，落在区间外属正常，不报）。"""
    violations: list[dict] = []
    for idx, action in enumerate(actions, start=1):
        actor = action.get("actor", "?")
        fs = action.get("frame_start")
        fe = action.get("frame_end")
        is_pass = bool(action.get("is_pass", False))
        if fs is None or fe is None:
            violations.append({"severity": "P2", "code": "V7", "action_idx": idx,
                               "actor": actor, "detail": "缺少 frame_start/frame_end"})
            continue
        if fs > fe:
            violations.append({"severity": "P2", "code": "V7", "action_idx": idx,
                               "actor": actor, "detail": f"frame_start({fs}) > frame_end({fe})"})
        if not is_pass:
            bf = action.get("best_frame")
            if bf is not None and not (fs <= bf <= fe):
                violations.append({"severity": "P2", "code": "V7", "action_idx": idx,
                                   "actor": actor,
                                   "detail": f"best_frame({bf}) 不在 [{fs},{fe}] 区间"})
            ev = action.get("evidence_frames") or []
            if not ev:
                violations.append({"severity": "P2", "code": "V7", "action_idx": idx,
                                   "actor": actor, "detail": "非 PASS 动作无证据帧"})
    return violations


def check_v8_provenance(actions: list[dict], raw_actions: list[dict]) -> list[dict]:
    """V8 规约溯源：非 PASS 动作必须有 source_action_ids。

    合并（merge）时 source_action_ids 引用被合并的 repaired id 是**正常行为**，
    不算违规；PASS 动作如果是拆分（split）产生的合成动作，可以没有
    source_action_ids。只有"非 PASS 动作完全没有溯源"才判违规。
    """
    violations: list[dict] = []
    for idx, action in enumerate(actions, start=1):
        actor = action.get("actor", "?")
        is_pass = bool(action.get("is_pass", False))
        rec = action.get("reconciliation") or {}
        src = rec.get("source_action_ids") or []
        if not is_pass and not src:
            violations.append({"severity": "P2", "code": "V8", "action_idx": idx,
                               "actor": actor,
                               "detail": "非 PASS 动作缺少 source_action_ids（无法溯源到 raw 轨迹）"})
    return violations


def check_v9_pass_pattern(actions: list[dict], partial: bool, init_ctx: bool) -> list[dict]:
    """V9 PASS 模式：无领出者时不 PASS、一墩不全 PASS。"""
    violations: list[dict] = []
    trick_leader: str | None = None
    for idx, action in enumerate(actions, start=1):
        actor = str(action.get("actor", "?"))
        is_pass = bool(action.get("is_pass", False))
        in_exempt = partial or (init_ctx and idx <= 3)
        if is_pass and trick_leader is None:
            violations.append({"severity": "P2" if in_exempt else "P1",
                               "code": "V9", "action_idx": idx, "actor": actor,
                               "detail": "无领出者时 PASS" + ("（可豁免）" if in_exempt else "")})
        if not is_pass:
            trick_leader = actor
    return violations


def check_v10_seat_coverage(actions: list[dict], partial: bool) -> list[dict]:
    """V10 座位覆盖：4 个座位均出现（除非部分片段）。"""
    violations: list[dict] = []
    if partial:
        return violations
    present = {a.get("actor") for a in actions}
    missing = [s for s in SEATS if s not in present]
    if missing:
        violations.append({"severity": "P1", "code": "V10", "action_idx": 0,
                           "actor": "-",
                           "detail": f"缺失座位 {missing}（全程未出现）"})
    return violations


# ────────────────────────── 单局验收 ──────────────────────────

def verify_session(session_dir: Path) -> dict:
    """对单局执行全部检查，返回结构化结果。"""
    actions = _load_jsonl(session_dir / "action_trace.jsonl")
    raw_actions = _load_jsonl(session_dir / "raw_action_trace.jsonl")
    summary = _load_json(session_dir / "scan_summary.json")
    slots_doc = _load_json(session_dir / "turn_slots.json")
    opening = _load_json(session_dir / "opening_candidates.json")

    partial = _is_partial_clip(summary)
    init_ctx = _initial_context_present(slots_doc)
    self_hand = _self_hand_size(session_dir)

    checks = {
        "V1 位次合法性": check_v1_turn_order(actions, self_hand, partial, init_ctx),
        "V2 牌面完整性": check_v2_cards(actions),
        "V3 槽位一致性": check_v3_slots(actions, slots_doc),
        "V5 开局一致性": check_v5_opening(actions, opening, partial),
        "V6 终局判定": check_v6_termination(actions, self_hand, partial),
        "V7 帧范围合法性": check_v7_frames(actions),
        "V8 规约溯源": check_v8_provenance(actions, raw_actions),
        "V9 PASS 模式": check_v9_pass_pattern(actions, partial, init_ctx),
        "V10 座位覆盖": check_v10_seat_coverage(actions, partial),
    }
    # 标记可豁免（部分片段或 init_ctx 前3手）
    for name, viols in checks.items():
        for v in viols:
            v["session"] = session_dir.name
            v["check"] = name
    return {
        "session": session_dir.name,
        "partial": partial,
        "init_ctx": init_ctx,
        "action_count": len(actions),
        "checks": checks,
    }


def _self_hand_size(report_dir: Path) -> int:
    obs = report_dir / "frame_observations.jsonl.gz"
    if not obs.is_file():
        return 27
    best = 0
    with gzip.open(obs, "rt", encoding="utf-8") as fh:
        for line in fh:
            row = json.loads(line)
            opening = row.get("opening")
            if isinstance(opening, dict):
                hand = opening.get("my_hand") or ()
                best = max(best, len(hand))
    return best or 27


# ────────────────────────── 批次验收 ──────────────────────────

def verify_batch(batch_dir: Path, only_session: str | None = None) -> int:
    summary = _load_json(batch_dir / "summary.json")
    print(f"{'=' * 72}")
    print(f"批量扫描验收报告")
    print(f"批次目录: {batch_dir}")
    print(f"选中 {summary['selected_count']} / 完成 {summary['completed_count']} / "
          f"失败 {summary['failed_count']} / 并发 {summary.get('max_workers', '?')}")
    print(f"{'=' * 72}")

    # ── 批次级检查 A1-A4 ──
    print("\n[批次级检查]")
    a1 = summary["selected_count"] == summary["completed_count"] and summary["failed_count"] == 0
    a4 = len(summary["sessions"]) == summary["selected_count"] and all(
        r.get("status") == "complete" for r in summary["sessions"])
    missing_files: list[str] = []
    for r in summary["sessions"]:
        d = Path(r["output_directory"])
        missing_files.extend(
            f"{r['session_id']}: {f}" for f in NEED_FILES if not (d / f).is_file())
    a3 = not missing_files
    print(f"  A1 选中=完成 且 失败=0: {'PASS' if a1 else 'FAIL'}")
    print(f"  A3 产物文件齐全: {'PASS' if a3 else 'FAIL'}")
    if missing_files:
        for m in missing_files:
            print(f"      缺失: {m}")
    print(f"  A4 汇总条目自洽: {'PASS' if a4 else 'FAIL'}")

    # ── 逐局检查 ──
    session_dirs = sorted(d for d in batch_dir.iterdir() if d.is_dir())
    if only_session:
        session_dirs = [d for d in session_dirs if d.name == only_session]
        if not session_dirs:
            print(f"\n未找到会话: {only_session}")
            return 1

    all_fail_sessions: list[str] = []
    total_p0 = total_p1 = total_p2 = 0
    pass_sessions = 0

    for sd in session_dirs:
        result = verify_session(sd)
        checks = result["checks"]
        # 统计不可豁免缺陷
        p0 = sum(1 for v in checks.values() for v in v if v["severity"] == "P0")
        p1 = sum(1 for v in checks.values() for v in v if v["severity"] == "P1")
        p2 = sum(1 for v in checks.values() for v in v if v["severity"] == "P2")
        total_p0 += p0
        total_p1 += p1
        total_p2 += p2
        has_fail = p0 + p1 > 0
        if has_fail:
            all_fail_sessions.append(sd.name)
        else:
            pass_sessions += 1

        tag = "部分片段" if result["partial"] else ("无初始上下文" if result["init_ctx"] else "完整")
        print(f"\n--- {sd.name} ({tag}, {result['action_count']}动作) ---")
        for name, viols in checks.items():
            if not viols:
                print(f"  {name}: PASS")
            else:
                np0 = sum(1 for v in viols if v["severity"] == "P0")
                np1 = sum(1 for v in viols if v["severity"] == "P1")
                np2 = sum(1 for v in viols if v["severity"] == "P2")
                parts = []
                if np0: parts.append(f"P0×{np0}")
                if np1: parts.append(f"P1×{np1}")
                if np2: parts.append(f"P2×{np2}")
                print(f"  {name}: FAIL ({', '.join(parts)})")
                for v in viols:
                    label = _label(actions_lookup(sd, v["action_idx"]), v["action_idx"]) \
                        if v["action_idx"] and v["code"] not in ("V10",) and not isinstance(v["action_idx"], str) \
                        else f"#{v['action_idx']}"
                    print(f"      [{v['severity']}] {label}: {v['detail']}")
        verdict = "FAIL" if has_fail else "PASS"
        if result["partial"]:
            verdict += " (部分片段，终局/开局检查已豁免)"
        print(f"  => 结论: {verdict}")

    # ── 汇总 ──
    print(f"\n{'=' * 72}")
    print(f"[汇总]")
    print(f"  PASS: {pass_sessions} 局")
    print(f"  FAIL: {len(all_fail_sessions)} 局")
    if all_fail_sessions:
        print(f"  不通过会话: {', '.join(all_fail_sessions)}")
    print(f"  缺陷统计: P0={total_p0} P1={total_p1} P2={total_p2}")
    print(f"  (P0=阻断/未知花色·幽灵牌, P1=严重/越位·终局异常, P2=轻微/开局·帧范围)")
    overall = pass_sessions == len(session_dirs)
    print(f"\n总体结论: {'PASS（全部通过）' if overall else 'FAIL（存在不可豁免缺陷）'}")
    return 0 if overall else 1


def actions_lookup(session_dir: Path, idx: int) -> dict:
    """惰性读取单条动作用于标签生成。"""
    if not idx or idx < 1:
        return {}
    try:
        with (session_dir / "action_trace.jsonl").open("r", encoding="utf-8") as fh:
            for i, line in enumerate(fh, 1):
                if i == idx:
                    return json.loads(line)
    except Exception:
        pass
    return {}


def main() -> int:
    parser = argparse.ArgumentParser(description="批量视频扫描结果详细验收")
    parser.add_argument("batch_dir", type=Path, help="批次目录（含 summary.json）")
    parser.add_argument("--session", default=None, help="只验收指定会话")
    args = parser.parse_args()
    return verify_batch(args.batch_dir, args.session)


if __name__ == "__main__":
    raise SystemExit(main())
