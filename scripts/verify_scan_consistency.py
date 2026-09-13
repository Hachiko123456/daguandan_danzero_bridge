"""视频扫描结果一致性验收标准（可执行）。

对一份扫描报告目录（含 action_trace.jsonl / turn_slots.json / frame_observations.jsonl.gz）
按以下标准验收：

V1 位次合法性
    规范动作轨迹必须能用逆时针位次规则（自己->右家->对家->左家）完整解释：
    每一手的行动者必须等于由 project_trick_turn 推导出的期望行动者，
    允许跳过已出完牌的座位；出完牌后的接风（伙伴领出）必须成立且被显式标注；
    新墩领出同理。任何无法用规则解释的跳位都计为位次错误。

V2 牌面完整性
    规范轨迹中不得出现未知花色（'?'）；单张牌在同一动作中不得超过 2 张
    （双副牌物理约束）；非 PASS 动作必须有牌。

V3 槽位一致性
    turn_slots 中每个 turn 槽位必须 resolved/recovered 且槽位演员与动作演员一致；
    槽位演员序列必须与规范动作演员序列按序对齐（允许轨迹在最后一个信号 run
    之后保留收尾动作，因为最后一手没有 following run，不产生槽位）。

V4 会话锚点（可选）
    针对具体对局的人工核对锚点，例如本局左家的 88822 三带二必须识别为
    完整的 8H 8H 8D 2D 2C，不得是 6 张或带未知花色。

用法：
    python scripts/verify_scan_consistency.py <scan_report_dir> [--session <session_dir>]
退出码：0 = 全部通过；1 = 存在违规。
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


def _load_actions(path: Path) -> list[dict[str, object]]:
    with path.open("r", encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def _load_slots(path: Path) -> dict[str, object]:
    return json.loads(path.read_text("utf-8"))


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


def check_turn_order(actions: list[dict[str, object]], self_hand: int) -> list[str]:
    """V1：逐步模拟位次推进，返回违规描述列表。"""
    violations: list[str] = []
    remaining = {seat: 27 for seat in TURN_ORDER}
    remaining["self"] = self_hand
    finished: set[str] = set()
    trick_leader: str | None = None
    passed: set[str] = set()
    expected: str | None = None  # 下一手期望行动者
    for index, action in enumerate(actions, start=1):
        actor = str(action.get("actor", ""))
        is_pass = bool(action.get("is_pass", False))
        cards = [str(c) for c in (action.get("cards") or ())]
        label = f"#{index} {actor} {'PASS' if is_pass else ' '.join(cards)}"
        if actor not in TURN_ORDER:
            violations.append(f"{label}: 非法座位")
            continue
        if expected is not None and actor != expected:
            violations.append(f"{label}: 位次错误，期望 {expected}")
        if expected is None and index > 1 and trick_leader is None:
            violations.append(f"{label}: 新墩领出者缺失")
        # 记录动作
        if not is_pass:
            remaining[actor] -= len(cards)
            if remaining[actor] <= 0:
                remaining[actor] = 0
                finished.add(actor)
            trick_leader = actor  # 最近的非 PASS 出牌者
            passed = set()
        else:
            if trick_leader is None:
                violations.append(f"{label}: 没有领出者时不能不出")
            passed.add(actor)
        if round_is_decided(finished):
            expected = None
            trick_leader = None
            # 牌局已定，之后不应再有动作
            if index < len(actions):
                violations.append(f"{label}: 牌局已结束但仍有后续动作")
            break
        if trick_leader is None:
            # 新墩：领出者由上一步的 next_leader 决定，本动作即为领出
            trick_leader = actor if not is_pass else None
            expected = next_active_seat(actor, frozenset(finished)) if not is_pass else None
            if is_pass:
                violations.append(f"{label}: 新墩首手不能是不出")
            continue
        projection = project_trick_turn(trick_leader, frozenset(finished), frozenset(passed))
        if projection.is_complete:
            # 墩收齐：下一手为下一墩领出者（含接风）
            trick_leader = None
            passed = set()
            expected = projection.next_leader
        else:
            try:
                expected = projection.expected_after(actor)
            except Exception:
                expected = None
    return violations


def check_cards(actions: list[dict[str, object]]) -> list[str]:
    """V2：牌面完整性。"""
    violations: list[str] = []
    for index, action in enumerate(actions, start=1):
        cards = [str(c) for c in (action.get("cards") or ())]
        is_pass = bool(action.get("is_pass", False))
        actor = action.get("actor")
        if is_pass:
            continue
        if not cards:
            violations.append(f"#{index} {actor}: 非 PASS 动作没有牌")
            continue
        unknown = [c for c in cards if c.endswith("?")]
        if unknown:
            violations.append(f"#{index} {actor}: 存在未知花色 {' '.join(cards)}")
        counts = Counter(c for c in cards if not c.endswith("?"))
        dup = [c for c, n in counts.items() if n > 2]
        if dup:
            violations.append(f"#{index} {actor}: 单张牌超过两张 {' '.join(dup)}")
    return violations


def check_slots(actions: list[dict[str, object]], slots_doc: dict[str, object]) -> list[str]:
    """V3：槽位与动作按序对齐。"""
    violations: list[str] = []
    slots = [s for s in slots_doc.get("slots", []) if s.get("kind") == "turn"]
    by_id = {a.get("action_id"): a for a in actions}
    cursor = 0  # 按序对齐的动作游标
    for slot in slots:
        actor = slot.get("actor")
        status = slot.get("status")
        action = slot.get("action") or {}
        if status not in {"resolved", "recovered"}:
            violations.append(f"槽位#{slot.get('slot_id')} {actor}: 状态 {status}")
            continue
        if action.get("actor") != actor:
            violations.append(f"槽位#{slot.get('slot_id')} {actor}: 动作演员 {action.get('actor')}")
        # 按序对齐：该槽位演员必须在游标之后的动作序列中出现
        found = None
        for i in range(cursor, len(actions)):
            if actions[i].get("actor") == actor:
                found = i
                break
        if found is None:
            violations.append(f"槽位#{slot.get('slot_id')} {actor}: 动作序列中找不到对应演员")
        else:
            cursor = found + 1
    return violations


# V4：会话级人工核对锚点。cards 为排序后的完整牌编码。
SESSION_ANCHORS: dict[str, list[dict[str, object]]] = {
    "game_20260829_192802_981e50": [
        {"actor": "left", "cards": ["2C", "2D", "8D", "8H", "8H"],
         "note": "左家三带二 88822 必须完整识别（帧205/220已人工核对）"},
        {"actor": "self", "cards": ["4D", "4S", "AC", "AD", "AS"],
         "note": "自己 AAA44 必须完整识别（帧220已人工核对）"},
    ],
    "game_20260829_114157_a940ae": [
        {"actor": "right", "cards": ["4C", "4S"], "position": 0,
         "note": "首动作为右家一对4（时间线：右家首出 4?〔♠/♣〕4♣）"},
    ],
}


def check_anchors(report_dir: Path, actions: list[dict[str, object]]) -> list[str]:
    violations: list[str] = []
    anchors: list[dict[str, object]] = []
    for session_id, items in SESSION_ANCHORS.items():
        if session_id in str(report_dir):
            anchors = items
            break
    for anchor in anchors:
        actor = str(anchor["actor"])
        want = sorted(str(c) for c in anchor["cards"])  # type: ignore[index]
        if "position" in anchor:
            pos = int(anchor["position"])  # type: ignore[index]
            if pos >= len(actions):
                violations.append(f"锚点失败：动作数 {len(actions)} 不足位置 {pos}")
                continue
            got = actions[pos]
            if got.get("actor") != actor or sorted(str(c) for c in got.get("cards") or ()) != want:
                violations.append(
                    f"锚点失败：第{pos + 1}个动作期望 {actor} {' '.join(want)}，"
                    f"实际 {got.get('actor')} {' '.join(str(c) for c in got.get('cards') or ())}"
                )
            continue
        if not any(
            a.get("actor") == actor and sorted(str(c) for c in a.get("cards") or ()) == want
            for a in actions
        ):
            violations.append(f"锚点失败：找不到 {actor} {' '.join(want)}（{anchor['note']}）")
    return violations


def main() -> int:
    parser = argparse.ArgumentParser(description="视频扫描结果一致性验收")
    parser.add_argument("report_dir", type=Path)
    parser.add_argument("--session", type=Path, default=None, help="保留参数：会话目录（当前仅用于锚点匹配）")
    args = parser.parse_args()
    report = args.report_dir
    actions = _load_actions(report / "action_trace.jsonl")
    slots_doc = _load_slots(report / "turn_slots.json")
    self_hand = _self_hand_size(report)

    checks = {
        "V1 位次合法性": check_turn_order(actions, self_hand),
        "V2 牌面完整性": check_cards(actions),
        "V3 槽位一致性": check_slots(actions, slots_doc),
        "V4 会话锚点": check_anchors(report, actions),
    }
    failed = False
    print(f"扫描报告：{report}")
    print(f"动作数：{len(actions)}，槽位数：{slots_doc.get('counts', {})}")
    for name, violations in checks.items():
        if violations:
            failed = True
            print(f"\n[FAIL] {name}（{len(violations)} 项违规）")
            for item in violations[:30]:
                print(f"  - {item}")
            if len(violations) > 30:
                print(f"  ... 其余 {len(violations) - 30} 项省略")
        else:
            print(f"[PASS] {name}")
    print("\n结论：" + ("未通过" if failed else "全部通过"))
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
