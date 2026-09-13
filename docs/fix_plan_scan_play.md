# 扫描出牌功能修复 — 修改方案（不含代码改动）

> 依据 `docs/verification_report_scan_play.md` 的验证结论。本文件只给方案，**尚未改动任何代码**。
> 目标：使 `docs/verification_standard_scan_play.md` 的步骤 A（pytest 30 全过）与步骤 C（两局 V1–V4 退出码 0）达标。

---

## 0. 待修项总览

| 编号 | 问题 | 现象证据 | 涉及文件 | 优先级 |
|---|---|---|---|---|
| **P2** | 跨墩 PASS 拆分产生**同一 run 的重复克隆** | 192802 V1 11→13；114157 V1 15→36 | `action_trace_reconciliation.py:239-290` | **最高** |
| **P1** | 幽灵过滤器误删**合法出牌** | 步骤 A 2 个单测新增失败 | `action_trace_reconciliation.py:339-369`（调用 `:885`） | 高 |
| **P3a** | 未知花色 `?` 未在最终轨迹中消解 | 192802 V2 残留 1 项 `#87 self QD Q?` | `action_trace_reconciliation.py`（收尾阶段） | 中 |
| **P3b** | 114157 首手一对 4 被识别为单张 | V4：`期望 right 4C 4S，实际 right 4C` | **识别层**（`recognition_service`），非本次 4 文件 | 中（需先定界） |

结论依赖关系：**P2 决定 V1；P1 决定步骤 A；P3a 决定 192802 的 V2；P3b 决定 114157 的 V4。** 四项全部清零才能整体 PASS。

---

## 1. P2（最高优先）：跨墩 PASS 拆分重复克隆

### 1.1 根因（已用报告数据证实）

位置：`_split_pass_actions_at_turn_boundaries`（`:239-290`），在 `reconcile_action_trace` 中于 `:774-775` 调用。

该函数对每个 PASS 动作，找出与其窗口重叠的**同座位 signal run**，为每个 run 生成一个克隆。问题在于**多个原始 PASS 面可以同时重叠同一个 run**：

192802 报告 `action_trace.jsonl` 的溯源字段直接暴露了这一点：

```
#11 right PASS 157..158   source_action_ids=['7@turn-159']    ← 原始 PASS 面 raw#7 (114..158)
#12 right PASS 187..188   source_action_ids=['11@turn-159']   ← 原始 PASS 面 raw#11(187..214)
#13 opposite PASS 191..192 source=['12@turn-187']
#52 self PASS 514..515    source=['49@turn-516']
#54 self PASS 522..523    source=['53@turn-516']     ← 与 #52 同为 turn-516
#57 self PASS 532..533    source=['53@turn-534']
#59 self PASS 539..540    source=['57@turn-534']     ← 与 #57 同为 turn-534
#62 self PASS 560..561    source=['57@turn-562']
#64 self PASS 566..567    source=['61@turn-562']     ← 与 #62 同为 turn-562
```

**同一 `turn-XXX`（即同一 signal run）被生成 2 个克隆 → 同一回合出现 2 条同座位 PASS**，破坏严格逆时针位次，直接造成 V1 违规；并派生"新墩首手不能是不出 / 新墩领出者缺失"。

机理：run 匹配条件过松（`:263-264`）

```python
run.frame_end >= item.start - 2 and run.frame_start <= item.end + 2
```

当一个 run 很长（示例中跨约 159..186）时，其**前**一个 PASS 面（结束于 158）与**后**一个 PASS 面（开始于 187）都满足该条件 → 两次克隆都锚到同一 run。

### 1.2 方案

**方案 A（推荐，最小且彻底）：按 `(actor, run.frame_start)` 去重**

- 在生成克隆前按 `(演员, run.frame_start)` 分组；每组只保留**证据帧与 run 区间 `[run.frame_start, run.frame_end]` 重叠度最高**的一个候选（并列时取 evidence 更靠内的）。
- 被淘汰的候选写入审计事件：`{"type":"drop","reason":"duplicate_pass_run_clone","action_ids":[...]}`，保持"可审计删除"的一贯风格。

**方案 B（辅助，收紧匹配）**

- 克隆要求其证据窗口**主体落在 run 内**，例如要求 `clone.start >= run.frame_start - 2`（不允许"前一个面"跳到后一个 run）。

**方案 C（复用现有判据）**

- 仅当现有 `_has_full_response_cycle(...)`（`:293-319`）或 `_actor_received_new_turn_between(...)`（`:322-333`）判定"该座位确实跨过了一个完整回合周期"时才允许拆分——这两个函数已在 `:920/:930/:933` 用于合并判定，可避免新引入语义。

> 建议 **A + B**：A 保证不出现同 run 重复，B 从源头收紧误匹配。C 可选加固。

### 1.3 验收

- 192802：V1 由 **13 → 0**（并预期 V3 维持 0）。
- 114157：V1 由 **36 → 0 或个位数**；其 V3 的 21 项"动作序列中找不到对应演员"预计随重复动作消除而改善（重复动作会耗尽 V3 的按序对齐游标）。
- 不得恶化 V2/V4。

---

## 2. P1：幽灵过滤器误删合法出牌

### 2.1 根因

位置：`_out_of_turn_card_ghost`（`:339-369`），调用点 `:885`。

判据为"该动作**所有**证据帧都不落在**演员自身**的 run 内（-2/+4 松弛），且 `item.start` 时刻由**他人**持有计时器 → 判幽灵并删除"。当**演员自身 run 根本不在观测窗口内**时（录像中途开始、信号已推进到下一家、开局领出者），合法出牌被误删：

- 单测 `test_pass_after_signal_advance_is_recovered_in_same_turn_window`：右家出 5 张（帧 10–15），但窗口内**没有 right 的 run**（信号已是 opposite）→ 被删。
- 单测 `test_initial_table_context_...`：对家带 `display_present_at_scan_start` 的领出牌 → 被删。

补位缺失：`_initial_context_required_pass_ids`（`:676-715`）只保护开局 **PASS** 徽标，**没有对等的领出牌保护**。

### 2.2 方案（建议 1+2+3 组合）

1. **时长守卫（最有效）**：真实幽灵是动画闪烁，通常仅 1–3 帧。新增常量 `_GHOST_MAX_EVIDENCE_FRAMES = 3`，仅当 `len(set(item.evidence_frames)) <= _GHOST_MAX_EVIDENCE_FRAMES` 时才启用幽灵判定。
   → 直接救回两例（right 6 帧、opposite 10 帧），且保留对短闪烁幽灵的过滤能力。
2. **证据不足守卫**：若 `signal_runs` 中**完全没有该演员的 run**，说明信号证据不足 → 返回 `False`（不删）。修正用例 1。
3. **开局上下文守卫**：`"display_present_at_scan_start" in item.original.get("uncertainty", ())` 的动作豁免。修正用例 2。
4. （可选加固）**领出牌守卫**：若该动作是其所在墩的 leader，不得判幽灵。

### 2.3 风险与对冲

- 放宽后可能**重新引入"邻座动画幽灵"**（这正是该过滤器要解决的 Bug 1 根因之一），导致 V1 回升。
- 对冲：**P2 与 P1 必须一起改、一起跑**；先单独落地 P2 确认 V1 改善，再叠加 P1，用"两局 V1/V2 回归 + 现有 30 个单测"双重把关；若 P1 导致 V1 回升，则改为只保留守卫 1+3（最小放宽）。

---

## 3. P3a：未知花色未消解（192802 V2 残留 1 项）

**现象**：`#87 self QD Q?`。标准明确"`?` 不允许出现在最终结果中"。

**设计冲突（需先决策）**：模块当前把未知花色保留为 `needs_review`（单测 `test_lone_one_frame_unknown_suit_is_preserved_for_review_without_complete_support` 要求 `A?` 被保留），但 V2 禁止 `?`。**两种取向互斥**，需二选一：

- **取向一（建议，满足现有标准）**：新增"收尾消费 pass"，在 canonical trace 输出前消解 `?`：
  1. 同动作内存在同点数已知牌（`QD` + `Q?`）且双副牌允许该编码出现 2 张 → 解析为同花色（`QD QD`）；
  2. 否则结合该座位剩余手牌/已出牌计数做唯一化；
  3. 仍无法唯一确定 → 从 canonical trace 移出并写审计（`reason="unresolved_unknown_suit"`，保持可追溯）。
- **取向二**：保留 `needs_review`，同步修改 V2 标准（允许 needs_review 动作含 `?`）——但这会改变验收口径，需你确认。

**风险**：强行解析可能把真实的 `QD QH` 误判为 `QD QD`。必须叠加"双副牌同编码上限 2"与"该座位手牌一致性"双重约束；无法唯一化时倾向"移出+审计"而非猜测。

---

## 4. P3b：114157 首手一对 4 识别为单张（V4 未过）

**现象**：锚点要求 `right 4C 4S`；离线扫描 `raw_action_trace.jsonl` 的 `raw id=1`（right，帧 128–160，**全部 35 帧一致**）只读到单张 `4C`（conf 0.90），`opening_candidates.json` 亦只有 `["4C"]`。

**对比证据**：同源录像的实时真值记录 `EVT-000003 right ['4?','4C']`，`source=opening_handoff_two_valid_anchor`，conf 0.61 —— **实时采集当时检出了两张**（其中一张花色未知）。

**定界结论**：差异发生在**识别层（牌框切分 / 花色判定）**，**不在本次修改的 4 个文件内**。reconciliation 只在已有读数上做归并，无法凭空补出第二张牌。因此：

- 不建议在 reconciliation 里硬编码"补一张同点数牌"。
- 建议排查方向：
  1. `ScreenshotRecognitionService` 在右家 ROI 该帧为何只输出 1 个牌框（第二张是否被裁剪/遮挡，或逐卡置信度低于阈值被过滤）；
  2. 复用实时侧的 `opening_handoff` 能力：开局领出时若同点数出现两张候选，允许保留 `4?` 参与 P3a 的消解；
  3. 若确认视频信息本身不足，则需与标准方确认该锚点期望是否应放宽为"首手含 4 张数≥1"。

---

## 5. 执行顺序与回归防线

**顺序（每步都跑步骤 A + B + C）**

1. **只修 P2** → 跑两局，确认 V1 明显下降（预期 114157 36 → 个位数）。
2. **叠加 P1** → 跑两局 + 30 单测，确认 V1 继续下降且不回升、2 个失败单测转绿。
3. **修 P3a** → 目标 192802 V2 = 0（需先确认第 3 节的取向）。
4. **排查 P3b** → 目标 114157 V4 = 0（识别层，先定界再改）。
5. **全量复跑** → 要求 pytest 30 全过 + 两局 V1–V4 退出码 0。

**回归防线**

- 把本次 2 个失败单测作为**守卫测试**保留（期望即"保留合法出牌"），确保 P1 不回归。
- **为 P2 新增单测**：构造"一个长 signal run 被前后两个 PASS 面同时覆盖"的输入，断言只产出 1 条 PASS（这是本次 V1 恶化的最小复现）。
- 建议给 `reconcile_action_trace` 增加一条输出不变量自检：canonical trace 中不得存在两条 `(actor, 所属 run)` 相同的动作；不得出现相邻同座位 PASS。

---

## 6. 改动面评估

| 文件 | 是否需要改 | 内容 |
|---|---|---|
| `application/action_trace_reconciliation.py` | **是** | P2 拆分去重/收紧；P1 三条守卫；P3a 收尾消解 |
| `application/action_trace_projection.py` | 可能 | 若 P3a 的消解放在投影阶段；否则不改 |
| `application/turn_slot_projection.py` | 预期不改 | V3 问题由 P2 的重复动作引发，修 P2 后应自愈；若仍不达标再评估 |
| `live/orchestrator.py` | 预期不改 | 遮挡门控在 192802 已见效（V4 通过）；P3b 若需复用 `opening_handoff` 才涉及 |
| `recognition_service`（识别层） | 待定 | P3b 根因所在，需先定界 |
| `tests/test_action_trace_reconciliation.py` | 是 | 保留 2 个守卫测试；新增 P2 重复克隆测试 |

> 说明：P1 与 P2 集中在 `action_trace_reconciliation.py` 两个函数内，改动面小、可独立回归，建议按第 5 节顺序分批落地。
