# 扫描出牌功能修复 — 验证报告

> 依据 `docs/verification_standard_scan_play.md` 执行。执行环境：`C:\project\python_project\daguandan_danzero_bridge`，
> 解释器 `C:\project\python_project\daguandan_danzero_bridge\.venv\Scripts\python.exe`（Python 3.12.0，pytest 9.1.1）。
> 执行时间：2026-09-11 13:51 ~ 14:30。

---

## 0. 总体结论

**未通过（FAIL）。** 步骤 B 通过，但**步骤 A 与步骤 C 均未达标**：回归测试新增 2 个失败；两局视频扫描的
V1–V4 验收**退出码均为 1**（要求为 0）。此外 V1 位次违规相比修复前**不降反升**，属新引入的回归。

| 步骤 | 内容 | 判定 |
|---|---|---|
| A | 语法编译 + 回归测试 | **FAIL**（编译通过；pytest 28 passed / **2 failed**，基线为 30 全过） |
| B | 两局视频扫描 | **PASS**（均 `status=complete`，三件产物齐全） |
| C | 两局 V1–V4 验收 | **FAIL**（两局均退出码 1；V1 恶化） |
| D | 实时管线回放（可选） | **不可判定**（不卡死，但未产出有效动作序列，0/2 条） |

修复目标（Bug 1 位次合法性）**未达成**，且在部分维度出现回归；Bug 2（遮挡识别）在 192802 局达成
（V4 锚点通过、V2 改善），但 114157 局锚点仍未通过。

---

## 1. 步骤 A：语法与回归测试

### 1.1 语法编译

```
python -m py_compile \
  src/daguandan_bridge/application/action_trace_projection.py \
  src/daguandan_bridge/application/action_trace_reconciliation.py \
  src/daguandan_bridge/application/turn_slot_projection.py \
  src/daguandan_bridge/live/orchestrator.py
```

结果：**4 个文件全部编译通过**（退出码 0，无输出）。

### 1.2 回归测试

```
python -m pytest tests/test_action_trace_reconciliation.py \
  tests/test_turn_slot_projection.py tests/test_video_scan.py \
  tests/test_scan_action_repair.py -q
```

结果：**2 failed, 28 passed in 2.21s**（基线为 30 全过，属"新增失败"）。

```
FAILED tests/test_action_trace_reconciliation.py::test_pass_after_signal_advance_is_recovered_in_same_turn_window
FAILED tests/test_action_trace_reconciliation.py::test_initial_table_context_keeps_only_passes_between_visible_leader_and_current_player
```

### 1.3 失败定性（关键）

按标准要求，需判断失败测试是否"编码了旧的错误行为"。经复现（脚本 `reports/_verify_dbg_recon.py`），
**两例失败均为同一根因，且不属于旧错误行为**，判定为**真回归**：

两个用例的失败动作都被 `action_trace_reconciliation.py` 内的新过滤器 `_out_of_turn_card_ghost`
（定义于第 339–369 行，调用点第 885 行）以 reason=`out_of_turn_card_ghost` 丢弃：

- **用例 1** `test_pass_after_signal_advance_is_recovered_in_same_turn_window`
  输入：右家出 5 张 `9S 6H JS QS KS`（帧 10–15），观测量窗口从帧 10 开始、`current_player_signal` 已为 `opposite`。
  实际输出仅剩 `[opposite PASS]`，**右家这手合法的 5 张出牌被整体丢弃**。
  期望为 `[right 出 5 张, opposite PASS]`。

- **用例 2** `test_initial_table_context_keeps_only_passes_between_visible_leader_and_current_player`
  输入：对家带 `display_present_at_scan_start` 的 5 张 `5C 6C 7C 8C 9C`（帧 0 已显示），首帧信号为 `self`。
  实际输出 `[left PASS, self PASS, right PASS]`，**开局已可见的领出者对家这手牌被丢弃**。

根因分析：

1. `_out_of_turn_card_ghost` 的判定是"若该动作的**所有**证据帧都不落在**演员自身**的 current-player run
   内（前后各留 -2/+4 帧松弛），且 `item.start` 时刻由**其他座位**持有计时器，则判为动画幽灵并丢弃"。
   当录像窗口从半途开始、或信号已推进到下一家时，**演员自身的 run 根本不在观测窗口内**，合法出牌被误判为幽灵。
2. 代码中已有 `_initial_context_required_pass_ids`（第 676–715 行）专门保护"开局已显示"场景下的 **PASS** 徽标，
   但**没有对等机制保护开局领出者的出牌动作**，导致该领出牌被后续的幽灵过滤误删。

<结论>：这两个测试断言的期望（保留合法出牌）是**符合领域规则的正确行为**，并非"张数优先"类的旧错误行为；
逃生条款不适用。故 **步骤 A 判定 FAIL**。同时也要指出：由于 C 步骤（端到端验收）同样未过，该失败并非孤立
的测试侧问题，而是与端到端位次回归相互印证。

---

## 2. 步骤 B：重跑两局视频扫描

两局均以后台任务方式运行（避免前台超时杀进程）。

| 会话 ID | 运行时长 | 结束状态 | 输出报告目录 |
|---|---|---|---|
| game_20260829_192802_981e50 | 10m28s | `status=complete` | `reports/video-scans/game_20260829_192802_981e50_20260911_135721` |
| game_20260829_114157_a940ae | 11m39s | `status=complete` | `reports/video-scans/game_20260829_114157_a940ae_20260911_135724` |

产物齐全性核对（两份均含三件必需产物）：

```
action_trace.jsonl            (224 KB / 216 KB)
turn_slots.json               ( 41 KB /  51 KB)
frame_observations.jsonl.gz   (156 KB / 179 KB)
```

**步骤 B 判定：PASS。**

---

## 3. 步骤 C：V1–V4 验收检查

命令：

```
python scripts/verify_scan_consistency.py reports/video-scans/<最新报告目录>
```

### 3.1 汇总对比（修复前基线 vs 修复后）

| 会话 | 标准 | 修复前基线 | 修复后 | 判定 |
|---|---|---|---|---|
| 192802 | V1 位次合法性 | 11 项违规 | **13 项违规** | **FAIL（恶化）** |
| 192802 | V2 牌面完整性 | 2 项违规 | **1 项违规** | **FAIL（仍违规）** |
| 192802 | V3 槽位一致性 | 15 项违规 | **0 项** | PASS（改善） |
| 192802 | V4 会话锚点 | 1 项违规 | **0 项** | PASS（改善） |
| 114157 | V1 位次合法性 | 15 项违规 | **36 项违规** | **FAIL（恶化）** |
| 114157 | V2 牌面完整性 | 0 项 | 0 项 | PASS |
| 114157 | V3 槽位一致性 | 41 项违规 | **21 项违规** | **FAIL（仍违规）** |
| 114157 | V4 会话锚点 | 1 项违规 | **1 项违规** | **FAIL（未改善）** |

- 192802：动作数 80 → 93；槽位 `signal_runs 84 / turn_slots 83 / resolved 83 / recovered 0 / needs_review 1`。
- 114157：动作数 85 → 101；槽位 `signal_runs 100 / turn_slots 99 / resolved 92 / recovered 7 / needs_review 4`。
- 两局脚本**退出码均为 1**（要求 0）。修复前基线 192802 的 V1/V2/V3/V4 = 11/2/15/1 被**完全复现**，
  说明验收脚本与执行方式与标准一致、可比。

### 3.2 违规明细

**192802 局** — V1（13 项，节选）：

```
#12 right PASS: 位次错误，期望 opposite
#17 right 8D 8C 8C 8S: 位次错误，期望 opposite
#46 right QH QD QS 4H 4C: 位次错误，期望 opposite
#52 self PASS: 位次错误，期望 left
#53 left PASS: 位次错误，期望 right
#57 self PASS: 位次错误，期望 left
#58 left PASS: 位次错误，期望 right
#62 self PASS: 位次错误，期望 left
#63 left PASS: 位次错误，期望 right
#67 self PASS: 位次错误，期望 left
#68 opposite PASS: 位次错误，期望 right
#70 opposite PASS: 位次错误，期望 self
#71 self PASS: 位次错误，期望 left
```

V2（1 项）：`#87 self: 存在未知花色 QD Q?`。V3、V4 全过（V4 锚点：left `8H 8H 8D 2D 2C`、
self `AD AC AS 4D 4S` 均命中）。

**114157 局** — V1（36 项，节选）：

```
#14 left PASS: 位次错误，期望 opposite
#15 opposite PASS: 位次错误，期望 self
#16 left PASS: 位次错误，期望 self / 没有领出者时不能不出 / 新墩首手不能是不出
#17 self 10H 10D 10S 4S 4S: 新墩领出者缺失
#19 left PASS: 位次错误，期望 opposite
#27 opposite PASS: 位次错误，期望 right / 新墩首手不能是不出
#62 right PASS: 位次错误，期望 self
#63 right PASS: 位次错误，期望 opposite
#78 right PASS: 位次错误，期望 self
#79 right PASS: 位次错误，期望 opposite
...
```

V3（21 项）：槽位 `#70 ~ #99` 均"动作序列中找不到对应演员"（按序对齐游标耗尽）。
V4（1 项）：`锚点失败：第 1 个动作期望 right 4C 4S，实际 right 4C` —— 右家首手一对 4 仍被识别为单张 4C。

### 3.3 回归根因分析

对比修复前后动作序列后发现，V1 恶化的直接来源是**成对重复出现的 PASS 伪动作**，由本修复引入的
"跨墩 PASS 拆分"逻辑（`_split_pass_actions_at_turn_boundaries`，调用点 `action_trace_reconciliation.py:774–775`）产生：

- 192802：`#11 right PASS + #12 right PASS`、`#52 self PASS + #54 self PASS`、`#57/#59 self PASS`、
  `#62/#64 self PASS`、`#67/#71 self PASS`（同一座位相邻两条 PASS）。
- 114157：`#62 right PASS + #63 right PASS`、`#78 right PASS + #79 right PASS` 等。

同一 PASS 徽标被切分成相邻两条同座位 PASS，破坏严格逆时针位次，并使个别 PASS 落在"新墩首手"位置，
派生"新墩首手不能是不出 / 新墩领出者缺失"。此外，114157 局 V1 中还残留多处 `left/opposite PASS` 连续出现，
说明跨墩 PASS 拆分在该局的切分点仍不正确。

> 代码注释（`action_trace_reconciliation.py:770–773`）声称"有信号证据时按回合边界拆分 PASS 总是安全的"，
> 但实测两局均出现同座位重复 PASS，该假设不成立。

同时，`_out_of_turn_card_ghost`（见 1.3）在真实扫描中同样可能误删合法出牌，与 V1 违规相互叠加。

---

## 4. 步骤 D：实时管线回放（可选）

**执行说明**：标准第 4 节给出的 D 示例无法直接执行——`replay_video_through_live_pipeline` 的实参第 2 位是必需的
`recognition_service`，且第 1 位应为**会话目录**而非视频文件（标准已注明"参数以该文件实际签名为准"）。
本次按 `live/replay.py:707` 实际签名、并参照 GUI 回放页（`gui/replay_page.py:325`）的调用方式补全后运行，
脚本 `reports/_verify_replay.py`，日志 `reports/_scan_verify_replay.log`。

**运行结果**（`use_live_pipeline=True`）：

| 会话 | 处理帧数 | 结束状态 | 产出动作数 | V1/V2 |
|---|---|---|---|---|
| game_20260829_192802_981e50 | 755 | `status=complete`（sealed） | **0** | 0（样本为空，无意义） |
| game_20260829_114157_a940ae | 887 | `status=complete`（sealed） | **2**（right 4C、opposite 9H） | 0（样本为空） |

- **不卡死**：两局均处理完全部帧并 sealed，满足"回放正常跑完不卡死"。
- **但未产出有效动作序列**：动作链近乎为空（0 / 2 条）。`runtime/timeline.jsonl` 中仅见
  `initial_state_confirmed / lead_player_confirmed / turn_started / advice_withheld / advice_recovery_failed /
  terminal_history_gap / game_end_detected / session_finalizing`，**没有任何 `player_played` 事件**
  （192802 局；114157 局仅 2 条）。因此 V1/V2 无法实质评估。

**对照实验（用于定位）**：对 192802 局以 `use_live_pipeline=False`（log 驱动路径）复跑，
结果同样 `processed_turn_count=0`。**即两种模式都近乎不产出动作**，说明该现象**并非实时管线特有**，
不能归因于本次"遮挡门控"改动；更可能是该回放路径在本次环境/参数下的共性问题（例如回放侧对
current-player 信号/播放位置门控的依赖）。114157 局回放产出的第 1 手 `right 4C` 与步骤 C 的 V4 锚点失败
（期望一对 `4C 4S`）方向一致，可佐证视频本身该处确为"一对 4 被识别为单张"。

**步骤 D 判定：不可判定 / 未通过实质验收（该步骤为可选）。**
理由：虽满足"不卡死"，但未产出可校验的动作序列，且标准未提供 D 的修复前基线；建议改由 GUI 回放页
（具备完整线程上下文与 `wait_for_position` 门控）复测，并补充修复前后的 D 基线后再行判定。
此结果**不影响** A、C 已给出的 FAIL 结论。

---

## 5. 复现指引

```
# A. 编译 + 回归
.venv/Scripts/python.exe -m py_compile src/daguandan_bridge/application/{action_trace_projection,action_trace_reconciliation,turn_slot_projection}.py src/daguandan_bridge/live/orchestrator.py
.venv/Scripts/python.exe -m pytest tests/test_action_trace_reconciliation.py tests/test_turn_slot_projection.py tests/test_video_scan.py tests/test_scan_action_repair.py -q

# B. 扫描（后台）
.venv/Scripts/python.exe scripts/run_video_scan.py data/profiles/tencent_daguandan/sessions/<会话ID>

# C. 验收
.venv/Scripts/python.exe scripts/verify_scan_consistency.py reports/video-scans/<最新报告目录>
```

本次验证新增的临时脚本（位于 `reports/`，非交付物）：
`_verify_dbg_recon.py`（用例复现）、`_verify_compare.py`（前后对比）、`_verify_replay.py`（步骤 D）。

---

## 6. 结论与建议

**最终判定：FAIL——修复尚未达到验收标准。**

1. **必须修复的回归**：跨墩 PASS 拆分产生同座位相邻重复 PASS，直接恶化 V1（两局均变差）。
   建议在拆分后增加"同座位相邻 PASS 去重/合并"约束，或收紧拆分触发条件。
2. **必须修复的回归**：`_out_of_turn_card_ghost` 误删演员自身信号 run 不在窗口内的合法出牌；
   需为"开局已显示领出者出牌"与"信号已推进"两种情形增加豁免（参照 `_initial_context_required_pass_ids` 的思路）。
3. **114157 局 V4 锚点未过**：右家首手一对 4 仍识别为单张，遮挡/配对识别在该局未生效。
4. **已达成**：V3 槽位一致性显著改善（15→0、41→21），192802 局 V4 锚点通过，V2 在 192802 由 2→1。

修复后需重跑步骤 A、B、C，要求 pytest 30 全过、两局 V1–V4 退出码均为 0，方可判定通过。
