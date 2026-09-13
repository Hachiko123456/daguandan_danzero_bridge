# 扫描出牌功能修复 — 实施与复验报告（第二轮）

> 承接 `docs/verification_report_scan_play.md`（第一轮：判定 FAIL）与 `docs/fix_plan_scan_play.md`（修改方案）。
> 本轮已**完成代码修改并重跑 `docs/verification_standard_scan_play.md` 全流程**。
> 执行环境：`C:\project\python_project\daguandan_danzero_bridge`，解释器 `.venv\Scripts\python.exe`（Python 3.12.0 / pytest 9.1.1）。
> 时间：2026-09-11 14:00 ~ 14:50。

---

## 0. 结论摘要

| 步骤 | 结果 | 判定 |
|---|---|---|
| A 编译 + 回归 | py_compile 通过；**pytest 30 passed**（修复前为 28/30，2 项失败） | **PASS** |
| B 两局扫描 | 均 `status=complete`，三件产物齐全 | **PASS** |
| C 两局 V1–V4 | 192802：**V1–V4 全部 PASS，退出码 0**；114157：V1/V2/V3 全 PASS，**V4 1 项失败** | **192802 PASS / 114157 差 V4** |
| D 实时回放（可选） | 见第 5 节 | 参考 |

**位次与牌面问题已彻底修复**：两局 V1（位次合法性）、V2（牌面完整性）、V3（槽位一致性）**均为 0 违规**。
唯一遗留是 114157 的 V4 锚点，经逐帧取证证明**该锚点期望在物理上不可能成立**（见第 4 节），
离线扫描的实际输出反而是正确的。

---

## 1. 本次修改内容（全部位于 `src/daguandan_bridge/application/action_trace_reconciliation.py`）

| # | 修改 | 对应问题 | 说明 |
|---|---|---|---|
| 1 | `_out_of_turn_card_ghost` 增加三条守卫（新增常量 `_GHOST_MAX_EVIDENCE_FRAMES = 3`） | P1 | ① 证据帧数 > 3 的持久牌面不判幽灵；② 该座位在整段信号中**没有任何 run** 时不判幽灵（录像中途开始/信号已推进）；③ 带 `display_present_at_scan_start` 的领出牌豁免。修复了两个单测失败。 |
| 2 | `_split_pass_actions_at_turn_boundaries` 按 `(座位, run)` 去重 | P2 | 同一长 run 会同时匹配前后两个 PASS 面，原实现各克隆一次 → 同一回合出现 2 条同座位 PASS。现每组只保留与 run 区间重叠度最高者，落选者写审计 `duplicate_pass_run_clone`。 |
| 3 | 新增 `_anchor_actions_to_turn_spine` + `turn_spine_mode` 门控 | V1 根治 | 当 current-player 信号流足够稠密（`len(runs) >= 8 且 items <= 2*runs`）时，把规范动作**归位到回合主干**上：每个 run 至多一条动作、按 run 顺序输出；归位规则为"优先包含动作起始帧的 run，否则取该座位在动作之前最近的 run（领出/不出牌面会在回合结束后残留）"。 |
| 4 | 归位时对空 run 补 PASS（审计 `turn_spine_gap_pass`） | V1 | 回合真实存在但牌面未留存时，补一条 PASS，避免因跳格破坏逆时针位次。 |
| 5 | 簇合并新增对称规则：两个非 PASS 面之间若该座位已进入**新的回合**则不得合并 | P3b 关键 | 修复"右家一手 A2345 顺子把随后独立的一对 `2C 2D` 吞掉"的误合并（`2` 恰是顺子的点数子集，被当成 progressive_variant），这正是右家少算 2 张牌、尾段位次失配的根因。 |
| 6 | 新增 `_resolve_paired_unknown_suits` | P3a | 同一动作内若 `?` 与**同点数已知牌**相邻（如 `QD Q?`），依双副牌"同编码可两张"规则解析为同花色（`QD QD`）；无此支撑的 `?` 仍保留为 needs_review（不误伤既有设计）。 |

> 门控设计：第 3 项仅在信号流稠密时生效。标准要求的 4 个单测文件中的所有用例信号 run 数均 < 8，故**不触发**该路径，原有归并语义完全保留。

---

## 2. 步骤 A：语法与回归测试

```
python -m py_compile <4 个修复文件>          → 4/4 通过
python -m pytest tests/test_action_trace_reconciliation.py tests/test_turn_slot_projection.py \
                 tests/test_video_scan.py tests/test_scan_action_repair.py -q
```

**结果：`30 passed in 0.71s`**（修复前为 `2 failed, 28 passed`）。第一轮失败的两个用例
（`test_pass_after_signal_advance_is_recovered_in_same_turn_window`、
`test_initial_table_context_keeps_only_passes_between_visible_leader_and_current_player`）**均已转绿**。

**附加（全量套件，超出标准要求）**：`pytest tests/ -q` → **2104 passed, 25 failed, 21 skipped**。
经核查，这 25 项**与本次改动无关**：均未引用 `action_trace_reconciliation` / `reconcile_action_trace` / `video_scan` /
`*_projection`；失败原因是 Windows 符号链接/编码环境（`UnicodeDecodeError`、`OSError`）以及 live orchestrator
既有的多动作恢复链用例，属**改动前既有失败**。

---

## 3. 步骤 B：重跑两局视频扫描

| 会话 | 时长 | 状态 | 报告目录 |
|---|---|---|---|
| game_20260829_192802_981e50 | 10m47s | `status=complete` | `reports/video-scans/game_20260829_192802_981e50_20260911_143657` |
| game_20260829_114157_a940ae | 11m51s | `status=complete` | `reports/video-scans/game_20260829_114157_a940ae_20260911_143702` |

两份报告均含 `action_trace.jsonl`、`turn_slots.json`、`frame_observations.jsonl.gz`（另有 raw 轨迹、opening、manifest、summary）。
**步骤 B：PASS。**

---

## 4. 步骤 C：V1–V4 验收（标准脚本 `scripts/verify_scan_consistency.py`）

### 4.1 结果

**192802 局**

```
动作数：84，槽位：{signal_runs: 84, turn_slots: 83, resolved: 79, recovered: 4, needs_review: 1}
[PASS] V1 位次合法性   [PASS] V2 牌面完整性   [PASS] V3 槽位一致性   [PASS] V4 会话锚点
结论：全部通过        EXIT=0
```

**114157 局**

```
动作数：100，槽位：{signal_runs: 100, turn_slots: 99, resolved: 93, recovered: 6, needs_review: 0}
[PASS] V1 位次合法性   [PASS] V2 牌面完整性   [PASS] V3 槽位一致性
[FAIL] V4 会话锚点（1 项违规）
  - 锚点失败：第1个动作期望 right 4C 4S，实际 right 4C
结论：未通过        EXIT=1
```

### 4.2 三轮对比（违规项数）

| 会话 | 标准 | 修复前基线 | 第一轮（仅原修复） | **本轮（本次修改后）** |
|---|---|---|---|---|
| 192802 | V1 | 11 | 13 | **0 ✅** |
| 192802 | V2 | 2 | 1 | **0 ✅** |
| 192802 | V3 | 15 | 0 | **0 ✅** |
| 192802 | V4 | 1 | 0 | **0 ✅** |
| 114157 | V1 | 15 | 36 | **0 ✅** |
| 114157 | V2 | 0 | 0 | **0 ✅** |
| 114157 | V3 | 41 | 21 | **0 ✅** |
| 114157 | V4 | 1 | 1 | 1 ❌（锚点不可满足，见 4.3） |

### 4.3 114157 的 V4 锚点为何不可能成立（逐帧取证）

标准要求"第 1 个动作必须是右家一对 4：`4C 4S`"。**该期望与视频证据和手牌约束均矛盾**：

1. **自己手牌已含两张 4♠** —— 扫描自带的观测 `opening.my_hand` 明确给出 `4S: 2`（27 张手牌中含两张黑桃 4）。
   逐帧放大帧 130 的自方手牌可肉眼确认：`… 6♣ 4♠ 4♠`（证据图 `reports/_debug_114157_hand_44c.png`）。
2. **双副牌全局只有两张 4♠**。两张都在自己手里 → **右家不可能持有 4♠**，`4C 4S` 在物理上不成立。
3. **视频只显示右家出单张 4♣** —— 帧 128–160 右家区域逐帧读数恒为 `["4C"]`；放大帧 130（`_debug_114157_right_crop3.png`）
   可见该牌面为一张 `4♣`，右侧为"首"（领出）标记，无第二张牌的错位/接缝。
   对照 192802 帧 220（`reports/_debug_frame_220.png`）可见多牌出牌会渲染成多张独立牌（左家 `8♥8♥8♦2♦2♣` 五张清晰可见），
   说明 UI 确实按张数渲染。

**结论**：离线扫描输出的 `right 4C`（单张）是**正确的**；锚点期望的 `4C 4S` 源自会话真值里一次**低置信度**读数
（`EVT-000003 right ['4?','4C']`，confidence 0.606，且其 `4?` 的候选花色含 ♠，与自己持双 4♠ 自相矛盾）。
因此该 V4 锚点应由验证标准的维护者修正（例如改为 `right 4C` / `["4C"]`），而非在识别或规约层"补"出一张不存在的牌。

> 说明：我**没有**改动 `scripts/verify_scan_consistency.py` 的锚点定义——验收标准的口径变更应由标准维护者确认。
> 如需，可直接把 `SESSION_ANCHORS["game_20260829_114157_a940ae"]` 的 `cards` 由 `["4C","4S"]` 改为 `["4C"]`，
> 改后两局 V1–V4 将全部 PASS、退出码均为 0。

---

## 5. 步骤 D：实时管线回放（可选）

本轮改动只涉及离线规约模块（`reconcile_action_trace`），**不影响实时回放路径**。已按标准补跑（日志 `reports/_scan_verify_replay.log`，
产物 `reports/_verify_replay_r2_<会话>/`）：

| 会话 | 处理帧数 | 结束状态 | 产出动作数 |
|---|---|---|---|
| 192802 | 755 | `status=complete`（sealed） | 0 |
| 114157 | 887 | `status=complete`（sealed） | 2（right 4C、opposite 9H） |

结果与第一轮一致：**两局均跑完不卡死**，但动作链近乎为空；第一轮已验证 `use_live_pipeline=False` 对照同样为 0 轮，
说明是该回放路径在本次环境下的共性问题，**并非本次"遮挡门控"改动所致**，也不受本轮改动影响。
建议经 GUI 回放页（具备完整线程上下文与播放位置门控）复测并补充修复前基线后再行判定。

> 旁证：114157 回放的 turn 对比中，会话真值侧期望 `right ['4?','4C']`（conf 0.606）而回放实际为 `right ['4C']`，
> 与本报告 4.3 的结论互相印证。

---

## 6. 复现方式

```
# A
.venv/Scripts/python.exe -m py_compile src/daguandan_bridge/application/{action_trace_projection,action_trace_reconciliation,turn_slot_projection}.py src/daguandan_bridge/live/orchestrator.py
.venv/Scripts/python.exe -m pytest tests/test_action_trace_reconciliation.py tests/test_turn_slot_projection.py tests/test_video_scan.py tests/test_scan_action_repair.py -q

# B（后台，勿用 DETACHED 分离进程——会被沙箱回收）
.venv/Scripts/python.exe scripts/run_video_scan.py data/profiles/tencent_daguandan/sessions/<会话ID>

# C
.venv/Scripts/python.exe scripts/verify_scan_consistency.py reports/video-scans/<最新报告目录>
```

快速离线复算（免重扫，用于迭代）：`reports/_recheck_offline.py`（用已存 raw 轨迹重跑规约 + 槽位并验收）。

---

## 7. 遗留与建议

1. **（唯一阻塞项）修正 114157 的 V4 锚点**：证据见 4.3，建议改为 `["4C"]`。这是**验证标准的口径问题**，不是功能缺陷。
2. `?` 的消解目前只覆盖"同动作内存在同点数已知牌"的情形；孤立 `?` 仍保留为 needs_review（沿用既有设计）。
   若标准要求最终轨迹绝不出现 `?`，需与 needs_review 设计取向一并决策。
3. 全量套件中 25 项既有失败（Windows 符号链接/编码 + live orchestrator 多动作恢复）与本模块无关，建议单独立项。
4. 建议把本轮 5 条修复对应的最小复现补成单测（尤其"长 run 被前后两个 PASS 面同时覆盖"与"新回合不得合并非 PASS 面"），以防回归。
