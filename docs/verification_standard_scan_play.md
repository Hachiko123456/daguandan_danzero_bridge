# 扫描出牌功能修复 — 完整验证标准

> 本文档是独立的验证作业指导书。执行者不需要了解修复过程，只需按步骤执行命令、
> 对照验收标准判定 PASS / FAIL，并输出验证结论。
> 所有路径相对于项目根目录 `C:\project\python_project\daguandan_danzero_bridge`。

## 1. 背景：被修复的两个 Bug

本项目是一个掼蛋（双副牌，4 人，自己/右家/对家/左家）出牌识别系统，
通过扫描对局视频（`game.avi`）重建完整出牌轨迹（`action_trace.jsonl`）。

- **Bug 1 位次错误**：出牌顺序必须严格按逆时针 `self → right → opposite → left → self`
  推进。修复前出现"左家直接到右家""对家直接到自己"等跳位。
  根因：跨墩的 PASS（不出）记录被错误合并、出牌动画在邻座区域产生幽灵动作。
- **Bug 2 遮挡误识别**：左家出 ≥5 张牌时，牌尾被我方"出牌/提示/不出"按钮遮挡，
  遮挡帧的识别结果（张数虚高、花色全为未知 `?`）污染了最终牌面。
  正确行为：等自己出完牌、按钮消失后再识别左家的牌，以干净帧为准。

修复涉及 4 个文件（仅供审查，验证以行为为准）：

- `src/daguandan_bridge/application/action_trace_projection.py`（遮挡判定 + 候选完整性优先）
- `src/daguandan_bridge/application/action_trace_reconciliation.py`（PASS 跨墩拆分 + 幽灵过滤 + 完整性优先）
- `src/daguandan_bridge/application/turn_slot_projection.py`（槽位恢复完整性优先）
- `src/daguandan_bridge/live/orchestrator.py`（实时管线：左家出牌期间按钮可见则暂停识别）

## 2. 领域规则（人工核对依据）

- 座位：`self`（自己，画面下方）、`right`（右家）、`opposite`（对家）、`left`（左家）。
- 位次：逆时针 `self → right → opposite → left → self`，已出完牌的座位被跳过。
- 每座位起手 27 张（自己可能因进贡等规则不同，以视频为准）。
- 一墩（trick）：一家领出后，其余三家依次跟牌或 PASS；三家全 PASS 则墩结束，
  最后出牌者的下一家领出下一墩；若最后出牌者已出完牌，由其伙伴**接风**领出。
- 牌编码：点数 + 花色，花色 `S/H/D/C`（黑桃/红桃/方块/梅花），`?` 表示未知花色
  （不允许出现在最终结果中）。双副牌：同一编码的牌最多出现 2 张。
  例：`8H 8H 8D 2D 2C` = 三带二 88822。

## 3. 测试对局

| 会话 ID | 会话目录 | 视频 |
|---|---|---|
| game_20260829_192802_981e50 | `data/profiles/tencent_daguandan/sessions/game_20260829_192802_981e50` | `video/game.avi`（含 `frame_index.jsonl`） |
| game_20260829_114157_a940ae | `data/profiles/tencent_daguandan/sessions/game_20260829_114157_a940ae` | `video/game.avi`（含 `frame_index.jsonl`） |

人工核对锚点（已逐帧确认，作为 V4 验收依据）：

- **192802 局**：左家必有一手三带二 `2C 2D 8D 8H 8H`（即 88822，共 5 张，
  不得识别为 6 张或含 `?`）；自己必有一手 `4D 4S AC AD AS`（AAA44）。
  证据帧：`reports/_debug_frame_205.png`（按钮遮挡左家 2♣ 的错误帧）与
  `reports/_debug_frame_220.png`（按钮消失后的干净帧）。
- **114157 局**：第 1 个动作必须是右家一对 4：`4C 4S`。

## 4. 验证步骤

### 步骤 A：语法与回归测试（约 1 分钟）

```bash
cd /c/project/python_project/daguandan_danzero_bridge
python -m py_compile \
  src/daguandan_bridge/application/action_trace_projection.py \
  src/daguandan_bridge/application/action_trace_reconciliation.py \
  src/daguandan_bridge/application/turn_slot_projection.py \
  src/daguandan_bridge/live/orchestrator.py

python -m pytest tests/test_action_trace_reconciliation.py \
  tests/test_turn_slot_projection.py tests/test_video_scan.py \
  tests/test_scan_action_repair.py -q
```

**通过标准**：编译无错误；pytest 全绿（修复前基线为 30 个测试全部通过，不允许新增失败）。
若个别测试失败，需先判断该测试是否编码了旧的错误行为（如"张数优先"）——
属于旧行为的测试应更新期望值并在验证报告中说明；否则判定 FAIL。

### 步骤 B：重跑两局视频扫描（每局约 5–15 分钟）

⚠️ 单局扫描超过 5 分钟，前台 shell 默认超时会杀进程。必须用后台分离进程 + 轮询，
或把 shell 超时调到 30 分钟以上。Windows 下推荐：

```bash
cd /c/project/python_project/daguandan_danzero_bridge
python - <<'EOF'
import subprocess, sys
from pathlib import Path
for game in ("game_20260829_192802_981e50", "game_20260829_114157_a940ae"):
    session = Path("data/profiles/tencent_daguandan/sessions") / game
    log = Path(f"reports/_scan_verify_{game}.log")
    with log.open("w", encoding="utf-8") as fh:
        subprocess.Popen(
            [sys.executable, "scripts/run_video_scan.py", str(session)],
            stdout=fh, stderr=subprocess.STDOUT,
            creationflags=0x00000008 | 0x00000200,  # DETACHED | NEW_PROCESS_GROUP
        )
        print(f"started {game}, log={log}")
EOF
```

轮询日志直到出现 `status=complete output=<报告目录>`（两局可并行）。
报告目录形如 `reports/video-scans/<会话ID>_<时间戳>`，取每个会话**最新**的一份。

**通过标准**：两局日志均出现 `status=complete`；报告目录内含
`action_trace.jsonl`、`turn_slots.json`、`frame_observations.jsonl.gz` 三个文件。

### 步骤 C：按验收标准检查扫描结果（每局约 10 秒）

对每个会话的最新报告目录执行：

```bash
python scripts/verify_scan_consistency.py reports/video-scans/<报告目录>
```

脚本自动执行以下四条标准，退出码 0 = 全部通过，1 = 存在违规：

| 标准 | 内容 | 通过条件 |
|---|---|---|
| **V1 位次合法性** | 用 `project_trick_turn` 逐步模拟位次推进，每个动作的演员必须等于规则推导的期望演员（含跳过已出完座位、接风、新墩领出） | 0 项违规 |
| **V2 牌面完整性** | 非 PASS 动作必须有牌；不得含未知花色 `?`；同一牌编码不超过 2 张 | 0 项违规 |
| **V3 槽位一致性** | `turn_slots.json` 每个 turn 槽位状态为 resolved/recovered，槽位演员与动作演员一致，且与动作序列按序对齐 | 0 项违规 |
| **V4 会话锚点** | 第 3 节中该会话的人工核对锚点必须在动作轨迹中出现且完全一致 | 0 项违规 |

**修复前基线**（192802 局旧报告 `game_20260829_192802_981e50_20260911_120153_756721`）：
V1 11 项违规、V2 2 项、V3 15 项、V4 1 项违规。
修复后的**通过标准是两局 V1–V4 全部 PASS（退出码 0）**。
114157 局修复前基线报告为 `game_20260829_114157_a940ae_20260911_131528`，
可对其运行同一脚本对比改进幅度。

### 步骤 D：实时管线回放验证（可选，每局约 10–20 分钟）

用实时管线回放两局视频，验证"按钮遮挡期间不识别左家出牌"的门控不导致卡死或漏牌：

```bash
python - <<'EOF'
import sys
sys.path.insert(0, "src")
from pathlib import Path
from daguandan_bridge.live.replay import replay_video_through_live_pipeline
for game in ("game_20260829_192802_981e50", "game_20260829_114157_a940ae"):
    session = Path("data/profiles/tencent_daguandan/sessions") / game
    result = replay_video_through_live_pipeline(session / "video" / "game.avi", use_live_pipeline=True)
    print(game, "->", result)
EOF
```

（入口见 `src/daguandan_bridge/live/replay.py`，参数以该文件实际签名为准；
同样需要用后台分离进程跑。）

**通过标准**：回放正常跑完不卡死；产出的动作序列同样满足 V1、V2 两条标准
（可把回放输出的动作轨迹整理成 `action_trace.jsonl` 格式后用同一脚本检查，
或直接检查位次推进与未知花色）。

## 5. 总体验收结论判定

| 项 | 必须 |
|---|---|
| A 回归测试 | 全部通过（除经论证属旧错误行为的测试已更新） |
| B 两局扫描 | 均 `status=complete`，产物齐全 |
| C 两局 V1–V4 | **全部 PASS，退出码 0** |
| D 回放（可选） | 不卡死，轨迹满足 V1/V2 |

验证报告需包含：每条标准的实际输出、违规明细（如有）、与修复前基线的对比、
以及最终 PASS / FAIL 结论。

## 6. 常见问题

- **扫描被 shell 超时杀掉**：见步骤 B，必须后台分离运行。
- **报告目录选错**：`reports/video-scans/` 下同一会话有多份历史报告（含修复前的），
  务必选时间戳最新的一份。
- **锚点找不到**：先在报告目录 `action_trace.jsonl` 中 grep `8H` 确认实际识别结果，
  若牌面对但张数/花色编码顺序不同，注意脚本比较的是排序后的牌集合。
