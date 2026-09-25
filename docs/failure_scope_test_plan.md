# Failure-scope acceptance test plan

## 目标

`run_failure_scope_acceptance.py` 是一个只读验收入口，组合三类证据：

1. 从 `data/profiles/tencent_daguandan/sessions` 发现已有可回放对局，按稳定 seed 选择 5 局并做 TruthLog advisor replay；
2. 对当前 profile 下最新的 `diagnostic_frames/000001.png` 和 `000002.png` 做真实识别；
3. 在开局安全边界上重复运行故障注入矩阵，检查故障只局部阻塞且不合成动作。

脚本只写报告目录或显式报告路径，不写入原始 session、生产源码或已有测试。

## session 发现与选择

- 只接受同时具备 `truth_log.json`、`video/game.avi`、`video/frame_index.jsonl` 的目录；`manual_diagnostic` 不作为对局回放候选。
- 优先级按证据加权：已验证 TruthLog、TruthLog 存在、`timeline.jsonl` 中存在 `initial_state_confirmed`、manifest 中可辨识的 live/listener/runtime 来源、视频和帧索引。
- 同分候选使用 `SHA256(seed + "\\0" + session_id)` 稳定排序，不依赖文件系统枚举顺序。
- 报告记录候选总数、选中的 session、证据旗标和稳定 key，默认选择 5 局。
- 回放使用现有 TruthLog advisor replay；advisor 的决策日志关闭，回放产物写入报告目录下的 `replay/`。

## 当前真实截图验收

默认自动选择最新的 `manual_diagnostic/**/diagnostic_frames` 帧对，也可以通过 `--diagnostic-frames` 显式指定目录。两张帧都必须满足：

- page stage 为 `table`，牌桌锚点可用；
- `my_hand` 恰好 27 张；
- round level 与 wild rank 都是 `10`；
- `lead_player` 与 `current_player` 都是 `self`；
- 通过 `OpeningTracker` 两帧确认后状态为 `READY_WAITING_FIRST_ACTION`（内部 reason 为 `ready_waiting_first_action`）；
- 没有真实首出事件时 `opening_action` 必须为空，不能伪造动作；
- `roi.critical_play_overlap` 可以出现，但必须是 warning，且 `opening_blocking` 必须为 false。当前 profile 的 overlap 只影响 action-blocking 诊断，不得把开局入口封死。

截图通过 `SessionDiagnosticFrameStore` 读取并校验 PNG/sidecar 哈希，识别过程不会重新捕获窗口。

## 故障注入矩阵

| 场景 | 注入 | 预期范围 | 必须证明 |
|---|---|---|---|
| `roi_overlap` | 保留真实 ROI overlap | warning-only | 不 opening-blocking、不 reset session |
| `pass_missing` | self 座位 pass 信号缺失 | seat-local | 仅 self 局部阻塞，不产生动作 |
| `timer_missing` | self 座位 timer 信号缺失 | seat-local | 仅 self 局部阻塞，不产生动作 |
| `button_missing` | self 座位 button 信号缺失 | seat-local | 仅 self 局部阻塞，不产生动作 |
| `single_frame_anchor_failure` | 单帧 anchor 低于阈值 | frame-local | 后续正常帧可恢复，不产生 opening action |
| `consecutive_page_unknown` | 连续两帧 page unknown | frame-local | 不升级为全局 session reset，正常帧可恢复 |
| `single_seat_action_uncertain` | right 单座位动作不确定 | seat-local | 只阻塞 right，其他座位范围不被污染 |
| `duplicate_frame` | 重复 frame identity | frame-local | 记录 `duplicate_frame`，不重复投票、不产生动作 |
| `out_of_order_frame` | 单帧时间戳乱序 | frame-local | 丢弃/重置局部证据，随后回到等待首动作 |

每个场景的报告都记录注入步骤、阻塞座位、tracker reason、恢复状态、动作列表和 synthetic action 数量。通过条件是：动作列表为空、synthetic action 为 0、没有 session reset，并在需要恢复的场景最终回到 `ready_waiting_first_action`。

## 命令

显示帮助：

```powershell
python scripts/run_failure_scope_acceptance.py --help
```

正式验收（TruthLog replay + 真实截图 + 故障矩阵）：

```powershell
python scripts/run_failure_scope_acceptance.py --output "$env:TEMP\daguandan-failure-scope-acceptance"
```

显式指定当前截图目录和报告文件：

```powershell
python scripts/run_failure_scope_acceptance.py `
  --diagnostic-frames "data/profiles/tencent_daguandan/sessions/manual_diagnostic/<capture>/diagnostic_frames" `
  --output "$env:TEMP\failure_scope_acceptance.json"
```

快速回归（跳过回放，只跑截图和故障矩阵）：

```powershell
python scripts/run_failure_scope_acceptance.py --skip-replay --output "$env:TEMP\failure-scope-smoke"
```

退出码：`0` 表示所有启用的验收项通过；`1` 表示报告生成但验收失败；`2` 表示参数、路径或运行错误。JSON 报告中的 `source_integrity.selected_sessions_unchanged` 必须为 `true`。

## pytest 回归

```powershell
pytest -q tests/test_failure_scope_acceptance.py
```

测试覆盖 CLI help、稳定五局选择、两张真实截图的精确字段、ROI warning/opening 非阻塞、完整故障矩阵、JSON 报告外置以及源 session 哈希不变。
