# 脚本目录说明

当前目录保留脚本的稳定路径，避免移动文件破坏发布脚本、测试和用户已有命令。
脚本按用途逻辑分组；后续新增脚本必须加入本说明并提供 `--help`（PowerShell 使用 `-?`）。

## 日常公开入口

| 用途 | 入口 | 说明 |
|---|---|---|
| 快速测试 | `run_test_layers.py` | 运行 unit/contract/visual_fixture/integration/replay/data_quality 六层测试 |
| 故障范围验收 | `run_failure_scope_acceptance.py` | 现有对局抽样、当前真实截图和异常范围检查 |
| 视觉回放 | `run_listener_regression.py` | 生产 Live V2 视觉监听回放 |
| 固定验收 | `run_five_session_acceptance.py` | 固定五局验收包装器 |
| Windows 检查 | `run_windows_smoke.py` | 真实窗口、DPI、采集链路 smoke |
| 窗口诊断 | `run_window_debug.py` | 只读窗口和单帧诊断 |
| session 预检 | `qualify_sessions.py` | 资源身份、TruthLog、视频和开局可观测性预检 |

## 发布工具

- `prepare_release_wheelhouse.ps1`：准备外部依赖 wheelhouse。
- `package_release.ps1`：执行 PyInstaller、原生审计、manifest、压缩和校验。
- `qualify_release.py` / `qualify_release.ps1`：验收发布包。
- `manage_release.py` / `manage_release.ps1`：安装、激活、回滚发布代次。
- `generate_release_lock.py`、`verify_release_inputs.py`、`generate_build_manifest.py` 等：发布内部工具，不作为日常入口。

批处理入口 `package_release.bat` 默认输出到 `release\current`，不再要求每次手工传入路径。

## 数据和 TruthLog

`validate_session_corpus.py`、`validate_truth_logs.py`、`build_session_dataset_catalog.py`、
`generate_missing_truth_logs.py`、`build_saveable_truthlogs.py`、`repair_truth_trick_ids.py`、
`verify_first_play_regions.py` 和 `run_video_scan.py` 属于数据维护/验证工具。

## 诊断和兼容工具

`diagnose_*.py`、`verify_terminal_placement_inference.py`、`trim_live_session.py`、
`run_shadow_live_replay.py` 等属于专项诊断或兼容验证，不应作为产品主流程入口。

## 待归档的旧入口

以下脚本仍保留原路径以兼容历史命令，但不应继续扩展：

- `audit_session_replays.py`：固定历史回放集合，功能可逐步并入 `run_listener_regression.py`。
- `run_full_flow_validation.py`：旧的一键 pytest + Shadow Live 入口，功能与六层测试及故障范围验收重叠。
- `run_selected_unverified_rescan.py`、`run_selected_unverified_rescan_quiet.py`：固定未验证数据重扫。
- `audit_batch_scan_batch.py`、`verify_batch_scan_detailed.py`、`verify_scan_consistency.py`：旧批量扫描链路。

归档前需要确认没有外部定时任务或人工流程依赖；归档时优先移动到 `scripts/legacy/` 并保留迁移说明，不直接删除。

## Help 合同

- Python：`python scripts/<name>.py --help` 必须退出码为 0 并说明用途、输入和输出。
- PowerShell：`powershell -File scripts/<name>.ps1 -?` 必须退出码为 0；脚本头部应有参数说明。
- JSON（如 `pyinstaller_live_v2_collection.json`）不是可执行脚本，不要求 help，但必须在发布说明中说明用途。
