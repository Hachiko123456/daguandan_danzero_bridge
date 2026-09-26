# 发布打包与增量缓存

## 入口与输出

直接运行 `package_release.bat`（无参数）。唯一 BAT 入口仍输出到
`release\current`，默认传入 `-OverwriteExisting -AllowDirtyDevelopmentBuild`。
默认 wheelhouse 在 `%LOCALAPPDATA%\Daguandan\wheelhouse`；缺少所有权标记或完成锁时由 BAT 准备。

正式发布直接调用 `scripts\package_release.ps1 -ReleaseRoot <外部新目录> -WheelhouseRoot <外部wheelhouse>`，**不传** `-AllowDirtyDevelopmentBuild`。该路径禁用缓存复用、创建新环境、执行 PyInstaller `--clean`，并要求干净源树及严格 manifest 验证。开发包不等于正式合格发布。

## staging 与旧包保护

默认每次构建在 `release\.staging\current-<时间>-<随机值>` 中开始，不能预先清空 current；外部输出的 staging 位于其父目录的 `.staging` 下。
新 dist、资源、原生审计、manifest 验证、ZIP、校验和和发布记录完成后才发布。

`Publish-ManagedRelease -Stage <stage> -Destination <current>` 先校验目录，再把 current 移为 current.previous，最后把 stage 移为 current。成功后保留 current.previous；若 stage 提升失败则回滚旧 current，保留失败 stage。已有 previous 只有通过所有权和内容检查才允许清理。构建启动时，如果 current 缺失但 previous 存在，会先检查并恢复 previous。

已有发布目录要求 `.daguandan-release-root` 内容为 `guandan.package-release-root/2`，顶层仅含受管理条目（包括 `build_metrics.json`），整棵目录和祖先路径不得含 symlink、junction 或其他 reparse point。未管理文件、错误 marker 均应拒绝，不得删除。`Clear-ManagedReleaseRoot` 只用于 previous/stage，不能构建前清空 current。输入路径按绝对路径验证；源/目标不能相同或嵌套，也不得发布到文件系统根。项目内仅默认 `release\current` 是特许输出；其他输出必须与项目根分离，wheelhouse 必须与项目及最终输出分离。

## 缓存行为

仅显式启用 `-AllowDirtyDevelopmentBuild` 时复用缓存（默认 BAT 已启用）：

- 环境：`.cache\release\envs\<environment_key前20hex>\env`
- PyInstaller 工作目录：`.cache\release\work\<work_key前20hex>\work`

短路径降低 Windows 长路径风险；receipt 仍校验完整 64 位十六进制 key 和内容。缓存不是旧 dist；每次都在新的 staging 中输出包。环境命中时不重新创建 venv、不重新安装依赖，但仍执行已安装依赖验证。

模块收集前会先在受校验的隔离环境中导入 `rlcard.envs`，准备其首次导入时展开的 JSON 数据，避免首次构建后 `_input_datas` 变化导致下一次无意义重建。

work key 保守绑定源码内容。通常业务源码修改后**环境可命中，work 图重建**；输入完全不变的二次打包两者都可命中。work 命中也不跳过后续打包、审计、manifest、归档和发布。receipt 用来发现损坏，不用于防御能同时修改缓存和 receipt 的攻击者。

## 计时、失败与测量边界

成功结果位于 `release\current\build_metrics.json`：

```json
{
  "schema": "guandan.release-build-metrics/1",
  "status": "passed",
  "total_seconds": 0.0,
  "environment_cache_hit": false,
  "work_cache_hit": false,
  "stages": [{"name": "阶段名称", "seconds": 0.0}]
}
```

发布前状态为 `validated`，发布完成后为 `passed`；构建失败为 `failed`，有 `error` 字段。已创建 staging 的构建在失败时尽力把报告保留在那里；更早的输入校验失败不保证有报告。失败不能假报成功。

冷/热实测期间必须冻结源码、文档及未跟踪文件集合；源身份复核会拒绝构建过程中的变更。对比两次 metrics 的总耗时、各阶段耗时及命中标志，而不是只看控制台的一项耗时。

回归命令：`python -m pytest tests/test_package_release_incremental.py -q`。
测试使用真实 Windows PowerShell 子进程、Parser AST 提取函数/分支、临时 owned 目录和指定 stage 的 Move-Item 故障注入，不执行打包主流程。它们验证发布/回滚、旧包保护、路径与 junction 拒绝、计时字段和缓存分支；**不代表真实 PyInstaller、最终 EXE 或桌面功能已经运行**，实际冷/热构建由发布主代理单独测量。
