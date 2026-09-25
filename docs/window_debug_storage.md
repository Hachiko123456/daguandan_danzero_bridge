# 窗口诊断报告存储

`WindowDebugStorage` 是窗口诊断报告的单一落盘边界。它只使用：

```text
<diagnostics-root>/window_debug/<run-id>/
```

`diagnostics-root` 默认来自现有 `DIAGNOSTICS_ROOT`；也可以传入 `runtime_root`，此时派生为 `<runtime-root>/diagnostics`。不会把报告写入 EXE 所在目录，也不接受任意输出路径。

## 默认隐私和文件

默认每次运行只允许 JSON 文件：

- `report.json`：一次报告，原子替换写入；
- `report.json`：窗口、捕获、识别、错误和 `events` 数组统一写入同一个文件；
- 不再生成 `events.jsonl` 或其他旁车 JSON 文件。

默认会遮蔽密钥、令牌、账号类字段和路径值；路径键即使只包含相对路径也会被遮蔽。截图、像素和编码图像字段会替换为占位文本。截图不会因为报告中出现了相关字段而被保存。

截图必须显式创建运行：

```python
run = storage.create_run("run-1", allow_screenshots=True)
storage.write_screenshot(run, "frame.png", image_bytes)
```

未显式 opt-in 时会抛出 `ScreenshotOptInRequired`。截图只允许安全的 `.png/.jpg/.jpeg` 文件名。

## 确定性保留

可在构造时设置 `max_runs` 和/或 `max_bytes`。清理只检查 `window_debug` 下直接的安全运行目录，不碰 `diagnostics` 下的其他目录（例如 sessions），也不会跟随符号链接或 Windows junction。

运行目录按 `run-id` 的大小写不敏感字典序从旧到新排序：

1. `max_runs` 超限时删除最旧目录；
2. `max_bytes` 超限时继续从最旧目录删除；
3. 排序后的最后一个安全运行始终受保护；即使 `max_runs=0`、极小字节预算或该运行本身超过预算，也至少保留这个最新安全运行，避免把当前证据全部清掉。

清理失败或包含 reparse point 的目录会被跳过并在 `CleanupResult.skipped_run_ids` 中报告。运行 ID 和 JSON/截图文件名都必须是单一路径段，因此 `..`、分隔符和越界路径会直接拒绝。

## 最小示例

```python
storage = WindowDebugStorage(max_runs=20, max_bytes=64 * 1024 * 1024)
run = storage.create_run()
storage.write_report(run, report)
storage.append_event(run, {"type": "probe_finished", "ok": True})
result = storage.cleanup()
```

该模块不修改 GUI、`live_controller.py`、`runtime_layout.py` 或 `config.py`；调用方只需把已有窗口诊断报告和事件交给这个存储边界。

