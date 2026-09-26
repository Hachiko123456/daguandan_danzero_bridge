# 监听截图、自动故障留证与跨电脑运行

## 验收约束

- 原始截图来自一个 `FrameSnapshot`。监听期间不得为了保存图片再次调用窗口捕获。
- 保存对象是最近采集帧；不能默认声称已完成深度识别。已完成的页面/识别证据按帧身份关联。
- 首帧未到或新采集代次没有合格帧时，提示等待下一帧重试，不静默回退独立截图。
- 停止前/失败前帧仍可保存，但不能重新参与新代次识别；界面展示来源、采集时间和帧龄。
- 从未监听且没有保留帧时可独立截图，明确标记 `manual_window_capture`。
- PNG 是用于逐像素复核的证据。录像输入可来自同帧，但 MJPG 编码、抽帧后的录像不是逐像素一致性依据。

## 自动故障证据

监听保留至多4帧的环形缓存（3张前帧和当前帧）。页面连续未知、采集/窗口/识别线程故障触发有界事件包；恢复后至多补1张恢复帧。

每次监听运行最多8个事件、保守预留64 MiB写入预算，单写入线程且等待队列最多4项。已入队但失败的写入也占用预算，避免磁盘异常导致重试风暴；队列拒绝不消耗配额，恢复帧队列暂满可在后续已验证帧重试。未取得首帧时报告 `no_frame`，不伪造故障截图。

捕获失败没有当前失败图时，旧帧只标记为上下文，不冒充失败图。深度识别恢复以实际消费帧的序列和采集时间为依据，不能用新提交帧或空轮询冒充恢复。

新问题现场使用统一 diagnostics 根下的 `cases/case_<stamp>_<8hex>/`：`case.json` 关联现场、配置、run 与 session，`frames/` 保存至少六位数字命名的 PNG/JSON 和 `incident_<id>.json`。截图路径与启动日志都以 `resolve_log_diagnostics_root` 的结果为准，不重新推导 profile/session 下的目录。已引用的证据不得为凑齐问题包而搬动或改写。

旧对局目录里的 `diagnostic_frames` 仍作为 legacy 证据保留，由 collector 按关联关系选择阅读，不自动迁移、不删除。目录由运行时返回，文件夹按钮不会硬编码开发机路径或退出极简模式。

元数据包括：frame ID、generation、capture_seq、捕获时间、来源阶段、窗口客户区/DPI/backend、标准化变换，以及可用的page锚点/按钮和识别摘要。

`raw_sha256`保留原有语义（标准化图像），新增明确别名`standardized_sha256`。原生客户区图像存在时另存`capture_raw_sha256`，不能将两者混淆。离线诊断解码PNG后直接识别，不重新裁剪或标准化。

页面未知超出快速恢复预算后降频探测，不再直接关闭监听；未知帧不会建立session或提交动作。窗口/线程硬故障和用户主动停止仍具有明确处理边界。加倍等待、未确认完整开局和页面未知保持分离；不降低任何模板阈值。

## 统一 diagnostics 目录

默认源码根是 `<项目目录>/diagnostics`；默认发布包根是 `<DaguandanAssistant.exe 所在目录>/diagnostics`。不依赖启动时工作目录、打包机目录或 `_MEIPASS`，也不再默认写入 `logs/diagnostics`。

```text
diagnostics/
  runs/
    .opening-budget.lock                固定的 DiagnosticBudget 协作锁
    <run_id>/                           启动报告、文本日志及滚动日志
      dependency-probes/<probe-id>.json 固定 dependency ID + 32 位十六进制 UUID
      opening/latest.json
      opening/incidents/OPEN-<ms>-<8hex>/
        incident.json
        opening_evidence.json
        repro.json
        recognition_trace.jsonl
  cases/case_<stamp>_<8hex>/
    case.json                           当前现场及关联证据清单
    config/                             可选的三个已知配置 JSON 快照
    frames/
      000001.png
      000001.json
      incident_<id>.json
  exports/
    DaguandanAssistant_problem_<stamp>_<random>.zip
```

`startup_report.json` 记录实际 build ID、平台、所选 profile 目录及配置/资源身份；不把“未选择 profile”伪装成已加载默认资源。`startup.jsonl`、`startup.log` 和异常日志用于复盘。

覆盖优先级保持不变：`DAGUANDAN_DIAGNOSTICS_ROOT` → `DAGUANDAN_DIAGNOSTICS_DIR` → 显式 `DAGUANDAN_DATA_ROOT/diagnostics` → 默认目录。报告记录实际选择。相对路径、受保护资源路径、重解析链和权限失败都会明确报告路径与原因，不静默切换 AppData/TEMP。早期诊断失败不阻止应用尝试启动，但不能伪称证据已写入；导出失败也不能伪称 ZIP 已生成。

**旧日志保持原样**：`logs/` 可继续存放通用日志，原 `logs/diagnostics`、profile/session 的 legacy 证据不搬动、不删除。collector 兼容选择旧根中的相关证据，而不是重跑 doctor 造一个空的新 run。原有 profile、session、用户配置、模板及 user-data generations 不因本次诊断路径统一而迁移；`LOCALAPPDATA` 对 frozen 用户数据的既有作用不变。

frozen 包内的 `diagnostics` 只是严格限定的可变证据命名空间，不是完整性检查的排除目录：只允许未登记的 `runs/<run_id>/` 文本日志/报告、符合命名约定的 case/配置/编号帧/事故 JSON、以及 `exports/` 的固定格式问题 ZIP。doctor 的 dependency-probes 及 opening 的 latest/事故文本仅限上面固定子路径和文件名；`runs/.opening-budget.lock` 仅允许该单一锁名，runs 下的 PNG/ROI 不放行。任意 EXE/DLL/Python、模型、其他二进制文件或额外目录中的文件不在例外内。manifest 已登记的任何文件（包括意外随包发布的证据）仍须大小、哈希一致；目录和文件的重解析链一律拒绝。显式外部 diagnostics 和旧 `logs` 路径仍兼容；显式旧 `logs` 根中的新 case/exports 同样只允许上述固定结构，并非放行任意图片或 ZIP；包内新布局根应直接选择 `diagnostics`，不要选择其中任意子目录作为新根。

## 双击生成一个问题 ZIP

1. 在解压目录双击 `Collect_Diagnostics.bat`；不需要填写参数表单。
2. 脚本只问一次是否包含截图：`Y` 包含，`N` 不包含。也可使用 `Collect_Diagnostics.bat --problem-no-images` 跳过询问；`--help` 查看简短说明。
3. 脚本调用同目录 EXE 的 `--export-problem` 无界面入口，读取已有问题证据，不重跑 doctor。**即使 GUI 无法启动，也应使用该 headless 入口导出**。
4. 默认在 `diagnostics/exports/` 查找 `DaguandanAssistant_problem_<stamp>_<random>.zip`；设置了覆盖目录时以实际根为准。自行检查后再发送给支持人员，脚本不自动上传。

一个 ZIP 应聚合所选/当前 case、当时配置、关联 run 与 session 的可用证据，不把当前导出进程的空 run 当成问题现场。没有 case、配置、run、session、图片或 legacy 证据时，清单必须明确列出缺失、排除或失败原因；不能伪造完整现场，也不能从无关对局凑数据。

**预算与隐私**：自动留证继续遵守前述事件数、队列和 64 MiB 预算；导出也受 collector 的文件数、单文件、总字节预算限制。超预算、读取失败及用户选择不含图片都应反映在 ZIP 清单中，不承诺“所有文件都已打包”。不默认复制整个 profile/model 目录。文本/config 经导出器的隐私处理；截图仍可能包含昵称、牌面及其他窗口内容，不因文本脱敏而变得无敏感信息。即使选择不含图片，也应在分享前检查配置和文本内容。

若 EXE 本身丢失或无法执行，BAT 会明确提示失败，不自行压缩、不伪造成功 ZIP；请先保留并手动复制现有 `diagnostics`（有覆盖时复制实际目录），同时保留相关旧 `logs`。手动复制的原始内容**未经导出脱敏**，不要直接当作安全分享包。不要为了排障删除原证据。

## 跨电脑运行

复制的是打包生成的`release/current/DaguandanAssistant.zip`，不是压缩源码或`.venv`。解压整个目录到有写权限的位置，直接运行`DaguandanAssistant.exe`；`_internal`、`data`和`build_manifest.json`必须一起保留。打包依赖和解释器随包携带。

包内`Launch_DaguandanAssistant.bat`是已有“安装/激活代次”启动器，不应把它当成首次解压的便携启动入口。便携运行请双击上述EXE。

同一发布包不保证任意Windows版本、设备权限、游戏皮肤/缩放都已经实测。窗口必须可识别且可捕获，路径/磁盘可写。不能复制另一台电脑的HWND、采集代次或旧故障帧冒充当前现场。只记录资源差异，不把用户合法的自定义ROI自动判为错误。

## 回归

合成测试始终覆盖并发队列、失败限额、旧代次、手动保存、故障/恢复和目录按钮。7张用户图片按相互独立的随机快照验证，绝不当成完整连续牌局。

外部真实图片测试显式设置`DAGUANDAN_CROSS_MACHINE_FRAMES`为PNG/JSON所在的`diagnostic_frames`目录后，运行`tests/test_cross_machine_listener_frames.py`。无语料时明确skip，不计为通过。测试输出全部进入tmp_path，原图和源profile只读。

这批图本机可识别为table，不构成另一台电脑page_unknown初始原因已定位的证据。新故障包保留真正失败帧，后续才能做确定归因。
