# 固定微信窗口与跨电脑运行：dry-run 验收契约

本文定义 `scripts/validate_fixed_window_portability.py` 的**报告型**验收契约。它只读取 profile 与运行时布局证据，输出可审计 JSON；不会启动或控制微信，不查找 HWND，不移动/缩放窗口，不采集屏幕，也不删除文件。

## 1. 固定窗口契约

验收对象是微信中“腾讯大掼蛋”小程序的客户区，而不是桌面坐标：

- `allow_resize` 必须显式为 `false`。缺失不等于 false，缺失直接不通过。
- `base_size` 是识别基准画面尺寸；当前 profile 约定为 `1280×720`。
- `target_client_size` 是实际固定客户区尺寸；当前约定为 `1280×764`，底部 44 px 由 `viewport_mode=bottom_aspect` 解释。
- `capture_backend` 必须显式为 `printwindow` 才能通过完整契约。`screen` 与 `gdi_screen` 只读取当前可见桌面；`auto` 可能回退到屏幕采集，不能证明遮挡场景。
- `allow_screen_fallback` 必须显式为 `false`。否则 PrintWindow 失败可能被屏幕采集掩盖，报告不能作为遮挡通过证据。
- 运行时必须提供至少一个可读取的布局证据：`runtime_layout.json`、`.daguandan-user-data-root.json` 或 `data/v1/active.json`。
- profile/持久化证据不得写入当前机器的绝对模型、模板、数据或会话路径。运行时布局自身描述的根目录绝对路径属于机器本地事实，报告会保留为 warning；它必须由目标机的 runtime root 推导，不能被 profile 固化。

严格通过（`summary.status=PASS`）表示静态契约完整，不表示已经在真实 Win32 桌面完成了窗口捕获。脚本退出码：`0` 为 PASS，`2` 为 FAIL；JSON 中保留每项证据、发现和副作用声明。

## 2. 运行方式

```powershell
python scripts/validate_fixed_window_portability.py `
  --root . `
  --profile data/profiles/tencent_daguandan/profile.json `
  --runtime-root "$env:DAGUANDAN_DATA_ROOT"
```

不指定 `--profile` 时读取 `data/profiles/*/profile.json` 的第一个排序结果。不指定 `--runtime-root` 时以工程根目录作为只读检查起点；这适合 source checkout 的静态预检。`--output report.json` 只是把同一份报告另存到调用方指定的位置，脚本不会创建父目录，也不会清理任何文件。

## 3. 验收矩阵

每个真实组合都要先运行脚本，再在该组合下执行“前景、部分遮挡、完全遮挡但未最小化”三类窗口场景。表中的“通过”是必须达到的观察结果；任何静态检查 FAIL、客户区尺寸漂移、黑帧、ROI 偏移或牌面识别退化，都算该组合失败。

| 维度 | 组合/场景 | PrintWindow 后端 | 屏幕采集后端 | 记录要求 |
|---|---|---|---|---|
| Windows | Win10 22H2、Win11 23H2 或更高的受支持版本 | 三类场景都应得到非黑、同尺寸客户区 | 仅前景场景可作为基线 | 记录版本、补丁、微信版本、profile 摘要 |
| DPI | 100%、125%、150%（若目标机还有其他缩放，增加同样一行） | 客户区物理像素仍为 `1280×764`；标准化后为 `1280×720` | 前景时同样；遮挡时不得声称通过 | 记录 Windows 缩放与实际 DPI；禁止只记录逻辑尺寸 |
| 分辨率 | 1280×720、1920×1080、2560×1440；多屏时分别记录主/副屏 | 固定客户区可放置且截图尺寸稳定 | 前景时稳定；窗口被覆盖即视为不可用 | 记录显示器物理分辨率、方向、主屏编号 |
| 前景 | 微信窗口可见且未被覆盖 | `backend=printwindow`，图像非黑，锚点/ROI 通过 | `backend=screen` 或 `gdi_screen` 可作为可见性基线 | 保存 JSON 与帧摘要，不要求保存敏感截图 |
| 部分遮挡 | 由记事本或其他窗口覆盖微信客户区的一部分 | 图像仍完整，ROI 与前景基线一致 | 必须标记 blocked/不可用，不得报告通过 | 记录遮挡窗口、遮挡比例、后端和结果 |
| 完全遮挡 | 另一窗口覆盖客户区，但微信未最小化 | 图像仍完整且尺寸一致；不能以屏幕内容替代 | 必须失败或明确 blocked | 记录 PrintWindow 是否实际使用，禁止静默回退 |
| 最小化/不可见 | 微信窗口最小化或不可见 | 不作为通过样本；应产生明确失败码 | 不作为通过样本 | 记录失败原因，不把黑帧当有效证据 |

推荐执行顺序为：Win10/Win11 × 100/125/150% DPI × 三种分辨率，先做前景基线，再做部分遮挡和完全遮挡。资源受限时，至少覆盖两种系统、两种 DPI、1920×1080 与 2560×1440，并且每种后端都要有一条明确结论。

## 4. JSON 审计字段

报告 schema 为 `guandan.fixed-window-portability-dry-run/1`，关键字段如下：

- `inputs`：实际读取的 profile、runtime-root 和布局文件；
- `observed_profile`：`allow_resize`、`capture_backend`、回退开关和几何配置；
- `checks`：逐项 PASS/FAIL/WARN 及其 evidence；
- `path_risks`：绝对路径、可接受的运行时根路径、相对路径及遍历风险；
- `findings`：可操作的阻断原因；
- `side_effects`：明确证明未控制微信、未采集屏幕、未调整窗口、未删除文件；
- `summary`：机器可读状态、计数与退出码对应关系。

报告中的绝对路径可能包含本机盘符，因此报告本身只适合在审计存储中使用；不要把绝对路径复制回 profile 或 session manifest。跨电脑验收应把同一份契约脚本和 profile 复制到目标机，重新解析目标机 runtime root，不能直接复用源机绝对路径。

## 5. 当前仓库的预期结果\r\n\r\n当前 `data/profiles/tencent_daguandan/profile.json` 已显式声明 `allow_resize=false`、`capture_backend=printwindow` 和 `allow_screen_fallback=false`，因此 source checkout 下脚本应为 `PASS`，但允许有一条 `runtime_layout_evidence=WARN`。发布/冻结运行仍必须提供 runtime-layout evidence；如果目标机不支持 `PrintWindow`，应在诊断中明确报告并由用户选择兼容采集策略，不能静默把被遮挡的屏幕帧当成可信证据。
