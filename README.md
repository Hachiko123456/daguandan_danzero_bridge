# 大掼蛋智能助手

这是一个独立的 Windows 11 项目，用于标记本地大掼蛋图片、维护识别模板，并调用本地策略模型给出建议。默认策略为 FableDan，也可切换到 DanZero；项目不会对游戏客户端执行点击、按键或自动出牌。

## 安装与启动

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe run.py
```

也可以双击 `start_gui.bat`。在“标记与模板”页选择包含截图的本地文件夹，即可开始配置区域或裁剪模板。

### 源码与发布版数据目录

源码运行仍使用仓库内的 `data/`，便于本机开发和现有流程保持不变。打包后的
EXE 会把发布目录中的 `data/` 当作不可写的资源种子；首次启动时按 build ID
原子复制到：

```text
%LOCALAPPDATA%\DaguandanAssistant\data\v1\generations\<build-id>\data\
```

发布版的 sessions、截图、模型/模板修改、日志、诊断、偏好和缓存都写在
`%LOCALAPPDATA%\DaguandanAssistant` 下，不会修改 EXE 所在目录。测试或便携
部署可用绝对路径环境变量 `DAGUANDAN_DATA_ROOT` 替换该根目录。

从旧便携版迁移时必须显式执行：

```powershell
.\DaguandanAssistant.exe --migrate-portable-data "D:\旧版目录"
```

迁移只复制白名单内的 profile、模板、模型、sessions 和截图，旧目录保持原样；
自动校准缓存、诊断缓存和未知文件不会导入。迁移会创建新的数据 generation 与
本地回执，不会修改或删除旧 generation。命令成功后会立即原子切换 active
pointer，因此下一次启动直接使用迁移后的数据；命令输出中的绝对 `receipt_path`
是恢复凭据。若要恢复迁移前的 pointer，执行：

```powershell
.\DaguandanAssistant.exe --restore-portable-migration `
  "%LOCALAPPDATA%\DaguandanAssistant\migration_backups\pm-....json"
```

恢复采用 compare-and-swap：只有当前 pointer 仍是该回执记录的迁移目标时才会
恢复；若之后已有新迁移、升级或手动 generation 切换，它会拒绝覆盖。恢复操作
写回同一回执并可安全重复执行，迁移生成的数据目录和旧便携目录都不会被删除。

旧版稳定基线注册后会保留一份不可变的批准副本，并另外创建实际运行副本。每次
查询版本状态、激活、回滚或通过 `Launch_DaguandanAssistant.ps1 -VerifyOnly`
启动前，都会逐文件校验运行副本中的 EXE、DLL、模板、模型和根目录资源。只有
以下明确的运行输出目录允许新增、修改或删除文件：`logs/**`、`reports/**`、
`diagnostics/**`，以及任意 profile 下的 `sessions/**`、`screenshots/**`、
`diagnostics/**`、`truth_log_batch_reports/**` 和 `models/benchmarks/**`。其他
文件缺失、变化或额外出现都会使校验关闭式失败；校验动作不会自动覆盖或修复
现场文件。

### 离线、锁定的发布构建

正式打包不复用开发虚拟环境中的包，也不会在构建时访问软件源。首先用唯一允许
联网的准备脚本生成外部 wheelhouse；它必须与仓库中提交的依赖、wheelhouse 和
CPython 工具链哈希完全一致：

```powershell
.\scripts\prepare_release_wheelhouse.ps1 `
  -WheelhouseRoot "C:\DaguandanBuildInputs\wheelhouse-cp312-win_amd64"
```

随后指定一个尚不存在、位于仓库外部的唯一输出目录进行打包：

```powershell
.\scripts\package_release.ps1 `
  -ReleaseRoot "C:\DaguandanBuilds\candidate-20260831-001" `
  -WheelhouseRoot "C:\DaguandanBuildInputs\wheelhouse-cp312-win_amd64"
```

脚本会在 ReleaseRoot 内创建全新的 build venv，并只使用
`--isolated --no-index --require-hashes` 安装锁定 wheel；构建环境会清除 Python、
Qt/QML、Java、Conda 和 Poppler 等宿主变量。PyInstaller 完成后必须通过原生 PE
来源审计；未知来源、冲突 ICU、Java/Anaconda/Poppler DLL、异常 Qt/CRT、UPX 或
缺失导入都会令构建失败。`native_dependency_audit.json`、完整安装包清单和所有
锁文件摘要均进入并受 `build_manifest.json` 哈希保护。

创建 build venv 之前的引导解释器统一使用 `-I -S`，不会加载宿主机的
`site-packages`、`.pth`、`sitecustomize` 或用户 site。脚本还会生成
`bootstrap_python_audit.json`，逐项确认 no-site/isolated 标志和 `sys.path` 只含
已锁定 CPython 根目录；该审计随发布包进入严格清单，正式资格流程会再次验证。

CPython 本身也不是只锁一个 `python.exe`：`python_runtime.lock.json` 完整记录
会影响 venv 与 PyInstaller 输出的基础运行时文件，包括 `Lib` 标准库、`venv`、
`encodings`、ensurepip wheels、`DLLs`、`libs`、解释器和 Python/CRT DLL。
校验会拒绝文件缺失、额外文件或任一哈希变化。工具链来源固定为 python.org 的
`python-3.12.0-amd64.exe`，安装包大小与 SHA256 写在
`release_toolchain.lock.json`。只有明确升级 CPython 时，才可在干净安装目录运行：

```powershell
.\.venv\Scripts\python.exe -I .\scripts\refresh_python_runtime_lock.py `
  --python .\.venv\Scripts\python.exe
```

生成后必须审查工具链来源、完整文件差异，并重新运行发布锁测试；不能为绕过某台
机器的校验失败而直接修改清单。

### 发布版固定 FableDan 基准

发布目录内双击 `Run_FableDan_Fixed_Benchmark.bat`。启动器会为本次运行分配唯一
结果文件，把它通过 `--benchmark-output` 明确传给 EXE，并在退出码为 0 后用
`-LiteralPath` 读取和校验该精确文件；不再从 EXE 目录猜测“最新结果”。默认结果
位于 `%LOCALAPPDATA%\DaguandanAssistant\benchmarks\`（设置
`DAGUANDAN_DATA_ROOT` 时位于该根目录的 `benchmarks\`），不会写入只读发布包。

自动化调用可自行指定一个尚不存在的绝对路径：

```powershell
.\DaguandanAssistant.exe --fabledan-fixed-benchmark `
  --benchmark-output "D:\Benchmark Results\candidate.json"
```

标准输出是 `guandan.fabledan-benchmark-cli/1` JSON，其中 `output_path` 是实际写入
的绝对路径，`output_sha256` 可用于后续审计。输出已存在、JSON 不合法或 EXE 返回
非零码时，启动器会原样失败，绝不回退到旧结果。

## 标记与模板

页面会递归读取所选本地文件夹中的 PNG、JPG、JPEG 和 BMP 图片；图片两侧提供上一张/下一张箭头，并显示当前序号，方便连续检查样本。

选择一张图片后会立即显示预览。点击“查看区域配置”打开独立配置窗口；在表格中多选区域后点击“标注选中区域”可显示不同颜色的框和中文名称，再点击同一个按钮即可隐藏这些框。编辑保存时请只选中一个区域；名称通过中文下拉框选择，`x/y/w/h` 会写回 `regions_config.json`，比例坐标会自动重算。区域配置包含“换桌 / 再来一局区域”，用于匹配结算页的对应按钮。

在“区域配置”模式下先选择区域名称，再用鼠标左键拖动图片框选，松开后点击“保存区域坐标”即可写回配置；图片预览缩放不会改变原图坐标。模板裁剪模式下，模板类型、模板标签和来源角色均可从下拉列表选择；模板标签会显示中文含义，也可以输入新标签。界面在普通窗口下上下排列预览与工具，在宽屏或全屏下自动切换为左右工作区。

“单图识别 / 策略测试”打开后会后台使用区域库和模板库自动识别当前截图：

- `level_rank` 区域 + `rank/*_level` 模板：识别当前级牌，并按当前掼蛋规则推导百搭级别；
- `my_hand` 区域 + `rank/suit` 手牌模板：识别我方手牌，牌面不完整或置信度不足时保留在输入框中手动补全；
- `*_play` 区域 + 出牌模板、`passed_*` 状态模板：识别“谁出了什么牌/谁不出”；
- `first_play_*` 状态模板：识别本轮首出者；`timer_*` 区域 + `timer/active` 模板：在画面存在活动计时器时识别当前行动者。

识别不到的字段会显示为“待手动确认”，不会把猜测值发送给策略模型。页面会把手牌显示为带颜色花色图案的牌面卡片，并在原图上绘制识别框、牌面和识别类别；页面整体采用分区滚动布局，鼠标滚轮不会改变识别下拉框的值。确认后点击“运行策略测试”，程序会构建 `GuanDanState`，异步调用所选模型，显示推荐动作、耗时、识别置信度、参数摘要和 `engine_input` JSON。当前识别基于项目模板的视觉匹配，不依赖 OCR；模板不足时仍可手动修正。

区域角色包括 `hand`（手牌）、`play`（出牌）、`anchor`（锚点）和 `generic`（通用）。模板记录和图片资产分别保存在 `templates_config.json` 与 `templates/`。

## 模板资产

现有腾讯大掼蛋模板已复制到：

```text
data/profiles/tencent_daguandan/templates/
```

在“标记与模板”页将模式切换为“模板裁剪”，从任意本地图片文件夹选择样本，拖拽图片框选 ROI，选择模板类型、标签和样本来源后点击“裁剪并保存模板”。
模板记录以表格展示，包含绝对坐标和比例坐标；按钮模板可直接选择或裁剪“超级加倍”“加倍”“要不起”等标签。
`rank`/`suit` 会提取前景并生成白底黑字模板，其余类型保留原始颜色；每条记录都会写入
`templates_config.json`，包含来源图片、来源角色、绝对坐标、比例坐标和标准画面尺寸。模板列表支持多选删除，删除不会影响原始图片。

支持的模板类型为 `rank`、`suit`、`anchor`、`button`、`status`、`effect` 和 `timer`；模板文件按类型保存到 `templates/<kind>/`。

牌型动画请在“标注与模板”页面选择 `effect` 后采集。下拉项已覆盖：单张、对子、三张、三带二、钢板、三连对、连对、顺子、炸弹、同花顺和天王炸。应定位到特效文字最清晰、对比最高的一帧，只框选特效主体，不要把出牌区牌面或动态背景一起框入；只有单一关键帧漏检时，才补充另一张代表性关键帧。`effect` 只用于暂停读牌，命中后会等待动画消退和短暂静止窗口，并不把模板标签当作牌型结论。

单图页面会单独显示图片识别耗时；打开页面后会后台预加载当前策略模型。测试结果会显示总耗时、模型初始化耗时和推理等分段耗时，便于区分首次加载与后续调用速度。动作按钮区会自动识别并标出“出牌”“不出”“要不起”“加倍”“超级加倍”等按钮值。

## 策略模型与更换模型

调用前必须确认级牌、百搭牌、当前行动者、本轮首出者与自己的手牌。截图本身不会发送给模型，只会把识别/确认后的 `GuanDanState` 转成策略输入。发布版支持 FableDan 与 DanZero；默认使用 FableDan。

```python
from daguandan_bridge.danzero import DanzeroAdvisor, GuanDanState

state = GuanDanState()
state.set_context(
    round_level="2",
    wild_rank="2",
    current_player="self",
    lead_player="self",
)
state.confirm_hand(("3S", "4H", "5D"))

advice = DanzeroAdvisor().recommend(state)
print(advice.cards)
```

源码版模型位于 `data/profiles/tencent_daguandan/models/`；发布版则使用当前
用户数据 generation 下同样的相对路径。FableDan 使用 `best.npz`，DanZero
使用 `danzero/q_network.ckpt`。请勿覆盖 EXE 旁边的只读种子；退出程序后替换
当前 generation 中的同名文件、再重新启动即可生效。也可通过
`DAGUANDAN_DANZERO_CKPT` 环境变量临时指定 DanZero 权重路径。

## 验证

```powershell
$env:QT_QPA_PLATFORM = "offscreen"
$env:PYTHONPATH = "src"
.\.venv\Scripts\python.exe -m pip install -r requirements-dev.txt
.\.venv\Scripts\python.exe -m pytest -q
```

## 半自动实时助手

“实时助手”只观察窗口、维护对局状态并给出所选策略模型的建议，不会点击、按键或控制游戏。完整的实时操作说明、状态流转、日志字段和测试命令见 [实时助手说明](docs/live_assistant.md)；下面是必须了解的使用方式。

1. 保持腾讯大掼蛋窗口尺寸稳定，点击“识别当前页面（持续监听）”。程序会自动切换到置顶的 `500×245` 极简推荐浮窗，顶部用四个紧凑座位格显示当前墩出牌情况，完整助手最小化到后台；浮窗和完整助手共用同一个控制器、识别器与策略模型，不会重复计算。
2. “保存方式”默认是“对局录制”：按同一监听阶段分别确认初始27张手牌、级牌和首出/首手证据，稳定比较不包含置信度浮点数或模板来源；不完整历史不会强行建局。选择“不保存”时不保存对局录像和session，但启动日志与限额诊断独立管理。原“全程录制”现为“完整牌桌录制”，只在准备、发牌和对局牌桌上录制；大厅、未知页面和结算页只监听，不持续录制媒体。该选择会记住，并在开始监听后锁定。
3. 首发候选只是单图提示，绝不作为开局承诺。加倍或超级加倍按钮存在时会清空该提示；真正的首发仍由运行中的开局状态机在加倍控件消失后确认。
4. 对局中可暂停/继续，也可手动“结束并封存”。检测到结算页的“继续游戏”或“换桌”模板后，会自动封存本局并恢复监听页面，无需再次点击开始按钮。

采集后端默认使用 `auto`：优先通过 `PrintWindow` 读取目标窗口，即使牌桌被其他窗口盖住也不把遮挡内容录入；只有后台窗口采集失败时才回退到可见屏幕采集。浮窗会显示实际使用的采集后端。回退到 `screen` / `gdi_screen` 后，只要检测到本程序或其他窗口覆盖牌桌客户区，当前帧会在录像、识别和策略计算之前被拒绝，流程暂停并明确提示移开窗口。单屏且无法把浮窗放到牌桌外时，不允许带遮挡继续推荐。

座位顺序固定为 `自己 → 右家 → 对家 → 左家 → 自己`。实时状态机始终只提交当前应行动座位的出牌区域；不出仅由对应座位的 `passed.png` 模板判定。两帧相同的明确花色结果即可确认；出现 `?` 时会额外观察一帧，连续三帧点数和张数一致即保留 `?` 提交。每个视觉出牌在短暂窗口内都会继续旁路复核；后续两帧出现完整且不同的稳定牌面时，可统一修正点数、花色、张数和牌型，规则层会从原动作位置原子重放全部后续状态，修正不会增加第二个动作或改变原 `action_id`。稳定可见的非空牌面视为已经发生的动作：规则层会枚举万能牌的所有解释，先选能压桌面的最强组合，但“不能压过重建桌面”只记审计告警，不再否决出牌或转换成不出。结构上不可能的瞬态 OCR 仍会过滤；若活动计时器已经切到下一家，穷尽采样后仍无法解释，则按物理牌继续推进，避免 `conflicting_valid_candidates` 阻断整局。命中牌型特效后，当前突发样本会作废，动画消退后至少经过 450ms 静止窗口才会重新读牌；延迟激活的新玩家区域即使牌面已经静止，也会立即进入采样。

Reducer 一旦从已确认动作推导出轮到自己，就会异步计算当前策略。时间线只记录已完成的“模型建议”或计算失败，不再显示短暂的“待确认/正在计算”中间状态；同一请求即使从后台完成变为当前可见也只展示一次。用户日志只突出“建议：不出”或“建议：出牌 · 牌型”、牌面和耗时，不显示内部请求编号；`visible` 与请求编号仍保留给内部旁证与审计。旧状态版本的结果会记录为 `advice_stale`，不会覆盖当前建议；任何包含当前手牌中已不存在牌张的建议都会在展示前被拦截并记录失败。

“对局动态”是可鼠标选择和复制的统一时间线：一条出牌事件只占一个时间线条目，牌面复用统一的花色卡片样式，而不是显示 `2D` 一类内部码；不出、回合、接风、头游/二游/三游/末游、模型建议和识别异常以不同颜色显示。`?` 只表示该张花色被遮挡，仍计入张数和牌型；悬停可查看候选花色。后续完整稳定牌面与正式动作不同时，统一生成 `event_correction` 并原位替换原动作，可修正点数、花色、张数与牌型；无法通过规则及后续历史重放的候选不会阻断整局。明确识别的牌仍严格受双副牌上限约束。固定的头游/二游/三游标识是最后一手无法读清时的低影响兜底：模板置信度必须至少为 0.90，且只能按头游 → 二游 → 三游推进；当前行动者还会等待六个稳定帧，让正常的最后一手优先提交。兜底只把该玩家标为出完并把剩余数设为 0，不伪造牌面。第三名出现后四个名次已确定，状态机不再虚构第四人的回合、接风或策略请求，只等待结算页的“再来一局/换桌”按钮自动封存。旧的“开始实时对局”和“最近动作快速纠错”控件已移除；历史 session、录像和日志不会因开始下一局而被清空。

## 对局日志、录像和异常包

### 开局与存储保护（2026-09）

- 可靠“要不起”按钮在两个独立新鲜采集时刻确认后可给出本地“不出”提示，不需要为该提示调用模型。即使正式历史正在等待补正，独立己方回合证据与合法按钮仍可提供这条规则提示；它不提交PASS、不推进历史、不解除模型安全门控。已经提交的模型任务不保证从未运行，诊断会区分这两者。
- 浮窗显示实际开局阶段；重连完成后继续显示手牌/首出确认进度，而不是永久停留在重连成功文案。过期候选不会清除本阶段已看见的动作或已完成事实。
- 明确大厅/结算页面停用开局异常计时和持续媒体写入。结算后封局、自动导出只执行一次，并继续监听下一局。
- 开局诊断默认每事故最多3帧/16MiB，每运行64MiB媒体、8MiB文本，全部运行512MiB媒体、64MiB开局文本；饱和时拒绝新增，不自动删除旧证据。配额扫描超时同样保守停用新增诊断，不能阻塞推荐或关闭窗口。
- 启动输出、异常日志、启动事件各保留一个1MiB当前段和一个旧段。原生崩溃追踪保留独立文件句柄，不参与普通日志轮转。
- 当前profile的session媒体总额度默认20GiB，也可在实时助手 UI 的“录像总容量”中调整；对应 `profile.json` 参数为正整数 `recording_max_total_bytes`。达到限额后停止新增录像、保留已录内容、标记PARTIAL并在 UI 中告警；识别和推荐继续，不自动删除旧录像。不同程序实例的录像预算不共享，勿将此视为多实例全局配额。
- `Clean-DaguandanDiagnostics.ps1`默认仅预览。完全退出助手并备份需分析事故后，加`-Apply`清理旧完成事故的frames/roi媒体；每次运行保留最新3个完整事故，保留文本、配置、模型、对局录像、ZIP以及未完成事故。删除不进入回收站，已清理事故不再具有完整图像证据。

### 回合响应与极简显示

- 正常座位绑定PASS优先走直接交接，不把“左家不出→己方”一律当快速历史事故。旧徽章、非法首出PASS、跨圈或没有独立采集时间的证据仍然拒绝。
- 己方刚出牌/不出后残留的按钮不代表又轮到己方。其他玩家的正常行动窗口只在同一响应身份内先确认按钮消失、再由两张新采集确认按钮重新出现时才进入恢复；不会因旧按钮或计时器抖动提前启动3秒证据过期。
- 局部历史恢复在可信己方机会出现后有2秒处理时间预算，缺证据时给出具体阻断原因，不再把8秒当只记日志的提示后继续重试几十秒。自动监听与独立规则提示仍可工作，但不得把后续轮次新牌填入旧回合。
- 普通高可信已确认动作不逐次阻塞复核；高风险前手复核保留1.2秒截止及修正版本隔离。用户执行后的旧建议不能因复核回调再次显示。
- 极简窗口默认只显示当前有效推荐牌型和牌、或“不出”，以及必要的一条错误。内部确认步骤、Q值、引擎耗时、导出路径和成功预选细节留在完整助手/诊断。旧局导出、旧采集代次及过期更新不能替换新建议。
- 四座位表面基线使用短时内存证据，最多4帧/16MiB，最长1.5秒；已消费动作设座位边界，不能借更早的画面把旧牌误认为新牌。不为此新增持续磁盘录像。
- 初始“不出”徽章不是永久否决条件：同一窗口后来确证消失并重新出现时可重新采用。有限短链可包含多位连续合法出牌，但每位非PASS都要有自己的新画面证据；己方明确交接时也保留有界语义读牌，避免低运动分数让实际新牌一直得不到读取。
- 名次徽章不提供牌面证据，也不能直接把仍有牌的座位清零。终局若还有相邻最后动作，程序先在2秒有界窗口内以两张不同采集、逐座位新ROI和合法顺序原子确认动作，再把名次事件绑定到最后动作；证据不足则保留缺口而不猜牌。
- 未分类、低置信度非空牌面、变化中的区域或识别异常一律是“不确定”，不能因为缺少合法候选就推成PASS。确定性安全回归覆盖了可见金色大王曾被误补“不出”的情况。
- 王的特殊皮肤可以在原有整牌候选、同标签模板和颜色门都成立时，用完整纵向JOKER字列补充确认；只在候选附近搜索，不使用边框相似度，不新增文字-only牌候选，不降低动作阈值。金色大王/蓝色小王已加入真实录像回归，但不等于所有皮肤与缩放均已覆盖。
- 处理管线使用真实单调时钟分段计时，摘要随manifest导出。阶段数、计数器、近期样本均有上限，近期百分位不代表整局或远程性能。录像零丢帧和模型ready不等同每轮推荐已在窗口显示。
- 正式JSONL仍保持同步持久化；派生时间线Markdown改为访问/封存时生成。采集按绝对节拍调度，但配置10fps不保证任意机器都达到10fps。录像尾帧完整性仍是正式分析准入的安全条件。

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\Clean-DaguandanDiagnostics.ps1
# 确认预览后才执行（不删除用户对局录像）：
powershell -NoProfile -ExecutionPolicy Bypass -File .\Clean-DaguandanDiagnostics.ps1 -Apply
```

当保存方式不是“不保存”时，每段录像的数据会物理隔离在当前 profile 下；
源码版位于仓库 `data/`，发布版位于上述用户数据 generation：

```text
data/profiles/tencent_daguandan/sessions/game_YYYYMMDD_HHMMSS_<id>/
  manifest.json
  timeline.jsonl
  timeline.md
  observations.jsonl.gz
  advice.jsonl
  video/game.avi
  video/frame_index.jsonl
  incidents/INC-0001/
```

- `timeline.jsonl` 是不可变事实源；`timeline.md` 是便于人和大模型阅读的中文投影。
- `observations.jsonl.gz` 保存候选、置信度、采用/拒绝原因和帧索引。
- `advice.jsonl` 保存完整 `engine_input`、建议状态和分段耗时。
- `frame_index.jsonl` 保存每个视频帧的原始单调时钟，不依赖播放器固定 FPS 推断业务时间。
- `manifest.json` 保存配置/模板哈希、帧数、丢帧数和本局性能指标。

候选冲突、超时、非法牌组、采集/编码中断和策略异常会生成 `incidents/INC-*`。事故目录包含状态前后、相关观察、关键帧、短视频、联系表、`media.json` 和自动生成的 `llm_report.md`；媒体编码失败时会保留 `media_error.json`，但不会阻止主录像释放和对局清单封存。相同录像警告会在五秒窗口内合并到同一事故包，避免重复复制整段环形缓冲。排查时优先发送 `llm_report.md` 与 `contact_sheet.png`；需要分析动画时再附 `clip.avi`。也可以在“对局回放”页点击“导出大模型诊断包”。

## 识别监听核心一键回归

修改识别监听逻辑后，不需要先新开一局。使用已人工核对的 `verified` TruthLog 会话运行：

```powershell
.\.venv\Scripts\python.exe scripts\run_listener_regression.py
```

默认会读取项目内的：

- `data\profiles\tencent_daguandan\sessions`；
- `data\profiles\tencent_daguandan`；
- 20 局 `label_status=verified` 的历史对局。

命令运行期间会在控制台显示单行进度条，包含总体百分比、当前 session、视觉监听帧进度和 FableDan 推荐进度；长时间没有动作提交时也能确认程序仍在运行。

该命令的主回归链路是：

```text
历史 AVI
  -> ScreenshotRecognitionService / 多牌桌锚点门禁
  -> LiveV2SessionRuntime / LiveV2 Vision Runtime
  -> ProductionRuleSession / LiveReducer
  -> TrustedGameSnapshot 直接投影到 FableDan
  -> TruthLog 比较与首个分歧证据
```

视觉回放使用当前识别监听核心；人工核对的 TruthLog 只作为期望结果基线，不替代视觉识别。报告同时包含：

牌桌页面门禁分别匹配右上角“规则”`table_anchor_1`、左上角“更多”`table_anchor_2` 和游戏标识 `game_logo_anchor`；任一固定 ROI 的匹配分数达到 `0.85` 即可确认牌桌，结算按钮仍具有更高优先级。回归报告会同时记录三个独立分数。FableDan Worker 不再创建第二个 Reducer 重放同一份历史，而是直接消费规则层已经校验的 `TrustedGameSnapshot`，避免未知花色选项在两套投影中被不同标准化。

- 视觉监听驱动的 FableDan 结果；
- 完整 TruthLog 驱动的 FableDan 隔离结果；
- 每局的动作缺失/新增/变化；
- 首个分歧帧、状态和截图证据；
- profile、模板和模型信息。

单独测试一局：

```powershell
.\.venv\Scripts\python.exe scripts\run_listener_regression.py `
  --session game_20260829_114157_a940ae
```

随机抽取 N 局（不放回）：

```powershell
.\.venv\Scripts\python.exe scripts\run_listener_regression.py `
  --random-count 3 `
  --seed 20260912
```

一次指定多个 session：

```powershell
.\.venv\Scripts\python.exe scripts\run_listener_regression.py `
  --session game_20260821_214641_6eed78 game_20260822_142942_e356d4 game_20260829_114157_a940ae
```

也可以重复使用 `--session`；两种写法会合并去重。多个 session 默认以 3 个有界线程并发处理，
每局仍然逐帧读取，不会预加载全部视频。使用 `--workers 1` 可退回串行，`--workers 2` 或 `--workers 3`
用于限制同时占用的识别、FableDan、OpenCV 和临时运行资源。报告会按命令行选择顺序合并生成，
不会因为并发完成顺序变化而打乱 session 顺序。

`--random-count 3` 表示从当前候选 verified session 中抽取 3 局；`--seed` 用于复现同一批抽样结果。
如果不指定 `--seed`，脚本会自动生成随机种子，并在控制台和报告中记录，后续可用该种子重跑。
`--session` 会先按指定 ID/路径过滤，随后 `--random-count` 可以在过滤结果中继续抽样。`--workers` 只控制 session 级并发，
默认值为 3，允许范围为 1～3；它不会把单个 session 的帧拆开并发，因此仍保持生产监听链路的逐帧顺序。

报告会写到 `reports\listener-core-regression\<run_id>\`，优先查看：

- `00_llm_summary.md`：大模型首要入口，包含结论、首个失败和读取顺序；
- `01_run_summary.json`：完整机器可读运行汇总；
- `02_artifact_manifest.json`：每个产物的作用、优先级和读取建议；
- `03_failures.json`：机器可读失败列表；
- `listener_core_regression.md`：兼容保留的简明结论；
- `listener_core_regression.json`：兼容保留的机器可读结论；
- `all_session_audit.json`：完整逐局审计；
- `sessions\...\first_divergence`：首个错误的证据。

默认只要 verified 会话出现监听、状态机或 FableDan 阻断错误，命令就返回非零退出码。draft 和无 TruthLog 会话可用 `--include-draft`、`--include-no-truth` 加入参考诊断，但不会把未确认基线当作严格真值。

会话测试工作台已移除，避免维护一条不经过实时监听核心的重复测试入口；历史回放和核心回归统一由命令行完成，对局回放页仍用于播放、查看帧和人工核对。

TruthLog 编辑器点击“保存 TruthLog”时还会执行双副牌物理牌库校验：同一张“点数+花色”（包括小王/大王）最多 2 张、同一点数最多 8 张、同一花色最多 26 张，并检查自己打出的牌必须能由初始手牌提供。校验失败会显示具体牌面和涉及的动作编号，**不会写入 `truth_log.json`，也不会通过“强制保存草稿”绕过**。已有历史文件仍可打开进行修正，校验只在保存/发布时阻断。

## 复现和回归测试

“对局回放”提供两条独立路径：

- “确定性状态重放”直接重放已保存事件，用于严格验证状态机、纠错和建议请求键；相同输入必须产生相同状态哈希。
- “重新视觉识别”按 `frame_index.jsonl` 的原始时间戳解码录像，把每一帧重新送入当前的特效门控、动态沉降、多帧共识和 Reducer 完整实时管线，再按 `turn_id` 与原时间线比较，输出 `visual_replay.jsonl` 和 `visual_replay_comparison.json`。选择事故后会直接跳到事故触发时间。

修复问题后，先在原事故目录的无损 PNG/ROI 上做精确测试，再对整局录像运行视觉复测，最后运行全部自动化测试。仓库内的 `tests/fixtures/live_sessions/golden_observations.json` 包含一段带“短暂假稳定 → 特效再次出现 → 最终稳定”的黄金对局，保证修复不会重新引入特效期间误提交。真实桌面帧率、识别 P95 和模型显示延迟仍应通过实际录像校准；在真实样本达到目标前，实时功能应视为实验性半自动功能。
