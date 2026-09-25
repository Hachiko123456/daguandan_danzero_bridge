# 大掼蛋助手：换机「静默卡死」定位 + 五问复核

> 评估日期：2026-09-22
> 评估对象：`daguandan_danzero_bridge` / profile `tencent_daguandan`（基准 1280×720，目标客户区 1280×764）
> 证据来源：`opening_gate.py`、`gui/live_controller.py`、`window_capture.py`、`image_io.py`、
> `capture_service.py`、`dpi.py`、`doctor.py`、`runtime_layout.py`、`profile.json`、`.gitignore`
> 前序文档：`docs/analysis_occlusion_and_portability.md`（问题 1–4 的实证部分，本文复核并收敛）

---

## 0. 结论摘要

1. **问题 5 不是"环境没装好"，而是链路上有 7 段 fail-closed 门禁，失败时只改浮窗文案、不产生任何错误。**
   换机后流程"走不下去"是这套设计的必然表现，不是偶发故障。
2. 其中 **4 段是机器耦合的**（窗口几何 / DPI / 采集后端 / 资产位置），所以同一份代码在 A 机能跑、
   在 B 机静默不动。
3. **`doctor.py` 的 `window_probe` / `capture_probe` / `recognition_probe` 全部是 `False`** ——
   现有自检完全不碰"真窗口 + 真采集 + 真识别"，因此换机失败**在原理上不可能被自检发现**。
   这是本次最值得补的一环。
4. **问题 3 的前提需要先纠正**：窗口尺寸不是"改不了"，而是**程序启动时会主动用 `SetWindowPos`
   把目标客户区强行改成 1280×764，改不成就直接拒绝启动**。这是监听的硬前置条件，不是外部约束。
5. 问题 1、2、4 沿用前序文档的实证结论，本文只做收敛和补强（第 3–5 节）。

---

## 1. 问题 5：换机后「不报错，但流程走不下去」

### 1.1 机制：整条链路是 fail-closed 的，而且没有告警通道

实时链路是：

```
定位目标窗口 → 锁定客户区 1280×764 → 采集一帧 → 标准化到 1280×720
→ 牌桌锚点门禁 → 开局门禁（6 条）→ 状态机 / Reducer → FableDan → 浮窗
```

- 第 1、2 段失败会**抛异常并弹出文案**（`live_controller.py:461-468` 捕获后 `error.emit`）。
- 第 5、6 段失败**只是把浮窗文案换掉**，`_publish_opening_status()`
  （`gui/live_controller.py:877-899`）把内部 `phase` 映射成一句中文提示，
  **不弹窗、不写 error、不进 incidents**。
- 第 7 段没通过，就没有建议 —— 用户看到的现象就是"程序开着、浮窗也在，但永远不给建议"。

`OpeningTracker.observe()`（`opening_gate.py:163-296`）在失败路径上全部是
`return OpeningGateEvaluation(False, reason, None, None)`，`reason` 只进 UI 文案和日志字段，
**没有任何一条会升级成用户可见的错误**。

### 1.2 定位器：浮窗文案 → 卡在哪一步

**这是最省事的排查方式** —— 换机后看浮窗上那行字，直接对应故障点：

| 浮窗文案 | 内部 reason | 实际含义 |
| --- | --- | --- |
| 已连接，等待进入牌桌 | `unknown` / `lobby` / `waiting_table` | **牌桌锚点没达到 0.85**，页面门禁一直没开 → 几何/识别问题 |
| 正在确认当前级牌 | `round_level_unresolved` | 级牌区域识别不到 |
| 正在确认起手牌，已识别 N 张 | `hand_count_mismatch` | 手牌读到的不是 27 张（N 就是实际读到的数，最有信息量） |
| 起手牌存在未确认花色，正在等待清晰画面 | `hand_unresolved` | 手牌里有 `?` → **只要有一张，就永远开不了局** |
| 起手牌识别有冲突，正在重新确认 | `hand_invalid` | 手牌不满足双副牌物理约束 |
| 已识别 N 张，等待首出确认 | `confirming_opening` / `opening_seed_invalid` | 首出动作读不到，或 `actor != lead` / 下一家顺序不合法 |
| 错过完整开局，当前 N 张，本局暂无法推荐 | `missed_opening` | 8 秒预算内没凑齐确认，本局放弃 |
| 完整开局已确认，正在建立对局 | `ready` | 正常 |

> 关键：**上面每一行都是"没报错"的状态。** 用户"换了电脑之后不报错但走不下去"，
> 必然是停在其中一行上。

### 1.3 换机后为什么会踩中这些门禁：按可能性排序

#### M1（最高）DPI 感知未生效 → 采集几何整体缩放

`dpi.py::enable_windows_dpi_awareness()` 是"尽力而为"的：依次尝试
`SetProcessDpiAwarenessContext` → `SetProcessDpiAwareness(2)` → `SetProcessDPIAware()`，
**全部失败时只返回 `success=False`，不阻断启动**（`dpi.py:36-79`）。

如果 B 机上这一步没拿到 Per-Monitor V2（组策略、以非标准方式启动、宿主环境限制等）：

- `GetClientRect` 返回的是**被虚拟化的逻辑坐标**，`SetWindowPos` 也按逻辑像素换算；
- 回读校验 `resized == (1280, 764)` **仍然会通过**（因为它校验的也是逻辑值）；
- 但 `PrintWindow` 拿到的位图内容是按系统 DPI 缩放后的渲染结果；
- 结果：**模板匹配的尺度整体偏离**，而 `match_settings.scales` 只覆盖 `0.9 / 0.95 / 1.0 / 1.05 / 1.1`
  （`profile.json:28-34`），**125% 需要 1.25、150% 需要 1.5，全在搜索范围之外**。

→ 所有模板命中率崩塌 → 牌桌锚点 < 0.85 → 浮窗永远显示「已连接，等待进入牌桌」。

这条能解释"为什么 A 机好、B 机全废"，而且**完全没有任何报错**。
A 机很可能是 100% 缩放，所以问题从来没暴露过。

> 需实测确认，见第 7 节。

#### M2（高）客户区最终不是 1280×764，或 44px 顶部留白假设失效

几何链条是三层叠加：

```
base_size = 1280×720
viewport_mode = bottom_aspect, aspect=16/9
target_client_size = 1280×764
```

`calculate_viewport_box`（`image_io.py:64-88`）：源 1280×764 的宽高比 1.675 < 1.778，
所以取 `Box(0, 764-720=44, 1280, 720)` —— **固定丢掉顶部 44 像素**，认为游戏画面在这 44px 之下。

在理想几何下 `scale = min(1280/1280, 720/720) = 1.0`，**不做任何重采样**，帧与模板逐像素对齐。
但只要偏离这个理想态，就会出事：

- 若客户区不是 1280×764（显示器比 1280 窄、小程序有最小尺寸限制、多屏缩放不同）→
  `SetWindowPos` 回读失败 → 第 2 段报错（这是**唯一会被明确报出来**的一种）；
- 若客户区是 1280×764 但**顶部留白不是 44px**（微信版本不同、小程序标题栏高度变了）→
  裁切窗口整体上下偏移 → 所有 ROI 相对画面错位 → 锚点分数掉下来；
- `standardize_to_base`（`image_io.py:135-157`）在宽高比不等于 16:9 时走的是
  **等比缩放 + 居中补边**，内容会整体位移，而 ROI 是按 `base_size` 绝对坐标定义的 → 全部错位；
- 而且 `aspect_compatible`（`image_io.py:166`）**只被记录，从未被用来拦截**
  （全仓只有 `opening_evidence.py:2014` 和 `window_e2e_validation.py:1693` 两处读取）；
- `profile.json:20` 是 `detect_black_bars: false`，所以 `image_io.py:120-127` 的
  `detect_content_viewport` **自适应裁剪根本没启用** —— 全靠固定 44px。

→ 静默表现为锚点门禁不过。

#### M3（中）定位到了"错的"那个窗口

`find_target_window`（`window_capture.py:82-121`）用 `EnumWindows` 枚举**顶层可见窗口**，
按 `window_title_keywords = ["大掼蛋（腾讯）"]` 做**精确优先 + 包含匹配**。

- 一个都匹配不到 → 抛 `WINDOW-NOT-FOUND`（硬报错，会看到）；
- 匹配到多个 → 抛 `WINDOW-AMBIGUOUS`（硬报错，会看到）；
- **但只匹配到一个"错的"窗口时，一切静默**：尺寸锁定会"成功"，
  改的是微信容器窗口，而不是游戏画面所在的那个窗口 → 内容与基准完全不同 → 锚点全灭。

换机后若微信版本不同、小程序改为内嵌在微信主窗口里（不再是独立的 `WeChatAppEx` 顶层窗口），
就会落到这一档。**这是"没有报错但完全不工作"的第二种典型成因。**

#### M4（中）时序预算在慢机器上不够

`OpeningTracker`（`opening_gate.py:117-296`）：

- `max_age_ms = 8000`：开始计时后 **8 秒**内必须完成确认，否则 `discard_candidates()` 重来；
- 需要 `hand_count >= 2` **且** `candidate_count >= 2`，即**至少两次独立采集**的同一语义确认；
- `observation_id` 重复的帧直接判 `duplicate_frame` 丢弃；
- 叠加 `match_settings.stable_frames = 3`（需要连续 3 帧稳定）。

换到性能更弱、或识别单帧更慢的机器上（DPI 更高 → 单帧更大 → 匹配更慢），
采样节拍跟不上，**8 秒内凑不齐 2 次确认 → 反复丢弃 → 永远停在"正在确认起手牌"/"等待首出确认"**。
这条与 M1/M2 会互相放大。

#### M5（中）资产与数据根不一致

- 源码运行读**仓库内 `data/`**；打包版读
  `%LOCALAPPDATA%\DaguandanAssistant\data\v1\generations\<build-id>\data\`
  （`runtime_layout.py:33` 定义 `DAGUANDAN_DATA_ROOT`，`:358-361` 回退 `LOCALAPPDATA`）。
- 好消息：`.gitignore:39` 明确写了 "Keep profile templates/configuration/model assets tracked"，
  模板目录和 `models/best.npz` 都是**入库的**（`.gitignore:74-75` 有一行
  `!data/profiles/*/models/best.npz` 的例外），所以重新 clone 不会丢模板。
- 风险在两处：
  1. 打包版换机器 / 换 build_id → 会**从 EXE 里的种子重新播种一份干净数据**，
     A 机手工调过的模板和配置不会跟过来；
  2. `data/profiles/*/hand_template_calibration.json` 在 `.gitignore:48` 被忽略 ——
     但这条**已经作废**（`portable_data_migration.py:456-458` 注明它是"已移除的自动校准缓存"，
     迁移时主动跳过），可以排除，不是元凶。

→ 若模板/模型缺失，`doctor.py:287-317` 的 `RESOURCE-TEMPLATES-JSON` 和
`models/best.npz` 检查会 FAIL。**先跑一次 doctor 就能排除这一档。**

#### M6（低，但有明确提示）采集后端回退后被自身浮窗遮挡

微信小程序是 GPU/WebGL 渲染，`PrintWindow` 可能返回黑屏并触发
`CAPTURE-BLACK-FRAME` 拒帧（`window_capture.py:238-239`，处理是正确的）。
回退到 `screen` / `gdi_screen` 后，只要检测到有窗口覆盖牌桌客户区，
**当前帧会在录像、识别、策略之前被拒绝，流程暂停**。

这一档有明确提示文案，且 `background` 下与换机无关。建议长期补 WGC
（Windows Graphics Capture）后端，并把 `find_screen_occluders`（`window_capture.py:253+`）
的遮挡判定从"任何顶层窗口"细化为"实际覆盖牌桌客户区的窗口"。

### 1.4 为什么现有自检发现不了

`doctor.py:351-362` 里 capabilities 是硬编码的：

```python
"window_probe": False,
"capture_probe": False,
"recognition_probe": False,
```

也就是说：**doctor 只检查依赖、平台、存储可写、构建完整性、模板文件存在性、模型文件存在性**，
它**不会去定位真窗口、不会真的截一帧、不会真的跑一次识别**。

于是"资产在不在"能查出来，**"资产在这台机器上匹不匹配得上"完全查不出来** ——
而这恰恰是换机失败的全部内容。

### 1.5 建议：补一个「换机预检」（环境契约）

这是本文最高优先级的建议。目标：**把静默的几何不匹配变成启动前的显式报错。**

建议新增 `scripts/preflight_environment.py`（或在 doctor 里把三个 probe 打开），按顺序做 5 步：

| 步骤 | 动作 | 通过标准 |
| --- | --- | --- |
| 1 | `find_target_window` 并打印 `hwnd` / `title` / `class` / 是否 `WeChatAppEx` | 能定位到**游戏画面所在**的窗口，且 class 符合预期 |
| 2 | 打印 `GetProcessDpiAwareness` 的**实际值**与显示器 DPI、系统缩放 | 进程 = `per_monitor`，且明确记录显示器缩放 |
| 3 | `lock_target_client_size` 后打印**期望 vs 实际**客户区，以及 `standardize_to_base` 返回的 `scale` / `aspect_error` / `padding` | 实际 = 1280×764；`scale == 1.0`；`padding == (0,0,0,0)` |
| 4 | 用当前帧跑 3 个牌桌锚点（`table_anchor_1` / `table_anchor_2` / `game_logo_anchor`），打印各自分数 | 至少一个 ≥ 0.85，且分数有明确余量（建议记录到 0.90 以上） |
| 5 | 用当前帧对全部模板做一次全量自检，按 kind 分组输出置信度分布 | 关键 kind（`anchor` / `rank` / `suit` / `status` / `timer`）的 P50 明显高于 `min_confidence=0.76` |

第 5 步输出应当落盘（例如 `diagnostics/preflight/<timestamp>.json`），
这样"换机前后两张报告"可以直接对比，**一眼看出是哪一步的分数掉了。**

配套的两个小改动（都很小、收益很大）：

1. **把 `target_client_size` / `viewport_mode` / 44px 偏移从隐式常量变成显式可配置 + 启动自检**，
   失败时给出"你的客户区是 1280×724，不是 1280×764，请检查微信版本"这类可操作指引；
2. **把 `aspect_compatible` 从"只记录"升级为"门禁"**：`aspect_error > aspect_tolerance` 时
   明确报错，而不是让它在后面变成锚点分数不够。**这一处改动能把 M2 从静默变成显式。**

### 1.6 不改代码就能做的三步

1. **看浮窗文案**，对照 §1.2 的表，确定卡在哪一步。
2. **跑一次 doctor**（`python -m daguandan_bridge.doctor` 或发布包的诊断入口），
   排除 M5（模板/模型缺失）。
3. **在出问题的机器上确认系统缩放比例**（设置 → 显示 → 缩放）以及"更改高 DPI 设置"的行为。
   如果 A 机是 100%、B 机是 125%/150%，基本就可以锁定 M1。

---

## 2. 问题 3 复核：小程序窗口「尺寸改不了」

**这个前提需要纠正。**

`gui/live_controller.py::start_listening()` 在开始监听前，一定会调用：

```python
lock_client = getattr(self.capture_service, "lock_target_client_size", None)
if callable(lock_client):
    try:
        lock_client(self.profile_name)          # → window_capture.resize_target_client
    except Exception as exc:
        self.opening_evidence.observe_failure(exc, stage="window")
        self.error.emit(f"无法锁定牌桌客户区尺寸：{exc}")
        return False                            # 监听根本起不来
```

（`live_controller.py:461-468`）

而 `resize_target_client`（`window_capture.py:154-216`）做的事是：
`ShowWindow(SW_RESTORE)` → 测出窗口边框内缩量 → `SetWindowPos` →
**回读客户区并校验是否等于 1280×764，不等就抛错**。

所以：

- **如果你在 A 机能正常起监听** → 说明 `SetWindowPos` 对微信小程序窗口是生效的，
  "窗口大小没办法改变"这件事其实不存在，A 机的窗口已经被程序改成 1280×764 了。
- **如果你在 B 机起不来** → 就是这里挡住的，报错文案是
  「无法锁定牌桌客户区尺寸：目标窗口未接受请求尺寸：期望 1280x764，实际 …」。

真正与"小程序"相关的坑不是尺寸，而是：

1. **后台截图黑屏**（GPU/WebGL 渲染）。项目已正确使用
   `PrintWindow + PW_RENDERFULLCONTENT (0x1|0x2)`，并对全黑帧显式拒绝
   （`window_capture.py:238-239`）—— **这两点做对了。**
2. **窗口是否还是独立顶层窗口**。微信改版后若小程序内嵌进主窗口，
   `EnumWindows` 就找不到了（见 §1.3 M3）。

**结论：不要再纠结"能不能改窗口大小"，要盯的是"改完之后实际拿到的是不是 1280×764、
以及内容偏移是不是和模板标定时一致"。** 后者才是换机失败的真实战场。

---

## 3. 问题 1 复核：左家 ≥5 张时花色被遮挡

前序文档已用 30 局 / 725 手真实数据把这件事钉死了，这里只收敛结论：

- 这是**左家专属 + ≥5 张**的几何耦合，不是"牌多就遮挡"：左家 5 张 27/57 = 47.4% 丢花色，
  而右家 5 张 0/40、右家 6 张 0/9、自己 5–6 张 0/40。**右家同样张数一次都不丢。**
- 遮挡源是**游戏自身的 UI 图层**（「不出 / 提示」按钮、座位等级徽章、牌型特效）压在左家牌列右下方。
  配置层印证：`left_play`(x135..485) 与 `button_actions`(x450..899) 在 x450..485 重叠；
  `passed_left`(x138..257) 整个落在 `left_play` 内。
- **点数在左上角始终外露**，所以是 `5?` / `A?` 而不是整张失败。

**处理方案延续前序的四层，但落地顺序建议收敛为：**

1. **先量化损失**（最该先做）：`advice.jsonl` 里已有 `suit_uncertain` / `suit_variant_count` /
   `suit_equivalence_class_count`，统计"有多少条建议因为历史里存在 `?` 而多算了变体、
   或在不同变体间结论不一致"。**如果这个数字很小，这就是体验问题而不是正确性问题**，
   后面三层都不必急。
2. **L1｜把左家出牌区的采样时机前移到"左家自己的回合内"**（性价比最高）：
   左家的牌在他自己回合就已落位，**那一刻「不出 / 提示」按钮还不存在**，是干净帧。
   项目已有 `burst_read` + 特效门控 + 静止窗口这套机制，只要把采样窗口从
   "轮到自己"改挂到"左家 action epoch 开始"即可。**不动识别算法。**
3. **L3｜遮挡感知门控**：项目已有 `button_actions` / `passed_*` 区域和按钮模板，
   读左家出牌区前先判断"按钮是否覆盖左家出牌区右下角"：
   覆盖 → 本帧只作点数证据，改取更早或更晚的帧。这与现有"命中牌型特效 → 丢弃突发样本"
   是同一个模式，属于既有机制的推广。
4. **L2 / L4** 按需：拆「角标区 + 花区」子 ROI；把 `?` 的候选集按**牌型结构**收窄
   （`live/card_uncertainty.py` 已有 `feasible_action_variants` /
   `state_variants_for_unknown_suits`）。

**不建议**：改 `left_play` 位置/大小（盖住的像素变不回来）、缩放窗口（几何等比变化，相对遮挡不变，
且见 §2 本来就动不了）、上 OCR（花色是图形不是文字）。

**注意一个与问题 5 的耦合点**：开局门禁的第 4 条是「手牌不含 `?`」（`opening_gate.py:325-326`），
**手牌里只要有一张花色被遮挡，开局就永远开不了**。左家遮挡问题不影响开局，
但**自己手牌**的遮挡会直接把整局卡死在门口 —— 排查换机问题时值得一并确认。

---

## 4. 问题 2 复核：同类开源项目与可抄的做法

### 4.1 同场景（小程序 + 模板匹配 + 后台截图）

| 项目 | 做法 | 对你的价值 |
| --- | --- | --- |
| `chushenshen/happy-fight-opencv`（GitHub + Gitee `css_code/happy-fight-opencv`） | 欢乐斗地主记牌器。`win32gui` 取窗口句柄 → 后台截屏 → `cv2` 模板匹配 → 识别开局/自家手牌/底牌/上下家出牌/剩余牌 | **架构和你几乎同源**。但它只有"每屏全量匹配"一条链路：无状态机、无证据链、无回放对账、无物理约束。**稳定来自"简单"，代价是错一次就一直错** |
| `FlysonBot/poker-counter`（中游斗地主） | 首次启动显示粉色叠加层让你拖区域；**每局开始时按手牌高度自动校准模板缩放比例，无需手动配置即可适配不同分辨率**；`config.yaml` 可自定义阈值 | ⭐ **这是本次最值得抄的一条**，直接对应你的问题 5：它不信任"固定模板 + 固定阈值"，而是**每局开局先自动校准一次尺度**。你现在的 `match_settings.scales` 是固定 5 档，**没有运行时校准** |
| `lemonpopdo/jj_card_tracker`（JJ 斗地主） | 模板匹配；**牌型校验 + 不合法的尝试纠正**；识别对手剩余牌数变化；**手动修正（左键 −1 / 右键 +1）+ 日志保存** | 与你的思路一致（你也做了牌型校验和原位修正）。它多一条：**给用户一个"就地人工纠正"的入口**，比自己重新开局便宜得多 |
| `tianqiraf` 的 `DouZero_For_HappyDouDiZhu` | PyQt5 + `PyAutoGUI` 截图 + 模板匹配 + DouZero 出建议 | **它的已知 bug 清单最值得看**：飞机带两对换行漏识别、**特效播放期间截图误判**、大小王无法区分、对家的王无法识别、**识别失败直接闪退**。你的 README 几乎逐条在解这些问题 |
| `wang11wei/FightLandlord` | 试过模板匹配 / SIFT / `pyautogui.locateAll()` | 反面教训，它的 README 结尾就一句：**「不同的分辨率要调整」** —— 和你现在的换机问题同款，且它没解 |
| `xiaooo-jian/Digital-image-processing-Final-Work` | 轮廓提取 + **腾讯云 OCR** | **反面教材**：把 OCR 放云端 → 网络抖动即失败、闪退。你选纯本地模板匹配是对的 |

### 4.2 同游戏（掼蛋 AI 引擎，解决"决策"不是"识别"）

- `AltmanD/guandan_mcc`（DanZero 原版，论文 2210.17087）
- `rezeu/GuandanProj`（基于 rlcard 魔改）
- `Mereithhh/guandan-lab`（在线大厅 + LLM Agent 的陪练训练器）
- `ablohui/ADAN`（首届人工智能掼蛋算法大赛第一名代码：理牌 / 主动出牌 / 被动出牌 / 进贡还贡四模块）
- 金山 `Calix-L/DanKS`：把掼蛋 AI 三代技术路线完整开源

你的 `danzero/_vendor/guandan_rlcard` 已经是 rlcard 的 guandan 分支。
**可借鉴的是 DanKS 的两段式设计**：先做有预算的结构化候选检索，
再让策略网络对少量候选做长期价值判断。你现在是把 `GuanDanState` 直接丢给 FableDan 全量打分，
改成"本地先枚举候选、模型只打分少量候选"会更快、更可解释。

### 4.3 视觉方案升级路线

- `pjgranieri/Portable-Real-Time-Poker-Assistant`：4 个 YOLOv8-nano 分别做牌/筹码/动作检测，
  多阈值置信度融合 + 自适应 ROI 裁剪 + 帧间 ROI 切换。**"模板匹配 → 检测模型"的范式**，
  对遮挡最鲁棒。你已有 30 局录像 + 104 模板 + `label_status=verified` 的 TruthLog，
  `application/dataset.py` 已在按 ROI 导出样本，**这条路的启动成本比一般项目低**。
- `newMeta98/ace-autopilot-poker-ai`：Moondream-2B 视觉模型 + DeepSeek 决策。
  单张推理 300ms+，对"轮到自己后几秒内出建议"太慢，只适合离线复盘。

### 4.4 别人怎么做得稳 vs 你已经做对的

**值得直接抄的 5 条工程习惯：**

1. **观测与决策分离**：识别层只产出"候选 + 置信度 + 拒绝理由"，状态层只做事务提交。
2. **一封不可变事实源 + 可重放**：出问题能定位，修完能验证。
3. **帧证据 + 多帧共识 + 时序门控**：两帧一致才提交；特效期间丢帧并等静止窗；
   按钮消失/重现用边沿检测而不是状态检查。
4. **物理约束兜底**：双副牌上限、同点数 ≤8、同花色 ≤26、自己打出的牌必须来自已知手牌。
5. **失败可恢复而不是重开**：出错后从原动作位置原子重放，`action_id` 不变。

**你已经做到、开源同类基本没做到的：**
`timeline.jsonl` 不可变事实源、`observations.jsonl.gz` 保留全部候选与拒绝理由、
`frame_index.jsonl` 用真实单调时钟而非播放器 FPS、`visual_replay` 按原始时间戳整局复测、
`incidents/` 自动生成 `llm_report.md` + 联系表、TruthLog 保存时的物理牌库校验、
`event_correction` 原位修正且不改变 `action_id`。

**在这批同类项目里，你的工程化程度明显高一层。** 唯一的短板不是工程素养，而是
**外部环境契约没有建模**（见第 5 节）。讽刺的是：开源记牌器们因为这个短板而"简单但脆"，
你则是"内部严谨、外部盲区"。

---

## 5. 问题 4 复核：架构是否合理

### 5.1 结论：合理，且工程化程度高；但有结构性盲区

**做得好的：**

- 分层清晰：`domain / application / infrastructure / presentation` +
  `live`（旧，仅服务回放）/ `live_v2`（生产）双代并存；
- `live_v2/architecture.py` 用**白名单强制导入边界**（`MODULE_DEPENDENCY_ALLOWLIST`）
  + **单模块 600 行上限** —— **"用测试守住架构"，这在同类项目里极少见**；
- 模型层 `fabledan` / `danzero` 各自 vendor 上游，边界干净；
- 可移植性在数据层面已经实际验证过：`sessions/game_20260918_*` 的 manifest 记录路径为
  `C:\project\python_project\daguandan_danzero_bridge`，当前工程在 `D:\project\daguandan_danzero_bridge`
  —— **换盘位跑过，session 按 profile 相对存放，没被绝对路径绑死。**

**两个结构风险：**

1. **规模**：`src` 下 234 个 Python 模块（`live` 21 / `live_v2` 32 / `application` 54）。
   旧 `live` 与新 `live_v2` 并存会持续消耗维护成本。
   建议写一份"哪些模块已冻结、只允许回放路径调用"的清单，并在 CI 上加一条断言。
2. **`live_v2/types.py` 被文档标注为"兼容外观层、不是新定义的地方"** ——
   这类"弃用但仍在用"的模块最容易失控，值得加注释 + 工具检查。

### 5.2 真正的架构短板：缺一层「环境契约」

这是本次分析最有价值的发现。

**架构把"内部边界"守得非常好（导入白名单、行数上限、事务化提交、可重放），
但没有把"外部世界的契约"显式建模。**

证据：窗口几何这一个概念，被拆散在**四个地方**，没有任何一处能在启动时声明
"我现在看到的东西，和模板标定时不是一回事"：

| 位置 | 承载了什么 |
| --- | --- |
| `profile.json` | `base_size` / `viewport_mode` / `viewport_aspect_ratio` / `target_client_size` / `aspect_ratio_tolerance` / `match_settings.scales` |
| `window_capture.py` | `SetWindowPos` + 回读校验（只校验"我要的尺寸给了没"） |
| `image_io.py` | 底部 720 的裁切、等比缩放补边、`aspect_compatible`（**只记录不拦截**） |
| `opening_gate.py` | `anchor_score >= 0.85` —— **几何错误的最终下游受害者** |

后果：**几何不匹配不会在几何层失败，而是飘到很远的锚点门禁那里，以"分数差一点"的形式表现，
并且没有任何一条路径把它升级成用户可见的错误。**

建议新增一个显式模块，例如 `environment_contract.py`（或 `live_v2/environment.py`），
职责只有三件事：

1. **声明**当前机器实际拿到的几何（客户区尺寸、DPI awareness、显示器缩放、
   `standardize_to_base` 返回的 `scale` / `padding` / `aspect_error`）；
2. **比对**它与 profile 声称的几何是否一致；
3. **不一致就显式失败**（而不是让锚点分数去承担这个信号）。

配套：`doctor.py:355-357` 的三个 `False` 打开，让自检真正去碰窗口和采集。
这一层补上之后，"换机"就从"玄学"变成"跑一次预检，哪一项红了修哪一项"。

### 5.3 另一个架构层面的观察：fail-closed 与可观测性不平衡

项目把"不确定就不动"（fail-closed）贯彻得非常彻底，这在**正确性**上是对的 ——
宁可不出建议，也不出错误建议。但**可观测性没有跟上**：

- 状态机把所有拒绝理由都算得很好（`observations.jsonl.gz` 里保留全部候选与拒绝理由）；
- 但**面向用户的出口只有一句浮窗文案**，而且这句文案被归类为"正常运行中的状态"，
  不是"错误"。

建议：给"连续 N 秒停留在同一个非 ready 的 opening phase"加一个**软告警**，
把当前的 `phase` + 相关分数（锚点分、手牌张数、级牌）打包成一条可导出的诊断，
而不是让它永远静默。**这不改变任何 fail-closed 语义，只是把"关门"这件事说出来。**

---

## 6. 优先级建议

| 优先级 | 事项 | 解决的 |
| --- | --- | --- |
| **P0** | 换机预检：定位窗口 / DPI / 客户区尺寸 / `scale`+`padding` / 锚点分数 / 全模板自检，输出落盘报告 | 问题 5 全部分支 |
| **P0** | 把 `aspect_compatible` 从"只记录"升级为门禁 | M2 静默 → 显式 |
| **P0** | 打开 `doctor` 的 `window_probe` / `capture_probe` / `recognition_probe` | 让自检能发现换机失败 |
| **P1** | `target_client_size` / 44px 偏移显式化 + 启动自检 + 可操作指引 | M2 / M3 |
| **P1** | 量化 `?` 对建议的实际损失（统计 `advice.jsonl` 的 `suit_*` 字段） | 问题 1 的投入决策 |
| **P1** | L1：左家出牌区采样时机前移到左家回合内 | 问题 1 主要收益，不动识别算法 |
| **P2** | 运行时模板尺度自动校准（抄 `FlysonBot/poker-counter`） | 换机/换版本的根因 |
| **P2** | 非必需重依赖剥离（`torch` / `rlcard` 可选化，只有 DanZero 需要） | 换机安装成本 |
| **P2** | L3：遮挡感知门控（复用现有特效门控模式） | 问题 1 |
| **P2** | 补 WGC 采集后端；细化 `find_screen_occluders` 判定 | M6 |
| **P3** | 用检测模型替代模板匹配 | 问题 1 长期方向 |

---

## 7. 需要你在机器上确认的事（本文的未验证假设）

以下四条我**无法在没有那台机器的前提下验证**，请对照确认：

1. **A 机与 B 机的系统缩放比例分别是多少？**（设置 → 系统 → 显示 → 缩放）
   若 A=100%、B=125%/150%，M1 基本成立。
2. **B 机上浮窗显示的是哪一行文案？** 对照 §1.2 的表可直接定位卡点。
3. **B 机上 `lock_target_client_size` 是通过还是抛错？**
   通过 → 问题在 M1/M2/M3；抛错 → 直接解决问题 3 那一环。
4. **B 机上 `python -m daguandan_bridge.doctor` 的 `RESOURCE-TEMPLATES-JSON` 和
   `models/best.npz` 两项是 PASS 还是 FAIL？** 用来排除 M5。

拿到这四个答案，问题 5 基本可以一次收敛。

---

## 附：本次的一处纠错

`.gitignore:48` 忽略了 `data/profiles/*/hand_template_calibration.json`，
初看像是"每台机器的校准文件没入库 → 换机丢校准"的元凶。
实际核查后排除：这是**已移除的自动校准缓存**，
`portable_data_migration.py:456-458` 明确注明迁移时主动跳过它，
`runtime_layout.py:615` 也只把它当"允许存在的额外文件"。
**不要按这条线索去查。**
