# 换机静默卡死：优化方案（待确认后实施）

> 日期：2026-09-22
> 状态：**方案待确认，尚未改动任何代码**
> 前置文档：`docs/analysis_portability_silent_stall.md`（本文修正其 §1.3 的 M1 判断）

---

## 0. 先纠正上一轮的判断

### 0.1 我上一轮判反了 DPI 的方向

我原先假设「A 机 100% 正常、B 机 125%/150% 失败，因为 `scales` 只覆盖 0.9~1.1」。
你给的数据是 **A=125% 正常、B=100% 失败**，方向正好相反。这条假设作废。

而且它本来就站不住：`dpi.py:36-79` 把进程设为 Per-Monitor V2，
`window_capture.py:198-216` 把客户区强制成 **1280×764 物理像素并回读校验**。
**这套设计的意图就是让采集结果与系统缩放无关** ——
A 机在 125% 下能正常工作，恰好证明这个机制是生效的。

### 0.2 几何不是变量，这一点可以确定为真

`resize_target_client` 的回读校验（`window_capture.py:210-215`）保证了：

- 客户端**要么**正好 1280×764，**要么**直接抛错 `return False`，监听根本起不来；
- 拿到 1280×764 之后，`calculate_viewport_box`（`image_io.py:64-88`）算出
  `Box(0, 44, 1280, 720)`（固定丢顶部 44px）；
- `standardize_to_base`（`image_io.py:135-157`）的 `scale = min(1280/1280, 720/720) = 1.0`，
  `padding = (0,0,0,0)`，`aspect_error = 0` —— **不重采样、不补边、逐像素对齐**。

所以 **"几何偏移导致 ROI 错位"这条路径在数学上不成立**（除非 44px 这个假设本身错了，
但那会让三张锚点一起垮，见下一节）。上一轮报告里的 **M2 基本可以划掉**。

### 0.3 「确认开局中…」这句话是有信息量的，而且信息量很大

来源是 `gui/recommendation_window.py:421-424`：

```python
elif state == "opening":
    phase = str(status.get("phase", "") or "")
    title = "等待开局" if phase in {"unknown", "lobby", "settlement", "waiting_table"} \
            else "确认开局中…"
    self._render_view(CompactViewState("opening", title))
```

调用链（`gui/live_controller.py`）：

| 行号 | 条件 | 发布的 phase | 浮窗标题 |
| --- | --- | --- | --- |
| 771-774 | `not page.allows_media`（没认出牌桌） | `unknown` / `lobby` / `settlement` | **等待开局** |
| 807-814 | 锚点分 < 0.85 | `waiting_table` | **等待开局** |
| 817-825 | 锚点已过，进入开局跟踪 | `evaluation.reason`（开局子阶段） | **确认开局中…** |

**你看到的是「确认开局中…」⇒ 已经在第 817 行那条路径上 ⇒
牌桌锚点门禁已经通过 ⇒ 几何和页面识别都没问题。**

这一步把范围收窄到了**开局门禁的 8 个子状态**里，M1（DPI）、M2（几何偏移）、
M3（窗口定位错）全部可以划掉。

### 0.4 还发现一个直接问题：浮窗把最关键的诊断信息扔掉了

`_publish_opening_status`（`live_controller.py:877-899`）本来已经算好了

```python
{"phase": phase, "reason": phase, "hand_count": count, "generation": ..., "message": "..."}
```

**但 `recommendation_window.py:421-424` 只取了标题，`message` 和 `hand_count` 全被丢弃。**
所以你只能看到"确认开局中…"，看不到是"已识别 26 张"还是"等待首出确认"。

**这是本次整件事里最该先修的一处 —— 它让问题从"可定位"变成"不可定位"。**

---

## 1. 卡点候选：8 个子状态

`opening_gate.py:163-296` / `:317-336`，任一不满足就停在那里、不报错：

| phase | 浮窗文案（现有，被丢弃） | 触发条件 | 换机相关度 |
| --- | --- | --- | --- |
| `round_level_unresolved` | 正在确认当前级牌 | 级牌区域匹配失败 | 高 |
| `hand_count_mismatch` | 正在确认起手牌，已识别 N 张 | 手牌 ≠ 27 张（**N 最有信息量**） | 高 |
| `hand_invalid` | 起手牌识别有冲突 | 手牌违反双副牌物理约束 | 中 |
| `hand_unresolved` | 起手牌存在未确认花色 | 手牌含 `?`（**一张就永久卡死**） | 中 |
| `opening_seed_invalid` | 已识别 N 张，等待首出确认 | 首出动作不合法 | 高 |
| `confirming_hand` | 已识别 N 张，正在确认起手牌 | 8 秒内只读到 1 次 | 中 |
| `confirming_opening` | 已识别 N 张，等待首出确认 | 8 秒内开局确认不足 2 次 | 中 |
| `missed_opening` | 错过完整开局，当前 N 张 | 8 秒预算用尽，本局放弃 | 中 |

**"没有对局数据生成"是这条链路的必然结果**：session 目录只在
`_start_pending_auto_session()`（开局确认后）才创建，开局没过就永远没有 session。

---

## 2. 优化方案

### P0-0｜先做一件 2 分钟、不写代码的事：比对两台机器的资产

**这一步最便宜，而且可能直接终结整个问题。**

在 A 机和 B 机上各跑一次，把输出贴出来对比：

```powershell
# 1. 数据根在哪（源码版 vs 打包版是两条完全不同的路径）
python -c "from daguandan_bridge.runtime_layout import *; print(globals().get('DATA_ROOT_ENV'))"
echo $env:DAGUANDAN_DATA_ROOT
dir "$env:LOCALAPPDATA\DaguandanAssistant\data\v1\generations"

# 2. 资产指纹（模板数量 + 关键文件哈希）
$p = "data\profiles\tencent_daguandan"
(Get-ChildItem "$p\templates" -Recurse -File).Count
Get-FileHash "$p\templates_config.json", "$p\profile.json", "$p\models\best.npz" -Algorithm SHA256
```

**判读：**

- 两份哈希/数量**不同** → 根因就是资产不一致（B 跑的是打包版新播种的干净 generation，
  或拷贝时漏了 `data/`）。方案立刻变成"资产同步"，下面 P0-1 之后的都不用做。
- 两份**完全相同** → 排除资产，问题在"同一段像素在两台机器上渲染不同"，进入 P0 全量诊断。

> 依据：`.gitignore:39` 明确保留模板与模型入库，所以"重新 clone 丢模板"不成立；
> 但打包版会按 `build_id` 从 EXE 种子重新播种（`runtime_layout.py:270-420`），
> 换 build_id 就会换一份数据。

### P0-1｜让浮窗把原因说出来（最小改动，最高收益）

**改动面：** `gui/recommendation_window.py:421-424`（+ 确认 `CompactViewState` 支持第二行 detail）

- `state == "opening"` 时把 `status["message"]` 渲染出来，而不是丢弃；
- 标题改成高区分度短标签，例如：
  `开局 · 手牌 26/27` / `开局 · 等首出` / `开局 · 级牌未知` / `开局 · 手牌含遮挡`；
- 浮窗体量小，建议用「标题 + 一行 detail」两行布局，保持 `500×245` 不变。

**验证：** 在 A 机上人为遮挡手牌一张花色，浮窗应显示「手牌含遮挡」，而不是"确认开局中…"。

### P0-2｜新增换机预检：`scripts/preflight_environment.py`

一次运行、输出一份 JSON（落盘 `diagnostics/preflight/<timestamp>.json`），
**在两台机器上各跑一次，diff 就能定位。** 采集项：

| 组 | 采集内容 |
| --- | --- |
| 窗口 | `hwnd` / `title` / `class` / 是否是独立顶层窗口；`EnumWindows` 全部候选列表 |
| 目标进程 DPI | 目标窗口所属进程的 DPI awareness（`GetWindowDpiAwarenessContext`） |
| 本进程 DPI | `get_windows_dpi_awareness()` 实际值、显示器 DPI、系统缩放 |
| 客户区 | 期望 1280×764 vs 实际；`GetClientRect` 物理值 |
| 标准化 | `standardize_to_base` 的 `scale` / `padding` / `aspect_error` / `content_box` |
| 锚点 | `table_anchor_1` / `table_anchor_2` / `game_logo_anchor` 三个独立分数 |
| 模板自检 | 按 `kind` 分组的置信度分布（P10/P50/P90）+ 低于 `min_confidence=0.76` 的模板清单 |
| 识别 | 手牌张数与 `?` 数量、级牌、当前行动者、按钮 |
| 资产 | 数据根路径、模板数量、`templates_config.json`/`profile.json`/`best.npz` 的 SHA256 |

**关键新增项是"目标进程 DPI awareness"**，理由是下面这条假设：

> **A/B 渲染差异假设（需预检确认）**
> 若 `WeChatAppEx` 本身是 **DPI-unaware**：
> - 在 A 机（125%）：它的 1280×764 物理客户区里，实际渲染是 1024×611 再被 Windows 拉伸 1.25×
>   → 字形偏软；**你的 104 个模板正是在这台机器上裁的。**
> - 在 B 机（100%）：渲染 1:1 原生清晰 → 与模板的相关性下降。
> - 粗锚点（「规则」「更多」这类大块 UI）仍能过 0.85，但**小字形模板
>   （`rank`/`suit`，`min_confidence=0.76`、`min_margin=0.04`）可能集体掉线**
>   → 正好表现为「锚点过了、卡确认开局」。
>
> 这条假设能同时解释：为什么 A 机能跑、为什么几何没问题、为什么卡在开局。

**这是唯一能一次性区分「资产差异 / 渲染差异 / 阈值太紧」的手段**，我认为是 P0 里最该做的。

### P0-3｜软告警：把「关门」说出来

同一非 `ready` 的 opening phase 连续超过 20 秒 → 生成一条可导出诊断
（含 `phase`、`hand_count`、锚点分、当帧截图），而不是让它永远静默。

**不改变任何 fail-closed 语义**，只是补上"可观测性"这一层。

### P1-1｜锚点驱动的几何自校正（替代固定 44px）

不再信任 `image_io.py:64-88` 里那个固定 44px 偏移，改为：
用当前帧实际匹配到的锚点位置，反推内容相对基准画面的**偏移量与尺度**，据此平移/缩放 ROI。

收益：同时覆盖换机、换微信版本、换游戏版本三种情况。

### P1-2｜运行时模板尺度自动校准

参考 `FlysonBot/poker-counter` 的做法：**每局开局时用自己手牌做基准校准**。
具体：手牌区识别出的连续牌列长度 → 反推该次运行的实际牌宽 → 与模板标定牌宽求比值，
把该比值作为本次运行的**首选 scale**（而不是现在固定枚举的 `0.9/0.95/1.0/1.05/1.1`）。

### P1-3｜打开 doctor 的三个 probe

`doctor.py:351-362` 的 `window_probe` / `capture_probe` / `recognition_probe` 目前硬编码 `False`。
打开它们，让自检真正去碰"窗口 + 采集 + 识别"。

### P2-1｜开局门禁的健壮性：手牌含 `?` 不该永久卡死

`opening_gate.py:325-326` 要求手牌 27 张且**不含任何 `?`**。
自己手牌只要一张花色被遮挡，整局就永远开不了 —— 这是**过度保守**，
而且它同时解释了"没有对局数据生成"。

`live_controller.py:827-831` 已经有"手牌暂时遮挡时启动 provisional 监听会话"的路径
（要求两次一致），可以放宽为：**带 unknown 手牌进入 provisional session，
由 `card_uncertainty.py` 现有的变体机制（`state_variants_for_unknown_suits`）承担不确定性**。

### P2-2｜8 秒预算改为帧数预算

`OpeningTracker.max_age_ms = 8000`（`opening_gate.py:117`）是墙钟时间。
慢机器上单帧识别更慢时，8 秒内凑不齐"2 次手牌观测 + 2 次开局确认 + `stable_frames=3`"，
会反复 `discard_candidates()`。建议改为**以采集帧数为预算**（例如 ≥12 帧），
并保留墙钟作为上限。

### P2-3｜其余（与本次卡死无直接关系，按需）

- 问题 1 左家遮挡：**L1**（左家出牌区采样时机前移到左家自己回合内，不动识别算法）
  → **L3**（复用现有 `button_actions` / `passed_*` 做遮挡感知门控）
  → 先做**量化损失**（统计 `advice.jsonl` 的 `suit_*` 字段）
- `torch` / `rlcard` 可选化（只有 DanZero 用；FableDan 走 numpy）
- 补 WGC 采集后端；细化 `find_screen_occluders` 的遮挡判定

---

## 3. 建议的落地批次

| 批次 | 内容 | 改动文件 | 风险 | 可独立验证 |
| --- | --- | --- | --- | --- |
| **批次 0** | P0-0 资产比对 | 无（只跑命令） | 无 | 是 |
| **批次 1** | P0-1 浮窗显示真实原因 + P0-3 软告警 | `gui/recommendation_window.py`、`gui/live_controller.py` | 低（纯展示层） | 是，人为遮挡手牌即可复现 |
| **批次 2** | P0-2 换机预检脚本 | 新增 `scripts/preflight_environment.py` | 低（只读，不改生产链路） | 是 |
| **批次 3** | 按批次 2 的结论做根因修复 | 待定（P1-1 / P1-2 / 资产同步 三选一或组合） | 中 | 是 |
| **批次 4** | P2-1 / P2-2 开局门禁放宽 | `opening_gate.py`、`gui/live_controller.py` | **中高**（碰状态机语义，需跑 `run_listener_regression.py` 全量回归） | 是 |
| **批次 5** | 问题 1（L1/L3）+ 依赖剥离 | 识别层、`requirements` | 中 | 是 |

**建议严格按批次推进：批次 0 → 1 → 2，拿到 B 机的预检报告后，再决定批次 3 怎么做。**
在批次 2 的结论出来之前做批次 3，等于继续猜。

---

## 4. 需要你先确认的三件事

1. **B 机是怎么部署的？** 拷了整个项目目录，还是装了打包后的 EXE？
   （决定资产比对那一步能不能直接定案）
2. **A / B 两机的微信版本、显示器分辨率**各是多少？是否同一个微信版本？
3. **这次先做哪一批？** 我的建议是**批次 0 + 批次 1 + 批次 2** 一起做 ——
   批次 0 是零成本，批次 1 是最小改动且立刻让问题可定位，批次 2 是决定性证据。

---

## 附：如果 B 机不方便跑脚本

退一步的降级路径：批次 0 的两条命令（模板数量 + 三个文件哈希）足以区分
"资产不一致"与"渲染不一致"，占整个判断的八成。先跑这个也行。
