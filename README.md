# 大掼蛋 DanZero 桥接器

这是一个独立的 Windows 11 项目，用于录制腾讯大掼蛋牌桌截图、保存对应模板资产，并通过 Python API 调用本地 DanZero 策略。项目不会对游戏客户端执行点击、按键或自动出牌。

## 安装与启动

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe run.py
```

也可以双击 `start_gui.bat`。在界面中先点击“开始预览”，确认画面后点击“开始本局录制”；录制按设置的间隔保存帧，点击“结束本局”关闭当前会话。

## 区域标注

“区域标注”页会递归读取本项目录制目录中的图片，并在图片左右提供上一张/下一张切换箭头：

```text
data/profiles/tencent_daguandan/screenshots/
```

选择一张录制截图后会立即显示图片预览；点击“查看区域配置”打开独立配置窗口，在表格中多选区域并点击“标注选中区域”可以查看不同颜色的框和中文名称。编辑保存时请只选中一个区域；名称和角色通过中文下拉框选择，`x/y/w/h` 会写回 `regions_config.json`，比例坐标会自动重算。标注页不会打开该目录之外的图片。

在“区域配置”模式下先选择区域名称和角色，再用鼠标左键拖动图片框选，松开后点击“保存区域坐标”即可写回配置；图片预览缩放不会改变原图坐标。区域配置模式只显示已选区域的 `x/y/w/h` 坐标表格，不显示模板字段。模板裁剪模式下，模板类型、模板标签和来源角色使用下拉框；模板标签会显示中文含义，也可以输入新标签。

“单图标注 / 测试 DanZero”打开后会后台使用区域库和模板库自动识别当前截图：

- `level_rank` 区域 + `rank/*_level` 模板：识别当前级牌，并按当前 DanZero 规则推导百搭级别；
- `my_hand` 区域 + `rank/suit` 手牌模板：识别我方手牌，牌面不完整或置信度不足时保留在输入框中手动补全；
- `*_play` 区域 + 出牌模板、`passed_*` 状态模板：识别“谁出了什么牌/谁不出”；
- `first_play_*` 状态模板：识别本轮首出者；`timer_*` 区域 + `timer/active` 模板：在画面存在活动计时器时识别当前行动者。

识别不到的字段会显示为“待手动确认”，不会用猜测值发送给 DanZero。页面会把手牌显示为带颜色花色图案的牌面卡片，并在原图上绘制识别框、牌面和识别类别；页面整体采用分区滚动布局，鼠标滚轮不会改变识别下拉框的值。确认后点击“测试 DanZero”，程序会构建 `GuanDanState`，异步调用本地 DanZero，显示推荐动作、耗时、识别置信度、参数摘要和 `engine_input` JSON。当前识别是基于项目模板的视觉匹配，不依赖 OCR；模板不足时仍可手动修正。

区域角色包括 `hand`（手牌）、`play`（出牌）、`anchor`（锚点）和 `generic`（通用）。模板记录和图片资产分别保存在 `templates_config.json` 与 `templates/`。

## 录制文件

默认 profile 为 `tencent_daguandan`。每局录制都会创建独立目录：

```text
data/profiles/tencent_daguandan/screenshots/
  game_YYYYMMDD_HHMMSS/
    000001.png
    000002.png
    session.json
```

`session.json` 包含录制开始/结束时间、间隔、帧数，以及首帧的窗口、DPI、客户区和标准化信息。

## 模板资产

现有腾讯大掼蛋模板已复制到：

```text
data/profiles/tencent_daguandan/templates/
```

在“区域标注”页将模式切换为“模板裁剪”即可迁移原项目的裁剪流程：图片只能从
`screenshots/` 目录选择，拖拽图片框选 ROI，选择模板类型、标签和样本来源后点击“裁剪并保存模板”。
模板记录以表格展示，包含绝对坐标和比例坐标；按钮模板可直接选择或裁剪“超级加倍”“加倍”“要不起”等标签。
`rank`/`suit` 会提取前景并生成白底黑字模板，其余类型保留原始颜色；每条记录都会写入
`templates_config.json`，包含来源图片、来源角色、绝对坐标、比例坐标和标准画面尺寸。模板列表支持多选删除，删除不会影响原始录制截图。

支持的模板类型为 `rank`、`suit`、`anchor`、`button`、`status`、`effect` 和 `timer`；模板文件按类型保存到 `templates/<kind>/`。

单图页面会单独显示图片识别耗时；打开单图页面后还会后台预加载 DanZero 模型。测试结果会显示 DanZero 总耗时、后台模型初始化耗时和模型推理等分段耗时，便于区分首次加载与后续调用速度。动作按钮区会自动识别并标出“出牌”“不出”“要不起”“加倍”“超级加倍”等按钮值。

## DanZero API

调用方最终必须确认级牌、百搭牌、当前行动者、本轮首出者与自己的手牌。截图本身不会发送给 DanZero，只会把识别/确认后的 `GuanDanState` 转成策略输入。

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

`DanzeroAdvisor` 会验证模型权重；如需替换权重，可将 `DAGUANDAN_DANZERO_CKPT` 环境变量设置为自定义权重文件路径。

## 验证

```powershell
$env:QT_QPA_PLATFORM = "offscreen"
$env:PYTHONPATH = "src"
.\.venv\Scripts\python.exe -m pip install -r requirements-dev.txt
.\.venv\Scripts\python.exe -m pytest -q
```
