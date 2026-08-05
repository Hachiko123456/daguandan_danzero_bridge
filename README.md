# 大掼蛋 DanZero 桥接器

这是一个独立的 Windows 11 项目，用于录制腾讯大掼蛋牌桌截图、保存对应模板资产，并通过 Python API 调用本地 DanZero 策略。

它不识别截图内容、不从图片自动推断牌局状态，也不会对游戏客户端执行点击、按键或自动出牌。

## 安装与启动

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe run.py
```

也可以双击 `start_gui.bat`。在界面中先点击“开始预览”，确认画面后点击“开始本局录制”；录制按设置的间隔保存帧，点击“结束本局”关闭当前会话。

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

模板仅作为随项目分发的资产；第一版不提供标注或自动识图功能。

## DanZero API

调用方必须手动确认级牌、百搭牌、当前行动者、本轮首出者与自己的手牌。截图不会自动发送给 DanZero。

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
