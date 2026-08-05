# 大掼蛋 DanZero 桥接器设计

## 目标

创建一个独立的 Windows 桌面项目，用于录制大掼蛋牌桌截图、保存现有腾讯大掼蛋模板资产，并向外部 Python 调用方提供 DanZero 出牌建议 API。

## 范围

- 提供截图预览、开始本局录制、结束本局录制与录制间隔设置。
- 每局录制保存至独立目录，截图按序号命名，并维护 `session.json` 元数据。
- 将现有 `tencent_daguandan/templates` 原样复制至新项目的同名 profile。
- 提供手动构造牌局状态后调用 DanZero 的稳定 Python API，并包含内置模型权重。
- 提供启动说明、资产位置说明和 API 调用示例。

## 非目标

- 不从截图自动识别手牌、出牌、级牌或回合状态。
- 不把截图自动转换成 DanZero 输入。
- 不向游戏客户端发送点击、键盘或自动出牌操作。
- 不迁移标注、自动识别和自动追踪界面。

## 架构

项目包名使用 `daguandan_bridge`，按三个独立边界组织：

1. `capture` 负责 Windows 窗口定位、客户区截图、标准化图片及录制会话落盘。
2. `gui` 仅负责录制界面和状态展示，调用 `capture` 服务，不依赖模板或 DanZero。
3. `danzero` 负责显式牌局状态、规则转换、模型权重加载和建议结果；它不读取截图，也不依赖 GUI。

默认 profile 为 `tencent_daguandan`。截图保存在 `data/profiles/tencent_daguandan/screenshots/game_YYYYMMDD_HHMMSS/`；模板保存在 `data/profiles/tencent_daguandan/templates/`。

## 数据流

截图录制：用户选择窗口配置并开始预览 -> 捕获服务返回标准化帧 -> GUI 定时请求保存 -> 会话目录写入 `000001.png` 等图片与 `session.json`。

DanZero：调用方构造完整 `GuanDanState` -> `DanzeroAdvisor.recommend()` 验证状态并调用本地模型 -> 返回建议牌组及诊断信息。调用方负责状态正确性。

## 错误处理

- 未找到目标窗口、截图后端不可用、无有效预览帧及无 profile 时，GUI 显示可读错误并暂停录制。
- 录制间隔必须在 0.1 至 60 秒之间；已结束会话拒绝继续写入。
- DanZero 缺少依赖、模型权重不存在或校验失败、牌局状态不完整时，API 抛出明确的领域错误，不尝试从图片猜测数据。

## 验证

- 测试创建录制会话、保存标准化帧、写入首帧元数据及结束会话。
- 测试随项目分发的模板目录非空。
- 测试 DanZero API 的模块可导入、模型权重可定位，以及无效状态返回明确错误。
- 对 GUI 执行无窗口（offscreen）初始化测试，确保录制页可创建。

## 依赖与约束

- Python 3.12，Windows 11。
- 使用 PySide6、PySide6-Fluent-Widgets、OpenCV、MSS、pywin32、NumPy 和 PyTorch。
- 新项目目录固定为 `C:\\project\\python_project\\daguandan_danzero_bridge`。
- 原项目与其 `data/profiles/tencent_daguandan/templates` 均只读；迁移通过复制而非移动完成。
