# 窗口校准/绑定服务

`daguandan_bridge.window_calibration` 把一次已经观测到的目标窗口客户区，接到
`runtime_layout` 已有的稳定持久化目录。它是纯构造函数加一个很薄的服务门面，
不会调用窗口调整、聚焦、移动、点击或按键 API。

## API

- `build_window_binding(...)`：只在内存中生成 `(binding_id, binding)`。
- `build_calibration(binding)`：从 binding 生成 JSON-safe calibration 文档。
- `save_window_calibration(layout, ...)`：同时调用既有的
  `save_window_binding` / `save_calibration`，返回两个文档和路径。
- `load_window_calibration(layout, profile_name, binding_id)`：加载并校验一对文档，
  用于 round-trip。
- `WindowCalibrationService(layout)`：提供 `save(**kwargs)` 与 `load(...)` 的可调用门面。

输入包括 profile 名称、语义 `application_id`、窗口类名、标题角色、实际
`ClientRect`、DPI、viewport size 和 base size。绑定 ID 只由语义身份与客户区
宽高决定，不包含 HWND、屏幕位置或绝对路径。因此同一语义和尺寸在不同机器上
仍会得到相同 ID，而不同客户区尺寸会得到不同 ID。

实际 `client_rect` 会被记录为观测证据；其中 `left`/`top` 仅是本次观测位置，
不会进入稳定 ID。JSON 中明确写入 `control_policy`，表示本服务只观察、绑定和
保存：**不会自动 resize，也不会控制窗口**。

保存位置由传入的 `RuntimeLayout` 决定，复用既有的 `calibrations/` 与
`window_bindings/` 稳定目录，并由 `runtime_layout` 的安全路径校验保证位于
`app_data_root` 下。模块不修改现有源码文件。
