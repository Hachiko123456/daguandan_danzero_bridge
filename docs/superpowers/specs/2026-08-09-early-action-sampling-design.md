# 首次动作采样延迟设计

## 目标

让实时 DanZero 的默认“两次有效牌型一致”策略在动作区域发生变化后立即开始采样，避免被全局 1000ms 等待延迟而错过短暂展示的牌面。

## 根因

`LiveOrchestrator._activate_zone()` 目前把构造参数 `settle_ms`（默认 1000ms）与策略的 `settle_ms` 取较大值。于是策略配置为 0ms 的 `two_valid_streak` 和 `valid_candidate_vote` 仍然实际等待 1000ms。

## 方案

- 将 `LiveOrchestrator` 的通用 `settle_ms` 默认值改为 0ms。
- `ZoneLifecycle` 的实际等待仍由当前动作识别策略的 `RecognitionStrategySpec.settle_ms` 决定。
- 保留 `max(self.settle_ms, strategy.settle_ms)`，使调用方显式传入更大等待时间时仍可覆盖策略。
- 因此：
  - `two_valid_streak`、`valid_candidate_vote`：立即采样；
  - `stable_single_shot`：400ms；
  - `reference_single_shot`：1000ms。

## 不在本次范围

- 不出模板的生命周期防残留。
- 接风规则、可信日志语义校验。
- UI、录像、DanZero 请求逻辑。

## 验收

新增回归测试证明：使用默认构造参数和 `two_valid_streak` 时，动作变化后的首帧即会调用出牌识别；使用 `reference_single_shot` 时，1000ms 前不会调用，1000ms 后才会调用。
