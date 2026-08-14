# 模型整局评测设计

## 1. 背景与决策

“对局回放与复测”负责录像播放、识别链路复测、TruthLog 编辑、状态重放和可信日志驱动的实时建议回放。这些能力用于检查采集、识别和状态管线，不应再承载某个历史决策点的模型探针。

本设计新增一个独立的“模型整局评测”能力。它读取一局 TruthLog，在每个我方真实动作发生前调用指定策略预测，然后仍用真实动作推进状态，最终生成逐决策明细和整局指标。首版不实现模型之间的闭环对战，也不改动实时助手、现有回放管线或 TruthLog 编辑器。

核心决策如下：

- 评测方法固定为 teacher-forced，不允许模型预测改变后续状态。
- “正式完整局评测”和“草稿诊断”是两种明确分开的输入模式，结果不可混合汇总。
- 受试策略显式选择为 DanZero、FableDan 模型或 FableDan RuleAgent 规则基线。
- FableDan 模型必须实际加载有效的 `fabledan_weights.npz`；缺失、损坏或运行时退化为 RuleAgent 时，不能记为 FableDan 模型成绩。
- 每次运行写入会话目录下不可覆盖的独立报告目录。

## 2. 目标与非目标

### 2.1 目标

- 用完整真实动作链评价模型在我方每个决策点上的离线选择质量。
- 保证每次预测只看到该动作之前的已知状态，杜绝未来动作或结果泄漏。
- 同时支持可发布的正式完整局结果和用于排错的草稿诊断结果。
- 用一致的明细格式比较 DanZero、FableDan 模型和 FableDan RuleAgent。
- 让失败、跳过、取消和模型资源问题可审计，不用降级或猜测掩盖问题。

### 2.2 非目标

- 不在本设计中实现 evaluator、命令行或 UI。
- 不替换现有 `replay_truth_through_live_advisor`，也不修改 `live/replay.py`。
- 不恢复单决策点探针或在回放页中嵌入模型诊断面板。
- 不计算反事实胜率、闭环胜率或“采用模型动作后的最终名次”。
- 不把 Q 值、logit 或规则打分解释为概率。

## 3. 术语与评测单位

- **真实动作**：TruthLog 当前行记录的 `actor/is_pass/cards`。
- **决策点**：某个 `actor == "self"` 的真实动作应用前的状态。
- **前缀**：初始状态加目标行之前的所有真实动作，不包含目标行及其未来。
- **预测动作**：受试策略在决策点返回的标准化动作。
- **teacher-forced**：先在真实前缀上预测，再把真实动作而非预测动作应用到状态，继续下一行。
- **正式完整局**：满足完整性、人工验证和终局条件，可进入正式汇总的 TruthLog。
- **草稿诊断**：未完成或未全部验证，但其已知前缀可以重放的 TruthLog；结果只供定位问题。
- **覆盖决策**：模型成功返回可比较动作的我方决策点。

动作比较使用规范化表示：不出为 `{is_pass: true, cards: []}`；出牌为 `{is_pass: false, cards: [...]}`，牌面按稳定规则排序并按多重集合比较。展示名称、牌序或模型内部 action index 不参与 exact 判定。

## 4. 输入资格

### 4.1 通用资格

正式完整局和草稿诊断都必须通过以下前置校验：

1. 会话目录存在，TruthLog 可解析，`source_session_id` 与所选会话一致。
2. `round_level`、`lead_player`、我方初始手牌和动作牌码合法；同一物理牌不超过双副牌上限，不接受带 `?` 的未知花色。
3. turn 编号从 1 连续，actor 合法，出牌非空、不出不带牌，trick id 为正且不递减。
4. 我方初始手牌必须是精确的 27 张；每个前缀经状态重放后都满足行动顺序、剩余牌数和动作合法性约束。
5. 至少存在一个我方决策点。
6. 所选策略通过严格的模型/规则运行时预检。

视频、帧索引和历史 `decisions.jsonl` 不是 teacher-forced 运行的必需输入；它们只可作为报告中的来源链接，不能代替 TruthLog 动作链。

### 4.2 正式完整局评测

正式模式只能读取已经落盘的 `<session>/truth_log.json`，并额外要求：

- 会话 manifest 已封存，TruthLog 本身为 `label_status == "verified"`。
- 每个 turn 都为 `verified`，有 provenance 和证据帧，且没有 uncertainty。
- `outcome.complete == true`，四家 finish order 完整，team result、reward 和 reward scheme 自洽。
- 完整动作链能够重放到终局，重放得到的出完顺序与 outcome 一致。

任何一项不满足都在开始预测前进入 `blocked`，不生成看似正式的部分分数。

只有状态为 `completed`、coverage 为 100% 且没有决策错误的正式运行，才具备进入跨局正式汇总的资格。`completed_with_errors` 的产物仍保留，但 `formal_eligible` 必须为 `false`。

### 4.3 草稿诊断

草稿模式可读取回放编辑器中的内存草稿快照或已落盘的 draft TruthLog，允许：

- 动作链尚未到终局；
- outcome 不完整；
- 日志或部分 turn 尚未 verified；
- 缺少证据帧。

它仍必须满足通用牌码和状态重放约束。校验器按前缀工作：无法重放的位置记录为错误，其后的依赖决策不再假装有效。所有草稿报告都写明 `evaluation_mode: "draft_diagnostic"`、`formal_eligible: false`，UI 使用“诊断”而非“评分”文案，且不进入正式榜单或跨局均值。

启动运行时必须冻结输入快照并记录 SHA-256。编辑器后续修改不会改变正在运行的输入；新内容必须发起新 run。

## 5. 受试策略与严格运行时身份

策略选择必须是枚举值，不能依赖现有“缺模型自动回退”的行为：

| strategy_id | 展示名 | 必需资源 | 允许的实际后端 |
| --- | --- | --- | --- |
| `danzero_model` | DanZero | DanZero 依赖和模型权重可加载 | DanZero 模型 |
| `fabledan_model` | FableDan 模型 | `<profile>/models/fabledan_weights.npz` 存在、可读、结构校验通过 | `numpy` |
| `fabledan_rule` | FableDan RuleAgent 规则基线 | vendored RuleAgent 可初始化 | `rule` |

运行前保存策略 ID、模型路径、文件摘要、适配器 schema、代码版本和实际 backend。DanZero 或 FableDan 模型资源缺失、不可读或初始化失败时，运行状态为 `blocked`。

当前 FableDan advisor 在 npz 缺失/损坏时可能自动使用 RuleAgent。整局评测层不得接受这一回退：

- 选择 `fabledan_model` 时，预检必须确认 `backend == "numpy"` 且 `status == "loaded"`。
- 缺少或损坏 npz 必须 `blocked`，`evaluated_decisions == 0`。
- 运行期间每次结果也必须保持 `strategy == "fabledan-numpy"`。若推理异常触发 backend 漂移，立即停止并记为 `failed`；此前明细可以保留，但不得重标为规则成绩。
- 只有显式选择 `fabledan_rule` 才能执行 RuleAgent，并在所有产物中标记为规则基线。

## 6. Teacher-forced 数据流与防泄漏边界

```text
会话 / 冻结的 TruthLog 快照
        │
        ├─ 输入校验 ──失败──> blocked + summary
        │
        ├─ 策略严格预检 ──失败──> blocked + summary
        │
        ▼
初始状态构建器
        │
        ▼
按 turn 顺序遍历
  ├─ 若 actor != self：校验并应用真实动作
  └─ 若 actor == self：
       1. 从“仅含过去”的当前状态生成不可变 snapshot
       2. 调用模型并计时
       3. 模型返回后才读取目标真实动作做比较
       4. 写一条 decision 明细
       5. 丢弃预测分支，应用目标真实动作
        │
        ▼
聚合 metrics → 原子写 summary / decisions / report
```

防泄漏必须是接口边界，而不只是调用顺序约定：

- 策略适配器只接收当前 `GuanDanState`/`LocalStrategySnapshot`、request id 和运行配置，不接收 TruthLog、目标行、实际动作、outcome 或未来 turn。
- evaluator 在模型调用完成之前不构造包含 actual action 的比较对象。
- 状态哈希只覆盖初始状态和已应用的真实前缀；报告可记录 `prefix_turn_count` 和 `state_before_sha256`，但不得把未来动作写入 engine input。
- 对手真实动作和我方真实动作都只在各自行到达时应用。预测动作永远不写回权威状态。
- 测试使用“未来污染”样本：保持目标前缀不变、任意修改目标后的动作和 outcome，目标点模型输入哈希及预测必须不变。

伪代码如下：

```python
state = build_state(frozen_truth.initial_state)
for turn in frozen_truth.turns:
    validate_actor_and_action(state, turn)
    if turn.actor == "self":
        state_before = state.local_snapshot()
        predicted = selected_strategy.predict(state_before)
        decision = compare(predicted, actual_from(turn))
        write_decision(decision)
    apply_truth_action(state, turn)  # 始终是真实动作
```

## 7. 运行状态机

内部过程状态可为 `created → validating → ready → running → finalizing`，对外最终状态固定为：

- `completed`：全部合格我方决策点成功预测、比较和写入，coverage 为 100%。
- `completed_with_errors`：至少一个决策成功，但一个或多个决策因可恢复的状态/策略错误未覆盖；指标使用明确分母，且正式资格为 false。
- `blocked`：预测开始前的输入或策略前置条件不满足，例如完整局未 verified、TruthLog 不可重放、DanZero 资源不可用、FableDan npz 缺失/损坏。此状态不得包含模型/规则混算结果。
- `cancelled`：用户请求取消。停止创建新决策，在安全点收尾并保存已有明细；正式资格为 false。
- `failed`：运行已经开始后发生无法安全继续的系统级错误、写盘错误或后端身份漂移。尽最大努力写出 summary 和已有明细，但不伪装为完成。

单个决策失败是否可恢复由错误分类决定。局部输入映射失败且后续前缀仍可验证时可继续并最终 `completed_with_errors`；权威状态无法推进、策略后端变化或产物不可写时必须 `failed`。`blocked` 与 `failed` 不可仅凭“有没有异常”互换，关键区别是是否已进入模型预测阶段。

## 8. 产物与可追溯性

每次运行创建唯一目录：

```text
<session>/model_evaluation_runs/<run_id>/
├── summary.json
├── decisions.jsonl
└── report.md
```

`run_id` 建议为 UTC 时间戳加短 UUID；目录不可复用或覆盖。写入先在同级临时目录完成，三个文件关闭并校验后再原子发布。即使 `blocked`，也应生成 summary 和 report；`decisions.jsonl` 可以为空。

### 8.1 summary.json

至少包含：

- schema/version、run id、创建/结束时间、最终 status、error summary；
- session id、TruthLog 来源、输入 SHA-256、evaluation mode、formal eligible；
- strategy id、展示名、实际 backend、模型路径和 SHA-256、适配器/代码版本；
- total turns、eligible self decisions、evaluated/error/skipped decisions；
- metrics、延迟统计、取消信息和报告相对路径。

### 8.2 decisions.jsonl

每个我方真实动作对应且只对应一行，按 turn 顺序写入，至少包含：

- decision id、turn id、trick id、prefix turn count、lead/follow 场景；
- state revision、state-before SHA-256、request id；
- predicted action、actual action、exact match、pass/play match；
- 策略 ID、实际 backend、模型摘要、latency ms；
- `status: evaluated|error|skipped`、稳定错误码和错误说明。

明细可以包含审计所需的当前合法动作摘要，但默认不复制庞大的特征矩阵。若未来提供 debug 附件，应另行版本化，且同样不得包含未来动作。

### 8.3 report.md

人类可读报告展示输入资格、运行身份、最终状态、核心指标、错误摘要和逐决策表。报告必须突出“teacher-forced，非闭环胜率”，并对草稿结果显示永久的诊断水印/提示。

## 9. 指标定义

所有比例同时输出 numerator、denominator 和 rate；分母为 0 时 rate 为 `null`，不得写 0 冒充真实成绩。

- **coverage**：`evaluated_decisions / eligible_self_decisions`。error 和 skipped 保留在总分母中。
- **exact action accuracy**：规范化预测动作与真实动作完全一致的数量 / evaluated decisions。
- **pass/play binary accuracy**：预测和真实的 `is_pass` 相同的数量 / evaluated decisions；同时输出 actual-pass、actual-play 两个分组的 count、correct、rate 和 2×2 confusion matrix。
- **lead/follow**：按动作前状态分组。`lead` 表示我方在新牌墩或当前无有效领出动作时决策，`follow` 表示响应当前领出动作；每组输出 eligible、evaluated、exact、pass/play 和 coverage。
- **latency**：只对成功模型调用统计 count、total、mean、p50、p95、max（毫秒）；预热耗时与单次推理耗时分开。

可选展示牌型、合法动作数等诊断分组，但不得改变上述稳定指标的语义。Q 值或规则分数只能作为审计字段，不能跨 DanZero/FableDan/RuleAgent 直接比较。

## 10. 独立 UI 草图

入口位于主导航的独立页面“模型整局评测”，不放回“对局回放与复测”页面。

```text
┌ 模型整局评测 ──────────────────────────────────────────────┐
│ 对局 [session 下拉] [刷新]  输入 [正式完整局 / 草稿诊断]     │
│ 资格  已验证完整局 / 3 个阻断项 [查看详情]                   │
│ 策略  (●) DanZero  ( ) FableDan 模型  ( ) RuleAgent 基线    │
│ 模型  <profile>/models/fabledan_weights.npz  [校验]          │
│ [开始整局评测] [取消]                  状态：running 12/28   │
├─────────────────────────────────────────────────────────────┤
│ coverage  exact  pass/play  lead  follow  p50/p95 latency   │
├─────────────────────────────────────────────────────────────┤
│ 回合 │ 场景 │ 真实动作 │ 预测动作 │ exact │ 延迟 │ 状态      │
├─────────────────────────────────────────────────────────────┤
│ [打开报告目录] [查看 report.md]                              │
└─────────────────────────────────────────────────────────────┘
```

交互约束：

- 先显示资格检查，再允许开始；阻断原因必须直接可见。
- 选择 FableDan 模型时显示 npz 路径、摘要和加载状态；选择 RuleAgent 时明确显示“规则基线，不使用 npz”。
- 正式与草稿使用不同徽标和说明，草稿不能通过换文案伪装为正式。
- 运行期间冻结会话、输入模式和策略；取消是协作式取消，已有报告可查看。
- 默认列表展示逐决策比较，不展示单决策“重跑探针”按钮。

## 11. 为什么当前 TruthLog 不能计算闭环胜率

当前 TruthLog 只有我方精确初始手牌、首出者、真实动作链和可选真实终局，不包含四家的完整初始牌分配。它足以沿真实历史重建我方可见状态并做 teacher-forced 决策比较，但不足以建立可重复的完整对局模拟器。

更重要的是，一旦把模型在第一个我方决策点的动作真正应用到状态，原 TruthLog 中之后的真实动作就属于另一条历史：其他玩家的应对、后续领出关系、各家剩余牌和合法动作都可能改变。继续照搬真实后续会产生非法或因果错误的轨迹。因此现有 outcome 只能作为这条真实对局的元数据，不能当作采用模型动作后的结果，也不能用于计算模型闭环胜率。

未来若要做闭环胜率，必须另行定义包含四家初始牌、规则环境、对手策略、随机种子、多局采样和置信区间的模拟协议；这不属于本设计。

## 12. 分阶段实施建议

1. **契约与校验**：定义 evaluation request/result schema、正式/草稿资格检查、稳定错误码和运行目录协议。
2. **严格策略适配**：为三个 strategy id 建立显式 factory 和运行时身份校验；隔离 FableDan 模型与 RuleAgent，增加缺/坏 npz 阻断测试。
3. **核心 evaluator**：实现冻结输入、teacher-forced 状态推进、未来隔离、逐决策比较、取消点和指标聚合。
4. **报告写入**：实现不可覆盖 run 目录、原子发布、summary/decisions/report 三类产物及失败收尾。
5. **无 UI 验证入口**：先提供应用服务或窄 CLI，使用伪策略和小型 TruthLog 固化指标及状态机测试。
6. **独立 UI**：最后接入导航、资格面板、进度、取消、明细和报告打开能力，不改现有回放主流程。
7. **跨局汇总（后续）**：只汇总 `formal_eligible == true` 的 completed 运行，并固定数据集版本和策略模型摘要。

## 13. 验收标准

- **AC-001**：主导航中的“模型整局评测”与“对局回放与复测”职责独立，回放页不存在单决策点探针、决策下拉或探针面板。
- **AC-002**：正式模式仅接受落盘、verified、无 uncertainty、outcome complete 且能重放至一致终局的完整 TruthLog。
- **AC-003**：草稿模式允许不完整/未验证输入，但永久标记为 `draft_diagnostic` 和 `formal_eligible: false`，不进入正式汇总。
- **AC-004**：每个我方决策点都在真实动作应用前预测，策略接口无法访问目标动作、未来 turn 或 outcome。
- **AC-005**：无论预测是什么，状态推进始终使用当前行真实动作；预测不会影响后续决策输入。
- **AC-006**：修改目标点之后的未来动作或 outcome，不改变该目标点的 state-before 哈希和预测结果。
- **AC-007**：可显式选择 `danzero_model`、`fabledan_model`、`fabledan_rule`，报告记录选择值和实际 backend。
- **AC-008**：选择 FableDan 模型时，缺失或损坏 npz 产生 `blocked` 且零 evaluated decisions；任何 RuleAgent 回退都不能计入模型结果。
- **AC-009**：通用输入校验覆盖 session 归属、精确牌码、27 张我方初始牌、turn/trick 连续性、动作合法性和前缀可重放性。
- **AC-010**：每次运行写入 `<session>/model_evaluation_runs/<run_id>/{summary.json,decisions.jsonl,report.md}`，run 不覆盖且产物可追溯到输入和模型摘要。
- **AC-011**：每个我方真实动作在 decisions 中恰有一条有序记录，包含前缀身份、lead/follow、预测、真实动作、比较、延迟和错误状态。
- **AC-012**：summary 输出带分子/分母的 coverage、exact、pass/play、lead/follow 指标，以及 count/mean/p50/p95/max latency。
- **AC-013**：最终状态只使用 `completed`、`completed_with_errors`、`blocked`、`cancelled`、`failed`，并符合本设计的阶段和错误语义。
- **AC-014**：取消保留已完成明细并停止新预测；后端漂移、权威状态无法推进或产物无法安全发布时不得伪装为 completed。
- **AC-015**：独立 UI 在开始前展示输入资格和策略运行时身份，运行中可取消，结束后可查看逐决策结果和报告目录，且没有单点探针入口。
- **AC-016**：报告明确声明 teacher-forced 不是闭环胜率，并说明缺少四家初始牌及首个反事实动作后真实后续失效；自动测试覆盖未来污染、真实动作推进、缺/坏 npz 阻断、RuleAgent 显式选择、指标分母和五种最终状态。
