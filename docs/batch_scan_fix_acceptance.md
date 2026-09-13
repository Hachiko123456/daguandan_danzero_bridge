# 「扫描未验证对局」问题修复 — 验收标准与验收结果

> 承接 `docs/batch_scan_issue_analysis.md`（排查与原因分析）。
> 本文定义**量化验收标准**并给出**实测验收结果**。修复时间：2026-09-11 21:00–21:25。

---

## 1. 修复内容（共 3 个源文件 + 3 条新测试）

| # | 文件 | 改动 |
|---|---|---|
| A1 | `application/video_scan.py` | 新增常量 `_MAX_RETAINED_SUIT_FRAMES = 24`；`_PendingSuitFrameStore` 变为**有界**缓存：超限淘汰最旧帧并计数 `evicted`，新增 `release()` |
| A2 | `application/video_scan.py` | `scan()` 在 `repair_scan_actions(...)` 之后（`finally`）调用 `pending_suit_frames.release()`，不再把 ~0.65 GB 挂到整局结束 |
| C1 | `application/video_scan.py` | 总帧数改为**循环前解析一次**（`total_frames`），不再每帧调用 `_total_frames()` |
| B1 | `application/unverified_batch_scan.py` | 进度改为**全批聚合**：维护每会话进度、`_overall_percent()` 单调不减；按"变化 ≥1% 或 ≥200 ms"**节流**；回调签名扩展为 `(session_id, done, total, frame, completed, overall_percent)` |
| C2 | `application/unverified_batch_scan.py` | 新增 `_available_memory_gb()`（`GlobalMemoryStatusEx`）；`recommended_workers()` 在可用内存 < 2 GB 时降为 1 |
| B2 | `gui/replay_page.py` | `UnverifiedBatchScanThread.progress` 扩为 6 参；新增 `_relay_progress()` 兼容旧 5 参调用；`_unverified_batch_progress()` 进度条只绑聚合值且**单调不回退**，会话/帧明细降为同行说明文字 |
| T | `tests/` | 新增 3 条回归测试：保留帧封顶与释放、聚合进度单调、低内存降并发 |

> 关键前提（已实测验证）：**回源 seek 读取与顺序读取逐像素完全一致**（两局各 33 个抽查点，0 不一致），
> 因此淘汰的帧由定向回源补齐，不会改变扫描结果。

---

## 2. 验收标准（量化）与验收结果

### 2.1 内存（对应现象②"跑久了卡死"）

| 编号 | 验收标准 | 修复前实测 | 修复后实测 | 结论 |
|---|---|---|---|---|
| **AC-1** | 单局扫描进程 Working Set 峰值 **≤ 250 MB** | 597 MB（560/755 帧，仍在爬升；外推峰值 ≈750 MB） | **≈140 MB**（全程 133–141 MB） | ✅ |
| **AC-2** | 运行期间 Working Set **不随时间增长**：后半程波动 ≤ ±10% | 阶梯式持续上涨（276→438→597+ MB） | 133–141 MB，波动 ≈6%，无增长趋势 | ✅ |
| **AC-3** | 保留帧数量 **≤ 24 帧**（该部分内存 ≤ 64 MB，原 653 MB/245 帧） | 245 帧 ≈ **653 MB**（隔离实测） | 24 帧 ≈ 64 MB（-90%） | ✅ 单测覆盖 |
| **AC-4** | 两局并发峰值 **≤ 500 MB** | ≈1.2–1.4 GB（推算） | 2 × ≈140 MB = **≈280 MB** | ✅ |

### 2.2 进度（对应现象①"进度跳来跳去"）

| 编号 | 验收标准 | 结果 |
|---|---|---|
| **AC-5** | 进度为**全批聚合**，百分比**单调不减**、末值 100 | ✅ 新增单测 `test_batch_progress_is_one_monotonic_aggregate_across_workers` 断言 `percents == sorted(percents)` 且末值 100 |
| **AC-6** | 存在"尚有会话未完成时的部分聚合"样本（证明不是只靠 completed 计数） | ✅ 同单测断言 `any(pct > 0 for pct, args in ... if args[4] == 0)` |
| **AC-7** | 进度节流：同百分比下间隔 <200 ms 不重复上报 | ✅ 常量 `_PROGRESS_MIN_INTERVAL_SEC = 0.2` + `_PROGRESS_MIN_DELTA_PERCENT = 1` |
| **AC-8** | 进度条**不回退**；会话/帧明细不再影响进度条 | ✅ `setValue(max(当前值, 聚合值))`，明细进 caption |
| **AC-9** | 兼容既有调用方（5 参回调） | ✅ `_relay_progress` 容错；既有 GUI 测试保持通过 |

### 2.3 正确性 / 无回归

| 编号 | 验收标准 | 结果 |
|---|---|---|
| **AC-10** | 扫描产物**逐字节不变**（`action_trace.jsonl` / `turn_slots.json` / `opening_candidates.json` / `raw_action_trace.jsonl` / `scan_summary.json`；`frame_observations.jsonl.gz` 解压后一致） | ✅ 两局**全部一致**（192802、114157） |
| **AC-11** | 扫描质量指标不回退 | ✅ 192802：V1–V4 **全 PASS**；114157：V1/V2/V3 PASS（V4 仅剩已证"锚点期望错误"的 1 项） |
| **AC-12** | 相关测试全绿 | ✅ **89 passed**（4 个标准测试文件 + `test_video_scan` + `test_unverified_batch_scan` + `test_replay_page` + `test_corpus_workbench_page` + `test_session_corpus_validation`） |
| **AC-13** | 资源感知并发：可用内存 < 2 GB 时 `recommended_workers()` 返回 1 | ✅ 新增单测 `test_recommended_workers_drops_to_one_when_free_memory_is_low` |

### 2.4 未纳入本轮（明确记录）

| 项 | 状态 | 原因 |
|---|---|---|
| **P1 批量扫描移出 GUI 进程** | 未实施 | 属架构性改动（QThread+线程池 → QProcess/子进程 worker），且**本环境无法启动 GUI 做端到端验证**。AC-1~AC-4 已消除卡死的根因（内存不再无界增长、低内存自动降并发），P1 作为后续加固项 |
| 保留帧 ROI 化（进一步降内存） | 未实施 | 需把识别层的座位区域盒暴露给扫描器，耦合度较高；当前 24 帧上限已达 <150 MB 目标 |

---

## 3. 实测证据摘要

**内存（外部每 10–15 s 采 `tasklist` 的 Working Set）**

```
修复前：276 MB → 321 → 343 → … → 438 → … → 597 MB（560/755 帧，仍在爬升）
修复后：两个并发扫描进程 62–83 MB → 133–141 MB，随后**稳定在 133–141 MB**（波动 ≈6%，无增长）
隔离实测：持有 245 帧 = 653 MB  →  上限 24 帧 = 64 MB
```

**产物一致性（sha256）**

```
192802：action_trace.jsonl / turn_slots.json / opening_candidates.json /
        raw_action_trace.jsonl / scan_summary.json   → 5/5 逐字节相同
        frame_observations.jsonl.gz 解压后 sha256 相同（仅 gzip 头部时间戳不同）
114157：同上，5/5 逐字节相同，解压观测 sha256 相同
```

**扫描质量**

```
192802（…_211228）：V1 PASS / V2 PASS / V3 PASS / V4 PASS
114157（…_211233）：V1 PASS / V2 PASS / V3 PASS / V4 FAIL（锚点期望 4C 4S 物理不可能，见 verification_report_scan_play_after_fix.md §4.3）
```

---

## 4. 验收判定

| 维度 | 判定 |
|---|---|
| 现象① 进度跳来跳去 | **已修复**（AC-5 ~ AC-9 全部通过） |
| 现象② 跑久了卡死 | **已修复**（AC-1 ~ AC-4 全部通过；内存从持续增长 597+ MB 降为稳定 ≈140 MB，并发 ≈280 MB） |
| 扫描正确性 | **无回归**（AC-10 ~ AC-12 全部通过） |
| 剩余加固项 | P1 子进程隔离（需 GUI 环境验证）、保留帧 ROI 化（可选） |

---

## 5. P0–P2 完成度对账（逐条）

对照 `docs/batch_scan_issue_analysis.md` §2.3 的方案清单：

| 方案项 | 状态 | 说明 |
|---|---|---|
| **P0-a** 保留帧剪枝 | **部分完成** | ① 硬上限兜底 → **已做**（`_MAX_RETAINED_SUIT_FRAMES = 24`，比方案建议的 120 更严，超限淘汰最旧帧并记 `evicted`）；② "`pending_seats` 变空即 `clear()`" → **有意未做**：未知花色的动作窗口正好落在该 pending 窗口内，清空会迫使所有重读都走回源、废掉缓存本身的意义；且硬上限已给出内存上界，无额外收益；③ "超限告警" → **未做**（有意）：为保持"扫描产物逐字节不变"这一更强的可验证性质，不向 `scan_summary.json`/`warnings` 追加字段；`evicted` 计数已挂在对象上，需要时可随时暴露 |
| **P0-b** 重读后显式释放 | **已完成** | `scan()` 在 `repair_scan_actions(...)` 的 `finally` 中调用 `pending_suit_frames.release()` |
| **P0-c** 只保留重读 ROI（非整帧） | **未做（有意延后）** | 收益 ~63 MB（64 MB → 约 1 MB），但需把识别层的座位区域盒暴露给扫描器 / 改造 `recognize_region` 接口，耦合与风险偏高；当前单局 140 MB 已优于目标（≤150 MB），故列为可选优化 |
| **P1** 批量扫描移出 GUI 进程 | **未做（有意延后）** | 架构性改动（QThread+线程池 → QProcess/子进程 worker），且**本环境无法启动 GUI 做端到端验证**；AC-1~AC-4 已消除卡死根因（内存有界 + 低内存自动降并发），故作为后续加固项 |
| **P2** 并发度资源感知 | **已完成** | `_available_memory_gb()` + 可用内存 < 2 GB 时降为 1（单测覆盖） |
| **P2** `_total_frames()` 缓存一次 | **已完成** | 改为循环前解析 `total_frames` |
| **P2** 进度节流 | **已完成** | ≥1% 或 ≥200 ms 才上报 |
| （现象①的 a/b/c/d） | **已完成** | 聚合口径 + 节流 + 进度条只绑聚合值 + 明细降级为 caption |

**方案自带的验证目标**（§3 末）：单局峰值 < 150 MB、2 路并发 < 400 MB、相关测试全绿 —— **三项均已达成**（实测 ≈140 MB / ≈280 MB / 89 passed）。

> 结论：**现象①与现象②的根因均已修复并验收通过**；P0–P2 清单中 **P0-a 为部分完成（子项 ②③ 有意取舍）**、**P0-c 与 P1 未实施**（均已给出理由与后续方案）。

---

## 6. 复现方式

```bash
# 单测（含新增 3 条）
.venv/Scripts/python.exe -m pytest tests/test_video_scan.py tests/test_unverified_batch_scan.py \
    tests/test_replay_page.py tests/test_corpus_workbench_page.py tests/test_session_corpus_validation.py \
    tests/test_action_trace_reconciliation.py tests/test_turn_slot_projection.py tests/test_scan_action_repair.py -q

# 内存复测：跑一局并在外部采样 Working Set
.venv/Scripts/python.exe scripts/run_video_scan.py data/profiles/tencent_daguandan/sessions/<会话ID>
tasklist | grep -i python        # 观察 WS 是否稳定在 ~140 MB

# 产物一致性（与修复前参考报告比对）
sha256sum reports/video-scans/<会话>_<新时间戳>/action_trace.jsonl \
          reports/video-scans/<会话>_<旧时间戳>/action_trace.jsonl
```
