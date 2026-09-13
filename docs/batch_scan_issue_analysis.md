# 「扫描未验证对局」两个问题的排查与解决方案

> **状态：已实施并验收通过。** 修复内容、量化验收标准与实测结果见 **`docs/batch_scan_fix_acceptance.md`**
> （内存：单局峰值 597+ MB → 稳定 ≈140 MB；两局并发 ≈280 MB；产物逐字节不变；89 项测试全绿）。
> 下文为原始排查与原因分析。
>
> 排查对象：GUI 回放页的 **扫描未验证对局** 批量扫描（`src/daguandan_bridge/application/unverified_batch_scan.py`
> + `gui/replay_page.py`），底层复用单局扫描 `application/video_scan.py` 的 `VideoActionScanner`。
> 排查时间：2026-09-11 20:00–20:20。本机：12 逻辑核，Windows。

---

## 0. 结论速览

| 现象 | 根因 | 定级 |
|---|---|---|
| ① 进度在两个会话间来回跳 | 服务端把**每个会话每帧**的进度原样透传；GUI 用"最后到达的那个会话"的 fraction 反推总进度 | 真 bug（口径错误） |
| ② 跑久了电脑卡死 | **主因**：保留帧内存 0.5–0.6 GB/局且**整局不释放**，并发 2 路 ≈1.2 GB，且与 GUI **同进程**；**次因**：单路占满约 1 个核、部分 Python 代码持 GIL 抢 UI 线程 | 真 bug（资源未回收 + 进程内运行） |

---

## 1. 现象①：进度跳来跳去

### 1.1 代码链路（证据）

1. **扫描端：每帧回调一次**（`video_scan.py:248-249`）
   ```python
   if on_progress is not None:
       on_progress(decoded_frames, _total_frames(capture, indexed), frame_index)
   ```
   `decoded_frames` 是**本会话**的帧计数，`total` 是**本会话**总帧数。

2. **批处理端：原样透传 + 只附一个"已完成会话数"**（`unverified_batch_scan.py:86-89, 104-106`）
   ```python
   def emit(session_id, done, total, frame):
       if on_progress is not None:
           with lock:
               on_progress(session_id, done, total, frame, completed)   # completed = 已结束的会话数
   ...
   on_progress=lambda done, total, frame: emit(item.session_id, done, total, frame)
   ```
   两个 worker 线程并发调用，**同一个回调被两个会话交替触发**。

3. **GUI 端：用"最后到达的那一条"反推总进度**（`replay_page.py:1288-1297`）
   ```python
   count = max(1, self._batch_selected_count)
   fraction = min(1.0, done / total) if total else 0.0     # ← 只取当前这条消息所属会话的比例
   percent = int(((completed + fraction) / count) * 100)   # ← completed 长期为 0
   ```
   于是：A 报到 40% → 显示 20%；B 报到 10% → 显示 5%（**回退**）；A 报到 41% → 21%……
   进度条在两个值之间来回跳，标签里"当前 {session_id} / 第 n 帧"也随之乱跳。
   另外两局总帧数不同（755 vs 887），跳动更明显。

### 1.2 解决方案

**a. 服务端改为"聚合口径"（核心）** —— 维护每个会话的最新进度，按**全批**计算：

```python
progress: dict[str, tuple[int, int]] = {}          # session_id -> (done, total)
def emit(session_id, done, total, frame):
    with lock:
        progress[session_id] = (done, total)
        started = sum(min(1.0, d / t) for d, t in progress.values() if t)
        overall = (completed + started) / max(1, len(selected))   # 0.0 ~ 1.0，单调不减
        on_progress(session_id, done, total, frame, completed, int(overall * 100))
```

**b. 节流**：只在 `overall` 百分比变化 ≥1 或距上次 ≥200 ms 时才发信号（现在是每帧 × 2 会话）。
可加 `"%d" 变化才 emit` 或 `time.monotonic()` 比较。

**c. GUI 只把进度条绑定 `overall`**（保证单调不回退），把"当前会话 + 帧明细"放到**第二行文字**作为明细：

```
批量扫描：42%（3/7 局完成）
  ├ 当前：game_20260829_114157_a940ae  第 512/887 帧
```

**d. 需要同步改的位置**（3 处，改动很小）：
- `unverified_batch_scan.py`：`emit()` 与 `on_progress` 签名（新增 `overall_percent`）
- `replay_page.py`：`UnverifiedBatchScanThread.progress = Signal(str, int, int, int, int, int)` 与 `_unverified_batch_progress(...)` 增加参数
- 现有测试 `tests/test_unverified_batch_scan.py` 用的是 `on_progress=lambda *args: ...`，**不受影响**

---

## 2. 现象②：跑久了电脑卡死

### 2.1 主因：保留帧内存不回收（实测）

`video_scan.py` 的 `_PendingSuitFrameStore`（`:133-172`）：

```python
if _has_unknown_suit(cards): self.pending_seats.add(seat)   # 某座位出现 '?' → 进入待重读
if self.pending_seats:
    self.frames[frame_index] = copy_frame(frame)            # ← 每帧复制整张 1280×720 BGR
for seat in tuple(self.pending_seats):                      # 该座位牌面变空/变 PASS → 移出
    ...
```

问题：`self.frames` **只增不减**（`grep` 全文件只有 `:166` 一处写入，没有任何删除/prune/clear），
要等**整局扫描结束**（`scan()` 返回）才随对象释放。

**实测保留量**（用 `reports/_diag_batch_mem.py` 逐帧重放上述逻辑得到）：

| 会话 | 观测帧数 | 保留帧数 | 占比 | 单帧 2.64 MB → 内存 |
|---|---:|---:|---:|---:|
| game_20260829_192802_981e50 | 755 | **245** | 32.5% | **≈ 0.63 GB** |
| game_20260829_114157_a940ae | 887 | **192** | 21.6% | **≈ 0.49 GB** |

**实测内存曲线**（本机跑真实一局 `video_scan`，每 10 s 外部采 `tasklist` 的 Working Set）：
```
276 MB → 321 → 343 → 344 → 345 → 346 → 346 → 353 → 343 → 345 → 438 → … → 597 MB (560/755 帧，仍在爬升)
```
阶梯式上涨（每开一个新的保留窗口涨一截），平台期只是被换到页面文件，**并没有释放**。
整局 7m51s 跑完（`status=complete`）。

**精确验证（决定性）**：单独做实验——按上面 6 个保留窗口把 245 帧真实解码并 `frame.copy()` 后驻留
（`reports/_verify_frame_mem.py`）：

```
基线 RSS = 39 MB
持有 245 帧 (720,1280,3) 后：RSS = 692 MB   ← 增量 653 MB，单帧 2.67 MB
结论：单局仅"保留帧"这一项常驻 ≈ 0.64 GB；批量 2 路并发 ≈ 1.28 GB
```

与 §2.1 表格测算的 0.63 GB 完全一致，也与外部采样（560 帧时 597 MB、预计收尾约 0.7 GB）吻合。

**并发放大**：批量 `max_workers = min(2, cpu//2)`，即**同时 2 局** → 常驻约 **1.2–1.3 GB**，
而且这 2 路与 GUI **在同一个进程**里（`UnverifiedBatchScanTask` 是 `QThread`，内部用 `ThreadPoolExecutor`）。
在内存偏小的机器上就会触发换页抖动 → 整机卡死。

### 2.2 次因：CPU 与 GIL

- 实测单局 ≈ **0.6 s/帧**（1.65 fps）≈ 占满 **1 个核**；2 路 ≈ 2 个核。
- 识别是 cv2 + numpy 混合，`matchTemplate`/`connectedComponents` 等释放 GIL，但**大量 Python 胶水代码仍持 GIL**。
- 扫描线程与 Qt 主线程**同进程**，GUI 主线程被抢 → 界面卡顿/无响应（低核机器尤甚）。
- 顺带：`_total_frames(capture, indexed)` 在**每帧**都被调用一次（`:249`），属可省的小开销。

### 2.3 解决方案（按性价比排序）

**P0-a 保留帧剪枝（改动小、收益最大）**
- `pending_seats` 变空时立即 `self.frames.clear()`；
- 更细：按"每个座位的待重读窗口"保留，窗口外的帧立即删除；
- 增加硬上限（如 120 帧）兜底：超限时告警并淘汰最旧帧。

**P0-b 重读完成后显式释放**
```python
actions = repair_scan_actions(actions, observations, reread=...)
pending_suit_frames.frames.clear()      # ← 新增：别让 0.6 GB 挂到整局结束
```

**P0-c 只保留重读需要的 ROI，而不是整帧**
`_suit_reread_callback` 的重读只针对某个座位的牌区，可只 `copy` 该座位的裁剪（几十 KB/帧），
把 2.64 MB/帧 降到 ~0.05 MB/帧，**直接减少 98% 内存**。（需确认重读是否只用一个座位区域。）

**P1 把批量扫描移出 GUI 进程（根治 UI 卡死与隔离）**
- 用 `QProcess` 或子进程跑一个 CLI worker（项目已有 `scripts/run_video_scan.py` 这样的现成入口），
  GUI 只负责发任务、收进度、收结果；
- 好处：内存峰值与崩溃都隔离在子进程，UI 始终流畅；子进程结束即彻底回收内存。
- 代价：需要把批处理编排（选会话、写 summary）也挪到 worker 或做成"GUI 编排 + worker 执行单局"。

**P2 资源感知的并发度**
- `recommended_workers()` 现在只看 CPU（`min(2, cpu//2)`），应同时看内存与"低配降为 1"；
  可读 `GlobalMemoryStatusEx`（ctypes，无需 psutil）。

**P2 其他小优化**
- `_total_frames()` 缓存一次，不要每帧调用；
- 进度节流（见现象①）。

---

## 3. 建议实施顺序

1. **P0-a + P0-b**（保留帧剪枝 + 重读后释放）—— 直接消掉 ~1.2 GB 峰值，改动集中在 `video_scan.py` 一个类。
2. **现象①的 a/b/c/d**（进度聚合 + 节流）—— 改动 3 处，体验立刻正常。
3. **P0-c**（ROI 化保留）—— 进一步把单局内存压到几十 MB。
4. **P1**（子进程化）—— 架构性改动，建议单独一轮。
5. **P2**（并发度资源感知 + 小优化）。

验证方式：改完后用本报告的探针脚本复测内存峰值（目标：单局峰值 < 150 MB，2 路并发 < 400 MB），
并确认 `tests/test_unverified_batch_scan.py`、`tests/test_video_scan.py` 全绿。

---

## 附：本次排查使用的临时脚本

- `reports/_diag_batch_mem.py` —— 逐帧重放保留逻辑，量化保留帧数与内存
- `reports/_probe_mem_cpu.py` / `reports/_selfprobe_scan.py` —— 内存/CPU 探针（后者为进程自测量）
- `reports/_selfprobe.log` —— 真实一局的采样日志
