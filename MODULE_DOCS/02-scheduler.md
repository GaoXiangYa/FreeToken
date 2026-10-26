# 02 · scheduler：调度循环、分块 prefill、CacheManager、前缀缓存提交、状态上报

> 行号基于 `377a3bc`。完整符号清单见 [`99-symbol-index/scheduler.md`](99-symbol-index/scheduler.md)。

---

## 1. 模块职责与在系统中的位置

`Scheduler` 是**引擎侧唯一的主循环**。它持有 `Engine`，从 tokenizer 进程收 `UserMsg`，把它们变成批次，调 `Engine.forward_batch`，再把采样出的 token 送回 tokenizer 进程。

- **上游**：`tokenizer/server.py` 经 ZMQ 送来 `UserMsg` / `AbortBackendMsg` / `CacheRebuildBackendMsg`。
- **下游**：`Engine.forward_batch`、`CacheManager`（页分配与前缀缓存）、`TokenPool`（token 存放）。
- **离线模式**：`llm.LLM(Scheduler)` 继承它并把 ZMQ 收发替换成两个函数（见 `14`）。

```
                ┌──────────── Scheduler.run_forever ────────────┐
ZMQ ──▶ io.py ──┤ overlap_loop:                                  │
 (UserMsg)      │   收消息 → 调度批次 N → 发射 N → 排空 N-1 ──▶ ZMQ
                └───────┬───────────────────┬───────────────────┘
                        │                   │
                  Engine.forward_batch   CacheManager（页/前缀缓存）
```

---

## 2. 文件清单

| 文件 | 行数 | 职责 | 阅读建议 |
|---|---:|---|---|
| `scheduler.py` | 962 | `Scheduler` 主类 + 主循环 + 批次准备/排空 + 重建 | **精读**循环与 `_process_one_msg`/`_prepare_batch`/`_process_last_data` |
| `cache.py` | 650 | `CacheManager`：页空闲链表、page table、前缀缓存 match/commit、GDN/SWA 第二币种 | **精读** |
| `prefill.py` | — | `ChunkedReq` / `PrefillAdder` / `PrefillManager`：准入、预算、分块 | 精读 |
| `decode.py` | — | `DecodeManager`：运行中请求集合 | 略读 |
| `io.py` | 133 | `SchedulerIOMixin`：ZMQ 收发 + TP 多 rank 广播 | 精读（不长） |
| `mm.py` | — | 多模态的 chunk 规划（`plan_mm_batch` 等） | 按需 |
| `status.py` | 144 | 吞吐日志（prefill/decode token/s） | 精读（很短） |
| `table.py` | — | `TableManager`：`table_idx` 与 `token_pool` 分配 | 略读 |
| `config.py` | 43 | `SchedulerConfig`（继承 `EngineConfig`）+ ZMQ 地址 | 精读 |
| `utils.py` | — | `PendingReq` | 略读 |

---

## 3. 核心数据结构

### 3.1 `ForwardInput` / `ForwardData`（`scheduler.py:54` / `:61`）

| 类型 | 内容 | 用途 |
|---|---|---|
| `ForwardInput` | `batch` + 采样参数 | `_schedule_next_batch` 的产物：一个已组好、待发射的批次 |
| `ForwardData` | `ForwardInput` + `ForwardOutput` | 已发射 **未排空**的批次；`overlap_loop` 在两次迭代之间传递它 |

`overlap_loop` 的整个设计就是围绕"持有上一轮的 `ForwardData`"展开的。

### 3.2 `PendingReq`（`utils.py:15`）

尚未被准入的请求包装：`input_len()`（`:25`）/ `output_len()`（`:29`）给准入预算用。**它不是 `Req`**——只有被 `PrefillAdder` 接纳后才变成 `Req`。

### 3.3 `ChunkedReq`（`prefill.py:31`）

`Req` 的子类，代表"一个 prompt 还没算完的中间块"。
- 重写 `_alloc_ids_buf`（`:32`）/ `append_host`（`:35`）：块之间不追加 token。
- **`can_decode()` 恒返回 False**（`:39`）：中间块绝不能采样。
- 续块时它被重新包装成新的 `ChunkedReq` 或最终 `Req`（`PrefillAdder._add_one_req` 里）。

### 3.4 `CacheManager`（`cache.py:32`）的内部状态

| 字段 | 含义 |
|---|---|
| `free_slots` | 页对齐的空闲 token 索引列表（`arange(num_pages)*page_size`） |
| `page_table` | 共享的 `[max_req+1, width]` int32 表（`[table_idx,pos] → KV 槽位`） |
| `prefix_cache` | radix 前缀缓存实例（三选一 + naive） |
| `is_hybrid` / `is_swa` | 是否线性注意力 / 滑窗模型的缓存 |
| `sliding_window_size` | SWA 窗口大小 |
| `linear_state_pool` | GDN 状态池（hybrid 时非空） |

**不变量**：`len(free_slots) + 已分配页数 == num_pages`，且 `free_slots` 的每个元素都是 `page_size` 的倍数（`check_integrity()`，`:516`，会断言这两条）。

### 3.5 `PrefillAdder`（`prefill.py:44`）的预算字段

| 字段 | 含义 |
|---|---|
| `token_budget` | 本次批次还能放多少 token |
| `reserved_size` | 为"第二/第三币种"（SWA 页、GDN 槽）预留的量 |
| `can_prefill` | 是否还有准入名额 |

`_kv_reservation_size()`（`:63`）把"这个请求额外要占的 KV 页"折算成 token 当量，供预算判断。

### 3.6 `SchedulerConfig`（`config.py:15`）

继承 `EngineConfig`，只加了引擎不需要的东西：`max_extend_tokens`（分块大小）、`cache_type`、`offline_mode`、`decode_log_interval`、`special_token_ckpt`，以及 **3 个 ZMQ 地址**：

| 地址 | 行 | 用途 |
|---|---:|---|
| `zmq_backend_addr` | 26 | tokenizer → scheduler（`ipc:///tmp/freetoken_0`） |
| `zmq_detokenizer_addr` | 30 | scheduler → detokenizer（`ft1`） |
| `zmq_scheduler_broadcast_addr` | 34 | TP rank0 → 其它 rank（`ft2`） |

三者都带 `_unique_suffix`（`pid=<pid>`），保证多实例不撞。

---

## 4. 函数详解

### 4.1 `scheduler.py` — `Scheduler`

| 方法 | 行 | 作用 | 关键行为 |
|---|---:|---|---|
| `__init__` | 65 | 装配全部子系统 | 建 Engine → 第二条流（`:70-73`）→ `TableManager` → `CacheManager` → `DecodeManager` → `PrefillManager` → 各种状态（`_last_data`、`_pending_rebuild`、`_abort_tombstones`）→ `SchedulerIOMixin.__init__` |
| `run_when_idle` | 154 | 空闲后台任务 | 基类空实现 |
| `rebuild_cache` | 160 | idle-only 运行时重建入口 | 忙则回 `busy`（不排队等待）；否则挂 `_pending_rebuild` |
| `overlap_loop` | 207 | **主循环（默认）** | 见 §5.1 |
| `normal_loop` | 263 | 非 overlap 版本 | `ENV.DISABLE_OVERLAP_SCHEDULING` 时使用：一轮内发射并排空同一批 |
| `run_forever` | 291 | 循环入口 | 按环境变量选 `overlap_loop` / `normal_loop`；`@torch.inference_mode()` |
| `shutdown` | 307 | 收尾 | 同步 → 屏障 → `engine.shutdown()` |
| `_process_last_data` | 312 | **排空上一批** | `copy_done.synchronize()`（:317）→ 逐 req 取 token → EOS/stop/length 判定 → `DetokenizeMsg` → 释放/提交前缀 → 状态上报 → `send_result` |
| `_match_stop_str` | 431 | stop string 匹配 | 只在已解码的**尾部**做，保证每步成本恒定 |
| `_kv_usage_pages` / `_mamba_slot_usage` / `_swa_token_usage` | 447/458/469 | 三种池的占用率 | 供日志与 reply 携带 |
| `_gpu_mem_bytes` | 481 | 本进程占用的 GPU 字节 | 用 torch 分配器的 reserved 统计（不做 device sync） |
| `_process_one_msg` | 488 | **消息分派** | `UserMsg`：上下文检查 → max_tokens 钳制 → `add_one_req`；`AbortBackendMsg`：in-flight 则打标记否则立即释放；`CacheRebuildBackendMsg`：挂起 |
| `_restore_linear_states` | 616 | 前缀命中时 COW 恢复 GDN 快照 | 必须在引擎流上、前向之前做 |
| `_free_req_resources` | 628 | 释放一个请求的全部资源 | 页 + GDN 槽 + 表行 |
| `_reply_rebuild` | 642 | 回重建结果 | |
| `_execute_pending_rebuild` | 661 | 在安全点执行重建 | 见 §5.6 |
| `_current_cache_geometry` | 733 | 当前池几何（回滚快照） | |
| `_log_cache_geometry` | 751 | 重建后一行式读出 | |
| `_prepare_batch` | 786 | **组批** | padding、`allocate_paged`、positions、page_table gather、GDN 表、`build_fla_metadata`、`attn_backend.prepare_metadata`、`Sampler.prepare` |
| `_gather_multimodal` | 847 | 多模态 gather 计划 | 交给引擎在前向之前执行 |
| `_schedule_next_batch` | 863 | 选下一个批次 | 先试 prefill，再试 decode |
| `_report_prompt_admissions` | 875 | 首次 prefill 计费回执 | **批次准备成功之后**才发布，保证恰好一次 |
| `_flush_abort_acks` | 890 | 刷 abort 确认 | 必须排在 `_process_last_data` 之后 |
| `_forward` | 898 | 调引擎 | `@torch.inference_mode()`；`engine.forward_batch` |
| `_make_positions` | 931 | 位置张量 | |
| `_make_mrope_positions` | 909 | `[3,N]` 的 mrope 行 | 图像 token 用它预计算的列，其它是序列索引+每请求 delta |
| `_make_input_tuple` / `_make_write_tuple` | 947/957 | 输入/写入的 `Indice2D` | write tuple 用固定形状 + `-1` 汇列 |

### 4.2 `cache.py` — `CacheManager`（关键部分）

| 方法 | 行 | 作用 | 关键行为 |
|---|---:|---|---|
| `__init__` | 33 | 建空闲链表与前缀缓存 | `free_slots = arange(num_pages)*page_size` |
| `prefix_replay_tokens` | 70 | 前缀命中必须留出的尾部 token | 某些池需要重算尾部 |
| `prefill_chunk_align` | 75 | 非末块的对齐粒度 | hybrid 快照只能捐赠在块边界 |
| `page_usage` | 81 | `(used, total)` 页 | |
| `_make_prefix_cache` | 89 | 按 cache_type 建三选一 | hybrid_radix 是内部自动选择，非用户可选 |
| `match_req` | 98 | **前缀匹配** | 调 radix `match`，返回句柄与命中长度 |
| `available_size` / `mamba_available_size` / `swa_available_size` | 114/120/125 | 三种币种的可用量 | hybrid/SWA 各有公式 |
| `ensure_swa_slots` / `ensure_mamba_slots` | 131/143 | 通过驱逐 LRU 节点腾出槽 | 不足则抛/拒绝准入 |
| `snapshot_toolcall_anchor` | 153 | 在 tool-call 锚点冻结 GDN 状态 | 写进 ping-pong 空闲槽 |
| `maybe_free_swa_out_of_window` | 180 | decode 时主动释放滑出窗口的 SWA 槽 | 有节流（`_swa_eviction_interval`，`:16`） |
| `free_swa_out_of_window_extend` | 219 | prefill 的对应版本 | 分配 chunk 前先回收 |
| `lock` / `unlock` | 242/251 | 冻结/解冻一个命中句柄 | 防止准入期间被驱逐 |
| `_free_swa` | 259 | 释放 SWA 槽 | 对同一批索引幂等 |
| `allocate_paged` | 265 | **按页分配并写 page table** | 不足时先从前缀缓存驱逐，再切 free_slots |
| `cache_req` | 286 | **提交前缀** | 分派到 `_cache_req_hybrid`/`_cache_req_swa`/plain |
| `_cache_req_hybrid` | 332 | hybrid 提交：KV + GDN 快照双币种 | |
| `_cache_req_swa` | 421 | SWA 提交 | `node.value` 是 full 池页 |
| `_padded_tail` / `_free_req_slots` | 495/504 | 请求自身页切片 / 释放 GDN 槽 | |
| `check_integrity` | 516 | 断言页守恒与页对齐 | 只在调试/测试路径 |
| `rebuild` | 556 | **原地重建** | 重指 page table、重置计数、**新建前缀缓存**（`RadixPrefixCache.reset()` 是未实现 stub，`:560`）；hybrid 还要 `reclaim_all_slots()` |
| `lazy_free_region` | 575 | 延迟释放区 | 提交期间不真正 `_free` |
| `_allocate` / `_free` / `_page_to_token` | 591/613/617 | 底层页操作 | `_free` 只收每 `page_size` 个索引 |
| `_write_page_table`（模块级） | 625 | 把分配结果写进请求行 | 写**整个尾页**，所以表宽必须对齐 32 |

### 4.3 `prefill.py`

| 符号 | 行 | 作用 |
|---|---:|---|
| `ChunkedReq` | 31 | 中间块请求（`can_decode=False`） |
| `PrefillAdder.__post_init__` | 59 | 由剩余预算初始化 |
| `_kv_reservation_size` | 63 | 额外 KV 页折算成 token |
| `_try_allocate_one` | 70 | 尝试准入一个：radix `match_req` + `lock` + `TableManager.allocate` |
| `_add_one_req` | 135 | 决定整块还是分块：`is_chunked = chunk_size < remain_len`，产生 `ChunkedReq` 或 `Req`，并把续块塞回 `pending_req.chunked_req` |
| `try_add_one` | 232 | 准入主循环（受 token_budget / reserved_size 约束） |
| `PrefillManager.add_one_req` | 282 | 入队 `PendingReq` |
| `PrefillManager.schedule_next_batch` | 294 | 组 prefill 批次 |
| `PrefillManager.abort_req` / `runnable` | 350/358 | 中止 / 是否还有活 |

`_add_one_req`（`:147-199`）的块大小同时受：token 预算、SWA 页跨度、对齐粒度、图像跨度安全（不能把一张图切成两半）约束。

### 4.4 `decode.py` / `table.py` / `io.py` / `status.py` / `mm.py`

| 文件 | 符号 | 行 | 作用 |
|---|---|---:|---|
| `decode.py` | `DecodeManager.filter_reqs`/`remove_req`/`abort_req`/`inflight_tokens`/`schedule_next_batch`/`runnable` | 14-38 | 运行中请求集合的增删与组批 |
| `table.py` | `TableManager.available_size`/`allocate`/`free`/`rebuild` | 14-23 | `table_idx` 与 `token_pool` 的分配器；`rebuild` 重指表并清空 |
| `io.py` | `SchedulerIOMixin.__init__` | 27 | 建 ZMQ 队列；多 rank 时 rank0 用 Pub 广播、其它用 Sub |
| | `sync_all_ranks` | 76 | TP CPU 组屏障 |
| | `_recv_msg_single_rank` / `_multi_rank0` / `_multi_rank1` | 79/88/109 | 三种收消息形态（单 rank / 广播方 / 跟随方） |
| | `_reply_tokenizer_rank0` / `rank1` | 124/132 | 回包；多条时打 `BatchTokenizerMsg` 信封 |
| `status.py` | `SchedulerStatusReporter.report_batch` | 26 | 分派到 prefill/decode 两种日志 |
| | `_report_prefill` | 60 | 打 `#new-seq/#new-token/#cached-token/token usage/#running-req/input throughput` |
| | `_report_decode` | 93 | 每 `decode_log_interval` 打一次 `gen throughput` |
| | `_usage_ratio`/`_mamba_msg`/`_swa_msg` | 127/131/139 | 格式化辅助 |
| `mm.py` | `mm_rows_after`/`mm_chunk_end`/`cut_image_spans`/`plan_mm_chunk`/`plan_mm_batch` | 14-67 | 分块 prefill 下图像跨块的规划 |
| `utils.py` | `PendingReq.input_len`/`output_len` | 25/29 | 准入预算用 |

---

## 5. 关键流程逐段解析

### 5.1 `overlap_loop`（`:207-261`）——全仓最重要的 55 行

```
① self._last_data = last_data                       # 暴露"已发射未排空"的批次
② blocking = 没有任何待处理工作
③ for msg in receive_msg(blocking): _process_one_msg(msg)   # CPU 收消息/准入，与 GPU 上的 N 并行
④ 若完全空闲且有 _pending_rebuild → _execute_pending_rebuild()
⑤ self.stream.wait_stream(self.engine.stream)      # ← 围栏 A
⑥ forward_input = _schedule_next_batch()           # 在 self.stream 上组批 N
⑦ if forward_input:
       with engine_stream_ctx:
           self.engine.stream.wait_stream(self.stream)   # ← 围栏 B（交接）
           _restore_linear_states(batch)                  # GDN COW 必须在引擎流上
           ongoing_data = (forward_input, _forward(forward_input))   # 发射，不等待
⑧ self.stream.wait_stream(self.engine.stream)      # ← 围栏 C
⑨ _process_last_data(last_data)                    # 排空 N-1：唯一的 host block
⑩ _flush_abort_acks()
```

三个围栏的理由（注释里都写了，值得逐条理解）：
- **A**：本次的 H2D token_pool 拷贝要排在上一批采样 token 的 scatter **之后**。否则复用刚释放的 `table_idx` 时，新 prompt 可能被上一任的残留写覆盖。
- **B**：引擎流要等 `self.stream` 上的元数据 staging 完成。
- **C**：排空时的 GPU 可见写（页释放/重指、SWA 哨兵 scatter）要排在刚发射的前向之后，否则可能改写仍在飞的图。

**所以这一步的真实结构是**：CPU 侧组批 N（在 `self.stream`）→ 发射 N（在 `engine.stream`）→ **CPU 排空 N-1**（第 ⑨ 步）。若 D2H 真是异步的，⑨ 的 `copy_done.synchronize()` 会立刻返回，于是"排空 N-1"与"GPU 跑 N"重叠——这就是设计意图。

### 5.2 一条 `UserMsg` → `PendingReq`（`_process_one_msg:488`）
```
UserMsg 分支（:494）
  → 上下文检查（:514）        # 超长/不支持的输入 → ErrorReply
  → max_tokens 钳制（:540）   # 不超过 max_seq_len - 已有长度
  → prefill_manager.add_one_req（:545） → PendingReq
AbortBackendMsg
  → 若该请求在本轮在飞（_last_data 里）→ 只置 req.aborted（:577-584）
  → 否则立即 _free_req_resources
CacheRebuildBackendMsg → 挂 _pending_rebuild
```
abort 的延迟释放是**为了安全**：前向正在读该请求的资源，不能在它下面释放。

### 5.3 分块 prefill（`scheduler_next_batch → PrefillAdder`）
```
PrefillManager.schedule_next_batch（prefill.py:294）
  → PrefillAdder(...)（:44）
  → try_add_one（:232） 循环：
       _try_allocate_one（:70）：radix match_req → lock → TableManager.allocate
       _add_one_req（:135）：
           chunk_size = min(remain_len, token_budget, ...)
           if chunk_size < remain_len:  ChunkedReq(...)   # 还要再来
           else:                        Req(...)
           续块塞回 pending_req.chunked_req
```
`ChunkedReq.can_decode()==False`（`prefill.py:39`）保证中间块**永不采样**——这是分块 prefill 的正确性基石。

### 5.4 `_prepare_batch`（`:786-845`）——每步宿主侧的工作
1. padding（`pad_batch`，让 bs 落到 graph 档位）；
2. `cache_manager.allocate_paged`（给新 token 分配页并写 page table）；
3. `positions` / `mrope_positions`；
4. `page_table[input_mapping]` gather 出本批输入；
5. GDN 的 `linear_table_idx` H2D；
6. `build_fla_metadata`（GDN 的 cu_seqlens 等）；
7. `attn_backend.prepare_metadata`（各后端不同，见 `09`）；
8. `Sampler.prepare`。
**这些全部在 `self.stream` 上**，正是被 overlap 藏起来的那部分。

### 5.5 `_process_last_data`（`:312-429`）——排空
```
copy_done.synchronize()                    # :317 唯一的 host block
with cache_manager.lazy_free_region():     # :320 提交期间延迟释放
  for req in batch.reqs:
      next_token = next_tokens_cpu[i]      # :347
      EOS / stop / 长度 判定                # :353-367
      DetokenizeMsg 入列                    # :374
      完成 → 释放资源 + cache_manager.cache_req(finished=...)  # :386-399 提交前缀
  状态上报                                  # :419
send_result(reply)                          # :429 → ZMQ
```

### 5.6 运行时重建（`:160 rebuild_cache` / `:661 _execute_pending_rebuild`）
```
rebuild_cache（消息到达）
  → 断言没有 runnable 的 prefill/decode；忙 → 立即回 "busy"（不排队）
  → 挂 _pending_rebuild
overlap_loop 的安全点：_pending_rebuild 非空 且 last_data is None 且无 runnable
  → _execute_pending_rebuild
       → engine.rebuild_runtime_cache（两阶段：校验预算 → 销毁 graph → 重建池 → 重捕）
       → cache_manager.rebuild（新前缀缓存）
       → 回成功 / 回滚
```
**重建会清空前缀缓存**，所以"改参数重跑"与"一开始就用该参数"的冷启动命中率不同。

---

## 6. 与其他模块的关系

| 方向 | 模块 | 接口 | 说明 |
|---|---|---|---|
| 使用 | `engine` | `Engine(config)`、`forward_batch`、`rebuild_runtime_cache` | 唯一执行入口 |
| 使用 | `kvcache` | `CacheManager` 构造时建池族/前缀缓存 | 三套 radix + 池族 |
| 使用 | `message` | 收发 `BaseBackendMsg` / `BaseTokenizerMsg` | 跨进程 |
| 使用 | `attention` | `prepare_metadata` / `prepare_for_replay` | 每步 |
| 使用 | `mm` | `plan_mm_batch` 等 | 多模态分块规划 |
| 被使用 | `server/api_server.py` | 构造 `Scheduler`（其实是通过 `server/launch.py` 的子进程） | |
| 被继承 | `llm.LLM` | 覆盖 `offline_receive_msg`/`offline_send_result` | 离线路径 |
| 被观测 | `server/stats.py` | 每个 reply 携带池占用 | `/v1/stats` |

---

## 7. 常见坑

1. **命名方向反直觉**：scheduler 从 tokenizer **收** `BaseBackendMsg`、**发** `BaseTokenizerMsg`。按事实记，不要按"谁是谁的后端"推。
2. **overlap 下"当前批次"有歧义**：可能已发射未排空。任何"为什么资源还没释放"的问题先看这里。
3. **`complete_one()` 在 forward 里调用**，所以日志用 `batch.log_new_tokens`（调度时快照）而不是 live req。
4. **abort 只打标记**（`:577-584`），真正释放在排空后。
5. **`_pending_rebuild` 只在完全空闲时执行**，忙时是"拒绝"而不是"等待"。
6. **`CacheManager.rebuild` 新建前缀缓存**（`:556-573`），不是清空复用。
7. **`lazy_free_region`**（`:575`）：提交期间释放被延迟，否则会把正在 insert 的页还回去。
8. **`_write_page_table` 写整个尾页** → `_page_table_width` 必须对齐（`engine.py:80`），这是 `-1` 汇列能工作的前提。

---

## 8. 自测问题

1. `overlap_loop` 里三个 `wait_stream` 各在防什么？去掉围栏 A 会出现什么症状？
2. `_last_data` 为什么必须存在？它让 abort 的处理复杂在哪？
3. 分块 prefill 的中间块为什么必须 `can_decode=False`？
4. `cache_req` 在什么时刻被调用？为什么要在 `lazy_free_region` 里？
5. 为什么 `PromptAdmittedMsg` 要等 `_prepare_batch` 成功后才发？
6. `rebuild_cache` 忙时为什么回 "busy" 而不是排队？
7. `_status` 日志里的 throughput 是怎么算的？为什么不能用 live req 的长度？
8. `normal_loop` 与 `overlap_loop` 的语义差别，以及什么时候该用前者？
