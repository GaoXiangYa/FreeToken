# 01 · core 与 engine：词汇表、配置、显存预算、CUDA graph、采样

> 行号基于 `377a3bc`。配套完整符号清单见 [`99-symbol-index/engine.md`](99-symbol-index/engine.md) 与 [`top-level.md`](99-symbol-index/top-level.md)。

---

## 1. 模块职责与在系统中的位置

`core.py` + `engine/` 是**执行层的心脏**：它把一份"解析好的配置"变成一个持有 GPU 资源的 `Engine`，并暴露**唯一的执行入口** `Engine.forward_batch(batch, args)`。

- **上游**：`scheduler/Scheduler.__init__` 构造 `Engine`（`scheduler.py:68`），此后每步调用 `forward_batch`。
- **下游**：`models`（模型前向）、`attention`（注意力后端）、`kvcache`（池）、`moe`（专家缓存）。
- **`core.py` 的特殊地位**：它对 `freetoken` 内部**零依赖**（实测 import 出度 0），是依赖图的叶子。所有人 import 它，它不 import 任何人——所以它才敢只定义 4 个类和 3 个函数，却成为全引擎的公共语言。

```
scheduler ──构造/每步调用──▶ Engine ──持有──▶ ctx(Context) ──全局单例──▶ attention/models/kvcache/moe
                                │
                                ├─ EngineConfig（不可变配置）
                                ├─ GraphRunner（CUDA graph）
                                ├─ Sampler
                                └─ 各池：kv_cache / linear_state_pool / moe_offload_cache
```

---

## 2. 文件清单

| 文件 | 行数 | 职责 | 阅读建议 |
|---|---:|---|---|
| `core.py` | 220 | `SamplingParams` / `Req` / `Batch` / `Context` + 全局 ctx | **精读**（全仓词汇表） |
| `engine/config.py` | 159 | `EngineConfig`（不可变配置对象 + `cached_property` 解析链） | 精读 |
| `engine/cache_budget.py` | 134 | 显存预算的**纯算术**策略（无 torch 副作用） | 精读（可单测） |
| `engine/engine.py` | 1,869 | `Engine`：初始化、权重装载、池分配、图捕获、前向、运行时重建 | 按方法选读（**不要顺序读**） |
| `engine/graph.py` | 233 | `GraphCaptureBuffer` + `GraphRunner`（CUDA graph 捕获/replay） | 精读 |
| `engine/sample.py` | 100 | `Sampler`（greedy / top-k / top-p，批内混合） | 精读 |
| `engine/__init__.py` | 6 | 仅 re-export | 略读 |

---

## 3. `core.py`：全引擎共享词汇表

### 3.1 `SamplingParams`（`core.py:18`）

| 字段 | 类型 | 默认 | 含义 |
|---|---|---|---|
| `temperature` | float | 0.0 | ≤0 即 greedy |
| `top_k` | int | -1 | -1 = 不限制 |
| `top_p` | float | 1.0 | 1.0 = 不限制 |
| `ignore_eos` | bool | False | 忽略 EOS 直到 max_tokens |
| `max_tokens` | int | 1024 | 输出预算 |
| `stop_strs` | list[str] | [] | 命中即结束，匹配串及其之后被裁掉 |

派生属性 `is_greedy`（`:29`）：`temperature <= 0 or top_k == 1`。**这一个属性决定整批是否走快速路径**（见 §7.4）。

### 3.2 `Req`（`core.py:34`）——一个请求的全部状态

`@dataclass(eq=False)`（需要可哈希/按身份比较，进入 `Set[Req]`）。

**基础字段**

| 字段 | 含义 |
|---|---|
| `input_ids` | **CPU** 上的 token 张量；实际指向内部 `_ids_buf` 的一段视图 |
| `table_idx` | 该请求在 `page_table` 里的行号（也是 `token_pool` 的槽位） |
| `cached_len` | 已经进 KV 缓存的前缀长度 |
| `output_len` | 输出预算（决定 `max_device_len`） |
| `uid` | 前端分配的用户请求 id |
| `sampling_params` | 采样参数 |
| `cache_handle` | 前缀缓存返回的句柄（`BaseCacheHandle`，锁定期间不可被驱逐） |
| `prompt_len` | 整个 prompt 的长度；0 表示"本请求的 input_ids 就是全部"（分块请求在块结束时才非 0） |

**派生字段**（`__post_init__`，`:70`）：`device_len`、`max_device_len`、`_ids_buf`。
`__post_init__` 里有 3 个断言，它们就是 `Req` 的不变量：

```python
assert self.input_ids.is_cpu                                     # 宿主侧持有
assert 0 <= self.cached_len < self.device_len <= self.max_device_len
```
含义：**已缓存长度必须严格小于已送入设备长度**（否则没有"待计算"的 token），且不超过总预算。第三条隐含在 `_alloc_ids_buf`（`:78`）——一次性分配 `max_device_len` 的缓冲，之后 `input_ids` 只是它的**视图**，这样追加 token 不需要重分配。

**长度语义**（三个属性必须分清，读调度代码时最易混）：

| 属性 | 公式 | 含义 |
|---|---|---|
| `remain_len` | `max_device_len - device_len` | 还能生成多少 token（预算未用尽） |
| `extend_len` | `device_len - cached_len` | **本次前向要算的 token 数** |
| `can_decode` | `remain_len > 0` | 是否还能继续 decode |

**方法**：`complete_one()`（`:91`）前进一步：`cached_len = device_len; device_len += 1`——这就是"采样出的 token 已被算进 KV"。
`append_host(next_token)`（`:95`）把新采样的 token 写进 `_ids_buf` 并滑动视图。
`_alloc_ids_buf`、`__repr__` 为辅助。

**线性注意力 / 滑窗 / 语义锚点字段**（非通用模型为 `None`，读 hybrid 模型时必看）：

| 字段 | 含义 |
|---|---|
| `linear_slot_idx` | GDN（线性注意力）循环状态的槽位号（类比 sglang 的 `mamba_pool_idx`） |
| `mamba_ping_pong` | overlap 调度下可捐赠的两个快照槽位 |
| `mamba_next_track_idx` / `mamba_last_track_seqlen` | 下一次快照的目标槽位 / 上次快照的块对齐长度 |
| `mamba_restore_src` | 前缀命中时，要从哪个树快照 COW 回活槽 |
| `swa_evicted_seqlen` | SWA radix：长度小于此的位置其窗口 KV 已被释放 |
| `decode_batch_idx` | SWA radix：已做过的 decode 前向数（第一次跳过主动释放） |
| `toolcall_anchor_len` | **语义锚点**：模型吐出 tool-call 开头 token 时的状态长度；客户端重写 tool call 后仍可从此复用 |
| `aborted` | overlap 下 abort 到达时只置位，真正释放推迟到批次排空后 |

### 3.3 `Batch`（`core.py:116`）——一次前向的工作集

`phase: Literal["prefill","decode"]` 是全仓最重要的分支条件。

| 字段 | 谁设置 | 含义 |
|---|---|---|
| `reqs` / `padded_reqs` | scheduler | 真实请求 / 补齐到 graph 档位后的请求（补齐用 dummy） |
| `input_ids` / `positions` | scheduler 或 graph buffer | 本次前向的 token 与位置 |
| `mrope_positions` | scheduler | `[3, n]` 多模态 3D rope 位置；此时 `positions` 仍保持序列索引语义 |
| `out_loc` | scheduler 或 graph buffer | 每个 token 的 KV 写入位置 |
| `linear_table_idx` | scheduler / graph buffer | GPU int64 的 per-request 槽位表（GDN decode 无需宿主循环 → 可图捕获） |
| `fla_metadata` | scheduler / graph buffer | GDN 的 cu_seqlens / cache_indices / continuation 标志，全层共享 |
| `active_table_idx` | scheduler | DSV4 的 page-table 行快照（防止下一步 allocate 破坏在飞图） |
| `attn_metadata` | attention backend | `prepare_metadata` 的产物 |
| `mm_embeds` / `mm_rows` | engine | 多模态软 token 及其落点行 |
| `mm_block_ends` | scheduler | 每个 token 所属图像块的结束位置（双向注意力边界） |
| `mm_encoder_jobs` / `mm_gather_plan` | scheduler | 本次要编码的 cache-miss 项 / 散布计划 |
| `log_new_tokens` / `log_cached_tokens` | PrefillManager | **调度时刻快照**：日志用它们，而不是读 live req（因为 forward 里 `complete_one()` 已推进 `cached_len`） |
| `prompt_admissions` | scheduler | `(uid, 完整 prompt 长度, 前缀命中数)`；只在首次 prefill 批次里非空 → 计费恰好一次 |

派生：`is_prefill`（`:162`）、`is_decode`（`:166`）、`get_attn_positions()`（`:169`，mrope 优先）、`size`/`padded_size`。

### 3.4 `Context`（`core.py:182`）——全局执行上下文

| 字段 | 含义 |
|---|---|
| `page_size` | KV 页大小（注意 `page_table` **永远按 page_size=1 记账**） |
| `page_table` | `[max_running_req+1, width]` int32，`[table_idx, pos] → KV 原始槽位` |
| `attn_backend` | 当前注意力后端 |
| `kv_cache` | 分页池实例 |
| `linear_state_pool` | GDN 状态池（非 hybrid 模型为 None） |
| `moe_offload_cache` | 专家卸载缓存（非 offload 策略为 None） |
| `_batch` | 当前正在前向的批次（`forward_batch` 上下文管理器压栈） |

`set_global_ctx` / `get_global_ctx`（`:212`/`:218`）：**全局单例**（只允许设置一次）。`forward_batch` 是 `@contextmanager`，禁止嵌套（`:201` 断言）。这就是为什么模型层可以在任何地方写 `get_global_ctx().attn_backend` 而不必层层传参。

---

## 4. `engine/config.py`：`EngineConfig`

`@dataclass(frozen=True)`（`:22`）——**不可变**，运行期要改就 `dataclasses.replace`。

### 4.1 字段分组

| 组 | 字段（节选） | 说明 |
|---|---|---|
| 模型 | `model_path`、`tp_info`、`dtype`、`hf_overrides` | `hf_overrides` 按 vLLM 语义应用：nested 段逐键更新，其它值整体替换 |
| 调度 | `max_running_req`（4）、`max_seq_len_override` | |
| 注意力 | `attention_backend`（"auto"）、`page_size`（1） | `page_size` 受模型强制：DSV4=128、TRTLLM=16/32/64、SWA=1 |
| MoE | `moe_strategy`、`moe_cache_size`/`rate`/`auto`、`kv_reserve_tokens`（8192）、`moe_cache_policy`（lru）、`moe_prefill_overlap`、`moe_cpu_*`、`moe_hybrid_max_fetch` | |
| 量化 | `quant_backend`（`layer[.kind]=name`） | |
| 显存 | `memory_ratio`（0.9）、`num_page_override`、`num_token_override` | `num_pages` 与 `num_tokens` 互斥 |
| SWA | `swa_full_tokens_ratio`（0.2）、`swa_num_pages_override` | 窗口池大小 = max(working-set floor, ratio × full tokens) |
| GDN | `linear_state_cache_ratio`（2.0） | 快照缓存 = `ceil(ratio × max_running_req)` 额外槽 |
| 多模态 | `mm: MultimodalConfig` | |
| 杂项 | `use_dummy_weight`、`use_pynccl`、`distributed_timeout`、`ple_backend`、`swa_decoder_replay`、`expert_load` | |

`__post_init__`（`:100`）只做一件事：把废弃的 `moe_backend` 折叠进 `moe_strategy`，两者同时给则报错。

### 4.2 解析链（`cached_property`，惰性且只算一次）

| 属性 | 行 | 作用 |
|---|---:|---|
| `hf_config` | 110 | `cached_load_hf_config(model_path, hf_overrides)` → HF config |
| `model_spec` | 114 | `get_model_spec(hf_config.architectures[0])` → 注册表查表 |
| `active_encoders` | 118 | 本进程真正会构建的编码塔（家族声明 ∩ checkpoint 有该 section ∩ 未被 `--mm-disable`） |
| `served_modalities` | 128 | 由 `active_encoders` 推出，暴露给 `/v1/stats` |
| `model_config` | 133 | **核心**：把不构建的编码器 section 置 None → `checkpoint_quant_config` → `set_quant_config` → 调家族的 `parse_config` |
| `max_seq_len` | 147 | 覆盖优先，否则 `rotary_config.table_positions` |
| `distributed_addr` | 157 | `tcp://127.0.0.1:2333` |

**关键点**：`model_config`（`:133-144`）是一个"副作用式"属性——它调用 `set_quant_config(quant)` 把量化配置装成进程全局。所以**第一次访问 `config.model_config` 会锁定量化方案**。

---

## 5. `engine/cache_budget.py`：显存预算（纯算术）

设计前提：**整个文件只有整数/字节运算，不 import torch、不碰 GPU**，因此可无 GPU 单测（`tests/engine/test_cache_budget.py`）。

| 函数 | 行 | 作用 | 公式 / 关键行为 |
|---|---:|---|---|
| `expert_bytes_per_slot(sources)` | 17 | 一个专家槽占多少 GPU 字节 | 按 bank 求和 `t[0][0].numel() * element_size()`；**不含** marlin/b12x 的固定 alpha（`[L*E]` 常驻，不随 cache_size 增长） |
| `net_cache_budget_bytes(ratio, baseline_free, weights, fixed)` | 31 | 可用于 MoE+KV 的净字节 | `int(ratio*baseline_free) - weights - fixed`。`(1-ratio)` 余量留给 CUDA graph/激活。**自称 single source of truth**（启动 sizing 与 rebuild 的 fit 检查都该用它） |
| `required_bytes(slots, pages, per_expert, per_page)` | 41 | 一个几何配置占多少 | `slots*per_expert + pages*per_page` |
| `plan_cache_budget(...)` | 48 | **MoE 优先的切分** | ① 先算 `hi = min(total_experts, max_slots)`；② `overlap = prefill_overlap and hi >= 2*num_experts`，`lo = 2E if overlap else E`；③ 预留 `kv_reserve_pages`，剩余按 per_expert 贪心给专家；④ KV 拿剩下的；⑤ 断言总和不超预算、`num_pages > 1` |
| `resolve_moe_cache_auto(...)` | 98 | `--moe-cache-auto` 的入口 | `net_cache_budget_bytes` **只应用一次 ratio**，再委托 `plan_cache_budget` |

**为什么 `lo = 2E`**：prefill 双缓冲要借用两个完整的专家层缓冲（见 `06` 的母题 4）。装不下就关掉 overlap 并把下限降回 `E`（`:72-73`）。
**为什么"MoE 优先"**：注释与代码一致——先预留 KV 地板，专家贪心吃满剩余。这使 `--moe-cache-auto` 天然偏向降低 miss 率。

---

## 6. `engine/graph.py`：CUDA graph

### 6.1 `GraphCaptureBuffer`（`:23`）——一组**静态**张量

| 字段 | shape | 说明 |
|---|---|---|
| `input_ids` / `out_loc` / `positions` | `[max_bs]` int32 | 每步用 `copy_from` 刷新 |
| `mrope_positions` | `[3, max_bs]` int32 或 None | 仅 mrope 模型分配 |
| `logits` | `[max_bs, vocab]` **fp32** | 注意显存：160×150k×4B ≈ 96 MB |
| `table_idx` | `[max_bs]` int32 | GDN 状态 gather/scatter 的槽位表 |
| `fla_cu_seqlens` | `[max_bs+1]` int32 | **常量 `arange`**，抓一次就不再变 → 省掉每步 H2D |

方法：
- `init(bs, vocab, device, mrope)`（`:35`）分配上述张量。
- `set_batch(batch)`（`:50`）**捕获时**把 buffer 的切片绑定进 batch（`batch.input_ids = self.input_ids[:bs]` 等），并构造 `FLAMetadata`。
- `copy_from(batch)`（`:67`）**每次 replay 前**把真实值拷进去（≤5 次 D2D）。

### 6.2 `GraphRunner`（`:105`）

| 方法 | 行 | 作用 |
|---|---:|---|
| `__init__` | 106 | 定 bs 阶梯 → 调 `_capture_graphs` |
| `_reset_moe_offload_cache` | 136 | 捕获前后重置专家缓存（保证捕获轨迹确定） |
| `_capture_graphs` | 140 | 见下方流程 |
| `can_use_cuda_graph(batch)` | 204 | `batch.is_decode and batch.size <= max_graph_bs` |
| `replay(batch)` | 207 | `copy_from` → 选 `graph_map[padded_size]` → `prepare_for_replay` → `g.replay()` → 返回 `logits[:size]` |
| `pad_batch(batch)` | 215 | 选**第一个 ≥ size 的已捕获档位**，用 dummy_req 补齐 |
| `destroy_cuda_graphs` | 224 | 丢引用 + `gc.collect()`（注释强调"drop references is the load-bearing step"），且必须在释放 NCCL 之前 |

`_determine_cuda_graph_bs`（`:79`）：显式列表优先；否则 `[1,2,4] + range(8, max+1, 8)`，`max` 默认 >80 GB 显存取 256、否则 160。

**捕获流程**（`:140-202`）：
1. `emit_progress("Capturing CUDA graphs / warming up", 0, 0)` —— 给 `/health` 一个不确定进度阶段；
2. `attn_backend.init_capture_graph(max_seq_len, bs_list)`；
3. `synchronize` → `empty_cache` → `reset_peak_memory_stats`；
4. 分配 `GraphCaptureBuffer.init(max_bs, ...)`；
5. **按 bs 从大到小**遍历：eager 前向一次（warmup，把懒分配/autotune 全逼出来）→ `with torch.cuda.graph(graph, pool=pool)` 再前向一次（捕获）→ reset 专家缓存 → **复用第一个 graph 的 mempool**；
6. 结束后再 reset 一次。

---

## 7. `engine/sample.py`：采样

### 7.1 `BatchSamplingArgs`（`:14`）
`temperatures` / `top_k` / `top_p` / `greedy_mask`，全部是 GPU 张量；`temperatures is None` 是"**整批 greedy**"的哨兵。

### 7.2 `Sampler.prepare(batch)`（`:59`）
- 全 greedy → `BatchSamplingArgs(temperatures=None)`，**不为任何 per-row 参数分配张量**（`:62-63`）。
- 混合 → 对 greedy 行填中性参数（temp 1.0 / top_k=vocab / top_p=1.0），并建 `greedy_mask`；`make_device_tensor` 用 `pin_memory=True` + `non_blocking`（`:21-22`）。

### 7.3 `Sampler.sample(logits, args)`（`:88`）
- `temperatures is None` → `torch.argmax`（`:92`）；
- 否则 `sample_impl`：flashinfer 优先，否则 Triton 回退；softmax（SM90+ 带 PDL）+ 四种采样之一（`:38-51`）；
- 有 `greedy_mask` 时**再跑一次 `argmax`** 并用 `torch.where` 覆盖（`:97`）——所以混合 batch 的成本 > 纯采样 batch。

### 7.4 性能含义（重要）
一个非 greedy 请求会让整批走概率采样路径，且多一次 argmax。**这是真实存在的性能悬崖**，与 `Sampler.prepare` 每步新建参数张量叠加。

---

## 8. `engine/engine.py`：`Engine`

### 8.1 `Engine.__init__`（`:331`）的启动顺序

| # | 步骤 | 行 | 说明 |
|---|---:|---|---|
| 1 | 断言 CUDA 未初始化 | 332 | 保证进程内只有一个 Engine |
| 2 | `set_tp_info` | 333 | 装分布式全局信息 |
| 3 | `set_quant_backend(_adjust_ftw_quant_backend(...))` | 334 | FTW checkpoint 会补全 `--quant-backend` 的自动项 |
| 4 | `_ensure_expandable_segments()` | 335 | **必须在第一次 CUDA 分配之前** |
| 5 | `bind_assigned_gpu(rank)` | 339 | 按 `--gpu` 绑定 |
| 6 | `_adjust_config(config)` | 340 | **配置收口**（`:1510`） |
| 7 | 建 `self.stream` 并设为当前流 | 342-343 | 引擎流；scheduler 另有自己的流 |
| 8 | `resolve_pool_class` + `Context` + `set_global_ctx` | 348-352 | 全局上下文在此建立 |
| 9 | `_init_communication` | 353 | TP 进程组 / pynccl |
| 10 | 量显存：`init_free_memory = free_max`（启动 sizing 用跨 rank **MAX**）、`_baseline_free = free_min`（rebuild 基线用 **MIN**，保证各 rank 一致） | 354-357 | |
| 11 | 模型初始化 + 权重装载（`_load_weights`） | 360+ | |
| 12 | 计算 KV 预算 → `_pool_cls.solve_num_pages` → `create_kv_pool` | 417-427 | 注意这里用的是 `_startup_kv_budget`，**不是** `cache_budget.py`（见 §10 坑 1） |
| 13 | `LinearStatePool`（GDN） | 429-444 | |
| 14 | `page_table` 分配 + `attach_page_table` | 446-458 | |
| 15 | `create_attention_backend` | 461-463 | |
| 16 | `Sampler` | 466 | |
| 17 | `dummy_req` + `GraphRunner`（含图捕获） | 469-497 | 图捕获是启动最慢的一段 |
| 18 | 若 prefill 后端是 triton → `_warmup_prefill()` | 498-500 | 预热 prefill 的 autotune 缓存 |

### 8.2 模块级辅助函数（`:41-324`）

| 函数 | 行 | 作用 | 关键行为 |
|---|---:|---|---|
| `_require_offload_cache_size` | 41 | 校验卸载缓存下限 | 至少每层每专家一个槽 |
| `_flashinfer_available` / `_sgl_flash_attn_available` | 54/60 | 后端可用性探针 | 供自动选型用 |
| `_startup_kv_budget` | 73 | 启动期 KV 预算 | `ratio × 预载前空闲 - 模型后空闲` 的形状；**第二份实现**（对比 §5） |
| `_page_table_width` | 80 | page table 列数 | 对齐到 32 的倍数（`_write_page_table` 会写整个尾页） |
| `_required_attn_types` | 88 | 本模型需要的 AttnType 集合 | 由 `kv_cache_group_specs()` 走一遍得到 |
| `_backend_parts_serve` / `_backend_requirements_met` | 104/111 | 后端能力判定 | 逗号分段 + 包/架构探针 |
| `_resolve_auto_attention_backend` | 124 | **自动选后端** | 按 AttnType 的候选顺序取第一个满足条件的；FULL-only 顺序 = `trtllm`(sm100) → `fa,fi`(sm90) → `fi` → `triton`；SWA-only 直接 `triton` |
| `_validate_attention_backend_choice` | 163 | 显式选择的校验 | 类型×能力不匹配就报错（含 trtllm 需 sm_100a cubin 的理由） |
| `_make_dummy_weight_state_dict` | 251 | `--dummy-weight` 用 | 不读 checkpoint，只造同形状张量 |
| `WeightLoadError` | 280 | 加载失败的专用异常 | 区分"checkpoint 读不了"与资源/配置失败 |
| `_is_resource_failure` / `_weight_load_context` | 284/295 | 失败归类与日志前缀 | 让 `/health` 的失败原因以 `WeightLoadError` 开头 |
| `_materialize_loaded_weight_state_dict` | 305 | 物化权重 | |
| `_profile_gpu` | 1133 | 取 GPU (name, uuid) | 供 bench 剖面 key |
| `_is_unified_memory_gpu` | 1141 | 是否统一内存（GB10/DGX Spark） | 是则 auto→fused |
| `_fused_resident_ok` | 1165 | fused 策略能否装下 | GGUF q4_0 与 DSV4 专家只能 offload |
| `_ensure_expandable_segments` | 1172 | 配 CUDA 分配器 | 减少碎片 |
| `_resolve_cache_type` | 1198 | 线性注意力模型强制某些 cache type | |
| `_adjust_dsv41_config` / `_adjust_dsv4_config` | 1207/1240 | 两个 DSV 模型的配置调和 | 在池存在之前调 |
| `_parse_cpu_layers_spec` / `_resolve_cpu_layers` / `_decode_target` / `_cpu_moe_executor_viable` | 1288/1317/1333/1349 | `--moe-cpu-layers` 的解析与判定 | 支持 id 列表 / 数量 / 比例 / auto |
| `_pin_budget_bytes` / `_bank_bytes` / `_pin_hint` / `_check_pin_budget` | 1370-1409 | 钉住内存预算 | 非 WSL 平台不设上限（返回 None） |
| `_auto_cpu_layers` | 1411 | `auto` 的层选择 | bank 装得下 pin 预算就全走 GPU |
| `_adjust_ftw_quant_backend` | 1453 | FTW 的 bank kernel 补全 | 标志与 FTW 打包内核冲突时以 FTW 为准 |
| `shared_offload_method` / `offload_expert_method` | 1479/1495 | 取模型用的专家方法 | meta-device 构建一次 |
| `_adjust_config` | 1510 | **归一化总收口** | 见 §9 |

### 8.3 `Engine` 的方法

| 方法 | 行 | 作用 | 注意 |
|---|---:|---|---|
| `_init_communication` | 503 | TP 进程组（pynccl 可选） | |
| `_load_weights` | 530 | 权重装载入口 | 内部转 `_load_weight_state_dict` |
| `_load_weight_state_dict` | 546 | 走 `models/weight.py:load_weight`，或 FTW / dummy | |
| `_load_resident_experts` | 564 | fused 策略下把专家装进 GPU bank | |
| `_warmup_encoders` / `_run_mm_encoder` | 579/586 | 多模态编码器预热 / 编码 cache-miss 项并写入 `batch.mm_embeds` | 在前向之前调用 |
| `_resolve_auto_moe_cache_size` | 611 | `--moe-cache-auto` → `(slots, pages, overlap)` | 委托 `cache_budget.py`；会把兄弟 GDN 状态池的字节也扣掉（`:620`） |
| `_init_offload_moe_cache` | 638 | 建 `OffloadMoeCache`、装 bank、分配槽 | |
| `_resolve_hybrid_fetch` | 766 | `--moe-hybrid-max-fetch -1` → 带宽匹配的 fetch 分式 | 读 `bench_profile`；无剖面则回退固定 cap=1 |
| `_init_cpu_moe_executor` | 797 | 建 CPU 执行器 | |
| `_sync_get_memory` | 835 | 跨 rank 取 min/max 空闲显存 | 返回元组，两个用途不同（见 §8.1 第 10 步） |
| `_target_moe_and_expert_bytes` | 856 | 给 rebuild 用的目标字节 | |
| `_resize_kv_pool` | 870 | rebuild 时重分配 KV 池 | |
| `_refresh_seq_state` | 881 | rebuild 后刷新序列状态 | |
| `rebuild_runtime_cache` | 898 | **运行时原地重建**：两阶段（先校验预算→再销毁 graph→重建池→重捕） | idle-only；会新建前缀缓存 |
| `forward_batch` | 1044 | **唯一执行入口** | 见 §9.3 |
| `_warmup_prefill` | 1067 | 预热 prefill 的 Triton autotune | |
| `shutdown` | 1127 | 清理 | 必须在释放 NCCL 前销毁图 |

---

## 9. 关键流程逐段解析

### 9.1 启动：从 `--model` 到"可以前向"
```
server/args.py:parse_args → ServerArgs
  → Scheduler.__init__ → Engine(config)
     → _adjust_config                      # 归一化
     → set_global_ctx(Context)             # 全局上下文
     → _load_weights → models/weight.load_weight
     → _startup_kv_budget → solve_num_pages → create_kv_pool
     → LinearStatePool(可选)
     → page_table + attach_page_table
     → create_attention_backend
     → Sampler
     → GraphRunner → _capture_graphs       # 启动最慢
     → _warmup_prefill (triton prefill)
```

### 9.2 `--moe-cache-auto` 的显存切分
```
Engine._resolve_auto_moe_cache_size
  → expert_bytes_per_slot(bank_sources)          # 一个槽多少字节
  → resolve_moe_cache_auto(...)                  # cache_budget.py:98
       → net_cache_budget_bytes(ratio, baseline, weights, fixed)
       → plan_cache_budget(...)                  # MoE 优先切分
  → 返回 (moe_cache_size, num_pages, prefill_overlap)
```
`plan_cache_budget` 内部顺序：**预留 KV 地板 → 专家贪心吃满 → KV 拿剩余 → 断言不超预算**。断言失败会在算术层报错，而不是等到 CUDA OOM（`:89-93`）。

### 9.3 一个 decode step（`forward_batch`，`:1044`）
```python
assert torch.cuda.current_stream() == self.stream
if batch.mm_gather_plan: self._run_mm_encoder(batch)          # ① 多模态编码
use_graph = self.graph_runner.can_use_cuda_graph(batch)      # ② 选 graph/eager
with self.ctx.forward_batch(batch), self.model.forward_host_ctx(batch, use_graph):
    logits = self.graph_runner.replay(batch) if use_graph else self.model.forward()
if self.cpu_moe_executor is not None: self.cpu_moe_executor.raise_if_unhealthy()
for req in batch.reqs: req.complete_one()                    # ③ 推进 cached_len
batch_logits = logits[: batch.size]
next_tokens_gpu = self.sampler.sample(batch_logits, args).to(torch.int32)   # ④ 图外采样
next_tokens_cpu = next_tokens_gpu.to("cpu", non_blocking=True)             # ⑤ D2H
copy_done_event = torch.cuda.Event(); copy_done_event.record(self.stream)   # ⑥ 事件
return ForwardOutput(next_tokens_gpu, next_tokens_cpu, copy_done_event)
```
- `@torch.inference_mode()` 装饰整个方法（`:1066`）。
- `forward_host_ctx`（`models/blocks.py:33`）是"主机喂数"的钩子：DSV4.1 Engram / Qwen4Exp PLE 在这时把行读进 pinned staging。
- **采样刻意留在图外**（`:1049-1050` vs `:1059-1060`），代价是每步宿主侧准备。

### 9.4 图捕获 → replay
捕获（`graph.py:167-198`）：倒序 bs → eager warmup → `torch.cuda.graph()` 捕获 → reset 专家缓存 → 共享 mempool。
replay（`graph.py:207-213`）：`copy_from`（≤5 次 D2D）→ `graph_map[padded_size]` → `prepare_for_replay`（3~4 次 D2D）→ `g.replay()`（**一次 `cudaGraphLaunch` 覆盖 embed+所有层+LM head**）→ 返回静态 logits 切片。

### 9.5 `_adjust_config`（`:1510`）——最后的归一化收口
它把"用户意图"变成"引擎可执行"：dense 模型丢弃 MoE 参数、单流模型折叠、DSV4/4.1 特化、`cuda_graph_max_bs` 钳到 `max_running_req`（`:1230-1237`，超了只警告）、`--moe-cache-policy` 等选择项校验、`_decode_target` 决定 CPU/GPU/hybrid 目标。**任何"我传了参数但没生效"的问题，先来这里查。**

---

## 10. 常见坑

1. **两份 KV 预算实现**：`engine/cache_budget.py` 自称 single source of truth，但**普通启动路径实际走** `_startup_kv_budget`（`engine.py:73`）+ `_pool_cls.solve_num_pages`（`:423`），是第二份实现。读 `memory_ratio` 相关代码时先确认读的是哪一份。
2. **`cuda_graph_max_bs` 被 `max_running_req` 钳住**（`:1230-1237`）。调大并发才能放开 graph 档位；反之想省显存可以降并发。
3. **`Req` 的三个长度语义**（`cached_len` / `device_len` / `max_device_len`）与三个属性（`remain_len` / `extend_len` / `can_decode`）必须分清，这是读调度代码的第一道坎。
4. **`config.model_config` 是带副作用的缓存属性**（`set_quant_config`），第一次访问即锁定量化方案；不要在别处重复构造。
5. **`complete_one()` 在 forward 内部调用**，所以日志必须用 `batch.log_new_tokens` 快照而不是 live req（`status.py:74-78` 有解释）。
6. **图捕获必须在释放 NCCL 之前销毁**（`graph.py:223`）；rebuild 的 "drop references" 是回收显存的必要条件而非可选优化。
7. **`_is_unified_memory_gpu` 会把 auto 解析成 fused**——在 GB10/DGX Spark 上，offload 只是同一块内存的两个名字。

---

## 11. 自测问题

1. `Req.extend_len` 与 `remain_len` 分别回答什么问题？在分块 prefill 的续块里，哪个会变、哪个不变？
2. 为什么 `core.py` 对内部模块零依赖这件事很重要？如果它 import 了 `models` 会破坏什么？
3. `plan_cache_budget` 为什么是"MoE 优先"而不是"KV 优先"？`lo = 2E` 的来源是什么？
4. CUDA graph 捕获前那次 eager 前向是干什么的？为什么捕获前后都要 reset 专家缓存？
5. `cuda_graph_max_bs` 和 `max_running_requests` 谁钳谁？为什么？
6. 一个 batch 里混入一个非 greedy 请求，会让整批多付出哪些成本？
7. `forward_host_ctx` 解决的是哪一类问题？举一个使用它的模型。
8. `_adjust_config` 之前和之后，同一份 config 的语义差别是什么？
