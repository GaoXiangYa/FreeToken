# 06 · moe：专家卸载缓存、LRU 准入、PCIe gather、双缓冲 prefill、CPU/混合执行

> 行号基于 `377a3bc`。完整符号清单见 [`99-symbol-index/moe.md`](99-symbol-index/moe.md)。

---

## 1. 模块职责与在系统中的位置

四种 MoE 策略里，本模块负责**后三种**并复用第一种的原语：

| 策略 | 专家常驻在哪 | 未命中怎么办 | 本模块的角色 |
|---|---|---|---|
| `fused` | GPU HBM（全部） | 不会未命中 | 复用 `fused.py` 的 `fused_topk`/`moe_align_block_size` |
| `offload` | **主机 pinned RAM** + GPU slot 缓存（LRU） | 走 PCIe 抓取 | `OffloadMoeCache` + `fast_index_copy` |
| `cpu` | 主机 RAM | CPU 算 | `CpuMoeExecutor` |
| `hybrid` | 主机 RAM | 部分抓取、部分 CPU 算（overlap） | 两者 + 带宽匹配分式 |

**为什么存在**：一个 MoE 模型有 100B-300B 参数，但每 token 只激活 3-10B。把非激活的专家放主机内存，GPU 只留一个 LRU 槽位池，就能在单卡上跑起来。代价是未命中要走 PCIe（实测 ~31 GB/s vs HBM ~3 TB/s，**每字节约差 100 倍**）。

```
                   ┌──────────── GPU ────────────┐
bank_sources ──▶  │ bank_caches[bank][cache_size] │ ← LRU slot 缓存（全层共享）
(pinned host,     │        ↑                     │
 每层一个张量)     │   fast_index_copy (gather)   │
                   └────────┼─────────────────────┘
                            │ 未命中 num_indices × row_bytes
                   ┌────────┴─────────────────────┐
                   │ pinned host banks (每层)      │
                   └──────────────────────────────┘
                            ▲
                   GEMM 按 topk_ids（已被改写成 slot id）取行
```

---

## 2. 文件清单

| 文件 | 行数 | 职责 | 阅读建议 |
|---|---:|---|---|
| `offload_cache.py` | 1,070 | `OffloadMoeCache`：主机 bank ↔ GPU slot 缓存、LRU 簿记、双缓冲、拷贝计划、统计 | **精读** |
| `cpu_executor.py` | 697 | `CpuMoeExecutor`：pinned IO、worker 池、flag 握手、ISA 分派、看门狗 | 精读 |
| `benchbw.py` | 999 | `ft bench bw` 的五个测量 + 裁决 | 读 `measure_*` 与 `recommend` |
| `host_banks.py` | 488 | `HostBank`（分配/pin/mlock/release）、`HostResidency`、`PinPipeline` | 精读 |
| `expert_banks.py` | 440 | `load_expert_banks` 分派、`ExpertBanks`、q4_0 provider、FTW 快路径 | 概览 |
| `offload_kernels.py` | 431 | LRU 委托（`ensure_experts`）、hybrid 三阶段 kernel、prefill 命中压缩 | **精读** |
| `fused.py` | 381 | resident 路径原语：`fused_topk`、`moe_align_block_size`、tuned config 查找、`fused_experts_impl` | 精读 |
| `expert_pieces.py` | 157 | checkpoint 张量 → 逐专家 piece；层 ↔ bank 层映射 | 概览 |
| `fused_nvfp4.py` / `fused_mxfp4.py` / `fused_ds_fp4.py` | 336 / 282 / 234 | 各格式的专家 GEMV / grouped GEMM | 按需 |
| `fused_fp8_block.py` / `fused_q4_0.py` | 48 / 53 | 同上（FP8 block / GGUF Q4_0） | 按需 |
| `bench_profile.py` | 191 | **torch-free** 的 profile 读取器：backend 推荐 + fetch 分式 | **精读** |
| `legacy_format.py` | 46 | `(kind,kernel)` ↔ 量化格式标签 | 略读 |
| `configs/` | — | tuned grouped-GEMM 配置（只有 H100 + triton 3.5.1 一份） | 略读 |

---

## 3. 核心数据结构

### 3.1 `OffloadMoeCache` 字段（`offload_cache.py:105`，全部在 `__post_init__:148` 建）

| 字段 | shape | dtype | 含义 |
|---|---|---|---|
| `slot_for_id` | `[num_layers, num_experts]` | int32 | 专家扁平 id → GPU 槽位；`-1` = 不在 |
| `id_of_slot` | `[cache_size]` | int32 | 反向映射；`-1` = 空槽 |
| `usage` | `[cache_size]` | int64 | 时间戳 LRU（`usage = step`） |
| `step` | `()` | int64 | 每次 `ensure_experts` 自增 |
| `active_mask` | `[num_experts]` | int32 | **不按层索引**，每层被覆盖写后立即读（hybrid 用） |
| `evict_slots` / `src_indices` | `[max(E, cache_size)]` | int32 | 本次要驱逐的槽 / 要拷贝的源行 |
| `num_indices` | `[1]` | int64 | 本次抓取数（cap 后） |
| `num_missing_full` | `[1]` | int64 | cap 前的未命中数（hybrid 统计） |
| `expert_recency` | `[num_layers, num_experts]` | int64 | hybrid 的专家近因 |
| `lru_stats` | `[num_layers, N_STATS]` | int64 | 内核内累加（可随图重放） |
| `decode_freq` | `[num_layers, num_experts]` | int64 | 可选路由直方图（**非图安全**） |
| `bank_sources` | `{bank: [L 个 [E,...] 张量]}` | — | 主机端 pinned bank（每层一个） |
| `bank_caches` | `{bank: [cache_size, ...]}` | — | GPU 端槽位池（**每 bank 一个，全层共享**） |

**不变量**：
- `cache_size >= 2 * num_experts`（prefill 双缓冲要借两整层，`:164-168` 断言）；
- 专家扁平 id = `layer_id * num_experts + expert`（`:176-178`），所以反查不需要解码；
- `bank_caches` 全部 bank 共用同一批 slot 下标（槽位是"跨 bank 的一条记录"）。

### 3.2 `_BANK_SCHEMAS`（`:37-79`）与 `_BANK_BYTES_PER_EXPERT`（`:87-97`）

每个量化格式需要哪些 bank：

| 格式标签 | banks | 备注 |
|---|---|---|
| `bf16` | gate_up, down | |
| `fp8_block` | gate_up, gate_up_scale, down, down_scale | 128×128 块 scale |
| `q4_0` | gate_up, down | 打包块字节，核内反量化 |
| `nvfp4` | gate_up_packed/scale/global, down_packed/scale/global | 6 个；w1/w3 各自 global |
| `nvfp4_marlin` / `nvfp4_b12x` | 4 个（global 折进 block scale，变常驻 alpha） | |
| `mxfp4_triton` | 6 个（blocks/scales/bias × 2） | transposed split-K 布局 |
| `ds_fp4` | 4 个（无 global） | DeepSeek-V4 专用 |

每槽字节公式（`_BANK_BYTES_PER_EXPERT`）决定 `--moe-cache-size` 的显存成本，例如 bf16 是 `3·I·H·2`，nvfp4 约 `0.25×`。

### 3.3 `HostBank` / `HostResidency` / `PinPipeline`（`host_banks.py`）

| 符号 | 行 | 含义 |
|---|---:|---|
| `HostBank` | 78 | page 对齐缓冲；`nbytes` 向上取整到 4096（O_DIRECT 要求）但 `tensor` 视图按精确 `nbytes` |
| 两种 backing | 60-110 | 默认惰性匿名 `mmap`；`FREETOKEN_BANK_CUDA_ALLOC` 时用 `cudaHostAlloc` 直接"生来就 pin" |
| `pin()` | 130 | `cudaHostRegister`（把已驻留页锁住） |
| `lock()` | 154 | `mlock` + 提升 `RLIMIT_MEMLOCK` |
| `release()` | 146 | `MADV_DONTNEED` |
| `HostResidency` | 43 | `PINNED` / `LOCKED` / `PAGEABLE`；**只有 PINNED 能喂 GPU 搬运路径**，其它两种必须 CPU 解码 |
| `PinPipeline` | 286 | 单后台线程串行处理 `cudaHostRegister`（驱动内部串行），并统一做 mlock（配额簿记非线程安全） |

**pin-after-fill 的理由**（`host_banks.py:1-16`）：**先填数据再注册**只锁页；若先注册一个惰性 mmap，注册时会把每一页都缺页+清零（注释举例 DSV4 ~137 GiB → ~47 s），而这次填充马上会被覆盖。

### 3.4 `ExpertBanks` / `ExpertView`（跨模块契约）
- `ExpertBanks`（`expert_banks.py:30`）由**量化内核拥有布局**（`BankSpec`），`load_expert_banks`（`:282`）负责填充/打包，FTW 有快路径。
- `ExpertView`（`layers/moe.py`）是"GEMM 看到的东西"：`(banks, slots=topk_ids)`——用 slot 索引行，而不是专家 id。

### 3.5 `benchbw` profile（JSON `version 4`）
`$XDG_CACHE_HOME/freetoken/benchbw/<gpu-uuid>.json`，字段：
- `gpu: {name, uuid}`（读时按 **name** 校验）
- `ceilings: {cpu_stream_read_gbs, ...}`
- `dtypes: {fmt: "hybrid"|"offload"}`（`--dtype` 的裁决）
- `dtype_kernels: {fmt: {cpu_moe_gbs, pcie_gather_gbs, cpu_moe_overlap_gbs, pcie_gather_overlap_gbs, recommended}}`
- `workloads: {...}`（`--model` 的逐 workload 明细）

---

## 4. 函数详解

### 4.1 `offload_cache.py`

| 方法 | 行 | 作用 | 关键行为 |
|---|---:|---|---|
| `__post_init__` | 148 | 建全部 device 张量与 bank 池 | 断言 `cache_size >= 2E` 等 |
| `set_bank_sources` | 294 | 绑定主机 bank | 同时按 residency 校验能否走 GPU 路径（`:322-340`） |
| `ensure_experts` | 846 | **准入**：把本层专家变常驻，并**原地**把 `topk_ids` 改写成 slot id | 调 `offload_kernels.ensure_experts`（外部 flashlib）；`collect_decode_freq` 时先打直方图 |
| `ensure_experts_hybrid` | 858 | hybrid 的 capped-fetch 准入 | 走仓库内 Triton kernel |
| `copy_missing` | 1014 | **抓取未命中** | 优先 fused 多 bank 一次 launch（`:1030-1046`）；`_copy_fused_ok` 为假才 per-bank（`:1048-1057`） |
| `bank_views` | 601 | 给 GEMM 的视图 | 可只取前 `num_experts` 个（非 overlap 路径） |
| `alphas_for_slots` | — | marlin/b12x 的常驻 alpha | 按 slot 取 |
| `prefetch_prefill_layer` | 671 | **双缓冲预取** | `buffer_id = layer_id % 2`；幂等；记录 ready 事件 |
| `wait_prefill_layer` | 821 | 等某层就绪并返回缓冲视图 | 计算流 `wait_event` |
| `release_prefill_layer` | 835 | 释放缓冲 | 记录 release 事件 |
| `materialize_layer` | 879 | 非 overlap 路径：整层同步拷贝 | |
| `_prefetch_split` | 749 | **命中/未命中分离** | 命中→设备侧 gather；未命中→一次批量 memcpy |
| `begin_prefill` | 648 | 每 chunk 一次的起始围栏 | 让 copy 流等计算流，防止借用槽被 decode 读 |
| `_hit_d2d_usable` | 706 | hit-D2D 可用性判定 | 不可用时**只警告一次**并回落（`:730-736`） |
| `reset` | — | 清空缓存状态 | 图捕获前后调用 |
| `record_decode_stats` | 907 | 旧的统计入口 | **已成 no-op**（改为核内累加 `lru_stats`） |
| `iter_offload_moe_layers` / `attach_offload_moe_cache` | 1060 / 1066 | 遍历/挂载到模型 | |

### 4.2 `offload_kernels.py`

| 函数 | 行 | 作用 |
|---|---:|---|
| `ensure_experts` | 19 | **委托外部 `flashlib.kernels.slot_cache.lru_ensure`**：传入 `expert_ids` 同时作为 `out_indices`（原地改写），并用 `id_base` 把层内 id 映射到全局 id 空间 |
| `_ensure_experts_hybrid_cpu` | 128 | hybrid 的 **CPU 镜像**（docstring 声称决策与 GPU 逐位一致） |
| `_ensure_experts_hybrid_kernel` | 290 | **hybrid 三阶段**：Phase1 活跃/未命中 + Q16 分摊；Phase2 驱逐选槽；Phase3 重写 slot / `-1` |
| `prefill_hit_compact` | 63 | prefill 时把命中行压紧成 gather 源 |
| `_decode_split_count` 等 | — | split 目标 program 数 |

### 4.3 `cpu_executor.py`

| 方法 | 行 | 作用 |
|---|---:|---|
| `__init__` / `PinPipeline` 接线 | — | 建 pinned IO、flag slots、worker 池 |
| `decode(layer_id, hidden, weights, ids)` | — | 一次 CPU 专家前向 |
| `submit` / `sync` | — | 两条路径：**stream memop**（默认，`cuStreamWriteValue64`/`cuStreamWaitValue64`）或 `cudaLaunchHostFunc` 回退 |
| `raise_if_unhealthy` | — | 看门狗：flag 挂起 ≥10 s 且协调器计数未前进才判死，抛大声错误 |
| `pick_isa`（C++ 侧） | — | scalar/avx2/avx512/avx512bf16，可用 `FREETOKEN_CPU_MOE_ISA`/`_SCALAR` 降级 |

### 4.4 `fused.py`（resident 路径原语）

| 函数 | 行 | 作用 |
|---|---:|---|
| `fused_topk` | 33 | 路由：softmax → top-k → 归一化（triton） |
| `moe_align_block_size` | 47 | 把 token 按专家排序分块（有 sgl_kernel 时用原生版） |
| `get_default_config` | 120 | 默认 tile：`M <= E` 时 16/32/64 GROUP=1，否则 64/64/32 GROUP=8 |
| `_get_tuned_moe_configs` | 144 | 读 tuned JSON（路径含 E/N/device/triton 版本） |
| `try_get_optimal_moe_config` | 168 | 取**最接近的 batch** 的配置；没有表就回默认 |
| `fused_experts_impl` | 192 | prefill：grouped GEMM |
| `fused_experts_decode_impl` | 300 | decode：per-route GEMV（固定 BLOCK_N=32/BLOCK_K=64） |

### 4.5 `bench_profile.py` 与 `benchbw.py`

| 函数 | 行 | 作用 |
|---|---:|---|
| `default_profile_path` | 39 | `benchbw/<gpu-uuid>.json` 或 legacy |
| `_usable_profile` | 80 | 读并**按 GPU name 校验**；不匹配返回 None |
| `load_backend_recommendation` | 114 | offload→hybrid 的裁决：全票才升级，否则保守 |
| `load_hybrid_fetch_fraction` | 156 | 优先重叠对 `pcie_ov/(pcie_ov+cpu_ov)`，退化 `pcie/cpu`，clamp [0,1] |
| `benchbw.measure_cpu_mem_bw` | 186 | STREAM 式主机带宽 |
| `measure_pcie_bw` | 258 | 线性 H2D/D2H 上限 |
| `measure_pcie_gather_bw` | 418 | **真实 rig**：`OffloadMoeCache.copy_missing()` over 合成 bank |
| `measure_cpu_moe_bw` | 508 | **真实** `CpuMoeExecutor` bs=1，64 步中位数 |
| `measure_overlap_bw` | 540 | 上面两者**并发**跑 2 s（这是分式的真实来源） |
| `recommend` | 600 | `cpu > threshold*pcie` → hybrid，否则 offload |

---

## 5. 关键流程逐段解析

### 5.1 offload decode 的每层（`layers/moe.py:285-287`）
```python
cache.ensure_experts(layer_id, topk_ids)   # ① 外部 LRU：改写 topk_ids 为 slot id
cache.copy_missing()                       # ② PCIe gather 未命中行
self._expert_gemm(..., views=cache.bank_views(), ...)   # ③ GEMM
```
**三段在同一 stream 上串行**：① 有设备端簿记开销；② 是带宽瓶颈；③ 是 HBM 受限。decode 没有重叠——只有 hybrid 把 ③ 的一部分挪到 CPU 并行。

### 5.2 hybrid 三阶段 kernel（`offload_kernels.py:290-410`）
```
Phase 1: 标出 active / missing；num_missing = sum(missing)
         Q16 解：lo = (M*frac)>>16
                 cost(F) = max(F*(65536-frac), (M-F)*frac)     # 归一化后的"较慢一侧"
                 max_fetch = lo 还是 lo+1？取决于 cost 谁小
         num_fetch = min(M, max_fetch)
Phase 2: 驱逐：活跃槽 usage 置 INT64_MAX 保护；其余按 argmin(usage) 选受害者
Phase 3: 抓取选中的 miss：slot_for_id[base+e] → 写回 topk_ids
         溢出未命中 → slot = -1（CPU 侧算）
```
目标函数的意义：`F = frac*M` 时两侧相等（**完美重叠**）。`frac` 来自实测的**并发带宽对**，见 §4.5。
抓取优先级：默认 `recency`（最近活跃的 miss 先抓，降低稳态 miss 率），可用 `FREETOKEN_HYBRID_FETCH=lowest_id` 退回"按 id 最小"。

### 5.3 CPU/GPU 并行握手（hybrid / cpu 策略）
```
GPU 侧（前端）：写 done=0、写 ready=1（两个 stream memop）
CPU 侧：常驻协调器轮询 ready → submit(task)+sync(spin) → release-store done=1
GPU 侧：cuStreamWaitValue64(done >= 1) → 继续
```
两条实现路径：
- **默认 flag 握手**（`FREETOKEN_CPU_MOE_FLAG_SYNC`）：等待不占 SM；
- **回退 `cudaLaunchHostFunc`**：注释记录 **~30-50 μs/回调 × 2 次/层/步 ≈ 6 ms/step**（75 层模型）。
注释还记录：早期用"常驻自旋 kernel"会把利用率钉在 99% 并触发 CPU 降频（GEMV worker -1.5×）。

### 5.4 双缓冲 prefill（`offload_cache.py:609-844`）
```
buffers = bank_caches[bank][:2E].view(2, E, ...)     # 每 bank 两个缓冲
begin_prefill():   copy_stream.wait_event(begin_event)   # 防止借用槽仍被 decode 读
_wait_prefill_overlap（每层）：
    prefetch(layer_id)  → buffer_id = layer_id % 2
    prefetch(layer_id+1)                                  # 预取深度 = 1 层
    wait_prefill_layer(layer_id)                          # 计算流等 ready 事件
    ... GEMM ...
    release_prefill_layer(layer_id)                       # 记录 release 事件
```
不变量：`cache_size >= 2E`（`engine/cache_budget.py:70-81` 的 `lo` 就是这个）；
hit-D2D 额外要求 `cache_size > 2E`（要留出"命中区"，`:721-725`）。

### 5.5 `_prefetch_split`（`:749-819`）——本文最值得学的一段
```
① 每 chunk 一次：在 copy 流上快照 slot_for_id，然后一次宿主同步
② 宿主侧分类 snap >= 2E：命中 vs 未命中（两个集合不相交，所以两流无需互定序）
③ 命中 → prefill_hit_compact 压紧 + fast_index_copy_multi_jit(blocks_per_bank=64)  D2D
④ 未命中 → 宿主按"合并成连续 expert 区间"构造条目 → **一次** cudaMemcpyBatchAsync
```
**256 KB 阈值的由来**（`offload_cache.py:17-26`，被二分定位出来的驱动行为）：
`cudaMemcpyBatchAsync` 在"大条目 + 小于 ~256 KB 条目"混在同一个 batch 时，会**静默退化为同步拷贝**（H100+CUDA13 实测：5-22 KB 条目旁放一个大条目就阻塞调用线程；≥253 KB 不触发）。后果是宿主被阻塞，"拷贝期本该盖住的 GEMM 又露出来"——gpt-oss 2048 tok 端到端 **-22%**。
对策：行字节 < `_SMALL_BANK_FEAT_BYTES=256KiB` 的 bank **整体作为一个 whole-layer 条目**发送并排除在 gather 外，保证 batch 里每个条目都 ≥256 KB。

### 5.6 `ft bench bw` → profile → auto
```
measure_cpu_mem_bw / measure_pcie_bw / measure_pcie_gather_bw / measure_cpu_moe_bw / measure_overlap_bw
  → recommend(cpu, pcie, threshold=2.0) → 写 benchbw/<gpu-uuid>.json (version 4)
启动时：
  engine._init_offload_moe_cache → load_backend_recommendation（auto 时 offload→hybrid）
  engine._resolve_hybrid_fetch  → load_hybrid_fetch_fraction（--moe-hybrid-max-fetch -1）
```
**关键纪律**：profile 按 GPU UUID 存、按 GPU **name** 校验、join key 是**专家格式**而非模型（比值由 format×hardware 决定，所以可迁移）。

### 5.7 host bank 的 pin-after-fill
```
HostBank 分配（mmap 或 cudaHostAlloc）
  → 用 O_DIRECT 读 checkpoint 填进去
  → PinPipeline 后台线程统一 cudaHostRegister（驱动内部串行 → 串行化反而更快）
  → 需要 mlock 的 LOCKED 层在同一线程里做（配额簿记非线程安全）
```

---

## 6. 与其他模块的关系

| 方向 | 模块 | 接口 | 说明 |
|---|---|---|---|
| 被调用 | `layers/moe.py`（`OffloadMoELayer`） | `ensure_experts` / `copy_missing` / `bank_views` / 预取三件套 | **唯一调用方** |
| 使用 | `kernel/fast_index_copy.py` + `csrc/jit/fast_index_copy.cuh` | 专家行 gather | 最性能关键的 kernel |
| 使用 | `kernel/batch_memcpy.py` | prefill 未命中批量 H2D | CUDA≥13 门控 |
| 使用 | `kernel/pinned.py` + `csrc/pinned_tensor.cpp` | 精确尺寸 pinned 分配 | |
| 使用 | `layers/quantization/moe/*` | 专家 GEMM 与 bank 布局（`BankSpec`） | 布局由内核拥有 |
| 使用 | `kernel/triton/*_moe.py` | 实际 GEMM/GEMV | |
| 被使用 | `engine/engine.py:638-795` | 初始化、槽位求解、hybrid 分式 | |
| **外部依赖** | `flashlib==0.3.0` | `lru_ensure` / `N_STATS` / `Stat` | **默认（非 hybrid）LRU 内核不在本仓库** |

---

## 7. 常见坑

1. **默认 LRU 内核不在本仓库**：`offload_kernels.py:8` 从 `flashlib` 导入。仓库内只有 hybrid 变体与它的 CPU 镜像——读不到不是漏看了。
2. **decode 的搬运不与 GEMM 重叠**（同一 stream 串行）。双缓冲是 **prefill-only**。
3. **hit-D2D 实际要求 CUDA >= 13.0**（`kernel/batch_memcpy.py:42-53`），而 `engine/config.py:52-53` 的注释写的是 12.8——**以代码为准**。
4. **`--moe-cache-policy` 只有 `lru` 一个选择**（`args.py:717`），是给未来留的占位。
5. **小 bank 必须整层发送**，否则 `cudaMemcpyBatchAsync` 静默退化（见 §5.5）。
6. **`collect_decode_freq` 不是图安全的**（`offload_cache.py:245-249` 自己写了），而 `lru_stats` 的累加是图安全的。
7. **pin-after-fill 不能反**：先注册惰性 mmap 会触发全量缺页清零（DSV4 ~47 s）。
8. **`FREETOKEN_SKIP_BANK_PIN` 只在做消融时用**，服务时设了 GPU 路径会读不到 bank。
9. **profile 在别的 GPU 上测的会被忽略**（不会误用），所以换机器要重跑 `ft bench bw`。

---

## 8. 自测问题

1. 一张 bf16 专家（H=2048,I=512）多大？8 个未命中在 31 GB/s 下要多久？换成 HBM 呢？
2. `ensure_experts` 为什么能原地改写 `topk_ids`？下游谁依赖这一点？
3. hybrid 的目标函数为什么在 `F = frac*M` 处平衡？`frac` 为什么用并发测量的那一对？
4. 双缓冲的 `cache_size >= 2E` 从哪来？为什么 hit-D2D 还要 `> 2E`？
5. 为什么 `<256 KB` 的 bank 要整层发送？不这么做会发生什么（含实测数字）？
6. `lru_stats` 为什么可以随 CUDA graph 重放而 `decode_freq` 不行？
7. pin-after-fill 的顺序为什么不能反？先注册会付出什么代价？
8. `ft bench bw` 的 2.0× 阈值在保护什么？什么情况下 hybrid 反而更慢？
