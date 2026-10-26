# FreeToken 推理优化阅读指南（性能向）

> 锚点基于 `377a3bc`。配合 [SOURCE_READING_GUIDE.md](SOURCE_READING_GUIDE.md)（全局地图）使用：本文件不重复讲架构，只讲**性能**。
> 本机无 GPU、无 torch，所以下面所有数字都标注了来源，凡是我没能验证的都标了"未验证"。

---

## 0. 先建立瓶颈模型（这一步决定后面所有取舍）

读性能代码最怕"什么都看，什么都记不住"。先花 20 分钟把这个引擎的**目标函数**想清楚：

**场景**：单张消费级显卡（RTX 30/40/50），MoE 模型有 100B~300B 参数但每 token 只激活 3B~10B。
**结论**：这类系统几乎总是**内存带宽受限**，而不是 FLOPS 受限。所以一切优化都在回答同一个问题：

> **如何让"必须搬的字节"更少、更早、且不阻塞计算？**

由此推出全仓的性能母题清单（第 1 节），以及四类必须被"藏起来"的延迟：

| 延迟 | 量级 | 藏它的手段 | 在哪 |
|---|---|---|---|
| kernel launch 开销 | 每 kernel 几 μs × 几百个 kernel | CUDA graph | `engine/graph.py` |
| CPU 调度/记账 | 每步几十~几百 μs | 双 stream 流水 | `scheduler/scheduler.py:207 overlap_loop` |
| GPU→CPU 同步等待 | 每步可达数百 μs | 非阻塞 D2H + event | `engine/engine.py:1058-1061` |
| 专家/权重搬运 | PCIe ~32-64 GB/s（`docs/models.md:66`） | LRU 缓存 + 预取 + CPU 协同 | `moe/` |

**仓库里仅有的定量锚点**（诚实清单，其余都要自己测）。标注 `[注释]` 的是源码注释里记录的实测值——**不是文档宣传值，是作者当时量出来的**，可信度更高：

| 锚点 | 值 | 出处 |
|---|---|---|
| PCIe 主机链路（文档口径） | ~32-64 GB/s | `docs/models.md:66` |
| **PCIe gather 实测** | **~31 GB/s**（对比 HBM ~3 TB/s） | `[注释]` `kernel/fast_index_copy.py:183` |
| gather 的网格依赖 | PCIe ~4K 线程/bank 即饱和；NVLink-C2C 需 ~32K（实测 8×1024 → 222，32×1024 → 412 GB/s） | `[注释]` `kernel/fast_index_copy.py:152-155` |
| GH200 上调 `FREETOKEN_H2D_BLOCKS_PER_BANK=32` | 专家 gather ~220 → ~410 GB/s | `docs/models.md:73` |
| **CPU MoE 最优单读带宽** | **~205 GB/s @ bs=1**（对比 PCIe ~55 GB/s） | `[注释]` `csrc/cpu_moe/cpu_moe_ext.cpp:1101-1105` |
| hybrid 判据阈值 | CPU 带宽 > **2.0 ×** PCIe gather | `docs/cli.md:206`（`--threshold`） |
| 非重叠搬运的代价 | gpt-oss 2048tok 端到端 **-22%**（同步拷贝"隐藏"了拷贝期本来能盖住的 GEMM） | `[注释]` `moe/offload_cache.py:17-25` |
| host-func 同步的成本 | ~30-50 μs/次 × 2 次/层/步 ≈ **6 ms/step**（75 层模型） | `[注释]` `moe/cpu_executor.py:32-45` |
| 每步拷贝的 launch 数 | 传统 per-bank 方式在 "6 bank × 36 层" 下 = **216 次/步**（暖缓存时几乎都是空 launch） | `[注释]` `kernel/csrc/jit/fast_index_copy.cuh:501-509` |
| 反量化往返 | CPU fp8 ~0.3 ms/层 @ H=4096 | `[注释]` `csrc/cpu_moe/cpu_moe_ext.cpp:1275` |
| 采样 token 预算 | Qwen3.5-35B-A3B-FP8 上 aime25 单题三条各 4.1k/10.7k/10.0k token | `tests/README.md:74` |

**没有的东西也要知道**：仓库里**没有**端到端 tok/s 的公开基线数字，`benchmarks/README.md` 只说明怎么跑。上表里最有用的一条是 **PCIe gather 31 GB/s vs HBM ~3 TB/s——每字节约差 100 倍**：这一个数字就足以解释整个架构为什么长成现在这样。

---

## 1. 六个优化母题（母题 → 机制 → 锚点 → 换来什么）

这是全仓性能代码的"索引"。**先记住这 6 条，再去看代码，你会发现所有细节都挂在它们下面。**

### 母题 1：双 CUDA stream 流水，把 CPU 调度藏到 GPU 计算背后
- 机制：`Scheduler.__init__` 建第二条流（`scheduler.py:70-73`，注释直说"use another stream to overlap metadata processing with computation"）。`overlap_loop`（`scheduler.py:207-261`）每轮：收消息 → **发射**批次 N → **排空**批次 N-1。两次 `stream.wait_stream(engine.stream)`（`:241`、`:258`）做双向跨流定序。
- 换来：CPU 的记账/取样/提交前缀缓存与 GPU 的前向重叠，GPU 不再每步空转。
- 关键细节：`self._last_data`（`:106-118` 注释）保存"已发射未排空"的批次，这正是 abort 必须"打标记延后释放"的原因。

### 母题 2：非阻塞 D2H + event，CPU 只等一个 token 的拷贝
- 机制（`engine/engine.py:1044-1062`，`@torch.inference_mode()`）：
  ```python
  next_tokens_gpu = self.sampler.sample(batch_logits, args).to(torch.int32)
  next_tokens_cpu = next_tokens_gpu.to("cpu", non_blocking=True)   # 设计意图：落到 pinned buffer
  copy_done_event = torch.cuda.Event(); copy_done_event.record(self.stream)
  ```
  排空侧只 `copy_done.synchronize()`（`scheduler.py:317`），而不是整步同步。
- 换来：设计上让 CPU 不必等整个 forward 完成，流水深度得以维持。
- 依赖：目标必须是 pinned 内存，否则 `non_blocking` 退化成同步。
- **⚠️ 实际代码里这个前提没被满足**——详见 [1.1](#11-一个值得学的读法示例non_blockingtrue-的前提可能没被满足)。这是我在这份代码里发现的唯一一处"优化机制可能没兑现"的地方。

### 母题 3：CUDA graph + bs 阶梯 + 显存池复用
- 机制（`engine/graph.py`）：
  - bs 阶梯 `[1,2,4] + range(8, max+1, 8)`，`max` 默认 >80GB 显存取 256，否则 160（`:79-98`）
  - `pad_batch`（`:215`）选**第一个 ≥ 实际 bs 的已捕获档位**，用 dummy req 补齐
  - 捕获前先做一次 eager warmup 前向（`:189-194`），把懒分配/autotune 全逼出来，否则会捕获进 graph
  - 按 bs **从大到小**捕获，然后复用第一个 graph 的 mempool（`:196-197`），省显存
  - MoE offload cache 每次捕获前 reset（`:195`），保证捕获轨迹确定
  - 静态缓冲 `GraphCaptureBuffer`（`:22-48`）：`input_ids/out_loc/positions/table_idx` + **`logits [max_bs, vocab] fp32`**（注意这笔显存：160 × 150k × 4B ≈ 96 MB）
  - `replay`（`:207-213`）顺序：`copy_from`（H2D 小量输入）→ `prepare_for_replay` → `g.replay()`
  - 只在 decode 且 `bs <= max_graph_bs` 时用（`:204-205`）；`_adjust_config` 把 graph bs 上限**钳到 `--max-running-requests`**（`engine.py:1230-1237`）
- 换来：一次 launch 顶几百个 kernel launch。
- 代价与坑：capture 期间要显存；rebuild 必须先 `destroy_cuda_graphs`（`:224-232`，注释说"drop references is load-bearing"），且必须在释放 NCCL 之前。

### 母题 4：预取 / 双缓冲 / 预启动 staging
- MoE prefill：slot cache 头部 `2*num_experts` 个 slot reshape 成两个权重缓冲（`moe/offload_cache.py:609-636`），`buffer_id = layer_id % 2`，预取 layer i+1 同时算 layer i（`layers/moe.py:391-401`）。
- 多模态编码器：`--mm-encoder-weights host` 把 tower 权重放主机 pinned bank，**两块两块地**流式送 GPU，只驻留两块（`docs/cli.md:119`）。
- 磁盘表：DSV4.1 Engram / Qwen4Exp PLE 的 `forward_host_ctx`（`models/blocks.py:33`，覆盖于 `models/deepseek_v41/engram_table.py:283`、`models/qwen4_exp/ple_disk.py:247`）在 launch **之前**完成 hash + 读行 + 填 pinned staging。
- 换来：搬运时间被计算掩盖。

> **⚠️ 一个必须记住的边界：这套双缓冲是 prefill-only 的。**
> `layers/moe.py:285-287` 的 decode 路径是 `ensure_experts` → `copy_missing()` → `_expert_gemm(...)`，**三段在同一 stream 上串行**——也就是说 **decode 时专家搬运并没有和该层的 GEMM 重叠**，每层都要先等数据到齐再算。
> 只有两类例外：（a）hybrid 把 **CPU 计算**塞进 GPU 工作之下（`layers/moe.py:298-350`）；（b）prefill 的 hit-D2D 把未命中批次放在 copy stream 上、命中在计算 stream 上并行 gather（`offload_cache.py:749-765, 777-819`）。
> 所以 offload 的 decode 每层代价 ≈ `LRU 核` + `PCIe gather` + `GEMM`，**三者相加**。这解释了为什么"提高专家命中率"是这个项目最有效的杠杆——它直接砍掉中间那项。

### 母题 5：钉住内存 + 零拷贝 + 绕过 page cache
- `kernel/pinned.py` + `csrc/pinned_tensor.cpp`：**精确尺寸**的 `cudaHostAlloc`（避开 cudaMallocHost 的 2 的幂膨胀）。
- 专家 bank 先在主机填好再 pin（`moe/host_banks.py`），`fast_index_copy` 直接从 pinned 主机内存 gather 到 GPU——这是 offload 模式的核心内核。
- 磁盘表：`csrc/row_store/`（O_DIRECT + pread / io_uring），跳过 page cache（PLE/Engram 表远超内存）。
- 换来：同等带宽下少一次拷贝、少一次内核页缓存污染。

### 母题 6：把"策略选择"从静态默认变成实测自适应
- `ft bench bw` 测本机带宽 → 写 `~/.cache/freetoken/benchbw/<gpu-uuid>.json` → 引擎启动时读（`moe/bench_profile.py`，**torch-free**）。
- 判据：CPU 带宽 > 2.0 × PCIe gather → 推荐 hybrid；负载/格式的裁决是"全票 hybrid 才 hybrid，否则保守回 offload"（`bench_profile.py:144-153`）。
- 分式：优先用**并发测量**的重叠带宽 `pcie_ov/(pcie_ov+cpu_ov)`，老 profile 退化为 `pcie/cpu`（`:185-190`），并 clamp 到 [0,1]。
- 其它 auto：注意力后端（`engine.py:124`）、MoE 策略（`engine.py:1653+`）、专家缓存大小（`engine.py:611`）、`--moe-cpu-layers auto`（`engine.py:1317`）。
- 换来：同一份代码在不同机器上自动落到不同策略，而不是靠用户拍参数。

---

## 2. 阅读顺序（五阶段）

**核心原则：先学"怎么测"，再读"怎么快"。** 这个仓库把测量做成了基础设施，跳过第 0 阶段会让你后面所有的判断都没有依据。

### 阶段 0：度量先行（半天）——先学会测，再谈优化

| 顺序 | 读什么 | 学什么 |
|---|---|---|
| 0.1 | `benchmarks/README.md`（31 行，全读） | 三个基准各自测什么、以及"带宽与 offload/hybrid 选型请用 `ft bench bw`" |
| 0.2 | `benchmarks/bench_offload_cache_copy.py` | **最小、无 checkpoint 的合成基准**（`ensure_experts` + `copy_missing` 扫 bank 布局 × cache 槽位 × batch × miss 率）。先读它学方法论 |
| 0.3 | `benchmark/perf.py:10 perf_cuda` + `:54 compare_memory_kernel_perf` | CUDA event + **用 CUDA graph 把 N 次迭代打包计时**（`:31-51`），再换算 GB/s（`:66`）。这是全仓微观基准的通用模板 |
| 0.4 | `moe/benchbw.py`（999 行，读文档字符串 + `measure_*` 函数 + 结尾的 `recommend`）+ `moe/bench_profile.py`（191 行，全读） | **完整的"测量 → profile → 策略"闭环**，全仓最值得抄的架构 |
| 0.5 | `scheduler/status.py`（144 行，全读） | 训练日志里的 `input throughput (token/s)` 和 `gen throughput (token/s)` 是怎么算的；注意它读的是**调度时刻快照** `batch.log_new_tokens`，而不是 live req（`:74-78` 有解释——因为 `complete_one()` 已经把 `cached_len` 推进了） |
| 0.6 | `server/stats.py:129 build_stats` | `/v1/stats` 的全部字段：`kv/mamba/swa` 池占用、`vram_bytes`、`decode_tps`/`prefill_tps`（5 秒滑窗，`:14`）、`p95_ms`、`ttft_mean_ms` |
| 0.7 | `server/args.py` 里的 `--decode-log-interval`、`--expert-load`、`--dummy-weight` | `--dummy-weight`（`args.py:341-345` → `engine.py:251 _make_dummy_weight_state_dict`）**不加载真权重**，用来把"运行时/调度开销"和"权重搬运开销"分离 |

**阶段产出**：你能说清"要优化什么，就先用哪个工具量它"。

### 阶段 1：单步 decode 延迟（1 天）——热路径本身

这是延迟的原子单位。按下面顺序读，最后应该能**按时间顺序复述一个 decode step 的 GPU 操作与同步点**。

```
engine/engine.py:1044-1062   forward_batch：graph or eager、采样、D2H、event
scheduler/scheduler.py:207-261  overlap_loop：谁先谁后、三次 wait_stream 围栏
scheduler/scheduler.py:312-349  排空：全步唯一的一次主机阻塞 copy_done.synchronize()（:317）
scheduler/scheduler.py:786-845  _prepare_batch：每步宿主侧到底做了多少事（全部在 self.stream 上）
engine/graph.py:22-48        GraphCaptureBuffer：哪些张量是"静态"的
engine/graph.py:79-98        bs 阶梯怎么定
engine/graph.py:140-202      捕获：warmup、倒序捕获、pool 复用、dummy slot、MoE cache reset
engine/graph.py:204-232      can_use_cuda_graph / replay / pad_batch / destroy
engine/sample.py:59-99       Sampler.prepare：全 greedy 时 **temps=None，直接跳过整个采样链**（:62-63）
engine/sample.py:88-99       greedy → torch.argmax；混合 batch → 概率采样 + greedy_mask 覆盖
engine/sample.py:38          enable_pdl=is_sm90_supported()（PDL 让下一个 kernel 提前发射）
kernel/csrc/jit/store.cu:24-52,110-119   decode 关键路径上的 KV 写入 kernel（warp-per-token、K+V 一趟、向量化 warp copy）
kernel/csrc/jit/index.cu:30-59,130-168   decode 关键路径上的 embedding gather（按行字节数决定行内 warp 切分）
kernel/pinned.py:1-68        钉住内存如何申请（D2H 非阻塞的前提，附 UVA/WDDM 注意事项）
```

**读完要能默写出这条 GPU 端有序路径**（稳态、`bs ≤ max_graph_bs`）：

```
token_pool gather → batch.input_ids                     (scheduler.py:900，eager)
GraphCaptureBuffer.copy_from：最多 5 次 D2D 拷贝          (graph.py:67-76)
attn_backend.prepare_for_replay：再 3~4 次 D2D            (graph.py:211；fa.py:144-149)
g.replay() —— 一次 cudaGraphLaunch 覆盖整个 forward      (graph.py:212)
采样：argmax 或 softmax+draw（eager，在图外）             (engine.py:1060)
.to(torch.int32) → D2H → copy_done_event.record          (engine.py:1060-1063)
token_pool scatter                                      (scheduler.py:904)
```

关键结论：**整个模型前向被压成了一次 `cudaGraphLaunch`**，所以这一步是"launch 开销受限"而不是"kernel 数量受限"；而**采样被刻意留在图外**（`engine.py:1049-1050` vs `:1059-1060`），代价是采样每步都要走一遍宿主侧准备（`Sampler.prepare` 每步新建参数张量）。

**这里有一个必答的问题**：`sampler.sample` 是**每个 decode step 都跑**的。全 greedy 的路径只有一次 `argmax`（`:91-92`），而混合 batch 会走 softmax+top-k/top-p（`:93`）——所以"同一批里混了一个采样请求"会让整批付出概率采样成本。这是真实存在的性能悬崖。更细一层：非 greedy 时还会**额外多跑一次完整 `argmax`**（`:97`，为了给 greedy 行保持 argmax 的确定性 tie-break），也就是说混合 batch 的成本 > 纯采样 batch。

#### 1.1 一个值得学的读法示例：`non_blocking=True` 的前提可能没被满足

这是我在核对时发现的一处**性能手法与实际代码不匹配**，也是"读性能代码"最值得掌握的检查动作。

**代码事实**（`engine/engine.py:1059-1063`，我逐行确认）：
```python
next_tokens_gpu = self.sampler.sample(batch_logits, args).to(torch.int32)
next_tokens_cpu = next_tokens_gpu.to("cpu", non_blocking=True)   # ← 目标张量是普通 pageable 内存
copy_done_event = torch.cuda.Event()
copy_done_event.record(self.stream)
```
`ForwardOutput.next_tokens_cpu` 是一个**每次调用新建的、非 pinned 的 CPU 张量**（`engine.py:326`）。我 grep 过 `engine.py`：`pin_memory` 只出现在专家 bank / PLE 表相关的路径上，**采样 token 的 D2H 没有 pinned staging buffer**。`torch` 的 `.to("cpu")` 每次分配新内存，不会复用 pinned 池。

**为什么这重要**：CUDA 对 pageable 目标的异步 D2H，**语义上退化为同步**——函数只在拷贝完成后返回（NVIDIA 文档明确：*"For transfers from device memory to pageable host memory, the function will return only once the copy has completed"*）。而这次拷贝排在整条 engine stream 之后（graph replay + 采样），所以：
- 宿主线程在 `_forward` 里被阻塞到**本步 GPU 工作全部结束**；
- 于是 `copy_done.synchronize()`（`scheduler.py:317`）等的是一个**早已完成**的事件，几乎是免费的；
- 而 `overlap_loop` 想藏起来的那部分 CPU 工作（排空 batch N-1、调度 batch N+1）**落在了 GPU 空闲的间隙里**。

换句话说：双 stream 管线的**定序**是对的，但**重叠的收益可能没兑现**。

**必须说清楚的边界**：以上是"代码事实 + CUDA 文档语义"推出来的，**我没有 GPU 无法量化**。它也可能是有意为之（例如 `bs` 极小、或为了让显存分配可预测）。判断它的方法就是本指南阶段 0 的工具：把 `--decode-log-interval` 调到 5，对比 CPU 侧调度耗时与步时；或直接给 `next_tokens_cpu` 一个持久的 pinned buffer（`kernel/pinned.py:42-45` 的 `alloc_pinned_tensor`，按 `max_graph_bs` 预分配）做 A/B。**这也是一个很适合作为你第一个"读源码发现问题"的练习题。**

**从这个例子提炼出的 4 个通用检查动作**（读任何性能代码都适用）：
1. **看到 `non_blocking=True`，先查目标是否 pinned**——否则它是装饰性的。
2. **看到 `non_blocking=True` / `async`，先看它排在谁后面**——异步只挪走了等待，没挪走依赖。
3. **看到 CUDA event + `synchronize()`，问"如果 D2H 本来就阻塞，这个 event 还有意义吗"**——冗余的同步器往往是"上游本该异步"的化石证据。
4. **看到"为了隐藏延迟而加的机制"（双流、双缓冲、预取），一定去核对它隐藏的那件事是否真的还能被藏**——机制存在不等于收益存在。

#### 1.2 不在这 6 个母题里、但同样值得记住的手法

| 手法 | 位置 | 换来了什么 |
|---|---|---|
| 整个调度循环包在 `torch.inference_mode()` 下 | `scheduler.py:290` | 步内没有 autograd/版本计数器开销 |
| GDN 的 `cu_seqlens` 用常量 `arange(bs+1)` + 常驻 slot map | `graph.py:31-32, 47, 61-65` | **省掉每步一次 H2D**，且地址对捕获稳定 |
| GDN decode 的 slot 索引用 GPU int64 张量而非宿主循环 | `core.py:120-125`、`scheduler.py:822-830` | decode 保持可捕获，无宿主同步 |
| CPU-MoE 握手用 **GPU stream memop（`cuStreamWaitValue`）而不是自旋 kernel** | `cpu_executor.py:38-46, 598-620` | 等待不占 SM；注释记录自旋版曾把利用率钉在 99% 并影响功耗联动 |
| 该握手用 pinned flag slot（每 (layer,bs) 一个） | `cpu_executor.py:293-296, 543` | 每个捕获的 decode graph 都能走这套握手 |
| row-store 磁盘 I/O 用 pinned wait/signal flag | `row_store.py:6-9, 46-58` | **被捕获的** lookup 可以用 stream memop 等宿主填充 |
| 写元组用固定形状 + `-1` 汇列，不在宿主过滤已完成行 | `scheduler.py:957-962` | 避免每步的形状变化与宿主分支 |
| stop-string 匹配只在已解码的短尾部做 | `scheduler.py:431-445` | 每步宿主侧反解码成本恒定（`O(尾部)` 而非 `O(全文)`） |
| 状态/显存读数**不做 device sync** | `engine.py:481-486`（注释：*"Cheap, no device sync"） | 观测不污染被观测对象 |
| 精确尺寸的 pinned 分配（扩展而非 torch 分配器） | `pinned.py:1-6`、`csrc/pinned_tensor.cpp:46-68` | torch 的 pinned 缓存会按 2 的幂取整（注释举例 70 GB → 128 GB） |
| 采样用多 CTA 切分词表 | `kernel/triton/sampling.py:1-20, 44-51` | bs=1 也能用满 SM，而不是单 CTA |

**这一节的读法**：这些手法都可以归纳成一句话——**"把每步的宿主侧成本变成常数，把不变量提到捕获/启动期"**。你读其它推理引擎时会反复见到同一批手法。

### 阶段 2：吞吐与显存（1 天）——把单步摊薄 / 把显存分对

延迟是分子，吞吐是"每步处理多少 token"和"显存够放多少并发"。

```
engine/cache_budget.py:31-95    ★ 显存预算的纯算术 single source of truth
engine/engine.py:73-77          注意：普通启动路径实际用的是 _startup_kv_budget + solve_num_pages（第二份实现）
engine/config.py:73-98          page_size / memory_ratio / swa_full_tokens_ratio 的默认与含义
engine/engine.py:1230-1237      CUDA graph bs 被 --max-running-requests 钳住（超了只警告）
scheduler/prefill.py:44-348     PrefillAdder：token 预算、分块、SWA/GDN 的"第二/第三币种"闸门
scheduler/scheduler.py:786-847  _prepare_batch：padding、page table、positions
kvcache/base.py:46-160          池的 sizing 接口
kvcache/radix_cache.py:236-262  _tree_walk：前缀命中率 = 直接省掉的 prefill 计算
```

**这里要形成的判断力**：`--memory-ratio` 往上调，KV 页与专家槽位就多（吞吐↑），但 CUDA graph/激活的余量变小（可能 OOM 或被迫关 graph）；`--moe-cache-auto` 是 MoE 优先分配（`cache_budget.py:58-95` 的注释说明了先预留 KV、再让专家贪心吃满），所以它天然偏向 MoE。**预算与策略是耦合的**，不要分开调。

### 阶段 3：MoE offload 性能（1-2 天）——项目卖点，也是最大的性能变量

```
docs/models.md:39-76            策略语义 + unified-memory / NVLink-C2C 特例（先读文档，代码会很顺）
layers/moe.py:261-296           ★ _decode_routed：一步的完整闭环
moe/offload_kernels.py:19-40    LRU 委托 flashlib.slot_cache（把 topk_ids 原地改写成 slot id）
moe/offload_cache.py:37-97      ★ _BANK_SCHEMAS + _BANK_BYTES_PER_EXPERT：每个格式的槽位字节数（决定 --moe-cache-size 的性价比）
moe/offload_cache.py:104-292    数据结构：slot_for_id / id_of_slot / usage / step / src_indices
moe/offload_cache.py:1014-1057  未命中搬运 fast_index_copy_*
moe/offload_cache.py:671-705    双缓冲预取：buffer_id = layer_id % 2 + ready/release 事件 + "未 release 不许复用"断言
moe/offload_cache.py:706-736    hit-D2D 的可用性判定与**降级要出声**（_hit_d2d_usable 会 log 一次原因）
moe/offload_cache.py:749-844    _prefetch_split：命中走设备侧 gather，未命中走一次 cudaMemcpyBatchAsync
moe/offload_kernels.py:290-410  hybrid：GPU 上的 Q16 整数解（见下面的不一致提示）
moe/cpu_executor.py             CPU 侧 submit/sync 协议与 ISA 分派
moe/host_banks.py               HostBank / pin-after-fill / PinPipeline
engine/engine.py:611-795        缓存大小求解 + q*（hybrid fetch fraction）解析
moe/bench_profile.py:156-191    分式怎么从实测带宽算出（含并发测量的重叠对）
```

**每个槽位多大？直接可算**（`_BANK_BYTES_PER_EXPERT`，`offload_cache.py:87-97`）——这决定了 `--moe-cache-size` 每加一格要花多少显存：

| 格式 | 每（专家,层）字节 | 相对 bf16 |
|---|---|---|
| bf16 | `3·I·H·2` | 1.0（基准） |
| fp8_block | `3·I·H` + 块 scale | ~0.5 |
| nvfp4 | `2I(H/2 + H/16 + 2) + H(I/2 + I/16 + 2)` | ~0.25 |
| mxfp4 | `2I(H/2 + H/32 + 2) + H(I/2 + I/32 + 2)` | ~0.25 |
| ds_fp4 | `2I(H/2 + H/32) + H(I/2 + I/32)` | ~0.25 |
| q4_0 | `2I(H/32)·18 + H(I/32)·18` | ~0.28 |

**必须搞清的一对权衡**（注意与母题 4 的边界一致——decode 是串行的）：
- **offload 每步每层 = `LRU 核` + `PCIe gather` + `GEMM`，三者相加**（同一 stream）。所以它与 miss 率**近似线性**：`--moe-cache-size` 增大 = 直接砍掉中间那项。
- **hybrid 每步 = `max(PCIe 抓取, CPU 溢出 GEMV)`**，CPU 那部分被塞进 GPU 工作之下（`layers/moe.py:298-350`），所以它是"取较大者"而不是"相加"。但它有 CPU 算力的下限，只在 CPU 带宽 > 2.0 × PCIe 时成立。

**一个绝佳的系统设计教训（值得单独记住）**：`moe/offload_cache.py:17-26` 记录了一个**被二分定位出来的驱动行为**——`cudaMemcpyBatchAsync` 在"大条目 + 小于 ~256 KB 的条目"混在同一个 batch 里时，会**静默退化为同步拷贝**（H100 + CUDA 13.0 实测：一个 5-22 KB 条目旁边放一个大条目，就阻塞调用线程直到传完；≥253 KB 的条目不触发）。后果：字节吞吐没变，但**宿主被阻塞**，于是"拷贝期本该被盖住的 GEMM 又露出来了"——gpt-oss 2048 tok 端到端 **-22%**。

对策同样值得学：把行字节 < 256 KB 的 bank **整体作为一个 whole-layer 条目**发送并排除在 gather 之外（`_SMALL_BANK_FEAT_BYTES = 256*1024`，`:26`、`:435-436`、`:801`），从而保证 batch 里每个条目都 ≥256 KB。

**推广**：把多个小 I/O 合成一个大 batch 调用，收益可能被"批内最小的那个元素"毁掉——**批处理 API 的性能特征往往由它的最坏成员决定**。这是读任何 batched API 封装时该问的问题。

> ### ⚠️ 文档与实现不一致（其一）：hybrid 抓取分式
>
> **`engine/engine.py:769-770` 与 `moe/offload_kernels.py:346` 的注释都写 `fetch_frac = pcie_bw / cpu_bw`，但实际喂进去的分式不是这个。**
>
> 实际来源在 `moe/bench_profile.py:156-191`：
> - **首选**（有并发测量时）：`pcie_gather_overlap_gbs / (pcie_ov + cpu_ov)` —— `:185-187`
> - **退化**（老 profile 无重叠对时）：`pcie_gather_gbs / cpu_moe_gbs` —— `:188-190`
>
> 哪个是对的？用引擎自己的物理模型推：设每专家抓取代价 `1/pcie`、CPU 计算代价 `1/cpu`，
> 抓 F 个、CPU 算 M-F 个，令两者相等得 **F = M · pcie/(pcie+cpu)** —— 与 `bench_profile` 的首选式一致。
> 内核里的目标函数 `max(F·(1-frac), (M-F)·frac)` 也确实在 `F = frac·M` 处平衡（`offload_kernels.py:349-354`）。
>
> **结论：以 `bench_profile.py` 为准，那两个 docstring 是旧的（只描述了退化式）。**
> 举例：pcie=100、cpu=150 GB/s 时，`pcie/cpu` = 0.67，而 `pcie/(pcie+cpu)` = 0.40 —— 差了 60% 以上的抓取量。
> 如果你按 docstring 去反推或手算调参，会系统性偏大。

**另外两处同类问题（顺手记下，都是"照注释理解会走偏"）**：

- **（其二）CUDA 版本要求写错了。** `engine/config.py:52-53` 的注释说 hit-D2D *"Needs CUDA >= 12.8"*，但真正的门在 `kernel/batch_memcpy.py:42-53`：它用到的 8 参数 `cudaMemcpyBatchAsync` 签名是 **CUDA 13.0 的**（12.8/12.9 多一个 `failIdx` 参数），并且运行时会检查 `torch.version.cuda >= (13, 0)`，否则直接抛 `RuntimeError`。`docs/cli.md:91` 说的也是 CUDA >= 13。**以代码为准。**
- **（其三）默认 LRU 内核不在本仓库。** `moe/offload_kernels.py:8` 导入的是外部依赖 `flashlib.kernels.slot_cache.lru_ensure`（`pyproject.toml:42` 钉死 `flashlib==0.3.0`）。**非 hybrid 的专家 LRU 准入逻辑你在源码里读不到**；仓库内只有 hybrid 变体（`offload_kernels.py:290-410`）和它的 CPU 镜像（`:128-188`）。这是阅读 MoE 时最大的一个盲区，别以为是漏看了。


### 阶段 4：内核与量化（按需，1 天）——把"哪条 kernel"和"哪个阈值"搞清楚

```
kernel/backend.py:26-38                    谁装了（find_spec + functools.cache，无副作用）——12 行讲清回退
layers/quantization/method.py:20-47       ★ select_kernel：全仓唯一的内核选择策略
layers/quantization/moe/base.py:137-167   MoEKernel 契约 + _common_reject：什么格式才配当候选
layers/quantization/moe/nvfp4.py:224-241, 342-374, 488-511   唯一真正的架构谓词（Marlin sm_80-99 / b12x sm_120+CUDA13 驱动）
kernel/triton/nvfp4_linear.py:53-116, 232-240, 523-543, 749-757  ★ 典范的 decode/prefill 阶梯
moe/fused.py:120-189                      get_default_config + try_get_optimal_moe_config + config JSON 路径规则
moe/fused_nvfp4.py:27-45                  离线扫出来的 decode tile 常量（注释里有实测百分比）
moe/fused_mxfp4.py:12-16, 200-204, 252-273  MXFP4_DECODE_MAX_TOKENS=16 + split-K 目标 program 数
attention/__init__.py:20-160 + engine/engine.py:124-160   能力矩阵 + 自动选择的硬件树
attention/base.py:94-130                  HybridBackend：prefill/decode 分流，且**只有 decode 侧进 CUDA graph**
kernel/triton/attention.py:21-42           smem 感知的 extend tile 选择（"不要随手改"）
kernel/fla/chunk_delta_h.py:22-51           故意只留单一 autotune 配置（多配置会破坏 in-place state pool）
layers/gguf.py:39-59                      最干净的阈值示例：_MMVQ_SAFE = 6
```

#### 4.1 选择策略只有一处：`select_kernel`

```
--quant-backend 显式给了名字  -> 直接实例化；若 unusable_reason 非空则**报错**（绝不静默降级）
否则按 candidates 顺序：
    跳过所有 unusable_reason 非空的
    取第一个 worth_it(cfg) == True 的
    都不满足 -> 取第一个可用的作为 fallback
```

两个必须记住的推论：
1. **`worth_it` 只是"可用内核之间的偏好"，不是门槛**（`method.py:33-46`）。
2. **所有谓词只在建层时求值一次，只依赖 config，不看 batch size、不看运行期形状**（`LinearConfig`/`MoEConfig`）。所以"内核选择"是 config 期决策，而"GEMV 还是 GEMM"是**运行期在核内/launcher 里**判断的——两者在不同的层，别混。

> **⚠️ 最实用的一条**：NVFP4 的 candidates 顺序是 `(triton, marlin, b12x)`（`moe/nvfp4.py:569`），而 triton 的闸门最松且 `worth_it=True`——**所以 `auto` 永远选 triton，marlin/b12x 只能靠 `--quant-backend moe.nvfp4=marlin|b12x` 显式指定**。`moe/nvfp4.py:505` 的注释也承认了这件事（"b12x stays behind triton in the table"）。如果你在 sm_120 上期待 b12x 自动生效，那是不会发生的。

#### 4.2 "为什么 decode 慢"的阈值清单

这是全仓最值得建一张小抄表的地方。**内核文件里那些"看起来像魔数"的常量，几乎都是 decode/prefill 的分水岭。**

| 位置 | 阈值 | 含义 |
|---|---|---|
| `layers/gguf.py:39` | `_MMVQ_SAFE = 6` | ≤6 token 走 MMVQ GEMV，否则 MMQ grouped |
| `kernel/triton/nvfp4_linear.py:768` | **M == 1** | 专用 GEMV |
| 同上 `:95, :754-757` | **M ≤ 64** | 核内 dot GEMM |
| 同上（doc `:28-31`） | **M > 64** | 反量化到 bf16 scratch + cuBLAS（核内反量化会按 M/BLOCK_M 重复反量化，H100 上慢约 3×） |
| `moe/fused_nvfp4.py:40-45` | K > 2048 | 深 K 时换窄 N tile 的 split（实测 18.6 vs 21.0 μs，短 K 反而退化） |
| `kernel/triton/mxfp8_linear.py:56` | `_GEMV_MAX_M = 256` | 覆盖 CUDA graph 阶梯上限；RTX PRO 6000 上 M=128 约 230 μs vs 930 μs |
| `kernel/triton/fp8_block_linear.py:142-143, 251-255` | M==1 / M≥64 | W8A16 split-K GEMV / W8A8 动态量化块 GEMM（block 32/4 or 64/8） |
| `kernel/triton/fp8_pertensor_linear.py:325-330` | **与 M 无关** | 是否量化激活是"部署属性"，不是 batch 属性——W8A16 在 M=1 只值 ~0.5% |
| `moe/fused_ds_fp4.py:132-137` | `_GROUPED_MIN_ROUTES = 768` | routes = T·top_k 低于此走 per-route GEMV，高于走 grouped GEMM |
| `moe/fused_mxfp4.py:12-16` | `MXFP4_DECODE_MAX_TOKENS = 16` | 常驻时 ≤16 token 走 decode 路径 |
| `moe/unquantized.py:31-36` | `is_prefill` | bf16 专家：grouped GEMM vs per-route GEMV |
| `attention/triton.py:91, 165-220` | `is_decode` + tile gate | extend vs decode 两套核 |

#### 4.3 tuned config 的真相：那张表**只在 H100 + triton 3.5.1 生效**

- `try_get_optimal_moe_config`（`moe/fused.py:168-189`）的键是 `(E, config 的 N=intermediate, device_name, triton.__version__)`，路径 `configs/triton_<ver>/E={E},N={N},device_name={dev}.json`，可用 `FREETOKEN_MOE_CONFIG_DIR` 覆盖。
- **仓库里只 ship 了一个文件**：`moe/configs/triton_3_5_1/E=128,N=704,device_name=NVIDIA_H100_80GB_HBM3.json`。所以**在任何其它设备/triton 版本上，全部退回 `get_default_config`（`fused.py:120-141`：M≤E 时用 16/32/64 GROUP_SIZE_M=1，否则 64/64/32 GROUP_SIZE_M=8）**。
- 这意味着"在 4090/5090 上摸到最优 MoE tile"这件事**仓库没给你**，得自己扫。想要就按 `benchmark/perf.py` 的门路扫，再放进 `FREETOKEN_MOE_CONFIG_DIR`。

#### 4.4 为什么到处是"手写死配置"而没有 `triton.autotune`

因为 **decode 路径要被 CUDA graph 捕获**，而 autotune 会在运行期基准测试、在捕获期做非法操作。文档写得很直白（`moe/fused_nvfp4.py:27-28`：*"Decode is captured into a CUDA graph, so the config must be fixed (no triton.autotune)"*）。所以：
- 离线的 tile 常量 + 注释里的实测百分比替代了 autotune；
- 需要 autotune 的地方只在 prefill/非图路径；
- `kernel/triton/autotune_cache.py` 用 `cache_results=True` 把扫出来的结果持久化，避免"每个新进程都在 warmup 里重新扫、启动要几秒"。

**这条规律值得推广**：在这个仓库里，看到"硬编码的 tile 常量"，先问"它是不是在 CUDA graph 里"——大概率是，而注释里的百分比就是当初的测量记录。

#### 4.5 内核作者真正在用的性能杠杆（读内核时按这 4 类找）

| 杠杆 | 实例与实测（来自注释） |
|---|---|
| **访存宽度/向量化** | FP4 码用 int32 宽载（每元素 8 码/4 字节）让 HBM 发出宽合并事务：gate/up GEMM 的访存上限从 ~65% 提到 ~85%（`moe/fused_nvfp4.py:160-174`） |
| **去掉 LUT gather** | e2m1→fp16 用位运算而非 L1 LUT：旧 decode GEMV 卡在权重带宽屋顶的 ~36%（H100 lm_head），位运算版到 ~64%，且位完全一致（`nvfp4_linear.py:12-19`） |
| **减少 smem 往返 / 换操作数** | "wdot" 交换操作数把反量化权重当作 MMA lhs，去掉每 tile 一次 smem 往返：从 ~50% Memory SOL 提上来（`nvfp4_linear.py:357-377`） |
| **grid 量化与波次** | split-K 目标 program 数（1024）与"接近恰好一波的 grid 会量化得很糟（一整波 + 一个拖尾小波）"——宁可两波（`nvfp4_linear.py:523-543`）；优先选能整除 K-tile 数的 split，让程序走无掩码的 EVEN_K 循环（快 15-30%，`:569-576`） |
| **PDL** | `is_sm90_supported()` 门控（`activation.py:45-49` 有理由说明）；`norm.py:18-19` 明确写了用途——**"hide launch latency in decode graph replays"** |
| **避免额外 kernel launch** | split-K 归约用自重置到达计数器融进 GEMV，省掉 M==1 decode linear 上每步 ~2 μs 的尾巴（`nvfp4_linear.py:100-116`） |
| **对齐** | 反量化 chunk 保持 N 256 对齐——奇数 N（如 13107）会让 cuBLAS 掉到约 4× 慢的 misaligned kernel（`nvfp4_linear.py:726-729`） |

#### 4.6 预编译 wheel 覆盖什么、不覆盖什么

- **只预编译 C++/CUDA 扩展内核，不含 Triton 内核，也不含 MoE tile 配置 JSON**（`kernel/aot.py:146-166`）：store / index / fast_index_copy / fast_index_copy_multi / batch_memcpy / radix。
- 形状覆盖是**手工维护的模型表** `kernel/aot_models.py:137-406`，其 docstring `:10-12` 警告：推导必须与运行期调用点**手工保持同步**，一旦漂移就会"按名字找不到预编译、静默退回 JIT（而 JIT 需要 nvcc）"。
- 结论：**Triton 内核的冷启动只靠 Triton 自己的磁盘缓存**（由 `autotune_cache.py` 填充），预编译 wheel 对它无效。所以"启动慢"要把这两条路分开查。

#### 4.7 注意力后端的性能事实

- 自动选择的硬件树（`engine.py:127-128` 的 docstring）：**sm_100 → `trtllm`；sm_90 + sgl_kernel → `"fa,fi"`；有 flashinfer → `fi`；否则 `triton`**。
- **`triton` 是唯一支持 SWA、也是唯一 `consumes_attn_spec=True` 的后端**——所以带窗口/sinks/sm_scale 的模型被钉在 triton（`engine.py:196-202`）。反过来，fi/fa/trtllm 遇到 per-call spec 会**报错而不是静默忽略**。
- 约束：`fi` 要求 `page_size == 1`（`fi.py:57, 66`）；`fa` 在 sm_120 上直接报错（"sgl-kernel ships no FA3/FA4 kernels for sm_120/121"）；`trtllm` 需要 sm_100a/103a 的 cubin。
- `HybridBackend`（`attention/base.py:94-130`）只在 `--attention-backend` 写成 `"prefill,decode"` 时构建，且**只有 decode 侧会被 CUDA graph 捕获**——所以"给 prefill 换个更快的后端""从不影响解码稳定性。
- 一个反直觉的默认：`fi` 明确用 `backend="fa2"` 而不是 fa3（`fi.py:120` 的注释："flashinfer fa3 is slow, use fa2 instead"）。


### 阶段 5：极致案例研究（选读，但收获最大）

`models/deepseek_v41/engram_table.py` 开头的 20 行 docstring（`:1-20`）是全仓最精彩的一段性能设计说明，值得逐字读：
- 每层一张 `[384M, 256]` fp8 表 = 98 GB + `[384M, 8]` scale 3 GB
- 表**留在 checkpoint 原处**，用 `RowStore` 映射成一段 extent，io_uring + O_DIRECT 每次只读一个 token 需要的 24 行（去重）
- CUDA graph 场景下：采样出的 token 在 GPU 上，所以主机要**读回**它 → 填进 graph 的 pinned staging → 释放一个 flag，被捕获的 `lookup` **WAIT** 等它
- **worker 线程必须在 graph launch 之前启动**：因为 40 层的 graph 太大，driver 的 launch 调用本身会在 GPU 停在 WAIT 时阻塞，若在 launch 之后再调度填充就会死锁
- 失败要在 launch 之后、输出被消费之前暴露

这不是"优化技巧"，这是**约束求解**——把它读懂，你对"图捕获 + 主机喂数"这一整类问题就有了判据。

对照读 `models/qwen4_exp/ple_disk.py:247`（同一个协议的另一个实现）和 `csrc/row_store/row_store_ext.cpp`（O_DIRECT / io_uring 的 C++ 侧）。

---

## 3. 调优旋钮总表

按"你想动什么"分组。**加粗的是最常动的**。

### (a) 显存预算切分
| 旋钮 | 作用 | 何时动 |
|---|---|---|
| **`--memory-ratio`**（0.9） | 可用显存的占比；`(1-ratio)` 留给 CUDA graph/激活 | OOM 就降；有余量想提并发就升 |
| `--num-pages` / `--num-tokens` | 直接钉 KV 容量（覆盖 auto） | 要做可复现实验时 |
| `--kv-reserve-tokens`（8192） | `--moe-cache-auto` 之前先给 KV 预留的 token 下限 | 长上下文时提高 |
| **`--moe-cache-size` / `--moe-cache-rate` / `--moe-cache-auto`** | GPU 专家槽位数 | 三选一；auto 默认开 |
| `--page-size`（1） | KV 页大小 | DSV4 强制 128；TRTLLM 需 16/32/64；SWA 强制 1 |
| `--swa-*` / `--num-swa-pages`（rebuild） | 窗口池大小 | SWA/DSV4 模型 |

### (b) MoE 策略
| 旋钮 | 作用 | 何时动 |
|---|---|---|
| **`--moe-strategy`**（auto） | fused / offload / cpu / hybrid | 想强制对比时显式指定；GB10 上 auto=fused |
| **`--quant-backend`** | 按 `layer.kind=name` 选核，如 `moe.nvfp4=b12x` | 换内核做 A/B |
| `--moe-cpu-layers` | offload 下哪些层交给 CPU decode（列表/数量/比例/auto） | fp8 专家不支持；Windows/WSL 用 auto |
| `--moe-hybrid-max-fetch`（-1=auto） | hybrid 每层每步最多抓多少专家 | 覆盖 profile 分式 |
| `--moe-cpu-threads`（物理核数） | CPU worker 线程数 | CPU 算力没吃满时 |
| `--moe-prefill-hit-d2d` | prefill 时命中专家走 D2D，只流未命中（CUDA≥13） | prefill 慢且命中率高时 |
| `--disable-moe-prefill-overlap` | 关掉权重的双缓冲预取 | 只在排查正确性时 |
| `--moe-cache-policy` | **只有 `lru` 一个选择**（`args.py:717`，`choices=["lru"]`；`offload_cache.py:149-151` 断言 `policy_ids={"lru":0}`）——是个为将来留的占位 | 现在无需动 |
| `--expert-load`（auto） | 专家 bank 读入主机 RAM 的并行/串行 | 启动慢且内存紧 |

### (c) 内核与后端
| 旋钮 | 作用 | 何时动 |
|---|---|---|
| `--attention-backend` / `--attn`（auto） | `trtllm`/`fi`/`fa`/`triton`/`dsv4_sparse`/`dsa`…；支持 `prefill,decode` 成对 | 想要 prefill/decode 用不同后端时 |
| `--dtype` | bf16/fp8 等 | 显存与精度权衡 |
| `--disable-pynccl` | 关掉 TP 的 pynccl | 只在排查 |

### (d) 调度与图
| 旋钮 | 作用 | 何时动 |
|---|---|---|
| **`--max-running-requests`**（4） | 并发上限；**同时钳住 CUDA graph bs 上限** | 吞吐主要旋钮 |
| `--cuda-graph-max-bs` / `--graph` | 捕获的最大 bs | 显存紧张就降；小值会让大 batch 掉出 graph |
| `--max-prefill-length` / `--max-extend-length`（8192） | 分块 prefill 的块大小 | 长 prompt 的 TTFT 与显存峰值 |
| `--decode-log-interval`（40） | 状态行频率 | 观察时调小 |

### (e) 环境变量（隐藏开关）
```bash
# 内核/JIT
FREETOKEN_DISABLE_JIT=1               # 发布验证：cache miss 直接失败
FREETOKEN_KERNEL_CACHE_DIR=...        # 预编译 .so 目录
FREETOKEN_DISABLE_KERNEL_CACHE=1
FREETOKEN_KERNEL_CACHE_JOBS / FREETOKEN_TRITON_CACHE_RESULTS
FREETOKEN_ALLOW_CUDA_MISMATCH=1       # nvcc 与 torch 的 CUDA major 不一致时放行
FREETOKEN_ROCM_ARCH / FREETOKEN_GGUF_HOST_CXX

# MoE 带宽与搬运（性能最关键的一组）
FREETOKEN_H2D_BLOCKS_PER_BANK=32      # 仅 NVLink-C2C 主机：gather 220→410 GB/s
FREETOKEN_HYBRID_FETCH=lowest_id      # 退回"按 id 最小抓取"，默认 recency
FREETOKEN_HYBRID_OVERLAP
FREETOKEN_SKIP_FAST_INDEX_COPY=1
FREETOKEN_FUSED_COPY
FREETOKEN_BANK_CUDA_ALLOC / FREETOKEN_SKIP_BANK_PIN / FREETOKEN_PIN_BUDGET_GB
FREETOKEN_BENCHBW_PATH                # 显式指定 profile

# CPU MoE
FREETOKEN_CPU_MOE_ISA / _SCALAR / _NO_VNNI / _NO_AVX512VNNI / _PF_BLOCKS / _SPIN / _FLAG_SYNC

# 调度/精度
FREETOKEN_DISABLE_OVERLAP_SCHEDULING=1   # 回到 normal_loop（排查 overlap 问题时）
FREETOKEN_MAMBA_SSM_DTYPE=float16        # GDN 状态池减半（精度换显存）
FREETOKEN_UNIFIED_MEMORY / FREETOKEN_SWA_EVICTION_INTERVAL

# 磁盘表
FREETOKEN_ENGRAM_IO_URING / FREETOKEN_ENGRAM_SYNC / FREETOKEN_PLE_IO_URING / FREETOKEN_PLE_SYNC
FREETOKEN_PLE_READER_THREADS / FREETOKEN_ROW_STORE_READER_THREADS
```
（完整 53 个：`grep -rhoE '"FREETOKEN_[A-Z0-9_]+"' python/ | sort -u`）

---

## 4. 优化循环：照着仓库自己的做法走

```
① 建立基线
   benchmarks/bench_decode_moe.py --backend offload,cpu,hybrid   # 端到端 bs=1 decode tok/s
   curl /v1/stats                                                # decode_tps / prefill_tps / kv 占用 / vram
   看 scheduler 日志的 "gen throughput (token/s)"（--decode-log-interval 调小到 5~10）

② 定位瓶颈在哪一类
   显存不够      -> cache_budget.py 的算术 + /v1/stats 的 kv/mamba/swa + vram_bytes
   decode 慢     -> 先看命中率/池占用，再看 MoE 策略（benchbw 判据），最后才看内核
   prefill 慢    -> 分块大小、前缀命中率、--moe-prefill-hit-d2d、双缓冲是否被关
   启动慢        -> benchmarks/bench_load_weight_generic.py（serial vs parallel O_DIRECT vs FTW）

③ 让策略自适应而不是手调
   ft bench bw [--dtype nvfp4,bf16]      # 每卡一次，写 <gpu-uuid>.json
   -> 之后 --moe-strategy auto 会自动读（bench_profile.py）

④ 只测一个变量
   基准脚本都是"每个后端一个子进程"，就是为了避免串扰；微观用 benchmark/perf.py:10 perf_cuda

⑤ 验证
   性能改动必须给 main 与本分支的 A/B（CONTRIBUTING.md 明确要求：tokens/s，prefill 相关还要 TTFT）
```

**`ft bench bw` 这条闭环值得单独记**（`moe/bench_profile.py`）：
- profile 路径 `$XDG_CACHE_HOME/freetoken/benchbw/<gpu-uuid>.json`，**按 GPU UUID 分开**（带宽随插槽不同）
- join key 是**专家格式**，不是模型——因为比值由 `(format, hardware)` 决定，所以一个 workload 的测量可迁移到同格式的别的模型（`bench_profile.py:6-9` 明确说明）
- 加载时校验 GPU 名字，不匹配就**忽略而不是信任**（`:104-110`）
- 结论保守：全票 hybrid 才 hybrid（`:144-153`）
- 分式优先取**并发测量**的重叠带宽（真实争用状态），老 profile 才退化为独立带宽相除（`:185-190`）

这套设计的可迁移经验：**测量产物要带硬件指纹、要与被测对象解耦、要在缺数据时保守回退**。

---

## 5. 性能上的常见误判（很容易踩）

1. **拿一次测量当结论**。仓库自己给了反面教材：`tests/kvcache/radix/README.md:52-69` 记录了一个 `>=` 边界 bug，**~8500 次随机运行都没复现**，因为随机轨迹几乎不会落在"两个墓碑之间恰好一个窗口"这种精确形状上。性能测量同理——均值会掩盖尾部，随机负载会掩盖边界。
2. **忽略采样协议的 token 预算**。`tests/README.md:74-76` 记录了同一个题 greedy ~4k token、推荐采样三条 4.1k/10.7k/10.0k——**用错预算测出来的 tok/s 没有可比性**。
3. **以为 `--memory-ratio` 只管 KV**。它同时管专家缓存和 graph/激活余量；调它会同时移动吞吐和稳定性。
4. **以为 hybrid 总是更快**。它的前提是 CPU 带宽显著高于 PCIe（阈值 2.0×）；否则 CPU 侧 GEMV 成为新的下限。
5. **忽略"混合 batch 拖累全批"**：一个非 greedy 请求会让整批走概率采样路径（`engine/sample.py:59-99`）。
6. **在 capture 范围外测 CUDA graph 收益**。`can_use_cuda_graph` 要求 decode 且 `bs <= max_graph_bs`（`graph.py:204`），而 graph bs 又被 `--max-running-requests` 钳住（`engine.py:1230`）——调大并发可能反而让某些档位的 graph 失效。
7. **忘了 rebuild 会清空前缀缓存**：`/v1/cache/rebuild` 是 idle-only 且会重建前缀缓存（`scheduler/cache.py:556-572`），所以"改完参数再跑"与"一开始就用该参数"的冷启动前缀命中率不同。
8. **CI 不跑 pytest**，也没有性能回归门禁。任何"变快了"的结论都必须自己 A/B。

### 5.1 声称做了、但实际没生效的（读性能代码时优先怀疑这一类）

这些我都逐行确认过。**它们不是 bug，但会让你对性能的判断失真**——尤其在你打算"照着注释理解优化"的时候。

| 位置 | 现象 | 影响 |
|---|---|---|
| `engine/engine.py:1061` | 采样 token 的 D2H 目标不是 pinned（见 1.1） | 双 stream 重叠可能未兑现 |
| `kernel/store.py:12`、`kernel/index.py:12` | PDL 代码路径**编译进去了**（`csrc/jit/store.cu:39,52`、`index.cu:48,58` 有 `griddepcontrol`），但默认 `use_pdl=False` | decode 关键路径上的两个 kernel 没用 PDL |
| `kernel/triton/sampling.py:134` | `softmax(..., enable_pdl=...)` 接受该参数但**函数体从未引用** | triton 采样回退路径上 PDL 是空开关 |
| `engine/sample.py:38` | 只有 flashinfer 路径真正用上 `enable_pdl` | 装了 flashinfer 才有收益 |
| `kernel/fast_index_copy.py:130-139, 205-209` | `fast_index_copy_jit(priority=..., sync_flag=...)`、`launch_high/launch_normal`、`update_copy_flag_jit` **无生产调用点**（只有 `aot.py:91-92` 提到名字） | 别以为有优先级/同步标志在起作用 |
| `kernel/pinned.py:27-39` | `copy_to_pinned_tensor` / `create_pinned_tensor_like` **只被测试使用** | 生产路径没有走它 |
| `engine/sample.py:88, 90` | `@nvtx_annotate("Sampler")` 与 `with torch.cuda.nvtx.range("Sampler")` **嵌套了两个同名 range** | profile 时不要被双份区间误导 |
| 各 `models/*/model.py` 的逐层 NVTX | 逐层 range 是**宿主侧调用**，在 graph replay 下不产生逐层区间 | **不要因为图捕获后的 trace 里没有逐层区间，就以为代码里没写**——图 replay 下整个 decode 步就是一个 graph 节点 |
| `engine/graph.py:191-192` | 注释说"Keep the offload cache warmed for capture"，但代码顺序是 warm(`:190`) → capture(`:193-194`) → reset(`:195`) | 注释读起来像是反的；以代码顺序为准 |
| `engine/config.py:52-53` | 注释说 hit-D2D "Needs CUDA >= 12.8"，实际门是 **13.0**（`kernel/batch_memcpy.py:42-53`） | 按 12.8 规划环境会直接抛错 |
| `moe/offload_kernels.py:8` | 默认 LRU 内核来自外部 `flashlib==0.3.0`，**不在本仓库** | 读不到非 hybrid 的 LRU 实现；只有 hybrid 变体在仓库内 |
| `moe/offload_cache.py:914-929, 849-853` | `lru_stats` 的累加**在图内可安全重放**（每步按真实路由累加，只有捕获时有一次性的热身增量，全生命周期 <0.1%）；但 `collect_decode_freq` **明确不是图安全的**（注释自己说了） | 采集统计时要分清哪个能进图 |
| `engine/config.py:55` | `moe_collect_stats` **没有对应的 CLI flag** | 想开得改 config 或代码 |

**这条经验的推广**：在这个仓库里，**documented ≠ implemented**。我已经核对出至少 4 处"文档/注释与实现不一致"（1.1 的 pinned、阶段 3 的 hybrid 分式、阶段 3 的 CUDA 版本、本表的 PDL/图注释）。所以读性能代码的正确姿势是：**注释告诉你作者想做什么，代码告诉你实际做了什么，两者都要读，并且永远以后者为准。**

---

## 6. 自测：能回答这 8 个问题就算读通了

1. 一个 decode step 里，设计上 CPU 应该在哪一刻不再需要等 GPU？**实际代码兑现了吗**？（提示：1.1）为什么那个设计在逻辑上不会读到未完成的 token？
2. CUDA graph 捕获前那次 eager 前向是干什么的？为什么 MoE offload cache 每次捕获前要 reset？
3. `--max-running-requests` 和 `--cuda-graph-max-bs` 的关系是什么（谁钳谁、为什么）？
4. offload 与 hybrid 分别受什么限制？`ft bench bw` 的 2.0× 阈值在保护什么？分式到底取自哪个公式？
5. 为什么 `fast_index_copy` 被当作最关键的内核？它省掉了什么？为什么预编译 wheel 不含 Triton 内核？
6. Engram 的填充线程为什么**必须在 graph launch 之前**启动？如果放在之后会发生什么？
7. 为什么 decode 路径上到处是"手写死的 tile 常量"而没有 `triton.autotune`？这跟哪条约束有关？
8. `auto` 模式下 NVFP4 专家**为什么永远不会自动选 marlin/b12x**？想用它们要怎么写命令？

---

## 7. 我可以继续带你走的线

- 「跟我一起数一个 decode step 的时间线」——逐 kernel 逐同步点，标出哪些是必须的、哪些是优化
- 「offload 的 miss 率 → 延迟模型」——用 `bench_offload_cache_copy.py` 的维度建一个纸面模型，预测 `--moe-cache-size` 的拐点
- 「`ft bench bw` 到 `--moe-strategy auto` 的完整闭环」——逐行
- 「hybrid 的 Q16 整数解在优化什么」——把那个 makespan 目标函数推一遍
- 「读一个 Triton 内核」——`norm.py` 或 `attention.py`，自底向上
- 「Engram/PLE 磁盘表的约束求解」——把 20 行 docstring 展开成设计文档
