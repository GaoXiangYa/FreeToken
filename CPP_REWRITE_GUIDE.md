# FreeToken → C++ 重写：读透路线与规格提取

> 目标读者：准备用 C++/CUDA 重写本项目的工程师。
> 与 [`SOURCE_READING_GUIDE.md`](SOURCE_READING_GUIDE.md)（怎么读懂）和 [`INFERENCE_OPTIMIZATION_READING_GUIDE.md`](INFERENCE_OPTIMIZATION_READING_GUIDE.md)（怎么调快）不同，本文只回答一件事：**为了重写，该读什么、读成什么样、怎么验证读对了。**
> 行号基准 `377a3bc`；模块级详解见 [`MODULE_DOCS/`](MODULE_DOCS/README.md)。
>
> **如果你的目的是学习（练系统/性能/算子能力）而不是上线，请先读 [`LEARNING_BY_REWRITING.md`](LEARNING_BY_REWRITING.md)——它给出的读法与本文件不同（自己写内核、精选模型族、以实验验收）。**

---

## 0. 结论先行：三条决定成败的判断

### 判断一：不要 1:1 翻译，切一条"竖切面"
全仓 **93,387 行 Python + 20,887 行 Triton + 13,360 行 C/C++**。等比例翻译 = 不可控工程。

**只移植最小但完整的一条纵切面**，它必须同时压到最难的四类契约：
```
HTTP(SSE) → tokenizer → 调度器(overlap) → 分页 KV + 前缀缓存 → 注意力 → 采样 → detokenize → SSE
                                        ↘ CUDA graph 重放
```
推荐第一条切片：**一个小 dense 模型 + 单卡 + bf16 + OpenAI chat/completions 流式 + 前缀缓存**。
理由：没有 MoE、没有量化、没有多模态，但**保留了全部并发/内存/时序难点**。它大约是全仓的 **15%**，却要求你读透 80% 的关键契约。

之后按 §7 的 P1→P6 逐层加回 fp8、MoE offload、多模型、多模态。

### 判断二：Python 代码此时是"规格"，不是"实现"
重写时你需要的不是"这行 Python 在做什么"，而是：
| 要提取的 | 从哪来 | 交付物 |
|---|---|---|
| **字节级数据布局** | `kvcache/`、`moe/_BANK_SCHEMAS`、`layers/quantization` | `struct` 定义 + 对齐/stride/dtype 表 |
| **不变量** | 源码里的 `assert`（如 `Req.__post_init__`、`cache_size >= 2E`） | C++ 的 `assert`/contract + 单测 |
| **状态机** | `Req` 生命周期、`Scheduler` 循环、parser 的流式状态 | 状态转移图 + 事件表 |
| **时序/同步契约** | `overlap_loop` 的三道围栏、双缓冲 ready/release、握手 | 时序图（stream/event 级） |
| **数值契约** | 量化格式、累加精度、reduction 顺序 | tolerance 预算 + 测试向量 |

**读代码的正确产出物是一份 5 页的 spec，而不是一段笔记。**

### 判断三：优先做"差分测试"，而不是优先写 C++ 代码
这是全仓给你的最大红利：
- 线上协议是 **msgpack + `__type__` 信封**（`utils/mp.py:25`、`message/utils.py`），**C++ 侧可以直接说同一种线协议** → 可以**一个进程一个进程地替换**，剩下的 Python 进程继续工作；
- 仓库自带**独立参考实现**（见 §6），可直接当 golden oracle。

组合起来：**Python 引擎跑 oracle，C++ 新模块影子运行，逐步比对中间张量/消息**。这比"全写完再联调"快一个数量级。

---

## 1. 可移植边界（用数据说话）

按"重写时要做什么"分类，而不是按目录：

| 类别 | 模块 | LOC | `torch.` 引用 | 重写策略 | 预估占比 |
|---|---|---:|---:|---|---:|
| **A. 已有 C/C++，可复用** | `kernel/csrc/`（除 vendored gguf 约 7,550 行） | ~5,810 | — | **直接用**：`pinned_tensor.cpp`、`jit/*.cu`、`row_store`、`cpu_moe`、`pynccl` 已经是生产级 CUDA/C++ | 6% |
| **B. 纯 I/O 与状态机（无张量语义）** | `daemon/`(3,057)、`shell/`(1,515)、`message/`(421)、`distributed/`(147)、顶层 CLI(2,513) | ~7,650 | **daemon=0**、shell=0、message=9、distributed=18 | 机械移植：HTTP/JSON/proc/日志。可放到最后，甚至**保留为辅助工具** | 8% |
| **C. 张量编排（torch 密集但算法直白）** | `scheduler/`(2,491)、`engine/`(2,495)、`kvcache/`(4,968)、`layers/`(4,420)、`server/`(11,312，torch 仅 4) | ~25,700 | engine 133 / scheduler 83 / kvcache 335 / layers 454 | **先写数据契约再写逻辑**；torch API 需逐一替换（§4） | 28% |
| **D. 数值内核（必须重写或换库）** | `kernel/triton`+`kernel/fla`(20,887)、`attention/`(4,219)、`moe/`(5,870) | ~31,000 | attention 698 / moe 493 | **优先换库**（flashinfer C++ / CUTLASS / sglang-kernel），只手写库不覆盖的部分 | 33% |
| **E. 模型定义与加载（可裁剪）** | `models/`(21,243) | 21,243 | **1,765（全仓最高）** | 只移植切面用到的族；其余按需 | 23% |
| **F. 格式与解析（纯逻辑，好移植）** | `checkpoint/`(1,165)、`tokenizer/`(1,218)、`mm/`(1,014)、`server` 的两个 parser(4,685) | ~8,100 | 很低 | 逐行移植，测试驱动 | 9% |

**读法上的直接推论**：
- **A + F 是"读了就能写"**（约 15% 代码，两三周）→ 先做，建立信心与测试骨架。
- **D 是"读了也要重新设计"**（33%）→ 只读契约与调用方式，算法换库，不要逐行翻译 Triton。
- **E 是"读了才知道能砍多少"**（23%）→ 先只读 `register.py` + 1 个族，确认骨架，其余 18 个族**现在不要读**。

---

## 2. 读法：四层剥离（每个模块都用这个模板）

对一个模块，按顺序只问四个问题：

```
① 语义层（必须 1:1 保留）  —— 这份代码保证了什么？
     不变量 / 状态转移 / 数值语义 / 错误分类
② 结构层（可映射）        —— 用什么 C++ 构造表达？
     dataclass→struct、Protocol/ABC→concept/接口、Union→variant、
     Optional→optional、dict→flat_map、生成器→迭代器/回调
③ 依赖层（必须替换/桥接）  —— 它借了 Python 生态什么？
     torch / transformers / jinja2 / fastapi / pydantic / zmq / triton / tvm-ffi
④ 噪声层（不必读）        —— 哪些是脚手架？
     CLI help、日志格式、per-agent 配置、desktop 适配、benchmark 脚本
```

**判定标准**：如果一句话说不清"这个模块保证什么"，说明还在结构层打转，没有到语义层。

### 每个模块要产出的一页 spec（模板）

```markdown
## <module> 规格
1. 职责边界：输入/输出/调用方（3 行）
2. 数据结构：字节布局表（字段/类型/对齐/不变量）
3. 状态机：状态 × 事件 → 动作（表格或图）
4. 时序：涉及哪些线程/stream/event，顺序约束是什么
5. 错误契约：哪些情况报错、错误如何分类/上抛
6. 数值契约：dtype 提升、累加精度、tolerance
7. 测试向量：从 tests/ 提取的可比对数据/golden
8. 可裁剪项：本模块哪些分支第一版不实现
```

---

## 3. 重写导向的阅读顺序（与"学习"顺序不同）

学习顺序是"跟着请求走"；**重写顺序是"先契约、再时序、最后算法"**。

### 第一遍：只读"契约"（约 4 天）——目标是画出 data layout 表
| 顺序 | 读什么 | 为什么对重写最要紧 |
|---|---|---|
| 1 | [`MODULE_DOCS/01-core-engine.md`](MODULE_DOCS/01-core-engine.md) §3 `Req`/`Batch`/`Context` | 全部状态字段 + 3 个不变量。这是 C++ 的 struct 雏形 |
| 2 | [`MODULE_DOCS/05-kvcache.md`](MODULE_DOCS/05-kvcache.md) §3-4 + `kvcache/base.py` | **page_table 寻址 + 池族 + 三套 radix 的 key 规则**。全项目最硬的数据契约 |
| 3 | [`MODULE_DOCS/06-moe.md`](MODULE_DOCS/06-moe.md) §3.1-3.2 | `slot_for_id`/`id_of_slot`/bank 布局 + 每槽字节公式 |
| 4 | [`MODULE_DOCS/10-layers-quant.md`](MODULE_DOCS/10-layers-quant.md) §4 + `moe/_BANK_SCHEMAS` | 每种量化格式的 bank 数量与行布局（决定 C++ 的内存布局） |
| 5 | [`MODULE_DOCS/09-attention.md`](MODULE_DOCS/09-attention.md) §3 | `AttnType` × 池族 × backend 的三层映射 |
| 6 | [`MODULE_DOCS/07-models-core.md`](MODULE_DOCS/07-models-core.md) §3-4 | `ModelSpec`/`ModelConfig`：模型→引擎的契约面 |
| 7 | [`MODULE_DOCS/14-tokenizer-message-llm.md`](MODULE_DOCS/14-tokenizer-message-llm.md) §3 | **msgpack 线协议全表**（逐进程替换的基础） |

**第一遍的产出**：`spec/00-layouts.md`——一张 `struct` 表 + 每张 GPU/主机张量的 shape/dtype/对齐/生命周期。
**验收**：能不看代码写出 `Req`、`Batch`、`page_table`、`OffloadMoeCache` 的字段与不变量。

### 第二遍：读"时序"（约 3 天）——目标是画时序图
| 顺序 | 读什么 | 关注点 |
|---|---|---|
| 1 | [`MODULE_DOCS/02-scheduler.md`](MODULE_DOCS/02-scheduler.md) §5.1 `overlap_loop` | **三道 `wait_stream` 围栏各自防什么**；这是 C++ 里最容易写错的地方 |
| 2 | [`MODULE_DOCS/01-core-engine.md`](MODULE_DOCS/01-core-engine.md) §6 graph.py + §9.3 | 捕获顺序（warmup→倒序捕获→共享 mempool）+ replay 前的 D2D 拷贝 |
| 3 | [`MODULE_DOCS/06-moe.md`](MODULE_DOCS/06-moe.md) §5.4-5.5 | 双缓冲 ready/release event + 256KB 批量拷贝阈值 |
| 4 | [`MODULE_DOCS/06-moe.md`](MODULE_DOCS/06-moe.md) §5.3 | CPU/GPU flag 握手（stream memop vs host-func） |
| 5 | [`MODULE_DOCS/03-server-http.md`](MODULE_DOCS/03-server-http.md) §6 | 进程拓扑 + ready 门 + abort 延迟释放 |

**产出**：`spec/01-timing.md`——每步的 stream/event 顺序表（C++ 里等价于 `cudaStreamWaitEvent` 调用序列）。
**验收**：能解释"去掉围栏 A 会出现什么症状"（提示：复用刚释放的 `table_idx` 时被上一任残留写覆盖）。

### 第三遍：读"算法"（按切面需要，逐模块）——目标是可移植的伪代码
只读切面用到的：一个模型族的 `model.py`/`attention.py`/`weight.py`、`layers` 的并行 Linear/MoE 装配、`kvcache` 的 `_tree_walk` 与 `allocate_paged`、`scheduler` 的 `PrefillAdder`。
**产出**：每个算法一份"输入/输出/不变量/复杂度"的伪代码。
**验收**：能对给定输入手算出输出（例如 page 匹配长度）。

### 明确"现在不要读"的清单（可省 2-3 周）
- **其余模型族**（`models/` 21k 行、19 个目录里只读切面用到的 1-2 个，`08` 篇有全表）；
- **55 个 Triton 文件里的 40+ 个**（只读切面用到的 attention/量化/采样，其余换库）；
- `daemon/`、`shell/`、`launch.py`、`control_cli.py`、`cache_report.py`（控制面与 UI）；
- `mm/`、`models/deepseek_v41`、`models/deepseek_v4`、`models/glm5_next`、`models/qwen4_exp`、`models/minimax_m3`（多模态与特殊结构）；
- `checkpoint/`（重写初期直接读 safetensors 即可，FTW 是加载优化不是语义）；
- `benchmarks/`、`scripts/`、`.github/`。

---

## 4. Python 生态依赖 → C++ 替换表

这是重写工作量的真正来源。逐个决策：

| Python 依赖 | 用在哪 | C++ 方案 | 风险/建议 |
|---|---|---|---|
| **torch（Tensor 语义）** | 全仓，1,765+ 引用（`models`） | ① **libtorch/ATen**（最快落地，保留 Tensor 语义）② 自研 Tensor + CUDA driver（干净但要重实现 stride/broadcast/dtype 提升） | **建议先用 ATen 打通全链，再逐个换掉热点**。不要一开始就自研 Tensor |
| **torch.cuda.Stream/Event** | 调度、graph | `cudaStream_t` / `cudaEvent_t` 直接用（语义 1:1） | 注意 torch 有"current stream"隐式上下文，C++ 必须显式传 |
| **torch CUDA graph** | `engine/graph.py` | `cudaGraph` API（1:1） | 捕获期禁止同步与分配——这条约束在 C++ 里一样 |
| **torch.distributed** | TP | `nccl` 直接用（psnccl 已在 `csrc/src/pynccl.cu`） | 建议先用 `ncclComm_t` |
| **transformers（tokenizer/config）** | `tokenizer/`、`models/*/config.py` | ① HF `tokenizers`（Rust，有 C API）② 自己读 `tokenizer.json` 实现 BPE | 配置解析手写即可（JSON） |
| **jinja2（chat template）** | `tokenizer/tokenize.py` | ① **`minja`**（llama.cpp 的 jinja2 子集）② 每族预编译成 C++ 代码 ③ 保留 Python sidecar | **最大的隐藏风险**：本仓有 4 个族相关决策点（thinking mode / DSV4 encoder / inline-system preserve-fold / effort），见 `MODULE_DOCS/14` §4.2 |
| **fastapi + pydantic** | `server/` | `cpp-httplib`/`Drogon` + `nlohmann::json` + 手写校验 | 11k 行里大部分是协议适配，逻辑直白 |
| **pyzmq** | 进程间 | `libzmq` 直接用，**保持 msgpack 信封不变** | **关键使能项**：可以逐进程替换 |
| **msgpack** | 线协议 | `msgpack-c` | 兼容 `__type__` 信封即可 |
| **Triton** | 20,887 行 | ① 换库：flashinfer C++ / CUTLASS/CuTe / sglang-kernel ② 手写 CUDA | **不要逐行翻译 Triton**；先看该 kernel 的"契约 + 阈值"，再找等价库 |
| **flashinfer / sgl_kernel / vllm** | 注意力、量化、采样 | 它们本身就是 C++/CUDA，直接链 | 这些是**你的朋友**，不是要替换的对象 |
| **tvm-ffi + JIT** | `kernel/utils.py` | 静态编译 + 编译期/启动期选择；保留 AOT wheel 思路 | 省掉运行时 nvcc |
| **flashlib（外部 LRU 内核）** | MoE 默认准入 | 必须自己写（源码不在仓库） | **唯一的"算法缺口"**；但仓库内有 hybrid 变体 `offload_kernels.py:290-410` + CPU 镜像可作参考与 oracle |
| **multiprocessing** | 进程编排 | `fork/exec` + 自有 supervisor | 建议 C++ 阶段**先改多线程单进程**（无 GIL） |
| **safetensors / huggingface_hub** | 权重 | `safetensors-cpp` 或直接按格式解析 | 格式简单 |
| **io_uring / O_DIRECT** | 磁盘表 | 直接系统调用 | `csrc/row_store` 已有实现可直接抄 |
| **PIL / numpy** | 多模态、序列化 | 第一版不移植多模态 | 省掉 |

---

## 5. 必须逐字节对齐的"数据契约清单"

重写时**先冻结这些**，否则后面所有差分测试都会失败：

| # | 契约 | 位置 | 冻结什么 |
|---|---|---|---|
| 1 | `page_table` 寻址 | `engine/engine.py:446-458`、`scheduler/cache.py:625` | `[max_req+1, width]` int32、`[table_idx, pos] → 原始 KV 槽位`、**永远按 page_size=1 记账**、width 对齐 32 |
| 2 | KV 池布局 | `kvcache/mha_pool.py`、`hybrid_swa_pool.py` | `[num_pages*page_size, heads, dim]` 的 flat 视图 + 写入位置 `out_loc` |
| 3 | expert slot 缓存 | `moe/offload_cache.py:170-252` | 5 个 int32/int64 映射表的 shape/dtype/sentinel（`-1`） |
| 4 | bank 行布局 | `_BANK_SCHEMAS`(37-79)、`_BANK_BYTES_PER_EXPERT`(87-97) | 每种量化格式几块 bank、每行字节公式 |
| 5 | 量化行格式 | `kvcache/dsv4/v41_row_format.py`、`layers/quantization/*` | packed 行的位域（fp8_e8m0_b32 等） |
| 6 | msgpack 信封 | `message/utils.py` | `__type__` 标签、`__raw_dict__` 保护、张量 `dtype/buffer/shape` |
| 7 | 输入/写入索引元组 | `scheduler.py:947/957` | `Indice2D` 的语义、`-1` 汇列 |
| 8 | 采样输出 | `engine/engine.py:1059-1063` | token dtype(int32) + D2H 时机 + event |
| 9 | 配置兼容 | `engine/config.py`、`ServerArgs`、`ModelConfig` | 必须能读**同一份** `config.json`/`quantization_config`/`generation_config.json` |
| 10 | 模型张量命名 | `models/register.py` 的 `checkpoint_roots/segments/packed` | 否则权重加载对不上 |

**做法**：为每一项写一个 C++ 的 `static_assert`/`sizeof` 检查 + 一个从 Python dump 出来的 golden 二进制/JSON，在 CI 里比对。

---

## 6. 把 `tests/` 变成你的 conformance suite（最重要的加速器）

这仓库的测试哲学是"期望值必须来自独立参照"，这正好是重写需要的。现成可用的 oracle：

| oracle | 位置 | 能校准什么 |
|---|---|---|
| **按页 radix 参考模型** | `tests/kvcache/radix/model.py` + `adapters.py` + `driver.py` | 三套前缀缓存的 match/insert/evict/lock 语义（README 记录 17/17 变异被捕获） |
| **vendored 参考 transformer** | `tests/models/deepseek_v41/reference/` | 整个模型层数值（含 packed pools / Engram / fp4 容差） |
| **GDN / QSA 参考实现** | `models/qwen4_exp/gdn_reference.py`、`tests/models/qwen4_exp/{ple_hf_ref,qsa_*}.py` | 线性注意力与稀疏注意力 |
| **dequant / GEMM oracle** | `tests/kernels/*`、`tests/moe/test_dsfp4_reference_semantics.py` | 量化内核与 MoE 语义（含 `slot=-1` 不读） |
| **hybrid LRU CPU 镜像** | `moe/offload_kernels.py:128-188` | 补上 flashlib 缺失的算法缺口 |
| **纯算术预算** | `tests/engine/test_cache_budget.py` | 显存切分（无 GPU 可跑） |
| **协议/解析器矩阵** | `tests/server/*` | 三协议 wire 格式、parser 流式状态机 |

**差分测试架构（建议第一天就搭）**：
```
        ┌─────────── Python 参考实现（不动）───────────┐
输入 ──▶│ tokenize → scheduler → attention → sample     │──▶ 输出 A
        └───────────────────────────────────────────────┘
        ┌─────────── C++ 实现（影子）──────────────────┐
输入 ──▶│ 逐模块替换，每模块与 A 比对中间结果           │──▶ 输出 B
        └───────────────────────────────────────────────┘
比对层：① 张量逐元素（按 dtype 定 tolerance）② 消息逐字段 ③ 最终 token 序列逐 token
```
三个层级的比对粒度：
1. **数值级**：fp32 1e-5、bf16 相对 1e-2、fp8/fp4 用参考容差（`test_reference_parity.py` 里 `QUANT_TOL=2e-2` 可参考）；
2. **结构级**：page table、slot 映射、bank 内容逐字节；
3. **行为级**：同一批请求 → 同一 token 序列（贪心采样下可完全一致）。

---

## 7. 分阶段路线（每阶段都有验收标准）

| 阶段 | 内容 | 用到的模块 | 验收 |
|---|---|---|---|
| **P0** | 冻结 §5 的 10 项契约；搭差分框架；把 `csrc/` 编进你的工程 | `kernel/csrc`、`MODULE_DOCS/01/05/06` | C++ 能 dump 出与 Python 一致的 `page_table`/bank 布局 |
| **P1** | **最小闭环**：单 dense 模型 + 单卡 bf16 + 贪心解码 + `/v1/chat/completions` 非流式 | `layers`、`attention`(fi 或 triton 后端)、`kvcache/mha`、`scheduler` 基础 | 与 Python 版**逐 token 一致**（贪心） |
| **P2** | 流式 SSE + tokenizer + parser + 分块 prefill | `server`(generation/parsers)、`tokenizer` | 同 prompt 的 SSE chunk 序列一致 |
| **P3** | 前缀缓存（radix）+ 连续批处理 + CUDA graph | `kvcache/radix`、`scheduler`、`engine/graph` | radix 语义过 `tests/kvcache/radix` 等价用例；吞吐 ≥ Python 版 |
| **P4** | fp8 量化（先 fp8_block）+ flashinfer 后端 | `layers/quantization`、`kernel/triton` 的等价库 | logits 在容差内；tokens 一致 |
| **P5** | MoE offload（先 fused，再 offload）+ slot 缓存 | `moe/`、`layers/moe` | Miss 率相同、输出一致；`ft bench bw` 级别的带宽 |
| **P6** | 其余：hybrid/cpu MoE、多模态、更多模型族、多卡 TP | 其余 | 按需 |

**每阶段的硬规则**：**不允许"先写完再测"**。一个模块的 C++ 实现在替换进链路前，必须能独立通过它的 oracle。

---

## 8. C++ 架构建议（与 Python 结构的映射）

Python 侧的多进程 + ZMQ 是因为 **GIL**。C++ 不需要照搬：

| Python 结构 | C++ 建议 | 理由 |
|---|---|---|
| 前端进程 + tokenizer worker + scheduler 进程 | **单进程多线程**（HTTP 线程池 + tokenizer 线程池 + 1 调度线程 + N CUDA 工作线程） | 无 GIL；省掉 msgpack 序列化与 IPC 延迟 |
| 迁移期 | **保留 msgpack 线协议 + ZMQ 边界**，逐进程替换 | 唯一可行的渐进迁移路径 |
| `torch.cuda.Stream` × 2（scheduler/engine） | 显式 `cudaStream_t`，加一个 **stream 归属文档** | Python 靠 "current stream" 隐式传递，C++ 必须显式 |
| torch caching allocator | 启动期预分配 arena（按模型+池配置算好），或接 `cudaMallocAsync` | 图捕获期不能分配 |
| `@dataclass` 全仓状态 | POD `struct` + `static_assert(sizeof)` + arena 分配 | 便于差分与序列化 |
| `assert` 不变量 | `contract_assert`/`CHECK`，Release 也可开 | 这些不变量是正确性边界 |
| Python 异常 | `std::expected<T, Err>` / 错误码分层（对应 `ErrorReplyMsg.code`） | 协议层要能映射成 OpenAI error code |
| `Registry`（45 行） | 静态注册表（编译期或启动期数组） | 后端/缓存的可枚举性要保留 |
| `HfConfig`/`ModelConfig` | JSON 读取 + 同一套字段名 | **配置必须二进制兼容**，否则无法 A/B |

**目录建议**（镜像 Python 分层，便于对照阅读）：
```
src/
  core/        Req, Batch, Context, Tensor 类型
  engine/      调度循环、graph、预算
  kvcache/     池、radix
  moe/         slot cache, banks, executor
  layers/      linear/moe/norm/rope + quantization
  attention/   backend 抽象 + fi/fa/triton 等价
  models/      registry + 切面用到的族
  server/      http + 三协议 + parsers
  tokenizer/   bpe + chat template
  kernel/      cuda/host 内核（直接抄 csrc）
  wire/        msgpack 信封（迁移期）
```

---

## 9. C++ 重写特有的陷阱

1. **GIL 曾经替你做了串行化**：Python 里"看起来原子"的代码在 C++ 多线程下会 race。重读 `overlap_loop` 时把"哪些状态只被调度线程碰"列成表。
2. **torch 的隐式语义是规格的一部分**：stride/contiguous、broadcast、dtype 提升、in-place 别名、`view` vs `copy`。用 ATen 可以白拿；自研 Tensor 必须逐条实现或**在接口上禁掉**。
3. **CUDA graph 的约束不变**：捕获期禁同步、禁分配、禁 autotune。Python 用"硬编码 tile 常量"绕开的坑，C++ 一样要绕。
4. **数值差异**：bf16/fp8/fp4 的舍入、累加顺序、reduction 顺序都会变。**先定 tolerance 预算，再写内核**；贪心解码对 logits 误差敏感（可能翻 token）。
5. **生命周期**：Python GC → C++ 必须明确所有权。重点看带"延迟释放"语义的地方：`Req.aborted`（在飞时只打标记）、`lazy_free_region`、`CacheManager.rebuild` 新建前缀缓存。
6. **不要照着注释写**：已确认至少 4 处注释与实现不一致（unpinned D2H、hybrid 分式、CUDA 12.8 vs 13.0、PDL 默认关）。**以代码为准**，详见 [`INFERENCE_OPTIMIZATION_READING_GUIDE.md`](INFERENCE_OPTIMIZATION_READING_GUIDE.md) §5.1。
7. **`flashlib` 的 LRU 算法不在仓库**：先用仓库内 hybrid 变体 + CPU 镜像复现语义，再用差分测试确认逐位一致。
8. **chat template 是最大的非算法风险**：4 个族相关决策点 + jinja2 语义。建议第一版只支持 1 个族、把模板**预编译**成 C++，并保留"Python 渲染 → C++ 对比"的差分用例。
9. **别信"全量重写更快"**：33% 的代码（Triton）应该换库而不是重写，23% 的代码（模型族）应该裁剪。真正要你写的是 B+C+F 约 40%。

---

## 10. 四周阅读计划（含产出物）

| 周 | 读 | 产出 | 验收（能做什么） |
|---|---|---|---|
| **W1** | `MODULE_DOCS/00` → `01` §3 + `05` §3-4 → `06` §3 → `10` §4 → `09` §3 | `spec/00-layouts.md`（struct 表 + 不变量） | 不看代码写出 `Req`/`Batch`/`page_table`/slot cache 字段；解释 `cache_size >= 2E` |
| **W2** | `02` §5 → `01` §6/§9.3 → `06` §5 → `03` §6 | `spec/01-timing.md`（stream/event 顺序表） | 画出一次 decode step 的 GPU 操作序列与三道围栏作用 |
| **W3** | 切面模型的 `config/model/attention/weight.py` + `layers` 装配 + `kvcache` 的 `_tree_walk`/`allocate_paged` + `scheduler.PrefillAdder` | `spec/02-algorithms.md`（伪代码 + 复杂度） | 能手算 page 匹配长度、分块 prefill 的块边界 |
| **W4** | `tests/` 的 oracle 清单 + 选定的 Triton/量化内核契约 | `spec/03-conformance.md`（golden 提取与 tolerance） | 跑通"Python dump → C++ 比对"的一条管线 |

**"读透"的判定标准（可自测）**：
1. 给出任意一个 `--flag`，能说出它从 CLI 到最终生效的函数链（C++ 侧对应哪个配置字段）；
2. 能写出一次请求的完整状态机（含分块 prefill、abort、前缀命中三条分支）；
3. 能画出一次 decode step 的 stream/event 时序，并解释每个同步的必要性；
4. 能逐字节描述一个 KV page 与一个 expert slot 的内存布局；
5. 拿到任意一个跨模块的 bug 现象，能定位到"契约崩了"还是"实现错了"。

---

## 11. 下一步我可以提供的帮助

按你的进度，下面每一项我都可以直接产出：

- **契约固化**：为 §5 的 10 项分别产出 C++ `struct` 定义 + `static_assert` + 从 Python dump golden 的脚本；
- **竖切面选型**：帮你选第一个模型 + 量化格式 + 注意力后端，并列出它实际需要移植的文件清单（含行数）；
- **差分脚手架**：Python dump 中间张量 + C++ 读取并比对的完整骨架；
- **单模块精读**：例如把 `_tree_walk` / `overlap_loop` / `PrefillAdder` / hybrid 三阶段 kernel 逐行拆成可移植伪代码；
- **Triton → CUDA 对照**：挑一个内核（如 nvfp4_linear 的 M==1 GEMV），给出 C++/CUDA 等价实现与性能要点；
- **聊天模板方案**：评估 minja / 预编译 / sidecar 三条路，并给出第一版实现范围。
