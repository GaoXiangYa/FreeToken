# 00 · 架构总纲：模块划分、依赖关系与数据流

> 本文件是全套模块文档的**总入口**。它回答三个问题：系统由哪些模块组成、模块之间怎么关联、一条数据怎么流过它们。
> 具体到每个函数的行为，看 `01`–`17` 各篇；想知道某个模块里"到底有哪些符号"，看 `99-symbol-index/`。
> 行号与依赖数据基于 commit `377a3bc`。

---

## 1. 系统定位

FreeToken 是一个**面向消费级硬件的 MoE 推理引擎**：把 GPU / CPU / 主机内存当作一个弹性异构平台，用"专家 LRU 缓存 + 按实测带宽自适应的 CPU-GPU 协同计算"，在单张消费级显卡上服务 100B–300B 参数的 MoE 模型。

它的代码可以按"**一条请求经过什么**"切成四层：

```
┌─────────────────────────────────────────────────────────────────────────┐
│ L4 接入层    server/  daemon/  shell/  cli.py  control_cli.py  launch.py │
│              HTTP 三协议、进程编排、控制面、TUI、命令分派、agent 配置      │
├─────────────────────────────────────────────────────────────────────────┤
│ L3 调度层    scheduler/  message/  tokenizer/  llm/                      │
│              调度循环、分块 prefill、批管理、前缀缓存提交、跨进程消息       │
├─────────────────────────────────────────────────────────────────────────┤
│ L2 执行层    engine/  core.py  attention/  kvcache/  moe/                │
│              显存预算、CUDA graph、注意力后端、分页 KV、专家卸载           │
├─────────────────────────────────────────────────────────────────────────┤
│ L1 模型层    models/  layers/  layers/quantization/  mm/  checkpoint/    │
│              架构注册表、权重加载、层原语、量化方法、多模态、FTW 格式       │
├─────────────────────────────────────────────────────────────────────────┤
│ L0 内核层    kernel/（triton/ fla/ csrc/）  distributed/  utils/         │
│              算子实现、编译与缓存、通信、通用工具                          │
└─────────────────────────────────────────────────────────────────────────┘
```

**规模**：`python/freetoken/` 439 个 Python 文件 / 93,387 行 / 4,597 个类与函数符号（含嵌套函数），另有 25 个 C++/CUDA 文件（13,360 行）。

---

## 2. 模块清单与文档对照

| # | 模块 | 一句话职责 | 规模 | 详见 |
|---|---|---:|---:|---|
| — | `core.py` | 全引擎共享词汇表：`Req` / `Batch` / `Context` | 220 行 | `01` |
| 01 | `engine/` | 显存预算、配置收口、权重装载、CUDA graph、单步前向、运行时重建 | 2,501 | `01` |
| 02 | `scheduler/` | 调度循环、分块 prefill、`CacheManager`、状态上报、重建决策 | 2,491 | `02` |
| 03 | `server/` | HTTP 三协议适配、进程编排、统计记账、中立生成核心 | 11,312 | `03` `04` |
| 04 | `moe/` | 专家卸载缓存、LRU 准入、PCIe gather、双缓冲、CPU/混合执行、带宽剖面 | 5,870 | `06` |
| 05 | `kvcache/` | 分页 KV 池族、三套 radix 前缀缓存、线性状态池、DSV4 分层池 | 4,968 | `05` |
| 06 | `models/` | 架构注册表、`ModelConfig` 契约、权重加载与分片融合、19 个模型族 | 21,243 | `07` `08` |
| 07 | `attention/` | 9 个注意力后端、能力矩阵、prefill/decode 分流、metadata 构造 | 4,219 | `09` |
| 08 | `layers/` | 层原语（并行 Linear/MoE/RoPE/norm）+ 量化方言与方法选择 | 4,420 | `10` |
| 09 | `kernel/` | Triton/FLA 算子、C++/CUDA 扩展、JIT/AOT 编译与缓存 | 23,649 | `11` `12` |
| 10 | `checkpoint/` | HF safetensors → FTW 快速加载格式的转换与读取 | 1,165 | `13` |
| 11 | `message/` | 跨进程消息类型与手写 codec（**必须 torch-free**） | 421 | `14` |
| 12 | `tokenizer/` | tokenize / detokenize worker 的实现 | 1,218 | `14` |
| 13 | `llm/` | 离线 API：`LLM(Scheduler)`，绕过 ZMQ 的最小可跑入口 | 128 | `14` |
| 14 | `mm/` | 多模态：占位符扩展、媒体获取、编码缓存、族处理器 | 1,014 | `15` |
| 15 | `daemon/` | torch-free 监督者：单实例、spawn/signal/日志/记账 | 3,057 | `16` |
| 16 | `shell/` | 终端 TUI 客户端 | 1,515 | `16` |
| 17 | `utils/` `distributed/` `benchmark/` | Registry、ZMQ 队列、HF 配置缓存、TP 信息、测量工具 | 1,689 | `17` |
| 18 | 顶层 CLI | `cli.py` `control_cli.py` `launch.py` `gpu_select.py` `env.py` `cache_report.py` | 2,513 | `16` |
| 19 | `tests/` `benchmarks/` `scripts/` | 可执行规格、性能基准、发布管线 | 141 测试 | `17` |

---

## 3. 模块依赖关系

### 3.1 实测依赖矩阵（`ast` 解析全部 `import`，单位：import 次数）

**最强的前 14 条边**（读代码时最该先理解的耦合）：

| 从 | 到 | 次数 | 说明 |
|---|---|---:|---|
| `models` | `layers` | 109 | 模型族用层原语拼网络——**全仓最强耦合** |
| `models` | `utils` | 72 | HF 配置缓存、日志、TP 信息 |
| `models` | `kernel` | 55 | 模型族的专属算子直接调 Triton |
| `models` | `core` | 49 | decoder layer 读 `Batch` / `Context` |
| `layers` | `kernel` | 43 | 层原语封装算子 |
| `moe` | `kernel` | 30 | offload gather、CPU MoE 扩展 |
| `models` | `distributed` | 28 | TP 切分 |
| `attention` | `kernel` | 23 | 后端调 Triton 注意力 |
| `engine` | `moe` | 15 | 卸载缓存初始化与策略解析 |
| `attention` | `core` | 13 | 读 `Batch` 的 page_table/positions |
| `attention` | `models` | 12 | 读 `ModelConfig` 的头数/维度 |
| `models` | `attention` | 11 | 声明 AttnType、拿 backend |
| `layers` | `moe` | 11 | `make_moe_layer` 与 expert view |
| `scheduler` | `kvcache` | 10 | 页分配与前缀缓存 |

### 3.2 分层依赖（关键：依赖方向基本单向向上）

```
L4  server  daemon  shell  cli
      │  ↓                        ↓
L3  scheduler ──→ message/tokenizer        llm ──(继承)──→ scheduler
      │
      ↓
L2  engine ──→ moe, kvcache, attention, models, layers, kernel
      │
      ↓
L1  models ──→ layers ──→ kernel
      │            ↑
      ├──→ attention ─┘
      ├──→ mm
      └──→ checkpoint(FTW)
      │
      ↓
L0  core  utils  distributed  kernel
```

三条**架构性事实**（读代码时能省很多力气）：

1. **`core.py` 的出度为 0** —— 它只 import `torch` 与 `typing`，对 `freetoken` 内部零依赖（`TYPE_CHECKING` 的引用不算）。所以它是整个依赖图的叶子，任何人都可以依赖它，它不依赖任何人。这也是为什么它只有 220 行却定义了全引擎的词汇表。
2. **`message/` 是共享的线上类型层，`tokenizer/` 是它的 worker 实现** —— 类型被前端、tokenizer worker、scheduler 三方 import，所以单独拆出来。注意 `message/utils.py` **确实 import torch**（要序列化张量载荷），因此它不是 torch-free 层；真正被强制 torch-free 的是 `daemon/`（`tests/daemon/test_daemon_import_safety.py` 用子进程 meta-path 拦截器强制）。
3. **`models/` 是最大的枢纽**（对 `layers` 109 次、对 `kernel` 55 次、对 `core` 49 次）——它同时面向"架构描述""权重加载""前向执行"三件事，所以它又被拆成 `07`（共享机制）和 `08`（19 个族）两篇。

### 3.3 反向依赖（"谁在用我"）

| 被依赖方 | 主要使用方 | 接口形态 |
|---|---|---|
| `core.Context` | `attention` / `models` / `kvcache` / `moe` | 全局单例 `get_global_ctx()`，在 `engine.forward_batch` 里用 `ctx.forward_batch(batch)` 压栈 |
| `engine.Engine.forward_batch` | `scheduler` | 唯一调用方；`@torch.inference_mode()` |
| `kvcache.BasePrefixCache` | `scheduler/cache.py`（`CacheManager`） | match/insert/lock/unlock/evict |
| `attention.BaseAttnBackend` | `models/<family>/attention.py` | `ctx.attn_backend.forward(q,k,v,layer_id,batch,spec)` |
| `layers.make_moe_layer` | `models/<family>` | 按 strategy 返回 `MoELayer` 或 `OffloadMoELayer` |
| `layers/quantization` 的 method | `layers/linear.py`、`layers/moe.py` | `quant_method_for(prefix)` → method → kernel |
| `moe.OffloadMoeCache` | `layers/moe.py`（`OffloadMoELayer`） | ensure_experts / copy_missing / bank_views |
| `utils.Registry` | `attention/__init__.py`、`kvcache/__init__.py` | 全仓唯一的插件注册机制（45 行） |
| `utils.mp.*Queue` | `scheduler/io.py`、`tokenizer/server.py` | ZMQ 四件套 |

---

## 4. 进程拓扑（部署时的真实形态）

`ft serve` **不是一个进程**，而是 `mp.Process` spawn 出来的一族（`server/launch.py:153-222`）：

```
                     ┌──────────────────────────────┐
   HTTP 客户端 ─────▶│ uvicorn/FastAPI 主进程        │
                     │  server/api_server.py         │
                     │  FrontendManager 路由 + SSE   │
                     └──────────┬───────────────────┘
                                │ ZMQ ipc:///tmp/freetoken_3.pid=<pid>
                     ┌──────────▼───────────────────┐
                     │ Tokenizer worker × N         │
                     │  tokenizer/server.py         │  持 transformers，无 GPU
                     └──────────┬───────────────────┘
                                │ ZMQ ipc:///tmp/freetoken_0.pid=<pid>
                     ┌──────────▼───────────────────┐
                     │ Scheduler 进程 × TP rank      │
                     │  scheduler/scheduler.py       │  持 Engine + GPU + 全部池
                     │  engine/engine.py             │
                     └──────────┬───────────────────┘
                                │ ZMQ ipc:///tmp/freetoken_2（rank0 → 其它 rank 广播）
                     ┌──────────▼───────────────────┐
                     │ Detokenizer worker           │
                     └──────────────────────────────┘

  另有独立进程：ft daemon（http://127.0.0.1:1900，torch-free 监督者）
  它只通过 loopback HTTP + 信号 + pidfile 管 ft serve，与 ZMQ 无关
```

**读代码时的第一条纪律：先问"这段代码跑在哪个进程"。**

---

## 5. 三条关键数据流

### 5.1 一条流式请求（详细分解见 `03`）

```
HTTP → GenSpec(中立) → TokenizeMsg ──ZMQ──▶ Tokenizer worker
     → UserMsg ──ZMQ──▶ Scheduler: PendingReq → PrefillAdder
     → Batch → engine.forward_batch（CUDA graph replay + 采样）
     → DetokenizeMsg ──ZMQ──▶ Detokenizer → UserReply ──ZMQ──▶ Frontend
     → SSE chunk
```
中立腰是 `server/generation.py` 的 `GenSpec`/`GenEvent`：**三种协议在此汇合，wire 类型不进引擎**。

### 5.2 一个 decode step（详细分解见 `01` `06`）

```
token_pool gather → copy_from(D2D×5) → prepare_for_replay(D2D×3~4)
→ g.replay()  ← 一次 cudaGraphLaunch 覆盖 embed + 所有层 + LM head
→ 采样（图外）→ int32 → D2H → copy_done_event.record
→ token_pool scatter
```
其中每一层的 MoE（offload 策略）是 `ensure_experts`（LRU）→ `copy_missing`（PCIe gather）→ `_expert_gemm`，**三段在同一 stream 串行**。

### 5.3 权重加载（详细分解见 `07` `13`）

```
checkpoint 目录
  ├─ config.json → cached_load_hf_config → architectures[0]
  │                                   ↓
  │                        models/register.py:_MODEL_REGISTRY → ModelSpec
  │                                   ↓ parse_config
  │                             ModelConfig（含 attention group / quant）
  ├─ safetensors 或 FTW  ──→ models/weight.py:load_weight
  │        （FTW 分支）         ├─ per-model iter_weights（分片 + 融合 + 堆叠专家）
  │                             ├─ load_state_dict
  │                             └─ finalize_quant → resident experts → moe/expert_banks
  └─ 显存预算：engine/cache_budget.py 决定 MoE 槽位 vs KV 页的切分
```

---

## 6. 横切关注点（不属于任何单篇，但处处出现）

| 关注点 | 机制 | 位置 | 影响面 |
|---|---|---|---|
| **插件可穷举** | `Registry`（45 行）+ 显式 dict/Enum | `utils/registry.py`；`attention/__init__.py:38`、`kvcache/__init__.py:24`、`models/register.py:88` | 全仓"有哪些后端/模型/缓存"都能列全 |
| **隐藏开关** | `EnvClassSingleton`：类属性自动变成 `FREETOKEN_<NAME>` | `env.py`；散落的 `os.getenv("FREETOKEN_...")` 共 53+ 个 | 性能调优的主要入口 |
| **日志分级** | `init_logger` + `info_rank0`/`warning_rank0` | `utils/logger.py` | TP 下只有 rank0 打日志 |
| **TP 分布式** | `DistributedInfo(rank,size)` + `is_primary()` | `distributed/` | 权重切分、all-reduce、ZMQ 广播 |
| **torch-free 边界** | `message/`、`daemon/` 不得 import torch | 由 `tests/daemon/test_daemon_import_safety.py` 强制 | 决定模块怎么拆 |
| **CUDA graph 安全** | decode 路径禁止 `triton.autotune`、统计累加必须可重放 | 各 Triton 内核注释 | 决定内核为什么用死常量 |

---

## 7. 推荐阅读顺序

| 顺序 | 读什么 | 为什么 |
|---|---|---|
| 1 | 本文件 + `01` | 建立词汇表（`Req`/`Batch`/`Context`）与配置收口 |
| 2 | `03` → `02` | 先看请求从哪进来，再看它怎么被调度 |
| 3 | `05` → `06` | 两笔最大的显存/带宽账：KV 与专家 |
| 4 | `09` → `10` | 注意力与量化：一次前向的两大组成部分 |
| 5 | `07` → `08` | 模型怎么被描述、怎么被加载 |
| 6 | `11` → `12` | 算子层（当手册查，不要顺序读） |
| 7 | `13` `14` `15` `16` `17` | 按需：FTW / 跨进程 / 多模态 / 控制面 / 基础设施 |

配套的主题资料（在仓库根目录）：
- [ARCHITECTURE.md](../ARCHITECTURE.md) —— **八视图架构图集**（本文的文字版图解，图更多、更直观）
- [SOURCE_READING_GUIDE.md](../SOURCE_READING_GUIDE.md) —— 全局源码阅读路线图
- [INFERENCE_OPTIMIZATION_READING_GUIDE.md](../INFERENCE_OPTIMIZATION_READING_GUIDE.md) —— 推理性能优化专题

---

## 8. 文档使用说明

- `01`–`17`：**手写模块详解**。每篇结构统一：模块职责 → 文件清单 → 核心数据结构（字段表）→ 函数详解（签名/行号/作用/调用方）→ 关键流程逐段解析 → 与其他模块关系 → 常见坑 → 自测题。
- `99-symbol-index/`：**自动生成**的符号索引（`ast` 解析，不执行代码），逐文件列出**全部**类/函数/方法、行号、首个 docstring 行。用来确认"某模块里到底有什么"，防止遗漏。
- 所有行号基于 `377a3bc`。代码变动后用符号名 grep 兜底。
- 标注"（未验证）"的条目表示无法从静态代码确证，需要上机确认。
