# FreeToken 系统架构图集

> 一套从**分层 → 依赖 → 部署 → 内存 → 数据流 → 时序 → 加载 → 抽象**的八视图架构说明。
> 每个视图都给了 **图（PNG/SVG）** 和 **ASCII 精确版**（便于终端阅读、grep 与贴进文档）。
> 行号与数字基准 `377a3bc`；模块细节见 [`MODULE_DOCS/`](MODULE_DOCS/README.md)。

---

## 0. 图索引

| # | 视图 | 文件 | 回答什么问题 |
|---|---|---|---|
| 01 | 分层架构 | [`diagrams/01-layers.png`](diagrams/01-layers.png) · [`.svg`](diagrams/01-layers.svg) · [`.dot`](diagrams/01-layers.dot) | 系统由哪几层组成，每层有哪些模块、多大 |
| 02 | 模块依赖 | [`diagrams/02-module-deps.png`](diagrams/02-module-deps.png) · [`.svg`](diagrams/02-module-deps.svg) · [`.dot`](diagrams/02-module-deps.dot) | 谁依赖谁（边上是实测 `import` 次数） |
| 03 | 进程拓扑 | [`diagrams/03-process-topology.png`](diagrams/03-process-topology.png) · [`.svg`](diagrams/03-process-topology.svg) · [`.dot`](diagrams/03-process-topology.dot) | `ft serve` 起几个进程、怎么通信、端口/ZMQ 地址 |
| 04 | 内存布局 | [`diagrams/04-memory-layout.png`](diagrams/04-memory-layout.png) · [`.svg`](diagrams/04-memory-layout.svg) · [`.dot`](diagrams/04-memory-layout.dot) | GPU/主机/磁盘各放什么、怎么算预算 |
| 05 | 请求数据流 | [`diagrams/05-request-flow.png`](diagrams/05-request-flow.png) · [`.svg`](diagrams/05-request-flow.svg) · [`.dot`](diagrams/05-request-flow.dot) | 一次请求的 12 跳 |
| 06 | decode 步时序 | [`diagrams/06-decode-step.png`](diagrams/06-decode-step.png) · [`.svg`](diagrams/06-decode-step.svg) · [`.dot`](diagrams/06-decode-step.dot) | 一条 decode step 的两条 stream 与三道围栏 |
| 07 | 权重加载流水线 | [`diagrams/07-weight-load.png`](diagrams/07-weight-load.png) · [`.svg`](diagrams/07-weight-load.svg) · [`.dot`](diagrams/07-weight-load.dot) | checkpoint 怎么变成 GPU 上的权重 |
| 08 | 关键抽象关系 | [`diagrams/08-abstractions.png`](diagrams/08-abstractions.png) · [`.svg`](diagrams/08-abstractions.svg) · [`.dot`](diagrams/08-abstractions.dot) | 核心类型之间的拥有/使用关系 |

`.dot` 是 Graphviz 源文件，改标签后 `dot -Tpng -Gdpi=140 x.dot -o x.png` 即可重渲染（中文字体已设为 `Noto Serif CJK SC`）。

---

## 1. 视图 01 · 分层架构

**一句话**：五层，依赖方向**基本单向向上**；越往下越接近硬件。

```
┌──────────────────────────────────────────────────────────────────────────────┐
│ L4 接入层   server 11,312 · daemon 3,057 · shell 1,515 · CLI 2,513            │
│   HTTP 三协议(openai/anthropic/responses) · FrontendManager · parser 4,685    │
│   ft daemon(:1900, torch-free) · ft shell(TUI) · ft 六命令分派                │
├──────────────────────────────────────────────────────────────────────────────┤
│ L3 调度层   scheduler 2,491 · message 421 · tokenizer 1,218 · llm 128          │
│   Scheduler(run_forever/overlap_loop/PrefillAdder/CacheManager)               │
│   message(msgpack 线上类型) · tokenizer worker · llm.LLM(离线入口)             │
├──────────────────────────────────────────────────────────────────────────────┤
│ L2 执行层   engine 2,495 · core 220 · attention 4,219 · kvcache 4,968 · moe 5,870 │
│   Engine(配置收口·池·CUDA graph·forward_batch) · core(Req/Batch/Context)       │
│   attention(9 后端+能力矩阵) · kvcache(分页池+3 套 radix) · moe(slot 缓存)      │
├──────────────────────────────────────────────────────────────────────────────┤
│ L1 模型层   models 21,243 · layers 4,420 · mm 1,014 · checkpoint 1,165        │
│   models(注册表 29 项/19 族/ModelConfig 契约) · layers(并行 Linear/MoE/norm)    │
│   quantization(6 种 QuantKind/select_kernel) · mm · checkpoint(FTW)            │
├──────────────────────────────────────────────────────────────────────────────┤
│ L0 内核与基础设施   kernel 23,649 · utils 967 · distributed 147                │
│   kernel/triton 15k · kernel/fla 6k · kernel/csrc(C++/CUDA 5.8k)              │
│   utils(Registry/ZMQ/HF 缓存) · distributed(NCCL/pynccl)                      │
└──────────────────────────────────────────────────────────────────────────────┘
        ↑ 层间唯一"主干调用"：HTTP → ZMQ → Scheduler → forward_batch()
          → create_model/load_weight → 算子
```

**记住这一层的一句话职责**：
- **L0** 算子与基础设施（不含业务语义）；
- **L1** "这个模型长什么样、权重怎么来"；
- **L2** "一份显存、一条请求怎么被执行"；
- **L3** "多条请求怎么排队与交叠"；
- **L4** "外面怎么说话、进程怎么活"。

---

## 2. 视图 02 · 模块依赖（实测）

边上的数字是 `ast` 解析出的 **`import` 次数**（绝对量，反映耦合强度）。

```
                         ┌──────── server ────────┐  ┌─ daemon ─┐ ┌ CLI ┐ ┌ shell ┐
                         │  9→utils  7→tokenizer  │  │ 0 torch  │
                         │  4→kvcache 3→message   │  └──────────┘
                         │  2→scheduler(spawn)    │
                         └───────────┬────────────┘
                                     ▼
                    ┌─────────── scheduler ──────────┐
                    │ 10→kvcache  6→core  4→message  │
                    └───────────┬────────────────────┘
                                ▼
        ┌─────────────────── engine ───────────────────┐
        │ 15→moe  7→models  5→kvcache  4→kernel        │
        │  3→layers  2→attention  2→core               │
        └───────┬───────────────────────────┬──────────┘
                ▼                           ▼
   ┌──────────────────── models ─────────────────────┐
   │ 109→layers ★  72→utils  55→kernel  49→core      │
   │  28→distributed  12→attention  11→…  8→moe      │
   └───────┬──────────────────┬──────────────────────┘
           ▼                  ▼
   ┌──── layers ────┐   ┌── attention ──┐   ┌──── kvcache ────┐
   │ 43→kernel      │   │ 23→kernel     │   │ 14→utils        │
   │ 11→moe         │   │ 13→core       │   │ 10→kernel       │
   └───────┬────────┘   │ 12→models     │   └────────┬────────┘
           │            └───────┬───────┘            │
           └────────┬───────────┴────────────────────┘
                    ▼
            ┌── kernel ──┐   ┌── utils ──┐   ┌ distributed ┐
            │ 7→utils    │   │ 8→models   │   │ NCCL/pynccl │
            └────────────┘   └────────────┘   └─────────────┘
```

**三条从依赖图读出来的架构事实**：

1. **`core.py` 出度 = 0**（只 import torch/typing）。它是依赖图的**叶子**——所以敢只写 220 行却定义全引擎词汇表。
2. **`models → layers` 109 次是全仓最强耦合**。模型族的本质就是"用 L1 的层原语拼网络"。
3. **`daemon` 的 torch 引用 = 0**（连 `utils` 都不 import，因为后者的 `__init__` 会拉 transformers）。这不是风格，是"不可崩"的硬约束（`tests/daemon/test_daemon_import_safety.py` 强制）。

---

## 3. 视图 03 · 进程拓扑

```
                     ┌──────────────┐
   HTTP 客户端 ─────▶│ uvicorn 主进程 │◀─── HTTP :1900 ────┐
  (curl/SDK/桌面/     │ FastAPI        │                   │
   ft shell)          │ FrontendManager│            ┌──────┴──────┐
        │ :1919       │ SSE · 准入     │            │ ft daemon    │
        └────────────▶└──┬──────┬──────┘            │ :1900        │
                          │      ▲                   │ torch-free   │
             ZMQ ft3      │      │ ZMQ ft4           │ 单实例 flock │
             TokenizeMsg  │      │ UserReply         └──────┬───────┘
                          ▼      │                          │
                 ┌───────────────┴──┐              信号 + pidfile
                 │ Tokenizer worker×N│                      │
                 │ (transformers)    │                      ▼
                 └────────┬──────────┘               ┌──────────────┐
                    ZMQ ft0│ UserMsg                │  supervise   │
                          ▼                        │  spawn/kill  │
                 ┌───────────────────────┐         └──────┬───────┘
                 │ Scheduler × TP rank    │◀───────────────┘
                 │ Engine + GPU + 全部池   │   (daemon 只经
                 └────────┬───────────────┘    loopback HTTP 交互)
                    ZMQ ft1│ DetokenizeMsg
                          ▼
                 ┌──────────────────┐
                 │ Detokenizer worker│  ← 增量解码 + UTF-8/stop 回退
                 └──────────────────┘
                 ▲
                 │ ZMQ ft2：TP rank0 → 其它 rank 广播
```

| 资源 | 值 | 说明 |
|---|---|---|
| serve HTTP | `127.0.0.1:1919` | 推理流量**直连**，不经 daemon |
| daemon HTTP | `127.0.0.1:1900` | 控制面（`/engine/*`、`/accounting/*`） |
| ZMQ | `ipc:///tmp/freetoken_{0..4}.pid=<pid>` | ft0 backend / ft1 detokenizer / ft2 broadcast / ft3 frontend / ft4 tokenizer |
| 就绪信号 | 字符串 `"Scheduler is ready"` | `server/launch.py:106`；`expected_acks = num_tokenizers + 2` |
| daemon↔engine | loopback HTTP + 信号 + pidfile | **不走 ZMQ** |

---

## 4. 视图 04 · 内存布局与预算

```
GPU HBM   预算 = memory_ratio × 预载前空闲显存（默认 0.9）
  ① 常驻权重        fused: dense+全部专家 ／ offload: 仅 dense
  ② KV 池           [num_pages × page_size, heads, head_dim]
  ③ GDN 状态池      num_slots × slot_states（仅 hybrid）
  ④ MoE 专家槽缓存   cache_size × per_slot_bytes × bank 数（全层共享；≥2E）
  ⑤ page_table + token_pool   [max_req+1, width] int32 ×2（同形）
  ⑥ 图捕获静态缓冲   [max_bs] ×4 + logits [max_bs, vocab] fp32（≈96 MB）
  ⑦ 激活/workspace = (1 - memory_ratio) 余量 + graph mempool

主机内存
  ⑧ 专家 bank_sources   每层 [E,...]；exact-size cudaHostAlloc + pin-after-fill + mlock
  ⑨ row_store staging   磁盘表批行 + wait/signal flag
  ⑩ PLE / Engram 常驻表  PLE 47.7 GiB；Engram scale 6 GiB
  ⑪ CPU-MoE pinned IO   x/ids/w/y + 每(layer,bs) flag slot

磁盘（O_DIRECT / io_uring）
  safetensors / FTW 分片 · Engram 98 GB/层 分片
```

**预算公式（单一真相源，`engine/cache_budget.py:31`）**：
```
net = memory_ratio × baseline_free - weights_bytes - fixed_cache_size
    → plan_cache_budget(): 先预留 kv_reserve_pages，专家贪心吃满，KV 拿剩余
    → 不变量：moe_cache_size ≥ 2×num_experts（prefill 双缓冲）/ >2E（hit-D2D）
```
**跨层流动只有两条**：④↔⑧ 专家搬运（PCIe 实测 ~31 GB/s vs HBM ~3 TB/s）、⑤↔⑨ 磁盘表按需读。

---

## 5. 视图 05 · 一次请求的 12 跳

| # | 位置 | 动作 | 跨边界 |
|---:|---|---|---|
| ① | `openai_api.py:118` | `POST /v1/chat/completions` | — |
| ② | `generation.py:136` | 转成协议中立的 `GenSpec` | — |
| ③ | `generation.py:293` | 准入 → 发 `TokenizeMsg`，**立即建 SSE 流** | ZMQ ft3 |
| ④ | `tokenizer/server.py` | chat template → input_ids | — |
| ⑤ | `message/backend.py:73` | `UserMsg` | ZMQ ft0 |
| ⑥ | `scheduler.py:488` | `PendingReq` → `PrefillAdder` | — |
| ⑦ | `scheduler.py:786` | 组批、`allocate_paged`、page_table | — |
| ⑧ | `engine/engine.py:1044` | 图 replay + 采样 | — |
| ⑨ | `scheduler.py:312` | 排空：EOS/stop、`cache_req` 提交前缀 | — |
| ⑩ | `tokenizer/detokenize.py:93` | 增量解码 + 两道 holdback | ZMQ ft1 |
| ⑪ | `api_server.py:246` | `UserReply` 唤醒 waiter | ZMQ ft4 |
| ⑫ | `openai_api.py:624` | SSE chunk → `data: [DONE]` | — |

**注意**：⑦⑧⑨ 发生在 `overlap_loop` 内，CPU 与 GPU 交叠；prompt token 的计费**不在**④，而在首次 prefill 准入后由 `PromptAdmittedMsg` 补。

---

## 6. 视图 06 · 一条 decode step 的时序

```
self.stream（调度侧）                                  engine.stream（执行侧）
─────────────────────                                 ─────────────────────────────
① 收消息+准入（与 N 批并行）
② 组批 N：allocate_paged / positions
   / metadata / Sampler.prepare
        │
        └──────── fence A: self.stream.wait(engine) ──────▶ ③ token_pool gather → input_ids
                                                             ④ copy_from ≤5×D2D
                                                                prepare_for_replay 3~4×
                                                             ⑤ g.replay()  ← 一次 cudaGraphLaunch
                                                                （embed + 所有层 + LM head）
                                                             ⑥ 采样（图外）argmax / top-k/p
                                                             ⑦ int32 → D2H
                                                                copy_done_event.record
        ┌──── fence C: self.stream.wait(engine) ◀──────────────┘
③ 排空 N-1：copy_done.sync()
   EOS/stop · cache_req · 上报
        │
        └─ 下一轮（此时 GPU 已开始跑 N+1）
```

**三道围栏各自防什么**（这是最容易写错的地方）：
- **A**（调度流等执行流）：本次 H2D 的 token 拷贝必须排在上一批采样写回**之后**——否则复用刚释放的 `table_idx` 时，新 prompt 会被上一任残留覆盖。
- **B**（执行流等调度流）：引擎流必须等元数据 staging 完成。
- **C**（调度流等执行流）：排空时的页释放/重指向、SWA 哨兵 scatter 必须排在刚发射的前向**之后**，否则会改写仍在飞的图。
- **设计意图**：发射 N 后 CPU 去排空 N-1，与 GPU 上的 N 重叠；**前提是 D2H 非阻塞**（目标需 pinned —— 此处代码未 pin，见 [`INFERENCE_OPTIMIZATION_READING_GUIDE.md`](INFERENCE_OPTIMIZATION_READING_GUIDE.md) §1.1）。

---

## 7. 视图 07 · 权重加载流水线

```
config.json(architectures[0])
   ▼
models/register._MODEL_REGISTRY[arch] → ModelSpec
   ▼
parse_config(HF cfg) → ModelConfig（含 attention_groups / quant 标记）
   ▼
checkpoint_quant_config → set_quant_config（进程全局）
   ▼
Engine._load_weights → models/weight.load_weight
   ├─ FTW? is_ftw_checkpoint ──是──▶ iter_ftw_weights
   └─ 否 ──▶ 每族 iter_weights(name, tensor)
   ▼
loader.py：shard_tensor(TP) → iter_merged_tensors(qkv/gate_up) → iter_stacked_experts
   ▼
load_state_dict → finalize_quant
   ├─ fused/常驻 ──▶ load_expert_banks → GPU banks
   └─ offload ────▶ 专家留 pinned host banks（pin-after-fill）
   ▼
（并行）resolve_moe_cache_auto → plan_cache_budget 决定 MoE/KV 切分
```

---

## 8. 视图 08 · 八个关键抽象

```
接入/协议      ServerArgs(⊂SchedulerConfig⊂EngineConfig) · GenSpec · message/*
                     │ 构造                              │ submit/SSE
请求状态      Req ──组批──▶ Batch ──压栈──▶ Context(全局单例 get_global_ctx())
(core.py)            │                          │
                     └── SamplingParams          ├─ page_table / kv_cache / linear_state_pool
                                                 └─ attn_backend / moe_offload_cache
执行与内存    Engine ─┬─▶ GraphRunner(capture/replay)
                      ├─▶ Sampler
                      ├─▶ CacheManager ──▶ BasePrefixCache(3 套 radix)
                      │                   └─▶ BaseKVCachePool(7 个池族)
                      └─▶ OffloadMoeCache ──▶ ExpertBanks(BankSpec 由量化内核拥有)
模型与算子    ModelSpec ──parse_config──▶ ModelConfig ──▶ AttnType(9 值) ──▶ BaseAttnBackend(9 后端)
                                             └────────▶ QuantConfig ──scheme_for──▶ QuantMethod
                                                          ──select_kernel──▶ kernel
```

**读这张图的方法**：三个"腰"决定了一切扩展点——
`ModelConfig`（模型声明需求）→ `AttnType`/`KVCacheGroupSpec`（需求翻译）→ `BaseAttnBackend`/`BaseKVCachePool`（实现选择）；
`QuantConfig` → `QuantMethod.select_kernel` → kernel；
`Req/Batch/Context` 是贯穿所有层的共享词汇。

---

## 9. 数字速查

| 项 | 值 |
|---|---|
| Python | 439 文件 / 93,387 行 / 4,597 符号 |
| C/C++（`csrc/`） | 13,360 行（含 vendored gguf 7,550 → 自有 ≈5,810） |
| Triton + FLA | 20,887 行（55 + 17 文件） |
| 模型族 | 19 个目录 / 29 个注册架构 |
| 注意力 | 9 个 `AttnType` / 9 个后端 |
| 量化 | 6 种 `QuantKind` / 7 种专家 bank 布局 |
| KV 页大小 | 默认 1；DSV4=128；TRTLLM=16/32/64；SWA=1 |
| CUDA graph bs | `[1,2,4] + range(8,max+1,8)`，max 默认 256(>80GB) / 160，且 ≤ `max_running_requests` |
| PCIe gather（实测） | ~31 GB/s（HBM ~3 TB/s；C2C ~450 GB/s） |
| hybrid 判据 | CPU 带宽 > 2.0 × PCIe gather |
| 环境变量 | 53+ 个 `FREETOKEN_*` |

---

## 10. 与其他文档的关系

| 想看 | 去哪 |
|---|---|
| 每个模块的字段/函数级详解 | [`MODULE_DOCS/`](MODULE_DOCS/README.md)（18 篇 + 4,597 符号索引） |
| 源码阅读顺序与坑 | [`SOURCE_READING_GUIDE.md`](SOURCE_READING_GUIDE.md) |
| 性能优化专题 | [`INFERENCE_OPTIMIZATION_READING_GUIDE.md`](INFERENCE_OPTIMIZATION_READING_GUIDE.md) |
| C++ 重写路线 | [`CPP_REWRITE_GUIDE.md`](CPP_REWRITE_GUIDE.md) |
| 学习导向的重写课程 | [`LEARNING_BY_REWRITING.md`](LEARNING_BY_REWRITING.md) |
