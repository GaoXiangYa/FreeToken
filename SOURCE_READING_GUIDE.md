# FreeToken 源码阅读指南

> 行号锚点基于 `377a3bc`（HEAD）。代码变动后行号会漂移，请用符号名 grep 兜底。
> 本机无 NVIDIA GPU、未安装 torch，所以本指南面向**静态阅读**：不假设你能跑起来，而是假设你能读、能 grep。

---

## 0. 先定位：这个项目是什么

FreeToken 是一个**面向消费级硬件的 MoE 推理引擎**，不是训练框架，也不是通用库。

一句话概括它的技术主张：*把 GPU / CPU / 主机内存当成一个弹性异构平台，用"专家缓存 + 按带宽自适应的 CPU-GPU 协同计算"把 290B+ 的 MoE 模型塞进一张游戏显卡*。

规模：`python/freetoken/` 约 **93k 行 Python**，另有 vendored 的 CUDA/ggml C++ 内核（`kernel/csrc/`）、Triton 内核（30 个文件）和 flash-linear-attention 内核（17 个文件）。141 个测试文件。

**阅读前必须建立的预期**：这个仓库的复杂度不在算法，而在**三维正交组合**：
`模型族(19) × 注意力类型(9) × MoE 后端(4) × 量化格式(6) × 硬件策略`。
如果你试图顺序读，会被组合爆炸淹没。正确策略是：**先抓住 4 个"腰"（waist）抽象，再挑一条纵向主线走通，其余当插件读**。

---

## 1. 心智模型：一条请求的旅程

### 1.1 进程拓扑（这是理解一切的地基）

`ft serve` **不是一个进程**，而是 `mp.Process` spawn 出来的一族进程，用 ZMQ over IPC 串起来（`server/launch.py:153-222`）：

```
uvicorn/FastAPI 主进程（HTTP + FrontendManager 路由/SSE）
   │  ZMQ ipc:///tmp/freetoken_3.pid=<pid>   (frontend -> tokenizer)
   ▼
Tokenizer worker × N（tokenize/detokenize，持 transformers，无 GPU）
   │  ZMQ ipc:///tmp/freetoken_0.pid=<pid>   (tokenizer -> scheduler)
   ▼
Scheduler 进程 × TP rank（持 Engine + GPU + KV pool + MoE cache）
   │  ZMQ ipc:///tmp/freetoken_2 (TP rank0 -> 其它 rank 广播)
   ▲
Detokenizer worker（增量解码、UTF-8/stop 回退）
```

- 地址定义：`scheduler/config.py:26-35`（ft0/ft1/ft2）、`server/args.py:80-89`（ft3/ft4）。
- `--daemon` 是**另一个进程**（torch-free 监督者），跟 ZMQ 无关，只通过 loopback HTTP + 信号 + pidfile 管 `ft serve` 子进程。见 `daemon/README.md`（95 行，强烈建议整读）。
- 就绪信号：字符串 `"Scheduler is ready"`（`server/launch.py:106`），由 `server/supervisor.py:99 drain_ready` 收齐 `num_tokenizers + 2` 个 ack 后，`maintenance_state: loading -> serving`。

**关键推论**：任何一个变量都可能活在被另一个进程拥有的对象里。读代码时先问"这段代码跑在哪个进程"。调试/加日志时要意识到消息要跨 3 次进程边界。

### 1.2 数据流：一次流式 `/v1/chat/completions`

这是全仓最重要的一条主线，值得逐跳读一遍（括号内是 `file:line`）：

1. 路由 `openai_api.py:118` → `handle_chat_completion:150`
2. 协议 → 中立请求：`chat_request_to_genspec`（`openai_api.py:59`）产出 `GenSpec`（`generation.py:136`）
3. 准入 + 提交：`submit_generation`（`generation.py:293`）→ `FrontendManager.new_user()`（`api_server.py:234`，准入闸门）→ 推 `TokenizeMsg`（`message/tokenizer.py:69`）到 ZMQ
4. 同时立刻建流：`stream_chat_completion_chunks`（`openai_api.py:240`）+ `StreamingResponse`（`:208`）
5. Tokenizer worker：`tokenizer/server.py:127` → `_tokenize_requests:84` → `TokenizeManager.tokenize`（`tokenizer/tokenize.py:66`）→ `apply_chat_template:159` → 产出 `UserMsg`（`message/backend.py:73`）推给 scheduler
6. Scheduler 收消息：`run_forever`（`scheduler/scheduler.py:291`）→ `overlap_loop:207` → `receive_msg`（`scheduler/io.py:79`）→ `_process_one_msg:488` → `UserMsg` 分支 `:494` → `prefill_manager.add_one_req:545`
7. 组批：`_schedule_next_batch`（`:863`）先试 prefill 再试 decode；分块 prefill 的核心在 `PrefillAdder`（`scheduler/prefill.py:44`）与 `ChunkedReq`（`prefill.py:31`）
8. 准备批次：`_prepare_batch`（`scheduler.py:786`）→ `cache_manager.allocate_paged`（`scheduler/cache.py:265`）分配 KV 页、写 page table
9. 前向：`_forward:898` → `engine.forward_batch`（`engine/engine.py:1044`）
10. 排空：`_process_last_data:312` —— `copy_done.synchronize()` 后取 token、判 EOS/stop/length、生成 `DetokenizeMsg`、`cache_manager.cache_req(...)` 提交前缀缓存、`send_result`→ZMQ
11. 增量反解码：`tokenizer/detokenize.py:93`（含 UTF-8 与 stop-prefix 回退）
12. 前端：`FrontendManager.listen`（`api_server.py:246`）→ `ack_map`/`event_map` → `_sse`（`openai_api.py:624`）→ `data: [DONE]`

**两个反直觉点**（很容易困住人）：
- **prompt token 数不是 tokenize 时记账的**，而是等首次 prefill 被准入后由 `PromptAdmittedMsg` 补回来的（`scheduler.py:875`）。所以非流式老接口 `/v1/completions` 和 `generate_events` 路径的计费点不一样。
- `overlap_loop`（默认）意味着**你正在看的批次可能已经发射但还没排空**——CPU 侧调度藏在 GPU 计算背后。abort 只是给 in-flight 请求打 `req.aborted` 标记，真正释放要等批次排空（`scheduler.py:577-584`）。

### 1.3 四个"腰"抽象（the waists）

全仓的复杂度被收敛到 4 个接口。**先把这 4 个抽象读懂，剩下的都是它们的实现**：

| 腰 | 位置 | 作用 |
|---|---|---|
| `Req` / `Batch` / `Context` | `core.py:34 / 116 / 182`（220 行） | 全引擎共享词汇表。`Context` 通过 `get_global_ctx()` 全局可达，注意力/模型层都靠它拿 `page_table`、`attn_backend`、`kv_cache`、`moe_offload_cache` |
| `ModelSpec` + `ModelConfig` | `models/register.py:18,88` + `models/config.py` | 模型族的注册契约：架构名 → 模块/类/`parse_config`/`iter_weights` + 权重名映射 + 注意力组声明 |
| `BaseAttnBackend` + `AttnType` | `attention/base.py:57 / 13` | 注意力后端 ABC；模型声明"需要哪些 AttnType"，引擎据此自动选后端 |
| `BaseKVCachePool` + `BasePrefixCache` | `kvcache/base.py:46 / 217` | 分页池与前缀缓存的 ABC；`AttnType → 池族` 的派发在 `kvcache/__init__.py:27 resolve_pool_class` |

扩展机制只有一个：`freetoken.utils.Registry`（`utils/registry.py`，**45 行**）。全仓只有两处用它（注意力后端 `attention/__init__.py:38`、缓存管理器 `kvcache/__init__.py:24`）；模型族和解析器用的是显式 dict/Enum（`models/register.py:88 _MODEL_REGISTRY`、`function_call_parser.py:3544 ToolCallParserEnum`）。所以"这个项目有哪些插件"是可以穷举的——这是它比同类工程好读的地方。

### 1.4 五个 CLI 命令的调用链（精确跳转表）

`ft` 本体是个 130 行的纯分发表（`cli.py:104 main` → `COMMANDS` → `globals()[runner_name]`）。

| 命令 | 调用链 |
|---|---|
| `ft serve` | `cli.py:104` → `:27` → `server/launch.py:125 launch_server` → `server/args.py:118 parse_args` → `launch.py:140 resolve_gpu_uuids` → `launch.py:153 start_subprocess`（spawn N 个 scheduler + tokenizer/detokenizer）→ `api_server.py:930 run_api_server` → `supervisor.py:99 drain_ready` |
| `ft shell` | `cli.py:34` → `shell/__init__.py:51 main`；带 `--model` 时 `launch_server(run_shell=True)`，否则纯 HTTP TUI |
| `ft launch <agent>` | `cli.py:40` → `launch.py:1013 main` → `:1030 resolve_server_url` / `:1031 discover_server_model` → `PREPARERS[agent]`（`:818`）→ `:1005 run_command`。**它不启动引擎**，需要一个已在跑的服务 |
| `ft ctl` | `cli.py:52` → `control_cli.py:352 main` → 对 1919 发 HTTP（`/health`、`/v1/stats`、`/generate`、`/v1/cache/status\|rebuild`、`/v1/requests`） |
| `ft daemon` | 带动词 → `daemon/__init__.py:14` → `client.py:161`（打 1900）；不带动词 → `daemon/server.py:104` → `SingleInstance.acquire:124` → `ServeManager:143` → uvicorn |

配置解析的**唯一入口是 `server/args.py:118 parse_args`**（dtype 推断 `:894-912`、TP `:913`），产出 `ServerArgs ⊂ SchedulerConfig ⊂ EngineConfig`；随后 `Scheduler(config)`（`scheduler/scheduler.py:65`）→ `Engine(config)`（`:68`）→ **`engine/engine.py:1510 _adjust_config` 是最后的归一化收口**（dense 模型丢掉 MoE 参数、单流折叠、DSV4/4.1 特化在 `:1207/:1240`）。池大小是纯算术，在 `engine/cache_budget.py:48/:98`。

daemon 与引擎的关系值得单独记一句：daemon **spawn**（不是 fork）`ft serve` 并让它成为会话/进程组组长（`start_new_session=True`），自己退出时**detach**，引擎活得比 daemon 长；新 daemon 通过 `serve.json` + starttime/argv 身份重新认领。两者之间是 loopback HTTP + 信号 + pidfile，**没有 ZMQ**。端口：daemon 1900，serve 1919。

---

## 2. 仓库地图

```
python/freetoken/
├── cli.py  control_cli.py  cache_report.py         CLI 三个入口（thin）
├── launch.py (1072)                                 ft launch：给 claude/codex/... 写配置并启动
├── core.py (220)                                    ★ 共享词汇表 Req/Batch/Context
├── env.py (94)                                      FREETOKEN_* 环境变量单例
├── gpu_select.py  version.py
├── daemon/ (16 文件)                                torch-free 监督者：flock、spawn、日志 SSE、记账 outbox
├── server/ (20 文件)                                HTTP 层 + 协议适配 + 进程编排 + 解析器
├── message/ (4 文件)                                跨进程消息 dataclass + 手写 codec（必须 torch-free）
├── tokenizer/ (6 文件)                              tokenize/detokenize worker 的实现
├── scheduler/ (11 文件)                             ★ 调度循环、分块 prefill、CacheManager、状态上报
├── engine/ (6 文件)                                 ★ Engine、EngineConfig、显存预算、CUDA graph
├── kvcache/ (13 + dsv4/)                            ★ 分页池 + 3 个 radix 前缀缓存
├── attention/ (17 文件)                             9 个注意力后端（FI/FA/TRTLLM/Triton/DSA/DSV4/...）
├── layers/ (10 + quantization/)                     融合算子 + 量化方法（线性和 MoE 两套）
├── moe/ (16 文件)                                   ★ 专家卸载缓存、CPU 执行器、各格式专家内核
├── models/ (19 个模型族包 + gguf/)                   ★ 注册表 + 每族 loader/forward/vision
├── kernel/ (csrc/ + triton/ + fla/)                 C++/CUDA/Triton 内核与 JIT 缓存
├── checkpoint/ (ftw.py 742)                         HF → FTW 快速加载格式
├── mm/ (processor/media/encoder_cache)              多模态：图像 → 嵌入
├── llm/llm.py (125)                                 ★ 离线 API：LLM(Scheduler)，最小可跑入口
├── platform: distributed/ shell/ benchmark/ utils/
```

**读 / 略读的粗判据**：
- **必读（spine）**：`core.py`、`engine/config.py`、`engine/cache_budget.py`、`models/register.py`、`kvcache/base.py`、`attention/base.py`、`message/*.py`、`llm/llm.py`。
- **主线必读（长但值）**：`scheduler/scheduler.py`(962)、`scheduler/prefill.py`、`scheduler/cache.py`(650)、`server/generation.py`、`engine/engine.py`（只读你关心的那几个方法）。
- **按需精读**：`moe/offload_cache.py`(1070)、`kvcache/{radix,swa_radix,hybrid_radix}_cache.py`、`models/<你要研究的族>/`。
- **基本可略**：`launch.py` 约 950/1072 行是六个 coding agent 的配置排列组合；`function_call_parser.py` 3756 行里 3000+ 行是各家模型的标记解析器；`moe/benchbw.py`(999) 只在你调 MoE 策略时读。

---

## 3. 五条阅读路线（按你的目标选一条）

### 路线 A：端到端 serving 主线（**默认推荐**，2-3 天）

目标：能画出上面那张 12 步数据流图，并说出每一步在哪个文件。

```
core.py:34-220                      ← 先花 20 分钟，这是全仓的词汇表
llm/llm.py:30-125                   ← 离线版 Scheduler：125 行看懂调度器接口，且**绕开 ZMQ**
server/generation.py:136,293,538    ← GenSpec/GenEvent：三种协议的中立腰
server/openai_api.py:118-208        ← 路由 → spec → submit → SSE
message/tokenizer.py + backend.py + frontend.py   ← 三种进程间消息
tokenizer/server.py:127-263         ← 桥接这三种消息的 worker
scheduler/scheduler.py:291-310      ← 循环骨架
scheduler/scheduler.py:207-261      ← overlap 调度（为什么"发射了但没排空"）
scheduler/scheduler.py:488-545      ← 消息 → PendingReq 准入
scheduler/prefill.py:44-348         ← 分块 prefill 的核心
scheduler/scheduler.py:786-906      ← prepare → forward
scheduler/scheduler.py:312-429      ← 排空 → DetokenizeMsg
server/generation.py:591-783        ← 语义事件 → 各协议 wire shape
```

**捷径**：如果你只想"让引擎跑起来看一眼"，`python/freetoken/llm/llm.py` 是最小入口——`LLM(model_path).generate(prompts, SamplingParams(...))`。它继承 `Scheduler` 但 `offline_mode=True`，把 ZMQ 收发换成两个函数覆盖（`offline_receive_msg`/`offline_send_result`）。这是全仓**信噪比最高的一个文件**。

### 路线 B：MoE offload（项目卖点，2-3 天）

先读 `docs/models.md:39-76`（四种策略的语义 + unified-memory / NVLink-C2C 特例），再进代码。

```
moe/__init__.py:9                       ← 策略词表 fused/offload/cpu/hybrid
layers/moe.py:261-296                   ← ★ _decode_routed：最短的完整闭环
moe/offload_cache.py:104-292            ← 核心数据结构（见下）
moe/offload_kernels.py:19-40            ← LRU 委托给 flashlib.slot_cache
moe/offload_cache.py:1014-1057          ← 未命中搬运 fast_index_copy_*
moe/offload_cache.py:609-636,671-844    ← 双缓冲 prefill 流式
engine/engine.py:638-764,766-795        ← 缓存初始化 + hybrid fetch 比例解析
layers/moe.py:408-440                   ← _expert_gemm：格式/内核派发点
moe/cpu_executor.py:550-627             ← CPU 侧执行
```

核心数据结构 `OffloadMoeCache`（`offload_cache.py:104`）三层：
- **主机端 pinned banks**：`bank_sources[bank] = list[L 个 [E,...] 张量]`（`:211`）
- **GPU 端 slot cache**：每 bank 一个 `[cache_size, ...]`（`:212`），**所有层共享一个全局池**（所以叫 "global LRU"）
- **磁盘**：无（只在加载期读 checkpoint）

专家身份是扁平整数：`id = layer_id * num_experts + expert`（`:176-178`）；`slot_for_id` / `id_of_slot` 双向映射，`usage` + `step` 就是 LRU 时钟。

**"q\*" 只存在于 README**（`README.md:20`），代码里没有 `q_star` 标识符。它的实现是 **hybrid fetch 比例**：`Engine._resolve_hybrid_fetch`（`engine.py:766-795`）读 `ft bench bw` 写下的 profile（按"专家格式 + GPU UUID"索引），算出 fraction；每个 decode step 在 GPU 上解一个整数规划，让 PCIe 抓取和 CPU 溢出计算**同时结束**（`offload_kernels.py:345-354`，Q16 定点）。这就是论文里的"带宽自适应协同计算"。

**双缓冲的对象是权重（专家 bank 行），不是激活**：slot cache 的头 `2*num_experts` 个 slot 被 reshape 成 `[2, E, ...]` 两个缓冲区（`offload_cache.py:609-636`），`buffer_id = layer_id % 2`。

### 路线 C：显存 / KV / 前缀缓存（2 天）

```
engine/cache_budget.py:31-95        ← 纯整数运算的显存预算策略（无 GPU 可读、可单测）
engine/config.py:73-98              ← page_size / memory_ratio / swa_full_tokens_ratio
kvcache/base.py:46-160              ← 池契约与 sizing 接口
kvcache/radix_cache.py:17-106,236-262   ← ★ 树、key_fn、split_at、_tree_walk
scheduler/cache.py:265-284,591-650  ← 页空闲链表、page table 写入、寻址
kvcache/hybrid_radix_cache.py:76-192    ← 第二币种（GDN 状态快照）
kvcache/swa_radix_cache.py:88-135,316-365  ← 墓碑与滑动窗口复用
kvcache/dsv4/window_tier.py:26-72,138-297  ← DSV4 的虚拟寻址
engine/engine.py:411-427,897-1042   ← 启动求解 + 运行时 rebuild
```

要点：
- **`page_size` 默认 1**，但 DSV4 强制 128；页是整个缓存的复用单位，所以 radix 的 key 是**按页**的 `tuple(ids[:(p+1)*page_size])`，不是按 token 的 trie。
- 三个 radix 实现是**兄弟类**：共享 `RadixTreeNode` + `split_at`，各自带一个"第二币种"（无 / 滑窗 / GDN 状态快照）。测试侧的参考模型 README（`tests/kvcache/radix/README.md`）是理解它们最好的文档，包括一个 `>=` 边界 bug 的精彩取证。
- "语义锚点"（README 说的 semantic anchor）在代码里**没有这个名字**。最接近的是 `Req.toolcall_anchor_len`（`core.py:62-64`）：模型吐出 tool-call 开头 token 时打标记，把 GDN 状态冻结进 ping-pong slot（`scheduler/cache.py:153-178`），并限制 SWA 驱逐不要越过它——这正是"tool call 重写后不用重算上下文"的实现。
- "弹性显存重分配"是**真的，但只在 idle 时、且会清空前缀缓存**：`POST /v1/cache/rebuild` → `scheduler.py:160-205`（忙就回 `busy`，不排队）→ `engine/engine.py:897 rebuild_runtime_cache`（两阶段：先校验预算再释放，最后重捕 CUDA graph）。不要理解成"负载中动态伸缩"。

### 路线 D：模型接入 / 加一个新模型族（1 天）

```
models/register.py:18,88,323,348    ← ModelSpec 契约 + _MODEL_REGISTRY
models/llama/ (5 文件, 290 行)       ← ★ 最规范的样板：dense、无 MoE、无多模态、无量化
  config.py:8 parse_config → ModelConfig
  model.py:18,71   decoder layer / LlamaModel / LlamaForCausalLM
  attention.py:17,70  → ctx.attn_backend.forward
  weight.py:23 iter_weights（分片 + 融合）
models/loader.py:25,128,275         ← shard_tensor / iter_merged_tensors / iter_stacked_experts
models/weight.py:209                ← load_weight 入口（含 FTW 分支 :226）
models/config.py                    ← AttentionGroupConfig / KVCacheGroupSpec（模型声明"我需要哪些 AttnType"）
engine/engine.py:530-562            ← _load_weights → state_dict → finalize_quant → 常驻专家
```

`qwen3/` 基本就是 `llama/` 加一个 `has_qk_norm=True`；`mistral/`、`qwen2/` 同理。**先读 llama，再按需看复杂度阶梯**：`gpt_oss`（MXFP4）→ `glm4_moe`（MoE）→ `deepseek_v4/v41`（DSV4 + 稀疏注意力 + Engram 表）→ `qwen4_exp`（GDN + QSA + 超连接 + PLE 表）。

FTW（`checkpoint/ftw.py`）：把 safetensors 的"未对齐 + 海量小 I/O"问题解决成**一段连续字节区域切成 ≤8 GiB 分片 + 一个 JSON 索引**，这样能用 O_DIRECT 并行读。`ft checkpoint` 的转换是**模型无关的**（`convert.py:165` 复用 `load_weight` + `load_expert_banks`），所以任何一个已注册的 safetensors 架构都隐式支持 FTW。

### 路线 E：内核层（当参考手册用，不要顺序读）

规模：`kernel/triton/` 实际 **55 个文件**（30 个顶层 + `dsv4/` 13 + `dsv41/` 6 + `qsa/` 6），约 15k 行；`kernel/fla/` 17 文件 5968 行；`kernel/csrc/` 25 文件 13360 行。注意 `kernel/triton/` **没有 `__init__.py`**（namespace package）。

**编译与缓存（不是 torch，是 tvm-ffi）**：除 gguf 之外全部走 tvm-ffi。
- `load_jit`（`kernel/utils.py:352`）内联 `csrc/jit/*` 源码编译；`load_aot`（`:292`）编译 `csrc/src/*`。
- 查找顺序 `_load_prebuilt`（`:252`）→ `_kernel_cache_dir`（`:210`）：`FREETOKEN_DISABLE_KERNEL_CACHE` → `FREETOKEN_KERNEL_CACHE_DIR` 覆盖 → `import freetoken_kernel_cache; get_jit_cache_dir()` → `<cache>/<name>/<name>.so` → `tvm_ffi.load_module`。miss 才 JIT；`FREETOKEN_DISABLE_JIT=1` 时 miss 直接硬报错（发布验证用）。
- **架构安全**：`_cuda_cflags`（`:81-91`）会补 `-gencode ... code=compute_XX`（最高架构的 PTX），这个字符串也会进入 tvm-ffi 的构建 hash，所以不同架构不会共用同一个 `.so`。nvcc 与 torch 的 CUDA major 不一致会在 `kernel/_toolchain.py:51-78` 直接报错（`FREETOKEN_ALLOW_CUDA_MISMATCH` 可越过）。
- 预编译 wheel 的内容 = `kernel/aot.py:146 default_kernel_specs()`（store / index / fast_index_copy / batch_memcpy / radix），按 `8.0 8.6 8.9 9.0 10.0 12.0` 构建。
- Triton 自己管磁盘缓存；`kernel/triton/autotune_cache.py:24-35` 加了 `cache_results=True` 让 autotune 结果持久化。`kernel/triton/e4m3_compat.py:42-44` 是唯一改 `TRITON_CACHE_DIR` 的地方（防止强制 e4m3 仿真污染正常缓存）。

**`setup.py` 构建三个 torch CppExtension**：
| 名字 | 源 | 作用 | 关键度 |
|---|---|---|---|
| `_pinned_tensor` | `csrc/pinned_tensor.cpp` (128) | 精确尺寸的 `cudaHostAlloc` 钉住内存（避开 2 的幂分配器膨胀） | offload 性能关键 |
| `_cpu_moe` | `csrc/cpu_moe/cpu_moe_ext.cpp` (2211) | CPU MoE decode；`cudaLaunchHostFunc` 图节点；bf16 GEMV 带 `target("avx512bf16")` + 运行时 `__builtin_cpu_supports` 分派 | CPU/hybrid 性能关键 |
| `_row_store` | `csrc/row_store/row_store_ext.cpp` (677，Linux only) | PLE/Engram 表的 O_DIRECT+pread / io_uring 行读取 | Qwen4Exp PLE 性能关键 |

运行时 JIT 的 `csrc/jit/` 里，**`fast_index_copy.cuh`（597 行）是全仓最性能关键的单个内核**——offload 模式下从 pinned 主机 bank 直接抓专家就是它。`csrc/src/` 里 `radix.cpp` 只是 CPU 侧 key 比较，`tensor.cpp` 是测试用（**可忽略**），`pynccl.cu` 是 TP 通信子。

**回退规则**（`pyproject.toml:78-91`）：`fi`=flashinfer、`sgl`=sglang-kernel（import 名 `sgl_kernel`）、`accel`=两者。探测是无副作用的 `find_spec`（`kernel/backend.py:26-38`），调用点就近分支（`layers/activation.py:10-16`、`layers/norm.py:11-16`、`layers/rotary.py:33-38`、`kernel/causal_conv1d.py:28-37`、`moe/fused.py:87-89`）。
注意力后端的自动选择在 `engine/engine.py:124 _resolve_auto_attention_backend`：按"架构是否满足 + 包是否装了 + 该后端的逗号分段是否覆盖所有必需 AttnType"取第一个。**FULL-only 模型的顺序是 `trtllm`(sm_100) → `fa,fi`(sm_90) → `fi` → `triton`；SWA-only 直接落到 `triton`**（`:140-148`）。所以纯 Triton 回退产出的就是注册名 `"triton"` 的注意力后端（`attention/__init__.py:82-92`，`attention/triton.py:142` → `kernel/triton/attention.py`）。模型专属的 `dsv4_sparse`/`dsv41_sparse`/`dsa`/`m3_sparse`/`qsa_sparse` 本身只有 Triton 实现，按 AttnType 优先于 FULL 候选。

**`fla/` 是 vendored 而非依赖**（`pyproject.toml` 里没有 fla）。来源在 `fla/__init__.py:1-38` 注明：GatedDeltaNet 路径来自 sglang 的 fla fork，KDA 来自 vLLM 的 `third_party/flash_linear_attention`。谁在用：
- `qwen3_5_moe` / `qwen4_exp` 的线性注意力层 → `chunk_gated_delta_rule`（`fla/chunk.py:128`，prefill）+ `fused_sigmoid_gating_delta_rule_update`（`fla/fused_sigmoid_gating_recurrent.py:247`，decode）
- `glm5_next` 的 **KDA**（Kimi Delta Attention）→ `chunk_kda_with_fused_gate`（`fla/kda.py:1215`）、`fused_recurrent_kda`（`:145`）
- 三者共用的 gated RMSNorm → `rms_norm_gated`（`fla/layernorm_gated.py:223`）

**`fla/__init__.py` 里有两个明写的雷**（读之前先看）：`chunk_kda_with_fused_gate` 会**覆写它的 `v` 实参**；`chunk_delta_h.py` 必须保留单一固定的 `triton.Config`——多配置 autotune 会破坏 in-place 的 state pool。

**`csrc/gguf/` 是唯一不走 tvm-ffi 的路径**：从 sgl-kernel vendored 进来，而后者是 llama.cpp 的移植（`gguf_kernel.cu:1-2` 标注了 vLLM commit `755ed7b`）。它由 torch `cpp_extension.load` 编译（`kernel/gguf.py:50-73`，模块名 `freetoken_gguf_kernels`）。**唯一消费者是 Gemma4 GGUF**（`models/gguf/config.py:22-24`）。算子按 batch size / 量化类型在 `layers/gguf.py:44-65` 分派：≤6 token 走 MMVQ，否则 MMQ，再否则反量化 + bf16 GEMM；Q4_0 专家 bank 走 `moe/fused_q4_0.py:30-46`。权重保持打包的 `[out, row_bytes]`，反量化在核内做。

**读 Triton 内核的高效姿势（以 `kernel/triton/norm.py` 为例，192 行）**：**自底向上读**——先读公开的 Python 包装（`:154-161`），再读 launcher（`:123-151`：校验形状、`None`→`empty_like`、算 stride、选 `num_warps`/`num_stages`/PDL），最后读 `@triton.jit` 主体。四步映射到 PyTorch 直觉：
1. `:42-44` `@triton.heuristics(_HEUR)` 叠在 `@triton.jit` 上；`jit` = "把一段你本来用 torch 写的逐元素循环，改成对单个 tile 写一遍"。
2. `:53-56` `tl.program_id(0/1)` 就是 `blockIdx`；`tl.arange(0, BLOCK)` + `mask = cols < H` 就是"一个向量 + 它的边界掩码"。
3. `:46-51` 签名里 `eps/H/BLOCK` 是 `tl.constexpr`（编译期常量 → 编译器特化、干掉掩码）；`sxa, sxb, ...` 是运行期 stride。
4. `:65-66` `tl.load(X + xoff + cols, mask=mask, other=0.0)` = 指针算术 + 掩码（就是带 stride 的张量读）；`:73-75` `tl.sum(x*x, axis=0)/H` 是规约，`tl.store(..., mask=mask)` 是写回，输出 dtype 来自 `OUT.dtype.element_ty`。
5. grid 在 Python 侧：`:145` 的 `_rmsnorm_kernel[(A, B)]` = 每 (row, head) 一个 program。

**新手可以完全忽略**：`csrc/include/freetoken/*`（tvm-ffi/ROCm 垫片）、`csrc/src/tensor.cpp`（测试算子）、`kernel/aot_models.py`（形状表）、`kernel/__main__.py`（生成 `.clangd`）、整个 `triton/dsv4|dsv41|qsa/` 以及 `triton/df11*.py`、`triton/minimax_m3_sparse.py`（都是单 checkpoint 专用）、`fla/` 的 `__init__.py` 之外的内部实现（vendored，自带"脏"的说明）。

---

## 4. 必读清单（按优先级）

### T0 —— 不读这些，后面都白读（约 900 行）
| 文件 | 行号 | 为什么 |
|---|---|---|
| `core.py` | 34-220 | `Req`/`Batch`/`Context`：全仓命名与状态的家底 |
| `engine/config.py` | 21-158 | `EngineConfig`：所有 flag 的最终归属；`ServerArgs ⊂ SchedulerConfig ⊂ EngineConfig` |
| `engine/cache_budget.py` | 31-95 | 显存怎么分：纯算术，`net_cache_budget_bytes` 自称 single source of truth |
| `models/register.py` | 18-120 | `ModelSpec` 契约 + 注册表（354 行，全读也快） |
| `kvcache/base.py` | 46-160 | 池 ABC + sizing 接口 |
| `attention/base.py` | 13-100 | `AttnType` 分类学 + 后端 ABC + `HybridBackend` |
| `message/{tokenizer,backend,frontend}.py` | 全部 | 跨进程消息定义（4 个文件都短） |
| `llm/llm.py` | 30-125 | 最小可跑的 Scheduler |
| `daemon/README.md` | 1-95 | 一页讲清控制面设计（为什么必须 torch-free） |

### T1 —— 主线精读（约 3000 行，按需）
`scheduler/scheduler.py`（循环 + `_process_one_msg` + `_prepare_batch` + `_process_last_data`）、`scheduler/prefill.py:44-348`、`scheduler/cache.py`、`server/generation.py`、`engine/engine.py`（选读：`__init__`、`_load_weights`、`forward_batch`、`rebuild_runtime_cache`）。

### T2 —— 深挖时才读
`moe/offload_cache.py`、`kvcache/{radix,swa_radix,hybrid_radix}_cache.py`、`kvcache/dsv4/`、`models/<目标族>/`、`server/function_call_parser.py`（只读 `BaseFormatDetector` 契约 + 一个具体 detector）。

**解析器的高效读法**：`function_call_parser.py:242-660` 是 `BaseFormatDetector` 契约（`bot_token`/`eot_token`/`detect_and_parse`/`parse_streaming_increment`/`supports_streaming`…）；然后挑一个最小的实现读完——`Qwen25Detector:913-1005` 最干净；最后读 `FunctionCallParser:3535-3724` 看服务层怎么驱动它。**不要从头读到尾**。

---

## 5. 把测试当规格说明书（本项目最大的红利）

这个仓库的测试写得像一份**独立参考实现**：`tests/README.md` 明确要求"期望值必须来自别的东西（PyTorch 参考、往返、CPU 镜像、活注册表），而不是复述实现"。所以你**用测试读代码，比用代码猜测试更划算**。

### 最值得先读的 5 个测试
| # | 文件 | 它钉住了什么 |
|---|---|---|
| 1 | `tests/kvcache/radix/test_plain_radix.py`（+ `radix/model.py`、`adapters.py`、`driver.py`） | 前缀缓存语义。**全仓最佳测试资产**：按页建键的独立参考模型 + 每步操作后跑不变量电池，注入 17 个变异全部捕获（`radix/README.md:47`）。读完你就懂了三套 radix 缓存的完整契约，包括那个 `>=` 边界 bug 的取证过程 |
| 2 | `tests/scheduler/test_scheduler_chunked_prefill.py` | 分块 prefill 的页会计与 overlap 顺序。**纯 CPU 可跑**，docstring 直接写了它复现的真实 bug（陈旧 `cache_handle` → 缓存完整性检查失败） |
| 3 | `tests/engine/test_cache_budget.py` | 专家缓存 vs KV 的显存切分，全是数字，**纯 CPU 可读** |
| 4 | `tests/moe/test_dsfp4_reference_semantics.py` | W4A8 专家契约：fp8 激活往返、swiglu 限幅、down epilogue 里的路由权重；用独立转写 + CPU/GPU 交叉验证 |
| 5 | `tests/daemon/test_daemon_import_safety.py` | 用子进程 meta-path 拦截器验证"daemon 不许 import torch"这条架构边界——**同进程检查会假通过，所以它必须这么做**。这是"测试如何保护架构"的范例 |

### 生产模块 → 可执行规格 对照（节选）
| 生产代码 | 最该读的测试 |
|---|---|
| `kvcache/radix_cache.py` / `swa_radix_cache.py` / `hybrid_radix_cache.py` | `tests/kvcache/radix/test_{plain,swa,hybrid}_radix.py` |
| `scheduler/{cache,prefill,decode}.py` | `tests/scheduler/test_scheduler_chunked_prefill.py`、`test_cost_accounting_core.py`、`test_swa_commit_lock_and_floor.py` |
| `engine/cache_budget.py` | `tests/engine/test_cache_budget.py` |
| `models/register.py` | `tests/models/test_models_registry.py`（遍历活注册表，新增条目自动覆盖） |
| `models/<family>/` 的数值正确性 | `tests/models/deepseek_v41/test_reference_parity.py`（对 vendored 参考 transformer 比 logits） |
| `moe/cpu_executor.py` | `tests/moe/test_dsfp4_reference_semantics.py`、`test_cpu_moe*.py` |
| `server/*_api.py` / `generation.py` | `tests/server/test_streaming_model_matrix.py`、`test_{openai,responses,anthropic}_api.py` |
| `server/args.py` 的 parser auto 推断 | `tests/server/test_parser_auto_selection.py` |
| `layers/rotary.py` | `tests/kernels/test_rotary.py`（按定义写的 YaRN 参考） |
| `checkpoint/ftw.py` | `tests/checkpoint/test_ftw_weights.py` |
| `attention/*` | `tests/attention/`、`tests/dsv4/`、`tests/kernels/test_triton_attention.py` |

### 标记与环境变量
- `slow` / `needs_weights` 只声明在 `pyproject.toml:122-124`；**`needs_weights` 本身不 skip 任何东西**，真正的跳过来自旁边的 `@needs_checkpoint` / `os.environ` skipif。
- `tests/conftest.py` 只有 10 行（一个 autouse 的 `QuantBackend` 重置）；`tests/kvcache/radix/conftest.py` 提供**确定性时钟**（LRU 断言必需）。
- 无 GPU 也能读/跑的：radix 全套、scheduler/engine/kvcache 的算术测试、daemon、tokenizer、全部 parser 测试、`test_rotary.py`、`test_models_registry.py`。
- **CI 不跑 pytest**（四个 workflow 都只做 wheel 构建与发布）。测试是本地/pre-PR 闸门——意味着它没有被 CI 强制的"必然正确性"，读的时候保留一点怀疑。

---

## 6. 导航技巧（效率差异最大的一节）

```bash
# 1) 一个 flag 怎么一路流到引擎：从 argparse 开始
grep -rn '"--moe-cache-auto"' python/freetoken/          # -> server/args.py
# 然后顺着 ServerArgs -> SchedulerConfig -> EngineConfig -> engine._adjust_config 看

# 2) 任何"策略/后端/格式"名词，先找它的注册表或 Enum
grep -rn "Registry\[" python/freetoken/
grep -rn "_MODEL_REGISTRY\|ToolCallParserEnum\|ReasoningParserEnum\|MOE_STRATEGIES" python/freetoken/

# 3) 一个 AttnType 用到哪去了
grep -rn "AttnType\.DSV4" python/freetoken/

# 4) 环境变量全清单（53 个）——隐藏开关大半在这里
grep -rhoE '"FREETOKEN_[A-Z0-9_]+"' python/ | sort -u
# 另有 env.py 里 ENV.* 形式的：DISABLE_OVERLAP_SCHEDULING、MAMBA_SSM_DTYPE 等

# 5) 内核缓存/jit 路径
grep -rn "load_jit\|load_aot\|KERNEL_CACHE_DIR" python/freetoken/kernel/

# 6) 消息类型全局表
grep -n "^class " python/freetoken/message/*.py

# 7) 找"这条路径上的所有分支"
ls python/freetoken/attention/ python/freetoken/kvcache/ python/freetoken/moe/
```

**推荐给别人讲代码时的做法**：拿一个具体 checkpoint（比如 Qwen3.6-35B-A3B-FP8），把它 `config.json` 的 `architectures[0]` 作为钥匙——`_MODEL_REGISTRY` 查表 → `ModelSpec` → 你只读那一个模型族 + 它声明的 `AttnType` 对应的池和后端。**永远从具体 checkpoint 反推，不要从抽象层往下遍历。**

---

## 7. 术语表（黑话对照）

| 术语 | 含义 | 出现地 |
|---|---|---|
| **FTW** | FreeToken Weight：自研快速加载格式，连续字节区 + 4096 对齐 + 分片 + JSON 索引 | `checkpoint/ftw.py` |
| **MoE strategy** | `fused`(全常驻GPU) / `offload`(专家在主机内存, GPU 只放 LRU slot 缓存) / `cpu`(未命中在 CPU 算) / `hybrid`(每步按比例拆分) | `moe/__init__.py:9` |
| **q\*** | 论文里的带宽自适应策略，代码里 = hybrid 的 fetch fraction | `engine.py:766`；README 命名 |
| **slot cache** | GPU 端专家槽位池，全层共享的全局 LRU | `offload_cache.py:212` |
| **GDN** | GatedDeltaNet，线性注意力/循环状态层；每请求需要独立 state slot | `kvcache/linear_state_pool.py` |
| **SWA** | Sliding Window Attention；与 radix 组合时用"墓碑"标记滑出窗口的 KV | `kvcache/swa_radix_cache.py` |
| **DSA / BSA / QSA** | 三种稀疏注意力：DeepSeek sparse / MiniMax-M3 block-sparse / Qwen 压缩块稀疏 | `AttnType` + `attention/*_sparse.py` |
| **DSV4 / DSV41** | DeepSeek-V4 / V4.1：window + 压缩 KV + Lightning Indexer 的多层池 | `kvcache/dsv4/` |
| **PLE** | Qwen3.8-Flash-Next 的 n-gram 嵌入表（47.7 GiB 常驻主机内存） | `models/qwen4_exp/ple.py` |
| **Engram** | DSV4.1 的两张 98 GiB 表，按需从 checkpoint 分片流式读 | `tests/models/deepseek_v41/test_engram.py` |
| **b12x / marlin** | NVFP4 专家 GEMM 的两种内核（flashinfer b12x / vLLM Marlin W4A16） | `layers/quantization/moe/nvfp4.py` |
| **NVFP4 / MXFP4 / MXFP8 / FP8-block / Q4_0** | 五类量化格式，各有 linear 与 moe 两套内核 | `layers/quantization/` |
| **prefill overlap** | 权重双缓冲：预取 layer i+1 的专家，同时算 layer i | `offload_cache.py:609-636` |
| **hybrid radix** | radix 缓存 + GDN 状态快照（第二币种），非用户可见 | `kvcache/hybrid_radix_cache.py` |
| **toolcall anchor** | 模型吐出 tool-call 开头 token 时冻结的状态边界，客户端重写后仍可复用 | `core.py:62-64` |
| **`Fi`/`fi`, `fa`, `trtllm`, `triton`** | 四个通用注意力后端（flashinfer / flash-attn / TensorRT-LLM / 纯 Triton 回退） | `attention/__init__.py:38+` |
| **unified memory GPU** | GB10/DGX Spark：CPU/GPU 共享 DRAM，`auto` 直接解析成 `fused` | `engine.py:1141` |
| **manylinux2_28 / cu130** | 发布平台：glibc 2.28+, CUDA 13.0, torch 2.11；nightly 只有 cp312 | `.github/workflows/` |

---

## 8. 容易迷路的 8 个坑

1. **`message/` vs `tokenizer/`**：*类型*在 `message/`，*worker 实现*在 `tokenizer/`；而且同一个 worker 函数同时承担 tokenize 和 detokenize 两个角色。
2. **命名方向**：scheduler 从 tokenizer 收到的是 `BaseBackendMsg`，发回去的是 `BaseTokenizerMsg`。名字看方向会觉得别扭——按事实记，别按直觉推。
3. **`engine/engine.py` 是 1868 行**，但它是个"装配车间"，不是算法。方法名就能导航：`_load_weights` / `_init_offload_moe_cache` / `_resolve_hybrid_fetch` / `_target_moe_and_expert_bytes` / `rebuild_runtime_cache`。**不要顺序读**。
4. **两份 KV 预算实现**：`engine/cache_budget.py` 自称 single source of truth，但普通（非 auto）启动路径走的是 `_startup_kv_budget`（`engine.py:73-77`）+ `solve_num_pages`（`base.py:82`），形状相同但是另一份实现。看到 `memory_ratio` 相关代码时先确认你读的是哪一份。
5. **`overlap_loop` 让"当前批次"有歧义**：可能已 launch 但未 drain。任何"为什么资源还没释放"的问题，答案通常在这里。
6. **两处 `launch.py`**：仓库根的 `launch.py` 是 `ft launch`（给 coding agent 写配置）；`server/launch.py` 是 `ft serve` 的进程编排。同名不同层。
7. **`git log` 只有 100 条提交**（快照式历史），不要指望用 history 学习设计演进。**真正的设计文档是**：`docs/*.md`、各 README（`daemon/`、`tests/kvcache/radix/`、`benchmarks/`、`freetoken-kernel-cache/`）、以及源码里的 docstring——这个仓库的 docstring 质量很高，尤其是那些"为什么"和边界条件。
8. **`needs_weights` 标记不会跳过任何测试**；而且 CI 不跑 pytest。不要以为"测试里没写 = 行为如此"。

---

## 9. 建议节奏

**第 1 天（建立地图，约 3 小时）**
`README.md` → `docs/quickstart.md` → `docs/cli.md` → `docs/models.md` → `daemon/README.md`
然后 `core.py` 全读，`engine/config.py` 全读，`models/register.py` 全读。
收尾：`llm/llm.py` 全读。此时你应该能回答"一次请求经过哪些进程"。

**第 2-3 天（serving 主线）**
路线 A 的 12 步，配合 `message/*.py`。读完能画出完整的时序图。

**第 4-5 天（挑一个纵深）**
- 想要"这个项目凭什么在消费级硬件跑大模型" → 路线 B（MoE）
- 想理解缓存系统 → 路线 C（KV/radix），并读 `tests/kvcache/radix/` 全套
- 想加模型 → 路线 D，读 `models/llama/` 然后对照一个负责的族

**第 2 周（横向）**
把 `AttnType` × `resolve_pool_class` × 后端 `supported_types` 三张表对齐，你会突然看懂那些"某个模型为什么必须用某个后端"的报错信息。然后按需读 `kernel/`。

**检验理解的方法**（比读更多代码有效）：
1. 任选一个 flag（如 `--moe-cpu-layers auto`），说出它从 CLI 到最终生效的函数调用链。
2. 说出 `AttnType.DSV41` 对应的池类、后端、以及为什么。
3. 解释为什么 `needs_weights` 不是跳过标记，以及 `tests/kvcache/radix/README.md` 里那个 `>=` 边界 bug 为什么 8500 次随机测试都没抓到。
4. 找出一处"同一逻辑有两份实现"的地方（提示：第 8 节第 4 条）并说明它们为什么没合并。

---

## 10. 我可以继续帮你做什么

告诉我一个具体的东西，我可以带你把这条线一次走通（含逐行讲解、调用链图、以及对应的测试）：

- 「`--moe-cache-auto` 到底怎么算出来的」→ 从 argparse 到 `plan_cache_budget` 到 CUDA 分配
- 「prefix cache 命中时到底复用了什么」→ radix `_tree_walk` + `CacheManager.lock/cache_req` 逐行
- 「我要加一个新模型族，从哪下手」→ 对照 `models/llama/` 给你一份清单
- 「tool call 解析是怎么在流式里做的」→ `BaseFormatDetector` 契约 + 一个 detector 逐行
- 「QSA / DSA 稀疏注意力怎么实现的」→ 池 + indexer + Triton 内核三层
- 「为什么这个 kernel 要这么写」→ 挑一个 Triton/CUDA 内核逐行讲

也可以让我针对某个子系统产出更细的模块级导读（像本指南的第 3 节那样，但细到函数）。
