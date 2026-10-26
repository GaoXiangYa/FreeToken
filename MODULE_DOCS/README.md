# FreeToken 模块文档

这套文档把 `python/freetoken` 的 **439 个 Python 文件 / 93,387 行 / 4,597 个类与函数符号（含嵌套函数）**拆成 18 篇模块文档 + 一份自动生成的符号索引。

> 行号基准：commit `377a3bc`。代码变动后请用符号名 grep 兜底。
> 本机无 GPU、无 torch，全部内容来自**静态源码阅读**；无法从代码确证的条目已标注"（未验证）"。

**目录结构**

```
MODULE_DOCS/
├── README.md                  ← 本文件（总索引与使用说明）
├── 00-architecture.md         ← 架构总纲（建议先读）
├── 01..17-<module>.md         ← 18 篇模块文档（编号即阅读顺序）
├── 99-symbol-index/           ← 20 个自动生成的符号索引（4,597 条，含行号）
└── tools/gen_symbol_index.py  ← 索引生成器（纯 ast，可重复执行）
```

---

## 1. 怎么用这套文档

| 你想做的事 | 从哪开始 |
|---|---|
| **第一次接触这个仓库** | [`00-architecture.md`](00-architecture.md)（架构总纲 + 依赖矩阵 + 数据流）→ [`01-core-engine.md`](01-core-engine.md)（词汇表） |
| **搞清楚一条请求怎么走** | [`03-server-http.md`](03-server-http.md) → [`02-scheduler.md`](02-scheduler.md) → [`01-core-engine.md`](01-core-engine.md) |
| **做性能优化** | 先看根目录的 [`INFERENCE_OPTIMIZATION_READING_GUIDE.md`](../INFERENCE_OPTIMIZATION_READING_GUIDE.md)，再按需查 [`06-moe.md`](06-moe.md) / [`05-kvcache.md`](05-kvcache.md) / [`12-kernel-triton-fla.md`](12-kernel-triton-fla.md) |
| **加一个新模型族** | [`07-models-core.md`](07-models-core.md) §8 的 6 步法 → [`08-models-families.md`](08-models-families.md) §9 的 8 步清单 |
| **查某个函数干什么** | 先用 [`99-symbol-index/`](99-symbol-index/) 定位（含行号），再跳到对应模块文档看详解 |
| **改内核 / 换量化格式** | [`10-layers-quant.md`](10-layers-quant.md)（选择策略与谓词）→ [`11-kernel-cpp.md`](11-kernel-cpp.md) / [`12-kernel-triton-fla.md`](12-kernel-triton-fla.md) |
| **看整体源码阅读路线** | 根目录的 [`SOURCE_READING_GUIDE.md`](../SOURCE_READING_GUIDE.md) |

---

## 2. 文档地图（18 篇 / 约 5563 行）

| # | 文档 | 覆盖的源码 | 符号数 | 一句话 |
|---|---|---|---:|---|
| 00 | [`00-architecture.md`](00-architecture.md) | — | — | **架构总纲**：分层、依赖矩阵（实测 import）、进程拓扑、三条数据流、横切关注点 |
| 01 | [`01-core-engine.md`](01-core-engine.md) | `core.py` + `engine/` | ~101 | `Req`/`Batch`/`Context`、`EngineConfig`、显存预算、CUDA graph、采样、`forward_batch` |
| 02 | [`02-scheduler.md`](02-scheduler.md) | `scheduler/` | ~130 | 调度循环与 overlap、分块 prefill、`CacheManager`、前缀提交、状态上报、运行时重建 |
| 03 | [`03-server-http.md`](03-server-http.md) | `server/`（除 parser） | ~250 | 进程编排、三种协议适配、中立 `GenSpec`、15 条 HTTP 路由、统计与记账 |
| 04 | [`04-server-parsers.md`](04-server-parsers.md) | `function_call_parser.py` + `reasoning_parser.py` | ~120 | 15 个工具调用 detector + 7 个思维链 parser、流式标记切分 |
| 05 | [`05-kvcache.md`](05-kvcache.md) | `kvcache/` | ~190 | 分页池族、三套 radix 前缀缓存、线性状态池、DSV4 分层池 |
| 06 | [`06-moe.md`](06-moe.md) | `moe/` | ~110 | 专家卸载缓存、LRU 准入、PCIe gather、双缓冲 prefill、CPU/hybrid 执行、带宽剖面 |
| 07 | [`07-models-core.md`](07-models-core.md) | `models/` 顶层 | ~62 | 架构注册表（29 项）、`ModelConfig` 契约、权重加载/分片/融合/堆叠 |
| 08 | [`08-models-families.md`](08-models-families.md) | `models/` 的 19 个族 | ~500 | 族 × 特性矩阵、四个代表族深度、15 族逐个说明、加新族清单 |
| 09 | [`09-attention.md`](09-attention.md) | `attention/` | ~160 | 9 个注意力后端、能力矩阵、metadata 构造、prefill/decode 分流 |
| 10 | [`10-layers-quant.md`](10-layers-quant.md) | `layers/`（含 `quantization/`） | ~220 | 层原语、量化方言、`select_kernel` 策略、格式→内核全景、阈值清单 |
| 11 | [`11-kernel-cpp.md`](11-kernel-cpp.md) | `kernel/` 顶层 + `csrc/` | ~100 | JIT/AOT 缓存、三个 torch 扩展、csrc 内核逐个详解、预编译 wheel |
| 12 | [`12-kernel-triton-fla.md`](12-kernel-triton-fla.md) | `kernel/triton/` + `kernel/fla/` | ~300 | Triton 算子全景、量化 kernel 的性能技巧、常量总表、FLA 与两条 landmine |
| 13 | [`13-checkpoint-ftw.md`](13-checkpoint-ftw.md) | `checkpoint/` | ~41 | FTW 格式规范、写入/读取侧、模型无关的转换、hotfix |
| 14 | [`14-tokenizer-message-llm.md`](14-tokenizer-message-llm.md) | `message/` + `tokenizer/` + `llm/` | ~85 | 跨进程消息全表与 codec、tokenize/detokenize worker、离线 API |
| 15 | [`15-mm.md`](15-mm.md) | `mm/` | ~40 | 占位符扩展、媒体获取与安全、编码缓存、7 个族处理器 |
| 16 | [`16-daemon-cli-shell.md`](16-daemon-cli-shell.md) | `daemon/` + `shell/` + 顶层 CLI | ~180 | torch-free 控制面（17 条路由）、TUI、6 个 CLI 命令链、6 个 agent |
| 17 | [`17-infra-tests-bench.md`](17-infra-tests-bench.md) | `utils/` + `distributed/` + `benchmark/` + `tests/` + `benchmarks/` + `scripts/` | ~120 | Registry、ZMQ 队列、TP、测量模板、141 个测试的方法论与映射 |

每篇结构统一：**模块职责 → 文件清单 → 核心数据结构（字段表）→ 函数详解（行号/作用/调用方）→ 关键流程逐段 → 与其他模块关系 → 常见坑 → 自测问题**。

---

## 3. 符号索引（`99-symbol-index/`，20 个文件）

**自动生成**（`ast` 解析，不执行代码，无 torch 依赖），逐文件列出**全部**类/函数/方法/嵌套函数、行号、docstring 首行。

- **覆盖度已核验**：源码 `ast` 可见 **4,597 个符号**，索引 **4,597 条，0 遗漏、0 重复**；439 个 `.py` 文件全部收录。
- 嵌套函数用 `↳` 与缩进标出层级（例如定义在 `match case` 分支里的 `get_mscale`）。
- 说明列取 docstring 首行；没有 docstring 时回退到签名首行。

| 文件 | 对应目录 |
|---|---|
| [`attention.md`](99-symbol-index/attention.md) · [`checkpoint.md`](99-symbol-index/checkpoint.md) · [`daemon.md`](99-symbol-index/daemon.md) · [`distributed.md`](99-symbol-index/distributed.md) · [`engine.md`](99-symbol-index/engine.md) · [`kernel.md`](99-symbol-index/kernel.md) · [`kvcache.md`](99-symbol-index/kvcache.md) · [`layers.md`](99-symbol-index/layers.md) · [`llm.md`](99-symbol-index/llm.md) · [`message.md`](99-symbol-index/message.md) · [`mm.md`](99-symbol-index/mm.md) · [`models.md`](99-symbol-index/models.md) · [`moe.md`](99-symbol-index/moe.md) · [`scheduler.md`](99-symbol-index/scheduler.md) · [`server.md`](99-symbol-index/server.md) · [`shell.md`](99-symbol-index/shell.md) · [`tokenizer.md`](99-symbol-index/tokenizer.md) · [`top-level.md`](99-symbol-index/top-level.md) · [`utils.md`](99-symbol-index/utils.md) · [`benchmark.md`](99-symbol-index/benchmark.md) | 同名模块 |

**用法**：
```bash
# 查某个模块里有哪些符号（行号 + 说明）
grep -n "^| [0-9]" MODULE_DOCS/99-symbol-index/engine.md

# 查某个符号在哪、干什么（跨全部模块）
grep -n "plan_cache_budget" MODULE_DOCS/99-symbol-index/*.md

# 查某个文件收录了哪些符号（文件段落内的行）
grep -n -A2 "^## .freetoken/kvcache/radix_cache.py" MODULE_DOCS/99-symbol-index/kvcache.md
```

**已知偏差**：索引里的"行数"是逻辑行数（末行无换行时按 1 行算），可能与 `wc -l` 差 1。**符号行号本身是准的**。

---

## 4. 与另外两份指南的关系

| 文档 | 定位 |
|---|---|
| **本目录（`MODULE_DOCS/`）** | **按模块组织**的 API 级参考：每个函数干什么、字段什么含义、模块间怎么关联 |
| [`../SOURCE_READING_GUIDE.md`](../SOURCE_READING_GUIDE.md) | **按阅读顺序组织**的路线图：先读什么后读什么、哪里容易迷路 |
| [`../INFERENCE_OPTIMIZATION_READING_GUIDE.md`](../INFERENCE_OPTIMIZATION_READING_GUIDE.md) | **按性能组织**的专题：瓶颈模型、六个优化母题、调优旋钮、性能误判 |
| [`../ARCHITECTURE.md`](../ARCHITECTURE.md) | **八视图架构图集**：分层/依赖/进程/内存/数据流/时序/加载/抽象（PNG+SVG+ASCII） |
| [`../CPP_REWRITE_GUIDE.md`](../CPP_REWRITE_GUIDE.md) | **面向 C++ 重写**：可移植边界、契约清单、依赖替换表、P0-P6 阶段 |
| [`../LEARNING_BY_REWRITING.md`](../LEARNING_BY_REWRITING.md) | **学习导向**：十级阶梯、12 个可复现实验、roofline 训练法、能力里程碑 |

五者互补：**要查**→本目录；**要学**→阅读指南；**要快**→优化指南；**要练能力**→学习课程。

---

## 5. 推荐的三条阅读路线

### 路线 A：新人通读（约 2 天）
`00` → `01` → `03` → `02` → `05` → `06` → `09` → `10` → `07` → `08` → 其余按需。

### 路线 B：性能工程师（约 1.5 天）
`00` §5-6 → `01` §6-9（CUDA graph / forward_batch）→ `02` §5.1（overlap_loop）→ `06`（MoE 每步代价）→ `05` §3-4（缓存与预算）→ `10` §6.5 + `12` §6（内核阈值）→ `17` §3（测量工具）。

### 路线 C：加模型 / 改内核（约 1 天）
`07` → `08` §2-4 + §9 → `10` §3-6 → `09` §3-4 → `12` §7（怎么读 Triton）。

---

## 6. 维护

### 6.1 重新生成符号索引

生成器已随文档提供：**`MODULE_DOCS/tools/gen_symbol_index.py`**（纯 `ast`，不 import 源码、不需要 torch/GPU）。

```bash
cd <repo root>
python3 MODULE_DOCS/tools/gen_symbol_index.py
# 输出每个模块的 "N files / M symbols"，末行是 TOTAL
```

改动代码后建议的流程：

```bash
# 1) 重生成索引
python3 MODULE_DOCS/tools/gen_symbol_index.py

# 2) 核验：索引符号数是否等于源码 ast 可见符号数（当前 4,597）
python3 - <<'PY'
import ast, os
n = 0
for dp, dn, fns in os.walk("python/freetoken"):
    dn[:] = [d for d in dn if d != "__pycache__"]
    for fn in fns:
        if fn.endswith(".py"):
            try:
                t = ast.parse(open(os.path.join(dp, fn), encoding="utf-8").read())
            except SyntaxError:
                continue
            n += sum(1 for x in ast.walk(t)
                     if isinstance(x, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)))
print("源码符号数:", n)
PY

# 3) 核验：文档里的 file:line 引用是否仍落在有效范围
grep -rhoE '`?[A-Za-z0-9_./-]+\.py:[0-9]+' MODULE_DOCS/*.md | wc -l
```

### 6.2 生成器的设计约束（改脚本时请保持）

- **纯 `ast`，绝不 import/执行源码**——否则在无 GPU/无 torch 的机器上会失败；
- 递归进函数体收集**嵌套函数**（用 `↳` 标层级），并处理 `if`/`try`/`with`/`for`/`while`/`match` 分支里的定义；
- 行号 = `def`/`class` 的起始行；
- 按**顶层目录**分桶，文件名 `99-symbol-index/<module>.md`；
- 说明列优先 docstring 首行，没有则回退签名首行。

### 6.3 加新文档时请保持

1. 文件名 `NN-<module>.md`，`NN` 与本文件 §2 的地图一致；
2. 沿用统一的 8 节结构：模块职责 → 文件清单 → 核心数据结构 → 函数详解 → 关键流程 → 与其他模块关系 → 常见坑 → 自测问题；
3. 每个事实落到 `文件:行`；
4. 区分"注释里记录的实测数字"与"作者的推断"；
5. 无法确证的写"（未验证）"，不要编；
6. 新增/改名文档后，同步更新本文件 §2 的地图与 §4 的路线。


---

## 7. 已知边界

- **`flashlib` 的 LRU 内核不在本仓库**（`moe/offload_kernels.py:8` 的外部依赖）——`06` 里只能给出 hybrid 变体与调用契约。
- **`kernel/triton/` 的模型专属子目录**（`dsv4/`、`dsv41/`、`qsa/`）在 `12` 中按目录概括，未逐文件展开。
- **19 个模型族**在 `08` 中四个深度、十五个概览；逐族逐函数需查 `99-symbol-index/models.md`。
- 所有"性能数字"都是**源码注释里记录的历史实测值**，不是在本机复现的结果。
