# 以"重写"为手段的学习课程：怎么读这个仓库才长能力

> 目标读者：**不为上线，只为练系统能力 / 性能分析 / 高性能算子**，打算把项目全部重写一遍的人。
> 与 [`CPP_REWRITE_GUIDE.md`](CPP_REWRITE_GUIDE.md) 的区别：那份为"尽快产出可用的 C++ 引擎"服务（换库、裁剪、竖切面）；**这份为"学到多少"服务**——所以该读的、该写的、该跳的都不一样。

---

## 0. 核心转变：从"必读"到"高学习密度"

生产型重写的原则是"只读必须的"。学习型重写要反过来：**读"独特、难、且有实测答案"的部分，跳过"必要但同质"的部分。**

| 生产型（上一份指南） | 学习型（本份） |
|---|---|
| 换库：flashinfer/CUTLASS 直接用 | **自己写**：Triton/CUDA 逐内核移植，这是练"算子能力"的唯一途径 |
| 裁剪：19 个模型族只留 1 个 | **精选**：留 2-3 个结构最特殊的族（DSV4.1/GDN）——它们教的东西最多 |
| 跳过 daemon/CLI | **同样跳过**（低学习密度，纯重复劳动） |
| 先跑通再优化 | **每个模块必须做到"能解释性能"才算完成** |

### 三种能力 → 对应可验证的练习

| 能力 | 仓库里对应内容 | 可验证的练习（做不出来就是没学会） |
|---|---|---|
| **系统能力** | 进程/线程模型、paged KV、radix、CUDA graph、offload cache、磁盘流式 | 从零写出一个**能跑前缀缓存 + 连续批处理**的引擎，且能解释每处同步的必要性 |
| **性能分析能力** | 注释里的实测百分比、`benchbw`、`perf_cuda`、`status.py` | 复现 12 个实验（§5），并给出 roofline 解释；能预测一次改动的收益方向与量级 |
| **高性能算子能力** | 55 个 Triton + 5.8k 行 CUDA + 量化格式 | Triton→CUDA 手写移植 ≥8 个内核，达到原版 ≥90% 带宽/算力 |

---

## 1. 学习型阅读的四条原则

### 原则一：读"为什么不用 X"，而不是"用了 X"
这个仓库的注释里藏着大量**设计否决**，这是判断力教材（比代码本身更值钱）：

| 否决 | 位置 | 学到什么 |
|---|---|---|
| decode 不能用 `triton.autotune` | `moe/fused_nvfp4.py:27-28` | 图捕获期不能做运行期基准测试 |
| `chunk_delta_h` 只能单配置 | `kernel/fla/chunk_delta_h.py:22-51` | autotune 会反复调用 in-place kernel → 破坏状态池（语义级约束） |
| 预算 MoE 优先而非 KV 优先 | `engine/cache_budget.py:58-95` | 目标函数决定分配顺序 |
| UMA GPU 上 auto→fused | `engine/engine.py:1141` | 物理拓扑改变"最优策略"的定义 |
| 显式 `--quant-backend` 不可用要报错 | `layers/quantization/method.py:20-47` | 静默降级是危险的工程选择 |
| 非 triton 后端遇 spec 要报错 | `attention/fi.py:209-212` | 静默忽略 ≠ 性能问题，而是正确性问题 |
| 批量拷贝 ≥256KB 才不退化 | `moe/offload_cache.py:17-26` | 批处理 API 由最坏成员决定 |

**读法**：看到 `# NOTE`/`# why`/`# without this` 就停下来，把"如果按直觉写会怎样"补出来。

### 原则二：读"有实测数字的注释"——它们是答案键
仓库注释里有大量 **A/B 百分比**（`~36% → ~64%`、`~65% → ~85%`、`-22% e2e`、`220 → 410 GB/s`）。这些不是宣传，是作者当时的测量记录。**它们让你能在自己实现之后判断"我做到 60% 是正常还是失败"。**

### 原则三：读测试里的"独立参照"，那是正确性判断力的来源
`tests/` 里有现成的独立参考模型（按页 radix 参考、vendored 参考 transformer、dequant oracle、CPU 镜像）。**读它们如何"独立地"定义期望值**，比读断言本身重要——这是你写差分测试时的模板。

### 原则四：读的顺序按"反馈速度"，不按"依赖顺序"
学习最怕"读三周才写出第一行能跑的东西"。正确的阶梯是：**先写反馈最快的（算子、内存），再写反馈最慢的（协议、多模型）。** 但引擎骨架要尽早有一个"能跑的最小闭环"（L2），否则你会失去坐标系。

---

## 2. 阅读的四类产出物（每读一个模块必须留下）

| 产出物 | 形式 | 用途 |
|---|---|---|
| **① 规格** | 5 页 markdown：布局/不变量/状态机/时序/错误契约 | 写 C++ 时的唯一依据（模板见 `CPP_REWRITE_GUIDE.md` §2） |
| **② roofline 记录** | 每个 kernel 一行：理论字节/FLOPs、实测时间、达成带宽、%峰值、优化手段 | 训练"数字感"；见 §6 模板 |
| **③ 复现实验** | 可运行脚本 + 实测值 + 与注释值对比 | **验证你真的读懂了机制**，而不是读懂了文字 |
| **④ 失败案例库** | "若按直觉写会怎样"的一句话记录 | 设计判断力；比成功案例更有迁移价值 |

**判定"读懂了"的唯一标准**：能产出 ③。只写笔记 = 没读懂。

---

## 3. 十级阶梯（每级都是"读 → 写 → 测 → 复盘"）

每级给出：读什么（最小集）→ 写什么 → 测什么 → 完成后你获得了什么。

### L0 · 张量与内存（1-2 周）
- **读**：`csrc/pinned_tensor.cpp`（精确尺寸分配、UVA/WDDM）、`csrc/jit/store.cu`（warp-per-token + 向量化）、`MODULE_DOCS/11-kernel-cpp.md` §3-4。
- **写**：自己的 `Tensor`（shape/stride/dtype/ownership）+ 显存 arena + pinned host 分配 + kernel launch 封装。
- **测**：pinned vs pageable 分配的时间/内存差异；`cudaHostAlloc` 精确尺寸 vs 2 的幂取整（仓库注释给了 70GB→128GB 的例）。
- **获得**：知道 torch 的 Tensor 抽象背后是什么；此后不再怕"内存/生命周期"这类问题。

### L1 · 单层前向（1-2 周）
- **读**：`models/llama/model.py:18`（decoder layer 残留结构）、`layers/{linear,norm,rotary,activation}.py`、`attention/fa.py`（或 triton 后端）。
- **写**：一个 decoder layer 的 eager 前向（qkv→rope→attn→o→mlp），用 cuBLAS 起步。
- **测**：与 Python 版单层输出逐元素比对（fp32 1e-5；bf16 用相对容差）。
- **获得**：能独立把一个层"翻译"成 CUDA/库调用。

### L2 · 第一个可跑闭环（2 周）★ 里程碑
- **读**：`tokenizer/tokenize.py`、`engine/sample.py`、`server/generation.py:136`（`GenSpec`）。
- **写**：tokenizer（BPE）+ greedy 采样 + 逐 token 解码循环 → **能生成文本的最小 C++ LLM**（无批处理、无缓存复用）。
- **测**：同一 prompt 贪心解码，**token 序列与 Python 版逐 token 一致**。
- **获得**：坐标系建立了——之后所有优化都有可对比的基线。**这一步做完再继续读数，效率高一个量级。**

### L3 · 分页 KV（2 周）
- **读**：`engine/engine.py:446-458`（page_table）、`kvcache/mha_pool.py`、`scheduler/cache.py:265`（`allocate_paged`）、`MODULE_DOCS/05` §3。
- **写**：页分配器 + page_table + KV 的 store/gather kernel。
- **测**：页守恒不变量（`free + used == total`、页对齐）、KV 内容与被覆盖前的中间张量逐字节一致。
- **获得**：**内存布局能力**——这是整个推理系统的地基。

### L4 · 连续批处理 + overlap（2-3 周）★ 最练并发
- **读**：`scheduler/scheduler.py:207-261`（overlap_loop、三道 `wait_stream`）、`scheduler/prefill.py:44`（`PrefillAdder`、`ChunkedReq`）、`MODULE_DOCS/02` §5。
- **写**：单进程多线程调度器（**不用 GIL 兜底**）+ 双 stream 流水 + 分块 prefill。
- **测**：同一批请求在 bs=1/8/32 下的吞吐；用 TSAN/竞争检测压一遍；**故意删掉一道围栏，复现它防的 bug**。
- **获得**：真正的并发/时序能力（这是 Python 里被 GIL 掩盖的部分）。

### L5 · 前缀缓存（2 周）★ 最练数据结构
- **读**：`kvcache/radix_cache.py:236`（`_tree_walk`）、`swa_radix_cache.py`（墓碑）、`hybrid_radix_cache.py`（第二币种）、`tests/kvcache/radix/README.md`（那个 `>=` 边界 bug 的取证）。
- **写**：radix 树 + lock/evict + 页引用计数。
- **测**：**直接拿仓库的参考模型当 oracle**（`tests/kvcache/radix/model.py` 按页建键）做随机操作序列比对；再复现"两个墓碑之间恰好一个窗口"的用例。
- **获得**：能把一个"有微妙不变量"的数据结构写对——这是系统能力的分水岭。

### L6 · CUDA graph（1-2 周）
- **读**：`engine/graph.py`（warmup→倒序捕获→共享 mempool→`copy_from`）、`MODULE_DOCS/01` §6。
- **写**：捕获/replay + 静态输入缓冲 + bs 阶梯（`[1,2,4]+range(8,max+1,8)`）+ pad 选择。
- **测**：launch 开销前后对比（µs 级）；验证"捕获期不能同步/分配"的具体报错形态。
- **获得**：`cudaGraph` API 的实战 + "什么能在图里"的判断力。

### L7 · 量化内核（3-4 周）★★ 核心算子训练
- **读**：`layers/quantization/method.py:20`（选择策略）、`kernel/triton/nvfp4_linear.py`（M==1/≤64/>64 三档 + 波次取舍 + 宽载 + 位技巧）、`fp8_block_linear.py`、`moe/fused_nvfp4.py`、`MODULE_DOCS/10` §4/§6.5。
- **写**：≥3 个内核的 CUDA 重写：FP8 block W8A16 GEMV、NVFP4 GEMV（用位技巧）、一个 grouped GEMM。
- **测**：**roofline 达成率**（§6）；与 Triton 版输出比对；复现注释里的百分比。
- **获得**：高性能算子的完整方法论：先算 roofline，再调 tile/split/访存宽度。

### L8 · MoE 与 offload（3-4 周）★★ 系统 + 算子 + I/O 三合一
- **读**：`layers/moe.py:261`（decode 三步）、`moe/offload_cache.py:170-252`（5 张映射表）、`moe/offload_kernels.py:19-40`（LRU 委托）+ `:290-410`（hybrid 三阶段）、`moe/host_banks.py`（pin-after-fill）、`csrc/jit/fast_index_copy.cuh`。
- **写**：slot 缓存 + 专家 gather kernel + 双缓冲 prefill + 一个 CPU 执行器（可选）。
- **测**：miss 率曲线随 `cache_size` 变化（与 Python 版一致）；gather 带宽随 `blocks_per_bank` 的饱和曲线。
- **获得**：**唯一一处能把"内存层级 + 内核 + 调度 + I/O"全串起来的训练**。

### L9 · 特殊系统（按兴趣选，每个 2-3 周）
| 选题 | 读 | 训练什么 |
|---|---|---|
| 磁盘流式表（Engram/PLE） | `models/deepseek_v41/engram_table.py:1-20`（20 行设计说明）、`csrc/row_store/` | O_DIRECT/io_uring + stream memop + 与图捕获的死锁规避 |
| 线性注意力（GDN/KDA） | `models/qwen3_5_moe/gdn.py`、`kernel/fla/__init__.py:1-48` | 分块算法 + in-place 状态池的约束 |
| 稀疏注意力（DSA/QSA/BSA） | `attention/{dsa,qsa_sparse,m3_sparse}.py` + 对应 Triton | 索引器 + 压缩块 + top-k 规划 |
| 多模态 | `mm/processor.py`、`models/*/vision.py` | 占位符扩展 + 编码缓存 |

---

## 4. 必须"深读"的 12 个文件（每个教会一件事）

| # | 文件 | 教会你 |
|---|---|---|
| 1 | `csrc/jit/fast_index_copy.cuh` | streaming 访存（`L1::no_allocate`/`st.wt`）+ grid 宽度与链路饱和 |
| 2 | `csrc/jit/store.cu` | warp-per-token、K/V 一趟、16/8/4 字节向量化 |
| 3 | `kernel/triton/nvfp4_linear.py` | 三档 dispatch + split-K 波次 + 宽载 + 位技巧 |
| 4 | `kernel/triton/attention.py:21-42` | smem 感知的 tile 选择 |
| 5 | `moe/offload_kernels.py:290-410` | 单 CTA 内做 LRU + makespan 分摊（Q16 整数解） |
| 6 | `moe/offload_cache.py:749-819` | 批量 I/O 的退化陷阱与对策（256KB） |
| 7 | `engine/graph.py:140-232` | 图捕获的完整工程细节 |
| 8 | `scheduler/scheduler.py:207-261` | 双 stream 流水的三道围栏 |
| 9 | `kvcache/radix_cache.py:236-262` | 按页 trie + 部分匹配分裂 |
| 10 | `kvcache/swa_radix_cache.py:88-135` | 墓碑与"第二币种"的锁纪律 |
| 11 | `engine/cache_budget.py:48-95` | 把策略写成纯算术（可无 GPU 单测） |
| 12 | `models/deepseek_v41/engram_table.py:1-20` | 约束求解式设计（图 + 主机喂数 + 死锁规避） |

---

## 5. 十二个可复现实验（答案在注释里）★ 性能分析训练的核心

每个实验：写最小复现 → 测 → 与"答案键"对比 → 解释差异。**这些是本仓库给你的、别处买不到的实验课。**

| # | 实验 | 答案键（注释里的实测） | 位置 |
|---|---|---|---|
| 1 | 空 kernel × 1000 vs CUDA graph replay | launch 开销量级 μs | `engine/graph.py` |
| 2 | pinned vs pageable 的异步 D2H | **仓库这里有个坑**：采样 token 的 D2H 目标未 pin → 退化为同步（见优化指南 §1.1） | `engine/engine.py:1061` |
| 3 | gather 带宽 vs `blocks_per_bank` | PCIe ~4K 线程/bank 饱和；C2C 需 ~32K（222→412 GB/s） | `kernel/fast_index_copy.py:152-155` |
| 4 | `cudaMemcpyBatchAsync` 的小条目退化 | 5-22KB 条目旁大条目 → 同步；≥253KB 不触发；e2e -22% | `moe/offload_cache.py:17-26` |
| 5 | 反量化：L1 LUT vs ALU 位技巧 | 36% → 64% 权重带宽屋顶 | `kernel/triton/nvfp4_linear.py:12-19` |
| 6 | FP4 码的 int32 宽载 vs 逐字节 | 65% → 85% 峰值读带宽 | `kernel/triton/nvfp4_fused_moe.py:160-174` |
| 7 | split-K 的波次量化 | `min_grid = wave - wave//32`；贴着一波会掉 | `kernel/triton/nvfp4_linear.py:523` |
| 8 | GEMV/GEMM 阈值 | M==1 / M≤64 / M>64（>64 回 cuBLAS 快 3×） | `kernel/triton/nvfp4_linear.py` |
| 9 | CPU MoE spin vs park | 每层一次 futex 唤醒的成本；spin kernel 会把利用率钉 99% 并降频 | `csrc/cpu_moe/cpu_moe_ext.cpp:1886` |
| 10 | radix 命中率 → TTFT | 命中 token 数直接省 prefill | `scheduler/status.py` 的日志字段 |
| 11 | MoE cache size → miss 率 → 步时 | 专家槽字节公式（`_BANK_BYTES_PER_EXPERT`）可算理论值 | `moe/offload_cache.py:87-97` |
| 12 | 图捕获 bs 阶梯的显存代价 | `logits [max_bs, vocab] fp32`（160×150k×4B ≈ 96MB） | `engine/graph.py:45` |

**做法**：建一个 `labs/` 目录，每个实验一个 `.cu` + 一个结果 md（含你的数值、注释数值、差异解释）。**这 12 个做完，性能分析能力就成型了。**

---

## 6. 内核移植方法论（Triton → CUDA 六步）

以 `nvfp4_linear` 的 M==1 GEMV 为例：

```
① 读契约：输入/输出 shape、dtype、是否 in-place、阈值（M==1）
② 算 roofline：需要读多少权重字节？算多少 FLOPs？
     → 结论："纯权重流式"，上限 = 权重字节 / 峰值带宽
③ 读 Triton 的决策，而不是语法：
     - 为什么 split-K（要凑够 ~1 wave 的 block）
     - 为什么 int32 宽载（合并事务）
     - 为什么位运算代替 LUT（LSU 串行化）
④ 用 CUDA 重写（先正确，后快）：thread mapping → 向量化访存 → 寄存器累加 → split-K 归约
⑤ 用 roofline 判断"还能不能快"：达到峰值带宽的百分比，几乎就是天花板
⑥ 回到注释对答案：注释说 64%，你到 55% → 差在 LSU/pipeline；到 62% → 收工
```

**记录模板**（每个内核一行，建议新建 `labs/kernels.md`）：

`labs/kernels.md`（每个内核一行的记录表）：

| kernel | 输入形状 | 理论字节 | 理论 FLOPs | 实测 μs | 达成 GB/s | %峰值 | 优化手段 | 剩余差距 |
|---|---|---|---|---|---|---|---|---|

**三个常见反模式（都在本仓库有反面教材）**：
1. **先调 tile 再算 roofline** —— 不知道上限在哪，调参就是碰运气（对照 `nvfp4_linear.py:523` 的注释，作者是**先算 wave** 再定 split）。
2. **用"利用率 100%"当目标** —— 自旋 kernel 能把利用率钉在 99% 但把 CPU 降频（`cpu_moe_ext.cpp:573-585`）。
3. **相信注释/文档而不是测量** —— 仓库里已确认至少 4 处注释与实现不一致（优化指南 §5.1）。

---

## 7. "全部重写"的现实边界与里程碑

### 现实边界（必须承认，否则会烂尾）
全量是 **93k Python + 21k Triton + 13k CUDA**，约 2-3 人年。对学习目标，**边际收益递减**的部分明确建议"适配而非重写"：

| 建议 | 内容 | 理由 |
|---|---|---|
| **必写**（约 55%） | tensor/allocator、内核、调度、KV/radix、graph、量化、MoE、磁盘 I/O | 三种能力的全部训练点都在这 |
| **可选** | 协议层/parser/tokenizer（BPE 自己写，chat template 用库） | 工程量大但学习密度中等 |
| **不重写**（约 25%） | 16/19 个模型族、`daemon`/`shell`/桌面适配、vendored GGUF dequant（5.8k 行 llama.cpp 移植） | 纯重复劳动 |

### 用"能力里程碑"而不是"功能清单"定义进度
| 里程碑 | 判据 | 对应阶梯 |
|---|---|---|
| M1 | 1B 模型贪心 decode，token 与 HF 逐 token 一致 | L0-L2 |
| M2 | bs=32 连续批处理，吞吐 ≥ Python 版 70% | L3-L4 |
| M3 | radix 语义通过仓库参考模型的随机操作比对 | L5 |
| M4 | CUDA graph 下 decode 步时与 Python 版持平 | L6 |
| M5 | fp8 GEMV 达到 Python 版 90% 带宽 | L7 |
| M6 | offload MoE 跑通 30B-A3B，miss 率曲线一致 | L8 |
| M7 | 自研磁盘表达到 O_DIRECT 上限 80%，且能在图内不死锁 | L9 |

**纪律**：main 分支**永远可跑**（每个里程碑都是一个能演示的 demo）；不允许"重构三个月没有可运行物"。

---

## 8. 每周节奏模板

```
周一   读：本阶梯"读什么"的那 1-2 个文件（只读，不写）
周二   写规格（5 页）+ 跑通该模块的 Python 版，记录基线数字
周三-四 写 C++（先正确，允许慢）
周五   测：与 Python 版比对 + roofline 记录 + 复现一个相关实验
周末   复盘：写"失败案例库"一条 + 更新 kernels.md/labs 结果
```

**每周的最低交付**：一个能跑的 C++ 模块 + 一次数值比对 + 一条 roofline 记录 + 一条"如果按直觉写会怎样"。

---

## 9. 最容易让学习项目烂尾的五个模式（提前打预防针）

1. **先读完再写** —— 93k 行读不完。正确做法：L2 之前就要有可跑闭环。
2. **一开始就自研 Tensor + 全套基础设施** —— 热情会耗尽。建议：先用 cuBLAS/ATen 打通 L1-L2，再回头替换（此时你才知道要什么）。
3. **只写不测** —— 没有差分测试，你在写"看起来对"的代码，能力不会长。
4. **追全部模型族** —— 第 3 个族之后学习曲线就平了。
5. **不敢删/不敢改** —— 这是你的练习项目，可以为了"练某件事"故意简化（例如只支持 page_size=1、只支持单卡）。

---

## 10. 如果只做一件事

**做 L4 + L5 + L7 三级**（并发调度 → 前缀缓存 → 量化内核），它们分别独立训练"系统/数据结构/算子"三种能力，且每一级都有现成的 oracle 可以验收。
其余级别是"把这三样能力组合起来"的应用练习——有精力再展开。
