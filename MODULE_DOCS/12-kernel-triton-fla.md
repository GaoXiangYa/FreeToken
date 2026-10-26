# 12 · kernel（Triton 与 FLA）：通用算子、量化线性/MoE、稀疏注意力、线性注意力内核

> 行号基于 `377a3bc`。完整符号清单见 [`99-symbol-index/kernel.md`](99-symbol-index/kernel.md)。`kernel/` 顶层与 `csrc/` 见 `11`。

---

## 1. 模块职责与在系统中的位置

`kernel/triton/` 与 `kernel/fla/` 是**纯 Python 实现的算子层**，承担两个角色：

1. **回退路径**：没装 flashinfer / sgl-kernel 时，通用算子（激活、norm、rope、注意力、采样、量化 linear/MoE）走 Triton；
2. **唯一实现**：所有**模型专属稀疏算子**（DSV4/DSV41/DSA/BSA/QSA）与**线性注意力**（GDN/KDA）只有 Triton/FLA 实现。

```
attention/*_sparse.py ──┐
layers/quantization/ ───┼──▶ kernel/triton/*  ──▶ Triton 编译器 ──▶ GPU
moe/fused*.py ──────────┘
models/{qwen3_5_moe,qwen4_exp,glm5_next} ──▶ kernel/fla/*  ──▶ Triton 编译器
```

`kernel/triton/` 是 **namespace package（没有 `__init__.py`）**；只有 `qsa/`、`dsv41/` 有。

---

## 2. `kernel/triton/` 文件全景表（55 文件 / 约 15,000 行）

### 2.1 通用注意力与采样

| 文件 | 行数 | 提供的算子 | 阅读建议 |
|---|---:|---|---|
| `attention.py` | 978 | extend（prefill）与 decode 两套注意力、`_select_extend_tile` 的 smem 自适应、split-KV 两级归约 | **精读** |
| `sampling.py` | 651 | softmax + top-k/top-p + 采样（split-vocab 多 CTA、cooperative launch） | 精读 |
| `rope.py` | 271 | RoPE / mRoPE 内核 | 按需 |
| `causal_conv1d_triton.py` | 682 | 因果深度可分离卷积（GDN 的 conv 段） | 按需 |
| `norm.py` | 193 | RMSNorm/LayerNorm（**最好的 Triton 读法样板**） | **精读** |
| `activation.py` | 200 | silu/gelu/swiglu 系列融合激活 | 按需 |

### 2.2 量化 Linear

| 文件 | 行数 | 提供的算子 |
|---|---:|---|
| `nvfp4_linear.py` | 819 | NVFP4 W4A16：M==1 GEMV / M≤64 核内 dot GEMM / M>64 dequant+cuBLAS；K-major 常驻布局；split-K |
| `fp8_block_linear.py` | 280 | FP8 block-scale：W8A16 GEMV（M=1）/ 动态 1×128 W8A8 GEMM |
| `fp8_pertensor_linear.py` | 357 | FP8 per-tensor：`_scaled_mm` W8A8 / Triton split-K GEMV / W8A16 GEMM |
| `mxfp8_linear.py` | 253 | MXFP8 W8A16 GEMV（`_GEMV_MAX_M=256`）/ dequant+cuBLAS |
| `nvfp4_dequant.py` | 134 | FP4 反量化独立内核（调试/参考用） |

### 2.3 量化与 bf16 MoE

| 文件 | 行数 | 提供的算子 |
|---|---:|---|
| `fused_moe.py` | 303 | 通用 fused MoE（routed GEMM + 归约） |
| `decode_moe.py` | 76 | decode 专用 per-route GEMV |
| `moe_align.py` | 329 | `moe_align_block_size`（按专家把 token 排序分块） |
| `moe_router.py` | 161 | 路由 top-k（含分组路由）、PDL |
| `moe_shared_gate.py` | 138 | shared expert 的门控与归约 |
| `nvfp4_fused_moe.py` | 334 | NVFP4 专家：int32 宽载 + 延迟 K 归约 |
| `mxfp4_moe.py` | 333 | MXFP4 / ds_fp4 专家（N-innermost 布局、split-K） |
| `fp8_blockscale_moe.py` | 263 | FP8 block-scale 专家（grouped prefill + GEMV decode） |

### 2.4 模型专属算子

| 文件 | 行数 | 服务谁 |
|---|---:|---|
| `minimax_m3_sparse.py` | 1,119 | MiniMax-M3 的 BSA block-sparse |
| `glm_dsa_sparse.py` | 432 | GLM 的 DSA 稀疏注意力 |
| `hc.py` / `mhc.py` | 396 / 348 | 超连接（GLM / Qwen4Exp） |
| `df11.py` / `df11_decode.py` | 319 / 172 | GLM-4.7 系 |
| `gemma4_fused.py` | 192 | Gemma-4 专用融合算子 |
| `kpool_compress.py` | 147 | GLM kpool 的压缩 |
| `ple.py` | 96 | PLE 表的行 gather |
| `autotune_cache.py` | 38 | **基础设施**：让 `@triton.autotune` 结果持久化 |
| `e4m3_compat.py` | 143 | **基础设施**：强制 fp8-e4m3 仿真时的独立缓存目录 |

### 2.5 三个子目录（各自服务一个模型族）

| 子目录 | 文件数 / 行数 | 服务谁 | 关键 kernel |
|---|---:|---|---|
| `dsv4/` | 13 / 约 2,100 | DeepSeek-V4 | `sparse_attn.py`(394)、`fp8_linear.py`(520)、`fused_moe.py`(327)、`indexer.py`(227)、`compress.py`、`router.py`、`sinkhorn.py`、`hadamard.py`、`hc.py`、`norm.py`、`rope.py`、`swiglu.py`、`bf16_linear.py` |
| `dsv41/` | 6 / 约 1,060 | DeepSeek-V4.1 | `sparse_attn.py`(324)、`topk.py`(266)、`pack.py`(184)、`indexer.py`(177)、`row_format.py`(93) |
| `qsa/` | 6 / 约 1,670 | Qwen3.8-Flash-Next | `topk.py`(627)、`attend.py`(361)、`compress.py`(332)、`score.py`(192)、`expand.py`(132) |

---

## 3. 关键算子详解

### 3.1 `triton/attention.py` —— 通用注意力

- **两套 kernel**：extend（prefill / 带缓存命中的 extend）与 decode。decode 是 **flash-decoding 式的两级 split-KV**（stage1 分块算部分 softmax，stage2 归约），`_MAX_KV_SPLITS = 8`（`:10`）、`_MIN_BLOCK_KV = 32`（`:11`）。
- **`_select_extend_tile(head_dim, block_d, smem_optin)`（`:21`）**：按 head_dim 与**共享内存余量**选 tile：
  - head_dim ≤ 128 → 128×64
  - ≤256 → 若 `(BLOCK_M + 2*BLOCK_N)*BLOCK_D*2 <= 0.8*smem_optin` 用 (128,64)，否则 (64,32)
  - 更大 → 32×64 / (32,32) / (16,16)
  注释说明：**更大的 tile 明显更快（head_dim 512 在 H100 上约 2×），只在放不下时缩**。
- 启动参数：extend `num_warps=8, num_stages=1`；decode stage1 `BLOCK_N=32, VALID_BLOCK_H=min(16,GQA), num_warps=4, num_stages=2`。

### 3.2 `triton/nvfp4_linear.py` —— 量化 Linear 的典范

**三档 dispatch**（按 M 选择，运行期判断）：

| 条件 | 走什么 | 位置 |
|---|---|---|
| `M == 1` | 专用 GEMV（split-K） | `:768` |
| `M <= _GEMM_MAX_INKERNEL_M (=64)` | 核内 dot GEMM | `:95`、`:755` |
| `M > 64` | 反量化到 bf16 scratch + cuBLAS | doc `:28-31` |

**为什么 M>64 要回 cuBLAS**：注释明确"核内反量化 GEMM 会把每个权重 tile 重复反量化 M/BLOCK_M 次，H100 上最慢可达 cuBLAS 的 3 倍"。

**几个可迁移的技巧**（注释都带实测数字）：
1. **int32 宽载**（`nvfp4_fused_moe.py:160-174`）：FP4 码按 int32 字加载（每 4 字节 8 个码），让 HBM 发出宽合并事务——"gate/up GEMM 的纯访存上限从 ~65% 提到 ~85%"。
2. **e2m1→fp16 的位技巧**（`:12-19`）：不用 L1 LUT gather（"gather 把 LSU 串行化，旧 decode GEMV 卡在权重带宽屋顶的 ~36%"），改用 ALU 位运算（"到 ~64%"，且逐位一致）。
3. **wdot 换操作数**（`:357-377`）：把反量化权重当 MMA 的 lhs，去掉每 tile 一次 smem 往返（"从 ~50% Memory SOL 提上来"）。
4. **split-K 的波次取舍**（`_pick_split_k_bm16:523`）："grid 落在恰好一波的 ~15% 内会量化得很糟（一整波 + 一个拖尾小波）——K 允许时取两波"。
5. **整除优先**（`:569-576`）：优先选能整除 K-tile 数的 split，让程序走无掩码的 EVEN_K 循环（"快 15-30%"）。
6. **融合 split-K 归约**（`:100-116`）：用自重置到达计数器省掉单独归约 kernel（"M==1 decode linear 上 ~2 μs 的尾巴"）。
7. **对齐**（`:726-729`）：dequant+cuBLAS 的 N 保持 256 对齐——"奇数 N（如 13107）会让 cuBLAS 掉到约 4× 慢的 misaligned kernel"。

### 3.3 `triton/sampling.py` —— 采样

- **split-vocab 多 CTA**：bs=1 也能用满 SM（而不是单 CTA 扫全词表）。
- **cooperative launch 的 occupancy 预算**：`_COOP_CTAS_PER_SM = 2`（`:525`），注释："融合核在 8 warps 下约 80 regs/thread；4/SM 会 cooperative launch 失败"。
- `softmax(..., enable_pdl=...)`（`:134` 附近）——**注意**：该参数在 Triton 回退路径里被接收但**函数体未引用**（见 `06`/`15` 的"声称但未生效"清单）。
- 路径选择：flashinfer 装了就用 flashinfer 的融合核（2 次 launch）；否则 Triton 回退是 2 次 softmax + 1-2 次 draw。

### 3.4 量化 MoE 的三文件 + bf16 MoE

共同分工：**prefill 走 grouped GEMM，decode 走 per-route GEMV**。

| 文件 | decode | prefill |
|---|---|---|
| `nvfp4_fused_moe.py` | marlin 风格 int32 GEMV | `_prefill_nvfp4_moe_kernel` grouped |
| `mxfp4_moe.py` | N-innermost 的 split-K GEMV | `_t` grouped（转置布局） |
| `fp8_blockscale_moe.py` | `BLOCK_N=16`（"bs=1 路由少 → 小 BLOCK_N 提高 program 数"） | grouped，`BLOCK_SIZE_M=64 if M>64 else 16` |
| `fused_moe.py` + `decode_moe.py`（bf16） | per-route GEMV，固定 `BLOCK_N=32/BLOCK_K=64/num_warps=8` | `fused_experts_impl` grouped |

### 3.5 `triton/norm.py` —— "如何读一个 Triton 内核"的样板（193 行）

见 §7。

### 3.6 两个基础设施文件

- **`autotune_cache.py`（38 行）**：`_SUPPORTS_AUTOTUNE_CACHE`（`:24`）探测 triton 的 `autotune` 是否支持 `cache_results`；`FREETOKEN_TRITON_CACHE_RESULTS=0`（`:29`）关掉；导出 `autotune_cache_kwargs`（`:33`）。**解决的问题**：没有它，`@triton.autotune` 会在每个新进程、每个新 key 的首次 launch 重新扫网格——而那段恰好发生在 warmup / 图捕获里，"启动要几秒"。
- **`e4m3_compat.py`（143 行）**：强制 fp8-e4m3 仿真时，把 `TRITON_CACHE_DIR` 指到独立目录（`~/.triton/cache-e4m3emu`），避免仿真产物污染正常缓存。

---

## 4. `kernel/fla/` 详解（17 文件 / 5,968 行）

**它是 vendored 的，不是依赖**（`pyproject.toml` 里没有 fla）。来源在 `fla/__init__.py:1-38` 注明：GDN 路径来自 sglang 的 fla fork，KDA 来自 vLLM 的 `third_party/flash_linear_attention`。

### 4.1 谁在用

| 模型族 | 入口 |
|---|---|
| `qwen3_5_moe` / `qwen4_exp` 的 GDN 层 | prefill: `chunk_gated_delta_rule`（`fla/chunk.py`）；decode: `fused_sigmoid_gating_delta_rule_update`（`fla/fused_sigmoid_gating_recurrent.py`）——由 `models/qwen3_5_moe/gdn_kernels.py` 调用 |
| `glm5_next` 的 KDA 层 | `chunk_kda_with_fused_gate`（`fla/kda.py:1215`）、`fused_recurrent_kda`（`:145`）、`fused_kda_gate`（`:1344`）——由 `models/glm5_next/kda.py` 调用 |
| 三者共用 | gated RMSNorm `rms_norm_gated`（`fla/layernorm_gated.py`），经 `layers/norm.py` 调用 |

### 4.2 prefill / decode 的分流是 **phase flag，不是 token 数**
```python
if batch.is_decode:  fused_recurrent_*   # 逐 token 循环，保持状态
else:                chunk_*             # 分块（chunk）算法，并行度高
```

### 4.3 关键常量
`CHUNK_SIZE = 64` 出现在三处：`fla/chunk.py:28`、`fla/chunk_delta_h.py:23`、`fla/utils.py:284`（`FLA_CHUNK_SIZE`，KDA 从这读）。chunk 内部用 `BC=16` 子块，把"10 个下三角 [BC,BC] 块"在寄存器里算完。
**TF32** 用于 block-merge 的 16×16 小矩阵乘（"SM90 上安全且 ~2× 快"）。

### 4.4 两个必须知道的 landmine

1. **`chunk_delta_h.py:22-51` 故意只留一个固定 `triton.Config`**。注释原文：这个 kernel 把最终状态 `ht` **原地写回** `initial_state`；如果有多个配置，autotune 的基准测试阶段会多次调用它来计时，**破坏 cache pool**，导致"第一个用户请求静默给出错误输出"。想调只能通过环境变量（`SGLANG_GDN_CHUNK_H_BV` / `..._NUM_WARPS` / `..._NUM_STAGES`）。
2. **`chunk_kda_with_fused_gate` 会覆写它的 `v` 实参**（`fla/__init__.py:32-34`）：输出被写进那个缓冲以省一次分配，"**绝不要传一个之后还要读的张量**"。
3. 另有两条：KDA 的 per-token 状态写入读 `ssm_state_indices` 要求**连续 [N,T] 块**（"materialize，绝不要 `expand()`"，`:36-37`）；`attention/linear.py:107-110` 断言 `km1 <= CHUNK_SIZE`。

### 4.5 其他性能开关
`FLA_USE_CUDA_GRAPH=1`（`fla/utils.py:277`）让 FLA 走 CUDA graph 优化路径；`FLA_USE_TMA=1`（`:288-293`）在 Hopper+ 上启用 TMA descriptor（solve_tril 快路径）。

---

## 5. 为什么这里到处是"手写死的常量"而不是 `@triton.autotune`

因为 **decode 路径要被 CUDA graph 捕获**，而 autotune 会在运行期基准测试（在捕获期做非法操作）。文档写得很直白（`moe/fused_nvfp4.py:27-28`）：
> "Decode is captured into a CUDA graph, so the config must be fixed (no triton.autotune, which benchmarks at run time)."

所以：
- **离线扫出来的常量 + 注释里的实测百分比**取代 autotune；
- 需要 autotune 的地方只在 prefill / 非图路径（并且要 persistence，见 `autotune_cache.py`）；
- `chunk_delta_h.py` 那种"必须单配置"的情况更进一步：多配置会破坏**语义**，不只是性能。

---

## 6. 阈值与调参常量总表（实用速查）

| 常量 | 值 | 文件:行 | 控制什么 | 理由（注释） |
|---|---|---|---|---|
| `_MAX_KV_SPLITS` | 8 | `triton/attention.py:10` | decode split-KV 最大切分 | |
| `_MIN_BLOCK_KV` | 32 | `triton/attention.py:11` | 最小 KV 块 | |
| extend tile | 见 §3.1 | `triton/attention.py:21` | prefill tile | smem 自适应；大 tile ~2× |
| `_GEMM_MAX_INKERNEL_M` | 64 | `triton/nvfp4_linear.py:95` | 核内 GEMM 的 M 上限 | 超过则反量化重复代价 > cuBLAS |
| M==1 判据 | 1 | `triton/nvfp4_linear.py:768` | 专用 GEMV | |
| split-K 目标 | 1024 | `triton/nvfp4_linear.py:63` | program 数 | 波次量化 |
| `_COOP_CTAS_PER_SM` | 2 | `triton/sampling.py:525` | cooperative launch 预算 | 80 regs/thread，4/SM 失败 |
| `_GEMV_MAX_M` | 256 | `triton/mxfp8_linear.py:56` | W8A16 GEMV 上限 | 覆盖 CUDA graph 阶梯最大 256 |
| num_warps(norm) | 4 if M≥1024 else 8 | `triton/norm.py:37-38` | RMSNorm 并行度 | "sglang 固定 w8 在这里亏 ~12%" |
| `_MMVQ_SAFE` | 6 | `layers/gguf.py:40` | GGUF 向量/矩阵核 | |
| `MXFP4_DECODE_MAX_TOKENS` | 16 | `moe/fused_mxfp4.py:12` | decode 路径切换 | |
| `_GROUPED_MIN_ROUTES` | 768 | `moe/fused_ds_fp4.py:132` | GEMV vs grouped GEMM | |
| `_DECODE_MARLIN_DEEPK_THRESHOLD` | 2048 | `moe/fused_nvfp4.py:40` | 深 K 换窄 N tile | 实测 18.6 vs 21.0 μs |
| `CHUNK_SIZE` / `FLA_CHUNK_SIZE` | 64 | `fla/chunk.py:28`、`fla/utils.py:284` | 线性注意力分块 | |
| `BC` | 16 | `fla/chunk_fwd.py` | chunk 内子块 | 寄存器内算下三角 |
| `kSpinWindow` | 50 ms | `csrc/cpu_moe/cpu_moe_ext.cpp` | CPU worker 自旋窗口 | 跳过每层 futex |

---

## 7. 如何读一个 Triton 内核（以 `triton/norm.py` 为例）

面向"会 PyTorch、不会 Triton"的读者。**自底向上读**：先公开包装 → 再 launcher → 最后 kernel 体。

| 步 | 看什么 | 在 `norm.py` 的位置 | 对应 PyTorch 直觉 |
|---:|---|---|---|
| 1 | 公开 API | `:154-161` | 一个 `nn.Module` 的 forward 替代品 |
| 2 | launcher / 包装 | `:123-151` | 校验形状、`None`→`empty_like`、算 stride、选 `num_warps`/`num_stages`/PDL |
| 3 | 装饰器 | `:42-44` | `@triton.heuristics` 在 `@triton.jit` 之上：编译期填 `BLOCK` |
| 4 | `program_id` | `:53-56` | `tl.program_id(0/1)` = `blockIdx.x/.y`；一个 program 处理一个 tile |
| 5 | 索引与掩码 | `:53-56` | `tl.arange(0, BLOCK)` + `mask = cols < H` = "向量 + 边界掩码" |
| 6 | 签名里的 `constexpr` | `:46-51` | `tl.constexpr` 是编译期常量 → 编译器特化并**干掉掩码**；裸名字是运行期 stride |
| 7 | load / 规约 / store | `:65-66`、`:73-75` | `tl.load(ptr + off, mask=..., other=0.0)` = 带掩码的 strided 读；`tl.sum(x*x, axis=0)/H` = 规约；`tl.store(..., mask=...)` |
| 8 | grid | `:145` | Python 侧 `kernel[(A, B)]` = 每 (row, head) 一个 program |

**关键认知**：Triton 里没有"张量"，只有**指针 + 索引算术 + 掩码**。所谓"tile"就是你手动切出来的一块，`BLOCK` 是它的边长。

---

## 8. 与其他模块的关系

| 方向 | 模块 | 接口 | 说明 |
|---|---|---|---|
| 被调用 | `attention/*_sparse.py` | 稀疏注意力内核 | 只有 Triton 实现 |
| 被调用 | `layers/quantization/linear/*`、`moe/*` | 量化 linear / MoE 内核 | 候选之一 |
| 被调用 | `moe/fused*.py` | 专家 GEMM/GEMV | |
| 被调用 | `models/{qwen3_5_moe,qwen4_exp,glm5_next}` | fla 的 GDN/KDA | |
| 被调用 | `layers/norm.py` | `rms_norm_gated` | |
| 被调用 | `layers/embedding.py` 等 | rope/激活/norm | |
| 依赖 | `kernel/backend.py` | 决定是否走 flashinfer | 见 `11` |
| 依赖 | `triton` 编译器 | 编译、磁盘缓存 | |

---

## 9. 阅读顺序、可跳过项、常见坑

### 阅读顺序（≤8）
1. `triton/norm.py` —— 学会读 Triton（§7）。
2. `triton/attention.py:10-42` —— tile 选择与 split-KV 的权衡。
3. `triton/nvfp4_linear.py:53-116, 232-240, 523-543, 749-757` —— decode/prefill 阶梯 + 波次取舍。
4. `triton/sampling.py:520-540` —— cooperative launch 的 occupancy 预算。
5. `triton/autotune_cache.py` —— 为什么需要持久化。
6. `fla/__init__.py:1-48` —— 线性注意力的范围、入口与两条 landmine。
7. `fla/chunk_delta_h.py:22-51` —— "为什么只能单配置"。
8. `triton/dsv4/sparse_attn.py` —— 一个模型专属稀疏注意力的样板。

### 新手可以安全跳过
`triton/dsv4/`、`dsv41/`、`qsa/` 的全部实现（除你要改对应模型）；`triton/df11*.py`、`minimax_m3_sparse.py`、`glm_dsa_sparse.py`、`gemma4_fused.py`、`kpool_compress.py`、`ple.py`（都是单族专用）；`fla/` 除 `__init__.py` 之外的内部实现（vendored，作者自己标注为"脏"）；`fla/{cumsum,index,l2norm,layernorm_gated}.py` 里除被调用签名外的部分。

### 常见坑
1. **decode 路径不能用 `@triton.autotune`**（图捕获）。看到硬编码常量先问"是不是在图里"。
2. **`chunk_delta_h.py` 的多配置会破坏语义**（in-place state pool），不是性能问题。
3. **`chunk_kda_with_fused_gate` 覆写 `v`**：传给它的张量之后不能再读。
4. **PDL 只在 sm_90+ 编译**（pre-Hopper 的 ptxas 会拒绝该 intrinsic）。
5. **`enable_pdl` 在 Triton 采样回退里是空开关**（接收但不使用）。
6. **`FLA_USE_TMA` / `FLA_USE_CUDA_GRAPH` 默认关**，要显式开。
7. **`triton/` 是 namespace package**：没有 `__init__.py`，`from kernel.triton import x` 的写法与普通包不同。

---

## 10. 自测问题

1. 为什么 `kernel/triton/norm.py` 适合作为第一个读的 Triton 文件？
2. `_select_extend_tile` 依据什么选 tile？为什么"大 tile 更快"却不能无条件用大的？
3. NVFP4 Linear 的三档 dispatch 分别在什么 M 下触发？为什么 M>64 要回 cuBLAS？
4. split-K 为什么偏好"能整除 K-tile 数"的方案？
5. 为什么 decode 路径不能用 autotune？那为什么还要 `autotune_cache.py`？
6. `chunk_delta_h.py` 只能有一个 `triton.Config` 的原因是什么？多配置会怎样？
7. `chunk_kda_with_fused_gate` 的"覆写 v"约束从哪来、违反会怎样？
8. GDN 的 prefill 和 decode 分别调哪个 fla 入口？分流依据是 token 数还是 phase？
