# 10 · layers 与 quantization：层原语、量化方言、内核选择策略、MoE/Linear 方法

> 行号基于 `377a3bc`。完整符号清单见 [`99-symbol-index/layers.md`](99-symbol-index/layers.md)（36 文件 / 343 符号）。

---

## 1. 模块职责与在系统中的位置

`layers/` 提供模型族"拼网络"所需的**层原语**（并行 Linear、MoE、norm、rope、激活、embedding）；`layers/quantization/` 决定**每个 Linear / 每个 MoE 层实际跑哪条 kernel**。

```
checkpoint 的 quantization_config
   │  ① 方言识别（compressed_tensors / modelopt / fp8 / mxfp4）
   ▼
QuantConfig（含 NameMap）           ← models/register.py:checkpoint_quant_config 构造
   │  ② set_quant_config（进程全局）
   ▼
建层时：quant_method_for(prefix)     ← configs/__init__.py:13
   │  ③ scheme_for(prefix)           # 运行时名 → checkpoint 名 → QuantScheme
   ▼
method_class(kind, layer_kind)       ← registry.py:51
   │  ④ 该 kind 的方法类（LinearMethod / MoEMethod）
   ▼
select_kernel(candidates, requested) ← method.py:20
   │  ⑤ 显式名 or 按顺序挑
   ▼
具体 kernel（triton / marlin / b12x / flashinfer / ggml / cuBLAS）
```

**关键设计**：选择是**两级**的——
- **config 期**（建层时，只看 config）：选哪个 kernel；
- **run 期**（前向内，看 token 数）：同一 kernel 内部走 GEMV 还是 GEMM。
两者常被混淆，读代码时务必分清。

---

## 2. 文件清单

### 2.1 层原语

| 文件 | 行数 | 职责 |
|---|---:|---|
| `layers/__init__.py` | 63 | 导出清单（就是"层原语全集"） |
| `linear.py` | 182 | `LinearColParallelMerged` / `LinearRowParallel` / `LinearQKVMerged` / `LinearOProj` / `LinearReplicated` |
| `moe.py` | 511 | `MoELayer`（常驻）/ `OffloadMoELayer`（卸载）/ `_expert_gemm` 派发 / `make_moe_layer` |
| `rotary.py` | 356 | RoPE / YaRN / mRoPE（`get_rope`） |
| `norm.py` | 206 | RMSNorm / LayerNorm / gated RMSNorm（`rms_norm_gated` 走 fla） |
| `activation.py` | 95 | `silu_and_mul` / `gelu_and_mul` / `gelu_tanh_and_mul` / `swigluoai_and_mul` / `swiglu_clamp_and_mul` |
| `embedding.py` | 149 | `VocabParallelEmbedding` / `ParallelLMHead`（词表并行） |
| `mhc.py` | 254 | 超连接（hyper-connections）残差流 |
| `gguf.py` | 129 | GGML 量化 Linear 的按 token 数分派 |
| `base.py` | 100 | `BaseOP` 与公共工具 |

### 2.2 量化

| 文件 | 行数 | 职责 |
|---|---:|---|
| `quantization/scheme.py` | 108 | `QuantKind` / `WeightDesc` / `QuantScheme` / 各 kind 的 scheme 构造器 |
| `quantization/registry.py` | 60 | `LayerKind`、`register_dialect`、`register_method`、`method_class` |
| `quantization/method.py` | 90 | **`select_kernel`** + `QuantMethod` 基类 + `finalize_quant` |
| `quantization/quant_backend.py` | 81 | `--quant-backend` 的解析（`layer.kind=name`）与安装 |
| `quantization/names.py` | 99 | 张量角色命名辅助 |
| `quantization/configs/base.py` | 131 | `QuantConfig` / `claims` / `from_hf` / `NameMap` |
| `quantization/configs/{compressed_tensors,modelopt,fp8,mxfp4}.py` | 74/81/79/30 | 四种 checkpoint 方言 |
| `quantization/linear/base.py` | 65 | `LinearConfig` + `LinearMethod` 基类 |
| `quantization/linear/{nvfp4,fp8_tensor,fp8_block,mxfp8,unquantized}.py` | 125/85/71/49/34 | 各 kind 的 Linear 方法与候选 |
| `quantization/moe/base.py` | 211 | `MoEConfig` / `BankSpec` / `ExpertView` / `MoEKernel` / `MoEMethod` / `_common_reject` / `gated_epilogue_reason` |
| `quantization/moe/{nvfp4,mxfp4,fp8_block,mxfp8,unquantized}.py` | 570/114/69/22/42 | 各 kind 的 MoE 方法与候选 |

---

## 3. 核心概念

### 3.1 `QuantKind`（`scheme.py:14`）与 `LayerKind`（`registry.py:11`）

| `QuantKind` | 含义 |
|---|---|
| `NONE` | bf16/fp16 未量化 |
| `FP8_TENSOR` | per-tensor fp8 |
| `FP8_BLOCK` | 块级 scale（128×128 或 32×32） |
| `MXFP8` | MX 格式 8bit |
| `NVFP4` | NVIDIA FP4（e2m1 + e4m3 块 scale + 行 global） |
| `MXFP4` | MX 格式 4bit（e8m0 块 scale） |

`LayerKind` = `LINEAR` | `MOE`。`(QuantKind, LayerKind)` 决定用哪个方法类。

### 3.2 `QuantScheme`（`scheme.py:44`）
`roles` 是该方法**声明的规范张量名**（如 `weight`/`weight_scale`/`weight_global`）；"checkpoint 里的哪些张量喂这些 role"是**方言 Config** 的事。关键方法 `has(role)`（`:67`）。
构造器：`fp8_tensor_scheme`（`:79`）、`fp8_block_scheme`（`:84`）、`fp8_block_size`（`:91`，128 或 32）、`mxfp8_scheme`（`:97`）、`nvfp4_scheme`（`:101`）、`mxfp4_scheme`（`:106`）。

### 3.3 `QuantConfig` 与 `scheme_for(prefix)`（`configs/base.py`）
- `claims`：该配置声明"我负责哪些张量"；
- `from_hf`：从 HF `quantization_config` 构造；
- `scheme_for(prefix)`：**把运行时属性路径翻译成 checkpoint 名后再查方案**——翻译靠 `NameMap`（见 `07` §3.4）。

### 3.4 方法契约

| 类 | 行 | 契约 |
|---|---:|---|
| `QuantMethod` | `method.py:50` | `candidates`（候选 kernel 类，按优先级）、`scheme`、`finalize` |
| `LinearMethod`（`linear/base.py`） | — | `create_weights` / `apply` / `finalize` |
| `MoEKernel`（`moe/base.py:137`） | — | `unusable_reason`(148) / `worth_it`(151) / `slot_limit`(154) / `layout`(170) / `pack`(173) / `apply`(177) |
| `MoEMethod`（`moe/base.py:180`） | — | 声明常驻专家；`layout`/`pack`/`apply` 转给 kernel；`cpu_format`(196) 说明 CPU 执行器能否服务 |
| `MoEConfig`（`moe/base.py:16`） | — | 谓词输入：`num_experts`/`hidden`/`intermediate`/`top_k`/`tp_*`/`scheme`/`activation`/`alpha`/`beta`/`limit`/`interleaved`/`act_block`/`has_bias`/`strategy`/`decode_target`/`dtype` |
| `LinearConfig`（`linear/base.py:15`） | — | `(in_features, out_features, output_sizes, scheme)` |

### 3.5 四种并行 Linear（`layers/linear.py`）

| 类 | 切分 | 用途 |
|---|---|---|
| `LinearColParallelMerged` | 输出维（列） | qkv / gate_up 的融合投影 |
| `LinearRowParallel` | 输入维（行） | o_proj / down_proj |
| `LinearQKVMerged` | 输出维（按头切） | 融合 qkv |
| `LinearOProj` | 输入维（按头切） | 输出投影 |
| `LinearReplicated` | 不切 | 小张量 |

---

## 4. 量化格式 → 内核 全景表

| 格式 | `QuantKind` | Linear 内核（候选顺序） | MoE 内核（候选顺序） | 门槛 | bank 布局 |
|---|---|---|---|---|---|
| **NVFP4** | `NVFP4` | triton(W4A16, K-major) → marlin → emulation | triton → marlin → b12x | marlin: sm_80-99 + vllm；b12x: sm_120 + CUDA13 驱动 | 6 banks（含 global） |
| **MXFP4** | `MXFP4` | **无 Linear 方法** | triton(ds_fp4 W4A8) / triton_gptoss | ds_fp4 需 `act_block`；gptoss 需 `has_bias` | 6 banks |
| **MXFP8** | `MXFP8` | triton W8A16 GEMV(M≤256) / dequant+cuBLAS | 不支持（`NotImplementedError`） | | — |
| **FP8 per-tensor** | `FP8_TENSOR` | torch `_scaled_mm`(W8A8) → triton → emulation | 无 MoE 方法 | 需 `input_scale`、sm≥89、e4m3 native | — |
| **FP8 block** | `FP8_BLOCK` | dsv4(e8m0) → triton(128×128) | triton only | block 必须 128（triton 分支）；K/输出维必须整除 | 4 banks |
| **Q4_0 (GGUF)** | `NONE`（走 gguf 路径） | ggml MMVQ / MMQ | ggml `ggml_moe_a8_vec` | 需要 GGUF checkpoint | 2 banks |
| **bf16** | `NONE` | cuBLAS | `fused`（triton grouped） | — | 2 banks |

---

## 5. 函数详解

### 5.1 `method.py` —— 全仓唯一的选择策略

| 符号 | 行 | 作用 | 关键行为 |
|---|---:|---|---|
| `KernelSelectionError` | 16 | 选不出来/显式选错 | |
| **`select_kernel(candidates, requested, cfg)`** | 20 | **选择** | 见 §6.2 |
| `QuantMethod` | 50 | 方法基类 | `candidates` 是一个按优先级排的**类元组** |
| `__init__` | 55 | 构造即选择 | `self.kernel = select_kernel(...)` |
| `scheme()` | 61 | 归一化方案 | |
| `finalize_quant(root)` | 64 | 对子树每个层调 `finalize` | 返回处理数量 |

### 5.2 `quant_backend.py`

| 符号 | 行 | 作用 |
|---|---:|---|
| `_kernel_names()` | 13 | 从注册表收集所有合法 kernel 名 |
| `_parse_table(text)` | 22 | 解析 `linear=marlin,moe=b12x` |
| `QuantBackend.parse(text)` | 40 | 解析 + 校验（不认识的 kernel 名报错） |
| **`QuantBackend.select(layer_kind, quant_kind)`** | 58 | 精确条目 > 层宽条目（且该 kernel 在表里） > `"auto"` |
| `set_quant_backend` / `get_quant_backend` | 73 / 79 | 进程全局安装（引擎在建模型前调一次） |

### 5.3 `registry.py`

| 符号 | 行 | 作用 |
|---|---:|---|
| `register_dialect(name)` | 23 | 注册一个 checkpoint 方言（装饰器） |
| `dialects()` | 31 | 列出 |
| `register_method(kind, layer_kind)` | 35 | 把方法类注册到 `(kind, layer_kind)` |
| `method_class(kind, layer_kind)` | 51 | 查表 |
| `methods_for(layer_kind)` | 58 | 列出某层类型的所有方法 |

### 5.4 `moe/base.py` 的共用谓词

| 函数 | 行 | 作用 |
|---|---:|---|
| `_common_reject(cfg, tp_ok, cpu_ok, plain_silu_only)` | 158 | 统一拒绝：**TP>1**、`decode_target != "gpu"` 当 `cpu_ok=False`、**任何 `has_bias`**、非 plain-silu 当 `plain_silu_only` |
| `gated_epilogue_reason(cfg)` | 118 | 拒绝：没有对应 epilogue 的激活、`interleaved`、`beta != 0`、alpha/limit 用在 silu/gelu/gelu_tanh 上 |
| `fused_piece` / `global_rows` / `fused_global` / `limit_or_inf` | 90/98/107/113 | 张量拼接与缩放辅助 |
| `is_resident(cfg)` | 133 | 是否常驻策略 |

### 5.5 各格式的 `unusable_reason`（具体谓词，读内核选择时必查）

| 格式/内核 | 位置（范围） | 拒绝条件 |
|---|---|---|
| nvfp4/triton | `moe/nvfp4.py:43-48` | `_common_reject(cpu_ok=True, plain_silu_only=False)` + `gated_epilogue_reason` |
| nvfp4/marlin | `moe/nvfp4.py:230-241` | vllm 未安装；`_common_reject(cpu_ok=False, plain_silu_only=True)`；`_marlin_symbols_ok()` 探测 vllm 的 `_custom_ops`/`fused_marlin_moe`/marlin transforms；`worth_it` = `(8,0) <= cc < (10,0)`；`max_slots=992` |
| nvfp4/b12x | `moe/nvfp4.py:493-506` + `:342-374` | cc<sm_120；flashinfer 缺；**驱动 CUDA < 13000**；`worth_it = intermediate >= 1024`；注释明确"**never auto-selected**" |
| mxfp4/triton | `moe/mxfp4.py:22-29` | `interleaved`；`(alpha,beta) != (1.0,0.0)`；**`act_block is None`** |
| mxfp4/triton_gptoss | `moe/mxfp4.py:67-70` | 需要 `has_bias` |
| fp8_block/moe | `moe/fp8_block.py:23-34` | `_common_reject(cpu_ok=False)`；block 必须 128；`gated_epilogue_reason`；拒绝 `apply_router_weight_on_input` |
| bf16/fused | `moe/unquantized.py:16-18` | 只有 `gated_epilogue_reason` |
| fp8_tensor/torch | `linear/fp8_tensor.py:23-33` | 需 `input_scale`；cc ≥ (8,9)；`e4m3_native()` |
| fp8_block/triton | `linear/fp8_block.py:45-50` | 拒绝 e8m0；block ≠ 128 |
| nvfp4/marlin（Linear） | `linear/nvfp4.py:40-55` | vllm 缺；`len(output_sizes) > 1`（融合投影每部分一个 global）；符号不可导入；`is_fp4_marlin_supported()` 假 |

### 5.6 `layers/moe.py`（MoE 层本体）

| 符号 | 行 | 作用 |
|---|---:|---|
| `MoELayer` | 29 | 常驻专家路径 |
| `OffloadMoELayer` | 148 | 卸载路径（持 `OffloadMoeCache`） |
| `_decode_routed` | 261 | **decode 主路径**：cpu-layer 分流 → hybrid 分流 → `ensure_experts` → `copy_missing` → `_expert_gemm` |
| `_decode_hybrid` | 298 | hybrid 的 CPU/GPU 并行 |
| `_prefill_routed` | 352 | prefill：overlap 走双缓冲，否则 `materialize_layer` + `copy_missing` |
| `_wait_prefill_overlap` | 391 | 预取 layer_id 与 layer_id+1，等 layer_id 就绪 |
| `_expert_gemm` | 408 | **派发点**：先量化方法，失败/不适用则 GGUF q4_0 回退 |
| `make_moe_layer` | 456 | 工厂：按 `strategy` 返回 `MoELayer` 或 `OffloadMoELayer` |

### 5.7 `layers/gguf.py`（最清晰的阈值示例）

| 符号 | 行 | 作用 |
|---|---:|---|
| `_MMVQ_SAFE = 6` | 40 | ≤6 token 走 MMVQ（向量核），否则 MMQ（矩阵核） |
| `fused_mul_mat_gguf(x, qweight, type)` | 43 | 三级分派：未量化 → `x @ w.T`；≤6 且类型支持 → `ggml_mul_mat_vec_a8`；支持 MMQ → `ggml_mul_mat_a8`；否则 dequant + GEMM |
| `GGUFLinear` | 69 | 权重的 native GGUF 块量化 Linear |

---

## 6. 关键流程逐段解析

### 6.1 启动期：量化绑定全链
```
checkpoint quantization_config
  → models/register.py:checkpoint_quant_config   # 用 NameMap 翻译名字，构造 QuantConfig
  → engine/config.py:141-142 set_quant_config(quant)   # 进程全局
  → 建层（models/<family> 的 __init__）
      → quant_method_for(prefix)        # configs/__init__.py:13
            → scheme_for(prefix)        # 运行时名 → checkpoint 名 → QuantScheme
            → method_class(kind, layer_kind)   # registry.py:51
      → 方法类构造 → QuantMethod.__init__ → select_kernel
```

### 6.2 `select_kernel` 的三分支（`method.py:20-47`）
```
if requested != "auto":
     实例化该 kernel
     if unusable_reason(cfg) 非空:  raise KernelSelectionError      # 显式指定绝不静默降级
     return it
else:
     for k in candidates:
         if unusable_reason(cfg): skipped.append(k); continue
         if worth_it(cfg): return k                    # ① 优先"值得"的
     return first_usable or raise                      # ② 否则取第一个可用的
```
**两个必须记住的推论**：
1. **`worth_it` 只是"可用内核之间的偏好"，不是门槛**。
2. NVFP4 MoE 的候选顺序是 `(triton, marlin, b12x)`（`moe/nvfp4.py:569`），而 triton 的闸门最松且 `worth_it=True` → **`auto` 永远选 triton；marlin/b12x 只能显式 `--quant-backend moe.nvfp4=marlin|b12x`**。`moe/nvfp4.py:505` 的注释也承认"b12x stays behind triton in the table"。

### 6.3 `--quant-backend` 的优先级
```
--quant-backend "moe.nvfp4=b12x"        # 只覆盖 MoE 的 nvfp4
--quant-backend "moe=b12x"              # 层宽：所有 kind 中"表里列了 b12x"的才生效
QuantBackend.select(LayerKind.MOE, QuantKind.NVFP4)
  → 精确条目 > 层宽条目（且该 kernel 在该 kind 的表里）> "auto"
FTW checkpoint：packed expert kernel 会被 `_adjust_ftw_quant_backend` 补上（engine.py:1453），
                与 flag 冲突时以 FTW 打包的为准
```

### 6.4 MoE 层的 forward 分流
```
make_moe_layer(strategy):
  fused   → MoELayer（专家常驻）
  offload → OffloadMoELayer
  cpu     → OffloadMoELayer 但 decode_target=cpu（`is_cpu_layer` 全层命中）
  hybrid  → OffloadMoELayer 且 decode_target=hybrid
OffloadMoELayer._decode_routed（:261）:
  is_cpu_layer → CpuMoeExecutor.decode        # ① CPU 层直接走执行器，不碰 ensure/copy
  decode_target == hybrid → _decode_hybrid    # ② hybrid：CPU/GPU 并行
  else → ensure_experts → copy_missing → _expert_gemm   # ③ 标准 offload
_expert_gemm（:408）:
  量化方法可用 → method.apply(ExpertView(...))
  否则 → GGUF q4_0 路径
```

### 6.5 decode/prefill 阈值清单（"为什么我的 decode 慢"）
| 阈值 | 值 | 位置 | 意义 |
|---|---|---|---|
| GGUF `_MMVQ_SAFE` | 6 | `layers/gguf.py:40` | ≤6 token 用向量核 |
| NVFP4 Linear | M==1 / M≤64 / M>64 | `kernel/triton/nvfp4_linear.py` | GEMV / 核内 dot GEMM / dequant+cuBLAS |
| NVFP4 decode 深 K | K>2048 | `moe/fused_nvfp4.py:40-45` | 换窄 N tile |
| MXFP8 Linear | `_GEMV_MAX_M = 256` | `kernel/triton/mxfp8_linear.py:56` | 覆盖 graph 阶梯上限 |
| FP8 block Linear | M==1 / M≥64 | `kernel/triton/fp8_block_linear.py` | W8A16 GEMV / W8A8 块 GEMM |
| FP8 per-tensor | **与 M 无关** | `kernel/triton/fp8_pertensor_linear.py:325-330` | 激活是否量化是部署属性 |
| ds_fp4 MoE | routes ≥ 768 | `moe/fused_ds_fp4.py:132-137` | per-route GEMV vs grouped GEMM |
| gpt-oss MXFP4 | ≤16 token | `moe/fused_mxfp4.py:12-16` | decode 路径切换 |
| bf16 fused MoE | `is_prefill` | `moe/unquantized.py:31-36` | grouped GEMM vs per-route GEMV |

---

## 7. 与其他模块的关系

| 方向 | 模块 | 接口 | 说明 |
|---|---|---|---|
| 被使用 | `models/<family>` | `make_moe_layer`、并行 Linear、norm、rope、激活、embedding | 拼网络的原材料 |
| 被使用 | `engine/config.py` | `set_quant_config` | 启动期注入 |
| 被使用 | `engine/engine.py:1510` | `set_quant_backend`、`_adjust_ftw_quant_backend` | |
| 使用 | `moe/` | `OffloadMoECache`、`ExpertBanks`、`ExpertView` | 专家权重与卸载 |
| 使用 | `layers/quantization/moe/*` | bank 布局（`BankSpec`） | 布局由内核拥有 |
| 使用 | `kernel/*` | 所有算子实现 | |
| 使用 | `checkpoint/ftw` | packed bank 的 kernel 决定 | 影响 `--quant-backend` |

---

## 8. 常见坑

1. **谓词只在建层时求值一次、只看 config**——不看 batch size。所以"某个核在小 batch 下是否可用"不是在这里决定的。
2. **`worth_it` 不是门槛**，只是偏好；`unusable_reason` 才是门槛。
3. **`auto` 下 NVFP4 永远选 triton**（marlin/b12x 要显式指定）。
4. **decode 路径不能有 `triton.autotune`**：图捕获期做基准测试是非法的 → 所以到处是手写死的 tile 常量。
5. **tuned config JSON 只 ship 了 H100 + triton 3.5.1 一份**：其它设备全走 `get_default_config`（`moe/fused.py:120-141`）。
6. **`--quant-backend` 与 FTW 冲突时以 FTW 为准**（`engine.py:1453-1476`）。
7. **FP8 per-tensor 的 W8A8/W8A16 选择与 M 无关**——别按 batch 大小猜。
8. **`layers/quantization/configs/` 只管"哪些张量喂哪些 role"**，不管选哪个 kernel；唯一例外是 `configs/fp8.py` 会设置 `act_block`，而它反过来是 ds_fp4 能否使用的前提。

---

## 9. 自测问题

1. `(QuantKind, LayerKind)` 到方法类、再到 kernel，中间经过哪几步？
2. `select_kernel` 里"显式指定"和"auto"的语义差别是什么？为什么显式指定不可用要报错？
3. 为什么 `auto` 永远不会选 marlin/b12x？想用要怎么写命令？
4. `_common_reject` 的 `cpu_ok` 参数在什么情况下会导致某个 kernel 被排除？
5. 为什么 decode 路径上到处是硬编码 tile，而不是 `triton.autotune`？
6. `scheme_for(prefix)` 里的 `prefix` 是什么？为什么需要 `NameMap`？
7. FP8 per-tensor 为什么"激活是否量化与 batch 无关"？
8. 一个 MoE 层 forward 时，`is_cpu_layer` / `decode_target == hybrid` / 其它 三种情况分别走什么路径？
