# 08 · models（模型族）：19 个架构的加载、前向与特殊算子

> 行号基于 `377a3bc`。完整符号清单见 [`99-symbol-index/models.md`](99-symbol-index/models.md)（149 文件 / 1,178 符号）。
> 本文件覆盖 19 个族子目录；顶层的注册表/加载机制见 [`07-models-core.md`](07-models-core.md)。

---

## 1. 模块职责与在系统中的位置

每个模型族只做四件事：**解析配置、拼 decoder、接注意力后端、声明权重名映射**。其余全部上收到 `models/` 顶层机制（`07`）。

```
models/register.py 的注册项（arch → module/class + 钩子名 + 名称映射 + encoders）
        │
        ▼
models/<family>/config.py   parse_config(HF config) ──▶ ModelConfig
        │                                          （含 attention_groups 声明）
        ▼
models/<family>/model.py    DecoderLayer / Model / ForCausalLM
        ├── attention.py    → ctx.attn_backend.forward(...)
        ├── moe.py          → make_moe_layer(...)（仅 MoE 族）
        ├── vision.py       → encode(item) / place_encoder_weights(mode)（仅多模态族）
        ├── 专属算子         → gdn.py / kda.py / hc.py / ple*.py / compress.py / ...
        └── weight.py       iter_weights(...)  ──▶ 被 models/loader.py 分片/融合/堆叠
```

**复杂度差异的来源**：注意力类型（FULL/SWA/GDN/KDA/DSA/DSV4/QSA/BSA）、是否 MoE、量化格式、是否多模态、有没有"表外"结构（Engram/PLE/超连接）。

---

## 2. 每族的文件结构约定

| 文件 | 何时存在 | 职责 |
|---|---|---|
| `__init__.py` | 总是 | re-export 模型类 |
| `config.py` | 总是 | `parse_config(hf_config) -> ModelConfig` |
| `model.py` | 总是 | `DecoderLayer` + `Model` + `ForCausalLM` |
| `attention.py` | 总是（少数把 attention 内联进 model.py） | 组 q/k/v 并调后端 |
| `weight.py` | 总是 | `iter_weights(...)`：yield 运行时名与张量 |
| `moe.py` | MoE 族 | 专家块与路由 |
| `vision.py` | 多模态族 | tower + `encode` + `place_encoder_weights` |
| `args.py` | DSV4/DSV4.1/GLM/`minimax_m3` 等 | 该族**特有**的配置参数解析（不是 CLI args） |
| `gdn.py` / `gdn_kernels.py` / `kda.py` | 线性注意力族 | 线性注意力的 prefill/decode 实现 |
| `hc.py` | qwen4_exp | 超连接残差流 |
| `ple.py` / `ple_disk.py` | qwen4_exp | PLE n-gram 表（常驻 / 磁盘） |
| `engram*.py` | deepseek_v41 | Engram 磁盘表与主机侧喂数 |
| `compress.py` / `compressor.py` / `indexer.py` | DSV4 系 | 压缩 KV 与稀疏索引器 |
| `gguf.py` | gemma4 / gguf | GGUF 变体的读取与前向 |
| `df11_*.py` | glm4_moe | 该族的特殊 embedding/linear |
| `rope.py` | deepseek_v41 | 该族的 rope 实现 |
| `gdn_reference.py` | qwen 系 | 测试用的参考实现 |

---

## 3. 族 × 特性矩阵（19 行全覆盖）

| 族 | arch 名（注册表） | 代码量 | 注意力类型 | MoE | 量化格式 | 多模态 | 特殊结构 |
|---|---|---:|---|---|---|---|---|
| `llama` | `LlamaForCausalLM` | 290 行 / 5 文件 | FULL | — | bf16 | — | **最简样板** |
| `qwen2` | `Qwen2ForCausalLM` | 288 / 5 | FULL | — | bf16 | — | llama + qkv bias |
| `qwen3` | `Qwen3ForCausalLM` | 296 / 5 | FULL | — | bf16/fp8 | — | llama + qk_norm |
| `qwen3_moe` | `Qwen3MoeForCausalLM` | 402 / 6 | FULL | ✓ | bf16 | — | 标准 MoE |
| `qwen3_5_moe` | `Qwen3_5MoeFor{,ConditionalGeneration}CausalLM`、`Qwen3_5For{CausalLM,ConditionalGeneration}` | 1,359 / 9 | **GDN + FULL**（混合） | ✓ | NVFP4 / FP8 | ✓（VL 变体） | 层类型交替；`gdn.py`；PLE 表（部分） |
| `qwen3_vl` | `Qwen3VLFor{,Moe}ConditionalGeneration` | 705 / 5 | FULL | 视变体 | bf16 | ✓ ViT | DeepStack 视觉 |
| `qwen4_exp` | `Qwen4ExpForConditionalGeneration` | 2,854 / 11 | **GDN + QSA**（36 GDN + 12 QSA） | ✓（512 专家 top-10 + shared） | NVFP4 | ✓ | **超连接 4 条残差流** + PLE n-gram 表 |
| `deepseek_v4` | `DeepseekV4ForCausalLM` | 1,816 / 10 | **DSV4**（window + compressed + indexer） | ✓ | FP4 专家 + fp8 dense | — | Manifold 超连接；无 `model.` 前缀 |
| `deepseek_v41` | `DeepseekV41ForCausalLM` | 2,273 / 13 | **DSV41** | ✓ | fp4 专家 / fp8 block-32 dense | ✓ ViT | **Engram 磁盘表（98 GB/层）**；`inference/config.json` |
| `glm4_moe` | `Glm4MoeForCausalLM` | 735 / 9 | FULL | ✓ | NVFP4（GLM-4.7） | — | 前 `first_k_dense_replace` 层稠密；shared expert |
| `glm_moe_dsa` | `GlmMoeDsaForCausalLM` | 796 / 8 | **DSA**（MLA + 稀疏索引器） | ✓ | NVFP4 | — | IndexShare（full 层拥有 indexer 槽） |
| `glm5_next` | `Glm5NextFor{,ConditionalGeneration}CausalLM` | 1,693 / 10 | **KDA**（线性注意力） | ✓ | NVFP4 | ✓ ViT | KDA 分块/循环；kpool 变体 |
| `gpt_oss` | `GptOssForCausalLM` | 746 / 6 | FULL | ✓ | **MXFP4**（transposed split-K） | — | 每专家 bias；harmony 通道 |
| `gemma4` | `Gemma4{,Unified}For{,ConditionalGeneration}CausalLM`、`Gemma4GGUFForCausalLM` | 1,681 / 8 | **SWA 混合**（window×3 + full） | ✓ | NVFP4 / **GGUF Q4_0** | ✓（含 unified 线性 embedder） | soft-token 预算 70~1120；post-norm eps 不同 |
| `mistral` | `Mistral{For,Mistral3For}ForCausalLM` | 291 / 5 | FULL + **SWA** | — | bf16 | （Mistral3 声明） | |
| `minimax_m2` | `MiniMaxM2ForCausalLM` | 398 / 6 | FULL | ✓ | NVFP4 | — | |
| `minimax_m3` | `MiniMaxM3SparseFor{,ConditionalGeneration}CausalLM` | 1,355 / 9 | **BSA**（GQA block-sparse） | ✓ | NVFP4 | ✓ CLIP ViT | 每稀疏层一块 index-key slab |
| `muse_glimmer` | `MuseGlimmerForConditionalGeneration` | 860 / 6 | FULL + **SWA×3/full**（full 层 NoPE） | — | NVFP4 W4A16 | ✓ 窗口 ViT | 无权重 qk norm、centered sandwich norm、softcap logits |
| `gguf` | （辅助，非族） | 539 / 5 | — | — | GGUF 读取/反量化/tokenizer | — | `GGUF_ARCH_TO_REGISTRY` shim |

> 注：`gguf/` 是**辅助目录**（reader/dequant/tokenizer），真正的 GGUF 前向在 `gemma4/gguf.py`；它通过伪造 `architectures` 让同一套 dispatch 也能服务 GGUF（`gguf/config.py:19,61-93`）。

---

## 4. 四个代表族深度解析

### 4.1 `llama`（290 行）——最简契约

| 文件 | 关键符号 | 行 | 作用 |
|---|---|---:|---|
| `config.py` | `parse_config` | 8 | HF config → `ModelConfig`（维度/头数/rope） |
| `model.py` | `LlamaDecoderLayer` | 18 | 一个 decoder 层 |
| | `LlamaModel` | 46 | embedding + N 层 + final norm |
| | `LlamaForCausalLM` | 71 | `BaseLLMModel`：`forward()` → logits |
| `attention.py` | `LlamaAttention` | 17 | 组 q/k/v → `ctx.attn_backend.forward` |
| `weight.py` | `iter_weights` | 23 | yield 运行时名 + 张量（q/k/v/o/gate/up/down/embed/lm_head） |

**一个 decoder layer 的 forward 顺序**（`model.py:18` 起）：
`input_layernorm → attention → residual → post_attention_layernorm → mlp → residual`。
`qwen2` = llama + qkv bias；`qwen3` = llama + `use_qk_norm=True`；`mistral` = llama + SWA。**所以读懂 llama 就等于读懂了 4 个族。**

### 4.2 `qwen3_5_moe`（1,359 行）——MoE + GDN 线性注意力 + 多模态

| 文件 | 作用 |
|---|---|
| `config.py` | 解析混合层表：哪些层是 GDN（`linear_attention` 组）、哪些是 full |
| `gdn.py` | `Qwen3_5GatedDeltaNet`（`:20`）+ `_DepthwiseConv1d`（`:13`）——**prefill 走 chunked、decode 走 recurrent** |
| `gdn_kernels.py` | 调 `kernel/fla/` 的入口（`chunk_gated_delta_rule` / `fused_sigmoid_gating_delta_rule_update`） |
| `moe.py` | MoE 块 → `make_moe_layer` |
| `attention.py` | full 层 → 后端 |
| `weight.py` | 融合投影名（`in_proj_qkvz` / `in_proj_ba` 等，见注册表的 `packed_modules_mapping`） |

**读它的要点**：层类型由 config 的**组声明**决定；GDN 层不走 `attn_backend`（`AttnType.LINEAR.backend_driven=False`），而是直接调 fla 算子并读 `batch.fla_metadata`。

### 4.3 `deepseek_v41`（2,273 行）——最复杂的一族

| 文件 | 作用 |
|---|---|
| `args.py`（226 行） | 从 checkpoint 的 `inference/config.json` 读**权威**模型参数（不是 HF config） |
| `config.py` | 组装 `ModelConfig` + `DSV4AttentionGroupConfig` |
| `attention.py` / `compressor.py` / `indexer.py` | DSV41 稀疏注意力：window + 跨层共享压缩 KV + indexer |
| `engram.py` / `engram_table.py` | **Engram 磁盘表**：每层 `[384M,256]` fp8（98 GB）+ `[384M,8]` scale（3 GB） |
| `vision.py` | ViT tower（host 流式权重） |
| `weight.py` | `iter_weights` + `ftw_side_files`（FTW 转换时把表所在分片拷到旁边） |
| `model.py` / `moe.py` / `rope.py` | 前向 / MoE / rope |

**`engram_table.py:1-20` 的 docstring 是全仓最值得读的一段性能设计**：表留在 checkpoint 原处、io_uring+O_DIRECT 按 token 读 24 行、值在 pinned staging 里拼成池行格式、CUDA graph 下用"宿主读回采样 token → 填 staging → 释放 flag → 图内 lookup WAIT"的协议，且**填充线程必须在 graph launch 之前启动**（否则 driver 的 launch 会阻塞在 WAIT 上死锁）。

### 4.4 `qwen4_exp`（2,854 行）——GDN + QSA + 超连接 + PLE

| 文件 | 作用 |
|---|---|
| `gdn.py` / `gdn_reference.py` | GDN 线性注意力（+ 测试用参考实现） |
| `attention.py` | QSA 稀疏注意力（压缩块 + pending ring） |
| `hc.py` | **超连接**：`grouped_plus_one_rms_norm`（`:29`）、`GatedResidual`（`:59`）——4 条并行残差流 |
| `ple.py` | PLE n-gram 表（常驻 pinned） |
| `ple_disk.py` | PLE 磁盘版：`PleRowSource`（`:43`）、`DiskRowTable`（`:103`），与 Engram 同一套 `forward_host_ctx` 协议 |
| `moe.py` / `model.py` / `weight.py` | MoE / 前向 / 权重 |

**读它的要点**：36 层 GDN + 12 层 QSA 交替；残差不是单流而是 4 条（`hc.py`）；PLE 表 47.7 GiB 常驻主机内存。

---

## 5. 其余 15 族的逐个说明

**`qwen2`（288/5）**：llama + qkv bias（`has_attn_bias`）。读法同 llama。

**`qwen3`（296/5）**：llama + `use_qk_norm=True`。与 llama 的 diff 通常只有几行。

**`qwen3_moe`（402/6）**：标准 MoE——`moe.py` 里 router + experts → `make_moe_layer`；`weight.py` 用 `_EXPERTS_PACKED` 探测专家容器。

**`qwen3_vl`（705/5）**：FULL 注意力 + **ViT tower**（`vision.py`）+ DeepStack 视觉注入；`config.py` 同时解析 `vision_config` 与 `text_config`。

**`deepseek_v4`（1,816/10）**：DSV4 稀疏（window + compressed + Lightning Indexer）；`compress.py`(480) 是最大的单文件；**checkpoint 没有 `model.` 前缀**（`checkpoint_roots` 特判）；Manifold 约束超连接。

**`glm4_moe`（735/9）**：DeepSeek 风格 MoE（`first_k_dense_replace=3`、1 个 shared expert、`routed_scaling_factor=2.5`、分组路由）；`df11_embedding.py`/`df11_linear.py` 是该族的特殊层。

**`glm_moe_dsa`（796/8）**：**DSA**（MLA latent KV + 稀疏索引器）；`attention.py` 接 `dsa` 后端；`args.py` 解析该族特有参数。IndexShare：full 层拥有 indexer 槽，follower 复用。

**`glm5_next`（1,693/10）**：**KDA**（Kimi Delta Attention，线性注意力）+ MoE + ViT；`kda.py` 分 chunked/recurrent 两路；`args.py` 特有参数；属于"GLM-5.3-Flash"。

**`gpt_oss`（746/6）**：FULL + **MXFP4 专家**（transposed split-K 布局）；每专家有 bias；`moe.py` 接 `triton_gptoss` kernel；harmony 通道格式（见 `04` 的 `GptOssDetector`）。

**`gemma4`（1,681/8）**：SWA 混合（window×3 + full，full 层 NoPE）+ MoE + **GGUF 变体**（`gguf.py` 475 行，含 Q4_0 专家 bank）；`gemma4_unified` 用线性 patch embedder 代替 ViT；soft-token 预算 70/140/280/560/1120；post-norm eps 与 pre-norm 不同（`post_norm_eps`）。

**`mistral`（291/5）**：dense + SWA。`Mistral3ForConditionalGeneration` 也映射到这个 module。

**`minimax_m2`（398/6）**：dense/MoE 的较早版本；结构接近 qwen3_moe。

**`minimax_m3`（1,355/9）**：**BSA**（GQA block-sparse，每稀疏层一块 index-key slab）+ MoE + CLIP 风格 ViT；`args.py` 解析该族参数；`_compression()` 读嵌套的 `img_token_compression_config`。

**`muse_glimmer`（860/6）**：FULL + SWA×3/full 模式；无权重 qk norm、centered `(1+w)` sandwich norm、softcapped logits、`output_multiplier`；窗口 ViT。

**`gguf`（539/5）**：辅助目录——`reader.py` 读 GGUF、`dequant.py` 反量化、`tokenizer.py`、`config.py`（`GGUF_ARCH_TO_REGISTRY` + 伪造 `architectures` 的 shim）。

---

## 6. 跨族共同模式（读任何新族都能套）

| 模式 | 说明 |
|---|---|
| **`parse_config` 三段式** | ① 读 HF config（或 `inference/config.json`）→ ② 填 `ModelConfig` 基础字段 → ③ 填 `attention_groups` + `RotaryConfig` + 量化标记 |
| **`iter_weights` 骨架** | 遍历 `layers.i.*`，对每个张量 yield `(运行时名, tensor)`；融合张量交给 `packed_modules_mapping` 在 `loader.py` 里拼 |
| **decoder layer 的残留结构** | `norm → attn → residual → norm → mlp/moe → residual`；`layer_id` 一路传到 `attn_backend.forward` |
| **接 MoE** | `moe.py` 里 `make_moe_layer(...)`，不要自己实现路由/专家 |
| **接注意力** | 只负责组 q/k/v 与传 `attn_spec`；寻址与内核在后端里 |
| **多模态协议** | 实现 `encode(item)` 与 `place_encoder_weights(mode)`；`config.py` 填 `vision_config`/`image_token_id` |
| **量化张量命名** | 必须与注册表的 `packed_modules_mapping`/`unquantized_modules` 对齐，否则量化查找会错位 |
| **rotary** | 用 `layers/rotary.py` 的 `get_rope`，只填 `RotaryConfig`（含 mrope 布局） |

---

## 7. 族 → 关键符号索引（每族 4-8 个）

| 族 | 关键符号（行） |
|---|---|
| llama | `parse_config`(config.py:8)、`LlamaDecoderLayer`(model.py:18)、`LlamaModel`(:46)、`LlamaForCausalLM`(:71)、`LlamaAttention`(attention.py:17)、`iter_weights`(weight.py:23) |
| qwen2 / qwen3 / mistral | 同名样板（config/model/attention/weight） |
| qwen3_moe | `parse_config`、`Qwen3MoeForCausalLM`、`Qwen3MoeSparseMoeBlock`(moe.py)、`iter_weights` |
| qwen3_5_moe | `Qwen3_5GatedDeltaNet`(gdn.py:20)、`_DepthwiseConv1d`(gdn.py:13)、`gdn_kernels.py` 的 fla 入口、`moe.py`、`iter_weights` |
| qwen3_vl | `QwenVLVisionModel`(vision.py:216)、`QwenVLVisionMixin`(:363)、`encode`(:371)、`parse_config` |
| qwen4_exp | `GroupedPlusOneRMSNorm`(hc.py:38)、`GatedResidual`(hc.py:59)、`PleRowSource`(ple_disk.py:43)、`DiskRowTable`(ple_disk.py:103)、`gdn.py`、`attention.py`(QSA) |
| deepseek_v4 | `DeepseekV4ForCausalLM`、`compress.py` 的压缩器、`ops.py`、`attention.py`、`moe.py`、`args.py` |
| deepseek_v41 | `EngramRowSource`(engram_table.py:68)、`EngramDiskTable`(:95)、`EngramHost`(:193)、`forward_host_ctx`(:283)、`compressor.py`、`indexer.py`、`vision.py`、`weight.py:ftw_side_files` |
| glm4_moe | `parse_config`、`df11_embedding.py`、`df11_linear.py`、`moe.py`、`mlp.py` |
| glm_moe_dsa | `attention.py`（DSABackend 接线）、`args.py`、`moe.py` |
| glm5_next | `kda.py`（chunked/recurrent）、`args.py`、`attention.py`、`vision.py`、`moe.py` |
| gpt_oss | `moe.py`（MXFP4 + bias）、`attention.py`、`model.py` |
| gemma4 | `parse_config`(config.py)、`Gemma4MMProcessor` 相关（vision.py 274 行）、`gguf.py`(475)、`moe.py`、`weight.py` |
| minimax_m2 | `model.py`、`moe.py`、`attention.py` |
| minimax_m3 | `attention.py`（BSA）、`args.py`、`vision.py`、`moe.py`、`mlp.py` |
| muse_glimmer | `attention.py`（SWA 模式）、`vision.py`、`model.py`（sandwich norm/softcap） |
| gguf | `GGUF_ARCH_TO_REGISTRY`(config.py:19)、`reader.py`、`dequant.py`、`tokenizer.py` |

---

## 8. 与其他模块的关系

| 族依赖 | 用到的接口 |
|---|---|
| `layers` | 并行 Linear、norm、rope、激活、embedding、`make_moe_layer` |
| `attention` | `ctx.attn_backend.forward(q,k,v,layer_id,batch,spec)` |
| `kvcache` | 通过 `ModelConfig.kv_cache_group_specs()` 间接声明池族 |
| `moe` | 专家 bank 与 offload cache（经 `layers.moe`） |
| `kernel/fla` | GDN（qwen3_5_moe / qwen4_exp）与 KDA（glm5_next） |
| `kernel/row_store` | Engram（deepseek_v41）/ PLE（qwen4_exp）的磁盘表 |
| `mm` | `mm_processor` 注册 + `encoders` 声明 |
| `layers/quantization` | 建层时由 `quant_method_for` 决定方法 |
| `models/` 顶层 | 注册表、`ModelConfig` 基类、`loader.py` 的分片/融合/堆叠 |

---

## 9. 加一个新模型族的 8 步清单

1. 建目录 `models/<family>/`，至少 `__init__.py` / `config.py` / `model.py` / `attention.py` / `weight.py`。
2. `config.py`：写 `parse_config(hf_config) -> ModelConfig`，填满基础字段 + `attention_groups` + `RotaryConfig`。
3. `model.py`：`DecoderLayer`（照抄 llama 的残留结构）+ `Model` + `ForCausalLM(BaseLLMModel)`。
4. `attention.py`：组 q/k/v，调 `ctx.attn_backend.forward`，传 `layer_id` 与 `attn_spec`。
5. `weight.py`：`iter_weights` yield **运行时名**（不是 checkpoint 名）。
6. `register.py`：加 `_MODEL_REGISTRY` 条目（arch → module/class + 名称映射 + encoders）。
7. 若多模态：加 `vision.py`（`encode` + `place_encoder_weights`）与 `mm_processor`。
8. 若 MoE：加 `moe.py` 调 `make_moe_layer`；确认 `packed_modules_mapping` 覆盖专家张量名。

**验证**：`tests/models/test_models_registry.py` 会遍历活注册表检查"注册项声明的类/函数确实存在"；跑 `pytest tests/models/` 能立刻抓到拼错的名字。

---

## 10. 常见坑

1. **`checkpoint_roots` 差异**：DSV4 无 `model.` 前缀；多模态包装在 `model.language_model.`。写错会导致权重名全对不上。
2. **`num_moe_layers != num_layers`**：`first_k_dense_replace` 让前几层稠密。
3. **`qwen3` 不是独立实现**，是 llama + 一个 flag——别把时间花在找差异上。
4. **GDN/KDA 层不走 `attn_backend`**：它们直接调 fla 算子，只借 `batch.fla_metadata`。
5. **同一族可有多个 arch 名**（如 gemma4 有 6 个、qwen3_5 有 4 个），反向"arch → 族"不是一对一。
6. **GGUF 走假 architectures shim**：`gemma4/gguf.py` 才是真正的 GGUF 前向。
7. **量化张量名必须与 `packed_modules_mapping` 对齐**，否则量化查找会落到错误的方案上（静默错误）。
8. **超连接（qwen4_exp）改变了残差流的数量**，照抄 llama 的 residual 写法会错。

---

## 11. 自测问题

1. 一个模型族最少要实现哪几个钩子？各自的输入输出是什么？
2. 为什么 `qwen3` / `mistral` / `qwen2` 读一个就够？
3. GDN 层为什么不经过 `attn_backend`？它靠什么拿每步的元数据？
4. `first_k_dense_replace` 影响哪些计算？
5. DeepSeek-V4.1 的 Engram 表为什么不能全部放进显存/内存？它的读路径是什么？
6. 多模态族需要额外实现哪两个接口？分别被谁调用？
7. 为什么 `iter_weights` yield 的是"运行时名"而不是 checkpoint 名？
8. `gemma4_unified` 与 `gemma4` 的差别在哪一层？
