# 07 · models（核心机制）：架构注册表、ModelConfig 契约、权重加载与分片融合

> 行号基于 `377a3bc`。完整符号清单见 [`99-symbol-index/models.md`](99-symbol-index/models.md)（1,178 个符号；本文件只覆盖**顶层共享机制**，19 个模型族见 `08`）。

---

## 1. 模块职责与在系统中的位置

`models/` 回答一个问题：**"一个 HF checkpoint 目录，怎么变成我能在 GPU 上 `forward()` 的对象？"**

它把这件事拆成三步，并把"每族不同"的部分压缩到 4 个钩子里：

```
config.json ─ architectures[0] ──▶ _MODEL_REGISTRY（arch → ModelSpec）
                                        │
                    ① parse_config ─────┤ 每族一个：HF config → ModelConfig
                                        │
safetensors/FTW ─ ② iter_weights ───────┤ 每族一个：yield (name, tensor)
                                        │
                    ③ model_cls ────────┘ 每族一个：nn.Module
                                        │
                        models/loader.py（分片/融合/堆叠，全族共用）
                                        │
                        Engine._load_weights → load_state_dict
```

- **上游**：`engine/engine.py:530 _load_weights`；`engine/config.py:114/133` 用 `model_spec`/`model_config`。
- **下游**：`layers`（层原语）、`attention`（后端）、`kvcache`（组声明）、`moe`（专家 bank）、`kernel`（专属算子）、`mm`（多模态处理器）。

**一句话概括设计**：模型族只描述"差异"，共性（张量名映射、分片、融合、专家堆叠、量化绑定、FTW）全部上收到本模块。所以加一个新族通常只需 4-5 个文件、几百行（`llama/` 只有 290 行）。

---

## 2. 文件清单

| 文件 | 行数 | 职责 | 阅读建议 |
|---|---:|---|---|
| `register.py` | 355 | `EncoderSpec` / `ModelSpec` / `_MODEL_REGISTRY` / `get_model_spec` / `NameMap` / `checkpoint_quant_config` | **精读**（355 行，全读也快） |
| `config.py` | 527 | `ModelConfig` 契约 + `RotaryConfig` + 注意力组声明 + 量化探测 | **精读**（字段多，重点看结构与 6 个 `attention_group_*` 方法） |
| `loader.py` | 311 | `shard_tensor` / `MergeRule` / `iter_merged_tensors` / `iter_stacked_experts` / `ShardReader` / NVFP4 融合 | **精读** |
| `weight.py` | 290 | `load_weight` 总入口（FTW 分支、并行 O_DIRECT、vision、q4_0 专家源） | **精读** |
| `blocks.py` | 94 | `BaseLLMModel`（forward/encode/forward_host_ctx/embed_input_ids）、`GatedMLP` | 精读（很短） |
| `weight_stream.py` | 123 | `BlockWeightStreamer`：把块权重放主机 bank、两块两块地流式送 GPU | 按需（多模态 tower 用） |
| `nvfp4_banks.py` | 139 | NVFP4 专家源的组织规格（`Nvfp4ExpertSourceSpec` + 逐专家 piece 迭代） | 按需 |
| `__init__.py` | 35 | `create_model` / `load_weight` / 配置类 re-export | 略读 |
| `gguf/` | — | GGUF 读取与"假 architectures"shim（见 `08` 的 gguf 族） | 按需 |

---

## 3. 注册表：`ModelSpec`（`register.py:19`）

### 3.1 字段

| 字段 | 类型 | 作用 | 谁读它 |
|---|---|---|---|
| `module` | str | 模型族所在模块（如 `freetoken.models.qwen3_5_moe`） | `get_model_class`（`:348`）|
| `model_cls` | str | 该模块里的模型类名 | `get_model_class` |
| `parse_config` | str | 配置解析函数名（默认 `parse_config`） | `EngineConfig.model_config`（`engine/config.py:143`）|
| `iter_weights` | str | 权重迭代器名（默认 `iter_weights`） | `weight.py:load_weight` |
| `checkpoint_roots` | tuple[(attr_path, ckpt_root)] | 运行时属性路径 ↔ checkpoint 根的前缀映射 | `NameMap`（量化配置查找、张量名翻译）|
| `checkpoint_segments` | tuple[(attr_path, ckpt_path)] | 段级重命名（如 `feed_forward.shared_mlp → mlp`） | 同上 |
| `packed_modules_mapping` | tuple[(fused_leaf, (src_leaves...))] | **融合模块 → 源张量名**（如 `qkv_proj → q_proj/k_proj/v_proj`） | `loader.iter_merged_tensors`、量化查找 |
| `unquantized_modules` | tuple[str] | checkpoint 量化配置覆盖但本族按 bf16 加载的张量 glob | `configs` 的 `unquantized` 判定 |
| `mm_processor` | str \| None | 多模态处理器 "module:Class"，None = 该族不收多模态 | `mm.processor.get_mm_processor` |
| `encoders` | tuple[EncoderSpec] | 该族注册的编码塔 | `EngineConfig.active_encoders` |

### 3.2 `EncoderSpec`（`register.py:12`）

| 字段 | 含义 |
|---|---|
| `kind` | `"vision"` / `"audio"`，也是 `--mm-disable` 接受的名字 |
| `config_key` | checkpoint config 里构建该塔的 section（如 `vision_config`） |
| `modalities` | 它实际服务的输入类型（如 `("image",)`） |

### 3.3 `_MODEL_REGISTRY`（`register.py:88`）—— 29 个架构全表

| # | arch 名 | 备注 |
|---:|---|---|
| 1 | `LlamaForCausalLM` | 最简样板 |
| 2 | `Qwen2ForCausalLM` | llama + qkv bias |
| 3 | `Qwen3ForCausalLM` | llama + qk_norm |
| 4 | `Qwen3MoeForCausalLM` | MoE |
| 5-6 | `Qwen3VLForConditionalGeneration` / `Qwen3VLMoeForConditionalGeneration` | 多模态（vision tower） |
| 7-9 | `MiniMaxM2ForCausalLM` / `MiniMaxM3SparseForConditionalGeneration` / `MiniMaxM3SparseForCausalLM` | M3 = block-sparse（BSA） |
| 10-11 | `DeepseekV4ForCausalLM` / `DeepseekV41ForCausalLM` | DSV4 / DSV4.1（无 `model.` 前缀；V4.1 多模态 + Engram） |
| 12-16 | `Qwen3_5MoeForConditionalGeneration` / `Qwen3_5MoeForCausalLM` / `Qwen3_5ForCausalLM` / `Qwen4ExpForConditionalGeneration` / `Qwen3_5ForConditionalGeneration` | Qwen3.5/3.6/3.8 系列；`qwen4_exp` = GDN+QSA+超连接+PLE |
| 17 | `MuseGlimmerForConditionalGeneration` | 窗口 ViT + SWA×3/full 模式 |
| 18-19 | `MistralForCausalLM` / `Mistral3ForConditionalGeneration` | |
| 20-24 | `Gemma4ForConditionalGeneration` / `Gemma4ForCausalLM` / `Gemma4UnifiedForConditionalGeneration` / `Gemma4UnifiedForCausalLM` / `Gemma4GGUFForCausalLM` | 含 GGUF 变体 |
| 25 | `GptOssForCausalLM` | MXFP4 专家 |
| 26 | `Glm4MoeForCausalLM` | |
| 27 | `GlmMoeDsaForCausalLM` | DSA 稀疏（GLM-5.2 系） |
| 28-29 | `Glm5NextForConditionalGeneration` / `Glm5NextForCausalLM` | KDA 线性注意力（GLM-5.3） |

`get_model_spec`（`:323`）在未注册时抛 `KeyError` 并列出支持项；`_load_attr`（`:330`）按 "module:attr" 惰性 import（注册时不 import 任何模型模块 → 启动快）。

### 3.4 `NameMap`：三张映射表

`checkpoint_quant_config`（`:335`）用一个 `NameMap(roots, segments, packed)` 把**运行时张量名**翻译成**checkpoint 张量名**，以便在 `quantization_config` 里查方案。

| 映射 | 例子 | 作用 |
|---|---|---|
| `checkpoint_roots` | `("model", "model.language_model")` | 运行时有 `model.`、checkpoint 里是 `model.language_model.`（多模态包装） |
| `checkpoint_segments` | `("feed_forward.shared_mlp", "mlp")` | 段级重命名（Gemma-4） |
| `packed_modules_mapping` | `("qkv_proj", ("q_proj","k_proj","v_proj"))` | 融合叶子 → 源叶子（量化配置按源名写） |

**具体例子**（Qwen3.5-MoE 的 `in_proj_qkvz`）：注册表里写作 `("in_proj_qkvz", ("in_proj_qkv","in_proj_z"))`，于是量化查找会分别用 `in_proj_qkv` 与 `in_proj_z` 两个 checkpoint 名去查。

---

## 4. `ModelConfig`（`config.py:259`）——模型与引擎之间的契约

### 4.1 必填字段（模型结构）

`num_layers` / `num_qo_heads` / `num_kv_heads` / `head_dim` / `hidden_size` / `vocab_size` / `intermediate_size` / `rms_norm_eps` / `rotary_config` / `hidden_act` / `tie_word_embeddings` / `num_experts` / `num_experts_per_tok` / `moe_intermediate_size` / `norm_topk_prob` / `model_type` / `architectures`

### 4.2 关键可选字段（按用途分组）

| 组 | 字段 | 含义 |
|---|---|---|
| **策略** | `moe_strategy`、`decode_target` | 引擎从 flag 写入；`decode_target` 决定哪些专家内核可用 |
| **量化** | `quant`、`expert_quant`、`weight_block_size`、`attn_quant`、`dense_quant`、`lm_head_quant`、`moe_weight_format` | 分别描述"专家 / 块大小 / 注意力 / 稠密 MLP / lm_head"各自的量化种类 |
| **MoE 细节** | `moe_enabled`、`shared_expert_intermediate_size`、`first_k_dense_replace`、`n_shared_experts`、`routed_scaling_factor`、`n_group`、`topk_group` | DeepSeek/GLM 风格的路由细节；`first_k_dense_replace` 决定"前几层是稠密"→ 专家只存在于后面的层 |
| **注意力** | `attention_groups`、`has_attn_bias`、`has_router_bias`、`attn_sm_scale`、`use_qk_norm` | |
| **数值** | `final_logit_softcapping`、`embedding_scale`、`post_norm_eps`、`output_multiplier`、`swiglu_limit`、`hidden_act_alpha` | 各族的小修正 |
| **多模态** | `vision_config`、`image_token_id` | |

### 4.3 派生属性（引擎/kvcache/attention 全靠它们做决策）

| 属性 | 行 | 含义 |
|---|---:|---|
| `is_moe` | 361 | 是否 MoE |
| `num_moe_layers` | 365 | **拥有稀疏 MoE 块的层数**（= `num_layers - first_k_dense_replace`）；专家槽位只按它算 |
| `is_multimodal` | 374 | |
| `model_is_mrope` | 378 | 任一 full 层用 3 轴 rope → 需要 mrope 位置 |
| `has_hybrid_attention` / `has_swa_attention` / `has_linear_attention` | 386/390/394 | 三种混合注意力 |
| `attention_group_for_layer` / `attn_type_for_layer` | 409/448 | 层 → 组 → **AttnType**（后端的选型依据） |
| `kv_cache_group_specs` | 460 | `[(KVCacheGroupSpec)]`：每组的层数/头数/头维/窗口/AttnType |
| `kv_cache_groups` | 516 | 归一成 `(num_layers, num_kv_heads, head_dim)` |

### 4.4 注意力组声明（`config.py:113-257`）

| 类 | 行 | 代表什么 |
|---|---:|---|
| `KVCacheGroupSpec` | 113 | 一个 KV 缓存组：`num_layers`/`is_swa`/`attn_type`/`sliding_window`… |
| `BaseAttentionGroupConfig` | 144 | 组基类（`owns_layer`） |
| `FullAttentionGroupConfig` | 153 | 全注意力组 |
| `SWAAttentionGroupConfig` | 174 | 滑窗组 |
| `LinearGatedDeltaGroupConfig` | 187 | GDN 线性注意力组 |
| `DSV4AttentionGroupConfig` | 204 | DSV4 三件套（window + compressed + indexer） |
| `SlotStateSpec` | 243 | 搭 `LinearStatePool` 槽位的额外 per-request 张量 |

**这是"模型声明需求、引擎选择实现"的关键接缝**：模型族在 `parse_config` 里填 `attention_groups`，引擎据此算 `_required_attn_types` 并选后端（见 `09`），`kvcache` 据此选池族（见 `05`）。

### 4.5 量化探测（`config.py:11/44`）

- `detect_expert_quant`（`:11`）：从 `quantization_config` 推出专家量化（`"nvfp4"` / `"fp8_block"` / …）。
- `detect_compressed_tensors_nvfp4`（`:44`）：识别 compressed-tensors W4A16 checkpoint。
- `mrope_layout_from_rope_params`（`:103`）：rope_parameters 的标志 → 布局名（`mrope_interleaved_glm` / `mrope_interleaved` / `contiguous`）。

---

## 5. 权重加载流水线

### 5.1 全链（`weight.py:209 load_weight`）

```
Engine._load_weights（engine.py:530）
  → _spec_for_model_path（weight.py:194）            # 由 config 定 spec
  → if is_ftw_checkpoint(model_path):                # FTW 快路径
        iter_ftw_weights(...)                        # 见 13
    else:
        每族的 iter_weights(...)                      # 逐张量 yield
  → shard_tensor（loader.py:25）                      # TP 分片
  → iter_merged_tensors（loader.py:128）              # qkv / gate_up 融合
  → iter_stacked_experts（loader.py:275）             # 专家堆叠
  → （可选）iter_expert_tensors_parallel（weight.py:68）# 并行 O_DIRECT 读专家
  → load_state_dict
  → finalize_quant（engine 侧）
  → _load_resident_experts（engine.py:564）→ moe/expert_banks
```

### 5.2 `shard_tensor`（`loader.py:25`）的切分维度规则

| 张量类别 | 切分维度 | 为什么 |
|---|---|---|
| `q_proj` / `k_proj` / `v_proj` / `gate_proj` / `up_proj` | **dim 0**（输出维） | 列并行：每个 rank 算一部分输出通道 |
| `o_proj` / `down_proj` | **dim 1**（输入维） | 行并行：每个 rank 吃一部分输入 |
| `embed_tokens` / `lm_head` | **vocab 区间** | 词表并行 |

### 5.3 融合与堆叠

| 机制 | 位置 | 作用 |
|---|---|---|
| `MergeRule` | `loader.py:19` | 描述"哪些源张量拼成哪个融合张量、沿哪个维度" |
| `iter_merged_tensors` | `loader.py:128` | 按 `packed_modules_mapping` 消费源张量并产出融合张量 |
| `_merge_info` | `loader.py:121` | 由 mapping 反查融合信息 |
| `iter_stacked_experts` | `loader.py:275` | 把 per-expert 张量堆成 `[E, ...]`（三种容器探测：per-expert / 预堆叠 / w123） |
| `_expert_stack_info` | `loader.py:265` | 探测用哪个堆叠策略 |
| `nvfp4_parts_ct` / `ct_nvfp4_fuse` / `ct_bf16_fuse` | `loader.py:210/222/249` | compressed-tensors 的 NVFP4 分量缓冲与融合（等齐 `.weight`/`.weight_scale`/`.weight_global` 再一次性产出） |

### 5.4 `ShardReader`（`loader.py:168`）
跨分片按名取张量，**分片文件句柄惰性打开**（`:176`）：避免一次性 mmap 全部 safetensors（大模型会爆虚拟地址/内存）。`names_in`（`:188`）用于探测某前缀下有哪些张量。

### 5.5 并行 O_DIRECT 专家读（`weight.py`）

| 函数 | 行 | 作用 |
|---|---:|---|
| `_read_shard_odirect_parallel` | 39 | 把整个 shard 用**分块多线程 O_DIRECT** 读进页对齐 mmap |
| `iter_expert_tensors_parallel` | 68 | 每族串行 `iter_weights` 的并行版 |
| `experts_scattered` | 155 | 慢路径信号：专家是否是"很多小张量"（决定并行读是否值得） |
| `drop_page_cache` | `loader.py:74` | 主动丢弃文件页缓存（banks + 整个 checkpoint 不能同时塞进主机内存） |

### 5.6 其它加载入口

| 函数 | 行 | 作用 |
|---|---:|---|
| `load_vision_weight` | 246 | 只读 vision encoder 张量（名字与 `load_weight` 一致） |
| `ftw_lacks_vision` | 255 | 旧版 FTW 没有 vision 张量 → 提示用 `--text-model-only` 或重新转换 |
| `load_q4_0_moe_expert_sources` | 265 | 加载（或 `dummy=True` 时伪造）打包的 GGUF Q4_0 专家源 bank |

### 5.7 `BlockWeightStreamer`（`weight_stream.py:49`）
"把一整摞块的权重放主机 bank，只驻留两块"的实现：
- `__init__`（`:52`）分配 bank 与两个 staging buffer；
- `_bind_all_to_host`（`:75`）把块张量绑到 host bank；
- `_prefetch`（`:88`）预取下一块；
- `device_bytes`（`:72`）报告设备端占用；
- `unstream`（`:81`）把所有权重放回设备成为常驻副本。
- `_slots`（`:28`）递归收集块的所有张量属性，**按 256 字节对齐**排布（`:28` 的 docstring）。
用户：多模态编码塔（`--mm-encoder-weights host`）。

### 5.8 `blocks.py`
- `BaseLLMModel`（`:?`）：`forward` / `encode`（多模态）/ `forward_host_ctx`（主机喂数钩子）/ `place_encoder_weights` / `embed_input_ids`。
- `GatedMLP`：标准 gated MLP 原语。
- `embed_input_ids`（`:51`）：处理多模态 pad 行——把 `batch.mm_embeds` 的前若干列 `index_copy_` 覆盖到对应行。

### 5.9 `nvfp4_banks.py`
`Nvfp4ExpertSourceSpec`（`:19`）+ `iter_nvfp4_expert_pieces`（`:66`）：为 NVFP4 专家逐 (expert, role) 产出 `gate/up/down` 的 codes 与 `_scale`。被 `moe/expert_banks` 消费。

---

## 6. 关键流程逐段解析

### 6.1 `create_model` 全链（`models/__init__.py:17`）
```
create_model(model_config)
  → get_model_class(model_config.architectures[0], model_config)     # register.py:348
        → spec = get_model_spec(arch)                                # :323
        → cls = _load_attr(spec.module, spec.model_cls)              # :330（惰性 import）
        → return cls(model_config)
```
注意：`parse_config` **不在**这里调用——它在 `EngineConfig.model_config`（`engine/config.py:133-144`）里就更早发生了。

### 6.2 张量名映射的完整例子（Qwen3.5-MoE 的 `in_proj_qkvz`）
```
运行时叶子：model.layers.3.linear_attn.in_proj_qkvz
  packed_modules_mapping: ("in_proj_qkvz", ("in_proj_qkv","in_proj_z"))
  checkpoint 侧：
     → model.layers.3.linear_attn.in_proj_qkv
     → model.layers.3.linear_attn.in_proj_z
  量化查找：对这两个源名分别查 quantization_config
  加载：iter_merged_tensors 把两者沿输出维拼成 in_proj_qkvz
```

### 6.3 FTW 分支
`is_ftw_checkpoint` 为真时，`iter_weights` 被换成 `iter_ftw_weights`（`checkpoint/ftw.py:365`），后续的分片/融合/堆叠**照常执行**——这就是"转换模型无关"的原因：FTW 只是换了字节读取层，不改语义。

### 6.4 量化配置注入
```
checkpoint_quant_config（register.py:335）
  → NameMap(roots, segments, packed) 翻译名字
  → 读 quantization_config → QuantConfig
engine/config.py:141-142: set_quant_config(quant)    # 进程全局
层构建时：quant_method_for(prefix) → 该层的方法/内核（见 10）
```

### 6.5 多模态权重的两种放置
```
place_encoder_weights("gpu")  → 全部常驻
place_encoder_weights("host") → BlockWeightStreamer 接管，只留两块
ftw_lacks_vision(path)        → 旧 FTW 没有 vision 张量，须 --text-model-only 或重转
```

---

## 7. 与其他模块的关系

| 方向 | 模块 | 接口 | 说明 |
|---|---|---|---|
| 被使用 | `engine` | `get_model_spec` / `create_model` / `load_weight` | 启动主链 |
| 使用 | `layers` | 层原语 | 模型族拼网络 |
| 使用 | `attention` | `create_attention_backend`（经 engine） | 模型族的 attention.py 拿后端 |
| 声明给 | `kvcache` | `kv_cache_group_specs()` | 决定池族 |
| 声明给 | `attention` | `attn_type_for_layer()` | 决定后端 |
| 使用 | `moe` | `load_expert_banks` | 专家 bank |
| 使用 | `mm` | `mm_processor` / `encoders` | 多模态 |
| 使用 | `checkpoint` | `iter_ftw_weights` / `load_ftw_banks` | FTW 路径 |
| 使用 | `layers/quantization` | `checkpoint_quant_config` → `set_quant_config` | 量化绑定 |
| 被使用 | `models/<19 families>` | 4 个钩子 | 见 `08` |

---

## 8. 读一个模型族的 6 步法

1. **`register.py` 找它那一行**：arch 名、`module`、`checkpoint_roots`、`packed_modules_mapping`、`encoders`。
2. **`config.py` 的 `parse_config`**：HF config → `ModelConfig`；注意它填了哪些 `attention_groups`。
3. **`model.py`**：decoder layer 的构造顺序（norm → attn → residual → norm → mlp/moe → residual），以及 `ForCausalLM.forward`。
4. **`attention.py`**：怎么把 q/k/v 交给 `ctx.attn_backend.forward`，`layer_id` 怎么传。
5. **`weight.py` 的 `iter_weights`**：yield 的名字与 `packed_modules_mapping` 是否对得上。
6. **回到 `loader.py`**：确认融合/堆叠/分片规则能覆盖它的张量名。

---

## 9. 常见坑

1. **`Qwen3` 就是 `llama` 加 `use_qk_norm=True`**，`mistral`/`qwen2` 同理——不要以为每个族都是独立实现。
2. **`checkpoint_roots` 有无 `model.` 前缀是坑**：DSV4 系没有 `model.`，多模态包装在 `model.language_model.`。
3. **`num_moe_layers != num_layers`**：`first_k_dense_replace` 让前几层是稠密，专家槽位只按前者算——算显存时用错会差很多。
4. **GGUF 走"假 architectures" shim**：`gguf/config.py` 伪造一个 `architectures` 字段，使同一套 dispatch 也能服务 GGUF。
5. **`parse_config` 不在 `create_model` 里调用**，在 `EngineConfig.model_config` 里更早发生。
6. **量化配置是进程全局**（`set_quant_config`），第一次访问 `config.model_config` 就锁定。
7. **`load_vision_weight` 与 `load_weight` 的名字规则必须一致**，否则 FTW 与 HF 两条路径的 vision 张量会对不上。

---

## 10. 自测问题

1. 一个 HF checkpoint 的 `architectures[0]` 到"能 forward 的模型"中间经过哪几步？各步由谁触发？
2. `ModelSpec` 的 `checkpoint_roots` / `checkpoint_segments` / `packed_modules_mapping` 三者分别在解决什么命名差异？
3. `ModelConfig.attention_groups` 被谁消费、消费来做什么？
4. `num_moe_layers` 与 `num_layers` 什么时候不相等？影响到哪些计算？
5. `shard_tensor` 对 `q_proj` 和 `o_proj` 分别切哪个维度，为什么？
6. 为什么说"FTW 转换是模型无关的"？
7. `BlockWeightStreamer` 与 `--mm-encoder-weights host` 是什么关系？
8. 加一个新模型族最少要写哪几个文件？各自职责是什么？
