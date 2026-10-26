# 09 · attention：注意力后端抽象、能力矩阵、9 个后端、prefill/decode 分流

> 行号基于 `377a3bc`。完整符号清单见 [`99-symbol-index/attention.md`](99-symbol-index/attention.md)。

---

## 1. 模块职责与在系统中的位置

注意力是"一个前向里最贵、且最依赖硬件的部分"。本模块把它抽象成**一个统一接口 + 一组可替换实现**，让模型层只写一句 `ctx.attn_backend.forward(q, k, v, layer_id, batch, attn_spec)`。

- **上游（调用方）**：`models/<family>/attention.py`（每个模型族的注意力装配）。
- **下游（被调用）**：`kvcache` 的 `page_table`/池视图、`kernel/triton/*` 与 `kernel/fla/*`、flashinfer / sgl-kernel / TensorRT-LLM 的 Python 绑定。
- **选型方**：`engine.py:124 _resolve_auto_attention_backend` —— 模型通过 `ModelConfig.kv_cache_group_specs()` 声明"我需要哪些 `AttnType`"，引擎据此挑后端。

```
models/<family>/attention.py
        │  forward(q,k,v,layer_id,batch,spec)
        ▼
attention/__init__.py  ← Registry + BackendInfo 能力矩阵（纯数据）
        │  create_attention_backend(name, model_config)
        ▼
fa / fi / trtllm / triton / dsa / dsv4_sparse / dsv41_sparse / m3_sparse / qsa_sparse
        │  prepare_metadata → forward
        ▼
kvcache.page_table（寻址） + kernel（真正计算）
```

---

## 2. 文件清单

| 文件 | 行数 | 职责 | 阅读建议 |
|---|---:|---|---|
| `base.py` | 130 | `AttnType` / `AttentionSpec` / `BaseAttnMetadata` / `BaseAttnBackend` / `HybridBackend` | **精读**（契约全在这） |
| `__init__.py` | 209 | `Registry` + `BackendInfo` 能力矩阵 + 9 个注册项 + `create_attention_backend` | **精读** |
| `fa.py` | 199 | FlashAttention 后端（sgl-kernel） | 精读 |
| `fi.py` | 314 | FlashInfer 后端 | 精读（有 page_size=1 等硬约束） |
| `triton.py` | 356 | 纯 Triton 后端（唯一同时支持 FULL+SWA） | 精读 |
| `trtllm.py` | 173 | TensorRT-LLM 后端（sm_100 only） | 概览 |
| `dsa.py` | 376 | DeepSeek/GLM 的 MLA + DSA 稀疏索引器 | 概览 + 关键方法 |
| `dsa_indexer.py` / `dsa_indexer_kpool.py` | 60 / 288 | DSA 索引器的两种实现（GLM-5.2 vs glm5_next 的 kpool） | 概览 |
| `dsv4_sparse.py` / `dsv4_compress.py` / `dsv4_indexer.py` | 291 / 154 / 87 | DSV4 的 window+compressed 稀疏注意力 | 概览 |
| `dsv41_sparse.py` | 438 | DSV4.1 的 packed-pool 版本 | 概览 |
| `m3_sparse.py` | 443 | MiniMax-M3 的 GQA block-sparse | 概览 |
| `qsa_sparse.py` | 542 | Qwen3.8-Flash-Next 的压缩块稀疏 | 概览 |
| `linear.py` | 141 | GDN 线性注意力的 `FLAMetadata` + 构建函数 | 精读（GDN 不走后端，只借它建 metadata） |
| `utils.py` | 24 | `BaseCaptureData`（图捕获公共基类） | 略读 |

---

## 3. 核心抽象

### 3.1 `AttnType`（`base.py:13`）——注意力类型分类学

**一个值对应一个 KV 池族**，这是本模块与 `kvcache` 的耦合点。

| 值 | 含义 | 对应池族 |
|---|---|---|
| `FULL` | 统一因果 MHA/GQA | `MHAKVCache` |
| `SWA` | 滑动窗口混合（window/sinks） | `HybridSWAKVCache` |
| `MLA` | 普通 latent-KV MLA | `MLAKVCache` |
| `DSA` | latent-KV MLA + DSA 稀疏索引器 | `DSAKVCache` |
| `DSV4` | DSV4 的 window + compressed 稀疏 | `DSV4PagedKVCache` |
| `DSV41` | DSV4.1：window + 跨层共享 compressed KV（打包 fp8/fp4 行） | `DSV41PagedKVCache` |
| `LINEAR` | GDN/mamba 循环状态层 | `LinearStatePool` |
| `BSA` | GQA block-sparse（MiniMax-M3） | `BSAKVCache` |
| `QSA` | 压缩块稀疏（Qwen3.8-Flash-Next） | `QSAKVCache` |

属性 `backend_driven`（`:36`）：**除 `LINEAR` 外都为 True**。原因写在注释里：线性层直接调 fla 算子（+ `batch.fla_metadata`），不经过注意力后端，所以它不约束后端选择。

### 3.2 `AttentionSpec`（`base.py:43`）

**per-call** 的注意力修饰，而不是 per-layer 的配置：

| 字段 | 含义 |
|---|---|
| `sliding_window` | 本层窗口（None = 全注意力） |
| `sm_scale` | 缩放因子覆盖 |
| `sinks` | attention sinks 张量 |
| `bidirectional_mm_blocks` | 多模态跨度内的行也看后面的 key（图像块内双向） |

### 3.3 `BaseAttnBackend`（`base.py:57`）——5 个抽象方法

| 方法 | 行 | 调用时机 | 契约 |
|---|---:|---|---|
| `forward(q,k,v,layer_id,batch,attn_spec=None)` | 59 | **每个前向、每层一次** | 返回 `[n_tokens, heads, head_dim]`；必须消费 `attn_spec`（不能消费就要报错） |
| `prepare_metadata(batch)` | 70 | `_prepare_batch` 里、forward 之前 | 构造 `batch.attn_metadata`（cu_seqlens / indices / page-table 切片） |
| `init_capture_graph(max_seq_len, bs_list)` | 73 | 启动图捕获前 | 分配静态捕获缓冲 |
| `prepare_for_capture(batch)` | 76 | 每次捕获前 | 把 batch 的寻址指到静态缓冲 |
| `prepare_for_replay(batch)` | 79 | **每次 replay 前** | 把本步寻址复制进静态缓冲并重指 |
| `reset_capture()` | 81 | 运行时 cache rebuild 前 | 丢弃捕获 scratch，让 `init_capture_graph` 能重跑（默认实现清 `capture`/`capture_bs`/`max_graph_bs`） |

### 3.4 `BaseAttnMetadata`（`base.py:52`）
只有一个抽象方法 `get_last_indices(bs)`：返回每个请求**最后一个 token 的下标**（decode 时用来定位要写入的 KV 位置）。

### 3.5 `HybridBackend`（`base.py:94`）
当 `--attention-backend` 写成 `"prefill,decode"` 两段时构造。`forward`（`:103`）与 `prepare_metadata`（`:115`）都按 `batch.is_prefill` 分流。
**关键约束**：`init_capture_graph`/`prepare_for_capture`/`prepare_for_replay`/`reset_capture`（`:119-128`）**全部只委托给 decode 后端**——因为只有 decode 会被图捕获。所以"给 prefill 换个更快的后端"不影响解码稳定性。

### 3.6 `BackendInfo`（`__init__.py:21`）——能力矩阵（纯数据）

| 字段 | 含义 |
|---|---|
| `supported_types` | 该后端能服务的 `AttnType` 集合 |
| `requires_flashinfer` / `requires_sgl_kernel` | 需要哪个可选包 |
| `requires_sm100` | 需要 sm_100 系 |
| `page_sizes` | 允许的 page_size（None = 任意）；配置期会把非法值强制到最后一个条目 |
| `consumes_attn_spec` | 是否消费 per-call `AttentionSpec`；不消费的后端遇到非 None 会**报错** |

---

## 4. 9 个后端总表

| 后端 | 注册行 | `supported_types` | 依赖 | 硬约束 | `consumes_attn_spec` | 适用场景 |
|---|---:|---|---|---|---|---|
| `trtllm` | 41 | FULL | flashinfer + sm_100 | 需 sm_100a/103a cubin；`page_sizes=(16,32,64)`；128 MiB workspace | ✗ | sm_100 上的最快 FULL |
| `fi` | 56 | FULL | flashinfer | **`page_size` 必须 =1**；decode 用 `backend="fa2"`（注释：fa3 慢） | ✗ | 通用 FULL（Hopper/Ampere） |
| `fa` | 69 | FULL | sgl-kernel | **sm_12x 直接报错**（无 FA3/FA4 cubin）；Hopper→FA3，sm_100→FA4 | ✗ | sm_90/sm_100 的 FULL |
| `triton` | 82 | **FULL + SWA** | 无 | 唯一 `consumes_attn_spec=True` | ✓ | 回退路径 / 所有窗口模型 |
| `dsv4_sparse` | 95 | DSV4 | 无（自研 Triton） | 需 DSV4 池 | — | DeepSeek-V4 |
| `dsv41_sparse` | 105 | DSV41 | 无 | 需 DSV41 池 | — | DeepSeek-V4.1 |
| `dsa` | 115 | DSA | 无 | 内部再分 `DSAIndexerMixin` / `KpoolPlan` 两种索引器 | — | GLM-5.2 / glm5_next |
| `m3_sparse` | 130 | BSA | 无 | 内部为 dense 前缀层再解析一个 FULL 后端（`_pick_inner_backend:55`） | — | MiniMax-M3 |
| `qsa_sparse` | 146 | QSA | 无 | 压缩 slab + pending ring（`_compress`/`_select`） | — | Qwen3.8-Flash-Next |

**自动选型顺序**（`engine.py:127-148` 的 docstring 与代码）：
```
FULL-only 模型：trtllm(sm_100) → "fa,fi"(sm_90) → fi(有 flashinfer) → triton
SWA 模型：      triton（唯一选择，:140-141）
带稀疏 AttnType：按类型直接选对应后端（优先于 FULL 候选）
```

### 4.1 四个通用后端的内部差异

| | `fa` | `fi` | `trtllm` | `triton` |
|---|---|---|---|---|
| metadata 里的寻址 | `page_table[..., :max_seqlen_k:page_size] / page_size`（`:108-113`） | 把 page_table 行**拼成很长的 `indices`**，`page_size=1`（`:225-260`） | 同 fa 的切片思路（`:126-131`） | `indptr`=cumsum(device lens)、`cu_seqlens_q`、`q_to_req`、`prefix_lens`，SWA 还要算 `swa_indices = translate_loc_from_full_to_swa(indices)`（`:222-273`） |
| 捕获数据 | `FACaptureData`（`:18`） | `FICaptureData`（`:36`），用 `one_tensor`/`indices` 复用静态缓冲 | `TRTLLMCaptureData`（`:17`） | `TritonCaptureData`（`:17`） |
| 独特机制 | `num_splits`/`pack_gqa` 暴露给用户调（注释"Can be tuned for speed"） | flashinfer `plan()` 每 metadata 调一次，且**用 event 保护**（复用了 pinned 宿主缓冲，`:147-190`）；`use_tensor_cores` 由 GQA≥4 决定（`:279`） | 128 MiB workspace（`:45-47`） | `_ensure_decode_scratch`（`:99`）预分配 split-KV 中间缓冲；`_swa_capture_enabled`/`_capture_swa_indices`/`_point_to_capture`（`:295-331`）处理 SWA 在图内翻译 |

### 4.2 五个模型专属后端

它们的共同结构：**自己拥有 `*Metadata` 类 + 自己构造寻址 + 调自研 Triton kernel**，并按需实现 `prepare_for_capture`/`prepare_for_replay` 把"本步寻址"复制进静态图缓冲。

| 后端 | metadata 类 | 关键辅助 | forward 结构 |
|---|---|---|---|
| `DSAAttnBackend`（`dsa.py:103`） | `DSAMetadata`（`:80`） | `_model_args`(141) 取 MLA/indexer 维度；`_build_index_slots`(148) IndexShare（full 层拥有 indexer 槽，follower 复用）；`_store_index`(161) 写入本次的 index key | `forward`(166) → `_attend`(195) 或 `mla_forward`(206) → `_decode`(235)/`_prefill`(288)；`_select_prefill`(261) 做 per-request 因果 top-k |
| `M3SparseAttnBackend`（`m3_sparse.py:150`） | `M3SparseMetadata`（`:128`） | `_k_rows`/`_v_rows`(191/195)；`_prefill_plan`(297) 每 forward 一次、层不变；`_block_base_view`(382) 取页表每 128 列 | `forward`(224) → `bsa_forward`(243) → `_decode`(272)/`_prefill`(341) |
| `QSASparseAttnBackend`（`qsa_sparse.py:101`） | `QSASparseMetadata`（`:73`） | `_qsa_group`(150)、`_cmp_pages`(162)、`_index_rope_*`(168/188)、`_plan_index_writes`(311)、`_update_index_cache`(345)、`_select`(385)/`_top_blocks`(444)、`_topk_scratch_width`(474)/`_scratch`(483) | `forward`(261) → `qsa_forward`(276) |
| `DSV4` 系（`dsv4_sparse.py` / `dsv41_sparse.py`） | 各自 metadata | compress/indexer 辅助模块（`dsv4_compress.py`/`dsv4_indexer.py`） | window + compressed 两级注意力 |
| `dsa_indexer_kpool.py` 的 `KpoolPlan`（`:53`） | — | 一次 forward 的 slab/ring 路由（层不变） | 被 `dsa` 后端用于 kpool 变体 |

---

## 5. 函数详解

### 5.1 `__init__.py`

| 符号 | 行 | 作用 | 关键行为 |
|---|---:|---|---|
| `BackendCreator`（Protocol） | 16 | 工厂签名 | `(model_config) -> BaseAttnBackend` |
| `BackendInfo` | 21 | 能力矩阵条目（见 §3.6） | 纯数据，注册时不 import kernel |
| `create_trtllm_backend` | 50 | 惰性 import 并构造 trtllm | 函数体内 import，避免启动即拉重库 |
| `create_fi_backend` / `create_fa_backend` / `create_triton_backend` | 63/76/89 | 同上 | |
| `create_dsv4_sparse_backend` / `create_dsv41_sparse_backend` / `create_dsa_backend` / `create_m3_sparse_backend` / `create_qsa_sparse_backend` | 99/109/119/140/155 | 五个模型专属后端 | |
| `attention_backend_info(name)` | 161 | 查能力矩阵 | 未注册则 KeyError |
| `validate_attn_backend(name, required)` | 165 | 校验后端能否服务所需类型 | 报错信息列出 supported |
| `create_attention_backend(name, config)` | 181 | **总入口** | 解析 `"a,b"` 两段 → `HybridBackend`；否则直接查表构造 |

### 5.2 `linear.py`（GDN 不走后端，但它在这里）

| 符号 | 行 | 作用 |
|---|---:|---|
| `FLAMetadata` | 13 | 每 forward 一次的 GDN 元数据（`cu_seqlens` / `cache_indices` / continuation 标志） |
| `build_fla_metadata` | 47 | 用 pinned 宿主 staging + `non_blocking` H2D 构造它 |
| `_build_track_metadata` | 94 | hybrid-radix 的额外轨：为每个跨 ×CHUNK 边界的请求建快照轨 |

---

## 6. 关键流程逐段解析

### 6.1 启动期：模型声明 → 引擎选型
```
ModelConfig.kv_cache_group_specs()        # models/config.py:460
  → 引擎 _required_attn_types(config)      # engine.py:88，去掉 LINEAR
  → _resolve_auto_attention_backend        # engine.py:124
       按 per-type 优先级生成候选：trtllm → "fa,fi" → fi → triton
       逐个检查：架构条件 → 包是否安装 → 逗号分段是否覆盖所有必需类型
  → create_attention_backend(name, model_config)   # __init__.py:181
  → 显式指定时走 _validate_attention_backend_choice（engine.py:163），不匹配直接报错
```

### 6.2 每次 forward 前的 `prepare_metadata`（三种 case）
以 fi 为例（`fi.py:225-260`），它要区分：
1. **全 decode**：`cu_seqlens_q = arange(bs+1)`，`indices` = page-table 行的拼接，`last_page_len` 全 1，`page_size=1`；
2. **冷 prefill**（无缓存命中）：`cu_seqlens_k = cu_seqlens_q`；
3. **带命中 extend**：`cu_seqlens_k` 要把 `cached_len` 算进去。
fa/trtllm 则简单些：把 page table 切片再除以 page_size（`fa.py:108-113`）。
triton 最复杂：除上面那些还要算 `q_to_req`、`prefix_lens`，SWA 模式还要 `translate_loc_from_full_to_swa`（`:244-251`）——而且**这一步在 replay 时也要重算**。

### 6.3 CUDA graph 的三段分工
```
init_capture_graph(max_seq_len, bs_list)   启动时一次：分配静态缓冲
prepare_for_capture(batch)                 每次捕获前：batch 的寻址指到静态缓冲
prepare_for_replay(batch)                  每次 replay 前：把本步寻址写进静态缓冲并重指
```
以 `fa` 为例（`fa.py:123-149`）：`init` 建 `FACaptureData`；`prepare_for_capture` 把 batch 的 `page_table`/`cu_seqlens` 绑到静态张量；`prepare_for_replay` 把本步的 `out_loc`/`positions`/`page_table` 切片 copy 进去。

### 6.4 为什么非 triton 后端遇到 `AttentionSpec` 要报错
`fi.py:209-212`、`fa.py:63-66`、`trtllm.py:58-61` 都是显式 raise。理由：这些后端的库调用没有 per-call window/sinks/sm_scale 参数。**静默忽略 spec 会导致窗口模型算错却看起来正常**，所以选择报错。这也是 SWA 模型被钉在 triton 的根本原因。

### 6.5 GDN 为什么不走 `attn_backend`
`AttnType.LINEAR.backend_driven == False`（`base.py:36-39`）。线性层在前向里直接调 `kernel/fla/*`（prefill 用 chunked，decode 用 recurrent），只需要 `batch.fla_metadata`，不需要 attention 后端参与寻址。

### 6.6 prefill/decode 用不同后端
```
--attention-backend "fa,fi"   → HybridBackend(prefill=fa, decode=fi)
```
`forward`/`prepare_metadata` 按 `batch.is_prefill` 分流（`base.py:103/115`）；四个捕获相关方法**只给 decode 后端**（`:119-128`）。

---

## 7. 与其他模块的关系

| 方向 | 模块 | 接口 | 说明 |
|---|---|---|---|
| 被调用 | `models/<family>/attention.py` | `ctx.attn_backend.forward(...)` | 每层一次 |
| 使用 | `kvcache` | `batch` 的 `page_table`、池视图、`translate_loc_from_full_to_swa` | 寻址来源 |
| 使用 | `kernel/triton`、`kernel/fla` | 各后端的计算内核 | 稀疏后端全是自研 Triton |
| 使用 | `layers` | `linear.py` 的 `FLAMetadata` 由 `build_fla_metadata` 造 | GDN |
| 被选择 | `engine` | `_resolve_auto_attention_backend` / `create_attention_backend` | 启动期 |
| 依赖 core | `core.Batch` / `Context` | 读 `batch.positions`、`attn_metadata`，写 `batch.attn_metadata` | |
| 声明方 | `models/config.py` | `KVCacheGroupSpec.attn_type` | 模型声明需求 |

---

## 8. 常见坑

1. **`fi` 必须 `page_size == 1`**（`fi.py:57,66` 的类型标注 + 断言）。设了别的 page_size 会直接失败。
2. **`fa` 在 sm_120/121 上直接报错**（`fa.py:38-43`），提示改用 `fi` 或 `triton`。
3. **`trtllm` 需要 sm_100a/103a 的 cubin**；`page_sizes=(16,32,64)` 且配置期会把非法值强制改成最后一个。
4. **只有 `triton` 消费 `AttentionSpec`** —— 窗口/sinks/sm_scale 模型因此只能用它。
5. **只有 decode 侧会被图捕获**（`HybridBackend` 只委托 decode）。
6. **`reset_capture` 是 rebuild 的前置**：不清 scratch，`init_capture_graph` 会重入崩溃。
7. **稀疏后端的 metadata 里"每 forward 一次、层不变"的计划要复用**（如 `m3_sparse._prefill_plan`、`qsa._plan_index_writes` 的注释），否则会 N 层重复计算。

---

## 9. 自测问题

1. 一个模型"需要哪些 AttnType"是从哪来的？谁把它翻译成后端名？
2. `AttnType.LINEAR` 为什么 `backend_driven=False`？这对后端选择有什么影响？
3. 为什么 SWA 模型在 `--attention-backend auto` 下只会得到 triton？
4. `prepare_metadata` / `prepare_for_capture` / `prepare_for_replay` 三者各在什么时机被调用？
5. `HybridBackend` 为什么只把捕获相关方法委托给 decode 侧？
6. `fi` 的 `indices` 与 `fa` 的 page-table 切片，本质上是同一件事的两种表示吗？
7. 一个后端不消费 `AttentionSpec` 时，为什么选择报错而不是忽略？
8. 加一个新注意力后端需要改哪几处（注册、能力矩阵、工厂、模型声明）？
