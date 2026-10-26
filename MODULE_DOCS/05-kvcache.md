# 05 - kvcache：分页 KV 池与前缀缓存

> 范围：`python/freetoken/kvcache/` 全部 20 个文件，约 4,970 行。
> 行号基于 `377a3bc`（与 `MODULE_DOCS/99-symbol-index/kvcache.md` 同源）；文中路径相对 `python/freetoken/`。
> 纯静态阅读，未执行任何代码，未加载 torch/GPU。

## 1. 模块职责与在系统中的位置

`kvcache` 负责两件互不重叠的事：

1. **物理存储**：按页（page）切分的连续显存张量，回答"第 `layer` 层第 `loc` 个 token 的 K/V 在哪"（`kvcache/base.py:159-172`）。
2. **前缀复用**：一棵按页为键的 radix 树，回答"这段 prompt 有多少 token 可以跳过 prefill"（`kvcache/base.py:217-285`）。

两者通过 engine 持有的 `page_table`（int32，`(max_running_req+1, width)`，**存 raw token loc 而非页号**）解耦（`engine/engine.py:446-458`）。三层映射如下：

```
 L3 复用层   scheduler/cache.py: CacheManager        (scheduler/cache.py:32)
             cache_type="naive"        -> NaivePrefixCache    (kvcache/naive_cache.py:16)
             cache_type="radix"        -> RadixPrefixCache    (kvcache/radix_cache.py:126)
             "hybrid_radix"（内部）    -> HybridRadixCache    (kvcache/hybrid_radix_cache.py:55)
             cache_type="swa_radix"    -> SWARadixCache       (kvcache/swa_radix_cache.py:66)
                              |
                              |  value = 一页的 full-pool loc 列表
                              v
 L2 虚拟层   ctx.page_table / engine.page_table  int32 [max_running_req+1, width]
             (engine/engine.py:450-455; 语义见 engine/engine.py:447 的"store raw locations")
                              |
                              |  DSV4/DSV41 再经 full_to_window 投影一层
                              v
 L1 物理层   model 的 kv_cache_group_specs()  (models/config.py:113-140)
             spec.attn_type --> resolve_pool_class()  (kvcache/__init__.py:27)
```

`AttnType -> 池族 -> 前缀缓存` 的对应关系（前缀缓存由 `cache_type` 选，不由 `AttnType` 选，只有 hybrid/SWA 是强绑定）：

```
 AttnType        pool family                      文件:类
 --------------- --------------------------------- ------------------------------------------
 FULL / 无 spec   MHAKVCache                        kvcache/mha_pool.py:12
 MLA             MLAKVCache                        kvcache/dsa_pool.py:26
 DSA (ratio==1)  DSAKVCache                        kvcache/dsa_pool.py:146
 DSA (ratio>1)   KpoolDSAKVCache                   kvcache/dsa_pool.py:208
 QSA             QSAKVCache                        kvcache/qsa_pool.py:38
 BSA             BSAKVCache                        kvcache/bsa_pool.py:27
 SWA             HybridSWAKVCache                  kvcache/hybrid_swa_pool.py:28
 DSV4            DSV4PagedKVCache                  kvcache/dsv4/v4_pool.py:41
 DSV41           DSV41PagedKVCache                 kvcache/dsv4/v41_pool.py:46
 LINEAR          （无 paged 池）LinearStatePool       kvcache/linear_state_pool.py:34
```

派发顺序是硬编码的优先级链，先命中先返回：DSV4（`kvcache/__init__.py:45-48`）→ DSV41（49-52）→ SWA（53-56）→ DSA（57-65）→ MLA（66-69）→ QSA（70-73）→ BSA（74-77）→ 默认 MHA（78-80）。同一次派发也被 engine 在**池子还不存在时**用于 sizing（`kvcache/__init__.py:28-31`），所以 `kv_cost` 必须是 `@classmethod`（`kvcache/base.py:71-73`）。

## 2. 文件清单表

| 文件 | 行数 | 主要符号 | 职责 |
|---|---:|---|---|
| `kvcache/__init__.py` | 318 | `resolve_pool_class`(27)、`create_kv_pool`(83)、`create_kvcache_pool`(142)、`create_prefix_cache`(301) | AttnType→池族派发、池工厂、前缀缓存注册表 |
| `kvcache/base.py` | 285 | `BaseKVCachePool`(46)、`BaseCacheHandle`(187)、`BasePrefixCache`(217)、`spec_kv_bytes_per_token`(19)、`WindowPoolSpec`(38) | 两套 ABC + 统一 cost/sizing 模板 |
| `kvcache/radix_cache.py` | 268 | `RadixTreeNode`(17)、`RadixPrefixCache`(126)、`_get_key_fn`(265) | 纯 KV 前缀缓存（无第二币种） |
| `kvcache/swa_radix_cache.py` | 455 | `SWARadixCache`(66)、`SWAMatch`(55)、`SWAEvictResult`(61) | SWA 第二币种（滑窗）+ tombstone 纪律 |
| `kvcache/hybrid_radix_cache.py` | 297 | `HybridRadixCache`(55)、`HybridMatch`(43)、`EvictResult`(50) | GDN 快照第二币种，donate-not-copy |
| `kvcache/mha_pool.py` | 148 | `MHAKVCache`(12) | 通用 GQA/MHA 分页池，`layer_ids` 重映射 |
| `kvcache/hybrid_swa_pool.py` | 418 | `HybridSWAKVCache`(28)、`_swa_paged_num_tokens`(408)、`_swa_pool_floor`(400) | full/swa 双 group 池 + 全局分页 swa 分配器 |
| `kvcache/linear_state_pool.py` | 301 | `LinearStatePool`(34)、`state_pool_bytes`(263)、`_linear_pool_num_slots`(280) | GDN 每请求 conv+SSM 状态槽 |
| `kvcache/dsa_pool.py` | 260 | `MLAKVCache`(26)、`DSAKVCache`(146)、`KpoolDSAKVCache`(208) | latent-KV + DSA 索引键 slab |
| `kvcache/qsa_pool.py` | 231 | `QSAKVCache`(38) | GQA 分页 + 压缩索引 slab + pending ring |
| `kvcache/bsa_pool.py` | 126 | `BSAKVCache`(27) | GQA 分页 + 块稀疏索引 slab |
| `kvcache/naive_cache.py` | 45 | `NaivePrefixCache`(16) | 零复用占位实现 |
| `kvcache/cache_status.py` | 194 | `compute_cache_unit_bytes`(16)、`compute_cache_floors`(83)、`compute_cache_pools`(136)、`compute_cache_status_meta`(166) | 桌面缓存面板的就绪期元数据 |
| `kvcache/dsv4/__init__.py` | 1 | - | 空包标记 |
| `kvcache/dsv4/window_tier.py` | 331 | `WindowTierPagedPool`(146)、`FreeListAllocator`(26)、`CompressStateRing`(75)、`window_state_loc`(138) | 滑窗分页池共享基座（DSV4/DSV41 共用） |
| `kvcache/dsv4/v4_pool.py` | 349 | `DSV4PagedKVCache`(41) | window/cmp/idx/state 四族缓冲 |
| `kvcache/dsv4/v4_cost_model.py` | 364 | `dsv4_pool_sizes`(156)、`dsv4_pool_bytes`(208)、`dsv4_solve_num_pages`(229) | DSV4 逐层逐 tier 精确字节模型 |
| `kvcache/dsv4/v41_pool.py` | 292 | `DSV41PagedKVCache`(46) | DSV41 packed byte tiers |
| `kvcache/dsv4/v41_cost_model.py` | 216 | `dsv41_pool_sizes`(75)、`private_window_layer_ids`(48) | DSV41 行格式驱动的字节模型 |
| `kvcache/dsv4/v41_row_format.py` | 69 | `RowFormat`(26)、`row_format`(62) | fp8/fp4 行布局与 scale 字节 |

## 3. 核心抽象

### 3.1 `BaseKVCachePool`（`kvcache/base.py:46`）

| 方法 | 行号 | 契约（谁调用、必须保证什么） |
|---|---|---|
| `window_spec` (classmethod) | 56-58 | 由 `cache_status.window_pool_spec`（`kvcache/cache_status.py:6-9`）调用。返回 `None` = 本池无窗口 tier；SWA 返回 `(1, _swa_pool_floor(config))`（`kvcache/hybrid_swa_pool.py:298-302`），DSV4 返回 `(P, floor_pages-1)`（`kvcache/dsv4/v4_pool.py:220-226`） |
| `window_pages` (property) | 60-63 | 基类恒 0；`cache_status.compute_cache_pools` 用它填 `num_swa_pages`（`kvcache/cache_status.py:151-154`）。必须与 `window_spec.page_size` 同单位 |
| `kv_cost` (classmethod, abstract) | 71-79 | 被 `solve_num_pages`(86)、`validate_rebuild`(127)、engine 自动规划（`engine/engine.py:619`）调用。只准计本族缓冲，GDN 状态与 MoE 由 engine 另加（`engine/engine.py:422,620`） |
| `solve_num_pages` (classmethod) | 81-96 | 启动时唯一入口（`engine/engine.py:423`）。`num_page_override` 优先（87）；结果必须 `>1` 否则 assert（90）。DSV4/DSV41 覆写它做逐 tier 字节二分（`kvcache/dsv4/v4_pool.py:177`、`v41_pool.py:153`） |
| `min_kv_tokens` (classmethod) | 98-101 | 默认一个 page 的 token 数；DSV4 改为窗口工作集 floor（`kvcache/dsv4/v4_pool.py:228-235`）。被 `cache_status.compute_cache_floors` 读作滑块下限（`kvcache/cache_status.py:103-109`） |
| `validate_rebuild` | 103-137 | **任何破坏性 free 之前**的预算校验，失败抛 `CacheRebuildRejected`（`kvcache/base.py:13-16,134-137`）。用 `inspect.signature(kv_cost)` 只转发本族声明过的 target（120-126），所以新池族加参数不必改基类签名 |
| `rebuild_from_config` (abstract) | 139-148 | engine `_resize_kv_pool` 调用（`engine/engine.py:878`）。必须**原地、保持对象身份**，由池自己补 dummy page（`kvcache/mha_pool.py:99`、`hybrid_swa_pool.py:353`、`v4_pool.py:288`） |
| `attach_page_table` | 150-152 | engine 初始化（`engine/engine.py:458`）与每次页表重分配（`engine/engine.py:895`）各调一次。DSV4 在此保存 `full_loc_map`（`kvcache/dsv4/window_tier.py:299-301`） |
| `unit_bytes` (abstract) | 154-157 | 返回 `(kv_bytes_per_token, swa_bytes_per_token)`，供滑块分母。`cache_status.compute_cache_unit_bytes` 直接读（`kvcache/cache_status.py:36-37`） |
| `k_cache` / `v_cache` | 159-163 | attention 后端按**全局 layer_id** 读（`attention/fa.py:72-73`、`attention/triton.py:152-153`）。有 `layer_ids` 的池必须重映射 |
| `store_kv` | 165-172 | attention 后端写当前 forward 的 K/V（`attention/fa.py:69`、`attention/triton.py:150`）。`out_loc` 是 full-pool token loc |
| `device` / `dtype` / `num_layers` | 174-184 | 只读属性；`num_layers` 是**模型**层数，不是 slab 数（`kvcache/mha_pool.py:38,147`） |

`needs_rebind_on_rebuild`（`kvcache/base.py:52-54`）是唯一的"重建副作用"声明：DSV4 系把 tier 缓冲绑进 per-forward model scratch，故为 `True`（`kvcache/dsv4/window_tier.py:163-164`），engine 在 rebuild 前调用 `model.mark_for_rebind()`（`engine/engine.py:876-877`）。

### 3.2 每 token 字节与窗口规格

`spec_kv_bytes_per_token(spec, config)`（`kvcache/base.py:19-35`）：

```
per_token = (1 if spec.mla else 2) * head_dim * div_even(num_kv_heads, tp, allow_replicate=True)
            * dtype.itemsize * num_layers                         # base.py:28-34
          + index_head_dim * num_index_layers * 2 // index_ratio   # base.py:35
```

- MLA 只存一块 slab，所以 V 别名 K，系数 1（`kvcache/base.py:29`、`kvcache/dsa_pool.py:77-79`）。
- 索引键 slab 的 2 字节是硬编码，与 `DSV4/DSA/BSA/QSA` 的 bf16 写入端逐个对齐（`kvcache/base.py:23-24`、`kvcache/dsa_pool.py:180-186`、`kvcache/bsa_pool.py:57-59`、`kvcache/qsa_pool.py:85-87`）。
- `index_ratio > 1` 时该 slab 的 ring/scratch 是定长，改在 `QSAKVCache.kv_cost` 的 `fixed_cache_size` 里计价（`kvcache/qsa_pool.py:173-178`）。

`WindowPoolSpec`（`kvcache/base.py:38-43`）只有两个字段：`page_size`（窗口页粒度）与 `min_pages`（**不含** reserved 页的最小可用容量）。消费方两处：滑块下限 `page_size * min_pages`（`kvcache/cache_status.py:123-125`）与展示用 `num_swa_pages`（`kvcache/cache_status.py:151-154`）。

### 3.3 句柄与结果类型

| 类型 | 行号 | 字段 | 契约 |
|---|---|---|---|
| `BaseCacheHandle` | 187-192 | `cached_len` | 冻结 dataclass；**必须先 lock 再用**（`kvcache/base.py:237`），否则可能被 evict |
| `MatchResult` | 209-214 | `cuda_handle`、`mamba_value=None` | `mamba_value` 只在 hybrid 路径被 `CacheManager.match_req` 透出（`scheduler/cache.py:106-110`） |
| `InsertResult` | 204-206 | `cached_len`、`handle` | `cached_len` 是"插入前已在缓存里的长度，调用方应释放"（`kvcache/base.py:205`），`scheduler/cache.py:313` 据此 free |
| `SizeInfo` | 195-201 | `evictable_size`、`protected_size` | `total_size` = 两者之和（200）；hybrid/SWA 只报 full 币种（`kvcache/hybrid_radix_cache.py:202-207`、`kvcache/swa_radix_cache.py:375-378`） |

### 3.4 `BasePrefixCache`（`kvcache/base.py:217`）

| 方法 | 行号 | 契约（谁调用、必须保证什么） |
|---|---|---|
| `lock_handle` | 219-230 | `scheduler/cache.py:242-257` 分派：SWA 走 `inc_lock`、hybrid 走 `inc_lock`、其余走 `lock_handle`。只改计数，不改树 |
| `match_prefix` | 233-243 | prefill 准入（`scheduler/prefill.py:76`）。不改树（plain 实际会 split+改时间戳，见 §6.1） |
| `insert_prefix` | 246-256 | commit 时调用（`scheduler/cache.py:305`）。会 align_down 到页边界（`kvcache/radix_cache.py:165`），尾部不入树 |
| `evict` | 259-272 | `CacheManager._allocate` 缺页时调用（`scheduler/cache.py:606`）。**evict 0 必须安全**；实际驱逐量可大于请求量；超 `evictable_size` 抛 `RuntimeError`（`kvcache/radix_cache.py:180-184`） |
| `reset` | 274-276 | plain radix 是 `NotImplementedError`（`kvcache/radix_cache.py:208-209`），所以 rebuild 改走"新建对象"（`scheduler/cache.py:556-568`） |
| `size_info` | 278-281 | 指标/水位；假量与 `evictable//page_size` 相加必须等于 `num_pages`（`scheduler/cache.py:544-552`） |
| `check_integrity` | 283-285 | plain 是空实现（`kvcache/radix_cache.py:218-219`）；SWA 检查 `full_ref >= swa_ref`（`kvcache/swa_radix_cache.py:380-385`）；hybrid 检查快照节点必持有 slot（`kvcache/hybrid_radix_cache.py:209-213`） |

## 4. 具体实现

### 4.1 分页池族

三个池共享同一套 `rebuild` 纪律：**先把引用置 None、cuda sync + empty_cache、再重新分配、最后刷新视图**，对象身份不变（`kvcache/mha_pool.py:60-83`、`hybrid_swa_pool.py:268-296`、`linear_state_pool.py:140-169`）。原因写在 `hybrid_swa_pool.py:277-280`：旧新缓冲同时活着会让本来放得下的 rebuild OOM。

**MHAKVCache**（`kvcache/mha_pool.py:12`）

| 缓冲区 | shape | dtype |
|---|---|---|
| `_kv_buffer` | `[2, num_storage_layers, num_pages, page_size, local_kv_heads, head_dim]`（50-54） | 模型 dtype |
| `_k_buffer`/`_v_buffer` | 上述的第 0/1 维切片（55-56） | 同上 |

`layer_ids is None` 时 `num_storage_layers = num_layers`；否则只分配 `len(layer_ids)` 层，并把全局 id 映射到 dense slot，非法 id 在构造时抛错（`kvcache/mha_pool.py:39-49`），读取时未映射的层抛 `KeyError`（106-112）。分配/释放由 `CacheManager` 的 `free_slots` 管，池本身**不做页级 free**。`unit_bytes` = 总字节 // (pages*page_size)（101-104）。

**HybridSWAKVCache**（`kvcache/hybrid_swa_pool.py:28`）

| 缓冲区 | shape | 说明 |
|---|---|---|
| `full_kv_pool.buffer` | `[2, L_full, num_full_pages, page_size, heads, d]`（59-66, 93-97） | full 层按页 |
| `swa_kv_pool.buffer` | `[2, L_swa, swa_num_tokens, 1, heads, d]`（67-74） | swa 层按 token，`inner_size=1` |
| `full_to_swa_index_mapping` | `[full_num_tokens + page_size + 1]` int64（150-155） | 末尾 `-1` 让 `last_loc=-1` 映射到 `-1`；**0 = 无 live swa slot** |
| `_swa_free` | `arange(1, swa_num_tokens)` int32（157） | slot 0 保留为 sentinel |

- `alloc_swa` 从 free-list 头部切片并写映射，不够则 `RuntimeError`（159-172）。
- `free_swa` 过滤 `>0` 后回填 free-list，并把映射重置为 0，因此对已释放槽**幂等**（174-186）。
- `store_kv` 对 swa 层先做 `translate_loc_from_full_to_swa`（203-223）。
- `rebuild` 的最后一步重建映射与 free-list（292-296），保证"新缓冲 + 新空树"自洽。

**LinearStatePool**（`kvcache/linear_state_pool.py:34`）

| 缓冲区 | shape | dtype |
|---|---|---|
| `conv_states` | `[n_layers, num_slots, local_conv_dim, conv_kernel_dim-1]`（67-71） | 模型 dtype |
| `recurrent_states` | `[n_layers, num_slots, local_v_heads, key_head_dim, value_head_dim]`（74-78） | `ssm_state_dtype()`，默认 fp32（18-20） |
| `slot_states[name]` | `[max(1,n_layers), num_slots, *spec.shape]`（98-107） | `spec.dtype or conv dtype` |

索引是 `Req.table_idx`，与 page_table 行号同一生命周期（`kvcache/linear_state_pool.py:36-39`）。`padding_slot=0`，free-list 是 `1..num_slots-1`（95-96），`alloc` 是 LIFO 且会抛 `RuntimeError`（126-132），`free` 接受 int/list/tensor（171-177）。`copy_from` 实现 COW-on-restore（188-194），`rebuild` 是**破坏性**的（140-148）。

**MLAKVCache / DSAKVCache / KpoolDSAKVCache**（`kvcache/dsa_pool.py`）

| 缓冲区 | shape | dtype |
|---|---|---|
| `MLAKVCache._kv_buffer` | `[1, L, num_pages, page_size, 1, latent_dim]`（62-68） | 模型 dtype |
| `DSAKVCache._index_k_buffer` | `[num_index_layers, pages*page_size, index_head_dim]`（174-186） | bf16，zeros |
| `KpoolDSAKVCache._index_k_buffer` | 行数改为 `pages*page_size//ratio + num_req_slots`（230-232） | bf16 |
| `KpoolDSAKVCache._tail_k/_tail_gate` | `[L, num_req_slots, index_ratio, index_head_dim]`（239-245） | bf16 |

- MLA 的 `v_cache` 直接返回 `k_cache`（77-79）；`store_kv` 把 `c_kv | k_rope` 分别写进一行两半（99-102）。
- 两块 slab 在**同一次** `_alloc` 里分配，rebuild 先置 None 再一起重建，避免"一块长了一块没长"（`kvcache/dsa_pool.py:174-177,188-190`）。
- Kpool 的压缩行地址是 `token_slot // index_ratio`，所以页共享/驱逐天然跟随（`kvcache/dsa_pool.py:209-219`）；未闭合的池写到 `cmp_scratch_base + table_idx`（234-237）。

**QSAKVCache**（`kvcache/qsa_pool.py:38`）

| 缓冲区 | shape | dtype |
|---|---|---|
| K/V | 继承 `MHAKVCache`（96-105） | 模型 dtype（强约束 2 字节，85-87） |
| `_cmp_k_buffer` | `[L, pages*page_size//ratio + num_req_slots, index_head_dim]`（119-126） | 模型 dtype |
| `_pending_ring` | `[num_req_slots, L, ring_capacity, index_head_dim]`（127-134） | 模型 dtype |
| `_rope_positions` | `[pages*page_size, 3]` int32，仅 mrope（136-140） | int32 |

`ring_capacity_for(ratio, spec) = ratio * ceil((ratio+spec)/ratio)`（48-51）。`rebuild` 的顺序是"先释放索引 tier → `super().rebuild` → 重新零化 → 重分配索引 tier → 失败时把 K/V 也置 None 再抛"（142-159），这是为了防止"K/V 变大而索引 slab 缺失"的静默错误。

**BSAKVCache**（`kvcache/bsa_pool.py:27`）：MHA 池 + `[L, pages*page_size, index_head_dim]` 的 zeros slab（81-91），rebuild 同样先弃索引再重建并在异常时把 K/V 置 None（93-108）。

### 4.2 三套 radix 对比

三者**只共享** `RadixTreeNode`/`split_at`/`key_fn`，其余各自实现（`tests/kvcache/radix/README.md:3-9`）。

| 维度 | `RadixPrefixCache` | `SWARadixCache` | `HybridRadixCache` |
|---|---|---|---|
| 第二币种 | 无 | 滑窗 swa KV，**不存槽位**：`node.value` 是 full-pool 页号，swa 经池的 full→swa 映射到达（`swa_radix_cache.py:6-11`） | GDN 状态快照槽 `node.mamba_value`（`hybrid_radix_cache.py:5-8`） |
| 第二币种锁 | - | `swa_ref_count` + `swa_uuid`（`swa_radix_cache.py:41-45`） | `mamba_ref_count`（`radix_cache.py:35`） |
| key 构造 | `page_size==1` 取 `x[0].item()`，否则 `tuple(x[:page_size].tolist())`（`radix_cache.py:265-268`） | 复用同一 `_get_key_fn`（`swa_radix_cache.py:72`） | 同（`hybrid_radix_cache.py:65`） |
| 匹配截断 | 仅按页 `align_down`（`radix_cache.py:250`） | 截到"回看最后一个 tombstone 的连续 live 长度 `>= sliding_window_size`"的最深节点（`swa_radix_cache.py:88-135`） | 截到路径上最深的**有 live 快照**节点（`hybrid_radix_cache.py:80-88`） |
| 驱逐 KV | 未锁**叶子** LRU（`radix_cache.py:179-206`） | `evict_full` 同样只打未锁叶子，且级联回收 tombstone 叶子（`swa_radix_cache.py:290-314`） | `evict_full` 同上，并顺带释放该节点的快照（`hybrid_radix_cache.py:151-169`） |
| 驱逐第二币种 | - | `evict_swa`：**内部节点也打**，原地 tombstone（只放 swa）；未锁叶子则两池都放（`swa_radix_cache.py:316-342`） | `evict_mamba`：内部节点原地 tombstone 快照，叶子两样都放（`hybrid_radix_cache.py:171-192`） |
| 墓碑 | 无 | `swa_tombstone`，`split_at` 时两半都继承、uuid 迁到 root 侧（`radix_cache.py:95-101`） | `mamba_value=None` 即"KV-only tombstone"，`split_at` 只会留在 suffix 侧（`radix_cache.py:29-34`） |
| 额外入口 | - | `trim_head_swa`（`swa_radix_cache.py:344-365`） | - |
| 完整性检查 | `pass`（`radix_cache.py:218-219`） | `full_ref >= swa_ref` 且 tombstone 不得持 swa 锁（`swa_radix_cache.py:380-385`） | 快照节点必持 slot id 且计数非负（`hybrid_radix_cache.py:209-213`） |
| 币种隔离 | 不受影响 | 两个计数都默认 0，plain 树结构不变（`radix_cache.py:43-45`） | 同上（`radix_cache.py:34-35`） |

`HybridRadixCache` 还有一个构造期硬约束：`CHUNK_SIZE % page_size == 0`，否则快照边界与 KV 节点边界不重合（`hybrid_radix_cache.py:57-62`）。

### 4.3 `dsv4/`

**`window_tier.py`：三块共享积木**

| 组件 | 行号 | 关键行为 |
|---|---|---|
| `FreeListAllocator` | 26-72 | 以 `page_unit` 为单位、LIFO；`alloc` 从**尾部**切（52-54），单位基址必为 `page_unit` 倍数（27-28,39），不足抛 `RuntimeError`（45-49） |
| `CompressStateRing` | 75-126 | `buffer` 形状 `[n_slots+1, 2*(1+overlap)*head_dim]` fp32（82-97）；最后一行是永久 scratch，每次写后清零 kv 并把 score 填 `-inf`（99-101,111-114）；`get_blocks`/`set_blocks` 用 `page_base + arange(ring_size)` 做块读写（116-126） |
| `WindowTierPagedPool` | 146-322 | full→window 投影、页原子分配、原地 rebuild、ABC 胶水 |

`WindowTierPagedPool` 的坐标纪律：

- `full_to_window`：`[full_token+1]` int64，`-1` = 未映射，最后一行是永久 `-1` 哨兵，所以 `gather(-1)` 安全（`kvcache/dsv4/window_tier.py:150-153,88-90`）。
- `translate_full_to_window` 把 full loc 转 int64 再 gather（182-184）；`cmp_rows` 用 floor 除法得到压缩行，负数保持负（186-189）。
- `bind_window_pages` 断言两个基址都页对齐（191-195）；`unbind_window_pages` 只重置 `>=0` 的项（197-198）。
- `_init_paged_state` 把**最后一个 full 页与最后一个 window 页**永久绑为 dummy 区，free-list 容量 `n_win_slots - P`（213-228）。
- `alloc_swa` 要求传入整页且页内连续升序，断言形如 `fbases[:,None] + arange(P)`（244-258）。
- `free_swa` 按页聚合计数，断言每页都被完整覆盖（`counts == P`），否则抛错（260-279）。
- `prefill_chunk_budget = max(P, (n_win_pages - reserved)//2 * P)`，因为一次批量 prefill 的窗口峰值约 2 倍 chunk（223-228）。

**`v4_pool.py`：四族缓冲**

| 族 | shape | dtype | 行号 |
|---|---|---|---|
| `window_pool[L]` | `[n_win_slots, head_dim]` | bf16 | 98-101 |
| `cmp_pool[L]` | `[cmp_blocks[L] + n_scratch, head_dim]` | bf16 | 123-127 |
| `idx_pool[L]` | `[idx_blocks[L] + n_scratch, index_head_dim]`，仅 ratio==4 | bf16 | 128-137 |
| `state_ring[L]` | `CompressStateRing(n_slots=state_slots[L], ring_size=ring_size_for_ratio(ratio), overlap=(ratio==4))` | fp32 | 153-161 |
| `indexer_state_ring[L]` | 仅 ratio==4，`head_dim=index_head_dim`，`overlap=True` | fp32 | 139-147 |

`ratio==0` 的层四个 tier 全为 `None`（111-120）。`n_scratch` 行永远不发给分配器，只用来接住"本步未闭合的压缩块"的行，使 masked scatter 图安全（`kvcache/dsv4/v4_pool.py:64-69`）。写入全是 `index_copy_`（314-325），读取由 `total_bytes` 严格镜像（327-342 ↔ `v4_cost_model.dsv4_pool_bytes`208-226）。

**`v41_pool.py`：packed byte tiers**

| 族 | shape / 宽度 | dtype | 行号 |
|---|---|---|---|
| `window_pool[L]`（共享层） | `[n_win_slots, _win_row_bytes(args)]` | uint8 | 83-86 |
| `window_pool[L]`（private 层） | `[n_scratch * P, _win_row_bytes(args)]` | uint8 | 83-86 |
| `main_pool[src]` | `[main_rows[src] + n_scratch, _main_row_bytes(args)]` | uint8 | 91-100 |
| `idx_pool[src]` | `[idx_rows[src] + n_scratch, _idx_row_bytes(args)]` | uint8 | 91-100 |
| `state_ring[src]` | `ring_size == compress_ratios[src]`，`overlap=False` | fp32 | 96-100 |

- 三层用三种行格式：`WIN_FMT=fp8_e8m0_b32`、`MAIN_FMT=fp4_e4m3_b16`、`IDX_FMT=fp4_e8m0_b32`（`kvcache/dsv4/v41_cost_model.py:26-28`），行布局是 `[values | scales]`（`v41_row_format.py:10-12`）。
- 私有窗口层（Decoder SWA Bounded Replay 的 decoder 层）不落共享页，槽位 `row*P + pos%P`（`kvcache/dsv4/v41_pool.py:117-124`、`v41_cost_model.py:48-57`）。
- 因此 `prefix_replay_tokens = P`（若存在私有窗口层），前缀命中必须给 prefill 留出最后一个窗口（`kvcache/dsv4/v41_pool.py:110-114`），由 `CacheManager.prefix_replay_tokens` 透出（`scheduler/cache.py:69-72`）。
- 读写都过 Triton 的 `pack_rows` / `unpack_rows`（249-278）。

**成本模型**

| 函数 | 行号 | 作用 |
|---|---|---|
| `dsv4_cache_per_page` | `v4_cost_model.py:67-96` | 把 window（×swa_ratio）/cmp（1/ratio）/idx（1/4）/两个 ring 合成**一个** bytes-per-P-token，只用于预算除法 |
| `dsv4_pool_sizes` | 156-202 | 从锚点 `full_token = num_pages*P` 独立推导每 tier；window 页数 `ceil(round(ratio*full)/P)` 并 cap 到 `num_pages`（164-168） |
| `dsv4_pool_bytes` | 208-226 | `total_bytes` 的精确镜像，含 scratch/sentinel 行 |
| `dsv4_solve_num_pages` | 229-265 | floor 用**页**表达；先二分下界，再倍增上界，最后精确二分（249-265） |
| `dsv4_auto_cost_model` | 271-286 | 仿射 `(per_page, fixed, min_reserve_tokens)`；slack 2GiB 吸收计划-实测偏差（268,284-285） |
| `_dsv4_window_floor_pages` | 315-326 | 唯一硬下限 = `min(prefill_reach_pages, 8) + reserved_window_pages(...)` |
| `DSV41` 对应族 | `v41_cost_model.py:106-172` | 同形，宽度改为行格式；`dsv41_auto_cost_model` 的 reserve 只到 `n0*P`（172） |

`reserved_window_pages` 是两条公式的唯一来源（`kvcache/dsv4/window_tier.py:129-135`），DSV4 侧只是转调（`v4_cost_model.py:30-36`）。

### 4.4 `cache_status.py`

| 函数 | 行号 | 输出 | 关键行为 |
|---|---|---|---|
| `window_pool_spec` | 6-9 | `WindowPoolSpec or None` | 转发到 `resolve_pool_class(...).window_spec(config)`，是"模型无关"的唯一窗口入口 |
| `compute_cache_unit_bytes` | 16-68 | `{kv_bytes_per_token, moe_bytes_per_expert, mamba_bytes_per_slot, swa_bytes_per_token}` | 全部从**真实已分配张量**量出来（29-51）；每项独立 try，失败降级为 0，绝不抛（59-68） |
| `compute_cache_floors` | 83-133 | `{kv_tokens, moe_experts, mamba_slots, swa_tokens}` | kv = `min_kv_tokens`（103-109）；mamba = `_linear_pool_min_slots - 1`（116-121）；swa = `spec.page_size*min_pages`（123-125） |
| `compute_cache_pools` | 136-163 | `{num_pages, page_size, moe_cache_size, num_mamba_slots, swa_page_size, num_swa_pages}` | 就绪时刻的真实几何；mamba 报**可用**槽（减 padding sink，160） |
| `compute_cache_status_meta` | 166-194 | `meta` 大包 | 拼 unit/floors/pools + `free_vram_bytes`（173）+ `swa_full_tokens_ratio`（176-179）+ `cache_budget_bytes`（185-193） |

`_pool_budget_free_vram_bytes`（71-80）读的是 `engine._post_weights_free`：权重加载后、任何 KV 池分配前的空闲显存，因此不受 allocator 缓存/CUDA graph 影响（`kvcache/cache_status.py:72-75`）。

## 5. 函数详解表（按文件分组）

### `base.py`

| 函数/方法 | 行号 | 作用 | 关键行为 | 调用方 |
|---|---:|---|---|---|
| `spec_kv_bytes_per_token` | 19 | 单个 group 的每 token 字节 | MLA 系数 1；索引项 `*2//ratio` | 各池 `kv_cost`（`mha_pool.py:89`、`hybrid_swa_pool.py:325`、`dsa_pool.py:117`、`qsa_pool.py:172`） |
| `solve_num_pages` | 81 | 启动页数求解 | `num_page_override` 短路；assert `>1` | `engine/engine.py:423` |
| `validate_rebuild` | 103 | rebuild 预算前置校验 | 反射 `kv_cost` 签名，只转发非 None target | `engine/engine.py:972` |
| `min_kv_tokens` | 98 | 滑块 KV 下限 | 默认 `page_size` | `cache_status.py:109` |
| `attach_page_table` | 150 | 重指向页表 | DSV4 保存 `full_loc_map` | `engine/engine.py:458,895` |

### `radix_cache.py`

| 函数/方法 | 行号 | 作用 | 关键行为 | 调用方 |
|---|---:|---|---|---|
| `RadixTreeNode.get_match_len` | 81 | 节点 key 与输入的最长公共前缀 | 走 kernel `fast_compare_key`（82-85） | `_tree_walk`(249)、SWA match(112)/insert(167)、hybrid `_walk`(287) |
| `RadixTreeNode.split_at` | 87 | 在 `pos` 处一分为二 | 新节点在 root 侧并继承 `ref_count`/`swa_ref_count`/`swa_tombstone`/`swa_uuid`，suffix 的 uuid 清空（91-104） | 三个 cache 的 walk/match/insert |
| `RadixCacheHandle.get_matched_indices` | 116 | 自叶向根收集 `value` 再反转拼接 | 返回一维 int32 full loc | `scheduler/prefill.py:124`、`scheduler/cache.py:326` |
| `lock_handle` | 141 | 加/解锁 | 每跨一个 `ref_count 0→1` 就在 evictable/protected 之间搬 `node.length`（148-157） | `scheduler/cache.py:249,257` |
| `insert_prefix` | 164 | 插入已 commit 的前缀 | `align_down` 后若已全匹配则不建节点；否则 clone 出 suffix（165-176） | `scheduler/cache.py:305` |
| `evict` | 179 | 按 token 数驱逐 | 叶子堆 + 父节点变叶后回堆（186-204） | `scheduler/cache.py:606` |
| `_collect_leave_nodes_for_evict` | 221 | 收集未锁叶子 | 显式栈，root 不算叶子（222-234） | `evict`(186) |
| `_tree_walk` | 236 | 按页走树 | 一次 `monotonic_ns` 作为本次 tic；部分匹配即 split 并返回（240-262） | `match_prefix`(161)、`insert_prefix`(167) |
| `_get_key_fn` | 265 | 页键函数 | `page_size==1` 走标量 key，省一次 tuple | 三个 cache 的 `__init__` |

### `swa_radix_cache.py`

| 函数/方法 | 行号 | 作用 | 关键行为 | 调用方 |
|---|---:|---|---|---|
| `match_prefix` | 88 | full 匹配 + 滑窗截断 | 遇 tombstone 先尝试在**父节点**提交边界再清零 live 计数（105-111）；循环后再判一次（129-131）；最后 `_stamp_path`（133） | `scheduler/cache.py:104,406,472` |
| `insert` | 137 | 插入并复活 tombstone | 三支：整节点复活 / 边界落在节点内则 split 后复活尾 / 仍出窗则只丢弃请求副本（174-206）；suffix 先 tombstone 再补一个 live 叶（217-228） | `scheduler/cache.py:442` |
| `_add_child` | 231 | 建子节点并记账 | tombstone 子节点只计 full_evictable | `insert`(224,228) |
| `inc_lock` | 243 | 双币种加锁 | full 每节点 +1；swa 只在窗口预算未满且非 tombstone 时 +1；首次覆盖整窗时分配 `swa_uuid`（250-267） | `CacheManager.lock`（`scheduler/cache.py:245`） |
| `dec_lock` | 270 | 双币种解锁 | 遇到持有本次 `swa_uuid` 的节点即停止 swa 递减（285-286） | `scheduler/cache.py:253` |
| `evict_full` | 290 | 驱逐 full KV | 未锁叶子；tombstone 叶只放 full，live 叶两池都放；级联回收（294-314） | `scheduler/cache.py:596` |
| `evict_swa` | 316 | 驱逐 swa | 内部节点/被 full 锁住的叶子原地 tombstone；未锁叶子两池都放并 unlink（321-342） | `CacheManager.ensure_swa_slots`（`scheduler/cache.py:136`） |
| `trim_head_swa` | 344 | 精确回收 prompt 头部的 swa | 先 `match_prefix` 强制切出边界（353），只打非叶、未锁、非 tombstone 的节点（360-363） | `scheduler/cache.py:471` |
| `check_integrity` | 380 | 结构不变量 | `full_ref >= swa_ref`；tombstone 不得有 swa 锁 | `CacheManager.check_integrity`（531） |
| `_cascade_swa_tombstone_leaves` | 393 | 叶删除后向上清 tombstone 叶 | 返回存活祖先，调用方必须替换原 parent（399-408） | `evict_full`(307)、`evict_swa`(335) |
| `_stamp_path` | 410 | 盖章 LRU | 事件 id ×`2^24` 后向 root 递减（414-420） | `match_prefix`(133) |

### `hybrid_radix_cache.py`

| 函数/方法 | 行号 | 作用 | 关键行为 | 调用方 |
|---|---:|---|---|---|
| `match_prefix` | 76 | 匹配后向上找最近快照 | 从叶向根回退，边回退边减 `end_len`（82-88） | `scheduler/cache.py:108` |
| `insert` | 90 | 插入 KV 并 donate 快照 | 到 root 或已有快照时返回 `mamba_exist=True` 让调用方释放（108-111）；否则挂上并把 `mamba_evictable` +1（112-114） | `scheduler/cache.py:362,376,399` |
| `inc_lock` | 118 | 快照 + full 路径加锁 | 先锁快照再锁路径，保证 `full_ref >= mamba_ref`（122-133） | `scheduler/cache.py:247` |
| `dec_lock` | 135 | 对称解锁 | 快照计数先减，再走路径 | `scheduler/cache.py:255` |
| `evict_full` | 151 | 逐叶驱逐 KV | 释放节点快照 + 级联（154-169） | `scheduler/cache.py:601` |
| `evict_mamba` | 171 | 按个数驱逐快照 | 叶子两样都放，内部节点只扔快照（183-191） | `scheduler/cache.py:147` |
| `check_integrity` | 209 | 快照节点自洽 | 有值即有 slot、计数非负 | `scheduler/cache.py:519` |
| `_cascade_tombstone_leaves` | 228 | 叶删除后清 KV-only 叶 | 保持"叶子必带 live 快照"不变量 | `evict_full`(165)、`evict_mamba`(188) |
| `_walk` | 278 | 走树 | 与 `_tree_walk` 同形，但 tic 语义相同（281-294） | `match_prefix`(80)、`insert`(98) |

### `__init__.py`

| 函数 | 行号 | 作用 | 关键行为 | 调用方 |
|---|---:|---|---|---|
| `resolve_pool_class` | 27 | AttnType → 池类 | 无 `kv_cache_group_specs` 时按 `dsv4_args` 兜底（34-42）；优先级链见 §1 | `create_kv_pool`(92)、engine sizing、`cache_status.window_pool_spec`(9) |
| `create_kv_pool` | 83 | 统一建池入口 | DSV4/DSV41 自带 `_init_paged_state`（104,119）；SWA 先算 `num_swa_tokens`（122-130）；所有族 `num_pages+1` 补 dummy（97,113,133） | `engine/engine.py:425` |
| `create_kvcache_pool` | 142 | 非 DSV4 分派 | hybrid-linear 断言只有 1 个 paged group（174-179）；QSA/Kpool 强制要 `num_req_slots`（211-212,250-251） | `create_kv_pool`(131) |
| `create_prefix_cache` | 301 | 按 `cache_type` 建前缀缓存 | 走 `SUPPORTED_CACHE_MANAGER` 注册表（281-292） | `CacheManager._make_prefix_cache`（`scheduler/cache.py:96`） |

### 池文件

| 函数/方法 | 行号 | 作用 | 关键行为 | 调用方 |
|---|---:|---|---|---|
| `MHAKVCache.__init__` | `mha_pool.py:25` | 建 KV slab | `layer_ids` 生成 `_layer_map`，越界抛 `ValueError`（44-49） | `create_kvcache_pool` |
| `MHAKVCache.rebuild` | 60 | 原地改页数 | 几何从旧 buffer 读，只改页数（67-83） | `rebuild_from_config`(96) |
| `HybridSWAKVCache.alloc_swa` | `hybrid_swa_pool.py:159` | full slot → swa slot | 从 free-list 头切，写映射（170-172） | `CacheManager.allocate_paged`（`scheduler/cache.py:283`） |
| `HybridSWAKVCache.free_swa` | 174 | 归还 swa slot | 过滤 `>0`，映射清 0（182-186） | `scheduler/cache.py:263` |
| `_swa_per_req_swa_floor` | 379 | 单请求不可回收 swa | `locked + window + interval + 2*ps`，special-token 再加一项（389-397） | `_swa_pool_floor`(405) |
| `_swa_paged_num_tokens` | 408 | radix swa 池容量 | `max(floor, pin|ratio*full) + 1`（413-418） | `create_kv_pool`(127)、`rebuild_from_config`(348) |
| `LinearStatePool.alloc` | `linear_state_pool.py:126` | LIFO 取槽 | 不足抛 `RuntimeError` | `scheduler/prefill.py:130-131` |
| `LinearStatePool.copy_from` | 188 | 快照 COW | 三类张量逐层拷贝 | `snapshot_toolcall_anchor`（`scheduler/cache.py:176`） |
| `state_pool_bytes` | 263 | 状态池总字节 | 无 linear group 但有 slot_states 则抛 | `engine/engine.py:422,620,978` |
| `QSAKVCache.ring_capacity_for` | `qsa_pool.py:48` | ring 深度 | `ratio*ceil((ratio+spec)/ratio)` | `kv_cost`(176) |
| `KpoolDSAKVCache.cmp_scratch_base` | `dsa_pool.py:234` | scratch 起始行 | = 影子行数，请求 `table_idx` 偏移 | attention 稀疏打分 |
| `BSAKVCache.rebuild` | `bsa_pool.py:93` | 重建两块 slab | 异常时把 K/V 也置 None（102-108） | `MHAKVCache.rebuild_from_config`(96) |

### `cache_status.py` / `dsv4/`

| 函数 | 行号 | 作用 | 关键行为 | 调用方 |
|---|---:|---|---|---|
| `window_pool_spec` | `cache_status.py:6` | 取窗口规格 | 唯一不依赖具体池类的窗口入口 | `server/stats.py:123`、`server/api_server.py:566`、`scheduler/scheduler.py:740` |
| `compute_cache_status_meta` | 166 | 面板大包 | 各段独立降级 | `server/launch.py:98` |
| `FreeListAllocator.alloc` | `dsv4/window_tier.py:41` | 取 n 个页单位 | 从尾部切，保证基址页对齐（52-54） | `WindowTierPagedPool.alloc_swa`(256) |
| `WindowTierPagedPool._init_paged_state` | 213 | 建窗口 free-list + dummy | 末页永久绑定（221-222），并算 chunk 预算（226-228） | `create_kv_pool`(104,119)、`rebuild`(297) |
| `WindowTierPagedPool.alloc_swa` | 244 | full 页 → window 页 | 断言整页升序（252-255） | `CacheManager.allocate_paged`（`scheduler/cache.py:283`） |
| `WindowTierPagedPool.free_swa` | 260 | 归还 window 页 | 断言整页（272-273），对已解绑页幂等 | `scheduler/cache.py:263` |
| `window_state_loc` | 138 | 由 window slot 派生 ring slot | `(ws//P)*ring_size + ws%ring_size`，`-1` 保持（141-143） | `WindowTierPagedPool.state_loc`(200) |
| `DSV4PagedKVCache.validate_rebuild` | `v4_pool.py:237` | DSV4 专用预算校验 | 低于 window floor 直接 reject（251-257） | `engine/engine.py:972` |
| `DSV4PagedKVCache.solve_num_pages` | 177 | 逐 tier 求解 | override 低于 floor 抛 `ValueError`（204-210） | engine startup |
| `dsv4_pool_bytes` | `v4_cost_model.py:208` | 精确字节 | 与 `total_bytes` 严格镜像 | `solve`(250,257)、`auto`(281) |
| `private_window_layer_ids` | `v41_cost_model.py:48` | 私有窗口层集合 | `swa_decoder_replay == "exact"` 时为空（55-57） | `v41_pool.__init__`(65)、cost model |

## 6. 关键流程

### 6.1 `_tree_walk`：按页匹配 → align_down → 分裂 → LRU 盖章

**数据结构变化**：命中的每个整节点 `timestamp` 被重写为本次 tic；部分命中时树中多出一个节点（root 侧为新建节点，suffix 为原节点）。

1. 取本次事件时间戳，一次调用只有一个 tic（`kvcache/radix_cache.py:240`）。
2. 以 `input_ids[prefix_len:]` 的**页键**查子节点，缺失即返回（243-245）。
3. 逐节点比对得到 `match_len`，再 `align_down` 到页（249-250）。
4. 若 `match_len != node.length`，在 `match_len` 处分裂，新前缀节点盖上本次 tic 后返回（254-257）。
5. 整节点命中则只更新 `timestamp`，继续下降（260）。

```python
# kvcache/radix_cache.py:249-257
match_len = node.get_match_len(input_ids[prefix_len:])
match_len = align_down(match_len, self.page_size)
prefix_len += match_len
if match_len != node.length:
    node = node.split_at(match_len)
    node.timestamp = tic
    return node, prefix_len
```

注意"匹配"这一步**会改树**（split + 时间戳），与 `BasePrefixCache.match_prefix` 文档里"不修改缓存"的措辞不同（`kvcache/base.py:236-237` vs `radix_cache.py:254-260`）。测试也据此把一次调用内所有节点视为同 age，LRU 断言必须用确定性时钟（`tests/kvcache/radix/README.md:38-40`）。

### 6.2 lock / unlock：evictable ↔ protected 计数如何防驱逐

**数据结构变化**：`ref_count` 沿 node→root 每个节点 +1/-1；`evictable_size` 与 `protected_size` 互补搬移，总量不变。

1. 加锁时每个 `ref_count == 0` 的节点把 `length` 从 evictable 搬到 protected（`kvcache/radix_cache.py:153-157`）。
2. 解锁时反向搬回，并断言计数不为负（144-151）。
3. root 的 `ref_count` 初始为 1，因此 root 永远不会成为驱逐候选（138-139）。
4. `evict` 的断言要求节点 `ref_count == 0`（196），所以任何仍被 handle 引用的路径都不可驱逐。
5. SWA 的双层锁要求先满足 full 再满足 swa，窗口边界用 `swa_uuid` 记住（`kvcache/swa_radix_cache.py:256-267`）。

```python
# kvcache/radix_cache.py:153-157
while not node.is_root():
    if node.ref_count == 0:
        self.evictable_size -= node.length
        self.protected_size += node.length
    node.ref_count += 1
    node = node.parent
```

`CacheManager` 的准入会"先 lock 再复核 available_size"，第二次不足则立刻 unlock（`scheduler/prefill.py:87-90`），因为 lock 本身会让可用量下降。

### 6.3 驱逐：时间戳堆 + 叶子级联

**数据结构变化**：叶子从 `parent.children` 删除；被删除叶的父节点若变叶且未锁，则**重新入堆**；`evictable_size` 逐个减去节点长度。

1. 收集所有 `ref_count == 0` 的叶子（`kvcache/radix_cache.py:221-234`）。
2. `heapify` 后按 `timestamp` 弹出最小者（179-187）。`__lt__` 只比 timestamp（108-109）。
3. 断言候选仍是"未锁叶子且非 root"，然后累加长度、收集 `value`（196-199）。
4. 从父节点的 children 删除自己（201）。
5. 父变叶且未锁则回堆（203-204），从而继续向上级联。

```python
# kvcache/radix_cache.py:195-204
node = heapq.heappop(leave_nodes)
assert node.ref_count == 0 and node.is_leaf() and not node.is_root()
evicted_size += node.length
evicted_indices.append(node.value)
self.evictable_size -= node.length
parent = node.parent
del parent.children[self.key_fn(node._key)]
if parent.is_leaf() and parent.ref_count == 0:
    heapq.heappush(leave_nodes, parent)
```

SWA/hybrid 多一层：级联回收要返回**存活的祖先**，调用方必须用它而不是原来的 parent，否则会 double-free 或 `KeyError`（`kvcache/swa_radix_cache.py:307-312`、`hybrid_radix_cache.py:165-168`）。SWA 的 LRU 顺序还被 `_stamp_path` 刻意做成"向 root 递减"，让靠近 root 的 SWA 节点先被回收（`swa_radix_cache.py:410-420`、`swa_radix_cache.py:17-19`）。

### 6.4 SWA tombstone 与 `match_since_tomb >= sliding_window_size` 边界

**数据结构变化**：`swa_tombstone=True` 的节点 full KV 保留、swa KV 视为已释放；`match_since_tomb` 是"自上一个 tombstone 以来连续 live 的 token 数"，初值 `+inf` 表示"到 root 全程无 tombstone"。

1. 初值 `inf`，候选边界停在 root（`kvcache/swa_radix_cache.py:95-99`）。
2. 遇到 tombstone 子节点：**若此前累计 `>= window`**，就在当前节点（tombstone 的父）提交边界；然后把计数清零（105-111）。
3. 累计只在非 tombstone 节点上增加（118-119,124-125）。
4. 循环结束后再判一次，覆盖"路径走完仍无 tombstone"的情形（129-131）。
5. 返回的 `kv` 只截到 `best_value_len`，`cached_len = kv.numel()`（134-135）。

```python
# kvcache/swa_radix_cache.py:105-111
if child.swa_tombstone:
    if match_since_tomb >= self.sliding_window_size:
        best_value_len = len(value)
        best_node = node
    match_since_tomb = 0
```

边界必须是 `>=` 而不是 `>`：恰好一个窗口的 live run 就足以覆盖一窗查询。测试 README 记录了这个 `>` 变异曾让整段匹配塌成 0（静默多算 prefill，答案仍正确），并给出唯一能钉住它的场景（`tests/kvcache/radix/README.md:53-69`）。

### 6.5 hybrid：GDN 快照如何把匹配截断到最深可恢复边界

**数据结构变化**：`_walk` 得到的叶节点不一定有快照；向上回退直到遇到第一个 `mamba_value is not None` 的节点，返回它的 KV 路径与该槽位。

1. `_walk` 得到最深匹配节点与路径（`kvcache/hybrid_radix_cache.py:80`）。
2. `end_len = _path_len(node)`（82），即该节点结束处的绝对长度。
3. 自叶向根：有快照就立刻返回 `(kv, end_len, snapshot, node)`（83-85）。
4. 否则 `end_len -= cur.length` 并上移（86-87）。
5. 到 root 仍无快照则返回空匹配、`mamba_value=None`（88）。

```python
# kvcache/hybrid_radix_cache.py:83-88
while not cur.is_root():
    if cur.mamba_value is not None:
        return HybridMatch(self._collect_kv(cur), end_len, cur.mamba_value, cur)
    end_len -= cur.length
    cur = cur.parent
return HybridMatch(self.empty, 0, None, self.root)
```

快照只挂在页对齐边界：`CHUNK_SIZE % page_size == 0` 是构造期断言（57-62），提交侧还会拒绝非页对齐的边界（`scheduler/cache.py:391-396`），因为把"编码了 L 个 token 的状态"挂到更短的节点上会让下次 COW 恢复出超前状态。donate 是零拷贝的：树接手槽位后调用方不再 free（`scheduler/cache.py:414-419`）。

### 6.6 DSV4 虚拟地址投影：full_to_window + 页原子 free list + 派生环槽位

**数据结构变化**：分配一页时 `full_to_window[full_page]` 从 `-1` 变成 window 页基址加偏移；释放时整页回到 `FreeListAllocator` 并把映射重置为 `-1`；ring slot 从不存储，每次从 window slot 现算。

1. 页表把 `(table_idx, pos)` 映射到 full loc（`engine/engine.py:446-458`）。
2. `translate_full_to_window` 用 int64 gather 得到 window slot，`-1` 落到哨兵行（`kvcache/dsv4/window_tier.py:182-184`）。
3. `alloc_swa` 按整页取基址，写 `[base, base+P)` 的偏移映射（244-258）。
4. `free_swa` 按页聚合、断言整页覆盖、重置映射并归还页（260-279）。
5. 压缩行由算术得到：`full_loc // ratio`（186-189）；ring slot 由 window slot 派生（138-143）。

```python
# kvcache/dsv4/window_tier.py:255-258
assert torch.equal(fi, fbases[:, None] + torch.arange(P, device=self._device)), "alloc_swa pages must be contiguous ascending"
wbases = self._win_alloc.alloc(fbases.numel())
offsets = torch.arange(P, dtype=torch.int64, device=self._device)
self.full_to_window[(fbases[:, None] + offsets).flatten()] = (wbases[:, None] + offsets).flatten()
```

```python
# kvcache/dsv4/window_tier.py:141-143
pages = torch.div(window_slot, P, rounding_mode="floor")
loc = pages * ring_size + (window_slot % ring_size)
return torch.where(window_slot < 0, torch.full_like(loc, -1), loc)
```

`ring_size | P` 保证不同页落在不相交的 ring 块（`window_tier.py:139-140`），所以 per-page 的 carry 块读写在并发页之间不会串（116-126）。dummy 页在 `_init_paged_state` 里被永久绑定，因此 CUDA graph 补齐的行总能落到真实槽位（213-222）。

## 7. 与其他模块的关系

| 模块 | 关系 | 证据 |
|---|---|---|
| `scheduler/cache.py` | **主要调用方**。`CacheManager` 持有 `free_slots`、`page_table`、`prefix_cache`，负责"虚拟页号 ↔ 前缀缓存 ↔ 第二币种池"的全部事务 | `scheduler/cache.py:32-63`；`match_req`(98)、`lock`(242)、`allocate_paged`(265)、`cache_req`(286)、`_allocate`(591) |
| `scheduler/prefill.py` | 准入侧：match → lock → 复核 available_size → 写 page_table 尾部 | `scheduler/prefill.py:76-125` |
| `attention/*` | 读路径：按全局 layer_id 取 `k_cache/v_cache`，按 `out_loc` 写；SWA 层经 `translate_loc_from_full_to_swa` | `attention/fa.py:69-73`、`attention/triton.py:150-153,251`、`attention/qsa_sparse.py:290-334`、`attention/m3_sparse.py:256-281` |
| `engine/engine.py` | sizing（`solve_num_pages`）、建池（`create_kv_pool`）、rebuild（`rebuild_from_config` + `validate_rebuild`）、页表重挂 | `engine/engine.py:421-425`、`870-878`、`881-895`、`972-982` |
| `engine/cache_budget.py` | `validate_rebuild` 的预算公式来源 | `engine/cache_budget.py:31-45` 被 `kvcache/base.py:120,128-132` 调用 |
| `models/` | `kv_cache_group_specs()` 决定池族、每层几何、`index_*`、`attn_type` | `models/config.py:113-140`；消费点 `kvcache/__init__.py:34-44`、`kvcache/mha_pool.py:89-93` |
| `kernel/` | `store_cache`、`fast_compare_key`、DSV41 `pack_rows/unpack_rows` | `kvcache/mha_pool.py:127-136`、`radix_cache.py:82-85`、`dsv4/v41_pool.py:250-278` |
| `server/` | 就绪期把 `compute_cache_status_meta` 塞进 readiness 响应；CLI 的 cache-type 取注册表 | `server/launch.py:98`、`server/args.py:133` |
| `tests/kvcache/radix/` | 三套 radix 的行为规范 + 变异测试门 | `tests/kvcache/radix/README.md:1-73` |

## 8. 常见坑

1. **key 按页不按 token。** `_get_key_fn` 在 `page_size>1` 时取前 `page_size` 个 id 组成 tuple（`kvcache/radix_cache.py:265-268`），树的重用单位是整页；token 级 trie 在 `page_size>1` 时是错的（`tests/kvcache/radix/README.md:23-27`）。
2. **insert 会静默截断。** `align_down(len(input_ids), page_size)` 之后的尾部永远不入树（`kvcache/radix_cache.py:165-166`），尾部由调用方按 `finished` 决定保留或释放（`scheduler/cache.py:314-330`）。
3. **rebuild 会新建前缀缓存对象。** `RadixPrefixCache.reset` 是 `NotImplementedError`（`kvcache/radix_cache.py:208-209`），所以 `CacheManager.rebuild` 直接 `_make_prefix_cache` 重建，并顺带 `reclaim_all_slots`（`scheduler/cache.py:556-572`）。任何跨 rebuild 缓存旧树引用的代码都会读到一个孤儿对象。
4. **`hybrid_radix` 不是用户可见的 cache-type。** 它只由 `_resolve_cache_type` 内部产出（`kvcache/__init__.py:295-298`）；用户选 `radix` 或 `naive`。
5. **所有池的 rebuild 都是 idle-only 且破坏性。** `LinearStatePool.rebuild` 明文要求"无请求持有槽位且旧树已丢弃"（`kvcache/linear_state_pool.py:144-148`），`HybridSWAKVCache` 的映射重建也只保证"与新空树自洽"（`hybrid_swa_pool.py:292-296`）。
6. **dummy page 的 `+1` 有两处必须一致。** 建池时 `num_pages+1`（`kvcache/__init__.py:97,113,133`），rebuild 时各池自己再 `+1`（`mha_pool.py:99`、`hybrid_swa_pool.py:353`、`v4_pool.py:288`）。漏掉一处就会少一页。
7. **`layer_ids` 是全局 id → dense slot 的重映射。** 对未映射的层 `k_cache` 抛 `KeyError` 而不是静默返回错层（`kvcache/mha_pool.py:106-112`）；`num_layers` 仍报模型层数（147）。
8. **swa 的 0 是"无 live slot"，不是合法槽位。** 映射初值 0、free-list 从 1 开始（`kvcache/hybrid_swa_pool.py:150-157`），`translate_loc_from_full_to_swa` 直接把 0 交给 gather（132-136）。任何把 0 当真实槽的代码都会静默读到 sentinel。
9. **`page_table` 存 raw token loc。** 注意力侧要自己按 `page_size` 切片再除（`attention/fa.py:108-113`），`CacheManager.free_slots` 也是 `page*page_size` 形式（`scheduler/cache.py:38`）。
10. **`size_info` 在 hybrid/SWA 下只报 full 币种。** 第二币种走 `mamba_evictable_size` / `swa_evictable_size`（`kvcache/hybrid_radix_cache.py:202-207`、`swa_radix_cache.py:375-378`），用 `size_info` 算 swa 水位会得到错数。
11. **rebuild 后旧视图失效。** 三个池的 `rebuild` 都会重建 `_k_buffer`/`_v_buffer` 等视图（`kvcache/mha_pool.py:70-83`），调用方必须每 forward 重新取（`linear_state_pool.py:42-44` 明说）。
12. **`prefill_chunk_budget` 会静默限制 chunk。** DSV4 把它设为约半个窗口工作集（`kvcache/dsv4/window_tier.py:223-228`），`CacheManager` 通过 `getattr` 拾取（`scheduler/cache.py:56-57`）；不存在时保持 `None`（67）。

### 自测问题

1. `page_size=4` 时，两个 prompt 的 token 序列前 4 个 token 相同但第 5 个不同，它们能共享 radix 节点吗？为什么 `_get_key_fn` 要这样设计？（提示：`kvcache/radix_cache.py:265-268`、`tests/kvcache/radix/README.md:23-27`）
2. 一个 SWA 请求在 decode 中，`match_prefix` 返回的 `cached_len` 为什么可能小于树的实际匹配长度？`match_since_tomb` 的初值为什么必须是 `inf`？（`kvcache/swa_radix_cache.py:95-111`、`README.md:53-69`）
3. `HybridRadixCache.insert` 在什么情况下返回 `mamba_exist=True`，返回值分别对应调用方的什么动作？（`kvcache/hybrid_radix_cache.py:108-115`、`scheduler/cache.py:364,380`）
4. 为什么 `rebuild_from_config` 要求"先置 None 再分配"，而不是直接赋新张量？举出两个池里的对应代码。（`kvcache/mha_pool.py:70-75`、`hybrid_swa_pool.py:281-288`）
5. DSV4 的 `alloc_swa` 为什么必须收整页且升序，而 `HybridSWAKVCache.alloc_swa` 没有这个要求？（`kvcache/dsv4/window_tier.py:252-255` vs `hybrid_swa_pool.py:159-172`；再想 `window_state_loc` 的 `ring_size | P` 前提）
6. `CacheManager.check_integrity` 对 SWA 用的是 `==` 而不是 `<=`，这个选择分别能抓住哪两类 bug？（`scheduler/cache.py:533-543`）
