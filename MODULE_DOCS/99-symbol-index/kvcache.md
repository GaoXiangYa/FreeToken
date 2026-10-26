# 符号索引：`kvcache`

自动生成（`ast` 解析，不执行代码）。**20 个文件 / 4968 行 / 404 个符号**。

> 用途：确认某个模块里**到底有哪些**类/函数/方法及其行号，避免遗漏。
> 行为详解请看同目录下的手写模块文档。
> 重新生成：`python3 MODULE_DOCS/tools/gen_symbol_index.py`

| 文件 | 行数 | 符号数 |
|---|---:|---:|
| `freetoken/kvcache/__init__.py` | 318 | 8 |
| `freetoken/kvcache/base.py` | 285 | 33 |
| `freetoken/kvcache/bsa_pool.py` | 126 | 8 |
| `freetoken/kvcache/cache_status.py` | 194 | 14 |
| `freetoken/kvcache/dsa_pool.py` | 260 | 31 |
| `freetoken/kvcache/hybrid_radix_cache.py` | 297 | 24 |
| `freetoken/kvcache/hybrid_swa_pool.py` | 418 | 35 |
| `freetoken/kvcache/linear_state_pool.py` | 301 | 27 |
| `freetoken/kvcache/mha_pool.py` | 148 | 13 |
| `freetoken/kvcache/naive_cache.py` | 45 | 12 |
| `freetoken/kvcache/qsa_pool.py` | 231 | 16 |
| `freetoken/kvcache/radix_cache.py` | 268 | 26 |
| `freetoken/kvcache/swa_radix_cache.py` | 455 | 25 |
| `freetoken/kvcache/dsv4/__init__.py` | 1 | 0 |
| `freetoken/kvcache/dsv4/v41_cost_model.py` | 216 | 17 |
| `freetoken/kvcache/dsv4/v41_pool.py` | 292 | 28 |
| `freetoken/kvcache/dsv4/v41_row_format.py` | 69 | 7 |
| `freetoken/kvcache/dsv4/v4_cost_model.py` | 364 | 18 |
| `freetoken/kvcache/dsv4/v4_pool.py` | 349 | 19 |
| `freetoken/kvcache/dsv4/window_tier.py` | 331 | 43 |

---

## `freetoken/kvcache/__init__.py`  (318 行)

| 行 | 符号 | 说明 |
|---:|---|---|
| 20 | **class `CacheManagerCreator`** | class CacheManagerCreator(Protocol) |
| 21 | **class `__call__`** | __call__(self, device |
| 27 | `resolve_pool_class()` | attn_type -> KV pool family, the dispatch shared by ``create_kv_pool`` and the |
| 83 | `create_kv_pool()` | Build the engine's KV pool for ``num_pages`` USABLE pages (the dummy page and every |
| 142 | `create_kvcache_pool()` | create_kvcache_pool( model_config |
| 282 | `create_naive_cache()` | create_naive_cache(device |
| 289 | `create_radix_cache()` | create_radix_cache(device |
| 301 | `create_prefix_cache()` | create_prefix_cache( device |

## `freetoken/kvcache/base.py`  (285 行)

| 行 | 符号 | 说明 |
|---:|---|---|
| 13 | **class `CacheRebuildRejected`** | A runtime cache rebuild was rejected BEFORE any destructive free (e.g. the |
| 19 | `spec_kv_bytes_per_token()` | One paged-KV group's bytes per token: (1\|2 slabs) x head_dim x local kv heads x dtype |
| 39 | **class `WindowPoolSpec`** | Window resize units and minimum usable capacity, excluding reserved pages. |
| 46 | **class `BaseKVCachePool`** | Base class for key-value caches. |
| 57 | **class `window_spec`** | window_spec(cls, config) -> WindowPoolSpec \| None |
| 61 | **class `window_pages`** | Current usable window pages in the units declared by window_spec. |
| 73 | **class `kv_cost`** | ``(cache_per_page, fixed_cache_size, page_tokens, min_reserve_tokens)`` for this |
| 82 | **class `solve_num_pages`** | Largest usable page count fitting ``available_memory`` (bytes measured by the |
| 99 | **class `min_kv_tokens`** | min_kv_tokens(cls, config) -> int |
| 103 | **class `validate_rebuild`** | Budget fit-check for a runtime rebuild target, BEFORE any destructive free. |
| 140 | **class `rebuild_from_config`** | Resize to ``num_pages`` USABLE pages in place, identity-preserving (callers keep |
| 150 | **class `attach_page_table`** | Called once at engine init and again on every page-table realloc. Pools routed by |
| 155 | **class `unit_bytes`** | ``(kv_bytes_per_token, swa_bytes_per_token)`` of the live pool -- the per-unit VRAM |
| 160 | **class `k_cache`** | k_cache(self, index |
| 163 | **class `v_cache`** | v_cache(self, index |
| 166 | **class `store_kv`** | store_kv( self, k |
| 176 | **class `device`** | device(self) -> torch.device |
| 180 | **class `dtype`** | dtype(self) -> torch.dtype |
| 184 | **class `num_layers`** | num_layers(self) -> int |
| 188 | **class `BaseCacheHandle`** | class BaseCacheHandle(ABC) |
| 192 | **class `get_matched_indices`** | get_matched_indices(self) -> torch.Tensor |
| 195 | **class `SizeInfo`** | class SizeInfo(NamedTuple) |
| 200 | **class `total_size`** | total_size(self) -> int |
| 204 | **class `InsertResult`** | class InsertResult(NamedTuple) |
| 209 | **class `MatchResult`** | class MatchResult(NamedTuple) |
| 217 | **class `BasePrefixCache`** | class BasePrefixCache(ABC) |
| 219 | **class `lock_handle`** | Lock or unlock a cache handle. |
| 233 | **class `match_prefix`** | Match prefix and return the indices of the matched prefix in the cache. |
| 246 | **class `insert_prefix`** | Insert a new prefix into the cache. |
| 259 | **class `evict`** | Evict some prefixes from the cache to free up space. |
| 275 | **class `reset`** | Reset the cache manager and the underlying cache. |
| 280 | **class `size_info`** | Get the size information of the cache. |
| 284 | **class `check_integrity`** | Check the integrity of the cache. Raise an error if the cache is corrupted. |

## `freetoken/kvcache/bsa_pool.py`  (126 行)

| 行 | 符号 | 说明 |
|---:|---|---|
| 27 | **class `BSAKVCache`** | MHA paged pool + the block-sparse index-key slab. |
| 37 | **class `__init__`** | __init__( self, num_kv_heads |
| 74 | **class `_zero_kv_slabs`** | _zero_kv_slabs(self) -> None |
| 81 | **class `_alloc_index_slab`** | _alloc_index_slab(self, num_pages |
| 93 | **class `rebuild`** | rebuild(self, num_pages |
| 110 | **class `unit_bytes`** | unit_bytes(self) -> tuple[int, int] |
| 118 | **class `index_k_cache`** | Row-flat index keys for a sparse-layer slot: ``[rows, index_head_dim]``. |
| 122 | **class `store_index_k`** | store_index_k(self, k |

## `freetoken/kvcache/cache_status.py`  (194 行)

| 行 | 符号 | 说明 |
|---:|---|---|
| 6 | `window_pool_spec()` | window_pool_spec(config) |
| 12 | `_supports_swa_ratio()` | _supports_swa_ratio(config) -> bool |
| 16 | `compute_cache_unit_bytes()` | Per-unit VRAM byte cost of each runtime cache, measured from the real allocated pool |
| 29 | &nbsp;&nbsp;&nbsp;↳ `_kv_swa()` | _kv_swa() -> tuple[int, int] |
| 39 | &nbsp;&nbsp;&nbsp;↳ `_moe()` | _moe() -> int |
| 47 | &nbsp;&nbsp;&nbsp;↳ `_mamba()` | _mamba() -> int |
| 71 | `_pool_budget_free_vram_bytes()` | The pool-budget baseline captured in ``Engine.__init__``: free VRAM after weights |
| 83 | `compute_cache_floors()` | Per-pool minimum (floor) unit counts the runtime cache-rebuild path enforces, in the |
| 103 | &nbsp;&nbsp;&nbsp;↳ `_kv()` | _kv() -> int |
| 111 | &nbsp;&nbsp;&nbsp;↳ `_moe()` | _moe() -> int |
| 116 | &nbsp;&nbsp;&nbsp;↳ `_mamba()` | _mamba() -> int |
| 123 | &nbsp;&nbsp;&nbsp;↳ `_swa()` | _swa() -> int |
| 136 | `compute_cache_pools()` | The ACTUAL pool sizes allocated at load, in API units. The frontend otherwise only |
| 166 | `compute_cache_status_meta()` | Full readiness ("meta", …) payload for the desktop cache panel: the per-unit VRAM costs |

## `freetoken/kvcache/dsa_pool.py`  (260 行)

| 行 | 符号 | 说明 |
|---:|---|---|
| 26 | **class `MLAKVCache`** | Paged latent-KV pool: ``[1, num_layers, num_pages, page_size, 1, latent_dim]``. |
| 37 | **class `__init__`** | __init__( self, latent_dim |
| 59 | **class `_local_layer`** | _local_layer(self, layer_id |
| 62 | **class `_alloc`** | _alloc(self, num_pages |
| 71 | **class `k_cache`** | Paged latent view ``[num_pages, page_size, latent_dim]``. |
| 77 | **class `v_cache`** | v_cache(self, layer_id |
| 81 | **class `latent_rows`** | Row-flat latent view ``[num_pages * page_size, latent_dim]``. |
| 86 | **class `store_kv`** | Scatter this forward's latent rows: ``c_kv`` [T, kv_lora_rank] and |
| 104 | **class `rebuild`** | In-place resize (frees the old slab first; object identity preserved -- |
| 114 | **class `kv_cost`** | kv_cost(cls, config) -> tuple[int, int, int, int] |
| 123 | **class `rebuild_from_config`** | rebuild_from_config( self, config, num_pages |
| 128 | **class `unit_bytes`** | unit_bytes(self) -> tuple[int, int] |
| 134 | **class `device`** | device(self) -> torch.device |
| 138 | **class `dtype`** | dtype(self) -> torch.dtype |
| 142 | **class `num_layers`** | num_layers(self) -> int |
| 146 | **class `DSAKVCache`** | MLA latent pool + the DSA index-key slab (one row per token per full-indexer |
| 150 | **class `__init__`** | __init__( self, latent_dim |
| 169 | **class `_index_rows`** | Index-slab row count: one row per token (KpoolDSAKVCache overrides to the |
| 174 | **class `_alloc`** | _alloc(self, num_pages |
| 188 | **class `rebuild`** | rebuild(self, num_pages |
| 192 | **class `unit_bytes`** | unit_bytes(self) -> tuple[int, int] |
| 200 | **class `index_k_cache`** | Row-flat index keys for a full-indexer layer slot: ``[rows, index_head_dim]``. |
| 204 | **class `store_index_k`** | store_index_k(self, k |
| 208 | **class `KpoolDSAKVCache`** | DSAKVCache with the index slab as a 1/ratio SHADOW of the KV pages, plus |
| 221 | **class `__init__`** | __init__(self, *args, num_req_slots |
| 230 | **class `_index_rows`** | _index_rows(self, num_pages |
| 235 | **class `cmp_scratch_base`** | First scratch row (== shadow row count); request ``table_idx`` offsets it. |
| 239 | **class `_alloc`** | _alloc(self, num_pages |
| 247 | **class `rebuild`** | rebuild(self, num_pages |
| 252 | **class `tail_k`** | Tail raw keys for an indexer layer slot: ``[num_req_slots, ratio, head_dim]``. |
| 256 | **class `tail_gate`** | tail_gate(self, slot |

## `freetoken/kvcache/hybrid_radix_cache.py`  (297 行)

| 行 | 符号 | 说明 |
|---:|---|---|
| 30 | **class `HybridCacheHandle`** | Lock handle for a matched hybrid prefix: the matched node (lock target) + the reusable |
| 39 | **class `get_matched_indices`** | get_matched_indices(self) -> torch.Tensor |
| 43 | **class `HybridMatch`** | class HybridMatch(NamedTuple) |
| 50 | **class `EvictResult`** | class EvictResult(NamedTuple) |
| 55 | **class `HybridRadixCache`** | class HybridRadixCache |
| 56 | **class `__init__`** | __init__(self, device |
| 76 | **class `match_prefix`** | Match the token prefix, then truncate the reusable length to the deepest node on |
| 90 | **class `insert`** | Insert the committed KV prefix and DONATE ``mamba_value`` at the (page-aligned) end |
| 118 | **class `inc_lock`** | Protect a matched node's snapshot (mamba ref on the node) and its KV path |
| 135 | **class `dec_lock`** | dec_lock(self, node |
| 151 | **class `evict_full`** | Evict KV tokens by LRU over UNLOCKED LEAF nodes (an internal node's KV is a prefix |
| 171 | **class `evict_mamba`** | Evict GDN snapshots by LRU over UNLOCKED snapshot-bearing nodes -- internal nodes |
| 195 | **class `full_evictable_size`** | full_evictable_size(self) -> int |
| 199 | **class `mamba_evictable_size`** | mamba_evictable_size(self) -> int |
| 203 | **class `size_info`** | KV-page currency, for code that reads a BasePrefixCache size_info (metrics/usage). |
| 209 | **class `check_integrity`** | check_integrity(self) -> None |
| 216 | **class `_free_node_mamba`** | _free_node_mamba(self, node |
| 223 | **class `_unlink`** | _unlink(self, node |
| 228 | **class `_cascade_tombstone_leaves`** | After a leaf is unlinked, eagerly reclaim the KV-only tombstone leaves it exposes |
| 242 | **class `_path_len`** | _path_len(self, node |
| 249 | **class `_collect_kv`** | _collect_kv(self, node |
| 258 | **class `_leaves`** | _leaves(self) -> List[RadixTreeNode] |
| 269 | **class `_snapshot_nodes`** | _snapshot_nodes(self) -> List[RadixTreeNode] |
| 278 | **class `_walk`** | _walk(self, input_ids |

## `freetoken/kvcache/hybrid_swa_pool.py`  (418 行)

| 行 | 符号 | 说明 |
|---:|---|---|
| 15 | **class `_LayerRef`** | class _LayerRef |
| 21 | **class `_KVGroupStorage`** | class _KVGroupStorage |
| 28 | **class `HybridSWAKVCache`** | SGLang-style wrapper for hybrid full/SWA attention KV storage. |
| 31 | **class `__init__`** | __init__( self, groups |
| 84 | **class `_allocate_group`** | _allocate_group( spec |
| 106 | **class `_build_layers_mapping`** | _build_layers_mapping( num_layers |
| 123 | **class `is_full_layer`** | is_full_layer(self, layer_id |
| 126 | **class `is_swa_layer`** | is_swa_layer(self, layer_id |
| 129 | **class `group_of`** | group_of(self, layer_id |
| 132 | **class `translate_loc_from_full_to_swa`** | translate_loc_from_full_to_swa(self, out_loc |
| 140 | **class `_init_swa_paged_state`** | (Re)build the global-paged SWA allocator state: the dense full->swa slot mapping |
| 159 | **class `alloc_swa`** | Allocate one swa slot per full slot and record full->swa in the mapping. The caller |
| 174 | **class `free_swa`** | Return the swa slots backing ``full_indices`` to the free-list and reset their |
| 188 | **class `swa_available_size`** | swa_available_size(self) -> int |
| 192 | **class `swa_paged`** | swa_paged(self) -> bool |
| 195 | **class `k_cache`** | k_cache(self, index |
| 199 | **class `v_cache`** | v_cache(self, index |
| 203 | **class `store_kv`** | store_kv( self, k |
| 226 | **class `full_num_tokens`** | full_num_tokens(self) -> int |
| 230 | **class `swa_num_tokens`** | swa_num_tokens(self) -> int |
| 234 | **class `device`** | device(self) -> torch.device |
| 238 | **class `dtype`** | dtype(self) -> torch.dtype |
| 242 | **class `num_layers`** | num_layers(self) -> int |
| 246 | **class `_group_geometry`** | _group_geometry(group |
| 253 | **class `_alloc_group`** | _alloc_group(geom |
| 268 | **class `rebuild`** | Reallocate both group buffers IN PLACE for new sizes. |
| 299 | **class `window_spec`** | window_spec(cls, config) -> WindowPoolSpec \| None |
| 305 | **class `window_pages`** | window_pages(self) -> int |
| 309 | **class `kv_cost`** | Full groups ride cache_per_page; the window pool is either a pinned absolute |
| 343 | **class `rebuild_from_config`** | rebuild_from_config( self, config, num_pages |
| 355 | **class `unit_bytes`** | unit_bytes(self) -> tuple[int, int] |
| 368 | `_naive_swa_num_tokens()` | Naive SWA pool = one window+headroom buffer per concurrent request. Sized so a request's |
| 379 | `_swa_per_req_swa_floor()` | One request's NON-EVICTABLE swa while it decodes, in tokens: |
| 400 | `_swa_pool_floor()` | The swa pool's hard floor: max_running_req x the per-request non-evictable footprint. |
| 408 | `_swa_paged_num_tokens()` | SWA pool size = max(concurrency floor, target tokens) + 1 (slot-0 sentinel). The target is |

## `freetoken/kvcache/linear_state_pool.py`  (301 行)

| 行 | 符号 | 说明 |
|---:|---|---|
| 18 | `ssm_state_dtype()` | Recurrent (SSM) state dtype, from FREETOKEN_MAMBA_SSM_DTYPE (default fp32). |
| 23 | `_linear_local_dims()` | TP-local ``(n_layers, conv_dim, v_heads)`` for the GDN state tensors -- the single |
| 34 | **class `LinearStatePool`** | Per-request recurrent state (conv + SSM) for GatedDeltaNet layers. |
| 47 | **class `__init__`** | __init__( self, group |
| 98 | **class `_alloc_slot_states`** | _alloc_slot_states(self, num_slots |
| 109 | **class `has_slot_state`** | has_slot_state(self, name |
| 112 | **class `slot_state`** | One declared sibling state, ``[num_slots, *shape]``; ``layer_id`` picks the layer row. |
| 123 | **class `num_free_slots`** | num_free_slots(self) -> int |
| 126 | **class `alloc`** | Pop ``n`` free slot ids (LIFO). Raises if the pool is exhausted. |
| 134 | **class `reclaim_all_slots`** | Restore the free-list to all non-padding slots. Idle-only: the caller (e.g. a |
| 140 | **class `rebuild`** | Reallocate the conv + recurrent state tensors for ``num_slots`` slots IN PLACE. |
| 171 | **class `free`** | Return slot ids to the free-list. Accepts an int, list, or 1-D tensor. |
| 179 | **class `clear_slots`** | Zero conv + recurrent state at ``slots`` across all linear layers (fresh sequence). |
| 188 | **class `copy_from`** | Copy a whole-sequence snapshot (conv + recurrent, all layers) from slot ``src`` to |
| 196 | **class `is_linear_layer`** | is_linear_layer(self, layer_id |
| 199 | **class `local_index`** | local_index(self, layer_id |
| 202 | **class `conv_state`** | conv_state(self, layer_id |
| 205 | **class `recurrent_state`** | recurrent_state(self, layer_id |
| 208 | **class `reset`** | Zero a slot across all linear layers (new request takes this table_idx). |
| 216 | **class `num_linear_layers`** | num_linear_layers(self) -> int |
| 220 | **class `num_slots`** | num_slots(self) -> int |
| 224 | **class `device`** | device(self) -> torch.device |
| 227 | **class `bytes_per_slot`** | Total state bytes for one request (all linear layers). |
| 238 | `linear_state_bytes_per_req()` | Linear-state bytes for one request across all linear layers (TP-local), plus any |
| 263 | `state_pool_bytes()` | Total GDN state-pool bytes at ``num_slots`` PHYSICAL slots (default: the startup |
| 280 | `_linear_pool_num_slots()` | LinearStatePool slot count. Hybrid-radix non-evictable peak is 4 slots per running request |
| 292 | `_linear_pool_min_slots()` | Floor on LinearStatePool slots that still runs: the non-evictable working set with a |

## `freetoken/kvcache/mha_pool.py`  (148 行)

| 行 | 符号 | 说明 |
|---:|---|---|
| 12 | **class `MHAKVCache`** | Base class for key-value caches. |
| 25 | **class `__init__`** | __init__( self, num_kv_heads |
| 60 | **class `rebuild`** | Reallocate the KV buffer for ``num_pages`` pages IN PLACE. |
| 86 | **class `kv_cost`** | kv_cost(cls, config) -> tuple[int, int, int, int] |
| 96 | **class `rebuild_from_config`** | rebuild_from_config( self, config, num_pages |
| 101 | **class `unit_bytes`** | unit_bytes(self) -> tuple[int, int] |
| 106 | **class `_dense`** | _dense(self, layer_id |
| 114 | **class `k_cache`** | k_cache(self, index |
| 117 | **class `v_cache`** | v_cache(self, index |
| 120 | **class `store_kv`** | store_kv( self, k |
| 139 | **class `device`** | device(self) -> torch.device |
| 143 | **class `dtype`** | dtype(self) -> torch.dtype |
| 147 | **class `num_layers`** | num_layers(self) -> int |

## `freetoken/kvcache/naive_cache.py`  (45 行)

| 行 | 符号 | 说明 |
|---:|---|---|
| 6 | **class `NaiveCacheHandle`** | class NaiveCacheHandle(BaseCacheHandle) |
| 9 | **class `__init__`** | __init__(self) |
| 12 | **class `get_matched_indices`** | get_matched_indices(self) -> torch.Tensor |
| 16 | **class `NaivePrefixCache`** | class NaivePrefixCache(BasePrefixCache) |
| 17 | **class `__init__`** | __init__(self, device |
| 23 | **class `lock_handle`** | lock_handle(self, handle |
| 26 | **class `match_prefix`** | match_prefix(self, input_ids |
| 29 | **class `insert_prefix`** | insert_prefix(self, input_ids |
| 32 | **class `evict`** | evict(self, size |
| 37 | **class `reset`** | reset(self) -> None |
| 41 | **class `size_info`** | size_info(self) -> SizeInfo |
| 44 | **class `check_integrity`** | check_integrity(self) -> None |

## `freetoken/kvcache/qsa_pool.py`  (231 行)

| 行 | 符号 | 说明 |
|---:|---|---|
| 38 | **class `QSAKVCache`** | MHA paged pool + the compressed index-key slab + the per-request pending ring. |
| 49 | **class `ring_capacity_for`** | Ring depth: one row per pending position, keyed ``position % capacity``; spec decode widens by the draft depth (vLLM sizing). |
| 53 | **class `__init__`** | __init__( self, num_kv_heads |
| 109 | **class `_zero_kv_slabs`** | _zero_kv_slabs(self) -> None |
| 115 | **class `_alloc_index_tiers`** | _alloc_index_tiers(self, num_pages |
| 142 | **class `rebuild`** | rebuild(self, num_pages |
| 162 | **class `kv_cost`** | kv_cost(cls, config) -> tuple[int, int, int, int] |
| 181 | **class `unit_bytes`** | unit_bytes(self) -> tuple[int, int] |
| 194 | **class `cmp_k_cache`** | Compressed index keys of one sparse layer: ``[rows, index_head_dim]``. |
| 198 | **class `pending_ring`** | One sparse layer's pending ring: ``[num_req_slots, ring_capacity, index_head_dim]``. |
| 203 | **class `rope_positions`** | ``[num_tokens, 3]`` int32 t/h/w rope position per KV slot (written by the QSA backend). |
| 209 | **class `cmp_scratch_base`** | First scratch row of ``cmp_k_cache``; row ``cmp_scratch_base + table_idx`` sinks a |
| 215 | **class `index_ratio`** | index_ratio(self) -> int |
| 219 | **class `index_head_dim`** | index_head_dim(self) -> int |
| 223 | **class `ring_capacity`** | ring_capacity(self) -> int |
| 227 | **class `num_req_slots`** | num_req_slots(self) -> int |

## `freetoken/kvcache/radix_cache.py`  (268 行)

| 行 | 符号 | 说明 |
|---:|---|---|
| 17 | **class `RadixTreeNode`** | class RadixTreeNode |
| 20 | **class `__init__`** | __init__(self, key_fn |
| 52 | **class `set_key_value`** | set_key_value(self, key |
| 58 | **class `set_parent`** | set_parent(self, parent |
| 63 | **class `length`** | length(self) -> int |
| 67 | **class `parent`** | parent(self) -> RadixTreeNode |
| 72 | **class `value`** | value(self) -> torch.Tensor |
| 75 | **class `is_root`** | is_root(self) -> bool |
| 78 | **class `is_leaf`** | is_leaf(self) -> bool |
| 81 | **class `get_match_len`** | get_match_len(self, input_ids |
| 87 | **class `split_at`** | split_at(self, pos |
| 108 | **class `__lt__`** | __lt__(self, other |
| 113 | **class `RadixCacheHandle`** | class RadixCacheHandle(BaseCacheHandle) |
| 116 | **class `get_matched_indices`** | get_matched_indices(self) -> torch.Tensor |
| 126 | **class `RadixPrefixCache`** | class RadixPrefixCache(BasePrefixCache) |
| 127 | **class `__init__`** | __init__(self, device |
| 141 | **class `lock_handle`** | lock_handle(self, handle |
| 160 | **class `match_prefix`** | match_prefix(self, input_ids |
| 164 | **class `insert_prefix`** | insert_prefix(self, input_ids |
| 179 | **class `evict`** | evict(self, size |
| 208 | **class `reset`** | reset(self) -> None |
| 212 | **class `size_info`** | size_info(self) -> SizeInfo |
| 218 | **class `check_integrity`** | check_integrity(self) -> None |
| 221 | **class `_collect_leave_nodes_for_evict`** | _collect_leave_nodes_for_evict(self) -> List[RadixTreeNode] |
| 236 | **class `_tree_walk`** | _tree_walk(self, input_ids |
| 265 | `_get_key_fn()` | _get_key_fn(page_size |

## `freetoken/kvcache/swa_radix_cache.py`  (455 行)

| 行 | 符号 | 说明 |
|---:|---|---|
| 36 | **class `SWACacheHandle`** | Lock handle for a matched SWA prefix: the windowed-safe matched node (lock target) + the |
| 46 | **class `get_matched_indices`** | get_matched_indices(self) -> torch.Tensor |
| 55 | **class `SWAMatch`** | class SWAMatch(NamedTuple) |
| 61 | **class `SWAEvictResult`** | class SWAEvictResult(NamedTuple) |
| 66 | **class `SWARadixCache`** | class SWARadixCache |
| 67 | **class `__init__`** | __init__(self, device |
| 88 | **class `match_prefix`** | Match the token prefix for the full layers, then truncate the reusable length to the |
| 137 | **class `insert`** | Insert the committed full KV prefix. ``kv_indices`` are the request's full-pool page |
| 231 | **class `_add_child`** | _add_child(self, parent |
| 243 | **class `inc_lock`** | Pin the matched node's KV path (full ref node..root) and the trailing window of live |
| 270 | **class `dec_lock`** | dec_lock(self, node |
| 290 | **class `evict_full`** | Evict full KV by LRU over UNLOCKED LEAF nodes (an internal node's KV is a prefix |
| 316 | **class `evict_swa`** | Evict swa KV by LRU over UNLOCKED, non-tombstone swa-bearing nodes -- internal nodes |
| 344 | **class `trim_head_swa`** | Tombstone the swa currency of the path strictly below ``keep_from`` (full KV stays), |
| 368 | **class `full_evictable_size`** | full_evictable_size(self) -> int |
| 372 | **class `swa_evictable_size`** | swa_evictable_size(self) -> int |
| 376 | **class `size_info`** | size_info(self) |
| 380 | **class `check_integrity`** | check_integrity(self) -> None |
| 388 | **class `_unlink`** | _unlink(self, node |
| 393 | **class `_cascade_swa_tombstone_leaves`** | After a leaf is unlinked, eagerly reclaim the swa-tombstone leaves it exposes upward |
| 410 | **class `_stamp_path`** | Stamp the matched path with strictly DECREASING timestamps toward root (deepest newest) |
| 422 | **class `_tick`** | _tick(self) -> int |
| 426 | **class `_leaves`** | _leaves(self) -> List[RadixTreeNode] |
| 437 | **class `_swa_nodes`** | _swa_nodes(self) -> List[RadixTreeNode] |
| 446 | **class `_all_nodes`** | _all_nodes(self) -> List[RadixTreeNode] |

## `freetoken/kvcache/dsv4/__init__.py`  (1 行)

_（无类/函数定义，模块级代码）_

## `freetoken/kvcache/dsv4/v41_cost_model.py`  (216 行)

| 行 | 符号 | 说明 |
|---:|---|---|
| 31 | `_win_row_bytes()` | _win_row_bytes(args) -> int |
| 35 | `_main_row_bytes()` | _main_row_bytes(args) -> int |
| 39 | `_idx_row_bytes()` | _idx_row_bytes(args) -> int |
| 43 | `_state_bytes()` | One compress-state ring slot: fp32 ``kv \| score`` of the latent width. |
| 48 | `private_window_layer_ids()` | Layers whose window KV is REQUEST-PRIVATE: kept in a per-page-table-row ring of ``window`` |
| 61 | **class `DSV41PoolSizes`** | Per-tier row counts derived from the budget anchor ``full_token``; the global tiers are |
| 75 | `dsv41_pool_sizes()` | dsv41_pool_sizes( num_pages |
| 92 | `dsv41_pool_bytes()` | Exact bytes a ``DSV41PagedKVCache`` built from ``sizes`` allocates (mirror of ``total_bytes``). |
| 106 | `dsv41_cache_per_page()` | Marginal bytes per P-token page across all tiers (window scaled by ``swa_ratio``). |
| 118 | `dsv41_kv_unit_bytes()` | FULL-tier bytes per full-history token (main + index pools + the mapping); the slider's |
| 127 | `dsv41_window_unit_bytes()` | WINDOW-tier bytes per window token (sliding KV on every shared-window layer + the state rings). |
| 137 | `dsv41_solve_num_pages()` | Largest budget-respecting pool; the window floor is honored in pages and the total is |
| 143 | &nbsp;&nbsp;&nbsp;↳ `_sizes()` | _sizes(num |
| 165 | `dsv41_auto_cost_model()` | Affine ``(cache_per_page, fixed_cache_size, min_reserve_tokens)`` for the MoE-first auto planner. |
| 178 | `_dsv41_swa_ratio()` | _dsv41_swa_ratio(config) -> float |
| 182 | `_dsv41_window_floor_pages()` | Minimum window pages: one prefill chunk's reach (capped at 8 pages) + the running requests' |
| 191 | `_dsv41_pool_sizes()` | Sizes for ``num_pages`` PHYSICAL pages (dummy included). Window precedence: an explicit |

## `freetoken/kvcache/dsv4/v41_pool.py`  (292 行)

| 行 | 符号 | 说明 |
|---:|---|---|
| 46 | **class `DSV41PagedKVCache`** | class DSV41PagedKVCache(WindowTierPagedPool) |
| 51 | **class `__init__`** | __init__( self, sizes |
| 78 | **class `_alloc_buffers`** | _alloc_buffers(self) -> None |
| 102 | **class `total_bytes`** | total_bytes(self) -> int |
| 111 | **class `prefix_replay_tokens`** | Prompt-tail tokens a prefix hit must leave to the prefill: bounded replay runs the decoder on |
| 117 | **class `is_private_window`** | is_private_window(self, layer_id |
| 120 | **class `ring_slots`** | Ring slots of ``positions`` (int64, broadcastable with ``table_rows``) for private window layers; |
| 127 | **class `source_of`** | source_of(self, layer_id |
| 132 | **class `main_pool_of`** | main_pool_of(self, layer_id |
| 135 | **class `idx_pool_of`** | idx_pool_of(self, layer_id |
| 138 | **class `ratio_of`** | ratio_of(self, layer_id |
| 143 | **class `kv_cost`** | kv_cost(cls, config) -> tuple[int, int, int, int] |
| 154 | **class `solve_num_pages`** | solve_num_pages(cls, config, available_memory |
| 190 | **class `window_spec`** | window_spec(cls, config) |
| 198 | **class `min_kv_tokens`** | min_kv_tokens(cls, config) -> int |
| 204 | **class `validate_rebuild`** | validate_rebuild( self, config, *, num_pages |
| 237 | **class `rebuild_from_config`** | rebuild_from_config(self, config, num_pages |
| 242 | **class `_drop_buffers`** | _drop_buffers(self) -> None |
| 245 | **class `unit_bytes`** | unit_bytes(self) -> tuple[int, int] |
| 249 | **class `store_window`** | store_window(self, kv |
| 254 | **class `store_main`** | store_main(self, latent |
| 259 | **class `store_index`** | store_index(self, k |
| 265 | **class `read_window`** | read_window(self, layer_id |
| 270 | **class `read_main`** | read_main(self, source |
| 275 | **class `read_index`** | read_index(self, source |
| 281 | **class `get_state`** | get_state(self, source |
| 284 | **class `set_state`** | set_state(self, source |
| 288 | **class `num_layers`** | num_layers(self) -> int |

## `freetoken/kvcache/dsv4/v41_row_format.py`  (69 行)

| 行 | 符号 | 说明 |
|---:|---|---|
| 26 | **class `RowFormat`** | class RowFormat |
| 33 | **class `value_bytes`** | value_bytes(self, dim |
| 37 | **class `scale_bytes`** | scale_bytes(self, dim |
| 43 | **class `row_bytes`** | row_bytes(self, dim |
| 46 | **class `num_groups`** | num_groups(self, dim |
| 49 | **class `validate_dim`** | validate_dim(self, dim |
| 62 | `row_format()` | row_format(name |

## `freetoken/kvcache/dsv4/v4_cost_model.py`  (364 行)

| 行 | 符号 | 说明 |
|---:|---|---|
| 30 | `dsv4_reserved_window_pages()` | Window pages the sliding pool must always keep for the concurrent working set; the |
| 39 | `ring_size_for_ratio()` | Compress-state ring slots per window page (non-speculative). |
| 48 | `_kv_bytes()` | _kv_bytes(args) -> int |
| 52 | `_index_bytes()` | _index_bytes(args) -> int |
| 56 | `_state_bytes()` | _state_bytes(args, ratio |
| 61 | `_idx_state_bytes()` | _idx_state_bytes(args) -> int |
| 67 | `dsv4_cache_per_page()` | Bytes per P-token across ALL tiers, summed over layers per ratio class. |
| 99 | `dsv4_kv_unit_bytes()` | FULL-tier bytes per full-history token: compressed KV + indexer KV + the full->window |
| 117 | `dsv4_window_unit_bytes()` | WINDOW(swa)-tier bytes per window token: the sliding KV pool (every layer) plus the |
| 135 | **class `DSV4PoolSizes`** | Per-tier slot counts derived from the budget anchor ``full_token``. |
| 156 | `dsv4_pool_sizes()` | dsv4_pool_sizes( num_pages |
| 208 | `dsv4_pool_bytes()` | Exact bytes a ``DSV4PagedKVCache`` built from ``sizes`` allocates (mirror of ``total_bytes`` |
| 229 | `dsv4_solve_num_pages()` | Largest budget-respecting pool: max ``num_pages`` with exact |
| 245 | &nbsp;&nbsp;&nbsp;↳ `_sizes()` | _sizes(num |
| 271 | `dsv4_auto_cost_model()` | Affine (cache_per_page, fixed_cache_size, min_reserve_tokens) for the MoE-first auto planner: |
| 309 | `_dsv4_swa_ratio()` | The live DSV4 window/full ratio from the serving config (config.swa_full_tokens_ratio; |
| 315 | `_dsv4_window_floor_pages()` | Minimum window pages = the live sliding working set the window pool must always hold: |
| 329 | `_dsv4_pool_sizes()` | _dsv4_pool_sizes(config, num_pages |

## `freetoken/kvcache/dsv4/v4_pool.py`  (349 行)

| 行 | 符号 | 说明 |
|---:|---|---|
| 41 | **class `DSV4PagedKVCache`** | class DSV4PagedKVCache(WindowTierPagedPool) |
| 42 | **class `__init__`** | __init__( self, sizes |
| 81 | **class `_alloc_buffers`** | (Re)allocate every physical buffer for the CURRENT ``self.sizes``. Shared by __init__ |
| 165 | **class `kv_cost`** | kv_cost(cls, config) -> tuple[int, int, int, int] |
| 178 | **class `solve_num_pages`** | solve_num_pages(cls, config, available_memory |
| 221 | **class `window_spec`** | window_spec(cls, config) |
| 229 | **class `min_kv_tokens`** | min_kv_tokens(cls, config) -> int |
| 237 | **class `validate_rebuild`** | validate_rebuild( self, config, *, num_pages |
| 282 | **class `rebuild_from_config`** | rebuild_from_config( self, config, num_pages |
| 290 | **class `unit_bytes`** | unit_bytes(self) -> tuple[int, int] |
| 295 | **class `_drop_buffers`** | _drop_buffers(self) -> None |
| 299 | **class `ring_size`** | ring_size(self, layer_id |
| 303 | **class `get_state`** | get_state(self, layer_id |
| 308 | **class `set_state`** | set_state(self, layer_id |
| 314 | **class `store_window`** | store_window(self, k |
| 317 | **class `store_compressed`** | store_compressed(self, kv |
| 322 | **class `store_indexer`** | store_indexer(self, k |
| 327 | **class `total_bytes`** | total_bytes(self) -> int |
| 345 | **class `num_layers`** | num_layers(self) -> int |

## `freetoken/kvcache/dsv4/window_tier.py`  (331 行)

| 行 | 符号 | 说明 |
|---:|---|---|
| 26 | **class `FreeListAllocator`** | LIFO free list over ``capacity // page_unit`` units; a unit's base slot is a multiple of |
| 30 | **class `__init__`** | __init__(self, capacity |
| 37 | **class `_fresh_free`** | _fresh_free(self) -> torch.Tensor |
| 41 | **class `alloc`** | Return ``n_units`` unit base slots (each a multiple of ``page_unit``). |
| 56 | **class `free`** | Return previously-allocated unit base slots to the pool (LIFO recycle). |
| 63 | **class `available`** | Free capacity in slots (units * page_unit). |
| 68 | **class `capacity`** | capacity(self) -> int |
| 71 | **class `reset`** | reset(self) -> None |
| 75 | **class `CompressStateRing`** | Per-layer fp32 compress-state ring: ``[n_slots + 1, 2*(1+overlap)*head_dim]``. |
| 82 | **class `__init__`** | __init__( self, n_slots |
| 99 | **class `_clear_scratch`** | _clear_scratch(self) -> None |
| 104 | **class `item_size`** | item_size(self) -> int |
| 107 | **class `get`** | Gather ``kv_score`` rows at ``state_loc`` (``-1`` -> scratch row). |
| 111 | **class `set`** | Scatter ``kv_score`` rows to ``state_loc`` then re-clear the scratch row. |
| 116 | **class `get_blocks`** | Batched per-row carry-block read: ``page_base`` ``[B]`` (``(window_slot // P) * ring_size``) |
| 122 | **class `set_blocks`** | Batched per-row carry-block write of ``[B, ring_size, 2*item]``; re-clears scratch. |
| 129 | `reserved_window_pages()` | Window pages the sliding pool must always keep for the concurrent working set: each |
| 138 | `window_state_loc()` | Ring slot of a window slot: ``(ws // P) * ring_size + ws % ring_size``; ``-1`` stays ``-1`` |
| 146 | **class `WindowTierPagedPool`** | A paged KV pool whose window tier is page-bound to the shared full-token page table. |
| 175 | **class `_alloc_buffers`** | _alloc_buffers(self) -> None |
| 178 | **class `_drop_buffers`** | Release every tier buffer before a rebuild reallocates them. |
| 182 | **class `translate_full_to_window`** | translate_full_to_window(self, full_locs |
| 187 | **class `cmp_rows`** | The compressed row of a full loc: ``full_loc // ratio`` (negative stays negative). |
| 191 | **class `bind_window_pages`** | bind_window_pages(self, full_page_base |
| 197 | **class `unbind_window_pages`** | unbind_window_pages(self, full_locs |
| 201 | **class `state_loc`** | state_loc(window_slot |
| 206 | **class `sliding_window_size`** | sliding_window_size(self) -> int |
| 210 | **class `prefill_chunk_budget`** | prefill_chunk_budget(self) -> int |
| 213 | **class `_init_paged_state`** | Build the pool-owned window free list + the tail dummy binding. The LAST full page and |
| 231 | **class `window_pages`** | window_pages(self) -> int |
| 235 | **class `swa_num_tokens`** | swa_num_tokens(self) -> int |
| 241 | **class `swa_available_size`** | swa_available_size(self) -> int |
| 244 | **class `alloc_swa`** | Bind one window page per incoming FULL page. ``full_indices`` must be whole ascending |
| 260 | **class `free_swa`** | Return the window pages backing these FULL locs and unbind the mapping. Page-atomic: the |
| 281 | **class `translate_loc_from_full_to_swa`** | translate_loc_from_full_to_swa(self, kv_indices |
| 285 | **class `rebuild`** | In-place resize to ``sizes`` (identity-preserving; free-before-alloc). The manager's |
| 299 | **class `attach_page_table`** | attach_page_table(self, page_table |
| 304 | **class `k_cache`** | k_cache(self, index |
| 307 | **class `v_cache`** | v_cache(self, index |
| 310 | **class `store_kv`** | store_kv(self, k, v, out_loc, layer_id) -> None |
| 314 | **class `store_window`** | store_window(self, kv |
| 317 | **class `device`** | device(self) -> torch.device |
| 321 | **class `dtype`** | dtype(self) -> torch.dtype |

