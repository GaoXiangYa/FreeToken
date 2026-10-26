# 符号索引：`attention`

自动生成（`ast` 解析，不执行代码）。**17 个文件 / 4219 行 / 274 个符号**。

> 用途：确认某个模块里**到底有哪些**类/函数/方法及其行号，避免遗漏。
> 行为详解请看同目录下的手写模块文档。
> 重新生成：`python3 MODULE_DOCS/tools/gen_symbol_index.py`

| 文件 | 行数 | 符号数 |
|---|---:|---:|
| `freetoken/attention/__init__.py` | 209 | 15 |
| `freetoken/attention/base.py` | 130 | 20 |
| `freetoken/attention/dsa.py` | 376 | 22 |
| `freetoken/attention/dsa_indexer.py` | 60 | 4 |
| `freetoken/attention/dsa_indexer_kpool.py` | 288 | 12 |
| `freetoken/attention/dsv41_sparse.py` | 438 | 49 |
| `freetoken/attention/dsv4_compress.py` | 154 | 14 |
| `freetoken/attention/dsv4_indexer.py` | 87 | 6 |
| `freetoken/attention/dsv4_sparse.py` | 291 | 23 |
| `freetoken/attention/fa.py` | 198 | 11 |
| `freetoken/attention/fi.py` | 313 | 19 |
| `freetoken/attention/linear.py` | 140 | 4 |
| `freetoken/attention/m3_sparse.py` | 443 | 21 |
| `freetoken/attention/qsa_sparse.py` | 542 | 27 |
| `freetoken/attention/triton.py` | 355 | 15 |
| `freetoken/attention/trtllm.py` | 172 | 10 |
| `freetoken/attention/utils.py` | 23 | 2 |

---

## `freetoken/attention/__init__.py`  (209 行)

| 行 | 符号 | 说明 |
|---:|---|---|
| 16 | **class `BackendCreator`** | class BackendCreator(Protocol) |
| 17 | **class `__call__`** | __call__(self, config |
| 21 | **class `BackendInfo`** | Declarative capability matrix entry for one backend. The engine's config-time |
| 50 | `create_trtllm_backend()` | create_trtllm_backend(config |
| 63 | `create_fi_backend()` | create_fi_backend(config |
| 76 | `create_fa_backend()` | create_fa_backend(config |
| 89 | `create_triton_backend()` | create_triton_backend(config |
| 99 | `create_dsv4_sparse_backend()` | create_dsv4_sparse_backend(config |
| 109 | `create_dsv41_sparse_backend()` | create_dsv41_sparse_backend(config |
| 119 | `create_dsa_backend()` | create_dsa_backend(config |
| 140 | `create_m3_sparse_backend()` | create_m3_sparse_backend(config |
| 155 | `create_qsa_sparse_backend()` | create_qsa_sparse_backend(config |
| 161 | `attention_backend_info()` | attention_backend_info(name |
| 165 | `validate_attn_backend()` | validate_attn_backend(backend |
| 181 | `create_attention_backend()` | create_attention_backend( backend |

## `freetoken/attention/base.py`  (130 行)

| 行 | 符号 | 说明 |
|---:|---|---|
| 13 | **class `AttnType`** | Attention-type taxonomy, one value per KV-pool family. A model's attention |
| 36 | **class `backend_driven`** | backend_driven(self) -> bool |
| 43 | **class `AttentionSpec`** | class AttentionSpec |
| 52 | **class `BaseAttnMetadata`** | class BaseAttnMetadata(ABC) |
| 54 | **class `get_last_indices`** | get_last_indices(self, bs |
| 57 | **class `BaseAttnBackend`** | class BaseAttnBackend(ABC) |
| 59 | **class `forward`** | forward( self, q |
| 70 | **class `prepare_metadata`** | prepare_metadata(self, batch |
| 73 | **class `init_capture_graph`** | init_capture_graph(self, max_seq_len |
| 76 | **class `prepare_for_capture`** | prepare_for_capture(self, batch |
| 79 | **class `prepare_for_replay`** | prepare_for_replay(self, batch |
| 81 | **class `reset_capture`** | Drop CUDA-graph capture scratch so ``init_capture_graph`` can re-run after a |
| 94 | **class `HybridBackend`** | class HybridBackend(BaseAttnBackend) |
| 95 | **class `__init__`** | __init__( self, prefill_backend |
| 103 | **class `forward`** | forward( self, q |
| 115 | **class `prepare_metadata`** | prepare_metadata(self, batch |
| 119 | **class `init_capture_graph`** | init_capture_graph(self, max_seq_len |
| 122 | **class `prepare_for_capture`** | prepare_for_capture(self, batch |
| 125 | **class `prepare_for_replay`** | prepare_for_replay(self, batch |
| 128 | **class `reset_capture`** | reset_capture(self) -> None |

## `freetoken/attention/dsa.py`  (376 行)

| 行 | 符号 | 说明 |
|---:|---|---|
| 53 | **class `KpoolPlan`** | glm5_next kpool: one forward's slab/ring routing (layer-invariant; the |
| 65 | **class `DSAIndexerInputs`** | Per-forward indexer tensors, built by the MODEL's indexer module and passed |
| 80 | **class `DSAMetadata`** | class DSAMetadata(BaseAttnMetadata) |
| 99 | **class `get_last_indices`** | get_last_indices(self, bs |
| 103 | **class `DSAAttnBackend`** | class DSAAttnBackend(DSAIndexerMixin, BaseAttnBackend) |
| 104 | **class `__init__`** | __init__(self, config |
| 141 | **class `_model_args`** | The MLA/indexer dims payload. GLM-5.2 reads glm_dsa_args; glm5_next's |
| 148 | **class `_build_index_slots`** | IndexShare (GLM-5.2): "full" layers own an indexer slot; followers reuse |
| 161 | **class `_store_index`** | Scatter this forward's index keys (kpool subclass adds gate scores and |
| 166 | **class `forward`** | forward(self, q, k, v, layer_id, batch, attn_spec |
| 170 | **class `prepare_metadata`** | prepare_metadata(self, batch |
| 195 | **class `_attend`** | Gathered-KV MLA over latent rows: q [b, m, H, 576] -> [b, m, H, 512]. |
| 206 | **class `mla_forward`** | Store this forward's latent rows and attend over the paged latent history. |
| 235 | **class `_decode`** | _decode(self, md, layer_id, q_nope, q_pe, inputs) -> torch.Tensor |
| 261 | **class `_select_prefill`** | Per-request causal top-k: ([1, m, K] physical rows, [1, m] counts). |
| 288 | **class `_prefill`** | _prefill(self, md, layer_id, q_nope, q_pe, batch, inputs) -> torch.Tensor |
| 330 | **class `init_capture_graph`** | init_capture_graph(self, max_seq_len |
| 338 | **class `_decode_rows`** | This decode step's per-request page-table rows [bs, W], gathered off the |
| 344 | **class `_stage_decode`** | Copy this step's addressing into the static graph buffers and point the |
| 353 | **class `prepare_for_capture`** | prepare_for_capture(self, batch |
| 364 | **class `prepare_for_replay`** | prepare_for_replay(self, batch |
| 370 | **class `reset_capture`** | reset_capture(self) -> None |

## `freetoken/attention/dsa_indexer.py`  (60 行)

| 行 | 符号 | 说明 |
|---:|---|---|
| 24 | **class `DSAIndexerMixin`** | class DSAIndexerMixin(IndexerBackendMixin) |
| 25 | **class `dsa_decode_scores`** | Fused head-reduced logits ``[bs, W]`` fp32 for a decode step: keys gathered |
| 38 | **class `dsa_prefill_logits`** | Head-reduced logits ``[m, kv_len]`` fp32 over a dense key slab (dsv4's |
| 51 | **class `dsa_map_rows`** | Selected POSITIONS -> physical rows, ``-1`` sentinel passed through. |

## `freetoken/attention/dsa_indexer_kpool.py`  (288 行)

| 行 | 符号 | 说明 |
|---:|---|---|
| 44 | **class `Glm5NextDSABackend`** | class Glm5NextDSABackend(DSAAttnBackend) |
| 45 | **class `__init__`** | __init__(self, config |
| 59 | **class `init_capture_graph`** | init_capture_graph(self, max_seq_len |
| 63 | **class `_stage_decode`** | _stage_decode(self, batch |
| 68 | **class `reset_capture`** | reset_capture(self) -> None |
| 73 | **class `_model_args`** | _model_args(self, config |
| 88 | **class `_build_index_slots`** | _build_index_slots(self, args, config |
| 97 | **class `_plan_kpool_writes`** | Per-token slab/ring routing for this forward; layer-invariant, so the |
| 152 | **class `_store_index`** | One fused kernel serves prefill and decode (layout and member |
| 181 | **class `_expand_and_tail`** | Selected pools -> token rows, tail tokens appended FIRST (fixed kpool-1 |
| 218 | **class `_decode`** | _decode(self, md, layer_id, q_nope, q_pe, inputs) -> torch.Tensor |
| 250 | **class `_select_prefill`** | Per-request causal top-k at pool granularity, expanded to token rows. |

## `freetoken/attention/dsv41_sparse.py`  (438 行)

| 行 | 符号 | 说明 |
|---:|---|---|
| 37 | **class `PrefillSegment`** | One request's slice of a flat prefill token stream. |
| 52 | **class `end`** | end(self) -> int |
| 57 | **class `SharedSelection`** | What the index-producing layers hand down the stack within ONE forward. |
| 74 | **class `DSV41AttnMetadata`** | class DSV41AttnMetadata(BaseAttnMetadata) |
| 90 | **class `get_last_indices`** | get_last_indices(self, bs |
| 94 | **class `stage_width`** | stage_width(self) -> int |
| 98 | **class `full_snapshot`** | full_snapshot(self) -> torch.Tensor |
| 105 | **class `window_ctx`** | Layer-invariant decode ring context ``(window_slots, prev_window_slots, |
| 123 | **class `private_window_ctx`** | The private-ring counterpart of ``window_ctx`` for request-private window layers: |
| 140 | **class `DSV41CaptureData`** | class DSV41CaptureData |
| 146 | **class `create`** | create(cls, max_bs |
| 154 | **class `DSV41SparseAttnBackend`** | class DSV41SparseAttnBackend(BaseAttnBackend) |
| 155 | **class `__init__`** | __init__(self, config |
| 167 | **class `pool`** | pool(self) |
| 171 | **class `forward`** | forward(self, q, k, v, layer_id, batch, attn_spec |
| 174 | **class `prepare_metadata`** | prepare_metadata(self, batch |
| 181 | **class `_prefill_metadata`** | The encoder pass runs every request's new tokens ``[cached_len, device_len)``. Exact mode runs |
| 215 | **class `init_capture_graph`** | init_capture_graph(self, max_seq_len |
| 221 | **class `prepare_for_capture`** | prepare_for_capture(self, batch |
| 226 | **class `prepare_for_replay`** | prepare_for_replay(self, batch |
| 229 | **class `_table_rows`** | _table_rows(self, batch |
| 233 | **class `_point_to_capture`** | _point_to_capture(self, batch |
| 246 | **class `metadata`** | metadata(self) -> DSV41AttnMetadata |
| 252 | **class `selection`** | selection(self) -> SharedSelection |
| 255 | **class `snapshot`** | snapshot(self) -> torch.Tensor |
| 259 | **class `window_slots_of`** | SHARED window slots of positions ``[lo, hi)`` off the request's LIVE full locs (slid-out -> -1). |
| 263 | **class `ring_slots_of`** | PRIVATE-ring slots of positions ``[lo, hi)`` of page-table row ``ti`` (``hi - lo <= win``). |
| 268 | **class `layer_window_slots_of`** | Where ``layer_id`` keeps the window KV of positions ``[lo, hi)``: its private ring or the shared pool. |
| 274 | **class `store_window`** | store_window(self, kv |
| 277 | **class `window_topk_prefill`** | Per-query window candidates for a prefill segment as GLOBAL window slots ``[1, n, win]`` |
| 292 | **class `source_of`** | source_of(self, layer_id |
| 295 | **class `compressed_rows_of`** | Main / index rows of the compressed groups whose ABSOLUTE first positions are |
| 301 | **class `locs_prefill`** | ``[1, end]`` int32: the request's live full locs for positions ``[0, end)`` -- what the indexer |
| 306 | **class `store_main`** | store_main(self, latent |
| 309 | **class `store_index`** | store_index(self, k |
| 312 | **class `decode_store_rows`** | Per-row decode destination: the completed group's arithmetic row, or the row's OWN scratch |
| 320 | **class `ring_page_base`** | ring_page_base(self, window_slots |
| 323 | **class `carry_state_loc`** | carry_state_loc(self, window_slot |
| 327 | **class `read_carry`** | The ``[ring_size, 2 * head_dim]`` carry block at ``window_slot``'s page (kv \| score). |
| 332 | **class `write_carry`** | write_carry(self, source |
| 336 | **class `read_carry_blocks`** | read_carry_blocks(self, source |
| 340 | **class `write_carry_blocks`** | write_carry_blocks(self, source |
| 344 | **class `write_boundary_carries`** | Persist the compressor carry at every window-page boundary ``B`` in ``(lo, hi]`` so a |
| 366 | **class `indexer_logits`** | ``[B, S, T]`` full-range scores (live columns written) or ``[B, S, NC]`` candidate scores; the |
| 377 | **class `select_topk`** | Exact causal top-k over ``[B, S, T]`` full-range scores (columns at or past ``live`` unread): |
| 387 | **class `select_topk_in_candidates`** | Top-k over ``[B, S, NC]`` candidate-aligned scores -> the picked compressed positions |
| 400 | **class `select_candidate_blocks`** | Level one of the Hierarchical Sparse Indexer: the ``topk_blocks`` best live ``block_size`` |
| 412 | **class `positions_to_rows`** | Compressed positions ``[B, S, K]`` (``-1`` masked) -> GLOBAL main rows ``locs[b, p * ratio] // ratio``. |
| 421 | **class `attend`** | Paged sparse attention over ``[window \| compressed]`` global rows; the compressed half reads |

## `freetoken/attention/dsv4_compress.py`  (154 行)

| 行 | 符号 | 说明 |
|---:|---|---|
| 25 | **class `CompressorBackendMixin`** | class CompressorBackendMixin |
| 36 | **class `compress_pool`** | compress_pool(self, layer_id |
| 39 | **class `compress_state_ring`** | compress_state_ring(self, layer_id |
| 42 | **class `compress_scratch_base`** | compress_scratch_base(self, layer_id |
| 46 | **class `compress_rows_of`** | Rows for blocks whose ABSOLUTE start positions are ``block_starts``, off the request's |
| 52 | **class `decode_compress_rows`** | Per-row decode store destination: the completed block's arithmetic row, or the row's |
| 67 | **class `scatter_compressed`** | scatter_compressed( self, layer_id |
| 74 | **class `carry_state_loc`** | The window page's ring block: ``(window_slot // P) * ring_size + arange(ring_size)``. |
| 79 | **class `ring_page_base`** | Per-row ring block base. Distinct window pages map to disjoint blocks, which is what |
| 84 | **class `read_carry`** | The ``[ring_size, 2 * item]`` carry block at ``window_slot``'s page -- kv half then |
| 93 | **class `write_carry`** | write_carry( self, layer_id |
| 107 | **class `read_carry_blocks`** | Per-row carry blocks ``[B, ring_size, 2 * item]`` for a decode step. |
| 114 | **class `write_carry_blocks`** | write_carry_blocks( self, layer_id |
| 121 | **class `write_boundary_carries`** | Persist the carry at EVERY window-page boundary in ``(lo, hi]``, so any future radix |

## `freetoken/attention/dsv4_indexer.py`  (87 行)

| 行 | 符号 | 说明 |
|---:|---|---|
| 19 | **class `IndexerBackendMixin`** | class IndexerBackendMixin |
| 20 | **class `indexer_keys`** | Blocks ``[0, n_blocks)`` of this request's indexer keys as a dense |
| 36 | **class `indexer_prefill_logits`** | Head-reduced scores ``[bsz, seqlen, n_blocks]`` over a dense key slab. |
| 44 | **class `indexer_decode_scores`** | Head-reduced scores ``[B, n_stage]`` for a decode step, gathering each block's key off |
| 58 | **class `indexer_select_prefill`** | Causal top-k over compressed blocks for a prefill/extend range. |
| 76 | **class `indexer_select_decode`** | Top-k over a decode step's staged scores. Columns past each row's live count already |

## `freetoken/attention/dsv4_sparse.py`  (291 行)

| 行 | 符号 | 说明 |
|---:|---|---|
| 47 | **class `DSV4AttnMetadata`** | Per-forward DSV4 addressing. |
| 69 | **class `get_last_indices`** | get_last_indices(self, bs |
| 73 | **class `stage_width`** | Static width of the compressed-candidate staging grids. A capture bakes it, so it |
| 80 | **class `full_snapshot`** | This decode batch's whole-history full locs (the buffer every decode gather reads). |
| 93 | **class `window_ctx`** | The layer-INVARIANT 128-ring context for a decode step: ``(window_slots, |
| 129 | **class `DSV4CaptureData`** | CUDA-graph input buffers. Addresses must stay stable for the life of the capture, so |
| 137 | **class `create`** | create(cls, max_bs |
| 146 | **class `DSV4SparseAttnBackend`** | class DSV4SparseAttnBackend(BaseAttnBackend, CompressorBackendMixin, IndexerBackendMixin) |
| 147 | **class `__init__`** | __init__(self, config |
| 159 | **class `pool`** | pool(self) |
| 163 | **class `forward`** | forward(self, q, k, v, layer_id, batch, attn_spec |
| 169 | **class `prepare_metadata`** | prepare_metadata(self, batch |
| 193 | **class `init_capture_graph`** | init_capture_graph(self, max_seq_len |
| 199 | **class `prepare_for_capture`** | prepare_for_capture(self, batch |
| 208 | **class `prepare_for_replay`** | prepare_for_replay(self, batch |
| 212 | **class `snapshot`** | The current decode batch's snapshot (see ``DSV4AttnMetadata.full_snapshot``); |
| 219 | **class `window_slots_of`** | Window slots for a prefill/extend range, off the request's LIVE full locs (positions |
| 224 | **class `win_cols_to_global`** | Per-query window columns (or -1) -> GLOBAL window-pool slots via ``slot_lut``. |
| 229 | **class `blocks_to_global`** | Compressed BLOCK indices (or -1) -> GLOBAL compressed rows: block b lives at |
| 245 | **class `store_window`** | store_window(self, kv |
| 248 | **class `attend`** | Paged sparse MLA attention over ``[window \| compressed]`` global slots. |
| 270 | **class `_table_rows`** | _table_rows(self, batch |
| 274 | **class `_point_to_capture`** | Stage this batch's rows into the captured snapshot, then hand the model a view of it. |

## `freetoken/attention/fa.py`  (198 行)

| 行 | 符号 | 说明 |
|---:|---|---|
| 18 | **class `FACaptureData`** | class FACaptureData(BaseCaptureData) |
| 23 | **class `FAMetadata`** | class FAMetadata(BaseAttnMetadata) |
| 32 | **class `get_last_indices`** | get_last_indices(self, bs |
| 36 | **class `FlashAttentionBackend`** | class FlashAttentionBackend(BaseAttnBackend) |
| 37 | **class `__init__`** | __init__(self, config |
| 54 | **class `forward`** | forward( self, q |
| 83 | **class `prepare_metadata`** | prepare_metadata(self, batch |
| 123 | **class `init_capture_graph`** | init_capture_graph(self, max_seq_len |
| 131 | **class `prepare_for_capture`** | prepare_for_capture(self, batch |
| 144 | **class `prepare_for_replay`** | prepare_for_replay(self, batch |
| 155 | `_fa_sgl_impl()` | _fa_sgl_impl( q |

## `freetoken/attention/fi.py`  (313 行)

| 行 | 符号 | 说明 |
|---:|---|---|
| 26 | `_next_power_of_2()` | _next_power_of_2(n |
| 36 | **class `FICaptureData`** | class FICaptureData(BaseCaptureData) |
| 38 | **class `one_tensor`** | one_tensor(self) -> torch.Tensor |
| 42 | **class `indices`** | indices(self) -> torch.Tensor |
| 47 | **class `FIMetadata`** | class FIMetadata(BaseAttnMetadata) |
| 65 | **class `__post_init__`** | __post_init__(self) -> None |
| 76 | **class `get_last_indices`** | get_last_indices(self, bs |
| 80 | **class `FlashInferBackend`** | class FlashInferBackend(BaseAttnBackend) |
| 81 | **class `__init__`** | __init__(self, config |
| 147 | **class `_initialize_metadata_once`** | _initialize_metadata_once(self, metadata |
| 192 | **class `_get_ones_cpu`** | _get_ones_cpu(self, bs |
| 200 | **class `forward`** | forward( self, q |
| 214 | &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;↳ `_flatten_cache()` | _flatten_cache(cache |
| 225 | **class `prepare_metadata`** | prepare_metadata(self, batch |
| 262 | **class `reset_capture`** | reset_capture(self) -> None |
| 269 | **class `init_capture_graph`** | init_capture_graph(self, max_seq_len |
| 279 | **class `use_tensor_cores`** | use_tensor_cores(self) -> bool |
| 286 | **class `prepare_for_capture`** | prepare_for_capture(self, batch |
| 308 | **class `prepare_for_replay`** | prepare_for_replay(self, batch |

## `freetoken/attention/linear.py`  (140 行)

| 行 | 符号 | 说明 |
|---:|---|---|
| 13 | **class `FLAMetadata`** | Per-forward GatedDeltaNet (flash-linear-attention) metadata, built once per |
| 47 | `build_fla_metadata()` | Build the per-forward GDN metadata. Uses pinned host staging + non_blocking H2D |
| 63 | &nbsp;&nbsp;&nbsp;↳ `gdn_slot()` | gdn_slot(r) |
| 94 | `_build_track_metadata()` | Hybrid-radix (extra_buffer): for each request that crosses a ×CHUNK boundary this |

## `freetoken/attention/m3_sparse.py`  (443 行)

| 行 | 符号 | 说明 |
|---:|---|---|
| 55 | `_pick_inner_backend()` | FULL backend for the dense leading layers, resolved through the engine's |
| 71 | &nbsp;&nbsp;&nbsp;↳ `_page_ok()` | _page_ok(name |
| 128 | **class `M3SparseMetadata`** | class M3SparseMetadata(BaseAttnMetadata) |
| 146 | **class `get_last_indices`** | get_last_indices(self, bs |
| 150 | **class `M3SparseAttnBackend`** | class M3SparseAttnBackend(BaseAttnBackend) |
| 151 | **class `__init__`** | __init__(self, config |
| 191 | **class `_k_rows`** | _k_rows(self, layer_id |
| 195 | **class `_v_rows`** | _v_rows(self, layer_id |
| 200 | **class `prepare_metadata`** | prepare_metadata(self, batch |
| 224 | **class `forward`** | forward( self, q |
| 243 | **class `bsa_forward`** | bsa_forward( self, q |
| 272 | **class `_decode`** | _decode(self, md, layer_id |
| 297 | **class `_prefill_plan`** | Per-request addressing plan, built ONCE per forward (layer-invariant -- |
| 341 | **class `_prefill`** | _prefill(self, md, layer_id |
| 371 | **class `init_capture_graph`** | init_capture_graph(self, max_seq_len |
| 382 | **class `_block_base_view`** | Every-``block_size``-th column of the page table: the per-block base |
| 389 | **class `_decode_block_rows`** | This decode step's per-request block-base rows [bs, W/128], gathered off |
| 399 | **class `_stage_decode`** | Copy this step's addressing into the static graph buffers and point the |
| 408 | **class `prepare_for_capture`** | prepare_for_capture(self, batch |
| 425 | **class `prepare_for_replay`** | prepare_for_replay(self, batch |
| 436 | **class `reset_capture`** | reset_capture(self) -> None |

## `freetoken/attention/qsa_sparse.py`  (542 行)

| 行 | 符号 | 说明 |
|---:|---|---|
| 58 | `_resolve_block_topk()` | The in-repo Triton block top-k, or None to fall back on torch.topk. |
| 73 | **class `QSASparseMetadata`** | class QSASparseMetadata(BaseAttnMetadata) |
| 97 | **class `get_last_indices`** | get_last_indices(self, bs |
| 101 | **class `QSASparseAttnBackend`** | class QSASparseAttnBackend(BaseAttnBackend) |
| 102 | **class `__init__`** | __init__(self, config |
| 150 | **class `_qsa_group`** | _qsa_group(config |
| 162 | **class `_cmp_pages`** | The compressed slab as ``[pages, page_size // ratio, 1, dim]``, the score kernel's |
| 168 | **class `_index_rope_cache`** | cos/sin table of the indexer rope: same rotary_dim and frequencies as the main |
| 188 | **class `_index_rope_rows`** | Per-token indexer cos/sin rows for [3, n] mrope positions. |
| 195 | **class `prepare_metadata`** | prepare_metadata(self, batch |
| 224 | **class `_block_base_view`** | Every-``page_size``-th column of the page table: the per-page base slots. A strided |
| 229 | **class `_block_table`** | _block_table(self, table_idx |
| 234 | **class `_stage_decode`** | Copy this step's addressing into the static graph buffers and point the metadata |
| 248 | **class `_snapshot_decode`** | Eager decode (not graph-staged): this step's rows, once per forward. The live |
| 261 | **class `forward`** | forward( self, q |
| 276 | **class `qsa_forward`** | qsa_forward( self, q |
| 311 | **class `_plan_index_writes`** | Per-token slab row and ring row for this forward; the other QSA layers reuse it |
| 345 | **class `_update_index_cache`** | Compress each closing group into the slab, then refresh the pending ring. |
| 385 | **class `_select`** | Score complete visible blocks, take the top-k, expand them to token indices. |
| 444 | **class `_top_blocks`** | Top ``block_topk`` complete blocks per row, row-relative, -1 padded. |
| 474 | **class `_topk_scratch_width`** | int32 columns per row the block top-k wants as scratch, 0 when it wants none. |
| 483 | **class `_scratch`** | A per-forward transient: the static decode buffer when it is wide enough (so a |
| 492 | **class `init_capture_graph`** | init_capture_graph(self, max_seq_len |
| 501 | &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;↳ `empty()` | empty(*shape |
| 521 | **class `prepare_for_capture`** | prepare_for_capture(self, batch |
| 531 | **class `prepare_for_replay`** | prepare_for_replay(self, batch |
| 537 | **class `reset_capture`** | reset_capture(self) -> None |

## `freetoken/attention/triton.py`  (355 行)

| 行 | 符号 | 说明 |
|---:|---|---|
| 17 | **class `TritonCaptureData`** | class TritonCaptureData(BaseCaptureData) |
| 25 | **class `create`** | create( cls, max_bs |
| 64 | **class `TritonMetadata`** | class TritonMetadata(BaseAttnMetadata) |
| 78 | **class `get_last_indices`** | get_last_indices(self, bs |
| 82 | **class `TritonAttentionBackend`** | class TritonAttentionBackend(BaseAttnBackend) |
| 83 | **class `__init__`** | __init__(self, config |
| 99 | **class `_ensure_decode_scratch`** | _ensure_decode_scratch( self, metadata |
| 133 | **class `forward`** | forward( self, q |
| 222 | **class `prepare_metadata`** | prepare_metadata(self, batch |
| 275 | **class `init_capture_graph`** | init_capture_graph(self, max_seq_len |
| 295 | **class `_swa_capture_enabled`** | _swa_capture_enabled(self) -> bool |
| 300 | **class `_capture_swa_indices`** | _capture_swa_indices(self) -> torch.Tensor \| None |
| 307 | **class `_point_to_capture`** | _point_to_capture(self, metadata |
| 332 | **class `prepare_for_capture`** | prepare_for_capture(self, batch |
| 351 | **class `prepare_for_replay`** | prepare_for_replay(self, batch |

## `freetoken/attention/trtllm.py`  (172 行)

| 行 | 符号 | 说明 |
|---:|---|---|
| 17 | **class `TRTLLMCaptureData`** | class TRTLLMCaptureData(BaseCaptureData) |
| 22 | **class `TRTLLMMetadata`** | class TRTLLMMetadata(BaseAttnMetadata) |
| 31 | **class `get_last_indices`** | get_last_indices(self, bs |
| 35 | **class `TensorRTLLMBackend`** | class TensorRTLLMBackend(BaseAttnBackend) |
| 36 | **class `__init__`** | __init__(self, config |
| 49 | **class `forward`** | forward( self, q |
| 101 | **class `prepare_metadata`** | prepare_metadata(self, batch |
| 141 | **class `init_capture_graph`** | init_capture_graph(self, max_seq_len |
| 151 | **class `prepare_for_capture`** | prepare_for_capture(self, batch |
| 164 | **class `prepare_for_replay`** | prepare_for_replay(self, batch |

## `freetoken/attention/utils.py`  (23 行)

| 行 | 符号 | 说明 |
|---:|---|---|
| 7 | **class `BaseCaptureData`** | class BaseCaptureData |
| 15 | **class `create`** | create(cls, max_bs |

