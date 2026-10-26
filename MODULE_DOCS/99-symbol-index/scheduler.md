# 符号索引：`scheduler`

自动生成（`ast` 解析，不执行代码）。**11 个文件 / 2491 行 / 130 个符号**。

> 用途：确认某个模块里**到底有哪些**类/函数/方法及其行号，避免遗漏。
> 行为详解请看同目录下的手写模块文档。
> 重新生成：`python3 MODULE_DOCS/tools/gen_symbol_index.py`

| 文件 | 行数 | 符号数 |
|---|---:|---:|
| `freetoken/scheduler/__init__.py` | 4 | 0 |
| `freetoken/scheduler/cache.py` | 650 | 33 |
| `freetoken/scheduler/config.py` | 43 | 7 |
| `freetoken/scheduler/decode.py` | 39 | 7 |
| `freetoken/scheduler/io.py` | 133 | 11 |
| `freetoken/scheduler/mm.py` | 91 | 5 |
| `freetoken/scheduler/prefill.py` | 359 | 16 |
| `freetoken/scheduler/scheduler.py` | 962 | 33 |
| `freetoken/scheduler/status.py` | 144 | 8 |
| `freetoken/scheduler/table.py` | 30 | 6 |
| `freetoken/scheduler/utils.py` | 36 | 4 |

---

## `freetoken/scheduler/__init__.py`  (4 行)

_（无类/函数定义，模块级代码）_

## `freetoken/scheduler/cache.py`  (650 行)

| 行 | 符号 | 说明 |
|---:|---|---|
| 16 | `_swa_eviction_interval()` | _swa_eviction_interval() -> int |
| 32 | **class `CacheManager`** | class CacheManager |
| 33 | **class `__init__`** | __init__(self, num_pages |
| 70 | **class `prefix_replay_tokens`** | Prompt-tail tokens a prefix hit must leave to the prefill (a pool whose model recomputes them). |
| 75 | **class `prefill_chunk_align`** | Granularity a non-final prefill chunk should end on. A hybrid snapshot is donated only |
| 81 | **class `page_usage`** | (used_pages, total_pages): allocated, non-evictable pages over the pool total |
| 89 | **class `_make_prefix_cache`** | _make_prefix_cache(self, device, page_size, type) |
| 98 | **class `match_req`** | match_req(self, req |
| 114 | **class `available_size`** | available_size(self) -> int |
| 120 | **class `mamba_available_size`** | Hybrid only: free GDN state slots + evictable (unlocked) tree snapshots. |
| 125 | **class `swa_available_size`** | SWA only: free swa-pool slots + (radix) evictable unlocked live tree swa tokens. |
| 131 | **class `ensure_swa_slots`** | Free swa-pool slots until >= ``n`` are available by tombstoning LRU tree swa nodes |
| 143 | **class `ensure_mamba_slots`** | Free GDN state slots until >= ``n`` are available by tombstoning LRU tree snapshots |
| 153 | **class `snapshot_toolcall_anchor`** | Freeze each decoding request's GDN state at its tool-call anchor, into the ping-pong |
| 180 | **class `maybe_free_swa_out_of_window`** | Proactively free each decoding request's now-out-of-window SWA slots, bounding its swa |
| 219 | **class `free_swa_out_of_window_extend`** | Prefill sibling of ``maybe_free_swa_out_of_window``: before allocating a chunk, return |
| 242 | **class `lock`** | lock(self, handle |
| 251 | **class `unlock`** | unlock(self, handle |
| 259 | **class `_free_swa`** | Free the swa-pool slots backing ``indices`` (full-pool slots). Idempotent over the |
| 265 | **class `allocate_paged`** | allocate_paged(self, reqs |
| 286 | **class `cache_req`** | cache_req(self, req |
| 332 | **class `_cache_req_hybrid`** | Hybrid (GDN) cache_req: commit KV like radix AND manage the GDN state snapshot. |
| 421 | **class `_cache_req_swa`** | SWA cache_req: commit the request's full KV prefix into the SWARadixCache (node.value = |
| 495 | **class `_padded_tail`** | The request's OWN slice [start, page_ceil(cached_len)) of the page table. A finish |
| 504 | **class `_free_req_slots`** | Return a finished request's GDN pool slots: both ping-pong slots, plus the live slot |
| 516 | **class `check_integrity`** | check_integrity(self) -> None |
| 556 | **class `rebuild`** | Re-point the page table and reset page accounting + prefix cache IN PLACE. |
| 575 | **class `lazy_free_region`** | lazy_free_region(self) |
| 576 | &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;↳ `lazy_free()` | lazy_free(indices |
| 591 | **class `_allocate`** | _allocate(self, needed_pages |
| 613 | **class `_free`** | _free(self, indices |
| 617 | **class `_page_to_token`** | _page_to_token(self, pages |
| 625 | `_write_page_table()` | _write_page_table( page_table |

## `freetoken/scheduler/config.py`  (43 行)

| 行 | 符号 | 说明 |
|---:|---|---|
| 8 | `_get_pid_suffix()` | _get_pid_suffix() -> str |
| 15 | **class `SchedulerConfig`** | class SchedulerConfig(EngineConfig) |
| 26 | **class `zmq_backend_addr`** | zmq_backend_addr(self) -> str |
| 30 | **class `zmq_detokenizer_addr`** | zmq_detokenizer_addr(self) -> str |
| 34 | **class `zmq_scheduler_broadcast_addr`** | zmq_scheduler_broadcast_addr(self) -> str |
| 38 | **class `max_forward_len`** | max_forward_len(self) -> int |
| 42 | **class `backend_create_detokenizer_link`** | backend_create_detokenizer_link(self) -> bool |

## `freetoken/scheduler/decode.py`  (39 行)

| 行 | 符号 | 说明 |
|---:|---|---|
| 10 | **class `DecodeManager`** | class DecodeManager |
| 14 | **class `filter_reqs`** | filter_reqs(self, reqs |
| 17 | **class `remove_req`** | remove_req(self, req |
| 20 | **class `abort_req`** | abort_req(self, uid |
| 28 | **class `inflight_tokens`** | inflight_tokens(self) -> int |
| 32 | **class `schedule_next_batch`** | schedule_next_batch(self) -> Batch \| None |
| 38 | **class `runnable`** | runnable(self) -> bool |

## `freetoken/scheduler/io.py`  (133 行)

| 行 | 符号 | 说明 |
|---:|---|---|
| 15 | **class `SchedulerIOMixin`** | Mixin class for Scheduler I/O operations. |
| 27 | **class `__init__`** | __init__(self, config |
| 67 | **class `run_when_idle`** | run_when_idle(self) |
| 70 | **class `offline_receive_msg`** | offline_receive_msg(self, blocking |
| 73 | **class `offline_send_result`** | offline_send_result(self, reply |
| 76 | **class `sync_all_ranks`** | sync_all_ranks(self) -> None |
| 79 | **class `_recv_msg_single_rank`** | _recv_msg_single_rank(self, blocking |
| 88 | **class `_recv_msg_multi_rank0`** | _recv_msg_multi_rank0(self, blocking |
| 109 | **class `_recv_msg_multi_rank1`** | _recv_msg_multi_rank1(self, blocking |
| 124 | **class `_reply_tokenizer_rank0`** | _reply_tokenizer_rank0(self, reply |
| 132 | **class `_reply_tokenizer_rank1`** | _reply_tokenizer_rank1(self, reply |

## `freetoken/scheduler/mm.py`  (91 行)

| 行 | 符号 | 说明 |
|---:|---|---|
| 14 | `mm_rows_after()` | Embedding rows of the item that prefill will gather once the first cached_len tokens are skipped. |
| 19 | `mm_chunk_end()` | The chunk end pulled back to an align multiple before any image span it would cut; end itself when that span already begins at or before chunk_lo (... |
| 32 | `cut_image_spans()` | The image spans the reqs' current chunks end inside of; mm_chunk_end leaves such a cut only for an image longer than the chunk. |
| 37 | `plan_mm_chunk()` | Encoder jobs and the embedding-row gather plan for the items overlapping the chunk window [window_lo, window_hi). |
| 67 | `plan_mm_batch()` | Jobs, plan, the batch rows of every gathered embedding row and, per batch token, the end of the image span holding it (0 for text), over the reqs i... |

## `freetoken/scheduler/prefill.py`  (359 行)

| 行 | 符号 | 说明 |
|---:|---|---|
| 26 | `_maybe_pinned()` | Pinning only buys the async H2D copy below; without a device it just raises. |
| 31 | **class `ChunkedReq`** | class ChunkedReq(Req) |
| 32 | **class `_alloc_ids_buf`** | _alloc_ids_buf(self) -> None |
| 35 | **class `append_host`** | append_host(self, next_token |
| 39 | **class `can_decode`** | can_decode(self) -> bool |
| 44 | **class `PrefillAdder`** | class PrefillAdder |
| 59 | **class `__post_init__`** | __post_init__(self) -> None |
| 63 | **class `_kv_reservation_size`** | Return the token-equivalent cost of the additional KV pages for a request. |
| 70 | **class `_try_allocate_one`** | _try_allocate_one(self, req |
| 135 | **class `_add_one_req`** | _add_one_req( self, pending_req |
| 232 | **class `try_add_one`** | try_add_one(self, pending_req |
| 274 | **class `PrefillManager`** | class PrefillManager |
| 282 | **class `add_one_req`** | add_one_req(self, req |
| 294 | **class `schedule_next_batch`** | schedule_next_batch(self, prefill_budget |
| 350 | **class `abort_req`** | abort_req(self, uid |
| 358 | **class `runnable`** | runnable(self) -> bool |

## `freetoken/scheduler/scheduler.py`  (962 行)

| 行 | 符号 | 说明 |
|---:|---|---|
| 49 | `_gib()` | _gib(n_bytes |
| 54 | **class `ForwardInput`** | class ForwardInput(NamedTuple) |
| 64 | **class `Scheduler`** | class Scheduler(SchedulerIOMixin) |
| 65 | **class `__init__`** | __init__(self, config |
| 154 | **class `run_when_idle`** | Called when the scheduler is idle to perform background tasks. |
| 160 | **class `rebuild_cache`** | Idle-only runtime cache rebuild: resize the MoE slot cache, KV pages, GDN (mamba) state |
| 207 | **class `overlap_loop`** | The main loop of overlapping scheduling and execution. |
| 263 | **class `normal_loop`** | normal_loop(self) -> None |
| 291 | **class `run_forever`** | run_forever(self) -> NoReturn |
| 307 | **class `shutdown`** | shutdown(self) -> None |
| 312 | **class `_process_last_data`** | _process_last_data(self, last_data |
| 431 | **class `_match_stop_str`** | First stop string present in this request's generated tail, else None. Decodes |
| 447 | **class `_kv_usage_pages`** | (used_pages, total_pages) of the KV page pool. |
| 458 | **class `_mamba_slot_usage`** | (used_slots, total_slots) of the GDN-state (mamba) pool for hybrid models, else None. |
| 469 | **class `_swa_token_usage`** | (used_tokens, total_tokens) of the window (swa) pool for SWA models, else None. |
| 481 | **class `_gpu_mem_bytes`** | Bytes this engine process holds on the GPU (torch's reserved caching-allocator |
| 488 | **class `_process_one_msg`** | _process_one_msg(self, msg |
| 616 | **class `_restore_linear_states`** | COW-restore a hybrid prefix hit's GDN snapshot into its freshly-allocated live slot |
| 628 | **class `_free_req_resources`** | _free_req_resources(self, req |
| 642 | **class `_reply_rebuild`** | _reply_rebuild(self, request_id |
| 661 | **class `_execute_pending_rebuild`** | _execute_pending_rebuild(self) -> None |
| 733 | **class `_current_cache_geometry`** | The pools' current (serving) sizes as rebuild_cache kwargs — the rollback snapshot and |
| 751 | **class `_log_cache_geometry`** | One-line readout of every pool's new size + VRAM after a rebuild changed them: |
| 786 | **class `_prepare_batch`** | _prepare_batch(self, batch |
| 847 | **class `_gather_multimodal`** | Plan the chunk's encoder jobs, gather rows and scatter rows over the batch; the engine runs them before the LM forward. |
| 863 | **class `_schedule_next_batch`** | _schedule_next_batch(self) -> ForwardInput \| None |
| 875 | **class `_report_prompt_admissions`** | Publish first-prefill accounting only after batch preparation succeeded. |
| 890 | **class `_flush_abort_acks`** | _flush_abort_acks(self) -> None |
| 898 | **class `_forward`** | _forward(self, forward_input |
| 909 | `_make_mrope_positions()` | [3, N] rope rows: an image request's prompt tokens use their precomputed columns, everything else is sequence index + per-request delta. |
| 931 | `_make_positions()` | _make_positions(batch |
| 947 | `_make_input_tuple()` | _make_input_tuple(batch |
| 957 | `_make_write_tuple()` | _make_write_tuple(batch |

## `freetoken/scheduler/status.py`  (144 行)

| 行 | 符号 | 说明 |
|---:|---|---|
| 11 | **class `SchedulerStatusReporter`** | class SchedulerStatusReporter |
| 20 | **class `__post_init__`** | __post_init__(self) -> None |
| 26 | **class `report_batch`** | report_batch( self, batch |
| 60 | **class `_report_prefill`** | _report_prefill( self, batch |
| 93 | **class `_report_decode`** | _report_decode( self, batch |
| 127 | `_usage_ratio()` | _usage_ratio(used |
| 131 | `_mamba_msg()` | GDN-state (mamba) pool occupancy for hybrid models; empty for the rest. |
| 139 | `_swa_msg()` | Window (swa) pool occupancy for SWA models; empty for the rest. |

## `freetoken/scheduler/table.py`  (30 行)

| 行 | 符号 | 说明 |
|---:|---|---|
| 4 | **class `TableManager`** | class TableManager |
| 5 | **class `__init__`** | __init__(self, max_running_reqs |
| 14 | **class `available_size`** | available_size(self) -> int |
| 17 | **class `allocate`** | allocate(self) -> int |
| 20 | **class `free`** | free(self, slot |
| 23 | **class `rebuild`** | Re-point the page table, reallocate the token pool, and free all slots. |

## `freetoken/scheduler/utils.py`  (36 行)

| 行 | 符号 | 说明 |
|---:|---|---|
| 15 | **class `PendingReq`** | class PendingReq |
| 25 | **class `input_len`** | input_len(self) -> int |
| 29 | **class `output_len`** | output_len(self) -> int |
| 34 | **class `ScheduleResult`** | class ScheduleResult |

