# 符号索引：`moe`

自动生成（`ast` 解析，不执行代码）。**16 个文件 / 5870 行 / 214 个符号**。

> 用途：确认某个模块里**到底有哪些**类/函数/方法及其行号，避免遗漏。
> 行为详解请看同目录下的手写模块文档。
> 重新生成：`python3 MODULE_DOCS/tools/gen_symbol_index.py`

| 文件 | 行数 | 符号数 |
|---|---:|---:|
| `freetoken/moe/__init__.py` | 17 | 1 |
| `freetoken/moe/bench_profile.py` | 191 | 7 |
| `freetoken/moe/benchbw.py` | 999 | 42 |
| `freetoken/moe/cpu_executor.py` | 697 | 19 |
| `freetoken/moe/expert_banks.py` | 440 | 14 |
| `freetoken/moe/expert_pieces.py` | 157 | 8 |
| `freetoken/moe/fused.py` | 381 | 8 |
| `freetoken/moe/fused_ds_fp4.py` | 234 | 5 |
| `freetoken/moe/fused_fp8_block.py` | 48 | 2 |
| `freetoken/moe/fused_mxfp4.py` | 282 | 6 |
| `freetoken/moe/fused_nvfp4.py` | 336 | 9 |
| `freetoken/moe/fused_q4_0.py` | 53 | 1 |
| `freetoken/moe/host_banks.py` | 488 | 38 |
| `freetoken/moe/legacy_format.py` | 46 | 4 |
| `freetoken/moe/offload_cache.py` | 1070 | 37 |
| `freetoken/moe/offload_kernels.py` | 431 | 13 |

---

## `freetoken/moe/__init__.py`  (17 行)

| 行 | 符号 | 说明 |
|---:|---|---|
| 13 | `is_offload_moe_strategy()` | is_offload_moe_strategy(strategy |

## `freetoken/moe/bench_profile.py`  (191 行)

| 行 | 符号 | 说明 |
|---:|---|---|
| 34 | `_cache_dir()` | _cache_dir() -> str |
| 39 | `default_profile_path()` | ``$XDG_CACHE_HOME/freetoken/benchbw/<gpu-uuid>.json``, or the legacy ``benchbw.json`` without a uuid. |
| 49 | `latest_profile_path()` | Newest ``benchbw/*.json``, else the legacy ``benchbw.json``, else None. |
| 72 | `_load()` | _load(path |
| 80 | `_usable_profile()` | The cached profile, or ``None`` when there is no file / it was benched on another GPU |
| 114 | `load_backend_recommendation()` | Bench-recommended offload-family backend for ``quant_format`` on this GPU, or ``None``. |
| 156 | `load_hybrid_fetch_fraction()` | Benched hybrid fetch fraction for ``quant_format``, or ``None``. |

## `freetoken/moe/benchbw.py`  (999 行)

| 行 | 符号 | 说明 |
|---:|---|---|
| 83 | `_forced_isa()` | Force the CPU MoE ISA tier for executors constructed in this block (None = auto). |
| 101 | **class `Workload`** | class Workload |
| 151 | `default_out_path()` | default_out_path(gpu_uuid |
| 158 | `_cgroup_mem_headroom()` | Free bytes under this process's cgroup v2 memory limit, or None if unlimited. |
| 173 | `_available_ram_bytes()` | Free host RAM, clamped to the cgroup memory limit so a container isn't over-estimated. |
| 186 | `measure_cpu_mem_bw()` | Aggregate host RAM read bandwidth (GB/s) across the physical cores (STREAM read). |
| 223 | &nbsp;&nbsp;&nbsp;↳ `worker()` | worker(idx |
| 258 | `measure_pcie_bw()` | Linear pinned<->device copy bandwidth (GB/s) for H2D and D2H, via CUDA events. |
| 268 | &nbsp;&nbsp;&nbsp;↳ `_timed()` | _timed(copy) -> float |
| 287 | `_offload_bank_specs()` | Per-bank (elems_per_expert, dtype) for the flat offload host banks (the gather). |
| 317 | `_expert_bytes()` | _expert_bytes(fmt |
| 321 | `_synth_experts()` | Expert count for the synthetic banks: as many as fit the memory budget (bandwidth is |
| 329 | `_cpu_moe_bank_sources()` | 3D pinned banks in the exact layout ``CpuMoeExecutor`` expects for ``fmt``. |
| 337 | &nbsp;&nbsp;&nbsp;↳ `pin()` | pin(*shape, dtype) |
| 387 | `_build_gather_rig()` | Production ``copy_missing`` rig: an ``OffloadMoeCache`` over synthetic pinned banks, |
| 418 | `measure_pcie_gather_bw()` | Real PCIe gather bandwidth (GB/s): pinned host banks -> GPU slot cache. |
| 451 | `_build_cpu_moe_executor()` | A production ``CpuMoeExecutor`` over synthetic banks (via a minimal cache stand-in). |
| 469 | `_cpu_moe_step_fns()` | ``(set_ids, run)`` closures for bs=1 decode steps on a constructed executor. |
| 482 | &nbsp;&nbsp;&nbsp;↳ `set_ids()` | set_ids(i |
| 485 | &nbsp;&nbsp;&nbsp;↳ `run()` | run() -> None |
| 491 | `_time_cpu_moe()` | bs=1 CPU MoE decode bandwidth (GB/s) for a constructed executor, median-timed. |
| 508 | `measure_cpu_moe_bw()` | Real CPU MoE GEMV bandwidth (GB/s) at bs=1, reading experts from pinned host banks. |
| 523 | &nbsp;&nbsp;&nbsp;↳ `run()` | run(isa |
| 540 | `measure_overlap_bw()` | Concurrent achieved bandwidths (GB/s): the CPU MoE GEMV and the PCIe gather running |
| 574 | &nbsp;&nbsp;&nbsp;↳ `cpu_side()` | cpu_side() -> None |
| 600 | `recommend()` | ``hybrid`` iff CPU bandwidth exceeds ``threshold`` x PCIe bandwidth, else ``offload``. |
| 608 | `_note()` | _note(entry |
| 612 | `_bench_format()` | Per (workload, format): real PCIe gather + real CPU MoE GEMV + hybrid/offload verdict. |
| 684 | `_atomic_write_json()` | _atomic_write_json(path |
| 693 | `run_benchbw()` | run_benchbw( out_path |
| 733 | &nbsp;&nbsp;&nbsp;↳ `_prog()` | _prog(label |
| 814 | `_gbs()` | _gbs(x) -> str |
| 818 | `_print_kernels()` | _print_kernels(kernels |
| 840 | `_print_report()` | _print_report(r |
| 867 | `_positive_int()` | _positive_int(s |
| 874 | `_nonneg_int()` | _nonneg_int(s |
| 881 | `_positive_float()` | _positive_float(s |
| 888 | `_model_list()` | _model_list(s |
| 900 | `_dtype_list()` | _dtype_list(s |
| 913 | `_format_list()` | _format_list(s |
| 924 | `_isa_list()` | _isa_list(s |
| 936 | `main()` | main(argv |

## `freetoken/moe/cpu_executor.py`  (697 行)

| 行 | 符号 | 说明 |
|---:|---|---|
| 77 | `compiled_extension_supports()` | Whether the compiled ``_cpu_moe`` extension can serve ``activation`` |
| 94 | `physical_core_cpus()` | One logical CPU per physical core, restricted to this process's affinity. |
| 121 | `_spin_wait_ok()` | Spin only if each pool thread owns a distinct CPU and two CPUs stay free for the main and CUDA callback threads. |
| 132 | `resolve_threads_and_affinity()` | Return (num_threads, core_ids) for the worker pool. |
| 157 | **class `CpuMoeExecutor`** | Decode-time CPU expert compute over an ``OffloadMoeCache``'s host banks |
| 161 | **class `__init__`** | __init__( self, cache, *, top_k |
| 347 | **class `_make_table`** | Build a CPU int64 tensor of per-layer base addresses for one bank. |
| 362 | **class `_resolve_banks`** | Return (pointer kwargs for the C++ ctor, (H, I)) for the given format. |
| 429 | **class `_resolve_q4_0_banks`** | Native GGUF Q4_0 schema (gemma4 GGUF): per-32 blocks (fp16 scale + 16 nibble |
| 456 | **class `_resolve_mxfp4_banks`** | gpt-oss mxfp4 ``mxfp4_triton`` schema: transposed split-K blocks/scales |
| 487 | **class `_resolve_dsfp4_banks`** | DeepSeek-V4 ``ds_fp4`` schema: row-major e2m1 (2/byte) + e8m0 per-32 block |
| 515 | **class `_io_for`** | _io_for(self, bs |
| 527 | **class `_task_for`** | _task_for(self, layer_id |
| 550 | **class `decode`** | One MoE layer of decode on the CPU. Returns a GPU [bs, H] tensor. |
| 565 | **class `decode_submit`** | Issue the D2H copies + the CPU-pool submit host node, then return without |
| 611 | **class `decode_sync`** | Issue the CPU-pool sync + the H2D result copy for a prior :meth:`decode_submit`, |
| 629 | **class `_watchdog_tick`** | One watchdog sampling round (called every 2 s by ``_watchdog_main``). |
| 670 | **class `raise_if_unhealthy`** | Raise if the flag watchdog fired (a doorbell stayed unanswered because the |
| 685 | `_watchdog_main()` | Watchdog daemon body: weakref-deref per tick so the thread never keeps a dead |

## `freetoken/moe/expert_banks.py`  (440 行)

| 行 | 符号 | 说明 |
|---:|---|---|
| 31 | **class `ExpertBanks`** | Loaded expert banks, normalized for ``OffloadMoeCache`` wiring. |
| 53 | `_dummy_fill()` | Random but finite bank contents for --use-dummy-weight. |
| 70 | `build_expert_banks()` | Fill host banks in the kernel's layout from a stream of expert pieces. |
| 122 | &nbsp;&nbsp;&nbsp;↳ `_fill()` | _fill(sink) -> None |
| 162 | `_q4_0_banks()` | _q4_0_banks(model_path, model_config, device, dtype, dummy, parallel=False, workers=8, chunk=_PARALLEL_CHUNK, decode_target="gpu", layer_sink=None)... |
| 188 | `_legacy_expert_banks()` | _legacy_expert_banks(model_path, model_config, device, dtype, dummy, parallel, workers, chunk, decode_target="gpu", layer_sink=None) -> ExpertBanks |
| 202 | `_method_expert_banks()` | _method_expert_banks(model_path, model_config, method, device, dummy, parallel, workers, chunk, layer_sink=None, resident=False) -> ExpertBanks |
| 214 | `_host_ram_fits_parallel()` | Best-effort: can free host RAM hold the expert banks plus the parallel reader's one |
| 242 | `ftw_bank_bytes()` | Total expert-bank bytes of an FTW checkpoint, from its metadata (no bank IO). |
| 255 | `bank_bytes_estimate()` | Estimated total expert-bank bytes of a raw checkpoint before loading it. |
| 282 | `load_expert_banks()` | Load (or fabricate, with ``dummy=True``) the expert banks. Two paths, both returning |
| 370 | &nbsp;&nbsp;&nbsp;↳ `_build()` | _build(par |
| 386 | `_echo_residency()` | Stamp an honored residency request onto the ExpertBanks; keep None (and warn) when no settle point consulted the plan. |
| 413 | `attach_resident_banks()` | Give each resident MoE layer its rows of the device ``banks``; bank layer ``i`` goes to the layer whose ``layer_id`` is ``i``. |

## `freetoken/moe/expert_pieces.py`  (157 行)

| 行 | 符号 | 说明 |
|---:|---|---|
| 20 | `num_moe_layers()` | num_moe_layers(config) -> int |
| 27 | `bank_layer_of()` | MoE-layer index of checkpoint layer ``layer``; None for the leading dense layers. |
| 33 | `_model_hook()` | _model_hook(spec, name |
| 40 | `iter_expert_pieces()` | The pieces of ``model_path``'s routed experts, stored as ``kind``. |
| 71 | `packed_expert_source_info()` | ``model.layers.N....experts.{gate_up_proj,down_proj}`` -> (N, leaf); None otherwise. |
| 84 | `stacked_expert_pieces()` | Whole-layer ``experts.gate_up_proj`` / ``experts.down_proj`` tensors -> one piece per layer. |
| 107 | `_bf16_pieces()` | _bf16_pieces(model_path |
| 122 | `per_expert_pieces()` | Group per-expert tensors into one piece per expert, in whatever order they arrive. |

## `freetoken/moe/fused.py`  (381 行)

| 行 | 符号 | 说明 |
|---:|---|---|
| 11 | `_torch_fused_topk()` | Pure-torch reference for the fused softmax router; tests compare the kernel against it. |
| 33 | `fused_topk()` | fused_topk( hidden_states |
| 47 | `moe_align_block_size()` | Aligns the token distribution across experts to be compatible with block |
| 120 | `get_default_config()` | get_default_config( M |
| 145 | `_get_tuned_moe_configs()` | _get_tuned_moe_configs( num_experts |
| 168 | `try_get_optimal_moe_config()` | try_get_optimal_moe_config( w1_shape |
| 192 | `fused_experts_impl()` | Returns ``hidden_states`` itself, overwritten with the routed output. A caller that |
| 300 | `fused_experts_decode_impl()` | fused_experts_decode_impl( hidden_states |

## `freetoken/moe/fused_ds_fp4.py`  (234 行)

| 行 | 符号 | 说明 |
|---:|---|---|
| 27 | `_compute_type()` | _compute_type(dtype |
| 37 | `_grouped_decode()` | Grouped per-route GEMM ``c[t,r,:] = a[row] @ dequant(W[slot])^T``. |
| 94 | `routed_experts_fp4()` | Full routed-expert output (summed over the top-k routes), excludes shared expert. |
| 140 | `_grouped_prefill()` | _grouped_prefill( a |
| 181 | `routed_experts_fp4_prefill()` | Grouped-GEMM counterpart of :func:`routed_experts_fp4` for dense prefill |

## `freetoken/moe/fused_fp8_block.py`  (48 行)

| 行 | 符号 | 说明 |
|---:|---|---|
| 16 | `fused_experts_fp8_block()` | Prefill: W8A8 fused grouped GEMM over the materialized-layer banks |
| 32 | `fused_experts_decode_fp8_block()` | Decode: W8A16 fused inline-dequant grouped GEMV -- reads the routed experts' fp8 rows |

## `freetoken/moe/fused_mxfp4.py`  (282 行)

| 行 | 符号 | 说明 |
|---:|---|---|
| 19 | `gpt_oss_swiglu()` | gpt_oss_swiglu(gate_up |
| 28 | `dequant_mxfp4_blocks()` | dequant_mxfp4_blocks( blocks |
| 75 | `run_mxfp4_prefill_experts_t()` | Prefill experts using the transposed weight layout shared with split-K decode |
| 189 | `_transpose_mxfp4_for_decode()` | HF MXFP4 [E, N, K//32, 16] blocks / [E, N, K//32] scales -> split-K GEMV |
| 200 | `_decode_split_count()` | Pick the split-K count so the GEMV launches ~target_programs program-rows. |
| 207 | `run_mxfp4_splitk_decode_experts()` | Split-K GEMV MoE decode over TRANSPOSED MXFP4 weights: |

## `freetoken/moe/fused_nvfp4.py`  (336 行)

| 行 | 符号 | 说明 |
|---:|---|---|
| 48 | `_tl_dtype()` | _tl_dtype(dt |
| 56 | `_decode_gemm()` | _decode_gemm( a |
| 94 | `_decode_gemm_marlin()` | Marlin-style decode GEMV: int32 wide loads + deferred reduction |
| 139 | `_fused_experts_decode_nvfp4()` | Shared decode body (gemm1 -> act -> gemm2 -> sum-reduce); ``gemm_fn`` is either |
| 181 | `fused_experts_decode_nvfp4_marlin()` | Decode inline-NVFP4 MoE using the Marlin-style int32 wide-load GEMV. |
| 206 | `fused_experts_decode_nvfp4_serial()` | Original LUT-gather decode (one program per route, full K reduction). Retained for |
| 232 | `_prefill_config()` | _prefill_config(M |
| 243 | `_prefill_gemm()` | _prefill_gemm( a |
| 283 | `fused_experts_nvfp4()` | Prefill inline-NVFP4 MoE. ``topk_ids`` index rows of the bank tensors in |

## `freetoken/moe/fused_q4_0.py`  (53 行)

| 行 | 符号 | 说明 |
|---:|---|---|
| 22 | `fused_experts_gguf_q4_0()` | fused_experts_gguf_q4_0( hidden_states |

## `freetoken/moe/host_banks.py`  (488 行)

| 行 | 符号 | 说明 |
|---:|---|---|
| 39 | **class `PinFailed`** | cudaHostRegister refused a bank: the host is out of pinnable RAM or over its pin quota. |
| 43 | **class `HostResidency`** | Residency class of a host bank layer. |
| 60 | `_env_born_pinned()` | ``FREETOKEN_BANK_CUDA_ALLOC`` tri-state: unset -> ``None`` (default applies), else the parsed boolean. |
| 68 | `born_pinned_default()` | Whether PINNED serving banks use cudaHostAlloc instead of mmap + register-after-fill. |
| 78 | **class `HostBank`** | A page-aligned host buffer + its torch view, page-locked on demand: allocate -> fill -> ``pin()``/``lock()``. |
| 88 | **class `__init__`** | __init__(self, shape |
| 120 | **class `residency`** | residency(self) -> HostResidency |
| 127 | **class `memoryview`** | memoryview(self) -> memoryview |
| 130 | **class `pin`** | cudaHostRegister the (now-filled) buffer -- pin-after-fill. |
| 146 | **class `release`** | Drop the resident pages; the address space stays valid, the contents become undefined. |
| 154 | **class `lock`** | mlock the (now-filled) buffer: resident without CUDA pin quota, but no device address -- only the CPU executor can serve a locked layer. |
| 176 | `_os_lock()` | _os_lock(addr |
| 202 | `alloc_banks()` | Allocate (lazy, unpinned) host banks from ``{name: (shape, dtype)}``. |
| 207 | `alloc_layer_banks()` | Allocate per-layer host banks: ``{name: ([num_experts, ...] row shape, dtype)}`` |
| 219 | **class `_ResidencyPlan`** | Per-layer ``HostResidency`` labels, ambiently visible to the bank settle points. |
| 226 | **class `__init__`** | __init__(self, labels |
| 232 | **class `residency_for`** | residency_for(self, layer_id |
| 236 | **class `record`** | One pageable bank downgrades the whole layer (a failed lock settles PAGEABLE). |
| 246 | `requested_residency()` | Install the ambient per-layer residency plan for the enclosed bank load (``None`` = no plan, everything pins). |
| 260 | `_settle()` | Route a filled bank to its residency class (PAGEABLE = leave the plain mmap). |
| 268 | `pin_banks()` | Settle every bank after it has been filled -- pin-after-fill by default. |
| 286 | **class `PinPipeline`** | Settle (pin or lock) filled banks while other banks are still being read. |
| 294 | **class `__init__`** | __init__(self) -> None |
| 302 | **class `_run`** | _run(self) -> None |
| 319 | **class `submit`** | submit(self, bank |
| 323 | **class `__call__`** | Layer-completion sink: queue every bank of the completed layer at its ambient :func:`requested_residency` label. |
| 332 | **class `_join`** | _join(self) -> None |
| 336 | **class `wait`** | wait(self) -> None |
| 341 | **class `__enter__`** | __enter__(self) -> "PinPipeline" |
| 344 | **class `__exit__`** | __exit__(self, exc_type, exc, tb) -> None |
| 351 | **class `LayerCompletionTracker`** | Fire a sink once per layer, when all of that layer's writes have landed. |
| 360 | **class `__init__`** | __init__( self, expected_per_layer |
| 373 | **class `note`** | note(self, layer_id |
| 382 | `read_file_into()` | Chunked multi-threaded O_DIRECT read of the whole file ``path`` into ``buf`` |
| 398 | &nbsp;&nbsp;&nbsp;↳ `rd()` | rd(o) |
| 415 | `_preadv_all()` | preadv into ``dst`` until ``need`` bytes have landed; O_DIRECT may return a short count. |
| 427 | `read_range_into()` | Chunked multi-threaded O_DIRECT read of ``path[file_offset : file_offset + nbytes]`` into ``buf`` at ``dest_offset``. Returns ``nbytes``. |
| 448 | &nbsp;&nbsp;&nbsp;↳ `rd()` | rd(i |

## `freetoken/moe/legacy_format.py`  (46 行)

| 行 | 符号 | 说明 |
|---:|---|---|
| 17 | `canonical_role()` | canonical_role(bank_name |
| 21 | `legacy_bank_names()` | Canonical role -> the bank name FTW files store for ``quant_format``. |
| 41 | `legacy_format_for()` | legacy_format_for(kind |
| 45 | `kind_kernel_for()` | kind_kernel_for(legacy_format |

## `freetoken/moe/offload_cache.py`  (1070 行)

| 行 | 符号 | 说明 |
|---:|---|---|
| 105 | **class `OffloadMoeCache`** | class OffloadMoeCache |
| 149 | **class `__post_init__`** | __post_init__(self) -> None |
| 294 | **class `set_bank_sources`** | Attach the host (CPU pinned) expert source banks and allocate a GPU slot |
| 369 | **class `_build_copy_plan`** | _build_copy_plan(self) -> None |
| 382 | **class `_build_fused_copy_plan`** | Precompute the fused multi-bank copy descriptor (base addrs + per-row bytes). |
| 446 | **class `validate_rebuild`** | Pure geometry validation of a rebuild target (no GPU side effects). |
| 468 | **class `rebuild`** | Resize the GPU slot cache + bookkeeping to ``cache_size`` IN PLACE. |
| 539 | **class `set_alphas`** | Attach the marlin/b12x per-expert global scales (``[L*E]``, GPU resident). |
| 558 | **class `set_cpu_executor`** | Attach the CPU MoE executor (``decode_target`` in {"cpu", "hybrid"}). |
| 570 | **class `is_cpu_layer`** | Whether ``layer_id`` decodes on the CPU executor (vs the GPU offload path). |
| 574 | **class `is_unpinned_layer`** | Whether ``layer_id``'s host banks have no device address (LOCKED/PAGEABLE): the GPU slot-gather paths cannot serve it. |
| 579 | **class `alphas_for_slots`** | Per-slot global scales for a decode call, or ``None`` when the format |
| 591 | **class `alphas_for_layer`** | Global scales for a full-layer prefill (overlap or materialize), where |
| 601 | **class `bank_views`** | Per-bank cache views in registration order: the full ``[S]`` slot cache |
| 609 | **class `_init_prefill_overlap_buffers`** | _init_prefill_overlap_buffers(self) -> None |
| 638 | **class `_invalidate_prefill_buffer`** | _invalidate_prefill_buffer(self, buffer_id |
| 648 | **class `begin_prefill`** | begin_prefill(self) -> None |
| 671 | **class `prefetch_prefill_layer`** | prefetch_prefill_layer(self, layer_id |
| 687 | &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;↳ `copy()` | copy() -> None |
| 706 | **class `_hit_d2d_usable`** | Whether the hit-D2D split can serve this prefill; logs the first fallback. |
| 738 | **class `_resolve_batch_memcpy`** | _resolve_batch_memcpy(self) -> bool |
| 749 | **class `_prefetch_split`** | Hit/miss-split prefetch of one expert layer into the double buffer. |
| 821 | **class `wait_prefill_layer`** | Full-layer ``[num_experts, ...]`` bank views for ``layer_id``, one per |
| 835 | **class `release_prefill_layer`** | release_prefill_layer(self, layer_id |
| 846 | **class `ensure_experts`** | ensure_experts(self, layer_id |
| 858 | **class `ensure_experts_hybrid`** | Capped-fetch LRU for the hybrid backend. |
| 879 | **class `materialize_layer`** | materialize_layer(self, layer_id |
| 886 | **class `reset`** | reset(self) -> None |
| 894 | **class `reset_stats`** | reset_stats(self) -> None |
| 907 | **class `record_decode_stats`** | No-op: ``ensure_experts`` accumulates into ``lru_stats`` inside its own launch. |
| 914 | **class `record_decode_stats_hybrid`** | Hybrid stats: full miss count (pre-cap), the PCIe-fetched count (capped), and |
| 931 | **class `decode_miss_stats`** | decode_miss_stats(self) -> dict |
| 954 | **class `decode_miss_stats_per_layer`** | Per-MoE-layer realized decode stats for one (reset_stats-delimited) window. |
| 982 | **class `decode_routing_stats`** | Per-layer decode routing concentration, for cache-skew analysis. |
| 1014 | **class `copy_missing`** | copy_missing(self) -> None |
| 1060 | `iter_offload_moe_layers()` | iter_offload_moe_layers(model) -> Iterator |
| 1066 | `attach_offload_moe_cache()` | attach_offload_moe_cache(model, cache |

## `freetoken/moe/offload_kernels.py`  (431 行)

| 行 | 符号 | 说明 |
|---:|---|---|
| 19 | `ensure_experts()` | Make this layer's routed experts resident; rewrite ``expert_ids`` to slot ids. |
| 43 | `ensure_experts_hybrid()` | Capped-fetch variant of ``ensure_experts`` (hybrid backend). |
| 63 | `prefill_hit_compact()` | Compact this layer's cache-resident experts into gather indices, device-side. |
| 85 | `materialize_layer()` | materialize_layer(cache, layer_id |
| 89 | `reset_cache()` | reset_cache(cache) -> None |
| 97 | `_ensure_experts_hybrid_gpu()` | _ensure_experts_hybrid_gpu( cache, layer_id |
| 128 | `_ensure_experts_hybrid_cpu()` | CPU reference mirror of the hybrid kernel (eviction/fetch decisions bit-identical to |
| 191 | `_materialize_layer_gpu()` | _materialize_layer_gpu(cache, layer_id |
| 208 | `_reset_cache_gpu()` | _reset_cache_gpu(cache) -> None |
| 227 | `_reset_cache_kernel()` | _reset_cache_kernel( slot_for_id_ptr, id_of_slot_ptr, usage_ptr, step_ptr, active_mask_ptr, num_indices_ptr, total_ids |
| 250 | `_materialize_layer_kernel()` | _materialize_layer_kernel( slot_for_id_ptr, id_of_slot_ptr, usage_ptr, step_ptr, evict_slots_ptr, src_indices_ptr, num_indices_ptr, layer_id |
| 291 | `_ensure_experts_hybrid_kernel()` | Capped-fetch timestamp-LRU (hybrid backend). |
| 414 | `_prefill_hit_compact_kernel()` | _prefill_hit_compact_kernel( slot_ptr,     # [num_experts] int32 |

