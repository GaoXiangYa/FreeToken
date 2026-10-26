# 符号索引：`engine`

自动生成（`ast` 解析，不执行代码）。**6 个文件 / 2495 行 / 92 个符号**。

> 用途：确认某个模块里**到底有哪些**类/函数/方法及其行号，避免遗漏。
> 行为详解请看同目录下的手写模块文档。
> 重新生成：`python3 MODULE_DOCS/tools/gen_symbol_index.py`

| 文件 | 行数 | 符号数 |
|---|---:|---:|
| `freetoken/engine/__init__.py` | 5 | 0 |
| `freetoken/engine/cache_budget.py` | 133 | 5 |
| `freetoken/engine/config.py` | 158 | 10 |
| `freetoken/engine/engine.py` | 1868 | 57 |
| `freetoken/engine/graph.py` | 232 | 14 |
| `freetoken/engine/sample.py` | 99 | 6 |

---

## `freetoken/engine/__init__.py`  (5 行)

_（无类/函数定义，模块级代码）_

## `freetoken/engine/cache_budget.py`  (133 行)

| 行 | 符号 | 说明 |
|---:|---|---|
| 17 | `expert_bytes_per_slot()` | Bytes one expert slot occupies on GPU: summed row bytes over all banks. |
| 31 | `net_cache_budget_bytes()` | Net GPU bytes available for the MoE + KV pools: ``memory_ratio`` of the pre-model |
| 41 | `required_bytes()` | GPU bytes a ``(moe_cache_size, num_pages)`` geometry occupies (MoE slots + KV pages). |
| 48 | `plan_cache_budget()` | Split ``budget_bytes`` MoE-first into (moe_cache_size, num_pages, prefill_overlap). |
| 98 | `resolve_moe_cache_auto()` | Resolve --moe-cache-auto into (moe_cache_size, num_pages, prefill_overlap). |

## `freetoken/engine/config.py`  (158 行)

| 行 | 符号 | 说明 |
|---:|---|---|
| 22 | **class `EngineConfig`** | class EngineConfig |
| 100 | **class `__post_init__`** | __post_init__(self) |
| 110 | **class `hf_config`** | hf_config(self) |
| 114 | **class `model_spec`** | model_spec(self) -> ModelSpec |
| 118 | **class `active_encoders`** | The encoder towers this process builds: the family registers them, the checkpoint config carries their section, --mm-disable did not name them. |
| 128 | **class `served_modalities`** | Modalities this process accepts. |
| 133 | **class `model_config`** | model_config(self) -> ModelConfig |
| 147 | **class `max_seq_len`** | max_seq_len(self) -> int |
| 153 | **class `max_forward_len`** | max_forward_len(self) -> int |
| 157 | **class `distributed_addr`** | distributed_addr(self) -> str |

## `freetoken/engine/engine.py`  (1868 行)

| 行 | 符号 | 说明 |
|---:|---|---|
| 41 | `_require_offload_cache_size()` | The offload MoE cache needs at least one slot per expert per layer. A too-small size |
| 54 | `_flashinfer_available()` | _flashinfer_available() -> bool |
| 60 | `_sgl_flash_attn_available()` | _sgl_flash_attn_available() -> bool |
| 73 | `_startup_kv_budget()` | Bytes available to the KV pool at startup: ratio-scaled pre-load free memory minus |
| 80 | `_page_table_width()` | Column count for the page table. ``_write_page_table`` writes WHOLE trailing pages, so the |
| 88 | `_required_attn_types()` | Backend-driving attention types of this model, from the group-spec walk |
| 104 | `_backend_parts_serve()` | _backend_parts_serve(name |
| 111 | `_backend_requirements_met()` | _backend_requirements_met(name |
| 124 | `_resolve_auto_attention_backend()` | First candidate (in per-type priority order) whose arch condition holds, |
| 163 | `_validate_attention_backend_choice()` | Config-time type x backend capability check for the resolved (or explicit) |
| 251 | `_make_dummy_weight_state_dict()` | _make_dummy_weight_state_dict( model_state |
| 280 | **class `WeightLoadError`** | The checkpoint itself could not be read. Resource and config failures keep their own type. |
| 284 | `_is_resource_failure()` | _is_resource_failure(exc |
| 295 | `_weight_load_context()` | Wrap a checkpoint read so the startup failure reason (log and /health) starts with WeightLoadError. |
| 305 | `_materialize_loaded_weight_state_dict()` | _materialize_loaded_weight_state_dict( model_state |
| 324 | **class `ForwardOutput`** | class ForwardOutput(NamedTuple) |
| 330 | **class `Engine`** | class Engine |
| 331 | **class `__init__`** | __init__(self, config |
| 503 | **class `_init_communication`** | _init_communication(self, config |
| 530 | **class `_load_weights`** | _load_weights(self, config |
| 546 | **class `_load_weight_state_dict`** | _load_weight_state_dict(self, config |
| 564 | **class `_load_resident_experts`** | _load_resident_experts(self, config |
| 579 | **class `_warmup_encoders`** | _warmup_encoders(self) -> None |
| 586 | **class `_run_mm_encoder`** | Encode the chunk's cache-miss items and gather its embedding rows into batch.mm_embeds, right before the LM forward. |
| 611 | **class `_resolve_auto_moe_cache_size`** | Resolve --moe-cache-auto into (moe_cache_size, num_pages, prefill_overlap). |
| 638 | **class `_init_offload_moe_cache`** | _init_offload_moe_cache(self, config |
| 766 | **class `_resolve_hybrid_fetch`** | Resolve --moe-hybrid-max-fetch -1 (auto) into a bandwidth-matched fetch fraction. |
| 797 | **class `_init_cpu_moe_executor`** | Build the persistent CPU MoE executor (decode-time expert compute). |
| 835 | **class `_sync_get_memory`** | Get the min and max free memory across TP ranks. |
| 856 | **class `_target_moe_and_expert_bytes`** | _target_moe_and_expert_bytes(self, moe_cache_size |
| 870 | **class `_resize_kv_pool`** | _resize_kv_pool(self, config, num_pages |
| 881 | **class `_refresh_seq_state`** | _refresh_seq_state(self, config) -> None |
| 898 | **class `rebuild_runtime_cache`** | Idle-only in-place resize of the MoE slot cache, KV page pool, GDN (mamba) state pool, |
| 1044 | **class `forward_batch`** | forward_batch(self, batch |
| 1067 | **class `_warmup_prefill`** | Compile the Triton prefill path before the first real request. |
| 1127 | **class `shutdown`** | shutdown(self) -> None |
| 1133 | `_profile_gpu()` | (name, uuid) of visible device ``index`` (default: the current, i.e. bound, device); (None, None) without CUDA. |
| 1141 | `_is_unified_memory_gpu()` | True when the GPU has no separate device memory (cudaDevAttrIntegrated): host |
| 1165 | `_fused_resident_ok()` | Whether the resident ('fused') MoE path can hold this model's experts; GGUF q4_0 and DeepSeek-V4 experts stay offload. |
| 1172 | `_ensure_expandable_segments()` | Default the CUDA allocator to expandable segments. |
| 1198 | `_resolve_cache_type()` | _resolve_cache_type(has_linear_attention |
| 1207 | `_adjust_dsv41_config()` | DeepSeek-V4.1 config reconciliation, the DSV4 policy with the replay knob: sync the |
| 1240 | `_adjust_dsv4_config()` | DSV4 engine-config reconciliation at config-resolution time (before the pool exists). |
| 1288 | `_parse_cpu_layers_spec()` | Parse ``--moe-cpu-layers``: an explicit MoE-layer id list (``"3,7,11"``), a count |
| 1317 | `_resolve_cpu_layers()` | MoE layer ids whose decode runs on the CPU executor. |
| 1333 | `_decode_target()` | Where routed experts decode, from the flags alone: hybrid co-compute, the CPU executor for some or all layers, or the GPU slot cache. |
| 1349 | `_cpu_moe_executor_viable()` | Whether an automatic CPU-decode decision may target the CPU MoE executor. |
| 1370 | `_pin_budget_bytes()` | Bytes this process can still safely cudaHostRegister, or None when the platform does not cap pinning (plain Linux). |
| 1383 | `_bank_bytes()` | _bank_bytes(config |
| 1389 | `_pin_hint()` | _pin_hint(reserved |
| 1398 | `_check_pin_budget()` | Stop a plain offload boot whose banks exceed a known pin budget before any bank is read. |
| 1411 | `_auto_cpu_layers()` | Pick CPU (locked) MoE layers for ``--moe-cpu-layers auto``: none while the banks fit the pin budget. |
| 1453 | `_adjust_ftw_quant_backend()` | --quant-backend with an FTW checkpoint's packed expert kernel filled in where the flag leaves that table automatic. |
| 1479 | `shared_offload_method()` | The expert method every offload MoE layer of ``model`` uses, or None for models whose MoE layers carry none (GGUF). |
| 1495 | `offload_expert_method()` | The offload expert method of ``config``'s model, from a meta-device build. |
| 1510 | `_adjust_config()` | _adjust_config(config |
| 1511 | &nbsp;&nbsp;&nbsp;↳ `override()` | override(attr |

## `freetoken/engine/graph.py`  (232 行)

| 行 | 符号 | 说明 |
|---:|---|---|
| 23 | **class `GraphCaptureBuffer`** | class GraphCaptureBuffer |
| 35 | **class `init`** | init( cls, bs |
| 50 | **class `set_batch`** | set_batch(self, batch |
| 67 | **class `copy_from`** | copy_from(self, batch |
| 79 | `_determine_cuda_graph_bs()` | _determine_cuda_graph_bs( cuda_graph_bs |
| 101 | `get_free_memory()` | get_free_memory(device |
| 105 | **class `GraphRunner`** | class GraphRunner |
| 106 | **class `__init__`** | __init__( self, stream |
| 136 | **class `_reset_moe_offload_cache`** | _reset_moe_offload_cache(self) -> None |
| 140 | **class `_capture_graphs`** | _capture_graphs(self, max_seq_len |
| 204 | **class `can_use_cuda_graph`** | can_use_cuda_graph(self, batch |
| 207 | **class `replay`** | replay(self, batch |
| 215 | **class `pad_batch`** | pad_batch(self, batch |
| 224 | **class `destroy_cuda_graphs`** | destroy_cuda_graphs(self) -> None |

## `freetoken/engine/sample.py`  (99 行)

| 行 | 符号 | 说明 |
|---:|---|---|
| 14 | **class `BatchSamplingArgs`** | class BatchSamplingArgs |
| 21 | `make_device_tensor()` | make_device_tensor(data |
| 25 | `sample_impl()` | sample_impl( logits |
| 55 | **class `Sampler`** | class Sampler |
| 59 | **class `prepare`** | prepare(self, batch |
| 89 | **class `sample`** | sample(self, logits |

