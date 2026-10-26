# 符号索引：`layers`

自动生成（`ast` 解析，不执行代码）。**36 个文件 / 4420 行 / 350 个符号**。

> 用途：确认某个模块里**到底有哪些**类/函数/方法及其行号，避免遗漏。
> 行为详解请看同目录下的手写模块文档。
> 重新生成：`python3 MODULE_DOCS/tools/gen_symbol_index.py`

| 文件 | 行数 | 符号数 |
|---|---:|---:|
| `freetoken/layers/__init__.py` | 62 | 0 |
| `freetoken/layers/activation.py` | 94 | 6 |
| `freetoken/layers/base.py` | 99 | 13 |
| `freetoken/layers/embedding.py` | 149 | 9 |
| `freetoken/layers/gguf.py` | 128 | 7 |
| `freetoken/layers/linear.py` | 181 | 16 |
| `freetoken/layers/mhc.py` | 253 | 11 |
| `freetoken/layers/moe.py` | 510 | 20 |
| `freetoken/layers/norm.py` | 205 | 26 |
| `freetoken/layers/rotary.py` | 355 | 18 |
| `freetoken/layers/quantization/__init__.py` | 52 | 0 |
| `freetoken/layers/quantization/method.py` | 89 | 7 |
| `freetoken/layers/quantization/names.py` | 98 | 9 |
| `freetoken/layers/quantization/quant_backend.py` | 80 | 8 |
| `freetoken/layers/quantization/registry.py` | 59 | 8 |
| `freetoken/layers/quantization/scheme.py` | 107 | 16 |
| `freetoken/layers/quantization/configs/__init__.py` | 38 | 3 |
| `freetoken/layers/quantization/configs/base.py` | 130 | 16 |
| `freetoken/layers/quantization/configs/compressed_tensors.py` | 73 | 4 |
| `freetoken/layers/quantization/configs/fp8.py` | 78 | 5 |
| `freetoken/layers/quantization/configs/modelopt.py` | 80 | 6 |
| `freetoken/layers/quantization/configs/mxfp4.py` | 29 | 3 |
| `freetoken/layers/quantization/linear/__init__.py` | 13 | 0 |
| `freetoken/layers/quantization/linear/base.py` | 64 | 12 |
| `freetoken/layers/quantization/linear/fp8_block.py` | 70 | 10 |
| `freetoken/layers/quantization/linear/fp8_tensor.py` | 84 | 10 |
| `freetoken/layers/quantization/linear/mxfp8.py` | 48 | 6 |
| `freetoken/layers/quantization/linear/nvfp4.py` | 124 | 11 |
| `freetoken/layers/quantization/linear/unquantized.py` | 33 | 4 |
| `freetoken/layers/quantization/moe/__init__.py` | 13 | 0 |
| `freetoken/layers/quantization/moe/base.py` | 210 | 29 |
| `freetoken/layers/quantization/moe/fp8_block.py` | 68 | 7 |
| `freetoken/layers/quantization/moe/mxfp4.py` | 113 | 12 |
| `freetoken/layers/quantization/moe/mxfp8.py` | 21 | 4 |
| `freetoken/layers/quantization/moe/nvfp4.py` | 569 | 28 |
| `freetoken/layers/quantization/moe/unquantized.py` | 41 | 6 |

---

## `freetoken/layers/__init__.py`  (62 行)

_（无类/函数定义，模块级代码）_

## `freetoken/layers/activation.py`  (94 行)

| 行 | 符号 | 说明 |
|---:|---|---|
| 9 | `silu_and_mul()` | silu_and_mul(x |
| 20 | `gelu_and_mul()` | gelu_and_mul(x |
| 31 | `gelu_tanh_and_mul()` | tanh-approximate GELU gate (`gelu_pytorch_tanh`) followed by elementwise mul. |
| 43 | `swigluoai_and_mul()` | SwiGLU-OAI (gpt-oss / MiniMax-M3 ``swigluoai``) over UNINTERLEAVED halves |
| 59 | `swiglu_clamp_and_mul()` | GLM-5.3 clamped SwiGLU over UNINTERLEAVED halves: ``clamp(gate, max=limit) * sigmoid(alpha * gate) * clamp(up, +-limit)``. |
| 71 | `gated_act_and_mul()` | The epilogue between the two expert GEMMs: ``activation`` over uninterleaved [gate; up] halves into ``out``. |

## `freetoken/layers/base.py`  (99 行)

| 行 | 符号 | 说明 |
|---:|---|---|
| 11 | `_concat_prefix()` | _concat_prefix(prefix |
| 15 | **class `BaseOP`** | class BaseOP |
| 17 | **class `forward`** | forward(self, *args |
| 19 | **class `state_dict`** | state_dict(self, *, prefix |
| 32 | **class `load_state_dict`** | load_state_dict( self, state_dict |
| 56 | **class `StateLessOP`** | class StateLessOP(BaseOP) |
| 57 | **class `__init__`** | __init__(self) |
| 60 | **class `load_state_dict`** | load_state_dict( self, state_dict |
| 70 | **class `state_dict`** | state_dict(self, *, prefix |
| 77 | **class `OPList`** | class OPList(BaseOP, Generic[T]) |
| 78 | **class `__init__`** | __init__(self, ops |
| 82 | **class `state_dict`** | state_dict(self, *, prefix |
| 88 | **class `load_state_dict`** | load_state_dict( self, state_dict |

## `freetoken/layers/embedding.py`  (149 行)

| 行 | 符号 | 说明 |
|---:|---|---|
| 15 | **class `VocabParallelEmbedding`** | class VocabParallelEmbedding(BaseOP) |
| 16 | **class `__init__`** | __init__( self, num_embeddings |
| 41 | **class `forward`** | forward(self, x |
| 61 | **class `ParallelLMHead`** | The head is a linear layer over the vocab shard: its weights come from ``quant_method`` |
| 67 | **class `__init__`** | __init__( self, num_embeddings |
| 92 | **class `finalize`** | finalize(self) -> None |
| 96 | **class `load_state_dict`** | load_state_dict( self, state_dict |
| 114 | **class `state_dict`** | state_dict( self, *, prefix |
| 125 | **class `forward`** | forward(self, x |

## `freetoken/layers/gguf.py`  (128 行)

| 行 | 符号 | 说明 |
|---:|---|---|
| 43 | `fused_mul_mat_gguf()` | y = x @ dequant(qweight).T, dispatched by batch size and quant type. |
| 68 | **class `GGUFLinear`** | Linear whose weight is a native GGUF block-quantized ``[out, row_bytes]`` tensor. |
| 71 | **class `__init__`** | __init__( self, in_features |
| 84 | **class `forward`** | forward(self, x |
| 91 | **class `GGUFEmbedding`** | Vocab embedding stored as a native GGUF block-quantized table. |
| 98 | **class `__init__`** | __init__( self, num_embeddings |
| 114 | **class `forward`** | forward(self, x |

## `freetoken/layers/linear.py`  (181 行)

| 行 | 符号 | 说明 |
|---:|---|---|
| 13 | **class `_LinearTPImpl`** | Real implementation of a linear layer with tensor parallelism. |
| 21 | **class `__init__`** | __init__( self, full_isize |
| 47 | **class `finalize`** | finalize(self) -> None |
| 50 | **class `forward`** | forward(self, x |
| 54 | **class `LinearReplicated`** | Linear layer where weights are replicated (not sharded) across all TP ranks. |
| 60 | **class `__init__`** | __init__( self, input_size |
| 80 | **class `LinearColParallelMerged`** | class LinearColParallelMerged(_LinearTPImpl) |
| 81 | **class `__init__`** | __init__( self, input_size |
| 101 | **class `LinearQKVMerged`** | class LinearQKVMerged(_LinearTPImpl) |
| 102 | **class `__init__`** | __init__( self, hidden_size |
| 128 | **class `LinearOProj`** | class LinearOProj(_LinearTPImpl) |
| 129 | **class `__init__`** | __init__( self, input_size |
| 150 | **class `forward`** | forward(self, x |
| 157 | **class `LinearRowParallel`** | class LinearRowParallel(_LinearTPImpl) |
| 158 | **class `__init__`** | __init__( self, input_size |
| 177 | **class `forward`** | forward(self, x |

## `freetoken/layers/mhc.py`  (253 行)

| 行 | 符号 | 说明 |
|---:|---|---|
| 39 | `hc_expand()` | [T, hidden] -> [T, n, hidden] by replication (model entry). |
| 44 | `hc_contract()` | [T, n, hidden] -> [T, hidden] by averaging (model exit). |
| 49 | `mhc_gates()` | The coefficient predictor H(X): returns (pre_mix [T, n] fp32, post_mix [T, n, 1] |
| 84 | `mhc_mix_input()` | The sublayer input ``sum_i pre_i * res_i``: [T, n, hidden] x [T, n] -> [T, hidden]. |
| 89 | `mhc_pre()` | Multi-pass: gates from ``residual``, pre applied to it right away. Returns |
| 105 | `mhc_pre_single_pass()` | Single-pass: gates from ``residual`` for THIS sublayer's post/comb and the NEXT |
| 123 | `mhc_post()` | out_j = sum_i comb_ij * residual_i + post_j * x; returns [T, n, hidden]. |
| 137 | `mhc_fused_post_pre_torch()` | Decomposed reference: hc_post then hc_pre; the fused triton kernel |
| 159 | `mhc_fused_post_pre()` | Apply the previous sublayer's hc_post, then this sublayer's hc_pre on the |
| 188 | `mhc_fused_post_pre_single_pass_torch()` | Decomposed single-pass reference: the previous sublayer's hc_post (skipped when |
| 213 | `mhc_fused_post_pre_single_pass()` | Single-pass sublayer boundary (DeepSeek-V4.1): fused triton on CUDA, torch elsewhere. |

## `freetoken/layers/moe.py`  (510 行)

| 行 | 符号 | 说明 |
|---:|---|---|
| 29 | **class `MoELayer`** | Resident routed experts. |
| 39 | **class `__init__`** | __init__( self, num_experts |
| 91 | **class `finalize`** | finalize(self) -> None |
| 95 | **class `_maybe_all_reduce`** | _maybe_all_reduce(self, hidden_states |
| 100 | **class `_resident_gemm`** | _resident_gemm( self, hidden_states |
| 112 | **class `routed_forward`** | Expert compute for an externally computed routing decision (``TopK``). |
| 134 | **class `forward`** | forward( self, hidden_states |
| 148 | **class `OffloadMoELayer`** | class OffloadMoELayer(MoELayer) |
| 149 | **class `__init__`** | __init__( self, layer_id |
| 192 | **class `forward`** | forward( self, hidden_states |
| 204 | **class `routed_forward`** | Expert compute for an externally computed routing decision (``TopK``). |
| 227 | **class `decode_forward`** | decode_forward( self, hidden_states |
| 240 | **class `prefill_forward`** | prefill_forward( self, hidden_states |
| 261 | **class `_decode_routed`** | On-demand load: ``ensure_experts`` rewrites ``topk_ids`` into cache slot |
| 298 | **class `_decode_hybrid`** | Hybrid decode: GPU computes cache hits + <=K freshly-fetched experts, the CPU |
| 352 | **class `_prefill_routed`** | Prefill movement: stream whole layers -- double-buffered behind the |
| 391 | **class `_wait_prefill_overlap`** | Double-buffer choreography for this layer's overlap prefill: kick off the |
| 408 | **class `_expert_gemm`** | _expert_gemm( self, cache |
| 443 | `iter_moe_layers()` | iter_moe_layers(model) -> Iterator[MoELayer] |
| 456 | `make_moe_layer()` | Build the experts layer for ``config.moe_strategy`` -- the one construction |

## `freetoken/layers/norm.py`  (205 行)

| 行 | 符号 | 说明 |
|---:|---|---|
| 9 | **class `RMSNorm`** | class RMSNorm(BaseOP) |
| 10 | **class `__init__`** | __init__(self, size |
| 22 | **class `forward`** | forward(self, x |
| 25 | **class `forward_inplace`** | forward_inplace(self, x |
| 29 | **class `GemmaRMSNorm`** | Gemma4-style RMSNorm backed directly by sgl_kernel. |
| 36 | **class `__init__`** | __init__(self, size |
| 54 | **class `_kernel_weight`** | _kernel_weight(self, x |
| 61 | **class `forward`** | forward(self, x |
| 72 | **class `forward_add_residual`** | forward_add_residual( self, x |
| 79 | **class `GemmaPlusOneRMSNorm`** | (1 + w)-scaled RMSNorm (Gemma semantics: the checkpoint stores ``scale - 1`` |
| 94 | **class `__init__`** | __init__(self, size |
| 107 | **class `_flat`** | _flat(self, x |
| 113 | **class `forward`** | forward(self, x |
| 116 | **class `forward_inplace`** | forward_inplace(self, x |
| 121 | **class `GemmaPlusOneRMSNormFused`** | (1 + w)-scaled RMSNorm with the fused-add-residual API of ``RMSNormFused`` |
| 126 | **class `__init__`** | __init__(self, size |
| 142 | **class `forward`** | forward( self, x |
| 151 | **class `RMSNormFused`** | class RMSNormFused(BaseOP) |
| 152 | **class `__init__`** | __init__(self, size |
| 165 | **class `forward`** | forward( self, x |
| 177 | **class `GatedRMSNorm`** | ``rmsnorm(x) * activation(z)`` in one fla kernel: the GDN / KDA output norm. |
| 180 | **class `__init__`** | __init__(self, size |
| 187 | **class `forward`** | forward(self, x |
| 196 | **class `LayerNorm`** | LayerNorm with bias on torch's fused kernel; the decoders use the RMSNorm family. |
| 199 | **class `__init__`** | __init__(self, size |
| 204 | **class `forward`** | forward(self, x |

## `freetoken/layers/rotary.py`  (355 行)

| 行 | 符号 | 说明 |
|---:|---|---|
| 12 | **class `RotaryEmbedding`** | class RotaryEmbedding(StateLessOP) |
| 13 | **class `__init__`** | __init__( self, head_size |
| 42 | **class `forward`** | forward( self, positions |
| 59 | `mrope_cos_sin_rows()` | Per-token [cos \| sin] rows for [3, n] positions: frequency slot i reads the cache row of axis section_table[i]. |
| 69 | `_mrope_torch()` | Pure-torch mrope for triton-less installs; same in-place NeoX contract as the kernel. |
| 96 | `build_section_table()` | Axis id (0=t, 1=h, 2=w) per frequency slot for a [t, h, w] section split. |
| 126 | **class `MRotaryEmbedding`** | 3-axis (t/h/w) rope over the parent's cos_sin_cache; section_table picks the axis row per frequency slot. Consumes positions [3, n]. |
| 129 | **class `__init__`** | __init__( self, *args, mrope_section |
| 145 | **class `forward`** | forward( self, positions |
| 168 | `rope_table_positions()` | Rows the cos/sin table needs: YaRN serves ``original_max_position_embeddings * factor``, |
| 177 | `_rope_frequencies()` | ``inv_freq`` over the table's half width, and the factor cos and sin are both scaled by. |
| 219 | &nbsp;&nbsp;&nbsp;↳ `get_mscale()` | get_mscale(scale |
| 237 | &nbsp;&nbsp;&nbsp;↳ `_find_correction_dim()` | _find_correction_dim(num_rotations |
| 266 | `_cos_sin_table()` | _cos_sin_table( head_size |
| 281 | `rope_cos_sin_table()` | The ``[positions, width]`` cos\|sin table, one per distinct rope: only proportional rope |
| 308 | `set_rope_device()` | set_rope_device(device |
| 314 | `get_rope()` | get_rope( head_dim |
| 324 | &nbsp;&nbsp;&nbsp;↳ `build()` | build() -> RotaryEmbedding |

## `freetoken/layers/quantization/__init__.py`  (52 行)

_（无类/函数定义，模块级代码）_

## `freetoken/layers/quantization/method.py`  (89 行)

| 行 | 符号 | 说明 |
|---:|---|---|
| 16 | **class `KernelSelectionError`** | class KernelSelectionError(RuntimeError) |
| 20 | `select_kernel()` | Pick one kernel from an ordered table: the requested name if usable, else the first usable one that is worth it. |
| 50 | **class `QuantMethod`** | class QuantMethod(ABC) |
| 55 | **class `__init__`** | __init__(self, cfg |
| 61 | **class `scheme`** | scheme(self) |
| 64 | `finalize_quant()` | Call ``quant_method.finalize(layer)`` on every layer under ``root``; returns the count. |
| 71 | &nbsp;&nbsp;&nbsp;↳ `walk()` | walk(op |

## `freetoken/layers/quantization/names.py`  (98 行)

| 行 | 符号 | 说明 |
|---:|---|---|
| 13 | `ancestors()` | ancestors(name |
| 22 | `name_set()` | Membership of a module or any parent in a list of literal names and glob patterns. |
| 28 | &nbsp;&nbsp;&nbsp;↳ `hit()` | hit(name |
| 34 | `substr_set()` | transformers' fp8 integration skips a module when ``entry + "."`` occurs in ``name + "."``. |
| 40 | `ct_set()` | compressed-tensors ``targets`` / ``ignore``: module names, ``re:`` regexes, or the class name Linear. |
| 58 | `is_routed_expert()` | is_routed_expert(name |
| 63 | **class `NameMap`** | FreeToken attribute path -> checkpoint module name(s), applied to every scheme_for query. |
| 75 | **class `to_checkpoint`** | to_checkpoint(self, prefix |
| 89 | `_replace_segment()` | _replace_segment(prefix |

## `freetoken/layers/quantization/quant_backend.py`  (80 行)

| 行 | 符号 | 说明 |
|---:|---|---|
| 13 | `_kernel_names()` | _kernel_names(layer_kind |
| 22 | `_parse_table()` | _parse_table(key |
| 30 | `_table_text()` | _table_text(table |
| 36 | **class `QuantBackend`** | class QuantBackend |
| 40 | **class `parse`** | parse(cls, text |
| 58 | **class `select`** | The kernel name for one table: its own entry, else the layer-wide entry when the table lists that kernel, else ``"auto"``. |
| 73 | `set_quant_backend()` | Install the kernel requests every layer built afterwards consults; the engine sets them once before create_model. |
| 79 | `get_quant_backend()` | get_quant_backend() -> QuantBackend |

## `freetoken/layers/quantization/registry.py`  (59 行)

| 行 | 符号 | 说明 |
|---:|---|---|
| 11 | **class `LayerKind`** | class LayerKind(Enum) |
| 15 | **class `__str__`** | __str__(self) -> str |
| 23 | `register_dialect()` | register_dialect(cls |
| 31 | `dialects()` | dialects() -> tuple[type, ...] |
| 35 | `register_method()` | register_method(kind |
| 39 | &nbsp;&nbsp;&nbsp;↳ `deco()` | deco(cls |
| 51 | `method_class()` | method_class(kind |
| 58 | `methods_for()` | methods_for(layer_kind |

## `freetoken/layers/quantization/scheme.py`  (107 行)

| 行 | 符号 | 说明 |
|---:|---|---|
| 14 | **class `QuantKind`** | The quantization kinds; one Method per (kind, layer kind). |
| 24 | **class `__str__`** | __str__(self) -> str |
| 35 | **class `WeightDesc`** | class WeightDesc |
| 44 | **class `QuantScheme`** | ``roles`` are the canonical tensor names the kind's Method declares; which checkpoint tensors feed them is the dialect Config's business. |
| 51 | **class `__init__`** | __init__(self, kind |
| 58 | **class `_key`** | _key(self) |
| 61 | **class `__eq__`** | __eq__(self, other |
| 64 | **class `__hash__`** | __hash__(self) -> int |
| 67 | **class `has`** | has(self, role |
| 70 | **class `__repr__`** | __repr__(self) -> str |
| 79 | `fp8_tensor_scheme()` | fp8_tensor_scheme(scale |
| 84 | `fp8_block_scheme()` | Square ``block x block`` e4m3 weight blocks, one scale each; ``block`` is the checkpoint's ``weight_block_size``. |
| 91 | `fp8_block_size()` | The square block edge of an FP8_BLOCK scheme (128 for DeepSeek-V3 style exports, 32 for DeepSeek-V4.1). |
| 97 | `mxfp8_scheme()` | mxfp8_scheme() -> QuantScheme |
| 101 | `nvfp4_scheme()` | nvfp4_scheme(*, input_scale |
| 106 | `mxfp4_scheme()` | mxfp4_scheme() -> QuantScheme |

## `freetoken/layers/quantization/configs/__init__.py`  (38 行)

| 行 | 符号 | 说明 |
|---:|---|---|
| 13 | `quant_method_for()` | The layer's Method from its config; a layer built without one is plain bf16. |
| 22 | `set_quant_config()` | Install the checkpoint's QuantConfig for the weight readers, which only get the model path; EngineConfig sets it once. |
| 28 | `get_quant_config()` | get_quant_config() -> QuantConfig \| None |

## `freetoken/layers/quantization/configs/base.py`  (130 行)

| 行 | 符号 | 说明 |
|---:|---|---|
| 19 | `cfg_get()` | cfg_get(cfg |
| 27 | `quantization_config_of()` | quantization_config_of(hf_config |
| 41 | **class `Stored`** | One checkpoint tensor behind a role: its suffix, and whether it holds the quant-side scale whose reciprocal the layer wants. |
| 48 | **class `QuantConfig`** | class QuantConfig(ABC) |
| 53 | **class `__init__`** | __init__(self, name_map |
| 60 | **class `claims`** | Whether this dialect reads the given ``quantization_config``. |
| 65 | **class `scheme_for_name`** | Scheme of one module by its checkpoint name; None means unquantized. |
| 68 | **class `scheme_for`** | scheme_for(self, prefix |
| 79 | **class `stored_tensors`** | role -> checkpoint tensor for every role the dialect stores for ``kind``. |
| 83 | **class `storage`** | role -> checkpoint tensor for one scheme's tensors. |
| 87 | **class `get_quant_method`** | get_quant_method(self, layer |
| 95 | **class `layer_config`** | The layer's Linear / MoE config; a dialect overrides it to add what only its checkpoint knows. |
| 100 | **class `from_hf`** | ``unquantized`` lists the modules the family keeps bf16 when the checkpoint's config does not (DeepSeek-V4's compressors). |
| 119 | **class `NoQuantConfig`** | No ``quantization_config``: every module is bf16. |
| 126 | **class `claims`** | claims(cls, q |
| 129 | **class `scheme_for_name`** | scheme_for_name(self, name |

## `freetoken/layers/quantization/configs/compressed_tensors.py`  (73 行)

| 行 | 符号 | 说明 |
|---:|---|---|
| 13 | **class `CompressedTensorsConfig`** | llm-compressor exports: ordered ``config_groups`` with ``targets`` (class names, names or ``re:``) minus ``ignore``. |
| 36 | **class `__init__`** | __init__(self, q |
| 44 | **class `scheme_for_name`** | scheme_for_name(self, name |
| 53 | **class `_scheme_of`** | _scheme_of(cls, spec |

## `freetoken/layers/quantization/configs/fp8.py`  (78 行)

| 行 | 符号 | 说明 |
|---:|---|---|
| 16 | **class `Fp8BlockConfig`** | HF ``quant_method: fp8`` (DeepSeek-V3 style 128x128 block scales) plus the DeepSeek-V4 e8m0 / fp4-expert |
| 37 | **class `__init__`** | __init__(self, q |
| 58 | **class `storage`** | storage(self, scheme |
| 64 | **class `scheme_for_name`** | scheme_for_name(self, name |
| 73 | **class `layer_config`** | layer_config(self, layer |

## `freetoken/layers/quantization/configs/modelopt.py`  (80 行)

| 行 | 符号 | 说明 |
|---:|---|---|
| 13 | **class `ModelOptConfig`** | NVIDIA ModelOpt exports: one ``quant_algo`` for every Linear minus ``ignore``, or |
| 35 | **class `claims`** | claims(cls, q |
| 39 | **class `__init__`** | __init__(self, q |
| 56 | **class `scheme_for_name`** | scheme_for_name(self, name |
| 62 | **class `_module_algo`** | _module_algo(self, name |
| 71 | **class `_scheme_of`** | _scheme_of(self, algo |

## `freetoken/layers/quantization/configs/mxfp4.py`  (29 行)

| 行 | 符号 | 说明 |
|---:|---|---|
| 13 | **class `Mxfp4Config`** | gpt-oss: the stacked ``experts`` containers are MXFP4 with biases, everything else bf16. |
| 22 | **class `__init__`** | __init__(self, q |
| 26 | **class `scheme_for_name`** | scheme_for_name(self, name |

## `freetoken/layers/quantization/linear/__init__.py`  (13 行)

_（无类/函数定义，模块级代码）_

## `freetoken/layers/quantization/linear/base.py`  (64 行)

| 行 | 符号 | 说明 |
|---:|---|---|
| 16 | **class `LinearConfig`** | What a linear kernel may look at to accept or decline a layer: the TP-local shape and the weight scheme. |
| 25 | **class `__post_init__`** | __post_init__(self) -> None |
| 32 | **class `from_layer`** | from_layer(cls, layer |
| 36 | **class `LinearKernel`** | One backend for one linear kind: says whether it can run a layer, then finalizes and applies it. |
| 41 | **class `unusable_reason`** | unusable_reason(self, cfg |
| 44 | **class `worth_it`** | worth_it(self, cfg |
| 47 | **class `finalize`** | finalize(self, layer |
| 51 | **class `apply`** | apply(self, layer |
| 54 | **class `LinearMethod`** | Declares a linear layer's weights for its kind; finalize and apply go to the kernel. |
| 58 | **class `create_weights`** | create_weights(self, layer |
| 60 | **class `finalize`** | finalize(self, layer |
| 63 | **class `apply`** | apply(self, layer |

## `freetoken/layers/quantization/linear/fp8_block.py`  (70 行)

| 行 | 符号 | 说明 |
|---:|---|---|
| 17 | `_e8m0()` | _e8m0(cfg |
| 21 | `layer_block_size()` | The block edge a finalized layer was declared with: ``K / (scale columns)``, exact by construction. |
| 26 | **class `Dsv4Fp8BlockLinearKernel`** | DeepSeek-V4's reference path: activations quantized to fp8 with power-of-two block scales, e8m0 weight scales read as codes. |
| 31 | **class `unusable_reason`** | unusable_reason(self, cfg |
| 34 | **class `apply`** | apply(self, layer |
| 40 | **class `TritonFp8BlockLinearKernel`** | W8A16 GEMV at M=1, dynamic 1x128 W8A8 GEMM above. |
| 45 | **class `unusable_reason`** | unusable_reason(self, cfg |
| 52 | **class `apply`** | apply(self, layer |
| 59 | **class `Fp8BlockLinearMethod`** | class Fp8BlockLinearMethod(LinearMethod) |
| 62 | **class `create_weights`** | create_weights(self, layer |

## `freetoken/layers/quantization/linear/fp8_tensor.py`  (84 行)

| 行 | 符号 | 说明 |
|---:|---|---|
| 18 | **class `TorchFp8TensorLinearKernel`** | Static W8A8 through torch._scaled_mm; needs the checkpoint's input_scale and fp8 tensor cores. |
| 23 | **class `unusable_reason`** | unusable_reason(self, cfg |
| 35 | **class `finalize`** | finalize(self, layer |
| 46 | **class `apply`** | apply(self, layer |
| 55 | **class `TritonFp8TensorLinearKernel`** | W8A16: split-K GEMV at M=1, GEMM above. |
| 60 | **class `apply`** | apply(self, layer |
| 66 | **class `EmulationFp8TensorLinearKernel`** | class EmulationFp8TensorLinearKernel(LinearKernel) |
| 69 | **class `apply`** | apply(self, layer |
| 77 | **class `Fp8TensorLinearMethod`** | class Fp8TensorLinearMethod(LinearMethod) |
| 80 | **class `create_weights`** | create_weights(self, layer |

## `freetoken/layers/quantization/linear/mxfp8.py`  (48 行)

| 行 | 符号 | 说明 |
|---:|---|---|
| 16 | **class `TritonMxfp8LinearKernel`** | W8A16: split-K GEMV up to M=256, bf16 dequant + cuBLAS above. |
| 21 | **class `apply`** | apply(self, layer |
| 27 | **class `EmulationMxfp8LinearKernel`** | class EmulationMxfp8LinearKernel(LinearKernel) |
| 30 | **class `apply`** | apply(self, layer |
| 40 | **class `Mxfp8LinearMethod`** | class Mxfp8LinearMethod(LinearMethod) |
| 43 | **class `create_weights`** | create_weights(self, layer |

## `freetoken/layers/quantization/linear/nvfp4.py`  (124 行)

| 行 | 符号 | 说明 |
|---:|---|---|
| 19 | **class `TritonNvfp4LinearKernel`** | W4A16 on the K-major resident layout. |
| 24 | **class `finalize`** | finalize(self, layer |
| 29 | **class `apply`** | apply(self, layer |
| 35 | **class `MarlinNvfp4LinearKernel`** | vLLM's fused FP4 Marlin GEMM (W4A16); needs one global scale for the whole weight. |
| 40 | **class `unusable_reason`** | unusable_reason(self, cfg |
| 57 | **class `finalize`** | finalize(self, layer |
| 79 | **class `apply`** | apply(self, layer |
| 94 | **class `EmulationNvfp4LinearKernel`** | Dequantize to bf16 with dequant_nvfp4, then a plain matmul; keeps the row-major layout. |
| 99 | **class `apply`** | apply(self, layer |
| 113 | **class `Nvfp4LinearMethod`** | class Nvfp4LinearMethod(LinearMethod) |
| 116 | **class `create_weights`** | create_weights(self, layer |

## `freetoken/layers/quantization/linear/unquantized.py`  (33 行)

| 行 | 符号 | 说明 |
|---:|---|---|
| 15 | **class `TorchLinearKernel`** | class TorchLinearKernel(LinearKernel) |
| 18 | **class `apply`** | apply(self, layer |
| 28 | **class `UnquantizedLinearMethod`** | class UnquantizedLinearMethod(LinearMethod) |
| 31 | **class `create_weights`** | create_weights(self, layer |

## `freetoken/layers/quantization/moe/__init__.py`  (13 行)

_（无类/函数定义，模块级代码）_

## `freetoken/layers/quantization/moe/base.py`  (210 行)

| 行 | 符号 | 说明 |
|---:|---|---|
| 16 | **class `MoEConfig`** | Everything an expert kernel may look at to accept or decline a layer and to size its banks. |
| 40 | **class `from_layer`** | from_layer(cls, layer |
| 62 | **class `local_intermediate`** | local_intermediate(self) -> int |
| 66 | **class `plain_silu`** | plain_silu(self) -> bool |
| 71 | **class `BankSpec`** | One bank of an expert layout, per expert (the E dim is prepended by the allocator). |
| 81 | **class `ExpertView`** | The weights an apply() call reads: layout roles -> [E, ...] (resident, materialized layer) or [S, ...] (slot cache). |
| 90 | `fused_piece()` | A gate_up-family piece, concatenating separate gate / up pieces along the row axis. |
| 98 | `global_rows()` | Per-expert global scale as one fp16 value per output row (a scalar per expert is broadcast). |
| 107 | `fused_global()` | fused_global(pieces |
| 113 | `limit_or_inf()` | limit_or_inf(layer |
| 118 | `gated_epilogue_reason()` | Why ``gated_act_and_mul`` cannot run ``cfg``'s activation quadruple, or None; the plain kinds have no alpha / limit and nothing takes beta or inter... |
| 133 | `is_resident()` | is_resident(layer |
| 137 | **class `MoEKernel`** | One backend for one expert kind: bank layout, packing from checkpoint pieces, and the fused forward. |
| 148 | **class `unusable_reason`** | unusable_reason(self, cfg |
| 151 | **class `worth_it`** | worth_it(self, cfg |
| 154 | **class `slot_limit`** | Most GPU cache slots the kernel can address for ``cfg``; None for no limit. |
| 158 | **class `_common_reject`** | _common_reject(self, cfg |
| 170 | **class `layout`** | layout(self, cfg |
| 173 | **class `pack`** | Write one batch of expert pieces into ``out`` rows; returns the GPU-resident per-expert values, if any. |
| 177 | **class `apply`** | apply(self, layer |
| 180 | **class `MoEMethod`** | Declares resident experts for its kind; layout, pack and apply go to the kernel. |
| 183 | **class `layout`** | layout(self) -> dict[str, BankSpec] |
| 186 | **class `pack`** | pack(self, pieces |
| 189 | **class `slot_limit`** | slot_limit(self) -> int \| None |
| 192 | **class `apply`** | apply(self, x |
| 196 | **class `cpu_format`** | cpu_format(self) -> str \| None |
| 199 | **class `create_weights`** | create_weights(self, layer |
| 204 | **class `finalize`** | finalize(self, layer |
| 207 | **class `resident_view`** | Layout roles -> the resident layer's tensors, for apply(). |

## `freetoken/layers/quantization/moe/fp8_block.py`  (68 行)

| 行 | 符号 | 说明 |
|---:|---|---|
| 14 | `_pad_scale()` | _pad_scale(rows |
| 20 | **class `TritonFp8BlockMoEKernel`** | class TritonFp8BlockMoEKernel(MoEKernel) |
| 23 | **class `unusable_reason`** | unusable_reason(self, cfg |
| 36 | **class `layout`** | layout(self, cfg |
| 45 | **class `pack`** | pack(self, pieces, cfg |
| 55 | **class `apply`** | apply(self, layer, x, topk_weights, topk_ids, view |
| 67 | **class `Fp8BlockMoEMethod`** | class Fp8BlockMoEMethod(MoEMethod) |

## `freetoken/layers/quantization/moe/mxfp4.py`  (113 行)

| 行 | 符号 | 说明 |
|---:|---|---|
| 14 | **class `TritonMxfp4MoEKernel`** | Standard OCP MXFP4 experts (DeepSeek-V4 / V4.1 ds_fp4): e2m1 pairs + e8m0 scales, no bias; |
| 22 | **class `unusable_reason`** | unusable_reason(self, cfg |
| 31 | **class `layout`** | layout(self, cfg |
| 40 | **class `pack`** | pack(self, pieces, cfg |
| 47 | **class `apply`** | apply(self, layer, x, topk_weights, topk_ids, view |
| 61 | **class `TritonGptossMxfp4MoEKernel`** | gpt-oss MXFP4: bias, alpha / limit swiglu, interleaved gate\|up rows, transposed split-K banks. |
| 67 | **class `unusable_reason`** | unusable_reason(self, cfg |
| 72 | **class `layout`** | layout(self, cfg |
| 83 | **class `pack`** | pack(self, pieces, cfg |
| 85 | &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;↳ `blocks_t()` | blocks_t(t) |
| 97 | **class `apply`** | apply(self, layer, x, topk_weights, topk_ids, view |
| 112 | **class `Mxfp4MoEMethod`** | class Mxfp4MoEMethod(MoEMethod) |

## `freetoken/layers/quantization/moe/mxfp8.py`  (21 行)

| 行 | 符号 | 说明 |
|---:|---|---|
| 11 | **class `Mxfp8MoEMethod`** | class Mxfp8MoEMethod(MoEMethod) |
| 14 | **class `__init__`** | __init__(self, *args, **kwargs) |
| 17 | **class `create_weights`** | create_weights(self, layer) -> None |
| 20 | **class `resident_view`** | resident_view(self, layer) -> ExpertView |

## `freetoken/layers/quantization/moe/nvfp4.py`  (569 行)

| 行 | 符号 | 说明 |
|---:|---|---|
| 37 | **class `TritonNvfp4MoEKernel`** | FreeToken's inline-dequant kernels over the native ModelOpt rows. |
| 43 | **class `unusable_reason`** | unusable_reason(self, cfg |
| 50 | **class `layout`** | layout(self, cfg |
| 61 | **class `pack`** | pack(self, pieces, cfg |
| 70 | **class `apply`** | apply(self, layer, x, topk_weights, topk_ids, view |
| 95 | `_marlin_symbols_ok()` | Probe the exact vLLM symbols the pack/forward paths below use. |
| 125 | `_marlin_pack_proj()` | Pack one expert projection (on GPU) into the Marlin layout. |
| 177 | `marlin_fused_experts()` | Marlin W4A16 fused MoE (two grouped GEMMs + activation + reduce). |
| 224 | **class `MarlinNvfp4MoEKernel`** | vLLM's fused Marlin MoE over pre-tiled banks; the globals fold into GPU-resident alphas. |
| 230 | **class `unusable_reason`** | unusable_reason(self, cfg |
| 240 | **class `worth_it`** | worth_it(self, cfg |
| 243 | **class `layout`** | layout(self, cfg |
| 254 | **class `pack`** | pack(self, pieces, cfg |
| 274 | **class `apply`** | apply(self, layer, x, topk_weights, topk_ids, view |
| 301 | `_b12x_symbols_ok()` | Probe the exact flashinfer symbols the pack/forward paths below use. |
| 330 | `_flashinfer_cuda_major()` | flashinfer's own toolkit-CUDA major; None if it can't be determined. Only a |
| 342 | `_b12x_unusable_reason()` | None if the flashinfer b12x decode/prefill kernel can actually run on this device, |
| 377 | `_b12x_swizzle_block_scales()` | Row-major per-16 fp8 block scales ``[n, N, K//16]`` -> flashinfer's expert-leading |
| 397 | `_b12x_small_workspace()` | _b12x_small_workspace(device |
| 409 | `b12x_fused_experts()` | SM12x W4A16 fused MoE over cache slots (same calling convention as |
| 488 | **class `B12xNvfp4MoEKernel`** | flashinfer's SM12x CuTe-DSL W4A16 MoE; the packed blocks reuse the native bank bytes. |
| 493 | **class `unusable_reason`** | unusable_reason(self, cfg |
| 504 | **class `worth_it`** | worth_it(self, cfg |
| 508 | **class `slot_limit`** | slot_limit(self, cfg |
| 513 | **class `layout`** | layout(self, cfg |
| 525 | **class `pack`** | pack(self, pieces, cfg |
| 561 | **class `apply`** | apply(self, layer, x, topk_weights, topk_ids, view |
| 568 | **class `Nvfp4MoEMethod`** | class Nvfp4MoEMethod(MoEMethod) |

## `freetoken/layers/quantization/moe/unquantized.py`  (41 行)

| 行 | 符号 | 说明 |
|---:|---|---|
| 12 | **class `FusedMoEKernel`** | class FusedMoEKernel(MoEKernel) |
| 16 | **class `unusable_reason`** | unusable_reason(self, cfg |
| 20 | **class `layout`** | layout(self, cfg |
| 26 | **class `pack`** | pack(self, pieces, cfg |
| 31 | **class `apply`** | apply(self, layer, x, topk_weights, topk_ids, view |
| 40 | **class `UnquantizedMoEMethod`** | class UnquantizedMoEMethod(MoEMethod) |

