# 符号索引：`kernel`

自动生成（`ast` 解析，不执行代码）。**91 个文件 / 23649 行 / 597 个符号**。

> 用途：确认某个模块里**到底有哪些**类/函数/方法及其行号，避免遗漏。
> 行为详解请看同目录下的手写模块文档。
> 重新生成：`python3 MODULE_DOCS/tools/gen_symbol_index.py`

| 文件 | 行数 | 符号数 |
|---|---:|---:|
| `freetoken/kernel/__init__.py` | 40 | 0 |
| `freetoken/kernel/__main__.py` | 52 | 1 |
| `freetoken/kernel/_toolchain.py` | 78 | 5 |
| `freetoken/kernel/aot.py` | 286 | 23 |
| `freetoken/kernel/aot_models.py` | 469 | 9 |
| `freetoken/kernel/backend.py` | 82 | 8 |
| `freetoken/kernel/batch_memcpy.py` | 69 | 4 |
| `freetoken/kernel/causal_conv1d.py` | 75 | 2 |
| `freetoken/kernel/fast_index_copy.py` | 209 | 11 |
| `freetoken/kernel/gguf.py` | 142 | 9 |
| `freetoken/kernel/index.py` | 54 | 3 |
| `freetoken/kernel/moe_impl.py` | 504 | 9 |
| `freetoken/kernel/pinned.py` | 68 | 7 |
| `freetoken/kernel/pynccl.py` | 79 | 9 |
| `freetoken/kernel/radix.py` | 20 | 2 |
| `freetoken/kernel/row_store.py` | 54 | 3 |
| `freetoken/kernel/store.py` | 42 | 2 |
| `freetoken/kernel/tensor.py` | 19 | 2 |
| `freetoken/kernel/utils.py` | 420 | 25 |
| `freetoken/kernel/fla/__init__.py` | 57 | 0 |
| `freetoken/kernel/fla/chunk.py` | 257 | 4 |
| `freetoken/kernel/fla/chunk_delta_h.py` | 357 | 3 |
| `freetoken/kernel/fla/chunk_fwd.py` | 416 | 2 |
| `freetoken/kernel/fla/chunk_o.py` | 174 | 3 |
| `freetoken/kernel/fla/cumsum.py` | 299 | 6 |
| `freetoken/kernel/fla/fused_recurrent.py` | 659 | 7 |
| `freetoken/kernel/fla/fused_sigmoid_gating_recurrent.py` | 372 | 2 |
| `freetoken/kernel/fla/index.py` | 35 | 3 |
| `freetoken/kernel/fla/kda.py` | 1396 | 21 |
| `freetoken/kernel/fla/kda_chunk_delta_h.py` | 394 | 3 |
| `freetoken/kernel/fla/l2norm.py` | 147 | 10 |
| `freetoken/kernel/fla/layernorm_gated.py` | 264 | 6 |
| `freetoken/kernel/fla/op.py` | 66 | 3 |
| `freetoken/kernel/fla/solve_tril.py` | 563 | 4 |
| `freetoken/kernel/fla/utils.py` | 356 | 23 |
| `freetoken/kernel/fla/wy_fast.py` | 156 | 2 |
| `freetoken/kernel/triton/activation.py` | 199 | 10 |
| `freetoken/kernel/triton/attention.py` | 977 | 11 |
| `freetoken/kernel/triton/autotune_cache.py` | 37 | 0 |
| `freetoken/kernel/triton/causal_conv1d_triton.py` | 681 | 8 |
| `freetoken/kernel/triton/decode_moe.py` | 75 | 1 |
| `freetoken/kernel/triton/df11.py` | 318 | 12 |
| `freetoken/kernel/triton/df11_decode.py` | 171 | 4 |
| `freetoken/kernel/triton/e4m3_compat.py` | 142 | 9 |
| `freetoken/kernel/triton/fp8_block_linear.py` | 279 | 9 |
| `freetoken/kernel/triton/fp8_blockscale_moe.py` | 262 | 6 |
| `freetoken/kernel/triton/fp8_pertensor_linear.py` | 356 | 13 |
| `freetoken/kernel/triton/fused_moe.py` | 302 | 7 |
| `freetoken/kernel/triton/gemma4_fused.py` | 191 | 4 |
| `freetoken/kernel/triton/glm_dsa_sparse.py` | 431 | 7 |
| `freetoken/kernel/triton/hc.py` | 395 | 11 |
| `freetoken/kernel/triton/kpool_compress.py` | 146 | 2 |
| `freetoken/kernel/triton/mhc.py` | 347 | 9 |
| `freetoken/kernel/triton/minimax_m3_sparse.py` | 1118 | 16 |
| `freetoken/kernel/triton/moe_align.py` | 328 | 7 |
| `freetoken/kernel/triton/moe_router.py` | 160 | 3 |
| `freetoken/kernel/triton/moe_shared_gate.py` | 137 | 6 |
| `freetoken/kernel/triton/mxfp4_moe.py` | 332 | 8 |
| `freetoken/kernel/triton/mxfp8_linear.py` | 252 | 6 |
| `freetoken/kernel/triton/norm.py` | 192 | 10 |
| `freetoken/kernel/triton/nvfp4_dequant.py` | 133 | 3 |
| `freetoken/kernel/triton/nvfp4_fused_moe.py` | 333 | 4 |
| `freetoken/kernel/triton/nvfp4_linear.py` | 818 | 22 |
| `freetoken/kernel/triton/ple.py` | 95 | 2 |
| `freetoken/kernel/triton/rope.py` | 270 | 5 |
| `freetoken/kernel/triton/sampling.py` | 650 | 33 |
| `freetoken/kernel/triton/dsv4/bf16_linear.py` | 70 | 2 |
| `freetoken/kernel/triton/dsv4/compress.py` | 67 | 2 |
| `freetoken/kernel/triton/dsv4/fp8_linear.py` | 519 | 17 |
| `freetoken/kernel/triton/dsv4/fused_moe.py` | 326 | 7 |
| `freetoken/kernel/triton/dsv4/hadamard.py` | 44 | 2 |
| `freetoken/kernel/triton/dsv4/hc.py` | 94 | 4 |
| `freetoken/kernel/triton/dsv4/indexer.py` | 226 | 4 |
| `freetoken/kernel/triton/dsv4/norm.py` | 55 | 2 |
| `freetoken/kernel/triton/dsv4/rope.py` | 87 | 2 |
| `freetoken/kernel/triton/dsv4/router.py` | 83 | 2 |
| `freetoken/kernel/triton/dsv4/sinkhorn.py` | 77 | 2 |
| `freetoken/kernel/triton/dsv4/sparse_attn.py` | 393 | 6 |
| `freetoken/kernel/triton/dsv4/swiglu.py` | 46 | 2 |
| `freetoken/kernel/triton/dsv41/__init__.py` | 9 | 0 |
| `freetoken/kernel/triton/dsv41/indexer.py` | 176 | 3 |
| `freetoken/kernel/triton/dsv41/pack.py` | 183 | 7 |
| `freetoken/kernel/triton/dsv41/row_format.py` | 92 | 3 |
| `freetoken/kernel/triton/dsv41/sparse_attn.py` | 323 | 6 |
| `freetoken/kernel/triton/dsv41/topk.py` | 265 | 9 |
| `freetoken/kernel/triton/qsa/__init__.py` | 18 | 0 |
| `freetoken/kernel/triton/qsa/attend.py` | 360 | 3 |
| `freetoken/kernel/triton/qsa/compress.py` | 331 | 6 |
| `freetoken/kernel/triton/qsa/expand.py` | 131 | 2 |
| `freetoken/kernel/triton/qsa/score.py` | 191 | 2 |
| `freetoken/kernel/triton/qsa/topk.py` | 626 | 18 |

---

## `freetoken/kernel/__init__.py`  (40 行)

_（无类/函数定义，模块级代码）_

## `freetoken/kernel/__main__.py`  (52 行)

| 行 | 符号 | 说明 |
|---:|---|---|
| 4 | `generate_clangd()` | generate_clangd() |

## `freetoken/kernel/_toolchain.py`  (78 行)

| 行 | 符号 | 说明 |
|---:|---|---|
| 19 | `_is_rocm()` | _is_rocm() -> bool |
| 24 | `_nvcc_path()` | _nvcc_path() -> str \| None |
| 32 | `nvcc_release()` | nvcc_release(nvcc |
| 43 | `torch_cuda_major()` | torch_cuda_major() -> int \| None |
| 51 | `check_nvcc_matches_torch()` | Refuse to nvcc-compile kernels across CUDA majors. |

## `freetoken/kernel/aot.py`  (286 行)

| 行 | 符号 | 说明 |
|---:|---|---|
| 38 | **class `KernelSpec`** | class KernelSpec |
| 43 | `_store_spec()` | _store_spec(element_size |
| 48 | &nbsp;&nbsp;&nbsp;↳ `build()` | build(build_directory |
| 60 | `_index_spec()` | _index_spec(element_size |
| 65 | &nbsp;&nbsp;&nbsp;↳ `build()` | build(build_directory |
| 78 | `_fast_index_copy_spec()` | _fast_index_copy_spec(feature_size |
| 84 | &nbsp;&nbsp;&nbsp;↳ `build()` | build(build_directory |
| 100 | `_fast_index_copy_update_flag_spec()` | _fast_index_copy_update_flag_spec() -> KernelSpec |
| 101 | &nbsp;&nbsp;&nbsp;↳ `build()` | build(build_directory |
| 112 | `_fast_index_copy_multi_spec()` | _fast_index_copy_multi_spec(num_threads |
| 115 | &nbsp;&nbsp;&nbsp;↳ `build()` | build(build_directory |
| 127 | `_batch_memcpy_spec()` | _batch_memcpy_spec() -> KernelSpec |
| 128 | &nbsp;&nbsp;&nbsp;↳ `build()` | build(build_directory |
| 139 | `_radix_spec()` | _radix_spec() -> KernelSpec |
| 140 | &nbsp;&nbsp;&nbsp;↳ `build()` | build(build_directory |
| 146 | `default_kernel_specs()` | default_kernel_specs() -> tuple[KernelSpec, ...] |
| 170 | `_force_source_build()` | _force_source_build() -> Iterator[None] |
| 188 | `_resolve_specs()` | _resolve_specs(specs |
| 206 | `_built_shared_object()` | _built_shared_object(build_dir |
| 217 | `_copy_kernel()` | _copy_kernel(build_dir |
| 231 | `_default_jobs()` | _default_jobs(num_specs |
| 238 | `compile_and_package_kernels()` | compile_and_package_kernels( *, out_dir |
| 258 | &nbsp;&nbsp;&nbsp;↳ `_build_one()` | _build_one(spec |

## `freetoken/kernel/aot_models.py`  (469 行)

| 行 | 符号 | 说明 |
|---:|---|---|
| 41 | **class `AotModel`** | One supported checkpoint family and the config fields shapes need. |
| 66 | `fp8_block_scale_pad()` | Trailing scale-bank dim padded so per-expert row bytes are 16B-aligned (fused copy). |
| 73 | `expert_bank_row_bytes()` | Per-expert row bytes for each offload bank a format registers. |
| 409 | `store_element_sizes()` | store_element_sizes(model |
| 413 | `index_variants()` | index_variants(model |
| 420 | `fast_index_copy_feature_sizes()` | fast_index_copy_feature_sizes(model |
| 431 | `aggregate_store_element_sizes()` | aggregate_store_element_sizes() -> tuple[int, ...] |
| 438 | `aggregate_index_variants()` | aggregate_index_variants() -> tuple[tuple[int, int], ...] |
| 450 | `aggregate_fast_index_copy_feature_sizes()` | aggregate_fast_index_copy_feature_sizes() -> tuple[int, ...] |

## `freetoken/kernel/backend.py`  (82 行)

| 行 | 符号 | 说明 |
|---:|---|---|
| 16 | `_importable()` | _importable(name |
| 27 | `is_flashinfer_installed()` | is_flashinfer_installed() -> bool |
| 32 | `is_sgl_kernel_installed()` | is_sgl_kernel_installed() -> bool |
| 37 | `is_vllm_installed()` | is_vllm_installed() -> bool |
| 42 | `device_capability()` | Compute capability of the current device as (major, minor); (0, 0) without CUDA. |
| 51 | `is_rocm()` | True when torch is built for ROCm (AMD GPU). |
| 58 | `driver_hip_version()` | ROCm driver version, or None if undetermined. |
| 69 | `driver_cuda_version()` | Max CUDA version the installed NVIDIA driver supports (``13000`` == CUDA 13.0), |

## `freetoken/kernel/batch_memcpy.py`  (69 行)

| 行 | 符号 | 说明 |
|---:|---|---|
| 15 | `_jit_batch_memcpy_module()` | _jit_batch_memcpy_module() -> Module |
| 23 | `_probe()` | One real 16-byte H2D through the binding. A version-gated build (panic |
| 42 | `load_batch_memcpy()` | Build (once), probe, and return the batch-memcpy entry point, or raise. |
| 57 | `batch_memcpy_jit()` | Enqueue one cudaMemcpyBatchAsync of ``len(sizes)`` independent copies. |

## `freetoken/kernel/causal_conv1d.py`  (75 行)

| 行 | 符号 | 说明 |
|---:|---|---|
| 18 | `causal_conv1d_varlen()` | Varlen (prefill) depthwise causal conv with silu; writes silu(conv) into ``x`` |
| 51 | `causal_conv1d_decode()` | Single-token (decode) causal conv update with silu; shifts+appends the new |

## `freetoken/kernel/fast_index_copy.py`  (209 行)

| 行 | 符号 | 说明 |
|---:|---|---|
| 21 | `_skip_fast_index_copy_enabled()` | _skip_fast_index_copy_enabled() -> bool |
| 26 | `_jit_update_flag_module()` | _jit_update_flag_module() -> Module |
| 35 | `_jit_fast_index_copy_module()` | _jit_fast_index_copy_module( *, feature_size |
| 62 | `_default_worker_threads()` | _default_worker_threads(feature_size |
| 70 | `_shrink_worker_feature_size()` | _shrink_worker_feature_size(feature_size |
| 78 | `default_worker_args()` | (worker_threads, worker_feature_size, num_block) of the default call path. |
| 91 | `fast_index_copy_jit()` | fast_index_copy_jit( dst |
| 143 | `_jit_fast_index_copy_multi_module()` | _jit_fast_index_copy_multi_module(*, num_threads |
| 152 | `default_blocks_per_bank()` | H2D gather grid width per bank. The link's bandwidth-delay product sets how many |
| 170 | `fast_index_copy_multi_jit()` | Fused multi-bank index copy: copy the same rows for every bank in ONE launch. |
| 205 | `update_copy_flag_jit()` | update_copy_flag_jit(sync_flag |

## `freetoken/kernel/gguf.py`  (142 行)

| 行 | 符号 | 说明 |
|---:|---|---|
| 25 | `_host_compiler()` | A host compiler nvcc + libtorch headers accept. |
| 43 | `_c_compiler_for()` | _c_compiler_for(cxx |
| 51 | `_module()` | _module() |
| 79 | `ggml_dequantize()` | Dequantize a packed GGUF weight ``[m, row_bytes]`` to a dense ``[m, n]`` tensor. |
| 86 | `ggml_mul_mat_vec_a8()` | MMVQ: small-batch GEMV with on-the-fly dequant. ``row`` = output features. |
| 93 | `ggml_mul_mat_a8()` | MMQ: large-batch quantized matmul. ``row`` = output features. |
| 100 | `ggml_moe_a8()` | MMQ grouped expert matmul over stacked experts ``weight[E, row, *]``. |
| 118 | `ggml_moe_a8_vec()` | MMVQ grouped expert GEMV over stacked experts ``weight[E, row, *]``. |
| 131 | `ggml_moe_get_block_size()` | ggml_moe_get_block_size(quant_type |

## `freetoken/kernel/index.py`  (54 行)

| 行 | 符号 | 说明 |
|---:|---|---|
| 16 | `_jit_index_module()` | _jit_index_module( element_size |
| 31 | `num_splits_for()` | Split factor for a row of ``element_size`` bytes; also used by the AOT |
| 41 | `indexing()` | indexing( weights |

## `freetoken/kernel/moe_impl.py`  (504 行)

| 行 | 符号 | 说明 |
|---:|---|---|
| 6 | `fused_moe_kernel_triton()` | fused_moe_kernel_triton( A |
| 65 | `moe_align_block_size_triton()` | moe_align_block_size_triton( topk_ids |
| 169 | `moe_sum_reduce_triton()` | moe_sum_reduce_triton(input |
| 200 | `mxfp4_fused_moe_kernel_t_triton()` | Grouped/sorted MXFP4 fused-MoE GEMM over the TRANSPOSED weight layout |
| 288 | `gpt_oss_swiglu_triton()` | gpt_oss_swiglu_triton( gate_up |
| 326 | `fused_moe_decode_kernel_triton()` | fused_moe_decode_kernel_triton( A |
| 399 | `gpt_oss_fused_routing()` | Fused softmax-over-top-k router for GPT-OSS (norm_topk_prob=True). |
| 443 | `get_fp4_lut()` | get_fp4_lut(device |
| 451 | `mxfp4_splitk_gemv_triton()` | Split-K expert GEMV over MXFP4 weights stored transposed as [E, K//2, N] |

## `freetoken/kernel/pinned.py`  (68 行)

| 行 | 符号 | 说明 |
|---:|---|---|
| 17 | `_load_pinned_extension()` | _load_pinned_extension() |
| 27 | `create_pinned_tensor_like()` | Create a CPU pinned tensor with the same size, stride, and dtype as input. |
| 33 | `copy_to_pinned_tensor()` | Copy a CPU tensor into exact-size cudaMallocHost pinned storage. |
| 42 | `alloc_pinned_tensor()` | Allocate an exact-size, uninitialized pinned host tensor via cudaHostAlloc. |
| 48 | `host_register()` | cudaHostRegister ``nbytes`` at ``addr`` as portable+mapped (pin-after-fill). |
| 54 | `_host_ptr_identity()` | _host_ptr_identity() -> bool |
| 59 | `device_ptr()` | Base address of ``t`` as the GPU must dereference it. |

## `freetoken/kernel/pynccl.py`  (79 行)

| 行 | 符号 | 说明 |
|---:|---|---|
| 16 | **class `PyNCCLCommunicator`** | class PyNCCLCommunicator |
| 18 | **class `all_reduce`** | all_reduce(self, input |
| 20 | **class `all_gather`** | all_gather(self, output |
| 22 | **class `get_buffer`** | get_buffer(self) -> int |
| 29 | `_load_nccl_module()` | _load_nccl_module() -> Module |
| 35 | `_get_pynccl_wrapper_cls()` | _get_pynccl_wrapper_cls() |
| 39 | **class `PyNCCLImpl`** | class PyNCCLImpl(tvm_ffi.Object) |
| 40 | **class `__init__`** | __init__(self, *args) |
| 46 | `init_pynccl()` | init_pynccl( *, tp_rank |

## `freetoken/kernel/radix.py`  (20 行)

| 行 | 符号 | 说明 |
|---:|---|---|
| 14 | `_load_radix_module()` | _load_radix_module() -> Module |
| 18 | `fast_compare_key()` | fast_compare_key(x |

## `freetoken/kernel/row_store.py`  (54 行)

| 行 | 符号 | 说明 |
|---:|---|---|
| 23 | `probe_wait_sync()` | ``mode``: ``auto`` (probe), ``wait`` (require memops) or ``gate`` (never use them). |
| 44 | `wait_reset()` | Enqueue WAIT(flag >= 1) then flag := 0 on ``stream`` (capturable). |
| 49 | `signal()` | Host side: release a pending WAIT (a release-store of 1). |

## `freetoken/kernel/store.py`  (42 行)

| 行 | 符号 | 说明 |
|---:|---|---|
| 16 | `_jit_store_module()` | _jit_store_module( element_size |
| 30 | `store_cache()` | store_cache( k_cache |

## `freetoken/kernel/tensor.py`  (19 行)

| 行 | 符号 | 说明 |
|---:|---|---|
| 14 | `_load_test_tensor_module()` | _load_test_tensor_module() -> Module |
| 18 | `test_tensor()` | test_tensor(x |

## `freetoken/kernel/utils.py`  (420 行)

| 行 | 符号 | 说明 |
|---:|---|---|
| 32 | `_is_rocm()` | _is_rocm() -> bool |
| 38 | `_cuda_arch_list()` | Archs a CUDA build targets: the AOT build's TVM_FFI_CUDA_ARCH_LIST, else the GPU this process is bound to. |
| 51 | `_rocm_arch_list()` | The single ROCm GPU this process is bound to, or an explicit cross-compile target. |
| 65 | `_pin_tvm_ffi_arch_ctx()` | Pin tvm-ffi to the resolved targets for one build, then restore the caller's environment. |
| 81 | `_cuda_cflags()` | CUDA nvcc flags for a kernel build. tvm-ffi emits one SASS cubin per arch in ``arch_list`` and no PTX, so add the PTX of the highest arch: a GPU ne... |
| 85 | &nbsp;&nbsp;&nbsp;↳ `_rank()` | _rank(a |
| 94 | `_hip_cflags()` | HIP flags for a kernel build on ROCm. |
| 101 | `_rocm_link_flags()` | Make ROCm's runtime library discoverable to JIT link commands. |
| 150 | **class `CppArgList`** | class CppArgList(list[str]) |
| 151 | **class `__str__`** | __str__(self) -> str |
| 155 | **class `KernelConfig`** | class KernelConfig(NamedTuple) |
| 161 | **class `template_args`** | template_args(self) -> str |
| 166 | `_make_name()` | _make_name(*args |
| 170 | `_env_enabled()` | _env_enabled(name |
| 174 | `_freetoken_version()` | _freetoken_version() -> str |
| 180 | `_version_parts()` | Split a PEP 440 version string into (release, local segments): |
| 187 | `_build_stamps()` | The `g<sha>` commit-stamp tokens of a local version segment list |
| 193 | `_kernel_cache_version_ok()` | Same release -- and, when both sides carry a `g<sha>` stamp, the same build. |
| 210 | `_kernel_cache_dir()` | _kernel_cache_dir() -> pathlib.Path \| None |
| 252 | `_load_prebuilt()` | _load_prebuilt(name |
| 276 | `_make_wrapper()` | _make_wrapper(tup |
| 281 | `make_cpp_args()` | make_cpp_args(*args |
| 282 | &nbsp;&nbsp;&nbsp;↳ `_convert()` | _convert(arg |
| 292 | `load_aot()` | load_aot( *args |
| 352 | `load_jit()` | load_jit( *args |

## `freetoken/kernel/fla/__init__.py`  (57 行)

_（无类/函数定义，模块级代码）_

## `freetoken/kernel/fla/chunk.py`  (257 行)

| 行 | 符号 | 说明 |
|---:|---|---|
| 31 | `chunk_gated_delta_rule_fwd()` | chunk_gated_delta_rule_fwd( q |
| 82 | **class `ChunkGatedDeltaRuleFunction`** | class ChunkGatedDeltaRuleFunction(torch.autograd.Function) |
| 87 | **class `forward`** | forward( ctx, q |
| 128 | `chunk_gated_delta_rule()` | Args: |

## `freetoken/kernel/fla/chunk_delta_h.py`  (357 行)

| 行 | 符号 | 说明 |
|---:|---|---|
| 53 | `chunk_gated_delta_rule_fwd_kernel_h_blockdim64()` | chunk_gated_delta_rule_fwd_kernel_h_blockdim64( k, v, w, v_new, g, gk, h, initial_state, initial_state_indices, cu_seqlens, chunk_offsets, T, H |
| 295 | `chunk_gated_delta_rule_fwd_h()` | chunk_gated_delta_rule_fwd_h( k |
| 328 | &nbsp;&nbsp;&nbsp;↳ `grid()` | grid(meta) |

## `freetoken/kernel/fla/chunk_fwd.py`  (416 行)

| 行 | 符号 | 说明 |
|---:|---|---|
| 40 | `chunk_gated_delta_rule_fwd_kkt_solve_kernel()` | Fused kernel: compute beta * K @ K^T (lower triangular) + solve_tril (I+A)^{-1} in one pass. |
| 339 | `chunk_gated_delta_rule_fwd_intra()` | GDN intra-chunk forward: fused kkt + solve_tril + recompute_w_u. |

## `freetoken/kernel/fla/chunk_o.py`  (174 行)

| 行 | 符号 | 说明 |
|---:|---|---|
| 30 | `chunk_fwd_kernel_o()` | chunk_fwd_kernel_o( q, k, v, h, g, o, cu_seqlens, chunk_indices, scale, T, H |
| 126 | `chunk_fwd_o()` | chunk_fwd_o( q |
| 148 | &nbsp;&nbsp;&nbsp;↳ `grid()` | grid(meta) |

## `freetoken/kernel/fla/cumsum.py`  (299 行)

| 行 | 符号 | 说明 |
|---:|---|---|
| 26 | `chunk_local_cumsum_scalar_kernel()` | chunk_local_cumsum_scalar_kernel( s, o, scale, cu_seqlens, chunk_indices, T, B |
| 86 | `chunk_local_cumsum_vector_kernel()` | chunk_local_cumsum_vector_kernel( s, o, scale, cu_seqlens, chunk_indices, T, B |
| 164 | `chunk_local_cumsum_scalar()` | chunk_local_cumsum_scalar( g |
| 207 | `chunk_local_cumsum_vector()` | chunk_local_cumsum_vector( g |
| 231 | &nbsp;&nbsp;&nbsp;↳ `grid()` | grid(meta) |
| 257 | `chunk_local_cumsum()` | chunk_local_cumsum( g |

## `freetoken/kernel/fla/fused_recurrent.py`  (659 行)

| 行 | 符号 | 说明 |
|---:|---|---|
| 32 | `fused_recurrent_gated_delta_rule_fwd_kernel()` | fused_recurrent_gated_delta_rule_fwd_kernel( q, k, v, g, beta, o, h0, ht, cu_seqlens, ssm_state_indices, num_accepted_tokens, a_log, g_bias, scale, N |
| 212 | `fused_recurrent_gated_delta_rule_fwd()` | fused_recurrent_gated_delta_rule_fwd( q |
| 296 | `fused_recurrent_gated_delta_rule_packed_decode_kernel()` | fused_recurrent_gated_delta_rule_packed_decode_kernel( mixed_qkv, a, b, A_log, dt_bias, o, h0, ht, ssm_state_indices, scale, stride_mixed_qkv_tok |
| 379 | `fused_recurrent_gated_delta_rule_packed_decode()` | fused_recurrent_gated_delta_rule_packed_decode( mixed_qkv |
| 521 | **class `FusedRecurrentFunction`** | class FusedRecurrentFunction(torch.autograd.Function) |
| 523 | **class `forward`** | forward( ctx, q |
| 556 | `fused_recurrent_gated_delta_rule()` | Args: |

## `freetoken/kernel/fla/fused_sigmoid_gating_recurrent.py`  (372 行)

| 行 | 符号 | 说明 |
|---:|---|---|
| 15 | `fused_sigmoid_gating_delta_rule_update_kernel()` | Fused kernel that combines sigmoid gating computation with recurrent delta rule update. |
| 247 | `fused_sigmoid_gating_delta_rule_update()` | Fused triton implementation of sigmoid gating delta rule update. |

## `freetoken/kernel/fla/index.py`  (35 行)

| 行 | 符号 | 说明 |
|---:|---|---|
| 12 | `prepare_lens()` | prepare_lens(cu_seqlens |
| 17 | `prepare_chunk_indices()` | prepare_chunk_indices( cu_seqlens |
| 30 | `prepare_chunk_offsets()` | prepare_chunk_offsets( cu_seqlens |

## `freetoken/kernel/fla/kda.py`  (1396 行)

| 行 | 符号 | 说明 |
|---:|---|---|
| 39 | `fused_recurrent_kda_fwd()` | fused_recurrent_kda_fwd( q |
| 145 | `fused_recurrent_kda()` | fused_recurrent_kda( q |
| 208 | `chunk_kda_scaled_dot_kkt_fwd_kernel_intra_sub_inter()` | chunk_kda_scaled_dot_kkt_fwd_kernel_intra_sub_inter( q, k, g, beta, A, Aqk, scale, cu_seqlens, chunk_indices, T, H |
| 314 | `chunk_kda_scaled_dot_kkt_fwd_kernel_intra_sub_intra()` | chunk_kda_scaled_dot_kkt_fwd_kernel_intra_sub_intra( q, k, g, beta, A, Aqk, scale, cu_seqlens, chunk_indices, T, H |
| 404 | `chunk_kda_scaled_dot_kkt_fwd()` | Compute beta * K * K^T. |
| 504 | `recompute_w_u_fwd_kernel()` | recompute_w_u_fwd_kernel( q, k, qg, kg, v, beta, w, u, A, gk, cu_seqlens, chunk_indices, T, H |
| 647 | `recompute_w_u_fwd()` | recompute_w_u_fwd( k |
| 706 | `chunk_gla_fwd_kernel_o()` | chunk_gla_fwd_kernel_o( q, v, g, h, o, A, cu_seqlens, chunk_indices, scale, T, H |
| 815 | `chunk_gla_fwd_o_gk()` | chunk_gla_fwd_o_gk( q |
| 834 | &nbsp;&nbsp;&nbsp;↳ `grid()` | grid(meta) |
| 871 | `kda_gate_cumsum_fwd_kernel()` | kda_gate_cumsum_fwd_kernel( g, A, y, g_bias, cu_seqlens, chunk_indices, cumsum_scale, beta, threshold, SAFE_GATE |
| 954 | `fused_kda_gate_chunk_cumsum()` | fused_kda_gate_chunk_cumsum( raw_g |
| 981 | &nbsp;&nbsp;&nbsp;↳ `grid()` | grid(meta) |
| 1007 | `_chunk_kda_fwd_with_cumulative_g()` | _chunk_kda_fwd_with_cumulative_g( q |
| 1092 | `chunk_kda_fwd()` | chunk_kda_fwd( q |
| 1133 | `chunk_kda_with_fused_gate_fwd()` | chunk_kda_with_fused_gate_fwd( q |
| 1181 | `chunk_kda()` | chunk_kda( q |
| 1215 | `chunk_kda_with_fused_gate()` | Run chunk KDA from raw gate projection using fused gate+cumsum. |
| 1275 | `kda_gate_fwd_kernel()` | kda_gate_fwd_kernel( g, A, y, g_bias, beta |
| 1344 | `fused_kda_gate()` | Forward pass for KDA gate: |
| 1375 | &nbsp;&nbsp;&nbsp;↳ `grid()` | grid(meta) |

## `freetoken/kernel/fla/kda_chunk_delta_h.py`  (394 行)

| 行 | 符号 | 说明 |
|---:|---|---|
| 54 | `chunk_gated_delta_rule_fwd_kernel_h_blockdim64()` | chunk_gated_delta_rule_fwd_kernel_h_blockdim64( k, v, w, v_new, g, gk, h, h0, ht, cu_seqlens, chunk_offsets, T, H |
| 332 | `chunk_gated_delta_rule_fwd_h()` | chunk_gated_delta_rule_fwd_h( k |
| 371 | &nbsp;&nbsp;&nbsp;↳ `grid()` | grid(meta) |

## `freetoken/kernel/fla/l2norm.py`  (147 行)

| 行 | 符号 | 说明 |
|---:|---|---|
| 24 | `l2norm_fwd_kernel1()` | l2norm_fwd_kernel1( x, y, D, BD |
| 55 | `l2norm_fwd_kernel()` | l2norm_fwd_kernel( x, y, eps, T, D |
| 73 | `l2norm_fwd()` | l2norm_fwd( x |
| 94 | &nbsp;&nbsp;&nbsp;↳ `grid()` | grid(meta) |
| 122 | **class `L2NormFunction`** | class L2NormFunction(torch.autograd.Function) |
| 126 | **class `forward`** | forward(ctx, x, eps=1e-6, output_dtype=None) |
| 130 | `l2norm()` | l2norm( x |
| 139 | **class `L2Norm`** | class L2Norm(nn.Module) |
| 141 | **class `__init__`** | __init__(self, eps |
| 146 | **class `forward`** | forward(self, x |

## `freetoken/kernel/fla/layernorm_gated.py`  (264 行)

| 行 | 符号 | 说明 |
|---:|---|---|
| 23 | `_device_context()` | _device_context(device |
| 29 | `_layer_norm_fwd_1pass_kernel()` | _layer_norm_fwd_1pass_kernel( X,  # pointer to the input Y,  # pointer to the output W,  # pointer to the weights B,  # pointer to the biases Z,  #... |
| 134 | `_get_sm_count()` | Get and cache the SM count for a given device. |
| 140 | `calc_rows_per_block()` | calc_rows_per_block(M |
| 147 | `_layer_norm_fwd()` | _layer_norm_fwd( x, weight, bias, eps, z=None, out=None, group_size=None, norm_before_gate=True, is_rms_norm=False, activation |
| 223 | `rms_norm_gated()` | If z is not None, we do norm(x) * silu(z) if norm_before_gate, else norm(x * silu(z)) |

## `freetoken/kernel/fla/op.py`  (66 行)

| 行 | 符号 | 说明 |
|---:|---|---|
| 26 | `safe_exp()` | safe_exp(x) |
| 33 | `gather()` | Gather operation that works when tl.gather is not supported. |
| 59 | `make_tensor_descriptor()` | make_tensor_descriptor( base, shape, strides, block_shape, _builder=None, ) |

## `freetoken/kernel/fla/solve_tril.py`  (563 行)

| 行 | 符号 | 说明 |
|---:|---|---|
| 43 | `solve_tril_16x16_kernel()` | solve_tril_16x16_kernel( A, Ai, cu_seqlens, chunk_indices, T, H |
| 118 | `merge_16x16_to_32x32_inverse_kernel()` | merge_16x16_to_32x32_inverse_kernel( A, Ai, cu_seqlens, chunk_indices, T, H |
| 243 | `merge_16x16_to_64x64_inverse_kernel()` | merge_16x16_to_64x64_inverse_kernel( A, Ai, cu_seqlens, chunk_indices, T, H |
| 512 | `solve_tril()` | Compute the inverse of the matrix I + A |

## `freetoken/kernel/fla/utils.py`  (356 行)

| 行 | 符号 | 说明 |
|---:|---|---|
| 38 | `check_environments()` | Checks the current operating system, Triton version, and Python version, |
| 75 | `get_abs_err()` | get_abs_err(x, y) |
| 79 | `get_err_ratio()` | get_err_ratio(x, y) |
| 85 | `assert_close()` | assert_close(prefix, ref, tri, ratio, warning=False, err_atol=1e-6) |
| 104 | `tensor_cache()` | A decorator that caches the most recent results of a function with tensor inputs. |
| 121 | &nbsp;&nbsp;&nbsp;↳ `wrapper()` | wrapper(*args |
| 146 | `input_guard()` | A decorator to make sure all input tensors are contiguous and set the device based on input tensors. |
| 152 | &nbsp;&nbsp;&nbsp;↳ `wrapper()` | wrapper(*args, **kwargs) |
| 186 | `require_version()` | Perform a runtime check of the dependency versions, using the exact same syntax used by pip. |
| 191 | &nbsp;&nbsp;&nbsp;↳ `decorator()` | decorator(fn) |
| 193 | &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;↳ `wrapper()` | wrapper(ctx, *args, **kwargs) |
| 214 | `checkpoint()` | checkpoint(fn) |
| 215 | &nbsp;&nbsp;&nbsp;↳ `wrapper()` | wrapper(*args, **kwargs) |
| 221 | `_cpu_device_warning()` | _cpu_device_warning() |
| 230 | `get_multiprocessor_count()` | get_multiprocessor_count(tensor_idx |
| 241 | `get_available_device()` | get_available_device() -> str |
| 250 | `_check_platform()` | _check_platform() -> Literal["nvidia", "amd", "intel", "musa"] |
| 298 | `get_all_max_shared_mem()` | get_all_max_shared_mem() |
| 311 | **class `Backend`** | class Backend(Enum) |
| 318 | **class `get_shared_memory`** | get_shared_memory(cls, arch |
| 326 | `check_shared_mem()` | check_shared_mem(arch |
| 342 | `custom_device_ctx()` | custom_device_ctx(index |
| 352 | `custom_device_ctx()` | custom_device_ctx(index |

## `freetoken/kernel/fla/wy_fast.py`  (156 行)

| 行 | 符号 | 说明 |
|---:|---|---|
| 23 | `recompute_w_u_fwd_kernel()` | recompute_w_u_fwd_kernel( k, v, beta, w, u, A, g, cu_seqlens, chunk_indices, T, H |
| 111 | `recompute_w_u_fwd()` | recompute_w_u_fwd( k |

## `freetoken/kernel/triton/activation.py`  (199 行)

| 行 | 符号 | 说明 |
|---:|---|---|
| 45 | `_pdl_supported()` | _pdl_supported() -> bool |
| 53 | `_fast_tanh()` | _fast_tanh(x) |
| 64 | `_fast_ex2()` | _fast_ex2(x) |
| 75 | `_act_and_mul_kernel()` | _act_and_mul_kernel( out_ptr, x_ptr, d, alpha, limit, ACT |
| 127 | `_act_and_mul()` | _act_and_mul( kind |
| 156 | `silu_and_mul()` | silu_and_mul(x |
| 160 | `gelu_and_mul()` | gelu_and_mul(x |
| 164 | `gelu_tanh_and_mul()` | gelu_tanh_and_mul(x |
| 168 | `swigluoai_and_mul()` | SwiGLU-OAI over UNINTERLEAVED halves (gate ``x[..., :d]``, up ``x[..., d:]``): |
| 182 | `swiglu_clamp_and_mul()` | GLM-5.3 clamped SwiGLU over UNINTERLEAVED halves: ``clamp(gate, max=limit) * sigmoid(alpha * gate) * clamp(up, +-limit)``. |

## `freetoken/kernel/triton/attention.py`  (977 行)

| 行 | 符号 | 说明 |
|---:|---|---|
| 15 | `_optin_smem_bytes()` | Per-block opt-in shared-memory budget for a CUDA device (0 if unavailable). |
| 21 | `_select_extend_tile()` | Pick ``(BLOCK_M, BLOCK_N)`` for the extend/prefill kernel, shared-memory aware. |
| 33 | &nbsp;&nbsp;&nbsp;↳ `fits()` | fits(block_m |
| 46 | `_paged_attention_kernel()` | _paged_attention_kernel( q_ptr, k_ptr, v_ptr, o_ptr, indptr_ptr, indices_ptr, q_to_req_ptr, q_pos_ptr, sm_scale, sinks_ptr, stride_qt, stride_qh, s... |
| 148 | `_decode_grouped_stage1_kernel()` | _decode_grouped_stage1_kernel( q_ptr, k_ptr, v_ptr, sm_scale, indptr_ptr, indices_ptr, q_pos_ptr, mid_o_ptr, mid_lse_ptr, num_kv_splits_ptr, stride... |
| 274 | `_decode_stage2_kernel()` | _decode_stage2_kernel( mid_o_ptr, mid_lse_ptr, o_ptr, indptr_ptr, q_pos_ptr, num_kv_splits_ptr, sinks_ptr, stride_mid_ob, stride_mid_oh, stride_mid... |
| 352 | `decode_paged_attention()` | SGLang-style split-k grouped decode attention for one query per request. |
| 470 | `_extend_attention_kernel()` | _extend_attention_kernel( q_ptr, k_ptr, v_ptr, o_ptr, qo_indptr_ptr, kv_indptr_ptr, kv_indices_ptr, prefix_lens_ptr, block_ends_ptr, sm_scale, sink... |
| 599 | `_extend_attention_split_kernel()` | _extend_attention_split_kernel( q_ptr, k_extend_ptr, v_extend_ptr, k_cache_ptr, v_cache_ptr, o_ptr, qo_indptr_ptr, kv_indptr_ptr, kv_indices_ptr, p... |
| 781 | `extend_paged_attention()` | Block-tiled causal prefill/extend attention over paged KV cache; block_ends holds per query token the end of the multimodal span it sits in (0 for ... |
| 911 | `paged_attention()` | Paged causal attention for one layer. |

## `freetoken/kernel/triton/autotune_cache.py`  (37 行)

_（无类/函数定义，模块级代码）_

## `freetoken/kernel/triton/causal_conv1d_triton.py`  (681 行)

| 行 | 符号 | 说明 |
|---:|---|---|
| 55 | `_causal_conv1d_fwd_tiled_kernel()` | _causal_conv1d_fwd_tiled_kernel( x_ptr, w_ptr, bias_ptr, initial_states_ptr, cache_indices_ptr, has_initial_states_ptr, query_start_loc_ptr, o_ptr,... |
| 249 | `_causal_conv1d_update_kernel()` | _causal_conv1d_update_kernel( x_ptr, w_ptr, bias_ptr, conv_state_ptr, conv_state_indices_ptr, o_ptr, batch |
| 422 | `causal_conv1d_varlen()` | causal_conv1d_varlen( x |
| 471 | &nbsp;&nbsp;&nbsp;↳ `grid()` | grid(META) |
| 520 | `causal_conv1d_decode()` | causal_conv1d_decode( x |
| 547 | &nbsp;&nbsp;&nbsp;↳ `grid()` | grid(META) |
| 594 | `causal_conv1d_fn()` | Varlen (prefill) depthwise causal conv with fused silu; conv_states updated |
| 633 | `causal_conv1d_update()` | Single-token (decode) causal conv update with silu; conv_state[idx] shifted |

## `freetoken/kernel/triton/decode_moe.py`  (75 行)

| 行 | 符号 | 说明 |
|---:|---|---|
| 6 | `fused_moe_decode_kernel()` | fused_moe_decode_kernel( a_ptr, b_ptr, c_ptr, topk_weights_ptr, topk_ids_ptr, total_routes |

## `freetoken/kernel/triton/df11.py`  (318 行)

| 行 | 符号 | 说明 |
|---:|---|---|
| 35 | `_huffman_lengths()` | Code length per symbol from a Huffman tree (length >= 1, even for a lone symbol). |
| 51 | &nbsp;&nbsp;&nbsp;↳ `walk()` | walk(node, depth) |
| 62 | `_length_limited_lengths()` | Huffman lengths capped at ``lmax`` by flooring rare-symbol frequencies (stays a valid |
| 75 | `_canonical_codes()` | Canonical Huffman codes: sort by (length, symbol), increment-and-shift. |
| 89 | `_build_lut()` | One ``2^lmax`` LUT mapping the next ``lmax`` peeked bits -> ``(symbol << 8) \| length``. |
| 104 | `_huffman_tables()` | From a 256-bin exponent histogram build the decode ``lut`` and the per-symbol |
| 120 | `df11_compress()` | Compress a 2-D BF16 ``weight`` into the DF11 tensor bundle (see module docstring). |
| 175 | `df11_decompress_ref()` | Pure-Python reference decoder (slow; for correctness tests on small tensors). |
| 186 | &nbsp;&nbsp;&nbsp;↳ `peek()` | peek(pos |
| 213 | `df11_compress_rows()` | Row-contiguous DF11: every ROW is an independently-decodable chunk. |
| 281 | `df11_decompress_rows_ref()` | Pure-Python reference decoder for the row-contiguous layout (small tensors only). |
| 291 | &nbsp;&nbsp;&nbsp;↳ `peek()` | peek(pos |

## `freetoken/kernel/triton/df11_decode.py`  (171 行)

| 行 | 符号 | 说明 |
|---:|---|---|
| 20 | `_df11_decode_kernel()` | _df11_decode_kernel( low8_ptr,        # [N]       uint8   |
| 78 | `df11_decompress()` | Decode a DF11 bundle back to its exact BF16 tensor. |
| 98 | `_df11_gather_kernel()` | _df11_gather_kernel( low8_ptr,        # [R*C]     uint8   |
| 150 | `df11_gather_decode()` | Decode the embedding rows ``ids`` from a row-contiguous DF11 bundle into ``[T, C]`` bf16. |

## `freetoken/kernel/triton/e4m3_compat.py`  (142 行)

| 行 | 符号 | 说明 |
|---:|---|---|
| 36 | `_env_force()` | _env_force() -> bool |
| 49 | `e4m3_native()` | Host-side twin of :func:`e4m3_native_cx`: True when kernels take fp8e4nv |
| 74 | `e4m3_kernel_view()` | An e4m3 tensor as the branched kernels expect it: unchanged when native, |
| 80 | `e4m3_act_dtype()` | Buffer dtype for quantized activations: fp8 when native, else bf16 (every |
| 87 | `e4m3_native_cx()` | Compile-time: does the compilation target have native fp8e4nv (sm_89+)? |
| 96 | `e4m3_u8_to_f32()` | Decode e4m3 bits (uint8) to fp32: place exp+mantissa in the fp16 field |
| 104 | `e4m3_u8_to_f16_x128()` | Decode e4m3 bits (uint8) to fp16 pre-scaled by 128 (the nvfp4 GEMM form: |
| 113 | `e4m3_f32_to_u8()` | Encode an fp32 value that already sits on the e4m3 grid (see :func:`round_e4m3`, \|x\| <= 448) |
| 127 | `round_e4m3()` | Round fp32 onto the e4m3 value grid (RNE), fp32 -> fp32, in a SINGLE |

## `freetoken/kernel/triton/fp8_block_linear.py`  (279 行)

| 行 | 符号 | 说明 |
|---:|---|---|
| 38 | `_act_quant_kernel()` | _act_quant_kernel( x_ptr, y_ptr, s_ptr, M, K, stride_xm, stride_xk, stride_ym, stride_yk, stride_sm, stride_sk, BLOCK_M |
| 63 | `per_token_group_quant_fp8()` | Quantize ``x`` ``[..., K]`` to fp8 with one fp32 scale per token per 128-K group. |
| 88 | `_block_fp8_gemm_kernel()` | ``BLOCK_N == BLOCK_K == 128`` so each tile maps to exactly one weight scale block |
| 126 | `block_fp8_matmul()` | ``y = (a_fp8 * a_scale) @ (weight * weight_scale)^T`` over 128-K blocks. |
| 163 | `_block_fp8_gemv_splitk_kernel()` | Each pid_k split covers ``kb_per`` 128-K-blocks (ceil-tiled, masked to ``n_kb``), so |
| 202 | `_splitk_reduce_kernel()` | _splitk_reduce_kernel(part_ptr, out_ptr, N, SPLIT_K |
| 212 | `_block_fp8_gemv()` | M==1 split-K GEMV. ``a`` [K]; ``weight`` [N,K] fp8, scale [N//128,K//128]. W8A16 when |
| 242 | `block_fp8_linear()` | ``y = x @ weight^T`` with block-fp8 weight. Decode (M=1) uses a W8A16 split-K GEMV |
| 261 | `dequant_block_fp8()` | Dequantize a block-fp8 weight ``[N, K]`` (+ scale ``[N//block, K//block]``) to bf16. |

## `freetoken/kernel/triton/fp8_blockscale_moe.py`  (262 行)

| 行 | 符号 | 说明 |
|---:|---|---|
| 24 | `_decode_fp8_moe_kernel()` | _decode_fp8_moe_kernel( a_ptr,        # [M, K] activation (compute dtype, bf16) w_ptr,        # [E, N, K] float8_e4m3fn s_ptr,        # [E, N//128,... |
| 78 | `_decode_gemm()` | _decode_gemm(a, w, s, c, topk_weights, topk_ids, mul_routed_weight, a_row_is_route) |
| 99 | `fused_experts_decode_fp8_blockscale()` | Decode (bs-1) inline-dequant block-fp8 MoE. ``topk_ids`` index rows of the expert |
| 131 | `_prefill_fp8_moe_kernel()` | W8A8: ``a`` is fp8 (per-128-K-group act scale ``a_scale``), ``w`` fp8 (per-128x128 |
| 199 | `_prefill_gemm()` | _prefill_gemm(a_fp8, a_scale, w, s, c, tw, sorted_ids, expert_ids, ntpp, num_valid, kernel_top_k, mul_routed_weight, cfg) |
| 217 | `fused_experts_fp8_blockscale()` | Prefill inline-dequant block-fp8 MoE. ``topk_ids`` index expert rows in [0, num_experts) |

## `freetoken/kernel/triton/fp8_pertensor_linear.py`  (356 行)

| 行 | 符号 | 说明 |
|---:|---|---|
| 43 | `_torch_version()` | _torch_version() -> tuple[int, int] |
| 49 | `rowwise_scaled_mm_ok()` | Whether row-wise ``torch._scaled_mm`` may be issued from a side stream on this GPU. |
| 76 | `weight_scale_segments()` | ``[start, end)`` row ranges over which ``weight_scale`` is constant (the fused parts). |
| 88 | `_segments_w8a8_ok()` | cuBLASLt needs 16-row aligned fp8 operands; more parts than a fused projection has |
| 98 | `_gemv_splitk_kernel()` | Each (pid_n, pid_k) computes the partial sum over ``kb_per`` BLOCK_K chunks for a |
| 133 | `_splitk_reduce_kernel()` | _splitk_reduce_kernel( part_ptr, scale_ptr, out_ptr, N, SPLIT_K |
| 146 | `_gemv()` | M==1 split-K GEMV. ``a`` [K] bf16; ``weight`` [N, K] fp8; ``weight_scale`` [N] fp32. |
| 177 | `_gemm_kernel()` | _gemm_kernel( a_ptr, w_ptr, scale_ptr, c_ptr, M, N, K, stride_am, stride_ak, stride_wn, stride_wk, stride_cm, stride_cn, BLOCK_M |
| 210 | `_gemm()` | M>1 W8A16 GEMM. ``a`` [M, K] bf16; ``weight`` [N, K] fp8; ``weight_scale`` [N] fp32. |
| 249 | `_static_quant_kernel()` | bf16 activation -> fp8-e4m3 under one broadcast per-tensor scale. |
| 259 | `_static_quant()` | One launch, no host sync: the scale is read from device memory inside the kernel, so |
| 271 | `_scaled_mm()` | ``a @ (weight_fp8 * weight_scale)^T`` as a W8A8 cuBLASLt GEMM. |
| 314 | `fp8_pertensor_linear()` | ``y = x @ (weight_fp8 * weight_scale)^T``. ``weight`` [N, K] fp8-e4m3, ``weight_scale`` |

## `freetoken/kernel/triton/fused_moe.py`  (302 行)

| 行 | 符号 | 说明 |
|---:|---|---|
| 6 | `moe_align_block_size_init_kernel()` | moe_align_block_size_init_kernel( sorted_token_ids_ptr, expert_ids_ptr, tokens_cnts_ptr, cumsum_ptr, sorted_token_ids_numel |
| 30 | `moe_align_block_size_stage1_kernel()` | moe_align_block_size_stage1_kernel( topk_ids_ptr, tokens_cnts_ptr, num_experts |
| 51 | `moe_align_block_size_stage2_kernel()` | moe_align_block_size_stage2_kernel( tokens_cnts_ptr, num_experts |
| 64 | `moe_align_block_size_stage3_kernel()` | moe_align_block_size_stage3_kernel( total_tokens_post_pad_ptr, tokens_cnts_ptr, cumsum_ptr, num_experts |
| 81 | `moe_align_block_size_stage4_kernel()` | moe_align_block_size_stage4_kernel( topk_ids_ptr, sorted_token_ids_ptr, expert_ids_ptr, tokens_cnts_ptr, cumsum_ptr, num_experts |
| 112 | `moe_sum_reduce_kernel()` | moe_sum_reduce_kernel( input_ptr, input_stride_0, input_stride_1, input_stride_2, output_ptr, output_stride_0, output_stride_1, token_num |
| 157 | `fused_moe_kernel()` | Implements the fused computation for a Mixture of Experts (MOE) using |

## `freetoken/kernel/triton/gemma4_fused.py`  (191 行)

| 行 | 符号 | 说明 |
|---:|---|---|
| 10 | `_gemma_dual_rmsnorm_residual_kernel()` | _gemma_dual_rmsnorm_residual_kernel( X1_ptr, W1_ptr, X2_ptr, W2_ptr, W3_ptr, Residual_ptr, Scalar_ptr, Out_ptr, stride_x1, stride_x2, stride_r, str... |
| 57 | `gemma_dual_rmsnorm_residual_scalar()` | gemma_dual_rmsnorm_residual_scalar( x1 |
| 106 | `_gemma4_routing_kernel()` | _gemma4_routing_kernel( logits_ptr, per_expert_scale_ptr, topk_weights_ptr, topk_ids_ptr, stride_logits_t, E |
| 159 | `gemma4_fused_routing()` | gemma4_fused_routing( logits |

## `freetoken/kernel/triton/glm_dsa_sparse.py`  (431 行)

| 行 | 符号 | 说明 |
|---:|---|---|
| 38 | `_glm_dsa_sparse_kernel()` | _glm_dsa_sparse_kernel( q_ptr, pool_ptr, o_ptr, idx_ptr, cnt_ptr, scale, H, TOPK, stride_qb, stride_qm, stride_qh, stride_qd, stride_pn, stride_pd,... |
| 108 | `_glm_dsa_decode_logits_kernel()` | One decode query per request, gathering its own index keys (GLM variant of |
| 164 | `glm_dsa_decode_logits()` | Head-reduced indexer logits ``[B, W]`` fp32 for a decode step. |
| 202 | `_glm_dsa_splitk_kernel()` | Stage 1 (decode flash-decoding): each program reduces one BLOCK_T-aligned slice of |
| 291 | `_glm_dsa_merge_kernel()` | Stage 2: log-sum-exp merge over the splits (no attention sink -- GLM DSA has none, |
| 334 | `_split_count()` | 0 = single-program kernel; decode splits until SMs are covered (dsv4 policy). |
| 347 | `glm_dsa_sparse_attn()` | Sparse MLA attention over gathered latent rows; returns ``[b, m, h, d_v]``. |

## `freetoken/kernel/triton/hc.py`  (395 行)

| 行 | 符号 | 说明 |
|---:|---|---|
| 18 | `_pdl_supported()` | _pdl_supported() -> bool |
| 23 | `_grouped_gemma_rmsnorm_kernel()` | _grouped_gemma_rmsnorm_kernel( x_ptr, w_ptr, y_ptr, stride_x, stride_y, DIM |
| 66 | `grouped_gemma_rmsnorm()` | grouped_gemma_rmsnorm( x |
| 93 | `_hc_silu_kernel()` | _hc_silu_kernel( x_ptr, y_ptr, stride_x, stride_y, DIM |
| 119 | `hc_silu()` | hc_silu(x |
| 137 | `_hc_gate_mix_kernel()` | _hc_gate_mix_kernel( x_ptr, g_ptr, y_ptr, stride_x, stride_g, stride_y, DIM |
| 174 | `hc_gate_mix()` | hc_gate_mix(x |
| 200 | `_hc_combine_kernel()` | _hc_combine_kernel( block_ptr, res_ptr, inj_ptr, out_ptr, stride_block, stride_res, stride_inj, stride_out, HC_DIM |
| 243 | `hc_combine()` | hc_combine( residual |
| 278 | `_hc_combine_norm_kernel()` | _hc_combine_norm_kernel( block_ptr, res_ptr, inj_ptr, w_ptr, out_ptr, y_ptr, stride_block, stride_res, stride_inj, stride_out, stride_y, HC_DIM |
| 345 | `hc_combine_norm()` | hc_combine_norm( residual |

## `freetoken/kernel/triton/kpool_compress.py`  (146 行)

| 行 | 符号 | 说明 |
|---:|---|---|
| 15 | `_compress_kpool_groups_kernel()` | _compress_kpool_groups_kernel( raw_k_ptr, raw_g_ptr, ring_k_ptr, ring_g_ptr, ape_ptr, ring_slots_ptr, token_to_req_ptr, query_start_loc_ptr, positi... |
| 118 | `kpool_compress_store()` | kpool_compress_store( k |

## `freetoken/kernel/triton/mhc.py`  (347 行)

| 行 | 符号 | 说明 |
|---:|---|---|
| 22 | `_mhc_stage1_kernel()` | Split-K stage of the fused mHC: each program owns one hidden slice of one |
| 82 | `_mhc_gates()` | Reduce token ``t``'s split partials and run the tiny gate math (sigmoid gates, |
| 147 | `_mhc_stage2_kernel()` | Gate math alone (the two-pass variant, whose stage 3 consumes these pre gates). |
| 163 | `_mhc_stage3_kernel()` | layer_input = sum_n pre_n * res_new_n, parallel over hidden chunks. |
| 182 | `_mhc_stage23_kernel()` | Single-pass mHC: the layer input mixes with the PREVIOUS sublayer's pre gates, so the |
| 213 | `_stage1_hidden_tile()` | The stage-1 hidden split. It fixes the fp32 reduction order over the hidden width, so it |
| 220 | `_mhc_fused_launch()` | The three stages behind both variants. ``pre_mix_in`` selects single-pass: the |
| 299 | `mhc_fused_post_pre_triton()` | Fused hc_post (skipped when ``post_mix is None``) + hc_pre. Three-stage |
| 323 | `mhc_fused_post_pre_single_pass_triton()` | Single-pass mHC (DeepSeek-V4.1): same fused stages, but the layer input is mixed |

## `freetoken/kernel/triton/minimax_m3_sparse.py`  (1118 行)

| 行 | 符号 | 说明 |
|---:|---|---|
| 50 | `_round_up()` | _round_up(a |
| 58 | `_compare_and_swap()` | _compare_and_swap(x, ids, flip, i |
| 83 | `_bitonic_merge()` | _bitonic_merge( x, ids, stage |
| 106 | `_index_block_score_kernel()` | _index_block_score_kernel( q_ptr,  # idx_q |
| 198 | `_topk_index_kernel()` | _topk_index_kernel( s_ptr,  # [num_heads, total_q, max_block] ti_ptr,  # [num_heads, total_q, topk] block_size |
| 292 | `_decode_index_score_kernel()` | _decode_index_score_kernel( q_ptr,  # idx_q |
| 372 | `_topk_index_partial_kernel()` | _topk_index_partial_kernel( s_ptr,  # score |
| 493 | `_topk_index_merge_kernel()` | _topk_index_merge_kernel( ts_partial_ptr,  # partial scores |
| 576 | `_gqa_sparse_fwd_kernel()` | _gqa_sparse_fwd_kernel( q_ptr,  # [total_q, num_heads, head_dim] k_ptr,  # K slab, row-flat |
| 707 | `_gqa_sparse_decode_kernel()` | _gqa_sparse_decode_kernel( q_ptr,  # [num_reqs, num_heads, head_dim] k_ptr,  # K slab, row-flat |
| 836 | `_merge_topk_attn_out_kernel()` | _merge_topk_attn_out_kernel( o_ptr,  # partials |
| 876 | `m3_index_score_prefill()` | Per-token index scores for each visible sparse block: score ``[num_idx_heads, |
| 914 | `m3_index_topk_prefill()` | Causal per-token top-k over precomputed block scores -> ``[num_idx_heads, |
| 943 | `m3_index_decode()` | Decode index block-score + top-k, both split-K (cudagraph-safe). Returns |
| 1025 | `m3_sparse_attn_prefill()` | GQA block-sparse attention over the selected blocks (prefill/extend). |
| 1061 | `m3_sparse_attn_decode()` | GQA block-sparse attention for decode (split-K over the top-k blocks + |

## `freetoken/kernel/triton/moe_align.py`  (328 行)

| 行 | 符号 | 说明 |
|---:|---|---|
| 58 | `_moe_align_small()` | _moe_align_small( topk_ids_ptr, sorted_token_ids_ptr, expert_ids_ptr, num_tokens_post_pad_ptr, cumsum_ptr,        # scratch [effective_E+1] |
| 109 | `_fill_and_count()` | _fill_and_count( topk_ids_ptr, sorted_token_ids_ptr, counts_ptr, fill_counter_ptr, numel,             # #valid flattened tokens sorted_numel,      ... |
| 139 | `_cumsum_experts()` | _cumsum_experts( counts_ptr, cumsum_ptr, num_tokens_post_pad_ptr, effective_E |
| 163 | `_fill_expert_ids()` | _fill_expert_ids( cumsum_ptr, expert_ids_ptr, num_tokens_post_pad_ptr, block_size |
| 192 | `_scatter()` | _scatter( topk_ids_ptr, sorted_token_ids_ptr, cumsum_ptr, fill_counter_ptr, numel, effective_E |
| 212 | `_div_ceil()` | _div_ceil(a |
| 216 | `moe_align_block_size()` | moe_align_block_size( topk_ids |

## `freetoken/kernel/triton/moe_router.py`  (160 行)

| 行 | 符号 | 说明 |
|---:|---|---|
| 22 | `_pdl_supported()` | _pdl_supported() -> bool |
| 27 | `_router_triton_kernel()` | _router_triton_kernel( scores_ptr, out_weights_ptr, out_indices_ptr, num_token_non_padded_ptr, M, stride_sm, stride_sn, stride_wm, stride_wk, strid... |
| 109 | `fused_topk_softmax()` | Softmax over all experts, top-k, then renormalize; ties keep the lowest expert id. |

## `freetoken/kernel/triton/moe_shared_gate.py`  (137 行)

| 行 | 符号 | 说明 |
|---:|---|---|
| 22 | `_pdl_supported()` | _pdl_supported() -> bool |
| 26 | `_reduction_warps()` | _reduction_warps(hidden_dim |
| 32 | `_gate_sigmoid_kernel()` | _gate_sigmoid_kernel( hidden_ptr, weight_ptr, gate_ptr, stride_h, HIDDEN |
| 59 | `shared_gate_sigmoid()` | Per-token ``sigmoid(hidden_states @ gate_weight)`` as fp32 [num_tokens]. |
| 80 | `_gate_mul_add_kernel()` | _gate_mul_add_kernel( routed_ptr, shared_ptr, gate_ptr, out_ptr, stride_r, stride_s, stride_o, HIDDEN |
| 110 | `shared_gate_mul_add()` | ``routed + gate[:, None] * shared`` into a fresh tensor. |

## `freetoken/kernel/triton/mxfp4_moe.py`  (332 行)

| 行 | 符号 | 说明 |
|---:|---|---|
| 15 | `_fp4_table_lut()` | _fp4_table_lut(nibble, lut_ptr) |
| 20 | `_e8m0_scale()` | _e8m0_scale(s_val) |
| 26 | `mxfp4_splitk_gemv_kernel()` | mxfp4_splitk_gemv_kernel( x_ptr, w_ptr, s_ptr, bias_ptr, expert_ids_ptr, out_ptr, lut_ptr, N, K |
| 91 | `mxfp4_splitk_reduce_kernel()` | mxfp4_splitk_reduce_kernel( partial_ptr, out_ptr, N, expert_wts_ptr, HAS_EXPERT_WTS |
| 116 | `_dequant_fp4_lut()` | _dequant_fp4_lut(nibble) |
| 129 | `mxfp4_fused_moe_kernel()` | mxfp4_fused_moe_kernel( a_ptr, b_blocks_ptr, b_scales_ptr, bias_ptr, c_ptr, topk_weights_ptr, sorted_token_ids_ptr, expert_ids_ptr, num_tokens_post... |
| 244 | `gpt_oss_swiglu_kernel()` | gpt_oss_swiglu_kernel( gate_up_ptr, out_ptr, total_routes, intermediate_size |
| 289 | `gpt_oss_routing_kernel()` | gpt_oss_routing_kernel( logits_ptr, topk_weights_ptr, topk_ids_ptr, stride_logits_t, E |

## `freetoken/kernel/triton/mxfp8_linear.py`  (252 行)

| 行 | 符号 | 说明 |
|---:|---|---|
| 32 | `mxfp8_dequant()` | Dequant: ``w[n, k] * 2**(codes[n, k//32] - 127)`` -> ``dtype``. |
| 60 | `_mxfp8_gemv_m1_splitk_kernel()` | M == 1 specialization: a plain fp32 multiply-reduce. The dot tile's 16-row |
| 106 | `_mxfp8_gemv_splitk_kernel()` | Each (pid_n, pid_k) accumulates ``kb_per`` BLOCK_K chunks of ``a[:M] @ w^T`` |
| 165 | `_splitk_reduce_kernel()` | _splitk_reduce_kernel( part_ptr, out_ptr, N, SPLIT_K |
| 182 | `_gemv()` | Small-M split-K GEMV. ``a`` [M, K] bf16 (M <= _GEMV_MAX_M); ``weight`` |
| 227 | `mxfp8_linear()` | ``y = x @ dequant(weight, scale_codes)^T``. Small batches |

## `freetoken/kernel/triton/norm.py`  (192 行)

| 行 | 符号 | 说明 |
|---:|---|---|
| 38 | `_num_warps()` | _num_warps(M |
| 44 | `_rmsnorm_kernel()` | _rmsnorm_kernel( OUT, X, W, eps |
| 80 | `_fused_add_rmsnorm_kernel()` | _fused_add_rmsnorm_kernel( X, R, W, eps |
| 115 | `_leading()` | _leading(t) |
| 123 | `_rmsnorm()` | _rmsnorm(input, weight, eps, out, gemma |
| 154 | `rmsnorm()` | rmsnorm(input, weight, eps |
| 158 | `gemma_rmsnorm()` | (1 + w)-scaled RMSNorm (Gemma/MiniMax-M3 semantics: the checkpoint stores |
| 164 | `_fused_add_rmsnorm()` | _fused_add_rmsnorm(input, residual, weight, eps, gemma |
| 182 | `fused_add_rmsnorm()` | fused_add_rmsnorm(input, residual, weight, eps |
| 186 | `gemma_fused_add_rmsnorm()` | In-place ``residual += input; input = gemma_rmsnorm(residual)`` (Gemma (1+w) |

## `freetoken/kernel/triton/nvfp4_dequant.py`  (133 行)

| 行 | 符号 | 说明 |
|---:|---|---|
| 19 | `_e2m1_lut()` | _e2m1_lut(device_index |
| 26 | `_dequant_nvfp4_kernel()` | _dequant_nvfp4_kernel( packed_ptr,      # [S, OUT, IN // 2] uint8 scale_ptr,       # [S, OUT, IN // 16] fp8-e4m3 per-block scale (checkpoint native... |
| 83 | `dequant_nvfp4()` | Dequantize the cache rows selected by ``slots`` into a compact dense tensor. |

## `freetoken/kernel/triton/nvfp4_fused_moe.py`  (333 行)

| 行 | 符号 | 说明 |
|---:|---|---|
| 41 | `_e2m1_lut()` | _e2m1_lut(device_index |
| 48 | `_decode_nvfp4_moe_kernel()` | Original LUT-gather serial-K decode GEMV. Kept as the A/B baseline; the production |
| 134 | `_decode_nvfp4_marlin_kernel()` | Marlin-style NVFP4 decode GEMV: wide int32 weight loads + deferred reduction. |
| 231 | `_prefill_nvfp4_moe_kernel()` | _prefill_nvfp4_moe_kernel( a_ptr,             # [M, K] activations packed_ptr,        # [S, N, K // 2] uint8 scale_ptr,         # [S, N, K // 16] f... |

## `freetoken/kernel/triton/nvfp4_linear.py`  (818 行)

| 行 | 符号 | 说明 |
|---:|---|---|
| 79 | `_gemm_wave()` | _gemm_wave(device |
| 110 | `_splitk_counters()` | _splitk_counters(n |
| 120 | `_nvfp4_pair_f32()` | Dequant two e2m1 codes held in an int32 at bits [3:0] and [19:16]. |
| 135 | `_split8()` | Contiguous ``[BLOCK_KW * 8]`` activations -> 8 ``[BLOCK_KW]`` vectors, a_j[w] = a[8w+j]. |
| 157 | `_nvfp4_gemv_kernel()` | _nvfp4_gemv_kernel( a_ptr,        # [K] activation (compute dtype), contiguous packed_ptr,   # [N, K // 8] int32 (8 fp4 codes per word, nibble j ->... |
| 225 | `_nvfp4_gemv_splitk_kernel()` | Split-K decode GEMV for small N: each ``(pid_n, pid_k)`` reduces ``tiles_per`` word- |
| 310 | `_gemv()` | M==1 W4A16 GEMV. ``a`` [K] compute-dtype; ``packed_i32`` logical [N, K//8] int32 |
| 379 | `_nvfp4_pair_f16()` | Dequant two e2m1 codes at int32 bits [3:0], [19:16] -> (lo, hi) fp16 = value * 2^-14. |
| 388 | `_nvfp4_gemm_kernel()` | _nvfp4_gemm_kernel( a_ptr,        # [M, K] activations packed_ptr,   # logical [N, K // 8] int32 (row- or K-major via strides) scale_ptr,    # logi... |
| 483 | `_split8_rows()` | Contiguous ``[ROWS, BLOCK_KW * 8]`` -> 8 ``[ROWS, BLOCK_KW]`` slices, a_j[r, w] = a[r, 8w+j]. |
| 501 | `_nvfp4_gemm_splitk_reduce_kernel()` | Split-K reduce for the dot GEMM. Kept as a separate launch on purpose: fusing it |
| 523 | `_pick_split_k_bm16()` | Split-K for the M <= 16 (decode-batch) grid. At these M the op is pure weight |
| 546 | `_gemm_inkernel()` | Batched-decode dot GEMM (M <= 64): weight read once, split-K keeps the SMs fed when |
| 612 | `_nvfp4_dequant_rows_kernel()` | _nvfp4_dequant_rows_kernel( packed_ptr,   # [N, K // 8] int32 scale_ptr,    # [N, K // 16] fp8-e4m3 gscale_ptr,   # [N] fp16 out_ptr,      # [N, K]... |
| 646 | `_nvfp4_dequant_cols_kernel()` | Dequant for the K-major resident layout: loads coalesce along N, the [K-tile, N-tile] |
| 692 | `_dequant_rows()` | Dequant logical ``packed_i32 [n, K//8]`` (+ scales) into preallocated ``out [n, K]``. |
| 716 | `_gemm_scratch()` | Dequant the weight to a bf16 scratch (N-chunked) and run cuBLAS. The dequant runs |
| 749 | `_gemm()` | M>1 W4A16 GEMM dispatch. Small M (batched decode, lm_head last-token batch) reads |
| 760 | `_linear_impl()` | Shared dispatch on a logical ``[N, K//8]`` int32 weight view (either storage order). |
| 779 | `nvfp4_dense_linear()` | ``y = x @ dequant(weight)^T`` on the checkpoint-native row-major layout. |
| 790 | `nvfp4_transpose_resident()` | Repack a row-major NVFP4 weight (``[N, K//2]`` uint8 + ``[N, K//16]`` fp8) into the |
| 802 | `nvfp4_dense_linear_t()` | ``y = x @ dequant(W)^T`` on the K-major resident layout from |

## `freetoken/kernel/triton/ple.py`  (95 行)

| 行 | 符号 | 说明 |
|---:|---|---|
| 27 | `_ple_gather_kernel()` | _ple_gather_kernel( table_ptr, ids_ptr, out_ptr, scale, num_rows, EMB_DIM |
| 63 | `ple_gather_rows()` | Gather ``row_ids`` from the host-resident table at ``table_ptr`` into ``out``. |

## `freetoken/kernel/triton/rope.py`  (270 行)

| 行 | 符号 | 说明 |
|---:|---|---|
| 29 | `_rope_tiled()` | _rope_tiled( Q, K, POS, CACHE, stride_qbs, stride_qh, stride_qd, stride_kbs, stride_kh, stride_kd, nnz, HEAD_Q, HEAD_K, rotary_dim, half, HAS_K |
| 94 | `apply_rope_with_cos_sin_cache_inplace()` | Drop-in for the vendored op; rotates query/key in place. |
| 147 | `_mrope_tiled()` | _mrope_tiled( Q, K, POS, CACHE, SEC, stride_qbs, stride_qh, stride_qd, stride_kbs, stride_kh, stride_kd, nnz, HEAD_Q, HEAD_K, rotary_dim, half, BLO... |
| 200 | `apply_mrope_with_cos_sin_cache_inplace()` | 3-axis rope: positions is [3, nnz], section_table [rotary_dim//2] maps each frequency slot to its axis row; NeoX layout only. |
| 244 | `apply_mrope_torch_fallback()` | Pure-torch mrope (Triton unavailable); same in-place contract as the kernel. |

## `freetoken/kernel/triton/sampling.py`  (650 行)

| 行 | 符号 | 说明 |
|---:|---|---|
| 42 | `_num_sm()` | _num_sm(device) |
| 46 | `_plan()` | Return (G, CHUNK): split each row into G column-chunks of size CHUNK. |
| 55 | `_next_pow2()` | _next_pow2(x) |
| 72 | `_sm_partial()` | _sm_partial( logits_ptr, pm_ptr, pl_ptr, temp_ptr, temp_scalar, HAS_TEMP |
| 103 | `_sm_finalize()` | _sm_finalize( logits_ptr, probs_ptr, pm_ptr, pl_ptr, temp_ptr, temp_scalar, HAS_TEMP |
| 134 | `softmax()` | softmax(logits, temperature=None, enable_pdl=None) |
| 165 | `top_p_renorm_probs()` | top_p_renorm_probs(probs, top_p) |
| 175 | `_draw_part()` | _draw_part(probs_ptr, thr_ptr, psum_ptr, V, G, CHUNK, row_stride, BLOCK_SIZE |
| 192 | `_draw_scan()` | _draw_scan(psum_ptr, choff_ptr, u_ptr, target_ptr, last_ptr, G, G_POW2 |
| 204 | `_draw_find()` | _draw_find(probs_ptr, thr_ptr, choff_ptr, target_ptr, psum_ptr, last_ptr, out_ptr, V, G, CHUNK, row_stride, BLOCK_SIZE |
| 238 | `_gen_u()` | _gen_u(B, device, seed, offset) |
| 252 | `_draw()` | _draw(probs, thr, seed, offset) |
| 271 | `_zeros_thr()` | _zeros_thr(B, dev) |
| 275 | `sampling_from_probs()` | sampling_from_probs(probs, indices=None, deterministic=True, generator=None, check_nan=False, seed=None, offset=None, return_valid=False) |
| 284 | `top_p_sampling_from_probs()` | top_p_sampling_from_probs(probs, top_p, indices=None, deterministic=True, generator=None, check_nan=False, seed=None, offset=None, return_valid=False) |
| 305 | `_topk_target()` | _topk_target(top_k, B, dev) |
| 312 | `_row_barrier()` | _row_barrier(bar_ptr, need) |
| 323 | `_bits_round()` | _bits_round( probs_ptr, base, start, end, hist_ptr, bar_ptr, target, lo, above, need, S |
| 357 | `_topk_fused()` | _topk_fused( probs_ptr, target_ptr, hist_ptr, bar_ptr, ksum_ptr, psum_ptr, u_ptr, out_ptr, tok_ptr, V, G, CHUNK, row_stride, DRAW |
| 382 | `_keep_tail()` | _keep_tail( probs_ptr, base, start, end, pid, row, cta, thr, bar_ptr, need, ksum_ptr, psum_ptr, u_ptr, out_ptr, tok_ptr, V, G, DRAW |
| 438 | `_pmass_round()` | _pmass_round( probs_ptr, base, start, end, priv_ptr, mass_ptr, bar_ptr, target, lo, above, need, S |
| 471 | `_topp_fused()` | _topp_fused( probs_ptr, tp_ptr, tk_ptr, hist_ptr, priv_ptr, mass_ptr, bar_ptr, ksumk_ptr, ksum_ptr, psum_ptr, u_ptr, out_ptr, tok_ptr, V, G, CHUNK,... |
| 528 | `_fused_plan()` | _fused_plan(B, V, device, force_single=False) |
| 538 | `_fused_launch()` | _fused_launch(probs, kernel, tk, tp, draw, seed, offset, force_single=False) |
| 575 | `_cooperative_key()` | _cooperative_key(probs, kernel, tk, draw) |
| 580 | `_is_cooperative_launch_error()` | _is_cooperative_launch_error(exc) |
| 585 | `_exact_launch()` | _exact_launch(probs, kernel, tk, tp, draw, seed, offset) |
| 600 | `_topk()` | _topk(probs, target, draw, seed=None, offset=None) |
| 607 | `_topp()` | _topp(probs, tp, tk, draw, seed=None, offset=None) |
| 614 | `_topp_target()` | _topp_target(top_p, B, dev) |
| 620 | `top_k_renorm_probs()` | top_k_renorm_probs(probs, top_k) |
| 625 | `top_k_sampling_from_probs()` | top_k_sampling_from_probs(probs, top_k, indices=None, deterministic=True, generator=None, check_nan=False, seed=None, offset=None, return_valid=False) |
| 634 | `top_k_top_p_sampling_from_probs()` | top_k_top_p_sampling_from_probs(probs, top_k, top_p, indices=None, filter_apply_order="top_k_first", deterministic=True, generator=None, check_nan=... |

## `freetoken/kernel/triton/dsv4/bf16_linear.py`  (70 行)

| 行 | 符号 | 说明 |
|---:|---|---|
| 23 | `_bf16_gemv_fp32_kernel()` | _bf16_gemv_fp32_kernel( x_ptr, w_ptr, out_ptr, N, K, stride_wn, stride_wk, BLOCK_N |
| 43 | `bf16_linear_fp32()` | ``out = x @ weight.T`` in fp32, reading ``weight`` (bf16) straight from HBM. |

## `freetoken/kernel/triton/dsv4/compress.py`  (67 行)

| 行 | 符号 | 说明 |
|---:|---|---|
| 21 | `_gated_pool_kernel()` | _gated_pool_kernel( kv_ptr, score_ptr, out_ptr, R, D, stride_kb, stride_kr, stride_kd, stride_sb, stride_sr, stride_sd, stride_ob, stride_od, BLOCK_D |
| 48 | `gated_pool()` | ``out[b,d] = sum_r kv[b,r,d] * softmax(score[b,:,d], dim=row)[r]``. |

## `freetoken/kernel/triton/dsv4/fp8_linear.py`  (519 行)

| 行 | 符号 | 说明 |
|---:|---|---|
| 47 | `_log2_ceil()` | Exact ceil(log2(v)) for v > 0 via IEEE-754 bit ops (matches fast_log2_ceil). |
| 56 | `_act_quant_fp8_kernel()` | Per-row, per-``BLOCK`` FP8 quant with ue8m0 (pow2) scale. ``s`` holds e8m0 codes. |
| 88 | `act_quant_fp8()` | Reference ``act_quant`` (ue8m0): returns ``(x_fp8 [M,K], scale_codes [M,K//block])`` |
| 109 | `_act_quant_inplace_kernel()` | Fused quant+dequant round-trip (reference ``inplace=True``), written to ``o_ptr`` as |
| 145 | `_round_fp4()` | Round to nearest float4_e2m1fn value in {0,.5,1,1.5,2,3,4,6} (signed), in FP32. |
| 166 | `act_quant_fp8_inplace()` | Reference ``act_quant(x, block, ue8m0, e8m0, inplace=True)``: FP8 quant+dequant |
| 182 | `act_quant_fp8_roundtrip()` | FP8 quant+dequant round-trip into a fresh contiguous BF16 tensor (fuses the copy -- |
| 200 | `fp4_act_quant_inplace()` | Reference ``fp4_act_quant(x, block, inplace=True)``: FP4 quant+dequant round-trip |
| 220 | `_fp8_act_gemm_kernel()` | ``BLOCK_K`` is the weight's square block edge, so each K step maps to exactly one scale |
| 288 | `_fp8_gemv_kernel()` | Small-M (decode) block-scaled fp8 GEMV/GEMM: one pass over the weight at near HBM bandwidth. |
| 365 | `_splitk_reduce_kernel()` | _splitk_reduce_kernel(part_ptr, out_ptr, M, N, SPLIT_K |
| 378 | `_gemv_cfg()` | Choose whole scale tiles and enough independent programs to cover the device. |
| 393 | `_fp8_gemv()` | ``x [M, groups, K]`` bf16 x block-diagonal ``weight [groups * N, K]`` fp8 -> ``[M, groups * N]`` |
| 423 | `grouped_w8a16_gemv()` | Decode form of the reference's grouped bf16 einsum ``tgd,grd->tgr`` over a block-diagonal fp8 |
| 436 | `_dequant_block_fp8_kernel()` | _dequant_block_fp8_kernel(w_ptr, sb_ptr, out_ptr, N, K, stride_wn, stride_sbn, stride_on, SLAB |
| 450 | `dequant_block_fp8()` | ``[N, K]`` block-scaled e4m3 -> bf16 (exact), one pass; the prefill-side companion of |
| 463 | `block_fp8_linear()` | ``y = act_quant(x) @ weight^T`` (reference FP8 path). |

## `freetoken/kernel/triton/dsv4/fused_moe.py`  (326 行)

| 行 | 符号 | 说明 |
|---:|---|---|
| 37 | `_e2m1_lut()` | _e2m1_lut(device_index |
| 44 | `_decode_dsfp4_moe_kernel()` | _decode_dsfp4_moe_kernel( a_ptr,             # [M, K] activations (compute dtype) packed_ptr,        # [S, N, K // 2] uint8 scale_ptr,         # [S... |
| 129 | `_swiglu_kernel()` | Fused SwiGLU: ``out = silu(min(gate, limit)) * clamp(up, -limit, limit)``. |
| 178 | `_fp4_bf16_bits()` | E2M1 nibble [tile] + e8m0 exponent-field shift -> bf16 bit pattern |
| 192 | `_prefill_dsfp4_moe_kernel()` | _prefill_dsfp4_moe_kernel( a_ptr,             # [M, K] activations (compute dtype, FP8 round-tripped) packed_ptr,        # [S, N, K // 2] uint8 sca... |
| 296 | `_compute_type()` | _compute_type(dtype |
| 300 | `fused_swiglu()` | ``[..., 2I] -> [..., I]`` SwiGLU in a single kernel (see ``_swiglu_kernel``). ``act_block > 0`` |

## `freetoken/kernel/triton/dsv4/hadamard.py`  (44 行)

| 行 | 符号 | 说明 |
|---:|---|---|
| 18 | `_hadamard_matrix()` | Sylvester-ordered Hadamard matrix (+-1), fp32 (the ``d**-0.5`` normalization is |
| 33 | `hadamard_transform()` | Normalized Hadamard transform on the last dim (matches ``rotate_activation`` = |

## `freetoken/kernel/triton/dsv4/hc.py`  (94 行)

| 行 | 符号 | 说明 |
|---:|---|---|
| 21 | `_hc_pre_combine_kernel()` | _hc_pre_combine_kernel( x_ptr, pre_ptr, y_ptr, M, D, stride_xm, stride_xh, stride_pm, stride_ym, HC |
| 38 | `_hc_post_combine_kernel()` | _hc_post_combine_kernel( a_ptr, res_ptr, post_ptr, comb_ptr, y_ptr, M, D, stride_am, stride_rm, stride_rh, stride_pm, stride_cm, stride_ym, stride_... |
| 57 | `hc_pre_combine()` | ``y[m,d] = sum_h pre[m,h] * x[m,h,d]``. ``x``: [M,HC,D] (fp32), ``pre``: [M,HC]. |
| 72 | `hc_post_combine()` | ``y[m,h,d] = post[m,h]*a[m,d] + sum_h' comb[m,h,h']*residual[m,h',d]``. |

## `freetoken/kernel/triton/dsv4/indexer.py`  (226 行)

| 行 | 符号 | 说明 |
|---:|---|---|
| 28 | `_indexer_logits_kernel()` | _indexer_logits_kernel( q_ptr, k_ptr, w_ptr, out_ptr, S, T, stride_qb, stride_qs, stride_qh, stride_qd, stride_kb, stride_kt, stride_kd, stride_wb,... |
| 79 | `indexer_logits()` | Head-reduced Lightning-Indexer logits ``[B, S, T]`` (fp32). |
| 119 | `_indexer_decode_logits_kernel()` | One decode query per row, gathering its own compressed keys. |
| 178 | `indexer_decode_logits()` | Head-reduced Lightning-Indexer logits ``[B, n_stage]`` for a decode step. |

## `freetoken/kernel/triton/dsv4/norm.py`  (55 行)

| 行 | 符号 | 说明 |
|---:|---|---|
| 19 | `_rmsnorm_kernel()` | _rmsnorm_kernel( x_ptr, w_ptr, out_ptr, M, D, eps, stride_xm, stride_om, BLOCK_D |
| 37 | `rms_norm()` | ``y = x * rsqrt(mean(x^2, -1) + eps) * weight`` (weight optional), fused. |

## `freetoken/kernel/triton/dsv4/rope.py`  (87 行)

| 行 | 符号 | 说明 |
|---:|---|---|
| 23 | `_rope_decode_kernel()` | _rope_decode_kernel( x_ptr, freqs_ptr, rope_dim, stride_x_batch, stride_x_head, stride_x_dim, stride_freq_pos, stride_freq_dim, IS_INVERSE |
| 61 | `rope_decode_inplace()` | In-place interleaved RoPE on the last ``rope_dim`` of ``x``, per-row complex ``freqs_cis``. |

## `freetoken/kernel/triton/dsv4/router.py`  (83 行)

| 行 | 符号 | 说明 |
|---:|---|---|
| 24 | `_router_select_kernel()` | _router_select_kernel( scores_ptr, bias_ptr, alternate_bias_ptr, alternate_mask_ptr, w_ptr, id_ptr, E, route_scale, stride_s, stride_w, stride_id, ... |
| 61 | `router_select()` | ``scores [T, E]`` fp32 (pre-activation gate logits), ``bias [E]`` -> ``(weights [T, topk] fp32, |

## `freetoken/kernel/triton/dsv4/sinkhorn.py`  (77 行)

| 行 | 符号 | 说明 |
|---:|---|---|
| 19 | `_hc_sinkhorn_kernel()` | _hc_sinkhorn_kernel( mixes_ptr, scale_ptr, base_ptr, pre_ptr, post_ptr, comb_ptr, n, stride_mn, stride_pn, stride_pon, stride_cn, HC |
| 56 | `hc_split_sinkhorn()` | Triton drop-in for :func:`freetoken.models.deepseek_v4.ops.hc_split_sinkhorn`. |

## `freetoken/kernel/triton/dsv4/sparse_attn.py`  (393 行)

| 行 | 符号 | 说明 |
|---:|---|---|
| 52 | `_sparse_attn_paged_kernel()` | _sparse_attn_paged_kernel( q_ptr, win_ptr, cmp_ptr, o_ptr, sink_ptr, idx_ptr, cnt_ptr, scale, H, TOPK, N_WINDOW, stride_qb, stride_qm, stride_qh, s... |
| 128 | `_sparse_attn_paged_splitk_kernel()` | Stage 1: each program reduces one BLOCK_T-aligned slice of the candidate list and writes |
| 217 | `_sparse_attn_splitk_merge_kernel()` | Stage 2: log-sum-exp merge over the splits. The attention sink joins here (once, as a |
| 264 | `split_count()` | How many ways to split the candidate axis; 0 means run the single-program kernel. |
| 283 | `sparse_attn_paged()` | Paged sparse MLA attention: gather KV from the two global pools inside the kernel. |
| 354 | `_sparse_attn_paged_splitk()` | _sparse_attn_paged_splitk( q, window_pool, cmp_pool, sink, idx, cnt, o, b, m, h, d, topk, n_window, softmax_scale, has_counts, stride_nb, stride_nm... |

## `freetoken/kernel/triton/dsv4/swiglu.py`  (46 行)

| 行 | 符号 | 说明 |
|---:|---|---|
| 20 | `_swiglu_kernel()` | _swiglu_kernel(gate_ptr, up_ptr, out_ptr, N, limit, BLOCK |
| 33 | `fused_swiglu()` | ``silu(clamp(gate, max=limit)) * clamp(up, -limit, limit)`` -> ``out_dtype``. |

## `freetoken/kernel/triton/dsv41/__init__.py`  (9 行)

_（无类/函数定义，模块级代码）_

## `freetoken/kernel/triton/dsv41/indexer.py`  (176 行)

| 行 | 符号 | 说明 |
|---:|---|---|
| 44 | `_score_tile()` | Score one tile of compressed positions ``pos`` against the query (``q [BLOCK_H, D]`` bf16, |
| 64 | `_indexer_logits_packed_kernel()` | _indexer_logits_packed_kernel( q_ptr, w_ptr, pool_ptr, locs_ptr, live_ptr, cand_ptr, out_ptr, T, NC, n_locs, stride_qb, stride_qs, stride_qh, strid... |
| 114 | `indexer_logits_packed()` | Head-reduced indexer logits: ``[B, S, T]`` over the full staged range (columns below ``live`` |

## `freetoken/kernel/triton/dsv41/pack.py`  (183 行)

| 行 | 符号 | 说明 |
|---:|---|---|
| 32 | `_e4m3_codes()` | fp32 (\|q\| <= 448) -> e4m3 bit pattern, RNE. |
| 41 | `_e4m3_round()` | fp32 -> nearest e4m3 value, as fp32. |
| 50 | `_pack_rows_kernel()` | One program per input row ``m``; writes packed bytes to pool row ``idx[m]`` (or ``m``). |
| 106 | `_unpack_rows_kernel()` | _unpack_rows_kernel( pool_ptr, idx_ptr, out_ptr, M, D |
| 125 | `pack_rows()` | Quantize ``x [M, D]`` (bf16 / fp16 / fp32) into packed rows. |
| 156 | `unpack_rows()` | Gather ``pool[row_ids]`` (or every row) and dequantize to ``[M, dim]`` in ``out_dtype``. |
| 176 | `quant_roundtrip()` | ``unpack(pack(x))`` -- the value the cache will hand back; the reference's ``inplace=True`` semantics. |

## `freetoken/kernel/triton/dsv41/row_format.py`  (92 行)

| 行 | 符号 | 说明 |
|---:|---|---|
| 30 | `fp4_nibble_to_f32()` | e2m1 code (uint8, 0..15) -> fp32: bit 3 sign, bits 2-1 exponent, bit 0 mantissa; |
| 42 | `fp4_f32_to_nibble()` | fp32 value already on the e2m1 grid ({0, .5, 1, 1.5, 2, 3, 4, 6}, signed) -> e2m1 code. |
| 55 | `load_rows()` | Gather + dequantize ``[T, D]`` fp32 from a packed pool. |

## `freetoken/kernel/triton/dsv41/sparse_attn.py`  (323 行)

| 行 | 符号 | 说明 |
|---:|---|---|
| 41 | `_attend_tile()` | One online-softmax step over a ``[BLOCK_T, D]`` fp32 KV tile (bf16-exact values). |
| 56 | `_sparse_attn_packed_kernel()` | _sparse_attn_packed_kernel( q_ptr, win_ptr, cmp_ptr, o_ptr, sink_ptr, idx_ptr, cnt_ptr, scale, H, TOPK, N_WINDOW, stride_qb, stride_qm, stride_qh, ... |
| 114 | `_sparse_attn_packed_splitk_kernel()` | Stage 1: each program reduces one BLOCK_T-aligned slice of the candidate list and writes |
| 196 | `_sparse_attn_splitk_merge_kernel()` | Stage 2: log-sum-exp merge over the splits; the attention sink joins here once, as a null |
| 233 | `split_count()` | How many ways to split the candidate axis; 0 means run the single-program kernel. Decode |
| 247 | `sparse_attn_packed()` | Paged sparse MLA attention over packed KV pools; see the module docstring for the contract. |

## `freetoken/kernel/triton/dsv41/topk.py`  (265 行)

| 行 | 符号 | 说明 |
|---:|---|---|
| 44 | `_load_entries()` | One slice of a level's input as ``(monotone key, column)``; key 0 = dead (past ``limit``, |
| 59 | `_dsv41_topk_level_kernel()` | One program per (row, slice): the slice's top-k into candidate slot ``split * TOP_K``. Slot |
| 99 | `_final_tile()` | _final_tile(src_row, col_row, out_row, limit, TOP_K |
| 110 | `_dsv41_topk_final_kernel()` | One program per row over the last level's live candidates (or the scores themselves when the |
| 140 | `topk_plan()` | ``[(input width, n_splits), ...]`` of the reduction levels before the final tile; empty when |
| 159 | `dsv41_topk()` | Exact top-``k`` columns of every ``scores`` row among its first ``live[row]`` columns. |
| 203 | `_dsv41_block_scores_kernel()` | Worker ``w`` of a row scores block tiles ``w, w + N_WORKERS, ...`` below the live block count: |
| 228 | `dsv41_block_scores()` | ``(block scores [R, cdiv(T, block_size)] fp32, live blocks [R] int32)``: each live block's best |
| 249 | `dsv41_candidate_blocks()` | Level one of the Hierarchical Sparse Indexer over ``[R, T]`` scores: the ``topk_blocks`` best |

## `freetoken/kernel/triton/qsa/__init__.py`  (18 行)

_（无类/函数定义，模块级代码）_

## `freetoken/kernel/triton/qsa/attend.py`  (360 行)

| 行 | 符号 | 说明 |
|---:|---|---|
| 14 | `_qsa_sparse_paged_gqa_splitk_kernel()` | _qsa_sparse_paged_gqa_splitk_kernel( q_ptr, k_cache_ptr, v_cache_ptr, indices_ptr, block_table_ptr, token_to_req_ptr, partial_output_ptr, partial_l... |
| 184 | `_qsa_merge_splitk_kernel()` | _qsa_merge_splitk_kernel( partial_output_ptr, partial_lse_ptr, output_ptr, stride_output_row, stride_output_head, num_rows, HEAD_DIM |
| 227 | `qsa_sparse_paged_attention()` | Run sparse GQA directly over paged BF16 K/V caches. |

## `freetoken/kernel/triton/qsa/compress.py`  (331 行)

| 行 | 符号 | 说明 |
|---:|---|---|
| 19 | `_compress_qsa_groups_kernel()` | _compress_qsa_groups_kernel( raw_keys_ptr, ring_ptr, ring_slots_ptr, token_to_req_ptr, query_start_loc_ptr, logical_positions_ptr, pooled_ptr, firs... |
| 107 | `_index_norm_rope_kernel()` | _index_norm_rope_kernel( x_ptr, positions_ptr, cos_sin_ptr, weight_ptr, out_ptr, dest_rows_ptr, stride_x_row, stride_out_row, stride_cos_sin_row, n... |
| 171 | `_store_qsa_rows_kernel()` | _store_qsa_rows_kernel( cache_ptr, slots_ptr, rows_ptr, stride_cache_block, stride_cache_token, stride_rows_row, num_rows, num_blocks, PAGE_SIZE |
| 205 | `qsa_compress_groups()` | Pool each row's closing group from the pending ring and this step's raw rows. |
| 253 | `qsa_index_norm_rope()` | Zero-centered RMSNorm then partial NeoX rope on [rows, head_dim] indexer rows. |
| 299 | `qsa_store_rows()` | Scatter rows into a ``[blocks, block_size, width]`` cache at ``block * block_size + |

## `freetoken/kernel/triton/qsa/expand.py`  (131 行)

| 行 | 符号 | 说明 |
|---:|---|---|
| 14 | `_expand_qsa_indices_kernel()` | _expand_qsa_indices_kernel( block_indices_ptr, query_positions_ptr, sequence_lengths_ptr, token_to_req_ptr, output_ptr, stride_blocks_row, stride_b... |
| 85 | `expand_qsa_block_indices()` | Expand compressed blocks and compact the causal tail of the open group. |

## `freetoken/kernel/triton/qsa/score.py`  (191 行)

| 行 | 符号 | 说明 |
|---:|---|---|
| 16 | `_qsa_mqa_paged_kernel()` | _qsa_mqa_paged_kernel( q_ptr, k_cache_ptr, page_table_ptr, token_to_req_ptr, query_positions_ptr, sequence_lengths_ptr, visible_blocks_ptr, logits_... |
| 118 | `qsa_mqa_paged()` | Compute QSA scores directly from a paged compressed-key cache. |

## `freetoken/kernel/triton/qsa/topk.py`  (626 行)

| 行 | 符号 | 说明 |
|---:|---|---|
| 49 | `_monotone_key()` | fp32 -> uint32 with the float order preserved; key 0 is reserved for a dead column. |
| 56 | `_load_keys()` | _load_keys(logits_row, columns, limit) |
| 63 | `_narrow()` | One radix step: pin the digit at ``shift`` and report whether the range is settled. |
| 80 | `_resident_prefix()` | Radix-select a register-resident tile; returns the winning key range and its tie budget. |
| 101 | `_tile_ranks()` | Rank every winner of one tile, and carry the running counts past it. |
| 118 | `_compact()` | Emit one tile's winners at their global ranks; returns the running counts after it. |
| 137 | `_live_candidates()` | Candidate slots level ``LEVEL`` filled for a row: its splits past the row's visible tail |
| 154 | `_emit_candidates()` | Write a resident tile's own top-k as ranked (key, column) candidates; returns how many. |
| 175 | `_qsa_block_topk_kernel()` | _qsa_block_topk_kernel( logits_ptr, visible_ptr, out_ptr, stride_logits_row, stride_out_row, num_columns, TOP_K |
| 245 | `_qsa_topk_split_kernel()` | Phase 1: one program per (row, chunk), writing the chunk's own top-k as candidates. |
| 284 | `_reduce_tile()` | Top-k of ``count`` candidates held as one resident tile; returns how many it wrote. |
| 307 | `_qsa_topk_reduce_kernel()` | A reduction level: one program per (row, ``RESIDENT`` candidates of level ``LEVEL - 1``), |
| 364 | `_merge_tile()` | Top-k of one resident tile of candidates; returns how many winners it wrote. |
| 389 | `_qsa_topk_merge_kernel()` | Phase 2: one program per row over the candidates the last level ``LEVEL`` left behind. |
| 453 | **class `_SplitPlan`** | ``chunk`` columns per first-level program, and the program count of every level: the |
| 461 | `_split_plan()` | The split geometry of a ``columns``-wide buffer, or None to keep the one-program path. |
| 479 | `qsa_block_topk_scratch_width()` | int32 columns of scratch ``qsa_block_topk`` wants per row; 0 when it needs none. |
| 485 | `qsa_block_topk()` | Top ``out.shape[1]`` columns of every ``logits`` row below ``visible``, -1 padded. |

