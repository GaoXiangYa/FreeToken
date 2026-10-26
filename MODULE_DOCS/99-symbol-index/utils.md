# 符号索引：`utils`

自动生成（`ast` 解析，不执行代码）。**9 个文件 / 967 行 / 94 个符号**。

> 用途：确认某个模块里**到底有哪些**类/函数/方法及其行号，避免遗漏。
> 行为详解请看同目录下的手写模块文档。
> 重新生成：`python3 MODULE_DOCS/tools/gen_symbol_index.py`

| 文件 | 行数 | 符号数 |
|---|---:|---:|
| `freetoken/utils/__init__.py` | 63 | 0 |
| `freetoken/utils/arch.py` | 100 | 10 |
| `freetoken/utils/hf.py` | 299 | 19 |
| `freetoken/utils/logger.py` | 129 | 9 |
| `freetoken/utils/misc.py` | 53 | 7 |
| `freetoken/utils/mp.py` | 151 | 29 |
| `freetoken/utils/progress.py` | 90 | 8 |
| `freetoken/utils/registry.py` | 45 | 8 |
| `freetoken/utils/torch_utils.py` | 37 | 4 |

---

## `freetoken/utils/__init__.py`  (63 行)

_（无类/函数定义，模块级代码）_

## `freetoken/utils/arch.py`  (100 行)

| 行 | 符号 | 说明 |
|---:|---|---|
| 12 | `_gfx_arch_from()` | _gfx_arch_from(value |
| 18 | `is_rocm()` | True when torch is built for ROCm (AMD GPU) instead of CUDA. |
| 25 | `get_rocm_gfx_arch()` | Return the current AMD GPU target (for example ``gfx1201``). |
| 58 | `_get_torch_cuda_version()` | _get_torch_cuda_version() -> Tuple[int, int] \| None |
| 69 | `is_arch_supported()` | capability >= (major, minor). Open-ended: newer archs also pass. Only use this |
| 79 | `_is_arch_family()` | _is_arch_family(major |
| 84 | `is_sm90_family()` | Exactly major 9 (Hopper). For sm_90a-only kernels (e.g. FA3). |
| 89 | `is_sm100_family()` | Exactly major 10 (datacenter Blackwell). For sm_100a/103a-only kernels |
| 95 | `is_sm90_supported()` | is_sm90_supported() -> bool |
| 99 | `is_sm100_supported()` | is_sm100_supported() -> bool |

## `freetoken/utils/hf.py`  (299 行)

| 行 | 符号 | 说明 |
|---:|---|---|
| 24 | **class `DisabledTqdm`** | class DisabledTqdm(tqdm) |
| 25 | **class `__init__`** | __init__(self, *args, **kwargs) |
| 31 | `load_tokenizer()` | load_tokenizer(model_path |
| 50 | `load_eos_token_ids()` | Return the full set of stop-token ids for generation. |
| 81 | `load_toolcall_anchor_id()` | The single token id of ``opener`` -- the wire format's unique tool-call opening |
| 98 | `load_generation_sampling()` | Recommended sampling defaults from ``generation_config.json`` (sglang's |
| 138 | **class `RawConfigShim`** | Attribute view over a checkpoint's raw ``config.json``. |
| 150 | **class `__init__`** | __init__(self, data |
| 153 | **class `__getattr__`** | __getattr__(self, name |
| 169 | **class `to_dict`** | to_dict(self) -> dict |
| 173 | `optional_hf_file()` | Local path of ``filename`` in a checkpoint dir or Hub repo; None when the checkpoint has no such file. |
| 184 | `_raw_config_json()` | _raw_config_json(model_path |
| 193 | `sidecar_quantization_config()` | The ``quantization_config`` an old ModelOpt export keeps only in ``hf_quant_config.json``, or None. |
| 205 | `_merge_sidecar_quantization_config()` | _merge_sidecar_quantization_config(config |
| 221 | `_load_hf_config()` | _load_hf_config(model_path |
| 237 | `_with_overrides()` | _with_overrides(config |
| 248 | `cached_load_hf_config()` | A fresh copy of the checkpoint's config with ``overrides`` applied as vLLM's --hf-overrides |
| 273 | `_weight_allow_patterns()` | _weight_allow_patterns(repo_id |
| 287 | `download_hf_weight()` | download_hf_weight(model_path |

## `freetoken/utils/logger.py`  (129 行)

| 行 | 符号 | 说明 |
|---:|---|---|
| 9 | `init_logger()` | Initialize the logger for the module with colors and pretty formatting. |
| 52 | **class `ColorFormatter`** | Formatter with colors and pretty output |
| 66 | **class `format`** | format(self, record) |
| 103 | &nbsp;&nbsp;&nbsp;↳ `_call_rank0()` | _call_rank0(msg, *args, _which, **kwargs) |
| 115 | **class `WrapperLogger`** | Custom logger to handle the color formatter. |
| 118 | **class `info_rank0`** | info_rank0(self, msg, *args, **kwargs) |
| 119 | **class `warning_rank0`** | warning_rank0(self, msg, *args, **kwargs) |
| 120 | **class `debug_rank0`** | debug_rank0(self, msg, *args, **kwargs) |
| 121 | **class `critical_rank0`** | critical_rank0(self, msg, *args, **kwargs) |

## `freetoken/utils/misc.py`  (53 行)

| 行 | 符号 | 说明 |
|---:|---|---|
| 4 | `call_if_main()` | Decorator to ensure a function will call when the script is run as main. |
| 20 | `div_even()` | Divides two integers. If allow_replicate=True, allows b > a when b % a == 0, returning 1. |
| 29 | `div_ceil()` | Divides two integers, rounding up |
| 34 | `align_ceil()` | Aligns a to the next multiple of b |
| 39 | `align_down()` | Aligns a to the previous multiple of b |
| 44 | `mem_GB()` | Format a byte count as a human-readable GiB string for logs. |
| 49 | **class `Unset`** | class Unset |

## `freetoken/utils/mp.py`  (151 行)

| 行 | 符号 | 说明 |
|---:|---|---|
| 12 | **class `ZmqPushQueue`** | class ZmqPushQueue(Generic[T]) |
| 13 | **class `__init__`** | __init__( self, addr |
| 24 | **class `put`** | put(self, obj |
| 28 | **class `stop`** | stop(self) |
| 33 | **class `ZmqAsyncPushQueue`** | class ZmqAsyncPushQueue(Generic[T]) |
| 34 | **class `__init__`** | __init__( self, addr |
| 45 | **class `put`** | put(self, obj |
| 49 | **class `stop`** | stop(self) |
| 54 | **class `ZmqPullQueue`** | class ZmqPullQueue(Generic[T]) |
| 55 | **class `__init__`** | __init__( self, addr |
| 66 | **class `get`** | get(self) -> T |
| 70 | **class `get_raw`** | get_raw(self) -> bytes |
| 73 | **class `decode`** | decode(self, raw |
| 76 | **class `empty`** | empty(self) -> bool |
| 79 | **class `stop`** | stop(self) |
| 84 | **class `ZmqAsyncPullQueue`** | class ZmqAsyncPullQueue(Generic[T]) |
| 85 | **class `__init__`** | __init__( self, addr |
| 96 | **class `get`** | get(self) -> T |
| 100 | **class `stop`** | stop(self) |
| 105 | **class `ZmqPubQueue`** | class ZmqPubQueue(Generic[T]) |
| 106 | **class `__init__`** | __init__( self, addr |
| 117 | **class `put_raw`** | put_raw(self, raw |
| 120 | **class `put`** | put(self, obj |
| 124 | **class `stop`** | stop(self) |
| 129 | **class `ZmqSubQueue`** | class ZmqSubQueue(Generic[T]) |
| 130 | **class `__init__`** | __init__( self, addr |
| 142 | **class `get`** | get(self) -> T |
| 146 | **class `empty`** | empty(self) -> bool |
| 149 | **class `stop`** | stop(self) |

## `freetoken/utils/progress.py`  (90 行)

| 行 | 符号 | 说明 |
|---:|---|---|
| 20 | `set_progress_sink()` | Install (or clear, with None) a global progress callback invoked (throttled) |
| 28 | `_on_primary()` | _on_primary() -> bool |
| 33 | `emit_progress()` | Push a one-off update to the installed sink for a phase that has no ``byte_bar`` — e.g. |
| 45 | **class `_SinkTqdm`** | A tqdm that also forwards its progress to the installed ``_PROGRESS_SINK``, |
| 49 | **class `__init__`** | __init__(self, *args, **kwargs) -> None |
| 54 | **class `update`** | update(self, n |
| 76 | `byte_bar()` | A byte-scaled bar (shows e.g. ``12.8GiB [00:02, 6.1GiB/s]``); ``update(nbytes)`` it |
| 84 | `count_bar()` | A plain count bar over an iterable (use when total bytes aren't known up front). |

## `freetoken/utils/registry.py`  (45 行)

| 行 | 符号 | 说明 |
|---:|---|---|
| 6 | **class `Registry`** | class Registry(Generic[T]) |
| 7 | **class `__init__`** | __init__(self, type |
| 12 | **class `register`** | register(self, name |
| 16 | &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;↳ `decorator()` | decorator(item |
| 22 | **class `info`** | info(self, name |
| 27 | **class `__getitem__`** | __getitem__(self, name |
| 32 | **class `supported_names`** | supported_names(self) -> List[str] |
| 35 | **class `assert_supported`** | assert_supported(self, names |

## `freetoken/utils/torch_utils.py`  (37 行)

| 行 | 符号 | 说明 |
|---:|---|---|
| 12 | `torch_dtype()` | torch_dtype(dtype |
| 23 | `nvtx_annotate()` | nvtx_annotate(name |
| 26 | &nbsp;&nbsp;&nbsp;↳ `decorator()` | decorator(fn) |
| 28 | &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;↳ `wrapper()` | wrapper(self, *args, **kwargs) |

