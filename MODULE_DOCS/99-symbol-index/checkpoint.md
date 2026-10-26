# 符号索引：`checkpoint`

自动生成（`ast` 解析，不执行代码）。**4 个文件 / 1165 行 / 50 个符号**。

> 用途：确认某个模块里**到底有哪些**类/函数/方法及其行号，避免遗漏。
> 行为详解请看同目录下的手写模块文档。
> 重新生成：`python3 MODULE_DOCS/tools/gen_symbol_index.py`

| 文件 | 行数 | 符号数 |
|---|---:|---:|
| `freetoken/checkpoint/__init__.py` | 19 | 0 |
| `freetoken/checkpoint/__main__.py` | 77 | 2 |
| `freetoken/checkpoint/convert.py` | 327 | 9 |
| `freetoken/checkpoint/ftw.py` | 742 | 39 |

---

## `freetoken/checkpoint/__init__.py`  (19 行)

_（无类/函数定义，模块级代码）_

## `freetoken/checkpoint/__main__.py`  (77 行)

| 行 | 符号 | 说明 |
|---:|---|---|
| 24 | `_parse_quant_backend()` | _parse_quant_backend(value |
| 34 | `main()` | main(argv |

## `freetoken/checkpoint/convert.py`  (327 行)

| 行 | 符号 | 说明 |
|---:|---|---|
| 35 | `_progress()` | _progress(phase |
| 40 | `_source_fingerprint()` | Identity of (checkpoint + quant + GPU capability), stored in the FTW index so |
| 69 | `_copy_metadata()` | Copy all non-weight files (config, tokenizer, remote-code, nested model configs) |
| 106 | **class `_ConvertSink`** | Layer-completion sink for ``load_expert_banks(layer_sink=...)``: writes each |
| 122 | **class `__init__`** | __init__(self, writer |
| 133 | **class `__call__`** | __call__(self, layer_id |
| 156 | **class `close`** | close(self) -> None |
| 161 | **class `num_layers`** | num_layers(self) -> int |
| 165 | `convert_checkpoint()` | Write ``model_path`` as an FTW checkpoint at ``out_dir``. Returns the index dict. |

## `freetoken/checkpoint/ftw.py`  (742 行)

| 行 | 符号 | 说明 |
|---:|---|---|
| 64 | `layer_bank_entry_name()` | Name of one per-layer ``experts_bank`` FTW entry; :func:`load_ftw_banks` groups |
| 70 | `_pread_into()` | POSIX positional read into ``mv`` at ``offset``, looping over any short preadv. |
| 89 | `_align_up()` | _align_up(n |
| 93 | `_dtype_str()` | _dtype_str(dt |
| 97 | `_dtype_of()` | _dtype_of(s |
| 101 | `_elsize()` | _elsize(dt |
| 105 | `is_ftw_checkpoint()` | True if ``path`` is a directory holding a FreeToken Weight (FTW) index. |
| 110 | `ftw_tensor_names()` | Names the FTW index lists for ``kinds`` (every kind when none is given). |
| 117 | `ftw_quant_format()` | The ``quant_format`` an FTW checkpoint's expert banks were packed for; None when ``path`` is not an FTW checkpoint or holds no banks. |
| 126 | **class `FTWWriter`** | Stream tensors into the FTW, rolling shard files at ``shard_limit``. |
| 135 | **class `__init__`** | __init__(self, out_dir |
| 148 | **class `_roll`** | _roll(self) -> None |
| 158 | **class `_write_raw`** | Write ``data`` into the FTW byte stream, splitting across shards at the limit. |
| 173 | **class `add_tensor`** | add_tensor(self, name |
| 191 | **class `finalize`** | finalize(self, meta |
| 208 | **class `FTWReader`** | Random-access reader over an FTW checkpoint. |
| 217 | **class `__init__`** | __init__(self, path |
| 236 | **class `meta`** | meta(self, key |
| 239 | **class `entries`** | entries(self, *kinds |
| 243 | **class `_ensure_mode`** | Resolve the read backend once: keep O_DIRECT if the filesystem accepts it, else |
| 262 | **class `_fd`** | _fd(self, file |
| 272 | **class `_map`** | _map(self, file |
| 291 | **class `close`** | close(self) -> None |
| 300 | **class `_pieces`** | Yield (file, file_off, dest_off, length) covering [global_off, +nbytes), |
| 320 | **class `read_into`** | Read one tensor's bytes into ``dest`` (length >= entry nbytes rounded to ALIGN). |
| 336 | &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;↳ `rd()` | rd(job) |
| 343 | &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;↳ `rd()` | rd(job) |
| 361 | `_transient_buffer()` | _transient_buffer(nbytes |
| 365 | `iter_ftw_weights()` | Yield ``(name, host_tensor)`` for the requested kinds, reading each tensor via |
| 386 | &nbsp;&nbsp;&nbsp;↳ `_put()` | _put(item) -> bool |
| 399 | &nbsp;&nbsp;&nbsp;↳ `_producer()` | _producer() |
| 435 | `_split_bank_entries()` | ``(all, alphas, flat regions, {base: {layer: entry}})`` of the ``experts_bank`` entries; None when there are none. |
| 472 | `load_ftw_banks()` | Reconstruct the offload :class:`ExpertBanks` from the FTW's ``experts_bank`` |
| 516 | &nbsp;&nbsp;&nbsp;↳ `_backing()` | _backing(layer_id |
| 585 | &nbsp;&nbsp;&nbsp;↳ `_read_alpha()` | _read_alpha(e) |
| 591 | &nbsp;&nbsp;&nbsp;↳ `_read_row()` | _read_row(job) |
| 598 | &nbsp;&nbsp;&nbsp;↳ `_read_layer()` | _read_layer(job) |
| 675 | `load_ftw_banks_to_device()` | :func:`load_ftw_banks` for resident experts: every bank lands on ``device``, one (bank, layer) at a time through a transient host buffer. |
| 704 | &nbsp;&nbsp;&nbsp;↳ `_read()` | _read(job) -> torch.Tensor |

