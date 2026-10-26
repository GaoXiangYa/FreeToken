# 符号索引：`mm`

自动生成（`ast` 解析，不执行代码）。**11 个文件 / 1014 行 / 80 个符号**。

> 用途：确认某个模块里**到底有哪些**类/函数/方法及其行号，避免遗漏。
> 行为详解请看同目录下的手写模块文档。
> 重新生成：`python3 MODULE_DOCS/tools/gen_symbol_index.py`

| 文件 | 行数 | 符号数 |
|---|---:|---:|
| `freetoken/mm/__init__.py` | 33 | 3 |
| `freetoken/mm/config.py` | 33 | 2 |
| `freetoken/mm/encoder_cache.py` | 87 | 10 |
| `freetoken/mm/media.py` | 112 | 5 |
| `freetoken/mm/processor.py` | 215 | 17 |
| `freetoken/mm/processors/deepseek_v41.py` | 88 | 6 |
| `freetoken/mm/processors/gemma4.py` | 104 | 10 |
| `freetoken/mm/processors/glm5_next.py` | 76 | 6 |
| `freetoken/mm/processors/minimax_m3.py` | 92 | 7 |
| `freetoken/mm/processors/muse_glimmer.py` | 79 | 6 |
| `freetoken/mm/processors/qwen_vl.py` | 95 | 8 |

---

## `freetoken/mm/__init__.py`  (33 行)

| 行 | 符号 | 说明 |
|---:|---|---|
| 12 | `restore_placeholder()` | input_ids with every content pad id put back to the family's placeholder token. |
| 17 | `mm_pad_value()` | mm_pad_value(item_hash |
| 21 | `check_mm_pad_shift()` | check_mm_pad_shift(vocab_size |

## `freetoken/mm/config.py`  (33 行)

| 行 | 符号 | 说明 |
|---:|---|---|
| 15 | **class `MultimodalConfig`** | class MultimodalConfig |
| 29 | **class `text_model_only`** | text_model_only(self) -> bool |

## `freetoken/mm/encoder_cache.py`  (87 行)

| 行 | 符号 | 说明 |
|---:|---|---|
| 16 | **class `_Entry`** | class _Entry |
| 22 | **class `EncoderCache`** | class EncoderCache |
| 23 | **class `__init__`** | __init__(self, storage |
| 28 | **class `register`** | Claim rows of the image for uid before any of its chunks run; a repeated image adds up. |
| 35 | **class `has`** | has(self, item_hash |
| 39 | **class `put`** | put(self, item_hash |
| 54 | **class `get_slice`** | get_slice(self, item_hash |
| 60 | **class `consume`** | Account rows uid gathered; the last row of the last holder frees the entry. |
| 72 | **class `release`** | Drop uid's claims whatever it gathered (abort); entries nobody else holds are freed. |
| 82 | **class `stats`** | stats(self) -> tuple[int, int] |

## `freetoken/mm/media.py`  (112 行)

| 行 | 符号 | 说明 |
|---:|---|---|
| 17 | `collect_image_refs()` | Pop the image refs out of rendered messages in prompt order; the bare {"type": "image"} parts stay for the chat template. |
| 32 | `image_reject_reason()` | None when this server can accept image inputs, else the client-facing reason. |
| 43 | `_check_media_domain()` | Reject a URL whose hostname is not in --allowed-media-domains; empty allowlist admits any domain. |
| 59 | `_load_local_media()` | Read a file:// ref; the resolved path must be a strict subpath of --allowed-local-media-path. |
| 80 | `fetch_image_bytes()` | Resolve collected image refs (URLs / base64) to raw bytes, in order. |

## `freetoken/mm/processor.py`  (215 行)

| 行 | 符号 | 说明 |
|---:|---|---|
| 21 | **class `PromptReplacement`** | The token sequence that replaces one placeholder; is_embed marks the positions that take image embeddings (None = all of them). |
| 28 | **class `select_token_id`** | select_token_id(cls, full |
| 31 | **class `embed_spans`** | Half-open [lo, hi) runs of embedding positions, relative to the start of full. |
| 46 | **class `MMResult`** | class MMResult |
| 53 | `_find_all()` | Start index of every non-overlapping occurrence of target. |
| 67 | **class `MMProcessor`** | How a checkpoint's images become items, tokens and positions; apply is the family-agnostic driver, subclasses fill the hooks. |
| 77 | **class `__init__`** | __init__(self, model_path |
| 83 | **class `_image_processor`** | The checkpoint's image processor, loaded on first use (tokenizer workers share the instance across threads). |
| 92 | **class `get_mm_processor_kwargs`** | Keyword arguments for the checkpoint processor call: the family's reading of the mm knobs, then mm.processor_kwargs on top; families with a token b... |
| 97 | **class `process`** | One MMItem per image with feature and hash; offsets are assigned by apply. |
| 101 | **class `prompt_replacement`** | The token sequence that stands in for this item's placeholder in the prompt. |
| 104 | **class `positions`** | [3, length] t/h/w rope positions and the decode delta; None for 1-D rope families. |
| 109 | **class `dummy_items`** | The smallest encodable item of each modality the family serves, for warming the encoders. |
| 112 | **class `apply`** | Swap each image placeholder for its replacement sequence and precompute rope positions. |
| 149 | `image_positions()` | t/h/w positions: text advances all three rows, an image freezes t and spreads h/w over its grid, then the running position jumps by max(h, w). |
| 179 | `content_hash()` | 62-bit hash of the wire bytes plus family extras (e.g. the grid, so transposed images differ). |
| 189 | `get_mm_processor()` | A processor for the checkpoint under the mm knobs, or None when no encoder is left to serve: the family registers none, the checkpoint ships none, ... |

## `freetoken/mm/processors/deepseek_v41.py`  (88 行)

| 行 | 符号 | 说明 |
|---:|---|---|
| 17 | `image_grid()` | image_grid(width, height, patch, ratio, min_pixels, max_tokens, max_wh_ratio=None) |
| 38 | **class `DeepseekV41MMProcessor`** | class DeepseekV41MMProcessor(MMProcessor) |
| 39 | **class `__init__`** | __init__(self, hf_config, model_path, mm) |
| 60 | **class `process`** | process(self, images) |
| 78 | **class `prompt_replacement`** | prompt_replacement(self, item) |
| 85 | **class `dummy_items`** | dummy_items(self, dtype, device) |

## `freetoken/mm/processors/gemma4.py`  (104 行)

| 行 | 符号 | 说明 |
|---:|---|---|
| 19 | `_soft_token_budget()` | Largest accepted budget within max_tokens. |
| 24 | **class `Gemma4MMProcessor`** | class Gemma4MMProcessor(MMProcessor) |
| 25 | **class `__init__`** | __init__(self, hf_config |
| 37 | **class `get_mm_processor_kwargs`** | get_mm_processor_kwargs(self, mm |
| 44 | **class `process`** | process(self, images |
| 68 | **class `prompt_replacement`** | prompt_replacement(self, item |
| 72 | **class `dummy_items`** | dummy_items(self, dtype |
| 86 | **class `Gemma4UnifiedMMProcessor`** | The gemma4_unified release: one 48x48 super-patch per soft token and no pooling; the processor call and its budgets are the same. |
| 89 | **class `__init__`** | __init__(self, hf_config |
| 93 | **class `dummy_items`** | dummy_items(self, dtype |

## `freetoken/mm/processors/glm5_next.py`  (76 行)

| 行 | 符号 | 说明 |
|---:|---|---|
| 16 | **class `Glm5NextMMProcessor`** | class Glm5NextMMProcessor(MMProcessor) |
| 17 | **class `__init__`** | __init__(self, hf_config |
| 26 | **class `get_mm_processor_kwargs`** | get_mm_processor_kwargs(self, mm |
| 35 | **class `process`** | process(self, images |
| 60 | **class `prompt_replacement`** | prompt_replacement(self, item |
| 64 | **class `dummy_items`** | dummy_items(self, dtype |

## `freetoken/mm/processors/minimax_m3.py`  (92 行)

| 行 | 符号 | 说明 |
|---:|---|---|
| 20 | `_compression()` | The native vision config carries the merge sizes flat; the checkpoint's own config nests them under img_token_compression_config. |
| 26 | **class `MiniMaxM3MMProcessor`** | class MiniMaxM3MMProcessor(MMProcessor) |
| 27 | **class `__init__`** | __init__(self, hf_config |
| 38 | **class `get_mm_processor_kwargs`** | get_mm_processor_kwargs(self, mm |
| 50 | **class `process`** | process(self, images |
| 75 | **class `prompt_replacement`** | prompt_replacement(self, item |
| 80 | **class `dummy_items`** | dummy_items(self, dtype |

## `freetoken/mm/processors/muse_glimmer.py`  (79 行)

| 行 | 符号 | 说明 |
|---:|---|---|
| 20 | **class `MuseGlimmerMMProcessor`** | class MuseGlimmerMMProcessor(MMProcessor) |
| 21 | **class `__init__`** | __init__(self, hf_config |
| 30 | **class `get_mm_processor_kwargs`** | get_mm_processor_kwargs(self, mm |
| 37 | **class `process`** | process(self, images |
| 62 | **class `prompt_replacement`** | prompt_replacement(self, item |
| 67 | **class `dummy_items`** | dummy_items(self, dtype |

## `freetoken/mm/processors/qwen_vl.py`  (95 行)

| 行 | 符号 | 说明 |
|---:|---|---|
| 16 | **class `QwenVLMMProcessor`** | class QwenVLMMProcessor(MMProcessor) |
| 17 | **class `__init__`** | __init__(self, hf_config |
| 27 | **class `get_mm_processor_kwargs`** | get_mm_processor_kwargs(self, mm |
| 39 | **class `process`** | process(self, images |
| 64 | **class `prompt_replacement`** | prompt_replacement(self, item |
| 69 | **class `image_grid`** | image_grid(self, item |
| 73 | **class `positions`** | positions(self, length |
| 83 | **class `dummy_items`** | dummy_items(self, dtype |

