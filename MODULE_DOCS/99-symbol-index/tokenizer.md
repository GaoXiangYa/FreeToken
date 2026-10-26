# 符号索引：`tokenizer`

自动生成（`ast` 解析，不执行代码）。**6 个文件 / 1218 行 / 47 个符号**。

> 用途：确认某个模块里**到底有哪些**类/函数/方法及其行号，避免遗漏。
> 行为详解请看同目录下的手写模块文档。
> 重新生成：`python3 MODULE_DOCS/tools/gen_symbol_index.py`

| 文件 | 行数 | 符号数 |
|---|---:|---:|
| `freetoken/tokenizer/__init__.py` | 3 | 0 |
| `freetoken/tokenizer/detokenize.py` | 147 | 8 |
| `freetoken/tokenizer/effort.py` | 301 | 8 |
| `freetoken/tokenizer/inline_system.py` | 173 | 8 |
| `freetoken/tokenizer/server.py` | 265 | 7 |
| `freetoken/tokenizer/tokenize.py` | 329 | 16 |

---

## `freetoken/tokenizer/__init__.py`  (3 行)

_（无类/函数定义，模块级代码）_

## `freetoken/tokenizer/detokenize.py`  (147 行)

| 行 | 符号 | 说明 |
|---:|---|---|
| 10 | `_is_chinese_char()` | Checks whether CP is the codepoint of a CJK character. |
| 35 | `find_printable_text()` | Returns the longest printable substring of text that contains only entire words. |
| 54 | `_stop_prefix_holdback()` | Length of the longest trailing suffix of ``text`` that is a proper prefix of some |
| 68 | **class `DecodeStatus`** | class DecodeStatus |
| 76 | **class `DetokenizeManager`** | class DetokenizeManager |
| 77 | **class `__init__`** | __init__( self, tokenizer |
| 87 | **class `discard`** | Drop a uid's decode state without a finished DetokenizeMsg. An aborted or errored |
| 93 | **class `detokenize`** | detokenize(self, msgs |

## `freetoken/tokenizer/effort.py`  (301 行)

| 行 | 符号 | 说明 |
|---:|---|---|
| 46 | `effective_efforts()` | The vocabulary a checkpoint is actually served (and advertised) with. |
| 76 | **class `EffortProfile`** | What one checkpoint's template/encoder accepts, learned by probing it. |
| 100 | **class `ThinkingProfile`** | A checkpoint's thinking controls, learned by probing its template. |
| 123 | `quantize_effort()` | Map a client's effort onto ``profile``; ``None`` means "send nothing". |
| 178 | `probe_effort_profile()` | Learn a checkpoint's effort vocabulary by rendering probes through it. |
| 245 | `_accepts_integer_budget()` | _accepts_integer_budget(render, base_kwargs, tools) -> bool |
| 261 | `probe_thinking_profile()` | Learn a checkpoint's thinking-toggle behavior by rendering the broadcast |
| 297 | `_renderings_differ()` | _renderings_differ(a |

## `freetoken/tokenizer/inline_system.py`  (173 行)

| 行 | 符号 | 说明 |
|---:|---|---|
| 14 | **class `InlineSystemError`** | An input instruction cannot be placed without corrupting tool ordering. |
| 18 | `has_inline_system()` | has_inline_system(messages |
| 29 | `_add_text()` | _add_text(message |
| 32 | &nbsp;&nbsp;&nbsp;↳ `join()` | join(a, b) |
| 50 | `_after_tool_results()` | _after_tool_results(messages |
| 74 | `normalize_inline_system()` | normalize_inline_system(messages |
| 112 | `_insertion()` | _insertion(base |
| 124 | `probe_inline_system()` | Conservatively require a distinct, prefix-stable system segment. |

## `freetoken/tokenizer/server.py`  (265 行)

| 行 | 符号 | 说明 |
|---:|---|---|
| 37 | `_unwrap_msg()` | _unwrap_msg(msg |
| 43 | `_prompt_admitted_reply()` | Translate the scheduler's admission signal onto the existing frontend usage path. |
| 54 | `_error_reply()` | _error_reply(msg |
| 60 | `_put_user_replies()` | _put_user_replies(send_frontend |
| 67 | `_send_generation_replies()` | Preserve the accounting barrier within one tokenizer queue drain. |
| 84 | `_tokenize_requests()` | Tokenize independently, returning backend work plus terminal frontend errors. |
| 127 | `tokenize_worker()` | tokenize_worker( *, tokenizer_path |

## `freetoken/tokenizer/tokenize.py`  (329 行)

| 行 | 符号 | 说明 |
|---:|---|---|
| 31 | `resolve_thinking_mode()` | Resolve the thinking mode (``"thinking"`` or ``"chat"``) for a chat request. |
| 54 | **class `TokenizeManager`** | class TokenizeManager |
| 55 | **class `__init__`** | __init__(self, tokenizer |
| 66 | **class `tokenize`** | tokenize(self, msgs |
| 103 | **class `render_prompt`** | The template/encoder half of ``tokenize``, exposed so the frontend can |
| 122 | **class `inline_system_mode`** | inline_system_mode(self, tools |
| 135 | **class `_render`** | Raw render, no effort sanitation — the probe needs unsupported values |
| 168 | **class `effort_profile`** | The checkpoint's effort vocabulary, probed on first use and cached |
| 181 | **class `thinking_profile`** | The checkpoint's thinking controls (toggle behavior + effort |
| 191 | **class `_probe_render`** | _probe_render( self, kwargs |
| 196 | **class `_sanitize_effort`** | _sanitize_effort(self, chat_template_kwargs |
| 220 | `_load_dsv4_encoder_if_needed()` | _load_dsv4_encoder_if_needed(tokenizer |
| 243 | `_apply_dsv4_chat_encoder()` | _apply_dsv4_chat_encoder( encoder |
| 284 | `_dsv4_tool_calls()` | The dsv4 encoder's contract is ``function.arguments`` = JSON-object STRING |
| 301 | `_dsv4_arguments_str()` | Missing/empty means no arguments (vLLM parity); anything else that is not |
| 324 | `_attach_tools_to_dsv4_messages()` | _attach_tools_to_dsv4_messages(messages |

