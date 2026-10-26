# 符号索引：`server`

自动生成（`ast` 解析，不执行代码）。**20 个文件 / 11312 行 / 488 个符号**。

> 用途：确认某个模块里**到底有哪些**类/函数/方法及其行号，避免遗漏。
> 行为详解请看同目录下的手写模块文档。
> 重新生成：`python3 MODULE_DOCS/tools/gen_symbol_index.py`

| 文件 | 行数 | 符号数 |
|---|---:|---:|
| `freetoken/server/__init__.py` | 3 | 0 |
| `freetoken/server/access_log_filter.py` | 89 | 4 |
| `freetoken/server/accounting.py` | 157 | 9 |
| `freetoken/server/anthropic_api.py` | 655 | 25 |
| `freetoken/server/anthropic_models.py` | 178 | 16 |
| `freetoken/server/api_models.py` | 149 | 15 |
| `freetoken/server/api_server.py` | 1050 | 45 |
| `freetoken/server/args.py` | 932 | 20 |
| `freetoken/server/control_api.py` | 80 | 5 |
| `freetoken/server/function_call_parser.py` | 3756 | 150 |
| `freetoken/server/generation.py` | 827 | 35 |
| `freetoken/server/launch.py` | 228 | 6 |
| `freetoken/server/model_meta.py` | 149 | 5 |
| `freetoken/server/openai_api.py` | 710 | 30 |
| `freetoken/server/reasoning_parser.py` | 929 | 46 |
| `freetoken/server/request_logger.py` | 177 | 10 |
| `freetoken/server/request_ring.py` | 101 | 14 |
| `freetoken/server/responses_api.py` | 780 | 26 |
| `freetoken/server/stats.py` | 176 | 13 |
| `freetoken/server/supervisor.py` | 186 | 14 |

---

## `freetoken/server/__init__.py`  (3 行)

_（无类/函数定义，模块级代码）_

## `freetoken/server/access_log_filter.py`  (89 行)

| 行 | 符号 | 说明 |
|---:|---|---|
| 38 | `_debug_logging_enabled()` | True when the engine's existing verbosity switch requests debug output. |
| 49 | **class `PollingAccessLogFilter`** | ``logging.Filter`` for the ``uvicorn.access`` logger. |
| 57 | **class `filter`** | filter(self, record |
| 80 | `install_polling_access_log_filter()` | Attach :class:`PollingAccessLogFilter` to uvicorn's access logger, once. |

## `freetoken/server/accounting.py`  (157 行)

| 行 | 符号 | 说明 |
|---:|---|---|
| 22 | **class `AdmissionClosedError`** | A generation tried to enter after the prepare-stop gate was closed. |
| 26 | **class `AccountingDrainError`** | The engine could not reach a sealed accounting state within the bounded stop barrier. |
| 30 | `_wait_for_idle()` | Wait until all admitted requests have emitted a terminal reply, without an unbounded wait. |
| 42 | `prepare_stop_accounting()` | Close admission, drain/abort accepted work, and return one idempotent sealed snapshot. |
| 111 | **class `PrepareStopBody`** | class PrepareStopBody(BaseModel) |
| 117 | `_is_loopback()` | _is_loopback(host |
| 133 | `register_accounting_routes()` | register_accounting_routes(app |
| 134 | &nbsp;&nbsp;&nbsp;↳ `_admission_closed()` | _admission_closed(_ |
| 140 | &nbsp;&nbsp;&nbsp;↳ `prepare_stop()` | prepare_stop(request |

## `freetoken/server/anthropic_api.py`  (655 行)

| 行 | 符号 | 说明 |
|---:|---|---|
| 70 | `_anthropic_stop()` | (stop_reason, stop_sequence). A stop-string hit is reported as 'stop_sequence' |
| 78 | `register_anthropic_routes()` | register_anthropic_routes( app |
| 84 | &nbsp;&nbsp;&nbsp;↳ `v1_messages()` | v1_messages(req |
| 94 | &nbsp;&nbsp;&nbsp;↳ `v1_messages_count_tokens()` | v1_messages_count_tokens(req |
| 99 | &nbsp;&nbsp;&nbsp;↳ `_validation_error()` | _validation_error(request |
| 110 | `handle_anthropic_messages()` | handle_anthropic_messages( req |
| 147 | `handle_anthropic_count_tokens()` | handle_anthropic_count_tokens(req |
| 186 | `convert_anthropic_prompt()` | (messages, template_tools, parser_tools, chat_template_kwargs) — the prompt |
| 313 | `convert_anthropic_to_genspec()` | convert_anthropic_to_genspec( req |
| 342 | `_content_text()` | Plain text of an Anthropic message content (str or list of blocks). |
| 351 | `_image_part()` | _image_part(source |
| 364 | `_tool_result_parts()` | The text of a tool_result and its image blocks as image parts. |
| 386 | `anthropic_full_response()` | anthropic_full_response( result |
| 417 | `_anthropic_usage()` | Anthropic billing semantics under --enable-cache-report: input_tokens EXCLUDES |
| 431 | `anthropic_event_stream()` | Format the protocol-neutral GenEvent stream into Anthropic SSE events. |
| 443 | &nbsp;&nbsp;&nbsp;↳ `_open_text()` | _open_text() -> str |
| 451 | &nbsp;&nbsp;&nbsp;↳ `_open_thinking()` | _open_thinking() -> str |
| 459 | &nbsp;&nbsp;&nbsp;↳ `_stop_block()` | _stop_block() -> list[str] |
| 484 | &nbsp;&nbsp;&nbsp;↳ `_open_tool()` | _open_tool(name |
| 502 | &nbsp;&nbsp;&nbsp;↳ `_tool_args_delta()` | _tool_args_delta(fragment |
| 614 | `_event()` | Encode an event as a named Anthropic SSE frame: ``event: <type>\ndata: <json>``. |
| 619 | `_tool_use_id()` | _tool_use_id(name |
| 624 | `_parse_json_args()` | _parse_json_args(arguments |
| 636 | `_validation_error_message()` | First pydantic error as an Anthropic-style 'field.path: message' line. |
| 647 | `_anthropic_error_response()` | _anthropic_error_response( status_code |

## `freetoken/server/anthropic_models.py`  (178 行)

| 行 | 符号 | 说明 |
|---:|---|---|
| 17 | **class `AnthropicError`** | class AnthropicError(BaseModel) |
| 22 | **class `AnthropicErrorResponse`** | class AnthropicErrorResponse(BaseModel) |
| 27 | **class `AnthropicUsage`** | class AnthropicUsage(BaseModel) |
| 34 | **class `AnthropicContentBlock`** | A content block in a message. ``type`` is left open (str) rather than a |
| 58 | **class `AnthropicMessage`** | class AnthropicMessage(BaseModel) |
| 66 | **class `AnthropicTool`** | class AnthropicTool(BaseModel) |
| 73 | **class `validate_input_schema`** | validate_input_schema(cls, v) |
| 81 | **class `AnthropicToolChoice`** | class AnthropicToolChoice(BaseModel) |
| 86 | **class `AnthropicMessagesRequest`** | Anthropic Messages API request (https://docs.anthropic.com/en/api/messages). |
| 107 | **class `validate_max_tokens`** | validate_max_tokens(cls, v) |
| 113 | **class `AnthropicCountTokensRequest`** | Anthropic count_tokens request (POST /v1/messages/count_tokens): the input |
| 125 | **class `AnthropicCountTokensResponse`** | class AnthropicCountTokensResponse(BaseModel) |
| 129 | **class `AnthropicDelta`** | Delta payload for streaming events. |
| 145 | **class `AnthropicMessagesResponse`** | class AnthropicMessagesResponse(BaseModel) |
| 157 | **class `model_post_init`** | model_post_init(self, __context) |
| 162 | **class `AnthropicStreamEvent`** | class AnthropicStreamEvent(BaseModel) |

## `freetoken/server/api_models.py`  (149 行)

| 行 | 符号 | 说明 |
|---:|---|---|
| 9 | **class `MessageContent`** | class MessageContent(BaseModel) |
| 16 | **class `Function`** | class Function(BaseModel) |
| 22 | **class `Tool`** | class Tool(BaseModel) |
| 27 | **class `ToolChoiceFunction`** | class ToolChoiceFunction(BaseModel) |
| 31 | **class `ToolChoiceObject`** | class ToolChoiceObject(BaseModel) |
| 36 | **class `StreamOptions`** | class StreamOptions(BaseModel) |
| 40 | **class `FunctionCall`** | class FunctionCall(BaseModel) |
| 45 | **class `ToolCall`** | class ToolCall(BaseModel) |
| 52 | **class `Message`** | class Message(BaseModel) |
| 63 | **class `ChatCompletionRequest`** | class ChatCompletionRequest(BaseModel) |
| 95 | **class `_sync_max_completion_tokens`** | _sync_max_completion_tokens(self) -> "ChatCompletionRequest" |
| 101 | **class `CompletionRequest`** | class CompletionRequest(BaseModel) |
| 125 | **class `_sync_max_completion_tokens`** | _sync_max_completion_tokens(self) -> "CompletionRequest" |
| 131 | **class `ModelCard`** | class ModelCard(BaseModel) |
| 147 | **class `ModelList`** | class ModelList(BaseModel) |

## `freetoken/server/api_server.py`  (1050 行)

| 行 | 符号 | 说明 |
|---:|---|---|
| 62 | `get_global_state()` | get_global_state() -> FrontendManager |
| 68 | `_terminate_backend_workers()` | Best-effort, non-blocking teardown of the backend worker processes on an orderly stop. |
| 87 | `_exit_after_backend_death()` | _exit_after_backend_death(grace_s |
| 88 | &nbsp;&nbsp;&nbsp;↳ `_stop()` | _stop() -> None |
| 100 | `_reap_backend_workers()` | Wait out a preceding ``_terminate_backend_workers`` and SIGKILL whatever is still |
| 114 | `_unwrap_msg()` | _unwrap_msg(msg |
| 125 | **class `GenerateRequest`** | class GenerateRequest(BaseModel) |
| 132 | **class `FrontendManager`** | class FrontendManager |
| 200 | **class `__post_init__`** | __post_init__(self) -> None |
| 204 | **class `frontend_tokenizer`** | Lazily build and cache the frontend-side tokenizer used by count_tokens (see the |
| 218 | **class `warm_frontend_tokenizer`** | Build the frontend tokenizer and probe its thinking profile off-thread, |
| 226 | &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;↳ `_warm()` | _warm() -> None |
| 234 | **class `new_user`** | new_user(self) -> int |
| 246 | **class `listen`** | listen(self) |
| 262 | **class `_resolve_rebuild`** | Terminal transition for a rebuild: rebuilding -> serving \| failed. This is the ONLY |
| 294 | **class `fail_pending_rebuilds`** | Resolve every in-flight rebuild waiter as failed. Called from the supervisor thread |
| 306 | &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;↳ `_resolve_all()` | _resolve_all() -> None |
| 318 | **class `_create_listener_once`** | _create_listener_once(self) |
| 324 | **class `send_one`** | send_one(self, msg |
| 328 | **class `wait_for_ack`** | wait_for_ack(self, uid |
| 350 | **class `stream_generate`** | stream_generate(self, uid |
| 361 | **class `stream_with_cancellation`** | stream_with_cancellation(self, generator, request |
| 373 | **class `abort_user`** | abort_user(self, uid |
| 383 | **class `shutdown`** | shutdown(self) |
| 392 | `lifespan()` | lifespan(_ |
| 403 | `install_cors()` | Attach CORS headers for browser/webview clients (e.g. the desktop app). |
| 443 | `_served_model_name()` | _served_model_name() -> str \| None |
| 450 | `_record_request_middleware()` | Time every generation request into the ring for /v1/requests + /v1/stats p95. Single- |
| 483 | **class `CacheRebuildRequest`** | class CacheRebuildRequest(BaseModel) |
| 503 | `dispatch_rebuild()` | Send a cache-rebuild request to the scheduler and await its result, managing the |
| 552 | `_resolve_num_swa_pages()` | External accepts num_swa_pages OR swa_full_tokens_ratio; internally only num_swa_pages |
| 577 | `cache_rebuild()` | Trigger a runtime KV/MoE cache resize. Blocks until the scheduler reports a result |
| 629 | `_cache_limits()` | Per-pool adjustable {min, max} bounds for the desktop cache sliders, so their ranges stay |
| 664 | &nbsp;&nbsp;&nbsp;↳ `ideal()` | ideal(unit_cost |
| 700 | `_reasoning_geometry()` | The ``geometry.reasoning`` block, from the frontend tokenizer's probed |
| 721 | `cache_geometry()` | Current cache geometry for the desktop cache panel. Each pool size resolves |
| 821 | `cache_status()` | cache_status() |
| 831 | `generate()` | generate(req |
| 858 | `_install_shell_stop_handlers()` | Shell mode runs uvicorn off the main thread, where its own signal capture is a no-op, so |
| 872 | &nbsp;&nbsp;&nbsp;↳ `_flag_shutdown()` | _flag_shutdown(signum, frame) -> None |
| 886 | `_serve_and_run_shell()` | Shell mode: serve the API here, and attach the terminal client to it over the loopback. |
| 930 | `run_api_server()` | Run the frontend API server (FastAPI + uvicorn) and wire it to the tokenizer process via ZMQ. |
| 993 | &nbsp;&nbsp;&nbsp;↳ `_on_ready()` | _on_ready() -> None |
| 1002 | &nbsp;&nbsp;&nbsp;↳ `_on_failure()` | _on_failure(message |
| 1014 | &nbsp;&nbsp;&nbsp;↳ `_on_meta()` | _on_meta(meta |

## `freetoken/server/args.py`  (932 行)

| 行 | 符号 | 说明 |
|---:|---|---|
| 18 | **class `_DeprecatedAlias`** | An old flag: warns at parse time, converts the value if asked, stores it. |
| 21 | **class `__init__`** | __init__(self, *args, new_flag |
| 25 | **class `__call__`** | __call__(self, parser, namespace, values, option_string=None) |
| 30 | `_nvfp4_entry()` | The --quant-backend entry an old --nvfp4-backend value stands for; auto stands for none. |
| 38 | **class `ServerArgs`** | class ServerArgs(SchedulerConfig) |
| 76 | **class `share_tokenizer`** | share_tokenizer(self) -> bool |
| 80 | **class `zmq_frontend_addr`** | zmq_frontend_addr(self) -> str |
| 84 | **class `zmq_tokenizer_addr`** | zmq_tokenizer_addr(self) -> str |
| 92 | **class `tokenizer_create_addr`** | tokenizer_create_addr(self) -> bool |
| 96 | **class `backend_create_detokenizer_link`** | backend_create_detokenizer_link(self) -> bool |
| 100 | **class `frontend_create_tokenizer_link`** | frontend_create_tokenizer_link(self) -> bool |
| 104 | **class `distributed_addr`** | distributed_addr(self) -> str |
| 108 | `_json_object()` | _json_object(text |
| 118 | `parse_args()` | Parse command line arguments and return an EngineConfig. |
| 136 | &nbsp;&nbsp;&nbsp;↳ `_parse_quant_backend()` | _parse_quant_backend(value |
| 145 | &nbsp;&nbsp;&nbsp;↳ `_parse_moe_cache_rate()` | _parse_moe_cache_rate(value |
| 154 | &nbsp;&nbsp;&nbsp;↳ `_positive_int()` | _positive_int(value |
| 163 | &nbsp;&nbsp;&nbsp;↳ `_lazy_gpu_arg()` | _lazy_gpu_arg(value |
| 168 | &nbsp;&nbsp;&nbsp;↳ `_infer_tool_call_parser()` | _infer_tool_call_parser(model_path |
| 219 | &nbsp;&nbsp;&nbsp;↳ `_infer_reasoning_parser()` | _infer_reasoning_parser(model_path |

## `freetoken/server/control_api.py`  (80 行)

| 行 | 符号 | 说明 |
|---:|---|---|
| 16 | `build_health()` | Full-lifecycle health doc: loading -> ok -> error. |
| 52 | `register_control_routes()` | register_control_routes( app |
| 58 | &nbsp;&nbsp;&nbsp;↳ `health()` | health() |
| 64 | &nbsp;&nbsp;&nbsp;↳ `list_requests()` | list_requests(since |
| 72 | &nbsp;&nbsp;&nbsp;↳ `stats()` | stats() |

## `freetoken/server/function_call_parser.py`  (3756 行)

| 行 | 符号 | 说明 |
|---:|---|---|
| 44 | **class `_OrjsonCompat`** | class _OrjsonCompat |
| 46 | **class `loads`** | loads(value |
| 63 | `_should_forward_unknown_tool()` | _should_forward_unknown_tool(name |
| 87 | **class `ToolCallItem`** | Simple encapsulation of the parsed ToolCall result for easier usage in streaming contexts. |
| 95 | **class `StreamingParseResult`** | Result of streaming incremental parsing. |
| 98 | **class `__init__`** | __init__(self, normal_text |
| 103 | `_first_existing_pos()` | _first_existing_pos(text |
| 108 | `_find_common_prefix()` | _find_common_prefix(s1 |
| 119 | `_partial_json_loads()` | Parse incomplete or partial JSON strings commonly encountered during streaming. |
| 148 | `_is_complete_json()` | _is_complete_json(input_str |
| 156 | `_parse_loose_json_value()` | _parse_loose_json_value(value |
| 169 | `_parse_first_json_value()` | _parse_first_json_value(text |
| 182 | `_split_top_level()` | _split_top_level(text |
| 206 | `_parse_gemma_value()` | _parse_gemma_value(value |
| 230 | `_parse_gemma_call_args()` | _parse_gemma_call_args(text |
| 242 | **class `BaseFormatDetector`** | Base class providing two sets of interfaces: one-time and streaming incremental. |
| 264 | **class `__init__`** | __init__(self) |
| 288 | **class `_get_tool_indices`** | Get a mapping of tool names to their indices in the tools list. |
| 304 | **class `parse_base_json`** | parse_base_json(self, action |
| 331 | **class `detect_and_parse`** | Parses the text in one go. Returns success=True if the format matches, otherwise False. |
| 339 | **class `_ends_with_partial_token`** | Check if buffer ends with a partial bot_token. |
| 352 | **class `_get_param_config`** | Extract the parameter properties (JSON schema) for one tool. |
| 363 | **class `_convert_param_value`** | Convert parameter value based on schema type. Safe alternative to eval(). |
| 399 | **class `_schema_param_type`** | Normalized schema type for a parameter; ``missing`` when undeclared. |
| 409 | **class `_json_escape_chunk`** | JSON-string-escape a fragment of a value (escaping is per-character, so |
| 414 | **class `block_close_tokens`** | Tokens that end a tool block — used to locate text AFTER the last call |
| 419 | **class `finish_streaming`** | End-of-stream drain: return residual buffered text that should be surfaced |
| 436 | **class `parse_streaming_increment`** | Streaming incremental parsing with tool validation. |
| 660 | **class `InvokeParamStreamMixin`** | Value-level streaming for invoke/parameter block formats (qwen3_coder, |
| 683 | **class `_ps_reset`** | _ps_reset(self) -> None |
| 691 | **class `_ps_convert_value`** | _ps_convert_value(self, key |
| 696 | **class `_ps_canonical_name`** | Hook: normalize an invoke's function name before validation (identity by |
| 701 | **class `_ps_trim_leading`** | _ps_trim_leading(self, text |
| 706 | **class `_ps_trim_trailing`** | _ps_trim_trailing(self, text |
| 711 | **class `parse_streaming_increment`** | parse_streaming_increment(self, new_text |
| 721 | &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;↳ `_emit()` | _emit(fragment |
| 728 | &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;↳ `_update_prev()` | _update_prev() -> None |
| 902 | **class `finish_streaming`** | finish_streaming(self) -> str |
| 913 | **class `Qwen25Detector`** | Detector for Qwen 2.5 and Qwen 3 model function call format. |
| 930 | **class `__init__`** | Initializes the detector with necessary state variables. |
| 940 | **class `has_tool_call`** | Check if the text contains a Qwen 2.5 format tool call. |
| 944 | **class `detect_and_parse`** | One-time parsing: Detects and parses tool calls in the provided text. |
| 970 | **class `parse_streaming_increment`** | Streaming incremental parsing for Qwen 2.5 tool calls. |
| 1002 | **class `finish_streaming`** | finish_streaming(self) -> str |
| 1008 | **class `MistralDetector`** | Detector for Mistral model function call format. |
| 1023 | **class `__init__`** | Initializes the detector with necessary state variables. |
| 1033 | **class `has_tool_call`** | Check if the text contains a Mistral format tool call. |
| 1037 | **class `detect_and_parse`** | One-time parsing: Detects and parses tool calls in the provided text. |
| 1069 | **class `_extract_json_array`** | Extract the JSON array part using bracket counting to handle nested brackets. |
| 1112 | **class `Llama32Detector`** | Detector for Llama 3.2 models with json tool call format. |
| 1122 | **class `__init__`** | __init__(self) |
| 1131 | **class `has_tool_call`** | Check if the text contains a Llama 3.2 format tool call. |
| 1137 | **class `detect_and_parse`** | Parse function calls from text, handling multiple JSON objects. |
| 1177 | **class `Glm47Detector`** | Detector for GLM-4.7/GLM-4.7-Flash model function call format. |
| 1212 | **class `__init__`** | __init__(self) |
| 1233 | **class `has_tool_call`** | Check if the text contains a GLM-4.7 format tool call. |
| 1237 | **class `_parse_xml_arguments`** | Parse XML-style arguments into a dictionary. |
| 1267 | **class `detect_and_parse`** | One-time parsing: Detects and parses tool calls in the provided text. |
| 1326 | **class `_g_reset`** | _g_reset(self) -> None |
| 1333 | **class `parse_streaming_increment`** | Value-level streaming for GLM-4.7: text outside blocks streams live; |
| 1346 | &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;↳ `_emit()` | _emit(fragment |
| 1353 | &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;↳ `_update_prev()` | _update_prev() -> None |
| 1529 | **class `finish_streaming`** | finish_streaming(self) -> str |
| 1542 | **class `DeepSeekV32Detector`** | Detector for DeepSeek V3.2 model function call format using DSML |
| 1573 | **class `__init__`** | __init__(self) |
| 1619 | **class `block_close_tokens`** | block_close_tokens(self) -> tuple |
| 1622 | **class `has_tool_call`** | has_tool_call(self, text |
| 1625 | **class `_param_fragment`** | Prefix-stable arguments fragment for one closed DSML parameter: the |
| 1645 | **class `_dsml_params_to_json`** | Convert DSML parameter tuples (name, is_str, value) to a JSON arguments string. |
| 1658 | **class `detect_and_parse`** | One-time parsing for DSML format tool calls. |
| 1687 | **class `parse_streaming_increment`** | Streaming incremental parsing for DSML tool calls, modeled on vLLM's |
| 1702 | &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;↳ `_emit_args()` | _emit_args(fragment |
| 1711 | &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;↳ `_update_prev_args()` | _update_prev_args() -> None |
| 1882 | **class `finish_streaming`** | finish_streaming(self) -> str |
| 1896 | **class `Qwen3CoderDetector`** | class Qwen3CoderDetector(InvokeParamStreamMixin, BaseFormatDetector) |
| 1927 | **class `__init__`** | __init__(self) |
| 1952 | **class `has_tool_call`** | has_tool_call(self, text |
| 1956 | **class `_parse_function_call`** | Parse a single <function=name>...</function> block into a ToolCallItem. |
| 1994 | **class `_build_partial_arguments_json`** | Build the current argument JSON from a partial XML tool-call body. |
| 2036 | **class `detect_and_parse`** | detect_and_parse(self, text |
| 2061 | **class `Gemma4Detector`** | FreeToken serving adapter for Gemma4's compact tool-call format. |
| 2064 | **class `__init__`** | __init__(self) |
| 2073 | **class `has_tool_call`** | has_tool_call(self, text |
| 2076 | **class `detect_and_parse`** | detect_and_parse(self, text |
| 2099 | **class `_g4_reset`** | _g4_reset(self) -> None |
| 2106 | **class `parse_streaming_increment`** | Value-level streaming for Gemma4's compact call syntax: text outside |
| 2122 | &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;↳ `_emit()` | _emit(fragment |
| 2129 | &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;↳ `_update_prev()` | _update_prev() -> None |
| 2338 | **class `finish_streaming`** | finish_streaming(self) -> str |
| 2351 | **class `MiniMaxDetector`** | class MiniMaxDetector(InvokeParamStreamMixin, BaseFormatDetector) |
| 2363 | **class `__init__`** | __init__(self) |
| 2381 | **class `has_tool_call`** | has_tool_call(self, text |
| 2384 | **class `detect_and_parse`** | detect_and_parse(self, text |
| 2416 | **class `MiniMaxM3Detector`** | FreeToken serving adapter for MiniMax-M3 namespace-delimited XML tool calls. |
| 2449 | **class `__init__`** | __init__(self) |
| 2461 | **class `has_tool_call`** | has_tool_call(self, text |
| 2464 | **class `_partial_marker_suffix`** | Length of the LONGEST buffer suffix that is a proper prefix of the |
| 2478 | **class `_scan_elements`** | Scan ``text`` for ``NS<k>...NS</k>`` elements; returns ``(items, stray)`` |
| 2534 | **class `_typed_leaf`** | Leaf dequoting. The template renders leaf values verbatim, so string |
| 2572 | **class `_union_leaf`** | ``anyOf``/``oneOf``: try each member's strict coercion in declared |
| 2604 | **class `_nested_value`** | _nested_value(self, raw |
| 2610 | **class `_structure`** | Composite value from scanned elements, threading the JSON schema down |
| 2619 | &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;↳ `_sub_schema()` | _sub_schema(key |
| 2655 | **class `_args_from_items`** | _args_from_items( self, func_name |
| 2674 | **class `_scan_invoke_interior`** | Collect parameter elements from ``pos`` (just past an invoke opener) |
| 2725 | **class `_parse_block`** | Parse a wrapper interior starting at ``pos`` (just past ``bot_token``), |
| 2767 | **class `detect_and_parse`** | Parse EVERY wrapper block (text between and after blocks stays content). |
| 2787 | **class `parse_streaming_increment`** | parse_streaming_increment(self, new_text |
| 2883 | **class `GptOssDetector`** | FreeToken serving adapter for Harmony ``to=functions.*`` tool calls. |
| 2904 | **class `__init__`** | __init__(self) |
| 2916 | **class `has_tool_call`** | has_tool_call(self, text |
| 2919 | **class `block_close_tokens`** | block_close_tokens(self) -> tuple |
| 2923 | **class `_partial_suffix`** | _partial_suffix(text |
| 2933 | **class `_earliest`** | _earliest(text |
| 2941 | **class `detect_and_parse`** | detect_and_parse(self, text |
| 2965 | **class `parse_streaming_increment`** | parse_streaming_increment(self, new_text |
| 3062 | **class `finish_streaming`** | finish_streaming(self) -> str |
| 3071 | **class `MuseGlimmerDetector`** | Detector for Muse Glimmer's ATEM tool-call protocol. |
| 3116 | **class `__init__`** | __init__(self, header_open |
| 3149 | **class `has_tool_call`** | has_tool_call(self, text |
| 3152 | **class `block_close_tokens`** | block_close_tokens(self) -> tuple |
| 3155 | **class `_ps_canonical_name`** | The template renders a bare-name tool's recipient namespace as |
| 3169 | **class `_enter_recipient`** | _enter_recipient(self, recipient |
| 3178 | **class `_channel_boundary`** | Earliest channel boundary in ``buf``: ``(pos, kind, payload)`` with kind |
| 3197 | **class `_finalize_truncated_invoke`** | The channel (or the whole stream) ended while the ATEM machinery was |
| 3239 | **class `finalize_streaming`** | End-of-stream hook (FunctionCallParser.finalize_stream): the generation |
| 3251 | **class `_channel_residue_end`** | Length of the leading run of ATEM closing markup (closing tags plus |
| 3281 | **class `_drop_channel_prose`** | Text inside a tool channel that is not ATEM markup is discarded (only |
| 3289 | **class `_warn_dropped_prose`** | _warn_dropped_prose(self) -> None |
| 3300 | **class `_run_mixin`** | Feed ``atem_part`` through the invoke/parameter machinery; whatever the |
| 3308 | **class `parse_streaming_increment`** | parse_streaming_increment(self, new_text |
| 3457 | **class `finish_streaming`** | finish_streaming(self) -> str |
| 3484 | **class `detect_and_parse`** | One-shot parse by replaying the streaming machinery on a fresh clone, |
| 3525 | **class `DeepSeekV41Detector`** | DeepSeek-V4.1's DSML: the V3.2 grammar with every tag name led by a space and the block |
| 3535 | **class `FunctionCallParser`** | Parser for function/tool calls in model outputs. |
| 3561 | **class `__init__`** | __init__(self, tools |
| 3577 | **class `has_tool_call`** | Check if the given text contains a tool call in the format supported by this parser. |
| 3592 | **class `parse_non_stream`** | One-time parsing of the full text to extract tool calls. |
| 3625 | **class `parse_stream_chunk`** | Streaming incremental parsing of chunks of text as they arrive. |
| 3646 | **class `parse_stream_events`** | Ordered streaming parse: ``[("text", str) \| ("calls", [ToolCallItem]), ...]`` |
| 3670 | **class `recover_truncated_call`** | Best-effort parse of a tool call cut off by end-of-generation while its |
| 3686 | **class `parse_streaming_increment`** | parse_streaming_increment(self, new_text |
| 3690 | **class `supports_streaming`** | Whether the underlying detector can parse incrementally; when False the |
| 3695 | **class `args_fragments_prefix_stable`** | Whether streamed argument fragments always concatenate to a prefix of the |
| 3700 | **class `finish_stream`** | End-of-stream drain: residual buffered text that is plain content (empty |
| 3705 | **class `unstreamed_arguments`** | Best-effort complete arguments JSON for a call whose argument stream was cut |
| 3717 | **class `finalize_stream`** | End-of-stream finalize: argument fragments that CLOSE a call cut off |
| 3730 | `toolcall_opener_for()` | The configured parser's unique tool-call opening marker, or None when the format has |
| 3737 | `_coerce_tools()` | _coerce_tools(tools |
| 3741 | `_coerce_tool()` | _coerce_tool(tool |

## `freetoken/server/generation.py`  (827 行)

| 行 | 符号 | 说明 |
|---:|---|---|
| 47 | **class `GenerationError`** | A request failed before producing output (surfaced via ``UserReply.error`` — e.g. a |
| 56 | **class `__init__`** | __init__(self, message |
| 70 | **class `ReasoningDelta`** | class ReasoningDelta |
| 75 | **class `ContentDelta`** | class ContentDelta |
| 80 | **class `ToolCallStart`** | A tool call opened mid-stream: its name is known, arguments follow as |
| 93 | **class `ToolCallArgsDelta`** | An incremental fragment of the open call's arguments JSON. Fragments always |
| 103 | **class `ToolCallsDelta`** | Complete call(s). On the streaming path this closes a ToolCallStart-opened |
| 112 | **class `GenDone`** | class GenDone |
| 124 | **class `GenResult`** | class GenResult |
| 136 | **class `GenSpec`** | Protocol-neutral 'what to generate'. Each wire protocol converts its request |
| 149 | **class `parse_tools`** | parse_tools(self) -> bool |
| 162 | `resolve_sampling()` | Map a protocol's sampling fields onto the engine's neutral SamplingParams, |
| 176 | &nbsp;&nbsp;&nbsp;↳ `pick()` | pick(value, key, framework) |
| 196 | `render_messages()` | Normalize OpenAI-shaped message dicts for the chat template: flatten text |
| 203 | `_render_message()` | _render_message(message |
| 239 | `_flatten_text_parts()` | Plain string for text-only parts; a template-ready part list with {"type": "image"} entries (source kept under freetoken_ref) when images are present. |
| 266 | `_resolve_images()` | Gate and fetch a request's images; every failure is a GenerationError. |
| 276 | `split_tool_lists()` | (template_tools, parser_tools): parser sees all tools; the template sees only |
| 293 | `submit_generation()` | Enqueue one generation from a GenSpec; return its uid. Every protocol adapter |
| 313 | `count_prompt_tokens()` | Token count of an already-converted (messages, tools, chat_template_kwargs) prompt, |
| 354 | `prerender_error()` | Render ``spec``'s prompt frontend-side, returning the failure a streaming |
| 385 | `_make_reasoning_parser()` | Build a reasoning parser for this generation, or None if the server has no |
| 421 | `_split_reasoning()` | Return ``(reasoning, content)``. A no-op (``("", text)``) when no reasoning |
| 430 | `_leaked_special_tokens()` | Special-token strings to strip from output. Empty (no-op) unless the dsv4 |
| 436 | `_make_tool_parser()` | Build the tool-call parser with its turn-start state read from the prompt: |
| 448 | `_parse_tool_response()` | _parse_tool_response( text |
| 464 | `_valid_json()` | _valid_json(text |
| 478 | `with_keepalive()` | Yield events from ``events``, interspersing the KEEPALIVE sentinel whenever |
| 503 | `_record_generation()` | Log one generation request into the request ring. Every protocol adapter converges here, |
| 538 | `generate_events()` | Wraps `_generate_events_impl` to log the request with its totals, read off the terminal |
| 568 | `generate_full()` | Wraps `_generate_full_impl` to log the request with its totals; the `finally` also records |
| 591 | `_generate_events_impl()` | Protocol-neutral streaming generation. Yields semantic events (reasoning / |
| 631 | &nbsp;&nbsp;&nbsp;↳ `_close_open_call()` | _close_open_call() -> ToolCallsDelta \| None |
| 650 | &nbsp;&nbsp;&nbsp;↳ `_route_tool_text()` | _route_tool_text(piece |
| 786 | `_generate_full_impl()` | Protocol-neutral non-streaming generation: accumulate, split reasoning, parse |

## `freetoken/server/launch.py`  (228 行)

| 行 | 符号 | 说明 |
|---:|---|---|
| 18 | `_report_startup_error()` | Tell the parent WHY this worker is dying — push an ("error", reason) ack before it exits, |
| 31 | `_detach_process_group()` | Shell mode only: move this worker out of the terminal's foreground process group. |
| 49 | `_run_tokenize_worker()` | Module-level so it survives the spawn pickle; exists only to detach the group first. |
| 58 | `_run_scheduler()` | _run_scheduler(args |
| 125 | `launch_server()` | launch_server( run_shell |
| 153 | &nbsp;&nbsp;&nbsp;↳ `start_subprocess()` | start_subprocess() -> "BackendHandle" |

## `freetoken/server/model_meta.py`  (149 行)

| 行 | 符号 | 说明 |
|---:|---|---|
| 24 | `thinking_toggle_kwargs()` | ``chat_template_kwargs`` for a protocol-level thinking on/off toggle |
| 33 | `derive_think_gears()` | ``(gears, default_gear, kwargs_per_gear)`` for the /v1/cache/status |
| 93 | `effort_toggle_kwargs()` | Fold a protocol-level reasoning-effort request into the template kwargs. |
| 122 | `moe_total_experts()` | Total routed-expert slots the model has: experts per layer x MoE layers. Matches the |
| 136 | `moe_cache_size()` | The configured MoE slot-cache size, resolving ``--moe-cache-rate`` to a slot count. |

## `freetoken/server/openai_api.py`  (710 行)

| 行 | 符号 | 说明 |
|---:|---|---|
| 48 | `_thinking_type()` | The DeepSeek-wire thinking toggle, or None for absent/foreign shapes |
| 59 | `chat_request_to_genspec()` | OpenAI ChatCompletionRequest -> GenSpec (the OpenAI 'to_sampling_params'). |
| 89 | `_all_tool_dicts()` | _all_tool_dicts(tools) -> list[dict[str, Any]] |
| 93 | `_maintenance_gate()` | 503 while the engine is not serving. Distinguishes the startup "loading" phase from a |
| 109 | `register_openai_routes()` | register_openai_routes( app |
| 115 | &nbsp;&nbsp;&nbsp;↳ `v1_root()` | v1_root() |
| 119 | &nbsp;&nbsp;&nbsp;↳ `v1_chat_completions()` | v1_chat_completions(req |
| 127 | &nbsp;&nbsp;&nbsp;↳ `v1_completions()` | v1_completions(req |
| 135 | &nbsp;&nbsp;&nbsp;↳ `v1_models()` | v1_models() |
| 150 | `handle_chat_completion()` | handle_chat_completion( req |
| 240 | `stream_chat_completion_chunks()` | Format generate_events() into the OpenAI chat.completion.chunk SSE stream. |
| 388 | `handle_completion()` | handle_completion( req |
| 460 | `stream_completion_chunks()` | stream_completion_chunks(uid |
| 519 | `create_error_response()` | create_error_response( message |
| 539 | `_resolve_sampling()` | _resolve_sampling( req |
| 556 | `_tools_for_template()` | _tools_for_template(req |
| 568 | `_should_parse_tools()` | _should_parse_tools(req |
| 572 | `_tool_calls_to_openai()` | _tool_calls_to_openai(calls |
| 589 | `_tool_call_deltas()` | OpenAI tool_calls stream deltas. ``start_index`` offsets the slot index so |
| 609 | `_tool_call_id()` | _tool_call_id(name |
| 614 | `_chat_chunk()` | _chat_chunk(req |
| 624 | `_sse()` | _sse(payload |
| 628 | `_reported_cached()` | The prefix-cache hit to report; 0 unless --enable-cache-report is set. |
| 633 | `_usage()` | _usage(prompt_tokens |
| 646 | `_response_format_unsupported()` | _response_format_unsupported(response_format |
| 651 | `_completion_unsupported_reason()` | _completion_unsupported_reason(req |
| 667 | `_is_token_prompt()` | _is_token_prompt(prompt |
| 678 | `_effort_fields()` | The checkpoint's probed effort vocabulary for /v1/models, or (None, None) |
| 699 | `_served_model_name()` | _served_model_name(state |
| 703 | `_model_context_length()` | The model ceiling, not `min(ceiling, KV budget)`: a rebuild moves the latter, and agents |

## `freetoken/server/reasoning_parser.py`  (929 行)

| 行 | 符号 | 说明 |
|---:|---|---|
| 44 | `strip_special_tokens()` | Remove special-token strings that leaked through the decode. |
| 69 | `_longest_harmony_partial_suffix()` | Length of the longest suffix of ``text`` that is a *proper* prefix of any |
| 83 | **class `ReasoningParseResult`** | Result of (incremental or one-shot) reasoning extraction. |
| 90 | **class `BaseReasoningParser`** | One-shot and streaming reasoning extraction. |
| 110 | **class `__init__`** | __init__( self, think_start_token |
| 128 | **class `detect_and_parse`** | One-shot parse of a complete completion. |
| 156 | **class `parse_streaming_increment`** | Incremental parse. |
| 217 | **class `_split_trailing_partial`** | Split off the longest suffix of ``text`` that is a *proper* prefix of a |
| 237 | **class `flush`** | Drain any residue left in the buffer at end-of-stream. |
| 259 | **class `DeepSeekV32ReasoningParser`** | Reasoning parser for DeepSeek-V4 and DeepSeek-V3.2 (same ``<think>`` + |
| 264 | **class `__init__`** | __init__( self, force_reasoning |
| 278 | **class `GptOssHarmonyReasoningParser`** | Reasoning parser for gpt-oss Harmony output. |
| 288 | **class `__init__`** | __init__(self, force_reasoning |
| 299 | **class `detect_and_parse`** | detect_and_parse(self, text |
| 307 | **class `parse_streaming_increment`** | parse_streaming_increment(self, new_text |
| 318 | **class `flush`** | flush(self) -> ReasoningParseResult |
| 328 | **class `_segment_end`** | Return ``(end_index, matched_token)`` for a segment body starting at |
| 348 | **class `_scan`** | _scan(self, text |
| 394 | **class `ThinkReasoningParser`** | Generic ``<think>``/``</think>`` reasoning parser for models that wrap their |
| 406 | **class `__init__`** | __init__(self, force_reasoning |
| 415 | **class `MiniMaxM3ReasoningParser`** | Reasoning parser for MiniMax-M3's ``<mm:think>...</mm:think>`` protocol. |
| 439 | **class `__init__`** | __init__(self, force_reasoning |
| 452 | **class `detect_and_parse`** | One-shot parse, positional and verbatim, matching the streaming path: |
| 494 | **class `parse_streaming_increment`** | parse_streaming_increment(self, new_text |
| 520 | **class `flush`** | flush(self) -> ReasoningParseResult |
| 535 | **class `GemmaThoughtReasoningParser`** | Reasoning parser for Gemma-4's thought channel: the model emits its thought, |
| 541 | **class `__init__`** | __init__(self, force_reasoning |
| 581 | `atem_marker_inside()` | True when a complete ATEM control token lies fully inside ``text[start:end]``. |
| 593 | `_longest_atem_partial_suffix()` | Length of the longest suffix of ``text`` that is a *proper* prefix of any ATEM |
| 605 | `atem_hold_len()` | Streaming hold-back: the longest suffix of ``text`` that could still grow into |
| 627 | **class `MuseGlimmerReasoningParser`** | Reasoning parser for Muse Glimmer's ATEM channel output. |
| 661 | **class `__init__`** | __init__( self, force_reasoning |
| 687 | **class `detect_and_parse`** | detect_and_parse(self, text |
| 700 | **class `parse_streaming_increment`** | parse_streaming_increment(self, new_text |
| 704 | **class `flush`** | flush(self) -> ReasoningParseResult |
| 708 | **class `_segment_body_end`** | Earliest segment-body boundary in ``buf``: ``(end, kind, payload)`` with |
| 726 | **class `_drain`** | _drain(self, *, final |
| 730 | &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;↳ `emit()` | emit(recipient |
| 738 | &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;↳ `begin_body()` | begin_body(recipient |
| 744 | &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;↳ `emit_seek()` | emit_seek(piece |
| 873 | **class `ReasoningParser`** | Wraps a reasoning detector for streaming and non-streaming use. |
| 887 | **class `__init__`** | __init__( self, reasoning_parser |
| 900 | **class `parse_non_stream`** | Return ``(reasoning_text, normal_text)`` for a complete completion. |
| 905 | **class `parse_stream_chunk`** | Return ``(reasoning_delta, normal_delta)`` for one streaming chunk. |
| 910 | **class `flush`** | Drain buffered residue at end-of-stream as ``(reasoning_delta, normal_delta)``. |
| 916 | `build_reasoning_parser()` | Construct the configured reasoning parser, or ``None`` when the server has |

## `freetoken/server/request_logger.py`  (177 行)

| 行 | 符号 | 说明 |
|---:|---|---|
| 56 | `enabled()` | enabled() -> bool |
| 60 | `_warn_once()` | _warn_once(msg |
| 67 | `init()` | Create the log directory and start the writer thread. No-op when |
| 75 | `_ensure_worker()` | Open the per-process log file (owner-only) and spawn the writer thread, |
| 106 | `_writer_loop()` | _writer_loop() -> None |
| 120 | `_shutdown()` | Best-effort flush of queued records at interpreter exit. |
| 131 | `flush()` | Block until queued records have been written. Best-effort; returns False |
| 138 | &nbsp;&nbsp;&nbsp;↳ `_waiter()` | _waiter() -> None |
| 146 | `_to_payload()` | Faithfully represent what the client sent: for a pydantic request model, |
| 155 | `log_request()` | Enqueue one record for an inbound API request. No-op unless |

## `freetoken/server/request_ring.py`  (101 行)

| 行 | 符号 | 说明 |
|---:|---|---|
| 15 | **class `RequestRecord`** | class RequestRecord |
| 29 | **class `RequestRing`** | class RequestRing |
| 30 | **class `__init__`** | __init__(self, capacity |
| 34 | **class `add`** | add(self, rec |
| 38 | **class `since`** | Return records with id >= cursor (up to limit) and the next cursor to poll with. |
| 56 | **class `p95_ms`** | p95_ms(self) -> int |
| 63 | **class `ttft_mean_ms`** | Mean TTFT over the records that have one. |
| 70 | **class `count`** | count(self) -> int |
| 78 | `record_request()` | record_request(rec |
| 82 | `requests_since()` | requests_since(cursor |
| 86 | `requests_p95_ms()` | requests_p95_ms() -> int |
| 90 | `requests_ttft_mean_ms()` | requests_ttft_mean_ms() -> int |
| 94 | `requests_count()` | requests_count() -> int |
| 98 | `reset()` | Test helper: clear the singleton ring. |

## `freetoken/server/responses_api.py`  (780 行)

| 行 | 符号 | 说明 |
|---:|---|---|
| 87 | **class `ResponsesRequest`** | The subset of the Responses request FreeToken honors (extra fields allowed). |
| 111 | `register_responses_routes()` | register_responses_routes( app |
| 117 | &nbsp;&nbsp;&nbsp;↳ `v1_responses()` | v1_responses(req |
| 135 | &nbsp;&nbsp;&nbsp;↳ `v1_responses_get()` | v1_responses_get(response_id |
| 139 | &nbsp;&nbsp;&nbsp;↳ `v1_responses_cancel()` | v1_responses_cancel(response_id |
| 143 | `handle_responses()` | handle_responses( req |
| 188 | `convert_responses_to_genspec()` | convert_responses_to_genspec( req |
| 257 | `_convert_input_item()` | _convert_input_item(item |
| 305 | `_merge_assistant_run()` | Coalesce one assistant turn's items (reasoning / message / function_call) |
| 334 | `_input_content()` | Like _input_text, but keeps input_image parts as template-ready image parts. |
| 357 | `_input_text()` | _input_text(content |
| 372 | `_stringify()` | _stringify(value |
| 382 | `_convert_tools()` | _convert_tools(tools |
| 407 | `build_responses_response()` | build_responses_response( result |
| 460 | `_response_obj()` | _response_obj( response_id |
| 481 | `_usage()` | _usage(prompt_tokens |
| 496 | `responses_stream_generator()` | responses_stream_generator( events |
| 505 | &nbsp;&nbsp;&nbsp;↳ `snapshot()` | snapshot(status, output, usage=None, incomplete_reason=None) |
| 526 | &nbsp;&nbsp;&nbsp;↳ `close_current()` | close_current() -> list[str] |
| 580 | &nbsp;&nbsp;&nbsp;↳ `open_function_call()` | open_function_call(name |
| 599 | &nbsp;&nbsp;&nbsp;↳ `args_delta_frame()` | args_delta_frame(fragment |
| 763 | **class `_Seq`** | class _Seq |
| 764 | **class `__init__`** | __init__(self) |
| 767 | **class `next`** | next(self) -> int |
| 772 | `_sse()` | _sse(event) -> str |
| 776 | `_error_response()` | _error_response(status_code |

## `freetoken/server/stats.py`  (176 行)

| 行 | 符号 | 说明 |
|---:|---|---|
| 13 | **class `StatsTracker`** | class StatsTracker |
| 14 | **class `__init__`** | __init__(self, window_s |
| 42 | **class `active`** | active(self) -> int |
| 46 | **class `inflight_uids`** | Stable snapshot used by prepare-stop to abort every still-admitted request. |
| 50 | **class `on_new_user`** | on_new_user(self, uid |
| 54 | **class `on_abort`** | on_abort(self, uid |
| 58 | **class `observe`** | observe(self, reply |
| 86 | **class `_rate`** | _rate(self, window |
| 97 | **class `decode_tps`** | decode_tps(self, now |
| 100 | **class `prefill_tps`** | prefill_tps(self, now |
| 104 | `derive_model_card()` | attn enum + moe bool + ctx from the model config; input_modalities is what the API accepts right now. |
| 122 | `_swa_page_size()` | _swa_page_size(config |
| 129 | `build_stats()` | Full /v1/stats doc. throughput is 0 when idle; kv/mamba/swa are null |

## `freetoken/server/supervisor.py`  (186 行)

| 行 | 符号 | 说明 |
|---:|---|---|
| 19 | `phase_slug()` | Normalize a progress-bar desc into the stable /health phase enum. Order matters: |
| 34 | **class `LoadProgress`** | class LoadProgress |
| 39 | **class `update`** | update(self, desc |
| 45 | **class `phase`** | phase(self) -> str |
| 50 | **class `BackendHandle`** | What start_subprocess hands back so run_api_server can supervise the workers it |
| 59 | **class `WorkerDied`** | class WorkerDied(Exception) |
| 60 | **class `__init__`** | __init__(self, message |
| 65 | `_first_dead()` | _first_dead(processes |
| 75 | `_as_error()` | The reason a dying worker pushed as ("error", reason), or None for any other ack — lets |
| 84 | `_drain_pending_error()` | After a worker death, poll the ack queue briefly for a final ("error", reason) ack still |
| 99 | `drain_ready()` | Block until ``expected_acks`` non-progress acks arrive, applying ("progress", …) |
| 114 | &nbsp;&nbsp;&nbsp;↳ `get()` | get(timeout |
| 142 | `run_backend_supervisor()` | Drain to readiness (watching liveness), flip the gate via ``on_ready``, then watch |
| 164 | &nbsp;&nbsp;&nbsp;↳ `_shutting_down()` | _shutting_down() -> bool |

