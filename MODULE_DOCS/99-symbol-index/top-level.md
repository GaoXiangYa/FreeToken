# 符号索引：`(top-level)`

自动生成（`ast` 解析，不执行代码）。**10 个文件 / 2513 行 / 156 个符号**。

> 用途：确认某个模块里**到底有哪些**类/函数/方法及其行号，避免遗漏。
> 行为详解请看同目录下的手写模块文档。
> 重新生成：`python3 MODULE_DOCS/tools/gen_symbol_index.py`

| 文件 | 行数 | 符号数 |
|---|---:|---:|
| `freetoken/__init__.py` | 5 | 0 |
| `freetoken/__main__.py` | 5 | 0 |
| `freetoken/cache_report.py` | 218 | 14 |
| `freetoken/cli.py` | 130 | 10 |
| `freetoken/control_cli.py` | 481 | 22 |
| `freetoken/core.py` | 220 | 22 |
| `freetoken/env.py` | 94 | 11 |
| `freetoken/gpu_select.py` | 287 | 17 |
| `freetoken/launch.py` | 1072 | 60 |
| `freetoken/version.py` | 1 | 0 |

---

## `freetoken/__init__.py`  (5 行)

_（无类/函数定义，模块级代码）_

## `freetoken/__main__.py`  (5 行)

_（无类/函数定义，模块级代码）_

## `freetoken/cache_report.py`  (218 行)

| 行 | 符号 | 说明 |
|---:|---|---|
| 20 | `_int()` | Read an int out of a server document defensively: a missing key, a null, and a 0 all |
| 30 | **class `CachePools`** | Which cache pools the served model actually has, read off its geometry. |
| 43 | **class `targets`** | targets(self) -> Tuple[str, ...] |
| 53 | **class `from_geometry`** | from_geometry(cls, geometry |
| 64 | `format_percent()` | format_percent(rate |
| 71 | `format_bytes()` | format_bytes(num_bytes |
| 77 | `format_tokens()` | ``<tokens> tok`` for a paged pool, with the page arithmetic spelled out -- a rebuild |
| 84 | `cache_rate()` | MoE residency: cached slots / the model's total routed experts (experts per layer x MoE |
| 94 | `pages_for_tokens()` | Round a token count up to whole pages. Pools are allocated in pages, so asking for 1000 |
| 103 | `pool_bytes()` | VRAM a pool holds: its unit count x the engine's per-unit cost. |
| 136 | `format_range()` | ``min..max`` a rebuild accepts for a pool, in the same unit the size column shows. |
| 148 | `cache_status_rows()` | ``(pool, size, allocated_bytes, resize_range)`` for every pool the model has, in report |
| 156 | &nbsp;&nbsp;&nbsp;↳ `_row()` | _row(pool |
| 177 | `format_cache_status()` | The served geometry as a table -- one row per pool the model has, carrying its size, the |

## `freetoken/cli.py`  (130 行)

| 行 | 符号 | 说明 |
|---:|---|---|
| 8 | `_print_help()` | _print_help(file |
| 27 | `_run_serve()` | _run_serve(argv |
| 34 | `_run_shell()` | _run_shell(argv |
| 40 | `_run_launch()` | _run_launch(argv |
| 46 | `_run_checkpoint()` | _run_checkpoint(argv |
| 52 | `_run_ctl()` | _run_ctl(argv |
| 58 | `_run_daemon()` | _run_daemon(argv |
| 64 | `_print_bench_help()` | _print_bench_help(file |
| 76 | `_run_bench()` | _run_bench(argv |
| 104 | `main()` | main(argv |

## `freetoken/control_cli.py`  (481 行)

| 行 | 符号 | 说明 |
|---:|---|---|
| 26 | **class `ControlCliError`** | class ControlCliError(Exception) |
| 27 | **class `__init__`** | __init__(self, message |
| 32 | `parse_count()` | parse_count(token |
| 43 | `_build_url()` | _build_url(base_url |
| 50 | `_decode_error_body()` | _decode_error_body(raw |
| 69 | `_request_json()` | _request_json( method |
| 100 | `_request_sse_text()` | _request_sse_text( method |
| 119 | `_open_request()` | _open_request( method |
| 152 | `_read_sse_text()` | _read_sse_text(response |
| 170 | `_fmt_bool()` | _fmt_bool(value |
| 176 | `_format_health()` | _format_health(doc |
| 194 | `_format_stats()` | _format_stats(doc |
| 236 | `_format_cache_status()` | The pool table (shared with the shell's ``/cache`` -- see freetoken.cache_report), plus |
| 248 | `_format_rebuild()` | _format_rebuild(doc |
| 261 | `_format_requests()` | _format_requests(doc |
| 280 | `_build_parser()` | _build_parser(prog |
| 312 | `_add_rebuild_args()` | _add_rebuild_args(parser |
| 345 | `_print_doc()` | _print_doc(doc |
| 352 | `main()` | main(argv |
| 412 | `_rebuild_body()` | Typed targets -> the rebuild body, converting the token counts to the page unit each |
| 441 | `_run_cache_rebuild()` | _run_cache_rebuild(args |
| 471 | `_cache_targets()` | The pools the user asked for, keyed by pool name, in the units they typed. |

## `freetoken/core.py`  (220 行)

| 行 | 符号 | 说明 |
|---:|---|---|
| 18 | **class `SamplingParams`** | class SamplingParams |
| 29 | **class `is_greedy`** | is_greedy(self) -> bool |
| 34 | **class `Req`** | class Req |
| 70 | **class `__post_init__`** | __post_init__(self) -> None |
| 78 | **class `_alloc_ids_buf`** | _alloc_ids_buf(self) -> None |
| 84 | **class `remain_len`** | remain_len(self) -> int |
| 88 | **class `extend_len`** | extend_len(self) -> int |
| 91 | **class `complete_one`** | complete_one(self) -> None |
| 95 | **class `append_host`** | append_host(self, next_token |
| 103 | **class `can_decode`** | can_decode(self) -> bool |
| 106 | **class `__repr__`** | __repr__(self) -> str |
| 116 | **class `Batch`** | class Batch |
| 162 | **class `is_prefill`** | is_prefill(self) -> bool |
| 166 | **class `is_decode`** | is_decode(self) -> bool |
| 169 | **class `get_attn_positions`** | get_attn_positions(self) -> torch.Tensor |
| 173 | **class `size`** | size(self) -> int |
| 177 | **class `padded_size`** | padded_size(self) -> int |
| 182 | **class `Context`** | class Context |
| 195 | **class `batch`** | batch(self) -> Batch |
| 200 | **class `forward_batch`** | forward_batch(self, batch |
| 212 | `set_global_ctx()` | set_global_ctx(ctx |
| 218 | `get_global_ctx()` | get_global_ctx() -> Context |

## `freetoken/env.py`  (94 行)

| 行 | 符号 | 说明 |
|---:|---|---|
| 8 | **class `BaseEnv`** | class BaseEnv |
| 9 | **class `_init`** | _init(self, name |
| 16 | **class `EnvVar`** | class EnvVar(BaseEnv, Generic[T]) |
| 17 | **class `__init__`** | __init__(self, default_value |
| 22 | **class `_init`** | _init(self, name |
| 30 | **class `__bool__`** | __bool__(self) |
| 33 | **class `__str__`** | __str__(self) |
| 40 | `_PARSE_MEM_BYTES()` | _PARSE_MEM_BYTES(mem |
| 59 | **class `EnvClassSingleton`** | class EnvClassSingleton |
| 79 | **class `__new__`** | __new__(cls) |
| 85 | **class `__init__`** | __init__(self) -> None |

## `freetoken/gpu_select.py`  (287 行)

| 行 | 符号 | 说明 |
|---:|---|---|
| 25 | `is_gpu_uuid()` | is_gpu_uuid(spec |
| 29 | `is_gpu_index()` | is_gpu_index(spec |
| 34 | `_canonical()` | A UUID in the exact form the driver matches (upper-case GPU- prefix), an index as-is. |
| 44 | `parse_gpu_spec()` | Split a --gpu value; ValueError on a bad entry, an empty value, or a mix of UUIDs and indices. |
| 55 | `gpu_arg()` | argparse type for a --gpu list. |
| 63 | `single_gpu_arg()` | argparse type for a single-GPU --gpu. |
| 71 | `_nvml_uuids()` | Full GPU UUIDs in physical (nvidia-smi) order, or None when NVML is unavailable. |
| 119 | `_match_uuid()` | The unique full UUID that ``spec`` prefixes, else ValueError. |
| 127 | `resolve_gpu_uuids()` | --gpu entries -> full GPU UUIDs, one per TP rank; raises ValueError on a bad entry. |
| 168 | `_preset_entry()` | The CUDA_VISIBLE_DEVICES entry ``spec`` selects, else ValueError. |
| 197 | `set_assigned_gpu()` | Publish this process's GPU before CUDA init; second call must agree. |
| 211 | `assign_gpu()` | Resolve one --gpu value and publish it for bind_assigned_gpu; no-op when the flag was not given. |
| 219 | `_visible_of_physical()` | CUDA ordinal of the physical GPU ``uuid`` (or unique prefix) among this process's visible devices. |
| 241 | `bind_assigned_gpu()` | torch.cuda.set_device this process's GPU and return the device. |
| 262 | `assigned_visible_gpu()` | Visible ordinal this process is pinned to, or None before it publishes or binds a GPU (= the current device). |
| 272 | `format_gpu_uuid()` | nvidia-smi form GPU-<uuid> from a uuid.UUID. |
| 277 | `gpu_identity()` | {index, name, uuid, total_bytes} of visible device ``index``. |

## `freetoken/launch.py`  (1072 行)

| 行 | 符号 | 说明 |
|---:|---|---|
| 52 | **class `ServerURL`** | class ServerURL |
| 58 | **class `ServedModel`** | class ServedModel |
| 68 | **class `LaunchContext`** | class LaunchContext |
| 77 | **class `CommandSpec`** | class CommandSpec |
| 84 | **class `AgentInstaller`** | class AgentInstaller |
| 130 | `_format_netloc()` | _format_netloc(host |
| 138 | `_connectable_host()` | _connectable_host(host |
| 146 | `resolve_server_url()` | resolve_server_url(server |
| 170 | `_get_json()` | _get_json(url |
| 176 | `_positive_int()` | _positive_int(value |
| 180 | `_stats_model()` | The /v1/stats model card; empty when the server predates it. |
| 190 | `_input_modalities()` | _input_modalities(stats_model |
| 196 | `discover_server_model()` | discover_server_model(server |
| 235 | `_backup_path()` | _backup_path(path |
| 239 | `_write_text_with_backup()` | _write_text_with_backup(path |
| 246 | `_write_json_with_backup()` | _write_json_with_backup(path |
| 250 | `_toml_string()` | _toml_string(value |
| 254 | `_remove_toml_section()` | _remove_toml_section(text |
| 271 | `_codex_config_path()` | _codex_config_path() -> Path |
| 275 | `_codex_profile_path()` | _codex_profile_path() -> Path |
| 279 | `_codex_catalog_path()` | _codex_catalog_path() -> Path |
| 283 | `_context_window()` | _context_window(ctx |
| 287 | `_max_output_tokens()` | Scales with the window: agents derive their compaction headroom from this. |
| 292 | `_accepts_images()` | _accepts_images(ctx |
| 296 | `_codex_catalog_entry()` | _codex_catalog_entry(model_id |
| 329 | `_ordered_model_ids()` | _ordered_model_ids(ctx |
| 337 | `_remove_toml_root_string_value()` | _remove_toml_root_string_value(text |
| 358 | `_codex_migrated_base_config_text()` | _codex_migrated_base_config_text(existing |
| 364 | `_codex_profile_text()` | _codex_profile_text(ctx |
| 378 | `prepare_codex()` | prepare_codex(ctx |
| 416 | `prepare_claude()` | prepare_claude(ctx |
| 437 | `_opencode_state_path()` | _opencode_state_path() -> Path |
| 441 | `_opencode_model_entries()` | _opencode_model_entries( model_ids |
| 457 | `_opencode_config()` | _opencode_config(ctx |
| 476 | `_load_json_object()` | _load_json_object(path |
| 488 | `_update_opencode_state()` | _update_opencode_state(path |
| 514 | `prepare_opencode()` | prepare_opencode(ctx |
| 523 | `_openclaw_config_path()` | _openclaw_config_path() -> Path |
| 527 | `_openclaw_sessions_path()` | _openclaw_sessions_path() -> Path |
| 531 | `_dict_child()` | _dict_child(parent |
| 540 | `_openclaw_model_entry()` | _openclaw_model_entry( model_id |
| 558 | `_openclaw_has_freetoken_provider()` | _openclaw_has_freetoken_provider(config |
| 568 | `_confirm_openclaw_first_patch()` | _confirm_openclaw_first_patch(ctx |
| 591 | `_patch_openclaw_config()` | _patch_openclaw_config( ctx |
| 634 | `_clear_openclaw_session_overrides()` | _clear_openclaw_session_overrides(primary_model |
| 657 | `prepare_openclaw()` | prepare_openclaw(ctx |
| 671 | `_hermes_config_path()` | _hermes_config_path() -> Path |
| 675 | `prepare_hermes()` | Nous Research's Hermes agent (hermes-agent.nousresearch.com). Its config.yaml `model:` |
| 721 | `_dsh_home()` | _dsh_home() -> Path |
| 726 | `_warn_dsh_node_version()` | _warn_dsh_node_version() -> None |
| 746 | `prepare_dsh()` | DeepSeek Harness (dsh). Like the codex profile, the wiring lives in its |
| 828 | `parse_argv()` | parse_argv(argv |
| 876 | `print_dry_run()` | print_dry_run(ctx |
| 893 | `_agent_binary_filename()` | _agent_binary_filename(binary |
| 899 | `_agent_binary_fallbacks()` | _agent_binary_fallbacks(agent |
| 920 | `resolve_agent_binary()` | resolve_agent_binary(agent |
| 934 | `_confirm_install()` | _confirm_install(installer |
| 954 | `ensure_agent_installed()` | ensure_agent_installed( agent |
| 1005 | `run_command()` | run_command(spec |
| 1013 | `main()` | main(argv |

## `freetoken/version.py`  (1 行)

_（无类/函数定义，模块级代码）_

