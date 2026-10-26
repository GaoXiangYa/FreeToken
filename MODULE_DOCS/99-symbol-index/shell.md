# 符号索引：`shell`

自动生成（`ast` 解析，不执行代码）。**5 个文件 / 1515 行 / 92 个符号**。

> 用途：确认某个模块里**到底有哪些**类/函数/方法及其行号，避免遗漏。
> 行为详解请看同目录下的手写模块文档。
> 重新生成：`python3 MODULE_DOCS/tools/gen_symbol_index.py`

| 文件 | 行数 | 符号数 |
|---|---:|---:|
| `freetoken/shell/__init__.py` | 82 | 3 |
| `freetoken/shell/__main__.py` | 4 | 0 |
| `freetoken/shell/client.py` | 327 | 19 |
| `freetoken/shell/render.py` | 350 | 34 |
| `freetoken/shell/tui.py` | 752 | 36 |

---

## `freetoken/shell/__init__.py`  (82 行)

| 行 | 符号 | 说明 |
|---:|---|---|
| 27 | `_wants_local_engine()` | _wants_local_engine(argv |
| 32 | `_build_parser()` | _build_parser(prog |
| 51 | `main()` | main(argv |

## `freetoken/shell/__main__.py`  (4 行)

_（无类/函数定义，模块级代码）_

## `freetoken/shell/client.py`  (327 行)

| 行 | 符号 | 说明 |
|---:|---|---|
| 39 | **class `ShellClientError`** | A request to the server failed, or the server refused it. Carries a message meant to be |
| 43 | **class `__init__`** | __init__(self, message |
| 49 | **class `ReasoningDelta`** | class ReasoningDelta |
| 54 | **class `ContentDelta`** | class ContentDelta |
| 59 | **class `TurnDone`** | class TurnDone |
| 69 | **class `Sampling`** | class Sampling |
| 76 | `_error_text()` | Pull a human-readable message out of the two error shapes the server emits: the OpenAI |
| 95 | **class `ShellClient`** | class ShellClient |
| 96 | **class `__init__`** | __init__( self, origin |
| 112 | **class `aclose`** | aclose(self) -> None |
| 117 | **class `_request_json_blocking`** | _request_json_blocking( self, method |
| 158 | **class `_request_json`** | _request_json( self, method |
| 170 | **class `health`** | health(self) -> dict[str, Any] |
| 173 | **class `stats`** | stats(self) -> dict[str, Any] |
| 176 | **class `cache_status`** | cache_status(self) -> dict[str, Any] |
| 179 | **class `cache_rebuild`** | Resize cache pools. Every count is in the endpoint's own unit -- slots for moe/mamba, |
| 205 | **class `model_id`** | The first id from ``/v1/models``. None when the server reports none (it is a single- |
| 217 | **class `wait_until_ready`** | Block until ``/health`` reports ok, returning that document. |
| 251 | **class `chat`** | Stream one turn. Yields reasoning/content deltas as the server splits them, then |

## `freetoken/shell/render.py`  (350 行)

| 行 | 符号 | 说明 |
|---:|---|---|
| 34 | `wrap_visual_lines()` | wrap_visual_lines(text |
| 45 | `wrap_line_by_cells()` | wrap_line_by_cells(line |
| 66 | `pad_cells()` | pad_cells(text |
| 71 | `clip_cells()` | clip_cells(text |
| 83 | `_ansi_dim()` | _ansi_dim(text |
| 87 | `_ansi_bold_dim()` | _ansi_bold_dim(text |
| 91 | `format_turn_header()` | format_turn_header(cmd |
| 106 | `format_think_block()` | format_think_block( content |
| 126 | `style_think_block()` | style_think_block(block |
| 137 | `clear_rows()` | clear_rows(num_rows |
| 151 | **class `ShellConsoleRenderer`** | Streams a turn to the terminal, folding reasoning into a dim ``Thinking...`` block and |
| 162 | **class `__init__`** | __init__( self, write |
| 177 | **class `begin_turn`** | begin_turn(self, cmd |
| 184 | **class `write_reasoning`** | write_reasoning(self, text |
| 190 | **class `write_content`** | First answer delta closes the live think block; after that content is plain text. |
| 199 | **class `finish_turn`** | finish_turn(self) -> None |
| 207 | **class `_render_think`** | _render_think(self, *, closed |
| 228 | **class `_write_stdout`** | _write_stdout(text |
| 233 | **class `ShellOutputBuffer`** | Coalesces per-token deltas into ~20 Hz repaints. Keeps the two channels in arrival |
| 237 | **class `__init__`** | __init__( self, renderer |
| 250 | **class `write_reasoning`** | write_reasoning(self, text |
| 253 | **class `write_content`** | write_content(self, text |
| 256 | **class `_append`** | _append(self, kind |
| 266 | **class `flush`** | flush(self) -> None |
| 278 | **class `ShellStatusLine`** | class ShellStatusLine |
| 279 | **class `__init__`** | __init__( self, format_status |
| 298 | **class `activate`** | activate(self) -> None |
| 303 | **class `deactivate`** | deactivate(self) -> None |
| 315 | **class `force`** | force(self) -> None |
| 319 | **class `maybe`** | maybe(self) -> None |
| 324 | **class `write_output`** | write_output(self, text |
| 329 | **class `_draw`** | _draw(self) -> None |
| 332 | **class `_render_status_line`** | _render_status_line(self) -> str |
| 341 | **class `_render_footer`** | _render_footer(self) -> str |

## `freetoken/shell/tui.py`  (752 行)

| 行 | 符号 | 说明 |
|---:|---|---|
| 60 | `_format_shell_model_label()` | _format_shell_model_label(model_id |
| 72 | `_apply_think_command()` | Resolve a ``/think`` argument to ``(new_gear, message)``. ``new_gear`` is the gear to use |
| 90 | `_cache_targets_hint()` | _cache_targets_hint(pools |
| 99 | `_cache_usage()` | The ``/cache`` usage line for THIS model: only its pools, with the unit each takes. |
| 102 | &nbsp;&nbsp;&nbsp;↳ `_units_clause()` | _units_clause(unit |
| 126 | `_parse_count()` | Parse a positive count with an optional ``k``/``m`` suffix (x1024 / x1024^2, case- |
| 142 | **class `CacheCommand`** | A parsed ``/cache`` invocation. ``kv``/``swa`` are token counts as typed -- the page |
| 154 | `_cache_error()` | _cache_error(message |
| 158 | `_parse_cache_command()` | Parse ``/cache`` arguments against the pools this model actually has. |
| 203 | **class `ShellStats`** | class ShellStats |
| 225 | **class `reset`** | reset(self) -> None |
| 233 | **class `mark_started`** | mark_started(self, now |
| 241 | **class `set_prompt_tokens`** | set_prompt_tokens(self, count |
| 247 | **class `add_completion_tokens`** | add_completion_tokens(self, count |
| 256 | **class `apply_usage`** | Replace the live estimates with the turn's authoritative usage. Completion tokens |
| 265 | **class `apply_stats_doc`** | Absorb a ``/v1/stats`` document. Every pool keeps last-known-value semantics: a |
| 287 | **class `apply_geometry`** | Absorb the ``geometry`` block of ``/v1/cache/status``: the MoE slot cache and its |
| 297 | **class `mark_finished`** | mark_finished(self, now |
| 301 | **class `tok_s`** | tok_s(self, now |
| 308 | **class `format`** | format(self, now |
| 333 | `_prompt_tokens_total()` | _prompt_tokens_total(doc |
| 340 | `_handle_cache_command()` | Shell ``/cache`` command: print the cache geometry, or resize any pool the model has at |
| 372 | &nbsp;&nbsp;&nbsp;↳ `_target()` | _target(pool |
| 439 | `_help_text()` | ``/help``, written against the served model: the thinking gears come from the server (a |
| 475 | `_sigint_cancels()` | Bind ^C to "cancel this turn" for as long as ``task`` runs. |
| 496 | `_shell_sampling()` | Sampling for a shell turn. Unset knobs stay unset so the server fills them from the |
| 507 | `_format_load_progress()` | _format_load_progress(doc |
| 518 | `run_shell()` | Attach to the FreeToken server at ``origin`` and run the terminal chat. |
| 531 | `_run_shell()` | _run_shell(client |
| 536 | &nbsp;&nbsp;&nbsp;↳ `_on_progress()` | One line per phase, then at most one per LOAD_PROGRESS_INTERVAL -- a 60 GiB |
| 596 | &nbsp;&nbsp;&nbsp;↳ `poll_stats()` | Keep the status bar live while a turn streams: pool occupancy, VRAM, and the prompt |
| 609 | &nbsp;&nbsp;&nbsp;↳ `run_turn()` | run_turn(cmd |
| 681 | &nbsp;&nbsp;&nbsp;↳ `handle_command()` | handle_command(cmd |
| 720 | &nbsp;&nbsp;&nbsp;↳ `_submit()` | _submit(event) |
| 724 | &nbsp;&nbsp;&nbsp;↳ `_insert_newline()` | _insert_newline(event) |
| 728 | &nbsp;&nbsp;&nbsp;↳ `_exit()` | _exit(event) |

