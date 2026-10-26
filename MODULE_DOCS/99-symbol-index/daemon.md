# 符号索引：`daemon`

自动生成（`ast` 解析，不执行代码）。**16 个文件 / 3057 行 / 205 个符号**。

> 用途：确认某个模块里**到底有哪些**类/函数/方法及其行号，避免遗漏。
> 行为详解请看同目录下的手写模块文档。
> 重新生成：`python3 MODULE_DOCS/tools/gen_symbol_index.py`

| 文件 | 行数 | 符号数 |
|---|---:|---:|
| `freetoken/daemon/__init__.py` | 31 | 1 |
| `freetoken/daemon/__main__.py` | 5 | 0 |
| `freetoken/daemon/accounting.py` | 160 | 14 |
| `freetoken/daemon/app.py` | 509 | 45 |
| `freetoken/daemon/checkpoint.py` | 174 | 10 |
| `freetoken/daemon/client.py` | 198 | 9 |
| `freetoken/daemon/logfmt.py` | 76 | 5 |
| `freetoken/daemon/logring.py` | 67 | 8 |
| `freetoken/daemon/metrics.py` | 134 | 8 |
| `freetoken/daemon/osproc.py` | 198 | 14 |
| `freetoken/daemon/pidfile.py` | 107 | 13 |
| `freetoken/daemon/proxy.py` | 129 | 12 |
| `freetoken/daemon/serve_manager.py` | 913 | 51 |
| `freetoken/daemon/server.py` | 231 | 8 |
| `freetoken/daemon/tailer.py` | 117 | 7 |
| `freetoken/daemon/version.py` | 8 | 0 |

---

## `freetoken/daemon/__init__.py`  (31 行)

| 行 | 符号 | 说明 |
|---:|---|---|
| 14 | `main()` | Single entry for ``ft daemon``. A leading client verb (``ft daemon status`` …) controls a |

## `freetoken/daemon/__main__.py`  (5 行)

_（无类/函数定义，模块级代码）_

## `freetoken/daemon/accounting.py`  (160 行)

| 行 | 符号 | 说明 |
|---:|---|---|
| 19 | **class `AccountingError`** | Base class for fail-closed accounting lifecycle errors. |
| 23 | **class `AccountingPrepareError`** | The engine's prepare-stop endpoint failed or returned a non-sealed snapshot. |
| 27 | **class `AccountingOutboxError`** | A receipt could not be durably persisted or read. |
| 31 | **class `PrepareStopUnavailable`** | The engine predates the prepare-stop endpoint (HTTP 404/405). |
| 38 | `_validate_receipt_id()` | _validate_receipt_id(receipt_id |
| 44 | `stable_receipt_id()` | Return the same receipt ID when a stop is retried for the same engine identity. |
| 49 | **class `AccountingOutbox`** | Crash-safe receipt outbox backed by one atomically-replaced JSON file per receipt. |
| 52 | **class `__init__`** | __init__( self, path |
| 62 | **class `persist`** | Durably store ``receipt`` and return the canonical persisted document. |
| 96 | **class `pending`** | Return all unacknowledged receipts in deterministic creation order. |
| 109 | **class `ack`** | Idempotently acknowledge and remove one receipt. |
| 125 | **class `_receipt_path`** | _receipt_path(self, receipt_id |
| 128 | **class `_read_one`** | _read_one(self, path |
| 148 | **class `_fsync_dir`** | Best-effort directory fsync; unsupported on Windows but required on Linux durability. |

## `freetoken/daemon/app.py`  (509 行)

| 行 | 符号 | 说明 |
|---:|---|---|
| 29 | **class `StartBody`** | class StartBody(BaseModel) |
| 35 | **class `StopBody`** | class StopBody(BaseModel) |
| 39 | **class `SwitchBody`** | class SwitchBody(StartBody) |
| 43 | **class `AccountingAckBody`** | class AccountingAckBody(BaseModel) |
| 47 | **class `CheckpointBody`** | class CheckpointBody(BaseModel) |
| 52 | **class `CancelBody`** | class CancelBody(BaseModel) |
| 56 | **class `BenchBody`** | class BenchBody(BaseModel) |
| 61 | `_bench_profile_path()` | _bench_profile_path(gpu_uuid |
| 72 | `_serve_gpu_uuid()` | The full UUID a serve's `--gpu` pins, or None when there is none or it cannot be resolved. |
| 91 | `_read_bench_profile()` | _read_bench_profile(path |
| 101 | `_bench_sse()` | _bench_sse(event |
| 105 | `_parse_ftbench()` | ``FTBENCH <done> <total> <label>`` -> a progress dict (mirrors ft checkpoint's FTCONVERT). |
| 116 | `build_app()` | build_app( *, manager, ring, probe, footprint_fn |
| 139 | &nbsp;&nbsp;&nbsp;↳ `_on_shutdown()` | _on_shutdown() -> None |
| 148 | &nbsp;&nbsp;&nbsp;↳ `require_token()` | require_token(x_ft_token |
| 154 | &nbsp;&nbsp;&nbsp;↳ `run()` | run(pool |
| 158 | &nbsp;&nbsp;&nbsp;↳ `resolve_port()` | resolve_port(explicit |
| 164 | &nbsp;&nbsp;&nbsp;↳ `accounting_error()` | accounting_error(exc |
| 182 | &nbsp;&nbsp;&nbsp;↳ `health()` | health() |
| 194 | &nbsp;&nbsp;&nbsp;↳ `engine_start()` | engine_start(body |
| 213 | &nbsp;&nbsp;&nbsp;↳ `engine_stop()` | engine_stop(body |
| 220 | &nbsp;&nbsp;&nbsp;↳ `shutdown_daemon()` | shutdown_daemon(request |
| 238 | &nbsp;&nbsp;&nbsp;↳ `engine_switch()` | engine_switch(body |
| 257 | &nbsp;&nbsp;&nbsp;↳ `accounting_pending()` | accounting_pending() |
| 265 | &nbsp;&nbsp;&nbsp;↳ `accounting_ack()` | accounting_ack(body |
| 274 | &nbsp;&nbsp;&nbsp;↳ `engine_status()` | engine_status() |
| 278 | &nbsp;&nbsp;&nbsp;↳ `engine_metrics()` | engine_metrics() |
| 283 | &nbsp;&nbsp;&nbsp;↳ `engine_health()` | engine_health() |
| 300 | &nbsp;&nbsp;&nbsp;↳ `engine_stats()` | engine_stats() |
| 310 | &nbsp;&nbsp;&nbsp;↳ `engine_logs()` | engine_logs(request |
| 318 | &nbsp;&nbsp;&nbsp;↳ `checkpoint_start()` | checkpoint_start(body |
| 329 | &nbsp;&nbsp;&nbsp;↳ `checkpoint_cancel()` | checkpoint_cancel(body |
| 333 | &nbsp;&nbsp;&nbsp;↳ `checkpoint_status()` | checkpoint_status() |
| 339 | &nbsp;&nbsp;&nbsp;↳ `bench_run()` | bench_run(body |
| 348 | &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;↳ `gen()` | gen() |
| 384 | &nbsp;&nbsp;&nbsp;↳ `bench_profile()` | bench_profile() |
| 385 | &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;↳ `read()` | read() -> dict \| None |
| 388 | &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;↳ `serve_gpu_uuid()` | serve_gpu_uuid() -> str \| None |
| 406 | `_engine_summary()` | _engine_summary(st |
| 416 | `_sse()` | _sse(rec |
| 420 | `_sse_gap()` | _sse_gap(dropped |
| 425 | `_log_stream()` | SSE log stream with replay + live tail. Correctness points: |
| 449 | &nbsp;&nbsp;&nbsp;↳ `push()` | push(rec |
| 450 | &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;↳ `_put()` | _put() -> None |
| 473 | &nbsp;&nbsp;&nbsp;↳ `gen()` | gen() |

## `freetoken/daemon/checkpoint.py`  (174 行)

| 行 | 符号 | 说明 |
|---:|---|---|
| 26 | **class `Job`** | class Job |
| 34 | **class `CheckpointManager`** | class CheckpointManager |
| 35 | **class `__init__`** | __init__( self, ring, *, python |
| 59 | **class `_default_spawn`** | _default_spawn(self, job_id |
| 66 | **class `start`** | start(self, job_id |
| 111 | **class `_monitor`** | _monitor(self, child, job_id |
| 133 | **class `cancel`** | cancel(self, job_id |
| 148 | **class `_terminate`** | SIGTERM → grace → SIGKILL on the child's group (shared by cancel() and the |
| 155 | **class `status`** | status(self) -> dict |
| 170 | **class `_emit`** | _emit(self, text |

## `freetoken/daemon/client.py`  (198 行)

| 行 | 符号 | 说明 |
|---:|---|---|
| 30 | **class `ClientError`** | class ClientError(Exception) |
| 31 | **class `__init__`** | __init__(self, message |
| 36 | `_effective_timeout()` | _effective_timeout(verb |
| 46 | `_request()` | _request(method |
| 68 | `_err_body()` | _err_body(raw |
| 83 | `_request_json()` | _request_json(method, url, path, *, body=None, query=None, token=None, timeout=10.0) -> dict |
| 94 | `_stream_logs()` | _stream_logs(url, since, token, timeout) -> None |
| 116 | `_build_parser()` | _build_parser(prog |
| 161 | `main()` | main(argv |

## `freetoken/daemon/logfmt.py`  (76 行)

| 行 | 符号 | 说明 |
|---:|---|---|
| 22 | `strip_ansi()` | strip_ansi(text |
| 26 | **class `LineAssembler`** | Feed raw decoded chunks; get back finalized display lines. Stateful across chunks (a line |
| 33 | **class `__init__`** | __init__(self) -> None |
| 37 | **class `feed`** | feed(self, text |
| 67 | **class `flush`** | flush(self) -> list[tuple[str, str]] |

## `freetoken/daemon/logring.py`  (67 行)

| 行 | 符号 | 说明 |
|---:|---|---|
| 23 | **class `LogRing`** | class LogRing |
| 24 | **class `__init__`** | __init__(self, capacity |
| 30 | **class `append`** | Append one record and fan it out to subscribers. ``kind`` is ``"line"`` (finalized), |
| 46 | **class `since`** | Records with ``seq >= cursor`` plus the next cursor to poll with (the all-time count). |
| 53 | **class `cursor`** | cursor(self) -> int |
| 57 | **class `subscribe`** | subscribe(self, push |
| 61 | **class `unsubscribe`** | unsubscribe(self, push |
| 65 | **class `subscriber_count`** | subscriber_count(self) -> int |

## `freetoken/daemon/metrics.py`  (134 行)

| 行 | 符号 | 说明 |
|---:|---|---|
| 19 | `engine_footprint()` | engine_footprint(pid |
| 28 | **class `FootprintCache`** | Single-flight + short-TTL cache over ``engine_footprint``. Clients poll metrics frequently |
| 34 | **class `__init__`** | __init__(self, *, ttl_s |
| 40 | **class `get`** | get(self, pid |
| 50 | `vram_bytes_for_pids()` | vram_bytes_for_pids(pids |
| 68 | `_nvml_ready()` | _nvml_ready() |
| 81 | `_nvml_process_vram()` | _nvml_process_vram() -> dict[int, int] \| None |
| 112 | `_smi_process_vram()` | _smi_process_vram() -> dict[int, int] |

## `freetoken/daemon/osproc.py`  (198 行)

| 行 | 符号 | 说明 |
|---:|---|---|
| 17 | `pid_alive()` | True iff a process with this pid currently exists. ``kill(pid, 0)`` is the portable probe: |
| 33 | `_read_proc()` | _read_proc(pid |
| 43 | `read_cmdline()` | The process argv as a list (``/proc/<pid>/cmdline`` is NUL-separated). [] if unavailable. |
| 54 | `_stat_fields()` | Split ``/proc/<pid>/stat`` past the comm field (which may itself contain spaces/parens). |
| 66 | `read_starttime()` | The process start time (stat field 22, clock ticks since boot). Combined with the pid this |
| 79 | `proc_pgid()` | The process group id (stat field 5). Used to (a) enumerate the serve's worker tree and |
| 91 | `_iter_all_pids()` | _iter_all_pids() -> Iterable[int] |
| 100 | `tree_pids()` | The serve and all its worker processes: every pid whose process group == ``root_pid``. |
| 116 | `signal_group()` | Signal the serve's whole process group. Resolves the pgid and asserts ``pgid == pid`` |
| 133 | `set_oom_score_adj()` | Best-effort write to ``/proc/<pid>/oom_score_adj``. Returns whether it stuck. |
| 145 | `read_pss_bytes()` | Proportional set size for one process from ``/proc/<pid>/smaps_rollup`` (shared pages |
| 159 | `is_ft_serve_on_port()` | Verify ``pid`` is (still) an ``ft serve`` bound to ``port`` — the re-adoption / liveness |
| 182 | `_argv_port()` | The integer following a standalone ``--port`` / ``-p`` in argv, else the serve default |
| 197 | `now_monotonic()` | now_monotonic() -> float |

## `freetoken/daemon/pidfile.py`  (107 行)

| 行 | 符号 | 说明 |
|---:|---|---|
| 18 | **class `AlreadyRunning`** | Another daemon holds the single-instance lock. |
| 22 | **class `SingleInstance`** | class SingleInstance |
| 23 | **class `__init__`** | __init__(self, path |
| 27 | **class `acquire`** | acquire(self) -> None |
| 42 | **class `release`** | release(self) -> None |
| 50 | **class `__enter__`** | __enter__(self) -> "SingleInstance" |
| 54 | **class `__exit__`** | __exit__(self, *exc) -> None |
| 59 | **class `ServeState`** | class ServeState |
| 68 | **class `ServeStateStore`** | class ServeStateStore |
| 69 | **class `__init__`** | __init__(self, path |
| 72 | **class `save`** | save(self, state |
| 81 | **class `clear`** | clear(self) -> None |
| 89 | **class `load`** | load(self) -> ServeState \| None |

## `freetoken/daemon/proxy.py`  (129 行)

| 行 | 符号 | 说明 |
|---:|---|---|
| 27 | `_camel_key()` | _camel_key(key |
| 31 | `to_camel()` | Recursively camelCase dict keys (uptime_s→uptimeS, done_bytes→doneBytes, …) so the daemon |
| 41 | **class `ServeProbe`** | class ServeProbe |
| 42 | **class `__init__`** | __init__( self, *, host |
| 63 | **class `health`** | health(self, port |
| 66 | **class `stats`** | stats(self, port |
| 69 | **class `fresh_stats`** | Fetch uncached stats for a legacy stop receipt. |
| 73 | **class `prepare_stop`** | Quiesce the engine and return its sealed final accounting snapshot. |
| 88 | **class `_cached`** | _cached(self, kind |
| 100 | **class `_fetch`** | _fetch(self, path |
| 113 | **class `_urlopen`** | _urlopen(url |
| 120 | **class `_urlopen_prepare`** | _urlopen_prepare(url |

## `freetoken/daemon/serve_manager.py`  (913 行)

| 行 | 符号 | 说明 |
|---:|---|---|
| 39 | **class `Conflict`** | A different serve (model/port/args) is already running; the client should switch(). |
| 44 | **class `ExitInfo`** | class ExitInfo |
| 52 | **class `PopenChild`** | A serve WE spawned. ``wait()`` is a real ``os.waitpid`` via Popen. |
| 55 | **class `__init__`** | __init__(self, proc |
| 64 | **class `wait`** | wait(self) -> ExitInfo |
| 68 | **class `poll`** | poll(self) -> ExitInfo \| None |
| 74 | **class `close`** | close(self) -> None |
| 78 | **class `AdoptedChild`** | A serve a PREVIOUS daemon spawned, re-adopted from the pidfile. It is NOT our child, so |
| 83 | **class `__init__`** | __init__( self, pid |
| 105 | **class `wait`** | wait(self) -> ExitInfo |
| 110 | **class `poll`** | poll(self) -> ExitInfo \| None |
| 113 | **class `close`** | close(self) -> None |
| 120 | `build_serve_command()` | The serve invocation + its log path. ``python -m freetoken.cli serve`` (NOT ``-m |
| 131 | `spawn_serve()` | Spawn the serve as a session leader with its stdout+stderr going to a real logfile fd (never |
| 155 | **class `ServeManager`** | class ServeManager |
| 156 | **class `__init__`** | __init__( self, ring, state_store, *, spawn_fn |
| 235 | **class `_default_spawn`** | _default_spawn(self, model |
| 241 | **class `_default_adopt`** | _default_adopt(self, state) |
| 258 | **class `start`** | start( self, model |
| 264 | **class `_start`** | _start( self, model |
| 327 | **class `stop`** | stop(self, timeout |
| 331 | **class `shutdown`** | Permanently close admission to new serves, then stop the current child. |
| 350 | **class `_stop`** | _stop(self, timeout |
| 404 | **class `switch`** | switch( self, model |
| 416 | **class `pending_accounting`** | pending_accounting(self) -> list[dict[str, Any]] |
| 419 | **class `ack_accounting`** | ack_accounting(self, receipt_id |
| 422 | **class `observe_accounting`** | Remember the latest validated totals for the current child. |
| 449 | **class `_prepare_and_persist_accounting`** | _prepare_and_persist_accounting( self, *, child, model |
| 466 | **class `_prepare_and_persist_accounting_locked`** | _prepare_and_persist_accounting_locked( self, *, child, model |
| 516 | **class `_persist_crash_accounting`** | Persist one identity-stable degraded receipt before a dead child is forgotten. |
| 551 | **class `_sealed_receipt`** | _sealed_receipt( self, prepared |
| 579 | **class `_best_effort_receipt`** | _best_effort_receipt( self, identity |
| 625 | **class `_snapshot_values`** | _snapshot_values( self, snapshot |
| 673 | **class `_degraded_receipt`** | _degraded_receipt( self, identity |
| 699 | **class `_child_identity`** | _child_identity(child) -> str |
| 703 | **class `_is_token_total`** | _is_token_total(value |
| 706 | **class `detach`** | Daemon is shutting down but the engine must OUTLIVE it: drop the |
| 721 | **class `readopt`** | At boot, re-attach a still-running serve recorded in the pidfile. Returns |
| 748 | **class `_begin_watch`** | _begin_watch(self, child) -> None |
| 758 | **class `_monitor`** | _monitor(self, child) -> None |
| 766 | **class `_reap`** | _reap(self, child, info |
| 817 | **class `_exit_line`** | _exit_line(self, child, info |
| 826 | **class `_restart_async`** | _restart_async(self, dead_child) -> None |
| 831 | &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;↳ `_run()` | _run() |
| 841 | **class `_maybe_apply_oom`** | _maybe_apply_oom(self, pid |
| 845 | **class `reapply_oom`** | Write the positive OOM score to the whole serve tree so the 22 GB serve — not the tiny |
| 861 | **class `current_pid`** | current_pid(self) -> int \| None |
| 865 | **class `serve_args`** | Engine args of the running (or last started) serve. |
| 870 | **class `status`** | status(self) -> dict |
| 895 | **class `_persist`** | _persist(self, child, model |
| 909 | **class `_emit`** | _emit(self, text |

## `freetoken/daemon/server.py`  (231 行)

| 行 | 符号 | 说明 |
|---:|---|---|
| 27 | `_default_state_dir()` | _default_state_dir() -> str |
| 34 | `_build_parser()` | _build_parser(prog |
| 66 | `_install_signal_hygiene()` | SIGHUP ignored so a ``wsl.exe --`` relay close can't kill the daemon; SIGPIPE |
| 87 | `_start_oom_reaper()` | Periodically rewrite the positive OOM score across the whole serve tree so mp-spawn workers |
| 91 | &nbsp;&nbsp;&nbsp;↳ `_run()` | _run() -> None |
| 104 | `main()` | main(argv |
| 137 | &nbsp;&nbsp;&nbsp;↳ `tailer_factory()` | tailer_factory(child) |
| 176 | &nbsp;&nbsp;&nbsp;↳ `shutdown_hook()` | shutdown_hook() -> None |

## `freetoken/daemon/tailer.py`  (117 行)

| 行 | 符号 | 说明 |
|---:|---|---|
| 26 | **class `LogTailer`** | class LogTailer(threading.Thread) |
| 27 | **class `__init__`** | __init__( self, log_path |
| 55 | **class `stop`** | stop(self) -> None |
| 58 | **class `run`** | run(self) -> None |
| 84 | **class `_follow`** | _follow(self, fh) -> None |
| 100 | **class `_consume`** | _consume(self, data |
| 105 | **class `_append`** | _append(self, kind |

## `freetoken/daemon/version.py`  (8 行)

_（无类/函数定义，模块级代码）_

