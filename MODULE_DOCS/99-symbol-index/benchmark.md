# 符号索引：`benchmark`

自动生成（`ast` 解析，不执行代码）。**2 个文件 / 575 行 / 37 个符号**。

> 用途：确认某个模块里**到底有哪些**类/函数/方法及其行号，避免遗漏。
> 行为详解请看同目录下的手写模块文档。
> 重新生成：`python3 MODULE_DOCS/tools/gen_symbol_index.py`

| 文件 | 行数 | 符号数 |
|---|---:|---:|
| `freetoken/benchmark/client.py` | 501 | 35 |
| `freetoken/benchmark/perf.py` | 74 | 2 |

---

## `freetoken/benchmark/client.py`  (501 行)

| 行 | 符号 | 说明 |
|---:|---|---|
| 19 | **class `BenchmarkTrace`** | class BenchmarkTrace |
| 27 | **class `BenchOneResult`** | class BenchOneResult |
| 32 | **class `as_json`** | as_json(self) -> List[float] |
| 36 | **class `from_json`** | from_json(raw |
| 43 | **class `RawResult`** | class RawResult |
| 51 | **class `Counter`** | class Counter |
| 55 | **class `inc`** | inc(self, n=1) |
| 59 | **class `dec`** | dec(self, n=1) |
| 65 | **class `Console`** | class Console |
| 74 | **class `update_input`** | update_input(self, n=1) |
| 80 | **class `update_output`** | update_output(self, n=1) |
| 85 | **class `update_prefill`** | update_prefill(self, n=1) |
| 90 | **class `update_decode`** | update_decode(self, n=1) |
| 94 | **class `inflight`** | inflight(self, n=1) |
| 100 | **class `log_stats`** | log_stats(self) |
| 113 | **class `BenchmarkResult`** | class BenchmarkResult |
| 116 | **class `as_json`** | as_json(self) -> List[List[float]] |
| 120 | **class `from_json`** | from_json(raw |
| 124 | `make_console()` | make_console(num_requests |
| 183 | `generate_prompt()` | Generate a prompt of approximately `n` tokens using the provided tokenizer. |
| 202 | `benchmark_one()` | benchmark_one( client |
| 251 | `benchmark_one_batch()` | benchmark_one_batch( client |
| 287 | `benchmark_trace()` | benchmark_trace( client |
| 300 | &nbsp;&nbsp;&nbsp;↳ `benchmark_timed()` | benchmark_timed(msg |
| 313 | `process_benchmark_results()` | process_benchmark_results(raw_data |
| 317 | `process_benchmark_results()` | process_benchmark_results(raw_data |
| 320 | `process_benchmark_results()` | process_benchmark_results( raw_data |
| 340 | &nbsp;&nbsp;&nbsp;↳ `_print_stats()` | _print_stats(times |
| 350 | &nbsp;&nbsp;&nbsp;↳ `_fmt()` | _fmt(x |
| 407 | `read_qwen_trace()` | read_qwen_trace( file_path |
| 413 | **class `JSONInput`** | class JSONInput(BaseModel) |
| 445 | `read_mooncake_trace()` | read_mooncake_trace( file_path |
| 451 | **class `JSONInput`** | class JSONInput(BaseModel) |
| 479 | `scale_traces()` | scale_traces( traces |
| 498 | `get_model_name()` | get_model_name(client |

## `freetoken/benchmark/perf.py`  (74 行)

| 行 | 符号 | 说明 |
|---:|---|---|
| 10 | `perf_cuda()` | perf_cuda( f |
| 54 | `compare_memory_kernel_perf()` | compare_memory_kernel_perf( *, baseline |

