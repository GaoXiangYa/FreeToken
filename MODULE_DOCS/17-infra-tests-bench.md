# 17 · 基础设施、测试与基准：Registry、ZMQ 队列、分布式、测量工具、测试方法论

> 行号基于 `377a3bc`。完整符号清单见 [`99-symbol-index/utils.md`](99-symbol-index/utils.md)、[`distributed.md`](99-symbol-index/distributed.md)、[`benchmark.md`](99-symbol-index/benchmark.md)。

---

## 1. 模块职责与在系统中的位置

这一篇覆盖三类"被所有模块使用"的东西：

| 组 | 内容 | 谁在用 |
|---|---|---|
| **基础设施** | `utils/`（Registry、ZMQ 队列、HF 配置缓存、日志、ROCm 探测、进度）、`distributed/`（TP 通信） | 几乎所有模块 |
| **测量工具** | `benchmark/`（微观计时 + 压测客户端）、`benchmarks/`（三个端到端基准）、`ft bench bw` | 性能工作、CI 之外的手工门禁 |
| **规格与发布** | `tests/`（141 个测试，充当**可执行规格**）、`scripts/` + `.github/workflows/`（wheel 构建与发布） | 改代码的人 |

**注意**：`utils/` 里有些文件会拉重依赖（`hf.py` → transformers），所以 `daemon/` 被禁止 import `freetoken.utils.*`（见 `16`）。

---

## 2. `utils/`（9 文件 / 967 行）

### 2.1 `registry.py`（46 行）——全仓唯一的插件机制

| 成员 | 行 | 契约 |
|---|---:|---|
| `Registry(type)` | 6 | 构造时给一个"类别名"，只用于报错 |
| `register(name, info=None)` | 12 | **装饰器**；重复注册同名 → `KeyError` |
| `info(name)` | 23 | 取注册时附带的 `BackendInfo` 等元数据 |
| `__getitem__(name)` | 28 | 取工厂；未注册 → `KeyError`（列支持项） |
| `supported_names()` | 33 | 列出全部名字 |
| `assert_supported(names)` | 36 | CLI 校验用；失败抛 `ArgumentTypeError` |

**全仓只有两处用它**：`attention/__init__.py:38`（`SUPPORTED_ATTENTION_BACKENDS`）与 `kvcache/__init__.py:24`（`SUPPORTED_CACHE_MANAGER`）。模型族用显式 dict（`models/register.py:88`），解析器用 Enum——**所以"有哪些插件"永远可以穷举**。

### 2.2 `mp.py`（152 行）——ZMQ / multiprocessing 队列

29 个符号，核心是四个队列类型 + 编解码封装：

| 类 | socket 类型 | 典型用途 |
|---|---|---|
| `ZmqPushQueue` | PUSH | 前端 → worker（单向、可多生产者） |
| `ZmqPullQueue` | PULL | worker 侧接收 |
| `ZmqPubQueue` | PUB | TP rank0 广播给其它 rank |
| `ZmqSubQueue` | SUB | 其它 rank 接收 |

关键参数：
- `create=True/False`：谁负责 bind、谁负责 connect（多实例不冲突靠 pid 后缀）；
- `encoder`/`decoder`：消息编解码函数（`BaseBackendMsg.encoder` 等）；
- `get_raw()` / `put_raw()`：**绕过解码**的原始转发——TP rank0 用它在广播时不重复序列化。

### 2.3 `hf.py`（299 行）——HF 配置与 checkpoint 解析
- `cached_load_hf_config(path, overrides)`：带缓存的 `AutoConfig` 读取；`overrides` 按 vLLM 语义应用（nested 段逐键更新，其它值整体替换）。
- 其余的 checkpoint 路径解析、`config.json` 读取、文件枚举工具。
**缓存键**要注意：同一个路径 + 不同 overrides 是两个缓存条目。

### 2.4 其余

| 文件 | 行数 | 内容 |
|---|---:|---|
| `misc.py` | 54 | `UNSET`/`Unset`、`div_ceil`/`div_even`/`align_down`/`mem_GB`、`call_if_main` |
| `logger.py` | 130 | `init_logger` + `info_rank0`/`warning_rank0`（TP 下只有 rank0 打日志） |
| `arch.py` | 101 | ROCm 的 gfx 架构探测（`get_rocm_gfx_arch`） |
| `progress.py` | 91 | `emit_progress`：加载进度 → `/health` 的就绪条 |
| `torch_utils.py` | 38 | `nvtx_annotate`（装饰器 + `layer_id_field`）、`set_default_dtype` |
| `__init__.py` | 64 | 统一 re-export（**这就是为什么 daemon 不能 import 它**） |

### 2.5 `distributed/`（3 文件 / 147 行）

| 文件 | 符号 | 作用 |
|---|---|---|
| `info.py` | `DistributedInfo(rank, size)`（`:7`）、`is_primary()`（`:14`）、`set_tp_info`（`:21`）/`get_tp_info`（`:28`）/`try_get_tp_info`（`:34`） | 全局 TP 信息（单例） |
| `impl.py` | `DistributedImpl`（`:16`）抽象 → `TorchDistributedImpl`（`:25`，torch.distributed）/ `PyNCCLDistributedImpl`（`:45`，pynccl）；`DistributedCommunicator`（`:63`）；`enable_pynccl_distributed`（`:73`）/`destroy_distributed`（`:93`） | 通信后端 |
| `__init__.py` | — | 装配 |

`all_reduce` / `all_gather` 是仅有的两个算子——足够 TP 的权重切分与前向 collective。

---

## 3. `benchmark/`（2 文件 / 575 行）——测量工具

### 3.1 `perf.py:10 perf_cuda` —— 全仓微观基准的模板

```python
def perf_cuda(f, *, init_stream=True, repetitions=10, cuda_graph_repetitions=10) -> float:
```
流程：建 stream → 先跑一次 `f()` 热身 → **用 CUDA graph 把 N 次迭代打包**（`:31-40`）→ `synchronize` → **再 replay 一次热身** → `tic.record()` → 跑 `repetitions` 次 → `toc.record()` → 返回 `dur / (N*repetitions)`。

为什么这么写：
- 用 CUDA graph 摊薄 launch 开销，测的是**稳态**；
- `tic/toc` 是 `torch.cuda.Event`，避免 host 计时误差；
- 第一次 replay 是为把缓存/页表都烧热。

`compare_memory_kernel_perf`（`:54`）：跑 baseline 与 our_impl，按 `memory_footprint / (dur*1e6)` 换算 GB/s。

### 3.2 `client.py`（501 行）——压测与 trace 回放

| 符号 | 行 | 作用 |
|---|---:|---|
| `BenchmarkTrace` | 19 | 一条回放轨迹（prompt/输出长度/时间） |
| `BenchOutResult` / `RawResult` | 27 / 43 | 单请求 / 原始结果 |
| `Counter` / `Console` | 51 / 65 | 进度与吞吐显示 |
| `BenchmarkResult` | 113 | 聚合结果 |
| `make_console` | 124 | 控制台 |
| `generate_prompt` | 183 | 生成压测 prompt |
| `process_benchmark_results` | 320 | 计算吞吐/延迟分位 |
| `read_qwen_trace` / `read_mooncake_trace` | 407 / 445 | 读取两种公开 trace 格式 |
| `scale_traces` | 479 | 按目标 QPS 缩放轨迹 |

---

## 4. 测试体系（`tests/`，141 个测试文件）

### 4.1 方法论（来自 `tests/README.md`）

测试的取舍原则非常明确，值得先读：
1. **只测"会静默失败"的东西**：一个看起来像数字的错误数字、某族悄悄失去工具调用解析、缓存返回错误槽位。
2. **测在它所在的地方**，不要通过暴露它的那层表面去测。
3. **期望值必须来自独立参照**：PyTorch 参考、CPU 镜像、往返、活注册表——**不能复述实现**。测试若只描述作者刚写的分支，就永远不会失败。
4. **应用面（CLI、TUI、access log、请求环、健康端点）不单测**：它们一用就露馅，且形态变得快。属于 `e2e/` 与"真跑一遍"。

### 4.2 目录 → 生产模块 映射全表

| 测试目录 | 文件数 | 被测子系统 | 代表文件 / 钉住什么 |
|---|---:|---|---|
| `kvcache/` | 17 | 分页池与三套 radix | `radix/test_plain_radix.py`（+ `radix/model.py` 按页参考模型、`driver.py` 不变量电池） |
| `scheduler/` | 13 | 分块 prefill、CacheManager、记账、SWA/GDN 锁 | `test_scheduler_chunked_prefill.py`（中间块不进缓存）、`test_cost_accounting_core.py` |
| `server/` | 18 | 三协议、流式、解析器、记账、维护态 | `test_streaming_model_matrix.py`、`test_function_call_parser.py`、`test_parser_auto_selection.py` |
| `models/` | 34 | 注册表、加载机制、各族数值 | `test_models_registry.py`（遍历活注册表）、`deepseek_v41/test_reference_parity.py`（对 vendored 参考比 logits） |
| `kernels/` | 23 | 注意力/量化/rope/JIT 缓存 | `test_rotary.py`（按定义写的 YaRN 参考）、`test_kernel_cache_version.py` |
| `moe/` | 12 | offload、CPU MoE、各格式专家 | `test_offload.py`、`test_dsfp4_reference_semantics.py`（dequant oracle） |
| `engine/` | 6 | 显存预算、后端矩阵、多模态编码 | `test_cache_budget.py`（纯算术，CPU 可跑） |
| `tokenizer/` | 4 | 模板、多模态 token 展开、thinking 决议 | `test_tokenize.py`、`test_mm_tokenize.py` |
| `attention/` `dsv4/` | 2 / 1 | 稀疏后端 | `test_dsv41_backend.py` |
| `checkpoint/` | 2 | FTW 权重与 hotfix | `test_ftw_weights.py`（往返） |
| `daemon/` | 2 | **import 安全**、serve manager | `test_daemon_import_safety.py`（子进程 meta-path 拦截器） |
| `mm/` `layers/` `utils/` | 2 / 1 / 2 | 处理器、超连接、ROCm 探测 | |
| `e2e/` | 2 | 真服务门禁 | `test_aime.py`（AIME 24/25/26 生成）、`test_cache_rebuild.py`（真实 rebuild） |

### 4.3 标记与门控

- `slow` / `needs_weights` 只在 `pyproject.toml:122-124` 声明。
- **`needs_weights` 本身不跳过任何测试**——它只是标签，真正跳过来自旁边的 `@needs_checkpoint` / `os.environ` skipif / 测试内 `pytest.skip`。
- 真实 checkpoint 的门控环境变量见 `tests/README.md:39-70`（`FREETOKEN_TEST_MODEL`、`FREETOKEN_AIME_*`、各族 `*_MODEL` 等）。

### 4.4 conftest

- 根 `tests/conftest.py` 只有 10 行：一个 autouse fixture 在测试后重置全局 `QuantBackend`（防止内核请求泄漏到下一个测试）。
- `tests/kvcache/radix/conftest.py` 提供**确定性时钟**（LRU 断言否则在负载机器上会 flaky）与 per-class 几何参数。
- `tests/models/qwen4_exp/conftest.py` 设一次 TP 信息并清 `core._GLOBAL_CTX`。

### 4.5 五个最值得先读的测试

| # | 文件 | 能学到什么 |
|---|---|---|
| 1 | `kvcache/radix/test_plain_radix.py` | 前缀缓存的完整契约。harness 拿**按页建键的独立参考模型**与缓存逐字段比对 + 每步跑不变量电池；README 记录注入 17 个变异全部捕获 |
| 2 | `scheduler/test_scheduler_chunked_prefill.py` | 分块 prefill 的页会计与 overlap 顺序；docstring 直接写了它复现的真实 bug |
| 3 | `engine/test_cache_budget.py` | 专家槽位 vs KV 页的切分算术，**纯 CPU 可跑** |
| 4 | `moe/test_dsfp4_reference_semantics.py` | W4A8 专家契约：fp8 激活往返、swiglu 限幅、路由权重在 down epilogue；独立转写 + CPU/GPU 交叉验证 |
| 5 | `daemon/test_daemon_import_safety.py` | 用**子进程** meta-path 拦截器保护"daemon 不许 import torch"（同进程检查会假通过） |

### 4.6 无 GPU 也能读/跑的测试
radix 全套、scheduler/engine/kvcache 的算术测试、daemon、tokenizer、全部 parser 测试、`test_rotary.py`、`test_models_registry.py`。
CUDA 相关：`kernels/` 大部分、`attention/`、`dsv4/`、`moe/` 多数、`models/deepseek_v41/`、`e2e/`。

### 4.7 CI 不跑 pytest
四个 workflow 都是 wheel 构建/发布。测试是**本地 / pre-PR 门禁**——意味着它没有被 CI 强制的必然正确性，读的时候保留一点怀疑。

---

## 5. `benchmarks/` 与 `scripts/`

### 5.1 三个基准

| 脚本 | 测什么 | 方法学 | 对应主张 |
|---|---|---|---|
| `bench_decode_moe.py` | bs=1 decode tok/s（端到端） | 每个后端起一个 `ft serve` 子进程，计时流式 `/v1/chat/completions` 的 token 到达；AIME-25 prompt；按 checkpoint 推荐采样；`ignore_eos` 保证步数精确 | README 的 MoE 服务性能主张；**唯一端到端数字** |
| `bench_load_weight_generic.py` | 专家 bank 启动加载时间 | 串行 vs 并行 O_DIRECT vs 预重打包 FTW，**各自独立子进程**并驱逐 page cache；Linux only；在 `/var/tmp` 里 staging FTW | FTW 快速格式的收益 |
| `bench_offload_cache_copy.py` | 每层 decode 的专家拷贝成本 | **合成、无 checkpoint**：`ensure_experts` + `copy_missing` 扫 bank 布局 × cache 槽位 × batch × miss 率 | 全局 LRU 专家缓存的设计 |

**先读 `bench_offload_cache_copy.py`**（无 checkpoint、最小 surface），它是理解 `OffloadMoeCache` 的最轻入口。
主机 RAM vs PCIe 带宽与 offload/hybrid 选型请用 `ft bench bw`（它写引擎读的 JSON profile，见 `06`）。

### 5.2 发布管线

| workflow | 触发 | 产物 |
|---|---|---|
| nightly | cron / 手动 | 自托管 `[self-hosted, linux, engine-build]` 在 `pytorch/manylinux2_28-builder:cuda13.0` 里构建 → 发布到 rolling `nightly` release 并移动 tag |
| promote-beta | 手动 | 把那对 wheel **逐字节复制**（不重建）到 FreeToken-Web 的 `beta` |
| release | `v*` tag | cp310–cp313 runtime wheel + 一个 `py3-none <version>+cu130` kernel-cache wheel；runtime 发 PyPI/TestPyPI，全部进 draft release |

平台矩阵：Linux x86_64（glibc 2.28 floor）、CUDA 13.0、torch 2.11 cu130；nightly 只有 cp312，release cp310–cp313；ROCm 在树内但实验性。

---

## 6. 函数详解（按文件分组）

### 6.1 `utils/`

| 函数/类 | 行 | 作用 |
|---|---:|---|
| `Registry.register/info/__getitem__/supported_names/assert_supported` | `registry.py:12/23/28/33/36` | 插件注册表全部 API |
| `Zmq*Queue` 四件套 + `get_raw`/`put_raw` | `mp.py` | 跨进程队列 |
| `cached_load_hf_config` | `hf.py` | 带 overrides 的缓存式 config 读取 |
| `init_logger` / `info_rank0` / `warning_rank0` | `logger.py` | 分级日志 |
| `div_ceil` / `div_even` / `align_down` / `mem_GB` / `UNSET` | `misc.py` | 算术与哨兵 |
| `get_rocm_gfx_arch` | `arch.py` | ROCm 架构探测 |
| `emit_progress` | `progress.py` | `/health` 的加载进度 |
| `nvtx_annotate` | `torch_utils.py:23` | NVTX 区间装饰器 |

### 6.2 `distributed/`

| 符号 | 行 | 作用 |
|---|---:|---|
| `DistributedInfo` / `is_primary` | `info.py:7/14` | TP 身份 |
| `set_tp_info` / `get_tp_info` / `try_get_tp_info` | `info.py:21/28/34` | 全局单例存取 |
| `TorchDistributedImpl` / `PyNCCLDistributedImpl` | `impl.py:25/45` | 两种通信后端 |
| `enable_pynccl_distributed` / `destroy_distributed` | `impl.py:73/93` | 切换与销毁 |

### 6.3 `benchmark/`

见 §3。

---

## 7. 与其他模块的关系

| 方向 | 模块 | 接口 | 说明 |
|---|---|---|---|
| 被几乎所有模块使用 | `utils` | re-export 的公共 API | 但 `daemon` 禁止使用 |
| 被使用 | `scheduler/io.py`、`tokenizer/server.py`、`server/api_server.py` | `Zmq*Queue` | 跨进程 |
| 被使用 | `engine`、`attention`、`models` | `DistributedInfo`、`all_reduce`/`all_gather` | TP |
| 被使用 | `attention/__init__.py`、`kvcache/__init__.py` | `Registry` | 两个注册表 |
| 被使用 | 全仓 | `init_logger`、`div_ceil`、`mem_GB` | |
| 测量 | `moe/benchbw.py`、`benchmarks/*` | `benchmark/perf.py` 的计时模板 | |

---

## 8. 常见坑

1. **`Registry` 重复注册同名会 `KeyError`**——插件的 import 顺序因此有意义。
2. **`needs_weights` 不跳过任何测试**；跳过来自 env guard。
3. **CI 不跑 pytest**：测试是本地门禁，别以为"没跑过就是对的"。
4. **`utils/__init__.py` 会拉重依赖**：这就是 daemon 不能 import 它的原因。
5. **`benchmark/perf.py:perf_cuda` 的第一次 replay 是热身**：去掉它会测到冷缓存。
6. **`Zmq*Queue` 的 `create` 决定 bind 还是 connect**：多个实例靠 pid 后缀区分，写错会撞 socket。
7. **`env.py` 的 `ENV.X` 只是环境变量的一部分**：还有 50+ 个散落的 `os.getenv`。
8. **`tests/README.md` 的测试哲学不是建议而是规则**：新增测试要能"在合理回归时失败"。

---

## 9. 自测问题

1. 全仓有几个 `Registry` 实例？为什么模型族不用它？
2. `ZmqPushQueue`/`PullQueue`/`PubQueue`/`SubQueue` 各对应什么通信模式？
3. `perf_cuda` 为什么要用 CUDA graph 打包 N 次迭代，还要先 replay 一次？
4. 测试的"期望值必须来自独立参照"具体指哪些参照？举三个。
5. `needs_weights` 与 `slow` 两个标记的真实语义差别是什么？
6. 为什么 `daemon/` 不能 import `freetoken.utils.*`？
7. `all_reduce` / `all_gather` 两种通信后端的切换开关是什么？
8. 发布管线里 nightly 与 release 的产物差别是什么？
