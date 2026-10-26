# 03 · server（HTTP 层）：进程编排、三协议适配、中立生成核心、统计与记账

> 行号基于 `377a3bc`。完整符号清单见 [`99-symbol-index/server.md`](99-symbol-index/server.md)（本文件不含两个 parser，见 `04`）。

---

## 1. 模块职责与在系统中的位置

`server/` 是**唯一面向客户端的一层**，同时承担三件不同的事：

1. **进程编排**：`server/launch.py` 把 scheduler 进程、tokenizer/detokenizer worker spawn 出来；
2. **协议适配**：OpenAI chat/completions、OpenAI Responses、Anthropic messages 三种 wire 协议 → 一个中立 `GenSpec`；
3. **前端状态**：准入、SSE 分发、统计、记账、维护态。

```
            ┌──────────────── uvicorn 主进程（本模块） ────────────────┐
HTTP 客户端 ─▶│ FastAPI 路由 → 协议适配器 → GenSpec → submit_generation │
            │        ▲                                    │           │
            │        │ UserReply                          │ TokenizeMsg
            │   FrontendManager.listen ◀── ZMQ(frontend) ──┘           │
            └────────┼────────────────────────────────────┼───────────┘
                     │                                    ▼
                     │                          ZMQ(backend) → Scheduler 进程
                     └──────────────── tokenizer/detokenizer worker ◀─────┘
```

端口：serve 默认 `127.0.0.1:1919`；ZMQ 用 `ipc:///tmp/freetoken_{0..4}.pid=<pid>`。

---

## 2. 文件清单

| 文件 | 行数 | 职责 | 阅读建议 |
|---|---:|---|---|
| `api_server.py` | 1,050 | FastAPI app、`FrontendManager`、ZMQ plumbing、`/v1/cache/*`、`/generate`、`run_api_server` | **精读** |
| `args.py` | 932 | `ServerArgs` 全部 CLI flag + `parse_args`（含 dtype/TP/parser auto 推断） | 精读（按需查 flag） |
| `generation.py` | 827 | **协议中立核心**：`GenSpec`/`GenEvent`、`submit_generation`、`generate_events/full`、模板渲染、采样解析 | **精读** |
| `openai_api.py` | 710 | OpenAI chat/completions + legacy completions + `/v1/models` | 精读 |
| `anthropic_api.py` | 655 | Anthropic `/v1/messages`、`count_tokens` | 概览 |
| `responses_api.py` | 780 | OpenAI `/v1/responses`（含 get/cancel） | 概览 |
| `launch.py` | 228 | spawn 各族进程、ready ack 收集 | **精读** |
| `supervisor.py` | 186 | 就绪门与存活监控 | 精读 |
| `accounting.py` | 157 | `/v1/admin/prepare-stop` 的 drain/abort 屏障 | 精读 |
| `stats.py` | 176 | `StatsTracker`（滑窗吞吐）+ `build_stats`（`/v1/stats` 文档） | 精读 |
| `api_models.py` / `anthropic_models.py` | 149 / 178 | 两种协议的 pydantic 请求模型 | 按需 |
| `request_ring.py` | 101 | 最近请求环形缓冲（`/v1/requests`） | 略读 |
| `request_logger.py` | 177 | 可选 JSONL 请求日志（后台线程） | 略读 |
| `access_log_filter.py` | 89 | 过滤桌面端轮询噪音 | 略读 |
| `control_api.py` | 80 | `/health`、`/v1/stats`、`/v1/requests` | 精读（短） |
| `model_meta.py` | 149 | 从配置派生的模型事实（effort/thinking kwargs 等） | 略读 |
| `function_call_parser.py` / `reasoning_parser.py` | 3,756 / 929 | 见 `04` | — |

---

## 3. 核心数据结构

### 3.1 `ServerArgs`（`args.py:38`）

继承链：**`ServerArgs` ⊂ `SchedulerConfig` ⊂ `EngineConfig`**。也就是说"服务端 flag"是"调度配置"的超集，而调度配置又是"引擎配置"的超集；越往下越接近 GPU。

| 组 | 字段 |
|---|---|
| 网络 | `server_host`（127.0.0.1）、`server_port`（1919）、`cors_origins` |
| 进程 | `num_tokenizer`（0 = tokenizer 与 detokenizer 共用一个 worker）、`share_tokenizer`（`:71`）、`shell_mode`（终端 shell 挂在本服务上，worker 要脱离前台进程组） |
| 模型/解析 | `served_model_name`、`tool_call_parser`（默认 `llama3`）、`reasoning_parser`、`sampling_defaults`（`model`/`none`）、`max_output_tokens`、`enable_cache_report` |
| 多模态 | `allowed_media_domains`、`allowed_local_media_path` |
| GPU | `gpu`（TP 顺序）、`gpu_assigned`（解析后的 UUID；None = NVML 不可用，由各 worker 自己解析） |
| 继承自 SchedulerConfig | `max_extend_tokens`、`cache_type`、`offline_mode`、`decode_log_interval`、ZMQ 地址 |
| 继承自 EngineConfig | 全部引擎/MoE/量化/显存旋钮 |

### 3.2 `GenSpec`（`generation.py:136`）——协议中立请求

| 字段 | 含义 |
|---|---|
| `messages` | 归一化、可直接喂模板的消息列表 |
| `sampling_params` | `core.SamplingParams` |
| `chat_template_kwargs` | 模板参数（thinking / effort 等） |
| `template_tools` | 模型看到的 tools（→ `TokenizeMsg.tools`） |
| `parser_tools` | **解析器**用的 tools；None 表示关闭解析 |
| `inline_system_policy` | `preserve` / `fold` / None |

派生 `parse_tools`（`:151`）。**设计意图写在 docstring 里**："no wire request type reaches the engine path"——三种协议只在适配器里存在。

### 3.3 `GenEvent` 家族（`generation.py:70-124`）

`ReasoningDelta`（`:70`）、`ContentDelta`（`:75`）、`ToolCallStart`（`:80`）、`ToolCallArgsDelta`（`:93`）、`ToolCallsDelta`（`:103`）、`GenDone`（`:112`）；`GenResult`（`:124`）是非流式结果。协议适配器只负责把这一组事件格式化成自己的 wire 形状。

### 3.4 `FrontendManager`（`api_server.py:132`）的内部状态

| 状态 | 含义 |
|---|---|
| `ack_map` | `uid → 已收到的最后一个 UserReply`（等待轮询用） |
| `event_map` | `uid → asyncio.Event`（唤醒等待者） |
| `maintenance_state` | `loading` / `serving` / `rebuilding`；决定是否 503 |
| `instance_id` | 本实例标识（`/v1/stats` 暴露） |
| `stats` | `StatsTracker` |
| `cache_pools` | 池几何元信息（来自 engine 的 meta） |
| `gpus` | GPU 身份列表（ready meta 到达后填充） |
| `frontend_tokenizer` | 只读的 `TokenizeManager`，供 `count_tokens` |

### 3.5 `StatsTracker`（`stats.py:13`）
5 秒滑窗（`:14`）：`_decode` / `_prefill` 两个 deque 记 `(时间, token 数)`；`decode_tps()`（`:97`）/`prefill_tps()`（`:100`）由 `_rate`（`:86`）算。另有 `active`（`:42`）、`inflight_uids`（`:46`）、`completed`、`prompt_tokens_total`、`completion_tokens_total`、`kv/mamba/swa` 占用、`vram_bytes`。

`build_stats(state, p95_ms, ttft_mean_ms)`（`stats.py:129`）组装 `/v1/stats` 的完整 JSON：`model`（`derive_model_card:104` 给 `attn`/`moe`/`input_modalities`）、`kv`/`mamba`/`swa`（无则 null）、`vram_bytes`、`throughput`、`requests`（含 `p95_ms`/`ttft_mean_ms`）。

---

## 4. HTTP 路由全表（15 条）

| 方法 + 路径 | 处理函数 | 行号 | 请求模型 | 返回 |
|---|---|---|---:|---|
| `POST /v1/chat/completions` | `v1_chat_completions` | `openai_api.py:118` | `ChatCompletionRequest` | JSON 或 SSE |
| `POST /v1/completions` | `v1_completions` | `openai_api.py:126` | 旧版 completion | JSON 或 SSE（**绕过 GenSpec**） |
| `GET /v1/models` | `list_models` | `openai_api.py:134` | — | 模型列表（含 `/v1/models` 供 launch 发现） |
| `POST /v1/responses` | `create_response` | `responses_api.py:116` | `ResponsesRequest` | JSON 或 SSE |
| `GET /v1/responses/{id}` | 取回 | `responses_api.py:134` | — | JSON |
| `POST /v1/responses/{id}/cancel` | 取消 | `responses_api.py:138` | — | JSON |
| `POST /v1/messages` | `create_message` | `anthropic_api.py:83` | `MessageRequest` | JSON 或 SSE |
| `POST /v1/messages/count_tokens` | 计 token | `anthropic_api.py:93` | 同 messages | JSON |
| `GET /health` | 健康 | `control_api.py:57` | — | 状态 + 加载进度 + `maintenance_state` |
| `GET /v1/stats` | 统计 | `control_api.py:71` | — | 见 §3.5 |
| `GET /v1/requests` | 请求环 | `control_api.py:63` | — | 最近请求 |
| `GET /v1/cache/status` | 缓存表 | `api_server.py:820` | — | 各池几何 + 上限 |
| `POST /v1/cache/rebuild` | 运行时重建 | `api_server.py:576` | `CacheRebuildRequest` | 几何或 `busy`/`failed` |
| `POST /generate` | 原始补全 | `api_server.py:830` | `GenerateRequest` | 文本（**不过 chat 模板**，冒烟用） |
| `POST /v1/admin/prepare-stop` | 停机前屏障 | `accounting.py:139` | — | drain/abort 后的记账结果（**仅 loopback**） |

---

## 5. 函数详解

### 5.1 `api_server.py` — `FrontendManager` 与装配

| 方法/函数 | 行 | 作用 | 关键行为 |
|---|---:|---|---|
| `get_global_state()` | 62 | 取全局 `FrontendManager` | 供路由闭包使用 |
| `_terminate_backend_workers` / `_exit_after_backend_death` / `_reap_backend_workers` | 68/87/100 | 后端进程死了怎么办 | 有 grace 定时器，避免端口挂着但引擎已死 |
| `_unwrap_msg` | 114 | 解 `Batch*Msg` 信封 | 统一成 list |
| `FrontendManager.__post_init__` | 200 | 初始化队列与事件 | |
| `frontend_tokenizer` / `warm_frontend_tokenizer` | 204/218 | 前端只读 tokenizer（count_tokens） | 惰性构建 |
| `new_user()` | 234 | **准入闸门** | 关闭时抛 `AdmissionClosedError`；分配 uid |
| `_resolve_rebuild` / `fail_pending_rebuilds` | 262/294 | 重建请求的等待与失败传播 | |
| `_create_listener_once` | 318 | 建 ZMQ 监听线程 | 只建一次 |
| `shutdown` | 383 | 停服 | 关准入 + 通知 worker |
| `send_one` | 324 | 往前端地址推一条消息 | |
| `listen` / `wait_for_ack` | 246 / 328 | **收 UserReply 并唤醒等待者** | 更新 `stats`、`ack_map`、`event_map` |
| `abort_user` | 373 | 客户端断连 → 发 `AbortMsg` | |
| `install_cors` | 403 | CORS 中间件 | |
| `_served_model_name` | 443 | 模型 id 决议 | |
| `cache_geometry` / `_cache_limits` | 721 / 629 | `/v1/cache/status` 的几何与上限 | |
| `_resolve_num_swa_pages` | 552 | SWA 页数的两种表达（绝对/比例） | |
| `run_api_server` | 930 | **主装配**：建 app、装路由、起监听、等 ready、起 uvicorn | |
| `_serve_and_run_shell` | 886 | shell 模式（同进程起引擎 + TUI） | |
| `_install_shell_stop_handlers` | 858 | shell 的停机信号 | |

### 5.2 `generation.py` — 中立生成核心

| 函数 | 行 | 作用 |
|---|---:|---|
| `resolve_sampling` | 162 | 请求未指定的采样参数从 `generation_config.json` 补（`--sampling-defaults model`） |
| `render_messages` / `_render_message` / `_flatten_text_parts` | 196 / 203 / 239 | 消息归一化（把各协议的消息形状压成模板可用的） |
| `split_tool_lists` | 276 | 把 tools 分成"给模型看的"和"给解析器用的" |
| `submit_generation` | 293 | **提交入口**：准入 → 发 `TokenizeMsg` |
| `prerender_error` | 354 | 前端侧预渲染，把坏模板变成干净的 400（不提交 SSE 头） |
| `_make_reasoning_parser` / `_split_reasoning` | 385 / 421 | 每请求构造 reasoning parser / 非流式整段切分 |
| `_make_tool_parser` / `_parse_tool_response` / `_valid_json` | 436 / 448 / 464 | 同上（工具调用） |
| `generate_events` | 538 | **流式主驱动**：把 reply 流变成 `GenEvent` 流 |
| `generate_full` | 568 | 非流式：收齐再拼 `GenResult` |

### 5.3 三协议适配器

| 文件 | 请求 → GenSpec | 事件 → wire |
|---|---|---|
| `openai_api.py` | `chat_request_to_genspec`（`:59`） | `stream_chat_completion_chunks`（`:240`）、`_sse`（`:624`）、usage chunk（`:371`）、`data: [DONE]`（`:385`） |
| `anthropic_api.py` | `convert_anthropic_to_genspec`（`:313`） | `anthropic_event_stream`（`:431`） |
| `responses_api.py` | `convert_responses_to_genspec`（`:188`） | `responses_stream_generator`（`:496`） |

**注意**：`chat_request_to_genspec` **不是**其它协议的公共枢轴——三个适配器各自独立转换到 `GenSpec`。

### 5.4 `launch.py` / `supervisor.py`

| 函数 | 行 | 作用 |
|---|---:|---|
| `launch_server` | `launch.py:125` | 解析 args → 解析 `--gpu` UUID → `start_subprocess` → `run_api_server` |
| `start_subprocess` | `launch.py:153` | spawn 世界：每 TP rank 一个 scheduler、1 个 detokenizer、N 个 tokenizer |
| `_run_scheduler` / `_run_tokenize_worker` | 同上文件 | 子进程入口 |
| `drain_ready` | `supervisor.py:99` | 收齐 `expected_acks = num_tokenizers + 2` 个 ack（含进度/元信息元组），全部到齐才 `_on_ready` |
| `BackendHandle` | `supervisor.py` | 子进程句柄与存活监控 |

### 5.5 `accounting.py` / `control_api.py` / `stats.py`

| 函数 | 行 | 作用 |
|---|---:|---|
| `install_prepare_stop(app)` / handler | `accounting.py:139` | `/v1/admin/prepare-stop`：关准入 → drain/abort → 产出最终记账；**只接受 loopback** |
| `install_control_api` + 三个 handler | `control_api.py:57/63/71` | `/health`、`/v1/requests`、`/v1/stats` |
| `StatsTracker.observe` | `stats.py:58` | 每条 reply 更新滑窗与累计 |
| `derive_model_card` | `stats.py:104` | `attn` 分类（`hybrid_linear`/`hybrid_swa`/`mha`）+ `moe` + `input_modalities` |
| `build_stats` | `stats.py:129` | 组装 `/v1/stats` |

---

## 6. 关键流程逐段解析

### 6.1 `ft serve` 进程编排与就绪门
```
cli.py _run_serve → server/launch.py:125 launch_server
  → args.parse_args（dtype/TP/parser auto 全在这）
  → resolve_gpu_uuids（--gpu 的 UUID ↔ 索引）
  → start_subprocess:
        mp.set_start_method("spawn")
        for i in range(world_size): mp.Process(_run_scheduler, ...)      # 每 TP rank
        mp.Process(_run_tokenize_worker, addr=detokenizer_addr, ...)     # detokenizer
        for i in range(num_tokenizer): mp.Process(_run_tokenize_worker, ...)
  → run_api_server:
        install_cors / 路由 / FrontendManager
        start_backend() → supervisor.drain_ready()
              等 expected_acks = num_tokenizer + 2
              就绪字符串来自 server/launch.py:106 的 "Scheduler is ready"
        _on_ready → maintenance_state: loading → serving（记录 ready_at）
        uvicorn.run(...)
```

### 6.2 一次流式 chat completions 的完整路径
```
① POST /v1/chat/completions → openai_api:118
② maintenance 闸门（rebuilding 时 503）→ handle_chat_completion:150
③ chat_request_to_genspec:59 → GenSpec
④ prerender_error:195 → 坏模板在提交 SSE 头之前变成 400
⑤ submit_generation:293 → state.new_user()（准入）→ send_one(TokenizeMsg)
⑥ 立刻建流：StreamingResponse + stream_with_cancellation
⑦ worker: tokenize → UserMsg → scheduler
⑧ scheduler: PrefillAdder → Batch → engine.forward_batch → DetokenizeMsg
⑨ worker: detokenize → UserReply（每 token 一条）
⑩ FrontendManager.listen:246 → stats.observe → 唤醒 wait_for_ack
⑪ 适配器把 GenEvent 格式化成 SSE chunk
⑫ 结束：usage chunk + data: [DONE]
```

### 6.3 三协议共享边界
共享：`generation.py` 的全部中立原语 + `GenSpec`/`GenEvent`。
不共享：请求 pydantic 模型（`api_models.py` vs `anthropic_models.py` vs `ResponsesRequest`）与事件→wire 的格式化函数。
**例外**：legacy `/v1/completions`（`openai_api.py:388-457`）**绕过 `GenSpec`**，直接构造 `TokenizeMsg`——所以它不走模板渲染、也不走工具/推理解析。

### 6.4 非流式 vs 流式的记账差异
- 流式：每个 token 一条 `UserReply`，用 `completion_tokens_delta` 累加；prompt token 由 `PromptAdmittedMsg` 转成的**一条** `UserReply`（`prompt_tokens_delta`）补。
- 非流式（`generate_full`）：收齐事件后一次性构造结果。
- **共性**：prompt 计费都不在 tokenize 时发生，而是等首次 prefill 认定（见 `02`、`14`）。

### 6.5 断连 → abort
```
客户端断开 → stream_with_cancellation 捕获 → abort_user:373
  → 发 AbortMsg → worker 转 AbortBackendMsg → scheduler
  → scheduler 置 req.aborted（在飞）或立即释放（不在飞）
```

### 6.6 维护态与 prepare-stop
```
POST /v1/cache/rebuild
  → dispatch_rebuild: 置 maintenance_state="rebuilding"，新生成请求 503
  → CacheRebuildMsg → scheduler（idle 才执行，忙则回 busy）
  → 结果回来 → 恢复 "serving"
POST /v1/admin/prepare-stop（仅 loopback，daemon 用）
  → 关准入 → drain 或 abort → 产出最终记账 receipt
```

---

## 7. 与其他模块的关系

| 方向 | 模块 | 接口 | 说明 |
|---|---|---|---|
| 使用 | `message/` | 收发 `TokenizeMsg` / `UserReply` / `CacheRebuild*` | 跨进程 |
| 使用 | `tokenizer/` | worker 实现（子进程） | |
| 使用 | `scheduler/` | 通过 ZMQ，不直接 import | 子进程隔离 |
| 使用 | `engine/` | 间接（scheduler 持有） | |
| 使用 | `04` 的两个 parser | 每请求构造 | |
| 使用 | `mm/media.py` | 图像引用解析与校验 | |
| 被使用 | `daemon/` | loopback HTTP（`/health`、`/v1/stats`、`/v1/admin/prepare-stop`） | 见 `16` |
| 被使用 | `launch.py`（根目录） | `ft launch` 通过 `/v1/models`、`/v1/stats` 发现服务 | |

---

## 8. 常见坑

1. **`message/` vs `tokenizer/`**：类型 vs 实现；同一个 worker 两角色。
2. **prompt 记账不在 tokenize 时**（`PromptAdmittedMsg` 才补）。
3. **legacy `/v1/completions` 绕过 `GenSpec`**：不走模板、不走解析器。
4. **overlap 下 `req.aborted` 只是标记**：真正释放在批次排空后。
5. **`count_tokens` 用的是前端那份只读 `TokenizeManager`**，不是 worker。
6. **shell 模式会强制 `cuda_graph_max_bs=1` 且 `max_running_req=1`**（`args.py:833`）。
7. **`--gpu` 给多个但 `tensor_parallel_size=1` 会直接报错**（`args.py:820-826`）。
8. **`/v1/admin/prepare-stop` 只接受 loopback**，即使推理 API 绑了 `0.0.0.0`。

---

## 9. 自测问题

1. `ft serve` 一共 spawn 了几类进程？各自持有 torch/GPU 吗？
2. 就绪门的 `expected_acks` 为什么是 `num_tokenizer + 2`？
3. `GenSpec` 里 `template_tools` 与 `parser_tools` 为什么要分开？
4. 为什么 legacy `/v1/completions` 要绕过 `GenSpec`？代价是什么？
5. 流式与非流式的 prompt 计费分别在哪一刻发生？
6. 维护态（rebuilding）下新请求会怎样？重建是"拒绝"还是"排队"？
7. `shell_mode` 改变了哪些默认值，为什么？
8. 客户端断连后，资源是在哪一刻真正释放的？
