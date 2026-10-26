# 14 · message / tokenizer / llm：跨进程消息、tokenize-detokenize worker、离线 API

> 行号基于 `377a3bc`。完整符号清单见 [`99-symbol-index/message.md`](99-symbol-index/message.md)、[`tokenizer.md`](99-symbol-index/tokenizer.md)、[`llm.md`](99-symbol-index/llm.md)。

---

## 1. 模块职责与在系统中的位置

三个模块解决三件不同的事，但常被混为一谈：

| 模块 | 是什么 | 关键事实 |
|---|---|---|
| `message/` | **跨进程消息类型 + 手写 codec** | 被前端、tokenizer worker、scheduler 三方 import；**import torch**（因为要序列化张量），所以不是 torch-free 层 |
| `tokenizer/` | **tokenize / detokenize worker 的实现** | 跑在独立进程里，持 transformers，无 GPU |
| `llm/` | **离线 API** | `LLM(Scheduler)` + `offline_mode=True`，用两个函数替换 ZMQ 收发——是"最小可跑入口" |

```
┌── 前端进程 ──┐   TokenizeMsg    ┌── tokenizer worker ──┐   UserMsg    ┌── scheduler ──┐
│ FrontendMgr │ ───────────────▶ │ tokenize_worker      │ ───────────▶ │ Scheduler     │
│             │ ◀─────────────── │  (tokenize/detokenize)│ ◀─────────── │               │
└─────────────┘   UserReply      └──────────────────────┘ DetokenizeMsg└───────────────┘
        ▲                                   ▲
        └──────── message/ 定义所有类型 ─────┘

离线路径：LLM.generate() ──▶ 直接构造 UserMsg / 直接收 DetokenizeMsg（无 ZMQ、无独立进程）
```

**结构性要点**：`message/` 只定义"线上跑什么"，`tokenizer/` 定义"收到之后怎么做"。这两件事分开是因为**同一个 worker 函数同时承担 tokenize 与 detokenize 两个角色**（靠构造参数区分），类型必须被两边共享。

---

## 2. 文件清单

| 文件 | 行数 | 职责 | 阅读建议 |
|---|---:|---|---|
| `message/backend.py` | — | worker → scheduler 的消息（`UserMsg` 等） | 精读 |
| `message/tokenizer.py` | — | 前端 ↔ worker ↔ scheduler 的消息（`TokenizeMsg` / `DetokenizeMsg` / `PromptAdmittedMsg`…） | **精读**（消息全表） |
| `message/frontend.py` | — | worker → 前端的消息（`UserReply` 等） | 精读 |
| `message/utils.py` | — | 手写 codec：`serialize_type` / `deserialize_type` + 张量载荷 | 精读（约 80 行） |
| `tokenizer/server.py` | — | **worker 主循环**（两种角色共用） | **精读** |
| `tokenizer/tokenize.py` | — | `TokenizeManager`：chat template / DSV4 encoder / thinking 决议 | 精读 |
| `tokenizer/detokenize.py` | — | `DetokenizeManager`：增量解码 + 两种 holdback | 精读 |
| `tokenizer/effort.py` | — | reasoning effort 的探针与清洗 | 按需 |
| `tokenizer/inline_system.py` | — | inline system 指令的 preserve/fold 策略 | 按需 |
| `llm/llm.py` | 125 | `LLM(Scheduler)` 离线入口 | **精读**（全仓信噪比最高） |

---

## 3. 消息协议（`message/`）

### 3.1 三个命名空间

**A. `tokenizer.py`（前端 ↔ worker ↔ scheduler）**

| 消息 | 方向 | 字段 |
|---|---|---|
| `TokenizeMsg` | 前端 → worker | `uid`、`text`（str 或 messages 列表）、`sampling_params`、`chat_template_kwargs`、`tools`、`images`、`inline_system_policy` |
| `DetokenizeMsg` | scheduler → detokenizer | `uid`、`next_token`、`finished`、`finish_reason`、`matched_stop`、`stop_strs`、以及**池占用快照**：`kv_used_pages`/`kv_total_pages`、`mamba_used_slots`/`mamba_total_slots`、`swa_used_tokens`/`swa_total_tokens`、`gpu_mem_bytes` |
| `PromptAdmittedMsg` | scheduler → worker | `uid`、`prompt_tokens`（**完整 prompt 长度，即使分块**）、`cached_tokens`（前缀命中数） |
| `AbortMsg` | 前端 → worker | `uid` |
| `CacheRebuildMsg` / `CacheRebuildResultMsg` | 前端 ↔ worker | `request_id`、四个尺寸、`mode`（只有 `if_idle`）、`status` |
| `ErrorReplyMsg` | scheduler → worker | `uid`、`error`、`code`（`context_length_exceeded` 等） |
| `BaseTokenizerMsg` / `BatchTokenizerMsg` | 基类 / 批信封 | `data: List[...]` |

**B. `backend.py`（worker → scheduler）**

| 消息 | 字段 |
|---|---|
| `UserMsg` | `uid`、`input_ids`（**CPU 1D int32**）、`sampling_params`、`mm_items`、`mrope_positions`（`[3, n]`）、`mrope_delta` |
| `AbortBackendMsg` | `uid` |
| `CacheRebuildBackendMsg` | 同 `CacheRebuildMsg` |
| `ExitMsg` | 空（停机信号） |
| `MMItem`（数据结构，非消息） | `modality`、`hash`、`pad_value`、`offsets`（半开区间列表）、`feature` 或 `precomputed_embeddings`（**恰有一个**）、`model_specific_data`（可当属性读） |

**C. `frontend.py`（worker → 前端）**

| 消息 | 字段 |
|---|---|
| `UserReply` | `uid`、`incremental_output`、`finished`、`prompt_tokens_delta`、`completion_tokens_delta`、`cached_tokens`、六项池占用、`gpu_mem_bytes`、`error`、`error_code`、`finish_reason`、`matched_stop` |
| `CacheRebuildReply` | `request_id`、`status`、四个尺寸、`error` |
| `BaseFrontendMsg` / `BatchFrontendMsg` | 基类 / 批信封 |

### 3.2 方向命名反直觉（按事实记）
- scheduler **收** `BaseBackendMsg`（`UserMsg` 属于"backend"这一族）；
- scheduler **发** `BaseTokenizerMsg`（`DetokenizeMsg` 属于"tokenizer"这一族）。

不要按"谁是谁的后端"去推，按这张表记。

### 3.3 codec（`message/utils.py`）

| 符号 | 行 | 作用 |
|---|---:|---|
| `_TYPE_KEY = "__type__"` | 8 | 类型标签 |
| `_RAW_DICT_KEY = "__raw_dict__"` | 13 | **保护用户自由 dict**：payload 里可能有字段恰好叫 `__type__`（tool JSON Schema、`chat_template_kwargs`），包一层避免被误当类型标签——否则一个请求就能打崩 tokenizer worker |
| `_serialize_any` | 15 | 递归编码；dict 里出现保留键就加 `__raw_dict__` 包装 |
| `serialize_type` | 28 | **张量特判**：必须是 CPU、contiguous；`__type__="Tensor"`、`dtype`、1-D 省略 shape（兼容旧线上格式）、bf16 走 `view(uint16)`（numpy 没有 bf16）后 `tobytes()` |
| `_deserialize_any` | 56 | 递归解码，含 `__raw_dict__` 还原 |
| `deserialize_type` | 75 | 按 `__type__` 还原；张量用 `np.frombuffer` + `torch.from_numpy`（bf16 再 `view` 回来） |

**注意**：它 import `torch` 与 `numpy`。所以"`message/` 是 torch-free"是**错的**——真正被要求 torch-free 的是 `daemon/`（见 `16`）。

---

## 4. tokenizer worker（`tokenizer/`）

### 4.1 `tokenizer/server.py` —— 一个函数、两个角色
主循环（`tokenize_worker`）从队列收消息，**按类型分派**：
- `TokenizeMsg` → `_tokenize_requests` → `TokenizeManager.tokenize` → 产出 `UserMsg` 推给 scheduler；
- `DetokenizeMsg` → `DetokenizeManager.detokenize` → 产出 `UserReply` 推给前端；
- `PromptAdmittedMsg` → 转成"提示 token 计费"的 `UserReply`（`completion_tokens_delta=0`）；
- `AbortMsg` / `CacheRebuildMsg` → 分别转发/回包；
- `ErrorReplyMsg` → 翻成终止性 `UserReply(error=...)`（否则前端会永远等不到 `finished`）。

**角色由构造参数决定**：`addr`/`backend_addr`/`frontend_addr`/`create`/`tokenizer_id` 不同 → 同一个函数承担 tokenize 或 detokenize。`Llm/launch.py:177-215` 里给它两次不同的 kwargs 就是这两个角色。

### 4.2 `TokenizeManager`（`tokenize/tokenize.py`）
`messages → chat template → input_ids`，中间有 4 个决策点：
1. **thinking mode 决议**（`resolve_thinking_mode`）：由模型族 + 请求参数决定是否开思考；
2. **DSV4 native encoder 分支**：DSV4 有自带的 `encoding/encoding_dsv4.py`，优先用它；
3. **inline system 策略**：`preserve` / `fold`（晚到的 system 指令放在原位还是折叠进首条）——由 checkpoint 的模板能力探测决定（`tokenizer/inline_system.py`）；
4. **effort 清洗**（`tokenizer/effort.py`）：`reasoning_effort` 的整数/档位如何塞进模板 kwargs。

### 4.3 `DetokenizeManager`（`detokenize.py`）
增量解码有两道 holdback，**都不能省**：
1. **partial-UTF-8 holdback**：一个多字节字符（如 emoji）可能被切成多个 token，先吐会出乱码 → 只吐"解码后完整"的部分。
2. **stop-prefix holdback**：已生成文本的尾部可能是某个 stop string 的前缀（如 stop=`"</answer>"`，当前尾部是 `"</ans"`）→ 先扣住，若后续补齐就裁掉，否则再吐。没有它就只能"先吐再撤回"，而 SSE 无法撤回。
3. 结束时还有 `matched_stop` 裁剪：把命中的 stop string 及其之后全部去掉。

### 4.4 前端为什么还有一份只读 `TokenizeManager`
`/v1/messages/count_tokens` 需要 tokenize 但不生成。前端持有一份只读实例（`api_server.py:204-216`），类型与 worker 共享、实例独立。

---

## 5. 离线 API（`llm/llm.py`，125 行）

### 5.1 构造
```python
config = SchedulerConfig(..., offline_mode=True)
super().__init__(config)     # 继承 Scheduler
```
`offline_mode=True` 让 `SchedulerIOMixin.__init__` **提前 return**（`scheduler/io.py:30-33`），并把 `receive_msg`/`send_result` 绑到两个可覆写方法上。

### 5.2 四个方法

| 方法 | 行 | 作用 |
|---|---:|---|
| `_tokenize_one` | 47 | str → encode；list → tensor（int32） |
| `offline_receive_msg` | 53 | **把待处理请求转成 `UserMsg`**：按 `prefill_budget` 截断、分配 uid、多模态走 `mm_processor.apply`；没有待处理且要求阻塞时抛 `RequestAllFinished` |
| `offline_send_result` | 89 | 收 `DetokenizeMsg` 累积 token；`PromptAdmittedMsg` 直接忽略（离线没有记账 sink） |
| `generate` | 100 | 公开 API：装 pending → `run_forever()` → 捕 `RequestAllFinished` → 组装 `{"text","token_ids"}` |

### 5.3 为什么它是"最小可跑入口"
- 不需要 HTTP、不需要 ZMQ、不需要 spawn 子进程；
- 但**跑的是同一个 `Scheduler` + 同一个 `Engine`**，所以行为与线上一致；
- 125 行读完就能理解调度器的输入输出契约。

用法（无 GPU 时无法实际运行，仅示意）：
```python
from freetoken.llm import LLM
from freetoken.core import SamplingParams
llm = LLM("/path/to/model")
print(llm.generate(["你好"], SamplingParams(max_tokens=32)))
```

---

## 6. 关键流程逐段解析

### 6.1 三进程接力
```
前端：submit_generation → TokenizeMsg ──ZMQ──▶
worker：tokenize_worker → TokenizeManager.tokenize → UserMsg ──ZMQ──▶
scheduler：add_one_req → ... → DetokenizeMsg ──ZMQ──▶
worker：DetokenizeManager.detokenize → UserReply ──ZMQ──▶
前端：FrontendManager.listen → SSE
```
一次请求跨 **3 次进程边界**。任何"消息丢了"的问题按这条链逐段查。

### 6.2 为什么需要两道 holdback（具体例子）
假设 stop=`"</answer>"`，模型逐 token 产出 `"</ans"`：
- 没有 stop-prefix holdback：`"</ans"` 会被当成正文吐出去；下一 token `"wer>"` 到达才发现命中 → 但前 6 个字符已经发给客户端，收不回。
- 有它：`"</ans"` 是 stop 的前缀 → 扣住不发；补齐后整体裁掉，客户端只看到 `"</answer>"` 之前的内容。
- UTF-8 同理：一个 4 字节 emoji 被切成 `[前2字节][后2字节]`，第一个 token 单独 `decode` 会产生替换字符。

### 6.3 `PromptAdmittedMsg` 的计费语义
- 它在**首次 prefill 批次准备成功之后**由 scheduler 发出（`scheduler.py:875`），所以即使 prefill 分块，`prompt_tokens` 也是**完整 prompt 长度**。
- `cached_tokens` 是前缀命中数；`PromptAdmittedMsg` 与 `prompt_tokens_delta` 在同一条 `UserReply` 上到达。
- 因此：**prompt 计费不在 tokenize 时发生**。这是读计费/统计代码时最容易搞错的一点。

### 6.4 离线 `generate` 的控制流
```
generate()
  → 重置 pending/status/counter
  → run_forever()                       # Scheduler 主循环（无 ZMQ）
      → offline_receive_msg(blocking)
           有活儿 → 返回 [UserMsg...]
           没活儿且 blocking → raise RequestAllFinished
      → ... 正常调度 ...
      → offline_send_result([DetokenizeMsg...])
  → except RequestAllFinished: pass     # 用异常表达"全部完成"
  → 组装结果
```

---

## 7. 与其他模块的关系

| 方向 | 模块 | 接口 | 说明 |
|---|---|---|---|
| 被使用 | `server/api_server.py` | `FrontendManager` 收发 | 前端侧 |
| 被使用 | `scheduler/io.py` | ZMQ 队列编解码 | 引擎侧 |
| 使用 | `core.SamplingParams` / `Req` | `UserMsg.sampling_params`、`UserMsg.input_ids` 最终变成 `Req` | |
| 使用 | `tokenizer/*` | worker 实现 | 与 `message/` 类型解耦 |
| 使用 | `mm/processor.py` | `mm_processor.apply`（离线路径） | 多模态 |
| 被继承 | `llm.LLM` | 继承 `Scheduler` | 离线 |
| 使用 | `utils.mp` | ZMQ 四件套 | |
| 影响 | `scheduler` | `PromptAdmittedMsg` 计费、`AbortMsg` | |

---

## 8. 常见坑

1. **`message/` 不是 torch-free**（`utils.py` import torch）。真正要求 torch-free 的是 `daemon/`。
2. **方向命名反直觉**：收 `BaseBackendMsg`、发 `BaseTokenizerMsg`。
3. **`message/` vs `tokenizer/`**：类型 vs 实现；同一个 worker 函数两种角色。
4. **UTF-8 holdback 与 stop-prefix holdback 都不能省**，且都是"宁可晚吐也不能撤回"。
5. **prompt 计费在 `PromptAdmittedMsg`，不在 tokenize**。
6. **`__raw_dict__` 不是洁癖**：用户 dict 里合法的 `__type__` 字段会把 decoder 打崩。
7. **离线模式不是另一套调度器**：它只是替换了 I/O 两端。

---

## 9. 自测问题

1. 为什么 `message/`（类型）与 `tokenizer/`（实现）要分开？
2. 同一个 worker 函数怎么同时扮演 tokenize 与 detokenize？
3. `DetokenizeMsg` 为什么携带六项池占用？它们是给谁看的？
4. stop-prefix holdback 与 `matched_stop` 裁剪分别解决什么？
5. 为什么 prompt token 的计费不在 tokenize 阶段？
6. `offline_mode=True` 到底改了什么？为什么它不改变调度行为？
7. 如果一个请求的 chat template 渲染失败，客户端会遇到什么？谁保证它不会挂死？
