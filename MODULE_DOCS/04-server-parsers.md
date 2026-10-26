# 04 · server parsers：工具调用解析与思维链解析（流式标记切分）

> 行号基于 `377a3bc`。完整符号清单见 [`99-symbol-index/server.md`](99-symbol-index/server.md) 的两个 parser 段落。

---

## 1. 模块职责与在系统中的位置

很多模型把**工具调用**和**思维链**写成**带内标记**混在正文里：

```
好的，我来查一下天气。<tool_call>{"name":"get_weather","arguments":{"city":"北京"}}</tool_call>
嗯，让我想想……<think>用户问的是天气，需要调用工具</think>
```

服务端不能把这些标记原样吐给客户端。本模块负责在**流式分块边界**上把标记与可见正文切开，并解析成结构化的 tool call / reasoning_content。

**难点不在正则，而在三件事**：
1. **跨 chunk**：标记可能被切在任意两个 token 之间（甚至 `bot_token` 本身被切成两半）。
2. **部分 JSON**：参数在流式到达时永远是不完整 JSON，必须先增量吐出"前缀稳定的片段"。
3. **绝不改写正文长度**：解析只做"归类与切分"，不能凭空增删字符，否则前端的增量拼接会错位。

- **上游**：`server/generation.py` 每个请求构造一对 parser（`_make_tool_parser:436`、`_make_reasoning_parser:385`）。
- **下游**：三种协议适配器把解析结果格式化成自己的 wire 格式（OpenAI `tool_calls` / Anthropic `tool_use` / Responses `function_call`）。
- **选型**：`server/args.py` 的 `--tool-call-parser auto` / `--reasoning-parser auto`，按架构名推断（测试 `tests/server/test_parser_auto_selection.py` 钉住）。

```
token 流
  │
  ├─▶ ReasoningParser  ──▶ (reasoning_content 增量, content 增量)
  │
  └─▶ FunctionCallParser ──▶ ToolCallItem 增量（name + args 片段）
                                │
                                ▼
                     三种协议的 wire 格式
```

---

## 2. 文件清单

| 文件 | 行数 | 职责 | 阅读建议 |
|---|---:|---|---|
| `function_call_parser.py` | 3,756 | 15 个工具调用 detector + 基类 + 门面类 | **先读基类契约（242-660）**，再读 1-2 个小 detector，其余按需 |
| `reasoning_parser.py` | 929 | 7 个思维链 parser + 基类 + 门面类 | 同上，基类 90-258 |
| `generation.py`（相关段） | — | 每请求构造 parser、驱动两级解析、输出事件 | 读 385-470 |

---

## 3. 核心抽象：`BaseFormatDetector`（`function_call_parser.py:242`）

### 3.1 契约（子类可覆写）

| 成员 | 行 | 契约 |
|---|---:|---|
| `bot_token` / `eot_token` | 类属性 | 工具块的开/闭标记 |
| `__init__` | 264 | 初始化流式状态 |
| `_get_tool_indices(tools)` | 288 | 工具名 → 索引映射（用于校验与 `tool_call_id`）|
| `parse_base_json(result, tools)` | 304 | 把解析出的 JSON 转成 `ToolCallItem` |
| `detect_and_parse(text, tools)` | 331 | **一次性**解析：`success=True/False` |
| `_ends_with_partial_token(buffer, token)` | 339 | 缓冲区是否以某个标记的**前半截**结尾（跨 chunk 的关键） |
| `_get_param_config(tools, func_name, param_name)` | 352 | 取某参数的 JSON schema |
| `_convert_param_value(value, param_config, func_name)` | 363 | 按 schema 类型转换（字符串→数字/布尔/对象），**安全的 `eval` 替代** |
| `_schema_param_type` / `_json_escape_chunk` | 399 / 409 | schema 类型归一 / 逐字符 JSON 转义 |
| `block_close_tokens()` | 414 | 结束工具块的标记集合（用于定位"最后一个 call 之后的文本"） |
| `finish_streaming()` | 419 | **流结束时的排空**：返回还缓冲着的残余文本 |
| `parse_streaming_increment(new_text, tools)` | 436 | **流式增量**：返回 `StreamingParseResult` |

### 3.2 三个能力标记（决定上层怎么用它）

| 标记 | 含义 | 影响 |
|---|---|---|
| `supports_streaming` | 是否支持逐 chunk 增量 | 否则退化为"攒完再解析"（首 token 延迟变差） |
| `args_fragments_prefix_stable` | 已吐出的参数片段是否为**最终值的前缀** | 若为真，前端可以安全地流式显示部分参数 |
| `toolcall_opener` | 该格式的工具调用开头 token | 服务端用它做**语义锚点**（`Req.toolcall_anchor_len`，见 `01`）：客户端重写 tool call 后仍可复用前缀 KV |

### 3.3 `InvokeParamStreamMixin`（`:660`）
invoke/parameter 风格格式（`qwen3_coder`、DSML 系、minimax、muse 等）的**值级流式**共享实现：
`_ps_reset`（`:683`）重置状态、`_ps_convert_value`（`:691`）值转换、`_ps_canonical_name`（`:696`）函数名归一钩子、`_ps_trim_leading`/`_ps_trim_trailing`（`:701/706`）去空白、`parse_streaming_increment`（`:711`）、`finish_streaming`（`:902`）。
**它解决的是同一类问题**：格式是"参数块"而非整段 JSON，所以要按参数逐个吐。

### 3.4 `BaseReasoningParser`（`reasoning_parser.py:90`）
契约与工具调用类似但更简单：进入/退出思考态、返回 `(reasoning_delta, content_delta)`。`ReasoningParseResult`（`:83`）是结果容器。
关键语义：**未闭合的思考块在流结束时怎么处理**（丢弃还是当正文吐）、**特殊 token 是否保留**。

---

## 4. 解析器全景表

### 4.1 工具调用（15 个 detector）

| Detector | 行 | CLI 名（`ToolCallParserEnum`） | 对应标记 |
|---|---:|---|---|
| `Qwen25Detector` | 913 | `qwen` / `qwen25` | `<tool_call>{...}</tool_call>` |
| `MistralDetector` | 1008 | `mistral` | `[TOOL_CALLS][{...}]` |
| `Llama32Detector` | 1112 | `llama3` | `<|python_tag|>{...}` |
| `Glm47Detector` | 1177 | `glm47` | XML 风格 `<arg_key>/<arg_value>` |
| `DeepSeekV32Detector` | 1542 | `deepseekv32` | DSLM `<｜DSML｜invoke ...>` |
| `Qwen3CoderDetector` | 1896 | `qwen3_coder` | `<function=name>...</function>` |
| `Gemma4Detector` | 2061 | `gemma4` | Gemma 专用（`_parse_gemma_value` 等辅助在 206-240） |
| `MiniMaxDetector` | 2351 | `minimax` | `<minimax:tool_call>` |
| `MiniMaxM3Detector` | 2416 | `minimax_m3` | `]<]minimax[>[<tool_call>` |
| `GptOssDetector` | 2883 | `gpt-oss` / `gpt_oss` | harmony `<|channel|>` |
| `MuseGlimmerDetector` | 3071 | `muse_glimmer` | muse 专用 |
| `DeepSeekV41Detector` | 3525 | `deepseekv41` | DSV4.1 |

（另有 `ToolCallItem`（`:87`）与 `StreamingParseResult`（`:95`）两个数据结构。）

### 4.2 思维链（8 个 CLI 名 → 7 个类）

| Parser | 行 | CLI 名 |
|---|---:|---|
| `ThinkReasoningParser` | 394 | `qwen3` / `glm` / `minimax`（`<think>` 系共用） |
| `DeepSeekV32ReasoningParser` | 259 | `deepseekv32` |
| `GptOssHarmonyReasoningParser` | 278 | `gpt_oss`（harmony channel） |
| `MiniMaxM3ReasoningParser` | 415 | `minimax_m3` |
| `GemmaThoughtReasoningParser` | 535 | `gemma4` |
| `MuseGlimmerReasoningParser` | 627 | `muse_glimmer` |

### 4.3 标记 → 归属（`TOOLS_TAG_LIST`，`:67`）
`<|plugin|>`、`<|tool_call>`、`<|tool_call_begin|>`、`<|channel|>`、`<function=`、`<tool_call>`、`<minimax:tool_call>`、`]<]minimax[>[<tool_call>`、`<|python_tag|>`、`[TOOL_CALLS]`、`<｜DSML｜function_calls>`、`<｜DSML｜tool_calls>`、`<｜DSML｜invoke`、`<｜DSML｜ calls>`、`<｜DSML｜ invoke`、`<atem:function_calls>`。
`grep -n "<你的标记>" TOOLS_TAG_LIST` 就能知道它归哪个 detector。

### 4.4 门面类

| 类 | 行 | 作用 |
|---|---:|---|
| `FunctionCallParser` | 3535 | 按名字构造 detector；提供 `has_tool_call` / `detect_and_parse` / `parse_streaming_increment` / `finish_streaming` 的统一入口；`ToolCallParserEnum` 就在它内部（`:3544-3559`） |
| `ReasoningParser` | 873 | 同上；`ReasoningParserEnum` 在 `:876-885`；`build_reasoning_parser`（`:916`）负责构造 |

`SUPPORTED_TOOL_CALL_PARSERS`（`:3727`）与 `SUPPORTED_REASONING_PARSERS`（`reasoning_parser.py:929`）把 enum 的 key 暴露给 CLI 校验。

---

## 5. 函数详解（除 detector 内部实现外的公共部分）

### 5.1 `function_call_parser.py` 的模块级工具

| 函数 | 行 | 作用 |
|---|---:|---|
| `_should_forward_unknown_tool(name)` | 63 | 未知工具是否放行：`FORWARD_UNKNOWN_TOOLS` 或名字含 `:`（MCP 风格 `server:tool`） |
| `_first_existing_pos` / `_find_common_prefix` | 103 / 108 | 找最早出现位置 / 两串公共前缀（增量输出用） |
| `_partial_json_loads` | 119 | **解析不完整 JSON**（流式核心） |
| `_is_complete_json` | 148 | 是否已完整 |
| `_parse_loose_json_value` / `_parse_first_json_value` | 156 / 169 | 宽松解析单个值 / 只取第一个值 |
| `_split_top_level` | 182 | 顶层分割（跳过嵌套括号内的分隔符） |
| `_parse_gemma_value` / `_parse_gemma_call_args` | 206 / 230 | Gemma 专用解析 |

### 5.2 `BaseFormatDetector` 的关键实现
- `parse_streaming_increment`（`:436-658`）是**最大的一个方法**：它维护缓冲、判断"标记是否可能被截断"（用 `_ends_with_partial_token`）、只在能确定边界时吐片段，否则继续攒。
- `detect_and_parse`（`:331`）是一次性路径，通常被非流式请求或有完整文本时使用。
- `_convert_param_value`（`:363`）按 schema 做类型转换——这是"模型吐字符串、但 schema 要求 number/boolean"时的兼容层。

### 5.3 `generation.py` 的驱动点（只引用）

| 函数 | 行 | 作用 |
|---|---:|---|
| `_make_reasoning_parser(spec, state)` | 385 | 每请求构造 reasoning parser |
| `_split_reasoning(text, spec, state)` | 421 | 非流式路径的整段切分 |
| `_make_tool_parser(spec, state)` | 436 | 每请求构造 tool parser |
| `_parse_tool_response(text, ...)` | 448 | 非流式路径的整段解析 |
| `_valid_json(text)` | 464 | 校验 |
| `build_reasoning_parser(...)` | `reasoning_parser.py:916` | 实际构造函数 |

---

## 6. 关键流程逐段解析

### 6.1 流式两级解析的执行顺序
```
每个 chunk 到来：
  1) reasoning_parser.parse_streaming_increment(chunk)
       → 产出 (reasoning_delta, content_delta)
       # 思考态内：文本进 reasoning_delta；思考态外：进 content_delta
  2) content_delta 再交给 tool_parser.parse_streaming_increment(content_delta)
       → 产出 ToolCallItem 增量
  3) 适配器把两类增量格式化成各自的 wire 事件
```
**顺序不能反**：工具标记可能出现在思考块内部（被 reasoning 吞掉），所以必须先做 reasoning 切分。

### 6.2 跨 chunk 的工具调用增量
```
缓冲累积文本
  → _ends_with_partial_token(buf, bot_token)?    # 标记被切成两半 → 继续攒
  → 找到完整 bot_token 后开始"在块内"状态
  → 块内：_partial_json_loads 反复尝试；能确定的部分参数立即吐出
  → 找到 eot_token → 块结束 → parse_base_json 产出完整 ToolCallItem
  → 流结束：finish_streaming() 吐出残余缓冲
```
`args_fragments_prefix_stable` 标记告诉上层："我吐的部分参数是最终值的前缀"，前端才能安全地边收边渲染。

### 6.3 解析失败/未知工具的回退
- 未知工具名：`_should_forward_unknown_tool`（`:63`）决定放行还是丢弃；放行由 `FREETOKEN_FORWARD_UNKNOWN_TOOLS` 控制，或名字含 `:`（MCP）。
- 格式不匹配：`detect_and_parse` 返回 `success=False`，正文原样输出（不吞字符）。
- **绝不静默丢字符**是这一层的硬要求：解析器只决定"这段属于正文还是参数"。

### 6.4 auto 推断
`--tool-call-parser auto` / `--reasoning-parser auto`：`server/args.py` 里按 checkpoint 架构名映射到 parser 名；`tests/server/test_parser_auto_selection.py` 遍历活注册表，保证新增模型族不会静默失去解析能力。

### 6.5 为什么 opener token 会影响缓存
detector 的 `toolcall_opener` 被服务端用来在模型吐出 tool-call 开头 token 时打**语义锚点**（`Req.toolcall_anchor_len`）。客户端通常会重写这段 tool call 再回传，锚点让"重写点之前"的前缀 KV 仍然可复用（见 `01` 的 `toolcall_anchor_len` 与 `02` 的 `snapshot_toolcall_anchor`）。

---

## 7. 与其他模块的关系

| 方向 | 模块 | 接口 | 说明 |
|---|---|---|---|
| 被驱动 | `server/generation.py` | 构造 + 每 chunk 调用 | 唯一调用方 |
| 选型 | `server/args.py` | auto 推断 | 架构名 → parser 名 |
| 依赖 | `core.SamplingParams` 无关；依赖 tokenizer 的 special token | | opener token 来自 tokenizer |
| 影响 | `scheduler` / `kvcache` | 通过 `toolcall_opener` → 语义锚点 | 间接 |
| 被消费 | `openai_api` / `anthropic_api` / `responses_api` | 把 `ToolCallItem` 转成各自 wire 格式 | |
| 测试 | `tests/server/test_function_call_parser.py`、`test_reasoning_parser_all_models.py`、`test_parser_auto_selection.py` | 遍历活注册表 | |

---

## 8. 如何只读一个 parser（省时读法）

3,756 行里绝大多数是**同构的重复**。按这 6 步读，1-2 小时能掌握：
1. `BaseFormatDetector`（`:242-660`）——契约与两个核心方法（`detect_and_parse` / `parse_streaming_increment`）。
2. `Qwen25Detector`（`:913-1005`）——**最干净的 detector**，对照基类看它只覆写了什么。
3. `MistralDetector`（`:1008`）——第二个样本，看差异。
4. `InvokeParamStreamMixin`（`:660`）——理解"参数块"族为什么需要值级流式。
5. `DeepSeekV32Detector`（`:1542`）——最复杂的 DSML 族，看它如何用 `block_close_tokens` 与 `_param_fragment`。
6. `FunctionCallParser`（`:3535-3724`）——门面类与 enum。
**可以跳过**：其余 detector 的内部实现，除非你要修某个具体模型。

---

## 9. 常见坑

1. **跨 chunk 边界**：`bot_token` 本身可能被切开，必须用 `_ends_with_partial_token` 攒。
2. **部分 JSON 是常态**，不是异常：流式路径永远在处理不完整数据。
3. **解析器不改写正文长度**：只归类，不增删字符。
4. **顺序**：先 reasoning 后 tool-call。
5. **未知工具默认可能被丢弃**：注意 `FREETOKEN_FORWARD_UNKNOWN_TOOLS` 与 MCP 的 `server:tool` 命名。
6. **一个 parser 名可对应多个类 / 多个名可对应一个类**（如 `qwen`/`qwen25` 都是 `Qwen25Detector`）——别假设一对一。
7. **非流式与流式是两条代码路径**（`detect_and_parse` vs `parse_streaming_increment`），修 bug 要两条都看。

---

## 10. 自测问题

1. 为什么必须先做 reasoning 切分、再做 tool-call 切分？
2. `_ends_with_partial_token` 解决的具体场景是什么？不做会怎样？
3. `args_fragments_prefix_stable` 这个能力标记是给谁用的、用来做什么决策？
4. `InvokeParamStreamMixin` 解决的是哪一类格式的问题？举两个使用它的 detector。
5. 一个未知工具名到达时，服务端有哪几种处理可能？
6. `toolcall_opener` 为什么会影响前缀缓存命中？
