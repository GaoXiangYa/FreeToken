# 符号索引：`llm`

自动生成（`ast` 解析，不执行代码）。**2 个文件 / 128 行 / 8 个符号**。

> 用途：确认某个模块里**到底有哪些**类/函数/方法及其行号，避免遗漏。
> 行为详解请看同目录下的手写模块文档。
> 重新生成：`python3 MODULE_DOCS/tools/gen_symbol_index.py`

| 文件 | 行数 | 符号数 |
|---|---:|---:|
| `freetoken/llm/__init__.py` | 3 | 0 |
| `freetoken/llm/llm.py` | 125 | 8 |

---

## `freetoken/llm/__init__.py`  (3 行)

_（无类/函数定义，模块级代码）_

## `freetoken/llm/llm.py`  (125 行)

| 行 | 符号 | 说明 |
|---:|---|---|
| 19 | **class `RequestAllFinished`** | class RequestAllFinished(Exception) |
| 24 | **class `RequestStatus`** | class RequestStatus |
| 30 | **class `LLM`** | class LLM(Scheduler) |
| 31 | **class `__init__`** | __init__(self, model_path |
| 47 | **class `_tokenize_one`** | _tokenize_one(self, prompt |
| 53 | **class `offline_receive_msg`** | offline_receive_msg(self, blocking |
| 89 | **class `offline_send_result`** | offline_send_result(self, reply |
| 100 | **class `generate`** | Offline generation; images is aligned with prompts: the raw image files of each prompt in placeholder order, or None. |

