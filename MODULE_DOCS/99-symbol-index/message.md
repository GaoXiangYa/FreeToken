# 符号索引：`message`

自动生成（`ast` 解析，不执行代码）。**5 个文件 / 421 行 / 35 个符号**。

> 用途：确认某个模块里**到底有哪些**类/函数/方法及其行号，避免遗漏。
> 行为详解请看同目录下的手写模块文档。
> 重新生成：`python3 MODULE_DOCS/tools/gen_symbol_index.py`

| 文件 | 行数 | 符号数 |
|---|---:|---:|
| `freetoken/message/__init__.py` | 44 | 0 |
| `freetoken/message/backend.py` | 97 | 14 |
| `freetoken/message/frontend.py` | 70 | 6 |
| `freetoken/message/tokenizer.py` | 118 | 11 |
| `freetoken/message/utils.py` | 92 | 4 |

---

## `freetoken/message/__init__.py`  (44 行)

_（无类/函数定义，模块级代码）_

## `freetoken/message/backend.py`  (97 行)

| 行 | 符号 | 说明 |
|---:|---|---|
| 13 | **class `BaseBackendMsg`** | class BaseBackendMsg |
| 14 | **class `encoder`** | encoder(self) -> Dict |
| 18 | **class `decoder`** | decoder(json |
| 23 | **class `BatchBackendMsg`** | class BatchBackendMsg(BaseBackendMsg) |
| 28 | **class `ExitMsg`** | class ExitMsg(BaseBackendMsg) |
| 33 | **class `MMItem`** | One multimodal input of a request (an image, video or audio clip). |
| 49 | **class `__getattr__`** | __getattr__(self, name |
| 56 | **class `num_tokens`** | num_tokens(self) -> int |
| 59 | **class `is_image`** | is_image(self) -> bool |
| 62 | **class `validate`** | validate(self) -> None |
| 68 | **class `encoder`** | encoder(self) -> Dict |
| 73 | **class `UserMsg`** | class UserMsg(BaseBackendMsg) |
| 85 | **class `AbortBackendMsg`** | class AbortBackendMsg(BaseBackendMsg) |
| 90 | **class `CacheRebuildBackendMsg`** | class CacheRebuildBackendMsg(BaseBackendMsg) |

## `freetoken/message/frontend.py`  (70 行)

| 行 | 符号 | 说明 |
|---:|---|---|
| 10 | **class `BaseFrontendMsg`** | class BaseFrontendMsg |
| 12 | **class `encoder`** | encoder(msg |
| 16 | **class `decoder`** | decoder(json |
| 21 | **class `BatchFrontendMsg`** | class BatchFrontendMsg(BaseFrontendMsg) |
| 26 | **class `UserReply`** | class UserReply(BaseFrontendMsg) |
| 62 | **class `CacheRebuildReply`** | class CacheRebuildReply(BaseFrontendMsg) |

## `freetoken/message/tokenizer.py`  (118 行)

| 行 | 符号 | 说明 |
|---:|---|---|
| 12 | **class `BaseTokenizerMsg`** | class BaseTokenizerMsg |
| 14 | **class `encoder`** | encoder(msg |
| 18 | **class `decoder`** | decoder(json |
| 23 | **class `BatchTokenizerMsg`** | class BatchTokenizerMsg(BaseTokenizerMsg) |
| 28 | **class `DetokenizeMsg`** | class DetokenizeMsg(BaseTokenizerMsg) |
| 54 | **class `PromptAdmittedMsg`** | Scheduler -> tokenizer accounting signal for an admitted prompt. |
| 69 | **class `TokenizeMsg`** | class TokenizeMsg(BaseTokenizerMsg) |
| 80 | **class `AbortMsg`** | class AbortMsg(BaseTokenizerMsg) |
| 85 | **class `CacheRebuildMsg`** | class CacheRebuildMsg(BaseTokenizerMsg) |
| 96 | **class `CacheRebuildResultMsg`** | class CacheRebuildResultMsg(BaseTokenizerMsg) |
| 108 | **class `ErrorReplyMsg`** | class ErrorReplyMsg(BaseTokenizerMsg) |

## `freetoken/message/utils.py`  (92 行)

| 行 | 符号 | 说明 |
|---:|---|---|
| 15 | `_serialize_any()` | _serialize_any(value |
| 29 | `serialize_type()` | serialize_type(self) -> Dict |
| 53 | `_deserialize_any()` | _deserialize_any(cls_map |
| 70 | `deserialize_type()` | deserialize_type(cls_map |

