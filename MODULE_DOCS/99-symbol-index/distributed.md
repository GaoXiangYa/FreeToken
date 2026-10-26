# 符号索引：`distributed`

自动生成（`ast` 解析，不执行代码）。**3 个文件 / 147 行 / 20 个符号**。

> 用途：确认某个模块里**到底有哪些**类/函数/方法及其行号，避免遗漏。
> 行为详解请看同目录下的手写模块文档。
> 重新生成：`python3 MODULE_DOCS/tools/gen_symbol_index.py`

| 文件 | 行数 | 符号数 |
|---|---:|---:|
| `freetoken/distributed/__init__.py` | 12 | 0 |
| `freetoken/distributed/impl.py` | 97 | 14 |
| `freetoken/distributed/info.py` | 38 | 6 |

---

## `freetoken/distributed/__init__.py`  (12 行)

_（无类/函数定义，模块级代码）_

## `freetoken/distributed/impl.py`  (97 行)

| 行 | 符号 | 说明 |
|---:|---|---|
| 16 | **class `DistributedImpl`** | class DistributedImpl(ABC) |
| 18 | **class `all_reduce`** | all_reduce(self, x |
| 21 | **class `all_gather`** | all_gather(self, x |
| 25 | **class `TorchDistributedImpl`** | class TorchDistributedImpl(DistributedImpl) |
| 26 | **class `all_reduce`** | all_reduce(self, x |
| 33 | **class `all_gather`** | all_gather(self, x |
| 45 | **class `PyNCCLDistributedImpl`** | class PyNCCLDistributedImpl(DistributedImpl) |
| 48 | **class `all_reduce`** | all_reduce(self, x |
| 52 | **class `all_gather`** | all_gather(self, x |
| 63 | **class `DistributedCommunicator`** | class DistributedCommunicator |
| 66 | **class `all_reduce`** | all_reduce(self, x |
| 69 | **class `all_gather`** | all_gather(self, x |
| 73 | `enable_pynccl_distributed()` | Enable PyNCCL-based distributed communication for tensor parallelism. |
| 93 | `destroy_distributed()` | Destroy all the distributed communication plugins. |

## `freetoken/distributed/info.py`  (38 行)

| 行 | 符号 | 说明 |
|---:|---|---|
| 7 | **class `DistributedInfo`** | class DistributedInfo |
| 11 | **class `__post_init__`** | __post_init__(self) |
| 14 | **class `is_primary`** | is_primary(self) -> bool |
| 21 | `set_tp_info()` | set_tp_info(rank |
| 28 | `get_tp_info()` | get_tp_info() -> DistributedInfo |
| 34 | `try_get_tp_info()` | try_get_tp_info() -> DistributedInfo \| None |

