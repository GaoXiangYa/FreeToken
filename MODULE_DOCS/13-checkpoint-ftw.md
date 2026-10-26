# 13 · `checkpoint/`：FTW 快速加载格式（转换与读取）

> 范围：`python/freetoken/checkpoint/`（`ftw.py` 742 行、`convert.py` 327 行、`__main__.py` 77 行、`__init__.py` 19 行）+ `docs/ftw-hotfix.md`（61 行）。
> 方法：无 GPU、无 torch 运行，纯静态阅读；行号基于 commit `377a3bc`（`MODULE_DOCS/99-symbol-index/checkpoint.md:6`），关键锚点已用 `sed` 复核。
> 简称：下文 `ftw.py` = `python/freetoken/checkpoint/ftw.py`，`convert.py` = `python/freetoken/checkpoint/convert.py`，`__main__.py` = `python/freetoken/checkpoint/__main__.py`。
> 其余短名按目录唯一匹配：`engine.py` → `python/freetoken/engine/`，`weight.py` → `python/freetoken/models/`，`expert_banks.py` / `host_banks.py` / `offload_cache.py` / `legacy_format.py` → `python/freetoken/moe/`，`cli.py` → `python/freetoken/cli.py`。
> 注：`99-symbol-index/checkpoint.md` 记的行数比 `wc -l` 多 1（743/328/78/20 vs 742/327/77/19），本篇用源码实际行号。

---

## 1. 模块职责与在系统中的位置

`checkpoint/` 位于 L1 模型层（`MODULE_DOCS/00-architecture.md:26`），职责一句话：**HF safetensors → FTW（FreeToken Weight）的离线转换，以及启动时模型无关的回读**（`MODULE_DOCS/00-architecture.md:52`）。它是"离线一次、启动多次"的加载加速层，不是新的权重语义：转换后把 `--model` 直接指向 FTW 目录，引擎靠目录里有没有 `freetoken_weight.json` 自动识别（`ftw.py:105-107`）。

### 1.1 为什么 safetensors 不够用

| 痛点 | 具体表现 | 出处 |
|---|---|---|
| **未对齐** | safetensors 把张量紧密排布、不做 padding，单个张量无法在任意 offset 上 `O_DIRECT`（offset / length / 目的地址三者都必须块对齐） | `ftw.py:5-7` |
| **海量小 I/O** | 旧的 per-bank 缓存原型给每个 expert bank 单独一个文件：只覆盖 expert、不覆盖 dense；而 norm / bias / router 这类长尾小张量会退化成成百上千次小读 | `ftw.py:7-9` |
| **两套路径** | dense 走 per-family 的 `iter_weights`，expert 走 per-bank 缓存，启动要跑两条互不相干的链路 | `ftw.py:18-23` |

### 1.2 FTW 的两个修复

| 修复 | 做法 | 出处 |
|---|---|---|
| **Aligned** | 每个张量起点 4096 对齐、尾部 pad 到 4096；shard 也在 4096 边界切。于是任意张量（或张量在某个 shard 内的切片）的 offset / 向上取整的 length / 目的地址全部块对齐，正好满足 `O_DIRECT`；张量大于 shard 就跨 shard，两端都对齐所以每段仍合法 | `ftw.py:11-17`、`ftw.py:182-189` |
| **Unified** | dense 权重存 `kind="weight"`（就是 `iter_weights` 的输出，post fusion / TP-shard，可直接喂 `load_state_dict`）；卸载 expert 存 `kind="experts_bank"`（后端 repack 之后的 per-expert bank + 用保留名区分的 alpha 向量） | `ftw.py:18-23`、`convert.py:6-10` |

因此转换器只需要**把现有 loader 跑一遍并落盘**，读取侧对模型一无所知（`ftw.py:22-23`、`convert.py:3-5`）。

---

## 2. 文件清单

| 文件 | 行数 | 职责 | 精读/略读 |
|---|---:|---|---|
| `ftw.py` | 742 | 格式常量与检测器、`FTWWriter`、`FTWReader`、`iter_ftw_weights`、`load_ftw_banks` / `load_ftw_banks_to_device` | 精读 |
| `convert.py` | 327 | `convert_checkpoint` 全流程、`_ConvertSink` 流式落盘、metadata 复制、fingerprint | 精读 |
| `__main__.py` | 77 | `ft checkpoint` 的 argparse、GPU 绑定、shard 取整与结果打印 | 精读（短） |
| `__init__.py` | 19 | 包级导出：`FTWReader` / `FTWWriter` / `is_ftw_checkpoint` / `iter_ftw_weights` / `load_ftw_banks` / `convert_checkpoint` | 略读 |
| `docs/ftw-hotfix.md` | 61 | 旧版 FTW 的补救手册：何时重转、何时用 `scripts/ftw_hotfix.py` 原位补条目 | 精读 |

补充：`load_ftw_banks_to_device` 与 `layer_bank_entry_name` **不在**包级导出里，只在 `ftw.py:738-742` 的 `__all__` 中。

---

## 3. FTW 格式规范

### 3.1 分片与磁盘布局

```
<dir>/freetoken_weight.json     # 索引：tensors[] + shards[] + meta
<dir>/freetoken-00000.ftw       # 一条逻辑连续字节区，按 <= shard_limit 物理切片
<dir>/freetoken-00001.ftw
<dir>/config.json, tokenizer*   # 复制过来的 metadata，使目录自包含
```

布局定义见 `ftw.py:25-30`；分片文件名模板 `freetoken-{:05d}.ftw`（`ftw.py:55`）；索引名 `freetoken_weight.json`（`ftw.py:50`）。

| 常量 | 值 | 位置 |
|---|---|---|
| `ALIGN` | 4096（本平台的页大小，即 O_DIRECT 块大小） | `ftw.py:53` |
| `DEFAULT_SHARD_LIMIT` | 8 GiB，必须是 `ALIGN` 的倍数 | `ftw.py:54`、`ftw.py:136` |
| `FORMAT_TAG` / `FORMAT_VERSION` | `"freetoken_weight"` / `1` | `ftw.py:51-52` |
| 读取 chunk 默认 | 8 MiB | `ftw.py:56`、`ftw.py:320-321` |
| bank 读取并发基线 | `_BANK_CONCURRENCY = 4`，实际池宽 16 | `ftw.py:57`、`ftw.py:604` |

### 3.2 4096 对齐不变式

| 不变式 | 保证方式 | 位置 |
|---|---|---|
| 每个张量起点对齐 | `assert global_off % ALIGN == 0`（写前断言） | `ftw.py:182` |
| 每个张量尾部 pad 到 4096 | `pad = _align_up(self._global) - self._global`，pad 字节也写进流 | `ftw.py:186-189` |
| shard 切点对齐 | 只在 `self._cur == shard_limit` 时 roll，而 limit 是 4096 倍数 | `ftw.py:165-171`、`ftw.py:136` |
| CLI 传入的 shard 上限对齐 | `shard_limit -= shard_limit % 4096` | `__main__.py:57-58` |
| 大张量跨 shard 仍合法 | 起点对齐 + 切点对齐 → 每段 offset 对齐；段长是 limit 的余量，也是 4096 倍数 | `ftw.py:11-17`、`ftw.py:158-171` |
| 读长度向上取整 | `rlen = _align_up(length)`，多读的尾部落在本张量的 padding 里，被视图丢弃 | `ftw.py:326`、`ftw.py:213-215` |
| 索引原子可见 | 先写 `freetoken_weight.json.tmp`，再 `os.replace` | `ftw.py:200-203` |
| 短读不吞错 | `preadv` 返回 0（EOF）直接 `OSError`，不让截断的 shard 静默加载垃圾 | `ftw.py:78-86` |

### 3.3 `freetoken_weight.json` 字段

| 字段 | 类型 | 含义 | 位置 |
|---|---|---|---|
| `format` | str | 固定 `"freetoken_weight"`；reader 打开时断言 | 写 `ftw.py:197`，断言 `ftw.py:220` |
| `version` | int | 格式版本，当前 1 | `ftw.py:197` |
| `align` | int | 4096 | `ftw.py:197` |
| `shard_limit` | int | 本次转换的分片上限 | `ftw.py:198` |
| `total_bytes` | int | 逻辑字节区总长（含张量间 padding，即 `_global`） | `ftw.py:142`、`ftw.py:171`、`ftw.py:198` |
| `tensors[]` | list | 每个张量一条，字段见下表 | `ftw.py:199` |
| `shards[]` | list | 每条 `{file, global_off, nbytes}` | `ftw.py:150-151`、`ftw.py:193-194` |
| `source_model_path` | str | 源 checkpoint 绝对路径 | `convert.py:308` |
| `fingerprint` | str/None | `(quant, arch, GPU cc, 各源文件 size+mtime)` 的 sha256 前 16 位 | `convert.py:40-56`、`convert.py:309` |
| `quant_format` | str/None | bank 被 repack 成的实际布局（`nvfp4_marlin` 等），启动时据此选 kernel | `convert.py:310-315`、`ftw.py:117-122` |
| `expert_bank_num_layers` | int/None | bank 的层数，读取时与模型 config 交叉校验 | `convert.py:316-319`、`ftw.py:444-450` |
| `counts` | dict | `{weight, experts_bank}` 条目数（alpha 计入 bank） | `convert.py:320` |
| `copied_metadata` | list | 复制过来的非权重文件相对路径 | `convert.py:69-103`、`convert.py:321` |
| `side_files` | list | `ftw_side_files` 钩子生成的文件（如 PLE 表） | `convert.py:231-238`、`convert.py:322` |

张量条目（`ftw.py:184-185`）：

| 字段 | 含义 |
|---|---|
| `name` | `kind="weight"` 时就是 `load_state_dict` 的 key；`kind="experts_bank"` 时是 bank 名或 `bank#L00007` |
| `kind` | `"weight"` 或 `"experts_bank"` |
| `dtype` | `str(torch.dtype)` 去掉 `torch.` 前缀（`ftw.py:93-94`） |
| `shape` | `list(t.shape)`；0-d scale 存 `[]` |
| `global_off` | 逻辑字节区偏移，4096 对齐 |
| `nbytes` | 未 padding 的原始字节数 |

### 3.4 两类条目与命名规则

| kind | 内容 | 写入方 | 读出方 |
|---|---|---|---|
| `weight` | `load_weight(...)` 的产出：dense 权重（embed / norm / attn / MLP / vision / MTP 等） | `convert.py:223-226`（默认 `add_tensor` 的 kind） | `iter_ftw_weights` 默认只读这一类（`ftw.py:365`） |
| `experts_bank` | 卸载 expert 的 per-expert bank；`gate_up_alpha` / `down_alpha` 虽同 kind，但靠**保留名**区分而不单列 kind | `convert.py:144`、`convert.py:268`、`convert.py:290` | `load_ftw_banks` / `load_ftw_banks_to_device`（`ftw.py:437-442`、`ftw.py:666-668`） |

命名规则：

| 场景 | 规则 | 位置 |
|---|---|---|
| 单层 bank 条目 | `f"{bank_name}#L{layer_id:05d}"` | `ftw.py:64-67` |
| 回解析正则 | `^(?P<base>.+)#L(?P<layer>\d{5})$`（5 位十进制，上限 99999 层） | `ftw.py:61` |
| 扁平布局 | 每 bank 一条，首维 = `num_layers * num_experts`；层数不进 shape，由调用方传 | `ftw.py:486-495`、`ftw.py:544-552` |
| 保留 alpha 名 | `("gate_up_alpha", "down_alpha")` | `ftw.py:58`、`ftw.py:441-442` |
| 旧文件名 → canonical role | `canonical_role(name)`，兼容 `gate_up_packed` 之类旧名 | `ftw.py:630`、`python/freetoken/moe/legacy_format.py:17-18` |

同一个 bank 名不允许同时出现扁平与 per-layer 两种布局（`ftw.py:462-463`）；per-layer 必须凑齐 `range(num_layers)`（`ftw.py:464-468`）。

---

## 4. 写入侧

### 4.1 `FTWWriter` 方法表

| 方法 | 行号 | 做什么 | 关键行为 |
|---|---:|---|---|
| `__init__` | `ftw.py:135` | 建目录、记 shard 上限与运行状态 | 断言 `shard_limit % ALIGN == 0`（`ftw.py:136`）；`_global` 是含 padding 的逻辑偏移（`ftw.py:142`） |
| `_roll` | `ftw.py:148` | 结束当前 shard、开下一个 | 把 `{file, global_off, nbytes}` 追加进 `_shards`；新 shard 的 `global_off = _global`（`ftw.py:150-156`） |
| `_write_raw` | `ftw.py:158` | 把一段字节写进逻辑流，跨 shard 时在 limit 处切开 | 循环 `take = min(n-off, limit-cur)`；三处状态同步递增（`ftw.py:164-171`） |
| `add_tensor` | `ftw.py:173` | 追加一个张量 | 先 `detach().cpu().contiguous()` 再 `view(torch.uint8)`（`ftw.py:174-175`）；**小张量（<= shard）提前 roll**，保证整块落在一个 shard（`ftw.py:177-180`）；写完 pad 到 4096（`ftw.py:186-189`） |
| `finalize` | `ftw.py:191` | 收尾索引并落盘 | 关闭最后一个 shard；索引 = 格式头 + `tensors` + `shards` + `**meta`；`.tmp` + `os.replace` 原子替换（`ftw.py:192-204`），返回索引 dict |

注意 `FTWWriter` 不是线程安全的（`convert.py:117-119`），所以 `_ConvertSink` 用一把锁串行化写入（`convert.py:128`、`convert.py:134`）。

### 4.2 `convert_checkpoint` 完整流程

| 步骤 | 行为 | 位置 |
|---:|---|---|
| 0 | 源已经是 FTW → `SystemExit` | `convert.py:185-186` |
| 1 | TP 检查：未设置就 `rank=0,size=1`；已设置且 `size != 1` → 拒绝（格式不记录 TP 布局，转换单进程） | `convert.py:187-195` |
| 2 | 选 device（默认 `cuda:0`）、`torch.cuda.set_device`、分配一个 0 元素张量以初始化 CUDA context | `convert.py:196-198` |
| 3 | 构造 `EngineConfig`（dtype / moe_strategy / quant_backend），拿到 `model_config` | `convert.py:200-202` |
| 4 | 判定 MoE：`is_moe` 为真则 `include_moe_experts=False`，并强制 `moe_strategy="offload"`、用 meta-device 构建模型取出 `offload_expert_method` | `convert.py:203-213`、`python/freetoken/engine/engine.py:1496-1504` |
| 5 | 开 `FTWWriter(out_dir, shard_limit=...)` | `convert.py:217` |
| 6 | **dense**：逐条消费 `load_weight(model_path, cpu, include_moe_experts=...)`，`add_tensor(kind="weight")`，累计字节发 `FTCONVERT dense` | `convert.py:221-229` |
| 7 | **side files**：若模型模块有 `ftw_side_files` 钩子，把它生成的文件写进 out_dir（不走索引） | `convert.py:231-238`、`python/freetoken/models/deepseek_v41/weight.py:178` |
| 8 | **experts**：先 `legacy_bank_names()` 建 canonical→文件名映射，`_ConvertSink` 挂到 `load_expert_banks(layer_sink=...)` | `convert.py:243-251` |
| 8a | 流式分支（`banks.streamed`）：每层完成即写 `#L%05d` 条目并 `bank.release()`；alpha 作为扁平保留名条目补写 | `convert.py:253-269`、`convert.py:133-154` |
| 8b | 物化分支：把 `banks.sources` 的 per-layer 列表 `torch.cat(dim=0)` 成一条扁平张量再写（转换期每 bank 一次 host RAM 峰值） | `convert.py:270-297` |
| 9 | 复制非权重 metadata（config / tokenizer / remote-code / 嵌套子目录），跳过权重后缀、safetensors 索引与 `.git`/`.cache`/`.freetoken_expert_cache` | `convert.py:299-300`、`convert.py:63-66`、`convert.py:69-103` |
| 10 | 计算 `fingerprint`，失败则置 None | `convert.py:302-305` |
| 11 | `writer.finalize({...})` 写入 `quant_format` / `expert_bank_num_layers` / `counts` / `copied_metadata` / `side_files`，返回索引 | `convert.py:307-324` |

### 4.3 为什么转换是模型无关的

| 论点 | 证据 |
|---|---|
| 不含任何 per-family 分支：dense 只调 `load_weight`，expert 只调 `load_expert_banks` | `convert.py:3-10`、`convert.py:181-182`、`convert.py:223`、`convert.py:249` |
| 存的字节就是 loader 的字节：没有额外融合 / 重排 / dtype 转换（dtype 已在 loader 内完成） | `convert.py:6-10`、`ftw.py:174` |
| 模型知识通过既有 loader 与可选的 `ftw_side_files` 钩子注入，而不是写进转换器 | `convert.py:232-238`、`python/freetoken/models/register.py`（`get_model_spec` / `_load_attr`） |
| 产物 TP 无关：转换强制单进程，索引里不记录 TP layout | `convert.py:177-178`、`convert.py:191-195` |

代价：转换必须在**有 GPU 的机器**上跑（CUDA context + expert repack，`convert.py:196-198`、`convert.py:249-251`），且产物与 `quant_format` 绑定（见第 9 节）。

---

## 5. 读取侧

### 5.1 `FTWReader` 方法表

| 方法 | 行号 | 做什么 | 关键点 |
|---|---:|---|---|
| `__init__` | `ftw.py:217` | 读索引、断言 format tag、按 `global_off` 排序 shards、建 name→entry 字典 | fd 与 mmap 缓存都是惰性字典（`ftw.py:224-225`） |
| `meta` | `ftw.py:236` | 读索引顶层字段 | 默认值可给 |
| `entries` | `ftw.py:239` | 按 kind 过滤张量条目 | 无参数 = 所有 kind |
| `_ensure_mode` | `ftw.py:243` | 探测一次文件系统是否接受 `O_DIRECT` | 只试第一个 shard；失败则 `_direct=0` 并告警（`ftw.py:252-259`），双检锁发布 `_probed`（`ftw.py:246-251`） |
| `_fd` | `ftw.py:262` | 取/开 shard 的 O_DIRECT fd 并缓存 | 只在首次打开时加锁（`ftw.py:265-269`） |
| `_map` | `ftw.py:272` | mmap 整个 shard 并缓存 memoryview | `MADV_SEQUENTIAL` 触发内核预读；映射后立刻 close fd（`ftw.py:278-288`） |
| `close` | `ftw.py:291` | 关 fd、释放并关闭 mmap | 由 `iter_ftw_weights` / `load_ftw_banks` 的 `finally` 调用 |
| `_pieces` | `ftw.py:300` | 把 `[global_off, +nbytes)` 映射成若干 shard 区间 | 断言 shard 连续；gap → `ValueError("FTW gap / misordered shards")`；越界 → `ValueError`（`ftw.py:306-318`） |
| `read_into` | `ftw.py:320` | 把一个张量读进调用方给的 `dest` memoryview | 先 `_ensure_mode`；按 8 MiB 切 job；distinct shard 单线程预开/预映射；`<=1` 个 job 串行，否则 `ThreadPoolExecutor(workers)`（`ftw.py:323-358`） |

### 5.2 `iter_ftw_weights` 生成器语义

| 语义 | 行为 | 位置 |
|---|---|---|
| 惰性 + 预取 | 后台守护线程 `FTW-prefetch` 先读第 n+1 个张量；`prefetch=2` 控制队列深度，峰值 host 内存约 `prefetch+1` 个张量 | `ftw.py:367-371`、`ftw.py:381`、`ftw.py:414-415` |
| 过滤在读之前 | `keep(name)` 为假的条目直接丢掉，**一个字节都不读** | `ftw.py:371`、`ftw.py:379-380` |
| 复用瞬时缓冲 | 每个张量一块匿名 `mmap(-1, align_up(nbytes))`，用 `torch.frombuffer` 零拷贝成张量；消费完即 `del tensor, buf` | `ftw.py:361-362`、`ftw.py:402-407`、`ftw.py:425` |
| 0-d 条目 | 单个 scale 用 `t.view(())` 还原成 0-d，而不是 `[1]` | `ftw.py:406-407` |
| 进度条 | 总量是 `keep` 之后所有条目的 `nbytes` 之和 | `ftw.py:416` |
| 提前 break 不挂死 | `_put` 只轮询 `cancel` 事件、不无限阻塞；`finally` 里 `cancel.set()` + `join()` + `reader.close()` | `ftw.py:386-397`、`ftw.py:426-430` |
| 异常延迟抛出 | producer 把异常收进 `err`，生成器耗尽后再 `raise err[0]` | `ftw.py:409-412`、`ftw.py:431-432` |
| 产出的是 host 张量 | 不 pin、不上 GPU；pin/搬运由消费方负责（`load_weight` 只负责 yield） | `ftw.py:367`、`python/freetoken/models/weight.py:209-243` |
| 默认只读 dense | `kinds=("weight",)`，expert 必须显式指定或走 `load_ftw_banks` | `ftw.py:365` |

### 5.3 `load_ftw_banks` 与 `load_ftw_banks_to_device`

`_split_bank_entries`（`ftw.py:435`）先把 `experts_bank` 条目分成 alpha / 扁平 / per-layer 三组，并用 `expert_bank_num_layers` 与模型 `num_moe_layers` 交叉校验（`ftw.py:444-450`）。之后两条布局走不同路径：

| 布局 | 条目形态 | 分配 | 读取 | 视图 |
|---|---|---|---|---|
| 扁平（旧文件 / 不可流式格式） | 每 bank 一条 `[L*E, ...]` | 每层一个按**对齐窗口** `[align_down(off), align_up(off+len))` 分配的 `HostBank(uint8)`（`ftw.py:555-563`） | 读整窗，窗口起点对齐（`ftw.py:591-596`） | 按 `head_pad` 切片再 `view(dtype).view(E, ...)`（`ftw.py:617-623`） |
| per-layer（可流式格式） | 每 `(bank, layer)` 一条 `#L%05d` | 每层一个形状与条目一致的 `HostBank`（`ftw.py:565-574`） | 断言起点对齐后直接读（`ftw.py:570`、`ftw.py:598-602`） | 直接就是 `[E, ...]`（`ftw.py:618-619`） |

装载细节：

| 主题 | 行为 | 位置 |
|---|---|---|
| 默认驻留 | `layer_residency` 缺省全 `PINNED`；长度必须等于 `num_layers` | `ftw.py:510-511` |
| born-pinned | `born_pinned_default()` 为真且该层是 PINNED 时用 `cudaHostAlloc`，否则惰性 mmap | `ftw.py:513-519`、`python/freetoken/moe/host_banks.py:68-75` |
| 边读边 pin | `PinPipeline` 用一条后台线程串行 `cudaHostRegister`/`mlock`，与剩余读重叠；`_read_*` 只 `submit` | `ftw.py:583-609`、`python/freetoken/moe/host_banks.py:286-300` |
| 并发度 | `min(max(4,16), max(n_jobs,1))` = `min(16, n_jobs)`，这是"大量小读"的优化 | `ftw.py:579-581`、`ftw.py:604` |
| mlock 失败降级 | 请求 LOCKED 但 bank 实际不是 LOCKED → 该层改记 `PAGEABLE`，并按 pinned / locked / pageable 打印 GiB 与层号 | `ftw.py:634-664` |
| alpha | 保持扁平 `[L*E]`，每条目一个 `HostBank`，不受行布局影响 | `ftw.py:532-534`、`ftw.py:666-672` |
| 名字归一 | 文件名 → `canonical_role` → 作为 `ExpertBanks.sources` 的 key | `ftw.py:626-630` |
| kind/kernel | 由 `quant_format` 反查 `LEGACY_FORMAT`，写进 `ExpertBanks` | `ftw.py:631-632`、`python/freetoken/moe/legacy_format.py:29-46` |
| 无 bank | 没有 `experts_bank` 条目时返回 `None`，调用方按 `is not None` 决定是否回落 | `ftw.py:437-439`、`ftw.py:527-529`、`python/freetoken/moe/expert_banks.py:335-340` |

`load_ftw_banks_to_device`（`ftw.py:675-735`）是 resident（fused）expert 的版本：把 alpha / 扁平 / per-layer 统一展开成 job 列表（`ftw.py:689-702`），用一个**单线程** `ThreadPoolExecutor` 预读下一个 `(bank, layer)`（`ftw.py:716-721`），读到 host 后立刻 `.to(device)` 并按 `canonical_role` 收进 `sources[name][layer_id]`（`ftw.py:722-725`）；任一时刻最多"一个预读缓冲 + 一个待搬缓冲"两块瞬时内存，不产生长期 host bank，返回的 `ExpertBanks` 不带 `layer_residency`（`ftw.py:735`）。

### 5.4 O_DIRECT 与 mmap 回退

| 问题 | 结论 | 位置 |
|---|---|---|
| 为什么优先 O_DIRECT | 磁盘 DMA 绕过 page cache，是"快"的选择而非正确性要求 | `ftw.py:226-231` |
| 何时回退 | `os.O_DIRECT` 不存在（非 Linux）或 `open(..., O_DIRECT)` 抛 `OSError`（tmpfs、部分 overlay/网络挂载） | `ftw.py:232`、`ftw.py:252-259` |
| 回退到什么 | 整 shard `mmap` + `MADV_SEQUENTIAL`，**不是**分块 buffered `preadv`：整片映射 + 内核预读比逐块 page-cache 读快得多 | `ftw.py:226-231`、`ftw.py:280-286` |
| 两条路径的读法 | direct：`_pread_into` 循环处理 short read；mmap：边界检查后切片赋值 | `ftw.py:335-351` |
| 线程模型 | 两种路径共用同一套 job 切分与线程池，只是 `rd` 闭包不同 | `ftw.py:331-358` |
| 探测粒度 | 只探测第一个 shard；若同一 FTW 跨了不同挂载点、后续 shard 拒绝 O_DIRECT，读取时才报错（未验证，仅代码推断） | `ftw.py:252-255` |

---

## 6. 函数详解表

| 函数 / 类方法 | 行号 | 作用 | 关键行为 | 调用方 |
|---|---:|---|---|---|
| `layer_bank_entry_name` | `ftw.py:64` | 生成 per-layer bank 条目名 | `#L` + 5 位十进制 | `convert.py:144` |
| `_pread_into` | `ftw.py:70` | 定位读满 memoryview | 循环 short `preadv`；EOF 抛 `OSError` | `ftw.py:339` |
| `_align_up` | `ftw.py:89` | 向上取整到 `ALIGN` | 默认 `a=ALIGN` | `ftw.py:187`、`ftw.py:326`、`ftw.py:362`、`ftw.py:558`、`ftw.py:698` |
| `_dtype_str` | `ftw.py:93` | dtype → 索引字符串 | 去掉 `torch.` 前缀 | `ftw.py:184` |
| `_dtype_of` | `ftw.py:97` | 索引字符串 → dtype | `getattr(torch, s)` | `ftw.py:404`、`ftw.py:533`、`ftw.py:549`、`ftw.py:571`、`ftw.py:689` |
| `_elsize` | `ftw.py:101` | dtype 字节宽度 | 空张量上取 `element_size` | `ftw.py:405`、`ftw.py:550`、`ftw.py:693`、`ftw.py:708` |
| `is_ftw_checkpoint` | `ftw.py:105` | 目录里是否有 FTW 索引 | 只看 `freetoken_weight.json` 是否存在 | `python/freetoken/models/weight.py:226`、`python/freetoken/engine/engine.py:538`、`python/freetoken/moe/expert_banks.py:328`、`convert.py:185` |
| `ftw_tensor_names` | `ftw.py:110` | 列出指定 kind 的张量名 | 无 kind = 全部；只读索引不读数据 | `python/freetoken/engine/engine.py:538`、`python/freetoken/models/weight.py:262`、`tests/checkpoint/test_ftw_weights.py:56` |
| `ftw_quant_format` | `ftw.py:117` | 取 `quant_format` | 非 FTW 返回 None | `python/freetoken/engine/engine.py:1461` |
| `FTWWriter.__init__` | `ftw.py:135` | 建目录、初始化状态 | 断言 shard_limit 对齐 | `convert.py:217`、`tests/checkpoint/test_ftw_weights.py:10` |
| `FTWWriter._roll` | `ftw.py:148` | 滚动 shard 文件 | 记录上一个 shard 的 `{file, global_off, nbytes}` | 内部 `ftw.py:161`、`ftw.py:166`、`ftw.py:180` |
| `FTWWriter._write_raw` | `ftw.py:158` | 逻辑流写入 | 在 limit 处切分并同步三处计数 | `ftw.py:183`、`ftw.py:189` |
| `FTWWriter.add_tensor` | `ftw.py:173` | 追加张量 + padding | CPU/contiguous 化；小张量提前 roll；起点断言对齐 | `convert.py:226`、`convert.py:268`、`convert.py:290`、`convert.py:143` |
| `FTWWriter.finalize` | `ftw.py:191` | 写索引并返回 | `.tmp` + `os.replace` | `convert.py:307` |
| `FTWReader.__init__` | `ftw.py:217` | 打开 FTW | 断言 format tag；shards 排序；tensor 建字典 | `ftw.py:377`、`ftw.py:521`、`ftw.py:681` |
| `FTWReader.meta` | `ftw.py:236` | 读顶层 meta | 带默认值 | `ftw.py:444`、`ftw.py:631`、`ftw.py:730` |
| `FTWReader.entries` | `ftw.py:239` | 按 kind 取条目 | 无 kind = 全部 | `ftw.py:378`、`ftw.py:437` |
| `FTWReader._ensure_mode` | `ftw.py:243` | 决定 direct / mmap | 双检锁 + 首个 shard 探测 | `ftw.py:323` |
| `FTWReader._fd` | `ftw.py:262` | 缓存 O_DIRECT fd | 首次打开加锁 | `ftw.py:331`、`ftw.py:339` |
| `FTWReader._map` | `ftw.py:272` | 缓存整 shard mmap | `MADV_SEQUENTIAL`；映射后关 fd | `ftw.py:331`、`ftw.py:345` |
| `FTWReader.close` | `ftw.py:291` | 释放 fd / mmap | 由各 loader 的 `finally` 保证 | `ftw.py:430`、`ftw.py:525`、`ftw.py:612`、`ftw.py:732` |
| `FTWReader._pieces` | `ftw.py:300` | 逻辑区 → shard 区间 | gap/越界 `ValueError` | `ftw.py:325` |
| `FTWReader.read_into` | `ftw.py:320` | 读一个张量进 `dest` | 8 MiB job、线程池、尾部向上取整 | `ftw.py:403`、`ftw.py:587`、`ftw.py:593`、`ftw.py:600`、`ftw.py:707`、`tests/checkpoint/test_ftw_weights.py:27` |
| `_transient_buffer` | `ftw.py:361` | 匿名对齐缓冲 | `mmap(-1, align_up(n))` | `ftw.py:402`、`ftw.py:706` |
| `iter_ftw_weights` | `ftw.py:365` | 流式回放 dense 权重 | 预取线程 + 队列 + 读前过滤 + 进度条 | `python/freetoken/models/weight.py:227`、`tests/checkpoint/test_ftw_weights.py:28` |
| `_split_bank_entries` | `ftw.py:435` | 切分 bank 条目 | 层数校验、扁平/per-layer 互斥、完整性断言 | `ftw.py:523`、`ftw.py:683` |
| `load_ftw_banks` | `ftw.py:472` | 重建 host `ExpertBanks` | 每层 `HostBank` + `PinPipeline` 边读边 pin + 驻留标签校正 | `python/freetoken/moe/expert_banks.py:332`、`benchmarks/bench_load_weight_generic.py:210` |
| `load_ftw_banks_to_device` | `ftw.py:675` | 重建 resident `ExpertBanks` | 单线程预读 + 逐块 `.to(device)` | `python/freetoken/moe/expert_banks.py:330` |
| `_progress` | `convert.py:35` | 机器可读进度 | 受 `FREETOKEN_CONVERT_PROGRESS==1` 控制 | `convert.py:154`、`convert.py:221`、`convert.py:288` |
| `_source_fingerprint` | `convert.py:40` | 源身份指纹 | quant/arch/cc + 源文件 size/mtime；失败可容忍 | `convert.py:303` |
| `_copy_metadata` | `convert.py:69` | 复制非权重文件 | 单文件 GGUF 走 metadata-only 分支 | `convert.py:300` |
| `_ConvertSink.__init__` | `convert.py:122` | 保存 writer 与名字映射 | `_seen` 防重复层 | `convert.py:248` |
| `_ConvertSink.__call__` | `convert.py:133` | 每层完成即写条目 | 加锁；逐 bank `add_tensor` + `release()`；发字节进度；作为 `layer_sink` 回调被 `LayerCompletionTracker` 触发 | 回调点 `python/freetoken/moe/host_banks.py:379`（在 `expert_banks.py:123` 接上） |
| `_ConvertSink.close` | `convert.py:156` | 关进度条 | 只在流式时创建 | `convert.py:254` |
| `_ConvertSink.num_layers` | `convert.py:161` | 实际见过的层数 | `len(_seen)` | `convert.py:255` |
| `convert_checkpoint` | `convert.py:165` | HF/GGUF → FTW | 见 4.2 全流程 | `__main__.py:60`、`benchmarks/bench_load_weight_generic.py:226` |
| `_parse_quant_backend` | `__main__.py:24` | 校验 `--quant-backend` | 交给 `QuantBackend.parse` | `__main__.py:41` |
| `main` | `__main__.py:34` | CLI 入口 | GPU 绑定、shard 取整、打印统计 | `python/freetoken/cli.py:47` |

---

## 7. 三个关键流程

### 7.1 `ft checkpoint` 全链

```
ft checkpoint                                                     cli.py:18,46-49
  └─ __main__.main(argv)                                          __main__.py:34
       ├─ argparse: --model/--out/--dtype/--moe-backend/
       │            --quant-backend/--shard-gib/--gpu            __main__.py:35-48
       ├─ assign_gpu + bind_assigned_gpu -> cuda:N                __main__.py:51-55
       ├─ shard_limit = int(gib<<30); -= %4096                    __main__.py:57-58
       └─ convert_checkpoint(...)                                 __main__.py:60-63
            ├─ guards: 已是 FTW / TP>1                             convert.py:185-195
            ├─ EngineConfig + model_config                         convert.py:200-202
            ├─ offload? -> meta model -> offload_expert_method     convert.py:203-213
            ├─ FTWWriter(out_dir, shard_limit)                     convert.py:217
            ├─ dense: load_weight -> add_tensor(kind=weight)       convert.py:223-229
            ├─ side files hook                                     convert.py:231-238
            ├─ experts: sink / cat                                 convert.py:243-297
            ├─ _copy_metadata(config/tokenizer/...)                convert.py:300
            ├─ fingerprint                                         convert.py:302-305
            └─ writer.finalize(meta) -> index                      convert.py:307-324
  └─ 打印 tensors / GiB / shards / quant_format / 耗时             __main__.py:65-72
```

### 7.2 启动时识别 FTW

```
Engine.__init__                                                    engine.py:330,334
  └─ _adjust_ftw_quant_backend(model_path, quant_backend)          engine.py:1453
       ├─ ftw_quant_format(path) -> None? 原样返回                  engine.py:1461-1462
       └─ kind_kernel_for(fmt) -> 用户显式选了别的 kernel? 报错      engine.py:1463-1475
Engine._load_weights                                               engine.py:530
  ├─ active_encoders && ftw_lacks_vision -> ValueError             engine.py:531-536
  ├─ resident && 无 experts_bank 条目 -> ValueError                engine.py:537-539
  ├─ load_weight(...) -> is_ftw_checkpoint? -> iter_ftw_weights    weight.py:209,226-227
  │    └─ FTWReader.read_into -> host tensor -> yield               ftw.py:217,320
  └─ model.load_state_dict(...) -> finalize_quant                   engine.py:541-543
```

### 7.3 expert bank 读到 GPU

```
engine 决定 resident / offload                                   engine.py:564-566,693
  └─ load_expert_banks(model_path, ..., layer_residency=...)      expert_banks.py:324
       └─ is_ftw_checkpoint && !dummy                             expert_banks.py:328
            ├─ resident: load_ftw_banks_to_device                  expert_banks.py:330
            │    ├─ _split_bank_entries                            ftw.py:683
            │    ├─ jobs: alpha / flat window / per-layer          ftw.py:689-702
            │    ├─ ThreadPoolExecutor(1) 预读下一个               ftw.py:716-721
            │    └─ host.to(device) -> ExpertBanks                 ftw.py:722-735
            └─ offload: load_ftw_banks                             expert_banks.py:332
                 ├─ 每 (bank, layer) 一个 HostBank                  ftw.py:555-574
                 ├─ read_into(workers=8, chunk=8 MiB)              ftw.py:591-602
                 ├─ pins.submit -> PinPipeline 串行 register/mlock  ftw.py:588-601
                 ├─ 校正 label（LOCKED 失败 -> PAGEABLE）           ftw.py:634-640
                 └─ ExpertBanks(quant_format, sources, alphas,    ftw.py:669-672
                                 layer_residency=applied)
  └─ cache.set_bank_sources(banks.sources, layer_residency=...)   engine.py:751, offload_cache.py:294
       └─ 每 bank 一条 [L, E, ...] -> GPU slot cache 建池          offload_cache.py:344-363
```

---

## 8. 与其他模块的关系

| 模块 / 文件 | 关系 | 锚点 |
|---|---|---|
| `models/weight.py` | `load_weight` 的 FTW 分支：dense 全部来自 `iter_ftw_weights`；`ftw_lacks_vision` 用 `ftw_tensor_names` 判断旧 FTW 缺视觉塔 | `python/freetoken/models/weight.py:209-243`、`:255-262` |
| `models/register.py` | 转换器用 `get_model_spec` / `_load_attr` 发现可选的 `ftw_side_files` 钩子 | `convert.py:232-238` |
| `engine/engine.py` | 启动三道 FTW 相关校验：quant backend 合并、缺视觉塔、resident 层却无 bank；装载后 `set_bank_sources` | `engine.py:334`、`:531-543`、`:751`、`:1453-1475` |
| `engine/config.py` | `EngineConfig(model_path, tp_info, dtype, moe_strategy, quant_backend)` 是转换器复用运行时配置的入口 | `convert.py:200-202` |
| `moe/expert_banks.py` | `load_expert_banks` 在 FTW 目录上优先走 `load_ftw_banks*`；`ExpertBanks` 是两侧的公共契约（`quant_format` / `sources` / alpha / `layer_residency` / `streamed` / `kind` / `kernel`） | `expert_banks.py:31-50`、`:324-340` |
| `moe/host_banks.py` | `HostBank`（页对齐、按需 pin/lock）、`alloc_banks`、`born_pinned_default`、`PinPipeline`；`HostResidency` 三态是 `layer_residency` 的取值域 | `host_banks.py:43-53`、`:68-75`、`:78-160`、`:202-204`、`:286-300` |
| `moe/offload_cache.py` | `set_bank_sources` 校验 bank schema / `layer_residency` 一致性，按 bank 建 GPU slot 池 | `offload_cache.py:294-337`、`:344-363` |
| `moe/legacy_format.py` | bank 名归一（`canonical_role`）、`quant_format` ↔ `(kind, kernel)` 双向表 | `legacy_format.py:17-46` |
| `cli.py` | `ft checkpoint` 子命令只做转发 | `python/freetoken/cli.py:18`、`:46-49`、`:99` |
| `benchmarks/bench_load_weight_generic.py` | `build` 模式调 `convert_checkpoint`，`ftw` 模式用同一 `load_expert_banks` 计时，baseline/parallel/ftw 三方 checksum 必须一致 | `benchmarks/bench_load_weight_generic.py:10-12`、`:210-226`、`:293-313` |
| `scripts/ftw_hotfix.py` + `docs/ftw-hotfix.md` | 不重转整模型的补救：按字节范围补 tensor、改名、补 PLE 表 / 视觉塔；只写需要改的条目 | `docs/ftw-hotfix.md:3-5`、`:24-25`、`:29-35`、`:44-51` |
| `tests/checkpoint/test_ftw_weights.py` | 行为规格：`keep` 在读前丢弃、`ftw_lacks_vision` 判定、`ftw_tensor_names` 顺序 | `tests/checkpoint/test_ftw_weights.py:19-56` |

---

## 9. 常见坑与自测

### 9.1 坑

| # | 坑 | 说明 | 位置 |
|---:|---|---|---|
| 1 | shard 上限没对齐 | `FTWWriter` 直接 `assert shard_limit % 4096 == 0`；CLI 自己会取整，手写脚本调 `convert_checkpoint` 时要自己保证 | `ftw.py:136`、`__main__.py:57-58` |
| 2 | 在无 GPU 的机器上转换 | 转换会建 CUDA context、跑 expert repack；不是纯 CPU 操作 | `convert.py:196-198`、`convert.py:249-251` |
| 3 | 用多 TP 进程转换 | 显式拒绝；FTW 只记录单进程（TP=1）的布局 | `convert.py:191-195` |
| 4 | 把 `--moe-backend` 当开关 | 该参数只为兼容保留；转换总是把 expert 写成 bank（`moe_strategy` 被强制为 `offload`） | `__main__.py:39-40`、`convert.py:203-213` |
| 5 | 旧 FTW 用 `fused` 转换 | expert 存成了 dense 权重，resident 模型启动时报 "keeps its routed experts as dense weights" | `engine.py:537-539` |
| 6 | 旧 FTW 缺视觉塔 | `active_encoders` 时报 `holds no vision encoder tensors`，按 `docs/ftw-hotfix.md` 补或重转 | `engine.py:531-536`、`docs/ftw-hotfix.md:35` |
| 7 | `quant_format` 与 `--quant-backend` 冲突 | 选了与文件不同的 kernel 会**在读 bank 之前**报错；`auto` 则自动采用文件里的 kernel | `engine.py:1463-1475` |
| 8 | 扁平 bank 的层不对齐 | 扁平区里单层的字节范围通常不是 4096 对齐，必须用"对齐窗口 + head_pad"读，不能直接 `read_into` 该层范围 | `ftw.py:486-495`、`ftw.py:555-563`、`ftw.py:617-623` |
| 9 | 层数不匹配 | 索引的 `expert_bank_num_layers` 与 config 的 `num_moe_layers` 不一致 → `RuntimeError` | `ftw.py:444-450` |
| 10 | 混用两种行布局 | 同名 bank 既有扁平又有 `#L` 条目 → `AssertionError`；per-layer 缺层也断言 | `ftw.py:462-468` |
| 11 | 把 `dest` 开小了 | `read_into` 要求 `len(dest) >= align_up(nbytes)`，因为尾块会向上取整读进 padding | `ftw.py:320-326` |
| 12 | 忽略 `keep` 的收益 | `keep` 在**读之前**过滤；视觉塔关闭时能省掉 GB 级 I/O | `ftw.py:371`、`weight.py:224-227` |
| 13 | 拿 0-d scale 当 `[1]` | 索引 `shape=[]` 的条目产出 0-d 张量 | `ftw.py:406-407` |
| 14 | 在 tmpfs / overlay 上期待 O_DIRECT | 会自动降级 mmap 并打 warning，性能曲线完全不同 | `ftw.py:252-259` |
| 15 | 重复转换到同一目录 | `_copy_metadata` 只复制不清理；`FTWWriter` 从头覆盖自己 roll 到的 shard，旧的高编号 shard 不会被索引引用但会留在磁盘上（未验证是否有其他清理路径） | `convert.py:69-103`、`ftw.py:148-156` |
| 16 | 转换中途崩溃 | 索引只在 `finalize` 写；崩溃后目录里可能有 shard 但没有（或只有旧的）`freetoken_weight.json`，重跑即可 | `ftw.py:191-204` |
| 17 | 把 `iter_ftw_weights` 的产出当 pinned | 它是普通 host 张量；pin 是 `load_ftw_banks` 侧 `HostBank` 的职责 | `ftw.py:365-372`、`ftw.py:583-609` |

### 9.2 五个自测问题

1. 为什么 FTW 既要每个张量尾部 pad 到 4096，又要求 shard 切点在 4096 边界？举一个"张量比 shard 大"的例子，说明为什么拆开后每一段仍满足 O_DIRECT 的 offset/length 对齐。（提示：`ftw.py:11-17`、`ftw.py:158-171`、`ftw.py:326`）
2. `FTWWriter.add_tensor` 在什么条件下会**提前** roll？为什么"小张量不跨 shard"值得单独一条分支，而大张量却允许跨？（`ftw.py:173-189`）
3. `iter_ftw_weights` 的 `keep` 为什么能省 I/O？如果消费者在生成器中途 `break`，预取线程如何退出、`reader.close()` 由谁保证？（`ftw.py:371`、`ftw.py:386-397`、`ftw.py:426-430`）
4. 扁平 bank 与 per-layer bank 的读取路径差在哪？为什么扁平布局必须分配"对齐窗口"并记 `head_pad`，而 per-layer 布局不用？（`ftw.py:486-499`、`ftw.py:555-574`、`ftw.py:614-624`）
5. 一个 `quant_format=nvfp4_marlin` 的 FTW，启动时传 `--quant-backend moe.nvfp4=b12x` 会发生什么？在哪一行被拒绝；如果传 `auto` 又会怎样？（`engine.py:1453-1475`、`ftw.py:117-122`、`legacy_format.py:29-37`）
