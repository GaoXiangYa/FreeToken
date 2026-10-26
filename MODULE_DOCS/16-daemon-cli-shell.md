# 16 · daemon / shell / CLI：控制面、终端 UI、命令分派

> 行号基于 `377a3bc`。完整符号清单见 [`99-symbol-index/daemon.md`](99-symbol-index/daemon.md)、[`shell.md`](99-symbol-index/shell.md)、[`top-level.md`](99-symbol-index/top-level.md)。

---

## 1. 模块职责与在系统中的位置

这里有**三条互不相干的线**，常被误当成一个东西：

| 线 | 组件 | 是什么 | 端口 |
|---|---|---|---|
| ① 命令分派 | `cli.py` + 6 个 runner | 130 行的纯分发表，`ft <verb>` → 具体模块 | — |
| ② 控制面 | `daemon/` | **torch-free 监督者**：管 `ft serve` 的生命周期、日志、记账 | 1900 |
| ③ 终端 UI | `shell/` | TUI 客户端（attach 到已有服务，或自带引擎） | — |

**关键边界**：daemon 管的是**引擎进程**，不是推理请求。推理流量由客户端**直连** `ft serve` 的 1919 端口，不经过 daemon。

```
client (ft ctl / curl / 桌面端)              推理流量 → 直连 serve:1919
        │ HTTP 控制面 :1900                          │
        ▼                                             ▼
   ft daemon ──spawn / signal / tail──▶  ft serve (模型 · 推理 · 可能崩)
   (不 import torch)                       └─ /health /v1/stats (自己的控制 API)
        ▲
   systemd   Restart=always · KillMode=process
```

另外两个顶层文件：`launch.py`（`ft launch`：给 6 个 coding agent 写 provider 配置）、`gpu_select.py`（`--gpu` 的 UUID 解析）。

---

## 2. 文件清单

### 2.1 daemon（16 文件 / 3,057 行）

| 文件 | 行数 | 职责 | 阅读建议 |
|---|---:|---|---|
| `README.md` | 95 | **设计参考**，全读 | **精读** |
| `serve_manager.py` | 913 | 监督一个 `ft serve` 子进程：start/stop/switch/reap/重认领 | **精读** |
| `app.py` | 509 | FastAPI 控制面路由（17 条） | 精读（路由表） |
| `server.py` | 231 | 守护进程入口：单实例锁、装配、detach 策略 | 精读 |
| `osproc.py` | 198 | `/proc` 辅助：进程组、starttime、oom_score_adj、PSS、argv 里的端口 | 概览 |
| `client.py` | 198 | `ft daemon <verb>` 的 HTTP 客户端 | 概览 |
| `checkpoint.py` | 174 | 受监督的 `ft checkpoint` 子进程 | 略读 |
| `accounting.py` | 160 | 持久化的"最终记账"收据 outbox | 精读 |
| `proxy.py` | 129 | `ServeProbe`：代理 `/health`、`/v1/stats`、`/v1/admin/prepare-stop` | 精读 |
| `pidfile.py` | 107 | `flock` 单实例 + `serve.json` 状态 | 精读 |
| `metrics.py` | 134 | RAM(PSS) + VRAM 足迹 | 略读 |
| `tailer.py` | 117 | 跟踪日志文件 | 略读 |
| `logfmt.py` / `logring.py` | 76 / 67 | ANSI/`\r` 归一化 / 环形缓冲 + SSE 回放 | 略读 |
| `version.py` / `__main__.py` / `__init__.py` | 8 / 5 / 31 | 版本 / 模块入口 / 动词分派 | 略读 |
| `ft-daemon.service` | 41 | systemd unit | 精读（很短） |

### 2.2 shell（5 文件 / 1,515 行）

| 文件 | 行数 | 职责 |
|---|---:|---|
| `tui.py` | 752 | 输入处理、内置命令、流式渲染主循环 |
| `render.py` | 350 | 流式输出渲染（markdown/工具块） |
| `client.py` | 327 | HTTP 客户端（流式消费） |
| `__init__.py` | 82 | `main`：两种模式分流 |
| `__main__.py` | 4 | `python -m` |

### 2.3 顶层 CLI 与平台（2,513 行）

| 文件 | 行数 | 职责 |
|---|---:|---|
| `launch.py` | 1,072 | `ft launch`：6 个 agent 的配置生成 + 安装 + 启动 |
| `control_cli.py` | 481 | `ft ctl` 的 HTTP 子命令 |
| `gpu_select.py` | 287 | `--gpu` 的 UUID ↔ 索引解析与绑定 |
| `cache_report.py` | 218 | 把 `/v1/cache/status` 渲染成人读的表格 |
| `env.py` | 94 | `EnvClassSingleton`：类属性 → `FREETOKEN_<NAME>` |
| `cli.py` | 130 | `ft` 的分发表 |

---

## 3. `ft` CLI

### 3.1 COMMANDS 表（`cli.py:93-101`）

| 子命令 | runner | 行 | 实际入口 |
|---|---|---:|---|
| `serve` | `_run_serve` | 27 | `server/launch.py:125 launch_server` |
| `shell` | `_run_shell` | 34 | `shell/__init__.py:51 main` |
| `launch` | `_run_launch` | 40 | `launch.py:1013 main` |
| `checkpoint` | `_run_checkpoint` | 46 | `checkpoint/__main__.py` |
| `ctl` | `_run_ctl` | 52 | `control_cli.py:352 main` |
| `daemon` | `_run_daemon` | 58 | `daemon/__init__.py:14 main` |
| `bench` | `_run_bench` | 76 | `moe/benchbw.py` 的入口 |

`main`（`cli.py:104`）只做三件事：处理 `-h/-V`、查 `COMMANDS`、`globals()[runner_name](args[1:])`。**没有任何逻辑**——这是故意的。

### 3.2 六个命令的调用链

| 命令 | 链 |
|---|---|
| `ft serve` | `cli:104` → `:27` → `server/launch.py:125` → `args.parse_args` → `resolve_gpu_uuids` → `start_subprocess`（spawn 各族）→ `api_server.run_api_server` |
| `ft shell` | `cli:34` → `shell/__init__.py:51`；带 `--model` → `launch_server(run_shell=True)`；否则纯 HTTP TUI |
| `ft launch` | `cli:40` → `launch.py:1013 main` → `parse_argv:828` → `resolve_server_url:146` / `discover_server_model:196` → `PREPARERS[agent]:818` → `ensure_agent_installed:954` → `run_command:1005`（**不启动引擎**） |
| `ft ctl` | `cli:52` → `control_cli.py:352` → 对 1919 发 HTTP |
| `ft daemon` | 带动词 → `daemon/__init__.py:14` → `client.py:161`（打 1900）；不带动词 → `server.py:104` → 单实例锁 → `ServeManager` → uvicorn |
| `ft checkpoint` | `cli:46` → `checkpoint/__main__.py` → `convert_checkpoint`（见 `13`） |

### 3.3 `gpu_select.py`（287 行）
`--gpu` 接受 **`nvidia-smi` 索引**或 **UUID 前缀**。`resolve_gpu_uuids` 把用户输入解析成完整 UUID 列表（TP rank 顺序），`bind_assigned_gpu(rank)` 在各 worker 里按 UUID 绑定设备。
为什么要 UUID：多卡机器上 CUDA 的枚举顺序与 `nvidia-smi` 不一定一致，用 UUID 才是稳定的。

### 3.4 `env.py`（94 行）
`EnvClassSingleton` 的魔法：类属性写成 `EnvInt(2048)` / `EnvBool(False)` / `EnvMem(...)` / `EnvStr(...)` / `EnvOption()`，单例初始化时遍历所有属性、按 `FREETOKEN_<ATTR_NAME>` 读环境变量并解析。文档里 `ENV.DISABLE_OVERLAP_SCHEDULING` 实际读的是 `FREETOKEN_DISABLE_OVERLAP_SCHEDULING`。
**注意**：另有一批环境变量是散落的 `os.getenv("FREETOKEN_...")`（共 53+ 个），不在这个单例里。

---

## 4. daemon（控制面）

### 4.1 为什么必须 torch-free（唯一不可协商的约束）

`daemon/` 只 import **stdlib + fastapi + uvicorn**（+ 少量 `/proc` 读取 + 可选 pynvml）。它**绝不** import `torch` / CUDA / flashinfer / `sgl_kernel`，也不 import `freetoken.server.*`（其 `__init__` 会拉 torch）或 `freetoken.utils.*`（会拉 transformers）。

理由：**CUDA 故障或原生扩展段错误会杀掉加载它的进程**——而 daemon 一个都不加载，所以它"崩不了"。所有有风险的活都在隔离的 `ft serve` 子进程里。

强制手段：`tests/daemon/test_daemon_import_safety.py` 用**子进程 + meta-path 拦截器**验证（同进程检查会假通过）。

### 4.2 启动

```
daemon/server.py:104 main
  → :124 SingleInstance.acquire()        # flock pidfile，保证最多一个 daemon
  → :143 ServeManager(...)
  → :164 manager.readopt()               # 重新认领上次留下的 serve
  → :207 uvicorn.run(...)                # 127.0.0.1:1900
```
状态目录默认 `~/.freetoken/daemon`（`$FREETOKEN_DAEMON_DIR`）：`daemon.pid`、`serve.json`、`logs/`、`accounting-outbox/`。

### 4.3 HTTP API 全表（17 条，`daemon/app.py`）

| 方法 + 路径 | 行 | 作用 |
|---|---:|---|
| `GET /health` | 181 | daemon 自身健康（**永不鉴权、永不失败**） |
| `POST /engine/start` | 193 | `{model, port, args[]}`；同一 `(model,port,args)` 幂等；同端口不同配置 → 409 |
| `POST /engine/stop` | 212 | 关准入 → drain/abort → 持久化最终记账 → SIGTERM→grace→SIGKILL |
| `POST /shutdown` | 219 | 关掉 daemon 自身 |
| `POST /engine/switch` | 237 | 一次串行的 stop-记账-start 事务 |
| `GET /accounting/pending` | 256 | 未确认的最终记账收据（客户端崩溃后可重放） |
| `POST /accounting/ack` | 264 | 幂等删除已应用的收据 |
| `GET /engine/status` | 273 | `{running,pid,model,port,uptimeS,lastExitCode,...}`（比任何单个 serve 活得久） |
| `GET /engine/metrics` | 277 | `{ramBytes,vramBytes}`（只算 serve 进程树） |
| `GET /engine/health` | 282 | 代理 serve 的 `/health` + daemon 可达性 |
| `GET /engine/stats` | 299 | 代理 serve 的 `/v1/stats` |
| `GET /engine/logs` | 309 | SSE 日志：去 ANSI、折叠 `\r`、环形回放、`id:<seq>`、`Last-Event-ID` 续传 |
| `POST /checkpoint/start` / `cancel` / `GET /checkpoint/status` | 317 / 328 / 332 | 受监督的 `ft checkpoint`（GPU 独占：先停 serve） |
| `POST /bench/run` / `GET /bench/profile` | 338 / 383 | 受监督的 `ft bench bw` |

鉴权：`--token`（或 `$FREETOKEN_DAEMON_TOKEN`）要求除 `/health` 外所有请求带 `X-FT-Token`。

### 4.4 `ServeManager` 的状态机

| 动作 | 关键点 |
|---|---|
| `start`（`:258`） | `build_serve_command`（`:120`）拼 `python -m freetoken.cli serve ...`；`_default_spawn`（`:235`）用 `Popen(..., start_new_session=True)` 让子进程成为**会话/进程组组长**（这样 killpg 能覆盖它的 mp worker）；日志写 `logs/serve-<port>.log` |
| `stop`（`:327`） | 先 `_prepare_and_persist_accounting`（`:449` → proxy 调 `/v1/admin/prepare-stop`）→ `signal_group` SIGTERM → grace → SIGKILL |
| `_begin_watch` / `_monitor` / `_reap`（`:748/758/766`） | 单一 reaper 线程，避免多线程 wait 竞争 |
| 重认领（`:721-744`） | 读 `serve.json` + 用 **starttime + argv** 校验 PID 身份（防 PID 复用），重新接管仍在跑的 serve |

### 4.5 自保策略（daemon 存在的意义）

| 策略 | 机制 |
|---|---|
| **单 owner** | flock pidfile；启动时**重新认领**仍在跑的 serve（PID 复用安全） |
| **engine 比 daemon 活得久** | 收到 SIGTERM 时 daemon **detach**（不杀 serve）；只有 `POST /engine/stop` 才杀。配 systemd 的 `KillMode=process` |
| **OOM 策略** | 周期性抬高 serve 进程树的 `oom_score_adj`，让内核优先杀 22 GB 的 serve 而不是小 daemon |
| **从不阻塞/从不抛** | spawn/kill/`/proc`/NVML/代理全在事件循环外；handler 错误 → 5xx；降级启动也能起 |
| **崩溃策略** | serve 崩了记录 `lastExitCode` 并上报，**不盲目重启**（盲目重启会在"模型太大导致 OOM"上死循环）；`--auto-restart` 才重启 |
| **force 语义** | `force` 永不隐式；是"显式恢复选择"，可能为不可观测的尾部记录 null 总计 |

### 4.6 持久化记账 outbox
`stop` 时引擎可能还没把最终用量报完。daemon 把"最终记账收据"**持久化**到 `accounting-outbox/`，客户端（桌面端）崩了也能通过 `/accounting/pending` + `/accounting/ack` 重放与确认；`ack` 幂等。

---

## 5. shell（TUI）

- `ft shell` 两种模式（`shell/__init__.py:51`）：
  - **attach**：`--server URL`（默认 `http://127.0.0.1:1919`），纯 HTTP，**不需要 GPU**，可跨机器；
  - **自带引擎**：`--model`/`--model-path` → `launch_server(run_shell=True)`，一个进程里既起引擎又聊天。
- `tui.py`（752 行）：输入处理、流式渲染主循环；内置命令 `/help`、`/think`、`/cache`、`/reset`。
- `render.py`（350 行）：把流式的增量文本渲染成终端输出（工具块、代码块等）。

---

## 6. `ft launch`（agent 配置生成）

### 6.1 六个 agent 的 PREPARERS（`launch.py:818`）

| agent | preparer | 行 | 写什么 | 关键差异 |
|---|---|---:|---|---|
| `codex` | `prepare_codex` | 378 | Codex 的 provider 配置 | 需要 image capability 声明 |
| `claude` | `prepare_claude` | 416 | Claude Code 的 `ANTHROPIC_BASE_URL` 等环境 | 不改配置文件 |
| `opencode` | `prepare_opencode` | 437 | opencode 的 provider | 需要 image 声明 |
| `openclaw` | `prepare_openclaw` | 523 | openclaw 配置 | 需要 image 声明 |
| `hermes` | `prepare_hermes` | 671 | hermes 配置 | 不需要声明 |
| `dsh` | `prepare_dsh` | 721 | DeepSeek Harness 配置 | 需要 image 声明 |

### 6.2 通用流程
```
resolve_server_url（默认 1919）→ discover_server_model（GET /v1/models）
  → PREPARERS[agent]（写配置）
  → ensure_agent_installed（缺就装）
  → run_command（把 -- 之后的参数原样转发）
```
`--dry-run` 只打印计划；`--config` 只配不启；`--install-only` 只装。

### 6.3 安全性
- **清空云 API key**（`ANTHROPIC_API_KEY`、`OPENAI_API_KEY` …）从子进程环境里剔除，防止 agent 静默回落到付费端点；
- `/v1/stats` 报 `image` 时，给需要声明的 agent 写 image-capable。

### 6.4 阅读规模提示
**1,072 行里约 950 行是 per-agent 配置排列组合**。真正通用的只有约 100 行：
`main`（`:1013-1068`）、`parse_argv`（`:828-873`）、`resolve_server_url`（`:146-167`）、`discover_server_model`（`:196-232`）、`run_command`（`:1005-1010`）。
其余：常量（`:19-48`）、dataclass（`:51-89`）、`AGENT_INSTALLERS`（`:91-127`）、TOML/JSON 辅助（`:235-268`）、六个 preparer（`:271-815`）、安装逻辑（`:893-1002`）。

---

## 7. 与其他模块的关系

| 方向 | 模块 | 接口 | 说明 |
|---|---|---|---|
| daemon → server | `server/` | loopback HTTP：`/health`、`/v1/stats`、`/v1/admin/prepare-stop` | **不走 ZMQ** |
| daemon → 子进程 | `ft serve` | `Popen(start_new_session=True)` + 信号 + pidfile | |
| `ft launch` → server | `server/` | `GET /v1/models`、`GET /v1/stats` | 发现模型与能力 |
| `ft ctl` → server | `server/` | 6 类 HTTP 端点 | 见 `03` |
| `ft shell` → server | `server/` | 流式 chat/completions | |
| `cache_report` ← server | `server/` | `/v1/cache/status` | 渲染成人读表格 |

---

## 8. 常见坑

1. **daemon 不 import torch 是硬约束**（不是风格偏好）。加一个 `import torch` 就会让"不可崩"这个卖点消失。
2. **`ft daemon` 带动词是客户端、不带动词是服务端**——同一个命令名两个角色。
3. **`ft launch` 不启动引擎**，它需要一个已经在跑的服务。
4. **有两个 `launch.py`**：仓库根的（agent 配置）与 `server/launch.py`（进程编排），同名不同层。
5. **`ft serve` 的推理流量不经过 daemon**：daemon 只管生命周期。
6. **`env.py` 的 `ENV.X` 与散落的 `os.getenv` 是两套**：找某个开关时两处都要查。
7. **`--gpu` 用 UUID 而不是索引**：多卡下 CUDA 枚举顺序与 `nvidia-smi` 可能不同。
8. **`start_new_session=True` 是 killpg 的前提**：serve 的子进程树能被一次性收掉。

---

## 9. 自测问题

1. daemon 为什么必须 torch-free？这条约束用什么测试强制？
2. daemon 与 serve 之间有哪三种交互方式？为什么不用 ZMQ？
3. "engine 比 daemon 活得久"是怎么实现的？systemd 里对应哪个配置？
4. `serve.json` 重认领如何防 PID 复用？
5. 卡住/崩溃时，daemon 为什么不盲目重启？
6. accounting outbox 解决什么场景？`ack` 为什么要幂等？
7. `ft launch` 的通用逻辑在哪几行？其余 950 行是什么？
8. `ft shell` 的 attach 模式为什么不需要 GPU？
