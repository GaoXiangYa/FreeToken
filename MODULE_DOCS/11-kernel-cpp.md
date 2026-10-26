# 11 · kernel（C++/CUDA 与构建层）：JIT/AOT 缓存、三个 torch 扩展、csrc 内核、预编译 wheel

> 行号基于 `377a3bc`。完整符号清单见 [`99-symbol-index/kernel.md`](99-symbol-index/kernel.md)（本文件只覆盖 `kernel/` 顶层 Python + `csrc/`；`triton/` 与 `fla/` 见 `12`）。

---

## 1. 模块职责与在系统中的位置

`kernel/` 有两个独立的构建体系，**不要混为一谈**：

| 体系 | 编译者 | 内容 | 加载函数 |
|---|---|---|---|
| **tvm-ffi** | `tvm_ffi.cpp.load_inline` / `load` | `csrc/jit/*`（运行时 JIT）与 `csrc/src/*`（AOT） | `load_jit`（`utils.py:352`）/ `load_aot`（`:292`） |
| **torch CppExtension** | `setup.py` 的 `BuildExtension` 与 `cpp_extension.load` | `csrc/pinned_tensor.cpp`、`csrc/cpu_moe/`、`csrc/row_store/`、`csrc/gguf/` | 直接 `import` 或 `load` |

- **上游**：`moe/`（专家 gather）、`kvcache/`（KV 写入）、`layers/`（embedding gather）、`models/`（磁盘表）、`distributed/`（pynccl）、`attention/`（部分元数据）。
- **下游**：GPU 驱动、nvcc、`freetoken-kernel-cache` 预编译 wheel。
- **门面探针**：`kernel/backend.py` 用无副作用的 `find_spec` 判断 flashinfer/sgl_kernel/vllm 是否可用，直接影响上层走哪条路径。

```
layers/moe  ──▶ fast_index_copy ──┐
kvcache     ──▶ store / index    ──┼─▶ load_jit / load_aot ──▶ 预编译 .so 或 nvcc
models      ──▶ row_store        ──┘        │
                                            └─▶ tvm_ffi.Module
cpu MoE     ──▶ _cpu_moe (torch ext)     pinned ──▶ _pinned_tensor (torch ext)
```

---

## 2. 文件清单

### 2.1 顶层 Python

| 文件 | 行数 | 职责 | 阅读建议 |
|---|---:|---|---|
| `utils.py` | 421 | **核心**：`KernelConfig`、`make_cpp_args`、`load_jit`/`load_aot`、缓存目录解析、架构参数、版本校验 | **精读** |
| `aot.py` | 287 | 预编译 wheel 的 kernel 规格与打包（`default_kernel_specs:146`） | 精读 |
| `aot_models.py` | 470 | 预编译的**形状覆盖表**（19 个模型的 store/index/fast_index_copy 尺寸推导） | 概览（读 `:10-27` 的推导说明） |
| `backend.py` | 83 | 包可用性探针 + `device_capability` + `driver_cuda_version` | 精读（很短） |
| `_toolchain.py` | 79 | nvcc 与 torch 的 CUDA major 一致性校验 | 略读 |
| `pinned.py` | 69 | 精确尺寸 pinned 分配（`alloc_pinned_tensor:42`）、`host_register:48`、`device_ptr:59` | 精读 |
| `fast_index_copy.py` | 210 | 专家行 gather 的 launcher（fused / per-bank，`blocks_per_bank` 启发式） | **精读** |
| `store.py` / `index.py` | 43 / 55 | KV 写入 / embedding gather 的 launcher | 精读 |
| `batch_memcpy.py` | 70 | `cudaMemcpyBatchAsync` 封装（**CUDA≥13** 门控 + 真实拷贝探测） | 精读 |
| `row_store.py` | 55 | 磁盘行读取（PLE/Engram）的 launcher + stream memop 同步 | 精读 |
| `gguf.py` | 143 | 唯一用 torch `cpp_extension.load` 的扩展（vendored ggml 量化 matmul） | 概览 |
| `pynccl.py` / `radix.py` / `tensor.py` | 80 / 21 / 20 | NCCL 通信 / CPU 前缀比较 / 测试算子 | 略读 |
| `moe_impl.py` | 505 | 把 Triton MoE 算子 re-export 出来 | 略读 |
| `causal_conv1d.py` | 76 | sgl_kernel 的 causal-conv1d + Triton 回退 | 略读 |
| `__init__.py` / `__main__.py` | 41 / 53 | 导出 / 生成 `.clangd` | 略读 |

### 2.2 `csrc/`（共 13,360 行）

| 子树 | 行数 | 构建方式 | 用途 | 性能关键度 |
|---|---:|---|---|---|
| `jit/fast_index_copy.cuh` | 597 | tvm-ffi JIT | **offload 专家 gather（最关键的单个内核）** | ★★★ |
| `jit/store.cu` / `jit/index.cu` / `jit/batch_memcpy.cuh` | 123 / 172 / 58 | JIT | KV 写入 / embedding gather / 批量拷贝 | ★★★ / ★★ / ★★ |
| `cpu_moe/cpu_moe_ext.cpp` | 2,211 | torch CppExtension | CPU MoE 解码、flag 握手、ISA 分派 | ★★★ |
| `row_store/row_store_ext.cpp` | 677 | torch CppExtension（Linux only） | PLE/Engram 的 O_DIRECT/io_uring 行读取 | ★★★ |
| `pinned_tensor.cpp` | 128 | torch CppExtension | 精确尺寸 `cudaHostAlloc`、`host_register`、`device_ptr` | ★★★ |
| `src/pynccl.cu` / `src/radix.cpp` / `src/tensor.cpp` | 188 / 44 / 31 | tvm-ffi AOT | NCCL 通信对象 / CPU 前缀比较 / 测试算子 | ★★ / ★ / 忽略 |
| `gguf/`（6 文件） | ~5,800 | torch `cpp_extension.load` | vendored 自 sgl-kernel→llama.cpp 的量化 matmul | ★★（仅 Gemma4 GGUF） |
| `include/freetoken/` | ~1,300 | 头文件 | tvm-ffi 辅助（`tensor.h` matcher、`warp.cuh`、`utils.cuh/h`、`nccl227.h`、`hip_compat.h`） | 基础设施 |

---

## 3. 构建与加载机制（本文重点）

### 3.1 `KernelConfig` 与"名字即缓存键"

`KernelConfig`（`utils.py:155`）是 NamedTuple：`num_threads` / `max_occupancy` / `use_pdl`。
`make_cpp_args(*args)`（`:281`）把配置展开成 C++ 模板参数列表，`_make_name`（`:166`）据此生成 kernel 名，例如 `freetoken__store_1024_128_1_false`。
**这个字符串同时是**：
- C++ 模板实例化的参数；
- 预编译 `.so` 的目录名与文件名；
- tvm-ffi 构建缓存的 key。
所以**任何参数变化都会 miss 预编译**，退化为 JIT。

### 3.2 `load_jit` / `load_aot` 的查找顺序

```
load_jit(name)（utils.py:352）/ load_aot(name)（:292）
  → _load_prebuilt(name)（:252）
        → _kernel_cache_dir()（:210）
             ① FREETOKEN_DISABLE_KERNEL_CACHE 开着 → None
             ② FREETOKEN_KERNEL_CACHE_DIR 覆盖
             ③ import freetoken_kernel_cache → get_jit_cache_dir()（:219, :246-249）
        → 版本/CUDA-major 配对校验（:227-244）
        → <cache>/<name>/<name>.so（:262）
        → tvm_ffi.load_module（:266）
  → 命中即返回
  → 未命中：
        FREETOKEN_DISABLE_JIT=1 → 直接抛错（发布验证用）
        否则 tvm_ffi.cpp.load_inline / load 编译
```

### 3.3 架构安全

- `_cuda_cflags(extra, arch_list)`（`:81`）除了 `-std=c++20 -O3`，还会追加 `-gencode ... code=compute_XX`（最高架构的 PTX）。这个字符串**也会进入 tvm-ffi 的构建 hash**，所以不同架构不会共用同一个 `.so`。
- `_cuda_arch_list()`（`:38`）优先 `TVM_FFI_CUDA_ARCH_LIST`，否则用当前绑定的 GPU 算 `major.minor`；ROCm 走 `_rocm_arch_list()`（`:51`）。
- `_pin_tvm_ffi_arch_ctx`（`:65`）在一次构建期间临时设置架构环境变量，结束后恢复——避免污染调用者。
- `_toolchain.py:51-78`：nvcc 与 torch 的 CUDA major 不一致时**直接报错**（`FREETOKEN_ALLOW_CUDA_MISMATCH` 可越过）。理由：链接到错误的 libtorch 会产出难以诊断的运行时错误。

### 3.4 `setup.py` 的三个 CppExtension

| 扩展名 | 源 | 链接 | 作用 |
|---|---|---|---|
| `_pinned_tensor` | `csrc/pinned_tensor.cpp` | cudart / amdhip64 | 精确尺寸 pinned 分配（避开 torch 的 2 的幂取整）、`host_register`、`device_ptr` 的 UVA/WDDM 处理 |
| `_cpu_moe` | `csrc/cpu_moe/cpu_moe_ext.cpp` | 同上 | CPU 专家计算、flag 握手、ISA 分派、看门狗 |
| `_row_store` | `csrc/row_store/row_store_ext.cpp` | 同上（Linux only） | O_DIRECT + pread / io_uring 行读取 |

三者都用 `BuildExtension.with_options(use_ninja=True)`，`-O3 -std=c++17`，include `csrc/include`。

### 3.5 `aot.py` + `aot_models.py`：预编译 wheel

`default_kernel_specs()`（`aot.py:146`）列出要预编译的 kernel：
5 个 `store` 规格、N 个 `index`、1 个 `fast_index_copy_update_flag`、M 个 `fast_index_copy`、3 个 `fast_index_copy_multi`（`blocks_per_bank` = 8/32/64）、1 个 `batch_memcpy`、1 个 `radix`。

形状从哪来：`aot_models.py` 的 `SUPPORTED_MODELS`（19 个模型条目）推导：
- `store_element_sizes`（`:409`）：`num_kv_heads * head_dim * 2`
- `index_variants`（`:413`）：`(hidden_size*2, num_splits_for(...))`
- `fast_index_copy_feature_sizes`（`:420`）：每个 offload bank 的行字节，**过滤到 128 的倍数**（per-bank kernel 按固定 128 字节步长拷贝）
- `expert_bank_row_bytes`（`:73`）与 `fp8_block_scale_pad`（`:66`）做实际计算

**设计警告（`:10-12` 的 docstring）**：这些推导必须与运行期调用点**手工保持同步**；一旦漂移，就会"按名字找不到预编译 → 静默退回 JIT（需要 nvcc）"。
架构列表默认 `8.0 8.6 8.9 9.0 10.0 12.0`（`freetoken-kernel-cache/build_backend.py`）。

---

## 4. csrc 内核逐个详解

### 4.1 `csrc/jit/fast_index_copy.cuh`（597 行）——最关键的单个内核

- **做什么**：把 offload 未命中的专家行从 pinned 主机内存（或 slot 缓存）gather 到 GPU 槽位。
- **为什么不是 memcpy**：源行是**离散的**（每层每专家的行地址不连续），所以必须按行索引；per-16 字节单元还要算 `row = u / units`（64 位除法）并取两个索引。
- **为什么快**：`ld.global.L1::no_allocate` + `st.global.wt` 的 streaming 访存——**不污染 L1/L2**（拷贝数据不会被复用）。
- **fused 变体**：`fast_index_copy_multi` 一次 launch 处理所有 bank；注释记录"6 banks × 36 层 = 216 次/步 launch，暖缓存时几乎都是空 launch"——fused 就是为了压 launch 数。
- **网格宽度决定链路是否饱和**：`blocks_per_bank` 默认 PCIe 用 8、NVLink-C2C 用 32（`fast_index_copy.py` 的启发式 + `FREETOKEN_H2D_BLOCKS_PER_BANK`）。注释记录 PCIe ~4K 线程/bank 即饱和、C2C 需 ~32K（8×1024 → 222，32×1024 → 412 GB/s）。

### 4.2 `csrc/jit/store.cu`（123 行）——KV 写入
一个 warp 处理一个 token，**K 与 V 一趟写完**，按 element_size 自动选 16/8/4 字节的 lane 打包并完全展开（`include/freetoken/warp.cuh`）。`__launch_bounds__(128, 1)`，grid = `ceil(len/4)`。
**在 decode 关键路径上**（图内）。

### 4.3 `csrc/jit/index.cu`（172 行）——embedding gather
一行（embedding 行）按行字节数切给多个 warp：`num_splits_for` 规则 = 行字节 %2048==0 → 4；%1024==0 → 2；否则 1。grid = `num_splits * num_indices`。
**在 decode 关键路径上**。

### 4.4 `csrc/jit/batch_memcpy.cuh`（58 行）
`cudaMemcpyBatchAsync` 的薄封装：一次 enqueue N 个独立拷贝。**CUDA 13 的 8 参数签名**（12.8/12.9 多一个 `failIdx`），所以 `kernel/batch_memcpy.py:42-53` 先校验 `torch.version.cuda >= (13,0)` 再用真实拷贝探测。
**只用于 prefill** 的未命中批量抓取。

### 4.5 `csrc/pinned_tensor.cpp`（128 行）
- `cudaHostAlloc(portable|mapped)` 的**精确尺寸**分配（torch 的 pinned 缓存会按 2 的幂取整：70 GB → 128 GB）。
- `host_register`：对已存在内存做 `cudaHostRegister`。
- `device_ptr`：返回 GPU 可解引用的地址——**只在 UVA 下成立，WDDM 下不成立**，所以要做平台判断。
- 还提供 `driver_cuda_version` 供 `backend.py:69` 使用。

### 4.6 `csrc/cpu_moe/cpu_moe_ext.cpp`（2,211 行）
- 暴露：pinned IO 分配、submit/sync、ISA 查询、flag 握手的 stream memop 解析。
- **握手协议**：mapped-pinned 的 int64 `ready/done/err` 数组，每 (layer, decode-bs) 一个槽；GPU 侧写 `done=0`/`ready=1` 然后 `cuStreamWaitValue64(done>=1)`；CPU 侧常驻协调器轮询→计算→release-store `done=1`。
- **ISA 分派**：`scalar`/`avx2`/`avx512`/`avx512bf16`，按 `__builtin_cpu_supports` 选，可用环境变量降级；bf16/fp4/W4A8-int8 各有独立选择器（VNNI vs AVX-VNNI）。
- 注释里记录的实测：host-func 回调 ~30-50 μs × 2 次/层/步 ≈ 6 ms/step；自旋 kernel 方案会把利用率钉在 99% 并触发 CPU 降频（GEMV -1.5×）——所以默认用 stream memop。

### 4.7 `csrc/row_store/row_store_ext.cpp`（677 行，Linux only）
磁盘固定宽度行的读取：O_DIRECT + `pread`，或 **io_uring 批量**。配 `kernel/row_store.py` 的 **stream memop 等待/信号**：被 CUDA graph 捕获的 lookup 可以等宿主填充完成（否则图里无法阻塞等 I/O）。

### 4.8 `csrc/src/`
- `pynccl.cu`（188）：NCCL 通信对象（TP 用），`-lnccl`。
- `radix.cpp`（44）：CPU `std::mismatch` 做前缀比较（被 `kvcache/radix_cache.py` 用）。
- `tensor.cpp`（31）：测试/校验用，**可忽略**。

### 4.9 `csrc/gguf/`
vendored 自 sgl-kernel 的 `csrc/quantization/gguf/`，而后者是 **llama.cpp** 的移植（`gguf_kernel.cu` 头注释标注了 vLLM commit）。`dispatch.h` 是裁剪过的类型分派 + warp shuffle 垫片，使 JIT 编译自包含。
**唯一消费者是 Gemma4 GGUF**（`models/gguf/config.py:22-24`）。它由 torch `cpp_extension.load` 编译（`kernel/gguf.py:50-73`），模块名 `freetoken_gguf_kernels`，优先 clang++/gcc-13。

### 4.10 `csrc/include/freetoken/`
`tensor.h`（496，tvm-ffi 的 Tensor matcher）、`utils.cuh/h`、`warp.cuh`（向量化 warp copy）、`nccl227.h`（571，NCCL 兼容层）、`hip_compat.h`（ROCm）。**基础设施，skim 即可**。

---

## 5. 函数详解（Python 层）

| 函数 | 行 | 作用 | 关键行为 |
|---|---:|---|---|
| `_is_rocm` | 32 | 平台判定 | 看 `torch.version.hip` |
| `_cuda_arch_list` / `_rocm_arch_list` | 38 / 51 | 目标架构 | 环境变量优先，否则当前 GPU |
| `_pin_tvm_ffi_arch_ctx` | 65 | 临时固定构建架构 | 上下文管理器，退出恢复 |
| `_cuda_cflags` / `_hip_cflags` / `_rocm_link_flags` | 81 / 94 / 101 | 编译参数 | gencode 进 hash |
| `CppArgList` | 150 | 模板参数列表 | 支持 int/bool/str |
| `KernelConfig` | 155 | 内核配置 | NamedTuple(num_threads, max_occupancy, use_pdl) |
| `_make_name` | 166 | 生成 kernel 名 | 名字即缓存键 |
| `_env_enabled` | 170 | 布尔环境变量 | |
| `_freetoken_version` / `_version_parts` / `_build_stamps` | 174 / 180 / 187 | 版本与 `+g<sha>` 解析 | 版本配对校验用 |
| `_kernel_cache_version_ok` | 193 | 缓存版本校验 | |
| `_kernel_cache_dir` | 210 | 缓存目录决议 | 三级优先级（见 §3.2） |
| `_load_prebuilt` | 252 | 加载预编译 `.so` | 失败返回 None |
| `_make_wrapper` / `make_cpp_args` | 276 / 281 | 生成内联包装源码 / 参数列表 | |
| `load_aot` | 292 | 加载 `csrc/src` 的 AOT 模块 | |
| `load_jit` | 352 | 加载/编译 `csrc/jit` 的内联源码 | |
| `is_flashinfer_installed` / `is_sgl_kernel_installed` / `is_vllm_installed` | `backend.py:27/32/37` | 包探针 | `functools.cache` + `find_spec`，无副作用 |
| `device_capability` / `is_rocm` / `driver_*_version` | `backend.py:42/51/58/69` | 硬件探针 | 上游用它选 kernel |
| `alloc_pinned_tensor` / `host_register` / `device_ptr` | `pinned.py:42/48/59` | 钉住内存操作 | 精确尺寸 |
| `_kernel_names` / `default_kernel_specs` | `aot.py` | 预编译清单 | |
| `compile_and_package_kernels` | `aot.py:238` | 构建并打包 wheel | |

---

## 6. 关键流程逐段解析

### 6.1 第一次调用某 JIT kernel
```
调用方（如 moe/offload_cache.py:1037）
  → fast_index_copy 的 launcher → load_jit(name, src, cflags)
        → _load_prebuilt(name)             # 预编译 wheel 里没有 → None
        → tvm_ffi.cpp.load_inline(...)      # 调 nvcc 编译（几十秒）
        → tvm-ffi 把产物写进自己的缓存目录
  → 后续进程启动时再 load_jit 同名 → 命中缓存（秒级）
```

### 6.2 预编译 wheel 命中 / miss
```
命中：load_jit → _load_prebuilt → freetoken_kernel_cache.get_jit_cache_dir()
        → <cache>/<name>/<name>.so → tvm_ffi.load_module    （毫秒级）
miss：FREETOKEN_DISABLE_JIT=1 → RuntimeError（发布验证用：任何 miss 立刻暴露）
      否则走 nvcc JIT
```
**注意**：wheel 只含 C++/CUDA 扩展内核，**不含 Triton 内核**——Triton 靠自己的磁盘缓存（`triton/autotune_cache.py` 填充）。

### 6.3 CPU-MoE 的一次前向
```
GPU（前端）：写 done=0 → 写 ready=1（两个 stream memop）
CPU 协调器：轮询 ready → submit(task) + sync(spin) → release-store done=1
GPU：cuStreamWaitValue64(done>=1) → 继续
```
回退路径（`FREETOKEN_CPU_MOE_FLAG_SYNC=0` 或槽位耗尽）：`cudaLaunchHostFunc`，贵 ~30-50 μs/次。

### 6.4 磁盘表的一行读取
```
token → hash → 行索引
  → RowStore extent（mmap 的 checkpoint 分片 + O_DIRECT）
  → io_uring / pread 批量读 24 行（去重）
  → 填进 pinned staging（拼成池行格式）
  → stream memop 释放（图内 lookup 在等它）
```

---

## 7. 与其他模块的关系

| 方向 | 模块 | 接口 | 说明 |
|---|---|---|---|
| 被使用 | `moe/` | `fast_index_copy`、`batch_memcpy`、`pinned` | offload 搬运 |
| 被使用 | `kvcache/` | `store`、`radix` | KV 写入 / 前缀比较 |
| 被使用 | `layers/embedding.py` | `index` | embedding gather |
| 被使用 | `models/deepseek_v41`、`models/qwen4_exp` | `row_store` + `pinned` | 磁盘表 |
| 被使用 | `moe/cpu_executor.py` | `_cpu_moe` 扩展、`pinned` | CPU 专家 |
| 被使用 | `distributed/` | `pynccl` | TP 通信 |
| 被使用 | `layers/`、`moe/` | `moe_impl`、`causal_conv1d` | 算子转发 |
| 使用 | `backend.py` | 包/硬件探针 | 决定走 flashinfer 还是 Triton |
| 产物 | `freetoken-kernel-cache` | 预编译 wheel | 省掉首次 nvcc |

---

## 8. 常见坑

1. **预编译 wheel 不含 Triton 内核**——"启动慢"要分开查这两条路。
2. **形状覆盖表是手工维护的**：`aot_models.py` 漂移 → 按名字 miss → 静默退回 JIT（而 JIT 需要 nvcc）。
3. **PDL 编译进了 `store`/`index`，但默认 `use_pdl=False`**（`store.py:12`、`index.py:12` 的 `DEFAULT_INDEX_KERNEL_CONFIG`）。
4. **`FREETOKEN_DISABLE_JIT=1` 是发布验证的正确姿势**：任何 cache miss 立刻失败，而不是悄悄编译几十秒。
5. **架构列表被环境变量污染会产出只含单个 arch 的 wheel**（`scripts/build-release-wheels.sh` 里对此有告警）。
6. **`tensor.cpp` 是测试算子**，别在它上面花时间。
7. **`cudaMemcpyBatchAsync` 的签名是 CUDA 13 的**：12.8/12.9 会直接抛错。
8. **`device_ptr` 依赖 UVA**：Windows/WDDM 下 pinned-mapped 内存不能直接当设备地址用。

---

## 9. 自测问题

1. 为什么 kernel 名要包含 `num_threads`/`max_occupancy`/`use_pdl`？这个设计带来什么代价？
2. `load_jit` 的查找顺序是什么？`FREETOKEN_DISABLE_KERNEL_CACHE` 与 `FREETOKEN_DISABLE_JIT` 分别控制什么？
3. 为什么 `-gencode` 参数要进入构建 hash？
4. 三个 torch CppExtension 各自提供什么？为什么它们不走 tvm-ffi？
5. `fast_index_copy` 为什么是 gather 而不是 memcpy？它的 `blocks_per_bank` 为什么与链路类型有关？
6. `cpu_moe_ext` 的 flag 握手为什么比 `cudaLaunchHostFunc` 省？自旋 kernel 方案的问题是什么？
7. 预编译 wheel 覆盖哪些 kernel、不覆盖哪些？形状是怎么推导出来的？
8. `row_store` 的 stream memop 在解决什么图捕获问题？
