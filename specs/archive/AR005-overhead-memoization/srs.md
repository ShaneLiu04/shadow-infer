# AR005 SRS — 推理管线常数开销消除（Kernel 解析记忆化 + KV INT4 批量打包）

## 1. 背景与目标

AR004 完成后，对 6 步优化管线做 CPU cProfile（真实 7B 配置：32 层/32 heads/head_dim 128/hidden 4096/intermediate 11008）暴露两个**实现级常数开销**，与模型规模无关、对真实 GPU 同样存在（CPU 侧因编译尝试失败被放大）：

1. `shadowinfer/kernels/__init__.py::_load_cuda_extensions` 仅缓存成功结果；CUDA 扩展加载失败时 `_cuda_ops` 保持 `None`，导致**每次 kernel 调用都重试 `cpp_extension.load`**。6 步管线中触发 3816 次，累计 12.4s（占总耗时 27.6s 的 45%），单次约 2.8ms。
2. `PackedKVCache._do_pack` 的 INT4 路径对每个 `(token, head)` 逐条调用 `_kernels._pack_int4`，128 token × 8 heads 的微基准中 pack 耗时 72.7ms（逐条参考实现 189ms）。AR002 已向量化量化，但**字节打包仍为逐条循环**。

目标：在保持数值/字节布局逐位一致（golden 等价）的前提下消除上述常数开销；全部 CPU 可验证，无大型实验。

## 2. 范围

**In Scope**
- Kernel 后端解析结果记忆化（成功/失败均缓存），失败告警只发一次，线程安全。
- `PackedKVCache` INT4 字节打包批量化（行内独立 padding），保留 `_do_pack_reference` 作为 golden。
- 单元/接口/性能测试与复现脚本、README/版本/ROADMAP 更新。

**Out of Scope**
- 不改变公开 API、默认精度计划、打包字节格式与存储布局。
- 不引入 GPU 编译器/内核改动；不做端到端大模型实验（服务器协议另行落档）。
- 不重构 ShadowKV 决策语义（AR003 已定）。

## 3. 功能需求

### 3.1 FR5.1 Kernel 后端解析记忆化

**描述：** `_load_cuda_extensions()` 首次调用后无论成功/失败都记录“已尝试”，后续 O(1) 返回缓存；并发首调用互斥。

**触发条件：** 任意 kernel wrapper（`quantize_per_channel_int8`、`_pack_int4`、`_unpack_int4`、`sparse_gemm_ffn` 等）被多次调用。

**期望行为：**
- 首次调用执行一次加载尝试（或返回不可用）；
- 后续调用不再进入 `cpp_extension.load`；
- 不可用告警最多一次；
- 返回值与首次一致（`None` 或 module）。

**异常处理：** 加载抛异常 → 捕获、缓存 `None`、告警一次，走 CPU fallback；不向调用方抛错。

**验收标准：**
- Given 强制不可用环境，When 连续调用 200 次 `_pack_int4`，Then 加载尝试计数 = 1；
- Given 并发 8 线程首次调用，When 同时进入，Then 加载尝试计数 = 1（锁保护）；
- Given 不可用环境，Then 告警仅出现一次。

### 3.2 FR5.2 KV INT4 批量字节打包

**描述：** 将 int4 量化结果按行批量打包为 uint8，替代逐 `(token, head)` 调用；行长度为奇数时逐行补零，保证与参考实现逐位一致。

**触发条件：** `PackedKVCache` 打包且存在 int4 精度条目。

**期望行为：**
- `packed_data["int4"]["k"/"v"]` 与 `_do_pack_reference` 字节完全相同；
- `byte_offsets` 与参考实现一致；
- `scale_k/scale_v`、`num_groups/group_size` 不变；
- 存储字节数（memory_bytes）不变。

**异常处理：** 空 int4 组直接跳过；head_dim 不被 group_size 整除时沿用自适应单组（AR003 行为）。

**验收标准：**
- Given 混合精度（fp32/fp16/int8/int4）K/V，When 向量化 pack vs 参考，Then k/v 字节张量 `torch.equal` 且 `byte_offsets` 完全一致（多组 seq/head/head_dim，含奇数行长）；
- Given 128 seq × 8 heads × head_dim 128，When 3 次中位对比，Then 向量化 pack <= 参考 30%（同进程对照）。

### 3.3 FR5.3 回归与可观测性

**描述：** 管线级行为不变；新增测试与复现脚本落档。

**验收标准：**
- Given 全量测试，Then 通过数不减少、覆盖率不低于当次基线；
- Given 复现脚本，Then 输出 loader 记忆化比值与 pack 同批比值。

### 3.4 FR5.4 文档与版本

- README v3.7.0 四检查点（特性表/版本历史/基准/结构与口径）；
- 版本号三处同步；ROADMAP 勾选 AOG/overhead 相关完成项；
- 服务器复现说明（无需 GPU 也可验证，GPU 上确认同布局）。

## 4. 非功能需求

| 类型 | 指标 | 要求 |
|------|------|------|
| 性能 | loader 记忆化 | <= 5% 的旧逐次重试耗时（200 次调用对照） |
| 性能 | int4 pack 向量化 | <= 参考 30%（同进程 3 次中位） |
| 正确性 | 字节布局 | `torch.equal` + `byte_offsets` 完全一致 |
| 兼容性 | API/格式 | 无公开签名与存储格式变更 |
| 质量 | 五件套 | 全绿；覆盖率不低于当次基线 |
| 环境 | 验证 | CPU-only 可验证 |

## 5. 约束与假设

- CUDA 扩展不可用时（本机）仍必须验证“失败不重试”路径。
- 线程安全仅作用于首次解析；解析完成后的只读缓存无锁。
- 参考实现保留，禁止删除。

## 6. 术语

- 记忆化（memoization）：同一无参解析结果在进程内缓存。
- 行内 padding：奇数行长补零后两两打包，避免跨行混字节。
