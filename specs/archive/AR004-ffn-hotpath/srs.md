# [AR004] 需求设计说明书

| 字段 | 内容 |
|------|------|
| AR 编号 | AR004 |
| AR 主题 | FFN 热路径与逐层状态正确性（fingerprint O(1) / 统计缓存 / 向量化 unpack / 层隔离 / 掩码修复） |
| 关联 SR | 无正式 SR；来源为 AR002/AR003 审计遗留缺口（FFN 支线约 57% 采样时间）与 AR004 微基准取证 |
| 日期 | 2026-10-18 |
| 状态 | Approved（门控第 12 轮 PASS） |

## 1. 背景与目标

AR003 完成后，端到端残余瓶颈集中在 FFN 支线。AR004 微基准（fast, CPU, 3 次重复，`ar004_bench.py`）定位到明确根因：

1. `PackedFFNWeight.dequantize_for_matmul` 占 `ffn.on_step` 累计时间 **39%**，其中 `_precision_plan_fingerprint`（每次 matmul 调用重建 O(channels) 元组）占 **37%**；真实 7B（intermediate_dim=11008）下机制外推该开销约放大 ~11x（未验证，待 GPU 实验）；
2. `on_step` 每步对 `precision_scales` 做 3 次 O(channels) 统计求和（quantization_info 聚合），占 **20%**，而精度计划是静态的（只在量化时变化）；
3. `unpack()` 仍为逐 channel Python 循环（首次反量化 / 重打包路径的热点，AR002 已缓存结果但未向量化计算本身）；
4. **正确性缺陷 FFN-P1**：`_weight_up/_weight_down/weight_up_quantized/channel_importance` 为单份全局状态，逐层调用时所有层共享第一层权重与量化计划——真实多层模型下结果错误（mock 路径因各层权重相同而掩盖）；
5. **正确性缺陷（取证发现）**：`_mixed_compute` 对 down 投影未应用通道掩码（up 输出被裁剪到 active 通道后 shape 不匹配），`channel_mask + packed weights` 组合直接抛错；
6. `quantize_mixed_precision` 逐 channel Python 循环（11008 次）——一次性成本，但影响冷启动与真实维度可用性。

**目标**：在不改变数值语义（golden 等价）的前提下消除 FFN 支线热点，并修复多层状态与掩码正确性。全部验证 CPU 可执行、快测试。

## 2. 需求范围

影响模块：`shadowinfer/ffn_optimizer/packed_weight.py`、`shadowinfer/ffn_optimizer/ffn_optimizer_agent.py`、`tests/`、`README.md`、`ROADMAP.md`、`pyproject.toml`、`shadowinfer/__init__.py`。

**In Scope：**
- FR4.1 精度计划指纹 O(1)（pack 时缓存，重打包失效）
- FR4.2 quantization_info 聚合统计缓存（量化时计算一次）
- FR4.3 `unpack` / 反量化向量化（按精度分组批量，golden 与逐 channel 参考等价 ≤1e-6）
- FR4.4 FFN-P1 逐层权重/量化/通道重要性状态隔离（layer_id 键）
- FR4.5 `_mixed_compute` down 投影通道掩码修复（列掩码）
- FR4.6 `quantize_mixed_precision` 向量化（分组批量，与逐 channel 实现等价）
- FR4.7 README v3.6.0 四检查点 + ROADMAP

**Out of Scope：**
- 真实 7B GPU 端到端实验（按用户指令暂停；微基准与单测作为本 AR 证据）
- CUDA sparse_gemm 内核变更、FFN-Compiler、CSR/CSC
- ShadowKV/QDrift 变更（AR003 已完成）
- AR005 AOG / AR006 宏块率控

## 3. 功能需求

### 3.1 精度计划指纹 O(1)

**描述：** `dequantize_for_matmul` 的缓存校验不得每次重建 O(channels) 元组。

**触发条件：** 每次 `dequantize_for_matmul()`（每 matmul 调用）。

**期望行为：** 指纹在 `_do_pack` 结束时计算一次并缓存（O(1) 读取，值语义比较）；`_do_pack` 重建计划时同步失效/更新。语义不变（同计划命中、不同计划重算）。

**异常处理：** 无（内部缓存，不可变）。

**验收标准：**
- Given packed 权重，When 连续 100 次 dequantize_for_matmul，Then 仅首次执行 unpack（unpack 调用计数 = 1）且总耗时中指纹开销 O(1)（微基准：1024 channels 下 fingerprint 耗时 ≤ 10µs）；
- Given 重新 pack（精度计划变化），Then 缓存失效并重算（golden 一致）。

### 3.2 量化统计缓存

**描述：** `on_step` 的 `quantization_info` 聚合（fp16/int8/int4 计数、压缩比）在量化时计算一次并缓存。

**触发条件：** 每次 `on_step`（非 full 路径）。

**期望行为：** 统计随精度计划变化更新；`on_step` 读取缓存，返回结构与数值不变。

**异常处理：** 状态重置（on_init）清空缓存。

**验收标准：**
- Given 同一 agent 连续多步（首步完成量化后），When 读取返回的 quantization_info，Then 与首次一致且数值正确（fp16/int8/int4 计数之和 = 通道数）；
- Given 效率：1024 channels 下 on_step 中热路径读取零迭代（硬断言）；统计读取耗时 ≤ 20µs（同机同批 3 次中位，诊断项）；取证基线：cProfile 中统计相关占 on_step 约 20%（见 baseline-profile.md §2），绝对耗时以 T007 同批对照为准。

说明：内存累计由“每步累加”改为“每层量化时一次”；单步返回的 quantization_info 字段与数值不变。

### 3.3 unpack 向量化

**描述：** `unpack()` 由逐 channel 循环改为按精度分组批量反量化。

**触发条件：** `unpack()` / `dequantize_for_matmul()` 首次调用 / 重打包。

**期望行为：** 输出 dense 张量与逐 channel 参考实现（保留为 golden）逐元素一致（≤1e-6）；fp32/fp16/int8/int4 混合均正确；pack_dim=0/1 均正确。

**异常处理：** 未知精度按 fp16（现有回退）。

**验收标准：**
- Given 混合精度 precision_map（含 int4 分组/非整除情形），When 向量化 unpack，Then 与 `_unpack_reference` max|Δ| ≤ 1e-6；
- Given 1024 channels，When 对比 unpack 耗时，Then 向量化实现 3 次中位 ≤ 参考实现 50%（微基准）。

### 3.4 逐层状态隔离（FFN-P1）

**描述：** 权重缓存、量化权重、通道重要性、精度计划按 `layer_id` 隔离。

**触发条件：** 同一 agent 依次处理不同 layer_id 的 `on_step`。

**期望行为：** 每层使用自己的权重与量化状态；on_init 清空全部层状态；无 layer_id（默认 0）行为不变（向后兼容）。

**异常处理：** 层状态数超过 max(1, num_layers) 时告警一次并继续运行（配置异常，不中断推理）。

**验收标准：**
- Given 两层不同权重（full 路径），When 依次 on_step，Then 第 2 层输出等于用第 2 层权重直接计算的结果（≤1e-6），第 1 层状态不被覆盖（再调用第 1 层结果稳定）；
- Given on_init，Then 层状态清空（重新进入冷启动）。

### 3.5 `_mixed_compute` 通道掩码修复

**描述：** 通道掩码对 up 投影取行、对 down 投影取列；混合路径与 full 路径在相同掩码下输出一致。

**触发条件：** `channel_pruning_enabled` 且 sensitivity < 0.7（channel_mask 非空）且混合/量化路径。

**期望行为：** 不再抛 shape 错误；输出与「同一 packed 权重反量化后的 dense 权重 + 同掩码的 full 计算」golden 一致（≤1e-6，比较对象同为量化权重）。

**异常处理：** 掩码全 False 时输出零向量（与 full 路径一致），告警一次。

**验收标准：**
- Given channel_mask + packed 权重，When `_mixed_compute`，Then 不抛错，且与「同一 packed 权重反量化后的 dense 权重 + 同掩码的 full 计算」≤1e-6（比较对象明确：同为量化权重）；
- Given 全 False 掩码，Then 不抛错且输出为 0。

### 3.6 `quantize_mixed_precision` 向量化

**描述：** 按阈值分组批量量化，替代逐 channel 循环。

**触发条件：** 首次混合路径 / 计划重建。

**期望行为：** 返回的 quantized_up/down 与 scales 与逐 channel 参考实现数值一致（float 比较 1e-6；scale 元组逐位）；packed 分支的 precision_map 不变。

**异常处理：** 全零通道 scale 处理与现状一致（保留原值）。

**验收标准：**
- Given 随机权重与重要性，When 向量化实现 vs `_quantize_mixed_precision_reference`，Then 输出 ≤1e-6、scales 完全一致；
- Given 1024 channels，Then 向量化实现 3 次中位 ≤ 参考 50%。

### 3.7 文档与版本同步

**描述：** 按仓库规则一完成 README v3.6.0 四检查点更新与版本号三处同步。

**触发条件：** T007 微基准与全量回归通过后。

**期望行为：** README 特性表/版本历史/性能基准（实测数据）/结构与口径（TOC 锚点、测试数、无 emoji）更新；`pyproject.toml` 与 `shadowinfer/__init__.py` 版本号与 README 一致（3.6.0）；ROADMAP 更新 FFN 项状态。

**异常处理：** 版本号不一致或 TOC 锚点失效视为未完成（禁止提交）。

**验收标准：**
- Given 更新完成，When 检查版本号，Then pyproject / `__version__` / README 徽章三处均为 3.6.0；
- Given README，When 检查内容，Then 无 emoji、TOC 锚点与章节一一对应、测试数与覆盖率与当次实跑一致；
- Given ROADMAP，When 检查 FFN 条目，Then 已标注 AR004 完成项与剩余项（真实 GPU 内核/编译器）。

## 4. 非功能需求

| 类型 | 指标 | 要求 |
|------|------|------|
| 性能 | 指纹（1024 ch） | ≤ 10µs/调用（3 次中位；同机同批） |
| 性能 | unpack / quantize 向量化 vs 同进程参考实现 | 3 次中位比值 ≤ 0.5（同批对照，参考实现保留在代码内） |
| 性能 | `ffn.on_step`（1024 ch/4 层/12 calls） | 3 次重复报告中位与区间；对照口径为同会话参考路径（reference 实现）比值，不设跨会话固定基线硬门禁 |
| 正确性 | golden | 向量化 vs 参考 max\|Δ\| ≤ 1e-6；golden 测试保留参考实现 |
| 兼容性 | 对外 API | 不破坏既有公开接口；新增内部缓存/property/可选参数（`_weight_up` 等私有属性除外） |
| 质量 | 五件套 | 全绿；覆盖率不低于项目门禁 75%（当次 pytest --cov 实测，并同步 README 两处数字；82.04% 仅作 v3.5.0 参考） |
| 环境 | 执行设备 | CPU-only 可验证 |

## 5. 约束与假设

**约束：**
- AGENTS.md 规则 1.1：性能数字基于 3 次重复中位 + 同机同参；微基准落档
- 不引入新依赖；参考实现保留作 golden（`_unpack_reference`、`_quantize_mixed_precision_reference`）
- 大规模 7B GPU 实验仍暂停（用户指令）；本 AR 证据为 CPU 微基准 + 单测

**假设：**
- 真实多层路径每层权重不同（FFN-P1 修复的必要性前提）
- 精度计划在量化后静态（缓存统计/指纹安全）

## 6. 术语说明

| 术语 | 定义 |
|------|------|
| 精度计划 | channel → precision 的映射（packed 权重 + scales） |
| fingerprint | 精度计划指纹，用于反量化结果缓存校验 |
| quantization_info | on_step 返回的量化统计聚合（计数/压缩比） |
| 层状态隔离 | 每 layer_id 独立的权重缓存/量化态/重要性 |
