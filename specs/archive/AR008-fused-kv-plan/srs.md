# [AR008] 需求设计说明书

| 字段 | 内容 |
|------|------|
| AR 编号 | AR008 |
| AR 主题 | FlashInfer 式 Plan/Run 融合：ShadowKV 压缩与打包单一量化 |
| 关联 SR | - |
| 日期 | 2026-10-19 |
| 状态 | Draft |

## 1. 背景与目标

FlashInfer 的核心工程思想是 **plan / run 分离**：把分页索引、indptr、批次元数据等
一次组合好，热路径只执行数据面计算；同一份计划可服务多次调用，避免重复推导。
ShadowInfer 的 KV 压缩路径存在同类浪费：

1. `ShadowKVAgent._compress_kv_vectorized` 每次调用都用 Python 双重循环重建
   `precision_map` 分组，并为每个 `(token, head)` 构造 scale 视图（生产路径下游并不使用）；
2. `PackedKVCache._do_pack` 对**同一份** `precision_map` 再建一次分组，并对**原始 K/V 再量化一次**；
3. 两段各自做 advanced indexing，产生重复 gather/拷贝。

基线（CPU，真实张量，5 次中位）：seq=512/heads=8/dim=64 时 compress 23.15 ms、
pack 4.84 ms、合计 27.71 ms；seq=1024 时 41.17 + 9.24 = 50.41 ms。
cProfile 显示 compress 的自耗时主要来自逐条 dict/切片，而非量化本身
（单组 int8 量化约 0.14 ms）。

**目标**：引入 `PrecisionPlan` 计划对象，压缩与打包共享同一份分组/索引/量化结果，
量化只做一次，同时保持字节级 golden 与既有公开行为。

## 2. 需求范围

**In Scope：**
- 新增 `shadowinfer/shadowkv/precision_plan.py`：`PrecisionPlan`（稠密精度码、分组切片、
  flat gather 索引、复用索引、fingerprint）。
- `_compress_kv_vectorized` 支持消费 plan；生产路径不再构造未使用的 scale dict。
- `PackedKVCache.pack` 支持消费压缩阶段已产生的分组量化结果（不重复量化/分组）。
- `compress_kv` 在一次调用内构建 plan 并贯穿 compress + pack。
- 参考实现与 `PackedKVCache.pack` 原路径保留，作为 golden 与回退。
- 真实张量微基准、UT、全量测试、服务器实验配置与复现文档。

**Out of Scope：**
- 不改变分页/存储布局（FlashInfer 的 page table 留作后续 AR）。
- 不新增 CUDA kernel，不改优化决策语义与对外 API（仅新增可选参数）。
- 不在本机运行 7B/长上下文实验。

## 3. 功能需求

### 3.1 FR8.1 PrecisionPlan 计划对象

**描述：** 从 `precision_map` / `reuse_mask` 一次性构建不可变计划。

**验收标准：**
- Given 混合精度与复用掩码，When 构建 plan，Then 提供稠密精度码、每个精度的
  连续索引切片、flat gather 索引、复用索引，且与旧路径分组逐项一致。
- 相同输入构建的 plan `fingerprint` 相同；精度或复用变化时 fingerprint 不同。

### 3.2 FR8.2 压缩消费计划

**描述：** `_compress_kv_vectorized` 通过 plan 完成 gather/量化/scatter。

**验收标准：**
- Given 同一 `(k, v, precision_map, reuse_mask)`，When 新旧路径执行，Then
  `k/v` 逐位一致、`scale_k_map/scale_v_map` 键集合与数值一致（默认构建）。
- Given `build_scale_map=False`，When 生产路径执行，Then 不构造 scale dict，
  且 `k/v` 与默认路径逐位一致。

### 3.3 FR8.3 打包复用压缩结果

**描述：** `PackedKVCache.pack` 接受压缩阶段的分组量化结果，跳过二次分组与量化。

**验收标准：**
- Given 同一输入与 plan，When 融合路径与原 pack 路径执行，Then `packed_data`
  的 `k/v/scale_*/indices/byte_offsets/group_size/num_groups` 逐位一致，
  `memory_bytes` 相同。
- Given 未提供预计算分组，When 调用 pack，Then 行为与现状完全一致（回退路径）。

### 3.4 FR8.4 生产集成与回退

**描述：** `compress_kv` 一次构建 plan，compress/pack 共享；异常时回退 reference。

**验收标准：**
- Given `use_packed_cache=True`，When `on_step` 压缩，Then pack 复用量化结果，
  且 KV 存储与统计口径不变。
- Given 融合路径抛错，Then 记录告警并回退 `_compress_kv_reference`，不产生错误数据。

### 3.5 FR8.5 性能与实验资产

- CPU 微基准覆盖 seq/heads/dim 多规模点，报告 3 次中位与区间。
- `configs/experiments/` 增加 AR008 服务器复现档位（7B/长上下文），README 写清命令与口径。

## 4. 非功能需求

| 类型 | 指标 | 要求 |
|------|------|------|
| 正确性 | golden | 新旧路径逐位一致（含奇数 head_dim、batch>1） |
| 性能 | compress+pack | 同批 CPU 微基准中位不劣于基线，目标 >=1.15x |
| 兼容性 | API | 现有调用零改动可用；新增参数均有默认值 |
| 质量 | 门禁 | 五件套全绿，覆盖率不低于 v3.9.0 |

## 5. 约束与假设

**约束：**
- 不使用 mock；基准与测试全部使用真实张量与真实 Agent。
- 大模型实验仅落档配置与协议，待服务器执行。

**假设：**
- scale map 在生产下游未被消费（已核对 `shadowkv_agent.py` / `orchestrator.py`）。
- 量化器为确定性实现，同一输入重复量化结果逐位一致。

## 6. 术语

| 术语 | 定义 |
|------|------|
| Plan/Run | 先构建索引/分组等元数据，再执行数据面计算 |
| PrecisionPlan | 本 AR 的不可变精度/复用计划对象 |
| 融合打包 | pack 复用 compress 的分组量化结果，避免二次量化 |
| golden | 保留的参考实现，用于逐位对照 |
