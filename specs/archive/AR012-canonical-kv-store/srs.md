# [AR012] 需求设计说明书：规范化紧凑 KV Store（向量化读/复用/就地驱逐）

| 字段 | 内容 |
|------|------|
| AR 编号 | AR012 |
| AR 主题 | Canonical packed KV store：向量化 read/reuse/evict + 单源内存 |
| 关联 SR | 无（系统审计驱动） |
| 日期 | 2026-10-20 |
| 状态 | Approved |

## 1. 背景与目标

FlashInfer 的 KV 管理核心是**紧凑、分页、可批量 gather 的单一存储**；PyramidInfer 的
heavy-hitter 驱逐只有在“真正释放字节”时才有意义。当前 `PackedKVCache` 虽然是紧凑格式，
但读路径仍是逐 token-head Python 循环，驱逐是“整层 unpack → 全量 re-pack”，导致：

本机真实测量（`B=1,H=8,D=64`，混合 fp16/int8/int4）：

| 规模 | `unpack` | `apply_reuse_mask`（50%） | `evict`（1/4 条目） |
|---|---|---|---|
| S=512 | 427.8 ms | 49.2 ms | 601.1 ms |
| S=1024 | 824.1 ms | 99.0 ms | 1417.9 ms |

同时存在三个真实缺陷：

1. INT4 的 scale 布局为 `[N*B, G, 1]`，`unpack` 却用 `scale[entry]` 索引，`B>1` 时
   读到错误 batch（实测 B=2 单 batch 相对 MSE 0.1617，B=1 为 0.0073）。
2. INT4 group size 在压缩（64）与打包（16）之间不一致，重复量化；驱逐后 group 又变成 64，
   未改动条目的字节数也会变化。
3. `evict` 把条目降成 INT4+零值，并未释放字节，且会把已硬驱逐（code 4）的条目“复活”。

**目标**：
- `unpack` / `apply_reuse_mask` / `evict` 全部张量化，语义与 reference 一致（INT4 修正后）；
- 统一 INT4 group size，并在打包时复用压缩阶段已量化的 INT4 payload；
- 驱逐改为**桶内就地删除**：真正释放字节、保留 group size、不复活；
- packed cache 成为生产路径的单一驻留表示（默认不再保留 dense shadow），`get_memory_usage`
  报告真实驻留字节；
- 提供可复现微基准与服务器协议。

## 2. 需求范围

**In Scope：**
- `shadowinfer/shadowkv/packed_kv_cache.py`：向量化 `unpack`、`apply_reuse_mask`、bucket-level `evict`、
  `resident_bytes()`、INT4 解码辅助。
- `shadowinfer/shadowkv/shadowkv_agent.py`：`_store_kv`/`_load_kv` 单源存储（可选 dense shadow）、
  INT4 压缩与打包 group size 统一、precomputed payload 传递。
- `shadowinfer/shadowkv/kv_cache_manager.py`：驻留字节口径。
- `shadowinfer/shadowkv/rd_allocator.py`：INT4 成本模型与 pack 统一（group size 参数可配）。
- 测试、基准、配置与服务器协议。

**Out of Scope：**
- 不改变精度分配/复用/驱逐的决策语义（只改变存储与读路径）。
- 不把 packed KV 直接接入 attention（后续 AR）。
- 不在本机运行 7B/长上下文大实验。
- 不实现 CUDA 自定义 page kernel。

## 3. 功能需求

### 3.1 FR12.1 向量化 unpack（含 INT4 batch 修正）

**描述：** 一次 gather/dequantize 每个精度桶，输出 dense `[B,H,S,D]`；缺失/硬驱逐条目为零。

**验收标准：**
- Given fp32/fp16/int8/int4 混合 map（B=1/2），When `unpack()`，Then 与
  `unpack_reference()`（保留的逐条目 golden）逐位/容差一致。
- Given B=2 全 INT4，When unpack，Then 每个 batch 的相对 MSE <= 0.02（修正前 0.16）。
- 512/1024 规模 unpack 相对当前实现加速 >=5x。

### 3.2 FR12.2 向量化 apply_reuse_mask

**描述：** 用 flat 索引批量复制同精度桶的 K/V/scale；跨精度复用必须显式处理（不静默跳过）。

**验收标准：**
- Given 50%/100% 复用 mask，When apply，Then 与逐条目 golden 逐位一致。
- 当前/上一 cache 的 batch/head_dim/group_size 不一致时抛 `ValueError`；跨精度条目保持
  当前量化值并在 `reuse_stats` 中计数（不静默）。
- 512/1024 规模加速 >=3x。

### 3.3 FR12.3 INT4 单一 group size 与单次量化

**描述：** 压缩、打包、驱逐、成本模型统一使用同一 `int4_group_size`（默认 16，可配）；
打包复用压缩阶段的 INT4 codes/scales，不重复量化。

**验收标准：**
- Given 同一 map 与 K/V，When compress→pack，Then packed 的 group_size/num_groups 与配置一致；
  INT4 量化调用次数为 1（压缩阶段）。
- 驱逐前后未改动条目的 packed 字节布局与 scale 数不变。
- `entry_bytes` 与实际 `memory_bytes` 的相对误差 <=1%。

### 3.4 FR12.4 桶内就地硬驱逐

**描述：** `evict(entries)` 直接从桶中删除条目（indices/K/V/scale/byte_offsets），不 unpack、不 re-pack。

**验收标准：**
- Given 混合精度 cache，When 驱逐 N 条，Then `len(indices)` 减少 N、`memory_bytes` 按条目真实下降、
  `unpack` 中被驱逐位置为零。
- 已缺失条目不会被复活；group_size/num_groups 不变。
- 512/1024 规模加速 >=10x；驱逐 1/4 条目后字节下降比例与条目占比相符（±5%）。

### 3.5 FR12.5 单源驻留与内存口径

**描述：** 默认 `store_dense_kv=False`（packed 存在时）不再保留 dense shadow；`_load_kv`
从 packed 反量化；`get_memory_usage` 返回 packed 驻留字节（含 indices/scale 元数据）。

**验收标准：**
- Given `use_packed_cache=True`，When on_step，Then `KVCacheEntry.k_tensor/v_tensor is None`，
  `_load_kv` 返回与 `packed.unpack()` 一致的结果。
- `get_memory_usage` 不把未保留的 dense 字节计入；`resident_bytes()` >= `memory_bytes()`。
- 关闭该选项时行为与 v3.13 逐位一致。

### 3.6 FR12.6 复现与回退

- 基准脚本 `ar012_bench.py` 可一键复跑并断言加速/一致性；
- `configs/experiments/s7_kv_store_gpu.yaml` + `server-experiment.md`；
- 全量门禁全绿，覆盖率不低于 82.99%。

## 4. 非功能需求

| 类型 | 指标 | 要求 |
|------|------|------|
| 性能 | unpack (512/1024) | >=5x |
| 性能 | apply_reuse_mask | >=3x |
| 性能 | evict | >=10x |
| 正确性 | INT4 B>1 相对 MSE | <=0.02 |
| 内存 | 驱逐 | 真实释放字节 |
| 兼容 | 默认 | 关闭 dense-shadow 开关时逐位不变 |
| 门禁 | 全量 | 五件套全绿 |

## 5. 约束与假设

- 所有测试使用真实张量；reference 实现保留为 golden。
- INT4 编码/字节布局不变，仅修正 scale 索引与 group size 一致性。
- `store_dense_kv` 默认在 packed 模式下关闭；直接使用 `KVCacheEntry` 的外部代码需走 `_load_kv`。
- 不改变 attention 的实际输入（本 AR 不宣称端到端模型加速）。

## 6. 术语

| 术语 | 定义 |
|------|------|
| 桶（bucket） | 某一精度的紧凑存储（indices + K/V + scales） |
| 就地驱逐 | 直接从桶删除条目并释放字节，不 unpack/repack |
| 单源存储 | packed cache 为唯一驻留表示，dense 仅按需反量化 |
| resident_bytes | 实际驻留字节（数据 + scale + 索引元数据估算） |
