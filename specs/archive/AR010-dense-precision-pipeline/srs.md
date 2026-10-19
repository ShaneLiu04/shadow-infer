# [AR010] 需求设计说明书：稠密精度码 KV 热路径

| 字段 | 内容 |
|------|------|
| AR 编号 | AR010 |
| AR 主题 | 稠密精度码作为 KV 热路径唯一状态 + 连续 workspace |
| 日期 | 2026-10-19 |
| 状态 | Draft |

## 1. 背景与目标

FlashInfer 的 plan/run 之所以能扩展到长上下文，是因为元数据在张量/紧凑数组中组合，
热路径不做逐元素 Python 解释；PyramidInfer 的长上下文分层预算同样依赖高效的元数据操作。
ShadowInfer v3.10/3.11 的 `PrecisionPlan` 与 `admit_budget` 仍是 **以嵌套 dict 为状态**：

本机实测（真实张量，单层）：

| 规模（seq×heads） | v3.11 `admit_budget` | v3.11 `PrecisionPlan.build` |
|---|---|---|
| 512×8 | 34.2 ms | 1.6 ms |
| 2048×8 | 130.7 ms | 10.5 ms |
| 4096×16 | 548.0 ms | 37.9 ms |
| 8192×32 | **2183.3 ms** | **168.4 ms** |

同时发现一个真实缺陷：当后端输出的 K/V 非连续时，`_compress_kv_with_plan` 使用
`zeros_like(...).view(...)` 触发 stride 错误，25% 的层静默回退到逐元素 reference
（本机 4 层管线：16 次压缩中 4 次回退，reference 累计 0.129s）。

**目标**：以 `int8 codes[seq, heads]` 作为热路径唯一精度状态，dict 仅在 API 边界转换一次；
分配/计划组合改为张量操作；修复非连续输入回退。长上下文单层准入+计划相对 v3.11 提速 ≥10x，
且分配结果与既有语义一致。

## 2. 范围

**In Scope：**
- `precision_plan.py`：`from_codes` / `codes_to_map` / `map_to_codes`；`build` 复用 `from_codes`。
- `rd_allocator.py`：`admit_codes`（RD 排序、sink/复用保护、可选驱逐），与 `admit_budget` 语义一致。
- `ShadowKVAgent`：`allocate_precision_codes`、`_admit_precision_codes`、`on_step` 稠密路径；
  `compress_kv(precision_codes=...)`；决策缓存保存 codes。
- 修复 `_compress_kv_with_plan` 输出 workspace 连续性，禁止非连续输入静默回退。
- 真实张量基准（seq 512-8192）、golden、UT、服务器协议。

**Out of Scope：**
- 不改变精度阈值/预算语义；不改 packed 字节布局。
- 不在本机运行 7B/长上下文模型前向大实验（服务器协议）。
- 不实现固定深度先验。

## 3. 功能需求

### 3.1 FR10.1 稠密精度码

**描述：** `codes` 为 `int8[seq, heads]`，0=fp32,1=fp16,2=int8,3=int4,4=evicted；
提供 codes↔dict 双向转换。

**验收：**
- Given 任意合法 precision_map，When `map_to_codes`→`codes_to_map`，Then 与输入逐项相等。
- Given codes，When `from_codes`，Then `groups`/`flat_indices`/`evicted_entries` 与 `build(map)` 逐项一致；
  `fingerprint` 相同、稳定。

### 3.2 FR10.2 稠密预算准入

**描述：** `admit_codes` 在 codes 上执行与 v3.11 相同的 RD 排序/降档/驱逐语义。

**验收：**
- Given 同一 importance/sensitivity/map/预算，When 分别 `admit_budget` 与 `admit_codes`，Then
  codes 对应结果逐项一致（含 sink/复用保护、预算合规、驱逐计数）。
- `admit_codes` 不修改输入 codes（返回新张量）。
- `rd_allocation=False` 时按 importance 排序。

### 3.3 FR10.3 生产稠密路径

**描述：** `on_step` 从 importance 直接生成 codes，完成 drift top-k 提升、预算准入、
计划/压缩/打包；仅在写入 `self.precision_map` 与决策缓存副本时转换为 dict 一次。

**验收：**
- Given 同一输入，When 稠密路径与 v3.11 dict 路径执行，Then 最终 precision_map、k/v、packed 字节一致。
- 决策缓存命中时复用 codes，不再从 dict 反解析。
- `compress_kv(precision_codes=...)` 与传 map 结果逐位一致。

### 3.4 FR10.4 连续 workspace

**描述：** 输出 K/V 使用连续 zeros，禁止 `zeros_like` 的潜在非连续 stride；plan 路径对非连续输入正常工作。

**验收：**
- Given 非连续 K/V（切片/转置构造），When plan 压缩，Then 与 legacy 逐位一致，不触发 reference 回退。
- 4 层真实后端管线中 fallback 次数为 0。

### 3.5 FR10.5 性能与复现

- 单层 `allocate+admit+plan` 在 8192×32 相对 v3.11 提速 ≥10x；512-4096 不劣化。
- 基准脚本、服务器实验档位与回填协议。

## 4. 非功能需求

| 类型 | 指标 | 要求 |
|------|------|------|
| 性能 | 8192×32 准入+计划 | ≥10x vs v3.11，目标 <200ms |
| 正确性 | golden | codes/plan/packed 与 v3.11 一致 |
| 回退 | 非连续输入 | 0 次 reference 回退 |
| 兼容 | 公共 API | 既有 dict 签名保留 |
| 门禁 | 全量 | 五件套全绿，覆盖率不降 |

## 5. 约束与假设

- 不使用 mock；基准使用真实张量与真实 Agent。
- ties 排序与 v3.11 一致（稳定排序，t 优先）。
- 转换 dict 的一次性 O(N) 成本保留在 API 边界。

## 6. 术语

| 术语 | 定义 |
|------|------|
| codes | `int8[seq,heads]` 精度码，含 evicted |
| 稠密路径 | 以 codes 为状态、无逐条目 dict 的 on_step 路径 |
| workspace | 压缩输出 K/V 的连续张量 |
