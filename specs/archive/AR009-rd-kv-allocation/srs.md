# [AR009] 需求设计说明书：率失真感知的 KV 精度/驱逐分配

| 字段 | 内容 |
|------|------|
| AR 编号 | AR009 |
| AR 主题 | PyramidInfer / FlashInfer 启发的 KV 预算分配：率失真排序 + 硬驱逐 |
| 日期 | 2026-10-19 |
| 状态 | Draft |

## 1. 背景与目标

PyramidInfer 的核心是「重击者（heavy hitter）保留 + 分层预算」：深层注意力更集中，
可保留更少 KV；FlashInfer 的核心是「plan/run 分离 + 预算内组合最优布局」。ShadowInfer
当前的预算准入（`_admit_precision_budget`）只按 **注意力重要性** 排序降档，存在两个实际限制：

1. **排序信号单一**：同等重要性下，KV 幅值大的条目量化误差更大，却被同等对待；
2. **预算硬下限不可达**：当总预算低于「sink fp16 + 全 int4」的字节底线时，只会告警并
   **继续超预算**，端侧无法保证内存上限。

本机对真实 `SimpleDiffusionLLM` 前向得到的逐层注意力与 K/V 做了 5 seed 原型验证：

- 以 `attention_importance × KV_absmax²` 排序（率失真近似）替代纯重要性排序，同字节下
  加权 KV 重建误差下降 **3.8% / 9.8% / 15.3%**（总预算 0.9 / 1.2 / 1.5 MB，5 seed 中位）；
- 纯 PyramidInfer 式「深层少分」层预算在本模型上收益 0-4% 且不稳定，因此本 AR **不采用
  固定深度先验**，而保留其「重击者保留」思想，用实测信号驱动分配；
- 启用硬驱逐后，预算低于 int4 底线时仍可满足预算（当前实现会超限）。

**目标**：新增率失真感知分配器与可选硬驱逐，在同等字节下降低 KV 重建误差，并提供
可复现的服务器验证协议。

## 2. 需求范围

**In Scope：**
- 新增 `shadowinfer/shadowkv/rd_allocator.py`：字节成本、KV 幅值敏感度、预算准入、硬驱逐。
- `_admit_precision_budget` 接入分配器；新增配置开关。
- `PrecisionPlan` / `PackedKVCache` / 压缩路径支持 `evicted` 状态（零字节、解包为零）。
- 保护 sink 前缀与复用条目；保持「只降不升」。
- 真实模型基准、UT、服务器复现配置与文档。

**Out of Scope：**
- 不改注意力计算/模型结构；不引入 mock。
- 不在本机运行 7B/长上下文大实验。
- 不实现固定深度先验（经原型验证收益不稳定）。

## 3. 功能需求

### 3.1 FR9.1 率失真排序信号

**描述：** 分配器使用 `importance × sensitivity` 作为降档/驱逐优先级；敏感度由真实 K/V
的 per-token-head 幅值平方给出。

**验收：**
- Given 真实 K/V 与重要性矩阵，When 计算敏感度，Then 输出 `[seq, heads]` 非负张量，
  与逐条目 `(absmax_k²+absmax_v²)/2` 一致（误差 <1e-6）。
- Given 两名目重要性相同但幅值不同，When 分配，Then 幅值大者优先保留高精度。

### 3.2 FR9.2 预算准入

**描述：** 在层预算内按率失真优先级降档；sink 与复用条目受保护。

**验收：**
- Given 预算可被降档满足，When 准入，Then predicted bytes ≤ budget，档位只降不升，
  sink 保持初始档位，复用条目不被降档。
- Given `rd_allocation_enabled=False`，Then 行为与 v3.10.0 完全一致（重要性排序 golden）。

### 3.3 FR9.3 可选硬驱逐

**描述：** 降档至 int4 仍超预算且 `hard_eviction_enabled=True` 时，按率失真优先级升序
驱逐非保护条目（`evicted`，0 字节）。

**验收：**
- Given 预算低于 int4 底线，When 启用硬驱逐，Then predicted/packed bytes ≤ budget；
  sink 与复用条目不驱逐；`unpack` 对驱逐条目返回 0。
- Given 未启用，Then v3.10.0 行为保持不变（告警 + 底线档）。
- Given 驱逐后的 packed cache 再次 `evict`，Then 不会复活已驱逐条目。

### 3.4 FR9.4 集成与配置

**描述：** 配置键：`rd_allocation_enabled`（默认 true）、`hard_eviction_enabled`（默认 false）。
`on_step` 将复用掩码作为保护集传入准入。

**验收：**
- Given 默认配置，Then RD 排序生效、无驱逐；Given 服务器档位，Then 显式开启驱逐。
- 统计：`precision_bits["evicted"]=0`，驱逐计数可观测。

### 3.5 FR9.5 性能与复现资产

- 本机基准：真实模型、3-5 seed、多预算，报告加权重建误差、预算合规、耗时。
- `configs/experiments/` 增加 AR009 档位；`server-experiment.md` 写清回填口径。

## 4. 非功能需求

| 类型 | 指标 | 要求 |
|------|------|------|
| 质量 | 同字节加权重建误差 | 1.2-1.5MB 预算内相对 v3.10.0 降低 ≥3%；0.9MB 允许在噪声内 |
| 合规 | 预算 | 启用驱逐时 100% ≤ budget |
| 兼容 | 关闭开关 | 与 v3.10.0 逐位一致 |
| 开销 | 敏感度计算 | 单层 <10% 压缩耗时（小模型实测） |
| 质量门禁 | 全量 | 五件套全绿 |

## 5. 约束与假设

- 不使用 mock；基准使用真实 `SimpleDiffusionLLM` 前向的注意力与 K/V。
- 本机模型为真实实现但随机初始化；服务器用预训练权重复核。
- 重要性-敏感度乘积是重建误差的一阶近似，最终以实测误差为准。

## 6. 术语

| 术语 | 定义 |
|------|------|
| sensitivity | 每 token-head KV 的幅值平方，近似量化误差尺度 |
| RD score | `importance × sensitivity`，用于降档/驱逐优先级 |
| evicted | 零字节状态，解包为零 |
| 重击者 | 高重要性/高敏感度、优先保留的 token-head |
