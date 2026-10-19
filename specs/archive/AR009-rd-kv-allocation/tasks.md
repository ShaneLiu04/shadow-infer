# [AR009] 任务跟踪

| 字段 | 内容 |
|------|------|
| AR 编号 | AR009 |
| 关联 srs.md | ./srs.md |
| 关联 design.md | ./design.md（待生成） |
| 创建日期 | 2026-10-19 |

## 任务列表

| ID | 任务描述 | 依赖 | 状态 | 备注 |
|----|---------|------|------|------|
| T001 | 基线/原型：PyramidInfer 分层先验与 RD 信号对比 | - | passing | 2026-10-19：分层先验 0-4% 不稳定；RD 3.8-15.3% |
| T002 | TDD Red：rd_allocator + evicted 路径 UT | T001 | passing | 2026-10-19：UT-9.1~9.13（子代理 Red） |
| T003 | 实现 `rd_allocator.py` 并接入 `_admit_precision_budget` | T002 | passing | 2026-10-19：13 passed；legacy 开关对比通过 |
| T004 | `evicted` 状态贯通 PrecisionPlan/compress/pack/unpack/evict | T003 | passing | 2026-10-19：零字节/解包零/不复活 |
| T005 | 真实模型基准 + 全量回归 + 五件套 | T004 | passing | 2026-10-19：772 passed / 82.93% |
| T006 | README v3.11.0 / ROADMAP / 服务器实验配置 | T005 | passing | 2026-10-19：README 四检查点、s4 档位、server-experiment |
| T007 | Review + ST + 归档 + Gitee 推送 | T006 | passing | 2026-10-19：归档 specs/archive/AR009-rd-kv-allocation；推送见提交 |

## 状态说明

- `pending` / `in_progress` / `passing` / `failed`

## 进度记录

### 2026-10-19 T001 原型

- 真实 `SimpleDiffusionLLM`（8 层/8 头/dim64/seq128），5 seed。
- 固定深度先验（inverted pyramid）在同字节下收益 0-4.1%，部分预算为负 → 不采用。
- RD 排序（importance×absmax²）同字节加权重建误差：0.9MB +3.80%，1.2MB +9.75%，1.5MB +15.25%。
- 硬驱逐可使 0.3-0.7MB 预算合规；当前实现会超 int4 底线。

## Review 记录（2026-10-19）

- 设计符合性：`rd_allocator` 纯函数接口、RD 排序、硬驱逐、保护集与 design §4 一致。
- 正确性：UT-9.1~9.13 覆盖敏感度、排序、预算、保护、evicted 全链路、统计与不复活。
- 兼容性：`rd_allocation_enabled=False` 时与 v3.10.0 重要性排序一致；默认开启 RD 不改内存上限。
- 风险：固定深度先验未采用（实测 0-4% 不稳定）；硬驱逐默认关闭，需显式开启；服务器精度待回填。
- 结论：PASS。

## ST 记录（2026-10-19）

- ST-9.1~ST-9.7：本机项全 PASS；ST-9.7 服务器执行待回填。
- 全量：772 passed / 1 skipped / 82.93%；五件套全绿。

## 阶段门控记录

- 2026-10-19：用户要求自动化执行 harnessX，跳过交互式 req/design 门控；Review/ST 按本文件执行。
