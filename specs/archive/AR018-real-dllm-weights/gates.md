# [AR018] T008 质量门禁实跑记录

> 验收口径（tasks.md T008）：五件套全绿；覆盖率 >= 84.71% 不回退；总用例 >= 1214
> 净增；覆盖率与用例数以当次实跑为准登记。本记录为 2026-10-24 实跑取证。

## 环境

| 项 | 值 |
|----|----|
| 机器 | Windows（本机开发环境，CPU） |
| Python | 3.11.9 |
| torch | 2.5.1+cu121（CPU 执行平面） |
| 范围 | shadowinfer/ tests/ benchmarks/ exporter/ scripts/（lint 四件）；mypy shadowinfer/；pytest 全仓 |

## 五件套结果（2026-10-24，T005-T008 全部提交内容就位后实跑）

| 工具 | 命令 | 结果 |
|------|------|------|
| black | `black --check shadowinfer/ tests/ benchmarks/ exporter/ scripts/` | 143 files would be left unchanged（exit 0） |
| isort | `isort --check-only shadowinfer/ tests/ benchmarks/ exporter/ scripts/` | Skipped 22 files，零改动（exit 0） |
| flake8 | `flake8 ... --max-line-length=100 --extend-ignore=E203,W503` | 零告警（exit 0） |
| mypy | `mypy shadowinfer/ --ignore-missing-imports` | Success: no issues found in 78 source files |
| pytest | `pytest`（全量，含覆盖率） | **1304 passed / 1 skipped**，覆盖率 **85.14%**（278.20s） |

## 净增核对

| 项 | AR017 基线（v3.19.0） | 本次实跑（v3.20.0） | 判定 |
|----|------------------------|----------------------|------|
| 总用例 | 1214 passed / 1 skipped | 1304 passed / 1 skipped | +90 净增（P1 29 + P5 5 + P2 11 + P3 26 + P4 19 = 90），通过 |
| 覆盖率 | 84.71% | 85.14% | +0.43pp 不回退，通过 |
| SimpleDiffusionLLM 1214 基线 | 1214 逐位回归 | 零改动通过（T004 薄委托锚点 + 全量绿） | 通过 |

## 说明

- 新增测试全部位于 `tests/test_diffusion_llama.py`（五分区 P1-P5，90 用例）；
  既有 1214 用例零修改零跳过（1 skipped 为既有 skip，非本 AR 引入）。
- 各任务 Green 时的中间全量回归记录见 tasks.md 各会话记录
  （T002 1248 / T003 1259 / T004 1285 / T005 1304），本记录为 T008 统一终验。
- 本机为 CPU 环境，本记录不含任何性能主张；s13 服务器场地断言（S1-S3）
  见 EXPERIMENTS.md（结果落档模板待回填）。
