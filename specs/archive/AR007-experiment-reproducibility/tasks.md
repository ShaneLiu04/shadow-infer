# [AR007] 任务跟踪

| 字段 | 内容 |
|------|------|
| AR 编号 | AR007 |
| 关联 srs.md | ./srs.md |
| 关联 design.md | ./design.md |
| 创建日期 | 2026-10-19 |

## 任务列表

| ID | 任务描述 | 依赖 | 状态 | 备注 |
|----|---------|------|------|------|
| T001 | 基线：确认现有入口/后端/产物落点 | - | passing | 2026-10-19：run_baseline/run_optimized、ProfilerAgent 输出目录、registry |
| T002 | 实现 `scripts/run_experiment.py`（FR7.1-FR7.8） | T001 | passing | 2026-10-19：真实后端、seed、repeat、摘要、产物隔离 |
| T003 | 规模档位配置 s1/s2/s3 + 长上下文模型模板 | T002 | passing | 2026-10-19：`configs/experiments/` |
| T004 | UT：真实后端、产物、重复统计、mock 拒绝、dry-run、摘要敏感性、模板展平、CLI | T002/T003 | passing | 2026-10-19：7 passed |
| T005 | README v3.9.0/版本三处/ROADMAP + 实验落档 | T004 | passing | 2026-10-19 |
| T006 | Review + ST + 归档 + Gitee 推送 | T005 | passing | 2026-10-19 |

## 进度记录

### 2026-10-19 T001-T002

- 复用 Orchestrator 的 `run_baseline` → `run_optimized` 对照路径。
- 发现并修复重复运行的状态泄漏：同一后端实例的 `_kv_cache` 会带入下一轮 baseline，
  导致 baseline KV 内存口径漂移（0.125 MB vs 0.046875 MB）；每轮开始 `set_kv_cache({})`。
- 发现 ProfilerAgent 使用自身 `output_dir`，仅设置 Orchestrator 顶层目录时产物泄漏到全局 `outputs/`；
  通过注入 `config["profiler"]["output_dir"]` 修复。

### 2026-10-19 T003-T004

- s1 CPU 冒烟档位；s2 7B GPU；s3 8K 长上下文 + 4GB 预算。
- UT-7.1~7.7 全部通过（真实 `PyTorchModelBackend`，无 mock）。

### 2026-10-19 T005

- s1 实测（CPU，6 步 × 3 次重复，seed=42）：
  baseline e2e 中位 41.89 ms（39.70-46.91）；optimized e2e 中位 1029.81 ms（1016.80-1060.44）；
  speedup 中位 0.041x（区间 0.039-0.044x）；KV 内存 1.75 MB → 1.65 MB，压缩比 1.061x。
- 口径说明：s1 用于验证编排与复现，不作为优化收益声称；优化管线 CPU 开销真实存在。

### 2026-10-19 T006

- Review：实现与设计一致；无 mock 路径；产物隔离与 seed 重置经 UT 验证。
- ST：ST-7.1~ST-7.7 全 PASS（见 st-cases.md）。
