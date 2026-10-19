# [AR007] 需求规格（SRS）：实验可复现编排

| 字段 | 内容 |
|------|------|
| AR 编号 | AR007 |
| 主题 | 真实模型实验的一键可复现编排 |
| 创建日期 | 2026-10-19 |
| 状态 | 已验收（2026-10-19） |

## 1. 背景

此前的性能/精度结论依赖临时脚本：种子、配置、模型规模、设备与产物目录散落在会话中，
跨会话复现成本高，且不同批次实验容易混用口径。框架需要一条**真实可用**的实验入口：
固定随机性、完整记录环境与配置、按重复次数给出中位数/区间，并将每次运行的产物隔离落盘。

约束：不引入 mock 后端；必须使用真实 `pytorch`（SimpleDiffusionLLM）或 `huggingface` 后端。

## 2. 功能需求

| ID | 需求 | 验收要点 |
|----|------|----------|
| FR7.1 | 一键 baseline/optimized 对照 | 同一后端实例、同一 prompt/步数；baseline 先行，optimized 复用 profiler 基线 |
| FR7.2 | 随机性控制 | 每次运行前 `torch.manual_seed(seed)`；后端 KV cache 在每个 repeat 前清空 |
| FR7.3 | 配置与环境取证 | 记录 config/model_config 全量、SHA-256 摘要、Python/torch/platform/device/git SHA、命令与 UTC 时间戳 |
| FR7.4 | 多次重复统计 | `--repeats N`，输出 median/min/max（e2e 延迟、speedup） |
| FR7.5 | 产物隔离 | 每个 repeat 独立 `run_i/`，profiler 的 baseline/optimized/comparison 全部落在该目录 |
| FR7.6 | 多规模档位 | s1（CPU smoke）、s2（7B GPU）、s3（8K 长上下文/4GB 预算）配置模板；模型配置支持仓库模板嵌套结构 |
| FR7.7 | 拒绝 mock | 显式选择 `mock`/`auto` 时抛错，防止误用假数据出结论 |
| FR7.8 | 机器可读 + 人读输出 | `result.json` + `summary.md`；dry-run 只输出计划不执行 |

## 3. 非功能需求

- NFR7.1：单次 s1（6 步）运行在 CPU 上 < 2 min。
- NFR7.2：相同 seed 下 baseline KV 内存结构跨 repeat 完全一致（会话内可验证）。
- NFR7.3：不修改既有 Orchestrator/Agent 对外 API；仅通过配置注入产物目录。
- NFR7.4：五件套全绿，新增 UT 使用真实后端。

## 4. 范围外

- 不替代 GPU 服务器上的大规模精度/性能实验；s2/s3 仅提供可执行模板与落档协议。
- 不引入新的推理算法或改变优化决策语义。
