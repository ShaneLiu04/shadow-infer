# [AR007] 设计说明：实验可复现编排

| 字段 | 内容 |
|------|------|
| AR 编号 | AR007 |
| 关联 srs.md | ./srs.md |
| 创建日期 | 2026-10-19 |

## 1. 总体方案

新增独立入口 `scripts/run_experiment.py`（薄层，不改核心推理代码）：

```
CLI args ──> 配置合并/展平 ──> run_experiment()
                               ├─ 环境与配置摘要（SHA-256）
                               ├─ 创建真实后端（registry）
                               └─ for repeat in range(N):
                                    ├─ backend.set_kv_cache({})   # 隔离上一轮状态
                                    ├─ torch.manual_seed(seed)
                                    ├─ Orchestrator(config=run_cfg)
                                    ├─ run_baseline()
                                    ├─ torch.manual_seed(seed)
                                    └─ run_optimized()
                               └─ 汇总 median/min/max ──> result.json + summary.md
```

## 2. 关键设计

### 2.1 真实后端与拒绝 mock
- `_build_backend` 仅接受 `pytorch` / `huggingface`（`hf` 为别名）；`mock`/`auto` 抛 `ValueError`。
- 后端实例在重复之间复用，但每轮清空 KV cache，避免上一轮压缩精度污染 baseline 统计。

### 2.2 随机性
- 模型加载、baseline、optimized 前分别 `torch.manual_seed(seed)`；Orchestrator 内部真实后端每步
  使用 `torch.randn` 作为输入，种子控制其序列。
- 时间型指标不承诺逐位复现；结构性指标（步数、KV 内存）在同一 seed 下可复现。

### 2.3 产物隔离
- `config["profiler"]["output_dir"] = run_dir`，因为 ProfilerAgent 从自身配置读取输出目录；
  Orchestrator 顶层 `output_dir`/`log_dir` 同步指向 run 目录。
- 这样 `profile_baseline.json`、`profile_optimized.json`、`profile_comparison.html` 不再泄漏到全局 `outputs/`。

### 2.4 配置与规模档位
- 支持扁平模型配置和仓库模板（`model` + `architecture` 嵌套）自动展平。
- s1 内置为 `--profile s1`；s2/s3 为 YAML（`configs/experiments/`），供服务器直接执行。

### 2.5 汇总口径
- baseline/optimized e2e、speedup 均输出 median/min/max（N 次重复）。
- KV 压缩比 = baseline 各层内存和 / optimized 各层内存和；同时保留两侧绝对值。
- 不把 s1 的 speedup 当作优化收益：s1 是复现性冒烟档位，CPU 上优化管线开销真实存在。

## 3. 错误处理

| 场景 | 行为 |
|------|------|
| 传入 mock/auto | 立即报错，不执行 |
| `repeats < 1` | 报错 |
| 无模型配置 | CLI 退出并提示 |
| dry-run | 仅返回计划（含摘要），不创建目录 |
| 后端加载失败 | 异常上抛，不静默回退 mock |
