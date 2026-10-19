# [AR008] 服务器复现实验协议：FlashInfer 式 Plan/Run 融合

本文件面向高速算力服务器（单卡 24GB+ 或 80GB），用于复现 AR008 的
`PrecisionPlan` 融合路径在真实模型上的性能。**本机仅完成 CPU 微基准与逐位 golden**，
服务器结果需按本协议回填到 README 的版本历史与基准表。

## 1. 目标与判定

| 项 | 内容 |
|----|------|
| 被测路径 | `compress_kv` → `PackedKVCache.pack`（plan 共享 + 单次量化） |
| 对照路径 | `git checkout v3.9.0`（AR007 末版）或本仓库 legacy 微基准 |
| 主指标 | end-to-end step latency、compress+pack 累计耗时、峰值显存 |
| 次指标 | KV 压缩比、packed 字节数、精度分布 |
| 正确性 | 同 seed 下 fused 与 legacy 的 `packed_data` 逐位一致（微基准已保证；服务器需抽查 1 个 shape） |
| 判定 | 3 次重复中位；compress+pack 中位不劣于 baseline，目标 >=1.15x；端到端不出现显著回退 |

> 端到端收益取决于模型规模、KV 预算与 GPU 访存；若 GPU 上 Python 开销被异步执行掩盖，
> 允许 compress+pack 收益显著而 e2e 收益有限，必须如实分别报告。

## 2. 环境

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
# 可选：安装 CUDA 扩展（无 Ninja 时自动 CPU fallback，仍可运行但性能口径不同）
pip install -e ".[dev]"
python -c "import torch; print(torch.__version__, torch.cuda.get_device_name(0))"
nvidia-smi
```

记录：GPU 型号/显存、驱动、CUDA、torch、git SHA（`git rev-parse HEAD`）。

## 3. 复现命令

### 3.1 微基准（必做，5 分钟）

```bash
python specs/archive/AR008-fused-kv-plan/ar008_bench.py
```

期望：输出 4 个形状的 legacy→fused 中位与比值，且脚本内的逐位断言全部通过
（失败会直接抛 `AssertionError`）。

### 3.2 7B 单卡（s2 档位）

```bash
python scripts/run_experiment.py \
  --config configs/experiments/s2_7b_gpu.yaml \
  --model-config configs/model_fast_dllm_7b.yaml \
  --backend hf --model-name <hf-model-id> \
  --device cuda --steps 32 --repeats 3 \
  --out results/ar008_s2
```

说明：
- `--backend hf` 使用真实 HuggingFace 权重；`pytorch` 仅内置小模型，不用于 7B 结论。
- 若模型需要自定义 tokenizer/输入，请用 `--backend hf --model-name <本地目录>`。
- 运行前确认 `configs/experiments/s2_7b_gpu.yaml` 中 `shadowkv.memory_budget_mb`
  与显卡匹配；显存不足时优先降 `steps` 或 `seq_len`，不要改 seed。

### 3.3 8K 长上下文（s3 档位）

```bash
python scripts/run_experiment.py \
  --config configs/experiments/s3_longctx_gpu.yaml \
  --model-config configs/experiments/model_7b_longctx.yaml \
  --backend hf --model-name <hf-model-id> \
  --device cuda --steps 24 --repeats 3 \
  --out results/ar008_s3
```

## 4. 产物与回填

每次运行产生：

```
results/ar008_s2/
├── result.json          # 环境、配置摘要、plan、每次运行原始数据、median/min/max
├── summary.md           # 人读汇总
└── run_i/
    ├── profile_baseline.json
    ├── profile_optimized.json
    └── profile_comparison.html
```

回填要求：
1. 将 `result.json` 的 `summary`（中位/区间）、`environment`、`plan.config_digest`
   提交到 AR 归档或 issue；
2. README《性能基准》新增服务器实测小节，标注 GPU 型号、CUDA/torch、seed、重复次数；
3. 若 e2e 回退，必须保留并解释（例如调度开销占比、GPU 异步掩盖），不得只报 compress 比值。

## 5. 故障排查

| 现象 | 处理 |
|------|------|
| `backend 'hf' not available` | 安装 `transformers`；或改用本地权重目录 |
| CUDA OOM | 降低 `--steps`/`seq_len`/`memory_budget_mb`；固定 seed 后重跑 |
| 微基准某形状断言失败 | 记录 shape/精度分布/硬件，停止回填；属正确性缺陷 |
| 结果与 CPU 微基准比值差异大 | 正常：CPU 路径受 Python 开销主导；分别报告设备口径 |
| `plan` 指纹变化 | 检查 precision_map/reuse 是否受未固定随机性影响；用同一 seed 重跑 |
