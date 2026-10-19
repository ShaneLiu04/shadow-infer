# [AR009] 服务器复现实验协议：率失真 KV 分配与硬驱逐

本协议用于在高速算力服务器上验证 AR009。**本机证据**仅覆盖真实模型架构、随机初始化
权重下的分配器与重建误差；预训练模型的端到端精度必须在服务器复核。

## 1. 目标与判定

| 项 | 内容 |
|----|------|
| 被测 | `rd_allocator`（RD 排序）与 `hard_eviction_enabled` |
| 对照 | `rd_allocation_enabled=false, hard_eviction_enabled=false`（v3.10.0） |
| 主指标 | 同 seed/同预算下：加权 KV 重建误差（离线可算）、packed 字节、峰值显存 |
| 精度代理 | 同一 prompt/step 下，压缩后 KV 与全精度 KV 的下一步 logits 余弦相似度 / MSE |
| 判定 | 同字节误差或 logits 指标不劣于 baseline（目标 >=3% 改善）；启用驱逐时 100% 预算合规 |
| 重复 | >=3 次，报告 median 与 min-max；GPU 型号/CUDA/torch/git SHA 必填 |

> 端到端文本质量需在 pretrained 模型上评估；若仅能测 logits 代理，必须明确标注。

## 2. 环境

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt && pip install -e ".[dev]"
python -c "import torch; print(torch.__version__, torch.cuda.get_device_name(0))"
nvidia-smi
```

## 3. 实验命令

### 3.1 分配器微基准（真实架构，5 分钟）

```bash
python specs/archive/AR009-rd-kv-allocation/ar009_bench.py
```

期望：同字节 1.2/1.5MB 档位 RD 误差低于 baseline；低于 int4 底线的预算 RD+驱逐全部合规。

### 3.2 7B 标准预算（s2）

```bash
python scripts/run_experiment.py \
  --config configs/experiments/s2_7b_gpu.yaml \
  --model-config configs/model_fast_dllm_7b.yaml \
  --backend hf --model-name <hf-model-id> \
  --device cuda --steps 32 --repeats 3 --out results/ar009_s2
```

### 3.3 8K 长上下文（s3）

```bash
python scripts/run_experiment.py \
  --config configs/experiments/s3_longctx_gpu.yaml \
  --model-config configs/experiments/model_7b_longctx.yaml \
  --backend hf --model-name <hf-model-id> \
  --device cuda --steps 24 --repeats 3 --out results/ar009_s3
```

### 3.4 紧预算/强制驱逐（s4，512MB）

```bash
python scripts/run_experiment.py \
  --config configs/experiments/s4_edge_eviction_gpu.yaml \
  --model-config configs/model_fast_dllm_7b.yaml \
  --backend hf --model-name <hf-model-id> \
  --device cuda --steps 24 --repeats 3 --out results/ar009_s4
```

## 4. 需要采集的证据

1. `result.json` 的 `summary`、`environment`、`plan.config_digest`；
2. 运行日志中的 `hard_eviction` 事件计数（`evicted` 数量）与 `layer budget floor` 告警
   （启用驱逐时不应出现预算超限告警）；
3. packed 字节：`run_i/profile_optimized.json` 或日志中的 `compressed_bytes`，须 ≤ 配置预算；
4. 精度代理：对同一 step 的全精度 KV 与压缩 KV 分别前向，记录 logits 余弦/MSE；
5. baseline 对照：同一命令将配置中两个开关设为 `false` 后重跑。

## 5. 回填

- README《性能基准》新增服务器实测小节（GPU/CUDA/seed/重复次数/预算）；
- 若端到端 logits 或困惑度改善 <3%，如实记录，不得只报微基准；
- 归档 AR009 `server-experiment.md` 的“服务器结果”小节，附 `result.json` 摘要。

## 6. 故障排查

| 现象 | 处理 |
|------|------|
| 始终无 `hard_eviction` 事件 | 预算可能仍高于 int4 底线；降低 `memory_budget_mb` 或 seq_len |
| 启用驱逐仍有 floor 告警 | 仅剩 sink/复用条目；提高预算或减少 sink_tokens |
| HF 模型 logits 形状不一致 | 固定同一个 batch/seq/step；使用 backend 输出的 logits |
| RD 收益为负 | 记录分布与预算；检查 `rd_allocation_enabled` 是否误关；不得调参掩盖 |
