# [AR010] 服务器复现实验协议：稠密精度码热路径

本协议用于在高速算力服务器上复核 AR010。**本机证据**覆盖真实张量与真实 4 层后端管线
（随机初始化权重，CPU），证明元数据路径逐位等价与降本；7B/8K 端到端延迟与精度必须在服务器复核。

## 1. 目标与判定

| 项 | 内容 |
|----|------|
| 被测 | `dense_precision_enabled=true`（codes 热路径） |
| 对照 | `dense_precision_enabled=false`（v3.11.0 dict 路径） |
| 正确性 | 同一配置/seed 下 precision_map、packed 字节、KV 输出逐位一致 |
| 主指标 | 元数据阶段（allocate+admit+plan）耗时；端到端每步延迟；峰值显存 |
| 判定 | 元数据阶段 >=10x（本机 8192×32 实测 18.00x，复跑 18.0-19.1x）；端到端不得劣化；输出一致 |
| 重复 | >=3 次，报告 median 与 min-max；GPU 型号/CUDA/torch/git SHA 必填 |

## 2. 环境

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt && pip install -e ".[dev]"
python -c "import torch; print(torch.__version__, torch.cuda.get_device_name(0))"
nvidia-smi
```

## 3. 实验命令

### 3.1 元数据微基准（真实张量，1 分钟）

```bash
python specs/archive/AR010-dense-precision-pipeline/ar010_bench.py
```

期望：2048/4096/8192 三档 codes 路径均不慢于 dict 路径；8192×32 比值 >=10x；
脚本内断言 map/predicted/evicted/fingerprint 完全一致。

### 3.2 4 层真实后端等价性（CPU 也可）

```bash
python -m shadowinfer optimize --model SimpleDiffusionLLM --backend pytorch \
  --device cuda --num-steps 8 --config configs/optimize_full.yaml
```

将配置中 `shadowkv.dense_precision_enabled` 分别设为 true/false，对比日志中的
`precision_map` 分布、`compressed_bytes` 与每步延迟。

### 3.3 8K 长上下文（s5）

```bash
python scripts/run_experiment.py \
  --config configs/experiments/s5_dense_precision_gpu.yaml \
  --model-config configs/experiments/model_7b_longctx.yaml \
  --backend hf --model-name <hf-model-id> \
  --device cuda --steps 24 --repeats 3 --out results/ar010_s5
```

对照：同命令加 `shadowkv.dense_precision_enabled=false`，输出目录改为 `results/ar010_s5_base`。

## 4. 本机结果（CPU，2026-10-19）

- `ar010_bench.py`（3 次中位[min-max]，生产包装）：512/2048/4096/8192 分别
  16.67x/17.11x/18.81x/18.00x；8192×32 为 2443.2→135.7 ms；独立复跑 15.7-19.6x；
  map/predicted/fingerprint 断言通过。
- `pipeline-check.py`（真实 4 层 `PyTorchModelBackend`，4 步）：`plan_calls=16, reference_calls=0`。
- 环境：Windows / Python 3.11.9 / torch 2.5.1（CPU），无 CUDA 计算。

## 5. 需要采集的证据

1. `result.json` 的 `summary`、`environment`、`plan.config_digest`；
2. 每步 `latency_ms` 中位数与 min-max（优化前后）；
3. 日志中不存在 `vectorized compress_kv failed, falling back to reference`；
4. packed 字节与 KV 内存（应与对照逐位一致）；
5. 若可测 logits/困惑度，记录同 seed 下与对照的差值。

## 6. 回填

- README《性能基准》新增服务器实测小节（GPU/CUDA/seed/重复次数/规模）；
- 若端到端收益不足或出现劣化，如实记录，并保留元数据微基准结论；
- 归档 AR010 `server-experiment.md` 的“服务器结果”小节，附 `result.json` 摘要。

## 7. 故障排查

| 现象 | 处理 |
|------|------|
| 出现 reference 回退日志 | 检查 K/V stride 与 head_dim；记录形状复现 |
| 两条路径 precision_map 不一致 | 检查 `dense_precision_enabled` 是否真正切换；核对 tie 输入 |
| 元数据加速比低于 10x | 确认规模为 8192×32；GPU 上 dict 路径占比可能不同，按实测记录 |
| 显存峰值变化 | 预期 packed 布局不变；若变化，检查 plan 是否由 codes 构建 |
