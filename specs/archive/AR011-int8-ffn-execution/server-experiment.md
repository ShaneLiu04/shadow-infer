# [AR011] 服务器复现实验协议：可执行 INT8 FFN

本协议用于在高速算力服务器上复核 AR011。**本机证据**覆盖真实张量微基准、真实中等模型端到端
与真实小模型 logits；7B 预训练模型的端到端延迟/质量必须在服务器复核。

## 1. 目标与判定

| 项 | 内容 |
|----|------|
| 被测 | `ffn.int8_execution_enabled=true`（真实 INT8 GEMM） |
| 对照 | `ffn.int8_execution_enabled=false`（FP32 FFN，v3.12.0 路径） |
| 主指标 | 端到端每步延迟、FFN 单层延迟、峰值显存、权重常驻内存 |
| 精度指标 | 同 prompt/step 的 logits 相对 L2、top-1 一致率；可选困惑度 |
| 判定 | 7B FFN 上投影（M>=16）加速 >=3x；端到端不慢于 baseline；精度退化在声明阈值内 |
| 重复 | >=3 次，报告 median 与 min-max；GPU/CUDA/torch/oneDNN/CPU 型号/git SHA 必填 |

> 本机小尺寸（K=128）INT8 反而更慢（0.22x），属于 GEMM 交叉点以下；服务器报告必须包含
> 实际 batch/seq 形状，不得把小模型结论外推到 7B。

## 2. 环境

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt && pip install -e ".[dev]"
python -c "import torch; print(torch.__version__, torch.cuda.get_device_name(0)); print(torch.backends.mkldnn.is_available())"
nvidia-smi
```

## 3. 实验命令

### 3.1 微基准（真实张量，1 分钟）

```bash
python specs/archive/AR011-int8-ffn-execution/ar011_bench.py
```

期望：`4096×11008` 且 `M>=16` 时 >=3x；7B 权重内存约 4x 降低；小模型 logits 相对误差 <=3%。

### 3.2 中等模型端到端（本机也可复跑）

```bash
python - <<'PY'
# 见 ar011_bench.py 的 _bench_medium_model；本机 1.42x，误差 0.46%
PY
```

### 3.3 7B 标准预算（s6）

```bash
python scripts/run_experiment.py \
  --config configs/experiments/s6_int8_ffn_gpu.yaml \
  --model-config configs/experiments/model_7b_longctx.yaml \
  --backend hf --model-name <hf-model-id> \
  --device cuda --steps 24 --repeats 3 --out results/ar011_s6
```

对照：同命令将 `ffn.int8_execution_enabled` 改为 `false`，输出目录改为 `results/ar011_s6_base`。

LLaMA/Mistral 类结构（`mlp.up_proj/down_proj/gate_proj`）与 GPT-2 类（`mlp.c_fc/c_proj`）由
`quantize_ffn_modules` 自动替换；若模型结构不匹配，后端返回 `supported=false`，日志会告警，
此时不得声称启用成功。

### 3.4 8K 长上下文（可选，s3 + s6 开关）

```bash
python scripts/run_experiment.py \
  --config configs/experiments/s6_int8_ffn_gpu.yaml \
  --model-config configs/experiments/model_7b_longctx.yaml \
  --backend hf --model-name <hf-model-id> \
  --device cuda --steps 24 --repeats 3 --out results/ar011_s3_int8
```

## 4. 需要采集的证据

1. `result.json` 的 `summary`、`environment`、`plan.config_digest`；
2. PyTorch profiler 中 `mlp.up_proj/down_proj` 的 kernel 名称与耗时（确认走 int8 GEMM，
   而非 dequantized FP32 回退；日志不得出现 `int8 _int_mm unavailable`）；
3. `int8_ffn_enabled` 事件中的 `weight_bytes` / `dense_bytes`（期望约 4x 差异）；
4. 每步 `latency_ms` 中位数与 min-max（baseline vs INT8）；
5. logits 相对 L2 / top-1 一致率（同 seed、同输入）；若测困惑度，报告数据集与样本数；
6. 峰值显存（`torch.cuda.max_memory_allocated`）。

## 5. 本机结果（CPU，2026-10-20）

- 7B 上投影（M=16/64/128，4096×11008）：6.74x / 5.93x / 4.80x（复跑 4.8-6.7x）；
  M=1 为 1.25x，反向投影 11008×4096 为 1.69x（如实记录）。
- 权重内存：3.99-4.00x 降低。
- 中等模型（4 层/hidden=1024/intermediate=4096/seq=256）：85.1→58.1 ms（1.47x，复跑 1.37-1.47x），
  logits 误差 0.46%。
- 小模型（4 层/8 头/dim64/seq128，3 seed）：相对误差 0.43%，top-1 99.22%。
- 小尺寸交叉点：128×128×512 为 0.23x；512×256×1024 为 0.59x。
- 环境：Windows / Python 3.11.9 / torch 2.5.1（CPU），无 CUDA 计算。
- 质量门禁：812 passed / 1 skipped，覆盖率 82.99%；black/isort/flake8/mypy 全绿。

## 6. 回填

- README《性能基准》新增服务器实测小节（GPU/CUDA/seed/重复/形状）；
- 若端到端收益不足或精度超阈，如实记录并保留微基准结论；
- 归档 `server-experiment.md` 的“服务器结果”小节，附 `result.json` 摘要。

## 7. 故障排查

| 现象 | 处理 |
|------|------|
| 日志出现 `_int_mm unavailable` | 检查 torch 版本/CPU 指令集/CUDA 支持；记录回退比例，不得当作 INT8 加速结果 |
| `supported=false` | 模型 FFN 命名不被识别；补充 suffix 或扩展 AR |
| 精度退化 >3% | 检查权重 scale/激活 scale 是否被覆盖；确认未重复量化；报告分布 |
| 端到端变慢但单层变快 | 其他算子/Agent 记账占比高；用 profiler 拆分并报告 |
| 显存未降 | HF 模型仍保留原 FP32 引用（`keep_fp32_reference=true` 或外部引用）；检查引用链 |
