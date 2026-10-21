# [AR013] 服务器复现实验协议：跨步复用与紧凑 KV 闭环执行

本协议用于在 GPU 服务器复核 AR013。本机证据（CPU、真实张量、真实
`SimpleDiffusionLLM` forward）覆盖受控 denoising 轨迹上的速度/误差/驻留；7B 规模与 GPU
延迟必须在服务器复核。

## 1. 目标与判据

| 项 | 内容 |
|----|------|
| 被测 | `execution.kv_lifecycle=replace` + `step_reuse_enabled=true` + `kv_execution=packed` |
| 对照 1 | `replace` + dense + 无复用（同上下文长度的正确 full baseline） |
| 对照 2 | `replace` + dense + 复用（隔离 packed 开销） |
| 对照 3 | `replace` + packed + 无复用（隔离复用收益） |
| 主指标 | 每步延迟 median[min-max]、30 步总时间、峰值显存、KV 驻留字节 |
| 质量指标 | 每步 logits 相对 L2、top-1 一致率；可选困惑度 |
| 判定 | 复用 >=1.15x 且 logits relL2 <=2%；packed 数据平面驻留（张量字节，含 scale）<=0.6x fp16 dense KV（int8），int4 需 group_size>=32 才可达 ~0.35x；全口径 resident（另含 48B/token-head Python 元数据估计）CPU 上约 0.72x，该元数据为 CPU 簿记、GPU flat-buffer 部署不承载；组合不劣于 1.10x（seed0 中位） |
| 重复 | 3 seed × 3 repeats，报告 median[min-max]；GPU/驱动/torch/git SHA 必填 |

> 本机 packed attention 使用 `PackedKVCache.unpack()` 反量化后计算，CPU 上比 dense 慢
> （原型 0.03-0.16x）。服务器不要以 CPU packed 延迟外推；packed 的收益是 KV 驻留，
> 组合速度收益来自跨步复用。

## 2. 环境

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt && pip install -e ".[dev]"
python -c "import torch; print(torch.__version__, torch.cuda.get_device_name(0))"
nvidia-smi
```

## 3. 本机基线（可复跑）

```bash
# 归档前：
python specs/changes/AR013-step-reuse-packed-execution/ar013_bench.py
# 归档后：
python specs/archive/AR013-step-reuse-packed-execution/ar013_bench.py
```

受控轨迹 `x_t = x0 + sigma_t * noise`（sigma 0.9→0.02，10 步），真实 8 层模型：

口径：dense_bytes 为 FP16 dense 基准（SRS D4）；resident 含 48B/token-head 元数据估计，
data-plane 仅计张量字节（含 scale）。驻留列格式：全口径 / 数据平面。

| 规模 L/H/D/HN/S | reuse 加速 | logits relL2 | 跳过率 | packed 驻留 (full/data-plane) | combined 加速 |
|---|---|---|---|---|---|
| 8/8/64/512/256 | 1.76-1.89x | 0.94-0.95% | 50% | 72% / 53% | 1.09-1.17x |
| 8/8/64/512/512 | 1.86-1.87x | 0.95-0.96% | 50% | 72% / 53% | 1.13-1.17x |
| 8/16/64/1024/256 | 1.93-1.94x | 0.96% | 50% | 72% / 53% | 1.42-1.44x |

（3 seed 实测中位，2026-10-20，CPU；combined seed 间最低 1.09x，如实记录。）

## 4. 服务器 7B 实验

### 4.1 全量对照（5 组）

```bash
# 0) baseline: replace + dense, no reuse
python scripts/run_experiment.py \
  --config configs/experiments/s8_step_reuse_gpu.yaml \
  --model-config configs/experiments/model_7b_longctx.yaml \
  --backend pytorch --device cuda --steps 30 --repeats 3 \
  --out results/ar013_s8_base
# 运行时将 execution.step_reuse_enabled=false, kv_execution=dense

# 1) reuse only: dense + reuse
#    step_reuse_enabled=true, kv_execution=dense
# 2) packed only: packed + reuse=false
#    kv_execution=packed, step_reuse_enabled=false
# 3) combined: packed + reuse
#    step_reuse_enabled=true, kv_execution=packed（s8 默认）
# 4) int8 packed（可选）：packed_precision=int8
```

每组的输出目录独立；每个 seed 修改 `--seed`（或 config 中 seed）后重复。

### 4.2 质量取证

- 保存每组每步 logits（或最后一步）到 `results/ar013_s8_*/logits.pt`；
- 以 baseline 为参考计算 `||logits - ref|| / ||ref||` 与 top-1 一致率；
- 质量门槛：reuse / combined 的 median relL2 <=2%；若超限，降低 `step_reuse_threshold`
  （0.1 → 0.05）并重跑。

### 4.3 显存与驻留

```bash
nvidia-smi --query-gpu=memory.used --format=csv -l 1
```

同时记录 `execution_stats.resident_bytes`、`data_plane_bytes` 与 `dense_bytes`（FP16 dense
基准）。GPU flat-buffer 部署以 `data_plane_bytes / dense_bytes` 为驻留比；CPU 簿记口径
`resident_bytes / dense_bytes` 一并记录用于对照。注意区分“KV 驻留”与“模型权重+激活峰值”。

## 5. 结果记录模板

| 组 | 每步延迟 median[min-max] | 30 步总时间 | 峰值显存 | KV 驻留 | relL2 | top-1 |
|---|---|---|---|---|---|---|
| baseline | | | | | | |
| reuse | | | | | | |
| packed | | | | | | |
| combined | | | | | | |
| int8 packed | | | | | | |

## 6. 结论落档

原始输出、配置、seed、git SHA 保存到 `specs/archive/AR013-step-reuse-packed-execution/`；
README 只引用结论性表格。未达标结果如实标注，不以受控轨迹替代真实模型结论。
