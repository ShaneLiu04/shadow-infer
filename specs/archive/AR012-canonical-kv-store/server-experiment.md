# [AR012] 服务器复现实验协议：规范化紧凑 KV Store

本协议用于在 GPU 服务器复核 AR012。本机证据（CPU、真实张量）覆盖向量化读写与驱逐微基准、
INT4 batch 修正、单源存储与全量回归；端到端模型与 GPU 驻留必须在服务器复核。

## 1. 目标与判据

| 项 | 内容 |
|----|------|
| 被测 | v3.14.0：向量化 `unpack` / `apply_reuse_mask` / 桶内 `evict`，`store_dense_kv=false` |
| 对照 | v3.13.0（`6e27ea4`..`1f49093`）同 flags；或 v3.14.0 下 `store_dense_kv=true` |
| 主指标 | 存储读路径延迟、驱逐延迟、KV 驻留字节（`resident_bytes`）、峰值显存、端到端每步延迟 |
| 精度指标 | `unpack` 与 golden 逐位一致；INT4 B>1 每 batch 相对 MSE；端到端 logits/top-1 |
| 判定 | 读/复用/驱逐相对基线加速 >=5x/3x/10x；INT4 每 batch MSE <=0.02；驱逐后字节真实下降；端到端不劣化 |
| 重复 | >=3 次，报告 median 与 min-max；GPU/CUDA/torch/git SHA 必填 |

> 注意：本 AR 不把压缩 KV 回灌 attention，因此**不宣称端到端模型加速**。端到端只验证
> 语义等价/不劣化与真实驻留下降。

## 2. 环境

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt && pip install -e ".[dev]"
python -c "import torch; print(torch.__version__, torch.cuda.get_device_name(0))"
nvidia-smi
```

## 3. 实验命令

### 3.1 微基准（真实张量，含断言）

```bash
python specs/changes/AR012-canonical-kv-store/ar012_bench.py
```

覆盖 S=512/1024/2048、B=1/H=8/D=64 混合精度；输出 unpack/reuse/evict 中位耗时与字节下降，
并断言 INT4 B=2 每 batch 相对 MSE <=0.02。

### 3.2 目标形状扫描（GPU）

将 `ar012_bench.py` 的 `run(...)` 扩展到目标模型的 `(seq, batch, heads, head_dim)`，至少 3 个
规模点（如 2K/8K/32K），并与 v3.13.0 的旧实现对照。记录：

- 各规模 unpack/reuse/evict median [min-max]；
- `memory_bytes()` 与 `resident_bytes()`；
- 驱逐前后实际驻留（`nvidia-smi` 与 manager 口径）。

### 3.3 端到端（s7）

```bash
python scripts/run_experiment.py \
  --config configs/experiments/s7_kv_store_gpu.yaml \
  --model-config configs/experiments/model_7b_longctx.yaml \
  --backend hf --model-name <hf-model-id> \
  --device cuda --steps 24 --repeats 3 --out results/ar012_s7
```

对照：v3.14.0 下 `shadowkv.store_dense_kv: true`（保留 dense shadow），输出到
`results/ar012_s7_dense`；以及 v3.13.0 同命令。

## 4. 结果记录模板

| 规模 (S/B/H/D) | unpack 旧/新 (ms) | reuse 旧/新 (ms) | evict 旧/新 (ms) | 驻留下降 | INT4 MSE (per batch) |
|---|---|---|---|---|---|
| 待填 | | | | | |

端到端：

| 版本/flags | 每步延迟 median [min-max] | 峰值显存 | logits 相对 L2 | top-1 一致率 |
|---|---|---|---|---|
| v3.13.0 | | | | |
| v3.14.0 dense=true | | | | |
| v3.14.0 dense=false | | | | |

## 5. 结论落档

实验数据（配置、原始输出、命令、git SHA）保存到 `specs/changes/AR012-canonical-kv-store/`；
README 仅引用结论性表格，未达标结果如实标注，不外推小尺寸结论。
