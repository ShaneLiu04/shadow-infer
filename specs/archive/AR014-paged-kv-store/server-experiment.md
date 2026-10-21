# [AR014] 服务器复现实验协议：分页增量 KV 驻留

本协议用于在 GPU 服务器复核 AR014。本机证据（CPU、真实张量、逐位 golden）覆盖 KV 执行平面
微基准与全量 UT；7B 规模 GPU 延迟与显存必须在服务器复核。

## 1. 目标与判定

| 项 | 内容 |
|----|------|
| 被测 | `execution.kv_lifecycle=append` + `kv_execution=paged`（`page_size=32`） |
| 对照 1 | `append` + `kv_execution=packed`（现行基线：每步全量 unpack + repack） |
| 对照 2 | `append` + `kv_execution=dense`（上界参照） |
| 主指标 | 每步延迟 median[min-max]（期望 paged 增长近平坦、packed 随 S 线性）、30 步总时间、峰值显存、KV 驻留字节 |
| 质量指标 | 每步 logits 相对 L2、top-1 一致率；fp16 臂必须逐位等于对照 1 |
| 统计键 | `paged` / `page_size` / `page_count` / `appended_tokens` / `appended_tokens_total` / `repack_tokens_saved` / `page_evictions` |
| 判定 | paged 每步延迟斜率显著低于 packed（参照本机斜率比 0.09-0.12）；fp16 logits relL2 = 0（逐位）；显存不高于 packed；驻留 `data_plane_bytes` 与 dense FP16 基准一致（fp16 无压缩，符合预期） |
| 重复 | 3 seed × 3 repeats，报告 median[min-max]；GPU/驱动/torch/git SHA 必填 |

> 本机 CPU 证据：KV 执行平面微基准（append 簿记）中位加速 9.3x（512）与 10.2x（2048），
> 斜率比 0.088-0.114，Python 堆瞬态峰值降约 51-57x，fp16 逐位等价；原始数据与诚实说明见
> `bench-results.md`。差额主要来自基线每步每层 O(S·H) 的 precision_map 重建与分组 Python
> 循环；GPU 上该 Python 开销占比可能不同，以服务器实测为准。

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
$env:PYTHONPATH='.'; python specs/changes/AR014-paged-kv-store/ar014_bench.py
# 归档后（Linux/bash）：
PYTHONPATH=. python specs/archive/AR014-paged-kv-store/ar014_bench.py
```

预注册断言（脚本内置，全部通过）：总加速 >=1.5x、斜率比 <0.60、Python 堆峰值 paged<baseline、
fp16 `torch.equal` 等价。规模点 (N=8, S_new=64, max_seq=512) 与 (N=16, S_new=128, max_seq=2048)，
维度 B=4/H=8/D=64/L=8 层/fp16/page_size=32，3 seeds。原始逐 seed 数据见 `bench-results.md`。

## 4. 服务器 7B 实验

### 4.1 对照臂（3 组）

```bash
# 0) baseline: append + packed（每步全量 unpack+repack）
python scripts/run_experiment.py \
  --config configs/experiments/s9_paged_longctx_gpu.yaml \
  --model-config configs/experiments/model_7b_longctx.yaml \
  --backend pytorch --device cuda --steps 30 --repeats 3 \
  --out results/ar014_s9_packed
# 运行时将 execution.kv_execution=packed

# 1) paged: append + paged（s9 默认臂）
#    kv_execution=paged, page_size=32（s9 默认）

# 2) dense: append + dense（上界参照）
#    kv_execution=dense
```

每组的输出目录独立；每个 seed 修改 `--seed` 后重复。

### 4.2 质量取证

- 保存每组每步 logits（或最后一步）到 `results/ar014_s9_*/logits.pt`。
- 以对照 1（packed）为参考计算 `||logits - ref|| / ||ref||` 与 top-1 一致率。
- 质量门槛：fp16 臂 paged relL2 = 0（逐位，与 packed 同精度解码等价）；若任何精度混合臂
  relL2 超限，降低步数或固定 `packed_precision=fp16` 重跑并在报告中如实标注。

### 4.3 显存与驻留

```bash
nvidia-smi --query-gpu=memory.used --format=csv -l 1
```

同时记录 `execution_stats.resident_bytes`、`data_plane_bytes`、`dense_bytes` 与 paged 专属键
（`page_count` 随 append 增长、`appended_tokens_total` == 总新 token 数、`repack_tokens_saved`
为相对全量 repack 节省的 token 数）。注意区分「KV 驻留」与「模型权重/激活峰值」。

## 5. 结果记录模板

| 臂 | 每步延迟 median[min-max] | 30 步总时 | 峰值显存 | KV 驻留 | page_count | repack_tokens_saved | relL2 | top-1 |
|---|---|---|---|---|---|---|---|---|
| packed | | | | | - | - | | |
| paged | | | | | | | | |
| dense | | | | | - | - | | |

## 6. 结论落档

原始输出、配置、seed、git SHA 保存至 `specs/archive/AR014-paged-kv-store/`；README 只引用
结论性表格。未达标结果如实标注，不以微基准替代真实模型结论。
