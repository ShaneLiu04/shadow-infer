# [AR016] 服务器复现实验协议：置信度感知块式并行解码

本协议用于在 GPU 服务器复核 AR016。本机证据（CPU、真实前向、四臂注册基准、108 UT）覆盖
机制正确性、fp32 精确性（逐位）、fp16/int8 量化取舍与端到端玩具基准；7B 规模 GPU 延迟、
显存与真实负载收益必须在服务器复核。

## 1. 目标与判定

| 项 | 内容 |
|----|------|
| 被测 | `PyTorchModelBackend.generate`（块式并行解码：逐块置信度解掩 + PagedKVStore 前缀页驻留，`page_precision=fp32`） |
| 对照 0 | `blockwise_generate_baseline`（同调度、每轮全量重算 forward_step + 段因果掩码；golden） |
| 对照 1/2 | `generate` + `page_precision=fp16` / `int8`（量化页平面） |
| 主指标 | 端到端总时间、逐块耗时（`stats.block_time_ms`）、总轮数、`computed_rows_total`、`kv_write_rows_total`（期望恒 == P+G） |
| 质量指标 | fp32 臂逐轮 logits relL2 + token 序列逐位一致性（vs 对照 0）；fp16/int8 臂 relL2 + token 一致率 + 轨迹对齐标记（逐轮解掩位置日志） |
| 判定（fp32 臂） | token 序列逐位一致且逐轮 relL2 <= 1e-6（3 seeds 最差；CPU 实测 0 / 3.25e-7） |
| 判定（fp16 臂） | relL2 <= 1e-4 且 token 逐位一致（CPU 实测 1.24e-5、9/9 逐位） |
| 判定（int8 臂） | token 一致率 >= 0.99（CPU 实测 [0.9961, 1.0]）；轨迹对齐点 relL2 <= 1e-3（CPU 实测 3.6e-4）；分叉点（阈值边缘调度置换 / 近并列 argmax 翻转）只报告 |
| 加速比 | fp32 vs 对照 0 报告 median[min-max]；CPU 玩具负载证据为 2.29x/3.68x/7.08x（S=256/512/1024 中位），S=256 诚实低于 3x（固定开销占比高）；GPU 7B 规模是否保持由服务器判定 |
| 重复 | 3 seed x 3 repeats；GPU/驱动/torch/git SHA 必填 |

> 本机 CPU 证据（bench-results.md）：fp32 臂与全量重算参考环路逐位一致（tokens + 全轮 logits
> relL2 <= 3.25e-7），KV 行写一次零重写零逐出（kv_write_rows == P+G，27/27 点）；加速由
> 「baseline 逐轮重算 L+W 行 vs 活跃前向 W 行」的行裁剪驱动（行缩减 4.9/8.4/15.4x）。
> GPU 复核重点：(1) 7B 维度（32 层 / 128 head_dim）下行裁剪收益是否随注意力占比上升而放大；
> (2) 真实 dLLM 置信度分布（预期远高于玩具模型 0.004-0.010）下 `confidence_threshold=0.9`
> 的并行解掩触发率与轮数收敛；(3) PagedKVStore gather 重建在 GPU 上的开销占比。

## 2. 环境

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt && pip install -e ".[dev]" && pip install pyyaml
python -c "import torch; print(torch.__version__, torch.cuda.get_device_name(0))"
nvidia-smi
```

## 3. 本机基线（可复跑）

```bash
# 归档前（Windows/PowerShell）：
$env:PYTHONPATH='.'; python specs/changes/AR016-blockwise-parallel-decoding/ar016_bench.py
# 归档后（Linux/bash）：
PYTHONPATH=. python specs/archive/AR016-blockwise-parallel-decoding/ar016_bench.py
```

终版预注册断言（脚本内置，判据 v3，全部通过）：A1-A9（见 bench-results.md §2/§3）。
校准历史（threshold 0.9 机制退化修订）与 A6 判据三轮修订（v1 bitwise 代理证伪 / v2
位置级实现 bug 与逻辑不完备 / v3 轨迹对齐终版）完整落档于 bench-results.md §1/§2.1。

## 4. 服务器 7B 实验

### 4.1 四臂命令

```bash
# 0) baseline：全量重算参考环路（golden）
python specs/archive/AR016-blockwise-parallel-decoding/serve_arm.py \
  --config configs/experiments/s11_blockwise_decoding_gpu.yaml \
  --model-config configs/experiments/model_7b_longctx.yaml \
  --arm baseline --seed 0 --device cuda --out results/ar016_s11_baseline_seed0

# 1) fp32 精确臂
python specs/archive/AR016-blockwise-parallel-decoding/serve_arm.py \
  --config configs/experiments/s11_blockwise_decoding_gpu.yaml \
  --model-config configs/experiments/model_7b_longctx.yaml \
  --arm fp32 --seed 0 --device cuda --out results/ar016_s11_fp32_seed0

# 2) fp16 / 3) int8：同上，--arm fp16 / --arm int8
# 每个 seed（0/1/2）重复；--prompt-len / --gen-len 可覆盖协议默认（256/512）
```

阈值扫描（报告性）：`confidence_threshold` 在 {0.9, 0.8, 0.6} 各跑 fp32 臂一次，
观察真实模型置信度分布下的并行解掩触发率（`stats.fallback_per_round` 占比）与
轮数收敛（`rounds_per_block`）；玩具模型证据表明加速比由行裁剪驱动、对阈值鲁棒，
但绝对时延随轮数下降而下降（bench-results.md §3.4）。

### 4.2 质量取证

- 每臂输出 `artifacts.pt`（sequences + 逐轮 logits + 逐轮解掩位置）与 `stats.json`。
- 以对照 0 为参考：fp32 臂断言 `torch.equal(sequences)` 且逐轮 relL2 <= 1e-6；
  fp16 断言逐位 + relL2 <= 1e-4；int8 报告 token 一致率与轨迹对齐标记
  （逐轮解掩位置相等 ∧ 终态逐位 ⟹ relL2 才与量化误差同口径，判据 v3，
  见 bench-results.md §2.1 调度置换伪影机制）。
- 若 fp32 臂出现非逐位：先核对两臂 `rounds_per_block` 与逐轮解掩位置（GPU
  非确定性内核可在近并列置信度上翻转调度；此类点按判据 v3 记轨迹分叉并
  报告 token 一致率，不判失败——CPU 上两臂共享调度从未分叉，GPU 首次复核
  需区分「数值误差」与「调度置换」）。

### 4.3 显存与页统计

```bash
nvidia-smi --query-gpu=memory.used --format=csv -l 1
```

同时记录 `stats` 键（`kv_write_rows_total` 应恒 == P+G；`computed_rows_total`、
`rounds_per_block`、`unmasked_per_round`、`fallback_per_round`、
`mean_confidence_per_round`、`block_time_ms`）；验证生成期间零页重写零逐出
（`backend.last_generate_stores` 的 `rewritten_tokens == 0`、
`page_evictions == 0`、`appended_tokens == P+G`，CPU UT 已证）。注意区分
「KV 驻存」与「模型权重/激活峰值」。

## 5. 结果记录模板

| 臂 | 端到端时间 median[min-max] | 逐块耗时中位 | 总轮数 | computed_rows | kv_writes | relL2 | token 一致 |
|---|---|---|---|---|---|---|---|
| baseline | | | | | 0（无页存） | - | - |
| fp32 | | | | | | | |
| fp16 | | | | | | | |
| int8 | | | | | | | |

## 6. 结论落档

原始输出、配置、seed、git SHA 保存至 `specs/archive/AR016-blockwise-parallel-decoding/`；
README 只引用结论性表格。未达标结果如实标注，不以玩具基准替代真实模型结论；CPU 端
小规模（S=256）加速低于 3x 的负结果与 GPU 实测并列呈现，不选择性引用。
