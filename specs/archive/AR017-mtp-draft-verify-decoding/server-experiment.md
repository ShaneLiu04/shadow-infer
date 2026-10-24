# [AR017] 服务器复现实验协议：块级 MTP 草拟-验证并行解码

本协议用于在 GPU 服务器复核 AR017。本机证据（CPU、真实前向、四臂注册基准、86 新增
UT）覆盖机制正确性、两臂精确性（draft-verify paged vs 全量重算 draft-verify 逐位
一致 + 调度整表相等，9/9 点）、零回退界与坏草稿拒收；**随机权重玩具上 draft-verify
与 mask-init 持平（诚实负结果）**，MTP 轮数削减与端到端收益 contingent 于真实预训练
模型草稿质量，必须在服务器以真实权重复核。

## 1. 目标与判定

| 项 | 内容 |
|----|------|
| 被测 | `PyTorchModelBackend.generate` + `draft_mode=True`（round-0 全位草拟 + 自一致迭代验证至不动点 + 定点 KV 复用封存，fp32 页） |
| 对照 0 | `blockwise_generate_baseline`（mask-init 全量重算，AR016 golden） |
| 对照 1 | `blockwise_generate_baseline` + `draft_mode=True`（baseline-dv 决策等价参考臂） |
| 对照 2 | `generate` mask-init 模式（AR016 管线，fp32 页；S1-S3 的比较臂） |
| 主指标 | 端到端总时间、逐块耗时（`stats.block_time_ms`）、`rounds_per_block` / `rounds_to_fixpoint` 分布、`changed_per_round` 轨迹、`draft_acceptance_rate`、`computed_rows_total`、`kv_write_rows_total`（期望恒 == P+G） |
| 判定 S4（精确性锚点，权重无关） | draft-verify paged token 序列与 baseline-dv 逐位一致且调度统计整表相等（`rounds_per_block` / `total_rounds` / `rounds_to_fixpoint` / `changed_per_round` / `forced_completions` / `draft_acceptance_rate`）；CPU 实测 9/9 |
| 判定 S1（性能，需真实权重） | draft-verify vs mask-init 端到端 >= 1.5x（3 seeds 中位） |
| 判定 S2（轮数，需真实权重） | 每块轮数削减 >= 2x（`rounds_per_block` 中位比；Fast-dLLM/Jacobi 文献预期块内 O(1)-O(3) 轮） |
| 判定 S3（质量，需真实权重） | draft-verify vs mask-init token 一致率 >= 0.90（一致率口径，不伪称 bitwise） |
| 随机权重预期 | 若 `model_weights` 为空（AR018 之前）：draft-verify == mask-init 持平（CPU 玩具实测中位比 0.997-1.039x、轮数无削减、一致率随机水平）——如实报告，**不判失败**；S1-S3 仅在真实权重下判定 |
| 重复 | 3 seed x 3 repeats；GPU/驱动/torch/git SHA 必填 |

> 本机 CPU 证据（bench-results.md）：draft-verify paged 与全量重算参考臂逐位一致 +
> 调度整表相等（9/9 点），KV 行写一次零重写（== P+G，27/27 项），闭式记账确定性成立；
> 玩具持平根因 = 随机权重无稳定 argmax 不动点（changed 震荡平台 0.672-1.0 不衰减，
> W∈{8,16,32} 与 cap∈{8,16} 均不定点）；自一致校验正确拒收低质草稿（acceptance
> 0.042-0.065，持续替换而非错误锁定）。GPU 复核重点：(1) 真实预训练 dLLM（LLaDA /
> Dream 类）草稿质量下的 `draft_acceptance_rate` 与块内收敛轮数（文献预期 O(1)-O(3)）；
> (2) 轮数削减对端到端延迟的传导（块内前向数 = 1 草拟 + k 验证 + 定点零封存）；
> (3) 定点 KV 复用（定点块零封存前向）在真实负载的触发率。

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
$env:PYTHONPATH='.'; python specs/changes/AR017-mtp-draft-verify-decoding/ar017_bench.py
# 归档后（Linux/bash）：
PYTHONPATH=. python specs/archive/AR017-mtp-draft-verify-decoding/ar017_bench.py
```

终版注册断言（脚本内置，v2 诚实口径，全部通过）：A1-A5（见 bench-results.md §2/§3）。
预注册修订史（v1 三门槛机理不可达 → v2 诚实口径，用户确认）与校准探针数据完整落档于
bench-results.md §1。

## 4. 服务器 7B 实验

### 4.1 四臂命令

```bash
# 0) baseline（mask-init 全量重算，golden）
python specs/archive/AR017-mtp-draft-verify-decoding/serve_arm.py \
  --config configs/experiments/s12_draft_verify_gpu.yaml \
  --model-config configs/experiments/model_7b_longctx.yaml \
  --arm baseline --mode mask-init --seed 0 --device cuda \
  --out results/ar017_s12_baseline_seed0

# 1) baseline-dv（draft-verify 全量重算，精确性参考臂）
python specs/archive/AR017-mtp-draft-verify-decoding/serve_arm.py \
  --config configs/experiments/s12_draft_verify_gpu.yaml \
  --model-config configs/experiments/model_7b_longctx.yaml \
  --arm baseline --mode draft-verify --seed 0 --device cuda \
  --out results/ar017_s12_baseline_dv_seed0

# 2) mask-init paged（比较臂）
python specs/archive/AR017-mtp-draft-verify-decoding/serve_arm.py \
  --config configs/experiments/s12_draft_verify_gpu.yaml \
  --model-config configs/experiments/model_7b_longctx.yaml \
  --arm fp32 --mode mask-init --seed 0 --device cuda \
  --out results/ar017_s12_maskinit_seed0

# 3) draft-verify paged（被测臂）
python specs/archive/AR017-mtp-draft-verify-decoding/serve_arm.py \
  --config configs/experiments/s12_draft_verify_gpu.yaml \
  --model-config configs/experiments/model_7b_longctx.yaml \
  --arm fp32 --mode draft-verify --seed 0 --device cuda \
  --out results/ar017_s12_draftverify_seed0

# 每个 seed（0/1/2）重复；--prompt-len / --gen-len 可覆盖协议默认（256/512）
```

真实权重（AR018 后）：`model_weights` 配置位填入预训练 dLLM 权重路径/仓库名
（适配器交付于 AR018），重跑 2)/3) 两臂并按 S1-S3 判定。

### 4.2 质量取证

- 每臂输出 `artifacts.pt`（sequences + 逐轮 logits + 逐轮解掩位置）与 `stats.json`
  （含 `mode`、`draft_mode` 与三新键 `draft_acceptance_rate` / `rounds_to_fixpoint` /
  `changed_per_round`）。
- **S4（权重无关，每次必查）**：draft-verify paged 臂断言 `torch.equal(sequences)`
  与 baseline-dv，且六个调度统计键整表相等；若非逐位，先核对两臂 `rounds_per_block`
  （GPU 非确定性内核可在近并列 argmax 上翻转验证轮替换；此类点记轨迹分叉并报告
  token 一致率，CPU 上两臂从未分叉）。
- **S1-S3（真实权重）**：draft-verify vs mask-init 端到端中位比、`rounds_per_block`
  中位比、token 一致率；同时报告 `draft_acceptance_rate` 分布与 `rounds_to_fixpoint`
  分布（定点块占比即「定点 KV 复用零封存前向」的触发率）。
- 震荡特征取证：`changed_per_round` 逐轮轨迹（真实权重预期快速衰减至 0；若出现玩具
  型平台不衰减，如实记录并分析草稿质量）。

### 4.3 显存与页统计

```bash
nvidia-smi --query-gpu=memory.used --format=csv -l 1
```

同时记录 `stats` 键（`kv_write_rows_total` 应恒 == P+G；`computed_rows_total` 按
`Σ W×(调度前向+封存前向)` 闭式，定点块封存前向 = 0）；验证生成期间零页重写零逐出
（`backend.last_generate_stores` 的 `rewritten_tokens == 0`、`page_evictions == 0`、
`appended_tokens == P+G`，CPU UT 已证）。

## 5. 结果记录模板

| 臂 | 端到端时间 median[min-max] | 总轮数/块 | rounds_to_fixpoint 分布 | acceptance | computed_rows | kv_writes | token 一致（vs 对照） |
|---|---|---|---|---|---|---|---|
| baseline (mask-init) | | | - | - | | 0（无页存） | - |
| baseline-dv | | | | | | 0（无页存） | S4 参考 |
| mask-init paged | | | - | - | | | - |
| draft-verify paged | | | | | | | S4: vs baseline-dv；S3: vs mask-init |

## 6. 结论落档

原始输出、配置、seed、git SHA 保存至 `specs/archive/AR017-mtp-draft-verify-decoding/`；
README 只引用结论性表格。未达标结果如实标注，不以玩具基准替代真实模型结论；CPU 端
「随机权重持平」负结果与 GPU 实测并列呈现，不选择性引用；S1-S3 在随机权重下不判定
（预期持平），仅真实权重（AR018）复核后进入 README 性能主张。
