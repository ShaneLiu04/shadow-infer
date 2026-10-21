# [AR015] 服务器复现实验协议：收敛感知宏块 KV 模式控制

本协议用于在 GPU 服务器复核 AR015。本机证据（CPU、真实张量、golden 容差、95 UT）覆盖机制正确性、
误差有界语义与端到端玩具基准；7B 规模 GPU 延迟、显存与真实负载收益必须在服务器复核。

## 1. 目标与判定

| 项 | 内容 |
|----|------|
| 被测 | `execution.kv_lifecycle=replace` + `kv_execution=paged` + `mode_control_enabled=true`（`macro_block_size=64`） |
| 对照 1 | `replace` + `dense`（无门控 golden，兼作 relL2 参考） |
| 对照 2 | `replace` + `packed` + step reuse（AR013 臂，threshold=0.1） |
| 保险臂 | 被测 + `force_refresh_every=2`（精度保险配置，CPU 证据的断言对象） |
| 主指标 | 每步延迟 median[min-max]、30 步总时间、峰值显存、冻结率曲线（`skipped_rows/(L*S)` 逐步） |
| 质量指标 | 每步 logits 相对 L2（vs 对照 1）、top-1 一致率 |
| 统计键 | `mode_control` / `macro_block_size` / `skip_blocks` / `delta_blocks` / `refresh_blocks` / `skipped_rows` / `rewritten_rows` / `projected_rows` |
| 判定（保险臂） | 全步 relL2 <= 3%（3 seeds 最差）；末半程平均冻结率 >= 30%；加速比与对照 2 比较**只报告不断言**（CPU 证据已证明玩具负载上 >=2x 不可达，见 bench-results.md 三轮预注册历史） |
| 边界臂 | `force_refresh_every=0`（默认臂）只报告：观察 SKIP 陈旧累积与漂移门自愈轨迹（CPU: relL2 峰值 6.7-10.1% 后回落） |
| 重复 | 3 seed x 3 repeats，报告 median[min-max]；GPU/驱动/torch/git SHA 必填 |

> 本机 CPU 证据（bench-results.md）：f2 保险臂 relL2 worst 1.88%/2.65%/1.99%（S=256/512/1024）、
> 末半程冻结 0.667、端到端加速 0.64-1.04x（诚实结论：CPU 玩具负载无收益，误差有界语义成立）。
> GPU 复核重点：真实块扩散负载的注意力耦合结构（局部/奇偶 vs 均匀）是否显著降低 SKIP 陈旧误差、
> 以及 GPU 上行粒度重组开销占比是否大幅下降——两者共同决定端到端收益是否存在。

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
$env:PYTHONPATH='.'; python specs/changes/AR015-macro-block-mode-control/ar015_bench.py
# 归档后（Linux/bash）：
PYTHONPATH=. python specs/archive/AR015-macro-block-mode-control/ar015_bench.py
```

终版预注册断言（脚本内置，f2 保险臂，全部通过）：全步 relL2 <= 3%、末半程冻结率 >= 30%；
加速比只报告。四臂 x 双轨迹（T1 均匀边界 / T2 块式半 AR）x 3 规模 x 3 seeds。
三轮预注册历史与 f-扫描不可行性证据见 `bench-results.md`。

## 4. 服务器 7B 实验

### 4.1 对照臂（4 组）

```bash
# 0) baseline: replace + dense（无门控 golden）
python scripts/run_experiment.py \
  --config configs/experiments/s10_mode_control_gpu.yaml \
  --model-config configs/experiments/model_7b_longctx.yaml \
  --backend pytorch --device cuda --steps 30 --repeats 3 \
  --out results/ar015_s10_dense
# 运行时将 execution.kv_execution=dense, mode_control_enabled=false

# 1) ar013: replace + packed + step reuse
#    kv_execution=packed, step_reuse_enabled=true, step_reuse_threshold=0.1,
#    mode_control_enabled=false

# 2) ar015 默认臂: replace + paged + mode control（s10 默认配置）
#    kv_execution=paged, mode_control_enabled=true, force_refresh_every=0

# 3) ar015_f2 保险臂: 同 2 + force_refresh_every=2（判定臂）
```

每组的输出目录独立；每个 seed 修改 `--seed` 后重复。

### 4.2 质量取证

- 保存每组每步 logits（或最后一步）到 `results/ar015_s10_*/logits.pt`。
- 以对照 0（dense）为参考计算 `||logits - ref|| / ||ref||` 与 top-1 一致率。
- 质量门槛：保险臂 relL2 <= 3%；默认臂如实报告累积-自愈轨迹，不作门槛。
- 若保险臂 relL2 超限，提高 `force_refresh_every` 刷新频率（如每 2 步强制全刷已是 f2；
  再超限说明真实负载耦合更强，如实记录并在报告中讨论 skip_threshold 调参空间）。

### 4.3 显存与模式统计

```bash
nvidia-smi --query-gpu=memory.used --format=csv -l 1
```

同时记录 `execution_stats` 模式控制八键（`mode_control`/`macro_block_size`/`skip_blocks`/
`delta_blocks`/`refresh_blocks`/`skipped_rows`/`rewritten_rows`/`projected_rows`）与冻结率曲线；
验证 SKIP 步 `rewritten_rows` 仅含活跃块行（零页写语义，CPU UT 已证）。注意区分「KV 驻留」
与「模型权重/激活峰值」。

## 5. 结果记录模板

| 臂 | 每步延迟 median[min-max] | 30 步总时 | 峰值显存 | 末半程冻结率 | relL2 | top-1 |
|---|---|---|---|---|---|---|
| dense | | | | - | - | - |
| ar013 | | | | - | | |
| ar015 (f=0) | | | | | | |
| ar015_f2 | | | | | | |

## 6. 结论落档

原始输出、配置、seed、git SHA 保存至 `specs/archive/AR015-macro-block-mode-control/`；README 只引用
结论性表格。未达标结果如实标注，不以玩具基准替代真实模型结论；CPU 端「无端到端收益」的负结果
与 GPU 实测并列呈现，不选择性引用。
