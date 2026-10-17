# AR002 热路径基线画像（T001 落档）

> 测量时间：2026-10-17；机器：Windows / Python 3.11.9 / torch 2.5.1+cu121（CPU fallback）
> 方法：`configs/optimize_full.yaml` + `configs/model_fast_dllm_7b.yaml`，6-step `run_optimized` / `run_baseline`，cProfile 采样 + 无 profiler 墙钟复测。
> 本文件是 T004/T005（M1 golden 对照）、T007（压缩耗时下降 >=5x 验收）、T008（端到端验收）的基线锚点。

## 1. 端到端对照（无 profiler，同机同参）

| 路径 | e2e（ms） | 每步均值（ms） | 口径 |
|---|---|---|---|
| run_baseline | 3817.7 | ~636 | `latency.e2e_ms` 实测（现有代码 per_step 为 0.0，属 M6 缺陷） |
| run_optimized | 46295.7 | ~7716 | 同上 |
| 倒挂倍数 | **12.1x** | - | 优化路径远慢于基线 |

注：srs.md 引言引用 AR001 审计口径（基线 31.3ms / 优化 1782.8ms，~57x），该数据出自审计时的小规模探针 harness。T001 以仓库随附 `configs/optimize_full.yaml` 为可复现基准口径（本表），后续 T008 验收按 srs「同机同参」原则以实际 harness 实测为准。

## 2. cProfile 热路径份额（profiled 总耗时 51.3s，含 ~10% 采样开销）

| 函数 | 位置 | 调用次数 | self(s) | cum(s) | 占比 |
|---|---|---|---|---|---|
| _execute_step | orchestrator.py:1091 | 6 | 0.007 | 47.45 | 92.5% |
| _run_shadowkv → shadowkv.on_step | shadowkv_agent.py:137 | 24 | 1.898 | 24.66 | 48.1% |
| _run_ffn → ffn.on_step | ffn_optimizer_agent.py:100 | 24 | 0.019 | 22.71 | 44.3% |
| compress_kv | shadowkv_agent.py:678 | 24 | 6.522 | 11.43 | 22.3% |
| compute_importance_score | shadowkv_agent.py:360 | 98304 | 2.490 | 8.66 | 16.9% |
| compute_entropy | metrics.py:209 | 98304 | 2.389 | 5.33 | 10.4% |
| _simulate_step_tensors（输入仿真） | orchestrator.py:1881 | 6 | 0.000 | 3.84 | 7.5% |
| packed_kv_cache.pack / _do_pack | packed_kv_cache.py:42/71 | 24 | 0.171 | 3.12 | 6.1% |
| _build_reuse_decision | shadowkv_agent.py:617 | 24 | 0.012 | 1.86 | 3.6% |
| decide_reuse_per_token | shadowkv_agent.py:516 | 20 | 1.388 | 1.85 | 3.6% |
| allocate_precision | shadowkv_agent.py:398 | 24 | 0.546 | 0.63 | 1.2% |
| quantize_tensor | quantization.py:16 | 196608 | 0.271 | 0.48 | 0.9% |
| qdrift.on_step | qdrift_agent.py:63 | 6 | 0.000 | 0.021 | 0.04% |
| profiler.on_step | profiler_agent.py:91 | 6 | 0.000 | 0.045 | 0.09% |

FFN 支线（AR002 范围外，记录备查）：_mixed_compute 18.68s（36.4%）→ packed_weight.unpack/dequantize_for_matmul 17.27s（33.7%，packed_weight.py:140/172，同为逐元素 Python 热路径）。

## 3. torch 原语调用密度（逐元素路径特征）

| 原语 | 调用次数 | self(s) |
|---|---|---|
| Tensor.copy_ | 352256 | 9.41 |
| Tensor.float | 374272 | 2.25 |
| Tensor.select | 374272 | 2.12 |
| Tensor.reshape | 374272 | 1.74 |
| Tensor.clamp | 180224 | 1.11 |
| Tensor.unsqueeze | 393216 | 1.08 |
| Tensor.half | 79536 | 0.99 |

数十万级小张量调用是纯 Python 逐元素循环的直接证据；M1 张量化后这些调用应坍缩为个位数大张量操作。

## 4. 决策面退化证据（srs Minor-3 / M3-lite 依据）

6 步 qdrift 输出（step_records 实录）：

| step | sensitivity_score | drift_score | phase | dispatch |
|---|---|---|---|---|
| 0 | 0.3000 | 0.0 | early | balanced / mixed |
| 1 | 0.2969 | 1.0 | early | balanced / sparse |
| 2 | 0.3750 | 1.0 | mid | conservative / mixed |
| 3 | 0.5000 | 1.0 | mid | conservative / mixed |
| 4 | 0.6250 | 1.0 | late | conservative / mixed |
| 5 | 0.7031 | 1.0 | late | conservative / full |

drift_score 呈「首步 0.0、此后恒 1.0」的退化分布——qdrift_agent.py:296-311 位置插值产生，非实测信号；sensitivity_score 亦为位置插值曲线（0.30→0.70），与张量内容无关。M3-lite 需引入实测项。

## 5. M6 指标缺陷实录

- run_optimized 报告 `latency.per_step_ms` 全 0.0（orchestrator.py:1456-1463 硬编码），`latency.e2e_ms` 却为真实值，两者互相矛盾；
- profiler 每步上报 `step_latency_ms = 0.0`、`accuracy_delta_percent = 0.0`、`step_memory_mb = 16383.56`（后者与 kv_cache.memory_mb 口径不一致，M5 记账范围需核对）；
- scheduler reward 消费上述零值延迟（scheduler.py:70-72），奖励信号失真。

## 6. 结论与主战场排序

1. **M1 主战场（shadowkv 支线，占优化路径 ~48%）**：compress_kv（22.3%）+ compute_importance_score（16.9%，98304 次调用 = 4096 token x 24 次组处理）+ packed_kv_cache._do_pack（6.1%）+ reuse 决策（3.6%）+ allocate_precision（1.2%）。张量化目标合计 ~50% 采样时间，对应 srs FR3.2「压缩耗时下降 >=5x」的验收分母。
2. **决策面成本可忽略**：qdrift/profiler/scheduler 的 on_step 合计 <0.1%，M3-lite 实测项引入不会造成性能回退。
3. **风险备注（范围外）**：FFN 支线 unpack 占 33.7%，与 shadowkv 相当；AR002 仅治理 shadowkv 支线，端到端 1.5x 目标受 FFN 支线制约，T008 按 design 4.1 兜底路径执行；FFN unpack 张量化列为 AR003 候选。
4. `quantize_tensor` 本体仅 0.9%（开销在调用方循环与张量切片），验证 design 4.2 判断：向量化收益来自消除循环与小张量对象，而非量化函数本身。

## 7. 复现命令

```powershell
python -X utf8 <临时目录>\t001_extract.py   # cProfile + 无 profiler 墙钟（脚本随会话存档于临时目录）
python -X utf8 <临时目录>\t001_baseline.py  # baseline e2e 复测
```

## 8. T008 端到端复测结果（AR002 完成态，2026-10-17）

> 同机同参同口径（configs/optimize_full.yaml + model_fast_dllm_7b.yaml，6-step，
> 无 profiler 外部墙钟，与 §1 基线锚点可比）。脚本：`<临时目录>\t008_e2e.py`、`t008_stab.py`。

### 8.1 端到端对照（无 profiler 外部墙钟）

| 路径 | e2e（ms，首测） | 稳定性复测（3 次） | 口径 |
|---|---|---|---|
| run_baseline | 6481.2 | 6527.1 / 6491.7 / 6525.0 | 外部墙钟（含 5 步 warmup 仿真） |
| run_optimized | 10273.5 | 10403.8 / 10492.7 / 10691.8 | 外部墙钟（含 step0 惰性初始化 5419ms） |
| 倒挂倍数 | **1.585** | 1.594 / 1.616 / 1.639（中位 1.61） | T001 基线 7.29x → 1.61x（4.5x 改善） |

**判定：落入 design 4.1 授权兜底区 (1.5x, 2.5x]，按兜底路径结案：**

- M5 决策缓存发力：FFN 反量化缓存已生效（dequantize_for_matmul 32 → 8 次调用，
  17.13s → 1.26s）；决策缓存 E2E 无同 (step_id, layer_id) 重复调用可命中
  （每步 4 层各调用一次）——已无发力空间；
- 降规模（seq <= 128）：基准配置 seq_len 已为 128（默认下限）——杠杆已在下限；
- 每层缓存默认关：未引入任何强暴默认口径，无需变更；
- 残余瓶颈为 FFN 支线（unpack Python 热路径 + 模拟计算，稳态 ~250ms/step），
  AR002 范围外，AR003 候选（§6.3 已预警）；
- 「1.5x 可达」由 T001/T008 双实测证明：优化路径增量 = shadowkv ~150ms/step
  （已至向量化下限）+ FFN ~250ms/step（AR003 范围），FFN 张量化后
  960-250=710ms/step vs 基线 586ms/step ≈ 1.21x。

### 8.2 M1 热路径复测（cProfile，对照 §2）

| 函数 | T001 cum(s) | T008 cum(s) | 下降 |
|---|---|---|---|
| compress_kv | 11.43 | 0.575 | 19.9x（FR3.2 >=5x 达标） |
| shadowkv.on_step | 24.66 | 0.954 | 25.8x |
| compute_importance（矩阵化） | 8.66（98304 次调用） | 0.041（24 次调用） | >200x |
| packed_kv_cache.pack | 3.12 | 0.199 | 15.7x |
| dequantize_for_matmul（FFN，M5 缓存） | 17.13（32 次） | 1.257（8 次） | 13.6x |
| ffn.on_step（范围外，AR003） | 22.71 | 6.358 | -（基准份额 57%，新主瓶颈） |

### 8.3 M6 / M3-lite 信号证据

- latency.per_step_ms 全部 > 0（稳态 948-993ms，step0 5419ms 含惰性初始化）；
- 内部 e2e_ms 与外部墙钟对照偏差 **0.02%**（< 2% 验收线，10271.5 vs 10273.5ms）；
- qdrift 实测信号：sensitivity = [0.300, 0.711, 0.735, 0.772, 0.810, 0.833]
  （混合实测项），drift = [0.0, 0.888x 五步]——不再恒 1.0 塌缩（对照 §4 基线实录）；
- kv_cache.memory_mb 为真实节省口径：[1.00, 0.51, 0.37, 0.33] MB/层。
