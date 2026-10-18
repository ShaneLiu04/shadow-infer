# AR004 T001 基线取证（baseline-profile.md）

| 字段 | 内容 |
|------|------|
| 取证日期 | 2026-10-18 |
| 执行设备 | CPU（Windows，torch 2.5.1 CPU-only） |
| 性质 | 缺陷与热点取证（微基准，非大型模型实验；收益对照于 T007 执行） |
| 复现脚本 | `specs/changes/AR004-ffn-hotpath/ar004_bench.py`（同目录，seed=42） |

## 1. 微基准配置

| 项 | 值 |
|----|----|
| hidden_dim / intermediate_dim | 256 / 1024（真实 7B 为 4096 / 11008，热点随通道数线性放大） |
| layers | 4（逐层调用 on_step） |
| seq / batch | 32 / 1 |
| 计算路径 | mixed_full（packed weights） |
| 通道剪枝 | 关闭（隔离热路径；掩码缺陷单独取证） |
| 计时 | 3 次独立 agent，每次 12 次 on_step，取中位 |

## 2. 基线数值

| 指标 | 实测 |
|------|------|
| `per_on_step_ms`（普通计时，3 次独立 agent，每次 warmup 后 12 calls） | 1.88 / 1.97 / 1.60，**中位 1.88 ms** |
| `per_on_step_ms`（第 2 次复跑） | 1.80 / 1.97 / 2.05，中位 1.97 ms；第 3 次 1.98 / 1.93 / 1.95，中位 1.95 ms（机器抖动；T007 以同批对照口径报告） |
| `_precision_plan_fingerprint` | **145.4~194.6 µs/调用**（复测 145.4 / 151.4 / 170.1 / 185.3 / 194.6 µs；1024 channels，O(channels) 元组重建） |
| `dequantize_for_matmul`（缓存命中） | 167.8~219.2 µs/调用（几乎全为指纹开销） |
| cProfile 占比（12 calls / 49 ms，profiler 开销下 ~4.08 ms/call） | `dequantize_for_matmul` **39%**（其中指纹 **37%**）；`quantization_info` 统计 **20%**（每步 3 次 O(channels) sum） |

## 3. 缺陷取证

**FFN-P1 层状态共享**：`_weight_up/_weight_down/weight_up_quantized/channel_importance/precision_scales` 均为单份全局状态；`on_step` 首次缓存后，无论 `layer_id` 为何都复用第一层权重。mock 路径各层权重相同而掩盖；真实多层模型输出错误。

**混合路径掩码缺陷**：`channel_pruning_enabled=True` + `packed weights` + sensitivity<0.7 时，`_mixed_compute` 仅对 up 投影应用行掩码，down 投影未应用列掩码：

```
RuntimeError: mat1 and mat2 shapes cannot be multiplied (32x716 and 1024x256)
```

（active 通道数 716，down 权重仍为完整 1024 列。）

脚本复现：`ar004_bench.py::repro_mixed_mask_defect()`，输出
`mask_defect: RuntimeError: mat1 and mat2 shapes cannot be multiplied (32x716 and 1024x256)`。

**FFN-P1 脚本复现**：`ar004_bench.py::repro_layer_state_defect()` 依次以 layer 0（w1）与 layer 1（w2）调用 on_step：
layer 1 输出与「w1 直接计算」max|Δ| = 0.0，与「w2 直接计算」max|Δ| = 0.5771 → 证明层 1 静默复用层 0 权重。

## 4. 复现命令

```powershell
python specs/changes/AR004-ffn-hotpath/ar004_bench.py
```

脚本输出：cProfile top 12、3 次 per_on_step 中位（计时前每 agent warmup）、指纹/dequant 微秒级耗时、掩码缺陷与 FFN-P1 复现结果。结果确定性（seed=42，CPU）。

## 5. 目标（T007 对照）

对照口径（规则 1.1）：T007 在同一会话、同配置下执行「优化路径 vs 参考路径」各 3 次，报告中位与区间，以**同批比值**判定；下表 1.88/1.97 ms 仅作取证参考，不作为跨会话固定基线。结构指标（指纹 ≤10µs、向量化 ≤50% 参考）为确定性硬门禁。

| 指标 | 取证基线（参考，不跨会话判定） | 目标 |
|------|------|------|
| per_on_step 中位 | 1.88 ms（复跑 1.97 ms） | 同批 A/B 比值报告；结构指标硬门禁（指纹/向量化） |
| fingerprint | 145.4~194.6 µs | ≤ 10 µs（同机同批） |
| 统计开销 | cProfile 20% of on_step | 迭代次数 0 + 计时 ≤20µs |
| 向量化 unpack/量化 | 参考实现 1x | ≤ 50%（同批 3 次中位） |

## 6. T007 优化后对照（2026-10-18，同进程同批，3 次中位）

| 指标 | 规模 | 优化路径 | 参考路径 | 比值 | 证据 |
|------|------|----------|----------|------|------|
| fingerprint | 256 ch | 0.2 µs | 53.5 µs | 0.004x | ar004_bench.py::structural_ab |
| fingerprint | 1024 ch | 0.1 µs | 138.4 µs | 0.001x | 同上 |
| unpack | 256 ch | 1.35 ms | 19.32 ms | 0.070x | `_unpack_reference` 同批 |
| unpack | 1024 ch | 3.47 ms | 79.67 ms | 0.044x | 同上 |
| quantize | 256 ch | 3.05 ms | 39.45 ms | 0.077x | `_quantize_mixed_precision_reference` 同批 |
| quantize | 1024 ch | 8.84 ms | 168.31 ms | 0.053x | 同上 |
| per_on_step_ms | 1024 ch/4 层/12 calls | 0.66-0.72 ms（复跑 0.59-0.77） | — | — | 绝对中位，不做跨会话比值 |

正确性：unpack/量化 golden <= 1e-6（含阈值 0.7/0.3、int4 非整除、pack_dim=1、全零通道）；掩码缺陷转 `mixed_full` 正常；层 2 输出 |直接计算差|=0.0。
测试与覆盖率：`732 passed, 1 skipped`，82.37%（当次 pytest --cov）。
说明：CPU 微基准；比值来自同进程参考实现对照，不设跨会话固定基线。
