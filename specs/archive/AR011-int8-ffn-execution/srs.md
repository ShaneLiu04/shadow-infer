# [AR011] 需求设计说明书：可执行 INT8 FFN 低比特运行路径

| 字段 | 内容 |
|------|------|
| AR 编号 | AR011 |
| AR 主题 | Executable INT8 FFN：真实低比特 GEMM + 动态激活量化 |
| 关联 SR | 无（架构审计驱动） |
| 日期 | 2026-10-20 |
| 状态 | Approved |

## 1. 背景与目标

FlashInfer 的核心价值不是“把元数据打包”，而是**让低比特/分页元数据直接驱动真实执行**：
quantized weights、per-group scales、plan/run 分离最终都落到可执行的 kernel。
PyramidInfer 同样说明：只有真正减少被计算的 token/通道，收益才成立。

当前 ShadowInfer 的 FFN 优化存在结构性缺口（真实代码审计）：

- `FFNOptimizerAgent` 生成 `PackedFFNWeight`，但 `dequantize_for_matmul()` 缓存完整 FP32 权重，
  每次前向仍调用 dense `F.linear`；稳态内存是 **packed + dense（比原始 FP32 更大）**，计算无收益。
- packed 只在 Agent 的记账路径中生效，模型 `SimpleDiffusionLLM` 的 `up_proj/down_proj` 从未使用低比特权重。
- 编排器读取 `ffn_optimizer` 配置段，而所有发布配置使用 `ffn:`，导致 `mixed_precision` / `sparse_update`
  等开关实际被忽略。

**目标**：实现一条**真实可执行**的 INT8 FFN 路径：

1. 权重按输出通道对称 INT8 量化（per-channel scale），激活按 token 动态 INT8 量化（per-token scale）；
2. 使用 `torch._int_mm` 执行真实 INT8 GEMM，再按 `scale_x × scale_w` 反量化并加 bias；
3. 直接替换模型 FFN 的 `up_proj` / `down_proj`，释放 FP32 权重（可选保留参考副本）；
4. 在 7B FFN 形状（4096×11008）上相对 FP32 达到 >=3x 单层加速、权重内存约 4x 降低；
5. 提供 FP32 golden 对照、精度证据、回退路径与服务器复现协议。

> 本机不运行 7B 模型前向；大模型端到端精度/延迟按 `server-experiment.md` 在服务器回填。

## 2. 需求范围

**In Scope：**
- 新增 `shadowinfer/ffn_optimizer/int8_linear.py`：`INT8Linear`（真实 INT8 GEMM、per-channel/per-token scale、
  bias、内存记账、不可用时确定性回退）。
- `SimpleDiffusionLLM`：`enable_int8_ffn()` / `restore_fp32_ffn()`，替换 FFN 线性层并释放 FP32 权重。
- `PyTorchModelBackend`：`enable_int8_ffn()` 接口与配置读取；返回量化统计。
- `HuggingFaceModelBackend`：按常见 FFN 命名（`mlp.up_proj/down_proj/gate_proj`、`mlp.c_fc/c_proj`）
  替换线性层；支持保留/恢复 FP32 参考，无匹配时返回不支持。
- Orchestrator：读取 `ffn:` 与 `ffn_optimizer:` 两套配置键；启用 INT8 执行时不再运行重复的 FFN Agent 记账路径；
  将真实执行统计传给 Profiler。
- 测试：数值 golden、形状/边界、内存、回退、端到端等价、配置接线。
- 基准：7B FFN 形状微基准、小模型端到端与 logits 误差、3 次重复、原始数据落档。
- `configs/experiments/s6_int8_ffn_gpu.yaml` 与服务器复现协议。

**Out of Scope：**
- 不在本机运行 7B/预训练模型大实验。
- 不实现 CUDA 自定义 kernel（使用 PyTorch `_int_mm`，不可用时回退）。
- 不改变 KV 压缩/分配语义；不实现 INT4 FFN 执行。
- 不承诺小尺寸模型（K<256）一定加速；如实报告交叉点。

## 3. 功能需求

### 3.1 FR11.1 真实 INT8 线性层

**描述：** `INT8Linear` 以对称 per-output-channel INT8 权重和 per-token 动态 INT8 激活执行矩阵乘。

**触发条件：** 输入 `x[..., K]`，模块持有 `int8[K, N]` 权重（转置存储）、`float32[N]` 权重 scale、可选 bias。

**期望行为：**
- 计算 `s_x = max|x_row| / 127`，`x_q = round(x / s_x).clamp(-127,127)`；
- `y_int32 = _int_mm(x_q, W_q)`；`y = y_int32 * (s_x * s_w) + bias`；
- 输出 dtype 与输入一致；空 scale 用 epsilon 防除零；
- 提供 `memory_bytes()`（int8 权重 + scale + bias）与 `dense_equivalent_bytes()`。

**异常处理：** `_int_mm` 不可用或形状不支持时，回退到“反量化权重 + F.linear”（仍为 INT8 存储），
记录一次 warning，结果与 INT8 语义一致；非法权重维度抛 `ValueError`。

**验收标准：**
- Given 随机 FP32 权重/输入，When 执行 `INT8Linear`，Then 输出与 FP32 golden 的相对误差 <= 3%
  且无 NaN/Inf。
- Given `[B,S,K]` 输入，When 前向，Then 输出 `[B,S,N]`，与 2D 展平结果一致。
- Given `_int_mm` 被模拟不可用（非数学 mock，注入异常），Then 回退结果与真实 INT8 路径一致。

### 3.2 FR11.2 模型级替换与真实执行

**描述：** `SimpleDiffusionLLM` 可将每个 block 的 `up_proj/down_proj` 替换为 `INT8Linear`。

**触发条件：** `enable_int8_ffn(keep_fp32_reference=False)`。

**期望行为：**
- 每层独立量化；返回 `{"layers": L, "weight_bytes": ..., "dense_bytes": ..., "fallback": bool}`；
- 默认释放原 FP32 线性层（不再持有），`keep_fp32_reference=True` 时保留供 golden 对照；
- `restore_fp32_ffn()` 在保留参考时恢复原模块，否则抛清晰错误；
- 重复调用幂等。

**验收标准：**
- Given 已加载模型，When `enable_int8_ffn()`，Then `isinstance(block.up_proj, INT8Linear)` 且 FP32 权重不再被引用
  （`keep_fp32_reference=False`）。
- Given 保留参考，When 同一输入分别走 FP32/INT8，Then logits 相对误差 <= 3%，top-1 一致率 >= 95%（真实小模型、3 seed）。
- Given 未启用，When 正常前向，Then 行为与 v3.12 完全一致（逐位）。

### 3.3 FR11.3 后端与编排接线

**描述：** PyTorch 后端与 Orchestrator 根据配置启用/停用 INT8 FFN，并避免重复记账。

**触发条件：** 配置 `ffn.int8_execution_enabled: true`（或顶层 `int8_ffn_enabled`）。

**期望行为：**
- `ModelBackend.enable_int8_ffn()` 默认返回“不支持”，PyTorch 实现真实替换；
- Orchestrator 在 `initialize()` 中启用，并记录 `int8_ffn` 统计；
- 启用后 `_run_ffn` 返回真实执行统计（不运行重复的 `FFNOptimizerAgent` 打包路径）；
- `ffn:` 与 `ffn_optimizer:` 两套键均可被读取，未知键不影响启动。

**验收标准：**
- Given `ffn.int8_execution_enabled=true`，When 运行一步，Then 模型 FFN 为 INT8 且 `_run_ffn` 不调用 Agent 打包。
- Given 配置为 false，When 运行一步，Then 走 v3.12 原有 FFN Agent 路径。
- Given 配置只有 `ffn:` 段，Then `mixed_precision` 等键能被 Agent 读取（修复键名不一致）。

### 3.4 FR11.4 性能与内存证据

**描述：** 在真实张量上量化 INT8 执行收益，并明确适用边界。

**验收标准：**
- 7B FFN 形状（M∈{16,64,128}，K=4096，N=11008）FP32 vs INT8，3 次中位：
  INT8 GEMM 加速 >=3x（实测 4.80-6.74x，复跑区间 4.8-6.7x）；M=1 与反向投影（11008×4096）为 1.25-1.69x，
  按实际区间报告；权重内存约 4x 降低。
- 中等规模真实模型（4 层/hidden=1024/intermediate=4096/seq=256）端到端实测加速 >=1.2x
  （本机 1.47x，复跑 1.37-1.47x），logits 相对误差 <=3%。
- 精度：真实 `SimpleDiffusionLLM`（4 层/8 头/dim64/seq128，3 seed）logits 相对误差与 top-1 一致率落档。

### 3.5 FR11.5 可复现与回退

**描述：** 基准脚本、配置档位、服务器协议与回退路径齐备。

**验收标准：**
- `specs/archive/AR011-int8-ffn-execution/ar011_bench.py` 可一键复跑并断言 golden/内存；
- `configs/experiments/s6_int8_ffn_gpu.yaml` 与 `server-experiment.md` 提供 7B/GPU 回填协议；
- 全量门禁（black/isort/flake8/mypy/pytest）全绿，覆盖率不下降。

## 4. 非功能需求

| 类型 | 指标 | 要求 |
|------|------|------|
| 性能 | 7B FFN（M>=16）单层加速 | >=3x（CPU 实测 4.80-6.74x） |
| 性能 | 中等模型端到端 | >=1.2x（CPU 实测 1.37-1.47x） |
| 内存 | FFN 权重 | 约 4x 降低（FP32→INT8+scale） |
| 精度 | logits 相对误差 | <=3%（小模型 3 seed） |
| 可靠性 | 回退 | `_int_mm` 不可用时确定性回退，不崩溃 |
| 兼容 | 默认行为 | 默认关闭，v3.12 路径逐位不变 |
| 门禁 | 全量 | 五件套全绿，覆盖率不低于 82.99% |

## 5. 约束与假设

**约束：**
- 不使用 mock；所有测试使用真实张量/真实模型。
- 不改变 KV 压缩与打包语义。
- `torch._int_mm` 为实验性 API；必须提供回退，且回退语义仍为 INT8 权重。
- 小尺寸 CPU GEMM 可能变慢，结论必须按实测区间报告。

**假设：**
- 目标部署以 7B 级 FFN 形状为主，INT8 GEMM 在该量级有明确收益（本机复跑 4.8-6.7x）。
- 随机初始化小模型可验证数值误差，但不能替代预训练模型质量评估。

## 6. 术语说明

| 术语 | 定义 |
|------|------|
| per-channel scale | 每个输出通道一个权重量化 scale |
| per-token scale | 每个 token（行）一个激活动态 scale |
| INT8 GEMM | `torch._int_mm` 的 int8×int8→int32 矩阵乘 |
| 可执行路径 | 低比特权重直接参与模型前向，而非仅记账/打包 |
