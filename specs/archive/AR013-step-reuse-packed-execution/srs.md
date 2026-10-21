# [AR013] 需求设计说明书：跨步复用与紧凑 KV 闭环执行

| 字段 | 内容 |
|------|------|
| AR 编号 | AR013 |
| AR 主题 | step-reuse-and-packed-execution |
| 关联 SR | 无（系统级执行闭环） |
| 日期 | 2026-10-20 |
| 状态 | Approved |

## 1. 背景与目标

AR007-AR012 已把 ShadowKV 的决策、压缩、打包与读写做到可复现、可量化，但真实推理仍存在两个
架构缺口：

1. **执行回路未闭合**：`PackedKVCache` 只被 agent 记账，`PyTorchModelBackend.forward_step` 始终
   使用自己的 dense KV；压缩后的 KV 从未进入 attention。KV 字节下降不等于推理驻留下降。
2. **Diffusion 步骤语义错误**：每个 denoising step 都把上一步 K/V 当 past 再 concat，序列固定时
   cache 随 step 线性增长（第 24 步实际 context ≈ 24×S），既慢又不符合 dLLM 语义。
3. **跨步冗余计算**：相邻 denoising step 的 hidden 漂移往往很小，但仍对全部层执行
   Q/K/V 投影、attention 与 FFN，缺少可执行的复用路径。

参考 SOTA：
- **FlashInfer**：plan/run 分离、紧凑分页 KV、批量 gather；启发“存储表示与执行表示解耦，执行时
  只做一次紧凑读取”。
- **DeepCache / Δ-DiT**：扩散去噪跨时间步复用高层特征；启发“漂移门控的块级复用”。
- **PyramidInfer / H2O**：按重要性保留 KV；已有 AR009/AR012 实现，本 AR 复用其打包结构。

**目标**：让压缩 KV 真正参与推理，并在真实 denoising 轨迹上以可量化的误差换到真实速度；
同时修正 diffusion KV 生命周期，形成“存储紧凑、执行可控、结果可复现”的闭环。

## 2. 需求范围

**In Scope：**
- `shadowinfer/models/simple_diffusion_llm.py`：可选的跨步块复用（drift-gated），默认关闭。
- 新增 `shadowinfer/execution/step_reuse.py`：复用状态、漂移度量、统计。
- `shadowinfer/core/backends/pytorch_backend.py`：`kv_lifecycle`、`kv_execution`（dense/packed）、
  复用状态持久化与统计。
- `shadowinfer/orchestrator.py`：执行配置下发、真实后端结果透传、每步执行统计。
- 真实模型基准脚本、s8 实验配置、服务器复现协议。
- 测试（真实张量 + 真实 forward）。

**Out of Scope：**
- 不修改 HuggingFace/通用 causal LM 的 block 结构；HF 后端保持现有 dense 路径。
- 不新增 CUDA kernel；packed 执行复用现有 `PackedKVCache` 张量化读路径。
- 不改变 ShadowKV 的精度/复用/驱逐决策语义。
- 不在本机跑 7B 大实验；服务器按协议复现。
- 不宣称 CPU 上 packed attention 比 dense 更快（原型显示相反）；packed 的收益是驻留字节，
  速度收益来自跨步复用。

## 3. 功能需求

### 3.1 FR13.1 Diffusion KV 生命周期

**描述：** 后端新增 `kv_lifecycle`（`"append"` | `"replace"`）。`replace` 用于固定长度
denoising：每步 attention 只使用当前步 K/V，不把上一步 concat 成 past。

**触发条件：** `PyTorchModelBackend` 配置 `kv_lifecycle="replace"`。

**期望行为：** `forward_step` 不向模型传 past KV；返回的每层 K/V 长度 == 当前输入长度。

**异常处理：** 非法值抛 `ValueError`；未配置时默认 `append`，与 v3.14.0 逐位一致。

**验收标准：**
- Given `replace` + 3 个连续 step、固定 S，When forward，Then 每步 `k_tensor.shape[2] == S`。
- Given `append`，When 连续 2 步 S=4，Then 第 2 步长度 == 8（旧行为）。

### 3.2 FR13.2 漂移门控块复用

**描述：** 新增 `StepReuseState`；模型逐 block 缓存 `(input, output, k, v)`。当
`||h - h_cached|| / ||h_cached|| <= threshold` 且已过 warmup 时，跳过该 block 的
Q/K/V/O、attention 与 FFN，输出 `cached_out + (h - h_cached)`；否则完整计算并刷新缓存。

**触发条件：** `step_reuse_enabled=True` 且 `kv_lifecycle="replace"`；首次 step 与 warmup 不
复用。

**期望行为：** 返回 `reuse_stats: {reused_layers, computed_layers, reused_ratio, drifts}`；
attention scores 复用上一步最后可用的矩阵（避免缓存 O(S²) 每层）。

**异常处理：** 形状不一致时清空缓存并完整计算，记录 reset 原因；`threshold<0` 抛 `ValueError`。

**验收标准：**
- Given 关闭复用，When 相同输入，Then logits 与 v3.14.0 逐位一致。
- Given 关闭复用，When 两次 `forward_step`，Then 不缓存且 `reused_layers=0`。
- Given 受控小漂移轨迹，When threshold=0.1，Then 至少跳过 10% block 且 logits 相对 L2 <= 0.02。
- Given warmup=2，When step 0/1，Then 不复用；step 2 起才可复用。

### 3.3 FR13.3 紧凑 KV 闭环执行

**描述：** 新增 `kv_execution`（`"dense"` | `"packed"`）。`packed` 模式下每层 K/V 以
`PackedKVCache` 为唯一驻留表示；下一步 attention 从 packed 反量化获取 K/V。

**触发条件：** `kv_execution="packed"`。

**期望行为：**
- 每步结束按 `shadowkv_mode` 选择精度（conservative=fp32、balanced=fp16、aggressive=int8），
  或 `packed_precision` 显式覆盖；调用真实 `PackedKVCache.pack`。
- `KVCacheEntry.k_tensor/v_tensor is None`，`packed_kv` 非空。
- 返回 `kv_execution_stats: {resident_bytes, data_plane_bytes, metadata_bytes,
  dense_bytes, compression_ratio, data_plane_ratio, precision}`。
- 下一步 `raw_cache` 通过 `packed.unpack()` 构建，logits 与 dense 路径在容差内一致。

**异常处理：** 不支持的精度抛 `ValueError`；pack 失败时记录告警并回退 dense（统计标记
`fallback=True`），不得静默。

**验收标准（Review 修正 2026-10-20：原 `resident_bytes < dense_bytes` 隐含 FP32 dense
基准，与 FR13.6 的 FP16 基准矛盾；`dense_bytes` 统一为 FP16 dense 基准，驻留改用双口径
记账——`data_plane_bytes` 仅计张量字节（含 scale，GPU flat-buffer 部署承载），
`resident_bytes` 另含 48B/token-head 的 Python 簿记估计（CPU 实现开销）。同精度 packed
（如 fp16）相对 FP16 dense 无压缩属预期，只验证数据平面不劣于 dense）：**
- Given `balanced` + packed，When 1 步，Then 每层 entry 无 dense 张量，
  `data_plane_bytes <= dense_bytes` 且 `metadata_bytes > 0`（记账透明）。
- Given packed 与 dense 同 precision，When 同输入，Then logits `allclose(atol=2e-3)`。
- Given `aggressive` int8，When append 连续 2 步，Then 第二步从 packed 读取 past 且无
  dense 张量；head_dim>=64 时 `data_plane_ratio <= 0.6`（小 head_dim 下 per-row scale
  与元数据占比上升，驻留目标是规模相关指标）。

### 3.4 FR13.4 复用与紧凑执行组合

**描述：** 同一 step 中，被复用的层直接保留上一步 packed K/V，不重复 pack；未复用层重新
计算并 pack。统计区分 `reused_layers` / `packed_layers` / `skipped_repack`。

**验收标准：**
- Given reuse 跳过 L 层，When packed 模式，Then 这 L 层的 `packed_kv` 对象保持同一引用，
  且总 pack 次数 == 计算层数。
- Given 全部层复用，When packed 模式，Then 本步无新 pack，resident bytes 不变。

### 3.5 FR13.5 后端与编排接线

**描述：** 后端新增 `configure_execution(...)` 与 `reset_execution_state()`；Orchestrator 从
`config["execution"]`（或 `shadowkv.execution` 兼容键）下发，并在每次 run 开始时 reset。

**验收标准：**
- Given 配置 `execution.step_reuse_enabled=true`，When `run_optimized`，Then 后端启用且每步
  profiler 可见 `execution` 统计；未配置时行为不变。
- 后端异常不得静默：告警 + `fallback` 标记。
- `run_optimized` 两次调用不共享复用状态（第二次 step 0 不复用）。

### 3.6 FR13.6 可复现实验与服务器协议

**描述：** 新增真实张量基准 `ar013_bench.py`：受控 denoising 轨迹（sigma 线性下降），
多规模（S=256/512/1024）、3 seed、3 repeats，比较 full baseline（replace + dense）与
step-reuse、packed、组合三者；断言速度与误差门槛。

**验收标准：**
- 脚本真实运行，无 mock；输出每档 median/min-max。
- 至少 2 个规模点满足：reuse 加速 >=1.15x 且 logits relL2 <= 0.02。
- packed 数据平面驻留（张量字节含 scale）<= dense fp16 的 0.6x（int8，head_dim>=64），
  int4 需 group_size>=32 方可达 ~0.35x（g=16 时约 0.375x，随 group_size 变化）；
  全口径 resident（含 48B/token-head Python 元数据估计）另行报告；logits relL2 <= 0.02。
  质量与驻留断言覆盖全部 seed，速度断言以 seed0 中位为准。
- 服务器配置 `s8_step_reuse_gpu.yaml` 与 `server-experiment.md` 包含 7B 维度、命令、
  baseline/优化对照、质量/延迟/显存记录模板。

## 4. 非功能需求

| 类型 | 指标 | 要求 |
|------|------|------|
| 性能 | 受控轨迹（S=256/512） | reuse >=1.15x，组合 >=1.10x |
| 质量 | logits 相对 L2 | <=2%（reuse），<=2%（packed） |
| 内存 | packed resident | int8 data-plane <=0.6x fp16 dense（全口径如实另报），int4 <=0.35x 需 group_size>=32 |
| 兼容 | 默认关闭 | logits 逐位一致 |
| 可靠性 | 状态隔离 | 每次 run reset；异常显式 fallback |
| 门禁 | 全量 | 五件套全绿，覆盖率不低于 83.23% |

## 5. 约束与假设

- 跨步复用是近似执行；必须同时报告误差与跳过率，不得只报速度。
- 受控轨迹不能替代真实 7B；服务器结果按协议回填，README 分开标注。
- packed attention 在 CPU 上比 dense 慢（原型 0.03-0.16x），本 AR 不以 CPU packed 加速为结论。
- HF 后端不启用 block 复用（结构不可控），只在协议中说明。
- 所有新增开关默认关闭，旧行为可逐位复现。

## 6. 术语

| 术语 | 定义 |
|------|------|
| 块复用 | 跳过整个 transformer block，复用缓存输出并加输入增量 |
| 漂移 | `||h - h_cached|| / ||h_cached||`，逐 block 计算 |
| replace 生命周期 | 固定长度 denoising 中每步 K/V 替换而非追加 |
| packed 执行 | attention 的 K/V 从 `PackedKVCache` 反量化获取 |
| 受控轨迹 | `x_t = x0 + sigma_t * noise`，sigma 单调下降的真实张量序列 |
