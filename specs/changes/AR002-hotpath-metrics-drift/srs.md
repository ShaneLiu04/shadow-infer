# [AR002] 需求设计说明书

| 字段 | 内容 |
|------|------|
| AR 编号 | AR002 |
| AR 主题 | 端侧热路径向量化、指标真实化与实测漂移灵敏度（hotpath-metrics-drift） |
| 关联 SR | 无（源自 AR001 深度审计的框架级发现，开源项目质量与创新专项） |
| 日期 | 2026-10-17 |
| 状态 | Draft |

## 1. 背景与目标

AR001 审计与端到端实测（6-step CPU：基线 31.3ms、优化 1782.8ms，慢 ~57 倍）揭示三类深层问题：

1. **热路径纯 Python 化**：ShadowKV 压缩/量化按 token×head 双层 Python 循环逐元素处理（shadowkv_agent.py:724-753），每元素一次 `quantize_tensor` 调用；打包（packed_kv_cache.py:100-104）同样逐元素。这是优化路径远慢于基线的首要根因，且在端侧（CPU/NPU）场景下被放大。
2. **性能指标虚构**：Profiler 采集的 perf_data 为硬编码零值（orchestrator.py:1456-1463：latency_ms=0.0、gpu_utilization=0.0、tokens_per_sec=0.0），调度器因此吃到零延迟奖励（scheduler.py:70-72 附近），内存账目与进程 RSS 无对账，README 的 8.6% KV 节存无实测依据——整个反馈控制回路建立在假数据上。
3. **灵敏度模型失真**：QDrift 的 sensitivity_score 实为对 step 位置做 cosine 调度插值（qdrift_agent.py:296-311 附近），与实际激活漂移无关；drift_score 恒为 1.0 说明信号失效。步感知调度因此退化为固定查表。

本 AR 目标：把这三条「反馈回路 + 热路径」主干线修成真实、可量化、CPU 端可验证的创新点，使优化路径在 CPU fallback 下开销可控（≤1.5× 基线端到端），指标可信（与墙钟误差 <2%），调度信号有实测依据（扰动响应 Spearman ≥ 0.9）。

## 2. 需求范围

**In Scope（本 AR 要做的）：**

- M6 指标完整性：Profiler 真实计时/内存对账；调度器奖励基于真实延迟；benchmarks 移除硬编码 mock 断言
- M1 决策面张量化：`compress_kv`/重要性计算/打包路径的向量化重写（torch 原生批量操作），golden test 保证逐元素语义一致
- M5 决策与打包记忆化 + 内存记账修复（含 on_init 重置 bug，shadowkv_agent.py:106 附近）
- M3-lite 实测漂移灵敏度：以最小接线用实测扰动响应替代位置插值（CPU 可验证）
- 按 AGENTS.md 规则一完成 README 四检查点更新（版本 v3.4.0）；AR 验收后按规则二推送 Gitee

**Out of Scope（本 AR 不做的）：**

- M2 模型后端契约变更（需协调多 backend，留 AR003）
- M4 基于真实轨迹数据的调度学习（需数据积累，留 AR003/AR004）
- M7 FFN channel_mask 行为变更（改变优化语义，需单独评审，留 AR004）
- CUDA kernel 编译与真实 GPU 适配（环境无 CUDA）
- 对外公开 API 行为变更（`__all__` 导出、Agent on_init/on_step 签名、CLI 参数、配置格式均不变）

## 3. 功能需求

### 3.1 M6：性能指标真实化

**描述：** Profiler 采集真实墙钟延迟与内存，消除硬编码零值；下游奖励与报告消费真实数据。

**触发条件：** 每次 `run_optimized`/`run_baseline` 的 step 执行与 Profiler 采集。

**期望行为：**
- step 执行路径以 `time.perf_counter` 计量真实延迟并写入 perf_data.latency_ms（替换 orchestrator.py:1456-1463 的零值）；
- memory_mb 与进程 RSS（或 torch 内存统计，取可行者）对账，偏差超过阈值时记录告警日志；
- 调度器奖励函数消费真实 latency_ms（零延迟吃奖励的缺陷修复，scheduler.py:70-72 附近）；
- benchmarks 中硬编码的 ~3x mock 加速断言改为以实测数据断言（或删除该断言并注明原因）。

**异常处理：** 计时器异常时回退为 0.0 并记录 warning 级结构化日志（不中断推理）。

**验收标准：**
- Given 一次 6-step CPU 流水线, When 运行完成, Then 报告中 step_latency_ms > 0 且与外部墙钟测量误差 < 2%；调度器奖励随真实延迟变化。
- Given benchmarks 套件, When 运行, Then 无硬编码加速比断言残留（grep 零命中）。

### 3.2 M1：ShadowKV 决策面张量化

**描述：** 将 compress_kv 的逐 token×head Python 循环、重要性计算与 KV 打包重写为 torch 批量（向量化）操作。

**触发条件：** ShadowKVAgent.on_step 的压缩与打包路径。

**期望行为：**
- 量化按 precision_map 分组为至多 3 个批次（fp16/int8/int4），组内整块张量量化；
- reuse_mask 以布尔张量索引一次性应用（被复用槽位直接拷贝 kv_previous 切片）；
- packed_kv_cache 打包路径同步向量化；
- 决策平面（decision_plane）输出（precision_map/reuse_mask/eviction）语义逐元素不变。

**异常处理：** 向量化路径抛错时记录日志并回退到原循环实现（保留为 reference 路径，供 golden test 与兜底）。

**验收标准：**
- Given 随机输入与随机决策, When 分别运行向量化与 reference 路径, Then precision_map/reuse_mask/压缩输出逐元素一致（golden test，容差 1e-6）。
- Given 6-step CPU 流水线, When 对比优化路径 step 耗时, Then 较 AR002 前的 ShadowKV 压缩耗时下降 ≥ 5 倍（cProfile 证据落档）。
- Given 全量测试, When 运行, Then 581+ 测试全部通过。

### 3.3 M5：决策记忆化与内存记账修复

**描述：** 同一 step 内重复调用的决策/打包结果记忆化复用；修复 on_init 状态重置与内存记账失真。

**触发条件：** 同一 step_id 的重复 on_step/决策查询。

**期望行为：**
- 决策缓存以 (step_id, 输入指纹) 为键，step 边界自动失效；
- on_init 正确重置全部缓存与累计状态（修复 shadowkv_agent.py:106 附近的重置缺失）；
- memory_mb 统计口径修正为「真实节省字节数 = 未压缩尺寸 − 压缩后尺寸」，禁止以张量元素数近似。

**异常处理：** 缓存指纹计算失败时直接旁路缓存（不缓存也不报错）。

**验收标准：**
- Given 同一 step 两次相同输入调用, When 对比, Then 第二次压缩耗时下降 ≥ 90% 且输出一致。
- Given 连续两次 initialize, When 检查内部状态, Then 无跨 run 状态泄漏（新增回归测试）。

### 3.4 M3-lite：实测漂移灵敏度

**描述：** QDrift 灵敏度以「对输入施加微小扰动的输出响应幅度」实测计算，替换纯位置插值；位置先验仅作正则项保留（权重可配置）。

**触发条件：** QDriftAgent.on_step 的 sensitivity_score 计算。

**期望行为：**
- 灵敏度 = 归一化的实测扰动响应（基于现有 query_current/activation_current 与其 previous 的漂移统计），位置先验以可配置权重 α 混入（默认 α=0.3）；
- drift_score 修复恒为 1.0 的归一化缺陷，恢复动态范围；
- CPU 上完整可验证（不依赖 CUDA）。

**异常处理：** 输入缺失（无 previous 张量）时回退为纯位置先验并记录日志。

**验收标准：**
- Given 构造的扰动响应序列与位置无关的测试输入, When 计算灵敏度, Then 灵敏度与实测扰动响应的 Spearman 相关 ≥ 0.9（新增统计测试）。
- Given 现有 qdrift 测试, When 运行, Then 全部通过或按新语义更新断言（更新须在测试注释中说明依据）。

### 3.5 README 四检查点更新（AGENTS.md 规则一）

**描述：** 本 AR 属性能改进创新，完成后必须更新 README：特性表（决策面张量化/真实指标/实测灵敏度条目修订）、版本历史（v3.4.0，同步 pyproject.toml 与 `__version__`）、性能基准（以本 AR 实测数据更新量化表格，禁止沿用旧数字）、结构与口径（锚点/无 emoji/失效引用零残留）。

**验收标准：**
- Given 本 AR 完成, When 检查 README, Then 四检查点全部满足且数据与当次实跑一致。

### 3.6 AR 完成后 Gitee 推送（AGENTS.md 规则二）

**描述：** Review 与 ST 验收并归档后，按 Runbook 推送 Gitee（小批量多次上传；已知 git 2.56 排除缺陷，线性提交正常逐个推送，双根/无祖先情形用 §3.5.1 微包法）。

**验收标准：**
- Given AR 通过验收, When 按 Runbook 推送, Then 三核验通过且远端无 AGENTS.md。

## 4. 非功能需求

| 类型 | 指标 | 要求 |
|------|------|------|
| 性能 | CPU fallback 端到端 | run_optimized 总耗时 ≤ 1.5 × run_baseline 总耗时（同机同参） |
| 质量 | 五件套门禁 | black/isort/flake8/mypy 零告警；pytest 全通过；覆盖率不低于 80.84% 基线 |
| 精度 | golden test | 向量化输出与 reference 逐元素一致（容差 1e-6） |
| 统计 | 灵敏度有效性 | Spearman ≥ 0.9（扰动响应测试） |
| 兼容性 | API | 对外接口行为不变（签名/返回结构/配置格式） |
| 安全 | token | 永不落盘（沿用 AGENTS.md 铁律） |

## 5. 约束与假设

**约束：**
- 全部改进在 CPU 环境可完整验证（本机无 CUDA；torch 2.5.1+cu121 走 CPU fallback）；
- 不改变对外 API 行为与配置格式（与 AR001 一致）；
- 推送遵守华为网关尺寸门：小批量多次上传，单批压缩后 ≤ 40KB；
- 版本号升 v3.4.0。

**假设：**
- 现有 581 测试中依赖旧语义（零值指标/位置插值）的断言允许按新真实语义更新，但须逐个注明依据；
- cProfile 计时抖动不影响 ≥5 倍这类量级结论。

## 6. 术语说明

| 术语 | 定义 |
|------|------|
| 热路径 | run_optimized 中每 step 必经的 ShadowKV 压缩/打包与决策计算代码 |
| golden test | 以旧实现为 reference，断言新实现输出逐元素一致的等价性测试 |
| 记忆化 | 以输入指纹为键缓存计算结果，同输入直接命中复用 |
| 位置插值 | 仅依赖 step 位置（cosine 调度）而非实测信号的灵敏度近似 |
| Spearman 相关 | 秩相关系数，用于验证灵敏度排序与实测扰动响应排序的一致性 |
