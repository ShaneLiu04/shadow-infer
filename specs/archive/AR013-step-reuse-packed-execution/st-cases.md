# [AR013] ST 验收测试用例

| 字段 | 内容 |
|------|------|
| AR 编号 | AR013 |
| 关联 srs.md | ./srs.md |
| 关联 design.md | ./design.md |
| 生成日期 | 2026-10-20 |

## 测试用例列表

### ST-001：replace 生命周期固定上下文长度

**关联需求：** srs.md §3.1 FR13.1 Diffusion KV 生命周期
**测试类型：** 正常路径
**优先级：** High

**前置条件：**
- Given `kv_lifecycle="replace"`，真实 `SimpleDiffusionLLM` 后端

**测试步骤：**
1. 配置 replace，连续执行 3 个 denoising step（同 S 输入）
2. 检查每步每层 KV entry（dense 或 packed）的序列长度

**期望结果：**
- Then 每步每层长度恒为 S（无跨步 concat 增长）

**实际结果：** 自动化 `tests/test_step_reuse_execution.py::TestKVLifecycle::test_replace_keeps_current_step_length` 通过（3 步，每层 seq_len==8）

**状态：** PASS

---

### ST-002：append 保留旧增长语义

**关联需求：** srs.md §3.1 FR13.1
**测试类型：** 边界条件（兼容性）
**优先级：** High

**前置条件：**
- Given `kv_lifecycle="append"`（默认）

**测试步骤：**
1. 连续执行 2 个 step
2. 检查第 1/2 步各层 k_tensor 序列长度

**期望结果：**
- Then 第 1 步 S、第 2 步 2S（v3.14.0 既有行为逐位保留）

**实际结果：** 自动化 `::TestKVLifecycle::test_append_preserves_legacy_growth` 通过（8→16）

**状态：** PASS

---

### ST-003：非法 lifecycle 拒绝

**关联需求：** srs.md §3.1 FR13.1（异常处理）
**测试类型：** 异常处理
**优先级：** Medium

**前置条件：**
- Given 任意后端实例

**测试步骤：**
1. 调用 `configure_execution(kv_lifecycle="grow")`

**期望结果：**
- Then 抛出 `ValueError`

**实际结果：** 自动化 `::TestKVLifecycle::test_invalid_lifecycle_raises` 通过（pytest.raises ValueError）

**状态：** PASS

---

### ST-004：默认关闭时逐位一致

**关联需求：** srs.md §3.2 FR13.2（验收标准 1/2）、§4 兼容性
**测试类型：** 边界条件（兼容性）
**优先级：** High

**前置条件：**
- Given 复用关闭（默认配置）

**测试步骤：**
1. 同一输入连续两次 `forward_step`
2. 对比两次 logits 与执行统计

**期望结果：**
- Then logits 逐位一致；`reused_layers=0`；无复用状态残留

**实际结果：** 自动化 `::TestReuseDisabled::test_two_calls_identical_and_no_state` 通过（torch.equal，reused_layers 为空，`_reuse_state` 为 None）

**状态：** PASS

---

### ST-005：受控小漂移轨迹复用（SRS 口径）

**关联需求：** srs.md §3.2 FR13.2（验收标准 3）
**测试类型：** 正常路径
**优先级：** High

**前置条件：**
- Given `step_reuse_enabled=True`、`threshold=0.1`、warmup=0，受控轨迹 `x0*3 + sigma*noise`（sigma 0.05/0.04/0.03）

**测试步骤：**
1. 复用后端与无复用参考后端分别执行 step 0/1
2. 检查复用层集合、复用率、drifts 与 logits 相对 L2

**期望结果：**
- Then 复用率 >= 10% block 且 logits relL2 <= 0.02；drifts 暴露且 <= threshold

**实际结果：** 自动化 `::TestReuseMechanics::test_controlled_trajectory_reuses_all_layers` 通过（2/2 层复用，ratio 1.0 >= 0.1，drifts 全层 <= 0.1，relL2 0.0026 <= 0.02）

**状态：** PASS

---

### ST-006：warmup 门控

**关联需求：** srs.md §3.2 FR13.2（验收标准 4）
**测试类型：** 边界条件
**优先级：** Medium

**前置条件：**
- Given `warmup_steps=2`

**测试步骤：**
1. 执行 step 0/1，检查 reused_layers
2. （step 2 起方可用）

**期望结果：**
- Then step 0/1 均不复用

**实际结果：** 自动化 `::TestReuseMechanics::test_warmup_blocks_reuse` 通过（step 0/1 reused_layers 为空）

**状态：** PASS

---

### ST-007：复用输出公式逐位一致

**关联需求：** srs.md §3.2 FR13.2（机制定义）
**测试类型：** 正常路径
**优先级：** High

**前置条件：**
- Given 手工设置 `LayerReuseEntry.hidden_in/hidden_out`（常量张量）

**测试步骤：**
1. 以 `x1 = x0 + 0.001` 执行 forward_step
2. 手工重算 `cached_out + (hidden_in(x1) - cached_in)` 经 out_norm/output_proj 的期望 logits

**期望结果：**
- Then 实际 logits 与期望 `allclose(atol=1e-6)`

**实际结果：** 自动化 `::TestReuseCorrectness::test_manual_cached_delta_formula` 通过（allclose 1e-6）

**状态：** PASS

---

### ST-008：首步形状不同清空缓存并记录原因

**关联需求：** srs.md §3.2 FR13.2（异常处理：清缓存重算，记录 reset 原因）
**测试类型：** 异常处理
**优先级：** High

**前置条件：**
- Given 缓存中存在与当前模型 num_heads/head_dim 不匹配的 KV entry

**测试步骤：**
1. 注入坏 shape entry（head_dim+8）
2. 执行 forward_step，检查 reset_reason、缓存重建与输出

**期望结果：**
- Then 全部缓存清空、`reset_reason="shape_mismatch"`、输出正确重建（有限值）；匹配时 reset_reason 为 None

**实际结果：** 自动化 `::TestShapeMismatchReset` 2 用例通过（正/反向）

**状态：** PASS

---

### ST-009：packed 模式结构（balanced）

**关联需求：** srs.md §3.3 FR13.3（验收标准 1，Review 修正后口径）
**测试类型：** 正常路径
**优先级：** High

**前置条件：**
- Given `kv_execution="packed"`、`packed_precision="fp16"`

**测试步骤：**
1. 执行 2 个 step
2. 检查 entry 无 dense 张量、记账关系与精度

**期望结果：**
- Then `k_tensor/v_tensor is None`、`packed_kv` 非空；`data_plane_bytes <= dense_bytes` 且 `metadata_bytes > 0`（FP16 基准双口径记账透明）

**实际结果：** 自动化 `::TestPackedExecution::test_packed_entries_and_quality` 通过（无 dense；resident==data_plane+metadata；metadata>0）

**状态：** PASS

---

### ST-010：packed 与 dense 同精度 logits 一致

**关联需求：** srs.md §3.3 FR13.3（验收标准 2）
**测试类型：** 正常路径
**优先级：** High

**前置条件：**
- Given packed(fp16) 与 dense 后端同 seed 同输入

**测试步骤：**
1. 两后端分别执行相同 2 步
2. 对比 logits

**期望结果：**
- Then `allclose(atol=2e-3)`

**实际结果：** 自动化 `::TestPackedExecution::test_packed_entries_and_quality` 通过（allclose 2e-3）；`::test_packed_next_step_has_no_dense_tensors` 通过（第 2 步 packed_layers==N）

**状态：** PASS

---

### ST-011：int8 连续 2 步从 packed 读取

**关联需求：** srs.md §3.3 FR13.3（验收标准 3，Review 修正：append 连续 2 步）
**测试类型：** 正常路径
**优先级：** High

**前置条件：**
- Given `aggressive` + `packed_precision="int8"`、append 模式、head_dim=64

**测试步骤：**
1. 连续 2 步（S=4/5），第 2 步 past 须从 packed 反量化读取
2. 检查每步输出 vs dense 后端、precision、data_plane_ratio、packed seq

**期望结果：**
- Then 每步 relL2 <= 0.02；`precision=="int8"`；`data_plane_ratio <= 0.6`；`packed_kv.seq_len == 4+5`

**实际结果：** 自动化 `::TestAggressivePacked::test_int8_two_steps_resident_and_quality` 通过（relL2 0.0075/0.0076，ratio 0.531，seq 9）

**状态：** PASS

---

### ST-012：pack 失败显式 fallback

**关联需求：** srs.md §3.3 FR13.3（异常处理：不得静默）
**测试类型：** 异常处理
**优先级：** High

**前置条件：**
- Given `PackedKVCache.pack` 抛出异常（monkeypatch 注入）

**测试步骤：**
1. 执行 forward_step，检查 fallback 标记、entry 类型、输出
2. 解除注入后再执行一步，检查恢复

**期望结果：**
- Then `fallback=True`、entry 回退 dense、输出与 dense 后端逐位一致、`_packed_cache` 清空；恢复后 packed 正常且 `fallback=False`

**实际结果：** 自动化 `::TestPackedFallback::test_pack_failure_falls_back_to_dense` 通过（torch.equal 输出一致；恢复步 packed_layers==N）

**状态：** PASS

---

### ST-013：非法执行配置拒绝

**关联需求：** srs.md §3.3 FR13.3 / §3.2 FR13.2（配置校验）
**测试类型：** 异常处理
**优先级：** Medium

**前置条件：**
- Given 任意后端实例

**测试步骤：**
1. 分别传入负 threshold、非法 `kv_execution`、非法 `packed_precision`

**期望结果：**
- Then 均抛 `ValueError`

**实际结果：** 自动化 `::TestInvalidExecutionConfig` 3 用例通过（pytest.raises ValueError）

**状态：** PASS

---

### ST-014：组合模式跳过重 pack

**关联需求：** srs.md §3.4 FR13.4
**测试类型：** 正常路径
**优先级：** High

**前置条件：**
- Given packed + reuse 同开，受控小漂移使全部层复用

**测试步骤：**
1. 执行 2 步，对比两步 packed 对象同一性与驻留
2. 检查 `packed_layers`/`skipped_repack` 统计

**期望结果：**
- Then 复用层保留同一 packed 对象、`packed_layers == len(computed_layers)`、`skipped_repack == len(reused_layers)`、驻留不变

**实际结果：** 自动化 `::TestCombinedReusePacked::test_skipped_layers_keep_packed_object` 通过（对象同一、resident 相等）

**状态：** PASS

---

### ST-015：Orchestrator 配置下发与每 run reset

**关联需求：** srs.md §3.5 FR13.5（验收标准 1/2）
**测试类型：** 正常路径
**优先级：** High

**前置条件：**
- Given config `execution` 块（replace + reuse 0.5）

**测试步骤：**
1. `initialize` 后检查后端配置生效
2. monkeypatch counting reset，连续两次 `run_optimized`

**期望结果：**
- Then 配置下发正确；每次 run 前 reset（计数 >= 2）

**实际结果：** 自动化 `::TestOrchestratorExecutionWiring::test_initialize_applies_execution_and_runs_reset` 通过（配置断言 + 计数 2）

**状态：** PASS

---

### ST-016：run_stream reset

**关联需求：** srs.md §3.5 FR13.5（每次 run reset）
**测试类型：** 边界条件
**优先级：** Medium

**前置条件：**
- Given 同上配置

**测试步骤：**
1. 连续两次消费 `run_stream` 生成器

**期望结果：**
- Then 每次流式 run 前 reset（计数 >= 2）

**实际结果：** 自动化 `::TestRunStreamReset::test_run_stream_resets_execution_state` 通过（计数 2）

**状态：** PASS

---

### ST-017：执行统计到达 shadowkv/profiler 结果

**关联需求：** srs.md §3.5 FR13.5（验收标准 3：profiler 可观测）、§3.2 FR13.2（drifts 暴露）
**测试类型：** 正常路径
**优先级：** High

**前置条件：**
- Given execution 配置 packed+reuse，真实后端 run_optimized

**测试步骤：**
1. 运行 2 步，检查末步 step 字典

**期望结果：**
- Then `shadowkv["execution"]` 与 `profiler["execution"]` 均含 `kv_execution`/`reused_ratio`/`drifts`/`data_plane_ratio`

**实际结果：** 自动化 `::TestExecutionStatsVisibility::test_execution_stats_reach_shadowkv_and_profiler` 通过（双通道字段断言）

**状态：** PASS

---

### ST-018：后端 forward 失败显式 fallback 标记

**关联需求：** srs.md §3.5 FR13.5（异常显式 fallback，不得静默）、§4 可靠性
**测试类型：** 异常处理
**优先级：** High

**前置条件：**
- Given 后端 `forward_step` 抛异常（monkeypatch 注入）

**测试步骤：**
1. run_optimized 1 步，检查 step 结果中的 execution 统计
2. 确认模拟回退路径不崩溃（下游 agent 契约保持）

**期望结果：**
- Then `execution["fallback"] is True` 且含 `backend_forward_failed` 原因；run 正常完成

**实际结果：** 自动化 `::TestOrchestratorFallbackMarker::test_backend_forward_failure_sets_fallback_marker` 通过（标记断言；修复了回退占位符 0-D 标量导致的 ShadowKV 4-D 契约崩溃，见缺陷记录 D-01）

**状态：** PASS

---

### ST-019：受控轨迹基准断言

**关联需求：** srs.md §3.6 FR13.6（验收标准 1/2/3）
**测试类型：** 集成（真实基准脚本）
**优先级：** High

**前置条件：**
- Given 3 规模 × 3 seed × 3 repeats，四组对照（baseline/reuse/packed/combined）

**测试步骤：**
1. 运行 `ar013_bench.py`（真实 SimpleDiffusionLLM forward，无 mock）
2. 检查内置断言：全 seed 质量（relL2<=0.02）与驻留（data-plane<=0.6）；seed0 速度（reuse>=1.15x、combined>=1.10x）

**期望结果：**
- Then 全部断言通过，输出 median/min-max

**实际结果：** 2026-10-20 实跑通过：reuse 1.76-1.94x、relL2 0.94-0.96%、跳过 50%；int8 packed 驻留 data-plane 53%（全口径 72%，FP16 dense 基准）、relL2 0.00%；combined 1.09-1.44x；脚本输出 "all AR013 benchmark assertions passed"

**状态：** PASS

---

### ST-020：服务器复现配置完整性

**关联需求：** srs.md §3.6 FR13.6（验收标准 4）
**测试类型：** 文档/配置检查（MANUAL）
**优先级：** Medium

**前置条件：**
- Given `configs/experiments/s8_step_reuse_gpu.yaml` 与 `server-experiment.md`

**测试步骤：**
1. 检查 yaml 头部 7B 维度注释（32 层/32 头/128/4096/11008/32000/8192）、完整命令（含 --seed）、记录要求、协议指引
2. 检查 server-experiment.md 的 5 组对照、质量/延迟/显存模板、双口径判定、bench 路径（changes/archive 双路径）

**期望结果：**
- Then 7B 维度、命令、对照、模板齐备，可按文档在服务器复现

**实际结果：** 人工核验通过（yaml L1-19 含 7B 维度与完整命令；server-experiment.md §1 判定行 FP16 双口径、§3 双路径、§4 五组对照与模板、§4.3 三字段记录）

**状态：** PASS

---

### ST-021：全量回归与覆盖率

**关联需求：** srs.md §4 非功能（覆盖率 >= 83.23%、无回归）
**测试类型：** 回归测试
**优先级：** High

**前置条件：**
- Given 仓库全量测试套件

**测试步骤：**
1. 运行 `pytest`（含覆盖率门禁 75）
2. 运行五件套质量门禁（black/isort/flake8/mypy/pytest）

**期望结果：**
- Then 全部通过、覆盖率不低于 AR012 基线（83.23%）、无回归

**实际结果：** 2026-10-20 实跑：876 passed / 1 skipped，覆盖率 83.37%（>= 83.23%）；black/isort/flake8/mypy 全绿；AR012 回归 `tests/test_kv_store.py` 41 passed

**状态：** PASS

---

## ST 执行报告

| 字段 | 内容 |
|------|------|
| 执行日期 | 2026-10-20 |
| 执行结果 | PASS |
| 执行轮次 | 第 1 轮 |

### 需求覆盖矩阵

| 需求 ID | 需求描述 | 测试用例 | 结果 |
|--------|---------|---------|------|
| §3.1 FR13.1 | Diffusion KV 生命周期 | ST-001, ST-002, ST-003 | PASS |
| §3.2 FR13.2 | 漂移门控跨步复用 | ST-004, ST-005, ST-006, ST-007, ST-008, ST-013 | PASS |
| §3.3 FR13.3 | 紧凑 KV 闭环执行 | ST-009, ST-010, ST-011, ST-012, ST-013 | PASS |
| §3.4 FR13.4 | 复用层跳过重 pack | ST-014 | PASS |
| §3.5 FR13.5 | 工程化接线 | ST-015, ST-016, ST-017, ST-018 | PASS |
| §3.6 FR13.6 | 可复现实验与服务器协议 | ST-019, ST-020 | PASS |
| §4 非功能 | 覆盖率/兼容/可靠性 | ST-004, ST-021 | PASS |

**需求覆盖率：** 7 / 7（100%）

### 测试执行汇总

| 类型 | 总计 | 通过 | 失败 | 阻塞 |
|------|------|------|------|------|
| 正常路径 | 10 | 10 | 0 | 0 |
| 边界条件 | 5 | 5 | 0 | 0 |
| 异常处理 | 4 | 4 | 0 | 0 |
| 集成/回归 | 2 | 2 | 0 | 0 |
| **合计** | **21** | **21** | **0** | **0** |

### 缺陷记录

| 编号 | 严重性 | 描述 | 处理方式 |
|------|-------|------|---------|
| D-01 | Major | ST-018 首跑暴露：后端 forward 失败的模拟回退路径中，0-D 标量占位符 `attention_scores`/`kv_current` 违反 ShadowKV 的 4-D 张量契约，`_run_shadowkv` 崩溃（ValueError: not enough values to unpack）——回退路径不可用（AR013 前即存在，AR013 的 fallback 标记使该路径首次被测试覆盖） | 已修复：orchestrator 失败分支按模型配置形状化占位符（`orchestrator.py` 失败分支），复测通过 |
| D-02 | Minor | Review D2：`test_m1_vectorization.py::test_compress_kv_perf_5x` 负载敏感断言偶发抖动（AR002 遗留，非 AR013 缺陷） | 延后处理：留待后续 AR 改为多次取中位 |

### 结论

> Go。21/21 用例通过，需求覆盖 7/7（100%），无未修复 Critical/Major 缺陷（D-01 已修复并复测），全量回归 876 passed / 1 skipped、覆盖率 83.37% 无回退，五件套全绿。D-02 为 AR002 遗留 Minor，已记录延后。
