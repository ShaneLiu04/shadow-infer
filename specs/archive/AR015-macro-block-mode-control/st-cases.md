# [AR015] ST 验收测试用例

| 字段 | 内容 |
|------|------|
| AR 编号 | AR015 |
| 关联 srs.md | ./srs.md |
| 生成日期 | 2026-10-21 |

## 测试用例列表

### ST-001：rewrite_tokens 三精度 golden 逐位等价

**关联需求：** srs.md §3.1 FR15.1
**测试类型：** 正常路径
**优先级：** High

**前置条件：**
- PagedKVStore 已 append 全量基线内容

**测试步骤：**
1. 先 append 全量再 `rewrite_tokens` 部分行 vs 直接 append 覆写后内容
2. fp16/int8/int4 三精度分别执行

**期望结果：**
- gather 输出逐元素相等（bitwise golden）

**实际结果：** PASS — `TestRewriteGoldenEquivalence`（tests/test_macro_mode_execution.py）fp16/int8/int4 三精度 `torch.equal` 全通过（ST 执行 95 passed 之一，2026-10-21）

**状态：** PASS

---

### ST-002：驱逐行拒绝覆写 + 失败原子性

**关联需求：** srs.md §3.1 FR15.1
**测试类型：** 异常处理
**优先级：** High

**前置条件：**
- 页内部分 `(token,head)` 已驱逐

**测试步骤：**
1. `rewrite_tokens` 覆盖含驱逐行的区间
2. 核验抛错后 store 全态（payload/scale/计数）不变

**期望结果：**
- 抛 `ValueError`；未覆写行逐位不变（失败原子性）

**实际结果：** PASS — `TestRewriteEvictedRows` + `TestRewriteValidation`（`_assert_unchanged` 全态快照核验，含 shape/precision/越界/驱逐四类）

**状态：** PASS

---

### ST-003：跨页宏块覆写与页不变量

**关联需求：** srs.md §3.1 FR15.1
**测试类型：** 边界条件
**优先级：** High

**前置条件：**
- macro=64、page=32 的跨页宏块

**测试步骤：**
1. 整宏块 `rewrite_tokens`
2. 核验两页各自覆写、未触碰页逐位不变、页边界/页 id 不变、`data_plane_bytes`/`resident_bytes` 不变

**期望结果：**
- 两页正确覆写；未触碰页逐位不变；`rewritten_tokens`/`rewritten_pages` 记账正确

**实际结果：** PASS — `TestRewriteCrossPageSplit` + `TestRewritePageInvariants` + `TestRewriteAccounting`

**状态：** PASS

---

### ST-004：构造收敛轨迹三态分类

**关联需求：** srs.md §3.2 FR15.2
**测试类型：** 正常路径
**优先级：** High

**前置条件：**
- 前 m 行冻结、后段漂移的构造轨迹；缓存已建立

**测试步骤：**
1. `MacroModeState.classify` 分类
2. 核验模式计数与活跃行集合

**期望结果：**
- 冻结段 SKIP、漂移段按行集合进 DELTA/REFRESH；batch 均匀性（任一 batch 超阈即活跃）；尾块实际行数比例

**实际结果：** PASS — 轨迹分类/边界/`TestMacroModeTailPartialBlock`（1/2 活跃→REFRESH）、零范数、batch 均匀性、跨层计数、`record_rows`-冻结闭环等 35 UT 全通过

**状态：** PASS

---

### ST-005：迟滞即时解除

**关联需求：** srs.md §3.2 FR15.2
**测试类型：** 边界条件
**优先级：** Medium

**前置条件：**
- 宏块处于 SKIP

**测试步骤：**
1. 下一步该宏块任一行输入突增漂移（超 `skip_threshold`）
2. 分类

**期望结果：**
- 当步即解除 SKIP（进入 DELTA/REFRESH），无需独立 unfreeze 参数

**实际结果：** PASS — `TestMacroModeHysteresisRelease`

**状态：** PASS

---

### ST-006：warmup 全 REFRESH + 缓存建立

**关联需求：** srs.md §3.2 FR15.2
**测试类型：** 正常路径
**优先级：** Medium

**前置条件：**
- `warmup_steps=2`

**测试步骤：**
1. 前 2 步分类
2. 核验模式与 entry 建立

**期望结果：**
- 全部 REFRESH 且缓存建立；`force_refresh_every` 周期步同理

**实际结果：** PASS — warmup 与 force_refresh 独立 UT 全通过

**状态：** PASS

---

### ST-007：行粒度分段前向 vs 全量前向

**关联需求：** srs.md §3.3 FR15.3
**测试类型：** 正常路径
**优先级：** High

**前置条件：**
- 单层 block、活跃行外输入与缓存一致

**测试步骤：**
1. 行粒度分段前向 vs 全量前向
2. 核验 K/V 组装、活跃行输出、冻结行输出

**期望结果：**
- K/V 组装逐位相等；活跃行输出 relL2 <= 1e-6（实测 ≈7.7e-8 末位噪声，SRS §3.3 修正口径）；冻结行输出=残差修正值；空活跃行合法（全 SKIP 层）

**实际结果：** PASS — 分段 vs dense、全 SKIP 回放、残差修正公式、DELTA/REFRESH record、`last_attention` 形状等 10 UT 全通过

**状态：** PASS

---

### ST-008：全 REFRESH 逐位回归锚点

**关联需求：** srs.md §3.3 FR15.3
**测试类型：** 回归测试
**优先级：** High

**前置条件：**
- 全 REFRESH（活跃=全部行）

**测试步骤：**
1. 分段路径 vs 现行路径输出与 K/V 对比

**期望结果：**
- 输出与 K/V 逐位一致（bitwise）

**实际结果：** PASS — `TestMacroModeFreshFullRefresh`（全活跃捷径逐位）

**状态：** PASS

---

### ST-009：模式控制关闭逐位回归

**关联需求：** srs.md §3.3 FR15.3
**测试类型：** 回归测试
**优先级：** High

**前置条件：**
- `mode_state=None`（模式控制关闭）

**测试步骤：**
1. forward_step 执行
2. 与 v3.16.0 参考路径输出对比

**期望结果：**
- 与 v3.16.0 输出逐位一致

**实际结果：** PASS — `TestMacroModeDisabledRegression`；全量套件含全部 921 基线用例（最终状态实跑 1016 passed / 1 skipped）

**状态：** PASS

---

### ST-010：三步端到端统计与 SKIP 零页写

**关联需求：** srs.md §3.4 FR15.4
**测试类型：** 正常路径
**优先级：** High

**前置条件：**
- `mode_control_enabled=true`（paged+replace）

**测试步骤：**
1. 端到端跑 3 步
2. 核验模式计数、`rewritten_rows`、SKIP 块页对象 `id()`

**期望结果：**
- 模式计数单调合理；`rewritten_rows == sum(活跃行)`；SKIP 块页 id 不变（零页写）；八统计键恒存在；首步冷启动全 REFRESH

**实际结果：** PASS — SKIP 零页写页身份不变、DELTA-REFRESH 分派精确预期（参考回放推导 rewritten==10）、首步冷启动、三步统计双通道 UT 全通过

**状态：** PASS

---

### ST-011：默认关闭全量回归不回退

**关联需求：** srs.md §3.4 FR15.4（兼容性 NFR）
**测试类型：** 回归测试
**优先级：** High

**前置条件：**
- `mode_control_enabled=false`（默认）

**测试步骤：**
1. 全量测试套件执行

**期望结果：**
- 921 基线用例全绿（逐位不变）；总用例 >= 921

**实际结果：** PASS — 最终状态全量实跑 **1016 passed / 1 skipped，覆盖率 84.25%**（基线 921/83.78% 全包含，+95 用例、+0.47pp；2026-10-21 两轮实跑一致）

**状态：** PASS

---

### ST-012：互斥与前置违规显式报错

**关联需求：** srs.md §3.4 FR15.4
**测试类型：** 异常处理
**优先级：** High

**前置条件：**
- 各违规配置组合

**测试步骤：**
1. 非 paged / 非 replace / 与 step_reuse 同开分别 configure

**期望结果：**
- 显式 `ValueError`（不静默禁用）

**实际结果：** PASS — 配置校验 UT（六新参显式 opt-in 校验，含互斥/前置七类 ValueError）

**状态：** PASS

---

### ST-013：rewrite 失败注入 fallback 与恢复

**关联需求：** srs.md §3.4 FR15.4
**测试类型：** 异常处理
**优先级：** High

**前置条件：**
- monkeypatch 注入 rewrite 失败（AR013 故障注入先例，零 mock 原则内的故障注入）

**测试步骤：**
1. 注入失败步执行
2. 后续步执行

**期望结果：**
- fallback 置位、输出可用（dense 全量重算）、模式状态 reset、后续步自动恢复；模型调用失败同理

**实际结果：** PASS — rewrite 失败 fallback + 后续步恢复、模型调用失败 fallback、shape 突变 reset（`TestModeShapeMutationReset`）、reset 隔离 UT 全通过

**状态：** PASS

---

### ST-014：Orchestrator 配置零改动透传与统计双通道

**关联需求：** srs.md §3.5 FR15.5
**测试类型：** 正常路径
**优先级：** Medium

**前置条件：**
- orchestrator 配置含 mode_control 参数

**测试步骤：**
1. 运行 orchestrator
2. 核验后端收到的参数与统计可见性

**期望结果：**
- `configure(**cfg)` 自动透传全部六参数；统计沿 `execution_stats` 与 profiler 双通道可见；无配置时零触碰

**实际结果：** PASS — `TestOrchestratorExecutionWiring`（cfg 透传/无配置零触碰；orchestrator.py 零改动经 git diff 核验）

**状态：** PASS

---

### ST-015：HF 后端不支持口径不崩溃

**关联需求：** srs.md §3.5 FR15.5
**测试类型：** 异常处理
**优先级：** Medium

**前置条件：**
- MockModelBackend（无 configure_execution 支持，对齐 HF 口径）

**测试步骤：**
1. 配置模式控制并运行

**期望结果：**
- 沿现有配置不支持口径处理（execution_unsupported 事件，不崩溃、明确告警）

**实际结果：** PASS — MockModelBackend 不支持口径 UT（orchestrator 记 execution_unsupported 不崩溃）

**状态：** PASS

---

### ST-016：bench 终版预注册断言 + 逐 seed 落档

**关联需求：** srs.md §3.6 FR15.6
**测试类型：** 正常路径
**优先级：** High

**前置条件：**
- `ar015_bench.py` 双轨迹四臂配置就绪；终版断言跑前登记（design §6.5 + bench-results 预注册历史）

**测试步骤：**
1. 全量 bench 执行（T1 均匀 / T2 块式半 AR × 3 规模 × 3 seeds × 四臂）
2. 核验终版断言（f2 保险臂）与逐 seed 落档

**期望结果：**
- f2 全步 relL2 <= 3%（3 seeds 最差）、末半程冻结率 >= 30%；加速比只报告不断言；逐 seed 原始数值与三轮预注册历史（含两次失败）完整落档 bench-results.md；未达标如实报告

**实际结果：** PASS — 2026-10-21 全量实跑：f2 relL2 worst 1.881%/2.649%/1.988%（S=256/512/1024）、freeze 0.667 全规模、脚本内置断言全 PASS；加速如实 0.642/0.831/1.043x（无断言）；T1 边界（0.516-0.789x、relL2 2.248-3.119%）与 f0 累积-自愈轨迹（峰值 6.4-10.1% 后回落）如实落档；README 按实测修订口径（「端到端无加速收益」诚实结论）

**状态：** PASS

---

### ST-017：s10 yaml + server-experiment 可复现性

**关联需求：** srs.md §3.6 FR15.6
**测试类型：** 正常路径
**优先级：** Medium

**前置条件：**
- 无

**测试步骤：**
1. 核验 `configs/experiments/s10_mode_control_gpu.yaml` 与 `server-experiment.md` 存在与内容完备性

**期望结果：**
- 四臂协议（dense/ar013/f0/f2）+ 维度注释 + --seed 复现命令 + 判定标准 + 记录模板齐备；GPU 不宣称（本机不宣称 GPU 结果）

**实际结果：** PASS — 两文件均存在（Test-Path 实核验）；s10 含四臂注释/维度/用法/统计键/判定；server-experiment 含环境/本机基线/四臂命令/质量取证/显存统计/记录模板/结论落档；Review X1 核验四处口径一致

**状态：** PASS

---

### ST-018：五件套质量门禁

**关联需求：** srs.md §4 NFR（质量门禁）
**测试类型：** 回归测试
**优先级：** High

**前置条件：**
- 无

**测试步骤：**
1. black/isort/flake8/mypy/pytest 全量执行

**期望结果：**
- 五件套全绿；覆盖率 >= 83.78%；总用例 >= 921

**实际结果：** PASS — black 136 文件 unchanged / isort exit=0 / flake8 exit=0 / mypy 74 文件 no issues / pytest **1016 passed / 1 skipped，覆盖率 84.25%**（>= 921 与 >= 83.78% 均超基线；最终状态实跑，2026-10-21）

**状态：** PASS

---

### ST-019：版本 3.17.0 四处同步

**关联需求：** srs.md §4 NFR（版本同步）
**测试类型：** 正常路径
**优先级：** Medium

**前置条件：**
- 无

**测试步骤：**
1. 核验 pyproject.toml / `__init__.py` / `__main__.py` / `test_ffn_hotpath.test_ut17` 版本一致

**期望结果：**
- 四处均 == 3.17.0

**实际结果：** PASS — 四处 3.16.0→3.17.0（git diff 核验）；`test_ffn_hotpath.py` 34 passed（含 test_ut17，2026-10-21 单独复验 + 全量套件内通过）

**状态：** PASS

---

### ST-020：README 四检查点与诚实纪律

**关联需求：** srs.md §4 NFR（文档/性能主张纪律）
**测试类型：** 正常路径
**优先级：** High

**前置条件：**
- 无

**测试步骤：**
1. 核验特性表 v3.17.0 条目、质量表、基准节、版本历史、TOC、emoji 扫描
2. 核验优越性声称可溯源

**期望结果：**
- 四检查点齐备；测试数/覆盖率以实跑为准；无 emoji；无超出实跑证据的优越性声称；未验证部分「机制说明（未验证）」口径

**实际结果：** PASS — 特性表 v3.17.0 条目（诚实结论口径：端到端无加速收益，交付物=正确性/误差有界/自愈/零页写）、质量表 1016/84.25%、基准节 T2/T1 双表（逐数字与 bench-results 核对无误）、版本历史 v3.17.0；TOC 章节级零失效锚点；emoji 扫描 clean；GPU 收益只给协议不给数字（Review C2/X3 核验）

**状态：** PASS

---

## 执行摘要

| 总计 | 通过 | 失败 | 阻塞 |
|------|------|------|------|
| 20 | 20 | 0 | 0 |

## ST 执行报告

| 字段 | 内容 |
|------|------|
| 执行日期 | 2026-10-21 |
| 执行结果 | PASS |
| 执行轮次 | 第 1 轮 |

### 需求覆盖矩阵

| 需求 ID | 需求描述 | 测试用例 | 结果 |
|--------|---------|---------|------|
| §3.1 | PagedKVStore.rewrite_tokens | ST-001, ST-002, ST-003 | PASS |
| §3.2 | MacroBlockModeController 分类器 | ST-004, ST-005, ST-006 | PASS |
| §3.3 | 行粒度分段前向 | ST-007, ST-008, ST-009 | PASS |
| §3.4 | 后端模式控制执行 | ST-010, ST-011, ST-012, ST-013 | PASS |
| §3.5 | Orchestrator 接线与回归 | ST-014, ST-015 | PASS |
| §3.6 | 基准、等价性与服务器协议 | ST-016, ST-017 | PASS |
| §4 NFR | 五件套/版本同步/文档纪律 | ST-018, ST-019, ST-020 | PASS |

**需求覆盖率：** 7 / 7（100%）

### 测试执行汇总

| 类型 | 总计 | 通过 | 失败 | 阻塞 |
|------|------|------|------|------|
| 正常路径 | 9 | 9 | 0 | 0 |
| 边界条件 | 3 | 3 | 0 | 0 |
| 异常处理 | 4 | 4 | 0 | 0 |
| 回归测试 | 4 | 4 | 0 | 0 |
| **合计** | **20** | **20** | **0** | **0** |

### 遗留问题

| 严重性 | 描述 | 处理方式 |
|-------|------|---------|
| Minor | tasks.md 各任务 UT 备注合计 96 与实测 95 差 1（T005 备注多数 1 条） | 延后处理（以实测 95 与 README/bench-results 口径为准；Review 记录已注明） |
| Minor | `last_attention` 跨层共享缺陷沿 AR013 现状（模式路径已标注活跃行子集） | 延后处理（独立 AR 候选） |
| Minor | AR013 遗留 test_m1_vectorization 偶发抖动（D-02） | 延后处理（沿 AR013 验收遗留口径） |

### 结论

> **Go**。全部 20 ST 用例 PASS（100% 需求覆盖）；无 Critical/Major 缺陷；五件套门禁全绿且覆盖率（84.25%）与用例数（1016）超基线；三轮预注册历史与「端到端无加速收益」负结果完整落档，诚实纪律合规；3 项 Minor 均为延后处理候选，不阻断归档。
