# [AR002] 任务跟踪

| 字段 | 内容 |
|------|------|
| AR 编号 | AR002 |
| 关联 srs.md | ./srs.md |
| 关联 design.md | ./design.md（待生成） |
| 创建日期 | 2026-10-17 |

## 任务列表

| ID | 任务描述 | 依赖 | 状态 | 备注 |
|----|---------|------|------|------|
| T001 | 热路径基线画像：cProfile 采集 6-step CPU 流水线，量化 ShadowKV 压缩/打包/决策/仲裁耗时占比，落档为后续验收对照基线 | - | passing | 2026-10-17 完成（optimize_full.yaml 口径复测：baseline 3817.7ms / optimized 46295.7ms = 12.1× 倒挂；shadowkv 支线 48.1%、FFN unpack 33.7% 双热点；drift_score 0.0→恒 1.0 退化实录；证据存 baseline-profile.md） |
| T002 | M6 指标真实化：step 真实 perf_counter 计时接入 perf_data；内存对账与告警；调度器真实延迟奖励；benchmarks 移除硬编码 mock 断言 | T001 | pending | FR3.1 |
| T003 | M6 语义测试更新：依赖零值指标的旧断言按真实语义修订（注明依据）；新增墙钟一致性测试 | T002 | pending | FR3.1 验收 |
| T004 | M1 决策面张量化：compress_kv 按 precision 分组批量量化、reuse_mask 张量索引、packed_kv_cache 向量化；保留 reference 循环路径为兜底 | T001 | pending | FR3.2 |
| T005 | M1 golden test：随机输入下向量化 vs reference 逐元素一致（容差 1e-6）+ 压缩耗时下降 ≥5 倍证据 | T004 | pending | FR3.2 验收 |
| T006 | M5 记忆化与记账修复：决策缓存（step 边界失效）、on_init 重置 bug 修复、memory_mb 真实节省口径；同 step 重复调用耗时下降 ≥90% 测试；含 design §7 增补的 FFN 反量化结果缓存 | T004 | pending | FR3.3 + design §7.2 |
| T007 | M3-lite 实测漂移灵敏度：扰动响应计算接入、位置先验 α 混合（默认 0.3）、drift_score 归一化修复、Spearman ≥ 0.9 统计测试 | T002 | pending | FR3.4 |
| T008 | 端到端性能验收：run_optimized ≤ 1.5 × run_baseline（CPU 实测落档）；五件套门禁全绿 | T002-T007 | pending | FR NFR |
| T009 | README 四检查点更新（v3.4.0：特性表/版本历史/性能基准实测表/结构口径）+ ROADMAP 同步 | T008 | pending | FR3.5 |
| T010 | Review 合规检查（S/D/C 维度）+ ST 验收（st-cases.md）+ 归档 | T009 | pending | |
| T011 | 按 Runbook 推送 Gitee（小批量多次；三核验；远端无 AGENTS.md 确认） | T010 | pending | FR3.6 |

## 状态说明

- `pending`：待开始
- `in_progress`：进行中（当前会话）
- `passing`：开发完成，测试通过
- `failed`：测试失败，需修复

## 进度记录

> 每个开发会话结束后追加，记录完成情况。

### 2026-10-17 需求阶段记录

- AR002 源起：AR001 深度审计子代理的框架级发现（18 项缺陷清单，重点：ShadowKV 纯 Python 热路径占优化路径 97.6% 开销、profiler 硬编码零值指标、QDrift 灵敏度为位置插值）+ 本机端到端实测（6-step CPU 基线 31.3ms vs 优化 1782.8ms，57 倍倒挂）
- 关键代码级论断已抽查证实：orchestrator.py:1456-1463（perf_data 全零硬编码）、shadowkv_agent.py:724-753（token×head 双层 Python 循环逐元素量化）
- 范围决策：采纳 M6→M1→M5→M3-lite；M2（backend 契约）/M4（轨迹数据学习）/M7（channel_mask 语义）留后续 AR
- 用户全局授权：全流程自动化执行，无需逐步确认（Gitee token 已持久化于 AGENTS.md，无需再索取）

## 阶段门控记录

> 由 sdd-phase-gate skill 在阶段门控审查后追加，记录每轮审查结果（PASS/FAIL + 轮次）。格式见 sdd-phase-gate SKILL.md Step 6。

### 2026-10-17 req 门控（第 1 轮）

- 结论：**PASS**（G1-G8 全 YES，独立审查子代理逐项复核，代码级论断全部抽查命中）
- Minor-1（G4）：1.5× 端到端目标的可达性测算需在 design.md 显式补充（5× 仅为验收下限，向量化预期 50×+；需写明 T008 未达标的迭代路径）
- Minor-2（G1）：tokens_per_sec / gpu_utilization 的取值口径需在 design.md 明确（CPU 环境前者由真实延迟派生、后者口径固定）
- Minor-3（G3）：drift_score 恒 1.0 的复现证据在 T001 baseline-profile.md 落档佐证
- 决定：准予进入设计阶段，3 项 Minor 随 design.md 编制一并处理

### 2026-10-17 design 门控

- 第 1 轮：**FAIL**（Important×1：决策缓存键 (step_id, layer_id) 与 srs FR3.3 输入指纹语义冲突且 6.4 指纹异常测试无实现主体；Minor×3：benchmarks 断言清理未落设计、scale 等价论证漏 batch 维、流程图 fork 误用）
- 修复：缓存改为「(step_id, layer_id) 定位键 + 输入指纹校验」（4.2.2 点 5，四处口径统一）；#3 表/4.4/6.3 增补 benchmarks 清理；scale 论证补 (N, batch, head_dim)→amax(dim=-1) 逐 batch 等价 + golden 覆盖 batch>1；fork 改 repeat 循环
- 第 2 轮：**PASS**（4 项全修复，无新 Critical/Important；2 项实现层 Minor 随编码处理：int4 scale 分组维注释 + quantize_batch 形状契约）
- 决定：准予进入开发阶段

### 2026-10-17 开发阶段 design 增补（T001 实证修订，用户已授权全流程自动化）

- 触发：T001 基线画像（真实后端净墙钟 7.29×）推翻 design 4.1 可达性测算输入（「FFN ≈ 11 ms」失真，实测 FFN on_step 22.95 s，其中 unpack 17.13 s 占 36.2%）
- 修订：design.md 新增 §7——M5 扩展「FFN 反量化结果缓存」（精度计划指纹校验，语义等价零风险）；可达性重估 1.47×（内部比）
- 依据：baseline-profile.md §3；srs FR3.3 记忆化语义在 FFN 平面的应用，M7（channel_mask 行为）仍 Out of Scope
- 决定：准予按修订后设计继续开发（T006 范围同步扩展）
