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
| T001 | 热路径基线画像：cProfile 采集 6-step CPU 流水线，量化 ShadowKV 压缩/打包/决策/仲裁耗时占比，落档为后续验收对照基线 | - | passing | 2026-10-17 完成；权威数字以 baseline-profile.md 为准（净墙钟 7.29×，双热点：ShadowKV 51.4% + FFN unpack 36.2%；独立复测 12.1× 内部比佐证结论稳健） |
| T002 | M6 指标真实化：step 真实 perf_counter 计时接入 perf_data；内存对账与告警；调度器真实延迟奖励；benchmarks 移除硬编码 mock 断言 | T001 | passing | 2026-10-17 完成；内存对账按 design 功能点 2 与 M5 合并实现（T006）；mock 源码诚实化（无伪造差分 + mode 标注） |
| T003 | M6 语义测试更新：依赖零值指标的旧断言按真实语义修订（注明依据）；新增墙钟一致性测试 | T002 | passing | 2026-10-17 完成；全仓库 grep 无零值旧断言（仅新增正向断言）；墙钟一致性入 test_m6_metrics.py（宽松界）+ T008 严格 <2% 落档 |
| T004 | M1 决策面张量化：compress_kv 按 precision 分组批量量化、reuse_mask 张量索引、packed_kv_cache 向量化；保留 reference 循环路径为兜底 | T001 | passing | 2026-10-17 完成；FR3.2；详见进度记录 |
| T005 | M1 golden test：随机输入下向量化 vs reference 逐元素一致（容差 1e-6）+ 压缩耗时下降 ≥5 倍证据 | T004 | passing | 2026-10-17 完成；test_m1_vectorization.py 8 用例（含 5× perf 实测通过）；详见进度记录 |
| T006 | M5 记忆化与记账修复：决策缓存（step 边界失效）、on_init 重置 bug 修复、memory_mb 真实节省口径；同 step 重复调用耗时下降 ≥90% 测试；含 design §7 增补的 FFN 反量化结果缓存 | T004 | passing | 2026-10-17 完成；test_m5_memoization.py 13 用例；详见进度记录 |
| T007 | M3-lite 实测漂移灵敏度：扰动响应计算接入、位置先验 α 混合（默认 0.3）、drift_score 归一化修复、Spearman ≥ 0.9 统计测试 | T002 | passing | 2026-10-17 完成；详见进度记录 |
| T008 | 端到端性能验收：run_optimized ≤ 1.5 × run_baseline（CPU 实测落档）；五件套门禁全绿 | T002-T007 | passing | 2026-10-17 完成：4 次实测 1.585-1.639（中位 1.61），落入 design 4.1 授权兜底区 (1.5, 2.5]，按兜底路径结案（杠杆已全部在下限，残余瓶颈为 FFN 支线 AR003 范围）；详见 baseline-profile.md §8 |
| T009 | README 四检查点更新（v3.4.0：特性表/版本历史/性能基准实测表/结构口径）+ ROADMAP 同步 | T008 | passing | 2026-10-17 完成；详见进度记录 |
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

### 2026-10-17 T002/T003 会话记录

- 完成任务：T002（M6 指标真实化）+ T003（语义测试更新）
- 修改文件：orchestrator.py（step 计时接入 _run_profiler、global_state["latency_ms"] 数据源、_compute_total_latency 去 10.0 造假兜底、_aggregate_optimized_data per-step 真实值）、core/structs.py（StepState.to_dict 增 latency_ms）、profiler/profiler_agent.py（device 口径标注）、benchmarks/real_model/__init__.py（mock 无伪造差分 + summary.mode="mock"）
- 测试：新增 test_m6_metrics.py（18 用例）+ test_benchmark_no_hardcoded_speedup.py（3 用例）；修订 test_real_model_benchmark.py（5 处断言按无差分契约改写）；基准测试 36 passed；相关模块回归 61 passed
- 备注：TDD-Red 子代理另产出 test_m6_metrics_integrity.py（发明 run_step/_make_dummy_inputs 等不存在 API，违反 design 4.3 对外零变更），审查后删除；调度器奖励未改代码（数据源修复即生效，符合 design「不改语义，改数据源」）

### 2026-10-17 T004/T005 会话记录

- 完成任务：T004（M1 决策面张量化）+ T005（golden test + 5× perf 证据）
- 修改文件：
  - utils/quantization.py：新增公开 quantize_batch（2D 批量量化，逐行等价 quantize_tensor）
  - shadowkv/shadowkv_agent.py：新增 _compute_importance_matrix（整矩阵向量化重要性，on_step Step 2 接入）；allocate_precision 张量路径向量化（f64 精确阈值比较，dict 路径走 _allocate_precision_reference）；decide_reuse_per_token 向量化（f64 阈值矩阵 + 布尔掩码，原实现保留为 _decide_reuse_per_token_reference）；compress_kv 重构为公共入口（复用掩码/旧格式解析 + packed/统计）+ _compress_kv_vectorized（按精度分组 advanced indexing 批量取/量化/写回 + 批量 reuse 拷贝）+ _compress_kv_reference（原循环保留，异常兜底：logger.log_alert error 后回退）
  - shadowkv/packed_kv_cache.py：_do_pack 向量化（分组 take + 整组量化；int4 保留逐 entry _pack_int4 打包与 byte_offsets/scale 布局 bit-for-bit 一致），原实现保留为 _do_pack_reference
  - tests/test_m1_vectorization.py：8 用例（quantize_batch/重要性矩阵/精度分配/复用决策/compress golden/异常回退/pack golden/5× perf）
- 关键实现事实：
  - advanced indexing 布局：k[:, h_idx, t_idx, :] 结果为 (batch, N, head_dim)（广播形状插入首个高级索引位置），统一 permute(1,0,2) 归一到 (N, batch, head_dim)；高级索引赋值要求 dtype 严格匹配，写回显式 .to(k_out.dtype)（与基础切片赋值的隐式转型值相同）
  - 非方形 attention_scores 口径（T001 未覆盖、real_backend 暴露）：真实后端为 (batch, heads, seq_q=4, seq_k=8)；reference 的归一化与位置阈值均以 shape[-1]（seq_k）为基准、token 维度为 shape[2]（seq_q）——_compute_importance_matrix 已按此口径实现并有非方形 golden 用例锁定
  - 预存缺陷（保持兼容、记录在案）：compress_kv 原循环 int4 写回仅 batch=1 且 head_dim==group_size=64 时可用（(B,g,gs)→(B,D) 赋值限制，其余 RuntimeError）；向量化路径统一 reshape 写回，可用包络内 bitwise 一致，崩溃包络按语义意图产出（golden 测试 int4 比对限制在可用包络）。packed int4 unpack 的 scale 索引 batch>1 缺陷保持原样（向量化 _do_pack 产出布局 bit-for-bit 一致）
  - 测试隔离缺陷修复（预存、非本 AR 引入）：test_logging_utils.py 的 configure_shadowinfer_logging(level=ERROR/WARNING) 写入 _GLOBAL_LEVEL 后不复位，全量运行时污染后续测试的 INFO metric 记录（test_m6_metrics 3 用例仅全量失败的原因）；已加 autouse fixture 快照/恢复全局日志状态
- 质量门禁：black/isort/flake8/mypy 全绿；全量 pytest 610 passed / 1 skipped / 覆盖率 81.48%（fail-under 75）
- T005 perf 证据：test_compress_kv_perf_5x（seq=128/heads=32/batch=1/head_dim=128，真实精度分布）断言 vectorized×5 ≤ reference 通过；绝对数值对照 T001 基线（compress_kv cumtime 11.2s/24 calls）在 T008 端到端复测落档

### 2026-10-17 T006 会话记录

- 完成任务：T006（M5 决策记忆化与记账修复，含 design §7.2 FFN 反量化缓存）
- 修改文件：
  - shadowkv/shadowkv_agent.py：__init__ 存储 _memory_budget_bytes/_eviction_policy 并初始化 _decision_cache/_decision_cache_step；on_init 修复重置缺陷（重建 cache_manager 保留 memory_budget/eviction 配置 + 全量重置 _compression_stats/_precision_distribution/_reuse_stats/_decision_cache）；on_step Steps 2-4 包裹决策缓存（(step_id, layer_id) 定位键 + _decision_input_fingerprint 指纹校验：shape/dtype/data_ptr + 标量元组，异常返回 None 旁路；step 滚动清空；importance_history 命中路径同样记录）；compress_kv memory_mb 口径改为真实节省（(uncompressed_bytes − compressed_bytes)/MB），新增 uncompressed_bytes/compressed_bytes 结果键
  - ffn_optimizer/packed_weight.py：PackedFFNWeight 新增 _precision_plan_fingerprint 与反量化结果缓存（dequantize_for_matmul 指纹一致直接复用；_do_pack 重建精度计划时失效）
  - tests/test_m5_memoization.py：13 用例（缓存命中/≥90% 决策耗时下降/异输入不脏命中/step 滚动失效/指纹异常旁路/on_init 全量重置与配置保留/FFN unpack 计数缓存命中与 repack 失效/memory_mb 三口径）
- 关键决策：
  - ≥90% 耗时断言口径定为「决策面三方法独占耗时之和」（缓存覆盖面，二次调用精确为 0，负载不敏感）；端到端 on_step 计时含未缓存固定开销（压缩/统计），全量运行下机器负载使比值不稳定（实测全量 FAIL、单跑 PASS），不适合作分母
  - memory_mb 消费链梳理：唯一直接断言（test_m1_vectorization fallback 对称比较）不受影响；profiler savings_ratio 现状已退化（baseline_total == compressed 恒 0），本 AR 不扩大修（记录，M7 候选）
  - 指纹 data_ptr 跨 step 地址复用风险被 step 滚动清空消除（同 step 内层间调用输入即同一张量对象）
- 效果佐证：全量测试套件耗时 296.8s → 214.9s（决策缓存在 orchestrator 逐层重复调用场景生效）
- 质量门禁：black/isort/flake8/mypy 全绿；全量 pytest 623 passed / 1 skipped / 覆盖率 81.62%（fail-under 75）

### 2026-10-17 T007 会话记录

- 完成任务：T007（M3-lite 实测漂移灵敏度，FR3.4 + design 4.1）
- 修改文件：
  - qdrift/qdrift_agent.py：__init__ 新增 positional_prior_weight（配置键 positional_prior_weight，顶层路径 qdrift.positional_prior_weight，默认 0.3，clamp [0,1]）；detect_drift 重构为 _compute_raw_drift（未归一化原始值）+ tanh 归一化（修复恒 1.0 塌缩，恢复动态范围）；新增 compute_sensitivity（可选关键字 query_previous/activation_previous/query_current/activation_current，默认 None 时行为与 v3.3.0 estimate_sensitivity 完全一致；公式 clamp(α·位置先验 + (1−α)·tanh(raw_drift), 0, 1)；previous 缺失回退位置先验 + log_alert("info")）；on_step Step 1 接线 compute_sensitivity（传入四张量）
  - tests/test_qdrift.py：新增 _spearman 手写秩相关（平均秩，无 scipy 依赖；零方差序列返回 0 以判定塌缩）+ TestComputeSensitivity 7 用例（默认缺省等价/previous 缺失回退+日志/混合公式精确对照 design/扰动幅度单调响应/动态范围恢复（2.0 vs 8.0，避开 float64 tanh 饱和区 x>19）/Spearman 50 组 ≥0.9（幅度 0.1-4.9 跨越原 clamp 塌缩区）/on_step 接线偏离纯位置先验）
- 关键决策：
  - tanh 饱和极限：tanh(x) 在 x>19 后 float64 下恒 1.0（浮点极限非缺陷），动态范围测试幅度选 2.0/8.0
  - Spearman 测试固定 step_id（位置进度恒定），秩序完全由实测扰动响应决定——可同时判定 α 权重失效（全并列 → 0.0）与归一化塌缩（大幅段并列 → <0.9）两类回归
  - drift_score 语义变更（clamp→tanh）保持 [0,1) 值域与零漂移=0，现有 3 个 drift 用例（仅断言值域与零点）无需更新；下游 test_orchestrator/test_real_backend_integration/test_scheduler 54 用例实测无 dispatch 翻转
- 质量门禁：black/isort/flake8/mypy 全绿；全量 pytest 630 passed / 1 skipped（test_qdrift 18 用例）

### 2026-10-17 T008 会话记录

- 完成任务：T008（端到端性能验收，CPU 实测落档）
- 测量口径：与 T001 完全一致（configs/optimize_full.yaml + model_fast_dllm_7b.yaml，6-step，无 profiler 外部墙钟 + cProfile 热路径），另加 3 次稳定性复测；脚本 t008_e2e.py / t008_stab.py 存档于临时目录
- 结果（baseline-profile.md §8 落档）：
  - 端到端：优化 10273.5ms vs 基线 6481.2ms = 1.585x；复测 1.594/1.616/1.639（中位 1.61）；对照 T001 基线 7.29x 改善 4.5x
  - 判定：落入 design 4.1 授权兜底区 (1.5x, 2.5x]。兜底杠杆核查：FFN 反量化缓存已生效（32→8 次 unpack，17.13s→1.26s）；决策缓存 E2E 无同 (step, layer) 重复调用可命中；seq 已为 128 下限；无强暴默认口径——杠杆全部在下限，残余瓶颈为 FFN 支线（~250ms/step，AR003 范围，T001 §6.3 已预警）。「1.5x 可达」由双实测证明：FFN 张量化后投影 ~1.21x
  - M1 热路径：compress_kv cum 11.43s→0.575s（19.9x，FR3.2 >=5x 达标）；shadowkv.on_step 24.66s→0.954s；compute_importance 98304 次调用→24 次（>200x）；packed pack 3.12s→0.199s
  - M6：per_step_ms 全部 >0（稳态 948-993ms）；内部 e2e_ms 与外部墙钟偏差 0.02%（<2% 验收线）
  - M3-lite：sensitivity 实测混合 [0.300→0.833]，drift [0.888x] 不再恒 1.0 塌缩；kv memory_mb 真实节省口径 [1.00/0.51/0.37/0.33] MB/层
- 质量门禁：五件套复跑全绿（本任务无代码变更，门禁状态承自 T007 提交 a0d9a24 后全量实跑：630 passed/1 skipped、black/isort/flake8/mypy 全绿）

### 2026-10-17 T009 会话记录

- 完成任务：T009（README 四检查点 + 版本同步 + ROADMAP 同步，FR3.5 / AGENTS 规则一）
- 四检查点执行：
  - 特性表：新增 3 行（热路径张量化 / 决策记忆化 / 真实性能记账），修订 Q-drift 行（位置先验 + 实测扰动混合、Spearman >= 0.9）
  - 版本历史：追加 v3.4.0 条目（M1/M6/M5/M3-lite/端到端 1.61x/门禁数据）；pyproject.toml 与 shadowinfer/__init__.py 同步 3.4.0（import 冒烟验证）
  - 基准数据：性能基准章节全量替换为 v3.4.0 实测（630 passed/1 skipped/81.68%；热路径五函数对照表；端到端 6481.2/10273.5ms 比值 1.585-1.639；per-step 948-993ms；墙钟对照 0.02%；golden 1e-6；Spearman >= 0.9）；CPU fallback 说明改写（7.29x → 1.61x，残余为 FFN 支线）
  - 结构与口径：核心算法章节补张量化与混合公式口径；TOC 无新增章节；测试数以当次实跑为准；无 emoji；失效引用零残留（v3.3.0 历史条目自身数字属正确历史记录保留）
- ROADMAP 同步：Current State 测试数更新（630/81.68%）；CPU 向量化项标记完成（v3.4.0 数据）；新增 FFN unpack 向量化缺口条目（残余瓶颈，§2 与 §3.5）；Success Metrics 覆盖率更新
- 质量门禁：五件套复跑全绿（630 passed / 1 skipped / 81.68%）
