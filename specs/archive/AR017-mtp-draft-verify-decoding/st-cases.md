# [AR017] ST 验收测试用例

| 字段 | 内容 |
|------|------|
| AR 编号 | AR017 |
| 关联 srs.md | ./srs.md |
| 生成日期 | 2026-10-23 |

## 测试用例列表

### ST-001：草拟轮全位 argmax 填充无 MASK 残留

**关联需求：** srs.md §3.1 FR17.1 验收 1
**测试类型：** 正常路径
**优先级：** High

**前置条件：**
- Given 全 MASK 块 round-0 logits [B, W, V]

**测试步骤：**
1. 调用 `draft_round(logits)`，检查输出 tokens/confidences

**期望结果：**
- Then 全部 W 位以 argmax 填充；confidences == softmax 最大值；形状/dtype 正确

**实际结果：** PASS（UT `TestDraftRound::test_all_masked_positions_filled_with_argmax` / `test_confidences_match_full_softmax_max` / `test_output_shapes_and_dtypes` / `test_tie_breaks_to_smallest_token_id`，2026-10-23 新鲜复跑 86 passed 之一）

**状态：** PASS

---

### ST-002：两轮收敛轨迹 rounds_to_fixpoint == 2

**关联需求：** srs.md §3.1 FR17.1 验收 2
**测试类型：** 正常路径
**优先级：** High

**前置条件：**
- Given 构造轨迹：首轮草稿部分位错误、次轮全部自一致

**测试步骤：**
1. 验证轮迭代，检查零替换判定与 rounds_to_fixpoint 记账

**期望结果：**
- Then 第 2 轮零替换且 rounds_to_fixpoint == 2；定点块末轮 changed == 0

**实际结果：** PASS（UT `TestVerifyRound::test_self_consistent_block_yields_zero_changes` + `TestDraftVerifyConvergenceDichotomy::test_real_forwards_convergence_dichotomy`：真实前向上 rtf ∈ {-1} ∪ [2, max] 二分性成立）

**状态：** PASS

---

### ST-003：永不收敛轨迹 max_rounds 强制收尾

**关联需求：** srs.md §3.1 FR17.1 验收 3
**测试类型：** 边界条件
**优先级：** High

**前置条件：**
- Given 永不收敛轨迹 + max_rounds=2（及退化 max_rounds=1）

**测试步骤：**
1. 运行调度至收尾，检查强制封存与 forced_completions 计数

**期望结果：**
- Then 强制封存完成（块内恒无 MASK，必然可封存）；forced_completions 计数正确；forced 块末轮 changed > 0

**实际结果：** PASS（UT `TestGenerateDraftVerifyPipeline::test_max_rounds_one_degrades_to_draft_only_completion` / `test_max_rounds_one_forces_every_block` / `test_default_max_rounds_forced_completions_bounded_by_blocks` + `TestDraftVerifyStatsClosedForm::test_total_rounds_and_forced_completions_parse_back`；基准 A5 forced == #{rtf==-1} 27/27 项）

**状态：** PASS

---

### ST-004：batch 元素独立收敛

**关联需求：** srs.md §3.1 FR17.1 验收 4
**测试类型：** 正常路径
**优先级：** Medium

**前置条件：**
- Given batch 内两元素收敛轮数不同（混合 prompt）

**测试步骤：**
1. 同一 batch 调度，检查逐元素决策与收敛互不干扰

**期望结果：**
- Then 各自独立收敛互不影响

**实际结果：** PASS（UT `TestVerifyRound::test_batch_elements_judged_independently` + `TestDraftVerifyBatchMixedPrompts::test_mixed_prompt_batch_completes_with_structural_stats`）

**状态：** PASS

---

### ST-005：退化输入与非法配置显式报错

**关联需求：** srs.md §3.1 FR17.1 异常处理
**测试类型：** 异常处理
**优先级：** High

**前置条件：**
- Given 空 logits / 非 3-D / 零宽块 / block_tokens 形状不符或非整型 / draft_mode 非 bool

**测试步骤：**
1. 逐一传入非法输入，检查异常类型

**期望结果：**
- Then 显式 ValueError（不崩溃、无半执行状态）

**实际结果：** PASS（UT 8 个 reject 用例：`test_reject_logits_not_3d` x2 / `test_reject_zero_width_block` x2 / `test_reject_block_tokens_not_2d` / `test_reject_block_tokens_shape_mismatch` / `test_reject_non_integer_block_tokens` + `TestDraftModeConfig::test_non_bool_draft_mode_raises_value_error`）

**状态：** PASS

---

### ST-006：draft_mode=True 管线正常完成与页驻留不变量

**关联需求：** srs.md §3.2 FR17.2 验收 1
**测试类型：** 正常路径
**优先级：** High

**前置条件：**
- Given draft_mode=True、真实 SimpleDiffusionLLM、随机 prompt（seed 固定）

**测试步骤：**
1. `generate` 全流程（多块/多规模/B=2），检查序列形状与页账本

**期望结果：**
- Then 序列形状 [B, P+G]、prompt 逐位保留、块内无 MASK 残留；轮间零页写、append 恰 P+G 各一次（`last_generate_stores` 核验）

**实际结果：** PASS（UT `TestGenerateDraftVerifyPipeline::test_draft_mode_completes_with_no_mask_residue` / `test_draft_mode_page_residency_ledger`；基准 kv_write_rows == P+G 27/27 项）

**状态：** PASS

---

### ST-007：draft_mode=False 默认路径逐位回归锚点

**关联需求：** srs.md §3.2 FR17.2 验收 2
**测试类型：** 回归测试
**优先级：** High

**前置条件：**
- Given draft_mode=False（显式与默认），与 v3.18.0 行为对照

**测试步骤：**
1. 同 prompt 同种子运行显式 False 与默认路径

**期望结果：**
- Then 与 v3.18.0 逐位一致（1128 基线零回退）

**实际结果：** PASS（UT `TestMaskInitRegressionAnchor::test_explicit_draft_mode_false_matches_default_path` / `test_draft_mode_false_generate_matches_baseline_bitwise`；T008 全量 1214 passed 含 AR016 112 用例零回归）

**状态：** PASS

---

### ST-008：fail-fast 清单全项（两模式）与零副作用

**关联需求：** srs.md §3.2 FR17.2 验收 3
**测试类型：** 异常处理
**优先级：** High

**前置条件：**
- Given 超长 / prompt 含 mask / 词表外 mask_token_id / gen_len 不整除 / 1-D / float dtype / 未加载后端

**测试步骤：**
1. 逐一传入非法输入（mask-init 与 draft-verify 两模式各覆盖）

**期望结果：**
- Then ValueError 且 `last_generate_stores is None`（零副作用）

**实际结果：** PASS（UT `TestDraftModeFailFast` 7 用例全参数化两模式）

**状态：** PASS

---

### ST-009：17 键统计契约与 acceptance 逐位核对

**关联需求：** srs.md §3.3 FR17.3 验收 1
**测试类型：** 正常路径
**优先级：** High

**前置条件：**
- Given 任一模式一次生成（含 capture 与否）

**测试步骤：**
1. 检查 stats 键集恒定与三新键语义（acceptance ∈ [0,1] 且与逐位核对一致）

**期望结果：**
- Then 17 键恒存在；两模式两臂键集一致；capture 只追加 block_logits_per_round

**实际结果：** PASS（UT `TestDraftVerifyStatsContract` 5 用例：seventeen_keys / mask_init_key_set_matches / 两臂 key_sets_match / new_keys_structure_across_round_budgets / capture_appends_only_capture_key；acceptance 闭式 `TestDraftVerifyStatsClosedForm::test_max_rounds_two_acceptance_closed_form`）

**状态：** PASS

---

### ST-010：统计闭式回代两模式各自成立

**关联需求：** srs.md §3.3 FR17.3 验收 2
**测试类型：** 边界条件
**优先级：** High

**前置条件：**
- Given 两模式同 prompt 同种子生成

**测试步骤：**
1. 回代 computed_rows == Σ W×(调度前向+封存前向)、total_rounds、forced、kv == P+G；mask-init 三新键 None 且 v3.18.0 公式保持；baseline 重算公式

**期望结果：**
- Then 各闭式关系成立（mask-init seal 恒 1 与 v3.18.0 等价；baseline 沿 Σ rounds×(L_j+W)、kv=0）

**实际结果：** PASS（UT `TestDraftVerifyStatsClosedForm` 5 用例 + `TestMaskInitStatsNewKeysNone` 2 用例 + `TestBaselineDraftVerifyStats` 3 用例）

**状态：** PASS

---

### ST-011：四臂注册断言全量判定与逐 seed 落档

**关联需求：** srs.md §3.4 FR17.4 验收 1（含 §5 约束「一致率口径不伪称 bitwise」与预注册修订 v2）
**测试类型：** 正常路径
**优先级：** High

**前置条件：**
- Given 注册配置（threshold=0.9，G∈{192,448,960}，seeds{0,1,2}，四臂）

**测试步骤：**
1. 运行 ar017_bench.py 全量（内置注册断言 v2：A1 两臂 bitwise / A2 调度整表 / A3 零回退 <=1.2x / A4 拒收 <=0.25 / A5 闭式记账）
2. 检查 bench-results.md 逐 seed 落档与诚实结论

**期望结果：**
- Then 全部注册断言按登记阈值判定 PASS（45/45）；未注册项（一致率/加速）实际值落档不隐藏；玩具持平负结果如实标注；预注册 v1→v2 修订史在案

**实际结果：** PASS（T006 全量实跑 2026-10-23：A1 9/9、A2 9/9、A3 区间 0.921-1.169x 9/9、A4 0.0424-0.0651 9/9、A5 27/27；bench-results.md §3.1 逐 seed 表 + §3.4 一致率 0.108-0.303 落档 + §3.5 诚实结论；预注册修订史 §1.3）

**状态：** PASS

---

### ST-012：复现命令可复现

**关联需求：** srs.md §3.4 FR17.4 验收 2
**测试类型：** 回归测试
**优先级：** High

**前置条件：**
- Given bench-results.md 登记的配置与命令（seed/线程/设备口径显式）

**测试步骤：**
1. 重新执行探针命令（`--probe --gen 192 --seed 0`），与落档数据对照

**期望结果：**
- Then 确定性结构值逐项一致（时间允许正常波动）

**实际结果：** PASS（2026-10-23 ST 期新鲜复跑：rounds=[8]x6 / forced=6 / accept=0.0651 / bitwise_vs_base-dv=True / schedule_match=True / rows=1728 / kv=256 / plateau=(0.734,1.0) / agree_vs_mask=0.3027——与 bench-results.md §3.1 G=192 seed=0 行及 §1 冒烟数据完全一致；耗时 4.564s/4.782s 属正常波动）

**状态：** PASS

---

### ST-013：s12 GPU 协议资产齐备可执行

**关联需求：** srs.md §3.5 FR17.5 验收 1
**测试类型：** 正常路径
**优先级：** Medium

**前置条件：**
- Given s12_draft_verify_gpu.yaml + server-experiment.md + serve_arm.py

**测试步骤：**
1. yaml 解析（execution 参数 / model_weights 预留位 / 服务器场地断言 S1-S4 注释）
2. serve_arm.py 配置装载与校验路径实跑（超长输入）
3. 检查 server-experiment.md 四臂命令/判定/记录模板齐备

**期望结果：**
- Then 资产可按文档在 GPU 服务器复现；本机不宣称 GPU 结果

**实际结果：** PASS（s12 yaml 解析通过：block_size=32 / draft_mode=false / model_weights=null 预留；serve_arm 校验路径实跑：`--prompt-len 8192 --gen-len 32` 正确 SystemExit「exceeds max_seq_len(8192)」exit=1，证明 yaml 装载/dims 解析/校验全链路可用；server-experiment.md §4.1 四臂命令、§1 S1-S4 判定、§5 记录模板齐备；随机权重预期持平口径写入 yaml 与协议）

**状态：** PASS

---

### ST-014：README 四检查点与版本 3.19.0 同步

**关联需求：** srs.md §3.6 FR17.6 验收 1
**测试类型：** 正常路径
**优先级：** High

**前置条件：**
- Given README.md 与版本四处

**测试步骤：**
1. 检查特性表 v3.19.0 行、基准节 v3.19.0 实测表、版本历史条目、复现命令行
2. 版本四处一致（pyproject / __init__ / __main__ / test_ut17）
3. 无 emoji；数据与 bench-results.md 一致；无超证据声称

**期望结果：**
- Then 四检查点齐备、数据以实跑为准、未验证内容为「机制说明/GPU 协议主张」口径

**实际结果：** PASS（ST 期脚本核验十项全 True：feature_row / bench_section / history_entry / repro_cmd 在位；pyproject / init / main / ut17 均 3.19.0；emoji_count=0；Review C3 已核 README 数据与 bench-results.md §3.2 逐项吻合、玩具持平负结果如实标注、GPU 主张明确为服务器场地断言）

**状态：** PASS

---

### ST-015：两臂 bitwise 精确性锚点（端到端）

**关联需求：** srs.md §5 约束（精确性锚点）+ design.md §6.3 精确性场景
**测试类型：** 正常路径
**优先级：** High

**前置条件：**
- Given draft-verify paged generate vs baseline draft-verify 全量重算，同种子

**测试步骤：**
1. 对比 sequences 逐位与调度统计整表（多规模）

**期望结果：**
- Then fp32 段因果时间不变性在草拟-验证调度下保持：tokens 逐位 + 整表相等

**实际结果：** PASS（UT `TestDraftVerifyTwoArmBitwiseAnchor::test_draft_verify_two_arm_bitwise_and_scheduling_anchor`（两规模）+ 基准 A1/A2 9/9 点全规模复核）

**状态：** PASS

---

### ST-016：全量测试套件零回归

**关联需求：** srs.md §4 非功能需求（兼容性：1128 基线零回退）
**测试类型：** 回归测试
**优先级：** High

**前置条件：**
- Given 版本 3.19.0 全量门禁

**测试步骤：**
1. 五件套全量实跑（black/isort/flake8/mypy/pytest 含覆盖率）

**期望结果：**
- Then >= 1128 用例、>= 84.25% 覆盖率、1128 基线零回退、四工具零告警

**实际结果：** PASS（T008 官方记录 2026-10-23：pytest 1214 passed / 1 skipped / 84.71%（1128 基线 + 86 新增零回退）；black 139 files unchanged；isort 通过；flake8 零告警；mypy Success 75 files。Review 后仅文档变更（8f7e083：design.md erratum + tasks.md 记录），无代码变更，记录有效；ST 期另作新文件 86 passed 新鲜复跑（21.71s））

**状态：** PASS

---

## 执行摘要

| 总计 | 通过 | 失败 | 阻塞 |
|------|------|------|------|
| 16 | 16 | 0 | 0 |

## ST 执行报告

| 字段 | 内容 |
|------|------|
| 执行日期 | 2026-10-23 |
| 执行结果 | PASS |
| 执行轮次 | 第 1 轮 |

### 需求覆盖矩阵

| 需求 ID | 需求描述 | 测试用例 | 结果 |
|--------|---------|---------|------|
| §3.1 FR17.1 | 草拟-验证调度器 | ST-001, ST-002, ST-003, ST-004, ST-005 | PASS |
| §3.2 FR17.2 | generate 管线 draft_mode 集成 | ST-006, ST-007, ST-008 | PASS |
| §3.3 FR17.3 | 生成统计扩展 | ST-009, ST-010 | PASS |
| §3.4 FR17.4 | 四臂基准与预注册断言 | ST-011, ST-012 | PASS |
| §3.5 FR17.5 | s12 GPU 服务器协议 | ST-013 | PASS |
| §3.6 FR17.6 | README 四检查点与版本同步 | ST-014 | PASS |

**需求覆盖率：** 6 / 6（100%）

### 测试执行汇总

| 类型 | 总计 | 通过 | 失败 | 阻塞 |
|------|------|------|------|------|
| 正常路径 | 8 | 8 | 0 | 0 |
| 边界条件 | 2 | 2 | 0 | 0 |
| 异常处理 | 2 | 2 | 0 | 0 |
| 回归测试 | 3 | 3 | 0 | 0 |
| 精确性/系统约束 | 1 | 1 | 0 | 0 |
| **合计** | **16** | **16** | **0** | **0** |

### 遗留问题

| 严重性 | 描述 | 处理方式 |
|-------|------|---------|
| Minor | design §6.1 第五分区「基准接口冒烟」以 erratum 记录（探针/全量实跑替代覆盖，Review D4，用户确认） | 已落档 design.md §6.1 erratum，不阻塞 |
| 信息 | 玩具模型上 draft-verify == mask-init 持平（随机权重无 argmax 不动点，机理结论）；性能主张 1.5x/2x/90% 为 s12 GPU 服务器场地断言，需 AR018 真实 dLLM 权重 | 已如实落档 bench-results.md §3.5 + README；AR018 接力 |

### 结论

> Go：16/16 用例 PASS、需求覆盖 6/6（100%）、无 Critical/Major/Minor 未处置缺陷（两项遗留均已落档且经用户确认）、1214 用例零回归、覆盖率 84.71% 超门槛。诚实纪律达标：玩具持平负结果如实标注，GPU 性能主张未在本机宣称。
