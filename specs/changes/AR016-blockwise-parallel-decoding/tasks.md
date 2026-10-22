# [AR016] 任务跟踪

| 字段 | 内容 |
|------|------|
| AR 编号 | AR016 |
| 关联 srs.md | ./srs.md |
| 关联 design.md | ./design.md（待生成）|
| 创建日期 | 2026-10-22 |

## 任务列表

| ID | 任务描述 | 依赖 | 状态 | 备注 |
|----|---------|------|------|------|
| T001 | ConfidenceSchedule + BlockDecodingConfig（FR16.1）：置信度并行解掩/保底 top-1/max_rounds 保底完成/配置校验；TDD：Red 子代理产 UT | - | passing | 2026-10-22 完成 |
| T002 | 模型 `forward_block`（FR16.2）：尾块连续活跃前向 + per-layer 前缀 K/V 条件 attention + 活跃行 K/V 返回；vs 全量前向 relL2<=1e-6 + K/V 逐位 | T001 | passing | 2026-10-22 完成 |
| T003 | 后端 `generate` 管线（FR16.3）：预填充 append / 逐块循环 / 封存 append / 轮间零页写 / fp32-fp16-int8 精度平面 / fail-fast 校验 | T002 | passing | 2026-10-22 完成 |
| T004 | 生成统计（FR16.4）：总量/逐块/派生键恒存在且解析一致 | T003 | passing | 2026-10-22 完成 |
| T005 | e2e 精确性（FR16.3/3.5 前置）：fp32 臂 vs baseline 环路 token 序列逐位一致 + 逐步 relL2<=1e-6 + 页写行为核验 | T003 | passing | 2026-10-22 完成（含 design 4.5 erratum） |
| T006 | 基准（FR16.5）：校准探针 -> 预注册登记 -> 四臂全量实跑 -> bench-results.md 逐 seed 落档 | T005 | passing | 2026-10-22 完成（A6 判据 v1→v3 两轮修订，见 bench-results §2.1） |
| T007 | s11 GPU yaml + server-experiment.md + README 四检查点 + 版本 3.18.0 四处同步 | T006 | passing | 2026-10-22 完成 |
| T008 | 五件套质量门禁全绿：black/isort/flake8/mypy/pytest；总用例 >= 1016；覆盖率 >= 84.25% | T007 | passing | 2026-10-22 完成（1124 passed / 84.56%） |

## 状态说明

- `pending`：待开始
- `in_progress`：进行中（当前会话）
- `passing`：开发完成，测试通过
- `failed`：测试失败，需修复

## 进度记录

> 每个开发会话结束后追加，记录完成情况。

### 2026-10-22 会话记录（T001）

- 完成任务：T001 ConfidenceSchedule + BlockDecodingConfig（FR16.1）
- 新增文件：`shadowinfer/execution/block_decoding.py`（BlockDecodingConfig 冻结校验 + ScheduleDecision + schedule_round 向量化调度）；`tests/test_block_decoding.py`（43 用例，Red 子代理产出）
- 修改文件：`shadowinfer/execution/__init__.py`（导出 BlockDecodingConfig / ScheduleDecision，design 文件结构要求）
- 测试：43 passed（含 3 规模随机黄金对照 vs 逐元素参考实现）；回归 test_macro_mode_execution.py 95 passed
- 测试缺陷修正（保留原意图，非弱化）：① 阈值边界用例原构造 conf=0.2<V=4 均匀下界 1/4，softmax max prob 数学上不可能低于 1/V，改为全均匀行+阈值 0.25 断言全解掩（判别 `>=` vs `>` 语义更强：严格大于会走保底 top-1）；② 保底并列用例原两行 argmax token 不同致 softmax 归约顺序 fp 差（0.70000005 vs 0.70000011）非真并列，改为位级相同行构造精确并列
- 备注：max_rounds 保底完成语义在 T003 generate 环路中体现（schedule_round 本身无轮概念）

### 2026-10-22 会话记录（T002）

- 完成任务：T002 模型 forward_block + 块因果 attention_mask 三层透传 + block_causal_mask（FR16.2 + design 功能点 3）
- 修改文件：`shadowinfer/models/simple_diffusion_llm.py`（forward_block 新增；forward_step/block.forward 增可选 attention_mask 默认 None；_compute_block 掩码应用+形状校验）；`shadowinfer/execution/block_decoding.py`（block_causal_mask）；`tests/test_block_decoding.py`（T002 分区 26 用例，Red 子代理产出）
- 测试：本文件 69 passed（T001 43 + T002 26）；关键证据：forward_block vs forward_step(块因果掩码) 块行 logits relL2<=1e-6 + 活跃行 K/V 逐位相等；前缀行 == 前缀-only 前向逐位（exp(-inf)=0 精确归零）；prefix_len=0 等价；两形态输入逐位一致
- 全量回归：1085 passed / 1 skipped / 84.36%（v3.17.0 基线 1016/84.25% 零回退，attention_mask=None 逐位安全）
- 实现要点：2D float 输入显式 ValueError（区别 token ids/hidden 形态，防 embed OOB 索引）；mask 与 kv_cache/reuse/mode 路径互斥显式报错（防静默忽略）

### 2026-10-22 会话记录（T003）

- 完成任务：T003 backend generate 管线（FR16.3）
- 修改文件：`shadowinfer/core/backends/pytorch_backend.py`（generate 方法 + last_generate_stores 只读属性；fresh per-call stores，不动 forward_step 服务路径状态）；`shadowinfer/execution/block_decoding.py`（GenerateResult）；`shadowinfer/execution/__init__.py`（导出）；`tests/test_block_decoding.py`（T003 分区 14 用例，Red 子代理产出）
- 测试：本文件 83 passed（T001 43 + T002 26 + T003 14）；页驻留证据：每层 appended_tokens == P+G 且 rewritten_tokens == 0 且 page_evictions == 0（轮间零页写充要）；max_rounds=1 保底完成 forced_completions >= 1；三精度平面完成；fail-fast 六类无副作用（stores None）
- 回归：backend 相关五套件 201 passed
- 实现要点：sealing pass 每块恒执行（全终态 forward_block 固化块 KV）；保底完成用最后一轮 logits argmax 填充；stats 本任务最小集（blocks/forced_completions/page_precision），完整八键集在 T004 落地

### 2026-10-22 会话记录（T004）

- 完成任务：T004 生成统计（FR16.4）
- 修改文件：`shadowinfer/core/backends/pytorch_backend.py`（generate 统计扩展：total_rounds/computed_rows_total/kv_write_rows_total/total_time_ms + 逐块 rounds_per_block/unmasked_per_round/fallback_per_round/mean_confidence_per_round + 派生 rows_saved_vs_baseline）；`tests/test_block_decoding.py`（T004 分区 12 用例，Red 子代理产出）
- 测试：本文件 95 passed（+12）；解析一致性：computed_rows == W×Σ(r+1)、kv_write_rows == P+G 且与页账本交叉验证、每块 unmasked 合计 == B×W（默认无保底）、rows_saved 闭式回代、保底路径记账（max_rounds=1）、fp16/int8 完整键集
- 统计语义决定：unmasked_per_round 为 batch 合计计数；fallback_per_round = progress_fallback.any()；mean_confidence_per_round = 该轮解掩位 softmax-max 均值；rows_saved_vs_baseline = Σ r_j×(L_j+W) − W×(r_j+1)（均不含 batch 因子，与 design 公式一致）
- 回归：test_block_decoding + test_pytorch_backend 111 passed

### 2026-10-22 会话记录（T005）

- 完成任务：T005 e2e 精确性（FR16.3 验收）+ `blockwise_generate_baseline` 参考环路实现 + `capture_block_logits` 可观测参数（generate/baseline 双侧，默认 False 逐位不变）
- **design 4.5 erratum（开发中发现的设计缺陷，已修正落档 design.md §4.5）**：原块因果掩码公式 `mask[i,j]=-inf iff i<L and j>=L` 在多块场景与方案 A「前缀 KV 时间不变」前提自相矛盾（block 1 轮次 prompt 行会看见已封存的 block 0）。T005 Red 测试的逐步 relL2 断言捕获该缺陷（block 1 恒定 5.2e-3，逐层定位：layer-0 K 逐位、layer-1 prompt K 差 0.1 = attention 上下文不同）。修正为段因果掩码 `segment_causal_mask`：`seg(i)=0 if i<P else 1+(i-P)//W`，`mask[i,j]=-inf iff seg(i)<seg(j)`——每行感受野封存后时间不变；单尾块情形与原公式逐位一致（T002 测试零改动）；期间还修正了一处调用参数误传（prefix_len 误当 prompt_len）
- 修正后实测：两臂**全轮 logits relL2 = 0.000e+00（逐位一致）**，优于 1e-6 容差预期；两规模点 (P=6,G=8) / (P=10,G=16) + max_rounds=1 保底路径 sequences 逐位一致；capture 无副作用；baseline 确定性；fp16 臂完成；baseline fail-fast 五类
- 修改文件：`shadowinfer/execution/block_decoding.py`（baseline + segment_causal_mask + validate_generate_inputs 共享校验）；`shadowinfer/core/backends/pytorch_backend.py`（generate 增 capture_block_logits + 复用共享校验）；`shadowinfer/execution/__init__.py`（导出掩码）；`tests/test_block_decoding.py`（T005 分区 13 用例，Red 子代理产出）；`specs/changes/AR016-blockwise-parallel-decoding/design.md`（§4.1/§4.2.2/§4.3 掩码口径 + §4.5 erratum）
- 测试：本文件 108 passed（95 + 13）；新增接口备注：blockwise_generate_baseline 与 generate 共享调度/校验/保底语义，stats 键集一致（computed_rows_total=Σ r_j×(L_j+W)、kv_write_rows_total=0、rows_saved_vs_baseline=0）

### 2026-10-22 会话记录（T006）

- 完成任务：T006 基准（FR16.5）：`ar016_bench.py`（注册化四臂基准脚本，断言内置）+ `bench-results.md`（校准叙事/注册表/判据修订记录/逐 seed 落档）
- **预注册修订（跑前登记）**：设计 §6.5 原定 threshold=0.9 在玩具模型上机制退化（softmax-max 置信度实际分布 0.004-0.010，0.9 下全部轮次 top-1 兜底、并行解掩零触发）→ 校准为 0.005（并行解掩 ~10-15% 轮次）+ 仅报告的阈值敏感性扫描 0.9/0.005/0.003
- **A6 判据两轮修订（全部落档 bench-results §2.1）**：v1「bitwise tokens ⟹ 对齐」被 run 1 证伪（调度置换可终态重收敛）；v2「位置级对齐」实现有 bug（baseline 漏列表 append → 空 zip 空洞真，run 2 的 A6 判定无效）且逻辑不完备（同位置 token 翻转同样破坏上下文）；v3 终版「逐轮解掩位置相等 ∧ 终态 tokens 逐位一致 ⟹ 轨迹相同」。诊断（临时脚本）定位完整机制：int8 噪声置换阈值边缘解掩顺序（同集合不同顺序），互换行自身 embedding 错位致行 relL2≈0.98、其余行 ~0.1-0.2%；轨迹对齐轮 int8 真实 logits relL2 2.0-3.6e-4（逐块剖面证实）；int8 K/V 直接往返 relL2 5.9e-3（utils/quantization.py per-channel 实现无缺陷）
- 第三次全量实跑（判据 v3）：**全部注册断言 PASS**——fp32 2.29x/3.68x/7.08x（S=256/512/1024 中位，tokens+逐轮 logits 全逐位）；fp16 relL2 ≤1.26e-5 全逐位；int8 一致率 [0.9961,1.0]、对齐点 relL2 3.6e-4；行缩减 4.9/8.4/15.4x；kv_writes 27/27 == P+G；G=192 诚实低于 3x（固定开销占比）
- 附带代码改动：generate/baseline 统计新增 `block_time_ms`（逐块耗时）与 `unmask_positions_per_round`（逐轮解掩位置日志，FR16.4「至少包含」增量键）；108 UT 回归零破坏
- 三跑交叉核对：fp32/fp16 指标一致，speedup 中位波动 <3.5%，int8 一致率区间稳定

### 2026-10-22 会话记录（T007 + T008）

- 完成任务：T007 GPU 协议/README/版本同步 + T008 五件套门禁
- T007 产物：`configs/experiments/s11_blockwise_decoding_gpu.yaml`（7B 四臂协议，阈值/块宽可配，注释含 CPU 证据口径与阈值扫描建议）；`specs/changes/AR016-blockwise-parallel-decoding/server-experiment.md`（目标判定/环境/本机基线/四臂命令/质量取证（判据 v3 口径）/显存页统计/记录模板/落档纪律）；`serve_arm.py`（单臂真实运行器，公开 API 零 mock，artifacts.pt + stats.json 落盘）
- README 四检查点：质量表 v3.18.0（1124/84.56%）；特性表新增 v3.18.0 行（含段因果 erratum、逐位精确性、加速 2.29/3.68/7.08x、int8 双态与判据修订史、S=256 诚实标注）；基准节新增 v3.18.0 实测表 + 复现命令；版本历史新增 v3.18.0 条目；TOC 无需变更（子节不入 TOC）；无 emoji
- 版本 3.18.0 四处同步：pyproject.toml / `shadowinfer/__init__.py` / `shadowinfer/__main__.py` / `tests/test_ffn_hotpath.py::test_ut17`
- T008 实测（全量）：**1124 passed / 1 skipped / 84.56%**（srs 门槛 ≥1016/≥84.25% 达标；v3.17.0 基线 1016/84.25% 零回退，净增 108 用例）；black/isort/flake8/mypy 全绿
- 修复记录：README 特性表编辑时曾误将 v3.18.0 行替换 v3.6.0 行，当场发现并恢复（v3.6.0 行完整保留）

### 2026-10-22 会话记录（Review 阶段）

- 审查方式：独立子代理（general）S/D/C 三维度审查，基线 base_sha=e264a7a，输入仅路径不含内容；mypy/black/isort/flake8 按任务声明采信 + 附加抽查
- 首轮结论：**FAIL**（2 项发现）：
  1. [Medium, S1/D2] `mask_token_id` 词表内校验缺失——srs.md FR16.1「mask_token_id 在词表内」与 design.md §4.2.2(7) fail-fast 清单均要求，但 `validate_generate_inputs` 未实现、无 UT（非法值将延迟至 embed 层 IndexError 晚爆）
  2. [Low, D2/D4] `forced_completions` 按块计数与 design §4.2.1 流程图「+= 残余数」不符且偏离未落档（两臂口径一致、SRS 未定义单位、注册数据按块口径使用，非精确性缺陷）
- 修复 1：`validate_generate_inputs` 新增区间校验 `0 <= mask_token_id < vocab_size → ValueError`（复审中按「在词表内」语义补齐负值下界）；两臂（generate / blockwise_generate_baseline）共享校验自动同步生效；新增 UT 2 例参数化（上界 GP_VOCAB / 下界 -1 × 两臂，generate 臂断言 `last_generate_stores is None` 无副作用）
- 修复 2：维持实现口径（块数），design.md §4.2.1 流程图修订为 `forced_completions += 1（按块计数）`，新增 §4.6 erratum 落档（偏离描述 + 处置 + 语义说明）
- 复审结论：**PASS**（独立子代理验证两项修复，证据齐全；附带非阻塞观察「负值未拦截」已在修复 1 中一并收口）
- 门禁复跑（修复后全量）：**1128 passed / 1 skipped / 84.56%**；black/isort/flake8/mypy 全绿；README 测试口径两处同步 1124→1128
- 审查其余维度：S2-S5 / D1 / D3 / C1-C3 全 YES（108 用例收数核实、影响面 18 文件核对、零 emoji、token 零入库、AGENTS.md 未跟踪）

## 阶段门控记录

> 由 sdd-phase-gate skill 在阶段门控审查后追加。
