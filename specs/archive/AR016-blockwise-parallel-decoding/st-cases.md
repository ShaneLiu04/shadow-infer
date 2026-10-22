# [AR016] ST 验收测试用例

| 字段 | 内容 |
|------|------|
| AR 编号 | AR016 |
| 关联 srs.md | ./srs.md |
| 生成日期 | 2026-10-22 |

## 测试用例列表

### ST-001：置信度过阈并行解掩与 argmax 贪心

**关联需求：** srs.md §3.1 FR16.1
**测试类型：** 正常路径
**优先级：** High

**前置条件：**
- 构造 logits（部分位置高置信、部分低置信）

**测试步骤：**
1. `schedule_round` 调度
2. 核验解掩集合、保持掩码集合、解掩 token、batch 元素独立性

**期望结果：**
- 过阈位置全部解掩；未过阈保持掩码；解掩 token == argmax；阈值边界 `>=` 语义；argmax 并列取最小位

**实际结果：** PASS — `TestScheduleRoundParallelUnmasking` / `TestScheduleRoundFallbackTop1` / `TestScheduleRoundThresholdBoundary`（tests/test_block_decoding.py，112 用例文件内通过，2026-10-22）

**状态：** PASS

---

### ST-002：保底 top-1 进度保证与 max_rounds 保底完成

**关联需求：** srs.md §3.1 FR16.1
**测试类型：** 正常路径
**优先级：** High

**前置条件：**
- 全低置信 logits；max_rounds=1 配置

**测试步骤：**
1. 全低置信首轮调度
2. max_rounds 耗尽收尾
3. 核验 forced_completions 记账（单位=块，design §4.6 erratum 口径）

**期望结果：**
- 恰好解掩 top-1（不死循环）；残余位全部强制解掩；forced_completions 计数正确

**实际结果：** PASS — 保底 top-1 首轮触发 UT、`TestGenerateForcedCompletion`（max_rounds=1 强制收尾）；单调解掩（已解掩不再解掩）UT 通过

**状态：** PASS

---

### ST-003：配置校验显式 ValueError

**关联需求：** srs.md §3.1 FR16.1
**测试类型：** 异常处理
**优先级：** High

**前置条件：**
- 非法配置组合

**测试步骤：**
1. `block_size < 1`、`max_rounds < 1`、`confidence_threshold` 出 (0,1]、`page_size < 1`、非法 `page_precision` 分别构造 Config
2. `mask_token_id` 出词表（>= vocab / < 0）经 generate 与 baseline 调用

**期望结果：**
- 全部显式 `ValueError`（fail-fast，不延迟到 embed 层）

**实际结果：** PASS — 配置校验 12 非法 + 7 合法边界参数化 UT；词表区间校验（Review 修复新增，`0 <= mask_token_id < vocab_size`）2 参数化 × 2 臂 UT（generate 臂断言 `last_generate_stores is None` 无副作用）；合规法边界（== vocab-1）由全量正常路径用例覆盖（mask_token_id=0 合法）

**状态：** PASS

---

### ST-004：forward_block vs 掩码 forward_step 精确性

**关联需求：** srs.md §3.2 FR16.2
**测试类型：** 正常路径
**优先级：** High

**前置条件：**
- 同一上下文（前缀 L 行 + 尾块 W 行），fp32

**测试步骤：**
1. `forward_block` vs `forward_step`（段因果掩码）
2. 核验块行 logits relL2 与活跃行 K/V

**期望结果：**
- 块行 logits relL2 <= 1e-6；活跃行 K/V 逐位相等；掩码与 kv_cache 组合显式拒绝

**实际结果：** PASS — `TestForwardBlockMatchesMaskedForwardStep`（relL2 实测 0.0 逐位，段因果掩码口径）；掩码组合 ValueError UT 通过

**状态：** PASS

---

### ST-005：prefix_len=0 与 attention_mask=None 回归

**关联需求：** srs.md §3.2 FR16.2
**测试类型：** 回归测试
**优先级：** High

**前置条件：**
- 首块无前缀；v3.17.0 默认路径锚点

**测试步骤：**
1. `prefix_len=0` 的 forward_block vs 全量前向
2. `attention_mask=None` 路径 vs v3.17.0 输出

**期望结果：**
- prefix_len=0 与全量前向一致；attention_mask=None 逐位回归（既有 forward_step 语义不变）

**实际结果：** PASS — `TestForwardBlockNoPrefix`；`TestAttentionMaskNoneRegression`（逐位 `torch.equal`）；全量套件 1128 passed 含 v3.17.0 全部基线用例零回退

**状态：** PASS

---

### ST-006：forward_block 形状校验

**关联需求：** srs.md §3.2 FR16.2
**测试类型：** 异常处理
**优先级：** Medium

**前置条件：**
- prefix_kv 层数/形状、x_block 长度不匹配输入

**测试步骤：**
1. 层数不符 / K/V 形状不符 / 长度不符分别调用

**期望结果：**
- 显式 `ValueError`

**实际结果：** PASS — 形状/层数/hidden 维不符三类 ValueError UT 通过（tests/test_block_decoding.py T002 分区）

**状态：** PASS

---

### ST-007：fp32 臂 generate vs baseline 逐位一致

**关联需求：** srs.md §3.3 FR16.3
**测试类型：** 正常路径
**优先级：** High

**前置条件：**
- 真实模型、随机 prompt（seed 固定）、fp32 页、多块多规模

**测试步骤：**
1. `generate` vs `blockwise_generate_baseline`（同调度同种子）
2. 核验 sequences 逐位、逐轮块行 logits relL2

**期望结果：**
- token 序列逐位一致；逐步 logits relL2 <= 1e-6；prompt 部分逐位保留；含保底完成路径

**实际结果：** PASS — T005 e2e：多块两规模（P=6/G=8、P=10/G=16）、B=2、fp32 逐位一致（relL2 实测 0.0）；基准规模（S=256/512/1024）tokens+逐轮 logits 全逐位（relL2 <= 3.25e-7，bench-results.md §3）

**状态：** PASS

---

### ST-008：零轮间页写与页驻留行为

**关联需求：** srs.md §3.3 FR16.3
**测试类型：** 正常路径
**优先级：** High

**前置条件：**
- paged 前缀驻留开启

**测试步骤：**
1. 生成全程检查 store 账本（公开只读 `last_generate_stores`）
2. 核验 append 次数、轮间页 id/页内容

**期望结果：**
- 预填充/封存各 append 一次；轮间零页写（页数与页 id 不变）；`appended_tokens == P + gen_len`

**实际结果：** PASS — 页驻留取证 UT（appended/rewritten/evictions 账本、轮间 page_count 与页 id 不变、`kv_writes == P+G`）；基准 27/27 组 kv_writes == P+G（bench-results.md A 类断言）

**状态：** PASS

---

### ST-009：fp16/int8 页精度权衡臂

**关联需求：** srs.md §3.3 FR16.3
**测试类型：** 正常路径
**优先级：** Medium

**前置条件：**
- `page_precision` fp16 / int8

**测试步骤：**
1. generate 全流程
2. 核验完成性与统计精度标识

**期望结果：**
- 正常完成；统计含精度标识；偏差由 FR16.5 基准量化

**实际结果：** PASS — fp16/int8 全流程完成 UT + 统计 `page_precision` 标识；基准实测：fp16 relL2 <= 1.26e-5 全逐位、int8 一致率 [0.9961, 1.0]（bench-results.md §3 落档）

**状态：** PASS

---

### ST-010：generate fail-fast 无副作用

**关联需求：** srs.md §3.3 FR16.3
**测试类型：** 异常处理
**优先级：** High

**前置条件：**
- 非法输入组合

**测试步骤：**
1. 超长 / prompt 含 mask / gen_len 不整除 / 1-D / float dtype / 未加载 backend 分别调用

**期望结果：**
- `ValueError`（带阶段诊断）；`last_generate_stores is None`（零副作用）

**实际结果：** PASS — `TestGenerateFailFast` 六类用例（含 `test_unloaded_backend_raises_runtime_error`），全部断言无副作用；`TestT005BaselineFailFast` 同口径覆盖 baseline 臂

**状态：** PASS

---

### ST-011：生成统计契约与闭式回代

**关联需求：** srs.md §3.4 FR16.4
**测试类型：** 正常路径
**优先级：** High

**前置条件：**
- 一次完整生成

**测试步骤：**
1. 检查 stats 全部键存在
2. 闭式回代：`computed_rows == Σ W×(rounds+1)`（含 sealing）、`kv_write_rows == P+G`、`rows_saved_vs_baseline`、`rounds_per_block`/`unmasked_per_round`/`mean_confidence_per_round` 与解析预期

**期望结果：**
- 键恒存在；数值与解析预期一致；两臂键集一致

**实际结果：** PASS — `_gp_assert_stats_structure` 十四键契约 + 闭式回代 UT（12 统计契约用例）；两臂 stats 键集一致（baseline computed_rows_total=Σ r_j×(L_j+W)、kv_write_rows_total=0、rows_saved_vs_baseline=0 语义落档）

**状态：** PASS

---

### ST-012：四臂基准注册断言与逐 seed 落档

**关联需求：** srs.md §3.5 FR16.5
**测试类型：** 正常路径
**优先级：** High

**前置条件：**
- `ar016_bench.py` 注册化四臂配置；断言跑前登记（bench-results.md §2）；校准探针先行（§1）

**测试步骤：**
1. 全量 bench 执行（四臂 × G∈{192,448,960} × seeds{0,1,2}，threshold=0.005，W=32，P=64，B=2，单线程 CPU）
2. 核验注册断言 A1-A9 与逐 seed 落档

**期望结果：**
- fp32 token 逐位 + relL2 <= 1e-6；加速与 KV 写削减按登记口径判定；未达标如实报告

**实际结果：** PASS — 2026-10-22 第三次全量实跑（判据 v3）：全部注册断言 PASS——fp32 加速 2.29x/3.68x/7.08x（S=256/512/1024 中位）、fp16 relL2 <= 1.26e-5 全逐位、int8 一致率 [0.9961,1.0]、行缩减 4.9/8.4/15.4x、kv_writes 27/27 == P+G；S=256 端到端 < 3x 诚实低于预注册目标（如实落档不构成 FAIL，属报告口径）；判据三轮修订史（v1 代理证伪/v2 实现bug无效/v3 终版）与阈值 0.9→0.005 校准史完整落档 bench-results.md §1/§2.1/§3

**状态：** PASS

---

### ST-013：s11 GPU 协议与服务器复现资产

**关联需求：** srs.md §3.6 FR16.6
**测试类型：** 正常路径
**优先级：** Medium

**前置条件：**
- 无

**测试步骤：**
1. 核验 `configs/experiments/s11_blockwise_decoding_gpu.yaml`、`server-experiment.md`、`serve_arm.py` 存在与内容完备性

**期望结果：**
- 7B 四臂协议（维度注释/阈值块宽可配/复现命令/判定标准/记录模板）齐备；本机不宣称 GPU 结果

**实际结果：** PASS — 五资产 Test-Path 实核验全存在（2026-10-22）；s11 含四臂/阈值扫描 {0.9,0.8,0.6} 建议/维度注释；server-experiment 含环境/本机基线/四臂命令/质量取证（判据 v3 口径）/显存页统计/记录模板；serve_arm 单臂运行器公开 API 零 mock

**状态：** PASS

---

### ST-014：五件套质量门禁

**关联需求：** srs.md §4 NFR（质量门禁）
**测试类型：** 回归测试
**优先级：** High

**前置条件：**
- 无

**测试步骤：**
1. black/isort/flake8/mypy/pytest 全量执行

**期望结果：**
- 五件套全绿；覆盖率 >= 84.25%；总用例 >= 1016；v3.17.0 基线 1016/84.25% 零回退

**实际结果：** PASS — black 133 文件 unchanged / isort 通过 / flake8 通过（--max-line-length=100）/ mypy 75 文件 no issues / pytest **1128 passed / 1 skipped，覆盖率 84.56%**（Review 修复后最终状态实跑，2026-10-22；门槛超基线 +112 用例 +0.31pp）

**状态：** PASS

---

### ST-015：版本 3.18.0 四处同步

**关联需求：** srs.md §4 NFR（版本同步）
**测试类型：** 正常路径
**优先级：** Medium

**前置条件：**
- 无

**测试步骤：**
1. 核验 pyproject.toml / `__init__.py` / `__main__.py` / `test_ffn_hotpath.test_ut17` 版本一致

**期望结果：**
- 四处均 == 3.18.0

**实际结果：** PASS — 四处实核验均 3.18.0（pyproject `version = "3.18.0"`、两处 `__version__ = "3.18.0"`、test_ut17 三处一致性断言，2026-10-22）

**状态：** PASS

---

### ST-016：README 四检查点与诚实纪律

**关联需求：** srs.md §3.6 FR16.6 + §4 NFR（文档/性能主张纪律）
**测试类型：** 正常路径
**优先级：** High

**前置条件：**
- 无

**测试步骤：**
1. 核验特性表 v3.18.0 条目、质量表、基准节、版本历史、TOC、emoji
2. 核验优越性声称可溯源至 bench-results.md 实跑

**期望结果：**
- 四检查点齐备；数据以实跑为准；无 emoji；无超证据声称；GPU 收益只给协议不宣称数字；S=256 低于 3x 诚实标注

**实际结果：** PASS — 特性表 v3.18.0 行（含段因果 erratum/判据修订史/int8 双态/S=256 诚实标注）、质量表 1128/84.56%（Review 修复后同步）、基准节 v3.18.0 实测表 + 复现命令（archive 路径）、版本历史 v3.18.0；emoji 扫描 clean（Review C3 核验）；加速数字全部溯源 bench-results.md §3 逐 seed 落档

**状态：** PASS

---

## 执行摘要

| 总计 | 通过 | 失败 | 阻塞 |
|------|------|------|------|
| 16 | 16 | 0 | 0 |

## ST 执行报告

| 字段 | 内容 |
|------|------|
| 执行日期 | 2026-10-22 |
| 执行结果 | PASS |
| 执行轮次 | 第 1 轮 |

### 需求覆盖矩阵

| 需求 ID | 需求描述 | 测试用例 | 结果 |
|--------|---------|---------|------|
| §3.1 FR16.1 | 置信度感知并行解掩调度 | ST-001, ST-002, ST-003 | PASS |
| §3.2 FR16.2 | 模型尾块活跃前向 forward_block | ST-004, ST-005, ST-006 | PASS |
| §3.3 FR16.3 | generate 管线与跨块前缀 KV 驻留 | ST-007, ST-008, ST-009, ST-010 | PASS |
| §3.4 FR16.4 | 生成统计与可观测 | ST-011 | PASS |
| §3.5 FR16.5 | 基准、精确性与预注册断言 | ST-012 | PASS |
| §3.6 FR16.6 | GPU 协议、README 与版本同步 | ST-013, ST-016 | PASS |
| §4 NFR | 五件套门禁 / 版本同步 | ST-014, ST-015 | PASS |

**需求覆盖率：** 7 / 7（100%）

### 测试执行汇总

| 类型 | 总计 | 通过 | 失败 | 阻塞 |
|------|------|------|------|------|
| 正常路径 | 9 | 9 | 0 | 0 |
| 边界条件 | 0 | 0 | 0 | 0 |
| 异常处理 | 3 | 3 | 0 | 0 |
| 回归测试 | 4 | 4 | 0 | 0 |
| **合计** | **16** | **16** | **0** | **0** |

### 遗留问题

| 严重性 | 描述 | 处理方式 |
|-------|------|---------|
| Minor | srs §1.2 预注册目标「端到端 >= 3x」在 S=256 未达（实测 2.29x）；S=512/1024 达标（3.68x/7.08x）。属如实报告口径非缺陷（FR16.5 验收允许未达标如实落档） | 已落档 bench-results.md §3 + README 诚实标注，关闭 |
| Minor | int8 臂轨迹分叉轮 relL2 0.17-0.21（调度置换伪影，对齐轮 2.0-3.6e-4）机理已诊断落档；GPU 7B 侧按 s11 协议复核 | 已落档 bench-results.md §2.1，服务器复核候选 |
| Minor | AR015 遗留三项（tasks.md UT 备注合计差 1 / last_attention 跨层共享 / test_m1_vectorization 偶发抖动）沿 AR015 验收口径延后 | 延后处理（独立 AR 候选） |

### 结论

> **Go 建议**。全部 16 ST 用例 PASS（100% 需求覆盖）；无 Critical/Major 缺陷；五件套门禁全绿（1128 passed / 84.56%，超 v3.17.0 基线零回退）；Review 首轮 FAIL 两项发现已修复并复审 PASS；预注册纪律合规（判据三轮修订史 + S=256 未达标诚实落档）；3 项 Minor 均已落档或延后，不阻断归档。
