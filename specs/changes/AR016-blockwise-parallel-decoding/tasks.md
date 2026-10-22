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
| T003 | 后端 `generate` 管线（FR16.3）：预填充 append / 逐块循环 / 封存 append / 轮间零页写 / fp32-fp16-int8 精度平面 / fail-fast 校验 | T002 | pending | FR16.3 |
| T004 | 生成统计（FR16.4）：总量/逐块/派生键恒存在且解析一致 | T003 | pending | FR16.4 |
| T005 | e2e 精确性（FR16.3/3.5 前置）：fp32 臂 vs baseline 环路 token 序列逐位一致 + 逐步 relL2<=1e-6 + 页写行为核验 | T003 | pending | FR16.3 |
| T006 | 基准（FR16.5）：校准探针 -> 预注册登记 -> 四臂全量实跑 -> bench-results.md 逐 seed 落档 | T005 | pending | FR16.5 |
| T007 | s11 GPU yaml + server-experiment.md + README 四检查点 + 版本 3.18.0 四处同步 | T006 | pending | FR16.6 + NFR |
| T008 | 五件套质量门禁全绿：black/isort/flake8/mypy/pytest；总用例 >= 1016；覆盖率 >= 84.25% | T007 | pending | NFR |

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

## 阶段门控记录

> 由 sdd-phase-gate skill 在阶段门控审查后追加。
