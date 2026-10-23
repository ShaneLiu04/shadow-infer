# [AR017] 任务跟踪

| 字段 | 内容 |
|------|------|
| AR 编号 | AR017 |
| 关联 srs.md | ./srs.md |
| 关联 design.md | ./design.md（待生成）|
| 创建日期 | 2026-10-23 |

## 任务列表

| ID | 任务描述 | 依赖 | 状态 | 备注 |
|----|---------|------|------|------|
| T001 | DraftVerify 调度原语：草拟轮全位填充 + 验证轮自一致替换 + 不动点检测 + 收尾；BlockDecodingConfig 增 `draft_mode` 字段与校验（FR17.1） | - | passing | 2026-10-23 完成；30 UT |
| T002 | backend `generate` draft_mode 分支 + `blockwise_generate_baseline` draft_mode 分支（决策等价参考臂）：页驻留协同（轮间零页写、封存一次 append / 定点 KV 复用零封存前向）、进度保证、fail-fast 共存、默认路径逐位回归锚点（FR17.2） | T001 | passing | 2026-10-23 完成；29 UT；seal_rounds 记账统一两模式 computed_rows 公式 |
| T003 | 统计扩展：draft_acceptance_rate / changed_per_round / rounds_to_fixpoint；两模式 17 键集一致 + 闭式回代（FR17.3） | T002 | passing | 2026-10-23 完成；18 UT；changed 口径 batch 聚合（design 4.3 已同步修订） |
| T004 | e2e 行为测试：draft-verify 正常完成（多块/多规模/B=2）、页驻留取证、batch 独立收敛、退化输入（FR17.1/17.2 验收） | T002 | passing | 2026-10-23 完成；9 UT；两臂 bitwise 锚点实测成立 |
| T005 | ar017_bench.py 三臂脚本 + 校准探针（threshold 扫描）+ 预注册断言登记（FR17.4 前半） | T003, T004 | pending | 探针不入库；断言跑前登记 bench-results.md |
| T006 | 三臂全量实跑 + bench-results.md 逐 seed 落档 + 判定（FR17.4 后半） | T005 | pending | >= 1.5x / >= 2x / >= 90% 预注册口径；未达标如实报告 |
| T007 | s12 yaml + server-experiment.md + serve_arm.py `--mode` 扩展 + README 四检查点 + 版本 3.19.0 四处同步（FR17.5/17.6） | T006 | pending | s12 前向兼容 AR018（model_weights 配置位预留） |
| T008 | 五件套质量门禁全量实跑（black/isort/flake8/mypy/pytest） | T007 | pending | 门槛：>= 1128 用例、>= 84.25%、1128 基线零回退 |

## 状态说明

- `pending`：待开始
- `in_progress`：进行中（当前会话）
- `passing`：开发完成，测试通过
- `failed`：测试失败，需修复

## 进度记录

> 每个开发会话结束后追加，记录完成情况。

### 2026-10-23 会话记录（T001）

- 完成任务：T001 DraftVerify 调度原语（Red 子代理 30 用例 → Green 主代理实现 → black 修正格式复验）
- 修改文件：`shadowinfer/execution/block_decoding.py`（+draft_mode 字段/校验；+`_validate_block_logits`/`draft_round`/`verify_round`）、`shadowinfer/execution/__init__.py`（导出两名）、`tests/test_draft_verify_decoding.py`（新增 30 用例三分区）
- 验证：新文件 30 passed；与 test_block_decoding.py 合跑 142 passed（AR016 112 用例零回归）；black/isort/flake8/mypy 全绿
- Red 证据：子代理实跑 ImportError（draft_round 不可导入），锁定 __init__ 导出契约

### 2026-10-23 会话记录（T002）

- 完成任务：T002 generate + baseline draft_mode 分支（Red 子代理 29 用例 → Green 主代理实现）
- 修改文件：`shadowinfer/core/backends/pytorch_backend.py`（generate 块内循环 draft_mode 分支 + 定点 KV 复用封存 + `seal_rounds` 记账）、`shadowinfer/execution/block_decoding.py`（baseline 块内循环 draft_mode 分支）、`tests/test_draft_verify_decoding.py`（+29 用例三分区）
- 关键实现：draft-verify 定点确认轮的 block_kv 直接用于封存 append（零额外封存前向）；forced 块（max_rounds 耗尽）补一次封存前向；`computed_rows_total`/`rows_saved_vs_baseline` 统一为 `W×(rounds+seal)` 公式（mask-init seal 恒 1，与 v3.18.0 公式等价）
- 验证：合跑 171 passed（draft_verify 59 + block_decoding 112 零回归）；divergence 锚点转绿（分支接线证明）；mask-init 回归锚点保持（draft_mode=False vs 默认路径逐位一致）；black/isort/flake8/mypy 全绿
- Red 证据：实现前 2 failed（draft_mode=True 被静默忽略、与 mask-init 输出相同）— 分支未接线的精准锚点

### 2026-10-23 会话记录（T003）

- 完成任务：T003 统计扩展（Red 子代理 18 用例 → Green 主代理实现）
- 修改文件：`shadowinfer/core/backends/pytorch_backend.py`（generate 三新键 + draft_snapshot 记账）、`shadowinfer/execution/block_decoding.py`（baseline 三新键）、`tests/test_draft_verify_decoding.py`（+18 用例四分区 + 共享断言助手）、`design.md`（4.3 表 changed_per_round 口径精化为 batch 聚合 B×W，与 unmasked_per_round 一致）
- 语义：draft_acceptance_rate = 终态 == 草稿的位比例（batch×blocks 均值）；rounds_to_fixpoint 定点块 = 确认轮调度前向数（>=2）、forced 块 = -1；changed_per_round 首项 B×W、后续 = 各验证轮替换总数；mask-init 三键恒 None；17 键契约两模式两臂一致
- 验证：合跑 189 passed（draft_verify 77 + block_decoding 112 零回归）；max_rounds=1 锚点（acceptance==1.0/全 -1/[[B×W]]×blocks/computed_rows==2·W·blocks）实测通过；black/isort/flake8/mypy 全绿
- Red 证据：实现前 18 failed（8 键集缺失 AssertionError + 10 KeyError 三新键），既有 59 用例全过

### 2026-10-23 会话记录（T004）

- 完成任务：T004 e2e 行为测试（子代理编写并实跑，9/9 一次通过——实现已在 T002/T003 交付）
- 修改文件：`tests/test_draft_verify_decoding.py`（+9 用例四分区）
- 核心证据：**draft-verify 两臂 bitwise 锚点成立**——generate（页驻留）vs blockwise_generate_baseline（全量重算）同种子 sequences 逐位一致 + rounds_to_fixpoint/changed_per_round/forced_completions/rounds_per_block/draft_acceptance_rate 整表相等（两规模 P=6/G=8、P=10/G=16）；fp32 段因果时间不变性在草拟-验证调度下保持
- 收敛二分性（真实前向）：max_rounds∈{2,3} 每块 rtf∈{-1}∪[2,max]；定点块末轮 changed==0、forced 块末轮 changed>0、forced_completions == -1 块数；capture 逐块同构（len == rounds_per_block，形状 [B,W,V]）
- 夹具陷阱记录：mask_token_id=0 兼作词表 token 0——「生成区无 mask_token_id」断言在草拟可合法产出 token 0 时种子依赖（P=10 规模 seed 2803/2841 触发）；测试种子已固化干净值（2807/2811）。「无 MASK 残留」的真正结构保证是 draft_round 的 copy_ 全填充，非 token 值观测
- 验证：全文件 86 passed；black/isort/flake8 全绿（既有 77 用例零改动）

## 阶段门控记录

> 由 sdd-phase-gate skill 在阶段门控审查后追加。
