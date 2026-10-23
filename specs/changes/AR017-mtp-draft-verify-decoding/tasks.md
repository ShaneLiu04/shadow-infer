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
| T005 | ar017_bench.py 三臂脚本 + 校准探针（threshold 扫描）+ 预注册断言登记（FR17.4 前半） | T003, T004 | passing | 2026-10-23 完成；探针证要纯 Jacobi 不收敛 → 预注册 v1→v2 修订（用户确认，诚实口径）；四臂（+baseline-dv 锚点臂）；断言已跑前登记 bench-results.md §2 |
| T006 | 三臂全量实跑 + bench-results.md 逐 seed 落档 + 判定（FR17.4 后半） | T005 | passing | 2026-10-23 完成；四臂全量实跑，注册断言 v2 全 PASS（A1-A5 各 9/9）；玩具持平如实落档，性能主张转 s12 GPU |
| T007 | s12 yaml + server-experiment.md + serve_arm.py `--mode` 扩展 + README 四检查点 + 版本 3.19.0 四处同步（FR17.5/17.6） | T006 | passing | 2026-10-23 完成；s12 前向兼容 AR018（model_weights 配置位预留）；1214 passed / 84.71% 实测数入 README |
| T008 | 五件套质量门禁全量实跑（black/isort/flake8/mypy/pytest） | T007 | passing | 2026-10-23 完成；1214 passed / 1 skipped / 84.71%（门槛 >= 1128 用例、>= 84.25% 均超）；1128 基线零回退 |

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

### 2026-10-23 会话记录（T005）

- 完成任务：T005 基准脚本 + 校准探针 + 预注册修订与断言登记（主代理直做，探针为校准性质）
- 校准探针（临时脚本，不入库）：probe1 实测 threshold=0.9 下 mask-init 与 draft-verify 均 8 轮+1 封存=9 前向/块持平，draft-verify 全块 forced 不定点，acceptance 0.039-0.086（cap=8/16）；probe2 实测 changed 逐轮轨迹呈 62-97% 震荡平台、不衰减、块宽不敏感（W∈{8,16,32} 全 forced）——随机权重模型无稳定 argmax 不动点，纯 Jacobi 收益 contingent 于真实预训练权重（Fast-dLLM/Jacobi 文献证据全部来自预训练模型）
- **预注册修订 v1→v2（用户 2026-10-23 确认诚实口径）**：v1 门槛（1.5x/2x/90%）机理不可达；v2 注册 A1 两臂 bitwise / A2 调度整表一致 / A3 零回退界 <=1.2x / A4 坏草稿拒收 acceptance <=0.25 / A5 闭式记账（rows/forced/kv）；性能主张转入 s12 GPU 协议（服务器场地断言，需 AR018 真实权重）。修订史落档 bench-results.md §1.3 + design.md §6.5 erratum + srs.md §3.4 修订记录
- 修改文件：`ar017_bench.py`（四臂：baseline / baseline-dv / mask-init paged / draft-verify paged；probe 模式 + 注册断言 v2 内置；阈值敏感性 0.005 report-only）、`bench-results.md`（新建：探针数据 + 修订史 + 断言跑前登记）、`design.md`（§6.5 erratum）、`srs.md`（§3.4 修订记录）
- 探针冒烟（G=192 seed=0，未注册）：A1/A2 锚点成立（bitwise_vs_base-dv=True, schedule_match=True）；ratio_vs_mask=0.895x；accept=0.0651；A5 闭式全 True（kv=256=P+G）；plateau=(0.734, 1.0)；agree_vs_mask=0.3027（随机水平，report-only）

### 2026-10-23 会话记录（T006）

- 完成任务：T006 四臂全量实跑 + 逐 seed 落档 + 判定（主代理直做）
- 实跑结果：**注册断言 v2 全部 PASS（45/45 检查，A1-A5 各 9/9）**——A1 两臂 bitwise 9/9、A2 调度整表一致 9/9、A3 零回退界（区间 0.921-1.169x，中位 ~1.0x）9/9、A4 坏草稿拒收（acceptance 0.0424-0.0651）9/9、A5 闭式记账（rows/forced/kv）27/27
- 诚实结论落档 bench-results.md §3：玩具上 draft-verify 与 mask-init 端到端持平（结构预测 9=9 前向/块实测成立）、轮数无削减；根因 = 随机权重无稳定 argmax 不动点；性能主张（1.5x/2x/90%）转 s12 GPU 协议服务器场地断言（需 AR018）；mask-init vs baseline bitwise 9/9 复核（AR016 锚点 0.9 口径）；threshold=0.005 敏感性点结构同型（结论阈值无关）
- 修改文件：`bench-results.md`（§3 全量结果 + §3.5 诚实结论与 README 引用口径）

### 2026-10-23 会话记录（T007）

- 完成任务：T007 s12 GPU 协议 + serve_arm 扩展 + README 四检查点 + 版本 3.19.0 四处同步（主代理直做）
- 新增 `configs/experiments/s12_draft_verify_gpu.yaml`：四臂协议（baseline / baseline-dv / mask-init paged / draft-verify paged）；`model_weights: null` 配置位预留 AR018（本 AR 随机权重口径注明）；服务器场地断言 S1-S3（1.5x/2x/90%，仅真实权重判定，随机权重预期持平 report-only）+ S4 精确性锚点（权重无关）
- 新增 `serve_arm.py`：`--arm {baseline,fp32,fp16,int8}` + `--mode {mask-init,draft-verify}`（映射 draft_mode）；stats.json 含 mode/draft_mode 与三新键；artifacts.pt 沿 AR016 结构
- 新增 `server-experiment.md`：四臂命令、质量取证（S4 每次必查 + S1-S3 真实权重）、`rounds_to_fixpoint`/`draft_acceptance_rate`/`changed_per_round` 取证、显存与页统计、记录模板、结论落档纪律（随机权重负结果并列呈现）
- README 四检查点：特性表 v3.19.0 行（机制 + 实测 + 诚实负结果 + GPU 协议主张口径）；基准节新增 v3.19.0 实测表（四臂，dv/mask 比中位 0.997-1.039x）+ 复现命令行；版本历史 v3.19.0 条目；TOC 零改动（无新增二级标题）
- 版本 3.19.0 四处同步：pyproject.toml / `__init__.py` / `__main__.py` / `test_ffn_hotpath.py::test_ut17_version_strings_consistent`（单测通过）
- 全量实跑（T008 前置取证）：1214 passed / 1 skipped / 84.71%（1128 基线 + 86 新增零回退，覆盖率 84.56% -> 84.71%）；black/isort/flake8/mypy 全绿

### 2026-10-23 会话记录（T008）

- 完成任务：T008 五件套质量门禁全量实跑（版本 3.19.0 同步后正式记录）
- 实跑结果（全绿）：pytest **1214 passed / 1 skipped / 84.71%**（门槛 >= 1128 用例、>= 84.25% 覆盖率均超；1128 基线零回退 + 86 新增；耗时 7:44）；black 139 files unchanged；isort 通过（Skipped 22）；flake8 零告警；mypy Success（75 files）
- 修改文件：无代码变更（门禁记录任务）

## 阶段门控记录

> 由 sdd-phase-gate skill 在阶段门控审查后追加。
