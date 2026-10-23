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
| T002 | backend `generate` draft_mode 分支 + `blockwise_generate_baseline` draft_mode 分支（决策等价参考臂）：页驻留协同（轮间零页写、封存一次 append / 定点 KV 复用零封存前向）、进度保证、fail-fast 共存、默认路径逐位回归锚点（FR17.2） | T001 | pending | 回归锚点：draft_mode=False 与 v3.18.0 逐位 |
| T003 | 统计扩展：draft_acceptance_rate / changed_per_round / rounds_to_fixpoint；两模式 17 键集一致 + 闭式回代（FR17.3） | T002 | pending | |
| T004 | e2e 行为测试：draft-verify 正常完成（多块/多规模/B=2）、页驻留取证、batch 独立收敛、退化输入（FR17.1/17.2 验收） | T002 | pending | |
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

## 阶段门控记录

> 由 sdd-phase-gate skill 在阶段门控审查后追加。
