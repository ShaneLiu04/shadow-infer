# [AR003] 任务跟踪

| 字段 | 内容 |
|------|------|
| AR 编号 | AR003 |
| 关联 srs.md | ./srs.md |
| 关联 design.md | ./design.md（待生成）|
| 创建日期 | 2026-10-17 |

## 任务列表

| ID | 任务描述 | 依赖 | 状态 | 备注 |
|----|---------|------|------|------|
| T001 | 基线取证落档：attention 契约双路径现状（mock=logits / 真实后端=probabilities）、双重 softmax 下精度档分布熵饱和实证、无界 importance_history 增长曲线、预算/驱逐死代码取证、旧默认参数（γ/sink/阈值）校准依据 → baseline-profile.md | - | passing | 2026-10-18 完成 |
| T002 | FR3.1 attention 契约显式化与单次归一化：配置键、orchestrator 按后端接线、标量/向量化双路径同步修复、熵饱和消除与回归锚点测试 | T001 | passing | 2026-10-18 完成（17 用例） |
| T003 | FR3.2 接收注意力重要性轴 + sink 前缀硬保护：主信号列向、rank 归一化、行熵辅助、sink_tokens 配置、双路径同步、sink 反转回归对照测试；含 FP6 公式单源化（ImportanceModel 委托共享核心、删 _history） | T002 | pending | 与 T004 增量结构协同设计 |
| T004 | FR3.3 IRAP 跨步 EMA 传播与常数内存历史：irap_decay 配置、增量 vs 全量等价 golden、无界 history 收敛为 EMA+标量统计、on_init 重置维持、形状增广规则 | T003 | pending | |
| T005 | FR3.4 预算准入与驱逐接线：store 同步 token-head 元数据、memory_pressure 消费与阈值动作、per-precision 字节估算修正（8x 误差）、sink 保护驱逐顺序、字节级断言测试 | T004 | pending | |
| T006 | FR3.5 ΔA 差分信号导出与 QDrift 可选消费：信号进 metrics/返回值、可选输入默认不融合、AR002 兼容性锚点全量回归 | T004 | pending | 融合方式 design 定 |
| T007 | 实验验证（AGENTS.md 规则 1.1 口径）：对照实验（修复前 vs 修复后重要性质量/精度分布/预算执行）、≥3 次重复取中位、多规模扫描（≥2-3 规模点）、统计/golden 证据、完整数据与复现命令落档本目录 | T005, T006 | pending | 收益声称的唯一依据 |
| T008 | 端到端验收与 AR002 口径复测：E2E 3 次中位 ≥ 1.4x 兜底、五件套全绿、覆盖率不低于 81.68%、跨步稳定性 | T007 | pending | |
| T009 | README 四检查点更新（v3.5.0：特性表/版本历史/基准数据/结构与口径）+ pyproject + __version__ + ROADMAP（IRAP 落地、AR004-AR006 规划入图） | T008 | pending | 规则一 + 规则 1.1 |
| T010 | Review（spec/design 合规审查）+ ST 验收（st-cases.md 生成与执行、Go/No-Go）+ AR 归档至 specs/archive/ | T009 | pending | |
| T011 | Gitee 推送（Runbook：与本地累积提交一并，含 69f666b TOC 提交；三核验） | T010 | pending | 规则二 |

## 状态说明

- `pending`：待开始
- `in_progress`：进行中（当前会话）
- `passing`：开发完成，测试通过
- `failed`：测试失败，需修复

## 进度记录

> 每个开发会话结束后追加，记录完成情况。

### 2026-10-18 会话记录

- 完成任务：T001 基线取证落档
- 修改文件：specs/changes/AR003-irap-signal-integrity/baseline-profile.md（新增）
- 取证要点：真实后端路径 imp 零动态范围（min=max=1.0，全 fp32）；sink token（接收 95.9% 注意力）被判 int4；历史 20 步 20 张量线性增长（投影 15.62 MB/层/千步）；预算死代码（超限 10 倍零动作）；基线 630 passed/81.68% 无回归
- 备注：基线 E2E 1.61x 引用 AR002 T008（同 HEAD 代码）

## 阶段门控记录

> 由 sdd-phase-gate skill 在阶段门控审查后追加，记录每轮审查结果（PASS/FAIL + 轮次）。格式见 sdd-phase-gate SKILL.md Step 6。

### 2026-10-17 req 门控记录

- 门控结果：PASS（第 1 轮）
- 审查项数：9 项（G1-G9；7 YES + 2 Minor WARN）
- 修复的问题：G3 两处模糊词已当场量化（FR3.1「非 fp32 档占比 ≥ 50% 且档位种类 ≥ 3」；FR3.2「档位严格高于对照组 + Spearman ≥ 0.9」）
- 审查代理：sdd-gate-reviewer

### 2026-10-17 design 门控记录

- 门控结果：PASS（第 1 轮）
- 审查项数：15 项（G1-G15；10 YES + 5 Minor WARN，无 Critical/Important NO）
- 修复的问题：全部 5 项 Minor 当场闭环——G2 srs 三处措辞对齐（驱逐复杂度/元数据接口/EMA 统计口径）；G3 共享核心与 KVM 新方法补参数表与异常契约；G13 补 UT-1b/UT-15b；G14 UT-14 补回归对照锚点；G15 补 get_importance_matrix None 分支、enforce_global_budget 返回值、ΔA 一步滞后断言、config 合法边界（β=0/1、sink_tokens=0）用例
- 审查代理：sdd-gate-reviewer


