# [AR001] 需求设计说明书

| 字段 | 内容 |
|------|------|
| AR 编号 | AR001 |
| AR 主题 | 代码深度检查、README 全方位优化与流程规则固化 |
| 关联 SR | 无（开源项目质量提升专项） |
| 日期 | 2026-10-17 |
| 状态 | Draft |

## 1. 背景与目标

ShadowInfer v3.2.2 的 README 存在数据矛盾（测试数 543/566/581 三处不一致）、版本历史错乱（v3.2.0 内容混入 v3.2.1）、目录缺章、无封面徽章等问题，直接影响开源项目的可信度与第一印象。同时代码层面存在过期 docstring（`__init__.py` 引用已删除的 ARCHITECTURE.md 等文档）、ROADMAP.md 相对链接错误等一致性缺陷。

本 AR 的目标：

1. 对代码做深度检查并全量修复发现的问题，建立可复跑的质量门禁基线；
2. 以实测数据为准全方位重塑 README（中文），配套高质量深色科技风 SVG 封面；
3. 将两条长期流程规则固化到 `AGENTS.md`：**每完成一次创新须全方位更新 README**；**每完成一个 AR 须按 Runbook 推送 Gitee**；
4. 本 AR 完成后执行首次 Gitee 推送，验证流程闭环。

## 2. 需求范围

**In Scope（本 AR 要做的）：**

- `shadowinfer/`、`tests/`、`exporter/`、`benchmarks/`、`scripts/` 包内代码问题的深度检查与全量修复
- `README.md` 全方位重塑（中文，无 emoji）
- `assets/banner.svg` 深色科技风封面横幅（新增，矢量可版本化）
- `ROADMAP.md` 链接与数据口径修复
- `AGENTS.md` 新增（含流程规则 + Gitee Push Runbook），并加入 `.gitignore` 使其**永不入库/上传**
- `.gitignore` 补充（AGENTS.md、.opencode/ 等）
- 版本历史补齐 v3.2.0，并新增本次变更条目（建议版本号 v3.3.0，待设计阶段确认）
- AR 完成后按 Runbook 推送 Gitee

**Out of Scope（本 AR 不做的）：**

- 不实现 ROADMAP 中未完成的算法/内核功能（CUDA 编译、Fast-dLLM-v2 真实适配器等）
- 不改变对外 API 行为与配置格式
- 不发布 PyPI 包、不建 arXiv 论文
- 不向 GitHub（origin）推送（用户未要求）

## 3. 功能需求

### 3.1 代码深度检查与全量修复

**描述：** 实跑五套质量门禁建立基线，人工深审核心模块，全量修复发现的问题。

**触发条件：** AR 开始时。

**期望行为：**
- 实跑 `black --check` / `isort --check-only` / `flake8` / `mypy` / `pytest`，记录基线（测试数、覆盖率、告警清单）作为文档数据的唯一事实来源；
- 深度审阅核心模块：`orchestrator.py`、`core/`、`shadowkv/`、`qdrift/`、`ffn_optimizer/`、`kernels/`、`engineering/`、`observability/`、`serving/`；
- 修复所有发现的问题：过期 docstring、死代码、逻辑缺陷、类型问题、格式问题。

**异常处理：** 若发现影响面过大的设计级缺陷（需改公开接口），记录到 ROADMAP 待办而非本 AR 内强改。

**验收标准：**
- Given 全套门禁实跑, When 修复完成, Then black/isort/flake8 零告警、mypy 无新增错误、pytest 全部通过且覆盖率不低于基线。

### 3.2 README 全方位重塑

**描述：** 以实测数据与修复后代码为准，重写 README（中文主体，无 emoji）。

**期望行为：**
- 封面横幅 + 徽章行（CI / License / Python / PyTorch / Code Style）置于文档顶部；
- 修复全部已识别问题：测试数据统一（以实跑为准）、TOC 补齐「Agent 插件、异步执行与早停」等缺失章节、版本历史补 v3.2.0 并按语义归位 v3.2.1 内容、新增本次变更条目；
- 重绘架构图（对齐、层次清晰）；性能基准章节以实测数据填充量化表格；
- 新增 Citation（BibTeX）、社区生态章节；
- 全文结构重组：项目简介→核心特性→快速开始→架构→模块详解→工程化→生态，保持中文、无 emoji。

**验收标准：**
- Given 修复后代码与实测数据, When 重写 README, Then 全部数据与实测一致、TOC 锚点与章节一一对应、无 emoji、相对链接全部有效。

### 3.3 高质量美观封面（无 emoji）

**描述：** 制作深色科技风 SVG 横幅作为 README 封面。

**期望行为：**
- 存放于 `assets/banner.svg`，矢量格式、可版本化；
- 深色（深蓝黑）渐变背景 + 科技感线条/光效 + 项目名 ShadowInfer + 副标题（面向端侧的 Diffusion LLM 推理优化框架）；
- 无任何 emoji 字符；SVG 语法合法，GitHub 渲染正常。

**验收标准：**
- Given README 顶部引用 `assets/banner.svg`, When 在 GitHub 渲染, Then 横幅完整显示（SVG 内文字元素无溢出/裁切，宽度完整呈现）、无 emoji。

### 3.4 关联文档修复

**描述：** 修复 README 之外的文档一致性问题。

**期望行为：**
- `ROADMAP.md`：`../README.md#开发贡献` 修正为 `README.md#开发贡献`；「566 passing tests」等数据口径与实测统一；
- `shadowinfer/__init__.py`：docstring 移除对已删除文档（ARCHITECTURE.md、AGENTS.md、TECHNICAL_SPEC.md、ENGINEERING.md）的引用，版本号描述与 `__version__` 一致。

**验收标准：**
- Given 修复后的文档, When 全文检索失效引用, Then 零残留。

### 3.5 AGENTS.md 流程规则固化

**描述：** 新建 `AGENTS.md`，固化两条长期流程规则与 Gitee 推送 Runbook。

**期望行为：**
- 规则一：每完成一次创新（新功能/算法/工程化改进），必须全方位更新优化 README（特性表、版本历史、基准数据、结构等）；
- 规则二：每完成一个 AR，必须按「Gitee Push 标准 Runbook（华为网关环境）」推送到 `https://gitee.com/liu-xingyan04/shadow-infer`；
- Runbook 内容适配本仓库（仓库地址、master 分支映射、账号 liu-xingyan04、代理 `http://proxy.huawei.com:8080`），保留：铁律（token 永不落盘、先诊断后归因、分批推送、三核验）、隔离干扰文件、标准推送命令、403 诊断决策树、分批推送法（每批压缩后 ≤40KB）、收尾核验、已知陷阱速查、提交信息风格；
- `AGENTS.md` 写入 `.gitignore`，永不提交、永不上传 Gitee。

**异常处理：** token 失效（401）时向用户索取新 token；网关拦截（403）按决策树处理，拆无可拆时上报用户。

**验收标准：**
- Given AGENTS.md 生成, When 执行 git add/commit, Then 该文件被 .gitignore 排除、不进入任何提交。

### 3.6 AR 完成后 Gitee 推送

**描述：** 本 AR 全部任务通过后，执行首次 Gitee 推送，验证流程闭环。

**期望行为：**
- 向用户索取私人令牌（不入库、不写进任何文件/commit/log）；
- 按	Runbook 执行：隔离干扰文件→构建提交（主题内聚的小提交）→设置代理→分批推送 `main` 到远端 `master`；
- 收尾三核验：`ls-remote` 远端 SHA == 本地、`git status` 干净、探针分支已删。

**验收标准：**
- Given AR 通过验收, When 按 Runbook 推送, Then 远端 master SHA == 本地 main SHA、AGENTS.md 不在远端仓库中。

## 4. 非功能需求

| 类型 | 指标 | 要求 |
|------|------|------|
| 质量门禁 | pytest | 全部通过，覆盖率 ≥ 基线（约 80%） |
| 质量门禁 | black / isort / flake8 | 零告警 |
| 安全 | token | 永不落盘（文档/代码/commit/config/日志） |
| 兼容性 | API | 对外接口行为不变 |
| 视觉 | README | 无 emoji、封面矢量渲染正常 |

## 5. 约束与假设

**约束：**
- README 与封面全程无 emoji；
- 文档数据以本地实跑结果为唯一事实来源；
- AGENTS.md 不上传 Gitee（通过 .gitignore 实现）；
- 推送走华为安全网关代理，单批 pack 压缩后 ≤40KB；
- Gitee 目标分支为 master，本地分支为 main，需做分支映射。

**假设：**
- 本机 Python 环境可运行全套门禁（torch 等依赖可安装）；
- Gitee 仓库 `liu-xingyan04/shadow-infer` 存在且 token 具备写权限；
- 用户会在推送阶段提供有效私人令牌。

## 6. 术语说明

| 术语 | 定义 |
|------|------|
| AR | Augmented Requirement，一次增量交付单元，过程资产存于 `specs/changes/ARxxx/` |
| 质量门禁 | black / isort / flake8 / mypy / pytest 五件套 |
| 网关尺寸门 | 华为安全网关 OversizedFiles 上传策略，单批 pack 压缩后约 >72-80KB 被拦截 |
| 三核验 | ls-remote SHA 比对、git status 干净、探针分支删除 |
