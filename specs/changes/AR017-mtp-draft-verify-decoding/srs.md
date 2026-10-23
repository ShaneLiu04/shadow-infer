# [AR017] 需求设计说明书：块级多 token 草拟-验证并行解码（MTP Draft-Verify）

| 字段 | 内容 |
|------|------|
| AR 编号 | AR017 |
| AR 主题 | 块级草拟-验证解码：round-0 全位并行草拟 + 自一致迭代收敛（Fast-dLLM MTP 对标） |
| 关联 SR | ROADMAP.md §4 创新提案 #5（Speculative Decoding for Diffusion LLM）；承接 AR016 |
| 日期 | 2026-10-23 |
| 状态 | Draft |

## 1. 背景与目标

### 1.1 痛点与机会（代码级与 SOTA 实证）

- **AR016 实测已证明：块内多轮精修是端到端延迟的主要驱动**。AR016 管线（v3.18.0）每块从全 MASK 初始化，只有过阈位置才解掩；真实 dLLM 置信度口径（threshold=0.9）下，玩具模型实测退化为逐轮 top-1 保底（每块最多 `max_rounds=8` 轮）。端到端加速 2.29-7.08x 的主要来源正是「baseline 每轮重算全上下文 vs 优化臂每轮只算 W 行」——轮数越多节省越大，但轮数本身就是待削减的浪费。
- **SOTA 空白**：Fast-dLLM（NVIDIA/MIT/HKU，LLaDA 27.6x）三件套中，本仓库已覆盖两项——块式 KV 缓存（AR016 更强：精确前缀 KV，零近似零漂移 vs Fast-dLLM 的 dual-cache 近似）与置信度感知并行解码（AR016 ConfidenceSchedule）。**唯一未覆盖的是 Multi-Token Prediction（MTP）**：块首单次前向并行草拟全部 token，以草稿代替全 MASK 初始化，后续验证轮大幅收敛。这是 Fast-dLLM 加速的关键组成。
- **机理对标**：MTP 草拟 + 验证等价于块内 Jacobi 式并行迭代（"Accelerating Transformer Inference via Jacobi Iteration", Ye et al.）：round-0 前向对全 MASK 块取各位 argmax 得完整草稿（草拟），随后每轮前向校验各位 argmax 与当前 token 的自一致性（不匹配位替换），直至不动点。块内轮数从「逐轮 top-1 的 O(W)」塌缩至「自一致收敛的 O(1)-O(3)」。
- **与精确前缀 KV 的协同**：草拟与验证轮全部复用 AR016 的页驻留前缀（轮间零页写、封存一次 append），MTP 只改块内初始化与调度语义，执行平面零新增写放大。

### 1.2 目标

在 AR016 块式半 AR 管线上新增**草拟-验证解码模式**（`draft_mode`，默认关闭）：

- **草拟**：块 round-0 前向（全 MASK 条件）→ 全部 W 位一次性以 argmax 填充（并行草稿）；
- **验证**：每轮前向计算各位 logits → `argmax == 当前 token` 的位置自一致（接受），不匹配位置以新 argmax 替换（原地重拟）→ 至不动点（零替换位）或 `max_rounds` 收尾；
- **进度保证**：草拟轮后块内恒无 MASK（全部有 token），封存必然可完成；不动点或收尾即封存；
- **统计**：`rounds_to_fixpoint`、`draft_acceptance_rate`（首轮草稿被最终保留比例）、`changed_per_round` 等新键，与 AR016 键集对齐（两模式键集一致，缺省值语义明确）；
- **性能**：多轮场景（threshold=0.9 口径）端到端 >= 1.5x vs AR016 mask-init 臂；每块轮数削减 >= 2x；token 一致率 >= 90%（与 mask-init 臂对比，一致率口径不伪称 bitwise）；
- **复现**：s12 GPU 协议（7B 四臂叙事，前向兼容 AR018 真实 dLLM 权重）+ server-experiment.md + README 四检查点 + 版本 3.19.0 四处同步。

### 1.3 SOTA 对照与差异化

| 项目 | 机制 | 本 AR 差异 |
|---|---|---|
| Fast-dLLM MTP | 块首全位草拟 + 置信度验证 | 同语义；验证采用自一致迭代（Jacobi 不动点），收敛判据显式可测；统计/预注册纪律 |
| Jacobi/Lookahead 解码（AR-LM） | 逐位并行定点迭代 | 移植到 dLLM 块内（MASK 条件草拟是 dLLM 特有优势：单次前向即得全块草稿） |
| AR016 mask-init（本仓库） | 过阈解掩 + 保底 top-1 | 保留为默认模式；draft-verify 为 opt-in；两臂统计键集一致可对照 |
| LLaDA 退火重掩码 | 低置信重掩码迭代 | Out of scope（本 AR 单调替换，不重掩码） |

## 2. 需求范围

| 项 | 内容 |
|---|---|
| AR 编号 | AR017 |
| 版本 | v3.19.0 |
| 影响面 | `shadowinfer/execution/block_decoding.py`（调度/草拟/统计扩展）、`shadowinfer/core/backends/pytorch_backend.py`（generate draft_mode 分支）、`shadowinfer/execution/__init__.py`（导出）、`tests/test_draft_verify_decoding.py`（新增）、`specs/changes/AR017-mtp-draft-verify-decoding/ar017_bench.py`、`configs/experiments/s12_draft_verify_gpu.yaml`、README.md、版本号同步 |
| 不破坏对外 API | `draft_mode` 为 `BlockDecodingConfig` 新增可选字段（默认 False）；既有 `generate`/`forward_block`/`forward_step` 签名不变；默认路径逐位回归（1128 基线） |

**In Scope（本 AR 要做的）：**
- 草拟-验证调度器（round-0 全位草拟、自一致迭代、不动点检测、max_rounds 收尾）；
- `generate` 管线 draft_mode 分支（页驻留协同：轮间零页写、封存一次 append）；
- 统计扩展（两模式键集一致 + 新键语义）；
- 三臂基准（baseline 全量重算 / mask-init / draft-verify）+ 校准探针 + 预注册断言 + 逐 seed 落档；
- s12 GPU 协议 + server-experiment.md（前向兼容 AR018 真实权重）+ serve_arm 扩展；
- README 四检查点 + 版本 3.19.0 四处同步。

**Out of Scope（本 AR 不做的）：**
- 真实预训练 dLLM 权重加载（拆分为 **AR018**，s12 协议前向兼容）；
- 跨块前瞻流水（dLLM-serve overlap 调度：块 j 处理中草拟块 j+1 的调度重叠，未来 AR）；
- 低置信重掩码退火（LLaDA 式）；置信度门控验证（confidence-gated accept）探针后若收益明确另落 AR；
- CUDA kernel / fused attention；
- HF 后端 / vLLM / SGLang 接线。

## 3. 功能需求

### 3.1 FR17.1 草拟-验证调度器（DraftVerifySchedule）

**描述：** `block_decoding.py` 新增草拟-验证调度原语，输入块内 logits 与当前块 token 状态，输出替换决策。

**触发条件：** `draft_mode=True` 时每轮前向后调用。

**期望行为：**
- **草拟轮（round 0）**：块全 MASK 初始化，前向后对全部 W 位取 argmax，整块一次性填充；
- **验证轮（round >= 1）**：对每位计算 `argmax(logits) == current_token`；自一致位保留；不匹配位以新 argmax 原地替换；
- **收敛判定**：某验证轮替换位数为 0 → 不动点，块完成（`rounds_to_fixpoint` 记录）；
- **收尾**：达 `max_rounds` 仍未不动点 → 以当前状态强制封存（块内恒无 MASK，必然可封存），计入 `forced_completions`（单位=块，沿 AR016 §4.6 erratum 口径）；
- batch 元素独立（互不影响）；替换决策确定性（并列取最小位，沿 AR016 惯例）。

**异常处理：** 空 logits / 零长度块等退化输入显式报错（沿 AR016 FR16.1 口径）。

**验收标准：**
- Given 全 MASK 块首轮 logits，When 草拟，Then 全部 W 位以 argmax 填充且无 MASK 残留；
- Given 构造两轮收敛轨迹（首轮草稿部分位错误、次轮全部自一致），When 验证迭代，Then 第 2 轮零替换且 `rounds_to_fixpoint == 2`；
- Given 永不收敛轨迹 + max_rounds=2，When 收尾，Then 强制封存且 `forced_completions` 计数正确；
- Given batch 内两元素收敛轮数不同，When 调度，Then 各自独立收敛互不干扰。

### 3.2 FR17.2 generate 管线 draft_mode 集成

**描述：** `PyTorchModelBackend.generate` 接入草拟-验证模式，与页驻留执行平面协同。

**触发条件：** `BlockDecodingConfig(draft_mode=True)` 调用 `generate`。

**期望行为：**
- 管线骨架沿 AR016：预填充 → 逐块循环 → 封存 append → 下一块；draft_mode 只改块内调度语义（草拟-验证 vs mask-init）；
- **页驻留不变量保持**：轮间零页写；页 append 仅预填充与封存各一次（`appended_tokens == P + G`）；
- **进度保证**：草拟轮后块内无 MASK；任何轮次退出后序列形状完整 `[B, P + gen_len]`；
- `draft_mode=False`（默认）路径与 v3.18.0 逐位一致（既有 UT 全部零改动通过）；
- `GenerateResult` 结构不变；prompt 部分逐位保留；
- 配置校验：`draft_mode` 为 bool；与 AR016 全部既有校验共存（`mask_token_id` 词表区间、长度整除等）。

**异常处理：** 沿 AR016 fail-fast 清单（超长/prompt 含 mask/词表外 mask_token_id/gen_len 不整除/非法 precision → `ValueError` 且零副作用）。

**验收标准：**
- Given draft_mode=True、真实模型、随机 prompt，When generate，Then 正常完成、序列形状正确、页驻留不变量成立（轮间零页写、append P+G 各一次）；
- Given draft_mode=False，When generate，Then 与 v3.18.0 输出逐位一致（回归锚点）；
- Given 非法输入，When generate（任一模式），Then `ValueError` 且 `last_generate_stores is None`。

### 3.3 FR17.3 生成统计扩展

**描述：** draft-verify 模式统计，与 AR016 十四键集对齐。

**期望行为：** `GenerateResult.stats` 在 AR016 十四键基础上新增：
- `draft_acceptance_rate`：各块首轮草稿 token 被最终封存保留的位比例（batch 总均值）；
- `changed_per_round`：逐块逐轮替换位数（嵌套列表）；
- `rounds_to_fixpoint`：逐块收敛轮数（未不动点则记 max_rounds）；
- mask-init 模式下新键以缺省语义存在（`draft_acceptance_rate=None` 或等价哨兵），**两模式键集一致**；
- 既有键语义零改动（`forced_completions` 单位=块；`computed_rows_total == Σ W×(轮数+1)` 含封存沿 AR016 口径）。

**验收标准：**
- Given 任一模式一次生成，When 检查统计，Then 键集恒定且数值与解析预期一致（17 键契约；draft-verify 模式 `draft_acceptance_rate ∈ [0,1]` 且与逐位核对一致）；
- Given 两模式同 prompt 同种子，When 对比统计，Then 既有闭式回代关系各自成立。

### 3.4 FR17.4 三臂基准与预注册断言（ar017_bench.py）

**描述：** 真实模型三臂基准 + 校准探针先行 + 预注册。

**期望行为：**
- 三臂：**baseline**（全量重算，沿 AR016）/ **mask-init**（AR016 优化管线）/ **draft-verify**（本 AR）；真实 `SimpleDiffusionLLM`、真实随机 prompt（seed 固定）、零 mock；
- 规模：沿 AR016 口径（G∈{192,448,960}，P=64，B=2，W=32，3 seeds，单线程 CPU）；**threshold=0.9 主口径**（多轮场景，MTP 收益的机制所在）+ 探针阈值扫描（校准史落档）；
- 指标：端到端时间、每块轮数（`rounds_per_block`/`rounds_to_fixpoint`）、`draft_acceptance_rate`、draft-verify vs mask-init token 一致率、vs baseline 一致率、computed_rows、kv_write_rows；
- **校准先行**（AR015/016 纪律）：正式断言登记前跑校准探针（不入库），按实测固化阈值（裕量 >= 20%），断言集跑前登记于 bench-results.md；
- 预注册门槛（用户确认口径，探针后修订需落档）：draft-verify vs mask-init 端到端 **>= 1.5x**（threshold=0.9）、每块轮数削减 **>= 2x**、token 一致率 **>= 90%**；未达标如实报告（三轮预注册历史模式沿用）。

**验收标准：**
- Given 三臂全量实跑，When 按登记口径判定，Then 全部注册断言按登记阈值判定并逐 seed 落档（未达标如实标注不构成隐藏）；
- Given 逐 seed 原始数值与配置，When 复现命令执行，Then 结果可复现（seed/线程/设备口径显式）。

### 3.5 FR17.5 s12 GPU 服务器协议

**描述：** `configs/experiments/s12_draft_verify_gpu.yaml` + `server-experiment.md` 更新 + `serve_arm.py` 扩展。

**期望行为：**
- s12：7B 三臂协议（LLaDA/Dream 叙事），`draft_mode`/`max_rounds`/阈值/块宽可配，维度注释与 `--seed` 复现命令；**前向兼容 AR018**（真实权重加载参数位预留：`model_weights` 配置位，本 AR 以随机权重口径注明）；
- server-experiment.md：三臂命令、质量取证（token 一致率、rounds_to_fixpoint 分布、draft_acceptance_rate）、显存与页统计、记录模板、判定标准、结论落档纪律；
- serve_arm.py：支持 `--mode {mask-init,draft-verify}`（baseline 臂已有），公开 API 零 mock。

**验收标准：**
- Given s12 yaml + server-experiment.md + serve_arm.py，Then 可按文档在 GPU 服务器复现（维度、命令、判定、模板齐备）；本机不宣称 GPU 结果。

### 3.6 FR17.6 README 四检查点与版本同步

**描述：** README 四检查点更新 + 版本 3.19.0 四处同步。

**期望行为：**
- 特性表 v3.19.0 行（MTP 草拟-验证、轮数削减实测、一致率口径诚实标注、与 Fast-dLLM/Jacobi 对标说明）；
- 质量表实测数、基准节 v3.19.0 实测表 + 复现命令、版本历史 v3.19.0；TOC 零残留；无 emoji；
- 版本 3.19.0 同步 `pyproject.toml`、`__init__.py`、`__main__.py`、`test_ffn_hotpath.test_ut17`。

**验收标准：**
- Given README，Then 四检查点齐备、数据以实跑为准、无超证据声称（未验证只给「机制说明（未验证）」口径）。

## 4. 非功能需求

| 类型 | 指标 | 要求 |
|------|------|------|
| 兼容性 | 新增 API 不破坏既有 | `draft_mode` 默认 False；forward_step/forward_block/generate 签名不变；全量测试逐位回归（1128 基线不回退） |
| 精确性口径 | 一致率 + 实测 | draft-verify vs mask-init 为一致率口径（>= 90% 预注册），**不伪称 bitwise**；vs baseline 同口径报告 |
| 性能主张纪律 | README 优越性数字 | 必须来自本 AR 实跑（三臂对照/多规模/3 seeds/实档）；未验证只能「机制说明（未验证）」口径 |
| 质量门禁 | 五件套 | black/isort/flake8/mypy/pytest 全绿；覆盖率 >= 84.25%（v3.18.0 基线 84.56% 不回退）；总用例 >= 1128 |
| 真实性 | 零 mock | 草拟/验证全部真实前向与真实页存储；无 monkeypatch 伪造数据路径（故障注入仅限异常测试） |
| 版本同步 | 3.19.0 | pyproject.toml、`__init__.py`、`__main__.py`、`test_ffn_hotpath.test_ut17` 四处 |
| 文档 | README 四检查点 | 特性表、版本历史 3.19.0、基准表实测行、TOC 零残留、无 emoji |

## 5. 约束与假设

**约束：**
- 一致率口径替代 bitwise 是本 AR 与 AR016 验收纪律的关键差异：草拟改变块内轨迹（草稿起步 vs 全 MASK 起步的收敛不动点不同），任何 bitwise 声称均不成立，README/基准只报实测一致率；
- `draft_mode=False` 默认路径必须与 v3.18.0 逐位一致（回归锚点，1128 用例零改动通过）；
- 轮数削减收益的前提是 mask-init 臂处于多轮场景（threshold=0.9 口径）；阈值塌缩场景（AR016 玩具校准 threshold=0.005）两臂同收敛，收益自然趋零，如实报告；
- `SimpleDiffusionLLM` 为随机权重参考模型：草拟质量（draft_acceptance_rate）无语义主张；本机主张 = 轮数削减 + 端到端加速 + 一致率，GPU 7B 语义质量按 s12 协议服务器复核；
- `PagedKVStore` 双账本口径不破：draft-verify 路径 append 语义沿 AR014/AR016，不改页布局/驱逐语义；
- 替换单调性：验证轮只做原地替换（不重掩码），轮数 <= max_rounds 确定性上界。

**假设：**
- 块内自一致迭代在真实 dLLM 上收敛轮数远小于 W（Fast-dLLM/Jacobi 文献实证；玩具模型实测验证）；
- 草拟前向（全 MASK 条件 argmax）在随机权重玩具模型上 acceptance 可能偏低——一致率门槛（>= 90%）以探针实测校准，未达标按预注册修订纪律落档；
- 跨块前瞻流水（overlap 草拟）收益叠加留待后续 AR（本 AR 先交付块内机制的正确性与实测）。

## 6. 术语说明

| 术语 | 定义 |
|---|---|
| 草拟（draft） | round-0 前向对全 MASK 块全位取 argmax 并整块填充的动作 |
| 验证（verify） | 每轮前向后校验各位 `argmax(logits) == current_token` 的自一致性 |
| 自一致迭代 | 不匹配位原地替换、重复前向直至不动点的过程（Jacobi 式） |
| 不动点（fixpoint） | 某验证轮替换位数为 0 的状态；`rounds_to_fixpoint` 记录达到轮数 |
| draft_acceptance_rate | 首轮草稿 token 被最终封存保留的位比例（batch 总均值） |
| 收尾（forced completion） | 达 max_rounds 未不动点时以当前状态强制封存（块内恒无 MASK） |
| mask-init | AR016 既有模式：全 MASK 起步、过阈解掩、保底 top-1 |
