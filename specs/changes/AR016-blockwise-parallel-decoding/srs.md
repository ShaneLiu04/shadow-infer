# [AR016] 需求设计说明书：置信度感知块式并行解码服务管线

| 字段 | 内容 |
|------|------|
| AR 编号 | AR016 |
| AR 主题 | 块式半 AR 生成环路 + 跨块精确前缀 KV 驻留（Fast-dLLM 对标） |
| 关联 SR | ROADMAP.md §2「Real Diffusion LLM model adapter (Fast-dLLM-v2 or equivalent)」、§4 创新提案 #5（并行解码）；承接 AR013/AR014/AR015 |
| 日期 | 2026-10-22 |
| 状态 | Draft |

## 1. 背景与目标

### 1.1 痛点与机会（代码级与 SOTA 实证）

- 本仓库 AR013（层复用）/AR014（分页 KV）/AR015（行粒度模式控制）全部在**合成去噪轨迹**上验证；AR015 的 s10 叙事假设「块式半 AR 服务」工作负载，但该负载本身从未实现——所有 KV 创新缺少真实落点，ROADMAP §2 最大未关闭缺口即「Real Diffusion LLM adapter（Fast-dLLM equivalent）」。
- SOTA 实证（Fast-dLLM，Nvidia/MIT/HKU，LLaDA 27.6x）：①分块 KV 缓存——相邻推理步 KV 激活余弦相似度 ≈ 1，前序块 KV 可复用（90%+ 激活重用）；②置信度感知并行解码——仅对置信度（max softmax 概率）≥ 阈值的 token 并行解掩，无 token 过阈时保底解掩 top-1；理论保证 (n+1)ε≤1 时贪心并行解码与顺序解码一致。
- Fast-dLLM 的 KV 复用是**近似**的（dual-cache 缓存双向 attention 下相邻步激活，余弦≈1 但非精确）。本 AR 的架构机会：**块式半 AR 协议下前缀上下文在块封存后永不变化**（序列按块增长，后块不进入已封存块的 attention 感受野）→ 前缀 KV 可**精确**复用，无近似误差、无漂移门、无误差累积——比 Fast-dLLM 更强的正确性主张。
- AR015 教训：冻结行**输出**回放引入跨行耦合误差（CPU 玩具负载 relL2 累积至 6.7-10.1%）。块式解码只需冻结行的 **K/V**（作为 attention 上下文），不需要冻结行输出——误差源根本消除。

### 1.2 目标

实现端到端**块式半 AR 生成管线**（置信度感知并行解码 + 跨块精确前缀 KV 驻留）：

- **生成环路**：prompt → 逐块生成：块内全掩码初始化 → 若干轮真实前向 → softmax 置信度 → 并行解掩过阈 token（保底 top-1 保进度）→ 块满封存（KV 入页）→ 下一块；
- **执行平面**：前缀 KV 以 `PagedKVStore` 按层驻留（prompt 预填充一次 + 块封存时 append 一次；**轮间零页写**）；每轮仅对当前块 W 行做投影/attention/FFN（尾块连续活跃前向），前缀 K/V 走页 gather；
- **精确性**：fp32 页下每轮 logits vs 全量重算 baseline relL2 <= 1e-6（attention 分块归约序噪声口径，AR015 T003 实证），贪心 token 轨迹与 baseline **完全一致**；fp16/int8 页为精度-内存权衡臂（实测报告）；
- **性能**：每轮计算行数 (P+jW) → W（玩具规模 ~9-16x 行计算削减）；KV 写行数 rounds×(P+jW) → 每行一次。目标：受控生成任务端到端 >= 3x vs 全量重算 baseline（保守预注册，实测为准）。

### 1.3 SOTA 对照与差异化

| 项目 | 机制 | 本 AR 差异 |
|---|---|---|
| Fast-dLLM | dual-cache 近似缓存前缀+后缀激活（余弦≈1） | 块式半 AR 协议下前缀 KV **精确**复用（上下文封存后不变）；页式驻留（fp16/int8 量化页）而非张量缓存；轮间零页写 |
| LLaDA 半 AR 重掩码 | 块内低置信度重掩码（退火） | 单调解掩（Fast-dLLM 式）+ 保底 top-1；重掩码出范围（轮数确定性优先） |
| AR015 模式控制 | 行粒度冻结行**输出**回放（近似，误差累积） | 只消费冻结行 **K/V**（精确）；活跃窗口恒为连续尾块（无需行索引重组） |
| 本仓库既有 serving | `/generate` 端点走 AR 语义单步 | 首条 dLLM 块式并行解码生成语义 |

## 2. 需求范围

| 项 | 内容 |
|---|---|
| AR 编号 | AR016 |
| 版本 | v3.18.0 |
| 影响面 | `shadowinfer/execution/block_decoding.py`（新增：调度/置信度/统计）、`shadowinfer/models/simple_diffusion_llm.py`（`forward_block` 尾块活跃前向）、`shadowinfer/core/backends/pytorch_backend.py`（`generate` 管线 + 前缀页驻留 + 统计）、`shadowinfer/execution/__init__.py`（导出）、`tests/test_block_decoding.py`（新增）、`specs/changes/AR016-blockwise-parallel-decoding/ar016_bench.py`、`configs/experiments/s11_blockwise_decoding_gpu.yaml`、README.md、版本号同步 |
| 不破坏对外 API | `generate` 为 backend 新增方法；既有 forward_step/configure_execution 签名不变；默认路径逐位回归 |

**In Scope（本 AR 要做的）：**
- 置信度感知并行解掩调度（阈值 + 保底 top-1 + max_rounds 保底完成）；
- 模型 `forward_block`：连续尾块活跃行前向，per-layer 前缀 K/V 条件 attention；
- 后端 `generate(prompt_ids, gen_len, config)`：预填充 → 逐块循环 → 封存 append；`PagedKVStore` 前缀驻留（page_precision fp32/fp16/int8 可配）；轮间零页写；
- 生成统计（轮数/解掩数/置信度轨迹/计算行数/KV 写行数/保底完成数）；
- 四臂基准（baseline 全量重算 / blockwise-fp32 / fp16 / int8）+ 校准后预注册断言 + 逐 seed 落档；
- s11 GPU 协议（LLaDA/Dream 7B 叙事）+ server-experiment.md。

**Out of Scope（本 AR 不做的）：**
- 低置信度重掩码（LLaDA 退火式）与自适应轮数策略；
- HF 后端 / vLLM / SGLang 的 generate 接线（后续 AR）；
- 真实预训练 dLLM 权重加载（GPU 侧按 s11 协议服务器复现）；
- CUDA kernel / fused attention；
- orchestrator 面向生成请求的调度改造（generate 为 backend 直连 API）。

## 3. 功能需求

### 3.1 FR16.1 置信度感知并行解掩调度（ConfidenceSchedule）

**描述：** 新增 `shadowinfer/execution/block_decoding.py`，实现 Fast-dLLM 式置信度感知并行解码调度。

**触发条件：** 每轮前向后，输入块内掩码位置的 logits。

**期望行为：**
- 置信度 = 掩码位置 softmax 概率最大值；
- 并行解掩：所有置信度 >= `confidence_threshold`（默认 0.9）的掩码位置一次性解掩（并行解码）；已解掩位置不再参与；
- **进度保证**：若无任何位置过阈，解掩置信度最高的单个位置（Fast-dLLM 保底语义，杜绝死循环）；
- **保底完成**：轮数达 `max_rounds` 仍有残掩码 → 全部强制解掩（取各自 argmax token），计入 `forced_completions` 统计；
- 解掩 token = 该位置 argmax token id（贪心）；
- 配置校验：`block_size >= 1`、`max_rounds >= 1`、`0 < confidence_threshold <= 1`、`gen_len % block_size == 0`、`mask_token_id` 在词表内且不出现在 prompt——非法配置显式 `ValueError`。

**异常处理：** 空 logits / 全零长度块等退化输入显式报错不崩溃。

**验收标准：**
- Given 构造 logits（部分位置高置信、部分低置信），When 调度，Then 过阈位置全部解掩、未过阈保持掩码、解掩 token == argmax；
- Given 全部低于阈值，When 调度，Then 恰好解掩 top-1 位置；
- Given 达 max_rounds 仍有掩码，When 收尾，Then 全部解掩且 forced_completions 计数正确；
- Given 非法配置，When 构造 Config，Then `ValueError`。

### 3.2 FR16.2 模型尾块活跃前向（forward_block）

**描述：** `SimpleDiffusionLLM.forward_block`：仅对连续尾块（当前块）W 行做完整 transformer 计算，前缀 L 行 K/V 由外部供给。

**触发条件：** 后端 generate 管线每轮调用；输入 `x_block`（token ids 或 hidden，[B, W] / [B, W, d]）、`prefix_len`（前缀行数 L）、`prefix_kv`（per-layer (k, v)，[B, H, L, D]）。

**期望行为：**
- 仅活跃 W 行做 embed/位置/时间嵌入、LN/Q/K/V/O 投影、attention、FFN；
- 每层 attention 的 K/V = `prefix_kv[layer]` 与活跃行新投影 K/V 沿 token 轴拼接；Q 仅活跃行；scores/softmax/attn·V 仅活跃行（W × (L+W) attention）；
- 返回：块内位置 logits [B, W, vocab] + 每层活跃行新 K/V（供封存 append）；
- 时间嵌入：`step_t / total_steps` 口径与 forward_step 一致，两臂相同映射（不影响精确性对比）；
- `prefix_len == 0`（首块无前缀）合法。

**异常处理：** prefix_kv 层数与模型不符 / K/V 形状与 (B,H,L,D) 不符 / x_block 长度 != W 配置 → 显式 `ValueError`。

**验收标准：**
- Given 同一上下文（前缀 L 行 + 尾块 W 行），When forward_block vs 全量 forward_step（fp32 前缀 KV），Then 块内 logits relL2 <= 1e-6（attention 分块归约序噪声口径）、活跃行 K/V 逐位相等；
- Given prefix_len=0，When forward_block，Then 与全量前向（仅 W 行）一致；
- Given 形状不符，When 调用，Then `ValueError`。

### 3.3 FR16.3 后端 generate 管线与跨块前缀 KV 驻留

**描述：** `PyTorchModelBackend.generate(prompt_ids, gen_len, config)` 实现完整块式半 AR 生成环路；前缀 KV 以 `PagedKVStore` 按层驻留。

**触发条件：** 显式调用 `generate`（新 API，不改既有 forward_step 语义）。

**期望行为：**
- **预填充**：prompt 行经一次全量前向（或 forward_block with prefix_len=0 的块语义），K/V 按层 append 入页（一次性）；
- **逐块循环**：块 j 初始化为 MASK → 轮循环：`forward_block`（前缀 K/V 从页 gather）→ 置信度调度 → 解掩；块满或保底完成 → **封存**：块行 K/V append 入页（一次性），进入块 j+1；
- **零轮间页写**：轮循环内 store 无任何 append/rewrite（页数与页内容不变）；页数仅在预填充与封存时增长；
- **精度平面**：`page_precision` fp32 / fp16 / int8 可配（默认 fp32；fp32 臂与全量重算逐位同 KV，fp16/int8 为量化权衡臂）；
- 生成序列：`[B, P + gen_len]`，prompt 部分逐位保留；
- 返回 `GenerateResult`：token 序列 + 逐块统计（见 FR16.4）+ 总耗时。

**异常处理：** prompt 超长（P + gen_len > max_seq_len）/ prompt 含 mask_token_id / 页 append 失败 → 显式报错（fail-fast，带阶段诊断信息）。

**验收标准：**
- Given fp32 页、真实模型、随机 prompt，When generate vs baseline 环路（同调度、每轮全量重算 forward_step），Then 生成 token 序列**逐位一致**、逐步 logits relL2 <= 1e-6；
- Given 生成全程，When 检查 store，Then 轮间 page_count 与页内容不变（零页写），append 仅发生 P/gen_len 行各一次；
- Given fp16/int8 页，When generate，Then 正常完成且统计含精度标识（精度权衡由 FR16.5 基准量化）；
- Given 非法输入（超长/prompt 含 mask），When generate，Then `ValueError`。

### 3.4 FR16.4 生成统计与可观测

**描述：** 生成全程统计，供基准与调试。

**期望行为：** `GenerateResult.stats` 至少含：
- 总量：`blocks`、`total_rounds`、`computed_rows_total`（活跃行前向累计）、`kv_write_rows_total`（页写行累计）、`forced_completions`、`total_time_ms`、`page_precision`；
- 逐块：`rounds_per_block`、`unmasked_per_round`（含保底标记）、`mean_confidence_per_round`；
- 派生：`rows_saved_vs_baseline`（baseline 计算行数 - 实际计算行数）。

**验收标准：**
- Given 一次生成，When 检查统计，Then 键恒存在、数值与解析预期一致（computed_rows == ΣW×rounds、kv_write_rows == P + gen_len）。

### 3.5 FR16.5 基准、精确性与预注册断言（ar016_bench.py）

**描述：** 真实模型四臂基准 + 校准后预注册。

**期望行为：**
- 四臂：**baseline**（同调度、每轮全量重算 forward_step）/ **blockwise-fp32**（精确臂）/ **blockwise-fp16** / **blockwise-int8**（量化权衡臂）；真实 `SimpleDiffusionLLM`，真实随机 prompt（seed 固定），无 mock；
- 规模点至少 3 个（如 P + G = 256/512/1024，L=8/16），3 seeds 报中位与区间；
- 指标：端到端总时间、逐块耗时、computed_rows、kv_write_rows、fp32 臂逐步 logits relL2 与 token 序列一致性、fp16/int8 臂 relL2 与 token 一致率；
- **校准先行**：正式预注册前跑校准探针（不入库），按实测固化阈值（含裕量），断言集跑前登记；
- 实验落档（规则 1.1）：逐 seed 原始数值、配置、复现命令入 `bench-results.md`；README 仅引用结论表。

**验收标准：**
- Given fp32 臂，When vs baseline，Then token 序列逐位一致且逐步 relL2 <= 1e-6（精确性断言）；
- Given 校准后预注册断言（预期含加速与 KV 写削减阈值），When 全量实跑，Then 按登记口径判定并如实落档（未达标如实报告，AR015 三轮预注册纪律沿用）；
- Given fp16/int8 臂，When vs baseline，Then relL2 与 token 一致率实测报告（校准后决定是否设断言）。

### 3.6 FR16.6 GPU 服务器协议、README 与版本同步

**描述：** s11 实验配置 + 服务器协议 + README 四检查点 + 版本 3.18.0 四处同步。

**期望行为：**
- `configs/experiments/s11_blockwise_decoding_gpu.yaml`：7B 块式解码协议（LLaDA/Dream 叙事：prompt + 多块生成，四臂，置信度阈值/块大小可配，维度注释与 --seed 命令）；
- `server-experiment.md`：GPU 复核协议（环境/本机基线/四臂命令/质量取证（token 一致率、relL2、top-1）/显存与页统计/记录模板/结论落档）；
- README：特性表 v3.18.0 条目（诚实口径）、质量表实测数、基准节、版本历史；版本 3.18.0 同步 `pyproject.toml`、`__init__.py`、`__main__.py`、`test_ffn_hotpath.test_ut17`。

**验收标准：**
- Given s11 yaml + server-experiment.md，Then 可按文档在服务器复现（维度、命令、判定、模板齐备）；
- Given README，Then 四检查点齐备、无 emoji、数据以实跑为准、无超证据声称。

## 4. 非功能需求

| 类型 | 指标 | 要求 |
|------|------|------|
| 兼容性 | 新增 API 不破坏既有 | forward_step/configure_execution 签名不变；全量测试逐位回归（1016 passed 基线不回退） |
| 精确性 | fp32 臂 | token 序列与 baseline 逐位一致；逐步 logits relL2 <= 1e-6 |
| 性能主张纪律 | README 优越性数字 | 必须来自本 AR 实跑（四臂对照/多规模/3 seeds/实档）；未验证只能「机制说明（未验证）」口径 |
| 质量门禁 | 五件套 | black/isort/flake8/mypy/pytest 全绿；覆盖率 >= 84.25%；总用例 >= 1016 |
| 真实性 | 零 mock | 生成环路全部真实前向与真实页存储；置信度来自真实 softmax；无 monkeypatch 伪造数据路径（故障注入仅限异常测试） |
| 版本同步 | 3.18.0 | pyproject.toml、`__init__.py`、`__main__.py`、`test_ffn_hotpath.test_ut17` 四处 |
| 文档 | README 四检查点 | 特性表、版本历史 3.18.0、基准表实测行、TOC 零残留、无 emoji |

## 5. 约束与假设

**约束：**
- 前缀 KV 精确性主张仅限 fp32 页（量化页的取舍如实报告，不做无损声称）；
- token 序列一致断言的前提是贪心解码且无阈值边界平局（校准探针验证；平局属浮点噪声边界，如实报告口径）；
- `SimpleDiffusionLLM` 为随机权重参考模型：生成 token 为真实前向输出但无语义质量主张；本机主张 = 精确性 + 计算行/KV 写削减 + 端到端加速，GPU 7B 语义质量按 s11 服务器复核；
- `PagedKVStore` 双账本口径不破：generate 路径 append 语义沿 AR014，不改页布局/驱逐语义；
- 单调解掩（无重掩码）：块内轮数 <= max_rounds 确定性上界。

**假设：**
- 块式半 AR 协议下已封存块上下文不变（序列按块增长，后续块不进入前序块 attention 感受野）——前缀 KV 精确复用的理论基础，UT 以逐步 logits 验证；
- CPU 玩具规模的行计算削减（~9-16x）足以在扣除 gather/调度开销后兑现 >= 3x 端到端加速（校准探针先行验证，实测为准）。

## 6. 术语说明

| 术语 | 定义 |
|------|------|
| 块（block） | 生成序列按 `block_size` 切分的连续 token 区间；生成与 KV 封存的基本单元 |
| 前缀（prefix） | prompt + 已封存块构成的上下文（长度 L）；其 K/V 精确驻留于页存储 |
| 封存（seal） | 块内全部 token 解掩完成后，该块 K/V 一次性 append 入页、上下文长度增长的动作 |
| 置信度 | 掩码位置 softmax 概率最大值；并行解掩的门槛信号 |
| 保底解掩 | 无位置过阈时解掩 top-1（进度保证）；max_rounds 耗尽时全部强制解掩（保底完成） |
| 零轮间页写 | 轮循环内页存储无 append/rewrite；页数仅在预填充与封存时变化 |
| computed_rows | 每轮活跃行数（W）累计；与 baseline 全量行数对照的削减度量 |
