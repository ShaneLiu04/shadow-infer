# [AR003] 需求设计说明书

| 字段 | 内容 |
|------|------|
| AR 编号 | AR003 |
| AR 主题 | 决策信号正确性修复与 IRAP 接收注意力重要性平面 |
| 关联 SR | 无正式 SR；来源为 AR002 期间全量代码审计（60+ 缺陷清单与三创新方向，存档于本目录 baseline-profile.md 引用） |
| 日期 | 2026-10-17 |
| 状态 | Draft |

## 1. 背景与目标

AR002 完成热点路径向量化与度量口径修复后，端到端加速收敛至 1.61x（中位），FFN 支线成为残余瓶颈。但对决策平面的深度审计发现一组**正确性级**缺陷，使 ShadowKV 的核心价值主张（按重要性分配 KV 精度）在语义上不成立：

1. **双重 softmax（KV-P2，Critical）**：参考后端 `SimpleDiffusionLLM` 返回 post-softmax 注意力权重（`simple_diffusion_llm.py:83`），而 agent 对其再次 softmax（`shadowkv_agent.py:439/:482`）；mock 路径（`orchestrator.py:1922` 生成 randn）则天然是 logits。同一入口两种语义，真实后端路径下重要性分布熵饱和、精度分配塌缩。
2. **重要性轴向相反（KV-P3，Critical）**：现实现以「每个 query 行的注意力熵」为重要性信号——衡量的是 query 的困惑度，而非 KV token 被关注的程度。与 H2O / StreamingLLM 文献的 received-attention 口径相反：注意力 sink（首 token 接收极高注意力但行熵极低）被判为最不重要，获得最低精度档。
3. **无界历史（KV-P9，Important）**：`importance_history` 每步 append 全量张量（`shadowkv_agent.py:265-267`），长会话下决策状态内存随步数线性增长，违背端侧常数内存约束。
4. **预算与驱逐为死代码（KV-P4/P8/P10，Critical/Important）**：`KVCacheManager` 具备 `memory_budget_bytes`/`eviction_policy`/`_token_head_meta` 全套基础设施（`kv_cache_manager.py:33-101`），但 `set_token_head_meta` 生产路径零调用、`memory_pressure` 由 orchestrator 计算并传入（`orchestrator.py:1350-1361`）却被 agent 忽略——「内存受限的 KV 压缩」闭环从未发生。

本 AR 以 **IRAP（Incremental Received-Attention Propagation，跨步 EMA 接收注意力传播）** 作为统一机制一并承载上述修复：重要性主信号改为列向接收注意力的跨步 EMA 累积（常数内存、增量计算、天然携带跨步稳定性），同步修复 softmax 契约与 sink 保护，并把预算/驱逐接线为生产路径。IRAP 同时产出 ΔA 跨步差分信号，为后续 AOG/宏块率控（AR005/AR006）提供输入。

**目标**：ShadowKV 决策平面从「机制存在但语义不成立」转变为「正确、有界、闭环」，且全部验证在 CPU 上可执行。

## 2. 需求范围

影响模块：`shadowinfer/shadowkv/shadowkv_agent.py`、`shadowinfer/shadowkv/kv_cache_manager.py`、`shadowinfer/shadowkv/eviction_policy.py`（如需）、`shadowinfer/orchestrator.py`（接线）、`shadowinfer/qdrift/qdrift_agent.py`（可选输入）、`tests/`、`README.md`、`ROADMAP.md`。

**In Scope（本 AR 要做的）：**
- attention 输入契约显式化（logits / probabilities）与单次归一化（FR3.1，KV-P2）
- 重要性主信号改为列向接收注意力 + sink 前缀硬保护（FR3.2，KV-P3）
- IRAP 跨步 EMA 传播，替代每步全量重算与无界 importance_history（FR3.3，KV-P9）
- 预算准入与驱逐接线为生产路径，消费 memory_pressure，修正 per-precision 字节估算（FR3.4，KV-P4/P8/P10）
- 导出 ΔA 跨步差分信号并支持 QDrift 可选消费（FR3.5）
- 上述修复的向量化路径与标量参考路径保持 golden 等价（AR002 口径维持）
- 基线取证、实验验证（AGENTS.md 规则 1.1 口径）、README/ROADMAP 更新

**Out of Scope（本 AR 不做的）：**
- FFN 支线修复（FFN-P1 单份权重缓存跨层共享、FFN-P3 逐通道量化循环、FFN-P4 双重上投影、unpack 向量化）——规划 AR004
- AOG 设备感知开销门控（ORC-P1 执行后仲裁、ORC-P3 逐层重复信号）——规划 AR005
- 宏块率控复用（SKIP/DELTA/REFRESH、按精度空闲表 slab 分配）——规划 AR006
- M2 后端契约变更（逐层 attention 供给、真实多层后端）——本 AR 全部修复在 agent 侧，不依赖后端改造
- int4 预存缺陷修复（packed unpack scale 索引 batch>1 错误等）——继续 bug-for-bug 维持，AR002 已建档
- RSS 级进程内存口径对账——维持张量级字节口径（AR002 srs §3.1 已修订）
- CUDA/加速器路径（全部验证 CPU-only）
- 需求/设计阶段不做任何创新收益实验（用户指示；实验集中在 T007/T008 执行）

## 3. 功能需求

### 3.1 attention 输入契约显式化与单次归一化

**描述：** 显式声明 agent 收到的 attention 张量语义（logits 或 probabilities），决策平面只做一次归一化，消除双重 softmax。

**触发条件：** `on_step` 收到 `attention_scores` 输入时。

**期望行为：**
- 新增配置键（命名于 design 阶段定，如 `attn_is_probabilities`）声明输入语义；orchestrator 按后端实际返回接线（`SimpleDiffusionLLM` 为 probabilities，mock randn 为 logits）
- probabilities 输入：跳过 softmax（仅做行归一化数值安全校正）；logits 输入：单次 softmax
- 标量参考路径（`compute_importance_score`）与向量化路径（`_compute_importance_matrix`）同步修复

**异常处理：** 配置缺失时按 mock 路径现状（logits）处理并 `log_alert("warning")` 一次；输入含 NaN/Inf 时按现有防御逻辑处理。

**验收标准：**
- Given 已行归一化的 attention 输入且重要性异质（构造强非均匀注意力，含明确高/低接收注意力 token），When 计算重要性并分档，Then 非 fp32 档 token 占比 ≥ 50% 且实际生效档位种类 ≥ 3（修复前该构造下全 fp32 塌缩——回归对照断言）
- Given logits 输入，When 归一化，Then 仅执行一次 softmax 且中间权重与修复前 bitwise 一致（中间量 golden 对照；最终重要性因 FR3.2 公式变更而不同，不作为等价断言对象）
- Given 同一输入，When 分别走标量与向量化路径，Then golden 等价 max|Δ| ≤ 1e-6（AR002 口径维持）

### 3.2 接收注意力重要性轴与 sink 前缀保护

**描述：** 重要性主信号改为列向接收注意力（KV token 被关注程度），并对注意力 sink 前缀做硬保护。

**触发条件：** 每步重要性计算与精度分档时。

**期望行为：**
- 主信号 = 当前步列向接收注意力（与 FR3.3 的 EMA 累积融合）；行熵降为辅助信号（权重于 design 阶段定）
- 前缀 `sink_tokens`（配置键，默认值 design 定）个 token 强制不降档、不驱逐
- 两条计算路径（标量/向量化）同步

**异常处理：** `sink_tokens ≥ seq_len` 时全部保护并告警（配置错误提示）。

**验收标准：**
- Given sink 模式 attention（首 token 接收注意力最高、行熵最低），When 分档，Then sink token 获最高精度档（修复前为最低档——回归对照测试显式断言此反转）
- Given 某中段 token 被高频跨行引用（对照组为低接收注意力 token），When 分档，Then 其档位严格高于对照组，且全序列档位与列向接收注意力的 Spearman 相关 ≥ 0.9
- Given 向量化与标量路径，Then golden 等价 ≤ 1e-6

### 3.3 IRAP 跨步 EMA 传播与常数内存历史

**描述：** 跨步重要性以 EMA 传播（A_new = γ·A_prev + 本步接收注意力增量），替代每步全量重算与无界 `importance_history`。

**触发条件：** 每步 `on_step`（存在 `kv_previous` 时增量更新；缺失时全量初始化）。

**期望行为：**
- 新增配置键 `irap_decay`（γ，默认 0.9，design 定）；EMA 状态 per-layer 常数大小（S×H）
- `importance_history` 无界 append 改为常数状态：EMA 张量（rank 归一化在分档时刻逐 head 进行，无需持久化标量统计）
- `on_init` 重置语义维持（AR002 M4 口径：保留配置、清空状态）

**异常处理：** 状态形状与当前步不匹配（如 seq 增长）时按增广规则处理（design 定）并告警；γ 越界（≤0 或 >1）配置校验报错。

**验收标准：**
- Given 连续多步，When 增量 EMA 更新，Then 与「全量重算等效定义」数值一致（容差 1e-6，等价性 golden 测试）
- Given 步数持续增长，When 检查决策状态内存，Then 历史相关字节不随步数增长（对象级断言，O(1)）
- Given `kv_previous` 缺失的第一步，When 计算，Then 正确初始化且无异常

### 3.4 预算准入与驱逐接线（消费 memory_pressure）

**描述：** 把 KVCacheManager 的预算/驱逐基础设施接线为生产路径：store 同步元数据、memory_pressure 超阈触发降档/驱逐、字节估算修正。

**触发条件：** `on_step` 收到 `memory_pressure` 输入（orchestrator 已传入）且超过动作阈值时；以及 store 压缩条目时。

**期望行为：**
- store 路径写入 per-token-head 重要性元数据（张量级矩阵；`set_token_head_meta` 对象级 API 保留兼容）
- `memory_pressure` 超过阈值（design 定）时按 IRAP 重要性升序降档/驱逐（sink 保护优先级最高，不可驱逐），动作后字节占用 ≤ `memory_budget_bytes`
- per-precision 字节估算修正为 bits×count 口径（修复 8x 误差，KV-P10）
- 预算预分配准入：新条目超预算时先降档再准入；常态降档在 precision_map 上以 O(选中集合) 完成（不触发重打包）；罕见全局驱逐路径允许 O(全量) 重打包（packed.evict，仅 pressure 超阈触发）

**异常处理：** 未配置 `memory_budget_bytes` 时保持现状（不触发预算动作）并告警一次；驱逐至 sink 保护仍超预算时保留保护集并告警（不静默丢数据）。

**验收标准：**
- Given 小预算 + 高低重要性混合 token，When memory_pressure 超阈，Then 低重要性 token 先被降档/驱逐且压缩后总字节 ≤ 预算（字节级断言）
- Given sink token，When 驱逐，Then sink token 存活
- Given memory_pressure 输入，When 超阈，Then 触发真实降档/驱逐动作（修复前为空操作——回归对照断言）
- Given 未超阈，Then 不发生任何驱逐（稳定性）

### 3.5 ΔA 跨步差分信号导出与 QDrift 可选消费

**描述：** 导出 IRAP 跨步差分信号（如 ‖A_new − A_prev‖/‖A_prev‖），并支持 QDrift 作为可选输入消费。

**触发条件：** 每步 `on_step` 计算完成后。

**期望行为：**
- 信号写入 step metrics（orchestrator 可观测），并进入 agent 返回值
- QDrift 新增可选输入（如 `importance_drift`），默认不融合（AR002 语义不变）；融合方式与权重于 design 定（若评估后仅导出不融合，需在 design 记录决策依据）

**异常处理：** 首步无 prev 时信号置 None/0 并跳过；QDrift 收到非法值时忽略并告警。

**验收标准：**
- Given 构造跨步重要性突变，When 导出 ΔA 信号，Then 信号非零且随突变幅度单调
- Given 默认配置（不融合），When 全量回归，Then AR002 既有测试全部不变（兼容性锚点）
- Given QDrift 开启融合，When 重要性突变，Then drift 响应方向正确（统计测试）

## 4. 非功能需求

| 类型 | 指标 | 要求 |
|------|------|------|
| 正确性 | 向量化 vs 标量参考 golden | max\|Δ\| ≤ 1e-6 维持（AR002 口径） |
| 兼容性 | 对外 API | 不破坏既有接口；仅新增可选参数/配置键/返回字段 |
| 性能 | 端到端 | 相对 AR002 基线 1.61x 不显著回退：3 次重复中位 ≥ 1.4x（兜底区间），实测落档 |
| 内存 | 决策状态 | 步数增长下 IRAP 历史字节 O(1)（对象级断言） |
| 质量 | 五件套 | black/isort/flake8/mypy/pytest 全绿；覆盖率不低于 AR002 口径（81.68%） |
| 环境 | 执行设备 | 全部验证 CPU-only 可执行（无 CUDA 依赖） |

## 5. 约束与假设

**约束：**
- AGENTS.md 规则 1.1：任何优于 baseline 的声称须先完成丰富实验（≥3 次重复取中位、多规模扫描、golden/统计证据、落档 specs），否则 README 仅能以「机制说明（未验证）」口径描述——本 AR 的 T007/T008 按此执行
- bug-for-bug 例外清单更新：KV-P2/P3/P4/P8/P9/P10 从例外清单移除（本 AR 修复）；int4 预存缺陷继续维持
- 不引入第三方依赖；不修改 M2 后端契约
- 需求与设计阶段不执行创新收益实验（用户指示，2026-10-17）

**假设：**
- `SimpleDiffusionLLM` 仍是唯一参考后端（post-softmax 返回）；mock 路径 randn 语义不变
- γ（irap_decay）、sink_tokens、行熵辅助权重、memory_pressure 动作阈值的默认值在 design 阶段定，并经 T001 基线取证校准
- 现有测试中依赖旧重要性语义的断言需要同步修订（修订需附根因说明，属预期内破坏性变更，限于测试文件）

## 6. 术语说明

| 术语 | 定义 |
|------|------|
| IRAP | Incremental Received-Attention Propagation：跨步 EMA 累积列向接收注意力的重要性传播机制 |
| 接收注意力（received attention） | 列向累加的注意力权重：某 KV token 被全部 query 关注的总量 |
| 注意力 sink | 序列前部持续接收极高注意力的 token（文献 StreamingLLM）；本 AR 指前缀 sink_tokens 个 token |
| EMA | 指数移动平均，A_new = γ·A_prev + 增量 |
| memory_pressure | orchestrator 计算的内存压力标量 ∈ [0,1]（total_mb/budget_mb 截断） |
| 精度档 | per-token-head 的 KV 存储精度（fp32/fp16/int8/int4） |
| ΔA 信号 | ‖A_new − A_prev‖/‖A_prev‖ 归一化跨步差分，衡量重要性平面稳定性 |
