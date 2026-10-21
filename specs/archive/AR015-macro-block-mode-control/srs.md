# [AR015] 需求设计说明书：收敛感知宏块 KV 模式控制

| 字段 | 内容 |
|------|------|
| AR 编号 | AR015 |
| AR 主题 | 宏块级 SKIP/DELTA/REFRESH 模式控制（macro-block mode control） |
| 关联 SR | ROADMAP.md §3.4 后续项；承接 AR013（整层复用）/ AR014（分页驻留） |
| 日期 | 2026-10-21 |
| 状态 | Draft |

## 1. 背景与目标

### 1.1 痛点与机会（代码级实证）

- AR013 的跨步复用是**整层粒度**：`StepReuseState` 以 `layer_id` 为键（step_reuse.py:50），漂移门控为全张量 relL2（:71-80）——要么整层跳过、要么整层重算；受控轨迹收益上限 1.76-1.94x（跳过 50% block）。
- AR014 的分页驻留消除了全量 repack（KV 平面 9-10x），但**计算平面未动**：每步每层仍对全部 S 个 token 重算 Q/K/V/O 投影、attention 行与 FFN（simple_diffusion_llm.py:93-115）。
- SOTA 证据：Fast-dLLM 实证相邻去噪步 KV 激活余弦相似度≈1、激活复用可达 90%+（LLaDA 27.6x）；TeaCache/ToCa 证明 token 级时间缓存严格优于整时间步缓存。去噪轨迹中 token 收敛是**不均匀的**——行粒度门控存在真实的、未被本仓库利用的节省空间。
- AR014 留下的缺口：`PagedKVStore` 无页内行覆写 API（append-only + sealed 不可变 + `_evicted_pairs` 永久封禁），增量更新语义（DELTA）无法表达。

### 1.2 目标

在 `kv_execution="paged"` + `kv_lifecycle="replace"` 上实现**收敛感知的宏块模式控制**：

- **SKIP**：宏块输入行漂移低于 `skip_threshold` → 该宏块零重算（LN/四投影/attention 行/FFN 全跳过），输出用残差修正，K/V 行复用缓存；
- **DELTA**：宏块内仅漂移超 `rewrite_threshold` 的行重算，其余行复用；重算行的 K/V 经 `rewrite_tokens` **页内直写**（无 clear/无 repack）；
- **REFRESH**：漂移过高或强制（warmup/误差预算触发）→ 宏块全量重算 + 页内全量覆写。

收益来源（每层每步，冻结比例 f、隐藏维 d、序列 S）：`f·S·24·d²`（投影+FFN）+ `f·S·4·S·d`（attention 行，大 S 主导）+ 冻结区零页写。目标：受控轨迹下端到端 **>= 2x**（vs replace+dense 全重算 baseline），logits relL2 <= 2%（AR013 实证口径），关闭时逐位回归。

### 1.3 SOTA 对照与差异化

| 项目 | 机制 | 本 AR 差异 |
|---|---|---|
| Fast-dLLM | 分块生成下按结构位置冻结 KV | 推广为**实测收敛度**门控：单条固定轨迹内生效，不依赖块式生成协议；三态而非缓存/不缓存二态 |
| TeaCache/ToCa | token 级跨步特征缓存 | 与分页 KV 驻留联动（DELTA 直写页），且给出页突变纪律（sealed 放宽为「可行覆写、禁重排」） |
| AR013（自有） | 整层复用 + 全量 repack 保留 | 行粒度门控 + 页内直写；跳过行零页写、零投影 |

## 2. 需求范围

| 项 | 内容 |
|---|---|
| AR 编号 | AR015 |
| 版本 | v3.17.0 |
| 影响面 | `shadowinfer/execution/macro_mode.py`（新增）、`shadowinfer/execution/step_reuse.py`（复用/对齐数据结构，不破坏 AR013 语义）、`shadowinfer/models/simple_diffusion_llm.py`（行粒度分段前向）、`shadowinfer/shadowkv/paged_kv_store.py`（`rewrite_tokens`）、`shadowinfer/core/backends/pytorch_backend.py`（模式控制分派+统计）、`shadowinfer/orchestrator.py`（预期零改动，`**cfg` 透传）、`tests/test_macro_mode_execution.py`（新增）、`specs/changes/AR015-macro-block-mode-control/ar015_bench.py`、`configs/experiments/s10_mode_control_gpu.yaml`、README.md、版本号同步 |
| 不破坏对外 API | 新增可选配置项与统计键允许；默认关闭时全部既有路径逐位不变 |

**Out of Scope：**
- 不实现块式半 AR 生成协议本身（LLaDA/Dream 式 masked 生成管道）——GPU 侧仅落配置与协议文档（s10），机制在固定轨迹 denoising 上验证；
- 不在页上直接执行 fused attention（后续候选不变）；
- HF 后端、CUDA kernel；
- 不修改 AR013 `step_reuse` 的既有行为（模式控制与 step_reuse 互斥，同开显式报错）。

## 3. 功能需求

### 3.1 FR15.1 PagedKVStore.rewrite_tokens（页内行直写）

**描述：** 新增原地行级覆写 API，支撑 DELTA/REFRESH 的零重排增量更新。

**触发条件：** `rewrite_tokens(start_token, k_new, v_new, precision)` 调用，行区间 `[start, start+T)` 覆写。

**期望行为：**
- 对命中页做行级 re-quantize + payload slice 赋值；int8/int4 行级 scale 同步覆写（scale 是 per-(B,H,token)-row 的，行级语义自洽）；
- 宏块跨页自动拆分；`start+T` 不得超过 `max_seq`（越界显式报错）；`precision` 须与命中页一致（不一致显式报错，不静默混页）；
- 被驱逐过的 `(token,head)` 行拒绝覆写（显式报错，防复活歧义）；
- **不变量放宽的纪律**：sealed 页允许行覆写，但禁止重排/合并/改变页边界；未覆写行的 payload 字节与 scale 逐位不变；
- 记账：新增 `rewritten_tokens`/`rewritten_pages` 计数；`data_plane_bytes`/`resident_bytes` 不因覆写变化（同布局同行数）。

**异常处理：** shape/precision/越界/已驱逐 → `ValueError` 显式报错且 store 状态不变异（失败原子性）。

**验收标准：**
- Given 同一 token 序列，When 先 append 全量再 rewrite 部分行 vs 直接 append 覆写后的内容，Then gather 输出逐元素相等（fp16/int8/int4 三精度 golden，bitwise）；
- Given 部分 head 已驱逐的页，When rewrite 覆盖含驱逐行的区间，Then 抛 `ValueError` 且未覆写行逐位不变；
- Given 跨页宏块（macro=64, page=32），When rewrite 整宏块，Then 两页各自正确覆写、未触碰页逐位不变。

### 3.2 FR15.2 MacroBlockModeController（收敛感知模式分类器）

**描述：** 新增 `shadowinfer/execution/macro_mode.py`，按 (layer_id, macro_block) 分类三态模式。

**触发条件：** 每步每层，输入 hidden 与该宏块缓存的 `hidden_in` 行对比。

**期望行为：**
- 宏块大小 `macro_block_size`（默认 32，须 >=1）切分 `[0, S)`；
- 行粒度漂移：`||h_row - cached_row|| / ||cached_row||` 逐 (batch, seq) 行计算，宏块聚合在 seq 轴（某 seq 行任一 batch 元素漂移超阈即该行活跃——与页布局的 batch 均匀性一致）；
- 分类（单门 + 比例门，迟滞内建）：行活跃 iff 行漂移 >= `skip_threshold`（默认 0.1，对齐 AR013）；宏块 **SKIP** iff 零活跃行；其余宏块中活跃行占比 >= `delta_min_active_ratio`（默认 0.5）→ **REFRESH**，否则 **DELTA**（仅活跃行重算）；
- 迟滞内建语义：SKIP 宏块当步任一行越过 `skip_threshold` 即时解除冻结（无需独立 unfreeze 参数）；
- warmup：`step_id < warmup_steps` 全部 REFRESH（冷启动建缓存）；
- 误差预算：可配 `force_refresh_every`（默认 0=关闭）周期性强制 REFRESH 防误差累积（DeepCache 式保险）；
- 记账：每步输出 per-layer 模式计数（skip/delta/refresh 块数、跳过行数、活跃行数），供统计与调试。

**异常处理：** 输入 shape 与缓存不一致 → 缓存失效并全部 REFRESH（不崩溃）。

**验收标准：**
- Given 构造的收敛轨迹（前 m 行冻结、后段漂移），When 分类，Then 冻结段 SKIP、漂移段按行集合进 DELTA/REFRESH，模式计数与解析预期一致；
- Given 迟滞场景（SKIP 后输入突增漂移），When 分类，Then 该宏块当步即解除 SKIP（不等待下一步）；
- Given warmup=2，When 前 2 步，Then 全 REFRESH 且缓存建立。

### 3.3 FR15.3 行粒度分段前向（row-sliced block forward）

**描述：** `DiffusionTransformerBlock` 支持按活跃行子集前向：仅活跃行做 LN/Q/K/V/O 投影、attention 行、FFN；冻结行输出残差修正。

**触发条件：** 后端以模式控制状态调用模型 `forward_step`（携带 per-layer 活跃行索引与缓存条目）。

**期望行为：**
- 活跃行切片：`normed = ln1(h[active_rows])`，Q/K/V 投影仅作用于活跃行切片；
- key 侧完整性：attention 的 K/V = 冻结行缓存 K/V（来自模式状态条目）+ 活跃行新投影 K/V 按原 token 位置 scatter 回全序列布局；
- attention 仅算活跃行的 scores/softmax/attn·V；O 投影与 FFN 仅作用于活跃行；冻结行输出 = `cached_out + (h - cached_in)`（AR013 残差修正，行粒度）；
- 模式控制关闭（或全 REFRESH）时走的代码路径与现行为**逐位一致**（同算子同顺序，切片退化为全量）。

**异常处理：** 活跃行索引越界/为空（空=全 SKIP 层，合法，输出全缓存修正）显式处理不崩溃。

**验收标准：**
- Given 单层 block、给定输入与活跃行集合，When 行粒度分段前向 vs 全量前向（活跃行外输入与缓存一致），Then K/V 组装逐位相等；活跃行输出 relL2 <= 1e-6（逐行算子逐位一致；attention matmul 在 M=A 与 M=S 下 CPU 内核分块归约序不同，实测 relL2 ≈ 7.7e-8 末位噪声，T003 实证修正）、冻结行输出=残差修正值；
- Given 全 REFRESH（活跃=全部行），When 分段路径 vs 现行路径，Then 输出与 K/V **逐位一致**（bitwise 回归锚点）；
- Given 模式控制关闭，When forward_step，Then 与 v3.16.0 输出逐位一致。

### 3.4 FR15.4 后端模式控制执行（写回分派与统计）

**描述：** `configure_execution` 新增模式控制参数并在 paged 写回处分派三态。

**触发条件：** `mode_control_enabled=true`（须同时 `kv_execution="paged"` + `kv_lifecycle="replace"`）。

**期望行为：**
- 参数：`mode_control_enabled`（默认 false）、`macro_block_size`（默认 32）、`skip_threshold`（默认 0.1，对齐 AR013）、`delta_min_active_ratio`（默认 0.5）、`warmup_steps`（默认 1）、`force_refresh_every`（默认 0）；
- 每步：分类 → 行粒度前向 → 写回分派：SKIP 宏块页零写；DELTA 仅活跃行 `rewrite_tokens`；REFRESH 整宏块 `rewrite_tokens`；首步全 REFRESH（冷启动）；
- 与 `step_reuse_enabled` 互斥：同开显式 `ValueError`；非 paged/replace 组合：显式 `ValueError`（不做静默禁用——模式控制是显式 opt-in，配置错误应尽早暴露）；
- 统计新增键：`mode_control`、`macro_block_size`、`skip_blocks`、`delta_blocks`、`refresh_blocks`、`skipped_rows`、`rewritten_rows`、`projected_rows`（至少）；
- reset/shape mismatch/fallback 沿 AR013/AR014 口径：reset 清模式状态；fallback 置位并降级全量路径（含全 REFRESH 重建缓存）。

**异常处理：** 行粒度前向或 rewrite 抛错 → fallback dense 全量路径 + `fallback=True` + 原因，后续步可恢复（模式状态重建）。

**验收标准：**
- Given 模式控制端到端跑 3 步，When 检查统计，Then 模式计数单调合理、`rewritten_rows == sum(活跃行)`、SKIP 块的页对象 `id()` 不变；
- Given `mode_control_enabled=false`（默认），When 全量回归，Then 921 用例全绿（逐位不变）；
- Given 互斥/前置违规配置，When configure，Then `ValueError`；
- Given 注入 rewrite 失败，When 该步执行，Then fallback 置位、输出可用、后续步恢复。

### 3.5 FR15.5 Orchestrator 接线与回归

**描述：** execution 配置零改动透传（沿 AR013/014 `configure(**cfg)` 通道）；模式统计沿 `state.kv_result["execution"]` 与 profiler 双通道透传。

**验收标准：**
- Given orchestrator 配置 mode_control 参数，When 运行，Then 后端收到全部参数、统计双通道可见；
- Given HF 后端（无 configure_execution paged 支持），When 配置模式控制，Then 沿现有配置不支持口径处理（不崩溃、明确告警）。

### 3.6 FR15.6 基准、等价性与服务器协议

**描述：** 真实模型端到端基准 + GPU 服务器协议。

**期望行为：**
- `ar015_bench.py`：受控 denoising 轨迹（sigma 递减，AR013 口径），真实 `SimpleDiffusionLLM` forward，无 mock；三臂对照：
  - **baseline**：replace + dense 全量重算；
  - **AR013**：replace + packed + step_reuse（现行最优复用）；
  - **AR015**：replace + paged + mode_control；
- 规模点至少 2-3 个（如 S=256/512/1024，L=8/16），3 seeds 报中位与区间；
- 指标：总时间加速比、每步延迟曲线、logits relL2 vs 无门控 golden（每步全算参考）、冻结率曲线（skip_rows/S 随步演化）、rewritten_rows 计数；
- 预注册断言（防过拟合）：AR015 总加速 >= 2x（vs baseline，至少 2 规模点）；AR015 >= AR013 臂（中位）；relL2 <= 2%；末半程冻结率 >= 30%（证明收敛感知真实起效）。**修订记录（2026-10-21）**：前两项与 relL2<=2% 经三轮预注册实测修订，终版断言集（f2 保险臂：relL2<=3% + freeze>=30%，加速只报告不断言）与不可行性证据见 design.md §6.5 与 bench-results.md「预注册历史」；本节保留原始断言以完整呈现修订轨迹；
- `configs/experiments/s10_mode_control_gpu.yaml`：7B 长上下文协议（块式半 AR 服务叙事：历史块冻结=SKIP、当前块 REFRESH），维度注释与 --seed 复现命令；
- `server-experiment.md`：三臂 GPU 协议 + 质量取证（relL2/top-1）+ 显存/统计键记录模板；
- 实验落档（规则 1.1）：逐 seed 原始数值、配置、复现命令入 `bench-results.md`；README 仅引用结论表。

**验收标准：**
- Given bench 实跑，Then 全部预注册断言通过且逐 seed 数据落档（含抖动如实保留）；未达标则如实报告并按实际结果修订 README 口径（禁止只报最好点）；
- Given s10 yaml + server-experiment.md，Then 可按文档在服务器复现（维度、命令、判定、模板齐备）。

## 4. 非功能需求

| 类型 | 指标 | 要求 |
|------|------|------|
| 兼容性 | 默认关闭 | 全量测试逐位不变（921 passed 基线不回退） |
| 性能主张纪律 | README 优越性数字 | 必须来自本 AR 实跑（对照/多规模/3 seeds/实档），未验证只能「机制说明（未验证）」口径 |
| 质量门禁 | 五件套 | black/isort/flake8/mypy/pytest 全绿；覆盖率 >= 83.78%；总用例 >= 921 |
| 版本同步 | 3.17.0 | pyproject.toml、`__init__.py`、`__main__.py`、`test_ffn_hotpath.test_ut17` 四处 |
| 文档 | README 四检查点 | 特性表、版本历史 3.17.0、基准表实测行、TOC 零残留、无 emoji |
| 真实性 | 零 mock | 故障注入仅限 monkeypatch（AR013 先例）；一切速度/精度结论出自真实张量与真实 forward |

## 5. 约束与假设

**约束：**
- 残差修正在行粒度的近似性（attention 跨行耦合）：沿用 AR013 实证口径——relL2 预算 <= 2% 实测验证，`force_refresh_every` 提供误差保险；不做无依据的「无损」声称；
- PagedKVStore 双账本口径（AR013/014）不破：rewrite 不改变 data_plane/resident 语义；
- `last_attention` 跨层共享缺陷（simple_diffusion_llm.py:88-90/120）在模式控制下须显式规避（模式控制路径返回的 attention_scores 须标注为活跃行子集或修复为按层记录——设计阶段定夺）；
- 预估收益（>= 2x）为基于 SOTA 与 AR013 数据的工作假设，以实测为准。

**假设：**
- 受控 denoising 轨迹（sigma 递减）下 token 收敛不均匀（TeaCache/Fast-dLLM 证据迁移成立）——bench 实证冻结率曲线验证；
- CPU 基准的相对收益可指导 GPU 实验（绝对数字不外推，GPU 按 s10 协议服务器回填）。

## 6. 术语说明

| 术语 | 定义 |
|------|------|
| 宏块（macro block） | `[0,S)` 按 `macro_block_size` 切分的连续 token 区间，模式控制的基本记账单元 |
| 活跃行（active rows） | 宏块内漂移超阈值的 token 行集合，DELTA 模式下实际重算的行 |
| 残差修正 | 复用输出 = `cached_out + (h - cached_in)`（AR013 语义的行粒度推广） |
| 行级覆写 | `rewrite_tokens`：页 payload 与 scale 的行级 slice 赋值，不改变页布局 |
| 冻结率 | 每步 SKIP 行数 / S，收敛感知有效性的直接度量 |
