# [AR018] ST 验收测试用例

| 字段 | 内容 |
|------|------|
| AR 编号 | AR018（真实 dLLM 权重执行平面） |
| 关联 srs.md | ./srs.md |
| 生成日期 | 2026-10-24 |
| 执行环境 | Windows / CPU / Python 3.11.9 / torch 2.5.1+cu121（本机零性能主张，见用例 ST-016） |

## 测试用例列表

### ST-001：真实格式 checkpoint 加载属性与权重保真

**关联需求：** srs.md §3.1 FR18.1（验收 1）
**测试类型：** 正常路径
**优先级：** High

**前置条件：**
- Given 程序生成的真实格式 fixture（单片与分片两种布局）

**测试步骤：**
1. `load_diffusion_llama` 加载单片与分片 fixture
2. 逐项核对模型属性（层数/头维/GQA 头数/vocab/mask_token_id/RoPE 参数/max_seq_len）
3. safetensors 原始回读与模型权重多重集逐张量 `torch.equal`

**期望结果：**
- Then 属性与 fixture config 逐项一致；权重张量 bitwise 一致（同 dtype 回读）

**实际结果：** 自动化：`tests/test_diffusion_llama.py` P1 分区 `TestLoaderSingleAndSharded` / `TestLoaderAttributes` / `TestLoaderWeightFidelity`（含跨布局 bitwise、bf16→fp32 无损、dtype=auto 保盘上 dtype）全部通过；2026-10-24 实跑 90/90。

**状态：** PASS

---

### ST-002：Llama/Qwen 双命名同一映射覆盖

**关联需求：** srs.md §3.1 FR18.1（验收 2）
**测试类型：** 边界条件
**优先级：** High

**前置条件：**
- Given Dream-7B（Qwen 风格，attention_bias true）与 LLaDA-8B（Llama 风格）两种命名的 fixture

**测试步骤：**
1. 分别以两种 style 生成 fixture 并加载
2. 核对同一权重命名映射（D9 单一映射）与 attention_bias 差异真实落盘

**期望结果：**
- Then 同一映射层覆盖两种命名，均加载成功且属性一致

**实际结果：** 自动化：`TestLoaderStylesAndTie`（双 style + tie 参数化）通过；fixture 侧 `TestFixtureGenAndPresets` 验证 style 差异（attention_bias 权重存在性）真实落盘。

**状态：** PASS

---

### ST-003：加载 fail-fast 定位与零部分加载

**关联需求：** srs.md §3.1 FR18.1（验收 3）
**测试类型：** 异常处理
**优先级：** High

**前置条件：**
- Given 缺权重/形状错/坏索引/缺字段（含 max_position_embeddings）/mask 越界/dtype 非法的 fixture

**测试步骤：**
1. 逐一注入故障并调用 `load_diffusion_llama`
2. 断言 `ValueError` 且消息含缺失项定位（字段名/张量名/文件名）

**期望结果：**
- Then 明确 `ValueError` 且零部分加载状态泄漏

**实际结果：** 自动化：P1 分区 14 个故障注入用例（`TestLoaderErrors` 等）全部命中；校验全前置于构造（函数式加载，失败路径不产生对象）。

**状态：** PASS

---

### ST-004：建模层 golden 对照矩阵（三态 mask × GQA/MHA × RoPE 偏移）

**关联需求：** srs.md §3.2 FR18.2（验收 1）
**测试类型：** 正常路径
**优先级：** High

**前置条件：**
- Given 随机权重小模型（固定种子）+ 手写逐算子参考实现（test 内联 golden）

**测试步骤：**
1. 前向 {全双向, 段因果} × {GQA kv2, MHA kv8} × {偏移 0, 块行偏移} 组合
2. 与参考实现对照（登记容差：KV relL2 <= 1e-5，logits <= 1e-6，argmax 逐位）

**期望结果：**
- Then 全部组合在登记容差内通过

**实际结果：** 自动化：`TestModelingGoldenComparison` 4×组合矩阵通过；容差口径与 design Erratum 7.1 一致（实测 logits 6.6e-7）。

**状态：** PASS

---

### ST-005：段因果分解正确性锚点

**关联需求：** srs.md §3.2 FR18.2（验收 2）
**测试类型：** 正常路径
**优先级：** High

**前置条件：**
- Given 段因果 mask 全序列前向结果

**测试步骤：**
1. 拆分执行：prefix 前向 + `forward_block` 块行前向（prefix_kv 条件）拼接
2. 两级锚点对照（KV 容差层 + logits 层 + argmax 逐位层）

**期望结果：**
- Then 拆分与全序列前向 logits 与 KV 一致

**实际结果：** 自动化：`TestSegmentCausalDecompositionAnchor` 通过（两级锚点全过）。

**状态：** PASS

---

### ST-006：MHA 零特判独立路径

**关联需求：** srs.md §3.2 FR18.2（验收 3）
**测试类型：** 边界条件
**优先级：** Medium

**前置条件：**
- Given MHA 配置（KV 头 == Q 头）

**测试步骤：**
1. MHA 前向并与参考实现对照
2. 确认无 GQA 特判分支（`num_kv_heads == num_heads` 时 `repeat_kv` n_rep=1 自然通过）

**期望结果：**
- Then 独立校验路径通过，GQA 逻辑零干扰

**实际结果：** 自动化：`TestMhaZeroSpecialCase`（含 n_rep==1 路径断言）通过。

**状态：** PASS

---

### ST-007：DiffusionLlama 两模式 generate 管线（17 键 + 页驻留）

**关联需求：** srs.md §3.3 FR18.3（验收 1）
**测试类型：** 正常路径
**优先级：** High

**前置条件：**
- Given DiffusionLlama fixture + 随机 prompt

**测试步骤：**
1. `blockwise_generate` 跑 mask-init 与 draft-verify 两模式
2. 核对 17 键统计契约（与 AR017 程序化提取的键集逐键一致）
3. 核对页驻留不变量（每层 appended_tokens == P+G、rewritten == 0、evictions == 0；GQA store num_heads == num_kv_heads、gather 原生宽度）

**期望结果：**
- Then 正常完成、17 键契约成立、页驻留不变量成立

**实际结果：** 自动化：`TestBlockwiseGenerateDiffusionLlama`（两模式 × 契约/页驻留/GQA store）通过；四臂 CLI 冒烟（ST-013）复证 mask-init/draft-verify 臂页计数器 == P+G == 192。

**状态：** PASS

---

### ST-008：SimpleDiffusionLLM 1214 基线逐位回归

**关联需求：** srs.md §3.3 FR18.3（验收 2）
**测试类型：** 回归测试
**优先级：** High

**前置条件：**
- Given `draft_mode=False` 的 SimpleDiffusionLLM 既有 1214 用例

**测试步骤：**
1. 全量回归（含薄委托 MHA 逐位锚点用例）
2. 核对既有测试文件零修改（除版本串）

**期望结果：**
- Then 零改动逐位通过

**实际结果：** 全量 1304 passed / 1 skipped（1214 基线 + 90 净增）；审查者独立重跑复核一致；git diff 核证 tests/ 除 test_diffusion_llama.py（新增）与 test_ffn_hotpath.py（版本串 3.19.0→3.20.0）外为空。

**状态：** PASS

---

### ST-009：GQA fixture 管线对照与段因果分解

**关联需求：** srs.md §3.3 FR18.3（验收 3）
**测试类型：** 正常路径
**优先级：** High

**前置条件：**
- Given GQA fixture（KV 头 2 < Q 头 8）

**测试步骤：**
1. `blockwise_generate`（页式）与全序列重算 `blockwise_generate_baseline` 对照
2. 核对段因果分解正确性锚点（FR18.2 锚点在管线语境成立）

**期望结果：**
- Then 一致率口径可报告且分解锚点成立

**实际结果：** 自动化：`TestThinDelegationMhaRegression`（MHA 逐位锚点）+ `TestBlockwiseGenerateDiffusionLlama`（GQA store/gather 契约）通过；块式 vs baseline 一致率指标可计算（随机 fixture 权重无数值门槛，S3 >= 0.90 为 s13 场地断言）。

**状态：** PASS

---

### ST-010：remasking 臂闭式收敛与统计完整

**关联需求：** srs.md §3.4 FR18.4（验收 1）
**测试类型：** 正常路径
**优先级：** High

**前置条件：**
- Given DiffusionLlama fixture

**测试步骤：**
1. `remasking_generate` 生成（confidence 与 random 策略）
2. 核对闭式：`passes_used == G // unmask_per_pass`、`mask_residuals == 0`、四统计键完整且与 17 键契约不相交
3. random 策略同种子两跑逐位复现；max_passes 耗尽 residual 闭式（batch 1/2）

**期望结果：**
- Then 有限步内全解掩、无 mask 残留、统计键完整

**实际结果：** 自动化：`TestRemaskingConfidenceConvergence` / `TestRemaskingRandomStrategy` / `TestRemaskingMaxPassesExhaustion` / `TestRemaskingFailFast`（9 项）通过；CLI 冒烟复证 `passes_used == 4`、`mask_residuals == 0`。

**状态：** PASS

---

### ST-011：remasking 臂与块式臂同 prompt 对照指标

**关联需求：** srs.md §3.4 FR18.4（验收 2）
**测试类型：** 正常路径
**优先级：** Medium

**前置条件：**
- Given 三种 mask 语义下的 remasking 臂与块式臂

**测试步骤：**
1. 两臂跑同 prompt，各自正常完成
2. 计算响应区一致率（与 17 键统计/页驻留互证）

**期望结果：**
- Then 各自正常完成且对照指标（一致率）可计算

**实际结果：** 自动化：`TestRemaskingVsBlockwiseComparability` 通过（一致率可计算、无数值门槛——随机权重无语义主张）；EXPERIMENTS.md §7 提供 S3 计算脚本（本机 fixture 实跑返回 0.0，属随机权重预期）。

**状态：** PASS

---

### ST-012：fixture 确定性与真实文件 IO

**关联需求：** srs.md §3.5 FR18.5（验收 1、2）
**测试类型：** 正常路径 + 边界条件
**优先级：** High

**前置条件：**
- Given fixture_gen.py 固定种子；全部 DiffusionLlama 测试

**测试步骤：**
1. 固定种子两次生成，逐文件 bitwise 比对（含分片 index 完备性）
2. 核对全部测试走真实文件 IO（加载/前向/管线无内存假对象）；PRESETS 模板自洽（heads*head_dim==hidden、mask < vocab）

**期望结果：**
- Then 产出 bitwise 一致；测试均基于真实文件 IO

**实际结果：** 自动化：`TestFixtureGenAndPresets`（确定性/双布局 index 完备/预设自洽/预设与 design 表逐字段一致）通过；90 用例全部通过 fixture 生成 + 真实 safetensors 读取路径（零 mock，review C2 维度核证分层与路径）。

**状态：** PASS

---

### ST-013：四臂 CLI 冒烟（s13 命令形态）

**关联需求：** srs.md §3.6 FR18.6（验收 1，命令形态部分）
**测试类型：** 集成测试
**优先级：** High

**前置条件：**
- Given s13 yaml + serve_arm.py + fixture 权重目录

**测试步骤：**
1. 以 `--model <fixture_dir>` 逐臂运行 baseline / mask-init / draft-verify / remasking（CPU、--dtype float32、--prompt-len 64 --gen-len 128）
2. 核验产出（artifacts.pt + stats.json）与不变量（remasking 残零；页计数器 == P+G；baseline kv_write_rows_total == 0）

**期望结果：**
- Then 四臂全部 exit 0 且不变量成立（服务器换真实权重目录即同形命令）

**实际结果：** 2026-10-24 实跑 4/4 exit 0；不变量核验脚本全过（remasking passes_used==4/residuals==0；每层 appended==192/rewritten==0；baseline 语义正确）。

**状态：** PASS

---

### ST-014：EXPERIMENTS.md 可独立复现性

**关联需求：** srs.md §3.6 FR18.6（验收 1，文档完备部分）
**测试类型：** 文档验证（MANUAL + 片段实跑）
**优先级：** High

**前置条件：**
- Given EXPERIMENTS.md（权重获取/环境/命令/取证/S1-S3 判定/落档模板/诚实纪律）+ s13 yaml + serve_arm.py

**测试步骤：**
1. 按 EXPERIMENTS.md §4.1 fixture 冒烟代码实跑（含 §4.2 真实权重加载冒烟片段）
2. 核对文档要素：HF/离线权重获取、PRESETS 核实表与 erratum 程序、四臂命令（2 模型 × 4 臂 × 3 seed）、取证要求、S1-S3 中位判定口径、落档模板全部「待回填」、诚实纪律五条
3. 核对 s13 yaml 复现命令指向 specs/changes/AR018-real-dllm-weights/serve_arm.py、归档 AR017 目录零改动

**期望结果：**
- Then 服务器操作者可按文档独立复现（无缺失步骤）；本机不宣称 GPU 结果

**实际结果：** §4.1/§4.2 片段 2026-10-24 在 fixture 上实跑通过（加载冒烟输出 `2 256 999`，review 轮 1 修复属性名后验证）；文档九节齐备、结果小节全部「待回填」；归档目录 diff 为空（review D3 维度核证）。

**状态：** PASS

---

### ST-015：README 四检查点与版本同步

**关联需求：** srs.md §3.7 FR18.7（验收 1）
**测试类型：** 文档验证
**优先级：** High

**前置条件：**
- Given README.md + 版本四处

**测试步骤：**
1. 核对特性表 v3.20.0 行（含 EXPERIMENTS.md 链接）、版本历史、基准节（fixture 结构/加载/数值口径 + s13 回填占位）、复现方式指引
2. 核对 TOC 零残留、全文无 emoji（程序化扫描）
3. 核对版本 3.20.0 四处一致（pyproject / `__init__` / `__main__` / 版本一致性 UT）

**期望结果：**
- Then 四检查点齐备、无超证据声称（性能主张零出现直至 s13 回填）

**实际结果：** 2026-10-24 程序化验证：emoji 扫描 NONE；`test_ut17_version_strings_consistent` 通过（1/1）；审查者 C3 维度逐项核证（README v3.20.0 行/小节仅结构口径、S1-S3 显式 s13 场地断言、版本四处 diff 核证）；顶部状态表 AR017 遗留 v3.18.0 已修正为 v3.20.0/1304/85.14%。

**状态：** PASS

---

### ST-016：非功能需求验证（零回退 / 零 mock / 门禁 / 诚实纪律）

**关联需求：** srs.md §4 非功能需求（兼容性/真实性/数值正确性/性能主张纪律/质量门禁/版本同步/文档）
**测试类型：** 回归测试 + NFR
**优先级：** High

**前置条件：**
- Given 全仓五件套门禁 + NFR 清单

**测试步骤：**
1. 五件套实跑（black/isort/flake8/mypy/pytest 含覆盖率）
2. 核对：1214 基线逐位回归（ST-008）；机制代码零语义改动（block_decoding/PagedKVStore diff 核证）；零 mock（全真实文件 IO）；覆盖率 >= 84.71% 不回退；本机零性能主张（README/EXPERIMENTS 无性能数字）

**期望结果：**
- Then 五件套全绿；NFR 全部达标

**实际结果：** 2026-10-24 T008 终验 + 审查者独立重跑：black 143 unchanged / isort 0 / flake8 0 / mypy 78 files 0 / pytest 1304 passed + 1 skipped + 85.14%（+0.43pp）；gates.md 落档四表；诚实纪律由 review C3 维度逐项核证（「省 4-8x 页内存」为 GQA 头数算术事实非本机性能主张）。

**状态：** PASS

---

## ST 执行报告

| 字段 | 内容 |
|------|------|
| 执行日期 | 2026-10-24 |
| 执行结果 | PASS |
| 执行轮次 | 第 1 轮 |

### 需求覆盖矩阵

| 需求 ID | 需求描述 | 测试用例 | 结果 |
|--------|---------|---------|------|
| §3.1 FR18.1 | 真实 checkpoint 加载器 | ST-001, ST-002, ST-003 | PASS |
| §3.2 FR18.2 | 建模层与数值对照 | ST-004, ST-005, ST-006 | PASS |
| §3.3 FR18.3 | 块式管线契约与 generate 接入 | ST-007, ST-008, ST-009 | PASS |
| §3.4 FR18.4 | remasking 基线臂 | ST-010, ST-011 | PASS |
| §3.5 FR18.5 | fixture 与真实性验证 | ST-012 | PASS |
| §3.6 FR18.6 | s13 协议与 EXPERIMENTS.md | ST-013, ST-014 | PASS |
| §3.7 FR18.7 | README 四检查点与版本同步 | ST-015 | PASS |
| §4 NFR | 非功能需求 | ST-016 | PASS |

**需求覆盖率：** 8 / 8（100%）

### 测试执行汇总

| 类型 | 总计 | 通过 | 失败 | 阻塞 |
|------|------|------|------|------|
| 正常路径 | 8 | 8 | 0 | 0 |
| 边界条件 | 3 | 3 | 0 | 0 |
| 异常处理 | 1 | 1 | 0 | 0 |
| 集成测试 | 1 | 1 | 0 | 0 |
| 回归测试 / NFR | 3 | 3 | 0 | 0 |
| **合计** | **16** | **16** | **0** | **0** |

自动化基础：全量 pytest 1304 passed / 1 skipped / 85.14%（2026-10-24 实跑，审查者独立复核一致）；四臂 CLI 冒烟 4/4 + 不变量核验；EXPERIMENTS.md §4 片段实跑。

### 遗留问题

| 严重性 | 描述 | 处理方式 |
|-------|------|---------|
| Minor | design 6.1「零部分加载（失败后无对象产生）」为结构性保证（函数式加载 + 校验全前置），14 个错误测试隐式覆盖，无专项显式断言 | 记录延后（review 轮 1 已落档 tasks.md） |
| Minor | baseline 臂在 DiffusionLlama 上无专项 pytest（AR016 golden + MHA 逐位锚点 + 四臂 CLI 冒烟 4/4 取证） | 记录延后（review 轮 1 已落档 tasks.md） |
| 待办（非缺陷） | PRESETS 文献口径待 s13 服务器以真实 config.json 核实（出入落 erratum）；S1-S3 场地断言待服务器实测回填 EXPERIMENTS.md §8 与 README 基准表 | 服务器侧执行（EXPERIMENTS.md §2.3/§9 程序已定义） |

### 结论

> **Go（建议）**：16/16 用例 PASS，需求覆盖 8/8（100%）；无 Critical/Major 缺陷；全量 1304 passed / 1 skipped 零回归，覆盖率 85.14% 不回退；两项 Minor 为结构性保证已落档；S1-S3 属服务器场地断言，本 AR 交付物（加载/建模/管线/remasking/s13 协议/文档）全部验收通过。本机零性能主张纪律贯穿全部交付物。
