# [AR018] 需求设计说明书：真实 dLLM 权重执行平面（DiffusionLlama）

| 字段 | 内容 |
|------|------|
| AR 编号 | AR018 |
| AR 主题 | 真实 dLLM 权重执行平面：自研 DiffusionLlama（Llama/Qwen 架构 + 双向 attention + GQA + RoPE），直接加载真实 LLaDA-8B / Dream-7B checkpoint，接入 AR016/017 块式分页 KV + draft-verify 全管线 |
| 关联 SR | ROADMAP.md §4 创新提案 #5（Speculative Decoding for Diffusion LLM）；承接 AR016（块式精确前缀 KV）/ AR017（MTP draft-verify）；Fast-dLLM 对标收官件 |
| 日期 | 2026-10-24 |
| 状态 | Draft |

## 1. 背景与目标

### 1.1 痛点与机会（代码级与 SOTA 实证）

- **机制已完成、载体是玩具**：AR016（块式精确前缀分页 KV）与 AR017（draft-verify 草拟-验证）在 v3.19.0 已全量交付（1214 用例、17 键统计契约、四臂基准），但执行载体是随机权重 `SimpleDiffusionLLM`。s12 协议的场地断言 S1（draft-verify vs mask-init 端到端 >= 1.5x）、S2（轮数削减 >= 2x）、S3（token 一致率 >= 0.90）全部悬置，等待真实权重（AR017 bench-results.md §2 明文移交本 AR）。
- **SOTA 收益全部来自真实权重**：Fast-dLLM（NVIDIA/MIT/HKU）的 LLaDA 27.6x 发表证据、Jacobi 式块内 O(1)-O(3) 轮收敛、草稿质量（acceptance）全部是预训练模型属性。本仓库机制对标已完成，唯一空白是真实 dLLM 权重执行平面。
- **关键实测发现（2026-10-24，本机验证）**：本机 transformers 5.15.1 **不存在 LLaDA / Dream 模型类**（无 `llada`/`dream` 模型目录、无 remask 生成机制、CONFIG_MAPPING 无对应条目；`diffllama` 为微软 Differential Transformer，同名无关）。5.x 的 dLLM 一等支持是 **DiffusionGemma**（Gemini Diffusion，encoder-decoder canvas 架构，`generation_diffusion_gemma.py`：外层 canvas 半 AR 循环 + 内层去噪 + EntropyBoundSampler + 编码器 KV 复用），架构与 decoder-only 的 LLaDA/Dream 不同，无法作为加载通道，仅作语义参照。**结论：AutoModel 路线不可行，「transformers >= 4.57 一等支持 LLaDA/Dream」的早前假设作废。**
- **自研执行平面 = 精确前缀 KV 的前提**：Fast-dLLM 用 dual-cache 近似（生成态/去噪态双份 KV 部分共享）+ 定制 CUDA attention kernel；本仓库的精确前缀 KV（零近似零漂移）要求对 attention 计算与 KV 注入有完全控制权——自研建模层是唯一能干净落地的路线（用户已定案，四问四答见 §2）。
- **GQA 缺口**：真实权重（LLaDA-8B 继承 LLaMA-3-8B：32 层 / 32 Q 头 / 8 KV 头；Dream-7B 继承 Qwen2.5-7B 口径：28 层 / 4 KV 头——设计期以真实 config.json 核实）均为 **GQA（KV 头数 < Q 头数）**；现有管线与 `PagedKVStore` 假设 MHA（`hidden_dim == num_heads * head_dim` 硬校验）。GQA 支持是本 AR 的硬需求。

### 1.2 目标

在 ShadowInfer 内自研 **DiffusionLlama 执行平面**并接入既有块式管线：

- **真实 checkpoint 加载**：解析真实 `config.json` + `safetensors` 权重（单文件与 `model.safetensors.index.json` 分片两种布局），Llama（LLaDA-8B）与 Qwen（Dream-7B）双命名映射，dtype 可配（checkpoint 原生 bf16 / fp32），形状与完整性 fail-fast 校验；
- **建模层**：RMSNorm + RoPE + SwiGLU FFN + GQA（repeat_kv 展开）+ **双向 attention**（dLLM 语义：response 各位可见全部 prompt 与块内全部位；块间段因果，与 Fast-dLLM 块式半 AR 口径一致）+ mask token；
- **块式管线契约**：实现 `forward_step`（prefill / 全序列重算，支持 `attention_mask`）与 `forward_block`（前缀 KV 条件 + 块内双向，K/V 返回契约与 AR016/017 管线一致）；`generate` 管线从 `SimpleDiffusionLLM` 具体类型协议化解耦，DiffusionLlama 零改动接入；
- **dLLM 原生生成基线臂**：低置信 remasking 全序列重算调度（LLaDA/Dream 官方生成语义），作为四臂对照的真 baseline；
- **本地验证（用户确认口径）**：微型真实格式 fixture checkpoint（程序自生成 tiny 尺寸 config.json + safetensors，真实命名/IO/解析/前向路径）+ 数值对照（与逐位手写参考实现容差/bitwise 对照）；
- **服务器复现**：s13 GPU 协议 + **EXPERIMENTS.md 独立文档**（权重获取、环境、四臂命令、S1-S3 判定、落档模板）+ `serve_arm.py --model` 真实权重路径支持；
- **README 四检查点 + 版本 3.20.0 四处同步**。

### 1.3 SOTA 对照与差异化

| 项目 | 机制 | 本 AR 差异 |
|---|---|---|
| Fast-dLLM（NVIDIA） | dual-cache 近似 KV + 定制 CUDA attention kernel + PMTG | 精确前缀分页 KV（零近似零漂移，AR016 已证）；纯 PyTorch 可移植执行平面（CPU/GPU 同码）；统计/预注册纪律 |
| transformers 5.x DiffusionGemma | encoder-decoder canvas 扩散 + EntropyBoundSampler + Cache 类复用 | decoder-only LLaDA/Dream 通用平面；块式管线契约直连（forward_block 前缀 KV 注入）；不依赖 HF 模型类版本 |
| HF remote code（LLaDA 官方仓库 modeling.py） | trust_remote_code 加载 | 自研平面版本免疫、attention 全控制权、GPU 服务器无需联网拉远程代码（用户已否决该路线） |
| dLLM-serve | 块式调度服务化 / overlap 流水 | Out of scope（未来 AR） |
| AR016/017（本仓库） | 玩具模型块式管线 | 执行平面泛化到真实权重；机制代码（调度/统计/页存储）零语义改动 |

## 2. 需求范围

| 项 | 内容 |
|---|---|
| AR 编号 | AR018 |
| 版本 | v3.20.0 |
| 影响面 | `shadowinfer/models/diffusion_llama.py`（新增：加载器 + 建模层 + 契约实现）、`shadowinfer/models/__init__.py`（导出）、`shadowinfer/execution/block_decoding.py`（模型类型协议化解耦，调度语义零改动）、`shadowinfer/core/backends/pytorch_backend.py`（generate 接入 GQA 感知的 KV 头处理）、`tests/test_diffusion_llama.py`（新增）、`tests/test_block_decoding.py`（协议化回归）、`configs/experiments/s13_real_dllm_gpu.yaml`（新增）、`EXPERIMENTS.md`（新增）、`specs/changes/AR018-real-dllm-weights/fixture_gen.py`（fixture 生成器）、`specs/changes/AR018-real-dllm-weights/serve_arm.py`（**新建**，沿 AR017 生长期惯例复制扩展 `specs/archive/AR017-mtp-draft-verify-decoding/serve_arm.py`，不改动归档文件）、README.md、版本号同步 |
| 不破坏对外 API | `SimpleDiffusionLLM` 全部路径逐位回归（1214 基线不回退）；`BlockDecodingConfig` 字段零改动；`generate`/`forward_block`/`forward_step` 既有签名不变；`PagedKVStore` 公开 API 不变（GQA 处理在装配层，不改页布局/驱逐语义） |

**In Scope（本 AR 要做的）：**
- 真实 checkpoint 加载器（config.json + safetensors 单片/分片、Llama/Qwen 双命名、dtype 可配、fail-fast）；
- DiffusionLlama 建模层（RMSNorm/RoPE/SwiGLU/GQA/双向 attention/mask token）与数值正确性对照；
- 块式管线契约（forward_step/forward_block/属性）与 generate 协议化接入（含 GQA KV 头装配）；
- dLLM 原生 remasking baseline 臂（全序列重算 + 低置信重掩，官方语义）；
- 微型真实格式 fixture 生成器 + 真实性/数值对照测试；
- s13 GPU 协议 + EXPERIMENTS.md + serve_arm --model 扩展；
- README 四检查点 + 版本 3.20.0 四处同步。

**Out of Scope（本 AR 不做的）：**
- CUDA fused attention kernel / FlashAttention 接线（Fast-dLLM 的 GPU kernel 层，纯 PyTorch 实现已满足机制验证与 s13 对照）；
- 服务化与 overlap 流水（dLLM-serve 范畴，未来 AR）；
- 自研 tokenizer（实验用 `AutoTokenizer`（本地 transformers 可用，不依赖模型类）；文本层不是执行平面核心）；
- LLaDA-V 多模态、训练/蒸馏/微调；
- 置信度调度与 draft-verify 的语义改动（AR016/017 机制零改动，只换执行平面）；
- 本机 GPU 性能主张（本机无 GPU；性能/质量主张全部 s13 服务器口径）。

## 3. 功能需求

### 3.1 FR18.1 真实 checkpoint 加载器

**描述：** `diffusion_llama.py` 提供 `load_diffusion_llama(path, dtype=..., device=...)`：从本地目录解析真实 HF 格式 checkpoint。

**触发条件：** 任何 DiffusionLlama 构造入口（fixture 与真实权重同一代码路径）。

**期望行为：**
- `config.json` 解析：Llama 命名（`hidden_size/num_hidden_layers/num_attention_heads/num_key_value_heads/intermediate_size/rope_theta/vocab_size/max_position_embeddings` 等）与 Qwen 命名兼容（Dream-7B 主口径）；`mask_token_id` 等扩散特有字段透传；未知字段忽略但记录；
- `max_seq_len` 推导：`max_position_embeddings`（Qwen 命名同字段）→ 模型属性 `max_seq_len`；**字段缺失时 fail-fast**（与整体校验口径一致），错误消息指明缺失字段名；
- 权重布局两种：`model.safetensors`（单片）与 `model.safetensors.index.json` + `model-000xx-of-000xx.safetensors`（分片，Dream-7B/LLaDA-8B 实际形态）；权重命名 `model.layers.N.self_attn.{q,k,v,o}_proj.weight`、`model.norm.weight`、`lm_head.weight`（含 `tie_word_embeddings` 两种口径）；
- dtype：`auto`（随 checkpoint，bf16 原样）与 `float32`（CPU 本地与数值对照口径）；
- fail-fast：权重缺失/多余、形状与 config 不符、mask_token_id 不在词表区间、分片索引与文件不一致 → 明确 `ValueError` 且零部分加载状态泄漏。

**验收标准：**
- Given 程序生成的真实格式 fixture（单片与分片两种），When 加载，Then 属性（层数/头维/GQA 头数/vocab/mask_token_id/RoPE 参数/max_seq_len ← `max_position_embeddings`）与 fixture config 逐项一致，权重张量 bitwise 一致（同 dtype 回读）；
- Given Dream-7B 命名与 LLaDA-8B 命名两种 fixture，When 加载，Then 同一映射层覆盖两种命名；
- Given 缺权重/形状错/坏索引的 fixture，When 加载，Then `ValueError` 消息含缺失项定位。

### 3.2 FR18.2 DiffusionLlama 建模层与数值对照

**描述：** Llama/Qwen 架构 dLLM 建模：RMSNorm、RoPE、SwiGLU、GQA attention、双向/段因果 mask 语义。

**期望行为：**
- attention 语义三态：(a) 全序列双向（dLLM 原生前向：全 token 互见）；(b) 段因果（块式管线：prefix 各行互见 + 当前块各行可见 prefix 与块内全部行，不可见未来块——与 AR016 `segment_causal_mask` 语义同构）；(c) 块行条件前向（forward_block：只算块行，以冻结 prefix K/V 为条件）；
- GQA：`num_key_value_heads < num_attention_heads` 时 KV 经 repeat_kv 展开参与 attention；MHA 权重（KV 头 == Q 头）零特判通过；
- RoPE：`rope_theta` 可配，位置偏移（forward_block 块行位置 = `[prefix_len, prefix_len+W)`）正确；
- 数值对照：以独立手写逐位参考实现（test 内联，非生产代码路径）为 golden，随机小权重固定种子下 logits 容差对照（fp32 口径 atol/rtol 显式登记）；GQA/MHA、双向/段因果、有无 RoPE 偏移各覆盖。

**验收标准：**
- Given 随机权重小模型 + 固定种子输入，When 前向（三态 mask × GQA/MHA），Then 与参考实现对照全部通过（登记容差内）；
- Given 段因果 mask 前向 vs 拆分（prefix 前向 + forward_block 块行前向）拼接，Then logits 与 KV 一致（块式分解的正确性锚点）；
- Given MHA 配置（KV 头 == Q 头），When 前向，Then 与 SimpleDiffusionLLM 无关的独立校验路径通过（GQA 逻辑零干扰）。

### 3.3 FR18.3 块式管线契约与 generate 接入

**描述：** DiffusionLlama 实现块式管线模型契约，`generate` 管线协议化解耦并接入。

**期望行为：**
- 契约：属性 `num_layers/num_heads/head_dim/max_seq_len/vocab_size/device`（GQA 展开视角）+ `forward_step(x, step_t, total_steps, kv_cache=None, attention_mask=None)`（prefill 与全序列重算；dLLM 无 timestep 语义，`step_t/total_steps` 接受但忽略）+ `forward_block(x_block, prefix_kv, prefix_len, step_t=0, total_steps=0)`（返回 `(logits [B,W,V], block_kv)`，`block_kv[layer] = (k,v)` 契约形状）；
- **forward_step 返回三元组** `(logits [B,T,V], kv, attn_weights)`，与既有管线解包口径逐点一致（`block_decoding.py` baseline 轮取 `logits`、`pytorch_backend.py` prefill 取 `kv`）：`kv` 为 `{layer_id: (k, v)}`，形状 `[B, H_*, T, D]`（`H_*` 为 KV 头原生数或展开后 Q 头数，与下条 GQA 装配口径联动、设计期定案并落档 design.md；同一模型内全部层与 forward_block 的 `block_kv` 必须同一口径）；`attn_weights` 允许为 `None`（管线既有调用点均不解包第三个元素的内容语义，仅占位对齐三元组）；
- `block_decoding.py`/`pytorch_backend.py` 的模型类型注解从 `SimpleDiffusionLLM` 放宽为结构化协议（Protocol/鸭子类型），调度与统计代码零语义改动；
- GQA 装配：页存储以 KV 头原生存（`PagedKVStore` 构造参数传 KV 头数）或装配层展开——设计期定案，约束：不改 `PagedKVStore` 公开 API 与页布局语义，MHA 路径逐位回归；
- `PyTorchModelBackend`（或新后端工厂入口）可用 DiffusionLlama 跑 `generate` 全管线（mask-init 与 draft-verify 两模式）；
- `SimpleDiffusionLLM` 路径 1214 基线逐位回归。

**验收标准：**
- Given DiffusionLlama fixture + 随机 prompt，When `generate`（两模式），Then 正常完成、17 键统计契约成立、页驻留不变量成立（轮间零页写、append P+G）；
- Given `draft_mode=False` 的 SimpleDiffusionLLM 既有用例，When 全量回归，Then 零改动逐位通过；
- Given GQA fixture（KV 头 < Q 头），When generate 与全序列重算 baseline 对照，Then 一致率口径报告且段因果分解正确性（FR18.2 锚点）成立。

### 3.4 FR18.4 dLLM 原生生成基线臂（remasking）

**描述：** 真实 dLLM 官方生成语义的 baseline：低置信 remasking 全序列重算调度（LLaDA/Dream 原生口径），作为四臂对照的真 baseline。

**期望行为：**
- 调度：response 区初始化全 mask → 每轮全序列前向（真实双向语义）→ 按置信度解掩固定数量 token（low-confidence remasking 口径，每轮解掩步长可配）→ 未解掩位重置为 mask → 至全部解掩或步数上限；
- 与既有 `blockwise_generate_baseline`（AR 的全量重算臂）并列，命名与 stats 口径明确区分；输出 token 序列 + 逐轮解掩统计；
- 该臂供 s13 四臂对照（remasking baseline / blockwise mask-init / draft-verify / 逐 token AR 参考（可选）），本机 fixture 上做结构正确性与收敛性测试（有限步内全解掩）。

**验收标准：**
- Given DiffusionLlama fixture，When remasking 生成，Then 有限步内全解掩、无 mask 残留、统计键完整；
- Given 三种 mask 语义（全双向/段因果/块行条件），When remasking 臂与块式臂跑同 prompt，Then 各自正常完成且对照指标（一致率）可计算。

### 3.5 FR18.5 微型真实格式 fixture 与真实性验证

**描述：** fixture 生成器（`specs/changes/AR018-real-dllm-weights/fixture_gen.py`）产出 tiny 尺寸真实格式 checkpoint；测试全走真实代码路径。

**期望行为：**
- 生成物：`config.json`（真实字段命名与结构）+ `model.safetensors`（单片）或分片布局（含 index json）；参数化：层数/头数/KV 头数/head_dim/vocab/mask_token_id/rope_theta/命名风格（llama/qwen）/tie_word_embeddings/dtype；固定种子确定性生成；
- LLaDA-8B 与 Dream-7B 的真实参数在生成器中作为**预设模板**（带「设计期以真实 config.json 核实」标注），fixture 默认 tiny 尺寸；
- 测试断言用结构/闭式/取值域关系（沿仓库惯例），数值对照锚点见 FR18.2；
- 生成器与测试不 mock 任何生产路径（加载/前向/管线全走真实代码）。

**验收标准：**
- Given fixture_gen.py 固定种子，When 两次生成，Then 产出 bitwise 一致（可复现）；
- Given 全部 DiffusionLlama 测试，When 运行，Then 均基于真实文件 IO（无内存假对象充当 checkpoint）。

### 3.6 FR18.6 s13 GPU 服务器协议与 EXPERIMENTS.md

**描述：** `configs/experiments/s13_real_dllm_gpu.yaml` + `EXPERIMENTS.md`（独立文档，README 链接）+ `serve_arm.py --model` 扩展。

**期望行为：**
- s13：真实权重四臂协议（Dream-7B 主验收、LLaDA-8B 次之），`model_weights`（本地权重目录路径）/`mask_token_id`/块宽/max_rounds/remasking 步长可配，`--seed` 复现命令，S1-S3 场地断言判定口径写入；
- EXPERIMENTS.md：权重获取（HF 下载或离线传输到服务器）、环境（CUDA/torch/依赖版本）、四臂运行命令、显存/页统计取证、S1-S3 判定标准、结果落档模板、诚实纪律（未达标如实标注）；
- serve_arm.py：**AR018 新建 `specs/changes/AR018-real-dllm-weights/serve_arm.py`**（复制扩展 AR017 归档版，归档文件零改动、s12 旧协议引用不受影响），`--model <path>` 指向真实 checkpoint 目录（替代 AR017 的 `model_weights: null` 占位），四臂均可指定；s13 yaml 的复现命令指向该新路径；
- 本机不宣称任何 GPU 结果；EXPERIMENTS.md 标注「服务器实测后回填」。

**验收标准：**
- Given s13 yaml + EXPERIMENTS.md + serve_arm.py，Then 服务器操作者可按文档独立复现（环境/命令/判定/模板齐备，无缺失步骤）。

### 3.7 FR18.7 README 四检查点与版本同步

**描述：** README 四检查点更新 + 版本 3.20.0 四处同步。

**期望行为：**
- 特性表 v3.20.0 行（真实 dLLM 权重执行平面、DiffusionLlama、GQA、双向/段因果语义、EXPERIMENTS.md 链接）；
- 基准节：本机 fixture 口径只写结构正确性/加载真实性（不写性能）；性能对照表标注「s13 服务器实测后回填」；
- 版本历史 v3.20.0；TOC 锚点零残留；无 emoji；
- 版本 3.20.0 同步 `pyproject.toml`、`__init__.py`、`__main__.py`、`test_ffn_hotpath`（UT18 版本一致性用例）。

**验收标准：**
- Given README，Then 四检查点齐备、无超证据声称（fixture 尺寸主张仅结构/加载/数值对照；性能主张零出现直至 s13 回填）。

## 4. 非功能需求

| 类型 | 指标 | 要求 |
|------|------|------|
| 兼容性 | 既有路径零回退 | SimpleDiffusionLLM 全路径逐位回归（1214 基线）；`BlockDecodingConfig`/`PagedKVStore` 公开 API 零改动 |
| 真实性 | 零 mock | 加载/前向/管线全走真实文件与真实代码路径；fixture 为合法集成夹具（真实格式与命名，仅尺寸缩小，用户已确认口径）；无 monkeypatch 伪造数据路径（故障注入仅限异常测试） |
| 数值正确性 | golden 对照 | 建模层与手写参考实现容差对照（fp32、容差登记）；段因果分解 vs 全序列前向一致性锚点 |
| 性能主张纪律 | README/文档 | 本机零性能主张（无 GPU）；性能/质量主张全部 s13 服务器口径；未验证只给「机制说明（未验证）」口径 |
| 质量门禁 | 五件套 | black/isort/flake8/mypy/pytest 全绿；覆盖率 >= 84.71%（v3.19.0 基线不回退）；总用例 >= 1214 |
| 版本同步 | 3.20.0 | pyproject.toml、`__init__.py`、`__main__.py`、版本一致性 UT 四处 |
| 文档 | README + EXPERIMENTS.md | 特性表、版本历史 3.20.0、基准表口径诚实、EXPERIMENTS.md 服务器手册齐备、TOC 零残留、无 emoji |

## 5. 约束与假设

**约束：**
- **架构参数零硬编码**：Dream-7B（Qwen2.5-7B 初始化口径：28 层 / GQA 4 KV 头 / vocab 151936 / mask_token_id 151669 / rope_theta 1000000——文献口径）与 LLaDA-8B（LLaMA-3-8B 初始化口径：32 层 / GQA 8 KV 头 / vocab 126464 / mask_token_id 126336）的真实参数**必须**在设计期以真实 `config.json` 核实（GPU 服务器下载或用户提供文件；本机网关已实测拦截 HF/arXiv/GitHub 全部通道），核实结果落档 design.md；核实前 fixture 预设模板以「文献口径，待核实」标注，代码完全 config 驱动；
- **块式臂是全双向 dLLM 的半 AR 近似**（与 Fast-dLLM 块式口径相同）：早期块不可见后期块，与全序列重算 baseline 的 token 一致率是一等指标（S3），README/文档不得伪称等价；
- **GQA 扩展不得破坏 MHA**：`PagedKVStore` 页布局/驱逐/双账本语义零改动，GQA 处理在装配层；SimpleDiffusionLLM 逐位回归是硬门；
- **dLLM 无 timestep 语义**：`forward_step/forward_block` 契约保留 `step_t/total_steps` 参数（管线兼容）但 DiffusionLlama 忽略之，文档明示；
- 机制代码（draft-verify/mask-init 调度、17 键统计、页存储）零语义改动——本 AR 只加执行平面，不改机制；
- RoPE 实现必须支持任意 `rope_theta` 与位置偏移（块行条件前向的连续性），预计算频率缓存。

**假设：**
- Dream-7B / LLaDA-8B 权重可在 GPU 服务器获取（HF 直连或离线传输），config.json 与权重命名与 HF 仓库公开口径一致；
- 7B/8B bf16 推理显存在单卡（>= 24GB）可承受，纯 PyTorch attention（非 fused kernel）在服务器 GPU 上可运行（速度非本 AR 验收门槛，正确性与 S1-S3 对照才是）；
- fixture tiny 尺寸（如 4 层 / hidden 256 / GQA 8:2 / vocab 1000）下随机权重的 remasking/块式调度收敛性与统计契约可测（结构正确性口径，无语义质量主张）。

## 6. 术语说明

| 术语 | 定义 |
|---|---|
| 执行平面（execution plane） | 模型权重加载 + 前向计算的完整真实路径（区别于调度/统计机制层） |
| DiffusionLlama | 本 AR 自研的 Llama/Qwen 架构 dLLM 建模层（双向 attention + mask token + GQA + RoPE） |
| GQA / repeat_kv | 分组查询注意力：KV 头数 < Q 头数；KV 头沿头维重复展开以参与 attention |
| 双向 attention（dLLM 语义） | response 各位可见全部 prompt 位与 response 内全部位（无因果下三角掩码） |
| 段因果（segment-causal） | 块式半 AR 口径：prefix 内互见 + 当前块可见 prefix 与块内全部位 + 不可见未来块 |
| remasking | dLLM 原生生成：每轮全序列前向后仅解掩高置信位，低置信位重置为 mask token |
| fixture checkpoint | 程序生成的 tiny 尺寸真实格式 checkpoint（config.json + safetensors，真实命名与 IO 路径） |
| 分片布局 | `model.safetensors.index.json` + 多个 `model-000xx-of-000xx.safetensors` 的 HF 权重组织形式 |
| S1-S3 场地断言 | s12 预注册的服务器断言（端到端 >= 1.5x / 轮数 >= 2x / 一致率 >= 0.90），s13 真实权重下判定 |
