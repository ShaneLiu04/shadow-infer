# [AR018] 任务分解：真实 dLLM 权重执行平面（DiffusionLlama）

> 需求来源：`specs/changes/AR018-real-dllm-weights/srs.md`（用户确认版）
> 设计来源：`specs/changes/AR018-real-dllm-weights/design.md`（用户确认版，方案 A + 设计决策 D1-D15）
> 纪律：零 mock（全真实文件 IO 与代码路径）；测试断言用结构/闭式/取值域关系 + 数值 golden 对照；
> 逐任务一提交（message 含【AR018】）；标准任务 TDD（Red 子代理写测试 → Green 主代理实现）；
> 校准/文档/门禁任务主代理直做；SimpleDiffusionLLM 1214 基线逐位回归是每个 Green 的硬门。

| 编号 | 来源 | 任务 | 产出 | 验收标准 | 执行方式 |
|---|---|---|---|---|---|
| T001 | FR18.5 | **fixture 生成器**（测试基建，先于一切）：`specs/changes/AR018-real-dllm-weights/fixture_gen.py`，参数化生成 tiny 真实格式 checkpoint（config.json + safetensors 单片 / 分片+index），支持 llama/qwen 双命名、GQA/MHA、tie_word_embeddings、dtype、固定种子；含 Dream-7B / LLaDA-8B 预设模板（文献口径，标注待核实） | fixture_gen.py + 测试可复用的生成入口 | 固定种子两次生成 bitwise 一致；两种命名/两种布局/GQA/MHA 均可生成；预设模板字段与 srs §5 文献口径一致 | 主代理直做 |
| T002 | FR18.1 | **真实 checkpoint 加载器**：`load_diffusion_llama(path, dtype, device)`——config.json 解析（Llama/Qwen 双命名、mask_token_id、rope_theta）、单片/分片权重加载、tie_word_embeddings、dtype 可配、fail-fast（缺权重/形状不符/坏索引/词表外 mask_token_id） | `shadowinfer/models/diffusion_llama.py`（加载部分）+ tests/test_diffusion_llama.py（加载 UT） | srs FR18.1 三条验收全部通过；异常路径零部分加载状态泄漏 | 标准 TDD |
| T003 | FR18.2 | **建模层与数值对照**：RMSNorm/RoPE（任意 theta + 位置偏移）/SwiGLU/GQA（repeat_kv）/三态 mask（全双向/段因果/块行条件）；手写参考实现 golden 对照（fp32 容差登记）；段因果分解 vs 全序列前向一致性锚点 | diffusion_llama.py（建模部分）+ UT | srs FR18.2 三条验收全部通过（三态 mask × GQA/MHA × RoPE 偏移对照矩阵） | 标准 TDD |
| T004 | FR18.3 | **块式管线契约与 generate 接入（含 D3 重构）**：DiffusionLlama 实现 forward_step（含 attention_mask、kv_cache 显式拒绝）/forward_block/属性契约；`pytorch_backend.py::generate` 编排体抽取为 `execution/blockwise_generate.py::blockwise_generate` 纯函数（cut-paste 零逻辑改动，backend 薄委托保持 `_last_generate_stores` 语义）；block_decoding.py 类型注解放宽为 BlockwiseModel 协议（调度/统计零语义改动）；GQA KV 头装配（store 以 `getattr(model, "num_kv_heads", model.num_heads)` 构造，PagedKVStore 公开 API 与页布局零改动）；DiffusionLlama 跑通两模式（17 键统计 + 页驻留不变量） | 契约实现 + blockwise_generate.py 纯函数 + 薄委托 + 协议化解耦 + test_block_decoding.py 回归 + UT | srs FR18.3 三条验收全部通过；SimpleDiffusionLLM 全量 1214 逐位回归零改动通过（重构锚点：薄委托与纯函数结果 bitwise 一致） | 标准 TDD |
| T005 | FR18.4 | **remasking baseline 臂**：dLLM 原生生成语义（全序列重算 + 低置信解掩 + 重置 mask，步长可配），输出序列 + 逐轮统计；与块式臂同 prompt 对照指标可计算 | remasking 调度实现 + UT | srs FR18.4 两条验收全部通过（有限步全解掩、无 mask 残留、统计完整） | 标准 TDD |
| T006 | FR18.6 | **s13 协议 + EXPERIMENTS.md + serve_arm 扩展**：`configs/experiments/s13_real_dllm_gpu.yaml`（四臂、model_weights、S1-S3 判定口径）；EXPERIMENTS.md 独立文档（权重获取/环境/命令/取证/落档模板，服务器实测后回填标注）；**新建** `specs/changes/AR018-real-dllm-weights/serve_arm.py`（复制扩展 AR017 归档版，归档零改动）支持 `--model` 真实权重路径 | s13 yaml + EXPERIMENTS.md + specs/changes/AR018-real-dllm-weights/serve_arm.py | 按 EXPERIMENTS.md 可独立复现（无缺失步骤）；s13 yaml 复现命令指向新 serve_arm.py；本机不宣称 GPU 结果 | 主代理直做 |
| T007 | FR18.7 | **README 四检查点 + 版本 3.20.0**：特性表 v3.20.0 行；基准节 fixture 口径只写结构/加载/数值正确性（零性能主张，s13 回填占位）；版本历史；EXPERIMENTS.md 链接；TOC 零残留无 emoji；版本四处同步（pyproject/`__init__.py`/`__main__.py`/版本一致性 UT） | README.md + 四处版本 | 四检查点齐备；无超证据声称；版本一致性 UT 通过 | 主代理直做 |
| T008 | §4 | **质量门禁全绿**：black/isort/flake8/mypy/pytest 五件套；覆盖率 >= 84.71% 不回退；总用例 >= 1214 净增 | 门禁实跑记录（落档本 AR 目录） | 五件套全绿；覆盖率与用例数以当次实跑为准登记 | 主代理直做 |

## 执行纪律

1. **TDD 链**：T002-T005 为标准任务——Red 阶段派发独立子代理按 srs 验收标准写测试（允许 import 未实现符号失败），Green 阶段主代理实现至全绿；每任务 Green 后跑 SimpleDiffusionLLM 回归子集，全量回归留 T008 统一取证；
2. **提交**：逐任务一提交，message 含【AR018】与任务号（沿仓库惯例 `feat(code): ...【AR018】T00x ...`）；
3. **架构参数核实**：Dream-7B / LLaDA-8B 真实 config.json 的获取与核实属设计期任务（srs §5 约束）；若核实结果与文献口径有出入，修订 fixture 预设模板并在 design.md 落档 erratum，代码 config 驱动不受影响；
4. **诚实纪律**：本机零性能主张；fixture 一致率/收敛性只作结构正确性口径；任何「优于 baseline」表述在 s13 服务器实测落档前禁止出现在 README/EXPERIMENTS.md；
5. **零 mock 红线**：故障注入仅限异常测试；生产路径禁止测试专用分支。

## 状态

| 编号 | 状态 | 提交 |
|---|---|---|
| T001 | passing | 2026-10-24 完成（验收自验 36/36：确定性/双 style/双布局/GQA/MHA/tie/bf16/预设一致性/异常路径） |
| T002 | pending | - |
| T003 | pending | - |
| T004 | pending | - |
| T005 | pending | - |
| T006 | pending | - |
| T007 | pending | - |
| T008 | pending | - |

## 进度记录

### 2026-10-24 会话记录（需求 + 设计 + T001）

- 完成阶段：requirements（srs.md + tasks.md，门控 FAIL→修复→复审 PASS）、design（方案 A + D1-D15，门控三轮 PASS：D7 两级锚点修订、PRESETS 补 head_dim、§3 旧口径清除）、T001
- 完成任务：T001 fixture 生成器（主代理直做）
- 修改文件：specs/changes/AR018-real-dllm-weights/{srs.md, tasks.md, design.md, fixture_gen.py}
- T001 验收：固定种子两次生成 bitwise 一致；llama/qwen 双 style（attention_bias 差异真实落盘）；单片/分片双布局（index 覆盖完备）；GQA/MHA 均可生成；tie 省略 lm_head；bf16 dtype；PRESET_TEMPLATES 与 design 4.2.2 文献口径表逐字段一致且内部自洽；异常路径 9 项 ValueError 全部命中
- 备注：PRESETS 为文献口径（本机网关拦截 HF/arXiv/GitHub，无法核实），s13 服务器核实后如有出入落 erratum；fixture 正式 pytest 用例随 T002 Red 阶段交付（design 6.1 P5 分区）
