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
| T002 | passing | 2026-10-24 完成（34/34：P1 加载器 29 + P5 fixture/预设 5；全量回归 1248 passed / 1 skipped / 84.87%） |
| T003 | passing | 2026-10-24 完成（11/11 P2 用例；全量回归 1259 passed / 1 skipped / 84.94%；D7 erratum 7.1 落档） |
| T004 | passing | 2026-10-24 完成（26 用例 P3；全量 1285 passed / 1 skipped / 85.06%；1214 MHA 基线逐位回归零改动通过） |
| T005 | passing | 2026-10-24 完成（19 用例 P4；全量 1304 passed / 1 skipped / 85.14%） |
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

### 2026-10-24 会话记录（T002，标准 TDD）

- 完成阶段：T002 Red（子代理写 tests/test_diffusion_llama.py，P1+P5 分区）+ T002 Green（主代理实现 shadowinfer/models/diffusion_llama.py）
- 修改文件：tests/test_diffusion_llama.py（Red）、shadowinfer/models/diffusion_llama.py（Green）、shadowinfer/models/__init__.py（导出）
- 实现要点：fail-fast 校验全前置于模块构造（路径→config 字段→维度→mask 范围→dtype 参数→文件存在→index 一致性→权重集合→形状）；期望权重名集合由 config 推导（D9 单一映射，非硬编码）；meta 设备构造 + `torch.utils.swap_tensors` 绑定真实权重（swap_tensors 连 Python 类型一起交换，替换张量须先包 nn.Parameter）；tie 时 lm_head 共享 embed_tokens 参数；inv_freq 为非持久 buffer（不出现在 named_parameters）；错误路径零部分加载（所有 ValueError 携带定位字段/张量/文件名）
- T002 验收：34/34（加载器 29 + fixture/预设 5）；权重保真以 safetensors 原始回读多重集逐张量 torch.equal 断言（单片/分片跨布局 bitwise 一致、bf16→fp32 无损、auto 保 dtype）；全量回归 1248 passed / 1 skipped / 84.87%，五门禁全绿（black/isort/flake8 0 issue，mypy 76 files 0 issue）
- 备注：forward_step/forward_block 为 NotImplementedError 占位（签名对齐 design 4.3），T003 实现建模层

### 2026-10-24 会话记录（T003，标准 TDD）

- 完成阶段：T003 Red（子代理写 P2 建模层数值对照测试 11 用例 4 类）+ T003 Green（主代理实现 diffusion_llama.py 建模层）
- 修改文件：tests/test_diffusion_llama.py（P2 追加 + 两处 Red 期修正）、shadowinfer/models/diffusion_llama.py（建模层）、design.md（Erratum 7.1）
- Red 期测试内容：手写参考实现（_ref_forward：RMSNorm/RoPE/SwiGLU/repeat_kv attention 显式逐算子）golden 对照（登记 atol=rtol=1e-5）× {全双向/段因果} × {GQA kv2/MHA kv8} × {偏移 0/块行偏移}；段因果分解两级锚点（KV 层 + logits rel-L2 层 + argmax 层）；RoPE 数学性质（inv_freq 公式 + 相邻位置旋转角差 + 位置偏移分解）；MHA 零特判独立路径
- Red 期两处主代理修正（子代理偏离，均已落测试内注释）：①参考实现 RoPE 配对从交错式 (x[2i],x[2i+1])（GPT-J 风格）改为分半式 (x[j],x[j+D/2))——D6「标准 rotate-half」= HF Llama/Qwen 的 cat(freqs,freqs) 配对，真实 LLaDA/Dream checkpoint 训练口径，交错式将在 s13 产生错误 logits；RoPE 角度断言同步改分半配对
- **Green 期实证否证与 Erratum 7.1**：D7 tier (i)「KV 投影跨形状逐位」被实测否证——KV 投影自身是 GEMM 输出，M 维变化切换 MKL microkernel（[12,256] vs [4,256] 共享行差 2.4e-7~1e-6；[9,256] vs [4,256] 恰同 kernel 逐位；FB 先例通过属形状幸运）；跨实现逐位亦不可行（SLEEF 向量化 cos vs 标量 libm cos ~1 ulp）。修订：KV 层改登记容差 KV_TOL=1e-5（rel-L2），logits 层（FB_TOL2=1e-6，实测 6.6e-7）与 argmax 层不变；语义错误 O(1) 与容差有 100 倍分离度。三处测试断言同步修订
- 实现要点：单层计算序严格同构参考（embedding→[rmsnorm→qkv proj→rope→repeat_kv→scores=q@k^T/sqrt(d)+mask→softmax→@v→o_proj→残差→rmsnorm→SwiGLU→残差]→norm→lm_head）；KV 为 post-RoPE KV-head 原生口径 [B,H_kv,T,D]（D2）；forward_step 校验 kv_cache 拒绝（D11）/ids 2-D 整型/超长/mask [S,S]|[1,1,S,S] 广播；forward_block 校验镜像 SimpleDiffusionLLM 口径（prefix_len>=0/prefix_kv 缺层/形状 [B,H_kv,L,D]）；cos/sin 以 inv_freq float32 计算后 cast 参数 dtype（bf16 兼容）；meta 构造路径复用 T002 加载器
- T003 验收：11/11 P2；全量回归 1259 passed / 1 skipped / 84.94%；五门禁全绿
- 备注：Gold 期发现的 `[9,256]` vs `[4,256]` 逐位属 kernel 幸运，RoPE 分解测试已同步改容差口径避免脆弱断言

### 2026-10-24 会话记录（T004，标准 TDD）

- 完成阶段：T004 Red（子代理写 P3 契约与管线测试 20 函数/26 用例 5 类）+ T004 Green（主代理实现 D3 抽取 + D4 协议化 + D1 GQA 装配）
- 修改文件：tests/test_diffusion_llama.py（P3 追加）、shadowinfer/execution/blockwise_generate.py（新建）、shadowinfer/execution/__init__.py（导出）、shadowinfer/execution/block_decoding.py（BlockwiseModel 协议 + 两处注解放宽 + baseline docstring 更新）、shadowinfer/core/backends/pytorch_backend.py（generate 薄委托 -202 行 +10 行）
- Red 期 17 键统计契约提取（pytorch_backend.py:964-986 程序化核对）：blocks/total_rounds/computed_rows_total/kv_write_rows_total/forced_completions/total_time_ms/page_precision/rounds_per_block/unmasked_per_round/fallback_per_round/mean_confidence_per_round/rows_saved_vs_baseline/unmask_positions_per_round/block_time_ms/draft_acceptance_rate/rounds_to_fixpoint/changed_per_round（draft_mode 三键非 draft 为 None；capture_block_logits 时第 18 键 block_logits_per_round）
- Green 实现要点：blockwise_generate 纯函数 = generate 体 cut-paste（唯一 delta：`self._model`→参数 + store 以 `getattr(model, "num_kv_heads", model.num_heads)` 构造）；backend.generate 保留 no-model RuntimeError 后薄委托，`_last_generate_stores` 语义逐字保持；BlockwiseModel 为 TYPE_CHECKING Protocol（forward_step 第三返回元素 Optional[Tensor] 兼容两模型），运行时零开销鸭子类型；backend 清理 4 个失效 import + time + List
- T004 验收：26 用例 = 契约 pin 11 + DiffusionLlama 两模式 generate（17 键/页驻留 appended_tokens==P+G/rewritten==0/GQA store num_heads==2/gather [B,2,P+G,D]）4 + fail-fast 6 + 薄委托 MHA 逐位回归 3 + 协议面 2；全量回归 1285 passed / 1 skipped / 85.06%（SimpleDiffusionLLM 1214 基线含内部逐位锚点零改动通过）；五门禁全绿（mypy 77 files）
- 备注：P3 中 14 用例 Red 期即过（pin T003 行为/后端语义/协议面），12 用例因 blockwise_generate 缺失失败——符合预期 Red 分布

### 2026-10-24 会话记录（T005，标准 TDD）

- 完成阶段：T005 Red（子代理写 P4 remasking 测试 18 函数/19 用例 5 类）+ T005 Green（主代理实现 execution/remasking.py）
- 修改文件：tests/test_diffusion_llama.py（P4 追加）、shadowinfer/execution/remasking.py（新建）、shadowinfer/execution/__init__.py（导出 RemaskingConfig/remasking_generate）
- 实现要点：RemaskingConfig frozen dataclass（mask_token_id/unmask_per_pass/max_passes/strategy）；remasking_generate 每轮全序列双向 forward_step（无 mask，D5(a)）→ softmax 置信度 → 已解掩排除 → 按 confidence（top-k）或 random（seeded Generator randperm）解掩 min(u, 剩余) 位 → 未选中位保持原状；独立 fail-fast 九项（共享口径沿 validate_generate_inputs 措辞 + gen_len>=1 无整除约束）；独立统计四键 passes_used/unmasked_per_pass/final_confidences/mask_residuals（与 17 键契约集合级断言不相交）
- T005 验收：19/19 P4（有限步全解掩闭式 passes_used==G//u 与 unmasked_per_pass 逐轮记账、前缀运行单调性、random 种子逐位复现、max_passes 耗尽 residual 闭式（batch 1/2）、fail-fast 九项、与块式臂同 prompt 一致率指标可计算无数值门槛）；全量回归 1304 passed / 1 skipped / 85.14%；五门禁全绿（mypy 78 files）
- 备注：Green 一次通过（Red 契约清晰）；本机 fixture 一致率仅结构口径，S3 >= 0.90 为 s13 服务器场地断言
