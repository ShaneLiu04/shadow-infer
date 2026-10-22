<p align="center">
  <img src="assets/banner.svg" alt="ShadowInfer — 面向端侧的 Diffusion LLM 推理优化框架" width="100%">
</p>

<p align="center">
  <a href="https://github.com/ShaneLiu04/ShadowInfer/actions/workflows/ci.yml"><img src="https://github.com/ShaneLiu04/ShadowInfer/actions/workflows/ci.yml/badge.svg" alt="CI"></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/license-MIT-blue.svg" alt="License: MIT"></a>
  <a href="pyproject.toml"><img src="https://img.shields.io/badge/python-3.10%2B-3776AB.svg" alt="Python 3.10+"></a>
  <img src="https://img.shields.io/badge/PyTorch-2.2%2B-EE4C2C.svg" alt="PyTorch 2.2+">
  <img src="https://img.shields.io/badge/code%20style-black-000000.svg" alt="Code style: black">
</p>

# ShadowInfer

> 面向端侧的 Diffusion LLM 推理优化框架 —— 在 4-8 GB 显存约束下实现低延迟、低精度损失的推理。

| 版本 | 测试 | 覆盖率 | 质量门禁 | 许可证 |
|------|------|--------|----------|--------|
| v3.18.0 | 1128 passed / 1 skipped | 84.56% | black / isort / flake8 / mypy / pytest 全绿 | MIT |

ShadowInfer 通过 **ShadowKV 决策平面**、**Q-drift 步感知调度**、**FFN 动态通道剪枝** 与多 Agent 协作架构，为 Diffusion LLM 的端侧 / 车端 GPU 部署提供从性能分析、策略优化到生产部署的完整链路。

---

## 目录

| 快速上手 | 架构与算法 | 工程化与生态 | 参考与贡献 |
|---------|-----------|-------------|-----------|
| [项目简介](#项目简介) | [架构设计](#架构设计) | [工程化与生产安全](#工程化与生产安全) | [项目结构](#项目结构) |
| [核心特性](#核心特性) | [核心模块](#核心模块) | [CLI 使用](#cli-使用) | [开发贡献](#开发贡献) |
| [快速开始](#快速开始) | [核心算法](#核心算法) | [模型后端](#模型后端) | [引用](#引用) |
| [性能基准](#性能基准) | [设计决策](#设计决策) | [Agent 插件、异步执行与早停](#agent-插件异步执行与早停) | [版本历史](#版本历史) |
| | | [可观测性与回归测试](#可观测性与回归测试) | [许可证](#许可证) |
| | | [分布式与 A/B 测试](#分布式与-ab-测试) | |

---

## 项目简介

Diffusion LLM 与自回归 LLM 的关键差异在于：它通过 **固定步数的 denoising** 过程从噪声生成文本，每步都是完整前向传播。这意味着：

- KV Cache 每步重建，但相邻 step 之间高度相似，具备复用空间；
- 不同 denoising step 对计算误差的敏感度差异显著（早期噪声高、可激进优化，后期噪声低、需保持精度）；
- 端侧 GPU 显存小、延迟敏感，需要算法与工程化手段结合。

ShadowInfer 将这些优化机会封装为四个核心 Agent，由统一编排器（Orchestrator）协调，并通过声明式策略 DSL、生产安全网与可观测性组件，把研究性优化变成可部署、可回滚、可审计的工程能力。

---

## 核心特性

| 特性 | 说明 |
|------|------|
| **ShadowKV 决策平面** | 形式化 Importance / Drift / Memory-Pressure 三维平面，统一输出 precision / reuse / eviction 决策。 |
| **IRAP 接收注意力重要性（v3.5.0，机制说明未验证）** | 重要性主信号由 query 行熵改为 KV 列向接收注意力的跨步 EMA 传播（Incremental Received-Attention Propagation）：常数内存、增量计算、对齐 H2O / StreamingLLM 语义；attention 输入契约显式化（logits / probabilities）并只归一化一次；注意力 sink 前缀硬保护。收益声称待大规模实验验证（protocol 见 `specs/changes/AR003-irap-signal-integrity/experiment-plan.md`）。 |
| **热路径张量化** | ShadowKV 重要性评分、精度分配、复用决策、压缩与打包全链路向量化，压缩耗时较逐元素实现下降约 20x（CPU 实测）；逐元素 reference 实现保留作 golden 对照与异常兜底。 |
| **字节级混合精度 KV Cache** | 支持 FP16 / INT8 / INT4 按需存储，per-token-head 动态分配精度；INT4 分组随 head_dim 自适应。 |
| **预算准入与重要性感知驱逐（v3.5.0）** | 压缩前按重要性升序降档（fp32→fp16→int8→int4 阶梯）以满足层预算，单次打包零重打包；显存压力超阈时跨层驱逐最低重要性 token-head（sink 保护）；字节估算按 per-precision bits 精确记账（修复 8x 低估）。 |
| **KV Cache 预取** | 基于 Q-drift 敏感度预测下一 step 复用 mask，提前 stage KV 数据。 |
| **Q-drift 步感知调度** | 敏感度由位置先验与实测扰动响应凸组合（默认权重 0.3 / 0.7），结合激活漂移动态选择 ShadowKV / FFN 工作模式；扰动响应与真实扰动幅度的 Spearman 秩相关 >= 0.9。 |
| **决策记忆化** | 同 (step, layer) 同输入的决策缓存（输入指纹校验、step 边界失效、异常旁路）与 FFN 反量化结果缓存，重复调用决策耗时下降 >= 90%。 |
| **真实性能记账** | latency.e2e_ms / per_step_ms 真实测量（内部计时与外部墙钟对照偏差 0.02%），KV memory 按真实节省字节记账。 |
| **FFN 打包向量化（v3.8.0）** | `PackedFFNWeight._do_pack` 按精度分组批量 gather/量化/字节打包；逐通道存储布局、scale、内存计账与指纹与参考实现逐位一致（含奇数长度、pack_dim 0/1）；同批 0.072x（8.41ms vs 116.56ms @1024×256）。复现脚本见 `specs/archive/AR006-ffn-pack-vectorization/`。 |
| **推理管线常数开销消除（v3.7.0）** | Kernel 后端解析结果记忆化（成功/失败均缓存，失败不再逐次重试 `cpp_extension.load`；同进程 200 次对照 0.022x）；KV INT4 字节打包改为行批量（`_pack_int4_rows`，奇数行内补零；字节布局与参考实现逐位一致，同批 0.016x）。复现脚本见 `specs/archive/AR005-overhead-memoization/`。 |
| **实验可复现编排（v3.9.0）** | `scripts/run_experiment.py` 一键执行真实后端 baseline/optimized 对照：固定 seed、每轮清空后端 KV cache、记录配置摘要与环境（Python/torch/设备/git SHA）、按 `--repeats` 输出 median/min/max、每次重复的 profiler 产物隔离到 `run_i/`；显式拒绝 mock 后端。内置 s1（CPU）档位，`configs/experiments/` 提供 s2（7B GPU）与 s3（8K 长上下文）模板。 |
| **FlashInfer 式 Plan/Run 融合（v3.10.0）** | 新增 `PrecisionPlan`：一次构建精度分组/flat 索引/复用索引/SHA-256 指纹，压缩与字节打包共享；生产路径不再构造未使用的逐条目 scale dict；pack 复用压缩阶段 fp16/int8 量化结果，消除重复分组与重复量化。保留 legacy 与逐元素 reference 作 golden，异常自动回退。CPU 多规模同批中位合计 2.81-3.74x（逐位一致）。 |
| **率失真 KV 预算分配（v3.11.0）** | `rd_allocator` 以 `attention_importance × KV 幅值敏感度` 排序（量化误差近似），替代纯重要性降档；sink 与复用条目受保护。可选硬驱逐（`hard_eviction_enabled`）在预算低于 int4 底线时按同序驱逐低价值 token-head（零字节，解包为零），保证内存上限不再被突破。真实 8 层模型 5 seed：同字节加权 KV 重建误差 1.2MB -3.94%、1.5MB -6.58%；0.9MB 持平（-0.26%）。 |
| **稠密精度码热路径（v3.12.0）** | 以 `int8[seq, heads]` codes 作为精度决策的唯一热路径状态：`allocate_precision_codes` 直接产出 codes，`admit_codes` 用稳定 argsort + 前缀和闭式求解降档/驱逐，`PrecisionPlan.from_codes` 张量化生成分组与 flat 索引，dict 仅在 API 边界转换一次。修复非连续 K/V 导致 `zeros_like().view()` 的静默回退（4 层管线 16 次压缩 0 次回退）。CPU 3 次中位 8192×32 元数据阶段 2443.2→135.7 ms（18.00x），512-4096 为 16.67-18.81x，输出与 v3.11 逐位一致。 |
| **可执行 INT8 FFN（v3.13.0）** | `INT8Linear` 按输出通道对称量化权重、按 token 动态量化激活，使用真实 `torch._int_mm` INT8 GEMM 后按 `scale_x × scale_w` 反量化；`SimpleDiffusionLLM`/HF 后端可直接替换 FFN 线性层并释放 FP32 权重（可选保留 golden）。7B FFN 上投影（4096×11008，M>=16）CPU 复跑 4.8-6.7x，权重内存 3.99x 降低；中等模型端到端 1.37-1.47x；小模型 logits 相对误差 0.43%、top-1 99.22%。`_int_mm` 不可用时确定性回退且只告警一次。 |
| **规范化紧凑 KV Store（v3.14.0）** | `PackedKVCache.unpack` / `apply_reuse_mask` / `evict` 全量张量化：unpack 按精度桶批量 gather+反量化（CPU 512-2048 为 101-219x），复用按 flat 索引批量拷贝（7.3-8.2x），驱逐改为桶内就地删除并真实释放字节（49-109x，25% 条目对应 25% 字节下降）。修正 INT4 `B>1` scale 的 batch 索引缺陷（B=2 每 batch 相对 MSE 0.0075/0.0072），统一压缩/打包/成本模型的 `int4_group_size`（默认 16）且只量化一次；默认 `store_dense_kv=false`，packed cache 成为单一驻留表示，`resident_bytes()` 与 `KVCacheManager` 统一口径（dense shadow 存在时同时计入，超预算先释放 shadow 再驱逐）。 |
| **跨步复用与紧凑 KV 闭环执行（v3.15.0）** | `StepReuseState` 逐 block 缓存 `(input, output, K, V)`，按相对 L2 漂移门控复用整个 block：跳过 Q/K/V/O、attention 与 FFN，输出 `cached_out + (h - cached_in)`；`kv_lifecycle=replace` 修正 diffusion 固定序列的 KV 无限追加；`kv_execution=packed` 让每层 K/V 以 `PackedKVCache` 驻留并在 append 模式从 packed 读取 past。真实受控 denoising 轨迹（3 seed × 3 repeats，CPU）：reuse 1.76-1.94x、logits relL2 0.94-0.96%、跳过 50% block；int8 packed 数据平面驻留为 FP16 dense 的 53%（全口径含元数据 72%）、relL2 0；组合 1.09-1.44x。默认关闭，旧路径逐位不变；7B/GPU 按 `s8_step_reuse_gpu.yaml` 在服务器复现。 |
| **分页增量 KV 驻留（v3.16.0）** | `PagedKVStore` 以 `[B,H,page_tokens,D]` 模型布局页存储 KV，append 仅量化/写入新 token 入页（向量化、零 Python 逐 token 循环），封存页 id 与数据指针不可变，读取走单遍 gather（零 permute/零 repack）；行级量化与 `PackedKVCache` 同元素逐位一致（fp32/fp16/int8/int4 golden `torch.equal`），支持按 token-head 驱逐（整页全驱逐真实释放字节、部分驱逐行置零、槽位空闲链复用、不复活）。`kv_execution="paged"`（`page_size` 可配，默认 32）消除 packed 路径每步全量 unpack+repack 与 precision_map 重建；执行统计新增 `paged`/`page_size`/`page_count`/`open_pages`/`appended_tokens_total`/`repack_tokens_saved`/`page_evictions` 等八键。KV 执行平面微基准（append 簿记，3 seed 中位，CPU）：加速 9.30x（S=512）/ 10.16x（S=2048），每步延迟斜率比为 packed 基线的 0.088-0.114，Python 堆瞬态峰值降约 51-57x；fp16 逐位等价、int8 驱逐前后逐位等价由 UT 证明。7B/GPU 长上下文按 `s9_paged_longctx_gpu.yaml` 在服务器复现。 |
| **收敛感知宏块 KV 模式控制（v3.17.0）** | `MacroModeState` 将序列按宏块（`macro_block_size`，默认 32）分类为 SKIP / DELTA / REFRESH 三态：漂移低于 `skip_threshold` 的行冻结（残差回放），活跃比例超 `delta_min_active_ratio` 的块整块重算，warmup 与 `force_refresh_every` 强制全刷作精度保险；活跃行输出与全量前向 relL2 <= 1e-6（K/V 组装逐位相等），分类全程向量化。`PagedKVStore.rewrite_tokens` 提供行级覆写（校验全先于变异、驱逐行拒写、未覆写行逐位不变），SKIP 块零页写。模型侧行粒度分段前向 + 后端三态写回分派（全 REFRESH 走 clear+append 快路径），任一失败自动降级 dense 重算并 reset。`mode_control_enabled`（paged+replace 显式 opt-in，与 step reuse 互斥），执行统计新增 `mode_control`/`macro_block_size`/`skip_blocks`/`delta_blocks`/`refresh_blocks`/`skipped_rows`/`rewritten_rows`/`projected_rows` 键，默认关闭逐位回归。**诚实结论（CPU 玩具负载，机制说明口径）**：端到端无加速收益（f2 保险臂 0.64-1.04x，f0 默认臂 0.49-1.06x；行粒度重组开销 ~ 所省计算 35-60%，冻结行误差随陈旧深度线性累积），交付物为正确性（1016 UT）、误差有界语义（f2 保险臂全步 relL2 <= 2.65%）、漂移门自愈与 SKIP 零页写；三轮预注册历史（含两次断言失败与不可行性证明）完整落档，GPU/真实负载按 `s10_mode_control_gpu.yaml` 复现。 |
| **置信度感知块式并行解码（v3.18.0）** | 新增 `shadowinfer/execution/block_decoding.py`：`BlockDecodingConfig`（冻结校验）+ `schedule_round` 向量化调度（置信度过阈位置并行解掩、无过阈时保底 top-1、argmax 并列取最小位、`max_rounds` 保底完成）；`SimpleDiffusionLLM.forward_block` 尾块活跃前向（仅算 W 个块行、prefix_kv 条件 attention、块内双向）+ `attention_mask` 三层透传；**段因果掩码 `segment_causal_mask`**（`seg(i)<seg(j) -> -inf`，保证前缀 KV 时间不变，多块场景替代会自相矛盾的块因果公式，开发期 erratum 落档）。`PyTorchModelBackend.generate` 生成管线：预填充 append -> 逐块「gather -> 活跃前向 -> 调度解掩」轮循环 -> 保底完成 -> sealing 前向固化块 KV -> append，**轮间零页写**（每层 fresh `PagedKVStore`，fp32/fp16/int8 页精度平面，fail-fast 校验无副作用）；`blockwise_generate_baseline` 全量重算参考环路（golden）；统计 14 键（含逐块耗时/逐轮解掩位置/兜底标记/置信度区间/`rows_saved_vs_baseline` 闭式）。**实测（CPU 玩具负载，B=2 单线程，3 seeds 中位）**：fp32 臂 vs 全量重算 **2.29x / 3.68x / 7.08x**（S=256/512/1024），token 序列与逐轮 logits 全逐位一致（relL2 <= 3.25e-7），计算行缩减 4.9/8.4/15.4x，KV 行恰写一次零重写零逐出（== P+G，27/27 点）；fp16 页 relL2 <= 1.26e-5 全逐位；int8 页 token 一致率 >= 0.996（轨迹对齐点 relL2 3.6e-4；阈值边缘调度置换/近并列翻转机理与 A6 判据三轮修订完整落档）；**S=256 加速 2.29x 低于 3x 如实标注**（gather 与 sealing 固定开销占比高）；阈值校准史（设计值 0.9 在玩具模型机制退化 -> 注册 0.005）与敏感性扫描落档。7B/GPU 按 `s11_blockwise_decoding_gpu.yaml` 在服务器复现。 |
| **FFN 热路径与逐层状态（v3.6.0）** | 精度计划指纹在打包时缓存（热路径 O(1)，同批对照下比值 0.001-0.004x）；quantization_info 按层缓存一次（热路径零迭代）；unpack/混合精度量化分组向量化（同批实测约 20x，参考实现保留 golden）；权重/量化态/重要性/mask/统计按 layer_id 隔离（修复 FFN-P1）；修复 mixed 路径 down 投影列掩码形状错误。微基准与复现脚本见 `specs/archive/AR004-ffn-hotpath/`。 |
| **FFN 动态通道剪枝** | 运行时按通道重要性或激活能量剪枝 FFN 中间通道，降低 FLOPs 与访存。 |
| **模型后端抽象** | 内置 `SimpleDiffusionLLM`、`PyTorchModelBackend`、`HuggingFaceModelBackend`，支持 entry-point 插件扩展。 |
| **Agent 插件系统** | 通过 `shadowinfer.agents` entry-point 注册第三方 Agent，Orchestrator 自动调度并收集输出。 |
| **异步 Agent 执行** | `AsyncTaskExecutor` 将 Profiler 聚合、A/B 统计、面板渲染等任务放到后台线程池，带超时与取消。 |
| **不确定性感知早停** | `UncertaintyAwareEarlyStopper` 根据相邻 step 输出变化动态停止 denoising，节省冗余计算。 |
| **声明式策略 DSL** | YAML / JSON 规则控制优化阈值与 Orchestrator 仲裁，替代硬编码阈值。 |
| **结构化日志** | 基于 `structlog` 的 JSON 行日志，支持按大小/时间轮转与运行时动态级别调整。 |
| **生产安全网** | 热配置重载、熔断器、五级优雅降级、令牌桶限流、健康监控；安全网降级在同 step 内不可被策略仲裁覆盖。 |
| **可观测性** | Prometheus 指标、OpenTelemetry 链路、Streamlit / Grafana 面板。 |
| **A/B 测试与回归追踪** | 策略分流 + 统计显著性检验 + JSONL 历史性能回归检测。 |
| **分布式脚手架** | Pipeline / Tensor 并行、多 GPU KV Cache 管理。 |

---

## 快速开始

### 环境要求

- Python 3.10+
- PyTorch 2.2+
- 推荐 CUDA 11.8+（CPU fallback 可用）

### 安装

```bash
git clone https://github.com/ShaneLiu04/ShadowInfer.git
cd ShadowInfer

python -m venv .venv
source .venv/bin/activate  # Linux/Mac
# .venv\Scripts\activate   # Windows

pip install -r requirements.txt
# 或开发模式安装
pip install -e ".[dev]"
```

### 运行测试

```bash
pytest
```

当前主分支：`876 passed, 1 skipped`，测试覆盖率 `83.37%`（pytest-cov，fail-under 75）。

### 30 秒跑通完整流水线

```python
from shadowinfer.orchestrator import Orchestrator

orch = Orchestrator(config_path="configs/optimize_full.yaml")
orch.initialize(model_config={"name": "SimpleDiffusionLLM"})

result = orch.run_full_pipeline(
    prompt="Diffusion LLM edge inference optimization.",
    num_steps=20,
)

print(f"Baseline latency:  {result['baseline_latency_ms']:.1f} ms")
print(f"Optimized latency: {result['optimized_latency_ms']:.1f} ms")
print(f"Speedup:           {result['speedup']:.2f}x")
print(f"Memory savings:    {result['memory_savings_ratio']:.1%}")
print(f"Accuracy drop:     {result['accuracy_drop']:.4f}")
print(f"Report:            {result['comparison_path']}")
```

### 可复现实验（v3.9.0）

```bash
# CPU 冒烟档位：6 步 x 3 次重复，真实 PyTorch 后端
python scripts/run_experiment.py --profile s1 --steps 6 --repeats 3 --device cpu --out results/s1

# 仅校验配置与摘要，不执行模型
python scripts/run_experiment.py --profile s1 --dry-run

# 服务器：7B GPU / 8K 长上下文（需 transformers 与本地或可下载权重）
python scripts/run_experiment.py --config configs/experiments/s2_7b_gpu.yaml \
    --model-config configs/model_fast_dllm_7b.yaml \
    --backend hf --model-name <hf-model-id> --device cuda --steps 32 --repeats 3
```

产物：`result.json`（环境、配置摘要、每次运行原始数据、median/min/max）与 `summary.md`；
`run_i/` 内含该轮的 `profile_baseline.json` / `profile_optimized.json` / `profile_comparison.html`。
`mock` 与 `auto` 后端会被拒绝，避免非真实数据进入结论。

### 单 Agent 使用示例

```python
import torch
from shadowinfer.core.config import Config
from shadowinfer.core.structs import StepConfig
from shadowinfer.qdrift import QDriftAgent
from shadowinfer.shadowkv import ShadowKVAgent
from shadowinfer.ffn_optimizer import FFNOptimizerAgent

config = Config.from_yaml("configs/optimize_full.yaml")
qdrift = QDriftAgent(config)
shadowkv = ShadowKVAgent(config)
ffn = FFNOptimizerAgent(config)

model_config = {"num_layers": 32, "hidden_dim": 4096, "num_heads": 32}
qdrift.on_init(model_config)
shadowkv.on_init(model_config)
ffn.on_init(model_config)

step_cfg = StepConfig(
    step_id=5, total_steps=50, noise_level=0.3,
    shadowkv_mode="aggressive", ffn_mode="sparse",
)

# Q-drift 调度
qd = qdrift.on_step(step_cfg, {
    "query_current": torch.randn(1, 128, 32, 128),
    "query_previous": torch.randn(1, 128, 32, 128),
    "activation_current": torch.randn(1, 128, 4096),
    "activation_previous": torch.randn(1, 128, 4096),
    "noise_level": 0.3,
})
print(qd["dispatch"])

# ShadowKV 压缩
sk = shadowkv.on_step(step_cfg, {
    "attention_scores": torch.randn(1, 32, 128, 128),
    "kv_current": {"k": torch.randn(1, 32, 128, 128), "v": torch.randn(1, 32, 128, 128)},
    "kv_previous": {"k": torch.randn(1, 32, 128, 128), "v": torch.randn(1, 32, 128, 128)},
    "layer_id": 0,
    "qdrift_signal": qd,
})
print(sk["compressed_kv"]["compression_ratio"])

# FFN 优化
ff = ffn.on_step(step_cfg, {
    "ffn_input_current": torch.randn(1, 128, 4096),
    "ffn_input_previous": torch.randn(1, 128, 4096),
    "ffn_output_previous": torch.randn(1, 128, 4096),
    "weights": {"up": torch.randn(11008, 4096), "down": torch.randn(4096, 11008)},
    "qdrift_signal": qd,
    "layer_id": 0,
})
print(ff["compute_stats"]["flops_saved"])
```

---

## 架构设计

ShadowInfer 采用**模块化多 Agent 架构**，由中心化 Orchestrator 统一协调，所有 Agent 通过 Profiling Bus 共享数据。

```
用户输入 / 配置（YAML / JSON）
        │
        ▼
┌─────────────────────────────────────┐
│             Orchestrator            │
│       编排 · 仲裁 · 安全网集成       │
└──┬─────────┬─────────┬─────────┬────┘
   │         │         │         │
   ▼         ▼         ▼         ▼
Profiler  ShadowKV  Q-drift  FFN Optimizer
   │         │         │         │
   └─────────┴────┬────┴─────────┘
                  ▼
           Profiling Bus
                  ▼
      PyTorch / CUDA / 自定义算子
                  ▼
          端侧 / 车端 GPU
```

端到端执行流程：

1. **初始化**：加载模型、配置优化策略、初始化各 Agent；
2. **Profiling 阶段（可选）**：运行基线，收集完整性能数据；
3. **Optimization 阶段**：按 denoising step 循环执行优化推理；
4. **Validation 阶段**：对比优化前后 accuracy / latency / throughput；
5. **输出**：综合报告 + 可视化结果。

---

## 核心模块

### Profiler Agent

- 自动统计 KV 精度分布（per-token、per-head、per-layer）；
- 统计 cache 复用率、q-drift 命中率、FFN 计算负载；
- 测量端到端 / per-step / per-layer 延迟；
- 追踪优化前后的生成质量变化。

输出产物：`profile_baseline.json`、`profile_optimized.json`、`profile_comparison.html`。

### ShadowKV Agent

解决 KV Cache 占用高、访存频繁的问题：

- **ImportanceModel**：基于 attention score 分布的 entropy 计算 per-token-head 重要性；
- **KVDecisionPlane**：在 Importance / Drift / Memory-Pressure 三维平面上输出 precision、reuse、eviction 决策；
- **PackedKVCache**：字节级打包，支持 FP16 / INT8 / INT4 混合精度；
- **EvictionPolicy**：显存预算超限时按重要性回收低价值 token-head；
- **Prefetch**：基于 Q-drift 敏感度预测下一步复用 mask，提前 stage KV。

### Q-drift Agent

利用 Diffusion LLM 不同 step 对误差敏感度不同的特点：

- **Step 敏感度估计**：根据 step index / noise level 输出 sensitivity_score（早期低、后期高）；
- **激活漂移检测**：比较相邻 step 的 query / activation 变化，输出 drift_score；
- **调度矩阵**：综合 sensitivity 和 drift，选择 ShadowKV 模式（aggressive / balanced / conservative）和 FFN 模式（sparse / mixed / full）。

### FFN Optimizer Agent

减少 FFN 层冗余矩阵计算：

- **通道重要性分析**：activation_magnitude / weight_magnitude / gradient_based；
- **混合精度量化**：重要通道 FP16，普通通道 INT8，不重要通道 INT4；
- **稀疏更新**：相邻 step 输入变化小时仅重算变化 token；
- **动态通道剪枝**：运行时按通道重要性或激活能量剪枝，并自动在精度敏感时禁用。

### 模型后端（ModelBackend）

统一后端抽象，已内置：

- `MockModelBackend`：用于框架测试；
- `PyTorchModelBackend`：包装 `SimpleDiffusionLLM`；
- `HuggingFaceModelBackend`：支持 `AutoModelForCausalLM`，含离线 tiny-model fallback；
- `VLLMModelBackend` / `SGLangModelBackend`：scaffolding（依赖可选）。

第三方后端可通过 `pyproject.toml` entry-point 注册到 `BackendRegistry`。

### 工程化模块

见下文 [工程化与生产安全](#工程化与生产安全)。

### 可观测性模块

- `MetricsRegistry`：Counter / Gauge / Histogram，支持 Prometheus 文本格式暴露；
- `Tracer`：OpenTelemetry 兼容 span 追踪；
- Dashboard：`shadowinfer dashboard` 启动 Streamlit 面板，也提供 Grafana JSON 模板。

### 分布式模块

- Pipeline Parallelism 微批处理脚手架；
- Tensor Parallelism all-reduce 脚手架；
- 多 GPU KV Cache 一致性管理。

### A/B 测试框架

- hash / 权重策略分流；
- paired t-test 统计显著性检验；
- 自动策略推广与回退。

---

## 核心算法

### ShadowKV 分层压缩

1. **Importance Scoring（v3.5.0：IRAP 接收注意力）**

   ```python
   # 0) 契约归一化：真实后端返回 post-softmax 概率（跳过 softmax，仅行和校正）
   #    mock / 原始 logits 路径：单次 softmax（绝不重复归一化）
   weights = probs if attn_is_probabilities else F.softmax(attn_logits, dim=-1)

   # 1) 列向接收注意力：KV token 被全部 query 关注的程度（H2O / StreamingLLM 口径）
   received[t, h] = weights[:, h, :, t].mean()

   # 2) IRAP 跨步 EMA 传播（常数内存，无历史列表）
   A[t, h] = gamma * A_prev[t, h] + (1 - gamma) * received[t, h]   # gamma 默认 0.9

   # 3) rank 归一化（逐 head 在 token 维，按排序位次；长尾稳健） + 行熵辅助（beta 默认 0.3）
   rank[t, h] = argsort(argsort(A[:, h]))[t] / seq_len
   importance = clamp(((1 - beta) * rank + beta * normalized_entropy) * layer_factor, max=1)

   # 4) sink 硬保护：前 sink_tokens（默认 4）个 token 强制最高档，且不可驱逐
   importance[:sink_tokens] = 1.0
   ```

   状态内存为每层 `seq_k x heads` 的张量（O(1) 于步数），跨步差分
   `delta_a = ||A_t - A_prev|| / ||A_prev||` 作为可选漂移信号导出给 Q-drift
   （`use_importance_drift`，默认关闭，一步滞后传递）。

2. **Precision Allocation**

   | 重要性分数 | 精度 | 存储格式 |
   |-----------|------|---------|
   | ≥ 0.8 | FP32 | 全精度 |
   | 0.5 - 0.8 | FP16 | 半精度 |
   | 0.2 - 0.5 | INT8 | 8-bit 量化 |
   | < 0.2 | INT4 | 4-bit 量化 / 复用 |

   配置显存预算（`memory_budget_mb`）时，精度分配后先做**预算准入**：
   非 sink 条目按重要性升序逐级降档，直至该层预估字节（per-precision bits
   + scale 精确记账）不超过 `预算 // 层数`；触底仍超时告警但保留 int4 底线。
   `memory_pressure >= eviction_pressure_threshold`（默认 0.9）时触发跨层
   全局驱逐（10% 迟滞带，sink 永不驱逐）。

3. **复用决策**
   ```python
   delta = torch.norm(kv_current - kv_previous) / torch.norm(kv_current)
   threshold = base_threshold * (1 - 0.5 * step_id / total_steps) * (1 - drift_score * 0.3)
   should_reuse = delta < threshold
   ```

以上三步在 v3.4.0 起均为整矩阵张量化实现（重要性评分 / 精度分配 / 复用决策 /
压缩 / 打包）；逐元素 reference 实现保留，作为单元测试的 golden 对照与张量路径
的异常兜底（命中异常时记录告警并回退，保证可用性）。

### Q-drift 调度矩阵

sensitivity_score 为位置先验与实测扰动响应的凸组合（v3.4.0 起）：

```python
sensitivity = clamp(alpha * positional_base + (1 - alpha) * tanh(raw_drift), 0, 1)
# alpha 默认 0.3（配置键 qdrift.positional_prior_weight）
# raw_drift 为相邻 step query/activation 的未归一化漂移（0.6/0.4 加权）
# previous 张量缺失时回退纯位置先验并记录日志
```

drift_score 采用 tanh 归一化以保持动态范围（避免大幅扰动下塌缩为恒 1.0）。

| sensitivity_score | drift_score | ShadowKV | FFN | 说明 |
|------------------|------------|----------|-----|------|
| < 0.3 | < 0.2 | aggressive | sparse | 最大优化 |
| < 0.3 | ≥ 0.2 | balanced | sparse | 适度优化 |
| 0.3 - 0.7 | < 0.2 | balanced | mixed | 平衡策略 |
| 0.3 - 0.7 | ≥ 0.2 | conservative | mixed | 保守策略 |
| ≥ 0.7 | any | conservative | full | 全精度 |

### FFN 通道剪枝

- `static_importance`：基于通道权重大小；
- `dynamic_activation`：基于当前 step 激活能量；
- 当 `sensitivity_score >= 0.7` 时自动禁用剪枝，避免精度敏感 step 受损；
- FLOPs 按 `active_ratio` 重新核算。

---

## 工程化与生产安全

ShadowInfer 内置完整的生产安全保障机制。

### 热配置重载（Hot Config Reloader）

- 文件监听 + 防抖，避免重载风暴；
- YAML 加载后通过业务规则验证（压缩比范围、温度 > 0、精度级别合法性等）；
- 验证通过才原子切换，失败保留旧配置；
- 配置变更后通知已注册 Agent 回调。

```python
from shadowinfer.engineering import HotConfigReloader

reloader = HotConfigReloader(config_path="configs/optimize_full.yaml")
reloader.start_watching(interval_sec=1.0, debounce_sec=0.5)
reloader.on_config_change(agent.on_config_update)
```

### 熔断器与优雅降级

**Circuit Breaker** 三态自动切换：

```
CLOSED ──(连续失败 ≥ threshold)──► OPEN ──(冷却超时)──► HALF_OPEN
  ▲                                      │
  │                                      │ (试探失败)
  └──(连续成功 ≥ threshold)─────────────┘
```

**Graceful Degradation** 五级降级体系：

| 等级 | 状态 | ShadowKV | Q-drift | FFN | Profiling | Observability |
|-----|------|----------|---------|-----|-----------|---------------|
| NONE | 正常 | 启用+复用 | 启用 | 启用+稀疏 | 每步 | 启用 |
| LIGHT | 轻度 | 启用+复用 | 启用 | 启用+稀疏 | 每5步 | 启用 |
| MODERATE | 中度 | 启用（无复用） | 启用 | 启用（无稀疏） | 每10步 | 启用 |
| SEVERE | 重度 | 启用（无复用） | 关闭 | 启用（无稀疏） | 每20步 | 关闭 |
| EMERGENCY | 紧急 | 关闭 | 关闭 | 关闭 | 关闭 | 关闭 |

### 令牌桶限流

```python
from shadowinfer.engineering import TokenBucketRateLimiter

limiter = TokenBucketRateLimiter(rate=100.0, burst=200)
if limiter.acquire():
    process_request()
```

### 健康监控与性能预算

- `HealthMonitor` 多维度健康检查；
- `PerformanceBudget` 跟踪 latency / memory，超限时触发降级；
- `TensorValidator` 运行时验证 tensor 形状、dtype、数值范围、NaN/Inf；
- `WeightHealthChecker` 检测 NaN / Inf / 全零 / 异常分布权重。

### Production Safety Net

一键整合所有保障机制：

```python
from shadowinfer.orchestrator import Orchestrator
from shadowinfer.engineering import ProductionSafetyNet

orch = Orchestrator(config_path="configs/optimize_full.yaml")
safety = ProductionSafetyNet()
orch.enable_safety_net(safety)
```

Orchestrator 在每次 step 前后自动调用 `check_before_inference()` 和 `report_after_inference()`，实现熔断、限流、降级、健康监控和性能预算跟踪。pre-flight 检查失败或抛异常时，本 step 自动降级到 conservative / full，且该安全降级在同 step 内不可被策略仲裁放宽。

---

## CLI 使用

安装后可通过 `python -m shadowinfer` 使用：

```bash
# 性能分析
python -m shadowinfer profiler \
    --model SimpleDiffusionLLM \
    --backend pytorch \
    --device cpu \
    --seed 42 \
    --num-steps 20 \
    --config configs/profiler_full.yaml

# 运行优化流水线
python -m shadowinfer optimize \
    --model SimpleDiffusionLLM \
    --backend pytorch \
    --device cpu \
    --seed 42 \
    --num-steps 20 \
    --config configs/optimize_full.yaml

# 启动 HTTP serving
python -m shadowinfer serve --config configs/serving.yaml

# 启动 Streamlit 面板
python -m shadowinfer dashboard --port 8501

# 运行基准测试并记录回归历史
python -m shadowinfer benchmark \
    --model SimpleDiffusionLLM \
    --backend pytorch \
    --device cpu \
    --num-steps 20

# 可复现 baseline/optimized 对照（v3.9.0，固定 seed + 环境取证 + 多次重复）
python scripts/run_experiment.py --profile s1 --steps 6 --repeats 3 --device cpu --out results/s1
```

---

## 模型后端

### 内置后端

| 后端 | 说明 |
|------|------|
| `mock` | 随机张量，用于框架测试 |
| `pytorch` | `SimpleDiffusionLLM` 真实 PyTorch 模型 |
| `huggingface` | HuggingFace `AutoModelForCausalLM`（含离线 tiny fallback） |
| `vllm` / `sglang` | 可选依赖的 scaffolding |

### 自定义后端

```python
from shadowinfer.core.model_backend import ModelBackend

class MyDiffusionBackend(ModelBackend):
    def load(self, model_name, device=None):
        self.model = load_my_model(model_name, device)

    def forward_step(self, x, step_cfg, kv_cache=None):
        output, kv_cache, attention = self.model.step(x, kv_cache)
        return {"output": output, "kv_cache": kv_cache, "attention_scores": attention}

    # ... get_kv_cache / set_kv_cache / get_model_config

orch = Orchestrator(config_path="configs/optimize_full.yaml")
orch.set_model_backend(MyDiffusionBackend())
orch.initialize()
orch.run_full_pipeline(prompt="hello", num_steps=20)
```

---

## Agent 插件、异步执行与早停

### Agent 插件系统

ShadowInfer 支持通过 `shadowinfer.agents` entry-point 注册自定义 Agent，Orchestrator 会在每步执行完内置 Agent 后自动调用插件 Agent，并将其输出写入 `step.outputs` 和广播消息。

```toml
[project.entry-points."shadowinfer.agents"]
my_agent = "my_package:MyAgent"
```

```python
from shadowinfer.core.base_agent import BaseAgent
from shadowinfer.core.structs import StepConfig

class MyAgent(BaseAgent):
    def on_init(self, model_config):
        self.model_dim = model_config["hidden_dim"]

    def on_step(self, step_config: StepConfig, inputs):
        return {"custom_metric": step_config.noise_level * self.model_dim}

    def on_shutdown(self):
        return None
```

```python
orch = Orchestrator(config={
    "extra_agents": [
        {"name": "my_agent", "agent": "my_agent", "config": {"enabled": True}}
    ]
})
orch.initialize(model_config={"name": "MyModel", "hidden_dim": 4096})
orch.run_optimized(prompt="hello", num_steps=20)
```

### 异步任务执行器

`AsyncTaskExecutor` 将非阻塞的后台任务（Profiler 聚合、A/B 统计、面板渲染）提交到线程池，支持超时等待与批量取消，避免阻塞主推理线程。

```python
from shadowinfer.core.async_executor import AsyncTaskExecutor

executor = AsyncTaskExecutor(max_workers=4)
future = executor.submit(expensive_summary, data)
result = executor.wait_for(future, timeout=5.0)
executor.shutdown()
```

### 不确定性感知早停

`UncertaintyAwareEarlyStopper` 观察相邻 denoising step 的输出相似度；当相似度持续高于阈值且达到最小步数后，Orchestrator 提前终止推理，减少冗余计算。

```python
orch = Orchestrator(config={
    "early_stop": {
        "enabled": True,
        "min_steps": 5,
        "max_steps": 50,
        "stability_window": 3,
        "similarity_threshold": 0.995,
        "metric": "relative_l2",
    }
})
result = orch.run_optimized(prompt="hello", num_steps=50)
print(result["early_stopped"], result["stopped_step"])
```

---

## 可观测性与回归测试

### Prometheus 指标

```python
from shadowinfer.observability import Counter, MetricsRegistry

registry = MetricsRegistry()
counter = Counter("inference_steps_total", "Total inference steps")
counter.inc(5)
print(registry.expose())
```

启动 exporter：`python -m exporter`，然后 scrape `http://localhost:8000/metrics`。

### OpenTelemetry 链路

```python
from shadowinfer.observability import Tracer

tracer = Tracer(service_name="shadowinfer")
tracer.set_otel_exporter("console")
# tracer.set_otel_exporter("otlp", endpoint="http://localhost:4317")

with tracer.start_span("inference_pipeline", {"model": "SimpleDiffusionLLM"}):
    with tracer.start_span("shadowkv_step"):
        pass
```

### 性能回归追踪

```python
from shadowinfer.benchmarking.regression import RegressionResult, RegressionTracker

result = RegressionResult(
    timestamp=time.time(), run_id="run-42", model="SimpleDiffusionLLM",
    backend="pytorch", num_steps=20, latency_ms=80.0, memory_mb=30.0,
    accuracy_drop=0.004, speedup=1.6, metadata={},
)
tracker = RegressionTracker("benchmarks/results/regression_history.jsonl")
tracker.record(result)
report = tracker.detect_regression(result)
print(report)
```

---

## 分布式与 A/B 测试

### 分布式推理（脚手架）

- **Pipeline Parallelism**：按 layer 切分模型，微批处理隐藏通信；
- **Tensor Parallelism**：按 head / hidden 切分，all-reduce 聚合；
- **多 GPU KV Cache**：跨设备一致性管理。

### A/B 测试

```python
from shadowinfer.ab_testing import ABTestRunner

ab = ABTestRunner(config={"strategy_a": "baseline", "strategy_b": "shadowkv_aggressive"})
ab.run_experiment(samples=1000)
print(ab.report())
```

- 支持 hash 分流与权重分流；
- paired t-test 判断指标差异显著性；
- 自动选择优胜策略并支持灰度推广。

---

## 性能基准

### 复现方式

```bash
# 全量测试套件（含覆盖率）
pytest

# 端到端流水线（6 步参考配置）
python -m shadowinfer optimize \
    --model SimpleDiffusionLLM \
    --backend pytorch \
    --device cpu \
    --num-steps 6 \
    --config configs/optimize_full.yaml

# 稠密精度码元数据基准（dict vs codes，含逐位一致性断言）
python specs/archive/AR010-dense-precision-pipeline/ar010_bench.py

# 可执行 INT8 FFN 基准（7B FFN 形状 / 中等模型 / logits golden）
python specs/archive/AR011-int8-ffn-execution/ar011_bench.py

# 紧凑 KV Store 读写/复用/驱逐基准（含 INT4 batch 修正与字节下降断言）
python specs/archive/AR012-canonical-kv-store/ar012_bench.py

# 跨步复用 + packed KV 执行基准（受控 denoising 轨迹，3 seed）
python specs/archive/AR013-step-reuse-packed-execution/ar013_bench.py

# 分页增量 KV 驻留基准（append 簿记平面，2 规模 × 3 seed，含预注册断言）
python specs/archive/AR014-paged-kv-store/ar014_bench.py

# 宏块模式控制基准（双轨迹 × 四臂 × 3 规模 × 3 seeds，含终版预注册断言）
python specs/archive/AR015-macro-block-mode-control/ar015_bench.py

# 块式并行解码基准（四臂 × 3 规模 × 3 seeds + 阈值敏感性，内置注册断言）
python specs/archive/AR016-blockwise-parallel-decoding/ar016_bench.py
```

### 实测数据（v3.18.0，2026-10-22，Windows / CPU）

置信度感知块式并行解码四臂基准：真实 `SimpleDiffusionLLM`（8 层 / 8 头 / 64 head_dim，
vocab 1000），P=64、W=32、B=2、`max_rounds=8`、单线程，threshold=0.005（校准注册值；
设计值 0.9 在玩具模型上全轮兜底退化，校准史见 bench-results.md），3 seeds
（`specs/archive/AR016-blockwise-parallel-decoding/ar016_bench.py`，内置注册断言全过）：

| 规模 S=P+G | fp32 加速中位（区间） | 计算行缩减 | fp32 relL2 / tokens | fp16 relL2 / tokens | int8 relL2（对齐点）/ 一致率 |
|---|---|---|---|---|---|
| 256 | 2.29x (2.21-2.32) | 4.9x | 0.0 / 逐位 | 1.24e-5 / 逐位 | 3.6e-4 / 1.000 |
| 512 | 3.68x (3.66-3.88) | 8.4x | 3.2e-7 / 逐位 | 1.25e-5 / 逐位 | - / >= 0.997 |
| 1024 | 7.08x (6.93-7.17) | 15.4x | 3.2e-7 / 逐位 | 1.26e-5 / 逐位 | - / >= 0.996 |

- **加速机制**：baseline 每轮全量重算 L+W 行 vs 活跃前向仅 W 行的行裁剪；KV 行恰写一次
  （`kv_write_rows == P+G`，27/27 点，零重写零逐出）；逐块耗时与逐轮调度全部入统计键。
- **诚实结论**：S=256 加速 2.29x 低于 3x（gather 重建与 sealing 前向固定开销占比高）；
  全并行极端阈值（0.003）下轮数塌缩至每块 1 轮，两臂同享收益、比率收敛至 ~1.18x——
  加速来自多轮场景的行裁剪复合，非并行解掩本身。
- **int8 双态**：轨迹对齐点（逐轮解掩位置与终态均与 baseline 一致）relL2 3.6e-4；
  阈值边缘调度置换 / 近并列 argmax 翻转点为错位伪影（~0.17-0.21），token 一致率
  仍 >= 0.996；A6 判据三轮修订（v1 bitwise 代理证伪 / v2 实现 bug / v3 轨迹对齐）
  与诊断证据完整落档 bench-results.md。

### 实测数据（v3.17.0，2026-10-21，Windows / CPU）

宏块模式控制端到端基准：真实 `SimpleDiffusionLLM` 六步去噪轨迹（sigma 0.9→0.02），
双轨迹（T1 均匀收敛边界案例 / T2 块式半 AR 目标负载），四臂（dense golden / AR013 step
reuse / AR015 默认 f=0 / AR015 保险 f=2），3 seeds
（`specs/archive/AR015-macro-block-mode-control/ar015_bench.py`）：

**T2 块式半 AR（目标负载，终版断言对象 = f2 保险臂）**：

| 规模 (L,H,D,S) | f2 relL2 最差（3 seeds） | f2 末半程冻结率中位 | f2 加速中位 | f0 加速中位 | f0 relL2 最差 |
|---|---|---|---|---|---|
| 8/8/64/256 | 1.881% | 0.667 | 0.642x | 0.515x | 7.150% |
| 16/8/64/512 | 2.649% | 0.667 | 0.831x | 0.725x | 10.083% |
| 8/8/64/1024 | 1.988% | 0.667 | 1.043x | 1.045x | 7.571% |

**T1 均匀收敛（边界案例，仅报告）**：f0 加速 0.516-0.789x、relL2 2.248-3.119%、
冻结 0.667；均匀漂移下行粒度分类退化为全有全无，DELTA 机制不出现。

- **终版预注册断言（f2 保险臂，全部通过）**：全步 relL2 <= 3%（实测最差 2.649%）；
  末半程平均冻结率 >= 30%（实测 0.667）。
- **诚实结论：端到端无加速收益**。f2 最优 1.043x（S=1024），S=256/512 为 0.64x/0.83x；
  加速比与 >= AR013 臂比较在二次预注册失败后取消断言只报告。f-扫描诊断（27 组）证明
  >=2x 与 <=2% 预算在本 CPU 玩具负载上不可同时达成：SKIP 冻结行残差回放不含 attention
  对尾部 K/V 演化的跨行耦合（误差 ~ 尾窗质量分数 x 陈旧深度 x 层数，逐步线性累积，
  S=512 十六层最差 10.1%），且 CPU 行粒度重组开销约为所省 dense 计算的 35-60%。
- **机制交付物（实测验证）**：漂移门自愈（f0 陈旧累积至峰值后触发部分 DELTA 与级联
  REFRESH，relL2 回落 2.6%）；f2 保险全步 relL2 有界（1.73-2.65%）；SKIP 步零页写
  （页 id 不变，UT 证明）；f2 冻结率 0.667 与强制刷新调度精确一致。
- 活跃行输出 vs 全量前向 relL2 <= 1e-6、K/V 组装逐位相等由 UT 覆盖
  （`tests/test_macro_mode_execution.py`，95 用例）；三轮预注册历史（原 T1+>=2x 断言
  失败、二次 T2+>=2x 失败、f-扫描不可行性证明、终版断言）完整落档于
  `specs/archive/AR015-macro-block-mode-control/bench-results.md`。
- 7B/GPU 真实块扩散负载（注意力耦合结构弱于均匀玩具）按 `s10_mode_control_gpu.yaml`
  与 `server-experiment.md` 在服务器复现；本机不宣称 GPU 结果。

### 实测数据（v3.16.0，2026-10-21，Windows / CPU）

KV 执行平面微基准（append 每步每层簿记）：真实张量，fp16，`B=4/H=8/D=64/L=8` 层，
`page_size=32`，3 seeds 报中位（`specs/archive/AR014-paged-kv-store/ar014_bench.py`）。
baseline 为现行 packed 路径（每步全量 unpack+repack + precision_map 重建），paged 为
`PagedKVStore` 增量入页 + 单遍 gather：

| 规模 (N 步, S_new, max_seq) | 加速中位 [区间] | 每步延迟斜率比 | Python 堆峰值 baseline→paged |
|---|---|---|---|
| 8, 64, 512 | **9.303x** [9.209-9.751] | 0.088 | 2408KB → 47KB |
| 16, 128, 2048 | **10.162x** [7.684-10.278] | 0.114 | 9975KB → 175KB |

- fp16 全序列 gather 与 packed 全量 pack 参考解码 `torch.equal` 逐位一致（两规模均通过）。
- 加速中 SRS 预估为 ~2x（仅张量算子口径 2×O(S) → 1×O(S)+O(S_new)）；实测 9-10x 的差额
  来自基线每步每层 O(S·H) 的 precision_map 重建与 `_do_pack` 分组 Python 循环，分页路径
  对 past token 完全绕开；两口径均如实报告，逐 seed 原始数据与诚实说明见
  `specs/archive/AR014-paged-kv-store/bench-results.md`。
- 规模 2 的 seed=1 加速 7.684x 低于其余 seed（10.2x+），为 CPU 调度抖动，如实保留。
- fp16 页无压缩，`data_plane_bytes` 与 FP16 dense 基准一致（比值 1.0）；分页收益在每步
  延迟斜率与 Python 瞬态，不在 fp16 驻留字节。
- int8/int4 逐位等价与驱逐等价（驱逐前后未触碰元素逐位不变、字节真实释放、不复活）由
  UT 覆盖（`tests/test_paged_kv_execution.py`）；端到端 logits 等价由
  `tests/test_paged_backend_execution.py` 覆盖（fp16 `torch.equal`）。
- 7B/GPU 延迟、显存与真实模型 logits 按 `server-experiment.md` 在服务器回填；本机不宣称
  7B 结果，也不以微基准替代真实模型结论。

### 实测数据（v3.15.0，2026-10-20，Windows / CPU）

真实 `SimpleDiffusionLLM` forward、真实张量；受控轨迹 `x_t = x0 + sigma_t * noise`
（sigma 0.9→0.02，10 步）；3 seed × 3 repeats 中位
（`specs/archive/AR013-step-reuse-packed-execution/ar013_bench.py`）：

| 规模 L/H/D/HN/S | reuse 加速 | reuse relL2 | 跳过 block | packed 驻留 (data-plane / 全口径) | packed relL2 | combined 加速 |
|---|---|---|---|---|---|---|
| 8/8/64/512/256 | 1.76-1.89x | 0.94-0.95% | 50% | 53% / 72% | 0.00% | 1.09-1.17x |
| 8/8/64/512/512 | 1.86-1.87x | 0.95-0.96% | 50% | 53% / 72% | 0.00% | 1.13-1.17x |
| 8/16/64/1024/256 | 1.93-1.94x | 0.96% | 50% | 53% / 72% | 0.00% | 1.42-1.44x |

- baseline 为 `replace + dense` 的完整重算；reuse 使用 threshold=0.1、warmup=1。
- packed 使用 int8；dense 基准为 FP16（SRS 口径），驻留列「data-plane」仅计张量字节（含
  scale，GPU flat-buffer 部署承载的部分），「全口径」另含 48B/token-head 的 Python 簿记
  估计（CPU 实现开销，GPU 部署不承载）；replace 模式下非复用层仍需重算，因此 packed 的
  收益是驻留而非 CPU 延迟；append 模式下 packed past 会真实参与下一步 attention。
- 组合模式同时启用 packed 与 reuse；跳过层保留同一 packed 对象、不重复 pack。
- 7B/GPU 的延迟、显存与真实模型 logits 按 `server-experiment.md` 在服务器回填；
  本机不宣称 7B 结果，也不以受控轨迹替代真实模型结论。

### 实测数据（v3.14.0，2026-10-20，Windows / CPU）

真实张量、混合精度 map（fp32/fp16/int8/int4 轮转）、`B=1/H=8/D=64`，3 次中位
（`specs/archive/AR012-canonical-kv-store/ar012_bench.py`）：

| 规模 S | unpack 旧→新 | 加速 | reuse 旧→新 | 加速 | evict 旧→新 | 加速 | 驱逐后字节 |
|---|---|---|---|---|---|---|---|
| 512 | 485.8→4.8 ms | **101.3x** | 84.2→11.5 ms | **7.3x** | 564.2→11.5 ms | **49.0x** | -25.0% |
| 1024 | 1016.1→4.8 ms | **212.1x** | 124.6→15.2 ms | **8.2x** | 951.3→15.4 ms | **61.9x** | -25.0% |
| 2048 | 1883.8→8.6 ms | **218.6x** | 266.0→32.4 ms | **8.2x** | 2899.3→26.6 ms | **109.2x** | -25.0% |

- INT4 `B=2` 每 batch 相对 MSE：K 为 0.0075 / 0.0072（修正前分别约 0.1617 / 0.0819）；
  修复的是 scale `[N*B,G,1]` 被按 `[N]` 读取的 batch 索引缺陷。
- 独立复跑受 CPU 调度影响，加速区间约 unpack 101-219x、reuse 7.3-8.2x、evict 49-109x；
  所有运行均满足门槛（5x/3x/10x）。
- 驱逐按每个精度桶各删 1/4 条目；实际字节与条目数同比例下降，未触碰条目反量化值逐位不变，
  `group_size`/`num_groups` 保持不变，重复驱逐不复活。
- `store_dense_kv=false` 时 packed 为单一驻留表示；`_load_kv` 按需反量化。
- 本 AR 不把压缩 KV 回灌 attention，**不宣称端到端模型加速**；GPU/7B 驻留与延迟按
  `specs/archive/AR012-canonical-kv-store/server-experiment.md` 在服务器回填。

### 实测数据（v3.13.0，2026-10-20，Windows / CPU）

真实张量与真实 `SimpleDiffusionLLM`（`specs/archive/AR011-int8-ffn-execution/ar011_bench.py`，
3 次中位[min-max]；`M` 为 token 行数）：

| FFN 形状（M×K×N） | FP32 | INT8 | 加速 | 权重内存 |
|---|---|---|---|---|
| 16×4096×11008 | 29.44 ms | 4.37 ms | **6.74x** | 3.99x |
| 64×4096×11008 | 35.02 ms | 5.90 ms | **5.93x** | 3.99x |
| 128×4096×11008 | 45.67 ms | 9.51 ms | **4.80x** | 3.99x |
| 1×4096×11008 | 4.51 ms | 3.61 ms | 1.25x | 3.99x |
| 64×11008×4096（反向） | 14.85 ms | 8.80 ms | 1.69x | 4.00x |

- 中等模型端到端（4 层 / hidden=1024 / intermediate=4096 / seq=256）：85.1→58.1 ms（**1.47x**），
  logits 相对误差 0.46%。
- 小模型（4 层/8 头/dim64/seq128，3 seed）：logits 相对误差 0.43%，top-1 一致率 99.22%。
- 独立复跑区间：7B 上投影 4.8-6.7x，中等模型 1.37-1.47x；小尺寸存在交叉点
  （128×128×512 为 0.23x，512×256×1024 为 0.59x），如实记录，不外推。
- 7B 预训练模型/GPU 的端到端延迟与困惑度按 `server-experiment.md` 在服务器回填；
  本机不宣称 7B 端到端结果。

### 实测数据（v3.12.0，2026-10-19，Windows / CPU）

真实张量单层，3 次中位（`specs/archive/AR010-dense-precision-pipeline/ar010_bench.py`，
seed=0，预算为同形状 fp32 全量的 18%）。两路径的 precision_map、predicted 字节、
驱逐计数与计划指纹均通过脚本内断言：

| 规模（seq×heads×dim） | v3.11 dict 路径（中位[min-max]） | v3.12 codes 路径（中位[min-max]） | 比值 |
|---|---|---|---|
| 512×32×128 | 175.0 ms [170.1-195.7] | 10.5 ms [10.5-10.7] | 16.67x |
| 2048×32×128 | 623.4 ms [612.1-625.2] | 36.4 ms [35.1-36.8] | 17.11x |
| 4096×32×128 | 1228.5 ms [1208.5-1230.8] | 65.3 ms [65.1-66.4] | 18.81x |
| 8192×32×128 | 2443.2 ms [2416.8-2467.1] | 135.7 ms [132.5-145.2] | **18.00x** |

每档为 3 次重复的中位与 min-max。独立复跑受 CPU 调度影响，比值区间约 15.7-19.6x
（8192×32 约 18.0-19.1x）；所有运行均满足 >=10x 且 8192 档 <200ms。计时覆盖
`allocate + admit + PrecisionPlan` 的生产包装（dict 路径为 `allocate_precision` +
`_admit_precision_budget` + `build`；codes 路径为 `allocate_precision_codes` +
`_admit_precision_codes` + `from_codes`，含 KV 敏感度计算）。4 层真实后端 4 步管线复核：
16 次压缩全部走 plan 路径，reference 回退 0 次。本机为 CPU 实测；GPU/7B 端到端延迟按
`server-experiment.md` 在服务器回填，本机不宣称端到端收益。

### 实测数据（v3.11.0，2026-10-19，Windows / CPU）

真实 `SimpleDiffusionLLM`（8 层/8 头/dim64/seq128，真实前向注意力与 K/V，随机初始化权重；
5 seed 中位，`specs/archive/AR009-rd-kv-allocation/ar009_bench.py`）：

| 总预算 | baseline 加权重建误差 | RD 加权重建误差 | 变化 | 实际字节 | 分配耗时（base/RD） |
|---|---|---|---|---|---|
| 0.9 MB | 0.011157 | 0.011128 | -0.26% | 899904 / 899904 | 88.1 / 85.9 ms |
| 1.2 MB | 0.009997 | 0.009603 | **-3.94%** | 1199424 / 1199424 | 83.4 / 80.4 ms |
| 1.5 MB | 0.008910 | 0.008324 | **-6.58%** | 1499904 / 1499904 | 75.5 / 83.4 ms |

低于 int4 底线的预算（RD + 硬驱逐）：

| 总预算 | baseline 实际字节 | 超预算 | RD+驱逐实际字节 | 是否合规 | 加权误差 | 分配耗时 |
|---|---|---|---|---|---|---|
| 0.3 MB | 892928 | 2.98x | 299264 | 是 | 0.7345 | 101.3 ms |
| 0.5 MB | 892928 | 1.79x | 499712 | 是 | 0.4736 | 87.0 ms |
| 0.7 MB | 892928 | 1.28x | 699392 | 是 | 0.2284 | 83.7 ms |

说明：0.9MB 档位接近 int4 底线，RD 与纯重要性差异在噪声内，如实标注；1.2/1.5MB 档位
RD 收益稳定。固定 inverted-pyramid 深度先验在本模型上未取得稳定收益，因此未作为默认策略。
GPU/7B 复现协议见 `specs/archive/AR009-rd-kv-allocation/server-experiment.md`。

### 实测数据（v3.10.0，2026-10-19，Windows / CPU）

同进程 5 次中位（`specs/archive/AR008-fused-kv-plan/ar008_bench.py`，seed=42），
fused（plan 共享）与 legacy 的 k/v 与 packed 数据逐位一致：

| 形状（seq×heads×dim） | compress legacy→plan | pack legacy→plan | 合计 legacy→fused | 比值 |
|---|---|---|---|---|
| 256×8×64 | 8.43→2.51 ms | 3.72→0.75 ms | 14.66→4.22 ms | 3.48x |
| 512×8×64 | 20.60→2.68 ms | 3.91→1.11 ms | 26.23→9.33 ms | 2.81x |
| 1024×8×64 | 43.15→3.77 ms | 7.73→1.41 ms | 47.26→12.65 ms | 3.74x |
| 512×16×128 | 39.47→5.63 ms | 9.86→1.71 ms | 53.98→15.54 ms | 3.47x |

收益主要来自消除逐条目 scale dict 构造、重复精度分组与重复 gather；int4 打包仍按
`PackedKVCache` 自己的 group size 重新量化以保证字节布局不变。GPU/7B 复现协议见
`specs/archive/AR008-fused-kv-plan/server-experiment.md`（本机不宣称端到端收益）。

### 实测数据（v3.9.0，2026-10-19，Windows / CPU）

可复现编排冒烟档位 s1（真实 `PyTorchModelBackend`，6 步 × 3 次重复，seed=42，
`results/ar007_s1/result.json`）：

| 指标 | 中位 | 区间（min-max） |
|------|------|-----------------|
| Baseline e2e | 41.89 ms | 39.70-46.91 ms |
| Optimized e2e | 1029.81 ms | 1016.80-1060.44 ms |
| Speedup | 0.041x | 0.039-0.044x |
| KV 内存 | 1.75 MB → 1.65 MB（压缩比 1.061x） | — |

该档位用于验证编排、随机性控制与产物隔离，不作为优化收益声称：CPU 上优化管线的调度/压缩开销
真实存在。收益判定请使用 s2/s3 在服务器上按同配置、同 seed、3 次重复执行。

### 实测数据（v3.8.0，2026-10-19，Windows / CPU fallback）

同进程同批对照（`specs/archive/AR006-ffn-pack-vectorization/ar006_bench.py`，seed=42，3 次中位）：

| 指标 | 规模 | 向量化 | 参考（逐通道） | 比值 |
|------|------|--------|----------------|------|
| FFN pack | 256×256 | 2.59 ms | 31.41 ms | 0.082x |
| FFN pack | 1024×256 | 8.41 ms | 116.56 ms | 0.072x |
| `_data` 布局 | 多组（含奇数长度/pack_dim=1） | 字节/dtype/shape/scales/fingerprint/memory 全一致 | — | golden |

该成本为冷启动一次性开销；4 步小模型 cProfile 中 `_do_pack` 曾是 AR005 后最大剩余项（0.071 s，8192 次 `select`）。

**质量**：743 passed / 1 skipped；覆盖率 82.53%；black / isort / flake8 / mypy 全绿。

### 实测数据（v3.7.0，2026-10-19，Windows / CPU fallback）

同进程同批对照（`specs/archive/AR005-overhead-memoization/ar005_bench.py`，seed=42，3 次中位）：

| 指标 | 规模 | 优化路径 | 旧/参考路径 | 比值 |
|------|------|----------|-------------|------|
| kernel 后端解析 | 200 次调用 | 64.2 µs/次 | 2866.1 µs/次（逐次重试） | 0.022x |
| KV INT4 打包 | 128 seq × 8 heads × 128 dim | 3.65 ms | 232.83 ms | 0.016x |
| 字节布局 | 多组 seq/head/dim | `torch.equal` + `byte_offsets` 一致 | — | golden |

根因取证（6 步管线 cProfile，7B 形状）：`_load_cuda_extensions` 被调用 3816 次、累计 12.38 s（占总 27.63 s 的 44.8%），因加载失败结果未被缓存而逐次重试。

**质量**：738 passed / 1 skipped；覆盖率 82.41%；black / isort / flake8 / mypy 全绿。
说明：以上为 CPU 微基准；GPU 服务器可按脚本复核（安装 CUDA 扩展后字节布局须与 CPU 一致）。

### 实测数据（v3.6.0，2026-10-18，Windows / CPU fallback）

同进程同批对照（`specs/archive/AR004-ffn-hotpath/ar004_bench.py`，seed=42，3 次中位）：

| 指标 | 规模 | 优化路径 | 参考路径 | 比值 |
|------|------|----------|----------|------|
| 精度计划指纹 | 256 ch | 0.2 µs | 53.5 µs | 0.004x |
| 精度计划指纹 | 1024 ch | 0.1 µs | 138.4 µs | 0.001x |
| unpack | 256 ch | 1.35 ms | 19.32 ms | 0.070x |
| unpack | 1024 ch | 3.47 ms | 79.67 ms | 0.044x |
| 混合精度量化 | 256 ch | 3.05 ms | 39.45 ms | 0.077x |
| 混合精度量化 | 1024 ch | 8.84 ms | 168.31 ms | 0.053x |

`ffn.on_step`（1024 ch / 4 层 / 12 calls）：优化后 0.66-0.72 ms（3 次中位，多次复跑区间 0.59-0.77 ms）；该值不与该版本之前的跨会话数据做比值宣称。

**正确性**：向量化 vs 保留的参考实现 max\|Δ\| <= 1e-6（unpack/量化含 0.7/0.3 阈值边界、int4 非整除、pack_dim=1、全零通道）；掩码缺陷由 RuntimeError 转为 `mixed_full` 正常且与 full golden 一致；两层不同权重输出各自等于直接计算（|Δ|=0）。
**质量**：732 passed / 1 skipped；覆盖率 82.37%；black / isort / flake8 / mypy 全绿。
上述为 CPU 微基准；7B 级端到端收益待 GPU 服务器按下方协议复现，不预设优越性数字。

### 实测数据（v3.5.0，2026-10-18，Windows / CPU fallback）

**质量与测试**

| 指标 | 实测值 | 说明 |
|------|--------|------|
| 测试通过率 | 704 passed / 1 skipped | 全量套件（新增 IRAP / 预算准入 / ΔA 信号 79 个用例） |
| 测试覆盖率 | 82.04% | pytest-cov（门禁 fail-under 75） |
| 静态检查 | black / isort / flake8 / mypy 全部零告警 | 提交前强制门禁 |
| 向量化 golden 精度 | vec vs 逐元素 reference max\|Δ\| <= 1e-6 | 两契约（logits / probabilities）、sink 边界、EMA 增量 vs 全量 |
| IRAP 常数内存 | 200 步后状态形状/字节不增长 | O(1) 于步数（事务级断言） |

> **v3.5.0 端到端性能：** 本版本变更决策语义（重要性信号与预算准入），未改变
> 压缩链路的数据通路复杂度；按当前项目指令暂停大规模端到端实验，E2E 指标以
> v3.4.0 实测为基线（1.61x 中位，CPU fallback），v3.5.0 的收益与精度影响
> **标记为「机制说明（未验证）」**，复现协议见下节与
> `specs/changes/AR003-irap-signal-integrity/experiment-plan.md`。

### 大规模复现实验指南（高速算力服务器）

目标：在 GPU 服务器上验证 IRAP 决策平面在真实 7B 级 Diffusion LLM 上的
压缩收益与精度影响。协议遵循仓库实验铁律（同机同参同种子、>=3 次重复取
中位、>=3 个规模点、golden/统计证据落档）。

**1. 环境**

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev,cuda]"     # 或按 pyproject 安装 torch 并启用 CUDA kernel
python -c "import torch; print(torch.cuda.get_device_name(0), torch.__version__)"
```

**2. 受测配置（3 个规模点）**

| 规模点 | 模型 | seq_len | denoising steps | batch | 预算 |
|--------|------|---------|-----------------|-------|------|
| S1 冒烟 | SimpleDiffusionLLM（内置，CPU/GPU 均可） | 128 | 6 | 1 | 不设 |
| S2 中等 | 开源 7B Diffusion LLM（如 LLaDA-8B / Dream-7B，HuggingFace 后端） | 512 | 32 | 1 | 6 GB |
| S3 长上下文 | 同上 | 2048 | 32 | 1 | 6 GB（压力触发驱逐） |

模型与超参见 `configs/model_fast_dllm_7b.yaml`；优化开关见
`configs/optimize_full.yaml`（`shadowkv.irap_decay / entropy_weight /
sink_tokens / eviction_pressure_threshold / memory_budget_mb`）。

**3. 对照与运行**

```bash
# 每组配置重复 3 次（不同进程），记录中位数与区间
for i in 1 2 3; do
  python -m shadowinfer optimize --model <MODEL> --backend pytorch \
    --device cuda --num-steps 32 --config configs/optimize_full.yaml \
    --prompt "<固定提示集>" --seed 42 --output outputs/irap_r$i
done
python -m shadowinfer profiler --model <MODEL> --backend pytorch --device cuda \
  --num-steps 32 --config configs/profiler_full.yaml --output outputs/profile
```

**4. 采集指标（落档到 `outputs/irap_r*/` 汇总表）**

- 延迟：e2e_ms、per_step_ms（内部计时与外部墙钟对照，偏差 <= 1% 才采信）；
- 显存：峰值显存、KV packed bytes（应 <= 预算；S3 应观察到 pressure 触发驱逐）；
- 质量：perplexity / 任务准确率相对基线变化（对照：`attn_is_probabilities`
  正确接线 vs 旧双重 softmax 路径）；
- 决策质量：sink token 精度档位（应为 fp32）、精度分布（非 fp32 占比、档位种类）、
  重要性 vs 列向接收注意力 Spearman（>= 0.9）；
- 稳定性：`delta_a` 步序列、驱逐计数、告警计数（触底告警应为 0 除 S3 极端设置）。

**5. 判定**

- 延迟与显存收益：3 次中位，报告区间；
- 精度影响：与基线差异的置信区间（>=3 种子）；
- 任何「优于 baseline」的结论必须有上述数据落档后才可写入 README。

### FFN 热路径复现协议（服务器）

目的：在 GPU 服务器上确认 v3.6.0 FFN 优化在真实维度下的冷启动与稳态收益，并复核多层状态隔离。

| 项 | 配置 |
|----|------|
| 规模点 | intermediate_dim / hidden_dim：256/256、1024/256、11008/4096（7B 级） |
| 路径 | `ffn_mode=mixed`，`use_packed_weights=True`，通道剪枝开/关各一组 |
| 层数 | 4 层不同权重（必须不同，用于 FFN-P1 回归） |
| 重复 | 同会话同批，优化路径 vs 参考路径（`_unpack_reference` / `_quantize_mixed_precision_reference`）各 3 次，报告中位与区间 |
| 硬门禁 | 指纹 <= 10 µs；向量化 <= 参考 50%；golden max|Δ| <= 1e-6；层 2 输出 == 直接计算 |
| 命令 | `python specs/archive/AR004-ffn-hotpath/ar004_bench.py`（可按 `HID/INT/LAYERS` 修改规模） |
| 落档 | 原始输出与配置写入 `specs/archive/AR004-ffn-hotpath/`，结论回写 README 性能基准 |

### 实测数据（v3.4.0，2026-10-17，Windows / CPU fallback）

**质量与测试**

| 指标 | 实测值 | 说明 |
|------|--------|------|
| 测试通过率 | 630 passed / 1 skipped | 全量套件，40 个测试文件 |
| 测试覆盖率 | 81.68% | pytest-cov（门禁 fail-under 75） |
| 静态检查 | black / isort / flake8 / mypy 全部零告警 | 提交前强制门禁 |

**热路径张量化收益（6-step 端到端 cProfile，对照 v3.3.0 逐元素实现）**

| 函数 | v3.3.0 | v3.4.0 | 下降 |
|------|--------|--------|------|
| ShadowKV compress_kv（压缩链路） | 11.43 s | 0.575 s | 19.9x |
| shadowkv.on_step（含决策面） | 24.66 s | 0.954 s | 25.8x |
| 重要性评分（矩阵化） | 8.66 s（98304 次调用） | 0.041 s（24 次调用） | > 200x |
| packed KV 打包 | 3.12 s | 0.199 s | 15.7x |
| FFN 反量化（M5 结果缓存） | 17.13 s（32 次） | 1.257 s（8 次） | 13.6x |

**端到端流水线（SimpleDiffusionLLM，6 steps，CPU，外部墙钟）**

| 指标 | 实测值 |
|------|--------|
| 基线端到端延迟 | 6481.2 ms |
| 优化路径端到端延迟 | 10273.5 ms |
| 优化 / 基线比值 | 1.585x（4 次复测 1.585-1.639，中位 1.61x；v3.3.0 为 7.29x） |
| 优化路径每步延迟 | 948-993 ms（全部真实测量，step 0 含惰性初始化 5419 ms） |
| 内部计时与外部墙钟偏差 | 0.02% |
| 向量化 golden 精度 | 与 reference 实现 bitwise 一致（容差 1e-6） |
| Q-drift 扰动响应 Spearman | >= 0.9（50 组合成扰动统计测试） |

> **CPU fallback 说明：** 当前主机无可用 CUDA，内核走 CPU fallback。v3.4.0 完成决策 / 压缩 / 打包热路径张量化后，优化路径与基线比值由 v3.3.0 的 7.29x 收敛至 1.61x；残余差距主要来自 FFN 反量化支线（unpack 仍为逐元素 Python 热路径，占优化路径采样时间约 57%），已列为下一阶段优化项。框架的目标收益场景是端侧 GPU + 真实 7B Diffusion LLM——GPU 环境启用 CUDA kernel 后收益进一步兑现。

完整基准套件：

```bash
python benchmarks/run_benchmarks.py --model Fast-dLLM-v2-7B --suite full
```

输出包括：

- `report.md` / `report.html`：端到端对比报告；
- `latency.csv` / `roofline.png`：延迟曲线与屋顶线分析；
- 回归历史 `benchmarks/results/regression_history.jsonl`。

---

## 设计决策

以下是项目关键架构决策的摘要，完整讨论已合并到本文档相关章节。

| ADR | 决策 | 理由 |
|-----|------|------|
| ADR-001 | 使用 ShadowKV 而非统一量化 | 关键 token 不可控损失 vs 动态 per-token-head 精度分配，压缩比可控且精度损失可追踪。 |
| ADR-002 | Q-drift 步感知调度 | Diffusion LLM 不同 step 误差敏感度不同，早期可激进、后期需保守。 |
| ADR-003 | 基于 Entropy 的重要性评分 | Attention 权重分布 entropy 能反映 token-head 的信息覆盖范围，无需额外训练。 |
| ADR-004 | 混合精度优于静态量化 | 静态量化固定压缩比、固定误差；混合精度按重要性自适应，可解释且与调度协同。 |
| ADR-005 | 多 Agent 架构 + 声明式策略 DSL | 将 Profiler / ShadowKV / Q-drift / FFN 解耦，通过 Profiling Bus 通信；仲裁阈值用 YAML 策略描述，可热更新、可审计。 |
| ADR-006 | 安全网降级优先级高于策略仲裁 | pre-flight 失败触发的 conservative 降级在同 step 内不可被仲裁放宽，保证生产安全机制的有效性。 |

---

## 项目结构

```
ShadowInfer/
├── .github/                        # GitHub 模板与工作流
│   ├── workflows/
│   │   └── ci.yml                  # CI/CD 流水线
│   ├── ISSUE_TEMPLATE/             # Issue 模板
│   └── PULL_REQUEST_TEMPLATE.md
├── assets/
│   └── banner.svg                  # 封面横幅（矢量）
├── LICENSE                         # MIT 许可证
├── README.md                       # 本文档（主文档）
├── ROADMAP.md                      # 项目路线图
├── pyproject.toml                  # 包元数据与工具配置
├── requirements.txt                # 依赖列表
├── requirements-lock.txt           # 锁定依赖
├── scripts/
│   ├── run.py                      # CLI 入口脚本
│   └── run_experiment.py           # 可复现实验编排（v3.9.0）
├── configs/                        # 配置示例
│   ├── experiments/                # s1-s5 实验档位
│   │   ├── s1_cpu_smoke.yaml
│   │   ├── s2_7b_gpu.yaml
│   │   ├── s3_longctx_gpu.yaml
│   │   ├── s4_edge_eviction_gpu.yaml
│   │   ├── s5_dense_precision_gpu.yaml
│   │   ├── s6_int8_ffn_gpu.yaml
│   │   └── model_7b_longctx.yaml
│   ├── model_fast_dllm_7b.yaml
│   ├── optimize_full.yaml
│   ├── policy_default.yaml
│   ├── profiler_full.yaml
│   └── serving.yaml
├── shadowinfer/                    # 核心包
│   ├── __init__.py
│   ├── __main__.py                 # python -m shadowinfer
│   ├── orchestrator.py             # 中心编排器
│   ├── core/                       # 基础抽象 + ModelBackend
│   │   ├── backends/               # 模型后端实现
│   │   │   ├── pytorch_backend.py
│   │   │   ├── huggingface_backend.py
│   │   │   ├── registry.py
│   │   │   └── ...
│   │   ├── model_backend.py        # 后端抽象
│   │   ├── policy.py               # 声明式策略 DSL
│   │   ├── scheduler.py            # 调度器
│   │   └── structs.py              # 核心数据结构
│   ├── serving/                    # HTTP serving 脚手架
│   ├── models/                     # 内置 Diffusion LLM 模型
│   ├── execution/                  # 跨步复用/执行控制（v3.15.0）
│   │   └── step_reuse.py
│   ├── profiler/                   # Profiling Agent
│   ├── shadowkv/                   # KV cache 优化
│   │   ├── shadowkv_agent.py
│   │   ├── kv_cache_manager.py
│   │   ├── packed_kv_cache.py
│   │   ├── precision_plan.py       # Plan/Run 精度计划 + 稠密 codes（v3.12.0）
│   │   ├── rd_allocator.py         # 率失真预算分配/硬驱逐（v3.11.0）
│   │   ├── decision_plane.py
│   │   ├── importance_model.py
│   │   └── eviction_policy.py
│   ├── qdrift/                     # 步感知调度
│   ├── ffn_optimizer/              # FFN 优化
│   │   ├── ffn_optimizer_agent.py
│   │   ├── int8_linear.py          # 可执行 INT8 FFN（v3.13.0）
│   │   └── packed_weight.py
│   ├── kernels/                    # CUDA 算子 + CPU fallback
│   ├── distributed/                # 多 GPU 脚手架
│   ├── ab_testing/                 # A/B 测试框架
│   ├── observability/              # 指标、追踪、面板
│   ├── benchmarking/               # Benchmark 套件
│   ├── engineering/                # 生产安全组件
│   ├── web_profiler/               # Streamlit 面板
│   └── utils/                      # 公共工具
├── benchmarks/                     # Benchmark 脚本
├── dashboards/                     # Grafana 面板模板
├── exporter/                       # Prometheus 导出器
├── notebooks/                      # Jupyter 笔记本
├── specs/                          # SDD 过程资产（AR 需求/设计/任务）
├── tests/                          # 单元与集成测试
└── outputs/                        # 生成的报告（gitignore）
```

---

## 开发贡献

感谢你对 ShadowInfer 的兴趣。欢迎通过以下方式参与：

- 在 [GitHub Issues](https://github.com/ShaneLiu04/ShadowInfer/issues) 报告 bug 或提出需求；
- 在 [Pull Requests](https://github.com/ShaneLiu04/ShadowInfer/pulls) 提交改进；
- 改进文档、教程或翻译；
- 分享你的使用场景与复现结果。

优先贡献方向（详见 [ROADMAP](ROADMAP.md)）：CUDA kernel 实现与测试、真实 Diffusion LLM 适配器、CPU 优化路径向量化、端侧 GPU 基准测试。

### 开发环境

```bash
git clone https://github.com/ShaneLiu04/ShadowInfer.git
cd ShadowInfer
python -m venv .venv
source .venv/bin/activate  # Linux/Mac
pip install -e ".[dev]"
```

### 代码规范

提交 PR 前请运行（全部通过才可合并）：

```bash
black --check shadowinfer/ tests/ benchmarks/ exporter/ scripts/
isort --check-only shadowinfer/ tests/ benchmarks/ exporter/ scripts/
flake8 shadowinfer/ tests/ benchmarks/ exporter/ scripts/ --max-line-length=100 --extend-ignore=E203,W503
mypy shadowinfer/ --ignore-missing-imports
pytest
```

### PR 流程

1. Fork 仓库并创建功能分支；
2. 保持提交清晰、聚焦；
3. 为新功能添加或更新测试；
4. 更新相关文档（本文档 `README.md`）；
5. 确保所有检查通过；
6. 提交 Pull Request 并清晰描述变更。

---

## 引用

如果 ShadowInfer 对你的研究或工程有帮助，欢迎引用：

```bibtex
@software{shadowinfer2026,
  title  = {ShadowInfer: Edge-Optimized Inference Framework for Diffusion LLMs},
  author = {ShadowInfer Contributors},
  year   = {2026},
  url    = {https://github.com/ShaneLiu04/ShadowInfer},
  version = {3.3.0}
}
```

---

## 版本历史

### v3.18.0（2026-10）

- **置信度调度**：新增 `shadowinfer/execution/block_decoding.py`；`BlockDecodingConfig` 冻结校验（块宽/轮数/阈值/页参数合法性），`schedule_round` 向量化（过阈置信度位置并行解掩 + 无过阈保底 top-1 + argmax 并列取最小位）；`GenerateResult` 携带序列与统计。
- **尾块活跃前向**：`SimpleDiffusionLLM.forward_block` 只计算 W 个块行（prefix_kv 条件 attention、块内双向、活跃行 K/V 返回）；`attention_mask` 三层透传（forward_step / block.forward / _compute_block 可选参默认 None 逐位回归）。
- **段因果掩码**：`segment_causal_mask`（`seg(i)=0 if i<P else 1+(i-P)//W`，`seg(i)<seg(j) -> -inf`）保证前缀 KV 时间不变；替代开发期发现自相矛盾的多块块因果公式（design 4.5 erratum 落档，单尾块情形逐位兼容）。
- **生成管线**：`PyTorchModelBackend.generate`（预填充 append -> 逐块轮循环 -> 保底完成 -> sealing 前向固化 KV -> append；每层 fresh `PagedKVStore`，fp32/fp16/int8 页精度平面，轮间零页写，fail-fast 六类校验无副作用，`capture_block_logits` 可观测，`last_generate_stores` 页驻留取证）；`blockwise_generate_baseline` 全量重算参考环路（同调度/校验/保底语义，golden）。
- **生成统计**：14 键恒存在（blocks / total_rounds / computed_rows_total / kv_write_rows_total / forced_completions / total_time_ms / page_precision / rounds_per_block / unmasked_per_round / fallback_per_round / mean_confidence_per_round / rows_saved_vs_baseline / block_time_ms / unmask_positions_per_round），解析一致性与页账本交叉验证由 UT 证明。
- **实测（CPU，四臂 × 3 规模 × 3 seeds，注册断言全过）**：fp32 臂 **2.29x / 3.68x / 7.08x**（S=256/512/1024 中位）与全量重算参考环路 token 序列 + 全轮 logits 逐位一致（relL2 <= 3.25e-7）；计算行缩减 4.9/8.4/15.4x；KV 行恰写一次（== P+G，27/27）；fp16 relL2 <= 1.26e-5 全逐位；int8 一致率 >= 0.996（对齐点 relL2 3.6e-4；调度置换伪影机理与 A6 判据三轮修订落档）；S=256 低于 3x 与全并行态比率收敛 ~1.18x 均如实标注；阈值 0.9 -> 0.005 校准史落档。
- **复现资产**：`specs/archive/AR016-blockwise-parallel-decoding/`（srs/design/tasks/bench-results/server-experiment/serve_arm），`configs/experiments/s11_blockwise_decoding_gpu.yaml`；1128 passed / 1 skipped，覆盖率 84.56%。

### v3.17.0（2026-10）

- **宏块模式分类**：新增 `shadowinfer/execution/macro_mode.py`；`MacroModeState` 按宏块将每层序列分类为 SKIP / DELTA / REFRESH（单门 `skip_threshold` + 比例门 `delta_min_active_ratio`），分类向量化（单层单次构建参考/有效行掩码，无逐块 `.item()`/`nonzero`）；warmup 步与 `force_refresh_every` 周期强制 REFRESH 作精度保险；entry 缺失/shape 失效逐块删除并强制全行活跃。
- **行粒度分段前向**：`SimpleDiffusionLLM` / `DiffusionTransformerBlock` 支持按活跃行子集前向（全活跃捷径逐位等价、全 SKIP 残差回放、部分活跃行切片 + key 侧 scatter 组装）；三态 record（REFRESH 整块、DELTA 活跃行、SKIP 不触碰）；`last_modes`/`last_active_rows`/`last_attention` 可观测。
- **行级页覆写**：`PagedKVStore.rewrite_tokens` 校验全先于变异（失败原子性）、驱逐行拒写、跨页自动拆分、未覆写行 payload 逐位不变；`rewritten_tokens`/`rewritten_pages` 记账。
- **后端分派与降级**：`configure_execution` 六新参（`mode_control_enabled` 等，paged+replace 显式 opt-in，与 step reuse 互斥 ValueError）；写回三态分派（SKIP 零页写、DELTA 活跃行、REFRESH 整块；全 REFRESH 走 clear+append 快路径，页布局逐位相同）；模型调用或写回失败自动降级 dense 重算并 reset；shape 突变清空模式状态；执行统计新增八键恒存在；默认关闭逐位回归。
- **实测（CPU，双轨迹 × 四臂 × 3 规模 × 3 seeds）**：f2 保险臂全步 relL2 <= 2.649%、末半程冻结率 0.667（终版预注册断言全过）；**端到端无加速收益（0.64-1.04x）如实报告**，f-扫描证明 >=2x 与 <=2% 在玩具负载不可同时达成，机制交付物为正确性、误差有界语义、漂移门自愈与 SKIP 零页写；三轮预注册历史完整落档。
- **复现资产**：`specs/archive/AR015-macro-block-mode-control/`（srs/design/tasks/st-cases/bench/server-experiment），`configs/experiments/s10_mode_control_gpu.yaml`；1016 passed / 1 skipped，覆盖率 84.25%。

### v3.16.0（2026-10）

- **PagedKVStore 核心机制**：新增 `shadowinfer/shadowkv/paged_kv_store.py`；`[B,H,page_tokens,D]` 模型布局页存储，append 增量入页（仅量化新 token，向量化）、封存页 id 与数据指针不可变、读取单遍 gather（零 permute/零 repack）；开放页填充以整页对象替换，页张量按实际 token 数精确分配。
- **行级量化逐位对齐**：fp16 直存、int8 行级 scale（dim=-1）、int4 行组 scale + 行级字节打包，与 `PackedKVCache.pack→unpack` 对同一元素 `torch.equal` 级一致（fp32/fp16/int8/int4 四精度 golden）。
- **分页驱逐**：`evict_tokens(token_head_set)` 支持全 head token 删除（整页清空真实释放字节并退出页序）与部分 head 零值行（fp 置 0 / int8 q=0 / int4 打包字节 0x88 解码 0.0）；槽位空闲链复用，`_evicted_pairs` 防二次计数与复活；驱逐前后未触碰元素逐位不变。
- **后端与编排接线**：`kv_execution="paged"`（`page_size` 可配，默认 32）；append 仅切新 token 入页、raw_cache 走 gather、replace 清页重入；shape mismatch 与 fallback 密集路径自适应；`KVCacheEntry.paged_kv` 字段与 orchestrator `_entry_kv_tensors` paged 分支；执行统计新增 `paged`/`page_size`/`page_count`/`open_pages`/`appended_tokens`/`appended_tokens_total`/`repack_tokens_saved`/`page_evictions` 八键。
- **实测（CPU，3 seed 中位）**：KV 执行平面微基准加速 9.30x（S=512）/ 10.16x（S=2048），每步延迟斜率比为 packed 基线的 0.088-0.114，Python 堆瞬态峰值降约 51-57x；fp16 逐位等价。
- **复现资产**：`specs/archive/AR014-paged-kv-store/`（srs/design/st-cases/bench/server-experiment），`configs/experiments/s9_paged_longctx_gpu.yaml`；921 passed / 1 skipped，覆盖率 83.78%。

### v3.15.0（2026-10）

- **跨步块复用**：新增 `shadowinfer/execution/step_reuse.py`；逐 block 缓存输入/输出/K/V，相对 L2 漂移低于阈值时跳过 Q/K/V/O、attention 与 FFN，输出 `cached_out + (h - cached_in)`；warmup、阈值、跳过率与误差全部可观测，默认关闭。
- **Diffusion KV 生命周期**：`kv_lifecycle=replace` 让固定序列 denoising 每步替换 K/V，不再把上一步 concat 成 past（旧 append 保留为默认，行为逐位不变）。
- **紧凑 KV 闭环执行**：`kv_execution=packed` 以 `PackedKVCache` 作为每层驻留表示；append 模式下下一步 attention 从 packed 反量化读取 past。复用层直接保留上一步 packed 对象，不重复 pack。
- **编排接线**：`execution` 配置在 `initialize` 下发，`run_baseline`/`run_optimized` 每次 reset，执行统计随 forward 返回。
- **实测（CPU，3 seed × 3 repeats）**：reuse 1.76-1.94x、logits relL2 0.94-0.96%、跳过 50% block；int8 packed 数据平面驻留为 FP16 dense 的 53%（全口径 72%）；组合 1.09-1.44x。
- **复现资产**：`specs/archive/AR013-step-reuse-packed-execution/`（srs/design/tasks/bench/server-experiment），`configs/experiments/s8_step_reuse_gpu.yaml`；876 passed / 1 skipped，覆盖率 83.37%。

### v3.14.0（2026-10）

- **紧凑 KV Store 热路径**：`PackedKVCache.unpack` / `apply_reuse_mask` / `evict` 全量张量化；unpack 按精度桶批量 gather+反量化，复用按 flat 索引批量拷贝，驱逐改为桶内就地删除（不再 unpack/re-pack）。
- **INT4 正确性**：修正 `B>1` 时 scale 按 `[entry]` 而非 `[entry,batch]` 读取的缺陷（B=2 每 batch 相对 MSE 0.0075/0.0072）；压缩、打包、驱逐与成本模型统一 `int4_group_size`（默认 16），打包复用压缩阶段的 INT4 payload，不再二次量化。
- **单源驻留**：新增 `store_dense_kv`（packed 模式下默认 `false`），packed cache 成为唯一驻留表示；`_load_kv` 按需反量化；新增 `resident_bytes()` 并统一 `KVCacheManager` 内存口径（dense shadow 存在时同时计入，超预算时先释放 shadow 再驱逐 packed 条目）。
- **实测（CPU，3 次中位）**：unpack 512-2048 为 101-219x，复用 7.3-8.2x，驱逐 49-109x；驱逐 1/4 条目后字节真实下降 25.0%，未触碰条目逐位不变。
- **复现资产**：`specs/archive/AR012-canonical-kv-store/`（srs/design/st-cases/bench/server-experiment），`configs/experiments/s7_kv_store_gpu.yaml`；853 passed / 1 skipped，覆盖率 83.23%。

### v3.13.0（2026-10）

- **可执行 INT8 FFN**：新增 `INT8Linear`（per-output-channel 权重 scale + per-token 激活 scale，真实 `_int_mm`），替代原有“打包但不执行”的 FFN 路径；`_int_mm` 不可用时确定性回退且仅告警一次。
- **模型级接线**：`SimpleDiffusionLLM.enable_int8_ffn()` 替换全部 FFN 上下投影并默认释放 FP32 权重，支持 `keep_fp32_reference`/`restore_fp32_ffn`；PyTorch 与 HuggingFace 后端均可启用（HF 按 `mlp.up_proj/down_proj/gate_proj`、`mlp.c_fc/c_proj` 匹配）。
- **编排与配置**：修复 `ffn:` / `ffn_optimizer:` 配置键不一致；启用 INT8 执行后跳过重复的 FFN Agent 记账路径，`_run_ffn` 返回真实执行统计。
- **实测（CPU，3 次中位）**：7B 上投影 M=16/64/128 为 6.74x/5.93x/4.80x（复跑 4.8-6.7x），权重内存 3.99x 降低；中等模型端到端 1.47x（复跑 1.37-1.47x）；小模型 logits 误差 0.43%、top-1 99.22%；小尺寸交叉点如实记录。
- **复现资产**：`specs/archive/AR011-int8-ffn-execution/`（srs/design/tasks/st-cases/bench/server-experiment），`configs/experiments/s6_int8_ffn_gpu.yaml`；812 passed / 1 skipped，覆盖率 82.99%。

### v3.12.0（2026-10）

- **稠密精度码 KV 热路径**：新增 `map_to_codes`/`codes_to_map`/`PrecisionPlan.from_codes`；`build` 委托 `from_codes`，dict 与 codes 两条路径构造完全一致；`int8[seq, heads]` 成为 on_step 的唯一精度状态。
- **稠密预算准入**：`admit_codes` 以稳定 argsort + 前缀和闭式求解降档与驱逐，语义与 `admit_budget` 逐项一致（含 sink/复用保护、RD 开关、驱逐计数），且不修改输入。
- **生产接入**：`allocate_precision_codes`、`_admit_precision_codes`、`compress_kv(precision_codes=...)`；决策缓存保存准入后 codes；`dense_precision_enabled`（默认 true）可切回 v3.11 dict 路径做 A/B。
- **缺陷修复**：`_compress_kv_with_plan` 改用连续 `torch.zeros` workspace，非连续 K/V 不再触发 stride 错误并静默回退；4 层真实管线 16/16 走 plan 路径。
- **实测（CPU，3 次中位[min-max]）**：元数据阶段 8192×32 2443.2→135.7 ms（18.00x，复跑 18.0-19.1x）；4096×32 18.81x；2048×32 17.11x；512×32 16.67x；输出逐位一致。
- **复现资产**：`specs/archive/AR010-dense-precision-pipeline/`（srs/design/tasks/bench/pipeline-check/server-experiment），`configs/experiments/s5_dense_precision_gpu.yaml`；792 passed / 1 skipped，覆盖率 82.99%。

### v3.11.0（2026-10）

- **率失真 KV 预算分配**：新增 `rd_allocator`，以 `importance × KV absmax²` 排序降档；sink/复用条目保护；`rd_allocation_enabled` 默认开启，关闭时与 v3.10.0 逐位一致。
- **可选硬驱逐**：`hard_eviction_enabled` 在预算低于 int4 底线时驱逐低价值 token-head（`evicted`，0 字节）；`PrecisionPlan`/compress/pack/unpack/evict 全链路支持，解包为零且不复活；内存上限 100% 合规。
- **实测（真实模型，5 seed）**：同字节加权 KV 重建误差 1.2MB -3.94%、1.5MB -6.58%；0.9MB -0.26%（噪声内）；低预算 baseline 超限 1.28-2.98x，RD+驱逐全部合规。
- **PyramidInfer 分析**：采用 heavy-hitter 保留思想；固定深度先验实测收益 0-4% 且不稳定，未作为默认，避免无依据层偏置。
- **复现资产**：`specs/archive/AR009-rd-kv-allocation/`（srs/design/tasks/st-cases/bench/server-experiment）；772 passed / 1 skipped，覆盖率 82.93%。

### v3.10.0（2026-10）

- **FlashInfer 式 Plan/Run 融合**：`PrecisionPlan` 一次构建精度码/分组/flat 索引/复用索引/SHA-256 指纹，`compress_kv` 与 `PackedKVCache.pack` 共享；生产路径跳过未使用的 scale dict，pack 复用 fp16/int8 量化结果。
- **正确性**：legacy 与逐元素 reference 保留为 golden；k/v、`scale_*`、`packed_data`、`byte_offsets`、`memory_bytes` 逐位一致（含 batch=2、奇数 head_dim int4、复用掩码）。
- **实测（CPU，5 次中位）**：compress+pack 512×8×64 26.23→9.33 ms（2.81x）；1024×8×64 47.26→12.65 ms（3.74x）；512×16×128 53.98→15.54 ms（3.47x）。
- **复现资产**：`specs/archive/AR008-fused-kv-plan/`（srs/design/tasks/st-cases/bench/server-experiment）；759 passed / 1 skipped，覆盖率 82.87%。
- **服务器实验**：`scripts/run_experiment.py` + `configs/experiments/s2_7b_gpu.yaml` / `s3_longctx_gpu.yaml`，按 `server-experiment.md` 回填。

### v3.9.0（2026-10）

- **实验可复现编排**：新增 `scripts/run_experiment.py`，真实后端 baseline/optimized 一键对照；固定 seed、每轮清空后端 KV cache、配置 SHA-256、环境与 git SHA 取证、`--repeats` 输出 median/min/max。
- **产物隔离**：Profiler 的 baseline/optimized/comparison 全部写入 `run_i/`，修复跨轮 KV cache 状态泄漏（baseline 口径漂移）。
- **规模档位**：s1 CPU 冒烟内置；`configs/experiments/` 提供 s2（7B GPU）与 s3（8K 长上下文 + 4GB）模板；模型模板支持嵌套结构自动展平；显式拒绝 mock/auto 后端。
- **实测（CPU，6 步 × 3 次）**：baseline 41.89 ms（39.70-46.91），optimized 1029.81 ms（1016.80-1060.44），KV 1.75→1.65 MB；该档位不作收益声称。751 passed / 1 skipped，覆盖率 82.48%。
- **复现资产**：`specs/archive/AR007-experiment-reproducibility/`（srs/design/tasks/st-cases）。

### v3.8.0（2026-10）

- **FFN 打包向量化**：`_do_pack` 按精度分组批量打包（fp32/fp16/int8/int4，pack_dim 0/1）；保留 `_do_pack_reference` 作 golden。
- **逐位一致**：`_data` 键/张量/dtype/shape、`byte_offsets` 语义、内存记账、精度计划指纹与参考完全相同（含奇数通道长度与单组自适应）。
- **实测（CPU，同进程 3 次中位）**：1024×256 pack 116.56 ms -> 8.41 ms（0.072x）；743 passed，覆盖率 82.53%。
- **复现资产**：`specs/archive/AR006-ffn-pack-vectorization/`（srs/design/tasks/baseline-profile/bench）。

### v3.7.0（2026-10）

- **Kernel 解析记忆化**：成功/失败结果均缓存 + 双检锁；不可用告警只发一次。消除热路径上每次 kernel 调用重试 `cpp_extension.load`（本机 3816 次 / 12.4s）。
- **KV INT4 批量打包**：`PackedKVCache._pack_int4_rows` 按行批量打包，奇数行内补零；k/v 字节、`byte_offsets`、scales 与 `_do_pack_reference` 逐位一致。
- **实测（CPU，同进程 3 次中位）**：loader 0.022x；INT4 pack 0.016x；738 passed，覆盖率 82.41%。
- **复现资产**：`specs/archive/AR005-overhead-memoization/`（srs/design/tasks/baseline-profile/bench）。

### v3.6.0（2026-10）

- **FFN 热路径优化**：精度计划指纹 pack 时缓存（O(1)）、quantization_info 按层缓存一次、unpack/混合精度量化分组向量化（参考实现保留 golden）。
- **逐层状态隔离（FFN-P1 修复）**：权重、量化态、重要性、掩码、统计按 layer_id 隔离；on_init 清空并重置累计；超层告警去重。
- **掩码正确性修复**：`_mixed_compute` down 投影响应列掩码；全 False 掩码输出零并告警一次（mixed/full 双路径）。
- **实测（CPU，同进程同批 3 次中位）**：on_step 0.66-0.72 ms（绝对中位）；指纹同批比值 0.001-0.004x；unpack 0.044-0.070x；量化 0.053-0.077x；732 passed，覆盖率 82.37%。
- **复现资产**：`specs/archive/AR004-ffn-hotpath/`（srs/design/tasks/baseline-profile/bench 脚本）与 README 服务器复现协议。

### v3.5.0（2026-10）

- 决策信号正确性修复（KV-P2/P3）：attention 输入契约显式化（`attn_is_probabilities`，真实后端 post-softmax 不再二次 softmax，logits 路径仅一次归一化）；重要性主体由 query 行熵改为 KV 列向接收注意力（对齐 H2O / StreamingLLM），修复 sink token 被判最低档的轴向反转；
- IRAP 接收注意力传播（FR3.3）：跨步 EMA（`irap_decay` 默认 0.9）取代无界 `importance_history`，决策状态常数内存（实测 200 步零增长）；rank 归一化 + 行熵辅助（`entropy_weight` 默认 0.3）+ sink 前缀硬保护（`sink_tokens` 默认 4）；公式单源化（`ImportanceModel` 委托共享核心）；
- 预算闭环（KV-P4/P8/P10）：pre-pack 降档准入（单次打包零重打包）、`memory_pressure` 消费与跨层驱逐（10% 迟滞）、per-precision 字节精确记账（修复 8x 低估）；
- ΔA 跨步差分信号（FR3.5）：每步导出并一步滞后供 Q-drift 可选融合（`use_importance_drift` 默认关闭，关闭时逐位不变）；
- 顺带修复：小 `head_dim`（不可被 64 整除）INT4 量化直接抛错缺陷（分组自适应）；
- 质量门禁全绿：`704 passed, 1 skipped`，覆盖率 `82.04%`；端到端收益与精度影响标记「机制说明（未验证）」，复现协议见《大规模复现实验指南》。

### v3.4.0（2026-10）

- 热路径张量化（M1）：ShadowKV 重要性评分 / 精度分配 / 复用决策 / 压缩 / 打包全链路向量化，压缩链路耗时 11.43s → 0.575s（19.9x），golden 测试保证与逐元素 reference bitwise 一致（容差 1e-6）；异常路径自动回退 reference 并记录告警；
- 指标真实化（M6）：latency.per_step_ms / e2e_ms 真实测量，修复 per_step 硬编码 0.0 与内外口径矛盾；内部计时与外部墙钟对照偏差 0.02%；
- 记忆化（M5）：同 (step, layer) 同输入决策缓存（指纹校验 + step 边界失效 + 异常旁路）与 FFN 反量化结果缓存（repack 自动失效）；on_init 修复累计状态未重置缺陷；kv memory_mb 改为真实节省字节口径；
- 实测漂移灵敏度（M3-lite）：Q-drift sensitivity 改为位置先验（权重 0.3）与实测扰动响应（tanh 归一化）凸组合，drift_score 修复恒 1.0 塌缩恢复动态范围；50 组合成扰动 Spearman >= 0.9 统计测试；
- 端到端：优化 / 基线比值 7.29x → 1.61x（CPU fallback，4 次实测中位）；残余差距为 FFN 反量化支线（下一阶段优化项）；
- 质量门禁全绿：`630 passed, 1 skipped`，覆盖率 `81.68%`。

### v3.3.0（2026-10）

- 修复安全网 pre-flight 降级被策略仲裁覆盖的逻辑缺陷：安全降级在同 step 内不可被仲裁放宽（ADR-006）；
- 清理 flake8 全部 20 处告警（未用导入、死代码、超长行）；
- 统一清理 79 处已删除文档的失效引用，全部改指 README.md 对应章节；
- README 全方位重塑：深色科技风 SVG 封面、徽章行、关键指标条、实测数据口径统一、架构图重绘、Citation 章节；
- 新增本地开发契约 AGENTS.md（不入库）：创新完成即更新 README、AR 完成即推送 Gitee；
- 质量门禁全绿：`581 passed, 1 skipped`，覆盖率 `80.84%`。

### v3.2.2（2026-07）

- 新增 `AgentPluginRegistry`，支持通过 `shadowinfer.agents` entry-point 注册第三方 Agent；
- Orchestrator 自动调度插件 Agent，输出写入 `step.outputs` 并随 STEP_RESULT 广播；
- 新增 `AsyncTaskExecutor`，支持后台任务提交、超时、取消与结果查询；
- 新增 `UncertaintyAwareEarlyStopper` 并接入 Orchestrator step 循环；
- 修复 HuggingFace 后端 `local_files_only=True` 离线 fallback。

### v3.2.1（2026-07）

- 新增 `ImportanceModel`、`KVDecisionPlane`、`EvictionPolicy`，构建 ShadowKV 决策平面；
- 支持 KV Cache 驱逐与预取；
- 新增 FFN 动态通道剪枝；
- 迁移日志到 `structlog`，支持 JSON、轮转、动态级别；
- 修复 HuggingFace 后端 `DynamicCache` 解包问题；
- 合并并精简所有文档到 `README.md`。

### v3.2.0（2026-07）

- 新增 `BackendRegistry` 与 entry-point 插件发现；
- 新增 vLLM / SGLang 后端 scaffolding；
- 新增声明式 Policy DSL，策略引擎接入 Orchestrator 仲裁；
- 重构 `Orchestrator`，支持 `run_stream()`、`cancel()`、event-sourcing snapshots；
- 新增 Kernel Auto-Tuning Cache 与 `KernelDispatcher`；
- 新增 `StepState` / `PipelineContext`。

### v3.1.0（2026-06）

- Prometheus / OpenTelemetry 可观测性；
- `PackedFFNWeight` 与稀疏 GEMM 路径；
- 性能回归追踪与 `shadowinfer benchmark` CLI；
- `SimpleDiffusionLLM`、PyTorch 后端、HuggingFace 后端；
- ShadowKV token / token-head 级复用。

### v3.0（2026-05）

- 手写 CUDA Kernel（INT8/INT4 量化、稀疏 GEMM）；
- CI/CD 流水线；
- 真实模型 Benchmark；
- 分布式推理脚手架；
- A/B 测试框架；
- Grafana Dashboard。

### v2.0（2026-04）

- 可观测性 + 基准测试 + 工程化组件；
- 完整 ADR、技术规范、工程文档。

### v1.0（2026-03）

- 四大核心模块 + 基础架构。

---

## 许可证

本项目采用 [MIT License](LICENSE) 开源。
