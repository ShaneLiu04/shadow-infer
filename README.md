# ShadowInfer

> 面向端侧的 Diffusion LLM 推理优化框架
>
> 版本：v3.2.2 | License：MIT

ShadowInfer 是一套用于 Diffusion LLM端侧/车端 GPU 推理的优化与评测框架。它通过 **ShadowKV 决策平面**、**Q-drift 步感知调度**、**FFN 动态通道剪枝** 和多 Agent 协作架构，在 4-8 GB 显存约束下实现低延迟、低精度损失的推理。

---

## 目录

- [项目简介](#项目简介)
- [核心特性](#核心特性)
- [技术亮点](#技术亮点)
- [快速开始](#快速开始)
- [架构设计](#架构设计)
- [核心模块](#核心模块)
- [核心算法](#核心算法)
- [工程化与生产安全](#工程化与生产安全)
- [CLI 使用](#cli-使用)
- [模型后端](#模型后端)
- [可观测性与回归测试](#可观测性与回归测试)
- [分布式与 A/B 测试](#分布式与-ab-测试)
- [性能基准](#性能基准)
- [设计决策](#设计决策)
- [项目结构](#项目结构)
- [开发贡献](#开发贡献)
- [版本历史](#版本历史)
- [许可证](#许可证)

---

## 项目简介

Diffusion LLM 与自回归 LLM 的关键差异在于：它通过 **固定步数的 denoising** 过程从噪声生成文本，每步都是完整前向传播。这意味着：

- KV Cache 每步重建，但相邻 step 之间高度相似，具备复用空间；
- 不同 denoising step 对计算误差的敏感度差异显著（早期噪声高、可激进优化，后期噪声低、需保持精度）；
- 端侧 GPU 显存小、延迟敏感，需要算法与工程化手段结合。

ShadowInfer 将这些优化机会封装为四个核心 Agent，由统一编排器（Orchestrator）协调，提供从性能分析、策略优化到生产部署的完整链路。

---

## 核心特性

| 特性 | 说明 |
|------|------|
| **ShadowKV 决策平面** | 形式化 Importance / Drift / Memory-Pressure 三维平面，统一输出 precision / reuse / eviction 决策。 |
| **字节级混合精度 KV Cache** | 支持 FP16 / INT8 / INT4 按需存储，per-token-head 动态分配精度。 |
| **重要性感知驱逐** | 显存预算超限时按重要性回收低价值 token-head，支持 LRU / age 策略。 |
| **KV Cache 预取** | 基于 Q-drift 敏感度预测下一 step 复用 mask，提前 stage KV 数据。 |
| **Q-drift 步感知调度** | 根据 step 敏感度和激活漂移动态选择 ShadowKV / FFN 工作模式。 |
| **FFN 动态通道剪枝** | 运行时按通道重要性或激活能量剪枝 FFN 中间通道，降低 FLOPs 与访存。 |
| **模型后端抽象** | 内置 `SimpleDiffusionLLM`、`PyTorchModelBackend`、`HuggingFaceModelBackend`，支持 entry-point 插件扩展。 |
| **Agent 插件系统** | 通过 `shadowinfer.agents` entry-point 注册第三方 Agent，Orchestrator 自动调度并收集输出。 |
| **异步 Agent 执行** | `AsyncTaskExecutor` 将 Profiler 聚合、A/B 统计、面板渲染等任务放到后台线程池，带超时与取消。 |
| **不确定性感知早停** | `UncertaintyAwareEarlyStopper` 根据相邻 step 输出变化动态停止 denoising，节省冗余计算。 |
| **结构化日志** | 基于 `structlog` 的 JSON 行日志，支持按大小/时间轮转与运行时动态级别调整。 |
| **生产安全网** | 热配置重载、熔断器、五级优雅降级、令牌桶限流、健康监控。 |
| **可观测性** | Prometheus 指标、OpenTelemetry 链路、Streamlit / Grafana 面板。 |
| **A/B 测试与回归追踪** | 策略分流 + 统计显著性检验 + JSONL 历史性能回归检测。 |
| **分布式脚手架** | Pipeline / Tensor 并行、多 GPU KV Cache 管理。 |

---

## 技术亮点

| 类别 | 技术 |
|------|------|
| 语言 | Python 3.10+，C++ / CUDA 扩展 |
| 框架 | PyTorch 2.x，transformers |
| 运行环境 | Linux（Ubuntu 22.04+）、Windows / WSL2 |
| 模型 | Diffusion LLM（Fast-dLLM-v2、SimpleDiffusionLLM） |
| KV 存储 | 字节级打包混合精度缓存（FP16 / INT8 / INT4） |
| 优化 | 键值缓存压缩、量化、稀疏计算、通道剪枝 |
| CUDA | 自定义逐通道 INT8 / INT4 算子、稀疏 GEMM、融合注意力（脚手架） |
| 算子调优 | 持久化 shape-aware 自动调优缓存 + CPU / CUDA 统一分发 |
| Backend 插件 | `ModelBackend` entry-point 注册表（含 vLLM / SGLang scaffolding） |
| Agent 插件 | `BaseAgent` entry-point 注册表，支持自定义 Agent 热插拔 |
| 异步执行 | 线程池任务执行器，支持超时、取消、结果查询 |
| 不确定性感知早停 | 滑动窗口输出相似度检测，动态终止 denoising |
| 策略 DSL | 声明式 YAML / JSON 规则控制优化阈值与仲裁 |
| 可观测性 | Prometheus metrics、OpenTelemetry traces、Grafana / Streamlit dashboard |
| 工程化 | 热配置重载、熔断、优雅降级、类型安全、限流 |
| 分布式 | 流水线 / 张量并行脚手架 |
| 回归测试 | 基于 JSONL 历史的性能回归框架 |

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

当前主分支：`543 passed`，测试覆盖率约 `80%`。

### 30 秒跑通完整流水线

```python
from shadowinfer.orchestrator import Orchestrator

orch = Orchestrator(config_path="configs/optimize_full.yaml")
orch.initialize(model_config={"name": "SimpleDiffusionLLM"})

result = orch.run_full_pipeline(
    prompt="Diffusion LLM edge inference optimization.",
    num_steps=20,
)

print(f"Baseline latency: {result['baseline']['latency_e2e_ms']:.1f} ms")
print(f"Optimized latency: {result['optimized']['latency_e2e_ms']:.1f} ms")
print(f"Speedup: {result['speedup']:.2f}x")
print(f"Accuracy drop: {result['accuracy_drop']:.4f}")
```

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
用户输入 / 配置
        │
        ▼
┌─────────────────────┐
│     Orchestrator    │
│  (编排器 / 协调器)   │
├─────────┬───────────┤
│         │           │
▼         ▼           ▼
Profiler  ShadowKV   Q-drift   FFN Optimizer
│         │           │         │
└─────────┴───────────┴─────────┘
            │
            ▼
      Profiling Bus
            │
            ▼
   PyTorch / CUDA / 自定义算子
            │
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

1. **Importance Scoring**
   ```python
   # 基于 attention entropy 计算 token-head 重要性
   weights = F.softmax(attn_weights, dim=-1)
   entropy = -(weights * torch.log(weights + 1e-10)).sum(dim=-1).mean()
   score = (entropy / math.log(seq_len)) * (1 + 0.1 * layer_id / num_layers)
   ```

2. **Precision Allocation**

   | 重要性分数 | 精度 | 存储格式 |
   |-----------|------|---------|
   | ≥ 0.8 | FP32 | 全精度 |
   | 0.5 - 0.8 | FP16 | 半精度 |
   | 0.2 - 0.5 | INT8 | 8-bit 量化 |
   | < 0.2 | INT4 | 4-bit 量化 / 复用 |

3. **复用决策**
   ```python
   delta = torch.norm(kv_current - kv_previous) / torch.norm(kv_current)
   threshold = base_threshold * (1 - 0.5 * step_id / total_steps) * (1 - drift_score * 0.3)
   should_reuse = delta < threshold
   ```

### Q-drift 调度矩阵

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

Orchestrator 在每次 step 前后自动调用 `check_before_inference()` 和 `report_after_inference()`，实现熔断、限流、降级、健康监控和性能预算跟踪。

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

完整基准套件：

```bash
python benchmarks/run_benchmarks.py --model Fast-dLLM-v2-7B --suite full
```

输出包括：

- `report.md` / `report.html`：端到端对比报告；
- `latency.csv` / `roofline.png`：延迟曲线与屋顶线分析；
- 回归历史 `benchmarks/results/regression_history.jsonl`。

> 注：当前主机无可用 CUDA，内核使用 CPU fallback；真实 GPU 环境可启用 CUDA kernel 获得更优性能。


---

## 设计决策

以下是项目关键架构决策的简要摘要，完整讨论已合并到本文档相关章节。

| ADR | 决策 | 理由 |
|-----|------|------|
| ADR-001 | 使用 ShadowKV 而非统一量化 | 关键 token 不可控损失 vs 动态 per-token-head 精度分配，压缩比 50-70% 且精度损失 < 1%。 |
| ADR-002 | Q-drift 步感知调度 | Diffusion LLM 不同 step 误差敏感度不同，早期可激进、后期需保守。 |
| ADR-003 | 基于 Entropy 的重要性评分 | Attention 权重分布 entropy 能反映 token-head 的信息覆盖范围，无需额外训练。 |
| ADR-004 | 混合精度优于静态量化 | 静态量化固定压缩比、固定误差；混合精度按重要性自适应，可解释且与调度协同。 |
| ADR-005 | 多 Agent 架构 | 将 Profiler / ShadowKV / Q-drift / FFN 解耦，通过 Profiling Bus 通信，便于独立迭代和扩展。 |

---

## 项目结构

```
ShadowInfer/
├── .github/                        # GitHub 模板与工作流
│   ├── workflows/
│   │   └── ci.yml                  # CI/CD 流水线
│   ├── ISSUE_TEMPLATE/             # Issue 模板
│   └── PULL_REQUEST_TEMPLATE.md
├── LICENSE                         # MIT 许可证
├── README.md                       # 本文档（唯一保留的文档）
├── ROADMAP.md                      # 项目路线图
├── pyproject.toml                  # 包元数据与工具配置
├── requirements.txt                # 依赖列表
├── requirements-lock.txt           # 锁定依赖
├── scripts/
│   └── run.py                      # CLI 入口脚本
├── configs/                        # 配置示例
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
│   ├── profiler/                   # Profiling Agent
│   ├── shadowkv/                   # KV cache 优化
│   │   ├── shadowkv_agent.py
│   │   ├── kv_cache_manager.py
│   │   ├── packed_kv_cache.py
│   │   ├── decision_plane.py
│   │   ├── importance_model.py
│   │   └── eviction_policy.py
│   ├── qdrift/                     # 步感知调度
│   ├── ffn_optimizer/              # FFN 优化
│   │   ├── ffn_optimizer_agent.py
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
├── tests/                          # 单元与集成测试
└── outputs/                        # 生成的报告（gitignore）
```

---

## 开发贡献

感谢你对 ShadowInfer 的兴趣！欢迎通过以下方式参与：

- 在 [GitHub Issues](https://github.com/ShaneLiu04/ShadowInfer/issues) 报告 bug 或提出需求；
- 在 [Pull Requests](https://github.com/ShaneLiu04/ShadowInfer/pulls) 提交改进；
- 改进文档、教程或翻译；
- 分享你的使用场景与复现结果。

### 开发环境

```bash
git clone https://github.com/ShaneLiu04/ShadowInfer.git
cd ShadowInfer
python -m venv .venv
source .venv/bin/activate  # Linux/Mac
pip install -e ".[dev]"
```

### 代码规范

提交 PR 前请运行：

```bash
black --check shadowinfer/ tests/ benchmarks/ exporter/
isort --check-only shadowinfer/ tests/ benchmarks/ exporter/
flake8 shadowinfer/ tests/ benchmarks/ exporter/
mypy shadowinfer/ --ignore-missing-imports
pytest tests/
```

### PR 流程

1. Fork 仓库并创建功能分支；
2. 保持提交清晰、聚焦；
3. 为新功能添加或更新测试；
4. 更新相关文档（本文档 `README.md`）；
5. 确保所有检查通过；
6. 提交 Pull Request 并清晰描述变更。

---

## 版本历史

### v3.2.2（2026-07）

- 新增 `AgentPluginRegistry`，支持通过 `shadowinfer.agents` entry-point 注册第三方 Agent；
- Orchestrator 自动调度插件 Agent，输出写入 `step.outputs` 并随 STEP_RESULT 广播；
- 新增 `AsyncTaskExecutor`，支持后台任务提交、超时、取消与结果查询；
- 新增 `UncertaintyAwareEarlyStopper` 并接入 Orchestrator step 循环；
- 修复 HuggingFace 后端 `local_files_only=True` 离线 fallback；
- 全量测试达 `581 passed, 1 skipped`，覆盖率 `80.41%`。

### v3.2.1（2026-07）

- 新增 `ImportanceModel`、`KVDecisionPlane`、`EvictionPolicy`，构建 ShadowKV 决策平面；
- 支持 KV Cache 驱逐与预取；
- 新增 FFN 动态通道剪枝；
- 迁移日志到 `structlog`，支持 JSON、轮转、动态级别；
- 修复 HuggingFace 后端 `DynamicCache` 解包问题；
- 合并并精简所有文档到 `README.md`。
- 新增 `BackendRegistry` 与 entry-point 插件发现；

- 新增 vLLM / SGLang 后端 scaffolding；
- 新增声明式 Policy DSL；
- 策略引擎接入 Orchestrator 仲裁。

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
