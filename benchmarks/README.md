# ShadowInfer Benchmark Suite

## 概述

ShadowInfer Benchmark Suite 提供了一套完整的 AI 推理性能基准测试工具，
涵盖延迟、吞吐量、内存、精度、可扩展性和 Roofline 模型分析六大维度。

对应文档：`plan-v2.md` Phase 2

---

## 运行方法

### 完整套件

```bash
python benchmarks/run_benchmarks.py --model Fast-dLLM-v2-7B --suite full
```

### 单独测试

```bash
# 延迟测试（P50/P95/P99）
python benchmarks/run_benchmarks.py --model Fast-dLLM-v2-7B --suite latency

# 吞吐量测试（不同 batch size）
python benchmarks/run_benchmarks.py --model Fast-dLLM-v2-7B --suite throughput

# 内存测试（峰值显存、碎片率）
python benchmarks/run_benchmarks.py --model Fast-dLLM-v2-7B --suite memory

# 可扩展性测试（不同 seq_len）
python benchmarks/run_benchmarks.py --model Fast-dLLM-v2-7B --suite scalability

# Roofline 分析
python benchmarks/run_benchmarks.py --model Fast-dLLM-v2-7B --suite roofline
```

### 高级参数

```bash
python benchmarks/run_benchmarks.py \
    --model Fast-dLLM-v2-7B \
    --suite full \
    --num-steps 100 \
    --warmup-steps 10 \
    --batch-sizes 1 2 4 8 16 \
    --seq-lengths 128 512 2048 4096 8192 \
    --output benchmarks/results/v2-7B \
    --gpu 0
```

---

## 结果解读

### 1. Latency Benchmark

- **latency_ms**: 单 step 延迟
- **P50 / P95 / P99**: 百分位延迟，反映尾延迟表现
- **std**: 延迟抖动程度

**目标**: P50 < 20ms, P99 < 50ms（7B 模型单 token）

### 2. Throughput Benchmark

- **throughput_tokens_per_sec**: tokens / 秒
- **batch_size**: 测试批次大小

**目标**: 随 batch size 线性扩展，直到内存或计算饱和

### 3. Memory Benchmark

- **peak_allocated_mb**: 峰值分配显存
- **peak_reserved_mb**: 峰值预留显存
- **fragmentation_ratio**: 碎片率 = (reserved - allocated) / reserved
- **utilization_ratio**: 利用率 = allocated / total

**目标**: fragmentation < 15%, utilization > 80%

### 4. Scalability Benchmark

- **seq_len**: 序列长度
- **latency_mean_ms**: 平均延迟
- **latency_p95_ms**: P95 延迟
- **peak_memory_mb**: 峰值内存

**目标**: 延迟随 seq_len 亚线性增长（得益于 ShadowKV 压缩）

### 5. Roofline 分析

Roofline 模型是理解性能瓶颈的经典方法：

- **横轴**: Operational Intensity (OI) = FLOPs / Bytes
- **纵轴**: Performance (GFLOPs/s)
- **Roofline 曲线**: min(峰值算力, 峰值带宽 × OI)
- **Ridge Point**: 拐点 = 峰值算力 / 峰值带宽

#### 瓶颈判定

| OI 位置 | 瓶颈类型 | 优化方向 |
|---------|----------|----------|
| OI < Ridge Point | Memory Bound | 提升数据复用、分块、量化压缩 |
| OI > Ridge Point | Compute Bound | 向量化、Kernel 融合、Tensor Core |
| OI ≈ Ridge Point | Balanced | 检查调度开销、线程占用率 |

#### 示例输出

```
Roofline Points:
  Attention: OI=8.00, Perf=25.00 GFLOPs/s, Efficiency=0.01%, Bottleneck=memory
  FFN: OI=32.00, Perf=62.50 GFLOPs/s, Efficiency=0.02%, Bottleneck=compute
  ShadowKV_Eviction: OI=2.00, Perf=16.67 GFLOPs/s, Efficiency=0.01%, Bottleneck=memory
  LayerNorm: OI=4.00, Perf=13.33 GFLOPs/s, Efficiency=0.00%, Bottleneck=memory
```

---

## 输出目录结构

```
benchmarks/results/
├── latency/
│   ├── measurements.csv      # 原始测量数据
│   ├── result.json           # 完整结果（含统计）
│   └── plots/
│       ├── timeseries.png
│       └── latency_distribution.png
├── throughput/
│   ├── measurements.csv
│   ├── result.json
│   └── plots/
├── memory/
│   ├── measurements.csv
│   ├── result.json
│   └── plots/
├── scalability/
│   ├── measurements.csv
│   ├── result.json
│   └── plots/
├── roofline/
│   ├── report.md             # 优化建议报告
│   ├── points.json           # Roofline 点数据
│   ├── config.json           # 分析器配置
│   └── roofline.png          # Roofline 图表
├── report.md                 # 汇总 Markdown 报告
└── report.html               # 汇总 HTML 报告
```

---

## 核心 API 使用

### 直接使用 BenchmarkRunner

```python
from shadowinfer.benchmarking import BenchmarkConfig, BenchmarkRunner

config = BenchmarkConfig(
    name="my_test",
    num_warmup_steps=5,
    num_measurement_steps=20,
    batch_sizes=[1, 2, 4, 8],
    seq_lengths=[128, 512, 2048],
)

runner = BenchmarkRunner(config)

# 延迟测试
result = runner.run_latency_benchmark(my_inference_fn, batch_size=1, seq_len=512)
print(result.get_summary())
result.export_json("results.json")

# 吞吐量测试
result = runner.run_throughput_benchmark(my_inference_fn, seq_len=512)
result.generate_plots("plots/")

# 内存测试
result = runner.run_memory_benchmark(my_inference_fn, batch_size=1, seq_len=512)

# 完整套件
results = runner.run_full_suite(my_inference_fn, batch_size=1, seq_len=512)
```

### 使用 RooflineAnalyzer

```python
from shadowinfer.benchmarking import RooflineAnalyzer

# A100-like specs: 312 TFLOPS FP16, 2039 GB/s
analyzer = RooflineAnalyzer(
    peak_compute_gflops=312_000.0,
    peak_memory_bandwidth_gbps=2_039.0,
)

# Analyze a single operation
point = analyzer.analyze_operation(
    name="Attention",
    flops=1.5e9,
    bytes_accessed=200e6,
    execution_time_ms=12.0,
)

print(f"OI={point.operational_intensity:.2f}, "
      f"Efficiency={point.efficiency()*100:.1f}%, "
      f"Bottleneck={point.bottleneck}")

# Generate report
report = analyzer.generate_optimization_report([point])
print(report)

# Generate plot (base64 PNG)
img_base64 = analyzer.generate_roofline_plot([point])
```

### 生成报告

```python
from shadowinfer.benchmarking import BenchmarkReport

report = BenchmarkReport(results)
report.generate("output_dir/")
# Generates: report.md, report.html, CSVs, JSONs, and plots
```

---

## 硬件配置参考

| GPU | Peak Compute (FP16) | Memory Bandwidth | Ridge Point |
|-----|---------------------|------------------|-------------|
| A100 | 312 TFLOPs/s | 2,039 GB/s | 153 |
| H100 | 989 TFLOPs/s | 3,350 GB/s | 295 |
| RTX 4090 | 82.6 TFLOPs/s | 1,008 GB/s | 82 |
| V100 | 125 TFLOPs/s | 900 GB/s | 139 |

---

## 扩展与定制

### 自定义 Benchmark 配置

```python
config = BenchmarkConfig(
    name="custom_benchmark",
    num_warmup_steps=10,
    num_measurement_steps=50,
    batch_sizes=[1, 4, 16, 32],
    seq_lengths=[256, 1024, 4096, 16384],
    model_names=["MyCustomModel"],
)
```

### 集成真实推理函数

```python
import torch

def real_inference_fn(batch_size=1, seq_len=512):
    input_ids = torch.randint(0, vocab_size, (batch_size, seq_len)).cuda()
    with torch.no_grad():
        outputs = model(input_ids)
    return {"num_tokens": batch_size * seq_len}

runner = BenchmarkRunner(config)
result = runner.run_latency_benchmark(real_inference_fn, batch_size=1, seq_len=512)
```

---

## 版本信息

- **Benchmark Suite Version**: 2.0
- **Requires**: Python 3.10+, PyTorch (optional), matplotlib (optional)
- **Corresponds to**: `plan-v2.md` Phase 2
