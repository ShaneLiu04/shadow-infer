#!/usr/bin/env python3
"""基准测试入口脚本。

用法：
    python benchmarks/run_benchmarks.py --model Fast-dLLM-v2-7B --suite full
    python benchmarks/run_benchmarks.py --model Fast-dLLM-v2-1.5B --suite latency
    python benchmarks/run_benchmarks.py --model Fast-dLLM-v2-7B --suite roofline

对应文档：plan-v2.md Phase 2

Version: 3.0
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict

# 添加项目根目录到 Python 路径
sys.path.insert(0, str(Path(__file__).parent.parent))

from shadowinfer.benchmarking import (  # noqa: E402
    BenchmarkConfig,
    BenchmarkReport,
    BenchmarkRunner,
    RooflineAnalyzer,
)

# Try to import torch for real inference; fall back to mock if unavailable
try:
    import torch

    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False


def _create_mock_inference_fn(model_name: str, device: str = "cpu") -> Any:
    """创建模拟推理函数，用于无 GPU 环境的基准测试演示。

    Args:
        model_name: 模型名称，用于决定延迟分布。
        device: 模拟设备。

    Returns:
        可调用推理函数。
    """
    import random

    base_latency = 15.0 if "7B" in model_name else 5.0

    def inference_fn(**kwargs: Any) -> Dict[str, Any]:
        batch_size = kwargs.get("batch_size", 1)
        seq_len = kwargs.get("seq_len", 128)
        # Simulate computation time roughly proportional to batch * seq_len
        delay = base_latency * (1 + 0.1 * (batch_size - 1)) * (seq_len / 128.0) ** 1.5
        delay *= random.uniform(0.9, 1.1)
        time.sleep(delay / 1000.0)
        return {
            "num_tokens": batch_size * seq_len,
            "latency_ms": delay,
        }

    return inference_fn


def _create_mock_roofline_inference_fn(model_name: str) -> Any:
    """创建模拟 Roofline 推理函数，返回各操作的 FLOPs 和内存访问。

    Args:
        model_name: 模型名称。

    Returns:
        返回包含 operations 列表的推理函数。
    """
    import random

    # Approximate FLOPs and memory for a transformer layer
    base_flops = 2e9 if "7B" in model_name else 5e8

    def inference_fn(**kwargs: Any) -> Dict[str, Any]:
        ops = [
            {
                "name": "Attention",
                "flops": base_flops * 0.3,
                "bytes_accessed": base_flops * 0.3 / 8.0,  # Low OI, memory bound
                "time_ms": 12.0 + random.uniform(-1.0, 1.0),
            },
            {
                "name": "FFN",
                "flops": base_flops * 0.5,
                "bytes_accessed": base_flops * 0.5 / 32.0,  # Higher OI, compute bound
                "time_ms": 8.0 + random.uniform(-0.5, 0.5),
            },
            {
                "name": "ShadowKV_Eviction",
                "flops": base_flops * 0.05,
                "bytes_accessed": base_flops * 0.05 / 2.0,  # Very low OI
                "time_ms": 3.0 + random.uniform(-0.2, 0.2),
            },
            {
                "name": "LayerNorm",
                "flops": base_flops * 0.02,
                "bytes_accessed": base_flops * 0.02 / 4.0,
                "time_ms": 1.5 + random.uniform(-0.1, 0.1),
            },
        ]
        return {"operations": ops}

    return inference_fn


def _setup_gpu(gpu_id: int) -> None:
    """设置 GPU 设备。

    Args:
        gpu_id: GPU 设备 ID。
    """
    if HAS_TORCH and torch.cuda.is_available():
        torch.cuda.set_device(gpu_id)
        torch.cuda.empty_cache()
        print(f"Using GPU: {torch.cuda.get_device_name(gpu_id)}")
    else:
        print("GPU not available, using CPU / mock mode.")


def _print_banner() -> None:
    """打印基准测试启动横幅。"""
    print("=" * 60)
    print("  ShadowInfer Benchmark Suite v2.0")
    print("  Performance Benchmarking & Roofline Analysis")
    print("=" * 60)


def run_latency_suite(model_name: str, output_dir: str, config: BenchmarkConfig) -> None:
    """运行延迟基准测试。

    Args:
        model_name: 模型名称。
        output_dir: 输出目录。
        config: 测试配置。
    """
    print("\n[Latency Benchmark]")
    inference_fn = _create_mock_inference_fn(model_name)
    runner = BenchmarkRunner(config)
    result = runner.run_latency_benchmark(inference_fn, batch_size=1, seq_len=512)

    print(result.get_summary())
    result.export_json(f"{output_dir}/latency/result.json")
    result.export_csv(f"{output_dir}/latency/measurements.csv")
    result.generate_plots(f"{output_dir}/latency/plots")


def run_throughput_suite(model_name: str, output_dir: str, config: BenchmarkConfig) -> None:
    """运行吞吐量基准测试。

    Args:
        model_name: 模型名称。
        output_dir: 输出目录。
        config: 测试配置。
    """
    print("\n[Throughput Benchmark]")
    inference_fn = _create_mock_inference_fn(model_name)
    runner = BenchmarkRunner(config)
    result = runner.run_throughput_benchmark(inference_fn, seq_len=512)

    print(result.get_summary())
    result.export_json(f"{output_dir}/throughput/result.json")
    result.export_csv(f"{output_dir}/throughput/measurements.csv")
    result.generate_plots(f"{output_dir}/throughput/plots")


def run_memory_suite(model_name: str, output_dir: str, config: BenchmarkConfig) -> None:
    """运行内存基准测试。

    Args:
        model_name: 模型名称。
        output_dir: 输出目录。
        config: 测试配置。
    """
    print("\n[Memory Benchmark]")
    inference_fn = _create_mock_inference_fn(model_name)
    runner = BenchmarkRunner(config)
    result = runner.run_memory_benchmark(inference_fn, batch_size=1, seq_len=512)

    print(result.get_summary())
    result.export_json(f"{output_dir}/memory/result.json")
    result.export_csv(f"{output_dir}/memory/measurements.csv")
    result.generate_plots(f"{output_dir}/memory/plots")


def run_scalability_suite(model_name: str, output_dir: str, config: BenchmarkConfig) -> None:
    """运行可扩展性基准测试。

    Args:
        model_name: 模型名称。
        output_dir: 输出目录。
        config: 测试配置。
    """
    print("\n[Scalability Benchmark]")
    inference_fn = _create_mock_inference_fn(model_name)
    runner = BenchmarkRunner(config)
    result = runner.run_scalability_benchmark(inference_fn, batch_size=1)

    print(result.get_summary())
    result.export_json(f"{output_dir}/scalability/result.json")
    result.export_csv(f"{output_dir}/scalability/measurements.csv")
    result.generate_plots(f"{output_dir}/scalability/plots")


def run_roofline_suite(model_name: str, output_dir: str, config: BenchmarkConfig) -> None:
    """运行 Roofline 分析。

    Args:
        model_name: 模型名称。
        output_dir: 输出目录。
        config: 测试配置。
    """
    print("\n[Roofline Analysis]")
    inference_fn = _create_mock_roofline_inference_fn(model_name)

    # Default analyzer: A100-like specs (312 TFLOPS FP16, 2039 GB/s)
    analyzer = RooflineAnalyzer(
        peak_compute_gflops=312000.0,
        peak_memory_bandwidth_gbps=2039.0,
    )

    runner = BenchmarkRunner(config)
    roofline_result = runner.run_roofline_benchmark(inference_fn, analyzer)

    print(f"Roofline Points: {len(roofline_result['points'])}")
    for point in roofline_result["points"]:
        print(
            f"  {point['name']}: OI={point['operational_intensity']:.2f}, "
            f"Perf={point['performance']:.2f} GFLOPs/s, "
            f"Efficiency={point['efficiency']*100:.1f}%, "
            f"Bottleneck={point['bottleneck']}"
        )

    # Save report
    roofline_dir = Path(output_dir) / "roofline"
    roofline_dir.mkdir(parents=True, exist_ok=True)
    (roofline_dir / "report.md").write_text(roofline_result["report"], encoding="utf-8")
    (roofline_dir / "points.json").write_text(
        json.dumps(roofline_result["points"], indent=2), encoding="utf-8"
    )
    (roofline_dir / "config.json").write_text(
        json.dumps(roofline_result["analyzer_config"], indent=2), encoding="utf-8"
    )

    # Save plot if available
    if roofline_result["plot_base64"]:
        import base64

        plot_path = roofline_dir / "roofline.png"
        plot_path.write_bytes(base64.b64decode(roofline_result["plot_base64"]))
        print(f"Roofline plot saved to: {plot_path}")

    print("\nOptimization Report Preview:")
    print("-" * 40)
    print(roofline_result["report"][:800] + "...")


def run_full_suite(model_name: str, output_dir: str, config: BenchmarkConfig) -> None:
    """运行完整测试套件。

    Args:
        model_name: 模型名称。
        output_dir: 输出目录。
        config: 测试配置。
    """
    print("\n[Full Benchmark Suite]")
    inference_fn = _create_mock_inference_fn(model_name)
    runner = BenchmarkRunner(config)
    results = runner.run_full_suite(inference_fn, batch_size=1, seq_len=512)

    # Add roofline
    roofline_fn = _create_mock_roofline_inference_fn(model_name)
    analyzer = RooflineAnalyzer(
        peak_compute_gflops=312000.0,
        peak_memory_bandwidth_gbps=2039.0,
    )
    roofline_result = runner.run_roofline_benchmark(roofline_fn, analyzer)

    # Generate consolidated report
    report = BenchmarkReport(results)
    report.generate(output_dir)

    # Save roofline separately
    roofline_dir = Path(output_dir) / "roofline"
    roofline_dir.mkdir(parents=True, exist_ok=True)
    (roofline_dir / "report.md").write_text(roofline_result["report"], encoding="utf-8")
    (roofline_dir / "points.json").write_text(
        json.dumps(roofline_result["points"], indent=2), encoding="utf-8"
    )
    if roofline_result["plot_base64"]:
        import base64

        (roofline_dir / "roofline.png").write_bytes(
            base64.b64decode(roofline_result["plot_base64"])
        )

    print("\n" + "=" * 60)
    print("Full suite completed. Results saved to:")
    for name in results.keys():
        print(f"  - {output_dir}/{name}/")
    print(f"  - {output_dir}/roofline/")
    print(f"  - {output_dir}/report.md")
    print(f"  - {output_dir}/report.html")
    print("=" * 60)


def main() -> None:
    """主入口函数。"""
    parser = argparse.ArgumentParser(
        description="ShadowInfer Benchmark Suite — Performance Benchmarking & Roofline Analysis",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s --model Fast-dLLM-v2-7B --suite full
  %(prog)s --model Fast-dLLM-v2-1.5B --suite latency --output results/1.5B
  %(prog)s --model Fast-dLLM-v2-7B --suite roofline --num-steps 30
        """,
    )
    parser.add_argument("--model", default="Fast-dLLM-v2-7B", help="模型名称")
    parser.add_argument(
        "--suite",
        default="full",
        choices=["full", "latency", "throughput", "memory", "accuracy", "scalability", "roofline"],
        help="测试套件类型",
    )
    parser.add_argument("--output", default="benchmarks/results", help="输出目录")
    parser.add_argument("--num-steps", type=int, default=50, help="测量步数（denoising steps）")
    parser.add_argument("--warmup-steps", type=int, default=5, help="warmup 步数")
    parser.add_argument("--gpu", type=int, default=0, help="GPU ID")
    parser.add_argument(
        "--batch-sizes", type=int, nargs="+", default=[1, 2, 4, 8], help="测试 batch sizes"
    )
    parser.add_argument(
        "--seq-lengths", type=int, nargs="+", default=[128, 512, 2048, 4096], help="测试序列长度"
    )

    args = parser.parse_args()
    _print_banner()
    _setup_gpu(args.gpu)

    config = BenchmarkConfig(
        name=f"{args.model}_benchmark",
        num_warmup_steps=args.warmup_steps,
        num_measurement_steps=args.num_steps,
        batch_sizes=args.batch_sizes,
        seq_lengths=args.seq_lengths,
        model_names=[args.model],
    )

    output_dir = args.output
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    start_time = time.time()

    if args.suite == "full":
        run_full_suite(args.model, output_dir, config)
    elif args.suite == "latency":
        run_latency_suite(args.model, output_dir, config)
    elif args.suite == "throughput":
        run_throughput_suite(args.model, output_dir, config)
    elif args.suite == "memory":
        run_memory_suite(args.model, output_dir, config)
    elif args.suite == "scalability":
        run_scalability_suite(args.model, output_dir, config)
    elif args.suite == "roofline":
        run_roofline_suite(args.model, output_dir, config)
    elif args.suite == "accuracy":
        print("\n[Accuracy Benchmark]")
        print("Note: Accuracy benchmark requires a baseline model reference.")
        print("Please implement baseline_fn and run programmatically.")

    elapsed = time.time() - start_time
    print(f"\nTotal time: {elapsed:.2f}s")
    print(f"Benchmark results saved to {output_dir}")


if __name__ == "__main__":
    main()
