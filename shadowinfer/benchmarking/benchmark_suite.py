"""标准化基准测试套件。

对应大厂实践：MLPerf, DeepBench 标准化测试
对应文档：plan-v2.md Phase 2.1

测试场景：
1. Latency Benchmark: 单 step 延迟（P50/P95/P99）
2. Throughput Benchmark: tokens/sec（不同 batch size）
3. Memory Benchmark: 峰值显存、内存碎片率
4. Accuracy Benchmark: perplexity, BLEU, 生成质量评分
5. Scalability Benchmark: 不同 seq_len（128/512/2048/4096）
6. Roofline Analysis: 计算密集型 vs 内存密集型分析

Version: 3.0
"""

from __future__ import annotations

import csv
import json
import os
import platform
import statistics
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from shadowinfer.benchmarking.roofline import RooflineAnalyzer, RooflinePoint
from shadowinfer.utils.metrics import Metrics

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False

try:
    import torch

    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

__version__ = "3.0"


@dataclass
class BenchmarkConfig:
    """基准测试配置。

    定义测试的名称、步数、batch sizes、seq lengths 和模型列表。

    Attributes:
        name: 测试名称。
        num_warmup_steps: warmup 步数，用于稳定 GPU 状态。
        num_measurement_steps: 正式测量步数。
        batch_sizes: 测试的 batch size 列表。
        seq_lengths: 测试的序列长度列表。
        model_names: 测试的模型名称列表。
    """

    name: str = "shadowinfer_benchmark"
    num_warmup_steps: int = 5
    num_measurement_steps: int = 20
    batch_sizes: List[int] = field(default_factory=lambda: [1, 2, 4, 8])
    seq_lengths: List[int] = field(default_factory=lambda: [128, 512, 2048, 4096])
    model_names: List[str] = field(default_factory=lambda: ["Fast-dLLM-v2-1.5B", "Fast-dLLM-v2-7B"])

    def to_dict(self) -> Dict[str, Any]:
        """将配置序列化为字典。"""
        return asdict(self)


@dataclass
class BenchmarkResult:
    """单轮基准测试结果。

    保存每次测量的原始数据和统计摘要，支持多种导出格式。

    Attributes:
        config: 本次测试的 BenchmarkConfig。
        measurements: 每次测量的原始数据列表，每个元素是指标字典。
        statistics: 统计摘要（mean, std, p50, p95, p99, min, max）。
        environment: 环境信息（CPU, GPU, CUDA, PyTorch 版本等）。
    """

    config: BenchmarkConfig
    measurements: List[Dict[str, float]] = field(default_factory=list)
    statistics: Dict[str, Dict[str, float]] = field(default_factory=dict)
    environment: Dict[str, str] = field(default_factory=dict)

    def get_summary(self) -> str:
        """生成文本摘要。

        Returns:
            Markdown 格式的统计摘要文本。
        """
        lines: List[str] = []
        lines.append(f"# Benchmark Result: {self.config.name}\n\n")

        # Environment info
        lines.append("## Environment\n\n")
        for key, value in self.environment.items():
            lines.append(f"- **{key}**: {value}\n")
        lines.append("\n")

        # Statistics table
        lines.append("## Statistics Summary\n\n")
        if self.statistics:
            metrics = list(next(iter(self.statistics.values())).keys())
            header = "| Metric | " + " | ".join(metrics) + " |\n"
            separator = "|--------|" + "|".join(["--------"] * len(metrics)) + "|\n"
            lines.append(header)
            lines.append(separator)
            for metric_name, stats in self.statistics.items():
                row = f"| {metric_name} | "
                row += " | ".join([f"{stats.get(m, 0.0):.4f}" for m in metrics])
                row += " |\n"
                lines.append(row)
        else:
            lines.append("No statistics available.\n")

        lines.append("\n")
        lines.append(f"**Total measurements:** {len(self.measurements)}\n")
        return "".join(lines)

    def export_csv(self, filepath: str) -> None:
        """导出原始测量数据为 CSV。

        Args:
            filepath: 输出 CSV 文件路径。
        """
        if not self.measurements:
            return

        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        fieldnames = list(self.measurements[0].keys())

        with open(filepath, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(self.measurements)

    def export_json(self, filepath: str) -> None:
        """导出结果为 JSON。

        Args:
            filepath: 输出 JSON 文件路径。
        """
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "config": self.config.to_dict(),
            "environment": self.environment,
            "statistics": self.statistics,
            "measurements": self.measurements,
        }
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, ensure_ascii=False)

    def generate_plots(self, output_dir: str) -> None:
        """生成可视化图表（matplotlib）并保存到目录。

        若 matplotlib 不可用，则跳过。

        Args:
            output_dir: 输出图表目录。
        """
        if not HAS_MATPLOTLIB or not self.measurements:
            return

        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        # Extract numeric keys
        numeric_keys = []
        for key in self.measurements[0].keys():
            if all(isinstance(m.get(key), (int, float)) for m in self.measurements):
                numeric_keys.append(key)

        if not numeric_keys:
            return

        # Time-series plot for all numeric metrics
        fig, axes = plt.subplots(
            nrows=(len(numeric_keys) + 1) // 2,
            ncols=2,
            figsize=(14, 4 * ((len(numeric_keys) + 1) // 2)),
        )
        if len(numeric_keys) == 1:
            axes = [axes]
        axes = axes.flatten() if hasattr(axes, "flatten") else [axes]

        for idx, key in enumerate(numeric_keys):
            ax = axes[idx]
            values = [m[key] for m in self.measurements]
            ax.plot(values, marker="o", markersize=3, linewidth=1)
            ax.set_title(f"{key} over time", fontsize=11)
            ax.set_xlabel("Step")
            ax.set_ylabel(key)
            ax.grid(True, linestyle="--", alpha=0.5)

        # Hide unused subplots
        for idx in range(len(numeric_keys), len(axes)):
            axes[idx].axis("off")

        plt.tight_layout()
        fig.savefig(output_path / "timeseries.png", dpi=150, bbox_inches="tight")
        plt.close(fig)

        # Histogram for latency-like metrics
        latency_keys = [k for k in numeric_keys if "latency" in k.lower() or "time" in k.lower()]
        if latency_keys:
            fig, axes = plt.subplots(
                nrows=(len(latency_keys) + 1) // 2,
                ncols=2,
                figsize=(14, 4 * ((len(latency_keys) + 1) // 2)),
            )
            if len(latency_keys) == 1:
                axes = [axes]
            axes = axes.flatten() if hasattr(axes, "flatten") else [axes]

            for idx, key in enumerate(latency_keys):
                ax = axes[idx]
                values = [m[key] for m in self.measurements]
                ax.hist(values, bins=min(20, len(values)), edgecolor="black", alpha=0.7)
                ax.axvline(
                    statistics.mean(values),
                    color="red",
                    linestyle="--",
                    label=f"Mean: {statistics.mean(values):.4f}",
                )
                ax.axvline(
                    statistics.median(values),
                    color="green",
                    linestyle="--",
                    label=f"Median: {statistics.median(values):.4f}",
                )
                ax.set_title(f"{key} Distribution", fontsize=11)
                ax.set_xlabel(key)
                ax.set_ylabel("Frequency")
                ax.legend()
                ax.grid(True, linestyle="--", alpha=0.5)

            for idx in range(len(latency_keys), len(axes)):
                axes[idx].axis("off")

            plt.tight_layout()
            fig.savefig(output_path / "latency_distribution.png", dpi=150, bbox_inches="tight")
            plt.close(fig)


def _collect_environment() -> Dict[str, str]:
    """收集运行环境信息。

    Returns:
        包含 CPU, GPU, CUDA, PyTorch 等版本信息的字典。
    """
    env: Dict[str, str] = {
        "platform": platform.platform(),
        "python_version": platform.python_version(),
        "cpu_count": str(os.cpu_count() or 0),
    }

    if HAS_TORCH:
        env["pytorch_version"] = torch.__version__
        env["cuda_available"] = str(torch.cuda.is_available())
        if torch.cuda.is_available():
            env["cuda_version"] = torch.version.cuda or "unknown"
            env["gpu_count"] = str(torch.cuda.device_count())
            env["gpu_name"] = torch.cuda.get_device_name(0)
            env["gpu_capability"] = str(torch.cuda.get_device_capability(0))
        else:
            env["cuda_version"] = "N/A"
            env["gpu_count"] = "0"
            env["gpu_name"] = "N/A"
            env["gpu_capability"] = "N/A"
    else:
        env["pytorch_version"] = "N/A"
        env["cuda_available"] = "False"

    return env


def _compute_statistics(measurements: List[Dict[str, float]]) -> Dict[str, Dict[str, float]]:
    """计算统计摘要。

    对每个数值指标计算 mean, std, p50, p95, p99, min, max。

    Args:
        measurements: 测量数据列表。

    Returns:
        嵌套字典，外层键为指标名，内层为统计量。
    """
    if not measurements:
        return {}

    numeric_keys = []
    for key in measurements[0].keys():
        if all(isinstance(m.get(key), (int, float)) for m in measurements):
            numeric_keys.append(key)

    stats: Dict[str, Dict[str, float]] = {}
    for key in numeric_keys:
        values = [m[key] for m in measurements]
        n = len(values)
        if n == 0:
            continue

        sorted_vals = sorted(values)
        mean = statistics.mean(values)
        std = statistics.stdev(values) if n > 1 else 0.0
        p50 = statistics.median(values)
        p95_idx = max(0, int(math.ceil(0.95 * n)) - 1)
        p99_idx = max(0, int(math.ceil(0.99 * n)) - 1)
        p95 = sorted_vals[p95_idx]
        p99 = sorted_vals[p99_idx]
        min_val = min(values)
        max_val = max(values)

        stats[key] = {
            "mean": float(mean),
            "std": float(std),
            "p50": float(p50),
            "p95": float(p95),
            "p99": float(p99),
            "min": float(min_val),
            "max": float(max_val),
        }

    return stats


import math  # noqa: E402, imported late for _compute_statistics


class BenchmarkRunner:
    """基准测试运行器。

    封装标准化的 benchmark 执行流程，包括 warmup、测量、
    统计计算和环境信息收集。

    Attributes:
        config: 当前测试配置。
    """

    def __init__(self, config: BenchmarkConfig) -> None:
        """初始化运行器。

        Args:
            config: 测试配置。
        """
        self.config = config

    def _warmup(self, inference_fn: Callable, **kwargs: Any) -> None:
        """执行 warmup 步骤。

        Args:
            inference_fn: 推理函数，接收 **kwargs。
            **kwargs: 传递给 inference_fn 的额外参数。
        """
        for _ in range(self.config.num_warmup_steps):
            inference_fn(**kwargs)
        if HAS_TORCH and torch.cuda.is_available():
            torch.cuda.synchronize()

    def _measure(
        self, inference_fn: Callable, metric_fn: Optional[Callable] = None, **kwargs: Any
    ) -> List[Dict[str, float]]:
        """执行测量步骤并收集数据。

        Args:
            inference_fn: 推理函数。
            metric_fn: 可选，每次测量后调用以获取额外指标。
            **kwargs: 传递给 inference_fn 的额外参数。

        Returns:
            测量数据列表。
        """
        measurements: List[Dict[str, float]] = []
        for _ in range(self.config.num_measurement_steps):
            start = time.perf_counter()
            result = inference_fn(**kwargs)
            if HAS_TORCH and torch.cuda.is_available():
                torch.cuda.synchronize()
            end = time.perf_counter()

            elapsed_ms = (end - start) * 1000.0
            record: Dict[str, float] = {"latency_ms": elapsed_ms}

            if metric_fn is not None:
                extra = metric_fn(result)
                if extra:
                    record.update(extra)

            measurements.append(record)
        return measurements

    def run_latency_benchmark(self, inference_fn: Callable, **kwargs: Any) -> BenchmarkResult:
        """运行延迟基准测试。

        测量单 step 推理延迟，输出 P50/P95/P99 统计。

        Args:
            inference_fn: 推理函数。
            **kwargs: 额外参数。

        Returns:
            BenchmarkResult 包含延迟统计。
        """
        self._warmup(inference_fn, **kwargs)
        measurements = self._measure(inference_fn, **kwargs)
        stats = _compute_statistics(measurements)
        env = _collect_environment()
        return BenchmarkResult(
            config=self.config,
            measurements=measurements,
            statistics=stats,
            environment=env,
        )

    def run_throughput_benchmark(self, inference_fn: Callable, **kwargs: Any) -> BenchmarkResult:
        """运行吞吐量基准测试。

        测试不同 batch size 下的 tokens/sec 吞吐量。

        Args:
            inference_fn: 推理函数，应返回 token 数量或包含 token 数的 dict。
            **kwargs: 额外参数，应包含可调 batch_size 的参数。

        Returns:
            BenchmarkResult 包含各 batch size 的吞吐量。
        """
        measurements: List[Dict[str, float]] = []
        for bs in self.config.batch_sizes:
            kwargs_with_bs = dict(kwargs)
            kwargs_with_bs["batch_size"] = bs
            self._warmup(inference_fn, **kwargs_with_bs)

            total_tokens = 0
            total_time_ms = 0.0
            for _ in range(self.config.num_measurement_steps):
                start = time.perf_counter()
                result = inference_fn(**kwargs_with_bs)
                if HAS_TORCH and torch.cuda.is_available():
                    torch.cuda.synchronize()
                end = time.perf_counter()

                elapsed_ms = (end - start) * 1000.0
                # Try to infer token count from result
                tokens = 0
                if isinstance(result, dict):
                    tokens = result.get("num_tokens", bs * kwargs_with_bs.get("seq_len", 128))
                elif isinstance(result, int):
                    tokens = result
                else:
                    tokens = bs * kwargs_with_bs.get("seq_len", 128)

                total_tokens += tokens
                total_time_ms += elapsed_ms

            avg_throughput = (
                total_tokens / (total_time_ms / 1000.0)
            ) / self.config.num_measurement_steps
            measurements.append(
                {
                    "batch_size": float(bs),
                    "throughput_tokens_per_sec": avg_throughput,
                    "total_time_ms": total_time_ms / self.config.num_measurement_steps,
                }
            )

        stats = _compute_statistics(measurements)
        env = _collect_environment()
        return BenchmarkResult(
            config=self.config,
            measurements=measurements,
            statistics=stats,
            environment=env,
        )

    def run_memory_benchmark(self, inference_fn: Callable, **kwargs: Any) -> BenchmarkResult:
        """运行内存基准测试。

        测量峰值显存、分配量、预留量和内存碎片率。

        Args:
            inference_fn: 推理函数。
            **kwargs: 额外参数。

        Returns:
            BenchmarkResult 包含内存指标。
        """
        measurements: List[Dict[str, float]] = []
        for _ in range(self.config.num_measurement_steps):
            if HAS_TORCH and torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats()
                torch.cuda.synchronize()

            inference_fn(**kwargs)

            if HAS_TORCH and torch.cuda.is_available():
                torch.cuda.synchronize()
                peak_allocated = torch.cuda.max_memory_allocated() / (1024**2)
                peak_reserved = torch.cuda.max_memory_reserved() / (1024**2)
                current_allocated = torch.cuda.memory_allocated() / (1024**2)
                current_reserved = torch.cuda.memory_reserved() / (1024**2)
                total = torch.cuda.get_device_properties(0).total_memory / (1024**2)
                fragmentation = (
                    (current_reserved - current_allocated) / current_reserved
                    if current_reserved > 0
                    else 0.0
                )
                utilization = current_allocated / total if total > 0 else 0.0

                measurements.append(
                    {
                        "peak_allocated_mb": float(peak_allocated),
                        "peak_reserved_mb": float(peak_reserved),
                        "current_allocated_mb": float(current_allocated),
                        "current_reserved_mb": float(current_reserved),
                        "fragmentation_ratio": float(fragmentation),
                        "utilization_ratio": float(utilization),
                        "total_memory_mb": float(total),
                    }
                )
            else:
                measurements.append(
                    {
                        "peak_allocated_mb": 0.0,
                        "peak_reserved_mb": 0.0,
                        "current_allocated_mb": 0.0,
                        "current_reserved_mb": 0.0,
                        "fragmentation_ratio": 0.0,
                        "utilization_ratio": 0.0,
                        "total_memory_mb": 0.0,
                    }
                )

        stats = _compute_statistics(measurements)
        env = _collect_environment()
        return BenchmarkResult(
            config=self.config,
            measurements=measurements,
            statistics=stats,
            environment=env,
        )

    def run_accuracy_benchmark(
        self,
        inference_fn: Callable,
        baseline_fn: Callable,
        **kwargs: Any,
    ) -> BenchmarkResult:
        """运行精度基准测试。

        对比优化实现与基线实现的 perplexity、BLEU 和相对误差。

        Args:
            inference_fn: 优化后的推理函数。
            baseline_fn: 基线（如 FP32）推理函数。
            **kwargs: 额外参数，需包含 input_data 或 logits/labels。

        Returns:
            BenchmarkResult 包含精度指标。
        """
        measurements: List[Dict[str, float]] = []

        for _ in range(self.config.num_measurement_steps):
            baseline_result = baseline_fn(**kwargs)
            optimized_result = inference_fn(**kwargs)

            metrics: Dict[str, float] = {}

            # If logits and labels are provided, compute perplexity
            if "logits" in kwargs and "labels" in kwargs:
                baseline_logits = (
                    baseline_fn(**kwargs) if callable(baseline_fn) else baseline_result
                )
                optimized_logits = optimized_result
                if HAS_TORCH and isinstance(optimized_logits, torch.Tensor):
                    ppl_baseline = Metrics.compute_perplexity(baseline_logits, kwargs["labels"])
                    ppl_optimized = Metrics.compute_perplexity(optimized_logits, kwargs["labels"])
                    metrics["perplexity_baseline"] = ppl_baseline
                    metrics["perplexity_optimized"] = ppl_optimized
                    metrics["perplexity_delta"] = ppl_optimized - ppl_baseline

            # If reference and candidate texts are provided, compute BLEU
            if "reference" in kwargs and "candidate" in kwargs:
                bleu = Metrics.compute_bleu_score(kwargs["reference"], kwargs["candidate"])
                metrics["bleu_score"] = bleu

            # Relative error if tensors are returned
            if (
                HAS_TORCH
                and isinstance(baseline_result, torch.Tensor)
                and isinstance(optimized_result, torch.Tensor)
            ):
                rel_err = Metrics.compute_relative_error(baseline_result, optimized_result)
                metrics["relative_error"] = rel_err
                cos_sim = Metrics.compute_cosine_similarity(baseline_result, optimized_result)
                metrics["cosine_similarity"] = cos_sim

            measurements.append(metrics)

        stats = _compute_statistics(measurements)
        env = _collect_environment()
        return BenchmarkResult(
            config=self.config,
            measurements=measurements,
            statistics=stats,
            environment=env,
        )

    def run_scalability_benchmark(self, inference_fn: Callable, **kwargs: Any) -> BenchmarkResult:
        """运行可扩展性基准测试（不同 seq_len）。

        测试模型在 128/512/2048/4096 等序列长度下的延迟和内存表现。

        Args:
            inference_fn: 推理函数，应接受 seq_len 参数。
            **kwargs: 额外参数。

        Returns:
            BenchmarkResult 包含各 seq_len 的指标。
        """
        measurements: List[Dict[str, float]] = []
        for seq_len in self.config.seq_lengths:
            kwargs_with_seq = dict(kwargs)
            kwargs_with_seq["seq_len"] = seq_len
            self._warmup(inference_fn, **kwargs_with_seq)

            latencies: List[float] = []
            peak_mems: List[float] = []
            for _ in range(self.config.num_measurement_steps):
                if HAS_TORCH and torch.cuda.is_available():
                    torch.cuda.reset_peak_memory_stats()
                    torch.cuda.synchronize()

                start = time.perf_counter()
                inference_fn(**kwargs_with_seq)
                if HAS_TORCH and torch.cuda.is_available():
                    torch.cuda.synchronize()
                end = time.perf_counter()

                latencies.append((end - start) * 1000.0)
                if HAS_TORCH and torch.cuda.is_available():
                    peak_mems.append(torch.cuda.max_memory_allocated() / (1024**2))
                else:
                    peak_mems.append(0.0)

            measurements.append(
                {
                    "seq_len": float(seq_len),
                    "latency_mean_ms": statistics.mean(latencies),
                    "latency_p95_ms": sorted(latencies)[
                        max(0, int(math.ceil(0.95 * len(latencies))) - 1)
                    ],
                    "peak_memory_mb": statistics.mean(peak_mems),
                }
            )

        stats = _compute_statistics(measurements)
        env = _collect_environment()
        return BenchmarkResult(
            config=self.config,
            measurements=measurements,
            statistics=stats,
            environment=env,
        )

    def run_roofline_benchmark(
        self,
        inference_fn: Callable,
        analyzer: RooflineAnalyzer,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """运行 Roofline 分析。

        分析推理函数中各操作的 FLOPs、内存访问和延迟，生成 Roofline 点。

        Args:
            inference_fn: 推理函数，应返回 flops, bytes_accessed, time_ms 的 dict。
            analyzer: RooflineAnalyzer 实例。
            **kwargs: 额外参数。

        Returns:
            包含 Roofline 点列表、报告和图表的字典。
        """
        # Collect operation data from the inference function
        points: List[RooflinePoint] = []
        for _ in range(self.config.num_measurement_steps):
            result = inference_fn(**kwargs)
            if isinstance(result, dict) and "operations" in result:
                for op in result["operations"]:
                    point = analyzer.analyze_operation(
                        name=op.get("name", "unknown"),
                        flops=op.get("flops", 0.0),
                        bytes_accessed=op.get("bytes_accessed", 0.0),
                        execution_time_ms=op.get("time_ms", 1.0),
                    )
                    points.append(point)

        # Generate report and plot
        report = analyzer.generate_optimization_report(points)
        plot_base64 = analyzer.generate_roofline_plot(points, title=f"Roofline: {self.config.name}")

        return {
            "points": [p.to_dict() for p in points],
            "report": report,
            "plot_base64": plot_base64,
            "analyzer_config": analyzer.to_dict(),
        }

    def run_full_suite(self, inference_fn: Callable, **kwargs: Any) -> Dict[str, BenchmarkResult]:
        """运行完整测试套件。

        依次执行 latency, throughput, memory, accuracy, scalability benchmark。

        Args:
            inference_fn: 推理函数。
            **kwargs: 额外参数。

        Returns:
            字典，键为测试名称，值为 BenchmarkResult。
        """
        return {
            "latency": self.run_latency_benchmark(inference_fn, **kwargs),
            "throughput": self.run_throughput_benchmark(inference_fn, **kwargs),
            "memory": self.run_memory_benchmark(inference_fn, **kwargs),
            "scalability": self.run_scalability_benchmark(inference_fn, **kwargs),
        }


class BenchmarkReport:
    """标准化性能报告生成器。

    将多个 BenchmarkResult 整合为完整的 Markdown / HTML 报告，
    包含环境信息、配置快照、统计摘要、对比分析和可视化图表。

    Attributes:
        results: 测试结果字典，键为测试名称。
    """

    def __init__(self, results: Dict[str, BenchmarkResult]) -> None:
        """初始化报告生成器。

        Args:
            results: 测试结果字典。
        """
        self.results = results

    def generate(self, output_dir: str) -> None:
        """生成完整报告。

        输出：
        1. 环境信息（environment.json）
        2. 配置快照（config_snapshot.json）
        3. 原始数据（CSV 格式）
        4. 统计摘要（statistics.json）
        5. 可视化图表（matplotlib 生成）
        6. Markdown 报告（report.md）
        7. HTML 报告（report.html）

        Args:
            output_dir: 输出目录。
        """
        out_path = Path(output_dir)
        out_path.mkdir(parents=True, exist_ok=True)

        for name, result in self.results.items():
            subdir = out_path / name
            subdir.mkdir(parents=True, exist_ok=True)
            result.export_csv(str(subdir / "measurements.csv"))
            result.export_json(str(subdir / "result.json"))
            result.generate_plots(str(subdir / "plots"))

        self.generate_markdown(str(out_path / "report.md"))
        self.generate_html(str(out_path / "report.html"))

    def generate_markdown(self, output_path: str) -> None:
        """生成 Markdown 报告。

        Args:
            output_path: 输出 Markdown 文件路径。
        """
        lines: List[str] = []
        lines.append("# ShadowInfer Benchmark Report\n\n")
        lines.append(f"**Generated:** {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")

        for name, result in self.results.items():
            lines.append(f"## {name.upper()} Benchmark\n\n")
            lines.append(result.get_summary())
            lines.append("\n---\n\n")

        Path(output_path).write_text("".join(lines), encoding="utf-8")

    def generate_html(self, output_path: str) -> None:
        """生成 HTML 报告。

        Args:
            output_path: 输出 HTML 文件路径。
        """
        html_parts: List[str] = []
        html_parts.append('<!DOCTYPE html>\n<html lang="en">\n<head>\n')
        html_parts.append('<meta charset="UTF-8">\n')
        html_parts.append(
            '<meta name="viewport" content="width=device-width, initial-scale=1.0">\n'
        )
        html_parts.append("<title>ShadowInfer Benchmark Report</title>\n")
        html_parts.append("<style>\n")
        html_parts.append("""
body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto,
    sans-serif; margin: 40px; background: #f8f9fa; color: #333; }
.container { max-width: 1200px; margin: 0 auto; background: white; padding: 40px;
    border-radius: 8px; box-shadow: 0 2px 8px rgba(0,0,0,0.1); }
h1 { color: #2c3e50; border-bottom: 3px solid #3498db;
    padding-bottom: 12px; }
h2 { color: #34495e; margin-top: 40px; border-left: 4px solid #3498db;
    padding-left: 16px; }
table { border-collapse: collapse; width: 100%; margin: 20px 0; }
th, td { border: 1px solid #ddd; padding: 10px 12px;
    text-align: left; }
th { background: #f1f5f9; font-weight: 600; }
tr:nth-child(even) { background: #fafbfc; }
.info { background: #e8f4f8; border-left: 4px solid #3498db;
    padding: 12px 16px; margin: 12px 0; border-radius: 4px; }
.warning { background: #fff3cd; border-left: 4px solid #f39c12;
    padding: 12px 16px; margin: 12px 0; border-radius: 4px; }
.success { background: #d4edda; border-left: 4px solid #27ae60;
    padding: 12px 16px; margin: 12px 0; border-radius: 4px; }
        """)
        html_parts.append('</style>\n</head>\n<body>\n<div class="container">\n')
        html_parts.append("<h1>ShadowInfer Benchmark Report</h1>\n")
        html_parts.append(
            f"<p><strong>Generated:</strong> {time.strftime('%Y-%m-%d %H:%M:%S')}</p>\n"
        )

        for name, result in self.results.items():
            html_parts.append(f"<h2>{name.upper()} Benchmark</h2>\n")

            # Environment
            if result.environment:
                html_parts.append('<div class="info">\n<strong>Environment</strong><br>\n')
                for k, v in result.environment.items():
                    html_parts.append(f"{k}: {v}<br>\n")
                html_parts.append("</div>\n")

            # Statistics table
            if result.statistics:
                html_parts.append("<table>\n<tr><th>Metric</th>")
                first_stats = next(iter(result.statistics.values()))
                for stat_key in first_stats.keys():
                    html_parts.append(f"<th>{stat_key}</th>")
                html_parts.append("</tr>\n")

                for metric_name, stats in result.statistics.items():
                    html_parts.append(f"<tr><td>{metric_name}</td>")
                    for stat_key in first_stats.keys():
                        val = stats.get(stat_key, 0.0)
                        html_parts.append(f"<td>{val:.4f}</td>")
                    html_parts.append("</tr>\n")
                html_parts.append("</table>\n")

            html_parts.append("<hr>\n")

        html_parts.append("</div>\n</body>\n</html>\n")
        Path(output_path).write_text("".join(html_parts), encoding="utf-8")
