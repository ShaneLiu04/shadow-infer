"""Roofline 模型分析器。

Roofline 模型是理解性能瓶颈的经典方法：
- 横轴：Operational Intensity (OI) = FLOPs / Bytes
- 纵轴：Performance (GFLOPs/s)
- 峰值受限于：内存带宽 或 计算峰值

对应大厂实践：MLPerf, Roofline Model 分析
对应文档：plan-v2.md Phase 2.2

Version: 3.0
"""

from __future__ import annotations

import base64
import io
import math
from dataclasses import dataclass
from typing import Any, Dict, List

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False

__version__ = "3.0"


@dataclass
class RooflinePoint:
    """Roofline 模型上的点。

    记录单个操作的性能特征，包括 Operational Intensity (OI)、
    实际性能、理论峰值以及瓶颈类型。

    Attributes:
        name: 操作名称（如 "Attention", "FFN", "ShadowKV"）。
        operational_intensity: OI = FLOPs / Bytes。
        performance: 实际性能 (GFLOPs/s)。
        theoretical_peak: 理论峰值 (GFLOPs/s)。
        bottleneck: 瓶颈类型，"memory" | "compute" | "balanced"。
    """

    name: str
    operational_intensity: float
    performance: float
    theoretical_peak: float
    bottleneck: str

    def efficiency(self) -> float:
        """计算效率 = 实际性能 / 理论峰值。

        Returns:
            效率百分比（0.0 ~ 1.0）。返回 -1.0 表示理论峰值为 0。
        """
        if self.theoretical_peak <= 0.0:
            return -1.0
        return min(1.0, self.performance / self.theoretical_peak)

    def headroom(self) -> float:
        """计算优化空间 = 理论峰值 - 实际性能。

        Returns:
            可提升的性能空间 (GFLOPs/s)，若理论峰值为 0 则返回 -1.0。
        """
        if self.theoretical_peak <= 0.0:
            return -1.0
        return self.theoretical_peak - self.performance

    def to_dict(self) -> Dict[str, Any]:
        """将 RooflinePoint 序列化为字典。

        Returns:
            包含所有字段及派生指标的字典。
        """
        return {
            "name": self.name,
            "operational_intensity": self.operational_intensity,
            "performance": self.performance,
            "theoretical_peak": self.theoretical_peak,
            "bottleneck": self.bottleneck,
            "efficiency": self.efficiency(),
            "headroom": self.headroom(),
        }


class RooflineAnalyzer:
    """Roofline 模型分析器。

    基于 Roofline 模型分析计算密集型 vs 内存密集型操作的性能瓶颈。
    核心思想：
    - 性能上限 = min(峰值算力, 峰值带宽 × OI)
    - 拐点 (ridge point) = 峰值算力 / 峰值带宽
    - OI < ridge_point: 内存带宽瓶颈
    - OI > ridge_point: 计算瓶颈

    Attributes:
        peak_compute: 理论计算峰值 (GFLOPs/s)。
        peak_memory: 理论内存带宽 (GB/s)。
        ridge_point: 拐点 OI 值。
    """

    def __init__(
        self,
        peak_compute_gflops: float,
        peak_memory_bandwidth_gbps: float,
    ) -> None:
        """初始化 Roofline 分析器。

        Args:
            peak_compute_gflops: 理论计算峰值 (GFLOPs/s)。
            peak_memory_bandwidth_gbps: 理论内存带宽 (GB/s)。

        Raises:
            ValueError: 若输入参数为非正值。
        """
        if peak_compute_gflops <= 0.0 or peak_memory_bandwidth_gbps <= 0.0:
            raise ValueError("Peak compute and bandwidth must be positive")
        self.peak_compute = float(peak_compute_gflops)
        self.peak_memory = float(peak_memory_bandwidth_gbps)
        self.ridge_point = self.peak_compute / self.peak_memory

    def analyze_operation(
        self,
        name: str,
        flops: float,
        bytes_accessed: float,
        execution_time_ms: float,
    ) -> RooflinePoint:
        """分析单个操作的性能瓶颈。

        计算 OI、实际性能、理论峰值，并判定瓶颈类型。

        Args:
            name: 操作名称标识。
            flops: 操作的总 FLOPs（浮点运算次数）。
            bytes_accessed: 访问的总字节数（内存移动量）。
            execution_time_ms: 执行时间（毫秒）。

        Returns:
            RooflinePoint 分析结果。

        Raises:
            ValueError: 若 flops、bytes_accessed 或 execution_time_ms 为负或零。
        """
        if flops < 0 or bytes_accessed < 0 or execution_time_ms <= 0:
            raise ValueError(
                "flops and bytes must be non-negative, execution_time must be positive"
            )

        if bytes_accessed == 0.0:
            operational_intensity = float("inf")
        else:
            operational_intensity = flops / bytes_accessed

        # Performance in GFLOPs/s
        execution_time_s = execution_time_ms / 1000.0
        performance = (flops / 1e9) / execution_time_s if execution_time_s > 0 else 0.0

        # Theoretical peak for this OI
        memory_bound_peak = self.peak_memory * operational_intensity  # GB/s * FLOPs/Byte = GFLOPs/s
        theoretical_peak = min(self.peak_compute, memory_bound_peak)

        if math.isinf(operational_intensity):
            bottleneck = "compute"
        elif operational_intensity < self.ridge_point * 0.8:
            bottleneck = "memory"
        elif operational_intensity > self.ridge_point * 1.2:
            bottleneck = "compute"
        else:
            bottleneck = "balanced"

        return RooflinePoint(
            name=name,
            operational_intensity=operational_intensity,
            performance=performance,
            theoretical_peak=theoretical_peak,
            bottleneck=bottleneck,
        )

    def analyze_model_layer(
        self,
        layer_name: str,
        attention_flops: float,
        attention_bytes: float,
        attention_time_ms: float,
        ffn_flops: float,
        ffn_bytes: float,
        ffn_time_ms: float,
    ) -> Dict[str, RooflinePoint]:
        """分析模型层的 Attention + FFN 两个主要组件。

        Args:
            layer_name: 层名称标识。
            attention_flops: Attention 模块的 FLOPs。
            attention_bytes: Attention 模块的内存访问字节数。
            attention_time_ms: Attention 执行时间（ms）。
            ffn_flops: FFN 模块的 FLOPs。
            ffn_bytes: FFN 模块的内存访问字节数。
            ffn_time_ms: FFN 执行时间（ms）。

        Returns:
            字典，键为 "attention" 和 "ffn"，值为 RooflinePoint。
        """
        return {
            "attention": self.analyze_operation(
                name=f"{layer_name}.Attention",
                flops=attention_flops,
                bytes_accessed=attention_bytes,
                execution_time_ms=attention_time_ms,
            ),
            "ffn": self.analyze_operation(
                name=f"{layer_name}.FFN",
                flops=ffn_flops,
                bytes_accessed=ffn_bytes,
                execution_time_ms=ffn_time_ms,
            ),
        }

    def generate_roofline_plot(
        self,
        points: List[RooflinePoint],
        title: str = "Roofline Performance Model",
    ) -> str:
        """生成 Roofline 图，返回 base64 编码的 PNG 字符串。

        使用 matplotlib 绘制 Roofline 曲线，并将各操作点标记在图上。

        Args:
            points: RooflinePoint 列表。
            title: 图表标题。

        Returns:
            base64 编码的 PNG 图像字符串。若 matplotlib 不可用，返回空字符串。
        """
        if not HAS_MATPLOTLIB:
            return ""

        # Filter valid finite points for plotting
        plot_points = [
            p
            for p in points
            if math.isfinite(p.operational_intensity) and math.isfinite(p.performance)
        ]

        if not plot_points:
            return ""

        fig, ax = plt.subplots(figsize=(10, 7))

        # Determine x-axis range
        all_ois = [p.operational_intensity for p in plot_points]
        min_oi = min(1.0, min(all_ois) * 0.5) if all_ois else 0.1
        max_oi = (
            max(self.ridge_point * 3.0, max(all_ois) * 3.0) if all_ois else self.ridge_point * 10.0
        )

        oi_range = np.logspace(np.log10(min_oi), np.log10(max_oi), 500)

        # Roofline curve: min(peak_compute, peak_memory * oi)
        roofline = np.minimum(self.peak_compute, self.peak_memory * oi_range)

        ax.plot(oi_range, roofline, "k-", linewidth=2.5, label="Roofline")
        ax.axvline(
            self.ridge_point,
            color="gray",
            linestyle="--",
            linewidth=1.0,
            label=f"Ridge Point (OI={self.ridge_point:.2f})",
        )

        # Color by bottleneck type
        color_map = {"memory": "#E74C3C", "compute": "#3498DB", "balanced": "#2ECC71"}
        for p in plot_points:
            color = color_map.get(p.bottleneck, "#9B59B6")
            ax.scatter(
                [p.operational_intensity],
                [p.performance],
                c=color,
                s=120,
                zorder=5,
                edgecolors="white",
                linewidths=1.5,
            )
            ax.annotate(
                p.name,
                (p.operational_intensity, p.performance),
                textcoords="offset points",
                xytext=(8, 8),
                fontsize=9,
                fontweight="bold",
            )

        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("Operational Intensity (FLOPs / Byte)", fontsize=12)
        ax.set_ylabel("Performance (GFLOPs/s)", fontsize=12)
        ax.set_title(title, fontsize=14, fontweight="bold")
        ax.legend(loc="upper left")
        ax.grid(True, which="both", linestyle="--", alpha=0.5)
        ax.set_xlim(min_oi, max_oi)
        ax.set_ylim(0.1, self.peak_compute * 2.0)

        # Add peak labels
        ax.text(
            max_oi * 0.6,
            self.peak_compute * 1.1,
            f"Peak Compute: {self.peak_compute:.1f} GFLOPs/s",
            fontsize=10,
            color="#3498DB",
            fontweight="bold",
        )
        ax.text(
            self.ridge_point * 0.3,
            self.peak_memory * self.ridge_point * 0.3 * 1.5,
            f"Peak Memory: {self.peak_memory:.1f} GB/s",
            fontsize=10,
            color="#E74C3C",
            fontweight="bold",
            rotation=35,
        )

        plt.tight_layout()

        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=150, bbox_inches="tight")
        plt.close(fig)
        buf.seek(0)
        img_base64 = base64.b64encode(buf.read()).decode("utf-8")
        return img_base64

    def generate_optimization_report(self, points: List[RooflinePoint]) -> str:
        """生成优化建议报告。

        识别三类操作并给出针对性优化建议：
        1. 内存瓶颈的操作（OI < ridge_point, efficiency < 50%）
        2. 计算瓶颈的操作（OI > ridge_point, efficiency < 50%）
        3. 已接近峰值的操作（efficiency > 80%）

        建议：
        - 内存瓶颈 → 提升 OI（数据复用、分块 tiling、量化压缩）
        - 计算瓶颈 → 提升计算效率（向量化、融合 kernel、减少 launch overhead）
        - 接近峰值 → 保持现状，或探索更高峰值硬件

        Args:
            points: RooflinePoint 列表。

        Returns:
            Markdown 格式的优化建议报告字符串。
        """
        lines: List[str] = []
        lines.append("# Roofline Optimization Report\n")
        lines.append(
            f"**Hardware Configuration:** Peak Compute = {self.peak_compute:.1f} GFLOPs/s, "
            f"Peak Memory Bandwidth = {self.peak_memory:.1f} GB/s, "
            f"Ridge Point OI = {self.ridge_point:.2f}\n"
        )

        memory_bottleneck = []
        compute_bottleneck = []
        near_peak = []
        balanced_low = []

        for p in points:
            eff = p.efficiency()
            if eff < 0:
                continue
            if p.bottleneck == "memory" and eff < 0.5:
                memory_bottleneck.append(p)
            elif p.bottleneck == "compute" and eff < 0.5:
                compute_bottleneck.append(p)
            elif eff > 0.8:
                near_peak.append(p)
            elif eff < 0.5:
                balanced_low.append(p)

        # Memory bottleneck section
        lines.append("## 1. Memory-Bound Operations (Priority: High)\n")
        if memory_bottleneck:
            lines.append(
                "These operations are limited by memory bandwidth. Focus on increasing OI:\n"
            )
            lines.append("| Operation | OI | Performance (GFLOPs/s) | Efficiency | Headroom |\n")
            lines.append("|-----------|-----|------------------------|------------|----------|\n")
            for p in memory_bottleneck:
                lines.append(
                    f"| {p.name} | {p.operational_intensity:.2f} | "
                    f"{p.performance:.2f} | {p.efficiency()*100:.1f}% | "
                    f"{p.headroom():.2f} |\n"
                )
            lines.append("\n**Optimization Suggestions:**\n")
            lines.append("- **Tiling / Blocking:** Reorder loops to improve cache locality.\n")
            lines.append(
                "- **Data Reuse:** Keep frequently accessed data in registers / shared memory.\n"
            )
            lines.append(
                "- **Quantization / Compression:** Reduce bytes moved "
                "(e.g., INT8 / INT4 KV cache).\n"
            )
            lines.append(
                "- **Kernel Fusion:** Merge element-wise ops to reduce memory round-trips.\n"
            )
        else:
            lines.append("No memory-bound operations with low efficiency found. Good job!\n")

        # Compute bottleneck section
        lines.append("\n## 2. Compute-Bound Operations (Priority: Medium)\n")
        if compute_bottleneck:
            lines.append(
                "These operations are limited by compute throughput. Focus on utilization:\n"
            )
            lines.append("| Operation | OI | Performance (GFLOPs/s) | Efficiency | Headroom |\n")
            lines.append("|-----------|-----|------------------------|------------|----------|\n")
            for p in compute_bottleneck:
                lines.append(
                    f"| {p.name} | {p.operational_intensity:.2f} | "
                    f"{p.performance:.2f} | {p.efficiency()*100:.1f}% | "
                    f"{p.headroom():.2f} |\n"
                )
            lines.append("\n**Optimization Suggestions:**\n")
            lines.append(
                "- **Vectorization:** Use wider SIMD / Tensor Core instructions (FP16/BF16/TF32).\n"
            )
            lines.append(
                "- **Kernel Fusion:** Combine multiple compute ops to reduce launch overhead.\n"
            )
            lines.append(
                "- **Instruction Mix:** Ensure FMA utilization; avoid data-dependent branches.\n"
            )
            lines.append("- **Occupancy:** Tune block/grid sizes for maximum SM utilization.\n")
        else:
            lines.append("No compute-bound operations with low efficiency found.\n")

        # Near peak section
        lines.append("\n## 3. Near-Peak Operations (Efficiency > 80%)\n")
        if near_peak:
            lines.append("These operations are already well-optimized:\n")
            lines.append("| Operation | OI | Performance (GFLOPs/s) | Efficiency |\n")
            lines.append("|-----------|-----|------------------------|------------|\n")
            for p in near_peak:
                lines.append(
                    f"| {p.name} | {p.operational_intensity:.2f} | "
                    f"{p.performance:.2f} | {p.efficiency()*100:.1f}% |\n"
                )
            lines.append(
                "\n**Suggestion:** Keep current implementation. Consider hardware "
                "upgrade for further gains.\n"
            )
        else:
            lines.append(
                "No operations near peak efficiency. Room for improvement across the board.\n"
            )

        # Balanced low efficiency
        if balanced_low:
            lines.append("\n## 4. Balanced but Low Efficiency Operations\n")
            lines.append("| Operation | OI | Performance (GFLOPs/s) | Efficiency |\n")
            lines.append("|-----------|-----|------------------------|------------|\n")
            for p in balanced_low:
                lines.append(
                    f"| {p.name} | {p.operational_intensity:.2f} | "
                    f"{p.performance:.2f} | {p.efficiency()*100:.1f}% |\n"
                )
            lines.append(
                "\n**Suggestion:** Both memory and compute are underutilized. "
                "Check for sync overhead, thread divergence, or suboptimal scheduling.\n"
            )

        return "".join(lines)

    def to_dict(self) -> Dict[str, float]:
        """导出分析器配置为字典。

        Returns:
            包含 peak_compute, peak_memory, ridge_point 的字典。
        """
        return {
            "peak_compute_gflops": self.peak_compute,
            "peak_memory_bandwidth_gbps": self.peak_memory,
            "ridge_point": self.ridge_point,
        }
