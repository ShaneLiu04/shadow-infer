"""ShadowInfer Benchmarking Package.

Provides performance benchmarking tools and Roofline model analysis
for evaluating AI inference performance.

Version: 3.0
Corresponds to: plan-v2.md Phase 2
"""

__version__ = "3.0"

from .benchmark_suite import (
    BenchmarkConfig,
    BenchmarkReport,
    BenchmarkResult,
    BenchmarkRunner,
)
from .roofline import RooflineAnalyzer, RooflinePoint

__all__ = [
    "RooflinePoint",
    "RooflineAnalyzer",
    "BenchmarkConfig",
    "BenchmarkResult",
    "BenchmarkRunner",
    "BenchmarkReport",
]
