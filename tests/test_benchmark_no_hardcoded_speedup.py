"""AR002/T002 基准 mock 源码与行为均不得包含硬编码加速差分。"""

import re
from pathlib import Path

import pytest

from benchmarks.real_model import BenchmarkConfig, RealModelBenchmark

BENCHMARK_SOURCE_PATH = (
    Path(__file__).resolve().parents[1] / "benchmarks" / "real_model" / "__init__.py"
)


class TestNoHardcodedSpeedupSource:
    def test_no_fabricated_speedup_markers(self):
        """源码不得残留伪造加速的标记文本。"""
        source = BENCHMARK_SOURCE_PATH.read_text(encoding="utf-8")
        assert "3x speedup" not in source
        assert "70% speedup" not in source
        assert "30% speedup" not in source

    def test_no_latency_multiplicative_differential(self):
        """_mock_model_step 不得存在针对 optimized 的 latency 乘法差分。"""
        source = BENCHMARK_SOURCE_PATH.read_text(encoding="utf-8")
        assert re.search(r"latency\s*\*=\s*0\.\d", source) is None
        assert re.search(r"latency\s*=\s*base_latency\s*\*\s*0\.\d", source) is None


class TestNoHardcodedSpeedupBehavior:
    def test_mock_speedup_approx_one(self):
        """无伪造差分时，mock 对比的 speedup 应近似 1.0。"""
        config = BenchmarkConfig(num_steps=10, num_prompts=2)
        benchmark = RealModelBenchmark(config)

        result = benchmark.run_comparison()

        assert result.speedup == pytest.approx(1.0, abs=0.05)
