"""
Tests for real model benchmark module.
"""

import os
import tempfile

import pytest

from benchmarks.real_model import BenchmarkConfig, RealModelBenchmark, RunResult, StepMetrics


class TestBenchmarkConfig:
    def test_default_config(self):
        config = BenchmarkConfig()
        assert config.model_name == "fast-dllm-1.5b"
        assert config.num_steps == 50
        assert config.num_prompts == 20

    def test_custom_config(self):
        config = BenchmarkConfig(model_name="test", num_steps=10, num_prompts=5)
        assert config.model_name == "test"
        assert config.num_steps == 10
        assert config.num_prompts == 5


class TestStepMetrics:
    def test_metrics_creation(self):
        metrics = StepMetrics(
            step_id=5,
            latency_ms=10.0,
            memory_mb=500.0,
            kv_cache_size_mb=50.0,
            ffn_compute_ratio=0.25,
            attention_time_ms=4.0,
            ffn_time_ms=5.0,
        )
        assert metrics.step_id == 5
        assert metrics.latency_ms == 10.0


class TestRunResult:
    def test_result_creation(self):
        steps = [StepMetrics(i, 10.0, 500.0, 10.0 * i, 0.25, 4.0, 5.0) for i in range(5)]
        result = RunResult(
            config_name="test",
            prompt_id=0,
            total_time_ms=100.0,
            avg_step_latency_ms=20.0,
            max_memory_mb=500.0,
            final_kv_cache_mb=40.0,
            steps=steps,
        )
        assert result.config_name == "test"
        assert len(result.steps) == 5


class TestRealModelBenchmark:
    def test_baseline_run(self):
        config = BenchmarkConfig(num_steps=10, num_prompts=1)
        benchmark = RealModelBenchmark(config)

        result = benchmark.run_baseline(0)

        assert result.config_name == "baseline"
        assert result.prompt_id == 0
        assert len(result.steps) == 10
        assert result.avg_step_latency_ms > 0

    def test_optimized_run(self):
        config = BenchmarkConfig(num_steps=10, num_prompts=1)
        benchmark = RealModelBenchmark(config)

        result = benchmark.run_optimized(0)

        assert result.config_name == "optimized"
        assert len(result.steps) == 10
        assert result.avg_step_latency_ms > 0

    def test_mock_has_no_fabricated_speedup(self):
        """Mock 对比不得携带硬编码加速差分：speedup 应近似 1（契约 G）。"""
        config = BenchmarkConfig(num_steps=10, num_prompts=2)
        benchmark = RealModelBenchmark(config)

        result = benchmark.run_comparison()

        assert abs(result.speedup - 1.0) < 0.05

    def test_optimized_uses_less_memory(self):
        """Mock 对比不得携带硬编码内存差分：memory_savings 应近似 0（契约 G）。"""
        config = BenchmarkConfig(num_steps=10, num_prompts=2)
        benchmark = RealModelBenchmark(config)

        result = benchmark.run_comparison()

        assert abs(result.memory_savings) < 0.05

    def test_comparison(self):
        config = BenchmarkConfig(num_steps=10, num_prompts=5)
        benchmark = RealModelBenchmark(config)

        result = benchmark.run_comparison()

        # AR002 M6 契约 G：mock 无优化差分，speedup/memory_savings 均应近似 1.0/0.0。
        assert result.speedup == pytest.approx(1.0, abs=0.05)
        assert abs(result.memory_savings) < 0.05
        assert result.accuracy_drop < 0.01
        assert result.latency_p50 > 0
        assert result.latency_p95 > 0
        assert result.latency_p99 > 0
        assert len(result.baseline_results) == 5
        assert len(result.optimized_results) == 5

    def test_generate_markdown_report(self):
        config = BenchmarkConfig(num_steps=5, num_prompts=2)
        benchmark = RealModelBenchmark(config)
        result = benchmark.run_comparison()

        fd, path = tempfile.mkstemp(suffix=".md")
        try:
            benchmark.generate_markdown_report(result, path)
            with open(path, "r", encoding="utf-8") as rf:
                content = rf.read()

            assert "ShadowInfer Benchmark Report" in content
            assert "Speedup" in content
            assert "Memory Efficiency" in content  # or "Avg Memory" in report
        finally:
            os.close(fd)
            os.unlink(path)

    def test_generate_json_report(self):
        import json

        config = BenchmarkConfig(num_steps=5, num_prompts=2)
        benchmark = RealModelBenchmark(config)
        result = benchmark.run_comparison()

        fd, path = tempfile.mkstemp(suffix=".json")
        try:
            benchmark.generate_json_report(result, path)
            with open(path, "r", encoding="utf-8") as rf:
                data = json.load(rf)

            assert "config" in data
            assert "summary" in data
            assert "baseline" in data
            assert "optimized" in data
            # AR002 M6 契约 G：mock 无优化差分，speedup 应近似 1.0。
            assert data["summary"]["speedup"] == pytest.approx(1.0, abs=0.05)
            # JSON 报告 summary 必须标注 mode 口径，避免 mock 数据被误读为实测。
            assert data["summary"]["mode"] == "mock"
        finally:
            os.close(fd)
            os.unlink(path)

    def test_generate_html_report(self):
        config = BenchmarkConfig(num_steps=5, num_prompts=2)
        benchmark = RealModelBenchmark(config)
        result = benchmark.run_comparison()

        fd, path = tempfile.mkstemp(suffix=".html")
        try:
            benchmark.generate_html_report(result, path)
            with open(path, "r", encoding="utf-8") as rf:
                content = rf.read()

            assert "<!DOCTYPE html>" in content
            assert "ShadowInfer Benchmark Report" in content
            assert "echarts" in content
        finally:
            os.close(fd)
            os.unlink(path)

    def test_step_latency_variance(self):
        """Test mock step latency stays sane without fabricated per-step gaps."""
        config = BenchmarkConfig(num_steps=10, num_prompts=1)
        benchmark = RealModelBenchmark(config)

        baseline = benchmark.run_baseline(0)
        optimized = benchmark.run_optimized(0)

        latencies = [s.latency_ms for s in optimized.steps]

        # AR002 M6 契约 G：无伪造差分，latency 仅有自然噪声/增长，全部为正
        # 且处于合理上界内。
        for value in latencies:
            assert 0.0 < value < 100.0

        # 优化前后同公式：任意 step 的 latency 比值不得出现伪造性偏离。
        for b_step, o_step in zip(baseline.steps, optimized.steps):
            ratio = o_step.latency_ms / b_step.latency_ms
            assert 0.5 <= ratio <= 2.0

        # Memory should still grow naturally
        memories = [s.memory_mb for s in optimized.steps]
        assert memories[-1] > memories[0]

    def test_kv_cache_growth(self):
        """Test KV cache grows over steps."""
        config = BenchmarkConfig(num_steps=10, num_prompts=1)
        benchmark = RealModelBenchmark(config)

        result = benchmark.run_optimized(0)

        kv_sizes = [s.kv_cache_size_mb for s in result.steps]

        for i in range(1, len(kv_sizes)):
            assert kv_sizes[i] >= kv_sizes[i - 1]

    def test_ffn_compute_ratio(self):
        """Test FFN compute ratio is between 0 and 1."""
        config = BenchmarkConfig(num_steps=10, num_prompts=1)
        benchmark = RealModelBenchmark(config)

        result = benchmark.run_optimized(0)

        for step in result.steps:
            assert 0.0 <= step.ffn_compute_ratio <= 1.0
