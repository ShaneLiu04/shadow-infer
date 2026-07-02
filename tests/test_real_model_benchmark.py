"""
Tests for real model benchmark module.
"""

import os
import tempfile

from benchmarks.real_model import (
    BenchmarkConfig,
    RealModelBenchmark,
    RunResult,
    StepMetrics,
)


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

    def test_optimized_faster_than_baseline(self):
        config = BenchmarkConfig(num_steps=10, num_prompts=1)
        benchmark = RealModelBenchmark(config)

        baseline = benchmark.run_baseline(0)
        optimized = benchmark.run_optimized(0)

        assert optimized.avg_step_latency_ms < baseline.avg_step_latency_ms

    def test_optimized_uses_less_memory(self):
        config = BenchmarkConfig(num_steps=10, num_prompts=1)
        benchmark = RealModelBenchmark(config)

        baseline = benchmark.run_baseline(0)
        optimized = benchmark.run_optimized(0)

        assert optimized.max_memory_mb < baseline.max_memory_mb

    def test_comparison(self):
        config = BenchmarkConfig(num_steps=10, num_prompts=5)
        benchmark = RealModelBenchmark(config)

        result = benchmark.run_comparison()

        assert result.speedup > 1.0
        assert result.memory_savings > 0.0
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
            assert data["summary"]["speedup"] > 1.0
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
        """Test that step latency varies realistically."""
        config = BenchmarkConfig(num_steps=10, num_prompts=1)
        benchmark = RealModelBenchmark(config)

        result = benchmark.run_optimized(0)

        latencies = [s.latency_ms for s in result.steps]

        # Early steps should be faster (more aggressive optimization)
        assert latencies[0] < latencies[-1] * 1.5

        # Memory should grow
        memories = [s.memory_mb for s in result.steps]
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
