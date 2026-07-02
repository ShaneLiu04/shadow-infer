"""
Tests for A/B testing framework.
"""

from shadowinfer.ab_testing import (
    ABTestFramework,
    MultiStrategyABTest,
    RequestMetrics,
    StatisticalTest,
    StrategyConfig,
    StrategyName,
    StrategyStats,
    TrafficSplitter,
)


class TestStrategyName:
    def test_strategy_values(self):
        assert StrategyName.BASELINE.value == "baseline"
        assert StrategyName.FULL_OPTIMIZATION.value == "full_optimization"


class TestStrategyConfig:
    def test_default_config(self):
        config = StrategyConfig(name=StrategyName.BASELINE)
        assert config.shadowkv_enabled is False
        assert config.qdrift_enabled is False
        assert config.ffn_enabled is False

    def test_full_optimization_config(self):
        config = StrategyConfig(
            name=StrategyName.FULL_OPTIMIZATION,
            shadowkv_enabled=True,
            qdrift_enabled=True,
            ffn_enabled=True,
        )
        assert config.shadowkv_enabled is True
        assert config.qdrift_enabled is True
        assert config.ffn_enabled is True


class TestRequestMetrics:
    def test_metrics_creation(self):
        metrics = RequestMetrics(
            request_id="req_123",
            strategy=StrategyName.BASELINE,
            latency_ms=50.0,
            memory_mb=4000.0,
            accuracy_score=0.95,
            success=True,
        )
        assert metrics.request_id == "req_123"
        assert metrics.success is True


class TestStrategyStats:
    def test_empty_stats(self):
        stats = StrategyStats(strategy=StrategyName.BASELINE)
        assert stats.request_count == 0
        assert stats.avg_latency == 0.0
        assert stats.p95_latency() == 0.0

    def test_avg_latency(self):
        stats = StrategyStats(strategy=StrategyName.BASELINE)
        stats.request_count = 3
        stats.total_latency_ms = 150.0
        assert stats.avg_latency == 50.0

    def test_success_rate(self):
        stats = StrategyStats(strategy=StrategyName.BASELINE)
        stats.request_count = 10
        stats.success_count = 9
        assert stats.success_rate == 0.9

    def test_p95_latency(self):
        stats = StrategyStats(strategy=StrategyName.BASELINE)
        stats.latencies = [10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0, 90.0, 100.0]
        stats.request_count = 10

        p95 = stats.p95_latency()
        # 95th percentile of 10 items: index 9 (0-indexed) = 100.0
        assert p95 == 100.0

    def test_p99_latency(self):
        stats = StrategyStats(strategy=StrategyName.BASELINE)
        stats.latencies = list(range(1, 101))
        stats.request_count = 100

        p99 = stats.p99_latency()
        # 99th percentile of 100 items: index 99 = 100.0
        assert p99 == 100.0


class TestTrafficSplitter:
    def test_deterministic_routing(self):
        splitter = TrafficSplitter(
            StrategyName.BASELINE,
            StrategyName.FULL_OPTIMIZATION,
            experiment_ratio=0.1,
        )

        # Same request ID always routes to same strategy
        strategy1 = splitter.route("req_123")
        strategy2 = splitter.route("req_123")
        assert strategy1 == strategy2

    def test_experiment_ratio(self):
        splitter = TrafficSplitter(
            StrategyName.BASELINE,
            StrategyName.FULL_OPTIMIZATION,
            experiment_ratio=0.1,
        )

        # Run many requests, check ratio
        experiment_count = 0
        total = 10000
        for i in range(total):
            strategy = splitter.route(f"req_{i}")
            if strategy == StrategyName.FULL_OPTIMIZATION:
                experiment_count += 1

        # Should be close to 10%
        ratio = experiment_count / total
        assert 0.08 < ratio < 0.12

    def test_route_multi(self):
        splitter = TrafficSplitter(StrategyName.BASELINE, StrategyName.FULL_OPTIMIZATION)

        strategies = [StrategyName.BASELINE, StrategyName.SHADOWKV_ONLY, StrategyName.FFN_ONLY]
        weights = [0.5, 0.3, 0.2]

        results = {}
        for i in range(10000):
            s = splitter.route_multi(f"req_{i}", strategies, weights)
            results[s] = results.get(s, 0) + 1

        # Check approximate ratios
        total = sum(results.values())
        baseline_ratio = results[StrategyName.BASELINE] / total
        assert 0.45 < baseline_ratio < 0.55


class TestStatisticalTest:
    def test_paired_t_test_identical(self):
        """Test with identical distributions: p-value should be high."""
        control = [50.0] * 100
        experiment = [50.0] * 100

        p_value = StatisticalTest.paired_t_test(control, experiment)
        assert p_value > 0.9  # Very high p-value (no difference)

    def test_paired_t_test_different(self):
        """Test with different distributions: p-value should be low."""
        control = [50.0] * 100
        experiment = [40.0] * 100

        p_value = StatisticalTest.paired_t_test(control, experiment)
        assert p_value < 0.05  # Significant difference

    def test_paired_t_test_insufficient_samples(self):
        """Test with insufficient samples."""
        p_value = StatisticalTest.paired_t_test([50.0], [40.0])
        assert p_value == 1.0  # Not enough data

    def test_is_significant(self):
        control = [50.0] * 100
        experiment = [40.0] * 100

        assert StatisticalTest.is_significant(control, experiment, alpha=0.05)
        assert not StatisticalTest.is_significant(control, experiment, alpha=0.0)

    def test_required_sample_size(self):
        """Test sample size calculation."""
        n = StatisticalTest.required_sample_size(expected_effect_size=0.2)
        assert n > 0
        assert n != float("inf")

    def test_required_sample_size_zero_effect(self):
        """Test with zero effect size."""
        n = StatisticalTest.required_sample_size(expected_effect_size=0.0)
        assert n == float("inf")


class TestABTestFramework:
    def test_initial_state(self):
        control = StrategyConfig(StrategyName.BASELINE)
        experiment = StrategyConfig(StrategyName.FULL_OPTIMIZATION)

        ab = ABTestFramework(control, experiment)

        assert ab.experiment_promoted is False
        assert ab.experiment_terminated is False

    def test_route_request(self):
        control = StrategyConfig(StrategyName.BASELINE)
        experiment = StrategyConfig(StrategyName.FULL_OPTIMIZATION)

        ab = ABTestFramework(control, experiment, experiment_ratio=0.1)

        strategy = ab.route_request("req_123")
        assert strategy in [StrategyName.BASELINE, StrategyName.FULL_OPTIMIZATION]

    def test_record_metrics(self):
        control = StrategyConfig(StrategyName.BASELINE)
        experiment = StrategyConfig(StrategyName.FULL_OPTIMIZATION)

        ab = ABTestFramework(control, experiment)

        ab.record_metrics(
            request_id="req_1",
            strategy=StrategyName.BASELINE,
            latency_ms=50.0,
            memory_mb=4000.0,
            accuracy=0.95,
            success=True,
        )

        stats = ab.stats[StrategyName.BASELINE]
        assert stats.request_count == 1
        assert stats.success_count == 1

    def test_comparison_report_structure(self):
        control = StrategyConfig(StrategyName.BASELINE)
        experiment = StrategyConfig(StrategyName.FULL_OPTIMIZATION)

        ab = ABTestFramework(control, experiment)

        # Add some data
        for i in range(100):
            ab.record_metrics(
                f"req_{i}",
                StrategyName.BASELINE,
                50.0 + i * 0.1,
                4000.0,
                0.95,
                True,
            )
            ab.record_metrics(
                f"req_{i}",
                StrategyName.FULL_OPTIMIZATION,
                40.0 + i * 0.1,
                2000.0,
                0.94,
                True,
            )

        report = ab.get_comparison_report()

        assert "control" in report
        assert "experiment" in report
        assert report["control"]["requests"] == 100
        assert report["experiment"]["requests"] == 100
        assert "statistical_tests" in report

    def test_should_not_promote_with_few_samples(self):
        control = StrategyConfig(StrategyName.BASELINE)
        experiment = StrategyConfig(StrategyName.FULL_OPTIMIZATION)

        ab = ABTestFramework(control, experiment, min_samples=100)

        # Add only 50 samples
        for i in range(50):
            ab.record_metrics(f"req_{i}", StrategyName.BASELINE, 50.0, 4000.0, 0.95, True)
            ab.record_metrics(f"req_{i}", StrategyName.FULL_OPTIMIZATION, 30.0, 2000.0, 0.95, True)

        assert not ab.should_promote_experiment()

    def test_promote_experiment(self):
        control = StrategyConfig(StrategyName.BASELINE)
        experiment = StrategyConfig(StrategyName.FULL_OPTIMIZATION)

        ab = ABTestFramework(control, experiment, min_samples=50, auto_promote=True)

        # Add enough samples with clear winner
        for i in range(100):
            ab.record_metrics(f"req_{i}", StrategyName.BASELINE, 50.0, 4000.0, 0.95, True)
            ab.record_metrics(f"req_{i}", StrategyName.FULL_OPTIMIZATION, 20.0, 1500.0, 0.95, True)

        # Should promote (lower latency + lower memory + same accuracy)
        assert ab.should_promote_experiment()

    def test_promote_and_route(self):
        control = StrategyConfig(StrategyName.BASELINE)
        experiment = StrategyConfig(StrategyName.FULL_OPTIMIZATION)

        ab = ABTestFramework(control, experiment)
        ab.experiment_promoted = True

        # All traffic should go to experiment
        for i in range(100):
            assert ab.route_request(f"req_{i}") == StrategyName.FULL_OPTIMIZATION

    def test_terminate_experiment(self):
        control = StrategyConfig(StrategyName.BASELINE)
        experiment = StrategyConfig(StrategyName.FULL_OPTIMIZATION)

        ab = ABTestFramework(control, experiment)
        ab.terminate_experiment()

        assert ab.experiment_terminated is True

        # All traffic should go to control
        for i in range(100):
            assert ab.route_request(f"req_{i}") == StrategyName.BASELINE

    def test_get_config(self):
        control = StrategyConfig(StrategyName.BASELINE)
        experiment = StrategyConfig(StrategyName.FULL_OPTIMIZATION)

        ab = ABTestFramework(control, experiment)

        config = ab.get_config_for_strategy(StrategyName.BASELINE)
        assert config.name == StrategyName.BASELINE


class TestMultiStrategyABTest:
    def test_route_request(self):
        strategies = [
            StrategyConfig(StrategyName.BASELINE),
            StrategyConfig(StrategyName.SHADOWKV_ONLY),
            StrategyConfig(StrategyName.FFN_ONLY),
        ]

        ab = MultiStrategyABTest(strategies)

        strategy = ab.route_request("req_123")
        assert strategy in [
            StrategyName.BASELINE,
            StrategyName.SHADOWKV_ONLY,
            StrategyName.FFN_ONLY,
        ]

    def test_record_and_rank(self):
        strategies = [
            StrategyConfig(StrategyName.BASELINE),
            StrategyConfig(StrategyName.SHADOWKV_ONLY),
        ]

        ab = MultiStrategyABTest(strategies)

        for i in range(50):
            ab.record_metrics(f"req_{i}", StrategyName.BASELINE, 50.0, 4000.0, 0.95, True)
            ab.record_metrics(f"req_{i}", StrategyName.SHADOWKV_ONLY, 30.0, 2000.0, 0.94, True)

        ranking = ab.get_ranking_report()

        assert len(ranking) == 2
        # SHADOWKV_ONLY should be first (lower latency)
        assert ranking[0]["strategy"] == StrategyName.SHADOWKV_ONLY.value

    def test_best_strategy(self):
        strategies = [
            StrategyConfig(StrategyName.BASELINE),
            StrategyConfig(StrategyName.SHADOWKV_ONLY),
        ]

        ab = MultiStrategyABTest(strategies)

        for i in range(50):
            ab.record_metrics(f"req_{i}", StrategyName.BASELINE, 50.0, 4000.0, 0.95, True)
            ab.record_metrics(f"req_{i}", StrategyName.SHADOWKV_ONLY, 30.0, 2000.0, 0.94, True)

        best = ab.get_best_strategy(min_samples=50)
        assert best == StrategyName.SHADOWKV_ONLY

    def test_best_strategy_insufficient_data(self):
        strategies = [StrategyConfig(StrategyName.BASELINE)]
        ab = MultiStrategyABTest(strategies)

        best = ab.get_best_strategy(min_samples=100)
        assert best is None
