"""
Tests for ShadowInfer Engineering module.

Covers: Hot Config, Circuit Breaker, Degradation, Type Safety, Rate Limiter
"""

import threading
import time

import pytest
import torch

from shadowinfer.engineering import (
    AgentConfigAdapter,
    BudgetExceededError,
    CircuitBreaker,
    CircuitBreakerOpen,
    CircuitState,
    ConfigSchemaValidator,
    DegradationLevel,
    DtypeError,
    GracefulDegradation,
    HealthMetric,
    HealthMonitor,
    HotConfigReloader,
    PerformanceBudget,
    ProductionSafetyNet,
    ReloadStatus,
    SafeInferenceContext,
    ShapeError,
    TensorValidator,
    TokenBucketRateLimiter,
    WeightHealthChecker,
)

# =============================================================================
# Hot Config Reloader Tests
# =============================================================================


class TestHotConfigReloader:

    def test_initial_load(self, tmp_path):
        """测试初始配置加载"""
        config_path = tmp_path / "test_config.yaml"
        config_path.write_text("""
optimization:
  enabled: true
shadowkv:
  enabled: true
  compression_ratio: 0.5
qdrift:
  enabled: true
ffn:
  enabled: true
constraints:
  max_accuracy_drop: 0.01
""")

        reloader = HotConfigReloader(str(config_path))
        assert reloader.get_status() == ReloadStatus.IDLE
        assert reloader.get_config_hash() != ""
        config = reloader.get_current_config()
        assert config is not None

    def test_validation_pass(self, tmp_path):
        """测试配置验证通过"""
        config_path = tmp_path / "test_config.yaml"
        config_path.write_text("""
optimization:
  enabled: true
shadowkv:
  enabled: true
  compression_ratio: 0.5
  precision_levels: ["fp32", "fp16", "int8"]
qdrift:
  enabled: true
  noise_schedule: "cosine"
  sensitivity_temperature: 1.0
ffn:
  enabled: true
  delta_threshold: 0.05
constraints:
  max_accuracy_drop: 0.01
""")

        reloader = HotConfigReloader(str(config_path))
        errors = reloader.validate_config(reloader._current_raw)
        assert len(errors) == 0

    def test_validation_fail_compression_ratio(self, tmp_path):
        """测试验证失败：压缩比超出范围"""
        config_path = tmp_path / "test_config.yaml"
        config_path.write_text("""
optimization:
  enabled: true
shadowkv:
  compression_ratio: 1.5
qdrift:
  sensitivity_temperature: 1.0
ffn:
  delta_threshold: 0.05
constraints:
  max_accuracy_drop: 0.01
""")

        reloader = HotConfigReloader(str(config_path))
        errors = reloader.validate_config(reloader._current_raw)
        assert any("compression_ratio" in e for e in errors)

    def test_validation_fail_temperature(self, tmp_path):
        """测试验证失败：温度 <= 0"""
        config_path = tmp_path / "test_config.yaml"
        config_path.write_text("""
optimization:
  enabled: true
shadowkv:
  compression_ratio: 0.5
qdrift:
  sensitivity_temperature: -1.0
ffn:
  delta_threshold: 0.05
constraints:
  max_accuracy_drop: 0.01
""")

        reloader = HotConfigReloader(str(config_path))
        errors = reloader.validate_config(reloader._current_raw)
        assert any("sensitivity_temperature" in e for e in errors)

    def test_validation_fail_precision_level(self, tmp_path):
        """测试验证失败：非法精度级别"""
        config_path = tmp_path / "test_config.yaml"
        config_path.write_text("""
optimization:
  enabled: true
shadowkv:
  compression_ratio: 0.5
  precision_levels: ["fp32", "fp16", "int8", "int2"]
qdrift:
  sensitivity_temperature: 1.0
ffn:
  delta_threshold: 0.05
constraints:
  max_accuracy_drop: 0.01
""")

        reloader = HotConfigReloader(str(config_path))
        errors = reloader.validate_config(reloader._current_raw)
        assert any("int2" in e for e in errors)

    def test_callback_notification(self, tmp_path):
        """测试配置变更回调通知"""
        config_path = tmp_path / "test_config.yaml"
        config_path.write_text("""
optimization:
  enabled: true
shadowkv:
  compression_ratio: 0.5
qdrift:
  sensitivity_temperature: 1.0
ffn:
  delta_threshold: 0.05
constraints:
  max_accuracy_drop: 0.01
""")

        reloader = HotConfigReloader(str(config_path))
        callback_called = threading.Event()

        def callback(old_cfg, new_cfg):
            callback_called.set()

        reloader.on_config_change(callback)

        # 模拟文件变更
        config_path.write_text("""
optimization:
  enabled: true
shadowkv:
  compression_ratio: 0.6
qdrift:
  sensitivity_temperature: 1.0
ffn:
  delta_threshold: 0.05
constraints:
  max_accuracy_drop: 0.01
""")

        reloader.force_reload()
        assert callback_called.wait(timeout=1.0)

    def test_force_reload(self, tmp_path):
        """测试强制重载"""
        config_path = tmp_path / "test_config.yaml"
        config_path.write_text("""
optimization:
  enabled: true
shadowkv:
  compression_ratio: 0.5
qdrift:
  sensitivity_temperature: 1.0
ffn:
  delta_threshold: 0.05
constraints:
  max_accuracy_drop: 0.01
""")

        reloader = HotConfigReloader(str(config_path))
        old_hash = reloader.get_config_hash()

        # 修改文件
        config_path.write_text("""
optimization:
  enabled: true
shadowkv:
  compression_ratio: 0.7
qdrift:
  sensitivity_temperature: 1.0
ffn:
  delta_threshold: 0.05
constraints:
  max_accuracy_drop: 0.01
""")

        result = reloader.force_reload()
        assert result is True
        assert reloader.get_config_hash() != old_hash

    def test_events_tracking(self, tmp_path):
        """测试事件记录"""
        config_path = tmp_path / "test_config.yaml"
        config_path.write_text("""
optimization:
  enabled: true
shadowkv:
  compression_ratio: 0.5
qdrift:
  sensitivity_temperature: 1.0
ffn:
  delta_threshold: 0.05
constraints:
  max_accuracy_drop: 0.01
""")

        reloader = HotConfigReloader(str(config_path))

        # 触发成功重载
        config_path.write_text("""
optimization:
  enabled: true
shadowkv:
  compression_ratio: 0.6
qdrift:
  sensitivity_temperature: 1.0
ffn:
  delta_threshold: 0.05
constraints:
  max_accuracy_drop: 0.01
""")
        reloader.force_reload()

        events = reloader.get_events(limit=10)
        assert len(events) >= 1
        assert events[-1].status == ReloadStatus.WATCHING


class TestAgentConfigAdapter:

    def test_extract_shadowkv_config(self, tmp_path):
        """测试提取 ShadowKV 配置"""
        from shadowinfer.core.config import Config

        config_path = tmp_path / "test_config.yaml"
        config_path.write_text("""
optimization:
  enabled: true
shadowkv:
  enabled: true
  compression_ratio: 0.5
  precision_levels: ["fp32", "fp16", "int8"]
  importance_thresholds:
    fp32: 0.8
  reuse:
    enabled: true
qdrift:
  enabled: true
ffn:
  enabled: true
constraints:
  max_accuracy_drop: 0.01
""")

        config = Config.from_yaml(str(config_path))
        shadowkv_cfg = AgentConfigAdapter.extract_shadowkv_config(config)

        assert shadowkv_cfg["enabled"] is True
        assert shadowkv_cfg["compression_ratio"] == 0.5
        assert shadowkv_cfg["precision_levels"] == ["fp32", "fp16", "int8"]

    def test_extract_qdrift_config(self, tmp_path):
        """测试提取 QDrift 配置"""
        from shadowinfer.core.config import Config

        config_path = tmp_path / "test_config.yaml"
        config_path.write_text("""
optimization:
  enabled: true
shadowkv:
  enabled: true
qdrift:
  enabled: true
  noise_schedule: "cosine"
  sensitivity_temperature: 1.2
  drift_method: "relative_l2"
ffn:
  enabled: true
constraints:
  max_accuracy_drop: 0.01
""")

        config = Config.from_yaml(str(config_path))
        qdrift_cfg = AgentConfigAdapter.extract_qdrift_config(config)

        assert qdrift_cfg["noise_schedule"] == "cosine"
        assert qdrift_cfg["sensitivity_temperature"] == 1.2

    def test_extract_constraints(self, tmp_path):
        """测试提取约束配置"""
        from shadowinfer.core.config import Config

        config_path = tmp_path / "test_config.yaml"
        config_path.write_text("""
optimization:
  enabled: true
shadowkv:
  enabled: true
qdrift:
  enabled: true
ffn:
  enabled: true
constraints:
  max_accuracy_drop: 0.01
  max_latency_ms: 100
  max_memory_mb: 4096
""")

        config = Config.from_yaml(str(config_path))
        constraints = AgentConfigAdapter.extract_constraints(config)

        assert constraints["max_accuracy_drop"] == 0.01
        assert constraints["max_latency_ms"] == 100
        assert constraints["max_memory_mb"] == 4096


# =============================================================================
# Circuit Breaker Tests
# =============================================================================


class TestCircuitBreaker:

    def test_initial_state(self):
        """测试初始状态为 CLOSED"""
        cb = CircuitBreaker(failure_threshold=3, recovery_timeout_sec=1.0)
        assert cb.get_state() == CircuitState.CLOSED

    def test_closed_to_open(self):
        """测试 CLOSED -> OPEN 转换"""
        cb = CircuitBreaker(failure_threshold=3, recovery_timeout_sec=1.0)

        cb.record_failure()
        cb.record_failure()
        assert cb.get_state() == CircuitState.CLOSED

        cb.record_failure()
        assert cb.get_state() == CircuitState.OPEN

    def test_open_blocks_execution(self):
        """测试 OPEN 状态阻止执行"""
        cb = CircuitBreaker(failure_threshold=1, recovery_timeout_sec=10.0)
        cb.record_failure()

        assert cb.get_state() == CircuitState.OPEN
        assert cb.can_execute() is False

    def test_open_to_half_open(self):
        """测试 OPEN -> HALF_OPEN 转换"""
        cb = CircuitBreaker(failure_threshold=1, recovery_timeout_sec=0.1)
        cb.record_failure()
        assert cb.get_state() == CircuitState.OPEN

        time.sleep(0.15)
        assert cb.can_execute() is True  # 冷却期结束，进入 HALF_OPEN
        assert cb.get_state() == CircuitState.HALF_OPEN

    def test_half_open_to_closed(self):
        """测试 HALF_OPEN -> CLOSED 转换"""
        cb = CircuitBreaker(failure_threshold=1, recovery_timeout_sec=0.1, half_open_max_calls=2)
        cb.record_failure()
        time.sleep(0.15)

        # HALF_OPEN 状态
        assert cb.can_execute() is True
        cb.record_success()
        cb.record_success()

        assert cb.get_state() == CircuitState.CLOSED

    def test_half_open_back_to_open(self):
        """测试 HALF_OPEN 失败回到 OPEN"""
        cb = CircuitBreaker(failure_threshold=1, recovery_timeout_sec=0.1)
        cb.record_failure()
        time.sleep(0.15)

        assert cb.can_execute() is True
        cb.record_failure()

        assert cb.get_state() == CircuitState.OPEN

    def test_success_resets_failures(self):
        """测试成功重置失败计数"""
        cb = CircuitBreaker(failure_threshold=3, recovery_timeout_sec=1.0)

        cb.record_failure()
        cb.record_failure()
        cb.record_success()  # 重置失败计数
        cb.record_failure()

        assert cb.get_state() == CircuitState.CLOSED  # 仍然只有 1 次失败

    def test_protect_decorator(self):
        """测试 protect 装饰器"""
        cb = CircuitBreaker(failure_threshold=1, recovery_timeout_sec=10.0)

        @cb.protect
        def reliable_function():
            return "success"

        @cb.protect
        def failing_function():
            raise ValueError("error")

        assert reliable_function() == "success"

        with pytest.raises(ValueError):
            failing_function()

        # 熔断器应该打开
        assert cb.get_state() == CircuitState.OPEN

        with pytest.raises(CircuitBreakerOpen):
            reliable_function()

    def test_state_change_callback(self):
        """测试状态变更回调"""
        cb = CircuitBreaker(failure_threshold=1, recovery_timeout_sec=0.1)

        transitions = []

        def on_change(old, new):
            transitions.append((old, new))

        cb.on_state_change(on_change)
        cb.record_failure()

        assert len(transitions) == 1
        assert transitions[0] == (CircuitState.CLOSED, CircuitState.OPEN)


# =============================================================================
# Graceful Degradation Tests
# =============================================================================


class TestGracefulDegradation:

    def test_initial_level(self):
        """测试初始降级等级为 NONE"""
        gd = GracefulDegradation()
        assert gd.get_current_level() == DegradationLevel.NONE

    def test_latency_light_degradation(self):
        """测试延迟触发轻度降级"""
        gd = GracefulDegradation(latency_threshold_ms=100.0, auto_degrade=True)

        # 模拟延迟超标
        for _ in range(5):
            gd.report_latency(150.0)  # > 100ms

        assert gd.get_current_level() == DegradationLevel.LIGHT

    def test_latency_moderate_degradation(self):
        """测试延迟触发中度降级"""
        gd = GracefulDegradation(latency_threshold_ms=100.0, auto_degrade=True)

        for _ in range(5):
            gd.report_latency(180.0)  # > 150ms

        assert gd.get_current_level() == DegradationLevel.MODERATE

    def test_memory_degradation(self):
        """测试显存触发降级"""
        gd = GracefulDegradation(memory_threshold_mb=7000.0, auto_degrade=True)

        for _ in range(5):
            gd.report_memory(7500.0)  # > 7000MB

        assert gd.get_current_level() == DegradationLevel.MODERATE

    def test_accuracy_degradation(self):
        """测试精度下降触发降级"""
        gd = GracefulDegradation(accuracy_drop_threshold=0.01, auto_degrade=True)

        for _ in range(5):
            gd.report_accuracy_drop(0.03)  # > 0.01

        assert gd.get_current_level() == DegradationLevel.MODERATE

    def test_consecutive_failures(self):
        """测试连续失败触发紧急降级"""
        gd = GracefulDegradation(consecutive_failures_threshold=3, auto_degrade=True)

        gd.report_failure()
        gd.report_failure()
        assert gd.get_current_level() == DegradationLevel.NONE

        gd.report_failure()
        assert gd.get_current_level() == DegradationLevel.EMERGENCY

    def test_success_resets_failures(self):
        """测试成功重置连续失败计数"""
        gd = GracefulDegradation(consecutive_failures_threshold=3, auto_degrade=True)

        gd.report_failure()
        gd.report_failure()
        gd.report_success()
        gd.report_failure()

        assert gd.get_current_level() == DegradationLevel.NONE

    def test_manual_force_level(self):
        """测试手动强制降级"""
        gd = GracefulDegradation()

        gd.force_level(DegradationLevel.SEVERE, "manual test")
        assert gd.get_current_level() == DegradationLevel.SEVERE

    def test_reset(self):
        """测试重置"""
        gd = GracefulDegradation()
        gd.force_level(DegradationLevel.MODERATE, "test")

        gd.reset()
        assert gd.get_current_level() == DegradationLevel.NONE

    def test_degradation_config(self):
        """测试降级配置映射"""
        gd = GracefulDegradation()

        config_none = gd.DEGRADATION_CONFIGS[DegradationLevel.NONE]
        assert config_none["shadowkv.enabled"] is True
        assert config_none["shadowkv.reuse.enabled"] is True

        config_severe = gd.DEGRADATION_CONFIGS[DegradationLevel.SEVERE]
        assert config_severe["shadowkv.reuse.enabled"] is False
        assert config_severe["qdrift.enabled"] is False

        config_emergency = gd.DEGRADATION_CONFIGS[DegradationLevel.EMERGENCY]
        assert config_emergency["shadowkv.enabled"] is False

    def test_degradation_callback(self):
        """测试降级回调"""
        gd = GracefulDegradation()

        callback_called = threading.Event()

        def on_degrade(old, new, reason):
            callback_called.set()

        gd.on_degrade(on_degrade)
        gd.force_level(DegradationLevel.LIGHT, "test")

        assert callback_called.wait(timeout=1.0)

    def test_actions_history(self):
        """测试降级动作历史"""
        gd = GracefulDegradation()

        gd.force_level(DegradationLevel.LIGHT, "reason1")
        gd.force_level(DegradationLevel.MODERATE, "reason2")

        history = gd.get_actions_history()
        assert len(history) == 2
        assert history[0].level == DegradationLevel.LIGHT
        assert history[1].level == DegradationLevel.MODERATE


# =============================================================================
# Token Bucket Rate Limiter Tests
# =============================================================================


class TestTokenBucketRateLimiter:

    def test_acquire_within_bucket(self):
        """测试桶内获取 token"""
        limiter = TokenBucketRateLimiter(rate=10.0, burst=5)

        # 初始 burst=5，可以获取 5 个
        for _ in range(5):
            assert limiter.acquire() is True

        # 第 6 个应该失败
        assert limiter.acquire() is False

    def test_token_refill(self):
        """测试 token 补充"""
        limiter = TokenBucketRateLimiter(rate=10.0, burst=5)

        # 消耗所有 token
        for _ in range(5):
            limiter.acquire()

        assert limiter.acquire() is False

        # 等待补充
        time.sleep(0.3)  # 3 个 token
        assert limiter.acquire() is True
        assert limiter.acquire() is True
        assert limiter.acquire() is True
        assert limiter.acquire() is False

    def test_wait_time(self):
        """测试等待时间计算"""
        limiter = TokenBucketRateLimiter(rate=10.0, burst=5)

        for _ in range(5):
            limiter.acquire()

        wait = limiter.wait_time(tokens=1)
        assert wait > 0.0

        wait = limiter.wait_time(tokens=10)
        assert wait > 0  # 更多 token 需要更久


# =============================================================================
# Health Monitor Tests
# =============================================================================


class TestHealthMonitor:

    def test_register_check(self):
        """测试注册健康检查"""
        monitor = HealthMonitor(check_interval_sec=0.1)

        def check_latency():
            return HealthMetric("latency", 50.0, 100.0, "below")

        monitor.register_check("latency", check_latency)
        assert "latency" in monitor._checks

    def test_health_status(self):
        """测试健康状态查询"""
        monitor = HealthMonitor(check_interval_sec=0.1)

        def check_latency():
            return HealthMetric("latency", 50.0, 100.0, "below")

        monitor.register_check("latency", check_latency)
        monitor.start()

        time.sleep(0.3)
        status = monitor.get_health_status()

        assert status["overall_healthy"] is True
        assert "latency" in status["checks"]

        monitor.stop()

    def test_unhealthy_status(self):
        """测试不健康状态"""
        monitor = HealthMonitor(check_interval_sec=0.1)

        def check_latency():
            return HealthMetric("latency", 150.0, 100.0, "below")

        monitor.register_check("latency", check_latency)
        monitor.start()

        time.sleep(0.3)
        status = monitor.get_health_status()

        assert status["overall_healthy"] is False
        assert status["checks"]["latency"]["healthy"] is False

        monitor.stop()

    def test_check_history(self):
        """测试检查历史"""
        monitor = HealthMonitor(check_interval_sec=0.1)

        def check_latency():
            return HealthMetric("latency", 50.0, 100.0, "below")

        monitor.register_check("latency", check_latency)
        monitor.start()

        time.sleep(0.5)
        history = monitor.get_check_history("latency", limit=10)

        assert len(history) > 0
        assert history[0].name == "latency"

        monitor.stop()


# =============================================================================
# Production Safety Net Tests
# =============================================================================


class TestProductionSafetyNet:

    def test_enable_disable(self):
        """测试启用和禁用"""
        safety = ProductionSafetyNet()

        safety.enable()
        assert safety._enabled is True

        safety.disable()
        assert safety._enabled is False

    def test_check_before_inference_disabled(self):
        """测试禁用时不检查"""
        safety = ProductionSafetyNet()

        can_proceed, reason = safety.check_before_inference()
        assert can_proceed is True
        assert reason is None

    def test_check_before_inference_circuit_open(self):
        """测试熔断器打开时阻止推理"""
        cb = CircuitBreaker(failure_threshold=1, recovery_timeout_sec=10.0)
        cb.record_failure()

        safety = ProductionSafetyNet(circuit_breaker=cb)
        safety.enable()

        can_proceed, reason = safety.check_before_inference()
        assert can_proceed is False
        assert "OPEN" in reason

        safety.disable()

    def test_check_before_inference_rate_limit(self):
        """测试限流"""
        limiter = TokenBucketRateLimiter(rate=1.0, burst=1)

        safety = ProductionSafetyNet(rate_limiter=limiter)
        safety.enable()

        # 第一次应该通过
        can_proceed, _ = safety.check_before_inference()
        assert can_proceed is True

        # 第二次应该被限流
        can_proceed, reason = safety.check_before_inference()
        assert can_proceed is False
        assert "Rate limit" in reason

        safety.disable()

    def test_report_after_inference(self):
        """测试推理后报告"""
        safety = ProductionSafetyNet()
        safety.enable()

        safety.report_after_inference(
            latency_ms=50.0, memory_mb=4000.0, accuracy_drop=0.005, success=True
        )

        status = safety.get_status()
        assert status["circuit_breaker"]["state"] == "closed"

        safety.disable()

    def test_report_failure_triggers_circuit(self):
        """测试失败报告触发熔断"""
        cb = CircuitBreaker(failure_threshold=1, recovery_timeout_sec=10.0)
        safety = ProductionSafetyNet(circuit_breaker=cb)
        safety.enable()

        safety.report_after_inference(
            latency_ms=50.0, memory_mb=4000.0, accuracy_drop=0.005, success=False
        )

        status = safety.get_status()
        assert status["circuit_breaker"]["state"] == "open"

        safety.disable()

    def test_status_report(self):
        """测试状态报告"""
        safety = ProductionSafetyNet()
        safety.enable()

        status = safety.get_status()
        assert "enabled" in status
        assert "circuit_breaker" in status
        assert "degradation" in status
        assert "rate_limiter" in status
        assert "health" in status

        safety.disable()


# =============================================================================
# Tensor Validator Tests
# =============================================================================


class TestTensorValidator:

    def test_validate_shape_correct(self):
        """测试形状验证通过"""
        tensor = torch.randn(1, 32, 128, 128)
        TensorValidator.validate_shape(tensor, [1, 32, 128, 128], name="test")

    def test_validate_shape_wrong_dims(self):
        """测试维度错误"""
        tensor = torch.randn(1, 32, 128)
        with pytest.raises(ShapeError):
            TensorValidator.validate_shape(tensor, [1, 32, 128, 128], name="test")

    def test_validate_shape_wrong_size(self):
        """测试尺寸错误"""
        tensor = torch.randn(1, 32, 128, 64)
        with pytest.raises(ShapeError):
            TensorValidator.validate_shape(tensor, [1, 32, 128, 128], name="test")

    def test_validate_dtype_correct(self):
        """测试 dtype 验证通过"""
        tensor = torch.randn(1, 32, 128, 128, dtype=torch.float32)
        TensorValidator.validate_dtype(tensor, [torch.float32], name="test")

    def test_validate_dtype_wrong(self):
        """测试 dtype 错误"""
        tensor = torch.randn(1, 32, 128, 128, dtype=torch.float64)
        with pytest.raises(DtypeError):
            TensorValidator.validate_dtype(tensor, [torch.float32], name="test")

    def test_validate_range_nan(self):
        """测试 NaN 检测"""
        tensor = torch.tensor([1.0, float("nan"), 3.0])
        with pytest.raises(ValueError):
            TensorValidator.validate_range(tensor, name="test")

    def test_validate_range_inf(self):
        """测试 Inf 检测"""
        tensor = torch.tensor([1.0, float("inf"), 3.0])
        with pytest.raises(ValueError):
            TensorValidator.validate_range(tensor, name="test")

    def test_validate_range_bounds(self):
        """测试范围边界"""
        tensor = torch.tensor([0.5, 1.0, 1.5])

        # 应该通过
        TensorValidator.validate_range(tensor, min_val=0.0, max_val=2.0, name="test")

        # 应该失败
        with pytest.raises(ValueError):
            TensorValidator.validate_range(tensor, min_val=0.0, max_val=1.0, name="test")

    def test_validate_kv_cache(self):
        """测试 KV Cache 验证"""
        k = torch.randn(1, 32, 128, 64)
        v = torch.randn(1, 32, 128, 64)

        TensorValidator.validate_kv_cache(
            k, v, expected_batch=1, expected_num_heads=32, expected_head_dim=64
        )

    def test_validate_kv_cache_mismatch(self):
        """测试 KV Cache 不匹配"""
        k = torch.randn(1, 32, 128, 64)
        v = torch.randn(1, 32, 128, 128)

        with pytest.raises(ShapeError):
            TensorValidator.validate_kv_cache(k, v)


# =============================================================================
# Performance Budget Tests
# =============================================================================


class TestPerformanceBudget:

    def test_track_step(self):
        """测试跟踪 step"""
        budget = PerformanceBudget(latency_budget_ms=200, memory_budget_mb=8192)

        budget.track_step(latency_ms=50.0, memory_mb=4000.0)
        budget.track_step(latency_ms=60.0, memory_mb=4500.0)

        assert not budget.is_exceeded()

    def test_latency_exceeded_non_strict(self):
        """测试非严格模式下延迟超限"""
        budget = PerformanceBudget(latency_budget_ms=100, memory_budget_mb=8192, strict=False)

        budget.track_step(latency_ms=150.0, memory_mb=4000.0)
        assert budget.is_latency_exceeded()

    def test_latency_exceeded_strict(self):
        """测试严格模式下延迟超限"""
        budget = PerformanceBudget(latency_budget_ms=100, memory_budget_mb=8192, strict=True)

        with pytest.raises(BudgetExceededError):
            budget.track_step(latency_ms=150.0, memory_mb=4000.0)

    def test_memory_exceeded_strict(self):
        """测试严格模式下显存超限"""
        budget = PerformanceBudget(latency_budget_ms=100, memory_budget_mb=4096, strict=True)

        with pytest.raises(BudgetExceededError):
            budget.track_step(latency_ms=50.0, memory_mb=5000.0)

    def test_get_stats(self):
        """测试统计信息"""
        budget = PerformanceBudget(latency_budget_ms=100, memory_budget_mb=8192)

        budget.track_step(latency_ms=50.0, memory_mb=4000.0)
        budget.track_step(latency_ms=60.0, memory_mb=4500.0)

        stats = budget.get_stats()
        assert stats["total_latency_ms"] == 110.0
        assert stats["avg_latency_ms"] == 55.0
        assert stats["max_latency_ms"] == 60.0
        assert stats["max_memory_mb"] == 4500.0
        assert stats["num_steps"] == 2

    def test_reset(self):
        """测试重置"""
        budget = PerformanceBudget(latency_budget_ms=100, memory_budget_mb=8192)

        budget.track_step(latency_ms=50.0, memory_mb=4000.0)
        budget.reset()

        assert budget.get_stats()["total_latency_ms"] == 0.0


# =============================================================================
# Weight Health Checker Tests
# =============================================================================


class TestWeightHealthChecker:

    def test_healthy_weights(self):
        """测试健康权重"""
        weights = {
            "layer1": torch.randn(100, 100),
            "layer2": torch.randn(100, 100),
        }

        issues = WeightHealthChecker.check_weights(weights)
        assert len(issues) == 0

    def test_nan_weights(self):
        """测试 NaN 权重"""
        weights = {
            "layer1": torch.tensor([1.0, float("nan"), 3.0]),
        }

        issues = WeightHealthChecker.check_weights(weights)
        assert any("NaN" in issue for issue in issues)

    def test_inf_weights(self):
        """测试 Inf 权重"""
        weights = {
            "layer1": torch.tensor([1.0, float("inf"), 3.0]),
        }

        issues = WeightHealthChecker.check_weights(weights)
        assert any("Inf" in issue for issue in issues)

    def test_zero_weights(self):
        """测试全零权重"""
        weights = {
            "layer1": torch.zeros(10, 10),
        }

        issues = WeightHealthChecker.check_weights(weights)
        assert any("all zeros" in issue for issue in issues)

    def test_attention_weights_healthy(self):
        """测试健康注意力权重"""
        weights = torch.softmax(torch.randn(1, 32, 128, 128), dim=-1)

        issues = WeightHealthChecker.check_attention_weights(weights)
        assert len(issues) == 0

    def test_attention_weights_negative(self):
        """测试注意力权重含负值"""
        weights = torch.randn(1, 32, 128, 128).clamp(-0.1, 1.0)

        issues = WeightHealthChecker.check_attention_weights(weights)
        assert any("negative" in issue for issue in issues)


# =============================================================================
# Config Schema Validator Tests
# =============================================================================


class TestConfigSchemaValidator:

    def test_valid_config(self):
        """测试有效配置"""
        config = {
            "optimization": {"enabled": True},
            "shadowkv": {
                "enabled": True,
                "compression_ratio": 0.5,
                "precision_levels": ["fp32", "fp16"],
            },
            "qdrift": {
                "noise_schedule": "cosine",
                "sensitivity_temperature": 1.0,
            },
            "ffn": {
                "delta_threshold": 0.05,
            },
            "constraints": {
                "max_accuracy_drop": 0.01,
            },
        }

        errors = ConfigSchemaValidator.validate(config)
        assert len(errors) == 0

    def test_invalid_compression_ratio(self):
        """测试无效压缩比"""
        config = {
            "optimization": {"enabled": True},
            "shadowkv": {
                "compression_ratio": 1.5,  # > 1.0
            },
        }

        errors = ConfigSchemaValidator.validate(config)
        assert any("compression_ratio" in e for e in errors)

    def test_invalid_noise_schedule(self):
        """测试无效噪声调度"""
        config = {
            "optimization": {"enabled": True},
            "qdrift": {
                "noise_schedule": "invalid",
            },
        }

        errors = ConfigSchemaValidator.validate(config)
        assert any("noise_schedule" in e for e in errors)

    def test_missing_required_field(self):
        """测试缺失必填字段"""
        config = {
            "shadowkv": {
                "compression_ratio": 0.5,
            }
        }

        errors = ConfigSchemaValidator.validate(config)
        assert any("optimization" in e for e in errors)

    def test_invalid_type(self):
        """测试类型错误"""
        config = {
            "optimization": {"enabled": True},
            "shadowkv": {
                "compression_ratio": "0.5",  # 应该是 float
            },
        }

        errors = ConfigSchemaValidator.validate(config)
        assert any("compression_ratio" in e for e in errors)


# =============================================================================
# Safe Inference Context Tests
# =============================================================================


class TestSafeInferenceContext:

    def test_context_manager(self):
        """测试上下文管理器"""
        with SafeInferenceContext() as ctx:
            assert ctx.is_healthy()

    def test_validate_input(self):
        """测试输入验证"""
        with SafeInferenceContext(validate_tensors=True) as ctx:
            tensor = torch.randn(1, 32, 128, 128)
            ctx.validate_input(tensor, "input", [1, 32, 128, 128])
            assert ctx.is_healthy()

    def test_validate_input_shape_error(self):
        """测试输入形状错误"""
        with SafeInferenceContext(validate_tensors=True, strict=False) as ctx:
            tensor = torch.randn(1, 32, 128)
            ctx.validate_input(tensor, "input", [1, 32, 128, 128])
            assert not ctx.is_healthy()

    def test_validate_output(self):
        """测试输出验证"""
        with SafeInferenceContext(validate_tensors=True) as ctx:
            tensor = torch.randn(1, 32, 128, 128)
            ctx.validate_output(tensor, "output", [1, 32, 128, 128])
            assert ctx.is_healthy()

    def test_track_performance(self):
        """测试性能跟踪"""
        budget = PerformanceBudget(latency_budget_ms=100, memory_budget_mb=8192)

        with SafeInferenceContext(budget=budget) as ctx:
            ctx.track_performance(latency_ms=50.0, memory_mb=4000.0)
            assert not budget.is_exceeded()

    def test_strict_mode_exception(self):
        """测试严格模式异常"""
        with pytest.raises(ShapeError):
            with SafeInferenceContext(validate_tensors=True, strict=True) as ctx:
                tensor = torch.randn(1, 32, 128)
                ctx.validate_input(tensor, "input", [1, 32, 128, 128])
                # 严格模式下应该抛出异常

    def test_non_strict_mode_catches_error(self):
        """测试非严格模式捕获错误"""
        with SafeInferenceContext(validate_tensors=True, strict=False) as ctx:
            tensor = torch.randn(1, 32, 128)
            ctx.validate_input(tensor, "input", [1, 32, 128, 128])

            errors = ctx.get_errors()
            assert len(errors) == 1
            assert "expected 4D" in errors[0]
