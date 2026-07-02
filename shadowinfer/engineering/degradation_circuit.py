"""
ShadowInfer Engineering: Degradation & Circuit Breaker Strategies
===============================================================

Production-grade degradation strategies for AI inference systems.

Features:
- Circuit Breaker: Trip on consecutive failures, auto-recovery with half-open
- Graceful Degradation: Fallback to simpler algorithms when resources constrained
- Rate Limiter: Token bucket for request throttling
- Health Monitor: Multi-dimensional health checks
- Alert Integration: Automatic alerting on degradation

Target: Big Tech AI Infra (SRE/DevOps best practices)
"""

import logging
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


class CircuitState(Enum):
    """熔断器状态"""

    CLOSED = "closed"  # 正常：请求通过
    OPEN = "open"  # 熔断：请求拒绝
    HALF_OPEN = "half_open"  # 半开：试探性请求


class DegradationLevel(Enum):
    """降级等级"""

    NONE = 0  # 无降级
    LIGHT = 1  # 轻度：减少 profiling 频率
    MODERATE = 2  # 中度：关闭非关键 agent
    SEVERE = 3  # 重度：最小功能模式
    EMERGENCY = 4  # 紧急：仅保留核心推理


@dataclass
class HealthMetric:
    """健康指标"""

    name: str
    value: float
    threshold: float
    direction: str  # "above" or "below" (value 相对于 threshold 的健康方向)
    timestamp: float = field(default_factory=time.time)

    def is_healthy(self) -> bool:
        if self.direction == "above":
            return self.value >= self.threshold
        else:
            return self.value <= self.threshold


@dataclass
class DegradationAction:
    """降级动作记录"""

    timestamp: float
    level: DegradationLevel
    reason: str
    actions_taken: List[str]
    metrics_at_trigger: Dict[str, float]


class CircuitBreaker:
    """
    熔断器（Circuit Breaker）。

    基于 Martin Fowler 的 Circuit Breaker 模式：
    - CLOSED: 正常服务，记录失败次数
    - OPEN: 失败次数超过阈值，拒绝请求，进入冷却期
    - HALF_OPEN: 冷却期结束，允许试探性请求

    状态转换：
    CLOSED --(failures >= threshold)--> OPEN --(timeout)--> HALF_OPEN --(success)--> CLOSED
                                            --(failure)--> OPEN

    Usage:
        cb = CircuitBreaker(failure_threshold=5, recovery_timeout=30.0)

        @cb.protect
        def inference_step(...) -> Tensor:
            # 如果熔断器 OPEN，自动抛出 CircuitBreakerOpen 异常
            ...

        # 手动记录成功/失败
        cb.record_success()
        cb.record_failure()
    """

    def __init__(
        self,
        failure_threshold: int = 5,
        recovery_timeout_sec: float = 30.0,
        half_open_max_calls: int = 3,
        name: str = "default",
    ):
        self.name = name
        self.failure_threshold = failure_threshold
        self.recovery_timeout_sec = recovery_timeout_sec
        self.half_open_max_calls = half_open_max_calls

        self._state = CircuitState.CLOSED
        self._failure_count = 0
        self._success_count = 0
        self._last_failure_time = 0.0
        self._half_open_calls = 0

        self._lock = threading.RLock()
        self._state_history: deque = deque(maxlen=100)

        self._on_state_change: Optional[Callable[[CircuitState, CircuitState], None]] = None

    def on_state_change(self, callback: Callable[[CircuitState, CircuitState], None]) -> None:
        """注册状态变更回调 (old_state, new_state)"""
        self._on_state_change = callback

    def _transition_to(self, new_state: CircuitState) -> None:
        """状态转换（带回调通知）"""
        with self._lock:
            old_state = self._state
            if old_state != new_state:
                self._state = new_state
                self._state_history.append((time.time(), old_state, new_state))
                logger.warning(
                    f"CircuitBreaker[{self.name}]: {old_state.value} -> {new_state.value}"
                )

                if self._on_state_change:
                    try:
                        self._on_state_change(old_state, new_state)
                    except Exception as e:
                        logger.warning(f"State change callback error: {e}")

    def can_execute(self) -> bool:
        """检查是否允许执行请求"""
        with self._lock:
            if self._state == CircuitState.CLOSED:
                return True

            elif self._state == CircuitState.OPEN:
                # 检查是否过了冷却期
                if time.time() - self._last_failure_time >= self.recovery_timeout_sec:
                    self._transition_to(CircuitState.HALF_OPEN)
                    self._half_open_calls = 0
                    return True
                return False

            elif self._state == CircuitState.HALF_OPEN:
                if self._half_open_calls < self.half_open_max_calls:
                    self._half_open_calls += 1
                    return True
                return False

            return False

    def record_success(self) -> None:
        """记录成功请求"""
        with self._lock:
            if self._state == CircuitState.HALF_OPEN:
                self._success_count += 1
                if self._success_count >= self.half_open_max_calls:
                    self._failure_count = 0
                    self._success_count = 0
                    self._transition_to(CircuitState.CLOSED)
            elif self._state == CircuitState.CLOSED:
                self._failure_count = 0  # 连续成功，重置失败计数

    def record_failure(self) -> None:
        """记录失败请求"""
        with self._lock:
            self._failure_count += 1
            self._last_failure_time = time.time()

            if self._state == CircuitState.CLOSED:
                if self._failure_count >= self.failure_threshold:
                    self._transition_to(CircuitState.OPEN)

            elif self._state == CircuitState.HALF_OPEN:
                self._transition_to(CircuitState.OPEN)

    def get_state(self) -> CircuitState:
        """获取当前状态"""
        with self._lock:
            return self._state

    def get_state_history(self) -> List[Tuple[float, CircuitState, CircuitState]]:
        """获取状态历史"""
        with self._lock:
            return list(self._state_history)

    def protect(self, func: Callable) -> Callable:
        """装饰器：保护函数免受熔断器影响"""

        def wrapper(*args, **kwargs):
            if not self.can_execute():
                raise CircuitBreakerOpen(f"CircuitBreaker[{self.name}] is OPEN")

            try:
                result = func(*args, **kwargs)
                self.record_success()
                return result
            except Exception:
                self.record_failure()
                raise

        return wrapper


class CircuitBreakerOpen(Exception):
    """熔断器打开异常"""


class GracefulDegradation:
    """
    优雅降级管理器。

    根据系统健康状态自动调整优化策略：
    - NONE: 全部优化开启
    - LIGHT: 减少 profiling 频率，降低 observability 开销
    - MODERATE: 关闭 ShadowKV 复用（保留量化），关闭 FFN 稀疏更新
    - SEVERE: 仅保留 ShadowKV 量化，关闭 Q-drift 调度
    - EMERGENCY: 全部优化关闭，仅保留核心推理

    触发条件：
    - 延迟持续超过阈值
    - 显存持续超过阈值
    - 连续 step 失败
    - 精度下降超过阈值
    """

    # 降级配置映射
    DEGRADATION_CONFIGS = {
        DegradationLevel.NONE: {
            "shadowkv.enabled": True,
            "shadowkv.reuse.enabled": True,
            "qdrift.enabled": True,
            "ffn.enabled": True,
            "ffn.sparse_update": True,
            "profiling.interval": 1,
            "observability.enabled": True,
        },
        DegradationLevel.LIGHT: {
            "shadowkv.enabled": True,
            "shadowkv.reuse.enabled": True,
            "qdrift.enabled": True,
            "ffn.enabled": True,
            "ffn.sparse_update": True,
            "profiling.interval": 5,  # 降低 profiling 频率
            "observability.enabled": True,
        },
        DegradationLevel.MODERATE: {
            "shadowkv.enabled": True,
            "shadowkv.reuse.enabled": False,  # 关闭复用
            "qdrift.enabled": True,
            "ffn.enabled": True,
            "ffn.sparse_update": False,  # 关闭稀疏更新
            "profiling.interval": 10,
            "observability.enabled": True,
        },
        DegradationLevel.SEVERE: {
            "shadowkv.enabled": True,
            "shadowkv.reuse.enabled": False,
            "qdrift.enabled": False,  # 关闭 Q-drift
            "ffn.enabled": True,
            "ffn.sparse_update": False,
            "profiling.interval": 20,
            "observability.enabled": False,  # 关闭可观测性
        },
        DegradationLevel.EMERGENCY: {
            "shadowkv.enabled": False,  # 全部关闭
            "shadowkv.reuse.enabled": False,
            "qdrift.enabled": False,
            "ffn.enabled": False,
            "ffn.sparse_update": False,
            "profiling.interval": 0,
            "observability.enabled": False,
        },
    }

    def __init__(
        self,
        latency_threshold_ms: float = 100.0,
        memory_threshold_mb: float = 7000.0,
        accuracy_drop_threshold: float = 0.02,
        consecutive_failures_threshold: int = 3,
        auto_degrade: bool = True,
    ):
        self.latency_threshold_ms = latency_threshold_ms
        self.memory_threshold_mb = memory_threshold_mb
        self.accuracy_drop_threshold = accuracy_drop_threshold
        self.consecutive_failures_threshold = consecutive_failures_threshold
        self.auto_degrade = auto_degrade

        self._current_level = DegradationLevel.NONE
        self._consecutive_failures = 0
        self._health_metrics: deque = deque(maxlen=100)
        self._actions_history: deque = deque(maxlen=50)

        self._lock = threading.RLock()
        self._on_degrade: Optional[Callable[[DegradationLevel, DegradationLevel, str], None]] = None

    def on_degrade(
        self, callback: Callable[[DegradationLevel, DegradationLevel, str], None]
    ) -> None:
        """注册降级回调 (old_level, new_level, reason)"""
        self._on_degrade = callback

    def report_metric(self, metric: HealthMetric) -> None:
        """报告健康指标"""
        with self._lock:
            self._health_metrics.append(metric)

            if self.auto_degrade:
                self._evaluate_degradation()

    def report_latency(self, latency_ms: float) -> None:
        """报告延迟指标"""
        self.report_metric(
            HealthMetric(
                name="latency_ms",
                value=latency_ms,
                threshold=self.latency_threshold_ms,
                direction="below",
            )
        )

    def report_memory(self, memory_mb: float) -> None:
        """报告显存指标"""
        self.report_metric(
            HealthMetric(
                name="memory_mb",
                value=memory_mb,
                threshold=self.memory_threshold_mb,
                direction="below",
            )
        )

    def report_accuracy_drop(self, drop: float) -> None:
        """报告精度下降指标"""
        self.report_metric(
            HealthMetric(
                name="accuracy_drop",
                value=drop,
                threshold=self.accuracy_drop_threshold,
                direction="below",
            )
        )

    def report_failure(self) -> None:
        """报告失败"""
        with self._lock:
            self._consecutive_failures += 1
            if self._consecutive_failures >= self.consecutive_failures_threshold:
                self._trigger_degradation(
                    DegradationLevel.EMERGENCY, f"{self._consecutive_failures} consecutive failures"
                )

    def report_success(self) -> None:
        """报告成功（重置失败计数）"""
        with self._lock:
            self._consecutive_failures = 0

    def _evaluate_degradation(self) -> None:
        """评估是否需要降级"""
        # 获取最近指标
        recent_metrics = list(self._health_metrics)
        if len(recent_metrics) < 5:
            return

        # 计算最近 5 个 step 的平均值
        recent = recent_metrics[-5:]

        latency_avg = sum(m.value for m in recent if m.name == "latency_ms") / max(
            1, sum(1 for m in recent if m.name == "latency_ms")
        )
        memory_avg = sum(m.value for m in recent if m.name == "memory_mb") / max(
            1, sum(1 for m in recent if m.name == "memory_mb")
        )
        accuracy_avg = sum(m.value for m in recent if m.name == "accuracy_drop") / max(
            1, sum(1 for m in recent if m.name == "accuracy_drop")
        )

        # 决定降级等级
        new_level = self._current_level
        reasons = []

        if latency_avg > self.latency_threshold_ms * 2:
            new_level = DegradationLevel(max(new_level.value, DegradationLevel.SEVERE.value))
            reasons.append(f"Latency {latency_avg:.1f}ms > {self.latency_threshold_ms * 2}ms")
        elif latency_avg > self.latency_threshold_ms * 1.5:
            new_level = DegradationLevel(max(new_level.value, DegradationLevel.MODERATE.value))
            reasons.append(f"Latency {latency_avg:.1f}ms > {self.latency_threshold_ms * 1.5}ms")
        elif latency_avg > self.latency_threshold_ms:
            new_level = DegradationLevel(max(new_level.value, DegradationLevel.LIGHT.value))
            reasons.append(f"Latency {latency_avg:.1f}ms > {self.latency_threshold_ms}ms")

        if memory_avg > self.memory_threshold_mb:
            new_level = DegradationLevel(max(new_level.value, DegradationLevel.MODERATE.value))
            reasons.append(f"Memory {memory_avg:.1f}MB > {self.memory_threshold_mb}MB")

        if accuracy_avg > self.accuracy_drop_threshold:
            new_level = DegradationLevel(max(new_level.value, DegradationLevel.MODERATE.value))
            reasons.append(f"Accuracy drop {accuracy_avg:.3f} > {self.accuracy_drop_threshold}")

        # 如果所有指标恢复正常，考虑升级
        if (
            latency_avg < self.latency_threshold_ms * 0.8
            and memory_avg < self.memory_threshold_mb * 0.8
            and accuracy_avg < self.accuracy_drop_threshold * 0.5
        ):
            if self._consecutive_failures == 0:
                new_level = DegradationLevel(max(0, new_level.value - 1))

        if new_level != self._current_level and reasons:
            self._trigger_degradation(new_level, "; ".join(reasons))

    def _trigger_degradation(self, new_level: DegradationLevel, reason: str) -> None:
        """触发降级"""
        with self._lock:
            old_level = self._current_level
            if old_level == new_level:
                return

            self._current_level = new_level

            # 记录降级动作
            config = self.DEGRADATION_CONFIGS.get(new_level, {})
            actions = [f"{k}={v}" for k, v in config.items()]

            metrics_at_trigger = {}
            for m in list(self._health_metrics)[-5:]:
                metrics_at_trigger[m.name] = m.value

            action = DegradationAction(
                timestamp=time.time(),
                level=new_level,
                reason=reason,
                actions_taken=actions,
                metrics_at_trigger=metrics_at_trigger,
            )
            self._actions_history.append(action)

            logger.warning(f"Degradation: {old_level.name} -> {new_level.name} | Reason: {reason}")

            if self._on_degrade:
                try:
                    self._on_degrade(old_level, new_level, reason)
                except Exception as e:
                    logger.warning(f"Degradation callback error: {e}")

    def get_current_level(self) -> DegradationLevel:
        """获取当前降级等级"""
        return self._current_level

    def get_current_config(self) -> Dict[str, Any]:
        """获取当前降级配置"""
        return self.DEGRADATION_CONFIGS.get(self._current_level, {}).copy()

    def get_actions_history(self) -> List[DegradationAction]:
        """获取降级历史"""
        with self._lock:
            return list(self._actions_history)

    def force_level(self, level: DegradationLevel, reason: str = "manual") -> None:
        """手动强制降级等级"""
        self._trigger_degradation(level, reason)

    def reset(self) -> None:
        """重置到无降级状态"""
        self._trigger_degradation(DegradationLevel.NONE, "manual reset")
        self._consecutive_failures = 0


class TokenBucketRateLimiter:
    """
    Token Bucket 限流器。

    用于控制请求速率，防止突发流量压垮系统。

    Usage:
        limiter = TokenBucketRateLimiter(rate=10.0, burst=20)

        if limiter.acquire():
            process_request()
        else:
            reject_request()  # 或排队等待
    """

    def __init__(self, rate: float, burst: int):
        """
        Args:
            rate: 每秒补充的 token 数
            burst: 桶容量（最大突发请求数）
        """
        self.rate = rate
        self.burst = burst
        self._tokens = burst
        self._last_update = time.time()
        self._lock = threading.Lock()

    def acquire(self, tokens: int = 1) -> bool:
        """
        尝试获取 token。

        Returns:
            True if tokens acquired, False otherwise
        """
        with self._lock:
            now = time.time()
            elapsed = now - self._last_update
            self._tokens = min(self.burst, self._tokens + elapsed * self.rate)
            self._last_update = now

            if self._tokens >= tokens:
                self._tokens -= tokens
                return True
            return False

    def get_available_tokens(self) -> float:
        """获取当前可用 token 数"""
        with self._lock:
            now = time.time()
            elapsed = now - self._last_update
            return min(self.burst, self._tokens + elapsed * self.rate)

    def wait_time(self, tokens: int = 1) -> float:
        """计算等待 token 可用的时间"""
        with self._lock:
            if self._tokens >= tokens:
                return 0.0
            needed = tokens - self._tokens
            return needed / self.rate


class HealthMonitor:
    """
    健康监控器。

    多维度健康检查，支持自定义检查项。
    """

    def __init__(self, check_interval_sec: float = 5.0):
        self.check_interval_sec = check_interval_sec
        self._checks: Dict[str, Callable[[], HealthMetric]] = {}
        self._results: Dict[str, List[HealthMetric]] = {}
        self._running = False
        self._thread: Optional[threading.Thread] = None

    def register_check(self, name: str, check_fn: Callable[[], HealthMetric]) -> None:
        """注册健康检查项"""
        self._checks[name] = check_fn
        self._results[name] = []

    def start(self) -> None:
        """启动监控"""
        self._running = True
        self._thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        """停止监控"""
        self._running = False
        if self._thread:
            self._thread.join(timeout=2.0)

    def _monitor_loop(self) -> None:
        """监控循环"""
        while self._running:
            for name, check_fn in self._checks.items():
                try:
                    metric = check_fn()
                    self._results[name].append(metric)

                    # 只保留最近 100 条
                    if len(self._results[name]) > 100:
                        self._results[name] = self._results[name][-100:]

                    if not metric.is_healthy():
                        logger.warning(
                            f"Health check failed: {name}={metric.value:.2f} "
                            f"(threshold={metric.threshold})"
                        )

                except Exception as e:
                    logger.error(f"Health check error for {name}: {e}")

            time.sleep(self.check_interval_sec)

    def get_health_status(self) -> Dict[str, Any]:
        """获取整体健康状态"""
        status = {}
        all_healthy = True

        for name, results in self._results.items():
            if results:
                latest = results[-1]
                status[name] = {
                    "healthy": latest.is_healthy(),
                    "value": latest.value,
                    "threshold": latest.threshold,
                    "timestamp": latest.timestamp,
                }
                if not latest.is_healthy():
                    all_healthy = False

        return {
            "overall_healthy": all_healthy,
            "checks": status,
            "timestamp": time.time(),
        }

    def get_check_history(self, name: str, limit: int = 20) -> List[HealthMetric]:
        """获取指定检查项的历史"""
        return self._results.get(name, [])[-limit:]


class ProductionSafetyNet:
    """
    生产安全网（Production Safety Net）。

    整合所有工程化保障机制：
    - Circuit Breaker: 防止级联故障
    - Graceful Degradation: 资源不足时自动降级
    - Rate Limiter: 防止突发流量
    - Health Monitor: 持续健康监控

    一键集成到 Orchestrator：
        safety_net = ProductionSafetyNet()
        orchestrator.enable_safety_net(safety_net)
    """

    def __init__(
        self,
        circuit_breaker: Optional[CircuitBreaker] = None,
        degradation: Optional[GracefulDegradation] = None,
        rate_limiter: Optional[TokenBucketRateLimiter] = None,
        health_monitor: Optional[HealthMonitor] = None,
    ):
        self.circuit_breaker = circuit_breaker or CircuitBreaker(
            failure_threshold=5, recovery_timeout_sec=30.0, name="inference"
        )
        self.degradation = degradation or GracefulDegradation()
        self.rate_limiter = rate_limiter or TokenBucketRateLimiter(rate=100.0, burst=200)
        self.health_monitor = health_monitor or HealthMonitor()

        self._enabled = False

    def enable(self) -> None:
        """启用安全网"""
        self._enabled = True
        self.health_monitor.start()
        logger.info("Production Safety Net enabled")

    def disable(self) -> None:
        """禁用安全网"""
        self._enabled = False
        self.health_monitor.stop()
        logger.info("Production Safety Net disabled")

    def check_before_inference(self) -> Tuple[bool, Optional[str]]:
        """
        推理前检查。

        Returns:
            (can_proceed, reason_if_not)
        """
        if not self._enabled:
            return True, None

        # 1. 熔断器检查
        if not self.circuit_breaker.can_execute():
            return False, f"Circuit breaker OPEN (state={self.circuit_breaker.get_state().value})"

        # 2. 限流检查
        if not self.rate_limiter.acquire():
            return (
                False,
                f"Rate limit exceeded (available={self.rate_limiter.get_available_tokens():.1f})",
            )

        # 3. 降级检查（仅日志警告，不阻止）
        level = self.degradation.get_current_level()
        if level != DegradationLevel.NONE:
            logger.warning(f"Running in degradation mode: {level.name}")

        return True, None

    def report_after_inference(
        self, latency_ms: float, memory_mb: float, accuracy_drop: float, success: bool
    ) -> None:
        """推理后报告"""
        if not self._enabled:
            return

        if success:
            self.circuit_breaker.record_success()
            self.degradation.report_success()
        else:
            self.circuit_breaker.record_failure()
            self.degradation.report_failure()

        self.degradation.report_latency(latency_ms)
        self.degradation.report_memory(memory_mb)
        self.degradation.report_accuracy_drop(accuracy_drop)

    def pre_flight_check(
        self,
        step_id: Optional[int] = None,
        step_config: Optional[Any] = None,
    ) -> Tuple[bool, Optional[str]]:
        """推理前检查（Orchestrator 集成接口）。

        Args:
            step_id: 当前 step 编号。
            step_config: 当前 step 配置。

        Returns:
            (can_proceed, reason_if_not)
        """
        return self.check_before_inference()

    def post_flight_check(
        self,
        latency_ms: float = 0.0,
        memory_mb: float = 0.0,
        accuracy_drop: float = 0.0,
        success: bool = True,
    ) -> None:
        """推理后报告（Orchestrator 集成接口）。

        Args:
            latency_ms: 当前 step 延迟（毫秒）。
            memory_mb: 当前 step 显存使用（MB）。
            accuracy_drop: 当前 step 精度下降比例。
            success: 推理是否成功。
        """
        self.report_after_inference(latency_ms, memory_mb, accuracy_drop, success)

    def get_status(self) -> Dict[str, Any]:
        """获取安全网状态"""
        return {
            "enabled": self._enabled,
            "circuit_breaker": {
                "state": self.circuit_breaker.get_state().value,
                "failure_count": getattr(self.circuit_breaker, "_failure_count", 0),
            },
            "degradation": {
                "current_level": self.degradation.get_current_level().name,
            },
            "rate_limiter": {
                "available_tokens": self.rate_limiter.get_available_tokens(),
            },
            "health": self.health_monitor.get_health_status(),
        }
