"""Serving metrics using the ShadowInfer observability backend.

对应文档：ROADMAP.md §3.6 Production & Serving
"""

from __future__ import annotations

import time
from typing import Any, Dict, Optional

from shadowinfer.observability.metrics import MetricsRegistry


class ServingMetrics:
    """Metrics collector for the HTTP serving layer."""

    def __init__(self, enabled: bool = True, registry: Optional[MetricsRegistry] = None) -> None:
        self.enabled = enabled
        self.registry = registry or MetricsRegistry()
        if not self.enabled:
            return

        self.requests_total = self.registry.counter(
            "shadowinfer_serving_requests_total", "累计 serving 请求数"
        )
        self.requests_failed = self.registry.counter(
            "shadowinfer_serving_requests_failed_total", "累计 serving 失败请求数"
        )
        self.requests_rate_limited = self.registry.counter(
            "shadowinfer_serving_requests_rate_limited_total", "累计被限流请求数"
        )
        self.requests_concurrency_limited = self.registry.counter(
            "shadowinfer_serving_requests_concurrency_limited_total", "累计并发超限请求数"
        )
        self.latency_histogram = self.registry.histogram(
            "shadowinfer_serving_latency_ms", "serving 端到端延迟分布（ms）"
        )
        self.concurrency_gauge = self.registry.gauge(
            "shadowinfer_serving_concurrency", "当前 serving 并发请求数"
        )

    def record_request(
        self,
        status: str,
        latency_ms: float,
        limited_by: Optional[str] = None,
    ) -> None:
        """Record a completed request."""
        if not self.enabled:
            return
        self.requests_total.inc()
        if latency_ms >= 0:
            self.latency_histogram.observe(latency_ms)
        if status.startswith("5"):
            self.requests_failed.inc()
        if limited_by == "rate":
            self.requests_rate_limited.inc()
        elif limited_by == "concurrency":
            self.requests_concurrency_limited.inc()

    def set_concurrency(self, value: int) -> None:
        if self.enabled:
            self.concurrency_gauge.set(float(value))

    def time_request(self) -> Any:
        """Context manager to time a request block."""
        return _ServingLatencyContext(self)

    def expose(self) -> str:
        """Expose metrics in Prometheus exposition format."""
        return self.registry.expose()


class _ServingLatencyContext:
    def __init__(self, metrics: ServingMetrics) -> None:
        self.metrics = metrics
        self.start = 0.0
        self.latency_ms = 0.0

    def __enter__(self) -> "_ServingLatencyContext":
        self.start = time.perf_counter()
        return self

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        self.latency_ms = (time.perf_counter() - self.start) * 1000.0
