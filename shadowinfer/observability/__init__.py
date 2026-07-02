"""ShadowInfer 可观测性模块。

对应大厂 AI Infra 招聘竞争力：
- Metrics (Prometheus 风格)
- Traces (OpenTelemetry 风格)
- Dashboard (实时可视化)

版本：v3.0
"""

__version__ = "3.0"

from .dashboard import DashboardData
from .metrics import (
    _PROMETHEUS_AVAILABLE,
    Counter,
    Gauge,
    Histogram,
    MetricsRegistry,
    Summary,
)
from .tracing import (
    _OTEL_AVAILABLE,
    CriticalPath,
    Span,
    Tracer,
    TracerProvider,
)

__all__ = [
    "Counter",
    "Gauge",
    "Histogram",
    "Summary",
    "MetricsRegistry",
    "Span",
    "Tracer",
    "CriticalPath",
    "TracerProvider",
    "DashboardData",
    "_PROMETHEUS_AVAILABLE",
    "_OTEL_AVAILABLE",
    "__version__",
]
