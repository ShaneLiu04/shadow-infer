"""ShadowInfer production serving scaffold.

Provides a small, dependency-light HTTP server that exposes ``/generate``,
``/health`` and ``/metrics`` endpoints over any
:class:`shadowinfer.core.model_backend.ModelBackend`.  The implementation uses
only the Python standard library so that it works on Windows/CPU-only hosts
where vLLM/SGLang are unavailable.

对应文档：ROADMAP.md §3.6 Production & Serving
"""

from shadowinfer.serving.backend import ServingBackend
from shadowinfer.serving.config import ServingConfig
from shadowinfer.serving.limiters import ConcurrencyLimiter, TokenBucketRateLimiter
from shadowinfer.serving.metrics import ServingMetrics
from shadowinfer.serving.server import create_server, serve_forever

__all__ = [
    "ConcurrencyLimiter",
    "ServingBackend",
    "ServingConfig",
    "ServingMetrics",
    "TokenBucketRateLimiter",
    "create_server",
    "serve_forever",
]
