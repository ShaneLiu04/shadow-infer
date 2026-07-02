"""ShadowInfer — Edge-optimized Diffusion LLM inference and profiling framework.

Docs: README.md, ARCHITECTURE.md, AGENTS.md, TECHNICAL_SPEC.md, ENGINEERING.md
Version: v3.1
"""

__version__ = "3.2.2"

from shadowinfer.core import (
    AgentRegistry,
    AgentState,
    BaseAgent,
    Config,
    ErrorRecord,
    KVCacheEntry,
    Message,
    ProfileResult,
    ProfilingBus,
    StepConfig,
    StepStats,
    load_config,
)
from shadowinfer.observability import (
    Counter,
    DashboardData,
    Gauge,
    Histogram,
    MetricsRegistry,
    Span,
    Summary,
    Tracer,
)
from shadowinfer.orchestrator import InferenceResult, Orchestrator

__all__ = [
    "BaseAgent",
    "AgentRegistry",
    "ProfilingBus",
    "KVCacheEntry",
    "StepConfig",
    "ProfileResult",
    "AgentState",
    "Message",
    "StepStats",
    "ErrorRecord",
    "Config",
    "load_config",
    "Orchestrator",
    "InferenceResult",
    "Counter",
    "Gauge",
    "Histogram",
    "Summary",
    "MetricsRegistry",
    "Span",
    "Tracer",
    "DashboardData",
]
