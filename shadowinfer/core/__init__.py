"""ShadowInfer Core Module.

Version: 3.0
Corresponds to: TECHNICAL_SPEC.md v2.0
"""

__version__ = "3.0"
__doc_version__ = "TECHNICAL_SPEC.md v2.0"

from .base_agent import AgentRegistry, BaseAgent
from .bus import MESSAGE_TYPES, ProfilingBus
from .config import Config, load_config
from .model_backend import MockModelBackend, ModelBackend
from .structs import (
    AgentState,
    ErrorRecord,
    KVCacheEntry,
    Message,
    PipelineContext,
    ProfileResult,
    StepConfig,
    StepState,
    StepStats,
)

__all__ = [
    "BaseAgent",
    "AgentRegistry",
    "ProfilingBus",
    "MESSAGE_TYPES",
    "AgentState",
    "ErrorRecord",
    "KVCacheEntry",
    "Message",
    "PipelineContext",
    "ProfileResult",
    "StepConfig",
    "StepState",
    "StepStats",
    "Config",
    "load_config",
    "ModelBackend",
    "MockModelBackend",
]
