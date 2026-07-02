"""ShadowKV Agent — KV Cache 压缩与复用优化模块。

对应文档：SHADOWKV_AGENT.md, TECHNICAL_SPEC.md §2.1 / §3.2
版本：v3.1
"""

from __future__ import annotations

__version__ = "3.1"

from shadowinfer.shadowkv.decision_plane import KVDecisionPlane, KVPlaneCoordinate, KVPlaneDecision
from shadowinfer.shadowkv.eviction_policy import (
    EvictionPolicy,
    ImportanceAgeEvictionPolicy,
    LeastImportantEvictionPolicy,
)
from shadowinfer.shadowkv.importance_model import ImportanceModel, TokenHeadScore
from shadowinfer.shadowkv.kv_cache_manager import KVCacheManager
from shadowinfer.shadowkv.packed_kv_cache import PackedKVCache
from shadowinfer.shadowkv.shadowkv_agent import ShadowKVAgent

__all__ = [
    "ShadowKVAgent",
    "KVCacheManager",
    "PackedKVCache",
    "ImportanceModel",
    "TokenHeadScore",
    "KVDecisionPlane",
    "KVPlaneCoordinate",
    "KVPlaneDecision",
    "EvictionPolicy",
    "LeastImportantEvictionPolicy",
    "ImportanceAgeEvictionPolicy",
]
