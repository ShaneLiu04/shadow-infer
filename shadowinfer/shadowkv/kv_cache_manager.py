"""KV Cache Manager — KV Cache 存储管理器。

处理多 layer 的 KV 存储和检索，支持：

* 内存统计与复用率追踪
* 可选的 packed byte-level KV cache 元数据
* 基于 ``EvictionPolicy`` 的显存预算强制（long-context 场景）
* 基于 Q-drift 敏感度的预取缓冲区（prefetch buffer）

对应文档：TECHNICAL_SPEC.md §3.1, SHADOWKV_AGENT.md, ROADMAP.md §ShadowKV
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Set, Tuple

import torch

from shadowinfer.core.structs import KVCacheEntry
from shadowinfer.shadowkv.eviction_policy import EvictionPolicy, LeastImportantEvictionPolicy
from shadowinfer.shadowkv.importance_model import TokenHeadScore


class KVCacheManager:
    """KV Cache 存储管理器。

    Args:
        num_layers: 层数。
        memory_budget_bytes: 可选显存预算（字节）。超出时按 eviction_policy 回收。
        eviction_policy: 回收策略实例。默认 ``LeastImportantEvictionPolicy``。
    """

    def __init__(
        self,
        num_layers: int,
        memory_budget_bytes: Optional[int] = None,
        eviction_policy: Optional[EvictionPolicy] = None,
    ):
        self.num_layers = num_layers
        self.memory_budget_bytes = memory_budget_bytes
        self.eviction_policy = eviction_policy or LeastImportantEvictionPolicy()

        self.cache: Dict[int, KVCacheEntry] = {}  # layer_id -> KVCacheEntry
        self._packed_cache: Dict[int, Any] = {}  # layer_id -> PackedKVCache
        self._reuse_count = 0
        self._total_access = 0
        self._reuse_history: Dict[int, List[bool]] = {}  # layer_id -> [is_reused, ...]

        # Per-token-head metadata used for eviction decisions.
        self._token_head_meta: Dict[int, Dict[Tuple[int, int], TokenHeadScore]] = {}

        # Prefetch buffer keyed by (layer_id, predicted_step_id) or
        # (layer_id, "next") for the immediate next step.
        self._prefetch_buffer: Dict[Tuple[int, str], KVCacheEntry] = {}

    # ------------------------------------------------------------------
    # Core store / retrieve
    # ------------------------------------------------------------------

    def store(self, layer_id: int, entry: KVCacheEntry) -> None:
        """存储 KV cache entry。

        如果配置了显存预算，存储后会调用回收策略强制预算。
        """
        self.cache[layer_id] = entry
        if entry.packed_kv is not None:
            self._packed_cache[layer_id] = entry.packed_kv
        if layer_id not in self._reuse_history:
            self._reuse_history[layer_id] = []
        self._reuse_history[layer_id].append(entry.is_reused)
        self._total_access += 1
        if entry.is_reused:
            self._reuse_count += 1

        self._enforce_budget(layer_id)

    def retrieve(self, layer_id: int) -> Optional[KVCacheEntry]:
        """检索 KV cache entry。"""
        return self.cache.get(layer_id, None)

    def store_packed(self, layer_id: int, packed_kv: Any) -> None:
        """存储打包后的 KV cache。"""
        if isinstance(packed_kv, dict):
            packed_kv = packed_kv.get("packed_kv", packed_kv)
        self._packed_cache[layer_id] = packed_kv

    def retrieve_packed(self, layer_id: int) -> Optional[Any]:
        """检索打包后的 KV cache。"""
        return self._packed_cache.get(layer_id, None)

    # ------------------------------------------------------------------
    # Token-head metadata & eviction
    # ------------------------------------------------------------------

    def set_token_head_meta(
        self, layer_id: int, meta: Dict[Tuple[int, int], TokenHeadScore]
    ) -> None:
        """Store per-token-head metadata for eviction decisions."""
        self._token_head_meta[layer_id] = meta

    def get_token_head_meta(self, layer_id: int) -> Dict[Tuple[int, int], TokenHeadScore]:
        """Return stored per-token-head metadata for a layer."""
        return dict(self._token_head_meta.get(layer_id, {}))

    def _enforce_budget(self, layer_id: int) -> None:
        """Evict token-heads if the cache exceeds the configured budget."""
        if self.memory_budget_bytes is None or self.memory_budget_bytes <= 0:
            return

        current_usage = self.get_memory_usage()
        if current_usage <= self.memory_budget_bytes:
            return

        entry = self.cache.get(layer_id)
        if entry is None or entry.packed_kv is None:
            return

        meta = self._token_head_meta.get(layer_id, {})
        if not meta:
            return

        packed = entry.packed_kv
        seq_len = packed.seq_len
        num_heads = packed.num_heads
        head_dim = packed.head_dim
        batch_size = packed.batch_size
        # Rough per-token-head byte estimate (mixed precision average).
        estimated_entry_bytes = batch_size * head_dim  # ~1 byte/value on average

        candidates = list(meta.values())
        to_evict = self.eviction_policy.select_for_eviction(
            candidates,
            budget_bytes=self.memory_budget_bytes,
            current_usage_bytes=current_usage,
            estimated_entry_bytes=estimated_entry_bytes,
        )

        if to_evict:
            packed.evict(to_evict)
            # Mark evicted entries in metadata.
            for _, token_idx, head_idx in to_evict:
                key = (token_idx, head_idx)
                if key in meta:
                    meta[key].importance = -1.0

    # ------------------------------------------------------------------
    # Prefetch buffer
    # ------------------------------------------------------------------

    def store_prefetched(
        self, layer_id: int, tag: str, entry: KVCacheEntry
    ) -> None:
        """Store a prefetched KV entry for a future step."""
        self._prefetch_buffer[(layer_id, tag)] = entry

    def retrieve_prefetched(self, layer_id: int, tag: str) -> Optional[KVCacheEntry]:
        """Retrieve a prefetched KV entry."""
        return self._prefetch_buffer.get((layer_id, tag), None)

    def clear_prefetch_buffer(self, tag: Optional[str] = None) -> None:
        """Clear prefetched entries. If ``tag`` is None, clear all."""
        if tag is None:
            self._prefetch_buffer.clear()
            return
        keys = [k for k in self._prefetch_buffer if k[1] == tag]
        for key in keys:
            del self._prefetch_buffer[key]

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def clear(self) -> None:
        """清空所有缓存。"""
        self.cache.clear()
        self._packed_cache.clear()
        self._reuse_count = 0
        self._total_access = 0
        self._reuse_history.clear()
        self._token_head_meta.clear()
        self._prefetch_buffer.clear()

    # ------------------------------------------------------------------
    # Statistics
    # ------------------------------------------------------------------

    def get_memory_usage(self) -> int:
        """获取总内存使用量（bytes）。"""
        total_bytes = 0
        for layer_id, entry in self.cache.items():
            packed = self._packed_cache.get(layer_id)
            if packed is not None and hasattr(packed, "memory_bytes"):
                total_bytes += packed.memory_bytes()
            else:
                total_bytes += self._tensor_bytes(entry.k_tensor)
                total_bytes += self._tensor_bytes(entry.v_tensor)
                if entry.scale_k is not None:
                    total_bytes += self._tensor_bytes(entry.scale_k)
                if entry.scale_v is not None:
                    total_bytes += self._tensor_bytes(entry.scale_v)
        return total_bytes

    def get_memory_usage_mb(self) -> float:
        """获取总内存使用量（MB）。"""
        return self.get_memory_usage() / (1024.0 * 1024.0)

    def get_reuse_stats(self) -> Dict[str, float]:
        """获取复用统计。"""
        stats: Dict[str, float] = {"global_reuse_rate": 0.0}
        if self._total_access > 0:
            stats["global_reuse_rate"] = self._reuse_count / self._total_access

        for layer_id, history in self._reuse_history.items():
            if history:
                stats[f"layer_{layer_id}_reuse_rate"] = sum(history) / len(history)
            else:
                stats[f"layer_{layer_id}_reuse_rate"] = 0.0

        return stats

    @staticmethod
    def _tensor_bytes(tensor: torch.Tensor) -> int:
        """计算 tensor 占用的字节数。"""
        return tensor.numel() * tensor.element_size()

    def __repr__(self) -> str:
        return (
            f"KVCacheManager(num_layers={self.num_layers}, "
            f"cached_layers={len(self.cache)}, "
            f"memory_mb={self.get_memory_usage_mb():.2f}, "
            f"budget_mb={self.memory_budget_bytes / (1024.0 * 1024.0) if self.memory_budget_bytes else None})"
        )
