"""KV Cache Manager — KV Cache 存储管理器。

处理多 layer 的 KV 存储和检索，支持：

* 内存统计与复用率追踪
* 可选的 packed byte-level KV cache 元数据
* 基于 ``EvictionPolicy`` 的显存预算强制（long-context 场景）
* 基于 Q-drift 敏感度的预取缓冲区（prefetch buffer）

对应文档：README.md《核心模块》《核心算法》章节
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

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
        # AR003 T005：张量级重要性矩阵（[seq_k, heads]），生产驱逐路径使用
        self._importance_matrices: Dict[int, torch.Tensor] = {}

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

    def set_importance_matrix(self, layer_id: int, importance: torch.Tensor) -> None:
        """存储张量级重要性矩阵（AR003 T005，FR3.4）。

        生产路径替代逐对象 ``set_token_head_meta``（对象级 API 保留兼容）。
        形状与既有层记录不一致时以最新为准并告警。
        """
        previous = self._importance_matrices.get(layer_id)
        if previous is not None and tuple(previous.shape) != tuple(importance.shape):
            import logging

            logging.getLogger("kv_cache_manager").warning(
                "importance matrix shape changed for layer %s: %s -> %s",
                layer_id,
                tuple(previous.shape),
                tuple(importance.shape),
            )
        self._importance_matrices[layer_id] = importance.detach().cpu()

    def get_importance_matrix(self, layer_id: int) -> Optional[torch.Tensor]:
        """返回层的重要性矩阵；未存储层返回 None。"""
        return self._importance_matrices.get(layer_id)

    def enforce_global_budget(self, budget_bytes: int, sink_tokens: int = 0) -> int:
        """跨层全局驱逐（AR003 T005）：按重要性升序驱逐非 sink token-head。

        目标用量为 ``budget_bytes * 0.9``（10% 迟滞）。驱逐通过
        ``PackedKVCache.evict`` 完成（降档至 int4 并清零），属于罕见路径，
        允许 O(全量) 重打包。sink 前缀（token_idx < sink_tokens）永不驱逐。

        Returns:
            实际驱逐的 token-head 条数；无 packed 层或无候选时为 0。
        """
        if budget_bytes is None or budget_bytes <= 0:
            return 0
        if not self._packed_cache:
            return 0

        target = int(budget_bytes * 0.9)
        evicted_total = 0
        evicted: set = set()

        candidates: List[Tuple[float, int, int, int]] = []
        for layer_id, matrix in self._importance_matrices.items():
            if layer_id not in self._packed_cache:
                continue
            values = matrix.detach().cpu()
            seq_len, num_heads = values.shape
            for token_idx in range(seq_len):
                if token_idx < sink_tokens:
                    continue
                for head_idx in range(num_heads):
                    candidates.append(
                        (float(values[token_idx, head_idx]), layer_id, token_idx, head_idx)
                    )
        candidates.sort(key=lambda item: item[0])

        idx = 0
        while self.get_memory_usage() > target and idx < len(candidates):
            chunk: Dict[int, set] = {}
            for _, layer_id, token_idx, head_idx in candidates[idx : idx + 64]:
                if (layer_id, token_idx, head_idx) in evicted:
                    continue
                chunk.setdefault(layer_id, set()).add((token_idx, head_idx))
            idx += 64
            if not chunk:
                continue
            for layer_id, token_heads in chunk.items():
                packed = self._packed_cache.get(layer_id)
                if packed is None:
                    continue
                packed.evict(token_heads)
                for token_idx, head_idx in token_heads:
                    evicted.add((layer_id, token_idx, head_idx))
                    evicted_total += 1

        if self.get_memory_usage() > target and idx >= len(candidates):
            import logging

            logging.getLogger("kv_cache_manager").warning(
                "global budget enforcement reached floor: usage=%s target=%s",
                self.get_memory_usage(),
                target,
            )
        return evicted_total

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
        # AR003 T005（KV-P10）：per-precision 字节估算修正——以实际 packed 字节
        # 均摊到每条 token-head（含 scale），替代旧 batch*head_dim（≈1B/值，
        # fp32 下低估 8x）
        seq_len = getattr(packed, "seq_len", 0) or 0
        num_heads = getattr(packed, "num_heads", 0) or 0
        total_entries = max(1, seq_len * num_heads)
        if hasattr(packed, "memory_bytes"):
            estimated_entry_bytes = max(1, int(packed.memory_bytes()) // total_entries)
        else:
            batch_size = getattr(packed, "batch_size", 1) or 1
            head_dim = getattr(packed, "head_dim", 1) or 1
            estimated_entry_bytes = batch_size * head_dim

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

    def store_prefetched(self, layer_id: int, tag: str, entry: KVCacheEntry) -> None:
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
        self._importance_matrices.clear()
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
        budget_mb = (
            self.memory_budget_bytes / (1024.0 * 1024.0) if self.memory_budget_bytes else None
        )
        return (
            f"KVCacheManager(num_layers={self.num_layers}, "
            f"cached_layers={len(self.cache)}, "
            f"memory_mb={self.get_memory_usage_mb():.2f}, "
            f"budget_mb={budget_mb})"
        )
