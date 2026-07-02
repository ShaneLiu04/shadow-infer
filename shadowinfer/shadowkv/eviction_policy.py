"""Eviction policies for ShadowKV long-context memory management.

When a ``KVCacheManager`` is configured with a ``memory_budget_bytes`` limit,
an ``EvictionPolicy`` decides which ``(layer, token, head)`` entries to drop.

对应文档：SHADOWKV_AGENT.md, TECHNICAL_SPEC.md §2.1, ROADMAP.md §Importance-aware eviction
"""

from __future__ import annotations

import abc
from typing import List, Set, Tuple

from shadowinfer.shadowkv.importance_model import TokenHeadScore


class EvictionPolicy(abc.ABC):
    """Abstract base for KV cache eviction policies."""

    @abc.abstractmethod
    def select_for_eviction(
        self,
        entries: List[TokenHeadScore],
        budget_bytes: int,
        current_usage_bytes: int,
        estimated_entry_bytes: int,
    ) -> Set[Tuple[int, int, int]]:
        """Return a set of ``(layer_id, token_idx, head_idx)`` to evict.

        Args:
            entries: All candidate token-head metadata records.
            budget_bytes: Configured memory budget.
            current_usage_bytes: Current total cache size in bytes.
            estimated_entry_bytes: Estimated bytes per token-head.

        Returns:
            Set of token-head coordinates to evict.
        """


class LeastImportantEvictionPolicy(EvictionPolicy):
    """Evict the lowest-importance token-heads first."""

    def select_for_eviction(
        self,
        entries: List[TokenHeadScore],
        budget_bytes: int,
        current_usage_bytes: int,
        estimated_entry_bytes: int,
    ) -> Set[Tuple[int, int, int]]:
        if current_usage_bytes <= budget_bytes or not entries:
            return set()

        target_reduction = current_usage_bytes - budget_bytes
        num_to_evict = min(
            len(entries),
            max(1, int(target_reduction // max(1, estimated_entry_bytes))),
        )

        sorted_entries = sorted(entries, key=lambda e: e.importance)
        evicted: Set[Tuple[int, int, int]] = set()
        for entry in sorted_entries[:num_to_evict]:
            evicted.add((entry.layer_id, entry.token_idx, entry.head_idx))
        return evicted


class ImportanceAgeEvictionPolicy(EvictionPolicy):
    """Evict token-heads with low ``importance * decay(age)`` scores.

    Args:
        age_decay: Factor controlling how fast old entries lose protection.
            Higher values make age more punitive.
    """

    def __init__(self, age_decay: float = 0.01):
        self.age_decay = age_decay

    def _score(self, entry: TokenHeadScore) -> float:
        # Lower score -> more likely to be evicted.
        return entry.importance * (1.0 - self.age_decay) ** entry.age

    def select_for_eviction(
        self,
        entries: List[TokenHeadScore],
        budget_bytes: int,
        current_usage_bytes: int,
        estimated_entry_bytes: int,
    ) -> Set[Tuple[int, int, int]]:
        if current_usage_bytes <= budget_bytes or not entries:
            return set()

        target_reduction = current_usage_bytes - budget_bytes
        num_to_evict = min(
            len(entries),
            max(1, int(target_reduction // max(1, estimated_entry_bytes))),
        )

        sorted_entries = sorted(entries, key=self._score)
        evicted: Set[Tuple[int, int, int]] = set()
        for entry in sorted_entries[:num_to_evict]:
            evicted.add((entry.layer_id, entry.token_idx, entry.head_idx))
        return evicted
