"""ShadowKV Decision Plane — formalize Importance / Precision / Reuse axes.

The decision plane maps every ``(token, head)`` into a 3-D coordinate:

* **Importance** (0..1): attention-based saliency of the token-head.
* **Drift** (0..1): relative L2 change of KV values vs. the previous step.
* **Memory Pressure** (0..1): current cache size vs. configured budget.

From this coordinate the plane derives three orthogonal decisions:

* **Precision**: FP32 / FP16 / INT8 / INT4 storage precision.
* **Reuse**: whether the token-head can be copied from the previous step.
* **Eviction**: whether the token-head should be dropped under memory pressure.

对应文档：SHADOWKV_AGENT.md, TECHNICAL_SPEC.md §2.1, ROADMAP.md §ShadowKV 2.0
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Set, Tuple

import torch


@dataclass(frozen=True)
class KVPlaneCoordinate:
    """A single point on the Importance / Drift / Memory-Pressure plane."""

    importance: float
    drift: float
    memory_pressure: float

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "importance", float(max(0.0, min(1.0, self.importance)))
        )
        object.__setattr__(self, "drift", float(max(0.0, min(1.0, self.drift))))
        object.__setattr__(
            self, "memory_pressure", float(max(0.0, min(1.0, self.memory_pressure)))
        )


@dataclass(frozen=True)
class KVPlaneDecision:
    """Per-token-head decision emitted by the decision plane."""

    precision: str
    should_reuse: bool
    should_evict: bool
    eviction_priority: float


class KVDecisionPlane:
    """Joint decision plane for ShadowKV precision, reuse, and eviction.

    Args:
        importance_thresholds: Mapping from precision name to minimum importance.
            Defaults match ShadowKVAgent: ``{"fp32": 0.8, "fp16": 0.5,
            "int8": 0.2, "int4": 0.0}``.
        reuse_base_threshold: Base relative-L2 reuse threshold.
        mode_adjustment: Mode offsets for precision thresholds.
            aggressive=-0.1, balanced=0.0, conservative=0.1.
        memory_budget_bytes: Optional absolute memory budget. When ``None``,
            eviction is disabled.
    """

    PRECISION_ORDER = ("fp32", "fp16", "int8", "int4")

    def __init__(
        self,
        importance_thresholds: Optional[Dict[str, float]] = None,
        reuse_base_threshold: float = 0.15,
        mode_adjustment: Optional[Dict[str, float]] = None,
        memory_budget_bytes: Optional[int] = None,
    ):
        self.importance_thresholds = importance_thresholds or {
            "fp32": 0.8,
            "fp16": 0.5,
            "int8": 0.2,
            "int4": 0.0,
        }
        self.reuse_base_threshold = reuse_base_threshold
        self.mode_adjustment = mode_adjustment or {
            "aggressive": -0.1,
            "balanced": 0.0,
            "conservative": 0.1,
        }
        self.memory_budget_bytes = memory_budget_bytes

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def decide(self, coord: KVPlaneCoordinate, mode: str = "balanced") -> KVPlaneDecision:
        """Return precision/reuse/eviction decision for a single coordinate."""
        precision = self.allocate_precision(coord.importance, mode)
        should_reuse = self.decide_reuse(coord, mode)
        should_evict, eviction_priority = self.decide_eviction(coord)
        return KVPlaneDecision(
            precision=precision,
            should_reuse=should_reuse,
            should_evict=should_evict,
            eviction_priority=eviction_priority,
        )

    def decide_all(
        self,
        importance_map: torch.Tensor,
        drift_map: torch.Tensor,
        memory_pressure: float,
        mode: str = "balanced",
    ) -> Dict[Tuple[int, int], KVPlaneDecision]:
        """Compute a decision for every ``(token_idx, head_idx)``.

        Args:
            importance_map: ``[seq_len, num_heads]`` tensor.
            drift_map: ``[seq_len, num_heads]`` tensor of per-token-head drift.
            memory_pressure: scalar in ``[0, 1]``.
            mode: ShadowKV mode.

        Returns:
            Dictionary mapping ``(token_idx, head_idx)`` to ``KVPlaneDecision``.
        """
        seq_len, num_heads = importance_map.shape
        decisions: Dict[Tuple[int, int], KVPlaneDecision] = {}
        for token_idx in range(seq_len):
            for head_idx in range(num_heads):
                coord = KVPlaneCoordinate(
                    importance=float(importance_map[token_idx, head_idx].item()),
                    drift=float(drift_map[token_idx, head_idx].item()),
                    memory_pressure=memory_pressure,
                )
                decisions[(token_idx, head_idx)] = self.decide(coord, mode=mode)
        return decisions

    def allocate_precision(self, importance: float, mode: str = "balanced") -> str:
        """Map an importance score to a precision string.

        Mirrors the allocation logic in ``ShadowKVAgent.allocate_precision``
        so that existing callers get identical results.
        """
        adjustment = self.mode_adjustment.get(mode, 0.0)
        thresholds = {
            key: max(0.0, value + adjustment)
            for key, value in self.importance_thresholds.items()
        }

        if importance >= thresholds["fp32"]:
            return "fp32"
        if importance >= thresholds["fp16"]:
            return "fp16"
        if importance >= thresholds["int8"]:
            return "int8"
        return "int4"

    def decide_reuse(self, coord: KVPlaneCoordinate, mode: str = "balanced") -> bool:
        """Return whether a token-head with ``coord`` can be reused.

        The effective reuse threshold is lowered by high importance (important
        heads must see smaller drift to be reused) and by high drift.
        """
        base = {"aggressive": 0.20, "balanced": 0.15, "conservative": 0.10}.get(
            mode, 0.15
        )
        # Sensitivity to current drift: the higher the drift, the tighter reuse.
        effective = base * (1.0 - 0.3 * coord.drift)
        # High-importance heads are held to a stricter standard.
        effective *= 1.0 - 0.3 * coord.importance
        effective = max(effective, 1e-5)
        return coord.drift < effective

    def decide_eviction(
        self, coord: KVPlaneCoordinate
    ) -> Tuple[bool, float]:
        """Return ``(should_evict, eviction_priority)``.

        Eviction is only enabled when ``memory_budget_bytes`` is set.
        The eviction priority is high for low-importance, low-drift entries
        (they contribute little and can be recomputed cheaply).
        """
        if self.memory_budget_bytes is None or self.memory_budget_bytes <= 0:
            return False, 0.0

        # Only evict when memory pressure is non-trivial.
        if coord.memory_pressure < 0.5:
            return False, 0.0

        # Priority: low importance + low drift are the safest to evict.
        priority = (1.0 - coord.importance) * (1.0 - coord.drift)
        threshold = 0.3 + 0.5 * coord.memory_pressure
        should_evict = priority > threshold
        return should_evict, priority

    def split_decisions(
        self,
        decisions: Dict[Tuple[int, int], KVPlaneDecision],
    ) -> Tuple[
        Dict[int, Dict[int, str]],
        Dict[Tuple[int, int], bool],
        Set[Tuple[int, int]],
    ]:
        """Split a decision map into precision_map, reuse_mask, and evict_set."""
        precision_map: Dict[int, Dict[int, str]] = {}
        reuse_mask: Dict[Tuple[int, int], bool] = {}
        evict_set: Set[Tuple[int, int]] = set()

        for (token_idx, head_idx), decision in decisions.items():
            precision_map.setdefault(token_idx, {})[head_idx] = decision.precision
            reuse_mask[(token_idx, head_idx)] = decision.should_reuse
            if decision.should_evict:
                evict_set.add((token_idx, head_idx))

        return precision_map, reuse_mask, evict_set

    # ------------------------------------------------------------------
    # Helpers that mirror the original ShadowKVAgent math
    # ------------------------------------------------------------------

    @staticmethod
    def compute_drift_map(
        kv_current_k: torch.Tensor,
        kv_current_v: torch.Tensor,
        kv_previous_k: torch.Tensor,
        kv_previous_v: torch.Tensor,
    ) -> torch.Tensor:
        """Compute per-token-head relative L2 drift.

        Returns:
            ``[seq_len, num_heads]`` tensor.
        """
        diff_k = kv_current_k - kv_previous_k
        diff_v = kv_current_v - kv_previous_v
        norm_diff_k = torch.norm(diff_k, dim=(0, 3))
        norm_diff_v = torch.norm(diff_v, dim=(0, 3))
        norm_cur_k = torch.norm(kv_current_k, dim=(0, 3)) + 1e-8
        norm_cur_v = torch.norm(kv_current_v, dim=(0, 3)) + 1e-8
        delta_k = norm_diff_k / norm_cur_k
        delta_v = norm_diff_v / norm_cur_v
        return ((delta_k + delta_v) / 2.0).detach().cpu().T  # [seq_len, num_heads]

    @staticmethod
    def compute_memory_pressure(
        current_bytes: int, budget_bytes: Optional[int]
    ) -> float:
        """Return memory pressure in ``[0, 1]``.

        Returns 0.0 when no budget is configured.
        """
        if budget_bytes is None or budget_bytes <= 0:
            return 0.0
        return float(min(1.0, current_bytes / max(1, budget_bytes)))

    @staticmethod
    def compute_importance_score(
        attention_scores: torch.Tensor,
        token_index: int,
        head_index: int,
        layer_index: int,
        num_layers: int,
    ) -> float:
        """Re-implementation of the original per-token-head importance formula.

        Kept here so the plane can be used independently of ``ShadowKVAgent``.
        """
        from shadowinfer.utils.metrics import Metrics

        attn_weights = attention_scores[:, head_index, token_index, :]
        weights = torch.nn.functional.softmax(attn_weights, dim=-1)
        entropy = Metrics.compute_entropy(weights)
        normalized_entropy = entropy / math.log(max(weights.shape[-1], 2))

        layer_factor = 1.0 + 0.1 * (layer_index / max(num_layers, 1))
        seq_len = weights.shape[-1]
        pos_factor = 1.2 if token_index < 5 or token_index > seq_len - 5 else 1.0

        score = normalized_entropy * layer_factor * pos_factor
        return float(min(score, 1.0))
