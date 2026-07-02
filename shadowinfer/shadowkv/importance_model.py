"""Importance model for ShadowKV.

Extracts per-token-head importance from attention scores and optionally from
historical KV drift. The model is intentionally lightweight (entropy + layer
factor + position factor) so it can run online inside the inference loop.

对应文档：SHADOWKV_AGENT.md, TECHNICAL_SPEC.md §2.1
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import torch
import torch.nn.functional as F

from shadowinfer.utils.metrics import Metrics


@dataclass
class TokenHeadScore:
    """Metadata describing a single ``(layer, token, head)`` entry."""

    layer_id: int
    token_idx: int
    head_idx: int
    importance: float
    drift: float = 0.0
    age: int = 0
    access_count: int = 1


class ImportanceModel:
    """Compute and track per-token-head importance scores.

    Args:
        num_layers: Total number of model layers.
        num_heads: Number of attention heads per layer.
        head_dim: Head dimension.
    """

    def __init__(
        self,
        num_layers: int = 32,
        num_heads: int = 32,
        head_dim: int = 128,
    ):
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.head_dim = head_dim
        self._history: Dict[int, List[torch.Tensor]] = {}

    def score(
        self,
        attention_scores: torch.Tensor,
        kv_current: Optional[Dict[str, torch.Tensor]] = None,
        kv_previous: Optional[Dict[str, torch.Tensor]] = None,
        layer_id: int = 0,
    ) -> torch.Tensor:
        """Return a ``[seq_len, num_heads]`` importance map.

        The score is based on normalized attention entropy, layer depth, and
        token position. This mirrors the original ``ShadowKVAgent`` formula so
        that behavior is unchanged when the model is swapped in.
        """
        batch_size, num_heads, seq_len, _ = attention_scores.shape
        importance_map = torch.zeros((seq_len, num_heads), device=attention_scores.device)

        for token_idx in range(seq_len):
            for head_idx in range(num_heads):
                importance_map[token_idx, head_idx] = self._compute_single(
                    attention_scores, token_idx, head_idx, layer_id
                )

        if kv_current is not None and kv_previous is not None:
            drift_map = self._compute_drift_map(
                kv_current["k"], kv_current["v"], kv_previous["k"], kv_previous["v"]
            )
            # Blend importance with inverse drift: stable heads (low drift)
            # are slightly more reusable, so importance can be discounted.
            importance_map = importance_map * (1.0 - 0.1 * drift_map.to(importance_map.device))
            importance_map = importance_map.clamp(0.0, 1.0)

        self._history.setdefault(layer_id, []).append(importance_map.detach().cpu())
        return importance_map

    def _compute_single(
        self,
        attention_scores: torch.Tensor,
        token_index: int,
        head_index: int,
        layer_index: int,
    ) -> float:
        attn_weights = attention_scores[:, head_index, token_index, :]
        weights = F.softmax(attn_weights, dim=-1)
        entropy = Metrics.compute_entropy(weights)
        normalized_entropy = entropy / math.log(max(weights.shape[-1], 2))

        layer_factor = 1.0 + 0.1 * (layer_index / max(self.num_layers, 1))
        seq_len = weights.shape[-1]
        pos_factor = 1.2 if token_index < 5 or token_index > seq_len - 5 else 1.0

        score = normalized_entropy * layer_factor * pos_factor
        return float(min(score, 1.0))

    @staticmethod
    def _compute_drift_map(
        k_cur: torch.Tensor,
        v_cur: torch.Tensor,
        k_prev: torch.Tensor,
        v_prev: torch.Tensor,
    ) -> torch.Tensor:
        diff_k = k_cur - k_prev
        diff_v = v_cur - v_prev
        norm_diff_k = torch.norm(diff_k, dim=(0, 3))
        norm_diff_v = torch.norm(diff_v, dim=(0, 3))
        norm_cur_k = torch.norm(k_cur, dim=(0, 3)) + 1e-8
        norm_cur_v = torch.norm(v_cur, dim=(0, 3)) + 1e-8
        delta_k = norm_diff_k / norm_cur_k
        delta_v = norm_diff_v / norm_cur_v
        return ((delta_k + delta_v) / 2.0).detach().cpu().T

    def build_token_head_scores(
        self,
        importance_map: torch.Tensor,
        drift_map: torch.Tensor,
        layer_id: int,
        step_id: int,
    ) -> List[TokenHeadScore]:
        """Convert tensor maps into a list of ``TokenHeadScore`` records."""
        seq_len, num_heads = importance_map.shape
        scores: List[TokenHeadScore] = []
        for token_idx in range(seq_len):
            for head_idx in range(num_heads):
                scores.append(
                    TokenHeadScore(
                        layer_id=layer_id,
                        token_idx=token_idx,
                        head_idx=head_idx,
                        importance=float(importance_map[token_idx, head_idx].item()),
                        drift=float(drift_map[token_idx, head_idx].item()),
                        age=step_id,
                        access_count=1,
                    )
                )
        return scores
