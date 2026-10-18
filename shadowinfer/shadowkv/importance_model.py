"""Importance model for ShadowKV.

Extracts per-token-head importance from attention scores and optionally from
historical KV drift. The model is intentionally lightweight so it can run
online inside the inference loop.

AR003（FR3.2/FR3.3）：重要性主信号改为列向接收注意力（received attention）
的跨步 EMA 传播（IRAP），共享核心 ``compute_irap_importance`` 为唯一公式
来源；旧的行熵主信号与位置加成已移除。

对应文档：README.md《核心模块》《核心算法》ShadowKV 章节
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F


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


def compute_irap_importance(
    weights: torch.Tensor,
    received_ema_prev: Optional[torch.Tensor] = None,
    irap_decay: float = 0.9,
    entropy_weight: float = 0.3,
    sink_tokens: int = 4,
    layer_index: int = 0,
    num_layers: int = 1,
) -> Tuple[torch.Tensor, torch.Tensor, Optional[float]]:
    """IRAP 重要性计算共享核心（design 4.1 / 4.3.2，唯一公式来源）。

    Args:
        weights: [batch, heads, seq_q, seq_k] 已完成契约归一化的概率权重。
        received_ema_prev: 上一步 EMA 状态 [seq_k, heads]；None 为冷启动
            （A = received，delta_a 返回 None）。序列增长零填充、收缩截断。
        irap_decay: EMA 衰减系数 γ ∈ (0, 1)。
        entropy_weight: 行熵辅助权重 β ∈ [0, 1]。
        sink_tokens: 前缀 sink 保护 token 数（≥0；强制最高档 1.0）。
        layer_index / num_layers: 层深度因子。

    Returns:
        (importance_map [seq_k, heads], new_ema [seq_k, heads], delta_a|None)

    Delta A 定义：``‖A_new − A_prev‖_F / (‖A_prev‖_F + 1e-12)``（跨步差分，
    供 QDrift 可选消费）。
    """
    seq_len = weights.shape[-1]
    log_seq = math.log(seq_len)
    if log_seq == 0.0:
        raise ZeroDivisionError("float division by zero")

    # 列向接收注意力 [heads, seq_k]：对 batch 与 query 维取 mean
    received = weights.mean(dim=(0, 2))

    if received_ema_prev is None:
        ema = received
        delta_a: Optional[float] = None
    else:
        prev = received_ema_prev
        if prev.shape[0] != seq_len:
            if prev.shape[0] < seq_len:
                pad = torch.zeros(
                    seq_len - prev.shape[0],
                    prev.shape[1],
                    dtype=prev.dtype,
                    device=prev.device,
                )
                prev = torch.cat([prev, pad], dim=0)
            else:
                prev = prev[:seq_len]
        prev_t = prev.transpose(0, 1)
        ema = irap_decay * prev_t + (1.0 - irap_decay) * received
        delta_a = float(torch.norm(ema - prev_t) / (torch.norm(prev_t) + 1e-12))

    # 逐 head 在 token 维的升序 dense-rank 归一化 ∈ (0, 1]
    order = received.argsort(dim=-1)
    ranks = order.argsort(dim=-1).to(torch.float64)
    rank_norm = (ranks + 1.0) / seq_len

    # 行熵辅助项（沿用 Metrics.compute_entropy 的 clamp + 全局重归一化语义）
    clamped = weights.clamp(min=1e-10)
    denom = clamped.sum(dim=(0, 3))
    probs = clamped / denom.unsqueeze(0).unsqueeze(-1)
    entropy = -(probs * torch.log(probs)).sum(dim=(0, 3))
    normalized_entropy = (entropy / log_seq).transpose(0, 1)  # [seq_q, heads]
    # 非方形输入口径：KV token t 无对应 query 行时熵辅助项取 0（rank 主信号
    # 仍覆盖全部 seq_k；方形自注意力下无此分支）
    if normalized_entropy.shape[0] != seq_len:
        padded_entropy = torch.zeros(
            seq_len,
            normalized_entropy.shape[1],
            dtype=normalized_entropy.dtype,
            device=normalized_entropy.device,
        )
        keep = min(seq_len, normalized_entropy.shape[0])
        padded_entropy[:keep] = normalized_entropy[:keep]
        normalized_entropy = padded_entropy

    layer_factor = 1.0 + 0.1 * (layer_index / max(num_layers, 1))
    blended = (1.0 - entropy_weight) * rank_norm.transpose(0, 1)
    blended = blended + entropy_weight * normalized_entropy
    importance = (blended * layer_factor).clamp(max=1.0).float()

    if sink_tokens > 0:
        importance[: min(sink_tokens, seq_len)] = 1.0

    return importance, ema.transpose(0, 1), delta_a


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

    def score(
        self,
        attention_scores: torch.Tensor,
        kv_current: Optional[Dict[str, torch.Tensor]] = None,
        kv_previous: Optional[Dict[str, torch.Tensor]] = None,
        layer_id: int = 0,
    ) -> torch.Tensor:
        """Return a ``[seq_len, num_heads]`` importance map (IRAP 共享核心)。

        输入按 logits 契约处理（单次 softmax）。历史状态不再在模型内累积
        （AR003 T003：删除无界 ``_history``，EMA 状态由调用方持有）。
        """
        weights = F.softmax(attention_scores, dim=-1)
        importance_map, _, _ = compute_irap_importance(
            weights,
            received_ema_prev=None,
            layer_index=layer_id,
            num_layers=self.num_layers,
        )

        if kv_current is not None and kv_previous is not None:
            drift_map = self._compute_drift_map(
                kv_current["k"], kv_current["v"], kv_previous["k"], kv_previous["v"]
            )
            # Blend importance with inverse drift: stable heads (low drift)
            # are slightly more reusable, so importance can be discounted.
            importance_map = importance_map * (1.0 - 0.1 * drift_map.to(importance_map.device))
            importance_map = importance_map.clamp(0.0, 1.0)

        return importance_map

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
