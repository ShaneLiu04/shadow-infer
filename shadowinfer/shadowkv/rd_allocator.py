"""Rate-distortion aware KV precision admission and optional hard eviction (AR009).

The allocator replaces importance-only downgrade ordering with
``importance * sensitivity`` where sensitivity is the per-token-head KV
amplitude squared (a cheap proxy for quantization error). When the layer budget
is below the int4 floor and hard eviction is enabled, the lowest-score
token-heads are dropped entirely (zero bytes).

All functions operate on real tensors; there are no mocks or probabilistic
shortcuts.
"""

from __future__ import annotations

from typing import Dict, Optional, Set, Tuple

import torch

EVICTED = "evicted"
DOWNGRADE_LADDER = {"fp32": "fp16", "fp16": "int8", "int8": "int4"}


def entry_bytes(precision: str, batch: int, head_dim: int) -> float:
    """Return approximate stored bytes for one token-head (K+V, incl. scales).

    Matches ``PackedKVCache.pack`` defaults: INT4 group size 16 when divisible,
    otherwise the whole ``head_dim`` as a single group.
    """
    p = precision.lower() if isinstance(precision, str) else precision
    if p == EVICTED:
        return 0.0
    if p == "int4":
        group = 16 if head_dim % 16 == 0 else head_dim
        per_tensor = batch * head_dim / 2.0 + batch * (head_dim // group) * 4
    elif p == "int8":
        per_tensor = batch * head_dim + batch * 4
    elif p == "fp32":
        per_tensor = batch * head_dim * 4
    else:
        per_tensor = batch * head_dim * 2
    return per_tensor * 2.0


def kv_sensitivity(k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """Per-token-head KV amplitude sensitivity, shape ``[seq, heads]``.

    Computed as ``(|k|max^2 + |v|max^2) / 2`` with the max taken across batch
    and head_dim. This is an O(N) proxy for quantization error magnitude and
    requires no full quantize.
    """
    if k.shape != v.shape:
        raise ValueError("k and v must have identical shapes")
    k_abs = k.detach().abs().amax(dim=0).amax(dim=-1)
    v_abs = v.detach().abs().amax(dim=0).amax(dim=-1)
    sensitivity = (k_abs.square() + v_abs.square()) / 2.0
    return sensitivity.transpose(0, 1).contiguous()


def admit_budget(
    importance: torch.Tensor,
    sensitivity: torch.Tensor,
    precision_map: Dict[int, Dict[int, str]],
    budget_bytes: int,
    batch: int,
    head_dim: int,
    sink_tokens: int = 0,
    rd_allocation: bool = True,
    allow_eviction: bool = False,
    protected: Optional[Set[Tuple[int, int]]] = None,
) -> Tuple[Dict[int, Dict[int, str]], int, int]:
    """Fit a precision map into ``budget_bytes``.

    Returns ``(map, predicted_bytes, evicted_count)``. Candidates are ranked by
    ``importance * sensitivity`` ascending (or importance only when
    ``rd_allocation`` is False). Sink tokens and ``protected`` entries are never
    downgraded or evicted.
    """
    result: Dict[int, Dict[int, str]] = {
        int(token_idx): dict(head_map) for token_idx, head_map in precision_map.items()
    }
    if budget_bytes is None or budget_bytes < 0:
        predicted = int(
            sum(
                entry_bytes(p, batch, head_dim)
                for head_map in result.values()
                for p in head_map.values()
            )
        )
        return result, predicted, 0

    seq_len = min(int(importance.shape[0]), max(result.keys(), default=0) + 1)
    num_heads = min(
        int(importance.shape[1]) if importance.dim() > 1 else 1,
        max((len(m) for m in result.values()), default=0),
    )
    if seq_len <= 0 or num_heads <= 0:
        predicted = int(
            sum(
                entry_bytes(p, batch, head_dim)
                for head_map in result.values()
                for p in head_map.values()
            )
        )
        return result, predicted, 0

    imp = importance.detach().to("cpu").float()[:seq_len, :num_heads]
    sens = sensitivity.detach().to("cpu").float()[:seq_len, :num_heads]
    if sens.shape != imp.shape:
        sens = torch.ones_like(imp)
    score = imp * sens if rd_allocation else imp

    protected = frozenset(protected or ())
    candidates = sorted(
        (
            (float(score[token_idx, head_idx]), token_idx, head_idx)
            for token_idx in range(seq_len)
            for head_idx in range(num_heads)
            if token_idx >= sink_tokens and (token_idx, head_idx) not in protected
        ),
        key=lambda item: item[0],
    )

    predicted = int(
        sum(
            entry_bytes(p, batch, head_dim)
            for head_map in result.values()
            for p in head_map.values()
        )
    )

    index = 0
    while predicted > budget_bytes and index < len(candidates):
        _, token_idx, head_idx = candidates[index]
        token_map = result.setdefault(token_idx, {})
        current = token_map.get(head_idx, "fp16")
        next_precision = DOWNGRADE_LADDER.get(current)
        if next_precision is None:
            index += 1
            continue
        predicted -= int(
            entry_bytes(current, batch, head_dim) - entry_bytes(next_precision, batch, head_dim)
        )
        token_map[head_idx] = next_precision
        if next_precision == "int4":
            index += 1

    evicted_count = 0
    if allow_eviction and predicted > budget_bytes:
        index = 0
        while predicted > budget_bytes and index < len(candidates):
            _, token_idx, head_idx = candidates[index]
            token_map = result.setdefault(token_idx, {})
            current = token_map.get(head_idx, "fp16")
            if current != EVICTED:
                predicted -= int(entry_bytes(current, batch, head_dim))
                token_map[head_idx] = EVICTED
                evicted_count += 1
            index += 1

    return result, int(predicted), evicted_count
