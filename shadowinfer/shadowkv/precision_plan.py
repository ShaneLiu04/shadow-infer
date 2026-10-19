"""Precision/reuse plan for ShadowKV compression and packing (AR008).

FlashInfer-style plan/run separation: build the grouping and flat gather
indices once, then reuse the same plan for both the compression working copy
and the byte-level packed cache. Plans are immutable snapshots and carry a
SHA-256 fingerprint for reproducibility.
"""

from __future__ import annotations

import hashlib
import struct
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import torch

PRECISION_ORDER: Tuple[str, ...] = ("fp32", "fp16", "int8", "int4")
EVICTED = "evicted"
PRECISION_CODES: Dict[str, int] = {
    "fp32": 0,
    "fp16": 1,
    "int8": 2,
    "int4": 3,
    EVICTED: 4,
}
CODE_PRECISIONS: Dict[int, str] = {code: name for name, code in PRECISION_CODES.items()}


def map_to_codes(
    precision_map: Dict[int, Dict[int, str]],
    seq_len: int,
    num_heads: int,
    strict: bool = True,
    device: Any = "cpu",
) -> torch.Tensor:
    """Convert a per-token/head precision map into an ``int8 [seq, heads]``.

    Missing entries default to ``fp16``. Unsupported names raise ``ValueError``
    when ``strict`` is true, otherwise they are normalized to ``fp16``.
    """
    rows: List[List[int]] = []
    for token_idx in range(seq_len):
        token_map = precision_map.get(token_idx, {})
        row: List[int] = []
        for head_idx in range(num_heads):
            raw = token_map.get(head_idx, "fp16")
            precision = raw.lower() if isinstance(raw, str) else raw
            if precision not in PRECISION_CODES:
                if strict:
                    raise ValueError(f"Unsupported precision: {raw}")
                precision = "fp16"
            row.append(PRECISION_CODES[precision])
        rows.append(row)
    return torch.tensor(rows, dtype=torch.int8, device=device)


def codes_to_map(codes: torch.Tensor) -> Dict[int, Dict[int, str]]:
    """Convert ``int8 [seq, heads]`` codes back into a precision map."""
    values = codes.detach().to("cpu").to(torch.int64).tolist()
    precision_map: Dict[int, Dict[int, str]] = {}
    for token_idx, row in enumerate(values):
        token_map: Dict[int, str] = {}
        for head_idx, code in enumerate(row):
            if code not in CODE_PRECISIONS:
                raise ValueError(f"Unsupported precision code: {code}")
            token_map[head_idx] = CODE_PRECISIONS[code]
        precision_map[token_idx] = token_map
    return precision_map


@dataclass(frozen=True)
class PrecisionPlan:
    """Immutable grouping/index plan shared by compress and pack."""

    seq_len: int
    num_heads: int
    codes: torch.Tensor
    groups: Dict[str, List[Tuple[int, int]]]
    flat_indices: Dict[str, torch.Tensor]
    reuse_entries: List[Tuple[int, int]]
    reuse_flat: torch.Tensor
    fingerprint: str
    evicted_entries: List[Tuple[int, int]]

    @classmethod
    def build(
        cls,
        precision_map: Dict[int, Dict[int, str]],
        seq_len: int,
        num_heads: int,
        reuse_mask: Optional[Dict[Tuple[int, int], bool]] = None,
        device: Any = "cpu",
        strict: bool = True,
    ) -> "PrecisionPlan":
        """Build a plan from per-token/head precision decisions.

        Args:
            precision_map: ``token_idx -> head_idx -> precision``.
            seq_len: number of cached tokens.
            num_heads: number of attention heads.
            reuse_mask: optional ``(token_idx, head_idx) -> bool`` reuse flags.
            device: device for index/code tensors.
            strict: raise on unsupported precision instead of normalizing to
                fp16 (compression semantics). ``PackedKVCache`` normalizes.

        Raises:
            ValueError: invalid shape or unsupported precision when ``strict``.
        """
        codes = map_to_codes(precision_map, seq_len, num_heads, strict=strict, device=device)
        return cls.from_codes(codes, reuse_mask=reuse_mask, device=device)

    @classmethod
    def from_codes(
        cls,
        codes: torch.Tensor,
        reuse_mask: Optional[Dict[Tuple[int, int], bool]] = None,
        device: Any = None,
    ) -> "PrecisionPlan":
        """Build a plan directly from dense ``int8 [seq, heads]`` codes (AR010)."""
        if codes.dim() != 2:
            raise ValueError(f"codes must be 2-D [seq, heads], got shape {tuple(codes.shape)}")
        seq_len, num_heads = int(codes.shape[0]), int(codes.shape[1])
        if seq_len <= 0 or num_heads <= 0:
            raise ValueError(f"seq_len and num_heads must be positive, got {seq_len}, {num_heads}")
        if device is None:
            device = codes.device

        codes = codes.detach().to(device=device, dtype=torch.int8).contiguous()
        working = codes.detach().to("cpu", dtype=torch.int8).contiguous()
        flat = working.reshape(-1).to(torch.int64)

        groups: Dict[str, List[Tuple[int, int]]] = {p: [] for p in PRECISION_ORDER}
        flat_indices: Dict[str, torch.Tensor] = {}
        evicted_entries: List[Tuple[int, int]] = []
        for code, precision in CODE_PRECISIONS.items():
            selected = (flat == code).nonzero(as_tuple=False).reshape(-1)
            if selected.numel() == 0:
                if precision != EVICTED:
                    flat_indices[precision] = torch.empty(0, dtype=torch.long, device=device)
                continue
            token_idx = torch.div(selected, num_heads, rounding_mode="floor")
            head_idx = torch.remainder(selected, num_heads)
            entries = list(zip(token_idx.tolist(), head_idx.tolist()))
            if precision == EVICTED:
                evicted_entries = entries
            else:
                groups[precision] = entries
                flat_indices[precision] = (head_idx * seq_len + token_idx).to(
                    device=device, dtype=torch.long
                )

        reuse_entries: List[Tuple[int, int]] = []
        if reuse_mask:
            reuse_entries = sorted(
                (int(token_idx), int(head_idx))
                for (token_idx, head_idx), flag in reuse_mask.items()
                if flag and 0 <= token_idx < seq_len and 0 <= head_idx < num_heads
            )
        reuse_flat = torch.tensor(
            [head_idx * seq_len + token_idx for token_idx, head_idx in reuse_entries],
            dtype=torch.long,
            device=device,
        )

        fingerprint = cls._fingerprint(codes, reuse_flat, seq_len, num_heads)
        return cls(
            seq_len=seq_len,
            num_heads=num_heads,
            codes=codes,
            groups=groups,
            flat_indices=flat_indices,
            reuse_entries=reuse_entries,
            reuse_flat=reuse_flat,
            fingerprint=fingerprint,
            evicted_entries=evicted_entries,
        )

    @staticmethod
    def _fingerprint(
        codes: torch.Tensor,
        reuse_flat: torch.Tensor,
        seq_len: int,
        num_heads: int,
    ) -> str:
        digest = hashlib.sha256()
        digest.update(struct.pack("<QQ", seq_len, num_heads))
        digest.update(codes.detach().to("cpu").contiguous().numpy().tobytes())
        digest.update(reuse_flat.detach().to("cpu").contiguous().numpy().tobytes())
        return digest.hexdigest()

    def reuse_set(self) -> set:
        """Return reuse entries as a set for O(1) membership checks."""
        return set(self.reuse_entries)
