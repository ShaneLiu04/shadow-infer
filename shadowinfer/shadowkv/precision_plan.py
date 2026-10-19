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
PRECISION_CODES: Dict[str, int] = {
    "fp32": 0,
    "fp16": 1,
    "int8": 2,
    "int4": 3,
}


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
        if seq_len <= 0 or num_heads <= 0:
            raise ValueError(f"seq_len and num_heads must be positive, got {seq_len}, {num_heads}")

        groups: Dict[str, List[Tuple[int, int]]] = {p: [] for p in PRECISION_ORDER}
        rows: List[List[int]] = []
        for token_idx in range(seq_len):
            token_map = precision_map.get(token_idx, {})
            codes_row: List[int] = []
            for head_idx in range(num_heads):
                raw = token_map.get(head_idx, "fp16")
                precision = raw.lower() if isinstance(raw, str) else raw
                if precision not in groups:
                    if strict:
                        raise ValueError(f"Unsupported precision: {raw}")
                    precision = "fp16"
                codes_row.append(PRECISION_CODES[precision])
                groups[precision].append((token_idx, head_idx))
            rows.append(codes_row)

        codes = torch.tensor(rows, dtype=torch.int8, device=device)
        flat_indices = {
            precision: torch.tensor(
                [head_idx * seq_len + token_idx for token_idx, head_idx in entries],
                dtype=torch.long,
                device=device,
            )
            for precision, entries in groups.items()
        }

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
