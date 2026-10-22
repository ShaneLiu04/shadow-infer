"""Confidence-aware blockwise parallel decoding primitives (AR016).

Implements the block-causal exact protocol scheduling layer (FR16.1): given
per-position logits for the current decoding block, every still-masked
position whose max softmax probability reaches ``confidence_threshold`` is
unmasked in parallel with its argmax token; batch elements with no
above-threshold masked position fall back to a guaranteed-progress top-1
unmask so every round advances at least one position. Already-unmasked
positions are never re-unmasked. Ties in the fallback resolve to the
smallest sequence position for determinism.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

PAGE_PRECISIONS = ("fp32", "fp16", "int8")


@dataclass(frozen=True)
class BlockDecodingConfig:
    """Configuration for blockwise confidence-aware parallel decoding."""

    block_size: int = 32
    max_rounds: int = 8
    confidence_threshold: float = 0.9
    mask_token_id: int = 0
    page_size: int = 32
    page_precision: str = "fp32"

    def validate(self) -> None:
        if self.block_size < 1:
            raise ValueError(f"block_size must be >= 1, got {self.block_size}")
        if self.max_rounds < 1:
            raise ValueError(f"max_rounds must be >= 1, got {self.max_rounds}")
        if not 0.0 < self.confidence_threshold <= 1.0:
            raise ValueError(
                "confidence_threshold must be within (0, 1], got " f"{self.confidence_threshold}"
            )
        if self.page_size < 1:
            raise ValueError(f"page_size must be >= 1, got {self.page_size}")
        if self.page_precision not in PAGE_PRECISIONS:
            raise ValueError(
                f"page_precision must be one of {PAGE_PRECISIONS}, got {self.page_precision}"
            )

    def __post_init__(self) -> None:
        self.validate()


@dataclass
class ScheduleDecision:
    """Result of one scheduling round over the current decoding block.

    ``unmask[b, w]`` marks positions to unmask this round; ``tokens`` holds
    the argmax token for unmasked positions (unspecified elsewhere);
    ``confidences`` holds the max softmax probability for positions that
    remain masked (0 for unmasked positions); ``progress_fallback[b]`` is
    True when element b was unmasked via the guaranteed-progress top-1
    fallback rather than parallel threshold unmasking.
    """

    unmask: torch.Tensor
    tokens: torch.Tensor
    confidences: torch.Tensor
    progress_fallback: torch.Tensor


def schedule_round(
    logits_block: torch.Tensor,
    masked: torch.Tensor,
    config: BlockDecodingConfig,
) -> ScheduleDecision:
    """Schedule one round of confidence-aware parallel unmasking.

    Args:
        logits_block: [B, W, V] position-wise logits for the current block.
        masked: [B, W] bool mask; True marks positions still holding the
            mask token (eligible for unmasking).
        config: BlockDecodingConfig with the confidence threshold.

    Returns:
        ScheduleDecision with per-position unmask decisions, argmax tokens,
        retained confidences, and per-element fallback flags.

    Raises:
        ValueError: If ``logits_block`` is not 3-D or ``masked`` does not
            match the [B, W] leading shape of ``logits_block``.
    """
    if logits_block.dim() != 3:
        raise ValueError(f"logits_block must be 3-D [B, W, V], got dim={logits_block.dim()}")
    if masked.dim() != 2 or masked.shape != logits_block.shape[:2]:
        raise ValueError(
            f"masked must be [B, W] matching logits_block leading shape "
            f"{tuple(logits_block.shape[:2])}, got {tuple(masked.shape)}"
        )

    probs = torch.softmax(logits_block, dim=-1)
    confidences, tokens = probs.max(dim=-1)

    above = masked & (confidences >= config.confidence_threshold)
    has_above = above.any(dim=1)
    has_masked = masked.any(dim=1)
    needs_fallback = has_masked & ~has_above

    masked_conf = confidences.masked_fill(~masked, float("-inf"))
    fallback_positions = masked_conf.argmax(dim=1)
    fallback_unmask = torch.zeros_like(above)
    fallback_unmask[needs_fallback, fallback_positions[needs_fallback]] = True

    unmask = above | fallback_unmask
    retained_confidences = confidences * (masked & ~unmask)

    return ScheduleDecision(
        unmask=unmask,
        tokens=tokens,
        confidences=retained_confidences,
        progress_fallback=needs_fallback,
    )


def block_causal_mask(
    seq_len: int,
    prefix_len: int,
    device: torch.device,
) -> torch.Tensor:
    """Build the block-causal additive attention mask (AR016, plan A).

    Rows before ``prefix_len`` (frozen prefix rows) may only attend to the
    prefix itself: ``mask[i, j] = -inf`` iff ``i < prefix_len`` and
    ``j >= prefix_len``; every other entry is 0. Active block rows attend
    to the full sequence.

    Args:
        seq_len: Total sequence length S (prefix + active block), >= 1.
        prefix_len: Number of frozen prefix rows L, 0 <= L <= S.
        device: Device to place the mask on.

    Returns:
        [S, S] float additive mask.

    Raises:
        ValueError: If ``seq_len < 1``, ``prefix_len < 0`` or
            ``prefix_len > seq_len``.
    """
    if seq_len < 1:
        raise ValueError(f"seq_len must be >= 1, got {seq_len}")
    if prefix_len < 0:
        raise ValueError(f"prefix_len must be >= 0, got {prefix_len}")
    if prefix_len > seq_len:
        raise ValueError(f"prefix_len ({prefix_len}) must not exceed seq_len ({seq_len})")

    mask = torch.zeros(seq_len, seq_len, device=device)
    if prefix_len > 0:
        mask[:prefix_len, prefix_len:] = float("-inf")
    return mask
