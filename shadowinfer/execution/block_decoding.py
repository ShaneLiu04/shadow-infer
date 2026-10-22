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

import time
from dataclasses import dataclass
from typing import TYPE_CHECKING, Dict

import torch

if TYPE_CHECKING:
    from shadowinfer.models.simple_diffusion_llm import SimpleDiffusionLLM

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


@dataclass
class GenerateResult:
    """Result of one blockwise parallel generation call (FR16.3).

    ``sequences`` holds the [B, P + G] long tensor whose prompt region is
    preserved bit-for-bit; ``stats`` carries generation statistics
    (FR16.4 keys are pinned by tests/test_block_decoding.py T004).
    """

    sequences: torch.Tensor
    stats: Dict[str, object]


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


def segment_causal_mask(
    seq_len: int,
    prefix_len: int,
    block_size: int,
    device: torch.device,
) -> torch.Tensor:
    """Build the segment-causal additive attention mask (AR016 erratum 4.5).

    Rows are grouped into segments: the prompt occupies segment 0
    ([0, prefix_len)) and every following block of ``block_size`` rows
    occupies segment 1, 2, ... A row may only attend to columns within
    its own segment or earlier segments:
    ``mask[i, j] = -inf`` iff ``seg(i) < seg(j)``.

    This keeps every row's receptive field time-invariant after sealing
    (prompt rows forever see only the prompt; block-j rows forever see
    the prompt plus blocks 0..j), which is what makes the baseline's
    in-context recomputed prefix K/V bit-identical to the sealed pages.
    For a single trailing block (seq_len == prefix_len + block_size) the
    result is element-wise identical to ``block_causal_mask``.

    Args:
        seq_len: Total sequence length S, >= 1.
        prefix_len: Prompt length P, 0 <= P <= S.
        block_size: Block width W, >= 1.
        device: Device to place the mask on.

    Returns:
        [S, S] float additive mask.

    Raises:
        ValueError: On seq_len < 1, prefix_len < 0, prefix_len > seq_len
            or block_size < 1.
    """
    if seq_len < 1:
        raise ValueError(f"seq_len must be >= 1, got {seq_len}")
    if prefix_len < 0:
        raise ValueError(f"prefix_len must be >= 0, got {prefix_len}")
    if prefix_len > seq_len:
        raise ValueError(f"prefix_len ({prefix_len}) must not exceed seq_len ({seq_len})")
    if block_size < 1:
        raise ValueError(f"block_size must be >= 1, got {block_size}")

    indices = torch.arange(seq_len, device=device)
    segments = torch.where(
        indices < prefix_len,
        torch.zeros_like(indices),
        1 + (indices - prefix_len) // block_size,
    )
    blocked = segments[:, None] < segments[None, :]
    return torch.where(blocked, float("-inf"), 0.0)


def validate_generate_inputs(
    model: SimpleDiffusionLLM,
    prompt_ids: torch.Tensor,
    gen_len: int,
    config: BlockDecodingConfig,
) -> int:
    """Shared fail-fast validation for both generation arms (FR16.3).

    Returns the prompt length P; raises ValueError before any state is
    created so failed calls have no side effects.
    """
    if prompt_ids.dim() != 2:
        raise ValueError(f"prompt_ids must be 2-D [B, P], got dim={prompt_ids.dim()}")
    if prompt_ids.dtype not in (torch.long, torch.int):
        raise ValueError(f"prompt_ids must be integer token ids, got dtype={prompt_ids.dtype}")
    prompt_len = int(prompt_ids.shape[1])
    if prompt_len + gen_len > model.max_seq_len:
        raise ValueError(
            f"prompt_len + gen_len ({prompt_len} + {gen_len}) exceeds model "
            f"max_seq_len ({model.max_seq_len})"
        )
    if bool((prompt_ids == config.mask_token_id).any()):
        raise ValueError(f"prompt_ids must not contain mask_token_id ({config.mask_token_id})")
    if gen_len % config.block_size != 0:
        raise ValueError(
            f"gen_len ({gen_len}) must be a multiple of block_size " f"({config.block_size})"
        )
    return prompt_len


def blockwise_generate_baseline(
    model: SimpleDiffusionLLM,
    prompt_ids: torch.Tensor,
    gen_len: int,
    config: BlockDecodingConfig,
    capture_block_logits: bool = False,
) -> GenerateResult:
    """Full-recompute reference loop for blockwise decoding (AR016, FR16.3).

    Mirror image of ``PyTorchModelBackend.generate`` sharing the same
    ``BlockDecodingConfig`` and ``schedule_round`` (decision-equivalent
    arms): every round recomputes the full [B, L_j + W] context through
    ``model.forward_step`` under the block-causal mask, so no prefix KV
    store and no sealing pass exist. Serves as the exactness reference
    for unit tests and the baseline arm for benchmarks.

    Args:
        model: SimpleDiffusionLLM instance (eval mode recommended).
        prompt_ids: [B, P] integer prompt token ids.
        gen_len: Tokens to generate; multiple of ``config.block_size``.
        config: BlockDecodingConfig (threshold/block/round settings).
        capture_block_logits: When True, stats additionally carries
            ``"block_logits_per_round"`` ([B, W, V] per round per block).

    Returns:
        GenerateResult whose stats follow the recompute semantics:
        ``computed_rows_total == sum(rounds_j * (L_j + W))``,
        ``kv_write_rows_total == 0``, ``rows_saved_vs_baseline == 0``.

    Raises:
        ValueError: On the same fail-fast conditions as ``generate``.
    """
    prompt_len = validate_generate_inputs(model, prompt_ids, gen_len, config)
    batch = int(prompt_ids.shape[0])
    width = config.block_size
    num_blocks = gen_len // width

    sequences = torch.cat(
        [
            prompt_ids.long(),
            torch.full((batch, gen_len), config.mask_token_id, dtype=torch.long),
        ],
        dim=1,
    )

    forced_completions = 0
    rounds_per_block: list = []
    unmasked_per_round: list = []
    fallback_per_round: list = []
    mean_confidence_per_round: list = []
    block_logits_per_round: list = []
    unmask_positions_per_round: list = []
    block_time_ms: list = []
    start_time = time.perf_counter()
    prefix_len = prompt_len
    for _ in range(num_blocks):
        block_start = time.perf_counter()
        block_tokens = sequences[:, prefix_len : prefix_len + width]
        masked = torch.ones(batch, width, dtype=torch.bool)
        rounds = 0
        logits: torch.Tensor = torch.empty(0)
        block_unmasked: list = []
        block_fallback: list = []
        block_confidence: list = []
        captured_rounds: list = []
        block_positions: list = []
        while bool(masked.any()) and rounds < config.max_rounds:
            full_ids = sequences[:, : prefix_len + width]
            mask = segment_causal_mask(prefix_len + width, prompt_len, width, sequences.device)
            full_logits, _, _ = model.forward_step(full_ids, 0, 0, attention_mask=mask)
            logits = full_logits[:, prefix_len:]
            decision = schedule_round(logits, masked, config)
            block_tokens[decision.unmask] = decision.tokens[decision.unmask]
            masked &= ~decision.unmask
            block_unmasked.append(int(decision.unmask.sum()))
            block_fallback.append(bool(decision.progress_fallback.any()))
            round_confidence = torch.softmax(logits, dim=-1).max(dim=-1).values
            block_confidence.append(float(round_confidence[decision.unmask].mean()))
            if capture_block_logits:
                captured_rounds.append(logits.detach().clone())
            block_positions.append(decision.unmask.detach().clone())
            rounds += 1
        if bool(masked.any()):
            remaining = logits.argmax(dim=-1)
            block_tokens[masked] = remaining[masked]
            forced_completions += 1
        rounds_per_block.append(rounds)
        unmasked_per_round.append(block_unmasked)
        fallback_per_round.append(block_fallback)
        mean_confidence_per_round.append(block_confidence)
        unmask_positions_per_round.append(block_positions)
        if capture_block_logits:
            block_logits_per_round.append(captured_rounds)
        block_time_ms.append((time.perf_counter() - block_start) * 1000.0)
        prefix_len += width
    total_time_ms = (time.perf_counter() - start_time) * 1000.0

    computed_rows_total = sum(
        rounds * (prompt_len + block_index * width + width)
        for block_index, rounds in enumerate(rounds_per_block)
    )
    stats: Dict[str, object] = {
        "blocks": num_blocks,
        "total_rounds": sum(rounds_per_block),
        "computed_rows_total": computed_rows_total,
        "kv_write_rows_total": 0,
        "forced_completions": forced_completions,
        "total_time_ms": float(total_time_ms),
        "page_precision": "fp32",
        "rounds_per_block": rounds_per_block,
        "unmasked_per_round": unmasked_per_round,
        "fallback_per_round": fallback_per_round,
        "mean_confidence_per_round": mean_confidence_per_round,
        "rows_saved_vs_baseline": 0,
        "unmask_positions_per_round": unmask_positions_per_round,
        "block_time_ms": block_time_ms,
    }
    if capture_block_logits:
        stats["block_logits_per_round"] = block_logits_per_round
    return GenerateResult(sequences=sequences, stats=stats)
