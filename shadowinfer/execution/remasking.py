"""dLLM-native remasking generation baseline arm (AR018 T005, FR18.4).

Implements the official dLLM generation semantics used by LLaDA / Dream
(low-confidence remasking): the response canvas starts fully masked and
every pass recomputes the FULL sequence through a genuinely bidirectional
``forward_step`` (no attention mask, design D5(a)), then unmasks a fixed,
configurable number of still-masked positions -- chosen by confidence
(highest softmax max first) or uniformly at random -- while all other
positions keep their current token (unmasked) or the mask token. The
loop runs until the canvas is fully unmasked or ``max_passes`` is hit.

This arm is deliberately a standalone module (design D8): it shares no
scheduling code with the blockwise pipeline and reports its own stats
key set (``passes_used`` / ``unmasked_per_pass`` / ``final_confidences``
/ ``mask_residuals``), which is explicitly NOT the 17-key blockwise
stats contract. It serves as the native-semantics baseline for the s13
four-arm comparison; on local fixtures only structural correctness and
finite-pass convergence are asserted (random weights carry no semantic
claims).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Dict, List, Optional

import torch

from shadowinfer.execution.block_decoding import GenerateResult

if TYPE_CHECKING:
    from shadowinfer.execution.block_decoding import BlockwiseModel

__all__ = ["RemaskingConfig", "remasking_generate"]

_REMASKING_STRATEGIES = ("confidence", "random")


@dataclass(frozen=True)
class RemaskingConfig:
    """Configuration for the remasking baseline arm (design D8).

    Attributes:
        mask_token_id: Diffusion mask token id (must be inside the model
            vocabulary; prompts must not contain it).
        unmask_per_pass: Number of still-masked positions unmasked per
            pass per batch element (>= 1).
        max_passes: Upper bound on passes (>= 1); exhaustion leaves the
            remaining mask tokens in place and records them in stats.
        strategy: ``"confidence"`` unmasks the highest-confidence masked
            positions first (LLaDA low-confidence remasking); ``"random"``
            picks uniformly among masked positions (literature control,
            reproducible via a seeded ``torch.Generator``).
    """

    mask_token_id: int
    unmask_per_pass: int
    max_passes: int
    strategy: str = "confidence"


def remasking_generate(
    model: "BlockwiseModel",
    prompt_ids: torch.Tensor,
    gen_len: int,
    config: RemaskingConfig,
    generator: Optional[torch.Generator] = None,
) -> GenerateResult:
    """dLLM-native remasking generation (design D5(a)/D8, FR18.4).

    Args:
        model: BlockwiseModel protocol instance; only ``forward_step``
            (full bidirectional recompute) is used.
        prompt_ids: [B, P] integer prompt token ids; must not contain
            ``config.mask_token_id``.
        gen_len: Number of tokens to generate G (>= 1; NO divisibility
            constraint -- this arm has no block structure).
        config: RemaskingConfig with mask token / pass / strategy
            settings.
        generator: Optional seeded torch.Generator used by the
            ``"random"`` strategy for reproducibility; unused by
            ``"confidence"``.

    Returns:
        GenerateResult with [B, P + G] sequences (prompt preserved
        bit-for-bit) and the standalone remasking stats keys:
        ``passes_used`` (int), ``unmasked_per_pass`` (list[int], batch
        totals per pass), ``final_confidences`` (list[float], mean
        confidence of the positions unmasked in each pass),
        ``mask_residuals`` (int, masked positions left at termination;
        0 when fully unmasked).

    Raises:
        ValueError: Fail-fast on invalid prompt (shape/dtype/mask
            content), gen_len < 1, over-length sequences, mask token
            id outside the vocabulary, unmask_per_pass < 1,
            max_passes < 1, or an unknown strategy. No state is created
            and no forward pass runs before validation passes.
    """
    if prompt_ids.dim() != 2:
        raise ValueError(f"prompt_ids must be 2-D [B, P], got dim={prompt_ids.dim()}")
    if prompt_ids.dtype not in (torch.long, torch.int):
        raise ValueError(f"prompt_ids must be integer token ids, got dtype={prompt_ids.dtype}")
    prompt_len = int(prompt_ids.shape[1])
    if gen_len < 1:
        raise ValueError(f"gen_len must be >= 1, got {gen_len}")
    if prompt_len + gen_len > model.max_seq_len:
        raise ValueError(
            f"prompt_len + gen_len ({prompt_len} + {gen_len}) exceeds model "
            f"max_seq_len ({model.max_seq_len})"
        )
    if not 0 <= config.mask_token_id < model.vocab_size:
        raise ValueError(
            f"mask_token_id ({config.mask_token_id}) must be within the model "
            f"vocabulary [0, vocab_size={model.vocab_size})"
        )
    if bool((prompt_ids == config.mask_token_id).any()):
        raise ValueError(f"prompt_ids must not contain mask_token_id ({config.mask_token_id})")
    if config.unmask_per_pass < 1:
        raise ValueError(f"unmask_per_pass must be >= 1, got {config.unmask_per_pass}")
    if config.max_passes < 1:
        raise ValueError(f"max_passes must be >= 1, got {config.max_passes}")
    if config.strategy not in _REMASKING_STRATEGIES:
        raise ValueError(
            f"strategy must be one of {_REMASKING_STRATEGIES}, got {config.strategy!r}"
        )

    batch = int(prompt_ids.shape[0])
    sequences = torch.cat(
        [
            prompt_ids.long(),
            torch.full((batch, gen_len), config.mask_token_id, dtype=torch.long),
        ],
        dim=1,
    )
    response = sequences[:, prompt_len:]
    masked = torch.ones(batch, gen_len, dtype=torch.bool)

    passes_used = 0
    unmasked_per_pass: List[int] = []
    final_confidences: List[float] = []
    while bool(masked.any()) and passes_used < config.max_passes:
        logits, _, _ = model.forward_step(sequences, 0, 0)
        probs = torch.softmax(logits[:, prompt_len:, :], dim=-1)
        confidence, tokens = probs.max(dim=-1)
        passes_used += 1
        pass_unmasked = 0
        pass_confidences: List[float] = []
        for b in range(batch):
            candidates = masked[b]
            remaining = int(candidates.sum())
            if remaining == 0:
                continue
            k = min(config.unmask_per_pass, remaining)
            if config.strategy == "confidence":
                masked_confidence = torch.where(
                    candidates, confidence[b], torch.finfo(confidence.dtype).min
                )
                chosen = torch.topk(masked_confidence, k).indices
            else:
                candidate_indices = candidates.nonzero(as_tuple=False).flatten()
                permutation = torch.randperm(remaining, generator=generator)
                chosen = candidate_indices[permutation[:k]]
            response[b, chosen] = tokens[b, chosen]
            masked[b, chosen] = False
            pass_unmasked += k
            pass_confidences.extend(confidence[b, chosen].tolist())
        unmasked_per_pass.append(pass_unmasked)
        final_confidences.append(
            float(sum(pass_confidences) / len(pass_confidences)) if pass_confidences else 0.0
        )

    stats: Dict[str, object] = {
        "passes_used": passes_used,
        "unmasked_per_pass": unmasked_per_pass,
        "final_confidences": final_confidences,
        "mask_residuals": int(masked.sum()),
    }
    return GenerateResult(sequences=sequences, stats=stats)
