"""Blockwise generate orchestration as a pure function (AR018 T004, D3).

Cut-paste of ``PyTorchModelBackend.generate``'s orchestration body
(AR016 + AR017 logic), zero semantic change: the same fail-fast checks
(same messages, same order, minus the backend-only "no model loaded"
RuntimeError), the same stats contract, the same page-residency
invariants (prefix K/V appended exactly twice per row lifetime --
prefill P rows plus each block's sealing W rows; decoding rounds only
gather). The backend method is now a thin delegation that keeps the
``_last_generate_stores`` semantics.

The single functional delta (AR018 design D1/D4): per-layer stores are
constructed with the model's native KV head count via
``getattr(model, "num_kv_heads", model.num_heads)`` so GQA models
(DiffusionLlama) page KV at native width. MHA models expose no
``num_kv_heads`` attribute and keep the exact previous behavior
bit-for-bit.
"""

from __future__ import annotations

import time
from typing import TYPE_CHECKING, Dict, List, Optional, Tuple

import torch

from shadowinfer.execution.block_decoding import (
    BlockDecodingConfig,
    GenerateResult,
    draft_round,
    schedule_round,
    validate_generate_inputs,
    verify_round,
)
from shadowinfer.shadowkv.paged_kv_store import PagedKVStore

if TYPE_CHECKING:
    from shadowinfer.execution.block_decoding import BlockwiseModel

__all__ = ["blockwise_generate"]


def blockwise_generate(
    model: "BlockwiseModel",
    prompt_ids: torch.Tensor,
    gen_len: int,
    config: BlockDecodingConfig,
    capture_block_logits: bool = False,
) -> Tuple[GenerateResult, Dict[int, PagedKVStore]]:
    """Blockwise confidence-aware parallel generation (AR016/AR017
    orchestration, pure-function form per AR018 D3).

    Batch-style API, fully independent of the forward_step serving
    path: fresh per-layer PagedKVStore instances are created per call.
    The prefix K/V is appended exactly twice per row lifetime -- once
    at prefill (P rows) and once at each block's sealing pass (W rows)
    -- while decoding rounds only gather (zero inter-round page writes).

    Args:
        model: BlockwiseModel protocol instance (attributes
            num_layers/num_heads/head_dim/max_seq_len/vocab_size/device
            plus forward_step/forward_block; GQA models may expose
            ``num_kv_heads``, in which case stores page KV at native
            width, design D1/D4).
        prompt_ids: [B, P] integer prompt token ids; must not contain
            ``config.mask_token_id``.
        gen_len: Number of tokens to generate G; must be a multiple of
            ``config.block_size``.
        config: BlockDecodingConfig with block/threshold/page settings.
        capture_block_logits: When True, per-round block logits are
            captured into ``stats["block_logits_per_round"]``.

    Returns:
        ``(GenerateResult, stores)`` where the result carries [B, P+G]
        sequences (prompt preserved bit-for-bit) and generation stats,
        and ``stores`` maps every layer to the PagedKVStore the
        generation used.

    Raises:
        ValueError: On invalid prompt shape/dtype, over-length
            sequences, mask tokens inside the prompt, mask token id out
            of the model vocabulary, or gen_len not divisible by the
            block size (all fail-fast: the pure function creates no
            stores and mutates no state).
    """
    prompt_len = validate_generate_inputs(model, prompt_ids, gen_len, config)
    batch = int(prompt_ids.shape[0])
    if bool((prompt_ids == config.mask_token_id).any()):
        raise ValueError(f"prompt_ids must not contain mask_token_id ({config.mask_token_id})")
    if gen_len % config.block_size != 0:
        raise ValueError(
            f"gen_len ({gen_len}) must be a multiple of block_size " f"({config.block_size})"
        )

    width = config.block_size
    num_blocks = gen_len // width
    num_layers = model.num_layers
    num_store_heads = getattr(model, "num_kv_heads", model.num_heads)

    stores: Dict[int, PagedKVStore] = {
        layer_id: PagedKVStore(
            batch_size=batch,
            num_heads=num_store_heads,
            head_dim=model.head_dim,
            page_size=config.page_size,
            device=model.device,
        )
        for layer_id in range(num_layers)
    }

    _, prefill_kv, _ = model.forward_step(prompt_ids, 0, 0)
    for layer_id in range(num_layers):
        k, v = prefill_kv[layer_id]
        stores[layer_id].append(k, v, config.page_precision)

    sequences = torch.cat(
        [
            prompt_ids.long(),
            torch.full((batch, gen_len), config.mask_token_id, dtype=torch.long),
        ],
        dim=1,
    )

    forced_completions = 0
    rounds_per_block: List[int] = []
    seal_rounds: List[int] = []
    unmasked_per_round: List[List[int]] = []
    fallback_per_round: List[List[bool]] = []
    mean_confidence_per_round: List[List[float]] = []
    block_logits_per_round: List[List[torch.Tensor]] = []
    unmask_positions_per_round: List[List[torch.Tensor]] = []
    block_time_ms: List[float] = []
    draft_accepted_positions = 0
    rounds_to_fixpoint: Optional[List[int]] = None
    changed_per_round: Optional[List[List[int]]] = None
    if config.draft_mode:
        rounds_to_fixpoint = []
        changed_per_round = []
    start_time = time.perf_counter()
    prefix_len = prompt_len
    for _ in range(num_blocks):
        block_start = time.perf_counter()
        block_tokens = sequences[:, prefix_len : prefix_len + width]
        rounds = 0
        logits: Optional[torch.Tensor] = None
        block_unmasked: List[int] = []
        block_fallback: List[bool] = []
        block_confidence: List[float] = []
        captured_rounds: List[torch.Tensor] = []
        block_positions: List[torch.Tensor] = []
        seal_kv: Optional[List] = None
        if config.draft_mode:
            block_changed: List[int] = []
            draft_snapshot: Optional[torch.Tensor] = None
            while True:
                prefix_kv = {layer_id: stores[layer_id].gather() for layer_id in range(num_layers)}
                logits, kv = model.forward_block(
                    block_tokens, prefix_kv, prefix_len, step_t=0, total_steps=0
                )
                rounds += 1
                if capture_block_logits:
                    captured_rounds.append(logits.detach().clone())
                if rounds == 1:
                    round_tokens, round_conf = draft_round(logits)
                    block_tokens.copy_(round_tokens)
                    draft_snapshot = block_tokens.detach().clone()
                    round_unmasked = int(batch * width)
                    round_positions = torch.ones(batch, width, dtype=torch.bool)
                    round_changed = int(batch * width)
                    fixpoint = False
                else:
                    changed, round_tokens, round_conf = verify_round(logits, block_tokens)
                    fixpoint = int(changed.sum()) == 0
                    if not fixpoint:
                        block_tokens[changed] = round_tokens[changed]
                    round_unmasked = 0
                    round_positions = torch.zeros(batch, width, dtype=torch.bool)
                    round_changed = int(changed.sum())
                block_unmasked.append(round_unmasked)
                block_fallback.append(False)
                block_confidence.append(float(round_conf.mean()))
                block_positions.append(round_positions)
                block_changed.append(round_changed)
                if fixpoint:
                    seal_kv = kv
                    break
                if rounds >= config.max_rounds:
                    break
            if seal_kv is None:
                forced_completions += 1
                prefix_kv = {layer_id: stores[layer_id].gather() for layer_id in range(num_layers)}
                _, seal_kv = model.forward_block(
                    block_tokens, prefix_kv, prefix_len, step_t=0, total_steps=0
                )
                seal_rounds.append(1)
            else:
                seal_rounds.append(0)
            block_kv = seal_kv
            draft_accepted_positions += int((block_tokens == draft_snapshot).sum())
            rounds_to_fixpoint.append(rounds if fixpoint else -1)
            changed_per_round.append(block_changed)
        else:
            masked = torch.ones(batch, width, dtype=torch.bool)
            while bool(masked.any()) and rounds < config.max_rounds:
                prefix_kv = {layer_id: stores[layer_id].gather() for layer_id in range(num_layers)}
                logits, _ = model.forward_block(
                    block_tokens, prefix_kv, prefix_len, step_t=0, total_steps=0
                )
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
                remaining = logits.argmax(dim=-1) if logits is not None else None
                if remaining is None:
                    raise RuntimeError("forced completion requires computed logits")
                block_tokens[masked] = remaining[masked]
                forced_completions += 1
            prefix_kv = {layer_id: stores[layer_id].gather() for layer_id in range(num_layers)}
            _, block_kv = model.forward_block(
                block_tokens, prefix_kv, prefix_len, step_t=0, total_steps=0
            )
            seal_rounds.append(1)
        for layer_id in range(num_layers):
            k, v = block_kv[layer_id]
            stores[layer_id].append(k, v, config.page_precision)
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
        width * (rounds + seal) for rounds, seal in zip(rounds_per_block, seal_rounds)
    )
    rows_saved_vs_baseline = sum(
        rounds * (prompt_len + block_index * width + width) - width * (rounds + seal)
        for block_index, (rounds, seal) in enumerate(zip(rounds_per_block, seal_rounds))
    )
    stats: Dict[str, object] = {
        "blocks": num_blocks,
        "total_rounds": sum(rounds_per_block),
        "computed_rows_total": computed_rows_total,
        "kv_write_rows_total": prompt_len + gen_len,
        "forced_completions": forced_completions,
        "total_time_ms": float(total_time_ms),
        "page_precision": config.page_precision,
        "rounds_per_block": rounds_per_block,
        "unmasked_per_round": unmasked_per_round,
        "fallback_per_round": fallback_per_round,
        "mean_confidence_per_round": mean_confidence_per_round,
        "rows_saved_vs_baseline": rows_saved_vs_baseline,
        "unmask_positions_per_round": unmask_positions_per_round,
        "block_time_ms": block_time_ms,
        "draft_acceptance_rate": (
            draft_accepted_positions / float(batch * width * num_blocks)
            if config.draft_mode
            else None
        ),
        "rounds_to_fixpoint": rounds_to_fixpoint,
        "changed_per_round": changed_per_round,
    }
    if capture_block_logits:
        stats["block_logits_per_round"] = block_logits_per_round
    return GenerateResult(sequences=sequences, stats=stats), stores
