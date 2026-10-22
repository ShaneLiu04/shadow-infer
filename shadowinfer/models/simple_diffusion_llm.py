"""A minimal, functional Diffusion-style LLM implemented in pure PyTorch.

This model is intentionally small so it can run on CPU inside the test suite,
but it implements the full transformer loop: token/embedding input, positional
and timestep embeddings, stacked transformer blocks with KV-cache support, and
a vocabulary projection that returns logits.
"""

from __future__ import annotations

import math
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from shadowinfer.execution.macro_mode import MacroModeConfig, MacroModeState
from shadowinfer.execution.step_reuse import StepReuseConfig, StepReuseState


def _default_intermediate_dim(hidden_dim: int) -> int:
    return 4 * hidden_dim


class DiffusionTransformerBlock(nn.Module):
    """One transformer block with multi-head self-attention and an FFN."""

    def __init__(
        self,
        hidden_dim: int,
        num_heads: int,
        head_dim: int,
        intermediate_dim: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.intermediate_dim = intermediate_dim

        self.ln1 = nn.LayerNorm(hidden_dim, device=device, dtype=dtype)
        self.q_proj = nn.Linear(hidden_dim, hidden_dim, device=device, dtype=dtype)
        self.k_proj = nn.Linear(hidden_dim, hidden_dim, device=device, dtype=dtype)
        self.v_proj = nn.Linear(hidden_dim, hidden_dim, device=device, dtype=dtype)
        self.o_proj = nn.Linear(hidden_dim, hidden_dim, device=device, dtype=dtype)

        self.ln2 = nn.LayerNorm(hidden_dim, device=device, dtype=dtype)
        self.up_proj = nn.Linear(hidden_dim, intermediate_dim, device=device, dtype=dtype)
        self.down_proj = nn.Linear(intermediate_dim, hidden_dim, device=device, dtype=dtype)

    def forward(
        self,
        hidden: torch.Tensor,
        kv_cache: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        layer_id: int = 0,
        reuse_state: Optional[StepReuseState] = None,
        reuse_config: Optional[StepReuseConfig] = None,
        step_id: int = 0,
        mode_state: Optional[MacroModeState] = None,
        mode_config: Optional[MacroModeConfig] = None,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor], torch.Tensor]:
        """Forward one block.

        Args:
            hidden: Input tensor of shape [batch, seq_len, hidden_dim].
            kv_cache: Optional previous (k, v) tuple of shape
                [batch, num_heads, past_len, head_dim].
            layer_id: Block index (for step reuse bookkeeping).
            reuse_state: Optional cross-step cache (AR013).
            reuse_config: Optional reuse policy (AR013).
            step_id: Current denoising step (warmup gate).
            mode_state: Optional macro-block mode control state (AR015).
            mode_config: Optional macro-block mode control policy (AR015).
            attention_mask: Optional additive attention mask of shape
                [seq_len, seq_len] or [1, 1, seq_len, seq_len] (AR016);
                None keeps the unmasked behavior bit-for-bit.

        Returns:
            A tuple of (updated hidden, new (k, v), attention weights).
        """
        batch, seq_len, _ = hidden.shape
        block_input = hidden

        if attention_mask is not None and (mode_state is not None or reuse_state is not None):
            raise ValueError(
                "attention_mask is not supported together with step-reuse "
                "or mode-control execution paths"
            )

        if mode_state is not None and mode_config is not None:
            return self._forward_mode_control(
                hidden, kv_cache, layer_id, step_id, mode_state, mode_config
            )

        if reuse_state is not None and reuse_config is not None:
            should_reuse, drift = reuse_state.should_reuse(layer_id, hidden, step_id, reuse_config)
            if drift is not None:
                reuse_state.last_drifts[layer_id] = drift
            if should_reuse:
                entry = reuse_state.entries[layer_id]
                reused = entry.hidden_out + (hidden - entry.hidden_in)
                reuse_state.last_reused_layers.add(layer_id)
                reuse_state.attention_reused = True
                attn = reuse_state.last_attention
                if attn is None:
                    attn = torch.empty(())
                return reused, (entry.k, entry.v), attn

        hidden, _, _, k, v, attn_weights = self._compute_block(hidden, kv_cache, attention_mask)

        if reuse_state is not None:
            reuse_state.record(layer_id, block_input, hidden, k, v)
            reuse_state.last_computed_layers.add(layer_id)
            reuse_state.last_attention = attn_weights.detach()

        return hidden, (k, v), attn_weights

    def _compute_block(
        self,
        hidden: torch.Tensor,
        kv_cache: Optional[Tuple[torch.Tensor, torch.Tensor]],
        attention_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Dense attention + FFN over the full sequence.

        Returns ``(hidden_out, k_new, v_new, k_full, v_full, attn_weights)``
        where ``k_new``/``v_new`` are this block's own projections and
        ``k_full``/``v_full`` additionally include the past cache.

        ``attention_mask`` is an additive mask (e.g. block-causal, AR016)
        of shape [seq_len, seq_len] or [1, 1, seq_len, seq_len]; None keeps
        the unmasked behavior bit-for-bit.
        """
        batch, seq_len, _ = hidden.shape

        if attention_mask is not None:
            if kv_cache is not None:
                raise ValueError("attention_mask is not supported together with kv_cache")
            mask_shape = tuple(attention_mask.shape)
            if mask_shape not in ((seq_len, seq_len), (1, 1, seq_len, seq_len)):
                raise ValueError(
                    f"attention_mask must be [S, S] or [1, 1, S, S] with "
                    f"S={seq_len}, got shape {mask_shape}"
                )

        normed = self.ln1(hidden)
        q = self.q_proj(normed).view(batch, seq_len, self.num_heads, self.head_dim)
        k = self.k_proj(normed).view(batch, seq_len, self.num_heads, self.head_dim)
        v = self.v_proj(normed).view(batch, seq_len, self.num_heads, self.head_dim)

        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        k_full, v_full = k, v

        if kv_cache is not None:
            past_k, past_v = kv_cache
            k_full = torch.cat([past_k, k], dim=2)
            v_full = torch.cat([past_v, v], dim=2)

        scores = torch.matmul(q, k_full.transpose(-2, -1)) / math.sqrt(self.head_dim)
        if attention_mask is not None:
            scores = scores + attention_mask
        attn_weights = F.softmax(scores, dim=-1)
        attn_out = torch.matmul(attn_weights, v_full)

        attn_out = attn_out.transpose(1, 2).contiguous().view(batch, seq_len, self.hidden_dim)
        hidden = hidden + self.o_proj(attn_out)

        ffn_out = self.down_proj(F.gelu(self.up_proj(self.ln2(hidden))))
        hidden = hidden + ffn_out

        return hidden, k, v, k_full, v_full, attn_weights

    def _forward_mode_control(
        self,
        hidden: torch.Tensor,
        kv_cache: Optional[Tuple[torch.Tensor, torch.Tensor]],
        layer_id: int,
        step_id: int,
        mode_state: MacroModeState,
        mode_config: MacroModeConfig,
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor], torch.Tensor]:
        """Row-granularity mode-controlled forward (SKIP/DELTA/REFRESH)."""
        batch, seq_len, _ = hidden.shape
        block_input = hidden
        block_size = mode_config.macro_block_size
        modes, active_rows = mode_state.classify(layer_id, hidden, step_id, mode_config)

        if len(active_rows) == seq_len:
            out, k_new, v_new, k_full, v_full, attn_weights = self._compute_block(hidden, kv_cache)
            for macro_idx, start in enumerate(range(0, seq_len, block_size)):
                end = min(start + block_size, seq_len)
                mode_state.record_rows(
                    layer_id,
                    macro_idx,
                    (start, end),
                    block_input[:, start:end, :],
                    out[:, start:end, :],
                    k_new[:, :, start:end, :],
                    v_new[:, :, start:end, :],
                )
            mode_state.last_attention = attn_weights.detach()
            return out, (k_full, v_full), attn_weights

        if kv_cache is not None:
            raise ValueError(
                "mode-controlled partial/skip forward does not support a past "
                "kv_cache (replace lifecycle rewrites the full sequence)"
            )

        macro_ranges = [
            (macro_idx, start, min(start + block_size, seq_len))
            for macro_idx, start in enumerate(range(0, seq_len, block_size))
        ]

        if not active_rows:
            out = torch.empty_like(block_input)
            k_full = torch.empty(
                batch,
                self.num_heads,
                seq_len,
                self.head_dim,
                dtype=block_input.dtype,
                device=block_input.device,
            )
            v_full = torch.empty_like(k_full)
            for macro_idx, start, end in macro_ranges:
                entry = mode_state.entries[(layer_id, macro_idx)]
                out[:, start:end, :] = entry.hidden_out + (
                    block_input[:, start:end, :] - entry.hidden_in
                )
                k_full[:, :, start:end, :] = entry.k
                v_full[:, :, start:end, :] = entry.v
            attn = mode_state.last_attention
            if attn is None:
                attn = torch.empty(())
            return out, (k_full, v_full), attn

        active_set = set(active_rows)
        active = torch.tensor(active_rows, dtype=torch.long, device=hidden.device)
        num_active = int(active.numel())

        h_act = block_input.index_select(1, active)
        normed = self.ln1(h_act)
        q = self.q_proj(normed).view(batch, num_active, self.num_heads, self.head_dim)
        k_act = self.k_proj(normed).view(batch, num_active, self.num_heads, self.head_dim)
        v_act = self.v_proj(normed).view(batch, num_active, self.num_heads, self.head_dim)
        q = q.transpose(1, 2)
        k_act = k_act.transpose(1, 2)
        v_act = v_act.transpose(1, 2)

        k_full = torch.empty(
            batch,
            self.num_heads,
            seq_len,
            self.head_dim,
            dtype=block_input.dtype,
            device=block_input.device,
        )
        v_full = torch.empty_like(k_full)
        out = torch.empty_like(block_input)

        for macro_idx, start, end in macro_ranges:
            local_frozen = [row - start for row in range(start, end) if row not in active_set]
            if not local_frozen:
                continue
            entry = mode_state.entries[(layer_id, macro_idx)]
            local_idx = torch.tensor(local_frozen, dtype=torch.long, device=hidden.device)
            global_idx = local_idx + start
            k_full.index_copy_(2, global_idx, entry.k.index_select(2, local_idx))
            v_full.index_copy_(2, global_idx, entry.v.index_select(2, local_idx))
            frozen_out = entry.hidden_out.index_select(1, local_idx) + (
                block_input.index_select(1, global_idx) - entry.hidden_in.index_select(1, local_idx)
            )
            out.index_copy_(1, global_idx, frozen_out)

        k_full.index_copy_(2, active, k_act)
        v_full.index_copy_(2, active, v_act)

        scores = torch.matmul(q, k_full.transpose(-2, -1)) / math.sqrt(self.head_dim)
        attn_weights = F.softmax(scores, dim=-1)
        attn_out = torch.matmul(attn_weights, v_full)
        attn_out = attn_out.transpose(1, 2).contiguous().view(batch, num_active, self.hidden_dim)
        h_act = h_act + self.o_proj(attn_out)
        h_act = h_act + self.down_proj(F.gelu(self.up_proj(self.ln2(h_act))))
        out.index_copy_(1, active, h_act)

        for macro_idx, start, end in macro_ranges:
            mode = modes[macro_idx]
            if mode == "SKIP":
                continue
            if mode == "REFRESH":
                mode_state.record_rows(
                    layer_id,
                    macro_idx,
                    (start, end),
                    block_input[:, start:end, :],
                    out[:, start:end, :],
                    k_full[:, :, start:end, :],
                    v_full[:, :, start:end, :],
                )
                continue
            entry = mode_state.entries[(layer_id, macro_idx)]
            local_active = [row - start for row in active_rows if start <= row < end]
            local_idx = torch.tensor(local_active, dtype=torch.long, device=hidden.device)
            global_idx = local_idx + start
            entry.hidden_in.index_copy_(
                1, local_idx, block_input.index_select(1, global_idx).detach()
            )
            entry.hidden_out.index_copy_(1, local_idx, out.index_select(1, global_idx).detach())
            entry.k.index_copy_(2, local_idx, k_full.index_select(2, global_idx).detach())
            entry.v.index_copy_(2, local_idx, v_full.index_select(2, global_idx).detach())

        mode_state.last_attention = attn_weights.detach()
        return out, (k_full, v_full), attn_weights


class SimpleDiffusionLLM(nn.Module):
    """Minimal Diffusion LLM with KV-cache injection per transformer block."""

    def __init__(
        self,
        num_layers: int = 4,
        num_heads: int = 4,
        head_dim: int = 32,
        hidden_dim: int = 128,
        vocab_size: int = 1000,
        max_seq_len: int = 128,
        intermediate_dim: Optional[int] = None,
        device: Optional[torch.device] = None,
        dtype: torch.dtype = torch.float32,
    ) -> None:
        super().__init__()
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.hidden_dim = hidden_dim
        self.vocab_size = vocab_size
        self.max_seq_len = max_seq_len
        self.intermediate_dim = intermediate_dim or _default_intermediate_dim(hidden_dim)
        self._device = device if device is not None else torch.device("cpu")
        self._dtype = dtype

        if hidden_dim != num_heads * head_dim:
            raise ValueError(
                f"hidden_dim ({hidden_dim}) must equal num_heads * head_dim "
                f"({num_heads * head_dim})"
            )

        self.embed = nn.Embedding(vocab_size, hidden_dim, device=self._device, dtype=self._dtype)
        self.pos_embed = nn.Embedding(
            max_seq_len, hidden_dim, device=self._device, dtype=self._dtype
        )
        self.time_embed = nn.Sequential(
            nn.Linear(1, hidden_dim, device=self._device, dtype=self._dtype),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim, device=self._device, dtype=self._dtype),
        )

        self.blocks = nn.ModuleList(
            DiffusionTransformerBlock(
                hidden_dim=hidden_dim,
                num_heads=num_heads,
                head_dim=head_dim,
                intermediate_dim=self.intermediate_dim,
                device=self._device,
                dtype=self._dtype,
            )
            for _ in range(num_layers)
        )

        self.out_norm = nn.LayerNorm(hidden_dim, device=self._device, dtype=self._dtype)
        self.output_proj = nn.Linear(
            hidden_dim, vocab_size, bias=False, device=self._device, dtype=self._dtype
        )
        self._fp32_ffn_reference: Optional[list] = None
        self._int8_ffn_stats: Optional[Dict[str, object]] = None

    def forward_step(
        self,
        x: torch.Tensor,
        step_t: int,
        total_steps: int,
        kv_cache: Optional[Dict[int, Tuple[torch.Tensor, torch.Tensor]]] = None,
        reuse_state: Optional[StepReuseState] = None,
        reuse_config: Optional[StepReuseConfig] = None,
        mode_state: Optional[MacroModeState] = None,
        mode_config: Optional[MacroModeConfig] = None,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict[int, Tuple[torch.Tensor, torch.Tensor]], torch.Tensor]:
        """Run one denoising step.

        Args:
            x: Either integer token ids of shape [batch, seq_len] or a float
                tensor of shape [batch, seq_len, hidden_dim].
            step_t: Current denoising step index (0-based).
            total_steps: Total number of denoising steps.
            kv_cache: Optional dictionary mapping layer index to a (k, v) tuple.
            reuse_state: Optional cross-step cache (AR013).
            reuse_config: Optional reuse policy (AR013).
            mode_state: Optional macro-block mode control state (AR015).
            mode_config: Optional macro-block mode control policy (AR015).
            attention_mask: Optional additive attention mask of shape
                [seq_len, seq_len] or [1, 1, seq_len, seq_len] (AR016);
                None keeps the unmasked behavior bit-for-bit.

        Returns:
            A tuple of (logits, updated kv_cache, attention_scores).
        """
        kv_cache = kv_cache or {}

        if x.dim() == 2:
            hidden = self.embed(x.long())
        else:
            hidden = x.to(dtype=self._dtype)

        batch, seq_len, hidden_dim = hidden.shape
        if hidden_dim != self.hidden_dim:
            raise ValueError(
                f"Input hidden_dim ({hidden_dim}) does not match model hidden_dim "
                f"({self.hidden_dim})"
            )

        positions = torch.arange(seq_len, device=hidden.device, dtype=torch.long)
        hidden = hidden + self.pos_embed(positions).unsqueeze(0)

        if total_steps > 0:
            ratio = torch.tensor([[step_t / total_steps]], device=hidden.device, dtype=hidden.dtype)
            hidden = hidden + self.time_embed(ratio).unsqueeze(1)

        updated_kv: Dict[int, Tuple[torch.Tensor, torch.Tensor]] = {}
        attention_scores: torch.Tensor = torch.empty(())
        if reuse_state is not None:
            reuse_state.begin_step()
        if mode_state is not None:
            mode_state.begin_step()
        for layer_id, block in enumerate(self.blocks):
            past = kv_cache.get(layer_id)
            hidden, new_kv, attn = block(
                hidden,
                past,
                layer_id=layer_id,
                reuse_state=reuse_state,
                reuse_config=reuse_config,
                step_id=step_t,
                mode_state=mode_state,
                mode_config=mode_config,
                attention_mask=attention_mask,
            )
            updated_kv[layer_id] = new_kv
            attention_scores = attn

        hidden = self.out_norm(hidden)
        logits = self.output_proj(hidden)
        return logits, updated_kv, attention_scores

    def forward_block(
        self,
        x_block: torch.Tensor,
        prefix_kv: Optional[Dict[int, Tuple[torch.Tensor, torch.Tensor]]] = None,
        prefix_len: int = 0,
        step_t: int = 0,
        total_steps: int = 0,
    ) -> Tuple[torch.Tensor, Dict[int, Tuple[torch.Tensor, torch.Tensor]]]:
        """Forward only the trailing active block of W rows (AR016, FR16.2).

        Computes the full transformer stack for the W active rows only,
        conditioning every layer's attention on the frozen prefix K/V
        supplied by the caller. Active rows attend to prefix + own rows
        (block-causal by construction: prefix K/V never changes).

        Args:
            x_block: [B, W] integer token ids or [B, W, hidden_dim] hidden
                states for the active block rows.
            prefix_kv: Per-layer (k, v) prefix tensors of shape
                [B, num_heads, prefix_len, head_dim]; may be None only when
                ``prefix_len == 0``.
            prefix_len: Number of frozen prefix rows L (>= 0). Positions for
                the active rows are [L, L+W).
            step_t: Current denoising step index (0-based).
            total_steps: Total denoising steps; the generation protocol
                fixes 0 (no time embedding), matching the baseline arm.

        Returns:
            A tuple of (logits [B, W, vocab_size], block_kv mapping each
            layer to its active-row (k, v) new projections
            [B, num_heads, W, head_dim]).

        Raises:
            ValueError: If ``x_block`` is not 2-D/3-D, its hidden dim does
                not match the model, ``prefix_len`` is negative,
                ``prefix_kv`` layer set is incomplete, or any prefix k/v
                shape deviates from [B, num_heads, prefix_len, head_dim].
        """
        if prefix_len < 0:
            raise ValueError(f"prefix_len must be >= 0, got {prefix_len}")
        if x_block.dim() == 2:
            if x_block.dtype not in (torch.long, torch.int):
                raise ValueError(
                    "2-D x_block must be integer token ids [B, W]; float "
                    "hidden states must be 3-D [B, W, hidden_dim]"
                )
            hidden = self.embed(x_block.long())
        elif x_block.dim() == 3:
            hidden = x_block.to(dtype=self._dtype)
        else:
            raise ValueError(
                f"x_block must be [B, W] token ids or [B, W, hidden_dim] "
                f"hidden states, got dim={x_block.dim()}"
            )

        batch, width, hidden_dim = hidden.shape
        if hidden_dim != self.hidden_dim:
            raise ValueError(
                f"Input hidden_dim ({hidden_dim}) does not match model hidden_dim "
                f"({self.hidden_dim})"
            )

        if prefix_kv is None:
            if prefix_len != 0:
                raise ValueError(f"prefix_kv is required when prefix_len={prefix_len} > 0")
        else:
            expected_layers = set(range(self.num_layers))
            if set(prefix_kv.keys()) != expected_layers:
                raise ValueError(
                    f"prefix_kv must cover exactly layers {sorted(expected_layers)}, "
                    f"got {sorted(prefix_kv.keys())}"
                )
            expected_shape = (batch, self.num_heads, prefix_len, self.head_dim)
            for layer_id, (past_k, past_v) in prefix_kv.items():
                if tuple(past_k.shape) != expected_shape or tuple(past_v.shape) != expected_shape:
                    raise ValueError(
                        f"prefix_kv[{layer_id}] must have shape {expected_shape}, "
                        f"got k={tuple(past_k.shape)}, v={tuple(past_v.shape)}"
                    )

        positions = torch.arange(
            prefix_len, prefix_len + width, device=hidden.device, dtype=torch.long
        )
        hidden = hidden + self.pos_embed(positions).unsqueeze(0)

        if total_steps > 0:
            ratio = torch.tensor([[step_t / total_steps]], device=hidden.device, dtype=hidden.dtype)
            hidden = hidden + self.time_embed(ratio).unsqueeze(1)

        block_kv: Dict[int, Tuple[torch.Tensor, torch.Tensor]] = {}
        for layer_id, block in enumerate(self.blocks):
            past = prefix_kv.get(layer_id) if prefix_kv is not None else None
            hidden, k_new, v_new, _, _, _ = block._compute_block(hidden, past)
            block_kv[layer_id] = (k_new, v_new)

        hidden = self.out_norm(hidden)
        logits = self.output_proj(hidden)
        return logits, block_kv

    def enable_int8_ffn(self, keep_fp32_reference: bool = False) -> Dict[str, object]:
        """Replace every FFN up/down projection with an executable INT8 linear.

        By default the original FP32 modules are released so their weights can be
        reclaimed. Set ``keep_fp32_reference=True`` for golden comparisons and a
        later :meth:`restore_fp32_ffn`.
        """
        from shadowinfer.ffn_optimizer.int8_linear import INT8Linear

        references = [] if keep_fp32_reference else None
        modules = 0
        weight_bytes = 0
        dense_bytes = 0
        fallback = False

        for block in self.blocks:
            for name in ("up_proj", "down_proj"):
                module = getattr(block, name)
                if isinstance(module, INT8Linear):
                    modules += 1
                    weight_bytes += module.memory_bytes()
                    dense_bytes += module.dense_equivalent_bytes()
                    fallback = fallback or not module.using_int_mm
                    continue
                if not isinstance(module, nn.Linear):
                    raise TypeError(f"{name} must be nn.Linear, got {type(module).__name__}")
                quantized = INT8Linear.from_linear(module)
                if references is not None:
                    references.append((block, name, module))
                setattr(block, name, quantized)
                modules += 1
                weight_bytes += quantized.memory_bytes()
                dense_bytes += quantized.dense_equivalent_bytes()
                fallback = fallback or not quantized.using_int_mm

        if references is not None:
            # Repeated enable(keep=True) on already-quantized layers captures
            # nothing new; preserve the original reference list for restore.
            existing = self._fp32_ffn_reference or []
            self._fp32_ffn_reference = (existing + references) if references else existing
        else:
            # Explicitly release any earlier reference so FP32 weights can be freed.
            self._fp32_ffn_reference = None
        stats: Dict[str, object] = {
            "layers": len(self.blocks),
            "modules": modules,
            "weight_bytes": weight_bytes,
            "dense_bytes": dense_bytes,
            "fallback": fallback,
        }
        self._int8_ffn_stats = stats
        return dict(stats)

    def restore_fp32_ffn(self) -> None:
        """Restore the FP32 FFN modules captured by ``keep_fp32_reference=True``."""
        references = self._fp32_ffn_reference
        if not references:
            raise RuntimeError(
                "no FP32 FFN reference available; enable_int8_ffn("
                "keep_fp32_reference=True) must be called before restore"
            )
        for block, name, module in references:
            setattr(block, name, module)
        self._fp32_ffn_reference = None
        self._int8_ffn_stats = None

    @property
    def int8_ffn_stats(self) -> Optional[Dict[str, object]]:
        if self._int8_ffn_stats is None:
            return None
        from shadowinfer.ffn_optimizer.int8_linear import INT8Linear

        stats = dict(self._int8_ffn_stats)
        stats["fallback"] = any(
            isinstance(getattr(block, name), INT8Linear) and not getattr(block, name).using_int_mm
            for block in self.blocks
            for name in ("up_proj", "down_proj")
        )
        return stats

    @property
    def device(self) -> torch.device:
        return self._device

    @property
    def dtype(self) -> torch.dtype:
        return self._dtype
