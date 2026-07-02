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
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor], torch.Tensor]:
        """Forward one block.

        Args:
            hidden: Input tensor of shape [batch, seq_len, hidden_dim].
            kv_cache: Optional previous (k, v) tuple of shape
                [batch, num_heads, past_len, head_dim].

        Returns:
            A tuple of (updated hidden, new (k, v), attention weights).
        """
        batch, seq_len, _ = hidden.shape

        normed = self.ln1(hidden)
        q = self.q_proj(normed).view(batch, seq_len, self.num_heads, self.head_dim)
        k = self.k_proj(normed).view(batch, seq_len, self.num_heads, self.head_dim)
        v = self.v_proj(normed).view(batch, seq_len, self.num_heads, self.head_dim)

        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        if kv_cache is not None:
            past_k, past_v = kv_cache
            k = torch.cat([past_k, k], dim=2)
            v = torch.cat([past_v, v], dim=2)

        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        attn_weights = F.softmax(scores, dim=-1)
        attn_out = torch.matmul(attn_weights, v)

        attn_out = attn_out.transpose(1, 2).contiguous().view(batch, seq_len, self.hidden_dim)
        hidden = hidden + self.o_proj(attn_out)

        ffn_out = self.down_proj(F.gelu(self.up_proj(self.ln2(hidden))))
        hidden = hidden + ffn_out

        return hidden, (k, v), attn_weights


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

    def forward_step(
        self,
        x: torch.Tensor,
        step_t: int,
        total_steps: int,
        kv_cache: Optional[Dict[int, Tuple[torch.Tensor, torch.Tensor]]] = None,
    ) -> Tuple[torch.Tensor, Dict[int, Tuple[torch.Tensor, torch.Tensor]], torch.Tensor]:
        """Run one denoising step.

        Args:
            x: Either integer token ids of shape [batch, seq_len] or a float
                tensor of shape [batch, seq_len, hidden_dim].
            step_t: Current denoising step index (0-based).
            total_steps: Total number of denoising steps.
            kv_cache: Optional dictionary mapping layer index to a (k, v) tuple.

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
            ratio = torch.tensor(
                [[step_t / total_steps]], device=hidden.device, dtype=hidden.dtype
            )
            hidden = hidden + self.time_embed(ratio).unsqueeze(1)

        updated_kv: Dict[int, Tuple[torch.Tensor, torch.Tensor]] = {}
        attention_scores: torch.Tensor = torch.empty(())
        for layer_id, block in enumerate(self.blocks):
            past = kv_cache.get(layer_id)
            hidden, new_kv, attn = block(hidden, past)
            updated_kv[layer_id] = new_kv
            attention_scores = attn

        hidden = self.out_norm(hidden)
        logits = self.output_proj(hidden)
        return logits, updated_kv, attention_scores

    @property
    def device(self) -> torch.device:
        return self._device

    @property
    def dtype(self) -> torch.dtype:
        return self._dtype
