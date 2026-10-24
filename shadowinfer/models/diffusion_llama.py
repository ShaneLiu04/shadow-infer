"""Real-checkpoint diffusion LLaMA (AR018): config parsing, checkpoint
loading, and the module/attribute skeleton.

``load_diffusion_llama`` loads HF-format diffusion LLM checkpoints
(``config.json`` plus ``model.safetensors`` or a sharded layout with
``model.safetensors.index.json``) following the Llama/Qwen weight naming
shared by LLaDA-8B and Dream-7B (design.md D9). All validation (config
fields, weight-name set, per-tensor shapes, index consistency) happens
before any module construction, so a rejected checkpoint never yields a
partially loaded model (design 4.2.2-1: zero-partial-load).

The modeling layer (``forward_step`` / ``forward_block``) lands with
T003; this module delivers the loader plane (FR18.1). The module
structure below exists so that the exposed parameter multiset equals the
on-disk tensor multiset bitwise, with tied ``lm_head`` sharing the
embedding parameter and RoPE ``inv_freq`` kept as a non-persistent
buffer (never a parameter).

PRESETS records the literature-scope architecture parameters of the two
real targets; it is a documentation/assertion object only -- the loading
path is fully config-driven (design D13).
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn.functional as F
from safetensors.torch import load_file
from torch import nn
from torch.utils import swap_tensors

__all__ = [
    "PRESETS",
    "DiffusionLlama",
    "DiffusionLlamaConfig",
    "load_diffusion_llama",
]

# Literature-scope architecture parameters (2026-10-24, UNVERIFIED on
# this machine: the corporate gateway blocks huggingface.co / arxiv /
# github, so these follow the published initialization-lineage claims
# documented in design.md 4.2.2). Must stay field-by-field identical to
# specs/archive/AR018-real-dllm-weights/fixture_gen.py PRESET_TEMPLATES;
# verify against the real config.json on the GPU server (s13) and record
# any mismatch as a design erratum.
PRESETS: Dict[str, Dict[str, Any]] = {
    "dream-7b": {
        "model_type": "dream",
        "num_hidden_layers": 28,
        "hidden_size": 3584,
        "num_attention_heads": 28,
        "num_key_value_heads": 4,
        "head_dim": 128,
        "intermediate_size": 18944,
        "vocab_size": 151936,
        "mask_token_id": 151669,
        "max_position_embeddings": 131072,
        "rope_theta": 1000000.0,
        "rms_norm_eps": 1e-6,
        "attention_bias": True,
        "tie_word_embeddings": False,
    },
    "llada-8b": {
        "model_type": "llada",
        "num_hidden_layers": 32,
        "hidden_size": 4096,
        "num_attention_heads": 32,
        "num_key_value_heads": 8,
        "head_dim": 128,
        "intermediate_size": 14336,
        "vocab_size": 126464,
        "mask_token_id": 126336,
        "max_position_embeddings": 8192,
        "rope_theta": 500000.0,
        "rms_norm_eps": 1e-5,
        "attention_bias": False,
        "tie_word_embeddings": False,
    },
}

_REQUIRED_CONFIG_FIELDS: Tuple[str, ...] = (
    "vocab_size",
    "hidden_size",
    "num_hidden_layers",
    "num_attention_heads",
    "num_key_value_heads",
    "intermediate_size",
    "max_position_embeddings",
)


@dataclass(frozen=True)
class DiffusionLlamaConfig:
    """Parsed architecture configuration (config.json subset).

    ``head_dim`` is ``None`` when config.json omits it; it is then
    derived as ``hidden_size // num_attention_heads`` (design D15).
    """

    vocab_size: int
    hidden_size: int
    num_hidden_layers: int
    num_attention_heads: int
    num_key_value_heads: int
    intermediate_size: int
    max_position_embeddings: int
    head_dim: Optional[int] = None
    rope_theta: float = 10000.0
    rms_norm_eps: float = 1e-5
    attention_bias: bool = False
    mask_token_id: Optional[int] = None
    tie_word_embeddings: bool = False


def _parse_config(checkpoint_dir: Path) -> DiffusionLlamaConfig:
    """Read and validate config.json (fail-fast, field-locating errors)."""
    config_path = checkpoint_dir / "config.json"
    if not config_path.is_file():
        raise ValueError(f"config.json not found in checkpoint directory: {config_path}")
    try:
        raw = json.loads(config_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"config.json is not valid JSON: {exc}") from exc
    if not isinstance(raw, dict):
        raise ValueError(f"config.json must contain a JSON object, got {type(raw).__name__}")

    missing = [field for field in _REQUIRED_CONFIG_FIELDS if field not in raw]
    if missing:
        raise ValueError(f"config.json is missing required field(s): {', '.join(missing)}")

    vocab_size = int(raw["vocab_size"])
    hidden_size = int(raw["hidden_size"])
    heads = int(raw["num_attention_heads"])
    kv_heads = int(raw["num_key_value_heads"])

    if heads < 1:
        raise ValueError(f"num_attention_heads ({heads}) must be >= 1")
    if kv_heads < 1:
        raise ValueError(f"num_key_value_heads ({kv_heads}) must be >= 1")
    if kv_heads > heads:
        raise ValueError(
            f"num_key_value_heads ({kv_heads}) must not exceed " f"num_attention_heads ({heads})"
        )
    if heads % kv_heads != 0:
        raise ValueError(
            f"num_attention_heads ({heads}) must be a multiple of "
            f"num_key_value_heads ({kv_heads}) for GQA repeat"
        )

    head_dim: Optional[int] = None
    if raw.get("head_dim") is not None:
        head_dim = int(raw["head_dim"])
        if head_dim < 1:
            raise ValueError(f"head_dim ({head_dim}) must be >= 1")
    resolved_head_dim = head_dim if head_dim is not None else hidden_size // heads
    if heads * resolved_head_dim != hidden_size:
        raise ValueError(
            f"hidden_size ({hidden_size}) must equal num_attention_heads * "
            f"head_dim ({heads} * {resolved_head_dim}) for the AR018 "
            "loader contract"
        )

    mask_token_id: Optional[int] = None
    if raw.get("mask_token_id") is not None:
        mask_token_id = int(raw["mask_token_id"])
        if not 0 <= mask_token_id < vocab_size:
            raise ValueError(
                f"mask_token_id ({mask_token_id}) must be within " f"[0, vocab_size={vocab_size})"
            )

    return DiffusionLlamaConfig(
        vocab_size=vocab_size,
        hidden_size=hidden_size,
        num_hidden_layers=int(raw["num_hidden_layers"]),
        num_attention_heads=heads,
        num_key_value_heads=kv_heads,
        intermediate_size=int(raw["intermediate_size"]),
        max_position_embeddings=int(raw["max_position_embeddings"]),
        head_dim=head_dim,
        rope_theta=float(raw.get("rope_theta", 10000.0)),
        rms_norm_eps=float(raw.get("rms_norm_eps", 1e-5)),
        attention_bias=bool(raw.get("attention_bias", False)),
        mask_token_id=mask_token_id,
        tie_word_embeddings=bool(raw.get("tie_word_embeddings", False)),
    )


def _expected_weight_shapes(
    config: DiffusionLlamaConfig,
) -> Dict[str, Tuple[int, ...]]:
    """Derive the expected {weight_name: shape} map from the config.

    Single naming map for both LLaDA-8B (Llama-style) and Dream-7B
    (Qwen-style) checkpoints (design D9). The expected set is derived,
    never hardcoded to a checkpoint's index.
    """
    head_dim = config.head_dim
    if head_dim is None:
        head_dim = config.hidden_size // config.num_attention_heads
    qkv_out = config.num_attention_heads * head_dim
    kv_out = config.num_key_value_heads * head_dim
    shapes: Dict[str, Tuple[int, ...]] = {
        "model.embed_tokens.weight": (config.vocab_size, config.hidden_size),
    }
    for layer in range(config.num_hidden_layers):
        prefix = f"model.layers.{layer}"
        shapes[f"{prefix}.self_attn.q_proj.weight"] = (qkv_out, config.hidden_size)
        shapes[f"{prefix}.self_attn.k_proj.weight"] = (kv_out, config.hidden_size)
        shapes[f"{prefix}.self_attn.v_proj.weight"] = (kv_out, config.hidden_size)
        shapes[f"{prefix}.self_attn.o_proj.weight"] = (qkv_out, config.hidden_size)
        if config.attention_bias:
            shapes[f"{prefix}.self_attn.q_proj.bias"] = (qkv_out,)
            shapes[f"{prefix}.self_attn.k_proj.bias"] = (kv_out,)
            shapes[f"{prefix}.self_attn.v_proj.bias"] = (kv_out,)
        shapes[f"{prefix}.mlp.gate_proj.weight"] = (
            config.intermediate_size,
            config.hidden_size,
        )
        shapes[f"{prefix}.mlp.up_proj.weight"] = (
            config.intermediate_size,
            config.hidden_size,
        )
        shapes[f"{prefix}.mlp.down_proj.weight"] = (
            config.hidden_size,
            config.intermediate_size,
        )
        shapes[f"{prefix}.input_layernorm.weight"] = (config.hidden_size,)
        shapes[f"{prefix}.post_attention_layernorm.weight"] = (config.hidden_size,)
    shapes["model.norm.weight"] = (config.hidden_size,)
    if not config.tie_word_embeddings:
        shapes["lm_head.weight"] = (config.vocab_size, config.hidden_size)
    return shapes


def _load_weight_dict(checkpoint_dir: Path) -> Dict[str, torch.Tensor]:
    """Load the raw checkpoint tensors (single-file or sharded layout).

    Sharded layout: the union of shard-file contents must match the
    index weight_map exactly (orphans and unresolved entries are both
    locating errors), and every referenced shard file must exist.
    """
    index_path = checkpoint_dir / "model.safetensors.index.json"
    if index_path.is_file():
        try:
            index = json.loads(index_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise ValueError(f"model.safetensors.index.json is not valid JSON: {exc}") from exc
        weight_map = index.get("weight_map") if isinstance(index, dict) else None
        if not isinstance(weight_map, dict) or not weight_map:
            raise ValueError(
                f"model.safetensors.index.json lacks a non-empty weight_map: " f"{index_path}"
            )
        merged: Dict[str, torch.Tensor] = {}
        for filename in sorted(set(weight_map.values())):
            shard_path = checkpoint_dir / filename
            if not shard_path.is_file():
                raise ValueError(f"shard file not found: {shard_path}")
            merged.update(load_file(str(shard_path)))
        orphans = sorted(set(merged) - set(weight_map))
        if orphans:
            raise ValueError(
                f"shard files carry weights absent from the index " f"weight_map: {orphans}"
            )
        unresolved = sorted(set(weight_map) - set(merged))
        if unresolved:
            raise ValueError(
                f"index weight_map references weights missing from the "
                f"shard files: {unresolved}"
            )
        return merged

    single_path = checkpoint_dir / "model.safetensors"
    if not single_path.is_file():
        raise ValueError(f"weight file not found: {single_path}")
    return dict(load_file(str(single_path)))


def _compute_inv_freq(config: DiffusionLlamaConfig, head_dim: int) -> torch.Tensor:
    """Precompute RoPE inverse frequencies (float32, design 4.2.2)."""
    exponents = torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim
    return 1.0 / (config.rope_theta**exponents)


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    """Standard rotate-half helper: [-x2, x1] over the two halves."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def _repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """GQA expand [B, H_kv, T, D] -> [B, H_kv * n_rep, T, D].

    Each KV head is repeated ``n_rep`` consecutive times so query head h
    attends to KV head h // n_rep; identity when ``n_rep == 1`` (MHA,
    zero special-casing, design D2).
    """
    if n_rep == 1:
        return hidden_states
    batch, num_kv_heads, seq_len, head_dim = hidden_states.shape
    expanded = hidden_states[:, :, None, :, :].expand(batch, num_kv_heads, n_rep, seq_len, head_dim)
    return expanded.reshape(batch, num_kv_heads * n_rep, seq_len, head_dim)


class DiffusionLlamaRMSNorm(nn.Module):
    """RMSNorm with a single ``weight`` parameter (Llama/Qwen naming)."""

    def __init__(self, hidden_size: int, eps: float) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.eps = eps

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        return hidden_states / torch.sqrt(variance + self.eps) * self.weight


class DiffusionLlamaMLP(nn.Module):
    """SwiGLU FFN: gate/up/down projections, no biases."""

    def __init__(self, config: DiffusionLlamaConfig) -> None:
        super().__init__()
        self.gate_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.up_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.down_proj = nn.Linear(config.intermediate_size, config.hidden_size, bias=False)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.down_proj(F.silu(self.gate_proj(hidden_states)) * self.up_proj(hidden_states))


class DiffusionLlamaAttention(nn.Module):
    """GQA attention: q/k/v/o projections (qkv biases per config).

    Returns, alongside the attention output, this call's own KV
    projections at KV-head-native width [B, num_kv_heads, T, D] AFTER
    RoPE (design D2: the pipeline consumes post-RoPE KV).
    """

    def __init__(self, config: DiffusionLlamaConfig, head_dim: int) -> None:
        super().__init__()
        self.num_heads = config.num_attention_heads
        self.num_kv_heads = config.num_key_value_heads
        self.head_dim = head_dim
        qkv_out = config.num_attention_heads * head_dim
        kv_out = config.num_key_value_heads * head_dim
        self.q_proj = nn.Linear(config.hidden_size, qkv_out, bias=config.attention_bias)
        self.k_proj = nn.Linear(config.hidden_size, kv_out, bias=config.attention_bias)
        self.v_proj = nn.Linear(config.hidden_size, kv_out, bias=config.attention_bias)
        self.o_proj = nn.Linear(qkv_out, config.hidden_size, bias=False)

    def forward(
        self,
        hidden_states: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        past_kv: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        batch, seq_len, _ = hidden_states.shape
        n_rep = self.num_heads // self.num_kv_heads

        q = self.q_proj(hidden_states)
        k = self.k_proj(hidden_states)
        v = self.v_proj(hidden_states)
        q = q.view(batch, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch, seq_len, self.num_kv_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch, seq_len, self.num_kv_heads, self.head_dim).transpose(1, 2)

        q = q * cos + _rotate_half(q) * sin
        k = k * cos + _rotate_half(k) * sin

        if past_kv is not None:
            past_k, past_v = past_kv
            k_all = torch.cat((past_k, k), dim=2)
            v_all = torch.cat((past_v, v), dim=2)
        else:
            k_all, v_all = k, v

        k_exp = _repeat_kv(k_all, n_rep)
        v_exp = _repeat_kv(v_all, n_rep)
        scores = torch.matmul(q, k_exp.transpose(-2, -1)) / math.sqrt(self.head_dim)
        if attention_mask is not None:
            scores = scores + attention_mask
        probs = torch.softmax(scores, dim=-1)
        attn = torch.matmul(probs, v_exp)
        attn = attn.transpose(1, 2).reshape(batch, seq_len, self.num_heads * self.head_dim)
        return self.o_proj(attn), k, v


class DiffusionLlamaDecoderLayer(nn.Module):
    """One decoder layer: attention + SwiGLU + dual RMSNorm + residual."""

    def __init__(self, config: DiffusionLlamaConfig, head_dim: int) -> None:
        super().__init__()
        self.self_attn = DiffusionLlamaAttention(config, head_dim)
        self.mlp = DiffusionLlamaMLP(config)
        self.input_layernorm = DiffusionLlamaRMSNorm(config.hidden_size, config.rms_norm_eps)
        self.post_attention_layernorm = DiffusionLlamaRMSNorm(
            config.hidden_size, config.rms_norm_eps
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        past_kv: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        normed = self.input_layernorm(hidden_states)
        attn_out, k, v = self.self_attn(normed, cos, sin, attention_mask, past_kv)
        hidden_states = hidden_states + attn_out
        normed = self.post_attention_layernorm(hidden_states)
        hidden_states = hidden_states + self.mlp(normed)
        return hidden_states, k, v


class DiffusionLlamaModel(nn.Module):
    """Transformer body: embedding, decoder layers, final norm."""

    def __init__(self, config: DiffusionLlamaConfig, head_dim: int) -> None:
        super().__init__()
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList(
            DiffusionLlamaDecoderLayer(config, head_dim) for _ in range(config.num_hidden_layers)
        )
        self.norm = DiffusionLlamaRMSNorm(config.hidden_size, config.rms_norm_eps)

    def forward(
        self,
        input_ids: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        prefix_kv: Optional[Dict[int, Tuple[torch.Tensor, torch.Tensor]]] = None,
    ) -> Tuple[torch.Tensor, Dict[int, Tuple[torch.Tensor, torch.Tensor]]]:
        hidden_states = self.embed_tokens(input_ids)
        kv: Dict[int, Tuple[torch.Tensor, torch.Tensor]] = {}
        for layer_id, layer in enumerate(self.layers):
            past_kv = prefix_kv.get(layer_id) if prefix_kv is not None else None
            hidden_states, k, v = layer(hidden_states, cos, sin, attention_mask, past_kv)
            kv[layer_id] = (k, v)
        hidden_states = self.norm(hidden_states)
        return hidden_states, kv


class DiffusionLlama(nn.Module):
    """Real-checkpoint diffusion LLaMA execution plane (AR018).

    Attribute contract (blockwise pipeline hard contract, srs FR18.2):
    ``num_layers``/``num_heads``/``num_kv_heads``/``head_dim``/
    ``max_seq_len``/``vocab_size``/``device``/``mask_token_id``/
    ``config``. ``num_kv_heads`` is the native KV head count (design D2):
    forward_step/forward_block expose KV at GQA-native width.
    """

    def __init__(self, config: DiffusionLlamaConfig) -> None:
        super().__init__()
        self.config = config
        if config.head_dim is not None:
            head_dim = config.head_dim
        else:
            head_dim = config.hidden_size // config.num_attention_heads
        self._resolved_head_dim = head_dim
        self.model = DiffusionLlamaModel(config, head_dim)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.register_buffer("inv_freq", _compute_inv_freq(config, head_dim), persistent=False)

    @property
    def num_layers(self) -> int:
        return self.config.num_hidden_layers

    @property
    def num_heads(self) -> int:
        return self.config.num_attention_heads

    @property
    def num_kv_heads(self) -> int:
        return self.config.num_key_value_heads

    @property
    def head_dim(self) -> int:
        return self._resolved_head_dim

    @property
    def max_seq_len(self) -> int:
        return self.config.max_position_embeddings

    @property
    def vocab_size(self) -> int:
        return self.config.vocab_size

    @property
    def mask_token_id(self) -> Optional[int]:
        return self.config.mask_token_id

    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device

    def _cos_sin(self, positions: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """cos/sin tables for absolute ``positions`` (rotate-half, D6).

        Computed in float32 (the inv_freq buffer dtype), then cast to the
        parameter dtype so bf16 checkpoints keep bf16 activations.
        """
        freqs = torch.outer(positions.to(self.inv_freq.dtype), self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        param_dtype = self.model.embed_tokens.weight.dtype
        return emb.cos().to(param_dtype), emb.sin().to(param_dtype)

    def _validate_input_ids(self, x: torch.Tensor, name: str) -> Tuple[int, int]:
        """Validate [B, S] integer token ids; return (batch, seq_len)."""
        if x.dim() != 2:
            raise ValueError(
                f"{name} must be [batch, seq_len] integer token ids, " f"got dim={x.dim()}"
            )
        if x.dtype not in (torch.long, torch.int):
            raise ValueError(
                f"{name} must be an integer tensor of token ids, " f"got dtype={x.dtype}"
            )
        seq_len = x.shape[1]
        if seq_len < 1:
            raise ValueError(f"{name} seq_len must be >= 1, got {seq_len}")
        if seq_len > self.max_seq_len:
            raise ValueError(f"{name} seq_len ({seq_len}) exceeds max_seq_len ({self.max_seq_len})")
        return x.shape[0], seq_len

    def forward_step(
        self,
        x: torch.Tensor,
        step_t: Optional[int] = None,
        total_steps: Optional[int] = None,
        kv_cache: Optional[Any] = None,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict[int, Tuple[torch.Tensor, torch.Tensor]], None]:
        """Full-sequence forward (prefill / full recompute / remasking arm).

        Args:
            x: [B, S] integer token ids.
            step_t/total_steps: Accepted but ignored (dLLMs carry no
                timestep semantics; pipeline-compat placeholders).
            kv_cache: Must be None. dLLM blockwise serving rejects
                AR-style incremental KV semantics (design D11).
            attention_mask: Optional additive mask, [S, S] or
                [1, 1, S, S]; None = fully bidirectional (design D5(a)).

        Returns:
            ``(logits [B, S, V], kv, None)`` where ``kv`` maps every layer
            to its post-RoPE (k, v) at KV-head-native width
            ``[B, num_kv_heads, S, head_dim]`` (design D2).

        Raises:
            ValueError: If ``kv_cache`` is not None, ``x`` is not [B, S]
                integer ids within max_seq_len, or ``attention_mask`` is
                not [S, S] / [1, 1, S, S].
        """
        if kv_cache is not None:
            raise ValueError(
                "DiffusionLlama.forward_step does not accept kv_cache: "
                "dLLM blockwise serving rejects AR-style incremental KV "
                "semantics (design D11); use forward_block with prefix_kv"
            )
        _, seq_len = self._validate_input_ids(x, "x")

        mask4: Optional[torch.Tensor] = None
        if attention_mask is not None:
            if attention_mask.dim() == 2:
                mask4 = attention_mask[None, None, :, :]
            elif attention_mask.dim() == 4:
                mask4 = attention_mask
            else:
                raise ValueError(
                    f"attention_mask must be [S, S] or [1, 1, S, S], got "
                    f"shape {tuple(attention_mask.shape)}"
                )
            if tuple(mask4.shape[-2:]) != (seq_len, seq_len):
                raise ValueError(
                    f"attention_mask must cover [S, S] = "
                    f"({seq_len}, {seq_len}), got {tuple(mask4.shape[-2:])}"
                )

        positions = torch.arange(seq_len, device=self.device)
        cos, sin = self._cos_sin(positions)
        hidden_states, kv = self.model(x, cos, sin, mask4, None)
        logits = self.lm_head(hidden_states)
        return logits, kv, None

    def forward_block(
        self,
        x_block: torch.Tensor,
        prefix_kv: Optional[Dict[int, Tuple[torch.Tensor, torch.Tensor]]] = None,
        prefix_len: int = 0,
        step_t: int = 0,
        total_steps: int = 0,
    ) -> Tuple[torch.Tensor, Dict[int, Tuple[torch.Tensor, torch.Tensor]]]:
        """Block-rows conditional forward (design D5(c)): compute only the
        tail W rows conditioned on the frozen ``prefix_kv``.

        Active rows sit at absolute positions ``[prefix_len, prefix_len +
        W)`` (RoPE offset, design D6) and attend to ``cat(prefix, block)``
        with no mask (the frozen prefix is the condition, not a peer).

        Args:
            x_block: [B, W] integer token ids for the active rows.
            prefix_kv: Per-layer post-RoPE (k, v) at KV-head-native width
                ``[B, num_kv_heads, prefix_len, head_dim]``; may be None
                only when ``prefix_len == 0``.
            prefix_len: Number of frozen prefix rows (>= 0).
            step_t/total_steps: Accepted but ignored.

        Returns:
            ``(logits [B, W, V], block_kv)`` where ``block_kv`` maps every
            layer to this call's own post-RoPE (k, v) rows at
            ``[B, num_kv_heads, W, head_dim]``.

        Raises:
            ValueError: If ``x_block`` is not [B, W] integer ids within
                max_seq_len, ``prefix_len`` is negative, ``prefix_kv`` is
                missing while ``prefix_len > 0``, its layer set is
                incomplete, or any k/v shape deviates from
                ``[B, num_kv_heads, prefix_len, head_dim]``.
        """
        if prefix_len < 0:
            raise ValueError(f"prefix_len must be >= 0, got {prefix_len}")
        batch, width = self._validate_input_ids(x_block, "x_block")

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
            expected_shape = (batch, self.num_kv_heads, prefix_len, self.head_dim)
            for layer_id, (past_k, past_v) in prefix_kv.items():
                if tuple(past_k.shape) != expected_shape or tuple(past_v.shape) != expected_shape:
                    raise ValueError(
                        f"prefix_kv[{layer_id}] must have shape {expected_shape}, "
                        f"got k={tuple(past_k.shape)}, v={tuple(past_v.shape)}"
                    )

        positions = torch.arange(prefix_len, prefix_len + width, device=self.device)
        cos, sin = self._cos_sin(positions)
        hidden_states, block_kv = self.model(x_block, cos, sin, None, prefix_kv)
        logits = self.lm_head(hidden_states)
        return logits, block_kv


def load_diffusion_llama(
    path: str | Path,
    dtype: str = "float32",
    device: str | torch.device = "cpu",
) -> DiffusionLlama:
    """Load a real HF-format diffusion LLM checkpoint into DiffusionLlama.

    Args:
        path: Checkpoint directory containing ``config.json`` and either
            ``model.safetensors`` or ``model.safetensors.index.json`` plus
            sharded safetensors files.
        dtype: ``"float32"`` (default) casts all weights to float32;
            ``"auto"`` keeps each tensor's on-disk dtype.
        device: Target device for all parameters.

    Returns:
        The loaded :class:`DiffusionLlama` (train mode; call ``.eval()``
        for inference).

    Raises:
        ValueError: On missing directory/config/files, missing or extra
            config fields (field-locating message), invalid dimensions,
            missing/extra checkpoint weights (name-locating message),
            index/shard inconsistency, shape mismatch (tensor-locating
            message), or an invalid ``dtype`` argument. All validation
            precedes module construction (zero partial load).
    """
    if dtype not in ("float32", "auto"):
        raise ValueError(f"dtype must be 'float32' or 'auto', got {dtype!r}")
    checkpoint_dir = Path(path)
    if not checkpoint_dir.is_dir():
        raise ValueError(f"checkpoint directory does not exist: {checkpoint_dir}")

    config = _parse_config(checkpoint_dir)
    weights = _load_weight_dict(checkpoint_dir)

    expected_shapes = _expected_weight_shapes(config)
    missing = sorted(set(expected_shapes) - set(weights))
    if missing:
        raise ValueError(f"checkpoint is missing expected weights: {missing}")
    extra = sorted(set(weights) - set(expected_shapes))
    if extra:
        raise ValueError(f"checkpoint carries unexpected weights: {extra}")
    for name in sorted(expected_shapes):
        expected_shape = expected_shapes[name]
        actual_shape = tuple(weights[name].shape)
        if actual_shape != expected_shape:
            raise ValueError(
                f"weight '{name}' has shape {actual_shape}, expected "
                f"{expected_shape} (from config.json)"
            )

    # Construct on the meta device (no random init cost for real 7B/8B
    # checkpoints), then bind the validated on-disk tensors directly via
    # swap_tensors (the standard meta-to-real materialization API).
    with torch.device("meta"):
        model = DiffusionLlama(config)
    params = dict(model.named_parameters())
    if config.tie_word_embeddings:
        expected_param_names = {name for name in params if name != "lm_head.weight"}
    else:
        expected_param_names = set(params)
    unbound = sorted(expected_param_names - set(weights))
    if unbound:
        raise ValueError(f"internal error: unbound model parameters: {unbound}")

    target = torch.float32 if dtype == "float32" else None
    for name, tensor in weights.items():
        if target is not None:
            moved = tensor.to(device=device, dtype=target)
        else:
            moved = tensor.to(device=device)
        # swap_tensors exchanges data AND Python type, so the replacement
        # must already be a Parameter for the module binding to survive.
        swap_tensors(params[name], nn.Parameter(moved))
    if config.tie_word_embeddings:
        model.lm_head.weight = model.model.embed_tokens.weight

    # Re-materialize the RoPE buffer on the target device (the meta
    # construction left it placeholder-only).
    model.register_buffer(
        "inv_freq",
        _compute_inv_freq(config, model._resolved_head_dim).to(device=device),
        persistent=False,
    )
    return model
