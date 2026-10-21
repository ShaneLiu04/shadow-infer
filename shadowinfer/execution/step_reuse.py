"""Cross-step block reuse state for diffusion denoising (AR013).

The state stores, per transformer block, the input/output hidden states and the
K/V tensors from the last full computation. A relative-L2 drift gate decides
whether the block can be skipped on the next step. Reusing a block does not
refresh the cache, which bounds accumulated drift against the last exact
computation.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Set, Tuple

import torch


@dataclass(frozen=True)
class StepReuseConfig:
    """Configuration for drift-gated block reuse."""

    enabled: bool = False
    threshold: float = 0.1
    warmup_steps: int = 1

    def validate(self) -> None:
        if self.threshold < 0:
            raise ValueError(f"step reuse threshold must be >= 0, got {self.threshold}")
        if self.warmup_steps < 0:
            raise ValueError(f"step reuse warmup_steps must be >= 0, got {self.warmup_steps}")

    def __post_init__(self) -> None:
        self.validate()


@dataclass
class LayerReuseEntry:
    """Cached block state from the last full computation."""

    hidden_in: torch.Tensor
    hidden_out: torch.Tensor
    k: torch.Tensor
    v: torch.Tensor


@dataclass
class StepReuseState:
    """Per-layer reuse cache plus execution statistics."""

    entries: Dict[int, LayerReuseEntry] = field(default_factory=dict)
    last_attention: Optional[torch.Tensor] = None
    last_reused_layers: Set[int] = field(default_factory=set)
    last_computed_layers: Set[int] = field(default_factory=set)
    last_drifts: Dict[int, float] = field(default_factory=dict)
    attention_reused: bool = False

    def reset(self) -> None:
        self.entries.clear()
        self.last_attention = None
        self.last_reused_layers.clear()
        self.last_computed_layers.clear()
        self.last_drifts.clear()
        self.attention_reused = False

    def begin_step(self) -> None:
        self.last_reused_layers = set()
        self.last_computed_layers = set()
        self.last_drifts = {}
        self.attention_reused = False

    def drift(self, layer_id: int, hidden: torch.Tensor) -> Optional[float]:
        entry = self.entries.get(layer_id)
        if entry is None:
            return None
        if entry.hidden_in.shape != hidden.shape:
            return None
        denom = entry.hidden_in.norm()
        if float(denom) <= 0.0:
            return None
        return float((hidden - entry.hidden_in).norm() / denom)

    def should_reuse(
        self,
        layer_id: int,
        hidden: torch.Tensor,
        step_id: int,
        config: StepReuseConfig,
    ) -> Tuple[bool, Optional[float]]:
        if not config.enabled or step_id < config.warmup_steps:
            return False, None
        drift = self.drift(layer_id, hidden)
        if drift is None:
            self.entries.pop(layer_id, None)
            return False, None
        return drift <= config.threshold, drift

    def record(
        self,
        layer_id: int,
        hidden_in: torch.Tensor,
        hidden_out: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
    ) -> None:
        self.entries[layer_id] = LayerReuseEntry(
            hidden_in=hidden_in.detach(),
            hidden_out=hidden_out.detach(),
            k=k.detach(),
            v=v.detach(),
        )

    def stats(self) -> Dict[str, Any]:
        reused = sorted(self.last_reused_layers)
        computed = sorted(self.last_computed_layers)
        total = len(reused) + len(computed)
        return {
            "reused_layers": reused,
            "computed_layers": computed,
            "reused_ratio": (len(reused) / total) if total else 0.0,
            "drifts": dict(self.last_drifts),
            "attention_reused": self.attention_reused,
        }
