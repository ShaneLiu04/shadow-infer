"""SGLang backend adapter scaffold.

This module provides a ``ModelBackend`` wrapper around SGLang. It uses optional
imports so that ShadowInfer remains installable on machines without SGLang.

When SGLang is installed, the backend can be registered via the
``shadowinfer.backends`` entry-point group. The current implementation is a
scaffold: it satisfies the ``ModelBackend`` interface and exposes model
configuration, but single-step diffusion forward is not yet implemented because
SGLang is optimized for auto-regressive generation rather than denoising-step
inference.

Version: 3.2.0
"""

from __future__ import annotations

from typing import Any, Dict, Optional

import torch

from shadowinfer.core.model_backend import ModelBackend
from shadowinfer.core.structs import KVCacheEntry, StepConfig

try:
    import sglang  # type: ignore

    _SGLANG_AVAILABLE = True
except Exception:
    sglang = None  # type: ignore
    _SGLANG_AVAILABLE = False


class SGLangModelBackend(ModelBackend):
    """SGLang-backed ``ModelBackend`` scaffold.

    Args:
        model_config: Optional model configuration dict.
        seed: Random seed for deterministic behavior.
    """

    def __init__(
        self,
        model_config: Optional[Dict[str, Any]] = None,
        seed: Optional[int] = None,
    ) -> None:
        self._model_config = model_config or {}
        self._seed = seed
        self._device: torch.device = torch.device("cpu")
        self._engine: Optional[Any] = None
        self._kv_cache: Dict[int, KVCacheEntry] = {}

    @classmethod
    def is_available(cls) -> bool:
        """Return True when SGLang can be imported."""
        return _SGLANG_AVAILABLE

    def load(
        self, model_name: str, device: Optional[str] = None, **kwargs: Any
    ) -> None:
        """Load the SGLang engine.

        The ``device`` argument is accepted for API compatibility; SGLang
        typically manages device placement internally.
        """
        if not self.is_available():
            raise RuntimeError(
                "SGLang backend is not available. Install SGLang and restart, e.g.: "
                "pip install sglang"
            )

        if self._seed is not None:
            torch.manual_seed(self._seed)

        self._device = torch.device(
            device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        )
        self._model_config["name"] = model_name
        self._engine = None  # Lazy instantiation on first forward_step.

    def forward_step(
        self,
        x: torch.Tensor,
        step_cfg: StepConfig,
        kv_cache: Optional[Dict[int, KVCacheEntry]] = None,
    ) -> Dict[str, Any]:
        """Execute a single denoising step via SGLang.

        .. note::
            This is a scaffolding method. A production implementation needs to
            map ShadowInfer's denoising-step semantics to SGLang's runtime or
            to a direct model forward call. Until then, the method raises
            ``NotImplementedError`` with integration guidance.
        """
        raise NotImplementedError(
            "SGLangModelBackend.forward_step() is not implemented yet. "
            "SGLang is auto-regressive; to support ShadowInfer diffusion steps, "
            "implement a single forward pass against the SGLang model runner and "
            "return {'output', 'kv_cache', 'attention_scores'}."
        )

    def get_kv_cache(self) -> Dict[int, KVCacheEntry]:
        """Return the current KV cache state."""
        return self._kv_cache

    def set_kv_cache(self, kv_cache: Dict[int, KVCacheEntry]) -> None:
        """Set the current KV cache state."""
        self._kv_cache = kv_cache or {}

    def get_model_config(self) -> Dict[str, Any]:
        """Return the model configuration."""
        return self._model_config

    def warmup(self, num_steps: int = 3) -> None:
        """Warmup is a no-op in the scaffold."""
        pass
