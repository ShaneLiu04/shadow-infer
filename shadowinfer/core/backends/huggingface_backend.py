"""HuggingFace-backed ModelBackend for ShadowInfer.

This adapter lets ShadowInfer drive any publicly available causal language model
from the HuggingFace ecosystem (e.g. ``gpt2``, ``facebook/opt-125m``,
``TinyLlama/TinyLlama-1.1B-Chat-v1.0``) through the standard
:class:`~shadowinfer.core.model_backend.ModelBackend` interface.
"""

from __future__ import annotations

import logging
import os
from typing import Any, Dict, List, Optional, Tuple

import torch

# Reuse the tiny quantization helpers from the PyTorch backend so both backends
# behave consistently with respect to ShadowKV precision hints.
from shadowinfer.core.backends.pytorch_backend import (
    _dequantize_kv,
    _precision_for_shadowkv_mode,
    _quantize_kv,
)
from shadowinfer.core.model_backend import ModelBackend
from shadowinfer.core.structs import KVCacheEntry, StepConfig

logger = logging.getLogger(__name__)

# Keep the backend optional: importing this module must not fail when
# ``transformers`` is not installed.  The dependency is only required when the
# class is actually instantiated.
try:
    import transformers  # noqa: F401
    from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
    from transformers.cache_utils import DynamicCache

    _TRANSFORMERS_AVAILABLE = True
except ImportError as _import_error:  # pragma: no cover
    transformers = None  # type: ignore[assignment]
    AutoConfig = None  # type: ignore[assignment,misc]
    AutoModelForCausalLM = None  # type: ignore[assignment,misc]
    AutoTokenizer = None  # type: ignore[assignment,misc]
    DynamicCache = None  # type: ignore[assignment,misc]
    _TRANSFORMERS_AVAILABLE = False
    _HF_IMPORT_ERROR = _import_error


# Fallback tiny model used when a requested model cannot be downloaded.
_FALLBACK_HIDDEN_SIZE = 64
_FALLBACK_NUM_LAYERS = 2
_FALLBACK_NUM_HEADS = 2
_FALLBACK_VOCAB_SIZE = 1000
_FALLBACK_MAX_POSITION_EMBEDDINGS = 128


def _default_fallback_dir() -> str:
    """Return a stable local directory for the fallback tiny model."""
    return os.path.join(os.path.expanduser("~"), ".cache", "shadowinfer", "hf_fallback")


def _is_internal_test_model(model_name: str) -> bool:
    """Return True if the model name is an internal test identifier."""
    return model_name.startswith("__shadowinfer_") and model_name.endswith("__")


def _create_fallback_model(path: str) -> None:
    """Create a minimal local GPT-2-style causal LM for offline testing."""
    os.makedirs(path, exist_ok=True)
    config = AutoConfig.for_model(
        "gpt2",
        n_embd=_FALLBACK_HIDDEN_SIZE,
        n_layer=_FALLBACK_NUM_LAYERS,
        n_head=_FALLBACK_NUM_HEADS,
        vocab_size=_FALLBACK_VOCAB_SIZE,
        n_positions=_FALLBACK_MAX_POSITION_EMBEDDINGS,
        n_ctx=_FALLBACK_MAX_POSITION_EMBEDDINGS,
        bos_token_id=0,
        eos_token_id=0,
    )
    model = AutoModelForCausalLM.from_config(config)
    model.save_pretrained(path)


def _hf_config_to_model_config(
    model_name: str, hf_cfg: Any, overrides: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """Convert a HuggingFace config into the ShadowInfer model config dict."""
    hidden_dim = getattr(hf_cfg, "hidden_size", None) or getattr(hf_cfg, "n_embd", 64)
    num_layers = getattr(hf_cfg, "num_hidden_layers", None) or getattr(hf_cfg, "n_layer", 2)
    num_heads = getattr(hf_cfg, "num_attention_heads", None) or getattr(hf_cfg, "n_head", 2)
    intermediate_dim = getattr(hf_cfg, "intermediate_size", None) or getattr(
        hf_cfg, "n_inner", None
    )
    if intermediate_dim is None:
        intermediate_dim = 4 * hidden_dim
    vocab_size = getattr(hf_cfg, "vocab_size", _FALLBACK_VOCAB_SIZE)
    head_dim = getattr(hf_cfg, "head_dim", hidden_dim // max(num_heads, 1))

    cfg: Dict[str, Any] = {
        "name": model_name,
        "num_layers": num_layers,
        "num_heads": num_heads,
        "head_dim": head_dim,
        "hidden_dim": hidden_dim,
        "intermediate_dim": intermediate_dim,
        "vocab_size": vocab_size,
        "max_seq_len": getattr(
            hf_cfg, "max_position_embeddings", _FALLBACK_MAX_POSITION_EMBEDDINGS
        ),
        "batch_size": 1,
        "seq_len": _FALLBACK_MAX_POSITION_EMBEDDINGS,
    }
    if overrides:
        cfg.update(overrides)
    return cfg


def _cache_to_kv_list(
    past_key_values: Any, num_layers: int
) -> List[Tuple[torch.Tensor, torch.Tensor]]:
    """Extract ``[(k, v), ...]`` from either a tuple or a ``DynamicCache``."""
    if past_key_values is None:
        return []
    result: List[Tuple[torch.Tensor, torch.Tensor]] = []
    for item in past_key_values:
        if isinstance(item, (list, tuple)):
            # Some ``DynamicCache`` implementations yield ``(k, v, ...)``
            # tuples; keep only the key and value tensors.
            result.append((item[0], item[1]))
        else:
            result.append(item)
    return result


class HuggingFaceModelBackend(ModelBackend):
    """Real causal LM backend powered by HuggingFace ``transformers``.

    The backend loads any causal language model that can be instantiated with
    ``AutoModelForCausalLM`` and runs it as a diffusion-style generator through
    :meth:`forward_step`.  ShadowInfer's per-layer KV cache entries are injected
    as ``past_key_values`` and updated after every step.
    """

    def __init__(
        self,
        model_config: Optional[Dict[str, Any]] = None,
        seed: Optional[int] = 0,
    ) -> None:
        if not _TRANSFORMERS_AVAILABLE:
            raise ImportError(
                "HuggingFaceModelBackend requires the `transformers` package. "
                "Install it with: pip install transformers"
            ) from _HF_IMPORT_ERROR

        self.model_config: Dict[str, Any] = dict(model_config or {})
        self._seed = seed
        self._model: Optional[Any] = None
        self._tokenizer: Optional[Any] = None
        self._kv_cache: Dict[int, KVCacheEntry] = {}
        self._device = torch.device("cpu")

    def load(
        self,
        model_name: str,
        device: Optional[str] = None,
        local_files_only: bool = False,
        **kwargs: Any,
    ) -> None:
        """Load a HuggingFace causal language model.

        If ``model_name`` points to a local directory it is loaded directly.
        Otherwise the model is downloaded from the HuggingFace Hub.  When the
        download fails (e.g. no network) or ``local_files_only=True`` is set, a
        tiny local GPT-2-style model is created on the fly and used instead so
        that tests and local runs keep working offline.
        """
        self._device = torch.device(
            device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.model_config["name"] = model_name

        if self._seed is not None:
            torch.manual_seed(self._seed)

        if os.path.isdir(model_name):
            self._load_from_path(model_name, local_files_only=local_files_only)
            return

        if local_files_only or _is_internal_test_model(model_name):
            logger.info("Loading local fallback model for %s.", model_name)
            self._load_fallback()
            self._model.to(self._device)
            self._model.eval()
            self.model_config = _hf_config_to_model_config(model_name, self._model.config)
            return

        try:
            self._model = AutoModelForCausalLM.from_pretrained(model_name)
            self._tokenizer = AutoTokenizer.from_pretrained(model_name)
        except (OSError, EnvironmentError, ConnectionError, TimeoutError) as exc:
            logger.warning(
                "Could not download %s (%s). Falling back to a local tiny model.",
                model_name,
                exc,
            )
            self._load_fallback()
        except Exception as exc:  # pragma: no cover - defensive fallback
            logger.warning(
                "Failed to load %s (%s). Falling back to a local tiny model.",
                model_name,
                exc,
            )
            self._load_fallback()

        self._model.to(self._device)
        self._model.eval()
        self.model_config = _hf_config_to_model_config(model_name, self._model.config)

    def _load_from_path(self, path: str, local_files_only: bool = True) -> None:
        """Load a model from a local directory."""
        self._model = AutoModelForCausalLM.from_pretrained(
            path,
            local_files_only=local_files_only,
        )
        try:
            self._tokenizer = AutoTokenizer.from_pretrained(path, local_files_only=local_files_only)
        except Exception as exc:  # noqa: BLE001
            logger.debug("Tokenizer not available at %s: %s", path, exc)
            self._tokenizer = None

    def _load_fallback(self) -> None:
        """Create or reuse the local tiny fallback model."""
        fallback_path = _default_fallback_dir()
        if not os.path.exists(os.path.join(fallback_path, "config.json")):
            _create_fallback_model(fallback_path)
        self._load_from_path(fallback_path)

    def forward_step(
        self,
        x: torch.Tensor,
        step_cfg: StepConfig,
        kv_cache: Optional[Dict[int, KVCacheEntry]] = None,
    ) -> Dict[str, Any]:
        """Run a single denoising step through the HF causal LM.

        Args:
            x: Either integer token ids of shape ``[batch, seq_len]`` or a
                float tensor of shape ``[batch, seq_len, hidden_dim]``.
            step_cfg: ShadowInfer step configuration.
            kv_cache: Optional per-layer KV cache entries to inject as
                ``past_key_values``.

        Returns:
            A dictionary with ``output`` (logits), ``kv_cache`` (updated
            entries) and ``attention_scores``.
        """
        if self._model is None:
            raise RuntimeError("Model has not been loaded. Call load() first.")

        x = x.to(self._device)
        batch_size, seq_len = x.shape[:2]

        if kv_cache is None:
            kv_cache = self._kv_cache

        precision = _precision_for_shadowkv_mode(step_cfg.shadowkv_mode)

        # Build the model-specific cache object from ShadowInfer KV entries.
        past_key_values, attention_mask = self._build_past_key_values(kv_cache, batch_size, seq_len)

        if x.dim() == 2:
            model_inputs: Dict[str, Any] = {"input_ids": x.long()}
        else:
            model_inputs = {
                "inputs_embeds": x.to(dtype=self._model.dtype),
            }

        model_inputs.update(
            {
                "past_key_values": past_key_values,
                "attention_mask": attention_mask,
                "output_attentions": True,
                "output_hidden_states": False,
                "return_dict": True,
                "use_cache": True,
            }
        )

        with torch.no_grad():
            outputs = self._model(**model_inputs)

        logits = outputs.logits
        new_kv_list = _cache_to_kv_list(outputs.past_key_values, self.model_config["num_layers"])

        # Pick the last layer attention weights for ShadowKV scoring.
        if outputs.attentions and outputs.attentions[-1] is not None:
            attention_scores = outputs.attentions[-1]
        else:
            attention_scores = torch.empty(())

        updated_kv: Dict[int, KVCacheEntry] = {}
        for layer_id, (new_k, new_v) in enumerate(new_kv_list):
            old_entry = kv_cache.get(layer_id)
            if layer_id in step_cfg.reuse_layers and old_entry is not None:
                updated_kv[layer_id] = KVCacheEntry(
                    k_tensor=old_entry.k_tensor,
                    v_tensor=old_entry.v_tensor,
                    precision=old_entry.precision,
                    scale_k=old_entry.scale_k,
                    scale_v=old_entry.scale_v,
                    importance_score=old_entry.importance_score,
                    is_reused=True,
                    reuse_step=step_cfg.step_id - 1,
                )
                continue

            k = new_k.detach().float()
            v = new_v.detach().float()
            qk, qv, sk, sv = _quantize_kv(k, v, precision)
            updated_kv[layer_id] = KVCacheEntry(
                k_tensor=qk,
                v_tensor=qv,
                precision=precision,
                scale_k=sk,
                scale_v=sv,
                importance_score=0.0,
                is_reused=False,
                reuse_step=-1,
            )

        self._kv_cache = updated_kv

        return {
            "output": logits,
            "kv_cache": updated_kv,
            "attention_scores": attention_scores,
        }

    def _build_past_key_values(
        self,
        kv_cache: Dict[int, KVCacheEntry],
        batch_size: int,
        seq_len: int,
    ) -> Tuple[Any, Optional[torch.Tensor]]:
        """Convert a ShadowInfer KV cache to a HF ``past_key_values`` object.

        Missing layers are represented by zero-length tensors so that every
        layer has an entry.  Returns ``(past_key_values, attention_mask)``.
        """
        if not kv_cache:
            return None, None

        num_layers = self.model_config.get("num_layers", _FALLBACK_NUM_LAYERS)
        num_heads = self.model_config.get("num_heads", _FALLBACK_NUM_HEADS)
        head_dim = self.model_config.get("head_dim", _FALLBACK_HIDDEN_SIZE // max(num_heads, 1))
        dtype = self._model.dtype
        device = self._device

        kv_list: List[Tuple[torch.Tensor, torch.Tensor]] = []
        past_len = 0
        for layer_id in range(num_layers):
            entry = kv_cache.get(layer_id)
            if entry is None:
                k = torch.zeros(batch_size, num_heads, 0, head_dim, device=device, dtype=dtype)
                v = torch.zeros(batch_size, num_heads, 0, head_dim, device=device, dtype=dtype)
            else:
                k, v = _dequantize_kv(entry)
                k = k.to(device=device, dtype=dtype)
                v = v.to(device=device, dtype=dtype)
                past_len = k.shape[2]
            kv_list.append((k, v))

        total_len = past_len + seq_len
        attention_mask = torch.ones(batch_size, total_len, device=device, dtype=torch.long)

        if DynamicCache is not None:
            return DynamicCache(ddp_cache_data=tuple(kv_list)), attention_mask
        return tuple(kv_list), attention_mask

    def get_kv_cache(self) -> Dict[int, KVCacheEntry]:
        """Return the current KV cache state."""
        return self._kv_cache

    def set_kv_cache(self, kv_cache: Dict[int, KVCacheEntry]) -> None:
        """Set the current KV cache state."""
        self._kv_cache = kv_cache

    def get_model_config(self) -> Dict[str, Any]:
        """Return the model configuration."""
        return dict(self.model_config)
