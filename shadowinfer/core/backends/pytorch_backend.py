"""PyTorch-backed ModelBackend that wraps SimpleDiffusionLLM."""

from __future__ import annotations

import logging
import os
from typing import Any, Dict, Optional

import torch

from shadowinfer.core.model_backend import ModelBackend
from shadowinfer.core.structs import KVCacheEntry, StepConfig
from shadowinfer.models import SimpleDiffusionLLM

logger = logging.getLogger(__name__)


def _precision_for_shadowkv_mode(mode: str) -> str:
    mapping = {
        "aggressive": "int8",
        "balanced": "fp16",
        "conservative": "fp32",
    }
    return mapping.get(mode, "fp16")


def _quantize_kv(
    k: torch.Tensor,
    v: torch.Tensor,
    precision: str,
) -> tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
    if precision == "fp32":
        return k, v, None, None
    if precision == "fp16":
        return k.half(), v.half(), None, None

    if precision in ("int8", "int4"):
        # For simplicity we use int8 symmetric per-tensor quantization.
        def _sym_int8(t: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
            tf = t.float()
            amax = tf.abs().max()
            scale = amax / 127.0 if amax > 0 else torch.tensor(1.0, device=t.device)
            q = torch.clamp(torch.round(tf / scale), -128, 127).to(torch.int8)
            return q, scale

        qk, sk = _sym_int8(k)
        qv, sv = _sym_int8(v)
        return qk, qv, sk, sv

    raise ValueError(f"Unsupported KV precision: {precision}")


def _dequantize_kv(entry: KVCacheEntry) -> tuple[torch.Tensor, torch.Tensor]:
    k, v = entry.k_tensor, entry.v_tensor
    if entry.precision == "int8":
        k = k.float() * (entry.scale_k if entry.scale_k is not None else 1.0)
        v = v.float() * (entry.scale_v if entry.scale_v is not None else 1.0)
    elif entry.precision == "fp16":
        k = k.float()
        v = v.float()
    return k, v


class PyTorchModelBackend(ModelBackend):
    """Real Diffusion LLM backend powered by SimpleDiffusionLLM."""

    def __init__(
        self,
        model_config: Optional[Dict[str, Any]] = None,
        seed: Optional[int] = 0,
    ) -> None:
        self.model_config = model_config or {}
        self._seed = seed
        self._model: Optional[SimpleDiffusionLLM] = None
        self._kv_cache: Dict[int, KVCacheEntry] = {}
        self._device = torch.device("cpu")

    def load(self, model_name: str, device: Optional[str] = None, **kwargs: Any) -> None:
        del kwargs  # unused
        self._device = torch.device(
            device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.model_config["name"] = model_name

        if self._seed is not None:
            torch.manual_seed(self._seed)

        cfg = self.model_config
        self._model = SimpleDiffusionLLM(
            num_layers=cfg.get("num_layers", 4),
            num_heads=cfg.get("num_heads", 4),
            head_dim=cfg.get("head_dim", 32),
            hidden_dim=cfg.get("hidden_dim", 128),
            vocab_size=cfg.get("vocab_size", 1000),
            max_seq_len=cfg.get("max_seq_len", 128),
            intermediate_dim=cfg.get("intermediate_dim"),
            device=self._device,
            dtype=torch.float32,
        )

        if os.path.isfile(model_name):
            logger.info("Loading weights from %s", model_name)
            state_dict = torch.load(
                model_name,
                map_location="cpu",
                weights_only=True,
            )
            self._model.load_state_dict(state_dict, strict=True)

        self._model.to(self._device)
        self._model.eval()

    def forward_step(
        self,
        x: torch.Tensor,
        step_cfg: StepConfig,
        kv_cache: Optional[Dict[int, KVCacheEntry]] = None,
    ) -> Dict[str, Any]:
        if self._model is None:
            raise RuntimeError("Model has not been loaded. Call load() first.")

        x = x.to(self._device)

        if kv_cache is None:
            kv_cache = self._kv_cache

        precision = _precision_for_shadowkv_mode(step_cfg.shadowkv_mode)

        # Build raw (float32) KV cache for the model from stored entries.
        raw_cache: Dict[int, tuple[torch.Tensor, torch.Tensor]] = {}
        for layer_id in range(self._model.num_layers):
            entry = kv_cache.get(layer_id)
            if entry is not None:
                raw_cache[layer_id] = _dequantize_kv(entry)

        logits, new_raw_kv, attention_scores = self._model.forward_step(
            x=x,
            step_t=step_cfg.step_id,
            total_steps=step_cfg.total_steps,
            kv_cache=raw_cache,
        )

        updated_kv: Dict[int, KVCacheEntry] = {}
        for layer_id in range(self._model.num_layers):
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

            k, v = new_raw_kv[layer_id]
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

    def get_kv_cache(self) -> Dict[int, KVCacheEntry]:
        return self._kv_cache

    def set_kv_cache(self, kv_cache: Dict[int, KVCacheEntry]) -> None:
        self._kv_cache = kv_cache

    def get_model_config(self) -> Dict[str, Any]:
        return dict(self.model_config)
