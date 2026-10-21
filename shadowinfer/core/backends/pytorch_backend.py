"""PyTorch-backed ModelBackend that wraps SimpleDiffusionLLM."""

from __future__ import annotations

import logging
import os
from typing import Any, Dict, Optional

import torch

from shadowinfer.core.model_backend import ModelBackend
from shadowinfer.core.structs import KVCacheEntry, StepConfig
from shadowinfer.execution.step_reuse import StepReuseConfig, StepReuseState
from shadowinfer.models import SimpleDiffusionLLM
from shadowinfer.shadowkv.packed_kv_cache import PackedKVCache

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
        self._packed_cache: Dict[int, PackedKVCache] = {}
        self._device = torch.device("cpu")
        self._int8_ffn_stats: Optional[Dict[str, Any]] = None
        self._kv_lifecycle = "append"
        self._kv_execution = "dense"
        self._packed_precision: Optional[str] = None
        self._step_reuse_enabled = False
        self._step_reuse_threshold = 0.1
        self._step_reuse_warmup_steps = 1
        self._int4_group_size = 16
        self._reuse_state: Optional[StepReuseState] = None
        self._last_execution_stats: Dict[str, Any] = {}

    def configure_execution(
        self,
        kv_lifecycle: str = "append",
        kv_execution: str = "dense",
        packed_precision: Optional[str] = None,
        step_reuse_enabled: bool = False,
        step_reuse_threshold: float = 0.1,
        step_reuse_warmup_steps: int = 1,
        int4_group_size: int = 16,
    ) -> None:
        """Configure the executable KV path (AR013)."""
        if kv_lifecycle not in ("append", "replace"):
            raise ValueError(f"Unsupported kv_lifecycle: {kv_lifecycle}")
        if kv_execution not in ("dense", "packed"):
            raise ValueError(f"Unsupported kv_execution: {kv_execution}")
        if packed_precision is not None and packed_precision not in (
            "fp32",
            "fp16",
            "int8",
            "int4",
        ):
            raise ValueError(f"Unsupported packed_precision: {packed_precision}")
        reuse_config = StepReuseConfig(
            enabled=bool(step_reuse_enabled),
            threshold=float(step_reuse_threshold),
            warmup_steps=int(step_reuse_warmup_steps),
        )
        reuse_config.validate()
        if int4_group_size < 1:
            raise ValueError(f"int4_group_size must be >= 1, got {int4_group_size}")

        self._kv_lifecycle = kv_lifecycle
        self._kv_execution = kv_execution
        self._packed_precision = packed_precision
        self._step_reuse_enabled = bool(step_reuse_enabled)
        self._step_reuse_threshold = float(step_reuse_threshold)
        self._step_reuse_warmup_steps = int(step_reuse_warmup_steps)
        self._int4_group_size = int(int4_group_size)
        self._reuse_config = reuse_config
        reuse_active = reuse_config.enabled and kv_lifecycle == "replace"
        self._reuse_state = StepReuseState() if reuse_active else None
        if step_reuse_enabled and kv_lifecycle != "replace":
            logger.warning("step_reuse_enabled requires kv_lifecycle='replace'; reuse disabled")
            self._step_reuse_enabled = False
            self._reuse_state = None

    def reset_execution_state(self) -> None:
        """Clear KV/packed/reuse state between independent runs."""
        self._kv_cache = {}
        self._packed_cache = {}
        self._last_execution_stats = {}
        if self._step_reuse_enabled:
            self._reuse_state = StepReuseState()
        else:
            self._reuse_state = None

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

        reset_reason: Optional[str] = None
        if kv_cache and self._model is not None:
            for entry in kv_cache.values():
                packed = getattr(entry, "packed_kv", None)
                if packed is not None:
                    if (
                        packed.num_heads != self._model.num_heads
                        or packed.head_dim != self._model.head_dim
                    ):
                        reset_reason = "shape_mismatch"
                        break
                elif entry.k_tensor is not None:
                    if (
                        entry.k_tensor.dim() != 4
                        or entry.k_tensor.shape[1] != self._model.num_heads
                        or entry.k_tensor.shape[3] != self._model.head_dim
                    ):
                        reset_reason = "shape_mismatch"
                        break
            if reset_reason is not None:
                logger.warning(
                    "KV cache shape mismatch detected, clearing cache (reason=%s)",
                    reset_reason,
                )
                self._kv_cache = {}
                self._packed_cache = {}
                kv_cache = {}
                if self._reuse_state is not None:
                    self._reuse_state.reset()

        precision = _precision_for_shadowkv_mode(step_cfg.shadowkv_mode)

        # AR013: replace semantics recompute current-step K/V without past.
        raw_cache: Dict[int, tuple[torch.Tensor, torch.Tensor]] = {}
        if self._kv_lifecycle == "append":
            for layer_id in range(self._model.num_layers):
                entry = kv_cache.get(layer_id)
                if entry is None:
                    continue
                if entry.packed_kv is not None:
                    k, v = entry.packed_kv.unpack()
                    raw_cache[layer_id] = (k, v)
                else:
                    raw_cache[layer_id] = _dequantize_kv(entry)

        reuse_active = self._step_reuse_enabled and self._reuse_state is not None
        reuse_config = (
            StepReuseConfig(
                enabled=True,
                threshold=self._step_reuse_threshold,
                warmup_steps=self._step_reuse_warmup_steps,
            )
            if reuse_active
            else None
        )
        logits, new_raw_kv, attention_scores = self._model.forward_step(
            x=x,
            step_t=step_cfg.step_id,
            total_steps=step_cfg.total_steps,
            kv_cache=raw_cache,
            reuse_state=self._reuse_state if reuse_active else None,
            reuse_config=reuse_config,
        )

        reused_layers = (
            set(self._reuse_state.last_reused_layers)
            if reuse_active and self._reuse_state is not None
            else set()
        )
        updated_kv: Dict[int, KVCacheEntry] = {}
        packed_count = 0
        skipped_repack = 0
        resident_bytes = 0
        data_plane_bytes = 0
        dense_bytes = 0
        fallback = False

        for layer_id in range(self._model.num_layers):
            old_entry = kv_cache.get(layer_id)
            k, v = new_raw_kv[layer_id]
            # SRS D4: dense baseline is FP16 (2 bytes/elem), not FP32.
            dense_bytes += k.numel() * 2 + v.numel() * 2

            if layer_id in reused_layers and old_entry is not None:
                updated_kv[layer_id] = old_entry
                skipped_repack += 1
                if old_entry.packed_kv is not None:
                    resident_bytes += old_entry.packed_kv.resident_bytes()
                    data_plane_bytes += old_entry.packed_kv.data_plane_bytes()
                elif old_entry.k_tensor is not None and old_entry.v_tensor is not None:
                    stored = (
                        old_entry.k_tensor.numel() * old_entry.k_tensor.element_size()
                        + old_entry.v_tensor.numel() * old_entry.v_tensor.element_size()
                    )
                    resident_bytes += stored
                    data_plane_bytes += stored
                continue

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
                    packed_kv=old_entry.packed_kv,
                )
                if old_entry.packed_kv is not None:
                    resident_bytes += old_entry.packed_kv.resident_bytes()
                    data_plane_bytes += old_entry.packed_kv.data_plane_bytes()
                elif old_entry.k_tensor is not None and old_entry.v_tensor is not None:
                    stored = (
                        old_entry.k_tensor.numel() * old_entry.k_tensor.element_size()
                        + old_entry.v_tensor.numel() * old_entry.v_tensor.element_size()
                    )
                    resident_bytes += stored
                    data_plane_bytes += stored
                continue

            if self._kv_execution == "packed":
                packed_precision = self._packed_precision or precision
                try:
                    seq_len = k.shape[2]
                    num_heads = k.shape[1]
                    precision_map = {
                        t: {h: packed_precision for h in range(num_heads)} for t in range(seq_len)
                    }
                    packed = PackedKVCache.pack(
                        k,
                        v,
                        precision_map,
                        group_size=self._int4_group_size,
                    )["packed_kv"]
                    updated_kv[layer_id] = KVCacheEntry(
                        k_tensor=None,
                        v_tensor=None,
                        precision="mixed",
                        importance_score=0.0,
                        is_reused=False,
                        reuse_step=-1,
                        packed_kv=packed,
                    )
                    self._packed_cache[layer_id] = packed
                    resident_bytes += packed.resident_bytes()
                    data_plane_bytes += packed.data_plane_bytes()
                    packed_count += 1
                except Exception as exc:  # pragma: no cover - defensive
                    logger.warning(
                        "packed KV execution failed at layer %s, falling back to dense: %s",
                        layer_id,
                        exc,
                    )
                    fallback = True
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
                    stored = qk.numel() * qk.element_size() + qv.numel() * qv.element_size()
                    resident_bytes += stored
                    data_plane_bytes += stored
                    self._packed_cache.pop(layer_id, None)
            else:
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
                stored = qk.numel() * qk.element_size() + qv.numel() * qv.element_size()
                resident_bytes += stored
                data_plane_bytes += stored
                self._packed_cache.pop(layer_id, None)

        self._kv_cache = updated_kv
        compression_ratio = (resident_bytes / dense_bytes) if dense_bytes > 0 else 0.0
        data_plane_ratio = (data_plane_bytes / dense_bytes) if dense_bytes > 0 else 0.0
        self._last_execution_stats = {
            "kv_lifecycle": self._kv_lifecycle,
            "kv_execution": self._kv_execution,
            "precision": self._packed_precision or precision,
            "reused_layers": sorted(reused_layers),
            "computed_layers": sorted(set(range(self._model.num_layers)) - reused_layers),
            "reused_ratio": (
                len(reused_layers) / self._model.num_layers if self._model.num_layers else 0.0
            ),
            "packed_layers": packed_count,
            "skipped_repack": skipped_repack,
            "resident_bytes": resident_bytes,
            "data_plane_bytes": data_plane_bytes,
            "metadata_bytes": max(resident_bytes - data_plane_bytes, 0),
            "dense_bytes": dense_bytes,
            "compression_ratio": compression_ratio,
            "data_plane_ratio": data_plane_ratio,
            "fallback": fallback,
            "reset_reason": reset_reason,
            "drifts": (
                dict(self._reuse_state.last_drifts) if reuse_active and self._reuse_state else {}
            ),
            "attention_reused": bool(
                reuse_active and self._reuse_state and self._reuse_state.attention_reused
            ),
        }

        return {
            "output": logits,
            "kv_cache": updated_kv,
            "attention_scores": attention_scores,
            "execution_stats": self._last_execution_stats,
        }

    @staticmethod
    def _entry_kv_tensors(entry: Any) -> Optional[tuple]:
        """Return dense (k, v) for dense or packed KV entries (AR013)."""
        if entry is None:
            return None
        if entry.k_tensor is not None and entry.v_tensor is not None:
            return entry.k_tensor, entry.v_tensor
        packed = getattr(entry, "packed_kv", None)
        if packed is not None:
            return packed.unpack()
        return None

    def get_kv_cache(self) -> Dict[int, KVCacheEntry]:
        return self._kv_cache

    def set_kv_cache(self, kv_cache: Dict[int, KVCacheEntry]) -> None:
        self._kv_cache = kv_cache

    def enable_int8_ffn(self, keep_fp32_reference: bool = False) -> Dict[str, Any]:
        """Replace the loaded model's FFN projections with executable INT8 linears."""
        if self._model is None:
            raise RuntimeError("Model has not been loaded. Call load() first.")
        stats = self._model.enable_int8_ffn(keep_fp32_reference=keep_fp32_reference)
        self._int8_ffn_stats = dict(stats)
        return {"supported": True, **stats}

    def get_int8_ffn_stats(self) -> Optional[Dict[str, Any]]:
        stats = getattr(self, "_int8_ffn_stats", None)
        return None if stats is None else dict(stats)

    def get_model_config(self) -> Dict[str, Any]:
        return dict(self.model_config)
