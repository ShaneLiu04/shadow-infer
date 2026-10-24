"""PyTorch-backed ModelBackend that wraps SimpleDiffusionLLM."""

from __future__ import annotations

import logging
import os
from typing import Any, Dict, Optional

import torch

from shadowinfer.core.model_backend import ModelBackend
from shadowinfer.core.structs import KVCacheEntry, StepConfig
from shadowinfer.execution.block_decoding import BlockDecodingConfig, GenerateResult
from shadowinfer.execution.blockwise_generate import blockwise_generate
from shadowinfer.execution.macro_mode import MacroModeConfig, MacroModeState
from shadowinfer.execution.step_reuse import StepReuseConfig, StepReuseState
from shadowinfer.models import SimpleDiffusionLLM
from shadowinfer.shadowkv.packed_kv_cache import PackedKVCache
from shadowinfer.shadowkv.paged_kv_store import PagedKVStore

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
        self._paged_cache: Dict[int, PagedKVStore] = {}
        self._device = torch.device("cpu")
        self._int8_ffn_stats: Optional[Dict[str, Any]] = None
        self._kv_lifecycle = "append"
        self._kv_execution = "dense"
        self._page_size = 32
        self._paged_appended_total = 0
        self._packed_precision: Optional[str] = None
        self._step_reuse_enabled = False
        self._step_reuse_threshold = 0.1
        self._step_reuse_warmup_steps = 1
        self._int4_group_size = 16
        self._reuse_state: Optional[StepReuseState] = None
        self._mode_control_enabled = False
        self._macro_block_size = 32
        self._mode_config: Optional[MacroModeConfig] = None
        self._mode_state: Optional[MacroModeState] = None
        self._last_execution_stats: Dict[str, Any] = {}
        self._last_generate_stores: Optional[Dict[int, PagedKVStore]] = None

    def configure_execution(
        self,
        kv_lifecycle: str = "append",
        kv_execution: str = "dense",
        packed_precision: Optional[str] = None,
        step_reuse_enabled: bool = False,
        step_reuse_threshold: float = 0.1,
        step_reuse_warmup_steps: int = 1,
        int4_group_size: int = 16,
        page_size: int = 32,
        mode_control_enabled: bool = False,
        macro_block_size: int = 32,
        skip_threshold: float = 0.1,
        delta_min_active_ratio: float = 0.5,
        mode_warmup_steps: int = 1,
        force_refresh_every: int = 0,
    ) -> None:
        """Configure the executable KV path (AR013; paged AR014; mode control AR015)."""
        if kv_lifecycle not in ("append", "replace"):
            raise ValueError(f"Unsupported kv_lifecycle: {kv_lifecycle}")
        if kv_execution not in ("dense", "packed", "paged"):
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
        if page_size < 1:
            raise ValueError(f"page_size must be >= 1, got {page_size}")
        mode_config = MacroModeConfig(
            enabled=bool(mode_control_enabled),
            macro_block_size=int(macro_block_size),
            skip_threshold=float(skip_threshold),
            delta_min_active_ratio=float(delta_min_active_ratio),
            warmup_steps=int(mode_warmup_steps),
            force_refresh_every=int(force_refresh_every),
        )
        if mode_config.enabled:
            if kv_execution != "paged":
                raise ValueError(
                    "mode_control_enabled requires kv_execution='paged', got "
                    f"{kv_execution!r} (explicit opt-in, no silent downgrade)"
                )
            if kv_lifecycle != "replace":
                raise ValueError(
                    "mode_control_enabled requires kv_lifecycle='replace', got "
                    f"{kv_lifecycle!r} (explicit opt-in, no silent downgrade)"
                )
            if reuse_config.enabled:
                raise ValueError(
                    "mode_control_enabled and step_reuse_enabled are mutually exclusive"
                )

        self._kv_lifecycle = kv_lifecycle
        self._kv_execution = kv_execution
        self._packed_precision = packed_precision
        self._page_size = int(page_size)
        self._step_reuse_enabled = bool(step_reuse_enabled)
        self._step_reuse_threshold = float(step_reuse_threshold)
        self._step_reuse_warmup_steps = int(step_reuse_warmup_steps)
        self._int4_group_size = int(int4_group_size)
        self._reuse_config = reuse_config
        self._mode_config = mode_config
        self._macro_block_size = mode_config.macro_block_size
        mode_active = mode_config.enabled and kv_execution == "paged"
        self._mode_state = MacroModeState() if mode_active else None
        self._mode_control_enabled = mode_active
        reuse_active = reuse_config.enabled and kv_lifecycle == "replace"
        self._reuse_state = StepReuseState() if reuse_active else None
        if step_reuse_enabled and kv_lifecycle != "replace":
            logger.warning("step_reuse_enabled requires kv_lifecycle='replace'; reuse disabled")
            self._step_reuse_enabled = False
            self._reuse_state = None

    def reset_execution_state(self) -> None:
        """Clear KV/packed/paged/reuse/mode state between independent runs."""
        self._kv_cache = {}
        self._packed_cache = {}
        self._paged_cache = {}
        self._paged_appended_total = 0
        self._last_execution_stats = {}
        if self._step_reuse_enabled:
            self._reuse_state = StepReuseState()
        else:
            self._reuse_state = None
        if self._mode_control_enabled:
            self._mode_state = MacroModeState()
        else:
            self._mode_state = None

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
                paged = getattr(entry, "paged_kv", None)
                if paged is not None:
                    if (
                        paged.num_heads != self._model.num_heads
                        or paged.head_dim != self._model.head_dim
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
                self._paged_cache = {}
                kv_cache = {}
                if self._reuse_state is not None:
                    self._reuse_state.reset()
                if self._mode_state is not None:
                    self._mode_state.reset()

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
                elif entry.paged_kv is not None:
                    k, v = entry.paged_kv.gather()
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
        mode_active = self._mode_control_enabled and self._mode_state is not None
        fallback = False
        try:
            logits, new_raw_kv, attention_scores = self._model.forward_step(
                x=x,
                step_t=step_cfg.step_id,
                total_steps=step_cfg.total_steps,
                kv_cache=raw_cache,
                reuse_state=self._reuse_state if reuse_active else None,
                reuse_config=reuse_config,
                mode_state=self._mode_state if mode_active else None,
                mode_config=self._mode_config if mode_active else None,
            )
        except Exception as exc:
            if not mode_active:
                raise
            logger.warning(
                "mode-controlled forward failed, degrading to dense full recompute: %s",
                exc,
            )
            fallback = True
            self._mode_state.reset()
            mode_active = False
            logits, new_raw_kv, attention_scores = self._model.forward_step(
                x=x,
                step_t=step_cfg.step_id,
                total_steps=step_cfg.total_steps,
                kv_cache=raw_cache,
                reuse_state=None,
                reuse_config=None,
                mode_state=None,
                mode_config=None,
            )

        reused_layers = (
            set(self._reuse_state.last_reused_layers)
            if reuse_active and self._reuse_state is not None
            else set()
        )
        updated_kv: Dict[int, KVCacheEntry] = {}
        packed_count = 0
        skipped_repack = 0
        paged_count = 0
        appended_this_step = 0
        repack_saved_total = 0
        resident_bytes = 0
        data_plane_bytes = 0
        dense_bytes = 0

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
                elif old_entry.paged_kv is not None:
                    resident_bytes += old_entry.paged_kv.resident_bytes()
                    data_plane_bytes += old_entry.paged_kv.data_plane_bytes()
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
                    paged_kv=old_entry.paged_kv,
                )
                if old_entry.packed_kv is not None:
                    resident_bytes += old_entry.packed_kv.resident_bytes()
                    data_plane_bytes += old_entry.packed_kv.data_plane_bytes()
                elif old_entry.paged_kv is not None:
                    resident_bytes += old_entry.paged_kv.resident_bytes()
                    data_plane_bytes += old_entry.paged_kv.data_plane_bytes()
                elif old_entry.k_tensor is not None and old_entry.v_tensor is not None:
                    stored = (
                        old_entry.k_tensor.numel() * old_entry.k_tensor.element_size()
                        + old_entry.v_tensor.numel() * old_entry.v_tensor.element_size()
                    )
                    resident_bytes += stored
                    data_plane_bytes += stored
                continue

            if self._kv_execution == "paged":
                page_precision = self._packed_precision or precision
                try:
                    store = (
                        old_entry.paged_kv
                        if old_entry is not None and old_entry.paged_kv is not None
                        else None
                    )
                    if store is None:
                        store = PagedKVStore(
                            k.shape[0],
                            k.shape[1],
                            k.shape[3],
                            page_size=self._page_size,
                            device=k.device,
                            int4_group_size=self._int4_group_size,
                        )
                    else:
                        store.validate_against(k.shape[0], k.shape[1], k.shape[3])
                    seq_len = k.shape[2]
                    mode_dispatch = (
                        mode_active and self._kv_lifecycle == "replace" and store.max_seq == seq_len
                    )
                    if mode_dispatch:
                        try:
                            rewritten = self._mode_rewrite_dispatch(
                                store, layer_id, k, v, page_precision
                            )
                            self._mode_state.record_rewritten(rewritten)
                            repack_saved_total += seq_len - rewritten
                        except Exception as exc:
                            logger.warning(
                                "mode rewrite dispatch failed at layer %s, degrading "
                                "to clear+append: %s",
                                layer_id,
                                exc,
                            )
                            fallback = True
                            self._mode_state.reset()
                            mode_active = False
                            store.clear()
                            store.append(k.contiguous(), v.contiguous(), page_precision)
                            appended_this_step += seq_len
                    else:
                        if self._kv_lifecycle == "replace":
                            if mode_active and store.max_seq > 0 and store.max_seq != seq_len:
                                self._mode_state.reset()
                            store.clear()
                            past = 0
                        else:
                            past = store.max_seq
                            if seq_len < past:
                                raise ValueError(
                                    f"paged append anomaly: model output seq {seq_len} "
                                    f"< stored past {past} at layer {layer_id}"
                                )
                        store.append(
                            k[:, :, past:, :].contiguous(),
                            v[:, :, past:, :].contiguous(),
                            page_precision,
                        )
                        appended_this_step += seq_len - past
                        repack_saved_total += past
                    updated_kv[layer_id] = KVCacheEntry(
                        k_tensor=None,
                        v_tensor=None,
                        precision="mixed",
                        importance_score=0.0,
                        is_reused=False,
                        reuse_step=-1,
                        paged_kv=store,
                    )
                    self._paged_cache[layer_id] = store
                    resident_bytes += store.resident_bytes()
                    data_plane_bytes += store.data_plane_bytes()
                    paged_count += 1
                    self._packed_cache.pop(layer_id, None)
                except Exception as exc:
                    logger.warning(
                        "paged KV execution failed at layer %s, falling back to dense: %s",
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
                    self._paged_cache.pop(layer_id, None)
            elif self._kv_execution == "packed":
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
                    self._paged_cache.pop(layer_id, None)
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
                self._paged_cache.pop(layer_id, None)

        self._kv_cache = updated_kv
        self._paged_appended_total += appended_this_step
        compression_ratio = (resident_bytes / dense_bytes) if dense_bytes > 0 else 0.0
        data_plane_ratio = (data_plane_bytes / dense_bytes) if dense_bytes > 0 else 0.0
        mode_stats = (
            self._mode_state.stats()
            if self._mode_control_enabled and self._mode_state is not None
            else {}
        )
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
            "paged": self._kv_execution == "paged",
            "page_size": self._page_size,
            "page_count": sum(store.page_count for store in self._paged_cache.values()),
            "open_pages": sum(store.open_pages for store in self._paged_cache.values()),
            "appended_tokens": appended_this_step,
            "appended_tokens_total": self._paged_appended_total,
            "repack_tokens_saved": repack_saved_total,
            "page_evictions": sum(store.page_evictions for store in self._paged_cache.values()),
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
            "mode_control": bool(self._mode_control_enabled),
            "macro_block_size": int(self._macro_block_size),
            "skip_blocks": mode_stats.get("skip_blocks", 0),
            "delta_blocks": mode_stats.get("delta_blocks", 0),
            "refresh_blocks": mode_stats.get("refresh_blocks", 0),
            "skipped_rows": mode_stats.get("skipped_rows", 0),
            "rewritten_rows": mode_stats.get("rewritten_rows", 0),
            "projected_rows": mode_stats.get("active_rows", 0),
        }

        return {
            "output": logits,
            "kv_cache": updated_kv,
            "attention_scores": attention_scores,
            "execution_stats": self._last_execution_stats,
        }

    def _mode_rewrite_dispatch(
        self,
        store: PagedKVStore,
        layer_id: int,
        k: torch.Tensor,
        v: torch.Tensor,
        page_precision: str,
    ) -> int:
        """Rewrite only the rows the classifier marked active (SKIP zero-write).

        DELTA blocks rewrite their contiguous active runs; REFRESH blocks
        rewrite the whole macro interval; SKIP blocks leave the pages
        untouched. Returns the number of rewritten token rows.
        """
        assert self._mode_state is not None and self._mode_config is not None
        modes = self._mode_state.last_modes.get(layer_id, {})
        active_rows = self._mode_state.last_active_rows.get(layer_id, [])
        active_set = set(active_rows)
        seq_len = k.shape[2]
        block_size = self._mode_config.macro_block_size
        all_refresh = not modes or all(mode == "REFRESH" for mode in modes.values())
        if all_refresh and page_precision in ("fp32", "fp16"):
            # Fast path: nothing is preserved, so a single full write beats
            # per-block rewrites (content and page layout are identical).
            store.clear()
            store.append(k.contiguous(), v.contiguous(), page_precision)
            return seq_len
        rewritten = 0
        for macro_idx, start in enumerate(range(0, seq_len, block_size)):
            end = min(start + block_size, seq_len)
            mode = modes.get(macro_idx, "REFRESH")
            if mode == "SKIP":
                continue
            if mode == "REFRESH":
                rewritten += store.rewrite_tokens(
                    start,
                    k[:, :, start:end, :].contiguous(),
                    v[:, :, start:end, :].contiguous(),
                    page_precision,
                )
                continue
            run_start = None
            for row in range(start, end + 1):
                if row < end and row in active_set:
                    if run_start is None:
                        run_start = row
                    continue
                if run_start is not None:
                    rewritten += store.rewrite_tokens(
                        run_start,
                        k[:, :, run_start:row, :].contiguous(),
                        v[:, :, run_start:row, :].contiguous(),
                        page_precision,
                    )
                    run_start = None
        return rewritten

    @staticmethod
    def _entry_kv_tensors(entry: Any) -> Optional[tuple]:
        """Return dense (k, v) for dense, packed or paged KV entries (AR013/14)."""
        if entry is None:
            return None
        if entry.k_tensor is not None and entry.v_tensor is not None:
            return entry.k_tensor, entry.v_tensor
        packed = getattr(entry, "packed_kv", None)
        if packed is not None:
            return packed.unpack()
        paged = getattr(entry, "paged_kv", None)
        if paged is not None:
            return paged.gather()
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

    @property
    def last_generate_stores(self) -> Optional[Dict[int, PagedKVStore]]:
        """Per-layer PagedKVStore instances from the most recent generate().

        Returns None before any successful generate call (and after a
        fail-fast validation error, which creates no stores).
        """
        return self._last_generate_stores

    def generate(
        self,
        prompt_ids: torch.Tensor,
        gen_len: int,
        config: BlockDecodingConfig,
        capture_block_logits: bool = False,
    ) -> GenerateResult:
        """Blockwise confidence-aware parallel generation (AR016, FR16.3).

        Thin delegation to the ``blockwise_generate`` pure function
        (AR018 D3): the orchestration body moved verbatim (zero logic
        change) so non-backend callers (serve arms, tests) share one
        implementation. ``_last_generate_stores`` semantics are kept
        exactly: after a successful call it holds the per-layer
        PagedKVStore instances this generation used; failed calls leave
        it untouched.

        Args:
            prompt_ids: [B, P] integer prompt token ids; must not contain
                ``config.mask_token_id``.
            gen_len: Number of tokens to generate G; must be a multiple of
                ``config.block_size``.
            config: BlockDecodingConfig with block/threshold/page settings.

        Returns:
            GenerateResult with [B, P + G] sequences (prompt preserved
            bit-for-bit) and generation stats.

        Raises:
            RuntimeError: If no model has been loaded.
            ValueError: On invalid prompt shape/dtype, over-length
                sequences, mask tokens inside the prompt, or gen_len not
                divisible by the block size (all fail-fast: no stores are
                created and no state is mutated).
        """
        if self._model is None:
            raise RuntimeError("generate requires a loaded model; call load() first")
        result, stores = blockwise_generate(
            self._model, prompt_ids, gen_len, config, capture_block_logits
        )
        self._last_generate_stores = stores
        return result
