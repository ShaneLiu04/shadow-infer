"""Executable INT8 linear layer for FFN weights (AR011).

Weights are quantized symmetrically per output channel; activations are
quantized dynamically per token row. When available, the matrix multiply runs
through ``torch._int_mm`` (int8 x int8 -> int32) and is dequantized with
``scale_x * scale_w``. If ``_int_mm`` is unavailable or rejects the shape, the
same integer matrix product is evaluated in float32 and the module reports the
fallback once. No math is mocked; FP32 weights are released after conversion.
"""

from __future__ import annotations

import logging
from typing import Dict, Optional

import torch
import torch.nn as nn

logger = logging.getLogger("shadowinfer.ffn_optimizer.int8_linear")

_ACT_SCALE_EPS = 1e-8
_WEIGHT_SCALE_EPS = 1e-8


class INT8Linear(nn.Module):
    """Linear layer with int8 weights, per-channel scales and dynamic activations."""

    def __init__(
        self,
        qweight: torch.Tensor,
        weight_scale: torch.Tensor,
        bias: Optional[torch.Tensor] = None,
        in_features: Optional[int] = None,
        out_features: Optional[int] = None,
    ) -> None:
        super().__init__()
        if qweight.dim() != 2:
            raise ValueError(f"qweight must be 2-D, got shape {tuple(qweight.shape)}")
        if qweight.dtype != torch.int8:
            raise ValueError(f"qweight must be int8, got {qweight.dtype}")
        in_features = in_features if in_features is not None else int(qweight.shape[0])
        out_features = out_features if out_features is not None else int(qweight.shape[1])
        if tuple(qweight.shape) != (in_features, out_features):
            raise ValueError(
                f"qweight shape {tuple(qweight.shape)} != ({in_features}, {out_features})"
            )
        if weight_scale.dim() != 1 or weight_scale.numel() != out_features:
            raise ValueError(
                f"weight_scale must have {out_features} elements, got {tuple(weight_scale.shape)}"
            )
        if bias is not None and bias.numel() != out_features:
            raise ValueError(f"bias must have {out_features} elements")

        self.in_features = int(in_features)
        self.out_features = int(out_features)
        self.register_buffer("qweight", qweight.to(torch.int8).contiguous())
        self.register_buffer("weight_scale", weight_scale.to(torch.float32).contiguous())
        if bias is None:
            self.register_buffer("bias", None)
        else:
            self.register_buffer("bias", bias.to(torch.float32).contiguous())
        self._int_mm_available = hasattr(torch, "_int_mm")
        self._fallback_warned = False

    @classmethod
    def from_linear(cls, linear: nn.Linear) -> "INT8Linear":
        """Quantize an ``nn.Linear`` (weights symmetric per output channel)."""
        if not isinstance(linear, nn.Linear):
            raise TypeError(f"expected nn.Linear, got {type(linear).__name__}")
        weight = linear.weight.detach().to(torch.float32)
        # weight is [out_features, in_features]; scale per output channel
        abs_max = weight.abs().amax(dim=1)
        scale = (abs_max / 127.0).clamp_min(_WEIGHT_SCALE_EPS)
        qweight = torch.round(weight / scale.unsqueeze(1)).clamp(-127, 127).to(torch.int8)
        bias = None if linear.bias is None else linear.bias.detach().to(torch.float32).clone()
        return cls(
            qweight=qweight.transpose(0, 1).contiguous(),
            weight_scale=scale.contiguous(),
            bias=bias,
            in_features=linear.in_features,
            out_features=linear.out_features,
        )

    @property
    def using_int_mm(self) -> bool:
        return self._int_mm_available

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.shape[-1] != self.in_features:
            raise ValueError(f"input last dim {x.shape[-1]} != in_features {self.in_features}")
        leading_shape = x.shape[:-1]
        flat = x.reshape(-1, self.in_features)
        if flat.numel() == 0:
            return torch.empty((*leading_shape, self.out_features), dtype=x.dtype, device=x.device)

        flat_f32 = flat.to(torch.float32)
        act_scale = (flat_f32.abs().amax(dim=1) / 127.0).clamp_min(_ACT_SCALE_EPS)
        x_q = torch.round(flat_f32 / act_scale.unsqueeze(1)).clamp(-127, 127).to(torch.int8)

        if self._int_mm_available:
            try:
                acc = torch._int_mm(x_q, self.qweight)
            except Exception as exc:  # pragma: no cover - platform dependent
                self._int_mm_available = False
                self._warn_fallback(exc)
                acc = self._fallback_matmul(x_q)
        else:
            if not self._fallback_warned:
                self._warn_fallback(None)
            acc = self._fallback_matmul(x_q)

        output = acc.to(torch.float32) * (act_scale.unsqueeze(1) * self.weight_scale.unsqueeze(0))
        if self.bias is not None:
            output = output + self.bias.unsqueeze(0)
        return output.to(x.dtype).reshape(*leading_shape, self.out_features)

    def _fallback_matmul(self, x_q: torch.Tensor) -> torch.Tensor:
        return torch.matmul(x_q.to(torch.float32), self.qweight.to(torch.float32))

    def _warn_fallback(self, exc: Optional[BaseException]) -> None:
        if self._fallback_warned:
            return
        self._fallback_warned = True
        detail = f": {exc}" if exc is not None else ""
        logger.warning("int8 _int_mm unavailable, using float32 integer-product fallback%s", detail)

    def memory_bytes(self) -> int:
        total = self.qweight.numel() + self.weight_scale.numel() * 4
        if self.bias is not None:
            total += self.bias.numel() * 4
        return int(total)

    def dense_equivalent_bytes(self) -> int:
        total = self.in_features * self.out_features * 4
        if self.bias is not None:
            total += self.out_features * 4
        return int(total)

    def extra_repr(self) -> str:
        return (
            f"in_features={self.in_features}, out_features={self.out_features}, "
            f"bias={self.bias is not None}, int_mm={self._int_mm_available}"
        )


def dequantized_weight(linear: INT8Linear) -> torch.Tensor:
    """Return the FP32 weight for diagnostics/golden comparisons only."""
    return linear.qweight.to(torch.float32) * linear.weight_scale.unsqueeze(0)


FFN_LINEAR_SUFFIXES = (
    "mlp.up_proj",
    "mlp.down_proj",
    "mlp.gate_proj",
    "mlp.c_fc",
    "mlp.c_proj",
)


def quantize_ffn_modules(
    model: nn.Module,
    suffixes: tuple = FFN_LINEAR_SUFFIXES,
    keep_fp32_reference: bool = False,
) -> Dict[str, object]:
    """Replace matching FFN ``nn.Linear`` modules in an arbitrary model (e.g. HF).

    Modules whose qualified name ends with one of ``suffixes`` and that are plain
    ``nn.Linear`` are converted to :class:`INT8Linear`. The function is
    idempotent: already-converted layers are counted without re-quantizing.
    """
    references = [] if keep_fp32_reference else None
    modules = 0
    weight_bytes = 0
    dense_bytes = 0
    fallback = False
    for module_name, module in list(model.named_modules()):
        if not any(module_name.endswith(suffix) for suffix in suffixes):
            continue
        if isinstance(module, INT8Linear):
            modules += 1
            weight_bytes += module.memory_bytes()
            dense_bytes += module.dense_equivalent_bytes()
            fallback = fallback or not module.using_int_mm
            continue
        if not isinstance(module, nn.Linear):
            continue
        parent_name, _, attribute = module_name.rpartition(".")
        parent = model.get_submodule(parent_name) if parent_name else model
        quantized = INT8Linear.from_linear(module)
        if references is not None:
            references.append((parent, attribute, module))
        setattr(parent, attribute, quantized)
        modules += 1
        weight_bytes += quantized.memory_bytes()
        dense_bytes += quantized.dense_equivalent_bytes()
        fallback = fallback or not quantized.using_int_mm
    if modules == 0:
        return {"supported": False}
    stats: Dict[str, object] = {
        "supported": True,
        "modules": modules,
        "weight_bytes": weight_bytes,
        "dense_bytes": dense_bytes,
        "fallback": fallback,
    }
    if references:
        stats["_references"] = references
    return stats


__all__ = [
    "FFN_LINEAR_SUFFIXES",
    "INT8Linear",
    "dequantized_weight",
    "quantize_ffn_modules",
]
