"""Packed FFN weight storage with per-output-channel mixed precision.

Supports FP16, INT8, and INT4 storage for FFN up/down projection weights,
reusing the INT4 packing utilities from ``shadowinfer.utils.quantization``
and the mixed-precision grouping ideas from
``shadowinfer.shadowkv.packed_kv_cache``.
"""

from __future__ import annotations

from typing import Any, Dict

import torch

from shadowinfer.utils.quantization import Quantizer


class PackedFFNWeight:
    """Byte-level packed FFN weight storage with per-channel mixed precision.

    The weight is stored as a collection of channel-wise slices. Each slice
    can be kept in FP16, INT8, or packed INT4, allowing real memory savings
    compared to a dense FP32 tensor.
    """

    PRECISION_BITS = {"fp32": 32, "fp16": 16, "int8": 8, "int4": 4}

    def __init__(
        self,
        shape: tuple,
        pack_dim: int = 0,
        device: Any = "cpu",
    ) -> None:
        """Initialize an empty packed weight container.

        Args:
            shape: Original dense weight shape.
            pack_dim: Dimension along which precision is assigned (0 for rows,
                1 for columns in a 2D weight).
            device: Torch device used for unpacked tensors.
        """
        self.shape = tuple(shape)
        self.pack_dim = pack_dim
        self.device = device
        self._data: Dict[int, Dict[str, Any]] = {}
        self._memory_bytes = 0
        self._original_bytes = int(
            torch.prod(torch.tensor(self.shape, dtype=torch.int64)).item() * 4
        )

    @classmethod
    def pack(
        cls,
        weight: torch.Tensor,
        precision_map: Dict[int, str],
        pack_dim: int = 0,
        group_size: int = 64,
    ) -> "PackedFFNWeight":
        """Pack a dense FP32 weight tensor into byte-level storage.

        Args:
            weight: Dense weight tensor, typically ``[out_features, in_features]``.
            precision_map: Mapping from channel index (along ``pack_dim``) to
                precision string (``"fp16"``, ``"int8"``, ``"int4"`` or ``"fp32"``).
            pack_dim: Dimension along which channels are indexed.
            group_size: INT4 quantization group size. If it does not divide the
                channel length, the channel length itself is used as a single group.

        Returns:
            A ``PackedFFNWeight`` instance.
        """
        obj = cls(weight.shape, pack_dim=pack_dim, device=weight.device)
        obj._do_pack(weight, precision_map, group_size=group_size)
        return obj

    def _do_pack(
        self,
        weight: torch.Tensor,
        precision_map: Dict[int, str],
        group_size: int = 64,
    ) -> None:
        """Internal packing implementation."""
        self._data = {}
        total_bytes = 0
        channels = weight.shape[self.pack_dim]

        for channel_idx in range(channels):
            precision = precision_map.get(channel_idx, "fp16").lower()
            if precision not in self.PRECISION_BITS:
                precision = "fp16"

            # Extract the channel slice and flatten it.
            slice_tensor = weight.select(self.pack_dim, channel_idx)
            flat = slice_tensor.reshape(-1).contiguous()
            channel_len = flat.numel()

            if precision == "fp32":
                stored = flat.float()
                total_bytes += stored.numel() * stored.element_size()
                self._data[channel_idx] = {"precision": "fp32", "tensor": stored}

            elif precision == "fp16":
                stored = flat.half()
                total_bytes += stored.numel() * stored.element_size()
                self._data[channel_idx] = {"precision": "fp16", "tensor": stored}

            elif precision == "int8":
                # Quantizer works along the last dimension; reshape to [1, channel_len].
                quantized, scale = Quantizer.quantize_int8(flat.unsqueeze(0))
                quantized = quantized.squeeze(0)
                scale = scale.squeeze(0)
                total_bytes += (
                    quantized.numel() * quantized.element_size()
                    + scale.numel() * scale.element_size()
                )
                self._data[channel_idx] = {
                    "precision": "int8",
                    "q": quantized,
                    "scale": scale,
                }

            elif precision == "int4":
                effective_group_size = group_size if channel_len % group_size == 0 else channel_len
                flat_2d = flat.unsqueeze(0)  # [1, channel_len]
                quantized, scale = Quantizer.quantize_int4(flat_2d, group_size=effective_group_size)
                packed = Quantizer.pack_int4(quantized.reshape(-1))
                total_bytes += (
                    packed.numel() * packed.element_size() + scale.numel() * scale.element_size()
                )
                self._data[channel_idx] = {
                    "precision": "int4",
                    "packed": packed,
                    "scale": scale,
                    "group_size": effective_group_size,
                    "num_groups": channel_len // effective_group_size,
                }

        self._memory_bytes = total_bytes

    def unpack(self) -> torch.Tensor:
        """Unpack the weight back to a dense FP32 tensor.

        Returns:
            Dense FP32 tensor with the original ``self.shape``.
        """
        out = torch.zeros(self.shape, dtype=torch.float32, device=self.device)

        for channel_idx, data in self._data.items():
            target = out.select(self.pack_dim, channel_idx)
            target_flat = target.reshape(-1)
            precision = data["precision"]

            if precision == "fp32":
                target_flat.copy_(data["tensor"].float())

            elif precision == "fp16":
                target_flat.copy_(data["tensor"].float())

            elif precision == "int8":
                dequantized = Quantizer.dequantize_int8(data["q"], data["scale"])
                target_flat.copy_(dequantized.reshape(-1))

            elif precision == "int4":
                channel_len = data["num_groups"] * data["group_size"]
                unpacked = Quantizer.unpack_int4(data["packed"], channel_len)
                unpacked_2d = unpacked.reshape(data["num_groups"], data["group_size"])
                dequantized = Quantizer.dequantize_int4(unpacked_2d, data["scale"])
                target_flat.copy_(dequantized.reshape(-1))

        return out

    def dequantize_for_matmul(self) -> torch.Tensor:
        """Return a dense tensor suitable for fallback matmul paths.

        Returns:
            FP32 dense tensor with the original ``self.shape``.
        """
        return self.unpack()

    def memory_bytes(self) -> int:
        """Return the actual byte-level storage size."""
        return self._memory_bytes

    def original_bytes(self) -> int:
        """Return the original dense FP32 storage size."""
        return self._original_bytes

    def memory_savings_bytes(self) -> int:
        """Return bytes saved versus the original dense FP32 tensor."""
        return self._original_bytes - self._memory_bytes

    def __repr__(self) -> str:
        return (
            f"PackedFFNWeight(shape={self.shape}, pack_dim={self.pack_dim}, "
            f"memory_bytes={self._memory_bytes}, "
            f"original_bytes={self._original_bytes})"
        )
