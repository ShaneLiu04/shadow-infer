"""Packed FFN weight storage with per-output-channel mixed precision.

Supports FP16, INT8, and INT4 storage for FFN up/down projection weights,
reusing the INT4 packing utilities from ``shadowinfer.utils.quantization``
and the mixed-precision grouping ideas from
``shadowinfer.shadowkv.packed_kv_cache``.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

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
        # AR002 M5（design §7.2）：反量化结果缓存——dequantize_for_matmul 首次
        # 调用后缓存稠密权重与精度计划指纹；_do_pack 重建精度计划时失效。
        self._dequant_cache: Optional[torch.Tensor] = None
        self._dequant_cache_fingerprint: Optional[tuple] = None
        # AR004 FR4.1：精度计划指纹在打包时构建一次，热路径 O(1) 读取。
        self._plan_fingerprint: tuple = ()

    def _precision_plan_fingerprint(self) -> tuple:
        """当前精度计划指纹：(channel_idx, precision) 有序元组（O(1) 读取缓存）。"""
        return self._plan_fingerprint

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
        self._dequant_cache = None
        self._dequant_cache_fingerprint = None
        total_bytes = 0
        channels = weight.shape[self.pack_dim]
        plan = []

        for channel_idx in range(channels):
            precision = precision_map.get(channel_idx, "fp16").lower()
            if precision not in self.PRECISION_BITS:
                precision = "fp16"
            plan.append((channel_idx, precision))

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
        self._plan_fingerprint = tuple(plan)

    def unpack(self) -> torch.Tensor:
        """Unpack the weight back to a dense FP32 tensor (vectorized).

        Channels are grouped by precision and dequantized in batched form,
        then scattered back into the dense tensor. ``_unpack_reference`` keeps
        the original per-channel implementation as a golden reference.
        """
        out = torch.zeros(self.shape, dtype=torch.float32, device=self.device)
        if not self._data:
            return out

        groups: Dict[str, list] = {}
        for channel_idx, data in self._data.items():
            groups.setdefault(data["precision"], []).append(channel_idx)

        for precision, channels in groups.items():
            indices = torch.tensor(channels, dtype=torch.long, device=self.device)
            if precision in ("fp32", "fp16"):
                block = torch.stack([self._data[ch]["tensor"] for ch in channels]).float()
            elif precision == "int8":
                q = torch.stack([self._data[ch]["q"] for ch in channels])
                scale = torch.stack([self._data[ch]["scale"] for ch in channels])
                block = Quantizer.dequantize_int8(q, scale)
            elif precision == "int4":
                block = self._dequantize_int4_batch(channels)
            else:
                block = torch.stack([self._data[ch]["tensor"] for ch in channels]).float()

            if self.pack_dim == 0:
                out[indices] = block
            else:
                out[:, indices] = block.transpose(0, 1)
        return out

    def _dequantize_int4_batch(self, channels: list) -> torch.Tensor:
        """Batched INT4 dequantization for channels sharing grouping layout."""
        sample = self._data[channels[0]]
        num_groups = sample["num_groups"]
        group_size = sample["group_size"]
        channel_len = num_groups * group_size

        packed = torch.stack([self._data[ch]["packed"] for ch in channels])
        high = (packed >> 4) & 0x0F
        low = packed & 0x0F
        interleaved = torch.stack([high, low], dim=-1).reshape(packed.shape[0], -1)
        int8_tensor = (interleaved.to(torch.int16) - 8).to(torch.int8)
        quantized = int8_tensor[:, :channel_len].reshape(-1, num_groups, group_size)

        scales = [self._data[ch]["scale"].reshape(num_groups, 1) for ch in channels]
        scale = torch.stack(scales)
        dequantized = quantized.float() * scale
        return dequantized.reshape(len(channels), channel_len)

    def _unpack_reference(self) -> torch.Tensor:
        """Golden per-channel unpack (pre-AR004 implementation)."""
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

        AR002 M5（design §7.2）：结果缓存——精度计划指纹一致时直接复用
        上次反量化结果（语义逐元素等价，同输入同输出）；指纹不一致或
        缓存为空时重算并写回缓存。
        """
        fingerprint = self._precision_plan_fingerprint()
        cache_fp = self._dequant_cache_fingerprint
        if self._dequant_cache is not None and (cache_fp is fingerprint or cache_fp == fingerprint):
            return self._dequant_cache
        dense = self.unpack()
        self._dequant_cache = dense
        self._dequant_cache_fingerprint = fingerprint
        return dense

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
