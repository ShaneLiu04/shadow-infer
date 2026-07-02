"""ShadowInfer 量化/反量化工具模块。

对应 TECHNICAL_SPEC.md §2.1.2 和 §2.3.2 中的量化策略。
"""

__version__ = "3.0"

from typing import List, Optional, Tuple

import torch


class Quantizer:
    """量化工具类。支持 FP32/FP16/INT8/INT4 精度。"""

    @staticmethod
    def quantize_tensor(
        tensor: torch.Tensor,
        precision: str,
        group_size: int = 64,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        量化 tensor。

        对应 TECHNICAL_SPEC.md §2.1.2:
        - FP32: 返回原 tensor
        - FP16: 转换为 half
        - INT8: per-channel 动态缩放
        - INT4: 分组量化 (group_size=64)

        Args:
            tensor: 待量化的张量，形状任意。
            precision: 目标精度，"fp32" / "fp16" / "int8" / "int4"。
            group_size: INT4 分组大小，默认 64。

        Returns:
            (quantized_tensor, scale)
            - scale 仅在 int8 / int4 时返回，否则为 None。
        """
        p = precision.lower()
        if p == "fp32":
            return tensor, None
        elif p == "fp16":
            return tensor.half(), None
        elif p == "int8":
            return Quantizer.quantize_int8(tensor)
        elif p == "int4":
            return Quantizer.quantize_int4(tensor, group_size=group_size)
        else:
            raise ValueError(f"Unsupported precision: {precision}")

    @staticmethod
    def dequantize_tensor(
        quantized: torch.Tensor,
        scale: Optional[torch.Tensor],
        precision: str,
    ) -> torch.Tensor:
        """
        反量化 tensor。

        Args:
            quantized: 量化后的张量。
            scale: 量化 scale，仅在 int8 / int4 时需要。
            precision: 原始量化精度。

        Returns:
            反量化后的浮点张量。
        """
        p = precision.lower()
        if p == "fp32":
            return quantized
        elif p == "fp16":
            return quantized.float()
        elif p == "int8":
            if scale is None:
                raise ValueError("scale is required for INT8 dequantization")
            return Quantizer.dequantize_int8(quantized, scale)
        elif p == "int4":
            if scale is None:
                raise ValueError("scale is required for INT4 dequantization")
            return Quantizer.dequantize_int4(quantized, scale)
        else:
            raise ValueError(f"Unsupported precision: {precision}")

    @staticmethod
    def quantize_int8(tensor: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        INT8 per-channel 量化。

        scale = max(abs(tensor)) / 127
        quantized = round(tensor / scale).clamp(-128, 127).to(int8)

        Args:
            tensor: 待量化张量。

        Returns:
            (quantized_tensor, scale)
        """
        if tensor.numel() == 0:
            scale = torch.tensor(1.0, device=tensor.device, dtype=tensor.dtype)
            return tensor.to(torch.int8), scale

        scale = tensor.abs().max(dim=-1, keepdim=True)[0] / 127.0
        # 避免除零
        scale = torch.where(scale == 0, torch.ones_like(scale), scale)
        quantized = torch.round(tensor / scale).clamp(-128, 127).to(torch.int8)
        return quantized, scale

    @staticmethod
    def quantize_int4(
        tensor: torch.Tensor,
        group_size: int = 64,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        INT4 分组量化。

        scale = max(abs(tensor)) / 7
        quantized = round(tensor / scale).clamp(-8, 7).to(int8)
        （INT4 值存储在 int8 中，两个值打包成一个 byte）

        Args:
            tensor: 待量化张量，最后一维长度需能被 group_size 整除。
            group_size: 每组元素数，默认 64。

        Returns:
            (quantized_tensor, scale)
            - quantized_tensor 形状为 (...), 每个元素存储一个 INT4 值（用 int8 表示）。
            - scale 形状为 (..., num_groups, 1)。
        """
        if tensor.numel() == 0:
            scale = torch.tensor(1.0, device=tensor.device, dtype=tensor.dtype)
            return tensor.to(torch.int8), scale

        last_dim = tensor.shape[-1]
        if last_dim % group_size != 0:
            raise ValueError(f"最后一维长度 {last_dim} 必须能被 group_size {group_size} 整除")

        num_groups = last_dim // group_size
        tensor_reshaped = tensor.reshape(*tensor.shape[:-1], num_groups, group_size)
        scale = tensor_reshaped.abs().max(dim=-1, keepdim=True)[0] / 7.0
        scale = torch.where(scale == 0, torch.ones_like(scale), scale)
        quantized = torch.round(tensor_reshaped / scale).clamp(-8, 7).to(torch.int8)
        return quantized, scale

    @staticmethod
    def dequantize_int8(
        quantized: torch.Tensor,
        scale: torch.Tensor,
    ) -> torch.Tensor:
        """
        INT8 反量化。

        公式：quantized.float() * scale

        Args:
            quantized: INT8 量化张量。
            scale: 量化 scale。

        Returns:
            反量化后的浮点张量。
        """
        return quantized.float() * scale

    @staticmethod
    def dequantize_int4(
        quantized: torch.Tensor,
        scale: torch.Tensor,
    ) -> torch.Tensor:
        """
        INT4 反量化。

        将分组量化后的 int8 值（每个值存一个 INT4）乘以 scale，
        并 reshape 回原始维度。

        Args:
            quantized: INT4 量化张量（存储为 int8），形状为 (..., num_groups, group_size)。
            scale: 量化 scale，形状为 (..., num_groups, 1)。

        Returns:
            反量化后的浮点张量。
        """
        dequantized = quantized.float() * scale
        # 恢复原始形状
        return dequantized.reshape(*quantized.shape[:-2], -1)

    @staticmethod
    def compute_quantization_error(
        original: torch.Tensor,
        quantized: torch.Tensor,
    ) -> float:
        """
        计算量化误差。

        公式：||original - quantized|| / ||original||

        Args:
            original: 原始张量。
            quantized: 量化/反量化后的张量。

        Returns:
            相对量化误差（float）。
        """
        if original.numel() == 0:
            return 0.0

        original_norm = torch.norm(original)
        if original_norm.item() == 0:
            return 0.0

        error_norm = torch.norm(original - quantized)
        return float(error_norm / original_norm)

    @staticmethod
    def pack_int4(int8_tensor: torch.Tensor) -> torch.Tensor:
        """
        将 INT8 存储的 INT4 值打包为 half-byte (2 values per byte)。

        假设 int8_tensor 中每个值都在 [-8, 7] 范围内，用 4-bit 有符号表示。
        打包方式：高 4 bits 存第一个值，低 4 bits 存第二个值。

        Args:
            int8_tensor: 形状为 (...), dtype=int8, 每个值在 [-8, 7]。

        Returns:
            packed: 形状为 (... // 2,), dtype=uint8。
        """
        if int8_tensor.numel() == 0:
            return torch.empty(0, dtype=torch.uint8, device=int8_tensor.device)

        # 偏移到无符号 0-15
        offset = (int8_tensor + 8).to(torch.uint8)

        # 如果元素个数为奇数，补一个零
        if offset.numel() % 2 != 0:
            pad = torch.zeros(1, dtype=torch.uint8, device=int8_tensor.device)
            offset = torch.cat([offset.flatten(), pad])

        offset = offset.flatten()
        high = offset[0::2] << 4
        low = offset[1::2] & 0x0F
        packed = high | low
        return packed

    @staticmethod
    def unpack_int4(
        packed: torch.Tensor,
        shape_or_num_elements: int | Tuple[int, ...] | List[int],
    ) -> torch.Tensor:
        """
        解包 INT4 值。

        Args:
            packed: 形状为 (...), dtype=uint8, 每字节存两个 INT4 值。
            shape_or_num_elements: 期望还原的元素个数，或目标形状。
                - int: 直接作为元素个数（保持旧接口兼容）。
                - tuple/list: 计算元素总数后还原并 reshape 为目标形状。

        Returns:
            int8_tensor: dtype=int8, 每个值在 [-8, 7]。
        """
        if packed.numel() == 0:
            target_shape = (
                shape_or_num_elements
                if isinstance(shape_or_num_elements, (tuple, list))
                else (shape_or_num_elements,)
            )
            return torch.empty(target_shape, dtype=torch.int8, device=packed.device)

        high = (packed >> 4) & 0x0F
        low = packed & 0x0F

        # 交错合并
        interleaved = torch.stack([high, low], dim=-1).flatten()
        # 偏移回有符号
        int8_tensor = (interleaved.to(torch.int16) - 8).to(torch.int8)

        if isinstance(shape_or_num_elements, int):
            return int8_tensor[:shape_or_num_elements]

        num_elements = 1
        for dim in shape_or_num_elements:
            num_elements *= dim
        return int8_tensor[:num_elements].reshape(shape_or_num_elements)
